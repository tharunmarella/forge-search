"""
Shared indexing helpers used by multiple endpoints.

These functions are used by:
- /index endpoint
- /watch endpoint  
- /scan endpoint
"""

import logging
from ..core import embeddings
from ..core.parser import parse_file, FileParseResult, SymbolDef
from ..storage import store
from ..intelligence.service_classifier import LLMServiceClassifier, classify_file_by_path_heuristic

logger = logging.getLogger(__name__)


def build_enriched_text(
    defn: SymbolDef,
    calls_map: dict[str, list[str]],
    called_by_map: dict[str, list[str]],
    imports: list[str],
    service_name: str | None = None,
    architectural_role: str | None = None,
    description: str | None = None,
    cross_file_calls: dict[str, dict[str, list[tuple[str, str]]]] | None = None,
) -> str:
    """
    Build a context-enriched text for a symbol to embed.

    Instead of embedding raw source code, we prepend structured context
    so the embedding captures *what this code means in the system*:
      - Service/module hierarchy
      - Architectural role and purpose
      - Where it lives (file path, module)
      - What it is (kind, parent type)
      - What it connects to (calls, called by, cross-file)
      - Its actual code

    This is how a developer reads code: with surrounding context.
    """
    parts: list[str] = []

    # Service/architecture context (Level 0)
    if service_name:
        parts.append(f"[Service] {service_name}")
    
    if architectural_role:
        parts.append(f"[Role] {architectural_role}")
        
    if description:
        parts.append(f"[Purpose] {description}")

    # Location context
    parts.append(f"[File] {defn.file_path}")

    # Module hint from file path (e.g., "database/connection_manager")
    module = defn.file_path.rsplit(".", 1)[0].replace("/src/", "/").replace("\\", "/")
    parts.append(f"[Module] {module}")

    # Symbol identity
    parts.append(f"[{defn.kind.capitalize()}] {defn.name}")

    if defn.parent:
        parts.append(f"[Parent] {defn.parent}")

    # Signature (the most important single line)
    if defn.signature:
        parts.append(f"[Signature] {defn.signature}")

    # In-file relationship context
    outgoing = calls_map.get(defn.name)
    if outgoing:
        unique_calls = list(dict.fromkeys(outgoing))[:15]
        parts.append(f"[Calls] {', '.join(unique_calls)}")

    incoming = called_by_map.get(defn.name)
    if incoming:
        unique_callers = list(dict.fromkeys(incoming))[:10]
        parts.append(f"[Called by] {', '.join(unique_callers)}")

    # Cross-file relationship context (for architecture view)
    if cross_file_calls and defn.name in cross_file_calls:
        info = cross_file_calls[defn.name]
        if info.get("callees"):
            external = [f"{n}@{f}" for n, f in info["callees"][:10]]
            parts.append(f"[Cross-file Calls] {', '.join(external)}")
        if info.get("callers"):
            external = [f"{n}@{f}" for n, f in info["callers"][:10]]
            parts.append(f"[Cross-file Callers] {', '.join(external)}")

    # Key imports for module-level context
    if imports:
        short_imports = [imp.split("::")[-1].split(".")[-1] for imp in imports[:8]]
        parts.append(f"[Imports] {', '.join(short_imports)}")

    # The actual source code
    parts.append("")
    parts.append(defn.content)

    return "\n".join(parts)


def build_context_maps(
    parse_result: FileParseResult,
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """
    From a file's parse result, build:
      - calls_map:     {caller_name: [callee_name, ...]}
      - called_by_map: {callee_name: [caller_name, ...]}
    """
    calls_map: dict[str, list[str]] = {}
    called_by_map: dict[str, list[str]] = {}

    # Set of defined symbol names in this file for filtering
    defined_names = {d.name for d in parse_result.definitions}

    for ref in parse_result.references:
        if not ref.context_name:
            continue
        # Only include if both caller and callee are defined in this file
        # (internal calls only — external calls come from imports)
        if ref.context_name in defined_names and ref.name in defined_names:
            calls_map.setdefault(ref.context_name, []).append(ref.name)
            called_by_map.setdefault(ref.name, []).append(ref.context_name)

    return calls_map, called_by_map


async def index_files_batch(workspace_id: str, files: list[dict]) -> dict:
    """
    Shared indexing logic used by both /index endpoint and watcher.
    
    Args:
        workspace_id: Workspace identifier
        files: List of {path, content} dicts
        
    Returns:
        {
            "files_indexed": int,
            "nodes_created": int,
            "relationships_created": int,
            "embeddings_generated": int
        }
    """
    total_nodes = 0
    total_rels = 0
    total_embeddings = 0
    files_indexed = 0

    # 1. Pre-parse and collect metadata for LLM-based service discovery
    files_to_index = []
    files_metadata = []
    
    for file_payload in files:
        try:
            existing_hash = await store.get_file_hash(workspace_id, file_payload["path"])
            parse_result = parse_file(file_payload["path"], file_payload["content"])

            if existing_hash == parse_result.content_hash:
                continue
            
            files_to_index.append((file_payload, parse_result))
            files_metadata.append({
                "path": file_payload["path"],
                "symbols": [d.name for d in parse_result.definitions]
            })
        except Exception as e:
            logger.error("Failed to pre-parse %s: %s", file_payload["path"], e)
            continue

    if not files_to_index:
        return {"files_indexed": 0, "nodes_created": 0, "relationships_created": 0, "embeddings_generated": 0}

    # 2. LLM-based Service Discovery & Classification
    classifier = LLMServiceClassifier(workspace_id)
    
    # Get existing services or discover new ones if this is a large batch
    existing_services = await store.get_services(workspace_id)
    if not existing_services or len(files_to_index) > 10:
        logger.info("Discovering workspace architecture for %s...", workspace_id)
        services = await classifier.discover_workspace_architecture(files_metadata)
    else:
        from ..intelligence.service_classifier import ServiceInfo
        services = [ServiceInfo(**s) for s in existing_services]

    # Classify files in this batch
    classifications = await classifier.classify_files_batch(files_metadata, services)

    # 3. Process each file with enriched context
    for file_payload, parse_result in files_to_index:
        try:
            calls_map, called_by_map = build_context_maps(parse_result)
            
            # Get cross-file calls for better embedding context
            defined_names = {d.name for d in parse_result.definitions}
            cross_file_calls = await store.get_external_calls(
                workspace_id, file_payload["path"], defined_names, parse_result.references
            )
            
            # Get LLM classification for this file
            file_cls = classifications.get(file_payload["path"])
            service_name = file_cls.service_name if file_cls else ""
            service_id = file_cls.service_id if file_cls else ""

            enriched_texts: dict[str, str] = {}
            symbol_classifications: dict[str, dict] = {}
            
            for defn in parse_result.definitions:
                uid = f"{defn.file_path}:{defn.name}:{defn.start_line}"
                
                # Use file-level classification for symbols
                arch_role = file_cls.architectural_role if file_cls else ""
                description = file_cls.description if file_cls else ""
                
                enriched_texts[uid] = build_enriched_text(
                    defn, 
                    calls_map, 
                    called_by_map, 
                    parse_result.imports,
                    service_name=service_name,
                    architectural_role=arch_role,
                    description=description,
                    cross_file_calls=cross_file_calls
                )
                
                symbol_classifications[uid] = {
                    "service_id": service_id,
                    "architectural_role": arch_role,
                    "description": description
                }

            # FALLBACK: embed whole file if no symbols were extracted
            if not enriched_texts and file_payload["content"].strip():
                file_uid = f"{file_payload['path']}:__file__:0"
                truncated = file_payload["content"][:5000]
                ext = file_payload["path"].rsplit(".", 1)[-1] if "." in file_payload["path"] else "unknown"
                
                arch_role = file_cls.architectural_role if file_cls else "File"
                description = file_cls.description if file_cls else ""
                
                enriched_texts[file_uid] = (
                    f"[Service] {service_name}\n"
                    f"[Role] {arch_role}\n"
                    f"[Purpose] {description}\n"
                    f"File: {file_payload['path']}\n"
                    f"Type: {ext}\n"
                    f"---\n{truncated}"
                )
                
                symbol_classifications[file_uid] = {
                    "service_id": service_id,
                    "architectural_role": arch_role,
                    "description": description
                }
                logger.info("Fallback: embedding whole file %s", file_payload["path"])

            if enriched_texts:
                texts_list = list(enriched_texts.values())
                uids_list = list(enriched_texts.keys())
                raw_embeddings = await embeddings.embed_batch(texts_list)
                embedding_map = dict(zip(uids_list, raw_embeddings))
                total_embeddings += sum(1 for v in embedding_map.values() if v is not None)
            else:
                embedding_map = {}

            stats = await store.index_file_result(
                workspace_id, 
                parse_result, 
                embedding_map, 
                enriched_texts,
                classifications=symbol_classifications
            )
            total_nodes += stats["nodes_created"]
            total_rels += stats["relationships_created"]

            if parse_result.references:
                await store.build_call_edges(workspace_id, parse_result.references)
                total_rels += len(parse_result.references)

            if parse_result.imports:
                await store.build_import_edges(
                    workspace_id, file_payload["path"], parse_result.imports
                )

            files_indexed += 1
        except Exception as e:
            logger.error("Failed to index %s: %s", file_payload["path"], e, exc_info=True)
            continue

    return {
        "files_indexed": files_indexed,
        "nodes_created": total_nodes,
        "relationships_created": total_rels,
        "embeddings_generated": total_embeddings,
    }
