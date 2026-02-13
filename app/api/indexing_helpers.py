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

logger = logging.getLogger(__name__)


def build_enriched_text(
    defn: SymbolDef,
    calls_map: dict[str, list[str]],
    called_by_map: dict[str, list[str]],
    imports: list[str],
) -> str:
    """
    Build a context-enriched text for a symbol to embed.

    Instead of embedding raw source code, we prepend structured context
    so the embedding captures *what this code means in the system*:
      - Where it lives (file path, module)
      - What it is (kind, parent type)
      - What it connects to (calls, called by)
      - Its actual code

    This is how a developer reads code: with surrounding context.
    """
    parts: list[str] = []

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

    # Relationship context -- what does this symbol connect to?
    outgoing = calls_map.get(defn.name)
    if outgoing:
        # Deduplicate and limit
        unique_calls = list(dict.fromkeys(outgoing))[:15]
        parts.append(f"[Calls] {', '.join(unique_calls)}")

    incoming = called_by_map.get(defn.name)
    if incoming:
        unique_callers = list(dict.fromkeys(incoming))[:10]
        parts.append(f"[Called by] {', '.join(unique_callers)}")

    # Key imports for module-level context
    if imports:
        # Keep short -- just the first few meaningful imports
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

    for file_payload in files:
        try:
            existing_hash = await store.get_file_hash(workspace_id, file_payload["path"])
            parse_result = parse_file(file_payload["path"], file_payload["content"])

            if existing_hash == parse_result.content_hash:
                continue

            calls_map, called_by_map = build_context_maps(parse_result)

            enriched_texts: dict[str, str] = {}
            for defn in parse_result.definitions:
                uid = f"{defn.file_path}:{defn.name}:{defn.start_line}"
                enriched_texts[uid] = build_enriched_text(
                    defn, calls_map, called_by_map, parse_result.imports,
                )

            # FALLBACK: embed whole file if no symbols were extracted
            if not enriched_texts and file_payload["content"].strip():
                file_uid = f"{file_payload['path']}:__file__:0"
                truncated = file_payload["content"][:5000]
                ext = file_payload["path"].rsplit(".", 1)[-1] if "." in file_payload["path"] else "unknown"
                enriched_texts[file_uid] = (
                    f"File: {file_payload['path']}\n"
                    f"Type: {ext}\n"
                    f"---\n{truncated}"
                )
                logger.info("Fallback: embedding whole file %s", file_payload["path"])

            if enriched_texts:
                texts_list = list(enriched_texts.values())
                uids_list = list(enriched_texts.keys())
                raw_embeddings = await embeddings.embed_batch(texts_list)
                embedding_map = dict(zip(uids_list, raw_embeddings))
                # Only count successful (non-None) embeddings
                total_embeddings += sum(1 for v in embedding_map.values() if v is not None)
            else:
                embedding_map = {}

            stats = await store.index_file_result(
                workspace_id, parse_result, embedding_map, enriched_texts
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
