"""
Tree-sitter based symbol extraction.

Uses the built-in tags.scm queries from tree-sitter grammar packages to extract
definitions and references. This follows the standardized tree-sitter tagging
convention used by GitHub code navigation.
"""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import tree_sitter_rust
import tree_sitter_python
import tree_sitter_javascript
import tree_sitter_typescript
import tree_sitter_go
from tree_sitter import Language, Parser, Query

logger = logging.getLogger(__name__)

# ── Language setup ────────────────────────────────────────────────

# Map file extensions to (language_func, package_module)
LANGUAGE_CONFIG: dict[str, tuple] = {
    "rs": (tree_sitter_rust.language, tree_sitter_rust),
    "py": (tree_sitter_python.language, tree_sitter_python),
    "js": (tree_sitter_javascript.language, tree_sitter_javascript),
    "ts": (tree_sitter_typescript.language_typescript, tree_sitter_typescript),
    "tsx": (tree_sitter_typescript.language_tsx, tree_sitter_typescript),
    "go": (tree_sitter_go.language, tree_sitter_go),
}


@lru_cache(maxsize=10)
def _get_language(ext: str) -> Language | None:
    """Get the tree-sitter Language for a file extension."""
    config = LANGUAGE_CONFIG.get(ext)
    if not config:
        return None
    lang_func, _ = config
    return Language(lang_func())


@lru_cache(maxsize=10)
def _get_tags_query(ext: str) -> Query | None:
    """Load the built-in tags.scm query for a language.
    
    These queries are maintained by tree-sitter and follow the standard
    @definition.* and @reference.* capture naming convention.
    
    Note: TypeScript/TSX use the JavaScript tags.scm since the TS tags.scm
    only covers TypeScript-specific constructs (.d.ts files), not regular
    functions/classes/arrow functions.
    """
    config = LANGUAGE_CONFIG.get(ext)
    if not config:
        return None
    
    lang_func, pkg = config
    lang = Language(lang_func())
    
    # For TS/TSX, use JavaScript tags (more comprehensive for actual code)
    # The TypeScript tags.scm only covers .d.ts type definitions
    if ext in ("ts", "tsx"):
        tags_pkg = tree_sitter_javascript
    else:
        tags_pkg = pkg
    
    # Find the tags.scm file in the package
    pkg_dir = os.path.dirname(tags_pkg.__file__)
    tags_path = os.path.join(pkg_dir, "queries", "tags.scm")
    
    if not os.path.exists(tags_path):
        logger.warning("No tags.scm found for %s at %s", ext, tags_path)
        return None
    
    try:
        query_text = Path(tags_path).read_text()
        return lang.query(query_text)
    except Exception as e:
        logger.warning("Failed to load tags.scm for %s: %s", ext, e)
        return None


def get_parser(ext: str) -> Parser | None:
    """Get a parser for the given file extension."""
    lang = _get_language(ext)
    if lang is None:
        return None
    return Parser(lang)


# ── Data models ───────────────────────────────────────────────────


@dataclass
class SymbolDef:
    """A symbol definition extracted from source code."""
    name: str
    kind: str  # "function", "method", "struct", "class", "enum", "interface", "module", "trait"
    file_path: str
    start_line: int
    end_line: int
    signature: str  # First line of the definition
    content: str  # Full source text of the symbol
    parent: str | None = None  # For methods: the parent struct/class name


@dataclass
class SymbolRef:
    """A reference (call/usage) to a symbol."""
    name: str
    file_path: str
    line: int
    context_name: str | None = None  # Enclosing function/method name


@dataclass
class FileParseResult:
    file_path: str
    content_hash: str
    definitions: list[SymbolDef] = field(default_factory=list)
    references: list[SymbolRef] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)  # Imported module paths


# ── Kind extraction from capture names ────────────────────────────
# Built-in tags.scm uses @definition.function, @definition.class, etc.
# We extract the kind from the capture name suffix.

def _extract_kind_from_capture(capture_name: str) -> str | None:
    """Extract the kind (function, class, method, etc.) from a capture name.
    
    Built-in tags.scm uses naming like:
    - @definition.function → "function"
    - @definition.class → "class"
    - @definition.method → "method"
    - @reference.call → "call"
    """
    if capture_name.startswith("definition."):
        return capture_name[len("definition."):]
    elif capture_name.startswith("reference."):
        return capture_name[len("reference."):]
    return None


# ── Main parser ───────────────────────────────────────────────────

def parse_file(file_path: str, content: str) -> FileParseResult:
    """Parse a source file and extract definitions + references.
    
    Uses the built-in tags.scm queries from tree-sitter grammar packages.
    These are the same queries used by GitHub for code navigation.
    """
    ext = Path(file_path).suffix.lstrip(".")
    content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]

    result = FileParseResult(
        file_path=file_path,
        content_hash=content_hash,
    )

    parser = get_parser(ext)
    if parser is None:
        return result

    source = content.encode("utf-8")
    tree = parser.parse(source)
    lines = content.split("\n")

    # Load the built-in tags.scm query for this language
    tags_query = _get_tags_query(ext)
    if tags_query is None:
        logger.debug("No tags.scm available for %s", ext)
        return result

    try:
        matches = tags_query.matches(tree.root_node)
        
        for _pattern_idx, captures in matches:
            # Find the @name capture (the identifier)
            name_nodes = captures.get("name", [])
            if not name_nodes:
                continue
            
            name_node = name_nodes[0]
            name = name_node.text.decode("utf-8")
            if len(name) < 2:
                continue
            
            # Find the definition or reference capture
            # Built-in queries use @definition.* or @reference.*
            def_node = None
            kind = None
            is_definition = False
            is_reference = False
            
            for capture_name, nodes in captures.items():
                if capture_name.startswith("definition."):
                    def_node = nodes[0] if nodes else None
                    kind = _extract_kind_from_capture(capture_name)
                    is_definition = True
                    break
                elif capture_name.startswith("reference."):
                    kind = _extract_kind_from_capture(capture_name)
                    is_reference = True
                    break
            
            if is_definition and def_node:
                start_line = def_node.start_point[0] + 1
                end_line = def_node.end_point[0] + 1
                
                # Get signature (first line of definition)
                sig_line_idx = def_node.start_point[0]
                signature = lines[sig_line_idx].strip() if sig_line_idx < len(lines) else ""
                
                # Get full content (capped at 10KB)
                max_bytes = 10_000
                end_byte = min(def_node.end_byte, def_node.start_byte + max_bytes)
                symbol_content = source[def_node.start_byte:end_byte].decode("utf-8", errors="replace")
                
                # Detect parent for methods
                parent = _find_parent_name(def_node, source)
                
                result.definitions.append(SymbolDef(
                    name=name,
                    kind=kind or "function",
                    file_path=file_path,
                    start_line=start_line,
                    end_line=end_line,
                    signature=signature,
                    content=symbol_content,
                    parent=parent,
                ))
            
            elif is_reference:
                line = name_node.start_point[0] + 1
                context = _find_enclosing_function(name_node, source)
                
                result.references.append(SymbolRef(
                    name=name,
                    file_path=file_path,
                    line=line,
                    context_name=context,
                ))
                
    except Exception as e:
        logger.warning("Tags query failed for %s: %s", file_path, e)

    # Extract imports (simplified)
    result.imports = _extract_imports(content, ext)

    return result


def _find_parent_name(node, source: bytes) -> str | None:
    """Walk up the tree to find a parent impl/class name."""
    current = node.parent
    while current:
        if current.type in ("impl_item", "class_definition", "class_declaration"):
            # Find the type/name child
            for child in current.children:
                if child.type in ("type_identifier", "identifier"):
                    return child.text.decode("utf-8")
        current = current.parent
    return None


def _find_enclosing_function(node, source: bytes) -> str | None:
    """Walk up the tree to find the enclosing function name."""
    current = node.parent
    while current:
        if current.type in (
            "function_item", "function_definition", "function_declaration",
            "method_definition", "method_declaration",
        ):
            for child in current.children:
                if child.type in ("identifier", "field_identifier", "property_identifier"):
                    return child.text.decode("utf-8")
        current = current.parent
    return None


def _extract_imports(content: str, ext: str) -> list[str]:
    """Quick regex-free import extraction."""
    imports = []
    for line in content.split("\n"):
        stripped = line.strip()
        if ext == "rs":
            if stripped.startswith("use ") and "::" in stripped:
                # "use crate::tools::search;" -> "crate::tools::search"
                path = stripped.removeprefix("use ").rstrip(";").strip()
                if "{" in path:
                    path = path.split("{")[0].rstrip("::")
                imports.append(path)
        elif ext == "py":
            if stripped.startswith("from ") or stripped.startswith("import "):
                imports.append(stripped)
        elif ext in ("js", "ts", "tsx"):
            if "import " in stripped and "from " in stripped:
                # Extract the module path
                parts = stripped.split("from ")
                if len(parts) > 1:
                    mod_path = parts[-1].strip().strip("'\"").strip(";")
                    imports.append(mod_path)
        elif ext == "go":
            if stripped.startswith('"') and stripped.endswith('"'):
                imports.append(stripped.strip('"'))
    return imports


# ── Utility ───────────────────────────────────────────────────────

SKIP_DIRS = {
    "node_modules", "target", ".git", "__pycache__", ".next",
    "dist", "build", "vendor", ".venv", "venv", ".tox",
    "coverage", ".cache", "reference-repos", ".forge",
}

CODE_EXTENSIONS = {"rs", "py", "js", "ts", "tsx", "go", "java", "cpp", "c", "h"}


def should_skip(path: Path) -> bool:
    return any(part in SKIP_DIRS for part in path.parts)


def is_code_file(path: Path) -> bool:
    return path.suffix.lstrip(".") in CODE_EXTENSIONS and not should_skip(path)
