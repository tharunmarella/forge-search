"""
Tree-sitter based symbol extraction.

Extracts definitions (functions, structs, classes, methods) and references
(calls, type usages) from source files. This data feeds into the Neo4j graph.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path

import tree_sitter_rust
import tree_sitter_python
import tree_sitter_javascript
import tree_sitter_typescript
import tree_sitter_go
from tree_sitter import Language, Parser

logger = logging.getLogger(__name__)

# ── Language setup ────────────────────────────────────────────────

LANGUAGES: dict[str, Language] = {}


def _init_languages():
    global LANGUAGES
    if LANGUAGES:
        return
    LANGUAGES = {
        "rs": Language(tree_sitter_rust.language()),
        "py": Language(tree_sitter_python.language()),
        "js": Language(tree_sitter_javascript.language()),
        "ts": Language(tree_sitter_typescript.language_typescript()),
        "tsx": Language(tree_sitter_typescript.language_tsx()),
        "go": Language(tree_sitter_go.language()),
    }


def get_parser(ext: str) -> Parser | None:
    _init_languages()
    lang = LANGUAGES.get(ext)
    if lang is None:
        return None
    p = Parser(lang)
    return p


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


# ── Tree-sitter queries per language ──────────────────────────────

# Rust definitions query
RUST_DEF_QUERY = """
(function_item name: (identifier) @name) @def
(struct_item name: (type_identifier) @name) @def
(enum_item name: (type_identifier) @name) @def
(trait_item name: (type_identifier) @name) @def
(impl_item type: (type_identifier) @name) @def
(type_item name: (type_identifier) @name) @def
(const_item name: (identifier) @name) @def
(static_item name: (identifier) @name) @def
(macro_definition name: (identifier) @name) @def
"""

# Rust references query - function calls and type usages
RUST_REF_QUERY = """
(call_expression function: (identifier) @name)
(call_expression function: (scoped_identifier name: (identifier) @name))
(call_expression function: (field_expression field: (field_identifier) @name))
(type_identifier) @name
(macro_invocation macro: (identifier) @name)
"""

# Python definitions
PYTHON_DEF_QUERY = """
(function_definition name: (identifier) @name) @def
(class_definition name: (identifier) @name) @def
"""

PYTHON_REF_QUERY = """
(call function: (identifier) @name)
(call function: (attribute attribute: (identifier) @name))
"""

# JavaScript/TypeScript definitions
JS_DEF_QUERY = """
(function_declaration name: (identifier) @name) @def
(class_declaration name: (identifier) @name) @def
(method_definition name: (property_identifier) @name) @def
(lexical_declaration (variable_declarator name: (identifier) @name value: (arrow_function))) @def
"""

JS_REF_QUERY = """
(call_expression function: (identifier) @name)
(call_expression function: (member_expression property: (property_identifier) @name))
"""

# Go definitions
GO_DEF_QUERY = """
(function_declaration name: (identifier) @name) @def
(method_declaration name: (field_identifier) @name) @def
(type_declaration (type_spec name: (type_identifier) @name)) @def
"""

GO_REF_QUERY = """
(call_expression function: (identifier) @name)
(call_expression function: (selector_expression field: (field_identifier) @name))
(type_identifier) @name
"""

DEF_QUERIES: dict[str, str] = {
    "rs": RUST_DEF_QUERY,
    "py": PYTHON_DEF_QUERY,
    "js": JS_DEF_QUERY,
    "ts": JS_DEF_QUERY,
    "tsx": JS_DEF_QUERY,
    "go": GO_DEF_QUERY,
}

REF_QUERIES: dict[str, str] = {
    "rs": RUST_REF_QUERY,
    "py": PYTHON_REF_QUERY,
    "js": JS_REF_QUERY,
    "ts": JS_REF_QUERY,
    "tsx": JS_REF_QUERY,
    "go": GO_REF_QUERY,
}


# ── Kind detection ────────────────────────────────────────────────

NODE_KIND_MAP = {
    # Rust
    "function_item": "function",
    "struct_item": "struct",
    "enum_item": "enum",
    "trait_item": "trait",
    "impl_item": "impl",
    "type_item": "type",
    "const_item": "const",
    "static_item": "static",
    "macro_definition": "macro",
    # Python
    "function_definition": "function",
    "class_definition": "class",
    # JS/TS
    "function_declaration": "function",
    "class_declaration": "class",
    "method_definition": "method",
    "lexical_declaration": "function",
    # Go
    "function_declaration": "function",
    "method_declaration": "method",
    "type_declaration": "struct",
    "type_spec": "struct",
}


# ── Main parser ───────────────────────────────────────────────────

def parse_file(file_path: str, content: str) -> FileParseResult:
    """Parse a source file and extract definitions + references."""
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

    # Extract definitions
    def_query_str = DEF_QUERIES.get(ext)
    if def_query_str:
        _init_languages()
        lang = LANGUAGES[ext]
        try:
            query = lang.query(def_query_str)
            matches = query.matches(tree.root_node)
            for _pattern_idx, match_dict in matches:
                name_nodes = match_dict.get("name", [])
                def_nodes = match_dict.get("def", [])
                if not name_nodes or not def_nodes:
                    continue

                name_node = name_nodes[0]
                def_node = def_nodes[0]

                name = name_node.text.decode("utf-8")
                if len(name) < 2:
                    continue

                start_line = def_node.start_point[0] + 1
                end_line = def_node.end_point[0] + 1
                kind = NODE_KIND_MAP.get(def_node.type, "function")

                # Get signature (first line of definition)
                sig_line_idx = def_node.start_point[0]
                signature = lines[sig_line_idx].strip() if sig_line_idx < len(lines) else ""

                # Get full content (capped at 100 lines for embedding)
                content_lines = lines[def_node.start_point[0]:min(def_node.end_point[0] + 1, def_node.start_point[0] + 100)]
                symbol_content = "\n".join(content_lines)

                # Detect parent for methods (Rust impl blocks)
                parent = _find_parent_name(def_node, source)

                result.definitions.append(SymbolDef(
                    name=name,
                    kind=kind,
                    file_path=file_path,
                    start_line=start_line,
                    end_line=end_line,
                    signature=signature,
                    content=symbol_content,
                    parent=parent,
                ))
        except Exception as e:
            logger.warning("Definition query failed for %s: %s", file_path, e)

    # Extract references
    ref_query_str = REF_QUERIES.get(ext)
    if ref_query_str:
        _init_languages()
        lang = LANGUAGES[ext]
        try:
            query = lang.query(ref_query_str)
            matches = query.matches(tree.root_node)
            for _pattern_idx, match_dict in matches:
                name_nodes = match_dict.get("name", [])
                if not name_nodes:
                    continue

                name_node = name_nodes[0]
                name = name_node.text.decode("utf-8")
                if len(name) < 2:
                    continue

                line = name_node.start_point[0] + 1

                # Find enclosing function
                context = _find_enclosing_function(name_node, source)

                result.references.append(SymbolRef(
                    name=name,
                    file_path=file_path,
                    line=line,
                    context_name=context,
                ))
        except Exception as e:
            logger.warning("Reference query failed for %s: %s", file_path, e)

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
