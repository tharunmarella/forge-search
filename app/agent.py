"""
LangGraph Agent Orchestrator for Forge.

Architecture Goal: Make a small model (Groq Kimi-K2) match Claude Opus performance
by providing it with "perfect context" through pre-enrichment.

The flow:
  1. User sends question + optional attached files
  2. BEFORE calling LLM, we gather:
     - Semantic search results from pgvector
     - Call chain traces for mentioned symbols  
     - Impact analysis if editing
     - Attached file contents from IDE
  3. Build a "perfect prompt" with all this context
  4. LLM just needs to reason + execute with full context
  5. For IDE tools (file ops), pause and return to IDE for execution
  6. Agent can use lookup_documentation tool for library docs on-demand
"""

from typing import Annotated, TypedDict, Literal
import json
import logging
import os
import re
import httpx

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langsmith import traceable

from . import store, embeddings, chat as chat_utils

# Documentation API configuration
CONTEXT7_API_URL = "https://mcp.context7.com/mcp"
DEVDOCS_API_URL = "https://devdocs.io"
CONTEXT7_API_KEY = os.getenv("CONTEXT7_API_KEY", "")

logger = logging.getLogger(__name__)

# ── System Prompt ──────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert senior software engineer working inside Forge IDE. You EXECUTE tasks, not describe them.

## CRITICAL RULES

1. **ALWAYS USE TOOLS** to perform actions. NEVER just describe what you would do.
   - WRONG: "I would use replace_in_file to change X to Y"
   - RIGHT: Actually call replace_in_file with the exact parameters
2. **DO the work, don't explain the work.** If the user asks you to refactor, USE the tools to make the changes.
3. **Read before editing**: Always use read_file to see exact current code before using replace_in_file
4. **replace_in_file old_str must match EXACTLY** — copy it character-for-character from read_file output
5. **For complex tasks**: Break into steps, execute each step with tools, verify with execute_command or read_file
6. **If something fails**: Analyze the error and try a different approach immediately
7. **ALWAYS VERIFY YOUR WORK** before finishing (see Verification section below)

## Tools Available

### Cloud Tools (use first for discovery)
- `codebase_search(query)`: Semantic search for code. Use to find relevant functions/classes.
- `trace_call_chain(symbol_name, direction, max_depth)`: Find what calls a function or what it calls.
- `impact_analysis(symbol_name, max_depth)`: Find all code affected by changing a symbol.
- `lookup_documentation(library, query)`: Look up official documentation for libraries/frameworks (e.g., "react", "fastapi"). Uses Context7 + DevDocs.io.

### File Tools (for reading and editing)
- `read_file(path, start_line, end_line)`: Read file contents. ALWAYS do this before editing.
- `replace_in_file(path, old_str, new_str)`: Replace exact text in a file. old_str must match exactly.
- `write_to_file(path, content)`: Write entire file. Only for new files.
- `execute_command(command)`: Run shell commands (grep, git, tests, etc.)

### LSP Tools (for type checking and code intelligence)
- `lsp_go_to_definition(path, line, column)`: Jump to where a symbol is defined.
- `lsp_find_references(path, line, column)`: Find all usages of a symbol.
- `lsp_hover(path, line, column)`: Get type info and docs for a symbol.
- `lsp_rename(path, line, column, new_name)`: Safe rename across the workspace.

## Workflow for Refactoring

1. Use `execute_command` with `grep -rn "old_name" --include="*.py"` to find ALL occurrences
2. Use `read_file` on each file to see the exact code around each occurrence
3. Use `replace_in_file` on each file to make the change
4. Use `execute_command` with grep again to verify no occurrences remain
5. **RUN VERIFICATION** (see below)

## VERIFICATION — MANDATORY BEFORE FINISHING

**You are NOT done until you verify your changes compile and pass checks.**

After making ALL edits, you MUST run a verification step using `execute_command`. Pick the right check based on the project:

| Project Type       | Verification Command                                    |
|--------------------|---------------------------------------------------------|
| Next.js            | `npm run build 2>&1 | tail -50` (catches "use client", SSR issues, imports) |
| TypeScript (non-Next) | `npx tsc --noEmit 2>&1 | head -50`                 |
| Python             | `python -m py_compile <file>` or `python -m mypy .`    |
| Rust               | `cargo check 2>&1 | tail -30`                          |
| Go                 | `go build ./... 2>&1 | tail -30`                       |
| General            | `npm test` / `pytest` / `cargo test` (if tests exist)  |

**IMPORTANT for Next.js**: `npx tsc --noEmit` does NOT catch Next.js-specific errors like missing `"use client"`. ALWAYS use `npm run build` for Next.js projects.

**If the check FAILS:**
1. Read the error output carefully
2. Fix the issues (e.g. missing imports, wrong types, missing "use client" for React hooks in Next.js App Router)
3. Re-run the verification
4. Repeat until it passes

**Common gotchas to check for:**
- Next.js App Router: Components using React hooks (useState, useEffect, etc.) MUST have `"use client"` at the top
- TypeScript: Missing type imports, wrong return types
- Python: Missing imports, indentation errors
- Rust: Borrow checker errors, missing trait implementations

NEVER report "done" with errors still present. Your job is only complete when the code compiles cleanly.

REMEMBER: You must CALL the tools. Do not write code blocks showing tool calls — actually invoke them."""


# ── State Definition ──────────────────────────────────────────────

class AgentState(TypedDict):
    """The state of the agent graph."""
    messages: Annotated[list[BaseMessage], add_messages]
    workspace_id: str
    # Pre-enriched context (set by first node)
    enriched_context: str
    # Attached files from IDE (live file contents)
    attached_files: dict[str, str]  # path -> content


# ── Tool Definitions (Cloud-side) ────────────────────────────────

# Note: These tools are defined with simple signatures. The workspace_id
# is injected by execute_server_tools from the state.

@tool
async def codebase_search(query: str) -> str:
    """
    Semantic code search - find code by meaning.
    Use for: understanding how things work, finding related code, exploring the codebase.
    Returns relevant code snippets with file paths and line numbers.
    """
    # The actual execution happens in execute_server_tools with workspace_id from state
    return "PENDING_SERVER_EXECUTION"


@tool  
async def trace_call_chain(symbol_name: str, direction: str = "both") -> str:
    """
    Deep call-chain traversal.
    Direction: 'upstream' (who calls this), 'downstream' (what does this call), 'both'.
    Returns the execution flow as a graph.
    """
    return "PENDING_SERVER_EXECUTION"


@tool
async def impact_analysis(symbol_name: str) -> str:
    """
    Blast radius analysis - find what's affected if you change this symbol.
    Returns all callers and dependent code.
    """
    return "PENDING_SERVER_EXECUTION"


@tool
async def lookup_documentation(library: str, query: str) -> str:
    """
    Look up documentation for a library/framework.
    Uses Context7 and DevDocs.io to find official documentation.
    
    Args:
        library: The library/framework name (e.g., "react", "fastapi", "langchain", "pandas")
        query: What you want to know (e.g., "how to use hooks", "async endpoints", "dataframe groupby")
    
    Returns:
        Relevant documentation snippets from official sources.
    """
    return "PENDING_SERVER_EXECUTION"


# ── Tool Definitions (IDE-side) ───────────────────────────────────
# These are "virtual" tools. When the agent calls them, the graph 
# pauses and returns to the IDE for execution.

@tool
def read_file(path: str) -> str:
    """Read the contents of a file from the user's machine."""
    return "PENDING_IDE_EXECUTION"


@tool
def write_to_file(path: str, content: str) -> str:
    """Create a new file or completely overwrite an existing one."""
    return "PENDING_IDE_EXECUTION"


@tool
def replace_in_file(path: str, old_str: str, new_str: str) -> str:
    """Replace text in a file. old_str must match exactly including whitespace."""
    return "PENDING_IDE_EXECUTION"


@tool
def execute_command(command: str) -> str:
    """Execute a shell command (git, npm, cargo, tests, etc.)."""
    return "PENDING_IDE_EXECUTION"


# ── LSP Tools (IDE-side, powered by language servers) ───────────

@tool
def lsp_go_to_definition(path: str, line: int, column: int) -> str:
    """
    Get the definition location for a symbol at a specific position.
    Uses the IDE's language server (rust-analyzer, pyright, etc.) for accurate results.
    Returns the file path and line number where the symbol is defined.
    """
    return "PENDING_IDE_EXECUTION"


@tool
def lsp_find_references(path: str, line: int, column: int) -> str:
    """
    Find ALL references to a symbol at a specific position.
    Uses the IDE's language server for accurate cross-file results.
    Great for understanding usage patterns and safe refactoring.
    Returns list of locations where the symbol is used.
    """
    return "PENDING_IDE_EXECUTION"


@tool
def lsp_hover(path: str, line: int, column: int) -> str:
    """
    Get type information and documentation for a symbol at a position.
    Uses the IDE's language server for accurate type info.
    Returns type signature, documentation, and other hover info.
    """
    return "PENDING_IDE_EXECUTION"


@tool
def lsp_rename(path: str, line: int, column: int, new_name: str) -> str:
    """
    Safely rename a symbol across the entire workspace.
    Uses the IDE's language server to find all occurrences and rename them atomically.
    This is the safest way to rename - better than find/replace.
    """
    return "PENDING_IDE_EXECUTION"


# Tool lists
SERVER_TOOLS = [codebase_search, trace_call_chain, impact_analysis, lookup_documentation]
IDE_TOOLS = [read_file, write_to_file, replace_in_file, execute_command]
LSP_TOOLS = [lsp_go_to_definition, lsp_find_references, lsp_hover, lsp_rename]
ALL_TOOLS = SERVER_TOOLS + IDE_TOOLS + LSP_TOOLS

IDE_TOOL_NAMES = {
    "read_file", "write_to_file", "replace_in_file", "execute_command",
    "lsp_go_to_definition", "lsp_find_references", "lsp_hover", "lsp_rename",
}
SERVER_TOOL_NAMES = {"codebase_search", "trace_call_chain", "impact_analysis", "lookup_documentation"}


# ── Pre-Enrichment Logic ──────────────────────────────────────────

@traceable(name="pre_enrichment", run_type="chain", tags=["enrichment"])
async def build_pre_enrichment(
    workspace_id: str,
    question: str,
    attached_files: dict[str, str] | None = None,
) -> str:
    """
    The secret sauce: gather ALL relevant context BEFORE the LLM sees the question.
    
    This is how we make a small model match Claude Opus - by doing the hard work
    of finding the right context upfront.
    """
    parts = []
    
    # 1. Attached files from IDE (live file contents)
    if attached_files:
        parts.append("## Live Files (from IDE)\n")
        for path, content in list(attached_files.items())[:5]:  # Limit to 5 files
            # Truncate very long files
            truncated = content[:8000] + "\n... (truncated)" if len(content) > 8000 else content
            parts.append(f"### `{path}`\n```\n{truncated}\n```\n")
        parts.append("")
    
    # 2. Semantic search based on the question
    try:
        logger.info("Pre-enrichment: searching for '%s' in workspace=%s", question[:100], workspace_id)
        query_emb = await embeddings.embed_query(question)
        search_results = await store.vector_search(workspace_id, query_emb, top_k=8)
        logger.info("Pre-enrichment: got %d search results for workspace=%s", len(search_results), workspace_id)
        
        if search_results:
            flat_results = [r["symbol"] for r in search_results]
            # Log what we're passing to build_context
            if flat_results:
                logger.info("Pre-enrichment: first result keys=%s, name=%s", 
                           list(flat_results[0].keys()), flat_results[0].get('name'))
            context_text = chat_utils.build_context_from_results(flat_results)
            if context_text:
                parts.append(context_text)
                logger.info("Pre-enrichment: built context len=%d", len(context_text))
            else:
                logger.warning("Pre-enrichment: build_context_from_results returned empty")
        else:
            logger.warning("Pre-enrichment: no results for query in workspace %s", workspace_id)
    except Exception as e:
        logger.error("Pre-enrichment search failed: %s", e, exc_info=True)
    
    # 3. Extract symbol names from question for trace analysis
    # Simple heuristic: look for CamelCase or snake_case identifiers
    symbols = extract_symbols_from_text(question)
    
    # 4. Trace call chains for mentioned symbols (limited)
    for sym in symbols[:2]:  # Max 2 symbols to avoid token explosion
        try:
            trace = await store.trace_call_chain(workspace_id, sym, direction="both", max_depth=2)
            if trace.get("nodes"):
                parts.append(f"\n## Call Chain: `{sym}`\n")
                for edge in trace.get("edges", [])[:10]:
                    parts.append(f"  `{edge['from']}` → `{edge['to']}`")
                parts.append("")
        except Exception as e:
            logger.debug("Trace failed for %s: %s", sym, e)
    
    # Note: Documentation lookup is now available as a tool (lookup_documentation)
    # The agent can call it on-demand instead of auto-fetching in pre-enrichment
    
    return "\n".join(parts).strip()


def extract_symbols_from_text(text: str) -> list[str]:
    """Extract potential symbol names from natural language text."""
    # CamelCase: FooBar, MyClass
    camel = re.findall(r'\b([A-Z][a-zA-Z0-9]+)\b', text)
    # snake_case: foo_bar, my_function  
    snake = re.findall(r'\b([a-z][a-z0-9]*(?:_[a-z0-9]+)+)\b', text)
    # Combined and deduplicated
    symbols = list(dict.fromkeys(camel + snake))
    return symbols[:5]  # Limit


# ── Documentation Lookup (Context7 + DevDocs.io) ──────────────────

@traceable(name="fetch_documentation", run_type="tool", tags=["documentation"])
async def _fetch_documentation(library: str, query: str) -> str:
    """
    Fetch documentation from Context7 and DevDocs.io.
    
    Context7: MCP-based, provides curated library docs with semantic search
    DevDocs.io: Free API for browsing official documentation
    """
    results = []
    
    # Try Context7 first (if API key is configured)
    if CONTEXT7_API_KEY:
        try:
            c7_result = await _query_context7(library, query)
            if c7_result:
                results.append(f"## Context7 Documentation\n\n{c7_result}")
        except Exception as e:
            logger.warning("Context7 lookup failed: %s", e)
    
    # Try DevDocs.io as fallback/supplement
    try:
        devdocs_result = await _query_devdocs(library, query)
        if devdocs_result:
            results.append(f"## DevDocs.io Documentation\n\n{devdocs_result}")
    except Exception as e:
        logger.warning("DevDocs lookup failed: %s", e)
    
    if results:
        return "\n\n---\n\n".join(results)
    else:
        return f"No documentation found for '{library}' with query '{query}'. Try a different library name or query."


@traceable(name="query_context7", run_type="retriever", tags=["documentation", "context7"])
async def _query_context7(library: str, query: str) -> str:
    """Query Context7 MCP API for library documentation."""
    async with httpx.AsyncClient(timeout=15.0) as client:
        # Step 1: Resolve library ID
        resolve_payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "resolve-library-id",
                "arguments": {"libraryName": library}
            }
        }
        
        headers = {"Content-Type": "application/json"}
        if CONTEXT7_API_KEY:
            headers["Authorization"] = f"Bearer {CONTEXT7_API_KEY}"
        
        response = await client.post(CONTEXT7_API_URL, json=resolve_payload, headers=headers)
        response.raise_for_status()
        resolve_result = response.json()
        
        # Extract library ID from response
        if "result" not in resolve_result or "content" not in resolve_result["result"]:
            logger.debug("Context7: No library found for %s", library)
            return ""
        
        content = resolve_result["result"]["content"]
        if not content or not isinstance(content, list):
            return ""
        
        # Parse the text content to find library ID
        text_content = content[0].get("text", "") if content else ""
        if not text_content or "No libraries found" in text_content:
            return ""
        
        # Extract first library ID (format: /library-name/version)
        lines = text_content.strip().split("\n")
        library_id = None
        for line in lines:
            if line.startswith("- "):
                # Extract ID from format: "- library-name: /library/version"
                parts = line.split(": ")
                if len(parts) >= 2:
                    library_id = parts[1].strip()
                    break
        
        if not library_id:
            return ""
        
        logger.info("Context7: Resolved %s -> %s", library, library_id)
        
        # Step 2: Query documentation
        query_payload = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "get-library-docs",
                "arguments": {
                    "context7CompatibleLibraryID": library_id,
                    "topic": query
                }
            }
        }
        
        response = await client.post(CONTEXT7_API_URL, json=query_payload, headers=headers)
        response.raise_for_status()
        docs_result = response.json()
        
        if "result" in docs_result and "content" in docs_result["result"]:
            doc_content = docs_result["result"]["content"]
            if doc_content and isinstance(doc_content, list):
                return doc_content[0].get("text", "")[:8000]  # Limit size
        
        return ""


@traceable(name="query_devdocs", run_type="retriever", tags=["documentation", "devdocs"])
async def _query_devdocs(library: str, query: str) -> str:
    """Query DevDocs.io API for documentation."""
    async with httpx.AsyncClient(timeout=15.0) as client:
        # Map common library names to DevDocs slugs (verified against docs.json)
        # Note: Some libraries like pydantic, langchain are NOT available on DevDocs
        slug_map = {
            "react": "react",
            "vue": "vue~3",
            "angular": "angular",
            "fastapi": "fastapi",
            "flask": "flask",
            "django": "django~5.2",
            "express": "express",
            "nextjs": "next.js",
            "next": "next.js",
            "typescript": "typescript",
            "python": "python~3.12",
            "rust": "rust",
            "go": "go",
            "nodejs": "node",
            "node": "node",
            "pandas": "pandas~2",
            "numpy": "numpy~2",
            "pytorch": "pytorch",
            "tensorflow": "tensorflow~2",
            "sqlalchemy": "sqlalchemy~2",
            "tailwind": "tailwindcss",
            "tailwindcss": "tailwindcss",
            "docker": "docker",
            "git": "git",
            "redis": "redis",
            "postgresql": "postgresql~16",
            "postgres": "postgresql~16",
            "mongodb": "mongoose",
            "graphql": "graphql",
            "webpack": "webpack~5",
            "vite": "vite",
            "jest": "jest",
            "cypress": "cypress",
            "axios": "axios",
            "lodash": "lodash~4",
            "moment": "moment",
            "d3": "d3~7",
            "sass": "sass",
            "css": "css",
            "html": "html",
            "dom": "dom",
            "javascript": "javascript",
            "js": "javascript",
        }
        
        slug = slug_map.get(library.lower(), library.lower())
        
        # Libraries not available on DevDocs - return early with helpful message
        not_available = {"pydantic", "langchain", "langgraph", "openai", "anthropic", "groq", "huggingface", "transformers"}
        if library.lower() in not_available:
            return f"Note: {library} documentation is not available on DevDocs.io. The agent can still help based on its training knowledge."
        
        try:
            # Get doc index from devdocs.io
            index_url = f"https://devdocs.io/docs/{slug}/index.json"
            response = await client.get(index_url)
            
            if response.status_code == 404:
                logger.debug("DevDocs: No docs found for slug %s", slug)
                return ""
            
            response.raise_for_status()
            index = response.json()
            
            # Search entries for matching items
            entries = index.get("entries", [])
            query_lower = query.lower()
            
            # Find matching entries by name or path
            matches = []
            for entry in entries:
                name = entry.get("name", "").lower()
                path = entry.get("path", "").lower()
                entry_type = entry.get("type", "").lower()
                if query_lower in name or query_lower in path or query_lower in entry_type:
                    matches.append(entry)
                    if len(matches) >= 5:  # Collect more matches
                        break
            
            if not matches:
                # Try partial word match
                query_words = query_lower.split()
                for entry in entries:
                    name = entry.get("name", "").lower()
                    entry_type = entry.get("type", "").lower()
                    if any(word in name or word in entry_type for word in query_words):
                        matches.append(entry)
                        if len(matches) >= 5:
                            break
            
            if not matches:
                return f"Found docs for {library} but no entries matching '{query}'. Available topics include: {', '.join(e['name'] for e in entries[:10])}"
            
            # Fetch content from the db.json (documents CDN)
            db_url = f"https://documents.devdocs.io/{slug}/db.json"
            db_response = await client.get(db_url)
            
            if db_response.status_code != 200:
                logger.debug("DevDocs: Could not fetch db.json for %s", slug)
                return f"Found matching topics for {library}: {', '.join(m['name'] for m in matches[:5])}"
            
            db = db_response.json()
            
            # Get content for matched entries
            results = []
            for match in matches[:3]:  # Limit to 3 full docs
                path = match.get("path", "")
                # db.json uses path without anchor (fragment)
                base_path = path.split('#')[0] if '#' in path else path
                
                if base_path in db:
                    html = db[base_path]
                    # Convert HTML to readable text
                    text = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL)
                    text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL)
                    text = re.sub(r'<pre[^>]*>(.*?)</pre>', r'\n```\n\1\n```\n', text, flags=re.DOTALL)
                    text = re.sub(r'<code[^>]*>(.*?)</code>', r'`\1`', text, flags=re.DOTALL)
                    text = re.sub(r'<h([1-6])[^>]*>(.*?)</h\1>', r'\n\n## \2\n\n', text, flags=re.DOTALL)
                    text = re.sub(r'<li[^>]*>(.*?)</li>', r'- \1\n', text, flags=re.DOTALL)
                    text = re.sub(r'<p[^>]*>(.*?)</p>', r'\1\n\n', text, flags=re.DOTALL)
                    text = re.sub(r'<[^>]+>', '', text)
                    text = re.sub(r'\n{3,}', '\n\n', text)
                    text = re.sub(r' +', ' ', text).strip()
                    
                    if text:
                        results.append(f"### {match['name']}\n\n{text[:4000]}")
            
            return "\n\n---\n\n".join(results) if results else ""
            
        except Exception as e:
            logger.warning("DevDocs query failed: %s", e)
            return ""


# ── Graph Nodes ───────────────────────────────────────────────────

@traceable(name="enrich_context_node", run_type="chain", tags=["enrichment"])
async def enrich_context(state: AgentState) -> dict:
    """First node: gather context BEFORE calling the LLM.
    
    Only runs on the FIRST turn of a conversation. On continuation turns
    (when tool results come back), we skip enrichment to avoid wasting
    time re-searching for the same context.
    """
    workspace_id = state['workspace_id']
    attached_files = state.get('attached_files', {})
    
    # Skip enrichment if we already have context (continuation turn)
    existing_context = state.get('enriched_context', '')
    if existing_context:
        logger.info("[enrich_context] Skipping - already have %d chars context", len(existing_context))
        return {}  # Don't overwrite existing context
    
    # Skip enrichment if the last message is a tool result (continuation)
    if state['messages'] and isinstance(state['messages'][-1], ToolMessage):
        logger.info("[enrich_context] Skipping - last message is ToolMessage (continuation)")
        return {}
    
    logger.info("[enrich_context] Starting enrichment for workspace=%s", workspace_id)
    
    # Get the user's question from the last human message
    question = ""
    for msg in reversed(state['messages']):
        if isinstance(msg, HumanMessage):
            question = msg.content
            break
    
    if not question:
        logger.warning("[enrich_context] No question found in messages")
        return {"enriched_context": ""}
    
    logger.info("[enrich_context] Question: %s", question[:100])
    
    # Build pre-enriched context
    context = await build_pre_enrichment(workspace_id, question, attached_files)
    
    logger.info("[enrich_context] Built context with %d chars", len(context))
    
    return {"enriched_context": context}


@traceable(name="call_model_node", run_type="chain", tags=["llm"])
async def call_model(state: AgentState) -> dict:
    """The 'Brain' node - LLM reasoning with full context."""
    enriched_context = state.get('enriched_context', '')
    
    logger.info("[call_model] enriched_context length=%d", len(enriched_context))
    
    # Build messages with system prompt and context
    messages_to_send = []
    
    # System prompt
    messages_to_send.append(SystemMessage(content=SYSTEM_PROMPT))
    
    # Add enriched context as a system message if we have it
    if enriched_context:
        context_msg = f"## Pre-gathered Context\n\n{enriched_context}\n\n---\n\nNow, answer the user's question using this context. If you need more information, use the available tools."
        messages_to_send.append(SystemMessage(content=context_msg))
        logger.info("[call_model] Added context message, total messages: %d", len(messages_to_send))
    else:
        logger.warning("[call_model] No enriched context to add!")
    
    # Add conversation history
    messages_to_send.extend(state['messages'])
    
    # Initialize model
    model_name = os.getenv("GROQ_MODEL", "moonshotai/kimi-k2-instruct-0905")
    model = ChatGroq(model=model_name, temperature=0.1)
    
    # Bind all tools
    model_with_tools = model.bind_tools(ALL_TOOLS)
    
    # Call the model
    response = await model_with_tools.ainvoke(messages_to_send)
    
    logger.info("[call_model] Got response, tool_calls=%s", 
                [tc['name'] for tc in response.tool_calls] if response.tool_calls else "none")
    
    return {"messages": [response]}


@traceable(name="execute_server_tools", run_type="chain", tags=["server-tools"])
async def execute_server_tools(state: AgentState) -> dict:
    """Execute tools that run on the server (search, trace, impact)."""
    workspace_id = state['workspace_id']
    last_message = state['messages'][-1]
    tool_outputs = []
    
    for tool_call in last_message.tool_calls:
        tool_name = tool_call['name']
        args = tool_call['args']
        
        try:
            if tool_name == "codebase_search":
                query = args.get('query', '')
                logger.info("codebase_search: query=%s, workspace=%s", query, workspace_id)
                query_emb = await embeddings.embed_query(query)
                results = await store.vector_search(workspace_id, query_emb, top_k=8)
                logger.info("codebase_search: got %d results", len(results))
                
                if results:
                    # Extract symbol dicts and build context
                    flat_results = [r["symbol"] for r in results]
                    content = chat_utils.build_context_from_results(flat_results)
                    logger.info("codebase_search: built context len=%d", len(content))
                else:
                    content = "No results found for this query. Try a different search term or use read_file to explore specific files."
                
            elif tool_name == "trace_call_chain":
                symbol = args.get('symbol_name', '')
                direction = args.get('direction', 'both')
                result = await store.trace_call_chain(workspace_id, symbol, direction=direction, max_depth=3)
                content = json.dumps(result, indent=2)
                
            elif tool_name == "impact_analysis":
                symbol = args.get('symbol_name', '')
                result = await store.impact_analysis(workspace_id, symbol, max_depth=3)
                content = json.dumps(result, indent=2)
            
            elif tool_name == "lookup_documentation":
                library = args.get('library', '')
                query = args.get('query', '')
                content = await _fetch_documentation(library, query)
                
            else:
                # Not a server tool, skip
                continue
                
        except Exception as e:
            content = f"Error executing {tool_name}: {e}"
            logger.error("Server tool %s failed: %s", tool_name, e)
            
        tool_outputs.append(ToolMessage(
            content=content,
            tool_call_id=tool_call['id']
        ))
    
    return {"messages": tool_outputs}


# ── Router Logic ──────────────────────────────────────────────────

def router(state: AgentState) -> Literal["server_tools", "pause", "end"]:
    """Decides whether to run server tools, pause for IDE, or finish."""
    last_message = state['messages'][-1]
    
    # If no tool calls, we're done
    if not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
        return "end"
    
    has_server_tools = False
    has_ide_tools = False
    
    for tc in last_message.tool_calls:
        if tc['name'] in IDE_TOOL_NAMES:
            has_ide_tools = True
        elif tc['name'] in SERVER_TOOL_NAMES:
            has_server_tools = True
    
    # Priority: if any IDE tools, pause for IDE callback
    # (server tools should have been executed already in previous iterations)
    if has_ide_tools:
        return "pause"
    
    if has_server_tools:
        return "server_tools"
    
    return "end"


def should_enrich(state: AgentState) -> Literal["enrich", "skip"]:
    """Only enrich on first message (not after tool results)."""
    # If we already have enriched context, skip
    if state.get('enriched_context'):
        return "skip"
    
    # If the last message is a tool result, skip enrichment
    if state['messages'] and isinstance(state['messages'][-1], ToolMessage):
        return "skip"
    
    return "enrich"


# ── Graph Construction ────────────────────────────────────────────

def create_agent():
    """Build the LangGraph agent."""
    workflow = StateGraph(AgentState)

    # Nodes
    workflow.add_node("enrich", enrich_context)
    workflow.add_node("agent", call_model)
    workflow.add_node("server_tools", execute_server_tools)

    # Entry: check if we need enrichment
    workflow.set_entry_point("enrich")
    
    # After enrichment, always call agent
    workflow.add_edge("enrich", "agent")
    
    # After agent, route based on tool calls
    workflow.add_conditional_edges(
        "agent",
        router,
        {
            "server_tools": "server_tools",
            "pause": END,  # Return to API/IDE for tool execution
            "end": END
        }
    )
    
    # After server tools, call agent again for next reasoning step
    workflow.add_edge("server_tools", "agent")

    return workflow.compile()


# Global agent instance
forge_agent = create_agent()
