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

### Search Tools (use in this priority order for finding code)
1. `codebase_search(query)`: **USE FIRST.** Semantic search — find code by meaning. Fast, searches the pre-built index.
2. `grep(pattern, path, glob)`: Literal/regex text search using ripgrep. Fast, respects .gitignore, skips binaries. Use when you know the exact string.
3. `trace_call_chain(symbol_name, direction, max_depth)`: Find what calls a function or what it calls.
4. `impact_analysis(symbol_name, max_depth)`: Find all code affected by changing a symbol.
5. `lookup_documentation(library, query)`: Look up official docs for libraries/frameworks.

### File Tools (for reading and editing)
- `read_file(path, start_line, end_line)`: Read file contents. ALWAYS do this before editing.
- `replace_in_file(path, old_str, new_str)`: Replace exact text in a file. old_str must match exactly.
- `write_to_file(path, content)`: Write entire file. Only for new files.
- `execute_command(command)`: Run shell commands (git, builds, tests, etc.)

### LSP Tools (for type checking and code intelligence)
- `lsp_go_to_definition(path, line, column)`: Jump to where a symbol is defined.
- `lsp_find_references(path, line, column)`: Find all usages of a symbol.
- `lsp_hover(path, line, column)`: Get type info and docs for a symbol.
- `lsp_rename(path, line, column, new_name)`: Safe rename across the workspace.

## SEARCH RULES — CRITICAL

**NEVER use `execute_command` with `grep` or `find` for searching code.**
- WRONG: `execute_command(command="grep -rn 'foo' .")`  ← SLOW, scans binaries, can escape workspace
- RIGHT: `grep(pattern="foo")`                          ← Uses ripgrep, fast, safe
- BEST:  `codebase_search(query="foo function")`        ← Semantic, finds related code too

The `grep` tool uses ripgrep internally (respects .gitignore, skips binaries, max 100KB files).
The `execute_command` tool should ONLY be used for: git, builds, tests, package managers, linters.

## Workflow for Refactoring

1. Use `codebase_search` or `grep` to find ALL occurrences (NEVER `execute_command` with grep)
2. Use `read_file` on each file to see the exact code around each occurrence
3. Use `replace_in_file` on each file to make the change
4. Use `grep` again to verify no occurrences remain
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

REMEMBER: You must CALL the tools. Do not write code blocks showing tool calls — actually invoke them.

## FORMATTING GUIDELINES

When including diagrams in your response:
- **Mermaid diagrams**: ALWAYS wrap in a fenced code block with the `mermaid` language tag:
  ```mermaid
  graph TD
    A[Start] --> B[End]
  ```
  The IDE will render these as interactive diagrams.
- **Code examples**: Use proper language tags (```python, ```typescript, etc.)
- **File references**: Use markdown links like `[filename](path/to/file)`"""

# Supplementary prompt injected for complex tasks that require a plan
PLANNING_PROMPT = """## PLANNING MODE — REQUIRED

This is a complex task that requires careful planning before execution.

**You MUST produce a structured plan BEFORE making any file changes.**

Output your plan as a numbered list under a `## Plan` heading. Each step should be:
- Concrete and actionable (not vague like "understand the code")
- Scoped to a single logical unit of work
- Ordered by dependency (do prerequisite steps first)

Example format:

## Plan
1. Read auth/session.py to understand current session handling
2. Create auth/jwt.py with JWT token generation and validation utilities
3. Update auth/login.py to return JWT tokens instead of session cookies
4. Update middleware/auth.py to validate JWT from Authorization header
5. Remove session-related code from auth/session.py
6. Update tests in tests/test_auth.py for the new JWT flow
7. Run `pytest tests/test_auth.py` and fix any failures

**Rules:**
- Aim for 3-8 steps (more for large tasks, fewer for smaller ones)
- Include a verification step at the end (run tests, type-check, build)
- Do NOT start executing tools yet — just output the plan
- After the plan, STOP and wait for confirmation to proceed

Once you have a plan, you will execute it step by step."""

# Prompt injected when executing with an active plan
PLAN_EXECUTION_PROMPT = """## PLAN EXECUTION MODE

You have a plan and you're executing it step by step.

{plan_display}

**Instructions:**
- Focus ONLY on the current step. Do NOT jump ahead.
- Use tools to complete the current step.
- When the current step is done, state "Step {current_step} complete" clearly.
- If a step fails, explain why and attempt to fix it before moving on.
- After ALL steps are done, run the verification step and report the final result."""


# ── State Definition ──────────────────────────────────────────────

class PlanStep(TypedDict):
    """A single step in the agent's execution plan."""
    number: int
    description: str
    status: str  # "pending", "in_progress", "done", "failed"


class AgentState(TypedDict):
    """The state of the agent graph."""
    messages: Annotated[list[BaseMessage], add_messages]
    workspace_id: str
    # Pre-enriched context (set by first node)
    enriched_context: str
    # Attached files from IDE (live file contents)
    attached_files: dict[str, str]  # path -> content
    # ── Task decomposition ────────────────────────────────────
    # Complexity: "simple" (Q&A, explain) or "complex" (multi-file, refactor, feature)
    task_complexity: str
    # Structured plan for complex tasks (empty for simple)
    plan_steps: list[PlanStep]
    # 1-indexed step the agent is currently executing (0 = no plan)
    current_step: int
    # Whether the planning phase has completed
    plan_complete: bool


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


# ── Task Complexity Classification ────────────────────────────────

# Keywords / patterns that signal a complex, multi-step task
_COMPLEX_VERBS = {
    "refactor", "restructure", "migrate", "rewrite", "redesign",
    "implement", "build", "create", "add", "develop",
    "convert", "port", "upgrade", "replace", "move",
    "split", "extract", "merge", "combine", "integrate",
    "optimize", "improve", "enhance", "fix all", "update all",
}

_COMPLEX_SIGNALS = {
    "across", "multiple files", "multi-file", "all files",
    "entire", "whole", "codebase-wide", "project-wide",
    "step by step", "end to end", "new feature",
    "authentication", "authorization", "database", "api",
}


def classify_task_complexity(question: str) -> str:
    """
    Classify whether a user's request is 'simple' or 'complex'.
    
    Simple: questions, explanations, single-file reads, quick lookups.
    Complex: multi-file edits, refactors, new features, migrations.
    
    This determines whether we force a planning phase before execution.
    """
    q_lower = question.lower().strip()
    
    # ── Definitely simple ──
    # Questions that ask for information, not action
    if q_lower.startswith(("what ", "how does ", "explain ", "show me ", "where ",
                           "why ", "which ", "describe ", "tell me ", "can you explain")):
        # Unless they combine with complex signals
        if not any(sig in q_lower for sig in ("and then", "and also", "after that")):
            return "simple"
    
    # Very short requests are usually simple
    if len(q_lower.split()) <= 5:
        return "simple"
    
    # ── Check for complex signals ──
    score = 0
    
    # Complex action verbs
    for verb in _COMPLEX_VERBS:
        if verb in q_lower:
            score += 2
            break
    
    # Multi-scope signals
    for signal in _COMPLEX_SIGNALS:
        if signal in q_lower:
            score += 1
    
    # Multiple files/components mentioned
    file_mentions = len(re.findall(r'\b\w+\.\w{1,4}\b', question))  # file.ext patterns
    if file_mentions >= 2:
        score += 1
    
    # Multiple action words (compound tasks)
    action_count = sum(1 for v in _COMPLEX_VERBS if v in q_lower)
    if action_count >= 2:
        score += 1
    
    # "and" connecting clauses often means multi-step
    if re.search(r'\band\b.*\b(then|also|after|next)\b', q_lower):
        score += 1
    
    return "complex" if score >= 2 else "simple"


# ── Plan Parsing ──────────────────────────────────────────────────

def parse_plan_from_response(text: str) -> list[PlanStep]:
    """
    Parse a structured plan from the LLM's response.
    
    Expects the LLM to produce a numbered list, e.g.:
    
    ## Plan
    1. Read the current auth module to understand the structure
    2. Create JWT utility functions in auth/jwt.py
    3. Update login endpoint to return JWT tokens
    4. Update middleware to validate JWT instead of sessions
    5. Run tests and fix any issues
    
    Returns a list of PlanStep dicts.
    """
    steps: list[PlanStep] = []
    
    # Look for numbered list items (1. xxx, 2. xxx, etc.)
    pattern = r'^\s*(\d+)[.)]\s*(.+?)\s*$'
    
    in_plan_section = False
    for line in text.split('\n'):
        stripped = line.strip()
        
        # Detect plan section header
        if re.match(r'^#{1,3}\s*(Plan|Task Plan|Execution Plan|Steps)', stripped, re.IGNORECASE):
            in_plan_section = True
            continue
        
        # Detect end of plan section (next heading or empty line after steps)
        if in_plan_section and stripped.startswith('#') and not re.match(r'^#{1,3}\s*(Plan|Step)', stripped, re.IGNORECASE):
            break
        
        # Parse numbered items
        match = re.match(pattern, stripped)
        if match:
            in_plan_section = True  # Auto-detect plan even without header
            number = int(match.group(1))
            description = match.group(2).strip()
            # Clean markdown formatting: **bold**, `code`, leading/trailing *
            description = re.sub(r'\*\*(.+?)\*\*', r'\1', description)
            description = re.sub(r'`(.+?)`', r'\1', description)
            description = description.strip('* ')
            
            if description:
                steps.append(PlanStep(
                    number=number,
                    description=description,
                    status="pending"
                ))
    
    # Re-number sequentially in case of gaps
    for i, step in enumerate(steps):
        step["number"] = i + 1
    
    return steps


def format_plan_for_prompt(steps: list[PlanStep], current_step: int) -> str:
    """Format the plan as a prompt section for the LLM to reference."""
    if not steps:
        return ""
    
    lines = ["## Your Execution Plan\n"]
    for step in steps:
        status_icon = {
            "pending": "⬜",
            "in_progress": "🔄",
            "done": "✅",
            "failed": "❌",
        }.get(step["status"], "⬜")
        
        marker = " ← YOU ARE HERE" if step["number"] == current_step else ""
        lines.append(f"{status_icon} {step['number']}. {step['description']}{marker}")
    
    lines.append("")
    lines.append(f"**Current step: {current_step} of {len(steps)}**")
    lines.append("Focus on completing the current step. Use tools to execute it, then move to the next.")
    
    return "\n".join(lines)


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

def _get_last_question(state: AgentState) -> str:
    """Extract the last human question from message history."""
    for msg in reversed(state['messages']):
        if isinstance(msg, HumanMessage):
            return msg.content
    return ""


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
    
    question = _get_last_question(state)
    if not question:
        logger.warning("[enrich_context] No question found in messages")
        return {"enriched_context": ""}
    
    logger.info("[enrich_context] Question: %s", question[:100])
    
    # Build pre-enriched context
    context = await build_pre_enrichment(workspace_id, question, attached_files)
    
    logger.info("[enrich_context] Built context with %d chars", len(context))
    
    return {"enriched_context": context}


@traceable(name="classify_task_node", run_type="chain", tags=["planning"])
async def classify_task(state: AgentState) -> dict:
    """Classify whether the task is simple or complex.
    
    Complex tasks get routed through a planning phase before execution.
    Simple tasks go directly to the agent.
    
    Runs only on the first turn; on continuations (tool results coming
    back) we preserve the existing classification.
    """
    # If already classified (continuation turn), keep it
    if state.get('task_complexity'):
        logger.info("[classify] Skipping - already classified as %s", state['task_complexity'])
        return {}
    
    # If last message is a tool result, this is a continuation — skip
    if state['messages'] and isinstance(state['messages'][-1], ToolMessage):
        return {}
    
    question = _get_last_question(state)
    if not question:
        return {"task_complexity": "simple", "plan_steps": [], "current_step": 0, "plan_complete": True}
    
    complexity = classify_task_complexity(question)
    logger.info("[classify] Task complexity: %s (question: %s)", complexity, question[:80])
    
    if complexity == "simple":
        return {
            "task_complexity": "simple",
            "plan_steps": [],
            "current_step": 0,
            "plan_complete": True,  # No plan needed
        }
    else:
        return {
            "task_complexity": "complex",
            "plan_steps": [],       # Will be filled by plan_task node
            "current_step": 0,
            "plan_complete": False,
        }


@traceable(name="plan_task_node", run_type="chain", tags=["planning", "llm"])
async def plan_task(state: AgentState) -> dict:
    """Generate a structured plan for complex tasks.
    
    Calls the REASONING model with planning-specific instructions.
    The LLM produces a numbered plan that gets parsed into PlanStep objects.
    No tools are bound — the LLM must ONLY output a plan, not execute.
    """
    enriched_context = state.get('enriched_context', '')
    question = _get_last_question(state)
    
    logger.info("[plan_task] Generating plan for: %s", question[:100])
    
    # Build planning prompt
    messages_to_send = [
        SystemMessage(content=SYSTEM_PROMPT),
        SystemMessage(content=PLANNING_PROMPT),
    ]
    
    # Add enriched context
    if enriched_context:
        ctx = enriched_context[:60_000]
        messages_to_send.append(SystemMessage(
            content=f"## Pre-gathered Context\n\n{ctx}\n\n---\n"
        ))
    
    # Add conversation history (just the human messages, not tool noise)
    for msg in state['messages']:
        if isinstance(msg, HumanMessage):
            messages_to_send.append(msg)
    
    # Use the reasoning model (stronger) for planning — NO tools bound
    model = ChatGroq(model=REASONING_MODEL, temperature=0.1)
    response = await model.ainvoke(messages_to_send)
    
    plan_text = response.content
    logger.info("[plan_task] Raw plan response (%d chars): %s", len(plan_text), plan_text[:300])
    
    # Parse the structured plan
    steps = parse_plan_from_response(plan_text)
    
    if steps:
        logger.info("[plan_task] Parsed %d plan steps:", len(steps))
        for s in steps:
            logger.info("  %d. %s", s["number"], s["description"])
        
        # Mark step 1 as in_progress
        steps[0]["status"] = "in_progress"
        
        # Build a short plan summary for the AI message (instead of raw LLM output)
        plan_summary = "## Plan\n" + "\n".join(
            f"{s['number']}. {s['description']}" for s in steps
        )
        
        return {
            "plan_steps": steps,
            "current_step": 1,
            "plan_complete": True,  # Planning phase done, ready to execute
            "messages": [
                # The plan as an AI message
                AIMessage(content=plan_summary),
                # A kick-start message so the LLM knows to START executing
                HumanMessage(content="Plan approved. Now execute step 1 using the tools. Do NOT just describe what to do — actually call the tools."),
            ],
        }
    else:
        # Failed to parse a plan — fall through to direct execution
        logger.warning("[plan_task] Could not parse plan steps, falling through to direct agent")
        return {
            "plan_steps": [],
            "current_step": 0,
            "plan_complete": True,
            "task_complexity": "simple",  # Downgrade
            "messages": [AIMessage(content=plan_text)],
        }


# ── Context Window Management ─────────────────────────────────────

# Rough char budget: 120K tokens ≈ 480K chars.  Reserve room for
# system prompt (~4K), enriched context (~20K), and model output (~4K).
MAX_HISTORY_CHARS = 200_000  # ~50K tokens for conversation history
MAX_TOOL_MSG_CHARS = 30_000  # Truncate individual tool results


def _truncate_messages(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Truncate oversized tool messages and trim old history if needed."""
    result = []
    total_chars = 0
    
    for msg in messages:
        if isinstance(msg, ToolMessage) and len(msg.content) > MAX_TOOL_MSG_CHARS:
            # Truncate large tool results (file dumps, command output)
            truncated = msg.content[:MAX_TOOL_MSG_CHARS] + (
                f"\n\n... (truncated from {len(msg.content)} chars)"
            )
            msg = ToolMessage(
                content=truncated,
                tool_call_id=msg.tool_call_id,
            )
        
        total_chars += len(msg.content)
        result.append(msg)
    
    # If total history is still too large, keep first 2 + last N messages
    if total_chars > MAX_HISTORY_CHARS:
        # Always keep: first HumanMessage (the question) + last few messages
        # Drop middle messages (old tool call/result pairs)
        if len(result) > 6:
            kept_start = result[:2]  # First human msg + first AI response
            kept_end = result[-4:]    # Last 4 messages (most recent context)
            dropped = len(result) - 6
            summary = AIMessage(content=f"[{dropped} earlier messages omitted to fit context window]")
            result = kept_start + [summary] + kept_end
            logger.warning("[truncate] Dropped %d messages, kept %d", dropped, len(result))
    
    return result


# ── Multi-Model Routing ───────────────────────────────────────────
#
# Strategy: use the best model for each phase of the agent loop.
#
#   REASONING model (GROQ_MODEL):
#     First call with enriched context + user question.
#     Needs strong reasoning to understand the codebase and plan.
#     Default: moonshotai/kimi-k2-instruct-0905
#
#   TOOL model (GROQ_TOOL_MODEL):
#     Subsequent calls that process tool results and decide next action.
#     Needs fast responses + reliable function calling.
#     Default: openai/gpt-oss-20b (3.6B active, native tool use)

REASONING_MODEL = os.getenv("GROQ_MODEL", "moonshotai/kimi-k2-instruct-0905")
TOOL_MODEL = os.getenv("GROQ_TOOL_MODEL", "openai/gpt-oss-20b")


def _pick_model(messages: list[BaseMessage], is_plan_execution: bool = False) -> str:
    """Pick model based on conversation phase.
    
    - Planning / first call → REASONING_MODEL
    - After tool results → TOOL_MODEL (faster iterations)
    - Plan execution (has plan context) → REASONING_MODEL for first step, TOOL for rest
    """
    has_tool_messages = any(isinstance(m, ToolMessage) for m in messages)
    
    if has_tool_messages:
        return TOOL_MODEL
    return REASONING_MODEL


@traceable(name="call_model_node", run_type="chain", tags=["llm"])
async def call_model(state: AgentState) -> dict:
    """The 'Brain' node - LLM reasoning with full context.
    
    Now plan-aware: if there's an active plan, it injects the plan display
    and current step instructions into the system prompt.
    """
    enriched_context = state.get('enriched_context', '')
    plan_steps = state.get('plan_steps', [])
    current_step = state.get('current_step', 0)
    
    logger.info("[call_model] enriched_context=%d chars, plan_steps=%d, current_step=%d",
                len(enriched_context), len(plan_steps), current_step)
    
    # Build messages with system prompt and context
    messages_to_send = []
    
    # System prompt
    messages_to_send.append(SystemMessage(content=SYSTEM_PROMPT))
    
    # ── Plan execution context ────────────────────────────────
    if plan_steps and current_step > 0:
        plan_display = format_plan_for_prompt(plan_steps, current_step)
        exec_prompt = PLAN_EXECUTION_PROMPT.format(
            plan_display=plan_display,
            current_step=current_step,
        )
        messages_to_send.append(SystemMessage(content=exec_prompt))
        logger.info("[call_model] Injected plan execution context (step %d/%d)", 
                    current_step, len(plan_steps))
    
    # Add enriched context as a system message if we have it
    if enriched_context:
        # Cap enriched context to avoid blowing up on its own
        ctx = enriched_context[:80_000] if len(enriched_context) > 80_000 else enriched_context
        context_msg = f"## Pre-gathered Context\n\n{ctx}\n\n---\n\nNow, answer the user's question using this context. If you need more information, use the available tools."
        messages_to_send.append(SystemMessage(content=context_msg))
        logger.info("[call_model] Added context message, total messages: %d", len(messages_to_send))
    else:
        logger.warning("[call_model] No enriched context to add!")
    
    # Add conversation history (with truncation)
    history = _truncate_messages(state['messages'])
    messages_to_send.extend(history)
    
    total_chars = sum(len(m.content) for m in messages_to_send)
    
    # Pick model based on conversation phase
    model_name = _pick_model(state['messages'], is_plan_execution=bool(plan_steps))
    logger.info("[call_model] Using %s | %d messages, %d chars", model_name, len(messages_to_send), total_chars)
    
    model = ChatGroq(model=model_name, temperature=0.1)
    
    # Bind all tools
    model_with_tools = model.bind_tools(ALL_TOOLS)
    
    # Call the model
    response = await model_with_tools.ainvoke(messages_to_send)
    
    logger.info("[call_model] Got response from %s, tool_calls=%s", 
                model_name,
                [tc['name'] for tc in response.tool_calls] if response.tool_calls else "none")
    
    # ── Plan step advancement ─────────────────────────────────
    # If there's an active plan, check if the LLM signalled step completion
    updates: dict = {"messages": [response]}
    
    if plan_steps and current_step > 0:
        resp_text = (response.content or "").lower()
        
        # Detect step completion signals
        step_done = (
            f"step {current_step} complete" in resp_text
            or f"step {current_step} done" in resp_text
            or f"completed step {current_step}" in resp_text
            or f"finished step {current_step}" in resp_text
        )
        
        if step_done and current_step <= len(plan_steps):
            # Mark current step done
            new_steps = [dict(s) for s in plan_steps]  # Copy
            new_steps[current_step - 1]["status"] = "done"
            
            next_step = current_step + 1
            if next_step <= len(new_steps):
                new_steps[next_step - 1]["status"] = "in_progress"
                logger.info("[call_model] ✅ Step %d done, advancing to step %d", current_step, next_step)
            else:
                logger.info("[call_model] ✅ Step %d done — ALL STEPS COMPLETE", current_step)
                next_step = 0  # Signal all done
            
            updates["plan_steps"] = new_steps
            updates["current_step"] = next_step
    
    return updates


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

def classify_router(state: AgentState) -> Literal["plan", "agent"]:
    """After classify: route complex tasks to plan, simple to agent."""
    complexity = state.get('task_complexity', 'simple')
    plan_complete = state.get('plan_complete', True)
    
    if complexity == "complex" and not plan_complete:
        logger.info("[classify_router] → plan (complex task, no plan yet)")
        return "plan"
    
    logger.info("[classify_router] → agent (simple or plan already done)")
    return "agent"


def agent_router(state: AgentState) -> Literal["server_tools", "pause", "end", "continue_plan"]:
    """After agent: route to server tools, pause for IDE, or finish.
    
    New: if a plan step was just completed and there are more steps,
    route back to agent to continue with the next step.
    """
    last_message = state['messages'][-1]
    plan_steps = state.get('plan_steps', [])
    current_step = state.get('current_step', 0)
    
    # If no tool calls, check if we should continue the plan
    if not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
        # If there's an active plan with remaining steps, loop back
        if plan_steps and current_step > 0:
            remaining = [s for s in plan_steps if s['status'] in ('pending', 'in_progress')]
            if remaining:
                logger.info("[agent_router] → continue_plan (step %d, %d remaining)", current_step, len(remaining))
                return "continue_plan"
        return "end"
    
    has_server_tools = False
    has_ide_tools = False
    
    for tc in last_message.tool_calls:
        if tc['name'] in IDE_TOOL_NAMES:
            has_ide_tools = True
        elif tc['name'] in SERVER_TOOL_NAMES:
            has_server_tools = True
    
    # Priority: if any IDE tools, pause for IDE callback
    if has_ide_tools:
        return "pause"
    
    if has_server_tools:
        return "server_tools"
    
    return "end"


# ── Graph Construction ────────────────────────────────────────────
#
# New flow with task decomposition:
#
#   enrich → classify ─┬─ (complex, no plan) → plan → agent ─┬→ server_tools → agent
#                       │                                      ├→ pause (IDE tools)
#                       └─ (simple or has plan) → agent ──┬────┴→ end
#                                                         └→ continue_plan → agent
#
# The continue_plan node handles step transitions: when the agent
# finishes a step (no tool calls) but there are more plan steps,
# it injects a kick-start message and loops back to the agent.

async def continue_plan(state: AgentState) -> dict:
    """Inject a kick-start message to advance to the next plan step."""
    current_step = state.get('current_step', 0)
    plan_steps = state.get('plan_steps', [])
    
    if current_step > 0 and current_step <= len(plan_steps):
        step_desc = plan_steps[current_step - 1]["description"]
        logger.info("[continue_plan] Kicking off step %d: %s", current_step, step_desc)
        return {
            "messages": [
                HumanMessage(content=f"Continue. Execute step {current_step}: {step_desc}. Use the tools now.")
            ]
        }
    
    return {}


def create_agent():
    """Build the LangGraph agent with task decomposition."""
    workflow = StateGraph(AgentState)

    # Nodes
    workflow.add_node("enrich", enrich_context)
    workflow.add_node("classify", classify_task)
    workflow.add_node("plan", plan_task)
    workflow.add_node("agent", call_model)
    workflow.add_node("server_tools", execute_server_tools)
    workflow.add_node("continue_plan", continue_plan)

    # ── Edges ─────────────────────────────────────────────────
    
    # Entry: always start with enrichment
    workflow.set_entry_point("enrich")
    
    # After enrichment, classify the task
    workflow.add_edge("enrich", "classify")
    
    # After classify, route based on complexity
    workflow.add_conditional_edges(
        "classify",
        classify_router,
        {
            "plan": "plan",
            "agent": "agent",
        }
    )
    
    # After planning, go to agent for execution
    workflow.add_edge("plan", "agent")
    
    # After agent, route based on tool calls
    workflow.add_conditional_edges(
        "agent",
        agent_router,
        {
            "server_tools": "server_tools",
            "pause": END,           # Return to API/IDE for tool execution
            "end": END,
            "continue_plan": "continue_plan",  # Step done, more to go
        }
    )
    
    # After server tools, call agent again for next reasoning step
    workflow.add_edge("server_tools", "agent")
    
    # After continue_plan kick-start, go back to agent
    workflow.add_edge("continue_plan", "agent")

    return workflow.compile()


# Global agent instance
forge_agent = create_agent()
