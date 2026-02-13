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
  4. LLM reasons + executes with full context and tools
  5. For complex tasks, agent uses create_plan/update_plan tools
     to manage a structured plan (shown live in the IDE)
  6. For IDE tools (file ops), pause and return to IDE for execution
  7. Agent can use lookup_documentation tool for library docs on-demand
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
from langchain_core.tools import tool
from langsmith import traceable

from . import store, embeddings, chat as chat_utils
from . import llm as llm_provider

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
   - If a library/tool command fails 2+ times with the same error → use `lookup_documentation` to check for API changes
   - Don't retry the same failing command more than twice
7. **ALWAYS VERIFY YOUR WORK** before finishing (see Verification section below)

## Tools Available

### Search & Documentation Tools
1. `codebase_search(query)`: **USE FIRST for finding code.** Semantic search — find code by meaning. Fast, searches the pre-built index.
2. `lookup_documentation(library, query)`: **USE for library/framework questions.** Look up official docs when:
   - You encounter library-specific errors or unexpected behavior
   - A library command fails (e.g., CLI tools, package managers)
   - You need to know correct API usage or configuration
   - Examples: `lookup_documentation("tailwindcss", "init configuration")`, `lookup_documentation("nextjs", "app router")` 
3. `grep(pattern, path, glob)`: Literal/regex text search using ripgrep. Use when you know the exact string.
4. `trace_call_chain(symbol_name, direction)`: Find what calls a function or what it calls.
5. `impact_analysis(symbol_name)`: Find all code affected by changing a symbol.

### File Tools (for reading and editing)
- `read_file(path)`: Read file contents. ALWAYS do this before editing.
- `list_files(path, recursive)`: List files in a directory.
- `glob(pattern, path)`: Find files matching a glob pattern (e.g. '**/*.py').
- `replace_in_file(path, old_str, new_str)`: Replace exact text in a file. old_str must match exactly.
- `write_to_file(path, content)`: Write entire file. Only for new files.
- `delete_file(path)`: Delete a file.
- `apply_patch(patch)`: Apply unified diff to modify multiple files at once.

### Command & Process Tools
- `execute_command(command)`: Run shell commands (git, builds, tests, etc.)
- `execute_background(command, label)`: Start long-running processes (dev servers, watchers).
- `read_process_output(pid, lines)`: Check logs/output from background processes.
- `check_process_status(pid)`: Check if a background process is still running.
- `kill_process(pid)`: Stop a background process.
- `wait_for_port(port, timeout)`: Wait for a server to be ready on a port.
- `check_port(port)`: Check if a port is in use.
- `kill_port(port)`: Kill whatever is using a port.

### Code Intelligence Tools
- `list_code_definition_names(path)`: List all symbols (functions, classes) in a file.
- `get_symbol_definition(symbol, path)`: Jump to where a symbol is defined.
- `find_symbol_references(symbol, path)`: Find all usages of a symbol.
- `diagnostics(path, fix)`: Get linter/compiler errors, optionally auto-fix.

### LSP Tools (powered by language servers)
- `lsp_go_to_definition(path, line, column)`: Jump to definition using LSP.
- `lsp_find_references(path, line, column)`: Find all usages via LSP.
- `lsp_hover(path, line, column)`: Get type info and docs for a symbol.
- `lsp_rename(path, line, column, new_name)`: Safe rename across the workspace.

### Planning Tools
- `create_plan(steps)`: Create an execution plan. The user sees it in the IDE with live status. **Use for tasks involving 3+ steps across multiple files** (refactoring, new features, migrations). Do NOT create plans for simple questions or single-file edits.
- `update_plan(step_number, status)`: Mark a step as "done", "in_progress", or "failed". **Call this every time you finish a step** so the user can track progress.
- `discard_plan()`: Remove the plan if your approach changes or the task is simpler than expected.

**Planning rules:**
- Create a plan BEFORE starting work on complex tasks
- Keep plans to 3-8 concrete steps
- Always include a verification step at the end
- Call `update_plan` to mark each step done — the user watches your progress live
- Do NOT create plans for: questions, explanations, single-file reads, quick edits

## SEARCH RULES — CRITICAL

**NEVER use `execute_command` with `grep` or `find` for searching code.**
- WRONG: `execute_command(command="grep -rn 'foo' .")`  ← SLOW, can escape workspace
- RIGHT: `grep(pattern="foo")`                          ← Uses ripgrep, fast, safe
- BEST:  `codebase_search(query="foo function")`        ← Semantic, finds related code too

Use `codebase_search` first for semantic/conceptual queries. Use `grep` when you need exact literal matches.
The `execute_command` tool should ONLY be used for: git, builds, tests, package managers, linters.

## AVOIDING RETRY LOOPS

**If a command fails twice with the same error, STOP and investigate:**

1. **Library/CLI tool errors** (e.g., `npx foo init` fails) → Use `lookup_documentation("foo", "init")`
2. **Missing file/command** → Check if you need to install it first, or if the API changed
3. **Configuration errors** → Search docs for correct config format or create the config manually

**NEVER retry the same failing command 3+ times.** If reinstalling doesn't fix it, the problem is conceptual (wrong command, API changed, missing prerequisite).

**Examples of bad patterns (retrying same failing command):**
```
# Frontend/Build Tools
npx <tool> init  → fails
npm install <tool>  → succeeds  
npx <tool> init  → fails AGAIN  ❌ STOP - API changed or not available

# Backend/CLI
<command> --init  → fails
pip/cargo install  → succeeds
<command> --init  → fails AGAIN  ❌ STOP - command doesn't exist or needs different flags

# Database/Services  
<cli> setup  → fails
apt/brew install  → succeeds
<cli> setup  → fails AGAIN  ❌ STOP - needs config file or different approach
```

**Correct recovery patterns:**

**Option 1 - Search documentation** (when you need to learn the API):
```
<command>  → fails twice
lookup_documentation("<library>", "<topic>")  ✓ Learn the correct way
```

**Option 2 - Create config manually** (PREFERRED for config files):
```
<init-command>  → fails twice
write_to_file("<config-file>", <standard-template>)  ✓ Skip the CLI, create directly
```

**Option 3 - Try alternative approach**:
```
<command-A>  → fails twice
<command-B>  ✓ Different tool/method that achieves same goal
```

**When to use each:**
- **Documentation**: Learning new APIs, understanding errors, feature discovery
- **Manual creation**: Config files (build tools, linters, formatters), known templates
- **Alternative**: When the original tool is deprecated/broken, use modern replacement

## Workflow for Refactoring

**For renaming a symbol** (preferred — safe, atomic):
1. Use `lsp_rename(path, line, column, new_name)` — renames across the entire workspace in one step
2. Run `diagnostics` on affected files to confirm

**For structural changes** (moving code, changing signatures, rewriting logic):
1. Use `codebase_search` or `grep` to find ALL occurrences
2. Use `read_file` on each file to see the exact code
3. Use `replace_in_file` on each file to make the change
4. Use `find_symbol_references(symbol)` or `lsp_find_references` to verify all call sites are updated
5. Use `grep` to verify no old occurrences remain
6. **RUN VERIFICATION** (see below)

## VERIFICATION — MANDATORY BEFORE FINISHING

**You are NOT done until you verify your changes are correct. Use ALL available checks.**

### Step 1: Code Intelligence Checks (USE THESE FIRST)

After edits, use the IDE's built-in code intelligence to catch problems before running builds:

- **`diagnostics(path)`** — Get linter/compiler errors for each file you edited. Fast, catches most issues immediately. If fix=True, auto-fixes simple problems.
- **`find_symbol_references(symbol)`** — After renaming or moving a symbol, check that all call sites are updated. Any broken reference = runtime crash.
- **`lsp_find_references(path, line, column)`** — Same but position-based, more accurate. Use for complex refactors.
- **`lsp_hover(path, line, column)`** — Verify types are correct after edits. If hover shows `any` or `unknown`, something is wrong.

**Refactoring verification pattern:**
1. `diagnostics(path)` on every file you edited
2. `find_symbol_references(symbol)` on every symbol you renamed, moved, or changed the signature of
3. Fix any issues found
4. Then run the build check below

### Step 2: Build/Compile Check

Run the project-level build to catch cross-file issues:

| Project Type       | Verification Command                                    |
|--------------------|---------------------------------------------------------|
| Next.js            | `npm run build 2>&1 | tail -50` (catches "use client", SSR issues, imports) |
| TypeScript (non-Next) | `npx tsc --noEmit 2>&1 | head -50`                 |
| Python             | `python -m py_compile <file>` or `python -m mypy .`    |
| Rust               | `cargo check 2>&1 | tail -30`                          |
| Go                 | `go build ./... 2>&1 | tail -30`                       |
| General            | `npm test` / `pytest` / `cargo test` (if tests exist)  |

**IMPORTANT for Next.js**: `npx tsc --noEmit` does NOT catch Next.js-specific errors like missing `"use client"`. ALWAYS use `npm run build` for Next.js projects.

**If any check FAILS:**
1. Read the error output carefully
2. Fix the issues (e.g. missing imports, wrong types, broken references)
3. Re-run the checks
4. Repeat until everything passes

NEVER report "done" with errors still present. Your job is only complete when diagnostics are clean AND the build passes.

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
    # The question we enriched for (used to detect topic shifts)
    enriched_question: str
    # Attached files from IDE (live file contents)
    attached_files: dict[str, str]  # path -> content
    # ── Plan state (managed by create_plan/update_plan/discard_plan tools) ──
    plan_steps: list[PlanStep]
    # 1-indexed step the agent is currently executing (0 = no plan)
    current_step: int


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


# ── Plan Management Tools (Cloud-side) ────────────────────────────
# These let the agent create and manage its own execution plan.
# The IDE shows the plan with live status updates (checkmarks, etc.).

@tool
async def create_plan(steps: list[str]) -> str:
    """
    Create an execution plan for a complex task.
    Use this BEFORE starting work on tasks that involve multiple files,
    refactoring, new features, or anything requiring 3+ coordinated steps.
    
    Args:
        steps: List of step descriptions in execution order.
               Each step should be a concrete, actionable item.
    
    Example:
        create_plan([
            "Read auth/session.py to understand current session handling",
            "Create auth/jwt.py with JWT token generation and validation",
            "Update auth/login.py to return JWT tokens",
            "Update middleware to validate JWT from Authorization header",
            "Run tests and fix any failures"
        ])
    """
    return "PENDING_SERVER_EXECUTION"


@tool
async def update_plan(step_number: int, status: str, new_description: str = "") -> str:
    """
    Update a plan step's status or description.
    Call this as you complete each step so the user can see your progress.
    
    Args:
        step_number: Which step to update (1-indexed).
        status: New status — one of: "done", "in_progress", "failed", "pending".
        new_description: Optional new description (leave empty to keep current).
    """
    return "PENDING_SERVER_EXECUTION"


@tool
async def discard_plan() -> str:
    """
    Discard the current plan entirely.
    Use when the approach needs to change fundamentally, or the task
    turned out to be simpler than expected and doesn't need a plan.
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


@tool
def grep(pattern: str, path: str = ".", glob: str = None, case_insensitive: bool = False) -> str:
    """
    Literal/regex text search using ripgrep. Fast, respects .gitignore.
    Use when you know the exact string to find (function name, error message, import).
    For semantic/conceptual search, use codebase_search instead.
    """
    return "PENDING_IDE_EXECUTION"


@tool
def list_files(path: str, recursive: bool = False) -> str:
    """List files in a directory."""
    return "PENDING_IDE_EXECUTION"


@tool
def delete_file(path: str) -> str:
    """Delete a file from the workspace."""
    return "PENDING_IDE_EXECUTION"


@tool
def apply_patch(patch: str) -> str:
    """Apply a unified diff patch to modify multiple files at once. The patch should be in standard unified diff format."""
    return "PENDING_IDE_EXECUTION"


@tool
def glob(pattern: str, path: str = ".") -> str:
    """Find files matching a glob pattern. E.g. '**/*.py' finds all Python files."""
    return "PENDING_IDE_EXECUTION"


@tool
def diagnostics(path: str, fix: bool = False) -> str:
    """Get linter/compiler errors for a file or directory. If fix=True, attempt auto-fix where supported."""
    return "PENDING_IDE_EXECUTION"


# ── Background Process Tools (IDE-side) ───────────────────────────

@tool
def execute_background(command: str, label: str = "") -> str:
    """Start a long-running command in background (e.g. dev servers). Returns process ID for later management."""
    return "PENDING_IDE_EXECUTION"


@tool
def read_process_output(pid: int, lines: int = 100) -> str:
    """Read recent output from a background process. Useful for checking server logs."""
    return "PENDING_IDE_EXECUTION"


@tool
def check_process_status(pid: int) -> str:
    """Check if a background process is still running."""
    return "PENDING_IDE_EXECUTION"


@tool
def kill_process(pid: int) -> str:
    """Stop a background process by its PID."""
    return "PENDING_IDE_EXECUTION"


@tool
def wait_for_port(port: int, timeout: int = 30, http_check: bool = False, path: str = "/") -> str:
    """Wait for a port to become available. Set http_check=True to also verify HTTP response is healthy (recommended for web servers)."""
    return "PENDING_IDE_EXECUTION"


@tool
def check_port(port: int) -> str:
    """Check if a port is currently in use."""
    return "PENDING_IDE_EXECUTION"


@tool
def kill_port(port: int) -> str:
    """Kill whatever process is using a port."""
    return "PENDING_IDE_EXECUTION"


# ── Code Intelligence Tools (IDE-side) ────────────────────────────

@tool
def list_code_definition_names(path: str) -> str:
    """List all symbol definitions (functions, classes, etc.) in a file. Great for understanding file structure."""
    return "PENDING_IDE_EXECUTION"


@tool
def get_symbol_definition(symbol: str, path: str = "") -> str:
    """Get the definition location and source code for a symbol. Uses tree-sitter for fast local lookup."""
    return "PENDING_IDE_EXECUTION"


@tool
def find_symbol_references(symbol: str, path: str = "") -> str:
    """Find all usages/references to a symbol across the codebase."""
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
PLAN_TOOLS = [create_plan, update_plan, discard_plan]
SERVER_TOOLS = [codebase_search, trace_call_chain, impact_analysis, lookup_documentation] + PLAN_TOOLS
IDE_FILE_TOOLS = [read_file, write_to_file, replace_in_file, delete_file, apply_patch, list_files, glob]
IDE_EXEC_TOOLS = [execute_command, execute_background, read_process_output, check_process_status, kill_process]
IDE_PORT_TOOLS = [wait_for_port, check_port, kill_port]
IDE_CODE_TOOLS = [list_code_definition_names, get_symbol_definition, find_symbol_references, diagnostics]
LSP_TOOLS = [lsp_go_to_definition, lsp_find_references, lsp_hover, lsp_rename]

IDE_TOOLS = IDE_FILE_TOOLS + IDE_EXEC_TOOLS + IDE_PORT_TOOLS + IDE_CODE_TOOLS + [grep]
ALL_TOOLS = SERVER_TOOLS + IDE_TOOLS + LSP_TOOLS

IDE_TOOL_NAMES = {
    # File operations
    "read_file", "write_to_file", "replace_in_file", "delete_file", "apply_patch",
    "list_files", "glob", "grep",
    # Process management
    "execute_command", "execute_background", "read_process_output",
    "check_process_status", "kill_process",
    # Port management
    "wait_for_port", "check_port", "kill_port",
    # Code intelligence
    "list_code_definition_names", "get_symbol_definition", "find_symbol_references", "diagnostics",
    # LSP
    "lsp_go_to_definition", "lsp_find_references", "lsp_hover", "lsp_rename",
}
PLAN_TOOL_NAMES = {"create_plan", "update_plan", "discard_plan"}
SERVER_TOOL_NAMES = {"codebase_search", "trace_call_chain", "impact_analysis", "lookup_documentation"} | PLAN_TOOL_NAMES


# ── Query Decomposition ───────────────────────────────────────────

DECOMPOSE_PROMPT = """You are a code search query optimizer. Given a user's question about a codebase, extract:

1. **search_queries**: 2-4 short, focused search queries optimized for finding relevant code in a semantic code index. Each query should target a different aspect (e.g., one for the module, one for the feature, one for the pattern). Think "what would the function signatures and class names look like?" not "what is the user asking?".

2. **symbols**: 0-3 exact code symbol names (function names, class names, variable names) mentioned or implied. Only real identifiers, not English words.

If the user's open files are provided, use the imports, function names, and class names you see there to make your queries and symbols more precise. Prefer real symbol names from the files over guesses.

Respond with ONLY valid JSON:
{"search_queries": ["query1", "query2", ...], "symbols": ["SymbolName", "func_name", ...]}

Examples:
- "refactor auth to use JWT" → {"search_queries": ["authentication login session handler", "JWT token verification", "user auth middleware"], "symbols": []}
- "fix the bug in UserService.get_profile" → {"search_queries": ["UserService get_profile method", "user profile data fetch"], "symbols": ["UserService", "get_profile"]}
- "how does the payment flow work?" → {"search_queries": ["payment processing checkout", "payment gateway integration", "order payment status"], "symbols": []}
- "update the create_order function to validate inventory" → {"search_queries": ["create_order function implementation", "inventory validation stock check"], "symbols": ["create_order"]}"""


def _extract_file_hints(attached_files: dict[str, str] | None) -> str:
    """Extract useful hints from attached files to improve query decomposition.
    
    Pulls out: import statements, function/class definitions, and file paths.
    This tells the decomposer what symbols actually exist in the codebase.
    """
    if not attached_files:
        return ""
    
    hints = []
    for path, content in list(attached_files.items())[:3]:  # Limit to 3 files
        file_hints = [f"File: {path}"]
        
        for line in content.split("\n")[:100]:  # Scan first 100 lines
            stripped = line.strip()
            # Imports (Python, JS/TS, Rust, Go)
            if stripped.startswith(("import ", "from ", "use ", "require(", "const ", "export ")):
                file_hints.append(f"  {stripped[:120]}")
            # Function/class/struct definitions
            elif stripped.startswith(("def ", "class ", "async def ", "fn ", "func ", "function ", "struct ", "interface ", "type ")):
                file_hints.append(f"  {stripped[:120]}")
        
        if len(file_hints) > 1:  # More than just the filename
            hints.append("\n".join(file_hints[:15]))  # Cap per file
    
    return "\n".join(hints)


@traceable(name="decompose_query", run_type="chain", tags=["enrichment", "llm"])
async def _decompose_query(question: str, attached_files: dict[str, str] | None = None) -> tuple[list[str], list[str]]:
    """Use the tool model to decompose a user question into focused search queries + symbols.
    
    If attached files are provided, extracts imports/definitions from them
    so the LLM knows what symbols actually exist in the codebase.
    
    Returns (search_queries, symbols). Falls back to the raw question on failure.
    """
    model = llm_provider.get_tool_model(temperature=0.0)
    
    # Build the user message with file context if available
    user_content = question
    file_hints = _extract_file_hints(attached_files)
    if file_hints:
        user_content = f"{question}\n\n--- User's open files (for reference) ---\n{file_hints}"
    
    try:
        response = await model.ainvoke([
            SystemMessage(content=DECOMPOSE_PROMPT),
            HumanMessage(content=user_content),
        ])
        
        text = response.content.strip()
        # Handle models that wrap JSON in markdown code blocks
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        
        result = json.loads(text)
        queries = result.get("search_queries", [question])
        symbols = result.get("symbols", [])
        
        # Validate
        queries = [q for q in queries if isinstance(q, str) and q.strip()][:4]
        symbols = [s for s in symbols if isinstance(s, str) and s.strip()][:3]
        
        if not queries:
            queries = [question]
        
        logger.info("[decompose] %d queries, %d symbols from: %s", len(queries), len(symbols), question[:80])
        return queries, symbols
        
    except Exception as e:
        logger.warning("[decompose] Failed (%s), falling back to raw question", e)
        return [question], []


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
    
    Uses query decomposition to turn a user instruction into multiple focused
    search queries, getting much better recall from the code index.
    """
    import asyncio
    parts = []
    
    # 1. Attached files from IDE (live file contents)
    if attached_files:
        parts.append("## Live Files (from IDE)\n")
        for path, content in list(attached_files.items())[:5]:  # Limit to 5 files
            truncated = content[:8000] + "\n... (truncated)" if len(content) > 8000 else content
            parts.append(f"### `{path}`\n```\n{truncated}\n```\n")
        parts.append("")
    
    # 2. Decompose question into focused search queries + symbols
    #    Pass attached files so the LLM can reference real symbol names
    search_queries, symbols = await _decompose_query(question, attached_files)
    logger.info("Pre-enrichment: %d queries=%s, %d symbols=%s", 
                len(search_queries), search_queries, len(symbols), symbols)
    
    # 3. Run all search queries in parallel for speed
    async def _search(query: str) -> list[dict]:
        try:
            query_emb = await embeddings.embed_query(query)
            results = await store.vector_search(workspace_id, query_emb, top_k=5)
            return results
        except Exception as e:
            logger.error("Search failed for query '%s': %s", query, e)
            return []
    
    all_search_results = await asyncio.gather(*[_search(q) for q in search_queries])
    
    # 4. Deduplicate results across queries (by symbol name/path)
    seen = set()
    unique_results = []
    for results in all_search_results:
        for r in results:
            sym = r.get("symbol", {})
            # Deduplicate by (name, file_path) to avoid showing the same code twice
            key = (sym.get("name", ""), sym.get("file_path", ""))
            if key not in seen:
                seen.add(key)
                unique_results.append(sym)
    
    logger.info("Pre-enrichment: %d unique results from %d queries", len(unique_results), len(search_queries))
    
    if unique_results:
        # Cap at 12 results to avoid token explosion
        context_text = chat_utils.build_context_from_results(unique_results[:12])
        if context_text:
            parts.append(context_text)
            logger.info("Pre-enrichment: built context len=%d", len(context_text))
    
    # 5. Trace call chains for extracted symbols
    for sym in symbols[:2]:
        try:
            trace = await store.trace_call_chain(workspace_id, sym, direction="both", max_depth=2)
            if trace.get("nodes"):
                parts.append(f"\n## Call Chain: `{sym}`\n")
                for edge in trace.get("edges", [])[:10]:
                    parts.append(f"  `{edge['from']}` → `{edge['to']}`")
                parts.append("")
        except Exception as e:
            logger.debug("Trace failed for %s: %s", sym, e)
    
    return "\n".join(parts).strip()




# ── Plan Display ──────────────────────────────────────────────────

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


# Cosine similarity threshold: below this, the topic has shifted enough to re-enrich.
# 0.65 means ~50% topical overlap — tolerant of rephrasing, triggers on real pivots.
TOPIC_SHIFT_THRESHOLD = 0.65


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


@traceable(name="enrich_context_node", run_type="chain", tags=["enrichment"])
async def enrich_context(state: AgentState) -> dict:
    """First node: gather context BEFORE calling the LLM.
    
    Behavior:
    - First turn: always enrich.
    - Tool result continuation (mid-task): always skip.
    - Follow-up question: compare embedding similarity with the question
      we originally enriched for. If the topic shifted significantly,
      re-enrich with the new question so the agent has fresh context.
    """
    workspace_id = state['workspace_id']
    attached_files = state.get('attached_files', {})
    existing_context = state.get('enriched_context', '')
    enriched_question = state.get('enriched_question', '')
    
    # ── Tool result continuation: ALWAYS skip (agent is mid-task) ──
    if state['messages'] and isinstance(state['messages'][-1], ToolMessage):
        logger.info("[enrich_context] Skipping - ToolMessage continuation")
        return {}
    
    question = _get_last_question(state)
    if not question:
        logger.warning("[enrich_context] No question found in messages")
        return {"enriched_context": "", "enriched_question": ""}
    
    # ── First turn: always enrich ──
    if not existing_context:
        logger.info("[enrich_context] First turn — enriching for: %s", question[:100])
        context = await build_pre_enrichment(workspace_id, question, attached_files)
        logger.info("[enrich_context] Built context with %d chars", len(context))
        return {"enriched_context": context, "enriched_question": question}
    
    # ── Follow-up question: check for topic shift ──
    if enriched_question and question != enriched_question:
        try:
            old_emb, new_emb = await embeddings.embed_query(enriched_question), await embeddings.embed_query(question)
            similarity = _cosine_similarity(old_emb, new_emb)
            logger.info("[enrich_context] Topic similarity: %.3f (threshold=%.2f) | old=%s | new=%s",
                       similarity, TOPIC_SHIFT_THRESHOLD, enriched_question[:60], question[:60])
            
            if similarity < TOPIC_SHIFT_THRESHOLD:
                logger.info("[enrich_context] Topic shift detected (%.3f < %.2f) — re-enriching",
                           similarity, TOPIC_SHIFT_THRESHOLD)
                context = await build_pre_enrichment(workspace_id, question, attached_files)
                logger.info("[enrich_context] Re-enriched with %d chars", len(context))
                return {"enriched_context": context, "enriched_question": question}
            else:
                logger.info("[enrich_context] Same topic (%.3f) — keeping existing context", similarity)
        except Exception as e:
            logger.warning("[enrich_context] Topic shift check failed (%s), keeping existing context", e)
    
    return {}


# ── Context Window Management ─────────────────────────────────────

# Rough char budget: 120K tokens ≈ 480K chars.  Reserve room for
# system prompt (~4K), enriched context (~20K), and model output (~4K).
MAX_HISTORY_CHARS = 200_000  # ~50K tokens for conversation history
MAX_TOOL_MSG_CHARS = 30_000  # Truncate individual tool results

SUMMARIZE_PROMPT = """Summarize this conversation segment concisely. Focus on:
- What files were read or edited (exact paths)
- What changes were made and why
- Key decisions or findings
- Any errors encountered and how they were resolved
- Current state of the task

Be brief (under 500 words). Use bullet points. Include exact file paths and symbol names."""


async def _summarize_dropped_messages(dropped_messages: list[BaseMessage]) -> str:
    """Use the tool model to summarize conversation messages that are being dropped.
    
    This preserves the key decisions, file edits, and findings from the middle
    of a long conversation instead of just saying "[N messages omitted]".
    """
    # Build a compact representation of the dropped messages
    parts = []
    for msg in dropped_messages:
        if isinstance(msg, HumanMessage):
            parts.append(f"[User]: {msg.content[:500]}")
        elif isinstance(msg, AIMessage):
            text = msg.content[:500] if msg.content else ""
            tools = ", ".join(tc["name"] for tc in (msg.tool_calls or []))
            if tools:
                parts.append(f"[Assistant]: {text}\n  Tools called: {tools}")
            elif text:
                parts.append(f"[Assistant]: {text}")
        elif isinstance(msg, ToolMessage):
            # Keep tool results short — just the first/last lines
            content = msg.content
            if len(content) > 300:
                content = content[:150] + "\n...\n" + content[-150:]
            parts.append(f"[Tool result]: {content}")
    
    conversation_text = "\n\n".join(parts)
    
    # Cap the input to avoid blowing up the summarization call itself
    if len(conversation_text) > 30_000:
        conversation_text = conversation_text[:30_000] + "\n\n... (further messages truncated for summarization)"
    
    try:
        model = llm_provider.get_tool_model(temperature=0.0)
        response = await model.ainvoke([
            SystemMessage(content=SUMMARIZE_PROMPT),
            HumanMessage(content=conversation_text),
        ])
        summary = response.content.strip()
        logger.info("[summarize] Summarized %d messages into %d chars", len(dropped_messages), len(summary))
        return summary
    except Exception as e:
        logger.warning("[summarize] Failed (%s), using fallback", e)
        # Fallback: extract just the tool names and file paths
        tool_names = set()
        for msg in dropped_messages:
            if isinstance(msg, AIMessage) and msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_names.add(tc["name"])
                    path = tc.get("args", {}).get("path", "")
                    if path:
                        tool_names.add(f"  {tc['name']}({path})")
        if tool_names:
            return f"[Summary of {len(dropped_messages)} earlier messages — tools used: {', '.join(sorted(tool_names))}]"
        return f"[{len(dropped_messages)} earlier messages omitted to fit context window]"


async def _truncate_messages(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Truncate oversized tool messages and trim old history if needed.
    
    When dropping middle messages, summarizes them using the tool model
    so the agent remembers what it did (files edited, decisions made).
    """
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
    # and SUMMARIZE the dropped middle instead of discarding it
    if total_chars > MAX_HISTORY_CHARS:
        if len(result) > 6:
            kept_start = result[:2]   # First human msg + first AI response
            kept_end = result[-4:]     # Last 4 messages (most recent context)
            dropped = result[2:-4]     # Middle messages to summarize
            
            logger.warning("[truncate] Summarizing %d middle messages to fit context", len(dropped))
            summary_text = await _summarize_dropped_messages(dropped)
            
            summary = AIMessage(content=f"## Summary of Earlier Work\n\n{summary_text}")
            result = kept_start + [summary] + kept_end
            logger.info("[truncate] Kept %d messages (2 start + summary + 4 end)", len(result))
    
    return result


# ── Multi-Model Routing ───────────────────────────────────────────
#
# Strategy: use the best model for each phase of the agent loop.
# Models are configured via app/llm.py (backed by LiteLLM).
#
#   REASONING model (LLM_REASONING_MODEL):
#     First call with enriched context + user question.
#     Needs strong reasoning to understand the codebase and plan.
#
#   TOOL model (LLM_TOOL_MODEL):
#     Subsequent calls that process tool results and decide next action.
#     Needs fast responses + reliable function calling.
#
# Works with ANY provider: Groq, Gemini, Claude, Fireworks, OpenAI, etc.
# Just set the model string: "groq/kimi-k2", "gemini/gemini-2.0-flash", etc.


def _pick_model_name(messages: list[BaseMessage], plan_steps: list[PlanStep] = None, current_step: int = 0) -> str:
    """Pick model string based on what the agent needs to do RIGHT NOW.
    
    Reasoning model (stronger, slower) for moments that need deep thinking:
      - First call (understand the question + codebase context)
      - Right after creating a plan (first execution step needs understanding)
      - After a tool failure (needs to reason about what went wrong)
      - When there's no plan yet but the task looks complex (agent might create one)
    
    Tool model (faster, cheaper) for straightforward execution:
      - Processing normal tool results (file contents, search results)
      - Continuing plan execution on routine steps
    """
    config = llm_provider.get_config()
    
    # No tool messages at all → first call, always use reasoning
    has_tool_messages = any(isinstance(m, ToolMessage) for m in messages)
    if not has_tool_messages:
        logger.info("[model_routing] → reasoning (first call)")
        return config.reasoning_model
    
    # Look at the last few messages to understand what just happened
    recent_tool_results = []
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            recent_tool_results.append(msg)
        elif isinstance(msg, AIMessage):
            break  # Stop at the last AI message
    
    # ── Check for failure signals → reasoning model ──
    for tool_msg in recent_tool_results:
        content_lower = tool_msg.content.lower() if tool_msg.content else ""
        if any(signal in content_lower for signal in (
            "error", "failed", "traceback", "exception", 
            "not found", "permission denied", "command failed",
            "no such file", "syntax error", "compile error",
        )):
            logger.info("[model_routing] → reasoning (tool failure detected)")
            return config.reasoning_model
    
    # ── Check if agent just created a plan → reasoning for first step ──
    if plan_steps and current_step == 1:
        # Just started executing — first step after planning needs reasoning
        any_done = any(s["status"] == "done" for s in plan_steps)
        if not any_done:
            logger.info("[model_routing] → reasoning (first step of new plan)")
            return config.reasoning_model
    
    # ── Check if the last AI message had no tool calls (agent was talking) ──
    # This means we're resuming after the agent gave a text response,
    # possibly asking for clarification or explaining something complex
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            if not msg.tool_calls:
                logger.info("[model_routing] → reasoning (resuming after text response)")
                return config.reasoning_model
            break
    
    # ── Default: tool model for routine execution ──
    logger.info("[model_routing] → tool (routine execution)")
    return config.tool_model


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
    
    # ── Active plan context (created by create_plan tool) ────
    if plan_steps and current_step > 0:
        plan_display = format_plan_for_prompt(plan_steps, current_step)
        messages_to_send.append(SystemMessage(content=plan_display))
        logger.info("[call_model] Injected active plan context (step %d/%d)", 
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
    history = await _truncate_messages(state['messages'])
    messages_to_send.extend(history)
    
    total_chars = sum(len(m.content) for m in messages_to_send)
    
    # Pick model based on what the agent needs to do right now
    model_name = _pick_model_name(state['messages'], plan_steps=plan_steps, current_step=current_step)
    logger.info("[call_model] Using %s | %d messages, %d chars", model_name, len(messages_to_send), total_chars)
    
    model = llm_provider.get_chat_model(model_name, temperature=0.1)
    
    # Bind all tools
    model_with_tools = model.bind_tools(ALL_TOOLS)
    
    # Call the model
    response = await model_with_tools.ainvoke(messages_to_send)
    
    logger.info("[call_model] Got response from %s, tool_calls=%s", 
                model_name,
                [tc['name'] for tc in response.tool_calls] if response.tool_calls else "none")
    
    return {"messages": [response]}


@traceable(name="execute_server_tools", run_type="chain", tags=["server-tools"])
async def execute_server_tools(state: AgentState) -> dict:
    """Execute tools that run on the server (search, trace, impact, plan management)."""
    workspace_id = state['workspace_id']
    last_message = state['messages'][-1]
    tool_outputs = []
    
    # Track plan state mutations from plan tools
    plan_steps = list(state.get('plan_steps', []))
    current_step = state.get('current_step', 0)
    plan_changed = False
    
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
            
            # ── Plan Management Tools ─────────────────────────────
            elif tool_name == "create_plan":
                step_descriptions = args.get('steps', [])
                if not step_descriptions:
                    content = "Error: steps list cannot be empty."
                else:
                    plan_steps = [
                        PlanStep(number=i + 1, description=desc, status="pending")
                        for i, desc in enumerate(step_descriptions)
                    ]
                    # Auto-mark step 1 as in_progress
                    plan_steps[0]["status"] = "in_progress"
                    current_step = 1
                    plan_changed = True
                    content = f"Plan created with {len(plan_steps)} steps. Step 1 is now in progress."
                    logger.info("[create_plan] Created %d steps", len(plan_steps))
            
            elif tool_name == "update_plan":
                step_num = args.get('step_number', 0)
                new_status = args.get('status', '')
                new_desc = args.get('new_description', '')
                
                if not plan_steps:
                    content = "Error: No active plan. Use create_plan first."
                elif step_num < 1 or step_num > len(plan_steps):
                    content = f"Error: Invalid step number {step_num}. Plan has {len(plan_steps)} steps."
                elif new_status not in ("done", "in_progress", "failed", "pending"):
                    content = f"Error: Invalid status '{new_status}'. Use: done, in_progress, failed, pending."
                else:
                    idx = step_num - 1
                    old_status = plan_steps[idx]["status"]
                    plan_steps[idx]["status"] = new_status
                    if new_desc:
                        plan_steps[idx]["description"] = new_desc
                    
                    # Auto-advance current_step when marking done
                    if new_status == "done" and step_num == current_step:
                        next_step = step_num + 1
                        if next_step <= len(plan_steps):
                            plan_steps[next_step - 1]["status"] = "in_progress"
                            current_step = next_step
                        else:
                            current_step = 0  # All steps done
                    
                    plan_changed = True
                    content = f"Step {step_num}: {old_status} → {new_status}"
                    if current_step == 0:
                        content += " | All steps complete!"
                    elif new_status == "done":
                        content += f" | Now on step {current_step}"
                    logger.info("[update_plan] Step %d: %s → %s (current=%d)", 
                               step_num, old_status, new_status, current_step)
            
            elif tool_name == "discard_plan":
                plan_steps = []
                current_step = 0
                plan_changed = True
                content = "Plan discarded."
                logger.info("[discard_plan] Plan cleared")
                
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
    
    result = {"messages": tool_outputs}
    
    # Propagate plan state changes
    if plan_changed:
        result["plan_steps"] = plan_steps
        result["current_step"] = current_step
    
    return result


# ── Router Logic ──────────────────────────────────────────────────

def agent_router(state: AgentState) -> Literal["server_tools", "pause", "end"]:
    """After agent: route to server tools, pause for IDE, or finish.
    
    Server tools (including plan management) are always processed first.
    If only IDE tools are present, we pause for the IDE to execute them.
    """
    last_message = state['messages'][-1]
    
    if not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
        return "end"
    
    has_server_tools = False
    has_ide_tools = False
    
    for tc in last_message.tool_calls:
        if tc['name'] in IDE_TOOL_NAMES:
            has_ide_tools = True
        elif tc['name'] in SERVER_TOOL_NAMES:
            has_server_tools = True
    
    # Server tools first (search, plan management, docs)
    if has_server_tools:
        return "server_tools"
    
    # Then IDE tools (pause for the IDE to execute)
    if has_ide_tools:
        return "pause"
    
    return "end"


def post_server_tools_router(state: AgentState) -> Literal["pause", "agent"]:
    """After server tools: check if the same AI message also had IDE tools.
    
    This handles the case where the agent calls both server tools (e.g.,
    update_plan) and IDE tools (e.g., read_file) in the same turn.
    Server tools are processed first, then we pause for IDE tools.
    """
    # Find the last AIMessage (it's behind the ToolMessages from server tools)
    for msg in reversed(state['messages']):
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                if tc['name'] in IDE_TOOL_NAMES:
                    logger.info("[post_server_router] → pause (mixed server + IDE tools)")
                    return "pause"
            break
    
    return "agent"


# ── Graph Construction ────────────────────────────────────────────
#
# Simplified flow with tool-based planning:
#
#   enrich → agent ─┬→ server_tools → post_router ─┬→ agent
#                    │                               └→ pause (IDE tools)
#                    ├→ pause (IDE tools only) → END
#                    └→ end → END
#
# The agent manages its own plan via create_plan/update_plan/discard_plan
# tools. No separate classifier, planning node, or continue_plan needed.

def create_agent():
    """Build the LangGraph agent with tool-based planning.
    
    The agent decides when to plan using create_plan/update_plan/discard_plan
    tools. This replaces the old classifier → planning node flow.
    """
    workflow = StateGraph(AgentState)

    # Nodes
    workflow.add_node("enrich", enrich_context)
    workflow.add_node("agent", call_model)
    workflow.add_node("server_tools", execute_server_tools)

    # ── Edges ─────────────────────────────────────────────────
    
    # Entry: pre-enrichment (semantic search, call chains)
    workflow.set_entry_point("enrich")
    
    # After enrichment, go to agent
    workflow.add_edge("enrich", "agent")
    
    # After agent, route based on tool calls
    workflow.add_conditional_edges(
        "agent",
        agent_router,
        {
            "server_tools": "server_tools",
            "pause": END,           # Return to API/IDE for tool execution
            "end": END,
        }
    )
    
    # After server tools, check if we also need to pause for IDE tools
    workflow.add_conditional_edges(
        "server_tools",
        post_server_tools_router,
        {
            "agent": "agent",       # No IDE tools, continue reasoning
            "pause": END,           # Mixed call — pause for IDE tools
        }
    )

    return workflow.compile()


# Global agent instance
forge_agent = create_agent()
