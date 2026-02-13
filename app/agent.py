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

SYSTEM_PROMPT = """You are an expert software engineer in Forge IDE. Execute tasks using tools — never just describe what you'd do.

## Rules

1. **Use tools.** Don't say "I would use X" — call X.
2. **Read before editing.** Always `read_file` before `replace_in_file`. The `old_str` must match exactly.
3. **Search smart.** Use `codebase_search` for meaning, `grep` for exact text. Never `execute_command` with grep/find.
4. **Don't loop.** If a command fails twice, use `lookup_documentation` or try a different approach. Never retry 3+ times.
5. **Rename safely.** Prefer `lsp_rename` over manual find/replace — it's atomic and cross-file.
6. **Plan adaptively.** Use `create_plan` for 3+ step tasks. Update progress with `update_plan`. If things aren't working (2+ failures), use `replan` to revise your strategy. Use `add_plan_step`/`remove_plan_step` to adjust scope as you learn more.
7. **Verify your work.** After edits: `diagnostics` on changed files → `find_symbol_references` on changed symbols → build/test command. Not done until checks pass.
8. **Dev servers and long-running processes:** NEVER use `execute_command` for servers or watchers (npm run dev, cargo watch, etc.) — it blocks until timeout. Instead:
   - `execute_background(command, label)` to start the process
   - `wait_for_port(port, timeout, http_check=True)` to wait until it's ready
   - `read_process_output(pid)` to check logs if something goes wrong
   Install-then-run commands should be split: `execute_command("npm install")` first, then `execute_background("npm run dev")`.
9. **Docker auto-start:** If a Docker command fails with "Cannot connect to the Docker daemon":
   - On macOS: `execute_command("open -a Docker")` then wait 10 seconds and retry
   - On Linux: `execute_command("sudo systemctl start docker")` then retry
   After starting Docker, wait for it to be ready before retrying the original command."""


# ── Master Planning Prompt (used when Claude is called for planning) ──
# This prompt extracts maximum value from expensive Claude calls by asking
# for deep analysis, comprehensive planning, and upfront risk identification.

MASTER_PLANNING_PROMPT = """## Master Planning Mode

You are being called with a powerful reasoning model for ONE purpose: create a comprehensive, production-quality execution plan. This is your ONE chance to think deeply before the faster execution model takes over.

### Your Planning Responsibilities

**1. ANALYZE THE REQUEST THOROUGHLY**
- What is the user ACTUALLY trying to achieve? (Not just what they said)
- What's the scope? (Single file fix vs architectural change vs new feature)
- What domain knowledge is needed? (Framework conventions, API patterns, security considerations)

**2. USE THE PRE-GATHERED CONTEXT**
IMPORTANT: You have already been given semantic search results in the "Pre-gathered Context" section above.
This contains relevant code snippets found via AI-powered codebase search.

- Study the code snippets in the pre-gathered context FIRST
- Reference specific file paths, function names, and patterns from that context
- Only call `codebase_search` if you need ADDITIONAL information not covered above
- Use `read_file` to see full file contents if snippets aren't enough
- Use `grep` to find exact symbol usages across the codebase

**3. CREATE A MASTERFUL PLAN**
Your plan should include:
- **Preparation steps** (read specific files you found via codebase_search)
- **Implementation steps** (concrete changes with EXACT file paths, function names, patterns to follow)
- **Verification steps** (tests to run, diagnostics to check, build commands)
- **Rollback awareness** (what to do if a step fails)

IMPORTANT: Reference actual code you found. Instead of "create a login page", say:
"Create pages/login.tsx following the pattern in pages/index.tsx (uses Hero component, Tailwind classes)"

Each step should be ATOMIC — completable without depending on later steps.
Order steps to minimize risk: read → small change → verify → larger changes.

**4. IDENTIFY RISKS UPFRONT**
In your response (before the plan), briefly note:
- Potential breaking changes or side effects
- Dependencies that might need updating
- Files that are touched by multiple steps (conflict risk)
- Things you're uncertain about that need exploration

**5. OPTIMIZE FOR THE EXECUTION MODEL**
The execution model is faster but less capable. Write steps that are:
- Clear and unambiguous (no "figure out how to X")
- Self-contained (include file paths, function names)
- Verifiable (each step should have a way to confirm success)

### Example of Good vs Bad Plan Steps

❌ Bad: "Create a login page" (too vague, no context)
✓ Good: "Create pages/login.tsx. Based on codebase_search results, follow the pattern from pages/index.tsx which uses: `import Layout from '../components/Layout'`, Tailwind classes like 'flex flex-col min-h-screen', and the existing form styling from the search results."

❌ Bad: "Update the authentication logic"
✓ Good: "In auth/session.py, modify `create_session()` to return JWT instead of session ID. Update return type annotation."

❌ Bad: "Fix any issues that come up"
✓ Good: "Run `pytest tests/auth/` and fix any failures related to session token format"

❌ Bad: "Refactor the codebase"
✓ Good: "Extract `validate_token()` from auth/middleware.py into auth/jwt.py, update 3 imports in auth/routes.py"

### Workflow Example
1. User asks: "Add login functionality"
2. Review the Pre-gathered Context above — it already has relevant code from semantic search
3. If you need more detail on a specific file, call `read_file("pages/index.tsx")`
4. If the context is missing something specific, call `codebase_search("session handling")`
5. Call `create_plan` with steps that reference actual code patterns you found

### Now: Think deeply, explore the codebase, then create your master plan."""


# ── Replan Prompt (used when Claude is called to recover from stuck state) ──

REPLAN_PROMPT = """## Strategic Replanning Mode

The current approach isn't working. You're being called with a powerful reasoning model to diagnose the problem and create a BETTER plan.

### Analyze What Went Wrong

1. **Review the failures** — What specifically failed? Error messages, wrong assumptions, missing dependencies?
2. **Identify root cause** — Was it a bad plan, unexpected codebase structure, or environmental issue?
3. **Learn from attempts** — What DID work? What can be preserved?

### Create a Revised Strategy

Use `replan(reason, new_steps, keep_completed=True)` to:
- Acknowledge what went wrong (the `reason` is shown to the user)
- Create new steps that avoid the same pitfalls
- Preserve completed work if it's still valid

### Common Recovery Patterns

- **Dependency issue?** → Add step to check/fix dependencies first
- **Wrong file/location?** → Add exploration step before modifying
- **Test failures?** → Add step to read test expectations, match them
- **Config mismatch?** → Add step to align with project conventions

### Now: Diagnose the issue, then use `replan` with a smarter approach."""



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
    # Which plan step we last enriched for (0 = no step-specific enrichment)
    enriched_step: int
    # Project profile: tech stack, versions, warnings (built once per workspace)
    project_profile: str
    # Attached files from IDE (live file contents)
    attached_files: dict[str, str]  # path -> content
    # Attached images from user (paste/screenshot) — list of {filename, data, mime_type}
    attached_images: list[dict]
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


@tool
async def add_plan_step(after_step: int, description: str) -> str:
    """
    Insert a new step into the current plan.
    Use when you discover the task needs additional work not in the original plan.
    
    Args:
        after_step: Insert the new step AFTER this step number (0 = insert at beginning).
        description: What the new step should accomplish.
    
    Example:
        # Discovered we need database migrations before updating the API
        add_plan_step(2, "Create database migration for new user_roles table")
    """
    return "PENDING_SERVER_EXECUTION"


@tool
async def remove_plan_step(step_number: int) -> str:
    """
    Remove a step from the current plan.
    Use when a step is no longer needed (already handled, or approach changed).
    
    Args:
        step_number: Which step to remove (1-indexed).
    """
    return "PENDING_SERVER_EXECUTION"


@tool
async def replan(reason: str, new_steps: list[str], keep_completed: bool = True) -> str:
    """
    Replace the current plan with a new one, optionally preserving completed work.
    Use when:
    - The original approach isn't working (2+ step failures)
    - You discovered the task is significantly different than expected
    - Major blockers require a different strategy
    
    Args:
        reason: Brief explanation of why re-planning is needed (shown to user).
        new_steps: The new list of steps to execute.
        keep_completed: If True, completed steps from the old plan are preserved
                       as a "completed" section for context. Default True.
    
    Example:
        replan(
            reason="Jest tests require different config than expected",
            new_steps=[
                "Check existing jest.config.js for test environment",
                "Update jest config to use jsdom environment",
                "Re-run failing tests with new config"
            ]
        )
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
PLAN_TOOLS = [create_plan, update_plan, discard_plan, add_plan_step, remove_plan_step, replan]
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
PLAN_TOOL_NAMES = {"create_plan", "update_plan", "discard_plan", "add_plan_step", "remove_plan_step", "replan"}
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


# ── Project Profile ───────────────────────────────────────────────
#
# Reads key config files and uses the tool model to understand the
# project's tech stack, versions, and detect compatibility issues
# BEFORE the agent writes any code.

# Config files to look for (checked in order, missing files are skipped)
PROJECT_CONFIG_FILES = [
    # Package managers / dependencies
    "package.json", "Cargo.toml", "requirements.txt", "pyproject.toml",
    "go.mod", "Gemfile", "pom.xml", "build.gradle",
    # Framework configs
    "next.config.js", "next.config.mjs", "next.config.ts",
    "vite.config.ts", "vite.config.js",
    "nuxt.config.ts", "angular.json", "svelte.config.js",
    # TypeScript / JavaScript
    "tsconfig.json", "jsconfig.json",
    # Styling
    "tailwind.config.js", "tailwind.config.ts", "postcss.config.js", "postcss.config.mjs",
    # UI libraries
    "components.json",  # Shadcn
    # Linting / formatting
    ".eslintrc.js", ".eslintrc.json", "eslint.config.js",
    ".prettierrc", ".prettierrc.json",
    # Environment / runtime
    ".nvmrc", ".node-version", ".python-version", ".tool-versions",
    "Dockerfile", "docker-compose.yml",
    # Git
    ".gitignore",
]

# Global CSS files to check for styling config
PROJECT_CSS_FILES = [
    "styles/globals.css", "src/styles/globals.css",
    "app/globals.css", "src/app/globals.css",
    "src/index.css", "styles/global.css",
]

PROFILE_ANALYSIS_PROMPT = """Analyze this project's configuration files and produce a concise project profile.

Extract:
1. **Framework** + exact version (e.g., "Next.js 13.4.0 (Pages Router)" or "Vite 5.2 + React 18")
2. **Language** + version (e.g., "TypeScript 5.3, strict mode")
3. **Styling** + version (e.g., "Tailwind CSS 4.1.18 with v4 CSS syntax")
4. **UI Library** (e.g., "Shadcn UI" if components.json present)
5. **Package manager** (npm/yarn/pnpm/bun — infer from lock file or config)
6. **Test framework** (jest, vitest, pytest, etc. or "none detected")
7. **Key conventions** (src/ vs app/ structure, module system ESM/CJS)

Then check for **WARNINGS** — version conflicts, configuration mismatches, or common pitfalls:
- Framework version vs dependency version compatibility
- Config file syntax vs installed version mismatches (e.g., Tailwind v4 CSS syntax but v3 config file format)
- Missing peer dependencies
- Outdated patterns that will cause issues

Format as a concise profile. Keep it under 400 words. Start warnings with "WARNING:" so they stand out.

Respond with ONLY the profile text, no JSON wrapping."""


@traceable(name="build_project_profile", run_type="chain", tags=["profile"])
async def build_project_profile(attached_files: dict[str, str] | None = None) -> str:
    """Build a project profile from config files.
    
    Reads key config files from attached_files (sent by IDE) and uses the
    tool model to analyze tech stack, versions, and detect conflicts.
    
    Returns a profile string to inject into the agent's context.
    """
    if not attached_files:
        return ""
    
    # Step 1: Collect config file contents from attached files
    config_contents = {}
    all_config_names = set(PROJECT_CONFIG_FILES + PROJECT_CSS_FILES)
    
    for path, content in attached_files.items():
        # Match by filename (attached files have full paths)
        filename = path.rsplit("/", 1)[-1] if "/" in path else path
        # Also check path suffixes for nested configs like styles/globals.css
        matches = (
            filename in all_config_names
            or any(path.endswith(f) for f in all_config_names)
        )
        if matches and content.strip():
            # Truncate very large configs (package-lock.json etc.)
            truncated = content[:3000] if len(content) > 3000 else content
            config_contents[path] = truncated
    
    if not config_contents:
        logger.info("[project_profile] No config files found in attached files")
        return ""
    
    logger.info("[project_profile] Found %d config files: %s", 
               len(config_contents), list(config_contents.keys()))
    
    # Step 2: Analyze with tool model
    files_text = "\n\n".join(
        f"### {path}\n```\n{content}\n```"
        for path, content in config_contents.items()
    )
    
    try:
        model = llm_provider.get_tool_model(temperature=0.0)
        response = await model.ainvoke([
            SystemMessage(content=PROFILE_ANALYSIS_PROMPT),
            HumanMessage(content=files_text),
        ])
        
        profile = response.content.strip()
        logger.info("[project_profile] Built profile (%d chars): %s", len(profile), profile[:200])
        return profile
        
    except Exception as e:
        logger.warning("[project_profile] Analysis failed (%s), building basic profile", e)
        # Fallback: just list what we found
        files_list = ", ".join(config_contents.keys())
        return f"Project config files detected: {files_list}"


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


@traceable(name="step_enrichment", run_type="chain", tags=["enrichment"])
async def _step_enrichment(workspace_id: str, step_description: str) -> str:
    """Lightweight supplementary search based on a plan step description.
    
    Unlike full pre-enrichment (query decomposition + multi-query + call chains),
    this is a single focused search — just enough to give the agent context
    for the current step without the overhead.
    """
    try:
        query_emb = await embeddings.embed_query(step_description)
        results = await store.vector_search(workspace_id, query_emb, top_k=5)
        
        if results:
            flat_results = [r["symbol"] for r in results]
            context_text = chat_utils.build_context_from_results(flat_results)
            if context_text:
                logger.info("[step_enrichment] Found %d results for step: %s", len(flat_results), step_description[:60])
                return f"\n\n## Context for Current Step\n\n{context_text}"
        
        return ""
    except Exception as e:
        logger.warning("[step_enrichment] Failed for step '%s': %s", step_description[:60], e)
        return ""


@traceable(name="enrich_context_node", run_type="chain", tags=["enrichment"])
async def enrich_context(state: AgentState) -> dict:
    """First node: gather context BEFORE calling the LLM.
    
    Behavior:
    - First turn: full enrichment (query decomposition + multi-query search).
    - Tool result continuation: check if plan step changed — if so, do a
      lightweight supplementary search for the new step's context.
    - Follow-up question: check embedding similarity — re-enrich if topic shifted.
    """
    workspace_id = state['workspace_id']
    attached_files = state.get('attached_files', {})
    existing_context = state.get('enriched_context', '')
    enriched_question = state.get('enriched_question', '')
    enriched_step = state.get('enriched_step', 0)
    plan_steps = state.get('plan_steps', [])
    current_step = state.get('current_step', 0)
    
    # ── Tool result continuation: check for plan step change ──
    if state['messages'] and isinstance(state['messages'][-1], ToolMessage):
        # If plan step advanced since last enrichment, supplement the context
        if (plan_steps and current_step > 0 
                and current_step != enriched_step 
                and current_step <= len(plan_steps)):
            step_desc = plan_steps[current_step - 1]["description"]
            logger.info("[enrich_context] Plan step changed (%d → %d): %s — supplementing context",
                       enriched_step, current_step, step_desc[:80])
            
            supplement = await _step_enrichment(workspace_id, step_desc)
            if supplement:
                # Append to existing context (don't replace — original context is still useful)
                updated_context = existing_context + supplement
                # Cap total context to avoid bloat across many steps
                if len(updated_context) > 100_000:
                    updated_context = updated_context[:100_000]
                return {"enriched_context": updated_context, "enriched_step": current_step}
            else:
                # No new results, just mark the step as enriched
                return {"enriched_step": current_step}
        
        logger.info("[enrich_context] Skipping - ToolMessage continuation (same step)")
        return {}
    
    question = _get_last_question(state)
    if not question:
        logger.warning("[enrich_context] No question found in messages")
        return {"enriched_context": "", "enriched_question": "", "enriched_step": 0}
    
    # ── First turn: full enrichment + project profile ──
    if not existing_context:
        import asyncio
        logger.info("[enrich_context] First turn — enriching for: %s", question[:100])
        
        # Build project profile and code enrichment in parallel
        existing_profile = state.get('project_profile', '')
        if not existing_profile:
            # Profile + enrichment concurrently
            profile_coro = build_project_profile(attached_files)
            enrich_coro = build_pre_enrichment(workspace_id, question, attached_files)
            profile, context = await asyncio.gather(profile_coro, enrich_coro)
            logger.info("[enrich_context] Built context=%d chars, profile=%d chars", len(context), len(profile))
            return {
                "enriched_context": context,
                "enriched_question": question,
                "enriched_step": current_step,
                "project_profile": profile,
            }
        else:
            # Profile already exists (from Redis cache), just enrich
            context = await build_pre_enrichment(workspace_id, question, attached_files)
            logger.info("[enrich_context] Built context with %d chars (profile cached)", len(context))
            return {"enriched_context": context, "enriched_question": question, "enriched_step": current_step}
    
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
                return {"enriched_context": context, "enriched_question": question, "enriched_step": 0}
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


# ── Self-Reflection (Stuck Detection) ─────────────────────────────

REFLECT_PROMPT = """## STOP AND REFLECT

You appear to be stuck. Before continuing, step back and think:

1. **What have you tried so far?** Look at the recent tool results above.
2. **Why isn't it working?** Identify the root cause, not just the symptoms.
3. **What should you do differently?** Consider:
   - Is the approach fundamentally wrong? (e.g., wrong file, wrong API, wrong assumption)
   - Are you missing a dependency or setup step?
   - Should you use `lookup_documentation` to check the correct API?
   - Should you `discard_plan` and try a completely different approach?
   - Should you ask the user for clarification?

**Do NOT retry the same failing approach.** Change strategy."""


def _detect_stuck_signals(messages: list[BaseMessage], plan_steps: list[PlanStep], current_step: int) -> str | None:
    """Scan recent conversation history for signs the agent is stuck.
    
    Returns a reflection prompt string if stuck, None otherwise.
    
    Detects:
    - Repeated errors (2+ consecutive tool failures)
    - Same tool called with same args (loop)
    - Plan step stuck for too many turns (3+ AI messages on same step)
    """
    if len(messages) < 4:
        return None  # Too early to judge
    
    # ── Collect recent AI messages and tool results ──
    recent_ai_calls: list[dict] = []  # [{name, args_key}]
    consecutive_errors = 0
    turns_on_current_step = 0
    
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            content_lower = (msg.content or "").lower()
            has_error = any(sig in content_lower for sig in (
                "error", "failed", "traceback", "exception",
                "command failed", "no such file", "permission denied",
            ))
            if has_error:
                consecutive_errors += 1
            else:
                consecutive_errors = 0  # Reset on success
                
        elif isinstance(msg, AIMessage):
            turns_on_current_step += 1
            
            # Track tool calls for loop detection
            for tc in (msg.tool_calls or []):
                # Create a hashable key from tool name + sorted args
                args_key = tc["name"] + ":" + json.dumps(tc.get("args", {}), sort_keys=True)[:200]
                recent_ai_calls.append({"name": tc["name"], "args_key": args_key})
            
            # Only look back ~8 turns
            if turns_on_current_step > 8:
                break
        
        elif isinstance(msg, HumanMessage):
            break  # Stop at the last user message
    
    # ── Check: 2+ consecutive errors ──
    if consecutive_errors >= 2:
        logger.warning("[reflect] STUCK: %d consecutive tool errors", consecutive_errors)
        
        # Extract failing command from recent messages for explicit "don't do this" guidance
        failing_commands = []
        for msg in reversed(messages[:20]):
            if isinstance(msg, AIMessage) and msg.tool_calls:
                for tc in msg.tool_calls:
                    if tc['name'] == 'execute_command':
                        failing_commands.append(tc['args'].get('command', '')[:50])
                    else:
                        failing_commands.append(tc['name'])
                    if len(failing_commands) >= 2:
                        break
            if len(failing_commands) >= 2:
                break
        
        dont_retry = ""
        if failing_commands:
            dont_retry = f"\n\n**DO NOT** retry these failing commands: {failing_commands}. Try a COMPLETELY different approach."
        
        if plan_steps and current_step > 0:
            return REFLECT_PROMPT + f"\n\n**CRITICAL: {consecutive_errors} consecutive tool errors.** STOP retrying the same approach.{dont_retry}\n\nOptions:\n1. `replan` - Create a new strategy\n2. `update_plan` - Mark this step failed and skip to next\n3. Ask the user for help or clarification"
        return REFLECT_PROMPT + f"\n\n**CRITICAL: {consecutive_errors} consecutive tool errors.** STOP retrying the same approach.{dont_retry}\n\nTry:\n1. Read error messages carefully and fix the root cause\n2. `lookup_documentation` for the failing tool/library\n3. Ask the user for help if you're stuck"
    
    # ── Check: same tool+args called twice (loop) ──
    if len(recent_ai_calls) >= 2:
        seen_calls = set()
        for call in recent_ai_calls:
            if call["args_key"] in seen_calls:
                logger.warning("[reflect] STUCK: duplicate tool call detected: %s", call["name"])
                if plan_steps and current_step > 0:
                    return REFLECT_PROMPT + f"\n\n**LOOP DETECTED: You called `{call['name']}` with identical arguments twice.**\n\n**DO NOT** call `{call['name']}` again with the same arguments. This will fail again.\n\nYou MUST:\n1. Use `replan` to create a new strategy, OR\n2. Use `update_plan` to mark this step failed and move on, OR\n3. Try a COMPLETELY different tool or approach"
                return REFLECT_PROMPT + f"\n\n**LOOP DETECTED: You called `{call['name']}` with identical arguments twice.**\n\n**DO NOT** repeat this call. You MUST try a different approach:\n1. Read the error and understand WHY it failed\n2. Use a different tool or different arguments\n3. Ask the user if you need more information"
            seen_calls.add(call["args_key"])
    
    # ── Check: plan step stuck for 4+ AI turns ──
    if plan_steps and current_step > 0 and turns_on_current_step >= 4:
        step_desc = plan_steps[current_step - 1]["description"] if current_step <= len(plan_steps) else "?"
        logger.warning("[reflect] STUCK: step %d ('%s') has been in progress for %d turns", 
                      current_step, step_desc, turns_on_current_step)
        return REFLECT_PROMPT + f"\n\n**Detected: step {current_step} ('{step_desc}') has been in progress for {turns_on_current_step} turns.** Options: use `add_plan_step` to break this into smaller steps, use `update_plan` to mark it failed and move on, or use `replan` if the overall approach needs to change."
    
    # ── Check: multiple failed steps in plan (suggests overall approach is wrong) ──
    if plan_steps:
        failed_count = sum(1 for s in plan_steps if s["status"] == "failed")
        if failed_count >= 2:
            logger.warning("[reflect] STUCK: %d plan steps have failed", failed_count)
            return REFLECT_PROMPT + f"\n\n**Detected: {failed_count} plan steps have failed.** The overall approach may be flawed. Use `replan` to create a new strategy based on what you've learned."
    
    return None


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


def _pick_model_name(messages: list[BaseMessage], plan_steps: list[PlanStep] = None, current_step: int = 0, is_reflecting: bool = False) -> str:
    """Pick model string based on what the agent needs to do RIGHT NOW.
    
    Planning model (Claude, best quality):
      - First call when no plan exists (will likely create plan)
      - Reflection mode suggesting replan
    
    Reasoning model (stronger, slower) for moments that need deep thinking:
      - Right after creating a plan (first execution step needs understanding)
      - After a tool failure (needs to reason about what went wrong)
    
    Tool model (faster, cheaper) for straightforward execution:
      - Processing normal tool results (file contents, search results)
      - Continuing plan execution on routine steps
    """
    config = llm_provider.get_config()
    planning_model = llm_provider.get_planning_model_name()
    
    # No tool messages at all → first call
    has_tool_messages = any(isinstance(m, ToolMessage) for m in messages)
    if not has_tool_messages:
        # First call with no existing plan → use planning model (likely to create plan)
        if not plan_steps and planning_model:
            logger.info("[model_routing] → planning model (first call, no plan yet)")
            return planning_model
        logger.info("[model_routing] → reasoning (first call)")
        return config.reasoning_model
    
    # Reflection mode: check if it's suggesting replan
    if is_reflecting:
        # If planning model is available AND reflection mentions replan, use it
        if planning_model:
            logger.info("[model_routing] → planning model (reflection/replan scenario)")
            return planning_model
        logger.info("[model_routing] → reasoning (reflection mode)")
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
    plan_steps = list(state.get('plan_steps', []))  # Make mutable copy
    current_step = state.get('current_step', 0)
    
    logger.info("[call_model] enriched_context=%d chars, plan_steps=%d, current_step=%d",
                len(enriched_context), len(plan_steps), current_step)
    
    # ── AUTO-REPLAN: Force replan after 8+ turns stuck on same step ────────────
    if plan_steps and current_step > 0:
        turns_on_step = 0
        plan_tools = {'create_plan', 'update_plan', 'replan', 'add_plan_step', 'remove_plan_step'}
        
        for msg in reversed(state['messages']):
            # First check if this is a plan-modifying tool - if so, stop counting
            # (we only want to count turns AFTER the current step started)
            if isinstance(msg, AIMessage) and msg.tool_calls:
                if any(tc['name'] in plan_tools for tc in msg.tool_calls):
                    # Found where the current step started, stop here
                    break
                # Regular tool call - count it
                turns_on_step += 1
            elif isinstance(msg, AIMessage) and not msg.tool_calls:
                # Text response - count it
                turns_on_step += 1
            
            if turns_on_step >= 20:  # Don't scan too far back
                break
        
        if turns_on_step >= 8 and current_step <= len(plan_steps):
            step_desc = plan_steps[current_step - 1].get("description", "?")
            logger.warning("[call_model] STUCK: step %d stuck for %d turns, forcing replan", 
                          current_step, turns_on_step)
            
            # Collect recent error messages to learn from
            recent_errors = []
            failed_tools = []
            for msg in reversed(state['messages'][:30]):
                if isinstance(msg, ToolMessage) and ("error" in msg.content.lower() or "failed" in msg.content.lower() or "exit code: 1" in msg.content.lower()):
                    error_snippet = msg.content[:150].replace('\n', ' ')
                    if error_snippet not in recent_errors:
                        recent_errors.append(error_snippet)
                if isinstance(msg, AIMessage) and msg.tool_calls:
                    for tc in msg.tool_calls:
                        if tc['name'] not in failed_tools:
                            failed_tools.append(tc['name'])
                if len(recent_errors) >= 3:
                    break
            
            # Build learnings from failures
            learnings = f"Step '{step_desc}' failed after {turns_on_step} attempts."
            if recent_errors:
                learnings += f" Errors encountered: {'; '.join(recent_errors[:2])}"
            if failed_tools:
                learnings += f" Tools tried: {', '.join(failed_tools[:5])}"
            
            # Mark current step as failed
            plan_steps[current_step - 1]["status"] = "failed"
            
            # Create new plan steps that avoid the same approach
            # We'll create a replan tool call that the model would have made
            replan_reason = f"Step {current_step} ('{step_desc}') failed repeatedly. {learnings}. Need a different approach."
            
            # Generate a synthetic replan tool call
            replan_tool_call = {
                "name": "replan",
                "args": {
                    "reason": replan_reason,
                    "new_steps": [
                        f"[REVISED] Find alternative approach for: {step_desc}",
                        *[s["description"] for s in plan_steps[current_step:] if s["status"] == "pending"]
                    ],
                    "keep_completed": True
                },
                "id": f"auto_replan_{current_step}"
            }
            
            replan_msg = AIMessage(
                content=f"🔄 **Auto-replanning:** Step {current_step} ('{step_desc}') failed after {turns_on_step} attempts. Creating a revised strategy...",
                tool_calls=[replan_tool_call]
            )
            
            return {
                "messages": [replan_msg],
                "plan_steps": plan_steps,
                "current_step": current_step,
            }
    
    # Build messages with system prompt and context
    messages_to_send = []
    
    # System prompt
    messages_to_send.append(SystemMessage(content=SYSTEM_PROMPT))
    
    # ── Project profile (tech stack, versions, warnings) ────
    project_profile = state.get('project_profile', '')
    is_first_call = not any(isinstance(m, ToolMessage) for m in state['messages'])
    
    # ── Determine model early so we know if we should inject planning prompt ──
    planning_model = llm_provider.get_planning_model_name()
    using_planning_model = is_first_call and not plan_steps and planning_model
    
    # ── Master Planning Prompt (when Claude is doing the initial analysis) ──
    if using_planning_model:
        messages_to_send.append(SystemMessage(content=MASTER_PLANNING_PROMPT))
        logger.info("[call_model] Injected MASTER_PLANNING_PROMPT (using planning model)")
        
        # Also inject full attached file contents for planning context
        attached_files = state.get('attached_files', {})
        if attached_files:
            files_context = "## User's Open Files (Full Content)\n\nThe user has these files open in their IDE. Use them to understand context, patterns, and code style:\n\n"
            for path, content in list(attached_files.items())[:8]:  # Up to 8 files for planning
                # Truncate very large files but be generous for planning
                if len(content) > 15000:
                    content = content[:15000] + f"\n\n... (truncated, {len(content)} chars total)"
                files_context += f"### {path}\n```\n{content}\n```\n\n"
            messages_to_send.append(SystemMessage(content=files_context))
            logger.info("[call_model] Injected %d attached files for planning", len(attached_files))
    
    if project_profile:
        messages_to_send.append(SystemMessage(
            content=f"## Project Profile\n\n{project_profile}\n\nIMPORTANT: If there are WARNINGs above, address them BEFORE starting the user's task. Fix version conflicts, configuration mismatches, or compatibility issues first."
        ))
        logger.info("[call_model] Injected project profile (%d chars)", len(project_profile))
    elif is_first_call:
        # No profile yet — tell the agent to read config files first
        messages_to_send.append(SystemMessage(
            content="## Project Profile: Not Yet Available\n\nBefore making any changes, read the project's key config files to understand the tech stack:\n- `read_file(\"package.json\")` or `read_file(\"Cargo.toml\")` or `read_file(\"requirements.txt\")`\n- Check framework config, styling config (tailwind.config.*, globals.css)\n- Check for version conflicts between dependencies\n\nThis prevents writing code that's incompatible with the project's environment."
        ))
        logger.info("[call_model] No profile — injected config-read guidance")
    
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
        context_msg = f"""## Pre-gathered Context (SEMANTIC SEARCH RESULTS)

The following code was found via AI-powered semantic search on the user's codebase.
USE THIS CODE as the primary reference for understanding patterns, conventions, and structure.

{ctx}

---

IMPORTANT: The semantic search above already queried the codebase for relevant code.
- Study the code snippets above to understand existing patterns before creating your plan
- Reference specific file paths, function names, and patterns from this context
- You can call `codebase_search` for ADDITIONAL context if needed, but start with what's above"""
        messages_to_send.append(SystemMessage(content=context_msg))
        logger.info("[call_model] Added context message, total messages: %d", len(messages_to_send))
    else:
        logger.warning("[call_model] No enriched context to add!")
    
    # ── Self-reflection: detect if the agent is stuck ────────
    reflection = _detect_stuck_signals(state['messages'], plan_steps, current_step)
    if reflection:
        messages_to_send.append(SystemMessage(content=reflection))
        logger.info("[call_model] Injected reflection prompt (agent appears stuck)")
        
        # If we have a planning model and there's an active plan, inject replan guidance
        if planning_model and plan_steps:
            messages_to_send.append(SystemMessage(content=REPLAN_PROMPT))
            logger.info("[call_model] Injected REPLAN_PROMPT (Claude for strategic recovery)")
    
    # Add conversation history (with truncation)
    history = await _truncate_messages(state['messages'])
    
    # ── Inject attached images into the last HumanMessage ────
    # Vision models expect images as part of the message content array.
    attached_images = state.get('attached_images', [])
    if attached_images and history:
        # Find the last HumanMessage in history and make it multimodal
        for i in range(len(history) - 1, -1, -1):
            if isinstance(history[i], HumanMessage):
                text_content = history[i].content
                # Build multimodal content: text + images
                content_parts = [{"type": "text", "text": text_content}]
                for img in attached_images:
                    data_uri = f"data:{img.get('mime_type', 'image/png')};base64,{img['data']}"
                    content_parts.append({
                        "type": "image_url",
                        "image_url": {"url": data_uri},
                    })
                history[i] = HumanMessage(content=content_parts)
                logger.info("[call_model] Attached %d images to HumanMessage", len(attached_images))
                break
    
    messages_to_send.extend(history)
    
    total_chars = sum(len(m.content) if isinstance(m.content, str) else len(str(m.content)) for m in messages_to_send)
    
    # Pick model based on what the agent needs to do right now
    # Reflection forces reasoning model — the agent needs to think its way out
    model_name = _pick_model_name(state['messages'], plan_steps=plan_steps, current_step=current_step, is_reflecting=bool(reflection))
    logger.info("[call_model] Using %s | %d messages, %d chars", model_name, len(messages_to_send), total_chars)
    
    model = llm_provider.get_chat_model(model_name, temperature=0.1)
    
    # Bind all tools
    model_with_tools = model.bind_tools(ALL_TOOLS)
    
    # Call the model (with fallback if planning model fails)
    try:
        response = await model_with_tools.ainvoke(messages_to_send)
    except Exception as e:
        # If planning model failed (e.g., no credits), fallback to reasoning model
        config = llm_provider.get_config()
        if model_name != config.reasoning_model:
            logger.warning("[call_model] %s failed (%s), falling back to %s", 
                          model_name, str(e)[:100], config.reasoning_model)
            fallback_model = llm_provider.get_chat_model(config.reasoning_model, temperature=0.1)
            model_with_tools = fallback_model.bind_tools(ALL_TOOLS)
            response = await model_with_tools.ainvoke(messages_to_send)
            model_name = config.reasoning_model  # Update for logging
        else:
            raise  # Re-raise if reasoning model itself failed
    
    logger.info("[call_model] Got response from %s, tool_calls=%s", 
                model_name,
                [tc['name'] for tc in response.tool_calls] if response.tool_calls else "none")
    
    # ── Detect empty response (no content AND no tool calls) ──
    has_content = response.content and response.content.strip()
    has_tools = response.tool_calls and len(response.tool_calls) > 0
    
    if not has_content and not has_tools:
        logger.warning("[call_model] EMPTY RESPONSE from %s - no content or tool calls!", model_name)
        # Return a fallback message so the conversation can continue
        fallback_msg = AIMessage(
            content="I encountered an issue processing this request. Let me try a different approach. Could you clarify what you'd like me to do, or I can check the current state of the files."
        )
        return {"messages": [fallback_msg]}
    
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
            
            elif tool_name == "add_plan_step":
                after_step = args.get('after_step', 0)
                description = args.get('description', '')
                
                if not plan_steps:
                    content = "Error: No active plan. Use create_plan first."
                elif after_step < 0 or after_step > len(plan_steps):
                    content = f"Error: Invalid position. after_step must be 0-{len(plan_steps)}."
                elif not description:
                    content = "Error: description cannot be empty."
                else:
                    # Insert new step after the specified position
                    new_step = PlanStep(
                        number=after_step + 1,  # Will be renumbered
                        description=description,
                        status="pending"
                    )
                    plan_steps.insert(after_step, new_step)
                    
                    # Renumber all steps
                    for i, step in enumerate(plan_steps):
                        step["number"] = i + 1
                    
                    # Adjust current_step if we inserted before it
                    if after_step < current_step:
                        current_step += 1
                    
                    plan_changed = True
                    content = f"Inserted new step {after_step + 1}: '{description[:50]}...' Plan now has {len(plan_steps)} steps."
                    logger.info("[add_plan_step] Inserted step after %d, total=%d", after_step, len(plan_steps))
            
            elif tool_name == "remove_plan_step":
                step_num = args.get('step_number', 0)
                
                if not plan_steps:
                    content = "Error: No active plan."
                elif step_num < 1 or step_num > len(plan_steps):
                    content = f"Error: Invalid step number {step_num}. Plan has {len(plan_steps)} steps."
                elif len(plan_steps) == 1:
                    content = "Error: Cannot remove the last step. Use discard_plan instead."
                else:
                    removed = plan_steps.pop(step_num - 1)
                    
                    # Renumber remaining steps
                    for i, step in enumerate(plan_steps):
                        step["number"] = i + 1
                    
                    # Adjust current_step
                    if step_num < current_step:
                        current_step -= 1
                    elif step_num == current_step and current_step > len(plan_steps):
                        current_step = len(plan_steps) if plan_steps else 0
                        if plan_steps and current_step > 0:
                            plan_steps[current_step - 1]["status"] = "in_progress"
                    
                    plan_changed = True
                    content = f"Removed step {step_num}: '{removed['description'][:40]}...' Plan now has {len(plan_steps)} steps."
                    logger.info("[remove_plan_step] Removed step %d, total=%d", step_num, len(plan_steps))
            
            elif tool_name == "replan":
                reason = args.get('reason', 'Re-planning required')
                new_step_descs = args.get('new_steps', [])
                keep_completed = args.get('keep_completed', True)
                
                if not new_step_descs:
                    content = "Error: new_steps list cannot be empty."
                else:
                    # Capture completed steps for context
                    completed_summary = ""
                    if keep_completed and plan_steps:
                        completed = [s for s in plan_steps if s["status"] == "done"]
                        if completed:
                            completed_summary = "Completed work preserved: " + "; ".join(
                                f"✓ {s['description'][:40]}" for s in completed
                            )
                    
                    # Create new plan
                    plan_steps = [
                        PlanStep(number=i + 1, description=desc, status="pending")
                        for i, desc in enumerate(new_step_descs)
                    ]
                    plan_steps[0]["status"] = "in_progress"
                    current_step = 1
                    plan_changed = True
                    
                    content = f"Re-planned: {reason}\nNew plan has {len(plan_steps)} steps. Step 1 is now in progress."
                    if completed_summary:
                        content += f"\n{completed_summary}"
                    logger.info("[replan] Reason='%s', new_steps=%d", reason, len(plan_steps))
                
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
