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
     - (Future) Context7 documentation
  3. Build a "perfect prompt" with all this context
  4. LLM just needs to reason + execute with full context
  5. For IDE tools (file ops), pause and return to IDE for execution
"""

from typing import Annotated, TypedDict, Literal
import json
import logging
import os
import re

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_core.tools import tool

from . import store, embeddings, chat as chat_utils

logger = logging.getLogger(__name__)

# ── System Prompt ──────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert senior software engineer with deep expertise in code analysis, refactoring, and development. You work inside Forge IDE, a powerful coding assistant.

## Your Capabilities

You have access to:
1. **Code Context**: Relevant code snippets, call chains, and impact analysis are provided before each question
2. **Cloud Tools**: Search the codebase semantically, trace call chains
3. **IDE Tools**: Read files, write files, replace text in files, execute commands
4. **LSP Tools**: Language server powered tools for accurate type info, references, and safe renames

## Rules

1. **Reference EXACT** function names, file paths, and line numbers from the provided context
2. **Be concise and actionable** - developers want solutions, not essays  
3. **Only reference what's in the context** - don't make up code or paths
4. **When suggesting changes**:
   - First, use codebase_search or read_file to see the current code
   - Show the exact file path and what to modify
   - Use replace_in_file for precise edits (match exact existing text)
   - Use write_to_file only for new files or complete rewrites
5. **For complex tasks**, break them into steps and verify each step worked before proceeding
6. **If something fails**, analyze the error and try a different approach
7. **For refactoring**, prefer LSP tools over manual find/replace:
   - Use `lsp_find_references` to find ALL usages before changing anything
   - Use `lsp_rename` for safe, atomic renames across the workspace

## Tool Usage Guidelines

### Cloud Tools (fast, use first)
- `codebase_search`: Use for finding code by meaning. Good for "how does X work?", "find the auth logic"
- `trace_call_chain`: Use to understand execution flow. Good for "what calls X?", "what does X call?"

### LSP Tools (accurate, use for refactoring)
- `lsp_go_to_definition`: Jump to where a symbol is defined. Needs file path + line + column.
- `lsp_find_references`: Find ALL usages of a symbol. Essential before refactoring.
- `lsp_hover`: Get type signature and documentation for a symbol.
- `lsp_rename`: Safely rename a symbol across the entire workspace. Atomic and accurate.

### File Tools (for reading and editing)
- `read_file`: Use to get exact current file contents before editing
- `replace_in_file`: Use for precise edits. The old_str must match EXACTLY including whitespace
- `write_to_file`: Use only for new files or complete file rewrites
- `execute_command`: Use for git, npm, cargo, tests, etc. Never use for destructive commands without confirmation"""


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
SERVER_TOOLS = [codebase_search, trace_call_chain, impact_analysis]
IDE_TOOLS = [read_file, write_to_file, replace_in_file, execute_command]
LSP_TOOLS = [lsp_go_to_definition, lsp_find_references, lsp_hover, lsp_rename]
ALL_TOOLS = SERVER_TOOLS + IDE_TOOLS + LSP_TOOLS

IDE_TOOL_NAMES = {
    "read_file", "write_to_file", "replace_in_file", "execute_command",
    "lsp_go_to_definition", "lsp_find_references", "lsp_hover", "lsp_rename",
}
SERVER_TOOL_NAMES = {"codebase_search", "trace_call_chain", "impact_analysis"}


# ── Pre-Enrichment Logic ──────────────────────────────────────────

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
    
    # 5. (Future) Fetch Context7 documentation for libraries mentioned
    # TODO: Integrate Context7 MCP for official docs
    
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


# ── Graph Nodes ───────────────────────────────────────────────────

async def enrich_context(state: AgentState) -> dict:
    """First node: gather context BEFORE calling the LLM."""
    workspace_id = state['workspace_id']
    attached_files = state.get('attached_files', {})
    
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
