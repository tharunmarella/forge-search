"""
Parallel Execution - Execute independent tasks concurrently.

Instead of sequential execution:
  Step 1 → wait → Step 2 → wait → Step 3

Do parallel where possible:
  Step 1 ┐
  Step 2 ├→ all execute concurrently → next steps
  Step 3 ┘

This dramatically speeds up execution for independent tasks.
"""

import asyncio
import logging
from typing import Any
from datetime import datetime

logger = logging.getLogger(__name__)


async def can_run_in_parallel(
    tasks: list[dict],
    llm_model: Any
) -> dict[str, bool]:
    """
    Use LLM to analyze which tasks can run in parallel.
    
    Args:
        tasks: List of task descriptions
        llm_model: Fast model for analysis
        
    Returns:
        Dict mapping task description -> can_run_parallel
    """
    if len(tasks) <= 1:
        return {tasks[0]["description"]: False} if tasks else {}
    
    tasks_text = "\n".join(
        f"{i+1}. {task['description']}"
        for i, task in enumerate(tasks)
    )
    
    prompt = f"""Analyze which of these tasks can run in parallel (independently):

TASKS:
{tasks_text}

Two tasks can run in parallel if:
- They don't depend on each other's output
- They don't modify the same files
- They don't conflict with each other

Respond with JSON:
{{
  "parallel_groups": [
    ["task 1 description", "task 2 description"],  // These can run together
    ["task 3 description"]  // This must run alone
  ],
  "reasoning": "brief explanation"
}}

Example:
- "Read file A" and "Read file B" → Can be parallel ✓
- "Create file" and "Read that file" → Cannot be parallel (dependency) ✗
- "Install package X" and "Install package Y" → Can be parallel ✓
"""
    
    try:
        import json
        response = await llm_model.ainvoke([{"role": "user", "content": prompt}])
        content = response.content.strip()
        
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        result = json.loads(content)
        
        # Build result dict
        parallel_map = {}
        for group in result["parallel_groups"]:
            can_parallel = len(group) > 1
            for task_desc in group:
                parallel_map[task_desc] = can_parallel
        
        logger.info(
            "[parallel] Analyzed %d tasks, %d groups found",
            len(tasks),
            len(result["parallel_groups"])
        )
        
        return parallel_map
        
    except Exception as e:
        logger.error("[parallel] Analysis failed: %s", e)
        
        # Fallback: assume all tasks must run sequentially
        return {task["description"]: False for task in tasks}


async def execute_tools_in_parallel(
    tool_calls: list[dict],
    executor_func: Any,
    max_concurrent: int = 3
) -> list[dict]:
    """
    Execute multiple tool calls in parallel.
    
    Args:
        tool_calls: List of tool call dicts
        executor_func: Async function that executes a single tool call
        max_concurrent: Maximum concurrent executions
        
    Returns:
        List of results in same order as tool_calls
    """
    if len(tool_calls) <= 1:
        # Single call, just execute
        if tool_calls:
            result = await executor_func(tool_calls[0])
            return [result]
        return []
    
    logger.info("[parallel] Executing %d tool calls concurrently (max=%d)", 
                len(tool_calls), max_concurrent)
    
    # Use semaphore to limit concurrency
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def execute_with_semaphore(tool_call: dict, index: int) -> tuple[int, dict]:
        """Execute with semaphore and track index."""
        async with semaphore:
            start_time = datetime.now()
            try:
                result = await executor_func(tool_call)
                elapsed = (datetime.now() - start_time).total_seconds()
                logger.info(
                    "[parallel] Tool %s completed in %.2fs",
                    tool_call.get("name", "?"),
                    elapsed
                )
                return (index, result)
            except Exception as e:
                logger.error("[parallel] Tool %s failed: %s", tool_call.get("name"), e)
                return (index, {"error": str(e), "tool_call": tool_call})
    
    # Execute all in parallel
    tasks = [
        execute_with_semaphore(tc, i)
        for i, tc in enumerate(tool_calls)
    ]
    
    start = datetime.now()
    results_with_indices = await asyncio.gather(*tasks)
    elapsed = (datetime.now() - start).total_seconds()
    
    logger.info("[parallel] All %d tools completed in %.2fs", len(tool_calls), elapsed)
    
    # Sort results by original index
    results_with_indices.sort(key=lambda x: x[0])
    results = [r for _, r in results_with_indices]
    
    return results


def group_tool_calls_by_independence(
    tool_calls: list[dict],
    dependency_analysis: dict = None
) -> list[list[dict]]:
    """
    Group tool calls into batches that can run in parallel.
    
    Args:
        tool_calls: List of tool calls
        dependency_analysis: Optional pre-computed dependency info
        
    Returns:
        List of groups, where each group can run in parallel
    """
    if len(tool_calls) <= 1:
        return [tool_calls] if tool_calls else []
    
    # Simple heuristic-based grouping (can be enhanced with LLM analysis)
    groups = []
    current_group = []
    
    for tc in tool_calls:
        tool_name = tc.get("name", "")
        
        # Tools that typically can't run in parallel with others
        exclusive_tools = {
            "execute_command", "execute_background",  # May have side effects
            "replace_in_file", "write_to_file",  # File modifications
        }
        
        # Safe parallel tools
        parallel_safe_tools = {
            "read_file", "list_files", "codebase_search",
            "grep", "find_symbol_references", "diagnostics"
        }
        
        if tool_name in exclusive_tools:
            # Finish current group and start new one
            if current_group:
                groups.append(current_group)
                current_group = []
            groups.append([tc])
        elif tool_name in parallel_safe_tools:
            # Can add to current group
            current_group.append(tc)
        else:
            # Unknown tool, be conservative
            if current_group:
                groups.append(current_group)
                current_group = []
            groups.append([tc])
    
    # Add remaining group
    if current_group:
        groups.append(current_group)
    
    logger.info(
        "[parallel] Grouped %d tool calls into %d batches",
        len(tool_calls),
        len(groups)
    )
    
    return groups


async def execute_with_optimal_parallelism(
    tool_calls: list[dict],
    executor_func: Any,
    enable_parallel: bool = True,
    max_concurrent: int = 3
) -> list[dict]:
    """
    Execute tool calls with optimal parallelism.
    
    This is the main entry point for parallel execution.
    
    Args:
        tool_calls: List of tool calls to execute
        executor_func: Async function that executes a single tool call
        enable_parallel: Whether to enable parallel execution
        max_concurrent: Max parallel executions
        
    Returns:
        List of results in original order
    """
    if not enable_parallel or len(tool_calls) <= 1:
        # Sequential execution
        results = []
        for tc in tool_calls:
            result = await executor_func(tc)
            results.append(result)
        return results
    
    # Group by independence
    groups = group_tool_calls_by_independence(tool_calls)
    
    all_results = []
    
    for group in groups:
        if len(group) == 1:
            # Single tool, execute directly
            result = await executor_func(group[0])
            all_results.append(result)
        else:
            # Multiple tools, execute in parallel
            group_results = await execute_tools_in_parallel(
                group,
                executor_func,
                max_concurrent
            )
            all_results.extend(group_results)
    
    return all_results


def estimate_time_savings(
    tool_calls: list[dict],
    sequential_time_estimate: float,
    parallel_groups: list[list[dict]]
) -> dict:
    """
    Estimate time savings from parallel execution.
    
    Returns:
        {
            "sequential_time": seconds,
            "parallel_time": seconds,
            "time_saved": seconds,
            "speedup": ratio
        }
    """
    if not parallel_groups:
        return {
            "sequential_time": sequential_time_estimate,
            "parallel_time": sequential_time_estimate,
            "time_saved": 0,
            "speedup": 1.0
        }
    
    # Estimate: each group takes time of longest task in group
    # Assume each tool takes ~2 seconds on average
    avg_tool_time = sequential_time_estimate / max(len(tool_calls), 1)
    
    parallel_time = 0
    for group in parallel_groups:
        # Group executes in time of longest task (assuming all same time)
        group_time = avg_tool_time
        parallel_time += group_time
    
    sequential_time = len(tool_calls) * avg_tool_time
    time_saved = sequential_time - parallel_time
    speedup = sequential_time / parallel_time if parallel_time > 0 else 1.0
    
    return {
        "sequential_time": sequential_time,
        "parallel_time": parallel_time,
        "time_saved": time_saved,
        "speedup": speedup
    }
