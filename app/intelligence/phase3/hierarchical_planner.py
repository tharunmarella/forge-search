"""
Hierarchical Planning - Tree-based plans that can branch and adapt.

Instead of flat plan steps that get nested descriptions like:
  "Fix errors from: Fix errors from: Fix errors from: npm install"

We use a tree structure that allows:
  - Subtasks (break complex steps into smaller ones)
  - Alternative approaches (try Plan B if Plan A fails)
  - Parallel execution (run independent tasks concurrently)
  - Proper failure handling (mark failed, try sibling approach)
"""

import logging
from typing import TypedDict, Literal
from datetime import datetime, timezone
import json

logger = logging.getLogger(__name__)


class PlanNode(TypedDict):
    """A node in the hierarchical plan tree."""
    id: str  # Unique ID like "1", "1.1", "1.2", "2"
    description: str
    status: Literal["pending", "in_progress", "completed", "failed", "skipped"]
    parent_id: str | None
    children: list[str]  # List of child IDs
    alternatives: list[str]  # Alternative approaches if this fails
    can_run_parallel: bool  # Can this run in parallel with siblings?
    metadata: dict  # Tool results, error messages, etc.


class HierarchicalPlan:
    """Tree-based plan structure."""
    
    def __init__(self):
        self.nodes: dict[str, PlanNode] = {}
        self.root_id = "0"
        self.current_node_id: str | None = None
        
    def create_from_flat_steps(self, flat_steps: list[dict]) -> None:
        """Convert flat plan steps to hierarchical structure."""
        # Create root
        self.nodes[self.root_id] = {
            "id": self.root_id,
            "description": "Root",
            "status": "completed",
            "parent_id": None,
            "children": [],
            "alternatives": [],
            "can_run_parallel": False,
            "metadata": {}
        }
        
        # Convert flat steps to top-level children
        for i, step in enumerate(flat_steps):
            node_id = str(i + 1)
            self.nodes[node_id] = {
                "id": node_id,
                "description": step["description"],
                "status": step.get("status", "pending"),
                "parent_id": self.root_id,
                "children": [],
                "alternatives": [],
                "can_run_parallel": False,
                "metadata": {}
            }
            self.nodes[self.root_id]["children"].append(node_id)
        
        # Set first pending node as current
        self._update_current_node()
    
    def _update_current_node(self) -> None:
        """Update current_node_id to next pending node."""
        # DFS to find first pending node
        def find_pending(node_id: str) -> str | None:
            node = self.nodes.get(node_id)
            if not node:
                return None
            
            if node["status"] == "pending":
                return node_id
            
            if node["status"] == "in_progress":
                # Check children first
                for child_id in node["children"]:
                    result = find_pending(child_id)
                    if result:
                        return result
                # No pending children, this is the current node
                return node_id
            
            # Check children
            for child_id in node["children"]:
                result = find_pending(child_id)
                if result:
                    return result
            
            return None
        
        self.current_node_id = find_pending(self.root_id)
    
    def add_subtask(self, parent_id: str, description: str) -> str:
        """Break a complex task into a subtask."""
        parent = self.nodes.get(parent_id)
        if not parent:
            raise ValueError(f"Parent node {parent_id} not found")
        
        # Create subtask with hierarchical ID
        child_id = f"{parent_id}.{len(parent['children']) + 1}"
        
        self.nodes[child_id] = {
            "id": child_id,
            "description": description,
            "status": "pending",
            "parent_id": parent_id,
            "children": [],
            "alternatives": [],
            "can_run_parallel": False,
            "metadata": {}
        }
        
        parent["children"].append(child_id)
        logger.info("[hierarchical_plan] Added subtask %s under %s", child_id, parent_id)
        
        return child_id
    
    def add_alternative(self, failed_node_id: str, description: str) -> str:
        """Add an alternative approach when a node fails."""
        failed_node = self.nodes.get(failed_node_id)
        if not failed_node:
            raise ValueError(f"Node {failed_node_id} not found")
        
        parent_id = failed_node["parent_id"]
        if not parent_id:
            raise ValueError("Cannot add alternative to root node")
        
        parent = self.nodes[parent_id]
        
        # Create alternative as sibling
        # Find next available ID at this level
        siblings = parent["children"]
        base_id = failed_node_id.rsplit(".", 1)[0] if "." in failed_node_id else self.root_id
        alt_id = f"{base_id}.{len(siblings) + 1}"
        
        self.nodes[alt_id] = {
            "id": alt_id,
            "description": f"Alternative: {description}",
            "status": "pending",
            "parent_id": parent_id,
            "children": [],
            "alternatives": [],
            "can_run_parallel": False,
            "metadata": {"is_alternative_to": failed_node_id}
        }
        
        parent["children"].append(alt_id)
        failed_node["alternatives"].append(alt_id)
        
        logger.info("[hierarchical_plan] Added alternative %s for failed %s", alt_id, failed_node_id)
        
        return alt_id
    
    def mark_failed_and_create_alternative(self, node_id: str, alternative_desc: str) -> str:
        """Mark node as failed and create alternative approach."""
        node = self.nodes.get(node_id)
        if not node:
            raise ValueError(f"Node {node_id} not found")
        
        node["status"] = "failed"
        alt_id = self.add_alternative(node_id, alternative_desc)
        
        # Update current node to the alternative
        self.current_node_id = alt_id
        self.nodes[alt_id]["status"] = "in_progress"
        
        return alt_id
    
    def get_parallel_tasks(self) -> list[PlanNode]:
        """Get all tasks that can run in parallel at current level."""
        if not self.current_node_id:
            return []
        
        current = self.nodes.get(self.current_node_id)
        if not current or not current["parent_id"]:
            return []
        
        parent = self.nodes[current["parent_id"]]
        
        # Find all siblings that can run in parallel and are pending
        parallel_tasks = []
        for sibling_id in parent["children"]:
            sibling = self.nodes[sibling_id]
            if (sibling["can_run_parallel"] and 
                sibling["status"] == "pending" and
                sibling_id != self.current_node_id):
                parallel_tasks.append(sibling)
        
        return parallel_tasks
    
    def to_flat_steps(self) -> list[dict]:
        """Convert back to flat steps for compatibility with existing code."""
        flat_steps = []
        
        def traverse(node_id: str, depth: int = 0):
            node = self.nodes.get(node_id)
            if not node or node_id == self.root_id:
                return
            
            # Add this node
            indent = "  " * depth
            flat_steps.append({
                "number": len(flat_steps) + 1,
                "description": f"{indent}{node['description']}",
                "status": node["status"],
                "metadata": node["metadata"]
            })
            
            # Traverse children
            for child_id in node["children"]:
                traverse(child_id, depth + 1)
        
        # Start from root's children
        for child_id in self.nodes[self.root_id]["children"]:
            traverse(child_id)
        
        return flat_steps
    
    def get_execution_summary(self) -> dict:
        """Get a summary of plan execution for debugging."""
        total = len(self.nodes) - 1  # Exclude root
        completed = sum(1 for n in self.nodes.values() if n["status"] == "completed" and n["id"] != self.root_id)
        failed = sum(1 for n in self.nodes.values() if n["status"] == "failed")
        in_progress = sum(1 for n in self.nodes.values() if n["status"] == "in_progress")
        pending = sum(1 for n in self.nodes.values() if n["status"] == "pending")
        
        return {
            "total_nodes": total,
            "completed": completed,
            "failed": failed,
            "in_progress": in_progress,
            "pending": pending,
            "current_node_id": self.current_node_id,
            "progress_percentage": (completed / total * 100) if total > 0 else 0
        }


async def detect_if_should_break_into_subtasks(
    step_description: str,
    failure_count: int,
    llm_model: any
) -> tuple[bool, list[str]]:
    """
    Use LLM to determine if a step should be broken into subtasks.
    
    Returns:
        (should_break, subtask_descriptions)
    """
    if failure_count < 3:
        return False, []
    
    prompt = f"""A plan step has failed {failure_count} times:

STEP: "{step_description}"

Is this step too complex and should be broken into smaller subtasks?

Respond with JSON:
{{
  "should_break": true/false,
  "reasoning": "why or why not",
  "subtasks": ["subtask 1", "subtask 2", ...]  // Only if should_break is true
}}

Guidelines:
- Break into subtasks if the step combines multiple actions
- Break if the step is vague ("fix the errors", "set up the project")
- Don't break if it's already atomic ("run npm install", "read file X")
"""
    
    try:
        response = await llm_model.ainvoke([{"role": "user", "content": prompt}])
        content = response.content.strip()
        
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        result = json.loads(content)
        
        logger.info(
            "[hierarchical_plan] Should break '%s'? %s",
            step_description[:50],
            result["should_break"]
        )
        
        return result["should_break"], result.get("subtasks", [])
        
    except Exception as e:
        logger.error("[hierarchical_plan] Error detecting subtasks: %s", e)
        return False, []


async def suggest_alternative_approach(
    failed_description: str,
    error_messages: list[str],
    llm_model: any
) -> str:
    """
    Use LLM to suggest an alternative approach after failure.
    
    Instead of retrying the same thing, the LLM suggests a fundamentally different approach.
    """
    errors_text = "\n".join(f"- {err[:200]}" for err in error_messages[:3])
    
    prompt = f"""A plan step has failed multiple times:

FAILED APPROACH: "{failed_description}"

ERRORS:
{errors_text}

Suggest a FUNDAMENTALLY DIFFERENT approach to achieve the same goal.

Requirements:
- Don't just add flags or change syntax
- Think about the root cause and attack it differently
- If the tool doesn't exist, suggest a different tool
- If the approach is wrong, suggest the correct approach

Respond with JSON:
{{
  "alternative_approach": "concise description of different approach",
  "reasoning": "why this is fundamentally different",
  "confidence": 0.0-1.0
}}
"""
    
    try:
        response = await llm_model.ainvoke([{"role": "user", "content": prompt}])
        content = response.content.strip()
        
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        result = json.loads(content)
        
        logger.info(
            "[hierarchical_plan] Alternative for '%s': %s",
            failed_description[:40],
            result["alternative_approach"][:60]
        )
        
        return result["alternative_approach"]
        
    except Exception as e:
        logger.error("[hierarchical_plan] Error suggesting alternative: %s", e)
        return f"Try a different approach to: {failed_description}"


def visualize_plan_tree(plan: HierarchicalPlan) -> str:
    """Generate a text visualization of the plan tree."""
    lines = []
    
    def traverse(node_id: str, prefix: str = "", is_last: bool = True):
        node = plan.nodes.get(node_id)
        if not node or node_id == plan.root_id:
            return
        
        # Status emoji
        status_emoji = {
            "pending": "⏸️",
            "in_progress": "▶️",
            "completed": "✅",
            "failed": "❌",
            "skipped": "⏭️"
        }.get(node["status"], "❓")
        
        # Current node indicator
        current = "👉 " if node_id == plan.current_node_id else ""
        
        # Branch characters
        branch = "└── " if is_last else "├── "
        
        lines.append(f"{prefix}{branch}{current}{status_emoji} [{node['id']}] {node['description']}")
        
        # Prepare prefix for children
        extension = "    " if is_last else "│   "
        child_prefix = prefix + extension
        
        # Traverse children
        children = node["children"]
        for i, child_id in enumerate(children):
            is_last_child = (i == len(children) - 1)
            traverse(child_id, child_prefix, is_last_child)
    
    lines.append("Plan Tree:")
    for child_id in plan.nodes[plan.root_id]["children"]:
        is_last = (child_id == plan.nodes[plan.root_id]["children"][-1])
        traverse(child_id, "", is_last)
    
    return "\n".join(lines)
