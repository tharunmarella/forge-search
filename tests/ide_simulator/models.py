"""Data models for the IDE Simulator."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, List
from datetime import datetime
from enum import Enum


class ToolStatus(Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    SKIPPED = "skipped"


@dataclass
class ToolExecution:
    """Record of a single tool execution."""
    tool_name: str
    tool_id: str
    args: dict[str, Any]
    success: bool
    output: str
    duration_ms: float
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def status(self) -> ToolStatus:
        return ToolStatus.SUCCESS if self.success else ToolStatus.FAILURE


@dataclass
class TurnLog:
    """Log of a single conversation turn."""
    turn_number: int
    history_length: int
    tool_executions: list[ToolExecution]
    api_time_ms: float
    status: str  # "requires_action", "done", "error"
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def tools_called(self) -> list[str]:
        return [t.tool_name for t in self.tool_executions]
    
    @property
    def all_succeeded(self) -> bool:
        return all(t.success for t in self.tool_executions)


@dataclass
class AgentTask:
    """Definition of a task to run."""
    question: str
    workspace_id: str
    workspace_path: str
    max_turns: int = 30
    timeout_seconds: int = 600
    attached_files: dict[str, str] | None = None
    
    # Optional hooks
    before_tool: Any = None  # Callable[[str, dict], None]
    after_tool: Any = None   # Callable[[str, dict, bool, str], None]
    on_turn_complete: Any = None  # Callable[[TurnLog], None]


@dataclass
class PlanStepResult:
    """A plan step returned by the agent."""
    number: int
    description: str
    status: str  # "pending", "in_progress", "done", "failed"


@dataclass
class TaskResult:
    """Result of running an agent task."""
    success: bool
    status: str  # "done", "max_turns", "timeout", "error"
    answer: str | None
    
    # Metrics
    total_turns: int
    total_time_seconds: float
    api_time_ms: float
    tool_time_ms: float
    
    # File changes
    files_created: list[str]
    files_modified: list[str]
    files_read: list[str]
    commands_executed: list[str]
    
    # Verification
    ran_build_check: bool
    build_check_passed: bool | None
    
    # Full logs
    turns: list[TurnLog]
    error: str | None = None
    
    # Task decomposition
    task_complexity: str | None = None
    plan_steps: list[PlanStepResult] | None = None
    current_step: int | None = None
    
    @property
    def total_tool_calls(self) -> int:
        return sum(len(t.tool_executions) for t in self.turns)
    
    @property
    def failed_tool_calls(self) -> int:
        return sum(1 for t in self.turns for e in t.tool_executions if not e.success)
    
    def summary(self) -> str:
        """Generate a human-readable summary."""
        lines = [
            f"{'='*60}",
            f"Task Result: {self.status.upper()}",
            f"{'='*60}",
            f"Turns: {self.total_turns}",
            f"Time: {self.total_time_seconds:.1f}s (API: {self.api_time_ms:.0f}ms, Tools: {self.tool_time_ms:.0f}ms)",
            f"Tool calls: {self.total_tool_calls} ({self.failed_tool_calls} failed)",
        ]
        
        # Plan info
        if self.task_complexity:
            lines.append(f"")
            lines.append(f"Task complexity: {self.task_complexity}")
        if self.plan_steps:
            lines.append(f"Plan ({len(self.plan_steps)} steps):")
            for step in self.plan_steps:
                icon = {"pending": "⬜", "in_progress": "🔄", "done": "✅", "failed": "❌"}.get(step.status, "?")
                lines.append(f"  {icon} {step.number}. {step.description}")
            if self.current_step:
                lines.append(f"Last active step: {self.current_step}")
        
        lines.extend([
            f"",
            f"Files created:  {self.files_created}",
            f"Files modified: {self.files_modified}",
            f"Files read:     {len(self.files_read)} files",
            f"Commands run:   {len(self.commands_executed)}",
            f"",
            f"Build check: {'Yes' if self.ran_build_check else 'No'}",
        ])
        if self.ran_build_check:
            lines.append(f"Build passed: {self.build_check_passed}")
        if self.answer:
            lines.append(f"\nAnswer: {self.answer[:500]}...")
        if self.error:
            lines.append(f"\nError: {self.error}")
        return "\n".join(lines)
