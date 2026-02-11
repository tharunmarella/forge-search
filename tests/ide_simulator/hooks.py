"""Hooks for customizing simulator behavior."""

from typing import Any, Callable
from abc import ABC, abstractmethod

from .models import ToolExecution, TurnLog


class SimulatorHooks(ABC):
    """
    Base class for simulator hooks.
    Inherit from this to customize behavior during agent runs.
    """
    
    def before_task(self, question: str, workspace: str) -> None:
        """Called before the task starts."""
        pass
    
    def after_task(self, success: bool, answer: str | None) -> None:
        """Called after the task completes."""
        pass
    
    def before_api_call(self, turn: int, messages_count: int) -> None:
        """Called before each API call to the agent."""
        pass
    
    def after_api_call(self, turn: int, status: str, tool_calls: list[dict]) -> None:
        """Called after each API call returns."""
        pass
    
    def before_tool(self, tool_name: str, tool_id: str, args: dict[str, Any]) -> dict[str, Any] | None:
        """
        Called before executing a tool.
        Return modified args, or None to skip execution.
        Raise an exception to abort the task.
        """
        return args
    
    def after_tool(self, execution: ToolExecution) -> str | None:
        """
        Called after executing a tool.
        Return a modified output, or None to use the original.
        """
        return None
    
    def on_turn_complete(self, log: TurnLog) -> None:
        """Called after each complete turn (API + all tools)."""
        pass
    
    def should_continue(self, turn: int, log: TurnLog) -> bool:
        """Return False to stop the task early."""
        return True


class LoggingHooks(SimulatorHooks):
    """Simple hooks that log all events."""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
    
    def before_task(self, question: str, workspace: str) -> None:
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Starting task: {question[:100]}...")
            print(f"Workspace: {workspace}")
            print(f"{'='*60}\n")
    
    def after_task(self, success: bool, answer: str | None) -> None:
        if self.verbose:
            status = "✓ SUCCESS" if success else "✗ FAILED"
            print(f"\n{'='*60}")
            print(f"Task {status}")
            if answer:
                print(f"Answer: {answer[:200]}...")
            print(f"{'='*60}\n")
    
    def before_api_call(self, turn: int, messages_count: int) -> None:
        if self.verbose:
            print(f"\n[Turn {turn}] Calling agent API ({messages_count} messages)...")
    
    def after_api_call(self, turn: int, status: str, tool_calls: list[dict]) -> None:
        if self.verbose:
            tools = [tc.get("name", "?") for tc in tool_calls]
            print(f"[Turn {turn}] Status: {status}, Tools: {tools}")
    
    def before_tool(self, tool_name: str, tool_id: str, args: dict[str, Any]) -> dict[str, Any]:
        if self.verbose:
            # Truncate long args for display
            display_args = {}
            for k, v in args.items():
                if isinstance(v, str) and len(v) > 100:
                    display_args[k] = v[:100] + "..."
                else:
                    display_args[k] = v
            print(f"  → {tool_name}: {display_args}")
        return args
    
    def after_tool(self, execution: ToolExecution) -> str | None:
        if self.verbose:
            status = "✓" if execution.success else "✗"
            output_preview = execution.output[:80].replace("\n", " ") if execution.output else ""
            print(f"    {status} ({execution.duration_ms:.0f}ms): {output_preview}...")
        return None
    
    def on_turn_complete(self, log: TurnLog) -> None:
        if self.verbose:
            print(f"[Turn {log.turn_number}] Complete: {len(log.tool_executions)} tools in {log.api_time_ms:.0f}ms")


class AssertionHooks(SimulatorHooks):
    """Hooks for making assertions during tests."""
    
    def __init__(self):
        self.tool_calls: list[tuple[str, dict]] = []
        self.errors: list[str] = []
    
    def after_tool(self, execution: ToolExecution) -> str | None:
        self.tool_calls.append((execution.tool_name, execution.args))
        if not execution.success:
            self.errors.append(f"{execution.tool_name}: {execution.output}")
        return None
    
    def assert_tool_called(self, tool_name: str, **expected_args) -> bool:
        """Check if a tool was called with specific args."""
        for name, args in self.tool_calls:
            if name == tool_name:
                if not expected_args:
                    return True
                if all(args.get(k) == v for k, v in expected_args.items()):
                    return True
        return False
    
    def assert_file_written(self, path: str) -> bool:
        """Check if a file was written."""
        return self.assert_tool_called("write_to_file", path=path)
    
    def assert_command_executed(self, command_contains: str) -> bool:
        """Check if a command containing the string was run."""
        for name, args in self.tool_calls:
            if name == "execute_command" and command_contains in args.get("command", ""):
                return True
        return False


class CompositeHooks(SimulatorHooks):
    """Combine multiple hooks together."""
    
    def __init__(self, *hooks: SimulatorHooks):
        self.hooks = list(hooks)
    
    def add(self, hook: SimulatorHooks):
        self.hooks.append(hook)
    
    def before_task(self, question: str, workspace: str) -> None:
        for h in self.hooks:
            h.before_task(question, workspace)
    
    def after_task(self, success: bool, answer: str | None) -> None:
        for h in self.hooks:
            h.after_task(success, answer)
    
    def before_api_call(self, turn: int, messages_count: int) -> None:
        for h in self.hooks:
            h.before_api_call(turn, messages_count)
    
    def after_api_call(self, turn: int, status: str, tool_calls: list[dict]) -> None:
        for h in self.hooks:
            h.after_api_call(turn, status, tool_calls)
    
    def before_tool(self, tool_name: str, tool_id: str, args: dict[str, Any]) -> dict[str, Any] | None:
        current_args = args
        for h in self.hooks:
            result = h.before_tool(tool_name, tool_id, current_args)
            if result is None:
                return None  # Skip execution
            current_args = result
        return current_args
    
    def after_tool(self, execution: ToolExecution) -> str | None:
        modified_output = None
        for h in self.hooks:
            result = h.after_tool(execution)
            if result is not None:
                modified_output = result
        return modified_output
    
    def on_turn_complete(self, log: TurnLog) -> None:
        for h in self.hooks:
            h.on_turn_complete(log)
    
    def should_continue(self, turn: int, log: TurnLog) -> bool:
        return all(h.should_continue(turn, log) for h in self.hooks)
