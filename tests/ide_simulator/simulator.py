"""Main IDE Simulator class."""

import time
import uuid
import httpx
from pathlib import Path
from datetime import datetime
from typing import Any

from .models import AgentTask, TaskResult, ToolExecution, TurnLog
from .tools import ToolExecutor
from .hooks import SimulatorHooks, LoggingHooks


class IDESimulator:
    """
    Simulates an IDE for testing AI agents autonomously.
    
    This class:
    - Sends requests to the forge-search API
    - Executes tool calls locally (file ops, commands)
    - Manages conversation state
    - Tracks metrics and provides hooks for customization
    """
    
    def __init__(
        self,
        workspace: str | Path,
        api_url: str = "http://localhost:8080",
        workspace_id: str | None = None,
        command_timeout: int = 120,
        hooks: SimulatorHooks | None = None,
    ):
        self.workspace = Path(workspace)
        self.api_url = api_url.rstrip("/")
        self.workspace_id = workspace_id or self.workspace.name
        self.command_timeout = command_timeout
        self.hooks = hooks or LoggingHooks(verbose=True)
        
        # Tool executor
        self.executor = ToolExecutor(self.workspace, command_timeout)
    
    def run_task(
        self,
        question: str,
        max_turns: int = 30,
        timeout_seconds: int = 600,
        attached_files: dict[str, str] | None = None,
    ) -> TaskResult:
        """
        Run a task and return the result.
        
        Args:
            question: The task/question for the agent
            max_turns: Maximum conversation turns before aborting
            timeout_seconds: Total timeout for the task
            attached_files: Optional dict of {path: content} for files to attach
        
        Returns:
            TaskResult with metrics, logs, and outcome
        """
        start_time = time.time()
        conversation_id = str(uuid.uuid4())
        
        self.hooks.before_task(question, str(self.workspace))
        
        turns: list[TurnLog] = []
        total_api_time = 0.0
        total_tool_time = 0.0
        current_turn = 0
        final_answer: str | None = None
        final_status = "error"
        error_msg: str | None = None
        
        # Initial request payload
        tool_results: list[dict] | None = None
        
        try:
            while current_turn < max_turns:
                current_turn += 1
                turn_start = time.time()
                
                # Check timeout
                if time.time() - start_time > timeout_seconds:
                    final_status = "timeout"
                    break
                
                # Build request
                payload: dict[str, Any] = {
                    "workspace_id": self.workspace_id,
                    "conversation_id": conversation_id,
                }
                
                if current_turn == 1:
                    # First turn - send question
                    payload["question"] = question
                    if attached_files:
                        payload["attached_files"] = attached_files
                else:
                    # Subsequent turns - send tool results
                    payload["tool_results"] = tool_results
                
                # Call API
                self.hooks.before_api_call(current_turn, current_turn)
                api_start = time.time()
                response = self._call_api(payload)
                api_time = (time.time() - api_start) * 1000
                total_api_time += api_time
                
                status = response.get("status", "error")
                tool_calls = response.get("tool_calls", [])
                answer = response.get("answer")
                
                self.hooks.after_api_call(current_turn, status, tool_calls)
                
                # Execute tools if needed
                tool_executions: list[ToolExecution] = []
                tool_results = []
                
                if status == "requires_action" and tool_calls:
                    for tc in tool_calls:
                        tool_name = tc.get("name", "unknown")
                        tool_id = tc.get("id", str(uuid.uuid4()))
                        args = tc.get("args", {})
                        
                        # Hook: before tool
                        modified_args = self.hooks.before_tool(tool_name, tool_id, args)
                        if modified_args is None:
                            # Skip this tool
                            continue
                        
                        # Execute
                        tool_start = time.time()
                        execution = self.executor.execute(tool_name, tool_id, modified_args)
                        total_tool_time += execution.duration_ms
                        
                        # Hook: after tool
                        modified_output = self.hooks.after_tool(execution)
                        if modified_output is not None:
                            execution.output = modified_output
                        
                        tool_executions.append(execution)
                        
                        # Build result for next API call
                        tool_results.append({
                            "call_id": tool_id,
                            "success": execution.success,
                            "output": execution.output,
                        })
                
                # Create turn log
                turn_log = TurnLog(
                    turn_number=current_turn,
                    history_length=current_turn,
                    tool_executions=tool_executions,
                    api_time_ms=api_time,
                    status=status,
                    timestamp=datetime.now()
                )
                turns.append(turn_log)
                
                self.hooks.on_turn_complete(turn_log)
                
                # Check if we should continue
                if not self.hooks.should_continue(current_turn, turn_log):
                    final_status = "aborted"
                    break
                
                # Check if done
                if status == "done":
                    final_status = "done"
                    final_answer = answer
                    break
                elif status == "error":
                    final_status = "error"
                    error_msg = answer or "Unknown error"
                    break
            else:
                # Exhausted max_turns
                final_status = "max_turns"
        
        except Exception as e:
            error_msg = str(e)
            final_status = "error"
        
        total_time = time.time() - start_time
        
        result = TaskResult(
            success=final_status == "done",
            status=final_status,
            answer=final_answer,
            total_turns=current_turn,
            total_time_seconds=total_time,
            api_time_ms=total_api_time,
            tool_time_ms=total_tool_time,
            files_created=self.executor.files_created,
            files_modified=self.executor.files_modified,
            files_read=self.executor.files_read,
            commands_executed=self.executor.commands_executed,
            ran_build_check=self.executor.ran_build_check,
            build_check_passed=self.executor.build_check_passed,
            turns=turns,
            error=error_msg,
        )
        
        self.hooks.after_task(result.success, final_answer)
        
        return result
    
    def _call_api(self, payload: dict) -> dict:
        """Make HTTP request to forge-search API."""
        with httpx.Client(timeout=120.0) as client:
            response = client.post(
                f"{self.api_url}/chat",
                json=payload,
            )
            response.raise_for_status()
            return response.json()
    
    def index_workspace(self, file_patterns: list[str] | None = None) -> dict:
        """
        Index the workspace in forge-search.
        
        Args:
            file_patterns: Optional list of glob patterns to include
        
        Returns:
            API response dict
        """
        # Collect files
        patterns = file_patterns or ["*.py", "*.ts", "*.tsx", "*.js", "*.jsx", "*.rs"]
        files = []
        
        for pattern in patterns:
            for path in self.workspace.rglob(pattern):
                if any(skip in str(path) for skip in ["node_modules", ".git", "__pycache__", "venv"]):
                    continue
                try:
                    content = path.read_text()
                    files.append({
                        "path": str(path.relative_to(self.workspace)),
                        "content": content,
                    })
                except Exception:
                    pass  # Skip unreadable files
        
        with httpx.Client(timeout=120.0) as client:
            response = client.post(
                f"{self.api_url}/index",
                json={
                    "workspace_id": self.workspace_id,
                    "files": files,
                }
            )
            response.raise_for_status()
            return response.json()
    
    def check_index_status(self) -> dict:
        """Check if workspace is indexed."""
        with httpx.Client(timeout=30.0) as client:
            response = client.get(
                f"{self.api_url}/index/status/{self.workspace_id}"
            )
            response.raise_for_status()
            return response.json()
