"""
IDE Simulator - A framework for testing AI agents autonomously.

This simulates the IDE side of the agent interaction:
- Executes tool calls (read_file, write_to_file, replace_in_file, execute_command)
- Manages conversation state
- Provides hooks for custom behavior
- Collects metrics and logs

Usage:
    from ide_simulator import IDESimulator, AgentTask
    
    sim = IDESimulator(
        workspace="/path/to/project",
        api_url="http://localhost:8080",
        workspace_id="my-project"
    )
    
    result = sim.run_task(
        question="Refactor function X to Y",
        max_turns=30
    )
    
    print(result.files_created)
    print(result.files_modified)
    print(result.success)
"""

from .simulator import IDESimulator
from .models import AgentTask, TaskResult, ToolExecution, TurnLog
from .tools import ToolExecutor, FileSystemTools, CommandTools
from .hooks import SimulatorHooks

__all__ = [
    "IDESimulator",
    "AgentTask", 
    "TaskResult",
    "ToolExecution",
    "TurnLog",
    "ToolExecutor",
    "FileSystemTools", 
    "CommandTools",
    "SimulatorHooks",
]
