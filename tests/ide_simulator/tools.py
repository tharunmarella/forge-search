"""Tool executors for the IDE Simulator."""

import os
import subprocess
import difflib
import time
from pathlib import Path
from typing import Any, Protocol
from abc import ABC, abstractmethod
from datetime import datetime

from .models import ToolExecution


class ToolHandler(Protocol):
    """Protocol for tool handlers."""
    def can_handle(self, tool_name: str) -> bool:
        ...
    
    def execute(self, tool_name: str, tool_id: str, args: dict[str, Any], workspace: Path) -> ToolExecution:
        ...


class FileSystemTools:
    """Handles file system operations: read, write, replace, list."""
    
    HANDLED_TOOLS = {"read_file", "write_to_file", "replace_in_file", "list_files", "list_directory"}
    
    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.files_read: set[str] = set()
        self.files_created: set[str] = set()
        self.files_modified: set[str] = set()
    
    def can_handle(self, tool_name: str) -> bool:
        return tool_name in self.HANDLED_TOOLS
    
    def execute(self, tool_name: str, tool_id: str, args: dict[str, Any], workspace: Path) -> ToolExecution:
        start = time.time()
        try:
            if tool_name == "read_file":
                output = self._read_file(args)
                success = True
            elif tool_name == "write_to_file":
                output = self._write_file(args)
                success = True
            elif tool_name == "replace_in_file":
                output = self._replace_in_file(args)
                success = True
            elif tool_name in ("list_files", "list_directory"):
                output = self._list_directory(args)
                success = True
            else:
                output = f"Unknown file tool: {tool_name}"
                success = False
        except Exception as e:
            output = f"Error: {e}"
            success = False
        
        duration = (time.time() - start) * 1000
        return ToolExecution(
            tool_name=tool_name,
            tool_id=tool_id,
            args=args,
            success=success,
            output=output,
            duration_ms=duration,
            timestamp=datetime.now()
        )
    
    def _resolve_path(self, path: str) -> Path:
        """Resolve a path relative to workspace."""
        if os.path.isabs(path):
            return Path(path)
        return self.workspace / path
    
    def _read_file(self, args: dict) -> str:
        path = self._resolve_path(args["path"])
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        content = path.read_text()
        self.files_read.add(str(path))
        return content
    
    def _write_file(self, args: dict) -> str:
        path = self._resolve_path(args["path"])
        content = args["content"]
        
        # Track if creating or modifying
        was_existing = path.exists()
        
        # Create parent dirs
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        
        if was_existing:
            self.files_modified.add(str(path))
        else:
            self.files_created.add(str(path))
        
        return f"Successfully wrote to {path}"
    
    def _replace_in_file(self, args: dict) -> str:
        path = self._resolve_path(args["path"])
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        content = path.read_text()
        old_str = args["old_str"]
        new_str = args["new_str"]
        
        if old_str not in content:
            # Show similar matches
            lines = content.split("\n")
            close = difflib.get_close_matches(old_str.split("\n")[0], lines, n=3, cutoff=0.5)
            hint = f"Similar lines: {close}" if close else ""
            raise ValueError(f"String to replace not found in file. {hint}")
        
        new_content = content.replace(old_str, new_str, 1)
        path.write_text(new_content)
        self.files_modified.add(str(path))
        
        return f"Successfully replaced content in {path}"
    
    def _list_directory(self, args: dict) -> str:
        path = self._resolve_path(args.get("path", "."))
        if not path.exists():
            raise FileNotFoundError(f"Directory not found: {path}")
        
        entries = []
        for item in sorted(path.iterdir()):
            if item.name.startswith("."):
                continue  # Skip hidden files
            if item.is_dir():
                entries.append(f"{item.name}/")
            else:
                entries.append(item.name)
        
        return "\n".join(entries) if entries else "(empty directory)"


class CommandTools:
    """Handles command execution."""
    
    HANDLED_TOOLS = {"execute_command"}
    
    # Commands that indicate build/type checking
    BUILD_CHECK_PATTERNS = [
        "npm run build",
        "yarn build",
        "pnpm build",
        "npx tsc",
        "tsc --noEmit",
        "mypy",
        "cargo build",
        "cargo check",
        "go build",
        "pytest",
        "npm test",
        "yarn test",
    ]
    
    def __init__(self, workspace: Path, timeout: int = 120):
        self.workspace = workspace
        self.timeout = timeout
        self.commands_executed: list[str] = []
        self.ran_build_check = False
        self.build_check_passed: bool | None = None
    
    def can_handle(self, tool_name: str) -> bool:
        return tool_name in self.HANDLED_TOOLS
    
    def execute(self, tool_name: str, tool_id: str, args: dict[str, Any], workspace: Path) -> ToolExecution:
        start = time.time()
        try:
            output = self._execute_command(args)
            success = True
        except subprocess.CalledProcessError as e:
            output = f"Command failed (exit {e.returncode}):\nstdout: {e.stdout}\nstderr: {e.stderr}"
            success = False
        except subprocess.TimeoutExpired:
            output = f"Command timed out after {self.timeout}s"
            success = False
        except Exception as e:
            output = f"Error: {e}"
            success = False
        
        duration = (time.time() - start) * 1000
        return ToolExecution(
            tool_name=tool_name,
            tool_id=tool_id,
            args=args,
            success=success,
            output=output,
            duration_ms=duration,
            timestamp=datetime.now()
        )
    
    def _execute_command(self, args: dict) -> str:
        command = args["command"]
        cwd = args.get("cwd") or self.workspace
        
        self.commands_executed.append(command)
        
        # Check if this is a build/verification command
        if any(pattern in command for pattern in self.BUILD_CHECK_PATTERNS):
            self.ran_build_check = True
        
        result = subprocess.run(
            command,
            shell=True,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=self.timeout
        )
        
        # Track build check result
        if self.ran_build_check and self.build_check_passed is None:
            self.build_check_passed = result.returncode == 0
        
        if result.returncode != 0:
            raise subprocess.CalledProcessError(
                result.returncode, 
                command,
                stdout=result.stdout,
                stderr=result.stderr
            )
        
        output = result.stdout
        if result.stderr:
            output += f"\n[stderr]: {result.stderr}"
        
        return output if output else "(command completed successfully)"


class LSPTools:
    """Handles LSP-like operations (simulated for testing)."""
    
    HANDLED_TOOLS = {
        "lsp_go_to_definition", 
        "lsp_find_references", 
        "lsp_hover", 
        "lsp_rename"
    }
    
    def __init__(self, workspace: Path):
        self.workspace = workspace
    
    def can_handle(self, tool_name: str) -> bool:
        return tool_name in self.HANDLED_TOOLS
    
    def execute(self, tool_name: str, tool_id: str, args: dict[str, Any], workspace: Path) -> ToolExecution:
        """
        For now, LSP tools return a simulated response.
        In a real implementation, these would connect to actual LSP servers.
        """
        start = time.time()
        
        # Simulated responses - in reality these would use actual LSP
        output = f"[LSP Simulated] {tool_name} for {args}"
        success = True
        
        duration = (time.time() - start) * 1000
        return ToolExecution(
            tool_name=tool_name,
            tool_id=tool_id,
            args=args,
            success=success,
            output=output,
            duration_ms=duration,
            timestamp=datetime.now()
        )


class ToolExecutor:
    """Main tool executor that delegates to specific handlers."""
    
    def __init__(self, workspace: Path, command_timeout: int = 120):
        self.workspace = workspace
        self.fs_tools = FileSystemTools(workspace)
        self.cmd_tools = CommandTools(workspace, timeout=command_timeout)
        self.lsp_tools = LSPTools(workspace)
        
        self.handlers: list[ToolHandler] = [
            self.fs_tools,
            self.cmd_tools,
            self.lsp_tools,
        ]
    
    def execute(self, tool_name: str, tool_id: str, args: dict[str, Any]) -> ToolExecution:
        """Execute a tool and return the result."""
        for handler in self.handlers:
            if handler.can_handle(tool_name):
                return handler.execute(tool_name, tool_id, args, self.workspace)
        
        # Unknown tool - return failure
        return ToolExecution(
            tool_name=tool_name,
            tool_id=tool_id,
            args=args,
            success=False,
            output=f"Unknown tool: {tool_name}. Available: read_file, write_to_file, replace_in_file, execute_command, list_files, lsp_*",
            duration_ms=0,
            timestamp=datetime.now()
        )
    
    def add_handler(self, handler: ToolHandler):
        """Add a custom tool handler."""
        self.handlers.insert(0, handler)  # Custom handlers take priority
    
    # Aggregated stats
    @property
    def files_read(self) -> list[str]:
        return list(self.fs_tools.files_read)
    
    @property
    def files_created(self) -> list[str]:
        return list(self.fs_tools.files_created)
    
    @property
    def files_modified(self) -> list[str]:
        return list(self.fs_tools.files_modified)
    
    @property
    def commands_executed(self) -> list[str]:
        return self.cmd_tools.commands_executed
    
    @property
    def ran_build_check(self) -> bool:
        return self.cmd_tools.ran_build_check
    
    @property
    def build_check_passed(self) -> bool | None:
        return self.cmd_tools.build_check_passed
