"""Tool executors for the IDE Simulator."""
from __future__ import annotations

import os
import subprocess
import difflib
import time
import socket
import signal
import threading
import tempfile
from pathlib import Path
from typing import Any, Protocol
from abc import ABC, abstractmethod
from datetime import datetime
from dataclasses import dataclass, field

from .models import ToolExecution


@dataclass
class BackgroundProcess:
    """Represents a background process with output capture."""
    pid: int
    command: str
    process: subprocess.Popen
    output_file: Path
    started_at: datetime = field(default_factory=datetime.now)
    
    @property
    def is_running(self) -> bool:
        return self.process.poll() is None
    
    @property
    def exit_code(self) -> int | None:
        return self.process.poll()
    
    @property
    def runtime_seconds(self) -> float:
        return (datetime.now() - self.started_at).total_seconds()


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


class ProcessTools:
    """Handles background process execution, monitoring, and port utilities."""
    
    HANDLED_TOOLS = {
        "execute_background",
        "read_process_output", 
        "check_process_status",
        "kill_process",
        "wait_for_port",
        "check_port",
        "kill_port"
    }
    
    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.background_processes: dict[int, BackgroundProcess] = {}
        self._output_dir = Path(tempfile.mkdtemp(prefix="forge_process_"))
    
    def can_handle(self, tool_name: str) -> bool:
        return tool_name in self.HANDLED_TOOLS
    
    def execute(self, tool_name: str, tool_id: str, args: dict[str, Any], workspace: Path) -> ToolExecution:
        start = time.time()
        try:
            if tool_name == "execute_background":
                output = self._execute_background(args)
            elif tool_name == "read_process_output":
                output = self._read_process_output(args)
            elif tool_name == "check_process_status":
                output = self._check_process_status(args)
            elif tool_name == "kill_process":
                output = self._kill_process(args)
            elif tool_name == "wait_for_port":
                output = self._wait_for_port(args)
            elif tool_name == "check_port":
                output = self._check_port(args)
            elif tool_name == "kill_port":
                output = self._kill_port(args)
            else:
                output = f"Unknown process tool: {tool_name}"
                raise ValueError(output)
            success = True
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
    
    def _execute_background(self, args: dict) -> str:
        """
        Execute a command in the background and return immediately.
        
        Args:
            command: The command to execute
            cwd: Working directory (optional, defaults to workspace)
            wait_seconds: Seconds to wait for initial output (default: 3)
        
        Returns:
            JSON-like string with pid, initial_output, and running status
        """
        command = args["command"]
        cwd = args.get("cwd") or str(self.workspace)
        wait_seconds = args.get("wait_seconds", 3)
        
        # Create output file for this process
        output_file = self._output_dir / f"proc_{int(time.time() * 1000)}.log"
        
        # Open file for writing output
        with open(output_file, "w") as f:
            process = subprocess.Popen(
                command,
                shell=True,
                cwd=str(cwd),
                stdout=f,
                stderr=subprocess.STDOUT,
                start_new_session=True  # Detach from parent
            )
        
        # Store the background process
        bg_proc = BackgroundProcess(
            pid=process.pid,
            command=command,
            process=process,
            output_file=output_file
        )
        self.background_processes[process.pid] = bg_proc
        
        # Wait briefly for initial output
        time.sleep(wait_seconds)
        
        # Read initial output
        initial_output = ""
        if output_file.exists():
            initial_output = output_file.read_text()[-4096:]  # Last 4KB
        
        return (
            f"Process started in background.\n"
            f"PID: {process.pid}\n"
            f"Running: {bg_proc.is_running}\n"
            f"Output file: {output_file}\n"
            f"--- Initial output ({wait_seconds}s) ---\n{initial_output}"
        )
    
    def _read_process_output(self, args: dict) -> str:
        """
        Read output from a background process.
        
        Args:
            pid: Process ID to read output from
            tail_lines: Number of lines from the end (default: 100)
            follow_seconds: Seconds to wait for new output (default: 0)
        """
        pid = args["pid"]
        tail_lines = args.get("tail_lines", 100)
        follow_seconds = args.get("follow_seconds", 0)
        
        if pid not in self.background_processes:
            raise ValueError(f"No background process with PID {pid}. Active PIDs: {list(self.background_processes.keys())}")
        
        bg_proc = self.background_processes[pid]
        
        # Optionally wait for more output
        if follow_seconds > 0:
            time.sleep(follow_seconds)
        
        # Read output file
        if not bg_proc.output_file.exists():
            return f"No output file found for PID {pid}"
        
        content = bg_proc.output_file.read_text()
        lines = content.split("\n")
        
        # Get last N lines
        if len(lines) > tail_lines:
            lines = lines[-tail_lines:]
            output = f"... (showing last {tail_lines} lines) ...\n" + "\n".join(lines)
        else:
            output = "\n".join(lines)
        
        status = "running" if bg_proc.is_running else f"exited (code: {bg_proc.exit_code})"
        runtime = f"{bg_proc.runtime_seconds:.1f}s"
        
        return (
            f"PID: {pid} | Status: {status} | Runtime: {runtime}\n"
            f"--- Output ---\n{output}"
        )
    
    def _check_process_status(self, args: dict) -> str:
        """
        Check status of a background process.
        
        Args:
            pid: Process ID to check (optional, if not provided returns all)
        """
        pid = args.get("pid")
        
        if pid is not None:
            if pid not in self.background_processes:
                raise ValueError(f"No background process with PID {pid}")
            
            bg_proc = self.background_processes[pid]
            return (
                f"PID: {pid}\n"
                f"Command: {bg_proc.command}\n"
                f"Running: {bg_proc.is_running}\n"
                f"Exit code: {bg_proc.exit_code}\n"
                f"Runtime: {bg_proc.runtime_seconds:.1f}s\n"
                f"Started: {bg_proc.started_at.isoformat()}"
            )
        
        # Return all processes
        if not self.background_processes:
            return "No background processes running."
        
        lines = ["Active background processes:"]
        for p_pid, bg_proc in self.background_processes.items():
            status = "running" if bg_proc.is_running else f"exited({bg_proc.exit_code})"
            lines.append(f"  PID {p_pid}: {status} | {bg_proc.runtime_seconds:.1f}s | {bg_proc.command[:50]}...")
        
        return "\n".join(lines)
    
    def _kill_process(self, args: dict) -> str:
        """
        Kill a background process.
        
        Args:
            pid: Process ID to kill
            force: Use SIGKILL instead of SIGTERM (default: False)
        """
        pid = args["pid"]
        force = args.get("force", False)
        
        if pid not in self.background_processes:
            # Try to kill any process with this PID
            try:
                sig = signal.SIGKILL if force else signal.SIGTERM
                os.kill(pid, sig)
                return f"Sent {'SIGKILL' if force else 'SIGTERM'} to PID {pid}"
            except ProcessLookupError:
                return f"No process with PID {pid}"
            except PermissionError:
                return f"Permission denied to kill PID {pid}"
        
        bg_proc = self.background_processes[pid]
        
        if not bg_proc.is_running:
            return f"Process {pid} already exited with code {bg_proc.exit_code}"
        
        if force:
            bg_proc.process.kill()
        else:
            bg_proc.process.terminate()
        
        # Wait briefly for it to exit
        try:
            bg_proc.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            bg_proc.process.kill()
        
        return f"Process {pid} terminated. Exit code: {bg_proc.exit_code}"
    
    def _wait_for_port(self, args: dict) -> str:
        """
        Wait until a port is accepting connections.
        
        Args:
            port: Port number to check
            host: Host to check (default: localhost)
            timeout: Max seconds to wait (default: 30)
            interval: Seconds between checks (default: 1)
        """
        port = args["port"]
        host = args.get("host", "localhost")
        timeout = args.get("timeout", 30)
        interval = args.get("interval", 1)
        
        start = time.time()
        attempts = 0
        
        while time.time() - start < timeout:
            attempts += 1
            try:
                with socket.create_connection((host, port), timeout=1):
                    elapsed = time.time() - start
                    return (
                        f"Port {port} is now accepting connections!\n"
                        f"Host: {host}\n"
                        f"Time waited: {elapsed:.1f}s\n"
                        f"Attempts: {attempts}"
                    )
            except (socket.timeout, ConnectionRefusedError, OSError):
                time.sleep(interval)
        
        raise TimeoutError(
            f"Port {port} on {host} did not become available within {timeout}s "
            f"({attempts} attempts)"
        )
    
    def _check_port(self, args: dict) -> str:
        """
        Check if a port is currently in use.
        
        Args:
            port: Port number to check
            host: Host to check (default: localhost)
        """
        port = args["port"]
        host = args.get("host", "localhost")
        
        try:
            with socket.create_connection((host, port), timeout=1):
                # Port is open, try to find what's using it
                try:
                    result = subprocess.run(
                        f"lsof -i :{port} -t",
                        shell=True,
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    pids = result.stdout.strip().split("\n") if result.stdout.strip() else []
                    pid_info = f"PIDs: {', '.join(pids)}" if pids else "PID: unknown"
                except Exception:
                    pid_info = "PID: could not determine"
                
                return f"Port {port} is IN USE on {host}. {pid_info}"
        except (socket.timeout, ConnectionRefusedError, OSError):
            return f"Port {port} is AVAILABLE on {host}."
    
    def _kill_port(self, args: dict) -> str:
        """
        Kill the process using a specific port.
        
        Args:
            port: Port number
            force: Use SIGKILL instead of SIGTERM (default: False)
        """
        port = args["port"]
        force = args.get("force", False)
        
        # Find PIDs using this port
        try:
            result = subprocess.run(
                f"lsof -i :{port} -t",
                shell=True,
                capture_output=True,
                text=True,
                timeout=5
            )
            pids = [p.strip() for p in result.stdout.strip().split("\n") if p.strip()]
        except Exception as e:
            raise RuntimeError(f"Failed to find process on port {port}: {e}")
        
        if not pids:
            return f"No process found using port {port}"
        
        killed = []
        failed = []
        
        for pid in pids:
            try:
                pid_int = int(pid)
                sig = signal.SIGKILL if force else signal.SIGTERM
                os.kill(pid_int, sig)
                killed.append(pid)
            except (ValueError, ProcessLookupError, PermissionError) as e:
                failed.append(f"{pid}: {e}")
        
        result_lines = []
        if killed:
            result_lines.append(f"Killed PIDs: {', '.join(killed)}")
        if failed:
            result_lines.append(f"Failed: {'; '.join(failed)}")
        
        return "\n".join(result_lines) if result_lines else "No action taken"
    
    def cleanup(self):
        """Clean up all background processes and temp files."""
        for pid, bg_proc in self.background_processes.items():
            if bg_proc.is_running:
                try:
                    bg_proc.process.terminate()
                    bg_proc.process.wait(timeout=5)
                except Exception:
                    bg_proc.process.kill()
        
        # Clean up output directory
        import shutil
        if self._output_dir.exists():
            shutil.rmtree(self._output_dir, ignore_errors=True)


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
        self.process_tools = ProcessTools(workspace)
        
        self.handlers: list[ToolHandler] = [
            self.fs_tools,
            self.cmd_tools,
            self.process_tools,
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
    
    @property
    def background_processes(self) -> dict[int, BackgroundProcess]:
        return self.process_tools.background_processes
    
    def cleanup(self):
        """Clean up resources (background processes, temp files)."""
        self.process_tools.cleanup()
