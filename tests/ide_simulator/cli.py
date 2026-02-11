"""CLI for running IDE Simulator tests."""

import argparse
import sys
import json
from pathlib import Path

from .simulator import IDESimulator
from .hooks import LoggingHooks, CompositeHooks, AssertionHooks


def main():
    parser = argparse.ArgumentParser(
        description="IDE Simulator - Test AI agents autonomously",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run a simple task
  python -m ide_simulator --workspace /path/to/project --task "Add error handling to main.py"
  
  # Run with custom settings
  python -m ide_simulator -w /project -t "Refactor auth" --max-turns 50 --timeout 300
  
  # Use a different API
  python -m ide_simulator -w /project -t "Task" --api http://localhost:9000
  
  # Quiet mode (less output)
  python -m ide_simulator -w /project -t "Task" --quiet
        """
    )
    
    parser.add_argument(
        "-w", "--workspace",
        required=True,
        help="Path to the workspace/project directory"
    )
    parser.add_argument(
        "-t", "--task",
        required=True,
        help="The task/question for the agent"
    )
    parser.add_argument(
        "--api",
        default="http://localhost:8080",
        help="API URL (default: http://localhost:8080)"
    )
    parser.add_argument(
        "--workspace-id",
        help="Workspace ID (default: workspace folder name)"
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=30,
        help="Maximum conversation turns (default: 30)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Total timeout in seconds (default: 600)"
    )
    parser.add_argument(
        "--command-timeout",
        type=int,
        default=120,
        help="Per-command timeout in seconds (default: 120)"
    )
    parser.add_argument(
        "--index",
        action="store_true",
        help="Index the workspace before running the task"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output result as JSON"
    )
    
    args = parser.parse_args()
    
    # Validate workspace
    workspace = Path(args.workspace)
    if not workspace.exists():
        print(f"Error: Workspace not found: {workspace}", file=sys.stderr)
        sys.exit(1)
    
    # Setup hooks
    hooks = LoggingHooks(verbose=not args.quiet)
    
    # Create simulator
    sim = IDESimulator(
        workspace=workspace,
        api_url=args.api,
        workspace_id=args.workspace_id,
        command_timeout=args.command_timeout,
        hooks=hooks,
    )
    
    # Index if requested
    if args.index:
        print("Indexing workspace...")
        try:
            result = sim.index_workspace()
            print(f"Indexed: {result}")
        except Exception as e:
            print(f"Indexing failed: {e}", file=sys.stderr)
            sys.exit(1)
    
    # Run task
    result = sim.run_task(
        question=args.task,
        max_turns=args.max_turns,
        timeout_seconds=args.timeout,
    )
    
    # Output
    if args.json:
        output = {
            "success": result.success,
            "status": result.status,
            "answer": result.answer,
            "total_turns": result.total_turns,
            "total_time_seconds": result.total_time_seconds,
            "files_created": result.files_created,
            "files_modified": result.files_modified,
            "commands_executed": result.commands_executed,
            "ran_build_check": result.ran_build_check,
            "build_check_passed": result.build_check_passed,
            "error": result.error,
        }
        print(json.dumps(output, indent=2))
    else:
        print(result.summary())
    
    # Exit code
    sys.exit(0 if result.success else 1)


if __name__ == "__main__":
    main()
