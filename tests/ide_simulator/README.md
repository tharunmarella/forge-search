# IDE Simulator

A framework for autonomous testing of the Forge AI agent.

## Quick Start

```python
from ide_simulator import IDESimulator

sim = IDESimulator(
    workspace="/path/to/project",
    api_url="http://localhost:8080",
)

result = sim.run_task("Refactor function X to Y")

print(result.summary())
print(result.files_modified)
print(result.ran_build_check)
```

## CLI Usage

```bash
# From forge-search/tests directory
python -m ide_simulator -w /path/to/project -t "Your task here"

# With options
python -m ide_simulator -w /project -t "Task" --max-turns 50 --index --quiet
```

## Custom Hooks

```python
from ide_simulator import IDESimulator
from ide_simulator.hooks import SimulatorHooks

class MyHooks(SimulatorHooks):
    def before_tool(self, tool_name, tool_id, args):
        print(f"About to run: {tool_name}")
        return args  # Return None to skip
    
    def after_tool(self, execution):
        if not execution.success:
            print(f"FAILED: {execution.output}")

sim = IDESimulator(workspace="/project", hooks=MyHooks())
```

## Result Object

```python
result.success          # bool
result.status           # "done", "max_turns", "timeout", "error"
result.files_created    # list[str]
result.files_modified   # list[str]
result.commands_executed # list[str]
result.ran_build_check  # bool
result.build_check_passed # bool | None
result.total_turns      # int
result.summary()        # Human-readable summary
```
