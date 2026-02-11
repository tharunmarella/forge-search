"""
Basic example: Run a simple task with the IDE Simulator.
"""

import sys
import tempfile
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ide_simulator import IDESimulator
from ide_simulator.hooks import LoggingHooks


def test_simple_file_creation():
    """Test that the agent can create a simple file."""
    
    # Create a temp workspace
    with tempfile.TemporaryDirectory() as workspace:
        workspace = Path(workspace)
        
        # Create initial file
        (workspace / "main.py").write_text("print('hello')\n")
        
        # Create simulator
        sim = IDESimulator(
            workspace=workspace,
            api_url="http://localhost:8080",
            hooks=LoggingHooks(verbose=True),
        )
        
        # Run task
        result = sim.run_task(
            question="Add a function called 'greet' that takes a name and returns 'Hello, {name}!'",
            max_turns=10,
        )
        
        # Print result
        print(result.summary())
        
        # Check outcome
        assert result.success, f"Task failed: {result.error}"
        assert "greet" in (workspace / "main.py").read_text(), "Function not found in file"


def test_command_execution():
    """Test that the agent can execute commands."""
    
    with tempfile.TemporaryDirectory() as workspace:
        workspace = Path(workspace)
        
        # Create a Python project
        (workspace / "main.py").write_text("""
def add(a, b):
    return a + b

if __name__ == "__main__":
    print(add(1, 2))
""")
        
        sim = IDESimulator(
            workspace=workspace,
            api_url="http://localhost:8080",
            hooks=LoggingHooks(verbose=True),
        )
        
        result = sim.run_task(
            question="Run the main.py script and tell me what it outputs",
            max_turns=5,
        )
        
        print(result.summary())
        
        # Should have run a command
        assert any("python" in cmd for cmd in result.commands_executed), \
            "Expected python command to be executed"


if __name__ == "__main__":
    print("=" * 60)
    print("Running: test_simple_file_creation")
    print("=" * 60)
    test_simple_file_creation()
    
    print("\n" + "=" * 60)
    print("Running: test_command_execution")
    print("=" * 60)
    test_command_execution()
    
    print("\n✓ All tests passed!")
