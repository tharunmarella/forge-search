"""
Verification test: Ensure the agent verifies its work.
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ide_simulator import IDESimulator
from ide_simulator.hooks import LoggingHooks, AssertionHooks, CompositeHooks


def test_agent_runs_build():
    """Test that agent runs build/verification after changes."""
    
    with tempfile.TemporaryDirectory() as workspace:
        workspace = Path(workspace)
        
        # Create a TypeScript project
        (workspace / "package.json").write_text("""{
  "name": "test-project",
  "scripts": {
    "build": "tsc",
    "typecheck": "tsc --noEmit"
  },
  "devDependencies": {
    "typescript": "^5.0.0"
  }
}""")
        
        (workspace / "tsconfig.json").write_text("""{
  "compilerOptions": {
    "strict": true,
    "target": "ES2020",
    "module": "commonjs",
    "outDir": "./dist"
  },
  "include": ["src/**/*"]
}""")
        
        (workspace / "src").mkdir()
        (workspace / "src" / "index.ts").write_text("""
export function greet(name: string): string {
    return `Hello, ${name}!`;
}
""")
        
        # Install dependencies (simulate)
        (workspace / "node_modules").mkdir()
        
        assertion_hooks = AssertionHooks()
        hooks = CompositeHooks(
            LoggingHooks(verbose=True),
            assertion_hooks,
        )
        
        sim = IDESimulator(
            workspace=workspace,
            api_url="http://localhost:8080",
            hooks=hooks,
        )
        
        result = sim.run_task(
            question="Add a new function 'farewell' that takes a name and returns 'Goodbye, {name}!'. Make sure it compiles correctly.",
            max_turns=15,
        )
        
        print(result.summary())
        
        # Check that build was run
        assert result.ran_build_check, "Agent should have run a build check"
        
        # Check function was added
        content = (workspace / "src" / "index.ts").read_text()
        assert "farewell" in content, "Function not added"


def test_agent_fixes_errors():
    """Test that agent fixes errors it introduces."""
    
    with tempfile.TemporaryDirectory() as workspace:
        workspace = Path(workspace)
        
        # Create Python project with pytest
        (workspace / "calculator.py").write_text("""
def add(a: int, b: int) -> int:
    return a + b

def subtract(a: int, b: int) -> int:
    return a - b
""")
        
        (workspace / "test_calculator.py").write_text("""
from calculator import add, subtract

def test_add():
    assert add(2, 3) == 5

def test_subtract():
    assert subtract(5, 3) == 2
""")
        
        assertion_hooks = AssertionHooks()
        hooks = CompositeHooks(
            LoggingHooks(verbose=True),
            assertion_hooks,
        )
        
        sim = IDESimulator(
            workspace=workspace,
            api_url="http://localhost:8080",
            hooks=hooks,
        )
        
        result = sim.run_task(
            question="Add a multiply function and a test for it. Run the tests to make sure they pass.",
            max_turns=20,
        )
        
        print(result.summary())
        
        # Check that pytest was run
        assert assertion_hooks.assert_command_executed("pytest"), \
            "Agent should have run pytest"
        
        # Check function was added
        calc_content = (workspace / "calculator.py").read_text()
        test_content = (workspace / "test_calculator.py").read_text()
        
        assert "multiply" in calc_content, "multiply function not added"
        assert "test_multiply" in test_content, "test not added"


if __name__ == "__main__":
    print("=" * 60)
    print("Running: test_agent_runs_build")
    print("=" * 60)
    test_agent_runs_build()
    
    print("\n" + "=" * 60)
    print("Running: test_agent_fixes_errors")
    print("=" * 60)
    test_agent_fixes_errors()
    
    print("\n✓ All verification tests passed!")
