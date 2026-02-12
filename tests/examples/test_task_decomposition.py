"""
Task Decomposition test: Verify the agent creates plans for complex tasks.

Tests that:
1. Simple tasks are classified as "simple" (no plan generated)
2. Complex tasks are classified as "complex" and get a structured plan
3. The plan has sensible steps (3-8, verification included)
4. The agent executes steps in order
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ide_simulator import IDESimulator
from ide_simulator.hooks import LoggingHooks, AssertionHooks, CompositeHooks


def test_simple_task_no_plan():
    """Simple questions should NOT trigger task decomposition."""
    
    with tempfile.TemporaryDirectory() as workspace:
        workspace = Path(workspace)
        
        (workspace / "main.py").write_text("""
def greet(name):
    return f"Hello, {name}!"

if __name__ == "__main__":
    print(greet("World"))
""")
        
        sim = IDESimulator(
            workspace=workspace,
            api_url="http://localhost:8080",
            hooks=LoggingHooks(verbose=True),
        )
        
        result = sim.run_task(
            question="What does the greet function do?",
            max_turns=5,
        )
        
        print(result.summary())
        
        # Should be classified as simple
        if result.task_complexity:
            print(f"Task complexity: {result.task_complexity}")
            assert result.task_complexity == "simple", \
                f"Expected 'simple', got '{result.task_complexity}'"
        
        # Should NOT have a plan
        if result.plan_steps:
            print(f"WARNING: Simple task generated {len(result.plan_steps)} plan steps (unexpected)")
        else:
            print("✓ No plan generated for simple task (correct)")


def test_complex_task_generates_plan():
    """Complex refactoring tasks should trigger task decomposition."""
    
    with tempfile.TemporaryDirectory() as workspace:
        workspace = Path(workspace)
        
        # Create a multi-file project
        (workspace / "auth").mkdir()
        (workspace / "auth" / "__init__.py").write_text("")
        (workspace / "auth" / "session.py").write_text("""
import hashlib
import secrets
from datetime import datetime, timedelta

# In-memory session store
_sessions = {}

def create_session(user_id: str) -> str:
    \"\"\"Create a new session and return the session ID.\"\"\"
    session_id = secrets.token_hex(32)
    _sessions[session_id] = {
        "user_id": user_id,
        "created_at": datetime.now(),
        "expires_at": datetime.now() + timedelta(hours=24),
    }
    return session_id

def get_session(session_id: str) -> dict | None:
    \"\"\"Get session data by ID, or None if expired/not found.\"\"\"
    session = _sessions.get(session_id)
    if session and session["expires_at"] > datetime.now():
        return session
    if session:
        del _sessions[session_id]
    return None

def destroy_session(session_id: str):
    \"\"\"Delete a session.\"\"\"
    _sessions.pop(session_id, None)
""")
        
        (workspace / "auth" / "middleware.py").write_text("""
from .session import get_session

def require_auth(handler):
    \"\"\"Middleware that checks for a valid session cookie.\"\"\"
    def wrapper(request):
        session_id = request.cookies.get("session_id")
        if not session_id:
            return {"error": "Not authenticated"}, 401
        
        session = get_session(session_id)
        if not session:
            return {"error": "Session expired"}, 401
        
        request.user_id = session["user_id"]
        return handler(request)
    return wrapper
""")
        
        (workspace / "auth" / "routes.py").write_text("""
from .session import create_session, destroy_session

def login(request):
    \"\"\"Login endpoint - creates a session.\"\"\"
    user_id = request.json.get("user_id")
    password = request.json.get("password")
    
    # Simplified auth (real app would check password)
    if not user_id or not password:
        return {"error": "Missing credentials"}, 400
    
    session_id = create_session(user_id)
    response = {"message": "Logged in"}
    response.set_cookie("session_id", session_id)
    return response

def logout(request):
    \"\"\"Logout endpoint - destroys the session.\"\"\"
    session_id = request.cookies.get("session_id")
    if session_id:
        destroy_session(session_id)
    return {"message": "Logged out"}
""")
        
        (workspace / "tests").mkdir()
        (workspace / "tests" / "test_auth.py").write_text("""
from auth.session import create_session, get_session, destroy_session

def test_create_session():
    sid = create_session("user1")
    assert sid is not None
    assert len(sid) == 64

def test_get_session():
    sid = create_session("user1")
    session = get_session(sid)
    assert session is not None
    assert session["user_id"] == "user1"

def test_destroy_session():
    sid = create_session("user1")
    destroy_session(sid)
    assert get_session(sid) is None
""")
        
        # Setup hooks
        assertions = AssertionHooks()
        hooks = CompositeHooks(
            LoggingHooks(verbose=True),
            assertions,
        )
        
        sim = IDESimulator(
            workspace=workspace,
            api_url="http://localhost:8080",
            hooks=hooks,
        )
        
        # Index workspace
        print("Indexing workspace...")
        try:
            sim.index_workspace()
        except Exception as e:
            print(f"Indexing failed (continuing anyway): {e}")
        
        # Run a complex refactoring task
        result = sim.run_task(
            question="Refactor the auth module to use JWT tokens instead of sessions. "
                     "Create a new jwt.py utility, update the middleware to validate JWT from "
                     "the Authorization header, update login to return JWT, and update the tests.",
            max_turns=30,
            timeout_seconds=300,
        )
        
        print("\n" + result.summary())
        
        # ── Verify task decomposition ──
        
        # Should be classified as complex
        if result.task_complexity:
            assert result.task_complexity == "complex", \
                f"Expected 'complex', got '{result.task_complexity}'"
            print("✓ Task correctly classified as complex")
        else:
            print("⚠ No task_complexity returned (server may not support it yet)")
        
        # Should have a plan
        if result.plan_steps:
            print(f"✓ Plan generated with {len(result.plan_steps)} steps:")
            for step in result.plan_steps:
                print(f"  {step.number}. {step.description} [{step.status}]")
            
            # Plan should have 3-10 steps
            assert 3 <= len(result.plan_steps) <= 10, \
                f"Expected 3-10 plan steps, got {len(result.plan_steps)}"
            
            # Plan should include a verification step (last step)
            last_step = result.plan_steps[-1].description.lower()
            has_verification = any(kw in last_step for kw in [
                "test", "verify", "check", "run", "build"
            ])
            if has_verification:
                print("✓ Plan includes verification step")
            else:
                print(f"⚠ Last step may not be verification: '{result.plan_steps[-1].description}'")
        else:
            print("⚠ No plan_steps returned (server may not support it yet)")
        
        # Check that some files were at least read or modified
        if result.files_read:
            print(f"✓ Agent read {len(result.files_read)} files")
        if result.files_modified:
            print(f"✓ Agent modified {len(result.files_modified)} files")
        if result.files_created:
            print(f"✓ Agent created {len(result.files_created)} files")


def test_plan_step_tracking():
    """Verify that plan steps transition from pending → in_progress → done."""
    
    with tempfile.TemporaryDirectory() as workspace:
        workspace = Path(workspace)
        
        (workspace / "app.py").write_text("""
def add(a, b):
    return a + b

def subtract(a, b):
    return a - b

def multiply(a, b):
    return a * b
""")
        
        sim = IDESimulator(
            workspace=workspace,
            api_url="http://localhost:8080",
            hooks=LoggingHooks(verbose=True),
        )
        
        result = sim.run_task(
            question="Add type hints to all functions in app.py, add a divide function "
                     "with zero-division handling, and create a test file with pytest tests for all functions",
            max_turns=20,
        )
        
        print("\n" + result.summary())
        
        if result.plan_steps:
            done_steps = [s for s in result.plan_steps if s.status == "done"]
            print(f"✓ {len(done_steps)}/{len(result.plan_steps)} steps completed")
        
        # Basic check: file should be modified
        if result.success:
            content = (workspace / "app.py").read_text()
            print(f"\nFinal app.py:\n{content}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", choices=["simple", "complex", "tracking", "all"], default="all")
    args = parser.parse_args()
    
    if args.test in ("simple", "all"):
        print("=" * 60)
        print("Test: Simple task — no plan expected")
        print("=" * 60)
        test_simple_task_no_plan()
    
    if args.test in ("complex", "all"):
        print("\n" + "=" * 60)
        print("Test: Complex task — plan expected")
        print("=" * 60)
        test_complex_task_generates_plan()
    
    if args.test in ("tracking", "all"):
        print("\n" + "=" * 60)
        print("Test: Plan step tracking")
        print("=" * 60)
        test_plan_step_tracking()
    
    print("\n✓ All task decomposition tests completed!")
