"""
Refactoring test: Test multi-file refactoring capabilities.
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ide_simulator import IDESimulator
from ide_simulator.hooks import LoggingHooks, AssertionHooks, CompositeHooks


def test_rename_function():
    """Test renaming a function across multiple files."""
    
    with tempfile.TemporaryDirectory() as workspace:
        workspace = Path(workspace)
        
        # Create multi-file project
        (workspace / "utils.py").write_text("""
def calculate_total(items):
    '''Calculate total price of items.'''
    return sum(item['price'] for item in items)

def format_currency(amount):
    return f"${amount:.2f}"
""")
        
        (workspace / "main.py").write_text("""
from utils import calculate_total, format_currency

def process_order(order):
    total = calculate_total(order['items'])
    return {
        'total': format_currency(total),
        'items': order['items']
    }

if __name__ == "__main__":
    order = {'items': [{'price': 10}, {'price': 20}]}
    print(process_order(order))
""")
        
        (workspace / "test_utils.py").write_text("""
from utils import calculate_total

def test_calculate_total():
    items = [{'price': 10}, {'price': 20}, {'price': 30}]
    assert calculate_total(items) == 60
""")
        
        # Setup hooks
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
        
        # Index first
        print("Indexing workspace...")
        sim.index_workspace()
        
        # Run refactoring task
        result = sim.run_task(
            question="Rename the function 'calculate_total' to 'compute_sum' across all files",
            max_turns=20,
        )
        
        print(result.summary())
        
        # Verify
        assert result.success, f"Task failed: {result.error}"
        
        # Check files were modified
        utils_content = (workspace / "utils.py").read_text()
        main_content = (workspace / "main.py").read_text()
        test_content = (workspace / "test_utils.py").read_text()
        
        assert "compute_sum" in utils_content, "utils.py not updated"
        assert "compute_sum" in main_content, "main.py not updated"
        assert "compute_sum" in test_content, "test_utils.py not updated"
        assert "calculate_total" not in utils_content, "Old name still in utils.py"
        
        print("✓ All files updated correctly!")


def test_extract_function():
    """Test extracting code into a new function."""
    
    with tempfile.TemporaryDirectory() as workspace:
        workspace = Path(workspace)
        
        (workspace / "processor.py").write_text("""
def process_data(data):
    # Validate data
    if not data:
        raise ValueError("Empty data")
    if not isinstance(data, list):
        raise ValueError("Data must be a list")
    if len(data) > 1000:
        raise ValueError("Too many items")
    
    # Process each item
    results = []
    for item in data:
        processed = item.upper() if isinstance(item, str) else str(item)
        results.append(processed)
    
    return results
""")
        
        sim = IDESimulator(
            workspace=workspace,
            api_url="http://localhost:8080",
            hooks=LoggingHooks(verbose=True),
        )
        
        result = sim.run_task(
            question="Extract the validation logic (lines 3-8) into a separate function called 'validate_data'",
            max_turns=15,
        )
        
        print(result.summary())
        
        content = (workspace / "processor.py").read_text()
        assert "def validate_data" in content, "New function not created"
        assert "validate_data(data)" in content, "Function not called"


if __name__ == "__main__":
    print("=" * 60)
    print("Running: test_rename_function")
    print("=" * 60)
    test_rename_function()
    
    print("\n" + "=" * 60)
    print("Running: test_extract_function")
    print("=" * 60)
    test_extract_function()
    
    print("\n✓ All refactoring tests passed!")
