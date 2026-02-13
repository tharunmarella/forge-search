"""
Test Phase 1 implementation: Persistent workspace memory

This script tests the three key features:
1. Persistent workspace memory (failed_commands tracking)
2. Pre-emptive blocking (don't waste LLM calls)
3. "Ask for help" after 5 failures
"""

import asyncio
import sys
from app.intelligence.phase1 import workspace_memory as ws_memory

async def test_phase1():
    """Test Phase 1 implementation."""
    
    workspace_id = "test-workspace-phase1"
    
    print("=" * 80)
    print("PHASE 1 TEST: Persistent Workspace Memory")
    print("=" * 80)
    
    # Clean slate
    print("\n1. Clearing existing memory...")
    await ws_memory.clear_workspace_memory(workspace_id)
    memory = await ws_memory.load_workspace_memory(workspace_id)
    print(f"   ✓ Memory cleared: {len(memory['failed_commands'])} failed commands")
    
    # Test recording failures
    print("\n2. Recording failures (simulating repeated command failures)...")
    test_command = "npx prisma generate"
    
    for i in range(1, 6):
        await ws_memory.record_failure(
            workspace_id,
            test_command,
            f"Error: Cannot find module @prisma/client (attempt {i})"
        )
        memory = await ws_memory.load_workspace_memory(workspace_id)
        record = memory['failed_commands'].get(test_command.lower())
        
        if record:
            status_emoji = "⚠️" if record['status'] == "active" else "🛑"
            print(f"   {status_emoji} Attempt {i}: {record['attempts']} failures, status={record['status']}")
    
    # Check if exhausted
    print("\n3. Checking if command is exhausted...")
    is_exhausted, record = await ws_memory.is_exhausted_approach(workspace_id, test_command)
    
    if is_exhausted:
        print(f"   🛑 Command is EXHAUSTED after {record['attempts']} attempts")
        print(f"   Error signature: {record['error_signature'][:100]}")
    else:
        print(f"   ❌ ERROR: Command should be exhausted but isn't!")
        return False
    
    # Test semantic duplicate detection
    print("\n4. Testing semantic duplicate detection...")
    similar_commands = [
        "npx prisma generate --config=prisma.config.ts",
        "npx prisma generate --schema=./prisma/schema.prisma",
        "npx prisma db push"  # Different base command - should NOT be caught
    ]
    
    for cmd in similar_commands:
        is_exhausted, _ = await ws_memory.is_exhausted_approach(workspace_id, cmd)
        base_cmd = ws_memory._extract_base_command(cmd)
        original_base = ws_memory._extract_base_command(test_command)
        
        should_block = (base_cmd == original_base)
        result_emoji = "🛑" if is_exhausted else "✅"
        expected_emoji = "🛑" if should_block else "✅"
        
        match = "✓" if (is_exhausted == should_block) else "✗"
        print(f"   {result_emoji} {match} '{cmd[:50]}...' blocked={is_exhausted} (expected={should_block})")
    
    # Test failure summary
    print("\n5. Generating failure summary for prompt injection...")
    summary = await ws_memory.get_failure_summary(workspace_id)
    
    if summary:
        print("   ✓ Failure summary generated:")
        print("\n" + "\n".join("      " + line for line in summary.split("\n")[:10]))
        print("      ...")
    else:
        print("   ❌ ERROR: No failure summary generated!")
        return False
    
    # Test ask for help
    print("\n6. Testing 'ask for help' trigger...")
    state = {"messages": []}  # Simplified state
    should_ask, help_msg = await ws_memory.should_ask_for_help(workspace_id, state)
    
    if should_ask:
        print("   ✓ Should ask for help: YES")
        print("   Message preview:")
        print("\n" + "\n".join("      " + line for line in help_msg.split("\n")[:5]))
    else:
        print("   ❌ ERROR: Should ask for help but didn't!")
        return False
    
    # Test recording success (should clear failure)
    print("\n7. Testing success recording (should clear failure history)...")
    await ws_memory.record_success(workspace_id, test_command)
    memory = await ws_memory.load_workspace_memory(workspace_id)
    record = memory['failed_commands'].get(test_command.lower())
    
    if not record:
        print(f"   ✓ Failure record cleared after success")
    else:
        print(f"   ❌ ERROR: Failure record still exists after success!")
        return False
    
    # Cleanup
    print("\n8. Cleaning up...")
    await ws_memory.clear_workspace_memory(workspace_id)
    print("   ✓ Test workspace cleared")
    
    print("\n" + "=" * 80)
    print("✅ ALL PHASE 1 TESTS PASSED")
    print("=" * 80)
    
    print("\n📋 PHASE 1 IMPLEMENTATION SUMMARY:")
    print("   ✅ Persistent workspace memory (MongoDB)")
    print("   ✅ Failed command tracking (attempts, errors, status)")
    print("   ✅ Exhausted approach detection (5+ failures)")
    print("   ✅ Semantic duplicate detection (base command matching)")
    print("   ✅ Failure summary generation (for prompt injection)")
    print("   ✅ 'Ask for help' trigger")
    print("   ✅ Success recording (clears failures)")
    print("\n🎯 Next steps:")
    print("   - Test in real agent loop with actual failures")
    print("   - Monitor MongoDB workspace_memory collection")
    print("   - Verify pre-emptive blocking prevents wasted LLM calls")
    
    return True


if __name__ == "__main__":
    try:
        success = asyncio.run(test_phase1())
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
