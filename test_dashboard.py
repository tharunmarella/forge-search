#!/usr/bin/env python3
"""
Quick test script to verify the traces dashboard API works.
"""

import asyncio
import os
import sys
sys.path.insert(0, '/Users/tharun/Documents/projects/forge-search')

# Load environment variables
from dotenv import load_dotenv
load_dotenv('/Users/tharun/Documents/projects/forge-search/.env')

from app.api.traces import list_traces, get_trace, get_trace_flow

async def test_dashboard():
    print("🧪 Testing Traces Dashboard API\n")
    
    # Test 1: List traces
    print("1️⃣ Testing list_traces()...")
    try:
        result = await list_traces()
        print(f"   ✅ Found {result['count']} traces")
        if result['traces']:
            trace = result['traces'][0]
            print(f"   📊 First trace: {trace.get('thread_id', 'N/A')[:20]}...")
            print(f"      Status: {trace.get('status')}")
            print(f"      Time: {trace.get('execution_time_ms')}ms")
            print(f"      Messages: {trace.get('message_count')}")
            return trace['_id']
        else:
            print("   ℹ️  No traces found in database")
            return None
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return None
    
async def test_trace_detail(trace_id):
    if not trace_id:
        return
    
    print(f"\n2️⃣ Testing get_trace({trace_id[:12]}...)...")
    try:
        trace = await get_trace(trace_id)
        print(f"   ✅ Retrieved trace details")
        print(f"      Workspace: {trace.get('workspace_id')}")
        print(f"      Timestamp: {trace.get('timestamp')}")
        response = trace.get('response', {})
        print(f"      Message count: {len(response.get('messages', []))}")
        print(f"      Has plan: {bool(response.get('plan_steps'))}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print(f"\n3️⃣ Testing get_trace_flow({trace_id[:12]}...)...")
    try:
        flow = await get_trace_flow(trace_id)
        print(f"   ✅ Generated flow graph")
        print(f"      Nodes: {len(flow['nodes'])}")
        print(f"      Edges: {len(flow['edges'])}")
        metadata = flow.get('metadata', {})
        print(f"      Workspace: {metadata.get('workspace_id')}")
        print(f"      Total nodes: {metadata.get('total_nodes')}")
        print(f"      Total edges: {metadata.get('total_edges')}")
    except Exception as e:
        print(f"   ❌ Error: {e}")

async def main():
    print("=" * 60)
    trace_id = await test_dashboard()
    if trace_id:
        await test_trace_detail(trace_id)
    print("\n" + "=" * 60)
    print("✅ Dashboard API tests complete!\n")
    print("🌐 Access dashboard at: http://localhost:8080/traces/dashboard/index.html")
    print("📊 API docs at: http://localhost:8080/docs")
    print("\n💡 To start the server:")
    print("   cd /Users/tharun/Documents/projects/forge-search")
    print("   python -m app.main")

if __name__ == "__main__":
    asyncio.run(main())
