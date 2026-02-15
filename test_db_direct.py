#!/usr/bin/env python3
"""Test the database type filtering directly."""

import asyncio
import os
from app.storage import store

async def test_db_type_filtering():
    """Test the vector_search_by_type function directly."""
    
    print("🧪 Testing Database Type Filtering\n")
    
    # Test with a dummy embedding (all zeros - just to test the SQL)
    dummy_embedding = [0.0] * 1024
    workspace_id = 'forge-search'
    
    # Test 1: Search functions
    print("1️⃣ Testing function/method filtering...")
    try:
        results = await store.vector_search_by_type(
            workspace_id, dummy_embedding, ['function', 'method'], top_k=5, score_threshold=0.0
        )
        print(f"✅ Found {len(results)} functions/methods")
        if results:
            for r in results[:2]:
                sym = r['symbol']
                print(f"   - {sym['name']} ({sym['kind']}) in {sym['file_path']}")
        print()
    except Exception as e:
        print(f"❌ Error: {e}\n")
    
    # Test 2: Search classes
    print("2️⃣ Testing class/struct/type filtering...")
    try:
        results = await store.vector_search_by_type(
            workspace_id, dummy_embedding, ['class', 'struct', 'type'], top_k=5, score_threshold=0.0
        )
        print(f"✅ Found {len(results)} classes/types")
        if results:
            for r in results[:2]:
                sym = r['symbol']
                print(f"   - {sym['name']} ({sym['kind']}) in {sym['file_path']}")
        print()
    except Exception as e:
        print(f"❌ Error: {e}\n")
    
    # Test 3: Search constants
    print("3️⃣ Testing constant filtering...")
    try:
        results = await store.vector_search_by_type(
            workspace_id, dummy_embedding, ['constant', 'const', 'static'], top_k=5, score_threshold=0.0
        )
        print(f"✅ Found {len(results)} constants")
        if results:
            for r in results[:2]:
                sym = r['symbol']
                print(f"   - {sym['name']} ({sym['kind']}) in {sym['file_path']}")
        print()
    except Exception as e:
        print(f"❌ Error: {e}\n")
    
    # Test 4: Search files
    print("4️⃣ Testing file filtering...")
    try:
        results = await store.vector_search_by_type(
            workspace_id, dummy_embedding, ['file'], top_k=5, score_threshold=0.0
        )
        print(f"✅ Found {len(results)} files")
        if results:
            for r in results[:2]:
                sym = r['symbol']
                print(f"   - {sym['name']} ({sym['kind']}) in {sym['file_path']}")
        print()
    except Exception as e:
        print(f"❌ Error: {e}\n")

if __name__ == "__main__":
    asyncio.run(test_db_type_filtering())