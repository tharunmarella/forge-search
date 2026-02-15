#!/usr/bin/env python3
"""Test the new type-filtered search tools."""

import asyncio
import json
from app.core.agent import AgentState

# Import the actual functions, not the tool wrappers
async def search_functions_test(query: str, state):
    """Test version of search_functions"""
    from app.core import embeddings
    from app.storage import store
    
    workspace_id = state['workspace_id']
    query_emb = await embeddings.embed_query(query)
    results = await store.vector_search_by_type(workspace_id, query_emb, ['function', 'method'], top_k=5)
    return f"Found {len(results)} functions/methods"

async def search_classes_test(query: str, state):
    """Test version of search_classes"""
    from app.core import embeddings
    from app.storage import store
    
    workspace_id = state['workspace_id']
    query_emb = await embeddings.embed_query(query)
    results = await store.vector_search_by_type(workspace_id, query_emb, ['class', 'struct', 'type'], top_k=5)
    return f"Found {len(results)} classes/types"

async def search_constants_test(query: str, state):
    """Test version of search_constants"""
    from app.core import embeddings
    from app.storage import store
    
    workspace_id = state['workspace_id']
    query_emb = await embeddings.embed_query(query)
    results = await store.vector_search_by_type(workspace_id, query_emb, ['constant', 'const', 'static'], top_k=5)
    return f"Found {len(results)} constants"

async def search_files_test(query: str, state):
    """Test version of search_files"""
    from app.core import embeddings
    from app.storage import store
    
    workspace_id = state['workspace_id']
    query_emb = await embeddings.embed_query(query)
    results = await store.vector_search_by_type(workspace_id, query_emb, ['file'], top_k=5)
    return f"Found {len(results)} files"

async def test_type_filtered_search():
    """Test all the new type-filtered search tools."""
    
    # Mock state for testing
    state = AgentState({
        'workspace_id': 'forge-search',
        'messages': [],
        'enriched_context': '',
        'attached_files': {},
        'attached_images': [],
        'plan_steps': [],
        'current_step': 0
    })
    
    print("🧪 Testing Type-Filtered Search Tools\n")
    
    # Test 1: Search functions
    print("1️⃣ Testing search_functions with 'authentication'...")
    try:
        result = await search_functions_test("authentication", state)
        print(f"✅ {result}\n")
    except Exception as e:
        print(f"❌ Error: {e}\n")
    
    # Test 2: Search classes  
    print("2️⃣ Testing search_classes with 'user model'...")
    try:
        result = await search_classes_test("user model", state)
        print(f"✅ {result}\n")
    except Exception as e:
        print(f"❌ Error: {e}\n")
    
    # Test 3: Search constants
    print("3️⃣ Testing search_constants with 'timeout'...")
    try:
        result = await search_constants_test("timeout", state)
        print(f"✅ {result}\n")
    except Exception as e:
        print(f"❌ Error: {e}\n")
    
    # Test 4: Search files
    print("4️⃣ Testing search_files with 'api'...")
    try:
        result = await search_files_test("api", state)
        print(f"✅ {result}\n")
    except Exception as e:
        print(f"❌ Error: {e}\n")

if __name__ == "__main__":
    asyncio.run(test_type_filtered_search())