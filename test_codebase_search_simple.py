#!/usr/bin/env python3
"""
Simple local test for codebase_search - focuses on the core logic.
This test runs without requiring a fully indexed database.
"""

import asyncio
import os
import sys

# Add the project root to path
sys.path.insert(0, os.path.dirname(__file__))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()


async def test_imports():
    """Test 1: Can we import the modules?"""
    print("\n" + "=" * 80)
    print("TEST 1: Module Imports")
    print("=" * 80)
    
    try:
        from app.core.agent import codebase_search
        print("✅ Successfully imported codebase_search")
    except Exception as e:
        print(f"❌ Failed to import codebase_search: {e}")
        return False
    
    try:
        from app.core import embeddings
        print("✅ Successfully imported embeddings module")
    except Exception as e:
        print(f"❌ Failed to import embeddings: {e}")
        return False
    
    try:
        from app.storage import store
        print("✅ Successfully imported store module")
    except Exception as e:
        print(f"❌ Failed to import store: {e}")
        return False
    
    return True


async def test_environment():
    """Test 2: Are required environment variables set?"""
    print("\n" + "=" * 80)
    print("TEST 2: Environment Configuration")
    print("=" * 80)
    
    all_good = True
    
    # Check DATABASE_URL
    db_url = os.getenv("DATABASE_URL")
    if db_url:
        # Mask password in display
        display_url = db_url.split("@")[0].split("://")[0] + "://***@" + db_url.split("@")[1] if "@" in db_url else db_url[:30] + "..."
        print(f"✅ DATABASE_URL: {display_url}")
    else:
        print("❌ DATABASE_URL: Not set")
        print("   Set in .env: DATABASE_URL=postgresql://user:pass@localhost:5432/dbname")
        all_good = False
    
    # Check VOYAGE_API_KEY
    voyage_key = os.getenv("VOYAGE_API_KEY")
    if voyage_key:
        print(f"✅ VOYAGE_API_KEY: ***{voyage_key[-4:]}")
    else:
        print("❌ VOYAGE_API_KEY: Not set")
        print("   Get one at https://dash.voyageai.com/")
        print("   Set in .env: VOYAGE_API_KEY=your-key-here")
        all_good = False
    
    # Check VOYAGE_MODEL
    voyage_model = os.getenv("VOYAGE_MODEL", "voyage-code-3")
    print(f"✅ VOYAGE_MODEL: {voyage_model}")
    
    return all_good


async def test_database_connection():
    """Test 3: Can we connect to the database?"""
    print("\n" + "=" * 80)
    print("TEST 3: Database Connection")
    print("=" * 80)
    
    try:
        from app.storage import store
        
        # Try a simple connection check
        is_connected = await store.check_connection()
        if is_connected:
            print("✅ Database connection successful")
            return True
        else:
            print("❌ Database connection check returned False")
            return False
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        print("\nTroubleshooting:")
        print("  1. Make sure PostgreSQL is running")
        print("  2. Check DATABASE_URL in .env")
        print("  3. Ensure pgvector extension is installed:")
        print("     psql -d your_db -c 'CREATE EXTENSION IF NOT EXISTS vector;'")
        return False


async def test_embeddings():
    """Test 4: Can we generate embeddings?"""
    print("\n" + "=" * 80)
    print("TEST 4: Embedding Generation")
    print("=" * 80)
    
    try:
        from app.core import embeddings
        
        test_query = "test function"
        print(f"Generating embedding for: '{test_query}'")
        
        embedding = await embeddings.embed_query(test_query)
        
        if embedding and len(embedding) > 0:
            print(f"✅ Embedding generated successfully")
            print(f"   Dimensions: {len(embedding)}")
            print(f"   First 5 values: {embedding[:5]}")
            return True
        else:
            print("❌ Embedding is empty or invalid")
            return False
            
    except Exception as e:
        print(f"❌ Embedding generation failed: {e}")
        print("\nTroubleshooting:")
        print("  1. Check VOYAGE_API_KEY in .env")
        print("  2. Verify your API key is valid")
        print("  3. Check your internet connection")
        return False


async def test_vector_search():
    """Test 5: Can we perform vector search?"""
    print("\n" + "=" * 80)
    print("TEST 5: Vector Search")
    print("=" * 80)
    
    try:
        from app.core import embeddings
        from app.storage import store
        
        workspace_id = "forge-search"  # Change to your workspace ID
        query = "search function"
        
        print(f"Searching workspace '{workspace_id}' for: '{query}'")
        
        # Generate query embedding
        query_emb = await embeddings.embed_query(query)
        
        # Perform search
        results = await store.vector_search(workspace_id, query_emb, top_k=5)
        
        if results:
            print(f"✅ Vector search successful")
            print(f"   Found {len(results)} results")
            
            # Show first result details
            if len(results) > 0:
                first = results[0]
                print(f"\n   Top result:")
                print(f"     Name: {first['symbol'].get('name', 'N/A')}")
                print(f"     Kind: {first['symbol'].get('kind', 'N/A')}")
                print(f"     File: {first['symbol'].get('file_path', 'N/A')}")
                print(f"     Score: {first.get('score', 0):.3f}")
            
            return True
        else:
            print("⚠️  No results found")
            print("   This might mean:")
            print("   1. The workspace hasn't been indexed yet")
            print("   2. The workspace ID is incorrect")
            print("   3. The query didn't match any code")
            return False
            
    except Exception as e:
        print(f"❌ Vector search failed: {e}")
        import traceback
        print("\n" + traceback.format_exc())
        return False


async def test_codebase_search_function():
    """Test 6: Can we run the full codebase_search function?"""
    print("\n" + "=" * 80)
    print("TEST 6: Full codebase_search Function")
    print("=" * 80)
    
    try:
        from app.core.agent import codebase_search
        
        workspace_id = "forge-search"  # Change to your workspace ID
        query = "LLM model configuration"
        
        # Create mock state
        state = {
            'workspace_id': workspace_id,
            'messages': [],
            'enriched_context': '',
            'attached_files': {},
            'attached_images': [],
            'plan_steps': [],
            'current_step': 0
        }
        
        print(f"Running codebase_search for: '{query}'")
        print(f"Workspace: {workspace_id}")
        
        # Call the underlying function (tool.func accesses the actual function)
        if hasattr(codebase_search, 'func'):
            result = await codebase_search.func(query, state)
        else:
            # Fallback: call invoke method
            result = await codebase_search.ainvoke({"query": query, "state": state})
        
        if result:
            print(f"✅ codebase_search executed successfully")
            print(f"   Result length: {len(result)} characters")
            
            # Show preview
            preview = result[:500] + "..." if len(result) > 500 else result
            print(f"\n   Preview:")
            print("   " + "-" * 76)
            for line in preview.split('\n')[:10]:
                print(f"   {line}")
            if len(result) > 500:
                print("   ...")
            print("   " + "-" * 76)
            
            return True
        else:
            print("⚠️  codebase_search returned empty result")
            return False
            
    except Exception as e:
        print(f"❌ codebase_search failed: {e}")
        import traceback
        print("\n" + traceback.format_exc())
        return False


async def main():
    """Run all tests in sequence."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "CODEBASE SEARCH - SIMPLE TEST" + " " * 29 + "║")
    print("╚" + "=" * 78 + "╝")
    
    results = {}
    
    # Test 1: Imports
    results['imports'] = await test_imports()
    if not results['imports']:
        print("\n❌ Cannot proceed without successful imports")
        return
    
    # Test 2: Environment
    results['environment'] = await test_environment()
    if not results['environment']:
        print("\n❌ Cannot proceed without proper environment setup")
        return
    
    # Test 3: Database
    results['database'] = await test_database_connection()
    if not results['database']:
        print("\n⚠️  Skipping remaining tests (database required)")
        print_summary(results)
        return
    
    # Test 4: Embeddings
    results['embeddings'] = await test_embeddings()
    if not results['embeddings']:
        print("\n⚠️  Skipping remaining tests (embeddings required)")
        print_summary(results)
        return
    
    # Test 5: Vector Search
    results['vector_search'] = await test_vector_search()
    
    # Test 6: Full codebase_search
    results['codebase_search'] = await test_codebase_search_function()
    
    # Print summary
    print_summary(results)


def print_summary(results):
    """Print test summary."""
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {test_name.replace('_', ' ').title()}")
    
    print()
    
    all_passed = all(results.values())
    if all_passed:
        print("🎉 All tests passed! The codebase_search tool is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    
    print()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        print(traceback.format_exc())
        sys.exit(1)
