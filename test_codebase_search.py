#!/usr/bin/env python3
"""
Local test script for the codebase_search tool.
Tests the semantic search functionality with detailed error reporting.
"""

import asyncio
import os
import sys
import traceback
from typing import Dict, Any

# Add the project root to path
sys.path.insert(0, os.path.dirname(__file__))


async def test_codebase_search():
    """Test the codebase_search tool with various queries."""
    
    print("=" * 80)
    print("🔍 CODEBASE SEARCH TOOL - LOCAL TEST")
    print("=" * 80)
    print()
    
    # Step 1: Check imports
    print("📦 Step 1: Checking imports...")
    try:
        from app.core.agent import AgentState, codebase_search
        from app.core import embeddings
        from app.storage import store
        print("✅ All imports successful")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("\nMake sure you have installed all dependencies:")
        print("  pip install -r requirements.txt")
        return
    print()
    
    # Step 2: Check environment variables
    print("🔑 Step 2: Checking environment variables...")
    required_vars = {
        "DATABASE_URL": os.getenv("DATABASE_URL"),
        "VOYAGE_API_KEY": os.getenv("VOYAGE_API_KEY"),
    }
    
    optional_vars = {
        "VOYAGE_MODEL": os.getenv("VOYAGE_MODEL", "voyage-code-3"),
        "GROQ_API_KEY": os.getenv("GROQ_API_KEY"),
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY"),
    }
    
    print("\nRequired environment variables:")
    for var, value in required_vars.items():
        status = "✅" if value else "❌"
        if "KEY" in var:
            display_value = "***" + value[-4:] if value else "(not set)"
        else:
            display_value = value[:50] + "..." if value and len(value) > 50 else value or "(not set)"
        print(f"  {status} {var}: {display_value}")
    
    print("\nOptional environment variables:")
    for var, value in optional_vars.items():
        status = "✅" if value else "⚠️ "
        if "KEY" in var:
            display_value = "***" + value[-4:] if value else "(not set)"
        else:
            display_value = str(value)
        print(f"  {status} {var}: {display_value}")
    
    if not required_vars["DATABASE_URL"]:
        print("\n❌ DATABASE_URL is required. Set it in your .env file.")
        print("   Example: DATABASE_URL=postgresql://user:pass@localhost:5432/forge_search")
        return
    
    if not required_vars["VOYAGE_API_KEY"]:
        print("\n❌ VOYAGE_API_KEY is required for embeddings. Get one at https://dash.voyageai.com/")
        print("   Set it in your .env file: VOYAGE_API_KEY=your-key-here")
        return
    print()
    
    # Step 3: Test database connection
    print("🗄️  Step 3: Testing database connection...")
    try:
        # Try to connect to the database
        connection_test = await store.get_workspace_metadata("test-connection")
        print("✅ Database connection successful")
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        print("\nTroubleshooting:")
        print("  1. Make sure PostgreSQL is running")
        print("  2. Check your DATABASE_URL in .env")
        print("  3. Run database migrations if needed")
        return
    print()
    
    # Step 4: Test embeddings model
    print("🤖 Step 4: Testing embeddings model...")
    try:
        test_text = "test embedding"
        test_embedding = await embeddings.embed_query(test_text)
        if test_embedding and len(test_embedding) > 0:
            print(f"✅ Embeddings working (dimension: {len(test_embedding)})")
        else:
            print("❌ Embeddings returned empty result")
            return
    except Exception as e:
        print(f"❌ Embeddings test failed: {e}")
        print("\nTroubleshooting:")
        print("  1. Check if you have the required model files")
        print("  2. Make sure sentence-transformers is installed")
        print("  3. Check EMBEDDINGS_MODEL in .env")
        return
    print()
    
    # Step 5: Set up test workspace
    print("🏗️  Step 5: Setting up test workspace...")
    workspace_id = "forge-search"  # Change this to your actual workspace ID
    
    state: Dict[str, Any] = {
        'workspace_id': workspace_id,
        'messages': [],
        'enriched_context': '',
        'attached_files': {},
        'attached_images': [],
        'plan_steps': [],
        'current_step': 0
    }
    
    print(f"   Using workspace: {workspace_id}")
    
    # Check if workspace has indexed data
    try:
        test_query_emb = await embeddings.embed_query("test")
        test_results = await store.vector_search(workspace_id, test_query_emb, top_k=1)
        
        if not test_results:
            print(f"⚠️  Warning: No indexed data found for workspace '{workspace_id}'")
            print("   You may need to index your codebase first.")
            print("   Run the indexing script or use the API to index files.")
        else:
            print(f"✅ Found indexed data ({len(test_results)} sample results)")
    except Exception as e:
        print(f"⚠️  Could not verify workspace data: {e}")
    print()
    
    # Step 6: Run codebase_search tests
    print("🔍 Step 6: Running codebase_search tests...")
    print()
    
    test_queries = [
        ("authentication logic", "Testing search for authentication-related code"),
        ("database queries", "Testing search for database operations"),
        ("API endpoints", "Testing search for API route handlers"),
        ("error handling", "Testing search for error handling patterns"),
        ("LLM model configuration", "Testing search for LLM setup"),
    ]
    
    for i, (query, description) in enumerate(test_queries, 1):
        print(f"Test {i}/{len(test_queries)}: {description}")
        print(f"Query: '{query}'")
        print("-" * 80)
        
        try:
            # Call the codebase_search function
            result = await codebase_search(query, state)
            
            if result:
                # Show first 500 chars of result
                result_preview = result[:500] + "..." if len(result) > 500 else result
                print(f"✅ Success! Got {len(result)} characters of results")
                print("\nPreview:")
                print(result_preview)
            else:
                print("⚠️  Search returned empty result")
                
        except Exception as e:
            print(f"❌ Error: {e}")
            print("\nFull traceback:")
            print(traceback.format_exc())
        
        print()
        print()
    
    # Step 7: Test with component focus
    print("🎯 Step 7: Testing with component focus...")
    print()
    
    try:
        print("Test: Searching for 'search' with component_focus='api'")
        print("-" * 80)
        
        result = await codebase_search(
            query="search implementation",
            state=state,
            component_focus="api"
        )
        
        if result:
            result_preview = result[:500] + "..." if len(result) > 500 else result
            print(f"✅ Success! Got {len(result)} characters of results")
            print("\nPreview:")
            print(result_preview)
        else:
            print("⚠️  Search returned empty result")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nFull traceback:")
        print(traceback.format_exc())
    
    print()
    print()
    
    # Final summary
    print("=" * 80)
    print("✨ TEST SUMMARY")
    print("=" * 80)
    print()
    print("If all tests passed:")
    print("  ✅ The codebase_search tool is working correctly!")
    print()
    print("If tests failed:")
    print("  1. Check the error messages above")
    print("  2. Verify your database has indexed data")
    print("  3. Check environment variables in .env")
    print("  4. Make sure all dependencies are installed")
    print()
    print("To index your codebase:")
    print("  1. Start the server: python -m app.main")
    print("  2. Use the /index endpoint or indexing API")
    print()


if __name__ == "__main__":
    try:
        asyncio.run(test_codebase_search())
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        print("\nFull traceback:")
        print(traceback.format_exc())
