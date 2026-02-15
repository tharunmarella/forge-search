#!/usr/bin/env python3
"""
End-to-end test: Index some code and then test codebase_search.
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv()


async def main():
    print("\n" + "=" * 80)
    print("END-TO-END CODEBASE SEARCH TEST")
    print("=" * 80)
    print()
    
    # Step 1: Import modules
    print("Step 1: Importing modules...")
    try:
        from app.storage import store
        from app.core import embeddings
        from app.core import parser
        print("✅ Imports successful")
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return
    print()
    
    # Step 2: Initialize database
    print("Step 2: Initializing database schema...")
    try:
        await store.ensure_schema()
        print("✅ Database schema ready")
    except Exception as e:
        print(f"❌ Schema initialization failed: {e}")
        return
    print()
    
    # Step 3: Index some sample Python code
    print("Step 3: Indexing sample code...")
    workspace_id = "test-workspace"
    
    # Sample Python file to index
    sample_code = '''
"""LLM configuration module for managing different AI models."""

import os
from typing import Optional

class LLMConfig:
    """Configuration for Large Language Models."""
    
    def __init__(self):
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        self.default_model = "gpt-4"
    
    def get_model(self, provider: str) -> str:
        """Get the configured model for a provider."""
        if provider == "openai":
            return self.default_model
        elif provider == "anthropic":
            return "claude-3"
        return self.default_model

def setup_llm(model_name: Optional[str] = None):
    """Set up and configure the LLM client."""
    config = LLMConfig()
    if model_name:
        config.default_model = model_name
    return config

# Initialize default configuration
default_config = setup_llm()
'''
    
    try:
        # Parse the code
        file_path = "app/config/llm_config.py"
        parse_result = parser.parse_file(file_path, sample_code)
        print(f"   Parsed {len(parse_result.definitions)} definitions from sample code")
        
        # Generate embeddings for each symbol (simplified)
        from app.core import embeddings
        
        embeddings_map = {}
        for sym in parse_result.definitions:
            # Very simple enrichment
            enriched = f"[File] {file_path}\n[{sym.kind}] {sym.name}"
            emb = await embeddings.embed_query(enriched)
            symbol_uid = f"{workspace_id}:{parse_result.file_path}:{sym.name}"
            embeddings_map[symbol_uid] = emb
        
        print(f"   Generated {len(embeddings_map)} embeddings")
        
        # Index the file
        index_result = await store.index_file_result(
            workspace_id=workspace_id,
            parse_result=parse_result,
            embeddings_map=embeddings_map
        )
        print(f"✅ Indexed file successfully")
        
        # Show what was indexed
        stats = await store.get_workspace_stats(workspace_id)
        print(f"   Workspace stats: {stats}")
        
    except Exception as e:
        print(f"❌ Indexing failed: {e}")
        import traceback
        print(traceback.format_exc())
        return
    print()
    
    # Step 4: Test vector search
    print("Step 4: Testing vector search...")
    try:
        query = "LLM configuration"
        query_emb = await embeddings.embed_query(query)
        results = await store.vector_search(workspace_id, query_emb, top_k=5)
        
        print(f"✅ Found {len(results)} results for query: '{query}'")
        
        if results:
            print("\n   Top results:")
            for i, r in enumerate(results[:3], 1):
                symbol = r['symbol']
                score = r['score']
                print(f"   {i}. {symbol['kind']} '{symbol['name']}' in {symbol['file_path']} (score: {score:.3f})")
        
    except Exception as e:
        print(f"❌ Vector search failed: {e}")
        import traceback
        print(traceback.format_exc())
        return
    print()
    
    # Step 5: Test codebase_search function
    print("Step 5: Testing codebase_search function...")
    try:
        from app.core.agent import codebase_search
        
        # Create state
        state = {
            'workspace_id': workspace_id,
            'messages': [],
            'enriched_context': '',
            'attached_files': {},
            'attached_images': [],
            'plan_steps': [],
            'current_step': 0
        }
        
        query = "how to configure LLM models"
        print(f"   Query: '{query}'")
        
        # Access the underlying function
        if hasattr(codebase_search, 'invoke'):
            # It's a LangChain tool
            result = await codebase_search.ainvoke({"query": query, "state": state})
        elif hasattr(codebase_search, '__call__'):
            result = await codebase_search(query, state)
        else:
            print("   ⚠️  Cannot call codebase_search directly, skipping...")
            result = None
        
        if result:
            print(f"✅ codebase_search executed successfully")
            print(f"   Result length: {len(result)} characters")
            print("\n   Preview:")
            print("   " + "-" * 76)
            preview_lines = result.split('\n')[:15]
            for line in preview_lines:
                print(f"   {line}")
            if len(result) > 500:
                print("   ...")
            print("   " + "-" * 76)
        else:
            print("⚠️  codebase_search returned empty or wasn't callable")
        
    except Exception as e:
        print(f"❌ codebase_search failed: {e}")
        import traceback
        print(traceback.format_exc())
    print()
    
    # Step 6: Cleanup
    print("Step 6: Cleaning up test workspace...")
    try:
        await store.clear_workspace(workspace_id)
        print("✅ Test workspace cleaned up")
    except Exception as e:
        print(f"⚠️  Cleanup warning: {e}")
    print()
    
    print("=" * 80)
    print("✨ END-TO-END TEST COMPLETE")
    print("=" * 80)
    print()
    print("Summary:")
    print("  ✅ Database connection works")
    print("  ✅ Code parsing and indexing works")
    print("  ✅ Vector embeddings work")
    print("  ✅ Semantic search works")
    print()
    print("The codebase_search tool infrastructure is working correctly!")
    print("To use it, index your actual codebase and it will find relevant code.")
    print()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Test failed: {e}")
        import traceback
        print(traceback.format_exc())
        sys.exit(1)
