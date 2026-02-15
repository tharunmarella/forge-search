import asyncio
import logging
import os
from app.api.indexing_helpers import index_files_batch
from app.storage import store

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_indexing():
    workspace_id = "test-architecture-workspace"
    
    # Sample files representing different "services"
    files = [
        {
            "path": "app/api/auth.py",
            "content": """
def login(username, password):
    \"\"\"Handles user login and returns a JWT token.\"\"\"
    return "token"

def verify_token(token):
    \"\"\"Verifies a JWT token.\"\"\"
    return True
"""
        },
        {
            "path": "app/core/search.py",
            "content": """
from app.api.auth import verify_token

def search_code(query, token):
    \"\"\"Performs semantic search after verifying token.\"\"\"
    if verify_token(token):
        return ["result1", "result2"]
    return []
"""
        },
        {
            "path": "app/storage/db.py",
            "content": """
class Database:
    def __init__(self, url):
        self.url = url
    
    def execute(self, query):
        print(f"Executing {query}")
"""
        }
    ]
    
    print(f"🚀 Starting indexing test for workspace: {workspace_id}")
    
    try:
        # 1. Ensure schema is up to date
        await store.ensure_schema()
        
        # 2. Clear workspace if exists
        await store.clear_workspace(workspace_id)
        
        # 3. Run indexing
        result = await index_files_batch(workspace_id, files)
        
        print("\n📊 Indexing Results:")
        print(f"Files indexed: {result['files_indexed']}")
        print(f"Nodes created: {result['nodes_created']}")
        print(f"Relationships created: {result['relationships_created']}")
        print(f"Embeddings generated: {result['embeddings_generated']}")
        
        # 4. Verify Services
        services = await store.get_services(workspace_id)
        print("\n🏢 Discovered Services:")
        for s in services:
            print(f"- {s['name']} ({s['id']}): {s['description']}")
            
        # 5. Verify Symbol Metadata
        async with store._cursor() as cur:
            await cur.execute(
                "SELECT name, service_id, architectural_role, description FROM symbols WHERE workspace_id = %s LIMIT 5",
                (workspace_id,)
            )
            rows = await cur.fetchall()
            print("\n📝 Sample Symbol Metadata:")
            for row in rows:
                print(f"- {row['name']}: service={row['service_id']}, role={row['architectural_role']}, desc={row['description']}")

    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
    finally:
        # Cleanup
        # await store.clear_workspace(workspace_id)
        pass

if __name__ == "__main__":
    asyncio.run(test_indexing())
