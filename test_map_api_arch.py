import asyncio
import logging
import os
import json
from app.storage import store

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_map_api():
    workspace_id = "test-architecture-workspace"
    
    print(f"🚀 Testing Map API for workspace: {workspace_id}")
    
    try:
        # 1. Test Level 0: Service Architecture View
        print("\n📍 Level 0: Service Architecture View")
        result0 = await store.get_project_map(workspace_id)
        print(f"Nodes: {len(result0['nodes'])}")
        print(f"Edges: {len(result0['edges'])}")
        for node in result0['nodes']:
            if node['kind'] == 'service':
                print(f"- Service: {node['name']} ({node['id']})")
                print(f"  Description: {node.get('description')}")
        for edge in result0['edges']:
            print(f"- Edge: {edge['from']} --({edge['type']})--> {edge['to']}")

        # 2. Test Level 1: Service Drill-down
        if result0['nodes']:
            service_id = result0['nodes'][0]['id']
            print(f"\n📍 Level 1: Drill-down into service: {service_id}")
            result1 = await store.get_project_map(workspace_id, focus_path=service_id)
            print(f"Nodes: {len(result1['nodes'])}")
            print(f"Edges: {len(result1['edges'])}")
            for node in result1['nodes']:
                print(f"- File: {node['name']} ({node['id']})")
                print(f"  Role: {node.get('description')}")

        # 3. Test Level 2: File Drill-down
        if result1['nodes']:
            file_path = result1['nodes'][0]['id']
            print(f"\n📍 Level 2: Drill-down into file: {file_path}")
            result2 = await store.get_project_map(workspace_id, focus_path=file_path)
            print(f"Nodes: {len(result2['nodes'])}")
            print(f"Edges: {len(result2['edges'])}")
            for node in result2['nodes']:
                print(f"- Symbol: {node['name']} ({node['kind']})")
                print(f"  Role/Desc: {node.get('description')}")

    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(test_map_api())
