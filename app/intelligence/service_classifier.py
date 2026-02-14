"""
LLM-based service detection and classification for hierarchical system architecture view.

Uses LLM reasoning to generalize service boundaries and generate architectural
metadata for embeddings and the project map.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..core import llm
from ..storage import store

logger = logging.getLogger(__name__)

@dataclass
class ServiceInfo:
    """Classified service for a file or symbol."""
    id: str
    name: str
    description: str
    service_type: str  # 'api', 'core', 'data', 'intelligence', 'gateway', 'utils'
    architectural_role: str = ""

@dataclass
class FileClassification:
    """LLM-generated classification for a single file."""
    file_path: str
    service_id: str
    service_name: str
    architectural_role: str
    description: str

class LLMServiceClassifier:
    """
    Uses LLM to discover and classify services in a workspace.
    """
    
    def __init__(self, workspace_id: str):
        self.workspace_id = workspace_id
        self.model = llm.get_reasoning_model(temperature=0.1)
        self._cache: Dict[str, FileClassification] = {}

    async def discover_workspace_architecture(self, files_metadata: List[Dict[str, Any]]) -> List[ServiceInfo]:
        """
        Analyze the whole workspace structure to identify high-level services.
        files_metadata: List of {path: str, symbols: List[str]}
        """
        # Prepare a condensed view of the project for the LLM
        project_summary = []
        for f in files_metadata[:100]:  # Cap to avoid context overflow
            path = f["path"]
            symbols = ", ".join(f.get("symbols", [])[:10])
            project_summary.append(f"- {path}: [{symbols}]")
        
        summary_text = "\n".join(project_summary)
        
        prompt = f"""You are an expert software architect. Analyze the following project structure and identify the high-level services/components (Level 0 blocks).

Project Structure (File Path: [Top Symbols]):
{summary_text}

Identify 4-8 logical services. For each service, provide:
1. A unique ID (e.g., service_auth)
2. A clear name (e.g., Authentication Service)
3. A brief description of its responsibility
4. Its service type (one of: api, core, data, intelligence, gateway, utils)

Respond ONLY with a JSON list of objects:
[
  {{"id": "service_id", "name": "Service Name", "description": "...", "service_type": "..."}},
  ...
]"""

        try:
            response = await llm.completion(
                messages=[{"role": "user", "content": prompt}],
                model=llm.get_config().reasoning_model,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            # Handle potential markdown wrapping
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            
            data = json.loads(content)
            services_data = data.get("services", data) if isinstance(data, dict) else data
            
            services = []
            for s in services_data:
                services.append(ServiceInfo(
                    id=s["id"],
                    name=s["name"],
                    description=s["description"],
                    service_type=s["service_type"]
                ))
            
            # Store services in database
            for s in services:
                await store.upsert_service(
                    self.workspace_id,
                    s.id,
                    s.name,
                    s.description,
                    s.service_type
                )
                
            return services
            
        except Exception as e:
            logger.error(f"Architecture discovery failed: {e}")
            return []

    async def classify_files_batch(self, files_metadata: List[Dict[str, Any]], services: List[ServiceInfo]) -> Dict[str, FileClassification]:
        """
        Assign each file to a discovered service and define its architectural role.
        """
        service_defs = "\n".join([f"- {s.id}: {s.name} ({s.description})" for s in services])
        
        # Process in smaller batches to stay within LLM limits
        batch_size = 20
        results = {}
        
        for i in range(0, len(files_metadata), batch_size):
            batch = files_metadata[i:i+batch_size]
            
            file_list = []
            for f in batch:
                path = f["path"]
                symbols = ", ".join(f.get("symbols", [])[:5])
                file_list.append(f"- {path} (symbols: {symbols})")
            
            files_text = "\n".join(file_list)
            
            prompt = f"""Assign each of the following files to one of these services:
{service_defs}

Files to classify:
{files_text}

For each file, determine:
1. service_id: The ID of the service it belongs to.
2. architectural_role: Its role (e.g., 'API Router', 'Data Model', 'Business Logic', 'Utility', 'Middleware').
3. description: A 1-sentence description of what this file does.

Respond ONLY with a JSON object where keys are file paths:
{{
  "path/to/file.py": {{
    "service_id": "...",
    "architectural_role": "...",
    "description": "..."
  }},
  ...
}}"""

            try:
                response = await llm.completion(
                    messages=[{"role": "user", "content": prompt}],
                    model=llm.get_config().reasoning_model,
                    response_format={"type": "json_object"}
                )
                
                content = response.choices[0].message.content
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                
                batch_results = json.loads(content)
                
                for path, data in batch_results.items():
                    # Find service name
                    s_name = next((s.name for s in services if s.id == data["service_id"]), "Unknown")
                    
                    classification = FileClassification(
                        file_path=path,
                        service_id=data["service_id"],
                        service_name=s_name,
                        architectural_role=data["architectural_role"],
                        description=data["description"]
                    )
                    results[path] = classification
                    self._cache[path] = classification
                    
            except Exception as e:
                logger.error(f"Batch classification failed: {e}")
                
        return results

    def get_cached_classification(self, file_path: str) -> Optional[FileClassification]:
        return self._cache.get(file_path)

# Heuristic fallback if LLM is unavailable or for initial indexing
def classify_file_by_path_heuristic(file_path: str) -> ServiceInfo:
    """
    Original heuristic-based classification.
    """
    path = file_path.replace("\\", "/").strip("/")
    parts = path.split("/")
    
    if not parts:
        return ServiceInfo("service_app", "Application", "Root application", "core")
        
    top = parts[0].lower()
    
    if top == "app":
        if len(parts) >= 2:
            second = parts[1].lower()
            if second == "api":
                return ServiceInfo("service_api", "API Gateway", "HTTP endpoints and routing", "gateway")
            if second == "core":
                return ServiceInfo("service_core", "Core Logic", "Core business logic", "core")
            if second == "storage":
                return ServiceInfo("service_storage", "Storage Layer", "Database and persistence", "data")
            if second == "intelligence":
                return ServiceInfo("service_intelligence", "Intelligence", "AI and reasoning", "intelligence")
            if second == "models":
                return ServiceInfo("service_models", "Data Models", "Schemas and models", "data")
        return ServiceInfo("service_app", "Application", "Main application module", "core")
        
    return ServiceInfo(f"service_{top}", top.title(), f"Module: {top}", "core")
