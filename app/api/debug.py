"""Debug and development endpoints."""

from fastapi import APIRouter, Depends
from ..utils import auth
from ..core import agent as agent_module

router = APIRouter()


@router.post("/pre-enrichment")
async def debug_pre_enrichment(
    req: dict,
    user: dict = Depends(auth.get_current_user)
):
    """
    Debug endpoint: show pre-enrichment context.
    
    Useful for understanding what semantic search results the agent sees
    before it makes decisions.
    
    Request:
        {
            "workspace_id": "my-project",
            "question": "how does authentication work?"
        }
    
    Response:
        {
            "workspace_id": "my-project",
            "question": "how does authentication work?",
            "context_length": 12345,
            "context": "... semantic search results ...",
            "has_content": true
        }
    """
    workspace_id = req.get("workspace_id")
    question = req.get("question")
    
    if not workspace_id or not question:
        return {
            "error": "Missing workspace_id or question",
            "workspace_id": workspace_id,
            "question": question,
        }
    
    try:
        # Build the pre-enrichment context (same as agent gets)
        context = await agent_module.build_pre_enrichment(workspace_id, question, {})
        return {
            "workspace_id": workspace_id,
            "question": question,
            "context_length": len(context),
            "context": context[:5000] if context else "",  # Truncate for readability
            "has_content": len(context) > 50,
        }
    except Exception as e:
        return {
            "error": str(e),
            "workspace_id": workspace_id,
            "question": question,
        }
