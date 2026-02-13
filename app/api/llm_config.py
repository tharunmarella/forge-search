"""LLM model configuration endpoints."""

from fastapi import APIRouter
from ..core import llm as llm_provider
from ..models import ActiveModelsResponse, SetModelRequest, SetModelResponse, ProviderInfo

router = APIRouter()


@router.get("/", response_model=ActiveModelsResponse)
async def get_models():
    """
    List available LLM providers, their models, and the current config.
    
    Providers with a configured API key show configured=true.
    Switch models at runtime with POST /models/set.
    """
    active = llm_provider.get_active_models()
    providers_raw = llm_provider.get_available_providers()
    
    providers = [
        ProviderInfo(
            provider=p["provider"],
            name=p["name"],
            env_key=p["env_key"],
            configured=p["configured"],
            models=p["models"],
        )
        for p in providers_raw
    ]
    
    return ActiveModelsResponse(
        reasoning_model=active["reasoning_model"],
        tool_model=active["tool_model"],
        providers=providers,
    )


@router.post("/set", response_model=SetModelResponse)
async def set_models(req: SetModelRequest):
    """
    Switch the active reasoning and/or tool model at runtime.
    
    Model string format: "provider/model-name"
    Examples:
        "groq/moonshotai/kimi-k2-instruct-0905"
        "gemini/gemini-2.0-flash"
        "anthropic/claude-sonnet-4-20250514"
        "fireworks_ai/accounts/fireworks/models/llama-v3p1-70b-instruct"
        "openai/gpt-4o"
    """
    if req.reasoning_model:
        llm_provider.set_reasoning_model(req.reasoning_model)
    if req.tool_model:
        llm_provider.set_tool_model(req.tool_model)
    
    config = llm_provider.get_config()
    
    changes = []
    if req.reasoning_model:
        changes.append(f"reasoning → {req.reasoning_model}")
    if req.tool_model:
        changes.append(f"tool → {req.tool_model}")
    
    return SetModelResponse(
        reasoning_model=config.reasoning_model,
        tool_model=config.tool_model,
        message=f"Updated: {', '.join(changes)}" if changes else "No changes",
    )
