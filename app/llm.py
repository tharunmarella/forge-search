"""
Unified LLM provider layer — powered by LiteLLM.

LiteLLM gives us 100+ providers through a single interface.
No per-provider connector code needed.

Model string format (LiteLLM convention):
    "groq/moonshotai/kimi-k2-instruct-0905"
    "gemini/gemini-2.0-flash"
    "anthropic/claude-sonnet-4-20250514"
    "fireworks_ai/accounts/fireworks/models/llama-v3p1-70b-instruct"
    "openai/gpt-4o"
    "mistral/mistral-large-latest"

API keys are read from env vars automatically:
    GROQ_API_KEY, GEMINI_API_KEY, ANTHROPIC_API_KEY,
    FIREWORKS_API_KEY, OPENAI_API_KEY, MISTRAL_API_KEY, etc.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

import litellm
from litellm import acompletion

# LangChain integration (via dedicated langchain-litellm package)
from langchain_litellm import ChatLiteLLM

logger = logging.getLogger(__name__)

# Suppress litellm's verbose logging by default
litellm.suppress_debug_info = True
litellm.set_verbose = False


# ── Provider Registry ──────────────────────────────────────────────
# Maps provider prefix → env var name for the API key.
# LiteLLM handles these automatically, but we track them for
# the /models endpoint so the UI knows what's available.

PROVIDERS: dict[str, dict] = {
    "groq": {
        "name": "Groq",
        "env_key": "GROQ_API_KEY",
        "models": [
            "groq/llama-3.3-70b-versatile",
            "groq/llama-3.1-8b-instant",
            "groq/mixtral-8x7b-32768",
            "groq/moonshotai/kimi-k2-instruct-0905",
            "groq/openai/gpt-oss-20b",
        ],
    },
    "gemini": {
        "name": "Google Gemini",
        "env_key": "GEMINI_API_KEY",
        "models": [
            "gemini/gemini-2.0-flash",
            "gemini/gemini-2.0-flash-lite",
            "gemini/gemini-1.5-pro",
            "gemini/gemini-1.5-flash",
        ],
    },
    "anthropic": {
        "name": "Anthropic (Claude)",
        "env_key": "ANTHROPIC_API_KEY",
        "models": [
            "anthropic/claude-opus-4-6",
            "anthropic/claude-sonnet-4-20250514",
            "anthropic/claude-3-5-haiku-20241022",
            "anthropic/claude-3-5-sonnet-20241022",
        ],
    },
    "fireworks_ai": {
        "name": "Fireworks AI",
        "env_key": "FIREWORKS_API_KEY",
        "models": [
            "fireworks_ai/accounts/fireworks/models/llama-v3p1-70b-instruct",
            "fireworks_ai/accounts/fireworks/models/llama-v3p1-405b-instruct",
            "fireworks_ai/accounts/fireworks/models/mixtral-8x22b-instruct",
        ],
    },
    "openai": {
        "name": "OpenAI",
        "env_key": "OPENAI_API_KEY",
        "models": [
            "openai/gpt-4o",
            "openai/gpt-4o-mini",
            "openai/gpt-4-turbo",
            "openai/o3-mini",
        ],
    },
    "mistral": {
        "name": "Mistral AI",
        "env_key": "MISTRAL_API_KEY",
        "models": [
            "mistral/mistral-large-latest",
            "mistral/mistral-medium-latest",
            "mistral/mistral-small-latest",
            "mistral/codestral-latest",
        ],
    },
    "deepseek": {
        "name": "DeepSeek",
        "env_key": "DEEPSEEK_API_KEY",
        "models": [
            "deepseek/deepseek-chat",
            "deepseek/deepseek-reasoner",
        ],
    },
}


# ── Active Model Configuration ────────────────────────────────────

@dataclass
class ModelConfig:
    """Runtime model configuration (mutable at runtime via /models/set)."""
    reasoning_model: str = ""
    tool_model: str = ""
    planning_model: str = ""  # Dedicated model for plan creation/replan (optional)

    def __post_init__(self):
        if not self.reasoning_model:
            self.reasoning_model = os.getenv(
                "LLM_REASONING_MODEL",
                os.getenv("GROQ_MODEL", "groq/moonshotai/kimi-k2-instruct-0905"),
            )
            # Auto-prefix with groq/ if it looks like a bare Groq model
            if not _has_provider_prefix(self.reasoning_model):
                self.reasoning_model = f"groq/{self.reasoning_model}"

        if not self.tool_model:
            self.tool_model = os.getenv(
                "LLM_TOOL_MODEL",
                os.getenv("GROQ_TOOL_MODEL", "groq/openai/gpt-oss-20b"),
            )
            if not _has_provider_prefix(self.tool_model):
                self.tool_model = f"groq/{self.tool_model}"
        
        # Planning model: defaults to reasoning model if not set
        # Use a strong model like Claude for high-quality plans
        if not self.planning_model:
            self.planning_model = os.getenv("LLM_PLANNING_MODEL", "")
            if self.planning_model and not _has_provider_prefix(self.planning_model):
                self.planning_model = f"anthropic/{self.planning_model}"


def _has_provider_prefix(model: str) -> bool:
    """Check if a model string already has a provider prefix."""
    known_prefixes = list(PROVIDERS.keys()) + [
        "azure", "huggingface", "ollama", "together_ai",
        "replicate", "cohere", "bedrock", "sagemaker",
    ]
    return any(model.startswith(f"{p}/") for p in known_prefixes)


# Singleton config — modified by /models/set endpoint
_config = ModelConfig()


def get_config() -> ModelConfig:
    return _config


def set_reasoning_model(model: str):
    _config.reasoning_model = model
    logger.info("Reasoning model set to: %s", model)


def set_tool_model(model: str):
    _config.tool_model = model
    logger.info("Tool model set to: %s", model)


def set_planning_model(model: str):
    _config.planning_model = model
    logger.info("Planning model set to: %s", model)


def get_planning_model_name() -> str | None:
    """Get the dedicated planning model, or None to use reasoning model."""
    return _config.planning_model if _config.planning_model else None


# ── LangChain Chat Model Factory ──────────────────────────────────

def get_chat_model(
    model: str | None = None,
    temperature: float = 0.1,
    **kwargs,
) -> ChatLiteLLM:
    """
    Create a LangChain-compatible chat model backed by LiteLLM.

    Works with ANY provider — just pass the litellm model string:
        get_chat_model("groq/kimi-k2-instruct-0905")
        get_chat_model("gemini/gemini-2.0-flash")
        get_chat_model("anthropic/claude-sonnet-4-20250514")

    If model is None, uses the configured reasoning model.
    """
    if model is None:
        model = _config.reasoning_model

    logger.debug("Creating chat model: %s (temp=%.2f)", model, temperature)
    return ChatLiteLLM(
        model=model,
        temperature=temperature,
        **kwargs,
    )


def get_reasoning_model(temperature: float = 0.1, **kwargs) -> ChatLiteLLM:
    """Get the configured reasoning model (for planning, first call)."""
    return get_chat_model(_config.reasoning_model, temperature=temperature, **kwargs)


def get_tool_model(temperature: float = 0.1, **kwargs) -> ChatLiteLLM:
    """Get the configured tool model (for tool-result processing)."""
    return get_chat_model(_config.tool_model, temperature=temperature, **kwargs)


# ── Direct LiteLLM Completion (for chat.py) ────────────────────────

async def completion(
    messages: list[dict],
    model: str | None = None,
    max_tokens: int = 1024,
    temperature: float = 0.1,
    **kwargs,
) -> dict:
    """
    Direct LiteLLM async completion.

    Returns the raw litellm response object.
    Use for the simple /chat endpoint that doesn't need LangGraph.
    """
    if model is None:
        model = _config.reasoning_model

    response = await acompletion(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        **kwargs,
    )
    return response


# ── Provider Discovery ─────────────────────────────────────────────

def get_available_providers() -> list[dict]:
    """
    Return providers that have API keys configured.

    Each entry: {provider, name, models[], configured: bool}
    """
    result = []
    for provider_id, info in PROVIDERS.items():
        api_key = os.getenv(info["env_key"], "")
        configured = bool(api_key)
        result.append({
            "provider": provider_id,
            "name": info["name"],
            "env_key": info["env_key"],
            "configured": configured,
            "models": info["models"],
        })
    return result


def get_active_models() -> dict:
    """Return currently active model configuration."""
    return {
        "reasoning_model": _config.reasoning_model,
        "tool_model": _config.tool_model,
        "planning_model": _config.planning_model or "(uses reasoning_model)",
    }


def list_all_models() -> list[str]:
    """List all known models across all providers."""
    models = []
    for info in PROVIDERS.values():
        models.extend(info["models"])
    return models


# ── Startup Log ────────────────────────────────────────────────────

def log_provider_status():
    """Log which providers are configured (call at startup)."""
    configured = []
    missing = []
    for provider_id, info in PROVIDERS.items():
        if os.getenv(info["env_key"], ""):
            configured.append(info["name"])
        else:
            missing.append(info["name"])

    logger.info(
        "LLM providers configured: %s",
        ", ".join(configured) if configured else "(none)",
    )
    if missing:
        logger.info(
            "LLM providers available (set API key to enable): %s",
            ", ".join(missing),
        )
    logger.info("Reasoning model: %s", _config.reasoning_model)
    logger.info("Tool model: %s", _config.tool_model)
    logger.info("Planning model: %s", _config.planning_model or "(uses reasoning model)")
