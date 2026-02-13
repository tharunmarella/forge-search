"""
Intelligent Model Routing - Choose the right model for the task.

Instead of hardcoding "use Claude for planning, GPT-4 for reasoning",
analyze the task and route to the optimal model based on:
- Task complexity
- Required capabilities
- Cost constraints
- Latency requirements
"""

import logging
from typing import Literal
import json

logger = logging.getLogger(__name__)


TaskType = Literal[
    "planning",  # Initial plan creation
    "simple_execution",  # Straightforward file operations
    "complex_reasoning",  # Debugging, understanding complex errors
    "error_analysis",  # Analyzing error messages
    "code_generation",  # Writing/modifying code
    "quick_decision",  # Simple yes/no, continue/stop decisions
]


async def analyze_task_requirements(
    task_description: str,
    context: dict,
    llm_model: any  # Fast model for analysis
) -> dict:
    """
    Analyze what capabilities are needed for the current task.
    
    Returns task analysis that helps route to optimal model.
    """
    prompt = f"""Analyze this task to determine what model capabilities are needed:

TASK: {task_description}

CONTEXT:
- Is first turn: {context.get('is_first_turn', False)}
- Has active plan: {context.get('has_plan', False)}
- Recent failures: {context.get('failure_count', 0)}
- Is stuck/reflecting: {context.get('is_stuck', False)}

Respond with JSON:
{{
  "task_type": "one of: planning, simple_execution, complex_reasoning, error_analysis, code_generation, quick_decision",
  "complexity": "low/medium/high",
  "requires_creativity": true/false,
  "requires_deep_reasoning": true/false,
  "can_use_fast_model": true/false,
  "recommended_model_tier": "fast/reasoning/planning",
  "reasoning": "brief explanation"
}}

Guidelines:
- planning: Creating initial strategy, needs smart model
- simple_execution: Reading files, running clear commands, can use fast model
- complex_reasoning: Stuck on errors, need to think through problems
- error_analysis: Understanding error messages, fast model OK
- code_generation: Writing code, medium model OK
- quick_decision: Simple choices, fast model OK
"""
    
    try:
        response = await llm_model.ainvoke([{"role": "user", "content": prompt}])
        content = response.content.strip()
        
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        analysis = json.loads(content)
        
        logger.info(
            "[model_router] Task='%s' → type=%s, tier=%s",
            task_description[:40],
            analysis["task_type"],
            analysis["recommended_model_tier"]
        )
        
        return analysis
        
    except Exception as e:
        logger.error("[model_router] Analysis failed: %s", e)
        
        # Fallback: conservative choice
        return {
            "task_type": "complex_reasoning",
            "complexity": "medium",
            "requires_creativity": True,
            "requires_deep_reasoning": True,
            "can_use_fast_model": False,
            "recommended_model_tier": "reasoning",
            "reasoning": "Analysis failed, using safe default"
        }


def select_model_based_on_analysis(
    analysis: dict,
    available_models: dict,
    budget_constraint: str = "balanced"  # "fast", "balanced", "quality"
) -> str:
    """
    Select optimal model based on task analysis and constraints.
    
    Args:
        analysis: Task analysis from analyze_task_requirements
        available_models: Dict of {tier: model_name}
        budget_constraint: Prefer speed or quality
        
    Returns:
        Model name to use
    """
    recommended_tier = analysis["recommended_model_tier"]
    
    # Budget constraint overrides
    if budget_constraint == "fast":
        # Always try to use fast model unless absolutely necessary
        if analysis["can_use_fast_model"]:
            return available_models.get("fast", available_models.get("reasoning"))
        else:
            return available_models.get("reasoning", available_models.get("fast"))
    
    elif budget_constraint == "quality":
        # Always use best model for important tasks
        if analysis["requires_deep_reasoning"] or analysis["task_type"] == "planning":
            return available_models.get("planning", available_models.get("reasoning"))
        else:
            return available_models.get("reasoning", available_models.get("fast"))
    
    else:  # balanced
        # Use recommended tier
        model = available_models.get(recommended_tier)
        if model:
            return model
        
        # Fallback to reasoning model
        return available_models.get("reasoning", available_models.get("fast"))


def estimate_token_cost(
    model_name: str,
    estimated_tokens: int
) -> float:
    """
    Estimate cost for a model call.
    
    This helps decide if it's worth using a more expensive model.
    """
    # Rough pricing (per 1M tokens)
    pricing = {
        "gpt-4o": 5.0,  # $5/1M input
        "gpt-4": 30.0,
        "claude-3-sonnet": 3.0,
        "claude-3-opus": 15.0,
        "groq-llama3": 0.05,  # Much cheaper
        "deepseek": 0.14,
    }
    
    # Get base price (use pattern matching)
    price_per_1m = 5.0  # Default
    for key, price in pricing.items():
        if key in model_name.lower():
            price_per_1m = price
            break
    
    return (estimated_tokens / 1_000_000) * price_per_1m


def should_use_cheaper_model(
    task_analysis: dict,
    current_token_usage: int,
    budget_limit: float = 0.50  # $0.50 per conversation
) -> bool:
    """
    Determine if we should downgrade to cheaper model to stay in budget.
    
    Args:
        task_analysis: Analysis of current task
        current_token_usage: Tokens used so far in conversation
        budget_limit: Max cost per conversation
        
    Returns:
        True if should use cheaper model
    """
    # If task is simple, always use cheap model
    if task_analysis["can_use_fast_model"]:
        return True
    
    # If we're approaching budget limit, use cheap model
    estimated_cost = estimate_token_cost("gpt-4o", current_token_usage)
    if estimated_cost > budget_limit * 0.8:  # 80% of budget used
        logger.warning(
            "[model_router] Approaching budget limit ($%.3f / $%.2f), switching to cheaper model",
            estimated_cost,
            budget_limit
        )
        return True
    
    return False


async def get_optimal_model_for_turn(
    state: dict,
    available_models: dict,
    budget_constraint: str = "balanced"
) -> str:
    """
    Main entry point: analyze current turn and select optimal model.
    
    Args:
        state: Current agent state with messages, plan, etc.
        available_models: Dict of available models by tier
        budget_constraint: Speed/quality preference
        
    Returns:
        Model name to use for this turn
    """
    # Extract context
    messages = state.get('messages', [])
    plan_steps = state.get('plan_steps', [])
    current_step = state.get('current_step', 0)
    
    # Determine what we're doing
    is_first_turn = len([m for m in messages if hasattr(m, 'content')]) <= 2
    has_plan = len(plan_steps) > 0
    
    # Count recent failures
    failure_count = 0
    for msg in messages[-10:]:
        if hasattr(msg, 'content') and msg.content:
            if any(sig in str(msg.content).lower() for sig in ['error', 'failed']):
                failure_count += 1
    
    is_stuck = failure_count >= 3
    
    # Get last user message to understand task
    task_description = "Continue execution"
    for msg in reversed(messages):
        if hasattr(msg, 'content') and msg.content:
            task_description = str(msg.content)[:200]
            break
    
    # Analyze task
    # Use fast model for the analysis itself to save cost
    fast_model = available_models.get("fast")
    if fast_model:
        from . import llm as llm_provider
        analysis_model = llm_provider.get_chat_model(fast_model, temperature=0)
        
        analysis = await analyze_task_requirements(
            task_description,
            {
                "is_first_turn": is_first_turn,
                "has_plan": has_plan,
                "failure_count": failure_count,
                "is_stuck": is_stuck
            },
            analysis_model
        )
    else:
        # Fallback if no fast model
        analysis = {
            "task_type": "planning" if is_first_turn else "complex_reasoning",
            "can_use_fast_model": not (is_first_turn or is_stuck),
            "recommended_model_tier": "planning" if is_first_turn else "reasoning"
        }
    
    # Select model
    model = select_model_based_on_analysis(analysis, available_models, budget_constraint)
    
    logger.info(
        "[model_router] Selected model: %s (task=%s, stuck=%s, budget=%s)",
        model,
        analysis.get("task_type"),
        is_stuck,
        budget_constraint
    )
    
    return model
