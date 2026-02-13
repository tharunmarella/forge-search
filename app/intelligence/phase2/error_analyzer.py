"""
Intelligent Error Analyzer - Use LLM to understand errors instead of hardcoded patterns.

This replaces hardcoded regex patterns with LLM-based analysis.
Benefits:
- Handles any error format/language
- Understands context and nuance
- Can suggest creative fixes
- No maintenance of regex patterns
"""

import logging
from typing import TypedDict
import json

logger = logging.getLogger(__name__)


class ErrorAnalysis(TypedDict):
    """Structured error analysis from LLM."""
    error_type: str
    root_cause: str
    is_fundamental_issue: bool  # Should we give up on this approach?
    suggested_fixes: list[str]
    confidence: float


ERROR_ANALYSIS_PROMPT = """You are an expert at analyzing software errors and suggesting fixes.

Analyze this error and provide structured output:

ERROR OUTPUT:
```
{error_output}
```

COMMAND THAT FAILED:
```
{command}
```

PREVIOUS ATTEMPTS (if any):
{previous_attempts}

Respond with JSON:
{{
  "error_type": "one of: missing_dependency, config_error, syntax_error, permission_error, version_mismatch, deprecated_api, command_not_found, runtime_error, network_error, other",
  "root_cause": "concise explanation of what's wrong",
  "is_fundamental_issue": true/false,  // true if this approach will NEVER work (wrong tool, nonexistent package, etc)
  "suggested_fixes": ["actionable fix 1", "actionable fix 2"],  // Ordered by likelihood to work
  "confidence": 0.0-1.0  // How confident are you in this analysis?
}}

Guidelines:
- If the error says "not found", "doesn't exist", or "deprecated", is_fundamental_issue = true
- If it's a missing dependency or config, is_fundamental_issue = false (fixable)
- Suggested fixes should be SPECIFIC commands or actions, not vague advice
- If you see the same error multiple times in previous attempts, is_fundamental_issue = true
"""


async def analyze_error_with_llm(
    error_output: str,
    command: str,
    previous_attempts: list[str],
    llm_model: any,  # Fast LLM model for analysis
) -> ErrorAnalysis:
    """
    Use LLM to analyze an error and extract insights.
    
    This replaces hardcoded regex patterns with intelligent analysis.
    """
    # Format previous attempts
    prev_text = "None" if not previous_attempts else "\n".join(
        f"{i+1}. {attempt}" for i, attempt in enumerate(previous_attempts[:5])
    )
    
    prompt = ERROR_ANALYSIS_PROMPT.format(
        error_output=error_output[:1000],  # Truncate very long errors
        command=command,
        previous_attempts=prev_text
    )
    
    try:
        # Call LLM with structured output
        response = await llm_model.ainvoke([
            {"role": "user", "content": prompt}
        ])
        
        # Parse JSON response
        content = response.content.strip()
        
        # Extract JSON from markdown code block if present
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        analysis = json.loads(content)
        
        logger.info(
            "[error_analyzer] Analysis: type=%s, fundamental=%s, confidence=%.2f",
            analysis["error_type"],
            analysis["is_fundamental_issue"],
            analysis["confidence"]
        )
        
        return analysis
        
    except Exception as e:
        logger.error("[error_analyzer] Failed to analyze error: %s", e)
        
        # Fallback to simple heuristic
        return {
            "error_type": "unknown",
            "root_cause": "Could not analyze error",
            "is_fundamental_issue": False,
            "suggested_fixes": ["Check the error message carefully"],
            "confidence": 0.3
        }


SEMANTIC_COMPARISON_PROMPT = """You are comparing two commands to determine if they're semantically similar.

FAILED COMMAND (already tried):
```
{failed_command}
```
Error: {failed_error}

NEW COMMAND (being attempted):
```
{new_command}
```

Are these commands trying to do the **same thing** in different ways?

Examples of semantic duplicates:
- "npx prisma generate" vs "npx prisma generate --config=foo" → YES (same base action)
- "npm install @nextui-org/theme" vs "npm i @nextui/theme" → YES (same package, different syntax)
- "npm install" vs "npm ci" → NO (different purposes)
- "npm install" vs "yarn install" → MAYBE (same goal, different tool)

Respond with JSON:
{{
  "is_semantic_duplicate": true/false,
  "similarity_score": 0.0-1.0,  // 0 = completely different, 1 = exactly the same
  "reasoning": "brief explanation",
  "recommendation": "continue/stop/try_different_tool"  // What should the agent do?
}}

Be strict: Only mark as duplicate if they're **truly** trying the same approach."""


async def compare_commands_semantically(
    new_command: str,
    failed_command: str,
    failed_error: str,
    llm_model: any
) -> dict:
    """
    Use LLM to compare commands semantically.
    
    This is more reliable than embedding-based similarity because the LLM
    understands the actual meaning and intent of commands.
    """
    prompt = SEMANTIC_COMPARISON_PROMPT.format(
        failed_command=failed_command,
        failed_error=failed_error[:300],
        new_command=new_command
    )
    
    try:
        response = await llm_model.ainvoke([
            {"role": "user", "content": prompt}
        ])
        
        content = response.content.strip()
        
        # Extract JSON
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        comparison = json.loads(content)
        
        logger.info(
            "[semantic_comparison] '%s' vs '%s': duplicate=%s, score=%.2f",
            new_command[:40],
            failed_command[:40],
            comparison["is_semantic_duplicate"],
            comparison["similarity_score"]
        )
        
        return comparison
        
    except Exception as e:
        logger.error("[semantic_comparison] Failed: %s", e)
        return {
            "is_semantic_duplicate": False,
            "similarity_score": 0.0,
            "reasoning": "Could not compare",
            "recommendation": "continue"
        }


USER_INTENT_ANALYSIS_PROMPT = """You are analyzing a user's message to understand their intent.

USER MESSAGE:
```
{user_message}
```

CONTEXT:
- There are {num_exhausted} exhausted approaches (failed 5+ times)
- Agent is currently stuck: {is_stuck}
- Current plan step: {current_step_desc}

What does the user want?

Respond with JSON:
{{
  "intent": "one of: continue_current_approach, try_different_approach, skip_step, provide_context, ask_question, acknowledge, new_task, force_retry, stop",
  "confidence": 0.0-1.0,
  "override_safety": false,  // true if user explicitly wants to retry exhausted approach
  "clarification_needed": false,  // true if you need more info from user
  "extracted_info": {{}}  // Any specific context/constraints mentioned
}}

Intent definitions:
- continue_current_approach: "ok", "yes", "go ahead", "continue"
- try_different_approach: "try something else", "different way"
- skip_step: "skip this", "move on", "not important"
- provide_context: User is giving new information about environment/setup
- ask_question: User is asking for clarification
- acknowledge: Simple "ok" during plan execution
- new_task: Completely new request
- force_retry: "keep trying", "retry", "do it again" (even if exhausted)
- stop: "stop", "cancel", "give up"
"""


async def analyze_user_intent(
    user_message: str,
    context: dict,
    llm_model: any
) -> dict:
    """
    Use LLM to understand what the user actually wants.
    
    This replaces simple keyword matching with intelligent intent parsing.
    """
    prompt = USER_INTENT_ANALYSIS_PROMPT.format(
        user_message=user_message,
        num_exhausted=context.get("num_exhausted", 0),
        is_stuck=context.get("is_stuck", False),
        current_step_desc=context.get("current_step", "none")
    )
    
    try:
        response = await llm_model.ainvoke([
            {"role": "user", "content": prompt}
        ])
        
        content = response.content.strip()
        
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        intent = json.loads(content)
        
        logger.info(
            "[user_intent] Message='%s' → intent=%s (confidence=%.2f, override=%s)",
            user_message[:50],
            intent["intent"],
            intent["confidence"],
            intent.get("override_safety", False)
        )
        
        return intent
        
    except Exception as e:
        logger.error("[user_intent] Failed: %s", e)
        return {
            "intent": "continue_current_approach",
            "confidence": 0.5,
            "override_safety": False,
            "clarification_needed": False,
            "extracted_info": {}
        }
