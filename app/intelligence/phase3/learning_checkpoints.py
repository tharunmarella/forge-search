"""
Learning Checkpoints - Agent pauses to consolidate knowledge before continuing.

Instead of blindly retrying, the agent can:
1. Pause and reflect on what it learned
2. Record key facts for future reference
3. Adjust its strategy based on learnings
4. Resume with better context
"""

import logging
from typing import TypedDict
from datetime import datetime, timezone
import json

logger = logging.getLogger(__name__)


class Checkpoint(TypedDict):
    """A learning checkpoint in the agent's execution."""
    checkpoint_id: str
    timestamp: str
    summary: str
    learned_facts: dict  # Key insights discovered
    failed_approaches: list[str]  # What didn't work
    successful_patterns: list[str]  # What did work
    next_steps: list[str]  # Planned actions based on learning
    confidence: float  # How confident in the learnings


async def create_checkpoint_with_llm(
    conversation_history: list,
    recent_errors: list[str],
    recent_successes: list[str],
    llm_model: any
) -> Checkpoint:
    """
    Use LLM to analyze recent activity and create a checkpoint.
    
    This helps the agent consolidate learnings instead of mindlessly retrying.
    """
    errors_text = "\n".join(f"- {err[:200]}" for err in recent_errors[:5])
    successes_text = "\n".join(f"- {s[:200]}" for s in recent_successes[:5])
    
    # Extract recent attempts
    attempts = []
    for msg in conversation_history[-20:]:
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            for tc in msg.tool_calls:
                if tc.get('name') in ['execute_command', 'execute_background']:
                    cmd = tc.get('args', {}).get('command', '')
                    if cmd:
                        attempts.append(cmd)
    
    attempts_text = "\n".join(f"{i+1}. {a}" for i, a in enumerate(attempts[-10:]))
    
    prompt = f"""You are consolidating your learnings before continuing.

RECENT ATTEMPTS:
{attempts_text}

RECENT ERRORS:
{errors_text}

RECENT SUCCESSES:
{successes_text}

Reflect on what you've learned and create a checkpoint.

Respond with JSON:
{{
  "summary": "brief summary of what you tried and learned",
  "learned_facts": {{
    "fact_key": "fact_value",
    // Example: "prisma_version": "5.0", "docker_not_installed": true
  }},
  "failed_approaches": ["approach 1 that didn't work", "approach 2 that didn't work"],
  "successful_patterns": ["what worked", "best practices discovered"],
  "next_steps": ["action 1 to try next", "action 2 if that fails"],
  "confidence": 0.0-1.0  // How confident are you in these learnings?
}}

Be specific about WHAT you learned, not just what happened.
Example:
- Bad: "npm install failed"
- Good: "Package @nextui-org/theme doesn't exist, should use @nextui-org/react instead"
"""
    
    try:
        response = await llm_model.ainvoke([{"role": "user", "content": prompt}])
        content = response.content.strip()
        
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        learning = json.loads(content)
        
        checkpoint: Checkpoint = {
            "checkpoint_id": f"cp_{int(datetime.now(timezone.utc).timestamp())}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "summary": learning["summary"],
            "learned_facts": learning["learned_facts"],
            "failed_approaches": learning["failed_approaches"],
            "successful_patterns": learning.get("successful_patterns", []),
            "next_steps": learning["next_steps"],
            "confidence": learning["confidence"]
        }
        
        logger.info(
            "[checkpoint] Created: %d facts learned, confidence=%.2f",
            len(checkpoint["learned_facts"]),
            checkpoint["confidence"]
        )
        
        return checkpoint
        
    except Exception as e:
        logger.error("[checkpoint] Error creating checkpoint: %s", e)
        
        # Return minimal checkpoint
        return {
            "checkpoint_id": f"cp_{int(datetime.now(timezone.utc).timestamp())}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "summary": "Checkpoint creation failed",
            "learned_facts": {},
            "failed_approaches": recent_errors[:3],
            "successful_patterns": recent_successes[:3],
            "next_steps": ["Review errors and try different approach"],
            "confidence": 0.3
        }


def format_checkpoint_for_prompt(checkpoint: Checkpoint) -> str:
    """Format a checkpoint for injection into system prompt."""
    message = f"## 📚 Learning Checkpoint (Created: {checkpoint['timestamp'][:19]})\n\n"
    message += f"**Summary**: {checkpoint['summary']}\n\n"
    
    if checkpoint['learned_facts']:
        message += f"**Key Facts Learned**:\n"
        for key, value in checkpoint['learned_facts'].items():
            message += f"- `{key}`: {value}\n"
        message += "\n"
    
    if checkpoint['failed_approaches']:
        message += f"**What Didn't Work**:\n"
        for approach in checkpoint['failed_approaches'][:5]:
            message += f"- {approach}\n"
        message += "\n"
    
    if checkpoint['successful_patterns']:
        message += f"**What Worked**:\n"
        for pattern in checkpoint['successful_patterns'][:5]:
            message += f"- {pattern}\n"
        message += "\n"
    
    if checkpoint['next_steps']:
        message += f"**Recommended Next Steps**:\n"
        for step in checkpoint['next_steps'][:3]:
            message += f"{checkpoint['next_steps'].index(step) + 1}. {step}\n"
    
    return message


async def should_create_checkpoint(
    state: dict,
    last_checkpoint_time: datetime | None
) -> bool:
    """
    Determine if it's time to create a checkpoint.
    
    Create checkpoint when:
    - Multiple failures (3+) since last checkpoint
    - Significant time elapsed (5+ minutes)
    - Before major strategy change
    - User explicitly asks for status
    """
    # Count recent failures
    recent_failures = 0
    for msg in state.get('messages', [])[-15:]:
        if hasattr(msg, 'content') and msg.content:
            content = str(msg.content).lower()
            if any(sig in content for sig in ['error', 'failed', 'exception']):
                recent_failures += 1
    
    # Time since last checkpoint
    if last_checkpoint_time:
        elapsed = (datetime.now(timezone.utc) - last_checkpoint_time).total_seconds()
        if elapsed < 60:  # Don't checkpoint more than once per minute
            return False
    
    # Checkpoint conditions
    if recent_failures >= 3:
        logger.info("[checkpoint] Should checkpoint: %d recent failures", recent_failures)
        return True
    
    if last_checkpoint_time and (datetime.now(timezone.utc) - last_checkpoint_time).total_seconds() > 300:
        logger.info("[checkpoint] Should checkpoint: 5+ minutes elapsed")
        return True
    
    return False


def merge_checkpoints(checkpoints: list[Checkpoint]) -> dict:
    """
    Merge multiple checkpoints into consolidated learnings.
    
    This helps when resuming from a long conversation - we don't repeat
    the same mistakes from hours ago.
    """
    if not checkpoints:
        return {"learned_facts": {}, "failed_approaches": [], "successful_patterns": []}
    
    merged = {
        "learned_facts": {},
        "failed_approaches": [],
        "successful_patterns": [],
        "checkpoints_count": len(checkpoints)
    }
    
    for cp in checkpoints:
        # Merge facts (later ones override earlier ones)
        merged["learned_facts"].update(cp.get("learned_facts", {}))
        
        # Accumulate failures (deduplicate)
        for approach in cp.get("failed_approaches", []):
            if approach not in merged["failed_approaches"]:
                merged["failed_approaches"].append(approach)
        
        # Accumulate successes (deduplicate)
        for pattern in cp.get("successful_patterns", []):
            if pattern not in merged["successful_patterns"]:
                merged["successful_patterns"].append(pattern)
    
    # Limit to most relevant items
    merged["failed_approaches"] = merged["failed_approaches"][-10:]
    merged["successful_patterns"] = merged["successful_patterns"][-10:]
    
    return merged
