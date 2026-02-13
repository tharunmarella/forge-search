"""
Adaptive Configuration - Learn optimal thresholds from actual agent behavior.

Instead of hardcoding values like "exhausted after 5 failures" or "similarity threshold 0.85",
this module learns what works for each workspace based on historical data.
"""

import logging
from typing import TypedDict
from datetime import datetime, timezone
from motor.motor_asyncio import AsyncIOMotorClient
import os

logger = logging.getLogger(__name__)

# MongoDB connection
MONGODB_URI = os.getenv("MONGODB_URI", "")
mongo_client = AsyncIOMotorClient(MONGODB_URI) if MONGODB_URI else None
db = mongo_client["forge_workspace_memory"] if mongo_client else None


class AdaptiveThresholds(TypedDict):
    """Learnable thresholds that adapt based on workspace behavior."""
    exhaustion_threshold: int  # After N failures, mark as exhausted
    ask_help_threshold: int  # After N failures, ask for help
    semantic_similarity_threshold: float  # For loop detection
    max_replans: int  # Max auto-replans before forcing skip
    confidence_threshold: float  # Min confidence to trust LLM analysis


# Default thresholds (conservative starting point)
DEFAULT_THRESHOLDS: AdaptiveThresholds = {
    "exhaustion_threshold": 5,
    "ask_help_threshold": 5,
    "semantic_similarity_threshold": 0.85,
    "max_replans": 2,
    "confidence_threshold": 0.7,
}


async def load_adaptive_config(workspace_id: str) -> AdaptiveThresholds:
    """
    Load adaptive configuration for a workspace.
    
    If no custom config exists, returns defaults.
    Over time, these values will be tuned based on what works for this workspace.
    """
    if not db:
        return DEFAULT_THRESHOLDS.copy()
    
    try:
        collection = db["adaptive_config"]
        doc = await collection.find_one({"workspace_id": workspace_id})
        
        if doc:
            thresholds = {
                k: doc.get(k, DEFAULT_THRESHOLDS[k])
                for k in DEFAULT_THRESHOLDS.keys()
            }
            logger.info(
                "[adaptive_config] Loaded for %s: exhaustion=%d, ask_help=%d",
                workspace_id,
                thresholds["exhaustion_threshold"],
                thresholds["ask_help_threshold"]
            )
            return thresholds
        else:
            return DEFAULT_THRESHOLDS.copy()
            
    except Exception as e:
        logger.error("[adaptive_config] Error loading: %s", e)
        return DEFAULT_THRESHOLDS.copy()


async def update_threshold_based_on_outcome(
    workspace_id: str,
    threshold_name: str,
    outcome: str,  # "success", "failure", "user_frustrated"
    context: dict = None
) -> None:
    """
    Adjust a threshold based on observed outcome.
    
    Example: If user asks for help at 3 failures (before our threshold of 5),
    we learn that ask_help_threshold should be lower for this workspace.
    """
    if not db:
        return
    
    try:
        collection = db["adaptive_config"]
        config = await load_adaptive_config(workspace_id)
        
        # Adjustment logic
        if threshold_name == "exhaustion_threshold":
            if outcome == "user_frustrated":
                # User got frustrated, we should have exhausted earlier
                config["exhaustion_threshold"] = max(3, config["exhaustion_threshold"] - 1)
                logger.info(
                    "[adaptive_config] Lowered exhaustion_threshold to %d for %s",
                    config["exhaustion_threshold"],
                    workspace_id
                )
            elif outcome == "premature_exhaustion":
                # We exhausted too early, command actually would have worked
                config["exhaustion_threshold"] = min(10, config["exhaustion_threshold"] + 1)
        
        elif threshold_name == "ask_help_threshold":
            if outcome == "help_needed_earlier":
                config["ask_help_threshold"] = max(2, config["ask_help_threshold"] - 1)
            elif outcome == "help_asked_too_soon":
                config["ask_help_threshold"] = min(8, config["ask_help_threshold"] + 1)
        
        elif threshold_name == "semantic_similarity_threshold":
            if outcome == "false_positive":
                # Blocked a command that wasn't actually a loop
                config["semantic_similarity_threshold"] = min(0.95, config["semantic_similarity_threshold"] + 0.05)
            elif outcome == "false_negative":
                # Missed a semantic loop
                config["semantic_similarity_threshold"] = max(0.70, config["semantic_similarity_threshold"] - 0.05)
        
        # Save updated config
        await collection.update_one(
            {"workspace_id": workspace_id},
            {
                "$set": {
                    **config,
                    "last_updated": datetime.now(timezone.utc).isoformat()
                }
            },
            upsert=True
        )
        
    except Exception as e:
        logger.error("[adaptive_config] Error updating: %s", e)


async def record_outcome_for_learning(
    workspace_id: str,
    event_type: str,
    details: dict
) -> None:
    """
    Record an outcome that can be used for future learning.
    
    Events:
    - command_succeeded_after_N_failures
    - command_exhausted_at_N_failures
    - user_asked_for_help_at_N_failures
    - semantic_loop_detected_with_similarity_X
    - etc.
    """
    if not db:
        return
    
    try:
        collection = db["learning_events"]
        
        await collection.insert_one({
            "workspace_id": workspace_id,
            "event_type": event_type,
            "details": details,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        # Analyze patterns and adjust thresholds
        await _analyze_and_adjust(workspace_id, event_type, details)
        
    except Exception as e:
        logger.error("[adaptive_config] Error recording outcome: %s", e)


async def _analyze_and_adjust(workspace_id: str, event_type: str, details: dict) -> None:
    """Internal: Analyze patterns and adjust thresholds."""
    
    # User explicitly asked for help before we offered
    if event_type == "user_asked_help" and "failure_count" in details:
        config = await load_adaptive_config(workspace_id)
        if details["failure_count"] < config["ask_help_threshold"]:
            await update_threshold_based_on_outcome(
                workspace_id,
                "ask_help_threshold",
                "help_needed_earlier"
            )
    
    # Command eventually succeeded after we marked it exhausted
    if event_type == "command_succeeded" and details.get("was_exhausted"):
        await update_threshold_based_on_outcome(
            workspace_id,
            "exhaustion_threshold",
            "premature_exhaustion"
        )
    
    # User said "stop" or "give up" - they're frustrated
    if event_type == "user_frustrated":
        await update_threshold_based_on_outcome(
            workspace_id,
            "exhaustion_threshold",
            "user_frustrated"
        )


async def get_config_summary(workspace_id: str) -> dict:
    """Get a summary of the adaptive config for this workspace."""
    config = await load_adaptive_config(workspace_id)
    
    # Compare to defaults
    changes = {}
    for key, value in config.items():
        default_value = DEFAULT_THRESHOLDS[key]
        if value != default_value:
            changes[key] = {
                "current": value,
                "default": default_value,
                "adjusted": value != default_value
            }
    
    return {
        "workspace_id": workspace_id,
        "thresholds": config,
        "changes_from_default": changes,
        "learning_enabled": True
    }
