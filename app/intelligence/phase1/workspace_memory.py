"""
Persistent Workspace Memory - Track failures, learnings, and exhausted approaches.

This module provides cross-trace memory so the agent can learn from past failures
and avoid repeating the same mistakes across multiple conversation traces.
"""

from typing import TypedDict, Any
from datetime import datetime, timezone
import logging
from motor.motor_asyncio import AsyncIOMotorClient
import os

logger = logging.getLogger(__name__)

# MongoDB connection
MONGODB_URI = os.getenv("MONGODB_URI", "")
mongo_client = AsyncIOMotorClient(MONGODB_URI) if MONGODB_URI else None
db = mongo_client["forge_workspace_memory"] if mongo_client else None


class FailureRecord(TypedDict):
    """Record of a failed command/approach."""
    command: str
    attempts: int
    first_failure: str  # ISO timestamp
    last_failure: str   # ISO timestamp
    error_signature: str  # First 200 chars of error
    status: str  # "active", "exhausted"
    alternatives_tried: list[str]


class WorkspaceMemory(TypedDict):
    """Persistent memory for a workspace."""
    workspace_id: str
    failed_commands: dict[str, FailureRecord]
    learned_facts: dict[str, Any]
    exhausted_approaches: list[str]
    last_updated: str


async def load_workspace_memory(workspace_id: str) -> WorkspaceMemory:
    """Load workspace memory from MongoDB."""
    if not db:
        logger.warning("[workspace_memory] MongoDB not configured, using empty memory")
        return _empty_memory(workspace_id)
    
    try:
        collection = db["workspace_memory"]
        doc = await collection.find_one({"workspace_id": workspace_id})
        
        if doc:
            # Remove MongoDB _id field
            doc.pop("_id", None)
            logger.info("[workspace_memory] Loaded memory for %s: %d failed commands, %d facts",
                       workspace_id, len(doc.get("failed_commands", {})), len(doc.get("learned_facts", {})))
            return doc
        else:
            logger.info("[workspace_memory] No existing memory for %s, creating new", workspace_id)
            return _empty_memory(workspace_id)
    except Exception as e:
        logger.error("[workspace_memory] Error loading memory for %s: %s", workspace_id, e)
        return _empty_memory(workspace_id)


async def save_workspace_memory(memory: WorkspaceMemory) -> None:
    """Save workspace memory to MongoDB."""
    if not db:
        logger.warning("[workspace_memory] MongoDB not configured, cannot save")
        return
    
    try:
        memory["last_updated"] = datetime.now(timezone.utc).isoformat()
        collection = db["workspace_memory"]
        
        await collection.update_one(
            {"workspace_id": memory["workspace_id"]},
            {"$set": memory},
            upsert=True
        )
        logger.info("[workspace_memory] Saved memory for %s", memory["workspace_id"])
    except Exception as e:
        logger.error("[workspace_memory] Error saving memory: %s", e)


async def record_failure(
    workspace_id: str,
    command: str,
    error_message: str,
    alternatives_tried: list[str] = None
) -> WorkspaceMemory:
    """Record a command failure and update memory."""
    memory = await load_workspace_memory(workspace_id)
    
    now = datetime.now(timezone.utc).isoformat()
    
    # Normalize command (strip whitespace, lowercase for comparison)
    command_key = command.strip().lower()
    
    if command_key in memory["failed_commands"]:
        # Update existing failure record
        record = memory["failed_commands"][command_key]
        record["attempts"] += 1
        record["last_failure"] = now
        if alternatives_tried:
            record["alternatives_tried"].extend(alternatives_tried)
        
        # Mark as exhausted after 5 attempts
        if record["attempts"] >= 5:
            record["status"] = "exhausted"
            if command_key not in memory["exhausted_approaches"]:
                memory["exhausted_approaches"].append(command_key)
                logger.warning("[workspace_memory] Marked '%s' as EXHAUSTED after %d attempts",
                             command[:60], record["attempts"])
    else:
        # Create new failure record
        memory["failed_commands"][command_key] = {
            "command": command,
            "attempts": 1,
            "first_failure": now,
            "last_failure": now,
            "error_signature": error_message[:200],
            "status": "active",
            "alternatives_tried": alternatives_tried or []
        }
        logger.info("[workspace_memory] Recorded first failure for '%s'", command[:60])
    
    await save_workspace_memory(memory)
    return memory


async def record_success(workspace_id: str, command: str) -> None:
    """Record a successful command (clear failure history)."""
    memory = await load_workspace_memory(workspace_id)
    
    command_key = command.strip().lower()
    
    # Remove from failed commands if it was there
    if command_key in memory["failed_commands"]:
        del memory["failed_commands"][command_key]
        logger.info("[workspace_memory] Cleared failure record for '%s' (succeeded)", command[:60])
    
    # Remove from exhausted approaches
    if command_key in memory["exhausted_approaches"]:
        memory["exhausted_approaches"].remove(command_key)
    
    await save_workspace_memory(memory)


async def is_exhausted_approach(workspace_id: str, command: str) -> tuple[bool, FailureRecord | None]:
    """Check if a command is an exhausted approach."""
    memory = await load_workspace_memory(workspace_id)
    
    command_key = command.strip().lower()
    
    # Direct match
    if command_key in memory["exhausted_approaches"]:
        record = memory["failed_commands"].get(command_key)
        return True, record
    
    # Check for semantic similarity (same base command with different flags)
    base_command = _extract_base_command(command)
    
    for exhausted in memory["exhausted_approaches"]:
        exhausted_base = _extract_base_command(exhausted)
        
        # Same base command = likely semantic duplicate
        if base_command == exhausted_base:
            record = memory["failed_commands"].get(exhausted)
            logger.warning("[workspace_memory] Semantic duplicate detected: '%s' similar to exhausted '%s'",
                         command[:50], exhausted[:50])
            return True, record
    
    return False, None


async def get_failure_summary(workspace_id: str) -> str:
    """Get a human-readable summary of failures for injection into prompts."""
    memory = await load_workspace_memory(workspace_id)
    
    if not memory["failed_commands"]:
        return ""
    
    exhausted = [
        record for record in memory["failed_commands"].values()
        if record["status"] == "exhausted"
    ]
    
    if not exhausted:
        return ""
    
    summary = "## ⚠️ EXHAUSTED APPROACHES (DO NOT RETRY)\n\n"
    summary += "The following commands have failed 5+ times and should NOT be retried:\n\n"
    
    for record in exhausted:
        summary += f"- `{record['command']}` (failed {record['attempts']} times)\n"
        summary += f"  - Error: {record['error_signature'][:100]}\n"
        if record["alternatives_tried"]:
            summary += f"  - Tried alternatives: {', '.join(record['alternatives_tried'][:3])}\n"
        summary += "\n"
    
    summary += "**If you need to solve this problem:**\n"
    summary += "1. Use `lookup_documentation` to find the correct approach\n"
    summary += "2. Ask the user for help with specific questions\n"
    summary += "3. Skip this step if it's not critical\n"
    
    return summary


def _empty_memory(workspace_id: str) -> WorkspaceMemory:
    """Create an empty memory structure."""
    return {
        "workspace_id": workspace_id,
        "failed_commands": {},
        "learned_facts": {},
        "exhausted_approaches": [],
        "last_updated": datetime.now(timezone.utc).isoformat()
    }


def _extract_base_command(command: str) -> str:
    """Extract the base command without flags/arguments."""
    # Remove common flags and arguments
    # e.g., "npx prisma generate --config=foo" -> "npx prisma generate"
    parts = command.strip().split()
    
    # Keep only the first 2-3 words (typically the actual command)
    # Skip obvious flags
    base_parts = []
    for part in parts:
        if part.startswith('-'):
            break
        base_parts.append(part)
        if len(base_parts) >= 3:
            break
    
    return ' '.join(base_parts).lower()


async def should_ask_for_help(workspace_id: str, state: dict) -> tuple[bool, str]:
    """
    Determine if the agent should ask the user for help.
    
    Returns:
        (should_ask, reason_message)
    """
    memory = await load_workspace_memory(workspace_id)
    
    # Check for exhausted approaches
    if memory["exhausted_approaches"]:
        exhausted_list = '\n'.join(f"- {cmd}" for cmd in memory["exhausted_approaches"][:5])
        return True, (
            "I've tried multiple approaches without success. The following have been exhausted:\n\n"
            f"{exhausted_list}\n\n"
            "**Could you help me by:**\n"
            "1. Suggesting a different approach\n"
            "2. Providing more context about the environment/setup\n"
            "3. Confirming if I should skip this step and move on"
        )
    
    # Check for high failure count on current command
    recent_failures = [
        record for record in memory["failed_commands"].values()
        if record["attempts"] >= 3 and record["status"] == "active"
    ]
    
    if recent_failures:
        failing_cmds = '\n'.join(f"- {r['command']} ({r['attempts']} attempts)" for r in recent_failures[:3])
        return True, (
            "I'm encountering repeated failures:\n\n"
            f"{failing_cmds}\n\n"
            "**Could you help me understand:**\n"
            "1. Is there something specific about your environment I should know?\n"
            "2. Should I try a completely different approach?\n"
            "3. Is this step optional?"
        )
    
    return False, ""


async def clear_workspace_memory(workspace_id: str) -> None:
    """Clear all memory for a workspace (useful for testing/reset)."""
    if not db:
        return
    
    try:
        collection = db["workspace_memory"]
        await collection.delete_one({"workspace_id": workspace_id})
        logger.info("[workspace_memory] Cleared memory for %s", workspace_id)
    except Exception as e:
        logger.error("[workspace_memory] Error clearing memory: %s", e)
