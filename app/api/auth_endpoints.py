"""Authentication endpoints - GitHub and Google OAuth."""

import time
import logging
from fastapi import APIRouter, Depends
from fastapi.responses import RedirectResponse
from ..utils import auth

logger = logging.getLogger(__name__)

router = APIRouter()

# Temporary storage for OAuth polling (session_id -> token)
_pending_auth = {}


@router.get("/github")
async def auth_github(state: str = ""):
    """Start GitHub OAuth flow — redirects to GitHub."""
    logger.info("[auth] GitHub OAuth start, state=%r", state)
    return RedirectResponse(auth.github_auth_url(state=state))


@router.get("/github/callback")
async def auth_github_callback(code: str, state: str = ""):
    """GitHub OAuth callback — creates user, returns JWT."""
    logger.info("[auth] GitHub callback received, state=%r", state)
    user_info = await auth.github_exchange_code(code)
    user_id = await auth.upsert_user(user_info)
    token = auth.create_token(user_id, user_info["email"], user_info["name"])
    
    # Polling-based auth: store token for IDE to poll
    if state.startswith("poll-"):
        session_id = state[5:]  # Remove "poll-" prefix
        _pending_auth[session_id] = {
            "token": token,
            "user_info": user_info,
            "expires": time.time() + 300,  # 5 minute expiry
        }
        # Clean up old entries
        now = time.time()
        for sid in list(_pending_auth.keys()):
            if _pending_auth[sid]["expires"] < now:
                del _pending_auth[sid]
        return auth.success_page(user_info, token)
    
    # Legacy: custom URL scheme (may not work on all platforms)
    if state.startswith("forge-ide"):
        return RedirectResponse(f"forge-ide://auth?token={token}")
    
    # For browser: show success page
    return auth.success_page(user_info, token)


@router.get("/google")
async def auth_google(state: str = ""):
    """Start Google OAuth flow — redirects to Google."""
    return RedirectResponse(auth.google_auth_url(state=state))


@router.get("/google/callback")
async def auth_google_callback(code: str, state: str = ""):
    """Google OAuth callback — creates user, returns JWT."""
    logger.info("[auth] Google callback received, state=%r", state)
    user_info = await auth.google_exchange_code(code)
    user_id = await auth.upsert_user(user_info)
    token = auth.create_token(user_id, user_info["email"], user_info["name"])
    
    # Polling-based auth: store token for IDE to poll
    if state.startswith("poll-"):
        session_id = state[5:]
        _pending_auth[session_id] = {
            "token": token,
            "user_info": user_info,
            "expires": time.time() + 300,
        }
        now = time.time()
        for sid in list(_pending_auth.keys()):
            if _pending_auth[sid]["expires"] < now:
                del _pending_auth[sid]
        return auth.success_page(user_info, token)
    
    if state.startswith("forge-ide"):
        return RedirectResponse(f"forge-ide://auth?token={token}")
    return auth.success_page(user_info, token)


@router.get("/me")
async def auth_me(user: dict = Depends(auth.require_user)):
    """Get current authenticated user."""
    return {"user_id": user["sub"], "email": user["email"], "name": user["name"]}


@router.get("/poll/{session_id}")
async def auth_poll(session_id: str):
    """
    Poll for OAuth token.
    
    IDE opens browser with state=poll-{session_id}, then polls this endpoint
    until the token is available (user completes OAuth in browser).
    
    Returns:
        - {"status": "pending"} if auth not complete
        - {"status": "success", "token": "...", "email": "...", "name": "..."} on success
        - {"status": "expired"} if session expired or not found
    """
    if session_id not in _pending_auth:
        # Only log every ~12th poll (once per minute at 5s interval) to avoid spam
        return {"status": "pending"}
    
    entry = _pending_auth[session_id]
    
    # Check expiry
    if entry["expires"] < time.time():
        del _pending_auth[session_id]
        logger.info("[auth] Poll session %s expired", session_id)
        return {"status": "expired"}
    
    # Token is ready - remove from store and return
    logger.info("[auth] Poll session %s SUCCESS, delivering token for %s", 
                session_id, entry["user_info"].get("email", "?"))
    del _pending_auth[session_id]
    return {
        "status": "success",
        "token": entry["token"],
        "email": entry["user_info"].get("email", ""),
        "name": entry["user_info"].get("name", ""),
    }
