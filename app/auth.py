"""
Authentication — GitHub/Google OAuth + JWT tokens.

Flow:
  1. User clicks "Sign in" in forge-ide
  2. Browser opens /auth/github (or /auth/google)
  3. User authorizes on GitHub/Google
  4. Callback creates user in DB, returns JWT
  5. forge-ide stores JWT, sends it with every request
  6. Middleware validates JWT on protected endpoints
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Optional

import httpx
import jwt
from fastapi import HTTPException, Request, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────

JWT_SECRET = os.getenv("JWT_SECRET", "change-me-in-production")
JWT_ALGORITHM = "HS256"
JWT_EXPIRY_DAYS = 30

GITHUB_CLIENT_ID = os.getenv("GITHUB_CLIENT_ID", "")
GITHUB_CLIENT_SECRET = os.getenv("GITHUB_CLIENT_SECRET", "")

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "")

APP_URL = os.getenv("APP_URL", "https://forge-search-production.up.railway.app")

# ── JWT ───────────────────────────────────────────────────────────

def create_token(user_id: str, email: str, name: str) -> str:
    payload = {
        "sub": user_id,
        "email": email,
        "name": name,
        "iat": int(time.time()),
        "exp": int((datetime.utcnow() + timedelta(days=JWT_EXPIRY_DAYS)).timestamp()),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


# ── Auth dependency ───────────────────────────────────────────────

security = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> Optional[dict]:
    """Returns user dict from JWT, or None if no token."""
    if credentials is None:
        return None
    return decode_token(credentials.credentials)


async def require_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> dict:
    """Returns user dict from JWT, or raises 401."""
    if credentials is None:
        raise HTTPException(status_code=401, detail="Authentication required")
    return decode_token(credentials.credentials)


# ── GitHub OAuth ──────────────────────────────────────────────────

def github_auth_url(state: str = "") -> str:
    return (
        f"https://github.com/login/oauth/authorize"
        f"?client_id={GITHUB_CLIENT_ID}"
        f"&redirect_uri={APP_URL}/auth/github/callback"
        f"&scope=user:email"
        f"&state={state}"
    )


async def github_exchange_code(code: str) -> dict:
    """Exchange GitHub OAuth code for user info."""
    async with httpx.AsyncClient() as client:
        # Get access token
        resp = await client.post(
            "https://github.com/login/oauth/access_token",
            json={
                "client_id": GITHUB_CLIENT_ID,
                "client_secret": GITHUB_CLIENT_SECRET,
                "code": code,
            },
            headers={"Accept": "application/json"},
        )
        token_data = resp.json()
        access_token = token_data.get("access_token")
        if not access_token:
            raise HTTPException(status_code=400, detail=f"GitHub OAuth failed: {token_data}")

        # Get user info
        resp = await client.get(
            "https://api.github.com/user",
            headers={"Authorization": f"Bearer {access_token}"},
        )
        user_data = resp.json()

        # Get email (might be private)
        email = user_data.get("email")
        if not email:
            resp = await client.get(
                "https://api.github.com/user/emails",
                headers={"Authorization": f"Bearer {access_token}"},
            )
            emails = resp.json()
            for e in emails:
                if e.get("primary"):
                    email = e["email"]
                    break

        return {
            "provider": "github",
            "provider_id": str(user_data["id"]),
            "email": email or "",
            "name": user_data.get("name") or user_data.get("login", ""),
            "avatar": user_data.get("avatar_url", ""),
            "username": user_data.get("login", ""),
        }


# ── Google OAuth ──────────────────────────────────────────────────

def google_auth_url(state: str = "") -> str:
    return (
        f"https://accounts.google.com/o/oauth2/v2/auth"
        f"?client_id={GOOGLE_CLIENT_ID}"
        f"&redirect_uri={APP_URL}/auth/google/callback"
        f"&response_type=code"
        f"&scope=email+profile"
        f"&state={state}"
    )


async def google_exchange_code(code: str) -> dict:
    """Exchange Google OAuth code for user info."""
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "https://oauth2.googleapis.com/token",
            data={
                "client_id": GOOGLE_CLIENT_ID,
                "client_secret": GOOGLE_CLIENT_SECRET,
                "code": code,
                "grant_type": "authorization_code",
                "redirect_uri": f"{APP_URL}/auth/google/callback",
            },
        )
        token_data = resp.json()
        access_token = token_data.get("access_token")
        if not access_token:
            raise HTTPException(status_code=400, detail=f"Google OAuth failed: {token_data}")

        resp = await client.get(
            "https://www.googleapis.com/oauth2/v2/userinfo",
            headers={"Authorization": f"Bearer {access_token}"},
        )
        user_data = resp.json()

        return {
            "provider": "google",
            "provider_id": user_data["id"],
            "email": user_data.get("email", ""),
            "name": user_data.get("name", ""),
            "avatar": user_data.get("picture", ""),
            "username": user_data.get("email", "").split("@")[0],
        }


# ── User DB operations ───────────────────────────────────────────

def ensure_user_table(conn):
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            email TEXT UNIQUE,
            name TEXT,
            avatar TEXT DEFAULT '',
            provider TEXT,
            provider_id TEXT,
            username TEXT DEFAULT '',
            created_at TIMESTAMP DEFAULT NOW(),
            last_login TIMESTAMP DEFAULT NOW()
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)")
    conn.commit()


def upsert_user(conn, user_info: dict) -> str:
    """Create or update user, return user ID."""
    user_id = hashlib.sha256(
        f"{user_info['provider']}:{user_info['provider_id']}".encode()
    ).hexdigest()[:16]

    cur = conn.cursor()
    cur.execute("""
        INSERT INTO users (id, email, name, avatar, provider, provider_id, username)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (id) DO UPDATE SET
            name = %s, avatar = %s, last_login = NOW()
        RETURNING id
    """, (user_id, user_info["email"], user_info["name"], user_info["avatar"],
          user_info["provider"], user_info["provider_id"], user_info["username"],
          user_info["name"], user_info["avatar"]))
    conn.commit()
    return user_id
