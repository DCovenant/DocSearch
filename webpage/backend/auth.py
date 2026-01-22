# OAuth authentication module for Microsoft Azure AD
from fastapi import HTTPException, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import httpx
import jwt
from jwt import PyJWKClient
import os
from datetime import datetime, timezone
from typing import Optional

# Azure AD configuration - set these in environment variables
TENANT_ID = os.getenv("AZURE_TENANT_ID", "")  # Your Azure tenant ID
CLIENT_ID = os.getenv("AZURE_CLIENT_ID", "")  # App registration client ID
CLIENT_SECRET = os.getenv("AZURE_CLIENT_SECRET", "")  # App registration secret
REDIRECT_URI = os.getenv("OAUTH_REDIRECT_URI", "http://localhost:3000/auth/callback")

# Microsoft OAuth endpoints
AUTHORITY = f"https://login.microsoftonline.com/{TENANT_ID}"
AUTH_URL = f"{AUTHORITY}/oauth2/v2.0/authorize"
TOKEN_URL = f"{AUTHORITY}/oauth2/v2.0/token"
JWKS_URL = f"https://login.microsoftonline.com/{TENANT_ID}/discovery/v2.0/keys"

# Session storage (in-memory for simplicity - use Redis in production)
sessions: dict[str, dict] = {}

security = HTTPBearer(auto_error=False)

def get_auth_url(state: str) -> str:
    """Generate Microsoft OAuth authorization URL"""
    params = {
        "client_id": CLIENT_ID,
        "response_type": "code",
        "redirect_uri": REDIRECT_URI,
        "scope": "openid email profile",
        "state": state,
        "response_mode": "query"
    }
    query = "&".join(f"{k}={v}" for k, v in params.items())
    return f"{AUTH_URL}?{query}"

async def exchange_code_for_token(code: str) -> dict:
    """Exchange authorization code for tokens"""
    async with httpx.AsyncClient() as client:
        response = await client.post(TOKEN_URL, data={
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
            "code": code,
            "redirect_uri": REDIRECT_URI,
            "grant_type": "authorization_code"
        })
        if response.status_code != 200:
            print(f"Token exchange failed: {response.text}")
            raise HTTPException(400, f"Token exchange failed: {response.json().get('error_description', response.text)}")
        return response.json()

def verify_token(id_token: str) -> dict:
    """Verify and decode Microsoft ID token"""
    jwks_client = PyJWKClient(JWKS_URL)
    signing_key = jwks_client.get_signing_key_from_jwt(id_token)

    return jwt.decode(
        id_token,
        signing_key.key,
        algorithms=["RS256"],
        audience=CLIENT_ID,
        options={"verify_exp": True}
    )

def create_session(user_info: dict) -> str:
    """Create session and return session ID"""
    import secrets
    session_id = secrets.token_urlsafe(32)
    sessions[session_id] = {
        "email": user_info.get("email") or user_info.get("preferred_username"),
        "name": user_info.get("name"),
        "created": datetime.now(timezone.utc).isoformat(),
        # "expires": (datetime.utcnow() + timedelta(hours=8)).isoformat()  # Disabled for now
    }
    return session_id

async def get_current_user(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> Optional[dict]:
    """Dependency to get current user from session token"""
    if not credentials:
        return None

    session_id = credentials.credentials
    session = sessions.get(session_id)

    if not session:
        return None

    # Expiry check disabled for now
    # if datetime.fromisoformat(session["expires"]) < datetime.utcnow():
    #     del sessions[session_id]
    #     return None

    return session

def require_auth(user: Optional[dict] = Depends(get_current_user)) -> dict:
    """Dependency that requires authentication"""
    if not user:
        raise HTTPException(401, "Authentication required")
    return user
