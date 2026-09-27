"""Authentication helpers for Acme Shop.

JWT access + refresh tokens; passwords hashed with bcrypt. The API gateway
validates the access token on every request; this module issues and decodes them.
Google social login (OAuth2/OIDC) issues the same token pair via /auth/google.
"""

import time

import bcrypt
import jwt  # PyJWT

JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_TTL_SECONDS = 900        # 15 minutes
REFRESH_TOKEN_TTL_SECONDS = 604800    # 7 days


def hash_password(password: str) -> bytes:
    """Hash a password with bcrypt. Never store plain-text passwords."""
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt())


def verify_password(password: str, hashed: bytes) -> bool:
    """Check a login password against its bcrypt hash."""
    return bcrypt.checkpw(password.encode(), hashed)


def issue_access_token(user_id: str, role: str, secret: str) -> str:
    """Short-lived JWT sent on every API call."""
    now = int(time.time())
    payload = {
        "sub": user_id, "role": role, "type": "access",
        "iat": now, "exp": now + ACCESS_TOKEN_TTL_SECONDS,
    }
    return jwt.encode(payload, secret, algorithm=JWT_ALGORITHM)


def issue_refresh_token(user_id: str, secret: str) -> str:
    """Long-lived JWT exchanged at /auth/refresh for a new access token."""
    now = int(time.time())
    payload = {
        "sub": user_id, "type": "refresh",
        "iat": now, "exp": now + REFRESH_TOKEN_TTL_SECONDS,
    }
    return jwt.encode(payload, secret, algorithm=JWT_ALGORITHM)


def decode_token(token: str, secret: str) -> dict:
    """Validate and decode a JWT. Raises jwt.InvalidTokenError if bad/expired."""
    return jwt.decode(token, secret, algorithms=[JWT_ALGORITHM])
