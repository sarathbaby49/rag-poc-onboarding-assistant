"""Authentication helpers for payments-service.

Users are authenticated with JWT (JSON Web Tokens). The API gateway validates
the token on the way in; this module is what issues and decodes them.
"""

import time

import jwt  # PyJWT

# In real life JWT_SECRET comes from the environment, never hard-coded.
JWT_ALGORITHM = "HS256"
TOKEN_TTL_SECONDS = 3600


def issue_token(user_id: str, secret: str) -> str:
    """Create a signed JWT for a user. Expires after TOKEN_TTL_SECONDS."""
    payload = {
        "sub": user_id,
        "iat": int(time.time()),
        "exp": int(time.time()) + TOKEN_TTL_SECONDS,
    }
    return jwt.encode(payload, secret, algorithm=JWT_ALGORITHM)


def get_user_token(user_id: str, secret: str) -> str:
    """Convenience wrapper used by the login endpoint."""
    return issue_token(user_id, secret)


def decode_token(token: str, secret: str) -> dict:
    """Validate and decode a JWT. Raises jwt.InvalidTokenError if bad/expired."""
    return jwt.decode(token, secret, algorithms=[JWT_ALGORITHM])
