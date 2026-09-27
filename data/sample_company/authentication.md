# Authentication — Acme Shop

Authentication is **JWT-based**, with optional **Google** social login. The auth
logic lives in `code/auth.py`; the API gateway validates tokens on every request.

## How it works
- **Login** (`POST /auth/login`) checks the email + password — passwords are
  hashed with **bcrypt**, never stored in plain text — and returns two tokens:
  - an **access token** (JWT, HS256), short-lived: **15 minutes**
  - a **refresh token**, long-lived: **7 days**, exchanged at `POST /auth/refresh`
    for a fresh access token
- **Social login**: `GET /auth/google` starts the **Google OAuth2 / OIDC** flow;
  the callback `GET /auth/google/callback` issues the same JWT pair.
- **Logout** (`POST /auth/logout`) revokes the refresh token.
- The **API gateway** validates the access token's signature and expiry on every
  request, then forwards the user id and role (`customer` / `admin`) downstream.

## Configuration (environment variables)
| Variable | Purpose |
| --- | --- |
| `JWT_SECRET` | signs and verifies JWTs |
| `ACCESS_TOKEN_TTL_SECONDS` | access-token lifetime (default 900 = 15 min) |
| `REFRESH_TOKEN_TTL_SECONDS` | refresh-token lifetime (default 604800 = 7 days) |
| `GOOGLE_OAUTH_CLIENT_ID`, `GOOGLE_OAUTH_CLIENT_SECRET` | Google social login |
