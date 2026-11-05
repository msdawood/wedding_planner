"""JWT authentication for the gateway (HS256 via PyJWT)."""
import hmac
import time
import uuid
from dataclasses import dataclass

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

import settings

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/token")


@dataclass(frozen=True)
class User:
    id: str


def assert_configured() -> None:
    """Fail fast at startup instead of issuing tokens signed with an empty/weak secret."""
    if len(settings.JWT_SECRET) < 32:
        raise RuntimeError("JWT_SECRET must be set to at least 32 characters")
    if not settings.API_USERNAME or not settings.API_PASSWORD:
        raise RuntimeError("API_USERNAME and API_PASSWORD must be set")


def authenticate(username: str, password: str) -> bool:
    # Constant-time comparison; evaluate both so timing doesn't reveal which was wrong.
    user_ok = hmac.compare_digest(username.encode(), settings.API_USERNAME.encode())
    pass_ok = hmac.compare_digest(password.encode(), settings.API_PASSWORD.encode())
    return user_ok and pass_ok


def create_token(user_id: str) -> tuple[str, int]:
    now = int(time.time())
    expires = settings.JWT_EXPIRES_MIN * 60
    payload = {"sub": user_id, "iss": settings.JWT_ISSUER, "iat": now, "exp": now + expires, "jti": uuid.uuid4().hex}
    return jwt.encode(payload, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM), expires


def verify_jwt_token(token: str) -> User | None:
    try:
        claims = jwt.decode(
            token, settings.JWT_SECRET, algorithms=[settings.JWT_ALGORITHM],
            issuer=settings.JWT_ISSUER, options={"require": ["exp", "iat", "sub", "iss"]})
    except jwt.PyJWTError:
        return None
    return User(id=str(claims["sub"]))


async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    """Verifies the bearer JWT. Fails with 401 if invalid or expired."""
    user = verify_jwt_token(token)
    if user is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or expired token",
                            headers={"WWW-Authenticate": "Bearer"})
    return user
