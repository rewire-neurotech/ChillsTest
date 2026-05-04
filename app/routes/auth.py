"""
Authentication routes for Jolter.

Supports:
  - Email / password registration and login
  - Google OAuth (frontend sends Google ID token, backend verifies)
  - JWT-based session tokens
"""

from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from sqlalchemy.orm import Session

from jose import JWTError, jwt
from passlib.context import CryptContext

from app.core.config import cfg
from app.db import get_db
from app.models import User

r = APIRouter(prefix="/api/auth", tags=["auth"])

# --- Password hashing ---
pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    return pwd_ctx.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_ctx.verify(plain, hashed)


# --- JWT helpers ---

def create_token(user_id: int, email: str) -> str:
    """Create a JWT access token."""
    payload = {
        "sub": str(user_id),
        "email": email,
        "exp": datetime.now(timezone.utc) + timedelta(hours=cfg.JWT_EXPIRATION_HOURS),
    }
    return jwt.encode(payload, cfg.JWT_SECRET, algorithm=cfg.JWT_ALGORITHM)


def decode_token(token: str) -> Optional[dict]:
    """Decode and verify a JWT. Returns payload dict or None."""
    try:
        payload = jwt.decode(token, cfg.JWT_SECRET, algorithms=[cfg.JWT_ALGORITHM])
        return payload
    except JWTError:
        return None


# --- Auth dependencies for use by other routes ---

_bearer = HTTPBearer(auto_error=False)


def get_current_user_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
    db: Session = Depends(get_db),
) -> Optional[User]:
    """
    Soft auth dependency.  Returns the User if a valid Bearer token is
    present, or None otherwise.  Use for endpoints that work for both
    logged-in and anonymous users.
    """
    if not credentials:
        return None
    payload = decode_token(credentials.credentials)
    if not payload:
        return None
    user_id = payload.get("sub")
    if not user_id:
        return None
    return db.query(User).filter(User.id == int(user_id)).first()


def get_current_user_required(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
    db: Session = Depends(get_db),
) -> User:
    """
    Strict auth dependency.  Returns the User or raises 401.
    Use for endpoints that require a logged-in user.
    """
    if not credentials:
        raise HTTPException(status_code=401, detail="Not authenticated")
    payload = decode_token(credentials.credentials)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token payload")
    user = db.query(User).filter(User.id == int(user_id)).first()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user


# --- Request / Response schemas ---

class RegisterRequest(BaseModel):
    email: str
    password: str
    name: Optional[str] = None


class LoginRequest(BaseModel):
    email: str
    password: str


class GoogleLoginRequest(BaseModel):
    credential: str  # Google ID token from frontend


class AuthResponse(BaseModel):
    token: str
    user_id: int
    email: str
    name: Optional[str] = None


class UserResponse(BaseModel):
    user_id: int
    email: str
    name: Optional[str] = None
    auth_provider: str


# --- Endpoints ---

@r.post("/register", response_model=AuthResponse)
def register(req: RegisterRequest, db: Session = Depends(get_db)):
    """Register a new user with email and password."""
    if not req.email or not req.email.strip():
        raise HTTPException(status_code=400, detail="Email is required")
    if not req.password or len(req.password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters")

    email = req.email.strip().lower()

    existing = db.query(User).filter(User.email == email).first()
    if existing:
        raise HTTPException(status_code=409, detail="Email already registered")

    user = User(
        email=email,
        password_hash=hash_password(req.password),
        name=req.name.strip() if req.name else None,
        auth_provider="local",
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    token = create_token(user.id, user.email)
    return AuthResponse(token=token, user_id=user.id, email=user.email, name=user.name)


@r.post("/login", response_model=AuthResponse)
def login(req: LoginRequest, db: Session = Depends(get_db)):
    """Login with email and password."""
    email = (req.email or "").strip().lower()
    user = db.query(User).filter(User.email == email).first()

    if not user or not user.password_hash:
        raise HTTPException(status_code=401, detail="Invalid email or password")

    if not verify_password(req.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    token = create_token(user.id, user.email)
    return AuthResponse(token=token, user_id=user.id, email=user.email, name=user.name)


@r.post("/google", response_model=AuthResponse)
def google_login(req: GoogleLoginRequest, db: Session = Depends(get_db)):
    """
    Login or register via Google.

    The frontend obtains a Google ID token using Google Identity Services
    and sends it here.  We verify it with Google and create / find the user.
    """
    import requests as http_requests

    # Verify the Google ID token via Google's tokeninfo endpoint
    try:
        resp = http_requests.get(
            "https://oauth2.googleapis.com/tokeninfo",
            params={"id_token": req.credential},
            timeout=10,
        )
    except Exception:
        raise HTTPException(status_code=502, detail="Could not reach Google for token verification")

    if resp.status_code != 200:
        raise HTTPException(status_code=401, detail="Invalid Google token")

    info = resp.json()

    # Verify the audience matches our client ID (if configured)
    if cfg.GOOGLE_CLIENT_ID and info.get("aud") != cfg.GOOGLE_CLIENT_ID:
        raise HTTPException(status_code=401, detail="Google token audience mismatch")

    google_email = info.get("email", "").strip().lower()
    google_id = info.get("sub", "")
    google_name = info.get("name", "")

    if not google_email or not google_id:
        raise HTTPException(status_code=401, detail="Could not extract email from Google token")

    # Find existing user by google_id or email
    user = db.query(User).filter(User.google_id == google_id).first()
    if not user:
        user = db.query(User).filter(User.email == google_email).first()

    if user:
        # Update google_id if not set (user registered with email, now linking Google)
        if not user.google_id:
            user.google_id = google_id
            user.auth_provider = "google"
            db.commit()
    else:
        # Create new user
        user = User(
            email=google_email,
            name=google_name or None,
            auth_provider="google",
            google_id=google_id,
        )
        db.add(user)
        db.commit()
        db.refresh(user)

    token = create_token(user.id, user.email)
    return AuthResponse(token=token, user_id=user.id, email=user.email, name=user.name)


@r.get("/me", response_model=UserResponse)
def get_me(user: User = Depends(get_current_user_required)):
    """Get the current logged-in user info."""
    return UserResponse(
        user_id=user.id,
        email=user.email,
        name=user.name,
        auth_provider=user.auth_provider,
    )


@r.get("/config")
def get_auth_config():
    """
    Public endpoint. Returns the Google Client ID so the frontend
    can initialize Google Identity Services.
    """
    return {"google_client_id": cfg.GOOGLE_CLIENT_ID or ""}
