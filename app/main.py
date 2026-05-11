import os
from pathlib import Path
from fastapi import FastAPI, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.core.config import cfg

# --- DB setup ---
try:
    from app.db import engine, SessionLocal, get_db
    from app.models import Base, PromoCode, PushSubscription
except Exception:
    engine = None
    Base = None
    SessionLocal = None
    PromoCode = None
    PushSubscription = None
    get_db = None


def _seed_promo_codes():
    """Insert default promo codes if missing. Idempotent -- safe to run every boot."""
    if SessionLocal is None or PromoCode is None:
        return
    defaults = ["CHILLS50", "JOLTER", "FRIEND"]
    db = SessionLocal()
    try:
        for code in defaults:
            existing = db.query(PromoCode).filter(PromoCode.code == code).first()
            if not existing:
                db.add(PromoCode(code=code, max_uses=0, is_active=True))
        db.commit()
    except Exception as e:
        print(f"[startup] Promo code seed error: {e}")
        db.rollback()
    finally:
        db.close()

app = FastAPI(
    title="Jolter Backend",
    version="0.2.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=cfg.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Security headers ---
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response

# Create output directory and mount for static audio files
os.makedirs(cfg.out_dir_path, exist_ok=True)
app.mount("/public", StaticFiles(directory=str(cfg.out_dir_path)), name="public")

# Mount assets folder
assets_path = Path(__file__).parent / "assets"
if assets_path.exists():
    app.mount("/assets", StaticFiles(directory=str(assets_path)), name="assets")

# Create DB tables
if Base is not None and engine is not None:
    try:
        Base.metadata.create_all(bind=engine)
        _seed_promo_codes()
    except Exception as e:
        print(f"[startup] DB table creation error: {e}")


# --- Health check ---
@app.get("/api/health")
def health():
    return {"status": "ok", "service": "jolter"}


# --- Routes ---
from app.routes.demo import r as demo_r
app.include_router(demo_r)

from app.routes.auth import r as auth_r
app.include_router(auth_r)

from app.routes.notes import r as notes_r
app.include_router(notes_r)

from app.routes.subscription import r as sub_r
app.include_router(sub_r)


# --- Service worker for push notifications ---
_SW_JS = """
self.addEventListener('install', e => self.skipWaiting());
self.addEventListener('activate', e => e.waitUntil(self.clients.claim()));
self.addEventListener('push', e => {
  let data = {title: 'ReWire', body: 'Your Jolt is ready'};
  try { data = e.data.json(); } catch(_) {}
  e.waitUntil(self.registration.showNotification(data.title, {
    body: data.body,
    icon: '/assets/jolter-logo.jpeg',
  }));
});
self.addEventListener('notificationclick', e => {
  e.notification.close();
  e.waitUntil(clients.matchAll({type:'window'}).then(cs => {
    for (const c of cs) { if (c.url.includes(self.location.origin)) return c.focus(); }
    return clients.openWindow('/');
  }));
});
""".strip()

@app.get("/sw.js")
def serve_sw():
    return Response(content=_SW_JS, media_type="application/javascript",
                    headers={"Cache-Control": "no-cache", "Service-Worker-Allowed": "/"})


# --- Push notification endpoints ---
class PushSubscribeRequest(BaseModel):
    endpoint: str
    p256dh: str
    auth: str

@app.get("/api/push/vapid-key")
def get_vapid_key():
    return {"vapid_public_key": cfg.VAPID_PUBLIC_KEY or ""}

@app.post("/api/push/subscribe")
def push_subscribe(req: PushSubscribeRequest, request: Request, db: Session = Depends(get_db)):
    from app.routes.auth import get_current_user_required
    user = get_current_user_required(request=request, db=db)
    # Upsert: avoid duplicate subscriptions for same endpoint
    existing = db.query(PushSubscription).filter(
        PushSubscription.user_id == user.id,
        PushSubscription.endpoint == req.endpoint,
    ).first()
    if existing:
        existing.p256dh = req.p256dh
        existing.auth = req.auth
    else:
        db.add(PushSubscription(
            user_id=user.id,
            endpoint=req.endpoint,
            p256dh=req.p256dh,
            auth=req.auth,
        ))
    db.commit()
    return {"status": "ok"}

@app.post("/api/push/unsubscribe")
def push_unsubscribe(req: PushSubscribeRequest, request: Request, db: Session = Depends(get_db)):
    from app.routes.auth import get_current_user_required
    user = get_current_user_required(request=request, db=db)
    sub = db.query(PushSubscription).filter(
        PushSubscription.user_id == user.id,
        PushSubscription.endpoint == req.endpoint,
    ).first()
    if sub:
        db.delete(sub)
        db.commit()
    return {"status": "ok"}


# --- Serve frontend ---
FRONTEND_PATH = Path(__file__).resolve().parent.parent / "frontend" / "rewire.html"

@app.get("/")
def serve_frontend():
    if FRONTEND_PATH.exists():
        return FileResponse(str(FRONTEND_PATH), media_type="text/html")
    return {"error": "frontend not found"}
