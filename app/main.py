import os
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.core.config import cfg

# --- DB setup ---
try:
    from app.db import engine
    from app.models import Base
except Exception:
    engine = None
    Base = None

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


# --- Serve frontend ---
FRONTEND_PATH = Path(__file__).resolve().parent.parent / "frontend" / "rewire.html"

@app.get("/")
def serve_frontend():
    if FRONTEND_PATH.exists():
        return FileResponse(str(FRONTEND_PATH), media_type="text/html")
    return {"error": "frontend not found"}
