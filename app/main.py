from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.chat import router as chat_router
from app.db.session import init_db
from app.observability.logging import setup_logging
from app.observability.metrics import metrics_payload

STATIC_DIR = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    init_db()
    print("[Lefty] ready")
    yield
    print("[Lefty] bye")


app = FastAPI(
    title="Lefty",
    version="0.2.0",
    description="Local shopping agent — catalog search + budget compare",
    lifespan=lifespan,
)

app.include_router(chat_router, prefix="/api/v1")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def root():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.get("/metrics")
async def metrics():
    body, content_type = metrics_payload()
    return Response(content=body, media_type=content_type)
