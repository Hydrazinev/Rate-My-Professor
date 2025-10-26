# main.py
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterable, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import (
    FileResponse,
    JSONResponse,
    PlainTextResponse,
    RedirectResponse,
    StreamingResponse,
)
from starlette.middleware.cors import CORSMiddleware

from agent import create_agent

# ---------- Logging ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("rmp")

# ---------- App ----------
app = FastAPI(title="Rate My Professor", version="1.0.0")

# CORS (adjust origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Files / Paths ----------
BASE_DIR = Path(__file__).resolve().parent
CHAT_HTML = BASE_DIR / "chat.html"                    # legacy location
PUBLIC_CHAT_HTML = BASE_DIR / "public" / "chat.html"  # Vercel CDN location

# ---------- Agent ----------
# Initialize once; missing env vars should be reflected in /healthz
try:
    agent = create_agent()
    logger.info("Agent created successfully.")
except Exception:
    logger.exception("Failed to create agent at startup.")
    agent = None  # Allow server to start; endpoints will handle None gracefully.

# ---------- Helpers ----------
def locate_chat_html() -> Optional[Path]:
    if CHAT_HTML.exists():
        return CHAT_HTML
    if PUBLIC_CHAT_HTML.exists():
        return PUBLIC_CHAT_HTML
    return None

def stream_agent_chunks(text: str) -> Iterable[str]:
    """
    Yield chunks from agent.stream(text, chat_history=None).
    Modify this to match your agent's streaming API if needed.
    """
    if agent is None:
        yield "Agent not initialized. Check environment variables."
        return
    # Example interface; adapt to your create_agent() implementation
    for chunk in agent.stream(text, chat_history=None):
        # Ensure each yielded item is a string
        yield str(chunk)

# ---------- Routes ----------
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "version": app.version,
        "has_agent": agent is not None,
        "port": os.environ.get("PORT"),
    }

@app.get("/healthz")
async def healthz():
    # Redacted health; only presence of required envs
    env_ok = {
        "OPENAI_API_KEY": bool(os.environ.get("OPENAI_API_KEY")),
        "PINECONE_API_KEY": bool(os.environ.get("PINECONE_API_KEY")),
        "PINECONE_REGION": bool(os.environ.get("PINECONE_REGION")),
        "PINECONE_INDEX_NAME": bool(os.environ.get("PINECONE_INDEX_NAME")),
        "EMBEDDING_MODEL": bool(os.environ.get("EMBEDDING_MODEL")),
        "RMP_AUTHORIZATION": bool(os.environ.get("RMP_AUTHORIZATION")),
    }
    return {"ok": True, "env": env_ok, "has_agent": agent is not None}

@app.get("/", include_in_schema=False)
async def root():
    # Prefer the CDN-served file if present (Vercel will serve /public/** statically)
    if PUBLIC_CHAT_HTML.exists():
        return RedirectResponse("/chat.html", status_code=307)
    html = locate_chat_html()
    if html:
        return FileResponse(str(html))
    # Fallback JSON tells where to find UI
    return JSONResponse(
        {
            "message": "Upload chat.html to /public or repo root.",
            "try": {
                "ui": "/chat.html",
                "docs": "/docs",
                "health": "/health",
                "sse": "/sse?q=hello",
                "ws": "/ws (not supported on Vercel serverless)",
            },
        }
    )

# ---------- SSE (Vercel-friendly streaming) ----------
STREAM_START = "<STREAM>"
STREAM_END = "<END>"

@app.get("/sse")
async def sse(request: Request, q: str):
    """
    Server-Sent Events endpoint for streaming model output.
    Use from the browser with: new EventSource('/sse?q=hello')
    """
    def gen():
        # Optional: early abort if client disconnects (best-effort)
        yield f"data: {STREAM_START}\n\n"
        try:
            for chunk in stream_agent_chunks(q):
                # If the client disconnected, stop (FastAPI/Starlette doesn't expose
                # a perfect check; this keeps it simple and robust)
                yield f"data: {chunk}\n\n"
        except Exception:
            logger.exception("SSE: error while streaming")
            yield "data: \n\n"
        finally:
            yield f"data: {STREAM_END}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")

# ---------- WebSockets (fine locally; not for Vercel serverless) ----------
@app.websocket("/ws-echo")
async def ws_echo(ws: WebSocket):
    await ws.accept()
    logger.info("ws-echo: accepted")
    try:
        while True:
            msg = await ws.receive_text()
            logger.info("ws-echo: recv=%r", msg)
            await ws.send_text(f"echo: {msg}")
    except WebSocketDisconnect as e:
        logger.info("ws-echo: client disconnected code=%s", e.code)
    except Exception:
        logger.exception("ws-echo: unhandled error")
        try:
            await ws.close(code=1011)
        except Exception:
            pass

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    logger.info("/ws: accepted")
    try:
        # Simple protocol: first message is the user text
        first = await ws.receive_text()
        await ws.send_text(STREAM_START)
        for chunk in stream_agent_chunks(first):
            await ws.send_text(chunk)
        await ws.send_text(STREAM_END)
    except WebSocketDisconnect as e:
        logger.info("/ws: client disconnected code=%s", e.code)
    except Exception:
        logger.exception("/ws: unhandled error")
        try:
            await ws.close(code=1011)
        except Exception:
            pass

# ---------- NOTE ----------
# Do NOT include `if __name__ == "__main__": uvicorn.run(...)` here.
# Vercel imports `app` from this module via index.py and manages the server.
