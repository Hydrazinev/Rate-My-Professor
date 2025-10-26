# main.py
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterable, Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from starlette.middleware.cors import CORSMiddleware

from agent import create_agent

# ---------- Logging ----------
logger = logging.getLogger("uvicorn.error")
logger.setLevel(logging.INFO)

# ---------- App setup ----------
app = FastAPI(title="Professor Chatbot", version="1.0.0")

# CORS (relax now; lock down later if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Files / Paths ----------
BASE_DIR = Path(__file__).resolve().parent
CHAT_HTML = BASE_DIR / "chat.html"                  # served if exists (Railway)
PUBLIC_CHAT_HTML = BASE_DIR / "public" / "chat.html"  # alt path if you keep it in /public

# ---------- Agent ----------
# Initialize once (env vars must be set in Railway)
try:
    agent = create_agent()
    logger.info("Agent created successfully.")
except Exception as e:
    logger.exception("Failed to create agent at startup.")
    agent = None  # We'll handle None at runtime so WS still accepts.


# ---------- Helpers ----------
def locate_chat_html() -> Optional[Path]:
    if CHAT_HTML.exists():
        return CHAT_HTML
    if PUBLIC_CHAT_HTML.exists():
        return PUBLIC_CHAT_HTML
    return None

def stream_agent_chunks(text: str) -> Iterable[str]:
    """Yield chunks from agent.stream(text, chat_history=None)."""
    if agent is None:
        # Don’t crash—surface a clear message to client
        yield "Server not fully initialized (agent unavailable). Check environment variables and logs."
        return
    # Your agent.stream should be an iterator/generator of str chunks
    for chunk in agent.stream(text, chat_history=None):
        if chunk:
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
    # Redacted health: don’t leak secrets, just presence
    env_ok = {
        "OPENAI_API_KEY": bool(os.environ.get("OPENAI_API_KEY")),
        "PINECONE_API_KEY": bool(os.environ.get("PINECONE_API_KEY")),
        "PINECONE_REGION": bool(os.environ.get("PINECONE_REGION")),
        "PINECONE_INDEX_NAME": bool(os.environ.get("PINECONE_INDEX_NAME")),
        "EMBEDDING_MODEL": bool(os.environ.get("EMBEDDING_MODEL")),
        "RMP_AUTHORIZATION": bool(os.environ.get("RMP_AUTHORIZATION")),
    }
    return {"ok": True, "env": env_ok, "has_agent": agent is not None}

@app.get("/")
async def serve_chat():
    html = locate_chat_html()
    if html:
        return FileResponse(str(html))
    # Fallback if you deploy UI on Vercel
    return JSONResponse(
        {
            "ok": True,
            "message": "Frontend not found on this server. If you host UI on Vercel, open that URL. WebSocket is at /ws",
            "ws": "/ws",
        }
    )

# --- Debug echo websocket (sanity check) ---
@app.websocket("/ws-echo")
async def ws_echo(ws: WebSocket):
    await ws.accept()
    logger.info("ws-echo: accepted")
    try:
        while True:
            msg = await ws.receive_text()
            logger.info(f"ws-echo: recv={msg!r}")
            await ws.send_text(f"echo: {msg}")
    except WebSocketDisconnect as e:
        logger.info(f"ws-echo: client disconnected code={e.code}")
    except Exception:
        logger.exception("ws-echo: unhandled error")
        try:
            await ws.close(code=1011)
        except Exception:
            pass

# --- Main chat websocket ---
STREAM_START = "<STREAM>"
STREAM_END = "<END>"

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    logger.info("/ws: accepted")
    try:
        while True:
            user_text = await ws.receive_text()
            logger.info(f"/ws: recv={user_text!r}")

            # Start streamed response
            await ws.send_text(STREAM_START)
            try:
                for chunk in stream_agent_chunks(user_text):
                    await ws.send_text(chunk)
            except Exception:
                logger.exception("/ws: error while streaming from agent")
                await ws.send_text("\n\n(⚠️ An error occurred while generating a response. Check server logs.)")
            finally:
                await ws.send_text(STREAM_END)

    except WebSocketDisconnect as e:
        logger.info(f"/ws: client disconnected code={e.code}")
    except Exception:
        logger.exception("/ws: unhandled error")
        try:
            await ws.close(code=1011)
        except Exception:
            pass


# ---------- Entrypoint (local dev only) ----------
if __name__ == "__main__":
    # Local dev run; Railway uses your Dockerfile CMD
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True, log_level="info")
