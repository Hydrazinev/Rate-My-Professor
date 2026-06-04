# main.py
from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import time
from collections import deque
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from langchain_core.messages import AIMessage, HumanMessage
from starlette.middleware.cors import CORSMiddleware

from agent import create_agent

# ---------- Logging ----------
logger = logging.getLogger("uvicorn.error")
logger.setLevel(logging.INFO)

# ---------- App setup ----------
app = FastAPI(title="Professor Chatbot", version="1.0.0")

# Fix 7 — CORS: read allowed origins from env var (default "*" for local dev)
_raw_origins = os.environ.get("ALLOWED_ORIGINS", "*")
_allowed_origins = [o.strip() for o in _raw_origins.split(",")] if _raw_origins != "*" else ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Paths ----------
BASE_DIR = Path(__file__).resolve().parent
CHAT_HTML = BASE_DIR / "chat.html"
PUBLIC_CHAT_HTML = BASE_DIR / "public" / "chat.html"

# ---------- Fix 6: JWT expiry check ----------
def _check_rmp_jwt() -> None:
    """Decode the RMP JWT (without verifying signature) and log its expiry."""
    token = os.environ.get("RMP_AUTHORIZATION", "")
    # Strip "Bearer " or "Basic " prefix
    parts = token.split()
    jwt = parts[-1] if parts else ""
    if not jwt or "." not in jwt:
        return
    try:
        # JWT payload is the second base64url segment
        payload_b64 = jwt.split(".")[1]
        # Add padding if needed
        padding = 4 - len(payload_b64) % 4
        payload_b64 += "=" * (padding % 4)
        payload = json.loads(base64.urlsafe_b64decode(payload_b64))
        exp = payload.get("exp")
        if exp:
            expires_in_days = (exp - time.time()) / 86400
            if expires_in_days < 0:
                logger.warning("⚠️  RMP JWT has ALREADY EXPIRED. RMP fallback will fail. Refresh your token.")
            elif expires_in_days < 7:
                logger.warning("⚠️  RMP JWT expires in %.1f days — refresh soon!", expires_in_days)
            else:
                logger.info("RMP JWT valid for %.0f more days.", expires_in_days)
    except Exception as e:
        logger.debug("Could not decode RMP JWT: %s", e)

# ---------- Agent ----------
try:
    agent = create_agent()
    logger.info("Agent created successfully.")
    _check_rmp_jwt()
except Exception:
    logger.exception("Failed to create agent at startup.")
    agent = None


# ---------- Rate limiter ----------

class RateLimiter:
    """
    Token-bucket style per-connection rate limiter.
    Allows at most `max_messages` within a rolling `window_seconds` window.
    """
    def __init__(self, max_messages: int = 5, window_seconds: float = 10.0) -> None:
        self.max_messages = max_messages
        self.window_seconds = window_seconds
        self._timestamps: deque[float] = deque()

    def is_allowed(self) -> bool:
        now = time.monotonic()
        # Expire old timestamps outside the rolling window
        while self._timestamps and now - self._timestamps[0] > self.window_seconds:
            self._timestamps.popleft()
        if len(self._timestamps) >= self.max_messages:
            return False
        self._timestamps.append(now)
        return True


# ---------- Chat history ----------
MAX_HISTORY_TURNS = 10  # keep the last N question-answer pairs per connection


# ---------- Fix 3: Startup warmup ----------
@app.on_event("startup")
async def warmup() -> None:
    """
    Pre-initialize all @cached_property objects (Pinecone, embeddings, LLM,
    retrieval chain) in the background so the first real user message is fast.
    """
    if agent is None:
        return

    async def _run() -> None:
        loop = asyncio.get_running_loop()
        try:
            logger.info("Warmup: initializing Pinecone + embeddings…")
            await loop.run_in_executor(None, lambda: agent._rag.lookup("warmup probe"))
            logger.info("Warmup: initializing retrieval chain…")
            await loop.run_in_executor(
                None,
                lambda: agent._retrieval_chain.invoke(
                    {"input": "warmup", "chat_history": []}
                ),
            )
            logger.info("Warmup complete — first message will be fast ✓")
        except Exception as e:
            logger.warning("Warmup failed (non-fatal): %s", e)

    asyncio.create_task(_run())


# ---------- Helpers ----------

def locate_chat_html() -> Optional[Path]:
    if CHAT_HTML.exists():
        return CHAT_HTML
    if PUBLIC_CHAT_HTML.exists():
        return PUBLIC_CHAT_HTML
    return None


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
    return JSONResponse({
        "ok": True,
        "message": "Frontend not found. If hosted on Vercel, open that URL. WebSocket is at /ws",
        "ws": "/ws",
    })


# ---------- WebSockets ----------

STREAM_START = "<STREAM>"
STREAM_END = "<END>"
RATE_LIMIT_MSG = "⚠️ Too many messages — please wait a moment before sending again."


@app.websocket("/ws-echo")
async def ws_echo(ws: WebSocket):
    """Debug echo endpoint — useful for verifying WebSocket connectivity."""
    await ws.accept()
    logger.info("ws-echo: accepted")
    try:
        while True:
            msg = await ws.receive_text()
            logger.info("ws-echo: recv=%r", msg)
            await ws.send_text(f"echo: {msg}")
    except WebSocketDisconnect as e:
        logger.info("ws-echo: disconnected code=%s", e.code)
    except Exception:
        logger.exception("ws-echo: unhandled error")
        try:
            await ws.close(code=1011)
        except Exception:
            pass


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    """
    Main chat WebSocket.

    Protocol per message:
      client  →  server  : plain text query
      server  →  client  : <STREAM> | token chunks … | <END>

    Features:
      - Per-connection chat history (last MAX_HISTORY_TURNS turns)
      - Per-connection rate limiting (5 msgs / 10 s)
      - True token-level streaming via agent.astream()
    """
    await ws.accept()
    logger.info("/ws: accepted")

    chat_history: list[AIMessage | HumanMessage] = []
    rate_limiter = RateLimiter(max_messages=5, window_seconds=10.0)

    try:
        while True:
            user_text = await ws.receive_text()
            logger.info("/ws: recv=%r", user_text[:120])

            # ---- History restore (sent by client on reconnect) ----
            try:
                parsed = json.loads(user_text)
                if isinstance(parsed, dict) and parsed.get("type") == "restore_history":
                    msgs = parsed.get("messages") or []
                    for entry in msgs:
                        role = entry.get("role", "")
                        content = entry.get("content", "")
                        if role == "user" and content:
                            chat_history.append(HumanMessage(content=content))
                        elif role == "assistant" and content:
                            chat_history.append(AIMessage(content=content))
                    if len(chat_history) > MAX_HISTORY_TURNS * 2:
                        chat_history = chat_history[-(MAX_HISTORY_TURNS * 2):]
                    logger.info("/ws: restored %d history messages", len(msgs))
                    continue  # don't treat this as a chat message
            except (json.JSONDecodeError, TypeError, AttributeError):
                pass  # not JSON — process as a normal chat message

            # ---- Rate limit check ----
            if not rate_limiter.is_allowed():
                await ws.send_text(STREAM_START)
                await ws.send_text(RATE_LIMIT_MSG)
                await ws.send_text(STREAM_END)
                continue

            # ---- Agent unavailable ----
            if agent is None:
                await ws.send_text(STREAM_START)
                await ws.send_text(
                    "Server not fully initialized (agent unavailable). "
                    "Check environment variables and logs."
                )
                await ws.send_text(STREAM_END)
                continue

            # ---- Stream response ----
            await ws.send_text(STREAM_START)
            response_parts: list[str] = []
            try:
                async for chunk in agent.astream(user_text, chat_history):
                    if chunk:
                        await ws.send_text(chunk)
                        response_parts.append(chunk)
            except Exception:
                logger.exception("/ws: error during agent.astream")
                await ws.send_text(
                    "\n\n(⚠️ An error occurred while generating a response. Check server logs.)"
                )
            finally:
                await ws.send_text(STREAM_END)

            # ---- Update per-connection chat history ----
            full_response = "".join(response_parts).strip()
            if full_response:
                chat_history.append(HumanMessage(content=user_text))
                chat_history.append(AIMessage(content=full_response))
                # Keep only the last N turns to prevent unbounded memory growth
                if len(chat_history) > MAX_HISTORY_TURNS * 2:
                    chat_history = chat_history[-(MAX_HISTORY_TURNS * 2):]

    except WebSocketDisconnect as e:
        logger.info("/ws: client disconnected code=%s", e.code)
    except Exception:
        logger.exception("/ws: unhandled error")
        try:
            await ws.close(code=1011)
        except Exception:
            pass


# ---------- Entrypoint (local dev only) ----------
if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True, log_level="info")
