from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterable

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import (
    JSONResponse,
    RedirectResponse,
    StreamingResponse,
)
from starlette.middleware.cors import CORSMiddleware

from agent import create_agent

# ----------------- Logging -----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("rmp")

# ----------------- App setup -----------------
app = FastAPI(title="Rate My Professor", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------- Agent init -----------------
try:
    agent = create_agent()
    logger.info("Agent created successfully.")
except Exception:
    logger.exception("Failed to create agent at startup.")
    agent = None


# ----------------- Helper -----------------
def stream_agent_chunks(text: str) -> Iterable[str]:
    if agent is None:
        yield "Agent not initialized. Check environment variables."
        return
    for chunk in agent.stream(text, chat_history=None):
        yield str(chunk)


# ----------------- Routes -----------------
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


# ✅ redirect to /chat.html (served by Vercel static CDN)
@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse("/chat.html", status_code=307)


# ----------------- SSE (for Vercel streaming) -----------------
STREAM_START = "<STREAM>"
STREAM_END = "<END>"

@app.get("/sse")
async def sse(request: Request, q: str):
    def gen():
        yield f"data: {STREAM_START}\n\n"
        try:
            for chunk in stream_agent_chunks(q):
                yield f"data: {chunk}\n\n"
        except Exception:
            logger.exception("SSE error while streaming")
            yield "data: \n\n"
        finally:
            yield f"data: {STREAM_END}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


# ----------------- WebSockets (work locally; not on Vercel) -----------------
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


# ----------------- NOTE -----------------
# ❌ DO NOT include `if __name__ == "__main__": uvicorn.run(...)`
# ✅ Vercel imports `app` automatically from index.py and runs it
