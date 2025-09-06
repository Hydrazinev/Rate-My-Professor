# main.py
from __future__ import annotations

import logging
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from starlette.middleware.cors import CORSMiddleware

from agent import create_agent

# ---------- App setup ----------
app = FastAPI(title="Professor Chatbot", version="1.0.0")

# Allow local development from any origin (you can lock this down later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# Static file path (chat.html should live next to this file)
BASE_DIR = Path(__file__).resolve().parent
CHAT_HTML = BASE_DIR / "chat.html"

# Initialize your RAG/LLM agent once (reads .env via AgentConfig)
agent = create_agent()

# ---------- Routes ----------
@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/")
async def serve_chat():
    # Serve the chat UI
    return FileResponse(str(CHAT_HTML))

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            # Receive user text from the browser
            user_text = await ws.receive_text()

            # Start streaming to the client
            await ws.send_text("<STREAM>")
            try:
                # Stream chunks from your agent
                for chunk in agent.stream(user_text, chat_history=None):
                    if chunk:
                        await ws.send_text(chunk)
            except Exception as e:
                await ws.send_text(f"\n\n(⚠️ error: {e})")
            finally:
                # Signal end of the streamed message
                await ws.send_text("<END>")
    except WebSocketDisconnect:
        # Client closed the tab or lost connection
        pass

# ---------- Entrypoint ----------
if __name__ == "__main__":
    logging.getLogger("uvicorn.error").setLevel(logging.INFO)
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
