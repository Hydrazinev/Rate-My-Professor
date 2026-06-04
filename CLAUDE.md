# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI-powered professor search chatbot. Users chat in a browser; the backend streams answers token-by-token over WebSocket. Answers come from a Pinecone vector index (RAG) first, falling back to the live RateMyProfessors GraphQL API.

## Commands

**Bulk-ingest real professors from RMP into Pinecone (~60–90 min, run once):**
```bash
python bulk_ingest.py                         # all 30 universities, 200 profs each
python bulk_ingest.py --limit 50 --unis "MIT,Stanford"  # quick test run
python bulk_ingest.py --skip-existing         # skip IDs already ingested
```

**Wipe dirty vectors and re-seed from scratch:**
```bash
python clean_index.py
```

**Run locally (dev with auto-reload):**
```bash
uvicorn main:app --reload --host 127.0.0.1 --port 8000
# or simply:
python main.py
```

**CLI interactive test (no server needed):**
```bash
python agent.py
```

**Install dependencies (uses uv):**
```bash
uv pip install -r requirements.txt
```

**Seed / populate Pinecone with professor data:**
```bash
# Expects: seed_professors.json in the project root (see data/sample.json for schema)
python ingest_professors.py
```

**Create Pinecone index manually (one-time):**
```bash
python create_index.py
```

**Docker (mirrors Railway production):**
```bash
docker build -t professor-chatbot .
docker run -p 8000:8000 --env-file .env professor-chatbot
```

## Required Environment Variables (`.env` or Railway)

| Variable | Default | Notes |
|---|---|---|
| `OPENAI_API_KEY` | — | Required |
| `PINECONE_API_KEY` | — | Required |
| `PINECONE_INDEX_NAME` | `professors-index` | |
| `PINECONE_REGION` | `us-east-1` | AWS serverless |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | |
| `RMP_AUTHORIZATION` | — | Full header: `Bearer <JWT>` **or** use `RMP_AUTH_TOKEN` + `RMP_AUTH_SCHEME` |
| `RMP_RMPAUTH_COOKIE` | — | Optional; JWT value of the `rmpAuth` cookie |

Health check routes `/health` and `/healthz` confirm which env vars are present without leaking values.

## Architecture

### Request Flow

```
Browser WebSocket (/ws)
  └─> main.py: ws_endpoint()  [per-connection chat_history + RateLimiter]
        └─> agent.py: ProfessorRaterAgent.astream()
              ├─ [1] Intent gate (_is_professor_intent)
              │        out-of-scope regex → conversational shortcut → keyword regex
              │        → name heuristic → LLM binary judge (gpt-4o-mini)
              ├─ [2] Conversational handler (greetings/help) — no RAG/RMP
              ├─ [3] Follow-up resolver (_resolve_followup)
              │        Short pronoun queries ("Is he easy?") get the context professor
              │        appended from the last AI message in chat_history
              ├─ [4] Top-rated shortcut → run_direct() (direct RMP, no RAG)
              ├─ [5] Comparison shortcut → _stream_comparison()
              │        Parallel profile fetch (ThreadPoolExecutor) + LLM streaming
              ├─ [6] Pre-flight Pinecone check (_docs_cover_query)
              │        lookup(top_k=4) → check if retrieved docs mention the queried
              │        professor name (both first+last in content). If not → RMP.
              ├─ [6a] RAG HIT → stream _retrieval_chain
              │        No NO_PROFESSOR_MARKER. If pre-flight said "relevant", trust it.
              └─ [6b] RAG MISS → run_direct() (deterministic RMP routing)
                        Extracts school/dept/name from query → calls the right RMP
                        function directly (no AgentExecutor, no hub.pull())
```

### WebSocket Protocol

`main.py` uses sentinel tokens:
```
server → client:  <STREAM>  token…token…token…  <END>
```
- Tokens are yielded as they come from the LLM — the client renders text progressively.
- At `<END>` the client re-renders with full card/comment parsing.
- The `NO PROFESSOR` marker is stripped client-side before display.
- Debug echo endpoint at `/ws-echo` for connectivity testing.

### Key Files

- **`agent.py`** — `ProfessorRaterAgent`. All heavy objects (`_llm`, `_rag`, `_embeddings`, `_retrieval_chain`) are `@cached_property`. `astream()` is the primary entry point — it uses LangChain's `.astream()` on the retrieval chain with a sliding-window buffer (`accumulated[:-marker_len]`) so the `NO_PROFESSOR_MARKER` is never sent to the client mid-stream. `stream()` / `invoke()` are sync fallbacks used by the CLI.
- **`rag.py`** — `RAG` class wraps `PineconeVectorStore`. Auto-creates a serverless Pinecone index if missing (infers dimension from a probe embedding). Exposes `get_retriever()` for the LangChain chain.
- **`tools/ratemyprofessor.py`** — Thin GraphQL client for RateMyProfessors. `run_direct(query, history)` is the main entry point — it parses the query (school, department, professor name) and calls the right function directly. No `AgentExecutor`. `_extract_school_name`, `_extract_department`, `_extract_professor_name_from_query` are the parsing helpers. `_format_professor_rows` produces the numbered-markdown format the frontend card parser expects. GraphQL queries live in `tools/*.graphql`.
- **`utils.py`** — Shared `extract_professor_names(answer_text, user_input)` used by both `agent.py` and `tools/ratemyprofessor.py` to parse professor names out of answer text for comment enrichment.
- **`main.py`** — FastAPI app. Per-connection state: `chat_history` (last `MAX_HISTORY_TURNS` turns) and a `RateLimiter` (5 msgs / 10 s). Serves `public/chat.html` as static file.
- **`ingest_professors.py`** — Seeds the Pinecone index from `seed_professors.json`. Each row needs: `professor_name`, `course`, `overall_rating`, `clarity`, `helpfulness`, `easiness`, `comment`.

### Key Design Decisions

**Why `_docs_cover_query` instead of `NO_PROFESSOR_MARKER`?**
The old approach had the LLM emit a literal string marker ("NO PROFESSOR") when it couldn't find the professor, then the streaming code used a sliding-window buffer to detect it mid-stream and trigger a fallback. This was brittle — the LLM could paraphrase it. The new approach checks upfront (before running the LLM at all) whether the retrieved Pinecone docs actually mention the queried professor's first + last name. If not, skip directly to RMP.

**Why `run_direct` instead of `AgentExecutor`?**
The RMP fallback use case is deterministic: "best CS professors at Stanford" → `get_top_professors("Stanford", "Computer Science")`. There's no need for multi-step tool-calling reasoning. Replacing the agent with direct routing eliminates ~300–800ms of LLM reasoning overhead, removes hub.pull() network calls at startup, and makes the code predictable.

**Why pre-flight lookup instead of re-using retrieval chain docs?**
We call `self._rag.lookup()` twice (pre-flight + inside `_retrieval_chain`). The small duplication avoids restructuring the LangChain chain internals and keeps `_docs_cover_query` as a clean boolean guard before any LLM call.

### Intent Filtering (5 layers, agent.py)

1. Explicit out-of-scope regex (cars, movies, food, medical, etc.) → reject
2. Conversational regex (greetings, "who are you") → canned reply, no RAG
3. Professor/course/education/rating keyword regex → accept
4. Name heuristic (2–5 word all-alpha string) → treat as professor name lookup
5. LLM binary classifier (`gpt-4o-mini`, "yes"/"no") as final arbiter

### Dual-Source Comment Enrichment

After an answer is assembled (RAG or RMP path), `utils.extract_professor_names()` parses names from the text. `_append_comments_if_missing()` then fetches live student comments from RMP and appends them if none are already present. The same logic runs in `run_tools()` for RMP-sourced answers.

### Deployment Split

- **Backend** — Railway (Docker). `CMD` in `Dockerfile` uses `${PORT:-8000}` for Railway's dynamic port injection.
- **Frontend** — Either served by FastAPI (`public/chat.html`) or deployed separately on Vercel (`vercel.json` rewrites `/` → `/chat.html`). `locate_chat_html()` in `main.py` checks both locations.

## Python Version & Tooling

- Python ≥ 3.11, < 3.13
- Package manager: **uv** (`uv pip install -r requirements.txt`)
- `pinecone` v5 (the modern SDK, **not** the old `pinecone-client`)
- `requirements.txt` is the single source of truth for deployment; `requirements/` contains environment-split variants (dev, prod, optional)
