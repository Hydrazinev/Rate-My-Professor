"""
clean_index.py — Wipe the Pinecone index and re-seed it from seed_professors.json.

Run once when you want to remove stale / badly-formatted cached entries:
    python clean_index.py

What it does:
  1. Deletes ALL vectors from the index (non-destructive to the index itself)
  2. Re-ingests seed_professors.json with the clean structured format
  3. Prints a summary

Safe to run multiple times — the ingest uses deterministic IDs so re-running
won't create duplicates.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings

INDEX_NAME     = os.environ.get("PINECONE_INDEX_NAME", "professors-index")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
SEED_FILE      = os.path.join(os.path.dirname(__file__), "seed_professors.json")


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:60]


def main() -> None:
    pinecone_key = os.environ.get("PINECONE_API_KEY")
    openai_key   = os.environ.get("OPENAI_API_KEY")
    if not pinecone_key:
        raise RuntimeError("Set PINECONE_API_KEY")
    if not openai_key:
        raise RuntimeError("Set OPENAI_API_KEY")

    pc    = Pinecone(api_key=pinecone_key)
    index = pc.Index(INDEX_NAME)

    # ── Step 1: count then wipe ──────────────────────────────────────────
    stats_before = index.describe_index_stats()
    count_before = stats_before.get("total_vector_count", 0)
    print(f"Vectors before wipe: {count_before}")

    if count_before > 0:
        print("Deleting all vectors…")
        index.delete(delete_all=True)
        print("Done.")
    else:
        print("Index already empty — skipping delete.")

    # ── Step 2: re-seed ──────────────────────────────────────────────────
    if not os.path.exists(SEED_FILE):
        print(f"No seed file at {SEED_FILE} — index is now empty.")
        return

    with open(SEED_FILE, encoding="utf-8") as f:
        rows = json.load(f)

    if not rows:
        print("Seed file is empty.")
        return

    print(f"\nRe-seeding {len(rows)} professors from {os.path.basename(SEED_FILE)}…")
    embedder = OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=openai_key)

    vectors = []
    for r in rows:
        name    = str(r["professor_name"])
        course  = str(r["course"])
        text = (
            f"{name} - {course}. "
            f"Overall {float(r['overall_rating'])}/5, "
            f"clarity {float(r['clarity'])}, "
            f"helpfulness {float(r['helpfulness'])}, "
            f"easiness {float(r['easiness'])}. "
            f"Comment: {r['comment']}"
        )
        base = f"{name}::{course}"
        vec_id = slugify(base) + "-" + hashlib.md5(base.encode()).hexdigest()[:8]
        vec = embedder.embed_query(text)
        vectors.append({
            "id": vec_id,
            "values": vec,
            "metadata": {
                "professor_name": name,
                "course": course,
                "overall_rating": float(r["overall_rating"]),
                "clarity":        float(r["clarity"]),
                "helpfulness":    float(r["helpfulness"]),
                "easiness":       float(r["easiness"]),
                "comment":        str(r["comment"]),
                "source":         "seed",
            },
        })

    BATCH = 100
    for i in range(0, len(vectors), BATCH):
        batch = vectors[i: i + BATCH]
        index.upsert(vectors=batch)
        print(f"  Upserted {i + len(batch)}/{len(vectors)}")

    stats_after = index.describe_index_stats()
    print(f"\nVectors after re-seed: {stats_after.get('total_vector_count', 0)}")
    print("Index is clean ✓")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
