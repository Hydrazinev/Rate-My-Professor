"""
Seed Pinecone with professor data using OpenAI embeddings.
Behavior preserved; refactor improves structure, typing, validation, and logging.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Dict, List

try:
    from dotenv import load_dotenv  # optional, best-effort
    load_dotenv()
except Exception:
    pass

from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings


# =============================================================================
# Config
# =============================================================================

INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "professors-index")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
PINECONE_REGION = os.environ.get("PINECONE_REGION", "us-east-1")
SEED_FILE = os.path.join(os.path.dirname(__file__), "seed_professors.json")
BATCH_SIZE = int(os.environ.get("SEED_BATCH_SIZE", "100"))  # behavior preserved


# =============================================================================
# Utilities
# =============================================================================

def slugify(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return text[:60]


def load_rows(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Seed file must be a JSON array of objects, got: {type(data).__name__}")
    return data


def get_dimension(emb_model: str, openai_key: str) -> int:
    # Probe once; keeps your original behavior.
    emb = OpenAIEmbeddings(model=emb_model, api_key=openai_key)
    vec = emb.embed_query("dimension probe")
    return len(vec)


def ensure_index(pc: Pinecone, name: str, dimension: int, *, region: str = PINECONE_REGION) -> None:
    exists = [i.name for i in pc.list_indexes()]
    if name in exists:
        desc = pc.describe_index(name)
        # v5 describe may act like an object or dict—support both:
        idx_dim = getattr(desc, "dimension", None) or (desc.get("dimension") if isinstance(desc, dict) else None)
        if idx_dim and idx_dim != dimension:
            raise RuntimeError(
                f"Existing index '{name}' has dimension {idx_dim}, but embeddings are {dimension}."
            )
        print(f"Index '{name}' already exists.")
        return

    print(f"Creating index '{name}' with dim={dimension} (cosine) in {region}…")
    pc.create_index(
        name=name,
        dimension=dimension,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=region),
    )

    # Wait until ready
    while True:
        desc = pc.describe_index(name)
        status = getattr(desc, "status", None) or desc.get("status", {})
        if status.get("ready"):
            print("Index is ready.")
            break
        print("Waiting for index to be ready…")
        time.sleep(2)


@dataclass
class SeedRow:
    professor_name: str
    course: str
    overall_rating: float
    clarity: float
    helpfulness: float
    easiness: float
    comment: str

    @classmethod
    def from_dict(cls, d: Dict) -> "SeedRow":
        # Keep behavior: cast numerics to float; require keys
        try:
            return cls(
                professor_name=str(d["professor_name"]),
                course=str(d["course"]),
                overall_rating=float(d["overall_rating"]),
                clarity=float(d["clarity"]),
                helpfulness=float(d["helpfulness"]),
                easiness=float(d["easiness"]),
                comment=str(d["comment"]),
            )
        except KeyError as e:
            raise ValueError(f"Missing required field in seed row: {e}") from e


def make_text_payload(r: SeedRow) -> str:
    # Preserve your original wording
    return (
        f"{r.professor_name} - {r.course}. "
        f"Overall {r.overall_rating}/5, clarity {r.clarity}, "
        f"helpfulness {r.helpfulness}, easiness {r.easiness}. "
        f"Comment: {r.comment}"
    )


def make_vector_id(r: SeedRow) -> str:
    base = f"{r.professor_name}::{r.course}"
    return slugify(base) + "-" + hashlib.md5(base.encode("utf-8")).hexdigest()[:8]


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    pinecone_key = os.environ.get("PINECONE_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")

    if not pinecone_key:
        raise RuntimeError("Set PINECONE_API_KEY (in .env or environment).")
    if not openai_key:
        raise RuntimeError("Set OPENAI_API_KEY (in .env or environment).")
    if not os.path.exists(SEED_FILE):
        raise FileNotFoundError(f"Seed file not found: {SEED_FILE}")

    rows_raw = load_rows(SEED_FILE)
    if not rows_raw:
        print("Seed file is empty; nothing to upsert.")
        return

    # Validate/normalize rows up front (keeps failure early & clear)
    rows: List[SeedRow] = [SeedRow.from_dict(r) for r in rows_raw]
    print(f"Loaded {len(rows)} rows from {os.path.basename(SEED_FILE)}")
    print(f"Using embedding model: {EMBEDDING_MODEL}")

    # Detect dimension and ensure index
    dimension = get_dimension(EMBEDDING_MODEL, openai_key)
    print(f"Detected embedding dimension: {dimension}")

    pc = Pinecone(api_key=pinecone_key)
    ensure_index(pc, INDEX_NAME, dimension, region=PINECONE_REGION)

    index = pc.Index(INDEX_NAME)
    embedder = OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=openai_key)

    vectors = []
    for r in rows:
        text = make_text_payload(r)
        vec = embedder.embed_query(text)
        vectors.append(
            {
                "id": make_vector_id(r),
                "values": vec,
                "metadata": {
                    "professor_name": r.professor_name,
                    "course": r.course,
                    "overall_rating": r.overall_rating,
                    "clarity": r.clarity,
                    "helpfulness": r.helpfulness,
                    "easiness": r.easiness,
                    "comment": r.comment,
                    "source": "seed_professors.json",
                },
            }
        )

    # Upsert in batches (same default size as your original)
    total = len(vectors)
    for i in range(0, total, BATCH_SIZE):
        batch = vectors[i : i + BATCH_SIZE]
        index.upsert(vectors=batch)
        print(f"Upserted {i + len(batch)}/{total}")

    print("Done. Your Pinecone index is seeded.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
