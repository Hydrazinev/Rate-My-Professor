"""
fix_text_metadata.py — One-time patch to add the "text" key to every vector
in the Pinecone index that is missing it.

LangChain's PineconeVectorStore requires a "text" key in vector metadata to
reconstruct Document.page_content on retrieval. The original bulk_ingest.py
forgot to store it, so every similarity_search() silently dropped all results
with the message "Found document with no `text` key. Skipping."

This script:
  1. Lists all vector IDs with the "rmp-prof-" prefix
  2. Fetches them in batches (no re-embedding needed — values are reused)
  3. Reconstructs the text from existing metadata fields
  4. Re-upserts each batch with "text" added

Run once:
    python fix_text_metadata.py

Or do a dry run first (shows what would change, touches nothing):
    python fix_text_metadata.py --dry-run
"""
from __future__ import annotations

import argparse
import os
import time
from typing import Dict, List

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from pinecone import Pinecone

INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "professors-index")
FETCH_BATCH = 100   # Pinecone fetch limit per call
UPSERT_BATCH = 100  # Pinecone upsert batch size


def _reconstruct_text(meta: Dict) -> str:
    """
    Rebuild the searchable text from stored metadata fields.
    Mirrors what bulk_ingest.build_vector_text() produces.
    """
    name      = (meta.get("professor_name") or "Unknown").strip()
    dept      = (meta.get("department")     or "Unknown dept").strip()
    school    = (meta.get("school")         or "Unknown school").strip()
    city      = (meta.get("school_city")    or "").strip()
    state     = (meta.get("school_state")   or "").strip()
    location  = ", ".join(filter(None, [city, state])) or "Unknown location"
    rating    = meta.get("avg_rating")
    n_ratings = int(meta.get("num_ratings") or 0)
    wta       = meta.get("would_take_again")

    rating_str = f"{rating}/5" if rating else "N/A"
    wta_str    = f"{wta}%" if wta is not None and float(wta) >= 0 else "N/A"

    return (
        f"{name} - {dept}. "
        f"School: {school}, {location}. "
        f"Overall {rating_str} ({n_ratings} ratings). "
        f"Would take again: {wta_str}."
    )


def main(dry_run: bool = False) -> None:
    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        raise RuntimeError("Set PINECONE_API_KEY in .env")

    pc    = Pinecone(api_key=api_key)
    index = pc.Index(INDEX_NAME)

    stats = index.describe_index_stats()
    total = stats.get("total_vector_count", 0)
    print(f"Index '{INDEX_NAME}': {total} vectors total")
    print(f"Mode: {'DRY RUN (no writes)' if dry_run else 'LIVE (will upsert)'}\n")

    # ── 1. Collect all IDs ───────────────────────────────────────────────────
    print("Listing all rmp-prof-* IDs…")
    all_ids: List[str] = []
    for batch in index.list(prefix="rmp-prof-"):
        chunk = batch if isinstance(batch, list) else [batch]
        all_ids.extend(chunk)

    print(f"Found {len(all_ids)} vectors with rmp-prof- prefix\n")

    # ── 2. Fetch, patch, re-upsert in batches ────────────────────────────────
    patched = 0
    already_ok = 0
    errors = 0

    for batch_start in range(0, len(all_ids), FETCH_BATCH):
        batch_ids = all_ids[batch_start : batch_start + FETCH_BATCH]

        try:
            result = index.fetch(ids=batch_ids)
        except Exception as e:
            print(f"  ✗ fetch error at offset {batch_start}: {e}")
            errors += len(batch_ids)
            continue

        to_upsert = []
        for vid, vec in result.vectors.items():
            meta = dict(vec.metadata or {})
            if "text" in meta:
                already_ok += 1
                continue

            meta["text"] = _reconstruct_text(meta)
            to_upsert.append({
                "id":     vid,
                "values": list(vec.values),
                "metadata": meta,
            })

        if to_upsert:
            patched += len(to_upsert)
            if not dry_run:
                for sub in range(0, len(to_upsert), UPSERT_BATCH):
                    index.upsert(vectors=to_upsert[sub : sub + UPSERT_BATCH])

        done = min(batch_start + FETCH_BATCH, len(all_ids))
        print(
            f"  [{done:>5}/{len(all_ids)}]  "
            f"patched={patched}  already_ok={already_ok}  errors={errors}",
            end="\r",
        )
        time.sleep(0.1)   # stay within rate limits

    print()
    print("\n" + "─" * 55)
    print(f"Done.")
    print(f"  Patched (text added) : {patched}")
    print(f"  Already had text     : {already_ok}")
    print(f"  Errors               : {errors}")
    if dry_run:
        print("\n(Dry run — nothing was written. Re-run without --dry-run to apply.)")
    else:
        print("\nRAG retrieval will now work correctly ✓")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Show what would change without writing")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
