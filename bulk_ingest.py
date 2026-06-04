"""
bulk_ingest.py — Bulk-load real professor data from RateMyProfessors into Pinecone.

Because the RMP GraphQL API has no pagination cursor, we issue multiple queries
per university with different search terms (departments, letter prefixes) to pull
different subsets of professors, then deduplicate by legacyId.

Usage:
    python bulk_ingest.py                          # uses built-in university list
    python bulk_ingest.py --limit 150              # max professors per university
    python bulk_ingest.py --unis "MIT,Stanford"    # custom university list
    python bulk_ingest.py --skip-existing          # skip IDs already in Pinecone

Estimated runtime: ~2-4 min per university at default settings.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from typing import Dict, List, Set

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings

from tools.ratemyprofessor import (
    send_graphql_request,
    load_query_from_file,
    format_professor_response,
    format_university_response,
    get_professor_comments_by_teacher_id,
)

# ── Config ────────────────────────────────────────────────────────────────────
INDEX_NAME      = os.environ.get("PINECONE_INDEX_NAME", "professors-index")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
BATCH_SIZE      = 50     # vectors per Pinecone upsert call
FETCH_PER_TERM  = 500    # professors to request per search term (RMP may cap lower)
COMMENT_LIMIT   = 3      # student comments to store per professor

# ── Default university list ───────────────────────────────────────────────────
DEFAULT_UNIVERSITIES = [
    # California State / UC system
    "California State University Long Beach",
    "California State University Northridge",
    "California State University Fullerton",
    "San Diego State University",
    "UCLA",
    "UC Berkeley",
    "UC San Diego",
    "UC Davis",
    "UC Santa Barbara",
    "UC Irvine",
    # Ivy / Elite
    "MIT",
    "Stanford University",
    "Harvard University",
    "Columbia University",
    "NYU",
    "University of Pennsylvania",
    "Yale University",
    "Princeton University",
    # Large state schools
    "University of Michigan",
    "University of Florida",
    "University of Texas Austin",
    "Ohio State University",
    "Penn State University",
    "Arizona State University",
    "University of Washington",
    "University of Illinois Urbana Champaign",
    # More popular ones
    "USC",
    "Boston University",
    "Purdue University",
    "University of Minnesota",
]

# Search terms we cycle through per university to pull different professor subsets
# (RMP returns at most FETCH_PER_TERM results per query, so multiple queries = more coverage)
SEARCH_TERMS = [
    "",                  # empty → default sort (usually most-rated first)
    "computer science",
    "mathematics",
    "biology",
    "chemistry",
    "physics",
    "english",
    "history",
    "psychology",
    "economics",
    "engineering",
    "business",
    "political science",
    "sociology",
    "philosophy",
    "art",
    "music",
    "nursing",
    "education",
    "communications",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def resolve_school_id(university_name: str) -> tuple[str | None, str | None]:
    """Return (graphql_id, display_name) for a university, or (None, None)."""
    query = load_query_from_file("search_university_by_name")
    variables = {"query": {"text": university_name}}
    resp = send_graphql_request(query, variables)
    schools = format_university_response(resp, limit=1)
    if not schools:
        return None, None
    school = schools[0]
    return school.get("id") or str(school.get("legacyId")), school.get("name", university_name)


def fetch_professors_for_school(
    school_id: str,
    target: int,
    verbose: bool = True,
) -> List[Dict]:
    """
    Pull up to `target` unique professors from a school by cycling through
    SEARCH_TERMS and deduplicating by legacyId.
    """
    query = load_query_from_file("search_teachers_by_school_id")
    seen_ids: Set[int] = set()
    all_profs: List[Dict] = []

    for term in SEARCH_TERMS:
        if len(all_profs) >= target:
            break

        variables = {
            "query": {"text": term, "schoolID": str(school_id)},
            "count": FETCH_PER_TERM,
        }
        try:
            resp = send_graphql_request(query, variables)
            batch = format_professor_response(resp, FETCH_PER_TERM)
        except Exception as e:
            if verbose:
                print(f"    ⚠  Term {term!r} failed: {e}")
            continue

        new = 0
        for prof in batch:
            lid = prof.get("legacyId")
            if lid and lid not in seen_ids:
                seen_ids.add(lid)
                all_profs.append(prof)
                new += 1
                if len(all_profs) >= target:
                    break

        if verbose and new:
            print(f"    term={term!r:25s} → +{new:4d} new  (total {len(all_profs)})")

        time.sleep(0.3)   # be polite to the API

    return all_profs[:target]


def build_vector_text(prof: Dict) -> str:
    """
    Build the structured text that gets embedded and stored.
    Matches the format used by ingest_professors.py so RAG retrieval
    is consistent regardless of data source.
    """
    first = (prof.get("firstName") or "").strip()
    last  = (prof.get("lastName")  or "").strip()
    name  = f"{first} {last}".strip() or "Unknown"
    dept  = prof.get("department") or "Unknown dept"
    school = prof.get("school") or "Unknown school"
    city   = prof.get("schoolCity") or ""
    state  = prof.get("schoolState") or ""
    location = ", ".join(filter(None, [city, state])) or "Unknown location"
    rating  = prof.get("avgRating") or "N/A"
    n_rates = prof.get("numRatings") or 0
    wta     = prof.get("wouldTakeAgainPercentRounded")
    wta_str = f"{wta}%" if wta is not None and wta >= 0 else "N/A"

    text = (
        f"{name} - {dept}. "
        f"School: {school}, {location}. "
        f"Overall {rating}/5 ({n_rates} ratings). "
        f"Would take again: {wta_str}."
    )

    comments = prof.get("comments") or []
    if comments:
        text += " Student comments: " + " | ".join(str(c) for c in comments[:2])

    return text


def build_vector_id(prof: Dict) -> str:
    """Stable, collision-resistant ID: rmp-prof-<hash(legacyId)>."""
    lid = prof.get("legacyId") or prof.get("id") or ""
    return "rmp-prof-" + hashlib.md5(str(lid).encode()).hexdigest()[:12]


def upsert_batch(index, vectors: List[Dict]) -> None:
    for i in range(0, len(vectors), BATCH_SIZE):
        index.upsert(vectors=vectors[i: i + BATCH_SIZE])


def get_existing_ids(index) -> Set[str]:
    """Return the set of vector IDs already in the index."""
    try:
        stats = index.describe_index_stats()
        # Only fetch IDs if the index is small enough to list
        total = stats.get("total_vector_count", 0)
        if total > 10_000:
            print(f"  Index has {total} vectors — skipping existing-ID check (too large).")
            return set()
        # list() is available in newer Pinecone SDKs
        if hasattr(index, "list"):
            ids: Set[str] = set()
            for id_batch in index.list(prefix="rmp-prof-"):
                ids.update(id_batch)
            return ids
    except Exception:
        pass
    return set()


# ── Main ──────────────────────────────────────────────────────────────────────

def main(
    universities: List[str],
    limit_per_uni: int,
    skip_existing: bool,
    fetch_comments: bool,
) -> None:
    pinecone_key = os.environ.get("PINECONE_API_KEY")
    openai_key   = os.environ.get("OPENAI_API_KEY")
    if not pinecone_key:
        raise RuntimeError("Set PINECONE_API_KEY in .env")
    if not openai_key:
        raise RuntimeError("Set OPENAI_API_KEY in .env")

    pc       = Pinecone(api_key=pinecone_key)
    index    = pc.Index(INDEX_NAME)
    embedder = OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=openai_key)

    stats = index.describe_index_stats()
    print(f"Pinecone index '{INDEX_NAME}': {stats.get('total_vector_count', 0)} vectors before ingest\n")

    existing_ids: Set[str] = get_existing_ids(index) if skip_existing else set()
    if skip_existing and existing_ids:
        print(f"  Will skip {len(existing_ids)} IDs already in the index.\n")

    grand_total   = 0
    grand_skipped = 0
    t0 = time.time()

    for uni_name in universities:
        print(f"{'─'*60}")
        print(f"University: {uni_name}")

        # ── 1. Resolve school ID ──────────────────────────────────
        school_id, display_name = resolve_school_id(uni_name)
        if not school_id:
            print(f"  ✗  Could not find school ID — skipping.\n")
            continue
        print(f"  Resolved → {display_name} (id={school_id})")

        # ── 2. Fetch professors ────────────────────────────────────
        print(f"  Fetching up to {limit_per_uni} professors…")
        profs = fetch_professors_for_school(school_id, target=limit_per_uni)
        print(f"  Fetched {len(profs)} unique professors.")

        if not profs:
            print("  Nothing to store.\n")
            continue

        # ── 3. Optionally fetch comments ───────────────────────────
        if fetch_comments:
            print(f"  Fetching comments for top {min(len(profs), 50)} professors…")
            for i, prof in enumerate(profs[:50]):
                try:
                    comments = get_professor_comments_by_teacher_id(prof.get("id"), limit=COMMENT_LIMIT)
                    prof["comments"] = comments
                except Exception:
                    prof["comments"] = []
                if (i + 1) % 10 == 0:
                    print(f"    {i + 1}/{min(len(profs), 50)} comments fetched")
                time.sleep(0.2)

        # ── 4. Build vectors ───────────────────────────────────────
        vectors   = []
        skipped   = 0
        t_embed_0 = time.time()

        for i, prof in enumerate(profs):
            vid = build_vector_id(prof)
            if skip_existing and vid in existing_ids:
                skipped += 1
                continue

            first = (prof.get("firstName") or "").strip()
            last  = (prof.get("lastName")  or "").strip()
            full_name = f"{first} {last}".strip()
            wta = prof.get("wouldTakeAgainPercentRounded")

            text = build_vector_text(prof)
            try:
                vec = embedder.embed_query(text)
            except Exception as e:
                print(f"    ⚠  Embedding failed for {full_name}: {e}")
                continue

            vectors.append({
                "id": vid,
                "values": vec,
                "metadata": {
                    # "text" is REQUIRED by LangChain's PineconeVectorStore to reconstruct
                    # Document.page_content on retrieval. Without it every doc is silently skipped.
                    "text":           text,
                    "professor_name": full_name,
                    "department":     prof.get("department") or "",
                    "school":         prof.get("school") or "",
                    "school_city":    prof.get("schoolCity") or "",
                    "school_state":   prof.get("schoolState") or "",
                    "avg_rating":     float(prof.get("avgRating") or 0),
                    "num_ratings":    int(prof.get("numRatings") or 0),
                    "would_take_again": float(wta) if wta is not None and wta >= 0 else -1.0,
                    "source":         "ratemyprofessors_bulk",
                    "cached_at":      time.time(),
                },
            })

            if (i + 1) % 25 == 0:
                elapsed = time.time() - t_embed_0
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                eta  = (len(profs) - i - 1) / rate if rate > 0 else 0
                print(f"    Embedded {i+1}/{len(profs)}  ({rate:.1f}/s, ETA {eta:.0f}s)")

        # ── 5. Upsert ──────────────────────────────────────────────
        if vectors:
            print(f"  Upserting {len(vectors)} vectors to Pinecone…")
            upsert_batch(index, vectors)
            grand_total   += len(vectors)
            grand_skipped += skipped
            print(f"  ✓  {len(vectors)} upserted, {skipped} skipped (already existed).\n")
        else:
            print(f"  Nothing new to upsert ({skipped} skipped).\n")

        time.sleep(1)   # brief pause between universities

    # ── Summary ───────────────────────────────────────────────────
    elapsed_total = time.time() - t0
    final_stats   = index.describe_index_stats()
    final_count   = final_stats.get("total_vector_count", 0)

    print(f"\n{'═'*60}")
    print(f"Bulk ingest complete in {elapsed_total:.0f}s")
    print(f"  Professors upserted : {grand_total}")
    print(f"  Skipped (existing)  : {grand_skipped}")
    print(f"  Pinecone total now  : {final_count}")
    print(f"{'═'*60}")
    print("\nRAG is now populated with real professor data ✓")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bulk-ingest professors from RMP into Pinecone.")
    parser.add_argument(
        "--unis",
        type=str,
        default="",
        help="Comma-separated university names. Defaults to built-in list of 30 universities.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=200,
        help="Max professors to fetch per university (default: 200).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip professor IDs already in the Pinecone index (default: True).",
    )
    parser.add_argument(
        "--comments",
        action="store_true",
        default=False,
        help="Fetch student comments for top 50 professors per university (slower).",
    )
    args = parser.parse_args()

    uni_list = (
        [u.strip() for u in args.unis.split(",") if u.strip()]
        if args.unis
        else DEFAULT_UNIVERSITIES
    )

    print(f"{'═'*60}")
    print(f"Bulk Professor Ingest")
    print(f"  Universities : {len(uni_list)}")
    print(f"  Limit/uni    : {args.limit}")
    print(f"  Skip existing: {args.skip_existing}")
    print(f"  Fetch comments: {args.comments}")
    print(f"  Total target : ~{len(uni_list) * args.limit:,} professors")
    print(f"{'═'*60}\n")

    try:
        main(
            universities=uni_list,
            limit_per_uni=args.limit,
            skip_existing=args.skip_existing,
            fetch_comments=args.comments,
        )
    except KeyboardInterrupt:
        print("\n\nInterrupted. Professors ingested so far are already saved in Pinecone.")
        sys.exit(130)
