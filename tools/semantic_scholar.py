"""
tools/semantic_scholar.py — Research paper lookup via the free Semantic Scholar API.

No API key required for basic use (rate-limited to 100 req/5 min).
Docs: https://api.semanticscholar.org/api-docs/

Usage:
    from tools.semantic_scholar import get_professor_papers, format_papers_markdown

    papers = get_professor_papers("Todd Ebert", school="CSULB", limit=5)
    print(format_papers_markdown("Todd Ebert", papers))
"""
from __future__ import annotations

import re
import time
import logging
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

_SS_BASE = "https://api.semanticscholar.org/graph/v1"
_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": "ProfessorChatBot/1.0 (research lookup)"})


# ── helpers ────────────────────────────────────────────────────────────────────

def _clean_name(name: str) -> str:
    """Strip non-alphabetic characters from a professor name for API queries."""
    return re.sub(r"[^A-Za-z\s\-']", "", name).strip()


def _safe_get(url: str, params: dict, timeout: int = 10) -> Optional[Dict[str, Any]]:
    try:
        r = _SESSION.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.HTTPError as e:
        if e.response is not None and e.response.status_code == 429:
            logger.warning("Semantic Scholar rate limit hit — backing off 2s")
            time.sleep(2.0)
        else:
            logger.debug("Semantic Scholar HTTP error %s: %s", url, e)
    except Exception as e:
        logger.debug("Semantic Scholar request failed %s: %s", url, e)
    return None


# ── public API ─────────────────────────────────────────────────────────────────

def get_professor_papers(
    professor_name: str,
    school: str = "",
    limit: int = 5,
) -> List[Dict[str, Any]]:
    """
    Fetch the top-cited research papers by a professor from Semantic Scholar.

    Returns a list of paper dicts:
        {title, year, citations, url, abstract, venue}
    Returns an empty list if nothing is found or the API is unavailable.
    """
    name_clean = _clean_name(professor_name)
    if not name_clean or len(name_clean) < 4:
        return []

    # ── 1. Search for the author ──────────────────────────────────────────────
    data = _safe_get(
        f"{_SS_BASE}/author/search",
        params={
            "query": name_clean,
            "fields": "name,affiliations,paperCount,citationCount",
            "limit": 8,
        },
    )
    if not data:
        return []

    authors: List[Dict] = data.get("data") or []
    if not authors:
        return []

    # Pick the best match: prefer author with school affiliation if given
    author = authors[0]
    if school:
        school_lower = school.lower()
        for a in authors:
            affiliations = [
                (aff.get("name") or "").lower()
                for aff in (a.get("affiliations") or [])
            ]
            if any(school_lower in aff or aff in school_lower for aff in affiliations):
                author = a
                break
        else:
            # Fall back to best name match
            name_lower = name_clean.lower()
            scored = sorted(
                authors,
                key=lambda a: (
                    a.get("name", "").lower() == name_lower,
                    int(a.get("paperCount") or 0),
                ),
                reverse=True,
            )
            if scored:
                author = scored[0]

    author_id = author.get("authorId")
    if not author_id:
        return []

    logger.debug(
        "Semantic Scholar: found author %r (id=%s, %d papers)",
        author.get("name"),
        author_id,
        author.get("paperCount", 0),
    )

    # ── 2. Fetch their top papers ─────────────────────────────────────────────
    time.sleep(0.15)   # stay polite to the free API
    papers_data = _safe_get(
        f"{_SS_BASE}/author/{author_id}/papers",
        params={
            "fields": "title,year,citationCount,externalIds,abstract,venue,publicationTypes",
            "limit": limit * 3,  # fetch more so we can sort + trim
        },
    )
    if not papers_data:
        return []

    raw_papers: List[Dict] = papers_data.get("data") or []

    # Sort by citation count (most impactful first)
    raw_papers.sort(key=lambda p: int(p.get("citationCount") or 0), reverse=True)

    result: List[Dict[str, Any]] = []
    for p in raw_papers[:limit]:
        external = p.get("externalIds") or {}
        doi = external.get("DOI")
        paper_id = p.get("paperId") or ""
        url = (
            f"https://doi.org/{doi}"
            if doi
            else f"https://www.semanticscholar.org/paper/{paper_id}"
        )

        abstract = (p.get("abstract") or "").strip()
        if len(abstract) > 220:
            abstract = abstract[:217] + "..."

        result.append({
            "title":    p.get("title") or "Untitled",
            "year":     p.get("year"),
            "citations": int(p.get("citationCount") or 0),
            "url":      url,
            "abstract": abstract,
            "venue":    (p.get("venue") or "").strip(),
        })

    return result


def format_papers_markdown(
    professor_name: str,
    papers: List[Dict[str, Any]],
    school: str = "",
) -> str:
    """
    Format a paper list into markdown that the frontend can display.

    Example output:
        📚 Research papers by **Todd Ebert** (via Semantic Scholar):

        1. **Some Paper Title** (2019) · 42 citations · _IEEE Transactions_
           > Brief abstract here...
           🔗 https://doi.org/...

        2. ...
    """
    if not papers:
        where = f" at {school}" if school else ""
        return (
            f"I couldn't find research papers for **{professor_name}**{where} "
            "on Semantic Scholar. They may publish under a different name, "
            "or their work may not be indexed there."
        )

    lines = [f"📚 Research papers by **{professor_name}** (via Semantic Scholar):\n"]
    for i, p in enumerate(papers, 1):
        year_str  = f" ({p['year']})" if p.get("year") else ""
        cite_str  = f" · **{p['citations']:,}** citations" if p.get("citations") else ""
        venue_str = f" · _{p['venue']}_" if p.get("venue") else ""

        lines.append(f"{i}. **{p['title']}**{year_str}{cite_str}{venue_str}")
        if p.get("abstract"):
            lines.append(f"   > {p['abstract']}")
        lines.append(f"   🔗 [{p['url']}]({p['url']})")
        lines.append("")

    lines.append(
        "_Results from [Semantic Scholar](https://www.semanticscholar.org) — "
        "citation counts update periodically._"
    )
    return "\n".join(lines)
