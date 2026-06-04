from __future__ import annotations

# =============================================================================
# RateMyProfessors LangChain tools & thin GraphQL client
# =============================================================================

from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

import base64
import json
import os
import re
from html import unescape as html_unescape

import requests
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_core.tools import StructuredTool

from utils import extract_professor_names


# =============================================================================
# Environment & constants
# =============================================================================

load_dotenv()

BASE_URL: str = os.getenv("RMP_BASE_URL", "https://www.ratemyprofessors.com/graphql")

_AUTHORIZATION_FULL: str = (os.getenv("RMP_AUTHORIZATION") or "").strip()
_AUTH_TOKEN_ONLY: str = (os.getenv("RMP_AUTH_TOKEN") or "").strip()
_AUTH_SCHEME: str = (os.getenv("RMP_AUTH_SCHEME") or "Bearer").strip()
_RMP_COOKIE: str = (os.getenv("RMP_RMPAUTH_COOKIE") or "").strip()

# Ignore system proxy settings — local bad proxies can break requests
_HTTP = requests.Session()
_HTTP.trust_env = False


# =============================================================================
# Types
# =============================================================================

JSON = Dict[str, Any]


class GraphQLPayload(TypedDict, total=False):
    query: str
    variables: Dict[str, Any]


# =============================================================================
# HTTP / headers helpers
# =============================================================================

def _build_auth_header() -> str:
    if _AUTHORIZATION_FULL:
        return _AUTHORIZATION_FULL
    if _AUTH_TOKEN_ONLY:
        return f"{_AUTH_SCHEME} {_AUTH_TOKEN_ONLY}"
    raise RuntimeError(
        "Missing RateMyProfessors auth.\n"
        "Provide either:\n"
        "  - RMP_AUTHORIZATION='Bearer <JWT>' (or 'Basic <base64>')\n"
        "  - RMP_AUTH_TOKEN='<token>' and optional RMP_AUTH_SCHEME='Bearer' (default)"
    )


def _default_headers() -> Dict[str, str]:
    headers = {
        "Accept": "*/*",
        "Content-Type": "application/json",
        "Origin": "https://www.ratemyprofessors.com",
        "Referer": "https://www.ratemyprofessors.com/",
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/127.0.0.0 Safari/537.36"
        ),
        "Authorization": _build_auth_header(),
    }
    if _RMP_COOKIE:
        headers["Cookie"] = f"rmpAuth={_RMP_COOKIE}"
    return headers


# =============================================================================
# GraphQL query loader
# =============================================================================

_THIS_DIR = Path(__file__).resolve().parent

_QUERY_CANDIDATES: Dict[str, List[str]] = {
    "search_professor_by_name": ["search_professor_by_name.graphql"],
    "search_university_by_name": ["search_university_by_name.graphql"],
    "search_teachers_by_school_id": [
        "search_teachers_by_school_id.graphql",
        "search_professor_by_school_id.graphql",
    ],
}


def _resolve_query_path(preferred_name: str) -> Path:
    candidates = _QUERY_CANDIDATES.get(preferred_name, [preferred_name])
    search_dirs = [_THIS_DIR, _THIS_DIR.parent, _THIS_DIR.parent / "tools"]
    for fname in candidates:
        for d in search_dirs:
            p = d / fname
            if p.exists():
                return p
    return _THIS_DIR / candidates[0]


def load_query_from_file(logical_name: str) -> str:
    path = _resolve_query_path(logical_name)
    if not path.exists():
        tried = ", ".join(_QUERY_CANDIDATES.get(logical_name, [logical_name]))
        raise FileNotFoundError(
            f"GraphQL file not found for '{logical_name}'. "
            f"Tried: {tried} in {_THIS_DIR}, its parent, and 'tools/'."
        )
    return path.read_text(encoding="utf-8")


# =============================================================================
# HTTP / GraphQL utilities
# =============================================================================

def send_graphql_request(
    query: str,
    variables: Optional[Dict[str, Any]] = None,
    timeout: int = 20,
) -> JSON:
    payload: GraphQLPayload = {"query": query, "variables": variables or {}}
    try:
        resp = _HTTP.post(
            BASE_URL,
            headers=_default_headers(),
            data=json.dumps(payload),
            timeout=timeout,
        )
        resp.raise_for_status()
        data: JSON = resp.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"HTTP error: {e}"}
    except json.JSONDecodeError:
        return {"error": "Invalid JSON in response"}

    if isinstance(data, dict) and data.get("errors"):
        return {"error": f"GraphQL errors: {data['errors']}"}

    return data


# =============================================================================
# Response formatters
# =============================================================================

def format_university_response(response: JSON, limit: int) -> List[JSON]:
    out: List[JSON] = []
    if not isinstance(response, dict) or "error" in response:
        return out
    new = response.get("data", {}).get("newSearch", {}) or {}
    edges = (new.get("schools", {}) or {}).get("edges", []) or []
    for edge in edges[:limit]:
        node = edge.get("node", {}) or {}
        out.append({
            "id": node.get("id"),
            "legacyId": node.get("legacyId"),
            "name": node.get("name"),
            "city": node.get("city"),
            "state": node.get("state"),
            "avgRatingRounded": node.get("avgRatingRounded"),
            "numRatings": node.get("numRatings"),
            "departments": [
                {"id": d.get("id"), "name": d.get("name")}
                for d in (node.get("departments") or [])
            ],
        })
    return out


def format_professor_response(response: JSON, limit: int) -> List[JSON]:
    out: List[JSON] = []
    if not isinstance(response, dict) or "error" in response:
        return out
    new = response.get("data", {}).get("newSearch", {}) or {}
    edges = (new.get("teachers", {}) or {}).get("edges", []) or []
    for edge in edges[:limit]:
        node = edge.get("node", {}) or {}
        school = node.get("school") or {}
        out.append({
            "id": node.get("id"),
            "legacyId": node.get("legacyId"),
            "firstName": node.get("firstName"),
            "lastName": node.get("lastName"),
            "department": node.get("department"),
            "school": school.get("name"),
            "schoolCity": school.get("city"),
            "schoolState": school.get("state"),
            "avgRating": node.get("avgRating"),
            "avgDifficulty": node.get("avgDifficulty"),
            "numRatings": node.get("numRatings"),
            "wouldTakeAgainPercentRounded": node.get("wouldTakeAgainPercentRounded"),
        })
    return out


def _extract_comments_from_professor_page(page_html: str, limit: int = 3) -> List[str]:
    if not page_html:
        return []
    patterns = [
        r'"comment":"((?:\\.|[^"\\])*)"',
        r'"teacherComment":"((?:\\.|[^"\\])*)"',
        r'"rComments":"((?:\\.|[^"\\])*)"',
    ]
    raw_comments: List[str] = []
    for pat in patterns:
        raw_comments.extend(re.findall(pat, page_html))
    cleaned: List[str] = []
    seen: set[str] = set()
    for raw in raw_comments:
        txt = raw.encode("utf-8").decode("unicode_escape")
        txt = html_unescape(txt)
        txt = txt.replace("\\n", " ").replace("\\r", " ").strip()
        txt = re.sub(r"\s+", " ", txt)
        if len(txt) < 8 or txt.lower() in seen:
            continue
        seen.add(txt.lower())
        cleaned.append(txt)
        if len(cleaned) >= limit:
            break
    return cleaned


def _teacher_id_from_legacy_id(legacy_id: Any) -> str:
    try:
        raw = f"Teacher-{int(legacy_id)}".encode("utf-8")
    except Exception:
        return ""
    return base64.b64encode(raw).decode("utf-8")


def get_professor_comments_by_teacher_id(teacher_id: Any, limit: int = 3) -> List[str]:
    if not teacher_id:
        return []
    query = """
    query TeacherComments($id: ID!, $count: Int!) {
      node(id: $id) {
        __typename
        ... on Teacher {
          ratings(first: $count) {
            edges {
              node { comment class date qualityRating }
            }
          }
        }
      }
    }
    """
    resp = send_graphql_request(query, {"id": str(teacher_id), "count": int(max(1, limit))})
    if not isinstance(resp, dict) or "error" in resp:
        return []
    edges = (
        resp.get("data", {}).get("node", {}).get("ratings", {}).get("edges", []) or []
    )
    comments: List[str] = []
    seen: set[str] = set()
    for edge in edges:
        node = (edge or {}).get("node", {}) or {}
        txt = (node.get("comment") or "").strip()
        if not txt:
            continue
        norm = re.sub(r"\s+", " ", txt).strip()
        if len(norm) < 8 or norm.lower() in seen:
            continue
        seen.add(norm.lower())
        comments.append(norm)
        if len(comments) >= limit:
            break
    return comments


def get_professor_comments_by_legacy_id(legacy_id: Any, limit: int = 3) -> List[str]:
    """Prefer GraphQL; fall back to HTML scraping."""
    teacher_id = _teacher_id_from_legacy_id(legacy_id)
    comments = get_professor_comments_by_teacher_id(teacher_id, limit=limit)
    if comments:
        return comments
    if not legacy_id:
        return []
    try:
        url = f"https://www.ratemyprofessors.com/professor/{legacy_id}"
        resp = _HTTP.get(
            url,
            headers={"User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36"
            )},
            timeout=15,
        )
        if resp.ok:
            return _extract_comments_from_professor_page(resp.text, limit=limit)
    except Exception:
        pass
    return []


# =============================================================================
# Public search functions
# =============================================================================

def get_professor(name: str, limit: int = 5) -> List[JSON]:
    """Search professors by free-text name."""
    query = load_query_from_file("search_professor_by_name")
    fetch_count = max(int(limit), 10)
    variables = {"query": {"text": name}, "count": fetch_count}
    resp = send_graphql_request(query, variables)
    out = format_professor_response(resp, fetch_count)

    q = " ".join((name or "").strip().lower().split())
    q_tokens = [t for t in re.findall(r"[a-z]+", q) if t]

    def _score(row: JSON) -> tuple[int, int, float]:
        first = str(row.get("firstName") or "").strip().lower()
        last = str(row.get("lastName") or "").strip().lower()
        full = " ".join([first, last]).strip()
        score = 0
        if q and full == q:
            score += 1000
        if q and q in full:
            score += 200
        if q_tokens:
            score += 60 * sum(1 for t in q_tokens if t == first or t == last)
            score += 20 * sum(1 for t in q_tokens if t in full)
        return (score, int(row.get("numRatings") or 0), float(row.get("avgRating") or 0.0))

    out = sorted(out, key=_score, reverse=True)[: int(limit)]
    for row in out:
        row["comments"] = get_professor_comments_by_teacher_id(row.get("id"), limit=3)
        if not row["comments"]:
            row["comments"] = get_professor_comments_by_legacy_id(row.get("legacyId"), limit=3)
    return out


def get_university(university: str, limit: int = 5) -> List[JSON]:
    """Search schools by free-text name."""
    query = load_query_from_file("search_university_by_name")
    variables = {"query": {"text": university}}
    resp = send_graphql_request(query, variables)
    return format_university_response(resp, limit)


def get_professors_by_university_id(
    school_id: str, professor_name: str, limit: int = 5
) -> List[JSON]:
    """Search teachers within a school by free-text name and school ID."""
    query = load_query_from_file("search_teachers_by_school_id")
    variables = {"query": {"text": professor_name, "schoolID": school_id}, "count": limit}
    resp = send_graphql_request(query, variables)
    out = format_professor_response(resp, limit)
    for row in out:
        row["comments"] = get_professor_comments_by_teacher_id(row.get("id"), limit=3)
        if not row["comments"]:
            row["comments"] = get_professor_comments_by_legacy_id(row.get("legacyId"), limit=3)
    return out


def get_top_professors(
    university: str,
    department: str = "",
    sort_by: str = "rating",
    limit: int = 5,
) -> List[JSON]:
    """
    Find top professors at a university, sorted by the requested metric.

    sort_by options:
      "rating"         — highest avgRating first (default)
      "easiest"        — lowest avgDifficulty first (easiest to pass)
      "hardest"        — highest avgDifficulty first
      "popular"        — most numRatings first
      "would_take_again" — highest wouldTakeAgainPercentRounded first
    """
    schools = get_university(university, limit=1)
    if not schools:
        return []
    school_id = schools[0].get("id") or schools[0].get("legacyId")
    if not school_id:
        return []

    fetch_limit = max(limit * 5, 25)
    query = load_query_from_file("search_teachers_by_school_id")
    variables = {
        "query": {"text": department or "", "schoolID": str(school_id)},
        "count": fetch_limit,
    }
    resp = send_graphql_request(query, variables)
    candidates = format_professor_response(resp, fetch_limit)
    # Only include professors who have actual ratings
    candidates = [c for c in candidates if (c.get("numRatings") or 0) > 0]

    sort_key = sort_by.lower().strip()
    if sort_key in ("easiest", "easy", "easy_grader", "lenient"):
        # Lower avgDifficulty = easier (ascending, but exclude professors with no difficulty data)
        with_diff = [c for c in candidates if c.get("avgDifficulty") is not None]
        without_diff = [c for c in candidates if c.get("avgDifficulty") is None]
        with_diff.sort(key=lambda r: float(r.get("avgDifficulty") or 2.5))
        candidates = with_diff + without_diff
    elif sort_key in ("hardest", "hard", "most_difficult", "difficult"):
        with_diff = [c for c in candidates if c.get("avgDifficulty") is not None]
        without_diff = [c for c in candidates if c.get("avgDifficulty") is None]
        with_diff.sort(key=lambda r: float(r.get("avgDifficulty") or 2.5), reverse=True)
        candidates = with_diff + without_diff
    elif sort_key in ("popular", "most_rated", "most_reviewed", "trending"):
        candidates.sort(key=lambda r: int(r.get("numRatings") or 0), reverse=True)
    elif sort_key in ("would_take_again", "would take again", "loved", "recommended"):
        candidates.sort(
            key=lambda r: float(r.get("wouldTakeAgainPercentRounded") or 0),
            reverse=True,
        )
    else:
        # Default: highest avgRating, tie-break by numRatings
        candidates.sort(
            key=lambda r: (float(r.get("avgRating") or 0), int(r.get("numRatings") or 0)),
            reverse=True,
        )

    top = candidates[:limit]
    for row in top:
        row["comments"] = get_professor_comments_by_teacher_id(row.get("id"), limit=3)
        if not row["comments"]:
            row["comments"] = get_professor_comments_by_legacy_id(row.get("legacyId"), limit=3)
    return top


def get_departments_for_school(university: str) -> List[str]:
    """Return a sorted list of department names for a university from RMP."""
    schools = get_university(university, limit=1)
    if not schools:
        return []
    depts = schools[0].get("departments") or []
    return sorted([d.get("name") for d in depts if d.get("name")])


# =============================================================================
# Query parsing helpers for direct routing
# =============================================================================

# Known abbreviations for universities
_SCHOOL_ABBREVS: List[tuple[str, str]] = [
    ("CSULB", "California State University Long Beach"),
    ("CSUN", "California State University Northridge"),
    ("CSUF", "California State University Fullerton"),
    ("SDSU", "San Diego State University"),
    ("UCSD", "UC San Diego"),
    ("UCSB", "UC Santa Barbara"),
    ("UCIRVINE", "UC Irvine"),
    ("UCI", "UC Irvine"),
    ("UCD", "UC Davis"),
    ("UCB", "UC Berkeley"),
    ("UCLA", "UCLA"),
    ("USC", "USC"),
    ("MIT", "MIT"),
    ("NYU", "NYU"),
    ("ASU", "Arizona State University"),
    ("OSU", "Ohio State University"),
    ("PSU", "Penn State University"),
    ("UW", "University of Washington"),
    ("UIUC", "University of Illinois Urbana Champaign"),
    ("UT", "University of Texas Austin"),
    ("UF", "University of Florida"),
    ("UM", "University of Michigan"),
]

# Department keyword → normalized name
_DEPT_KEYWORDS: List[tuple[str, str]] = [
    ("computer science", "Computer Science"),
    (" cs ", "Computer Science"),
    ("mathematics", "Mathematics"),
    (" math ", "Mathematics"),
    ("calculus", "Mathematics"),
    ("biology", "Biology"),
    ("chemistry", "Chemistry"),
    ("organic chemistry", "Chemistry"),
    ("physics", "Physics"),
    ("english", "English"),
    ("history", "History"),
    ("psychology", "Psychology"),
    ("economics", "Economics"),
    ("engineering", "Engineering"),
    ("electrical engineering", "Electrical Engineering"),
    ("mechanical engineering", "Mechanical Engineering"),
    ("civil engineering", "Civil Engineering"),
    ("business", "Business"),
    ("political science", "Political Science"),
    ("sociology", "Sociology"),
    ("philosophy", "Philosophy"),
    ("art", "Art"),
    ("music", "Music"),
    ("nursing", "Nursing"),
    ("education", "Education"),
    ("communications", "Communications"),
    ("statistics", "Statistics"),
    ("data science", "Computer Science"),
    ("machine learning", "Computer Science"),
    ("artificial intelligence", "Computer Science"),
]

_TOP_RATED_RE = re.compile(
    r"\b(best|top|highest[\s-]?rated|top[\s-]?rated|easiest|hardest|most\s+popular|most\s+recommended)\b",
    re.IGNORECASE,
)
_EASY_RE = re.compile(
    r"\b(easiest?|easy\s+grader|lenient|low\s+difficulty|not\s+hard|chill\s+professor)\b",
    re.IGNORECASE,
)
_HARD_RE = re.compile(
    r"\b(hardest?|most\s+difficult|strict\s+grader|tough\s+grader|challenging)\b",
    re.IGNORECASE,
)
_POPULAR_RE = re.compile(
    r"\b(most\s+popular|most\s+rated|most\s+reviewed|most\s+known|trending)\b",
    re.IGNORECASE,
)
_LOVED_RE = re.compile(
    r"\b(most\s+loved|students?\s+love|highly\s+recommended|best\s+liked|most\s+recommended)\b",
    re.IGNORECASE,
)
_DEPT_LIST_RE = re.compile(
    r"\b(what\s+departments?|list\s+departments?|available\s+departments?|departments?\s+(?:at|in|for)|show\s+departments?)\b",
    re.IGNORECASE,
)


def _extract_school_name(query: str) -> str:
    """Extract a university name from a natural language query."""
    # Check abbreviations first (exact word-boundary match)
    for abbr, full in _SCHOOL_ABBREVS:
        if re.search(r"\b" + re.escape(abbr) + r"\b", query, re.IGNORECASE):
            return full

    # "at/from [Capitalized Words]" pattern
    m = re.search(
        r"\b(?:at|from)\s+((?:[A-Z][A-Za-z]+\s+){0,5}[A-Z][A-Za-z]+)",
        query,
    )
    if m:
        candidate = m.group(1).strip()
        # Filter out obvious non-school words
        _skip = {"The", "A", "An", "He", "She", "It", "They", "His", "Her", "This", "That"}
        if len(candidate) >= 4 and candidate not in _skip:
            return candidate

    return ""


def _extract_department(query: str) -> str:
    """Extract department/subject from query."""
    q_lower = " " + query.lower() + " "
    # Try longest match first
    for keyword, normalized in sorted(_DEPT_KEYWORDS, key=lambda x: len(x[0]), reverse=True):
        if keyword.lower() in q_lower:
            return normalized
    return ""


def _extract_course_number(query: str) -> tuple[str, str]:
    """
    Extract department code + course number from queries like 'CS 462', 'MATH 201A'.
    Returns (dept_code, course_number) or ('', '').
    """
    _FALSE_POS = {'THE', 'AND', 'FOR', 'NOT', 'BUT', 'ARE', 'WAS', 'HAS',
                  'TOP', 'WHO', 'WHY', 'HOW', 'CAN', 'DID', 'GET', 'GOT'}
    m = re.search(r'\b([A-Za-z]{2,6})\s*(\d{2,4}[A-Za-z]?)\b', query)
    if m:
        code = m.group(1).upper()
        if code not in _FALSE_POS:
            return code, m.group(2).upper()
    return '', ''


# Map 2-6 letter dept codes → full department name
_COURSE_CODE_MAP: dict[str, str] = {
    'CS': 'Computer Science', 'CECS': 'Computer Science', 'CMPSC': 'Computer Science',
    'MATH': 'Mathematics', 'MAT': 'Mathematics',
    'PHYS': 'Physics', 'PHY': 'Physics',
    'CHEM': 'Chemistry', 'CHE': 'Chemistry',
    'BIO': 'Biology', 'BIOL': 'Biology',
    'ENGL': 'English', 'ENG': 'English',
    'HIST': 'History',
    'PSYC': 'Psychology', 'PSY': 'Psychology',
    'ECON': 'Economics', 'ECO': 'Economics',
    'BUS': 'Business', 'MGMT': 'Business', 'MBA': 'Business',
    'POLS': 'Political Science', 'POLI': 'Political Science',
    'SOC': 'Sociology',
    'PHIL': 'Philosophy',
    'ART': 'Art',
    'MUS': 'Music',
    'NURS': 'Nursing',
    'EDUC': 'Education',
    'COMM': 'Communications',
    'STAT': 'Statistics',
    'ECE': 'Electrical Engineering', 'EE': 'Electrical Engineering',
    'ME': 'Mechanical Engineering', 'MECH': 'Mechanical Engineering',
    'CE': 'Civil Engineering', 'CIV': 'Civil Engineering',
}


def _extract_min_rating(query: str) -> float | None:
    """
    Extract a minimum rating threshold from natural language.
    e.g. "rating above 4", "at least 4.5 stars", "4+ rating", "rated over 3.5"
    """
    patterns = [
        r'rating\s+(?:above|over|greater\s+than|at\s+least|more\s+than|>=?)\s*(\d+(?:\.\d+)?)',
        r'(?:above|over|at\s+least|>=?)\s*(\d+(?:\.\d+)?)\s*(?:stars?|rating)',
        r'(\d+(?:\.\d+)?)\+\s*(?:stars?|rating)',
        r'(?:minimum|min)\s+(?:rating\s+(?:of\s+)?)?(\d+(?:\.\d+)?)',
        r'rated\s+(?:above|over|at\s+least)\s+(\d+(?:\.\d+)?)',
    ]
    for pat in patterns:
        m = re.search(pat, query, re.IGNORECASE)
        if m:
            val = float(m.group(1))
            if 0.0 <= val <= 5.0:
                return val
    return None


def _extract_wta_filter(query: str) -> float | None:
    """
    Extract a minimum 'would take again' percentage from natural language.
    e.g. "would take again above 80%", "students love them (90% would retake)"
    """
    patterns = [
        r'would\s+take\s+again\s+(?:above|over|at\s+least|>=?)\s*(\d+)%?',
        r'(\d+)%?\s+would\s+take\s+again',
        r'retake\s+(?:rate\s+)?(?:above|over|at\s+least)\s*(\d+)%?',
    ]
    for pat in patterns:
        m = re.search(pat, query, re.IGNORECASE)
        if m:
            val = float(m.group(1))
            if 0 <= val <= 100:
                return val
    return None


def _suggest_fuzzy_name(name: str, rows: List[JSON]) -> List[str]:
    """
    Given a queried name and candidate professor rows, return close-matching full names
    using difflib (handles single-letter typos, transpositions, etc.).
    """
    import difflib
    candidate_names = []
    for row in rows:
        fn = (row.get('firstName') or '').strip()
        ln = (row.get('lastName') or '').strip()
        full = f"{fn} {ln}".strip()
        if full:
            candidate_names.append(full)
    return difflib.get_close_matches(name, candidate_names, n=3, cutoff=0.55)


def _extract_professor_name_from_query(query: str) -> str:
    """
    Extract a professor name from a user query string.
    Returns an empty string if no clear name is found.

    All name patterns are applied case-sensitively ([A-Z][a-z]+) so that lowercase
    prepositions like 'at', 'from', 'in' are never captured as part of the name.
    """
    _UNI_WORDS = {"California", "State", "University", "College", "Institute",
                  "North", "South", "East", "West", "New", "Los", "San"}

    # Pattern for a capitalized name: one or more "Word" tokens, all title-cased
    _CAP_NAME = r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}"

    # 0. Middle initial: "Jeffrey H. Cohen" → "Jeffrey Cohen"
    m = re.search(r"\b([A-Z][a-z]+)\s+[A-Z]\.\s*([A-Z][a-z]+)\b", query)
    if m:
        candidate = f"{m.group(1)} {m.group(2)}"
        if not any(w in _UNI_WORDS for w in candidate.split()):
            return candidate

    # 1. Quoted name: "Todd Ebert" or 'Todd Ebert'
    m = re.search(r'["\'](' + _CAP_NAME + r')["\']', query)
    if m:
        return m.group(1)

    # 2. "professor/prof/dr <Name>" — find keyword, then pick up the name after it
    m = re.search(r"(?i)\b(?:professor|prof\.?|dr\.?|instructor)\s+", query)
    if m:
        after = query[m.end():]
        nm = re.match(_CAP_NAME, after)
        if nm:
            return nm.group(0)

    # 3. "<Name> at/from <School>" — name followed by a school preposition
    m = re.search(r"^(" + _CAP_NAME + r")\s+(?:at|from|in)\s+[A-Z]", query)
    if m:
        candidate = m.group(1)
        if not any(w in _UNI_WORDS for w in candidate.split()):
            return candidate

    # 4. "find/search/about/rate <Name>"
    m = re.search(r"(?i)\b(?:find|search|rate|look\s+up|tell\s+me\s+about)\s+", query)
    if m:
        after = query[m.end():]
        nm = re.match(_CAP_NAME, after)
        if nm:
            candidate = nm.group(0)
            if not any(w in _UNI_WORDS for w in candidate.split()):
                return candidate

    # 4b. Preposition + Name: "similar to <Name>", "by <Name>", "about <Name>",
    #     "for <Name>", "of <Name>", "like <Name>" — covers research/similar queries
    m = re.search(
        r"(?i)\b(?:similar\s+to|by|about|for|of|like|alternatives?\s+to)\s+(" + _CAP_NAME + r")\b",
        query,
    )
    if m:
        candidate = m.group(1)
        if not any(w in _UNI_WORDS for w in candidate.split()):
            return candidate

    # 5. Bare capitalized 2-3 word query (the entire query is the name)
    bare = query.strip()
    if re.match(r"^[A-Z][a-z]+(\s+[A-Z][a-z]+){1,2}$", bare):
        return bare

    # 6. Last-resort: first capitalized 2-word sequence not containing university words
    for m in re.finditer(r"\b([A-Z][a-z]+\s+[A-Z][a-z]+)\b", query):
        candidate = m.group(1)
        if not any(w in _UNI_WORDS for w in candidate.split()):
            return candidate

    return ""


def _format_professor_rows(rows: List[JSON], title: str = "") -> str:
    """
    Format professor rows into markdown that the frontend card parsers understand.

    Produces:
        [title]

        1. **Name**
        - Department: X
        - University: X
        - City: X, State
        - Average Rating: X/5 (N ratings)
        - Would Take Again: X%

        Student comments for Name:
        - comment1
        - comment2
    """
    if not rows:
        return "I couldn't find any professors matching your query."

    parts: List[str] = []
    if title:
        parts.append(title)
        parts.append("")

    for i, row in enumerate(rows, 1):
        first = (row.get("firstName") or "").strip()
        last = (row.get("lastName") or "").strip()
        name = f"{first} {last}".strip() or "Unknown"
        dept = row.get("department") or "N/A"
        school = row.get("school") or "N/A"
        city = (row.get("schoolCity") or "").strip()
        state = (row.get("schoolState") or "").strip()
        location = ", ".join(filter(None, [city, state])) or "N/A"
        rating = row.get("avgRating")
        n_ratings = int(row.get("numRatings") or 0)
        wta = row.get("wouldTakeAgainPercentRounded")

        rating_str = f"{rating}/5 ({n_ratings} ratings)" if rating else "N/A"
        wta_str = f"{wta}%" if wta is not None and wta >= 0 else "N/A"

        difficulty = row.get("avgDifficulty")
        if difficulty is not None:
            diff_label = "Easy" if float(difficulty) <= 2.0 else "Medium" if float(difficulty) <= 3.5 else "Hard"
            difficulty_str = f"{float(difficulty):.1f}/5 ({diff_label})"
        else:
            difficulty_str = None

        legacy_id = row.get("legacyId")
        profile_url = (
            f"https://www.ratemyprofessors.com/professor/{legacy_id}"
            if legacy_id else ""
        )

        parts.append(f"{i}. **{name}**")
        parts.append(f"- Department: {dept}")
        parts.append(f"- University: {school}")
        parts.append(f"- City: {location}")
        parts.append(f"- Average Rating: {rating_str}")
        if difficulty_str:
            parts.append(f"- Difficulty: {difficulty_str}")
        parts.append(f"- Would Take Again: {wta_str}")
        if profile_url:
            parts.append(f"- Profile: {profile_url}")
        parts.append("")

    # Include comments for the first result
    top = rows[0]
    comments = [c for c in (top.get("comments") or []) if isinstance(c, str) and c.strip()][:3]
    if comments:
        first = (top.get("firstName") or "").strip()
        last = (top.get("lastName") or "").strip()
        top_name = f"{first} {last}".strip()
        parts.append(f"Student comments for {top_name}:")
        for c in comments:
            parts.append(f"- {c}")

    return "\n".join(parts)


# =============================================================================
# Direct routing entry point — replaces AgentExecutor
# =============================================================================

def run_direct(query: str, chat_history: List[Any]) -> str:
    """
    Route the query directly to the right RMP function.

    Fixes vs v1:
      - Top-rated with no school gives a helpful prompt instead of garbage
      - Course number queries (CS 462) route to dept professors
      - Rating filter ("above 4.0") applied post-fetch
      - Disambiguation: multiple different professors → ask user to clarify
      - Fuzzy name suggestion when exact search returns nothing
      - Fetches up to 10 results (frontend handles "show more" locally)
    """
    school  = _extract_school_name(query)
    dept    = _extract_department(query)
    name    = _extract_professor_name_from_query(query)
    min_rating = _extract_min_rating(query)
    min_wta    = _extract_wta_filter(query)
    course_code, course_num = _extract_course_number(query)

    def _apply_filters(rows: List[JSON]) -> List[JSON]:
        if min_rating is not None:
            rows = [r for r in rows if float(r.get("avgRating") or 0) >= min_rating]
        if min_wta is not None:
            rows = [r for r in rows if float(r.get("wouldTakeAgainPercentRounded") or 0) >= min_wta]
        return rows

    # Backwards compat alias
    _apply_rating_filter = _apply_filters

    # ── 0a. Department listing ("what departments does MIT have?") ───────────
    if _DEPT_LIST_RE.search(query) and school:
        depts = get_departments_for_school(school)
        if depts:
            dept_list = "\n".join(f"- {d}" for d in depts)
            return (
                f"Here are the departments available at **{school}** on RateMyProfessors:\n\n"
                f"{dept_list}\n\n"
                f"You can ask about professors in any of these departments!"
            )
        return f"I couldn't find department info for **{school}**."

    # ── 0b. Course number query (e.g. "CS 462 at CSULB") ────────────────────
    if course_code and course_num:
        full_dept = _COURSE_CODE_MAP.get(course_code, course_code)
        if school:
            schools = get_university(school, limit=1)
            if schools:
                sid = schools[0].get("id") or str(schools[0].get("legacyId", ""))
                if sid:
                    rows = get_professors_by_university_id(sid, full_dept, limit=10)
                    rows = _apply_filters(rows)
                    if rows:
                        return _format_professor_rows(
                            rows,
                            title=(
                                f"Here are professors who may teach "
                                f"{course_code} {course_num} ({full_dept}) at {school}:"
                            ),
                        )
        # No school — search globally by dept
        rows = get_professor(full_dept, limit=8)
        rows = _apply_filters(rows)
        if rows:
            rows.sort(key=lambda r: float(r.get("avgRating") or 0), reverse=True)
            return _format_professor_rows(
                rows[:5],
                title=f"Here are top-rated {full_dept} professors (for {course_code} courses):",
            )
        return (
            f"I couldn't find professors for {course_code} {course_num}. "
            "Try specifying the university too."
        )

    # ── 1. Sorted/top query at a specific university ─────────────────────────
    is_sort_query = bool(
        _TOP_RATED_RE.search(query) or _EASY_RE.search(query) or
        _HARD_RE.search(query) or _POPULAR_RE.search(query) or _LOVED_RE.search(query)
    )
    if is_sort_query and school:
        # Pick the right sort mode
        if _EASY_RE.search(query):
            sort_by, label = "easiest", "easiest"
        elif _HARD_RE.search(query):
            sort_by, label = "hardest", "hardest"
        elif _POPULAR_RE.search(query):
            sort_by, label = "popular", "most popular"
        elif _LOVED_RE.search(query):
            sort_by, label = "would_take_again", "most loved"
        else:
            sort_by, label = "rating", "top-rated"

        rows = get_top_professors(school, department=dept, sort_by=sort_by, limit=10)
        rows = _apply_filters(rows)
        if rows:
            title = f"Here are the {label} professors at {school}"
            if dept:
                title += f" for {dept}"
            title += ":"
            return _format_professor_rows(rows, title=title)
        if min_rating:
            return (
                f"No professors at {school} with a rating ≥ {min_rating} found. "
                "Try lowering the threshold."
            )
        return (
            f"I couldn't find professors at {school}. "
            "The university may not be listed on RateMyProfessors, or try the full name."
        )

    # ── 2. Sort query with no school → helpful prompt ────────────────────────
    if is_sort_query and not school:
        if dept:
            return (
                f"To find the best **{dept}** professors I need a university name. "
                f"Try: *\"best {dept} professors at [University]\"*"
            )
        return (
            "To find top-rated or easiest professors please specify a university. "
            "Example: *\"easiest CS professors at UCLA\"* or *\"best professors at MIT\"*"
        )

    # ── 3. Specific professor name ────────────────────────────────────────────
    if name:
        rows = get_professor(name, limit=8)
        rows = _apply_filters(rows)

        if rows:
            # If school mentioned, prefer that school's professors
            if school:
                school_lower = school.lower()
                matching = [r for r in rows if school_lower in (r.get("school") or "").lower()]
                if matching:
                    rows = matching

            # Detect disambiguation: multiple genuinely different people
            unique_people: dict[str, JSON] = {}
            for r in rows:
                fn = (r.get("firstName") or "").strip()
                ln = (r.get("lastName") or "").strip()
                key = f"{fn} {ln}".strip().lower()
                if key not in unique_people:
                    unique_people[key] = r

            if len(unique_people) > 1:
                # Check that they differ by more than just university
                options: List[str] = []
                for i, (_, r) in enumerate(list(unique_people.items())[:4], 1):
                    fn = (r.get("firstName") or "").strip()
                    ln = (r.get("lastName") or "").strip()
                    display = f"{fn} {ln}".strip()
                    school_name = r.get("school") or "Unknown University"
                    rating = r.get("avgRating") or "N/A"
                    n_ratings = r.get("numRatings") or 0
                    options.append(
                        f"{i}. **{display}** — {school_name} "
                        f"({rating}/5, {n_ratings} ratings)"
                    )
                return (
                    f"I found **{len(unique_people)} different professors** named '{name}':\n\n"
                    + "\n".join(options)
                    + "\n\nWhich one did you mean? "
                    "Reply with their number, or add their university "
                    "(e.g. *'Todd Ebert at CSULB'*)."
                )

            # Single person (possibly at multiple schools) — show top 3 listings
            return _format_professor_rows(rows[:3])

        # No results — try fuzzy suggestion via last-name search
        last_name = name.split()[-1] if name.split() else name
        broader = get_professor(last_name, limit=12)
        suggestions = _suggest_fuzzy_name(name, broader)
        if suggestions:
            suggestion_list = "\n".join(f"- {s}" for s in suggestions)
            return (
                f"I couldn't find **'{name}'** exactly. Did you mean:\n\n"
                f"{suggestion_list}\n\n"
                "Try the full name or add their university."
            )
        return (
            f"I couldn't find a professor named **'{name}'** on RateMyProfessors. "
            "Check the spelling or try their full name."
        )

    # ── 4. Professors at a school (no specific name or top-rated) ────────────
    if school:
        schools = get_university(school, limit=1)
        if schools:
            sid = schools[0].get("id") or str(schools[0].get("legacyId", ""))
            if sid:
                rows = get_professors_by_university_id(sid, dept or "", limit=10)
                rows = _apply_filters(rows)
                if rows:
                    return _format_professor_rows(
                        rows, title=f"Here are professors at {school}:"
                    )
        return f"I couldn't find **'{school}'** on RateMyProfessors."

    # ── 5. Last resort: raw query search ─────────────────────────────────────
    rows = get_professor(query, limit=5)
    rows = _apply_filters(rows)
    if rows:
        return _format_professor_rows(rows)

    return (
        "I couldn't find any professors matching your query. "
        "Try including the professor's full name, "
        "or specify a university (e.g. *'Todd Ebert at CSULB'*)."
    )


# Backward-compatible alias — anything that still calls run_tools() gets run_direct()
run_tools = run_direct


# =============================================================================
# LangChain tool wrappers (kept for potential future use)
# =============================================================================

class GetProfessorArgs(BaseModel):
    name: str = Field(..., title="The name of the professor to be searched.")
    limit: int = Field(5, title="Max number of results to return.", ge=1, le=20)


class GetUniversityArgs(BaseModel):
    university: str = Field(..., title="The name of the university to be searched.")
    limit: int = Field(5, title="Max number of results to return.", ge=1, le=20)


class GetProfessorsByUniversityIDArgs(BaseModel):
    school_id: str = Field(..., title="The ID of the school.")
    professor_name: str = Field(..., title="Free-text filter for the professor name.")
    limit: int = Field(5, title="Max number of results to return.", ge=1, le=20)


class GetTopProfessorsArgs(BaseModel):
    university: str = Field(..., title="University name to search within.")
    department: str = Field("", title="Optional department filter (e.g. 'Computer Science').")
    sort_by: str = Field(
        "rating",
        title="Sort metric: 'rating' | 'would_take_again'. Default: rating.",
    )
    limit: int = Field(5, title="Max number of results to return.", ge=1, le=20)


rate_tools = [
    StructuredTool.from_function(
        func=get_professor,
        name="GetProfessor",
        description="Search professors globally by name.",
        args_schema=GetProfessorArgs,
    ),
    StructuredTool.from_function(
        func=get_university,
        name="GetUniversity",
        description="Search universities by name to get their ID.",
        args_schema=GetUniversityArgs,
    ),
    StructuredTool.from_function(
        func=get_professors_by_university_id,
        name="GetProfessorsByUniversityID",
        description="Search professors within a university by school_id and optional name filter.",
        args_schema=GetProfessorsByUniversityIDArgs,
    ),
    StructuredTool.from_function(
        func=get_top_professors,
        name="GetTopProfessors",
        description=(
            "Find the top-rated professors at a university, optionally filtered by department. "
            "sort_by options: 'rating' (default) or 'would_take_again'."
        ),
        args_schema=GetTopProfessorsArgs,
    ),
]


# =============================================================================
# Manual quick test
# =============================================================================

if __name__ == "__main__":
    print("Auth present?", bool(_AUTHORIZATION_FULL or _AUTH_TOKEN_ONLY))
    print("Trying run_direct('Todd Ebert at CSULB')...")
    print(run_direct("Find professor Todd Ebert at CSULB", []))
