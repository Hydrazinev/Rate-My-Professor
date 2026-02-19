from __future__ import annotations

# =============================================================================
# RateMyProfessors LangChain tools & thin GraphQL client
# -----------------------------------------------------------------------------
# Behavior is intentionally unchanged. This refactor focuses on:
# - Code organization & readability
# - Idiomatic naming & structure
# - Consistent formatting and comments
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
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain.agents import AgentExecutor, create_tool_calling_agent


# =============================================================================
# Environment & constants
# =============================================================================

load_dotenv()

# Public RMP GraphQL endpoint
BASE_URL: str = os.getenv("RMP_BASE_URL", "https://www.ratemyprofessors.com/graphql")

# Authorization can be provided in either of these ways:
# 1) RMP_AUTHORIZATION="Bearer <JWT>" (or "Basic <base64>")
# 2) RMP_AUTH_TOKEN="<token>" and optional RMP_AUTH_SCHEME="Bearer" (default)
_AUTHORIZATION_FULL: str = (os.getenv("RMP_AUTHORIZATION") or "").strip()
_AUTH_TOKEN_ONLY: str = (os.getenv("RMP_AUTH_TOKEN") or "").strip()
_AUTH_SCHEME: str = (os.getenv("RMP_AUTH_SCHEME") or "Bearer").strip()

# Optional cookie (helpful sometimes). Provide only the JWT value from the rmpAuth cookie.
_RMP_COOKIE: str = (os.getenv("RMP_RMPAUTH_COOKIE") or "").strip()

# Use a session that ignores system proxy env; local bad proxies can break requests.
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
    """
    Build the Authorization header value from environment variables.

    Raises:
        RuntimeError: If no suitable auth configuration is present.
    """
    if _AUTHORIZATION_FULL:
        # User provided full header value like "Bearer ey..." or "Basic abc..."
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
# GraphQL query loader (robust to filename variants & working dir)
# =============================================================================

_THIS_DIR = Path(__file__).resolve().parent

# Map "logical" names to observed on-disk filename variants.
_QUERY_CANDIDATES: Dict[str, List[str]] = {
    "search_professor_by_name": ["search_professor_by_name.graphql"],
    "search_university_by_name": ["search_university_by_name.graphql"],
    "search_teachers_by_school_id": [
        "search_teachers_by_school_id.graphql",
        # Alternate naming seen in some repos:
        "search_professor_by_school_id.graphql",
    ],
}


def _resolve_query_path(preferred_name: str) -> Path:
    """
    Resolve a .graphql file from likely locations and filename variants.

    Search order:
      - this file's folder (tools/)
      - parent folder (project root candidate)
      - parent/tools (defensive fallback)
    """
    candidates = _QUERY_CANDIDATES.get(preferred_name, [preferred_name])
    search_dirs = [
        _THIS_DIR,
        _THIS_DIR.parent,
        _THIS_DIR.parent / "tools",
    ]
    for fname in candidates:
        for d in search_dirs:
            p = d / fname
            if p.exists():
                return p
    # If nothing found, return a path that will raise a clearer error later.
    return _THIS_DIR / candidates[0]


def load_query_from_file(logical_name: str) -> str:
    """
    Load a GraphQL query by logical name (e.g., 'search_professor_by_name').

    Raises:
        FileNotFoundError: if a matching .graphql file cannot be found.
    """
    path = _resolve_query_path(logical_name)
    if not path.exists():
        tried = ", ".join(_QUERY_CANDIDATES.get(logical_name, [logical_name]))
        raise FileNotFoundError(
            f"GraphQL file not found for '{logical_name}'. "
            f"Tried names: {tried} in {_THIS_DIR}, its parent, and 'tools/'."
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
    """
    Send a GraphQL request to RateMyProfessors.

    Returns:
        dict: Parsed JSON. On HTTP/JSON/GraphQL error, returns {"error": "<message>"}.
    """
    payload: GraphQLPayload = {"query": query, "variables": variables or {}}
    try:
        resp = _HTTP.post(
            BASE_URL,
            headers=_default_headers(),
            data=json.dumps(payload),
            timeout=timeout,
        )
        # Some GraphQL servers return 200 OK even when the body contains errors.
        resp.raise_for_status()
        data: JSON = resp.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"HTTP error: {e}"}
    except json.JSONDecodeError:
        return {"error": "Invalid JSON in response"}

    # Surface GraphQL "errors" array as a normalized error.
    if isinstance(data, dict) and data.get("errors"):
        return {"error": f"GraphQL errors: {data['errors']}"}

    return data


# =============================================================================
# Response formatters (normalize API shapes)
# =============================================================================

def format_university_response(response: JSON, limit: int) -> List[JSON]:
    """
    Normalize the 'schools' search result.
    """
    out: List[JSON] = []
    if not isinstance(response, dict) or "error" in response:
        return out

    new = response.get("data", {}).get("newSearch", {}) or {}
    edges = (new.get("schools", {}) or {}).get("edges", []) or []
    for edge in edges[:limit]:
        node = edge.get("node", {}) or {}
        out.append(
            {
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
            }
        )
    return out


def format_professor_response(response: JSON, limit: int) -> List[JSON]:
    """
    Normalize the 'teachers' search result.
    """
    out: List[JSON] = []
    if not isinstance(response, dict) or "error" in response:
        return out

    new = response.get("data", {}).get("newSearch", {}) or {}
    edges = (new.get("teachers", {}) or {}).get("edges", []) or []
    for edge in edges[:limit]:
        node = edge.get("node", {}) or {}
        school = node.get("school") or {}
        out.append(
            {
                "id": node.get("id"),
                "legacyId": node.get("legacyId"),
                "firstName": node.get("firstName"),
                "lastName": node.get("lastName"),
                "department": node.get("department"),
                "school": school.get("name"),
                "schoolCity": school.get("city"),
                "schoolState": school.get("state"),
                "avgRating": node.get("avgRating"),
                "numRatings": node.get("numRatings"),
                "wouldTakeAgainPercentRounded": node.get(
                    "wouldTakeAgainPercentRounded"
                ),
            }
        )
    return out


def _extract_comments_from_professor_page(page_html: str, limit: int = 3) -> List[str]:
    """
    Extract rating comments from the public professor page HTML.

    This is intentionally best-effort because the page structure can change.
    """
    if not page_html:
        return []

    # Pull comment text from JSON blobs embedded in the page.
    # Keep multiple patterns because frontend payload keys can vary.
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
        if len(txt) < 8:
            continue
        if txt.lower() in seen:
            continue
        seen.add(txt.lower())
        cleaned.append(txt)
        if len(cleaned) >= limit:
            break

    return cleaned


def _teacher_id_from_legacy_id(legacy_id: Any) -> str:
    """Convert legacy numeric ID to GraphQL Teacher node ID."""
    try:
        raw = f"Teacher-{int(legacy_id)}".encode("utf-8")
    except Exception:
        return ""
    return base64.b64encode(raw).decode("utf-8")


def get_professor_comments_by_teacher_id(teacher_id: Any, limit: int = 3) -> List[str]:
    """
    Fetch recent student comments directly from GraphQL Teacher.ratings.
    Returns [] on any failure.
    """
    if not teacher_id:
        return []
    query = """
    query TeacherComments($id: ID!, $count: Int!) {
      node(id: $id) {
        __typename
        ... on Teacher {
          ratings(first: $count) {
            edges {
              node {
                comment
                class
                date
                qualityRating
              }
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
        resp.get("data", {})
        .get("node", {})
        .get("ratings", {})
        .get("edges", [])
        or []
    )
    comments: List[str] = []
    seen: set[str] = set()
    for edge in edges:
        node = (edge or {}).get("node", {}) or {}
        txt = (node.get("comment") or "").strip()
        if not txt:
            continue
        norm = re.sub(r"\s+", " ", txt).strip()
        if len(norm) < 8:
            continue
        key = norm.lower()
        if key in seen:
            continue
        seen.add(key)
        comments.append(norm)
        if len(comments) >= limit:
            break
    return comments


def get_professor_comments_by_legacy_id(legacy_id: Any, limit: int = 3) -> List[str]:
    """
    Compatibility helper. Prefer GraphQL comments by teacher ID.
    Returns [] on any failure.
    """
    teacher_id = _teacher_id_from_legacy_id(legacy_id)
    comments = get_professor_comments_by_teacher_id(teacher_id, limit=limit)
    if comments:
        return comments
    # Fallback to legacy HTML scraping only if GraphQL fails.
    if not legacy_id:
        return []
    try:
        url = f"https://www.ratemyprofessors.com/professor/{legacy_id}"
        resp = _HTTP.get(
            url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/127.0.0.0 Safari/537.36"
                )
            },
            timeout=15,
        )
        if resp.ok:
            return _extract_comments_from_professor_page(resp.text, limit=limit)
    except Exception:
        pass
    return []


# =============================================================================
# Public search functions (stable external API)
# =============================================================================

def get_professor(name: str, limit: int = 5) -> List[JSON]:
    """
    Search professors by free-text name.
    """
    query = load_query_from_file("search_professor_by_name")
    # Pull a wider candidate set, then rank locally for better name precision.
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
        # Prefer rows with more ratings when textual score ties.
        num_ratings = int(row.get("numRatings") or 0)
        avg = float(row.get("avgRating") or 0.0)
        return (score, num_ratings, avg)

    out = sorted(out, key=_score, reverse=True)[: int(limit)]
    for row in out:
        row["comments"] = get_professor_comments_by_teacher_id(row.get("id"), limit=3)
        if not row["comments"]:
            row["comments"] = get_professor_comments_by_legacy_id(row.get("legacyId"), limit=3)
    return out


def get_university(university: str, limit: int = 5) -> List[JSON]:
    """
    Search schools by free-text name.
    """
    query = load_query_from_file("search_university_by_name")
    variables = {"query": {"text": university}}
    resp = send_graphql_request(query, variables)
    return format_university_response(resp, limit)


def get_professors_by_university_id(
    school_id: str, professor_name: str, limit: int = 5
) -> List[JSON]:
    """
    Search teachers within a school by free-text name and school ID.
    """
    query = load_query_from_file("search_teachers_by_school_id")
    variables = {"query": {"text": professor_name, "schoolID": school_id}, "count": limit}
    resp = send_graphql_request(query, variables)
    out = format_professor_response(resp, limit)
    for row in out:
        row["comments"] = get_professor_comments_by_teacher_id(row.get("id"), limit=3)
        if not row["comments"]:
            row["comments"] = get_professor_comments_by_legacy_id(row.get("legacyId"), limit=3)
    return out


# =============================================================================
# LangChain tool wrappers
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
        description="Search universities by name.",
        args_schema=GetUniversityArgs,
    ),
    StructuredTool.from_function(
        func=get_professors_by_university_id,
        name="GetProfessorsByUniversityID",
        description="Search professors within a university by school_id and text filter.",
        args_schema=GetProfessorsByUniversityIDArgs,
    ),
]

# Small, general-purpose agent that can decide which tool to call.
# NOTE: Keeping import-time construction to preserve current behavior.
_llm = ChatOpenAI(model="gpt-4o-mini")
try:
    _prompt = hub.pull("hwchase17/openai-tools-agent")
except Exception:
    # Fallback prompt for offline/local environments where LangSmith hub is unreachable.
    _prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful professor search assistant. Use tools to find accurate professor and university data.",
            ),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )
_agent = create_tool_calling_agent(llm=_llm, tools=rate_tools, prompt=_prompt)
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=_agent,
    tools=rate_tools,
    verbose=True,
    handle_parsing_errors=True,
)


def run_tools(input: str, chat_history: List[Any]) -> str:
    """
    Run the tools using a simple tool-calling agent. Returns a text answer.
    """
    out = agent_executor.invoke({"input": input, "chat_history": chat_history})
    text = out.get("output", "") or ""

    def _extract_professor_names(output_text: str, query_text: str) -> List[str]:
        names: List[str] = []

        # 1) Numbered markdown list: 1. **Dr. Jane Doe**
        names.extend(re.findall(r"(?:^|\n)\s*\d+\.\s+\*\*([^*\n]+)\*\*", output_text))

        # 2) Narrative: "found a professor named Dr. Jane Doe:"
        names.extend(re.findall(r"found\s+a\s+professor\s+named\s+([^\n:]+)", output_text, flags=re.IGNORECASE))
        names.extend(re.findall(r"found\s+(?:dr\.?|professor)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})", output_text, flags=re.IGNORECASE))
        names.extend(re.findall(r"details\s+for\s+professor\s+([^\n:]+)", output_text, flags=re.IGNORECASE))

        # 3) Bullet fields: - Name: Jane Doe or - **Name:** Jane Doe
        names.extend(
            re.findall(
                r"^\s*[-*]\s+\*{0,2}\s*Name\s*\*{0,2}\s*:\s*(.+)$",
                output_text,
                flags=re.IGNORECASE | re.MULTILINE,
            )
        )

        # 4) Standalone lines containing a professor title.
        names.extend(
            re.findall(
                r"^\s*((?:Dr\.?|Professor)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\s*$",
                output_text,
                flags=re.MULTILINE,
            )
        )

        # 5) Fallback from query itself: professor "Name" / professor Name
        quoted = re.findall(r'professor\s+"([^"]+)"', query_text, flags=re.IGNORECASE)
        names.extend(quoted)
        plain = re.findall(
            r"(?:professor|dr\.?)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})",
            query_text,
            flags=re.IGNORECASE,
        )
        names.extend(plain)

        cleaned: List[str] = []
        seen: set[str] = set()
        for n in names:
            name = re.sub(r"\s+", " ", (n or "").strip().strip("-").strip())
            name = re.sub(r"^(?:Name\s*:\s*)", "", name, flags=re.IGNORECASE).strip()
            # Trim sentence tails/noise.
            name = re.split(r"\s*(?:\.|,|;|\(|-)\s*", name, maxsplit=1)[0].strip()
            name = re.sub(r"\s+here\s+are.*$", "", name, flags=re.IGNORECASE).strip()
            name = re.sub(r"\s+at\s+[A-Z].*$", "", name).strip()
            if not name:
                continue
            key = name.lower()
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(name)
        return cleaned

    # If the agent did not include comments, enrich known professor names with comments.
    has_comment_block = bool(
        re.search(r"comments?\s+from\s+students?\s*:|student\s+comments?\s+for\s+[^:\n]+\s*:", text, re.IGNORECASE)
    )
    if not has_comment_block:
        name_matches = _extract_professor_names(text, input)

        blocks: List[str] = []
        seen: set[str] = set()
        for name in name_matches[:4]:
            n = name.strip()
            if not n or n.lower() in seen:
                continue
            seen.add(n.lower())
            rows = get_professor(n, limit=1)
            if not rows:
                continue
            comments = rows[0].get("comments") or []
            comments = [c for c in comments if isinstance(c, str) and c.strip()][:3]
            if comments:
                block = [f"Student comments for {n}:"]
                for c in comments:
                    block.append(f"- {c}")
                blocks.append("\n".join(block))

        if blocks:
            cleaned = re.sub(
                r"(?im)^.*(?:don['’]t\s+have|do\s+not\s+have|unfortunately[^.\n]*|not\s+available[^.\n]*).*(?:comment|comments)[^.\n]*\.?\s*$",
                "",
                text,
            ).strip()
            base = cleaned or text.strip()
            text = base.rstrip() + "\n\n" + "\n\n".join(blocks)

    return text


# =============================================================================
# Manual quick test
# =============================================================================

if __name__ == "__main__":
    print("Auth present? ", bool(_AUTHORIZATION_FULL or _AUTH_TOKEN_ONLY))
    print("Trying 'John'...")
    print(get_professor("John", limit=1))
