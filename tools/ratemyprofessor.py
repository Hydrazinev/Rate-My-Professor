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

import json
import os

import requests
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_core.tools import StructuredTool
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
        resp = requests.post(
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


# =============================================================================
# Public search functions (stable external API)
# =============================================================================

def get_professor(name: str, limit: int = 5) -> List[JSON]:
    """
    Search professors by free-text name.
    """
    query = load_query_from_file("search_professor_by_name")
    variables = {"query": {"text": name}, "count": limit}
    resp = send_graphql_request(query, variables)
    return format_professor_response(resp, limit)


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
    return format_professor_response(resp, limit)


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
_prompt = hub.pull("hwchase17/openai-tools-agent")
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
    return out.get("output", "")


# =============================================================================
# Manual quick test
# =============================================================================

if __name__ == "__main__":
    print("Auth present? ", bool(_AUTHORIZATION_FULL or _AUTH_TOKEN_ONLY))
    print("Trying 'John'...")
    print(get_professor("John", limit=1))
