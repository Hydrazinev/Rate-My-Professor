"""
Shared utilities for the Professor Chatbot.
"""
from __future__ import annotations

import re
from typing import List


def extract_professor_names(answer_text: str, user_input: str) -> List[str]:
    """
    Extract professor names from an answer string and the original user query.

    Uses multiple heuristics in priority order:
      1. Numbered markdown list items:  1. **Dr. Jane Doe**
      2. Narrative phrases:             "found a professor named X" / "details for professor X"
      3. Bullet field lines:            - Name: X  /  - **Name:** X
      4. Standalone titled lines:       Dr. Jane Doe  /  Professor Jane Doe
      5. Query fallback:                professor "X"  /  professor X

    Returns a deduplicated, cleaned list (max 4 items is the caller's concern).
    """
    names: List[str] = []

    # 1) Numbered markdown: 1. **Dr. Jane Doe**
    names.extend(re.findall(r"(?:^|\n)\s*\d+\.\s+\*\*([^*\n]+)\*\*", answer_text))

    # 2) Narrative patterns
    names.extend(re.findall(
        r"found\s+a\s+professor\s+named\s+([^\n:]+)", answer_text, flags=re.IGNORECASE
    ))
    names.extend(re.findall(
        r"found\s+(?:dr\.?|professor)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})",
        answer_text, flags=re.IGNORECASE,
    ))
    names.extend(re.findall(
        r"details\s+for\s+professor\s+([^\n:]+)", answer_text, flags=re.IGNORECASE
    ))

    # 3) Bullet field: - Name: X  or  - **Name:** X
    names.extend(re.findall(
        r"^\s*[-*]\s+\*{0,2}\s*Name\s*\*{0,2}\s*:\s*(.+)$",
        answer_text, flags=re.IGNORECASE | re.MULTILINE,
    ))

    # 4) Standalone titled lines: Dr. / Professor prefix
    names.extend(re.findall(
        r"^\s*((?:Dr\.?|Professor)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\s*$",
        answer_text, flags=re.MULTILINE,
    ))

    # 5) Fallback from user query
    names.extend(re.findall(r'professor\s+"([^"]+)"', user_input, flags=re.IGNORECASE))
    names.extend(re.findall(
        r"(?:professor|dr\.?)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})",
        user_input, flags=re.IGNORECASE,
    ))

    # Clean and deduplicate
    cleaned: List[str] = []
    seen: set[str] = set()
    for raw in names:
        name = re.sub(r"\s+", " ", (raw or "").strip().strip("-").strip())
        name = re.sub(r"^(?:Name\s*:\s*)", "", name, flags=re.IGNORECASE).strip()
        # Trim trailing sentence noise
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
