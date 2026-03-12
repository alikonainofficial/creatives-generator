from __future__ import annotations

import re


def extract_json(text: str) -> str:
    """Robustly extract a JSON object from a Gemini response.

    Handles:
    - Markdown code fences (```json ... ```)
    - Thinking preamble / prose before or after the JSON
    - Varied whitespace and newlines before the opening brace
    """
    text = text.strip()

    # Try extracting from a markdown code fence first
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence_match:
        return fence_match.group(1).strip()

    # Fall back: find the outermost {...} block in the text
    brace_match = re.search(r"\{[\s\S]*\}", text)
    if brace_match:
        return brace_match.group(0).strip()

    # Some models occasionally emit a top-level object body without the opening brace,
    # e.g. `"segments": [...] }`. Try to normalize that into a valid JSON object.
    # We intentionally keep this heuristic narrow to avoid masking real errors.
    # If the text starts with a quoted key (e.g. `"segments":`) but does *not*
    # start with `{`, wrap it with a leading brace.
    stripped = text.lstrip()
    if stripped.startswith('"') and not stripped.startswith("{"):
        # Avoid adding an extra closing brace if one is already present at the end.
        if stripped.endswith("}"):
            return "{" + stripped
        return "{" + stripped + "}"

    # Return as-is and let json.loads raise a clear error
    return text
