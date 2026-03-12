"""Demo selector: pick the best-matching app demo for a given job.

Selection strategy:
  1. Parse each demo_id into a set of lowercase tokens by splitting on both
     "_" and "-".  This handles variable-length IDs and settings that are
     embedded inside the book-name component (e.g. "ItEndsWithUs-BedTime1").
     Gender is always the last "_"-delimited component.
  2. Filter by gender: keep demos whose last component is "MAN", "WMN", or
     "All", matched against the job's gender ("male" / "female").
     If no demo survives gender filtering, fall back to all demos.
  3. Score by keyword: for each keyword found in the full script text, add
     +5 if any token in the demo_id contains the corresponding setting
     substring (case-insensitive).  The token search naturally handles both
     explicit setting components ("Kitchen") and embedded ones ("BedTime1").
  4. Return the highest-scoring demo (first one on ties).  If no demo scores
     above 0, a random demo is returned from the candidate pool.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from logging_config import get_logger
from sheets.demo_reader import DemoRow


_logger = get_logger(__name__)

# keyword in script text → setting substring to look for in any demo_id token
_KEYWORD_MAP: dict[str, str] = {
    "bed":       "bed",        # matches "Bed", "BedTime1", …
    "bedtime":   "bed",
    "kitchen":   "kitchen",
    "plane":     "plane",
    "table":     "table",
    "living":    "living",     # matches "LivingRoom"
    "bookstore": "bookstore",
}


@dataclass
class _ParsedDemo:
    row: DemoRow
    gender: str               # last "_"-component: "MAN" | "WMN" | "All"
    id_tokens: set[str] = field(default_factory=set)  # all lowercase tokens


def _parse_demo(row: DemoRow) -> _ParsedDemo:
    """Build a _ParsedDemo by splitting demo_id on both '_' and '-'."""
    underscore_parts = row.demo_id.split("_")
    gender = underscore_parts[-1] if underscore_parts else ""

    tokens: set[str] = set()
    for part in underscore_parts:
        for sub in part.split("-"):
            if sub:
                tokens.add(sub.lower())

    return _ParsedDemo(row=row, gender=gender, id_tokens=tokens)


def _gender_matches(demo_gender: str, job_gender: str) -> bool:
    if not job_gender:
        return True
    g = job_gender.lower()
    if g == "male":
        return demo_gender in ("MAN", "All")
    if g == "female":
        return demo_gender in ("WMN", "All")
    return True


def _score_demo(parsed: _ParsedDemo, script_lower: str) -> int:
    """Keyword-based environment score.

    For each keyword found in the script, add +5 if any demo_id token
    contains the corresponding setting substring.
    """
    score = 0
    for keyword, setting_substr in _KEYWORD_MAP.items():
        if keyword in script_lower:
            if any(setting_substr in token for token in parsed.id_tokens):
                score += 5
    return score


def select_demo(
    demos: list[DemoRow],
    script_text: str,
    gender: str,
) -> DemoRow:
    """Return the best-matching demo for a given job.

    Args:
        demos: All rows from the App Demo sheet (only demo_id + video_url
            are required — no demo_type column needed).
        script_text: Full concatenated script text used for keyword scoring.
        gender: Job gender string ("male", "female", or empty).

    Raises:
        ValueError: If demos is empty.
    """
    if not demos:
        raise ValueError("No demos available in the App Demo sheet.")

    parsed = [_parse_demo(d) for d in demos]

    gender_filtered = [p for p in parsed if _gender_matches(p.gender, gender)]
    if gender_filtered:
        candidates = gender_filtered
        fallback_reason = None
    else:
        candidates = parsed
        fallback_reason = f"no demos matched gender '{gender}'"
        _logger.warning(
            "No demos match gender '%s' — falling back to all demos",
            gender,
            extra={"total_demos": len(parsed)},
        )

    script_lower = script_text.lower()
    scored = sorted(
        candidates,
        key=lambda p: _score_demo(p, script_lower),
        reverse=True,
    )

    best_score = _score_demo(scored[0], script_lower)
    if best_score > 0:
        best = scored[0]
    else:
        best = random.choice(candidates)
        _logger.info(
            "No keyword matches — picking random demo",
            extra={"demo_id": best.row.demo_id, "pool_size": len(candidates)},
        )

    _logger.info(
        "Demo selected",
        extra={
            "demo_id": best.row.demo_id,
            "gender": best.gender,
            "score": _score_demo(best, script_lower),
            "candidates": len(candidates),
            "fallback_reason": fallback_reason,
        },
    )
    return best.row
