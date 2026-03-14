from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import logging
from pydantic import BaseModel, Field

from pipeline.json_utils import extract_json
from pipeline.video_analysis import VideoAnalysisResult
from services.gemini import GeminiClient


PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "script_rewrite.txt"

logger = logging.getLogger(__name__)


def compute_script_limits(target_duration_s: int) -> dict[str, int]:
    """Derive Gemini prompt constraints from a target spoken duration.

    The formula is calibrated so that target_duration_s=20 reproduces the
    original hardcoded values exactly:
      max_words=45, min_words=30, ideal_min_s=14, ideal_max_s=18, max_segments=4
    """
    max_words = max(15, int(target_duration_s * 2.25))
    min_words = max(10, int(target_duration_s * 1.50))
    ideal_max_s = max(8, int(target_duration_s * 0.90))
    ideal_min_s = max(5, int(target_duration_s * 0.70))
    max_segments = max(2, min(6, round(target_duration_s / 5)))
    return {
        "target_duration_s": target_duration_s,
        "max_words": max_words,
        "min_words": min_words,
        "ideal_min_s": ideal_min_s,
        "ideal_max_s": ideal_max_s,
        "max_segments": max_segments,
    }


class ScriptSegment(BaseModel):
    segment_id: int
    dialogue: str
    is_demo: bool = False
    # Optional metadata for app demo insertion segments. These fields are
    # populated when the upstream JSON uses the n8n-style schema with
    # "insert_demo", "demo_type", and "mode".
    demo_type: str | None = None
    mode: str | None = None


class RewrittenScript(BaseModel):
    segments: list[ScriptSegment] = Field(default_factory=list)
    total_word_count: int = 0


def rewrite_script(
    original_dialogue: list[str],
    gender: str,
    analysis: VideoAnalysisResult,
    gemini: GeminiClient,
    target_duration_s: int = 20,
) -> RewrittenScript:
    """Call Gemini to rewrite the script for Speechify branding."""
    prompt_template = PROMPT_PATH.read_text()

    limits = compute_script_limits(target_duration_s)
    logger.info(
        "Script rewrite started",
        extra={
            "target_duration_s": target_duration_s,
            "max_words": limits["max_words"],
            "min_words": limits["min_words"],
            "max_segments": limits["max_segments"],
        },
    )

    # Build JSON payloads expected by the prompt template.
    video_format_summary_json = json.dumps(
        analysis.video_summary.model_dump(),
        ensure_ascii=False,
    )
    spoken_dialogue_json = json.dumps(
        [d.model_dump() for d in analysis.spoken_dialogue],
        ensure_ascii=False,
    )
    static_on_screen_text_json = json.dumps(
        [t.model_dump() for t in analysis.on_screen_text if t.text_type == "static_overlay"],
        ensure_ascii=False,
    )

    # Avoid str.format() here because the prompt contains literal JSON braces.
    prompt = (
        prompt_template
        .replace("{{ target_duration_s }}", str(limits["target_duration_s"]))
        .replace("{{ max_words }}", str(limits["max_words"]))
        .replace("{{ min_words }}", str(limits["min_words"]))
        .replace("{{ ideal_min_s }}", str(limits["ideal_min_s"]))
        .replace("{{ ideal_max_s }}", str(limits["ideal_max_s"]))
        .replace("{{ max_segments }}", str(limits["max_segments"]))
        .replace("{{ video_format_summary_json }}", video_format_summary_json)
        .replace("{{ spoken_dialogue_json }}", spoken_dialogue_json)
        .replace("{{ static_on_screen_text_json }}", static_on_screen_text_json)
    )

    raw = gemini.generate_text(prompt)
    json_text = extract_json(raw)

    try:
        data: dict[str, Any] = json.loads(json_text)
    except json.JSONDecodeError as e:
        # Log full context for debugging, but keep the raised message concise and actionable.
        logger.exception(
            "Failed to decode Gemini script JSON",
            extra={"raw_response_preview": raw[:500]},
        )

        preview = json_text[:200].replace("\n", "\\n")
        raise RuntimeError(
            "Script rewrite JSON parsing failed. "
            f"Error: {e}. "
            f"Preview of extracted JSON: '{preview}'. "
            "Check the Gemini 'script_rewrite.txt' prompt to ensure it returns a single JSON object "
            "with either a top-level 'script' array (n8n-style) or a 'segments' array "
            "matching the RewrittenScript schema."
        ) from e

    # Primary path: n8n-style schema with top-level "script" array of segments.
    if isinstance(data, dict) and "script" in data and isinstance(data["script"], list):
        segments: list[ScriptSegment] = []
        for idx, seg in enumerate(data["script"], start=1):
            if not isinstance(seg, dict):
                continue
            text = str(seg.get("text", "")).strip()
            if not text:
                continue
            is_demo = bool(seg.get("insert_demo", False))
            demo_type = seg.get("demo_type")
            mode = seg.get("mode")
            segments.append(
                ScriptSegment(
                    segment_id=idx,
                    dialogue=text,
                    is_demo=is_demo,
                    demo_type=demo_type,
                    mode=mode,
                )
            )

        total_word_count = sum(len(s.dialogue.split()) for s in segments)
        logger.info(
            "Script rewrite completed",
            extra={
                "target_duration_s": target_duration_s,
                "total_word_count": total_word_count,
                "max_words": limits["max_words"],
                "within_limit": total_word_count <= limits["max_words"],
                "segments_count": len(segments),
            },
        )
        return RewrittenScript(segments=segments, total_word_count=total_word_count)

    # Fallback: assume the JSON already matches the RewrittenScript schema.
    try:
        return RewrittenScript.model_validate(data)
    except Exception as e:
        logger.exception(
            "Failed to validate Gemini script JSON against RewrittenScript schema",
            extra={
                "data_keys": list(data.keys()) if isinstance(data, dict) else None,
            },
        )
        raise RuntimeError(
            "Script rewrite validation failed. "
            f"Error: {e}. "
            f"Top-level keys: {list(data.keys()) if isinstance(data, dict) else 'n/a'}. "
            "Ensure the JSON matches either the n8n-style schema "
            "{ 'script': [{ 'speaker': 'main', 'text': str, 'insert_demo': bool, 'demo_type': ..., 'mode': ... }] } "
            "or the RewrittenScript schema: "
            "{ 'segments': [{ 'segment_id': int, 'dialogue': str, 'is_demo': bool }], 'total_word_count': int }."
        ) from e
