from __future__ import annotations

import json
import re
from typing import Any

from logging_config import get_logger
from services.gemini import GeminiClient

_logger = get_logger(__name__)

REQUIRED_BLUEPRINT_KEYS = frozenset(
    {"video_type", "environment", "performance", "structure", "editing", "constraints"}
)

# Mirrors the Gemini prompt from the n8n "Generate Video Prompt Blueprint" node exactly.
_BLUEPRINT_PROMPT = """\
System:

You are a short-form video format analyst.
Output MUST be valid JSON only.
No markdown. No commentary. No extra keys.


User:

Given this source video analysis JSON:

{analysis_json}

Create a VIDEO_BLUEPRINT JSON with this exact schema:

{{
"video_type": "talking_head | expressive_no_dialogue | caption_driven | b_roll_heavy | mixed",
"environment": {{
"setting": "string",
"camera_style": "string",
"framing": "string",
"lighting": "string",
"depth_of_field": "string"
}},
"performance": {{
"base_energy": 1,
"motion_intensity": 1,
"facial_expressiveness": 1,
"delivery_style": "string"
}},
"structure": {{
"has_spoken_dialogue": true,
"caption_heavy": false,
"estimated_beats": 4
}},
"editing": {{
"cut_style": "none | jump_cuts | quick_cuts | cinematic",
"music_presence": "none | subtle | strong"
}},
"constraints": {{
"lock_environment": true,
"use_reference_identity": true
}}
}}

Rules:
\t•\tInfer video_type from spoken_dialogue length, roll_structure, and editing_style.
\t•\tIf spoken_dialogue is empty but captions exist → expressive_no_dialogue or caption_driven.
\t•\tbase_energy: 1 (calm) to 5 (high emotional intensity).
\t•\tmotion_intensity: 1 (still) to 5 (high movement).
\t•\tfacial_expressiveness: 1 (neutral) to 5 (highly expressive).
\t•\testimated_beats: approximate number of emotional shifts in spoken_dialogue.
\t•\tDo NOT include subject appearance details.
\t•\tDo NOT include dialogue text.
\t•\tOutput valid JSON only.\
"""


def validate_blueprint(prompt_template: Any) -> bool:
    """Return True if prompt_template is a valid, fully-formed blueprint dict."""
    if not prompt_template:
        return False
    try:
        parsed = (
            json.loads(prompt_template)
            if isinstance(prompt_template, str)
            else prompt_template
        )
        if not isinstance(parsed, dict):
            return False
        return REQUIRED_BLUEPRINT_KEYS.issubset(parsed.keys())
    except (json.JSONDecodeError, TypeError):
        return False


_PERSONA_BLUEPRINT_PROMPT = """\
System:

You are a short-form video format analyst specialising in persona-driven talking-head ads.
Output MUST be valid JSON only.
No markdown. No commentary. No extra keys.


User:

Given this persona ad context:

Persona: {persona}
Pain point: {pain_point}
Speechify solution: {speechify_solution}
Hook emotion: {hook_emotion}
Scene description: {scene_description}
Gender: {gender}
Number of script clips: {clip_count}

Create a VIDEO_BLUEPRINT JSON with this exact schema:

{{
"video_type": "talking_head | expressive_no_dialogue | caption_driven | b_roll_heavy | mixed",
"environment": {{
"setting": "string",
"camera_style": "string",
"framing": "string",
"lighting": "string",
"depth_of_field": "string"
}},
"performance": {{
"base_energy": 1,
"motion_intensity": 1,
"facial_expressiveness": 1,
"delivery_style": "string"
}},
"structure": {{
"has_spoken_dialogue": true,
"caption_heavy": false,
"estimated_beats": 4
}},
"editing": {{
"cut_style": "none | jump_cuts | quick_cuts | cinematic",
"music_presence": "none | subtle | strong"
}},
"constraints": {{
"lock_environment": true,
"use_reference_identity": true
}}
}}

Rules:
\t•\tThis is always a talking-head ad with spoken dialogue — set video_type accordingly.
\t•\tDerive environment fields from the scene_description.
\t•\tbase_energy: 1 (calm) to 5 (high emotional intensity) — infer from hook_emotion and pain_point.
\t•\tmotion_intensity: 1 (still) to 5 (high movement) — talking-head ads are typically 1–2.
\t•\tfacial_expressiveness: 1 (neutral) to 5 (highly expressive) — infer from hook_emotion.
\t•\testimated_beats: use clip_count as a guide.
\t•\tDo NOT include subject appearance details.
\t•\tDo NOT include dialogue text.
\t•\tOutput valid JSON only.\
"""


def generate_blueprint_from_persona(
    persona: str,
    pain_point: str,
    speechify_solution: str,
    hook_emotion: str,
    scene_description: str,
    gender: str,
    clip_count: int,
    gemini: GeminiClient,
) -> dict[str, Any]:
    """Generate a video blueprint from persona context (no source video required).

    Used by the persona pipeline at Phase A time so the Producer always finds a
    valid prompt_template and never hits the 'no analysis_json' dead-end.
    """
    prompt = _PERSONA_BLUEPRINT_PROMPT.format(
        persona=persona,
        pain_point=pain_point,
        speechify_solution=speechify_solution,
        hook_emotion=hook_emotion,
        scene_description=scene_description or "Not available",
        gender=gender or "unknown",
        clip_count=clip_count,
    )
    raw = gemini.generate_text(prompt)

    cleaned = raw.strip()
    cleaned = re.sub(r"```json\s*", "", cleaned)
    cleaned = re.sub(r"```", "", cleaned).strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Persona blueprint JSON parsing failed: {exc}. Preview: {cleaned[:300]}"
        ) from exc

    if not isinstance(parsed, dict) or not REQUIRED_BLUEPRINT_KEYS.issubset(parsed.keys()):
        raise RuntimeError(
            f"Persona blueprint missing required keys. Got: {list(parsed.keys() if isinstance(parsed, dict) else [])}"
        )

    _logger.info("Persona video blueprint generated", extra={"video_type": parsed.get("video_type")})
    return parsed


def generate_blueprint(analysis_json: str, gemini: GeminiClient) -> dict[str, Any]:
    """Call Gemini to generate a video blueprint from analysis_json.

    Ports the n8n 'Generate Video Prompt Blueprint' → 'Clean Blueprint JSON' nodes.
    """
    prompt = _BLUEPRINT_PROMPT.format(analysis_json=analysis_json)
    raw = gemini.generate_text(prompt)

    cleaned = raw.strip()
    cleaned = re.sub(r"```json\s*", "", cleaned)
    cleaned = re.sub(r"```", "", cleaned).strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Blueprint JSON parsing failed: {exc}. Preview: {cleaned[:300]}"
        ) from exc

    if not isinstance(parsed, dict) or not REQUIRED_BLUEPRINT_KEYS.issubset(parsed.keys()):
        raise RuntimeError(
            f"Blueprint missing required keys. Got: {list(parsed.keys() if isinstance(parsed, dict) else [])}"
        )

    _logger.info("Video blueprint generated", extra={"video_type": parsed.get("video_type")})
    return parsed
