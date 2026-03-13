from __future__ import annotations

import json
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any

import httpx
from google import genai
from google.genai import types

from logging_config import get_logger


def _extract_json_from_response(text: str) -> str:
    """Minimal JSON extractor for Gemini responses (handles markdown fences)."""
    text = text.strip()
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence_match:
        return fence_match.group(1).strip()
    brace_match = re.search(r"\{[\s\S]*\}", text)
    if brace_match:
        return brace_match.group(0).strip()
    return text


def _image_mime_type(url: str) -> str:
    """Infer the MIME type of an image URL from its extension."""
    lower = url.lower().split("?")[0]
    if lower.endswith(".png"):
        return "image/png"
    if lower.endswith(".webp"):
        return "image/webp"
    if lower.endswith(".gif"):
        return "image/gif"
    return "image/jpeg"


class GeminiClient:
    def __init__(self, api_key: str | None = None):
        key = api_key or os.environ["GEMINI_API_KEY"]
        self._client = genai.Client(api_key=key)
        self._logger = get_logger(__name__, service="gemini")

    def analyze_video(self, video_url: str, prompt: str) -> str:
        """Analyze a publicly accessible video URL using Gemini 3.1 Pro preview."""
        start = time.time()

        # Detect mime type from URL extension; default to mp4
        url_lower = video_url.lower().split("?")[0]
        if url_lower.endswith(".webm"):
            mime_type = "video/webm"
        elif url_lower.endswith(".mov"):
            mime_type = "video/quicktime"
        elif url_lower.endswith(".avi"):
            mime_type = "video/avi"
        else:
            mime_type = "video/mp4"

        video_part = types.Part.from_uri(file_uri=video_url, mime_type=mime_type)

        result = self._client.models.generate_content(
            model="gemini-3.1-pro-preview",
            contents=[video_part, prompt],
        )
        elapsed = time.time() - start
        self._logger.info(
            "Gemini video analysis completed",
            extra={"video_url": video_url, "duration_seconds": round(elapsed, 2)},
        )
        return result.text

    def generate_text(self, prompt: str) -> str:
        """Generate text using Gemini 3.1 Pro preview."""
        start = time.time()
        result = self._client.models.generate_content(
            model="gemini-3.1-pro-preview",
            contents=prompt,
        )
        elapsed = time.time() - start
        self._logger.info(
            "Gemini text generation completed",
            extra={"duration_seconds": round(elapsed, 2)},
        )
        return result.text

    def analyze_persona_image(self, image_url: str) -> dict[str, str]:
        """Analyze a persona image: return detected gender and scene description.

        Returns a dict with keys 'gender' ("male"/"female"/"unknown") and
        'scene_description' (1-2 sentence environment/setting summary).
        """
        prompt = (
            "Analyze the person and environment in this image. "
            "Return a JSON object with exactly two fields: "
            "'gender' (one of: 'male', 'female', 'unknown') and "
            "'scene_description' (1-2 sentences describing the setting, environment, and mood visible in the image). "
            "Return only valid JSON, no markdown, no commentary."
        )

        image_part = types.Part.from_uri(
            file_uri=image_url, mime_type=_image_mime_type(image_url)
        )

        start = time.time()
        result = self._client.models.generate_content(
            model="gemini-3.1-pro-preview",
            contents=[image_part, prompt],
        )
        elapsed = time.time() - start

        try:
            data = json.loads(_extract_json_from_response(result.text))
        except (json.JSONDecodeError, AttributeError):
            data = {}

        gender = str(data.get("gender", "unknown")).strip().lower()
        if gender not in ("male", "female"):
            gender = "unknown"
        scene_description = str(data.get("scene_description", "")).strip()

        self._logger.info(
            "Gemini persona image analysis completed",
            extra={
                "image_url": image_url,
                "duration_seconds": round(elapsed, 2),
                "gender": gender,
            },
        )
        return {"gender": gender, "scene_description": scene_description}

    def detect_gender(self, image_url: str) -> str:
        """Detect speaker gender from a publicly accessible image URL."""
        prompt = (
            "Look at the person in this image. "
            "Respond with ONLY one word: 'male' or 'female'. "
            "No other text."
        )

        image_part = types.Part.from_uri(
            file_uri=image_url, mime_type=_image_mime_type(image_url)
        )

        start = time.time()
        result = self._client.models.generate_content(
            model="gemini-3.1-pro-preview",
            contents=[image_part, prompt],
        )
        elapsed = time.time() - start
        gender = result.text.strip().lower()
        normalized = gender if gender in ("male", "female") else "unknown"
        self._logger.info(
            "Gemini gender detection completed",
            extra={"image_url": image_url, "duration_seconds": round(elapsed, 2), "gender": normalized},
        )
        return normalized

    def select_voice_for_image(
        self,
        image_url: str,
        voices: list[dict[str, Any]],
        vo_hint: str = "",
    ) -> dict[str, Any]:
        """Select the best ElevenLabs voice for the character visible in the image.

        voices: list of compact voice dicts (voice_id, name, gender, accent, …).
        vo_hint: optional operator instruction to bias the selection.

        Returns {"voice_id": str, "name": str, "reasoning": str}.
        Raises ValueError if Gemini returns a voice_id not present in the catalog
        (guards against hallucinated IDs).
        """
        hint_block = (
            f"\nOperator instruction: {vo_hint.strip()}" if vo_hint.strip() else ""
        )
        prompt = (
            "You are a casting director selecting a text-to-speech voice for the "
            "character in this image.\n\n"
            f"Available voices:\n{json.dumps(voices, indent=2)}\n"
            f"{hint_block}\n\n"
            "Pick the single voice that best fits the character's apparent age, "
            "gender, energy, and personality.\n"
            'Return a JSON object with exactly these three fields:\n'
            '  "voice_id": the exact voice_id string from the list above\n'
            '  "name": the corresponding voice name\n'
            '  "reasoning": one sentence explaining why this voice fits\n'
            "Return only valid JSON — no markdown fences, no extra keys."
        )

        image_part = types.Part.from_uri(
            file_uri=image_url, mime_type=_image_mime_type(image_url)
        )

        start = time.time()
        result = self._client.models.generate_content(
            model="gemini-3.1-pro-preview",
            contents=[image_part, prompt],
            config={"response_mime_type": "application/json"},
        )
        elapsed = time.time() - start

        try:
            data: dict[str, Any] = json.loads(_extract_json_from_response(result.text))
        except (json.JSONDecodeError, AttributeError) as exc:
            raise ValueError(f"Gemini returned non-JSON response: {exc}") from exc

        known_ids = {v["voice_id"] for v in voices}
        if data.get("voice_id") not in known_ids:
            raise ValueError(
                f"Gemini returned voice_id={data.get('voice_id')!r} "
                "which is not present in the provided catalog"
            )

        self._logger.info(
            "Gemini voice selection completed",
            extra={
                "voice_id": data.get("voice_id"),
                "voice_name": data.get("name"),
                "duration_seconds": round(elapsed, 2),
            },
        )
        return data
