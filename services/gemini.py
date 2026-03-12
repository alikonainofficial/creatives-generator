from __future__ import annotations

import json
import os
import re
import tempfile
import time
from pathlib import Path

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

        url_lower = image_url.lower().split("?")[0]
        if url_lower.endswith(".png"):
            mime_type = "image/png"
        elif url_lower.endswith(".webp"):
            mime_type = "image/webp"
        elif url_lower.endswith(".gif"):
            mime_type = "image/gif"
        else:
            mime_type = "image/jpeg"

        image_part = types.Part.from_uri(file_uri=image_url, mime_type=mime_type)

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

        url_lower = image_url.lower().split("?")[0]
        if url_lower.endswith(".png"):
            mime_type = "image/png"
        elif url_lower.endswith(".webp"):
            mime_type = "image/webp"
        elif url_lower.endswith(".gif"):
            mime_type = "image/gif"
        else:
            mime_type = "image/jpeg"

        image_part = types.Part.from_uri(file_uri=image_url, mime_type=mime_type)

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
