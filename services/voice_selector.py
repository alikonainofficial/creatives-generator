from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import httpx

from logging_config import get_logger

if TYPE_CHECKING:
    from services.gemini import GeminiClient

_logger = get_logger(__name__, service="voice_selector")

_BASE_URL = "https://api.elevenlabs.io"
_CATALOG_TTL_SECONDS = 3600  # re-fetch at most once per hour

# Module-level voice catalog cache: (voices, fetched_at_epoch)
_catalog_cache: tuple[list[dict[str, Any]], float] | None = None


@dataclass
class VoiceSelectionResult:
    voice_id: str
    name: str
    reasoning: str


def get_my_voices(api_key: str) -> list[dict[str, Any]]:
    """Fetch the user's ElevenLabs voice library with in-process TTL caching.

    Returns a list of compact voice dicts with only the fields relevant for
    AI matching (no preview URLs, no binary data):
        {voice_id, name, gender?, age?, accent?, use_case?, description?}

    Empty/null fields are omitted to keep the AI context compact.
    Cache is invalidated after CATALOG_TTL_SECONDS (1 hour).
    """
    global _catalog_cache

    now = time.time()
    if _catalog_cache is not None and (now - _catalog_cache[1]) < _CATALOG_TTL_SECONDS:
        _logger.debug("Voice catalog served from in-process cache")
        return _catalog_cache[0]

    _logger.info("Fetching ElevenLabs voice catalog from API")
    with httpx.Client(timeout=30.0) as client:
        resp = client.get(
            f"{_BASE_URL}/v1/voices",
            headers={"xi-api-key": api_key},
            params={"show_legacy": "false"},
        )
        resp.raise_for_status()

    raw_voices: list[dict[str, Any]] = resp.json().get("voices", [])
    voices: list[dict[str, Any]] = []

    for v in raw_voices:
        labels: dict[str, Any] = v.get("labels") or {}
        entry: dict[str, Any] = {
            "voice_id": v.get("voice_id", ""),
            "name": (v.get("name") or "").strip(),
            "gender": (labels.get("gender") or "").strip().lower(),
            "age": (labels.get("age") or "").strip(),
            "accent": (labels.get("accent") or "").strip(),
            "use_case": (labels.get("use_case") or "").strip(),
            "description": (
                labels.get("description") or v.get("description") or ""
            ).strip(),
        }
        # Strip empty fields so the AI context stays compact.
        entry = {k: val for k, val in entry.items() if val}
        if entry.get("voice_id"):
            voices.append(entry)

    _catalog_cache = (voices, now)
    _logger.info("Voice catalog cached", extra={"voice_count": len(voices)})
    return voices


def _filter_by_gender(
    voices: list[dict[str, Any]], gender: str
) -> list[dict[str, Any]]:
    """Return voices whose gender label matches. Falls back to the full list if empty."""
    if not gender or gender == "unknown":
        return voices
    filtered = [v for v in voices if v.get("gender", "") == gender]
    if not filtered:
        _logger.warning(
            "No voices found for gender='%s' — using full catalog as fallback", gender
        )
        return voices
    return filtered


def select_voice_for_subject(
    image_url: str,
    gender: str,
    elevenlabs_api_key: str,
    gemini: GeminiClient,
    vo_hint: str = "",
) -> VoiceSelectionResult | None:
    """Deterministically select the best ElevenLabs voice for the subject in the image.

    Phase 1 (deterministic): Fetch voice catalog (cached) and pre-filter by gender.
    Phase 2 (AI):            Gemini vision picks the best match from the filtered list.

    Returns None on any failure — callers should fall back to the configured default.
    """
    try:
        all_voices = get_my_voices(elevenlabs_api_key)
    except Exception as exc:
        _logger.warning(
            "Failed to fetch voice catalog (%s) — skipping voice selection", exc
        )
        return None

    if not all_voices:
        _logger.warning("Voice catalog is empty — skipping voice selection")
        return None

    candidates = _filter_by_gender(all_voices, gender)
    _logger.info(
        "Voice candidates after gender filter",
        extra={"total": len(all_voices), "candidates": len(candidates), "gender": gender},
    )

    try:
        selected = gemini.select_voice_for_image(image_url, candidates, vo_hint=vo_hint)
    except Exception as exc:
        _logger.warning(
            "Gemini voice selection failed (%s) — skipping voice selection", exc
        )
        return None

    result = VoiceSelectionResult(
        voice_id=selected["voice_id"],
        name=selected.get("name", ""),
        reasoning=selected.get("reasoning", ""),
    )
    _logger.info(
        "Voice selected for subject",
        extra={
            "voice_id": result.voice_id,
            "name": result.name,
            "gender_filter": gender,
            "candidates": len(candidates),
        },
    )
    return result
