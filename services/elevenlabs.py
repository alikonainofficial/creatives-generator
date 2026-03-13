from __future__ import annotations

import os
from typing import Any

import fal_client
import httpx

from logging_config import get_logger

_logger = get_logger(__name__, service="elevenlabs")

# "Adam" — clear American English male; override via ELEVENLABS_VOICE_ID env var.
_DEFAULT_VOICE_ID = "pNInz6obpgDQGcFmaJgB"
_DEFAULT_MODEL_ID = "eleven_multilingual_v2"
_DEFAULT_OUTPUT_FORMAT = "mp3_44100_128"

_BASE_URL = "https://api.elevenlabs.io"


class ElevenLabsClient:
    """ElevenLabs TTS client with full SSML / phoneme-tag support.

    Plain text usage:
        client.text_to_speech("Hello world")

    IPA phoneme control (fixes mispronunciations):
        client.text_to_speech(
            '<speak>Download <phoneme alphabet="ipa" ph="ˈspiːtʃɪfaɪ">'
            'Speechify</phoneme> today.</speak>'
        )

    The client uploads generated audio to fal.ai ephemeral storage so it
    can be passed directly to fal.ai downstream models (Kling LipSync).
    """

    def __init__(self, api_key: str | None = None):
        self._api_key = api_key or os.environ.get("ELEVENLABS_API_KEY", "")
        self._default_voice_id = os.environ.get("ELEVENLABS_VOICE_ID", _DEFAULT_VOICE_ID)

    def text_to_speech(
        self,
        text: str,
        voice_id: str | None = None,
        model_id: str = _DEFAULT_MODEL_ID,
        output_format: str = _DEFAULT_OUTPUT_FORMAT,
        voice_settings: dict[str, Any] | None = None,
    ) -> bytes:
        """Convert text (or SSML) to speech. Returns raw mp3 bytes.

        To fix mispronounced words, use SSML phoneme tags:
            <speak>
              <phoneme alphabet="ipa" ph="ˈspiːtʃɪfaɪ">Speechify</phoneme>
            </speak>

        IPA and x-arpabet alphabets are both supported on eleven_multilingual_v2.
        """
        if not self._api_key:
            raise RuntimeError(
                "ELEVENLABS_API_KEY is not set. "
                "Add it to your .env to enable pronunciation fix."
            )

        vid = voice_id or self._default_voice_id
        url = f"{_BASE_URL}/v1/text-to-speech/{vid}"

        payload: dict[str, Any] = {"text": text, "model_id": model_id}
        if voice_settings:
            payload["voice_settings"] = voice_settings

        headers = {
            "xi-api-key": self._api_key,
            "Content-Type": "application/json",
            "Accept": "audio/mpeg",
        }

        _logger.info(
            "Generating ElevenLabs TTS",
            extra={
                "voice_id": vid,
                "model_id": model_id,
                "text_preview": text[:80],
            },
        )

        with httpx.Client(timeout=60.0, transport=httpx.HTTPTransport(retries=3)) as client:
            resp = client.post(
                url,
                headers=headers,
                params={"output_format": output_format},
                json=payload,
            )
            resp.raise_for_status()

        _logger.info(
            "ElevenLabs TTS complete",
            extra={"audio_bytes": len(resp.content), "voice_id": vid},
        )
        return resp.content

    def text_to_speech_and_upload(
        self,
        text: str,
        voice_id: str | None = None,
    ) -> str:
        """Generate TTS audio and upload to fal.ai ephemeral storage.

        Returns a publicly accessible URL that can be passed directly to
        fal.ai models (Kling LipSync, Demucs, ffmpeg-api).
        """
        audio_bytes = self.text_to_speech(text, voice_id=voice_id)
        fal_url: str = fal_client.upload(audio_bytes, "audio/mpeg")
        _logger.info("TTS audio uploaded to fal storage", extra={"fal_url": fal_url})
        return fal_url
