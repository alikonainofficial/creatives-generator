from __future__ import annotations

from typing import Callable

from logging_config import get_logger
from services.fal_queue import FalQueueClient

_MODEL = "fal-ai/workflow-utilities/auto-subtitle"

# Caption poll is faster than Kling (transcription + FFmpeg burn-in, not video gen).
_DEFAULT_POLL_INTERVAL = 10
_DEFAULT_MAX_ATTEMPTS = 30  # up to 5 min


def add_captions(
    video_url: str,
    fal_queue: FalQueueClient,
    poll_interval: int = _DEFAULT_POLL_INTERVAL,
    max_poll_attempts: int = _DEFAULT_MAX_ATTEMPTS,
    on_poll: Callable[[int, str], None] | None = None,
) -> str:
    """Burn word-by-word captions into a video using fal-ai/workflow-utilities/auto-subtitle.

    One word is displayed at a time, white bold text with a black stroke, positioned
    near the bottom of the frame. Uses ElevenLabs transcription for word-level timing.

    Returns the URL of the captioned video (FAL ephemeral storage).

    Raises RuntimeError if the job fails or no URL is found in the response.
    """
    logger = get_logger(__name__, service="caption_svc")

    queue_result = fal_queue.submit(
        _MODEL,
        {
            "video_url": video_url,
            "words_per_subtitle": 1,
            "font_color": "white",
            "font_name": "Montserrat",
            "font_weight": "bold",
            "font_size": 100,
            "stroke_color": "black",
            "stroke_width": 3,
            "background_color": "none",
            "position": "bottom",
            "y_offset": 75,
            "enable_animation": True,
            "language": "en",
        },
    )

    logger.info(
        "Caption job submitted",
        extra={"request_id": queue_result.request_id, "video_url": video_url},
    )

    result = fal_queue.wait_for_completion(
        queue_result,
        poll_interval=poll_interval,
        max_attempts=max_poll_attempts,
        on_poll=on_poll,
    )

    video = result.get("video", {})
    captioned_url: str = video.get("url", "") if isinstance(video, dict) else str(video)

    if not captioned_url:
        raise RuntimeError(f"No video URL in auto-subtitle response: {result}")

    logger.info(
        "Caption job complete",
        extra={"captioned_url": captioned_url, "request_id": queue_result.request_id},
    )
    return captioned_url
