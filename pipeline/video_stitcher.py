from __future__ import annotations

from typing import Any, Callable

from logging_config import get_logger
from services.fal_queue import FalQueueClient

_logger = get_logger(__name__)

FFMPEG_MERGE_MODEL = "fal-ai/ffmpeg-api/merge-videos"


def stitch_clips(
    video_urls: list[str],
    fal_queue: FalQueueClient,
    poll_interval: int = 30,
    max_poll_attempts: int = 40,
    on_poll: Callable[[int, str], None] | None = None,
) -> str:
    """Submit a merge job via FAL FFmpeg API and return the merged video URL.

    Mirrors the n8n 'Merge Clips' → poll → 'Get Final Video' flow.
    """
    _logger.info("Submitting clip merge job", extra={"clip_count": len(video_urls)})

    submit_result = fal_queue.submit(FFMPEG_MERGE_MODEL, {"video_urls": video_urls})
    merged = fal_queue.wait_for_completion(
        submit_result,
        poll_interval=poll_interval,
        max_attempts=max_poll_attempts,
        on_poll=on_poll,
    )

    video_url = _extract_merged_url(merged)
    if not video_url:
        raise RuntimeError(f"FFmpeg merge returned no video URL. Response: {merged}")

    _logger.info("Clip merge complete", extra={"merged_url": video_url})
    return video_url


def _extract_merged_url(result: dict[str, Any]) -> str:
    if isinstance(result.get("video"), dict):
        return result["video"].get("url", "")
    return result.get("url") or result.get("output_url", "")
