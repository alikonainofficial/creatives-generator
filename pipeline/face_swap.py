from __future__ import annotations

from services.fal_ai import FalAiClient


def face_swap(
    frame_url: str,
    reference_image_url: str,
    fal: FalAiClient,
) -> str:
    """Submit face swap job to Fal.ai and block until complete. Returns result image URL."""
    return fal.face_swap(
        target_image_url=frame_url,
        reference_image_url=reference_image_url,
    )
