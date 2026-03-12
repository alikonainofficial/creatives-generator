from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

from logging_config import get_logger
from services.fal_queue import FalQueueClient, QueueSubmitResult

_logger = get_logger(__name__)

KLING_MODEL = "fal-ai/kling-video/o3/standard/image-to-video"
FFMPEG_MERGE_MODEL = "fal-ai/ffmpeg-api/merge-videos"


@dataclass
class ClipGenerationContext:
    clip_key: str
    clip_index: int
    total_clips: int
    dialogue: str
    duration_s: int
    beat_number: int
    clip_energy: int
    kling_prompt: str
    kling_start_frame_url: str
    kling_end_frame_url: str | None


def build_clip_context(
    clip_index: int,
    total_clips: int,
    dialogue: str,
    duration_s: int,
    clip_key: str,
    swapped_start_frame_url: str,
    blueprint: dict[str, Any],
    last_frame_url: str | None = None,
    start_frame_url: str | None = None,
) -> ClipGenerationContext:
    """Build the Kling generation context for a clip.

    Exact port of the n8n 'Build Generation Context' node:
    - Beat engine: energy ramps through the video, drops at the last beat.
    - SceneLock: odd clips use swapped_start_frame_url as start (no end frame).
                  even clips use last_frame_url as start, swapped_start_frame_url as end.
    - Format-aware prompts keyed on blueprint.video_type.
    """
    is_odd = clip_index % 2 == 1

    # ---- Beat engine ----
    beats = int(blueprint.get("structure", {}).get("estimated_beats") or 4)
    beats = max(1, beats)
    beat_size = total_clips / beats
    beat_number = min(beats, math.ceil(clip_index / beat_size))
    base_energy = int(blueprint.get("performance", {}).get("base_energy") or 3)

    if beat_number < beats:
        energy = min(5, base_energy + math.floor((beat_number / beats) * 2))
    else:
        energy = max(2, base_energy - 1)

    # ---- Motion text ----
    motion_intensity = int(blueprint.get("performance", {}).get("motion_intensity") or 2)
    if motion_intensity <= 2:
        motion_text = "Very minimal movement."
    elif motion_intensity == 3:
        motion_text = "Moderate natural movement."
    else:
        motion_text = "Expressive movement allowed."

    # ---- SceneLock frame logic ----
    if is_odd:
        start_frame = swapped_start_frame_url
        end_frame: str | None = None
    else:
        start_frame = last_frame_url or start_frame_url or swapped_start_frame_url
        end_frame = swapped_start_frame_url

    if not start_frame:
        raise ValueError(f"No valid SceneLock start frame for clip_index={clip_index}")

    # ---- Format-aware prompt ----
    env = blueprint.get("environment", {})
    perf = blueprint.get("performance", {})
    video_type = blueprint.get("video_type", "mixed")

    environment_block = (
        f"Environment:\n"
        f"{env.get('setting', '')}.\n"
        f"Camera: {env.get('camera_style', '')}.\n"
        f"Framing: {env.get('framing', '')}.\n"
        f"Lighting: {env.get('lighting', '')}.\n"
        f"Depth of field: {env.get('depth_of_field', '')}."
    ).strip()

    base_constraints = (
        "Constraints:\n"
        "- Lock environment.\n"
        "- No scene changes.\n"
        "- Use reference identity exactly as provided.\n"
        "- Do NOT alter facial structure, hair, or clothing.\n"
        "- Do NOT add any on-screen text."
    )

    if video_type == "talking_head":
        prompt = (
            f"{environment_block}\n\n"
            f"The subject delivers directly to camera:\n\n"
            f'"{dialogue}"\n\n'
            f"Energy level: {energy}/5.\n"
            f"Delivery style: {perf.get('delivery_style', '')}.\n"
            f"{motion_text}\n\n"
            f"{base_constraints}"
        ).strip()

    elif video_type == "expressive_no_dialogue":
        prompt = (
            f"{environment_block}\n\n"
            "No spoken dialogue.\n\n"
            "Performance is conveyed purely through facial expression and subtle movement.\n\n"
            f"Energy level: {energy}/5.\n"
            f"{motion_text}\n\n"
            "Constraints:\n"
            "- Lock environment.\n"
            "- Use reference identity exactly as provided.\n"
            "- Do NOT alter facial structure, hair, or clothing.\n"
            "- Do NOT add any on-screen text."
        ).strip()

    elif video_type == "b_roll_heavy":
        prompt = (
            f"{environment_block}\n\n"
            f"The spoken line is delivered as voiceover:\n\n"
            f'"{dialogue}"\n\n'
            "Visual focus may include environment details or subtle subject motion.\n\n"
            f"Energy level: {energy}/5.\n"
            f"{motion_text}\n\n"
            "Constraints:\n"
            "- Maintain visual continuity.\n"
            "- No abrupt camera changes.\n"
            "- Use reference identity exactly as provided.\n"
            "- Do NOT alter facial structure, hair, or clothing.\n"
            "- Do NOT add any on-screen text."
        ).strip()

    else:  # mixed, caption_driven, or unknown
        prompt = (
            f"{environment_block}\n\n"
            f"The subject delivers:\n\n"
            f'"{dialogue}"\n\n'
            "Blend direct address with subtle environmental presence.\n\n"
            f"Energy level: {energy}/5.\n"
            f"Delivery style: {perf.get('delivery_style', '')}.\n"
            f"{motion_text}\n\n"
            "Constraints:\n"
            "- Lock environment.\n"
            "- Maintain visual continuity.\n"
            "- Use reference identity exactly as provided.\n"
            "- Do NOT alter facial structure, hair, or clothing.\n"
            "- Do NOT add any on-screen text."
        ).strip()

    return ClipGenerationContext(
        clip_key=clip_key,
        clip_index=clip_index,
        total_clips=total_clips,
        dialogue=dialogue,
        duration_s=duration_s,
        beat_number=beat_number,
        clip_energy=energy,
        kling_prompt=prompt,
        kling_start_frame_url=start_frame,
        kling_end_frame_url=end_frame,
    )


def submit_clip_to_kling(
    context: ClipGenerationContext,
    fal_queue: FalQueueClient,
) -> QueueSubmitResult:
    """Submit a clip generation job to Kling via the FAL queue API."""
    arguments: dict[str, Any] = {
        "prompt": context.kling_prompt,
        "image_url": context.kling_start_frame_url,
        "duration": context.duration_s,
        "generate_audio": True,
    }
    if context.kling_end_frame_url:
        arguments["end_image_url"] = context.kling_end_frame_url

    _logger.info(
        "Submitting clip to Kling",
        extra={
            "clip_key": context.clip_key,
            "clip_index": context.clip_index,
            "duration_s": context.duration_s,
            "has_end_frame": context.kling_end_frame_url is not None,
            "energy": context.clip_energy,
        },
    )
    return fal_queue.submit(KLING_MODEL, arguments)


def extract_video_url_from_kling_result(result: dict[str, Any]) -> str:
    """Pull the video URL out of a Kling result dict."""
    url = (
        result.get("video", {}).get("url")
        if isinstance(result.get("video"), dict)
        else result.get("url", "")
    )
    if not url:
        url = result.get("output_url", "")
    if not url:
        raise RuntimeError(f"Cannot find video URL in Kling response: {result}")
    return url


def build_last_frame_url(clip_video_url: str, duration_s: float) -> str:
    """Derive a Cloudinary frame-extraction URL for the last frame of a clip.

    Port of the n8n 'Build Next Clip Start Frame URL' node.
    Injects `so_<time>` after `/video/upload/` and swaps the extension to .jpg.
    """
    last_time = max(0.1, duration_s - 0.1)
    injected = clip_video_url.replace(
        "/video/upload/", f"/video/upload/so_{last_time}/"
    )
    last_frame_url = re.sub(
        r"\.(mp4|mov|webm|mkv|avi)(\?.*)?$",
        r".jpg\2",
        injected,
        flags=re.IGNORECASE,
    )
    return last_frame_url
