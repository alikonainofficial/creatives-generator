from __future__ import annotations

import math
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fal_client
import httpx

from logging_config import get_logger
from services.fal_queue import FalQueueClient, QueueSubmitResult

_logger = get_logger(__name__)

KLING_MODEL = "fal-ai/kling-video/o3/standard/image-to-video"
FFMPEG_MERGE_MODEL = "fal-ai/ffmpeg-api/merge-videos"
KLING_LIPSYNC_MODEL = "fal-ai/kling-video/lipsync/audio-to-video"

# Non-vocal Demucs stems that make up the ambient background track.
# "vocals" is intentionally excluded — that's Kling's mispronounced speech.
_BACKGROUND_STEMS = ("drums", "bass", "other", "guitar", "piano")

_LIPSYNC_POLL_INTERVAL = 15  # seconds between polls (LipSync is faster than gen)
_LIPSYNC_MAX_ATTEMPTS = 24  # up to 6 min


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
    injected = clip_video_url.replace("/video/upload/", f"/video/upload/so_{last_time}/")
    last_frame_url = re.sub(
        r"\.(mp4|mov|webm|mkv|avi)(\?.*)?$",
        r".jpg\2",
        injected,
        flags=re.IGNORECASE,
    )
    return last_frame_url


# ── Pronunciation Fix Pipeline ────────────────────────────────────────────────


def apply_pronunciation_fix(
    kling_video_url: str,
    dialogue: str,
    fal_queue: FalQueueClient,
    fal_ai: "FalAiClient",
    elevenlabs: "ElevenLabsClient",
    clip_key: str = "",
    poll_interval: int = _LIPSYNC_POLL_INTERVAL,
    max_poll_attempts: int = _LIPSYNC_MAX_ATTEMPTS,
) -> str:
    """Replace Kling's audio with ElevenLabs TTS while preserving background ambience.

    Pipeline:
      1. Download Kling video → extract audio track (local ffmpeg)
      2. Upload audio → fal Demucs → separate vocals from ambient background
      3. ElevenLabs TTS (with SSML phoneme control) → upload to fal storage
      4. Kling LipSync (original video + TTS audio) → lip-synced video
      5. Merge non-vocal background stems → mix with TTS (sidechain ducking)
      6. Replace audio on lip-synced video → upload to fal storage → return URL

    Requires ffmpeg to be installed and on PATH.
    Raises on hard failures — callers should wrap in try/except and fall back
    to the original kling_video_url if desired.
    """
    # Lazy imports to avoid circular deps at module load time.
    from services.elevenlabs import ElevenLabsClient  # noqa: F401 (type check only)
    from services.fal_ai import FalAiClient  # noqa: F401 (type check only)

    _logger.info(
        "Applying pronunciation fix",
        extra={"clip_key": clip_key, "text_preview": dialogue[:80]},
    )

    with tempfile.TemporaryDirectory() as _tmpdir:
        tmp = Path(_tmpdir)

        # ── 1. Download Kling video ───────────────────────────────────────────
        video_path = tmp / "kling.mp4"
        _download_to_file(kling_video_url, video_path)

        # ── 1b. Extract audio track ───────────────────────────────────────────
        audio_path = tmp / "kling_audio.mp3"
        _ffmpeg(["-i", str(video_path), "-vn", "-q:a", "0", str(audio_path)])

        # ── 2. Demucs stem separation ─────────────────────────────────────────
        audio_fal_url: str = fal_client.upload_file(str(audio_path))
        all_stems = fal_ai.demucs_separate(audio_fal_url)
        bg_stem_urls = {k: v for k, v in all_stems.items() if k in _BACKGROUND_STEMS}

        _logger.info(
            "Demucs stems ready",
            extra={
                "clip_key": clip_key,
                "background_stems": list(bg_stem_urls.keys()),
                "has_vocals": "vocals" in all_stems,
            },
        )

        # ── 3. ElevenLabs TTS ─────────────────────────────────────────────────
        tts_fal_url = elevenlabs.text_to_speech_and_upload(dialogue)

        # ── 4. Kling LipSync ──────────────────────────────────────────────────
        _logger.info("Submitting Kling LipSync job", extra={"clip_key": clip_key})
        lipsync_submit = fal_queue.submit(
            KLING_LIPSYNC_MODEL,
            {"video_url": kling_video_url, "audio_url": tts_fal_url},
        )
        lipsync_result = fal_queue.wait_for_completion(
            lipsync_submit,
            poll_interval=poll_interval,
            max_attempts=max_poll_attempts,
        )
        lipsync_url = _extract_lipsync_video_url(lipsync_result)
        _logger.info("LipSync complete", extra={"clip_key": clip_key, "lipsync_url": lipsync_url})

        # ── 5 & 6. Download, mix audio, replace on video ──────────────────────
        lipsync_path = tmp / "lipsync.mp4"
        _download_to_file(lipsync_url, lipsync_path)

        tts_path = tmp / "tts.mp3"
        _download_to_file(tts_fal_url, tts_path)

        final_path = tmp / "final.mp4"

        if bg_stem_urls:
            stem_paths: list[Path] = []
            for stem_name, url in bg_stem_urls.items():
                p = tmp / f"stem_{stem_name}.mp3"
                _download_to_file(url, p)
                stem_paths.append(p)

            bg_path = tmp / "background.mp3"
            _merge_audio_stems(stem_paths, bg_path)
            _mix_tts_and_background(lipsync_path, tts_path, bg_path, final_path)
        else:
            # No separable background found — just replace with clean TTS.
            _replace_audio(lipsync_path, tts_path, final_path)

        # ── Upload final video to fal ephemeral storage ───────────────────────
        final_fal_url: str = fal_client.upload_file(str(final_path))

        _logger.info(
            "Pronunciation fix complete",
            extra={"clip_key": clip_key, "final_url": final_fal_url},
        )
        return final_fal_url


# ── Private helpers ───────────────────────────────────────────────────────────


def _download_to_file(url: str, dest: Path) -> None:
    with httpx.Client(timeout=180.0, follow_redirects=True) as client:
        resp = client.get(url)
        resp.raise_for_status()
    dest.write_bytes(resp.content)


def _ffmpeg(args: list[str], timeout: int = 300) -> None:
    """Run ffmpeg with the given arguments. Raises RuntimeError on failure."""
    cmd = ["ffmpeg", "-y", "-loglevel", "error"] + args
    result = subprocess.run(cmd, capture_output=True, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg exited {result.returncode}: "
            f"{result.stderr.decode('utf-8', errors='replace')}"
        )


def _merge_audio_stems(stem_paths: list[Path], output: Path) -> None:
    """Merge N audio stem files into one mixed track using ffmpeg amix."""
    import shutil

    n = len(stem_paths)
    if n == 1:
        shutil.copy(str(stem_paths[0]), str(output))
        return

    inputs: list[str] = []
    for p in stem_paths:
        inputs += ["-i", str(p)]

    filter_str = "".join(f"[{i}:a]" for i in range(n)) + f"amix=inputs={n}:duration=longest[out]"
    _ffmpeg(inputs + ["-filter_complex", filter_str, "-map", "[out]", str(output)])


def _mix_tts_and_background(
    video_path: Path,
    tts_path: Path,
    bg_path: Path,
    output: Path,
) -> None:
    """Mix TTS (foreground) with auto-ducked background, replacing the video's audio.

    Uses sidechain compression so background ducks under speech and rises
    naturally in pauses — the same technique used in professional dubbing.

    ffmpeg input map:
      0 = lipsync video (video+audio, we take only video)
      1 = tts speech (foreground)
      2 = background ambient (compressed by sidechain)
    """
    _ffmpeg(
        [
            "-i",
            str(video_path),
            "-i",
            str(tts_path),
            "-i",
            str(bg_path),
            "-filter_complex",
            "[2:a][1:a]sidechaincompress="
            "threshold=0.003:ratio=20:attack=5:release=100"
            "[comp_bg];"
            "[comp_bg][1:a]amix=inputs=2:duration=first[mixed]",
            "-map",
            "0:v",
            "-map",
            "[mixed]",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            str(output),
        ]
    )


def _replace_audio(video_path: Path, audio_path: Path, output: Path) -> None:
    """Replace the audio track on a video with a clean audio file.

    Pads the audio with silence (apad) so the full video duration is
    preserved even when the replacement audio is shorter than the video.
    """
    _ffmpeg(
        [
            "-i",
            str(video_path),
            "-i",
            str(audio_path),
            "-filter_complex",
            "[1:a]apad[a]",
            "-map",
            "0:v",
            "-map",
            "[a]",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            str(output),
        ]
    )


def _extract_lipsync_video_url(result: dict[str, Any]) -> str:
    """Pull the video URL from a Kling LipSync result dict."""
    url = (
        result.get("video", {}).get("url")
        if isinstance(result.get("video"), dict)
        else result.get("url", "")
    )
    if not url:
        url = result.get("output_url", "")
    if not url:
        raise RuntimeError(f"Cannot find video URL in LipSync response: {result}")
    return url
