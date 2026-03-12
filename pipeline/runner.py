from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from logging_config import get_logger
from pipeline.clip_timing import TimedClip, assign_clip_durations
from pipeline.context import JobContext, build_context
from pipeline.face_swap import face_swap
from pipeline.script_writer import RewrittenScript, rewrite_script
from pipeline.video_analysis import VideoAnalysisResult, analyze_video, compute_anchor_frame_timestamp
from services.cloudinary_svc import CloudinaryService
from services.fal_ai import FalAiClient
from services.gemini import GeminiClient


@dataclass
class StepResult:
    name: str
    success: bool
    duration_seconds: float
    output: Any = None
    error: str | None = None


@dataclass
class PipelineResult:
    context: JobContext | None = None
    analysis: VideoAnalysisResult | None = None
    anchor_frame_url: str | None = None
    swapped_image_url: str | None = None
    gender: str | None = None
    script: RewrittenScript | None = None
    clips: list[TimedClip] = field(default_factory=list)
    steps: list[StepResult] = field(default_factory=list)
    success: bool = False
    error: str | None = None

    @property
    def total_duration(self) -> float:
        return sum(s.duration_seconds for s in self.steps)


ProgressCallback = Callable[[str, str], None]  # (step_name, message)


def _run_step(
    name: str,
    logger,
    fn: Callable,
    progress_cb: ProgressCallback | None,
) -> tuple[Any, StepResult]:
    logger.info("Starting pipeline step", extra={"step": name})
    if progress_cb:
        progress_cb(name, f"Starting {name}...")

    start = time.time()
    try:
        result = fn()
        elapsed = time.time() - start
        step = StepResult(name=name, success=True, duration_seconds=round(elapsed, 2), output=result)
        logger.info(
            "Completed pipeline step",
            extra={"step": name, "duration_seconds": round(elapsed, 2)},
        )
        if progress_cb:
            progress_cb(name, f"✓ {name} completed in {elapsed:.1f}s")
        return result, step
    except Exception as e:
        elapsed = time.time() - start
        step = StepResult(name=name, success=False, duration_seconds=round(elapsed, 2), error=str(e))
        logger.exception(
            "Pipeline step failed",
            extra={
                "step": name,
                "duration_seconds": round(elapsed, 2),
            },
        )
        if progress_cb:
            progress_cb(name, f"✗ {name} failed: {e}")
        raise


def run_pipeline(
    source_video_url: str,
    reference_image_url: str,
    video_cache: dict,
    progress_cb: ProgressCallback | None = None,
    regenerate_image_only: bool = False,
) -> PipelineResult:
    """Run the video cloning pipeline for a single job.

    When regenerate_image_only=True, we:
    - Reuse cached analysis, anchor frame, gender, and script
    - Force a new face swap + Cloudinary upload
    - Recompute clip timings from the cached script
    This lets callers keep all non-image data stable while regenerating the image.
    """
    result = PipelineResult()

    run_id = str(uuid.uuid4())
    logger = get_logger(
        __name__,
        run_id=run_id,
        source_video_url=source_video_url,
        reference_image_url=reference_image_url,
        regenerate_image_only=regenerate_image_only,
    )

    gemini = GeminiClient()
    fal = FalAiClient()
    cloudinary_svc = CloudinaryService()

    # Step 1: Build context
    try:
        logger.info("Starting pipeline", extra={"step": "Build Context"})
        ctx, step = _run_step(
            "Build Context",
            logger,
            lambda: build_context(source_video_url, reference_image_url),
            progress_cb,
        )
        result.context = ctx
        result.steps.append(step)
    except Exception as e:
        result.error = f"[Build Context] {type(e).__name__}: {e}"
        return result

    # Enrich logger with stable job context once we have it.
    logger = get_logger(
        __name__,
        run_id=run_id,
        source_video_url=source_video_url,
        reference_image_url=reference_image_url,
        regenerate_image_only=regenerate_image_only,
        video_key=ctx.video_key,
        job_key=ctx.job_key,
    )

    logger.info(
        "Context built for pipeline run",
        extra={
            "cloud_name": ctx.cloud_name,
            "video_public_id": ctx.video_public_id,
        },
    )

    # Initialize / fetch per-video cache entry. This acts as a lightweight
    # checkpoint store so we can resume work across runs in both single and batch modes.
    cache_entry = video_cache.setdefault(ctx.video_key, {})

    # Within a single video, we may run the pipeline multiple times with
    # different reference images. Analysis and anchor frame are purely
    # video-level, but the swapped image, detected gender, and script all
    # depend on the reference image. Track those in a separate per-reference
    # bucket so we don't accidentally reuse image-specific data across rows.
    ref_variants = cache_entry.setdefault("image_variants", {})
    ref_entry = ref_variants.setdefault(ctx.reference_image_url, {})

    # Step 2: Check cache / Analyze video
    analysis: VideoAnalysisResult | None = cache_entry.get("analysis")
    cache_hit = analysis is not None

    logger.info(
        "Video analysis cache check",
        extra={"cache_hit": cache_hit},
    )

    if analysis is not None:
        if progress_cb:
            progress_cb("Video Analysis", "Cache hit — skipping Gemini video analysis")
        result.steps.append(StepResult("Video Analysis (cached)", True, 0.0, analysis))
    else:
        if regenerate_image_only:
            result.error = "[Video Analysis] Cannot regenerate image-only because analysis is not cached. Run full pipeline first."
            return result
        try:
            analysis, step = _run_step(
                "Video Analysis",
                logger,
                lambda: analyze_video(ctx.source_video_url, gemini),
                progress_cb,
            )
            cache_entry["analysis"] = analysis
            result.steps.append(step)
        except Exception as e:
            result.error = f"[Video Analysis] {type(e).__name__}: {e}"
            return result

    result.analysis = analysis

    # Step 3: Compute anchor frame + build Cloudinary URL
    frame_url: str | None = cache_entry.get("anchor_frame_url")
    logger.info(
        "Anchor frame cache check",
        extra={"cache_hit": frame_url is not None},
    )

    if frame_url is not None:
        if progress_cb:
            progress_cb(
                "Anchor Frame",
                "Cache hit — reusing anchor frame & frame URL"
                if regenerate_image_only
                else "Cache hit — skipping anchor frame & frame URL computation",
            )
        result.anchor_frame_url = frame_url
        result.steps.append(StepResult("Anchor Frame (cached)", True, 0.0, frame_url))
        result.steps.append(StepResult("Frame URL (cached)", True, 0.0, frame_url))
    else:
        if regenerate_image_only:
            result.error = "[Anchor Frame] Cannot regenerate image-only because anchor frame is not cached. Run full pipeline first."
            return result
        try:
            timestamp, step = _run_step(
                "Anchor Frame",
                logger,
                lambda: compute_anchor_frame_timestamp(analysis),
                progress_cb,
            )
            frame_url, step2 = _run_step(
                "Frame URL",
                logger,
                lambda: cloudinary_svc.build_frame_url(ctx.cloud_name, ctx.video_public_id, timestamp),
                progress_cb,
            )
            result.anchor_frame_url = frame_url
            cache_entry["anchor_frame_url"] = frame_url
            result.steps.extend([step, step2])
        except Exception as e:
            result.error = f"[Anchor Frame] {type(e).__name__}: {e}"
            return result

    # Step 4 & 5: Face swap + Upload to Cloudinary
    # We cache the final swapped image URL (post-upload). If present, we can
    # safely skip both the face swap and upload steps on subsequent runs.
    swapped_final_url: str | None = ref_entry.get("swapped_image_url")
    logger.info(
        "Swapped image cache check",
        extra={"cache_hit": swapped_final_url is not None},
    )
    if swapped_final_url is not None and not regenerate_image_only:
        if progress_cb:
            progress_cb("Face Swap", "Cache hit — skipping face swap and upload")
        result.swapped_image_url = swapped_final_url
        result.steps.append(StepResult("Face Swap (cached)", True, 0.0, swapped_final_url))
        result.steps.append(StepResult("Upload to Cloudinary (cached)", True, 0.0, swapped_final_url))
    else:
        try:
            face_step_name = "Face Swap (regen)" if regenerate_image_only and swapped_final_url is not None else "Face Swap"
            swapped_url, step = _run_step(
                face_step_name,
                logger,
                lambda: face_swap(frame_url, ctx.reference_image_url, fal),
                progress_cb,
            )
            result.swapped_image_url = swapped_url
            result.steps.append(step)
        except Exception as e:
            result.error = f"[Face Swap] {type(e).__name__}: {e}"
            return result

        try:
            upload_step_name = "Upload to Cloudinary (regen)" if regenerate_image_only and swapped_final_url is not None else "Upload to Cloudinary"
            uploaded_url, step = _run_step(
                upload_step_name,
                logger,
                # Let Cloudinary assign the public_id (mirrors n8n pipeline behavior)
                lambda: cloudinary_svc.upload_image(swapped_url),
                progress_cb,
            )
            result.swapped_image_url = uploaded_url
            ref_entry["swapped_image_url"] = uploaded_url
            result.steps.append(step)
        except Exception as e:
            result.error = f"[Upload to Cloudinary] {type(e).__name__}: {e}"
            return result

    # Step 6: Detect gender
    gender: str | None = ref_entry.get("gender")
    logger.info(
        "Gender cache check",
        extra={"cache_hit": gender is not None},
    )
    if gender is not None:
        if progress_cb:
            progress_cb(
                "Gender Detection",
                "Reusing cached gender from previous run"
                if regenerate_image_only
                else "Cache hit — skipping gender detection",
            )
        result.gender = gender
        result.steps.append(StepResult("Gender Detection (cached)", True, 0.0, gender))
    else:
        if regenerate_image_only:
            # For image-only regenerations, we don't need to re-run gender
            # detection. If it isn't cached, just skip this step rather than
            # failing the entire regeneration.
            if progress_cb:
                progress_cb(
                    "Gender Detection",
                    "Skipping gender detection for image-only regeneration (no cached gender).",
                )
            result.gender = None
            result.steps.append(
                StepResult(
                    name="Gender Detection (skipped for regen)",
                    success=True,
                    duration_seconds=0.0,
                    output=None,
                )
            )
        else:
            try:
                gender, step = _run_step(
                    "Gender Detection",
                    logger,
                    lambda: gemini.detect_gender(result.swapped_image_url),
                    progress_cb,
                )
                result.gender = gender
                ref_entry["gender"] = gender
                result.steps.append(step)
            except Exception as e:
                result.error = f"[Gender Detection] {type(e).__name__}: {e}"
                return result

    # Step 7: Check script cache / Rewrite script
    # We now treat the script as **video-level** data so that multiple
    # reference images for the same source video can reuse the same script.
    # For backward compatibility with older cache entries, we also look for
    # a script stored under the per-reference image bucket.
    cached_script = cache_entry.get("script") or ref_entry.get("script")
    logger.info(
        "Script cache check",
        extra={"cache_hit": cached_script is not None},
    )

    if cached_script:
        if progress_cb:
            progress_cb(
                "Script Rewrite",
                "Script cache hit — reusing video-level script for this reference image",
            )
        script = cached_script
        result.steps.append(StepResult("Script Rewrite (cached)", True, 0.0, script))
    else:
        if regenerate_image_only:
            result.error = "[Script Rewrite] Cannot regenerate image-only because script is not cached. Run full pipeline first."
            return result
        try:
            original_lines = [d.text for d in analysis.spoken_dialogue]
            script, step = _run_step(
                "Script Rewrite",
                logger,
                lambda: rewrite_script(
                    original_lines,
                    result.gender or "unknown",
                    analysis,
                    gemini,
                ),
                progress_cb,
            )
            result.steps.append(step)
            # Save script to cache at the video level so subsequent jobs for
            # the same source video (even with different reference images)
            # reuse the same script instead of regenerating it.
            cache_entry["script"] = script
            # Also persist on the reference image entry to gracefully handle
            # any legacy callers that still expect script there.
            ref_entry["script"] = script
        except Exception as e:
            result.error = f"[Script Rewrite] {type(e).__name__}: {e}"
            return result

    result.script = script

    # Step 8: Assign clip durations
    try:
        clips, step = _run_step(
            "Clip Timing",
            logger,
            lambda: assign_clip_durations(script),
            progress_cb,
        )
        result.clips = clips
        result.steps.append(step)
    except Exception as e:
        result.error = f"Clip timing failed: {e}"
        return result

    result.success = True
    logger.info(
        "Pipeline run completed successfully",
        extra={
            "total_duration_seconds": result.total_duration,
            "steps_count": len(result.steps),
        },
    )
    return result
