from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

import gspread

from logging_config import get_logger
from pipeline.clip_generator import (
    build_clip_context,
    build_last_frame_url,
    extract_video_url_from_kling_result,
    submit_clip_to_kling,
)
from services.caption_svc import add_captions
from pipeline.video_blueprint import generate_blueprint, validate_blueprint
from pipeline.video_stitcher import stitch_clips
from services.cloudinary_svc import CloudinaryService
from services.fal_queue import FalQueueClient
from services.gemini import GeminiClient
from services.google_drive import GoogleDriveService
from sheets.producer_reader import ClipRow, GeneratingJob, read_clips_for_job
from sheets.producer_writer import (
    claim_job,
    mark_job_done,
    mark_job_error,
    save_blueprint_to_job,
    update_clip,
)


# Signature: (job_key, event_type, data)
# event_type is one of the string constants below; data is event-specific.
ProgressCallback = Callable[[str, str, Any], None]

# ---- Event types ----
EVT_BLUEPRINT = "blueprint"
EVT_BLUEPRINT_READY = "blueprint_ready"
EVT_CLIPS_LOADED = "clips_loaded"
EVT_CLIP_START = "clip_start"
EVT_CLIP_POLL = "clip_poll"
EVT_CLIP_DONE = "clip_done"
EVT_CLIP_ERROR = "clip_error"
EVT_STITCH_START = "stitch_start"
EVT_STITCH_POLL = "stitch_poll"
EVT_STITCH_DONE = "stitch_done"
EVT_DRIVE_UPLOAD = "drive_upload"
EVT_DRIVE_ERROR = "drive_error"
EVT_DEMO_INJECT = "demo_inject"
EVT_DEMO_SKIP = "demo_skip"
EVT_CAPTION_START = "caption_start"
EVT_CAPTION_POLL = "caption_poll"
EVT_CAPTION_DONE = "caption_done"
EVT_CAPTION_ERROR = "caption_error"
EVT_JOB_DONE = "job_done"
EVT_JOB_ERROR = "job_error"


@dataclass
class ClipStatus:
    clip_key: str
    clip_index: int
    status: str  # queued | generating | done | error | skipped
    error: str = ""
    clip_video_url: str = ""


@dataclass
class ProducerJobResult:
    job_key: str
    success: bool
    fal_video_url: str = ""
    drive_video_url: str = ""
    error: str = ""
    clips: list[ClipStatus] = field(default_factory=list)


def run_producer_job(
    job: GeneratingJob,
    jobs_worksheet: gspread.Worksheet,
    clips_worksheet: gspread.Worksheet,
    drive_folder_id: str,
    demos_worksheet: gspread.Worksheet | None = None,
    progress_cb: ProgressCallback | None = None,
    poll_interval: int = 30,
    max_poll_attempts: int = 40,
    enable_captions: bool = False,
) -> ProducerJobResult:
    """Run the complete producer pipeline for one job.

    Phases (mirrors n8n Producer v2):
      1. Validate / generate video blueprint
      2. Sequential clip generation (SceneLock frame chaining)
      2.5. Optionally burn word-by-word captions into each generated clip
      3. Stitch clips via FAL FFmpeg
      4. Upload final video to Google Drive
      5. Write URLs + status back to sheet

    When enable_captions=True, captions are applied to each Kling-generated clip before
    stitching. The app demo clip (injected via _inject_demo_if_needed) is never captioned.
    The SceneLock last-frame chain always uses the original (pre-caption) Cloudinary URL so
    the face-swap anchor is not contaminated by burned-in text.
    """
    logger = get_logger(__name__, job_key=job.job_key)
    gemini = GeminiClient()
    fal_queue = FalQueueClient()
    cloudinary_svc = CloudinaryService()
    drive_svc = GoogleDriveService()

    def emit(event_type: str, data: Any = None) -> None:
        if progress_cb:
            progress_cb(job.job_key, event_type, data)

    def fail(reason: str) -> ProducerJobResult:
        logger.error("Producer job failed: %s", reason)
        mark_job_error(jobs_worksheet, job.row_index, reason)
        emit(EVT_JOB_ERROR, reason)
        return ProducerJobResult(job_key=job.job_key, success=False, error=reason)

    # ── Claim: transition generating → producing so concurrent workers/sessions
    # won't pick up this row.  Any failure path calls mark_job_error which
    # overwrites the status, so this is purely a forward-only guard.
    claim_job(jobs_worksheet, job.row_index)

    # ── Phase 1: Blueprint ────────────────────────────────────────────────────
    emit(EVT_BLUEPRINT, "Validating blueprint")
    blueprint: dict[str, Any]

    if validate_blueprint(job.prompt_template):
        blueprint = (
            json.loads(job.prompt_template)
            if isinstance(job.prompt_template, str)
            else job.prompt_template
        )
        logger.info("Using existing blueprint from sheet")
    else:
        if not job.analysis_json:
            return fail(
                "No valid prompt_template and no analysis_json — cannot generate blueprint."
            )
        logger.info("Generating blueprint from analysis_json via Gemini")
        try:
            blueprint = generate_blueprint(job.analysis_json, gemini)
            save_blueprint_to_job(
                jobs_worksheet, job.row_index, json.dumps(blueprint, ensure_ascii=False)
            )
        except Exception as exc:
            return fail(f"[Blueprint] {type(exc).__name__}: {exc}")

    emit(EVT_BLUEPRINT_READY, blueprint.get("video_type", "unknown"))

    # ── Phase 1b: Validate required fields ───────────────────────────────────
    if not job.swapped_start_frame_url:
        return fail(
            "No start frame image available for this job. "
            "For Jobs sheet: ensure the batch pipeline has run and written swapped_start_frame_url. "
            "For Personas sheet: ensure the Persona Batch pipeline has run and persona_image is populated."
        )

    # ── Phase 2: Load clips ───────────────────────────────────────────────────
    clips = read_clips_for_job(clips_worksheet, job.job_key)
    if not clips:
        return fail(f"No clips found in Clips sheet for job_key='{job.job_key}'.")

    total_clips = len(clips)
    clip_statuses: list[ClipStatus] = [
        ClipStatus(
            clip_key=c.clip_key,
            clip_index=c.clip_index,
            status=c.status,
            clip_video_url=c.clip_video_url,
        )
        for c in clips
    ]
    emit(EVT_CLIPS_LOADED, {"total": total_clips, "job_key": job.job_key})

    # ── Phase 3: Sequential clip generation ──────────────────────────────────
    # last_frame_urls[clip_index] = Cloudinary last-frame URL of that clip,
    # used as the start frame for the next even-indexed clip.
    last_frame_urls: dict[int, str] = {}

    # Pre-populate from clips that already have a start_frame_url written
    # (supports resumption when the pipeline was previously interrupted).
    for clip in clips:
        if clip.start_frame_url and clip.clip_index > 1:
            last_frame_urls[clip.clip_index - 1] = clip.start_frame_url

    for clip in clips:
        cs = next(s for s in clip_statuses if s.clip_key == clip.clip_key)

        # Already finished — collect last frame for chaining and continue.
        if clip.status == "done" and clip.clip_video_url:
            last_frame_urls[clip.clip_index] = build_last_frame_url(
                clip.clip_video_url, clip.duration_s
            )
            cs.status = "done"
            cs.clip_video_url = clip.clip_video_url
            logger.info("Skipping already-done clip", extra={"clip_key": clip.clip_key})
            continue

        # Reset clips stuck in "generating" from a crashed previous run.
        if clip.status == "generating":
            update_clip(clips_worksheet, clip.row_index, {"status": "queued"})
            clip.status = "queued"  # noqa: PLW2901 — mutate local copy only

        if clip.status not in ("queued", "error"):
            cs.status = "skipped"
            continue

        # ── 3a. Lock clip ─────────────────────────────────────────────────
        cs.status = "generating"
        emit(EVT_CLIP_START, {"clip_index": clip.clip_index, "clip_key": clip.clip_key})
        update_clip(clips_worksheet, clip.row_index, {"status": "generating"})

        # ── 3b. Build context ─────────────────────────────────────────────
        try:
            ctx = build_clip_context(
                clip_index=clip.clip_index,
                total_clips=total_clips,
                dialogue=clip.dialogue,
                duration_s=clip.duration_s,
                clip_key=clip.clip_key,
                swapped_start_frame_url=job.swapped_start_frame_url,
                blueprint=blueprint,
                last_frame_url=last_frame_urls.get(clip.clip_index - 1),
                start_frame_url=clip.start_frame_url or None,
            )
        except Exception as exc:
            err = f"[BuildContext clip={clip.clip_index}] {exc}"
            logger.exception("Failed to build clip context", extra={"clip_key": clip.clip_key})
            cs.status = "error"
            cs.error = err
            update_clip(clips_worksheet, clip.row_index, {"status": "error", "error": err})
            continue

        # ── 3c. Submit to Kling ───────────────────────────────────────────
        try:
            queue_result = submit_clip_to_kling(ctx, fal_queue)
        except Exception as exc:
            err = f"[KlingSubmit clip={clip.clip_index}] {exc}"
            logger.exception("Failed to submit to Kling", extra={"clip_key": clip.clip_key})
            cs.status = "error"
            cs.error = err
            update_clip(clips_worksheet, clip.row_index, {"status": "error", "error": err})
            continue

        # Persist task ID and frame URLs immediately (survivable on crash).
        update_clip(
            clips_worksheet,
            clip.row_index,
            {
                "kling_task_id": queue_result.request_id,
                "start_frame_url": ctx.kling_start_frame_url,
                "end_frame_url": ctx.kling_end_frame_url or "",
            },
        )

        # ── 3d. Poll Kling ────────────────────────────────────────────────
        def _poll_cb(attempt: int, status: str) -> None:
            emit(
                EVT_CLIP_POLL,
                {"clip_index": clip.clip_index, "attempt": attempt, "status": status},
            )

        try:
            fal_result = fal_queue.wait_for_completion(
                queue_result,
                poll_interval=poll_interval,
                max_attempts=max_poll_attempts,
                on_poll=_poll_cb,
            )
        except Exception as exc:
            err = f"[KlingPoll clip={clip.clip_index}] {exc}"
            logger.exception("Clip polling failed", extra={"clip_key": clip.clip_key})
            cs.status = "error"
            cs.error = err
            update_clip(clips_worksheet, clip.row_index, {"status": "error", "error": err})
            continue

        # ── 3e. Extract video URL ─────────────────────────────────────────
        try:
            raw_video_url = extract_video_url_from_kling_result(fal_result)
        except Exception as exc:
            err = f"[KlingResult clip={clip.clip_index}] {exc}"
            cs.status = "error"
            cs.error = err
            update_clip(clips_worksheet, clip.row_index, {"status": "error", "error": err})
            continue

        # ── 3f. Upload to Cloudinary ──────────────────────────────────────
        try:
            uploaded_url = cloudinary_svc.upload_video(
                raw_video_url,
                folder=f"ai-generated/clips/{job.job_key}",
                public_id=f"{job.job_key}_clip_{clip.clip_index}",
            )
        except Exception as exc:
            err = f"[CloudinaryUpload clip={clip.clip_index}] {exc}"
            logger.exception("Cloudinary clip upload failed", extra={"clip_key": clip.clip_key})
            cs.status = "error"
            cs.error = err
            update_clip(clips_worksheet, clip.row_index, {"status": "error", "error": err})
            continue

        # ── 3g. Optionally add word-by-word captions ───────────────────────
        # The captioned clip is uploaded to Cloudinary so the URL is persistent
        # across session boundaries (resumption after a failed stitch).
        # The SceneLock last-frame chain always uses the original (pre-caption)
        # uploaded_url so Kling never receives a frame with burned-in text.
        final_clip_url = uploaded_url
        if enable_captions:
            emit(EVT_CAPTION_START, {"clip_index": clip.clip_index, "clip_key": clip.clip_key})

            def _caption_poll_cb(attempt: int, status: str) -> None:
                emit(
                    EVT_CAPTION_POLL,
                    {"clip_index": clip.clip_index, "attempt": attempt, "status": status},
                )

            try:
                captioned_fal_url = add_captions(
                    uploaded_url,
                    fal_queue,
                    on_poll=_caption_poll_cb,
                )
                # Re-upload captioned clip to Cloudinary for persistence.
                final_clip_url = cloudinary_svc.upload_video(
                    captioned_fal_url,
                    folder=f"ai-generated/clips/{job.job_key}",
                    public_id=f"{job.job_key}_clip_{clip.clip_index}_captioned",
                )
                emit(
                    EVT_CAPTION_DONE,
                    {"clip_index": clip.clip_index, "clip_key": clip.clip_key, "url": final_clip_url},
                )
                logger.info(
                    "Clip captioned",
                    extra={"clip_key": clip.clip_key, "captioned_url": final_clip_url},
                )
            except Exception as exc:
                # Non-fatal: warn and fall back to the uncaptioned clip.
                warn = f"[Caption clip={clip.clip_index}] {exc} — using uncaptioned clip"
                logger.warning(warn, extra={"clip_key": clip.clip_key})
                emit(EVT_CAPTION_ERROR, {"clip_index": clip.clip_index, "error": warn})
                final_clip_url = uploaded_url

        # ── 3h. Mark clip done + update sheet ─────────────────────────────
        cs.status = "done"
        cs.clip_video_url = final_clip_url
        update_clip(
            clips_worksheet,
            clip.row_index,
            {"status": "done", "clip_video_url": final_clip_url},
        )
        emit(
            EVT_CLIP_DONE,
            {"clip_index": clip.clip_index, "clip_key": clip.clip_key, "url": final_clip_url},
        )

        # ── 3i. Chain last frame to next clip (always uses original) ───────
        # Use the pre-caption Cloudinary URL so Kling never receives a frame
        # with burned-in text as its SceneLock start-frame input.
        clip_last_frame = build_last_frame_url(uploaded_url, clip.duration_s)
        last_frame_urls[clip.clip_index] = clip_last_frame

        next_clip = next((c for c in clips if c.clip_index == clip.clip_index + 1), None)
        if next_clip:
            update_clip(
                clips_worksheet,
                next_clip.row_index,
                {"start_frame_url": clip_last_frame},
            )

        logger.info(
            "Clip done",
            extra={"clip_key": clip.clip_key, "clip_index": clip.clip_index},
        )

    # ── Phase 4: Check all clips done ─────────────────────────────────────────
    done_clips = [cs for cs in clip_statuses if cs.status == "done" and cs.clip_video_url]
    if len(done_clips) < total_clips:
        errored = [cs for cs in clip_statuses if cs.status == "error"]
        return fail(
            f"Not all clips succeeded: {len(done_clips)}/{total_clips} done, "
            f"{len(errored)} errored."
        )

    done_clips.sort(key=lambda cs: cs.clip_index)
    video_urls = [cs.clip_video_url for cs in done_clips]

    # ── Phase 4.5: Inject app demo (if configured) ────────────────────────────
    if demos_worksheet is not None:
        try:
            video_urls, demo_id = _inject_demo_if_needed(
                video_urls=video_urls,
                job=job,
                demos_worksheet=demos_worksheet,
                logger=logger,
            )
            if demo_id:
                emit(EVT_DEMO_INJECT, {"demo_id": demo_id, "total_clips": len(video_urls)})
            else:
                emit(EVT_DEMO_SKIP, "No insert_demo segment found in script")
        except Exception as exc:
            logger.warning(
                "Demo injection failed — continuing without demo: %s",
                exc,
                extra={"job_key": job.job_key},
            )
            emit(EVT_DEMO_SKIP, f"Demo injection error: {exc}")

    # ── Phase 5: Stitch clips ─────────────────────────────────────────────────
    emit(EVT_STITCH_START, {"clip_count": len(video_urls)})
    try:
        fal_video_url = stitch_clips(
            video_urls,
            fal_queue,
            poll_interval=poll_interval,
            max_poll_attempts=max_poll_attempts,
            on_poll=lambda attempt, status: emit(
                EVT_STITCH_POLL, {"attempt": attempt, "status": status}
            ),
        )
    except Exception as exc:
        return fail(f"[Stitch] {type(exc).__name__}: {exc}")

    emit(EVT_STITCH_DONE, fal_video_url)

    # ── Phase 6: Upload to Google Drive ───────────────────────────────────────
    emit(EVT_DRIVE_UPLOAD, f"Uploading {job.job_key}.mp4 to Google Drive")
    drive_video_url = ""
    try:
        file_info = drive_svc.upload_video_from_url(
            video_url=fal_video_url,
            filename=f"{job.job_key}.mp4",
            folder_id=drive_folder_id,
        )
        drive_video_url = file_info.get("webViewLink", "")
    except Exception as exc:
        # Non-fatal: save the Fal URL as a fallback so the job isn't lost.
        logger.exception("Google Drive upload failed", extra={"job_key": job.job_key})
        drive_video_url = fal_video_url
        emit(EVT_DRIVE_ERROR, f"Drive upload failed ({exc}) — using Fal URL as fallback.")

    # ── Phase 7: Finalise ─────────────────────────────────────────────────────
    mark_job_done(
        jobs_worksheet,
        job.row_index,
        fal_video_url=fal_video_url,
        drive_video_url=drive_video_url,
    )
    emit(EVT_JOB_DONE, {"fal_video_url": fal_video_url, "drive_video_url": drive_video_url})

    logger.info(
        "Producer job complete",
        extra={
            "job_key": job.job_key,
            "fal_video_url": fal_video_url,
            "drive_video_url": drive_video_url,
        },
    )
    return ProducerJobResult(
        job_key=job.job_key,
        success=True,
        fal_video_url=fal_video_url,
        drive_video_url=drive_video_url,
        clips=clip_statuses,
    )


# ── Demo injection helper ─────────────────────────────────────────────────────


def _inject_demo_if_needed(
    video_urls: list[str],
    job: "GeneratingJob",
    demos_worksheet: "gspread.Worksheet",
    logger: Any,
) -> tuple[list[str], str]:
    """Parse the job's script JSON, find an insert_demo segment, select the
    best demo, and return (new_video_urls, demo_id).

    Returns the original list unchanged (with empty demo_id) when:
      - final_script_json is absent or unparseable
      - no segment is marked insert_demo / is_demo
      - the App Demo sheet is empty

    Raises ValueError if demos exist but none survive gender filtering
    (surfaces to caller so it can decide whether to abort or skip).
    """
    from pipeline.demo_selector import select_demo
    from sheets.demo_reader import read_demo_rows

    _no_injection: tuple[list[str], str] = (video_urls, "")

    if not job.final_script_json:
        logger.info("No final_script_json on job — skipping demo injection")
        return _no_injection

    try:
        script_data = json.loads(job.final_script_json)
    except json.JSONDecodeError:
        logger.warning(
            "final_script_json is not valid JSON — skipping demo injection",
            extra={"job_key": job.job_key},
        )
        return _no_injection

    # Support both n8n-style {"script": [..., {"insert_demo": true}]}
    # and Python-style {"segments": [..., {"is_demo": true}]}.
    segments: list[Any] = script_data.get("script") or script_data.get("segments") or []

    insert_idx: int | None = None
    for i, seg in enumerate(segments):
        if not isinstance(seg, dict):
            continue
        if seg.get("insert_demo") or seg.get("is_demo"):
            insert_idx = i
            break

    if insert_idx is None:
        logger.info("No insert_demo segment in script — skipping demo injection")
        return _no_injection

    # Full script text for keyword scoring (same as n8n fullScript).
    full_script = " ".join(
        str(seg.get("text") or seg.get("dialogue") or "")
        for seg in segments
        if isinstance(seg, dict)
    )

    demos = read_demo_rows(demos_worksheet)
    if not demos:
        logger.warning("App Demo sheet is empty — skipping demo injection")
        return _no_injection

    best_demo = select_demo(demos, full_script, job.gender)

    # insert_idx is 0-based position in segments array.
    # clip_index is 1-based and maps 1:1 to segment position.
    # video_urls is 0-based: video_urls[0] = clip 1, video_urls[1] = clip 2, …
    # We want to insert the demo AFTER segment at insert_idx, so after
    # clip (insert_idx + 1), meaning we splice at position (insert_idx + 1).
    url_insert_pos = insert_idx + 1
    new_urls = list(video_urls)
    new_urls.insert(url_insert_pos, best_demo.video_url)

    logger.info(
        "App demo injected into clip list",
        extra={
            "demo_id": best_demo.demo_id,
            "demo_url": best_demo.video_url,
            "insert_after_clip_index": insert_idx + 1,
            "total_clips_with_demo": len(new_urls),
        },
    )
    return new_urls, best_demo.demo_id
