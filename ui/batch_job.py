from __future__ import annotations

import queue
import threading
import time
import traceback
from typing import Any

import streamlit as st

from cache import get_session_cache, load_sheet_cache, save_to_sheet_cache
from logging_config import get_logger, setup_logging
from pipeline.parallel_runner import (
    DEFAULT_BATCH_WORKERS,
    MAX_BATCH_WORKERS,
    EVT_ALL_DONE,
    EVT_JOB_PROGRESS,
    EVT_JOB_RESULT,
    EVT_JOB_START,
    BatchWorkerResult,
    ParallelBatchRunner,
)
from pipeline.runner import PipelineResult, run_pipeline
from sheets.auth import get_gspread_client
from sheets.reader import SheetJob, get_worksheet, read_queued_jobs
from sheets.writer import update_job_status, write_job_result


logger = get_logger(__name__, ui_mode="batch")

# ── Session-state keys ────────────────────────────────────────────────────────
_KEY_RUNNING = "batch_running"
_KEY_QUEUE = "batch_event_queue"
_KEY_JOB_STATES = "batch_job_states"
_KEY_JOB_MAP = "batch_job_map"          # job_key → SheetJob (for post-run summary)
_KEY_LOG = "batch_log"
_KEY_RESULTS = "batch_results"          # list[BatchWorkerResult]


def render() -> None:
    setup_logging()
    st.header("Clone Batch")
    st.caption(
        "Analyze source videos, swap faces from a reference image, and generate scripts — "
        "in bulk via Google Sheet."
    )

    with st.expander("ℹ️ How this works & what to prepare", expanded=False):
        st.markdown(
            """
**What this mode does**

Reads every row with `status = queued` from the **Jobs** tab of your Google Sheet, runs the full
pipeline per row (video analysis → face swap → script rewrite), and writes results back to the sheet.
Use this when you have multiple videos to process in one go.

**Typical workflow:**
1. Fill in your Google Sheet's **Jobs** tab with the required columns (see below).
2. Set each row's `status` column to `queued`.
3. Paste the sheet URL here, confirm tab names, and click **Load & Process Queued Jobs**.
4. When this run finishes, rows are left in `generating` status — switch to the **Producer** tab to generate the final stitched videos.

---

**Required columns in the Jobs tab**

| Column | Description |
|---|---|
| `source_video_url` | Publicly accessible URL of the original ad video. |
| `reference_image_url` | Publicly accessible URL of the face photo to swap in. |
| `status` | Set to `queued` for rows you want processed. Other values are skipped. |

Column order does not matter. Additional columns are ignored.

---

**What gets written back to the sheet**

| Column | Description |
|---|---|
| `status` | Updated to `analyzing` → `generating` (or `error`). |
| `job_key` | Unique identifier for this job, used by the Producer tab. |
| `swapped_start_frame_url` | URL of the face-swapped anchor frame image. |
| `final_script_json` | Full rewritten script as JSON. |
| `analysis_json` | Raw Gemini video-analysis output. |
| `face_time_start` / `face_time_end` | Timestamps (seconds) of the best face window in the source video. |
| `extracted_frame_url` | URL of the raw anchor frame before face swap. |
| `total_clips` | Number of script segments generated. |
| `new_frame_gender` | Detected speaker gender (`male` / `female`). |
| `Date generated` | Timestamp of when this row was processed. |
| `error` / `error_message` | Populated only when `status = error`. |

Clips are also written to the **Clips** tab (one row per clip segment).

---

**Optional: VideoCache tab**

If you have a `VideoCache` tab in your sheet, the pipeline will load previously completed video
analyses from it, skipping re-analysis for videos it has already seen. Results are also saved back
so future runs stay fast.
            """
        )

    # Handle any pending image-only regeneration before routing so that results
    # are reflected on the same render cycle.
    regen_target_row = st.session_state.pop("batch_regen_target_row", None)
    if regen_target_row is not None:
        sheet_ctx = st.session_state.get("batch_last_sheet_context")
        completed_jobs: list[dict[str, Any]] | None = st.session_state.get(
            "batch_last_completed_jobs"
        )
        if isinstance(sheet_ctx, dict) and isinstance(completed_jobs, list):
            job = next(
                (j for j in completed_jobs if j.get("row_index") == regen_target_row), None
            )
            if job is not None:
                st.session_state["batch_regen_in_progress"] = True
                try:
                    _regenerate_single_batch_job_image(job, sheet_ctx)
                finally:
                    st.session_state["batch_regen_in_progress"] = False

    # ── Active run ────────────────────────────────────────────────────────────
    if st.session_state.get(_KEY_RUNNING):
        _render_active_run()
        return

    # ── Config form ───────────────────────────────────────────────────────────
    with st.form("batch_form"):
        sheet_url = st.text_input(
            "Google Sheet URL or ID",
            placeholder="https://docs.google.com/spreadsheets/d/...",
        )
        jobs_tab = st.text_input("Jobs Tab Name", value="Jobs")
        clips_tab = st.text_input("Clips Tab Name", value="Clips")
        cache_tab = st.text_input("VideoCache Tab Name (optional)", value="VideoCache")
        target_duration_s = st.number_input(
            "Target video length (seconds, excluding app demo)",
            min_value=10,
            max_value=60,
            value=20,
            step=5,
            help=(
                "The desired spoken duration of the generated script, excluding the app demo cutaway. "
                "Controls word count limits and segment count passed to Gemini. "
                "Default (20s) matches the original behaviour."
            ),
        )
        num_workers = st.number_input(
            "Parallel workers",
            min_value=1,
            max_value=MAX_BATCH_WORKERS,
            value=DEFAULT_BATCH_WORKERS,
            step=1,
            help=(
                f"Number of jobs to process simultaneously (1–{MAX_BATCH_WORKERS}). "
                "Each worker opens its own API connections. "
                "Increase for large batches; keep at 1 to process sequentially."
            ),
        )
        submitted = st.form_submit_button("Load & Process Queued Jobs", type="primary")

    if not submitted:
        _render_batch_review_section()
        return

    if not sheet_url.strip():
        st.error("Please enter a Google Sheet URL or ID.")
        return

    logger.info(
        "Batch submission received",
        extra={
            "sheet_url_or_id": sheet_url.strip(),
            "jobs_tab": jobs_tab.strip(),
            "clips_tab": clips_tab.strip(),
            "cache_tab": cache_tab.strip(),
            "num_workers": int(num_workers),
            "target_duration_s": int(target_duration_s),
        },
    )

    # ── Connect + discover jobs ───────────────────────────────────────────────
    try:
        client = get_gspread_client()

        sheet_id = sheet_url.strip()
        if "spreadsheets/d/" in sheet_id:
            sheet_id = sheet_id.split("spreadsheets/d/")[1].split("/")[0]

        spreadsheet = client.open_by_key(sheet_id)
        jobs_ws = get_worksheet(spreadsheet, jobs_tab.strip())

        try:
            get_worksheet(spreadsheet, clips_tab.strip())
        except ValueError:
            st.warning(
                f"Clips tab '{clips_tab}' not found — clips will not be written to sheet."
            )

        # Pre-load video cache from sheet (shared read-only baseline for all workers).
        video_cache = get_session_cache()
        cache_ws_available = False
        try:
            cache_ws = get_worksheet(spreadsheet, cache_tab.strip())
            sheet_cache = load_sheet_cache(cache_ws)
            video_cache.update(sheet_cache)
            cache_ws_available = True
            st.info(f"Loaded {len(sheet_cache)} cached entries from VideoCache tab.")
        except ValueError:
            pass

        jobs = read_queued_jobs(jobs_ws)

    except Exception as e:
        logger.exception("Failed to connect to Google Sheet", extra={"sheet_url": sheet_url})
        st.error("Failed to connect to Google Sheet:")
        st.exception(e)
        traceback.print_exc()
        return

    if not jobs:
        st.info("No queued jobs found in the sheet.")
        return

    actual_workers = min(int(num_workers), len(jobs))
    st.info(
        f"Found {len(jobs)} queued job(s). "
        f"Starting {actual_workers} parallel worker(s)…"
    )

    # ── Bootstrap shared state ────────────────────────────────────────────────
    event_queue: queue.Queue[tuple[str, str, Any]] = queue.Queue()
    cache_lock = threading.Lock()

    job_states: dict[str, dict[str, Any]] = {
        f"row_{job.row_index}": {
            "job_key": f"row_{job.row_index}",
            "url": job.source_video_url,
            "status": "queued",
            "step": "",
            "error": "",
            "row_index": job.row_index,
        }
        for job in jobs
    }
    job_map: dict[str, SheetJob] = {f"row_{job.row_index}": job for job in jobs}

    # Capture config for worker closures.
    _sheet_id = sheet_id
    _jobs_tab = jobs_tab.strip()
    _clips_tab = clips_tab.strip()
    _cache_tab = cache_tab.strip() if cache_ws_available else ""
    _shared_cache = video_cache
    _cache_lock = cache_lock
    _target_duration_s = int(target_duration_s)

    def _run_worker(job: SheetJob) -> None:
        """Process one Clone Batch job with its own isolated gspread connections."""
        job_key = f"row_{job.row_index}"
        event_queue.put((job_key, EVT_JOB_START, None))

        try:
            # Each worker gets its own gspread client — auth tokens and HTTP
            # sessions are never shared across threads.
            worker_client = get_gspread_client()
            worker_spread = worker_client.open_by_key(_sheet_id)
            worker_jobs_ws = get_worksheet(worker_spread, _jobs_tab)

            worker_clips_ws = None
            try:
                worker_clips_ws = get_worksheet(worker_spread, _clips_tab)
            except ValueError:
                pass

            worker_cache_ws = None
            try:
                if _cache_tab:
                    worker_cache_ws = get_worksheet(worker_spread, _cache_tab)
            except ValueError:
                pass

            # Thread-local copy of the shared cache for safe concurrent reads.
            with _cache_lock:
                local_cache = dict(_shared_cache)

            def progress_cb(step: str, msg: str) -> None:
                event_queue.put((job_key, EVT_JOB_PROGRESS, {"step": step, "msg": msg}))

            try:
                update_job_status(worker_jobs_ws, job.row_index, "analyzing")
            except Exception:
                pass

            result: PipelineResult = run_pipeline(
                source_video_url=job.source_video_url,
                reference_image_url=job.reference_image_url,
                video_cache=local_cache,
                progress_cb=progress_cb,
                target_duration_s=_target_duration_s,
            )

            try:
                write_job_result(worker_jobs_ws, worker_clips_ws, job.row_index, result)
            except Exception as exc:
                logger.warning(
                    "Failed to write job result",
                    extra={"row_index": job.row_index, "error": str(exc)},
                )

            # Merge this worker's cache updates back into the shared cache and
            # persist to the VideoCache sheet so future runs can skip these steps.
            if result.context is not None:
                try:
                    video_key = result.context.video_key
                    entry = local_cache.get(video_key, {})
                    if isinstance(entry, dict):
                        entry.setdefault("source_video_url", job.source_video_url)
                        if result.analysis and result.analysis.face_visibility.best_window:
                            window = result.analysis.face_visibility.best_window
                            entry.setdefault("face_time_start", window.time_start_s)
                            entry.setdefault("face_time_end", window.time_end_s)
                        entry.setdefault("status", "done")
                        if result.script is not None:
                            entry["script"] = result.script
                        with _cache_lock:
                            _shared_cache[video_key] = entry
                        if worker_cache_ws is not None:
                            save_to_sheet_cache(worker_cache_ws, video_key, entry)
                except Exception:
                    pass

            batch_result = BatchWorkerResult(
                job_key=job_key,
                success=result.success,
                result=result,
                error=result.error if not result.success else None,
            )

        except Exception as exc:
            logger.exception(
                "Unhandled error in clone batch worker",
                extra={"row_index": job.row_index},
            )
            batch_result = BatchWorkerResult(
                job_key=job_key,
                success=False,
                error=f"Unhandled exception: {exc}",
            )

        event_queue.put((job_key, EVT_JOB_RESULT, batch_result))

    # ── Persist sheet context for the review / regen UI ───────────────────────
    st.session_state["batch_last_sheet_context"] = {
        "sheet_id": sheet_id,
        "jobs_tab": jobs_tab.strip(),
        "clips_tab": clips_tab.strip(),
        "cache_tab": cache_tab.strip(),
    }

    # ── Launch parallel runner ─────────────────────────────────────────────────
    runner = ParallelBatchRunner(
        jobs=jobs,
        worker_fn=_run_worker,
        num_workers=actual_workers,
        event_queue=event_queue,
    )

    st.session_state[_KEY_QUEUE] = event_queue
    st.session_state[_KEY_JOB_STATES] = job_states
    st.session_state[_KEY_JOB_MAP] = job_map
    st.session_state[_KEY_LOG] = []
    st.session_state[_KEY_RESULTS] = []
    st.session_state[_KEY_RUNNING] = True

    runner.start()
    st.rerun()


# ── Active-run renderer ───────────────────────────────────────────────────────


def _render_active_run() -> None:
    """Drain the event queue, update state, re-render, and auto-refresh."""
    event_queue: queue.Queue = st.session_state[_KEY_QUEUE]
    job_states: dict[str, dict] = st.session_state[_KEY_JOB_STATES]
    log: list[str] = st.session_state[_KEY_LOG]
    results: list[BatchWorkerResult] = st.session_state.get(_KEY_RESULTS, [])

    all_done = False
    while True:
        try:
            job_key, event_type, data = event_queue.get_nowait()
        except queue.Empty:
            break

        if event_type == EVT_ALL_DONE:
            all_done = True
            break

        ts = time.strftime("%H:%M:%S")
        state = job_states.get(job_key, {})

        if event_type == EVT_JOB_START:
            state["status"] = "analyzing"
            log.append(f"[{ts}] {job_key} — started")

        elif event_type == EVT_JOB_PROGRESS:
            if isinstance(data, dict):
                step = data.get("step", "")
                state["step"] = step
                state["status"] = step.lower().replace(" ", "_")
                log.append(f"[{ts}] {job_key} — {step}")

        elif event_type == EVT_JOB_RESULT:
            batch_result: BatchWorkerResult = data
            if batch_result.success:
                state["status"] = "complete"
            else:
                state["status"] = "error"
                state["error"] = batch_result.error or "Unknown error"
            results.append(batch_result)
            indicator = "✓" if batch_result.success else "✗"
            log.append(
                f"[{ts}] {job_key} — {indicator} "
                f"{'done' if batch_result.success else batch_result.error or ''}"
            )

    st.session_state[_KEY_RESULTS] = results

    # Status table
    st.subheader("Batch Processing Status")
    table_rows = [
        {
            "Job": s["job_key"],
            "Video": (s["url"][-60:] if len(s["url"]) > 60 else s["url"]),
            "Status": s["status"],
            "Step": s.get("step", ""),
            "Error": (s["error"][:80] if s.get("error") else ""),
        }
        for s in job_states.values()
    ]
    st.dataframe(table_rows, width="stretch")

    if log:
        with st.expander("Event log", expanded=False):
            st.text("\n".join(log[-50:]))

    if all_done:
        st.session_state[_KEY_RUNNING] = False
        _finalize_run(results)
    else:
        active = sum(
            1
            for s in job_states.values()
            if s["status"] not in ("complete", "error", "queued")
        )
        st.info(
            f"Processing — {active} job(s) active in parallel. "
            "Page refreshes every 5 seconds."
        )
        time.sleep(5)
        st.rerun()


def _finalize_run(results: list[BatchWorkerResult]) -> None:
    """Build the completed-jobs summary and render the review section."""
    job_map: dict[str, SheetJob] = st.session_state.get(_KEY_JOB_MAP, {})

    completed_jobs: list[dict[str, Any]] = []
    for br in results:
        job = job_map.get(br.job_key)
        pipeline_result: PipelineResult | None = br.result
        completed_jobs.append(
            {
                "row_index": job.row_index if job else 0,
                "source_video_url": job.source_video_url if job else "",
                "reference_image_url": job.reference_image_url if job else "",
                "swapped_image_url": (
                    pipeline_result.swapped_image_url if pipeline_result else None
                ),
                "success": br.success,
                "error": br.error,
            }
        )
    st.session_state["batch_last_completed_jobs"] = completed_jobs

    done_count = sum(1 for br in results if br.success)
    fail_count = len(results) - done_count

    if fail_count == 0:
        st.success(f"All {done_count} job(s) completed successfully!")
    else:
        st.warning(f"{done_count} completed, {fail_count} failed.")

    if any(not br.success for br in results):
        st.subheader("Failed jobs")
        for br in results:
            if not br.success:
                st.error(f"**{br.job_key}**: {br.error}")

    _render_batch_review_section()


# ── Post-run review ───────────────────────────────────────────────────────────


def _render_batch_review_section() -> None:
    completed_jobs: list[dict[str, Any]] | None = st.session_state.get(
        "batch_last_completed_jobs"
    )
    sheet_ctx: dict[str, str] | None = st.session_state.get("batch_last_sheet_context")

    if not completed_jobs or not sheet_ctx:
        return

    st.subheader("Review generated images (last batch)")

    options = [
        f"Row {job['row_index']} — {'OK' if job['success'] else 'ERROR'}"
        for job in completed_jobs
    ]
    selected_label = st.selectbox(
        "Select a job to review",
        options,
        key="batch_review_selected_job",
    )

    selected_idx = options.index(selected_label)
    job = completed_jobs[selected_idx]

    if job.get("swapped_image_url"):
        st.image(
            job["swapped_image_url"],
            caption=f"Row {job['row_index']} swapped image",
            width="stretch",
        )
    else:
        st.info("This job did not produce a swapped image.")

    regen_col1, regen_col2 = st.columns(2)
    with regen_col1:
        st.caption("If you like this result, no action is needed.")
    with regen_col2:
        regen_in_progress: bool = bool(
            st.session_state.get("batch_regen_in_progress", False)
        )
        clicked = st.button(
            "Regenerate image only for this row",
            key="batch_regen_image_only",
            disabled=regen_in_progress,
        )
        if regen_in_progress:
            st.caption("Image regeneration in progress...")
        if clicked:
            st.session_state["batch_regen_target_row"] = job["row_index"]
            st.info("Regenerating image with the same settings...")
            st.rerun()


def _regenerate_single_batch_job_image(
    job: dict[str, Any], sheet_ctx: dict[str, str]
) -> None:
    """Regenerate only the swapped image for a completed batch job row."""
    try:
        client = get_gspread_client()
        spreadsheet = client.open_by_key(sheet_ctx["sheet_id"])
        jobs_ws = get_worksheet(spreadsheet, sheet_ctx["jobs_tab"])

        try:
            clips_ws = get_worksheet(spreadsheet, sheet_ctx["clips_tab"])
        except ValueError:
            clips_ws = None

        video_cache = get_session_cache()
        cache_ws = None
        try:
            cache_ws = get_worksheet(spreadsheet, sheet_ctx["cache_tab"])
            sheet_cache = load_sheet_cache(cache_ws)
            video_cache.update(sheet_cache)
        except ValueError:
            cache_ws = None

        row_index = job["row_index"]

        result = run_pipeline(
            source_video_url=job["source_video_url"],
            reference_image_url=job["reference_image_url"],
            video_cache=video_cache,
            progress_cb=None,
            regenerate_image_only=True,
        )

        if not result.success:
            st.error(f"Failed to regenerate image for row {row_index}: {result.error}")
            return

        update_job_status(
            jobs_ws,
            row_index,
            "generating",
            extra_fields={"swapped_start_frame_url": result.swapped_image_url or ""},
        )

        try:
            if cache_ws is not None and result.context is not None:
                video_key = result.context.video_key
                entry = video_cache.get(video_key, {})
                if isinstance(entry, dict) and entry:
                    save_to_sheet_cache(cache_ws, video_key, entry)
        except Exception:
            pass

        job["swapped_image_url"] = result.swapped_image_url
        job["success"] = result.success
        job["error"] = result.error

        st.success(f"Regenerated image for row {row_index}. The sheet has been updated.")
    except Exception as e:
        st.error(f"Failed to regenerate image for row {job.get('row_index')}: {e}")
