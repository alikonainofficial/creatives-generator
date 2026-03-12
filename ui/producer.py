from __future__ import annotations

import queue
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import streamlit as st

from logging_config import get_logger, setup_logging
from pipeline.producer_runner import (
    EVT_BLUEPRINT,
    EVT_BLUEPRINT_READY,
    EVT_CLIP_DONE,
    EVT_CLIP_ERROR,
    EVT_CLIP_POLL,
    EVT_CLIP_START,
    EVT_CLIPS_LOADED,
    EVT_DEMO_INJECT,
    EVT_DEMO_SKIP,
    EVT_DRIVE_ERROR,
    EVT_DRIVE_UPLOAD,
    EVT_JOB_DONE,
    EVT_JOB_ERROR,
    EVT_STITCH_DONE,
    EVT_STITCH_POLL,
    EVT_STITCH_START,
    ProducerJobResult,
    run_producer_job,
)
from sheets.auth import get_gspread_client
from sheets.producer_reader import GeneratingJob, read_generating_jobs
from sheets.reader import get_worksheet


logger = get_logger(__name__, ui_mode="producer")

# ── Session-state keys ────────────────────────────────────────────────────────
_KEY_RUNNING = "producer_running"
_KEY_THREAD = "producer_thread"
_KEY_QUEUE = "producer_event_queue"
_KEY_JOB_STATES = "producer_job_states"
_KEY_RESULTS = "producer_results"
_KEY_LOG = "producer_log"

_DEFAULT_WORKERS = 5
_MAX_WORKERS = 10


def render() -> None:
    setup_logging()
    st.header("Producer")
    st.caption(
        "Generate Kling video clips, stitch them, and upload to Google Drive "
        "for all jobs currently in 'generating' status."
    )

    with st.expander("ℹ️ How this works & what to prepare", expanded=False):
        st.markdown(
            """
**What this mode does**

This is **Phase 2** of the pipeline — it picks up where **Batch Job** or **Persona Batch** left off.

For every job with `status = generating` in your sheet it:
1. Validates (or generates) the video blueprint — a frame-chained plan for all clips.
2. Loads clip rows from the **Clips** tab.
3. Submits each clip to **Kling AI** sequentially, polling until complete.
4. Optionally injects an **app demo** clip at a configured position.
5. Stitches all clips into a single video via **FAL FFmpeg**.
6. Uploads the final video to **Google Drive** and writes the Drive URL back to the sheet.
7. Marks the job `done`.

> **Run Clone Batch or Persona Batch first.** Producer only works on rows that those modes have
> already processed (they populate `status = generating` and write the Clips tab).

---

**Inputs required**

| Field | Description |
|---|---|
| **Google Sheet URL or ID** | Same sheet used in the earlier Batch Job / Persona Batch run. |
| **Jobs Tab Name** | Tab containing jobs at `status = generating`. Use `Jobs` for Batch Job output, or `Personas` for Persona Batch output. |
| **Clips Tab Name** | Tab containing clip rows written by the earlier pipeline run (default: `Clips`). |
| **App Demo Tab Name** | *(Optional)* Tab with app demo videos to inject. Leave blank to skip. Required columns: `demo_id`, `demo_type`, `video_url`. |
| **Google Drive Folder ID** | The folder ID from the Google Drive URL where final videos will be uploaded. |
| **Poll interval (seconds)** | How often to check Kling / FFmpeg job status. 30s is a safe default; lower values increase API usage. |

---

**What gets written back to the sheet**

| Column | Description |
|---|---|
| `status` | Updated to `done` (or `error`). |
| `fal_video_url` | URL of the stitched video hosted on Fal. |
| `drive_video_url` | Google Drive URL of the final uploaded video. |
| `clip_video_url` *(Clips tab)* | Individual Kling clip URL per clip row. |
| `kling_task_id` *(Clips tab)* | Kling task ID for debugging. |

---

**Tips**

- Clips that already have `status = done` and a `clip_video_url` are **skipped** — safe to re-run after a partial failure.
- Clips stuck in `generating` (from a crashed run) are automatically **reset to queued** on the next run.
- Use the **Event log** expander during an active run to see step-by-step progress.
            """
        )

    # ── Active run ────────────────────────────────────────────────────────────
    if st.session_state.get(_KEY_RUNNING):
        _render_active_run()
        return

    # ── Config form ───────────────────────────────────────────────────────────
    with st.form("producer_form"):
        sheet_url = st.text_input(
            "Google Sheet URL or ID",
            placeholder="https://docs.google.com/spreadsheets/d/...",
        )
        col1, col2 = st.columns(2)
        with col1:
            jobs_tab = st.text_input(
                "Jobs Tab Name",
                value="Jobs",
                help="Use 'Personas' for persona batch mode.",
            )
            clips_tab = st.text_input("Clips Tab Name", value="Clips")
            demo_tab = st.text_input(
                "App Demo Tab Name (optional)",
                value="",
                help=(
                    "Name of the sheet tab containing app demo videos "
                    "(columns: demo_id, demo_type, video_url). "
                    "Leave blank to skip demo injection."
                ),
            )
        with col2:
            drive_folder_id = st.text_input(
                "Google Drive Folder ID",
                placeholder="1u7p69c7JVeHUUaq8GVA0UUccFr3rgmnR",
                help="The folder ID from the Drive URL. Final videos are uploaded here.",
            )
            poll_interval = st.number_input(
                "Poll interval (seconds)",
                min_value=10,
                max_value=120,
                value=30,
                step=5,
                help="How often to check Kling/FFmpeg job status.",
            )
            num_workers = st.number_input(
                "Parallel workers",
                min_value=1,
                max_value=_MAX_WORKERS,
                value=_DEFAULT_WORKERS,
                step=1,
                help=(
                    f"Number of jobs to process simultaneously (1–{_MAX_WORKERS}). "
                    "Each worker handles one job end-to-end with its own API connections. "
                    "Clips within a single job are still generated sequentially."
                ),
            )

        submitted = st.form_submit_button("▶ Start Production", type="primary")

    if not submitted:
        _render_previous_results()
        return

    if not sheet_url.strip():
        st.error("Please enter a Google Sheet URL or ID.")
        return
    if not drive_folder_id.strip():
        st.error("Please enter a Google Drive Folder ID.")
        return

    # ── Connect + discover jobs ───────────────────────────────────────────────
    try:
        client = get_gspread_client()
        sheet_id = sheet_url.strip()
        if "spreadsheets/d/" in sheet_id:
            sheet_id = sheet_id.split("spreadsheets/d/")[1].split("/")[0]

        spreadsheet = client.open_by_key(sheet_id)
        jobs_ws = get_worksheet(spreadsheet, jobs_tab.strip())
        clips_ws = get_worksheet(spreadsheet, clips_tab.strip())

        demos_ws = None
        if demo_tab.strip():
            demos_ws = get_worksheet(spreadsheet, demo_tab.strip())

        jobs = read_generating_jobs(jobs_ws)
    except Exception as exc:
        st.error("Failed to connect to Google Sheet:")
        st.exception(exc)
        traceback.print_exc()
        return

    if not jobs:
        st.info(
            f"No jobs with status='generating' found in the '{jobs_tab}' tab. "
            "Run the Clone Batch or Persona Batch pipeline first."
        )
        return

    actual_workers = min(int(num_workers), len(jobs))
    st.info(
        f"Found {len(jobs)} job(s) to produce. "
        f"Starting {actual_workers} parallel worker(s)…"
    )

    # ── Bootstrap shared state ────────────────────────────────────────────────
    event_queue: queue.Queue[tuple[str, str, Any]] = queue.Queue()
    job_states: dict[str, dict[str, Any]] = {
        job.job_key: {
            "job_key": job.job_key,
            "row": job.row_index,
            "status": "queued",
            "current_step": "—",
            "clips_done": 0,
            "clips_total": 0,
            "fal_video_url": "",
            "drive_video_url": "",
            "error": "",
        }
        for job in jobs
    }

    st.session_state[_KEY_QUEUE] = event_queue
    st.session_state[_KEY_JOB_STATES] = job_states
    st.session_state[_KEY_LOG] = []
    st.session_state[_KEY_RESULTS] = []
    st.session_state[_KEY_RUNNING] = True

    # Capture loop variables for the worker closure.
    _sheet_id = sheet_id
    _jobs_tab = jobs_tab.strip()
    _clips_tab = clips_tab.strip()
    _demo_tab = demo_tab.strip()
    _drive_folder_id = drive_folder_id.strip()
    _poll_interval = int(poll_interval)

    def _run_worker(job: GeneratingJob) -> ProducerJobResult:
        """Process one job end-to-end with its own isolated gspread connections.

        Each worker opens a fresh gspread client so auth token refresh and the
        underlying HTTP session are never shared across threads.
        """
        worker_client = get_gspread_client()
        worker_spread = worker_client.open_by_key(_sheet_id)
        worker_jobs_ws = get_worksheet(worker_spread, _jobs_tab)
        worker_clips_ws = get_worksheet(worker_spread, _clips_tab)
        worker_demos_ws = get_worksheet(worker_spread, _demo_tab) if _demo_tab else None

        event_queue.put((job.job_key, "job_start", None))
        try:
            result = run_producer_job(
                job=job,
                jobs_worksheet=worker_jobs_ws,
                clips_worksheet=worker_clips_ws,
                drive_folder_id=_drive_folder_id,
                demos_worksheet=worker_demos_ws,
                progress_cb=lambda jk, et, d: event_queue.put((jk, et, d)),
                poll_interval=_poll_interval,
                max_poll_attempts=40,
            )
        except Exception as exc:
            result = ProducerJobResult(
                job_key=job.job_key,
                success=False,
                error=f"Unhandled exception: {exc}",
            )
            logger.exception("Unhandled error in producer job", extra={"job_key": job.job_key})

        event_queue.put((job.job_key, "_result", result))
        return result

    def _producer_thread() -> None:
        with ThreadPoolExecutor(max_workers=actual_workers) as executor:
            futures = {executor.submit(_run_worker, job): job for job in jobs}
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception:
                    # Exceptions are already caught and enqueued inside _run_worker;
                    # this guard is a last-resort safety net.
                    pass
        event_queue.put(("__all__", "_done", None))

    thread = threading.Thread(target=_producer_thread, daemon=True)
    st.session_state[_KEY_THREAD] = thread
    thread.start()

    st.rerun()


# ── Active-run renderer ───────────────────────────────────────────────────────


def _render_active_run() -> None:
    """Drain the event queue, update state, re-render, and auto-refresh."""
    event_queue: queue.Queue = st.session_state[_KEY_QUEUE]
    job_states: dict[str, dict] = st.session_state[_KEY_JOB_STATES]
    log: list[str] = st.session_state[_KEY_LOG]
    results: list[ProducerJobResult] = st.session_state[_KEY_RESULTS]

    # Drain all pending events
    all_done = False
    while True:
        try:
            job_key, event_type, data = event_queue.get_nowait()
        except queue.Empty:
            break

        if event_type == "_done":
            all_done = True
            break

        if event_type == "_result":
            results.append(data)
            if data.success:
                job_states[job_key]["status"] = "done"
            else:
                job_states[job_key]["status"] = "error"
                job_states[job_key]["error"] = data.error
            continue

        _apply_event(job_states, log, job_key, event_type, data)

    # Render status table
    st.subheader("Production Status")
    table_rows = [
        {
            "Job Key": s["job_key"],
            "Status": s["status"],
            "Step": s["current_step"],
            "Clips": f"{s['clips_done']}/{s['clips_total']}" if s["clips_total"] else "—",
            "Error": s["error"][:80] if s["error"] else "",
        }
        for s in job_states.values()
    ]
    st.dataframe(table_rows, use_container_width=True)

    # Recent log
    if log:
        with st.expander("Event log", expanded=False):
            st.text("\n".join(log[-50:]))

    if all_done:
        st.session_state[_KEY_RUNNING] = False
        _render_completion_summary(results)
    else:
        running_count = sum(1 for s in job_states.values() if s["status"] == "running")
        st.info(
            f"Production running — {running_count} job(s) active in parallel. "
            "Page refreshes every 5 seconds."
        )
        time.sleep(5)
        st.rerun()


def _apply_event(
    job_states: dict[str, dict],
    log: list[str],
    job_key: str,
    event_type: str,
    data: Any,
) -> None:
    """Update job_states in place and append to log based on the event."""
    state = job_states.get(job_key, {})
    ts = time.strftime("%H:%M:%S")

    if event_type == "job_start":
        state["status"] = "running"
        log.append(f"[{ts}] {job_key} — starting")

    elif event_type == EVT_BLUEPRINT:
        state["current_step"] = "Blueprint"
        log.append(f"[{ts}] {job_key} — {data}")

    elif event_type == EVT_BLUEPRINT_READY:
        state["current_step"] = f"Blueprint ready ({data})"
        log.append(f"[{ts}] {job_key} — blueprint: {data}")

    elif event_type == EVT_CLIPS_LOADED:
        total = data.get("total", 0) if isinstance(data, dict) else 0
        state["clips_total"] = total
        state["current_step"] = f"0/{total} clips"
        log.append(f"[{ts}] {job_key} — {total} clip(s) loaded")

    elif event_type == EVT_CLIP_START:
        idx = data.get("clip_index", "?") if isinstance(data, dict) else "?"
        state["current_step"] = f"Generating clip {idx}"
        log.append(f"[{ts}] {job_key} — clip {idx} started")

    elif event_type == EVT_CLIP_POLL:
        if isinstance(data, dict):
            state["current_step"] = (
                f"Clip {data.get('clip_index')} — poll {data.get('attempt')} "
                f"({data.get('status')})"
            )

    elif event_type == EVT_CLIP_DONE:
        state["clips_done"] = state.get("clips_done", 0) + 1
        idx = data.get("clip_index", "?") if isinstance(data, dict) else "?"
        total = state.get("clips_total", "?")
        state["current_step"] = f"{state['clips_done']}/{total} clips done"
        log.append(f"[{ts}] {job_key} — clip {idx} done")

    elif event_type == EVT_CLIP_ERROR:
        log.append(f"[{ts}] {job_key} — clip error: {data}")

    elif event_type == EVT_DEMO_INJECT:
        demo_id = data.get("demo_id", "?") if isinstance(data, dict) else "?"
        total = data.get("total_clips", "?") if isinstance(data, dict) else "?"
        state["current_step"] = f"Demo injected ({demo_id})"
        log.append(f"[{ts}] {job_key} — demo injected: {demo_id} ({total} clips total)")

    elif event_type == EVT_DEMO_SKIP:
        log.append(f"[{ts}] {job_key} — demo skipped: {data}")

    elif event_type == EVT_STITCH_START:
        state["current_step"] = "Stitching clips"
        log.append(f"[{ts}] {job_key} — stitching {data.get('clip_count', '?')} clips")

    elif event_type == EVT_STITCH_POLL:
        if isinstance(data, dict):
            state["current_step"] = f"Stitch poll {data.get('attempt')} ({data.get('status')})"

    elif event_type == EVT_STITCH_DONE:
        state["current_step"] = "Stitch complete"
        log.append(f"[{ts}] {job_key} — stitch done: {data}")

    elif event_type == EVT_DRIVE_UPLOAD:
        state["current_step"] = "Uploading to Drive"
        log.append(f"[{ts}] {job_key} — {data}")

    elif event_type == EVT_DRIVE_ERROR:
        log.append(f"[{ts}] {job_key} — ⚠ {data}")

    elif event_type == EVT_JOB_DONE:
        state["status"] = "done"
        if isinstance(data, dict):
            state["fal_video_url"] = data.get("fal_video_url", "")
            state["drive_video_url"] = data.get("drive_video_url", "")
        log.append(f"[{ts}] {job_key} — ✓ done")

    elif event_type == EVT_JOB_ERROR:
        state["status"] = "error"
        state["error"] = str(data)
        log.append(f"[{ts}] {job_key} — ✗ {data}")


# ── Post-run summary ──────────────────────────────────────────────────────────


def _render_completion_summary(results: list[ProducerJobResult]) -> None:
    done = [r for r in results if r.success]
    failed = [r for r in results if not r.success]

    if not failed:
        st.success(f"All {len(done)} job(s) produced successfully!")
    else:
        st.warning(f"{len(done)} succeeded, {len(failed)} failed.")

    if done:
        st.subheader("Completed videos")
        for r in done:
            with st.expander(r.job_key):
                if r.drive_video_url:
                    st.markdown(f"**Google Drive:** [{r.job_key}.mp4]({r.drive_video_url})")
                if r.fal_video_url:
                    st.markdown(f"**Fal URL:** {r.fal_video_url}")

    if failed:
        st.subheader("Failed jobs")
        for r in failed:
            st.error(f"**{r.job_key}**: {r.error}")

    st.session_state[_KEY_RESULTS] = results


def _render_previous_results() -> None:
    results: list[ProducerJobResult] | None = st.session_state.get(_KEY_RESULTS)
    if not results:
        return
    st.divider()
    st.subheader("Previous run results")
    _render_completion_summary(results)
