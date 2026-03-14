from __future__ import annotations

import queue
import time
import traceback
from typing import Any

import streamlit as st

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
from pipeline.persona_runner import PersonaPipelineResult, run_persona_pipeline
from sheets.auth import get_gspread_client
from sheets.persona_reader import PersonaSheetJob, read_queued_persona_jobs
from sheets.persona_writer import write_persona_result
from sheets.reader import get_worksheet
from sheets.writer import update_job_status


logger = get_logger(__name__, ui_mode="persona_batch")

# ── Session-state keys ────────────────────────────────────────────────────────
_KEY_RUNNING = "persona_running"
_KEY_QUEUE = "persona_event_queue"
_KEY_JOB_STATES = "persona_job_states"
_KEY_JOB_MAP = "persona_job_map"        # job_key → PersonaSheetJob
_KEY_LOG = "persona_log"
_KEY_RESULTS = "persona_results"        # list[BatchWorkerResult]


def render() -> None:
    setup_logging()
    st.header("Persona Batch")
    st.caption(
        "Generate scripts and clips for each persona row in a Google Sheet — "
        "no video analysis or face swap needed."
    )

    with st.expander("ℹ️ How this works & what to prepare", expanded=False):
        st.markdown(
            """
**What this mode does**

A lighter-weight pipeline designed for **persona-based ads** where you already have a persona image
and don't need to analyze or face-swap a source video.

For each `queued` row in the **Personas** tab it:
1. Uses the `persona_image` directly as the Kling seed (no video analysis, no face swap).
2. Rewrites a script tailored to the persona's pain point and solution using Gemini.
3. Splits the script into timed clip segments and writes them to the **Clips** tab.

**Typical workflow:**
1. Fill in your **Personas** tab with the required columns (see below).
2. Set each row's `status` to `queued`.
3. Paste the sheet URL, confirm tab names, and click **Load & Process Queued Personas**.
4. When this run finishes, rows move to `generating` status — switch to the **Producer** tab to generate the final videos.

---

**Required columns in the Personas tab**

| Column | Description |
|---|---|
| `job_key` | A unique identifier you assign to this persona job. |
| `persona` | Description of the target persona (e.g. "Busy professional, 30s, reads on commute"). |
| `pain_point` | The problem this persona faces that Speechify solves. |
| `speechify_solution` | How Speechify specifically addresses the pain point. |
| `hook_sample` | An example hook line or tone to guide the script. |
| `hook_emotion` | The emotional tone to aim for (e.g. "relief", "excitement"). |
| `persona_image` | Publicly accessible URL of the persona's face/headshot image. |
| `status` | Set to `queued` for rows to process. Other values are skipped. |

---

**What gets written back to the sheet**

| Column | Description |
|---|---|
| `status` | Updated to `analyzing` → `generating` (or `error`). |
| `new_frame_gender` | Detected speaker gender from the persona image (`male` / `female`). |
| `final_script_json` | Full rewritten script as JSON. |
| `prompt_template` | The prompt template used for this persona's script. |
| `total_clips` | Number of clip segments generated. |
| `Date generated` | Timestamp of when this row was processed. |

Clips are also written to the **Clips** tab (one row per clip segment).
            """
        )

    # ── Active run ────────────────────────────────────────────────────────────
    if st.session_state.get(_KEY_RUNNING):
        _render_active_run()
        return

    # ── Config form ───────────────────────────────────────────────────────────
    with st.form("persona_batch_form"):
        sheet_url = st.text_input(
            "Google Sheet URL or ID",
            placeholder="https://docs.google.com/spreadsheets/d/...",
        )
        personas_tab = st.text_input("Personas Tab Name", value="Personas")
        clips_tab = st.text_input("Clips Tab Name", value="Clips")
        num_workers = st.number_input(
            "Parallel workers",
            min_value=1,
            max_value=MAX_BATCH_WORKERS,
            value=DEFAULT_BATCH_WORKERS,
            step=1,
            help=(
                f"Number of personas to process simultaneously (1–{MAX_BATCH_WORKERS}). "
                "Each worker opens its own API connections. "
                "Increase for large batches; keep at 1 to process sequentially."
            ),
        )
        submitted = st.form_submit_button("Load & Process Queued Personas", type="primary")

    if not submitted:
        _render_persona_review_section()
        return

    if not sheet_url.strip():
        st.error("Please enter a Google Sheet URL or ID.")
        return

    logger.info(
        "Persona batch submission received",
        extra={
            "sheet_url_or_id": sheet_url.strip(),
            "personas_tab": personas_tab.strip(),
            "clips_tab": clips_tab.strip(),
            "num_workers": int(num_workers),
        },
    )

    # ── Connect + discover jobs ───────────────────────────────────────────────
    try:
        client = get_gspread_client()

        sheet_id = sheet_url.strip()
        if "spreadsheets/d/" in sheet_id:
            sheet_id = sheet_id.split("spreadsheets/d/")[1].split("/")[0]

        spreadsheet = client.open_by_key(sheet_id)
        personas_ws = get_worksheet(spreadsheet, personas_tab.strip())

        try:
            get_worksheet(spreadsheet, clips_tab.strip())
        except ValueError:
            st.warning(
                f"Clips tab '{clips_tab}' not found — clips will not be written to sheet."
            )

        jobs = read_queued_persona_jobs(personas_ws)

    except Exception as e:
        logger.exception(
            "Failed to connect to Google Sheet",
            extra={"sheet_id": sheet_url.strip()},
        )
        st.error("Failed to connect to Google Sheet:")
        st.exception(e)
        traceback.print_exc()
        return

    if not jobs:
        st.info("No queued persona rows found in the sheet.")
        return

    actual_workers = min(int(num_workers), len(jobs))
    st.info(
        f"Found {len(jobs)} queued persona(s). "
        f"Starting {actual_workers} parallel worker(s)…"
    )

    # ── Bootstrap shared state ────────────────────────────────────────────────
    event_queue: queue.Queue[tuple[str, str, Any]] = queue.Queue()

    job_states: dict[str, dict[str, Any]] = {
        job.job_input.job_key: {
            "job_key": job.job_input.job_key,
            "persona": job.job_input.persona.split("\n")[0][:50],
            "status": "queued",
            "step": "",
            "error": "",
            "row_index": job.row_index,
        }
        for job in jobs
    }
    job_map: dict[str, PersonaSheetJob] = {
        job.job_input.job_key: job for job in jobs
    }

    # Capture config for worker closures.
    _sheet_id = sheet_id
    _personas_tab = personas_tab.strip()
    _clips_tab = clips_tab.strip()

    def _run_worker(job: PersonaSheetJob) -> None:
        """Process one Persona Batch job with its own isolated gspread connections."""
        job_key = job.job_input.job_key
        event_queue.put((job_key, EVT_JOB_START, None))

        try:
            # Each worker gets its own gspread client — auth tokens and HTTP
            # sessions are never shared across threads.
            worker_client = get_gspread_client()
            worker_spread = worker_client.open_by_key(_sheet_id)
            worker_personas_ws = get_worksheet(worker_spread, _personas_tab)

            worker_clips_ws = None
            try:
                worker_clips_ws = get_worksheet(worker_spread, _clips_tab)
            except ValueError:
                pass

            def progress_cb(step: str, msg: str) -> None:
                event_queue.put((job_key, EVT_JOB_PROGRESS, {"step": step, "msg": msg}))

            try:
                update_job_status(worker_personas_ws, job.row_index, "analyzing")
            except Exception:
                pass

            result: PersonaPipelineResult = run_persona_pipeline(
                job=job.job_input,
                progress_cb=progress_cb,
            )

            try:
                write_persona_result(
                    worker_personas_ws, worker_clips_ws, job.row_index, result
                )
            except Exception as exc:
                logger.warning(
                    "Failed to write persona result",
                    extra={"row_index": job.row_index, "error": str(exc)},
                )

            batch_result = BatchWorkerResult(
                job_key=job_key,
                success=result.success,
                result=result,
                error=result.error if not result.success else None,
            )

        except Exception as exc:
            logger.exception(
                "Unhandled error in persona batch worker",
                extra={"row_index": job.row_index, "job_key": job.job_input.job_key},
            )
            batch_result = BatchWorkerResult(
                job_key=job_key,
                success=False,
                error=f"Unhandled exception: {exc}",
            )

        event_queue.put((job_key, EVT_JOB_RESULT, batch_result))

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
    st.subheader("Persona Processing Status")
    table_rows = [
        {
            "Job Key": s["job_key"],
            "Persona": s["persona"],
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
            f"Processing — {active} persona(s) active in parallel. "
            "Page refreshes every 5 seconds."
        )
        time.sleep(5)
        st.rerun()


def _finalize_run(results: list[BatchWorkerResult]) -> None:
    """Build the completed-jobs summary and render the review section."""
    job_map: dict[str, PersonaSheetJob] = st.session_state.get(_KEY_JOB_MAP, {})

    completed_jobs: list[dict[str, Any]] = []
    for br in results:
        job = job_map.get(br.job_key)
        pipeline_result: PersonaPipelineResult | None = br.result
        completed_jobs.append(
            {
                "row_index": job.row_index if job else 0,
                "job_key": br.job_key,
                "persona": (
                    job.job_input.persona.split("\n")[0][:50] if job else br.job_key
                ),
                "persona_image_url": (
                    job.job_input.persona_image_url if job else ""
                ),
                "gender": pipeline_result.gender if pipeline_result else None,
                "clips_count": (
                    len(pipeline_result.clips) if pipeline_result else 0
                ),
                "success": br.success,
                "error": br.error,
            }
        )
    st.session_state["persona_batch_last_completed"] = completed_jobs

    done_count = sum(1 for br in results if br.success)
    fail_count = len(results) - done_count

    if fail_count == 0:
        st.success(f"All {done_count} persona(s) processed successfully!")
    else:
        st.warning(f"{done_count} completed, {fail_count} failed.")

    if any(not br.success for br in results):
        st.subheader("Failed personas")
        for br in results:
            if not br.success:
                st.error(f"**{br.job_key}**: {br.error}")

    _render_persona_review_section()


# ── Post-run review ───────────────────────────────────────────────────────────


def _render_persona_review_section() -> None:
    completed_jobs: list[dict[str, Any]] | None = st.session_state.get(
        "persona_batch_last_completed"
    )
    if not completed_jobs:
        return

    st.subheader("Review generated personas (last batch)")

    options = [
        f"Row {job['row_index']} — {job['persona']} — {'OK' if job['success'] else 'ERROR'}"
        for job in completed_jobs
    ]
    selected_label = st.selectbox(
        "Select a persona to review",
        options,
        key="persona_review_selected",
    )

    selected_idx = options.index(selected_label)
    job = completed_jobs[selected_idx]

    col1, col2 = st.columns(2)
    with col1:
        if job.get("persona_image_url"):
            st.image(
                job["persona_image_url"],
                caption=f"Persona image — {job['persona']}",
                width="stretch",
            )
        else:
            st.info("No persona image available.")
    with col2:
        st.metric("Detected gender", job.get("gender") or "—")
        st.metric("Clips generated", job.get("clips_count") or 0)
        if not job["success"]:
            st.error(f"Error: {job.get('error')}")
