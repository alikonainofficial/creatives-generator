from __future__ import annotations

import traceback
from typing import Any

import streamlit as st

from logging_config import get_logger, setup_logging
from pipeline.persona_runner import run_persona_pipeline
from sheets.auth import get_gspread_client
from sheets.persona_reader import read_queued_persona_jobs
from sheets.persona_writer import write_persona_result
from sheets.reader import get_worksheet
from sheets.writer import update_job_status


logger = get_logger(__name__, ui_mode="persona_batch")


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

    with st.form("persona_batch_form"):
        sheet_url = st.text_input(
            "Google Sheet URL or ID",
            placeholder="https://docs.google.com/spreadsheets/d/...",
        )
        personas_tab = st.text_input("Personas Tab Name", value="Personas")
        clips_tab = st.text_input("Clips Tab Name", value="Clips")
        submitted = st.form_submit_button("Load & Process Queued Personas", type="primary")

    if submitted:
        if not sheet_url.strip():
            st.error("Please enter a Google Sheet URL or ID.")
            return

        logger.info(
            "Persona batch submission received",
            extra={
                "sheet_url_or_id": sheet_url.strip(),
                "personas_tab": personas_tab.strip(),
                "clips_tab": clips_tab.strip(),
            },
        )

        try:
            client = get_gspread_client()

            sheet_id = sheet_url.strip()
            if "spreadsheets/d/" in sheet_id:
                sheet_id = sheet_id.split("spreadsheets/d/")[1].split("/")[0]

            spreadsheet = client.open_by_key(sheet_id)
            personas_ws = get_worksheet(spreadsheet, personas_tab.strip())

            try:
                clips_ws = get_worksheet(spreadsheet, clips_tab.strip())
            except ValueError:
                clips_ws = None
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

        st.info(f"Found {len(jobs)} queued persona(s). Starting processing...")

        job_statuses: dict[int, dict[str, str]] = {
            job.row_index: {
                "job_key": job.job_input.job_key,
                "persona": job.job_input.persona.split("\n")[0][:50],
                "status": "pending",
                "error": "",
            }
            for job in jobs
        }

        status_placeholder = st.empty()

        def render_status_table() -> None:
            rows = [
                {
                    "Row": row_idx,
                    "Job Key": info["job_key"],
                    "Persona": info["persona"],
                    "Status": info["status"],
                    "Error": info["error"],
                }
                for row_idx, info in job_statuses.items()
            ]
            status_placeholder.dataframe(rows, width="stretch")

        render_status_table()

        completed_jobs: list[dict[str, Any]] = []

        for job in jobs:
            logger.info(
                "Starting persona batch row",
                extra={
                    "row_index": job.row_index,
                    "job_key": job.job_input.job_key,
                },
            )
            job_statuses[job.row_index]["status"] = "analyzing"
            render_status_table()

            try:
                update_job_status(personas_ws, job.row_index, "analyzing")
            except Exception:
                pass

            def make_progress_cb(row_idx: int):
                def cb(step_name: str, _message: str) -> None:
                    job_statuses[row_idx]["status"] = step_name.lower().replace(" ", "_")
                    render_status_table()
                return cb

            result = run_persona_pipeline(
                job=job.job_input,
                progress_cb=make_progress_cb(job.row_index),
            )

            if result.success:
                job_statuses[job.row_index]["status"] = "complete"
                logger.info(
                    "Persona batch row completed",
                    extra={"row_index": job.row_index, "job_key": job.job_input.job_key},
                )
            else:
                job_statuses[job.row_index]["status"] = "error"
                error_msg = result.error or "Unknown error"
                job_statuses[job.row_index]["error"] = error_msg
                logger.warning(
                    "Persona batch row failed",
                    extra={
                        "row_index": job.row_index,
                        "job_key": job.job_input.job_key,
                        "error": error_msg,
                    },
                )

            completed_jobs.append(
                {
                    "row_index": job.row_index,
                    "job_key": job.job_input.job_key,
                    "persona": job.job_input.persona.split("\n")[0][:50],
                    "persona_image_url": job.job_input.persona_image_url,
                    "gender": result.gender,
                    "clips_count": len(result.clips),
                    "success": result.success,
                    "error": result.error,
                }
            )

            try:
                write_persona_result(personas_ws, clips_ws, job.row_index, result)
            except Exception as e:
                st.warning(f"Failed to write results for row {job.row_index}: {e}")

            render_status_table()

        complete_count = sum(1 for v in job_statuses.values() if v["status"] == "complete")
        error_count = sum(1 for v in job_statuses.values() if v["status"] == "error")

        if error_count == 0:
            st.success(f"All {complete_count} persona(s) processed successfully!")
        else:
            st.warning(f"{complete_count} completed, {error_count} failed.")

        st.session_state["persona_batch_last_completed"] = completed_jobs

    _render_persona_review_section()


def _render_persona_review_section() -> None:
    completed_jobs: list[dict[str, Any]] | None = st.session_state.get("persona_batch_last_completed")
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
            st.image(job["persona_image_url"], caption=f"Persona image — {job['persona']}", width="stretch")
        else:
            st.info("No persona image available.")
    with col2:
        st.metric("Detected gender", job.get("gender") or "—")
        st.metric("Clips generated", job.get("clips_count") or 0)
        if not job["success"]:
            st.error(f"Error: {job.get('error')}")
