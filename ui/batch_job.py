from __future__ import annotations

import asyncio
import json
import threading
import time
from typing import Any

import streamlit as st
import traceback

from cache import get_session_cache, load_sheet_cache, save_to_sheet_cache
from logging_config import get_logger, setup_logging
from pipeline.runner import PipelineResult, run_pipeline
from sheets.auth import get_gspread_client
from sheets.reader import read_queued_jobs, get_worksheet
from sheets.writer import write_job_result, update_job_status


logger = get_logger(__name__, ui_mode="batch")


def render() -> None:
    setup_logging()
    st.header("Clone Batch")
    st.caption("Analyze source videos, swap faces from a reference image, and generate scripts — in bulk via Google Sheet.")

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

    with st.form("batch_form"):
        sheet_url = st.text_input(
            "Google Sheet URL or ID",
            placeholder="https://docs.google.com/spreadsheets/d/...",
        )
        jobs_tab = st.text_input("Jobs Tab Name", value="Jobs")
        clips_tab = st.text_input("Clips Tab Name", value="Clips")
        cache_tab = st.text_input("VideoCache Tab Name (optional)", value="VideoCache")
        submitted = st.form_submit_button("Load & Process Queued Jobs", type="primary")

    if submitted:
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
            },
        )

        # Connect to sheet
        try:
            client = get_gspread_client()

            sheet_id = sheet_url.strip()
            if "spreadsheets/d/" in sheet_id:
                sheet_id = sheet_id.split("spreadsheets/d/")[1].split("/")[0]

            spreadsheet = client.open_by_key(sheet_id)
            jobs_ws = get_worksheet(spreadsheet, jobs_tab.strip())

            try:
                clips_ws = get_worksheet(spreadsheet, clips_tab.strip())
            except ValueError:
                clips_ws = None
                st.warning(
                    f"Clips tab '{clips_tab}' not found — clips will not be written to sheet."
                )

            # Load video cache from sheet if available
            video_cache = get_session_cache()
            try:
                cache_ws = get_worksheet(spreadsheet, cache_tab.strip())
                sheet_cache = load_sheet_cache(cache_ws)
                video_cache.update(sheet_cache)
                st.info(f"Loaded {len(sheet_cache)} cached entries from VideoCache tab.")
            except ValueError:
                cache_ws = None

            jobs = read_queued_jobs(jobs_ws)

        except Exception as e:
            # Surface full exception details so debugging is easier.
            logger.exception(
                "Failed to connect to Google Sheet",
                extra={"sheet_id": sheet_id},
            )
            st.error("Failed to connect to Google Sheet:")
            st.exception(e)
            # Also print full traceback to the server logs for deeper inspection.
            traceback.print_exc()
            return

        if not jobs:
            st.info("No queued jobs found in the sheet.")
            return

        st.info(f"Found {len(jobs)} queued job(s). Starting processing...")

        # Live status table
        job_statuses = {job.row_index: {"url": job.source_video_url, "status": "pending", "error": ""} for job in jobs}

        status_placeholder = st.empty()

        def render_status_table() -> None:
            rows = []
            for row_idx, info in job_statuses.items():
                rows.append({
                    "Row": row_idx,
                    "Video": info["url"][-60:] if len(info["url"]) > 60 else info["url"],
                    "Status": info["status"],
                    "Error": info["error"],
                })
            status_placeholder.dataframe(rows, width="stretch")

        render_status_table()

        # Persist the sheet context so we can later regenerate images for specific
        # rows without re-running the entire batch.
        st.session_state["batch_last_sheet_context"] = {
            "sheet_id": sheet_id,
            "jobs_tab": jobs_tab.strip(),
            "clips_tab": clips_tab.strip(),
            "cache_tab": cache_tab.strip(),
        }

        completed_jobs: list[dict[str, Any]] = []

        # Process jobs sequentially (can extend to parallel later)
        for job in jobs:
            logger.info(
                "Starting batch job row",
                extra={
                    "row_index": job.row_index,
                    "source_video_url": job.source_video_url,
                    "reference_image_url": job.reference_image_url,
                },
            )
            job_statuses[job.row_index]["status"] = "analyzing"
            render_status_table()

            # Mark as in-progress in sheet
            try:
                update_job_status(jobs_ws, job.row_index, "analyzing")
            except Exception:
                pass

            def make_progress_cb(row_idx: int):
                def cb(step_name: str, message: str) -> None:
                    job_statuses[row_idx]["status"] = step_name.lower().replace(" ", "_")
                    render_status_table()
                return cb

            result = run_pipeline(
                source_video_url=job.source_video_url,
                reference_image_url=job.reference_image_url,
                video_cache=video_cache,
                progress_cb=make_progress_cb(job.row_index),
            )

            if result.success:
                job_statuses[job.row_index]["status"] = "complete"
                logger.info(
                    "Batch job row completed",
                    extra={"row_index": job.row_index, "success": True},
                )
            else:
                job_statuses[job.row_index]["status"] = "error"
                error_msg = result.error or "Unknown error"
                job_statuses[job.row_index]["error"] = error_msg
                logger.warning(
                    "Batch job row failed",
                    extra={"row_index": job.row_index, "success": False, "error": error_msg},
                )

            # Track for post-batch review.
            completed_jobs.append(
                {
                    "row_index": job.row_index,
                    "source_video_url": job.source_video_url,
                    "reference_image_url": job.reference_image_url,
                    "swapped_image_url": result.swapped_image_url,
                    "success": result.success,
                    "error": result.error,
                }
            )

            # Write result back to sheet
            try:
                write_job_result(jobs_ws, clips_ws, job.row_index, result)
            except Exception as e:
                st.warning(f"Failed to write results for row {job.row_index}: {e}")

            # Persist updated video-level cache entry for this job back to the
            # VideoCache sheet so future batch runs can resume from completed steps.
            try:
                if cache_ws is not None and result.context is not None:
                    video_key = result.context.video_key
                    entry = video_cache.get(video_key, {})
                    if isinstance(entry, dict):
                        # Enrich the cache entry with metadata needed for the VideoCache sheet.
                        entry.setdefault("source_video_url", job.source_video_url)

                        if result.analysis and result.analysis.face_visibility.best_window:
                            window = result.analysis.face_visibility.best_window
                            entry.setdefault("face_time_start", window.time_start_s)
                            entry.setdefault("face_time_end", window.time_end_s)

                        # Mark this cache row as completed so it's easy to filter.
                        entry.setdefault("status", "done")

                        # Flatten the latest script onto the top-level cache entry so
                        # that it can be serialized into the VideoCache tab
                        # (`final_script_json` column) by `save_to_sheet_cache`.
                        if result.script is not None:
                            entry["script"] = result.script

                        if entry:
                            save_to_sheet_cache(cache_ws, video_key, entry)
            except Exception:
                # Cache persistence is best-effort; ignore failures here.
                pass

            render_status_table()

        complete_count = sum(1 for v in job_statuses.values() if v["status"] == "complete")
        error_count = sum(1 for v in job_statuses.values() if v["status"] == "error")

        if error_count == 0:
            st.success(f"All {complete_count} jobs completed successfully!")
        else:
            st.warning(f"{complete_count} completed, {error_count} failed.")

        # Persist summary so the user can review and optionally regenerate images
        # without re-running the entire batch.
        st.session_state["batch_last_completed_jobs"] = completed_jobs

    # Handle any pending image-only regeneration request before rendering the
    # review UI so that progress / results are reflected immediately.
    regen_target_row = st.session_state.pop("batch_regen_target_row", None)
    if regen_target_row is not None:
        sheet_ctx = st.session_state.get("batch_last_sheet_context")  # type: ignore[assignment]
        completed_jobs: list[dict[str, Any]] | None = st.session_state.get("batch_last_completed_jobs")  # type: ignore[assignment]
        if isinstance(sheet_ctx, dict) and isinstance(completed_jobs, list):
            job = next((j for j in completed_jobs if j.get("row_index") == regen_target_row), None)
            if job is not None:
                st.session_state["batch_regen_in_progress"] = True
                try:
                    _regenerate_single_batch_job_image(job, sheet_ctx)
                finally:
                    st.session_state["batch_regen_in_progress"] = False

    # Regardless of whether we just ran a batch, if we have a previous batch
    # summary in session_state, show the review / regeneration UI so that
    # interacting with the dropdown doesn't make it disappear.
    _render_batch_review_section()


def _render_batch_review_section() -> None:
    completed_jobs: list[dict[str, Any]] | None = st.session_state.get("batch_last_completed_jobs")  # type: ignore[assignment]
    sheet_ctx: dict[str, str] | None = st.session_state.get("batch_last_sheet_context")  # type: ignore[assignment]

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

    # Map selection back to job dict
    selected_idx = options.index(selected_label)
    job = completed_jobs[selected_idx]

    if job.get("swapped_image_url"):
        st.image(job["swapped_image_url"], caption=f"Row {job['row_index']} swapped image", width="stretch")
    else:
        st.info("This job did not produce a swapped image.")

    regen_col1, regen_col2 = st.columns(2)
    with regen_col1:
        st.caption("If you like this result, no action is needed.")
    with regen_col2:
        regen_in_progress: bool = bool(st.session_state.get("batch_regen_in_progress", False))  # type: ignore[assignment]
        clicked = st.button(
            "Regenerate image only for this row",
            key="batch_regen_image_only",
            disabled=regen_in_progress,
        )
        if regen_in_progress:
            st.caption("Image regeneration in progress...")
        if clicked:
            # Defer the actual regeneration work to the top-level render() cycle
            # so we can control button disabling and messaging via session_state.
            st.session_state["batch_regen_target_row"] = job["row_index"]
            st.info("Regenerating image with the same settings...")
            st.rerun()


def _regenerate_single_batch_job_image(job: dict[str, Any], sheet_ctx: dict[str, str]) -> None:
    """Regenerate only the swapped image for a completed batch job row."""
    try:
        client = get_gspread_client()
        spreadsheet = client.open_by_key(sheet_ctx["sheet_id"])
        jobs_ws = get_worksheet(spreadsheet, sheet_ctx["jobs_tab"])

        try:
            clips_ws = get_worksheet(spreadsheet, sheet_ctx["clips_tab"])
        except ValueError:
            clips_ws = None

        # Rebuild / refresh the video cache from the sheet (best-effort).
        video_cache = get_session_cache()
        cache_ws = None
        try:
            cache_ws = get_worksheet(spreadsheet, sheet_ctx["cache_tab"])
            sheet_cache = load_sheet_cache(cache_ws)
            video_cache.update(sheet_cache)
        except ValueError:
            cache_ws = None

        row_index = job["row_index"]

        # Run the pipeline in "image-only" mode so we keep analysis/script/etc.
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

        # Only update the swapped image URL — leave all other columns (gender,
        # script, clips, etc.) untouched so we don't accidentally blank them out.
        update_job_status(jobs_ws, row_index, "generating", extra_fields={
            "swapped_start_frame_url": result.swapped_image_url or "",
        })

        # Persist updated cache entry back to the sheet if possible.
        try:
            if cache_ws is not None and result.context is not None:
                video_key = result.context.video_key
                entry = video_cache.get(video_key, {})
                if isinstance(entry, dict) and entry:
                    save_to_sheet_cache(cache_ws, video_key, entry)
        except Exception:
            pass

        # Update in-memory record so the new image is visible in the review UI.
        job["swapped_image_url"] = result.swapped_image_url
        job["success"] = result.success
        job["error"] = result.error

        st.success(f"Regenerated image for row {row_index}. The sheet has been updated.")
    except Exception as e:
        st.error(f"Failed to regenerate image for row {job.get('row_index')}: {e}")
