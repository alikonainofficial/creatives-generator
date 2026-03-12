from __future__ import annotations

import threading
import time
from typing import Any

import streamlit as st

from cache import get_session_cache, load_disk_cache, save_disk_cache
from logging_config import get_logger, setup_logging
from pipeline.runner import PipelineResult, run_pipeline


logger = get_logger(__name__, ui_mode="single_job")


def render() -> None:
    setup_logging()
    st.header("Single Job")
    st.caption("Process one video at a time with live progress.")

    with st.expander("ℹ️ How this works & what to prepare", expanded=False):
        st.markdown(
            """
**What this mode does**

Runs the full video-cloning pipeline for a single video in one shot:
1. Downloads and analyzes the source video with Gemini to find the best face window.
2. Extracts the anchor frame and swaps the reference face in via Fal.ai.
3. Rewrites the video script using Gemini, splitting it into timed clips.

This mode is best for **testing a single video** before running a full batch.

---

**Inputs**

| Field | What to provide |
|---|---|
| **Source Video URL** | A publicly accessible video URL (Cloudinary, S3, etc.). This is the original ad video whose speaker face will be replaced. |
| **Reference Image URL** | A publicly accessible photo of the face you want to swap in. Clear, front-facing headshots work best. |

---

**Outputs**

After the pipeline finishes you will see:
- **Anchor Frame** — the extracted frame from the source video used as the swap base.
- **Swapped Face** — the anchor frame with the reference face applied.
- **Generated Script** — a table of clip segments with dialogue, duration, and word count.
- **Step timing details** (expandable) — per-step duration and any errors.

Results are kept in your session. Use **Regenerate image only** to re-run just the face swap without re-analyzing the video.
            """
        )

    # If we have a previous successful run, always show it so the user can
    # review or regenerate the image without re-running the full pipeline.
    last_result: PipelineResult | None = st.session_state.get("single_job_last_result")  # type: ignore[assignment]
    last_inputs: dict[str, str] | None = st.session_state.get("single_job_last_inputs")  # type: ignore[assignment]

    with st.form("single_job_form"):
        source_video_url = st.text_input(
            "Source Video URL",
            placeholder="https://res.cloudinary.com/your-cloud/video/upload/...",
        )
        reference_image_url = st.text_input(
            "Reference Image URL",
            placeholder="https://example.com/reference-face.jpg",
        )
        submitted = st.form_submit_button("Run Pipeline", type="primary")

    # Handle "Regenerate image only" action from a previous run.
    regenerate_requested = st.session_state.pop("single_job_regenerate_image_only", False)
    if regenerate_requested and last_inputs is not None:
        logger.info(
            "Single-job image regeneration requested",
            extra={
                "source_video_url": last_inputs.get("source_video_url"),
                "reference_image_url": last_inputs.get("reference_image_url"),
            },
        )
        # Mark regeneration as in progress so the UI can disable the button.
        st.session_state["single_job_regen_in_progress"] = True
        try:
            _run_pipeline_with_progress(
                source_video_url=last_inputs["source_video_url"],
                reference_image_url=last_inputs["reference_image_url"],
                regenerate_image_only=True,
            )
            last_result = st.session_state.get("single_job_last_result")  # type: ignore[assignment]
        finally:
            # Always clear the in-progress flag when the run completes.
            st.session_state["single_job_regen_in_progress"] = False

    # Handle fresh run submission.
    if submitted:
        if not source_video_url.strip():
            st.error("Please enter a Source Video URL.")
            return
        if not reference_image_url.strip():
            st.error("Please enter a Reference Image URL.")
            return

        logger.info(
            "Single-job pipeline submission",
            extra={
                "source_video_url": source_video_url.strip(),
                "reference_image_url": reference_image_url.strip(),
            },
        )

        st.session_state["single_job_last_inputs"] = {
            "source_video_url": source_video_url.strip(),
            "reference_image_url": reference_image_url.strip(),
        }
        _run_pipeline_with_progress(
            source_video_url=source_video_url.strip(),
            reference_image_url=reference_image_url.strip(),
            regenerate_image_only=False,
        )
        last_result = st.session_state.get("single_job_last_result")  # type: ignore[assignment]

    # Always show the latest result (if any) so the user can review / regenerate.
    if last_result is not None:
        _display_result(last_result)


def _run_pipeline_with_progress(
    source_video_url: str,
    reference_image_url: str,
    regenerate_image_only: bool,
) -> None:
    # Progress tracking
    step_messages: list[tuple[str, str]] = []
    result_holder: list[PipelineResult] = []
    error_holder: list[str] = []
    done_event = threading.Event()

    def progress_cb(step_name: str, message: str) -> None:
        step_messages.append((step_name, message))

    def run_in_thread() -> None:
        try:
            # Merge in-memory session cache with any persisted disk cache so we
            # can resume work across server restarts.
            cache = get_session_cache()
            if not cache:
                cache.update(load_disk_cache())

            result = run_pipeline(
                source_video_url=source_video_url,
                reference_image_url=reference_image_url,
                video_cache=cache,
                progress_cb=progress_cb,
                regenerate_image_only=regenerate_image_only,
            )
            result_holder.append(result)
            # Persist updated cache snapshot to disk so subsequent runs (and
            # future sessions) can reuse completed steps.
            save_disk_cache(cache)
        except Exception as e:
            error_holder.append(str(e))
        finally:
            done_event.set()

    thread = threading.Thread(target=run_in_thread, daemon=True)
    thread.start()

    status_label = "Regenerating image..." if regenerate_image_only else "Running pipeline..."

    # Live progress display
    with st.status(status_label, expanded=True) as status_widget:
        last_count = 0
        while not done_event.is_set():
            if len(step_messages) > last_count:
                for _, msg in step_messages[last_count:]:
                    st.write(msg)
                last_count = len(step_messages)
            time.sleep(0.5)

        # Flush remaining messages
        for _, msg in step_messages[last_count:]:
            st.write(msg)

        if error_holder:
            status_widget.update(label="Pipeline crashed", state="error")
            st.error(error_holder[0])
            # Clear any previous successful result so we don't show stale data.
            st.session_state.pop("single_job_last_result", None)
            return

        if not result_holder:
            status_widget.update(label="No result returned", state="error")
            st.session_state.pop("single_job_last_result", None)
            return

        result = result_holder[0]

        if result.success:
            label = "Image regenerated!" if regenerate_image_only else "Pipeline complete!"
            status_widget.update(label=label, state="complete")
        else:
            status_widget.update(label=f"Pipeline failed: {result.error}", state="error")
            # Do not overwrite previous good result on failure.
            return

    # Persist latest successful result for review / regeneration.
    if result_holder and result_holder[0].success:
        st.session_state["single_job_last_result"] = result_holder[0]


def _display_result(result: PipelineResult) -> None:
    if not result.success:
        st.error(f"Pipeline failed: {result.error}")
        _show_step_summary(result)
        return

    st.success(f"Pipeline completed in {result.total_duration:.1f}s")

    # Images side by side
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Anchor Frame")
        if result.anchor_frame_url:
            st.image(result.anchor_frame_url, width="stretch")
        else:
            st.caption("No anchor frame")

    with col2:
        st.subheader("Swapped Face")
        if result.swapped_image_url:
            st.image(result.swapped_image_url, width="stretch")

            regen_col1, regen_col2 = st.columns(2)
            with regen_col1:
                st.caption("If you like this result, no action is needed.")
            with regen_col2:
                regen_in_progress: bool = st.session_state.get("single_job_regen_in_progress", False)  # type: ignore[assignment]
                clicked = st.button(
                    "Regenerate image only",
                    key="single_job_regen_image_only",
                    disabled=regen_in_progress,
                )
                if regen_in_progress:
                    st.caption("Image regeneration in progress...")
                if clicked:
                    # Trigger a regeneration on the next run; the top-level
                    # render() function will pick this up and call the pipeline
                    # in image-only mode using the last saved inputs.
                    st.session_state["single_job_regenerate_image_only"] = True
                    st.info("Regenerating image with the same settings...")
                    # Immediately rerun so the next render cycle sees the flag
                    # and starts the pipeline with live progress.
                    st.rerun()
        else:
            st.caption("No swapped image")

    # Script table
    if result.clips:
        st.subheader("Generated Script")
        clip_data = [
            {
                "Segment": clip.segment_id,
                "Dialogue": clip.dialogue,
                "Duration (s)": clip.duration_seconds,
                "Words": clip.word_count,
                "Demo": "Yes" if clip.is_demo else "No",
            }
            for clip in result.clips
        ]
        st.dataframe(clip_data, width="stretch")

        total_duration = sum(c.duration_seconds for c in result.clips)
        total_words = sum(c.word_count for c in result.clips)

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Clips", len(result.clips))
        col2.metric("Total Words", total_words)
        col3.metric("Est. Duration", f"{total_duration:.1f}s")

    # Detected gender
    if result.gender:
        st.caption(f"Detected speaker gender: **{result.gender}**")

    _show_step_summary(result)


def _show_step_summary(result: PipelineResult) -> None:
    with st.expander("Step timing details"):
        for step in result.steps:
            icon = "✓" if step.success else "✗"
            status = f"{step.duration_seconds:.2f}s"
            if not step.success and step.error:
                status += f" — {step.error}"
            st.caption(f"{icon} **{step.name}** — {status}")
