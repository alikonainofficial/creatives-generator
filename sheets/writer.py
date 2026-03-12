from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import gspread

from pipeline.clip_timing import TimedClip
from pipeline.runner import PipelineResult


def update_job_status(
    worksheet: gspread.Worksheet,
    row_index: int,
    status: str,
    extra_fields: dict[str, Any] | None = None,
) -> None:
    """Update the status column for a job row.

    Column *order* is ignored; we always resolve updates by header name.
    Header matching is case-insensitive so the sheet is the source of truth
    for the actual column names.
    """
    headers = worksheet.row_values(1)
    if not headers:
        return

    # Map lowercase header -> (original header, column index)
    header_map: dict[str, tuple[str, int]] = {
        h.strip().lower(): (h, idx) for idx, h in enumerate(headers, start=1) if h.strip()
    }

    updates: list[tuple[int, Any]] = []

    # Always update the status column if present.
    status_entry = header_map.get("status")
    if status_entry:
        _, status_col = status_entry
        updates.append((status_col, status))

    # Update any extra fields by matching header name case-insensitively.
    if extra_fields:
        for key, value in extra_fields.items():
            entry = header_map.get(key.strip().lower())
            if entry:
                _, col_idx = entry
                updates.append((col_idx, value))

    for col_idx, value in updates:
        worksheet.update_cell(row_index, col_idx, value)


def write_job_result(
    jobs_worksheet: gspread.Worksheet,
    clips_worksheet: gspread.Worksheet | None,
    row_index: int,
    result: PipelineResult,
) -> None:
    """Write pipeline results back to the Jobs sheet and append clips to Clips tab.

    This function is header-driven: it looks up columns by name, so the
    physical order of columns in the Google Sheet does not matter.
    """
    if not result.success:
        error_msg = result.error or "Unknown error"
        update_job_status(
            jobs_worksheet,
            row_index,
            "error",
            # Support either an "error" or "error_message" column depending
            # on how the sheet is configured.
            extra_fields={
                "error": error_msg,
                "error_message": error_msg,
            },
        )
        return

    analysis_json = ""
    face_time_start: float | str = ""
    face_time_end: float | str = ""
    if result.analysis:
        analysis_json = result.analysis.model_dump_json()
        window = result.analysis.face_visibility.best_window
        if window is not None:
            face_time_start = window.time_start_s
            face_time_end = window.time_end_s

    script_json = ""
    if result.clips:
        # Serialize the final script in the n8n-compatible schema expected by
        # downstream tooling:
        # {
        #   "script": [
        #     {"speaker": "main", "text": str, "insert_demo": bool, "demo_type": null | str, "mode": null | str},
        #     ...
        #   ]
        # }
        script_entries: list[dict[str, Any]] = []

        # Build a lookup from segment_id -> ScriptSegment so we can attach
        # demo metadata (demo_type/mode) from the rewrite stage to the clips.
        segment_meta: dict[int, Any] = {}
        if result.script is not None:
            for seg in result.script.segments:
                segment_meta[seg.segment_id] = seg

        for clip in result.clips:
            seg = segment_meta.get(clip.segment_id)

            # Prefer the segment's is_demo flag if available to keep behavior
            # aligned with the original script segmentation.
            insert_demo = bool(getattr(seg, "is_demo", clip.is_demo))

            demo_type = None
            mode = None
            if seg is not None and insert_demo:
                demo_type = getattr(seg, "demo_type", None)
                mode = getattr(seg, "mode", None)

            script_entries.append(
                {
                    "speaker": "main",
                    "text": clip.dialogue,
                    "insert_demo": insert_demo,
                    # Only the single insertion segment should carry a concrete
                    # demo_type/mode; all non-demo segments keep these as null.
                    "demo_type": demo_type,
                    "mode": mode,
                }
            )
        script_json = json.dumps({"script": script_entries}, ensure_ascii=False)
    elif result.script:
        # Fallback for scenarios where clips were not generated but a script is
        # available: preserve the original RewrittenScript JSON.
        script_json = result.script.model_dump_json()

    completed_at = datetime.utcnow()
    extra = {
        "analysis_json": analysis_json,
        "swapped_start_frame_url": result.swapped_image_url or "",
        "final_script_json": script_json,
        # Human-readable timestamp in the "Date generated" column instead of a
        # raw ISO field, per sheet expectations.
        "Date generated": completed_at.strftime("%Y-%m-%d %H:%M:%S UTC"),
        # Expose the anchor frame metadata directly on the Jobs sheet so
        # downstream tooling can consume it without re-parsing analysis_json.
        "face_time_start": face_time_start,
        "face_time_end": face_time_end,
        "extracted_frame_url": result.anchor_frame_url or "",
        "total_clips": len(result.clips) if result.clips else 0,
        # Surface detected gender so downstream tools (including n8n) can
        # read it directly from the Jobs tab.
        "new_frame_gender": result.gender or "",
    }

    # If a job_key is available, surface a row-scoped variant on the Jobs tab
    # so downstream processes can correlate rows with Clips entries. This keeps
    # keys unique even when multiple rows share the same source video URL.
    row_scoped_job_key: str | None = None
    if result.context is not None and getattr(result.context, "job_key", None):
        base_job_key = result.context.job_key
        row_scoped_job_key = f"{base_job_key}_row_{row_index}"
        extra["job_key"] = row_scoped_job_key

    # After the analysis/script/image pipeline completes, the next step is
    # downstream video generation. Reflect that by marking the job as
    # "generating" rather than "complete" here.
    update_job_status(jobs_worksheet, row_index, "generating", extra_fields=extra)

    # Append clips to Clips tab
    if clips_worksheet and result.clips and result.context:
        # Use the same row-scoped job_key that we wrote to the Jobs tab so
        # each Clips entry maps back to a unique job row, even when multiple
        # rows share the same underlying video.
        if row_scoped_job_key is not None:
            job_key = row_scoped_job_key
        else:
            job_key = result.context.job_key

        # Resolve clip columns by header name so we can respect any column order.
        clip_headers = clips_worksheet.row_values(1)
        header_to_col: dict[str, int] = {
            h.strip(): idx for idx, h in enumerate(clip_headers, start=1) if h.strip()
        }

        rows_to_append = []
        for clip in result.clips:
            row_values = ["" for _ in clip_headers]

            # job_key
            col = header_to_col.get("job_key")
            if col is not None:
                row_values[col - 1] = job_key

            # clip_index (segment id)
            col = header_to_col.get("clip_index")
            if col is not None:
                row_values[col - 1] = clip.segment_id

            # dialogue
            col = header_to_col.get("dialogue")
            if col is not None:
                row_values[col - 1] = clip.dialogue

            # duration_s (rounded down to nearest whole second)
            col = header_to_col.get("duration_s")
            if col is not None:
                try:
                    # Always floor to an integer number of seconds.
                    duration_int = int(clip.duration_seconds)
                except (TypeError, ValueError):
                    duration_int = clip.duration_seconds
                row_values[col - 1] = duration_int

            # clip_key: <job_key>_clip_<clip_index>
            col = header_to_col.get("clip_key")
            if col is not None:
                row_values[col - 1] = f"{job_key}_clip_{clip.segment_id}"

            # Initial clip status
            col = header_to_col.get("status")
            if col is not None:
                row_values[col - 1] = "queued"

            rows_to_append.append(row_values)

        if rows_to_append:
            clips_worksheet.append_rows(rows_to_append)
