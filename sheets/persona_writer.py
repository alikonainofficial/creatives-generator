from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import gspread

from pipeline.persona_runner import PersonaPipelineResult
from sheets.writer import update_job_status


def write_persona_result(
    personas_worksheet: gspread.Worksheet,
    clips_worksheet: gspread.Worksheet | None,
    row_index: int,
    result: PersonaPipelineResult,
) -> None:
    """Write persona pipeline results back to the Personas sheet and Clips tab.

    Writes to the Personas tab:
        new_frame_gender, final_script_json, Date generated, total_clips, status

    Appends clip rows to the Clips tab (same schema as existing jobs):
        job_key, clip_index, dialogue, duration_s, status, clip_key
    """
    if not result.success or result.job_input is None:
        error_msg = result.error or "Unknown error"
        update_job_status(
            personas_worksheet,
            row_index,
            "error",
            extra_fields={
                "error": error_msg,
                "error_message": error_msg,
            },
        )
        return

    script_json = ""
    if result.clips:
        script_entries: list[dict[str, Any]] = []

        segment_meta: dict[int, Any] = {}
        if result.script is not None:
            for seg in result.script.segments:
                segment_meta[seg.segment_id] = seg

        for clip in result.clips:
            seg = segment_meta.get(clip.segment_id)
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
                    "demo_type": demo_type,
                    "mode": mode,
                }
            )
        script_json = json.dumps({"script": script_entries}, ensure_ascii=False)
    elif result.script:
        script_json = result.script.model_dump_json()

    completed_at = datetime.utcnow()
    extra: dict[str, Any] = {
        "new_frame_gender": result.gender or "",
        "final_script_json": script_json,
        "Date generated": completed_at.strftime("%Y-%m-%d %H:%M:%S UTC"),
        "total_clips": len(result.clips) if result.clips else 0,
    }

    # Write the generated blueprint as prompt_template so the Producer always
    # finds a valid blueprint and never hits the no-analysis_json dead-end.
    if result.blueprint:
        extra["prompt_template"] = json.dumps(result.blueprint, ensure_ascii=False)

    update_job_status(personas_worksheet, row_index, "generating", extra_fields=extra)

    if clips_worksheet and result.clips and result.job_input:
        job_key = result.job_input.job_key

        clip_headers = clips_worksheet.row_values(1)
        header_to_col: dict[str, int] = {
            h.strip(): idx for idx, h in enumerate(clip_headers, start=1) if h.strip()
        }

        rows_to_append = []
        for clip in result.clips:
            row_values = ["" for _ in clip_headers]

            col = header_to_col.get("job_key")
            if col is not None:
                row_values[col - 1] = job_key

            col = header_to_col.get("clip_index")
            if col is not None:
                row_values[col - 1] = clip.segment_id

            col = header_to_col.get("dialogue")
            if col is not None:
                row_values[col - 1] = clip.dialogue

            col = header_to_col.get("duration_s")
            if col is not None:
                try:
                    row_values[col - 1] = int(clip.duration_seconds)
                except (TypeError, ValueError):
                    row_values[col - 1] = clip.duration_seconds

            col = header_to_col.get("clip_key")
            if col is not None:
                row_values[col - 1] = f"{job_key}_clip_{clip.segment_id}"

            col = header_to_col.get("status")
            if col is not None:
                row_values[col - 1] = "queued"

            rows_to_append.append(row_values)

        if rows_to_append:
            clips_worksheet.append_rows(rows_to_append)
