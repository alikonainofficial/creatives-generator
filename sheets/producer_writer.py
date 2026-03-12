from __future__ import annotations

from datetime import datetime
from typing import Any

import gspread

from sheets.writer import update_job_status


def update_clip(
    clips_worksheet: gspread.Worksheet,
    row_index: int,
    fields: dict[str, Any],
) -> None:
    """Update specific fields on a Clips sheet row, resolved by column name."""
    headers = clips_worksheet.row_values(1)
    header_map = {
        h.strip(): idx for idx, h in enumerate(headers, start=1) if h.strip()
    }
    for field, value in fields.items():
        col_idx = header_map.get(field)
        if col_idx is not None:
            clips_worksheet.update_cell(row_index, col_idx, value)


def save_blueprint_to_job(
    jobs_worksheet: gspread.Worksheet,
    row_index: int,
    prompt_template_json: str,
) -> None:
    """Persist the generated blueprint JSON back to the Jobs sheet row."""
    update_job_status(
        jobs_worksheet,
        row_index,
        "generating",
        extra_fields={"prompt_template": prompt_template_json},
    )


def mark_job_done(
    jobs_worksheet: gspread.Worksheet,
    row_index: int,
    fal_video_url: str,
    drive_video_url: str,
) -> None:
    """Mark a job row as done and write both video URLs."""
    update_job_status(
        jobs_worksheet,
        row_index,
        "done",
        extra_fields={
            "fal_video_url": fal_video_url,
            "drive_video_url": drive_video_url,
            # final_video_url mirrors the Drive link so n8n-compatible consumers
            # that reference final_video_url continue to work.
            "final_video_url": drive_video_url,
            "Date generated": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        },
    )


def mark_job_error(
    jobs_worksheet: gspread.Worksheet,
    row_index: int,
    error: str,
) -> None:
    """Mark a job row as errored."""
    update_job_status(
        jobs_worksheet,
        row_index,
        "error",
        extra_fields={"error": error},
    )


def claim_job(jobs_worksheet: gspread.Worksheet, row_index: int) -> None:
    """Transition a job from 'generating' → 'producing'.

    Written immediately when a worker picks up a job so that a concurrent
    session (or a second worker) won't attempt to process the same row.
    """
    update_job_status(jobs_worksheet, row_index, "producing")
