from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gspread


@dataclass
class GeneratingJob:
    row_index: int
    job_key: str
    swapped_start_frame_url: str
    analysis_json: str
    prompt_template: str
    raw_row: dict[str, Any]
    final_script_json: str = ""
    gender: str = ""  # "male" | "female" | "" — from new_frame_gender column


@dataclass
class ClipRow:
    row_index: int
    clip_key: str
    job_key: str
    clip_index: int
    dialogue: str
    duration_s: int
    status: str
    start_frame_url: str
    end_frame_url: str
    clip_video_url: str
    raw_row: dict[str, Any]


def _read_records(worksheet: gspread.Worksheet) -> list[dict[str, Any]]:
    """Return all data rows as dicts keyed by lower-cased headers, deduplicating headers."""
    all_values = worksheet.get_all_values()
    if not all_values:
        return []

    raw_headers = all_values[0]
    header_keys: list[str | None] = []
    seen: set[str] = set()

    for h in raw_headers:
        key = h.strip().lower()
        if not key or key in seen:
            header_keys.append(None)
            continue
        seen.add(key)
        header_keys.append(key)

    records: list[dict[str, Any]] = []
    for row_values in all_values[1:]:
        record: dict[str, Any] = {}
        for idx, key in enumerate(header_keys):
            if key is None:
                continue
            record[key] = row_values[idx] if idx < len(row_values) else ""
        records.append(record)

    return records


def read_generating_jobs(jobs_worksheet: gspread.Worksheet) -> list[GeneratingJob]:
    """Read all rows with status='generating' from a Jobs (or Personas) worksheet."""
    records = _read_records(jobs_worksheet)

    jobs: list[GeneratingJob] = []
    for i, row in enumerate(records, start=2):  # row 1 is header
        if str(row.get("status", "")).strip().lower() != "generating":
            continue

        job_key = str(row.get("job_key", "")).strip()
        if not job_key:
            continue

        # For Jobs sheets: swapped_start_frame_url is written by the batch pipeline (face swap result).
        # For Personas sheets: swapped_start_frame_url is absent — fall back to persona_image,
        # which serves as the Kling seed image directly without a face-swap step.
        swapped_start_frame_url = str(row.get("swapped_start_frame_url", "")).strip()
        if not swapped_start_frame_url:
            swapped_start_frame_url = str(row.get("persona_image", "")).strip()

        jobs.append(
            GeneratingJob(
                row_index=i,
                job_key=job_key,
                swapped_start_frame_url=swapped_start_frame_url,
                analysis_json=str(row.get("analysis_json", "")).strip(),
                prompt_template=str(row.get("prompt_template", "")).strip(),
                raw_row=dict(row),
                final_script_json=str(row.get("final_script_json", "")).strip(),
                gender=str(row.get("new_frame_gender", "")).strip().lower(),
            )
        )

    return jobs


def read_clips_for_job(
    clips_worksheet: gspread.Worksheet, job_key: str
) -> list[ClipRow]:
    """Read all Clips rows for a given job_key, sorted by clip_index."""
    records = _read_records(clips_worksheet)

    clips: list[ClipRow] = []
    for i, row in enumerate(records, start=2):
        if str(row.get("job_key", "")).strip() != job_key:
            continue

        try:
            clip_index = int(row.get("clip_index", 0) or 0)
        except (ValueError, TypeError):
            clip_index = 0

        try:
            duration_s = int(row.get("duration_s", 5) or 5)
        except (ValueError, TypeError):
            duration_s = 5

        clips.append(
            ClipRow(
                row_index=i,
                clip_key=str(row.get("clip_key", "")).strip(),
                job_key=str(row.get("job_key", "")).strip(),
                clip_index=clip_index,
                dialogue=str(row.get("dialogue", "")).strip(),
                duration_s=duration_s,
                status=str(row.get("status", "")).strip().lower(),
                start_frame_url=str(row.get("start_frame_url", "")).strip(),
                end_frame_url=str(row.get("end_frame_url", "")).strip(),
                clip_video_url=str(row.get("clip_video_url", "")).strip(),
                raw_row=dict(row),
            )
        )

    clips.sort(key=lambda c: c.clip_index)
    return clips
