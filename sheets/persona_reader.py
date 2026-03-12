from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

import gspread

from pipeline.persona_runner import PersonaJobInput


@dataclass
class PersonaSheetJob:
    row_index: int  # 1-based row in the sheet
    job_input: PersonaJobInput
    status: str
    raw_row: dict[str, Any]


def _resolve_job_key_col(header_keys: list[str | None]) -> int | None:
    """Return the 1-based column index of the job_key header, or None."""
    for idx, key in enumerate(header_keys):
        if key == "job_key":
            return idx + 1
    return None


def read_queued_persona_jobs(worksheet: gspread.Worksheet) -> list[PersonaSheetJob]:
    """Read all rows with status == 'queued' from the Personas worksheet.

    Expected columns (case-insensitive):
        job_key, persona, pain_point, speechify_solution,
        hook_sample, hook_emotion, persona_image, status

    If job_key is empty for a queued row, a UUID is generated and written
    back to the sheet so downstream writers can reference it.
    """
    all_values = worksheet.get_all_values()
    if not all_values:
        return []

    raw_headers = all_values[0]
    header_keys: list[str | None] = []
    seen: set[str] = set()

    for h in raw_headers:
        key = h.strip().lower()
        if not key:
            header_keys.append(None)
            continue
        if key in seen:
            header_keys.append(None)
            continue
        seen.add(key)
        header_keys.append(key)

    job_key_col = _resolve_job_key_col(header_keys)

    records: list[dict[str, Any]] = []
    for row_values in all_values[1:]:
        record: dict[str, Any] = {}
        for idx, key in enumerate(header_keys):
            if key is None:
                continue
            if idx >= len(row_values):
                continue
            record[key] = row_values[idx]
        records.append(record)

    jobs: list[PersonaSheetJob] = []
    cells_to_update: list[gspread.Cell] = []

    for i, row in enumerate(records, start=2):  # row 1 is header
        status = str(row.get("status", "")).strip().lower()
        if status != "queued":
            continue

        job_key = str(row.get("job_key", "")).strip()
        persona = str(row.get("persona", "")).strip()
        pain_point = str(row.get("pain_point", "")).strip()
        speechify_solution = str(row.get("speechify_solution", "")).strip()
        hook_sample = str(row.get("hook_sample", "")).strip()
        hook_emotion = str(row.get("hook_emotion", "")).strip()
        persona_image_url = str(row.get("persona_image", "")).strip()

        if not persona_image_url:
            continue

        if not job_key:
            if job_key_col is None:
                continue
            job_key = uuid.uuid4().hex[:12]
            cells_to_update.append(gspread.Cell(row=i, col=job_key_col, value=job_key))

        jobs.append(
            PersonaSheetJob(
                row_index=i,
                job_input=PersonaJobInput(
                    job_key=job_key,
                    persona=persona,
                    pain_point=pain_point,
                    speechify_solution=speechify_solution,
                    hook_sample=hook_sample,
                    hook_emotion=hook_emotion,
                    persona_image_url=persona_image_url,
                ),
                status=status,
                raw_row=dict(row),
            )
        )

    if cells_to_update:
        worksheet.update_cells(cells_to_update)

    return jobs
