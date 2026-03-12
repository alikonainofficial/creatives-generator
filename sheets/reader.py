from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gspread


@dataclass
class SheetJob:
    row_index: int  # 1-based row in the sheet
    source_video_url: str
    reference_image_url: str
    status: str
    raw_row: dict[str, Any]


def read_queued_jobs(worksheet: gspread.Worksheet) -> list[SheetJob]:
    """Read all rows with status == 'queued' from the Jobs worksheet."""
    # `get_all_records()` raises if the header row has duplicate names. Some
    # sheets may legitimately have repeated headers (e.g. multiple feedback
    # columns), so we reconstruct records manually while ignoring duplicate
    # header names beyond the first occurrence.
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
            # Ignore subsequent duplicate headers (e.g. repeated feedback cols).
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
            if idx >= len(row_values):
                continue
            record[key] = row_values[idx]
        records.append(record)

    jobs: list[SheetJob] = []

    for i, row in enumerate(records, start=2):  # row 1 is header
        status = str(row.get("status", "")).strip().lower()
        if status != "queued":
            continue

        source_url = str(row.get("source_video_url", "")).strip()
        ref_url = str(row.get("reference_image_url", "")).strip()

        if not source_url or not ref_url:
            continue

        jobs.append(
            SheetJob(
                row_index=i,
                source_video_url=source_url,
                reference_image_url=ref_url,
                status=status,
                raw_row=dict(row),
            )
        )

    return jobs


def get_worksheet(spreadsheet: gspread.Spreadsheet, tab_name: str) -> gspread.Worksheet:
    """Return a worksheet by name, raise if not found."""
    try:
        return spreadsheet.worksheet(tab_name)
    except gspread.WorksheetNotFound:
        raise ValueError(f"Sheet tab '{tab_name}' not found. Available: {[ws.title for ws in spreadsheet.worksheets()]}")
