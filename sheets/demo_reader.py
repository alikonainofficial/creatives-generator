from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gspread


@dataclass
class DemoRow:
    demo_id: str
    demo_type: str
    video_url: str
    raw_row: dict[str, Any]


def read_demo_rows(worksheet: gspread.Worksheet) -> list[DemoRow]:
    """Read all rows from the App Demo worksheet.

    Expected columns (case-insensitive): demo_id, demo_type, video_url.
    Rows missing demo_id or video_url are skipped.
    """
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

    rows: list[DemoRow] = []
    for row_values in all_values[1:]:
        record: dict[str, Any] = {}
        for idx, key in enumerate(header_keys):
            if key is None:
                continue
            record[key] = row_values[idx] if idx < len(row_values) else ""

        demo_id = str(record.get("demo_id", "")).strip()
        video_url = str(record.get("video_url", "")).strip()
        if not demo_id or not video_url:
            continue

        rows.append(
            DemoRow(
                demo_id=demo_id,
                demo_type=str(record.get("demo_type", "")).strip(),
                video_url=video_url,
                raw_row=record,
            )
        )

    return rows
