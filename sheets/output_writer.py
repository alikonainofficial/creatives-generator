from __future__ import annotations

from datetime import datetime

import gspread


def append_output_row(
    output_worksheet: gspread.Worksheet,
    job_id: str,
    input_link: str,
    creative_link: str,
    pipeline_type: str,
) -> None:
    """Append a single result row to the output tracking sheet.

    Column matching is header-driven and case-insensitive so the physical
    column order in the sheet does not matter. The expected columns are:
    Date, Job ID, Input Link, Creative Link, Pipeline Type.
    """
    headers = output_worksheet.row_values(1)
    if not headers:
        return

    header_map: dict[str, int] = {
        h.strip().lower(): idx
        for idx, h in enumerate(headers, start=1)
        if h.strip()
    }

    row_values: list[str] = [""] * len(headers)

    field_map: dict[str, str] = {
        "date": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "job id": job_id,
        "input link": input_link,
        "creative link": creative_link,
        "pipeline type": pipeline_type,
    }

    for col_name, value in field_map.items():
        col_idx = header_map.get(col_name)
        if col_idx is not None:
            row_values[col_idx - 1] = value

    output_worksheet.append_row(row_values, value_input_option="USER_ENTERED")
