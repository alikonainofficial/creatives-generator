from __future__ import annotations

import os
from pathlib import Path

import gspread


def get_gspread_client() -> gspread.Client:
    """Return an authenticated gspread client using service account credentials."""
    sa_json_path = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON", "./credentials/service_account.json")

    path = Path(sa_json_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Service account JSON not found at: {sa_json_path}\n"
            "Set GOOGLE_SERVICE_ACCOUNT_JSON in your .env file."
        )

    return gspread.service_account(filename=str(path))
