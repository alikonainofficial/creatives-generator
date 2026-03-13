from __future__ import annotations

import io
import os
import time

import httpx
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload

from logging_config import get_logger


class GoogleDriveService:
    """Upload files to Google Drive using a service account."""

    SCOPES = ["https://www.googleapis.com/auth/drive"]

    def __init__(self, service_account_file: str | None = None):
        path = service_account_file or os.environ.get(
            "GOOGLE_SERVICE_ACCOUNT_JSON", "./credentials/service_account.json"
        )
        creds = Credentials.from_service_account_file(path, scopes=self.SCOPES)
        self._service = build("drive", "v3", credentials=creds, cache_discovery=False)
        self._logger = get_logger(__name__, service="google_drive")

    def upload_video_from_url(
        self,
        video_url: str,
        filename: str,
        folder_id: str,
        mime_type: str = "video/mp4",
    ) -> dict:
        """Download a video from a URL and upload it to a Google Drive folder.

        Returns the Drive API file resource with id, webViewLink, webContentLink.
        """
        start = time.time()
        self._logger.info("Downloading video for Drive upload", extra={"video_url": video_url})

        with httpx.Client(follow_redirects=True, timeout=300.0, transport=httpx.HTTPTransport(retries=3)) as client:
            resp = client.get(video_url)
            resp.raise_for_status()
            content = resp.content

        self._logger.info(
            "Video downloaded, uploading to Drive",
            extra={"size_bytes": len(content), "file_name": filename},
        )

        file_metadata = {"name": filename, "parents": [folder_id]}
        media = MediaIoBaseUpload(io.BytesIO(content), mimetype=mime_type, resumable=True)

        file = (
            self._service.files()
            .create(
                body=file_metadata,
                media_body=media,
                fields="id,webViewLink,webContentLink",
                supportsAllDrives=True,
            )
            .execute()
        )

        elapsed = time.time() - start
        self._logger.info(
            "Uploaded video to Google Drive",
            extra={
                "file_name": filename,
                "folder_id": folder_id,
                "file_id": file.get("id"),
                "duration_seconds": round(elapsed, 2),
            },
        )
        return file
