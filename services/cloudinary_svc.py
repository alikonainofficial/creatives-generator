from __future__ import annotations

import os

import cloudinary
import cloudinary.uploader
import httpx


class CloudinaryService:
    def __init__(self):
        cloudinary.config(
            cloud_name=os.environ.get("CLOUDINARY_CLOUD_NAME", ""),
            api_key=os.environ.get("CLOUDINARY_API_KEY", ""),
            api_secret=os.environ.get("CLOUDINARY_API_SECRET", ""),
            secure=True,
        )

    def build_frame_url(self, cloud_name: str, video_public_id: str, timestamp: float) -> str:
        """Build a Cloudinary frame extraction URL at a specific timestamp.

        Uses /video/upload/ with the so_ (start offset) transformation, which tells
        Cloudinary to extract that frame and serve it as a JPEG.
        """
        ts_str = f"{timestamp:.1f}"
        url = (
            f"https://res.cloudinary.com/{cloud_name}/video/upload"
            f"/so_{ts_str}/{video_public_id}.jpg"
        )
        return url

    def upload_image(self, image_url: str, public_id: str | None = None) -> str:
        """Upload an image from a URL to Cloudinary and return the secure URL."""
        kwargs: dict = {"resource_type": "image"}
        if public_id:
            kwargs["public_id"] = public_id

        result = cloudinary.uploader.upload(image_url, **kwargs)
        return result["secure_url"]

    def upload_video(
        self,
        video_url: str,
        folder: str = "",
        public_id: str | None = None,
    ) -> str:
        """Upload a video from a URL to Cloudinary and return the secure URL.

        Uses the unsigned upload preset pattern from the n8n pipeline:
        folder=ai-generated/clips/{job_key}, public_id={job_key}_clip_{index}.
        """
        kwargs: dict = {"resource_type": "video"}
        if folder:
            kwargs["folder"] = folder
        if public_id:
            kwargs["public_id"] = public_id

        result = cloudinary.uploader.upload(video_url, **kwargs)
        return result["secure_url"]
