from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import urlparse


@dataclass
class JobContext:
    source_video_url: str
    reference_image_url: str
    cloud_name: str
    video_public_id: str
    video_key: str
    job_key: str


def build_context(source_video_url: str, reference_image_url: str) -> JobContext:
    """Parse a Cloudinary video URL to extract cloud_name and video_public_id."""
    parsed = urlparse(source_video_url)

    # Expected: https://res.cloudinary.com/{cloud_name}/video/upload/.../{public_id}.{ext}
    path_parts = parsed.path.lstrip("/").split("/")

    if len(path_parts) < 4 or path_parts[1] != "video":
        raise ValueError(
            f"Cannot parse Cloudinary video URL: {source_video_url}\n"
            "Expected format: https://res.cloudinary.com/{{cloud_name}}/video/upload/.../{{public_id}}.ext"
        )

    cloud_name = path_parts[0]

    # Everything after "upload/" (skipping version segment if present) up to the last part
    upload_idx = path_parts.index("upload")
    remaining = path_parts[upload_idx + 1 :]

    # Strip version segment (v followed by digits)
    if remaining and re.match(r"^v\d+$", remaining[0]):
        remaining = remaining[1:]

    # Join remaining parts, strip file extension from last part
    if remaining:
        last = remaining[-1]
        last_no_ext = last.rsplit(".", 1)[0]
        remaining[-1] = last_no_ext
        video_public_id = "/".join(remaining)
    else:
        video_public_id = parsed.path.rsplit("/", 1)[-1].rsplit(".", 1)[0]

    video_key = f"{cloud_name}/{video_public_id}"

    # Derive a job key from the video public_id (last path component)
    job_key = video_public_id.rsplit("/", 1)[-1]

    return JobContext(
        source_video_url=source_video_url,
        reference_image_url=reference_image_url,
        cloud_name=cloud_name,
        video_public_id=video_public_id,
        video_key=video_key,
        job_key=job_key,
    )
