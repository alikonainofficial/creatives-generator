from __future__ import annotations

import os
import time

import fal_client

from logging_config import get_logger


class FalAiClient:
    def __init__(self, api_key: str | None = None):
        key = api_key or os.environ.get("FAL_KEY", "")
        if key:
            os.environ["FAL_KEY"] = key
        self._logger = get_logger(__name__, service="fal_ai")

    def face_swap(
        self,
        target_image_url: str,
        reference_image_url: str,
        model: str = "fal-ai/nano-banana-2/edit",
    ) -> str:
        """Submit identity swap to Fal.ai Nano Banana 2, block until complete, return output image URL.

        Uses fal_client.subscribe() which handles polling internally — no asyncio needed.
        """
        start = time.time()
        prompt = (
            "IMAGE ROLES:\n\n"
            "Image 0 = SCREENSHOT (Base Scene Authority)\n"
            "Image 1 = REFERENCE (Identity Authority)\n\n"
            "TASK:\n"
            "Perform a strict identity replacement on the main human subject in Image 0 "
            "using Image 1 as the complete identity authority.\n\n"
            "BASE SCENE RULES (Image 0):\n"
            "- Preserve the entire scene exactly\n"
            "- Preserve environment and objects\n"
            "- Preserve camera angle and framing\n"
            "- Preserve lighting and exposure\n"
            "- Preserve composition and crop\n"
            "- Preserve 9:16 composition\n"
            "- Preserve pose and gesture\n"
            "- Preserve head angle and orientation\n"
            "- Preserve eye direction and gaze\n"
            "- Preserve facial expression\n\n"
            "IDENTITY RULES (Image 1):\n"
            "The final subject must match Image 1 exactly in biological identity.\n\n"
            "Must match from Image 1:\n"
            "- Facial structure and proportions\n"
            "- Skin tone and undertone\n"
            "- Skin texture and pigmentation\n"
            "- Eyes (shape and color)\n"
            "- Nose and mouth shape\n"
            "- Hairline and hairstyle compatible with head shape\n"
            "- Overall head identity\n"
            "- Ethnic appearance cues\n\n"
            "BODY IDENTITY TRANSFER:\n"
            "All visible skin must match Image 1:\n\n"
            "- Face skin tone\n"
            "- Neck skin tone\n"
            "- Arms skin tone\n"
            "- Hands skin tone\n"
            "- Fingers skin tone\n"
            "- Ears skin tone\n\n"
            "No original skin tone from Image 0 may remain if it conflicts with Image 1.\n\n"
            "BODY STRUCTURE AUTHORITY:\n"
            "The biological body structure must match the gender of Image 1.\n\n"
            "Adjust the following if needed to match Image 1:\n"
            "- Shoulder width\n"
            "- Neck thickness\n"
            "- Arm thickness\n"
            "- Hand size\n"
            "- Chest structure\n"
            "- Torso proportions\n"
            "- Waist shape\n"
            "- Overall body proportions\n"
            "- Muscle mass distribution\n\n"
            "If Image 0 contains a body inconsistent with the gender of Image 1, you MUST "
            "modify the body structure to match the natural anatomy of Image 1.\n\n"
            "Example corrections:\n"
            "- If Image 0 shows a male torso but Image 1 is female → convert torso to "
            "natural female anatomy.\n"
            "- If Image 0 shows a female torso but Image 1 is male → convert torso to "
            "natural male anatomy.\n\n"
            "The final anatomy must look natural for the gender of Image 1.\n\n"
            "GENDER AUTHORITY:\n"
            "The final subject must fully match the biological gender of Image 1.\n\n"
            "Ignore gender cues from Image 0 if they conflict.\n\n"
            "This includes:\n"
            "- facial structure\n"
            "- body proportions\n"
            "- posture\n"
            "- clothing compatibility\n"
            "- anatomical features\n\n"
            "The final person must appear as a natural male or natural female consistent "
            "with Image 1.\n\n"
            "CLOTHING COMPATIBILITY:\n"
            "If the clothing in Image 0 is compatible with the gender of Image 1, "
            "preserve the clothing exactly.\n\n"
            "If the clothing strongly conflicts with the gender of Image 1, adapt the "
            "clothing style to match the gender of Image 1 while preserving:\n"
            "- color\n"
            "- material\n"
            "- lighting\n"
            "- overall visual style\n\n"
            "The clothing must look natural for the person from Image 1.\n\n"
            "ENVIRONMENT:\n"
            "Do NOT regenerate the scene.\n"
            "Do NOT change background.\n"
            "Do NOT move objects.\n"
            "Do NOT relight the scene.\n\n"
            "LIGHTING:\n"
            "Preserve:\n"
            "- Light direction\n"
            "- Light intensity\n"
            "- Color temperature\n"
            "- Shadows\n"
            "- Exposure\n\n"
            "Do not apply filters or beautification.\n\n"
            "TEXT REMOVAL:\n"
            "Remove all subtitles and all text.\n"
            "Do not recreate text.\n"
            "No ghosting.\n\n"
            "ANATOMY CONSISTENCY:\n"
            "Do not create mismatched anatomy such as:\n"
            "- female face on male torso\n"
            "- male face on female body\n"
            "- incorrect shoulder or chest structure\n\n"
            "The entire visible body must belong to the same person identity from Image 1.\n\n"
            "QUALITY REQUIREMENTS:\n"
            "- Strict photorealism\n"
            "- Natural anatomy\n"
            "- Clean edges\n"
            "- No halos\n"
            "- No ghosting\n"
            "- No morph blending\n"
            "- No double faces\n\n"
            "FINAL REQUIREMENT:\n"
            "The final image must look like the person from Image 1 was physically "
            "present in Image 0 when the screenshot was captured.\n\n"
            "OUTPUT:\n"
            "Return exactly one final edited image."
        )

        result = fal_client.subscribe(
            model,
            arguments={
                "prompt": prompt,
                "image_urls": [target_image_url, reference_image_url],
                "num_images": 1,
                "aspect_ratio": "9:16",
                "output_format": "png",
                "resolution": "1K",
                "safety_tolerance": 4,
            },
        )
        elapsed = time.time() - start
        self._logger.info(
            "Fal.ai face swap completed",
            extra={
                "target_image_url": target_image_url,
                "reference_image_url": reference_image_url,
                "model": model,
                "duration_seconds": round(elapsed, 2),
            },
        )
        return _extract_image_url(result)


def _extract_image_url(result: object) -> str:
    """Pull the output image URL out of a Fal.ai result dict."""
    if not isinstance(result, dict):
        raise RuntimeError(f"Unexpected Fal.ai response type: {type(result)}")

    if "image" in result:
        img = result["image"]
        return img["url"] if isinstance(img, dict) else str(img)

    if "images" in result:
        images = result["images"]
        if images:
            return images[0]["url"] if isinstance(images[0], dict) else str(images[0])

    for key in ("output", "result", "url"):
        if key in result:
            val = result[key]
            if isinstance(val, str):
                return val
            if isinstance(val, dict):
                return val.get("url", "")

    raise RuntimeError(f"Cannot find image URL in Fal.ai response: {result}")
