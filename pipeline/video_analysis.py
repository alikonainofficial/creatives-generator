from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from pipeline.json_utils import extract_json
from services.gemini import GeminiClient


PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "video_analysis.txt"


# These models are shaped to match the JSON schema defined in
# prompts/video_analysis.txt. Field names intentionally mirror the prompt keys
# (e.g. time_start_s) so we can validate the Gemini response directly.


class TimeWindow(BaseModel):
    time_start_s: float
    time_end_s: float


class FaceVisibility(BaseModel):
    is_clearly_visible: bool
    best_window: TimeWindow | None = None
    reason_if_not_visible: str | None = None


class RollStructure(BaseModel):
    contains_a_roll: bool
    contains_b_roll: bool
    dominant_roll_type: str
    estimated_a_roll_percentage: float


class OnScreenText(BaseModel):
    time_start_s: float
    time_end_s: float
    text: str
    position: str
    text_type: str


class DialogueLine(BaseModel):
    speaker: str = "main"
    time_start_s: float
    time_end_s: float
    text: str


class MainSubject(BaseModel):
    role: str
    appearance_non_sensitive: str
    clothing: str
    notable_objects: str


class VideoSummary(BaseModel):
    format_name: str
    setting_environment: str
    camera_framing: str
    lighting: str
    subject_count: int
    main_subject: MainSubject
    actions_movements: list[str]
    editing_style: str
    music_sfx: str


class VideoAnalysisResult(BaseModel):
    video_summary: VideoSummary
    roll_structure: RollStructure
    on_screen_text: list[OnScreenText] = Field(default_factory=list)
    spoken_dialogue: list[DialogueLine] = Field(default_factory=list)
    face_visibility: FaceVisibility


def analyze_video(video_url: str, gemini: GeminiClient) -> VideoAnalysisResult:
    """Call Gemini to analyze the video and return structured results."""
    prompt = PROMPT_PATH.read_text()

    raw = gemini.analyze_video(video_url, prompt)
    data: dict[str, Any] = json.loads(extract_json(raw))
    return VideoAnalysisResult.model_validate(data)


def compute_anchor_frame_timestamp(analysis: VideoAnalysisResult) -> float:
    """Return the midpoint timestamp from the best face visibility window.

    If no suitable window exists, fall back to 0.0 seconds.
    """
    window = analysis.face_visibility.best_window
    if window is None:
        return 0.0
    return (window.time_start_s + window.time_end_s) / 2.0
