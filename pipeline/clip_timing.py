from __future__ import annotations

from dataclasses import dataclass

from pipeline.script_writer import RewrittenScript, ScriptSegment


WORDS_PER_SECOND = 2.5
MIN_CLIP_DURATION = 3.0
MAX_CLIP_DURATION = 7.0


@dataclass
class TimedClip:
    segment_id: int
    dialogue: str
    is_demo: bool
    duration_seconds: float
    word_count: int


def assign_clip_durations(script: RewrittenScript) -> list[TimedClip]:
    """Assign durations to each segment based on word count."""
    clips: list[TimedClip] = []

    for seg in script.segments:
        word_count = len(seg.dialogue.split())
        raw_duration = word_count / WORDS_PER_SECOND
        duration = max(MIN_CLIP_DURATION, min(MAX_CLIP_DURATION, raw_duration))

        clips.append(
            TimedClip(
                segment_id=seg.segment_id,
                dialogue=seg.dialogue,
                is_demo=seg.is_demo,
                duration_seconds=round(duration, 1),
                word_count=word_count,
            )
        )

    return clips
