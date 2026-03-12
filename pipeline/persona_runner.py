from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from logging_config import get_logger
from pipeline.clip_timing import TimedClip, assign_clip_durations
from pipeline.json_utils import extract_json
from pipeline.runner import StepResult
from pipeline.script_writer import RewrittenScript, ScriptSegment
from pipeline.video_blueprint import generate_blueprint_from_persona
from services.gemini import GeminiClient


PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "persona_script.txt"

ProgressCallback = Callable[[str, str], None]  # (step_name, message)


@dataclass
class PersonaJobInput:
    job_key: str
    persona: str
    pain_point: str
    speechify_solution: str
    hook_sample: str
    hook_emotion: str
    persona_image_url: str


@dataclass
class PersonaPipelineResult:
    job_input: PersonaJobInput | None = None
    gender: str | None = None
    scene_description: str | None = None
    script: RewrittenScript | None = None
    clips: list[TimedClip] = field(default_factory=list)
    blueprint: dict | None = None
    steps: list[StepResult] = field(default_factory=list)
    success: bool = False
    error: str | None = None

    @property
    def total_duration(self) -> float:
        return sum(s.duration_seconds for s in self.steps)


def _run_step(
    name: str,
    logger,
    fn: Callable,
    progress_cb: ProgressCallback | None,
) -> tuple[Any, StepResult]:
    logger.info("Starting persona pipeline step", extra={"step": name})
    if progress_cb:
        progress_cb(name, f"Starting {name}...")

    start = time.time()
    try:
        result = fn()
        elapsed = time.time() - start
        step = StepResult(name=name, success=True, duration_seconds=round(elapsed, 2), output=result)
        logger.info(
            "Completed persona pipeline step",
            extra={"step": name, "duration_seconds": round(elapsed, 2)},
        )
        if progress_cb:
            progress_cb(name, f"✓ {name} completed in {elapsed:.1f}s")
        return result, step
    except Exception as e:
        elapsed = time.time() - start
        step = StepResult(name=name, success=False, duration_seconds=round(elapsed, 2), error=str(e))
        logger.exception(
            "Persona pipeline step failed",
            extra={"step": name, "duration_seconds": round(elapsed, 2)},
        )
        if progress_cb:
            progress_cb(name, f"✗ {name} failed: {e}")
        raise


def _rewrite_persona_script(
    job: PersonaJobInput,
    gender: str,
    scene_description: str,
    gemini: GeminiClient,
) -> RewrittenScript:
    """Call Gemini to write a Speechify script from persona context."""
    prompt_template = PROMPT_PATH.read_text()

    persona_json = json.dumps(
        {
            "name": job.persona.split("\n")[0].strip(),
            "description": job.persona,
            "pain_point": job.pain_point,
            "speechify_solution": job.speechify_solution,
        },
        ensure_ascii=False,
    )

    prompt = (
        prompt_template
        .replace("{{ persona_json }}", persona_json)
        .replace("{{ hook_sample }}", job.hook_sample)
        .replace("{{ hook_emotion }}", job.hook_emotion)
        .replace("{{ scene_description }}", scene_description or "No scene description available.")
        .replace("{{ gender }}", gender)
    )

    raw = gemini.generate_text(prompt)
    json_text = extract_json(raw)

    try:
        data: dict[str, Any] = json.loads(json_text)
    except json.JSONDecodeError as e:
        preview = json_text[:200].replace("\n", "\\n")
        raise RuntimeError(
            "Persona script JSON parsing failed. "
            f"Error: {e}. "
            f"Preview: '{preview}'."
        ) from e

    if isinstance(data, dict) and "script" in data and isinstance(data["script"], list):
        segments: list[ScriptSegment] = []
        for idx, seg in enumerate(data["script"], start=1):
            if not isinstance(seg, dict):
                continue
            text = str(seg.get("text", "")).strip()
            if not text:
                continue
            is_demo = bool(seg.get("insert_demo", False))
            demo_type = seg.get("demo_type")
            mode = seg.get("mode")
            segments.append(
                ScriptSegment(
                    segment_id=idx,
                    dialogue=text,
                    is_demo=is_demo,
                    demo_type=demo_type,
                    mode=mode,
                )
            )
        total_word_count = sum(len(s.dialogue.split()) for s in segments)
        return RewrittenScript(segments=segments, total_word_count=total_word_count)

    try:
        return RewrittenScript.model_validate(data)
    except Exception as e:
        raise RuntimeError(
            f"Persona script validation failed: {e}. "
            f"Top-level keys: {list(data.keys()) if isinstance(data, dict) else 'n/a'}."
        ) from e


def run_persona_pipeline(
    job: PersonaJobInput,
    progress_cb: ProgressCallback | None = None,
) -> PersonaPipelineResult:
    """Run the persona-based pipeline for a single persona row.

    Steps:
      1. Analyze persona_image → gender + scene_description
      2. Rewrite script using persona context
      3. Assign clip durations
      4. Generate video blueprint from persona context → written to prompt_template

    No video analysis, face swap, or Cloudinary involved.
    No caching — each run is fresh.
    """
    result = PersonaPipelineResult(job_input=job)

    run_id = str(uuid.uuid4())
    logger = get_logger(
        __name__,
        run_id=run_id,
        job_key=job.job_key,
        persona_image_url=job.persona_image_url,
    )

    logger.info("Starting persona pipeline", extra={"job_key": job.job_key})
    gemini = GeminiClient()

    # Step 1: Analyze persona image (gender + scene description)
    try:
        image_data, step = _run_step(
            "Image Analysis",
            logger,
            lambda: gemini.analyze_persona_image(job.persona_image_url),
            progress_cb,
        )
        result.gender = image_data["gender"]
        result.scene_description = image_data["scene_description"]
        result.steps.append(step)
    except Exception as e:
        result.error = f"[Image Analysis] {type(e).__name__}: {e}"
        return result

    # Step 2: Rewrite script using persona context
    try:
        script, step = _run_step(
            "Script Generation",
            logger,
            lambda: _rewrite_persona_script(
                job,
                result.gender or "unknown",
                result.scene_description or "",
                gemini,
            ),
            progress_cb,
        )
        result.script = script
        result.steps.append(step)
    except Exception as e:
        result.error = f"[Script Generation] {type(e).__name__}: {e}"
        return result

    # Step 3: Assign clip durations
    try:
        clips, step = _run_step(
            "Clip Timing",
            logger,
            lambda: assign_clip_durations(script),
            progress_cb,
        )
        result.clips = clips
        result.steps.append(step)
    except Exception as e:
        result.error = f"[Clip Timing] {type(e).__name__}: {e}"
        return result

    # Step 4: Generate video blueprint from persona context so the Producer
    # always has a valid prompt_template and never hits the no-analysis_json path.
    try:
        blueprint, step = _run_step(
            "Blueprint Generation",
            logger,
            lambda: generate_blueprint_from_persona(
                persona=job.persona,
                pain_point=job.pain_point,
                speechify_solution=job.speechify_solution,
                hook_emotion=job.hook_emotion,
                scene_description=result.scene_description or "",
                gender=result.gender or "unknown",
                clip_count=len(result.clips),
                gemini=gemini,
            ),
            progress_cb,
        )
        result.blueprint = blueprint
        result.steps.append(step)
    except Exception as e:
        result.error = f"[Blueprint Generation] {type(e).__name__}: {e}"
        return result

    result.success = True
    logger.info(
        "Persona pipeline completed successfully",
        extra={
            "job_key": job.job_key,
            "total_duration_seconds": result.total_duration,
            "clips_count": len(result.clips),
        },
    )
    return result
