from __future__ import annotations

"""Video analysis cache helpers.

- Single-job mode: primarily uses an in-session dict stored in st.session_state,
  augmented with a simple disk-backed cache so work survives server restarts.
- Batch mode: can optionally persist cache entries to a Google Sheet tab.
"""

from pathlib import Path
from typing import Any


_DISK_CACHE_PATH = Path(__file__).parent / "video_cache.json"


def _load_rewritten_script_from_json(script_json: str):
    """Parse a script JSON blob into a RewrittenScript instance.

    Supports both legacy RewrittenScript schema:
        {"segments": [...], "total_word_count": int}
    and the n8n-compatible schema:
        {"script": [{"speaker": "main", "text": str, "insert_demo": bool, ...}, ...]}
    """
    import json

    from pipeline.script_writer import RewrittenScript, ScriptSegment

    try:
        data = json.loads(script_json)
    except Exception:
        return None

    # Primary path: n8n-style schema with top-level "script" array of segments.
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

    # Fallback: assume the JSON already matches the RewrittenScript schema.
    try:
        return RewrittenScript.model_validate(data)
    except Exception:
        return None


def _script_to_n8n_json(script_obj: Any) -> str:
    """Serialize a RewrittenScript-style object into the n8n-compatible schema.

    Output shape:
        {"script": [{"speaker": "main", "text": str, "insert_demo": bool, ...}, ...]}
    """
    import json

    # Best-effort: if the object exposes .segments, use that; otherwise fall back
    # to its own JSON representation.
    segments = getattr(script_obj, "segments", None)
    if segments is None:
        try:
            return script_obj.model_dump_json()
        except Exception:
            return ""

    script_entries: list[dict[str, Any]] = []
    for seg in segments:
        dialogue = str(getattr(seg, "dialogue", "")).strip()
        if not dialogue:
            continue

        insert_demo = bool(getattr(seg, "is_demo", False))
        demo_type = getattr(seg, "demo_type", None) if insert_demo else None
        mode = getattr(seg, "mode", None) if insert_demo else None

        script_entries.append(
            {
                "speaker": "main",
                "text": dialogue,
                "insert_demo": insert_demo,
                "demo_type": demo_type,
                "mode": mode,
            }
        )

    return json.dumps({"script": script_entries}, ensure_ascii=False)


def get_session_cache() -> dict[str, Any]:
    """Return or initialize the in-session cache dict."""
    try:
        import streamlit as st
        if "video_cache" not in st.session_state:
            st.session_state["video_cache"] = {}
        return st.session_state["video_cache"]
    except Exception:
        # Fallback for non-Streamlit contexts
        return {}


def load_disk_cache() -> dict[str, Any]:
    """Load cache entries from the local JSON cache file (if present).

    This is used to allow the pipeline to resume work across Streamlit server
    restarts in single-job mode.
    """
    import json

    if not _DISK_CACHE_PATH.exists():
        return {}

    try:
        raw = json.loads(_DISK_CACHE_PATH.read_text())
    except Exception:
        return {}

    cache: dict[str, Any] = {}

    for video_key, row in raw.items():
        entry: dict[str, Any] = {}

        analysis_json = str(row.get("analysis_json", "")).strip()
        if analysis_json:
            from pipeline.video_analysis import VideoAnalysisResult

            try:
                entry["analysis"] = VideoAnalysisResult.model_validate_json(analysis_json)
            except Exception:
                pass

        script_json = str(row.get("final_script_json", "")).strip()
        if script_json:
            script = _load_rewritten_script_from_json(script_json)
            if script is not None:
                entry["script"] = script

        anchor_url = str(row.get("anchor_frame_url", "")).strip()
        if anchor_url:
            entry["anchor_frame_url"] = anchor_url

        swapped_url = str(row.get("swapped_image_url", "")).strip()
        if swapped_url:
            entry["swapped_image_url"] = swapped_url

        gender = str(row.get("gender", "")).strip()
        if gender:
            entry["gender"] = gender

        if entry:
            cache[video_key] = entry

    return cache


def save_disk_cache(cache: dict[str, Any]) -> None:
    """Persist the in-memory cache dict to a local JSON file.

    We only serialize the fields needed to resume work:
    - analysis (as JSON)
    - script (as JSON)
    - anchor_frame_url
    - swapped_image_url
    - gender
    """
    import json

    serializable: dict[str, dict[str, Any]] = {}

    for video_key, entry in cache.items():
        if not isinstance(entry, dict):
            continue

        row: dict[str, Any] = {}

        if "analysis" in entry:
            try:
                row["analysis_json"] = entry["analysis"].model_dump_json()
            except Exception:
                pass

        if "script" in entry:
            try:
                row["final_script_json"] = _script_to_n8n_json(entry["script"])
            except Exception:
                pass

        if "anchor_frame_url" in entry:
            row["anchor_frame_url"] = entry["anchor_frame_url"]

        if "swapped_image_url" in entry:
            row["swapped_image_url"] = entry["swapped_image_url"]

        if "gender" in entry:
            row["gender"] = entry["gender"]

        if row:
            serializable[video_key] = row

    try:
        _DISK_CACHE_PATH.write_text(json.dumps(serializable, indent=2))
    except Exception:
        # Disk persistence is best-effort; ignore failures.
        pass


def load_sheet_cache(video_cache_worksheet) -> dict[str, Any]:
    """Load cache entries from a VideoCache Google Sheet tab."""
    import json

    cache: dict[str, Any] = {}

    try:
        records = video_cache_worksheet.get_all_records()
        for row in records:
            video_key = str(row.get("video_key", "")).strip()
            if not video_key:
                continue

            entry: dict[str, Any] = {}

            analysis_json = str(row.get("analysis_json", "")).strip()
            if analysis_json:
                from pipeline.video_analysis import VideoAnalysisResult
                try:
                    entry["analysis"] = VideoAnalysisResult.model_validate_json(analysis_json)
                except Exception:
                    pass

            script_json = str(row.get("final_script_json", "")).strip()
            if script_json:
                script = _load_rewritten_script_from_json(script_json)
                if script is not None:
                    entry["script"] = script

            # Historically this column was named "anchor_frame_url". In the current
            # Sheet layout it's stored as "extracted_frame_url". Support both so the
            # cache logic does not depend on column order or legacy naming.
            anchor_url_raw = row.get("extracted_frame_url", "") or row.get("anchor_frame_url", "")
            anchor_url = str(anchor_url_raw).strip()
            if anchor_url:
                entry["anchor_frame_url"] = anchor_url

            if entry:
                cache[video_key] = entry
    except Exception:
        pass

    return cache


def save_to_sheet_cache(video_cache_worksheet, video_key: str, entry: dict[str, Any]) -> None:
    """Save a cache entry to the VideoCache sheet tab."""
    import json

    # Build serializable values from the in-memory cache entry.
    analysis_json = ""
    if "analysis" in entry:
        try:
            analysis_json = entry["analysis"].model_dump_json()
        except Exception:
            pass

    script_json = ""
    if "script" in entry:
        try:
            script_json = _script_to_n8n_json(entry["script"])
        except Exception:
            pass

    anchor_url = entry.get("anchor_frame_url", "")

    try:
        # Read header row once so we can map by column name instead of position.
        headers = video_cache_worksheet.row_values(1)
        header_to_col: dict[str, int] = {
            h.strip(): idx for idx, h in enumerate(headers, start=1) if h.strip()
        }

        # Helper to resolve the canonical column name for the extracted frame.
        def _frame_header_name() -> str | None:
            if "extracted_frame_url" in header_to_col:
                return "extracted_frame_url"
            if "anchor_frame_url" in header_to_col:
                return "anchor_frame_url"
            return None

        # Prepare field values we know how to persist.
        field_values: dict[str, Any] = {}
        if analysis_json:
            field_values["analysis_json"] = analysis_json
        if script_json:
            field_values["final_script_json"] = script_json

        frame_header = _frame_header_name()
        if frame_header and anchor_url:
            field_values[frame_header] = anchor_url

        # Optional metadata: source_video_url, face_time_start/end, status.
        source_url = entry.get("source_video_url")
        if source_url:
            field_values["source_video_url"] = source_url

        # Face window times (if present on the cache entry).
        face_time_start = entry.get("face_time_start")
        face_time_end = entry.get("face_time_end")
        if face_time_start is not None:
            field_values["face_time_start"] = face_time_start
        if face_time_end is not None:
            field_values["face_time_end"] = face_time_end

        status = entry.get("status")
        if status:
            field_values["status"] = status

        # Locate existing row for this video_key, if any.
        target_row: int | None = None
        records = video_cache_worksheet.get_all_records()
        for i, row in enumerate(records, start=2):
            if str(row.get("video_key", "")).strip() == video_key:
                target_row = i
                break

        if target_row is not None:
            # Update existing row cells by header name.
            for header_name, value in field_values.items():
                col_idx = header_to_col.get(header_name)
                if col_idx is not None:
                    video_cache_worksheet.update_cell(target_row, col_idx, value)
        else:
            # Append a new row, aligning values to the header order.
            row_values = ["" for _ in headers]

            # Always write the video_key if that column exists.
            vk_col = header_to_col.get("video_key")
            if vk_col is not None:
                row_values[vk_col - 1] = video_key

            for header_name, value in field_values.items():
                col_idx = header_to_col.get(header_name)
                if col_idx is not None:
                    row_values[col_idx - 1] = value

            video_cache_worksheet.append_row(row_values)
    except Exception:
        # Sheet persistence is best-effort; ignore failures.
        pass
