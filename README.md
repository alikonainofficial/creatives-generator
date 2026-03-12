# Video Cloner

Speechify-branded video creation pipeline — Python/Streamlit replacement for the n8n workflow.

## Setup

1. Copy `.env.example` to `.env` and fill in your API keys:
   ```
   cp .env.example .env
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. For Google Sheets batch mode, place your service account JSON at the path specified in `GOOGLE_SERVICE_ACCOUNT_JSON`.

## Running

```
streamlit run app.py
```

## Modes

### Single Job
Enter a Cloudinary video URL and a reference image URL. The pipeline runs step-by-step with live progress.

### Batch via Google Sheets
Provide a Google Sheet URL with a Jobs tab containing columns:
- `source_video_url`
- `reference_image_url`
- `status` (set to `queued` to process)

Results are written back to:
- `Jobs` tab: status, analysis_json, swapped_start_frame_url, final_script_json
- `Clips` tab: one row per dialogue clip

## Pipeline Steps

1. **Build Context** — Parse Cloudinary URL to extract cloud_name, video_public_id
2. **Video Analysis** — Gemini 1.5 Pro analyzes the video, returns structured JSON
3. **Anchor Frame** — Extract best face frame URL via Cloudinary's `so_T` parameter
4. **Face Swap** — Fal.ai async face swap with polling
5. **Upload to Cloudinary** — Store swapped image in your Cloudinary account
6. **Gender Detection** — Gemini Flash detects speaker gender from swapped image
7. **Script Rewrite** — Gemini Flash rewrites script for Speechify branding (30-45 words, 2-4 clips)
8. **Clip Timing** — Assign durations (~2.5 words/sec, clamped 3-7s per clip)

## Cache

- **Single job mode**: uses Streamlit session_state (in-memory per session)
- **Batch mode**: reads/writes a `VideoCache` sheet tab to avoid re-analyzing the same video
