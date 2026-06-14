# FusionOCR Tester

A small web tool to test [FusionOCR](https://github.com/MxxScott/FusionOCR) transcriptions
with **per-engine comparison**, **human review**, and **AI vision verification**.

Upload a handwritten script image → FusionOCR transcribes it → you see each line
with what every OCR engine read, edit/flag anything wrong, optionally let an AI
vision model double-check against the image, then export a corrected JSON.

## Run

```bash
pip install -r requirements.txt
uvicorn server:app --reload --port 8200
# open http://localhost:8200
```

The server looks for FusionOCR and an Ollama vision model; both are optional.
Status is shown as two badges in the header.

| Capability | Real | Fallback |
|---|---|---|
| OCR | imports `run_ocr` from `FUSIONOCR_PATH` | realistic **mock** transcription |
| AI verify | Ollama vision model (`llava:7b`) checks each line vs the image | **engine vote** — majority across the per-engine reads |

So it runs end-to-end with nothing installed, then upgrades automatically once
FusionOCR / Ollama are present.

## Configuration (env vars)

| Var | Default | Purpose |
|---|---|---|
| `FUSIONOCR_PATH` | `Z:\Code\Python\FusionOCR` | folder containing FusionOCR's `main.py` (with `run_ocr`) |
| `OLLAMA_BASE` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_VISION_MODEL` | `llava:7b` | vision model for AI verification (`ollama pull llava:7b`) |

## How verification works

- **Human review** — every line is editable; flag uncertain lines, add or delete
  lines. Confidence and flags come straight from FusionOCR.
- **AI verification** — "Run AI verification" sends the (possibly edited) lines plus
  the image to the server. Choose **Flagged only** (fast) or **All lines**. Each
  suggestion shows the original vs. the AI's reading; Accept applies it and clears
  the flag, Reject dismisses it. "Accept all" applies every suggestion at once.
- **Export** — downloads the corrected transcription as JSON (`*_verified.json`),
  ready to feed into a marking pipeline (e.g. Verdikt).

## Expected `run_ocr` output

The server normalizes FusionOCR's result, but it works best when `run_ocr(image_path)`
returns something like:

```json
{
  "full_text": "line one\nline two",
  "overall_confidence": 0.78,
  "lines": [
    {
      "line": 1,
      "text": "fused best-guess text",
      "confidence": 0.74,
      "flagged": false,
      "engines": {
        "trocr_hw": "...", "trocr_pr": "...", "easyocr": "...", "tesseract": "..."
      }
    }
  ]
}
```

`engines` (per-line, `{engine: text}`) is what powers the comparison view. Missing
fields are tolerated: confidence defaults to 0.8, and lines under 0.65 are auto-flagged.

## Endpoints

| Method | Path | Body | Returns |
|---|---|---|---|
| `GET` | `/api/health` | – | OCR/AI availability |
| `POST` | `/api/ocr` | multipart `image` | normalized transcription |
| `POST` | `/api/verify` | multipart `payload` (JSON) + optional `image` | per-line suggestions |
