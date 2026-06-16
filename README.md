# FusionOCR Tester

A browser-based test bench for [FusionOCR](https://github.com/MxxScott/FusionOCR). Upload a
handwritten script image and FusionOCR transcribes it; the tester shows every line alongside
what each individual OCR engine (TrOCR, EasyOCR, Tesseract…) read, so you can see exactly
where the engines agreed or disagreed. You can edit and flag lines by hand, then run an AI
vision model to double-check each line against the original image — and export a clean,
corrected transcription as JSON.

It's built to run on **one host** and be used from **any device**: the OCR models and the
vision model live on the host and download only once; every other device just opens the URL
in a browser. Nothing heavy to install per device. If FusionOCR or the AI model aren't
present, it degrades gracefully to realistic mock data and a per-engine majority vote, so the
full workflow works end-to-end with zero setup.

## Run locally (Python)

```bash
pip install -r requirements.txt
uvicorn server:app --reload --port 8200
# open http://localhost:8200
```

## Run once, use from any device (Docker — recommended)


```bash
# 1. (optional) tell it where FusionOCR lives — enables real OCR
cp .env.example .env        # then edit FUSIONOCR_PATH

# 2. start the host (FusionOCR-Tester API + Ollama for AI verification)
docker compose up -d --build

# 3. one-time: pull the vision model into the Ollama volume
docker exec fusionocr-ollama ollama pull llava:7b

# 4. open it
#    on this machine:        http://localhost:8200
#    from another device:    http://<this-host-LAN-ip>:8200   (e.g. http://192.168.1.20:8200)
```

**Why this avoids the repeated download:** the OCR model caches (`HF_HOME`, `TORCH_HOME`,
`EASYOCR_MODULE_PATH`) and the Ollama models are mapped to named volumes (`model_cache`,
`ollama_models`). They're fetched on first use and reused forever after — even across
`docker compose up` rebuilds. Only this one host ever downloads them.

> Running the **full** FusionOCR ML stack in the container needs its Python deps added to the
> image — see the commented block in the `Dockerfile`. Until then the container runs the
> tester on mock data (plus real Tesseract, which is tiny), which is enough to exercise the
> whole UI.

### Reaching the host from outside your network

On the same Wi-Fi, the LAN IP above just works. To use it from anywhere (phone on mobile data,
another location), put a free tunnel in front of port 8200 — no router config needed:

| Tool | One-liner | Notes |
|---|---|---|
| **Tailscale** | install on host + device, then `http://<host-tailscale-ip>:8200` | private mesh VPN; best for personal use |
| **Cloudflare Tunnel** | `cloudflared tunnel --url http://localhost:8200` | gives a public `*.trycloudflare.com` URL |
| **ngrok** | `ngrok http 8200` | quick public URL; free tier rotates the address |

Treat any public tunnel as exposing the tool to the internet — add auth or keep it private
(Tailscale) if the scripts are sensitive.

## Configuration (env vars)

| Var | Default | Purpose |
|---|---|---|
| `FUSIONOCR_PATH` | `Z:\Code\Python\FusionOCR` | folder containing FusionOCR's `main.py` (with `run_ocr`) |
| `OLLAMA_BASE` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_VISION_MODEL` | `llava:7b` | vision model for AI verification (`ollama pull llava:7b`) |

## How verification works

- **Human review** — every line is editable; flag uncertain lines, add or delete lines.
  Confidence and flags come straight from FusionOCR.
- **AI verification** — "Run AI verification" sends the (possibly edited) lines plus the image
  to the server. Choose **Flagged only** (fast) or **All lines**. Each suggestion shows the
  original vs. the AI's reading; Accept applies it and clears the flag, Reject dismisses it.
  "Accept all" applies every suggestion at once. With no Ollama vision model available, it
  falls back to a **majority vote** across the per-engine reads.
- **Export** — downloads the corrected transcription as JSON (`*_verified.json`), ready to
  feed into a marking pipeline (e.g. Verdikt).

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

`engines` (per-line, `{engine: text}`) is what powers the comparison view. Missing fields are
tolerated: confidence defaults to 0.8, and lines under 0.65 are auto-flagged.

## Endpoints

| Method | Path | Body | Returns |
|---|---|---|---|
| `GET` | `/api/health` | – | OCR/AI availability |
| `POST` | `/api/ocr` | multipart `image` | normalized transcription |
| `POST` | `/api/verify` | multipart `payload` (JSON) + optional `image` | per-line suggestions |
