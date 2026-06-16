"""
FusionOCR Tester - server
=========================
A small FastAPI app to test FusionOCR transcriptions with both human review
and AI (vision-LLM) verification.

  Browser UI  ─►  POST /api/ocr      (run FusionOCR on an uploaded image)
              ─►  POST /api/verify   (AI check each line against the image)

It imports FusionOCR's `run_ocr` from FUSIONOCR_PATH if available; otherwise it
serves realistic MOCK data so you can drive the whole UI with no OCR installed.

Run:
    pip install -r requirements.txt
    uvicorn server:app --reload --port 8200
    # open http://localhost:8200

Env:
    FUSIONOCR_PATH        path to the FusionOCR project (default Z:\\Code\\Python\\FusionOCR)
    OLLAMA_BASE           Ollama URL for AI verification (default http://localhost:11434)
    OLLAMA_VISION_MODEL   vision model (default llava:7b)
"""

import os
import sys
import json
import base64
import logging
import tempfile
from pathlib import Path
from typing import Optional

import requests
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("fusionocr-tester")

HERE            = Path(__file__).parent
FUSIONOCR_PATH  = os.getenv("FUSIONOCR_PATH", r"Z:\Code\Python\FusionOCR")
OLLAMA_BASE     = os.getenv("OLLAMA_BASE", "http://localhost:11434")
OLLAMA_VISION   = os.getenv("OLLAMA_VISION_MODEL", "llava:7b")

# Engine ids FusionOCR fuses together (used for the per-engine comparison UI).
ENGINES = ["trocr_handwritten", "trocr_printed", "easyocr", "tesseract"]

app = FastAPI(title="FusionOCR Tester", version="1.0.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


# ─── Mock data ───────────────────────────────────────────────────────────────
# A realistic fused transcription with engine disagreement, so the UI is
# meaningful even without FusionOCR or a real image.

MOCK_LINES = [
    {
        "line": 1, "text": "Plants need sunlight water and nutrients to grow.",
        "confidence": 0.91, "flagged": False,
        "engines": {
            "trocr_handwritten": "Plants need sunlight water and nutrients to grow.",
            "trocr_printed": "Plants need sunlight water and nutrients to grow.",
            "easyocr":  "Plants need sunlight water and nutrients to grow.",
            "tesseract":"Plants need sunlight water and nutrlents to grow.",
        },
    },
    {
        "line": 2, "text": "Photosynthsis is when plants make food from sunlight.",
        "confidence": 0.74, "flagged": False,
        "engines": {
            "trocr_handwritten": "Photosynthsis is when plants make food from sunlight.",
            "trocr_printed": "Photosynthesis is when plants make food from sunlight.",
            "easyocr":  "Photosynthsis is when plants make food from sunlight.",
            "tesseract":"Photosynthsis is when plonts make food from sunlight.",
        },
    },
    {
        "line": 3, "text": "They use carbon dioxid and warter to make glucose.",
        "confidence": 0.61, "flagged": True,
        "engines": {
            "trocr_handwritten": "They use carbon dioxid and warter to make glucose.",
            "trocr_printed": "They use carbon dioxide and water to make glucose.",
            "easyocr":  "They use carbon dioxid and warter to make glucoze.",
            "tesseract":"They use corbon dioxid and worter to moke glucose.",
        },
    },
    {
        "line": 4, "text": "Oxygen is relesed as a biproduct.",
        "confidence": 0.68, "flagged": True,
        "engines": {
            "trocr_handwritten": "Oxygen is relesed as a biproduct.",
            "trocr_printed": "Oxygen is released as a by-product.",
            "easyocr":  "Oxygen is relesed as a biproduct.",
            "tesseract":"Oxygen is relesed as a blproduct.",
        },
    },
    {
        "line": 5, "text": "The green pigment in leaves is called chlorophyll.",
        "confidence": 0.88, "flagged": False,
        "engines": {
            "trocr_handwritten": "The green pigment in leaves is called chlorophyll.",
            "trocr_printed": "The green pigment in leaves is called chlorophyll.",
            "easyocr":  "The green pigment in leaves is called chlorophyll.",
            "tesseract":"The green pigment in leaves is called chlorophy11.",
        },
    },
    {
        "line": 6, "text": "It absorbs red and blue lite but reflects green.",
        "confidence": 0.72, "flagged": False,
        "engines": {
            "trocr_handwritten": "It absorbs red and blue lite but reflects green.",
            "trocr_printed": "It absorbs red and blue light but reflects green.",
            "easyocr":  "It absorbs red and blue lite but reflects green.",
            "tesseract":"It absorbs red and blue lite but reflects green.",
        },
    },
    {
        "line": 7, "text": "Without enough light plants cannot photosinthesize well.",
        "confidence": 0.83, "flagged": False,
        "engines": {
            "trocr_handwritten": "Without enough light plants cannot photosinthesize well.",
            "trocr_printed": "Without enough light plants cannot photosynthesize well.",
            "easyocr":  "Without enough light plants cannot photosinthesize well.",
            "tesseract":"Without enough light plants connot photosinthesize well.",
        },
    },
    {
        "line": 8, "text": "Plants also need minerals from the soil like nitragen.",
        "confidence": 0.65, "flagged": True,
        "engines": {
            "trocr_handwritten": "Plants also need minerals from the soil like nitragen.",
            "trocr_printed": "Plants also need minerals from the soil like nitrogen.",
            "easyocr":  "Plants also need minerals from the soil like nitragen.",
            "tesseract":"Plonts also need minerals from the soil like nitragen.",
        },
    },
]


def _mock_result(source: str = "mock") -> dict:
    lines = [json.loads(json.dumps(l)) for l in MOCK_LINES]  # deep copy
    return {
        "source": source,
        "mode": "mock",
        "engines_used": ENGINES,
        "overall_confidence": round(sum(l["confidence"] for l in lines) / len(lines), 3),
        "full_text": "\n".join(l["text"] for l in lines),
        "lines": lines,
    }


# ─── FusionOCR adapter ───────────────────────────────────────────────────────

def _run_fusionocr(image_path: str) -> Optional[dict]:
    """Call the external FusionOCR project, or return None if unavailable."""
    if not Path(FUSIONOCR_PATH).exists():
        log.warning("FusionOCR not found at %s - using mock data", FUSIONOCR_PATH)
        return None
    try:
        if FUSIONOCR_PATH not in sys.path:
            sys.path.insert(0, FUSIONOCR_PATH)
        from main import run_ocr  # provided by the FusionOCR project
        return run_ocr(image_path)
    except Exception as e:  # noqa: BLE001 - surface any import/run failure as mock
        log.exception("FusionOCR run failed (%s) - using mock data", e)
        return None


def _normalize(raw: dict, source: str) -> dict:
    """Coerce a FusionOCR result into the UI contract, tolerating shape varis."""
    lines_in = raw.get("lines", [])
    # flagged_lines may be a list of ints or of {"line": n} dicts — tolerate both
    flagged_set = set()
    for r in raw.get("flagged_lines", []):
        flagged_set.add(r.get("line") if isinstance(r, dict) else r)

    lines = []
    for i, l in enumerate(lines_in, start=1):
        num  = l.get("line", i)
        conf = l.get("confidence", l.get("score", 0.8))
        eng  = l.get("engines", {})
        # engines may be {name: text} or {name: {text, confidence}}
        engines = {}
        if isinstance(eng, dict):
            for k, v in eng.items():
                engines[k] = v.get("text", "") if isinstance(v, dict) else str(v)
        flagged = l.get("flagged", num in flagged_set or conf < 0.65)
        lines.append({
            "line": num,
            "text": l.get("text", ""),
            "confidence": round(float(conf), 3),
            "flagged": bool(flagged),
            "engines": engines,
        })

    engines_used = raw.get("engines_used")
    if not engines_used:
        seen = []
        for l in lines:
            for k in l["engines"]:
                if k not in seen:
                    seen.append(k)
        engines_used = seen or ENGINES

    overall = raw.get("overall_confidence")
    if overall is None and lines:
        overall = round(sum(l["confidence"] for l in lines) / len(lines), 3)

    return {
        "source": source,
        "mode": "fusionocr",
        "engines_used": engines_used,
        "overall_confidence": overall or 0.0,
        "full_text": raw.get("full_text") or "\n".join(l["text"] for l in lines),
        "lines": lines,
    }


# ─── AI verification ─────────────────────────────────────────────────────────

def _ollama_available() -> bool:
    try:
        r = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def _vision_prompt(text: str, line_num: int) -> str:
    return (
        "You are checking an OCR transcription of handwritten text. "
        f'The OCR says line {line_num} reads: "{text}". '
        "Look carefully at the handwriting in the image. Return ONLY a JSON object: "
        '{"accurate": true|false, "corrected_text": "<what it really says>", '
        '"issues": "<short note or none>", "confidence": "high|medium|low"}'
    )


def _parse_json(raw: str) -> Optional[dict]:
    raw = (raw or "").strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        # try to find the first {...}
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end != -1:
            try:
                return json.loads(raw[start:end + 1])
            except json.JSONDecodeError:
                return None
        return None


def _verify_line_vision(line: dict, image_b64: str) -> dict:
    prompt = _vision_prompt(line["text"], line["line"])
    try:
        r = requests.post(
            f"{OLLAMA_BASE}/api/generate",
            json={"model": OLLAMA_VISION, "prompt": prompt,
                  "images": [image_b64], "stream": False},
            timeout=120,
        )
        r.raise_for_status()
        data = _parse_json(r.json().get("response", ""))
    except Exception as e:  # noqa: BLE001
        log.warning("vision verify failed for line %s: %s", line["line"], e)
        data = None

    if not data:
        return _verify_line_engines(line, method="vision_failed")

    corrected = (data.get("corrected_text") or line["text"]).strip()
    return {
        "line": line["line"],
        "original_text": line["text"],
        "corrected_text": corrected,
        "accurate": bool(data.get("accurate", corrected.lower() == line["text"].lower())),
        "issues": data.get("issues", "none"),
        "confidence": data.get("confidence", "medium"),
        "method": "vision",
        "changed": corrected.lower() != line["text"].lower(),
    }


def _verify_line_engines(line: dict, method: str = "engine_vote") -> dict:
    """Fallback: majority vote across the per-engine reads."""
    texts = [t for t in line.get("engines", {}).values() if t]
    suggestion = line["text"]
    issues = "none"
    conf = "low"
    if texts:
        from collections import Counter
        winner, count = Counter(texts).most_common(1)[0]
        agree = count / len(texts)
        if winner != line["text"] and agree >= 0.5:
            suggestion = winner
            issues = f"{count}/{len(texts)} engines agree on a different reading"
            conf = "medium" if agree >= 0.75 else "low"
        elif len(set(texts)) == 1:
            conf = "high"
    changed = suggestion.lower() != line["text"].lower()
    return {
        "line": line["line"],
        "original_text": line["text"],
        "corrected_text": suggestion,
        "accurate": not changed,
        "issues": issues,
        "confidence": conf,
        "method": method,
        "changed": changed,
    }


# ─── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "fusionocr_present": Path(FUSIONOCR_PATH).exists(),
        "fusionocr_path": FUSIONOCR_PATH,
        "ollama_available": _ollama_available(),
        "vision_model": OLLAMA_VISION,
    }


@app.post("/api/ocr")
async def api_ocr(image: UploadFile = File(...)) -> dict:
    """Run FusionOCR on an uploaded image (or return mock data)."""
    suffix = Path(image.filename or "upload").suffix or ".png"
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(await image.read())
    try:
        raw = _run_fusionocr(path)
        if raw is None:
            return _mock_result(source=image.filename or "mock")
        return _normalize(raw, source=image.filename or path)
    finally:
        try: os.remove(path)
        except OSError: pass


@app.post("/api/verify")
async def api_verify(
    payload: str = Form(...),
    image: Optional[UploadFile] = File(None),
) -> dict:
    """AI-verify a (possibly human-edited) transcription against the image."""
    data = json.loads(payload)
    lines = data.get("lines", [])
    mode  = data.get("mode", "flagged")  # 'flagged' | 'all'

    image_b64 = None
    use_vision = False
    if image is not None and _ollama_available():
        image_b64 = base64.b64encode(await image.read()).decode()
        use_vision = True

    targets = [l for l in lines if l.get("flagged")] if mode == "flagged" else list(lines)

    verified = []
    for l in targets:
        if use_vision:
            verified.append(_verify_line_vision(l, image_b64))
        else:
            verified.append(_verify_line_engines(l))

    changed = [v for v in verified if v["changed"]]
    return {
        "method": "vision" if use_vision else "engine_vote",
        "mode": mode,
        "lines_checked": len(verified),
        "suggestions": changed,
        "verified": verified,
    }


# ─── Static UI ───────────────────────────────────────────────────────────────

@app.get("/")
def index() -> FileResponse:
    return FileResponse(HERE / "static" / "index.html")


@app.get("/favicon.ico")
def favicon() -> Response:
    return Response(status_code=204)
