# Running the tester with real FusionOCR (Windows)

Run these in **PowerShell** on your machine. The tester process runs FusionOCR
in-process, so FusionOCR's dependencies must be installed in the *same* venv as
the tester.

## 0. Remove the broken clone stub
A partial `.git` got left here by an earlier clone attempt. Delete it first:
```powershell
Remove-Item -Recurse -Force C:\Users\lawizi\Documents\GitHub\FusionOCR
```

## 1. Clone FusionOCR (next to the tester)
```powershell
cd C:\Users\lawizi\Documents\GitHub
git clone https://github.com/MxxScott/FusionOCR.git
```

## 2. Install Tesseract system-wide (one-time)
Windows build: https://github.com/UB-Mannheim/tesseract/wiki — install and make
sure `tesseract.exe` is on your PATH.

## 3. One venv with BOTH tester + FusionOCR deps
```powershell
cd C:\Users\lawizi\Documents\GitHub\FusionOCR-Tester
.\.venv\Scripts\Activate.ps1            # the venv already here
pip install -r requirements.txt
pip install -r ..\FusionOCR\requirements.txt   # heavy: torch, transformers, easyocr, ...
```
GPU: if you have an NVIDIA card, install the CUDA build of torch for speed
(https://pytorch.org/get-started/locally/); CPU works but is slow.

## 4. Point the tester at FusionOCR and run
```powershell
$env:FUSIONOCR_PATH = "C:\Users\lawizi\Documents\GitHub\FusionOCR"
uvicorn server:app --port 8200          # omit --reload for OCR runs (avoids reloading heavy models)
```
Open http://localhost:8200, upload a script image, click **Run FusionOCR**.

- **First run downloads ~4.6GB of TrOCR models** to `C:\Users\lawizi\FusionOCR-models`
  (this is the C-drive space you freed up). Subsequent runs are fast.
- Check it's wired up: http://localhost:8200/api/health → `"fusionocr_present": true`.

## 5. (Optional) AI verification
```powershell
# requires Ollama installed + running
ollama pull llava:7b
```
Without it, "Run AI verification" falls back to per-engine majority vote.

## Troubleshooting
- `fusionocr_present: false` → `FUSIONOCR_PATH` is wrong, or the folder lacks `main.py`.
- Import errors on Run → FusionOCR's deps aren't in this venv (re-do step 3).
- Tesseract not found → not on PATH (step 2).
- OOM on GPU → lower `TROCR_BATCH` in FusionOCR's `main.py`.
