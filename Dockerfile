FROM python:3.11-slim

# System deps. Tesseract is small and lets the tester run one real OCR engine
# even without the full FusionOCR stack. libgl/glib are needed by EasyOCR/OpenCV.
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# To run the FULL FusionOCR stack inside the container, also install its Python
# dependencies (this is the heavy ~6GB of wheels + model downloads at runtime):
#
#   COPY fusionocr-requirements.txt .
#   RUN pip install --no-cache-dir -r fusionocr-requirements.txt
#       # torch, transformers, easyocr, pytesseract, pillow, ...
#
# The model *weights* are not baked into the image — they download on first run
# into the /models volume (see docker-compose.yml), so they persist and are
# fetched only once per host.

COPY . .

EXPOSE 8200
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8200"]
