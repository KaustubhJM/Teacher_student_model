# Website backend (FastAPI, server/app.py) for Render — see render.yaml.
# The website itself (web/) is deployed on Vercel, which forwards /api/* here
# (see vercel.json). The CLI (classroom.py) is not part of this image's job.
# Debian 13 pinned: the apt package names below are trixie's (glib is
# libglib2.0-0t64 there).
FROM python:3.11-slim-trixie

# libportaudio2: server.app -> explainer -> teacher_natural -> neural_tts /
#   voice_io import sounddevice at startup; without PortAudio the import
#   raises OSError and the server never starts (no audio device is needed).
# libgl1 + libglib2.0-0t64: OpenCV, used by RapidOCR for image-only PDF pages.
# libgomp1: OpenMP runtime for faiss / torch.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libportaudio2 libgl1 libglib2.0-0t64 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.hf_cache

# CPU-only PyTorch first, so sentence-transformers doesn't pull the default
# CUDA build (several GB of NVIDIA libraries that are useless here).
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch
COPY requirements.txt .
RUN pip install -r requirements.txt

# Bake the embedding model and the reranker into the image. rag/vector_index.py
# and rag/reranker.py load them with local_files_only=True first, so the
# server starts without downloading anything.
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('all-MiniLM-L6-v2'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512)"

COPY . .

# Everything that must survive a restart/redeploy lives on the Render disk
# mounted at /var/data (render.yaml): class records, uploaded sources and the
# rag caches. .rag_cache is a fixed relative path in the code, so it is
# linked onto the disk at start-up (the disk only exists at runtime).
ENV CLASSROOM_WEB_DB_PATH=/var/data/classes.db \
    CLASSROOM_WEB_UPLOAD_DIR=/var/data/uploads \
    OCR_CACHE_DIR=/var/data/ocr_cache \
    VISION_CACHE_DIR=/var/data/vision_cache

CMD ["sh", "-c", "mkdir -p /var/data/rag_cache && ln -sfn /var/data/rag_cache /app/.rag_cache && exec uvicorn server.app:app --host 0.0.0.0 --port ${PORT:-10000}"]
