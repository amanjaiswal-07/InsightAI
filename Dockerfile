# Single-container image: FastAPI backend + built React UI on one port.
# Used for the Hugging Face Spaces demo, but works with any Docker host.
#
#   docker build -t insightai .
#   docker run -p 7860:7860 -e GROQ_API_KEY=... insightai

# ── 1. Build the React frontend ──────────────────────────────────────────────
FROM node:22-slim AS frontend
WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# ── 2. Python runtime ────────────────────────────────────────────────────────
FROM python:3.12-slim

# Hugging Face Spaces runs containers as uid 1000
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    HF_HOME=/home/user/.cache/huggingface \
    PYTHONUNBUFFERED=1 \
    PORT=7860
WORKDIR /home/user/app

COPY requirements.txt .
# CPU-only torch is ~10x smaller than the default CUDA build
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .
COPY --from=frontend --chown=user /frontend/dist ./frontend/dist

USER user
# Download the embedding + reranker models at build time so cold starts are fast
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

EXPOSE 7860
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
