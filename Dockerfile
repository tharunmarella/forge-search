# forge-search — Code Intelligence API
# Runs on Railway Pro (CPU only, no GPU needed)
#
# Embeddings: jina-code-v2 (baked into image, ~600MB RAM)
# Store: numpy + SQLite (persistent volume)
# LLM: Groq API (external, free tier)

FROM python:3.10-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download embedding model (baked in = no cold start download)
RUN python3 -c "\
from sentence_transformers import SentenceTransformer; \
m = SentenceTransformer('jinaai/jina-embeddings-v2-base-code', trust_remote_code=True); \
print(f'Model cached: dims={m.get_sentence_embedding_dimension()}')"

# Copy app
COPY app/ ./app/

# Data directory for SQLite (mount Railway volume here)
RUN mkdir -p /app/data
ENV STORE_DB_PATH=/app/data/forge_search.db

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
