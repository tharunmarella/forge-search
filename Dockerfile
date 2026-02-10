# forge-search — Code Intelligence API
# Lightweight: no torch, no local models, no GPU
# Embeddings via Jina AI API, LLM via Groq API

FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

RUN mkdir -p /app/data
ENV STORE_DB_PATH=/app/data/forge_search.db

EXPOSE 8080

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
