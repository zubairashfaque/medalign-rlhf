FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY configs/ ./configs/
COPY scripts/ ./scripts/

RUN pip install --no-cache-dir poetry && \
    poetry config virtualenvs.create false && \
    poetry install --only main --no-interaction --no-ansi

EXPOSE 8000
CMD ["python", "scripts/serve.py", "--gguf", "/models/model-Q4_K_M.gguf", "--use-rag", "--port", "8000"]
