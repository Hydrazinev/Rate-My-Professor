FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

# Optional: default for local; Railway will override PORT
ENV PORT=8000

# Force shell, and print PORT so we can see it in logs
ENTRYPOINT ["sh","-c"]
CMD ["echo PORT=${PORT}; uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]

