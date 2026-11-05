FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

WORKDIR /app

# Dependencies first so code changes don't invalidate this layer.
COPY requirements.txt .
RUN pip install -r requirements.txt

# Code lives at the top of /app (flat imports: `import settings`, `from tasks import ...`).
COPY src/ /app/
COPY resources/ /app/resources/

# Never run as root.
RUN useradd --create-home --uid 1000 app && chown -R app /app
USER app

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
