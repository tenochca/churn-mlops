FROM python:3.14-slim

COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH" \
    CHURN_MODEL_URI=/app/models/churn_model

WORKDIR /app

# 1. Dependencies only (cached until pyproject.toml or uv.lock change)
COPY pyproject.toml uv.lock .python-version README.md ./
RUN uv sync --frozen --no-dev --no-install-project

# 2. Project code and the exported model
COPY src/ ./src/
COPY models/churn_model ./models/churn_model
RUN uv sync --frozen --no-dev

EXPOSE 8000
CMD ["uvicorn", "churn_mlops.api:app", "--host", "0.0.0.0", "--port", "8000"]
