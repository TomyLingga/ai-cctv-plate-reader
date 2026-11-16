# Linux image for alpr-id. Not the main deployment path: on the Windows PC the service
# runs directly (start-alpr.bat or install-autostart.ps1). GPU in Docker on Windows only
# works for Linux containers through WSL2.
FROM python:3.12-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.13.4 /uv /uvx /bin/

# cpu or gpu: which ONNX Runtime build to install (see pyproject.toml).
ARG ALPR_EXTRA=cpu
ENV ALPR_EXTRA=${ALPR_EXTRA} \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PYTHONUNBUFFERED=1

# Shared libraries needed by opencv-python.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first, so this layer is reused when only the source changes.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --extra ${ALPR_EXTRA} --no-install-project

COPY src ./src
RUN uv sync --frozen --no-dev --extra ${ALPR_EXTRA}

EXPOSE 5000
CMD ["sh", "-c", "exec uv run --frozen --no-dev --extra ${ALPR_EXTRA} alpr-id serve"]
