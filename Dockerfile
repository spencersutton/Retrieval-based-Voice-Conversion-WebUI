FROM nvidia/cuda:12.2.2-devel-ubuntu22.04
ARG DEBIAN_FRONTEND=noninteractive
EXPOSE 7865


RUN apt-get update && apt-get install -y \
    curl ca-certificates \
    git \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY ./.python-version ./.python-version
COPY ./pyproject.toml ./pyproject.toml
COPY ./uv.lock ./uv.lock
# Download the latest installer
ADD https://astral.sh/uv/install.sh /uv-installer.sh

# Run the installer then remove it
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Ensure the installed binary is on the `PATH`
ENV PATH="/root/.local/bin/:$PATH"

# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    pkg-config \
    libcairo2-dev \
    && rm -rf /var/lib/apt/lists/*

RUN uv sync --locked
# Place executables in the environment at the front of the path
ENV PATH="/app/.venv/bin:$PATH"
COPY ./tools/download_models.py ./tools/download_models.py
RUN uv run tools/download_models.py
COPY . .
# VOLUME [ "/app/weights", "/app/opt" ]
EXPOSE 7865
CMD ["python",  "web_ui.py"]
