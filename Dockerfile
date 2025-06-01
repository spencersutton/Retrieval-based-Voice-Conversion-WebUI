FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04
ARG DEBIAN_FRONTEND=noninteractive
EXPOSE 7865

WORKDIR /app

COPY ./.python-version ./.python-version
COPY ./pyproject.toml ./pyproject.toml
COPY ./uv.lock ./uv.lock

RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    curl ca-certificates \
    git \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Download the latest installer
ADD https://astral.sh/uv/install.sh /uv-installer.sh

# Run the installer then remove it
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Ensure the installed binary is on the `PATH`
ENV PATH="/root/.local/bin/:$PATH"

# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy

RUN uv sync --locked
# Place executables in the environment at the front of the path
ENV PATH="/app/.venv/bin:$PATH"
COPY ./tools/download_models.py ./tools/download_models.py
RUN uv run tools/download_models.py
COPY . .
# VOLUME [ "/app/weights", "/app/opt" ]
EXPOSE 7865
CMD ["python",  "infer-web.py"]
