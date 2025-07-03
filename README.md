# RVC WebGUI Fork

This fork aims to improve upon the [original RVC project](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) with several enhancements:

- Dependency Updates: All dependencies have been updated, and uv is used for package management, promising general performance improvements.

- Improved Gradio UI: The user interface has been significantly cleaned up, making it easier to use. A notable addition is the ability to upload audio training data directly from the Web GUI, which is particularly useful for server deployments.

- Better Docker Support: Enhanced Docker integration, providing a docker-compose example for easy deployment with GPU support.

## Getting Started

### Run with UV

> Install [uv](https://docs.astral.sh/uv/#installation) if you haven't already.

#### Clone the repo

```bash
git clone https://github.com/yamada-sexta/rvc-webgui-fork.git
cd rvc-webgui-fork
```

```bash
uv sync --lock
uv run ./tools/download_models.py
uv run ./web_ui.py
```

### Docker Compose

```yml
services:
  rvc-fork-server:
    build:
      context: https://github.com/yamada-sexta/rvc-webgui-fork.git
      dockerfile: Dockerfile
    restart: "unless-stopped"
    shm_size: '16gb'
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    ports:
      - "7865:7865"
    volumes:
      - ./data/datasets:/app/datasets
      - ./data/weights:/app/assets/weights
      - ./data/logs:/app/logs
      - /app/logs/mute
```