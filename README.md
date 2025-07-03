# RVC WebGUI Fork

<div align="center">

[![License](https://img.shields.io/github/license/yamada-sexta/rvc-webgui-fork?style=for-the-badge)](https://github.com/yamada-sexta/rvc-webgui-fork/blob/main/THIRD_PARTY_NOTICES.md)
[![GitHub stars](https://img.shields.io/github/stars/yamada-sexta/rvc-webgui-fork?style=for-the-badge)](https://github.com/yamada-sexta/rvc-webgui-fork/stargazers)
[![GitHub last commit](https://img.shields.io/github/last-commit/yamada-sexta/rvc-webgui-fork?style=for-the-badge)](https://github.com/yamada-sexta/rvc-webgui-fork/commits/main)
[![GitHub issues](https://img.shields.io/github/issues/yamada-sexta/rvc-webgui-fork?style=for-the-badge)](https://github.com/yamada-sexta/rvc-webgui-fork/issues)
[![GitHub pull requests](https://img.shields.io/github/issues-pr/yamada-sexta/rvc-webgui-fork?style=for-the-badge)](https://github.com/yamada-sexta/rvc-webgui-fork/pulls)

</div>

This fork aims to improve upon the [original RVC project](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) with several enhancements:

- Dependency Updates: All dependencies have been updated, and uv is used for package management, promising general performance improvements.

- Improved Gradio UI: The user interface has been significantly cleaned up, making it easier to use. A notable addition is the ability to upload audio training data directly from the Web GUI, which is particularly useful for server deployments.

- Better Docker Support: Enhanced Docker integration, providing a docker-compose example for easy deployment with GPU support.

## Getting Started

### Run with UV

> Install [uv](https://docs.astral.sh/uv/#installation) if you haven't already.

```bash
git clone https://github.com/yamada-sexta/rvc-webgui-fork.git
cd rvc-webgui-fork
uv sync --lock
uv run ./tools/download_models.py
uv run ./web_ui.py
```

### Docker Compose

Create a `docker-compose.yml` similar to this:

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

Then run

```bash
docker-compose up -d
```

Then visit `localhost:7865` in your browser.

## Contributing

We have a more open policy to contribution compared to the original.

Please refer [this guid](./CONTRIBUTING.md) for more details.
