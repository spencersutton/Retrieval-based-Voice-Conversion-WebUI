import argparse
import json
import logging
import sys
from multiprocessing import cpu_count
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


CONFIG_FILES = [
    "v2/32k.json",
    "v2/40k.json",
    "v2/48k.json",
]


class Config:
    _instance = None
    python_cmd: str

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        self.device = "cuda:0"
        self.is_half = True
        self.use_jit = False
        self.n_cpu = 0
        self.gpu_mem = None
        self.instead = ""
        self.preprocess_per = 3.7

        # Load config JSON files
        self.json_config: dict[str, dict[str, dict[str, object]]] = {}
        for config_file in CONFIG_FILES:
            config_path = Path("configs")
            src = config_path / config_file
            dst = config_path / "inuse" / config_file
            if not dst.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_text(src.read_text())

            with dst.open("r") as f:
                self.json_config[config_file] = json.load(f)

        # Parse command-line arguments
        exe = sys.executable or "python"
        parser = argparse.ArgumentParser()
        parser.add_argument("--port", type=int, default=7865, help="Listen port")
        parser.add_argument("--pycmd", type=str, default=exe, help="Python command")
        parser.add_argument("--colab", action="store_true", help="Launch in colab")
        parser.add_argument(
            "--noparallel", action="store_true", help="Disable parallel processing"
        )
        parser.add_argument(
            "--noautoopen",
            action="store_true",
            help="Do not open in browser automatically",
        )
        args = parser.parse_args()

        self.python_cmd = args.pycmd
        self.listen_port = args.port if 0 <= args.port <= 65535 else 7865
        self.iscolab = args.colab
        self.noparallel = args.noparallel
        self.noautoopen = args.noautoopen

        # Detect device and configure precision
        if torch.cuda.is_available():
            i_device = int(self.device.split(":")[-1])
            logger.info("Found GPU %s", torch.cuda.get_device_name(i_device))
            total_memory = torch.cuda.get_device_properties(i_device).total_memory
            self.gpu_mem = int(total_memory / (1024**3) + 0.4)
            if self.gpu_mem <= 4:
                self.preprocess_per = 3.0
        elif torch.mps.is_available():
            logger.info("Found Apple Silicon GPU")
            self.device = self.instead = "mps"
            self.is_half = False
            self.use_fp32_config()
        else:
            logger.info("No supported GPU found")
            self.device = self.instead = "cpu"
            self.is_half = False
            self.use_fp32_config()

        # Set CPU count if not set
        if self.n_cpu == 0:
            self.n_cpu = cpu_count()

        # Set config values based on precision and memory
        if self.is_half:
            self.x_pad, self.x_query, self.x_center, self.x_max = 3, 10, 60, 65
        else:
            self.x_pad, self.x_query, self.x_center, self.x_max = 1, 6, 38, 41

        if self.gpu_mem is not None and self.gpu_mem <= 4:
            self.x_pad, self.x_query, self.x_center, self.x_max = 1, 5, 30, 32

        if self.instead:
            logger.info(f"Use {self.instead} instead")

        logger.info(
            f"Half-precision floating-point: {self.is_half}, device: {self.device}"
        )

    def use_fp32_config(self) -> None:
        for config_file in CONFIG_FILES:
            # Update in-memory config
            self.json_config[config_file]["train"]["fp16_run"] = False

            # Update config file on disk
            path = Path("configs/inuse") / config_file
            path.write_text(path.read_text().replace("true", "false"))

            logger.info(f"Set {config_file} to use fp32")

        self.preprocess_per = 3.0
        logger.info(f"preprocess_per set to {self.preprocess_per}")
