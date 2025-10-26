import argparse
import json
import logging
import sys
from collections.abc import Callable
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)


version_config_list = [
    "v2/32k.json",
    "v2/40k.json",
    "v2/48k.json",
]


def singleton_variable(func: Callable[..., Any]) -> Callable[..., Any]:
    def wrapper(*args, **kwargs):
        if not wrapper.instance:
            wrapper.instance = func(*args, **kwargs)
        return wrapper.instance

    wrapper.instance = None
    return wrapper


@singleton_variable
class Config:
    python_cmd: str

    def __init__(self) -> None:
        self.device = "cuda:0"
        self.is_half = True
        self.use_jit = False
        self.n_cpu = 0
        self.json_config = self.load_config_json()
        self.gpu_mem = None
        (
            self.python_cmd,
            self.listen_port,
            self.iscolab,
            self.noparallel,
            self.noautoopen,
        ) = self.arg_parse()
        self.instead = ""
        self.preprocess_per = 3.7
        self.x_pad, self.x_query, self.x_center, self.x_max = self.device_config()

    @staticmethod
    def load_config_json() -> dict[str, dict[str, object]]:
        configs: dict[str, dict[str, object]] = {}
        for config_file in version_config_list:
            src = Path("configs") / config_file
            dst = Path("configs/inuse") / config_file

            if not dst.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_text(src.read_text())

            with dst.open("r") as f:
                configs[config_file] = json.load(f)

        return configs

    @staticmethod
    def arg_parse() -> tuple[str, int, bool, bool, bool]:
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
        cmd_opts = parser.parse_args()

        cmd_opts.port = cmd_opts.port if 0 <= cmd_opts.port <= 65535 else 7865

        return (
            cmd_opts.pycmd,
            cmd_opts.port,
            cmd_opts.colab,
            cmd_opts.noparallel,
            cmd_opts.noautoopen,
        )

    # has_mps is only available in nightly pytorch (for now) and MasOS 12.3+.
    # check `getattr` and try it for compatibility
    @staticmethod
    def has_mps() -> bool:
        if not torch.backends.mps.is_available():
            return False
        try:
            torch.zeros(1).to(torch.device("mps"))
            return True
        except Exception:
            return False

    def use_fp32_config(self) -> None:
        for config_file in version_config_list:
            # Set fp16_run to False in config
            self.json_config[config_file]["train"]["fp16_run"] = False

            # Update config file on disk
            path = Path("configs/inuse") / config_file
            config_text = path.read_text()
            updated_text = config_text.replace("true", "false")
            path.write_text(updated_text)

            logger.info(f"Overwrote {config_file} to use fp32")

        self.preprocess_per = 3.0
        logger.info(f"Set preprocess_per to {self.preprocess_per}")

    def device_config(self) -> tuple[int, int, int, int]:
        # Detect device and configure precision
        if torch.cuda.is_available():
            i_device = int(self.device.split(":")[-1])
            logger.info("Found GPU %s", self.torch.cuda.get_device_name(i_device))
            self.gpu_mem = int(
                torch.cuda.get_device_properties(i_device).total_memory / (1024**3)
                + 0.4
            )
            if self.gpu_mem <= 4:
                self.preprocess_per = 3.0
        elif self.has_mps():
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
            x_pad, x_query, x_center, x_max = 3, 10, 60, 65
        else:
            x_pad, x_query, x_center, x_max = 1, 6, 38, 41

        if self.gpu_mem is not None and self.gpu_mem <= 4:
            x_pad, x_query, x_center, x_max = 1, 5, 30, 32

        if self.instead:
            logger.info(f"Use {self.instead} instead")

        logger.info(
            f"Half-precision floating-point: {self.is_half}, device: {self.device}"
        )
        return x_pad, x_query, x_center, x_max
