import logging
import os
import shutil
import warnings
from pathlib import Path

import fairseq
import torch
from dotenv import load_dotenv

from configs.config import Config
from i18n.i18n import I18nAuto
from infer.modules.vc.modules import VC

# Environment and logging setup
load_dotenv()
for lib in ["numba", "httpx", "fairseq"]:
    logging.getLogger(lib).setLevel(logging.WARNING)
os.environ["OPENBLAS_NUM_THREADS"] = "1"

logger = logging.getLogger(__name__)
cwd = Path.cwd()

# Directories to clean up
cleanup_dirs = [
    cwd / "TEMP",
    cwd / "runtime" / "Lib" / "site-packages" / "infer_pack",
    cwd / "runtime" / "Lib" / "site-packages" / "uvr5_pack",
]
for d in cleanup_dirs:
    shutil.rmtree(d, ignore_errors=True)

# Ensure required directories exist
for d in [cwd / "TEMP", cwd / "logs", cwd / "assets" / "weights"]:
    d.mkdir(exist_ok=True)
os.environ["TEMP"] = str(cwd / "TEMP")

warnings.filterwarnings("ignore")
torch.manual_seed(114514)


config: Config = Config()
vc = VC(config)


if config.dml:

    def forward_dml(ctx, x, scale):  # type: ignore
        ctx.scale = scale
        res = x.clone().detach()
        return res

    fairseq.modules.grad_multiply.GradMultiply.forward = forward_dml  # type: ignore
i18n = I18nAuto()
logger.info(i18n)

# Get GPU count
ngpu = torch.cuda.device_count()
gpu_infos, mem = [], []
if_gpu_ok = False
gpu_keywords = [
    "10",
    "16",
    "20",
    "30",
    "40",
    "A2",
    "A3",
    "A4",
    "P4",
    "A50",
    "500",
    "A60",
    "70",
    "80",
    "90",
    "M4",
    "T4",
    "TITAN",
    "4060",
    "L",
    "6000",
]

if torch.cuda.is_available() and ngpu > 0:
    for i in range(ngpu):
        gpu_name = torch.cuda.get_device_name(i)
        if any(k in gpu_name.upper() for k in gpu_keywords):
            if_gpu_ok = True
            gpu_infos.append(f"{i}\t{gpu_name}")
            mem.append(
                int(torch.cuda.get_device_properties(i).total_memory / 1024**3 + 0.4)
            )

if if_gpu_ok and gpu_infos:
    gpu_info = "\n".join(gpu_infos)
    default_batch_size = min(mem) // 2
    gpus = "-".join(str(i) for i in range(len(gpu_infos)))
else:
    gpu_info = i18n(
        "Unfortunately, you don't have a usable graphics card to support your training."
    )
    default_batch_size = 1
    gpus = ""


weight_root = os.getenv("WEIGHT_ROOT", "assets/weights")
index_root = os.getenv("INDEX_ROOT", "logs")
outside_index_root = os.getenv("OUTSIDE_INDEX_ROOT", "assets/indices")
rmvpe_root = os.getenv("RMVPE_ROOT", "assets/rmvpe")

names = []
for path in Path(weight_root).iterdir():
    name = path.name
    print(f"Checking: {name}")
    if name.endswith(".pth"):
        names.append(name)
index_paths = [""]  # Fix for gradio 5


def lookup_indices(root: str):
    # shared.index_paths
    index_paths.extend(
        [
            f"{root_dir}/{name}"
            for root_dir, dirs, files in os.walk(root, topdown=False)
            for name in files
            if name.endswith(".index") and "trained" not in name
        ]
    )


for r in [index_root, outside_index_root]:
    lookup_indices(r)

sr_dict = {
    "32k": 32000,
    "40k": 40000,
    "48k": 48000,
}
