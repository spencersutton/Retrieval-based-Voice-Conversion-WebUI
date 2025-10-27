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

load_dotenv()
logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("fairseq").setLevel(logging.WARNING)
os.environ["OPENBLAS_NUM_THREADS"] = "1"


logger = logging.getLogger(__name__)
now_dir = Path.cwd()
tmp = now_dir / "TEMP"
shutil.rmtree(tmp, ignore_errors=True)
shutil.rmtree(
    now_dir / "runtime" / "Lib" / "site-packages" / "infer_pack", ignore_errors=True
)
shutil.rmtree(
    now_dir / "runtime" / "Lib" / "site-packages" / "uvr5_pack", ignore_errors=True
)
tmp.mkdir(exist_ok=True)
(now_dir / "logs").mkdir(exist_ok=True)
(now_dir / "assets" / "weights").mkdir(exist_ok=True)
os.environ["TEMP"] = str(tmp)
warnings.filterwarnings("ignore")
torch.manual_seed(114514)


config: Config = Config()
vc = VC(config)


if config.dml:

    def forward_dml(ctx, x, scale):
        ctx.scale = scale
        res = x.clone().detach()
        return res

    fairseq.modules.grad_multiply.GradMultiply.forward = forward_dml
i18n = I18nAuto()
logger.info(i18n)
# Get GPU count
ngpu = torch.cuda.device_count()
gpu_infos: list[str] = []
mem: list[int] = []
if_gpu_ok: bool = False

if torch.cuda.is_available() or ngpu != 0:
    for i in range(ngpu):
        gpu_name = torch.cuda.get_device_name(i)
        if any(
            value in gpu_name.upper()
            for value in [
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
        ):
            # A10#A100#V100#A40#P40#M40#K80#A4500
            if_gpu_ok = True  # 至少有一张能用的N卡
            gpu_infos.append(f"{i}\t{gpu_name}")
            mem.append(
                int(
                    torch.cuda.get_device_properties(i).total_memory
                    / 1024
                    / 1024
                    / 1024
                    + 0.4
                )
            )
if if_gpu_ok and len(gpu_infos) > 0:
    gpu_info = "\n".join(gpu_infos)
    default_batch_size = min(mem) // 2
else:
    gpu_info = i18n(
        "Unfortunately, you don't have a usable graphics card to support your training."
    )
    default_batch_size = 1
gpus = "-".join([i[0] for i in gpu_infos])


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


lookup_indices(index_root)
lookup_indices(outside_index_root)

sr_dict = {
    "32k": 32000,
    "40k": 40000,
    "48k": 48000,
}
