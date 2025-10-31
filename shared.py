import logging
import os
import shutil
import warnings
from pathlib import Path
from typing import Final

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
tmp = Path.cwd() / "TEMP"
shutil.rmtree(tmp, ignore_errors=True)
shutil.rmtree(Path.cwd() / "runtime/Lib/site-packages/infer_pack", ignore_errors=True)
shutil.rmtree(Path.cwd() / "runtime/Lib/site-packages/uvr5_pack", ignore_errors=True)
tmp.mkdir(parents=True, exist_ok=True)
(log_dir := Path.cwd() / "logs").mkdir(parents=True, exist_ok=True)
(assets_dir := Path.cwd() / "assets/weights").mkdir(parents=True, exist_ok=True)
os.environ["TEMP"] = str(tmp)
warnings.filterwarnings("ignore")
torch.manual_seed(114514)

config: Config = Config()
vc = VC(config)

F0_DIR_NAME: Final = "2a_f0"
F0_NSF_DIR_NAME: Final = "2b-f0nsf"
GT_WAVS_DIR_NAME: Final = "0_gt_wavs"
WAVS_16K_DIR_NAME: Final = "1_16k_wavs"
FEATURE_DIMENSION: Final = 256
FEATURE_DIMENSION_V2: Final = 768
FEATURE_DIR_NAME: Final = f"3_feature{FEATURE_DIMENSION}"
FEATURE_DIR_NAME_V2: Final = f"3_feature{FEATURE_DIMENSION_V2}"

if config.dml:

    def forward_dml(ctx, x: torch.Tensor, scale: float) -> torch.Tensor:  # type: ignore
        ctx.scale = scale
        res = x.clone().detach()
        return res

    fairseq.modules.grad_multiply.GradMultiply.forward = forward_dml  # type: ignore

i18n = I18nAuto()
logger.info(i18n)
# Get GPU count
_n_gpu = torch.cuda.device_count()
_gpu_infos: list[str] = []
_mem: list[int] = []
_if_gpu_ok: bool = False

if torch.cuda.is_available() or _n_gpu != 0:
    for i in range(_n_gpu):
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
            _if_gpu_ok = True  # At least one usable GPU available
            _gpu_infos.append(f"{i}\t{gpu_name}")
            _mem.append(
                int(
                    torch.cuda.get_device_properties(i).total_memory
                    / 1024
                    / 1024
                    / 1024
                    + 0.4
                )
            )
if _if_gpu_ok and len(_gpu_infos) > 0:
    gpu_info = "\n".join(_gpu_infos)
    default_batch_size = min(_mem) // 2
else:
    gpu_info = i18n(
        "Unfortunately, you don't have a usable graphics card to support your training."
    )
    default_batch_size = 1
gpus = "-".join([i[0] for i in _gpu_infos])

weight_root = Path(os.getenv("WEIGHT_ROOT", "assets/weights"))
index_root = Path(os.getenv("INDEX_ROOT", "logs"))
outside_index_root = Path(os.getenv("OUTSIDE_INDEX_ROOT", "assets/indices"))
rmvpe_root = Path(os.getenv("RMVPE_ROOT", "assets/rmvpe"))

names = [p.name for p in weight_root.iterdir() if p.name.endswith(".pth")]

# Initialize index_paths for gradio compatibility
index_paths = [""]


def lookup_indices(root: Path):
    index_paths.extend(
        [str(path) for path in root.rglob("*.index") if "trained" not in path.name]
    )


lookup_indices(index_root)
lookup_indices(outside_index_root)
