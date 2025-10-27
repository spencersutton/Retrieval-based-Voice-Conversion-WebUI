import os
import sys
import traceback
from pathlib import Path

import fairseq
import fairseq.checkpoint_utils
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from fairseq.data.dictionary import Dictionary
from torch.serialization import safe_globals

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

_device = sys.argv[1]
_n_part = int(sys.argv[2])
_i_part = int(sys.argv[3])
if len(sys.argv) == 7:
    _exp_dir = Path(sys.argv[4])
    _version = sys.argv[5]
    _is_half = sys.argv[6].lower() == "true"
else:
    i_gpu = sys.argv[4]
    _exp_dir = Path(sys.argv[5])
    os.environ["CUDA_VISIBLE_DEVICES"] = str(i_gpu)
    _version = sys.argv[6]
    _is_half = sys.argv[7].lower() == "true"

if "privateuseone" not in _device:
    _device = "cpu"
    if torch.cuda.is_available():
        _device = "cuda"
    elif torch.backends.mps.is_available():
        _device = "mps"
else:
    import torch_directml  # pyright: ignore[reportMissingImports]

    _device = torch_directml.device(torch_directml.default_device())

    def forward_dml(
        ctx: fairseq.modules.grad_multiply.GradMultiply,  # pyright: ignore[reportAttributeAccessIssue]
        x: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        ctx.scale = scale
        res = x.clone().detach()
        return res

    fairseq.modules.grad_multiply.GradMultiply.forward = forward_dml  # pyright: ignore[reportAttributeAccessIssue]

_f = (_exp_dir / "extract_f0_feature.log").open("a+")


def printt(strr: str):
    print(strr)
    _f.write(f"{strr}\n")
    _f.flush()


printt(" ".join(sys.argv))
model_path = "assets/hubert/hubert_base.pt"

wav_dir = _exp_dir / "1_16k_wavs"
out_path = _exp_dir / ("3_feature256" if _version == "v1" else "3_feature768")
out_path.mkdir(parents=True, exist_ok=True)


# wave must be 16k, hop_size=320
def _readwave(wav_path: Path, normalize: bool = False):
    wav, sr = sf.read(wav_path)
    assert sr == 16000
    feats = torch.from_numpy(wav).float()
    if feats.dim() == 2:  # double channels
        feats = feats.mean(-1)
    assert feats.dim() == 1, feats.dim()
    if normalize:
        with torch.no_grad():
            feats = F.layer_norm(feats, feats.shape)
    feats = feats.view(1, -1)
    return feats


# HuBERT model
printt(f"load model(s) from {model_path}")
# if hubert model is exist
if not os.access(model_path, os.F_OK):
    printt(
        f"Error: Extracting is shut down because {model_path} does not exist, you may download it from https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main"
    )
    exit(0)

with safe_globals([Dictionary]):
    models, saved_cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
        [model_path],
        suffix="",
    )
model = models[0]
model = model.to(_device)
printt(f"move model to {_device}")
if _is_half:
    if _device not in ["mps", "cpu"]:
        model = model.half()
model.eval()

todo = sorted([p.name for p in wav_dir.iterdir() if p.is_file()])[_i_part::_n_part]
n = max(1, len(todo) // 10)  # 最多打印十条
if len(todo) == 0:
    printt("no-feature-todo")
else:
    printt(f"all-feature-{len(todo)}")
    for idx, file in enumerate(todo):
        try:
            if file.endswith(".wav"):
                wav_path = wav_dir / file
                out_path = out_path / file.replace("wav", "npy")

                if out_path.exists():
                    continue

                assert saved_cfg is not None
                feats = _readwave(wav_path, normalize=saved_cfg.task.normalize)
                padding_mask = torch.BoolTensor(feats.shape).fill_(False)
                inputs = {
                    "source": (
                        feats.half().to(_device)
                        if _is_half and _device not in ["mps", "cpu"]
                        else feats.to(_device)
                    ),
                    "padding_mask": padding_mask.to(_device),
                    "output_layer": 9 if _version == "v1" else 12,  # layer 9
                }
                with torch.no_grad():
                    logits = model.extract_features(**inputs)
                    feats = (
                        model.final_proj(logits[0]) if _version == "v1" else logits[0]
                    )

                feats = feats.squeeze(0).float().cpu().numpy()
                if np.isnan(feats).sum() == 0:
                    np.save(out_path, feats, allow_pickle=False)
                else:
                    printt(f"{file}-contains nan")
                if idx % n == 0:
                    printt(f"now-{len(todo)},all-{idx},{file},{feats.shape}")
        except Exception:
            printt(traceback.format_exc())
    printt("all-feature-done")
