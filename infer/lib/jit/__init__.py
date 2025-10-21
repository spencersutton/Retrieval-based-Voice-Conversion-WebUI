import pickle
import time
from collections import OrderedDict
from io import BytesIO
from pathlib import Path

import torch


def _export(
    model: torch.nn.Module,
    device=None,
    is_half: bool = False,
) -> dict:
    if device is None:
        device = torch.device("cpu")
    model = model.half() if is_half else model.float()
    model.eval()
    model_jit = torch.jit.script(model)
    model_jit.to(device)
    model_jit = model_jit.half() if is_half else model_jit.float()
    buffer = BytesIO()
    # model_jit=model_jit.cpu()
    torch.jit.save(model_jit, buffer)
    del model_jit
    cpt = OrderedDict()
    cpt["model"] = buffer.getvalue()
    cpt["is_half"] = is_half
    return cpt


def load(path: str):
    with Path(path).open("rb") as f:
        return pickle.load(f)


def _save(ckpt: dict, save_path: str):
    with Path(save_path).open("wb") as f:
        pickle.dump(ckpt, f)


def rmvpe_jit_export(
    model_path: str,
    save_path: str | None = None,
    device=None,
    is_half=False,
):
    if device is None:
        device = torch.device("cpu")
    if not save_path:
        save_path = model_path.rstrip(".pth")
        save_path += ".half.jit" if is_half else ".jit"
    if "cuda" in str(device) and ":" not in str(device):
        device = torch.device("cuda:0")
    from .get_rmvpe import get_rmvpe

    model = get_rmvpe(model_path, device)
    ckpt = _export(model, device, is_half)
    ckpt["device"] = str(device)
    _save(ckpt, save_path)
    return ckpt
