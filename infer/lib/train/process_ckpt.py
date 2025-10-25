import traceback
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch

from i18n.i18n import I18nAuto
from infer.lib.train.utils import HParams

i18n = I18nAuto()


def savee(
    ckpt: dict[str, torch.Tensor],
    sr: int,
    if_f0: bool,
    name: str,
    epoch: int,
    hps: HParams,
) -> str:
    try:
        opt: OrderedDict[str, object] = OrderedDict()
        opt["weight"] = {}
        for key in ckpt.keys():
            if "enc_q" in key:
                continue
            opt["weight"][key] = ckpt[key].half()
        opt["config"] = [
            hps.data.filter_length // 2 + 1,
            32,
            hps.model.inter_channels,
            hps.model.hidden_channels,
            hps.model.filter_channels,
            hps.model.n_heads,
            hps.model.n_layers,
            hps.model.kernel_size,
            hps.model.p_dropout,
            hps.model.resblock,
            hps.model.resblock_kernel_sizes,
            hps.model.resblock_dilation_sizes,
            hps.model.upsample_rates,
            hps.model.upsample_initial_channel,
            hps.model.upsample_kernel_sizes,
            hps.model.spk_embed_dim,
            hps.model.gin_channels,
            hps.data.sampling_rate,
        ]
        opt["info"] = f"{epoch}epoch"
        opt["sr"] = sr
        opt["f0"] = if_f0
        torch.save(opt, f"assets/weights/{name}.pth")
        return "Success."
    except Exception:
        return traceback.format_exc()


def show_info(path: str) -> str:
    try:
        a = torch.load(path, map_location="cpu")
        return "模型信息:{}\n采样率:{}\n模型是否输入音高引导:{}".format(
            a.get("info", "None"),
            a.get("sr", "None"),
            a.get("f0", "None"),
        )
    except Exception:
        return traceback.format_exc()


def extract_small_model(path: str, name: str, sr: str, if_f0: str, info: str) -> str:
    try:
        ckpt = torch.load(path, map_location="cpu")
        if "model" in ckpt:
            ckpt = ckpt["model"]
        opt: OrderedDict[str, object] = OrderedDict()
        opt["weight"] = {}
        for key in ckpt.keys():
            if "enc_q" in key:
                continue
            opt["weight"][key] = ckpt[key].half()
        if sr == "40k":
            opt["config"] = [
                1025,
                32,
                192,
                192,
                768,
                2,
                6,
                3,
                0,
                "1",
                [3, 7, 11],
                [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                [10, 10, 2, 2],
                512,
                [16, 16, 4, 4],
                109,
                256,
                40000,
            ]
        elif sr == "48k":
            opt["config"] = [
                1025,
                32,
                192,
                192,
                768,
                2,
                6,
                3,
                0,
                "1",
                [3, 7, 11],
                [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                [12, 10, 2, 2],
                512,
                [24, 20, 4, 4],
                109,
                256,
                48000,
            ]
        elif sr == "32k":
            opt["config"] = [
                513,
                32,
                192,
                192,
                768,
                2,
                6,
                3,
                0,
                "1",
                [3, 7, 11],
                [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                [10, 8, 2, 2],
                512,
                [20, 16, 4, 4],
                109,
                256,
                32000,
            ]
        if info == "":
            info = "Extracted model."
        opt["info"] = info
        opt["sr"] = sr
        opt["f0"] = int(if_f0)
        torch.save(opt, f"assets/weights/{name}.pth")
        return "Success."
    except Exception:
        return traceback.format_exc()


def change_info(path: str, info: str, name: str) -> str:
    try:
        ckpt = torch.load(path, map_location="cpu")
        ckpt["info"] = info
        if name == "":
            name = Path(path).name
        torch.save(ckpt, f"assets/weights/{name}")
        return "Success."
    except Exception:
        return traceback.format_exc()


def merge(
    path1: str,
    path2: str,
    alpha1: torch.Tensor,
    sr: str,
    f0: object,
    info: str,
    name: str,
) -> str:
    try:

        def extract(ckpt: dict[str, dict[str, Any]]) -> OrderedDict[str, Any]:
            a = ckpt["model"]
            opt: OrderedDict[str, object] = OrderedDict()
            opt["weight"] = {}
            for key in a.keys():
                if "enc_q" in key:
                    continue
                opt["weight"][key] = a[key]
            return opt

        ckpt1: dict[str, Any] = torch.load(path1, map_location="cpu")
        ckpt2: dict[str, Any] = torch.load(path2, map_location="cpu")
        cfg = ckpt1["config"]
        if "model" in ckpt1:
            ckpt1 = extract(ckpt1)
        else:
            ckpt1 = ckpt1["weight"]
        if "model" in ckpt2:
            ckpt2 = extract(ckpt2)
        else:
            ckpt2 = ckpt2["weight"]
        if sorted(ckpt1.keys()) != sorted(ckpt2.keys()):
            return "Fail to merge the models. The model architectures are not the same."
        opt: OrderedDict[str, Any] = OrderedDict()
        opt["weight"] = {}
        for key in ckpt1.keys():
            if key == "emb_g.weight" and ckpt1[key].shape != ckpt2[key].shape:
                min_shape0 = min(ckpt1[key].shape[0], ckpt2[key].shape[0])
                merged_emb_weight = alpha1 * (ckpt1[key][:min_shape0].float()) + (
                    1 - alpha1
                ) * (ckpt2[key][:min_shape0].float())
                opt["weight"][key] = (merged_emb_weight).half()
            else:
                opt["weight"][key] = (
                    alpha1 * (ckpt1[key].float()) + (1 - alpha1) * (ckpt2[key].float())
                ).half()

        opt["config"] = cfg
        opt["sr"] = sr
        opt["f0"] = 1 if f0 == i18n("是") else 0
        opt["info"] = info
        torch.save(opt, f"assets/weights/{name}.pth")
        return "Success."
    except Exception:
        return traceback.format_exc()
