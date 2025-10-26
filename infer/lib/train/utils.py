import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import torch
from scipy.io import wavfile

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger(__name__)


def latest_checkpoint_path(dir_path: Path, pattern: str = "G_*.pth") -> str:
    files = sorted(
        dir_path.glob(pattern),
        key=lambda f: int("".join(filter(str.isdigit, f.stem))),
    )
    latest = str(files[-1])
    logger.debug(f"Latest checkpoint: {latest}")
    return latest


def load_wav_to_torch(full_path: Path) -> tuple[torch.FloatTensor, int]:
    sampling_rate, data = cast("tuple[int, np.ndarray]", wavfile.read(full_path))  # pyright: ignore[reportUnknownMemberType]
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate


def load_filepaths_and_text(filename: Path, split: str = "|") -> list[list[str]]:
    try:
        lines = filename.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        lines = filename.read_text().splitlines()
    return [line.strip().split(split) for line in lines]


def get_logger(model_dir: Path, filename: Path = Path("train.log")) -> logging.Logger:
    model_dir.mkdir(parents=True, exist_ok=True)
    log_file = model_dir / filename

    logger = logging.getLogger(model_dir.name)
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")
    handler = logging.FileHandler(log_file)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    return logger


@dataclass
class HParamsData:
    filter_length: int
    hop_length: int
    max_wav_value: float
    mel_fmax: int
    mel_fmin: int
    n_mel_channels: int
    sampling_rate: int
    win_length: int
    training_files: str | None = None


@dataclass
class HParamsTrain:
    batch_size: int
    betas: tuple[float, float]
    c_kl: float
    c_mel: float
    epochs: int
    eps: float
    fp16_run: bool
    init_lr_ratio: int
    learning_rate: float
    log_interval: int
    lr_decay: float
    seed: int
    segment_size: int
    warmup_epochs: int


@dataclass
class HParamsModel:
    filter_channels: int
    gin_channels: int
    hidden_channels: int
    inter_channels: int
    kernel_size: int
    n_heads: int
    n_layers: int
    p_dropout: float
    resblock_dilation_sizes: list[list[int]]
    resblock_kernel_sizes: list[int]
    resblock: str
    spk_embed_dim: int
    upsample_initial_channel: int
    upsample_kernel_sizes: list[int]
    upsample_rates: list[int]
    use_spectral_norm: bool


@dataclass
class HParams:
    data: HParamsData
    model: HParamsModel
    train: HParamsTrain

    model_dir: Path = Path()
    experiment_dir: Path = Path()
    save_every_epoch: bool = False
    name: str = ""
    total_epoch: int = 0
    pretrainG: str = ""
    pretrainD: str = ""
    gpus: str = ""
    sample_rate: int = 0
    if_f0: bool = False
    if_latest: int = 0
    save_every_weights: str = "0"
    if_cache_data_in_gpu: int = 0


def get_hparams() -> HParams:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-se",
        "--save_every_epoch",
        type=int,
        required=True,
        help="checkpoint save frequency (epoch)",
    )
    parser.add_argument(
        "-te", "--total_epoch", type=int, required=True, help="total_epoch"
    )
    parser.add_argument(
        "-pg", "--pretrainG", type=str, default="", help="Pretrained Generator path"
    )
    parser.add_argument(
        "-pd", "--pretrainD", type=str, default="", help="Pretrained Discriminator path"
    )
    parser.add_argument("-g", "--gpus", type=str, default="0", help="split by -")
    parser.add_argument(
        "-bs", "--batch_size", type=int, required=True, help="batch size"
    )
    parser.add_argument(
        "-e", "--experiment_dir", type=str, required=True, help="experiment dir"
    )  # -m
    parser.add_argument(
        "-sr", "--sample_rate", type=str, required=True, help="sample rate, 32k/40k/48k"
    )
    parser.add_argument(
        "-sw",
        "--save_every_weights",
        type=str,
        default="0",
        help="save the extracted model in weights directory when saving checkpoints",
    )
    parser.add_argument(
        "-f0",
        "--if_f0",
        type=int,
        required=True,
        help="use f0 as one of the inputs of the model, 1 or 0",
    )
    parser.add_argument(
        "-l",
        "--if_latest",
        type=int,
        required=True,
        help="if only save the latest G/D pth file, 1 or 0",
    )
    parser.add_argument(
        "-c",
        "--if_cache_data_in_gpu",
        type=int,
        required=True,
        help="if caching the dataset in GPU memory, 1 or 0",
    )
    args = parser.parse_args()
    name = args.experiment_dir
    experiment_dir = Path("./logs") / args.experiment_dir

    config_save_path = experiment_dir / "config.json"
    config = json.loads(config_save_path.read_text(encoding="utf-8"))

    hparams = HParams(**config)
    hparams.train = HParamsTrain(**config["train"])
    hparams.model = HParamsModel(**config["model"])
    hparams.data = HParamsData(**config["data"])
    hparams.model_dir = hparams.experiment_dir = experiment_dir
    hparams.save_every_epoch = args.save_every_epoch
    hparams.name = name
    hparams.total_epoch = args.total_epoch
    hparams.pretrainG = args.pretrainG
    hparams.pretrainD = args.pretrainD
    hparams.gpus = args.gpus
    hparams.train.batch_size = args.batch_size
    hparams.sample_rate = args.sample_rate
    hparams.if_f0 = args.if_f0
    hparams.if_latest = args.if_latest
    hparams.save_every_weights = args.save_every_weights
    hparams.if_cache_data_in_gpu = args.if_cache_data_in_gpu
    hparams.data.training_files = f"{experiment_dir}/filelist.txt"
    return hparams
