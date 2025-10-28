import argparse
import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.io.wavfile import read

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging


def load_checkpoint(
    checkpoint_path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    load_opt: int = 1,
):
    assert checkpoint_path.is_file()
    checkpoint_dict = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )

    saved_state_dict = checkpoint_dict["model"]
    if hasattr(model, "module"):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    new_state_dict = {}
    for k, v in state_dict.items():  # 模型需要的shape
        try:
            new_state_dict[k] = saved_state_dict[k]
            if saved_state_dict[k].shape != state_dict[k].shape:
                logger.warning(
                    "shape-%s-mismatch|need-%s|get-%s",
                    k,
                    state_dict[k].shape,
                    saved_state_dict[k].shape,
                )  #
                raise KeyError
        except Exception:
            logger.info("%s is not in the checkpoint", k)  # pretrain缺失的
            new_state_dict[k] = v  # 模型自带的随机值
    if hasattr(model, "module"):
        model.module.load_state_dict(new_state_dict, strict=False)
    else:
        model.load_state_dict(new_state_dict, strict=False)
    logger.info("Loaded model weights")

    iteration = checkpoint_dict["iteration"]
    learning_rate = checkpoint_dict["learning_rate"]
    if (
        optimizer is not None and load_opt == 1
    ):  ###加载不了，如果是空的的话，重新初始化，可能还会影响lr时间表的更新，因此在train文件最外围catch
        optimizer.load_state_dict(checkpoint_dict["optimizer"])
    logger.info(f"Loaded checkpoint '{checkpoint_path}' (epoch {iteration})")
    return model, optimizer, learning_rate, iteration


def save_checkpoint(model, optimizer, learning_rate, iteration, checkpoint_path):
    logger.info(
        f"Saving model and optimizer state at epoch {iteration} to {checkpoint_path}"
    )
    if hasattr(model, "module"):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    torch.save(
        {
            "model": state_dict,
            "iteration": iteration,
            "optimizer": optimizer.state_dict(),
            "learning_rate": learning_rate,
        },
        checkpoint_path,
    )


def latest_checkpoint_path(dir_path: Path, regex: str = "G_*.pth"):
    f_list = list(dir_path.glob(regex))
    if len(f_list) == 0:
        return None
    f_list.sort(key=lambda f: int("".join(filter(str.isdigit, str(f)))))
    x = f_list[-1]
    logger.debug(x)
    return x


def load_wav_to_torch(full_path):
    sampling_rate, data = read(full_path)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate


def load_filepaths_and_text(filename, split="|"):
    try:
        with open(filename, encoding="utf-8") as f:
            filepaths_and_text = [line.strip().split(split) for line in f]
    except UnicodeDecodeError:
        with open(filename) as f:
            filepaths_and_text = [line.strip().split(split) for line in f]

    return filepaths_and_text


def get_hparams():
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
        "-v", "--version", type=str, required=True, help="model version"
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
    experiment_dir = os.path.join("./logs", args.experiment_dir)

    config_save_path = os.path.join(experiment_dir, "config.json")
    with open(config_save_path) as f:
        config = json.load(f)

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
    hparams.version = args.version
    hparams.gpus = args.gpus
    hparams.train.batch_size = args.batch_size
    hparams.sample_rate = args.sample_rate
    hparams.if_f0 = args.if_f0
    hparams.if_latest = args.if_latest
    hparams.save_every_weights = args.save_every_weights
    hparams.if_cache_data_in_gpu = args.if_cache_data_in_gpu
    hparams.data.training_files = f"{experiment_dir}/filelist.txt"
    return hparams


def check_git_hash(model_dir: str):
    source_dir = os.path.dirname(os.path.realpath(__file__))
    if not os.path.exists(os.path.join(source_dir, ".git")):
        logger.warning(
            f"{source_dir} is not a git repository, therefore hash value comparison will be ignored."
        )
        return

    cur_hash = subprocess.getoutput("git rev-parse HEAD")

    path = os.path.join(model_dir, "githash")
    if os.path.exists(path):
        saved_hash = open(path).read()
        if saved_hash != cur_hash:
            logger.warning(
                f"git hash values are different. {saved_hash[:8]}(saved) != {cur_hash[:8]}(current)"
            )
    else:
        open(path, "w").write(cur_hash)


def get_logger(model_dir, filename="train.log"):
    global logger
    logger = logging.getLogger(os.path.basename(model_dir))
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    h = logging.FileHandler(os.path.join(model_dir, filename))
    h.setLevel(logging.DEBUG)
    h.setFormatter(formatter)
    logger.addHandler(h)
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

    model_dir: str = ""
    experiment_dir: str = ""
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
    version: str = "v2"
