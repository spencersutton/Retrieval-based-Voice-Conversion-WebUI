import argparse
import glob
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from scipy.io.wavfile import read

MATPLOTLIB_FLAG = False

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging


def load_checkpoint(checkpoint_path, model, optimizer=None, load_opt=1):
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")

    saved_state_dict = checkpoint_dict["model"]
    if hasattr(model, "module"):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    new_state_dict = {}
    for k, v in state_dict.items():  # shape required by the model
        try:
            new_state_dict[k] = saved_state_dict[k]
            if saved_state_dict[k].shape != v.shape:
                logger.warning(
                    "shape-%s-mismatch|need-%s|get-%s",
                    k,
                    v.shape,
                    saved_state_dict[k].shape,
                )
                raise KeyError
        except:
            logger.info("%s is not in the checkpoint", k)  # missing in checkpoint
            new_state_dict[k] = v  # use model's own random value
    if hasattr(model, "module"):
        model.module.load_state_dict(new_state_dict, strict=False)
    else:
        model.load_state_dict(new_state_dict, strict=False)
    logger.info("Loaded model weights")

    iteration = checkpoint_dict["iteration"]
    learning_rate = checkpoint_dict["learning_rate"]
    # If loading fails and optimizer is empty, reinitialize.
    # This may also affect the learning rate scheduler update,
    # so catch this at the outermost level in the train file.
    if optimizer is not None and load_opt == 1:
        optimizer.load_state_dict(checkpoint_dict["optimizer"])
    logger.info("Loaded checkpoint '{}' (epoch {})".format(checkpoint_path, iteration))
    return model, optimizer, learning_rate, iteration


def save_checkpoint(model, optimizer, learning_rate, iteration, checkpoint_path):
    logger.info(
        "Saving model and optimizer state at epoch {} to {}".format(
            iteration, checkpoint_path
        )
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


def summarize(
    writer,
    global_step,
    scalars=None,
    histograms=None,
    images=None,
    audios=None,
    audio_sampling_rate=22050,
):
    if audios is None:
        audios = {}
    if images is None:
        images = {}
    if histograms is None:
        histograms = {}
    if scalars is None:
        scalars = {}
    for k, v in scalars.items():
        writer.add_scalar(k, v, global_step)
    for k, v in histograms.items():
        writer.add_histogram(k, v, global_step)
    for k, v in images.items():
        writer.add_image(k, v, global_step, dataformats="HWC")
    for k, v in audios.items():
        writer.add_audio(k, v, global_step, audio_sampling_rate)


def latest_checkpoint_path(dir_path, regex="G_*.pth"):
    f_list = glob.glob(os.path.join(dir_path, regex))
    f_list.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
    x = f_list[-1]
    logger.debug(x)
    return x


def plot_spectrogram_to_numpy(spectrogram):
    global MATPLOTLIB_FLAG
    if not MATPLOTLIB_FLAG:
        import matplotlib

        matplotlib.use("Agg")
        MATPLOTLIB_FLAG = True
        mpl_logger = logging.getLogger("matplotlib")
        mpl_logger.setLevel(logging.WARNING)
    import matplotlib.pylab as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation="none")
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()

    fig.canvas.draw()
    rgba_buffer = fig.canvas.buffer_rgba()  # type: ignore
    data = np.frombuffer(rgba_buffer, dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    data = data[:, :, :3]  # Remove alpha channel, keep only RGB
    plt.close()
    return data


def load_wav_to_torch(full_path):
    sampling_rate, data = read(full_path)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate


def load_filepaths_and_text(filename: Path, split="|"):
    try:
        with filename.open(encoding="utf-8") as f:
            filepaths_and_text = [line.strip().split(split) for line in f]
    except UnicodeDecodeError:
        with filename.open() as f:
            filepaths_and_text = [line.strip().split(split) for line in f]

    return filepaths_and_text


def get_hparams(init=True):
    """
    todo:
      Final seven items:
        Save frequency, total epochs                  done
        batch size                                    done
        pretrainG, pretrainD                          done
        GPU id: os.en["CUDA_VISIBLE_DEVICES"]         done
        if_latest                                     done
      Model: if_f0                                    done
      Sample rate: auto-select from config            done
      Whether to cache dataset in GPU: if_cache_data_in_gpu done

      -m:
        Automatically determine training_files path, modify hps.data.training_files in train_nsf_load_pretrain.py    done
      -c is no longer needed
    """
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
    )
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
    with open(config_save_path, "r") as f:
        config = json.load(f)

    hparams = HParams.from_dict(config)
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
    hparams.data.training_files = Path(experiment_dir) / "filelist.txt"
    return hparams


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
class _TrainConfig:
    """Training hyperparameters configuration."""

    log_interval: int = 200
    seed: int = 1234
    epochs: int = 20000
    learning_rate: float = 1e-4
    betas: tuple[float, float] = field(default_factory=lambda: (0.8, 0.99))
    eps: float = 1e-9
    batch_size: int = 4
    fp16_run: bool = True
    lr_decay: float = 0.999875
    segment_size: int = 17280
    init_lr_ratio: float = 1
    warmup_epochs: int = 0
    c_mel: float = 45
    c_kl: float = 1.0


@dataclass
class DataConfig:
    """Data processing hyperparameters configuration."""

    max_wav_value: float = 32768.0
    sampling_rate: int = 48000
    filter_length: int = 2048
    hop_length: int = 480
    win_length: int = 2048
    n_mel_channels: int = 128
    mel_fmin: float = 0.0
    mel_fmax: Optional[float] = None
    training_files: Path = Path()


@dataclass
class _ModelConfig:
    """Model architecture hyperparameters configuration."""

    inter_channels: int = 192
    hidden_channels: int = 192
    filter_channels: int = 768
    n_heads: int = 2
    n_layers: int = 6
    kernel_size: int = 3
    p_dropout: float = 0
    resblock: str = "1"
    resblock_kernel_sizes: List[int] = field(default_factory=lambda: [3, 7, 11])
    resblock_dilation_sizes: List[List[int]] = field(
        default_factory=lambda: [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
    )
    upsample_rates: List[int] = field(default_factory=lambda: [12, 10, 2, 2])
    upsample_initial_channel: int = 512
    upsample_kernel_sizes: List[int] = field(default_factory=lambda: [24, 20, 4, 4])
    use_spectral_norm: bool = False
    gin_channels: int = 256
    spk_embed_dim: int = 109


@dataclass
class HParams:
    """Complete hyperparameters configuration combining train, data, and model configs."""

    train: _TrainConfig = field(default_factory=_TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: _ModelConfig = field(default_factory=_ModelConfig)

    # Additional runtime parameters
    model_dir: str = ""
    experiment_dir: str = ""
    save_every_epoch: int = 0
    name: str = ""
    total_epoch: int = 0
    pretrainG: str = ""
    pretrainD: str = ""
    version: str = ""
    gpus: str = "0"
    sample_rate: int = 48000
    if_f0: int = 1
    if_latest: int = 1
    save_every_weights: str = "0"
    if_cache_data_in_gpu: int = 0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HParams":
        """Create HParams from a dictionary, handling nested configs."""
        train_config = data.get("train", {})
        data_config = data.get("data", {})
        model_config = data.get("model", {})

        return cls(
            train=_TrainConfig(**train_config),
            data=DataConfig(**data_config),
            model=_ModelConfig(**model_config),
        )
