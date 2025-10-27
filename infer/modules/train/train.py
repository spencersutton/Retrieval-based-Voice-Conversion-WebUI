import argparse
import datetime
import json
import logging
import os
import random
import sys
import traceback
from dataclasses import asdict
from pathlib import Path
from time import sleep
from time import time as ttime

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from infer.lib.infer_pack import commons
from infer.lib.infer_pack.models import MultiPeriodDiscriminator
from infer.lib.infer_pack.models import SynthesizerTrnMs768NSFsid as RVC_Model_f0
from infer.lib.infer_pack.models import SynthesizerTrnMs768NSFsid_nono as RVC_Model_nof0
from infer.lib.train import params
from infer.lib.train.data_utils import (
    DistributedBucketSampler,
    TextAudioCollate,
    TextAudioCollateMultiNSFsid,
    TextAudioLoader,
    TextAudioLoaderMultiNSFsid,
)
from infer.lib.train.losses import (
    discriminator_loss,
    feature_loss,
    generator_loss,
    kl_loss,
)
from infer.lib.train.mel_processing import mel_spectrogram_torch, spec_to_mel_torch

# Constants
DEVICE_TYPE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
torch.set_default_device(DEVICE_TYPE)

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger(__name__)


def latest_checkpoint_path(dir_path: Path, pattern: str = "G_*.pth") -> Path:
    """Find the latest checkpoint file matching the given pattern.

    Args:
        dir_path: Directory to search for checkpoints
        pattern: Glob pattern to match checkpoint files

    Returns:
        Path to the latest checkpoint file
    """
    files = sorted(
        dir_path.glob(pattern),
        key=lambda f: int("".join(filter(str.isdigit, f.stem))),
    )
    latest = files[-1]
    logger.debug(f"Latest checkpoint: {latest}")
    return latest


def get_model_state_dict(
    model: torch.nn.parallel.DistributedDataParallel,
) -> dict[str, torch.Tensor]:
    """Extract state dict from model, handling DDP wrapper."""
    return model.module.state_dict() if hasattr(model, "module") else model.state_dict()


def load_model_state_dict(
    model: torch.nn.parallel.DistributedDataParallel,
    state_dict: dict[str, torch.Tensor],
) -> None:
    """Load state dict into model, handling DDP wrapper."""
    if hasattr(model, "module"):
        model.module.load_state_dict(state_dict, strict=False)
    else:
        model.load_state_dict(state_dict, strict=False)


def load_checkpoint(
    checkpoint_path: Path,
    model: torch.nn.parallel.DistributedDataParallel,
    optimizer: torch.optim.Optimizer | None = None,
    load_opt: int = 1,
) -> int:
    """Load model and optionally optimizer state from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load state into
        optimizer: Optional optimizer to load state into
        load_opt: Whether to load optimizer state (1) or not (0)

    Returns:
        Training iteration/epoch number
    """
    assert checkpoint_path.is_file()
    checkpoint_dict = torch.load(str(checkpoint_path), map_location="cpu")

    saved_state_dict = checkpoint_dict["model"]
    state_dict = get_model_state_dict(model)
    new_state_dict: dict[str, torch.Tensor] = {}

    for k, v in state_dict.items():
        try:
            new_state_dict[k] = saved_state_dict[k]
            if saved_state_dict[k].shape != v.shape:
                logger.warning(
                    "Shape mismatch for %s: expected %s, got %s",
                    k,
                    v.shape,
                    saved_state_dict[k].shape,
                )
                raise KeyError
        except Exception:
            logger.info("%s is not in the checkpoint", k)
            new_state_dict[k] = v

    load_model_state_dict(model, new_state_dict)
    logger.info("Loaded model weights")

    iteration = checkpoint_dict["iteration"]

    # If loading optimizer state fails or optimizer is None, reinitialize it.
    if optimizer is not None and load_opt == 1:
        optimizer.load_state_dict(checkpoint_dict["optimizer"])

    logger.info(f"Loaded checkpoint '{checkpoint_path}' (epoch {iteration})")

    # Clean up checkpoint dict to free memory
    del checkpoint_dict
    del saved_state_dict
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return iteration


def save_checkpoint(
    model: torch.nn.parallel.DistributedDataParallel,
    optimizer: torch.optim.Optimizer,
    learning_rate: float,
    iteration: int,
    checkpoint_path: Path,
) -> None:
    """Save model and optimizer state to checkpoint file.

    Args:
        model: Model to save
        optimizer: Optimizer to save
        learning_rate: Current learning rate
        iteration: Current training iteration/epoch
        checkpoint_path: Path to save checkpoint to
    """
    logger.info(
        f"Saving model and optimizer state at epoch {iteration} to {checkpoint_path}"
    )
    checkpoint = {
        "model": get_model_state_dict(model),
        "iteration": iteration,
        "optimizer": optimizer.state_dict(),
        "learning_rate": learning_rate,
    }
    torch.save(checkpoint, checkpoint_path)

    # Clean up to reduce memory after save
    del checkpoint
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def save_weights(
    ckpt: dict[str, torch.Tensor],
    sr: int,
    if_f0: bool,
    name: str,
    epoch: int,
    hps: params.HParams,
) -> str:
    """Extract and save model weights for inference.

    Args:
        ckpt: Model state dict
        sr: Sample rate
        if_f0: Whether model uses f0
        name: Model name for filename
        epoch: Current epoch
        hps: Hyperparameters

    Returns:
        Success message or error traceback
    """
    try:
        # Convert to half precision and exclude encoder_q for inference
        weights = {k: v.half() for k, v in ckpt.items() if "enc_q" not in k}
        config = [
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
        opt = {
            "weight": weights,
            "config": config,
            "info": f"{epoch}epoch",
            "sr": sr,
            "f0": if_f0,
        }
        save_path = Path("assets/weights") / f"{name}.pth"
        torch.save(opt, save_path)

        # Clean up
        del weights
        del opt

        return "Success."
    except Exception:
        return traceback.format_exc()


class EpochRecorder:
    """Records time elapsed between epoch events for logging."""

    def __init__(self) -> None:
        self.last_time = ttime()

    def record(self) -> str:
        """Record current time and return formatted elapsed time string."""
        now_time = ttime()
        elapsed = now_time - self.last_time
        self.last_time = now_time
        elapsed_str = str(datetime.timedelta(seconds=elapsed))
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"[{timestamp}] | ({elapsed_str})"


def main(hps: params.HParams) -> None:
    """Initialize training environment and launch training processes."""
    # Determine number of GPUs
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
    elif torch.backends.mps.is_available():
        n_gpus = 1
    else:
        logger.warning("NO GPU DETECTED: falling back to CPU - this may take a while")
        n_gpus = 1

    # Set up distributed training environment
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(random.randint(20000, 55555))

    hps.model_dir.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(hps.model_dir / "train.log")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")
    )

    logger.addHandler(file_handler)

    # Launch training processes
    processes = [
        mp.Process(target=run, args=(rank, n_gpus, hps)) for rank in range(n_gpus)
    ]
    for proc in processes:
        proc.start()
    for proc in processes:
        proc.join()


def run(rank: int, n_gpus: int, hps: params.HParams) -> None:
    """Main training loop for a single GPU process.

    Args:
        rank: GPU rank for distributed training
        n_gpus: Total number of GPUs
        hps: Hyperparameters
    """
    # Distributed setup
    dist.init_process_group(
        backend="gloo", init_method="env://", world_size=n_gpus, rank=rank
    )
    torch.manual_seed(hps.train.seed)
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)

    # Dataset and collate function
    assert hps.data.training_files is not None
    training_path = Path(hps.data.training_files)
    if hps.if_f0:
        train_dataset = TextAudioLoaderMultiNSFsid(training_path, hps.data)
        collate_fn = TextAudioCollateMultiNSFsid()
    else:
        train_dataset = TextAudioLoader(training_path, hps.data)
        collate_fn = TextAudioCollate()

    # Sampler and DataLoader
    sampler = DistributedBucketSampler(
        train_dataset,
        hps.train.batch_size * n_gpus,
        [100, 200, 300, 400, 500, 600, 700, 800, 900],
        num_replicas=n_gpus,
        rank=rank,
        shuffle=True,
    )
    loader = DataLoader(
        train_dataset,
        num_workers=4,
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_fn,
        batch_sampler=sampler,
        persistent_workers=True,
        prefetch_factor=8,
    )

    # Model initialization
    model_args = {
        "in_channels": hps.data.filter_length // 2 + 1,
        "segment_size": hps.train.segment_size // hps.data.hop_length,
        "is_half": hps.train.fp16_run,
        "spec_channels": hps.data.filter_length // 2 + 1,
        **asdict(hps.model),
    }
    if hps.if_f0:
        model_args["sr"] = hps.sample_rate
        net_g = RVC_Model_f0(**model_args)
    else:
        net_g = RVC_Model_nof0(**model_args)
    net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm)

    # Move models to device and wrap with DDP
    if torch.cuda.is_available():
        net_g = DDP(net_g.cuda(rank), device_ids=[rank])
        net_d = DDP(net_d.cuda(rank), device_ids=[rank])
    else:
        net_g = DDP(net_g)
        net_d = DDP(net_d)

    # Optimizers
    optim_params = {
        "lr": hps.train.learning_rate,
        "betas": hps.train.betas,
        "eps": hps.train.eps,
    }
    optim_g = torch.optim.AdamW(net_g.parameters(), **optim_params)
    optim_d = torch.optim.AdamW(net_d.parameters(), **optim_params)

    # Resume or load pretrained
    global_step = 0
    try:
        model_dir = Path(hps.model_dir)
        load_checkpoint(latest_checkpoint_path(model_dir, "D_*.pth"), net_d, optim_d)
        epoch_str = load_checkpoint(
            latest_checkpoint_path(model_dir, "G_*.pth"), net_g, optim_g
        )
        global_step = (epoch_str - 1) * len(loader)
        if rank == 0:
            logger.info("Loaded checkpoints")
    except Exception:
        epoch_str = 1
        global_step = 0
        if hps.pretrainG:
            if rank == 0:
                logger.info(f"Loading pretrained G: {hps.pretrainG}")
            ckpt = torch.load(hps.pretrainG, map_location="cpu")["model"]
            load_model_state_dict(net_g, ckpt)
            del ckpt  # Free memory immediately
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if hps.pretrainD:
            if rank == 0:
                logger.info(f"Loading pretrained D: {hps.pretrainD}")
            ckpt = torch.load(hps.pretrainD, map_location="cpu")["model"]
            load_model_state_dict(net_d, ckpt)
            del ckpt  # Free memory immediately
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    scaler = torch.GradScaler(enabled=hps.train.fp16_run)
    cache: list[object] = []

    # Training loop
    for epoch in range(epoch_str, hps.train.epochs + 1):
        loader.batch_sampler.set_epoch(epoch)  # type: ignore
        net_g.train()
        net_d.train()

        # Prepare data iterator with caching support
        def move_batch_to_device(batch: tuple[object, ...]) -> tuple[object, ...]:
            """Move batch tensors to GPU if available."""
            if not torch.cuda.is_available():
                return batch
            return tuple(
                x.cuda(rank, non_blocking=True) if isinstance(x, torch.Tensor) else x
                for x in batch
            )

        if hps.if_cache_data_in_gpu:
            if not cache:
                # Build cache on first epoch
                cache = [move_batch_to_device(info) for info in loader]
            random.shuffle(cache)
            data_iterator = cache
        else:
            data_iterator = loader

        epoch_recorder = EpochRecorder()

        for batch_idx, info in enumerate(data_iterator):
            # Unpack batch data
            if hps.if_f0:
                (
                    phone,
                    phone_lengths,
                    pitch,
                    pitchf,
                    spec,
                    spec_lengths,
                    wave,
                    _,
                    sid,
                ) = info
            else:
                phone, phone_lengths, spec, spec_lengths, wave, _, sid = info

            # Move to device if not cached
            if not hps.if_cache_data_in_gpu:
                batch = move_batch_to_device(info)
                if hps.if_f0:
                    (
                        phone,
                        phone_lengths,
                        pitch,
                        pitchf,
                        spec,
                        spec_lengths,
                        wave,
                        _,
                        sid,
                    ) = batch
                else:
                    (phone, phone_lengths, spec, spec_lengths, wave, _, sid) = batch

            # Forward pass
            with torch.autocast(device_type=DEVICE_TYPE, enabled=hps.train.fp16_run):
                if hps.if_f0:
                    (
                        y_hat,
                        ids_slice,
                        _x_mask,
                        z_mask,
                        (_z, z_p, m_p, logs_p, _m_q, logs_q),
                    ) = net_g(
                        phone,
                        phone_lengths,
                        pitch,
                        pitchf,
                        spec,
                        spec_lengths,
                        sid,
                    )
                else:
                    (
                        y_hat,
                        ids_slice,
                        _x_mask,
                        z_mask,
                        (_z, z_p, m_p, logs_p, _m_q, logs_q),
                    ) = net_g(phone, phone_lengths, spec, spec_lengths, sid)

                mel = spec_to_mel_torch(
                    spec,
                    hps.data.filter_length,
                    hps.data.n_mel_channels,
                    hps.data.sampling_rate,
                    hps.data.mel_fmin,
                    hps.data.mel_fmax,
                )
                y_mel = commons.slice_segments(
                    mel, ids_slice, hps.train.segment_size // hps.data.hop_length
                )
                with torch.autocast(device_type=DEVICE_TYPE, enabled=False):
                    y_hat_mel = mel_spectrogram_torch(
                        y_hat.float().squeeze(1),
                        hps.data.filter_length,
                        hps.data.n_mel_channels,
                        hps.data.sampling_rate,
                        hps.data.hop_length,
                        hps.data.win_length,
                        hps.data.mel_fmin,
                        hps.data.mel_fmax,
                    )
                if hps.train.fp16_run:
                    y_hat_mel = y_hat_mel.half()
                wave = commons.slice_segments(
                    wave, ids_slice * hps.data.hop_length, hps.train.segment_size
                )

                # Discriminator
                y_d_hat_r, y_d_hat_g, _, _ = net_d(wave, y_hat.detach())
                with torch.autocast(device_type=DEVICE_TYPE, enabled=False):
                    loss_disc = discriminator_loss(y_d_hat_r, y_d_hat_g)

            optim_d.zero_grad()
            scaler.scale(loss_disc).backward()
            scaler.unscale_(optim_d)
            scaler.step(optim_d)

            # Generator
            with torch.autocast(device_type=DEVICE_TYPE, enabled=hps.train.fp16_run):
                y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(wave, y_hat)
                with torch.autocast(device_type=DEVICE_TYPE, enabled=False):
                    loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
                    loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl
                    loss_fm = feature_loss(fmap_r, fmap_g)
                    loss_gen, _ = generator_loss(y_d_hat_g)
                    loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl

            optim_g.zero_grad()
            scaler.scale(loss_gen_all).backward()
            scaler.unscale_(optim_g)
            scaler.step(optim_g)
            scaler.update()

            global_step += 1

            # Periodic memory cleanup to reduce fragmentation
            if batch_idx % 100 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Save checkpoints
        if epoch % hps.save_every_epoch == 0 and rank == 0:
            ckpt_suffix = global_step if hps.if_latest == 0 else 2333333
            save_checkpoint(
                net_g,
                optim_g,
                hps.train.learning_rate,
                epoch,
                hps.model_dir / f"G_{ckpt_suffix}.pth",
            )
            save_checkpoint(
                net_d,
                optim_d,
                hps.train.learning_rate,
                epoch,
                hps.model_dir / f"D_{ckpt_suffix}.pth",
            )
            if hps.save_every_weights == "1":
                status = save_weights(
                    get_model_state_dict(net_g),
                    hps.sample_rate,
                    hps.if_f0,
                    f"{hps.name}_e{epoch}_s{global_step}",
                    epoch,
                    hps,
                )
                logger.info(f"saving ckpt {hps.name}_e{epoch}: {status}")

        if rank == 0:
            logger.info(f"====> Epoch: {epoch} {epoch_recorder.record()}")

        # Training complete
        if epoch >= hps.total_epoch and rank == 0:
            logger.info("Training is done. The program is closed.")
            status = save_weights(
                get_model_state_dict(net_g),
                hps.sample_rate,
                hps.if_f0,
                hps.name,
                epoch,
                hps,
            )
            logger.info(f"saving final ckpt: {status}")
            sleep(1)
            os._exit(2333333)


if __name__ == "__main__":
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

    hps = params.HParams(**config)
    hps.train = params.HParamsTrain(**config["train"])
    hps.model = params.HParamsModel(**config["model"])
    hps.data = params.HParamsData(**config["data"])
    hps.model_dir = hps.experiment_dir = experiment_dir
    hps.save_every_epoch = args.save_every_epoch
    hps.name = name
    hps.total_epoch = args.total_epoch
    hps.pretrainG = args.pretrainG
    hps.pretrainD = args.pretrainD
    hps.gpus = args.gpus
    hps.train.batch_size = args.batch_size
    hps.sample_rate = args.sample_rate
    hps.if_f0 = args.if_f0
    hps.if_latest = args.if_latest
    hps.save_every_weights = args.save_every_weights
    hps.if_cache_data_in_gpu = args.if_cache_data_in_gpu
    hps.data.training_files = f"{experiment_dir}/filelist.txt"

    os.environ["CUDA_VISIBLE_DEVICES"] = hps.gpus.replace("-", ",")

    # Disable deterministic and benchmark for cuDNN
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False

    # Use 'spawn' method for multiprocessing
    mp.set_start_method("spawn")
    main(hps)
