import datetime
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
from infer.lib.train import utils
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

# Initialize these as None; they will be set when training is actually started
hps: utils.HParams | None = None
n_gpus: int = 0
global_step = 0

DEVICE_TYPE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger(__name__)


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.parallel.DistributedDataParallel,
    optimizer: torch.optim.Optimizer | None = None,
    load_opt: int = 1,
) -> tuple[torch.nn.Module, torch.optim.Optimizer | None, float, int]:
    checkpoint_file = Path(checkpoint_path)
    assert checkpoint_file.is_file()
    checkpoint_dict = torch.load(str(checkpoint_file), map_location="cpu")

    saved_state_dict = checkpoint_dict["model"]
    # Get the current model's state dict
    state_dict = (
        model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    )
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
            # Missing in checkpoint, use model's own random value
            logger.info("%s is not in the checkpoint", k)
            new_state_dict[k] = v

    # Load the new state dict into the model
    if hasattr(model, "module"):
        model.module.load_state_dict(new_state_dict, strict=False)
    else:
        model.load_state_dict(new_state_dict, strict=False)
    logger.info("Loaded model weights")

    iteration = checkpoint_dict["iteration"]
    learning_rate = checkpoint_dict["learning_rate"]

    # If loading optimizer state fails or optimizer is None, reinitialize it.
    # If empty, may affect LR scheduler updates, so catch at the outermost train file.
    if optimizer is not None and load_opt == 1:
        optimizer.load_state_dict(checkpoint_dict["optimizer"])

    logger.info(f"Loaded checkpoint '{checkpoint_path}' (epoch {iteration})")
    return model, optimizer, learning_rate, iteration


def save_checkpoint(
    model: torch.nn.parallel.DistributedDataParallel,
    optimizer: torch.optim.Optimizer | None,
    learning_rate: float,
    iteration: int,
    checkpoint_path: Path,
) -> None:
    logger.info(
        f"Saving model and optimizer state at epoch {iteration} to {checkpoint_path}"
    )
    state_dict = (
        model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    )
    if optimizer is None:
        raise ValueError("Optimizer must not be None when saving checkpoint.")
    checkpoint = {
        "model": state_dict,
        "iteration": iteration,
        "optimizer": optimizer.state_dict(),
        "learning_rate": learning_rate,
    }
    torch.save(checkpoint, checkpoint_path)


def store_model_checkpoint(
    ckpt: dict[str, torch.Tensor],
    sr: int,
    if_f0: bool,
    name: str,
    epoch: int,
    hps: utils.HParams,
) -> str:
    try:
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
        torch.save(opt, Path("assets/weights") / f"{name}.pth")
        return "Success."
    except Exception:
        return traceback.format_exc()


class EpochRecorder:
    def __init__(self) -> None:
        self.last_time = ttime()

    def record(self) -> str:
        now_time = ttime()
        elapsed = now_time - self.last_time
        self.last_time = now_time
        elapsed_str = str(datetime.timedelta(seconds=elapsed))
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"[{timestamp}] | ({elapsed_str})"


def main() -> None:
    assert hps is not None

    # Determine number of GPUs
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
    elif torch.backends.mps.is_available():
        n_gpus = 1
    else:
        print("NO GPU DETECTED: falling back to CPU - this may take a while")
        n_gpus = 1

    # Set up distributed training environment
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(random.randint(20000, 55555))

    logger = utils.get_logger(hps.model_dir)

    # Launch training processes
    processes = [
        mp.Process(target=run, args=(rank, n_gpus, hps, logger))
        for rank in range(n_gpus)
    ]
    for proc in processes:
        proc.start()
    for proc in processes:
        proc.join()


def run(rank: int, n_gpus: int, hps: utils.HParams, logger: logging.Logger) -> None:
    global global_step

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
    bucket_sizes = [100, 200, 300, 400, 500, 600, 700, 800, 900]
    train_sampler = DistributedBucketSampler(
        train_dataset,
        hps.train.batch_size * n_gpus,
        bucket_sizes,
        num_replicas=n_gpus,
        rank=rank,
        shuffle=True,
    )
    train_loader = DataLoader(
        train_dataset,
        num_workers=4,
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_fn,
        batch_sampler=train_sampler,
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
    try:
        _, _, _, epoch_str = load_checkpoint(
            utils.latest_checkpoint_path(hps.model_dir, "D_*.pth"), net_d, optim_d
        )
        _, _, _, epoch_str = load_checkpoint(
            utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"), net_g, optim_g
        )
        global_step = (epoch_str - 1) * len(train_loader)
        if rank == 0:
            logger.info("Loaded checkpoints")
    except Exception:
        epoch_str = 1
        global_step = 0
        if hps.pretrainG:
            ckpt = torch.load(hps.pretrainG, map_location="cpu")["model"]
            (net_g.module if hasattr(net_g, "module") else net_g).load_state_dict(ckpt)
            if rank == 0:
                logger.info(f"Loaded pretrained G: {hps.pretrainG}")
        if hps.pretrainD:
            ckpt = torch.load(hps.pretrainD, map_location="cpu")["model"]
            (net_d.module if hasattr(net_d, "module") else net_d).load_state_dict(ckpt)
            if rank == 0:
                logger.info(f"Loaded pretrained D: {hps.pretrainD}")

    scaler = torch.GradScaler(enabled=hps.train.fp16_run)
    cache: list[object] = []

    # Training loop
    for epoch in range(epoch_str, hps.train.epochs + 1):
        train_loader.batch_sampler.set_epoch(epoch)  # type: ignore
        net_g.train()
        net_d.train()

        # Prepare data iterator
        def cache_batch(info: tuple[object, ...]) -> tuple[object, ...]:
            if hps.if_f0:
                (
                    phone,
                    phone_lengths,
                    pitch,
                    pitchf,
                    spec,
                    spec_lengths,
                    wave,
                    wave_lengths,
                    sid,
                ) = info
                batch = (
                    phone,
                    phone_lengths,
                    pitch,
                    pitchf,
                    spec,
                    spec_lengths,
                    wave,
                    wave_lengths,
                    sid,
                )
            else:
                phone, phone_lengths, spec, spec_lengths, wave, wave_lengths, sid = info
                batch = (
                    phone,
                    phone_lengths,
                    spec,
                    spec_lengths,
                    wave,
                    wave_lengths,
                    sid,
                )
            if torch.cuda.is_available():
                batch = tuple(
                    x.cuda(rank, non_blocking=True)
                    if isinstance(x, torch.Tensor)
                    else x
                    for x in batch
                )
            return batch

        if hps.if_cache_data_in_gpu and not cache:
            for batch_idx, info in enumerate(train_loader):  # type: ignore
                cache.append((batch_idx, cache_batch(info)))
            random.shuffle(cache)
            data_iterator = cache
        elif hps.if_cache_data_in_gpu:
            random.shuffle(cache)
            data_iterator = cache
        else:
            data_iterator = enumerate(train_loader)  # type: ignore

        epoch_recorder = EpochRecorder()

        for batch_idx, info in data_iterator:  # type: ignore
            if hps.if_f0:
                (
                    phone,
                    phone_lengths,
                    pitch,
                    pitchf,
                    spec,
                    spec_lengths,
                    wave,
                    wave_lengths,
                    sid,
                ) = info
            else:
                phone, phone_lengths, spec, spec_lengths, wave, wave_lengths, sid = info
            if not hps.if_cache_data_in_gpu and torch.cuda.is_available():
                phone = phone.cuda(rank, non_blocking=True)
                phone_lengths = phone_lengths.cuda(rank, non_blocking=True)
                if hps.if_f0:
                    assert pitch is not None and pitchf is not None
                    pitch = pitch.cuda(rank, non_blocking=True)
                    pitchf = pitchf.cuda(rank, non_blocking=True)
                sid = sid.cuda(rank, non_blocking=True)
                spec = spec.cuda(rank, non_blocking=True)
                spec_lengths = spec_lengths.cuda(rank, non_blocking=True)
                wave = wave.cuda(rank, non_blocking=True)
                wave_lengths = wave_lengths.cuda(rank, non_blocking=True)

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
                    loss_disc, _, _ = discriminator_loss(y_d_hat_r, y_d_hat_g)

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

        # Save checkpoints (inlined)
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
                ckpt = (
                    net_g.module.state_dict()
                    if hasattr(net_g, "module")
                    else net_g.state_dict()
                )
                checkpoint_filepath = store_model_checkpoint(
                    ckpt,
                    hps.sample_rate,
                    hps.if_f0,
                    f"{hps.name}_e{epoch}_s{global_step}",
                    epoch,
                    hps,
                )
                logger.info(f"saving ckpt {hps.name}_e{epoch}:{checkpoint_filepath}")

        if rank == 0:
            logger.info(f"====> Epoch: {epoch} {epoch_recorder.record()}")
        if epoch >= hps.total_epoch and rank == 0:
            logger.info("Training is done. The program is closed.")
            ckpt = (
                net_g.module.state_dict()
                if hasattr(net_g, "module")
                else net_g.state_dict()
            )
            checkpoint_filepath = store_model_checkpoint(
                ckpt, hps.sample_rate, hps.if_f0, hps.name, epoch, hps
            )
            logger.info(f"saving final ckpt:{checkpoint_filepath}")
            sleep(1)
            os._exit(2333333)


if __name__ == "__main__":
    hps = utils.get_hparams()
    os.environ["CUDA_VISIBLE_DEVICES"] = hps.gpus.replace("-", ",")
    n_gpus = len(hps.gpus.split("-"))

    # Disable deterministic and benchmark for cuDNN
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False

    # Use 'spawn' method for multiprocessing
    mp.set_start_method("spawn")
    main()
