import logging
import traceback
from pathlib import Path

import ffmpeg
import numpy as np
import parselmouth
import pyworld
import soundfile as sf
import torch
import torch.nn.functional as F
from fairseq import checkpoint_utils

from configs.config import Config

logger = logging.getLogger(__name__)

config = Config()


def _load_audio(file: Path, sr: int):
    """Load audio file using ffmpeg."""
    try:
        out, _ = (
            ffmpeg.input(str(file), threads=0)
            .output("-", format="f32le", acodec="pcm_f32le", ac=1, ar=sr)
            .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
        )
    except Exception as e:
        traceback.print_exc()
        raise RuntimeError(f"Failed to load audio: {e}") from e
    return np.frombuffer(out, np.float32).flatten()


class FeatureExtractor:
    """Extract pitch (f0) and model features from audio files."""

    def __init__(self, sample_rate=16000, hop_size=160):
        self.sr = sample_rate
        self.hop = hop_size
        self.f0_bin = 256
        self.f0_max = 1100.0
        self.f0_min = 50.0
        self.f0_mel_min = 1127 * np.log(1 + self.f0_min / 700)
        self.f0_mel_max = 1127 * np.log(1 + self.f0_max / 700)

    def _compute_f0(self, path: Path, f0_method: str, device="cpu"):
        """Compute f0 using various methods."""
        x = _load_audio(path, self.sr)
        p_len = x.shape[0] // self.hop

        if f0_method == "pm":
            time_step = 160 / 16000 * 1000
            f0 = (
                parselmouth.Sound(x, self.sr)
                .to_pitch_ac(
                    time_step=time_step / 1000,
                    voicing_threshold=0.6,
                    pitch_floor=self.f0_min,
                    pitch_ceiling=self.f0_max,
                )
                .selected_array["frequency"]
            )
            pad_size = (p_len - len(f0) + 1) // 2
            if pad_size > 0 or p_len - len(f0) - pad_size > 0:
                f0 = np.pad(
                    f0, [[pad_size, p_len - len(f0) - pad_size]], mode="constant"
                )
        elif f0_method == "harvest":
            f0, t = pyworld.harvest(
                x.astype(np.double),
                fs=self.sr,
                f0_ceil=self.f0_max,
                f0_floor=self.f0_min,
                frame_period=1000 * self.hop / self.sr,
            )
            f0 = pyworld.stonemask(x.astype(np.double), f0, t, self.sr)
        elif f0_method == "dio":
            f0, t = pyworld.dio(
                x.astype(np.double),
                fs=self.sr,
                f0_ceil=self.f0_max,
                f0_floor=self.f0_min,
                frame_period=1000 * self.hop / self.sr,
            )
            f0 = pyworld.stonemask(x.astype(np.double), f0, t, self.sr)
        elif f0_method in ["rmvpe", "rmvpe_gpu"]:
            if not hasattr(self, "model_rmvpe"):
                from infer.lib.rmvpe import RMVPE

                logger.info("Loading rmvpe model")
                self.model_rmvpe = RMVPE(
                    "assets/rmvpe/rmvpe.pt", is_half=config.is_half, device=device
                )
            f0 = self.model_rmvpe.infer_from_audio(x, thred=0.03)
        else:
            raise ValueError(f"Unknown f0 method: {f0_method}")

        return f0

    def _coarse_f0(self, f0: np.ndarray):
        """Convert f0 to coarse representation."""
        f0_mel = 1127 * np.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - self.f0_mel_min) * (
            self.f0_bin - 2
        ) / (self.f0_mel_max - self.f0_mel_min) + 1

        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > self.f0_bin - 1] = self.f0_bin - 1
        f0_coarse = np.rint(f0_mel).astype(int)
        assert f0_coarse.max() <= 255 and f0_coarse.min() >= 1, (
            f0_coarse.max(),
            f0_coarse.min(),
        )
        return f0_coarse

    def extract_f0_for_files(
        self,
        file_paths: list,
        f0_method: str,
        log_file: Path,
        device="cpu",
    ):
        """Extract f0 features for a list of audio files."""
        if len(file_paths) == 0:
            msg = "no-f0-todo"
            logger.info(msg)
            with log_file.open("a") as f:
                f.write(f"{msg}\n")
            return

        msg = f"todo-f0-{len(file_paths)}"
        logger.info(msg)
        with log_file.open("a") as f:
            f.write(f"{msg}\n")

        n = max(len(file_paths) // 5, 1)
        for idx, (inp_path, opt_path1, opt_path2) in enumerate(file_paths):
            try:
                if idx % n == 0:
                    msg = f"f0ing,now-{idx},all-{len(file_paths)},-{inp_path}"
                    logger.info(msg)
                    with log_file.open("a") as f:
                        f.write(f"{msg}\n")

                if (
                    opt_path1.with_suffix(".npy").exists()
                    and opt_path2.with_suffix(".npy").exists()
                ):
                    continue

                featur_pit = self._compute_f0(inp_path, f0_method, device)
                np.save(opt_path2, featur_pit, allow_pickle=False)  # nsf
                coarse_pit = self._coarse_f0(featur_pit)
                np.save(opt_path1, coarse_pit, allow_pickle=False)  # ori
            except Exception:
                msg = f"f0fail-{idx}-{inp_path}-{traceback.format_exc()}"
                logger.error(msg)
                with log_file.open("a") as f:
                    f.write(f"{msg}\n")

    def extract_model_features_for_files(
        self,
        file_paths: list,
        log_file: Path,
        version: str,
    ):
        """Extract HuBERT model features for audio files."""
        if len(file_paths) == 0:
            msg = "no-feature-todo"
            logger.info(msg)
            with log_file.open("a") as f:
                f.write(f"{msg}\n")
            return

        device = "cuda" if torch.cuda.is_available() else "cpu"

        model_path = "assets/hubert/hubert_base.pt"
        if not Path(model_path).exists():
            msg = f"Error: {model_path} does not exist. Download from https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main"
            logger.error(msg)
            with log_file.open("a") as f:
                f.write(f"{msg}\n")
            return

        msg = f"Loading model from {model_path}"
        logger.info(msg)
        with log_file.open("a") as f:
            f.write(f"{msg}\n")
        models, saved_cfg, _ = checkpoint_utils.load_model_ensemble_and_task(
            [model_path], suffix=""
        )
        assert saved_cfg is not None, "Failed to load model configuration."

        model = models[0].to(device)
        if config.is_half and device not in ["mps", "cpu"]:
            model = model.half()
        model.eval()

        msg = f"all-feature-{len(file_paths)}"
        logger.info(msg)
        with log_file.open("a") as f:
            f.write(f"{msg}\n")

        n = max(1, len(file_paths) // 10)
        for idx, (wav_path, out_path) in enumerate(file_paths):
            try:
                if out_path.exists():
                    continue

                # Read audio
                wav, sr = sf.read(str(wav_path))
                assert sr == 16000, f"Sample rate must be 16000, got {sr}"
                feats = torch.from_numpy(wav).float()
                if feats.dim() == 2:
                    feats = feats.mean(-1)
                assert feats.dim() == 1
                with torch.no_grad():
                    feats = (
                        F.layer_norm(feats, feats.shape)
                        if saved_cfg.task.normalize
                        else feats
                    )
                feats = feats.view(1, -1)

                # Extract features
                padding_mask = torch.BoolTensor(feats.shape).fill_(False)
                inputs = {
                    "source": (
                        feats.half().to(device)
                        if config.is_half and device not in ["mps", "cpu"]
                        else feats.to(device)
                    ),
                    "padding_mask": padding_mask.to(device),
                    "output_layer": 9 if version == "v1" else 12,
                }
                with torch.no_grad():
                    logits = model.extract_features(**inputs)
                    feats_out = (
                        model.final_proj(logits[0]) if version == "v1" else logits[0]
                    )

                feats_out = feats_out.squeeze(0).float().cpu().numpy()
                if np.isnan(feats_out).sum() == 0:
                    np.save(out_path, feats_out, allow_pickle=False)
                else:
                    msg = f"{wav_path.name}-contains nan"
                    logger.warning(msg)
                    with log_file.open("a") as f:
                        f.write(f"{msg}\n")

                if idx % n == 0:
                    msg = f"now-{idx},all-{len(file_paths)},{wav_path.name},{feats_out.shape}"
                    logger.info(msg)
                    with log_file.open("a") as f:
                        f.write(f"{msg}\n")
            except Exception:
                msg = f"feature-fail-{idx}-{wav_path}-{traceback.format_exc()}"
                logger.error(msg)
                with log_file.open("a") as f:
                    f.write(f"{msg}\n")

        msg = "all-feature-done"
        logger.info(msg)
        with log_file.open("a") as f:
            f.write(f"{msg}\n")
