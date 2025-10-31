import re
import traceback
from collections.abc import Generator
from multiprocessing import Process
from pathlib import Path
from typing import Literal

import fairseq.checkpoint_utils
import gradio as gr
import numpy as np
import parselmouth
import pyworld
import soundfile as sf
import torch
import torch.nn.functional as F
from fairseq.data.dictionary import Dictionary
from torch.serialization import safe_globals

import shared
from infer.lib.audio import load_audio
from infer.lib.rmvpe import RMVPE
from shared import i18n

f0_GPU_visible = not shared.config.dml


def _write_to_log(log_file: Path, message: str):
    """Write message to log file and print it."""
    print(message)
    current_content = log_file.read_text(encoding="utf-8") if log_file.exists() else ""
    log_file.write_text(current_content + f"{message}\n", encoding="utf-8")


def _parse_f0_feature_log(content: str) -> tuple[int, int]:
    """
    Parses log content to extract the highest 'now' and 'all' values from lines matching the pattern:
    'f0ing,now-<number>,all-<number>,...'
    """
    max_now = 0
    max_all = 1
    pattern = re.compile(r"f0ing,now-(\d+),all-(\d+)")

    for line in content.splitlines():
        match = pattern.search(line)
        if match:
            try:
                current_now = int(match.group(1))
                current_all = int(match.group(2))
                max_now = max(max_now, current_now)
                max_all = max(max_all, current_all)
            except ValueError:
                print(f"Warning: Could not parse numbers from line: {line}")

    return max_now, max_all


class F0FeatureExtractor:
    """Handles F0 and feature extraction with support for multiple methods."""

    def __init__(self, exp_dir: str, log_file: Path):
        self.exp_dir = exp_dir
        self.log_file = log_file
        self.fs = 16000
        self.hop = 160
        self.f0_bin = 256
        self.f0_max = 1100.0
        self.f0_min = 50.0
        self.f0_mel_min = 1127 * np.log(1 + self.f0_min / 700)
        self.f0_mel_max = 1127 * np.log(1 + self.f0_max / 700)
        self.model_rmvpe = None

    def printt(self, message: str):
        """Log message to both console and file."""
        print(message)
        _write_to_log(self.log_file, message)

    def compute_f0(
        self,
        path: Path,
        f0_method: Literal["pm", "harvest", "dio", "rmvpe"],
        is_half: bool = False,
        device: str = "cpu",
    ) -> np.ndarray:
        """Compute F0 using the specified method."""
        x = load_audio(path, self.fs)
        p_len = x.shape[0] // self.hop

        if f0_method == "pm":
            time_step = 160 / 16000 * 1000
            f0 = (
                parselmouth.Sound(x, self.fs)
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
            f0, t = pyworld.harvest(  # type: ignore
                x.astype(np.double),
                fs=self.fs,
                f0_ceil=self.f0_max,
                f0_floor=self.f0_min,
                frame_period=1000 * self.hop / self.fs,
            )
            f0 = pyworld.stonemask(x.astype(np.double), f0, t, self.fs)  # type: ignore
        elif f0_method == "dio":
            f0, t = pyworld.dio(  # type: ignore
                x.astype(np.double),
                fs=self.fs,
                f0_ceil=self.f0_max,
                f0_floor=self.f0_min,
                frame_period=1000 * self.hop / self.fs,
            )
            f0 = pyworld.stonemask(x.astype(np.double), f0, t, self.fs)  # type: ignore
        elif f0_method == "rmvpe":
            if self.model_rmvpe is None:
                self.printt("Loading rmvpe model")
                self.model_rmvpe = RMVPE(
                    "assets/rmvpe/rmvpe.pt", is_half=is_half, device=device
                )
            f0 = self.model_rmvpe.infer_from_audio(x, thred=0.03)
        else:
            raise ValueError(f"Unknown f0_method: {f0_method}")

        return f0

    def coarse_f0(self, f0: np.ndarray) -> np.ndarray:
        """Convert F0 to coarse representation."""
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

    def extract_f0_batch(
        self,
        paths: list[tuple[Path, Path, Path]],
        f0_method: Literal["pm", "harvest", "dio", "rmvpe"],
        is_half: bool = False,
        device: str = "cpu",
    ):
        """Extract F0 for a batch of files."""
        if not paths:
            self.printt("no-f0-todo")
            return
        self.printt(f"todo-f0-{len(paths)}")
        n = max(len(paths) // 5, 1)
        for idx, (inp_path, opt_path1, opt_path2) in enumerate(paths):
            try:
                if idx % n == 0:
                    self.printt(f"f0ing,now-{idx},all-{len(paths)},-{inp_path}")
                if (
                    opt_path1.with_suffix(".npy").exists()
                    and opt_path2.with_suffix(".npy").exists()
                ):
                    continue
                featur_pit = self.compute_f0(inp_path, f0_method, is_half, device)
                np.save(opt_path2, featur_pit, allow_pickle=False)  # nsf
                coarse_pit = self.coarse_f0(featur_pit)
                np.save(opt_path1, coarse_pit, allow_pickle=False)  # ori
            except Exception:
                self.printt(f"f0fail-{idx}-{inp_path}-{traceback.format_exc()}")


class FeatureExtractor:
    """Handles HuBERT feature extraction."""

    def __init__(self, exp_dir: str, log_file: Path):
        self.exp_dir = Path(exp_dir)
        self.log_file = Path(log_file)
        self.model = None
        self.saved_cfg = None

    def printt(self, message: str):
        """Log message to both console and file."""
        print(message)
        _write_to_log(self.log_file, message)

    def load_model(self, model_path: Path, device: str, is_half: bool) -> bool:
        """Load HuBERT model."""
        if not model_path.exists():
            self.printt(
                f"Error: Extracting is shut down because {model_path} does not exist, "
                "you may download it from https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main"
            )
            return False

        try:
            self.printt(f"load model(s) from {model_path}")
            with safe_globals([Dictionary]):
                models, saved_cfg, _task = (
                    fairseq.checkpoint_utils.load_model_ensemble_and_task(
                        [model_path],
                        suffix="",
                    )
                )
            self.model = models[0]
            self.saved_cfg = saved_cfg

            self.model = self.model.to(device)
            self.printt(f"move model to {device}")

            if is_half and device not in ["mps", "cpu"]:
                self.model = self.model.half()

            self.model.eval()
            return True
        except Exception:
            self.printt(f"Error loading model: {traceback.format_exc()}")
            return False

    def readwave(self, wav_path: Path, normalize: bool = False) -> torch.Tensor:
        """Read and prepare audio file."""
        wav, sr = sf.read(wav_path)
        assert sr == 16000, f"Sample rate must be 16000, got {sr}"

        feats = torch.from_numpy(wav).float()
        if feats.dim() == 2:
            feats = feats.mean(-1)
        assert feats.dim() == 1, feats.dim()

        if normalize:
            with torch.no_grad():
                feats = F.layer_norm(feats, feats.shape)

        feats = feats.view(1, -1)
        return feats

    def extract_features_batch(
        self,
        file_list: list[Path],
        device: str,
        version: Literal["v1", "v2"],
        is_half: bool,
    ):
        """Extract features for a batch of files."""
        wavPath = self.exp_dir / shared.WAVS_16K_DIR_NAME
        outPath = (
            self.exp_dir / shared.FEATURE_DIR_NAME
            if version == "v1"
            else self.exp_dir / shared.FEATURE_DIR_NAME_V2
        )
        outPath.mkdir(parents=True, exist_ok=True)

        n = max(1, len(file_list) // 10)
        if not file_list:
            self.printt("no-feature-todo")
            return
        self.printt(f"all-feature-{len(file_list)}")
        for idx, file in enumerate(file_list):
            try:
                if file.suffix != ".wav":
                    continue
                wav_path = wavPath / file.name
                out_path = outPath / file.name.replace("wav", "npy")

                if out_path.exists():
                    continue

                assert self.saved_cfg is not None
                feats = self.readwave(wav_path, normalize=self.saved_cfg.task.normalize)
                padding_mask = torch.BoolTensor(feats.shape).fill_(False)

                inputs = {
                    "source": (
                        feats.half().to(device)
                        if is_half and device not in ["mps", "cpu"]
                        else feats.to(device)
                    ),
                    "padding_mask": padding_mask.to(device),
                    "output_layer": 9 if version == "v1" else 12,
                }

                with torch.no_grad():
                    assert self.model is not None
                    logits = self.model.extract_features(**inputs)
                    feats = (
                        self.model.final_proj(logits[0])
                        if version == "v1"
                        else logits[0]
                    )

                feats = feats.squeeze(0).float().cpu().numpy()
                if np.isnan(feats).sum() == 0:
                    np.save(out_path, feats, allow_pickle=False)
                else:
                    self.printt(f"{file}-contains nan")

                if idx % n == 0:
                    self.printt(f"now-{len(file_list)},all-{idx},{file},{feats.shape}")
            except Exception:
                self.printt(traceback.format_exc())

        self.printt("all-feature-done")


def _extract_f0_feature(
    gpus_str: str,
    n_p: int,
    f0method: str,
    if_f0: bool,
    exp_dir: str,
    version: Literal["v1", "v2"],
    gpus_rmvpe: str,
    progress: gr.Progress = gr.Progress(),
) -> Generator[str]:
    """Extract F0 and feature from audio files efficiently using multiprocessing."""

    def update_progress(content: str):
        now, all_count = _parse_f0_feature_log(content)
        if all_count > 0:
            progress(
                float(now) / all_count, desc=f"{now}/{all_count} Features extracted..."
            )

    log_dir_path = Path.cwd() / "logs" / exp_dir
    log_dir_path.mkdir(parents=True, exist_ok=True)
    log_file = log_dir_path / "extract_f0_feature.log"
    log_file.write_text("", encoding="utf-8")

    # Extract F0 if needed
    if if_f0:
        _write_to_log(log_file, f"Starting F0 extraction with method: {f0method}")

        f0_extractor = F0FeatureExtractor(str(log_dir_path), log_file)
        inp_root = log_dir_path / shared.WAVS_16K_DIR_NAME
        opt_root1 = log_dir_path / shared.F0_DIR_NAME
        opt_root2 = log_dir_path / shared.F0_NSF_DIR_NAME

        opt_root1.mkdir(parents=True, exist_ok=True)
        opt_root2.mkdir(parents=True, exist_ok=True)

        paths: list[tuple[Path, Path, Path]] = []
        for inp_path in sorted(inp_root.iterdir()):
            if "spec" in str(inp_path):
                continue
            name = inp_path.name
            opt_path1 = opt_root1 / name
            opt_path2 = opt_root2 / name
            paths.append((inp_path, opt_path1, opt_path2))

        # Determine device for RMVPE
        rmvpe_device = "cpu"
        if f0method == "rmvpe_gpu" and torch.cuda.is_available():
            rmvpe_device = "cuda"
        elif f0method == "rmvpe_gpu" and shared.config.dml:
            import torch_directml  # type: ignore  # noqa: PLC0415

            rmvpe_device = torch_directml.device(torch_directml.default_device())

        # Run F0 extraction with multiprocessing
        if f0method != "rmvpe_gpu":
            # Multi-process for CPU-based methods
            ps = []
            for i in range(n_p):
                p = Process(
                    target=f0_extractor.extract_f0_batch,
                    args=(paths[i::n_p], f0method, False, "cpu"),
                )
                ps.append(p)
                p.start()
            for p in ps:
                p.join()
        else:
            # Multi-GPU or multi-device for RMVPE
            if gpus_rmvpe != "-":
                gpus_rmvpe_list = gpus_rmvpe.split("-")
                ps = []
                for idx, gpu_id in enumerate(gpus_rmvpe_list):
                    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
                    p = Process(
                        target=f0_extractor.extract_f0_batch,
                        args=(
                            paths[idx :: len(gpus_rmvpe_list)],
                            "rmvpe",
                            shared.config.is_half,
                            device,
                        ),
                    )
                    ps.append(p)
                    p.start()
                for p in ps:
                    p.join()
            else:
                # DML device
                f0_extractor.extract_f0_batch(paths, "rmvpe", False, rmvpe_device)

        _write_to_log(log_file, "F0 extraction completed")
        yield log_file.read_text(encoding="utf-8")

    # Feature extraction
    _write_to_log(log_file, f"Starting feature extraction with version: {version}")

    feature_extractor = FeatureExtractor(str(log_dir_path), log_file)
    model_path = Path("assets/hubert/hubert_base.pt")

    # Determine device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    elif shared.config.dml:
        import torch_directml  # type: ignore  # noqa: PLC0415

        device = torch_directml.device(torch_directml.default_device())

    # Load model
    if not feature_extractor.load_model(model_path, device, shared.config.is_half):
        yield log_file.read_text(encoding="utf-8")
        return

    wavPath = log_dir_path / shared.WAVS_16K_DIR_NAME
    all_files = sorted(wavPath.iterdir())

    # Split files across processes
    gpus = gpus_str.split("-")
    ps = []

    for idx, gpu_id in enumerate(gpus):
        device_for_extraction = "cpu"
        if torch.cuda.is_available():
            device_for_extraction = f"cuda:{gpu_id}"
        elif shared.config.dml:
            import torch_directml  # type: ignore # noqa: PLC0415

            device_for_extraction = torch_directml.device(
                torch_directml.default_device()
            )

        file_subset = all_files[idx :: len(gpus)]
        p = Process(
            target=feature_extractor.extract_features_batch,
            args=(file_subset, device_for_extraction, version, shared.config.is_half),
        )
        ps.append(p)
        p.start()

    for p in ps:
        p.join()

    _write_to_log(log_file, "Feature extraction completed")
    yield log_file.read_text(encoding="utf-8")


def _change_f0_method(f0_method: str):
    """Update GPU visibility based on F0 method."""
    return {
        "visible": f0_GPU_visible if f0_method == "rmvpe_gpu" else False,
        "__type__": "update",
    }


def extract_pitch_config(
    experiment_name: gr.Textbox,
    use_f0: gr.Radio,
    model_version: gr.Radio,
    cpu_count: gr.Slider,
):
    """Create UI components for pitch extraction configuration."""
    gr.Markdown(value=i18n("## Extract Pitch"))
    with gr.Row():
        with gr.Column():
            gpus6 = gr.Textbox(
                label=i18n("以-分隔输入使用的卡号, 例如   0-1-2   使用卡0和卡1和卡2"),
                value=shared.gpus,
                interactive=True,
                visible=f0_GPU_visible,
            )
            gr.Textbox(
                label=i18n("GPU Info"),
                value=shared.gpu_info,
                visible=f0_GPU_visible,
            )
        with gr.Column():
            gr.Markdown(
                value=i18n(
                    """### Select pitch extraction algorithm:
                - PM speeds up vocal input.
                - DIO speeds up high-quality speech on weaker CPUs.
                - Harvest is higher quality but slower.
                - RMVPE is the best and slightly CPU/GPU-intensive."""
                )
            )
            f0method8 = gr.Radio(
                label="Method",
                choices=["pm", "harvest", "dio", "rmvpe", "rmvpe_gpu"],
                value="rmvpe_gpu",
                interactive=True,
            )
            gpus_rmvpe = gr.Textbox(
                label=i18n(
                    "rmvpe卡号配置: 以-分隔输入使用的不同进程卡号,例如0-0-1使用在卡0上跑2个进程并在卡1上跑1个进程"
                ),
                value=f"{shared.gpus}-{shared.gpus}",
                interactive=True,
                visible=f0_GPU_visible,
            )
        with gr.Column():
            extract_f0_btn = gr.Button(i18n("Extract"), variant="primary")
            info2 = gr.Textbox(label=i18n("Info"), value="", max_lines=8)
            f0method8.change(
                fn=_change_f0_method,
                inputs=[f0method8],
                outputs=[gpus_rmvpe],
            )
            extract_f0_btn.click(
                _extract_f0_feature,
                [
                    gpus6,
                    cpu_count,
                    f0method8,
                    use_f0,
                    experiment_name,
                    model_version,
                    gpus_rmvpe,
                ],
                [info2],
                api_name="train_extract_f0_feature",
            )

    return f0method8, gpus_rmvpe
