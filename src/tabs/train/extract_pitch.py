import re
import traceback
import typing
from collections.abc import Generator
from multiprocessing import get_context
from pathlib import Path
from typing import Final, Literal

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

f0_GPU_visible: Final = not shared.config.dml

HUBERT_PATH: Final = Path("assets/hubert/hubert_base.pt")

FS: Final = 16000
HOP: Final = 160
F0_BIN: Final = 256
F0_MAX: Final = 1100.0
F0_MIN: Final = 50.0
F0_MEL_MIN: Final[float] = 1127 * np.log(1 + F0_MIN / 700)
F0_MEL_MAX: Final[float] = 1127 * np.log(1 + F0_MAX / 700)

DEVICE = "cpu"
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
elif shared.config.dml:
    import torch_directml  # type: ignore

    DEVICE = torch_directml.device(torch_directml.default_device())


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

    def __init__(self, log_file: Path):
        self.log_file = log_file

        self.model_rmvpe = None

    def _printt(self, message: str):
        """Log message to both console and file."""
        print(message)
        _write_to_log(self.log_file, message)

    def _compute_f0(
        self,
        path: Path,
        f0_method: Literal["pm", "harvest", "dio", "rmvpe"],
        is_half: bool = False,
        device: str = "cpu",
    ) -> np.ndarray:
        """Compute F0 using the specified method."""
        x = load_audio(path, FS)
        p_len = x.shape[0] // HOP

        match f0_method:
            case "pm":
                time_step = 160 / 16000 * 1000
                f0 = (
                    parselmouth.Sound(x, FS)
                    .to_pitch_ac(
                        time_step=time_step / 1000,
                        voicing_threshold=0.6,
                        pitch_floor=F0_MIN,
                        pitch_ceiling=F0_MAX,
                    )
                    .selected_array["frequency"]
                )
                pad_size = (p_len - len(f0) + 1) // 2
                if pad_size > 0 or p_len - len(f0) - pad_size > 0:
                    f0 = np.pad(
                        f0, [[pad_size, p_len - len(f0) - pad_size]], mode="constant"
                    )
            case "harvest":
                f0, t = pyworld.harvest(  # type: ignore
                    x.astype(np.double),
                    fs=FS,
                    f0_ceil=F0_MAX,
                    f0_floor=F0_MIN,
                    frame_period=1000 * HOP / FS,
                )
                f0 = pyworld.stonemask(x.astype(np.double), f0, t, FS)  # type: ignore
            case "dio":
                f0, t = pyworld.dio(  # type: ignore
                    x.astype(np.double),
                    fs=FS,
                    f0_ceil=F0_MAX,
                    f0_floor=F0_MIN,
                    frame_period=1000 * HOP / FS,
                )
                f0 = pyworld.stonemask(x.astype(np.double), f0, t, FS)  # type: ignore
            case "rmvpe":
                if self.model_rmvpe is None:
                    self._printt("Loading rmvpe model")
                    self.model_rmvpe = RMVPE(
                        "assets/rmvpe/rmvpe.pt", is_half=is_half, device=device
                    )
                f0 = self.model_rmvpe.infer_from_audio(x, thred=0.03)
            case _:
                typing.assert_never(f0_method)
                raise ValueError(f"Unknown f0_method: {f0_method}")

        return f0

    def _coarse_f0(self, f0: np.ndarray) -> np.ndarray:
        """Convert F0 to coarse representation."""
        f0_mel = 1127 * np.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - F0_MEL_MIN) * (F0_BIN - 2) / (
            F0_MEL_MAX - F0_MEL_MIN
        ) + 1

        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > F0_BIN - 1] = F0_BIN - 1
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
            self._printt("no-f0-todo")
            return
        self._printt(f"todo-f0-{len(paths)}")
        n = max(len(paths) // 5, 1)
        for idx, (inp_path, opt_path1, opt_path2) in enumerate(paths):
            try:
                if idx % n == 0:
                    self._printt(f"f0ing,now-{idx},all-{len(paths)},-{inp_path}")
                if (
                    opt_path1.with_suffix(".npy").exists()
                    and opt_path2.with_suffix(".npy").exists()
                ):
                    continue
                featur_pit = self._compute_f0(inp_path, f0_method, is_half, device)
                np.save(opt_path2, featur_pit, allow_pickle=False)  # nsf
                coarse_pit = self._coarse_f0(featur_pit)
                np.save(opt_path1, coarse_pit, allow_pickle=False)  # ori
            except Exception:
                self._printt(f"f0fail-{idx}-{inp_path}-{traceback.format_exc()}")


class FeatureExtractor:
    """Handles HuBERT feature extraction."""

    def __init__(self, exp_dir: Path, log_file: Path):
        self.exp_dir = Path(exp_dir)
        self.log_file = Path(log_file)
        self.model = None
        self.saved_cfg = None

    def _printt(self, message: str):
        """Log message to both console and file."""
        print(message)
        _write_to_log(self.log_file, message)

    def load_model(self, model_path: Path, device: str, is_half: bool) -> bool:
        """Load HuBERT model."""
        if not model_path.exists():
            self._printt(
                f"Error: Extracting is shut down because {model_path} does not exist, "
                "you may download it from https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main"
            )
            return False

        try:
            self._printt(f"load model(s) from {model_path}")
            with safe_globals([Dictionary]):
                models, saved_cfg, _task = (
                    fairseq.checkpoint_utils.load_model_ensemble_and_task(
                        [str(model_path)],
                        suffix="",
                    )
                )
            self.model = models[0]
            self.saved_cfg = saved_cfg

            self.model = self.model.to(device)
            self._printt(f"move model to {device}")

            if is_half and device not in ["mps", "cpu"]:
                self.model = self.model.half()

            self.model.eval()
            return True
        except Exception:
            self._printt(f"Error loading model: {traceback.format_exc()}")
            return False

    def _read_wave(self, wav_path: Path, normalize: bool = False) -> torch.Tensor:
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
        wav_path = self.exp_dir / shared.WAVS_16K_DIR_NAME
        out_path = (
            self.exp_dir / shared.FEATURE_DIR_NAME
            if version == "v1"
            else self.exp_dir / shared.FEATURE_DIR_NAME_V2
        )
        out_path.mkdir(parents=True, exist_ok=True)

        n = max(1, len(file_list) // 10)
        if not file_list:
            self._printt("no-feature-todo")
            return
        self._printt(f"all-feature-{len(file_list)}")
        for idx, file in enumerate(file_list):
            try:
                if file.suffix != ".wav":
                    continue
                wav_path = wav_path / file.name
                out_path = out_path / file.name.replace("wav", "npy")

                if out_path.exists():
                    continue

                assert self.saved_cfg is not None
                feats = self._read_wave(
                    wav_path, normalize=self.saved_cfg.task.normalize
                )
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
                    self._printt(f"{file}-contains nan")

                if idx % n == 0:
                    self._printt(f"now-{len(file_list)},all-{idx},{file},{feats.shape}")
            except Exception:
                self._printt(traceback.format_exc())

        self._printt("all-feature-done")


def _extract_features_worker(
    file_list: list[Path],
    exp_dir: Path,
    log_file: Path,
    device: str,
    version: Literal["v1", "v2"],
    is_half: bool,
) -> None:
    """Worker function to extract features in a child process. Loads model fresh."""
    feature_extractor = FeatureExtractor(exp_dir, log_file)

    # Load model fresh in this process
    if not feature_extractor.load_model(HUBERT_PATH, device, is_half):
        return

    feature_extractor.extract_features_batch(file_list, device, version, is_half)


def _extract_f0_feature(
    gpus_str: str,
    n_p: int,
    f0_method: Literal["pm", "harvest", "dio", "rmvpe", "rmvpe_gpu"],
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
    log_file.touch()

    # Extract F0 if needed
    if if_f0:
        _write_to_log(log_file, f"Starting F0 extraction with method: {f0_method}")

        f0_extractor = F0FeatureExtractor(log_file)
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
            paths.append((inp_path, opt_root1 / name, opt_root2 / name))

        # Run F0 extraction with multiprocessing
        if f0_method != "rmvpe_gpu":
            # Multi-process for CPU-based methods
            ctx = get_context("spawn")
            ps = [
                ctx.Process(
                    target=f0_extractor.extract_f0_batch,
                    args=(paths[i::n_p], f0_method, False, "cpu"),
                )
                for i in range(n_p)
            ]
            for p in ps:
                p.start()
            for p in ps:
                p.join()
        else:
            # Multi-GPU or multi-device for RMVPE
            if gpus_rmvpe != "-":
                gpus_rmvpe_list = gpus_rmvpe.split("-")
                ctx = get_context("spawn")
                ps = [
                    ctx.Process(
                        target=f0_extractor.extract_f0_batch,
                        args=(
                            paths[idx :: len(gpus_rmvpe_list)],
                            "rmvpe",
                            shared.config.is_half,
                            f"cuda:{gpu_id}" if DEVICE == "cuda" else "cpu",
                        ),
                    )
                    for idx, gpu_id in enumerate(gpus_rmvpe_list)
                ]
                for p in ps:
                    p.start()
                for p in ps:
                    p.join()
            else:
                # DML device
                f0_extractor.extract_f0_batch(paths, "rmvpe", False, DEVICE)

        _write_to_log(log_file, "F0 extraction completed")
        log_content = log_file.read_text(encoding="utf-8")
        update_progress(log_content)
        yield log_content

    # Feature extraction
    _write_to_log(log_file, f"Starting feature extraction with version: {version}")

    wav_path = log_dir_path / shared.WAVS_16K_DIR_NAME
    all_files = sorted(wav_path.iterdir())

    # Split files across processes
    gpus = gpus_str.split("-")
    ctx = get_context("spawn")
    ps: list = []

    for idx, gpu_id in enumerate(gpus):
        device_for_extraction = DEVICE
        if DEVICE == "cuda":
            device_for_extraction = f"cuda:{gpu_id}"

        file_subset = all_files[idx :: len(gpus)]
        p = ctx.Process(
            target=_extract_features_worker,
            args=(
                file_subset,
                log_dir_path,
                log_file,
                device_for_extraction,
                version,
                shared.config.is_half,
            ),
        )
        ps.append(p)
        p.start()

    for p in ps:
        p.join()

    _write_to_log(log_file, "Feature extraction completed")
    log_content = log_file.read_text(encoding="utf-8")
    update_progress(log_content)
    yield log_content


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
            gpus = gr.Textbox(
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
            f0_method = gr.Radio(
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
            f0_method.change(
                fn=_change_f0_method,
                inputs=[f0_method],
                outputs=[gpus_rmvpe],
            )
        with gr.Column():
            extract_f0_btn = gr.Button(i18n("Extract"), variant="primary")
            info = gr.Textbox(label=i18n("Info"), value="", max_lines=8)

            extract_f0_btn.click(
                _extract_f0_feature,
                [
                    gpus,
                    cpu_count,
                    f0_method,
                    use_f0,
                    experiment_name,
                    model_version,
                    gpus_rmvpe,
                ],
                [info],
                api_name="train_extract_f0_feature",
            )

    return f0_method, gpus_rmvpe
