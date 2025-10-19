import logging
import multiprocessing
import os
import shutil
import sys
import traceback
import warnings
from collections.abc import Generator
from multiprocessing import Process
from pathlib import Path
from subprocess import Popen
from time import sleep

import ffmpeg
import gradio as gr
import numpy as np
import parselmouth
import soundfile as sf
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from fairseq import checkpoint_utils
from fairseq.modules.grad_multiply import GradMultiply

import pyworld
from configs.config import Config
from i18n.i18n import I18nAuto

cwd = Path.cwd()
sys.path.append(str(cwd))
load_dotenv()

logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# Cleanup runtime packages
for pack_dir in ["infer_pack", "uvr5_pack"]:
    shutil.rmtree(cwd / f"runtime/Lib/site-packages/{pack_dir}", ignore_errors=True)


warnings.filterwarnings("ignore")
torch.manual_seed(114514)

multiprocessing.set_start_method("spawn", force=True)
config = Config()


if config.dml:

    def forward_dml(ctx, x, scale):
        ctx.scale = scale
        res = x.clone().detach()
        return res

    GradMultiply.forward = forward_dml

i18n = I18nAuto()
logger.info(i18n)


def get_gpu_info():
    n_gpu = torch.cuda.device_count()
    details = []
    if torch.cuda.is_available() and n_gpu > 0:
        for i in range(n_gpu):
            name = torch.cuda.get_device_name(i)
            details.append(f"{i}\t{name}")
    info = "\n".join(details)
    gpus = "-".join(info[0] for info in details)
    return info, gpus


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


class _FeatureExtractor:
    """Extract pitch (f0) and model features from audio files."""

    def __init__(self, sample_rate=16000, hop_size=160):
        self.sr = sample_rate
        self.hop = hop_size
        self.f0_bin = 256
        self.f0_max = 1100.0
        self.f0_min = 50.0
        self.f0_mel_min = 1127 * np.log(1 + self.f0_min / 700)
        self.f0_mel_max = 1127 * np.log(1 + self.f0_max / 700)

    def compute_f0(self, path: Path, f0_method: str, device="cpu"):
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

    def coarse_f0(self, f0: np.ndarray):
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

                featur_pit = self.compute_f0(inp_path, f0_method, device)
                np.save(opt_path2, featur_pit, allow_pickle=False)  # nsf
                coarse_pit = self.coarse_f0(featur_pit)
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


_sr_dict = {
    "32k": 32000,
    "40k": 40000,
    "48k": 48000,
}


# Module-level worker functions for multiprocessing
def _worker_process_f0_gpu(gpu_id: str, _process_idx: int, paths: list, log_file: Path):
    """Worker function for GPU-based F0 extraction."""
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    extractor = _FeatureExtractor()
    extractor.extract_f0_for_files(paths, "rmvpe_gpu", log_file, device="cuda")


def _worker_process_f0_cpu(
    _process_idx: int, paths: list, log_file: Path, extract_method: str
):
    """Worker function for CPU-based F0 extraction."""
    extractor = _FeatureExtractor()
    extractor.extract_f0_for_files(paths, extract_method, log_file, device="cpu")


def _worker_process_features_gpu(
    gpu_id: str, paths: list, log_file: Path, version: str
):
    """Worker function for GPU-based feature extraction."""
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    extractor = _FeatureExtractor()
    extractor.extract_model_features_for_files(paths, log_file, version)


def _preprocess_dataset(
    training_file: gr.FileData, exp_dir: str, sample_rate_str: str, n_p: int
):
    """Preprocess dataset by resampling audio files."""
    training_dir = Path(str(training_file)).parent
    sr = _sr_dict[sample_rate_str]

    log_dir = cwd / "logs" / exp_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "preprocess.log"
    log_file.touch()

    cmd = (
        f'"{config.python_cmd}" infer/modules/train/preprocess.py '
        f'"{training_dir}" {sr} {n_p} "{log_dir}" {config.noparallel} {config.preprocess_per:.1f}'
    )
    logger.info("Execute: %s", cmd)
    p = Popen(cmd, shell=True)
    while p.poll() is None:
        yield log_file.read_text()
        sleep(1)

    log = log_file.read_text()
    logger.info(log)
    yield log


def _extract_pitch_features(
    gpus: str,
    num_cpu_processes: int,
    extract_method: str,
    should_guide: bool,
    project_dir: str | Path,
    version: str,
    gpu_ids_rmvpe: str,
) -> Generator[str]:
    """Extract pitch features and model features using parallel processing."""
    log_dir = cwd / "logs" / project_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / "extract_f0_feature.log"
    log_file.unlink(missing_ok=True)
    log_file.touch()

    # Step 1: Extract pitch (f0) if pitch guidance is enabled
    if should_guide:
        input_root = log_dir / "1_16k_wavs"
        opt_root1 = log_dir / "2a_f0"
        opt_root2 = log_dir / "2b-f0nsf"
        opt_root1.mkdir(parents=True, exist_ok=True)
        opt_root2.mkdir(parents=True, exist_ok=True)

        # Build list of files to process
        file_paths = [
            [file_path, opt_root1 / file_path.name, opt_root2 / file_path.name]
            for file_path in sorted(input_root.iterdir())
            if file_path.is_file() and "spec" not in file_path.name
        ]

        # Determine device for RMVPE
        if extract_method == "rmvpe_gpu" and gpu_ids_rmvpe != "-":
            # Multi-GPU RMVPE extraction
            gpu_ids = gpu_ids_rmvpe.split("-")
            n_processes = len(gpu_ids)

            processes = [
                Process(
                    target=_worker_process_f0_gpu,
                    args=(gpu_id, idx, file_paths[idx::n_processes], log_file),
                )
                for idx, gpu_id in enumerate(gpu_ids)
            ]
            for p in processes:
                p.start()

            while any(p.is_alive() for p in processes):
                yield log_file.read_text()
                sleep(1)

            for p in processes:
                p.join()

        elif extract_method == "rmvpe_gpu":
            # DirectML RMVPE extraction
            try:
                import torch_directml  # type: ignore

                device = torch_directml.device(torch_directml.default_device())
            except ImportError:
                device = "cpu"
                logger.warning("torch_directml not available, falling back to CPU")

            extractor = _FeatureExtractor()
            # Process in main thread for DML
            extractor.extract_f0_for_files(file_paths, "rmvpe", log_file, device=device)
            yield log_file.read_text()
        else:
            # CPU-based extraction (pm, harvest, dio, rmvpe on CPU)
            processes = [
                Process(
                    target=_worker_process_f0_cpu,
                    args=(
                        i,
                        file_paths[i::num_cpu_processes],
                        log_file,
                        extract_method,
                    ),
                )
                for i in range(num_cpu_processes)
            ]
            for p in processes:
                p.start()

            while any(p.is_alive() for p in processes):
                yield log_file.read_text()
                sleep(1)

            for p in processes:
                p.join()

        # Final yield after f0 extraction
        log = log_file.read_text()
        logger.info(log)
        yield log

    # Step 2: Extract model features using GPUs
    wav_path = log_dir / "1_16k_wavs"
    out_path = log_dir / ("3_feature256" if version == "v1" else "3_feature768")
    out_path.mkdir(parents=True, exist_ok=True)

    # Build list of wav files to process
    wav_files = [
        (wav_file, out_path / wav_file.with_suffix(".npy").name)
        for wav_file in sorted(wav_path.iterdir())
        if wav_file.suffix == ".wav"
    ]

    gpu_list = gpus.split("-")

    processes = [
        Process(
            target=_worker_process_features_gpu,
            args=(gpu_id, wav_files[idx :: len(gpu_list)], log_file, version),
        )
        for idx, gpu_id in enumerate(gpu_list)
    ]
    for p in processes:
        p.start()

    while any(p.is_alive() for p in processes):
        yield log_file.read_text()
        sleep(1)

    for p in processes:
        p.join()

    log = log_file.read_text()
    logger.info(log)
    yield log


_GPUVisible = not config.dml


def _change_extraction_method(method):
    return {"visible": method == "rmvpe_gpu" and _GPUVisible, "__type__": "update"}


def main():

    gpu_info, gpus = get_gpu_info()

    with gr.Blocks(title="RVC WebUI") as app:
        gr.Markdown("## RVC WebUI")
        gr.Markdown(
            value=i18n(
                "This software is open source under the MIT license. "
                "The author has no control over the software. "
                "Users and those who distribute sounds generated by the software "
                "are solely responsible for their actions. <br>If you do not accept these terms, "
                "you may not use or reference any code or files in this package. "
                "See <b>LICENSE</b> in the root directory for details."
            )
        )
        project_dir = gr.Textbox(label=i18n("Enter project name"), value="mi-test")
        training_file = gr.File(
            label=i18n("Upload training file"),
            file_count="single",
        )
        gr_sample_rate = gr.Radio(
            label=i18n("Target sample rate"),
            choices=["40k", "48k"],
            value="40k",
            interactive=True,
        )

        num_cpu_processes = gr.Slider(
            minimum=0,
            maximum=config.n_cpu,
            step=1,
            label=i18n(
                "Number of CPU processes for pitch extraction and data processing"
            ),
            value=int(np.ceil(config.n_cpu / 1.5)),
            interactive=True,
        )

        preprocess_output = gr.Textbox(
            label=i18n("Output Information"),
            value="",
            max_lines=8,
            lines=4,
            autoscroll=True,
        )
        preprocess_button = gr.Button(i18n("Process data"), variant="primary")
        preprocess_button.click(
            _preprocess_dataset,
            [training_file, project_dir, gr_sample_rate, num_cpu_processes],
            [preprocess_output],
            api_name="train_preprocess",
        )

        include_pitch_guidance = gr.Radio(
            label=i18n(
                "Does the model use pitch guidance? (Required for singing, optional for speech)"
            ),
            choices=[True, False],
            value=True,
            interactive=True,
        )
        gr_version = gr.Radio(
            label=i18n("Version"),
            choices=["v1", "v2"],
            value="v2",
            interactive=True,
            visible=True,
        )

        gpu_ids_input = gr.Textbox(
            label=i18n(
                "Enter GPU IDs separated by '-', e.g. 0-1-2 to use GPU 0, 1, and 2"
            ),
            value=gpus,
            interactive=True,
            visible=_GPUVisible,
        )

        pitch_extraction_method = gr.Radio(
            label=i18n(
                "Select pitch extraction algorithm: For singing, use pm for speed; "
                "for high-quality speech but slow CPU, use dio for speed; harvest is "
                "better quality but slower; rmvpe has the best effect and uses some CPU/GPU"
            ),
            choices=["pm", "harvest", "dio", "rmvpe", "rmvpe_gpu"],
            value="rmvpe_gpu",
            interactive=True,
        )
        gpu_ids_rmvpe = gr.Textbox(
            label=i18n(
                "rmvpe GPU configuration: Enter different process GPU IDs separated by '-',"
                " e.g. 0-0-1 runs 2 processes on GPU 0 and 1 process on GPU 1"
            ),
            value=f"{gpus}-{gpus}",
            interactive=True,
            visible=_GPUVisible,
        )
        btn_extract_features = gr.Button(i18n("Extract Features"), variant="primary")
        feature_extraction_output = gr.Textbox(
            label=i18n("Output Information"),
            value="",
            max_lines=8,
            lines=4,
            autoscroll=True,
        )
        pitch_extraction_method.change(
            fn=_change_extraction_method,
            inputs=[pitch_extraction_method],
            outputs=[gpu_ids_rmvpe],
        )
        btn_extract_features.click(
            _extract_pitch_features,
            [
                gpu_ids_input,
                num_cpu_processes,
                pitch_extraction_method,
                include_pitch_guidance,
                project_dir,
                gr_version,
                gpu_ids_rmvpe,
            ],
            [feature_extraction_output],
            api_name="train_extract_f0_feature",
        )

        if config.iscolab:
            app.queue(max_size=1022).launch(share=True)
        else:
            app.queue(max_size=1022).launch(
                server_name="0.0.0.0",
                inbrowser=not config.noautoopen,
                server_port=config.listen_port,
                quiet=True,
            )


if __name__ == "__main__":
    # Create necessary directories
    for dir_path in ["logs", "assets/weights"]:
        (cwd / dir_path).mkdir(parents=True, exist_ok=True)

    main()
