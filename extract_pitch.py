import logging
import multiprocessing
import os
import shutil
import sys
import warnings
from collections.abc import Generator
from multiprocessing import Process
from pathlib import Path
from subprocess import Popen
from time import sleep

import gradio as gr
import numpy as np
import torch
from dotenv import load_dotenv
from fairseq.modules.grad_multiply import GradMultiply

from configs.config import Config
from i18n.i18n import I18nAuto
from infer.lib.train.extract import FeatureExtractor

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
    gpus = "-".join(info[0] for info in details)
    return gpus


_sr_dict = {
    "32k": 32000,
    "40k": 40000,
    "48k": 48000,
}


# Module-level worker functions for multiprocessing
def _worker_process_f0_gpu(paths: list, log_file: Path, gpu_id: str):
    """Worker function for GPU-based F0 extraction."""
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    FeatureExtractor().extract_f0_for_files(paths, "rmvpe_gpu", log_file, device="cuda")


def _worker_process_f0_cpu(paths: list, log_file: Path, extract_method: str):
    """Worker function for CPU-based F0 extraction."""
    FeatureExtractor().extract_f0_for_files(
        paths, extract_method, log_file, device="cpu"
    )


def _worker_process_features_gpu(
    paths: list, log_file: Path, gpu_id: str, version: str
):
    """Worker function for GPU-based feature extraction."""
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    FeatureExtractor().extract_model_features_for_files(paths, log_file, version)


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
                    args=(file_paths[idx::n_processes], log_file, gpu_id),
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

            # Process in main thread for DML
            FeatureExtractor().extract_f0_for_files(
                file_paths, "rmvpe", log_file, device=device
            )
            yield log_file.read_text()
        else:
            # CPU-based extraction (pm, harvest, dio, rmvpe on CPU)
            processes = [
                Process(
                    target=_worker_process_f0_cpu,
                    args=(
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
            args=(wav_files[idx :: len(gpu_list)], log_file, gpu_id, version),
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

    gpus = get_gpu_info()

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
