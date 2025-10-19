import logging
import shutil
import sys
import threading
import warnings
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

cwd = Path.cwd()
sys.path.append(str(cwd))
load_dotenv()

logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# Cleanup runtime packages
for pack_dir in ["infer_pack", "uvr5_pack"]:
    shutil.rmtree(cwd / f"runtime/Lib/site-packages/{pack_dir}", ignore_errors=True)

# Create necessary directories
for dir_path in ["logs", "assets/weights"]:
    (cwd / dir_path).mkdir(parents=True, exist_ok=True)
warnings.filterwarnings("ignore")
torch.manual_seed(114514)


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


gpu_info, gpus = get_gpu_info()


def _wait_for_process(done, process_or_processes: Popen | list[Popen]):
    """Wait for subprocess(es) to complete."""
    processes = (
        process_or_processes
        if isinstance(process_or_processes, list)
        else [process_or_processes]
    )
    while any(p.poll() is None for p in processes):
        sleep(0.5)
    done[0] = True


_sr_dict = {
    "32k": 32000,
    "40k": 40000,
    "48k": 48000,
}


def _if_done(done, p):
    while 1:
        if p.poll() is None:
            sleep(0.5)
        else:
            break
    done[0] = True


def _preprocess_dataset(
    training_file: gr.FileData, exp_dir: str, sample_rate_str: str, n_p: int
):
    training_dir = Path(str(training_file)).parent
    sr = _sr_dict[sample_rate_str]  # Sample rate

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
    done = [False]
    threading.Thread(target=_if_done, args=(done, p)).start()
    while True:
        yield log_file.read_text()
        sleep(1)
        if done[0]:
            break
    log = log_file.read_text()
    logger.info(log)
    yield log


def _extract_pitch_features(
    gpus: str,
    num_cpu_processes: int,
    extract_method: str,
    should_guide: bool,
    project_dir: str | Path,
    extractor_version_id: str,
    gpu_ids_rmvpe: str,
):
    """Extract pitch features and model features using parallel processing."""
    log_dir = cwd / "logs" / project_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / "extract_f0_feature.log"
    log_file.unlink(missing_ok=True)
    log_file.touch()

    def _stream_logs_until_complete(processes):
        """Stream log file contents while processes are running."""
        processes = processes if isinstance(processes, list) else [processes]
        while any(p.poll() is None for p in processes):
            yield log_file.read_text()
            sleep(1)
        # Final log read after completion
        log = log_file.read_text()
        logger.info(log)
        yield log

    # Step 1: Extract pitch (f0) if pitch guidance is enabled
    if should_guide:
        extract_path = "infer/modules/train/extract"

        if extract_method == "rmvpe_gpu" and gpu_ids_rmvpe != "-":
            # Multi-GPU RMVPE extraction
            ids = gpu_ids_rmvpe.split("-")
            processes = [
                Popen(
                    f'"{config.python_cmd}" {extract_path}/extract_f0_rmvpe.py {len(ids)} {idx} '
                    f'{gpu_id} "{log_dir}" {config.is_half}',
                    shell=True,
                )
                for idx, gpu_id in enumerate(ids)
            ]
        elif extract_method == "rmvpe_gpu":
            # DirectML RMVPE extraction
            cmd = f'"{config.python_cmd}" {extract_path}/extract_f0_rmvpe_dml.py "{log_dir}"'
            logger.info("Execute: %s", cmd)
            processes = [Popen(cmd, shell=True)]
        else:
            # CPU-based extraction (pm, harvest, dio, rmvpe)
            cmd = (
                f'"{config.python_cmd}" {extract_path}/extract_f0_print.py '
                f'"{log_dir}" {num_cpu_processes} {extract_method}'
            )
            logger.info("Execute: %s", cmd)
            processes = [Popen(cmd, shell=True)]

        yield from _stream_logs_until_complete(processes)

    # Step 2: Extract model features using GPUs
    gpu_list = gpus.split("-")
    processes = [
        Popen(
            f'"{config.python_cmd}" infer/modules/train/extract_feature_print.py '
            f'{config.device} {len(gpu_list)} {idx} {n_g} "{log_dir}" {extractor_version_id} {config.is_half}',
            shell=True,
        )
        for idx, n_g in enumerate(gpu_list)
    ]
    yield from _stream_logs_until_complete(processes)


_GPUVisible = not config.dml


def _change_extraction_method(method):
    return {"visible": method == "rmvpe_gpu" and _GPUVisible, "__type__": "update"}


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
        label=i18n("Number of CPU processes for pitch extraction and data processing"),
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

    speaker_id = gr.Slider(
        minimum=0,
        maximum=4,
        step=1,
        label=i18n("Please specify speaker id"),
        value=0,
        interactive=True,
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
        label=i18n("Enter GPU IDs separated by '-', e.g. 0-1-2 to use GPU 0, 1, and 2"),
        value=gpus,
        interactive=True,
        visible=_GPUVisible,
    )
    gpu_status_display = gr.Textbox(
        label=i18n("GPU Information"),
        value=gpu_info,
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
