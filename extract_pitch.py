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

now_dir = Path.cwd()
sys.path.append(str(now_dir))
load_dotenv()

logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# Cleanup runtime packages
for pack_dir in ["infer_pack", "uvr5_pack"]:
    shutil.rmtree(now_dir / f"runtime/Lib/site-packages/{pack_dir}", ignore_errors=True)

# Create necessary directories
for dir_path in ["logs", "assets/weights"]:
    (now_dir / dir_path).mkdir(parents=True, exist_ok=True)
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

ngpu = torch.cuda.device_count()
gpu_infos = []

# Supported GPU models for training
SUPPORTED_GPU_MODELS = [
    "10",
    "16",
    "20",
    "30",
    "40",
    "A2",
    "A3",
    "A4",
    "P4",
    "A50",
    "500",
    "A60",
    "70",
    "80",
    "90",
    "M4",
    "T4",
    "TITAN",
    "4060",
    "L",
    "6000",
]

if torch.cuda.is_available() and ngpu > 0:
    for i in range(ngpu):
        gpu_name = torch.cuda.get_device_name(i)
        if any(value in gpu_name.upper() for value in SUPPORTED_GPU_MODELS):
            gpu_infos.append(f"{i}\t{gpu_name}")

gpu_info = (
    "\n".join(gpu_infos) if gpu_infos else i18n("很遗憾您这没有能用的显卡来支持您训练")
)
gpus = "-".join(info[0] for info in gpu_infos)


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


def _extract_f0_feature(
    gpus: str,
    n_p,
    f0method: str,
    if_f0,
    exp_dir: str | Path,
    extractor_version_id,
    gpus_rmvpe,
):
    gpu_list = gpus.split("-")

    logs_directory = now_dir / "logs" / exp_dir
    log_file_path = logs_directory / "extract_f0_feature.log"

    logs_directory.mkdir(parents=True, exist_ok=True)
    log_file_path.touch()
    extract_path = "infer/modules/train/extract"
    if if_f0:
        if f0method != "rmvpe_gpu":
            cmd = f'"{config.python_cmd}" {extract_path}/extract_f0_print.py "{logs_directory}" {n_p} {f0method}'
            logger.info("Execute: %s", cmd)
            p = Popen(cmd, shell=True, cwd=now_dir)
            done = [False]
            threading.Thread(target=_wait_for_process, args=(done, p)).start()
        elif gpus_rmvpe != "-":
            gpus_rmvpe = gpus_rmvpe.split("-")
            length = len(gpus_rmvpe)
            ps = [
                Popen(
                    f'"{config.python_cmd}" {extract_path}/extract_f0_rmvpe.py {length} {idx} '
                    f'{n_g} "{logs_directory}" {config.is_half}',
                    shell=True,
                    cwd=now_dir,
                )
                for idx, n_g in enumerate(gpus_rmvpe)
            ]
            done = [False]
            threading.Thread(target=_wait_for_process, args=(done, ps)).start()
        else:
            cmd = f'"{config.python_cmd}" {extract_path}/extract_f0_rmvpe_dml.py "{logs_directory}"'
            logger.info("Execute: %s", cmd)
            Popen(cmd, shell=True, cwd=now_dir).wait()
            done = [True]
        while True:
            yield log_file_path.read_text()
            sleep(1)
            if done[0]:
                break
        log = log_file_path.read_text()
        logger.info(log)
        yield log

    length = len(gpu_list)
    ps = [
        Popen(
            f'"{config.python_cmd}" infer/modules/train/extract_feature_print.py '
            f'{config.device} {length} {idx} {n_g} "{logs_directory}" {extractor_version_id} {config.is_half}',
            shell=True,
            cwd=now_dir,
        )
        for idx, n_g in enumerate(gpu_list)
    ]
    done = [False]
    threading.Thread(target=_wait_for_process, args=(done, ps)).start()
    while True:
        yield log_file_path.read_text()
        sleep(1)
        if done[0]:
            break
    log = log_file_path.read_text()
    logger.info(log)
    yield log


_F0GPUVisible = not config.dml


def _change_f0_method(f0method8):
    return {"visible": f0method8 == "rmvpe_gpu" and _F0GPUVisible, "__type__": "update"}


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
    training_data_directory = gr.Textbox(
        label=i18n("Enter training folder path"),
        value=i18n("E:\\VoiceAudio+Annotations\\YonezuKenshi\\src"),
    )
    speaker_id = gr.Slider(
        minimum=0,
        maximum=4,
        step=1,
        label=i18n("Please specify speaker id"),
        value=0,
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
    include_pitch_guidance = gr.Radio(
        label=i18n(
            "Does the model use pitch guidance? (Required for singing, optional for speech)"
        ),
        choices=[True, False],
        value=True,
        interactive=True,
    )
    gr_experiment_dir = gr.Textbox(label=i18n("Enter experiment name"), value="mi-test")
    gr_version = gr.Radio(
        label=i18n("Version"),
        choices=["v1", "v2"],
        value="v2",
        interactive=True,
        visible=True,
    )
    gr_sample_rate = gr.Radio(
        label=i18n("Target sample rate"),
        choices=["40k", "48k"],
        value="40k",
        interactive=True,
    )
    gpu_ids_input = gr.Textbox(
        label=i18n("Enter GPU IDs separated by '-', e.g. 0-1-2 to use GPU 0, 1, and 2"),
        value=gpus,
        interactive=True,
        visible=_F0GPUVisible,
    )
    gpu_status_display = gr.Textbox(
        label=i18n("GPU Information"),
        value=gpu_info,
        visible=_F0GPUVisible,
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
    gpus_rmvpe = gr.Textbox(
        label=i18n(
            "rmvpe GPU configuration: Enter different process GPU IDs separated by '-',"
            " e.g. 0-0-1 runs 2 processes on GPU 0 and 1 process on GPU 1"
        ),
        value=f"{gpus}-{gpus}",
        interactive=True,
        visible=_F0GPUVisible,
    )
    btn_extract_features = gr.Button(i18n("Extract Features"), variant="primary")
    feature_extraction_output = gr.Textbox(
        label=i18n("Output Information"), value="", max_lines=8
    )
    pitch_extraction_method.change(
        fn=_change_f0_method,
        inputs=[pitch_extraction_method],
        outputs=[gpus_rmvpe],
    )
    btn_extract_features.click(
        _extract_f0_feature,
        [
            gpu_ids_input,
            num_cpu_processes,
            pitch_extraction_method,
            include_pitch_guidance,
            gr_experiment_dir,
            gr_version,
            gpus_rmvpe,
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
