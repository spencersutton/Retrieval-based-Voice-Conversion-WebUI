import re
import threading
from collections.abc import Generator
from pathlib import Path
from subprocess import Popen
from typing import Literal

import gradio as gr

import shared
from shared import i18n
from tabs.train.train_tab import monitor_log_with_progress

f0_GPU_visible = not shared.config.dml


def _wait_for_process(done_event: threading.Event, p: Popen):
    """Wait for a single process to complete and signal completion."""
    p.wait()
    done_event.set()


def _wait_for_processes(done_event: threading.Event, processes: list[Popen]):
    """Wait for all processes to complete and signal completion."""
    for p in processes:
        p.wait()
    done_event.set()


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
    """Extract F0 and feature from audio files."""

    def update_progress(content: str):
        now, all_count = _parse_f0_feature_log(content)
        progress(
            float(now) / all_count, desc=f"{now}/{all_count} Features extracted..."
        )

    log_dir_path = Path.cwd() / "logs" / exp_dir
    log_dir_path.mkdir(parents=True, exist_ok=True)
    log_file = log_dir_path / "extract_f0_feature.log"
    log_file.touch()

    def run_commands(cmds: list[str], wait_all: bool = True):
        """Execute commands and monitor progress."""
        done_event = threading.Event()
        ps = [Popen(cmd, shell=True, cwd=Path.cwd()) for cmd in cmds]
        for cmd in cmds:
            shared.logger.info("Execute: " + cmd)

        if wait_all:
            threading.Thread(
                target=_wait_for_processes, args=(done_event, ps), daemon=True
            ).start()
        else:
            threading.Thread(
                target=_wait_for_process, args=(done_event, ps[0]), daemon=True
            ).start()

        return monitor_log_with_progress(
            log_file, done_event, update_progress, poll_interval=1
        )

    # Extract F0 if needed
    if if_f0:
        if f0method != "rmvpe_gpu":
            cmd = (
                f'"{shared.config.python_cmd}" '
                "infer/modules/train/extract/extract_f0_print.py "
                f'"{log_dir_path}" {n_p} {f0method}'
            )
            log = run_commands([cmd], wait_all=False)
        else:
            if gpus_rmvpe != "-":
                gpus_rmvpe_list = gpus_rmvpe.split("-")
                length = len(gpus_rmvpe_list)
                cmds = [
                    (
                        f'"{shared.config.python_cmd}" '
                        "infer/modules/train/extract/extract_f0_rmvpe.py "
                        f'{length} {idx} {n_g} "{log_dir_path}" {shared.config.is_half} '
                    )
                    for idx, n_g in enumerate(gpus_rmvpe_list)
                ]
                log = run_commands(cmds)
            else:
                cmd = (
                    f'"{shared.config.python_cmd}" '
                    "infer/modules/train/extract/extract_f0_rmvpe_dml.py "
                    f'"{log_dir_path}" '
                )
                shared.logger.info("Execute: " + cmd)
                p = Popen(cmd, shell=True, cwd=Path.cwd())
                p.wait()
                log = log_file.read_text()
                shared.logger.info(log)
        yield log

    # Feature extraction for each GPU part
    gpus = gpus_str.split("-")
    length = len(gpus)
    cmds = [
        (
            f'"{shared.config.python_cmd}" '
            f"infer/modules/train/extract_feature_print.py "
            f'{shared.config.device} {length} {idx} {n_g} "{log_dir_path}" {version} {shared.config.is_half}'
        )
        for idx, n_g in enumerate(gpus)
    ]
    log = run_commands(cmds)
    yield log


def _change_f0_method(f0_method: str):
    # Show GPU config only for rmvpe_gpu method
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
