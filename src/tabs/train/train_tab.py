import datetime
import threading
from pathlib import Path
from time import sleep

import gradio as gr
import numpy as np

import shared
from shared import i18n
from tabs.train.configure_training_parameters import configure_training_parameters
from tabs.train.extract_pitch_config import extract_pitch_config
from tabs.train.preprocess import create_preprocess_section

# Constants
LOG_POLL_INTERVAL = 0.5


def monitor_log_with_progress(
    log_file: Path,
    done_event: threading.Event,
    progress_callback,  # type: ignore
    poll_interval: float = LOG_POLL_INTERVAL,
) -> str:
    """Monitor a log file and update progress until completion."""
    while not done_event.is_set():
        progress_callback(log_file.read_text())
        sleep(poll_interval)
    return log_file.read_text()


def create_train_tab():
    with gr.TabItem(i18n("Train")):
        # Experiment Config
        with gr.Group():
            gr.Markdown(value=i18n("## Experiment Config"))
            with gr.Row():
                current_date = datetime.date.today().strftime("%Y-%m-%d")
                experiment_name = gr.Textbox(
                    label=i18n("Experiment Name"), value=f"experiment_{current_date}"
                )
                target_sr = gr.Radio(
                    label=i18n("Target Sample Rate"),
                    choices=["40k", "48k"],
                    value="48k",
                    interactive=True,
                )
                use_f0 = gr.Radio(
                    label=i18n("Pitch Guidance"),
                    choices=[True, False],
                    value=True,
                    interactive=True,
                )
                model_version = gr.Radio(
                    label=i18n("Version"),
                    choices=["v1", "v2"],
                    value="v2",
                    interactive=True,
                    visible=True,
                )
                cpu_count = gr.Slider(
                    minimum=0,
                    maximum=shared.config.n_cpu,
                    step=1,
                    label=i18n("CPU Process Count"),
                    value=int(np.ceil(shared.config.n_cpu / 1.5)),
                    interactive=True,
                )

        # Preprocess
        with gr.Group():
            spk_id = create_preprocess_section(experiment_name, target_sr, cpu_count)

        # Extract Pitch
        with gr.Group():
            f0method8, gpus_rmvpe = extract_pitch_config(
                experiment_name, use_f0, model_version, cpu_count
            )

        # Training Config
        with gr.Group():
            configure_training_parameters(
                experiment_name,
                target_sr,
                use_f0,
                model_version,
                spk_id,
                f0method8,
                gpus_rmvpe,
            )
