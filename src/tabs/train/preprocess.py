import shutil
import threading
from collections.abc import Generator
from pathlib import Path

import gradio as gr

import shared
from infer.modules.train.preprocess import preprocess_trainset
from shared import i18n
from tabs.train.log import monitor_log_with_progress


def _preprocess_dataset(
    audio_dir: Path,
    exp_dir: Path,
    sr: str,
    n_p: int,
    progress: gr.Progress = gr.Progress(),
) -> Generator[str]:
    """
    Preprocesses an audio dataset by validating the input directory, counting files, and running a preprocessing script.
    Progress is reported via a Gradio progress object, and status messages are yielded throughout the process.

    Args:
        audio_dir (Path): Path to the directory containing audio files to preprocess.
        exp_dir (Path): Path to the experiment directory for storing logs.
        sr (str): Sample rate key to look up the actual sample rate value.
        n_p (int): Number of processes or parallel workers to use for preprocessing.
        progress (gr.Progress, optional): Gradio progress object for reporting progress. Defaults to gr.Progress().

    Yields:
        str: Status messages, warnings, errors, and the final log content.
    """
    # Validate audio_dir and count files
    if not audio_dir.is_dir():
        error_msg = (
            f"Error: Audio directory '{audio_dir}' not found or is not a directory."
        )
        shared.logger.error(error_msg)
        yield error_msg
        return

    try:
        file_names = [f for f in audio_dir.iterdir() if f.is_file()]
        actual_file_count = len(file_names)
    except OSError as e:
        error_msg = f"Error: Could not access audio directory '{audio_dir}': {e}"
        shared.logger.error(error_msg)
        yield error_msg
        return

    shared.logger.info(
        f"Found {actual_file_count} files in audio directory: {audio_dir}"
    )

    if actual_file_count == 0:
        warning_msg = f"Warning: No files found in '{audio_dir}'. Preprocessing script will run, but may not find items to process."
        shared.logger.warning(warning_msg)
        yield warning_msg
        if progress:
            progress(
                1.0,
                desc=f"No files found in {audio_dir}. Preprocessing step initiated.",
            )
        return

    sr_int = int(sr.replace("k", "000"))
    log_dir = Path.cwd() / "logs" / exp_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "preprocess.log"
    log_file.touch()

    # Run preprocess_trainset directly in a thread
    done_event = threading.Event()

    def run_preprocess():
        try:
            preprocess_trainset(
                inp_root=audio_dir,
                sr=sr_int,
                n_p=n_p,
                exp_dir=log_dir,
                per=shared.config.preprocess_per,
                no_parallel=shared.config.no_parallel,
            )
        finally:
            done_event.set()

    threading.Thread(target=run_preprocess, daemon=True).start()

    def update_progress(content: str):
        count = content.count("Success")
        progress(
            float(count) / actual_file_count,
            desc=f"Processed {count}/{actual_file_count} audio...",
        )

    log = monitor_log_with_progress(log_file, done_event, update_progress)
    shared.logger.info(log)
    yield log


def _preprocess_meta(
    experiment_name: str,
    audio_dir_str: str,
    audio_files_str: list[str] | None,
    sr: str,
    n_p: int,
    progress: gr.Progress = gr.Progress(),
):
    audio_dir = Path(audio_dir_str)
    save_dir = audio_dir / experiment_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # Copy selected audio files to experiment directory if provided
    if audio_files_str:
        audio_files = [Path(f) for f in audio_files_str]
        total_files = len(audio_files)
        for idx, audio_file in enumerate(audio_files):
            shutil.copy(audio_file, save_dir / audio_file.name)
            progress(idx / total_files, desc="Copying files...")

    # Run preprocessing on the prepared directory
    yield from _preprocess_dataset(
        audio_dir=save_dir,
        exp_dir=Path(experiment_name),
        sr=sr,
        n_p=n_p,
        progress=progress,
    )


def create_preprocess_section(
    experiment_name: gr.Textbox, target_sr: gr.Radio, cpu_count: gr.Slider
):
    gr.Markdown(value=i18n("## Preprocess"))
    spk_id = gr.Slider(
        minimum=0,
        maximum=4,
        step=1,
        label=i18n("Speaker ID"),
        value=0,
        interactive=True,
        visible=False,
    )
    with gr.Row():
        with gr.Column():
            audio_data_root = gr.Textbox(
                label=i18n("Audio Directory"),
                value=i18n("./datasets"),
            )
            audio_files = gr.Files(
                type="filepath", label=i18n("Audio Files"), file_types=["audio"]
            )
        with gr.Column():
            preprocessing_btn = gr.Button(i18n("Preprocess"), variant="primary")
            info1 = gr.Textbox(
                label=i18n("Info"),
                value="",
                lines=4,
            )
            preprocessing_btn.click(
                _preprocess_meta,
                [
                    experiment_name,
                    audio_data_root,
                    audio_files,
                    target_sr,
                    cpu_count,
                ],
                [info1],
                api_name="train_preprocess",
            )
    return spk_id
