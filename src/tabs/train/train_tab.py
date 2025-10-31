import datetime
import json
import os
import platform
import re
import subprocess
import threading
import traceback
from pathlib import Path
from random import shuffle
from subprocess import Popen
from time import sleep
from typing import Literal

import faiss
import gradio as gr
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans

import shared
from shared import i18n
from tabs.train.extract_pitch_config import extract_pitch_config
from tabs.train.preprocess import create_preprocess_section

# Constants
MUTE_DIR_NAME = "mute"
KMEANS_MAX_SAMPLES = 2e5
KMEANS_N_CLUSTERS = 10000
INDEX_BATCH_SIZE = 8192
LOG_POLL_INTERVAL = 0.5


def _get_feature_dir_name(version: Literal["v1", "v2"]) -> str:
    """Get feature directory name based on version."""
    return shared.FEATURE_DIR_NAME if version == "v1" else shared.FEATURE_DIR_NAME_V2


def _get_pretrained_path(version: Literal["v1", "v2"], if_f0: bool) -> str:
    """Get pretrained model directory path."""
    return "" if version == "v1" else "_v2"


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


def _get_pretrained_models(path_str: str, f0_str: str, sample_rate: str):
    """Get paths to pretrained generator and discriminator models."""
    base_dir = Path(f"assets/pretrained{path_str}")
    gen_path = base_dir / f"{f0_str}G{sample_rate}.pth"
    dis_path = base_dir / f"{f0_str}D{sample_rate}.pth"

    gen_exists = gen_path.exists()
    dis_exists = dis_path.exists()

    if not gen_exists:
        shared.logger.warning("%s not exist, will not use pretrained model", gen_path)
    if not dis_exists:
        shared.logger.warning("%s not exist, will not use pretrained model", dis_path)

    return (
        str(gen_path) if gen_exists else "",
        str(dis_path) if dis_exists else "",
    )


def _change_sample_rate(sample_rate: str, if_f0: bool, version: Literal["v1", "v2"]):
    """Update pretrained model paths when sample rate changes."""
    path_str = _get_pretrained_path(version, if_f0)
    f0_str = "f0" if if_f0 else ""
    return _get_pretrained_models(path_str, f0_str, sample_rate)


def _change_version(sample_rate: str, if_f0: bool, version: Literal["v1", "v2"]):
    """Update pretrained model paths and sample rate choices when version changes."""
    # Adjust sample rate for v1 if needed
    if sample_rate == "32k" and version == "v1":
        sample_rate = "40k"
    path_str = _get_pretrained_path(version, if_f0)
    f0_str = "f0" if if_f0 else ""
    # Set available choices based on version
    choices = ["40k", "48k"] if version == "v1" else ["40k", "48k", "32k"]
    sr_update = {"choices": choices, "__type__": "update", "value": sample_rate}
    return (*_get_pretrained_models(path_str, f0_str, sample_rate), sr_update)


def _change_f0(if_f0: bool, sample_rate: str, version: Literal["v1", "v2"]):
    """Update UI visibility and pretrained model paths when f0 setting changes."""
    path_str = _get_pretrained_path(version, if_f0)
    visible_update = {"visible": if_f0, "__type__": "update"}
    f0_str = "f0" if if_f0 else ""
    gen_path, dis_path = _get_pretrained_models(path_str, f0_str, sample_rate)
    return visible_update, visible_update, gen_path, dis_path


def _parse_epoch_from_train_log_line(line: str) -> int | None:
    # Pattern 1: Train Epoch: X [...]
    match = re.search(r"Train Epoch:\s*(\d+)", line)
    if match:
        return int(match.group(1))

    # Pattern 2: ====> Epoch: X [...]
    match = re.search(r"====> Epoch:\s*(\d+)", line)
    if match:
        return int(match.group(1))

    return None


def _get_mute_paths(sample_rate: str, version: Literal["v1", "v2"], if_f0: bool):
    """Get paths for mute files used in training."""
    mute_dir = Path.cwd() / "logs" / MUTE_DIR_NAME

    mute_gt_wavs = mute_dir / shared.GT_WAVS_DIR_NAME / f"mute{sample_rate}.wav"
    dir_name = (
        shared.FEATURE_DIR_NAME if version == "v1" else shared.FEATURE_DIR_NAME_V2
    )
    mute_feature = mute_dir / dir_name / "mute.npy"

    if if_f0:
        mute_f0 = mute_dir / shared.F0_DIR_NAME / "mute.wav.npy"
        mute_f0nsf = mute_dir / shared.F0_NSF_DIR_NAME / "mute.wav.npy"
        return mute_gt_wavs, mute_feature, mute_f0, mute_f0nsf

    return mute_gt_wavs, mute_feature, None, None


def _build_filelist(
    gt_wavs_dir: Path,
    feature_dir: Path,
    f0_dir: Path | None,
    f0nsf_dir: Path | None,
    spk_id: str,
    sample_rate: str,
    version: Literal["v1", "v2"],
) -> list[str]:
    """Build file list for training."""
    # Collect file names for training
    names = {p.stem for p in gt_wavs_dir.iterdir() if p.is_file()} & {
        p.stem for p in feature_dir.iterdir() if p.is_file()
    }

    if f0_dir and f0nsf_dir:
        names &= {p.stem for p in f0_dir.iterdir() if p.is_file()}
        names &= {p.stem for p in f0nsf_dir.iterdir() if p.is_file()}

    # Build filelist
    opt = []
    if f0_dir and f0nsf_dir:
        opt.extend(
            [
                f"{gt_wavs_dir / (name + '.wav')}|{feature_dir / (name + '.npy')}|{f0_dir / (name + '.wav.npy')}|{f0nsf_dir / (name + '.wav.npy')}|{spk_id}"
                for name in names
            ]
        )
    else:
        opt.extend(
            [
                f"{gt_wavs_dir / (name + '.wav')}|{feature_dir / (name + '.npy')}|{spk_id}"
                for name in names
            ]
        )

    # Add mute files
    mute_paths = _get_mute_paths(sample_rate, version, f0_dir is not None)
    if f0_dir and f0nsf_dir:
        mute_gt, mute_feat, mute_f0, mute_f0nsf = mute_paths
        opt.extend(
            [f"{mute_gt}|{mute_feat}|{mute_f0}|{mute_f0nsf}|{spk_id}" for _ in range(2)]
        )
    else:
        mute_gt, mute_feat, _, _ = mute_paths
        opt.extend([f"{mute_gt}|{mute_feat}|{spk_id}" for _ in range(2)])

    shuffle(opt)
    return opt


_scalar_history = []


def _click_train(
    exp_dir_str: str,
    sample_rate: str,
    if_f0: bool,
    spk_id: str,
    save_epoch: int,
    total_epoch: int,
    batch_size: int,
    if_save_latest: bool,
    pretrained_G: str,
    pretrained_D: str,
    gpus: str,
    if_cache_gpu: bool,
    if_save_every_weights: bool,
    version: Literal["v1", "v2"],
    progress: gr.Progress = gr.Progress(),
):
    """Main training function."""
    # Setup experiment directories
    exp_dir = Path.cwd() / "logs" / exp_dir_str
    exp_dir.mkdir(parents=True, exist_ok=True)
    gt_wavs_dir = exp_dir / shared.GT_WAVS_DIR_NAME
    feature_dir = exp_dir / _get_feature_dir_name(version)

    # Setup F0 directories if needed
    f0_dir = exp_dir / shared.F0_DIR_NAME if if_f0 else None
    f0nsf_dir = exp_dir / shared.F0_NSF_DIR_NAME if if_f0 else None

    # Build and save filelist
    filelist = _build_filelist(
        gt_wavs_dir, feature_dir, f0_dir, f0nsf_dir, spk_id, sample_rate, version
    )

    # Collect file names for training
    f0_dir = None
    f0nsf_dir = None
    if if_f0:
        f0_dir = exp_dir / shared.F0_DIR_NAME
        f0nsf_dir = exp_dir / shared.F0_NSF_DIR_NAME
        names = (
            {p.stem for p in gt_wavs_dir.iterdir() if p.is_file()}
            & {p.stem for p in feature_dir.iterdir() if p.is_file()}
            & {p.stem for p in f0_dir.iterdir() if p.is_file()}
            & {p.stem for p in f0nsf_dir.iterdir() if p.is_file()}
        )
    else:
        names = {p.stem for p in gt_wavs_dir.iterdir() if p.is_file()} & {
            p.stem for p in feature_dir.iterdir() if p.is_file()
        }

    # Build filelist for training
    opt = []
    feature_dir_name = (
        shared.FEATURE_DIR_NAME if version == "v1" else shared.FEATURE_DIR_NAME_V2
    )
    mute_dir = Path.cwd() / "logs" / "mute"
    mute_gt_wavs = mute_dir / shared.GT_WAVS_DIR_NAME / f"mute{sample_rate}.wav"
    mute_feature = mute_dir / feature_dir_name / "mute.npy"

    if if_f0:
        mute_f0 = mute_dir / shared.F0_DIR_NAME / "mute.wav.npy"
        mute_f0nsf = mute_dir / shared.F0_NSF_DIR_NAME / "mute.wav.npy"
        assert f0_dir is not None and f0nsf_dir is not None
        opt.extend(
            [
                (
                    f"{gt_wavs_dir / (name + '.wav')}"
                    f"|{feature_dir / (name + '.npy')}"
                    f"|{f0_dir / (name + '.wav.npy')}"
                    f"|{f0nsf_dir / (name + '.wav.npy')}"
                    f"|{spk_id}"
                )
                for name in names
            ]
        )
        opt.extend(
            [
                f"{mute_gt_wavs}|{mute_feature}|{mute_f0}|{mute_f0nsf}|{spk_id}"
                for _ in range(2)
            ]
        )
    else:
        opt.extend(
            [
                f"{gt_wavs_dir / (name + '.wav')}|{feature_dir / (name + '.npy')}|{spk_id}"
                for name in names
            ]
        )
        opt.extend([f"{mute_gt_wavs}|{mute_feature}|{spk_id}" for _ in range(2)])

    shuffle(opt)
    filelist_path = exp_dir / "filelist.txt"
    filelist_path.write_text("\n".join(filelist), encoding="utf-8")
    shared.logger.debug("Write filelist done")

    # Log pretrained model info
    shared.logger.info("Use gpus: %s", str(gpus))
    if not pretrained_G:
        shared.logger.info("No pretrained Generator")
    if not pretrained_D:
        shared.logger.info("No pretrained Discriminator")

    # Prepare config
    config_path = (
        f"v1/{sample_rate}.json"
        if version == "v1" or sample_rate == "40k"
        else f"v2/{sample_rate}.json"
    )
    config_save_path = exp_dir / "config.json"
    if not config_save_path.exists():
        config_save_path.write_text(
            json.dumps(
                shared.config.json_config[config_path],
                ensure_ascii=False,
                indent=4,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    # Build and execute training command
    cmd_parts = [
        f'"{shared.config.python_cmd}" infer/modules/train/train.py',
        f'-e "{exp_dir_str}"',
        f"-sr {sample_rate}",
        f"-f0 {1 if if_f0 else 0}",
        f"-bs {batch_size}",
    ]
    if gpus:
        cmd_parts.append(f"-g {gpus}")
    cmd_parts.extend(
        [
            f"-te {total_epoch}",
            f"-se {save_epoch}",
        ]
    )
    if pretrained_G:
        cmd_parts.append(f"-pg {pretrained_G}")
    if pretrained_D:
        cmd_parts.append(f"-pd {pretrained_D}")
    cmd_parts.extend(
        [
            f"-l {1 if if_save_latest == i18n('Yes') else 0}",
            f"-c {1 if if_cache_gpu == i18n('Yes') else 0}",
            f"-sw {1 if if_save_every_weights == i18n('Yes') else 0}",
            f"-v {version}",
        ]
    )

    cmd = " ".join(cmd_parts)
    shared.logger.info("Execute: " + cmd)

    # Run training and update progress/plot
    current_epoch = 0
    p = Popen(cmd, shell=True, cwd=Path.cwd(), stdout=subprocess.PIPE)
    scalar_count = 0

    while True:
        assert p.stdout is not None
        line_bytes = p.stdout.readline()
        if not line_bytes:
            break
        line = line_bytes.decode("utf-8", errors="ignore")
        shared.logger.info(f"{line}")

        if line.startswith("SCALAR_DICT: "):
            try:
                scalar_dict = json.loads(line.replace("SCALAR_DICT: ", ""))
                scalar_dict["index"] = scalar_count
                scalar_count += 1
                _scalar_history.append(scalar_dict)
                df = pd.DataFrame(_scalar_history)
                print(f"history: {_scalar_history}")
                yield ("", df)
            except Exception:
                pass

        current_epoch = _parse_epoch_from_train_log_line(line) or current_epoch
        progress(current_epoch / total_epoch, desc="Training...")

    p.wait()
    yield (
        f"Training finished with exit code {p.returncode}.",
        pd.DataFrame(_scalar_history),
    )


def _train_index(
    experiment_name: str,
    version: Literal["v1", "v2"],
    progress: gr.Progress = gr.Progress(),
):
    """Train FAISS index for voice conversion."""
    exp_dir = Path("logs") / experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    feature_dir = exp_dir / _get_feature_dir_name(version)
    if not feature_dir.exists():
        return "Please perform feature extraction first!"

    feature_files = sorted([p for p in feature_dir.iterdir() if p.is_file()])
    if len(feature_files) == 0:
        return "Please perform feature extraction first!"

    progress(0.05, desc="Loading features...")

    infos = []
    npys = [np.load(path) for path in feature_files]
    big_npy = np.concatenate(npys, 0)
    big_npy_idx = np.arange(big_npy.shape[0])
    np.random.default_rng().shuffle(big_npy_idx)
    big_npy = big_npy[big_npy_idx]

    # Apply KMeans if dataset is large
    if big_npy.shape[0] > KMEANS_MAX_SAMPLES:
        infos.append(
            f"Trying to perform KMeans on {big_npy.shape[0]} samples to {KMEANS_N_CLUSTERS} centers."
        )
        progress(0.2, desc="Performing KMeans...")
        try:
            big_npy = (
                MiniBatchKMeans(
                    n_clusters=KMEANS_N_CLUSTERS,
                    verbose=True,
                    batch_size=256 * shared.config.n_cpu,
                    compute_labels=False,
                    init="random",
                )
                .fit(big_npy)
                .cluster_centers_
            )
        except Exception:
            info = traceback.format_exc()
            shared.logger.info(info)
            infos.append(info)
            yield "\n".join(infos)

    np.save(exp_dir / "total_fea.npy", big_npy)
    n_ivf = min(int(16 * np.sqrt(big_npy.shape[0])), big_npy.shape[0] // 39)
    infos.append(f"{big_npy.shape},{n_ivf}")

    progress(0.5, desc="Training FAISS index...")
    feature_dim = (
        shared.FEATURE_DIMENSION if version == "v1" else shared.FEATURE_DIMENSION_V2
    )
    index = faiss.index_factory(feature_dim, f"IVF{n_ivf},Flat")
    infos.append("training")
    index_ivf = faiss.extract_index_ivf(index)
    index_ivf.nprobe = 1
    index.train(big_npy)

    index_file_name = (
        f"IVF{n_ivf}_Flat_nprobe_{index_ivf.nprobe}_{experiment_name}_{version}.index"
    )
    faiss.write_index(index, f"{exp_dir}/trained_{index_file_name}")

    progress(0.7, desc="Adding vectors to index...")
    infos.append("Adding vectors to index...")

    # Add vectors in batches
    for i in range(0, big_npy.shape[0], INDEX_BATCH_SIZE):
        index.add(big_npy[i : i + INDEX_BATCH_SIZE])

    added_index_file_path = f"{exp_dir}/added_{index_file_name}"
    faiss.write_index(index, added_index_file_path)
    infos.append(f"Successfully built index: {added_index_file_path}")

    # Link to external directory
    try:
        link = os.link if platform.system() == "Windows" else os.symlink
        link(
            added_index_file_path,
            f"{shared.outside_index_root}/{experiment_name}_{index_file_name}",
        )
        infos.append(f"Linked index to external directory: {shared.outside_index_root}")
    except Exception:
        infos.append(
            f"Failed to link index to external directory: {shared.outside_index_root}"
        )

    progress(1.0, desc="Indexing complete!")
    yield "\n".join(infos)


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
            gr.Markdown(value=i18n("## Training Config"))
            with gr.Row():
                save_epoch = gr.Slider(
                    minimum=1,
                    maximum=50,
                    step=1,
                    label=i18n("Save Frequency"),
                    value=5,
                    interactive=True,
                )
                total_epoch = gr.Slider(
                    minimum=2,
                    maximum=1000,
                    step=1,
                    label=i18n("Total Epochs"),
                    value=20,
                    interactive=True,
                )
                batch_size = gr.Slider(
                    minimum=1,
                    maximum=40,
                    step=1,
                    label=i18n("Batch Size per GPU"),
                    value=shared.default_batch_size,
                    interactive=True,
                )
                if_save_latest = gr.Radio(
                    label=i18n("Only Save Latest Model"),
                    choices=[i18n("Yes"), i18n("No")],
                    value=i18n("No"),
                    interactive=True,
                )
                if_cache_gpu = gr.Radio(
                    label=i18n("Cache Data to GPU (Recommend for Data < 10 mins)"),
                    choices=[i18n("Yes"), i18n("No")],
                    value=i18n("No"),
                    interactive=True,
                )
                if_save_every_weights = gr.Radio(
                    label=i18n("Save Finalized Model Every Time"),
                    choices=[i18n("Yes"), i18n("No")],
                    value=i18n("No"),
                    interactive=True,
                )
            with gr.Row():
                pretrained_G = gr.Textbox(
                    label=i18n("Base Model G"),
                    value="assets/pretrained_v2/f0D48k.pth",
                    interactive=True,
                )
                pretrained_D = gr.Textbox(
                    label=i18n("Base Model D"),
                    value="assets/pretrained_v2/f0D48k.pth",
                    interactive=True,
                )
                target_sr.change(
                    _change_sample_rate,
                    [target_sr, use_f0, model_version],
                    [pretrained_G, pretrained_D],
                )
                model_version.change(
                    _change_version,
                    [target_sr, use_f0, model_version],
                    [pretrained_G, pretrained_D, target_sr],
                )
                use_f0.change(
                    _change_f0,
                    [use_f0, target_sr, model_version],
                    [f0method8, gpus_rmvpe, pretrained_G, pretrained_D],
                )
                gpus = gr.Textbox(
                    label=i18n(
                        "以-分隔输入使用的卡号, 例如   0-1-2   使用卡0和卡1和卡2"
                    ),
                    value=shared.gpus,
                    interactive=True,
                )
                train_btn = gr.Button(i18n("Train"), variant="primary")
                index_btn = gr.Button(i18n("Extra Feature Index"), variant="primary")

                training_plot = gr.LinePlot(
                    label=i18n("Training Metrics"),
                    x="index",  # Use the DataFrame's index as the x-axis (epochs)
                    y=["loss/g/total", "loss/d/total"],  # type: ignore # TODO: Fix type ignore
                )
                training_info = gr.Textbox(label=i18n("Info"), value="", max_lines=10)
                train_btn.click(
                    _click_train,
                    [
                        experiment_name,
                        target_sr,
                        use_f0,
                        spk_id,
                        save_epoch,
                        total_epoch,
                        batch_size,
                        if_save_latest,
                        pretrained_G,
                        pretrained_D,
                        gpus,
                        if_cache_gpu,
                        if_save_every_weights,
                        model_version,
                    ],
                    [training_info, training_plot],
                    api_name="train_start",
                )
                index_btn.click(
                    _train_index, [experiment_name, model_version], training_info
                )
