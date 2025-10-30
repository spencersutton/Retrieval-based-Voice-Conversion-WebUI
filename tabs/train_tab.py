import datetime
import json
import os
import platform
import re
import shutil
import subprocess
import threading
import traceback
from collections.abc import Generator
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

ProgressComponent = gr.Progress

F0GPUVisible = not shared.config.dml


def change_f0_method(f0method8: str):
    if f0method8 == "rmvpe_gpu":
        visible = F0GPUVisible
    else:
        visible = False
    return {"visible": visible, "__type__": "update"}


def if_done(done_flag: list[bool], p: Popen):
    p.wait()
    done_flag[0] = True


def if_done_multi(done_flag: list[bool], p_objs: list[Popen]):
    for p_obj in p_objs:
        p_obj.wait()
    done_flag[0] = True


def preprocess_dataset(
    audio_dir: Path,
    exp_dir: Path,
    sr: str,  # pyright: ignore[reportRedeclaration]
    n_p: int,
    progress: gr.Progress = gr.Progress(),
) -> Generator[str, None, None]:
    # 1. Validate audio_dir and count files
    if not audio_dir.is_dir():
        error_msg = (
            f"Error: Audio directory '{audio_dir}' not found or is not a directory."
        )
        shared.logger.error(error_msg)
        yield error_msg
        return

    actual_file_count = 0
    try:
        # List all entries in the directory and filter for files
        file_names = [name for name in audio_dir.iterdir() if name.is_file()]
        actual_file_count = len(file_names)
        info_msg = f"Found {actual_file_count} files in audio directory: {audio_dir}"
        shared.logger.info(info_msg)

        if actual_file_count == 0:
            warning_msg = f"Warning: No files found in '{audio_dir}'. Preprocessing script will run, but may not find items to process."
            shared.logger.warning(warning_msg)
            yield warning_msg
            # Update progress to indicate nothing to process, but the step is "complete"
            if progress:
                progress(
                    1.0,
                    desc=f"No files found in {audio_dir}. Preprocessing step initiated.",
                )
            return

    except OSError as e:
        error_msg = (
            f"Error: Could not access audio directory '{audio_dir}' to count files: {e}"
        )
        shared.logger.error(error_msg)
        yield error_msg
        return
    sr: int = shared.sr_dict[sr]
    log_dir = Path.cwd() / "logs" / exp_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "preprocess.log"
    log_file.touch()
    cmd = f'"{shared.config.python_cmd}" infer/modules/train/preprocess.py "{audio_dir}" {sr} {n_p} "{log_dir}" {shared.config.noparallel} {shared.config.preprocess_per:.1f}'
    shared.logger.info("Execute: " + cmd)
    p = Popen(cmd, shell=True)
    # 煞笔gr, popen read都非得全跑完了再一次性读取, 不用gr就正常读一句输出一句;只能额外弄出一个文本流定时读
    done = [False]
    threading.Thread(target=if_done, args=(done, p)).start()

    log_file_path = Path.cwd() / "logs" / exp_dir / "preprocess.log"
    while True:
        file_content = log_file_path.read_text()
        count = file_content.count("Success")
        progress(
            float(count) / actual_file_count,
            desc=f"Processed {count}/{actual_file_count} audio...",
        )
        sleep(0.5)
        if done[0]:
            break
    log = log_file_path.read_text()
    shared.logger.info(log)
    yield log


def preprocess_meta(
    experiment_name: str,
    audio_dir_str: str,
    audio_files_str: list[str] | None,
    sr: str,
    n_p: int,
    progress: gr.Progress = gr.Progress(),
):
    audio_dir = Path(audio_dir_str)
    audio_files = [Path(f) for f in audio_files_str] if audio_files_str else None
    save_dir = audio_dir / experiment_name
    save_dir.mkdir(parents=True, exist_ok=True)

    if audio_files is not None:
        for idx, audio_file in enumerate(audio_files):
            shutil.copy(audio_file, save_dir / audio_file.name)
            progress(idx / len(audio_files), "Copying files...")

    yield from preprocess_dataset(
        audio_dir=save_dir,
        exp_dir=Path(experiment_name),
        sr=sr,
        n_p=n_p,
        progress=progress,
    )


def parse_f0_feature_log(content: str) -> tuple[int, int]:
    max_now: int | None = 0
    max_all: int | None = 1
    # Regex to capture the numbers after "now-" and "all-"
    # Pattern breakdown:
    # f0ing,       # Literal string "f0ing,"
    # now-(\d+)    # "now-" followed by one or more digits (captured as group 1)
    # ,all-(\d+)   # ",all-" followed by one or more digits (captured as group 2)
    # The rest of the line (e.g., ",-filepath") is ignored by this regex.
    pattern = re.compile(r"f0ing,now-(\d+),all-(\d+)")

    for line in content.splitlines():
        match = pattern.search(line)
        if match:
            try:
                current_now_str = match.group(1)
                current_all_str = match.group(2)

                current_now = int(current_now_str)
                current_all = int(current_all_str)

                if max_now is None or current_now > max_now:
                    max_now = current_now

                if max_all is None or current_all > max_all:
                    max_all = current_all
            except ValueError:
                print(f"Warning: Could not parse numbers from line: {line}")

    return max_now, max_all


def extract_f0_feature(
    gpus_str: str,
    n_p: int,
    f0method: str,
    if_f0: bool,
    exp_dir: str,
    version: str,
    gpus_rmvpe: str,  # pyright: ignore[reportRedeclaration]
    progress: gr.Progress = gr.Progress(),
) -> Generator[str, None, None]:
    def update_progress(content: str):
        now, all = parse_f0_feature_log(content)
        progress(float(now) / all, desc=f"{now}/{all} Features extracted...")

    gpus = gpus_str.split("-")
    log_dir_path = Path.cwd() / "logs" / exp_dir
    log_dir_path.mkdir(parents=True, exist_ok=True)
    log_file = log_dir_path / "extract_f0_feature.log"
    log_file.touch()
    if if_f0:
        if f0method != "rmvpe_gpu":
            cmd = f'"{shared.config.python_cmd}" infer/modules/train/extract/extract_f0_print.py "{log_dir_path}" {n_p} {f0method}'
            shared.logger.info("Execute: " + cmd)
            p = Popen(cmd, shell=True, cwd=Path.cwd())
            # 煞笔gr, popen read都非得全跑完了再一次性读取, 不用gr就正常读一句输出一句;只能额外弄出一个文本流定时读
            done = [False]
            threading.Thread(target=if_done, args=(done, p)).start()
        else:
            if gpus_rmvpe != "-":
                gpus_rmvpe: list[str] = gpus_rmvpe.split("-")
                length = len(gpus_rmvpe)
                ps = []
                for idx, n_g in enumerate(gpus_rmvpe):
                    cmd = f'"{shared.config.python_cmd}" infer/modules/train/extract/extract_f0_rmvpe.py {length} {idx} {n_g} "{Path.cwd()}/logs/{exp_dir}" {shared.config.is_half} '
                    shared.logger.info("Execute: " + cmd)
                    p = Popen(cmd, shell=True, cwd=Path.cwd())
                    ps.append(p)
                done = [False]
                threading.Thread(
                    target=if_done_multi,  #
                    args=(
                        done,
                        ps,
                    ),
                ).start()
            else:
                cmd = (
                    shared.config.python_cmd
                    + f' infer/modules/train/extract/extract_f0_rmvpe_dml.py "{log_dir_path}" '
                )
                shared.logger.info("Execute: " + cmd)
                p = Popen(cmd, shell=True, cwd=Path.cwd())
                p.wait()
                done = [True]
        while True:
            update_progress(log_file.read_text())
            sleep(1)
            if done[0]:
                break
        log = log_file.read_text()
        shared.logger.info(log)
    # 对不同part分别开多进程
    length = len(gpus)
    ps = []
    for idx, n_g in enumerate(gpus):
        cmd = f'"{shared.config.python_cmd}" infer/modules/train/extract_feature_print.py {shared.config.device} {length} {idx} {n_g} "{Path.cwd()}/logs/{exp_dir}" {version} {shared.config.is_half}'
        shared.logger.info("Execute: " + cmd)
        p = Popen(cmd, shell=True, cwd=Path.cwd())
        ps.append(p)
    # 煞笔gr, popen read都非得全跑完了再一次性读取, 不用gr就正常读一句输出一句;只能额外弄出一个文本流定时读
    done = [False]
    threading.Thread(
        target=if_done_multi,
        args=(
            done,
            ps,
        ),
    ).start()
    while True:
        update_progress(log_file.read_text())
        sleep(1)
        if done[0]:
            break
    log = log_file.read_text()
    shared.logger.info(log)
    yield log


def get_pretrained_models(path_str: str, f0_str: str, sr2: str):
    if_pretrained_generator_exist = os.access(
        f"assets/pretrained{path_str}/{f0_str}G{sr2}.pth", os.F_OK
    )
    if_pretrained_discriminator_exist = os.access(
        f"assets/pretrained{path_str}/{f0_str}D{sr2}.pth", os.F_OK
    )
    if not if_pretrained_generator_exist:
        shared.logger.warning(
            "assets/pretrained%s/%sG%s.pth not exist, will not use pretrained model",
            path_str,
            f0_str,
            sr2,
        )
    if not if_pretrained_discriminator_exist:
        shared.logger.warning(
            "assets/pretrained%s/%sD%s.pth not exist, will not use pretrained model",
            path_str,
            f0_str,
            sr2,
        )
    return (
        (
            f"assets/pretrained{path_str}/{f0_str}G{sr2}.pth"
            if if_pretrained_generator_exist
            else ""
        ),
        (
            f"assets/pretrained{path_str}/{f0_str}D{sr2}.pth"
            if if_pretrained_discriminator_exist
            else ""
        ),
    )


def change_sr2(sr2: str, if_f0_3: bool, version: str):
    path_str = "" if version == "v1" else "_v2"
    f0_str = "f0" if if_f0_3 else ""
    return get_pretrained_models(path_str, f0_str, sr2)


def change_version19(sr2: str, if_f0_3: bool, version: str):
    path_str = "" if version == "v1" else "_v2"
    if sr2 == "32k" and version == "v1":
        sr2 = "40k"
    to_return_sr2 = (
        {"choices": ["40k", "48k"], "__type__": "update", "value": sr2}
        if version == "v1"
        else {"choices": ["40k", "48k", "32k"], "__type__": "update", "value": sr2}
    )
    f0_str = "f0" if if_f0_3 else ""
    return (
        *get_pretrained_models(path_str, f0_str, sr2),
        to_return_sr2,
    )


def change_f0(if_f0_3: bool, sr2: str, version: str):
    path_str = "" if version == "v1" else "_v2"
    return (
        {"visible": if_f0_3, "__type__": "update"},
        {"visible": if_f0_3, "__type__": "update"},
        *get_pretrained_models(path_str, "f0" if if_f0_3 else "", sr2),
    )


def parse_epoch_from_train_log_line(line: str) -> int | None:
    """
    Parse a single log line and extract the current epoch number if present.

    Args:
        line (bytes): A single log line in bytes.

    Returns:
        Optional[int]: The epoch number if found, otherwise None.
    """

    # Pattern 1: Train Epoch: X [...]
    match = re.search(r"Train Epoch:\s*(\d+)", line)
    if match:
        return int(match.group(1))

    # Pattern 2: ====> Epoch: X [...]
    match = re.search(r"====> Epoch:\s*(\d+)", line)
    if match:
        return int(match.group(1))

    return None


scalar_history = []


def click_train(
    exp_dir1: str,
    sr2: str,
    if_f0_3: bool,
    spk_id5: str,
    save_epoch10: int,
    total_epoch11: int,
    batch_size12: int,
    if_save_latest13: bool,
    pretrained_G14: str,
    pretrained_D15: str,
    gpus16: str,
    if_cache_gpu17: bool,
    if_save_every_weights18: bool,
    version: Literal["v1", "v2"],
    progress: gr.Progress = gr.Progress(),
):
    # Setup experiment directories
    exp_dir = Path.cwd() / "logs" / exp_dir1
    exp_dir.mkdir(parents=True, exist_ok=True)
    gt_wavs_dir = exp_dir / shared.GT_WAVS_DIR_NAME
    feature_dir = exp_dir / (
        shared.FEATURE_DIR_NAME if version == "v1" else shared.FEATURE_DIR_NAME_V2
    )

    # Collect file names for training
    f0_dir = None
    f0nsf_dir = None
    if if_f0_3:
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
    fea_dim = 256 if version == "v1" else 768
    mute_dir = Path.cwd() / "logs" / "mute"
    mute_gt_wavs = mute_dir / shared.GT_WAVS_DIR_NAME / f"mute{sr2}.wav"
    mute_feature = mute_dir / f"3_feature{fea_dim}" / "mute.npy"

    if if_f0_3:
        mute_f0 = mute_dir / shared.F0_DIR_NAME / "mute.wav.npy"
        mute_f0nsf = mute_dir / shared.F0_NSF_DIR_NAME / "mute.wav.npy"
        assert f0_dir is not None and f0nsf_dir is not None
        opt.extend(
            [
                f"{gt_wavs_dir / (name + '.wav')}|{feature_dir / (name + '.npy')}|{f0_dir / (name + '.wav.npy')}|{f0nsf_dir / (name + '.wav.npy')}|{spk_id5}"
                for name in names
            ]
        )
        opt.extend(
            [
                f"{mute_gt_wavs}|{mute_feature}|{mute_f0}|{mute_f0nsf}|{spk_id5}"
                for _ in range(2)
            ]
        )
    else:
        opt.extend(
            [
                f"{gt_wavs_dir / (name + '.wav')}|{feature_dir / (name + '.npy')}|{spk_id5}"
                for name in names
            ]
        )
        opt.extend([f"{mute_gt_wavs}|{mute_feature}|{spk_id5}" for _ in range(2)])

    shuffle(opt)
    filelist_path = exp_dir / "filelist.txt"
    filelist_path.write_text("\n".join(opt), encoding="utf-8")
    shared.logger.debug("Write filelist done")

    # Prepare config
    shared.logger.info("Use gpus: %s", str(gpus16))
    if pretrained_G14 == "":
        shared.logger.info("No pretrained Generator")
    if pretrained_D15 == "":
        shared.logger.info("No pretrained Discriminator")
    config_path = (
        f"v1/{sr2}.json" if version == "v1" or sr2 == "40k" else f"v2/{sr2}.json"
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

    # Helper to build command line arguments for train.py
    def build_train_cmd(
        gpus: str,
        pretrained_G: str,
        pretrained_D: str,
        save_latest: bool,
        cache_gpu: bool,
        save_every_weights: bool,
    ):
        args = [
            f'"{shared.config.python_cmd}" infer/modules/train/train.py',
            f'-e "{exp_dir1}"',
            f"-sr {sr2}",
            f"-f0 {1 if if_f0_3 else 0}",
            f"-bs {batch_size12}",
        ]
        if gpus:
            args.append(f"-g {gpus}")
        args.extend(
            [
                f"-te {total_epoch11}",
                f"-se {save_epoch10}",
            ]
        )
        if pretrained_G:
            args.append(f"-pg {pretrained_G}")
        if pretrained_D:
            args.append(f"-pd {pretrained_D}")
        args.extend(
            [
                f"-l {1 if save_latest == i18n('Yes') else 0}",
                f"-c {1 if cache_gpu == i18n('Yes') else 0}",
                f"-sw {1 if save_every_weights == i18n('Yes') else 0}",
                f"-v {version}",
            ]
        )
        return " ".join(args)

    cmd = build_train_cmd(
        gpus16,
        pretrained_G14,
        pretrained_D15,
        if_save_latest13,
        if_cache_gpu17,
        if_save_every_weights18,
    )
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
                scalar_history.append(scalar_dict)
                df = pd.DataFrame(scalar_history)
                print(f"history: {scalar_history}")
                yield ("", df)
            except Exception:
                pass

        current_epoch = parse_epoch_from_train_log_line(line) or current_epoch
        progress(current_epoch / total_epoch11, desc="Training...")

    p.wait()
    yield (
        f"Training finished with exit code {p.returncode}.",
        pd.DataFrame(scalar_history),
    )


def train_index(exp_dir1: str, version: str, progress: gr.Progress = gr.Progress()):
    exp_dir = Path("logs") / exp_dir1
    exp_dir.mkdir(parents=True, exist_ok=True)
    feature_dir = exp_dir / (
        shared.FEATURE_DIR_NAME if version == "v1" else shared.FEATURE_DIR_NAME_V2
    )
    if not feature_dir.exists():
        return "Please perform feature extraction first!"
    listdir_res = sorted([p for p in feature_dir.iterdir() if p.is_file()])
    if len(listdir_res) == 0:
        return "Please perform feature extraction first!"

    progress(0.05, desc="Loading features...")  # Initial progress update
    infos = []
    npys = []
    for path in listdir_res:
        phone = np.load(str(path))
        npys.append(phone)
    big_npy = np.concatenate(npys, 0)
    big_npy_idx = np.arange(big_npy.shape[0])
    rng = np.random.default_rng()
    rng.shuffle(big_npy_idx)
    big_npy = big_npy[big_npy_idx]
    if big_npy.shape[0] > 2e5:
        infos.append(
            f"Trying to perform KMeans on {big_npy.shape[0]} samples to 10k centers."
        )
        progress(0.2, desc="Performing KMeans...")  # Progress update for KMeans
        try:
            big_npy = (
                MiniBatchKMeans(
                    n_clusters=10000,
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

    np.save(f"{exp_dir}/total_fea.npy", big_npy)
    n_ivf = min(int(16 * np.sqrt(big_npy.shape[0])), big_npy.shape[0] // 39)
    infos.append(f"{big_npy.shape},{n_ivf}")
    progress(0.5, desc="Training FAISS index...")  # Progress update for training
    index = faiss.index_factory(256 if version == "v1" else 768, f"IVF{n_ivf},Flat")
    infos.append("training")
    index_ivf = faiss.extract_index_ivf(index)  #
    index_ivf.nprobe = 1
    index.train(big_npy)
    faiss.write_index(
        index,
        f"{exp_dir}/trained_IVF{n_ivf}_Flat_nprobe_{index_ivf.nprobe}_{exp_dir1}_{version}.index",
    )
    progress(0.7, desc="Adding vectors to index...")
    infos.append("Adding vectors to index...")
    batch_size_add = 8192
    for i in range(0, big_npy.shape[0], batch_size_add):
        index.add(big_npy[i : i + batch_size_add])
    faiss.write_index(
        index,
        f"{exp_dir}/added_IVF{n_ivf}_Flat_nprobe_{index_ivf.nprobe}_{exp_dir1}_{version}.index",
    )
    infos.append(
        f"Successfully built index: added_IVF{n_ivf}_Flat_nprobe_{index_ivf.nprobe}_{exp_dir1}_{version}.index"
    )
    try:
        link = os.link if platform.system() == "Windows" else os.symlink
        link(
            f"{exp_dir}/added_IVF{n_ivf}_Flat_nprobe_{index_ivf.nprobe}_{exp_dir1}_{version}.index",
            f"{shared.outside_index_root}/{exp_dir1}_IVF{n_ivf}_Flat_nprobe_{index_ivf.nprobe}_{exp_dir1}_{version}.index",
        )
        infos.append(
            f"Linked index to external directory: {shared.outside_index_root}"
        )  # Original: "链接索引到外部-%s"
    except Exception:
        infos.append(
            f"Failed to link index to external directory: {shared.outside_index_root}"
        )  # Original: "链接索引到外部-%s失败"
    progress(1.0, desc="Indexing complete!")  # Final progress update
    yield "\n".join(infos)


def one_click_training(
    exp_dir1: str,
    sr2: str,
    if_f0_3: bool,
    trainset_dir4: str,
    spk_id5: str,
    np7: int,
    f0method8: str,
    save_epoch10: int,
    total_epoch11: int,
    batch_size12: int,
    if_save_latest13: bool,
    pretrained_G14: str,
    pretrained_D15: str,
    gpus16: str,
    if_cache_gpu17: bool,
    if_save_every_weights18: bool,
    version: Literal["v1", "v2"],
    gpus_rmvpe: str,
) -> Generator[str]:
    infos: list[str] = []

    def get_info_str(strr: str) -> str:
        infos.append(strr)
        return "\n".join(infos)

    yield get_info_str(shared.i18n("step1: processing data..."))
    [
        get_info_str(_)
        for _ in preprocess_dataset(Path(trainset_dir4), Path(exp_dir1), sr2, np7)
    ]

    yield get_info_str(shared.i18n("step2: extracting feature & pitch"))
    [
        get_info_str(_)
        for _ in extract_f0_feature(
            gpus16, np7, f0method8, if_f0_3, exp_dir1, version, gpus_rmvpe
        )
    ]

    yield get_info_str(shared.i18n("step3a:正在训练模型"))
    click_train(
        exp_dir1,
        sr2,
        if_f0_3,
        spk_id5,
        save_epoch10,
        total_epoch11,
        batch_size12,
        if_save_latest13,
        pretrained_G14,
        pretrained_D15,
        gpus16,
        if_cache_gpu17,
        if_save_every_weights18,
        version,
    )
    yield get_info_str(
        i18n("训练结束, 您可查看控制台训练日志或实验文件夹下的train.log")
    )

    [get_info_str(_) for _ in train_index(exp_dir1, version)]
    yield get_info_str(i18n("全流程结束！"))


def create_train_tab():
    with gr.TabItem(i18n("Train")):
        with gr.Group():
            gr.Markdown(value=i18n("## Experiment Config"))
            with gr.Row():
                current_date = datetime.date.today()
                formatted_date = current_date.strftime("%Y-%m-%d")
                experiment_name = gr.Textbox(
                    label=i18n("Experiment Name"), value=f"experiment_{formatted_date}"
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

        with gr.Group():
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
                        preprocess_meta,
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
        with gr.Group():
            gr.Markdown(value=i18n("## Extract Pitch"))
            with gr.Row():
                with gr.Column():
                    gpus6 = gr.Textbox(
                        label=i18n(
                            "以-分隔输入使用的卡号, 例如   0-1-2   使用卡0和卡1和卡2"
                        ),
                        value=shared.gpus,
                        interactive=True,
                        visible=F0GPUVisible,
                    )
                    gr.Textbox(
                        label=i18n("GPU Info"),
                        value=shared.gpu_info,
                        visible=F0GPUVisible,
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
                            "rmvpe卡号配置：以-分隔输入使用的不同进程卡号,例如0-0-1使用在卡0上跑2个进程并在卡1上跑1个进程"
                        ),
                        value=f"{shared.gpus}-{shared.gpus}",
                        interactive=True,
                        visible=F0GPUVisible,
                    )
                with gr.Column():
                    extract_f0_btn = gr.Button(i18n("Extract"), variant="primary")
                    info2 = gr.Textbox(label=i18n("Info"), value="", max_lines=8)
                    f0method8.change(
                        fn=change_f0_method,
                        inputs=[f0method8],
                        outputs=[gpus_rmvpe],
                    )
                    extract_f0_btn.click(
                        extract_f0_feature,
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
                if_save_latest13 = gr.Radio(
                    label=i18n("Only Save Latest Model"),
                    choices=[i18n("Yes"), i18n("No")],
                    value=i18n("No"),
                    interactive=True,
                )
                if_cache_gpu17 = gr.Radio(
                    label=i18n("Cache Data to GPU (Recommend for Data < 10 mins)"),
                    choices=[i18n("Yes"), i18n("No")],
                    value=i18n("No"),
                    interactive=True,
                )
                if_save_every_weights18 = gr.Radio(
                    label=i18n("Save Finalized Model Every Time"),
                    choices=[i18n("Yes"), i18n("No")],
                    value=i18n("No"),
                    interactive=True,
                )
            with gr.Row():
                pretrained_G14 = gr.Textbox(
                    label=i18n("Base Model G"),
                    value="assets/pretrained_v2/f0D48k.pth",
                    interactive=True,
                )
                pretrained_D15 = gr.Textbox(
                    label=i18n("Base Model D"),
                    value="assets/pretrained_v2/f0D48k.pth",
                    interactive=True,
                )
                target_sr.change(
                    change_sr2,
                    [target_sr, use_f0, model_version],
                    [pretrained_G14, pretrained_D15],
                )
                model_version.change(
                    change_version19,
                    [target_sr, use_f0, model_version],
                    [pretrained_G14, pretrained_D15, target_sr],
                )
                use_f0.change(
                    change_f0,
                    [use_f0, target_sr, model_version],
                    [f0method8, gpus_rmvpe, pretrained_G14, pretrained_D15],
                )
                gpus16 = gr.Textbox(
                    label=i18n(
                        "以-分隔输入使用的卡号, 例如   0-1-2   使用卡0和卡1和卡2"
                    ),
                    value=shared.gpus,
                    interactive=True,
                )
                train_btn = gr.Button(i18n("Train"), variant="primary")
                index_btn = gr.Button(i18n("Extra Feature Index"), variant="primary")
                one_click_btn = gr.Button(i18n("Train Everything"), variant="primary")

                training_plot = gr.LinePlot(
                    label=i18n("Training Metrics"),
                    x="index",  # Use the DataFrame's index as the x-axis (epochs)
                    y=["loss/g/total", "loss/d/total"],  # type: ignore # TODO: Fix type ignore
                )
                training_info = gr.Textbox(label=i18n("Info"), value="", max_lines=10)
                train_btn.click(
                    click_train,
                    [
                        experiment_name,
                        target_sr,
                        use_f0,
                        spk_id,
                        save_epoch,
                        total_epoch,
                        batch_size,
                        if_save_latest13,
                        pretrained_G14,
                        pretrained_D15,
                        gpus16,
                        if_cache_gpu17,
                        if_save_every_weights18,
                        model_version,
                    ],
                    [training_info, training_plot],
                    api_name="train_start",
                )
                index_btn.click(
                    train_index, [experiment_name, model_version], training_info
                )
                one_click_btn.click(
                    one_click_training,
                    [
                        experiment_name,
                        target_sr,
                        use_f0,
                        audio_data_root,
                        spk_id,
                        cpu_count,
                        f0method8,
                        save_epoch,
                        total_epoch,
                        batch_size,
                        if_save_latest13,
                        pretrained_G14,
                        pretrained_D15,
                        gpus16,
                        if_cache_gpu17,
                        if_save_every_weights18,
                        model_version,
                        gpus_rmvpe,
                    ],
                    training_info,
                    api_name="train_start_all",
                )
