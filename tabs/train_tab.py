import datetime
import json
import os
import pathlib
import platform
import re
import shutil
import subprocess
import threading
import traceback
from collections.abc import Generator
from random import shuffle
from subprocess import Popen
from time import sleep

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


def if_done_multi(done_flag, p_objs):
    for p_obj in p_objs:
        p_obj.wait()
    done_flag[0] = True


def preprocess_dataset(
    audio_dir: str, exp_dir: str, sr: int, n_p: int, progress=gr.Progress()
) -> Generator[str, None, None]:
    # 1. Validate audio_dir and count files
    if not os.path.isdir(audio_dir):
        error_msg = (
            f"Error: Audio directory '{audio_dir}' not found or is not a directory."
        )
        shared.logger.error(error_msg)
        yield error_msg
        return

    actual_file_count = 0
    try:
        # List all entries in the directory and filter for files
        file_names = [
            name
            for name in os.listdir(audio_dir)
            if os.path.isfile(os.path.join(audio_dir, name))
        ]
        actual_file_count = len(file_names)
        info_msg = f"Found {actual_file_count} files in audio directory: {audio_dir}"
        shared.logger.info(info_msg)
        # yield info_msg # Optionally yield this information to the UI

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
    except OSError as e:
        error_msg = (
            f"Error: Could not access audio directory '{audio_dir}' to count files: {e}"
        )
        shared.logger.error(error_msg)
        yield error_msg
        return
    sr = shared.sr_dict[sr]
    os.makedirs(f"{shared.now_dir}/logs/{exp_dir}", exist_ok=True)
    f = open(f"{shared.now_dir}/logs/{exp_dir}/preprocess.log", "w")
    f.close()
    cmd = f'"{shared.config.python_cmd}" infer/modules/train/preprocess.py "{audio_dir}" {sr} {n_p} "{shared.now_dir}/logs/{exp_dir}" {shared.config.noparallel} {shared.config.preprocess_per:.1f}'
    shared.logger.info("Execute: " + cmd)
    p = Popen(cmd, shell=True)
    # 煞笔gr, popen read都非得全跑完了再一次性读取, 不用gr就正常读一句输出一句;只能额外弄出一个文本流定时读
    done = [False]
    threading.Thread(
        target=if_done,
        args=(
            done,
            p,
        ),
    ).start()

    while True:
        with open(f"{shared.now_dir}/logs/{exp_dir}/preprocess.log") as f:
            file_content = f.read()
            count = file_content.count("Success")
            progress(
                float(count) / actual_file_count,
                desc=f"Processed {count}/{actual_file_count} audio...",
            )
        sleep(0.5)
        if done[0]:
            break
    with open(f"{shared.now_dir}/logs/{exp_dir}/preprocess.log") as f:
        log = f.read()
    shared.logger.info(log)
    yield log


def preprocess_meta(
    experiment_name: str,
    audio_dir: str,
    audio_files: list[str] | None,
    sr: int,
    n_p: int,
    progress=gr.Progress(),
):
    save_dir = f"{audio_dir}/{experiment_name}"
    os.makedirs(save_dir, exist_ok=True)

    if audio_files is not None:
        for idx, audio_file in enumerate(audio_files):
            audio_file_name = os.path.basename(audio_file)
            shutil.copy(audio_file, f"{save_dir}/{audio_file_name}")
            progress(idx / len(audio_files), "Copying files...")

    yield from preprocess_dataset(
        audio_dir=save_dir,
        exp_dir=experiment_name,
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
    gpus: str,
    n_p: int,
    f0method: str,
    if_f0: bool,
    exp_dir: str,
    version19: str,
    gpus_rmvpe: str,
    progress: gr.Progress = gr.Progress(),
) -> Generator[str, None, None]:
    def update_progress(content: str):
        now, all = parse_f0_feature_log(content)
        progress(float(now) / all, desc=f"{now}/{all} Features extracted...")

    gpus: list[str] = gpus.split("-")
    os.makedirs(f"{shared.now_dir}/logs/{exp_dir}", exist_ok=True)
    f = open(f"{shared.now_dir}/logs/{exp_dir}/extract_f0_feature.log", "w")
    f.close()
    if if_f0:
        if f0method != "rmvpe_gpu":
            cmd = f'"{shared.config.python_cmd}" infer/modules/train/extract/extract_f0_print.py "{shared.now_dir}/logs/{exp_dir}" {n_p} {f0method}'
            shared.logger.info("Execute: " + cmd)
            p = Popen(cmd, shell=True, cwd=shared.now_dir)
            # 煞笔gr, popen read都非得全跑完了再一次性读取, 不用gr就正常读一句输出一句;只能额外弄出一个文本流定时读
            done = [False]
            threading.Thread(
                target=if_done,
                args=(
                    done,
                    p,
                ),
            ).start()
        else:
            if gpus_rmvpe != "-":
                gpus_rmvpe = gpus_rmvpe.split("-")
                length = len(gpus_rmvpe)
                ps = []
                for idx, n_g in enumerate(gpus_rmvpe):
                    cmd = f'"{shared.config.python_cmd}" infer/modules/train/extract/extract_f0_rmvpe.py {length} {idx} {n_g} "{shared.now_dir}/logs/{exp_dir}" {shared.config.is_half} '
                    shared.logger.info("Execute: " + cmd)
                    p = Popen(cmd, shell=True, cwd=shared.now_dir)
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
                    + f' infer/modules/train/extract/extract_f0_rmvpe_dml.py "{shared.now_dir}/logs/{exp_dir}" '
                )
                shared.logger.info("Execute: " + cmd)
                p = Popen(cmd, shell=True, cwd=shared.now_dir)
                p.wait()
                done = [True]
        while True:
            with open(f"{shared.now_dir}/logs/{exp_dir}/extract_f0_feature.log") as f:
                update_progress(f.read())
            sleep(1)
            if done[0]:
                break
        with open(f"{shared.now_dir}/logs/{exp_dir}/extract_f0_feature.log") as f:
            log = f.read()
        shared.logger.info(log)
    # 对不同part分别开多进程
    """
    n_part=int(sys.argv[1])
    i_part=int(sys.argv[2])
    i_gpu=sys.argv[3]
    exp_dir=sys.argv[4]
    os.environ["CUDA_VISIBLE_DEVICES"]=str(i_gpu)
    """
    length = len(gpus)
    ps = []
    for idx, n_g in enumerate(gpus):
        cmd = f'"{shared.config.python_cmd}" infer/modules/train/extract_feature_print.py {shared.config.device} {length} {idx} {n_g} "{shared.now_dir}/logs/{exp_dir}" {version19} {shared.config.is_half}'
        shared.logger.info("Execute: " + cmd)
        p = Popen(cmd, shell=True, cwd=shared.now_dir)
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
        with open(f"{shared.now_dir}/logs/{exp_dir}/extract_f0_feature.log") as f:
            update_progress(f.read())
        sleep(1)
        if done[0]:
            break
    with open(f"{shared.now_dir}/logs/{exp_dir}/extract_f0_feature.log") as f:
        log = f.read()
    shared.logger.info(log)
    yield log


def get_pretrained_models(path_str: str, f0_str: str, sr2: int):
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


def change_sr2(sr2: int, if_f0_3, version19):
    path_str = "" if version19 == "v1" else "_v2"
    f0_str = "f0" if if_f0_3 else ""
    return get_pretrained_models(path_str, f0_str, sr2)


def change_version19(sr2: int, if_f0_3: bool, version19: str):
    path_str = "" if version19 == "v1" else "_v2"
    if sr2 == "32k" and version19 == "v1":
        sr2 = "40k"
    to_return_sr2 = (
        {"choices": ["40k", "48k"], "__type__": "update", "value": sr2}
        if version19 == "v1"
        else {"choices": ["40k", "48k", "32k"], "__type__": "update", "value": sr2}
    )
    f0_str = "f0" if if_f0_3 else ""
    return (
        *get_pretrained_models(path_str, f0_str, sr2),
        to_return_sr2,
    )


def change_f0(if_f0_3: bool, sr2, version19):  # f0method8,pretrained_G14,pretrained_D15
    path_str = "" if version19 == "v1" else "_v2"
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
    sr2: int,
    if_f0_3,
    spk_id5,
    save_epoch10,
    total_epoch11,
    batch_size12,
    if_save_latest13: str,
    pretrained_G14,
    pretrained_D15,
    gpus16,
    if_cache_gpu17,
    if_save_every_weights18,
    version19,
    progress=gr.Progress(),
):
    # Generating file list
    exp_dir = f"{shared.now_dir}/logs/{exp_dir1}"
    os.makedirs(exp_dir, exist_ok=True)
    gt_wavs_dir = f"{exp_dir}/0_gt_wavs"
    feature_dir = (
        f"{exp_dir}/3_feature256" if version19 == "v1" else f"{exp_dir}/3_feature768"
    )
    if if_f0_3:
        f0_dir = f"{exp_dir}/2a_f0"
        f0nsf_dir = f"{exp_dir}/2b-f0nsf"
        names = (
            {name.split(".")[0] for name in os.listdir(gt_wavs_dir)}
            & {name.split(".")[0] for name in os.listdir(feature_dir)}
            & {name.split(".")[0] for name in os.listdir(f0_dir)}
            & {name.split(".")[0] for name in os.listdir(f0nsf_dir)}
        )
    else:
        names = {name.split(".")[0] for name in os.listdir(gt_wavs_dir)} & {
            name.split(".")[0] for name in os.listdir(feature_dir)
        }
    opt = []
    for name in names:
        if if_f0_3:
            opt.append(
                "{}/{}.wav|{}/{}.npy|{}/{}.wav.npy|{}/{}.wav.npy|{}".format(
                    gt_wavs_dir.replace("\\", "\\\\"),
                    name,
                    feature_dir.replace("\\", "\\\\"),
                    name,
                    f0_dir.replace("\\", "\\\\"),
                    name,
                    f0nsf_dir.replace("\\", "\\\\"),
                    name,
                    spk_id5,
                )
            )
        else:
            opt.append(
                "{}/{}.wav|{}/{}.npy|{}".format(
                    gt_wavs_dir.replace("\\", "\\\\"),
                    name,
                    feature_dir.replace("\\", "\\\\"),
                    name,
                    spk_id5,
                )
            )
    fea_dim = 256 if version19 == "v1" else 768
    if if_f0_3:
        for _ in range(2):
            opt.append(
                f"{shared.now_dir}/logs/mute/0_gt_wavs/mute{sr2}.wav|{shared.now_dir}/logs/mute/3_feature{fea_dim}/mute.npy|{shared.now_dir}/logs/mute/2a_f0/mute.wav.npy|{shared.now_dir}/logs/mute/2b-f0nsf/mute.wav.npy|{spk_id5}"
            )
    else:
        for _ in range(2):
            opt.append(
                f"{shared.now_dir}/logs/mute/0_gt_wavs/mute{sr2}.wav|{shared.now_dir}/logs/mute/3_feature{fea_dim}/mute.npy|{spk_id5}"
            )
    shuffle(opt)
    with open(f"{exp_dir}/filelist.txt", "w") as f:
        f.write("\n".join(opt))
    shared.logger.debug("Write filelist done")
    # 生成config#无需生成config
    shared.logger.info("Use gpus: %s", str(gpus16))
    if pretrained_G14 == "":
        shared.logger.info("No pretrained Generator")
    if pretrained_D15 == "":
        shared.logger.info("No pretrained Discriminator")
    if version19 == "v1" or sr2 == "40k":
        config_path = f"v1/{sr2}.json"
    else:
        config_path = f"v2/{sr2}.json"
    config_save_path = os.path.join(exp_dir, "config.json")
    if not pathlib.Path(config_save_path).exists():
        with open(config_save_path, "w", encoding="utf-8") as f:
            json.dump(
                shared.config.json_config[config_path],
                f,
                ensure_ascii=False,
                indent=4,
                sort_keys=True,
            )
            f.write("\n")
    if gpus16:
        cmd = '"{}" infer/modules/train/train.py -e "{}" -sr {} -f0 {} -bs {} -g {} -te {} -se {} {} {} -l {} -c {} -sw {} -v {}'.format(
            shared.config.python_cmd,
            exp_dir1,
            sr2,
            1 if if_f0_3 else 0,
            batch_size12,
            gpus16,
            total_epoch11,
            save_epoch10,
            f"-pg {pretrained_G14}" if pretrained_G14 != "" else "",
            f"-pd {pretrained_D15}" if pretrained_D15 != "" else "",
            1 if if_save_latest13 == i18n("Yes") else 0,
            1 if if_cache_gpu17 == i18n("Yes") else 0,
            1 if if_save_every_weights18 == i18n("Yes") else 0,
            version19,
        )
    else:
        cmd = '"{}" infer/modules/train/train.py -e "{}" -sr {} -f0 {} -bs {} -te {} -se {} {} {} -l {} -c {} -sw {} -v {}'.format(
            shared.config.python_cmd,
            exp_dir1,
            sr2,
            1 if if_f0_3 else 0,
            batch_size12,
            total_epoch11,
            save_epoch10,
            f"-pg {pretrained_G14}" if pretrained_G14 != "" else "",
            f"-pd {pretrained_D15}" if pretrained_D15 != "" else "",
            1 if if_save_latest13 == shared.i18n("Yes") else 0,
            1 if if_cache_gpu17 == shared.i18n("Yes") else 0,
            1 if if_save_every_weights18 == shared.i18n("Yes") else 0,
            version19,
        )
    shared.logger.info("Execute: " + cmd)
    current_epoch = 0
    p = Popen(cmd, shell=True, cwd=shared.now_dir, stdout=subprocess.PIPE)
    scalar_count = 0
    while True:
        line = p.stdout.readline()
        if not line:
            break
        # the real code does filtering here
        line: str = line.decode("utf-8", errors="ignore")
        shared.logger.info(f"{line}")

        if line.startswith("SCALAR_DICT: "):
            try:
                scalar_dict: dict = json.loads(line.replace("SCALAR_DICT: ", ""))
                scalar_dict["index"] = scalar_count
                scalar_count += 1

                # Step 1: Append the dictionary to the history
                scalar_history.append(scalar_dict)

                # Step 2: Convert the history list to a pandas DataFrame
                df = pd.DataFrame(scalar_history)
                # Returning the plot data will update the plot component
                # The yield statement is necessary to update the Gradio UI in real-time
                # within a loop.
                print(f"history: {scalar_history}")
                yield (
                    "",
                    df,
                )  # Yielding the empty string updates info3, and plot_data updates the plot
            except Exception:
                pass

        current_epoch = parse_epoch_from_train_log_line(line) or current_epoch
        progress(current_epoch / total_epoch11, desc="Training...")

    p.wait()
    yield (
        "Training finished with exit code {return_code}.",
        pd.DataFrame(scalar_history),
    )


def train_index(exp_dir1: str, version19: str, progress=gr.Progress()):
    exp_dir = f"logs/{exp_dir1}"
    os.makedirs(exp_dir, exist_ok=True)
    feature_dir = (
        f"{exp_dir}/3_feature256" if version19 == "v1" else f"{exp_dir}/3_feature768"
    )
    if not os.path.exists(feature_dir):
        return "Please perform feature extraction first!"
    listdir_res = list(os.listdir(feature_dir))
    if len(listdir_res) == 0:
        return "Please perform feature extraction first!"

    progress(0.05, desc="Loading features...")  # Initial progress update
    infos = []
    npys = []
    for name in sorted(listdir_res):
        phone = np.load(f"{feature_dir}/{name}")
        npys.append(phone)
    big_npy = np.concatenate(npys, 0)
    big_npy_idx = np.arange(big_npy.shape[0])
    np.random.shuffle(big_npy_idx)
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
    index = faiss.index_factory(256 if version19 == "v1" else 768, f"IVF{n_ivf},Flat")
    infos.append("training")
    index_ivf = faiss.extract_index_ivf(index)  #
    index_ivf.nprobe = 1
    index.train(big_npy)
    faiss.write_index(
        index,
        f"{exp_dir}/trained_IVF{n_ivf}_Flat_nprobe_{index_ivf.nprobe}_{exp_dir1}_{version19}.index",
    )
    progress(0.7, desc="Adding vectors to index...")
    infos.append("Adding vectors to index...")
    batch_size_add = 8192
    for i in range(0, big_npy.shape[0], batch_size_add):
        index.add(big_npy[i : i + batch_size_add])
    faiss.write_index(
        index,
        f"{exp_dir}/added_IVF{n_ivf}_Flat_nprobe_{index_ivf.nprobe}_{exp_dir1}_{version19}.index",
    )
    infos.append(
        f"Successfully built index: added_IVF{n_ivf}_Flat_nprobe_{index_ivf.nprobe}_{exp_dir1}_{version19}.index"
    )
    try:
        link = os.link if platform.system() == "Windows" else os.symlink
        link(
            f"{exp_dir}/added_IVF{n_ivf}_Flat_nprobe_{index_ivf.nprobe}_{exp_dir1}_{version19}.index",
            f"{shared.outside_index_root}/{exp_dir1}_IVF{n_ivf}_Flat_nprobe_{index_ivf.nprobe}_{exp_dir1}_{version19}.index",
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
    exp_dir1,
    sr2,
    if_f0_3,
    trainset_dir4,
    spk_id5,
    np7,
    f0method8,
    save_epoch10,
    total_epoch11,
    batch_size12,
    if_save_latest13,
    pretrained_G14,
    pretrained_D15,
    gpus16,
    if_cache_gpu17,
    if_save_every_weights18,
    version19,
    gpus_rmvpe,
):
    infos: list[str] = []

    def get_info_str(strr):
        infos.append(strr)
        return "\n".join(infos)

    yield get_info_str(shared.i18n("step1: processing data..."))
    [get_info_str(_) for _ in preprocess_dataset(trainset_dir4, exp_dir1, sr2, np7)]

    yield get_info_str(shared.i18n("step2: extracting feature & pitch"))
    [
        get_info_str(_)
        for _ in extract_f0_feature(
            gpus16, np7, f0method8, if_f0_3, exp_dir1, version19, gpus_rmvpe
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
        version19,
    )
    yield get_info_str(
        i18n("训练结束, 您可查看控制台训练日志或实验文件夹下的train.log")
    )

    [get_info_str(_) for _ in train_index(exp_dir1, version19)]
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
                    info1 = gr.Textbox(label=i18n("Info"), value="")
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
                    y=["loss/g/total", "loss/d/total"],
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
