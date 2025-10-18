import json
import logging
import os
import platform
import shutil
import sys
import threading
import traceback
import warnings
from pathlib import Path
from random import shuffle
from subprocess import Popen
from time import sleep

import faiss
import gradio as gr
import numpy as np
import torch
from dotenv import load_dotenv
from fairseq.modules.grad_multiply import GradMultiply
from sklearn.cluster import MiniBatchKMeans

from configs.config import Config
from i18n.i18n import I18nAuto
from infer.modules.vc.modules import VC

now_dir = Path.cwd()
sys.path.append(str(now_dir))
load_dotenv()

logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

tmp = now_dir / "TEMP"
shutil.rmtree(tmp, ignore_errors=True)
shutil.rmtree(now_dir / "runtime/Lib/site-packages/infer_pack", ignore_errors=True)
shutil.rmtree(now_dir / "runtime/Lib/site-packages/uvr5_pack", ignore_errors=True)
tmp.mkdir(parents=True, exist_ok=True)
(now_dir / "logs").mkdir(parents=True, exist_ok=True)
(now_dir / "assets/weights").mkdir(parents=True, exist_ok=True)
os.environ["TEMP"] = str(tmp)
warnings.filterwarnings("ignore")
torch.manual_seed(114514)


config = Config()
vc = VC(config)


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
mem = []
if_gpu_ok = False

if torch.cuda.is_available() or ngpu != 0:
    for i in range(ngpu):
        gpu_name = torch.cuda.get_device_name(i)
        if any(
            value in gpu_name.upper()
            for value in [
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
        ):

            if_gpu_ok = True
            gpu_infos.append(f"{i}\t{gpu_name}")
            mem.append(
                int(
                    torch.cuda.get_device_properties(i).total_memory
                    / 1024
                    / 1024
                    / 1024
                    + 0.4
                )
            )
if if_gpu_ok and len(gpu_infos) > 0:
    gpu_info = "\n".join(gpu_infos)
    default_batch_size = min(mem) // 2
else:
    gpu_info = i18n("很遗憾您这没有能用的显卡来支持您训练")
    default_batch_size = 1
gpus = "-".join([i[0] for i in gpu_infos])


weight_root = Path(os.getenv("weight_root", ""))
weight_uvr5_root = Path(os.getenv("weight_uvr5_root", ""))
index_root = Path(os.getenv("index_root", ""))
outside_index_root = Path(os.getenv("outside_index_root", ""))

names = []
for name in weight_root.iterdir():
    if name.suffix == ".pth":
        names.append(name)
index_paths = []


def _lookup_indices(index_root: os.PathLike):
    for root, _dirs, files in os.walk(index_root, topdown=False):
        found_files = [Path(f) for f in files]
        for name in found_files:
            if name.suffix == ".index" and "trained" not in name.stem:
                index_paths.append(f"{root}/{name}")


_lookup_indices(index_root)
_lookup_indices(outside_index_root)
_uvr5_names = []
for name in weight_uvr5_root.iterdir():
    if name.suffix == ".pth" or "onnx" in name.name:
        _uvr5_names.append(name.stem)


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


def _if_done_multi(done, ps):
    while 1:

        flag = 1
        for p in ps:
            if p.poll() is None:
                flag = 0
                sleep(0.5)
                break
        if flag == 1:
            break
    done[0] = True


def _preprocess_dataset(trainset_dir: str, exp_dir: str, sr: str, n_p: int):
    sr = _sr_dict[sr]  # type: ignore

    logs_directory = now_dir / "logs" / exp_dir
    logs_directory.mkdir(parents=True, exist_ok=True)
    (logs_directory / "preprocess.log").touch()
    cmd = f'"{config.python_cmd}" infer/modules/train/preprocess.py "{trainset_dir}" {sr} {n_p} "{now_dir}/logs/{exp_dir}" {config.noparallel} {config.preprocess_per:.1f}'
    logger.info("Execute: %s", cmd)
    p = Popen(cmd, shell=True)
    done = [False]
    threading.Thread(
        target=_if_done,
        args=(
            done,
            p,
        ),
    ).start()
    while True:
        with (now_dir / "logs" / exp_dir / "preprocess.log").open("r") as f:
            yield (f.read())
        sleep(1)
        if done[0]:
            break
    with (now_dir / "logs" / exp_dir / "preprocess.log").open("r") as f:
        log = f.read()
    logger.info(log)
    yield log


def _extract_f0_feature(  # noqa: PLR0913
    gpus, n_p, f0method, if_f0, exp_dir, version19, gpus_rmvpe
):
    gpus = gpus.split("-")

    logs_directory = now_dir / "logs" / exp_dir
    logs_directory.mkdir(parents=True, exist_ok=True)
    (logs_directory / "extract_f0_feature.log").touch()
    if if_f0:
        if f0method != "rmvpe_gpu":
            cmd = f'"{config.python_cmd}" infer/modules/train/extract/extract_f0_print.py "{now_dir}/logs/{exp_dir}" {n_p} {f0method}'
            logger.info("Execute: %s", cmd)
            p = Popen(cmd, shell=True, cwd=now_dir)
            done = [False]
            threading.Thread(
                target=_if_done,
                args=(
                    done,
                    p,
                ),
            ).start()
        elif gpus_rmvpe != "-":
            gpus_rmvpe = gpus_rmvpe.split("-")
            leng = len(gpus_rmvpe)
            ps = []
            for idx, n_g in enumerate(gpus_rmvpe):
                cmd = f'"{config.python_cmd}" infer/modules/train/extract/extract_f0_rmvpe.py {leng} {idx} {n_g} "{now_dir}/logs/{exp_dir}" {config.is_half}'
                logger.info("Execute: %s", cmd)
                p = Popen(cmd, shell=True, cwd=now_dir)
                ps.append(p)
            done = [False]
            threading.Thread(
                target=_if_done_multi,
                args=(
                    done,
                    ps,
                ),
            ).start()
        else:
            cmd = (
                config.python_cmd
                + f' infer/modules/train/extract/extract_f0_rmvpe_dml.py "{now_dir}/logs/{exp_dir}"'
            )
            logger.info("Execute: %s", cmd)
            p = Popen(cmd, shell=True, cwd=now_dir)
            p.wait()
            done = [True]
        while True:
            with (now_dir / "logs" / exp_dir / "extract_f0_feature.log").open("r") as f:
                yield (f.read())
            sleep(1)
            if done[0]:
                break
        with (now_dir / "logs" / exp_dir / "extract_f0_feature.log").open("r") as f:
            log = f.read()
        logger.info(log)
        yield log

    leng = len(gpus)
    ps = []
    for idx, n_g in enumerate(gpus):
        cmd = f'"{config.python_cmd}" infer/modules/train/extract_feature_print.py {config.device} {leng} {idx} {n_g} "{now_dir}/logs/{exp_dir}" {version19} {config.is_half}'
        logger.info("Execute: %s", cmd)
        p = Popen(cmd, shell=True, cwd=now_dir)
        ps.append(p)
    done = [False]
    threading.Thread(
        target=_if_done_multi,
        args=(
            done,
            ps,
        ),
    ).start()
    while True:
        with (now_dir / "logs" / exp_dir / "extract_f0_feature.log").open("r") as f:
            yield (f.read())
        sleep(1)
        if done[0]:
            break
    with (now_dir / "logs" / exp_dir / "extract_f0_feature.log").open("r") as f:
        log = f.read()
    logger.info(log)
    yield log


def _get_pretrained_models(path_str: str, f0_str: str, sr2: str):
    if_pretrained_generator_exist = os.access(
        f"assets/pretrained{path_str}/{f0_str}G{sr2}.pth", os.F_OK
    )
    if_pretrained_discriminator_exist = os.access(
        f"assets/pretrained{path_str}/{f0_str}D{sr2}.pth", os.F_OK
    )
    if not if_pretrained_generator_exist:
        logger.warning(
            "assets/pretrained%s/%sG%s.pth not exist, will not use pretrained model",
            path_str,
            f0_str,
            sr2,
        )
    if not if_pretrained_discriminator_exist:
        logger.warning(
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


def _change_sr2(sr2: str, if_f0_3: bool, version19: str):
    path_str = "" if version19 == "v1" else "_v2"
    f0_str = "f0" if if_f0_3 else ""
    return _get_pretrained_models(path_str, f0_str, sr2)


def _change_version19(sr2, if_f0_3, version19):
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
        *_get_pretrained_models(path_str, f0_str, sr2),
        to_return_sr2,
    )


def _change_f0(if_f0_3, sr2, version19):
    path_str = "" if version19 == "v1" else "_v2"
    return (
        {"visible": if_f0_3, "__type__": "update"},
        {"visible": if_f0_3, "__type__": "update"},
        *_get_pretrained_models(path_str, "f0" if if_f0_3 else "", sr2),
    )


def _click_train(
    exp_dir1: str,
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
):
    exp_dir = now_dir / "logs" / exp_dir1
    exp_dir.mkdir(parents=True, exist_ok=True)
    gt_wavs_dir = exp_dir / "0_gt_wavs"
    feature_dir = (
        exp_dir / "3_feature256" if version19 == "v1" else exp_dir / "3_feature768"
    )
    if if_f0_3:
        f0_dir = exp_dir / "2a_f0"
        f0nsf_dir = exp_dir / "2b-f0nsf"
        names = (
            {name.stem for name in gt_wavs_dir.iterdir()}
            & {name.stem for name in feature_dir.iterdir()}
            & {name.stem for name in f0_dir.iterdir()}
            & {name.stem for name in f0nsf_dir.iterdir()}
        )
    else:
        names = {name.stem for name in gt_wavs_dir.iterdir()} & {
            name.stem for name in feature_dir.iterdir()
        }
    opt = []
    for name in names:
        if if_f0_3:
            opt.append(
                f"{gt_wavs_dir}/{name}.wav|{feature_dir}/{name}.npy|{f0_dir}/{name}.wav.npy|{f0nsf_dir}/{name}.wav.npy|{spk_id5}"
            )
        else:
            opt.append(f"{gt_wavs_dir}/{name}.wav|{feature_dir}/{name}.npy|{spk_id5}")
    fea_dim = 256 if version19 == "v1" else 768
    if if_f0_3:
        for _ in range(2):
            opt.append(
                f"{now_dir}/logs/mute/0_gt_wavs/mute{sr2}.wav|{now_dir}/logs/mute/3_feature{fea_dim}/mute.npy|{now_dir}/logs/mute/2a_f0/mute.wav.npy|{now_dir}/logs/mute/2b-f0nsf/mute.wav.npy|{spk_id5}"
            )
    else:
        for _ in range(2):
            opt.append(
                f"{now_dir}/logs/mute/0_gt_wavs/mute{sr2}.wav|{now_dir}/logs/mute/3_feature{fea_dim}/mute.npy|{spk_id5}"
            )
    shuffle(opt)
    with Path(exp_dir).open("w") as f:
        f.write("\n".join(opt))
    logger.debug("Write filelist done")
    logger.info("Use gpus: %s", str(gpus16))
    if pretrained_G14 == "":
        logger.info("No pretrained Generator")
    if pretrained_D15 == "":
        logger.info("No pretrained Discriminator")
    if version19 == "v1" or sr2 == "40k":
        config_path = "v1/%s.json" % sr2
    else:
        config_path = "v2/%s.json" % sr2
    config_save_path = exp_dir / "config.json"
    if not config_save_path.exists():
        with Path(config_save_path).open("w", encoding="utf-8") as f:
            json.dump(
                config.json_config[config_path],
                f,
                ensure_ascii=False,
                indent=4,
                sort_keys=True,
            )
            f.write("\n")
    if gpus16:
        cmd = f'"{config.python_cmd}" infer/modules/train/train.py -e "{exp_dir1}" -sr {sr2} -f0 {1 if if_f0_3 else 0} -bs {batch_size12} -g {gpus16} -te {total_epoch11} -se {save_epoch10} {f"-pg {pretrained_G14}" if pretrained_G14 != "" else ""} {f"-pd {pretrained_D15}" if pretrained_D15 != "" else ""} -l {1 if if_save_latest13 == i18n("是") else 0} -c {1 if if_cache_gpu17 == i18n("是") else 0} -sw {1 if if_save_every_weights18 == i18n("是") else 0} -v {version19}'
    else:
        cmd = f'"{config.python_cmd}" infer/modules/train/train.py -e "{exp_dir1}" -sr {sr2} -f0 {1 if if_f0_3 else 0} -bs {batch_size12} -te {total_epoch11} -se {save_epoch10} {f"-pg {pretrained_G14}" if pretrained_G14 != "" else ""} {f"-pd {pretrained_D15}" if pretrained_D15 != "" else ""} -l {1 if if_save_latest13 == i18n("是") else 0} -c {1 if if_cache_gpu17 == i18n("是") else 0} -sw {1 if if_save_every_weights18 == i18n("是") else 0} -v {version19}'
    logger.info("Execute: %s", cmd)
    p = Popen(cmd, shell=True, cwd=now_dir)
    p.wait()
    return "训练结束, 您可查看控制台训练日志或实验文件夹下的train.log"


def _train_index(exp_dir1, version19):  # noqa: PLR0915
    exp_dir = Path(f"logs/{exp_dir1}")
    exp_dir.mkdir(parents=True, exist_ok=True)
    feature_dir = (
        exp_dir / "3_feature256" if version19 == "v1" else exp_dir / "3_feature768"
    )
    if not feature_dir.exists():
        return "请先进行特征提取!"
    listdir_res = list(feature_dir.iterdir())
    if len(listdir_res) == 0:
        return "请先进行特征提取！"
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
        infos.append(f"Trying doing kmeans {big_npy.shape[0]} shape to 10k centers.")
        yield "\n".join(infos)
        try:
            big_npy = (
                MiniBatchKMeans(
                    n_clusters=10000,
                    verbose=True,
                    batch_size=256 * config.n_cpu,
                    compute_labels=False,
                    init="random",
                )
                .fit(big_npy)
                .cluster_centers_
            )
        except:
            info = traceback.format_exc()
            logger.info(info)
            infos.append(info)
            yield "\n".join(infos)

    np.save(f"{exp_dir}/total_fea.npy", big_npy)
    n_ivf = min(int(16 * np.sqrt(big_npy.shape[0])), big_npy.shape[0] // 39)
    infos.append(f"{big_npy.shape},{n_ivf}")
    yield "\n".join(infos)
    index = faiss.index_factory(256 if version19 == "v1" else 768, f"IVF{n_ivf},Flat")

    infos.append("training")
    yield "\n".join(infos)
    index_ivf = faiss.extract_index_ivf(index)
    index_ivf.nprobe = 1
    index.train(big_npy)
    faiss.write_index(
        index,
        f"{exp_dir}/trained_IVF{n_ivf}_Flat_nprobe_{index_ivf.nprobe}_{exp_dir1}_{version19}.index",
    )
    infos.append("adding")
    yield "\n".join(infos)
    batch_size_add = 8192
    for i in range(0, big_npy.shape[0], batch_size_add):
        index.add(big_npy[i : i + batch_size_add])
    faiss.write_index(
        index,
        f"{exp_dir}/added_IVF{n_ivf}_Flat_nprobe_{index_ivf.nprobe}_{exp_dir1}_{version19}.index",
    )
    infos.append(
        f"成功构建索引 added_IVF{n_ivf}_Flat_nprobe_{index_ivf.nprobe}_{exp_dir1}_{version19}.index"
    )
    try:
        file_name = (
            f"IVF{n_ivf}_Flat_nprobe_{index_ivf.nprobe}_{exp_dir1}_{version19}.index"
        )
        source_path = Path(exp_dir) / f"added_{file_name}"
        target_path = outside_index_root / f"{exp_dir1}_{file_name}"
        if platform.system() == "Windows":
            source_path.hardlink_to(target_path)
        else:
            source_path.symlink_to(target_path)
        infos.append(f"链接索引到外部-{outside_index_root}")
    except:
        infos.append(f"链接索引到外部-{outside_index_root}失败")

    yield "\n".join(infos)


def _train1key(  # noqa: PLR0913
    exp_dir1,
    sr2,
    if_f0_3,
    trainset_dir4: str,
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
    infos = []

    def get_info_str(strr):
        infos.append(strr)
        return "\n".join(infos)

    yield get_info_str(i18n("step1:正在处理数据"))
    [get_info_str(_) for _ in _preprocess_dataset(trainset_dir4, exp_dir1, sr2, np7)]

    yield get_info_str(i18n("step2:正在提取音高&正在提取特征"))
    [
        get_info_str(_)
        for _ in _extract_f0_feature(
            gpus16, np7, f0method8, if_f0_3, exp_dir1, version19, gpus_rmvpe
        )
    ]

    yield get_info_str(i18n("step3a:正在训练模型"))
    _click_train(
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

    [get_info_str(_) for _ in _train_index(exp_dir1, version19)]
    yield get_info_str(i18n("全流程结束！"))


_F0GPUVisible = not config.dml


def _change_f0_method(f0method8):
    if f0method8 == "rmvpe_gpu":
        visible = _F0GPUVisible
    else:
        visible = False
    return {"visible": visible, "__type__": "update"}


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
    with gr.Tabs():
        with gr.TabItem(i18n("Train")):
            with gr.Group():
                # trainset_dir4 = gr.File(
                #     label=i18n("Upload training file"),
                #     file_count="single",
                # )
                trainset_dir4 = gr.Textbox(
                    label=i18n("Enter training folder path"),
                    value=i18n("E:\\VoiceAudio+Annotations\\YonezuKenshi\\src"),
                )
                gr.Markdown(
                    value=i18n(
                        "step2b: Use CPU to extract pitch (if the model includes pitch), "
                        "use GPU to extract features (select GPU IDs)"
                    )
                )
                with gr.Row():
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
                        label=i18n(
                            "Number of CPU processes for pitch extraction and data processing"
                        ),
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
                    gr_experiment_dir = gr.Textbox(
                        label=i18n("Enter experiment name"), value="mi-test"
                    )
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
                    with gr.Column():
                        gpu_ids_input = gr.Textbox(
                            label=i18n(
                                "Enter GPU IDs separated by '-', e.g. 0-1-2 to use GPU 0, 1, and 2"
                            ),
                            value=gpus,
                            interactive=True,
                            visible=_F0GPUVisible,
                        )
                        gpu_status_display = gr.Textbox(
                            label=i18n("GPU Information"),
                            value=gpu_info,
                            visible=_F0GPUVisible,
                        )
                    with gr.Column():
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
                            value="%s-%s" % (gpus, gpus),
                            interactive=True,
                            visible=_F0GPUVisible,
                        )
                    btn_extract_features = gr.Button(
                        i18n("Extract Features"), variant="primary"
                    )
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
            with gr.Group():
                gr.Markdown(
                    value=i18n(
                        "step3: Fill in training settings, start training model and index"
                    )
                )
                with gr.Row():
                    save_epoch_frequency = gr.Slider(
                        minimum=1,
                        maximum=50,
                        step=1,
                        label=i18n("Save frequency (save_every_epoch)"),
                        value=5,
                        interactive=True,
                    )
                    total_training_epochs = gr.Slider(
                        minimum=2,
                        maximum=1000,
                        step=1,
                        label=i18n("Total training epochs (total_epoch)"),
                        value=20,
                        interactive=True,
                    )
                    gpu_batch_size = gr.Slider(
                        minimum=1,
                        maximum=40,
                        step=1,
                        label=i18n("Batch size per GPU"),
                        value=default_batch_size,
                        interactive=True,
                    )
                    should_save_latest_model = gr.Radio(
                        label=i18n("Only save the latest ckpt file to save disk space"),
                        choices=[i18n("Yes"), i18n("No")],
                        value=i18n("No"),
                        interactive=True,
                    )
                    use_gpu_cache = gr.Radio(
                        label=i18n(
                            "Cache all training set to GPU memory. For small data under 10min, caching can speed up training. For large data, caching may cause out-of-memory and doesn't speed up much."
                        ),
                        choices=[i18n("Yes"), i18n("No")],
                        value=i18n("No"),
                        interactive=True,
                    )
                    is_save_every_weight = gr.Radio(
                        label=i18n(
                            "Save the final small model to the weights folder at each save time point"
                        ),
                        choices=[i18n("Yes"), i18n("No")],
                        value=i18n("No"),
                        interactive=True,
                    )
                with gr.Row():
                    gr_pretrained_G14 = gr.Textbox(
                        label=i18n("Load pretrained base model G path"),
                        value="assets/pretrained_v2/f0G40k.pth",
                        interactive=True,
                    )
                    gr_pretrained_D15 = gr.Textbox(
                        label=i18n("Load pretrained base model D path"),
                        value="assets/pretrained_v2/f0D40k.pth",
                        interactive=True,
                    )
                    gr_sample_rate.change(
                        _change_sr2,
                        [gr_sample_rate, include_pitch_guidance, gr_version],
                        [gr_pretrained_G14, gr_pretrained_D15],
                    )
                    gr_version.change(
                        _change_version19,
                        [gr_sample_rate, include_pitch_guidance, gr_version],
                        [gr_pretrained_G14, gr_pretrained_D15, gr_sample_rate],
                    )
                    include_pitch_guidance.change(
                        _change_f0,
                        [include_pitch_guidance, gr_sample_rate, gr_version],
                        [
                            pitch_extraction_method,
                            gpus_rmvpe,
                            gr_pretrained_G14,
                            gr_pretrained_D15,
                        ],
                    )
                    input_gpu_ids = gr.Textbox(
                        label=i18n(
                            "Enter GPU IDs separated by '-', e.g. 0-1-2 to use GPU 0, 1, and 2"
                        ),
                        value=gpus,
                        interactive=True,
                    )
                    btn_train_model = gr.Button(i18n("Train Model"), variant="primary")
                    btn_train_feature_index = gr.Button(
                        i18n("Train Feature Index"), variant="primary"
                    )
                    btn_one_click_training = gr.Button(
                        i18n("One-click Training"), variant="primary"
                    )
                    training_output_info = gr.Textbox(
                        label=i18n("Output Information"), value="", max_lines=10
                    )
                    btn_train_model.click(
                        _click_train,
                        [
                            gr_experiment_dir,
                            gr_sample_rate,
                            include_pitch_guidance,
                            speaker_id,
                            save_epoch_frequency,
                            total_training_epochs,
                            gpu_batch_size,
                            should_save_latest_model,
                            gr_pretrained_G14,
                            gr_pretrained_D15,
                            input_gpu_ids,
                            use_gpu_cache,
                            is_save_every_weight,
                            gr_version,
                        ],
                        training_output_info,
                        api_name="train_start",
                    )
                    btn_train_feature_index.click(
                        _train_index,
                        [gr_experiment_dir, gr_version],
                        training_output_info,
                    )
                    btn_one_click_training.click(
                        _train1key,
                        [
                            gr_experiment_dir,
                            gr_sample_rate,
                            include_pitch_guidance,
                            training_data_directory,
                            speaker_id,
                            num_cpu_processes,
                            pitch_extraction_method,
                            save_epoch_frequency,
                            total_training_epochs,
                            gpu_batch_size,
                            should_save_latest_model,
                            gr_pretrained_G14,
                            gr_pretrained_D15,
                            input_gpu_ids,
                            use_gpu_cache,
                            is_save_every_weight,
                            gr_version,
                            gpus_rmvpe,
                        ],
                        training_output_info,
                        api_name="train_start_all",
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
