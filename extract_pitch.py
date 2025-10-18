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
from gradio.components import FormComponent
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


class ToolButton(gr.Button, FormComponent):
    """Small button with single emoji as text, fits inside gradio forms"""

    def __init__(self, **kwargs):
        super().__init__(variant="secondary", **kwargs)

    def get_block_name(self):
        return "button"


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


def _change_choices():
    names = []
    for name in weight_root.iterdir():
        if name.suffix == ".pth":
            names.append(name)
    index_paths = []
    for root, _dirs, files in os.walk(index_root, topdown=False):
        found_files = [Path(f) for f in files]
        for name in found_files:
            if name.suffix == ".index" and "trained" not in name.stem:
                index_paths.append(f"{root}/{name}")
    return {"choices": sorted(names), "__type__": "update"}, {
        "choices": sorted(index_paths),
        "__type__": "update",
    }


def _clean():
    return {"value": "", "__type__": "update"}


def _export_onnx(ModelPath, ExportedPath):
    from infer.modules.onnx.export import export_onnx as eo  # noqa: PLC0415

    eo(ModelPath, ExportedPath)


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


def _preprocess_dataset(trainset_dir: str, exp_dir, sr, n_p):
    sr = _sr_dict[sr]

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


def _get_pretrained_models(path_str, f0_str, sr2):
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


def _change_sr2(sr2, if_f0_3, version19):
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


def _change_info_(ckpt_path: Path):
    if not (ckpt_path.parent / "train.log").exists():
        return {"__type__": "update"}, {"__type__": "update"}, {"__type__": "update"}
    try:
        with (ckpt_path.parent / "train.log").open("r") as f:
            info = eval(f.read().strip("\n").split("\n")[0].split("\t")[-1])
            sr, f0 = info["sample_rate"], info["if_f0"]
            version = "v2" if ("version" in info and info["version"] == "v2") else "v1"
            return sr, str(f0), version
    except:
        traceback.print_exc()
        return {"__type__": "update"}, {"__type__": "update"}, {"__type__": "update"}


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
            "本软件以MIT协议开源, 作者不对软件具备任何控制力, 使用软件者、传播软件导出的声音者自负全责. <br>如不认可该条款, 则不能使用或引用软件包内任何代码和文件. 详见根目录<b>LICENSE</b>."
        )
    )
    with gr.Tabs():
        with gr.TabItem(i18n("Train")):
            gr.Markdown(
                value=i18n(
                    "step1: Fill in experiment configuration. Experiment data is placed under logs, each experiment has a folder, you need to manually enter the experiment name path, which contains experiment configuration, logs, and model files obtained from training."
                )
            )
            with gr.Row():
                gr_experiment_dir = gr.Textbox(
                    label=i18n("Enter experiment name"), value="mi-test"
                )
                gr_sample_rate = gr.Radio(
                    label=i18n("Target sample rate"),
                    choices=["40k", "48k"],
                    value="40k",
                    interactive=True,
                )
                if_f0_3 = gr.Radio(
                    label=i18n(
                        "Does the model include pitch guidance (required for singing, optional for speech)"
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
                np7 = gr.Slider(
                    minimum=0,
                    maximum=config.n_cpu,
                    step=1,
                    label=i18n(
                        "Number of CPU processes for pitch extraction and data processing"
                    ),
                    value=int(np.ceil(config.n_cpu / 1.5)),
                    interactive=True,
                )
            with (
                gr.Group()
            ):  # Temporarily single-person, will support up to 4 people later # Data processing
                gr.Markdown(
                    value=i18n(
                        "step2a: Automatically traverse all files in the training folder that can be decoded into audio and perform slicing and normalization, generating 2 wav folders in the experiment directory; currently only supports single-person training."
                    )
                )
                with gr.Row():
                    trainset_dir4 = gr.Textbox(
                        label=i18n("输入训练文件夹路径"),
                        value=i18n("E:\\语音音频+标注\\米津玄师\\src"),
                    )
                    spk_id5 = gr.Slider(
                        minimum=0,
                        maximum=4,
                        step=1,
                        label=i18n("请指定说话人id"),
                        value=0,
                        interactive=True,
                    )
                    but1 = gr.Button(i18n("处理数据"), variant="primary")
                    info1 = gr.Textbox(label=i18n("输出信息"), value="")
                    but1.click(
                        _preprocess_dataset,
                        [trainset_dir4, gr_experiment_dir, gr_sample_rate, np7],
                        [info1],
                        api_name="train_preprocess",
                    )
            with gr.Group():
                gr.Markdown(
                    value=i18n(
                        "step2b: 使用CPU提取音高(如果模型带音高), 使用GPU提取特征(选择卡号)"
                    )
                )
                with gr.Row():
                    with gr.Column():
                        gpus6 = gr.Textbox(
                            label=i18n(
                                "以-分隔输入使用的卡号, 例如   0-1-2   使用卡0和卡1和卡2"
                            ),
                            value=gpus,
                            interactive=True,
                            visible=_F0GPUVisible,
                        )
                        gpu_info9 = gr.Textbox(
                            label=i18n("显卡信息"),
                            value=gpu_info,
                            visible=_F0GPUVisible,
                        )
                    with gr.Column():
                        f0method8 = gr.Radio(
                            label=i18n(
                                "选择音高提取算法:输入歌声可用pm提速,高质量语音但CPU差可用dio提速,harvest质量更好但慢,rmvpe效果最好且微吃CPU/GPU"
                            ),
                            choices=["pm", "harvest", "dio", "rmvpe", "rmvpe_gpu"],
                            value="rmvpe_gpu",
                            interactive=True,
                        )
                        gpus_rmvpe = gr.Textbox(
                            label=i18n(
                                "rmvpe卡号配置：以-分隔输入使用的不同进程卡号,例如0-0-1使用在卡0上跑2个进程并在卡1上跑1个进程"
                            ),
                            value="%s-%s" % (gpus, gpus),
                            interactive=True,
                            visible=_F0GPUVisible,
                        )
                    but2 = gr.Button(i18n("特征提取"), variant="primary")
                    info2 = gr.Textbox(label=i18n("输出信息"), value="", max_lines=8)
                    f0method8.change(
                        fn=_change_f0_method,
                        inputs=[f0method8],
                        outputs=[gpus_rmvpe],
                    )
                    but2.click(
                        _extract_f0_feature,
                        [
                            gpus6,
                            np7,
                            f0method8,
                            if_f0_3,
                            gr_experiment_dir,
                            gr_version,
                            gpus_rmvpe,
                        ],
                        [info2],
                        api_name="train_extract_f0_feature",
                    )
            with gr.Group():
                gr.Markdown(value=i18n("step3: 填写训练设置, 开始训练模型和索引"))
                with gr.Row():
                    gr_save_epoch10 = gr.Slider(
                        minimum=1,
                        maximum=50,
                        step=1,
                        label=i18n("保存频率save_every_epoch"),
                        value=5,
                        interactive=True,
                    )
                    gr_total_epoch11 = gr.Slider(
                        minimum=2,
                        maximum=1000,
                        step=1,
                        label=i18n("总训练轮数total_epoch"),
                        value=20,
                        interactive=True,
                    )
                    gr_batch_size12 = gr.Slider(
                        minimum=1,
                        maximum=40,
                        step=1,
                        label=i18n("每张显卡的batch_size"),
                        value=default_batch_size,
                        interactive=True,
                    )
                    gr_if_save_latest13 = gr.Radio(
                        label=i18n("是否仅保存最新的ckpt文件以节省硬盘空间"),
                        choices=[i18n("是"), i18n("否")],
                        value=i18n("否"),
                        interactive=True,
                    )
                    gr_if_cache_gpu17 = gr.Radio(
                        label=i18n(
                            "是否缓存所有训练集至显存. 10min以下小数据可缓存以加速训练, 大数据缓存会炸显存也加不了多少速"
                        ),
                        choices=[i18n("是"), i18n("否")],
                        value=i18n("否"),
                        interactive=True,
                    )
                    gr_if_save_every_weights18 = gr.Radio(
                        label=i18n(
                            "是否在每次保存时间点将最终小模型保存至weights文件夹"
                        ),
                        choices=[i18n("是"), i18n("否")],
                        value=i18n("否"),
                        interactive=True,
                    )
                with gr.Row():
                    gr_pretrained_G14 = gr.Textbox(
                        label=i18n("加载预训练底模G路径"),
                        value="assets/pretrained_v2/f0G40k.pth",
                        interactive=True,
                    )
                    gr_pretrained_D15 = gr.Textbox(
                        label=i18n("加载预训练底模D路径"),
                        value="assets/pretrained_v2/f0D40k.pth",
                        interactive=True,
                    )
                    gr_sample_rate.change(
                        _change_sr2,
                        [gr_sample_rate, if_f0_3, gr_version],
                        [gr_pretrained_G14, gr_pretrained_D15],
                    )
                    gr_version.change(
                        _change_version19,
                        [gr_sample_rate, if_f0_3, gr_version],
                        [gr_pretrained_G14, gr_pretrained_D15, gr_sample_rate],
                    )
                    if_f0_3.change(
                        _change_f0,
                        [if_f0_3, gr_sample_rate, gr_version],
                        [f0method8, gpus_rmvpe, gr_pretrained_G14, gr_pretrained_D15],
                    )
                    gpus16 = gr.Textbox(
                        label=i18n(
                            "以-分隔输入使用的卡号, 例如   0-1-2   使用卡0和卡1和卡2"
                        ),
                        value=gpus,
                        interactive=True,
                    )
                    but3 = gr.Button(i18n("训练模型"), variant="primary")
                    but4 = gr.Button(i18n("训练特征索引"), variant="primary")
                    but5 = gr.Button(i18n("一键训练"), variant="primary")
                    info3 = gr.Textbox(label=i18n("输出信息"), value="", max_lines=10)
                    but3.click(
                        _click_train,
                        [
                            gr_experiment_dir,
                            gr_sample_rate,
                            if_f0_3,
                            spk_id5,
                            gr_save_epoch10,
                            gr_total_epoch11,
                            gr_batch_size12,
                            gr_if_save_latest13,
                            gr_pretrained_G14,
                            gr_pretrained_D15,
                            gpus16,
                            gr_if_cache_gpu17,
                            gr_if_save_every_weights18,
                            gr_version,
                        ],
                        info3,
                        api_name="train_start",
                    )
                    but4.click(_train_index, [gr_experiment_dir, gr_version], info3)
                    but5.click(
                        _train1key,
                        [
                            gr_experiment_dir,
                            gr_sample_rate,
                            if_f0_3,
                            trainset_dir4,
                            spk_id5,
                            np7,
                            f0method8,
                            gr_save_epoch10,
                            gr_total_epoch11,
                            gr_batch_size12,
                            gr_if_save_latest13,
                            gr_pretrained_G14,
                            gr_pretrained_D15,
                            gpus16,
                            gr_if_cache_gpu17,
                            gr_if_save_every_weights18,
                            gr_version,
                            gpus_rmvpe,
                        ],
                        info3,
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
