import json
import logging
import os
import pathlib
import platform
import shutil
import threading
import traceback
import warnings
from collections.abc import Generator
from random import shuffle
from subprocess import Popen
from time import sleep

import faiss
import gradio as gr
import numpy as np
import torch
from dotenv import load_dotenv
from sklearn.cluster import MiniBatchKMeans

from configs.config import Config
from i18n.i18n import I18nAuto
from infer.lib.train.process_ckpt import (
    change_info,
    extract_small_model,
    merge,
    show_info,
)
from infer.modules.vc.modules import VC

load_dotenv()

_now_dir = os.getcwd()

logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

_logger = logging.getLogger(__name__)

_tmp = os.path.join(_now_dir, "TEMP")
shutil.rmtree(_tmp, ignore_errors=True)
shutil.rmtree(f"{_now_dir}/runtime/Lib/site-packages/infer_pack", ignore_errors=True)
shutil.rmtree(f"{_now_dir}/runtime/Lib/site-packages/uvr5_pack", ignore_errors=True)
os.makedirs(_tmp, exist_ok=True)
os.makedirs(os.path.join(_now_dir, "logs"), exist_ok=True)
os.makedirs(os.path.join(_now_dir, "assets/weights"), exist_ok=True)
os.environ["TEMP"] = _tmp
warnings.filterwarnings("ignore")
torch.manual_seed(114514)


_config = Config()
_vc = VC(_config)


_i18n = I18nAuto()
_logger.info(_i18n)
# 判断是否有能用来训练和加速推理的N卡
_ngpu = torch.cuda.device_count()
_gpu_infos: list[str] = []
_mem: list[int] = []
_if_gpu_ok = False

if torch.cuda.is_available() or _ngpu != 0:
    for i in range(_ngpu):
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
            # A10#A100#V100#A40#P40#M40#K80#A4500
            _if_gpu_ok = True  # 至少有一张能用的N卡
            _gpu_infos.append(f"{i}\t{gpu_name}")
            _mem.append(
                int(
                    torch.cuda.get_device_properties(i).total_memory
                    / 1024
                    / 1024
                    / 1024
                    + 0.4
                )
            )
if _if_gpu_ok and len(_gpu_infos) > 0:
    _default_batch_size = min(_mem) // 2
else:
    _default_batch_size = 1
gpus = "-".join([i[0] for i in _gpu_infos])

_weight_root = os.getenv("weight_root")
_index_root = os.getenv("index_root", "logs")
_outside_index_root = os.getenv("outside_index_root")

names: list[str] = []
for name in os.listdir(_weight_root):
    if name.endswith(".pth"):
        names.append(name)
index_paths: list[str] = []


def _lookup_indices(index_root: str | None) -> None:
    if index_root is None:
        return
    global index_paths
    for root, _dirs, files in os.walk(index_root, topdown=False):
        for name in files:
            if name.endswith(".index") and "trained" not in name:
                index_paths.append(f"{root}/{name}")


_lookup_indices(_index_root)
_lookup_indices(_outside_index_root)


def _change_choices() -> tuple[dict[str, object], dict[str, object]]:
    names: list[str] = []
    for name in os.listdir(_weight_root):
        if name.endswith(".pth"):
            names.append(name)
    index_paths_local: list[str] = []
    for root, _dirs, files in os.walk(_index_root, topdown=False):
        for name in files:
            if name.endswith(".index") and "trained" not in name:
                index_paths_local.append(f"{root}/{name}")
    return {"choices": sorted(names), "__type__": "update"}, {
        "choices": sorted(index_paths_local),
        "__type__": "update",
    }


def _clean() -> dict[str, object]:
    return {"value": "", "__type__": "update"}


def _export_onnx(model_path: str, exported_path: str) -> None:
    from infer.modules.onnx.export import export_onnx as eo

    eo(model_path, exported_path)


_sr_dict = {
    "32k": 32000,
    "40k": 40000,
    "48k": 48000,
}


def _if_done(done: list[object], p: Popen[bytes]) -> None:
    while 1:
        if p.poll() is None:
            sleep(0.5)
        else:
            break
    done[0] = True


def _if_done_multi(done: list[object], ps: list[Popen[bytes]]) -> None:
    while 1:
        # poll==None means the process has not ended
        # As long as one process has not ended, keep looping
        flag = 1
        for p in ps:
            if p.poll() is None:
                flag = 0
                sleep(0.5)
                break
        if flag == 1:
            break
    done[0] = True


def _preprocess_dataset(
    trainset_dir: str,
    exp_dir: str,
    sr_str: str,
    n_p: int,
) -> Generator[str]:
    sr = _sr_dict[sr_str]
    os.makedirs(f"{_now_dir}/logs/{exp_dir}", exist_ok=True)
    f = open(f"{_now_dir}/logs/{exp_dir}/preprocess.log", "w")
    f.close()
    cmd = f'"{_config.python_cmd}" infer/modules/train/preprocess.py "{trainset_dir}" {sr} {n_p} "{_now_dir}/logs/{exp_dir}" {_config.noparallel} {_config.preprocess_per:.1f}'
    _logger.info("Execute: " + cmd)

    p = Popen(cmd, shell=True)
    # 煞笔gr, popen read都非得全跑完了再一次性读取, 不用gr就正常读一句输出一句;只能额外弄出一个文本流定时读
    done = [False]
    threading.Thread(
        target=_if_done,
        args=(
            done,
            p,
        ),
    ).start()
    while 1:
        with open(f"{_now_dir}/logs/{exp_dir}/preprocess.log") as f:
            yield (f.read())
        sleep(1)
        if done[0]:
            break
    with open(f"{_now_dir}/logs/{exp_dir}/preprocess.log") as f:
        log = f.read()
    _logger.info(log)
    yield log


def _extract_f0_feature(
    gpu_str: str,
    n_p: int,
    f0method: str,
    if_f0: bool,
    exp_dir: str,
    gpus_rmvpe_str: str,
):
    gpus = gpu_str.split("-")
    os.makedirs(f"{_now_dir}/logs/{exp_dir}", exist_ok=True)
    f = open(f"{_now_dir}/logs/{exp_dir}/extract_f0_feature.log", "w")
    f.close()
    if if_f0:
        if f0method != "rmvpe_gpu":
            cmd = f'"{_config.python_cmd}" infer/modules/train/extract/extract_f0_print.py "{_now_dir}/logs/{exp_dir}" {n_p} {f0method}'
            _logger.info("Execute: " + cmd)
            p = Popen(cmd, shell=True, cwd=_now_dir)
            # 煞笔gr, popen read都非得全跑完了再一次性读取, 不用gr就正常读一句输出一句;只能额外弄出一个文本流定时读
            done = [False]
            threading.Thread(
                target=_if_done,
                args=(
                    done,
                    p,
                ),
            ).start()
        else:
            if gpus_rmvpe_str != "-":
                gpus_rmvpe = gpus_rmvpe_str.split("-")
                leng = len(gpus_rmvpe)
                ps = []
                for idx, n_g in enumerate(gpus_rmvpe):
                    cmd = f'"{_config.python_cmd}" infer/modules/train/extract/extract_f0_rmvpe.py {leng} {idx} {n_g} "{_now_dir}/logs/{exp_dir}" {_config.is_half} '
                    _logger.info("Execute: " + cmd)
                    p = Popen(cmd, shell=True, cwd=_now_dir)
                    ps.append(p)
                # 煞笔gr, popen read都非得全跑完了再一次性读取, 不用gr就正常读一句输出一句;只能额外弄出一个文本流定时读
                done = [False]
                threading.Thread(
                    target=_if_done_multi,  #
                    args=(
                        done,
                        ps,
                    ),
                ).start()
            else:
                cmd = (
                    _config.python_cmd
                    + f' infer/modules/train/extract/extract_f0_rmvpe_dml.py "{_now_dir}/logs/{exp_dir}" '
                )
                _logger.info("Execute: " + cmd)
                p = Popen(cmd, shell=True, cwd=_now_dir)
                p.wait()
                done = [True]
        while 1:
            with open(f"{_now_dir}/logs/{exp_dir}/extract_f0_feature.log") as f:
                yield (f.read())
            sleep(1)
            if done[0]:
                break
        with open(f"{_now_dir}/logs/{exp_dir}/extract_f0_feature.log") as f:
            log = f.read()
        _logger.info(log)
        yield log
    # 对不同part分别开多进程
    leng = len(gpus)
    ps: list[Popen[bytes]] = []
    for idx, n_g in enumerate(gpus):
        cmd = f'"{_config.python_cmd}" infer/modules/train/extract_feature_print.py {_config.device} {leng} {idx} {n_g} "{_now_dir}/logs/{exp_dir}" {_config.is_half}'
        _logger.info("Execute: " + cmd)
        p = Popen(cmd, shell=True, cwd=_now_dir)
        ps.append(p)
    # 煞笔gr, popen read都非得全跑完了再一次性读取, 不用gr就正常读一句输出一句;只能额外弄出一个文本流定时读
    done = [False]
    threading.Thread(
        target=_if_done_multi,
        args=(
            done,
            ps,
        ),
    ).start()
    while 1:
        with open(f"{_now_dir}/logs/{exp_dir}/extract_f0_feature.log") as f:
            yield (f.read())
        sleep(1)
        if done[0]:
            break
    with open(f"{_now_dir}/logs/{exp_dir}/extract_f0_feature.log") as f:
        log = f.read()
    _logger.info(log)
    yield log


def _get_pretrained_models(path_str: str, f0_str: str, sr2: str):
    if_pretrained_generator_exist = os.access(
        f"assets/pretrained{path_str}/{f0_str}G{sr2}.pth", os.F_OK
    )
    if_pretrained_discriminator_exist = os.access(
        f"assets/pretrained{path_str}/{f0_str}D{sr2}.pth", os.F_OK
    )
    if not if_pretrained_generator_exist:
        _logger.warning(
            "assets/pretrained%s/%sG%s.pth not exist, will not use pretrained model",
            path_str,
            f0_str,
            sr2,
        )
    if not if_pretrained_discriminator_exist:
        _logger.warning(
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


def _change_sr2(sr2: str, if_f0_3: bool):
    path_str = "" if False else "_v2"
    f0_str = "f0" if if_f0_3 else ""
    return _get_pretrained_models(path_str, f0_str, sr2)


def _change_f0(if_f0_3: bool, sr2: str):
    path_str = "" if False else "_v2"
    return (
        {"visible": if_f0_3, "__type__": "update"},
        {"visible": if_f0_3, "__type__": "update"},
        *_get_pretrained_models(path_str, "f0" if if_f0_3 else "", sr2),
    )


def _click_train(
    exp_dir1: str,
    sr2: str,
    if_f0_3: bool,
    spk_id5: int,
    save_epoch10: int,
    total_epoch11: int,
    batch_size12: int,
    if_save_latest13: bool,
    pretrained_G14: str,
    pretrained_D15: str,
    gpus16: str,
    if_cache_gpu17: bool,
    if_save_every_weights18: bool,
):
    # 生成filelist
    exp_dir = f"{_now_dir}/logs/{exp_dir1}"
    os.makedirs(exp_dir, exist_ok=True)
    gt_wavs_dir = f"{exp_dir}/0_gt_wavs"
    feature_dir = f"{exp_dir}/3_feature256" if False else f"{exp_dir}/3_feature768"
    f0_dir = ""
    f0nsf_dir = ""
    if if_f0_3:
        f0_dir = f"{exp_dir}/2a_f0"
        f0nsf_dir = f"{exp_dir}/2b-f0nsf"
        names = (
            set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)])
            & set([name.split(".")[0] for name in os.listdir(feature_dir)])
            & set([name.split(".")[0] for name in os.listdir(f0_dir)])
            & set([name.split(".")[0] for name in os.listdir(f0nsf_dir)])
        )
    else:
        names = set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)]) & set(
            [name.split(".")[0] for name in os.listdir(feature_dir)]
        )
    opt: list[str] = []
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
    fea_dim = 256 if False else 768
    if if_f0_3:
        for _ in range(2):
            opt.append(
                f"{_now_dir}/logs/mute/0_gt_wavs/mute{sr2}.wav|{_now_dir}/logs/mute/3_feature{fea_dim}/mute.npy|{_now_dir}/logs/mute/2a_f0/mute.wav.npy|{_now_dir}/logs/mute/2b-f0nsf/mute.wav.npy|{spk_id5}"
            )
    else:
        for _ in range(2):
            opt.append(
                f"{_now_dir}/logs/mute/0_gt_wavs/mute{sr2}.wav|{_now_dir}/logs/mute/3_feature{fea_dim}/mute.npy|{spk_id5}"
            )
    shuffle(opt)
    with open(f"{exp_dir}/filelist.txt", "w") as f:
        f.write("\n".join(opt))
    _logger.debug("Write filelist done")
    # 生成config#无需生成config

    _logger.info("Use gpus: %s", str(gpus16))
    if pretrained_G14 == "":
        _logger.info("No pretrained Generator")
    if pretrained_D15 == "":
        _logger.info("No pretrained Discriminator")
    config_path = f"v2/{sr2}.json"
    config_save_path = os.path.join(exp_dir, "config.json")
    if not pathlib.Path(config_save_path).exists():
        with open(config_save_path, "w", encoding="utf-8") as f:
            json.dump(
                _config.json_config[config_path],
                f,
                ensure_ascii=False,
                indent=4,
                sort_keys=True,
            )
            f.write("\n")
    if gpus16:
        cmd = '"{}" infer/modules/train/train.py -e "{}" -sr {} -f0 {} -bs {} -g {} -te {} -se {} {} {} -l {} -c {} -sw {}'.format(
            _config.python_cmd,
            exp_dir1,
            sr2,
            1 if if_f0_3 else 0,
            batch_size12,
            gpus16,
            total_epoch11,
            save_epoch10,
            f"-pg {pretrained_G14}" if pretrained_G14 != "" else "",
            f"-pd {pretrained_D15}" if pretrained_D15 != "" else "",
            1 if if_save_latest13 == _i18n("是") else 0,
            1 if if_cache_gpu17 == _i18n("是") else 0,
            1 if if_save_every_weights18 == _i18n("是") else 0,
        )
    else:
        cmd = '"{}" infer/modules/train/train.py -e "{}" -sr {} -f0 {} -bs {} -te {} -se {} {} {} -l {} -c {} -sw {}'.format(
            _config.python_cmd,
            exp_dir1,
            sr2,
            1 if if_f0_3 else 0,
            batch_size12,
            total_epoch11,
            save_epoch10,
            f"-pg {pretrained_G14}" if pretrained_G14 != "" else "",
            f"-pd {pretrained_D15}" if pretrained_D15 != "" else "",
            1 if if_save_latest13 == _i18n("是") else 0,
            1 if if_cache_gpu17 == _i18n("是") else 0,
            1 if if_save_every_weights18 == _i18n("是") else 0,
        )
    _logger.info("Execute: " + cmd)
    p = Popen(cmd, shell=True, cwd=_now_dir)
    p.wait()
    return "训练结束, 您可查看控制台训练日志或实验文件夹下的train.log"


def _train_index(exp_dir1: str) -> Generator[str]:
    _logger.info("Start training index for %s", exp_dir1)

    exp_dir = f"logs/{exp_dir1}"
    os.makedirs(exp_dir, exist_ok=True)
    feature_dir = f"{exp_dir}/3_feature768"

    if not os.path.exists(feature_dir):
        return "请先进行特征提取!"

    listdir_res = list(os.listdir(feature_dir))
    if len(listdir_res) == 0:
        return "请先进行特征提取！"

    # Load and concatenate all feature files
    infos: list[str] = []
    npys: list[np.ndarray] = []
    for name in sorted(listdir_res):
        npys.append(np.load(f"{feature_dir}/{name}"))

    big_npy = np.concatenate(npys, 0)

    # Shuffle and optionally compress features
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
                    batch_size=256 * _config.n_cpu,
                    compute_labels=False,
                    init="random",
                )
                .fit(big_npy)
                .cluster_centers_
            )
        except Exception:
            infos.append(traceback.format_exc())
            _logger.info(infos[-1])
            yield "\n".join(infos)

    np.save(f"{exp_dir}/total_fea.npy", big_npy)

    # Build FAISS index
    n_ivf = min(int(16 * np.sqrt(big_npy.shape[0])), big_npy.shape[0] // 39)
    infos.append(f"{big_npy.shape},{n_ivf}")
    yield "\n".join(infos)

    index = faiss.index_factory(768, f"IVF{n_ivf},Flat")
    index_ivf = faiss.extract_index_ivf(index)
    index_ivf.nprobe = 1
    index.train(big_npy)

    # Save training index
    trained_index_path = (
        f"{exp_dir}/trained_IVF{n_ivf}_Flat_nprobe_{index_ivf.nprobe}_{exp_dir1}.index"
    )
    faiss.write_index(index, trained_index_path)

    # Add vectors and save final index
    infos.append("adding")
    yield "\n".join(infos)

    for i in range(0, big_npy.shape[0], 8192):
        index.add(big_npy[i : i + 8192])

    added_index_path = (
        f"{exp_dir}/added_IVF{n_ivf}_Flat_nprobe_{index_ivf.nprobe}_{exp_dir1}.index"
    )
    faiss.write_index(index, added_index_path)
    infos.append(f"成功构建索引 {os.path.basename(added_index_path)}")

    # Link to external index root if configured
    if _outside_index_root:
        try:
            link_fn = os.link if platform.system() == "Windows" else os.symlink
            external_path = f"{_outside_index_root}/{exp_dir1}_IVF{n_ivf}_Flat_nprobe_{index_ivf.nprobe}_{exp_dir1}.index"
            link_fn(added_index_path, external_path)
            infos.append(f"链接索引到外部-{_outside_index_root}")
        except Exception:
            infos.append(f"链接索引到外部-{_outside_index_root}失败")

    yield "\n".join(infos)


def _train1key(
    exp_dir1: str,
    sr2: str,
    if_f0_3: bool,
    trainset_dir4: str,
    spk_id5: int,
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
    gpus_rmvpe: str,
):
    infos = []

    def get_info_str(strr):
        infos.append(strr)
        return "\n".join(infos)

    # step1:处理数据
    yield get_info_str(_i18n("step1:正在处理数据"))
    [get_info_str(_) for _ in _preprocess_dataset(trainset_dir4, exp_dir1, sr2, np7)]

    # step2a:提取音高
    yield get_info_str(_i18n("step2:正在提取音高&正在提取特征"))
    [
        get_info_str(_)
        for _ in _extract_f0_feature(
            gpus16, np7, f0method8, if_f0_3, exp_dir1, gpus_rmvpe
        )
    ]

    # step3a:训练模型
    yield get_info_str(_i18n("step3a:正在训练模型"))
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
    )
    yield get_info_str(
        _i18n("训练结束, 您可查看控制台训练日志或实验文件夹下的train.log")
    )

    # step3b:训练索引
    [get_info_str(_) for _ in _train_index(exp_dir1)]
    yield get_info_str(_i18n("全流程结束！"))


def _change_info_(ckpt_path: str):
    if not os.path.exists(ckpt_path.replace(os.path.basename(ckpt_path), "train.log")):
        return {"__type__": "update"}, {"__type__": "update"}, {"__type__": "update"}
    try:
        with open(ckpt_path.replace(os.path.basename(ckpt_path), "train.log")) as f:
            info = eval(f.read().strip("\n").split("\n")[0].split("\t")[-1])
            sr, f0 = info["sample_rate"], info["if_f0"]
            return sr, str(f0)
    except Exception:
        traceback.print_exc()
        return {"__type__": "update"}, {"__type__": "update"}, {"__type__": "update"}


def _change_f0_method(f0method8: str):
    visible = f0method8 == "rmvpe_gpu"
    return {"visible": visible, "__type__": "update"}


with gr.Blocks(title="RVC WebUI") as app:
    gr.Markdown("## RVC WebUI")
    gr.Markdown(
        value=_i18n(
            "本软件以MIT协议开源, 作者不对软件具备任何控制力, 使用软件者、传播软件导出的声音者自负全责. <br>如不认可该条款, 则不能使用或引用软件包内任何代码和文件. 详见根目录<b>LICENSE</b>."
        )
    )
    with gr.Tabs():
        with gr.TabItem(_i18n("模型推理")):
            with gr.Row():
                sid0 = gr.Dropdown(label=_i18n("推理音色"), choices=sorted(names))
                with gr.Column():
                    refresh_button = gr.Button(
                        _i18n("刷新音色列表和索引路径"), variant="primary"
                    )
                    clean_button = gr.Button(_i18n("卸载音色省显存"), variant="primary")
                spk_item = gr.Slider(
                    minimum=0,
                    maximum=2333,
                    step=1,
                    label=_i18n("请选择说话人id"),
                    value=0,
                    visible=False,
                    interactive=True,
                )
                clean_button.click(
                    fn=_clean, inputs=[], outputs=[sid0], api_name="infer_clean"
                )
            with gr.TabItem(_i18n("单次推理")):
                with gr.Group():
                    with gr.Row():
                        with gr.Column():
                            vc_transform0 = gr.Number(
                                label=_i18n("变调(整数, 半音数量, 升八度12降八度-12)"),
                                value=0,
                            )
                            input_audio0 = gr.Textbox(
                                label=_i18n(
                                    "输入待处理音频文件路径(默认是正确格式示例)"
                                ),
                                placeholder="C:\\Users\\Desktop\\audio_example.wav",
                            )
                            file_index1 = gr.Textbox(
                                label=_i18n(
                                    "特征检索库文件路径,为空则使用下拉的选择结果"
                                ),
                                placeholder="C:\\Users\\Desktop\\model_example.index",
                                interactive=True,
                            )
                            file_index2 = gr.Dropdown(
                                label=_i18n("自动检测index路径,下拉式选择(dropdown)"),
                                choices=sorted(index_paths),
                                interactive=True,
                            )
                            f0method0 = gr.Radio(
                                label=_i18n(
                                    "选择音高提取算法,输入歌声可用pm提速,harvest低音好但巨慢无比,crepe效果好但吃GPU,rmvpe效果最好且微吃GPU"
                                ),
                                choices=(["pm", "harvest", "crepe", "rmvpe"]),
                                value="rmvpe",
                                interactive=True,
                            )

                        with gr.Column():
                            resample_sr0 = gr.Slider(
                                minimum=0,
                                maximum=48000,
                                label=_i18n(
                                    "后处理重采样至最终采样率，0为不进行重采样"
                                ),
                                value=0,
                                step=1,
                                interactive=True,
                            )
                            rms_mix_rate0 = gr.Slider(
                                minimum=0,
                                maximum=1,
                                label=_i18n(
                                    "输入源音量包络替换输出音量包络融合比例，越靠近1越使用输出包络"
                                ),
                                value=0.25,
                                interactive=True,
                            )
                            protect0 = gr.Slider(
                                minimum=0,
                                maximum=0.5,
                                label=_i18n(
                                    "保护清辅音和呼吸声，防止电音撕裂等artifact，拉满0.5不开启，调低加大保护力度但可能降低索引效果"
                                ),
                                value=0.33,
                                step=0.01,
                                interactive=True,
                            )
                            filter_radius0 = gr.Slider(
                                minimum=0,
                                maximum=7,
                                label=_i18n(
                                    ">=3则使用对harvest音高识别的结果使用中值滤波，数值为滤波半径，使用可以削弱哑音"
                                ),
                                value=3,
                                step=1,
                                interactive=True,
                            )
                            index_rate1 = gr.Slider(
                                minimum=0,
                                maximum=1,
                                label=_i18n("检索特征占比"),
                                value=0.75,
                                interactive=True,
                            )
                            f0_file = gr.File(
                                label=_i18n(
                                    "F0曲线文件, 可选, 一行一个音高, 代替默认F0及升降调"
                                ),
                                visible=False,
                            )

                            refresh_button.click(
                                fn=_change_choices,
                                inputs=[],
                                outputs=[sid0, file_index2],
                                api_name="infer_refresh",
                            )
                with gr.Group():
                    with gr.Column():
                        but0 = gr.Button(_i18n("转换"), variant="primary")
                        with gr.Row():
                            vc_output1 = gr.Textbox(label=_i18n("输出信息"))
                            vc_output2 = gr.Audio(
                                label=_i18n("输出音频(右下角三个点,点了可以下载)")
                            )

                        but0.click(
                            _vc.vc_single,
                            [
                                spk_item,
                                input_audio0,
                                vc_transform0,
                                f0_file,
                                f0method0,
                                file_index1,
                                file_index2,
                                index_rate1,
                                filter_radius0,
                                resample_sr0,
                                rms_mix_rate0,
                                protect0,
                            ],
                            [vc_output1, vc_output2],
                            api_name="infer_convert",
                        )
            with gr.TabItem(_i18n("批量推理")):
                gr.Markdown(
                    value=_i18n(
                        "批量转换, 输入待转换音频文件夹, 或上传多个音频文件, 在指定文件夹(默认opt)下输出转换的音频. "
                    )
                )
                with gr.Row():
                    with gr.Column():
                        vc_transform1 = gr.Number(
                            label=_i18n("变调(整数, 半音数量, 升八度12降八度-12)"),
                            value=0,
                        )
                        opt_input = gr.Textbox(
                            label=_i18n("指定输出文件夹"), value="opt"
                        )
                        file_index3 = gr.Textbox(
                            label=_i18n("特征检索库文件路径,为空则使用下拉的选择结果"),
                            value="",
                            interactive=True,
                        )
                        file_index4 = gr.Dropdown(
                            label=_i18n("自动检测index路径,下拉式选择(dropdown)"),
                            choices=sorted(index_paths),
                            interactive=True,
                        )
                        f0method1 = gr.Radio(
                            label=_i18n(
                                "选择音高提取算法,输入歌声可用pm提速,harvest低音好但巨慢无比,crepe效果好但吃GPU,rmvpe效果最好且微吃GPU"
                            ),
                            choices=(["pm", "harvest", "crepe", "rmvpe"]),
                            value="rmvpe",
                            interactive=True,
                        )
                        format1 = gr.Radio(
                            label=_i18n("导出文件格式"),
                            choices=["wav", "flac", "mp3", "m4a"],
                            value="wav",
                            interactive=True,
                        )

                        refresh_button.click(
                            fn=lambda: _change_choices()[1],
                            inputs=[],
                            outputs=file_index4,
                            api_name="infer_refresh_batch",
                        )

                    with gr.Column():
                        resample_sr1 = gr.Slider(
                            minimum=0,
                            maximum=48000,
                            label=_i18n("后处理重采样至最终采样率，0为不进行重采样"),
                            value=0,
                            step=1,
                            interactive=True,
                        )
                        rms_mix_rate1 = gr.Slider(
                            minimum=0,
                            maximum=1,
                            label=_i18n(
                                "输入源音量包络替换输出音量包络融合比例，越靠近1越使用输出包络"
                            ),
                            value=1,
                            interactive=True,
                        )
                        protect1 = gr.Slider(
                            minimum=0,
                            maximum=0.5,
                            label=_i18n(
                                "保护清辅音和呼吸声，防止电音撕裂等artifact，拉满0.5不开启，调低加大保护力度但可能降低索引效果"
                            ),
                            value=0.33,
                            step=0.01,
                            interactive=True,
                        )
                        filter_radius1 = gr.Slider(
                            minimum=0,
                            maximum=7,
                            label=_i18n(
                                ">=3则使用对harvest音高识别的结果使用中值滤波，数值为滤波半径，使用可以削弱哑音"
                            ),
                            value=3,
                            step=1,
                            interactive=True,
                        )
                        index_rate2 = gr.Slider(
                            minimum=0,
                            maximum=1,
                            label=_i18n("检索特征占比"),
                            value=1,
                            interactive=True,
                        )
                with gr.Row():
                    dir_input = gr.Textbox(
                        label=_i18n(
                            "输入待处理音频文件夹路径(去文件管理器地址栏拷就行了)"
                        ),
                        placeholder="C:\\Users\\Desktop\\input_vocal_dir",
                    )
                    inputs = gr.File(
                        file_count="multiple",
                        label=_i18n("也可批量输入音频文件, 二选一, 优先读文件夹"),
                    )

                with gr.Row():
                    but1 = gr.Button(_i18n("转换"), variant="primary")
                    vc_output3 = gr.Textbox(label=_i18n("输出信息"))

                    but1.click(
                        _vc.vc_multi,
                        [
                            spk_item,
                            dir_input,
                            opt_input,
                            inputs,
                            vc_transform1,
                            f0method1,
                            file_index3,
                            file_index4,
                            index_rate2,
                            filter_radius1,
                            resample_sr1,
                            rms_mix_rate1,
                            protect1,
                            format1,
                        ],
                        [vc_output3],
                        api_name="infer_convert_batch",
                    )
                sid0.change(
                    fn=_vc.get_vc,
                    inputs=[sid0, protect0, protect1],
                    outputs=[spk_item, protect0, protect1, file_index2, file_index4],
                    api_name="infer_change_voice",
                )
        with gr.TabItem(_i18n("训练")):
            gr.Markdown(
                value=_i18n(
                    "step1: 填写实验配置. 实验数据放在logs下, 每个实验一个文件夹, 需手工输入实验名路径, 内含实验配置, 日志, 训练得到的模型文件. "
                )
            )
            with gr.Row():
                exp_dir1 = gr.Textbox(label=_i18n("输入实验名"), value="mi-test")
                sr2 = gr.Radio(
                    label=_i18n("目标采样率"),
                    choices=["40k", "48k"],
                    value="40k",
                    interactive=True,
                )
                if_f0_3 = gr.Radio(
                    label=_i18n("模型是否带音高指导(唱歌一定要, 语音可以不要)"),
                    choices=[True, False],
                    value=True,
                    interactive=True,
                )
                np7 = gr.Slider(
                    minimum=0,
                    maximum=_config.n_cpu,
                    step=1,
                    label=_i18n("提取音高和处理数据使用的CPU进程数"),
                    value=int(np.ceil(_config.n_cpu / 1.5)),
                    interactive=True,
                )
            with gr.Group():  # 暂时单人的, 后面支持最多4人的#数据处理
                gr.Markdown(
                    value=_i18n(
                        "step2a: 自动遍历训练文件夹下所有可解码成音频的文件并进行切片归一化, 在实验目录下生成2个wav文件夹; 暂时只支持单人训练. "
                    )
                )
                with gr.Row():
                    trainset_dir4 = gr.Textbox(
                        label=_i18n("输入训练文件夹路径"),
                        value=_i18n("E:\\语音音频+标注\\米津玄师\\src"),
                    )
                    spk_id5 = gr.Slider(
                        minimum=0,
                        maximum=4,
                        step=1,
                        label=_i18n("请指定说话人id"),
                        value=0,
                        interactive=True,
                    )
                    but1 = gr.Button(_i18n("处理数据"), variant="primary")
                    info1 = gr.Textbox(label=_i18n("输出信息"), value="")
                    but1.click(
                        _preprocess_dataset,
                        [trainset_dir4, exp_dir1, sr2, np7],
                        [info1],
                        api_name="train_preprocess",
                    )
            with gr.Group():
                gr.Markdown(
                    value=_i18n(
                        "step2b: 使用CPU提取音高(如果模型带音高), 使用GPU提取特征(选择卡号)"
                    )
                )
                with gr.Row():
                    with gr.Column():
                        gpus6 = gr.Textbox(
                            label=_i18n(
                                "以-分隔输入使用的卡号, 例如   0-1-2   使用卡0和卡1和卡2"
                            ),
                            value=gpus,
                            interactive=True,
                        )
                    with gr.Column():
                        f0method8 = gr.Radio(
                            label=_i18n(
                                "选择音高提取算法:输入歌声可用pm提速,高质量语音但CPU差可用dio提速,harvest质量更好但慢,rmvpe效果最好且微吃CPU/GPU"
                            ),
                            choices=["pm", "harvest", "dio", "rmvpe", "rmvpe_gpu"],
                            value="rmvpe_gpu",
                            interactive=True,
                        )
                        gpus_rmvpe = gr.Textbox(
                            label=_i18n(
                                "rmvpe卡号配置：以-分隔输入使用的不同进程卡号,例如0-0-1使用在卡0上跑2个进程并在卡1上跑1个进程"
                            ),
                            value=f"{gpus}-{gpus}",
                            interactive=True,
                        )
                    but2 = gr.Button(_i18n("特征提取"), variant="primary")
                    info2 = gr.Textbox(label=_i18n("输出信息"), value="", max_lines=8)
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
                            exp_dir1,
                            gpus_rmvpe,
                        ],
                        [info2],
                        api_name="train_extract_f0_feature",
                    )
            with gr.Group():
                gr.Markdown(value=_i18n("step3: 填写训练设置, 开始训练模型和索引"))
                with gr.Row():
                    save_epoch10 = gr.Slider(
                        minimum=1,
                        maximum=50,
                        step=1,
                        label=_i18n("保存频率save_every_epoch"),
                        value=5,
                        interactive=True,
                    )
                    total_epoch11 = gr.Slider(
                        minimum=2,
                        maximum=1000,
                        step=1,
                        label=_i18n("总训练轮数total_epoch"),
                        value=20,
                        interactive=True,
                    )
                    batch_size12 = gr.Slider(
                        minimum=1,
                        maximum=40,
                        step=1,
                        label=_i18n("每张显卡的batch_size"),
                        value=_default_batch_size,
                        interactive=True,
                    )
                    if_save_latest13 = gr.Radio(
                        label=_i18n("是否仅保存最新的ckpt文件以节省硬盘空间"),
                        choices=[_i18n("是"), _i18n("否")],
                        value=_i18n("否"),
                        interactive=True,
                    )
                    if_cache_gpu17 = gr.Radio(
                        label=_i18n(
                            "是否缓存所有训练集至显存. 10min以下小数据可缓存以加速训练, 大数据缓存会炸显存也加不了多少速"
                        ),
                        choices=[_i18n("是"), _i18n("否")],
                        value=_i18n("否"),
                        interactive=True,
                    )
                    if_save_every_weights18 = gr.Radio(
                        label=_i18n(
                            "是否在每次保存时间点将最终小模型保存至weights文件夹"
                        ),
                        choices=[_i18n("是"), _i18n("否")],
                        value=_i18n("否"),
                        interactive=True,
                    )
                with gr.Row():
                    pretrained_G14 = gr.Textbox(
                        label=_i18n("加载预训练底模G路径"),
                        value="assets/pretrained_v2/f0G40k.pth",
                        interactive=True,
                    )
                    pretrained_D15 = gr.Textbox(
                        label=_i18n("加载预训练底模D路径"),
                        value="assets/pretrained_v2/f0D40k.pth",
                        interactive=True,
                    )
                    sr2.change(
                        _change_sr2,
                        [sr2, if_f0_3],
                        [pretrained_G14, pretrained_D15],
                    )
                    if_f0_3.change(
                        _change_f0,
                        [if_f0_3, sr2],
                        [f0method8, gpus_rmvpe, pretrained_G14, pretrained_D15],
                    )
                    gpus16 = gr.Textbox(
                        label=_i18n(
                            "以-分隔输入使用的卡号, 例如   0-1-2   使用卡0和卡1和卡2"
                        ),
                        value=gpus,
                        interactive=True,
                    )
                    but3 = gr.Button(_i18n("训练模型"), variant="primary")
                    but4 = gr.Button(_i18n("训练特征索引"), variant="primary")
                    but5 = gr.Button(_i18n("一键训练"), variant="primary")
                    info3 = gr.Textbox(label=_i18n("输出信息"), value="", max_lines=10)
                    but3.click(
                        _click_train,
                        [
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
                        ],
                        info3,
                        api_name="train_start",
                    )
                    but4.click(_train_index, [exp_dir1], info3)
                    but5.click(
                        _train1key,
                        [
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
                            gpus_rmvpe,
                        ],
                        info3,
                        api_name="train_start_all",
                    )

        with gr.TabItem(_i18n("ckpt处理")):
            with gr.Group():
                gr.Markdown(value=_i18n("模型融合, 可用于测试音色融合"))
                with gr.Row():
                    ckpt_a = gr.Textbox(
                        label=_i18n("A模型路径"), value="", interactive=True
                    )
                    ckpt_b = gr.Textbox(
                        label=_i18n("B模型路径"), value="", interactive=True
                    )
                    alpha_a = gr.Slider(
                        minimum=0,
                        maximum=1,
                        label=_i18n("A模型权重"),
                        value=0.5,
                        interactive=True,
                    )
                with gr.Row():
                    sr_ = gr.Radio(
                        label=_i18n("目标采样率"),
                        choices=["40k", "48k"],
                        value="40k",
                        interactive=True,
                    )
                    if_f0_ = gr.Radio(
                        label=_i18n("模型是否带音高指导"),
                        choices=[_i18n("是"), _i18n("否")],
                        value=_i18n("是"),
                        interactive=True,
                    )
                    info__ = gr.Textbox(
                        label=_i18n("要置入的模型信息"),
                        value="",
                        max_lines=8,
                        interactive=True,
                    )
                    name_to_save0 = gr.Textbox(
                        label=_i18n("保存的模型名不带后缀"),
                        value="",
                        max_lines=1,
                        interactive=True,
                    )
                with gr.Row():
                    but6 = gr.Button(_i18n("融合"), variant="primary")
                    info4 = gr.Textbox(label=_i18n("输出信息"), value="", max_lines=8)
                but6.click(
                    merge,
                    [
                        ckpt_a,
                        ckpt_b,
                        alpha_a,
                        sr_,
                        if_f0_,
                        info__,
                        name_to_save0,
                    ],
                    info4,
                    api_name="ckpt_merge",
                )
            with gr.Group():
                gr.Markdown(
                    value=_i18n("修改模型信息(仅支持weights文件夹下提取的小模型文件)")
                )
                with gr.Row():
                    ckpt_path0 = gr.Textbox(
                        label=_i18n("模型路径"), value="", interactive=True
                    )
                    info_ = gr.Textbox(
                        label=_i18n("要改的模型信息"),
                        value="",
                        max_lines=8,
                        interactive=True,
                    )
                    name_to_save1 = gr.Textbox(
                        label=_i18n("保存的文件名, 默认空为和源文件同名"),
                        value="",
                        max_lines=8,
                        interactive=True,
                    )
                with gr.Row():
                    but7 = gr.Button(_i18n("修改"), variant="primary")
                    info5 = gr.Textbox(label=_i18n("输出信息"), value="", max_lines=8)
                but7.click(
                    change_info,
                    [ckpt_path0, info_, name_to_save1],
                    info5,
                    api_name="ckpt_modify",
                )
            with gr.Group():
                gr.Markdown(
                    value=_i18n("查看模型信息(仅支持weights文件夹下提取的小模型文件)")
                )
                with gr.Row():
                    ckpt_path1 = gr.Textbox(
                        label=_i18n("模型路径"), value="", interactive=True
                    )
                    but8 = gr.Button(_i18n("查看"), variant="primary")
                    info6 = gr.Textbox(label=_i18n("输出信息"), value="", max_lines=8)
                but8.click(show_info, [ckpt_path1], info6, api_name="ckpt_show")
            with gr.Group():
                gr.Markdown(
                    value=_i18n(
                        "模型提取(输入logs文件夹下大文件模型路径),适用于训一半不想训了模型没有自动提取保存小文件模型,或者想测试中间模型的情况"
                    )
                )
                with gr.Row():
                    ckpt_path2 = gr.Textbox(
                        label=_i18n("模型路径"),
                        value="E:\\codes\\py39\\logs\\mi-test_f0_48k\\G_23333.pth",
                        interactive=True,
                    )
                    save_name = gr.Textbox(
                        label=_i18n("保存名"), value="", interactive=True
                    )
                    sr__ = gr.Radio(
                        label=_i18n("目标采样率"),
                        choices=["32k", "40k", "48k"],
                        value="40k",
                        interactive=True,
                    )
                    if_f0__ = gr.Radio(
                        label=_i18n("模型是否带音高指导,1是0否"),
                        choices=["1", "0"],
                        value="1",
                        interactive=True,
                    )
                    info___ = gr.Textbox(
                        label=_i18n("要置入的模型信息"),
                        value="",
                        max_lines=8,
                        interactive=True,
                    )
                    but9 = gr.Button(_i18n("提取"), variant="primary")
                    info7 = gr.Textbox(label=_i18n("输出信息"), value="", max_lines=8)
                    ckpt_path2.change(_change_info_, [ckpt_path2], [sr__, if_f0__])
                but9.click(
                    extract_small_model,
                    [ckpt_path2, save_name, sr__, if_f0__, info___],
                    info7,
                    api_name="ckpt_extract",
                )

        with gr.TabItem(_i18n("Onnx导出")):
            with gr.Row():
                ckpt_dir = gr.Textbox(
                    label=_i18n("RVC模型路径"), value="", interactive=True
                )
            with gr.Row():
                onnx_dir = gr.Textbox(
                    label=_i18n("Onnx输出路径"), value="", interactive=True
                )
            with gr.Row():
                infoOnnx = gr.Label(label="info")
            with gr.Row():
                butOnnx = gr.Button(_i18n("导出Onnx模型"), variant="primary")
            butOnnx.click(
                _export_onnx, [ckpt_dir, onnx_dir], infoOnnx, api_name="export_onnx"
            )

        tab_faq = _i18n("常见问题解答")
        with gr.TabItem(tab_faq):
            try:
                if tab_faq == "常见问题解答":
                    with open("docs/cn/faq.md", encoding="utf8") as f:
                        info = f.read()
                else:
                    with open("docs/en/faq_en.md", encoding="utf8") as f:
                        info = f.read()
                gr.Markdown(value=info)
            except Exception:
                gr.Markdown(traceback.format_exc())

    if _config.iscolab:
        app.queue(max_size=1022).launch(share=True)
    else:
        app.queue(max_size=1022).launch(
            server_name="0.0.0.0",
            inbrowser=not _config.noautoopen,
            server_port=_config.listen_port,
            quiet=True,
        )
