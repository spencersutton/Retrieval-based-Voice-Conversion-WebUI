import json
import logging
import multiprocessing
import os
import platform
import shutil
import sys
import traceback
import warnings
from collections.abc import Generator
from multiprocessing import Process
from pathlib import Path
from random import shuffle
from subprocess import Popen
from time import sleep
from typing import Any

import faiss
import gradio as gr
import numpy as np
import torch
from dotenv import load_dotenv
from fairseq.modules.grad_multiply import GradMultiply
from sklearn.cluster import MiniBatchKMeans

from configs.config import Config
from i18n.i18n import I18nAuto
from infer.lib.train.extract import FeatureExtractor

cwd = Path.cwd()
sys.path.append(str(cwd))
load_dotenv()

logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

weight_root = Path(os.getenv("weight_root", ""))
weight_uvr5_root = Path(os.getenv("weight_uvr5_root", ""))
index_root = Path(os.getenv("index_root", ""))
outside_index_root = Path(os.getenv("outside_index_root", ""))


# Cleanup runtime packages
for pack_dir in ["infer_pack", "uvr5_pack"]:
    shutil.rmtree(cwd / f"runtime/Lib/site-packages/{pack_dir}", ignore_errors=True)


warnings.filterwarnings("ignore")
torch.manual_seed(114514)

multiprocessing.set_start_method("spawn", force=True)
config = Config()


if config.dml:

    def forward_dml(ctx: Any, x: torch.Tensor, scale: float) -> torch.Tensor:
        ctx.scale = scale
        return x.clone().detach()

    GradMultiply.forward = forward_dml

i18n = I18nAuto()
logger.info(i18n)


def get_gpu_info() -> str:
    n_gpu = torch.cuda.device_count()
    details = []
    if torch.cuda.is_available() and n_gpu > 0:
        for i in range(n_gpu):
            name = torch.cuda.get_device_name(i)
            details.append(f"{i}\t{name}")
    gpus = "-".join(info[0] for info in details)
    return gpus


# Module-level worker functions for multiprocessing
def _worker_process_f0_gpu(paths: list[Any], log_file: Path, gpu_id: str) -> None:
    """Worker function for GPU-based F0 extraction."""
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    FeatureExtractor().extract_f0_for_files(paths, "rmvpe_gpu", log_file, device="cuda")


def _worker_process_f0_cpu(
    paths: list[Any], log_file: Path, extract_method: str
) -> None:
    """Worker function for CPU-based F0 extraction."""
    FeatureExtractor().extract_f0_for_files(
        paths, extract_method, log_file, device="cpu"
    )


def _worker_process_features_gpu(
    paths: list[Any], log_file: Path, gpu_id: str, version: str
) -> None:
    """Worker function for GPU-based feature extraction."""
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    FeatureExtractor().extract_model_features_for_files(paths, log_file, version)


def _preprocess_dataset(
    training_file: gr.FileData, exp_dir: str, sample_rate_str: str, n_p: int
) -> Generator[str, None, None]:
    """Preprocess dataset by resampling audio files."""
    training_dir = Path(str(training_file)).parent
    sr = int(sample_rate_str[:-1]) * 1000

    log_dir = cwd / "logs" / exp_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "preprocess.log"
    log_file.touch()

    cmd = (
        f'"{config.python_cmd}" infer/modules/train/preprocess.py '
        f'"{training_dir}" {sr} {n_p} "{log_dir}" {config.noparallel} {config.preprocess_per:.1f}'
    )
    logger.info("Execute: %s", cmd)
    p = Popen(cmd, shell=True)
    while p.poll() is None:
        yield log_file.read_text()
        sleep(1)

    log = log_file.read_text()
    logger.info(log)
    yield log


def _extract_pitch_features(
    gpus: str,
    num_cpu_processes: int,
    extract_method: str,
    should_guide: bool,
    project_dir: str | Path,
    version: str,
    gpu_ids_rmvpe: str,
) -> Generator[str, None, None]:
    """Extract pitch features and model features using parallel processing."""
    log_dir = cwd / "logs" / project_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / "extract_f0_feature.log"
    log_file.unlink(missing_ok=True)
    log_file.touch()

    # Step 1: Extract pitch (f0) if pitch guidance is enabled
    if should_guide:
        input_root = log_dir / "1_16k_wavs"
        opt_root1 = log_dir / "2a_f0"
        opt_root2 = log_dir / "2b-f0nsf"
        opt_root1.mkdir(parents=True, exist_ok=True)
        opt_root2.mkdir(parents=True, exist_ok=True)

        # Build list of files to process
        file_paths = [
            [file_path, opt_root1 / file_path.name, opt_root2 / file_path.name]
            for file_path in sorted(input_root.iterdir())
            if file_path.is_file() and "spec" not in file_path.name
        ]

        # Determine device for RMVPE
        if extract_method == "rmvpe_gpu" and gpu_ids_rmvpe != "-":
            # Multi-GPU RMVPE extraction
            gpu_ids = gpu_ids_rmvpe.split("-")
            n_processes = len(gpu_ids)

            processes = [
                Process(
                    target=_worker_process_f0_gpu,
                    args=(file_paths[idx::n_processes], log_file, gpu_id),
                )
                for idx, gpu_id in enumerate(gpu_ids)
            ]
            for p in processes:
                p.start()

            while any(p.is_alive() for p in processes):
                yield log_file.read_text()
                sleep(1)

            for p in processes:
                p.join()

        elif extract_method == "rmvpe_gpu":
            # DirectML RMVPE extraction
            try:
                import torch_directml  # type: ignore

                device = torch_directml.device(torch_directml.default_device())
            except ImportError:
                device = "cpu"
                logger.warning("torch_directml not available, falling back to CPU")

            # Process in main thread for DML
            FeatureExtractor().extract_f0_for_files(
                file_paths, "rmvpe", log_file, device=device
            )
            yield log_file.read_text()
        else:
            # CPU-based extraction (pm, harvest, dio, rmvpe on CPU)
            processes = [
                Process(
                    target=_worker_process_f0_cpu,
                    args=(
                        file_paths[i::num_cpu_processes],
                        log_file,
                        extract_method,
                    ),
                )
                for i in range(num_cpu_processes)
            ]
            for p in processes:
                p.start()

            while any(p.is_alive() for p in processes):
                yield log_file.read_text()
                sleep(1)

            for p in processes:
                p.join()

        # Final yield after f0 extraction
        log = log_file.read_text()
        logger.info(log)
        yield log

    # Step 2: Extract model features using GPUs
    wav_path = log_dir / "1_16k_wavs"
    out_path = log_dir / ("3_feature256" if version == "v1" else "3_feature768")
    out_path.mkdir(parents=True, exist_ok=True)

    # Build list of wav files to process
    wav_files = [
        (wav_file, out_path / wav_file.with_suffix(".npy").name)
        for wav_file in sorted(wav_path.iterdir())
        if wav_file.suffix == ".wav"
    ]

    gpu_list = gpus.split("-")

    processes = [
        Process(
            target=_worker_process_features_gpu,
            args=(wav_files[idx :: len(gpu_list)], log_file, gpu_id, version),
        )
        for idx, gpu_id in enumerate(gpu_list)
    ]
    for p in processes:
        p.start()

    while any(p.is_alive() for p in processes):
        yield log_file.read_text()
        sleep(1)

    for p in processes:
        p.join()

    log = log_file.read_text()
    logger.info(log)
    yield log


_GPUVisible = not config.dml


def _get_pretrained_models(path: str, f0: str, sr: str) -> tuple[str, str]:
    gen_path = f"assets/pretrained{path}/{f0}G{sr}.pth"
    gen_exists = os.access(gen_path, os.F_OK)
    disc_path = f"assets/pretrained{path}/{f0}D{sr}.pth"
    disc_exists = os.access(disc_path, os.F_OK)
    if not gen_exists:
        logger.warning("%s not exist, will not use pretrained model", gen_path)
    if not disc_exists:
        logger.warning("%s not exist, will not use pretrained model", disc_path)
    return (
        gen_path if gen_exists else "",
        disc_path if disc_exists else "",
    )


def _change_sr(sr: str, if_f0: bool, version: str) -> tuple[str, str]:
    """Change sample rate and return corresponding pretrained model paths."""
    path_str = "" if version == "v1" else "_v2"
    f0_str = "f0" if if_f0 else ""
    return _get_pretrained_models(path_str, f0_str, sr)


def _change_f0(if_f0: bool, sample_rate: str, version: str):
    path_str = "" if version == "v1" else "_v2"
    update: dict[str, bool | str] = {"visible": if_f0, "__type__": "update"}
    return (
        update,
        update,
        *_get_pretrained_models(path_str, "f0" if if_f0 else "", sample_rate),
    )


def _change_version(
    sample_rate: str, if_f0: bool, version: str
) -> tuple[str, str, dict[str, Any]]:
    path_str = "" if version == "v1" else "_v2"
    if sample_rate == "32k" and version == "v1":
        sample_rate = "40k"
    sample_rate_update = (
        {"choices": ["40k", "48k"], "__type__": "update", "value": sample_rate}
        if version == "v1"
        else {
            "choices": ["40k", "48k", "32k"],
            "__type__": "update",
            "value": sample_rate,
        }
    )
    f0_str = "f0" if if_f0 else ""
    return (
        *_get_pretrained_models(path_str, f0_str, sample_rate),
        sample_rate_update,
    )


def _click_train(
    project_dir: str,
    sr: str,
    if_f0: bool,
    spk_id: int,
    save_epoch: int,
    total_epoch: int,
    batch_size: int,
    if_save_latest: str,
    pretrained_G14: str,
    pretrained_D15: str,
    gpus: str,
    if_cache_gpu: str,
    if_save_every_weights: str,
    version: str,
) -> str:
    p_dir = Path.cwd() / "logs" / project_dir
    p_dir.mkdir(parents=True, exist_ok=True)
    """Project directory"""
    gt_wavs_dir = p_dir / "0_gt_wavs"
    feature_dir = p_dir / "3_feature256" if version == "v1" else p_dir / "3_feature768"
    if if_f0:
        f0_dir = p_dir / "2a_f0"
        f0nsf_dir = p_dir / "2b-f0nsf"
        names = (
            {name.stem for name in gt_wavs_dir.iterdir()}
            & {name.stem for name in feature_dir.iterdir()}
            & {name.stem for name in f0_dir.iterdir()}
            & {name.stem for name in f0nsf_dir.iterdir()}
        )
    else:
        names = {x.stem for x in gt_wavs_dir.iterdir()} & {
            x.stem for x in feature_dir.iterdir()
        }
    opt: list[str] = []
    for name in names:
        if if_f0:
            opt.append(
                f"{gt_wavs_dir}/{name}.wav|{feature_dir}/{name}.npy|{f0_dir}/{name}.wav.npy|{f0nsf_dir}/{name}.wav.npy|{spk_id}"
            )
        else:
            opt.append(f"{gt_wavs_dir}/{name}.wav|{feature_dir}/{name}.npy|{spk_id}")
    fea_dim = 256 if version == "v1" else 768
    if if_f0:
        for _ in range(2):
            opt.append(
                f"{Path.cwd()}/logs/mute/0_gt_wavs/mute{sr}.wav|{Path.cwd()}/logs/mute/3_feature{fea_dim}/mute.npy|{Path.cwd()}/logs/mute/2a_f0/mute.wav.npy|{Path.cwd()}/logs/mute/2b-f0nsf/mute.wav.npy|{spk_id}"
            )
    else:
        for _ in range(2):
            opt.append(
                f"{Path.cwd()}/logs/mute/0_gt_wavs/mute{sr}.wav|{Path.cwd()}/logs/mute/3_feature{fea_dim}/mute.npy|{spk_id}"
            )
    shuffle(opt)
    with (p_dir / "filelist.txt").open("w") as f:
        f.write("\n".join(opt))
    logger.debug("Write filelist done")
    logger.info("Use gpus: %s", str(gpus))
    if pretrained_G14 == "":
        logger.info("No pretrained Generator")
    if pretrained_D15 == "":
        logger.info("No pretrained Discriminator")
    if version == "v1" or sr == "40k":
        config_path = "v1/%s.json" % sr
    else:
        config_path = "v2/%s.json" % sr
    config_save_path = p_dir / "config.json"
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
    if gpus:
        cmd = f'"{config.python_cmd}" infer/modules/train/train.py -e "{project_dir}" -sr {sr} -f0 {1 if if_f0 else 0} -bs {batch_size} -g {gpus} -te {total_epoch} -se {save_epoch} {f"-pg {pretrained_G14}" if pretrained_G14 != "" else ""} {f"-pd {pretrained_D15}" if pretrained_D15 != "" else ""} -l {1 if if_save_latest == i18n("是") else 0} -c {1 if if_cache_gpu == i18n("是") else 0} -sw {1 if if_save_every_weights == i18n("是") else 0} -v {version}'
    else:
        cmd = f'"{config.python_cmd}" infer/modules/train/train.py -e "{project_dir}" -sr {sr} -f0 {1 if if_f0 else 0} -bs {batch_size} -te {total_epoch} -se {save_epoch} {f"-pg {pretrained_G14}" if pretrained_G14 != "" else ""} {f"-pd {pretrained_D15}" if pretrained_D15 != "" else ""} -l {1 if if_save_latest == i18n("是") else 0} -c {1 if if_cache_gpu == i18n("是") else 0} -sw {1 if if_save_every_weights == i18n("是") else 0} -v {version}'
    logger.info("Execute: %s", cmd)
    p = Popen(cmd, shell=True, cwd=Path.cwd())
    p.wait()
    return "Training complete. You can view the training log in the console or in the train.log file in the experiment folder."


def _train_index(project_dir: Path, version: str) -> Generator[str, None, str]:
    project_dir = Path("logs") / project_dir
    project_dir.mkdir(parents=True, exist_ok=True)
    feature_dir = (
        project_dir / "3_feature256"
        if version == "v1"
        else project_dir / "3_feature768"
    )
    if not feature_dir.exists():
        return "Please extract features first!"
    listdir_res = list(feature_dir.iterdir())
    if len(listdir_res) == 0:
        return "Please extract features first!"
    infos = []
    npys = []
    for name in sorted(listdir_res):
        phone = np.load(name)
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

    np.save(f"{project_dir}/total_fea.npy", big_npy)
    n_ivf = min(int(16 * np.sqrt(big_npy.shape[0])), big_npy.shape[0] // 39)
    infos.append(f"{big_npy.shape},{n_ivf}")
    yield "\n".join(infos)
    index = faiss.index_factory(256 if version == "v1" else 768, f"IVF{n_ivf},Flat")

    infos.append("training")
    yield "\n".join(infos)
    index_ivf = faiss.extract_index_ivf(index)
    index_ivf.nprobe = 1
    index.train(big_npy)
    file_name = (
        f"IVF{n_ivf}_Flat_nprobe_{index_ivf.nprobe}_{project_dir.stem}_{version}.index"
    )
    faiss.write_index(
        index,
        f"{project_dir}/trained_{file_name}.index",
    )
    infos.append("adding")
    yield "\n".join(infos)
    batch_size_add = 8192
    for i in range(0, big_npy.shape[0], batch_size_add):
        index.add(big_npy[i : i + batch_size_add])
    faiss.write_index(
        index,
        f"{project_dir}/added_{file_name}.index",
    )
    infos.append(f"Successfully built index: added_{file_name}.index")
    try:
        source_path = project_dir / f"added_{file_name}"
        target_path = outside_index_root / f"{project_dir.stem}_{file_name}"
        if platform.system() == "Windows":
            target_path.hardlink_to(source_path)
        else:
            target_path.symlink_to(source_path)
        infos.append(f"Linked index to external directory - {outside_index_root}")
    except:
        infos.append(
            f"Failed to link index to external directory - {outside_index_root}"
        )

    yield "\n".join(infos)
    return "Index training complete."


if __name__ == "__main__":
    # Create necessary directories
    for dir_path in ["logs", "assets/weights"]:
        (cwd / dir_path).mkdir(parents=True, exist_ok=True)

    gpus = get_gpu_info()

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
        project_dir = gr.Textbox(label=i18n("Enter project name"), value="mi-test")
        training_file = gr.File(
            label=i18n("Upload training file"),
            file_count="single",
        )
        sample_rate = gr.Radio(
            label=i18n("Target sample rate"),
            choices=["40k", "48k"],
            value="40k",
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

        preprocess_output = gr.Textbox(
            label=i18n("Output Information"),
            value="",
            max_lines=8,
            lines=4,
            autoscroll=True,
        )
        preprocess_button = gr.Button(i18n("Process data"), variant="primary")
        preprocess_button.click(
            _preprocess_dataset,
            [training_file, project_dir, sample_rate, num_cpu_processes],
            [preprocess_output],
            api_name="train_preprocess",
        )

        include_pitch_guidance = gr.Radio(
            label=i18n(
                "Does the model use pitch guidance? (Required for singing, optional for speech)"
            ),
            choices=[True, False],
            value=True,
            interactive=True,
        )
        version = gr.Radio(
            label=i18n("Version"),
            choices=["v1", "v2"],
            value="v2",
            interactive=True,
            visible=True,
        )

        gpu_ids_input = gr.Textbox(
            label=i18n(
                "Enter GPU IDs separated by '-', e.g. 0-1-2 to use GPU 0, 1, and 2"
            ),
            value=gpus,
            interactive=True,
            visible=_GPUVisible,
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
        gpu_ids_rmvpe = gr.Textbox(
            label=i18n(
                "rmvpe GPU configuration: Enter different process GPU IDs separated by '-',"
                " e.g. 0-0-1 runs 2 processes on GPU 0 and 1 process on GPU 1"
            ),
            value=f"{gpus}-{gpus}",
            interactive=True,
            visible=_GPUVisible,
        )
        btn_extract_features = gr.Button(i18n("Extract Features"), variant="primary")
        feature_extraction_output = gr.Textbox(
            label=i18n("Output Information"),
            value="",
            max_lines=8,
            lines=4,
            autoscroll=True,
        )
        pitch_extraction_method.change(
            fn=lambda method: {
                "visible": method == "rmvpe_gpu" and _GPUVisible,
                "__type__": "update",
            },
            inputs=[pitch_extraction_method],
            outputs=[gpu_ids_rmvpe],
        )
        btn_extract_features.click(
            _extract_pitch_features,
            [
                gpu_ids_input,
                num_cpu_processes,
                pitch_extraction_method,
                include_pitch_guidance,
                project_dir,
                version,
                gpu_ids_rmvpe,
            ],
            [feature_extraction_output],
            api_name="train_extract_f0_feature",
        )

        default_batch_size = int(
            torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024 + 0.4
        )
        gr.Markdown(
            value=i18n(
                "step3: Fill in training settings and start training model and index"
            )
        )
        save_interval = gr.Slider(
            minimum=1,
            maximum=50,
            step=1,
            label=i18n("Save frequency (save_every_epoch)"),
            value=5,
            interactive=True,
        )
        total_epochs = gr.Slider(
            minimum=2,
            maximum=1000,
            step=1,
            label=i18n("Total training epochs (total_epoch)"),
            value=20,
            interactive=True,
        )
        batch_size = gr.Slider(
            minimum=1,
            maximum=40,
            step=1,
            label=i18n("Batch size per GPU"),
            value=default_batch_size,
            interactive=True,
        )
        if_save_latest = gr.Radio(
            label=i18n("Only save the latest ckpt file to save disk space"),
            choices=[i18n("Yes"), i18n("No")],
            value=i18n("No"),
            interactive=True,
        )
        if_cache = gr.Radio(
            label=i18n(
                "Cache all training set to GPU memory. For small data under 10min, caching can speed up training. Large data may cause out-of-memory and doesn't speed up much."
            ),
            choices=[i18n("Yes"), i18n("No")],
            value=i18n("No"),
            interactive=True,
        )
        if_save_every_weights = gr.Radio(
            label=i18n(
                "Save the final small model to the weights folder at every save point"
            ),
            choices=[i18n("Yes"), i18n("No")],
            value=i18n("No"),
            interactive=True,
        )
        pretrained_G14 = gr.Textbox(
            label=i18n("Load pretrained base model G path"),
            value="assets/pretrained_v2/f0G40k.pth",
            interactive=True,
        )
        pretrained_D15 = gr.Textbox(
            label=i18n("Load pretrained base model D path"),
            value="assets/pretrained_v2/f0D40k.pth",
            interactive=True,
        )
        use_pitch_guidance = gr.Radio(
            label=i18n(
                "Does the model use pitch guidance? (Required for singing, optional for speech)"
            ),
            choices=[True, False],
            value=True,
            interactive=True,
        )
        pitch_extraction_algorithm = gr.Radio(
            label=i18n(
                "Select pitch extraction algorithm: For singing, use pm for speed; for high-quality speech but slow CPU, use dio for speed; harvest is better quality but slower; rmvpe has the best effect and uses some CPU/GPU"
            ),
            choices=["pm", "harvest", "dio", "rmvpe", "rmvpe_gpu"],
            value="rmvpe_gpu",
            interactive=True,
        )

        gpus_rmvpe = gr.Textbox(
            label=i18n(
                "rmvpe GPU configuration: Enter different process GPU IDs separated by '-', e.g. 0-0-1 runs 2 processes on GPU 0 and 1 process on GPU 1"
            ),
            value="%s-%s" % (gpus, gpus),
            interactive=True,
            visible=_GPUVisible,
        )
        use_pitch_guidance.change(
            _change_f0,
            [use_pitch_guidance, sample_rate, version],
            [pitch_extraction_algorithm, gpus_rmvpe, pretrained_G14, pretrained_D15],
        )

        sample_rate.change(
            _change_sr,
            [sample_rate, use_pitch_guidance, version],
            [pretrained_G14, pretrained_D15],
        )
        version.change(
            _change_version,
            [sample_rate, use_pitch_guidance, version],
            [pretrained_G14, pretrained_D15, sample_rate],
        )

        gpu_ids_input = gr.Textbox(
            label=i18n(
                "Enter GPU IDs separated by '-', e.g. 0-1-2 to use GPU 0, 1, and 2"
            ),
            value=gpus,
            interactive=True,
        )
        speaker_id = gr.Slider(
            minimum=0,
            maximum=4,
            step=1,
            label=i18n("Please specify speaker id"),
            value=0,
            interactive=True,
        )
        train_model = gr.Button(i18n("Train Model"), variant="primary")
        btn_train_feature_index = gr.Button(
            i18n("Train Feature Index"), variant="primary"
        )
        output_info_textbox = gr.Textbox(
            label=i18n("Output Information"),
            value="",
            lines=4,
            max_lines=10,
        )

        train_model.click(
            _click_train,
            [
                project_dir,
                sample_rate,
                use_pitch_guidance,
                speaker_id,
                save_interval,
                total_epochs,
                batch_size,
                if_save_latest,
                pretrained_G14,
                pretrained_D15,
                gpu_ids_input,
                if_cache,
                if_save_every_weights,
                version,
            ],
            output_info_textbox,
            api_name="train_start",
        )
        btn_train_feature_index.click(
            _train_index, [project_dir, version], output_info_textbox
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
