import traceback
import logging
from typing import Any, Dict, List, Literal, Optional, Union
import gradio as gr
import resampy

from configs.config import Config
logger = logging.getLogger(__name__)

import numpy as np
import soundfile as sf
import torch
from io import BytesIO

from infer.lib.audio import load_audio, wav2
from infer.lib.infer_pack.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)
from infer.modules.vc.pipeline import Pipeline
from infer.modules.vc.utils import *


def resample_audio(
    audio_array: np.ndarray,  # Your input audio array, potentially stereo
    orig_sr: int,
    target_sr: int,
):
    # Check if the audio is stereo and downmix to mono
    if audio_array.ndim > 1 and audio_array.shape[1] > 1:
        # print("Detected stereo audio, downmixing to mono.")
        # Average the channels to create a mono signal
        audio_mono = audio_array.mean(axis=1)
    else:
        # Already mono or 1D array
        audio_mono = audio_array.flatten()  # Ensure it's 1D in case it's (N, 1)

    # print(f"Mono audio shape after downmixing: {audio_mono.shape}")

    if audio_mono.size < 10:  # A reasonable minimum length for resampling
        raise ValueError(
            f"Mono audio signal length ({audio_mono.size}) is too small to resample from {orig_sr} to {target_sr}. "
            "Ensure the audio file contains actual sound data."
        )

    # Perform resampling on the mono signal
    resampled_audio = resampy.resample(audio_mono, orig_sr, target_sr)
    # print(f"Resampled audio shape: {resampled_audio.shape}")
    return resampled_audio


class VC:
    def __init__(self: "VC", config: Config):
        # self.config = config
        self.n_spk: Optional[int] = None
        self.tgt_sr: Optional[int] = None
        self.net_g: Optional[
            Union[
                SynthesizerTrnMs256NSFsid,
                SynthesizerTrnMs256NSFsid_nono,
                SynthesizerTrnMs768NSFsid,
                SynthesizerTrnMs768NSFsid_nono,
            ]
        ] = None
        self.pipeline: Optional[Pipeline] = None
        self.cpt: Optional[Dict[str, Any]] = None
        self.version: Optional[str] = None
        self.if_f0: Optional[int] = None
        self.hubert_model: Optional[HubertModel] = None
        self.config: Config = config

    def get_vc(self: "VC", sid: Optional[str], *to_return_protect):
        if sid is None or sid == "":
            logger.warning("No SID")
            return (
            {"visible": True, "value": 0.5, "__type__": "update"},
            {"choices": [], "value": "", "__type__": "update"}
        )
        logger.info("Get sid: " + sid)

        to_return_protect0 = {
            "visible": self.if_f0 != 0,
            "value": (
                to_return_protect[0] if self.if_f0 != 0 and to_return_protect else 0.5
            ),
            "__type__": "update",
        }

        if sid == "" or sid == []:
            if (
                self.hubert_model is not None
            ):  # 考虑到轮询, 需要加个判断看是否 sid 是由有模型切换到无模型的
                logger.info("Clean model cache")
                del (self.net_g, self.n_spk, self.hubert_model, self.tgt_sr)  # ,cpt
                self.hubert_model = self.net_g = self.n_spk = self.hubert_model = (
                    self.tgt_sr
                ) = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                ###楼下不这么折腾清理不干净
                self.if_f0 = self.cpt.get("f0", 1)
                self.version = self.cpt.get("version", "v1")
                if self.version == "v1":
                    if self.if_f0 == 1:
                        self.net_g = SynthesizerTrnMs256NSFsid(
                            *self.cpt["config"], is_half=self.config.is_half
                        )
                    else:
                        self.net_g = SynthesizerTrnMs256NSFsid_nono(*self.cpt["config"])
                elif self.version == "v2":
                    if self.if_f0 == 1:
                        self.net_g = SynthesizerTrnMs768NSFsid(
                            *self.cpt["config"], is_half=self.config.is_half
                        )
                    else:
                        self.net_g = SynthesizerTrnMs768NSFsid_nono(*self.cpt["config"])
                del self.net_g, self.cpt
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            return (
                {
                    "visible": True,
                    "value": to_return_protect0,
                    "__type__": "update",
                },
                "",
                "",
            )
        person = f'{os.getenv("weight_root")}/{sid}'
        logger.info(f"Loading: {person}")

        self.cpt = torch.load(person, map_location="cpu", weights_only=False)
        self.tgt_sr = self.cpt["config"][-1]
        self.cpt["config"][-3] = self.cpt["weight"]["emb_g.weight"].shape[0]  # n_spk
        self.if_f0 = self.cpt.get("f0", 1)
        self.version = self.cpt.get("version", "v1")

        synthesizer_class = {
            ("v1", 1): SynthesizerTrnMs256NSFsid,
            ("v1", 0): SynthesizerTrnMs256NSFsid_nono,
            ("v2", 1): SynthesizerTrnMs768NSFsid,
            ("v2", 0): SynthesizerTrnMs768NSFsid_nono,
        }

        self.net_g = synthesizer_class.get(
            (self.version, self.if_f0), SynthesizerTrnMs256NSFsid
        )(*self.cpt["config"], is_half=self.config.is_half)

        del self.net_g.enc_q

        self.net_g.load_state_dict(self.cpt["weight"], strict=False)
        self.net_g.eval().to(self.config.device)
        if self.config.is_half:
            self.net_g = self.net_g.half()
        else:
            self.net_g = self.net_g.float()

        self.pipeline = Pipeline(self.tgt_sr, self.config)
        n_spk = self.cpt["config"][-3]
        index = {"value": get_index_path_from_model(sid), "__type__": "update"}
        logger.info("Select index: " + index["value"])
        res = (
            (
                to_return_protect0,
                index,
            )
            # if to_return_protect
            # else {"visible": True, "maximum": n_spk, "__type__": "update"}
        )
        logger.info(f"Result {res}")

        return res

    def vc_single(
        self: "VC",
        # sid: int,
        sr_and_audio: Optional[Tuple[int, np.ndarray]],
        f0_up_key: int,
        f0_method: str,
        # file_index: Optional[str],  # Path to .index file from textbox
        file_index2: Optional[str],  # Path to .index file from dropdown
        index_rate: float,
        # filter_radius: int,  # Typically an integer for radius
        resample_sr: int,  # Target sample rate, typically an integer
        rms_mix_rate: float,
        protect: float,
        progress: gr.Progress = gr.Progress(),
    ) -> Tuple[str, Optional[Tuple[int, np.ndarray]]]:
        file_index = None
        f0_file = None
        sid = 0
        filter_radius = 3
        # if input_audio_path is None:
        #     return "Audio is required", None
        # if audio is None:
        # return "Audio is required", None
        f0_up_key = int(f0_up_key)
        try:
            if sr_and_audio is None:
                return "Audio is required", None

            original_sr, audio = sr_and_audio
            if original_sr != 16000:
                # print(f"Resampling audio from {original_sr} Hz to {16000} Hz")
                audio = resample_audio(audio, original_sr, 16000)
            audio_max: np.float64 = np.abs(audio).max() / 0.95
            if audio_max > 1:
                audio /= audio_max
            times = [0, 0, 0]

            if self.hubert_model is None:
                # torch.serialization.add_safe_globals([Dictionary])
                self.hubert_model = load_hubert(self.config)

            if file_index:
                file_index = (
                    file_index.strip(" ")
                    .strip('"')
                    .strip("\n")
                    .strip('"')
                    .strip(" ")
                    .replace("trained", "added")
                )
            elif file_index2:
                file_index = file_index2
            else:
                file_index = ""

            audio_opt: np.ndarray = self.pipeline.pipeline(
                model=self.hubert_model,
                net_g=self.net_g,
                sid=sid,
                audio=audio,
                input_audio_path="NA",
                times=times,
                f0_up_key=f0_up_key,
                f0_method=f0_method,
                file_index=file_index,
                index_rate=index_rate,
                if_f0=self.if_f0,
                filter_radius=filter_radius,
                tgt_sr=self.tgt_sr,
                resample_sr=resample_sr,
                rms_mix_rate=rms_mix_rate,
                version=self.version,
                protect=protect,
                f0_file=f0_file,
                progress=progress,
            )
            if self.tgt_sr != resample_sr >= 16000:
                tgt_sr = resample_sr
            else:
                tgt_sr = self.tgt_sr
            index_info = (
                "Index:\n%s." % file_index
                if os.path.exists(file_index)
                else "Index not used."
            )
            return (
                "Success.\n%s\nTime:\nnpy: %.2fs, f0: %.2fs, infer: %.2fs."
                % (index_info, *times),
                (tgt_sr, audio_opt),
            )
        except:
            info = traceback.format_exc()
            logger.warning(info)
            return f"Failed with error:\n{info}", None

    def vc_multi(
        self: "VC",
        sid: int,
        dir_path: str,
        opt_root: str,
        paths: List[Union[str, Any]],  # Paths can be strings or gradio File objects
        f0_up_key: Union[int, float],
        f0_method: str,
        file_index: Optional[str],
        file_index2: Optional[str],
        index_rate: float,
        filter_radius: int,
        resample_sr: int,
        rms_mix_rate: float,
        protect: float,
        format1: str,
    ):
        try:
            dir_path = (
                dir_path.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
            )  # 防止小白拷路径头尾带了空格和"和回车
            opt_root = opt_root.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
            os.makedirs(opt_root, exist_ok=True)
            try:
                if dir_path != "":
                    paths = [
                        os.path.join(dir_path, name) for name in os.listdir(dir_path)
                    ]
                else:
                    paths = [path.name for path in paths]
            except:
                traceback.print_exc()
                paths = [path.name for path in paths]
            infos = []
            for path in paths:
                info, opt = self.vc_single(
                    sid,
                    path,
                    f0_up_key,
                    None,
                    f0_method,
                    file_index,
                    file_index2,
                    # file_big_npy,
                    index_rate,
                    filter_radius,
                    resample_sr,
                    rms_mix_rate,
                    protect,
                )
                if "Success" in info:
                    try:
                        tgt_sr, audio_opt = opt
                        if format1 in ["wav", "flac"]:
                            sf.write(
                                "%s/%s.%s"
                                % (opt_root, os.path.basename(path), format1),
                                audio_opt,
                                tgt_sr,
                            )
                        else:
                            path = "%s/%s.%s" % (
                                opt_root,
                                os.path.basename(path),
                                format1,
                            )
                            with BytesIO() as wavf:
                                sf.write(wavf, audio_opt, tgt_sr, format="wav")
                                wavf.seek(0, 0)
                                with open(path, "wb") as outf:
                                    wav2(wavf, outf, format1)
                    except:
                        info += traceback.format_exc()
                infos.append("%s->%s" % (os.path.basename(path), info))
                yield "\n".join(infos)
            yield "\n".join(infos)
        except:
            yield traceback.format_exc()
