import traceback
import logging
from typing import Any, Dict, List, Optional, Union

import librosa

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

            logger.info("No SID")
            return
        logger.info("Get sid: " + sid)

        to_return_protect0 = {
            "visible": self.if_f0 != 0,
            "value": (
                to_return_protect[0] if self.if_f0 != 0 and to_return_protect else 0.5
            ),
            "__type__": "update",
        }
        to_return_protect1 = {
            "visible": self.if_f0 != 0,
            "value": (
                to_return_protect[1] if self.if_f0 != 0 and to_return_protect else 0.33
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
                {"visible": False, "__type__": "update"},
                {
                    "visible": True,
                    "value": to_return_protect0,
                    "__type__": "update",
                },
                {
                    "visible": True,
                    "value": to_return_protect1,
                    "__type__": "update",
                },
                "",
                "",
            )
        person = f'{os.getenv("weight_root")}/{sid}'
        logger.info(f"Loading: {person}")

        self.cpt = torch.load(person, map_location="cpu")
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

        return (
            (
                {"visible": True, "maximum": n_spk, "__type__": "update"},
                to_return_protect0,
                to_return_protect1,
                index,
                index,
            )
            if to_return_protect
            else {"visible": True, "maximum": n_spk, "__type__": "update"}
        )

    def vc_single(
        self: "VC",
        sid: int,  # Speaker ID, typically an integer
        input_audio_path: Optional[str],
        f0_up_key: Union[
            int, float
        ],  # Pitch change, can be int or float from gr.Number
        f0_file: Optional[str],  # Path to F0 file, if provided
        f0_method: str,
        file_index: Optional[str],  # Path to .index file from textbox
        file_index2: Optional[str],  # Path to .index file from dropdown
        index_rate: float,
        filter_radius: int,  # Typically an integer for radius
        resample_sr: int,  # Target sample rate, typically an integer
        rms_mix_rate: float,
        protect: float,
    ):
        if input_audio_path is None:
            return "Audio is required", None
        f0_up_key = int(f0_up_key)
        try:
            audio = load_audio(input_audio_path, 16000)
            audio_max: np.float64 = np.abs(audio).max() / 0.95
            if audio_max > 1:
                audio /= audio_max
            times = [0, 0, 0]

            if self.hubert_model is None:
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
                file_index = ""  # 防止小白写错，自动帮他替换掉

            audio_opt = self.pipeline.pipeline(
                self.hubert_model,
                self.net_g,
                sid,
                audio,
                input_audio_path,
                times,
                f0_up_key,
                f0_method,
                file_index,
                index_rate,
                self.if_f0,
                filter_radius,
                self.tgt_sr,
                resample_sr,
                rms_mix_rate,
                self.version,
                protect,
                f0_file,
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
            return info, (None, None)

    def vc_long(
        self: "VC",
        sid: int,  # Speaker ID, typically an integer
        input_audio_path: Optional[str],
        f0_up_key: Union[
            int, float
        ],  # Pitch change, can be int or float from gr.Number
        f0_file: Optional[str],  # Path to F0 file, if provided
        f0_method: str,
        file_index: Optional[str],  # Path to .index file from textbox
        file_index2: Optional[str],  # Path to .index file from dropdown
        index_rate: float,
        filter_radius: int,  # Typically an integer for radius
        resample_sr: int,  # Target sample rate, typically an integer
        rms_mix_rate: float,
        protect: float,
        chunk_length: int = 15,  # Add a parameter for chunk length in seconds
        overlap_length: int = 2,  # Add a parameter for overlap length in seconds
    ):
        if input_audio_path is None:
            return "Audio is required", None
        f0_up_key = int(f0_up_key)

        try:
            audio, sr = sf.read(
                input_audio_path
            )  # Use soundfile to get sample rate directly
            audio = (audio * 32768).astype(
                np.float32
            )  # Convert to int16 range, then to float32
            # Handle stereo to mono conversion if necessary (common for RVC)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)

            # Resample to 16000Hz for Hubert if needed
            if sr != 16000:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                sr = 16000

            audio_max = np.abs(audio).max() / 0.95
            if audio_max > 1:
                audio /= audio_max

            if self.hubert_model is None:
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

            # --- Chunking Logic ---
            total_length_samples = audio.shape[0]
            chunk_length_samples = chunk_length * sr
            overlap_samples = overlap_length * sr

            output_audio_chunks = []
            start_time = 0

            total_inference_time = [0, 0, 0]  # To accumulate times

            while start_time < total_length_samples:
                end_time = min(start_time + chunk_length_samples, total_length_samples)
                current_chunk = audio[start_time:end_time]

                # Add overlap to the beginning of the chunk (except for the first chunk)
                # and to the end (except for the last chunk)
                current_overlap_start = max(0, start_time - overlap_samples)
                current_overlap_end = min(
                    total_length_samples, end_time + overlap_samples
                )

                # Adjust chunk to include overlap for processing
                chunk_for_processing = audio[current_overlap_start:current_overlap_end]

                # Calculate the start and end of the original chunk within the processed chunk
                original_chunk_start_in_processed = start_time - current_overlap_start
                original_chunk_end_in_processed = (
                    original_chunk_start_in_processed + current_chunk.shape[0]
                )

                if chunk_for_processing.shape[0] == 0:
                    break  # Avoid processing empty chunks

                times = [0, 0, 0]  # Times for current chunk

                logger.info(
                    f"Processing chunk from {start_time/sr:.2f}s to {end_time/sr:.2f}s"
                )

                audio_opt_chunk = self.pipeline.pipeline(
                    self.hubert_model,
                    self.net_g,
                    sid,
                    chunk_for_processing,  # Pass the chunk with overlap
                    input_audio_path,  # Keep original path for logging if needed
                    times,
                    f0_up_key,
                    f0_method,
                    file_index,
                    index_rate,
                    self.if_f0,
                    filter_radius,
                    self.tgt_sr,
                    resample_sr,
                    rms_mix_rate,
                    self.version,
                    protect,
                    f0_file,
                )

                # Accumulate times
                total_inference_time[0] += times[0]
                total_inference_time[1] += times[1]
                total_inference_time[2] += times[2]

                # Extract the non-overlapping part from the processed chunk
                if audio_opt_chunk is not None:
                    # Determine the actual start and end of the original chunk within the *output* of the pipeline
                    # This assumes the pipeline maintains the relative timing and only changes content
                    output_sr = (
                        self.tgt_sr
                        if self.tgt_sr != resample_sr >= 16000
                        else resample_sr
                    )

                    # Calculate indices for the non-overlapping part in the output chunk
                    # This is crucial for seamless stitching
                    overlap_start_output_samples = overlap_samples * output_sr // sr

                    # If it's the first chunk, take from the beginning, otherwise from after the overlap
                    chunk_output_start_idx = (
                        0 if start_time == 0 else overlap_start_output_samples
                    )

                    # If it's the last chunk, take to the end, otherwise up to the end of the non-overlap
                    chunk_output_end_idx = (
                        audio_opt_chunk.shape[0]
                        if end_time == total_length_samples
                        else (
                            original_chunk_end_in_processed * output_sr // sr
                            - (
                                overlap_samples * output_sr // sr
                                if start_time + chunk_length_samples
                                < total_length_samples
                                else 0
                            )
                        )
                    )

                    # Ensure indices are within bounds
                    chunk_output_end_idx = min(
                        chunk_output_end_idx, audio_opt_chunk.shape[0]
                    )

                    # Take the non-overlapping portion of the output chunk
                    # If it's the very first chunk, we take from the beginning up to the point *before* the next overlap starts.
                    # If it's a middle chunk, we take the part *after* the previous overlap and *before* the next overlap.
                    # If it's the very last chunk, we take from *after* the previous overlap to the end.

                    # Simplified overlap handling for stitching:
                    # For the first chunk, take the whole thing and deal with potential overlap at the end.
                    # For middle chunks, trim `overlap_samples` from the start and end.
                    # For the last chunk, trim `overlap_samples` from the start and take to the end.

                    # A more robust overlap-add method would involve crossfading the overlaps.
                    # For simplicity, let's just trim for now.

                    if start_time == 0:  # First chunk
                        # Take the processed audio up to the point where the overlap for the *next* chunk would begin
                        non_overlap_end_idx = min(
                            audio_opt_chunk.shape[0],
                            chunk_length_samples * output_sr // sr,
                        )
                        output_audio_chunks.append(
                            audio_opt_chunk[:non_overlap_end_idx]
                        )
                    else:  # Subsequent chunks
                        # Take from the start of the current chunk's non-overlapping part
                        # up to the end of the chunk (or end of non-overlapping part if it's a middle chunk)
                        non_overlap_start_idx = overlap_samples * output_sr // sr
                        non_overlap_end_idx = min(
                            audio_opt_chunk.shape[0],
                            non_overlap_start_idx
                            + chunk_length_samples * output_sr // sr,
                        )

                        # If this is the last chunk, take everything from the non_overlap_start_idx to the end
                        if end_time == total_length_samples:
                            output_audio_chunks.append(
                                audio_opt_chunk[non_overlap_start_idx:]
                            )
                        else:
                            # For middle chunks, we might still have a trailing overlap
                            output_audio_chunks.append(
                                audio_opt_chunk[
                                    non_overlap_start_idx:non_overlap_end_idx
                                ]
                            )

                    # A more advanced solution would be to use a proper overlap-add method to prevent clicks
                    # For now, this is a basic trimming strategy.

                start_time += (
                    chunk_length_samples - overlap_samples
                )  # Move by non-overlapping part

            # Concatenate all processed chunks
            audio_opt = np.concatenate(output_audio_chunks)

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
                % (index_info, *total_inference_time),
                (tgt_sr, audio_opt),
            )
        except Exception:
            info = traceback.format_exc()
            logger.warning(info)
            return info, (None, None)

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
