import multiprocessing
import os
import sys
import traceback
from pathlib import Path

import librosa
import numpy as np
from scipy import signal
from scipy.io import wavfile

import shared
from infer.lib.audio import load_audio
from infer.lib.slicer2 import Slicer

# Global log file handle
f = None


def println(strr: str):
    print(strr)
    if f is not None:
        f.write(f"{strr}\n")
        f.flush()


class PreProcess:
    def __init__(self, sr: int, exp_dir: Path, per: float = 3.7):
        self.sr = sr
        self.per = per
        self.overlap = 0.3
        self.tail = self.per + self.overlap
        self.max = 0.9
        self.alpha = 0.75
        self.exp_dir = exp_dir
        self.gt_wavs_dir = exp_dir / shared.GT_WAVS_DIR_NAME
        self.wavs16k_dir = exp_dir / shared.WAVS_16K_DIR_NAME

        self.slicer = Slicer(
            sr=sr,
            threshold=-42,
            min_length=1500,
            min_interval=400,
            hop_size=15,
            max_sil_kept=500,
        )
        self.bh, self.ah = signal.butter(N=5, Wn=48, btype="high", fs=self.sr)

        self.exp_dir.mkdir(exist_ok=True)
        self.gt_wavs_dir.mkdir(exist_ok=True)
        self.wavs16k_dir.mkdir(exist_ok=True)

    def norm_write(self, tmp_audio: np.ndarray, idx0: int, idx1: int) -> None:
        tmp_max = np.abs(tmp_audio).max()
        if tmp_max > 2.5:
            print(f"{idx0}-{idx1}-{tmp_max}-filtered")
            return
        # Normalize audio
        tmp_audio = (tmp_audio / tmp_max * (self.max * self.alpha)) + (
            1 - self.alpha
        ) * tmp_audio

        gt_path = self.gt_wavs_dir / f"{idx0}_{idx1}.wav"
        wavs16k_path = self.wavs16k_dir / f"{idx0}_{idx1}.wav"

        wavfile.write(str(gt_path), self.sr, tmp_audio.astype(np.float32))
        tmp_audio_16k = librosa.resample(tmp_audio, orig_sr=self.sr, target_sr=16000)
        wavfile.write(str(wavs16k_path), 16000, tmp_audio_16k.astype(np.float32))

    def pipeline(self, path: Path, idx0: int):
        try:
            audio = load_audio(str(path), self.sr)
            audio = signal.lfilter(self.bh, self.ah, audio)

            idx1 = 0
            for audio_slice in self.slicer.slice(audio):
                i = 0
                while True:
                    start = int(self.sr * (self.per - self.overlap) * i)
                    i += 1
                    if len(audio_slice[start:]) > self.tail * self.sr:
                        tmp_audio = audio_slice[start : start + int(self.per * self.sr)]
                        self.norm_write(tmp_audio, idx0, idx1)
                        idx1 += 1
                    else:
                        tmp_audio = audio_slice[start:]
                        self.norm_write(tmp_audio, idx0, idx1)
                        idx1 += 1
                        break
            println(f"{path}\t-> Success")
        except Exception:
            println(f"{path}\t-> {traceback.format_exc()}")

    def pipeline_mp(self, infos: list[tuple[Path, int]]) -> None:
        for path, idx0 in infos:
            self.pipeline(path, idx0)

    def pipeline_mp_inp_dir(self, inp_root: Path, n_p: int, noparallel: bool) -> None:
        try:
            infos = [
                (inp_root / name, idx)
                for idx, name in enumerate(sorted(os.listdir(inp_root)))
            ]
            if noparallel:
                for i in range(n_p):
                    self.pipeline_mp(infos[i::n_p])
            else:
                ps = []
                for i in range(n_p):
                    p = multiprocessing.Process(
                        target=self.pipeline_mp, args=(infos[i::n_p],)
                    )
                    ps.append(p)
                    p.start()
                for p in ps:
                    p.join()
        except Exception:
            println(f"Fail. {traceback.format_exc()}")


def preprocess_trainset(
    inp_root: Path,
    sr: int,
    n_p: int,
    exp_dir: Path,
    per: float,
    noparallel: bool = False,
):
    global f

    # Open log file
    log_path = Path(f"{exp_dir}/preprocess.log")
    f = log_path.open("a+")

    try:
        pp = PreProcess(sr, exp_dir, per)
        println("start preprocess")
        pp.pipeline_mp_inp_dir(inp_root, n_p, noparallel)
        println("end preprocess")
    finally:
        if f is not None:
            f.close()
            f = None


if __name__ == "__main__":
    import sys

    print(*sys.argv[1:])
    inp_root = Path(sys.argv[1])
    sr = int(sys.argv[2])
    n_p = int(sys.argv[3])
    exp_dir = Path(sys.argv[4])
    noparallel = sys.argv[5] == "True"
    per = float(sys.argv[6])

    preprocess_trainset(inp_root, sr, n_p, exp_dir, per, noparallel)
