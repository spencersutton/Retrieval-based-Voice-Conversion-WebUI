import multiprocessing
import sys
import traceback
from pathlib import Path
from typing import TextIO

import librosa
import numpy as np
from scipy import signal
from scipy.io import wavfile

import shared
from infer.lib.audio import load_audio
from infer.lib.slicer2 import Slicer


class PreProcess:
    def __init__(
        self, sr: int, exp_dir: Path, per: float = 3.7, log_file: TextIO | None = None
    ):
        self.sr = sr
        self.per = per
        self.overlap = 0.3
        self.tail = self.per + self.overlap
        self.max = 0.9
        self.alpha = 0.75
        self.exp_dir = exp_dir
        self.log_file = log_file
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
        self.bh, self.ah = signal.butter(N=5, Wn=48, btype="high", fs=self.sr)  # type: ignore

        self.exp_dir.mkdir(exist_ok=True)
        self.gt_wavs_dir.mkdir(exist_ok=True)
        self.wavs16k_dir.mkdir(exist_ok=True)

    def log(self, message: str) -> None:
        """Print message to console and optionally write to log file."""
        print(message)
        if self.log_file is not None:
            self.log_file.write(f"{message}\n")
            self.log_file.flush()

    def norm_write(self, tmp_audio: np.ndarray, idx0: int, idx1: int) -> None:
        """Normalize and write audio to both sample rate directories."""
        tmp_max = np.abs(tmp_audio).max()
        if tmp_max > 2.5:
            print(f"{idx0}-{idx1}-{tmp_max}-filtered")
            return

        # Normalize audio
        tmp_audio = (tmp_audio / tmp_max * (self.max * self.alpha)) + (
            1 - self.alpha
        ) * tmp_audio

        # Write at original sample rate
        gt_path = self.gt_wavs_dir / f"{idx0}_{idx1}.wav"
        wavfile.write(str(gt_path), self.sr, tmp_audio.astype(np.float32))

        # Resample and write at 16kHz
        tmp_audio_16k = librosa.resample(tmp_audio, orig_sr=self.sr, target_sr=16000)
        wavs16k_path = self.wavs16k_dir / f"{idx0}_{idx1}.wav"
        wavfile.write(str(wavs16k_path), 16000, tmp_audio_16k.astype(np.float32))

    def pipeline(self, path: Path, idx0: int):
        """Process a single audio file: load, filter, slice, and write segments."""
        try:
            audio = load_audio(str(path), self.sr)
            audio = signal.lfilter(self.bh, self.ah, audio)

            idx1 = 0
            for audio_slice in self.slicer.slice(audio):
                idx1 = self._process_audio_slice(audio_slice, idx0, idx1)

            self.log(f"{path}\t-> Success")
        except Exception:
            self.log(f"{path}\t-> {traceback.format_exc()}")

    def _process_audio_slice(
        self, audio_slice: np.ndarray, idx0: int, idx1: int
    ) -> int:
        """Process a single audio slice by splitting it into overlapping segments."""
        segment_step = int(self.sr * (self.per - self.overlap))
        segment_length = int(self.per * self.sr)
        tail_length = int(self.tail * self.sr)

        i = 0
        while True:
            start = segment_step * i
            remaining = audio_slice[start:]

            if len(remaining) > tail_length:
                # Extract full segment
                tmp_audio = remaining[:segment_length]
            else:
                # Last segment
                tmp_audio = remaining
                self.norm_write(tmp_audio, idx0, idx1)
                idx1 += 1
                break

            self.norm_write(tmp_audio, idx0, idx1)
            idx1 += 1
            i += 1

        return idx1

    def pipeline_mp(self, infos: list[tuple[Path, int]]) -> None:
        """Process multiple audio files sequentially."""
        for path, idx0 in infos:
            self.pipeline(path, idx0)

    def pipeline_mp_inp_dir(self, inp_root: Path, n_p: int, no_parallel: bool) -> None:
        """Process all audio files in a directory, optionally in parallel."""
        try:
            infos = [(path, idx) for idx, path in enumerate(sorted(inp_root.iterdir()))]

            if no_parallel:
                # Sequential processing
                for i in range(n_p):
                    self.pipeline_mp(infos[i::n_p])
            else:
                # Parallel processing
                processes = []
                for i in range(n_p):
                    p = multiprocessing.Process(
                        target=self.pipeline_mp, args=(infos[i::n_p],)
                    )
                    processes.append(p)
                    p.start()

                for p in processes:
                    p.join()
        except Exception:
            self.log(f"Fail. {traceback.format_exc()}")


def preprocess_trainset(
    inp_root: Path,
    sr: int,
    n_p: int,
    exp_dir: Path,
    per: float,
    no_parallel: bool = False,
):
    """Preprocess audio training dataset with logging."""
    log_path = exp_dir / "preprocess.log"

    with log_path.open("a+") as log_file:
        pp = PreProcess(sr, exp_dir, per, log_file)
        pp.log("start preprocess")
        pp.pipeline_mp_inp_dir(inp_root, n_p, no_parallel)
        pp.log("end preprocess")


if __name__ == "__main__":
    if len(sys.argv) < 7:
        print(
            "Usage: python preprocess.py <inp_root> <sr> <n_p> <exp_dir> <no_parallel> <per>"
        )
        sys.exit(1)

    inp_root = Path(sys.argv[1])
    sr = int(sys.argv[2])
    n_p = int(sys.argv[3])
    exp_dir = Path(sys.argv[4])
    no_parallel = sys.argv[5] == "True"
    per = float(sys.argv[6])

    preprocess_trainset(inp_root, sr, n_p, exp_dir, per, no_parallel)
