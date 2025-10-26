import multiprocessing
import sys
import traceback
from io import TextIOWrapper
from pathlib import Path

import librosa
import numpy as np
from scipy import signal
from scipy.io import wavfile

from infer.lib.audio import load_audio
from infer.lib.slicer2 import Slicer

_input_root: Path = Path()
_sample_rate: int = 0
_num_processes: int = 0
_exp_dir: Path = Path()
_no_parallel: bool = False
_per: float = 0.0
_f: TextIOWrapper | None = None


def _println(strr: str) -> None:
    print(strr)
    assert _f is not None
    _f.write(f"{strr}\n")
    _f.flush()


class _PreProcess:
    bh: np.ndarray
    ah: np.ndarray

    def __init__(self, sample_rate: int, exp_dir: Path, per: float = 3.7) -> None:
        self.slicer = Slicer(
            sample_rate=sample_rate,
            threshold=-42,
            min_length=1500,
            min_interval=400,
            hop_size=15,
            max_sil_kept=500,
        )
        self.sr = sample_rate
        self.bh, self.ah = signal.butter(N=5, Wn=48, btype="high", fs=self.sr)  # type: ignore
        self.per = per
        self.overlap = 0.3
        self.tail = self.per + self.overlap
        self.max = 0.9
        self.alpha = 0.75
        self.exp_dir = exp_dir
        self.gt_wavs_dir = exp_dir / "0_gt_wavs"
        self.wavs16k_dir = exp_dir / "1_16k_wavs"
        self.exp_dir.mkdir(exist_ok=True, parents=True)
        self.gt_wavs_dir.mkdir(exist_ok=True, parents=True)
        self.wavs16k_dir.mkdir(exist_ok=True, parents=True)

    def norm_write(self, audio: np.ndarray, idx0: int, idx1: int) -> None:
        max_val = np.abs(audio).max()
        if max_val > 2.5:
            print(f"{idx0}-{idx1}-{max_val}-filtered")
            return

        # Normalize audio
        normalized_audio = (audio / max_val * (self.max * self.alpha)) + (
            1 - self.alpha
        ) * audio

        # Save ground truth wav
        gt_wav_path = self.gt_wavs_dir / f"{idx0}_{idx1}.wav"
        wavfile.write(str(gt_wav_path), self.sr, normalized_audio.astype(np.float32))

        # Resample and save 16k wav
        audio_16k = librosa.resample(normalized_audio, orig_sr=self.sr, target_sr=16000)
        wav16k_path = self.wavs16k_dir / f"{idx0}_{idx1}.wav"
        wavfile.write(str(wav16k_path), 16000, audio_16k.astype(np.float32))

    def pipeline(self, path: str, idx0: int) -> None:
        try:
            audio = load_audio(path, self.sr)
            audio = signal.lfilter(self.bh, self.ah, audio)

            idx1 = 0
            for sliced_audio in self.slicer.slice(audio):
                start = 0
                length = len(sliced_audio)
                per_samples = int(self.per * self.sr)
                overlap_samples = int(self.overlap * self.sr)
                tail_samples = int(self.tail * self.sr)

                while start < length:
                    end = start + per_samples
                    if length - start > tail_samples:
                        segment = sliced_audio[start:end]
                        self.norm_write(segment, idx0, idx1)
                        idx1 += 1
                        start += per_samples - overlap_samples
                    else:
                        segment = sliced_audio[start:]
                        self.norm_write(segment, idx0, idx1)
                        idx1 += 1
                        break
            _println(f"{path}\t-> Success")
        except Exception:
            _println(f"{path}\t-> {traceback.format_exc()}")

    def pipeline_mp(self, infos: list[tuple[str, int]]) -> None:
        for path, idx0 in infos:
            self.pipeline(path, idx0)

    def pipeline_mp_inp_dir(self, input_root: Path, num_processes: int) -> None:
        try:
            infos = [
                (str(path), idx)
                for idx, path in enumerate(sorted(input_root.iterdir()))
            ]
            if _no_parallel:
                # Run sequentially, split work by process count for consistency
                for chunk in (infos[i::num_processes] for i in range(num_processes)):
                    self.pipeline_mp(chunk)
            else:
                # Run in parallel using multiprocessing
                processes = [
                    multiprocessing.Process(
                        target=self.pipeline_mp, args=(infos[i::num_processes],)
                    )
                    for i in range(num_processes)
                ]
                for p in processes:
                    p.start()
                for p in processes:
                    p.join()
        except Exception:
            _println(f"Fail. {traceback.format_exc()}")


def _preprocess_trainset(
    input_root: Path,
    sample_rate: int,
    num_processes: int,
    exp_dir: Path,
    per: float,
) -> None:
    pp = _PreProcess(sample_rate, exp_dir, per)
    _println("start preprocess")
    pp.pipeline_mp_inp_dir(input_root, num_processes)
    _println("end preprocess")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        print(*sys.argv[1:])
        _input_root = Path(sys.argv[1])
        _sample_rate = int(sys.argv[2])
        _num_processes = int(sys.argv[3])
        _exp_dir = Path(sys.argv[4])
        _no_parallel = sys.argv[5] == "True"
        _per = float(sys.argv[6])
        _f = (_exp_dir / "preprocess.log").open("a+", encoding="utf-8")

    _preprocess_trainset(
        _input_root,
        _sample_rate,
        _num_processes,
        _exp_dir,
        _per,
    )
