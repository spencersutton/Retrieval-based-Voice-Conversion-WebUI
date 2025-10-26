import logging
import sys
import traceback
from multiprocessing import Process
from pathlib import Path

import numpy as np
import parselmouth
import pyworld

from infer.lib.audio import load_audio

logging.getLogger("numba").setLevel(logging.WARNING)

exp_dir = Path(sys.argv[1])
f = (exp_dir / "extract_f0_feature.log").open("a+")


def printt(strr: str) -> None:
    print(strr)
    f.write(f"{strr}\n")
    f.flush()


n_p = int(sys.argv[2])
f0method = sys.argv[3]


class FeatureInput:
    def __init__(self, sample_rate: int = 16000, hop_size: int = 160) -> None:
        self.fs = sample_rate
        self.hop = hop_size

        self.f0_bin = 256
        self.f0_max = 1100.0
        self.f0_min = 50.0
        self.f0_mel_min = 1127 * np.log(1 + self.f0_min / 700)
        self.f0_mel_max = 1127 * np.log(1 + self.f0_max / 700)

    def compute_f0(self, path: Path, f0_method: str) -> np.ndarray:
        x = load_audio(path, self.fs)
        p_len = x.shape[0] // self.hop
        if f0_method == "pm":
            time_step = 160 / 16000 * 1000
            f0_min = 50
            f0_max = 1100
            f0 = (
                parselmouth.Sound(x, self.fs)
                .to_pitch_ac(
                    time_step=time_step / 1000,
                    voicing_threshold=0.6,
                    pitch_floor=f0_min,
                    pitch_ceiling=f0_max,
                )
                .selected_array["frequency"]
            )
            pad_size = (p_len - len(f0) + 1) // 2
            if pad_size > 0 or p_len - len(f0) - pad_size > 0:
                f0 = np.pad(
                    f0, [[pad_size, p_len - len(f0) - pad_size]], mode="constant"
                )
        elif f0_method == "harvest":
            f0, t = pyworld.harvest(  # type: ignore
                x.astype(np.double),
                fs=self.fs,
                f0_ceil=self.f0_max,
                f0_floor=self.f0_min,
                frame_period=1000 * self.hop / self.fs,
            )
            f0 = pyworld.stonemask(x.astype(np.double), f0, t, self.fs)  # type: ignore
        elif f0_method == "dio":
            f0, t = pyworld.dio(
                x.astype(np.double),
                fs=self.fs,
                f0_ceil=self.f0_max,
                f0_floor=self.f0_min,
                frame_period=1000 * self.hop / self.fs,
            )
            f0 = pyworld.stonemask(x.astype(np.double), f0, t, self.fs)
        elif f0_method == "rmvpe":
            if not hasattr(self, "model_rmvpe"):
                from infer.lib.rmvpe import RMVPE

                print("Loading rmvpe model")
                self.model_rmvpe = RMVPE(
                    "assets/rmvpe/rmvpe.pt", is_half=False, device="cpu"
                )
            f0 = self.model_rmvpe.infer_from_audio(x, thred=0.03)
        return f0

    def coarse_f0(self, f0: np.ndarray) -> np.ndarray:
        f0_mel = 1127 * np.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - self.f0_mel_min) * (
            self.f0_bin - 2
        ) / (self.f0_mel_max - self.f0_mel_min) + 1

        # use 0 or 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > self.f0_bin - 1] = self.f0_bin - 1
        f0_coarse = np.rint(f0_mel).astype(int)
        assert f0_coarse.max() <= 255 and f0_coarse.min() >= 1, (
            f0_coarse.max(),
            f0_coarse.min(),
        )
        return f0_coarse

    def go(self, paths: list[tuple[Path, Path, Path]], f0_method: str) -> None:
        if len(paths) == 0:
            printt("no-f0-todo")
        else:
            printt(f"todo-f0-{len(paths)}")
            n = max(len(paths) // 5, 1)  # Each process prints at most 5 messages
            for idx, (inp_path, opt_path1, opt_path2) in enumerate(paths):
                try:
                    if idx % n == 0:
                        printt(f"f0ing,now-{idx},all-{len(paths)},-{inp_path}")
                    if (
                        opt_path1.with_suffix(".npy").exists()
                        and opt_path2.with_suffix(".npy").exists()
                    ):
                        continue
                    featur_pit = self.compute_f0(inp_path, f0_method)
                    np.save(
                        opt_path2,
                        featur_pit,
                        allow_pickle=False,
                    )  # nsf
                    coarse_pit = self.coarse_f0(featur_pit)
                    np.save(
                        opt_path1,
                        coarse_pit,
                        allow_pickle=False,
                    )  # ori
                except Exception:
                    printt(f"f0fail-{idx}-{inp_path}-{traceback.format_exc()}")


if __name__ == "__main__":
    printt(" ".join(sys.argv))
    featureInput = FeatureInput()
    paths: list[tuple[Path, Path, Path]] = []
    inp_root = exp_dir / "1_16k_wavs"
    opt_root1 = exp_dir / "2a_f0"
    opt_root2 = exp_dir / "2b-f0nsf"

    opt_root1.mkdir(parents=True, exist_ok=True)
    opt_root2.mkdir(parents=True, exist_ok=True)
    for inp_path_obj in sorted(inp_root.iterdir()):
        name = inp_path_obj.name
        if "spec" in name:
            continue
        opt_path1 = opt_root1 / name
        opt_path2 = opt_root2 / name
        paths.append((inp_path_obj, opt_path1, opt_path2))

    ps = []
    for i in range(n_p):
        p = Process(
            target=featureInput.go,
            args=(
                paths[i::n_p],
                f0method,
            ),
        )
        ps.append(p)
        p.start()
    for i in range(n_p):
        ps[i].join()
