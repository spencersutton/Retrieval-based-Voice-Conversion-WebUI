import os
import sys
import traceback
from pathlib import Path

import numpy as np

from infer.lib.audio import load_audio
from infer.lib.rmvpe import RMVPE

now_dir = Path.cwd()
sys.path.append(str(now_dir))

n_part = int(sys.argv[1])
i_part = int(sys.argv[2])
i_gpu = sys.argv[3]

exp_dir = Path(sys.argv[4])
is_half = sys.argv[5]
os.environ["CUDA_VISIBLE_DEVICES"] = str(i_gpu)
f = (exp_dir / "extract_f0_feature.log").open("a+")


def printt(strr: str):
    print(strr)
    f.write(f"{strr}\n")
    f.flush()


class FeatureInput:
    def __init__(self, samplerate: int = 16000, hop_size: int = 160):
        self.fs = samplerate
        self.hop = hop_size

        self.f0_bin = 256
        self.f0_max = 1100.0
        self.f0_min = 50.0
        self.f0_mel_min = 1127 * np.log(1 + self.f0_min / 700)
        self.f0_mel_max = 1127 * np.log(1 + self.f0_max / 700)

    def compute_f0(self, path: Path, f0_method: str):
        x = load_audio(str(path), self.fs)
        f0 = None
        if f0_method == "rmvpe":
            if not hasattr(self, "model_rmvpe"):
                print("Loading rmvpe model")
                self.model_rmvpe = RMVPE(
                    "assets/rmvpe/rmvpe.pt", is_half=is_half, device="cuda"
                )
            f0 = self.model_rmvpe.infer_from_audio(x, thred=0.03)
        return f0

    def coarse_f0(self, f0: np.ndarray):
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

    def go(self, paths: list[tuple[Path, Path, Path]], f0_method: str):
        if len(paths) == 0:
            printt("no-f0-todo")
        else:
            printt(f"todo-f0-{len(paths)}")
            n = max(len(paths) // 5, 1)  # 每个进程最多打印5条
            for idx, (inp_path, opt_path1, opt_path2) in enumerate(paths):
                try:
                    if idx % n == 0:
                        printt(f"f0ing,now-{idx},all-{len(paths)},-{inp_path}")
                    if (
                        Path(opt_path1).with_suffix(".npy").exists()
                        and Path(opt_path2).with_suffix(".npy").exists()
                    ):
                        continue
                    featur_pit = self.compute_f0(inp_path, f0_method)
                    if featur_pit is None:
                        printt(f"f0fail-{idx}-{inp_path}-f0 extraction returned None")
                        continue
                    np.save(opt_path2, featur_pit, allow_pickle=False)  # nsf
                    coarse_pit = self.coarse_f0(featur_pit)
                    np.save(opt_path1, coarse_pit, allow_pickle=False)  # ori
                except Exception:
                    printt(f"f0fail-{idx}-{inp_path}-{traceback.format_exc()}")


if __name__ == "__main__":
    printt(" ".join(sys.argv))
    featureInput = FeatureInput()
    paths = []
    inp_root = exp_dir / "1_16k_wavs"
    opt_root1 = exp_dir / "2a_f0"
    opt_root2 = exp_dir / "2b-f0nsf"

    opt_root1.mkdir(parents=True, exist_ok=True)
    opt_root2.mkdir(parents=True, exist_ok=True)
    for name in sorted(inp_root.iterdir()):
        if "spec" in str(name):
            continue
        paths.append([name, opt_root1 / name.name, opt_root2 / name.name])
    try:
        featureInput.go(paths[i_part::n_part], "rmvpe")
    except Exception:
        printt(f"f0_all_fail-{traceback.format_exc()}")
