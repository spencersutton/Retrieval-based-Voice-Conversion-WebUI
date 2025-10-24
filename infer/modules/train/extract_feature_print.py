import os
import sys
import traceback

import fairseq
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from fairseq.data.dictionary import Dictionary
from torch.serialization import safe_globals

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

_device = sys.argv[1]
_n_part = int(sys.argv[2])
_i_part = int(sys.argv[3])
if len(sys.argv) == 6:
    _exp_dir = sys.argv[4]
    _is_half = sys.argv[5].lower() == "true"
else:
    _i_gpu = sys.argv[4]
    _exp_dir = sys.argv[5]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_i_gpu)
    _is_half = sys.argv[6].lower() == "true"
_device = "cpu"
if torch.cuda.is_available():
    _device = "cuda"
elif torch.backends.mps.is_available():
    _device = "mps"

_f = open("%s/extract_f0_feature.log" % _exp_dir, "a+")


def printt(strr):
    print(strr)
    _f.write("%s\n" % strr)
    _f.flush()


printt(" ".join(sys.argv))
_model_path = "assets/hubert/hubert_base.pt"

printt("exp_dir: " + _exp_dir)
_wavPath = "%s/1_16k_wavs" % _exp_dir
_outPath = "%s/3_feature768" % _exp_dir
os.makedirs(_outPath, exist_ok=True)


# wave must be 16k, hop_size=320
def readwave(wav_path, normalize=False):
    wav, sr = sf.read(wav_path)
    assert sr == 16000
    feats = torch.from_numpy(wav).float()
    if feats.dim() == 2:  # double channels
        feats = feats.mean(-1)
    assert feats.dim() == 1, feats.dim()
    if normalize:
        with torch.no_grad():
            feats = F.layer_norm(feats, feats.shape)
    feats = feats.view(1, -1)
    return feats


# HuBERT model
printt("load model(s) from {}".format(_model_path))
# if hubert model is exist
if not os.access(_model_path, os.F_OK):
    printt(
        "Error: Extracting is shut down because %s does not exist, you may download it from https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main"
        % _model_path
    )
    exit(0)
with safe_globals([Dictionary]):
    models, saved_cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
        [_model_path],
        suffix="",
    )
_model = models[0]
_model = _model.to(_device)
printt("move model to %s" % _device)
if _is_half:
    if _device not in ["mps", "cpu"]:
        _model = _model.half()
_model.eval()

_todo = sorted(list(os.listdir(_wavPath)))[_i_part::_n_part]
_n = max(1, len(_todo) // 10)  # 最多打印十条
if len(_todo) == 0:
    printt("no-feature-todo")
else:
    printt("all-feature-%s" % len(_todo))
    for idx, file in enumerate(_todo):
        try:
            if file.endswith(".wav"):
                wav_path = "%s/%s" % (_wavPath, file)
                out_path = "%s/%s" % (_outPath, file.replace("wav", "npy"))

                if os.path.exists(out_path):
                    continue

                feats = readwave(wav_path, normalize=saved_cfg.task.normalize)
                padding_mask = torch.BoolTensor(feats.shape).fill_(False)
                inputs = {
                    "source": (
                        feats.half().to(_device)
                        if _is_half and _device not in ["mps", "cpu"]
                        else feats.to(_device)
                    ),
                    "padding_mask": padding_mask.to(_device),
                    "output_layer": 12,  # layer 9
                }
                with torch.no_grad():
                    logits = _model.extract_features(**inputs)
                    feats = logits[0]

                feats = feats.squeeze(0).float().cpu().numpy()
                if np.isnan(feats).sum() == 0:
                    np.save(out_path, feats, allow_pickle=False)
                else:
                    printt("%s-contains nan" % file)
                if idx % _n == 0:
                    printt("now-%s,all-%s,%s,%s" % (len(_todo), idx, file, feats.shape))
        except Exception:
            printt(traceback.format_exc())
    printt("all-feature-done")
