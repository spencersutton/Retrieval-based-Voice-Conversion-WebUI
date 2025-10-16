import os
from typing import List, Tuple, Union

from fairseq import checkpoint_utils
from configs.config import Config
from fairseq.models.hubert.hubert import HubertModel
import shared


def get_index_path_from_model(sid: str) -> str:
    return next(
        (
            f
            for f in [
                os.path.join(root, name)
                for root, _, files in os.walk(shared.index_root, topdown=False)
                for name in files
                if name.endswith(".index") and "trained" not in name
            ]
            if sid.split(".")[0] in f
        ),
        "",
    )


def load_hubert(config: Config) -> HubertModel:  # hubert_model is a torch.nn.Module
    models: List[HubertModel]

    models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
        ["assets/hubert/hubert_base.pt"],
        suffix="",
    )
    hubert_model = models[0]
    hubert_model = hubert_model.to(config.device)
    if config.is_half:
        try:
            hubert_model = hubert_model.half()
        except Exception as e:
            print(
                "Warning: could not convert HuBERT to half — keeping float32. Error:",
                e,
            )
            hubert_model = hubert_model.float()
    else:
        hubert_model = hubert_model.float()
    return hubert_model.eval()
