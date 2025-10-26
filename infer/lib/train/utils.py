from dataclasses import dataclass
from pathlib import Path


@dataclass
class HParamsData:
    filter_length: int
    hop_length: int
    max_wav_value: float
    mel_fmax: int
    mel_fmin: int
    n_mel_channels: int
    sampling_rate: int
    win_length: int
    training_files: str | None = None


@dataclass
class HParamsTrain:
    batch_size: int
    betas: tuple[float, float]
    c_kl: float
    c_mel: float
    epochs: int
    eps: float
    fp16_run: bool
    init_lr_ratio: int
    learning_rate: float
    log_interval: int
    lr_decay: float
    seed: int
    segment_size: int
    warmup_epochs: int


@dataclass
class HParamsModel:
    filter_channels: int
    gin_channels: int
    hidden_channels: int
    inter_channels: int
    kernel_size: int
    n_heads: int
    n_layers: int
    p_dropout: float
    resblock_dilation_sizes: list[list[int]]
    resblock_kernel_sizes: list[int]
    resblock: str
    spk_embed_dim: int
    upsample_initial_channel: int
    upsample_kernel_sizes: list[int]
    upsample_rates: list[int]
    use_spectral_norm: bool


@dataclass
class HParams:
    data: HParamsData
    model: HParamsModel
    train: HParamsTrain

    experiment_dir: Path = Path()
    gpus: str = ""
    if_cache_data_in_gpu: int = 0
    if_f0: bool = False
    if_latest: int = 0
    model_dir: Path = Path()
    name: str = ""
    pretrainD: str = ""
    pretrainG: str = ""
    sample_rate: int = 0
    save_every_epoch: bool = False
    save_every_weights: str = "0"
    total_epoch: int = 0
