import math
from collections.abc import Iterable

import torch
from torch.nn import functional as F


def init_weights(m: torch.nn.Module, mean: float = 0.0, std: float = 0.01) -> None:
    classname = m.__class__.__name__
    if (
        classname.find("Conv") != -1
        and hasattr(m, "weight")
        and isinstance(m.weight, torch.Tensor)
    ):
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    return int((kernel_size * dilation - dilation) / 2)


def kl_divergence(
    m_p: torch.Tensor, logs_p: torch.Tensor, m_q: torch.Tensor, logs_q: torch.Tensor
) -> torch.Tensor:
    """KL(P||Q)"""
    kl = (logs_q - logs_p) - 0.5
    kl += (
        0.5 * (torch.exp(2.0 * logs_p) + ((m_p - m_q) ** 2)) * torch.exp(-2.0 * logs_q)
    )
    return kl


def rand_gumbel(shape: torch.Size | tuple[int, ...]) -> torch.Tensor:
    """Sample from the Gumbel distribution, protect from overflows."""
    uniform_samples = torch.rand(shape) * 0.99998 + 0.00001
    return -torch.log(-torch.log(uniform_samples))


def rand_gumbel_like(x: torch.Tensor) -> torch.Tensor:
    g = rand_gumbel(x.size()).to(dtype=x.dtype, device=x.device)
    return g


def slice_segments(
    x: torch.Tensor, ids_str: torch.Tensor, segment_size: int = 4
) -> torch.Tensor:
    ret = torch.zeros_like(x[:, :, :segment_size])
    for i in range(x.size(0)):
        idx_str = ids_str[i]
        idx_end = idx_str + segment_size
        ret[i] = x[i, :, idx_str:idx_end]
    return ret


def slice_segments2(
    x: torch.Tensor, ids_str: torch.Tensor, segment_size: int = 4
) -> torch.Tensor:
    ret = torch.zeros_like(x[:, :segment_size])
    for i in range(x.size(0)):
        idx_str = ids_str[i]
        idx_end = idx_str + segment_size
        ret[i] = x[i, idx_str:idx_end]
    return ret


def rand_slice_segments(
    x: torch.Tensor, x_lengths: int | torch.Tensor | None = None, segment_size: int = 4
) -> tuple[torch.Tensor, torch.Tensor]:
    b, _d, t = x.size()
    if x_lengths is None:
        x_lengths = t
    ids_str_max = x_lengths - segment_size + 1
    ids_str = (torch.rand([b]).to(device=x.device) * ids_str_max).to(dtype=torch.long)
    ret = slice_segments(x, ids_str, segment_size)
    return ret, ids_str


def get_timing_signal_1d(
    length: int, channels: int, min_timescale: float = 1.0, max_timescale: float = 1.0e4
) -> torch.Tensor:
    position = torch.arange(length, dtype=torch.float)
    num_timescales = channels // 2
    log_timescale_increment = math.log(float(max_timescale) / float(min_timescale)) / (
        num_timescales - 1
    )
    inv_timescales = min_timescale * torch.exp(
        torch.arange(num_timescales, dtype=torch.float) * -log_timescale_increment
    )
    scaled_time = position.unsqueeze(0) * inv_timescales.unsqueeze(1)
    signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], 0)
    signal = F.pad(signal, [0, 0, 0, channels % 2])
    signal = signal.view(1, channels, length)
    return signal


def add_timing_signal_1d(
    x: torch.Tensor, min_timescale: float = 1.0, max_timescale: float = 1.0e4
) -> torch.Tensor:
    _b, channels, length = x.size()
    signal = get_timing_signal_1d(length, channels, min_timescale, max_timescale)
    return x + signal.to(dtype=x.dtype, device=x.device)


def cat_timing_signal_1d(
    x: torch.Tensor,
    min_timescale: float = 1.0,
    max_timescale: float = 1.0e4,
    axis: int = 1,
) -> torch.Tensor:
    _b, channels, length = x.size()
    signal = get_timing_signal_1d(length, channels, min_timescale, max_timescale)
    return torch.cat([x, signal.to(dtype=x.dtype, device=x.device)], axis)


def subsequent_mask(length: int) -> torch.Tensor:
    mask = torch.tril(torch.ones(length, length)).unsqueeze(0).unsqueeze(0)
    return mask


@torch.jit.script
def fused_add_tanh_sigmoid_multiply(
    input_a: torch.Tensor, input_b: torch.Tensor, n_channels: torch.Tensor
) -> torch.Tensor:
    n_channels_int = n_channels[0]
    in_act = input_a + input_b
    t_act = torch.tanh(in_act[:, :n_channels_int, :])
    s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
    acts = t_act * s_act
    return acts


def convert_pad_shape(pad_shape: list[list[int]]) -> list[int]:
    return torch.tensor(pad_shape).flip(0).reshape(-1).int().tolist()


def shift_1d(x: torch.Tensor) -> torch.Tensor:
    x = F.pad(x, convert_pad_shape([[0, 0], [0, 0], [1, 0]]))[:, :, :-1]
    return x


def sequence_mask(
    length: torch.Tensor, max_length: int | torch.Tensor | None = None
) -> torch.Tensor:
    if max_length is None:
        max_length = int(length.max().item())
    if isinstance(max_length, torch.Tensor):
        max_length = int(max_length.item())
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


def generate_path(duration: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    duration: [b, 1, t_x]
    mask: [b, 1, t_y, t_x]
    """

    b, _, t_y, t_x = mask.shape
    cum_duration = torch.cumsum(duration, -1)

    cum_duration_flat = cum_duration.view(b * t_x)
    path = sequence_mask(cum_duration_flat, t_y).to(mask.dtype)
    path = path.view(b, t_x, t_y)
    path = path - F.pad(path, convert_pad_shape([[0, 0], [1, 0], [0, 0]]))[:, :-1]
    path = path.unsqueeze(1).transpose(2, 3) * mask
    return path


def clip_grad_value_(
    parameters: Iterable[torch.Tensor] | torch.Tensor,
    clip_value: float | None,
    norm_type: int | float = 2,
) -> float:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    if clip_value is not None:
        clip_value = float(clip_value)

    total_norm = 0.0
    for p in parameters:
        if p.grad is not None:
            param_norm = p.grad.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
            if clip_value is not None:
                p.grad.clamp_(min=-clip_value, max=clip_value)
    total_norm = total_norm ** (1.0 / norm_type)
    return total_norm
