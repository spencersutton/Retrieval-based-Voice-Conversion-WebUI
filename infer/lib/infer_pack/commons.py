import torch


def init_weights(
    m: torch.nn.Module,
    mean: float = 0.0,
    std: float = 0.01,
):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        assert isinstance(m.weight.data, torch.Tensor)
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size: int, dilation: int = 1):
    return int((kernel_size * dilation - dilation) / 2)


def slice_segments(
    x: torch.Tensor,
    ids_str: torch.Tensor,
    segment_size: int = 4,
) -> torch.Tensor:
    ret = torch.zeros_like(x[:, :, :segment_size])
    for i in range(x.size(0)):
        idx_str = ids_str[i]
        idx_end = idx_str + segment_size
        ret[i] = x[i, :, idx_str:idx_end]
    return ret


def slice_segments2(
    x: torch.Tensor,
    ids_str: torch.Tensor,
    segment_size: int = 4,
) -> torch.Tensor:
    ret = torch.zeros_like(x[:, :segment_size])
    for i in range(x.size(0)):
        idx_str = ids_str[i]
        idx_end = idx_str + segment_size
        ret[i] = x[i, idx_str:idx_end]
    return ret


def rand_slice_segments(
    x: torch.Tensor,
    x_lengths: int | None = None,
    segment_size: int = 4,
):
    b, _d, t = x.size()
    if x_lengths is None:
        x_lengths = t
    ids_str_max = x_lengths - segment_size + 1
    ids_str = (torch.rand([b]).to(device=x.device) * ids_str_max).to(dtype=torch.long)
    ret = slice_segments(x, ids_str, segment_size)
    return ret, ids_str


def subsequent_mask(length: int):
    mask = torch.tril(torch.ones(length, length)).unsqueeze(0).unsqueeze(0)
    return mask


@torch.jit.script  # type: ignore
def fused_add_tanh_sigmoid_multiply(
    input_a: torch.Tensor,
    input_b: torch.Tensor,
    n_channels: torch.Tensor,
):
    n_channels_int = n_channels[0]
    in_act = input_a + input_b
    t_act = torch.tanh(in_act[:, :n_channels_int, :])
    s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
    acts = t_act * s_act
    return acts


def sequence_mask(
    length: torch.Tensor,
    max_length: int | None = None,
) -> torch.Tensor:
    if max_length is None:
        max_length = int(length.max())
    assert max_length is not None
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)
