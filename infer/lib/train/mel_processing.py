import torch
from librosa.filters import mel as librosa_mel_fn


def dynamic_range_compression_torch(
    x: torch.Tensor,
    C: int = 1,
    clip_val: float = 1e-5,
):
    """
    PARAMS
    ------
    C: compression factor
    """
    return torch.log(torch.clamp(x, min=clip_val) * C)


def spectral_normalize_torch(magnitudes: torch.Tensor) -> torch.Tensor:
    return dynamic_range_compression_torch(magnitudes)


# Reusable banks
_mel_basis: dict[str, torch.Tensor] = {}
_hann_window: dict[str, torch.Tensor] = {}


def spectrogram_torch(
    y: torch.Tensor,
    n_fft: int,
    sampling_rate: int,
    hop_size: int,
    win_size: int,
    center: bool = False,
) -> torch.Tensor:
    """Convert waveform into Linear-frequency Linear-amplitude spectrogram.

    Args:
        y             :: (B, T) - Audio waveforms
        n_fft
        sampling_rate
        hop_size
        win_size
        center
    Returns:
        :: (B, Freq, Frame) - Linear-frequency Linear-amplitude spectrogram
    """

    # Window - Cache if needed
    global _hann_window
    dtype_device = str(y.dtype) + "_" + str(y.device)
    wnsize_dtype_device = str(win_size) + "_" + dtype_device
    if wnsize_dtype_device not in _hann_window:
        _hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(
            dtype=y.dtype, device=y.device
        )

    # Padding
    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect",
    )
    y = y.squeeze(1)

    # Complex Spectrogram :: (B, T) -> (B, Freq, Frame, RealComplex=2)
    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=_hann_window[wnsize_dtype_device],
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )

    # Linear-frequency Linear-amplitude spectrogram :: (B, Freq, Frame, RealComplex=2) -> (B, Freq, Frame)
    spec = torch.sqrt(spec.real.pow(2) + spec.imag.pow(2) + 1e-6)
    return spec


def spec_to_mel_torch(
    spec: torch.Tensor,
    n_fft: int,
    num_mels: int,
    sampling_rate: int,
    fmin: float,
    fmax: float,
) -> torch.Tensor:
    # MelBasis - Cache if needed
    global _mel_basis
    dtype_device = str(spec.dtype) + "_" + str(spec.device)
    fmax_dtype_device = str(fmax) + "_" + dtype_device
    if fmax_dtype_device not in _mel_basis:
        mel = librosa_mel_fn(
            sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax
        )
        mel_tensor = torch.from_numpy(mel)  # pyright: ignore[reportUnknownMemberType]
        _mel_basis[fmax_dtype_device] = mel_tensor.to(
            dtype=spec.dtype, device=spec.device
        )

    # Mel-frequency Log-amplitude spectrogram :: (B, Freq=num_mels, Frame)
    melspec = torch.matmul(_mel_basis[fmax_dtype_device], spec)
    melspec = spectral_normalize_torch(melspec)
    return melspec


def mel_spectrogram_torch(
    y: torch.Tensor,
    n_fft: int,
    num_mels: int,
    sampling_rate: int,
    hop_size: int,
    win_size: int,
    fmin: float,
    fmax: float,
    center: bool = False,
):
    """Convert waveform into Mel-frequency Log-amplitude spectrogram.

    Args:
        y       :: (B, T)           - Waveforms
    Returns:
        melspec :: (B, Freq, Frame) - Mel-frequency Log-amplitude spectrogram
    """
    # Linear-frequency Linear-amplitude spectrogram :: (B, T) -> (B, Freq, Frame)
    spec = spectrogram_torch(y, n_fft, sampling_rate, hop_size, win_size, center)

    # Mel-frequency Log-amplitude spectrogram :: (B, Freq, Frame) -> (B, Freq=num_mels, Frame)
    melspec = spec_to_mel_torch(spec, n_fft, num_mels, sampling_rate, fmin, fmax)

    return melspec
