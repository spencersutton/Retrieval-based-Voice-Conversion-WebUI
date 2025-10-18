"""Type stubs for pyworld

PyWORLD is a Python wrapper for WORLD, a high-quality speech analysis,
manipulation and synthesis system. For more info see:
https://github.com/mmorise/World
"""

from typing import Tuple

import numpy as np

__version__: str

def harvest(
    x: np.ndarray,
    fs: int,
    f0_floor: float = 71.0,
    f0_ceil: float = 800.0,
    frame_period: float = 5.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract F0 using Harvest algorithm.

    Args:
        x: Input audio signal as a 1D numpy array (numpy.ndarray[Any, numpy.dtype[numpy.floating[Any]]])
        fs: Sampling frequency in Hz (int)
        f0_floor: Minimum fundamental frequency to be detected (float, default: 71.0)
        f0_ceil: Maximum fundamental frequency to be detected (float, default: 800.0)
        frame_period: Frame length in milliseconds (float, default: 5.0)

    Returns:
        f0: Extracted F0 trajectory (numpy.ndarray[Any, numpy.dtype[numpy.floating[Any]]])
        t: Time axis in seconds (numpy.ndarray[Any, numpy.dtype[numpy.floating[Any]]])
    """

def dio(
    x: np.ndarray,
    fs: int,
    f0_floor: float = 71.0,
    f0_ceil: float = 800.0,
    frame_period: float = 5.0,
    allowed_range: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract F0 using DIO algorithm.

    Args:
        x: Input audio signal as a 1D numpy array (numpy.ndarray[Any, numpy.dtype[numpy.floating[Any]]])
        fs: Sampling frequency in Hz (int)
        f0_floor: Minimum fundamental frequency to be detected (float, default: 71.0)
        f0_ceil: Maximum fundamental frequency to be detected (float, default: 800.0)
        frame_period: Frame length in milliseconds (float, default: 5.0)
        allowed_range: Allowed range for pitch (float, default: 0.1)

    Returns:
        f0: Extracted F0 trajectory (numpy.ndarray[Any, numpy.dtype[numpy.floating[Any]]])
        t: Time axis in seconds (numpy.ndarray[Any, numpy.dtype[numpy.floating[Any]]])
    """

def stonemask(
    x: np.ndarray,
    f0: np.ndarray,
    t: np.ndarray,
    fs: int,
) -> np.ndarray:
    """Refine F0 trajectory using StoneMask algorithm.

    This function refines the F0 trajectory estimated by Harvest or DIO.

    Args:
        x: Input audio signal as a 1D numpy array (numpy.ndarray[Any, numpy.dtype[numpy.floating[Any]]])
        f0: F0 trajectory to be refined (numpy.ndarray[Any, numpy.dtype[numpy.floating[Any]]])
        t: Time axis in seconds (numpy.ndarray[Any, numpy.dtype[numpy.floating[Any]]])
        fs: Sampling frequency in Hz (int)

    Returns:
        Refined F0 trajectory (numpy.ndarray[Any, numpy.dtype[numpy.floating[Any]]])
    """

def synthesize(
    f0: np.ndarray,
    sp: np.ndarray,
    ap: np.ndarray,
    fs: int,
    frame_period: float = 5.0,
) -> np.ndarray:
    """Synthesize speech from spectral parameters.

    Args:
        f0: F0 trajectory (numpy.ndarray[Any, numpy.dtype[numpy.floating[Any]]])
        sp: Spectral envelope (numpy.ndarray[Any, numpy.dtype[numpy.floating[Any]]], 2D array)
        ap: Aperiodicity (numpy.ndarray[Any, numpy.dtype[numpy.floating[Any]]], 2D array)
        fs: Sampling frequency in Hz (int)
        frame_period: Frame length in milliseconds (float, default: 5.0)

    Returns:
        Synthesized audio signal (numpy.ndarray[Any, numpy.dtype[numpy.floating[Any]]])
    """

def cheaptrick(
    x: np.ndarray,
    f0: np.ndarray,
    t: np.ndarray,
    fs: int,
    q1: float = -0.15,
    f0_floor: float = 71.0,
) -> np.ndarray:
    """Extract spectral envelope using CheapTrick algorithm.

    Args:
        x: Input audio signal as a 1D numpy array (numpy.ndarray[Any, numpy.dtype[numpy.floating[Any]]])
        f0: F0 trajectory (numpy.ndarray[Any, numpy.dtype[numpy.floating[Any]]])
        t: Time axis in seconds (numpy.ndarray[Any, numpy.dtype[numpy.floating[Any]]])
        fs: Sampling frequency in Hz (int)
        q1: Spectral envelope parameter (float, default: -0.15)
        f0_floor: Minimum fundamental frequency (float, default: 71.0)

    Returns:
        Spectral envelope (numpy.ndarray[Any, numpy.dtype[numpy.floating[Any]]], 2D array)
    """

def d4c(
    x: np.ndarray,
    f0: np.ndarray,
    t: np.ndarray,
    fs: int,
    threshold: float = 0.85,
) -> np.ndarray:
    """Extract aperiodicity using D4C algorithm.

    Args:
        x: Input audio signal as a 1D numpy array (numpy.ndarray[Any, numpy.dtype[numpy.floating[Any]]])
        f0: F0 trajectory (numpy.ndarray[Any, numpy.dtype[numpy.floating[Any]]])
        t: Time axis in seconds (numpy.ndarray[Any, numpy.dtype[numpy.floating[Any]]])
        fs: Sampling frequency in Hz (int)
        threshold: Voicing threshold (float, default: 0.85)

    Returns:
        Aperiodicity (numpy.ndarray[Any, numpy.dtype[numpy.floating[Any]]], 2D array)
    """

def get_cheaptrick_fft_size(fs: int) -> int:
    """Get FFT size for CheapTrick.

    Args:
        fs: Sampling frequency in Hz (int)

    Returns:
        FFT size (int)
    """
