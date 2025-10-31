import av
import numpy as np


def load_audio(file: str, sr: int) -> np.ndarray:
    """Load and resample audio file to mono at the specified sample rate.

    Args:
        file: Path to the audio file
        sr: Target sample rate

    Returns:
        Flattened numpy array of audio samples

    Raises:
        RuntimeError: If audio loading fails
    """
    try:
        with av.open(file, "r") as container:
            stream = next(s for s in container.streams if s.type == "audio")
            resampler = av.audio.resampler.AudioResampler(  # type: ignore
                format="flt", layout="mono", rate=sr
            )

            audio_data = []
            for frame in container.decode(stream):
                resampled = resampler.resample(frame)
                if not resampled:
                    continue

                # Normalize to list for uniform processing
                frames = resampled if isinstance(resampled, list) else [resampled]

                # Collect audio arrays from all frames
                audio_data.extend(f.to_ndarray() for f in frames)

            if not audio_data:
                return np.array([], dtype=np.float32)

            return np.concatenate(audio_data, axis=1).flatten()
    except Exception as e:
        raise RuntimeError(f"Failed to load audio with PyAV: {e}") from e
