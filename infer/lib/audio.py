import av
import numpy as np


def load_audio(file: str, sr: int) -> np.ndarray:
    try:
        with av.open(file, "r") as container:
            stream = next(s for s in container.streams if s.type == "audio")

            resampler = av.audio.resampler.AudioResampler(
                format="flt", layout="mono", rate=sr
            )

            audio_data = []
            for frame in container.decode(stream):
                # Resample returns either a frame or a list of frames
                resampled = resampler.resample(frame)
                if not resampled:
                    continue
                if isinstance(resampled, list):
                    frames = resampled
                else:
                    frames = [resampled]

                for f in frames:
                    arr = f.to_ndarray()
                    audio_data.append(arr)

            return np.concatenate(audio_data, axis=1).flatten()
    except Exception as e:
        raise RuntimeError(f"Failed to load audio with PyAV: {e}") from e
