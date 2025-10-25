import av
import av.audio.frame
import numpy as np


def wav2(i: str, o: str, format: str) -> None:
    inp = av.open(i, "r")
    if format == "m4a":
        format = "mp4"
    out = av.open(o, "w", format=format)
    if format == "ogg":
        format = "libvorbis"
    if format == "mp4":
        format = "aac"

    ostream = out.add_stream(format)  # pyright: ignore[reportUnknownMemberType]

    for frame in inp.decode(audio=0):
        for p in ostream.encode(frame):  # type: ignore
            out.mux(p)  # type: ignore

    for p in ostream.encode(None):  # type: ignore
        out.mux(p)  # pyright: ignore[reportUnknownArgumentType]

    out.close()
    inp.close()


def load_audio(file: str, sr: int) -> np.ndarray:
    try:
        with av.open(file, "r") as container:
            stream = next(s for s in container.streams if s.type == "audio")

            resampler = av.AudioResampler(format="flt", layout="mono", rate=sr)

            audio_data: list[np.ndarray] = []
            for frame in container.decode(stream):
                if not isinstance(frame, av.audio.frame.AudioFrame):
                    continue
                # Resample returns either a frame or a list of frames
                resampled = resampler.resample(frame)
                if not resampled:
                    continue
                assert isinstance(resampled, list)
                frames = resampled

                for f in frames:
                    arr = f.to_ndarray()
                    audio_data.append(arr)

            return np.concatenate(audio_data, axis=1).flatten()
    except Exception as e:
        raise RuntimeError(f"Failed to load audio with PyAV: {e}") from e
    return np.array([])
