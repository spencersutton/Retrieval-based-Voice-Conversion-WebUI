import platform
import numpy as np
import av
import re
import av.audio.frame


def wav2(i, o, format):
    inp = av.open(i, "rb")
    if format == "m4a":
        format = "mp4"
    out = av.open(o, "wb", format=format)
    if format == "ogg":
        format = "libvorbis"
    if format == "mp4":
        format = "aac"

    ostream = out.add_stream(format)

    for frame in inp.decode(audio=0):
        for p in ostream.encode(frame):
            out.mux(p)

    for p in ostream.encode(None):
        out.mux(p)

    out.close()
    inp.close()


def load_audio(file: str, sr: int) -> np.ndarray:
    try:
        with av.open(file, "r") as container:
            stream = next(s for s in container.streams if s.type == "audio")

            resampler = av.AudioResampler(format="flt", layout="mono", rate=sr)

            audio_data = []
            for frame in container.decode(stream):
                if not isinstance(frame, av.audio.frame.AudioFrame):
                    continue
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
    return np.array([])


def clean_path(path_str):
    if platform.system() == "Windows":
        path_str = path_str.replace("/", "\\")
    path_str = re.sub(
        r"[\u202a\u202b\u202c\u202d\u202e]", "", path_str
    )  # 移除 Unicode 控制字符
    return path_str.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
