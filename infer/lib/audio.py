import platform
import numpy as np
import av
import re

def wav2(input_path: str, output_path: str, format: str):
    inp = av.open(input_path, "rb")
    if format == "m4a":
        format = "mp4"
    out = av.open(output_path, "wb", format=format)
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
        with av.open(file, 'r') as container:
            # Get the first audio stream
            stream = next(s for s in container.streams if s.type == 'audio')
            
            # Set the output format: single channel, float32, and the target sample rate
            stream.layout = 'mono'
            stream.format = 'flt'
            stream.rate = sr

            audio_data = []
            for frame in container.decode(stream):
                # Convert the PyAV frame to a NumPy array and append
                audio_data.append(frame.to_ndarray())
            
            # Concatenate all the frames into a single NumPy array
            return np.concatenate(audio_data, axis=1).flatten()
    except Exception as e:
        # A more specific error might be raised by av.open or during decoding
        raise RuntimeError(f"Failed to load audio with PyAV: {e}")


def clean_path(path_str: str) -> str:
    if platform.system() == "Windows":
        path_str = path_str.replace("/", "\\")
    path_str = re.sub(
        r"[\u202a\u202b\u202c\u202d\u202e]", "", path_str
    )  # 移除 Unicode 控制字符
    return path_str.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
