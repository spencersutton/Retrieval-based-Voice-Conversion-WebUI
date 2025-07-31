import os
import sys
import json
import re
import time
import shutil
import multiprocessing
from multiprocessing import Queue, cpu_count
from typing import List, Literal, Optional
import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Gtk, Adw, GLib, Gio
from dotenv import load_dotenv
import numpy as np
import torch
import sounddevice as sd
import librosa
import torch.nn.functional as F
import torchaudio.transforms as tat
from tools.torchgate import TorchGate
from infer.lib import rtrvc as rvc_for_realtime
from configs.config import Config

# --- Environment and Path Setup ---
load_dotenv()
os.environ["OMP_NUM_THREADS"] = "4"
if sys.platform == "darwin":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
now_dir = os.getcwd()
sys.path.append(now_dir)


def printt(strr: str, *args):
    if len(args) == 0:
        print(strr)
    else:
        print(strr % args)


def get_device_samplerate():
    return int(sd.query_devices(device=sd.default.device[0])["default_samplerate"])


def phase_vocoder(
    a: torch.Tensor, b: torch.Tensor, fade_out: torch.Tensor, fade_in: torch.Tensor
) -> torch.Tensor:
    window = torch.sqrt(fade_out * fade_in)
    fa: torch.Tensor = torch.fft.rfft(a * window)
    fb: torch.Tensor = torch.fft.rfft(b * window)
    absab = torch.abs(fa) + torch.abs(fb)
    n = a.shape[0]
    if n % 2 == 0:
        absab[1:-1] *= 2
    else:
        absab[1:] *= 2
    phia = torch.angle(fa)
    phib = torch.angle(fb)
    deltaphase = phib - phia
    deltaphase = deltaphase - 2 * np.pi * torch.floor(deltaphase / 2 / np.pi + 0.5)
    w = 2 * np.pi * torch.arange(n // 2 + 1).to(a) + deltaphase
    t = torch.arange(n).unsqueeze(-1).to(a) / n
    result = (
        a * (fade_out**2)
        + b * (fade_in**2)
        + torch.sum(absab * torch.cos(w * t + phia), -1) * window / n
    )
    return result


current_dir = os.getcwd()
inp_q = Queue()
opt_q = Queue()
n_cpu = min(cpu_count(), 8)


GUI_TITLE = "RVC GUI"


class GUIConfig:
    pth_path: str = ""
    index_path: str = ""
    pitch: int = 0
    formant: float = 0.0
    sr_type: str = "sr_model"
    block_time: float = 0.25  # in second
    threshold: int = -60
    crossfade_time: float = 0.05
    extra_time: float = 2.5
    I_noise_reduce: bool = False
    O_noise_reduce: bool = False
    use_pv: bool = False
    rms_mix_rate: float = 0.0
    index_rate: float = 0.0
    n_cpu: int = min(n_cpu, 4)
    f0method: Literal["harvest", "crepe", "rmvpe", "fcpe"] = "fcpe"
    sg_hostapi: str = ""
    sg_input_device: str = ""
    sg_output_device: str = ""
    samplerate: int = -1
    channels: int = -1


class VCState:
    # VC state
    rvc: rvc_for_realtime.RVC
    zc: int
    block_frame: int
    block_frame_16k: int
    crossfade_frame: int
    sola_buffer_frame: int
    sola_search_frame: int
    extra_frame: int

    input_wav: torch.Tensor
    input_wav_denoise: torch.Tensor
    input_wav_res: torch.Tensor
    rms_buffer: np.ndarray
    sola_buffer: torch.Tensor
    nr_buffer: torch.Tensor
    output_buffer: torch.Tensor

    skip_head: int
    return_length: int

    fade_in_window: torch.Tensor
    fade_out_window: torch.Tensor

    resampler: tat.Resample
    resampler2: Optional[tat.Resample]

    tg: TorchGate

    def __init__(
        self,
        gui_config: GUIConfig,
        rvc_config: Config,
        last_state: Optional["VCState"] = None,
    ):
        torch.cuda.empty_cache()

        self.rvc = rvc_for_realtime.RVC(
            gui_config.pitch,
            gui_config.formant,
            gui_config.pth_path,
            gui_config.index_path,
            gui_config.index_rate,
            gui_config.n_cpu,
            inp_q,
            opt_q,
            rvc_config,
            last_state.rvc if last_state else None,
        )

        gui_config.samplerate = (
            self.rvc.tgt_sr
            if gui_config.sr_type == "sr_model"
            else get_device_samplerate()
        )

        gui_config.channels = get_device_channels()

        self.zc = gui_config.samplerate // 100
        self.block_frame = (
            int(np.round(gui_config.block_time * gui_config.samplerate / self.zc))
            * self.zc
        )

        self.block_frame_16k = 160 * self.block_frame // self.zc

        self.crossfade_frame = (
            int(np.round(gui_config.crossfade_time * gui_config.samplerate / self.zc))
            * self.zc
        )
        self.sola_buffer_frame = min(self.crossfade_frame, 4 * self.zc)
        self.sola_search_frame = self.zc
        self.extra_frame = (
            int(np.round(gui_config.extra_time * gui_config.samplerate / self.zc))
            * self.zc
        )

        self.input_wav = torch.zeros(
            self.extra_frame
            + self.crossfade_frame
            + self.sola_search_frame
            + self.block_frame,
            device=rvc_config.device,
            dtype=torch.float32,
        )

        self.input_wav_denoise: torch.Tensor = self.input_wav.clone()
        self.input_wav_res = torch.zeros(
            160 * self.input_wav.shape[0] // self.zc,
            device=rvc_config.device,
            dtype=torch.float32,
        )
        self.rms_buffer = np.zeros(4 * self.zc, dtype="float32")
        self.sola_buffer = torch.zeros(
            self.sola_buffer_frame, device=rvc_config.device, dtype=torch.float32
        )

        self.nr_buffer = self.sola_buffer.clone()
        self.output_buffer = self.input_wav.clone()
        self.skip_head = self.extra_frame // self.zc

        self.return_length = (
            self.block_frame + self.sola_buffer_frame + self.sola_search_frame
        ) // self.zc
        self.fade_in_window: torch.Tensor = (
            torch.sin(
                0.5
                * np.pi
                * torch.linspace(
                    0.0,
                    1.0,
                    steps=self.sola_buffer_frame,
                    device=rvc_config.device,
                    dtype=torch.float32,
                )
            )
            ** 2
        )

        self.fade_out_window: torch.Tensor = 1 - self.fade_in_window
        self.resampler = tat.Resample(
            orig_freq=gui_config.samplerate,
            new_freq=16000,
            dtype=torch.float32,
        ).to(rvc_config.device)
        if self.rvc.tgt_sr != gui_config.samplerate:
            self.resampler2 = tat.Resample(
                orig_freq=self.rvc.tgt_sr,
                new_freq=gui_config.samplerate,
                dtype=torch.float32,
            ).to(rvc_config.device)
        else:
            self.resampler2 = None
        self.tg = TorchGate(
            sr=gui_config.samplerate, n_fft=4 * self.zc, prop_decrease=0.9
        ).to(rvc_config.device)


class UiState:
    stream: Optional[sd.Stream] = None
    delay_time = 0
    hostapis: List[str] = []
    input_devices: List[str] = []
    output_devices: List[str] = []
    input_devices_indices: List[int] = []
    output_devices_indices: List[int] = []

    vc_state: Optional[VCState] = None

    def __init__(self) -> None:
        self.gui_config = GUIConfig()
        self.config = Config()
        self.function = "vc"

    def start_stream(self):
        if self.stream is not None:
            print("Already started...")
            return
        if self.vc_state is None:
            print("No VC state!")
            return

        self.stream = sd.Stream(
            callback=self.audio_callback,
            blocksize=self.vc_state.block_frame,
            samplerate=self.gui_config.samplerate,
            channels=self.gui_config.channels,
            dtype="float32",
        )

        self.stream.start()

    def stop_stream(self):
        if self.stream is None:
            print("No stream")
            return
        self.stream.abort()
        self.stream.close()
        self.stream = None

    def start_vc(self):
        self.vc_state = VCState(self.gui_config, self.config, self.vc_state)
        self.start_stream()

    def __str__(self):
        json_output = json.dumps(self, indent=4, cls=StateEncoder)
        return json_output

    def audio_callback(
        self, input_data: np.ndarray, output_data: np.ndarray, frames, times, status
    ) -> None:
        """
        Processing audio
        """
        if self.vc_state is None:
            print("VC State isn't initialized...")
            return

        state = self.vc_state

        start_time = time.perf_counter()

        input_data = librosa.to_mono(input_data.T)

        if self.gui_config.threshold > -60:
            input_data = np.append(self.rms_buffer, input_data)
            rms = librosa.feature.rms(
                y=input_data,
                frame_length=4 * state.zc,
                hop_length=state.zc,
            )[:, 2:]
            state.rms_buffer[:] = input_data[-4 * self.zc :]
            input_data = input_data[2 * self.zc - self.zc // 2 :]
            db_threshold = (
                librosa.amplitude_to_db(rms, ref=1.0)[0] < self.gui_config.threshold
            )
            for i in range(db_threshold.shape[0]):
                if db_threshold[i]:
                    input_data[i * state.zc : (i + 1) * self.zc] = 0
            input_data = input_data[self.zc // 2 :]

        state.input_wav[: -state.block_frame] = state.input_wav[
            state.block_frame :
        ].clone()
        state.input_wav[-input_data.shape[0] :] = torch.from_numpy(input_data).to(
            self.config.device
        )
        state.input_wav_res[: -state.block_frame_16k] = state.input_wav_res[
            state.block_frame_16k :
        ].clone()

        # input noise reduction and resampling
        if self.gui_config.I_noise_reduce:
            state.input_wav_denoise[: -state.block_frame] = state.input_wav_denoise[
                state.block_frame :
            ].clone()
            input_wav = state.input_wav[-state.sola_buffer_frame - state.block_frame :]
            input_wav = state.tg(
                input_wav.unsqueeze(0), state.input_wav.unsqueeze(0)
            ).squeeze(0)
            input_wav[: state.sola_buffer_frame] *= state.fade_in_window
            input_wav[: state.sola_buffer_frame] += (
                state.nr_buffer * state.fade_out_window
            )
            state.input_wav_denoise[-state.block_frame :] = input_wav[
                : state.block_frame
            ]
            state.nr_buffer[:] = input_wav[state.block_frame :]
            state.input_wav_res[-state.block_frame_16k - 160 :] = state.resampler(
                state.input_wav_denoise[-state.block_frame - 2 * state.zc :]
            )[160:]
        else:
            state.input_wav_res[-160 * (input_data.shape[0] // state.zc + 1) :] = (
                state.resampler(state.input_wav[-input_data.shape[0] - 2 * state.zc :])[
                    160:
                ]
            )
        # infer
        if self.function == "vc":
            infer_wav = state.rvc.infer(
                state.input_wav_res,
                state.block_frame_16k,
                state.skip_head,
                state.return_length,
                state.gui_config.f0method,
            )
            if state.resampler2 is not None:
                infer_wav = state.resampler2(infer_wav)
        elif self.gui_config.I_noise_reduce:
            infer_wav = state.input_wav_denoise[self.extra_frame :].clone()
        else:
            infer_wav = state.input_wav[state.extra_frame :].clone()
        # output noise reduction
        if self.gui_config.O_noise_reduce and self.function == "vc":
            state.output_buffer[: -state.block_frame] = state.output_buffer[
                state.block_frame :
            ].clone()
            state.output_buffer[-state.block_frame :] = infer_wav[-state.block_frame :]
            infer_wav = state.tg(
                infer_wav.unsqueeze(0), state.output_buffer.unsqueeze(0)
            ).squeeze(0)
        # volume envelop mixing
        if self.gui_config.rms_mix_rate < 1 and self.function == "vc":
            if self.gui_config.I_noise_reduce:
                input_wav = state.input_wav_denoise[state.extra_frame :]
            else:
                input_wav = state.input_wav[state.extra_frame :]
            rms1 = librosa.feature.rms(
                y=input_wav[: infer_wav.shape[0]].cpu().numpy(),
                frame_length=4 * state.zc,
                hop_length=state.zc,
            )
            rms1 = torch.from_numpy(rms1).to(self.config.device)
            rms1 = F.interpolate(
                rms1.unsqueeze(0),
                size=infer_wav.shape[0] + 1,
                mode="linear",
                align_corners=True,
            )[0, 0, :-1]
            rms2 = librosa.feature.rms(
                y=infer_wav[:].cpu().numpy(),
                frame_length=4 * state.zc,
                hop_length=state.zc,
            )
            rms2 = torch.from_numpy(rms2).to(self.config.device)
            rms2 = F.interpolate(
                rms2.unsqueeze(0),
                size=infer_wav.shape[0] + 1,
                mode="linear",
                align_corners=True,
            )[0, 0, :-1]
            rms2 = torch.max(rms2, torch.zeros_like(rms2) + 1e-3)
            infer_wav *= torch.pow(
                rms1 / rms2, torch.tensor(1 - self.gui_config.rms_mix_rate)
            )
        # SOLA algorithm from https://github.com/yxlllc/DDSP-SVC
        conv_input = infer_wav[
            None, None, : state.sola_buffer_frame + state.sola_search_frame
        ]
        cor_nom = F.conv1d(conv_input, state.sola_buffer[None, None, :])
        cor_den = torch.sqrt(
            F.conv1d(
                conv_input**2,
                torch.ones(1, 1, state.sola_buffer_frame, device=self.config.device),
            )
            + 1e-8
        )
        if sys.platform == "darwin":
            _, sola_offset = torch.max(cor_nom[0, 0] / cor_den[0, 0])
            sola_offset = sola_offset.item()
        else:
            sola_offset = torch.argmax(cor_nom[0, 0] / cor_den[0, 0])
        printt("sola_offset = %d", int(sola_offset))
        infer_wav = infer_wav[sola_offset:]
        if "privateuseone" in str(self.config.device) or not self.gui_config.use_pv:
            infer_wav[: state.sola_buffer_frame] *= state.fade_in_window
            infer_wav[: state.sola_buffer_frame] += (
                state.sola_buffer * state.fade_out_window
            )
        else:
            infer_wav[: state.sola_buffer_frame] = phase_vocoder(
                state.sola_buffer,
                infer_wav[: state.sola_buffer_frame],
                state.fade_out_window,
                state.fade_in_window,
            )
        state.sola_buffer[:] = infer_wav[
            state.block_frame : state.block_frame + state.sola_buffer_frame
        ]
        output_data[:] = (
            infer_wav[: state.block_frame]
            .repeat(self.gui_config.channels, 1)
            .t()
            .cpu()
            .numpy()
        )
        total_time = time.perf_counter() - start_time
        printt("Infer time: %.2f", total_time)


class StateEncoder(json.JSONEncoder):
    def default(self, o):
        if o == Config:
            return "Config {...}"
        # if isinstance(o, (UiState, GUIConfig, Config)):
        # Convert custom objects to their dictionary representation
        if isinstance(o, sd.Stream):
            # Cannot serialize a stream object, return a placeholder
            return "SoundDevice Stream Object (Not Serializable)"
        # Let the base class default method raise the TypeError for other types

        return o.__dict__
        return super().default(o)


def get_device_sample_rate():
    return int(sd.query_devices(device=sd.default.device[0])["default_samplerate"])


def get_device_channels():
    max_input_channels = sd.query_devices(device=sd.default.device[0])[
        "max_input_channels"
    ]
    max_output_channels = sd.query_devices(device=sd.default.device[1])[
        "max_output_channels"
    ]
    return min(max_input_channels, max_output_channels, 2)


def update_devices(state: UiState, hostapi_name: Optional[str] = None):
    """Get devices"""
    sd._terminate()
    sd._initialize()

    devices = sd.query_devices()
    hostapis = sd.query_hostapis()

    for hostapi in hostapis:
        for device_idx in hostapi["devices"]:
            devices[device_idx]["hostapi_name"] = hostapi["name"]

    state.hostapis = [hostapi["name"] for hostapi in hostapis]

    if hostapi_name not in state.hostapis:
        hostapi_name = state.hostapis[0]

    for d in devices:
        # Filter for devices matching the selected host API
        if d.get("hostapi_name") == hostapi_name:
            # Use .get() for a safe fallback from index to name
            device_identifier = d.get("index", d["name"])
            # Add to input lists if it's an input device
            if d["max_input_channels"] > 0:
                state.input_devices.append(d["name"])
                state.input_devices_indices.append(device_identifier)
            # Add to output lists if it's an output device
            if d["max_output_channels"] > 0:
                state.output_devices.append(d["name"])
                state.output_devices_indices.append(device_identifier)

    print(state)


def set_devices(state: UiState, input_device, output_device):
    """Set devices"""
    sd.default.device[0] = state.input_devices_indices[
        state.input_devices.index(input_device)
    ]
    sd.default.device[1] = state.output_devices_indices[
        state.output_devices.index(output_device)
    ]
    printt("Input device: %s:%s", str(sd.default.device[0]), input_device)
    printt("Output device: %s:%s", str(sd.default.device[1]), output_device)


class MainWindow(Gtk.ApplicationWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        GLib.set_application_name(GUI_TITLE)

        self.main_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.main_box.set_css_classes(["main-box"])
        self.set_child(self.main_box)

        action = Gio.SimpleAction.new("about", None)
        self.add_action(action)

        self.button = Gtk.Button(label="Hello")
        self.main_box.append(self.button)
        self.button.connect("clicked", self.hello)

        self.set_default_size(600, 250)
        self.set_title(GUI_TITLE)

    def hello(self, button: Gtk.Button):
        # print("hello!")

        state = UiState()
        update_devices(state)


class MyApp(Adw.Application):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.connect("activate", self.on_activate)

    def on_activate(self, app):
        self.win = MainWindow(application=app)
        self.win.present()


css_provider = Gtk.CssProvider()
css_provider.load_from_path("gtk.css")

app = MyApp(application_id="com.example.GtkRVC")
app.run(sys.argv)
