import os
import sys
import json
import re
import time
import shutil
import multiprocessing
from multiprocessing import Queue, cpu_count
from typing import Callable, List, Literal, Optional
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
    # sg_hostapi: str = ""
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
        rvc_config.use_jit = False

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
    # print(f"devices: {devices}")
    hostapis = sd.query_hostapis()
    for hostapi in hostapis:
        # print(f"Hostapi: {hostapi}")
        for device_idx in hostapi["devices"]:
            devices[device_idx]["hostapi_name"] = hostapi["name"]

    state.hostapis = [hostapi["name"] for hostapi in hostapis]

    if hostapi_name not in state.hostapis:
        hostapi_name = state.hostapis[0]
    state.input_devices = [
        d["name"]
        for d in devices
        if d["max_input_channels"] > 0 and d["hostapi_name"] == hostapi_name
    ]
    state.output_devices = [
        d["name"]
        for d in devices
        if d["max_output_channels"] > 0 and d["hostapi_name"] == hostapi_name
    ]
    state.input_devices_indices = [
        d["index"] if "index" in d else d["name"]
        for d in devices
        if d["max_input_channels"] > 0 and d["hostapi_name"] == hostapi_name
    ]
    state.output_devices_indices = [
        d["index"] if "index" in d else d["name"]
        for d in devices
        if d["max_output_channels"] > 0 and d["hostapi_name"] == hostapi_name
    ]

    print(state)


def set_devices(
    state: UiState,
    input_device: Optional[str] = None,
    output_device: Optional[str] = None,
):
    """Set devices"""
    if input_device is not None:
        sd.default.device[0] = state.input_devices_indices[
            state.input_devices.index(input_device)
        ]
        printt("Input device: %s:%s", str(sd.default.device[0]), input_device)

    if output_device is not None:
        sd.default.device[1] = state.output_devices_indices[
            state.output_devices.index(output_device)
        ]
        printt("Output device: %s:%s", str(sd.default.device[1]), output_device)


class MainWindow(Adw.ApplicationWindow):
    state: UiState = UiState()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.set_default_size(500, 500)
        # Note: The title is now set on the HeaderBar, not the window.

        # 1. Create the ToolbarView as the main container
        toolbar_view = Adw.ToolbarView()
        self.set_content(toolbar_view)

        # 2. Create the HeaderBar and add it to the ToolbarView's top
        header = Adw.HeaderBar()
        header.set_title_widget(Adw.WindowTitle(title=GUI_TITLE))
        toolbar_view.add_top_bar(header)

        # --- Reload Button ---
        self.reload_device_btn = Gtk.Button(label="Reload")
        self.reload_device_btn.set_icon_name("view-refresh-symbolic")
        self.reload_device_btn.connect("clicked", self.reload_device)
        header.pack_start(self.reload_device_btn)

        # --- NEW: Start/Stop Button and Spinner ---
        self.header_spinner = Gtk.Spinner()
        header.pack_end(self.header_spinner)

        self.start_stop_btn = Gtk.Button(label="Start")
        self.start_stop_btn.set_icon_name("media-playback-start-symbolic")
        # "suggested-action" makes it the primary (often blue) button
        self.start_stop_btn.get_style_context().add_class("suggested-action")
        self.start_stop_btn.connect("clicked", self.on_start_stop_clicked)
        header.pack_end(self.start_stop_btn)

        # 3. Create your page content as before
        main_group = Adw.PreferencesGroup()
        page = Adw.PreferencesPage()
        page.add(main_group)

        # 4. Set the page as the main content of the ToolbarView
        toolbar_view.set_content(page)

        # --- Input Device Dropdown ---
        # (The rest of the code for creating rows and dropdowns is identical)
        self.input_devices_list = Gtk.StringList()
        self.input_dropdown = Gtk.DropDown(
            model=self.input_devices_list, valign=Gtk.Align.CENTER
        )
        self.input_dropdown.connect(
            "notify::selected-item", self.on_input_device_changed
        )

        input_row = Adw.ActionRow(title="Input Device")
        input_row.add_suffix(self.input_dropdown)
        main_group.add(input_row)

        # --- Output Device Dropdown ---
        self.output_devices_list = Gtk.StringList()
        self.output_dropdown = Gtk.DropDown(
            model=self.output_devices_list, valign=Gtk.Align.CENTER
        )
        self.output_dropdown.connect(
            "notify::selected-item", self.on_output_device_changed
        )

        output_row = Adw.ActionRow(title="Output Device")
        output_row.add_suffix(self.output_dropdown)
        main_group.add(output_row)

        # Initial population of devices
        self.reload_device(None)

        # --- Model .pth File Input ---
        self.model_path_row = Adw.EntryRow(title="Model Path (.pth)")
        model_browse_btn = Gtk.Button(icon_name="document-open-symbolic")
        model_browse_btn.connect("clicked", self.on_open_model_path_clicked)
        self.model_path_row.add_suffix(model_browse_btn)
        main_group.add(self.model_path_row)

        # --- Index .index File Input ---
        self.index_path_row = Adw.EntryRow(title="Index File Path (.index)")
        index_browse_btn = Gtk.Button(icon_name="document-open-symbolic")
        index_browse_btn.connect("clicked", self.on_open_index_path_clicked)
        self.index_path_row.add_suffix(index_browse_btn)
        main_group.add(self.index_path_row)

    def on_open_model_path_clicked(self, widget):
        """Handler to open a file chooser for the .pth model file."""

        def s(file: str):
            self.state.gui_config.pth_path = file

        self._show_file_chooser(
            "Select Model File",
            self.model_path_row,
            pattern="*.pth",
            mime="application/octet-stream",  # A generic mime type
            on_file_path=s,
        )

    def on_open_index_path_clicked(self, widget):
        """Handler to open a file chooser for the .index file."""

        def s(file: str):
            self.state.gui_config.index_path = file

        self._show_file_chooser(
            "Select Index File",
            self.index_path_row,
            pattern="*.index",
            mime="application/octet-stream",  # A generic mime type
            on_file_path=s,
        )

    def _show_file_chooser(
        self,
        title: str,
        entry_row: Adw.EntryRow,
        pattern: str,
        mime: str,
        on_file_path: Optional[Callable],
    ):
        """Generic method to create and show a Gtk.FileChooserDialog."""
        # Create a filter for the specific file type
        file_filter = Gtk.FileFilter()
        file_filter.set_name(f"Files ({pattern})")
        file_filter.add_pattern(pattern)
        file_filter.add_mime_type(mime)

        # Create a filter for all files
        all_files_filter = Gtk.FileFilter()
        all_files_filter.set_name("All Files")
        all_files_filter.add_pattern("*")

        filters = Gio.ListStore(item_type=Gtk.FileFilter)
        filters.append(file_filter)
        filters.append(all_files_filter)

        dialog = Gtk.FileDialog(
            title=title, default_filter=file_filter, filters=filters
        )

        # Handle the dialog response
        def on_response(dialog: Gtk.FileDialog, result):
            # if response == Gtk.ResponseType.ACCEPT:
            #     file_path = dialog.get_file().get_path()
            #     entry_row.set_text(file_path)
            # dialog.get_data
            try:
                file = dialog.open_finish(result)
                if file is not None:
                    file_path = file.get_path()
                    print(f"File path is {file_path}")
                    entry_row.set_text(file_path)
                    # self.state.gui_config.
                    # on_file_path()
                    if on_file_path is not None:
                        on_file_path(file_path)
                # Handle loading file from here
            except GLib.Error as error:
                print(f"Error opening file: {error.message}")

            # dialog.destroy()

        # dialog.connect("response", on_response)
        dialog.open(parent=self, callback=on_response)

    def reload_device(self, button: Gtk.Button | None):
        update_devices(self.state)
        input_devices: List[str] = self.state.input_devices
        output_devices: List[str] = self.state.output_devices

        # Clear existing models
        while self.input_devices_list.get_n_items() > 0:
            self.input_devices_list.remove(0)
        while self.output_devices_list.get_n_items() > 0:
            self.output_devices_list.remove(0)

        # Populate models with new devices
        for device in input_devices:
            self.input_devices_list.append(device)
        for device in output_devices:
            self.output_devices_list.append(device)

        # Set a default selection
        if self.input_devices_list.get_n_items() > 0:
            self.input_dropdown.set_selected(0)
        if self.output_devices_list.get_n_items() > 0:
            self.output_dropdown.set_selected(0)

    def on_input_device_changed(self, dropdown: Gtk.DropDown, _param):
        selected_item = dropdown.get_selected_item()
        if selected_item is not None:
            device_name = selected_item.get_string()
            print(f"🎤 Input device selected: {device_name}")

            set_devices(state=self.state, input_device=device_name)

    def on_output_device_changed(self, dropdown: Gtk.DropDown, _param):
        selected_item = dropdown.get_selected_item()
        if selected_item is not None:
            device_name = selected_item.get_string()
            print(f"🔊 Output device selected: {device_name}")
            # set_devices(device_name)
            set_devices(state=self.state, output_device=device_name)

    def on_start_stop_clicked(self, widget):
        if self.state.stream is None:
            # --- START SEQUENCE ---
            # 1. Enter loading state: disable buttons and start spinner
            # self.start_stop_btn.set_sensitive(False)
            # self.reload_device_btn.set_sensitive(False)
            # self.header_spinner.start()

            # # 2. Run blocking function in a background thread
            # thread = threading.Thread(target=self._start_vc_thread)
            # thread.daemon = True
            # thread.start()
            self.state.start_vc()
        else:
            # --- STOP SEQUENCE ---
            # 1. Call the stop function (assumed to be fast)
            self.state.stop_stream()

            # 2. Revert button to "Start" state
            self.start_stop_btn.set_label("Start")
            self.start_stop_btn.set_icon_name("media-playback-start-symbolic")
            self.is_running = False
            self.reload_device_btn.set_sensitive(True)


class MyApp(Adw.Application):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.connect("activate", self.on_activate)

    def on_activate(self, app):
        self.win = MainWindow(application=app)
        self.win.present()


if __name__ == "__main__":
    css_provider = Gtk.CssProvider()
    css_provider.load_from_path("gtk.css")

    app = MyApp(application_id="com.example.GtkRVC")
    app.run(sys.argv)
