import os
from typing import List
import gradio as gr

import shared
from shared import PITCH_METHODS, PitchMethod, i18n


def clean():
    return {"value": "", "__type__": "update"}


def change_choices():
    names = []
    for name in os.listdir(shared.weight_root):
        if name.endswith(".pth"):
            names.append(name)
    index_paths = [""]
    for root, dirs, files in os.walk(shared.index_root, topdown=False):
        for name in files:
            if name.endswith(".index") and "trained" not in name:
                index_paths.append("%s/%s" % (root, name))
    return {"choices": sorted(names), "__type__": "update"}, {
        "choices": sorted(index_paths),
        "__type__": "update",
    }


def get_pitch_methods() -> List[PitchMethod]:
    if not hasattr(shared.config, "dml"):
        # Handle cases where shared.config.dml might not exist
        return PITCH_METHODS

    return (
        [method for method in PITCH_METHODS if method != "crepe"]
        if shared.config.dml
        else PITCH_METHODS
    )


def get_model_list() -> List[str]:
    return sorted(shared.names)


def get_index_paths() -> List[str]:
    return sorted(shared.index_paths)


def create_inference_tab(app: gr.Blocks):

    with gr.TabItem(i18n("Inference")):
        gr.api(
            get_model_list,
            api_name="get_model_list",
        )
        gr.api(get_pitch_methods, api_name="get_pitch_methods")
        gr.api(get_index_paths, api_name="get_index_paths")
        with gr.Row():
            with gr.Column():
                model_dropdown = gr.Dropdown(
                    label=i18n("Model"), choices=sorted(shared.names)
                )
                with gr.Column():
                    refresh_btn = gr.Button(i18n("Refresh"), variant="primary")
                with gr.TabItem(i18n("Basic")):
                    audio_input = gr.Audio(
                        label=i18n("Input Audio"),
                        type="numpy",
                    )
                    convert_btn = gr.Button(i18n("Convert"), variant="primary")
                    autoplay_checkbox = gr.Checkbox(label=i18n("Autoplay"), value=False)

                    vc_file_output = gr.Audio(
                        label=i18n("Output Audio"),
                    )

                    def set_autoplay(x: bool):
                        print(f"Set auto play: {x}")
                        return {"autoplay": x, "__type__": "update"}

                    autoplay_checkbox.input(
                        set_autoplay,
                        [autoplay_checkbox],
                        [vc_file_output],
                    )

            with gr.Column():
                pitch_offset = gr.Slider(
                    label="Pitch Offset",
                    minimum=-24,
                    maximum=24,
                    step=1,
                    value=0,
                )
                resample_sr0 = gr.Slider(
                    minimum=0,
                    maximum=48000,
                    label=i18n("Resample Rate (Skip if it is 0)"),
                    value=0,
                    step=1,
                    interactive=True,
                )
                rms_mix_rate0 = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label=i18n(
                        # "输入源音量包络替换输出音量包络融合比例，越靠近1越使用输出包络"
                        "RMS Mix Rate"
                    ),
                    value=0.25,
                    interactive=True,
                )
                protect0 = gr.Slider(
                    minimum=0,
                    maximum=0.5,
                    label=i18n(
                        # "保护清辅音和呼吸声，防止电音撕裂等artifact，拉满0.5不开启，调低加大保护力度但可能降低索引效果"
                        "Protect 0 (Reduce Artifact)"
                    ),
                    value=0.33,
                    step=0.01,
                    interactive=True,
                )
                index_rate1 = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label=i18n("检索特征占比"),
                    value=0.75,
                    interactive=True,
                )
            with gr.Column():
                file_index2 = gr.Dropdown(
                    label=i18n("Index File"),
                    choices=sorted(shared.index_paths),
                    interactive=True,
                    allow_custom_value=True,
                    value="",
                )
                f0method0 = gr.Radio(
                    label=i18n("Pitch Method"),
                    choices=get_pitch_methods(),
                    value="rmvpe",
                    interactive=True,
                )
                vc_log_output = gr.Textbox(label=i18n("Log info"))

        convert_btn.click(
            shared.vc.vc_single,
            [
                audio_input,
                pitch_offset,
                f0method0,
                file_index2,
                index_rate1,
                resample_sr0,
                rms_mix_rate0,
                protect0,
            ],
            [vc_log_output, vc_file_output],
            api_name="infer_convert",
        )
        refresh_btn.click(
            fn=change_choices,
            inputs=[],
            outputs=[model_dropdown, file_index2],
            api_name="infer_refresh",
        )
        model_dropdown.change(
            fn=shared.vc.get_vc,
            inputs=[
                model_dropdown,
                protect0,
            ],  # Use protect0 and protect1 from Basic/Batch tab
            outputs=[protect0, file_index2],
            api_name="infer_change_voice",
        )
        app.load(
            fn=shared.vc.get_vc,
            inputs=[
                model_dropdown,
                protect0,
            ],  # Use the components themselves to get their initial values
            outputs=[protect0, file_index2],
        )
