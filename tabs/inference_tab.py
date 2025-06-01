import os
from typing import List
import gradio as gr

import shared
from shared import i18n

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

def create_inference_tab(app: gr.Blocks):
    with gr.TabItem(i18n("Inference")):
        with gr.Row():
            model_dropdown = gr.Dropdown(
                label=i18n("Model"), choices=sorted(shared.names)
            )
            with gr.Column():
                refresh_btn = gr.Button(
                    i18n("Refresh"), variant="primary"
                )  # Use the shared button
                
                unload_btn = gr.Button(i18n("Unload Model"), variant="primary")
            spk_item = gr.Slider(
                minimum=0,
                maximum=2333,
                step=1,
                label=i18n("请选择说话人id"),
                value=0,
                visible=False,
                interactive=True,
            )
            unload_btn.click(
                fn=clean,
                inputs=[],
                outputs=[model_dropdown],
                api_name="infer_clean",
            )

        with gr.TabItem(i18n("Basic")):
            with gr.Group():
                with gr.Row():
                    with gr.Column():
                        vc_transform0 = gr.Slider(
                            label="Pitch Offset",
                            minimum=-24,
                            maximum=24,
                            step=1,
                            value=0,
                        )
                        input_audio = gr.Audio(
                            label=i18n("Input Audio"),
                            type="numpy",
                        )
                        file_index1 = gr.Textbox(
                            label=i18n("Manual Index File"),
                            placeholder="/path/to/model_example.index",
                            interactive=True,
                        )
                        file_index2 = gr.Dropdown(
                            label=i18n("Auto Detected Index"),
                            choices=sorted(shared.index_paths),
                            interactive=True,
                        )
                        f0method0 = gr.Radio(
                            label=i18n("Pitch Method"),
                            choices=(
                                ["pm", "harvest", "crepe", "rmvpe"]
                                if shared.config.dml == False
                                else ["pm", "harvest", "rmvpe"]
                            ),
                            value="rmvpe",
                            interactive=True,
                        )
                    with gr.Column():
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
                        filter_radius0 = gr.Slider(
                            minimum=0,
                            maximum=7,
                            label=i18n(
                                # ">=3则使用对harvest音高识别的结果使用中值滤波，数值为滤波半径，使用可以削弱哑音"
                                "Filter Radius (Legacy)"
                            ),
                            value=3,
                            step=1,
                            interactive=True,
                        )
                        index_rate1 = gr.Slider(
                            minimum=0,
                            maximum=1,
                            label=i18n("检索特征占比"),
                            value=0.75,
                            interactive=True,
                        )
                        f0_file = gr.File(
                            label=i18n(
                                "F0曲线文件, 可选, 一行一个音高, 代替默认F0及升降调"
                            ),
                            visible=False,
                        )

            with gr.Group():
                with gr.Column():
                    but0 = gr.Button(i18n("Convert"), variant="primary")
                    with gr.Row():
                        vc_log_output = gr.Textbox(label=i18n("Log info"))
                        vc_file_output = gr.Audio(
                            label=i18n("输出音频(右下角三个点,点了可以下载)")
                        )

                    but0.click(
                        shared.vc.vc_single,
                        [
                            spk_item,
                            input_audio,
                            vc_transform0,
                            f0_file,
                            f0method0,
                            file_index1,
                            file_index2,
                            index_rate1,
                            filter_radius0,
                            resample_sr0,
                            rms_mix_rate0,
                            protect0,
                        ],
                        [vc_log_output, vc_file_output],
                        api_name="infer_convert",
                    )

        model_dropdown.change(
            fn=shared.vc.get_vc,
            inputs=[
                model_dropdown,
                protect0,
            ],  # Use protect0 and protect1 from Basic/Batch tab
            outputs=[spk_item, protect0, file_index2],
            api_name="infer_change_voice",
        )
        refresh_btn.click(
                    fn=change_choices,
                    inputs=[],
                    outputs=[model_dropdown, file_index2],
                    api_name="infer_refresh",
                )
        app.load(
            fn=shared.vc.get_vc,
            inputs=[
                model_dropdown,
                protect0,
            ],  # Use the components themselves to get their initial values
            outputs=[spk_item, protect0, file_index2],
        )
