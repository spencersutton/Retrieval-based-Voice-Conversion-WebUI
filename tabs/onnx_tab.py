import gradio as gr

from shared import i18n


def export_onnx(ModelPath: str, ExportedPath: str):
    from infer.modules.onnx.export import export_onnx as eo

    eo(ModelPath, ExportedPath)


def create_onnx_tab():
    with gr.TabItem(i18n("Onnx导出")):
        with gr.Row():
            ckpt_dir = gr.Textbox(label=i18n("RVC模型路径"), value="", interactive=True)
        with gr.Row():
            onnx_dir = gr.Textbox(
                label=i18n("Onnx输出路径"), value="", interactive=True
            )
        with gr.Row():
            infoOnnx = gr.Label(label="info")
        with gr.Row():
            butOnnx = gr.Button(i18n("导出Onnx模型"), variant="primary")
        butOnnx.click(
            export_onnx, [ckpt_dir, onnx_dir], infoOnnx, api_name="export_onnx"
        )
