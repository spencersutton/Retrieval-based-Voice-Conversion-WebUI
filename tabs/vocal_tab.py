import gradio as gr
from PolUVR.utils import PolUVR_UI


def create_vocal_tab():
    with gr.TabItem("UVR5"):
        PolUVR_UI()
