import gradio as gr
import shared

from PolUVR.utils import PolUVR_UI


def create_vocal_tab():
    with gr.TabItem(i18n("Vocal Preprocessing")):
        PolUVR_UI()
