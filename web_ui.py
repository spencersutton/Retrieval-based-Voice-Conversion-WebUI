import shared
from tabs.ckpt_processing_tab import create_ckpt_processing_tab
from tabs.faq_tab import create_faq_tab
from tabs.inference_tab import create_inference_tab

import gradio as gr

from tabs.onnx_tab import create_onnx_tab
from tabs.train_tab import create_train_tab
from tabs.vocal_tab import create_vocal_tab

import torch
from fairseq.data.dictionary import Dictionary
import fairseq
torch.serialization.add_safe_globals({
    'fairseq.data.dictionary.Dictionary': fairseq.data.dictionary.Dictionary
})

torch.serialization.add_safe_globals([Dictionary])

with gr.Blocks(title="RVC WebUI Fork") as app:
    gr.Markdown("## RVC WebUI Fork")
    with gr.Tabs():
        create_inference_tab(app=app)
        create_vocal_tab()
        create_train_tab()
        create_ckpt_processing_tab()

    if shared.config.iscolab:
        app.queue(max_size=1022).launch(share=True)
    else:
        app.queue(max_size=1022).launch(
            server_name="0.0.0.0",
            inbrowser=not shared.config.noautoopen,
            server_port=shared.config.listen_port,
            quiet=True,
        )
