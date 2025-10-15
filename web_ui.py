import shared
import fairseq
import torch
import gradio as gr
import git
import logging
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware import Middleware
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from tabs.ckpt_processing_tab import create_ckpt_processing_tab

# from tabs.faq_tab import create_faq_tab
# from tabs.onnx_tab import create_onnx_tab
from tabs.inference_tab import create_inference_tab
from tabs.train_tab import create_train_tab
from tabs.vocal_tab import create_vocal_tab

torch.serialization.add_safe_globals([fairseq.data.dictionary.Dictionary])


with gr.Blocks(title="RVC WebUI Fork") as app:
    repo = git.Repo(search_parent_directories=True)
    gr.Markdown(
        f"## RVC WebUI Fork ({repo.active_branch}) ({repo.head.object.hexsha[:7]})"
    )
    with gr.Tabs():
        create_inference_tab(app=app)
        create_vocal_tab()
        create_train_tab()
        create_ckpt_processing_tab()

    if shared.config.iscolab:
        app.queue(max_size=1022).launch(share=True)
    else:
        # enable Gradio queuing but don't call .launch() so we can mount into FastAPI
        app.queue(max_size=1022)

        # create FastAPI app and add CORS middleware so browser JS clients can call /config etc.
        fastapi_app = FastAPI()

        # restrict to the actual frontend origin when using credentials
        ALLOWED_ORIGINS = ["https://rvc-rest-gui.pages.dev"]

        fastapi_app.add_middleware(
            CORSMiddleware,
            allow_origins=ALLOWED_ORIGINS,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Private Network Access: browsers send Access-Control-Request-Private-Network on preflight
        # when a public context requests the loopback address. Respond with
        # Access-Control-Allow-Private-Network: true to allow those requests.
        from starlette.responses import Response

        @fastapi_app.middleware("http")
        async def _allow_private_network(request, call_next):
            origin = request.headers.get("origin")
            allow_origin_value = origin if origin in ALLOWED_ORIGINS else None

            # handle preflight requests (OPTIONS) - ensure necessary CORS + PNA headers are present
            if request.method == "OPTIONS":
                req_headers = request.headers.get("access-control-request-headers", "*")
                headers = {}
                if allow_origin_value:
                    headers["Access-Control-Allow-Origin"] = allow_origin_value
                    headers["Vary"] = "Origin"
                    headers["Access-Control-Allow-Credentials"] = "true"
                headers.update(
                    {
                        "Access-Control-Allow-Methods": "*",
                        "Access-Control-Allow-Headers": req_headers,
                        "Access-Control-Allow-Private-Network": "true",
                    }
                )
                return Response(status_code=200, headers=headers)

            response = await call_next(request)
            # ensure PNA header is present on normal responses as well
            response.headers["Access-Control-Allow-Private-Network"] = "true"
            if allow_origin_value:
                response.headers["Access-Control-Allow-Origin"] = allow_origin_value
                response.headers["Vary"] = "Origin"
                response.headers["Access-Control-Allow-Credentials"] = "true"
            return response

        # mount the Gradio Blocks app into FastAPI and run with uvicorn
        gr.mount_gradio_app(fastapi_app, app, path="/")

        import uvicorn

        # reduce noisy logs
        logging.getLogger("uvicorn.access").disabled = True
        logging.getLogger("uvicorn.error").setLevel(logging.WARNING)
        logging.getLogger("fastapi").setLevel(logging.WARNING)
        logging.getLogger("gradio").setLevel(logging.WARNING)

        uvicorn.run(
            fastapi_app,
            host="0.0.0.0",
            port=shared.config.listen_port,
            log_level="warning",
            access_log=False,
        )
