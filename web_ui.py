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

        # Use a middleware that echoes the Origin header (allows "all origins" even when credentials are used).
        # WARNING: this effectively permits any origin to send credentialed requests. Only use if acceptable.
        from starlette.responses import Response

        @fastapi_app.middleware("http")
        async def _cors_and_private_network(request, call_next):
            origin = request.headers.get("origin")

            # handle preflight requests (OPTIONS)
            if request.method == "OPTIONS":
                req_headers = request.headers.get("access-control-request-headers", "*")
                headers = {
                    "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
                    "Access-Control-Allow-Headers": req_headers,
                    "Access-Control-Allow-Private-Network": "true",
                }
                if origin:
                    headers["Access-Control-Allow-Origin"] = origin
                    headers["Vary"] = "Origin"
                    headers["Access-Control-Allow-Credentials"] = "true"
                return Response(status_code=200, headers=headers)

            response = await call_next(request)
            # add headers on regular responses too
            response.headers["Access-Control-Allow-Private-Network"] = "true"
            if origin:
                response.headers["Access-Control-Allow-Origin"] = origin
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
