from collections.abc import Callable

import git
import gradio as gr
import torch
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

import shared
from tabs.ckpt_processing_tab import create_ckpt_processing_tab
from tabs.inference_tab import create_inference_tab
from tabs.train.train_tab import create_train_tab
from tabs.vocal_tab import create_vocal_tab

# Try to import the Dictionary class in a way compatible with different fairseq versions,
# and only register it with torch.serialization if the import succeeds.
try:
    from fairseq.data.dictionary import Dictionary as FairseqDictionary
except Exception:
    try:
        from fairseq.data import Dictionary as FairseqDictionary
    except Exception:
        FairseqDictionary = None

if FairseqDictionary is not None:
    torch.serialization.add_safe_globals([FairseqDictionary])


if __name__ == "__main__":
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

            # Use FastAPI's CORSMiddleware to properly handle preflight and CORS headers.
            fastapi_app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],  # or restrict to specific origins if needed
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )

            @fastapi_app.middleware("http")
            async def _cors_and_private_network(request: Request, call_next: Callable):
                response = await call_next(request)
                response.headers["Access-Control-Allow-Private-Network"] = "true"
                origin = request.headers.get("origin")
                if origin:
                    response.headers["Access-Control-Allow-Origin"] = origin
                    response.headers["Vary"] = "Origin"
                    response.headers["Access-Control-Allow-Credentials"] = "true"
                return response

            # mount the Gradio Blocks app into FastAPI and run with uvicorn
            gr.mount_gradio_app(fastapi_app, app, path="/gradio")

            # Redirect root URL to /gradio
            from fastapi.responses import RedirectResponse

            @fastapi_app.get("/")
            async def redirect_to_gradio():
                return RedirectResponse(url="/gradio")

            import uvicorn

            print(f"Listening on port http://0.0.0.0:{shared.config.listen_port}")

            uvicorn.run(
                fastapi_app,
                host="0.0.0.0",
                port=shared.config.listen_port,
                log_level="warning",
                access_log=False,
            )
