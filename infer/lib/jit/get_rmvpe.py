from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from infer.lib.rmvpe import E2E


def get_rmvpe(
    model_path: str = "assets/rmvpe/rmvpe.pt",
    device: torch.device = torch.device("cpu"),
) -> "E2E":
    from infer.lib.rmvpe import E2E  # noqa: PLC0415

    model = E2E(4, 1, (2, 2))
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt)
    model.eval()
    model = model.to(device)
    return model
