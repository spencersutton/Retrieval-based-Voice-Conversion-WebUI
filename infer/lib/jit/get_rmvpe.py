import torch

from infer.lib.rmvpe import E2E


def get_rmvpe(model_path="assets/rmvpe/rmvpe.pt", device=None):
    if device is None:
        device = torch.device("cpu")

    model = E2E(4, 1, (2, 2))
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt)
    model.eval()
    model = model.to(device)
    return model
