import torch


def get_rmvpe(
    model_path: str = "assets/rmvpe/rmvpe.pt",
    device: torch.device = torch.device("cpu"),
):
    from infer.lib.rmvpe import E2E

    model = E2E(4, 1, (2, 2))
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt)
    model.eval()
    model = model.to(device)
    return model
