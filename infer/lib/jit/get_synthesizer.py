import torch


def get_synthesizer(
    pth_path: str,
    device: torch.device = torch.device("cpu"),
):
    from infer.lib.infer_pack.models import (
        SynthesizerTrnMs768NSFsid,
        SynthesizerTrnMs768NSFsid_nono,
    )

    cpt = torch.load(pth_path, map_location=torch.device("cpu"))

    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]
    if_f0 = cpt.get("f0", 1)
    if if_f0 == 1:
        net_g = SynthesizerTrnMs768NSFsid(*cpt["config"], is_half=False)
    else:
        net_g = SynthesizerTrnMs768NSFsid_nono(*cpt["config"])
    del net_g.enc_q

    net_g.load_state_dict(cpt["weight"], strict=False)
    net_g = net_g.float()
    net_g.eval().to(device)
    net_g.remove_weight_norm()
    return net_g, cpt
