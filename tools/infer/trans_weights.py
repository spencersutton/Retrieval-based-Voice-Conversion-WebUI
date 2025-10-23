import torch

a = torch.load(
    r"E:\codes\py39\vits_vc_gpu_train\logs\ft-mi-no_opt-no_dropout\G_1000.pth"
)["model"]  # sim_nsf#
for key in a.keys():
    a[key] = a[key].half()


torch.save(a, "ft-mi-no_opt-no_dropout.pt")  #
