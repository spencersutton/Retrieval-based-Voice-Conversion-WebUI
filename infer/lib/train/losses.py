import torch


def feature_loss(
    fmap_r: list[list[torch.Tensor]], fmap_g: list[list[torch.Tensor]]
) -> torch.Tensor:
    """Calculate feature matching loss between real and generated feature maps.

    Accumulates mean absolute difference between corresponding feature maps.
    """
    loss = torch.tensor(0.0, dtype=torch.float32)
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            rl = rl.float().detach()  # noqa: PLW2901
            gl = gl.float()  # noqa: PLW2901
            loss += torch.mean(torch.abs(rl - gl))

    return loss * 2


def discriminator_loss(
    disc_real_outputs: list[torch.Tensor], disc_generated_outputs: list[torch.Tensor]
) -> torch.Tensor:
    """Calculate discriminator loss using least-squares GAN objective.

    Real outputs should be close to 1, generated outputs should be close to 0.
    """
    loss = torch.tensor(0.0, dtype=torch.float32)
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1 - dr.float()) ** 2)
        g_loss = torch.mean(dg.float() ** 2)
        loss = loss + r_loss + g_loss

    return loss


def generator_loss(
    disc_outputs: list[torch.Tensor],
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    gen_losses: list[torch.Tensor] = []
    for dg in disc_outputs:
        l = torch.mean((1 - dg.float()) ** 2)
        gen_losses.append(l)

    # Sum all losses - avoid accumulating as scalar
    loss = torch.stack(gen_losses).sum() if gen_losses else torch.tensor(0.0)
    return loss, gen_losses


def kl_loss(
    z_p: torch.Tensor,
    logs_q: torch.Tensor,
    m_p: torch.Tensor,
    logs_p: torch.Tensor,
    z_mask: torch.Tensor,
) -> torch.Tensor:
    """
    z_p, logs_q: [b, h, t_t]
    m_p, logs_p: [b, h, t_t]
    """
    z_p = z_p.float()
    logs_q = logs_q.float()
    m_p = m_p.float()
    logs_p = logs_p.float()
    z_mask = z_mask.float()

    kl = logs_p - logs_q - 0.5
    kl += 0.5 * ((z_p - m_p) ** 2) * torch.exp(-2.0 * logs_p)
    kl = torch.sum(kl * z_mask)
    l = kl / torch.sum(z_mask)
    return l
