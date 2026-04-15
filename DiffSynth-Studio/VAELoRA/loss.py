import torch
import torch.nn.functional as F
from utils import *

def multi_layer_ab_lora_vae_loss(
    model,
    layer_inputs,
    recon_outputs,
    mu,
    logvar,
    beta_kl=1e-6,
    beta_wp=0.1,
    beta_delta=0.0,
):
    num_layers = len(layer_inputs)

    total_A_recon = 0.0
    total_B_recon = 0.0
    total_A_wp = 0.0
    total_B_wp = 0.0
    total_delta = 0.0

    for i, (inp, out) in enumerate(zip(layer_inputs, recon_outputs)):
        A = inp["A"]
        B = inp["B"]
        A_hat = out["A"]
        B_hat = out["B"]

        A_recon = F.mse_loss(A_hat, A, reduction="mean")
        B_recon = F.mse_loss(B_hat, B, reduction="mean")

        A_wp = (A.abs() * (A - A_hat).abs()).mean()
        B_wp = (B.abs() * (B - B_hat).abs()).mean()

        total_A_recon += A_recon
        total_B_recon += B_recon
        total_A_wp += A_wp
        total_B_wp += B_wp

        if beta_delta > 0:
            p = model.lora_patterns[i]
            out_dim, in_dim = p['dim']
            rank = model.rank

            Bsz = A.shape[0]
            for b in range(Bsz):
                A_mat = unflatten_A(A[b], rank, in_dim)
                B_mat = unflatten_B(B[b], out_dim, rank)

                A_hat_mat = unflatten_A(A_hat[b], rank, in_dim)
                B_hat_mat = unflatten_B(B_hat[b], out_dim, rank)

                delta = B_mat @ A_mat
                delta_hat = B_hat_mat @ A_hat_mat
                total_delta += F.mse_loss(delta, delta_hat, reduction="mean")
    
    total_A_recon = total_A_recon / num_layers
    total_B_recon = total_B_recon / num_layers
    total_A_wp = total_A_wp / num_layers
    total_B_wp = total_B_wp / num_layers

    kl = -0.5 * torch.mean(
        torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    )

    loss = recon + beta_wp * wp + beta_kl * kl

    if beta_delta > 0:
        total_delta = total_delta / num_layers
        loss = loss + beta_delta * total_delta
    
    return {
        "loss": loss,
        "recon": recon,
        "A_recon": total_A_recon,
        "B_recon": total_B_recon,
        "wp": wp,
        "A_wp": total_A_wp,
        "B_wp": total_B_wp,
        "kl": kl,
        "delta": total_delta if beta_delta > 0 else torch.tensor(0.0, device=mu.device),
    }
