import torch
from loss import multi_layer_ab_lora_vae_loss
from utils import *
def train_one_epoch_ab_vae(
    model,
    dataloader,
    optimizer,
    device,
    beta_kl=1e-6,
    beta_wp=0.1,
    beta_delta=0.05
):
    model.train()

    total = {
        "loss": 0.0,
        "recon": 0.0,
        "A_recon": 0.0,
        "B_recon": 0.0,
        "wp": 0.0,
        "A_wp": 0.0,
        "B_wp": 0.0,
        "kl": 0.0,
        "delta": 0.0
    }

    n = 0

    for layer_inputs in dataloader:
        layer_inputs = move_ab_layer_inputs_to_device(layer_inputs, device)
