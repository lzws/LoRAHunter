from sympy import flatten
import torch
import torch.nn as nn
import torch.nn.functional as F


# 每一列是独立的


def default_lora_patterns(num_blocks=60):
    lora_patterns = []
    lora_dict = {
        "attn.add_k_proj": (3072, 3072),
        "attn.add_q_proj": (3072, 3072),
        "attn.add_v_proj": (3072, 3072),
        "attn.to_add_out": (3072, 3072),
        "attn.to_k": (3072, 3072),
        "attn.to_out.0": (3072, 3072),
        "attn.to_q": (3072, 3072),
        "attn.to_v": (3072, 3072),
        "img_mlp.net.2": (12288, 3072),
        "img_mod.1": (3072, 18432),
        "txt_mlp.net.2": (12288, 3072),
        "txt_mod.1": (3072, 18432),
    }

    for i in range(num_blocks):
        for suffix in lora_dict:
            lora_patterns.append({
                "name": f"transformer_blocks.{i}.{suffix}",
                "dim": lora_dict[suffix],   # (out_dim, in_dim)
                "type": suffix,
                "block_id": i,
            })
    return lora_patterns



def shape_key(out_dim, in_dim, max_rank):
    return f"{out_dim}x{in_dim}_r{max_rank}"

def pad_lora_AB(A, B, trget_rank):
    """
    A: [r, in_dim]
    B: [out_dim, r]
    return:
        A_pad: [target_rank, in_dim]
        B_pad: [out_dim, target_rank]
        orig_rank: int
    """
    r, in_dim = A.shape
    out_dim, r2 = B.shape
    assert r == r2
    assert r <= trget_rank

    A_pad = torch.zeros(target_rank, in_dim, dtype=A.dtype, device=A.device)
    B_pad = torch.zeros(out_dim, target_rank, dtype=B.dtype, device=B.device)

    A_pad[:r, :] = A
    B_pad[:, :r] = B

    return A_pad, B_pad, r

def flatten_A(A):
    """
    A: [r, in_dim]
    return: [r * in_dim]
    """
    return A.reshape(-1)

def flatten_B(B):
    """
    B: [out_dim, r]
    return: [out_dim * r]
    """
    return B.reshape(-1)

def unflatten_A(x, rank, in_dim):
    return x.view(rank, in_dim)

def unflatten_B(x, out_dim, rank):
    return x.view(out_dim, rank)

def flattern_lora_AB(A, B):
    """
    A: [r, in_dim]
    B: [out_dim_r]
    """
    vec_B = B.reshape(-1)
    vec_A = A.transpose(0, 1).reshape(-1)
    return torch.cat([vec_B, vec_A], dim=0)


def unflatten_lora_AB(x, out_dim, in_dim, rank):
    """
    x: [out_dim * rank + in_dim * rank]
    """

    len_B = out_dim * rank
    len_A = in_dim * rank

    vec_B = x[:len_B]
    vec_A = x[len_B:len_B + len_A]

    B = vec_B.view(out_dim, rank)
    A_t = vec_A.View(in_dim, rank)
    A = A_t.transpose(0, 1).contiguous()
    return A, B


def sample_to_ab_layer_inputs(sample, lora_patterns):
    """
    sample: lora state dict
    return:
        list of dicts:
            {
                "A":[A_dim],
                "B":[B_dim]
            }
    """
    outputs = []
    for p in lora_patterns:
        name = p["name"]
        A = sample[name]['A']
        B = sample[name]['B']

        outputs.append({
            "A": flatten_A(A),
            "B": flatten_B(B)
        })
    return outputs


def collate_ab_lora_samples(samples, lora_patterns):
    """
    return:
        layer_inputs: list of len L
            each item:
                {
                    "A": [B, A_dim],
                    "B": [B, B_dim]
                }
    """  
    per_sample = [sample_to_ab_layer_inputs(s, lora_patterns) for s in samples]
    num_layers = len(lora_patterns)

    batch_layer_inputs = []
    for i in range(num_layers):
        A_list = [per_sample[b][i]["A"] for b in range(len(samples))]
        B_list = [per_sample[b][i]["B"] for b in range(len(samples))]

        batch_layer_inputs.append({
            "A": torch.stack(A_list, dim=0),
            "B": torch.stack(B_list, dim=0),
        })
    return batch_layer_inputs

def move_ab_layer_inputs_to_device(layer_inputs, device):
    outputs = []
    for item in layer_inputs:
        outputs.append({
            "A": item["A"].to(device).float(),
            "B": item["B"].to(device).float(),
        })
    return outputs

def extract_lora_sample_from_state_dict(state_dict, lora_patterns):
    sample = {}
    for p in lora_patterns:
        name = p["name"]
        A_key = f"{name}.lora_A.weight"
        B_key = f"{name}.lora_B.weight"

        if A_key not in state_dict or B_key not in state_dict:
            raise KeyError(f"Missing LoRA keys for layer: {name}")

        A = state_dict[A_key].detach().float()   # [r, in_dim]
        B = state_dict[B_key].detach().float()   # [out_dim, r]

        sample[name] = {"A": A, "B": B}
    return sample


def sample_to_ab_layer_inputs_with_padding(sample, lora_patterns, max_rank):
    """
    sample[name] = {
        "A": [r, in_dim],
        "B": [out_dim, r]
    }

    return:
        list of dicts:
        {
            "A": [max_rank * in_dim],
            "B": [out_dim * max_rank],
            "rank": int
        }
    """
    outputs = []
    for p in lora_patterns:
        name = p["name"]
        A = sample[name]["A"]
        B = sample[name]["B"]

        A_pad, B_pad, r = pad_lora_AB(A, B, target_rank=max_rank)

        outputs.append({
            "A": flatten_A(A_pad),
            "B": flatten_B(B_pad),
            "rank": r,
        })
    return outputs


def collate_ab_lora_samples_with_padding(samples, lora_patterns, max_rank):
    per_sample = [
        sample_to_ab_layer_inputs_with_padding(s, lora_patterns, max_rank)
        for s in samples
    ]
    num_layers = len(lora_patterns)

    batch_layer_inputs = []
    for i in range(num_layers):
        A_list = [per_sample[b][i]["A"] for b in range(len(samples))]
        B_list = [per_sample[b][i]["B"] for b in range(len(samples))]
        rank_list = [per_sample[b][i]["rank"] for b in range(len(samples))]

        batch_layer_inputs.append({
            "A": torch.stack(A_list, dim=0),
            "B": torch.stack(B_list, dim=0),
            "rank": torch.tensor(rank_list, dtype=torch.long),
        })

    return batch_layer_inputs
