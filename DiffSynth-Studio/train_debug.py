import os
import math
import json
import argparse
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn

from accelerate import Accelerator, DistributedDataParallelKwargs
from transformers import get_cosine_schedule_with_warmup
from safetensors.torch import load_file
from safetensors.torch import save_file as save_safetensors

from models import TextImageEncoder
from encoder import LoRAEncoder
from dataaset2 import LoRADataset, lora_collate_fn

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def gather_with_grad(x):
    if x is None:
        return None
    if not dist.is_available() or not dist.is_initialized():
        return x
    gathered = dist_nn.all_gather(x)
    return torch.cat(gathered, dim=0)


class PromptAdapter(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: Optional[int] = None, residual: bool = True):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = out_dim * 2

        self.residual = residual and (in_dim == out_dim)
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net(x)
        if self.residual:
            y = y + x
        return y


class LoRARetrieverTrainingModel(torch.nn.Module):
    def __init__(
        self,
        task="clipemb",
        L=1,
        dtype=torch.bfloat16,
        embed_dim=768,
        encoder_intermediate_size=2560,
        num_encoder_layers=4,
        num_probes=4,
        lora_encoder_path=None,
        loss_type="contrastive_loss",
        lambda_img=0.1,
        lambda_text=0.1,
        block_type="block",
        head_mode="single",
        pooling="cls",
        prompt_in_dim=768,
        prompt_adapter_hidden_dim=4096,
        prompt_adapter_residual=True,
    ):
        super().__init__()
        self.task = task
        self.loss_type = loss_type
        self.lambda_img = lambda_img
        self.lambda_text = lambda_text
        self.dtype = dtype
        self.device_holder = None

        if task == "clipemb":
            self.clip_encoder = TextImageEncoder().to(dtype=dtype)
        else:
            self.clip_encoder = None

        self.lora_encoder = LoRAEncoder(
            L=L,
            embed_dim=embed_dim,
            encoder_intermediate_size=encoder_intermediate_size,
            num_encoder_layers=num_encoder_layers,
            num_probes=num_probes,
            block_type=block_type,
            head_mode=head_mode,
            pooling=pooling,
        ).to(dtype=dtype)

        self.logit_scale = torch.nn.Parameter(
            torch.tensor(math.log(1.0 / 0.07), dtype=torch.float32)
        )

        if task == "gclemb":
            self.prompt_adapter = PromptAdapter(
                in_dim=prompt_in_dim,
                out_dim=embed_dim,
                hidden_dim=prompt_adapter_hidden_dim,
                residual=prompt_adapter_residual,
            ).to(dtype=dtype)
        else:
            self.prompt_adapter = None

        if lora_encoder_path is not None:
            ckpt = load_file(lora_encoder_path)

            has_prefixed_lora = any(k.startswith("lora_encoder.") for k in ckpt.keys())
            has_prompt_adapter = any(k.startswith("prompt_adapter.") for k in ckpt.keys())
            has_logit_scale = "logit_scale" in ckpt

            if has_prefixed_lora:
                lora_sd = {
                    k[len("lora_encoder."):]: v
                    for k, v in ckpt.items()
                    if k.startswith("lora_encoder.")
                }
                self.lora_encoder.load_state_dict(lora_sd, strict=True)
                print("Load lora_encoder from full retriever ckpt:", lora_encoder_path)
            else:
                self.lora_encoder.load_state_dict(ckpt, strict=True)
                print("Load lora_encoder from lora-only ckpt:", lora_encoder_path)

            self.lora_encoder = self.lora_encoder.to(dtype=dtype)

            if self.task == "gclemb" and self.prompt_adapter is not None and has_prompt_adapter:
                prompt_adapter_sd = {
                    k[len("prompt_adapter."):]: v
                    for k, v in ckpt.items()
                    if k.startswith("prompt_adapter.")
                }
                self.prompt_adapter.load_state_dict(prompt_adapter_sd, strict=True)
                self.prompt_adapter = self.prompt_adapter.to(dtype=dtype)
                print("Load prompt_adapter from full retriever ckpt:", lora_encoder_path)

            if self.task == "gclemb" and has_logit_scale:
                self.logit_scale.data.copy_(ckpt["logit_scale"].to(self.logit_scale.device, dtype=self.logit_scale.dtype))
                print("Load logit_scale from full retriever ckpt:", lora_encoder_path)

                # self.logit_scale = torch.nn.Parameter(
                #     torch.tensor(math.log(1.0 / 0.07), dtype=torch.float32)
                # )



        if self.clip_encoder is not None:
            for param in self.clip_encoder.parameters():
                param.requires_grad = False
            self.clip_encoder.eval()

        self.loss_map = {
            "contrastive_loss": self.contrastive_loss,
            "weighted_contrastive_loss": self.weighted_contrastive_loss,
            "mse_loss": self.mse_regression_loss,
            "mse_raw_loss": self.mse_regression_loss_raw,
            "smooth_l1_regression_loss": self.smooth_l1_regression_loss,
            "cosine_smoothl1_loss": self.cosine_smoothl1_loss,
        }

    def to(self, *args, **kwargs):
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(*args, **kwargs)
        if device is not None:
            self.device_holder = device
        super().to(*args, **kwargs)
        return self

    @property
    def device(self):
        if self.device_holder is not None:
            return self.device_holder
        return next(self.parameters()).device

    def encode_batch(self, batch):
        lora_text = batch["lora_text"]
        diff_vecs = batch["diff_vec"]
        loras = batch["lora"]
        txt_embs = batch["txtemb"]
        weights = batch.get("weight", None)

        # model_files = batch["model_file"]

        # print("+++"*10)
        # for jj in range(8):
        #     print(f"model_file: {model_files[jj]}, des: {lora_text[jj][:150]}")
        # print("+++"*10)

        target_device = self.device
        target_dtype = self.dtype

        lora_text_embs = []
        lora_img_embs = []

        for lora in loras:
            lora = {
                k: v.to(device=target_device, dtype=target_dtype, non_blocking=True)
                for k, v in lora.items()
            }
            lora_emb = self.lora_encoder(lora)
            if isinstance(lora_emb, tuple):
                lora_text_emb, lora_img_emb = lora_emb
                lora_text_embs.append(lora_text_emb)
                lora_img_embs.append(lora_img_emb)
            else:
                lora_text_embs.append(lora_emb)
                lora_img_embs.append(lora_emb)

        lora_text_embs = torch.cat(lora_text_embs, dim=0)
        lora_img_embs = torch.cat(lora_img_embs, dim=0)

        if self.task == "clipemb":
            with torch.no_grad():
                txt_embs = self.clip_encoder.encoding_text(lora_text)
        else:
            txt_embs = txt_embs.to(dtype=target_dtype, device=target_device)

        if self.task == "gclemb":
            txt_embs = self.prompt_adapter(txt_embs)

        if diff_vecs is not None:
            diff_vecs = diff_vecs.to(dtype=target_dtype, device=target_device)

        out = {
            "lora_text_embs": lora_text_embs,
            "lora_img_embs": lora_img_embs,
            "txt_embs": txt_embs,
            "diff_vecs": diff_vecs,
        }

        if weights is not None:
            out["weights"] = weights.to(device=target_device, dtype=torch.float32)

        return out

    def mse_regression_loss(self, emb_a, emb_b):
        emb_a = F.normalize(emb_a, p=2, dim=-1)
        emb_b = F.normalize(emb_b, p=2, dim=-1)
        return F.mse_loss(emb_a, emb_b, reduction="mean")

    def mse_regression_loss_raw(self, emb_a, emb_b):
        return F.mse_loss(emb_a, emb_b, reduction="mean")

    def smooth_l1_regression_loss(self, emb_a, emb_b, beta=1.0):
        emb_a = F.normalize(emb_a, p=2, dim=-1)
        emb_b = F.normalize(emb_b, p=2, dim=-1)
        return F.smooth_l1_loss(emb_a, emb_b, reduction="mean", beta=beta)

    def cosine_smoothl1_loss(self, emb_a, emb_b, alpha=1.0, beta=1.0):
        emb_a = F.normalize(emb_a, p=2, dim=-1)
        emb_b = F.normalize(emb_b, p=2, dim=-1)
        cos_loss = 1.0 - (emb_a * emb_b).sum(dim=-1).mean()
        reg_loss = F.smooth_l1_loss(emb_a, emb_b, reduction="mean", beta=beta)
        return alpha * cos_loss + reg_loss

    def contrastive_loss(self, emb_a, emb_b):
        emb_a = F.normalize(emb_a, p=2, dim=-1)
        emb_b = F.normalize(emb_b, p=2, dim=-1)

        logit_scale = self.logit_scale.clamp(max=math.log(100.0))
        scale = logit_scale.exp()

        logits = (emb_a @ emb_b.t()) * scale

        # print(f"logits: \n {logits}")
        # print("\n")

        labels = torch.arange(emb_a.size(0), device=emb_a.device)

        loss_a2b = F.cross_entropy(logits, labels)
        loss_b2a = F.cross_entropy(logits.t(), labels)
        return (loss_a2b + loss_b2a) / 2

    def weighted_contrastive_loss(self, emb_a, emb_b, weights):
        emb_a = F.normalize(emb_a, p=2, dim=-1)
        emb_b = F.normalize(emb_b, p=2, dim=-1)

        logit_scale = self.logit_scale.clamp(max=math.log(100.0))
        scale = logit_scale.exp()

        logits = (emb_a @ emb_b.t()) * scale
        labels = torch.arange(emb_a.size(0), device=emb_a.device)

        loss_a2b = F.cross_entropy(logits, labels, reduction="none")
        loss_b2a = F.cross_entropy(logits.t(), labels, reduction="none")

        weights = weights.float()
        weights = weights / weights.mean().clamp_min(1e-6)

        return ((loss_a2b * weights).mean() + (loss_b2a * weights).mean()) / 2

    def compute_loss(self, lora_text_embs, lora_img_embs, txt_embs, diff_vecs=None, weights=None):
        if self.loss_type == "weighted_contrastive_loss":
            if weights is None:
                raise ValueError("weighted_contrastive_loss requires weights")

            loss_text = self.weighted_contrastive_loss(lora_text_embs, txt_embs, weights)
            debug_text = self.analyze_similarity(
                lora_text_embs,
                txt_embs,
                prefix="text"
            )
            loss_img = torch.tensor(0.0, device=lora_text_embs.device, dtype=lora_text_embs.dtype)
            loss = loss_text

            cos_text = F.cosine_similarity(
                F.normalize(lora_text_embs, dim=-1),
                F.normalize(txt_embs, dim=-1),
                dim=-1
            ).mean()
            cos_img = torch.tensor(0.0, device=lora_text_embs.device, dtype=lora_text_embs.dtype)
            cos_teacher = torch.tensor(0.0, device=lora_text_embs.device, dtype=lora_text_embs.dtype)
            mse_teacher = torch.tensor(0.0, device=lora_text_embs.device, dtype=lora_text_embs.dtype)

            out = {
                "loss": loss,
                "loss_text": loss_text.detach(),
                "loss_img": loss_img.detach(),
                "cos_text": cos_text.detach(),
                "cos_img": cos_img.detach(),
                "cos_teacher": cos_teacher.detach(),
                "mse_teacher": mse_teacher.detach(),
            }
            out.update(debug_text)
            return out

        loss_text = self.loss_map[self.loss_type](lora_text_embs, txt_embs)
        debug_text = self.analyze_similarity(
                lora_text_embs,
                txt_embs,
                prefix="text"
            )
        if self.lambda_img > 0:
            loss_img = self.loss_map[self.loss_type](lora_img_embs, diff_vecs)
        else:
            loss_img = torch.tensor(0.0, device=lora_text_embs.device, dtype=lora_text_embs.dtype)

        loss = self.lambda_text * loss_text + self.lambda_img * loss_img

        cos_text = F.cosine_similarity(lora_text_embs, txt_embs, dim=-1).mean()
        if self.lambda_img > 0:
            
            cos_img = F.cosine_similarity(lora_img_embs, diff_vecs, dim=-1).mean()
            cos_teacher = F.cosine_similarity(txt_embs, diff_vecs, dim=-1).mean()
            mse_teacher = F.mse_loss(txt_embs, diff_vecs, reduction="mean")
        else:
            cos_img = torch.tensor(0.0, device=lora_text_embs.device, dtype=lora_text_embs.dtype)
            cos_teacher = torch.tensor(0.0, device=lora_text_embs.device, dtype=lora_text_embs.dtype)
            mse_teacher = torch.tensor(0.0, device=lora_text_embs.device, dtype=lora_text_embs.dtype)

        out = {
            "loss": loss,
            "loss_text": loss_text.detach(),
            "loss_img": loss_img.detach(),
            "cos_text": cos_text.detach(),
            "cos_img": cos_img.detach(),
            "cos_teacher": cos_teacher.detach(),
            "mse_teacher": mse_teacher.detach(),
        }
        out.update(debug_text)
        return out



    @torch.no_grad()
    def analyze_similarity(self, emb_a, emb_b, prefix="text"):
        """
        emb_a: [B, D]
        emb_b: [B, D]
        """
        emb_a = F.normalize(emb_a.float(), p=2, dim=-1)
        emb_b = F.normalize(emb_b.float(), p=2, dim=-1)

        logits = torch.matmul(emb_a, emb_b.t())
        labels = torch.arange(logits.size(0), device=logits.device)

        pred_a2b = logits.argmax(dim=1)
        pred_b2a = logits.t().argmax(dim=1)

        acc_a2b = (pred_a2b == labels).float().mean()
        acc_b2a = (pred_b2a == labels).float().mean()

        diag_mean = logits.diag().mean()
        if logits.numel() > logits.shape[0]:
            offdiag_mean = (logits.sum() - logits.diag().sum()) / (logits.numel() - logits.shape[0])
        else:
            offdiag_mean = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

        margin = diag_mean - offdiag_mean

        sim_aa = torch.matmul(emb_a, emb_a.t())
        sim_bb = torch.matmul(emb_b, emb_b.t())

        aa_diag = sim_aa.diag().mean()
        bb_diag = sim_bb.diag().mean()

        if sim_aa.numel() > sim_aa.shape[0]:
            aa_offdiag = (sim_aa.sum() - sim_aa.diag().sum()) / (sim_aa.numel() - sim_aa.shape[0])
        else:
            aa_offdiag = torch.tensor(0.0, device=sim_aa.device, dtype=sim_aa.dtype)

        if sim_bb.numel() > sim_bb.shape[0]:
            bb_offdiag = (sim_bb.sum() - sim_bb.diag().sum()) / (sim_bb.numel() - sim_bb.shape[0])
        else:
            bb_offdiag = torch.tensor(0.0, device=sim_bb.device, dtype=sim_bb.dtype)

        # 每个 query 的 hardest negative
        masked_logits = logits.clone()
        masked_logits.fill_diagonal_(-1e9)
        hardest_neg_mean = masked_logits.max(dim=1).values.mean()

        return {
            f"{prefix}_acc_a2b": acc_a2b.detach(),
            f"{prefix}_acc_b2a": acc_b2a.detach(),
            f"{prefix}_diag": diag_mean.detach(),
            f"{prefix}_offdiag": offdiag_mean.detach(),
            f"{prefix}_margin": margin.detach(),
            f"{prefix}_hardneg": hardest_neg_mean.detach(),

            f"{prefix}_aa_diag": aa_diag.detach(),
            f"{prefix}_aa_offdiag": aa_offdiag.detach(),
            f"{prefix}_bb_diag": bb_diag.detach(),
            f"{prefix}_bb_offdiag": bb_offdiag.detach(),

            # 只取前几个值方便打印
            f"{prefix}_logits_preview": logits[:8, :8].detach().cpu(),
            f"{prefix}_sim_aa_preview": sim_aa[:8, :8].detach().cpu(),
            f"{prefix}_sim_bb_preview": sim_bb[:8, :8].detach().cpu(),
        }


class ModelLogger:
    def __init__(self, output_path):
        self.output_path = output_path

    def on_epoch_end(self, accelerator, model, epoch_id):
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            unwrapped = accelerator.unwrap_model(model)
            os.makedirs(self.output_path, exist_ok=True)

            if unwrapped.task == "gclemb":
                state_dict = {}
                for k, v in unwrapped.lora_encoder.state_dict().items():
                    state_dict[f"lora_encoder.{k}"] = v.detach().cpu()
                for k, v in unwrapped.prompt_adapter.state_dict().items():
                    state_dict[f"prompt_adapter.{k}"] = v.detach().cpu()
                state_dict["logit_scale"] = unwrapped.logit_scale.detach().cpu()
                path = os.path.join(self.output_path, f"retriever-{epoch_id}.safetensors")
                save_safetensors(state_dict, path)
            else:
                state_dict = unwrapped.lora_encoder.state_dict()
                path = os.path.join(self.output_path, f"lora_encoder-{epoch_id}.safetensors")
                accelerator.save(state_dict, path, safe_serialization=True)


def parse_args():
    parser = argparse.ArgumentParser()

    # /shark/zhiwen/LoRAHunter/sd_encoder/train_sd_lora_dataset_20k.jsonl
    # /shark/zhiwen/LoRAHunter/rank_encoder/train_gcl_topk.jsonl
    # /shark/zhiwen/LoRAHunter/DiffSynth-Studio/rank_dataset/train_rank_lora_all.jsonl
    # train_rank_lora_all_filtered.jsonl
    parser.add_argument("--metadata_path", type=str, required=False, default='/shark/zhiwen/LoRAHunter/DiffSynth-Studio/rank_dataset/train_rank_lora_test.jsonl')
    parser.add_argument("--emb_path", type=str, required=False, default='/shark/zhiwen/LoRAHunter/DiffSynth-Studio/Diffimage-2-emb')
    parser.add_argument("--txt_emb_path", type=str, required=False, default='/shark/zhiwen/LoRAHunter/DiffSynth-Studio/train_qwenlora_text_emb_title_des')
    parser.add_argument("--prompt_emb_path", type=str, required=False, default='/shark/zhiwen/LoRAHunter/train_set_prompt_emb_20k_2')
    parser.add_argument("--gcl_emb_path", type=str, required=False, default='/shark/zhiwen/LoRAHunter/train_gcl_promptemb')
    parser.add_argument("--output_dir", type=str, required=False, default="models/lora_encode")
    parser.add_argument("--info", type=str, required=False, default="")

    parser.add_argument("--L", type=int, required=False, default=2)
    parser.add_argument("--embed_dim", type=int, required=False, default=2048)
    parser.add_argument("--encoder_intermediate_size", type=int, required=False, default=3072)
    parser.add_argument("--num_encoder_layers", type=int, required=False, default=10)
    parser.add_argument("--num_probes", type=int, required=False, default=20)
    parser.add_argument("--block_type", type=str, required=False, default="block2")
    parser.add_argument("--head_mode", type=str, required=False, default="single") # single | dual
    parser.add_argument("--pooling", type=str, required=False, default="cls_mean") # cls | cls_mean
    parser.add_argument("--lora_encoder_path", required=False, default=None)

    parser.add_argument("--prompt_in_dim", type=int, default=2048)
    parser.add_argument("--prompt_adapter_hidden_dim", type=int, default=4096)
    parser.add_argument("--prompt_adapter_residual", type=bool, default=True)

    parser.add_argument("--task", type=str, default="qwenemb")  # clipemb | qwenemb | promptemb | gclemb
    parser.add_argument("--torch_dtype", required=False, default="bf16")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--loss_type",
        type=str,
        default="contrastive_loss",
        choices=[
            "contrastive_loss",
            "weighted_contrastive_loss",
            "mse_loss",
            "mse_raw_loss",
            "smooth_l1_regression_loss",
            "cosine_smoothl1_loss",
        ],
    )
    parser.add_argument("--lambda_img", type=float, default=0)
    parser.add_argument("--lambda_text", type=float, default=1)

    parser.add_argument("--warmup_ratio", type=float, default=0.005)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--logging_steps", type=int, default=2)

    return parser.parse_args()


def main():
    args = parse_args()
    accelerator = Accelerator(
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=False)]
    )

    dataset = LoRADataset(
        task=args.task,
        head_mode=args.head_mode,
        metadata_path=args.metadata_path,
        emb_path=args.emb_path,
        txt_emb_path=args.txt_emb_path,
        prompt_emb_path=args.prompt_emb_path,
        gcl_emb_path=args.gcl_emb_path,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=lora_collate_fn,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False,
    )

    dtype = torch.bfloat16 if args.torch_dtype == "bf16" else torch.float32

    model = LoRARetrieverTrainingModel(
        dtype=dtype,
        task=args.task,
        L=args.L,
        embed_dim=args.embed_dim,
        encoder_intermediate_size=args.encoder_intermediate_size,
        num_encoder_layers=args.num_encoder_layers,
        num_probes=args.num_probes,
        lora_encoder_path=args.lora_encoder_path,
        loss_type=args.loss_type,
        lambda_img=args.lambda_img,
        lambda_text=args.lambda_text,
        block_type=args.block_type,
        head_mode=args.head_mode,
        pooling=args.pooling,
        prompt_in_dim=args.prompt_in_dim,
        prompt_adapter_hidden_dim=args.prompt_adapter_hidden_dim,
        prompt_adapter_residual=args.prompt_adapter_residual,
    )

    if args.task == "gclemb":
        optimizer = torch.optim.AdamW(
            [
                {"params": model.lora_encoder.parameters(), "weight_decay": args.weight_decay},
                {"params": model.prompt_adapter.parameters(), "weight_decay": args.weight_decay},
                {"params": [model.logit_scale], "weight_decay": 0.0},
            ],
            lr=args.lr,
        )
    elif args.loss_type in ["contrastive_loss", "weighted_contrastive_loss"]:
        optimizer = torch.optim.AdamW(
            [
                {"params": model.lora_encoder.parameters(), "weight_decay": args.weight_decay},
                {"params": [model.logit_scale], "weight_decay": 0.0},
            ],
            lr=args.lr,
        )
    else:
        optimizer = torch.optim.AdamW(
            model.lora_encoder.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

    train_set_name = args.metadata_path.split("/")[-1].split(".")[0]
    model_save_path = f"models/lora_encoder/{args.task}/{train_set_name}/{args.loss_type}"
    os.makedirs(model_save_path, exist_ok=True)
    model_ids = len(os.listdir(model_save_path))

    if accelerator.is_main_process:
        p = f"{model_save_path}/{model_ids}"
        os.makedirs(p, exist_ok=True)
        with open(f"{p}/config.json", "w", encoding="utf-8") as f:
            json.dump(args.__dict__, f, ensure_ascii=False, indent=2)

    model_logger = ModelLogger(f"{model_save_path}/{model_ids}")

    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    num_update_steps_per_epoch = math.ceil(len(dataloader))
    max_train_steps = args.num_epochs * num_update_steps_per_epoch
    num_warmup_steps = int(args.warmup_ratio * max_train_steps)

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=max_train_steps,
    )

    if accelerator.is_main_process:
        print("=" * 80)
        print("[Training Config]")
        print(f"metadata_path: {args.metadata_path}")
        print(f"task: {args.task}")
        print(f"loss_type: {args.loss_type}")
        print(f"gcl_emb_path: {args.gcl_emb_path}")
        print(f"dataset size: {len(dataset)}")
        print(f"per-device batch size: {args.batch_size}")
        print(f"num processes (gpus): {accelerator.num_processes}")
        print(f"global batch size: {args.batch_size * accelerator.num_processes * accelerator.gradient_accumulation_steps}")
        print(f"num_epochs: {args.num_epochs}")
        print(f"max_train_steps: {max_train_steps}")
        print(f"warmup_steps: {num_warmup_steps}")
        print(f"lr: {args.lr}")
        print(f"embed_dim: {args.embed_dim}")
        print(f"prompt_in_dim: {args.prompt_in_dim}")
        print(f"prompt_adapter_hidden_dim: {args.prompt_adapter_hidden_dim}")
        print(f"save path: {model_save_path}/{model_ids}")
        print("=" * 80)

    global_step = 0
    for epoch in range(args.num_epochs):
        model.train()

        for step, batch in enumerate(dataloader):
            with accelerator.accumulate(model):
                encoded = model.module.encode_batch(batch) if hasattr(model, "module") else model.encode_batch(batch)

                local_lora_text_embs = encoded["lora_text_embs"]
                local_lora_img_embs = encoded["lora_img_embs"]
                local_txt_embs = encoded["txt_embs"]
                local_diff_vecs = encoded["diff_vecs"]
                local_weights = encoded.get("weights", None)

                raw_model = model.module if hasattr(model, "module") else model

                if args.loss_type == "weighted_contrastive_loss":
                    global_lora_text_embs = gather_with_grad(local_lora_text_embs)
                    global_txt_embs = gather_with_grad(local_txt_embs)
                    global_weights = gather_with_grad(local_weights)

                    outputs = raw_model.compute_loss(
                        global_lora_text_embs,
                        global_lora_text_embs,
                        global_txt_embs,
                        diff_vecs=None,
                        weights=global_weights,
                    )

                elif args.loss_type == "contrastive_loss":
                    global_lora_text_embs = gather_with_grad(local_lora_text_embs)
                    # global_lora_img_embs = gather_with_grad(local_lora_img_embs)
                    global_txt_embs = gather_with_grad(local_txt_embs)
                    # global_diff_vecs = gather_with_grad(local_diff_vecs)

                    outputs = raw_model.compute_loss(
                        global_lora_text_embs,
                        global_lora_text_embs,
                        global_txt_embs,
                        diff_vecs=None,
                    )
                else:
                    outputs = raw_model.compute_loss(
                        local_lora_text_embs,
                        local_lora_img_embs,
                        local_txt_embs,
                        local_diff_vecs,
                    )

                loss = outputs["loss"]

                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            global_step += 1
            if global_step % args.logging_steps == 0:
                msg = (
                    f"Epoch [{epoch+1}/{args.num_epochs}] "
                    f"Step [{global_step}/{max_train_steps}] "
                    f"Loss: {outputs['loss'].item():.4f} "
                    f"Text: {outputs['loss_text'].item():.4f} "
                    f"Img: {outputs['loss_img'].item():.4f} "
                    f"LR: {lr_scheduler.get_last_lr()[0]:.8f} "
                    f"LR_opt: {optimizer.param_groups[0]['lr']:.8f} | "
                    f"CosT: {outputs['cos_text'].item():.8f} "
                    f"CosI: {outputs['cos_img'].item():.8f} "
                    f"CosTI: {outputs['cos_teacher'].item():.8f} | "
                    f"MSE_TI: {outputs['mse_teacher'].item():.8f}"
                )

                if "text_acc_a2b" in outputs:
                    msg += (
                        f" | AccT: {outputs['text_acc_a2b'].item():.4f}/{outputs['text_acc_b2a'].item():.4f}"
                        f" DiagT: {outputs['text_diag'].item():.4f}"
                        f" OffT: {outputs['text_offdiag'].item():.4f}"
                        f" MarginT: {outputs['text_margin'].item():.4f}"
                        f" HardNegT: {outputs['text_hardneg'].item():.4f}"
                        f" | LL_off: {outputs['text_aa_offdiag'].item():.4f}"
                        f" TT_off: {outputs['text_bb_offdiag'].item():.4f}"
                    )

                accelerator.print(msg)

        if epoch % 2 == 0 or epoch == args.num_epochs - 1:
            model_logger.on_epoch_end(accelerator, model, epoch)


if __name__ == "__main__":
    main()
# nohup accelerate launch --num_processes 2 --gpu_ids 0,1,2,3,4,5,6,7,8 --main_process_port=29501 train_debug.py > zlog/train_gclemb/qwenemb_gcl_all.log 2>&1 &