import argparse
import json
import math
import os

import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn
import torch.nn.functional as F
from accelerate import Accelerator
from safetensors.torch import load_file
from transformers import CLIPModel, CLIPProcessor, get_cosine_schedule_with_warmup, set_seed

from checkpoint import load_checkpoint, save_checkpoint
from dataset import LoRADataset, lora_collate_fn
from lora_encoder import LoRAEncoder


class CLIPTextEncoder(torch.nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.model = CLIPModel.from_pretrained(model_name).float()
        self.processor = CLIPProcessor.from_pretrained(model_name)
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        self.model.eval()

    @torch.no_grad()
    def forward(self, texts):
        inputs = self.processor(
            text=list(texts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        )
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
        features = self.model.get_text_features(**inputs)
        if hasattr(features, "pooler_output"):
            features = features.pooler_output
        return features


class RetrieverModel(torch.nn.Module):
    def __init__(self, args, dtype):
        super().__init__()
        self.task = args.task
        self.loss_type = args.loss_type
        self.lambda_text = args.lambda_text
        self.lambda_img = args.lambda_img
        self.compute_dtype = dtype
        head_mode = args.head_mode
        if head_mode == "auto":
            head_mode = "dual" if args.task == "clipemb" else "single"
        self.lora_encoder = LoRAEncoder(
            embed_dim=args.embed_dim,
            max_position_embeddings=args.max_position_embeddings,
            num_encoder_layers=args.num_encoder_layers,
            encoder_intermediate_size=args.encoder_intermediate_size,
            L=args.L,
            num_probes=args.num_probes,
            block_type=args.block_type,
            head_mode=head_mode,
            pooling=args.pooling,
            out_dim=args.embed_dim,
            text_out_dim=args.text_out_dim,
            img_out_dim=args.img_out_dim,
        ).to(dtype=dtype)
        self.clip_encoder = CLIPTextEncoder(args.clip_model) if args.task == "clipemb" else None
        self.logit_scale = torch.nn.Parameter(torch.tensor(math.log(1.0 / 0.07), dtype=torch.float32))

    def train(self, mode=True):
        super().train(mode)
        if self.clip_encoder is not None:
            self.clip_encoder.eval()
        return self

    @property
    def device(self):
        return next(self.lora_encoder.parameters()).device

    def encode_loras(self, loras):
        text_features = []
        image_features = []
        for lora in loras:
            lora = {
                key: value.to(self.device, dtype=self.compute_dtype, non_blocking=True)
                for key, value in lora.items()
            }
            output = self.lora_encoder(lora)
            if isinstance(output, tuple):
                text_feature, image_feature = output
            else:
                text_feature = output
                image_feature = output
            text_features.append(text_feature)
            image_features.append(image_feature)
        return torch.cat(text_features, dim=0), torch.cat(image_features, dim=0)

    def encode_batch(self, batch):
        lora_text, lora_image = self.encode_loras(batch["lora"])
        target = batch["target"].to(self.device, dtype=self.compute_dtype, non_blocking=True)
        if self.task == "clipemb":
            text_target = self.clip_encoder(batch["text"]).to(self.device, dtype=self.compute_dtype)
            image_target = target
        else:
            text_target = target
            image_target = None
        return {
            "lora_text": lora_text,
            "lora_image": lora_image,
            "text_target": text_target,
            "image_target": image_target,
            "weight": None if batch["weight"] is None else batch["weight"].to(self.device),
        }

    def contrastive_loss(self, left, right, weights=None):
        left = F.normalize(left, p=2, dim=-1)
        right = F.normalize(right, p=2, dim=-1)
        scale = self.logit_scale.clamp(max=math.log(100.0)).exp()
        logits = left @ right.transpose(0, 1) * scale
        labels = torch.arange(left.size(0), device=left.device)
        loss_left = F.cross_entropy(logits, labels, reduction="none")
        loss_right = F.cross_entropy(logits.transpose(0, 1), labels, reduction="none")
        if weights is not None:
            weights = weights.float() / weights.float().mean().clamp_min(1e-6)
            loss_left = loss_left * weights
            loss_right = loss_right * weights
        return (loss_left.mean() + loss_right.mean()) * 0.5

    def compute_loss(self, encoded):
        if self.task == "gclemb":
            loss = self.contrastive_loss(
                encoded["lora_text"],
                encoded["text_target"],
                encoded["weight"] if self.loss_type == "weighted_contrastive_loss" else None,
            )
            text_cos = F.cosine_similarity(encoded["lora_text"], encoded["text_target"], dim=-1).mean()
            return {"loss": loss, "loss_text": loss, "loss_img": loss.detach() * 0, "cos_text": text_cos}
        text_loss = self.contrastive_loss(encoded["lora_text"], encoded["text_target"])
        image_loss = self.contrastive_loss(encoded["lora_image"], encoded["image_target"])
        loss = self.lambda_text * text_loss + self.lambda_img * image_loss
        text_cos = F.cosine_similarity(encoded["lora_text"], encoded["text_target"], dim=-1).mean()
        image_cos = F.cosine_similarity(encoded["lora_image"], encoded["image_target"], dim=-1).mean()
        return {"loss": loss, "loss_text": text_loss, "loss_img": image_loss, "cos_text": text_cos, "cos_img": image_cos}


def gather_with_grad(value):
    if not dist.is_available() or not dist.is_initialized():
        return value
    return torch.cat(dist_nn.all_gather(value), dim=0)


def gather_without_grad(value):
    if not dist.is_available() or not dist.is_initialized():
        return value
    outputs = [torch.empty_like(value) for _ in range(dist.get_world_size())]
    dist.all_gather(outputs, value)
    return torch.cat(outputs, dim=0)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--task", choices=["clipemb", "gclemb"], required=True)
    parser.add_argument("--lora-root", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--gcl-emb-root", default=None)
    parser.add_argument("--diff-emb-root", default=None)
    parser.add_argument("--clip-model", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--init-checkpoint", default=None)
    parser.add_argument("--L", type=int, default=1)
    parser.add_argument("--embed-dim", type=int, default=768)
    parser.add_argument("--text-out-dim", type=int, default=None)
    parser.add_argument("--img-out-dim", type=int, default=None)
    parser.add_argument("--max-position-embeddings", type=int, default=150)
    parser.add_argument("--encoder-intermediate-size", type=int, default=2560)
    parser.add_argument("--num-encoder-layers", type=int, default=4)
    parser.add_argument("--num-probes", type=int, default=4)
    parser.add_argument("--block-type", choices=["block", "block2"], default="block")
    parser.add_argument("--head-mode", choices=["auto", "single", "dual"], default="auto")
    parser.add_argument("--pooling", choices=["cls", "cls_mean"], default="cls")
    parser.add_argument("--loss-type", choices=["contrastive_loss", "weighted_contrastive_loss"], default=None)
    parser.add_argument("--lambda-text", type=float, default=1.0)
    parser.add_argument("--lambda-img", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging-steps", type=int, default=20)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.task == "clipemb" and not args.clip_model:
        raise ValueError("--clip-model is required for clipemb")
    if args.loss_type is None:
        args.loss_type = "weighted_contrastive_loss" if args.task == "gclemb" else "contrastive_loss"
    set_seed(args.seed)
    accelerator = Accelerator()
    if args.dtype == "bf16" and accelerator.device.type == "cuda":
        dtype = torch.bfloat16
    elif args.dtype == "fp16" and accelerator.device.type == "cuda":
        dtype = torch.float16
    else:
        dtype = torch.float32
    dataset = LoRADataset(
        metadata_path=args.metadata,
        task=args.task,
        lora_root=args.lora_root,
        cache_dir=args.cache_dir,
        gcl_emb_root=args.gcl_emb_root,
        diff_emb_root=args.diff_emb_root,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=lora_collate_fn,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    model = RetrieverModel(args, dtype)
    if args.init_checkpoint:
        config, lora_state, logit_scale, _ = load_checkpoint(args.init_checkpoint)
        model.lora_encoder.load_state_dict(lora_state, strict=True)
        if logit_scale is not None:
            model.logit_scale.data.copy_(logit_scale)
        args.init_config = config
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    total_steps = args.num_epochs * max(1, len(dataloader))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * args.warmup_ratio),
        num_training_steps=total_steps,
    )
    output_dir = os.path.abspath(args.output_dir)
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as handle:
            json.dump(vars(args), handle, ensure_ascii=False, indent=2)
    accelerator.wait_for_everyone()
    step_id = 0
    for epoch in range(args.num_epochs):
        model.train()
        for batch in dataloader:
            with accelerator.accumulate(model):
                raw_model = model.module if hasattr(model, "module") else model
                local = raw_model.encode_batch(batch)
                if args.task == "gclemb":
                    global_encoded = dict(local)
                    global_encoded["lora_text"] = gather_with_grad(local["lora_text"])
                    global_encoded["text_target"] = gather_without_grad(local["text_target"])
                    global_encoded["weight"] = None if local["weight"] is None else gather_without_grad(local["weight"])
                else:
                    global_encoded = dict(local)
                    global_encoded["lora_text"] = gather_with_grad(local["lora_text"])
                    global_encoded["lora_image"] = gather_with_grad(local["lora_image"])
                    global_encoded["text_target"] = gather_without_grad(local["text_target"])
                    global_encoded["image_target"] = gather_without_grad(local["image_target"])
                outputs = raw_model.compute_loss(global_encoded)
                accelerator.backward(outputs["loss"])
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            step_id += 1
            if step_id % args.logging_steps == 0:
                accelerator.print(
                    f"epoch={epoch + 1} step={step_id} loss={outputs['loss'].item():.5f} "
                    f"text={outputs['loss_text'].item():.5f} cos={outputs['cos_text'].item():.5f}"
                )
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            raw_model = accelerator.unwrap_model(model)
            save_checkpoint(
                os.path.join(output_dir, f"retriever-epoch-{epoch + 1:04d}.safetensors"),
                raw_model.lora_encoder,
                raw_model.logit_scale,
            )
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        raw_model = accelerator.unwrap_model(model)
        save_checkpoint(os.path.join(output_dir, "retriever-final.safetensors"), raw_model.lora_encoder, raw_model.logit_scale)


if __name__ == "__main__":
    main()
