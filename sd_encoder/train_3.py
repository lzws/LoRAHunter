import torch, os
from accelerate import Accelerator, DistributedDataParallelKwargs
from tqdm import tqdm
from transformers import CLIPTokenizer, CLIPModel, get_cosine_schedule_with_warmup
import pandas as pd
import random
import torch.nn.functional as F
from models import TextImageEncoder
from encoder import LoRAEncoder

from dataset import LoRADataset,lora_collate_fn
# from diffsynth.core import load_state_dict
import math
import argparse
import os,json
# from diffsynth.utils.lora import GeneralLoRALoader

import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn
from safetensors.torch import load_file

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class LoRARetrieverTrainingModel(torch.nn.Module):
    def __init__(
        self,
        task='clipemb',
        L=1,
        dtype=torch.bfloat16,
        embed_dim=768,
        encoder_intermediate_size=2560,
        num_encoder_layers=4,
        num_probes=4,
        lora_encoder_path=None,
        loss_type = 'contrastive_loss',
        lambda_img = 0.1,
        lambda_text = 0.1,
        block_type='block',
        head_mode='single',
        pooling='cls'
    ):
        super().__init__()
        self.task = task

        if task == 'clipemb':
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
            pooling=pooling
        ).to(dtype=dtype)
        if lora_encoder_path is not None:
            self.lora_encoder.load_state_dict(load_file(lora_encoder_path))
            self.lora_encoder = self.lora_encoder.to(dtype=dtype)
            print("Load lora encoder from", lora_encoder_path)

        self.logit_scale = torch.nn.Parameter(
                torch.tensor(math.log(1.0 / 0.07), dtype=torch.float)
            )

        if self.clip_encoder is not None:
            for param in self.clip_encoder.parameters():
                param.requires_grad = False
            self.clip_encoder.eval()

        self.lambda_img = lambda_img
        self.lambda_text = lambda_text
        self.dtype = dtype
        self.loss_type = loss_type

        self.loss_map = {
            'contrastive_loss': self.contrastive_loss,
            'mse_loss': self.mse_regression_loss,
            "mse_raw_loss":self.mse_regression_loss_raw,
            'smooth_l1_regression_loss': self.smooth_l1_regression_loss,
            'cosine_smoothl1_loss': self.cosine_smoothl1_loss,
        }

    def to(self, *args, **kwargs):
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(*args, **kwargs)
        if device is not None:
            self.device = device
        if dtype is not None:
            self.torch_dtype = dtype
        super().to(*args, **kwargs)
        return self

    def encode_batch(self, batch):
        lora_text = batch['lora_text']
        diff_vecs = batch['diff_vec']   # [B, 768] or [B,1,768]
        loras = batch['lora']
        txt_embs = batch['txtemb']

        # 1) encode lora
        # lora_embs = []
        target_device = self.device
        target_dtype = self.dtype

        lora_text_embs = []
        lora_img_embs = []

        for lora in loras:
            lora = {
                k: v.to(device=target_device, dtype=target_dtype, non_blocking=True)
                for k, v in lora.items()
            }
            lora_emb = self.lora_encoder(lora)  # [1, D]
            if isinstance(lora_emb, tuple):
                lora_text_emb, lora_img_emb = lora_emb
                lora_text_embs.append(lora_text_emb)
                lora_img_embs.append(lora_img_emb)
            else:
                lora_text_embs.append(lora_emb)
                lora_img_embs.append(lora_emb)

        lora_text_embs = torch.cat(lora_text_embs, dim=0)  # [B, D]
        lora_img_embs = torch.cat(lora_img_embs, dim=0) # [B, D]


        # 2) text emb
        if self.task == 'clipemb':
            with torch.no_grad():
                txt_embs = self.clip_encoder.encoding_text(lora_text)  # [B, D]
        else:
            txt_embs = txt_embs.to(dtype=target_dtype, device=target_device)



        diff_vecs = diff_vecs.to(dtype=target_dtype, device=target_device)

        return {
            # "lora_embs": lora_embs,
            "lora_text_embs": lora_text_embs,
            "lora_img_embs": lora_img_embs,
            "txt_embs": txt_embs,
            "diff_vecs": diff_vecs,
        }
    
    def mse_regression_loss(self, emb_a, emb_b):
        emb_a = F.normalize(emb_a, p=2, dim=-1)
        emb_b = F.normalize(emb_b, p=2, dim=-1)
        return F.mse_loss(emb_a, emb_b, reduction='mean')
    
    def mse_regression_loss_raw(self, emb_a, emb_b):
        return F.mse_loss(emb_a, emb_b, reduction='mean')

    
    def smooth_l1_regression_loss(self, emb_a, emb_b, beta=1.0):
        emb_a = F.normalize(emb_a, p=2, dim=-1)
        emb_b = F.normalize(emb_b, p=2, dim=-1)
        loss = F.smooth_l1_loss(emb_a, emb_b, reduction='mean', beta=beta)
        return loss
    
    def cosine_smoothl1_loss(self, emb_a, emb_b, alpha=1.0, beta=1.0):
        emb_a = F.normalize(emb_a, p=2, dim=-1)
        emb_b = F.normalize(emb_b, p=2, dim=-1)

        cos_loss = 1.0 - (emb_a * emb_b).sum(dim=-1).mean()
        reg_loss = F.smooth_l1_loss(emb_a, emb_b, reduction='mean', beta=beta)

        return alpha * cos_loss + reg_loss
    
    

    def contrastive_loss(self, emb_a, emb_b):
        emb_a = F.normalize(emb_a, p=2, dim=-1)
        emb_b = F.normalize(emb_b, p=2, dim=-1)

        # clamp for stability
        logit_scale = self.logit_scale.clamp(max=math.log(100.0))
        scale = logit_scale.exp()

        logits = (emb_a @ emb_b.t()) * scale
        labels = torch.arange(emb_a.size(0), device=emb_a.device)

        loss_a2b = F.cross_entropy(logits, labels)
        loss_b2a = F.cross_entropy(logits.t(), labels)
        loss = (loss_a2b + loss_b2a) / 2
        return loss

    def compute_loss(self, lora_text_embs, lora_img_embs, txt_embs, diff_vecs):
        # loss_text = self.contrastive_loss(lora_embs, txt_embs)
        # loss_img = self.contrastive_loss(lora_embs, diff_vecs)

        loss_text = self.loss_map[self.loss_type](lora_text_embs, txt_embs)
        loss_img = self.loss_map[self.loss_type](lora_img_embs, diff_vecs)

        loss = self.lambda_text * loss_text + self.lambda_img * loss_img

        # ---------------------------
        # cosine alignment
        # ---------------------------
        cos_text = F.cosine_similarity(lora_text_embs, txt_embs, dim=-1).mean()
        cos_img = F.cosine_similarity(lora_img_embs, diff_vecs, dim=-1).mean()
        cos_teacher = F.cosine_similarity(txt_embs, diff_vecs, dim=-1).mean()

        # ---------------------------
        # mse diagnostics
        # ---------------------------
        mse_teacher = F.mse_loss(txt_embs, diff_vecs, reduction="mean")

        return {
            "loss": loss,
            "loss_text": loss_text.detach(),
            "loss_img": loss_img.detach(),

            "cos_text": cos_text.detach(),
            "cos_img": cos_img.detach(),
            "cos_teacher": cos_teacher.detach(),

            "mse_teacher": mse_teacher.detach(),
        }

    def trainable_modules(self):
        return list(self.lora_encoder.parameters()) + [self.logit_scale]


def gather_with_grad(x):
    if not dist.is_available() or not dist.is_initialized():
        return x
    gathered = dist_nn.all_gather(x)
    return torch.cat(gathered, dim=0)

class ModelLogger:
    def __init__(self, output_path, remove_prefix_in_ckpt=None):
        self.output_path = output_path
        self.remove_prefix_in_ckpt = remove_prefix_in_ckpt
        
    
    def on_step_end(self, loss):
        pass
    
    
    def on_epoch_end(self, accelerator, model, epoch_id):
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            state_dict = accelerator.unwrap_model(model).lora_encoder.state_dict()
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, f"lora_encoder-{epoch_id}.safetensors")
            accelerator.save(state_dict, path, safe_serialization=True)

            # state_dict = accelerator.unwrap_model(model).text_encoder.state_dict()
            # os.makedirs(self.output_path, exist_ok=True)
            # path = os.path.join(self.output_path, f"epoch-text_encoder-{epoch_id}.safetensors")
            # accelerator.save(state_dict, path, safe_serialization=True)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--metadata_path", type=str, required=False, default='/shark/zhiwen/LoRAHunter/sd_encoder/train_sd_lora_dataset_20k.jsonl')
    parser.add_argument("--emb_path", type=str, required=False, default='/shark/zhiwen/LoRAHunter/Diffimage-SD-emb-qwen')
    parser.add_argument("--txt_emb_path", type=str, required=False, default='/shark/zhiwen/LoRAHunter/train_set_txtemb_10k')
    parser.add_argument("--output_dir", type=str, required=False, default="models/lora_encode")
    parser.add_argument("--info", type=str, required=False, default=", 不用prompt训练，直接用lora emb 和diff_vec做对比学习, 8卡全局batch对比学习")

    # lora encoder config
    parser.add_argument("--L", type=int, required=False, default=2)
    parser.add_argument("--embed_dim", type=int, required=False, default=2048) # 768 | 2048
    parser.add_argument("--encoder_intermediate_size", type=int, required=False, default=3072) # 2560 | 3072
    parser.add_argument("--num_encoder_layers", type=int, required=False, default=8) # 8 | 12
    parser.add_argument("--num_probes", type=int, required=False, default=16) # 8 | 16
    parser.add_argument("--block_type", type=str, required=False, default="block2") # block | block2
    parser.add_argument("--head_mode", type=str, required=False, default="dual")  # single | dual
    parser.add_argument("--pooling", type=str, required=False, default="cls_mean")   # cls | cls_mean
    parser.add_argument("--lora_encoder_path", required=False, default=None)

    # training setting
    parser.add_argument("--task", type=str, default="qwenemb") # clipemb ｜ qwenemb  使用clip模型的embedding， 还是用 qwenvl embedding 模型进行训练loraencoder
    parser.add_argument("--torch_dtype", required=False, default="bf16") # bf16 | float
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--loss_type", type=str, default="contrastive_loss") # contrastive_loss | mse_loss | mse_raw_loss | smooth_l1_regression_loss | cosine_smoothl1_loss
    parser.add_argument("--lambda_img", type=float, default=1)
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
    device = accelerator.device
    # lora_encoder_path = "/shark/zhiwen/LoRAHunter/sd_encoder/models/lora_encoder/clipemb/train_sd_lora_dataset_2/4/lora_encoder-149.safetensors"

    dataset = LoRADataset(
        task=args.task,
        metadata_path=args.metadata_path,
        emb_path=args.emb_path,
        txt_emb_path=args.txt_emb_path,
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
    if args.torch_dtype == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float

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
    )

    # optimizer = torch.optim.AdamW(
    #     model.trainable_modules(),
    #     lr=args.lr,
    #     weight_decay=args.weight_decay,
    # )
    if args.loss_type == 'contrastive_loss':
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

    train_set_name = args.metadata_path.split('/')[-1].split('.')[0]
    model_save_path = f"models/lora_encoder/{args.task}/{train_set_name}/{args.loss_type}"
    os.makedirs(model_save_path, exist_ok=True)
    model_ids = len(os.listdir(model_save_path))

    if accelerator.is_main_process:
        p = f"{model_save_path}/{model_ids}"
        os.makedirs(p, exist_ok=True)
        with open(f'{p}/config.json', 'w', encoding='utf-8') as f:
            json.dump(args.__dict__, f)

    model_logger = ModelLogger(f'{model_save_path}/{model_ids}', remove_prefix_in_ckpt="lora_encoder.")

    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    num_update_steps_per_epoch = math.ceil(len(dataloader))
    max_train_steps = args.num_epochs * num_update_steps_per_epoch
    num_warmup_steps = int(args.warmup_ratio * max_train_steps)

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=max_train_steps,
    )
    # lr_scheduler = accelerator.prepare(lr_scheduler)

    if accelerator.is_main_process:
        print("=" * 80)
        print("[Training Config]")
        print(f"metadata_path: {args.metadata_path}")
        print(f"emb_path: {args.emb_path}")
        print(f"txt_emb_path: {args.txt_emb_path}")
        print(f"Dataset size: {len(dataset)}")
        print(f"Per-device batch size: {args.batch_size}")
        print(f"Num processes (GPUs): {accelerator.num_processes}")
        print(f"Global batch size: {args.batch_size * accelerator.num_processes * accelerator.gradient_accumulation_steps}")
        print(f"Gradient accumulation steps: {accelerator.gradient_accumulation_steps}")
        print(f"len(dataloader) per process: {len(dataloader)}")
        print(f"num_update_steps_per_epoch: {num_update_steps_per_epoch}")
        print(f"num_epochs: {args.num_epochs}")
        print(f"max_train_steps: {max_train_steps}")
        print(f"warmup_ratio: {args.warmup_ratio}")
        print(f"num_warmup_steps: {num_warmup_steps}")
        print(f"loss_type: {args.loss_type}")
        print(f"lambda_img: {args.lambda_img}")
        print(f"lambda_text: {args.lambda_text}")
        print(f"block_type: {args.block_type}")
        print(f"torch_dtype: {dtype}")
        print(f"head_mode: {args.head_mode}")
        print(f"pooling: {args.pooling}")
        print(f"base lr: {args.lr}")
        print(f"optimizer lr (init): {optimizer.param_groups[0]['lr']}")
        print(f"save path: {model_save_path}/{model_ids}")
        print("=" * 80)

    global_step = 0
    for epoch in range(args.num_epochs):
        for step, batch in enumerate(dataloader):
            with accelerator.accumulate(model):
                # 1) local encode
                encoded = model.module.encode_batch(batch) if hasattr(model, "module") else model.encode_batch(batch)

                # local_lora_embs = encoded["lora_embs"]
                local_lora_text_embs = encoded["lora_text_embs"]
                local_lora_img_embs = encoded["lora_img_embs"]
                local_txt_embs = encoded["txt_embs"]
                local_diff_vecs = encoded["diff_vecs"]

                raw_model = model.module if hasattr(model, "module") else model
                if args.loss_type == 'contrastive_loss':

                    # 2) global gather with grad
                    # global_lora_embs = gather_with_grad(local_lora_embs)
                    global_lora_text_embs = gather_with_grad(local_lora_text_embs)
                    global_lora_img_embs = gather_with_grad(local_lora_img_embs)
                    global_txt_embs = gather_with_grad(local_txt_embs)
                    global_diff_vecs = gather_with_grad(local_diff_vecs)

                    # 3) compute global contrastive loss
                    outputs = raw_model.compute_loss(
                        global_lora_text_embs,
                        global_lora_img_embs,
                        global_txt_embs,
                        global_diff_vecs,
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
                accelerator.print(
                    f"Epoch [{epoch+1}/{args.num_epochs}] "
                    f"Step [{global_step}/{max_train_steps}] "
                    f"Loss: {outputs['loss'].item():.4f} "
                    f"Text: {outputs['loss_text'].item():.4f} "
                    f"Img: {outputs['loss_img'].item():.4f} "
                    f"LR: {lr_scheduler.get_last_lr()[0]:.8f} "
                    f"LR_opt: {optimizer.param_groups[0]['lr']:.8f} |"
                    f"CosT: {outputs['cos_text'].item():.8f} "
                    f"CosI: {outputs['cos_img'].item():.8f} "
                    f"CosTI: {outputs['cos_teacher'].item():.8f} | "
                    f"MSE_TI: {outputs['mse_teacher'].item():.8f} | "
                )

        if epoch % 2 == 0 or epoch == args.num_epochs - 1:
            model_logger.on_epoch_end(accelerator, model, epoch)


if __name__ == "__main__":
    main()
    
# nohup accelerate launch --num_processes 8 --gpu_ids 0,1,2,3,4,5,6,7,8 --main_process_port=29501 train_3.py > zlog/train_clipemb/clipemb_dataset20k_contrastive_loss.log 2>&1 &
# nohup accelerate launch --num_processes 8 --gpu_ids 0,1,2,3,4,5,6,7,8 --main_process_port=29501 train_3.py > zlog/train_qwenemb/qwenemb_dataset20k_contrastive_loss_blocck2_dual_0.log 2>&1 &