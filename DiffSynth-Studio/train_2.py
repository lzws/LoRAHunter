
import torch, os
from accelerate import Accelerator, DistributedDataParallelKwargs
from tqdm import tqdm
from transformers import CLIPTokenizer, CLIPModel, get_cosine_schedule_with_warmup
import pandas as pd
import random
import torch.nn.functional as F
from encoder import TextImageEncoder
from LoRAEncoder import LoRAEncoder
from FusionEncoder import FusionEncoder
from dataset import LoRADataset
from diffsynth.core import load_state_dict
import math
import argparse
import os,json
from diffsynth.utils.lora import GeneralLoRALoader
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class CombineEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_encoder = LoRAEncoder()
        self.fusion_encoder = FusionEncoder()


    def to(self, *args, **kwargs):
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(*args, **kwargs)
        if device is not None:
            self.device = device
        if dtype is not None:
            self.torch_dtype = dtype
        super().to(*args, **kwargs)
        return self
    
    def forward(self, lora_paths, txt_embs, diff_vecs):
        lora_embs = []
        if isinstance(lora_paths, str):
            lora_paths = [lora_paths]

        for lora_path in lora_paths:
            lora = load_state_dict(lora_path, torch_dtype=txt_embs.dtype, device=txt_embs.device)
            lora_emb = self.lora_encoder(lora) # [1, 768]
            lora_embs.append(lora_emb)
        lora_embs = torch.cat(lora_embs, dim=0) # [B, 768]

        fusion_emb = self.fusion_encoder(txt_embs, diff_vecs, lora_embs) # [B, 768]
        return fusion_emb


class LoRARetrieverTrainingModel(torch.nn.Module):
    def __init__(self,L=1,dtype=torch.float):
        super().__init__()
        self.clip_encoder = TextImageEncoder().to(dtype=dtype)
        self.lora_encoder = LoRAEncoder(L=L)
    
        for param in self.clip_encoder.parameters():
            param.requires_grad = False
        self.lora_loader = GeneralLoRALoader()

    def to(self, *args, **kwargs):
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(*args, **kwargs)
        if device is not None:
            self.device = device
        if dtype is not None:
            self.torch_dtype = dtype
        super().to(*args, **kwargs)
        return self
        
    def forward(self, batch):
        lora_paths = batch['model_file']
        prompts = batch['prompt'] # list([p1,p2..])
        lora_text = batch['lora_text']
        diff_vecs = batch['diff_vec'] # [B,1,768]

        B = len(prompts)

        # target prompts emb
        # prompt_embs = self.clip_encoder.encoding_text(prompts)

        # text and diff_vec
        txt_embs = self.clip_encoder.encoding_text(lora_text) # [B, 768]
        diff_vecs = diff_vecs.squeeze(1).to(dtype=txt_embs.dtype,device=txt_embs.device) # [B, 768]

        prompt_embs = txt_embs + diff_vecs # [B, 768]

        lora_embs=[]
        for lora_path in lora_paths:
            lora = load_state_dict(lora_path, torch_dtype=txt_embs.dtype, device=txt_embs.device)
            
            lora = self.lora_loader.convert_state_dict(lora)
            lora_emb = self.lora_encoder(lora) # [1, 768]
            lora_embs.append(lora_emb)
        lora_embs = torch.cat(lora_embs, dim=0) # [B, 768]

        loss = self.contrastive_loss(lora_embs, prompt_embs)

        return loss


    def contrastive_loss(self,fuse_emb, text_emb, temperature=0.07):
        # 1) 归一化
        fuse_emb = F.normalize(fuse_emb, p=2, dim=-1)
        text_emb = F.normalize(text_emb, p=2, dim=-1)

        # 2) 计算相似度矩阵 [B, B]
        logits = fuse_emb @ text_emb.t() / temperature

        # 3) 构造标签：第 i 个 image 对应第 i 个 text
        labels = torch.arange(fuse_emb.size(0), device=fuse_emb.device)

        # 4) 双向对比损失
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.t(), labels)

        loss = (loss_i2t + loss_t2i) / 2
        return loss


    def trainable_modules(self):
        return self.lora_encoder.parameters()



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

    parser.add_argument("--metadata_path", type=str, required=False, default='/shark/zhiwen/LoRAHunter/DiffSynth-Studio/train_available_lora_dataset.jsonl')
    parser.add_argument("--output_dir", type=str, required=False, default="models/lora_encode")
    parser.add_argument("--info", type=str, required=False, default=", 不用prompt训练，直接用lora emb 和diff_vec做对比学习")

    # lora encoder config
    parser.add_argument("--L", type=int, required=False, default=1)


    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)


    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--logging_steps", type=int, default=2)

    return parser.parse_args()

def main():
    args = parse_args()
    accelerator = Accelerator(kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=False)])
    device = accelerator.device

    dataset = LoRADataset(metadata_path=args.metadata_path)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=args.batch_size, num_workers=args.num_workers)

    model = LoRARetrieverTrainingModel(L=args.L)

    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=args.lr, weight_decay=args.weight_decay,)

    

    train_set_name = args.metadata_path.split('/')[-1].split('.')[0]
    model_save_path = f"models/lora_encoder/{train_set_name}"
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
    lr_scheduler = accelerator.prepare(lr_scheduler)

    global_step = 0
    for epoch in range(args.num_epochs):
        for step, batch in enumerate(dataloader):
            # print(f'data:{data}')
            with accelerator.accumulate(model):
                # 
                loss = model(batch)
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            global_step += 1
            if global_step % args.logging_steps == 0:
                accelerator.print(
                    f"Epoch [{epoch+1}/{args.num_epochs}] "
                    f"Step [{global_step}/{max_train_steps}] "
                    f"Loss: {loss.item():.4f} "
                    f"LR: {lr_scheduler.get_last_lr()[0]:.8f}"
                )
        if epoch % 2 == 0 or epoch==args.num_epochs-1:
            model_logger.on_epoch_end(accelerator, model, epoch) 
        # model_logger.on_epoch_end(accelerator, model, epoch)


if __name__ == "__main__":
    main()
    

# nohup accelerate launch --num_processes 4 --gpu_ids 4,5,6,7 --main_process_port=29502 train_2.py > zlog/train_lora_encoder_cls_avail_1e4.log 2>&1 &