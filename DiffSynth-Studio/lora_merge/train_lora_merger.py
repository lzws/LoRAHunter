import os,json,math

from torch._C import device

from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
import torch
import torch.nn as nn
import os 
import pandas as pd
from torch.utils.data import DataLoader, dataset
from collections import defaultdict
import torch.distributed as dist
from LoRAFusion import LoRAMerger,replace_target_modules_with_lora_merger,load_lora,clear_lora
from diffsynth.diffusion.loss import FlowMatchSFTLoss
import argparse
from accelerate import Accelerator, DistributedDataParallelKwargs
from transformers import get_cosine_schedule_with_warmup
from PIL import Image
from torchvision.transforms import v2
import random
from diffsynth.core.data.operators import LoadImage, ImageCropAndResize
import gc
# 训练融合模块
# 每次加载10个lora进行训练，用其中一个lora的text和图片作为监督信号

# 1. 构建数据集
class LoRAMergerDataset(torch.utils.data.Dataset):
    def __init__(self, load_lora_nums = 10,base_path="/shark/zhiwen/LoRAHunter/DiffSynth-Studio/models/train/Qwen-Image_lora", metadata_path="/shark/zhiwen/LoRAHunter/DiffSynth-Studio/LoRA_Merger_train_data.csv"):
        metadatas = pd.read_csv(metadata_path)
        self.base_path = base_path
        self.device='cpu'
        self.load_lora_nums = load_lora_nums

        self.lora_name = metadatas['lora_name'].tolist()
        # 预先建立 name -> 所有对应索引
        self.name_to_indices = defaultdict(list)
        for idx, name in enumerate(self.lora_name):
            self.name_to_indices[name].append(idx)
        self.unique_names = list(self.name_to_indices.keys())

        self.image = metadatas['image'].tolist()
        self.prompt = metadatas['prompt'].tolist()
        self.model_file = metadatas['model_file'].tolist()

        self.max_resolution = 1920 * 1080


    
    def sample_other_name_indices(self, target_name, k=9, step=0):
        rank = dist.get_rank() if dist.is_initialized() else 0

        candidate_names = [name for name in self.unique_names if name != target_name]

        if len(candidate_names) < k:
            raise ValueError(f"不同name数量不足: 需要{k}个, 但只有{len(candidate_names)}个")

        g = torch.Generator(device=self.device)
        g.manual_seed(42 + rank * 100000 + step)

        perm = torch.randperm(len(candidate_names), generator=g, device=self.device)
        selected_name_idx = perm[:k].tolist()
        sampled_names = [candidate_names[i] for i in selected_name_idx]

        sampled_indices = []
        for name in sampled_names:
            indices = self.name_to_indices[name]
            rand_idx = torch.randint(
                0, len(indices), (1,), generator=g, device=self.device
            ).item()
            sampled_indices.append(indices[rand_idx])

        return sampled_indices

    def __getitem__(self, index):

        tar_lora = self.lora_name[index]
        tar_image = self.image[index]
        tar_prompt = self.prompt[index]
        tar_model_file = os.path.join(self.base_path, self.model_file[index])

        other_indices = self.sample_other_name_indices(tar_lora,k=self.load_lora_nums-1,step=index)

        model_files = [tar_model_file]+[os.path.join(self.base_path, self.model_file[i]) for i in other_indices]

        image = LoadImage()(tar_image)
        image = ImageCropAndResize(None, None, 1920 * 1080, 16, 16)(image)

        return {
            "tar_lora": tar_lora, #[B]
            "tar_image": image,
            "tar_prompt": tar_prompt,
            "tar_model_file": tar_model_file,
            "other_indices": other_indices,
            "model_files": model_files #[10,B]
        }

    def __len__(self):
        return len(self.lora_name)
        

class LoRAMergerTrainingModel(torch.nn.Module):
    def __init__(self,device=device):
        super().__init__()
        # load qwen pipe
        self.pipe = QwenImagePipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device=device,
            model_configs=[
                ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
                ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors"),
                ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
            ],
            tokenizer_config=ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="tokenizer/"),
        )

        # Training mode
        self.pipe.scheduler.set_timesteps(1000, training=True)
        self.pipe.freeze_except([])

        # add lora merger layer
        moelora_target_modules = "to_q,to_k,to_v,add_q_proj,add_k_proj,add_v_proj,to_out.0,to_add_out,img_mlp.net.2,img_mod.1,txt_mlp.net.2,txt_mod.1"
        model_ = replace_target_modules_with_lora_merger(
            getattr(self.pipe, 'dit'),
            target_modules=moelora_target_modules.split(","),
        )
        setattr(self.pipe, 'dit', model_.to(self.pipe.device,self.pipe.torch_dtype))

        self.use_gradient_checkpointing = False
        self.use_gradient_checkpointing_offload = False
        self.extra_inputs=[]

        self.task_to_loss = {
            "sft": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
        }


    def to(self, *args, **kwargs):
        for name, model in self.named_children():
            model.to(*args, **kwargs)
        return self


    def trainable_modules(self):

        trainable_modules = filter(lambda p: p.requires_grad, self.parameters())
        return trainable_modules


    def trainable_param_names(self):
        trainable_param_names = list(filter(lambda named_param: named_param[1].requires_grad, self.named_parameters()))
        trainable_param_names = set([named_param[0] for named_param in trainable_param_names])
        return trainable_param_names

    def get_pipeline_inputs(self, data):
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {"negative_prompt": ""}
        inputs_shared = {
            # Please do not modify the following parameters
            # unless you clearly know what this will cause.
            "cfg_scale": 1,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "edit_image_auto_resize": True,
            "zero_cond_t": False
        }
        # Assume you are using this pipeline for inference,
        # please fill in the input parameters.
        if isinstance(data["image"], list):
            inputs_shared.update({
                "input_image": data["image"],
                "height": data["image"][0].size[1],
                "width": data["image"][0].size[0],
            })
        else:
            inputs_shared.update({
                "input_image": data["image"],
                "height": data["image"].size[1],
                "width": data["image"].size[0],
            })
        inputs_shared = self.parse_extra_inputs(data, self.extra_inputs, inputs_shared)
        return inputs_shared, inputs_posi, inputs_nega

    def parse_extra_inputs(self, data, extra_inputs, inputs_shared):
        controlnet_keys_map = (
            ("blockwise_controlnet_", "blockwise_controlnet_inputs",),
            ("controlnet_", "controlnet_inputs"),
        )
        controlnet_inputs = {}
        for extra_input in extra_inputs:
            for prefix, name in controlnet_keys_map:
                if extra_input.startswith(prefix):
                    if name not in controlnet_inputs:
                        controlnet_inputs[name] = {}
                    controlnet_inputs[name][extra_input.replace(prefix, "")] = data[extra_input]
                    break
            else:
                inputs_shared[extra_input] = data[extra_input]
        for name, params in controlnet_inputs.items():
            inputs_shared[name] = [ControlNetInput(**params)]
        return inputs_shared

    def transfer_data_to_device(self, data, device, torch_float_dtype=None):
        if data is None:
            return data
        elif isinstance(data, torch.Tensor):
            data = data.to(device)
            if torch_float_dtype is not None and data.dtype in [torch.float, torch.float16, torch.bfloat16]:
                data = data.to(torch_float_dtype)
            return data
        elif isinstance(data, tuple):
            data = tuple(self.transfer_data_to_device(x, device, torch_float_dtype) for x in data)
            return data
        elif isinstance(data, list):
            data = list(self.transfer_data_to_device(x, device, torch_float_dtype) for x in data)
            return data
        elif isinstance(data, dict):
            data = {i: self.transfer_data_to_device(data[i], device, torch_float_dtype) for i in data}
            return data
        else:
            return data

    def export_trainable_state_dict(self, state_dict, remove_prefix=None):
        trainable_param_names = self.trainable_param_names()
        state_dict = {name: param for name, param in state_dict.items() if name in trainable_param_names}
        if remove_prefix is not None:
            state_dict_ = {}
            for name, param in state_dict.items():
                if name.startswith(remove_prefix):
                    name = name[len(remove_prefix):]
                state_dict_[name] = param
            state_dict = state_dict_
        return state_dict

    def forward(self,data):
        tar_lora = data["tar_lora"]
        tar_image = data["tar_image"]
        tar_prompt = data["tar_prompt"]
        tar_model_file = data["tar_model_file"]
        other_indices = data["other_indices"]
        model_files = data["model_files"]
        B = len(tar_lora)

        inputs = self.get_pipeline_inputs({"prompt": tar_prompt, "image": tar_image})
        inputs = self.transfer_data_to_device(inputs, self.pipe.device, self.pipe.torch_dtype)

        # load lora
        with torch.no_grad():
            clear_lora(self.pipe)
            print(f"load {len(model_files)} lora")
            for lora_path in model_files:
                load_lora(self.pipe, lora_path)

            # for name, module in self.pipe.dit.named_modules():
            #     if isinstance(module, LoRAMerger):
            #         print(name, len(module.lora_A_weights))
            #         break
        
        # loss
        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
        loss = self.task_to_loss['sft'](self.pipe, *inputs)
        clear_lora(self.pipe)
        return loss


class ModelLogger(object):
    def __init__(self, output_path, remove_prefix_in_ckpt=None,state_dict_converter=lambda x:x):
        self.output_path = output_path
        self.remove_prefix_in_ckpt = remove_prefix_in_ckpt

    def on_step_end(self, loss):
        pass

    def on_epoch_end(self, accelerator, model, epoch_id):
        accelerator.wait_for_everyone()
        state_dict = accelerator.get_state_dict(model)
        if accelerator.is_main_process:
            state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(state_dict, remove_prefix=self.remove_prefix_in_ckpt)
            # state_dict = self.state_dict_converter(state_dict)
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, f"epoch-{epoch_id}.safetensors")
            accelerator.save(state_dict, path, safe_serialization=True)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--metadata_path", type=str, required=False, default='/shark/zhiwen/LoRAHunter/DiffSynth-Studio/LoRA_Merger_train_data.csv')
    parser.add_argument("--output_dir", type=str, required=False, default="models/lora_merger")
    parser.add_argument("--info", type=str, required=False, default="")
    # lora encoder config

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)


    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--logging_steps", type=int, default=1)

    return parser.parse_args()

def main():
    args = parse_args()
    accelerator = Accelerator(kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])
    device = accelerator.device
    dataset = LoRAMergerDataset(load_lora_nums=4,metadata_path=args.metadata_path)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=args.batch_size, num_workers=args.num_workers,collate_fn=lambda x: x[0])

    model = LoRAMergerTrainingModel(device=device)
    # 打印可以训练的参数
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=args.lr, weight_decay=args.weight_decay,)

    train_set_name = args.metadata_path.split('/')[-1].split('.')[0]
    model_save_path = f"{args.output_dir}/{train_set_name}"
    os.makedirs(model_save_path, exist_ok=True)
    model_ids = len(os.listdir(model_save_path))
    if accelerator.is_main_process:
        p = f"{model_save_path}/{model_ids}"
        os.makedirs(p, exist_ok=True)
        with open(f'{p}/config.json', 'w') as f:
            json.dump(args.__dict__, f)
    model_logger = ModelLogger(f'{model_save_path}/{model_ids}', remove_prefix_in_ckpt="pipe.dit")
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
                optimizer.zero_grad()
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
            
            gc.collect()
            torch.cuda.empty_cache()
        if epoch % 1 == 0 or epoch==args.num_epochs-1:
            model_logger.on_epoch_end(accelerator, model, epoch) 
        # model_logger.on_epoch_end(accelerator, model, epoch)

if __name__ == "__main__":

    # dataset = LoRAMergerDataset()
    # dataloader = DataLoader(dataset, batch_size=1, shuffle=True,collate_fn=lambda x: x[0])

    # batch = next(iter(dataloader))
    # for k,v in batch.items():
    #     if 'model_files' in k:
    #         print(k,v)

    main()

# nohup accelerate launch --num_processes 4 --gpu_ids 0,2,3,4 --main_process_port=29502 train_lora_merger.py > zlog/train_merger_2.log 2>&1 & 


    
