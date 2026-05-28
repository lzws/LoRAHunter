from diffsynth import FluxImagePipeline, ModelManager,load_state_dict
from diffsynth.models.lora import FluxLoRAConverter
from diffsynth.pipelines.flux_image import lets_dance_flux
from dataset import LoraDataset
from merger import LoraPatcher,LoraPatcher2
from utils import load_lora
import torch, os
from accelerate import Accelerator, DistributedDataParallelKwargs
from tqdm import tqdm
import random


class LoRAMergerTrainingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu", model_id_list=["FLUX.1-dev"])
        model_manager = ModelManager(torch_dtype=torch.bfloat16, device='cuda')
        model_manager.load_models([
            "/shark/zhiwen/LoRA_retrieve/lora/models/FLUX/FLUX.1-dev/text_encoder/model.safetensors",
            "/shark/zhiwen/LoRA_retrieve/lora/models/FLUX/FLUX.1-dev/text_encoder_2",
            "/shark/zhiwen/LoRA_retrieve/lora/models/FLUX/FLUX.1-dev/ae.safetensors",
            "/shark/zhiwen/LoRA_retrieve/lora/models/FLUX/FLUX.1-dev/flux1-dev.safetensors"
        ])
        self.pipe = FluxImagePipeline.from_model_manager(model_manager)
        self.lora_patcher = LoraPatcher().to(dtype=torch.bfloat16)
        self.pipe.enable_auto_lora()
        self.freeze_parameters()
        self.switch_to_training_mode()
        self.use_gradient_checkpointing = True
        self.state_dict_converter = FluxLoRAConverter.align_to_diffsynth_format
        self.device = "cuda"
        
        
    def to(self, *args, **kwargs):
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(*args, **kwargs)
        if device is not None:
            self.device = device
        if dtype is not None:
            self.torch_dtype = dtype
        super().to(*args, **kwargs)
        return self
        
        
    def switch_to_training_mode(self):
        self.pipe.scheduler.set_timesteps(1000, training=True)


    def freeze_parameters(self):
        self.pipe.requires_grad_(False)
        self.pipe.eval()
        self.pipe.denoising_model().train()
        self.lora_patcher.requires_grad_(True)


    def forward(self, batch):
        # Data
        text, image = batch[0]["text"], batch[0]["image"].unsqueeze(0)
        text_2, image_2 = batch[1]["text"], batch[1]["image"].unsqueeze(0)
        num_lora = torch.randint(1, len(batch)+1, (1,))[0]
        lora_state_dicts = [
            # self.state_dict_converter(load_lora(batch[i]["model_file"], device=self.device)) for i in range(num_lora)
            FluxLoRAConverter().align_to_all_format(load_state_dict(data[i]["model_file"], torch_dtype=torch.bfloat16, device=self.device)) for i in range(num_lora)
        ]
        if lora_state_dicts[0] == {}:
            print("empty lora +++++++++++++++++")
        lora_alphas = None

        # if random.random() < 0.25:
        #     text = ['']

        # Prepare input parameters
        self.pipe.device = self.device
        prompt_emb = self.pipe.encode_prompt(text, positive=True)
        prompt_emb_2 = self.pipe.encode_prompt(text_2, positive=True)
        
        latents = self.pipe.vae_encoder(image.to(dtype=self.pipe.torch_dtype, device=self.device))
        latents_2 = self.pipe.vae_encoder(image_2.to(dtype=self.pipe.torch_dtype, device=self.device))

        noise = torch.randn_like(latents)
        timestep_id = torch.randint(0, self.pipe.scheduler.num_train_timesteps, (1,))
        timestep = self.pipe.scheduler.timesteps[timestep_id].to(device=self.device)
        extra_input = self.pipe.prepare_extra_input(latents)
        extra_input_2 = self.pipe.prepare_extra_input(latents_2)

        noisy_latents = self.pipe.scheduler.add_noise(latents, noise, timestep)

        noisy_latents_2 = self.pipe.scheduler.add_noise(latents_2, noise, timestep)

        training_target = self.pipe.scheduler.training_target(latents, noise, timestep)


        # Compute loss
        # merged weight + lora1
        noise_pred_m1 = lets_dance_flux(
            self.pipe.dit,
            hidden_states=noisy_latents, timestep=timestep, **prompt_emb, **extra_input,
            lora_state_dicts=[lora_state_dicts[0]], lora_alphas=lora_alphas, lora_patcher=self.lora_patcher,
            use_gradient_checkpointing=self.use_gradient_checkpointing
        )

        # merged weight + lora2
        noise_pred_m2 = lets_dance_flux(
            self.pipe.dit,
            hidden_states=noisy_latents, timestep=timestep, **prompt_emb, **extra_input,
            lora_state_dicts=[lora_state_dicts[1]], lora_alphas=lora_alphas, lora_patcher=self.lora_patcher,
            use_gradient_checkpointing=self.use_gradient_checkpointing
        )

        # lora1 weight + lora1
        with torch.no_grad():
            noise_pred_l1 = lets_dance_flux(
                self.pipe.dit,
                hidden_states=noisy_latents, timestep=timestep, **prompt_emb, **extra_input,
                lora_state_dicts=[lora_state_dicts[0]], lora_alphas=lora_alphas, lora_patcher=None,
                use_gradient_checkpointing=self.use_gradient_checkpointing
            )
        
        # lora2 weight + lora2
        with torch.no_grad():
            noise_pred_l2 = lets_dance_flux(
                self.pipe.dit,
                hidden_states=noisy_latents, timestep=timestep, **prompt_emb, **extra_input,
                lora_state_dicts=[lora_state_dicts[1]], lora_alphas=lora_alphas, lora_patcher=None,
                use_gradient_checkpointing=self.use_gradient_checkpointing
            )
        

        print(f"loss: {loss}")
        return loss
    
    
    def trainable_modules(self):
        return self.lora_patcher.parameters()


class ModelLogger:
    def __init__(self, output_path, remove_prefix_in_ckpt=None):
        self.output_path = output_path
        self.remove_prefix_in_ckpt = remove_prefix_in_ckpt
        
    
    def on_step_end(self, loss):
        pass
    
    
    def on_epoch_end(self, accelerator, model, epoch_id):
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            state_dict = accelerator.unwrap_model(model).lora_patcher.state_dict()
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, f"epoch-{epoch_id}.safetensors")
            accelerator.save(state_dict, path, safe_serialization=True)


if __name__ == '__main__':
    model = LoRAMergerTrainingModel()
    # metadata_path = "prompts/train_prompts/good_loras_diff_length.csv"
    # metadata_path = 'prompts/train_prompts/loras_extend_caption_repeat_diff_length.csv'
    # metadata_path = '/shark/zhiwen/lora_diff_length/prompts/train_prompts/good_loras.csv'
    metadata_path = '/shark/zhiwen/lora_diff_length/prompts/train_prompts/fgao.csv'
    dataset = LoraDataset("data/lora/models/", metadata_path, steps_per_epoch=100, loras_per_item=1)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=1, num_workers=1, collate_fn=lambda x: x[0])
    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=5e-6)
    model_logger = ModelLogger("models/lora_merger/fgao_notext_5e5")
    accelerator = Accelerator(kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=False)])
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    
    for epoch_id in range(50):
        for data in tqdm(dataloader):
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                loss = model(data)
                accelerator.backward(loss)
                optimizer.step()
        model_logger.on_epoch_end(accelerator, model, epoch_id)

# nohup accelerate launch --num_processes 4 --gpu_ids 4,5,6,7 --main_process_port=29502 train_merger_one.py > zlog/train_merger_fgao_notext_5e6.log 2>&1 &
# 149729