import sys
import os,json
import pandas as pd
import pickle

import types
from dataclasses import dataclass
from typing import List, Optional

from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
from diffusers import LMSDiscreteScheduler
import torch
from PIL import Image
import argparse
from safetensors.torch import load_file
from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler, DPMSolverMultistepScheduler
import gc

@dataclass
class AdapterInfo:
    adapter_id: str
    alias: str
    title: str
    base_model: str
    description: str
    tags: List[str]
    trigger_words: List[str]
    stats: dict
    download_url: str
    llm_description: Optional[str] = None
    image_urls: Optional[List[str]] = None
    image_prompts: Optional[List[str]] = None
    image_negative_prompts: Optional[List[str]] = None
    weight: float = 0.8

    def __repr__(self):
        return (
            f"AdapterInfo(adapter_id={self.adapter_id!r}, alias={self.alias!r}, "
            f"title={self.title!r}, description={self.description!r}, "
            f"llm_description={self.llm_description!r})"
        )


if 'stylus' not in sys.modules:
    stylus_mod = types.ModuleType('stylus')
    sys.modules['stylus'] = stylus_mod

if 'stylus.refiner' not in sys.modules:
    refiner_mod = types.ModuleType('stylus.refiner')
    sys.modules['stylus.refiner'] = refiner_mod
    setattr(sys.modules['stylus'], 'refiner', refiner_mod)

mod_name = 'stylus.refiner.fetch_adapter_metadata'
if mod_name not in sys.modules:
    fam_mod = types.ModuleType(mod_name)
    sys.modules[mod_name] = fam_mod
    setattr(sys.modules['stylus.refiner'], 'fetch_adapter_metadata', fam_mod)

setattr(sys.modules[mod_name], 'AdapterInfo', AdapterInfo)

# 先把pkl里面的信息提取出来


adapter_metadata_path = "/mnt/nas2/zhiwen/LoRARetriver/stylusdata/cache/sd_adapters.pkl"


def load_adapter_metadata(path):
    with open(path, "rb") as f:
        adapters = pickle.load(f)
    return adapters

def check_file_size(file_path):
    # 获取文件大小（字节）
    size_in_bytes = os.path.getsize(file_path)
    # 1 MB = 1024 * 1024 字节
    is_less_than_1mb = size_in_bytes < (1024 * 1024) / 2
    return is_less_than_1mb


###### SD model utils ######
def extract_text_encoder_ckpt(ckpt_path):
    full_ckpt = torch.load(ckpt_path)
    new_ckpt = {}
    for key in full_ckpt.keys():
        if 'text_encoder.text_model' in key:
            new_ckpt[key.replace("text_encoder.", "")] = full_ckpt[key]
    return new_ckpt

def generate_images(model_name, prompts_path, save_path, device='cuda:0', guidance_scale = 7.5, image_size=512, ddim_steps=50, num_samples=10, from_case=0, folder_suffix='imagenette', origin_or_target='target'):
    '''
    Function to generate images from diffusers code
    
    The program requires the prompts to be in a csv format with headers 
        1. 'case_number' (used for file naming of image)
        2. 'prompt' (the prompt used to generate image)
        3. 'seed' (the inital seed to generate gaussion noise for diffusion input)
    
    Parameters
    ----------
    model_name : str
        name of the model to load.
    prompts_path : str
        path for the csv file with prompts and corresponding seeds.
    save_path : str
        save directory for images.
    device : str, optional
        device to be used to load the model. The default is 'cuda:0'.
    guidance_scale : float, optional
        guidance value for inference. The default is 7.5.
    image_size : int, optional
        image size. The default is 512.
    ddim_steps : int, optional
        number of denoising steps. The default is 100.
    num_samples : int, optional
        number of samples generated per prompt. The default is 10.
    from_case : int, optional
        The starting offset in csv to generate images. The default is 0.

    Returns
    -------
    None.

    '''
    if model_name == 'SD-v1-4':
        dir_ = "CompVis/stable-diffusion-v1-4"
    elif model_name == 'SD-V2':
        dir_ = "stabilityai/stable-diffusion-2-base"
    elif model_name == 'SD-V2-1':
        dir_ = "stabilityai/stable-diffusion-2-1-base"
    else:
        dir_ = "CompVis/stable-diffusion-v1-4" # all the erasure models built on SDv1-4
    
    dir_ = "/shark/zhiwen/LoRAHunter/DiffSynth-Studio/models/AI-ModelScope/stable-diffusion-v1-5"

    target_ckpt = None
    save_path = f'{save_path}/original_SD/visualizations_{folder_suffix}'
        
    # 1. Load the autoencoder model which will be used to decode the latents into image space.
    vae = AutoencoderKL.from_pretrained(dir_, subfolder="vae")
    # 2. Load the tokenizer and text encoder to tokenize and encode the text.
    tokenizer = CLIPTokenizer.from_pretrained(dir_, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(dir_, subfolder="text_encoder")
    # 3. The UNet model for generating the latents.
    unet = UNet2DConditionModel.from_pretrained(dir_, subfolder="unet")

    scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)

    vae.to(device)
    text_encoder.to(device)
    unet.to(device)
    torch_device = device
    df = pd.read_csv(prompts_path)

    folder_path = f'{save_path}/{model_name}'
    os.makedirs(folder_path, exist_ok=True)

    for _, row in df.iterrows():
        prompt = [str(row.prompt)]*num_samples
        seed = row.evaluation_seed
        case_number = row.case_number
        if case_number<from_case:
            continue

        height = image_size                        # default height of Stable Diffusion
        width = image_size                         # default width of Stable Diffusion

        num_inference_steps = ddim_steps           # Number of denoising steps

        guidance_scale = guidance_scale            # Scale for classifier-free guidance

        generator = torch.manual_seed(seed)        # Seed generator to create the inital latent noise

        batch_size = len(prompt)

        text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")

        text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]

        max_length = text_input.input_ids.shape[-1]
        uncond_input = tokenizer(
            [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]

        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        latents = torch.randn(
            (batch_size, unet.in_channels, height // 8, width // 8),
            generator=generator,
        )
        latents = latents.to(torch_device)

        scheduler.set_timesteps(num_inference_steps)

        latents = latents * scheduler.init_noise_sigma

        from tqdm.auto import tqdm

        scheduler.set_timesteps(num_inference_steps)

        for t in tqdm(scheduler.timesteps):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)

            latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)

            # predict the noise residual
            with torch.no_grad():
                noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = scheduler.step(noise_pred, t, latents).prev_sample

        # scale and decode the image latents with vae
        latents = 1 / 0.18215 * latents
        with torch.no_grad():
            image = vae.decode(latents).sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (image * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]
        for num, im in enumerate(pil_images):
            im.save(f"{folder_path}/{case_number}_{num}.png")



def get_sd_model(dir_,device="cuda"):
    # 1. Load the autoencoder model which will be used to decode the latents into image space.
    vae = AutoencoderKL.from_pretrained(dir_, subfolder="vae")
    # 2. Load the tokenizer and text encoder to tokenize and encode the text.
    tokenizer = CLIPTokenizer.from_pretrained(dir_, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(dir_, subfolder="text_encoder")
    # 3. The UNet model for generating the latents.
    unet = UNet2DConditionModel.from_pretrained(dir_, subfolder="unet")

    # unet_path = '/shark/zhiwen/LoRAHunter/DiffSynth-Studio/models/SDv1-5-Unet/counterfeitV30_v30.safetensors'
    # state_dict = load_file(unet_path)

    # unet.load_state_dict(state_dict)

    scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)

    vae.to(device)
    text_encoder.to(device)
    unet.to(device)
    torch_device = device

    return vae, tokenizer, text_encoder, unet, scheduler


def generate_img(device='cuda'):
    # load sd model
    model_path = "/shark/zhiwen/LoRAHunter/DiffSynth-Studio/models/AI-ModelScope/stable-diffusion-v1-5"
    vae, tokenizer, text_encoder, unet, scheduler = get_sd_model(model_path,device)

    torch_device = device

    height = 512                        # default height of Stable Diffusion
    width = 512                         # default width of Stable Diffusion

    num_inference_steps = 50           # Number of denoising steps

    guidance_scale = 7.5            # Scale for classifier-free guidance
    
    seed = 42
    generator = torch.manual_seed(seed)        # Seed generator to create the inital latent noise

    prompt = ["a cat and a girl is playing ball"]
    batch_size = len(prompt)

    text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")

    text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]

    max_length = text_input.input_ids.shape[-1]
    uncond_input = tokenizer(
        [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]

    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

    latents = torch.randn(
        (batch_size, unet.in_channels, height // 8, width // 8),
        generator=generator,
    )
    latents = latents.to(torch_device)

    scheduler.set_timesteps(num_inference_steps)

    latents = latents * scheduler.init_noise_sigma

    from tqdm.auto import tqdm

    scheduler.set_timesteps(num_inference_steps)

    for t in tqdm(scheduler.timesteps):
        # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
        latent_model_input = torch.cat([latents] * 2)

        latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)

        # predict the noise residual
        with torch.no_grad():
            noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

        # perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # compute the previous noisy sample x_t -> x_t-1
        latents = scheduler.step(noise_pred, t, latents).prev_sample

    # scale and decode the image latents with vae
    latents = 1 / 0.18215 * latents
    with torch.no_grad():
        image = vae.decode(latents).sample

    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    im = pil_images[0]
    im.save("sd_v1-5_rea.jpg")

def load_sd_from_single_file(model_path,device="cuda"):
    pipe = StableDiffusionPipeline.from_single_file(model_path, torch_dtype=torch.float16, local_files_only=True)
    # pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    return pipe

def dynamic_load_lora(model_path,device="cuda"):

    # 1. 初始化基础 Pipeline (只加载一次)
    base_model = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(
        base_model, 
        torch_dtype=torch.float16
    ).to("cuda")

    # 假设你有一堆 LoRA 文件路径
    lora_list = [
        "./loras/style_anime.safetensors",
        "./loras/character_ganyu.safetensors",
        "./loras/pose_action.safetensors",
        # ... 还有 100 个 ...
    ]

    prompt = "masterpiece, best quality, 1girl"

    # 2. 循环处理
    for lora_path in lora_list:
        print(f"--- 正在处理: {lora_path} ---")
        
        try:
            # A. 动态加载当前 LoRA
            # 注意：这里不需要 adapter_name，除非你要混合
            pipe.load_lora_weights(lora_path)
            
            # B. 生成图像
            image = pipe(
                prompt, 
                num_inference_steps=20,
                cross_attention_kwargs={"scale": 0.8} # 动态权重
            ).images[0]
            
            # 保存
            image.save(f"output_{os.path.basename(lora_path)}.png")
            
        finally:
            # C. 【关键步骤】用完立刻卸载！
            # 这行代码会把 LoRA 权重从显存中抹去
            pipe.unload_lora_weights()
            
            # D. (可选但推荐) 强制清理 Python 垃圾回收和 CUDA 缓存
            # 确保显存彻底释放，防止碎片化
            gc.collect()
            torch.cuda.empty_cache()

    print("所有 LoRA 处理完毕，显存已释放。")

if __name__ == "__main__":
    unet_path = '/shark/zhiwen/LoRAHunter/DiffSynth-Studio/models/SDv1-5-Unet/counterfeitV30_v30.safetensors'
    lora_files = ["/shark/zhiwen/LoRAHunter/sd_lora/lzwecnu/civitai-lora-66k-70k/28/280276.safetensors",
        "/shark/zhiwen/LoRAHunter/sd_lora/lzwecnu/civitai-lora-66k-70k/28/289562.safetensors",
        ]
    base_model = "/shark/zhiwen/LoRAHunter/DiffSynth-Studio/models/AI-ModelScope/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(
        base_model, 
        torch_dtype=torch.float16
    ).to("cuda")
    for lora_file in lora_files:
        print(f"--- 正在处理: {lora_file} ---")
        pipe.load_lora_weights(lora_file)
    image = pipe(
        "a cat", 
        num_inference_steps=20,
    ).images[0]


    # generate_img()