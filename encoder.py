import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from Qwen3_VL_Embedding.src.models.qwen3_vl_embedding import Qwen3VLEmbedder

# "openai/clip-vit-large-patch14"
class TextImageEncoder(torch.nn.Module):
    def __init__(self, model_name="/shark/zhiwen/LoRAHunter/DiffSynth-Studio/models/AI-ModelScope/clip-vit-large-patch14", dtype=torch.float):
        super().__init__()
        # self.device = device
        self.dtype = dtype
        # self.model, self.preprocess = clip.load('ViT-L/14@336px', download_root=model_name, device=device)
        print(f"load clip model from: {model_name}")
        self.model = CLIPModel.from_pretrained(model_name).to(dtype=torch.float)
        self.processor = CLIPProcessor.from_pretrained(model_name)



    @torch.no_grad()
    def encoding_image(self,image_path=''):
        image = Image.open(image_path)
        image_input = self.processor(images=image, return_tensors="pt").to(self.model.device)
        image_features = self.model.get_image_features(**image_input)
        return image_features.pooler_output #[1,768]
        

    @torch.no_grad()
    def encoding_text(self,text=''):
        text_input = self.processor(text=text, return_tensors="pt", max_length=77, truncation=True, padding=True).to(self.model.device)
        text_features = self.model.get_text_features(**text_input)
        return text_features.pooler_output #[1, 768]


class LoRAEncoder(torch.nn.Module):
    def __init__(self, model_name="", device='cuda:7'):
        super().__init__()
        self.device = device


class QwenVLEncoder(torch.nn.Module):
    def __init__(self, model_name="/shark/zhiwen/LoRAHunter/DiffSynth-Studio/models/Qwen/Qwen3-VL-Embedding-2B", device='cuda'):
        super().__init__()
        self.device = device
        self.model = Qwen3VLEmbedder(model_name_or_path=model_name,device = torch.device(device))
        print(f"load model from: {model_name}")

    @torch.no_grad()
    def encoding_text(self, text):
        if isinstance(text, str):
            text = [text]

        queries = [{"text": t} for t in text]
        text_features = self.model.process(queries) # [N, 2048]
        return text_features
    
    @torch.no_grad()
    def encoding_image(self, image_path):
        if isinstance(image_path, str):
            image_path = [image_path]

        queries = [{"image": i} for i in image_path]
        image_features = self.model.process(queries) # [N, 2048]
        return image_features

    @torch.no_grad()
    def encoding_images(self, inputs):
        image_features = self.model.get_embeddings(inputs) # [N, 2048]
        return image_features

            
if __name__ == '__main__':

    clip_encoder = QwenVLEncoder(device="cuda:4")
    text = 'This LoRA adapter focuses on generating images of male figures wearing school uniforms, particularly athletic-style tracksuits commonly associated with Chinese high schools. The clothing is consistently depicted with a structured design: short-sleeved zip-up jackets featuring color-blocked panels in blue, white, and navy, often with vertical stripes along the sleeves and side seams. Matching shorts and striped socks complete the ensemble. The garments appear to be made of lightweight, slightly shiny synthetic fabric, suggesting a sporty, functional material. The subject is typically positioned outdoors in natural settings such as school sports fields or campuses, with blurred greenery and distant buildings creating a soft, real-world backdrop. Lighting is bright and even, simulating daylight with natural shadows, contributing to a clean, realistic aesthetic. The adapter emphasizes the upper body and facial features, maintaining clear detail in the face and clothing while keeping the background softly out of focus. It appears to strengthen the visual presence of the subject, enhancing contrast and clarity in the figure relative to the surroundings. The overall mood is calm and youthful, evoking a sense of everyday student life. While the exact pose may vary, the composition tends to center on full-body or three-quarter views that highlight the uniform’s design and fit'
    text = 'This LoRA adapter focuses on generating images of male figures wearing school uniforms, particularly athletic-style tracksuits commonly associated with Chinese high schools. The clothing is consistently depicted with a structured design: short-sleeved zip-up jackets featuring color-blocked panels in blue, white, and navy, often with vertical stripes along the sleeves and side seams. Matching shorts and striped socks complete the ensemble. The garments appear to be made of lightweight, slightly shiny synthetic fabric, suggesting a sporty, functional material. The overall mood is calm and youthful, evoking a sense of everyday student life. While the exact pose may vary, the composition tends to center on full-body or three-quarter views that highlight the uniform’s design and fit'
    text = 'This LoRA induces a flat, abstract style with simplified geometric forms and uniform color fields. It transforms subjects into stylized 2D representations featuring clean edges and solid blocks, removing texture, shading, and realistic depth. Outputs resemble digital illustrations or graphic designs, prioritizing clarity and form over naturalism. Figures are reduced to essential shapes against plain backgrounds, creating a cohesive, minimalist aesthetic akin to modern interface graphics. The adapter focuses on stylistic abstraction and visual simplicity rather than specific subjects or environments.'
    
    image_path = '/shark/zhiwen/LoRAHunter/DiffSynth-Studio/DiffImage/Qwen/Animals/Hybrid Creatures/0/42.jpg'
    image = Image.open(image_path).convert('RGB')
    text_features = clip_encoder.encoding_image([image,image])

    print(text_features.shape)

    # print(text_features.last_hidden_state.shape)
    # print(text_features.pooler_output.shape)

    # t_emb = text_features.pooler_output
    # image = '/shark/zhiwen/LoRAHunter/DiffSynth-Studio/DiffImage/Qwen/Animals/Hybrid Creatures/0/42.jpg'
    # image_features = clip_encoder.encoding_image(image)
    # print(image_features.last_hidden_state.shape)
    # print(image_features.pooler_output.shape)
    # i_emb = image_features.pooler_output

    # t_emb = t_emb / t_emb.norm(dim=-1, keepdim=True)
    # i_emb = i_emb / i_emb.norm(dim=-1, keepdim=True)
    # similarity = (t_emb @ i_emb.T)
    # print(similarity)



