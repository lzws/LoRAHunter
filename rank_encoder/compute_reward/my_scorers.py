import torch
import torch.nn.functional as F

from PIL import Image
from typing import List
from transformers import AutoModel, CLIPModel, CLIPProcessor


class BaseScorer:
    def score(self, prompt: str, images: List[Image.Image]) -> List[float]:
        raise NotImplementedError


class CLIPScoreScorer(BaseScorer):
    def __init__(
        self,
        model_name: str = "openai/clip-vit-large-patch14",
        device: str = "cuda:0",
    ):
        self.device = device
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(device)
        self.model.eval()

    @torch.no_grad()
    def score(self, prompt: str, images: List[Image.Image]) -> List[float]:
        image_inputs = self.processor(
            images=images,
            return_tensors="pt",
            padding=True,
        )
        text_inputs = self.processor(
            text=[prompt] * len(images),
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        image_inputs = {k: v.to(self.device) for k, v in image_inputs.items()}
        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}

        image_embs = self.model.get_image_features(**image_inputs).pooler_output
        
        text_embs = self.model.get_text_features(**text_inputs).pooler_output

        image_embs = F.normalize(image_embs, dim=-1)
        text_embs = F.normalize(text_embs, dim=-1)

        scores = (image_embs * text_embs).sum(dim=-1)
        return scores.float().cpu().tolist()


class PickScoreScorer(BaseScorer):
    def __init__(
        self,
        model_name: str = "yuvalkirstain/PickScore_v1",
        processor_name: str = "/shark/zhiwen/LoRAHunter/rank_encoder/models/AI-ModelScope/PickScore_v1",
        device: str = "cuda:0",
    ):
        self.device = device

        # PickScore 常用搭配是 CLIPProcessor
        self.processor = CLIPProcessor.from_pretrained(processor_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()

    @torch.no_grad()
    def score(self, prompt: str, images: List[Image.Image]) -> List[float]:
        image_inputs = self.processor(
            images=images,
            return_tensors="pt",
            padding=True,
        )
        text_inputs = self.processor(
            text=[prompt] * len(images),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        )

        image_inputs = {k: v.to(self.device) for k, v in image_inputs.items()}
        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}

        image_embs = self.model.get_image_features(**image_inputs).pooler_output
        text_embs = self.model.get_text_features(**text_inputs).pooler_output

        image_embs = F.normalize(image_embs, dim=-1)
        text_embs = F.normalize(text_embs, dim=-1)

        scores = (image_embs * text_embs).sum(dim=-1)
        return scores.float().cpu().tolist()





def build_scorers(device: str):
    scorers = {
        "clip": CLIPScoreScorer(
            model_name="/shark/zhiwen/LoRAHunter/rank_encoder/models/muse/openai-clip-vit-large-patch14",
            device=device,
        ),
        "pick": PickScoreScorer(
            model_name="/shark/zhiwen/LoRAHunter/rank_encoder/models/AI-ModelScope/PickScore_v1",
            device=device,
        ),
    }
    return scorers
