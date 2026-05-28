import torch
from models import TextImageEncoder,QwenVLEncoder
import torch.nn.functional as F
import json

class LoRARetriever:
    def __init__(self, index_path, device="cuda:7"):
        self.device = device
        # self.clip_encoder = TextImageEncoder().to(device)
        self.clip_encoder = QwenVLEncoder(device=device).to(device)
        self.clip_encoder.eval()

        index_data = torch.load(index_path, map_location="cpu")
        print(f"keys: {index_data.keys()}")
        self.model_files = index_data["model_files"]
        self.lora_embs = index_data["embeddings"]   # [N, D]
        print(f"[Index] loaded index: {self.lora_embs.shape}")

        # normalize once
        self.lora_embs = F.normalize(self.lora_embs, dim=-1).to(device)

    @torch.no_grad()
    def retrieve(self, query_text, top_k=5):
        text_emb = self.clip_encoder.encoding_text([query_text])   # [1, D]
        text_emb = F.normalize(text_emb, dim=-1)

        scores = text_emb @ self.lora_embs.t()   # [1, N]
        top_scores, top_indices = torch.topk(scores, k=top_k, dim=-1)

        results = []
        for score, idx in zip(top_scores[0].tolist(), top_indices[0].tolist()):
            results.append({
                "model_file": self.model_files[idx],
                "score": score,
            })
        return results





def call_lora():
    index_path = "rank_dataset/qwenlora_filtered_query_index_qwen.pt"
    retriever = LoRARetriever(index_path=index_path, device="cuda:4")

    test_data_path = 'test_data/diffusiondb_test_200.jsonl'
    with open(test_data_path, 'r') as f:
        test_datas = [json.loads(line) for line in f.readlines()]

    for i,data in enumerate(test_datas):
        print(i)
        # extract_concept = data['extract_concept']
        data['retrieval_results'] = {}
        for ec in [1]:
            keyword = data['prompt']
            print(keyword)
            # retrieval_description = ec['retrieval_description']

            
            top5 = retriever.retrieve(query_text=keyword,top_k=150)
            data['retrieval_results'][keyword] = top5
            # res_datas.append(data)
        with open("test_data/diffusiondb_test_200_calllora_totalpool_qwen.jsonl", 'a') as f:
            f.write(json.dumps(data)+'\n')

if __name__ == "__main__":
    call_lora()