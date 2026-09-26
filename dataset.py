import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
from safetensors.torch import load_file
from diffusers.loaders import StableDiffusionLoraLoaderMixin


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _extract_tensor(value):
    if torch.is_tensor(value):
        return value.float()
    if isinstance(value, dict):
        for key in ("embedding", "emb", "prompt_embedding", "text_embedding", "diff_vec", "vector"):
            if key in value and torch.is_tensor(value[key]):
                return value[key].float()
        tensors = [item for item in value.values() if torch.is_tensor(item)]
        if len(tensors) == 1:
            return tensors[0].float()
    if isinstance(value, (list, tuple)):
        return torch.tensor(value, dtype=torch.float32)
    raise TypeError(f"unsupported embedding object: {type(value)}")


def load_tensor(path):
    return _extract_tensor(_torch_load(path))


def _canonical_lora_keys(state_dict):
    return any(key.endswith(".down.weight") for key in state_dict)


def load_lora_state(path):
    path = str(path)
    suffix = Path(path).suffix.lower()
    if suffix in {".pt", ".pth", ".bin"}:
        state_dict = _torch_load(path)
    elif suffix == ".safetensors":
        state_dict = load_file(path, device="cpu")
        if not _canonical_lora_keys(state_dict):
            state_dict = StableDiffusionLoraLoaderMixin.lora_state_dict(path)[0]
    else:
        raise ValueError(f"unsupported LoRA file: {path}")
    if not isinstance(state_dict, dict):
        raise TypeError(f"LoRA file must contain a state dict: {path}")
    return {key: value.float() for key, value in state_dict.items() if torch.is_tensor(value)}


def resolve_lora_path(row, lora_root=None, cache_dir=None):
    candidates = []
    for key in ("lora_path", "model_path"):
        if row.get(key):
            candidates.append(Path(row[key]))
    model_file = row.get("model_file")
    if model_file:
        model_file = Path(model_file)
        if cache_dir is not None:
            candidates.append(Path(cache_dir) / model_file.with_suffix(".pt"))
            candidates.append(Path(cache_dir) / model_file.with_suffix(".pth"))
        if lora_root is not None:
            candidates.append(Path(lora_root) / model_file)
        candidates.append(model_file)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"cannot resolve LoRA path for row: {row}")


def _sharded_path(root, identifier, suffix=""):
    identifier = str(identifier)
    return Path(root) / identifier[:2] / identifier[2:4] / f"{identifier.replace('/', '__')}{suffix}.pth"


def _gcl_path(row, root):
    for key in ("gcl_embedding_path", "gcl_emb_path", "target_embedding_path"):
        if row.get(key):
            return Path(row[key])
    if row.get("iid") is None or root is None:
        return None
    iid = str(row["iid"])
    return Path(root) / iid[:2] / f"{iid}.pth"


def _diff_path(row, root):
    if row.get("diff_vec_path"):
        return Path(row["diff_vec_path"])
    if row.get("diff_embedding_path"):
        return Path(row["diff_embedding_path"])
    if row.get("adapter_id") is None or root is None:
        return None
    return _sharded_path(root, row["adapter_id"], "_diffvec")


class LoRADataset(Dataset):
    def __init__(
        self,
        metadata_path,
        task,
        lora_root=None,
        cache_dir=None,
        gcl_emb_root=None,
        diff_emb_root=None,
    ):
        if task not in {"clipemb", "gclemb"}:
            raise ValueError(f"unsupported task: {task}")
        self.rows = read_jsonl(metadata_path)
        self.task = task
        self.lora_root = lora_root
        self.cache_dir = cache_dir
        self.gcl_emb_root = gcl_emb_root
        self.diff_emb_root = diff_emb_root

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        lora_path = resolve_lora_path(row, self.lora_root, self.cache_dir)
        lora = load_lora_state(lora_path)
        text = row.get("llm_description", row.get("prompt", row.get("text", "")))
        sample = {
            "model_file": row.get("model_file", str(lora_path)),
            "lora": lora,
            "text": text,
            "row": row,
        }
        if self.task == "clipemb":
            diff_path = _diff_path(row, self.diff_emb_root)
            if diff_path is None or not diff_path.exists():
                raise FileNotFoundError(f"missing diff vector for row {index}: {row}")
            sample["target"] = load_tensor(diff_path).flatten()
            sample["weight"] = None
        else:
            if "gcl_embedding" in row:
                target = _extract_tensor(row["gcl_embedding"])
            else:
                gcl_path = _gcl_path(row, self.gcl_emb_root)
                if gcl_path is None or not gcl_path.exists():
                    raise FileNotFoundError(f"missing gcl embedding for row {index}: {row}")
                target = load_tensor(gcl_path)
            sample["target"] = target.flatten()
            sample["weight"] = torch.tensor(float(row.get("weight", 1.0)), dtype=torch.float32)
        return sample


def lora_collate_fn(batch):
    targets = torch.stack([item["target"] for item in batch], dim=0)
    weights = None
    if batch[0]["weight"] is not None:
        weights = torch.stack([item["weight"] for item in batch], dim=0)
    return {
        "model_file": [item["model_file"] for item in batch],
        "lora": [item["lora"] for item in batch],
        "text": [item["text"] for item in batch],
        "target": targets,
        "weight": weights,
        "row": [item["row"] for item in batch],
    }
