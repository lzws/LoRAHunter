import json
from pathlib import Path

from safetensors.torch import load_file, save_file


def checkpoint_paths(path):
    path = Path(path)
    if path.is_dir():
        candidates = sorted(
            list(path.glob("retriever-*.safetensors"))
            + list(path.glob("lora_encoder-*.safetensors"))
        )
        if not candidates:
            raise FileNotFoundError(f"no checkpoint found in {path}")
        checkpoint = candidates[-1]
        config = path / "config.json"
    else:
        checkpoint = path
        config = path.parent / "config.json"
    if not config.exists():
        raise FileNotFoundError(f"missing config.json next to {checkpoint}")
    return checkpoint, config


def load_checkpoint(path):
    checkpoint, config_path = checkpoint_paths(path)
    with open(config_path, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    state_dict = load_file(str(checkpoint), device="cpu")
    has_prefix = any(key.startswith("lora_encoder.") for key in state_dict)
    if has_prefix:
        lora_state = {
            key[len("lora_encoder."):]: value
            for key, value in state_dict.items()
            if key.startswith("lora_encoder.")
        }
    else:
        lora_state = {key: value for key, value in state_dict.items() if key != "logit_scale"}
    return config, lora_state, state_dict.get("logit_scale"), checkpoint


def save_checkpoint(path, lora_encoder, logit_scale=None):
    state_dict = {
        f"lora_encoder.{key}": value.detach().float().cpu().contiguous()
        for key, value in lora_encoder.state_dict().items()
    }
    if logit_scale is not None:
        state_dict["logit_scale"] = logit_scale.detach().float().cpu().contiguous()
    save_file(state_dict, str(path))
