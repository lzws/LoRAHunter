import argparse
from pathlib import Path

import torch

from checkpoint import load_checkpoint
from dataset import load_lora_state, read_jsonl, resolve_lora_path
from lora_encoder import LoRAEncoder


def make_encoder(config):
    task = config.get("task", "gclemb")
    head_mode = config.get("head_mode", "auto")
    if head_mode == "auto":
        head_mode = "dual" if task == "clipemb" else "single"
    return LoRAEncoder(
        embed_dim=config.get("embed_dim", 768),
        max_position_embeddings=config.get("max_position_embeddings", 150),
        num_encoder_layers=config.get("num_encoder_layers", 4),
        encoder_intermediate_size=config.get("encoder_intermediate_size", 2560),
        L=config.get("L", 1),
        num_probes=config.get("num_probes", 4),
        block_type=config.get("block_type", "block"),
        head_mode=head_mode,
        pooling=config.get("pooling", "cls"),
        out_dim=config.get("embed_dim", 768),
        text_out_dim=config.get("text_out_dim"),
        img_out_dim=config.get("img_out_dim"),
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--lora-root", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    config, lora_state, _, _ = load_checkpoint(args.checkpoint)
    model = make_encoder(config).to(args.device, dtype=torch.float32)
    model.load_state_dict(lora_state, strict=True)
    model.eval()
    rows = read_jsonl(args.metadata)
    model_files = []
    embeddings = []
    with torch.no_grad():
        for start in range(0, len(rows), args.batch_size):
            batch_rows = rows[start:start + args.batch_size]
            for row in batch_rows:
                path = resolve_lora_path(row, args.lora_root, args.cache_dir)
                lora = load_lora_state(path)
                lora = {key: value.to(args.device, dtype=torch.float32) for key, value in lora.items()}
                output = model(lora)
                if isinstance(output, tuple):
                    output = output[0]
                embeddings.append(torch.nn.functional.normalize(output[0].float().cpu(), dim=-1))
                model_files.append(row.get("model_file", str(path)))
            print(f"indexed {min(start + args.batch_size, len(rows))}/{len(rows)}")
    index = {
        "model_files": model_files,
        "embeddings": torch.stack(embeddings, dim=0),
        "metadata": rows,
        "task": config.get("task"),
        "dim": int(embeddings[0].numel()) if embeddings else 0,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(index, output)
    print(f"saved {len(model_files)} embeddings to {output}")


if __name__ == "__main__":
    main()
