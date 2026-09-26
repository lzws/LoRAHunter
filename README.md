# LORAHUNTER: DECODING SEMANTIC REPRESENTATIONS FROM LOW-RANK ADAPTER WEIGHTS


## Installation

```bash
pip install -r requirements.txt
```

Training and index construction are intended to run in a CUDA environment. `clipemb` requires a local Hugging Face CLIP model, for example a local copy of `openai/clip-vit-large-patch14`.

## LoRA Files

Each LoRA file can be a `.pt`, `.pth`, or `.safetensors` file.

The encoder expects canonical LoRA keys such as:

```text
unet.down_blocks.0.attentions.0.proj_in.lora.down.weight
unet.down_blocks.0.attentions.0.proj_in.lora.up.weight
```

Additional keys are allowed. The encoder only uses the `down.weight` and `up.weight` keys covered by the default Stable Diffusion 1.5 UNet LoRA patterns.

For original Diffusers LoRA safetensors, the loader attempts to use Diffusers' LoRA key conversion logic. To reduce data-loading overhead, converting the files to canonical `.pt` state dictionaries before training is recommended.

LoRA paths are resolved in the following order:

1. `lora_path` in the JSONL record
2. `model_path` in the JSONL record
3. `cache_dir/model_file`, with the extension replaced by `.pt` or `.pth`
4. `lora_root/model_file`
5. `model_file` itself

## Training Data Format

Training metadata uses JSONL, with one LoRA sample per line. Path fields can be absolute paths or paths relative to the corresponding command-line root.

### clipemb

`clipemb` uses a frozen CLIP text embedding as text supervision

```json
{
  "model_file": "civitai/12345.safetensors",
  "adapter_id": "12345",
  "llm_description": "a watercolor portrait of a woman",
  "diff_vec_path": "/data/diff_vectors/12345.pth",
  "lora_path": "/data/loras/civitai/12345.safetensors"
}
```

Required fields:

- `model_file` or `lora_path` / `model_path`
- `llm_description`, or alternatively `prompt` or `text`


### gclemb


```json
{
  "iid": "000183",
  "model_file": "civitai/12345.safetensors",
  "prompt": "a watercolor portrait of a woman",
  "weight": 1.0,
  "gcl_embedding_path": "/data/gcl_embeddings/00/000183.pth",
  "lora_path": "/data/loras/civitai/12345.safetensors"
}
```

Required fields:

- `model_file` or `lora_path` / `model_path`
- `gcl_embedding_path`, or `iid` together with `--gcl-emb-root`
- `weight` is optional and defaults to `1.0`
- `prompt`, or alternatively `llm_description` or `text`


## Training

### clipemb

```bash
accelerate launch --num_processes 1 train.py \
  --task clipemb \
  --metadata /data/train_clipemb.jsonl \
  --lora-root /data/loras \
  --cache-dir /data/lora_cache \
  --diff-emb-root /data/diff_vectors \
  --clip-model /data/models/clip-vit-large-patch14 \
  --output-dir /data/checkpoints/sd_clipemb \
  --embed-dim 768 \
  --head-mode dual \
  --batch-size 16 \
  --num-epochs 30 \
  --lr 1e-5
```

### gclemb

```bash
accelerate launch --num_processes 1 train.py \
  --task gclemb \
  --metadata /data/train_gclemb.jsonl \
  --lora-root /data/loras \
  --cache-dir /data/lora_cache \
  --gcl-emb-root /data/gcl_embeddings \
  --output-dir /data/checkpoints/sd_gclemb \
  --embed-dim 768 \
  --head-mode single \
  --loss-type weighted_contrastive_loss \
  --batch-size 16 \
  --num-epochs 30 \
  --lr 1e-5
```

The training output directory contains:

```text
config.json
retriever-epoch-0001.safetensors
retriever-epoch-0002.safetensors
retriever-final.safetensors
```


## Building a LoRA Embedding Index

The index contains the output of the trained LoRA encoder. All stored embeddings are L2-normalized.

```bash
python build_index.py \
  --metadata /data/retrieval_loras.jsonl \
  --checkpoint /data/checkpoints/sd_gclemb \
  --output /data/index/sd_gclemb.pt \
  --lora-root /data/loras \
  --cache-dir /data/lora_cache \
  --device cuda:0
```

`retrieval_loras.jsonl` only needs to provide `model_file` or a LoRA path field. The index stores `model_files`, normalized `embeddings`, and the corresponding metadata.

## Embedding-Only LoRA-Set Selection

The input JSONL contains one query per line and may include a prompt and a list of concepts:

```json
{
  "prompt": "a watercolor portrait of a woman in a garden",
  "concepts": [
    {"name": "watercolor", "description": "watercolor painting style"},
    {"name": "portrait", "description": "female portrait"}
  ]
}
```

### Retrieval

The query and concept texts are encoded by the same frozen CLIP encoder and compared with LoRA embeddings using cosine similarity.

```bash
python select_lora_set.py \
  --task clipemb \
  --index /data/index/sd_loraemb.pt \
  --input /data/test_prompts.jsonl \
  --output /data/results/sd_loraemb.jsonl \
  --clip-model /data/models/clip-vit-large-patch14 \
  --candidate-k-full 50 \
  --concept-top-k 10 \
  --max-unique-loras 4 \
  --num-combos 5
```



