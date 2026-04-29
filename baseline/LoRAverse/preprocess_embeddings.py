#!/usr/bin/env python3
"""
使用CLIP预计算LoRA embedding的脚本

用法:
    python preprocess_clip.py -i lora_data.json -o embeddings.npz --clip-model openai/clip-vit-large-patch14
    
    # 使用较小的模型测试
    python preprocess_clip.py -i data.json -o embeddings.npz --clip-model openai/clip-vit-base-patch32 --batch-size 64
"""

import json
import argparse
import sys
from pathlib import Path

try:
    from adapter_database_clip import AdapterDatabase
except ImportError:
    print("Error: adapter_database_clip.py not found")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description='Precompute LoRA embeddings with CLIP')
    parser.add_argument('-i', '--input', required=True, help='Input JSON file')
    parser.add_argument('-o', '--output', required=True, help='Output .npz file')
    parser.add_argument('--clip-model', default='openai/clip-vit-large-patch14',
                       help='CLIP model name (paper uses ViT-L/14)')
    parser.add_argument('--batch-size', '-b', type=int, default=32)
    parser.add_argument('--save-metadata', action='store_true', default=True,
                       help='Save descriptions and metadata')
    
    args = parser.parse_args()
    
    print("="*70)
    print("LoRAverse CLIP Embedding Preprocessing")
    print("="*70)
    
    # 加载数据
    print(f"\nLoading data from {args.input}")
    with open(args.input, 'r') as f:
        data = json.load(f)
    
    print(f"Found {len(data)} adapters")
    
    # 验证ID唯一性
    ids = [item.get('id') for item in data]
    if len(set(ids)) != len(ids):
        print("ERROR: Duplicate IDs found!")
        sys.exit(1)
    
    print(f"ID uniqueness: verified ({len(set(ids))} unique)")
    
    # 计算embedding
    print(f"\nInitializing CLIP: {args.clip_model}")
    db = AdapterDatabase(use_clip=True, clip_model=args.clip_model)
    
    print(f"\nComputing embeddings (batch_size={args.batch_size})...")
    db.add_adapters(data, batch_size=args.batch_size)
    
    # 验证映射
    print("\nVerifying ID <-> embedding mapping...")
    if db.verify_id_embedding_mapping():
        print("✓ Verification passed")
    else:
        print("✗ Verification failed!")
        sys.exit(1)
    
    # 保存
    print(f"\nSaving to {args.output}")
    db.save_embeddings(args.output, save_metadata=args.save_metadata)
    
    # 输出统计
    output_path = Path(args.output)
    size_mb = output_path.stat().st_size / 1024 / 1024
    
    print("\n" + "="*70)
    print("Preprocessing complete!")
    print("="*70)
    print(f"Adapters processed: {len(db.adapters)}")
    print(f"Embedding model: {args.clip_model}")
    print(f"Embedding shape: {db.embeddings_matrix.shape}")
    print(f"Output file: {args.output} ({size_mb:.2f} MB)")
    print(f"\nUsage:")
    print(f"  from adapter_database_clip import AdapterDatabase")
    print(f"  db = AdapterDatabase.load_embeddings('{args.output}')")


if __name__ == "__main__":
    main()


# python preprocess_clip.py \
#     --input lora_metadata.json \
#     --output embeddings_clip.npz \
#     --clip-model openai/clip-vit-large-patch14 \
#     --batch-size 64
