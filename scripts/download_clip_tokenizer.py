#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
下载 CLIP tokenizer 到本地
只需要运行一次，之后就可以离线使用
"""
import os
from transformers import CLIPTokenizer

# 下载目录
tokenizer_dir = "data/clip_tokenizer"
os.makedirs(tokenizer_dir, exist_ok=True)

print("=" * 80)
print("下载 CLIP Tokenizer 到本地")
print("=" * 80)
print(f"目标目录: {tokenizer_dir}")
print()

try:
    print("1. 从 HuggingFace 下载 tokenizer...")
    tokenizer = CLIPTokenizer.from_pretrained(
        "openai/clip-vit-base-patch16",
        cache_dir=None  # 使用默认缓存
    )
    
    print("2. 保存到本地目录...")
    tokenizer.save_pretrained(tokenizer_dir)
    
    print()
    print("=" * 80)
    print("✓ 下载成功!")
    print("=" * 80)
    print(f"Tokenizer 已保存到: {tokenizer_dir}")
    print()
    print("文件列表:")
    for file in os.listdir(tokenizer_dir):
        file_path = os.path.join(tokenizer_dir, file)
        size = os.path.getsize(file_path) / 1024  # KB
        print(f"  - {file:30s} ({size:.1f} KB)")
    print()
    print("现在可以离线使用了！")
    print("=" * 80)
    
    # 测试加载
    print()
    print("3. 测试离线加载...")
    tokenizer_local = CLIPTokenizer.from_pretrained(tokenizer_dir)
    print("   ✓ 离线加载成功!")
    
    # 测试 tokenize
    text = "chop down a tree and collect one wood log"
    tokens = tokenizer_local(
        text,
        return_tensors="pt",
        padding="max_length",
        max_length=77,
        truncation=True
    )
    print(f"   ✓ Tokenize 测试成功! 输出形状: {tokens['input_ids'].shape}")
    print()
    print("=" * 80)
    
except Exception as e:
    print(f"✗ 下载失败: {e}")
    import traceback
    traceback.print_exc()

