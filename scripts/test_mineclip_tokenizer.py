#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试 MineCLIP 的 tokenizer
"""
import torch
from mineclip import MineCLIP

print("=" * 80)
print("检查 MineCLIP 的 tokenizer")
print("=" * 80)

# 创建模型
config = {
    "arch": "vit_base_p16_fz.v2.t2",
    "pool_type": "attn.d2.nh8.glusw",
    "resolution": (160, 256),
    "image_feature_dim": 512,
    "mlp_adapter_spec": "v0-2.t0",
    "hidden_dim": 512
}

model = MineCLIP(**config)
model.eval()

# 检查是否有 tokenizer
print("\n1. 检查 tokenizer 属性:")
if hasattr(model, 'tokenizer'):
    print(f"   ✓ model.tokenizer 存在: {type(model.tokenizer)}")
else:
    print(f"   ✗ model.tokenizer 不存在")

if hasattr(model, 'text_encoder'):
    print(f"   ✓ model.text_encoder 存在: {type(model.text_encoder)}")
else:
    print(f"   ✗ model.text_encoder 不存在")

if hasattr(model, 'clip_model'):
    print(f"   ✓ model.clip_model 存在: {type(model.clip_model)}")
    if hasattr(model.clip_model, 'tokenizer'):
        print(f"   ✓ model.clip_model.tokenizer 存在: {type(model.clip_model.tokenizer)}")
    
    if hasattr(model.clip_model, 'text_model'):
        print(f"   ✓ model.clip_model.text_model 存在")

# 尝试使用 transformers tokenizer
print("\n2. 尝试使用 transformers tokenizer:")
try:
    from transformers import CLIPTokenizer
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16")
    print(f"   ✓ tokenizer 加载成功")
    
    # 测试 tokenize
    text = "chop down a tree and collect one wood log"
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    print(f"   文本: {text}")
    print(f"   Token IDs 形状: {tokens['input_ids'].shape}")
    print(f"   Token IDs: {tokens['input_ids']}")
    
    # 测试 encode_text
    with torch.no_grad():
        text_features = model.encode_text(tokens['input_ids'])
        print(f"   ✓ encode_text 成功! 输出形状: {text_features.shape}")
        
except Exception as e:
    print(f"   ✗ 失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)

