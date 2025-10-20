#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试 MineCLIP 的正确使用方法
"""
import torch
from mineclip import MineCLIP
from transformers import CLIPTokenizer

print("=" * 80)
print("测试 MineCLIP 的正确使用方法")
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

# 加载 tokenizer
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16")

# 测试 1: 固定长度 77
print("\n1. 测试固定长度 77 (max_length=77):")
text = "chop down a tree and collect one wood log"
tokens = tokenizer(
    text, 
    return_tensors="pt", 
    padding="max_length",  # padding 到 max_length
    max_length=77,         # 固定长度 77
    truncation=True
)
print(f"   文本: {text}")
print(f"   Token IDs 形状: {tokens['input_ids'].shape}")

try:
    with torch.no_grad():
        text_features = model.encode_text(tokens['input_ids'])
        print(f"   ✓ encode_text 成功! 输出形状: {text_features.shape}")
        print(f"   特征范数: {text_features.norm().item():.4f}")
except Exception as e:
    print(f"   ✗ 失败: {e}")

# 测试 2: 图像编码
print("\n2. 测试图像编码 (forward_image_features):")
fake_image = torch.randn(1, 3, 160, 256)
print(f"   图像形状: {fake_image.shape}")

try:
    with torch.no_grad():
        image_features = model.forward_image_features(fake_image)
        print(f"   ✓ forward_image_features 成功! 输出形状: {image_features.shape}")
        print(f"   特征范数: {image_features.norm().item():.4f}")
except Exception as e:
    print(f"   ✗ 失败: {e}")

# 测试 3: 计算相似度
print("\n3. 测试图像-文本相似度:")
try:
    with torch.no_grad():
        # 归一化特征
        image_features_norm = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # 计算余弦相似度
        similarity = (image_features_norm @ text_features_norm.T).squeeze()
        print(f"   ✓ 相似度计算成功: {similarity.item():.4f}")
        print(f"   相似度范围: [-1, 1]，值越大越相似")
        
        # 归一化到 [0, 1]
        similarity_normalized = (similarity + 1) / 2
        print(f"   归一化相似度: {similarity_normalized.item():.4f}")
        
except Exception as e:
    print(f"   ✗ 失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("总结:")
print("  1. tokenizer 需要设置 max_length=77, padding='max_length'")
print("  2. 图像编码使用 forward_image_features(image)")
print("  3. 文本编码使用 encode_text(token_ids)")
print("  4. 相似度 = image_features @ text_features.T")
print("=" * 80)

