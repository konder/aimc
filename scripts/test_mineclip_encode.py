#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试 MineCLIP 的编码方法
"""
import torch
import numpy as np
from mineclip import MineCLIP

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

print("=" * 80)
print("测试 MineCLIP 编码方法")
print("=" * 80)

# 1. 测试 forward_image_features
print("\n1. 测试 forward_image_features:")
import inspect
sig = inspect.signature(model.forward_image_features)
print(f"   签名: {sig}")

try:
    # 创建一个假的图像 (1, 3, 160, 256)
    fake_image = torch.randn(1, 3, 160, 256)
    print(f"   输入形状: {fake_image.shape}")
    
    result = model.forward_image_features(fake_image)
    print(f"   ✓ 成功! 输出形状: {result.shape}")
except Exception as e:
    print(f"   ✗ 失败: {e}")

# 2. 测试 encode_video
print("\n2. 测试 encode_video:")
sig = inspect.signature(model.encode_video)
print(f"   签名: {sig}")

try:
    # 创建一个假的视频 (batch, frames, 3, 160, 256)
    fake_video = torch.randn(1, 16, 3, 160, 256)  # 16帧
    print(f"   输入形状: {fake_video.shape}")
    
    result = model.encode_video(fake_video)
    print(f"   ✓ 成功! 输出形状: {result.shape}")
except Exception as e:
    print(f"   ✗ 失败: {e}")

# 3. 测试 encode_text
print("\n3. 测试 encode_text:")
sig = inspect.signature(model.encode_text)
print(f"   签名: {sig}")

try:
    # 创建假的 token (batch, seq_len)
    fake_tokens = torch.randint(0, 1000, (1, 77))
    print(f"   输入形状: {fake_tokens.shape}")
    
    result = model.encode_text(fake_tokens)
    print(f"   ✓ 成功! 输出形状: {result.shape}")
except Exception as e:
    print(f"   ✗ 失败: {e}")

# 4. 测试单帧作为视频
print("\n4. 测试单帧作为视频 (1帧):")
try:
    # 创建单帧"视频" (1, 1, 3, 160, 256)
    fake_single_frame = torch.randn(1, 1, 3, 160, 256)
    print(f"   输入形状: {fake_single_frame.shape}")
    
    result = model.encode_video(fake_single_frame)
    print(f"   ✓ 成功! 输出形状: {result.shape}")
except Exception as e:
    print(f"   ✗ 失败: {e}")

print("\n" + "=" * 80)
print("结论:")
print("  - 对于单帧图像，使用 encode_video(images.unsqueeze(1))")
print("  - images 形状应该是 (batch, 3, H, W)")
print("  - unsqueeze(1) 后变成 (batch, 1, 3, H, W)")
print("=" * 80)

