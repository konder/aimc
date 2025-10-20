#!/usr/bin/env python
"""
测试 MineCLIP 模型加载和推理
验证：
1. mineclip 包是否正确安装
2. 能否加载预训练权重 (attn.pth)
3. 推理是否正常工作
"""

import os
import sys
import numpy as np
import torch

print("=" * 70)
print("测试 MineCLIP 模型")
print("=" * 70)
print()

# 1. 检查 mineclip 包
print("[1/5] 检查 mineclip 包...")
try:
    import mineclip
    print(f"  ✓ mineclip 包已安装")
    print(f"  位置: {mineclip.__file__}")
except ImportError as e:
    print(f"  ✗ mineclip 包未安装: {e}")
    print(f"  安装命令: pip install git+https://github.com/MineDojo/MineCLIP")
    sys.exit(1)

print()

# 2. 检查模型类
print("[2/5] 检查 MineCLIP 模型类...")
try:
    from mineclip import MineCLIP
    print(f"  ✓ MineCLIP 类可用")
except ImportError as e:
    print(f"  ✗ 无法导入 MineCLIP: {e}")
    sys.exit(1)

print()

# 3. 尝试创建模型（attn 和 avg）
print("[3/5] 创建模型实例...")

for variant in ["attn", "avg"]:
    try:
        print(f"\n  测试 {variant} 变体:")
        model = MineCLIP(variant=variant)
        print(f"    ✓ {variant} 模型创建成功")
        
        # 检查模型结构
        total_params = sum(p.numel() for p in model.parameters())
        print(f"    参数量: {total_params:,}")
        
    except Exception as e:
        print(f"    ✗ {variant} 模型创建失败: {e}")

print()

# 4. 测试加载预训练权重
print("[4/5] 测试加载预训练权重...")

model_paths = [
    "data/mineclip/attn.pth",
    "data/attn.pth",
    "../data/mineclip/attn.pth",
]

loaded = False
for path in model_paths:
    if os.path.exists(path):
        print(f"\n  找到模型文件: {path}")
        try:
            model = MineCLIP(variant="attn")
            checkpoint = torch.load(path, map_location="cpu")
            
            # 尝试加载
            if isinstance(checkpoint, dict):
                # 可能是完整的 checkpoint
                if 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'])
                elif 'model' in checkpoint:
                    model.load_state_dict(checkpoint['model'])
                else:
                    model.load_state_dict(checkpoint)
            else:
                model.load_state_dict(checkpoint)
            
            print(f"  ✓ 权重加载成功")
            print(f"  文件大小: {os.path.getsize(path) / 1024 / 1024:.1f} MB")
            loaded = True
            break
            
        except Exception as e:
            print(f"  ✗ 加载失败: {e}")
    
if not loaded:
    print(f"\n  ⚠️ 未找到预训练权重")
    print(f"  请下载并放到以下位置之一:")
    for path in model_paths:
        print(f"    - {path}")
    print(f"\n  下载地址: https://github.com/MineDojo/MineCLIP")

print()

# 5. 测试推理
print("[5/5] 测试推理...")

try:
    # 创建模型
    model = MineCLIP(variant="attn")
    model.eval()
    
    # 创建测试数据
    # 图像：batch=1, channels=3, height=160, width=256
    test_image = torch.rand(1, 3, 160, 256)
    
    # 文本
    test_texts = ["chop down a tree and collect wood"]
    
    print(f"\n  测试输入:")
    print(f"    图像: {test_image.shape}")
    print(f"    文本: {test_texts}")
    
    # 推理
    with torch.no_grad():
        # 编码图像
        image_features = model.encode_image(test_image)
        print(f"\n  图像特征: {image_features.shape}")
        
        # 编码文本
        text_features = model.encode_text(test_texts)
        print(f"  文本特征: {text_features.shape}")
        
        # 计算相似度
        similarity = torch.cosine_similarity(image_features, text_features, dim=-1)
        similarity = (similarity + 1.0) / 2.0  # 映射到 [0, 1]
        
        print(f"\n  相似度分数: {similarity.item():.4f}")
        print(f"  ✓ 推理成功！")
        
except Exception as e:
    print(f"  ✗ 推理失败: {e}")
    import traceback
    traceback.print_exc()

print()
print("=" * 70)
print("测试完成")
print("=" * 70)
print()

# 总结
print("下一步:")
if loaded:
    print("  1. ✓ MineCLIP 已就绪，可以开始训练")
    print("  2. 运行: scripts/train_get_wood.sh test --mineclip")
else:
    print("  1. 下载预训练权重 (attn.pth)")
    print("  2. 放到 data/mineclip/ 或 data/ 目录")
    print("  3. 重新运行此脚本验证")

print()

