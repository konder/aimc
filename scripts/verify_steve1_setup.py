#!/usr/bin/env python3
"""
验证 STEVE-1 配置和设备设置
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

print("="*70)
print("STEVE-1 环境验证")
print("="*70)

# 1. 检查 PyTorch
print("\n1. PyTorch 检测:")
try:
    import torch
    print(f"   ✅ PyTorch 版本: {torch.__version__}")
    print(f"   ✅ CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   ✅ CUDA 版本: {torch.version.cuda}")
    else:
        print(f"   ℹ️  将使用 CPU 运行")
except Exception as e:
    print(f"   ❌ PyTorch 错误: {e}")

# 2. 检查配置
print("\n2. STEVE-1 配置:")
try:
    from src.training.steve1.config import DEVICE, MINECLIP_CONFIG, PRIOR_INFO, DATA_DIR
    print(f"   ✅ 设备设置: {DEVICE}")
    print(f"   ✅ DATA_DIR: {DATA_DIR}")
    print(f"   ✅ MineCLIP 权重路径: {MINECLIP_CONFIG['ckpt']['path']}")
    print(f"   ✅ Prior 权重路径: {PRIOR_INFO['model_path']}")
    
    # 检查文件是否存在
    mineclip_exists = os.path.exists(MINECLIP_CONFIG['ckpt']['path'])
    prior_exists = os.path.exists(PRIOR_INFO['model_path'])
    
    print(f"\n   MineCLIP 权重文件存在: {'✅' if mineclip_exists else '❌'}")
    print(f"   Prior 权重文件存在: {'✅' if prior_exists else '❌'}")
    
except Exception as e:
    print(f"   ❌ 配置错误: {e}")
    import traceback
    traceback.print_exc()

# 3. 检查 Hugging Face 缓存
print("\n3. Hugging Face 缓存:")
try:
    hf_cache = os.environ.get('HF_HOME', '~/.cache/huggingface')
    hf_cache = os.path.expanduser(hf_cache)
    print(f"   HF_HOME: {hf_cache}")
    
    tokenizer_path = os.path.join(
        hf_cache, 
        "hub/models--openai--clip-vit-base-patch16/snapshots"
    )
    
    if os.path.exists(tokenizer_path):
        snapshots = os.listdir(tokenizer_path)
        snapshots = [s for s in snapshots if not s.startswith('.')]
        print(f"   ✅ 找到 {len(snapshots)} 个 snapshot(s)")
        for snapshot in snapshots:
            snapshot_dir = os.path.join(tokenizer_path, snapshot)
            if os.path.isdir(snapshot_dir):
                files = os.listdir(snapshot_dir)
                files = [f for f in files if not f.startswith('.')]
                print(f"      - {snapshot[:12]}... ({len(files)} 个文件)")
    else:
        print(f"   ⚠️  缓存目录不存在: {tokenizer_path}")
        
except Exception as e:
    print(f"   ❌ 缓存检查错误: {e}")

# 4. 测试 Tokenizer 加载
print("\n4. Tokenizer 加载测试:")
try:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "openai/clip-vit-base-patch16",
        local_files_only=True
    )
    print(f"   ✅ Tokenizer 加载成功")
    print(f"   ✅ 词汇表大小: {len(tokenizer)}")
except Exception as e:
    print(f"   ❌ Tokenizer 加载失败: {e}")

print("\n" + "="*70)
print("验证完成！")
print("="*70)

