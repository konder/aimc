#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
检查 MineCLIP 对象的可用方法
"""
import sys
sys.path.insert(0, '/Users/nanzhang/aimc/src')

try:
    from mineclip import MineCLIP
    
    # 创建一个 MineCLIP 实例
    config = {
        "arch": "vit_base_p16_fz.v2.t2",
        "pool_type": "attn.d2.nh8.glusw",
        "resolution": (160, 256),
        "image_feature_dim": 512,
        "mlp_adapter_spec": "v0-2.t0",
        "hidden_dim": 512
    }
    
    model = MineCLIP(**config)
    
    print("=" * 80)
    print("MineCLIP 对象的所有属性和方法:")
    print("=" * 80)
    
    # 获取所有属性和方法
    all_attrs = dir(model)
    
    # 分类显示
    print("\n📋 公开方法 (不以 _ 开头):")
    public_methods = [attr for attr in all_attrs if not attr.startswith('_') and callable(getattr(model, attr))]
    for method in sorted(public_methods):
        print(f"  - {method}")
    
    print("\n🔧 公开属性 (不以 _ 开头):")
    public_attrs = [attr for attr in all_attrs if not attr.startswith('_') and not callable(getattr(model, attr))]
    for attr in sorted(public_attrs):
        print(f"  - {attr}: {type(getattr(model, attr))}")
    
    # 检查特定方法
    print("\n🔍 检查常见编码方法:")
    encode_methods = [
        'encode_image',
        'encode_video', 
        'encode_text',
        'forward',
        'compute_video',
        'compute_image',
        'compute_text',
    ]
    
    for method in encode_methods:
        if hasattr(model, method):
            attr = getattr(model, method)
            print(f"  ✓ {method}: {type(attr)}")
            if callable(attr):
                import inspect
                try:
                    sig = inspect.signature(attr)
                    print(f"    参数: {sig}")
                except:
                    print(f"    (无法获取签名)")
        else:
            print(f"  ✗ {method}: 不存在")
    
    print("\n" + "=" * 80)
    
except Exception as e:
    print(f"错误: {e}")
    import traceback
    traceback.print_exc()

