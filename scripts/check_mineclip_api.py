#!/usr/bin/env python
"""
检查 MineCLIP 的正确 API
"""

print("检查 mineclip 包的 API...")
print()

# 1. 导入 mineclip
try:
    import mineclip
    print("✓ mineclip 包已导入")
    print()
    
    # 2. 查看包的内容
    print("mineclip 包含的内容:")
    attrs = [attr for attr in dir(mineclip) if not attr.startswith('_')]
    for attr in attrs:
        print(f"  - {attr}")
    print()
    
    # 3. 检查是否有 MineCLIP 类
    if hasattr(mineclip, 'MineCLIP'):
        print("✓ 发现 MineCLIP 类")
        MineCLIP = mineclip.MineCLIP
        
        # 检查初始化参数
        import inspect
        sig = inspect.signature(MineCLIP.__init__)
        print(f"  MineCLIP.__init__ 参数: {sig}")
        print()
    
    # 4. 检查是否有分开的类
    for variant in ['MineCLIP_attn', 'MineCLIP_avg', 'MineCLIPAttn', 'MineCLIPAvg']:
        if hasattr(mineclip, variant):
            print(f"✓ 发现 {variant} 类")
    
    # 5. 检查模块结构
    print()
    print("mineclip 模块结构:")
    if hasattr(mineclip, '__file__'):
        print(f"  文件位置: {mineclip.__file__}")
    
    # 尝试查看子模块
    if hasattr(mineclip, '__path__'):
        import os
        mineclip_dir = mineclip.__path__[0]
        print(f"  包目录: {mineclip_dir}")
        print("  包含文件:")
        for item in os.listdir(mineclip_dir):
            if not item.startswith('_') and not item.startswith('.'):
                print(f"    - {item}")
    
except ImportError as e:
    print(f"✗ 无法导入 mineclip: {e}")
except Exception as e:
    print(f"✗ 发生错误: {e}")
    import traceback
    traceback.print_exc()

