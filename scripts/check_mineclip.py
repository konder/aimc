#!/usr/bin/env python
"""
检查 MineCLIP 在 MineDojo 中的可用性
"""

import sys

print("=" * 70)
print("检查 MineCLIP 可用性")
print("=" * 70)
print()

# 1. 检查 MineDojo 版本
try:
    import minedojo
    print(f"✓ MineDojo 版本: {minedojo.__version__}")
except Exception as e:
    print(f"✗ MineDojo 导入失败: {e}")
    sys.exit(1)

print()

# 2. 检查 MineDojo 中的 MineCLIP 相关属性
print("检查 MineDojo 的 MineCLIP 相关功能:")
print("-" * 70)

# 检查顶层模块
minedojo_attrs = dir(minedojo)
mineclip_related = [attr for attr in minedojo_attrs if 'clip' in attr.lower()]
print(f"MineDojo 顶层包含 'clip' 的属性: {mineclip_related if mineclip_related else '无'}")

# 3. 检查 sim 模块
try:
    from minedojo import sim
    sim_attrs = dir(sim)
    sim_clip_related = [attr for attr in sim_attrs if 'clip' in attr.lower()]
    print(f"MineDojo.sim 包含 'clip' 的属性: {sim_clip_related if sim_clip_related else '无'}")
except Exception as e:
    print(f"✗ 无法导入 minedojo.sim: {e}")

# 4. 检查 sim.wrappers
try:
    from minedojo.sim import wrappers
    wrapper_attrs = dir(wrappers)
    print(f"\nMineDojo.sim.wrappers 可用的包装器:")
    for attr in wrapper_attrs:
        if not attr.startswith('_'):
            print(f"  - {attr}")
except Exception as e:
    print(f"✗ 无法导入 minedojo.sim.wrappers: {e}")

print()

# 5. 尝试直接导入 MineCLIP
print("尝试导入 MineCLIP 相关模块:")
print("-" * 70)

attempts = [
    ("minedojo.mineclip", "MineDojo 内置 MineCLIP"),
    ("minedojo.sim.mineclip", "sim.mineclip"),
    ("minedojo.sim.wrappers.MineCLIPWrapper", "MineCLIP Wrapper"),
]

for module_path, description in attempts:
    try:
        parts = module_path.split('.')
        if len(parts) == 2:
            exec(f"from {parts[0]} import {parts[1]}")
        elif len(parts) == 3:
            exec(f"from {parts[0]}.{parts[1]} import {parts[2]}")
        print(f"✓ {description}: {module_path} - 可用")
    except ImportError as e:
        print(f"✗ {description}: {module_path} - 不可用")
    except Exception as e:
        print(f"⚠ {description}: {module_path} - 错误: {e}")

print()

# 6. 检查是否有独立的 mineclip 包
print("检查独立的 mineclip 包:")
print("-" * 70)
try:
    import mineclip
    print(f"✓ 独立的 mineclip 包已安装")
    print(f"  版本: {getattr(mineclip, '__version__', '未知')}")
    print(f"  位置: {mineclip.__file__}")
    print(f"  可用属性: {[attr for attr in dir(mineclip) if not attr.startswith('_')][:10]}")
except ImportError:
    print(f"✗ 独立的 mineclip 包未安装")
    print(f"  可以尝试安装: pip install mineclip")

print()

# 7. 创建一个环境并检查其属性
print("创建 MineDojo 环境并检查:")
print("-" * 70)
try:
    env = minedojo.make(task_id="harvest_1_log", image_size=(160, 256))
    env_attrs = dir(env)
    env_clip_related = [attr for attr in env_attrs if 'clip' in attr.lower() or 'reward' in attr.lower()]
    print(f"环境对象包含 'clip' 或 'reward' 的属性:")
    for attr in env_clip_related[:10]:
        print(f"  - {attr}")
    
    # 检查环境是否有 MineCLIP 相关方法
    if hasattr(env, 'get_mineclip_reward'):
        print("  ✓ 环境有 get_mineclip_reward 方法")
    else:
        print("  ✗ 环境没有 get_mineclip_reward 方法")
    
    env.close()
except Exception as e:
    print(f"✗ 创建环境失败: {e}")

print()
print("=" * 70)
print("检查完成")
print("=" * 70)

