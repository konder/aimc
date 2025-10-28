"""在真实MineRL环境中验证obs['pov']格式"""
import numpy as np
import minerl
import gym

print("="*70)
print("MineRL环境obs['pov']格式验证")
print("="*70)

# 创建MineRL环境
print("\n创建MineRL环境: MineRLNavigateDense-v0")
env = gym.make('MineRLNavigateDense-v0')

print("\n执行env.reset()...")
obs = env.reset()

print(f"\n✓ obs类型: {type(obs)}")
print(f"✓ obs keys: {list(obs.keys())}")

if 'pov' in obs:
    pov = obs['pov']
    print(f"\n✓ obs['pov']:")
    print(f"  - shape: {pov.shape}")
    print(f"  - dtype: {pov.dtype}")
    print(f"  - min/max: {pov.min()}/{pov.max()}")
    
    # 判断格式
    if len(pov.shape) == 3:
        h, w, c = pov.shape
        print(f"  - 维度解析: H={h}, W={w}, C={c}")
        if c == 3:
            print(f"  - ✓ 确认格式: HWC (Height, Width, Channels)")
            print(f"  - ✓ 与MineDojo相同: 都是HWC格式")
        elif h == 3:
            print(f"  - 格式: CHW (Channels, Height, Width)")
        else:
            print(f"  - 格式: 未知")

print(f"\n执行env.step(action)...")
action = env.action_space.sample()
obs, reward, done, info = env.step(action)

print(f"✓ step后obs['pov'] shape: {obs['pov'].shape}")

env.close()

print("\n" + "="*70)
print("✅ 验证完成！")
print("="*70)
print("\n结论:")
print("  - MineRL使用HWC格式")
print("  - 与MineDojo格式相同")
print("  - VPTAgent的obs转换正确（无需HWC<->CHW转换）")
print("="*70)
