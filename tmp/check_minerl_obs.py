"""检查MineRL环境的实际obs格式"""
import minerl
import gym
import numpy as np

print("="*70)
print("MineRL环境obs格式检查")
print("="*70)

try:
    # 创建MineRL环境
    env = gym.make('MineRLNavigateDense-v0')
    obs = env.reset()
    
    print(f"\nMineRL env.reset()返回:")
    print(f"  类型: {type(obs)}")
    
    if isinstance(obs, dict):
        print(f"  keys: {list(obs.keys())}")
        if 'pov' in obs:
            print(f"\n  obs['pov']:")
            print(f"    shape: {obs['pov'].shape}")
            print(f"    dtype: {obs['pov'].dtype}")
            print(f"    min/max: {obs['pov'].min()}/{obs['pov'].max()}")
            
            # 判断格式
            shape = obs['pov'].shape
            print(f"    维度: {shape}")
            
            if len(shape) == 3:
                if shape[2] == 3:
                    print(f"    格式: HWC (shape[-1]=3表示channels在最后)")
                elif shape[0] == 3:
                    print(f"    格式: CHW (shape[0]=3表示channels在最前)")
                else:
                    print(f"    格式: 未知")
    
    # 执行一步
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    
    print(f"\nMineRL env.step()返回的obs:")
    if isinstance(obs, dict) and 'pov' in obs:
        print(f"  obs['pov'] shape: {obs['pov'].shape}")
    
    env.close()
    print("\n✓ 测试完成")
    
except Exception as e:
    print(f"\n❌ 错误: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
