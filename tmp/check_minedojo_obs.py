"""检查MineDojo环境返回的观察格式"""
import minedojo

env = minedojo.make(task_id="harvest_1_log", image_size=(160, 256))
obs = env.reset()

print("="*70)
print("MineDojo观察格式检查")
print("="*70)
print(f"\nobs类型: {type(obs)}")
print(f"obs键: {obs.keys() if isinstance(obs, dict) else 'N/A'}")

if isinstance(obs, dict) and 'rgb' in obs:
    print(f"\nobs['rgb']:")
    print(f"  类型: {type(obs['rgb'])}")
    print(f"  shape: {obs['rgb'].shape}")
    print(f"  dtype: {obs['rgb'].dtype}")
    print(f"  min/max: {obs['rgb'].min()}/{obs['rgb'].max()}")

env.close()
print("\n" + "="*70)
