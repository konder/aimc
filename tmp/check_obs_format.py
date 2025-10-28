"""检查MineDojo环境返回的obs格式"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

print("="*70)
print("检查MineDojo环境的obs格式")
print("="*70)

try:
    import minedojo
    
    env = minedojo.make(task_id="harvest_1_log", image_size=(160, 256))
    obs = env.reset()
    
    print(f"\nMineDojo env.reset()返回:")
    print(f"  类型: {type(obs)}")
    
    if isinstance(obs, dict):
        print(f"  keys: {obs.keys()}")
        if 'rgb' in obs:
            print(f"\n  obs['rgb']:")
            print(f"    shape: {obs['rgb'].shape}")
            print(f"    dtype: {obs['rgb'].dtype}")
            print(f"    min/max: {obs['rgb'].min()}/{obs['rgb'].max()}")
            print(f"    格式推断: ", end="")
            if obs['rgb'].shape[2] == 3:
                print("HWC (Height, Width, Channels)")
            elif obs['rgb'].shape[0] == 3:
                print("CHW (Channels, Height, Width)")
            else:
                print("未知")
    else:
        print(f"  shape: {obs.shape}")
        print(f"  dtype: {obs.dtype}")
    
    env.close()
    
except Exception as e:
    print(f"\n❌ MineDojo测试失败: {e}")

print("\n" + "="*70)
print("查看MineRL环境的obs格式（从文档）")
print("="*70)

print("""
根据MineRL官方文档：
- MineRL环境返回: Dict observation space
- obs["pov"]: RGB图像观察
- 格式: numpy.ndarray
- shape: (height, width, 3) - HWC格式
- dtype: uint8
- 范围: [0, 255]

结论：MineRL使用HWC格式
""")

print("="*70)
