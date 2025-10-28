"""调试VPT输入shape"""
import numpy as np
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path.cwd()))

from src.training.vpt import VPTAgent

# 创建Agent
agent = VPTAgent(
    vpt_weights_path="data/pretrained/vpt/rl-from-early-game-2x.weights",
    device='mps',
    verbose=False
)

# Monkey patch _env_obs_to_agent来打印shape
original_env_obs_to_agent = agent.vpt_agent._env_obs_to_agent

def debug_env_obs_to_agent(minerl_obs):
    print(f"\n[DEBUG] _env_obs_to_agent输入:")
    print(f"  minerl_obs['pov'] shape: {minerl_obs['pov'].shape}")
    print(f"  minerl_obs['pov'] dtype: {minerl_obs['pov'].dtype}")
    result = original_env_obs_to_agent(minerl_obs)
    print(f"\n[DEBUG] _env_obs_to_agent输出:")
    print(f"  result['img'] shape: {result['img'].shape}")
    print(f"  result['img'] dtype: {result['img'].dtype}")
    return result

agent.vpt_agent._env_obs_to_agent = debug_env_obs_to_agent

# 测试
print("="*70)
print("测试MineDojo观察")
print("="*70)

fake_obs = np.random.randint(0, 255, (160, 256, 3), dtype=np.uint8)
print(f"\n输入MineDojo观察 shape: {fake_obs.shape}  # (H, W, C)")

try:
    action = agent.predict(fake_obs)
    print(f"\n✓ 成功! 输出动作: {action}")
except Exception as e:
    print(f"\n❌ 错误: {e}")

print("\n" + "="*70)
