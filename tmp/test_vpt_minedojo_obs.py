"""测试VPT Agent处理MineDojo观察格式"""

import numpy as np
from src.training.vpt import VPTAgent

print("="*70)
print("测试VPT Agent处理MineDojo观察格式")
print("="*70)

# 创建Agent
agent = VPTAgent(
    vpt_weights_path="data/pretrained/vpt/rl-from-early-game-2x.weights",
    device='auto',
    verbose=False
)

agent.reset()

# 测试1: numpy数组格式（旧格式）
print("\n1. 测试numpy数组格式:")
obs_array = np.random.randint(0, 255, (160, 256, 3), dtype=np.uint8)
action1 = agent.predict(obs_array)
print(f"   输入: numpy array shape={obs_array.shape}")
print(f"   输出: {action1}")
print("   ✓ numpy数组格式测试通过")

# 测试2: MineDojo字典格式（新格式）
print("\n2. 测试MineDojo字典格式:")
obs_dict = {'rgb': np.random.randint(0, 255, (160, 256, 3), dtype=np.uint8)}
action2 = agent.predict(obs_dict)
print(f"   输入: dict with 'rgb' key, shape={obs_dict['rgb'].shape}")
print(f"   输出: {action2}")
print("   ✓ MineDojo字典格式测试通过")

print("\n" + "="*70)
print("✅ 所有观察格式测试通过！")
print("="*70)
