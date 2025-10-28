"""测试VPT Agent with debug"""
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

from src.training.vpt import VPTAgent

print("="*70)
print("测试VPT Agent (启用debug)")
print("="*70)

# 创建Agent (启用debug_actions)
agent = VPTAgent(
    vpt_weights_path="data/pretrained/vpt/rl-from-early-game-2x.weights",
    device='auto',
    verbose=True,
    debug_actions=True  # 启用调试
)

agent.reset()

# 测试
fake_obs = np.random.randint(0, 255, (160, 256, 3), dtype=np.uint8)
print(f"\n输入观察 shape: {fake_obs.shape}")

try:
    action = agent.predict(fake_obs)
    print(f"\n✓ 成功! 输出动作: {action}")
except Exception as e:
    print(f"\n❌ 错误: {e}")

