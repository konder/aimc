from src.training.steve1 import load_steve1_agent
from src.training.steve1.config import DEVICE
import gym

# 加载模型（自动检测设备类型）
agent = load_steve1_agent(
    weights_path="data/weights/steve1/steve1_weights.pt",
    prior_path="data/weights/prior/prior_weights.pt",
    device=DEVICE
)

# 创建环境
env = gym.make("MineRLBasaltFindCave-v0")

# 运行评估
obs = env.reset()
for step in range(500):
    # 根据文本指令生成动作
    action = agent.predict(obs, text="mine stone")
    obs, reward, done, info = env.step(action)
    env.render()
    
    if done:
        break

env.close()