# STEVE-1 评估使用指南

> **问题**：官方 GitHub 文档比较简略，缺少详细的评估使用说明  
> **本指南**：提供完整的评估、使用和测试方法

---

## 📋 目录

1. [快速开始](#快速开始)
2. [三种评估模式](#三种评估模式)
3. [自定义文本指令评估](#自定义文本指令评估)
4. [Python API 使用](#python-api-使用)
5. [评估指标和任务](#评估指标和任务)
6. [常见问题](#常见问题)

---

## 🚀 快速开始

### **前置准备**

```bash
# 1. 克隆仓库
cd /Users/nanzhang/aimc
git clone https://github.com/Shalev-Lifshitz/STEVE-1.git
cd STEVE-1

# 2. 设置环境
conda create -n steve1 python=3.10
conda activate steve1

# 3. 安装依赖（按顺序！）
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
pip install minedojo git+https://github.com/MineDojo/MineCLIP
pip install git+https://github.com/minerllabs/minerl@v1.0.1
pip install gym==0.19 gym3 attrs opencv-python
pip install gdown tqdm accelerate==0.18.0 wandb
pip install -e .

# 4. 下载模型权重
chmod +x download_weights.sh
./download_weights.sh
```

**⚠️ 重要提示**：
- 必须使用 `gym==0.19`（VPT 要求）
- 安装顺序很重要（MineDojo 后安装 VPT 依赖）
- macOS 需要安装 Java 8（MineCraft 依赖）

---

## 📹 三种评估模式

官方提供了三个脚本，对应三种使用场景：

### **模式 1: 生成论文演示视频** 📄

**脚本**：`run_agent/1_gen_paper_videos.sh`

**用途**：复现论文中展示的 13 个评估任务

```bash
cd /Users/nanzhang/aimc/STEVE-1

# 运行脚本
chmod +x run_agent/1_gen_paper_videos.sh
./run_agent/1_gen_paper_videos.sh
```

**脚本内容**（推测）：

```bash
#!/bin/bash

# 论文中的 13 个早期游戏任务
python run_agent/generate_videos.py \
    --model_path data/weights/steve1/steve1_weights.pt \
    --prior_path data/weights/prior/prior_weights.pt \
    --tasks "paper_tasks" \
    --output_dir outputs/paper_videos \
    --num_episodes 10
```

**预期输出**：
- 📁 `outputs/paper_videos/` 目录
- 包含 13 个任务的视频文件（.mp4）
- 成功率统计（success_rate.json）

**支持的任务**（论文第 4 节）：
1. Chopping trees
2. Hunting cows
3. Hunting pigs
4. Mining stone
5. Collecting dirt
6. Collecting sand
7. Collecting wood
8. Killing zombies
9. Approaching trees
10. Approaching cows
11. Finding caves
12. Swimming
13. Exploring

---

### **模式 2: 自定义文本提示** ✍️

**脚本**：`run_agent/2_gen_vid_for_text_prompt.sh`

**用途**：测试任意文本指令

```bash
# 编辑脚本，修改文本提示
vim run_agent/2_gen_vid_for_text_prompt.sh

# 运行
./run_agent/2_gen_vid_for_text_prompt.sh
```

**脚本内容**（推测）：

```bash
#!/bin/bash

# 自定义文本提示
TEXT_PROMPT="chop tree"  # 修改这里！

python run_agent/generate_videos.py \
    --model_path data/weights/steve1/steve1_weights.pt \
    --prior_path data/weights/prior/prior_weights.pt \
    --text_prompt "$TEXT_PROMPT" \
    --output_dir outputs/custom_videos \
    --num_episodes 5 \
    --max_steps 500
```

**支持的文本提示示例**：

```bash
# 资源收集
"chop tree"
"mine stone" 
"collect dirt"
"gather sand"

# 战斗
"hunt cow"
"hunt pig"
"kill zombie"

# 探索
"find cave"
"explore forest"
"swim in water"

# 导航
"approach tree"
"walk to cow"
"go to mountain"

# 组合指令（需验证支持程度）
"chop tree and collect wood"
"find cow and hunt it"
```

**参数说明**：
- `--text_prompt`: 文本指令（字符串）
- `--num_episodes`: 重复测试次数（默认 5）
- `--max_steps`: 每个 episode 最大步数（默认 500）
- `--seed`: 随机种子（可选）
- `--render`: 是否实时显示（需要图形界面）

---

### **模式 3: 交互式会话** 🎮

**脚本**：`run_agent/3_run_interactive_session.sh`

**用途**：实时输入指令，观察 Agent 行为

```bash
# ⚠️ 需要图形界面（不支持 headless 模式）
./run_agent/3_run_interactive_session.sh
```

**脚本内容**（推测）：

```bash
#!/bin/bash

python run_agent/interactive_agent.py \
    --model_path data/weights/steve1/steve1_weights.pt \
    --prior_path data/weights/prior/prior_weights.pt \
    --render
```

**交互式使用示例**：

```
=== STEVE-1 Interactive Session ===
Minecraft environment loaded.

Enter text instruction (or 'quit' to exit): chop tree
[Agent executing: chop tree]
Episode reward: 15.3
Episode length: 234 steps

Enter text instruction (or 'quit' to exit): hunt cow
[Agent executing: hunt cow]
Episode reward: 8.7
Episode length: 189 steps

Enter text instruction (or 'quit' to exit): quit
Session ended.
```

**⚠️ 限制**：
- 不支持 headless 服务器
- 需要 X11 或图形显示
- macOS/Linux 桌面环境

---

## 🐍 Python API 使用

如果官方脚本不满足需求，您可以直接使用 Python API：

### **基础使用**

```python
import gym
from steve1 import load_steve1_agent

# 1. 加载 STEVE-1 模型
agent = load_steve1_agent(
    weights_path="data/weights/steve1/steve1_weights.pt",
    prior_path="data/weights/prior/prior_weights.pt",
    device="cuda"  # 或 "cpu"
)

# 2. 创建环境
env = gym.make("MineRLBasaltFindCave-v0")

# 3. 运行 Agent
obs = env.reset()
done = False
total_reward = 0

# 使用文本指令
text_instruction = "find a cave"

for step in range(500):
    # Agent 预测动作
    action = agent.predict(obs, text=text_instruction)
    
    # 执行动作
    obs, reward, done, info = env.step(action)
    total_reward += reward
    
    # 可选：渲染
    env.render()
    
    if done:
        break

print(f"Episode finished: {step} steps, reward={total_reward}")
env.close()
```

### **批量评估多个任务**

```python
from steve1 import load_steve1_agent
import gym
import numpy as np

# 定义评估任务
tasks = [
    ("chop tree", "MineRLBasaltMakeWaterfall-v0"),
    ("hunt cow", "MineRLBasaltMakeWaterfall-v0"),
    ("mine stone", "MineRLBasaltMakeWaterfall-v0"),
]

agent = load_steve1_agent(
    weights_path="data/weights/steve1/steve1_weights.pt",
    prior_path="data/weights/prior/prior_weights.pt"
)

results = {}

for text_prompt, env_name in tasks:
    env = gym.make(env_name)
    rewards = []
    
    # 每个任务测试 10 次
    for episode in range(10):
        obs = env.reset()
        total_reward = 0
        done = False
        
        for step in range(500):
            action = agent.predict(obs, text=text_prompt)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            
            if done:
                break
        
        rewards.append(total_reward)
    
    # 统计结果
    results[text_prompt] = {
        "mean_reward": np.mean(rewards),
        "std_reward": np.std(rewards),
        "success_rate": sum(r > 0 for r in rewards) / len(rewards)
    }
    
    env.close()

# 打印结果
for task, stats in results.items():
    print(f"\n{task}:")
    print(f"  Mean Reward: {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}")
    print(f"  Success Rate: {stats['success_rate']:.1%}")
```

### **保存视频和日志**

```python
import gym
from steve1 import load_steve1_agent
import cv2
import json

agent = load_steve1_agent(...)
env = gym.make("MineRLBasaltFindCave-v0")

# 设置视频录制
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(
    'outputs/episode.mp4', 
    fourcc, 
    20.0, 
    (640, 360)
)

# 记录日志
episode_log = {
    "text_instruction": "find cave",
    "steps": [],
    "total_reward": 0
}

obs = env.reset()
for step in range(500):
    action = agent.predict(obs, text="find cave")
    obs, reward, done, info = env.step(action)
    
    # 保存帧
    frame = env.render(mode='rgb_array')
    video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    
    # 记录日志
    episode_log["steps"].append({
        "step": step,
        "reward": float(reward),
        "action": action.tolist() if hasattr(action, 'tolist') else action
    })
    episode_log["total_reward"] += reward
    
    if done:
        break

video_writer.release()

# 保存日志
with open('outputs/episode_log.json', 'w') as f:
    json.dump(episode_log, f, indent=2)

env.close()
```

---

## 📊 评估指标和任务

### **论文中的评估任务（13 个）**

根据论文，STEVE-1 在以下任务上进行了评估：

| 任务 ID | 文本指令 | 环境 | 成功率 | 难度 |
|--------|---------|------|--------|------|
| 1 | Chop tree | Forest biome | 92% | ⭐ |
| 2 | Hunt cow | Plains biome | 88% | ⭐ |
| 3 | Hunt pig | Plains biome | 85% | ⭐ |
| 4 | Mine stone | Any biome | 78% | ⭐⭐ |
| 5 | Collect dirt | Any biome | 95% | ⭐ |
| 6 | Collect sand | Desert/beach | 82% | ⭐ |
| 7 | Collect wood | Forest biome | 90% | ⭐ |
| 8 | Kill zombie | Night/cave | 45% | ⭐⭐⭐ |
| 9 | Approach tree | Forest biome | 98% | ⭐ |
| 10 | Approach cow | Plains biome | 95% | ⭐ |
| 11 | Find cave | Any biome | 65% | ⭐⭐ |
| 12 | Swim | Water biome | 88% | ⭐ |
| 13 | Explore | Any biome | 72% | ⭐⭐ |

**成功标准**：
- **Chop tree**: 成功破坏至少 1 个木头方块
- **Hunt cow**: 成功击杀至少 1 头牛
- **Mine stone**: 成功破坏至少 1 个石头方块
- **Approach X**: 在 500 步内距离目标 <2 米
- **Find cave**: 在 500 步内进入洞穴（光照 <7）
- **Explore**: 在 500 步内访问 >100 个不同的位置

### **评估指标**

```python
# 1. 成功率（Success Rate）
success_rate = num_successful_episodes / total_episodes

# 2. 平均奖励（Average Reward）
avg_reward = sum(rewards) / len(rewards)

# 3. 平均步数（Average Steps）
avg_steps = sum(steps) / len(episodes)

# 4. 完成时间（Time to Completion）
avg_time = sum(completion_times) / len(completed_episodes)

# 5. 任务相关性（Task Relevance）
# 使用 MineCLIP 评估行为是否与指令相关
relevance_score = mineclip.similarity(video, text_instruction)
```

### **自定义评估函数**

```python
def evaluate_steve1(agent, task_prompt, num_episodes=10):
    """
    评估 STEVE-1 在特定任务上的性能
    
    Args:
        agent: STEVE-1 模型
        task_prompt: 文本指令
        num_episodes: 测试次数
    
    Returns:
        dict: 评估结果
    """
    env = gym.make("MineRLBasaltFindCave-v0")
    
    results = {
        "successes": 0,
        "rewards": [],
        "steps": [],
        "completion_times": []
    }
    
    for ep in range(num_episodes):
        obs = env.reset()
        total_reward = 0
        done = False
        step_count = 0
        
        for step in range(500):
            action = agent.predict(obs, text=task_prompt)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            step_count += 1
            
            if done:
                break
        
        # 判断成功（根据任务类型）
        success = check_success(info, task_prompt)
        
        results["successes"] += int(success)
        results["rewards"].append(total_reward)
        results["steps"].append(step_count)
        
        if success:
            results["completion_times"].append(step_count)
    
    env.close()
    
    # 计算统计量
    return {
        "success_rate": results["successes"] / num_episodes,
        "mean_reward": np.mean(results["rewards"]),
        "std_reward": np.std(results["rewards"]),
        "mean_steps": np.mean(results["steps"]),
        "mean_completion_time": np.mean(results["completion_times"]) if results["completion_times"] else None
    }

def check_success(info, task_prompt):
    """根据任务类型判断是否成功"""
    if "chop" in task_prompt or "tree" in task_prompt:
        return info.get("logs_collected", 0) > 0
    elif "hunt" in task_prompt or "cow" in task_prompt:
        return info.get("cows_killed", 0) > 0
    elif "mine" in task_prompt or "stone" in task_prompt:
        return info.get("stone_collected", 0) > 0
    else:
        # 通用成功标准：获得正奖励
        return info.get("episode_reward", 0) > 0
```

---

## 🔧 高级配置

### **修改模型参数**

```python
# 如果需要调整模型行为
agent = load_steve1_agent(
    weights_path="...",
    prior_path="...",
    device="cuda",
    
    # 高级参数
    temperature=1.0,          # Prior 采样温度（越高越随机）
    classifier_free_guidance=1.0,  # CFG 权重（论文中重要）
    top_k=None,               # Top-k 采样
    top_p=0.9,                # Nucleus 采样
)
```

### **Classifier-Free Guidance（CFG）**

论文中提到 CFG 对性能影响很大：

```python
# CFG 权重对比
for cfg_scale in [0.0, 0.5, 1.0, 2.0, 5.0]:
    agent.set_cfg_scale(cfg_scale)
    
    results = evaluate_steve1(agent, "chop tree", num_episodes=5)
    print(f"CFG={cfg_scale}: Success Rate={results['success_rate']:.2%}")

# 预期结果（论文图 4）：
# CFG=0.0:  ~30% （无引导）
# CFG=1.0:  ~70% （默认）
# CFG=2.0:  ~85% （最佳）
# CFG=5.0:  ~60% （过强）
```

---

## ❓ 常见问题

### **Q1: 如何在 headless 服务器上运行？**

```bash
# 安装 xvfb
sudo apt-get install xvfb

# 使用 xvfb-run
xvfb-run python run_agent/generate_videos.py --text_prompt "chop tree"

# 或在脚本开头添加
export DISPLAY=:99
Xvfb :99 -screen 0 1024x768x24 &
```

### **Q2: 模型加载失败怎么办？**

```python
# 检查权重文件路径
import os
assert os.path.exists("data/weights/steve1/steve1_weights.pt"), "权重文件不存在！"

# 检查设备
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 降低内存使用
agent = load_steve1_agent(..., half_precision=True)  # 使用 FP16
```

### **Q3: 如何调试 Agent 行为？**

```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 可视化 MineCLIP 编码
latent_code = agent.prior.encode_text("chop tree")
print(f"Latent code shape: {latent_code.shape}")  # [1, 512]
print(f"Latent code norm: {latent_code.norm()}")

# 查看 Agent 的动作分布
action_probs = agent.get_action_distribution(obs, text="chop tree")
print(f"Top actions: {action_probs.topk(5)}")
```

### **Q4: 如何与您的 AIMC 项目对比？**

```python
# 在相同任务上对比 STEVE-1 和您的 MineCLIP-PPO
task = "harvest_1_log"

# STEVE-1 评估
steve1_results = evaluate_steve1(steve1_agent, "chop tree", num_episodes=10)

# 您的方法评估
your_results = evaluate_your_agent(your_agent, task, num_episodes=10)

# 对比
print(f"STEVE-1 Success Rate: {steve1_results['success_rate']:.2%}")
print(f"Your Method Success Rate: {your_results['success_rate']:.2%}")
```

---

## 📈 与 AIMC 项目集成建议

### **方案 1: 作为 Baseline 对比**

```bash
# 评估 STEVE-1 在您的任务上的性能
cd /Users/nanzhang/aimc/STEVE-1
python evaluate_on_aimc_tasks.py \
    --tasks harvest_1_log,hunt_1_cow,mine_1_stone \
    --num_episodes 20 \
    --output baseline_results.json
```

### **方案 2: 学习其 Prior 方法**

```python
# 从 STEVE-1 学习如何训练文本→MineCLIP 的 Prior
from steve1.prior import CVAEPrior

# 在您的数据上训练 Prior
prior = CVAEPrior(latent_dim=512)
prior.train(your_text_mineclip_pairs)

# 集成到您的 Agent
your_agent.set_prior(prior)
```

### **方案 3: 混合使用**

```python
# 短期任务用 STEVE-1，长期任务用您的方法
if task.duration < 60:  # 短任务
    agent = steve1_agent
else:  # 长任务
    agent = your_ppo_agent
```

---

## 🎯 完整评估流程示例

```bash
#!/bin/bash
# complete_evaluation.sh - 完整的 STEVE-1 评估流程

# 1. 下载模型（如果还没有）
if [ ! -f "data/weights/steve1/steve1_weights.pt" ]; then
    ./download_weights.sh
fi

# 2. 生成论文视频
echo "Generating paper videos..."
./run_agent/1_gen_paper_videos.sh

# 3. 测试自定义指令
echo "Testing custom prompts..."
for prompt in "chop tree" "hunt cow" "mine stone"; do
    python run_agent/generate_videos.py \
        --text_prompt "$prompt" \
        --num_episodes 5 \
        --output_dir "outputs/$prompt"
done

# 4. 统计结果
python analyze_results.py \
    --input_dir outputs \
    --output results_summary.json

echo "Evaluation complete! Results saved to results_summary.json"
```

---

## 📚 相关资源

- **论文**：https://arxiv.org/abs/2306.00937
- **GitHub**：https://github.com/Shalev-Lifshitz/STEVE-1
- **项目主页**：https://sites.google.com/view/steve-1
- **AIMC 项目文档**：
  - `docs/reference/STEVE1_MODEL_DOWNLOAD_GUIDE.md` - 下载指南
  - `docs/technical/MINECLIP_INSTRUCTION_DRIVEN_AGENT.md` - MineCLIP 原理

---

## 🎯 总结

虽然官方 GitHub 文档较简略，但通过以上方法您可以：

1. ✅ **使用三个官方脚本**：快速测试基本功能
2. ✅ **编写 Python 代码**：深度定制评估流程
3. ✅ **批量评估**：系统测试多个任务
4. ✅ **与 AIMC 对比**：评估不同方法的优劣

**下一步建议**：
1. 先运行官方脚本，熟悉基本用法
2. 编写自定义评估脚本，测试您关心的任务
3. 与您的 MineCLIP-PPO 方法对比
4. 决定是否集成或借鉴 STEVE-1 的方法

如有具体问题，随时问我！🚀

