# MineDojo 技能训练加速指南

## 问题背景

从零开始强化学习训练Minecraft技能（如砍树、采矿等）面临的核心挑战：

- ❌ **稀疏奖励**：智能体可能需要数千步才能获得第一次正向奖励
- ❌ **探索困难**：动作空间大（数百个离散动作组合），状态空间更大
- ❌ **训练时间长**：可能需要数百万步（几天到几周）才能学会基础技能
- ❌ **样本效率低**：纯强化学习需要大量试错

**目标**：训练多个技能并通过agent组合评估 → 需要高效的单技能训练方法

---

## 🚀 方案一：模仿学习（Imitation Learning）【推荐】

### 1.1 使用MineDojo的YouTube视频数据集

MineDojo提供了**大规模YouTube游戏视频数据集**，包含数千小时的人类玩家游戏录像。

#### 实现步骤

**第一步：使用MineCLIP作为奖励函数**

MineCLIP是MineDojo提供的预训练视觉-语言模型，可以将文字任务描述与游戏画面关联。

```python
import minedojo
from minedojo.sim import MinecraftSim
import torch

# 创建带MineCLIP奖励的环境
env = minedojo.make(
    task_id="harvest_log",  # 砍树任务
    image_size=(160, 256),
    reward_fn="mineclip",  # 使用MineCLIP作为奖励
    use_voxel=False,
)

# MineCLIP会根据任务描述自动计算密集奖励
# 例如："chop down a tree" → 靠近树木、挥动斧头、获得木头都会有奖励
```

**优点**：
- ✅ 将稀疏奖励转为密集奖励
- ✅ 无需人工标注数据
- ✅ 可以直接用文字描述新任务

**第二步：行为克隆（Behavior Cloning）**

如果有人类演示数据，可以先用监督学习预训练策略。

```python
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
import numpy as np

# 1. 收集人类演示数据（可以自己玩或使用MineDojo数据集）
# MineDojo提供了contractor数据集（人类专家录制）
env = minedojo.make(
    task_id="harvest_log",
    image_size=(160, 256),
)

# 2. 行为克隆预训练
# 假设你有demonstrations.pkl文件（obs, actions对）
from imitation.algorithms import bc
from imitation.data import rollout

# 加载演示数据
demonstrations = load_demonstrations("demos/harvest_log.pkl")

# 训练行为克隆模型
bc_trainer = bc.BC(
    observation_space=env.observation_space,
    action_space=env.action_space,
    demonstrations=demonstrations,
    batch_size=32,
)
bc_trainer.train(n_epochs=100)

# 3. 用行为克隆模型初始化PPO
model = PPO("CnnPolicy", env, verbose=1)
model.policy.load_state_dict(bc_trainer.policy.state_dict())

# 4. 继续强化学习微调
model.learn(total_timesteps=500000)
```

**第三步：使用VPT（Video Pre-Training）**

OpenAI开发的VPT方法，可以从未标注的YouTube视频学习。

参考项目：
- [Video Pre-Training (VPT)](https://github.com/openai/Video-Pre-Training)
- [MineRL BASALT Competition](https://www.aicrowd.com/challenges/neurips-2022-minerl-basalt-competition)

```bash
# 安装VPT相关依赖
pip install video-pre-training
pip install minerl

# 使用预训练的VPT模型
from vpt import load_model

# 加载预训练基础模型
model = load_model("vpt-foundation-2x.model")

# 在MineDojo环境中使用
# VPT模型可以作为初始策略，然后在MineDojo中微调
```

---

## 🎯 方案二：课程学习（Curriculum Learning）

### 2.1 渐进式任务难度

不要直接训练复杂任务，而是从简单任务开始逐步增加难度。

**示例：砍树技能的课程设计**

```python
# 阶段1：导航 - 学习移动和视角控制
curriculum = [
    # Level 1: 简单移动（平地，目标近）
    {
        "task": "navigate_to_block",
        "target": "oak_log",
        "distance": 10,
        "terrain": "flat",
        "timesteps": 50000,
    },
    
    # Level 2: 复杂导航（有障碍，目标远）
    {
        "task": "navigate_to_block",
        "target": "oak_log",
        "distance": 50,
        "terrain": "forest",
        "timesteps": 100000,
    },
    
    # Level 3: 砍树（近距离，斧头在手）
    {
        "task": "harvest_log",
        "initial_items": [{"type": "wooden_axe", "quantity": 1}],
        "spawn_near_tree": True,
        "timesteps": 100000,
    },
    
    # Level 4: 完整任务（需要找树、砍树）
    {
        "task": "harvest_log",
        "initial_items": [],
        "spawn_near_tree": False,
        "timesteps": 200000,
    },
]

# 训练循环
for level in curriculum:
    print(f"Training level: {level['task']}")
    
    # 创建环境
    env = create_custom_env(**level)
    
    # 如果有上一阶段的模型，加载它
    if previous_model is not None:
        model = PPO.load(previous_model)
        model.set_env(env)
    else:
        model = PPO("CnnPolicy", env)
    
    # 训练当前阶段
    model.learn(total_timesteps=level['timesteps'])
    
    # 保存模型供下一阶段使用
    previous_model = f"checkpoints/curriculum_level_{level['task']}.zip"
    model.save(previous_model)
```

### 2.2 自动课程学习（Automatic Curriculum Learning）

使用PLR（Prioritized Level Replay）等方法自动调整任务难度。

```python
# 伪代码示例
class AutoCurriculumEnv:
    def __init__(self):
        self.difficulty_level = 0
        self.success_rate = []
    
    def reset(self):
        # 根据最近成功率调整难度
        recent_success = np.mean(self.success_rate[-10:])
        
        if recent_success > 0.7:
            self.difficulty_level += 1  # 增加难度
        elif recent_success < 0.3:
            self.difficulty_level = max(0, self.difficulty_level - 1)  # 降低难度
        
        # 生成相应难度的环境配置
        config = self.generate_config(self.difficulty_level)
        return self.env.reset(config)
```

---

## 🎁 方案三：奖励塑形（Reward Shaping）

### 3.1 手工设计中间奖励

为任务的中间步骤提供奖励，引导智能体学习。

**示例：砍树任务的奖励函数**

```python
class RewardShapedEnv:
    def __init__(self, base_env):
        self.base_env = base_env
        self.previous_inventory = {}
        self.previous_distance_to_tree = float('inf')
        
    def step(self, action):
        obs, reward, done, info = self.base_env.step(action)
        
        # 原始任务奖励（获得木头）
        shaped_reward = reward
        
        # 1. 靠近树木 → +0.01
        current_distance = self.get_distance_to_nearest_tree(obs)
        if current_distance < self.previous_distance_to_tree:
            shaped_reward += 0.01
        self.previous_distance_to_tree = current_distance
        
        # 2. 面向树木 → +0.005
        if self.is_facing_tree(obs):
            shaped_reward += 0.005
        
        # 3. 手持工具 → +0.02
        if self.is_holding_axe(obs):
            shaped_reward += 0.02
        
        # 4. 攻击树木 → +0.1
        if self.is_attacking_tree(obs, action):
            shaped_reward += 0.1
        
        # 5. 获得木头 → +1.0（原始任务奖励）
        # 已经包含在base reward中
        
        return obs, shaped_reward, done, info
    
    def get_distance_to_nearest_tree(self, obs):
        # 使用voxel数据或视觉检测找最近的树
        # 简化实现：假设MineDojo提供
        voxel = obs.get('voxels', None)
        if voxel is not None:
            # 计算到最近oak_log方块的距离
            tree_positions = np.where(voxel['block_name'] == 'oak_log')
            if len(tree_positions[0]) > 0:
                distances = np.sqrt(np.sum(tree_positions**2, axis=0))
                return np.min(distances)
        return float('inf')
    
    def is_facing_tree(self, obs):
        # 检查视野中心是否有树木
        # 可以用简单的图像处理或MineCLIP
        pass
    
    def is_holding_axe(self, obs):
        # 检查当前手持物品
        inventory = obs.get('inventory', {})
        return 'axe' in inventory.get('mainhand', {}).get('type', '')
    
    def is_attacking_tree(self, obs, action):
        # 检查是否在攻击动作且面向树木
        return action == ATTACK_ACTION and self.is_facing_tree(obs)
```

**使用方法**：

```python
# 创建包装后的环境
base_env = minedojo.make(task_id="harvest_log")
shaped_env = RewardShapedEnv(base_env)

# 正常训练
model = PPO("CnnPolicy", shaped_env)
model.learn(total_timesteps=500000)
```

### 3.2 使用潜在空间距离作为奖励

利用预训练的视觉编码器（如MineCLIP、VPT特征提取器）计算状态与目标的相似度。

```python
import torch
from mineclip import MineCLIP

class LatentDistanceReward:
    def __init__(self, target_description="a player chopping down a tree"):
        # 加载MineCLIP模型
        self.mineclip = MineCLIP()
        self.target_description = target_description
        
        # 编码目标描述
        self.target_embedding = self.mineclip.encode_text(target_description)
    
    def compute_reward(self, observation):
        # 编码当前观察
        obs_embedding = self.mineclip.encode_image(observation)
        
        # 计算余弦相似度作为奖励
        similarity = torch.cosine_similarity(
            obs_embedding, 
            self.target_embedding
        )
        
        # 转换为奖励（0到1之间）
        reward = (similarity + 1) / 2
        return reward.item()
```

---

## 🧠 方案四：使用预训练模型

### 4.1 MineCLIP作为视觉编码器

不要从随机初始化开始，使用预训练的MineCLIP作为特征提取器。

```python
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
from mineclip import MineCLIP

class MineCLIPFeaturesExtractor(BaseFeaturesExtractor):
    """使用预训练MineCLIP作为特征提取器"""
    
    def __init__(self, observation_space, features_dim=512):
        super().__init__(observation_space, features_dim)
        
        # 加载预训练MineCLIP
        self.mineclip = MineCLIP()
        
        # 冻结MineCLIP参数（可选）
        for param in self.mineclip.parameters():
            param.requires_grad = False
        
        # 添加任务特定的头部
        self.head = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, features_dim),
            nn.ReLU()
        )
    
    def forward(self, observations):
        # 使用MineCLIP编码图像
        features = self.mineclip.encode_image(observations)
        
        # 通过任务特定头部
        return self.head(features)

# 使用自定义特征提取器
policy_kwargs = dict(
    features_extractor_class=MineCLIPFeaturesExtractor,
    features_extractor_kwargs=dict(features_dim=512),
)

model = PPO(
    "CnnPolicy",
    env,
    policy_kwargs=policy_kwargs,
    verbose=1
)

model.learn(total_timesteps=500000)
```

**优点**：
- ✅ 利用大规模预训练知识
- ✅ 更快收敛
- ✅ 更好的泛化能力

### 4.2 使用VPT基础模型

```python
# 加载OpenAI的VPT基础模型
from vpt import load_vpt_model

# 加载foundation模型（在YouTube视频上预训练）
vpt_model = load_vpt_model("foundation-2x")

# 方式1：直接微调VPT模型
# 在MineDojo环境中继续训练VPT模型

# 方式2：使用VPT特征作为初始化
# 将VPT的权重迁移到你的策略网络
model = PPO("CnnPolicy", env)
model.policy.features_extractor.load_state_dict(
    vpt_model.img_encoder.state_dict(), 
    strict=False
)
```

---

## 🤖 方案五：分层强化学习（Hierarchical RL）

### 5.1 技能库设计

将复杂任务分解为可复用的低级技能。

**技能层次结构示例**：

```
高级任务：制作木制工具
├── 技能1：寻找树木（导航）
├── 技能2：砍树（获取木头）
├── 技能3：打开物品栏
└── 技能4：合成物品

低级技能：
- 移动（前后左右）
- 转向（改变视角）
- 跳跃
- 攻击
- 使用物品
```

**实现方式**：

```python
class SkillLibrary:
    """技能库 - 存储和管理已学习的低级技能"""
    
    def __init__(self):
        self.skills = {}
    
    def add_skill(self, name, policy_path, description=""):
        """添加新技能"""
        self.skills[name] = {
            "policy": PPO.load(policy_path),
            "description": description,
        }
    
    def get_skill(self, name):
        """获取技能策略"""
        return self.skills[name]["policy"]
    
    def list_skills(self):
        """列出所有技能"""
        return list(self.skills.keys())


class HierarchicalAgent:
    """分层智能体 - 高级策略选择技能，低级策略执行"""
    
    def __init__(self, skill_library):
        self.skill_library = skill_library
        self.high_level_policy = None  # 选择技能的策略
        self.current_skill = None
        self.skill_steps = 0
        self.max_skill_steps = 100  # 每个技能最多执行100步
    
    def select_skill(self, obs, task_embedding):
        """高级策略：根据观察和任务选择技能"""
        # 使用一个简单的网络选择技能
        skill_probs = self.high_level_policy(obs, task_embedding)
        skill_idx = torch.argmax(skill_probs)
        
        skill_names = self.skill_library.list_skills()
        return skill_names[skill_idx]
    
    def act(self, obs, task_description):
        """执行动作"""
        # 如果当前没有技能或技能已执行足够步数，选择新技能
        if self.current_skill is None or self.skill_steps >= self.max_skill_steps:
            task_emb = encode_task(task_description)
            self.current_skill = self.select_skill(obs, task_emb)
            self.skill_steps = 0
        
        # 使用当前技能的策略执行动作
        policy = self.skill_library.get_skill(self.current_skill)
        action, _ = policy.predict(obs)
        self.skill_steps += 1
        
        return action


# 使用示例
# 1. 训练基础技能
skill_lib = SkillLibrary()

# 训练"导航到树木"技能
nav_env = create_navigation_env(target="oak_log")
nav_model = PPO("CnnPolicy", nav_env)
nav_model.learn(total_timesteps=100000)
nav_model.save("skills/navigate_to_tree.zip")
skill_lib.add_skill("navigate_to_tree", "skills/navigate_to_tree.zip")

# 训练"砍树"技能（假设已经在树旁边）
chop_env = create_chopping_env(spawn_near_tree=True)
chop_model = PPO("CnnPolicy", chop_env)
chop_model.learn(total_timesteps=100000)
chop_model.save("skills/chop_tree.zip")
skill_lib.add_skill("chop_tree", "skills/chop_tree.zip")

# 2. 训练高级策略（组合技能）
agent = HierarchicalAgent(skill_lib)
# 训练高级策略选择正确的技能序列...
```

### 5.2 使用Options框架

```python
from stable_baselines3 import PPO
import numpy as np

class Option:
    """一个可重用的技能/选项"""
    
    def __init__(self, policy, initiation_set, termination_fn):
        self.policy = policy  # 策略网络
        self.initiation_set = initiation_set  # 何时可以启动
        self.termination_fn = termination_fn  # 何时终止
        self.active = False
    
    def can_initiate(self, state):
        """检查是否可以在当前状态启动此选项"""
        return self.initiation_set(state)
    
    def should_terminate(self, state):
        """检查是否应该终止此选项"""
        return self.termination_fn(state)
    
    def get_action(self, state):
        """使用此选项的策略选择动作"""
        return self.policy.predict(state)[0]


# 定义选项
option_navigate = Option(
    policy=PPO.load("skills/navigate.zip"),
    initiation_set=lambda s: True,  # 任何状态都可以导航
    termination_fn=lambda s: is_near_tree(s),  # 靠近树木时终止
)

option_chop = Option(
    policy=PPO.load("skills/chop.zip"),
    initiation_set=lambda s: is_near_tree(s),  # 只有在树旁才能砍树
    termination_fn=lambda s: has_collected_log(s),  # 获得木头时终止
)
```

---

## 🎮 方案六：人机协作（Human-in-the-Loop）

### 6.1 收集人类演示

最直接的方法：自己玩游戏，收集高质量演示数据。

```python
import minedojo
import pickle

def collect_demonstrations(task_id, num_episodes=10):
    """使用键盘控制收集演示数据"""
    
    env = minedojo.make(
        task_id=task_id,
        image_size=(160, 256),
    )
    
    demonstrations = []
    
    for episode in range(num_episodes):
        obs = env.reset()
        episode_data = {"observations": [], "actions": []}
        done = False
        
        print(f"\n=== Episode {episode + 1}/{num_episodes} ===")
        print("Use keyboard to control:")
        print("  W/A/S/D: Move")
        print("  Mouse: Look around")
        print("  Space: Jump")
        print("  Left Click: Attack")
        print("  Q: Quit episode\n")
        
        while not done:
            # 显示当前画面
            env.render()
            
            # 获取键盘输入（需要实现keyboard_to_action函数）
            action = keyboard_to_action()
            
            if action == "quit":
                break
            
            # 执行动作
            next_obs, reward, done, info = env.step(action)
            
            # 记录数据
            episode_data["observations"].append(obs)
            episode_data["actions"].append(action)
            
            obs = next_obs
            
            if reward > 0:
                print(f"✓ Reward: {reward}")
        
        demonstrations.append(episode_data)
        print(f"Episode completed: {len(episode_data['actions'])} steps")
    
    # 保存演示数据
    output_file = f"demonstrations/{task_id}_demos.pkl"
    with open(output_file, 'wb') as f:
        pickle.dump(demonstrations, f)
    
    print(f"\n✓ Saved {num_episodes} demonstrations to {output_file}")
    env.close()
    
    return demonstrations


# 使用
demos = collect_demonstrations("harvest_log", num_episodes=20)
```

### 6.2 交互式学习（Interactive Learning）

智能体在训练过程中遇到困难时请求人类帮助。

```python
class InteractiveLearning:
    """交互式学习 - 智能体可以请求人类帮助"""
    
    def __init__(self, env, model):
        self.env = env
        self.model = model
        self.human_intervention_threshold = 0.1  # 置信度阈值
        self.demonstration_buffer = []
    
    def train_episode(self):
        obs = self.env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            # 获取模型的动作和置信度
            action, _, _, confidence = self.model.predict(
                obs, 
                return_all=True
            )
            
            # 如果模型不确定，请求人类帮助
            if confidence < self.human_intervention_threshold:
                print("🤔 Agent is uncertain. Please demonstrate the action.")
                action = get_human_action()  # 获取人类输入
                
                # 将人类演示加入buffer
                self.demonstration_buffer.append((obs, action))
                
                # 每收集10个人类演示，进行一次行为克隆更新
                if len(self.demonstration_buffer) >= 10:
                    self.update_from_demonstrations()
            
            obs, reward, done, info = self.env.step(action)
            episode_reward += reward
        
        return episode_reward
    
    def update_from_demonstrations(self):
        """使用人类演示更新策略"""
        # 简化的行为克隆更新
        for obs, action in self.demonstration_buffer:
            self.model.policy.learn_from_expert(obs, action)
        
        self.demonstration_buffer.clear()
        print("✓ Updated policy from human demonstrations")
```

---

## 📊 方案七：离线强化学习（Offline RL）

### 7.1 使用现有数据集

MineDojo和MineRL提供了大量人类玩家的游戏数据。

```python
# 使用MineRL数据集
import minerl

# 下载数据集
data = minerl.data.make("MineRLTreechop-v0")

# 加载轨迹
trajectories = []
for state, action, reward, next_state, done in data.batch_iter(
    batch_size=32, 
    num_epochs=1
):
    trajectories.append((state, action, reward, next_state, done))

# 使用离线RL算法训练（如CQL, IQL, BCQ）
from d3rlpy.algos import CQLConfig
from d3rlpy.dataset import MDPDataset

# 创建离线数据集
dataset = MDPDataset(
    observations=...,
    actions=...,
    rewards=...,
    terminals=...,
)

# 训练CQL模型
cql = CQLConfig().create()
cql.fit(dataset, n_steps=100000)
```

### 7.2 CQL（Conservative Q-Learning）

```python
# 使用d3rlpy库实现CQL
from d3rlpy.algos import CQL
from d3rlpy.dataset import MDPDataset

# 假设已经收集了离线数据
offline_dataset = load_offline_data("harvest_log_data.h5")

# 创建CQL模型
cql = CQL(
    learning_rate=3e-4,
    batch_size=256,
    use_gpu=True,
)

# 纯离线训练
cql.fit(
    offline_dataset,
    n_steps=500000,
)

# 保存模型
cql.save_model("checkpoints/cql_harvest_log.pt")
```

---

## 🔄 方案八：多任务学习和迁移学习

### 8.1 多任务训练

同时训练多个相关任务，共享底层表示。

```python
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv

# 创建多个相关任务的环境
def make_multi_task_env():
    """创建多任务环境"""
    
    # 随机选择一个任务
    tasks = [
        "harvest_log",
        "harvest_oak_log",  
        "harvest_birch_log",
        "harvest_spruce_log",
    ]
    
    task = np.random.choice(tasks)
    env = minedojo.make(task_id=task, image_size=(160, 256))
    
    return env


# 创建多个多任务环境
envs = SubprocVecEnv([make_multi_task_env for _ in range(4)])

# 训练多任务策略
model = PPO("CnnPolicy", envs, verbose=1)
model.learn(total_timesteps=1000000)

# 这个模型可以泛化到不同类型的砍树任务
```

### 8.2 从简单任务迁移

先在简单任务上训练，然后迁移到复杂任务。

```python
# 1. 在简单任务上训练（例如：harvest_milk，奖励更容易获得）
simple_env = minedojo.make("harvest_milk")
model = PPO("CnnPolicy", simple_env)
model.learn(total_timesteps=200000)

# 2. 保存视觉编码器的权重
torch.save(
    model.policy.features_extractor.state_dict(),
    "checkpoints/visual_encoder_pretrained.pth"
)

# 3. 在目标任务上训练，使用预训练的视觉编码器
target_env = minedojo.make("harvest_log")
target_model = PPO("CnnPolicy", target_env)

# 加载预训练权重
target_model.policy.features_extractor.load_state_dict(
    torch.load("checkpoints/visual_encoder_pretrained.pth")
)

# 可选：冻结前几层
for param in list(target_model.policy.features_extractor.parameters())[:10]:
    param.requires_grad = False

# 继续训练
target_model.learn(total_timesteps=500000)
```

---

## 🛠️ 实施建议

### 最优组合方案（推荐）

根据你的需求（训练多个技能并组合），我推荐以下组合：

```
方案一（模仿学习） + 方案二（课程学习） + 方案四（预训练模型）
```

**具体实施步骤**：

#### 第1阶段：使用MineCLIP进行奖励塑形（1-2周）

```python
# 创建密集奖励环境
env = minedojo.make(
    task_id="harvest_log",
    image_size=(160, 256),
    reward_fn="mineclip",  # 使用MineCLIP
)

# 使用MineCLIP特征提取器
policy_kwargs = dict(
    features_extractor_class=MineCLIPFeaturesExtractor,
)

model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs)
model.learn(total_timesteps=500000)
```

**预期效果**：
- 训练时间减少50-70%
- 更快看到有意义的行为
- 更好的收敛性

#### 第2阶段：课程学习训练多个技能（2-4周）

为每个技能设计课程：

```python
skills_curriculum = {
    "chop_tree": [
        {"difficulty": "easy", "spawn_near": True, "has_axe": True},
        {"difficulty": "medium", "spawn_near": False, "has_axe": True},
        {"difficulty": "hard", "spawn_near": False, "has_axe": False},
    ],
    
    "mine_stone": [
        {"difficulty": "easy", "spawn_near": True, "has_pickaxe": True},
        {"difficulty": "medium", "spawn_near": False, "has_pickaxe": True},
        {"difficulty": "hard", "spawn_near": False, "has_pickaxe": False},
    ],
    
    "hunt_animal": [
        {"difficulty": "easy", "animal": "cow", "spawn_near": True},
        {"difficulty": "medium", "animal": "sheep", "spawn_near": False},
        {"difficulty": "hard", "animal": "chicken", "spawn_near": False},
    ],
}

# 依次训练每个技能
for skill_name, curriculum in skills_curriculum.items():
    print(f"\n{'='*60}")
    print(f"Training skill: {skill_name}")
    print(f"{'='*60}\n")
    
    previous_model = None
    
    for level in curriculum:
        print(f"  Level: {level['difficulty']}")
        
        # 创建环境
        env = create_skill_env(skill_name, level)
        
        # 加载上一阶段模型或创建新模型
        if previous_model:
            model = PPO.load(previous_model)
            model.set_env(env)
        else:
            model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs)
        
        # 训练
        model.learn(total_timesteps=200000)
        
        # 保存
        model_path = f"skills/{skill_name}_{level['difficulty']}.zip"
        model.save(model_path)
        previous_model = model_path
    
    # 保存最终技能
    final_path = f"skills/{skill_name}_final.zip"
    model.save(final_path)
    print(f"✓ Skill {skill_name} trained and saved to {final_path}")
```

#### 第3阶段：组合技能和高级策略（1-2周）

```python
# 创建技能库
skill_library = SkillLibrary()
skill_library.add_skill("chop_tree", "skills/chop_tree_final.zip")
skill_library.add_skill("mine_stone", "skills/mine_stone_final.zip")
skill_library.add_skill("hunt_animal", "skills/hunt_animal_final.zip")

# 训练高级策略组合技能
hierarchical_agent = HierarchicalAgent(skill_library)

# 在复杂任务上训练（例如："make wooden tools"）
complex_env = minedojo.make("make_wooden_pickaxe")
train_hierarchical_policy(hierarchical_agent, complex_env)
```

### 时间和资源估算

| 方案 | 预期加速 | 实施难度 | 所需资源 |
|------|----------|----------|----------|
| 方案一：模仿学习（MineCLIP） | **3-5x** | ⭐⭐ 中等 | MineDojo自带 |
| 方案二：课程学习 | **2-3x** | ⭐ 简单 | 需要设计课程 |
| 方案三：奖励塑形 | **2-4x** | ⭐⭐⭐ 较难 | 需要任务知识 |
| 方案四：预训练模型 | **3-10x** | ⭐⭐ 中等 | MineCLIP/VPT |
| 方案五：分层RL | **大型项目** | ⭐⭐⭐⭐ 很难 | 复杂系统设计 |
| 方案六：人机协作 | **5-10x** | ⭐ 简单 | 需要人工时间 |
| 方案七：离线RL | **10x+** | ⭐⭐⭐⭐ 很难 | 大量离线数据 |
| 方案八：多任务学习 | **2-3x** | ⭐⭐ 中等 | 多GPU |

### 快速原型（1周内看到效果）

如果你想快速验证，从这里开始：

```python
# quick_start.py - 快速开始脚本

import minedojo
from stable_baselines3 import PPO

# 1. 使用MineCLIP奖励（最重要！）
env = minedojo.make(
    task_id="harvest_log",
    image_size=(160, 256),
    reward_fn="mineclip",  # 密集奖励
)

# 2. 简单的奖励塑形
class SimpleRewardWrapper:
    def __init__(self, env):
        self.env = env
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        
        # 额外奖励：手持工具
        if self.holding_tool(obs):
            reward += 0.01
        
        # 额外奖励：面向方块
        if self.facing_block(obs):
            reward += 0.005
        
        return obs, reward, done, info
    
    # ... 其他方法

env = SimpleRewardWrapper(env)

# 3. 训练
model = PPO("CnnPolicy", env, verbose=1, tensorboard_log="logs/")
model.learn(total_timesteps=200000)  # 先用较少步数测试

# 4. 保存
model.save("checkpoints/quick_harvest.zip")

print("✓ Quick start training completed!")
print("Check TensorBoard: tensorboard --logdir logs/")
```

---

## 📚 参考资料

### 论文
- [MineDojo: Building Open-Ended Embodied Agents with Internet-Scale Knowledge](https://arxiv.org/abs/2206.08853)
- [Video PreTraining (VPT): Learning to Act by Watching Unlabeled Online Videos](https://arxiv.org/abs/2206.11795)
- [MineCLIP: Aligning Language and Video in Minecraft](https://openai.com/research/vpt)

### 代码库
- [MineDojo官方库](https://github.com/MineDojo/MineDojo)
- [OpenAI VPT](https://github.com/openai/Video-Pre-Training)
- [MineRL数据集](https://github.com/minerllabs/minerl)
- [imitation库](https://github.com/HumanCompatibleAI/imitation) - 行为克隆和模仿学习
- [d3rlpy](https://github.com/takuseno/d3rlpy) - 离线强化学习

### 工具
- [MineCLIP模型](https://github.com/MineDojo/MineCLIP)
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)

---

## 总结

训练Minecraft技能的关键是**解决稀疏奖励问题**。最有效的方法是：

1. ✅ **使用MineCLIP提供密集奖励**（最简单、最有效）
2. ✅ **课程学习**（从简单到复杂）
3. ✅ **利用预训练模型**（MineCLIP、VPT）
4. ✅ **收集少量人类演示**（bootstrap初始策略）

**不要**从纯随机策略开始！这是最慢、最低效的方式。

**推荐路径**：
- 第1周：MineCLIP + 简单奖励塑形 → 快速验证
- 第2-4周：课程学习训练多个基础技能
- 第5-6周：分层策略组合技能
- 第7周+：在实际任务中评估和优化

祝你训练顺利！🚀

