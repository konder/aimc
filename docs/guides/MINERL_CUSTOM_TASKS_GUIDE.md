# MineRL 自定义任务指南

**日期**: 2025-11-06  
**状态**: ✅ 可用  
**环境**: MineRL

---

## 概述

本指南介绍如何在 MineRL 环境中创建自定义任务，参考 MineDojo 的任务设计。

### 设计原则

1. **基于 MineRL 现有环境** - 使用 MineRL Obtain 系列作为基础
2. **自定义奖励函数** - 实现稀疏或密集奖励
3. **任务完成检测** - 检查 inventory 中的目标物品
4. **兼容 STEVE-1** - 确保与 STEVE-1 评估框架兼容

---

## 架构设计

### 核心类

#### `MineRLCustomTaskWrapper`

包装器类，在 MineRL 环境基础上添加自定义功能。

```python
class MineRLCustomTaskWrapper(gym.Wrapper):
    """
    参数:
        base_env_name: 基础 MineRL 环境名称
        target_item: 目标物品名称
        target_count: 目标数量
        max_steps: 最大步数
        reward_type: 奖励类型 ('sparse' 或 'dense')
    """
```

**功能**:
- ✅ 自定义奖励计算
- ✅ 任务完成检测
- ✅ 超时检测
- ✅ 详细的 info 信息

---

## 已实现的任务

### 1. Harvest Log (获得木头)

**环境ID**: `MineRLHarvestLog-v0`

```python
import gym

env = gym.make('MineRLHarvestLog-v0')
obs, info = env.reset()

done = False
while not done:
    action = agent.get_action(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    
    if reward > 0:
        print("✓ 获得木头！")

env.close()
```

**参数**:
- 基础环境: `MineRLObtainDiamond-v0`
- 目标物品: `log`
- 目标数量: 1
- 最大步数: 2000
- 奖励类型: `sparse` (完成时 +1.0)

---

### 2. Harvest Iron Ingot (冶炼铁锭)

**环境ID**: `MineRLHarvestIronIngot-v0`

```python
env = gym.make('MineRLHarvestIronIngot-v0')
```

**参数**:
- 基础环境: `MineRLObtainIronPickaxe-v0`
- 目标物品: `iron_ingot`
- 目标数量: 1
- 最大步数: 6000
- 奖励类型: `sparse`

---

### 3. Harvest Diamond (获得钻石)

**环境ID**: `MineRLHarvestDiamond-v0`

```python
env = gym.make('MineRLHarvestDiamond-v0')
```

**参数**:
- 基础环境: `MineRLObtainDiamond-v0`
- 目标物品: `diamond`
- 目标数量: 1
- 最大步数: 18000
- 奖励类型: `sparse`

---

## 创建新任务

### 方法 1: 使用 `MineRLCustomTaskWrapper`

最简单的方式，直接创建包装器实例：

```python
from src.envs import MineRLCustomTaskWrapper
import gym

# 创建自定义任务：获得 5 个苹果
env = MineRLCustomTaskWrapper(
    base_env_name='MineRLObtainDiamond-v0',
    target_item='apple',
    target_count=5,
    max_steps=3000,
    reward_type='sparse',
)

obs, info = env.reset()
# ... 使用环境
```

---

### 方法 2: 继承创建新类

为常用任务创建专门的类：

```python
from src.envs import MineRLCustomTaskWrapper

class HarvestAppleEnv(MineRLCustomTaskWrapper):
    """获得苹果任务"""
    
    def __init__(self, count=1):
        super().__init__(
            base_env_name='MineRLObtainDiamond-v0',
            target_item='apple',
            target_count=count,
            max_steps=2000,
            reward_type='sparse',
        )
```

**注册到 gym**:

```python
import gym

gym.register(
    id='MineRLHarvestApple-v0',
    entry_point='src.envs.minerl_custom_tasks:HarvestAppleEnv',
)
```

---

## 奖励机制

### 稀疏奖励 (Sparse Reward)

只在任务完成时给予奖励：

```python
reward_type='sparse'

# 奖励值:
# - 完成任务: +1.0
# - 未完成:   0.0
```

**适用场景**:
- 简单任务（如获得木头）
- 需要强化学习探索的任务

---

### 密集奖励 (Dense Reward)

根据进度给予部分奖励：

```python
reward_type='dense'

# 奖励值:
# - 每获得 1 个目标物品: +1.0 / target_count
# - 总奖励累积到 1.0
```

**适用场景**:
- 复杂任务（如获得钻石）
- 需要逐步引导的任务

---

## 与 STEVE-1 集成

### 评估示例

```python
import gym
from steve1.utils.mineclip_agent_env_utils import load_mineclip_agent_env
import torch as th

# 1. 加载 STEVE-1
agent, mineclip, _ = load_mineclip_agent_env(
    in_model='data/weights/vpt/2x.model',
    in_weights='data/weights/steve1/steve1.weights',
    seed=42,
)

# 2. 创建自定义环境
env = gym.make('MineRLHarvestLog-v0')

# 3. 编码指令
instruction = "chop tree"
with th.no_grad():
    text_embed = mineclip.encode_text(instruction).cpu().numpy()

# 4. 运行评估
obs, _ = env.reset()
agent.reset()

done = False
while not done:
    with th.no_grad():
        action = agent.get_action(obs, text_embed)
    
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    
    if reward > 0:
        print(f"✓ 任务完成！步数: {info['steps']}")

env.close()
```

---

## 完整评估脚本

使用项目提供的评估脚本：

```bash
# 运行 Harvest Log 评估
bash scripts/run_steve1_harvest_log.sh

# 或直接运行 Python 脚本
python scripts/test_steve1_evaluation.py
```

**评估脚本功能**:
- ✅ 自动加载 STEVE-1 和 MineCLIP
- ✅ 创建自定义环境
- ✅ 运行多次试验
- ✅ 计算成功率和平均步数
- ✅ 生成 JSON 报告
- ✅ 支持中英文指令

---

## 任务配置

### 基础 MineRL 环境选择

| 基础环境 | 适用任务 | 说明 |
|---------|---------|------|
| `MineRLObtainDiamond-v0` | 简单采集 | 森林环境，有树木 |
| `MineRLObtainIronPickaxe-v0` | 冶炼任务 | 有矿石，适合铁矿相关 |
| `MineRLNavigate-v0` | 导航任务 | 开阔地形 |
| `MineRLBasaltFindCave-v0` | 探索任务 | 洞穴环境 |

---

## 目标物品名称

### 常见物品

| MineDojo 名称 | MineRL 名称 | 说明 |
|--------------|------------|------|
| `log` | `log`, `log2` | 原木（各种树木） |
| `planks` | `planks` | 木板 |
| `stick` | `stick` | 木棍 |
| `crafting_table` | `crafting_table` | 工作台 |
| `iron_ingot` | `iron_ingot` | 铁锭 |
| `diamond` | `diamond` | 钻石 |
| `stone` | `stone` | 石头 |
| `cobblestone` | `cobblestone` | 圆石 |

### 检测多种物品

对于有多个变体的物品（如木头），包装器会自动检测：

```python
# 自动检测所有木头类型
log_types = [
    'log', 'log2', 
    'oak_log', 'birch_log', 'spruce_log',
    'jungle_log', 'acacia_log', 'dark_oak_log'
]
```

---

## 任务完成检测

### Inventory 检查

包装器从以下来源检查物品：

1. **observation['inventory']** - 首选
2. **info['inventory']** - 备选

```python
def _get_item_count(self, obs, info):
    # 尝试从 obs 获取
    if 'inventory' in obs:
        inventory = obs['inventory']
        if self.target_item in inventory:
            return int(inventory[self.target_item])
    
    # 尝试从 info 获取
    if 'inventory' in info:
        inventory = info['inventory']
        if self.target_item in inventory:
            return int(inventory[self.target_item])
    
    return 0
```

---

## 参考 MineDojo 任务

本实现参考了 MineDojo 的任务设计：

### MineDojo 任务分类

| 类别 | 数量 | 示例 |
|------|------|------|
| Harvest | 895 | harvest_1_log, harvest_1_iron_ingot |
| Combat | 462 | combat_zombie_forest_* |
| TechTree | 213 | techtree_from_barehand_to_iron_sword |
| Survival | 2 | survival, survival_sword_food |

### 对应关系

| MineDojo 任务 | MineRL 自定义任务 | 说明 |
|--------------|------------------|------|
| `harvest_1_log` | `MineRLHarvestLog-v0` | 获得 1 个木头 |
| `harvest_1_iron_ingot` | `MineRLHarvestIronIngot-v0` | 冶炼 1 个铁锭 |
| `harvest_1_diamond` | `MineRLHarvestDiamond-v0` | 获得 1 个钻石 |

**详细参考**: `docs/reference/MINEDOJO_TASKS_REFERENCE.md`

---

## 限制和注意事项

### ⚠️ 当前限制

1. **基于现有 MineRL 环境**
   - 无法完全自定义世界生成
   - 受限于基础环境的配置

2. **Inventory 检测**
   - 依赖 MineRL 提供的 inventory 信息
   - 部分环境可能不提供完整的 inventory

3. **任务类型**
   - 主要支持 Harvest 类任务
   - Combat 和 TechTree 任务需要额外实现

---

## 后续扩展

### 计划功能

- [ ] Combat 类任务（战斗僵尸等）
- [ ] TechTree 类任务（科技树进度）
- [ ] 自定义世界生成（通过 Malmo XML）
- [ ] 密集奖励函数优化
- [ ] 多目标任务支持

---

## 文件结构

```
src/envs/
├── __init__.py                  # 包初始化和环境注册
├── minerl_custom_tasks.py       # 自定义任务包装器
└── harvest_log.py               # （已废弃）旧实现

scripts/
├── test_steve1_evaluation.py    # 评估脚本
└── run_steve1_harvest_log.sh    # 运行脚本

docs/guides/
└── MINERL_CUSTOM_TASKS_GUIDE.md # 本文档

docs/reference/
└── MINEDOJO_TASKS_REFERENCE.md  # MineDojo 任务参考
```

---

## 参考资料

- **MineRL 官方文档**: https://minerl.readthedocs.io/
- **MineRL GitHub**: https://github.com/minerllabs/minerl
- **MineDojo 文档**: https://docs.minedojo.org
- **STEVE-1 论文**: STEVE-1: A Generative Model for Text-to-Behavior in Minecraft

---

**文档版本**: 1.0  
**最后更新**: 2025-11-06  
**维护者**: AIMC 项目团队

