# MineRLHarvestEnv 动态奖励配置使用指南

## 概述

MineRLHarvestEnv 现已支持动态奖励配置，允许为任意物品设置自定义奖励规则，支持单物品和多物品组合奖励。

## 功能特性

### 1. 单物品奖励
为单个物品设置奖励规则。

```yaml
env_config:
  reward_config:
    - entity: "oak_log"      # 物品名称（MineCraft inventory 中的名称）
      amount: 1              # 目标数量
      reward: 100            # 达到目标时的总奖励
  task_complete_on_any: true # 达到目标就完成任务
```

### 2. 多物品组合奖励
为多个物品分别设置奖励规则。

```yaml
env_config:
  reward_config:
    - entity: "oak_log"
      amount: 1
      reward: 20
    - entity: "acacia_log"
      amount: 1
      reward: 80
  task_complete_on_any: true  # 任意物品达标就完成
```

### 3. 所有物品都要达标
要求所有配置的物品都达到目标才完成任务。

```yaml
env_config:
  reward_config:
    - entity: "oak_log"
      amount: 1
      reward: 50
    - entity: "birch_log"
      amount: 1
      reward: 50
  task_complete_on_any: false  # 不是任意一种
  task_complete_on_all: true   # 所有物品都要达标
```

### 4. 大量物品增量奖励
为大量物品设置奖励，系统会自动分配增量奖励。

```yaml
env_config:
  reward_config:
    - entity: "oak_log"
      amount: 5              # 目标5个
      reward: 500            # 总奖励500分
                             # 每获得1个奖励 500/5 = 100分
  task_complete_on_any: true
```

## 配置参数说明

### reward_config
奖励配置列表，每个配置项包含：

- **entity** (string): 物品名称，必须与 MineCraft inventory 中的名称一致
  - 例如：`oak_log`, `birch_log`, `diamond`, `iron_ingot`, `dirt` 等
  
- **amount** (int): 目标数量，达到此数量时认为该物品完成
  
- **reward** (float): 该物品的总奖励值
  - 系统会自动计算增量奖励：每获得1个物品奖励 = reward / amount

### task_complete_on_any
- **类型**: boolean
- **默认值**: true
- **说明**: 是否任意物品达到目标就完成任务
- **适用场景**: 多物品可选任务（如"采集橡木或金合欢木"）

### task_complete_on_all
- **类型**: boolean
- **默认值**: false
- **说明**: 是否所有物品都达到目标才完成任务
- **适用场景**: 多物品必需任务（如"同时采集橡木和桦木"）
- **注意**: 当此项为 true 时，task_complete_on_any 应设为 false

## 在 eval_tasks.yaml 中使用

### 示例1: 单物品任务

```yaml
- task_id: "harvest_oak_log_custom"
  category: "harvest"
  difficulty: "easy"
  description: "采集1个橡木原木（自定义奖励）"
  env_name: "MineRLHarvestEnv-v0"
  
  en_instruction: "chop oak tree"
  zh_instruction: "砍橡树"
  
  env_config:
    reward_config:
      - entity: "oak_log"
        amount: 1
        reward: 100
    task_complete_on_any: true
  
  max_steps: 300
```

### 示例2: 多物品可选任务

```yaml
- task_id: "harvest_any_log"
  category: "harvest"
  difficulty: "easy"
  description: "采集任意类型的木头"
  env_name: "MineRLHarvestEnv-v0"
  
  en_instruction: "collect any type of log"
  zh_instruction: "采集任意木头"
  
  env_config:
    reward_config:
      - entity: "oak_log"
        amount: 1
        reward: 100
      - entity: "birch_log"
        amount: 1
        reward: 100
      - entity: "spruce_log"
        amount: 1
        reward: 100
    task_complete_on_any: true  # 获得任意一种就完成
  
  max_steps: 500
```

### 示例3: 多物品组合任务

```yaml
- task_id: "harvest_mixed_logs"
  category: "harvest"
  difficulty: "hard"
  description: "同时采集橡木和桦木"
  env_name: "MineRLHarvestEnv-v0"
  
  en_instruction: "collect both oak log and birch log"
  zh_instruction: "同时采集橡木和桦木"
  
  env_config:
    reward_config:
      - entity: "oak_log"
        amount: 1
        reward: 50
      - entity: "birch_log"
        amount: 1
        reward: 50
    task_complete_on_any: false
    task_complete_on_all: true  # 需要所有物品都达标
  
  max_steps: 1000
```

### 示例4: 大量物品任务

```yaml
- task_id: "harvest_10_logs"
  category: "harvest"
  difficulty: "hard"
  description: "采集10个原木"
  env_name: "MineRLHarvestEnv-v0"
  
  en_instruction: "collect 10 logs"
  zh_instruction: "采集10个原木"
  
  env_config:
    reward_config:
      - entity: "oak_log"
        amount: 10
        reward: 1000  # 每个木头奖励 1000/10 = 100分
    task_complete_on_any: true
  
  max_steps: 2000
```

## 奖励计算机制

### 增量奖励
系统采用增量奖励机制，每获得一个物品就立即给予奖励：

```
单个物品增量奖励 = 总奖励 / 目标数量
```

**示例**：
- 配置：`{"entity": "oak_log", "amount": 5, "reward": 500}`
- 每获得1个橡木：奖励 500/5 = 100分
- 获得5个橡木后：总奖励 = 500分

### 多物品奖励累加
当配置多个物品时，各物品的奖励独立计算：

```
总奖励 = Σ(各物品的累积奖励)
```

**示例**：
- 配置：
  - 橡木：1个奖励20分
  - 金合欢：1个奖励80分
- 获得1个橡木：奖励 = 20分
- 再获得1个金合欢：奖励 = 80分，总奖励 = 100分

## 常见物品名称参考

### 原木类 (Logs)
- `oak_log` - 橡木原木
- `birch_log` - 桦木原木
- `spruce_log` - 云杉原木
- `jungle_log` - 丛林原木
- `acacia_log` - 金合欢原木
- `dark_oak_log` - 深色橡木原木

### 矿物类 (Ores & Minerals)
- `diamond` - 钻石
- `iron_ingot` - 铁锭
- `gold_ingot` - 金锭
- `coal` - 煤炭
- `emerald` - 绿宝石

### 方块类 (Blocks)
- `dirt` - 泥土
- `stone` - 石头
- `cobblestone` - 圆石
- `sand` - 沙子
- `gravel` - 沙砾

### 食物类 (Food)
- `apple` - 苹果
- `bread` - 面包
- `cooked_beef` - 熟牛肉
- `milk_bucket` - 牛奶桶

### 工具类 (Tools)
- `wooden_pickaxe` - 木镐
- `stone_pickaxe` - 石镐
- `iron_pickaxe` - 铁镐
- `wooden_axe` - 木斧
- `crafting_table` - 工作台

完整列表请参考：`src/envs/minerl_harvest.py` 中的 `ALL_ITEMS`

## 测试验证

### 运行单元测试

```bash
# 基础环境测试
python -m pytest tests/test_minerl_harvest_dynamic_rewards.py -v
```

### 运行演示脚本

```bash
# 通过 run_minedojo_x86.sh 运行
./scripts/run_minedojo_x86.sh python scripts/test_dynamic_reward_config.py
```

### 测试覆盖范围

单元测试涵盖：
1. ✓ 默认配置
2. ✓ 单物品自定义配置
3. ✓ 多物品组合配置
4. ✓ 单物品增量奖励
5. ✓ 多物品任意达标完成
6. ✓ 多物品全部达标完成
7. ✓ 环境重置功能
8. ✓ 超时机制
9. ✓ 配置格式验证

## 代码实现

### 核心类：MineRLHarvestWrapper

位置：`src/envs/minerl_harvest.py`

```python
class MineRLHarvestWrapper(gym.Wrapper):
    def __init__(self, env, reward_config=None, 
                 task_complete_on_any=True, 
                 task_complete_on_all=False):
        """
        Args:
            env: 被包装的环境
            reward_config: 奖励配置列表
            task_complete_on_any: 任意物品达标就完成
            task_complete_on_all: 所有物品达标才完成
        """
```

### 环境注册

```python
# src/envs/__init__.py
from .minerl_harvest import register_minerl_harvest_env

register_minerl_harvest_env()
```

### 创建环境

```python
import gym

# 使用默认配置
env = gym.make('MineRLHarvestEnv-v0')

# 使用自定义配置（需要修改评估框架）
env = gym.make('MineRLHarvestEnv-v0', 
               reward_config=[...],
               task_complete_on_any=True)
```

## 下一步工作

1. **修改评估框架**：更新 `STEVE1Evaluator` 以支持从任务配置传递 `env_config`
2. **实际测试**：在真实环境中运行配置好的任务
3. **性能调优**：根据实际表现调整奖励值
4. **扩展物品**：添加更多物品类型的任务

## 注意事项

1. **物品名称**：必须与 MineCraft inventory 中的名称完全一致（小写+下划线）
2. **奖励值**：建议根据任务难度合理设置，避免过高或过低
3. **完成条件**：`task_complete_on_any` 和 `task_complete_on_all` 不能同时为 true
4. **超时设置**：根据任务难度合理设置 `max_steps`
5. **环境重置**：每次 episode 开始前会自动重置物品计数

## 相关文件

- 核心实现：`src/envs/minerl_harvest.py`
- 单元测试：`tests/test_minerl_harvest_dynamic_rewards.py`
- 演示脚本：`scripts/test_dynamic_reward_config.py`
- 配置文件：`config/eval_tasks.yaml`
- 使用指南：本文档

## 反馈与改进

如有问题或建议，请创建 issue 或联系开发团队。

