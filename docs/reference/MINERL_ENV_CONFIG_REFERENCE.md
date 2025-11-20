# MineRL 环境配置参考

本文档说明 `MineRLHarvestDefaultEnv-v0` 环境支持的所有配置参数。

---

## 📋 支持的配置参数

### 1. 奖励配置 (Reward Configuration)

#### `reward_config`
**类型**: `List[Dict]`  
**必需**: 否  
**默认**: `None`

定义任务的奖励结构。

**格式**:
```yaml
reward_config:
  - entity: "oak_log"      # 物品名称
    amount: 1              # 目标数量
    reward: 100            # 奖励值
  - entity: "cobblestone"
    amount: 10
    reward: 50
```

**说明**:
- `entity`: 物品的内部名称（MineRL 格式，如 `oak_log`, `cobblestone`）
- `amount`: 需要收集的数量
- `reward`: 达到目标时的奖励分数

#### `reward_rule`
**类型**: `str`  
**必需**: 否  
**默认**: `"any"`  
**可选值**: `"any"`, `"all"`, `"none"`

定义任务完成条件。

**说明**:
- `"any"`: 完成任意一个目标即可
- `"all"`: 需要完成所有目标
- `"none"`: 不检查任务完成

---

### 2. 世界生成配置 (World Generation)

#### `specified_biome`
**类型**: `str`  
**必需**: 否  
**默认**: `None` (随机世界)

指定单一 biome 世界。

**格式**:
```yaml
specified_biome: "desert"
```

**支持的 biome**:
- 温暖: `desert`, `savanna`, `badlands`
- 温带: `plains`, `forest`, `flower_forest`, `birch_forest`, `dark_forest`, `swamp`
- 寒冷: `taiga`, `snowy_taiga`, `snowy_tundra`
- 海洋: `ocean`, `deep_ocean`, `frozen_ocean`, `warm_ocean`, `lukewarm_ocean`, `cold_ocean`
- 丛林: `jungle`, `bamboo_jungle`
- 山地: `mountains`, `snowy_mountains`, `wooded_mountains`
- 其他: `beach`, `snowy_beach`, `mushroom_fields`, `river`, `frozen_river`

**参考**: `docs/reference/MINERL_BIOME_REFERENCE.md`

---

### 3. 时间配置 (Time Configuration)

支持两种配置格式：

#### 格式1: 字典格式 (推荐用于代码)
```yaml
time_condition:
  start_time: 6000                # 起始时间（游戏刻）
  allow_passage_of_time: false    # 是否允许时间流逝
```

#### 格式2: 单独字段 (推荐用于 YAML)
```yaml
start_time: 6000                  # 起始时间（游戏刻）
allow_time_passage: false         # 是否允许时间流逝
```

**参数说明**:

##### `start_time`
**类型**: `int`  
**必需**: 否  
**默认**: `6000` (白天正午)  
**范围**: `0-24000`

游戏开始时的时间。

**常用值**:
- `0`: 日出
- `6000`: 正午 ☀️ (推荐)
- `12000`: 日落
- `18000`: 午夜 🌙

##### `allow_passage_of_time` / `allow_time_passage`
**类型**: `bool`  
**必需**: 否  
**默认**: `false`

是否允许时间流逝。

**说明**:
- `false`: 时间固定，不会变化（推荐用于训练）
- `true`: 时间正常流逝，会有昼夜循环

---

### 4. 生物生成配置 (Mob Spawning)

支持两种配置格式：

#### 格式1: 字典格式 (推荐用于代码)
```yaml
spawning_condition:
  allow_spawning: true    # 是否允许生物生成
```

#### 格式2: 单独字段 (推荐用于 YAML)
```yaml
allow_mob_spawn: false    # 是否允许生物生成
```

**参数说明**:

##### `allow_spawning` / `allow_mob_spawn`
**类型**: `bool`  
**必需**: 否  
**默认**: `true`

是否允许敌对生物生成。

**说明**:
- `true`: 允许生物生成（牛、猪、羊、僵尸、骷髅等）
- `false`: 禁用生物生成（更安全，推荐用于采集任务）

**注意**: 被动动物（牛、猪、羊）通常在世界生成时已存在，此选项主要影响后续生成。

---

### 5. 初始物品配置 (Initial Inventory)

#### `initial_inventory`
**类型**: `List[Dict]`  
**必需**: 否  
**默认**: `[]` (空手)

玩家初始拥有的物品。

**格式**:
```yaml
initial_inventory:
  - type: "wooden_axe"      # 物品类型
    quantity: 1             # 数量
  - type: "bucket"
    quantity: 1
  - type: "bread"
    quantity: 16
```

**说明**:
- `type`: 物品的内部名称（MineRL 格式）
- `quantity`: 物品数量

**常用物品**:
- 工具: `wooden_axe`, `stone_axe`, `wooden_pickaxe`, `wooden_shovel`
- 食物: `bread`, `cooked_beef`, `apple`
- 容器: `bucket`, `bowl`
- 材料: `stick`, `planks`, `cobblestone`

---

### 6. 图像配置 (Image Configuration)

#### `image_size`
**类型**: `List[int]` 或 `Tuple[int, int]`  
**必需**: 否  
**默认**: 全局配置或 `(160, 256)` (MineCLIP 标准)

POV 观察图像的尺寸。

**格式**:
```yaml
image_size: [160, 256]    # [height, width]
```

**说明**:
- 格式: `[height, width]`
- MineRL 内部会转换为 `(width, height)`
- 推荐: `[160, 256]` (MineCLIP/STEVE-1 标准)
- 训练: 可以使用更小尺寸如 `[64, 64]` 加速

---

### 7. Episode 配置

#### `max_episode_steps`
**类型**: `int`  
**必需**: 否  
**默认**: `2000`

每个 episode 的最大步数。

**格式**:
```yaml
max_episode_steps: 500
```

**说明**:
- 1 步 ≈ 50ms (20 FPS)
- 500 步 ≈ 25 秒
- 2000 步 ≈ 100 秒

---

## 📝 完整配置示例

### 示例1: 简单采集任务（推荐用于 YAML）

```yaml
tasks:
  - task_id: harvest_wood_forest
    env_name: MineRLHarvestDefaultEnv-v0
    en_instruction: "chop tree and get a log"
    
    env_config:
      # Biome 配置
      specified_biome: forest
      
      # 时间配置（单独字段格式）
      start_time: 6000              # 正午
      allow_time_passage: false     # 固定时间
      
      # 生物生成配置（单独字段格式）
      allow_mob_spawn: false        # 禁用生物
      
      # 奖励配置
      reward_config:
        - entity: "oak_log"
          amount: 1
          reward: 100
      reward_rule: "any"
      
      # Episode 配置
      max_episode_steps: 500
    
    n_trials: 3
```

### 示例2: 复杂任务（使用字典格式）

```yaml
tasks:
  - task_id: harvest_milk_plains
    env_name: MineRLHarvestDefaultEnv-v0
    en_instruction: "find cow and get milk"
    
    env_config:
      # Biome 配置
      specified_biome: plains
      
      # 时间配置（字典格式）
      time_condition:
        start_time: 6000
        allow_passage_of_time: false
      
      # 生物生成配置（字典格式）
      spawning_condition:
        allow_spawning: true
      
      # 初始物品
      initial_inventory:
        - type: "bucket"
          quantity: 1
      
      # 奖励配置
      reward_config:
        - entity: "milk_bucket"
          amount: 1
          reward: 100
      
      # 图像配置
      image_size: [160, 256]
      
      max_episode_steps: 1000
```

### 示例3: 多目标任务

```yaml
tasks:
  - task_id: gather_resources
    env_name: MineRLHarvestDefaultEnv-v0
    en_instruction: "gather wood and cobblestone"
    
    env_config:
      specified_biome: plains
      
      # 时间和生成配置（单独字段）
      start_time: 6000
      allow_time_passage: false
      allow_mob_spawn: false
      
      # 初始工具
      initial_inventory:
        - type: "wooden_axe"
          quantity: 1
        - type: "wooden_pickaxe"
          quantity: 1
      
      # 多目标奖励
      reward_config:
        - entity: "oak_log"
          amount: 5
          reward: 50
        - entity: "cobblestone"
          amount: 10
          reward: 50
      reward_rule: "all"    # 需要完成所有目标
      
      max_episode_steps: 2000
```

---

## 🔄 配置格式转换

代码会自动转换两种格式：

### YAML 格式（单独字段）→ Python 格式（字典）

**YAML**:
```yaml
start_time: 6000
allow_time_passage: false
allow_mob_spawn: false
```

**自动转换为**:
```python
time_condition = {
    "start_time": 6000,
    "allow_passage_of_time": False
}
spawning_condition = {
    "allow_spawning": False
}
```

### 日志输出

```
==============================
创建 MineRL Harvest 环境
==============================
  reward_config: 1 项
  reward_rule: any
  initial_inventory: None
  specified_biome: forest
  🔄 构建 time_condition: {'start_time': 6000, 'allow_passage_of_time': False}
  time_condition: {'start_time': 6000, 'allow_passage_of_time': False}
  🔄 构建 spawning_condition: {'allow_spawning': False}
  spawning_condition: {'allow_spawning': False}
  image_size: (160, 256)
  max_episode_steps: 500
```

---

## ⚙️ 代码中使用

### Python 代码示例

```python
import gym
from src.envs.minerl_harvest_default import register_minerl_harvest_default_env

# 注册环境
register_minerl_harvest_default_env()

# 方式1: 使用单独字段（推荐）
env = gym.make(
    'MineRLHarvestDefaultEnv-v0',
    specified_biome="forest",
    start_time=6000,
    allow_time_passage=False,
    allow_mob_spawn=False,
    initial_inventory=[{"type": "wooden_axe", "quantity": 1}],
    reward_config=[{"entity": "oak_log", "amount": 1, "reward": 100}],
    max_episode_steps=500
)

# 方式2: 使用字典格式
env = gym.make(
    'MineRLHarvestDefaultEnv-v0',
    specified_biome="forest",
    time_condition={"start_time": 6000, "allow_passage_of_time": False},
    spawning_condition={"allow_spawning": False},
    initial_inventory=[{"type": "wooden_axe", "quantity": 1}],
    reward_config=[{"entity": "oak_log", "amount": 1, "reward": 100}],
    max_episode_steps=500
)

obs = env.reset()
# ... 使用环境
env.close()
```

---

## 🐛 故障排查

### 问题1: 配置没有生效

**症状**: 时间或生物配置没有按预期工作

**检查**:
1. 查看日志中是否有 "🔄 构建 time_condition" 或 "🔄 构建 spawning_condition"
2. 确认 YAML 缩进正确
3. 确认字段名拼写正确

**常见错误**:
```yaml
# ❌ 错误: 缩进不对
env_config:
start_time: 6000

# ✅ 正确: 缩进对齐
env_config:
  start_time: 6000
```

### 问题2: 字段名混淆

**MineRL vs MineDojo**:

| 配置 | MineRL (字典) | MineRL (单独) | MineDojo |
|------|--------------|---------------|----------|
| 时间 | `time_condition` | `start_time`, `allow_time_passage` | `start_time`, `allow_time_passage` |
| 生成 | `spawning_condition` | `allow_mob_spawn` | `allow_mob_spawn` |
| 物品 | `initial_inventory` | `initial_inventory` | `initial_inventory` |
| Biome | `specified_biome` | `specified_biome` | `specified_biome` |

---

## 📚 相关文档

- **Biome 参考**: `docs/reference/MINERL_BIOME_REFERENCE.md`
- **任务配置**: `config/eval_tasks.yaml`
- **环境实现**: `src/envs/minerl_harvest_default.py`
- **工具函数**: `src/utils/steve1_mineclip_agent_env_utils.py`

---

**版本**: v1.0.0  
**日期**: 2025-11-20  
**状态**: ✅ 已验证

