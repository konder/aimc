# 物品名称映射实现总结

**日期**: 2025-11-20  
**版本**: v1.0  
**状态**: ✅ 完成

---

## 📖 概述

实现了 MineRL 和 MineDojo 环境之间的物品名称自动转换功能，解决了两个环境使用不同物品命名规范的问题。

### 问题背景

- **MineRL** 使用具体变体名称：`oak_planks`, `spruce_planks`, `oak_log`
- **MineDojo** 使用通用名称：`planks`, `log`
- **配置文件** (`eval_tasks.yaml`) 统一使用 MineRL 格式

### 解决方案

创建物品名称映射器 (`item_name_mapper.py`)，在 MineDojo 环境初始化时自动转换物品名称。

---

## 🎯 实现的功能

### 1. 物品名称映射表

**核心映射** (100+ 物品):

| MineRL 名称 | MineDojo 名称 | 类别 |
|------------|--------------|------|
| `oak_planks` | `planks` | 木材 |
| `spruce_planks` | `planks` | 木材 |
| `oak_log` | `log` | 原木 |
| `oak_sapling` | `sapling` | 树苗 |
| `lapis_lazuli` | `dye` | 染料 |
| `sugar_cane` | `reeds` | 甘蔗 |
| `dandelion` | `yellow_flower` | 花朵 |
| `poppy` | `red_flower` | 花朵 |
| `white_wool` | `wool` | 羊毛 |

**完整映射**: 见 `src/envs/item_name_mapper.py`

### 2. 转换函数

#### 2.1 基础转换

```python
# MineRL → MineDojo
minerl_to_minedojo('oak_planks')  # → 'planks'
minerl_to_minedojo('minecraft:oak_log')  # → 'log'

# MineDojo → MineRL
minedojo_to_minerl('planks')  # → 'oak_planks'
minedojo_to_minerl('log')  # → 'oak_log'
```

#### 2.2 配置转换

```python
# initial_inventory 转换
inventory = [
    {'type': 'oak_planks', 'quantity': 2},
    {'type': 'stick', 'quantity': 4}
]
converted = convert_initial_inventory(inventory, 'minedojo')
# → [{'type': 'planks', 'quantity': 2}, {'type': 'stick', 'quantity': 4}]

# reward_config 转换
rewards = [
    {'entity': 'oak_planks', 'amount': 1, 'reward': 100},
    {'entity': 'stick', 'amount': 4, 'reward': 50}
]
converted = convert_reward_config(rewards, 'minedojo')
# → [{'entity': 'planks', 'amount': 1, 'reward': 100}, ...]
```

### 3. MineDojo 环境集成

在 `MineDojoBiomeEnvSpec.create_env()` 中自动转换：

```python
# 1. initial_inventory 转换
if self.initial_inventory:
    converted_inventory = convert_initial_inventory(
        self.initial_inventory, 
        target_env='minedojo'
    )
    # 创建 InventoryItem 对象...

# 2. reward_config 转换
if 'reward_config' in self.kwargs:
    reward_config = self.kwargs.pop('reward_config')
    converted_rewards = convert_reward_config(reward_config, target_env='minedojo')
    
    # 提取并转换为 MineDojo 格式
    self.kwargs['target_names'] = [item['entity'] for item in converted_rewards]
    self.kwargs['target_quantities'] = [item['amount'] for item in converted_rewards]
    self.kwargs['reward_weights'] = {
        item['entity']: item['reward'] 
        for item in converted_rewards
    }
```

---

## 📊 转换示例

### 示例 1: 初始物品栏

**配置文件** (`eval_tasks.yaml`):

```yaml
initial_inventory:
  - type: "oak_planks"
    quantity: 2
  - type: "stick"
    quantity: 4
```

**转换过程**:

```
1. 读取配置 (MineRL 格式)
   ↓
2. convert_initial_inventory(inventory, 'minedojo')
   ↓
3. 物品名称转换
   - oak_planks → planks
   - stick → stick (不变)
   ↓
4. 创建 InventoryItem 对象
   ↓
5. 传递给 minedojo.make()
```

**日志输出**:

```
📦 处理初始物品栏 (2 项)...
  🔄 物品名称转换: oak_planks → planks
✓ 初始物品: 2 项
  - slot 0: planks x2
  - slot 1: stick x4
```

### 示例 2: 奖励配置

**配置文件** (`eval_tasks.yaml`):

```yaml
reward_config:
  - entity: "oak_planks"
    amount: 1
    reward: 100
  - entity: "stick"
    amount: 4
    reward: 50
reward_rule: "any"
```

**转换过程**:

```
1. 读取配置 (MineRL 格式)
   ↓
2. convert_reward_config(reward_config, 'minedojo')
   ↓
3. 物品名称转换
   - oak_planks → planks
   - stick → stick (不变)
   ↓
4. 转换为 MineDojo 格式
   - target_names: ['planks', 'stick']
   - target_quantities: [1, 4]
   - reward_weights: {'planks': 100, 'stick': 50}
   ↓
5. 传递给 minedojo.make()
```

**日志输出**:

```
🎯 处理奖励配置...
  🔄 奖励物品转换: oak_planks → planks
✓ 奖励配置转换完成:
  target_names: ['planks', 'stick']
  target_quantities: [1, 4]
  reward_weights: {'planks': 100, 'stick': 50}
```

---

## 🔍 环境配置支持分析

### MineRL 环境支持的配置

| 配置项 | 支持 | 说明 |
|--------|------|------|
| `initial_inventory` | ✅ | 使用 `type` 字段 |
| `reward_config` | ✅ | 列表格式，包含 `entity`, `amount`, `reward` |
| `reward_rule` | ✅ | "any" 或 "all" |
| `world_generator` | ✅ | 嵌套配置 |
| `time_condition` | ✅ | 嵌套配置 |
| `spawning_condition` | ✅ | 嵌套配置 |

### MineDojo 环境支持的配置

| 配置项 | 支持 | 说明 |
|--------|------|------|
| `initial_inventory` | ✅ | 使用 `type` 或 `name` 字段，**需要转换物品名称** |
| `target_names` | ✅ | 目标物品列表，**需要转换物品名称** |
| `target_quantities` | ✅ | 目标数量列表 |
| `reward_weights` | ✅ | 奖励权重字典 |
| `task_id` | ✅ | "open-ended" 用于自定义任务 |
| `generate_world_type` | ✅ | "default", "flat", "specified_biome" |
| `specified_biome` | ✅ | 指定生物群系 |
| `allow_time_passage` | ✅ | 扁平化配置 |
| `allow_mob_spawn` | ✅ | 扁平化配置 |

### 不兼容的配置

**MineDojo 不支持**:
- `reward_rule` - 自动处理
- `world_generator.force_reset` - 忽略
- `time_condition` (嵌套) - 需要转换为扁平化
- `spawning_condition` (嵌套) - 需要转换为扁平化

**MineRL 不支持**:
- `task_id` - 忽略
- `generate_world_type` - 忽略
- `specified_biome` - 忽略

---

## 📝 使用指南

### 1. 配置文件编写

在 `eval_tasks.yaml` 中统一使用 **MineRL 格式**：

```yaml
env_config:
  # 初始物品 - 使用 MineRL 格式
  initial_inventory:
    - type: "oak_planks"  # 具体变体
      quantity: 2
    - type: "stick"
      quantity: 4
  
  # 奖励配置 - 使用 MineRL 格式
  reward_config:
    - entity: "oak_planks"  # 具体变体
      amount: 1
      reward: 100
  reward_rule: "any"
  
  # MineDojo 专用配置
  task_id: "open-ended"
  generate_world_type: "specified_biome"
  specified_biome: "forest"
```

### 2. 环境创建

```python
# MineDojo 环境会自动转换物品名称
env = gym.make(
    'MineDojoHarvestEnv-v0',
    **env_config  # 包含 MineRL 格式的物品名称
)
# → 内部自动转换为 MineDojo 格式
```

### 3. 添加新的物品映射

在 `src/envs/item_name_mapper.py` 中添加：

```python
MINERL_TO_MINEDOJO_ITEM_MAP = {
    # ... 现有映射 ...
    
    # 新增映射
    "new_minerl_item": "new_minedojo_item",
}

MINEDOJO_TO_MINERL_ITEM_MAP = {
    # ... 现有映射 ...
    
    # 新增反向映射
    "new_minedojo_item": "new_minerl_item",
}
```

---

## ✅ 验证清单

- [x] **物品名称映射表**
  - [x] 100+ 常用物品映射
  - [x] 双向映射（MineRL ↔ MineDojo）
  - [x] 特殊物品处理（planks, log, sapling等）

- [x] **转换函数**
  - [x] `minerl_to_minedojo()` - 基础转换
  - [x] `minedojo_to_minerl()` - 反向转换
  - [x] `convert_initial_inventory()` - 初始物品栏转换
  - [x] `convert_reward_config()` - 奖励配置转换
  - [x] 自动移除 `minecraft:` 前缀

- [x] **MineDojo 环境集成**
  - [x] `initial_inventory` 自动转换
  - [x] `reward_config` 自动转换
  - [x] `target_names` 自动转换
  - [x] 转换日志输出

- [x] **测试验证**
  - [x] 单元测试（内置）
  - [x] 配置转换测试
  - [x] 日志输出验证

---

## 📚 相关文档

- **物品名称映射器**: `src/envs/item_name_mapper.py`
- **环境配置兼容性**: `docs/reference/ENV_CONFIG_COMPATIBILITY.md`
- **MineDojo 环境包装器**: `src/envs/minedojo_harvest.py`
- **MineRL 物品列表**: https://github.com/minerllabs/minerl/blob/cdeae668c2f334e3c9117adf651b5a94436b45f8/minerl/herobraine/hero/mc.py#L535
- **MineDojo 物品列表**: https://github.com/MineDojo/MineDojo/blob/2731bc27394269643b43828d9db8ab3a364601f0/minedojo/sim/mc_meta/mc.py#L4

---

## 🎉 总结

✅ **实现完成！**

- **100+ 物品映射**
- **自动转换功能**
- **完整的日志输出**
- **双向兼容性**

**现在可以在配置文件中使用 MineRL 格式的物品名称，MineDojo 环境会自动转换！** 🚀

---

**最后更新**: 2025-11-20

