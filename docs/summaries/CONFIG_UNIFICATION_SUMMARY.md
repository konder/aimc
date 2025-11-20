# 环境配置统一总结

**日期**: 2025-11-20  
**版本**: v2.0  
**状态**: ✅ 完成

---

## 📖 概述

统一了 MineRL 和 MineDojo 环境的配置格式，简化了配置项，提高了可维护性。

### 改进目标

1. ✅ 统一 `image_size` 和 `resolution` 为一个配置
2. ✅ 统一 `initial_inventory` 使用 `type` 字段
3. ✅ 统一 `reward_config` 使用 `entity` 和 `amount` 字段
4. ✅ 简化世界生成配置，只保留 `specified_biome`
5. ✅ 移除冗余的时间和生成配置

---

## 🎯 配置统一方案

### 1. 图像尺寸统一

**之前**:
```yaml
# MineRL 格式
resolution: (640, 320)  # (width, height)

# MineDojo 格式
image_size: [320, 640]  # [height, width]
```

**现在**:
```yaml
# 统一格式（自动识别并转换）
image_size: [320, 640]  # [height, width]
# 或
image_size: (320, 640)  # (height, width)
```

**转换逻辑**:
- 自动识别 `resolution` 并转换为 `image_size`
- `resolution: (width, height)` → `image_size: (height, width)`
- 日志输出: `🔄 配置转换: resolution=(640, 320) → image_size=(320, 640)`

---

### 2. 初始物品栏统一

**之前**:
```yaml
initial_inventory:
  - name: "oak_planks"  # MineDojo 格式
    quantity: 2
  - type: "stick"       # MineRL 格式
    quantity: 4
```

**现在**:
```yaml
# 统一使用 'type' 字段
initial_inventory:
  - type: "oak_planks"
    quantity: 2
  - type: "stick"
    quantity: 4
```

**转换逻辑**:
- 自动将 `name` 转换为 `type`
- 自动将 `entity` 转换为 `type`
- 日志输出: `🔄 initial_inventory: 'name' → 'type' (oak_planks)`

---

### 3. 奖励配置统一

**之前**:
```yaml
# MineRL 格式
reward_config:
  - entity: "oak_planks"
    amount: 1
    reward: 100

# MineDojo 格式
target_names: ["planks"]
target_quantities: [1]
reward_weights: {"planks": 100}
```

**现在**:
```yaml
# 统一使用 MineRL 格式
reward_config:
  - entity: "oak_planks"
    amount: 1
    reward: 100
```

**转换逻辑**:
- MineDojo 的 `target_names` + `target_quantities` + `reward_weights` → `reward_config`
- 自动将 `type` 或 `name` 转换为 `entity`
- 自动将 `quantity` 转换为 `amount`
- 日志输出: `🔄 配置转换: target_names/target_quantities → reward_config (2 项)`

---

### 4. 世界生成配置简化

**之前**:
```yaml
# MineRL 格式
world_generator:
  force_reset: true

# MineDojo 格式
generate_world_type: "specified_biome"
specified_biome: "forest"
```

**现在**:
```yaml
# 统一格式（仅 MineDojo 支持）
specified_biome: "forest"  # 指定生物群系
# 或
# (不指定，使用默认世界)
```

**转换逻辑**:
- 移除 `world_generator` (MineRL 专用，MineDojo 不支持)
- 移除显式的 `generate_world_type` (自动推断)
- 如果 `specified_biome` 不为空 → `generate_world_type = "specified_biome"`
- 如果 `specified_biome` 为空 → `generate_world_type = "default"`
- 日志输出: `🌍 自动设置: generate_world_type='specified_biome' (biome=forest)`

---

### 5. 时间和生成配置简化

**之前**:
```yaml
# MineRL 嵌套格式
time_condition:
  allow_passage_of_time: false
  start_time: 6000

spawning_condition:
  allow_spawning: true

# MineDojo 扁平格式
allow_time_passage: false
allow_mob_spawn: true
```

**现在**:
```yaml
# 统一扁平格式
start_time: 6000           # 起始时间
allow_mob_spawn: true      # 是否允许生物生成
# (时间默认不流逝，不需要配置)
```

**转换逻辑**:
- 移除 `time_condition` (嵌套格式)
- 移除 `allow_passage_of_time` (MineRL 格式)
- 移除 `allow_time_passage` (MineDojo 格式) - 时间默认不流逝
- 移除 `spawning_condition` (嵌套格式)
- 统一使用 `allow_mob_spawn` (移除 `allow_spawning`)
- 日志输出: `🔄 移除配置: time_condition (使用扁平化配置)`

---

## 📊 配置对照表

### 统一后的标准配置

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `image_size` | `Tuple[int, int]` | `(160, 256)` | 图像尺寸 (height, width) |
| `initial_inventory` | `List[Dict]` | `[]` | 初始物品，使用 `type` 字段 |
| `reward_config` | `List[Dict]` | - | 奖励配置，使用 `entity`, `amount`, `reward` |
| `specified_biome` | `str` (可选) | `None` | 指定生物群系（仅 MineDojo） |
| `world_seed` | `str` (可选) | `"minedojo_biome"` | 世界种子 |
| `start_time` | `int` | `6000` | 起始时间 (0-24000) |
| `allow_mob_spawn` | `bool` | `False` | 是否允许生物生成 |
| `spawn_in_village` | `bool` | `False` | 是否在村庄生成 |
| `break_speed_multiplier` | `float` | `1.0` | 破坏速度倍数 |

### 移除的配置项

| 配置项 | 原因 |
|--------|------|
| `resolution` | 统一为 `image_size` |
| `generate_world_type` | 由 `specified_biome` 自动推断 |
| `world_generator` | MineRL 专用，MineDojo 不支持 |
| `time_condition` | 简化为扁平配置 |
| `allow_passage_of_time` | 时间默认不流逝 |
| `allow_time_passage` | 时间默认不流逝 |
| `spawning_condition` | 简化为扁平配置 |
| `allow_spawning` | 统一为 `allow_mob_spawn` |

---

## 🔄 配置转换示例

### 示例 1: MineRL 格式 → 统一格式

**输入** (MineRL 格式):
```yaml
resolution: (640, 320)
initial_inventory:
  - name: "oak_planks"
    quantity: 2
target_names: ["oak_planks"]
target_quantities: [1]
reward_weights: {"oak_planks": 100}
world_generator:
  force_reset: true
time_condition:
  allow_passage_of_time: false
  start_time: 6000
spawning_condition:
  allow_spawning: true
```

**输出** (统一格式):
```yaml
image_size: (320, 640)
initial_inventory:
  - type: "oak_planks"
    quantity: 2
reward_config:
  - entity: "oak_planks"
    amount: 1
    reward: 100
start_time: 6000
allow_mob_spawn: true
```

**日志输出**:
```
🔄 配置转换: resolution=(640, 320) → image_size=(320, 640)
🔄 配置转换: target_names/target_quantities → reward_config (1 项)
🔄 移除配置: world_generator (MineRL 专用)
🔄 移除配置: time_condition (使用扁平化配置)
🔄 移除配置: spawning_condition (使用扁平化配置)
```

### 示例 2: MineDojo 格式 → 统一格式

**输入** (MineDojo 格式):
```yaml
image_size: [320, 640]
initial_inventory:
  - type: "planks"
    quantity: 2
target_names: ["planks"]
target_quantities: [1]
reward_weights: {"planks": 100}
generate_world_type: "specified_biome"
specified_biome: "forest"
allow_time_passage: false
allow_mob_spawn: false
```

**输出** (统一格式):
```yaml
image_size: [320, 640]
initial_inventory:
  - type: "planks"
    quantity: 2
reward_config:
  - entity: "planks"
    amount: 1
    reward: 100
specified_biome: "forest"
allow_mob_spawn: false
```

**日志输出**:
```
🔄 配置转换: target_names/target_quantities → reward_config (1 项)
🔄 移除配置: allow_time_passage (时间默认不流逝)
🌍 自动设置: generate_world_type='specified_biome' (biome=forest)
```

---

## 📝 使用指南

### 1. 配置文件编写

在 `eval_tasks.yaml` 中使用统一格式：

```yaml
env_config:
  # 图像尺寸
  image_size: [320, 640]  # [height, width]
  
  # 初始物品（使用 'type' 字段）
  initial_inventory:
    - type: "oak_planks"
      quantity: 2
    - type: "stick"
      quantity: 4
  
  # 奖励配置（使用 'entity', 'amount', 'reward'）
  reward_config:
    - entity: "oak_planks"
      amount: 1
      reward: 100
  
  # 世界生成（仅指定 biome，可选）
  specified_biome: "forest"  # 不指定则使用默认世界
  
  # 时间和生成
  start_time: 6000
  allow_mob_spawn: true
```

### 2. 环境创建

```python
import gym

# 配置会自动标准化
env = gym.make(
    'MineDojoHarvestEnv-v0',
    **env_config
)
```

### 3. 兼容旧配置

旧格式的配置会自动转换：

```python
# 旧格式（仍然支持）
old_config = {
    'resolution': (640, 320),  # 自动转换
    'initial_inventory': [
        {'name': 'oak_planks', 'quantity': 2}  # 自动转换
    ],
    'target_names': ['oak_planks'],  # 自动转换
    'target_quantities': [1],
    'reward_weights': {'oak_planks': 100},
    'world_generator': {'force_reset': True},  # 自动移除
    'time_condition': {  # 自动转换
        'allow_passage_of_time': False,
        'start_time': 6000
    }
}

# 创建环境时自动标准化
env = gym.make('MineDojoHarvestEnv-v0', **old_config)
```

---

## 🛠️ 实现细节

### 配置标准化器

**文件**: `src/envs/config_normalizer.py`

**核心函数**:
1. `normalize_image_size()` - 统一图像尺寸
2. `normalize_initial_inventory()` - 统一初始物品栏
3. `normalize_reward_config()` - 统一奖励配置
4. `normalize_world_generation()` - 简化世界生成
5. `normalize_spawn_and_time()` - 简化时间和生成配置
6. `normalize_env_config()` - 主入口函数

### MineDojo 环境集成

**文件**: `src/envs/minedojo_harvest.py`

**修改点**:
1. `__init__` 方法中调用 `normalize_env_config()`
2. 移除 `generate_world_type` 参数（自动推断）
3. 移除 `allow_time_passage` 参数（默认 False）
4. 根据 `specified_biome` 自动设置 `generate_world_type`

---

## ✅ 验证清单

- [x] **图像尺寸统一**
  - [x] `resolution` → `image_size` 转换
  - [x] 自动识别 (width, height) 和 (height, width)
  - [x] 日志输出

- [x] **初始物品栏统一**
  - [x] `name` → `type` 转换
  - [x] `entity` → `type` 转换
  - [x] 保持 `quantity` 字段

- [x] **奖励配置统一**
  - [x] `target_names` + `target_quantities` → `reward_config` 转换
  - [x] `type`/`name` → `entity` 转换
  - [x] `quantity` → `amount` 转换
  - [x] 日志输出

- [x] **世界生成简化**
  - [x] 移除 `world_generator`
  - [x] 移除显式的 `generate_world_type`
  - [x] 根据 `specified_biome` 自动推断
  - [x] 日志输出

- [x] **时间和生成简化**
  - [x] 移除 `time_condition`
  - [x] 移除 `allow_passage_of_time`
  - [x] 移除 `allow_time_passage`
  - [x] 移除 `spawning_condition`
  - [x] `allow_spawning` → `allow_mob_spawn` 统一
  - [x] 日志输出

- [x] **测试验证**
  - [x] MineRL 格式转换测试
  - [x] MineDojo 格式转换测试
  - [x] 混合格式转换测试

---

## 📚 相关文档

- **配置标准化器**: `src/envs/config_normalizer.py`
- **环境配置兼容性**: `docs/reference/ENV_CONFIG_COMPATIBILITY.md`
- **物品名称映射**: `src/envs/item_name_mapper.py`
- **MineDojo 环境**: `src/envs/minedojo_harvest.py`

---

## 🎉 总结

✅ **配置统一完成！**

**简化的配置项**:
- `image_size` (统一图像尺寸)
- `initial_inventory` (使用 `type`)
- `reward_config` (使用 `entity`, `amount`)
- `specified_biome` (简化世界生成)
- `start_time` + `allow_mob_spawn` (简化时间和生成)

**移除的冗余配置**:
- `resolution`, `generate_world_type`, `world_generator`
- `time_condition`, `allow_passage_of_time`, `allow_time_passage`
- `spawning_condition`, `allow_spawning`

**现在配置更简洁、更统一、更易维护！** 🚀

---

**最后更新**: 2025-11-20

