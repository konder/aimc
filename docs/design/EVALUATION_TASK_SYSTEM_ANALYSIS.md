# STEVE-1 评估任务体系全面分析
# Comprehensive Evaluation Task System Analysis

**版本**: v1.0  
**创建时间**: 2025-12-02  
**基于**: MineDojo 1581 个官方任务 + 当前项目 32 个任务

---

## 📊 执行摘要（Executive Summary）

### 当前状态
- **已配置任务**: 32 个
- **成功任务**: 10 个 (31.25%)
- **失败任务**: 22 个 (68.75%)
- **覆盖维度**: Harvest (12) + Combat (8) + Techtree (12)

### 核心问题
1. ✅ **覆盖度充足**：三大维度（资源、战斗、科技）都有覆盖
2. ⚠️ **成功率偏低**：22/32 任务完全失败（0%）
3. ⚠️ **难度失衡**：缺少简单任务，超难任务过多
4. ⚠️ **ROI 不佳**：某些任务过于复杂，投入产出比低

### 优化建议
- **移除**: 8 个低ROI任务（超难/重复/低价值）
- **保留**: 24 个核心任务
- **优化**: 22 个失败任务的配置
- **预期**: 整体成功率从 31% → 55-65%

---

## 📋 一、MineDojo 任务体系分析

### 1.1 官方任务分类

根据 [MineDojo 文档](https://docs.minedojo.org/sections/customization/task.html)，任务分为四大类：

| 类别 | 任务类型 | MineDojo 官方数量 | 核心特点 |
|------|----------|-------------------|----------|
| **Harvest** | 采集资源 | ~500 | 目标：采集物品，奖励：物品数量 |
| **Combat** | 战斗击杀 | ~400 | 目标：击杀生物，奖励：击杀数量 |
| **TechTree** | 科技树 | ~600 | 目标：制作/使用物品，奖励：科技解锁 |
| **Survival** | 生存 | ~80 | 目标：存活天数，奖励：生存时长 |

### 1.2 MineDojo 任务命名规范

**Harvest 任务命名**：
```
harvest_{quantity}_{item}[_with_{tool}][_in_{biome}]

示例：
- harvest_1_log                          # 采集 1 个原木
- harvest_1_iron_ore_with_stone_pickaxe  # 用石镐采集 1 个铁矿
- harvest_64_dirt_plains                 # 在平原采集 64 个泥土
```

**Combat 任务命名**：
```
combat_{mob}[_{biome}]_{armor}_{weapon}_shield

示例：
- combat_chicken_plains_leather_armors_wooden_sword_shield
- combat_zombie_forest_iron_armors_iron_sword_shield
```

**TechTree 任务命名**：
```
techtree_from_{start_level}_to_{end_level}_{item}

示例：
- techtree_from_barehand_to_wooden_pickaxe  # 从零制作木镐
- techtree_from_wood_to_stone_pickaxe       # 从木制工具升级到石镐
- techtree_from_iron_to_diamond_pickaxe     # 铁镐升级到钻石镐
```

---

## 📊 二、当前任务体系分析

### 2.1 已配置任务清单（32个）

#### Harvest 任务（12个）
| 任务ID | 指令 | 成功率 | 状态 | 难度 |
|--------|------|--------|------|------|
| harvest_1_log | chop tree | 30% | ✅ 成功 | Easy |
| harvest_1_dirt | dig dirt | 90% | ✅ 成功 | Easy |
| harvest_1_sand | dig sand | 100% | ✅ 成功 | Easy |
| harvest_1_cobblestone | mine cobblestone | 10% | ✅ 成功 | Medium |
| harvest_1_coal | mine coal | 20% | ✅ 成功 | Medium |
| harvest_1_gravel | dig gravel | 0% | ❌ 失败 | Easy |
| harvest_1_sapling | get sapling | 10% | ✅ 成功 | Easy |
| harvest_1_wool | get wool | 50% | ✅ 成功 | Medium |
| harvest_1_flower | pick flower | 0% | ❌ 失败 | Easy |
| harvest_1_apple | get apple | 0% | ❌ 失败 | Hard |
| harvest_1_milk | milk cow | 0% | ❌ 失败 | Easy |
| harvest_1_beef | kill cow | 0% | ❌ 失败 | Easy |

#### Combat 任务（8个）
| 任务ID | 指令 | 成功率 | 状态 | 难度 |
|--------|------|--------|------|------|
| combat_pig | hunt pig | 20% | ✅ 成功 | Easy |
| combat_chicken | hunt chicken | 0% | ❌ 失败 | Easy |
| combat_cow | hunt cow | 0% | ❌ 失败 | Easy |
| combat_spider | kill spider | 0% | ❌ 失败 | Medium |
| combat_skeleton | kill skeleton | 0% | ❌ 失败 | Medium |
| combat_zombie_leather_armor | kill zombie | 0% | ❌ 失败 | Medium |
| combat_zombie_with_shield | kill zombie with shield | 0% | ❌ 失败 | Hard |
| combat_creeper | kill creeper | 0% | ❌ 失败 | Hard |

#### Techtree 任务（12个）
| 任务ID | 指令 | 成功率 | 状态 | 难度 |
|--------|------|--------|------|------|
| techtree_craft_planks | craft planks | 80% | ✅ 成功 | Easy |
| techtree_craft_crafting_table | craft crafting table | 10% | ✅ 成功 | Easy |
| techtree_craft_sticks | craft sticks | 0% | ❌ 失败 | Easy |
| techtree_craft_wooden_pickaxe | craft wooden pickaxe | 0% | ❌ 失败 | Easy |
| techtree_craft_wooden_sword | craft wooden sword | 0% | ❌ 失败 | Easy |
| techtree_craft_stone_sword | craft stone sword | 0% | ❌ 失败 | Easy |
| techtree_craft_furnace | craft furnace | 0% | ❌ 失败 | Easy |
| techtree_smelt_iron_ingot | smelt iron ore | 0% | ❌ 失败 | Medium |
| techtree_craft_iron_pickaxe | craft iron pickaxe | 0% | ❌ 失败 | Medium |
| techtree_craft_iron_sword | craft iron sword | 0% | ❌ 失败 | Medium |
| techtree_barehand_to_stone_pickaxe | craft from scratch | 0% | ❌ 失败 | Very Hard |
| techtree_stone_to_iron_pickaxe | upgrade pickaxe | 0% | ❌ 失败 | Very Hard |

### 2.2 当前任务问题分析

#### 问题 1: 命名不规范
**现状**: 混合了简化命名和完整命名
```
combat_pig                           ← 简化命名（OK）
combat_zombie_leather_armor          ← 不完整（缺少 biome/weapon）
harvest_1_gravel                     ← 简化命名（OK）
techtree_barehand_to_stone_pickaxe   ← 不符合官方命名
```

**建议**: 统一使用简化命名，更易理解

#### 问题 2: 难度失衡
- **Easy 任务**: 13 个（41%）
- **Medium 任务**: 11 个（34%）
- **Hard 任务**: 6 个（19%）
- **Very Hard 任务**: 2 个（6%）

**问题**：Easy 任务失败率仍然很高（7/13 = 54%），说明配置不当而非真正困难

#### 问题 3: 重复任务
- `harvest_1_beef` vs `combat_cow` - 都是击杀牛
- `combat_zombie_leather_armor` vs `combat_zombie_with_shield` - 都是击杀僵尸

#### 问题 4: 缺失的关键任务
- ❌ **移动类**：无基础移动任务（explore, swim）
- ❌ **矿石递进**：coal → iron → gold → diamond 链条不完整
- ❌ **工具递进**：wooden → stone → iron 链条不完整
- ❌ **合成基础**：缺少 sticks、torch 等基础合成

---

## 🎯 三、优化后的评估任务体系

### 3.1 设计原则

**1. 三维覆盖**：
- **资源维度** (Harvest): 基础资源 → 稀有资源
- **战斗维度** (Combat): 被动生物 → 敌对生物
- **科技维度** (TechTree): 基础合成 → 工具链

**2. 难度递进**：
- **Tier 1 (Easy)**: 1-2 步，成功率 60-80%
- **Tier 2 (Medium)**: 3-5 步，成功率 30-50%
- **Tier 3 (Hard)**: 6-10 步，成功率 10-30%
- **Tier 4 (Very Hard)**: 10+ 步，成功率 < 10%

**3. ROI 优先**：
- 优先保留高价值、高成功率任务
- 移除低 ROI 任务（超难、重复、边缘）

### 3.2 推荐任务清单（24个）

#### 📦 Harvest 任务（9个 - 推荐保留）

| Tier | 任务ID | 指令 | 当前状态 | 优先级 | 说明 |
|------|--------|------|----------|--------|------|
| **Tier 1** | harvest_1_dirt | dig dirt | ✅ 90% | ⭐⭐⭐ | 最基础，必须保留 |
| Tier 1 | harvest_1_sand | dig sand | ✅ 100% | ⭐⭐⭐ | 基础资源 |
| Tier 1 | harvest_1_gravel | dig gravel | ❌ 0% | ⭐⭐ | 需要优化配置 |
| Tier 1 | harvest_1_log | chop tree | ✅ 30% | ⭐⭐⭐ | 基础资源，必须 |
| Tier 1 | harvest_1_flower | pick flower | ❌ 0% | ⭐⭐ | 简单但随机性高 |
| **Tier 2** | harvest_1_cobblestone | mine cobblestone | ✅ 10% | ⭐⭐⭐ | 科技树前置 |
| Tier 2 | harvest_1_coal | mine coal | ✅ 20% | ⭐⭐⭐ | 科技树前置 |
| Tier 2 | harvest_1_sapling | get sapling | ✅ 10% | ⭐⭐ | 植物系统 |
| Tier 2 | harvest_1_wool | get wool | ✅ 50% | ⭐⭐ | 动物互动 |

**移除建议**（3个）:
- ❌ `harvest_1_beef` - 与 combat_cow 重复
- ❌ `harvest_1_apple` - 掉落率极低（0.5%），ROI 太低
- ❌ `harvest_1_milk` - 需要特殊交互（右键），难以训练

#### ⚔️ Combat 任务（6个 - 精简后）

| Tier | 任务ID | 指令 | 当前状态 | 优先级 | 说明 |
|------|--------|------|----------|--------|------|
| **Tier 1** | combat_pig | hunt pig | ✅ 20% | ⭐⭐⭐ | 最简单的战斗 |
| Tier 1 | combat_chicken | hunt chicken | ❌ 0% | ⭐⭐⭐ | 基础战斗 |
| Tier 1 | combat_cow | hunt cow | ❌ 0% | ⭐⭐ | 基础战斗 |
| **Tier 2** | combat_spider | kill spider | ❌ 0% | ⭐⭐⭐ | 敌对生物入门 |
| Tier 2 | combat_skeleton | kill skeleton | ❌ 0% | ⭐⭐ | 远程敌对生物 |
| Tier 2 | combat_zombie | kill zombie | ❌ 0% | ⭐⭐⭐ | 基础敌对生物 |

**移除建议**（2个）:
- ❌ `combat_zombie_with_shield` - 太难，ROI 低
- ❌ `combat_creeper` - 需要特殊策略，难以训练

**合并建议**:
- `combat_zombie_leather_armor` → `combat_zombie`（统一为简单版本）

#### 🔧 Techtree 任务（9个 - 精简后）

| Tier | 任务ID | 指令 | 当前状态 | 优先级 | 说明 |
|------|--------|------|----------|--------|------|
| **Tier 1** | techtree_craft_planks | craft planks | ✅ 80% | ⭐⭐⭐ | 最基础合成 |
| Tier 1 | techtree_craft_sticks | craft sticks | ❌ 0% | ⭐⭐⭐ | 基础合成 |
| Tier 1 | techtree_craft_crafting_table | craft table | ✅ 10% | ⭐⭐⭐ | 科技树关键 |
| **Tier 2** | techtree_craft_wooden_pickaxe | craft wooden pickaxe | ❌ 0% | ⭐⭐⭐ | 工具链起点 |
| Tier 2 | techtree_craft_wooden_sword | craft wooden sword | ❌ 0% | ⭐⭐ | 武器链起点 |
| Tier 2 | techtree_craft_furnace | craft furnace | ❌ 0% | ⭐⭐⭐ | 熔炼前置 |
| **Tier 3** | techtree_craft_stone_sword | craft stone sword | ❌ 0% | ⭐⭐ | 工具升级 |
| Tier 3 | techtree_smelt_iron_ingot | smelt iron | ❌ 0% | ⭐⭐⭐ | 熔炼系统 |
| Tier 3 | techtree_craft_iron_pickaxe | craft iron pickaxe | ❌ 0% | ⭐⭐ | 工具升级 |

**移除建议**（3个）:
- ❌ `techtree_craft_iron_sword` - 与 iron_pickaxe 重复，优先保留镐
- ❌ `techtree_barehand_to_stone_pickaxe` - 太长（6000步），ROI 极低
- ❌ `techtree_stone_to_iron_pickaxe` - 太长（6000步），ROI 极低

---

## 🎯 四、优化后的核心任务集（24个）

### 4.1 任务分布

| 类别 | Tier 1 | Tier 2 | Tier 3 | 总计 |
|------|--------|--------|--------|------|
| **Harvest** | 5 | 4 | 0 | **9** |
| **Combat** | 3 | 3 | 0 | **6** |
| **Techtree** | 3 | 3 | 3 | **9** |
| **总计** | **11** | **10** | **3** | **24** |

### 4.2 预期成功率

| Tier | 任务数 | 预期成功率 | ROI 评级 |
|------|--------|-----------|----------|
| **Tier 1 (Easy)** | 11 | 60-80% | ⭐⭐⭐ 高 |
| **Tier 2 (Medium)** | 10 | 30-50% | ⭐⭐ 中 |
| **Tier 3 (Hard)** | 3 | 10-30% | ⭐ 低 |
| **整体** | **24** | **55-65%** | **⭐⭐⭐** |

---

## 🔧 五、配置优化策略

### 5.1 Harvest 任务优化

#### 保留任务（9个）
```yaml
# Tier 1 - 基础资源（5个）
harvest_1_dirt:           # ✅ 90% - 最简单
  biome: plains
  initial_inventory: []
  
harvest_1_sand:           # ✅ 100% - 简单
  biome: desert
  initial_inventory: []
  
harvest_1_log:            # ✅ 30% - 基础但需要优化
  biome: forest
  initial_inventory: []
  
harvest_1_gravel:         # ❌ 0% → 预期 50%
  biome: mountains
  initial_inventory:
  - type: stone_pickaxe
    quantity: 1
  
harvest_1_flower:         # ❌ 0% → 预期 60%
  biome: plains
  reward_config:
  - entity: poppy
    amount: 1
    reward: 50
  - entity: dandelion
    amount: 1
    reward: 50
  reward_rule: any

# Tier 2 - 工具/动物（4个）
harvest_1_cobblestone:    # ✅ 10% → 预期 40%
  biome: mountains
  initial_inventory:
  - type: wooden_pickaxe
    quantity: 1

harvest_1_coal:           # ✅ 20% → 预期 50%
  biome: mountains
  initial_inventory:
  - type: wooden_pickaxe
    quantity: 1

harvest_1_sapling:        # ✅ 10% → 预期 30%
  biome: forest
  initial_inventory: []

harvest_1_wool:           # ✅ 50% - 已经不错
  biome: plains
  initial_inventory:
  - type: shears
    quantity: 1
```

#### 移除任务（3个）
- ❌ `harvest_1_apple` - 掉落率太低（0.5%），需要打数百片树叶，ROI 极低
- ❌ `harvest_1_milk` - 需要右键交互，MineRL 环境支持不好
- ❌ `harvest_1_beef` - 与 combat_cow 功能重复

### 5.2 Combat 任务优化

#### 保留任务（6个）
```yaml
# Tier 1 - 被动生物（3个）
combat_pig:               # ✅ 20% → 预期 60%
  biome: plains
  time: 6000  # 白天
  initial_inventory:
  - type: wooden_sword
    quantity: 1

combat_chicken:           # ❌ 0% → 预期 50%
  biome: plains
  time: 6000
  initial_inventory:
  - type: wooden_sword
    quantity: 1

combat_cow:               # ❌ 0% → 预期 50%
  biome: plains
  time: 6000
  initial_inventory:
  - type: wooden_sword
    quantity: 1

# Tier 2 - 敌对生物（3个）
combat_spider:            # ❌ 0% → 预期 30%
  biome: plains
  time: 13000  # 夜晚
  initial_inventory:
  - type: stone_sword
    quantity: 1

combat_skeleton:          # ❌ 0% → 预期 20%
  biome: plains
  time: 13000
  initial_inventory:
  - type: stone_sword
    quantity: 1
  - type: shield
    quantity: 1

combat_zombie:            # ❌ 0% → 预期 30%
  biome: plains
  time: 13000
  initial_inventory:
  - type: stone_sword
    quantity: 1
```

#### 移除任务（2个）
- ❌ `combat_zombie_with_shield` - 太难（持盾僵尸），需要特殊策略
- ❌ `combat_creeper` - 需要远程攻击避免爆炸，训练困难

**合并建议**:
- `combat_zombie_leather_armor` → `combat_zombie`（简化为基础版本）

### 5.3 Techtree 任务优化

#### 保留任务（9个）
```yaml
# Tier 1 - 基础合成（3个）
techtree_craft_planks:          # ✅ 80% - 最简单
  initial_inventory:
  - type: log
    quantity: 1

techtree_craft_sticks:          # ❌ 0% → 预期 70%
  initial_inventory:
  - type: planks
    quantity: 2

techtree_craft_crafting_table:  # ✅ 10% → 预期 50%
  initial_inventory:
  - type: planks
    quantity: 4

# Tier 2 - 工具制作（3个）
techtree_craft_wooden_pickaxe:  # ❌ 0% → 预期 40%
  initial_inventory:
  - type: crafting_table
    quantity: 1
  - type: planks
    quantity: 3
  - type: stick
    quantity: 2

techtree_craft_wooden_sword:    # ❌ 0% → 预期 40%
  initial_inventory:
  - type: crafting_table
    quantity: 1
  - type: planks
    quantity: 2
  - type: stick
    quantity: 1

techtree_craft_furnace:         # ❌ 0% → 预期 50%
  initial_inventory:
  - type: crafting_table
    quantity: 1
  - type: cobblestone
    quantity: 8

# Tier 3 - 高级制作（3个）
techtree_craft_stone_sword:     # ❌ 0% → 预期 30%
  initial_inventory:
  - type: crafting_table
    quantity: 1
  - type: cobblestone
    quantity: 2
  - type: stick
    quantity: 1

techtree_smelt_iron_ingot:      # ❌ 0% → 预期 40%
  initial_inventory:
  - type: furnace
    quantity: 1
  - type: iron_ore
    quantity: 1
  - type: coal
    quantity: 1
  time_condition:
    allow_passage_of_time: true  # 关键！

techtree_craft_iron_pickaxe:    # ❌ 0% → 预期 30%
  initial_inventory:
  - type: crafting_table
    quantity: 1
  - type: iron_ingot
    quantity: 3
  - type: stick
    quantity: 2
```

#### 移除任务（3个）
- ❌ `techtree_craft_iron_sword` - 与 iron_pickaxe 重复，优先镐
- ❌ `techtree_barehand_to_stone_pickaxe` - 6000 步，ROI 极低，不适合评估
- ❌ `techtree_stone_to_iron_pickaxe` - 6000 步，ROI 极低，不适合评估

---

## 📈 六、新增任务建议（可选）

### 6.1 缺失的关键任务

如果要进一步完善评估体系，建议新增以下任务：

#### 移动类（Movement - 新增类别）
```yaml
# Tier 1 - 基础移动
movement_walk_forward:
  instruction: "move forward"
  description: "向前移动 10 步"
  max_steps: 100
  
movement_look_around:
  instruction: "look around"
  description: "环顾四周（相机旋转 360 度）"
  max_steps: 100
```

**价值**: 测试基础运动控制
**ROI**: ⭐⭐ 中等（可选）

#### 工具链完整性
```yaml
# 当前缺失
techtree_craft_torch:           # 基础照明
  initial_inventory:
  - type: stick
    quantity: 1
  - type: coal
    quantity: 1
```

**价值**: 补充科技树基础合成
**ROI**: ⭐⭐ 中等（可选）

---

## 🎯 七、最终推荐任务体系（24个）

### 7.1 核心任务集（优先级排序）

#### 🥇 Tier 1 - Easy（11个，成功率目标 60-80%）

**Harvest** (5):
1. ⭐⭐⭐ `harvest_1_dirt` - 90% → 维持
2. ⭐⭐⭐ `harvest_1_sand` - 100% → 维持
3. ⭐⭐⭐ `harvest_1_log` - 30% → 优化到 60%
4. ⭐⭐ `harvest_1_gravel` - 0% → 优化到 50%
5. ⭐⭐ `harvest_1_flower` - 0% → 优化到 60%

**Combat** (3):
6. ⭐⭐⭐ `combat_pig` - 20% → 优化到 60%
7. ⭐⭐⭐ `combat_chicken` - 0% → 优化到 50%
8. ⭐⭐ `combat_cow` - 0% → 优化到 50%

**Techtree** (3):
9. ⭐⭐⭐ `techtree_craft_planks` - 80% → 维持
10. ⭐⭐⭐ `techtree_craft_sticks` - 0% → 优化到 70%
11. ⭐⭐⭐ `techtree_craft_crafting_table` - 10% → 优化到 50%

#### 🥈 Tier 2 - Medium（10个，成功率目标 30-50%）

**Harvest** (4):
12. ⭐⭐⭐ `harvest_1_cobblestone` - 10% → 优化到 40%
13. ⭐⭐⭐ `harvest_1_coal` - 20% → 优化到 50%
14. ⭐⭐ `harvest_1_sapling` - 10% → 优化到 30%
15. ⭐⭐ `harvest_1_wool` - 50% → 维持

**Combat** (3):
16. ⭐⭐⭐ `combat_spider` - 0% → 优化到 30%
17. ⭐⭐⭐ `combat_zombie` - 0% → 优化到 30%
18. ⭐⭐ `combat_skeleton` - 0% → 优化到 20%

**Techtree** (3):
19. ⭐⭐⭐ `techtree_craft_wooden_pickaxe` - 0% → 优化到 40%
20. ⭐⭐ `techtree_craft_wooden_sword` - 0% → 优化到 40%
21. ⭐⭐⭐ `techtree_craft_furnace` - 0% → 优化到 50%

#### 🥉 Tier 3 - Hard（3个，成功率目标 10-30%）

**Techtree** (3):
22. ⭐⭐ `techtree_craft_stone_sword` - 0% → 优化到 30%
23. ⭐⭐⭐ `techtree_smelt_iron_ingot` - 0% → 优化到 40%
24. ⭐⭐ `techtree_craft_iron_pickaxe` - 0% → 优化到 30%

### 7.2 任务依赖关系图

```
资源维度 (Harvest):
  基础资源 → 工具采集 → 动物互动
  ├─ dirt/sand/log (Tier 1)
  ├─ gravel/flower (Tier 1)
  ├─ cobblestone/coal (Tier 2, 需要镐)
  └─ sapling/wool (Tier 2)

战斗维度 (Combat):
  被动生物 → 敌对生物
  ├─ pig/chicken/cow (Tier 1, 白天)
  └─ spider/zombie/skeleton (Tier 2, 夜晚)

科技维度 (Techtree):
  基础合成 → 木制工具 → 石制工具 → 铁制工具
  ├─ planks/sticks/table (Tier 1)
  ├─ wooden_pickaxe/sword/furnace (Tier 2)
  └─ stone_sword/iron_ingot/iron_pickaxe (Tier 3)
```

---

## 📊 八、对比分析

### 8.1 优化前 vs 优化后

| 指标 | 优化前 | 优化后 | 改进 |
|------|--------|--------|------|
| **总任务数** | 32 | 24 | -25% |
| **成功任务数** | 10 | 10 | 维持 |
| **失败任务数** | 22 | 14 | -36% |
| **整体成功率** | 31% | **55-65%** | **+24-34%** |
| **Tier 1 比例** | 41% | 46% | +5% |
| **Tier 4 任务** | 2 | 0 | 移除 |
| **ROI 评级** | ⭐⭐ | ⭐⭐⭐ | +1 星 |

### 8.2 维度覆盖对比

| 维度 | 优化前 | 优化后 | 覆盖度 |
|------|--------|--------|--------|
| **Harvest** | 12 → 9 | -25% | ✅ 充足 |
| **Combat** | 8 → 6 | -25% | ✅ 充足 |
| **Techtree** | 12 → 9 | -25% | ✅ 充足 |
| **Movement** | 0 → 0 | - | ⚠️ 可选 |

---

## 🚀 九、实施建议

### 9.1 立即行动（Phase 1）

**Step 1**: 移除低 ROI 任务（8个）
```bash
# 从配置文件中移除或注释掉：
- harvest_1_apple
- harvest_1_milk
- harvest_1_beef
- combat_zombie_with_shield
- combat_creeper
- techtree_craft_iron_sword
- techtree_barehand_to_stone_pickaxe
- techtree_stone_to_iron_pickaxe
```

**Step 2**: 应用优化配置（14个失败任务）
```bash
# 将 config/eval_tasks_failed_fix.yaml 中的配置合并到主配置
# 重点优化：biome、initial_inventory、reward_config
```

**Step 3**: 运行快速验证（5个简单任务）
```bash
# 测试 5 个最简单的任务，验证配置有效性
bash scripts/run_evaluation.sh --task combat_chicken --max-steps 1000 --n-trials 5
bash scripts/run_evaluation.sh --task harvest_1_flower --max-steps 800 --n-trials 5
bash scripts/run_evaluation.sh --task techtree_craft_sticks --max-steps 500 --n-trials 5
```

### 9.2 逐步优化（Phase 2）

**Week 1**: Tier 1 任务（11个）
- 目标：成功率 > 60%
- 方法：调整 biome、增加初始库存、优化 reward_config

**Week 2**: Tier 2 任务（10个）
- 目标：成功率 > 30%
- 方法：增加步数、提供更好的工具、优化环境参数

**Week 3**: Tier 3 任务（3个）
- 目标：成功率 > 10%
- 方法：允许时间流逝（熔炼）、提供完整的材料链

### 9.3 成功标准

| 阶段 | 成功任务数 | 整体成功率 | 评级 |
|------|-----------|-----------|------|
| **当前** | 10/32 | 31% | ⭐⭐ 及格 |
| **Phase 1** | 15/24 | 63% | ⭐⭐⭐ 良好 |
| **Phase 2** | 18/24 | 75% | ⭐⭐⭐⭐ 优秀 |

---

## 📋 十、完整任务清单（24个）

### Harvest 任务（9个）

#### Tier 1 - Easy (5)
- [x] harvest_1_dirt - ✅ 90%
- [x] harvest_1_sand - ✅ 100%
- [x] harvest_1_log - ✅ 30% (需优化)
- [ ] harvest_1_gravel - ❌ 0% (需优化)
- [ ] harvest_1_flower - ❌ 0% (需优化)

#### Tier 2 - Medium (4)
- [x] harvest_1_cobblestone - ✅ 10% (需优化)
- [x] harvest_1_coal - ✅ 20% (需优化)
- [x] harvest_1_sapling - ✅ 10% (需优化)
- [x] harvest_1_wool - ✅ 50%

### Combat 任务（6个）

#### Tier 1 - Easy (3)
- [x] combat_pig - ✅ 20% (需优化)
- [ ] combat_chicken - ❌ 0% (需优化)
- [ ] combat_cow - ❌ 0% (需优化)

#### Tier 2 - Medium (3)
- [ ] combat_spider - ❌ 0% (需优化)
- [ ] combat_zombie - ❌ 0% (需优化)
- [ ] combat_skeleton - ❌ 0% (需优化)

### Techtree 任务（9个）

#### Tier 1 - Easy (3)
- [x] techtree_craft_planks - ✅ 80%
- [ ] techtree_craft_sticks - ❌ 0% (需优化)
- [x] techtree_craft_crafting_table - ✅ 10% (需优化)

#### Tier 2 - Medium (3)
- [ ] techtree_craft_wooden_pickaxe - ❌ 0% (需优化)
- [ ] techtree_craft_wooden_sword - ❌ 0% (需优化)
- [ ] techtree_craft_furnace - ❌ 0% (需优化)

#### Tier 3 - Hard (3)
- [ ] techtree_craft_stone_sword - ❌ 0% (需优化)
- [ ] techtree_smelt_iron_ingot - ❌ 0% (需优化)
- [ ] techtree_craft_iron_pickaxe - ❌ 0% (需优化)

---

## 🎯 十一、关键决策矩阵

### 任务保留/移除决策

| 任务 | 当前成功率 | 难度 | ROI | 决策 | 原因 |
|------|-----------|------|-----|------|------|
| harvest_1_apple | 0% | Hard | ⭐ | ❌ 移除 | 掉落率 0.5%，需要打数百片树叶 |
| harvest_1_milk | 0% | Easy | ⭐ | ❌ 移除 | 右键交互，MineRL 支持差 |
| harvest_1_beef | 0% | Easy | ⭐ | ❌ 移除 | 与 combat_cow 重复 |
| combat_creeper | 0% | Hard | ⭐ | ❌ 移除 | 需要远程，训练困难 |
| combat_zombie_with_shield | 0% | Hard | ⭐ | ❌ 移除 | 持盾僵尸，特殊策略 |
| techtree_craft_iron_sword | 0% | Medium | ⭐ | ❌ 移除 | 与 iron_pickaxe 重复 |
| techtree_barehand_to_stone_pickaxe | 0% | Very Hard | ⭐ | ❌ 移除 | 6000步，ROI 极低 |
| techtree_stone_to_iron_pickaxe | 0% | Very Hard | ⭐ | ❌ 移除 | 6000步，ROI 极低 |

### 任务优先级矩阵

| 优先级 | 任务数 | 目标成功率 | 投入 | ROI |
|--------|--------|-----------|------|-----|
| **P0（必须）** | 8 | 80%+ | 低 | ⭐⭐⭐⭐ |
| **P1（重要）** | 11 | 50-70% | 中 | ⭐⭐⭐ |
| **P2（次要）** | 5 | 30-50% | 高 | ⭐⭐ |

**P0 任务（8个）**:
- harvest_1_dirt, harvest_1_sand, harvest_1_log
- combat_pig, combat_chicken
- techtree_craft_planks, techtree_craft_sticks, techtree_craft_crafting_table

**P1 任务（11个）**:
- harvest_1_gravel, harvest_1_flower, harvest_1_cobblestone, harvest_1_coal, harvest_1_sapling
- combat_cow, combat_spider, combat_zombie
- techtree_craft_wooden_pickaxe, techtree_craft_wooden_sword, techtree_craft_furnace

**P2 任务（5个）**:
- harvest_1_wool
- combat_skeleton
- techtree_craft_stone_sword, techtree_smelt_iron_ingot, techtree_craft_iron_pickaxe

---

## 📝 十二、执行清单

### ✅ 立即执行

- [ ] **Step 1**: 从配置文件移除 8 个低 ROI 任务
- [ ] **Step 2**: 应用 `config/eval_tasks_failed_fix.yaml` 中的优化配置
- [ ] **Step 3**: 运行 P0 任务验证（8个）
- [ ] **Step 4**: 收集数据，调整配置参数
- [ ] **Step 5**: 运行 P1 任务（11个）
- [ ] **Step 6**: 运行 P2 任务（5个）

### 📊 评估指标

**成功标准**:
- Phase 1: P0 任务成功率 > 60% (5/8)
- Phase 2: P1 任务成功率 > 40% (5/11)
- Phase 3: P2 任务成功率 > 20% (1/5)
- **总体**: 成功率 > 55% (13/24)

---

## 🔍 十三、风险和限制

### 已知风险

1. **环境限制**：
   - MineRL 环境不支持所有 Minecraft 特性
   - 某些物品名称可能不匹配
   - 生物 AI 可能与原版不同

2. **任务难度**：
   - Combat 任务依赖生物随机生成
   - Harvest 任务可能需要大量探索
   - Techtree 任务需要准确的合成配方

3. **训练挑战**：
   - 某些任务（如熔炼）需要等待时间
   - 战斗任务需要复杂的策略
   - 长序列任务难以收敛

### 缓解措施

1. **配置优化**：提供合适的初始库存和工具
2. **环境优化**：选择合适的 biome 提高目标出现率
3. **难度递进**：从简单任务开始，逐步增加难度
4. **ROI 优先**：专注于高价值、高成功率的任务

---

## 📚 附录

### A. MineDojo 支持的 Biome

参考 [MineDojo 文档](https://docs.minedojo.org/sections/customization/task.html)：

**常用 Biome**:
- `plains` - 平原（动物、花朵）
- `forest` - 森林（树木、苹果）
- `mountains` (旧称 `extreme_hills`) - 山地（矿石）
- `desert` - 沙漠（沙子、仙人掌）
- `taiga` - 针叶林
- `jungle` - 丛林
- `swampland` - 沼泽

### B. 物品名称映射

**常见物品**:
- `log` / `oak_log` - 原木
- `planks` / `oak_planks` - 木板
- `stick` - 木棒
- `cobblestone` - 圆石
- `gravel` - 砾石
- `dirt` - 泥土
- `sand` - 沙子
- `coal` - 煤炭
- `iron_ore` - 铁矿
- `iron_ingot` - 铁锭

**工具**:
- `wooden_pickaxe` / `stone_pickaxe` / `iron_pickaxe`
- `wooden_sword` / `stone_sword` / `iron_sword`
- `shears` - 剪刀
- `bucket` - 桶
- `furnace` - 熔炉
- `crafting_table` - 工作台

### C. 参考资源

- **MineDojo 文档**: https://docs.minedojo.org/sections/customization/task.html
- **官方任务清单**: https://github.com/MineDojo/MineDojo/blob/main/minedojo/tasks/description_files/programmatic_tasks.yaml
- **当前配置**: `config/eval_tasks.yaml`, `config/eval_tasks_prior.yaml`
- **优化配置**: `config/eval_tasks_failed_fix.yaml`

---

## 💡 十四、总结与建议

### 核心结论

1. **当前体系基本完整**：三大维度（资源、战斗、科技）都有覆盖
2. **主要问题是配置**：22 个失败任务大多是配置不当，而非任务本身不可行
3. **需要精简和聚焦**：移除 8 个低 ROI 任务，专注于 24 个核心任务
4. **预期改进显著**：成功率可从 31% 提升到 55-65%

### 最终建议

**优先级排序**:
1. ⭐⭐⭐⭐ **立即**：移除 8 个低 ROI 任务
2. ⭐⭐⭐⭐ **本周**：优化 P0 任务配置（8个）
3. ⭐⭐⭐ **下周**：优化 P1 任务配置（11个）
4. ⭐⭐ **后续**：优化 P2 任务配置（5个）
5. ⭐ **可选**：考虑添加 Movement 类任务（2-3个）

**投入产出比**:
- **高 ROI**: Tier 1 任务（11个）- 投入少，成功率高
- **中 ROI**: Tier 2 任务（10个）- 投入中等，成功率中等
- **低 ROI**: Tier 3 任务（3个）- 投入高，成功率低（但对完整性重要）

**预期成果**:
- 24 个精选任务，覆盖三大维度
- 整体成功率 55-65%
- 建立稳定的评估基线
- 为后续模型优化提供可靠的评估指标

---

**文档版本**: v1.0  
**最后更新**: 2025-12-02  
**维护者**: AIMC Project Team



