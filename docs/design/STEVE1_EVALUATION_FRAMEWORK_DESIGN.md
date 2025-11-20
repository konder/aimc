# Steve1 模型三维评估框架设计方案
# Steve1 Model Three-Dimensional Evaluation Framework Design

> **版本**: v1.0  
> **创建日期**: 2025-11-20  
> **设计目标**: 基于MineDojo 1,581个预设任务，构建全面评估Steve1模型的Harvest、Combat、TechTree三维任务体系  
> **参考文档**: [MineDojo Programmatic Tasks](https://docs.minedojo.org/sections/core_api/sim.html#programmatic-tasks)

---

## 📋 目录

1. [设计背景](#设计背景)
2. [MineDojo任务体系分析](#minedojo任务体系分析)
3. [三维评估框架设计](#三维评估框架设计)
4. [任务选择策略](#任务选择策略)
5. [环境配置策略](#环境配置策略)
6. [指令优化指南](#指令优化指南)
7. [评估指标体系](#评估指标体系)
8. [实施建议](#实施建议)

---

## 设计背景

### Steve1模型特点

Steve1是基于VPT（Video Pre-Training）的多模态Minecraft智能体：

- **核心机制**: 通过MineCLIP将文本指令和视觉画面关联
- **训练方式**: 使用视频时序数据（后一帧作为前一帧的目标）
- **关键特性**: 
  - ✅ 支持自然语言指令控制
  - ⚠️ 对文字提示非常敏感
  - ⚠️ 指令措辞会显著影响成功率

### 评估目标

构建一套**科学、全面、可扩展**的评估体系，用于：

1. **能力基准测试**: 评估Steve1在不同维度的基础能力
2. **指令优化**: 找到最适合Steve1的指令表达方式
3. **弱点识别**: 发现模型在特定任务类型上的不足
4. **进化追踪**: 为模型改进提供量化指标

---

## MineDojo任务体系分析

根据[MineDojo官方文档](https://docs.minedojo.org/sections/core_api/sim.html#programmatic-tasks)，编程任务统计如下：

| 任务类别 | 数量 | 占比 | 特点 |
|---------|------|------|------|
| **Harvest** | 895 | 56.6% | 采集资源，覆盖面最广 |
| **Combat** | 471 | 29.8% | 战斗生物，测试反应和策略 |
| **TechTree** | 213 | 13.5% | 科技发展，测试规划能力 |
| **Survival** | 2 | 0.1% | 综合生存，最高难度 |
| **总计** | 1,581 | 100% | 编程任务总数 |

### 任务命名规则

MineDojo任务遵循标准命名模式：

```
Harvest:   harvest_<数量>_<物品>_[biome]_[条件]
Combat:    combat_<生物>_[biome]_[装备]_[武器]_[副手]
TechTree:  techtree_from_<起点>_to_<目标>
Survival:  survival[_条件]
```

**示例**:
- `harvest_1_milk` - 采集1桶牛奶
- `harvest_wool_with_shears_and_sheep` - 有剪刀和羊的情况下采羊毛
- `combat_zombie_forest_leather_armors_wooden_sword_shield` - 森林中穿皮甲用木剑盾牌打僵尸
- `techtree_from_barehand_to_wooden_sword` - 从空手到木剑
- `techtree_from_iron_to_diamond_pickaxe` - 从铁器到钻石镐

---

## 三维评估框架设计

### 设计原则

1. **渐进式难度**: 从简单到复杂，分4-5个难度等级
2. **代表性采样**: 从1,581个任务中精选40-50个核心任务
3. **能力覆盖**: 确保覆盖各维度的关键能力点
4. **指令友好**: 优先选择可用简洁指令表达的任务
5. **可扩展性**: 预留扩展接口，方便后续增加任务

### 三维能力矩阵

```
维度         Level 1        Level 2        Level 3        Level 4        Level 5
────────────────────────────────────────────────────────────────────────────
Harvest      基础采集       动物互动       工具使用       矿石挖掘       复杂制造
             (空手)        (牛奶/羊毛)    (镐/斧/铲)     (煤/铁/金)     (熔炉/合成)

Combat       被动生物       装备战斗       简单敌对       困难敌对       Boss战斗
             (鸡/猪)       (木剑+皮甲)    (僵尸/蜘蛛)    (骷髅/爬行者)   (末影人/凋灵)

TechTree     基础合成       木制工具       石制工具       铁制工具       钻石工具
             (木板/棍子)    (木镐/木剑)    (石镐/熔炉)    (铁镐/铁甲)    (钻石镐/剑)
```

---

## 任务选择策略

### Harvest维度 (13-15个任务)

#### Level 1: 基础采集 (3-4个)
最简单的资源获取，验证基本移动和交互能力

| MineDojo Task ID | 中文描述 | 推荐指令 | 环境配置 |
|-----------------|---------|---------|---------|
| `harvest_1_log` | 砍1个原木 | `chop tree` | forest |
| `harvest_1_dirt` | 挖1个泥土 | `dig dirt` | 任意 |
| `harvest_1_sand` | 挖1个沙子 | `dig sand` | desert/beach |
| `harvest_1_cobblestone` (提供木镐) | 挖1个圆石 | `mine stone` | mountains |

**关键能力**: 移动、视角控制、基础交互

#### Level 2: 动物互动 (3-4个)
需要寻找并与生物互动

| MineDojo Task ID | 中文描述 | 推荐指令 | 环境配置 |
|-----------------|---------|---------|---------|
| `harvest_1_milk` | 获取牛奶 | `milk cow` | plains (无biome更好) |
| `harvest_wool_with_shears` | 剪羊毛 | `shear sheep` | plains |
| `harvest_1_beef` | 获取牛肉 | `kill cow` | plains |
| `harvest_1_leather` | 获取皮革 | `get leather from cow` | plains |

**关键能力**: 生物识别、接近、互动、物品栏管理

**指令优化建议**:
- ✅ `milk cow` (简洁动词+名词)
- ⚠️ `find cow and get milk` (复合动作，容易混淆)
- ⚠️ `use bucket on cow` (过于具体)

#### Level 3: 工具使用 (3-4个)
使用工具采集特定资源

| MineDojo Task ID | 中文描述 | 推荐指令 | 环境配置 |
|-----------------|---------|---------|---------|
| `harvest_1_coal` (提供木镐) | 挖煤炭 | `mine coal` | mountains/extreme_hills |
| `harvest_1_iron_ore` (提供石镐) | 挖铁矿 | `mine iron ore` | mountains/extreme_hills |
| `harvest_1_gold_ore` (提供铁镐) | 挖金矿 | `mine gold ore` | badlands |
| `harvest_1_diamond` (提供铁镐) | 挖钻石 | `mine diamond` | 任意 |

**关键能力**: 工具选择、矿石识别、深度挖掘

#### Level 4: 植物采集 (2-3个)
寻找和采集地表植物资源

| MineDojo Task ID | 中文描述 | 推荐指令 | 环境配置 |
|-----------------|---------|---------|---------|
| `harvest_1_red_flower` | 采集花朵 | `pick flower` | flower_forest |
| `harvest_1_apple` | 获取苹果 | `get apple from tree` | forest |
| `harvest_1_sapling` | 获取树苗 | `get sapling` | forest |

**关键能力**: 资源搜索、稀有物品识别

---

### Combat维度 (10-12个任务)

#### Level 1: 被动生物 (3个)
最简单的战斗，验证基本攻击能力

| MineDojo Task ID | 中文描述 | 推荐指令 | 装备 |
|-----------------|---------|---------|-----|
| `combat_chicken_plains_wooden_sword` | 击杀鸡 | `kill chicken` | 木剑 |
| `combat_pig_plains_wooden_sword` | 击杀猪 | `kill pig` | 木剑 |
| `combat_cow_plains_wooden_sword` | 击杀牛 | `kill cow` | 木剑 |

**关键能力**: 目标锁定、攻击时机、追击

#### Level 2: 装备战斗 (2-3个)
使用装备与生物战斗

| MineDojo Task ID | 中文描述 | 推荐指令 | 装备 |
|-----------------|---------|---------|-----|
| `combat_zombie_plains_leather_armors_wooden_sword_shield` | 穿皮甲打僵尸 | `kill zombie` | 皮甲套+木剑+盾 |
| `combat_spider_forest_leather_armors_wooden_sword` | 打蜘蛛 | `kill spider` | 皮甲套+木剑 |

**关键能力**: 装备使用、防御、生命值管理

#### Level 3: 简单敌对生物 (2-3个)
与基础敌对生物战斗

| MineDojo Task ID | 中文描述 | 推荐指令 | 装备 |
|-----------------|---------|---------|-----|
| `combat_zombie_plains_iron_armors_iron_sword_shield` | 铁装打僵尸 | `kill zombie` | 铁甲套+铁剑+盾 |
| `combat_spider_forest_iron_armors_iron_sword` | 铁装打蜘蛛 | `kill spider` | 铁甲套+铁剑 |
| `combat_cave_spider_plains_iron_armors_iron_sword` | 打洞穴蜘蛛 | `kill cave spider` | 铁甲套+铁剑 |

**关键能力**: 战斗策略、走位、combo攻击

#### Level 4: 困难敌对生物 (2-3个)
远程攻击或特殊机制的敌对生物

| MineDojo Task ID | 中文描述 | 推荐指令 | 装备 |
|-----------------|---------|---------|-----|
| `combat_skeleton_plains_iron_armors_iron_sword_shield` | 打骷髅 | `kill skeleton` | 铁甲套+铁剑+盾 |
| `combat_creeper_plains_diamond_armors_diamond_sword` | 打苦力怕 | `kill creeper` | 钻石甲套+钻石剑 |
| `combat_witch_swampland_diamond_armors_diamond_sword_shield` | 打女巫 | `kill witch` | 钻石甲套+钻石剑+盾 |

**关键能力**: 距离控制、躲避远程攻击、高级战术

---

### TechTree维度 (14-16个任务)

#### Level 1: 基础合成 (4-5个)
最基本的物品转换和合成

| MineDojo Task ID | 中文描述 | 推荐指令 | 初始物品 |
|-----------------|---------|---------|---------|
| `techtree_planks` | 制作木板 | `craft planks` | log x1 |
| `techtree_crafting_table` | 制作工作台 | `craft crafting table` | planks x4 |
| `techtree_sticks` | 制作木棍 | `craft sticks` | planks x2 |
| `techtree_torch` | 制作火把 | `craft torch` | coal x1, stick x1 |
| `techtree_chest` | 制作箱子 | `craft chest` | planks x8, crafting_table x1 |

**关键能力**: GUI交互、配方识别、基础合成

#### Level 2: 木制工具 (3-4个)
第一级工具制作

| MineDojo Task ID | 中文描述 | 推荐指令 | 初始物品 |
|-----------------|---------|---------|---------|
| `techtree_wooden_pickaxe` | 制作木镐 | `craft wooden pickaxe` | planks x3, stick x2, crafting_table |
| `techtree_wooden_sword` | 制作木剑 | `craft wooden sword` | planks x2, stick x1, crafting_table |
| `techtree_wooden_axe` | 制作木斧 | `craft wooden axe` | planks x3, stick x2, crafting_table |
| `techtree_from_barehand_to_wooden_pickaxe` | 从空手到木镐 | `make wooden pickaxe` | 空手开始 |

**关键能力**: 工具合成链、多步骤规划

#### Level 3: 石制工具 (3-4个)
升级到石器时代

| MineDojo Task ID | 中文描述 | 推荐指令 | 初始物品 |
|-----------------|---------|---------|---------|
| `techtree_stone_pickaxe` | 制作石镐 | `craft stone pickaxe` | cobblestone x3, stick x2, crafting_table |
| `techtree_stone_sword` | 制作石剑 | `craft stone sword` | cobblestone x2, stick x1, crafting_table |
| `techtree_furnace` | 制作熔炉 | `craft furnace` | cobblestone x8, crafting_table |
| `techtree_from_barehand_to_stone_pickaxe` | 从空手到石镐 | `make stone pickaxe` | 空手开始 |

**关键能力**: 挖掘圆石、科技升级、熔炉解锁

#### Level 4: 铁制工具 (3-4个)
冶炼和铁器制作

| MineDojo Task ID | 中文描述 | 推荐指令 | 初始物品 |
|-----------------|---------|---------|---------|
| `techtree_iron_ingot` | 冶炼铁锭 | `smelt iron` | iron_ore x1, coal x1, furnace x1 |
| `techtree_iron_pickaxe` | 制作铁镐 | `craft iron pickaxe` | iron_ingot x3, stick x2, crafting_table |
| `techtree_iron_sword` | 制作铁剑 | `craft iron sword` | iron_ingot x2, stick x1, crafting_table |
| `techtree_from_stone_to_iron_pickaxe` | 从石器到铁镐 | `make iron pickaxe` | 石镐+工作台 |

**关键能力**: 熔炉使用、燃料管理、冶炼流程

#### Level 5: 钻石工具 (挑战) (1-2个)
最高级工具制作

| MineDojo Task ID | 中文描述 | 推荐指令 | 初始物品 |
|-----------------|---------|---------|---------|
| `techtree_diamond_pickaxe` | 制作钻石镐 | `craft diamond pickaxe` | diamond x3, stick x2, crafting_table |
| `techtree_from_iron_to_diamond_pickaxe` | 从铁器到钻石镐 | `make diamond pickaxe` | 铁镐+工作台 |

**关键能力**: 深度挖掘、钻石寻找、完整科技树

---

## 环境配置策略

### Biome选择原则

基于[MineDojo文档](https://docs.minedojo.org/sections/core_api/sim.html#programmatic-tasks)和实践经验：

#### 1. Harvest任务Biome推荐

| 资源类型 | 推荐Biome | 理由 |
|---------|----------|-----|
| 木头/树叶 | `forest`, `birch_forest` | 树木密集，易于找到 |
| 动物产品(牛奶/羊毛) | `plains`, 无指定 | 开阔平坦，动物生成率高 |
| 矿石(煤/铁/金) | `mountains`, `extreme_hills` | 露天矿脉多，地形起伏利于找矿 |
| 钻石 | 不指定或`mountains` | 需要深度挖掘，biome影响不大 |
| 花朵 | `flower_forest`, `plains` | 地表植被丰富 |
| 沙子 | `desert`, `beach` | 大量沙子 |

#### 2. Combat任务Biome推荐

| 战斗类型 | 推荐Biome | 理由 |
|---------|----------|-----|
| 被动生物 | `plains` | 开阔，便于追击 |
| 夜间怪物 | `plains`, `forest` | 平坦或有遮挡的多样地形 |
| 特殊生物(女巫) | `swampland` | 女巫小屋生成地 |

#### 3. TechTree任务Biome推荐

| 发展阶段 | 推荐Biome | 理由 |
|---------|----------|-----|
| 木制工具 | `forest` | 木材充足 |
| 石制工具 | `forest` + `mountains` | 需要木材和石头 |
| 铁制工具 | `mountains`, `extreme_hills` | 需要铁矿 |
| 钻石工具 | 不指定 | 需要深度挖掘 |

### 时间和生物生成配置

```yaml
# 白天作业（推荐大部分任务）
time_condition:
  start_time: 6000              # 正午
  allow_passage_of_time: false  # 固定时间

spawning_condition:
  allow_spawning: true          # 允许生物生成

# 夜间战斗（敌对生物）
time_condition:
  start_time: 18000             # 午夜
  allow_passage_of_time: false

spawning_condition:
  allow_spawning: true
```

---

## 指令优化指南

### Steve1指令设计原则

基于Steve1对文字的敏感性，遵循以下原则：

#### 1. 简洁性原则
**推荐**: 动词 + 名词 (2-3词)
```
✅ chop tree
✅ milk cow
✅ kill zombie
✅ craft pickaxe
```

**避免**: 复杂句式、多个动作
```
❌ find cow and get milk
❌ go to tree and chop it down
❌ search for zombie and kill it
```

#### 2. 具体性原则
**推荐**: 明确目标对象
```
✅ mine coal
✅ dig dirt
✅ shear sheep
```

**避免**: 模糊表达
```
❌ get resources
❌ find things
❌ do task
```

#### 3. 游戏术语原则
**推荐**: 使用Minecraft玩家常用术语
```
✅ mine ore (不是 "excavate ore")
✅ smelt iron (不是 "heat iron in furnace")
✅ craft sword (不是 "make sword")
```

#### 4. 一致性原则
同类任务使用相同的指令模式：

```yaml
# 采集类
harvest_1_log:      "chop tree"
harvest_1_coal:     "mine coal"
harvest_1_iron_ore: "mine iron ore"

# 动物类
harvest_1_milk:     "milk cow"
harvest_1_wool:     "shear sheep"
harvest_1_beef:     "kill cow"

# 制作类
techtree_wooden_pickaxe: "craft wooden pickaxe"
techtree_stone_sword:    "craft stone sword"
techtree_iron_helmet:    "craft iron helmet"
```

### 指令测试策略

对于成功率低的任务，尝试以下变体：

```python
# 牛奶任务示例
variants = [
    "milk cow",              # 首选：简洁动词+名词
    "get milk",              # 备选1：只关注目标
    "use bucket on cow",     # 备选2：明确工具
    "collect milk from cow", # 备选3：完整描述
]

# 逐个测试，记录成功率
for instruction in variants:
    success_rate = evaluate(task_id, instruction, n_trials=10)
    print(f"{instruction}: {success_rate}%")
```

---

## 评估指标体系

### 1. 任务级指标

| 指标 | 计算方式 | 说明 |
|-----|---------|-----|
| **成功率** | 成功次数 / 总试验次数 | 主要指标 |
| **平均步数** | Σ(成功任务步数) / 成功次数 | 效率指标 |
| **首次成功步数** | 首次成功的步数 | 探索能力 |
| **稳定性** | 成功试验中步数的标准差 | 一致性 |

### 2. 维度级指标

```
维度得分 = Σ(任务成功率 × 任务权重) / Σ(任务权重)

权重分配:
- Level 1: 权重 1.0 (基础能力)
- Level 2: 权重 1.5 (重要能力)
- Level 3: 权重 2.0 (核心能力)
- Level 4: 权重 2.5 (高级能力)
- Level 5: 权重 3.0 (挑战能力)
```

### 3. 综合评分

```
Steve1综合得分 = 
    Harvest得分 × 0.40 +
    Combat得分  × 0.30 +
    TechTree得分 × 0.30

说明:
- Harvest权重最高(40%)，因为是最基础且应用最广的能力
- Combat和TechTree各30%，反映战斗和发展能力
```

### 4. 能力雷达图

生成六边形雷达图展示各子维度能力：

```
       Harvest_Basic
            /\
           /  \
Harvest_Tool    Combat_Passive
          |    |
TechTree_Basic  Combat_Hostile
           \  /
            \/
       TechTree_Advanced
```

---

## 实施建议

### Phase 1: 基础评估 (Week 1-2)

**目标**: 建立基准，验证框架

```bash
# 1. 运行Level 1任务(最简单)
python -m src.evaluation.run_evaluation \
  --config config/eval_tasks_comprehensive.yaml \
  --difficulty easy \
  --n_trials 5

# 2. 分析结果，调整指令
# 3. 重新测试成功率低的任务(<30%)
```

**预期输出**:
- Harvest Level 1: 60-80%成功率
- Combat Level 1: 50-70%成功率
- TechTree Level 1: 40-60%成功率

### Phase 2: 全面评估 (Week 3-4)

**目标**: 完整跑通所有任务

```bash
# 运行全部任务
python -m src.evaluation.run_evaluation \
  --config config/eval_tasks_comprehensive.yaml \
  --n_trials 10
```

**预期时间**:
- 40个任务 × 10次试验 × 平均1000步 × 0.05秒/步 = 约5.5小时

### Phase 3: 指令优化 (Week 5-6)

**目标**: 针对低成功率任务优化指令

1. 识别成功率<40%的任务
2. 为每个任务设计3-5个指令变体
3. A/B测试找到最佳指令
4. 更新配置文件

### Phase 4: 扩展评估 (Week 7+)

**目标**: 增加任务覆盖面

1. 从MineDojo 1,581个任务中选择新任务
2. 补充当前弱项维度的任务
3. 添加多目标组合任务
4. 长期生存任务

---

## 附录

### A. MineDojo任务访问方式

```python
import minedojo

# 获取所有任务ID
all_task_ids = minedojo.tasks.ALL_PROGRAMMATIC_TASK_IDS
print(f"Total tasks: {len(all_task_ids)}")  # 1581

# 获取任务指令
task_prompt, task_guidance = minedojo.tasks.ALL_PROGRAMMATIC_TASK_INSTRUCTIONS[task_id]

# 创建任务
env = minedojo.make(
    task_id="harvest_1_milk",
    image_size=(160, 256)
)
```

### B. 配置文件模板

参见: `config/eval_tasks_comprehensive.yaml`

### C. 参考资源

1. [MineDojo官方文档](https://docs.minedojo.org/sections/core_api/sim.html#programmatic-tasks)
2. [MineDojo GitHub](https://github.com/MineDojo/MineDojo)
3. [Steve1论文](https://arxiv.org/abs/2306.00937)
4. 本地文档: `docs/reference/MINEDOJO_TASKS_REFERENCE.md`

---

## 版本历史

- **v1.0** (2025-11-20): 初始版本，基于MineDojo 1,581任务设计三维评估框架

---

**文档维护者**: AIMC项目团队  
**最后更新**: 2025-11-20

