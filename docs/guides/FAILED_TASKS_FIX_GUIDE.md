# 失败任务修复指南
# Failed Tasks Fix Guide

**版本**: v1.0  
**创建时间**: 2025-12-02  
**配置文件**: `config/eval_tasks_failed_fix.yaml`

## 📋 概述

本指南提供了 22 个失败任务（0% 成功率）的完整配置方案，包括：
- 7 个 Combat 任务（战斗类）
- 5 个 Harvest 任务（采集类）
- 10 个 Techtree 任务（科技树）

## 🎯 配置原则

### 1. Biome（群系）选择

| Biome | 用途 | 适用任务 |
|-------|------|----------|
| `plains` | 动物（鸡、牛、羊）、花朵 | combat_chicken, combat_cow, harvest_1_flower |
| `forest` | 树木（原木、苹果、树苗） | harvest_1_log, harvest_1_apple |
| `mountains` | 矿石（铁矿、煤炭、圆石） | harvest_1_iron_ore, harvest_1_coal |
| `desert` | 沙子、仙人掌 | harvest_1_sand |

### 2. 时间条件

| 时间 | 游戏刻 | 用途 |
|------|--------|------|
| 白天 | 6000 | 友好生物、常规任务 |
| 夜晚 | 13000 | 敌对生物（僵尸、骷髅、蜘蛛、爬行者）|

**特殊设置**：
- `allow_passage_of_time: false` - 大部分任务（时间冻结）
- `allow_passage_of_time: true` - 熔炼类任务（需要时间流逝）

### 3. 初始库存策略

| 任务类型 | 提供物品 | 原因 |
|----------|----------|------|
| Combat | 武器（剑、弓）、防具（盾牌）| 提高战斗效率和生存率 |
| Harvest | 工具（镐、桶、剑） | 提供必要的采集工具 |
| Techtree | 前置材料（木板、圆石、铁锭）| 缩短制作流程 |

### 4. 奖励配置

**物品类任务**：
```yaml
reward_config:
- entity: gravel          # 物品名称
  amount: 1               # 数量
  reward: 100             # 奖励分数
reward_rule: any          # 任意一个完成即可
```

**战斗类任务**：
```yaml
reward_config:
- event: kill_entity     # 事件类型
  entity_type: chicken   # 生物类型
  reward: 100
reward_rule: any
```

## 📊 任务难度分级

### Easy（简单）
**预期成功率**: 50-80%

- `combat_chicken`, `combat_cow` - 击杀友好生物
- `harvest_1_beef`, `harvest_1_flower`, `harvest_1_milk` - 简单采集
- `techtree_craft_sticks`, `techtree_craft_wooden_*` - 单步制作

**特点**：1-2 步完成，有充足初始资源

### Medium（中等）
**预期成功率**: 30-60%

- `combat_skeleton`, `combat_spider`, `combat_zombie_*` - 击杀敌对生物
- `harvest_1_apple`, `harvest_1_iron_ore` - 需要探索或挖掘
- `techtree_smelt_iron_ingot`, `techtree_craft_iron_*` - 多步制作

**特点**：3-5 步，需要特定条件或策略

### Hard（困难）
**预期成功率**: 10-40%

- `combat_creeper` - 需要远程攻击，避免爆炸
- `combat_zombie_with_shield` - 持盾僵尸，需要强武器

**特点**：需要特殊策略或技巧

### Very Hard（超难）
**预期成功率**: 5-20%

- `techtree_barehand_to_stone_pickaxe` - 完整科技树（砍树→木板→木棒→工作台→木镐→挖圆石→石镐）
- `techtree_stone_to_iron_pickaxe` - 石镐升级（挖铁矿→熔炼→制作铁镐）

**特点**：需要完成多个步骤，耗时长（3000-6000 步）

## 🔧 配置示例

### 示例 1: combat_chicken（简单战斗）

```yaml
- task_id: combat_chicken
  env_name: MineRLHarvestDefaultEnv-v0
  en_instruction: hunt chicken
  
  env_config:
    specified_biome: plains
    
    initial_inventory:
    - type: wooden_sword
      quantity: 1
    
    reward_config:
    - event: kill_entity
      entity_type: chicken
      reward: 100
    reward_rule: any
    
    time_condition:
      start_time: 6000
      allow_passage_of_time: false
    spawning_condition:
      allow_spawning: true
  
  max_steps: 1000
  n_trials: 3
```

### 示例 2: harvest_1_iron_ore（挖矿）

```yaml
- task_id: harvest_1_iron_ore
  env_name: MineRLHarvestDefaultEnv-v0
  en_instruction: mine iron ore
  
  env_config:
    specified_biome: mountains
    
    initial_inventory:
    - type: stone_pickaxe    # 必须用石镐
      quantity: 1
    
    reward_config:
    - entity: iron_ore
      amount: 1
      reward: 100
    reward_rule: any
  
  max_steps: 3000  # 找矿需要时间
  n_trials: 3
```

### 示例 3: techtree_smelt_iron_ingot（熔炼）

```yaml
- task_id: techtree_smelt_iron_ingot
  env_name: MineRLHarvestDefaultEnv-v0
  en_instruction: smelt iron ore
  
  env_config:
    initial_inventory:
    - type: furnace
      quantity: 1
    - type: iron_ore
      quantity: 1
    - type: coal
      quantity: 1
    
    reward_config:
    - entity: iron_ingot
      amount: 1
      reward: 100
    reward_rule: any
    
    time_condition:
      allow_passage_of_time: true  # 关键！
  
  max_steps: 2000
  n_trials: 3
```

## ⚠️ 重要注意事项

### Combat 任务

1. **生物生成随机性**：生物生成位置和数量是随机的，可能需要探索
2. **夜晚设置**：敌对生物需要 `start_time: 13000`
3. **生成开关**：必须设置 `allow_spawning: true`
4. **武器选择**：
   - 近战生物（猪、牛、鸡）：木剑足够
   - 远程生物（骷髅）：需要盾牌
   - 爆炸生物（爬行者）：建议弓箭

### Harvest 任务

1. **harvest_1_apple**：苹果掉落概率低（约 0.5%），需要打很多树叶
2. **harvest_1_iron_ore**：铁矿生成在 y=0-64，需要向下挖掘
3. **harvest_1_milk**：需要接近牛并右键使用桶

### Techtree 任务

1. **熔炼任务**：必须设置 `allow_passage_of_time: true`，否则熔炉不工作
2. **合成配方**：确保提供正确的材料数量和类型
3. **超难任务**：
   - `barehand_to_stone_pickaxe`：需要 6000 步
   - `stone_to_iron_pickaxe`：需要挖掘、熔炼、制作多个步骤

### 环境限制

1. **物品名称**：可能需要调整（如 `log` vs `oak_log`）
2. **生物 AI**：MineRL 的生物行为可能与原版不同
3. **合成系统**：某些复杂合成可能不被支持

## 🚀 使用方法

### 方法 1: 测试单个任务

```bash
# 1. 从 config/eval_tasks_failed_fix.yaml 复制任务配置
# 2. 添加到 config/eval_tasks.yaml 或 config/eval_tasks_prior.yaml
# 3. 运行评估
bash scripts/run_evaluation.sh --task combat_chicken
```

### 方法 2: 使用专家录制验证

```bash
# 手动验证任务可行性
bash scripts/record_expert_demo.sh --task combat_chicken --fullscreen

# 确认：
# - 初始库存是否正确
# - 生物是否生成
# - 奖励是否触发
# - 任务是否自动完成
```

### 方法 3: 批量评估

```bash
# 使用完整配置文件运行评估
python src/evaluation/eval_framework.py \
  --config config/eval_tasks_failed_fix.yaml \
  --task-set combat_tasks \
  --n-trials 10
```

## 🔍 调试建议

### 1. 检查物品/生物名称

```python
from minerl.herobraine.hero.mc import ALL_ITEMS
print(ALL_ITEMS)
```

### 2. 检查群系配置

```python
# 确认群系名称与 BIOME_ID_MAP 匹配
# 参考: src/envs/minerl_harvest_default.py

BIOME_ID_MAP = {
    "plains": 1,
    "forest": 4,
    "mountains": 3,
    "desert": 2,
    ...
}
```

### 3. 增加步数和尝试次数

```yaml
max_steps: 6000  # 增加到最大
n_trials: 10     # 多次尝试
```

### 4. 启用详细日志

```bash
# 录制时会显示详细的库存变化和奖励信息
bash scripts/record_expert_demo.sh --task combat_chicken

# 观察：
# [INVENTORY] 📦 库存变化: ...
# [REWARD] 🎉 获得奖励: ...
# [STATUS] Done=True
```

## 📈 预期改进

| 指标 | 原始 | 预期 |
|------|------|------|
| 失败任务数 | 22/22 (100%) | 预计 10-12/22 (45-55%) |
| Easy 任务成功率 | 0% | 50-80% |
| Medium 任务成功率 | 0% | 30-60% |
| Hard 任务成功率 | 0% | 10-40% |
| Very Hard 任务成功率 | 0% | 5-20% |

## ✅ 任务清单

### Combat 任务（7个）

- [ ] combat_chicken - 击杀鸡（Easy）
- [ ] combat_cow - 击杀牛（Easy）
- [ ] combat_creeper - 击杀爬行者（Hard）
- [ ] combat_skeleton - 击杀骷髅（Medium）
- [ ] combat_spider - 击杀蜘蛛（Medium）
- [ ] combat_zombie_leather_armor - 击杀僵尸（Medium）
- [ ] combat_zombie_with_shield - 击杀持盾僵尸（Hard）

### Harvest 任务（5个）

- [ ] harvest_1_apple - 获取苹果（Medium）
- [ ] harvest_1_beef - 获取牛肉（Easy）
- [ ] harvest_1_flower - 采花（Easy）
- [ ] harvest_1_iron_ore - 挖铁矿（Medium）
- [ ] harvest_1_milk - 挤奶（Easy）

### Techtree 任务（10个）

- [ ] techtree_craft_sticks - 制作木棒（Easy）
- [ ] techtree_craft_wooden_pickaxe - 制作木镐（Easy）
- [ ] techtree_craft_wooden_sword - 制作木剑（Easy）
- [ ] techtree_craft_stone_sword - 制作石剑（Easy）
- [ ] techtree_craft_furnace - 制作熔炉（Easy）
- [ ] techtree_smelt_iron_ingot - 熔炼铁锭（Medium）
- [ ] techtree_craft_iron_pickaxe - 制作铁镐（Medium）
- [ ] techtree_craft_iron_sword - 制作铁剑（Medium）
- [ ] techtree_barehand_to_stone_pickaxe - 从零到石镐（Very Hard）
- [ ] techtree_stone_to_iron_pickaxe - 石镐到铁镐（Very Hard）

## 📚 相关文档

- **配置文件**: `config/eval_tasks_failed_fix.yaml`
- **环境定义**: `src/envs/minerl_harvest_default.py`
- **群系参考**: `BIOME_REFERENCE.md`
- **评估框架**: `docs/guides/EVALUATION_FRAMEWORK_GUIDE.md`
- **专家录制**: `docs/guides/EXPERT_DEMO_RECORDING_GUIDE.md`

## 🔄 更新记录

- **2025-12-02**: 初始版本，包含所有 22 个失败任务的配置方案
