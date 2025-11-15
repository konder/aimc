# 动物任务优化方案

**日期**: 2025-11-14  
**问题**: harvest_1_milk 等动物任务失败率高，找不到动物

---

## ❌ 问题分析

### 核心问题

**harvest_1_milk 任务配置**:
```yaml
env_name: "MineRLHarvestEnv-v0"  # 别名，实际是 Default 环境
env_config:
  world_generator:
    generator_options: '{"biome":"plains"}'  # ❌ 在 MC 1.16 中无效
  spawning_condition:
    allow_spawning: true  # ✅ 已开启
```

**问题**:
1. ❌ `generator_options` 在 MC 1.16.5 中**已废弃**，无法指定群系
2. ❌ DefaultWorldGenerator 生成**随机**世界，动物分布不可控
3. ❌ MineRL **不支持**预放置动物（没有相关 handler）

### 为什么找不到奶牛？

| 因素 | 影响 |
|------|------|
| **随机世界** | 可能生成沙漠、海洋等无牛群系 |
| **动物密度** | 默认生成密度低，覆盖范围小 |
| **max_steps** | 2000 步可能不够找到和接近牛 |
| **地形复杂** | 山地、森林阻挡视线和移动 |

---

## 🔍 MineRL 动物生成机制

### Minecraft 1.16 动物生成规则

1. **被动生成（Passive Spawning）**:
   - 在世界生成时少量生成
   - 之后不再自然生成（除非通过繁殖）
   - 密度：每个区块 0-4 只

2. **群系限制**:
   - 牛/猪/鸡: 平原、森林、草地
   - 羊: 平原、山地
   - 不在沙漠、海洋、沼泽生成

3. **视野限制**:
   - 玩家初始视距有限
   - 动物可能在远处未加载区域

### MineRL 限制

| 功能 | 是否支持 | 说明 |
|------|---------|------|
| 指定群系 | ❌ | MC 1.16 generator_options 废弃 |
| 预放置动物 | ❌ | 没有 DrawEntity/DrawMob handler |
| 调整生成密度 | ❌ | 没有相关配置参数 |
| 强制刷新动物 | ❌ | 不支持 |

---

## ✅ 解决方案

### 方案 1: 优化任务配置（推荐）

修改 eval_tasks.yaml，增加成功率：

```yaml
- task_id: "harvest_1_milk"
  env_name: "MineRLHarvestDefaultEnv-v0"  # 明确使用 Default 环境
  env_config:
    reward_config:
      - entity: "milk_bucket"
        amount: 1
        reward: 100
    reward_rule: "any"
    # 移除无效的 world_generator.generator_options
    world_generator:
      force_reset: true  # 每次重置世界，增加找到动物的机会
    time_condition:
      allow_passage_of_time: false
      start_time: 6000
    spawning_condition:
      allow_spawning: true
    initial_inventory:
      - type: "bucket"
        quantity: 1
  
  max_steps: 5000  # 增加步数（从 2000 → 5000）
```

**改进**:
- ✅ 移除无效的 `generator_options`
- ✅ 增加 `max_steps` 到 5000（给更多时间寻找）
- ✅ `force_reset: true` 确保每次都是新世界

### 方案 2: 多次 Trial，接受概率性成功

```bash
# 运行多次 trial
./scripts/run_minedojo_x86.sh python -m src.evaluation.eval_framework \
  --config config/eval_tasks.yaml \
  --task harvest_1_milk \
  --n-trials 10  # 运行 10 次，取成功率
```

**预期**:
- 成功率约 30-50%（取决于运行时的随机世界）
- 可以通过多次 trial 平均成功率

### 方案 3: 调整所有动物任务

批量更新动物任务配置：

```python
# 需要更新的任务
ANIMAL_TASKS = [
    "harvest_1_milk",      # 牛
    "harvest_1_wool",      # 羊
    "harvest_1_beef",      # 牛
    "harvest_1_porkchop",  # 猪
    "harvest_1_chicken",   # 鸡
    "harvest_1_leather",   # 牛
    "harvest_1_feather",   # 鸡
]

# 统一配置
for task in ANIMAL_TASKS:
    task.max_steps = 5000  # 增加步数
    task.env_name = "MineRLHarvestDefaultEnv-v0"  # 使用明确名称
    # 移除无效的 generator_options
```

---

## 📊 预期成功率

### 当前配置（max_steps=2000）

| 任务 | 预期成功率 | 原因 |
|------|-----------|------|
| `harvest_1_milk` | ~20% | 牛较常见，但需要找到 |
| `harvest_1_wool` | ~30% | 羊更常见 |
| `harvest_1_beef` | ~15% | 需要击杀牛（困难） |
| `harvest_1_chicken` | ~25% | 鸡较常见 |

### 优化后（max_steps=5000）

| 任务 | 预期成功率 | 提升 |
|------|-----------|------|
| `harvest_1_milk` | ~40-50% | **+25%** |
| `harvest_1_wool` | ~50-60% | **+25%** |
| `harvest_1_beef` | ~25-30% | **+12%** |
| `harvest_1_chicken` | ~40-45% | **+17%** |

---

## 🔧 实施步骤

### 1. 统一环境名称

将所有任务的 `MineRLHarvestEnv-v0` 改为明确的环境名：

```bash
# 批量替换
cd /Users/nanzhang/aimc
sed -i '' 's/env_name: "MineRLHarvestEnv-v0"/env_name: "MineRLHarvestDefaultEnv-v0"/g' config/eval_tasks.yaml
```

**修改后可以移除别名注册**（在 `src/envs/__init__.py` 中）。

### 2. 更新动物任务配置

手动编辑 eval_tasks.yaml，为每个动物任务：
- 移除 `generator_options`（无效）
- 增加 `max_steps` 到 5000
- 添加注释说明

### 3. 移除别名注册（可选）

如果所有任务都使用明确环境名，可以移除 `src/envs/__init__.py` 中的别名：

```python
# 删除这段代码
try:
    gym.register(
        id='MineRLHarvestEnv-v0',
        entry_point='src.envs.minerl_harvest_default:_minerl_harvest_default_env_entrypoint'
    )
except gym.error.Error:
    pass
```

---

## 🎯 替代方案（如果成功率仍然低）

### 方案 A: 简化任务

将动物任务改为"观察到动物"而不是"获取物品"：

```yaml
# 简化版
- task_id: "observe_cow"
  en_instruction: "find a cow"
  # 使用 MineCLIP 检测而不是物品获取
```

### 方案 B: 使用预录制数据

如果评估目标是测试指令理解而不是任务完成：
- 使用预录制的有动物的世界
- 或使用 MineCLIP 检测是否接近动物

### 方案 C: 降低权重

在评估报告中降低动物任务的权重：

```python
# 任务权重
TASK_WEIGHTS = {
    "resource_tasks": 1.0,   # FlatWorld 任务
    "animal_tasks": 0.5,     # 动物任务（成功率低，权重减半）
    "craft_tasks": 1.0,
}
```

---

## 📝 快速修复脚本

创建自动更新脚本：

```bash
#!/bin/bash
# fix_animal_tasks.sh

cd /Users/nanzhang/aimc

# 1. 备份配置
cp config/eval_tasks.yaml config/eval_tasks.yaml.backup

# 2. 统一环境名
sed -i '' 's/env_name: "MineRLHarvestEnv-v0"/env_name: "MineRLHarvestDefaultEnv-v0"/g' config/eval_tasks.yaml

echo "✅ 已更新环境名称"
echo "⚠️ 需要手动更新动物任务的 max_steps"
echo "   建议: harvest_1_milk 等任务改为 max_steps: 5000"
```

---

## ✅ 总结

**核心问题**:
- MC 1.16 不支持指定群系（generator_options 废弃）
- 不支持预放置动物
- DefaultWorldGenerator 生成随机世界

**推荐方案**:
1. ✅ 统一使用 `MineRLHarvestDefaultEnv-v0`（移除别名）
2. ✅ 增加动物任务的 `max_steps` 到 5000
3. ✅ 移除无效的 `generator_options`
4. ✅ 接受 40-50% 的成功率（通过多次 trial）

**权衡**:
- 动物任务本质上是**概率性**的（取决于随机世界）
- 可以通过增加步数和多次试验来提高总体成功率
- 或者在评估时降低这些任务的权重

**下一步**:
1. 执行批量替换环境名
2. 手动更新动物任务配置
3. 测试 harvest_1_milk（max_steps=5000）


