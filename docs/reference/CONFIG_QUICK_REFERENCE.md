# 📋 配置快速参考

**最后更新**: 2025-11-20

---

## 🎯 标准配置格式

```yaml
env_config:
  # 图像尺寸 [height, width]
  image_size: [320, 640]
  
  # 初始物品（使用 'type' 字段）
  initial_inventory:
    - type: "bucket"
      quantity: 1
  
  # 奖励配置（使用 'entity', 'amount', 'reward'）
  reward_config:
    - entity: "milk_bucket"
      amount: 1
      reward: 100
  
  # 世界生成（可选，仅 MineDojo 支持）
  specified_biome: "forest"
  world_seed: "test_seed"
  task_id: "open-ended"
  
  # 时间和生成
  start_time: 6000
  allow_mob_spawn: true
  
  # 其他
  max_episode_steps: 500
```

---

## ✅ 支持的配置项

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `image_size` | `[int, int]` | `[160, 256]` | 图像尺寸 [height, width] |
| `initial_inventory` | `List[Dict]` | `[]` | 初始物品，使用 `type` 字段 |
| `reward_config` | `List[Dict]` | - | 奖励配置，使用 `entity`, `amount`, `reward` |
| `specified_biome` | `str` | `None` | 指定生物群系（仅 MineDojo） |
| `world_seed` | `str` | - | 世界种子 |
| `task_id` | `str` | - | MineDojo 任务 ID |
| `start_time` | `int` | `6000` | 起始时间 (0-24000) |
| `allow_mob_spawn` | `bool` | `False` | 是否允许生物生成 |
| `spawn_in_village` | `bool` | `False` | 是否在村庄生成 |
| `break_speed_multiplier` | `float` | `1.0` | 破坏速度倍数 |
| `max_episode_steps` | `int` | - | 最大步数 |

---

## ❌ 不支持的配置项

| 配置项 | 原因 | 替代方案 |
|--------|------|----------|
| `resolution` | 已统一 | 使用 `image_size` |
| `generate_world_type` | 自动推断 | 使用 `specified_biome` |
| `allow_time_passage` | 默认不流逝 | 无需配置 |
| `reward_rule` | 自动处理 | 无需配置 |
| `world_generator` | MineDojo 不支持 | 无 |
| `time_condition` | 已扁平化 | 使用 `start_time` |
| `spawning_condition` | 已扁平化 | 使用 `allow_mob_spawn` |
| `allow_passage_of_time` | 已移除 | 无需配置 |
| `allow_spawning` | 已统一 | 使用 `allow_mob_spawn` |

---

## 🔄 物品名称映射

### 常用物品

| MineRL | MineDojo |
|--------|----------|
| `oak_planks` | `planks` |
| `spruce_planks` | `planks` |
| `oak_log` | `log` |
| `oak_sapling` | `sapling` |
| `lapis_lazuli` | `dye` |
| `sugar_cane` | `reeds` |
| `dandelion` | `yellow_flower` |
| `poppy` | `red_flower` |
| `white_wool` | `wool` |

**完整映射**: 见 `src/envs/item_name_mapper.py`

---

## 📝 使用示例

### MineDojo 任务

```yaml
- task_id: "harvest_1_log"
  env_name: "MineDojoHarvestEnv-v0"
  env_config:
    specified_biome: "forest"
    world_seed: "harvest_log_test"
    task_id: "open-ended"
    image_size: [320, 640]
    initial_inventory:
      - type: "bucket"
        quantity: 1
    start_time: 6000
    allow_mob_spawn: false
    max_episode_steps: 500
```

### MineRL 任务

```yaml
- task_id: "harvest_1_milk"
  env_name: "MineRLHarvestEnv-v0"
  env_config:
    reward_config:
      - entity: "milk_bucket"
        amount: 1
        reward: 100
    initial_inventory:
      - type: "bucket"
        quantity: 1
    start_time: 6000
    allow_mob_spawn: true
```

---

## 🛠️ 工具和脚本

### 配置清理脚本

```bash
# 清理配置文件（会自动备份）
python scripts/clean_eval_tasks_config.py

# Dry-run 模式（只显示变更）
python scripts/clean_eval_tasks_config.py --dry-run

# 指定输出文件
python scripts/clean_eval_tasks_config.py -o config/eval_tasks_clean.yaml
```

### 配置验证

```bash
# 检查不支持的配置项
grep -n "generate_world_type\|allow_time_passage\|reward_rule\|world_generator\|time_condition\|spawning_condition" config/eval_tasks.yaml

# 应该返回空（或返回 0）
```

---

## 📚 相关文档

| 文档 | 说明 |
|------|------|
| `CONFIG_UNIFICATION_COMPLETE.md` | 配置统一完成报告 |
| `CONFIG_CLEANUP_REPORT.md` | 配置清理报告 |
| `docs/summaries/CONFIG_UNIFICATION_SUMMARY.md` | 详细的统一总结 |
| `docs/summaries/ITEM_NAME_MAPPING_IMPLEMENTATION.md` | 物品映射实现 |
| `src/envs/config_normalizer.py` | 配置标准化器 |
| `src/envs/item_name_mapper.py` | 物品名称映射器 |

---

## 🆘 常见问题

### Q: 旧配置还能用吗？

**A**: 可以！旧格式会自动转换，但建议使用新格式。

### Q: 如何恢复原配置？

**A**: `cp config/eval_tasks.yaml.backup config/eval_tasks.yaml`

### Q: 如何添加新物品映射？

**A**: 编辑 `src/envs/item_name_mapper.py`，添加到 `MINERL_TO_MINEDOJO_ITEM_MAP`

### Q: 配置转换失败怎么办？

**A**: 检查日志中的 `🔄` 转换信息，参考文档修正配置

---

**快速开始**: 直接使用标准配置格式编写新任务！ 🚀

