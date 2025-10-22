# MineDojo Programmatic Tasks 完整参考手册

> **最后更新**: 2025-10-20  
> **MineDojo版本**: 官方最新版  
> **任务总数**: 1,572 个编程任务

---

## 📋 目录

1. [任务概览](#任务概览)
2. [任务分类](#任务分类)
3. [Harvest 采集任务](#harvest-采集任务)
4. [Combat 战斗任务](#combat-战斗任务)
5. [TechTree 科技树任务](#techtree-科技树任务)
6. [Survival 生存任务](#survival-生存任务)
7. [奖励机制](#奖励机制)
8. [使用示例](#使用示例)

---

## 任务概览

MineDojo提供了**1,572个**内置的Programmatic Tasks（编程任务），涵盖了Minecraft游戏的各个方面。这些任务可用于训练和评估具身智能体。

### 任务统计

| 类别 | 数量 | 占比 |
|------|------|------|
| **Harvest (采集)** | 895 | 56.9% |
| **Combat (战斗)** | 462 | 29.4% |
| **TechTree (科技树)** | 213 | 13.5% |
| **Survival (生存)** | 2 | 0.1% |
| **总计** | 1,572 | 100% |

---

## 任务分类

### 任务命名规则

所有任务ID遵循以下命名模式：

```
<类型>_<数量>_<目标物品>_[环境]_[初始条件]
```

**示例**:
- `harvest_1_paper`: 采集1个纸张
- `harvest_8_apple_plains`: 在平原生物群系采集8个苹果
- `combat_zombie_forest_iron_armors_diamond_sword_shield`: 在森林中用钻石剑、盾牌和铁盔甲战斗僵尸

---

## Harvest 采集任务

### 概述

采集任务要求智能体收集特定数量的物品。共有**895个**采集任务，涵盖以下难度级别：

1. **基础采集** - 收集原始资源（不需要工具）
2. **手工制作** - 需要合成台制作物品
3. **熔炉冶炼** - 需要熔炉和燃料
4. **特定生物群系** - 在指定环境中采集

### 1. 基础采集任务

#### 原始资源 (Raw Resources)

| 任务ID | 中文名称 | 完成条件 | 奖励 | 难度 |
|--------|----------|----------|------|------|
| `harvest_1_milk` | 采集牛奶 | 获得1桶牛奶 | 稀疏奖励 | ⭐ |
| `harvest_1_log` | 采集木头 | 获得1个原木 | 稀疏奖励 | ⭐ |
| `harvest_1_dirt` | 采集泥土 | 获得1个泥土 | 稀疏奖励 | ⭐ |
| `harvest_1_grass` | 采集草方块 | 获得1个草方块 | 稀疏奖励 | ⭐ |
| `harvest_8_log` | 采集8个木头 | 获得8个原木 | 稀疏奖励 | ⭐⭐ |
| `harvest_8_dirt` | 采集8个泥土 | 获得8个泥土 | 稀疏奖励 | ⭐ |

**说明**:
- 这些任务不需要任何工具
- 智能体从空手开始
- 任务完成时获得单次奖励（稀疏奖励）

#### 食物与农作物

| 任务ID | 中文名称 | 资源类型 | 获取方式 |
|--------|----------|----------|----------|
| `harvest_1_apple` | 采集苹果 | 食物 | 破坏橡树树叶 |
| `harvest_1_wheat` | 采集小麦 | 农作物 | 收割成熟小麦 |
| `harvest_1_carrot` | 采集胡萝卜 | 农作物 | 收割胡萝卜 |
| `harvest_1_potato` | 采集土豆 | 农作物 | 收割土豆 |
| `harvest_1_beetroot` | 采集甜菜根 | 农作物 | 收割甜菜根 |
| `harvest_1_pumpkin` | 采集南瓜 | 农作物 | 收割南瓜 |
| `harvest_1_reeds` | 采集甘蔗 | 农作物 | 收割甘蔗 |

#### 动物掉落物

| 任务ID | 中文名称 | 来源动物 | 获取方式 |
|--------|----------|----------|----------|
| `harvest_1_beef` | 采集生牛肉 | 牛 | 击杀牛 |
| `harvest_1_porkchop` | 采集生猪肉 | 猪 | 击杀猪 |
| `harvest_1_chicken` | 采集生鸡肉 | 鸡 | 击杀鸡 |
| `harvest_1_mutton` | 采集生羊肉 | 羊 | 击杀羊 |
| `harvest_1_rabbit` | 采集生兔肉 | 兔子 | 击杀兔子 |
| `harvest_1_fish` | 采集生鱼 | 鱼 | 钓鱼或击杀鱼 |
| `harvest_1_feather` | 采集羽毛 | 鸡 | 击杀鸡 |
| `harvest_1_wool` | 采集羊毛 | 羊 | 剪羊或击杀羊 |
| `harvest_1_egg` | 采集鸡蛋 | 鸡 | 鸡自动掉落 |

#### 怪物掉落物

| 任务ID | 中文名称 | 来源怪物 | 难度 |
|--------|----------|----------|------|
| `harvest_1_bone` | 采集骨头 | 骷髅 | ⭐⭐⭐ |
| `harvest_1_string` | 采集线 | 蜘蛛 | ⭐⭐⭐ |
| `harvest_1_ender_pearl` | 采集末影珍珠 | 末影人 | ⭐⭐⭐⭐ |
| `harvest_1_slime_ball` | 采集史莱姆球 | 史莱姆 | ⭐⭐⭐⭐ |
| `harvest_1_skull` | 采集头颅 | 凋灵骷髅 | ⭐⭐⭐⭐⭐ |
| `harvest_1_totem_of_undying` | 采集不死图腾 | 卫道士 | ⭐⭐⭐⭐⭐ |

### 2. 手工制作任务

需要使用合成台制作物品。

#### 工具类

| 任务ID | 中文名称 | 所需材料 | 难度 |
|--------|----------|----------|------|
| `harvest_1_crafting_table` | 制作工作台 | 4个木板 | ⭐ |
| `harvest_1_stick` | 制作木棍 | 2个木板 | ⭐ |
| `harvest_1_torch` | 制作火把 | 1木棍+1煤炭 | ⭐⭐ |
| `harvest_1_shears` | 制作剪刀 | 2个铁锭 | ⭐⭐⭐ |
| `harvest_1_flint_and_steel` | 制作打火石 | 1铁锭+1燧石 | ⭐⭐⭐ |
| `harvest_1_bucket` | 制作桶 | 3个铁锭 | ⭐⭐⭐ |
| `harvest_1_fishing_rod` | 制作钓鱼竿 | 3木棍+2线 | ⭐⭐⭐ |

#### 建筑类

| 任务ID | 中文名称 | 所需材料 | 用途 |
|--------|----------|----------|------|
| `harvest_1_chest` | 制作箱子 | 8个木板 | 存储 |
| `harvest_1_furnace` | 制作熔炉 | 8个圆石 | 冶炼 |
| `harvest_1_ladder` | 制作梯子 | 7个木棍 | 攀爬 |
| `harvest_1_fence` | 制作栅栏 | 木棍+木板 | 围栏 |
| `harvest_1_fence_gate` | 制作栅栏门 | 木棍+木板 | 大门 |
| `harvest_1_trapdoor` | 制作活板门 | 6个木板 | 机关 |
| `harvest_1_bed` | 制作床 | 3羊毛+3木板 | 重生点 |

#### 食物类

| 任务ID | 中文名称 | 所需材料 | 恢复饥饿度 |
|--------|----------|----------|-----------|
| `harvest_1_bowl` | 制作碗 | 3个木板 | - |
| `harvest_1_bread` | 制作面包 | 3个小麦 | 5 |
| `harvest_1_cookie` | 制作曲奇 | 2小麦+1可可豆 | 2 |
| `harvest_1_cake` | 制作蛋糕 | 糖+鸡蛋+小麦+牛奶 | 14 |
| `harvest_1_mushroom_stew` | 制作蘑菇煲 | 碗+蘑菇 | 6 |
| `harvest_1_rabbit_stew` | 制作兔肉煲 | 兔肉+蔬菜 | 10 |
| `harvest_1_golden_apple` | 制作金苹果 | 苹果+8金锭 | 4+特效 |

#### 带工作台版本

这些任务智能体初始拥有一个工作台：

| 任务ID模式 | 说明 |
|-----------|------|
| `harvest_1_<item>_with_crafting_table` | 初始拥有工作台 |
| `harvest_8_<item>_with_crafting_table` | 初始拥有工作台，需8个 |

**示例**:
- `harvest_1_arrow_with_crafting_table`: 初始有工作台，制作1个箭
- `harvest_1_chest_with_crafting_table`: 初始有工作台，制作1个箱子

### 3. 熔炉冶炼任务

需要使用熔炉和燃料。

#### 食物烹饪

| 任务ID | 中文名称 | 原材料 | 烹饪时间 |
|--------|----------|--------|----------|
| `harvest_1_cooked_beef` | 烹饪熟牛肉 | 生牛肉 | 10秒 |
| `harvest_1_cooked_porkchop` | 烹饪熟猪肉 | 生猪肉 | 10秒 |
| `harvest_1_cooked_chicken` | 烹饪熟鸡肉 | 生鸡肉 | 10秒 |
| `harvest_1_cooked_mutton` | 烹饪熟羊肉 | 生羊肉 | 10秒 |
| `harvest_1_cooked_rabbit` | 烹饪熟兔肉 | 生兔肉 | 10秒 |
| `harvest_1_cooked_fish` | 烹饪熟鱼 | 生鱼 | 10秒 |
| `harvest_1_baked_potato` | 烤土豆 | 土豆 | 10秒 |

#### 矿物冶炼

| 任务ID | 中文名称 | 原材料 | 产出 |
|--------|----------|--------|------|
| `harvest_1_iron_ingot` | 冶炼铁锭 | 铁矿石 | 1个铁锭 |
| `harvest_1_gold_ingot` | 冶炼金锭 | 金矿石 | 1个金锭 |
| `harvest_1_glass` | 烧制玻璃 | 沙子 | 1个玻璃 |
| `harvest_1_stone` | 烧制石头 | 圆石 | 1个石头 |
| `harvest_1_brick` | 烧制砖 | 粘土球 | 1个砖 |
| `harvest_1_quartz` | 冶炼石英 | 下界石英矿 | 1个石英 |
| `harvest_1_emerald` | 获取绿宝石 | 绿宝石矿 | 1个绿宝石 |
| `harvest_1_netherbrick` | 烧制下界砖 | 下界岩 | 1个下界砖 |

#### 带熔炉版本

这些任务智能体初始拥有熔炉和燃料：

| 任务ID模式 | 说明 |
|-----------|------|
| `harvest_1_<item>_with_furnace_and_fuel` | 初始拥有熔炉和燃料 |
| `harvest_8_<item>_with_furnace_and_fuel` | 初始拥有熔炉和燃料，需8个 |

### 4. 生物群系特定任务

在特定的生物群系中完成采集任务。

#### 生物群系类型

| 生物群系 | 英文ID | 特征 |
|---------|--------|------|
| 平原 | `plains` | 开阔平坦，动物多 |
| 森林 | `forest` | 树木茂密，木材丰富 |
| 针叶林 | `taiga` | 云杉为主，狼群出没 |
| 丛林 | `jungle` | 热带雨林，可可豆 |
| 沼泽 | `swampland` | 水域多，史莱姆 |
| 沙漠 | `desert` | 无树木，仙人掌 |
| 海洋 | `ocean` | 深水，鱼类丰富 |
| 山地 | `extreme_hills` | 高海拔，资源丰富 |

#### 任务示例

| 任务ID | 中文名称 | 说明 |
|--------|----------|------|
| `harvest_1_log_plains` | 平原采集原木 | 在平原生物群系采集1个原木 |
| `harvest_8_apple_forest` | 森林采集苹果 | 在森林生物群系采集8个苹果 |
| `harvest_1_wheat_jungle` | 丛林采集小麦 | 在丛林生物群系采集1个小麦 |
| `harvest_1_beef_swampland` | 沼泽采集牛肉 | 在沼泽生物群系采集1个牛肉 |
| `harvest_1_iron_ingot_taiga` | 针叶林冶炼铁锭 | 在针叶林生物群系冶炼1个铁锭 |

---

## Combat 战斗任务

### 概述

战斗任务要求智能体击败特定的生物。共有**462个**战斗任务，涵盖不同的：
- 目标生物（敌对/中立/被动）
- 生物群系
- 装备等级（皮革/铁/钻石盔甲，木/铁/钻石剑）

### 1. 敌对生物战斗

#### 常见敌对生物

| 任务ID模式 | 中文名称 | 生物特性 | 基础难度 |
|-----------|----------|----------|----------|
| `combat_zombie_*` | 僵尸战斗 | 近战，慢速 | ⭐⭐ |
| `combat_skeleton_*` | 骷髅战斗 | 远程弓箭 | ⭐⭐⭐ |
| `combat_spider_*` | 蜘蛛战斗 | 快速爬墙 | ⭐⭐ |
| `combat_creeper_*` | 爬行者战斗 | 会爆炸！ | ⭐⭐⭐⭐ |
| `combat_witch_*` | 女巫战斗 | 投掷药水 | ⭐⭐⭐⭐ |
| `combat_enderman_*` | 末影人战斗 | 瞬移攻击 | ⭐⭐⭐⭐ |

#### 高级敌对生物

| 任务ID模式 | 中文名称 | 位置 | 难度 |
|-----------|----------|------|------|
| `combat_blaze_nether_*` | 烈焰人战斗 | 下界 | ⭐⭐⭐⭐⭐ |
| `combat_wither_skeleton_nether_*` | 凋灵骷髅战斗 | 下界要塞 | ⭐⭐⭐⭐⭐ |
| `combat_shulker_end_*` | 潜影贝战斗 | 末地 | ⭐⭐⭐⭐⭐ |
| `combat_guardian_ocean_*` | 守卫者战斗 | 海底神殿 | ⭐⭐⭐⭐⭐ |

### 2. 中立/被动生物战斗

#### 被动动物

| 任务ID模式 | 中文名称 | 说明 |
|-----------|----------|------|
| `combat_cow_*` | 牛战斗 | 被动，容易击败 |
| `combat_pig_*` | 猪战斗 | 被动，容易击败 |
| `combat_sheep_*` | 羊战斗 | 被动，容易击败 |
| `combat_chicken_*` | 鸡战斗 | 被动，快速移动 |
| `combat_horse_*` | 马战斗 | 中立，快速 |
| `combat_rabbit_*` | 兔子战斗 | 被动，非常快 |

#### 特殊生物

| 任务ID模式 | 中文名称 | 特殊性 |
|-----------|----------|--------|
| `combat_wolf_taiga_*` | 狼战斗 | 中立，群体攻击 |
| `combat_llama_extreme_hills_*` | 羊驼战斗 | 中立，会吐口水 |
| `combat_squid_ocean_*` | 鱿鱼战斗 | 水下，会逃跑 |

### 3. 装备组合

战斗任务通过装备组合来调整难度：

#### 盔甲等级

| 盔甲类型 | ID标识 | 防御值 | 说明 |
|---------|--------|--------|------|
| 皮革盔甲 | `leather_armors` | 28% | 最低防护 |
| 铁盔甲 | `iron_armors` | 60% | 中等防护 |
| 钻石盔甲 | `diamond_armors` | 80% | 最高防护 |

#### 武器等级

| 武器类型 | ID标识 | 伤害 | 耐久 |
|---------|--------|------|------|
| 木剑 | `wooden_sword` | 4 | 60 |
| 铁剑 | `iron_sword` | 6 | 251 |
| 钻石剑 | `diamond_sword` | 7 | 1562 |

#### 副手装备

| 装备 | 说明 |
|------|------|
| `shield` | 盾牌，可格挡攻击 |

### 4. 任务命名示例

```
combat_<生物>_<生物群系>_<盔甲>_<武器>_<副手>
```

**示例**:
- `combat_zombie_forest_leather_armors_wooden_sword_shield`
  - 在森林中
  - 穿着全套皮革盔甲
  - 持木剑和盾牌
  - 战斗僵尸

- `combat_enderman_plains_diamond_armors_diamond_sword_shield`
  - 在平原中
  - 穿着全套钻石盔甲
  - 持钻石剑和盾牌
  - 战斗末影人

### 5. 空手战斗任务

部分任务要求空手战斗被动生物：

| 任务ID | 中文名称 | 难度 |
|--------|----------|------|
| `combat_cow_forest_barehand` | 空手打牛（森林） | ⭐ |
| `combat_pig_plains_barehand` | 空手打猪（平原） | ⭐ |
| `combat_sheep_extreme_hills_barehand` | 空手打羊（山地） | ⭐ |
| `combat_bat_forest_barehand` | 空手打蝙蝠（森林） | ⭐⭐ |

---

## TechTree 科技树任务

### 概述

科技树任务模拟Minecraft的进度系统，要求智能体从基础资源逐步发展到高级装备。共有**213个**科技树任务。

### 1. 起点类型

科技树任务有不同的起点：

| 起点类型 | ID标识 | 说明 |
|---------|--------|------|
| 徒手开始 | `from_barehand_to_*` | 完全空手，最困难 |
| 有木材 | `from_wood_to_*` | 初始拥有木材 |
| 有石器 | `from_stone_to_*` | 初始拥有石器工具 |
| 有铁器 | `from_iron_to_*` | 初始拥有铁器工具 |
| 有金器 | `from_gold_to_*` | 初始拥有金器工具 |
| 有钻石 | `from_diamond_to_*` | 初始拥有钻石工具 |

### 2. 工具科技树

#### 木制工具（最基础）

| 任务ID | 中文名称 | 制作路径 |
|--------|----------|----------|
| `techtree_from_barehand_to_wooden_sword` | 徒手→木剑 | 采集木头→木板→木棍→木剑 |
| `techtree_from_barehand_to_wooden_pickaxe` | 徒手→木镐 | 采集木头→木板→木棍→木镐 |
| `techtree_from_barehand_to_wooden_axe` | 徒手→木斧 | 采集木头→木板→木棍→木斧 |
| `techtree_from_barehand_to_wooden_hoe` | 徒手→木锄 | 采集木头→木板→木棍→木锄 |
| `techtree_from_barehand_to_wooden_shovel` | 徒手→木铲 | 采集木头→木板→木棍→木铲 |

#### 石制工具

| 任务ID | 中文名称 | 制作路径 |
|--------|----------|----------|
| `techtree_from_barehand_to_stone_sword` | 徒手→石剑 | 木工具→挖圆石→石剑 |
| `techtree_from_barehand_to_stone_pickaxe` | 徒手→石镐 | 木工具→挖圆石→石镐 |
| `techtree_from_wood_to_stone_sword` | 木材→石剑 | 木板→木镐→圆石→石剑 |

#### 铁制工具

| 任务ID | 中文名称 | 制作路径 |
|--------|----------|----------|
| `techtree_from_barehand_to_iron_sword` | 徒手→铁剑 | 木→石→铁矿→熔炉→铁锭→铁剑 |
| `techtree_from_barehand_to_iron_pickaxe` | 徒手→铁镐 | 木→石→铁矿→熔炉→铁锭→铁镐 |
| `techtree_from_stone_to_iron_sword` | 石器→铁剑 | 铁矿→熔炉→铁锭→铁剑 |

#### 钻石工具（最高级）

| 任务ID | 中文名称 | 制作路径 |
|--------|----------|----------|
| `techtree_from_barehand_to_diamond_sword` | 徒手→钻石剑 | 完整发展链 |
| `techtree_from_barehand_to_diamond_pickaxe` | 徒手→钻石镐 | 完整发展链 |
| `techtree_from_iron_to_diamond_sword` | 铁器→钻石剑 | 挖钻石→制作钻石剑 |

### 3. 盔甲科技树

#### 皮革盔甲

| 任务ID | 中文名称 | 所需材料 |
|--------|----------|----------|
| `techtree_from_barehand_to_leather_helmet` | 徒手→皮革头盔 | 5个皮革 |
| `techtree_from_barehand_to_leather_chestplate` | 徒手→皮革胸甲 | 8个皮革 |
| `techtree_from_barehand_to_leather_leggings` | 徒手→皮革护腿 | 7个皮革 |
| `techtree_from_barehand_to_leather_boots` | 徒手→皮革靴子 | 4个皮革 |

#### 铁盔甲

| 任务ID | 中文名称 | 所需材料 |
|--------|----------|----------|
| `techtree_from_barehand_to_iron_helmet` | 徒手→铁头盔 | 5个铁锭 |
| `techtree_from_barehand_to_iron_chestplate` | 徒手→铁胸甲 | 8个铁锭 |
| `techtree_from_barehand_to_iron_leggings` | 徒手→铁护腿 | 7个铁锭 |
| `techtree_from_barehand_to_iron_boots` | 徒手→铁靴子 | 4个铁锭 |

#### 钻石盔甲

| 任务ID | 中文名称 | 所需材料 |
|--------|----------|----------|
| `techtree_from_barehand_to_diamond_helmet` | 徒手→钻石头盔 | 5个钻石 |
| `techtree_from_barehand_to_diamond_chestplate` | 徒手→钻石胸甲 | 8个钻石 |
| `techtree_from_barehand_to_diamond_leggings` | 徒手→钻石护腿 | 7个钻石 |
| `techtree_from_barehand_to_diamond_boots` | 徒手→钻石靴子 | 4个钻石 |

### 4. 特殊科技树

#### 弓箭科技

| 任务ID | 中文名称 | 所需材料 |
|--------|----------|----------|
| `techtree_from_barehand_to_archery` | 徒手→弓箭术 | 木棍+线→弓，燧石+木棍+羽毛→箭 |

#### 爆炸物科技

| 任务ID | 中文名称 | 所需材料 |
|--------|----------|----------|
| `techtree_from_barehand_to_explosives` | 徒手→爆炸物 | 火药+沙子→TNT |

#### 红石科技

| 任务ID前缀 | 中文名称 | 类别 |
|-----------|----------|------|
| `techtree_*_to_redstone_redstone_block` | →红石块 | 基础 |
| `techtree_*_to_redstone_torch` | →红石火把 | 基础 |
| `techtree_*_to_redstone_repeater` | →红石中继器 | 逻辑 |
| `techtree_*_to_redstone_comparator` | →红石比较器 | 逻辑 |
| `techtree_*_to_redstone_piston` | →活塞 | 机械 |
| `techtree_*_to_redstone_dispenser` | →发射器 | 机械 |
| `techtree_*_to_redstone_dropper` | →投掷器 | 机械 |
| `techtree_*_to_redstone_observer` | →侦测器 | 传感 |

---

## Survival 生存任务

### 概述

生存任务考察智能体的综合生存能力。共有**2个**生存任务。

### 任务详情

| 任务ID | 中文名称 | 完成条件 | 难度 |
|--------|----------|----------|------|
| `survival` | 基础生存 | 在Minecraft世界中存活尽可能长时间 | ⭐⭐⭐⭐⭐ |
| `survival_sword_food` | 携带剑和食物生存 | 初始拥有铁剑和食物，尽可能长时间生存 | ⭐⭐⭐⭐ |

### 生存要素

智能体需要管理以下生存要素：

| 要素 | 说明 | 重要性 |
|------|------|--------|
| **生命值** (Health) | 避免受伤和死亡 | ⭐⭐⭐⭐⭐ |
| **饥饿值** (Hunger) | 定期进食保持饥饿值 | ⭐⭐⭐⭐⭐ |
| **氧气值** (Oxygen) | 水下需要管理氧气 | ⭐⭐⭐ |
| **护甲值** (Armor) | 装备盔甲减少伤害 | ⭐⭐⭐⭐ |
| **经验值** (XP) | 击杀怪物获得经验 | ⭐⭐ |

---

## 奖励机制

### 1. 奖励类型

MineDojo任务支持两种奖励类型：

#### 稀疏奖励 (Sparse Reward)

- **特点**: 只在任务完成时给予奖励
- **奖励值**: +1.0（成功），0（未完成）
- **适用任务**: 大部分harvest和combat任务
- **训练难度**: 较高，需要探索

**示例**:
```python
# harvest_1_paper任务
# 只有当智能体获得1个纸张时，才获得+1奖励
reward = 1.0 if has_paper else 0.0
```

#### 密集奖励 (Dense Reward)

- **特点**: 根据进度给予部分奖励
- **奖励值**: 0~1.0之间的连续值
- **适用任务**: 部分复杂任务
- **训练难度**: 较低，容易学习

**示例**:
```python
# 复杂科技树任务
# 完成中间步骤也会获得部分奖励
reward = progress / total_steps
```

### 2. 任务成功判定

不同类型任务的成功判定：

| 任务类型 | 成功条件 | 检测方式 |
|---------|---------|---------|
| Harvest | 物品栏中拥有目标物品 | 检查inventory |
| Combat | 目标生物死亡 | 检查entity状态 |
| TechTree | 拥有目标装备 | 检查inventory和equipment |
| Survival | 存活时间 | 持续步数 |

### 3. 奖励计算示例

#### Harvest任务

```python
# harvest_8_apple - 采集8个苹果
def compute_reward(inventory):
    apple_count = count_item(inventory, 'apple')
    if apple_count >= 8:
        return 1.0  # 任务完成
    else:
        return 0.0  # 未完成
```

#### Combat任务

```python
# combat_zombie_forest_* - 击败僵尸
def compute_reward(world_state):
    if target_zombie_killed(world_state):
        return 1.0  # 任务完成
    elif agent_died(world_state):
        return 0.0  # 智能体死亡，任务失败
    else:
        return 0.0  # 进行中
```

#### TechTree任务

```python
# techtree_from_barehand_to_iron_sword - 制作铁剑
def compute_reward(inventory, equipment):
    has_iron_sword = check_item(inventory, 'iron_sword') or \
                     check_item(equipment, 'iron_sword')
    if has_iron_sword:
        return 1.0  # 任务完成
    else:
        return 0.0  # 未完成
```

---

## 使用示例

### 1. 基础使用

```python
import minedojo

# 创建一个简单的采集任务
env = minedojo.make(
    task_id="harvest_1_milk",
    image_size=(160, 256)
)

obs = env.reset()
done = False

while not done:
    # 智能体选择动作
    action = agent.get_action(obs)
    
    # 执行动作
    obs, reward, done, info = env.step(action)
    
    if reward > 0:
        print("任务完成！获得牛奶")

env.close()
```

### 2. 战斗任务示例

```python
import minedojo

# 创建一个战斗任务
env = minedojo.make(
    task_id="combat_zombie_forest_iron_armors_diamond_sword_shield",
    image_size=(160, 256)
)

obs = env.reset()
# 初始状态：智能体在森林中，穿着铁盔甲，持钻石剑和盾牌

done = False
while not done:
    action = agent.get_action(obs)
    obs, reward, done, info = env.step(action)
    
    if reward > 0:
        print("成功击败僵尸！")

env.close()
```

### 3. 科技树任务示例

```python
import minedojo

# 创建一个科技树任务
env = minedojo.make(
    task_id="techtree_from_barehand_to_iron_pickaxe",
    image_size=(160, 256)
)

obs = env.reset()
# 初始状态：完全空手

# 智能体需要：
# 1. 采集木头
# 2. 制作木板和木棍
# 3. 制作工作台
# 4. 制作木镐
# 5. 挖掘圆石
# 6. 制作熔炉
# 7. 挖掘铁矿
# 8. 冶炼铁锭
# 9. 制作铁镐

done = False
steps = 0
while not done and steps < 10000:
    action = agent.get_action(obs)
    obs, reward, done, info = env.step(action)
    steps += 1
    
    if reward > 0:
        print(f"成功制作铁镐！用时{steps}步")

env.close()
```

### 4. 生存任务示例

```python
import minedojo

# 创建生存任务
env = minedojo.make(
    task_id="survival",
    image_size=(160, 256)
)

obs = env.reset()

survival_time = 0
done = False

while not done:
    action = agent.get_action(obs)
    obs, reward, done, info = env.step(action)
    survival_time += 1
    
    # 监控生存状态
    health = obs['life_stats']['life'][0]
    food = obs['life_stats']['food'][0]
    
    if health < 5:
        print(f"警告：生命值过低！当前：{health}")
    if food < 5:
        print(f"警告：饥饿值过低！当前：{food}")

print(f"生存了{survival_time}步")
env.close()
```

### 5. 批量测试任务

```python
import minedojo

# 测试多个harvest任务
harvest_tasks = [
    "harvest_1_milk",
    "harvest_1_log",
    "harvest_1_paper",
    "harvest_1_apple",
]

results = {}

for task_id in harvest_tasks:
    env = minedojo.make(task_id=task_id, image_size=(160, 256))
    obs = env.reset()
    
    done = False
    success = False
    steps = 0
    
    while not done and steps < 5000:
        action = agent.get_action(obs)
        obs, reward, done, info = env.step(action)
        steps += 1
        
        if reward > 0:
            success = True
            break
    
    results[task_id] = {
        'success': success,
        'steps': steps
    }
    
    env.close()

# 打印结果
for task_id, result in results.items():
    status = "✓" if result['success'] else "✗"
    print(f"{status} {task_id}: {result['steps']}步")
```

---

## 附录

### A. 任务ID快速查询

#### 常用Harvest任务

```
harvest_1_milk          # 采集牛奶（最简单）
harvest_1_log           # 采集木头
harvest_1_apple         # 采集苹果
harvest_1_wheat         # 采集小麦
harvest_1_paper         # 制作纸张
harvest_1_crafting_table # 制作工作台
harvest_1_torch         # 制作火把
harvest_1_iron_ingot    # 冶炼铁锭
harvest_1_cooked_beef   # 烹饪熟牛肉
```

#### 常用Combat任务

```
combat_cow_forest_barehand                          # 最简单
combat_zombie_forest_leather_armors_wooden_sword_shield  # 初级战斗
combat_spider_plains_iron_armors_iron_sword_shield      # 中级战斗
combat_enderman_plains_diamond_armors_diamond_sword_shield # 高级战斗
```

#### 常用TechTree任务

```
techtree_from_barehand_to_wooden_pickaxe  # 最基础
techtree_from_barehand_to_stone_sword     # 进阶
techtree_from_barehand_to_iron_sword      # 中级
techtree_from_barehand_to_diamond_pickaxe # 高级
```

### B. 难度评级说明

| 等级 | 符号 | 说明 | 预计步数 |
|------|------|------|---------|
| 非常简单 | ⭐ | 基础操作，容易完成 | < 100 |
| 简单 | ⭐⭐ | 需要基本策略 | 100-500 |
| 中等 | ⭐⭐⭐ | 需要多步规划 | 500-2000 |
| 困难 | ⭐⭐⭐⭐ | 需要复杂策略 | 2000-10000 |
| 非常困难 | ⭐⭐⭐⭐⭐ | 需要高级技能 | > 10000 |

### C. 观察空间说明

MineDojo提供丰富的观察空间：

| 观察项 | 类型 | 说明 |
|--------|------|------|
| `rgb` | 图像 | 第一人称视角画面 |
| `inventory` | 字典 | 背包物品信息 |
| `equipment` | 字典 | 装备信息 |
| `life_stats` | 字典 | 生命、饥饿、护甲值等 |
| `location_stats` | 字典 | 位置、生物群系、光照等 |
| `voxels` | 数组 | 周围方块信息 |

### D. 动作空间说明

MineDojo使用MultiDiscrete动作空间：

| 动作维度 | 取值范围 | 说明 |
|---------|---------|------|
| 前进/后退 | [0, 1, 2] | 0=不动，1=前进，2=后退 |
| 左移/右移 | [0, 1, 2] | 0=不动，1=左移，2=右移 |
| 跳跃/潜行 | [0, 1, 2, 3] | 跳跃、潜行、冲刺 |
| 相机俯仰 | [0-24] | 上下看 |
| 相机偏航 | [0-24] | 左右看 |
| 功能按键 | [0-7] | 攻击、使用、放置等 |
| 合成/冶炼 | [0-243] | 选择合成配方 |
| 物品选择 | [0-35] | 选择物品栏槽位 |

---

## 参考资料

- **MineDojo官方网站**: https://minedojo.org
- **论文**: MineDojo: Building Open-Ended Embodied Agents with Internet-Scale Knowledge
- **GitHub**: https://github.com/MineDojo/MineDojo
- **文档**: https://docs.minedojo.org

---

**文档版本**: 1.0  
**生成时间**: 2025-10-20  
**维护者**: AIMC项目团队


