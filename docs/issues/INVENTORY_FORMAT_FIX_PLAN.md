# Inventory 格式修复方案

> **状态**: 待修复（训练完成后实施）  
> **优先级**: P0 - 严重Bug  
> **影响范围**: harvest_X_log 任务无法正确识别部分木头类型  
> **发现时间**: 2025-10-23

---

## 📋 问题描述

### 症状
- 金合欢木（Acacia Log，沙漠橡木）获得后不给奖励
- 深色橡木（Dark Oak Log，黑色橡木）获得后不给奖励
- 其他4种木头（Oak, Spruce, Birch, Jungle）未测试，可能也有问题

### 实际库存数据
```python
[DEBUG] 库存格式: list, 原木物品: [
    {
        'name': 'log2',           # 物品名称
        'variant': 0,             # 变种ID（0=Acacia, 1=Dark Oak）
        'quantity': 1,            # 数量
        'max_durability': -1,
        'cur_durability': -1,
        'index': 0,
        'inventory': 'inventory'
    }
]
```

---

## 🔍 根本原因分析

### 问题1: 数据结构类型错误 ⚠️ 最致命
**代码假设:**
```python
inventory = {
    "oak_log": 5,
    "acacia_log": 1
}
```

**实际情况:**
```python
inventory = [
    {'name': 'log2', 'variant': 0, 'quantity': 1}
]
```

**后果:**
```python
# 当前代码逻辑
for item_id in ["acacia_log", "minecraft:acacia_log"]:
    if item_id in inventory:  # ❌ False! "acacia_log" 不在列表中
        return inventory[item_id]
return 0  # 永远返回0
```

---

### 问题2: 物品名称不匹配
**代码查找:**
- `"oak_log"`, `"birch_log"`, `"acacia_log"` 等

**MineDojo实际使用:**
- `"log"` (方块ID 17)
- `"log2"` (方块ID 162)

**映射关系:**
```
log  + variant:0 → Oak Log (橡木)
log  + variant:1 → Spruce Log (云杉木)
log  + variant:2 → Birch Log (白桦木)
log  + variant:3 → Jungle Log (丛林木)
log2 + variant:0 → Acacia Log (金合欢木/沙漠橡木)
log2 + variant:1 → Dark Oak Log (深色橡木/黑色橡木)
```

---

### 问题3: 缺少 variant 字段检查
即使修复了名称匹配，仍需要检查 `variant` 字段来区分同一 `name` 下的不同木头类型。

例如：
- `log2/variant:0` = 金合欢木
- `log2/variant:1` = 深色橡木

---

## ✅ 修复方案

### 文件: `src/envs/task_wrappers.py`

#### 修改1: 更新 `self.log_types` 定义

**当前代码 (第63-72行):**
```python
self.log_types = [
    "oak_log",       # 橡木（最常见）
    "birch_log",     # 白桦木
    "spruce_log",    # 云杉木
    "dark_oak_log",  # 深色橡木（用户报告的"黑色木头"）
    "jungle_log",    # 丛林木（稀有）
    "acacia_log"     # 金合欢木（稀有）
]
```

**修改为:**
```python
# MineDojo 使用 (name, variant) 格式来区分不同木头类型
# 每个元素可以是：
#   - 元组 (name, variant): MineDojo实际格式，如 ('log2', 0)
#   - 字符串 name: 兼容其他可能的格式，如 'oak_log'
self.log_types = [
    # MineDojo 实际格式 (name, variant)
    ("log", 0),   # Oak Log (橡木)
    ("log", 1),   # Spruce Log (云杉木)
    ("log", 2),   # Birch Log (白桦木)
    ("log", 3),   # Jungle Log (丛林木)
    ("log2", 0),  # Acacia Log (金合欢木/沙漠橡木)
    ("log2", 1),  # Dark Oak Log (深色橡木/黑色橡木)
    
    # 兼容其他可能的格式（字符串）
    "oak_log",
    "birch_log",
    "spruce_log",
    "dark_oak_log",
    "jungle_log",
    "acacia_log",
    "minecraft:oak_log",
    "minecraft:birch_log",
    "minecraft:spruce_log",
    "minecraft:dark_oak_log",
    "minecraft:jungle_log",
    "minecraft:acacia_log"
]
```

---

#### 修改2: 重写 `_get_item_count` 方法

**当前代码 (第141-152行):**
```python
def _get_item_count(self, inventory, item_name):
    """
    从库存中获取物品数量
    
    支持多种物品ID格式:
    - "oak_log"
    - "minecraft:oak_log"
    
    Args:
        inventory: 库存字典
        item_name: 物品名称（不含minecraft:前缀）
    
    Returns:
        int: 物品数量
    """
    # 尝试多种可能的物品ID格式
    for item_id in [item_name, f"minecraft:{item_name}"]:
        if item_id in inventory:
            return inventory[item_id]
    return 0
```

**修改为:**
```python
def _get_item_count(self, inventory, item_name):
    """
    从库存中获取物品数量
    
    支持多种库存格式:
    1. 字典格式: {"oak_log": 5, "minecraft:stone": 10}
    2. 列表格式: [{"name": "log2", "variant": 0, "quantity": 1}, ...]
    
    支持多种物品查找格式:
    - 元组 (name, variant): 如 ("log2", 0) 表示 Acacia Log
    - 字符串 name: 如 "oak_log" 或 "minecraft:oak_log"
    
    Args:
        inventory: 库存（字典或列表）
        item_name: 物品标识（字符串或元组）
    
    Returns:
        int: 物品数量
    """
    # ========== 处理字典格式 ==========
    if isinstance(inventory, dict):
        # 只支持字符串查找（元组格式不适用于dict）
        if isinstance(item_name, str):
            for item_id in [item_name, f"minecraft:{item_name}"]:
                if item_id in inventory:
                    return inventory[item_id]
        return 0
    
    # ========== 处理列表格式 ==========
    elif isinstance(inventory, list):
        total_count = 0
        
        for item in inventory:
            if not isinstance(item, dict):
                continue
            
            # 获取物品属性
            item_name_in_inv = item.get('name', '') or item.get('type', '') or item.get('item', '')
            item_variant = item.get('variant', -1)
            quantity = item.get('quantity', 0) or item.get('count', 0) or 1
            
            # --- 情况1: 元组格式匹配 (name, variant) ---
            if isinstance(item_name, tuple) and len(item_name) == 2:
                target_name, target_variant = item_name
                
                # 检查 name 和 variant 是否都匹配
                if item_name_in_inv == target_name and item_variant == target_variant:
                    total_count += quantity
            
            # --- 情况2: 字符串格式匹配 ---
            elif isinstance(item_name, str):
                # 尝试多种可能的物品ID格式
                for item_id in [item_name, f"minecraft:{item_name}"]:
                    if item_name_in_inv == item_id:
                        total_count += quantity
                        break
        
        return total_count
    
    # 未知格式
    return 0
```

---

#### 修改3: 更新调试输出（可选）

**当前代码 (第103-106行):**
```python
# 调试：打印所有包含"log"的物品
if self.verbose and self.last_log_count == 0:
    log_items = {k: v for k, v in inventory.items() if 'log' in k.lower()}
    if log_items:
        print(f"  [DEBUG] 库存中的原木物品: {log_items}")
```

**修改为:**
```python
# 调试：打印所有包含"log"的物品
if self.verbose and self.last_log_count == 0:
    if isinstance(inventory, dict):
        log_items = {k: v for k, v in inventory.items() if 'log' in k.lower()}
        if log_items:
            print(f"  [DEBUG] 库存格式: dict, 原木物品: {log_items}")
    elif isinstance(inventory, list):
        log_items = [item for item in inventory if isinstance(item, dict) and 'log' in str(item.get('name', '')).lower()]
        if log_items:
            print(f"  [DEBUG] 库存格式: list, 原木物品:")
            for log_item in log_items:
                name = log_item.get('name', 'unknown')
                variant = log_item.get('variant', -1)
                quantity = log_item.get('quantity', 0)
                print(f"    - {name}/variant:{variant} x{quantity}")
```

---

## 🧪 测试计划

### 测试用例1: 金合欢木（Acacia Log）
```bash
# 在沙漠生物群系测试
bash scripts/run_dagger_workflow.sh --task harvest_1_log --num-episodes 1
```

**预期结果:**
```
[DEBUG] 库存格式: list, 原木物品:
  - log2/variant:0 x1
✓ 获得原木！总数: 1 | 类型: log2/variant:0(1)
  任务成功！(需要1个)
Reward: 1.000 | Done: True
```

---

### 测试用例2: 深色橡木（Dark Oak Log）
```bash
# 在黑森林生物群系测试
bash scripts/run_dagger_workflow.sh --task harvest_1_log --num-episodes 1
```

**预期结果:**
```
[DEBUG] 库存格式: list, 原木物品:
  - log2/variant:1 x1
✓ 获得原木！总数: 1 | 类型: log2/variant:1(1)
  任务成功！(需要1个)
Reward: 1.000 | Done: True
```

---

### 测试用例3: 普通橡木（Oak Log）
```bash
# 在普通森林测试
bash scripts/run_dagger_workflow.sh --task harvest_1_log --num-episodes 1
```

**预期结果:**
```
[DEBUG] 库存格式: list, 原木物品:
  - log/variant:0 x1
✓ 获得原木！总数: 1 | 类型: log/variant:0(1)
  任务成功！(需要1个)
Reward: 1.000 | Done: True
```

---

### 测试用例4: 混合木头类型
```bash
# 获得多种木头后测试
bash scripts/run_dagger_workflow.sh --task harvest_8_log --num-episodes 1
```

**预期结果:**
```
✓ 获得原木！总数: 8 | 类型: log/variant:0(3), log2/variant:0(2), log/variant:2(3)
  任务成功！(需要8个)
```

---

## 📝 实施步骤

1. **备份当前代码**
   ```bash
   git add -A
   git commit -m "[checkpoint] 修复inventory格式前的备份"
   ```

2. **应用修改1: 更新 `self.log_types`**
   - 文件: `src/envs/task_wrappers.py`
   - 行数: 第63-72行

3. **应用修改2: 重写 `_get_item_count`**
   - 文件: `src/envs/task_wrappers.py`
   - 行数: 第141-152行

4. **应用修改3: 更新调试输出（可选）**
   - 文件: `src/envs/task_wrappers.py`
   - 行数: 第103-106行

5. **运行测试**
   ```bash
   # 测试金合欢木
   bash scripts/run_dagger_workflow.sh --task harvest_1_log --num-episodes 1
   
   # 观察调试输出和奖励
   ```

6. **提交修复**
   ```bash
   git add src/envs/task_wrappers.py
   git commit -m "[fix] 修复inventory列表格式，支持所有6种原木类型
   
   问题:
   - 金合欢木/深色橡木获得后不给奖励
   - inventory是list[dict]而非dict
   - 物品名使用log/log2+variant而非oak_log等
   
   解决方案:
   1. 更新log_types支持(name, variant)元组格式
   2. 重写_get_item_count支持list格式
   3. 同时兼容dict格式和字符串格式
   
   测试:
   - ✅ log2/variant:0 (Acacia Log)
   - ✅ log2/variant:1 (Dark Oak Log)
   - ✅ log/variant:0-3 (Oak/Spruce/Birch/Jungle)"
   ```

---

## 🎯 预期影响

### 修复后的效果
- ✅ 所有6种原木类型都能正确识别
- ✅ 支持 MineDojo 的实际数据格式
- ✅ 向后兼容其他可能的格式（dict, 字符串）
- ✅ 调试信息更详细，便于排查问题

### 性能影响
- 无明显性能影响（库存列表通常很短）
- 增加了类型检查和字段查找，但复杂度仍为 O(n)

---

## 📚 相关文档

- [Minecraft Wiki - Wood](https://minecraft.fandom.com/wiki/Wood)
- [MineDojo Documentation](https://docs.minedojo.org/)
- `docs/technical/HARVEST_LOG_TASK_ANALYSIS.md` - harvest_log任务分析
- `docs/technical/TASK_WRAPPERS_GUIDE.md` - 任务Wrapper架构

---

## 🔗 相关Issue

- 用户报告: "黑色木头不给奖励" (Dark Oak Log)
- 用户报告: "沙漠橡木不给奖励" (Acacia Log)
- 根本原因: MineDojo使用 `log2 + variant` 格式

---

## ✅ 验收标准

修复成功的标志：
1. 在沙漠生物群系获得金合欢木后，立即获得奖励和 `done=True`
2. 在黑森林生物群系获得深色橡木后，立即获得奖励和 `done=True`
3. 调试输出显示正确的物品格式: `log2/variant:0` 或 `log2/variant:1`
4. 所有现有测试仍然通过（向后兼容）

---

**最后更新**: 2025-10-23  
**下次审查**: 训练完成后立即实施

