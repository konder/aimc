# MineDojo 官方动作空间定义（已验证）

> **来源**: [MineDojo官方文档 - Action Space](https://docs.minedojo.org/sections/core_api/action_space.html)

---

## ⚠️ **重要修正**

对照官方文档，我们之前的理解有以下**错误**：

### **❌ 错误1: Jump维度的理解**
```python
# ❌ 我们之前认为:
# [2] jump: 0=不跳, 1=跳跃, 2=?, 3=?

# ✅ 官方正确定义:
# [2] Jump, sneak, and sprint: 
#     0=noop, 1=jump, 2=sneak, 3=sprint
```

### **❌ 错误2: Functional动作的理解**
```python
# ❌ 我们之前认为:
# [5] functional: 0=无, 3=攻击, 其他未知

# ✅ 官方正确定义:
# [5] Functional actions:
#     0=noop
#     1=use (使用物品/放置方块)
#     2=drop (丢弃物品)
#     3=attack (攻击) ✅ 我们验证正确
#     4=craft (合成)
#     5=equip (装备)
#     6=place (放置)
#     7=destroy (销毁)
```

### **❌ 错误3: Craft和Smelt的关系**
```python
# ❌ 我们之前认为:
# [6] craft: 合成
# [7] smelt: 熔炼

# ✅ 官方正确定义:
# [6] Argument for "craft": 244个合成配方（包含熔炉）
# [7] Argument for "equip", "place", "destroy": 36个物品栏槽位
```

---

## 📊 **官方完整定义**

```python
MultiDiscrete([3, 3, 4, 25, 25, 8, 244, 36])
```

| 索引 | 名称 | 详细说明 | 取值范围 |
|------|------|---------|---------|
| 0 | **Forward/Backward** | 前后移动 | 3个选项 |
| | | 0: noop (停止) | |
| | | 1: forward (前进) | |
| | | 2: back (后退) | |
| 1 | **Left/Right** | 左右移动 | 3个选项 |
| | | 0: noop (停止) | |
| | | 1: move left (左移) | |
| | | 2: move right (右移) | |
| 2 | **Jump/Sneak/Sprint** | 跳跃/潜行/疾跑 | 4个选项 |
| | | 0: noop (无) | |
| | | 1: jump (跳跃) | |
| | | 2: sneak (潜行) ⭐ 新发现 | |
| | | 3: sprint (疾跑) ⭐ 新发现 | |
| 3 | **Camera Delta Pitch** | 相机俯仰角增量 | 25个选项 |
| | | 0: -180度 | |
| | | 12: 0度 (中心) | |
| | | 24: +180度 | |
| 4 | **Camera Delta Yaw** | 相机偏航角增量 | 25个选项 |
| | | 0: -180度 | |
| | | 12: 0度 (中心) | |
| | | 24: +180度 | |
| 5 | **Functional Actions** | 功能动作 | 8个选项 |
| | | 0: noop (无动作) | |
| | | 1: use (使用物品) ⭐ | |
| | | 2: drop (丢弃物品) ⭐ | |
| | | 3: attack (攻击/挖掘) ✅ | |
| | | 4: craft (合成) ⭐ | |
| | | 5: equip (装备物品) ⭐ | |
| | | 6: place (放置方块) ⭐ | |
| | | 7: destroy (销毁物品) ⭐ | |
| 6 | **Craft Argument** | 合成配方参数 | 244个配方 |
| | | 0: 无合成 | |
| | | 1-243: 各种物品的合成配方 | |
| | | (包含工作台和熔炉配方) | |
| 7 | **Inventory Argument** | 物品栏参数 | 36个槽位 |
| | | 用于 equip/place/destroy | |
| | | 0-35: 物品栏槽位索引 | |

---

## 🎯 **更新后的按键映射建议**

### **基础动作（已确认）**

```python
# 移动控制
'w' → [1, 0, 0, 12, 12, 0, 0, 0]  # 前进
's' → [2, 0, 0, 12, 12, 0, 0, 0]  # 后退
'a' → [0, 1, 0, 12, 12, 0, 0, 0]  # 左移
'd' → [0, 2, 0, 12, 12, 0, 0, 0]  # 右移

# 视角控制（已确认）
'i' → [0, 0, 0, 8, 12, 0, 0, 0]   # 向上看
'k' → [0, 0, 0, 16, 12, 0, 0, 0]  # 向下看
'j' → [0, 0, 0, 12, 8, 0, 0, 0]   # 向左看
'l' → [0, 0, 0, 12, 16, 0, 0, 0]  # 向右看

# 功能动作（已确认）
'f' → [0, 0, 0, 12, 12, 3, 0, 0]  # 攻击/砍树 ✅
```

### **新增可用动作**

```python
# 跳跃/潜行/疾跑
' ' (空格) → [0, 0, 1, 12, 12, 0, 0, 0]  # 跳跃
'c' → [0, 0, 2, 12, 12, 0, 0, 0]         # 潜行 ⭐ 新
'shift' → [0, 0, 3, 12, 12, 0, 0, 0]     # 疾跑 ⭐ 新

# 其他功能动作（可选）
'r' → [0, 0, 0, 12, 12, 1, 0, 0]  # 使用物品（右键） ⭐ 新
't' → [0, 0, 0, 12, 12, 2, 0, 0]  # 丢弃物品 ⭐ 新
```

---

## 📝 **官方文档关键信息**

### **复合动作空间设计**

引用官方文档：

> "We design a compound action space. At each step the agent chooses **one movement action** (forward, backward, camera actions, etc.) and **one optional functional action** (attack, use, craft, etc.)."

这意味着：
- ✅ 可以同时执行移动和功能动作
- ✅ 例如：`[1, 0, 0, 12, 12, 3, 0, 0]` = 前进 + 攻击 ✅ 正确

### **相机角度计算**

官方定义：
```
pitch/yaw: 0 = -180度, 12 = 0度, 24 = +180度
```

计算公式：
```python
angle = (value - 12) * 15 度

# 示例
pitch=0  → (0-12)*15  = -180度 (最大向上)
pitch=8  → (8-12)*15  = -60度  (向上看) ✅ 我们使用的
pitch=12 → (12-12)*15 = 0度    (水平)
pitch=16 → (16-12)*15 = +60度  (向下看) ✅ 我们使用的
pitch=24 → (24-12)*15 = +180度 (最大向下)
```

### **Action Masks（动作掩码）**

官方提供了动作掩码来指示哪些动作是有效的：

```python
# 功能动作掩码
obs["masks"]["action_type"]       # 形状: (8,)

# 动作参数掩码
obs["masks"]["action_arg"]        # 形状: (8, 1)

# 装备掩码
obs["masks"]["equip"]             # 形状: (36,)

# 放置掩码
obs["masks"]["place"]             # 形状: (36,)

# 销毁掩码
obs["masks"]["destroy"]           # 形状: (36,)

# 合成/熔炼掩码
obs["masks"]["craft_smelt"]       # 形状: (244,)
```

---

## ✅ **验证结果总结**

### **我们正确的部分** ✅
1. ✅ 维度0-1（前后左右移动）- 完全正确
2. ✅ 维度3-4（相机pitch/yaw）- 完全正确
3. ✅ 维度5=3（攻击动作）- 已验证正确
4. ✅ 组合动作（移动+攻击）- 正确理解

### **我们错误的部分** ❌
1. ❌ 维度2：不仅是跳跃，还包括**潜行(2)**和**疾跑(3)**
2. ❌ 维度5：其他功能动作的含义未知 → 官方已明确定义
3. ❌ 维度7：不是smelt，而是**inventory槽位参数**

### **新发现的可用动作** ⭐
1. ⭐ **Sneak (潜行)**: `action[2] = 2` - 可用于隐蔽或慢速移动
2. ⭐ **Sprint (疾跑)**: `action[2] = 3` - 可用于快速移动
3. ⭐ **Use (使用)**: `action[5] = 1` - 右键使用物品
4. ⭐ **Drop (丢弃)**: `action[5] = 2` - 丢弃物品
5. ⭐ **Equip (装备)**: `action[5] = 5` + `action[7] = slot` - 装备物品
6. ⭐ **Place (放置)**: `action[5] = 6` + `action[7] = slot` - 放置方块
7. ⭐ **Destroy (销毁)**: `action[5] = 7` + `action[7] = slot` - 销毁物品

---

## 🔄 **推荐的更新**

### **对于砍树任务**

**必需动作**（保持不变）:
```python
'w' → 前进
's' → 后退
'a'/'d' → 左右移动
'i'/'k' → 上下看
'j'/'l' → 左右看
'f' → 攻击
' ' → 跳跃
```

**可选增强**（新增）:
```python
'shift' → 疾跑 (快速靠近树木)
'c' → 潜行 (精确对准树干)
```

### **对于复杂任务**

如果未来需要做更复杂的任务（如建造、合成），可以添加：
```python
'r' → 使用物品 (functional=1)
't' → 丢弃物品 (functional=2)
'e' → 装备物品 (functional=5)
'b' → 放置方块 (functional=6)
```

---

## 📚 **官方文档链接**

- **Action Space**: https://docs.minedojo.org/sections/core_api/action_space.html
- **Observation Space**: https://docs.minedojo.org/sections/core_api/observation_space.html
- **MineDojo API**: https://docs.minedojo.org/

---

## 🎯 **建议的后续行动**

1. ✅ **更新 `config/keyboard_mapping.yaml`**
   - 添加sneak和sprint的说明
   - 更新functional动作的完整定义

2. ✅ **更新 `KEYBOARD_REFERENCE_CARD.md`**
   - 添加可选的疾跑和潜行按键

3. ✅ **保持当前录制工具不变**
   - 砍树任务不需要sneak/sprint
   - 保持简单即可

4. ⚪ **未来考虑**
   - 如需要做合成/建造任务，再添加其他功能动作

---

**验证日期**: 2025-10-21  
**官方文档版本**: 最新  
**验证状态**: ✅ 已完成，发现并修正错误

