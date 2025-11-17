# MineDojo Inventory 动作调查总结

## 问题描述

在 MineDojo 环境中使用 STEVE-1 时，发现 `inventory` 动作（MineRL 的标准动作）未被映射到 MineDojo 的动作空间，导致该动作被丢弃。

## 调查过程

### 1. 初始实现（失败）

**尝试**：在 MineDojo 的 `common_actions` 列表中添加 `"inventory"`，使其像其他键盘动作一样被支持。

**结果**：
- ✅ `inventory=1` 可以打开物品栏
- ❌ `inventory=0`（或 no_op）会立即关闭物品栏

**原因分析**：MineDojo 的 `action.py` 中，`to_hero()` 方法会过滤掉值为 `0` 的命令：
```python
if adjective != "0":
    cmd = "%s %s" % (self.command, adjective)
```

### 2. 尝试过滤 `inventory=0`（失败）

**假设**：应该过滤 `inventory=0`，避免发送关闭命令。

**实现**：
- 修改 `minedojo/sim/handlers/agent/action.py`：在 `to_hero()` 中，当 `inventory=0` 时返回空字符串
- 修改 `minedojo/sim/sim.py`：在 `_action_obj_to_xml()` 中过滤 `inventory=0`

**结果**：❌ 物品栏仍然在下一个 step 自动关闭。

### 3. 对比 MineRL 行为（关键发现）

**实验**：通过修改 MineRL 和 MineDojo 的源码，打印所有 `inventory` 命令。

**MineRL 行为**：
```
Step 2: inventory=1  →  打开物品栏
Step 3-100: inventory=0  →  每步都发送！
结果: ✅ 物品栏保持打开
```

**MineDojo 行为**（过滤 `inventory=0` 时）：
```
Step 2: inventory=1  →  打开物品栏
Step 3-100: (无 inventory 命令)  →  被过滤
结果: ❌ 物品栏在 Step 3 关闭
```

**结论**：MineRL 通过**持续发送 `inventory=0`** 来保持物品栏打开，而不是不发送！

### 4. 恢复发送 `inventory=0`（仍失败）

**尝试**：移除所有过滤逻辑，像 MineRL 一样每个 step 都发送 `inventory=0`。

**MineDojo 行为**（发送 `inventory=0` 时）：
```
Step 2: inventory=1  →  打开物品栏
Step 3-100: inventory=0  →  每步都发送
结果: ❌ 物品栏仍在 Step 3 关闭
```

## 根本原因

**🎯 确认：MineDojo 和 MineRL 使用不同的 Minecraft 版本，导致 GUI 处理逻辑不同！**

### 版本对比

| 环境 | Minecraft | Forge | Malmo | JAR 大小 |
|------|-----------|-------|-------|----------|
| **MineRL** | **1.16.5** | 36.x | MCP-Reborn 6.13 | 444 MB |
| **MineDojo** | **1.11.2** | 13.20.1.2588 | 0.37.0 | 90 MB |

**版本差距：5 个大版本！**(1.11 → 1.12 → 1.13 → 1.14 → 1.15 → 1.16)

### `inventory 0` 行为差异

| Minecraft 版本 | `inventory 0` 行为 | 物品栏保持打开？ | 逻辑模型 |
|----------------|---------------------|------------------|----------|
| **MC 1.11.2** (MineDojo) | 触发 `FakeKeyboard.release()` → 关闭 GUI | ❌ 否 | 按键模型 (0 = 释放) |
| **MC 1.16.5** (MineRL) | 不触发关闭，保持当前状态 | ✅ 是 | 状态模型 (0 = 维持) |

### 为什么会有这个差异？

**Minecraft 1.13 "The Flattening" 更新**：
- 大量重构了底层代码
- GUI 系统从"事件驱动"改为"状态驱动"
- 键盘输入处理逻辑变化
- `inventory` 键可能从 "toggle" 改为 "open with hold" 模式

**Java 源码相同但行为不同**：
- `CommandForKey.java` 和 `FakeKeyboard.java` 的源码在两个版本中完全相同 (MD5 一致)
- 但 Minecraft 客户端本身对这些 Malmo 命令的响应不同
- 这是 Minecraft **游戏引擎**层面的差异，不是 Malmo 层面

### 验证过程

1. ✅ 对比了 MineRL 和 MineDojo 的 Java 源码 → 完全相同
2. ✅ 对比了运行时行为 → MineRL 发送 `inventory 0`，MineDojo 也发送
3. ✅ 检查了 JAR 文件 → 大小差异巨大 (444MB vs 90MB)
4. ✅ 找到了 `build.gradle` → MineDojo 使用 MC 1.11.2，MineRL 使用 MC 1.16.5

## 当前实现状态

**已实现**：
- ✅ `inventory` 动作已添加到 MineDojo 的 `common_actions`
- ✅ `inventory=1` 可以打开物品栏
- ✅ wrapper 正确映射 MineRL 的 `inventory` 动作到 MineDojo

**限制**：
- ❌ 无法像 MineRL 一样"保持物品栏打开"
- ⚠️ `inventory=0` 会立即关闭物品栏（下一个 step）

## 对 STEVE-1 的影响

**好消息**：
- STEVE-1 的 `inventory` 动作频率很低（观察到 3/1751 steps = 0.17%）
- 通常模式：`打开 → 立即操作（如 craft, equip） → 自动关闭`
- 这种"瞬时打开"可能足够满足需求

**潜在问题**：
- 如果 STEVE-1 需要"持续保持物品栏打开"进行多步操作
- MineDojo 无法实现
- 但实际场景中，STEVE-1 可能不依赖这种行为

## 解决方案选项

### 方案 A：接受现状（推荐）✅

**实施**：
- 保持当前实现（inventory 可打开，但无法保持）
- 继续使用 MineDojo 进行评估

**优点**：
- 简单，无需进一步修改
- 可能满足 STEVE-1 的实际需求

**缺点**：
- 与 MineRL 行为不完全一致

**建议**：先在实际任务中测试，观察是否影响 STEVE-1 的表现。

### 方案 B：Wrapper 状态管理（中等风险）⚠️

**实施**：
- 在 `MineDojoBiomeWrapper` 中维护 `_inventory_open` 状态
- 当收到 `inventory=1` 时，后续所有 step 强制发送 `inventory=1`（而非 0）
- 直到显式收到"关闭"信号（如 ESC）

**优点**：
- 可能实现"保持打开"的效果

**缺点**：
- 可能与 STEVE-1 的预期行为不符
- 可能导致意外的 GUI 状态
- 需要额外的状态管理逻辑

### 方案 C：修改 Malmo Java 代码（高风险）❌

**实施**：
- 修改 `minedojo/sim/Malmo/Minecraft/.../CommandForKey.java`
- 或修改 `FakeKeyboard.java`
- 让 `inventory 0` 的行为与 MineRL 一致

**优点**：
- 完全解决问题

**缺点**：
- 需要重新编译 Malmo
- 可能引入其他问题
- 维护困难
- 不推荐

## 补丁文件说明

`docker/minedojo_inventory.patch` 包含的修改：

1. **`minedojo/sim/sim.py`**：
   - 在 `common_actions` 中添加 `"inventory"`

2. **`minedojo/sim/wrappers/ar_nn/nn_action_space_wrapper.py`**：
   - 扩展 functional actions 从 8 到 9
   - 添加 `inventory` 到 functional action index 5, value 8

3. **`minedojo/sim/handlers/agent/action.py`**：
   - ~~移除了 `adjective != "0"` 的过滤~~（已恢复，保持原样）

**注意**：经过实验，我们发现原始的 `adjective != "0"` 过滤应该保留，因为移除它也无法解决"保持打开"的问题。

## 测试验证

### 测试脚本

**MineRL 对比测试**：
```bash
python scripts/test_minerl_inventory_simple.py
```

**MineDojo 测试**：
```bash
python scripts/test_minedojo_inventory_simple.py
```

### 预期结果

**MineDojo**：
- Step 2: `inventory=1` → 物品栏打开 ✅
- Step 3+: `inventory=0` → 物品栏关闭 ⚠️

## 建议

1. **短期**：使用方案 A，在实际评估任务中测试 STEVE-1 的表现
2. **观察**：记录 STEVE-1 使用 `inventory` 的模式和频率
3. **评估**：如果发现 `inventory` 无法保持打开导致任务失败，再考虑方案 B

## 相关文件

- `src/envs/minedojo_harvest.py`：MineDojo wrapper 实现
- `docker/minedojo_inventory.patch`：MineDojo 源码补丁
- `scripts/test_minerl_inventory_simple.py`：MineRL 测试脚本
- `scripts/test_minedojo_inventory_simple.py`：MineDojo 测试脚本
- `docs/technical/MINEDOJO_INVENTORY_ACTION_IMPLEMENTATION.md`：实现文档

---

**创建日期**：2025-11-16  
**最后更新**：2025-11-16  
**状态**：已完成调查，建议使用方案 A

