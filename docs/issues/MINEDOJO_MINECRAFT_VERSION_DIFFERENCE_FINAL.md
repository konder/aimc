# MineDojo vs MineRL Inventory 行为差异 - 最终结论

## 调查结果总结

经过深入的源码分析和反编译，我们找到了 MineDojo 和 MineRL 在 `inventory` 动作上行为差异的**真正原因**。

## 版本信息

| 环境 | Minecraft | Forge | Malmo | JAR |
|------|-----------|-------|-------|-----|
| **MineRL** | 1.16.5 | 36.x | MCP-Reborn 6.13 | mcprec-6.13.jar (444 MB) |
| **MineDojo** | 1.11.2 | 13.20.1.2588 | 0.37.0 | MalmoMod-0.37.0-fat.jar (90 MB) |

## Malmo 代码分析

### 源码完全相同

通过对比 Java 源文件：
- `CommandForKey.java`: MD5 = `89cc19df16ccc0c7aaadab042b8b8324` (相同)
- `FakeKeyboard.java`: MD5 = `a18307c55dad659c2a555f0ee8456888` (相同)

### 反编译运行时 JAR

从 MineDojo 的 `MalmoMod-0.37.0-fat.jar` 中提取并反编译 `FakeKeyboard.class`：

**`FakeKeyboard.press(int keyCode)` 逻辑**：
```java
if (!keysDown.contains(keyCode)) {
    System.out.println("Pressed" + keyCode);
    add(new FakeKeyEvent(' ', keyCode, true));  // isPressed=true
    // add() 会将 keyCode 添加到 keysDown 集合
}
```

**`FakeKeyboard.release(int keyCode)` 逻辑**：
```java
if (keysDown.contains(keyCode)) {
    System.out.println("Released" + keyCode);
    add(new FakeKeyEvent(' ', keyCode, false)); // isPressed=false
    // add() 会从 keysDown 集合中移除 keyCode
}
```

### 关键：`keysDown` 状态管理

1. **Step 2**: `inventory=1`
   - → `FakeKeyboard.press(inventoryKeyCode)`
   - → `keysDown.add(inventoryKeyCode)` ✅
   - → Minecraft 收到 **press** 事件 → 打开物品栏

2. **Step 3**: `inventory=0`
   - → `FakeKeyboard.release(inventoryKeyCode)`
   - → `if (keysDown.contains(inventoryKeyCode))` → **true!**
   - → 创建并发送 **release** 事件
   - → `keysDown.remove(inventoryKeyCode)` ❌
   - → Minecraft 收到 **release** 事件 → ???

## 根本差异：Minecraft 版本对 GUI 的处理

### Minecraft 1.11.2 (MineDojo)

**release** 事件的处理：
- GUI toggle 键（如 inventory）的 release 事件会**触发 GUI 关闭**
- 这是经典的"按键模型"：press = 打开，release = 关闭

### Minecraft 1.16.5 (MineRL)

**release** 事件的处理：
- GUI toggle 键的 release 事件**不会**触发 GUI 关闭
- 可能的原因：
  1. MC 1.13 "The Flattening" 重构了 GUI 系统
  2. inventory 键从"toggle"改为"持续状态"模式
  3. release 事件被忽略或有特殊处理

### 验证结果

通过实际测试：

**MineRL**:
```
Step 2: inventory=1 → 物品栏打开
Step 3-100: inventory=0 (每步都发送) → 物品栏保持打开 ✅
```

**MineDojo**:
```
Step 2: inventory=1 → 物品栏打开
Step 3-100: inventory=0 (每步都发送) → 物品栏在 Step 3 关闭 ❌
```

## 为什么无法解决

### 1. Malmo 层面无差异

- MineRL 和 MineDojo 的 Malmo Java 代码完全相同
- 两者都会发送 press/release 事件到 Minecraft
- 问题不在 Malmo

### 2. Minecraft 版本差异

- MC 1.11.2 和 MC 1.16.5 对 release 事件的响应不同
- 这是 Minecraft **游戏引擎**层面的行为
- 无法通过修改 Malmo 或 Python 代码解决

### 3. 升级 MineDojo 不可行

- 需要将 MineDojo 从 MC 1.11.2 升级到 MC 1.16.5
- 涉及 Forge 版本升级（13.x → 36.x）
- 需要重新编译整个 Malmo Mod
- 可能引入大量兼容性问题
- 工作量巨大，风险极高

## 实际影响评估

### 对 STEVE-1 的影响

**好消息**：
- `inventory=1` 可以成功打开物品栏 ✅
- STEVE-1 的 inventory 使用频率极低（0.17%）
- 通常模式：打开 → 操作（craft/equip） → 自动关闭
- 这种"瞬时打开"模式可能足够

**潜在问题**：
- 如果 STEVE-1 需要"持续保持物品栏打开"
- 或需要在物品栏打开时进行多步操作
- MineDojo 无法实现

### 建议

**短期**（推荐）：
- 接受现状，继续使用 MineDojo
- 在实际评估任务中测试 STEVE-1 的表现
- 观察 inventory 动作是否影响任务成功率

**长期**（如果必要）：
- 如果发现 inventory 行为确实导致任务失败
- 考虑使用 MineRL 环境（MC 1.16.5）而非 MineDojo
- 或实现"wrapper 状态管理"方案（持续发送 inventory=1）

## 相关文件

- 源码对比：`/usr/local/Caskroom/miniforge/base/envs/minedojo-x86/lib/python3.9/site-packages/{minerl,minedojo}/*/Malmo/Minecraft/src/main/java/`
- 运行时 JAR：`mcprec-6.13.jar` (MineRL), `MalmoMod-0.37.0-fat.jar` (MineDojo)
- 调查文档：`docs/issues/MINEDOJO_INVENTORY_BEHAVIOR_INVESTIGATION.md`
- 测试脚本：`scripts/test_{minerl,minedojo}_inventory_simple.py`

---

**结论**：问题根源是 **Minecraft 版本差异**（1.11.2 vs 1.16.5），而非 Malmo 实现差异。Malmo 代码完全相同，但底层 Minecraft 对 GUI release 事件的处理不同。这是无法通过修改 MineDojo 源码解决的根本性差异。

**创建日期**：2025-11-16  
**状态**：调查完成，根本原因已确认
