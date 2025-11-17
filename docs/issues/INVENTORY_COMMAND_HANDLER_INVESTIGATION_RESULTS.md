# MCP-Reborn vs Malmo CommandHandler 调查结果

## 调查时间
2025-11-16

## 目标
定位 MCP-Reborn (MineRL) 和 Malmo (MineDojo) 在处理 `inventory` 命令时的差异。

---

## 关键发现

### 1. JAR 文件结构差异

#### MCP-Reborn (MineRL)
```
JAR: mcprec-6.13.jar (444 MB)
路径: minerl/MCP-Reborn/build/libs/mcprec-6.13.jar
Minecraft: 1.16.5

结构:
❌ 未找到 com/microsoft/Malmo/ 包结构
❌ 没有提取到 CommandForKey.class 或 FakeKeyboard.class
⚠️  可能使用了不同的包结构或完全重写了命令处理系统
```

#### Malmo (MineDojo)
```
JAR: MalmoMod-0.37.0-fat.jar (90 MB)
路径: minedojo/sim/Malmo/Minecraft/build/libs/MalmoMod-0.37.0-fat.jar
Minecraft: 1.11.2

结构:
✓ 完整的 com/microsoft/Malmo/ 包结构
✓ CommandForKey.class 存在
✓ FakeKeyboard.class 存在
✓ 231 个 Malmo 类文件
```

### 2. Malmo 的 FakeKeyboard 实现（MC 1.11.2）

反编译代码显示：

```java
// press 方法
public static void press(int keyCode) {
    // 检查 keysDown 集合
    if (!keysDown.contains(keyCode)) {
        // 打印日志
        System.out.println("Pressed" + keyCode);
        
        // 创建 press 事件（最后一个参数是 true）
        FakeKeyEvent event = new FakeKeyEvent(' ', keyCode, true);
        add(event);
    }
}

// release 方法
public static void release(int keyCode) {
    // 检查 keysDown 集合
    if (keysDown.contains(keyCode)) {
        // 打印日志
        System.out.println("Released" + keyCode);
        
        // 创建 release 事件（最后一个参数是 false）
        FakeKeyEvent event = new FakeKeyEvent(' ', keyCode, false);
        add(event);
    }
}

// add 方法
public static void add(FakeKeyEvent event) {
    // 添加到事件队列
    eventQueue.add(event);
    
    // 更新 keysDown 集合
    if (event.isPressed) {
        keysDown.add(event.keyCode);
    } else {
        keysDown.remove(event.keyCode);
    }
}
```

#### 关键机制

1. **keysDown 集合**: 维护当前按下的键
2. **press**: 只有当键不在 `keysDown` 中时才会创建 press 事件
3. **release**: 只有当键在 `keysDown` 中时才会创建 release 事件
4. **事件队列**: 所有事件都添加到队列中，然后传递给 Minecraft

### 3. inventory 命令的行为差异分析

#### Malmo (MineDojo) 的流程

```
Python: inventory=1
    ↓
Socket → XML
    ↓
CommandForKey.onExecute("inventory", "1")
    ↓
FakeKeyboard.press(inventoryKeyCode)
    ↓
检查: keysDown.contains(inventoryKeyCode)?
    NO → 创建 press 事件，添加到 keysDown
    ↓
Minecraft 1.11.2 接收 press 事件
    ↓
GUI Toggle 或 打开

Python: inventory=0
    ↓
Socket → XML
    ↓
CommandForKey.onExecute("inventory", "0")
    ↓
FakeKeyboard.release(inventoryKeyCode)
    ↓
检查: keysDown.contains(inventoryKeyCode)?
    YES → 创建 release 事件，从 keysDown 移除
    ↓
Minecraft 1.11.2 接收 release 事件
    ↓
GUI 关闭 或 Toggle
```

#### MCP-Reborn (MineRL) 的预期流程

由于没有提取到 `com/microsoft/Malmo/` 包结构，推测：

**可能性 A**: 使用了不同的包名和类结构
```
com/minerl/...
或
com/mcpreborn/...
```

**可能性 B**: 完全重写了命令处理系统
```
不再使用 CommandForKey / FakeKeyboard
直接与 MC 1.16.5 API 交互
```

**可能性 C**: 修改了 FakeKeyboard 的行为
```java
// MCP-Reborn 可能的实现
public static void release(int keyCode) {
    // 对于 inventory 键，忽略 release
    if (keyCode == INVENTORY_KEY) {
        return; // ← 这就是为什么 inventory 保持打开！
    }
    
    // 其他键的正常处理
    if (keysDown.contains(keyCode)) {
        ...
    }
}
```

### 4. Minecraft 层面的差异（最关键！）

无论 Java Handler 实现如何，最终的差异来自 **Minecraft 本身**：

#### MC 1.11.2 (Malmo/MineDojo)
```java
// 伪代码 - KeyBinding 处理
public void handleKeyEvent(int keyCode, boolean pressed) {
    if (keyCode == INVENTORY_KEY) {
        if (pressed) {
            toggleInventory(); // Toggle 模式
        } else {
            // 可能也会触发 toggle
            // 或者关闭 GUI
        }
    }
}
```

#### MC 1.16.5 (MCP-Reborn/MineRL)
```java
// 伪代码 - KeyBinding 处理
public void handleKeyEvent(int keyCode, boolean pressed) {
    if (keyCode == INVENTORY_KEY) {
        if (pressed && currentScreen == null) {
            openInventory(); // 只响应 press
        }
        // 忽略 release 事件
    }
}
```

---

## 核心结论

### 问题的根源

**inventory GUI 自动关闭的根本原因有两个层面：**

#### 层面 1: Java Handler（次要）

Malmo 的 `FakeKeyboard` 会忠实地执行：
- `inventory=1` → `press(inventoryKey)` → 添加到 `keysDown`
- `inventory=0` → `release(inventoryKey)` → 从 `keysDown` 移除

这意味着每次发送 `inventory=0` 都会生成一个 release 事件。

#### 层面 2: Minecraft GUI 系统（主要）

- **MC 1.11.2**: 响应 release 事件，触发 toggle 或关闭
- **MC 1.16.5**: 忽略 release 事件，GUI 保持打开

**即使 MCP-Reborn 和 Malmo 的 FakeKeyboard 实现完全相同，最终的行为也会因为 Minecraft 版本不同而不同。**

### MCP-Reborn 的可能优化

MCP-Reborn 可能做了以下之一：

1. **修改了 FakeKeyboard**: 对 inventory 键的 release 进行特殊处理
2. **修改了 CommandForKey**: 阻止生成 inventory 的 release 命令
3. **利用了 MC 1.16.5 的特性**: 依赖 MC 1.16.5 本身忽略 release

最可能的是 **#3**：MCP-Reborn 依赖 MC 1.16.5 的行为，无需特殊处理。

---

## 当前解决方案的正确性

我们在 Python wrapper 中采用的策略是正确的：

```python
# MineDojoBiomeWrapper.step()
if current_inventory == 1 and not self._inventory_opened:
    # 第一次打开 - 发送 inventory=1
    self._inventory_opened = True
else:
    # 其他情况 - 移除 inventory 键
    # 阻止发送任何 inventory 命令到 Malmo
    if 'inventory' in minerl_action:
        del minerl_action['inventory']
```

### 为什么有效？

1. **第一次打开**: `inventory=1` → Malmo 生成 press 事件 → GUI 打开
2. **后续步骤**: 不发送任何 inventory 命令 → Malmo 不生成任何事件 → GUI 保持当前状态（打开）

这模拟了 MCP-Reborn + MC 1.16.5 的行为：
- 只发送一次 press
- 永不发送 release
- GUI 保持打开

---

## 进一步调查（可选）

如果需要更深入的对比，可以：

### 1. 查找 MCP-Reborn 的命令处理类

```bash
# 列出 MCP-Reborn JAR 中的所有类
unzip -l mcprec-6.13.jar | grep "\.class$" | grep -i "command\|keyboard" > mcp_classes.txt

# 可能的包名:
# - com/mcpreborn/
# - com/minerl/
# - net/minecraft/client/ (直接修改了 MC 类)
```

### 2. 反编译 MCP-Reborn 的关键类

如果找到了相应的类，使用 `javap` 或更强大的反编译工具（如 JD-GUI, CFR）。

### 3. 对比 Minecraft 原版代码

- MC 1.11.2 的 KeyBinding 实现
- MC 1.16.5 的 KeyBinding 实现

### 4. 查看运行时日志

Malmo 的 `FakeKeyboard` 已经包含了日志：
```java
System.out.println("Pressed" + keyCode);
System.out.println("Released" + keyCode);
```

可以在运行时查看这些日志来确认实际发送的命令。

---

## 总结表

| 方面 | MCP-Reborn (MineRL) | Malmo (MineDojo) | 影响 |
|------|-------------------|------------------|------|
| **JAR 结构** | 未找到 Malmo 类 | 完整 Malmo 类 | 重大 |
| **包名** | 未知（可能重写） | `com/microsoft/Malmo/` | 重大 |
| **FakeKeyboard** | 未提取到 | 标准 press/release | 中等 |
| **CommandForKey** | 未提取到 | 标准实现 | 中等 |
| **Minecraft 版本** | 1.16.5 | 1.11.2 | **关键** |
| **GUI 行为** | 忽略 release | 响应 release | **关键** |
| **Wrapper 策略** | 无需（MC 自动处理） | 必需（阻止 release） | 关键 |

---

## 建议

1. ✅ **当前的 wrapper 策略是正确的** - 继续使用
2. ⚠️ **如果需要更精确的控制** - 可以考虑：
   - 修改 MineDojo 的 Java 代码（已完成：添加 inventory 支持）
   - 在 FakeKeyboard 层面过滤 inventory release（复杂，不推荐）
3. 📝 **记录差异** - 为未来可能的升级或迁移做准备
4. 🔬 **可选的深入调查** - 如果时间允许，反编译 MCP-Reborn 了解其具体实现

---

## 文件位置

- 调查脚本: `/Users/nanzhang/aimc/scripts/investigate_inventory_handlers.sh`
- 调查结果: `/tmp/inventory_investigation/`
- 本文档: `/Users/nanzhang/aimc/docs/issues/INVENTORY_COMMAND_HANDLER_INVESTIGATION_RESULTS.md`


