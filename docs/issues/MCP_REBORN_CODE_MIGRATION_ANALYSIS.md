# 移植 MCP-Reborn 按键处理机制的可行性分析

## 背景

用户观察到：
- **MC 1.11.2** (MineDojo): 原生按 E 键 → 打开，保持，再按 E → 关闭 ✅
- **MC 1.16.5** (MineRL): 原生按 E 键 → 打开，保持，再按 E → 关闭 ✅

**两者原生行为完全相同！**

问题在于：
- **Malmo (MineDojo)**: 使用事件驱动（press/release）→ GUI 自动关闭 ❌
- **MCP-Reborn (MineRL)**: 使用状态驱动（keys list）→ GUI 保持打开 ✅

---

## 问题根源

### Malmo 的事件处理

```
Python: inventory=1
    ↓
FakeKeyboard.press(E) → 发送 KeyEvent(pressed=true)
    ↓
Minecraft 接收 press 事件 → 打开 GUI
    ↓
Python: inventory=0
    ↓
FakeKeyboard.release(E) → 发送 KeyEvent(pressed=false)
    ↓
Minecraft 接收 release 事件 → ❌ 可能触发 toggle，关闭 GUI
```

**关键问题**：Malmo 的 `FakeKeyboard` 会发送 **release 事件**，这在 Minecraft 中可能触发意外的行为。

### MCP-Reborn 的状态处理

```
Python: inventory=1
    ↓
KeyboardListener.State(keys=["key.keyboard.e"])
    ↓
Minecraft 看到 E 键在 keys 列表中 → 打开 GUI
    ↓
Python: inventory=0
    ↓
KeyboardListener.State(keys=[])  # E 键不在列表中
    ↓
Minecraft 看到 E 键不在列表中 → ✅ 不触发任何事件，GUI 保持打开
```

**优势**：状态驱动只传递"哪些键当前被按下"，不会生成多余的 release 事件。

---

## 方案分析

### 方案 A：直接移植 MCP-Reborn 的 EnvServer + KeyboardListener

#### 可行性：❌ **不可行**

**原因**：

1. **MineDojo 使用官方 Malmo**
   - Malmo JAR 是预编译的
   - 无法修改 Malmo 的核心通信协议
   - Malmo 期望接收 XML 格式的命令消息

2. **MCP-Reborn 修改了 Minecraft 原生类**
   ```
   net/minecraft/client/KeyboardListener.java  # 直接修改 MC 类
   net/minecraft/client/settings/KeyBinding.java  # 直接修改 MC 类
   ```
   - MineDojo 无法修改 Minecraft 1.11.2 的原生类
   - 需要重新编译整个 Minecraft + Forge

3. **通信协议不兼容**
   - Malmo: XML over TCP + CommandForKey → FakeKeyboard
   - MCP-Reborn: Text over TCP + 直接调用 MC API

#### 工作量评估：🔴 **极高**

- 需要 fork MineDojo 的 Malmo
- 需要重新编译 Minecraft 1.11.2 + Forge
- 需要重写 MineDojo 的通信层
- 预计：**几周到几个月**的工作量

---

### 方案 B：在 Malmo 的 FakeKeyboard 层面修改（避免 release 事件）

#### 可行性：⚠️ **理论可行，但有风险**

**思路**：修改 Malmo 的 `FakeKeyboard.java`，对 inventory 键特殊处理：

```java
// 修改 /usr/local/.../minedojo/sim/Malmo/.../FakeKeyboard.java
public static void release(int keyCode) {
    // 特殊处理：如果是 inventory 键，忽略 release
    if (keyCode == INVENTORY_KEY_CODE) {
        System.out.println("Ignoring inventory release");
        return;  // ← 直接返回，不发送 release 事件
    }
    
    // 其他键的正常处理
    if (keysDown.contains(keyCode)) {
        System.out.println("Released" + keyCode);
        FakeKeyEvent event = new FakeKeyEvent(' ', keyCode, false);
        add(event);
    }
}
```

#### 优势：

✅ 工作量小（只修改一个方法）
✅ 不需要重新编译 Minecraft
✅ 不需要修改通信协议

#### 劣势：

❌ 仍然需要修改和重新编译 Malmo JAR
❌ 可能影响其他依赖 release 事件的功能
❌ 不是通用解决方案（只针对 inventory）

#### 工作量评估：🟡 **中等**

- 修改 Malmo Java 源码
- 重新编译 Malmo JAR
- 测试验证
- 预计：**几天**的工作量

---

### 方案 C：在 MineDojo Python 层面过滤（当前方案）

#### 可行性：✅ **已实现且有效**

**思路**：在 Python wrapper 层面阻止发送 `inventory=0`：

```python
# src/envs/minedojo_harvest.py (已实现)
def step(self, minerl_action: Dict):
    current_inventory = minerl_action.get('inventory', 0)
    
    if current_inventory == 1 and not self._inventory_opened:
        # 第一次打开
        self._inventory_opened = True
    else:
        # 后续删除 inventory 键，不发送任何命令
        if 'inventory' in minerl_action:
            del minerl_action['inventory']
    
    # 转换并执行动作
    ...
```

#### 优势：

✅ **无需修改 Java 代码**
✅ **无需重新编译任何东西**
✅ **完全在 Python 层面实现**
✅ **已验证有效**
✅ **维护成本低**

#### 劣势：

⚠️ 不是"真正"的状态驱动（仍然基于 Malmo 的事件驱动）
⚠️ 依赖于 wrapper 的状态管理

#### 工作量评估：🟢 **已完成**

- 已实现并测试
- 预计：**0 额外工作量**

---

## 深入分析：为什么原生 MC 行为相同，但 Malmo 行为不同？

### 原生 Minecraft（你的测试）

```
用户按下 E 键
    ↓
OS 生成 KeyEvent(pressed=true)
    ↓
Minecraft KeyBinding 系统
    ↓
检测到 E 键 press → 打开 GUI
    ↓
用户松开 E 键
    ↓
OS 生成 KeyEvent(pressed=false)
    ↓
Minecraft KeyBinding 系统
    ↓
检测到 E 键 release → ✅ 忽略（MC 的 inventory 只响应 press）
    ↓
GUI 保持打开
    ↓
用户再次按下 E 键
    ↓
Minecraft KeyBinding 系统
    ↓
检测到 E 键 press → 关闭 GUI（toggle）
```

### Malmo 的问题

```
Python: inventory=1
    ↓
FakeKeyboard.press(E) → 模拟 press 事件
    ↓
Minecraft 接收 → 打开 GUI ✅
    ↓
Python: inventory=0
    ↓
FakeKeyboard.release(E) → 模拟 release 事件
    ↓
Minecraft 接收 → ❌ 可能的问题：
    1. release 被误解为新的"按键"
    2. release 触发了某种 toggle 逻辑
    3. release 与下一个 press 的时序问题
```

**关键差异**：

在原生 MC 中：
- Press → Open
- Release → **被忽略**
- Press → Close (toggle)

在 Malmo + MC 中：
- `inventory 1` → Press → Open
- `inventory 0` → Release → **可能被误解为某种事件**
- 导致 GUI 意外关闭

---

## 建议

### 最佳方案：保持当前的 Python Wrapper 策略（方案 C）

**理由**：

1. ✅ **已经工作**：当前方案已经成功模拟了 MCP-Reborn 的行为
2. ✅ **成本最低**：无需修改 Java 代码或重新编译
3. ✅ **维护简单**：所有逻辑都在 Python 层面，易于调试和修改
4. ✅ **足够好**：对于 STEVE-1 的使用场景完全满足需求

### 可选方案：修改 Malmo FakeKeyboard（方案 B）

**适用场景**：

如果你需要：
- 更"纯粹"的解决方案
- 不依赖 wrapper 状态管理
- 让 MineDojo 的行为完全一致于 MCP-Reborn

**步骤**：

1. 修改 `/usr/local/.../minedojo/sim/Malmo/.../FakeKeyboard.java`
2. 在 `release()` 方法中添加 inventory 键的特殊处理
3. 重新编译 Malmo JAR：
   ```bash
   cd /usr/local/.../minedojo/sim/Malmo/Minecraft
   ./gradlew shadowJar
   ```
4. 测试验证

**风险**：
- 可能影响其他功能
- 需要维护自己的 Malmo fork

### 不推荐：完全移植 MCP-Reborn 架构（方案 A）

**理由**：
- 工作量太大（几周到几个月）
- 需要深入修改 MineDojo 核心
- 维护成本高
- 收益不明显（当前方案已足够好）

---

## 实验建议

如果你想验证 Malmo 的 release 事件是否是问题根源，可以：

### 实验 1：在 MineDojo 中添加日志

修改 MineDojo 的 `action.py`，打印实际发送的命令：

```python
# /usr/local/.../minedojo/sim/handlers/agent/action.py
def to_hero(self, x):
    cmd = ""
    verb = self.command
    # ... 处理 x ...
    cmd += "{} {}".format(verb, adjective)
    
    # 添加日志
    if verb == "inventory":
        import sys
        print(f"[to_hero] inventory command: '{cmd}'", file=sys.stderr, flush=True)
    
    return cmd
```

然后运行测试，观察实际发送的命令。

### 实验 2：强制不发送 inventory 0

在 wrapper 中完全阻止 `inventory=0`：

```python
# 已经实现的策略
if 'inventory' in minerl_action and not self._should_send_inventory:
    del minerl_action['inventory']
```

如果这样能让 inventory 保持打开，就证明了 release 事件是问题根源。

---

## 结论

**你的观察非常正确**：MC 1.11.2 和 1.16.5 的原生 GUI 行为是相同的。

**问题的根源**：Malmo 的 **事件驱动模型**（press/release）与 Minecraft 的 **toggle 行为**不匹配。

**最佳解决方案**：
- **短期**：保持当前的 Python wrapper 策略 ✅
- **中期**：如果需要，修改 Malmo 的 FakeKeyboard
- **长期**：不推荐完全移植 MCP-Reborn 架构（成本太高）

**当前方案已经足够好**，它成功模拟了 MCP-Reborn 的行为，对 STEVE-1 完全够用！

---

## 文件位置

本文档：`docs/issues/MCP_REBORN_CODE_MIGRATION_ANALYSIS.md`


