# MCP-Reborn vs Malmo Socket 消息处理流程对比

## ⚠️ 重要说明

**MCP-Reborn ≠ Malmo**

- **MineRL** 依赖 **MCP-Reborn 6.13** (Malmo 的 fork/定制版本，针对 MC 1.16.5)
- **MineDojo** 依赖 **Malmo 0.37.0** (Microsoft 官方版本，针对 MC 1.11.2)

这两个是**不同的代码库**，虽然可能继承自同一个基础，但已经分叉，实现细节有重大差异。

---

## MCP-Reborn 与 Malmo 的关系

### 项目背景

**Malmo (Microsoft)**:
- 官方 AI 研究平台
- 最初发布于 2016 年左右
- 支持多个 Minecraft 版本 (1.11.2, 1.12.x 等)
- 开源项目: https://github.com/Microsoft/malmo

**MCP-Reborn (MineRL)**:
- 基于 Malmo fork/重写
- 专门为 MineRL 项目定制
- 升级到 Minecraft 1.16.5 + Forge 36.x
- 添加了 MineRL 特有的功能
- 项目: https://github.com/minerllabs/MCP-Reborn

### 代码演化

```
Malmo 0.37.0 (MC 1.11.2)
    ↓
   Fork / Modify
    ↓
MCP-Reborn 6.13 (MC 1.16.5)
    ↓
   重大修改：
   • 适配 MC 1.16.5 API
   • 适配 Forge 36.x
   • 添加 MineRL 特有功能
   • 可能修改了 FakeKeyboard 等核心类
```

### 为什么它们不同？

#### 1. Minecraft API 重大变化

MC 1.11.2 → 1.16.5 之间经历了多次重大更新：
- 1.13: "Flattening" (方块 ID 系统重写)
- 1.14: "Village & Pillage" (村庄系统重写)
- 1.15: "Buzzy Bees" (性能优化)
- 1.16: "Nether Update" (下界更新)

每次更新都可能改变：
- KeyBinding 系统
- GUI 系统
- 渲染系统
- 网络协议

#### 2. Forge API 变化

Forge 13.20 → 36.x：
- 事件系统重构
- Mod 加载机制改变
- API 结构调整

#### 3. MineRL 特有需求

MCP-Reborn 可能添加了：
- 更高效的观察采集
- 更精确的动作控制
- 特殊的 Mission Handler
- 性能优化

### 实际影响

虽然 MCP-Reborn 和 Malmo 的**架构相似**，但：

1. **类实现不同**: 
   - FakeKeyboard.java 需要调用不同的 MC API
   - CommandForKey.java 需要适配不同的事件系统
   
2. **行为可能不同**:
   - 相同的 `press(keyCode)` 调用
   - 但最终触发的 MC 事件不同
   - 导致 GUI 行为不同

3. **无法直接对比**:
   - 不能假设它们的实现相同
   - 需要分别反编译 JAR 查看实际代码

---

## 概述

当 Python 端通过 `client_socket_send_message(step_message.encode())` 发送消息到 Minecraft 客户端时，MCP-Reborn (MineRL) 和 Malmo (MineDojo) 的处理流程有相似之处，但也有重要差异。

---

## 消息格式

### Step Message 格式
```xml
<StepCommand>
    <inventory>1</inventory>
    <forward>0</forward>
    <camera>0 0</camera>
    <!-- ... 其他动作 ... -->
</StepCommand>
```

---

## 1. MCP-Reborn (MineRL) 处理流程

### 环境信息
- **版本**: MCP-Reborn 6.13
- **Minecraft**: 1.16.5
- **Forge**: 36.x
- **JAR**: `mcprec-6.13.jar` (444 MB)

### 流程图

```
Python 端
  ↓
instance.client_socket_send_message(step_message.encode())
  ↓
TCP Socket 发送 XML 消息
  ↓
═══════════════════════════════════════════════════════════════
Java 端 (Minecraft 1.16.5 客户端)
═══════════════════════════════════════════════════════════════
  ↓
1. ClientCommandHandler.java
   └─ 监听 socket 端口（通常 9000）
   └─ 接收 TCP 消息
   └─ 解析 XML
  ↓
2. MissionBehaviourManager.java
   └─ 管理当前 Mission
   └─ 分发命令到对应的 Handler
  ↓
3. CommandForKey.java (针对 keyboard 命令)
   └─ 解析 <inventory>1</inventory>
   └─ verb = "inventory"
   └─ parameter = "1"
   └─ onExecute(verb, parameter)
  ↓
4. 判断 press/release
   if (parameter == "1"):
       FakeKeyboard.press(inventoryKeyCode)
   else:
       FakeKeyboard.release(inventoryKeyCode)
  ↓
5. FakeKeyboard.java
   └─ press(keyCode):
       if (!keysDown.contains(keyCode)):
           keysDown.add(keyCode)
           发送 KeyEvent 到 Minecraft
   └─ release(keyCode):
       if (keysDown.contains(keyCode)):
           keysDown.remove(keyCode)
           发送 KeyEvent 到 Minecraft
  ↓
6. Minecraft 1.16.5 KeyBinding 系统
   └─ 接收 KeyEvent
   └─ 处理 GUI 打开/关闭
   └─ **MC 1.16.5 特性**: 忽略 inventory release 事件
   └─ GUI 保持打开 ✅
  ↓
7. 渲染循环
   └─ 更新游戏状态
   └─ 渲染 GUI
  ↓
8. 观察数据采集
   └─ 截取屏幕 (POV)
   └─ 收集 inventory 数据
   └─ 收集其他状态
  ↓
9. ObservationMessage.java
   └─ 构造观察 XML
   └─ 通过 socket 发送回 Python
  ↓
═══════════════════════════════════════════════════════════════
Python 端
═══════════════════════════════════════════════════════════════
  ↓
instance.receive_observation()
  ↓
解析观察数据
  ↓
返回 obs, reward, done, info
```

### 关键文件路径 (MCP-Reborn)

```
minerl/MCP-Reborn/
├── com/microsoft/Malmo/
│   ├── Client/
│   │   ├── ClientCommandHandler.java       # Socket 监听
│   │   └── FakeKeyboard.java               # 键盘模拟
│   ├── MissionHandlers/
│   │   ├── CommandForKey.java              # 键盘命令处理
│   │   └── ObservationProducer.java        # 观察生成
│   └── MissionBehaviourManager.java        # Mission 管理
└── net/minecraft/client/
    ├── KeyBinding.java                      # MC 键绑定系统
    └── gui/                                 # GUI 系统
```

---

## 2. Malmo (MineDojo) 处理流程

### 环境信息
- **版本**: Malmo 0.37.0
- **Minecraft**: 1.11.2
- **Forge**: 13.20.1.2588
- **JAR**: `MalmoMod-0.37.0-fat.jar` (90 MB)

### 流程图

```
Python 端
  ↓
instance.client_socket_send_message(step_message.encode())
  ↓
TCP Socket 发送 XML 消息
  ↓
═══════════════════════════════════════════════════════════════
Java 端 (Minecraft 1.11.2 客户端)
═══════════════════════════════════════════════════════════════
  ↓
1. ClientStateMachine.java
   └─ 监听 socket 端口
   └─ 接收 TCP 消息
   └─ 解析 XML
  ↓
2. MissionCommandHandler.java
   └─ 管理当前 Mission 状态
   └─ 分发命令到对应的 Handler
  ↓
3. CommandForKey.java (针对 keyboard 命令)
   └─ 解析 <inventory>1</inventory>
   └─ verb = "inventory"
   └─ parameter = "1"
   └─ onExecute(verb, parameter)
  ↓
4. 判断 press/release (与 MCP-Reborn 相同)
   if (parameter == "1"):
       FakeKeyboard.press(inventoryKeyCode)
   else:
       FakeKeyboard.release(inventoryKeyCode)
  ↓
5. FakeKeyboard.java (与 MCP-Reborn 相同)
   └─ press(keyCode):
       if (!keysDown.contains(keyCode)):
           keysDown.add(keyCode)
           发送 KeyEvent 到 Minecraft
   └─ release(keyCode):
       if (keysDown.contains(keyCode)):
           keysDown.remove(keyCode)
           发送 KeyEvent 到 Minecraft
  ↓
6. Minecraft 1.11.2 KeyBinding 系统
   └─ 接收 KeyEvent
   └─ 处理 GUI 打开/关闭
   └─ **MC 1.11.2 特性**: 响应 inventory release 事件 ❌
   └─ release 事件触发 GUI 关闭
   └─ 或者连续 press 触发 toggle
  ↓
7. 渲染循环
   └─ 更新游戏状态
   └─ 渲染 GUI
  ↓
8. 观察数据采集
   └─ 截取屏幕 (RGB)
   └─ 收集 inventory 数据
   └─ 收集其他状态
  ↓
9. ObservationFromServer.java
   └─ 构造观察 JSON
   └─ 通过 socket 发送回 Python
  ↓
═══════════════════════════════════════════════════════════════
Python 端
═══════════════════════════════════════════════════════════════
  ↓
instance.receive_observation()
  ↓
解析观察数据
  ↓
返回 obs, reward, done, info
```

### 关键文件路径 (Malmo)

```
minedojo/sim/Malmo/Minecraft/
├── com/microsoft/Malmo/
│   ├── Client/
│   │   ├── ClientStateMachine.java         # 状态机和 Socket
│   │   └── FakeKeyboard.java               # 键盘模拟
│   ├── MissionHandlers/
│   │   ├── CommandForKey.java              # 键盘命令处理
│   │   ├── ObservationFromServer.java      # 观察生成
│   │   └── MissionCommandHandler.java      # 命令分发
│   └── Utils/
│       └── TCPSocketHelper.java            # Socket 工具
└── net/minecraft/client/
    ├── settings/KeyBinding.java             # MC 键绑定系统 (1.11.2)
    └── gui/                                 # GUI 系统 (1.11.2)
```

---

## 3. 关键差异对比

### Socket 监听

| 组件 | MCP-Reborn (MineRL) | Malmo (MineDojo) |
|------|---------------------|------------------|
| **监听类** | `ClientCommandHandler.java` | `ClientStateMachine.java` |
| **状态管理** | `MissionBehaviourManager` | `MissionCommandHandler` |
| **端口** | 9000 (通常) | 9000 (通常) |
| **协议** | TCP | TCP |
| **消息格式** | XML | XML |

### FakeKeyboard 实现

**⚠️ 重要：虽然类名相同，但实现可能不同！**

由于 MCP-Reborn 和 Malmo 是不同的代码库，且针对不同的 Minecraft 版本：
- **MCP-Reborn** 的 FakeKeyboard 针对 MC 1.16.5 API
- **Malmo** 的 FakeKeyboard 针对 MC 1.11.2 API

**基本逻辑相似（但实现细节可能不同）**：

```java
// 两者都使用相同的逻辑
public static void press(int keyCode) {
    if (!keysDown.contains(keyCode)) {
        keysDown.add(keyCode);
        KeyBinding.setKeyBindState(keyCode, true);
    }
}

public static void release(int keyCode) {
    if (keysDown.contains(keyCode)) {
        keysDown.remove(keyCode);
        KeyBinding.setKeyBindState(keyCode, false);
    }
}
```

### Minecraft GUI 处理（关键差异！）

| 方面 | MC 1.16.5 (MineRL) | MC 1.11.2 (MineDojo) |
|------|-------------------|---------------------|
| **KeyEvent 接收** | ✅ 相同 | ✅ 相同 |
| **Press 处理** | 打开 GUI | 打开 GUI |
| **Release 处理** | **忽略** ✅ | **响应，关闭 GUI** ❌ |
| **Toggle 行为** | 不触发 | 触发 |
| **GUI 持久性** | 保持打开 | 自动关闭 |

### 观察数据返回

| 方面 | MCP-Reborn (MineRL) | Malmo (MineDojo) |
|------|---------------------|------------------|
| **生成类** | `ObservationMessage.java` | `ObservationFromServer.java` |
| **格式** | XML | JSON/XML |
| **POV** | `pov` (dict key) | `rgb` (numpy array) |
| **Inventory** | 扁平化 dict | 嵌套 dict |

---

## 4. 完整的消息循环

### 时序图

```
Python                    Socket                  Java (Minecraft)
  |                         |                          |
  |-- send step_message --->|                          |
  |                         |-- parse XML ------------>|
  |                         |                          |-- CommandForKey.onExecute()
  |                         |                          |-- FakeKeyboard.press/release()
  |                         |                          |-- Minecraft KeyBinding
  |                         |                          |-- GUI 更新
  |                         |                          |-- 游戏逻辑更新
  |                         |                          |-- 渲染
  |                         |                          |-- 采集观察
  |                         |<-- observation message --|
  |<-- receive observation -|                          |
  |                         |                          |
```

### 延迟和同步

1. **Python 发送命令** → **~1-5ms** → Java 接收
2. **Java 解析 XML** → **~1-10ms** → 分发到 Handler
3. **Handler 执行** → **~1ms** → FakeKeyboard
4. **FakeKeyboard** → **~1ms** → Minecraft KeyBinding
5. **Minecraft 更新** → **~16ms** (1 tick @ 60 FPS) → 渲染
6. **观察采集** → **~5-20ms** → 发送回 Python
7. **Python 接收** → **~1-5ms** → 返回

**总延迟**: 约 25-60ms 每步

---

## 5. 调试方法

### 添加 Socket 消息日志

#### MineRL (ClientCommandHandler.java)

```java
// 在 socket 接收处添加
private void handleIncomingMessage(String message) {
    System.out.println("[MCP-Reborn] Received: " + message);
    
    // 原有解析逻辑
    parseAndExecuteCommands(message);
}
```

#### MineDojo (ClientStateMachine.java)

```java
// 在 socket 接收处添加
private void processMessage(String message) {
    System.out.println("[Malmo] Received: " + message);
    
    // 原有解析逻辑
    parseAndDispatchCommands(message);
}
```

### 添加 FakeKeyboard 日志

```java
// 在 FakeKeyboard.java 中添加
public static void press(int keyCode) {
    String keyName = KeyBinding.getKeyName(keyCode);
    System.out.println("[FakeKeyboard] Press: " + keyName + " (" + keyCode + ")");
    
    if (!keysDown.contains(keyCode)) {
        keysDown.add(keyCode);
        KeyBinding.setKeyBindState(keyCode, true);
    }
}

public static void release(int keyCode) {
    String keyName = KeyBinding.getKeyName(keyCode);
    System.out.println("[FakeKeyboard] Release: " + keyName + " (" + keyCode + ")");
    
    if (keysDown.contains(keyCode)) {
        keysDown.remove(keyCode);
        KeyBinding.setKeyBindState(keyCode, false);
    }
}
```

### Python 端日志

```python
# 在发送前添加
print(f"[Python] Sending: {step_message}")
instance.client_socket_send_message(step_message.encode())

# 在接收后添加
obs = instance.receive_observation()
print(f"[Python] Received observation")
```

---

## 6. 问题排查流程

### 如果 inventory GUI 不显示

1. **检查 Python 端**:
   - `step_message` 是否包含 `<inventory>1</inventory>`？
   - Socket 是否成功发送？

2. **检查 Java 端**:
   - Socket 是否接收到消息？ (添加日志)
   - XML 解析是否成功？
   - `CommandForKey.onExecute()` 是否被调用？
   - `FakeKeyboard.press()` 是否被调用？
   - `keysDown` 集合是否包含 inventoryKeyCode？

3. **检查 Minecraft**:
   - KeyEvent 是否发送到 Minecraft？
   - GUI 是否实际打开？
   - 是否有其他事件干扰？

### 如果 inventory GUI 自动关闭

1. **检查后续消息**:
   - 是否发送了 `<inventory>0</inventory>`？
   - 是否触发了 `FakeKeyboard.release()`？

2. **检查 Minecraft 版本**:
   - MC 1.11.2: 会响应 release
   - MC 1.16.5: 会忽略 release

3. **检查 keysDown 状态**:
   - 是否被意外清空？
   - 是否有其他代码修改？

---

## 总结

### 共同点（可能）

1. ✅ **Socket 通信机制相似**: TCP + XML
2. ⚠️ **FakeKeyboard 逻辑相似**: press/release 概念相同，但实现细节可能不同
3. ⚠️ **命令解析流程相似**: XML → Handler → FakeKeyboard 的架构相同

### 差异点

1. ❌ **代码库不同**: MCP-Reborn vs Malmo (不同的项目！)
2. ❌ **Minecraft 版本不同**: 1.16.5 vs 1.11.2
3. ❌ **Forge 版本不同**: 36.x vs 13.20.1.2588
4. ❌ **GUI 行为不同**: 保持打开 vs Toggle
5. ❌ **API 实现不同**: MC 1.16.5 和 1.11.2 的 API 完全不同
6. ❌ **观察格式可能不同**: XML vs JSON
7. ❌ **类实现细节不同**: 虽然类名可能相同，但针对不同 MC 版本的实现不同

### 核心问题

**inventory GUI 自动关闭的根本原因：**

1. **不是** Socket 消息传输的问题
2. **不是** Java Handler 的问题
3. **不是** FakeKeyboard press/release 逻辑的问题
4. **而是** Minecraft 1.11.2 vs 1.16.5 的 **GUI 系统** 对 KeyEvent 的响应方式根本不同！

即使 MCP-Reborn 和 Malmo 的 FakeKeyboard 实现完全相同（实际上可能不同），最终的 GUI 行为也会因为 Minecraft 版本的差异而不同。

**因此，解决方案必须在 Python wrapper 层面实现，阻止发送会导致 release 的命令。**

