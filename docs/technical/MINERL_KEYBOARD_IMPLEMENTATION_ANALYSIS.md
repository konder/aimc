# MineRL (MCP-Reborn) 按键实现机制分析

## 目录
1. [整体架构](#整体架构)
2. [核心文件列表](#核心文件列表)
3. [数据流转流程](#数据流转流程)
4. [关键代码分析](#关键代码分析)
5. [与 MineDojo (Malmo) 的对比](#与-minedojo-malmo-的对比)

---

## 整体架构

```
Python (minerl)
    ↓
MCP-Reborn Java Server (EnvServer)
    ↓
KeyboardListener (MC 1.16.5 原生)
    ↓
KeyBinding.setKeyBindState() + onTick()
    ↓
Minecraft 游戏逻辑
```

---

## 核心文件列表

### 1. Python 层面

#### `/usr/local/Caskroom/miniforge/base/envs/minedojo-x86/lib/python3.9/site-packages/minerl/herobraine/hero/mc.py`
- **作用**: MineRL 的主要 Python 接口，处理动作转换
- **关键方法**: `_to_hero(action)` - 将 Gym action 转换为 MineRL Hero 格式

#### `/usr/local/Caskroom/miniforge/base/envs/minedojo-x86/lib/python3.9/site-packages/minerl/herobraine/hero/handlers/agent/actions/keyboard.py`
- **作用**: 定义键盘动作的处理器
- **关键类**: `KeybasedCommandAction` - 处理 WASD、跳跃、潜行等键盘动作

---

### 2. Java 层面（MCP-Reborn）

#### `/usr/local/Caskroom/miniforge/base/envs/minedojo-x86/lib/python3.9/site-packages/minerl/MCP-Reborn/src/main/java/com/minerl/multiagent/env/EnvServer.java`
- **作用**: MCP-Reborn 的核心服务器，接收 Python 发送的动作命令
- **关键方法**: 
  - `handleMessage(String message)` - 处理来自 Python 的消息
  - 解析 JSON 格式的动作命令

#### `/usr/local/Caskroom/miniforge/base/envs/minedojo-x86/lib/python3.9/site-packages/minerl/MCP-Reborn/src/main/java/net/minecraft/client/KeyboardListener.java`
- **作用**: Minecraft 1.16.5 原生的键盘监听器（被 MCP-Reborn 修改）
- **关键方法**: 
  - `onKey(long window, int key, int scancode, int action, int modifiers)` - 处理按键事件
  - **核心机制**: 使用 `KeyBinding.setKeyBindState()` + `KeyBinding.onTick()`

#### `/usr/local/Caskroom/miniforge/base/envs/minedojo-x86/lib/python3.9/site-packages/minerl/MCP-Reborn/src/main/java/net/minecraft/client/settings/KeyBinding.java`
- **作用**: MC 1.16.5 的按键绑定管理
- **关键方法**:
  - `setKeyBindState(InputMappings.Input key, boolean held)` - 设置按键状态
  - `onTick(InputMappings.Input key)` - 增加 `pressTime` 计数器
  - `setPressed(boolean pressed)` - 内部方法，实际设置 pressed 状态

---

## 数据流转流程

### 完整流程（以 `forward` 为例）

```
Step 1: Python 发送动作
    action = {'forward': 1}
    ↓
    minerl/herobraine/hero/mc.py: _to_hero(action)
    ↓
    生成 JSON: {"forward": 1}
    ↓
    通过 socket 发送到 MCP-Reborn

Step 2: MCP-Reborn 接收并处理
    EnvServer.handleMessage(json_string)
    ↓
    解析 JSON，识别 "forward" 动作
    ↓
    查找对应的 KeyBinding (mc.gameSettings.keyBindForward)
    ↓
    调用键盘处理逻辑

Step 3: 设置 Minecraft 按键状态
    KeyboardListener.onKey(...)
    ↓
    InputMappings.Input input = InputMappings.Type.KEYSYM.getOrMakeInput(keyCode)
    ↓
    KeyBinding.setKeyBindState(input, true)  // 设置按下状态
    ↓
    KeyBinding.onTick(input)  // 触发 pressTime++
    ↓
    KeyBinding 内部: setPressed(true)

Step 4: Minecraft 游戏逻辑读取
    Minecraft.tick()
    ↓
    检查 KeyBinding.isPressed()
    ↓
    执行相应的游戏逻辑（如前进移动）
```

---

## 关键代码分析

### 1. MineRL KeyboardListener.java 核心代码

```java
// 位置: net/minecraft/client/KeyboardListener.java (MC 1.16.5)
// 行号: ~485-504

public void onKey(long window, int key, int scancode, int action, int modifiers) {
    // ... 省略前面的代码 ...
    
    if (key != InputMappings.INPUT_INVALID.getKeyCode()) {
        InputMappings.Input inputmappings$input = InputMappings.Type.KEYSYM.getOrMakeInput(key);
        
        if (action == GLFW.GLFW_PRESS) {
            // 按键按下
            KeyBinding.setKeyBindState(inputmappings$input, true);
            KeyBinding.onTick(inputmappings$input);  // ← 关键！触发 onTick
        } else if (action == GLFW.GLFW_RELEASE) {
            // 按键释放
            KeyBinding.setKeyBindState(inputmappings$input, false);
        }
    }
}
```

**关键点**:
- ✅ **按下时调用 `onTick`**：这是触发游戏逻辑的关键
- ✅ **使用 `InputMappings.Input` 对象**：MC 1.16.5 的新 API
- ✅ **状态驱动**：按下后状态保持，直到显式释放

---

### 2. MineRL KeyBinding.java 核心代码

```java
// 位置: net/minecraft/client/settings/KeyBinding.java (MC 1.16.5)

public static void onTick(InputMappings.Input key) {
    KeyBinding keybinding = HASH.get(key);
    if (keybinding != null) {
        ++keybinding.pressTime;  // ← 增加按压计数器
    }
}

public static void setKeyBindState(InputMappings.Input key, boolean held) {
    KeyBinding keybinding = HASH.get(key);
    if (keybinding != null) {
        keybinding.setPressed(held);  // ← 设置 pressed 状态
    }
}

protected void setPressed(boolean pressed) {
    this.pressed = pressed;
}

public boolean isPressed() {
    return this.pressed;
}
```

**关键点**:
- `pressTime`：记录按键被触发的次数
- `pressed`：当前按键状态（true=按下，false=释放）
- 游戏逻辑通过 `isPressed()` 检查按键状态

---

### 3. Minecraft 游戏逻辑读取按键

```java
// 位置: net/minecraft/client/Minecraft.java (MC 1.16.5)
// 游戏主循环

public void tick() {
    // ...
    if (this.gameSettings.keyBindForward.isPressed()) {
        // 执行前进逻辑
        this.player.movementInput.moveForward = 1.0F;
    }
    
    if (this.gameSettings.keyBindInventory.isKeyDown()) {
        // 打开/关闭物品栏
        this.displayGuiScreen(new InventoryScreen(this.player));
    }
    // ...
}
```

---

## 与 MineDojo (Malmo) 的对比

| 对比项 | MineRL (MC 1.16.5) | MineDojo (MC 1.11.2 Malmo) |
|--------|-------------------|---------------------------|
| **架构** | 状态驱动 | 事件驱动 |
| **按键处理** | `KeyboardListener` 直接操作 | `CommandForKey` + `FakeKeyboard` |
| **API** | `InputMappings.Input` 对象 | `int keyCode` |
| **onTick 调用** | ✅ 按下时调用 `KeyBinding.onTick()` | ❌ 未调用 `KeyBinding.onTick()` |
| **状态保持** | ✅ 状态驱动，按下后保持 | ❌ 事件驱动，press/release 成对 |
| **inventory 行为** | ✅ 打开后保持 | ❌ 自动关闭 |

---

## MineDojo 需要的修改

### 当前问题
MineDojo 的 `CommandForKey.java` 调用了 `KeyBinding.setKeyBindState(keyCode, true)`，但**没有调用 `KeyBinding.onTick(keyCode)`**，导致：
- `pressTime` 没有增加
- 游戏逻辑可能无法正确触发

### 解决方案（已实现）

```java
// 位置: CommandForKey.java (MineDojo/Malmo)

private static void setKeyBindingStateDirect(KeyBinding keyBinding, boolean pressed) {
    try {
        int keyCode = keyBinding.getKeyCode();
        
        // 1. 设置按键状态
        KeyBinding.setKeyBindState(keyCode, pressed);
        
        // 2. 如果是按下，调用 onTick（模拟 MineRL）
        if (pressed) {
            KeyBinding.onTick(keyCode);  // ← 新增！
            System.out.println("[CommandForKey] Direct state set + onTick: " 
                + keyBinding.getKeyDescription() + " (keyCode=" + keyCode + ")");
        } else {
            System.out.println("[CommandForKey] Direct state set: " 
                + keyBinding.getKeyDescription() + " = false (keyCode=" + keyCode + ")");
        }
    } catch (Exception e) {
        System.err.println("[CommandForKey] Failed to set key state directly: " + e.getMessage());
        e.printStackTrace();
    }
}
```

---

## 验证方法

### 1. 检查 Java 日志
```bash
grep "Direct state set + onTick" /tmp/test.log
```

预期输出:
```
[CommandForKey] Direct state set + onTick: key.inventory (keyCode=18)
```

### 2. 观察游戏行为
- **Step 1**: 发送 `inventory=1` → GUI 打开
- **Step 2**: 发送移动命令 → GUI 保持打开
- **Step 3**: 发送 `inventory=0` 或 no-op → GUI 仍然保持打开 ✅

---

## 总结

### MineRL 的核心机制
1. **状态驱动**: 按键按下后状态保持，不需要持续发送
2. **onTick 触发**: 每次按下都调用 `onTick()`，增加 `pressTime`
3. **原生 API**: 直接使用 MC 1.16.5 的 `KeyboardListener` 和 `KeyBinding`

### MineDojo 的修改要点
1. ✅ 绕过 `FakeKeyboard`，直接调用 `KeyBinding.setKeyBindState()`
2. ✅ **新增 `KeyBinding.onTick()` 调用**（关键！）
3. ✅ 对 `inventory` 使用状态驱动，不发送 `inventory=0`

---

## 参考文件路径

### MineRL (MCP-Reborn)
```
/usr/local/Caskroom/miniforge/base/envs/minedojo-x86/lib/python3.9/site-packages/minerl/
├── MCP-Reborn/src/main/java/
│   ├── com/minerl/multiagent/env/EnvServer.java
│   └── net/minecraft/client/
│       ├── KeyboardListener.java    ← 核心：处理按键，调用 onTick
│       └── settings/KeyBinding.java ← 核心：管理按键状态
└── herobraine/hero/
    ├── mc.py                         ← Python 接口
    └── handlers/agent/actions/
        └── keyboard.py               ← 键盘动作定义
```

### MineDojo (Malmo)
```
/usr/local/Caskroom/miniforge/base/envs/minedojo-x86/lib/python3.9/site-packages/minedojo/
└── sim/Malmo/Minecraft/src/main/java/
    └── com/microsoft/Malmo/MissionHandlers/
        ├── CommandForKey.java        ← 核心：需要修改，添加 onTick
        └── FakeKeyboard.java         ← 原机制：事件驱动（被绕过）
```

---

## 下一步验证

1. **运行测试脚本**，观察 Java 日志中是否出现 `"Direct state set + onTick"`
2. **观察游戏窗口**，验证 inventory GUI 是否保持打开
3. **保存 POV 截图**，确认每个 step 的 GUI 状态

如果 `onTick` 调用成功，MineDojo 的 `inventory` 行为应该与 MineRL 一致！✅


