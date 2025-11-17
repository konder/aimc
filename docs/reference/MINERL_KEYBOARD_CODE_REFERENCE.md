# MineRL 按键实现核心代码

## 1. KeyboardListener.java (MC 1.16.5)

**位置**: `/usr/local/Caskroom/miniforge/base/envs/minedojo-x86/lib/python3.9/site-packages/minerl/MCP-Reborn/src/main/java/net/minecraft/client/KeyboardListener.java`

**核心逻辑** (行 365-369):

```java
if (flag1) {
    KeyBinding.setKeyBindState(inputmappings$input, false);
} else {
    KeyBinding.setKeyBindState(inputmappings$input, true);
    KeyBinding.onTick(inputmappings$input);  // ← 关键！按下时调用 onTick
}
```

**关键点**:
- ✅ **按键按下时调用 `onTick()`**
- ✅ **状态驱动**: 设置后状态保持
- ✅ **使用 `InputMappings.Input` 对象** (MC 1.16.5 API)

---

## 2. KeyBinding.java (MC 1.16.5)

**位置**: `/usr/local/Caskroom/miniforge/base/envs/minedojo-x86/lib/python3.9/site-packages/minerl/MCP-Reborn/src/main/java/net/minecraft/client/settings/KeyBinding.java`

### onTick 方法

```java
public static void onTick(InputMappings.Input key) {
    KeyBinding keybinding = HASH.get(key);
    if (keybinding != null) {
        ++keybinding.pressTime;  // ← 增加按压计数器
    }
}
```

**作用**:
- 增加 `pressTime` 计数器
- 游戏逻辑根据 `pressTime` 来判断按键是否被触发

### setKeyBindState 方法

```java
public static void setKeyBindState(InputMappings.Input key, boolean held) {
    KeyBinding keybinding = HASH.get(key);
    if (keybinding != null) {
        keybinding.setPressed(held);  // ← 设置 pressed 状态
    }
}
```

**作用**:
- 设置 `pressed` 状态
- 游戏逻辑通过 `isPressed()` 检查状态

---

## 3. 与 MineDojo (Malmo MC 1.11.2) 的对比

### API 差异

| 项目 | MineRL (MC 1.16.5) | MineDojo (MC 1.11.2) |
|------|-------------------|---------------------|
| **参数类型** | `InputMappings.Input` 对象 | `int keyCode` |
| **方法签名** | `setKeyBindState(InputMappings.Input key, boolean held)` | `setKeyBindState(int keyCode, boolean held)` |
| **方法签名** | `onTick(InputMappings.Input key)` | `onTick(int keyCode)` |

### 行为差异

| 行为 | MineRL | MineDojo (修改前) | MineDojo (修改后) |
|------|--------|------------------|------------------|
| **调用 setKeyBindState** | ✅ Yes | ✅ Yes | ✅ Yes |
| **调用 onTick** | ✅ Yes | ❌ No | ✅ Yes |
| **inventory 保持打开** | ✅ Yes | ❌ No | ✅ Yes (预期) |

---

## 4. MineDojo 的修改（CommandForKey.java）

**修改前**:

```java
private static void setKeyBindingStateDirect(KeyBinding keyBinding, boolean pressed) {
    KeyBinding.setKeyBindState(keyBinding.getKeyCode(), pressed);
    // ❌ 没有调用 onTick
}
```

**修改后**:

```java
private static void setKeyBindingStateDirect(KeyBinding keyBinding, boolean pressed) {
    int keyCode = keyBinding.getKeyCode();
    
    // 1. 设置按键状态
    KeyBinding.setKeyBindState(keyCode, pressed);
    
    // 2. 如果是按下，调用 onTick（模拟 MineRL）
    if (pressed) {
        KeyBinding.onTick(keyCode);  // ✅ 新增！
        System.out.println("[CommandForKey] Direct state set + onTick: " 
            + keyBinding.getKeyDescription() + " (keyCode=" + keyCode + ")");
    }
}
```

---

## 5. 验证方法

### 检查 Java 日志

```bash
grep "Direct state set + onTick" /tmp/test.log
```

**预期输出**:
```
[CommandForKey] Direct state set + onTick: key.inventory (keyCode=18)
```

### 观察游戏行为

1. **Step 1**: 发送 `inventory=1` → GUI 打开
2. **Step 2-N**: 发送其他命令 → GUI 保持打开 ✅

---

## 6. 核心发现总结

### MineRL 的关键机制

1. **状态驱动**: 
   - 按键按下后，状态保持为 `pressed=true`
   - 不需要持续发送按键命令

2. **onTick 触发**: 
   - 每次按下调用 `onTick()`
   - 增加 `pressTime` 计数器
   - 游戏逻辑根据 `pressTime > 0` 来处理

3. **原生 API 集成**: 
   - 直接使用 MC 1.16.5 的 `KeyboardListener` 和 `KeyBinding`
   - 与原生 Minecraft 行为完全一致

### MineDojo 缺失的环节

- ❌ **未调用 `onTick()`**: 导致 `pressTime` 不增加
- ❌ **事件驱动**: `FakeKeyboard` 使用 press/release 成对事件
- ❌ **inventory 自动关闭**: 因为 release 事件触发了关闭逻辑

### 修改后的 MineDojo

- ✅ **调用 `onTick()`**: 模拟 MineRL 行为
- ✅ **状态驱动**: 对 `inventory` 使用状态驱动
- ✅ **预期一致性**: inventory GUI 应该保持打开

---

## 7. 相关文件路径

### MineRL (MCP-Reborn)

```
/usr/local/Caskroom/miniforge/base/envs/minedojo-x86/lib/python3.9/site-packages/minerl/
└── MCP-Reborn/src/main/java/
    └── net/minecraft/client/
        ├── KeyboardListener.java     ← 核心：按下时调用 onTick
        └── settings/KeyBinding.java  ← 核心：管理 pressed + pressTime
```

### MineDojo (Malmo)

```
/usr/local/Caskroom/miniforge/base/envs/minedojo-x86/lib/python3.9/site-packages/minedojo/
└── sim/Malmo/Minecraft/src/main/java/
    └── com/microsoft/Malmo/MissionHandlers/
        └── CommandForKey.java        ← 已修改：添加 onTick 调用
```

---

## 8. 下一步

1. **运行测试脚本**，验证 `onTick` 是否被调用
2. **观察 Java 日志**，确认输出 `"Direct state set + onTick"`
3. **观察游戏窗口**，验证 inventory GUI 是否保持打开
4. **保存截图**，记录每个 step 的 GUI 状态

如果一切正常，MineDojo 的 `inventory` 行为应该与 MineRL 完全一致！✅


