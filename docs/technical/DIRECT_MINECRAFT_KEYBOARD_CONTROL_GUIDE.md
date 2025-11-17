# 直接操作 Minecraft 客户端键盘状态 - 技术方案

**日期**: 2025-11-16  
**目标**: 绕过 Malmo 的 `FakeKeyboard`，直接向 MC 客户端发送键盘状态

---

## 🎯 核心思路

**借鉴 MCP-Reborn 的成功经验**，使用**状态驱动**而非**事件驱动**：

```
Python: inventory=1
    ↓
Malmo 接收命令
    ↓
跳过 FakeKeyboard.press(18)  ← 事件驱动（会触发 release）
    ↓
直接设置 KeyboardListener.State(["key.keyboard.e"])  ← 状态驱动
    ↓
Minecraft 看到 E 键在按下状态 → 打开 GUI
    ↓
Python: inventory=0
    ↓
直接设置 KeyboardListener.State([])  ← E 键不在列表中
    ↓
Minecraft 看到 E 键不在按下状态 → 但 GUI 保持打开 ✅
```

---

## 📋 实施步骤

### 步骤 1: 修改 Minecraft 的 `KeyboardListener`

**文件位置**:
```
/usr/local/Caskroom/miniforge/base/envs/minedojo-x86/lib/python3.9/site-packages/minedojo/sim/Malmo/Minecraft/build/sources/main/java/net/minecraft/client/KeyboardListener.java
```

**添加代码**:

```java
package net.minecraft.client;

import net.minecraft.client.settings.KeyBinding;
import java.util.Collection;
import java.util.HashSet;
import java.util.Set;

public class KeyboardListener {
    
    // 现有代码...
    
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // 新增：外部状态设置（MCP-Reborn 风格）
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    /**
     * 键盘状态（包含当前按下的键）
     */
    public static class State {
        public final Set<String> keys;      // 当前按下的键
        public final Set<String> newKeys;   // 新按下的键
        public final String chars;          // 字符输入
        
        public State(Collection<String> keys, Collection<String> newKeys, String chars) {
            this.keys = new HashSet<>();
            this.keys.addAll(keys);
            this.newKeys = new HashSet<>();
            this.newKeys.addAll(newKeys);
            this.chars = chars;
        }
    }
    
    /**
     * 直接设置键盘状态（供 Malmo 调用）
     * 
     * @param state 新的键盘状态
     */
    public void setExternalState(State state) {
        System.out.println("[KeyboardListener] Setting external state: " + state.keys);
        
        // 更新所有键绑定的状态
        for (String keyName : state.keys) {
            // 查找对应的键绑定
            InputMappings.Input key = InputMappings.getInputByName(keyName);
            if (key != null) {
                KeyBinding.setKeyBindState(key, true);
                System.out.println("[KeyboardListener]   Set key: " + keyName + " = true");
            }
        }
        
        // 释放不在列表中的键
        // （如果需要的话，取决于 MC 1.11.2 的行为）
    }
}
```

### 步骤 2: 修改 Malmo 的 `CommandForKey`

**文件位置**:
```
/usr/local/Caskroom/miniforge/base/envs/minedojo-x86/lib/python3.9/site-packages/minedojo/sim/Malmo/Minecraft/build/sources/main/java/com/microsoft/Malmo/MissionHandlers/CommandForKey.java
```

**修改 `execute` 方法**:

```java
package com.microsoft.Malmo.MissionHandlers;

import net.minecraft.client.KeyboardListener;
import net.minecraft.client.Minecraft;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.Set;

public class CommandForKey extends CommandBase {
    
    // 跟踪当前按下的键（跨所有 CommandForKey 实例共享）
    private static Set<String> currentPressedKeys = new HashSet<>();
    
    // 动作到键名的映射
    private static final Map<String, String> ACTION_TO_KEY = new HashMap<String, String>() {{
        put("forward", "key.keyboard.w");
        put("back", "key.keyboard.s");
        put("left", "key.keyboard.a");
        put("right", "key.keyboard.d");
        put("inventory", "key.keyboard.e");
        put("jump", "key.keyboard.space");
        put("sneak", "key.keyboard.left.shift");
        put("drop", "key.keyboard.q");
    }};
    
    @Override
    public boolean execute(String verb, String parameter) {
        String keyName = ACTION_TO_KEY.get(verb.toLowerCase());
        
        if (keyName != null) {
            boolean pressed = !parameter.equalsIgnoreCase("0");
            
            System.out.println("[CommandForKey] Direct mode: " + verb + " → " + keyName + " = " + pressed);
            
            // 更新当前按下的键集合
            if (pressed) {
                currentPressedKeys.add(keyName);
            } else {
                currentPressedKeys.remove(keyName);
            }
            
            // 创建新的键盘状态
            KeyboardListener.State newState = new KeyboardListener.State(
                new ArrayList<>(currentPressedKeys),
                new ArrayList<>(),
                ""
            );
            
            // 直接设置到 Minecraft
            Minecraft mc = Minecraft.getInstance();
            if (mc != null && mc.keyboardListener != null) {
                mc.keyboardListener.setExternalState(newState);
            }
            
            return true;
        }
        
        // 回退到原始 FakeKeyboard 方式（用于未映射的动作）
        return super.execute(verb, parameter);
    }
}
```

### 步骤 3: 重新编译 Malmo JAR

```bash
cd /usr/local/Caskroom/miniforge/base/envs/minedojo-x86/lib/python3.9/site-packages/minedojo/sim/Malmo/Minecraft

# 备份原始 JAR
cp build/libs/MalmoMod-*.jar build/libs/MalmoMod-original.jar

# 清理并重新编译
./gradlew clean shadowJar

# 如果编译成功
echo "✓ 新 JAR 已生成"
```

### 步骤 4: 测试

创建测试脚本验证 inventory GUI 是否保持打开：

```python
import minedojo
import time

env = minedojo.make(task_id="open-ended", image_size=(160, 256))
obs = env.reset()

# Step 1: 打开 inventory
action = env.action_space.no_op()
action[5] = 8  # functional action: inventory
obs, _, _, _ = env.step(action)
env.render()
time.sleep(1)
print("Step 1: inventory=1 (打开)")

# Step 2-5: 发送 no-op (inventory=0)
for i in range(4):
    action = env.action_space.no_op()
    action[5] = 0  # no-op
    obs, _, _, _ = env.step(action)
    env.render()
    time.sleep(0.5)
    print(f"Step {i+2}: inventory=0 (应该保持打开)")

print("\n观察 GUI 是否保持打开？")
env.close()
```

---

## 🔧 关键技术细节

### 1. `KeyboardListener.State` 的结构

```java
State {
    keys: Set<String>,      // 当前按下的键列表（例如 ["key.keyboard.e"]）
    newKeys: Set<String>,   // 新按下的键（用于触发事件）
    chars: String           // 字符输入
}
```

### 2. 为什么这个方法有效？

**Malmo 的问题**:
```
inventory=0 → FakeKeyboard.release(18) → 触发 release 事件 → GUI 关闭
```

**直接状态管理**:
```
inventory=0 → keys 列表不包含 "key.keyboard.e" → Minecraft 只看到键不再按下 → 不触发任何事件 → GUI 保持
```

### 3. MC 1.11.2 vs MC 1.16.5

虽然 `KeyboardListener.State` 是 MCP-Reborn (MC 1.16.5) 的实现，但：
- **MC 1.11.2 也有类似的键盘状态管理机制**
- 可能需要适配 MC 1.11.2 的 API
- 核心思路相同：**状态驱动 > 事件驱动**

---

## ⚠️ 潜在问题

### 1. MC 1.11.2 的 API 差异

- `KeyboardListener` 的类结构可能不同
- 需要查阅 MC 1.11.2 的反编译代码

### 2. Malmo 的其他组件依赖 `FakeKeyboard`

- 如果完全禁用 `FakeKeyboard`，可能影响其他功能
- 建议：**只对 `inventory` 使用直接状态管理，其他动作保持 `FakeKeyboard`**

### 3. 编译错误

- MC 1.11.2 的 `InputMappings` 可能不存在
- 需要找到等效的类/方法

---

## 🎯 简化方案（推荐）

如果完全修改 Malmo 太复杂，可以考虑：

### **只针对 `inventory` 命令特殊处理**

在 `CommandForKey.java` 中：

```java
@Override
public boolean execute(String verb, String parameter) {
    // 特殊处理 inventory
    if (verb.equalsIgnoreCase("inventory")) {
        boolean pressed = !parameter.equalsIgnoreCase("0");
        
        if (pressed) {
            // 只在按下时调用 FakeKeyboard.press
            FakeKeyboard.press(18);
        }
        // 不调用 FakeKeyboard.release(18)，让键保持按下状态
        
        return true;
    }
    
    // 其他命令使用原始逻辑
    return super.execute(verb, parameter);
}
```

**效果**:
- `inventory=1` → `press(18)` → GUI 打开
- `inventory=0` → 什么都不做 → GUI 保持打开 ✅

---

## 📊 对比总结

| 方案 | 复杂度 | 效果 | 风险 |
|------|--------|------|------|
| **完全状态驱动**（修改 KeyboardListener） | ⭐⭐⭐⭐⭐ | ✅✅✅ | ⚠️⚠️⚠️ API 差异 |
| **混合方式**（只修改 inventory） | ⭐⭐⭐ | ✅✅ | ⚠️ 其他功能可能受影响 |
| **简化方案**（禁用 inventory release） | ⭐ | ✅ | ⚠️ 最小 |

---

## ✅ 结论

1. **理论上可行**：直接操作 MC 客户端键盘状态是可行的
2. **参考 MCP-Reborn**：它成功实现了这个机制
3. **推荐简化方案**：先尝试只禁用 `inventory` 的 `release` 调用
4. **如果需要完整方案**：需要深入研究 MC 1.11.2 的键盘 API

你想先尝试哪个方案？🤔


