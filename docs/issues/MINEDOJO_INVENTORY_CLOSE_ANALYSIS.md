# MineDojo Inventory 关闭问题分析报告

**日期**: 2025-11-15  
**问题**: MineDojo 环境中 `inventory 0` 会关闭物品栏，而 MineRL 不会

---

## 🔍 问题现象

### MineRL 行为
```python
action['inventory'] = 1  # Step 2: 打开物品栏 ✓
action['inventory'] = 0  # Step 3+: 物品栏保持打开 ✓
```

### MineDojo 行为
```python
action['inventory'] = 1  # Step 2: 打开物品栏 ✓
action['inventory'] = 0  # Step 3+: 物品栏立即关闭 ❌
```

### 命令日志（两者完全一致）
```
[DEBUG] inventory action: x=1, cmd='inventory 1'  # 两者都发送
[DEBUG] inventory action: x=0, cmd='inventory 0'  # 两者都发送
```

---

## 🔬 源码分析

### 1. Java 层代码（Malmo）

#### CommandForKey.onExecute()
**路径**: 
- MineDojo: `minedojo/sim/Malmo/Minecraft/.../CommandForKey.java`
- MineRL: `minerl/Malmo/Minecraft/.../CommandForKey.java`

**代码**（两者完全相同）:
```java
public boolean onExecute(String verb, String parameter, MissionInit missionInit) {
    if (verb != null && verb.equalsIgnoreCase(keyHook.getCommandString())) {
        if (parameter != null && parameter.equalsIgnoreCase(DOWN_COMMAND_STRING)) {
            FakeKeyboard.press(keyHook.getKeyCode());  // inventory 1
        } else if (parameter != null && parameter.equalsIgnoreCase(UP_COMMAND_STRING)) {
            FakeKeyboard.release(keyHook.getKeyCode());  // inventory 0
        }
        return true;
    }
    return false;
}
```

#### FakeKeyboard.press() / release()
**路径**:
- MineDojo: `minedojo/sim/Malmo/Minecraft/.../FakeKeyboard.java`
- MineRL: `minerl/Malmo/Minecraft/.../FakeKeyboard.java`

**代码**（两者完全相同）:
```java
public static void press(int key) {
    if (!keysDown.contains(key)) {  // 只有键未按下时才触发
        add(new FakeKeyEvent(' ', key, true));
    }
}

public static void release(int key) {
    if (keysDown.contains(key)) {  // 只有键已按下时才触发释放
        add(new FakeKeyEvent(' ', key, false));
    }
}

public static void add(FakeKeyEvent event) {
    eventQueue.add(event);
    if (event.state) {
        keysDown.add(event.key);    // 记录按下状态
    } else {
        keysDown.remove(event.key); // 移除按下状态
    }
}
```

**关键发现**:
- `inventory 1` → `press()` → 如果键未按下，添加 `FakeKeyEvent(true)` → `keysDown.add(key)`
- `inventory 0` → `release()` → **如果键已按下（在 `keysDown` 中），添加 `FakeKeyEvent(false)`** → `keysDown.remove(key)`

### 2. Python 层代码

#### MineDojo action.py
```python
def to_hero(self, x):
    cmd = ""
    cmd += "{} {}".format(self.command, adjective)  # ✅ 不过滤 0
    return cmd
```

#### MineRL action.py
```python
def to_hero(self, x):
    cmd = ""
    cmd += "{} {}".format(self.command, adjective)  # ✅ 不过滤 0
    return cmd
```

**结论**: Python 层代码一致，都发送 `"inventory 0"` 命令。

---

## 💡 问题根源假设

### 假设 1: 状态管理差异

MineRL 和 MineDojo 的 Java 代码完全相同，但行为不同，可能的原因：

1. **`keysDown` 状态不同步**
   - MineDojo: `keysDown` 正确追踪了 `inventory 1` 的按下状态
   - MineRL: `keysDown` 可能没有正确追踪，导致 `release()` 不执行

2. **Minecraft 事件循环差异**
   - MineDojo 的 Minecraft 版本/配置可能更"敏感"，立即处理所有键盘事件
   - MineRL 的版本/配置可能有延迟或过滤机制

3. **GUI 状态检测**
   - Minecraft 的 GUI 系统可能检测到物品栏已打开
   - 当收到 `FakeKeyEvent(false)` 时，GUI 触发关闭逻辑

### 假设 2: Minecraft 版本差异

- MineRL: Minecraft 1.16.5
- MineDojo: Minecraft 1.16.5（理论上相同，但可能有配置差异）

可能的差异点：
- 键盘事件处理优先级
- GUI 渲染循环
- Forge Mod 加载顺序

---

## 🧪 验证方法

### 方法 1: 检查 `keysDown` 状态

修改 `FakeKeyboard.java` 添加详细日志：

```java
public static void release(int key) {
    System.out.println("[DEBUG] release() called for key: " + key);
    System.out.println("[DEBUG] keysDown.contains(" + key + "): " + keysDown.contains(key));
    if (keysDown.contains(key)) {
        System.out.println("[DEBUG] Adding release event");
        add(new FakeKeyEvent(' ', key, false));
    } else {
        System.out.println("[DEBUG] Skipping release (key not down)");
    }
}
```

**预期**:
- MineRL: `release()` 调用但 `keysDown.contains()` 返回 `false`，跳过释放
- MineDojo: `release()` 调用且 `keysDown.contains()` 返回 `true`，执行释放

### 方法 2: 强制跳过 `inventory 0`

在 Python 层过滤：

```python
def to_hero(self, x):
    if self.command == "inventory" and adjective == "0":
        return ""  # 不发送 inventory 0
    cmd = ""
    cmd += "{} {}".format(self.command, adjective)
    return cmd
```

---

## ✅ 解决方案

### 方案 A: Python 层过滤（推荐）

修改 MineDojo 的 `action.py`:

```python
def to_hero(self, x):
    # ... (existing logic to get adjective) ...
    
    cmd = ""
    # 特殊处理：inventory 只在 1 时发送，0 时不发送
    if self.command == "inventory" and adjective == "0":
        return ""
    
    cmd += "{} {}".format(self.command, adjective)
    return cmd
```

**优点**:
- 简单，不需要修改 Java 代码
- 只影响 inventory 动作
- 与 MineRL 行为一致

**缺点**:
- 治标不治本，没有解决根本原因

---

### 方案 B: Java 层修复（彻底）

修改 `CommandForKey.java` 的 `onExecute()`:

```java
public boolean onExecute(String verb, String parameter, MissionInit missionInit) {
    if (verb != null && verb.equalsIgnoreCase(keyHook.getCommandString())) {
        // 特殊处理：inventory 只响应按下（1），忽略释放（0）
        if (verb.equals("inventory")) {
            if (parameter != null && parameter.equalsIgnoreCase(DOWN_COMMAND_STRING)) {
                FakeKeyboard.press(keyHook.getKeyCode());
            }
            // 忽略 UP_COMMAND_STRING (0)
            return true;
        }
        
        // 其他键的正常处理
        if (parameter != null && parameter.equalsIgnoreCase(DOWN_COMMAND_STRING)) {
            FakeKeyboard.press(keyHook.getKeyCode());
        } else if (parameter != null && parameter.equalsIgnoreCase(UP_COMMAND_STRING)) {
            FakeKeyboard.release(keyHook.getKeyCode());
        } else {
            return false;
        }
        return true;
    }
    return false;
}
```

**优点**:
- 从根源解决问题
- 完全控制 inventory 行为

**缺点**:
- 需要重新编译 Minecraft Mod
- 需要分发修改后的 JAR 文件

---

### 方案 C: 状态管理（中间方案）

在 Wrapper 中维护 inventory 状态：

```python
class MineDojoBiomeWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._inventory_open = False  # 追踪物品栏状态
    
    def _convert_action_to_minedojo(self, minerl_action: Dict) -> np.ndarray:
        # ... (existing code) ...
        
        # Functional (index 5)
        if minerl_action.get('inventory', 0):
            if not self._inventory_open:
                minedojo_action[5] = 8  # 只在未打开时发送打开命令
                self._inventory_open = True
        else:
            # inventory=0 时不发送任何命令
            pass
        
        # ... (rest of code) ...
```

**优点**:
- 不需要修改 MineDojo 源码
- 在 Wrapper 层面控制

**缺点**:
- 需要维护额外状态
- 可能与实际 GUI 状态不同步

---

## 🎯 推荐方案

**短期**: 方案 A（Python 层过滤）
- 修改 `minedojo/sim/handlers/agent/action.py`
- 在 `docker/minedojo_inventory.patch` 中添加此修改

**长期**: 方案 B（Java 层修复）
- 提交 PR 到 MineDojo 仓库
- 或在项目中维护自定义 Minecraft Mod

---

## 📝 后续行动

1. ✅ 验证方法 1（添加日志）确认 `keysDown` 状态
2. ⏳ 实施方案 A（Python 层过滤）
3. ⏳ 更新 `minedojo_inventory.patch`
4. ⏳ 测试验证

---

**维护者**: AIMC 项目团队  
**最后更新**: 2025-11-15


