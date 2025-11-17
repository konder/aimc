# MineDojo Inventory 行为分析 - Press+Release 假设

## 用户观察结果

### 测试 1：连续发送 action[5]=8

```
Step 7:  [0 0 0 12 12 8 0 0]  → 显示 ✅
Step 8:  [0 0 0 12 12 8 0 0]  → 不显示 ❌
Step 9:  [0 0 0 12 12 8 0 0]  → 不显示 ❌
```

**关键观察**：如果是简单的 toggle，Step 9 应该显示，但实际上不显示！

### 测试 2：间隔发送 action[5]=8 和 action[5]=0

```
Step 22: [0 0 0 12 12 8 0 0]  → 显示 ✅
Step 23: [0 0 0 12 12 0 0 0]  → 不显示 ❌
Step 24: [0 0 0 12 12 8 0 0]  → 显示 ✅
Step 25: [0 0 0 12 12 0 0 0]  → 不显示 ❌
```

**关键观察**：action[5]=0 起到了"重置"的作用。

---

## 假设 1：每次 press() 自动产生 press+release 事件对

### 可能的实现

```java
// Malmo FakeKeyboard 可能的实现
public static void press(int keyCode) {
    if (!keysDown.contains(keyCode)) {
        // 发送 press 事件
        sendKeyEvent(keyCode, true);
        keysDown.add(keyCode);
        
        // ⚠️ 关键：在同一帧内自动发送 release？
        sendKeyEvent(keyCode, false);
        // 但不从 keysDown 移除，留给显式的 release() 调用
    }
    // 如果 keysDown 中已有 keyCode → 忽略
}
```

### 流程分析

**Step 7**: action[5]=8
- `keysDown` 初始为空
- `press(E_KEY)` 被调用
  - `!keysDown.contains(E_KEY)` → true
  - 发送 **press 事件** → MC GUI 打开 ✅
  - 发送 **release 事件** → MC GUI 关闭？
  - `keysDown.add(E_KEY)` → `keysDown = {E_KEY}`
- **但用户看到 GUI 显示**，说明什么？

可能的情况：
1. **MC 只响应 press，忽略 release**？
2. **press 和 release 事件之间有优先级**？
3. **GUI 在帧结束时才更新**，press 优先级更高？

**Step 8**: action[5]=8
- `keysDown = {E_KEY}`（Step 7 留下的）
- `press(E_KEY)` 被调用
  - `!keysDown.contains(E_KEY)` → **false**
  - **被忽略！不发送任何事件！**
- MC 没收到任何事件
- **用户看到 GUI 不显示**

这说明 Step 7 结束时，GUI 确实已经关闭了！
但为什么 Step 7 执行时用户看到的是"显示"？

**可能原因**：
- 用户观察的时机是 Step 7 执行**过程中**
- 此时 press 事件已触发，release 事件还未生效
- 或者 release 事件在下一帧才生效

**Step 9**: action[5]=8
- 同 Step 8，被忽略

---

## 假设 2：Malmo 在每帧结束时自动 release 所有 keysDown

### 可能的实现

```java
// Malmo 帧循环（伪代码）
public void tickFrame() {
    // 1. 处理来自 Python 的命令
    processCommands();  // 包含 press()/release() 调用
    
    // 2. 执行 Minecraft tick
    minecraft.tick();
    
    // 3. 帧结束时自动 release 所有按键
    for (int key : keysDown) {
        sendKeyEvent(key, false);  // 发送 release 事件
    }
    keysDown.clear();  // 清空状态
}
```

### 流程分析

**Step 7**: action[5]=8
- **帧开始**：`keysDown = {}`
- Python 发送 "inventory 1"
- `press(E_KEY)`：
  - 发送 press 事件 → MC GUI 打开 ✅
  - `keysDown = {E_KEY}`
- MC tick
- **帧结束**：
  - 自动发送 release 事件（针对 E_KEY）
  - MC 收到 release → GUI toggle？
  - `keysDown.clear()` → `keysDown = {}`

**关键问题**：Step 7 结束时 GUI 应该已经关闭（因为 release），但用户看到的是"显示"。

**可能原因**：
1. **渲染延迟**：GUI 关闭在下一帧才渲染
2. **MC 只响应 press**：release 不触发 toggle
3. **用户观察时机**：在 `env.render()` 调用时，press 已生效但 release 未生效

**Step 8**: action[5]=8
- **帧开始**：`keysDown = {}`（Step 7 帧结束时已清空）
- Python 发送 "inventory 1"
- `press(E_KEY)`：
  - `!keysDown.contains(E_KEY)` → **true**（因为已清空）
  - 应该发送 press 事件！
  - 应该再次打开 GUI！

但用户看到的是"不显示"！矛盾了！

---

## 假设 3：keysDown 不会自动清空 + press 既触发 press 也触发 release

### 混合模型

```java
public static void press(int keyCode) {
    if (!keysDown.contains(keyCode)) {
        // 发送 press 事件
        sendKeyEvent(keyCode, true);
        keysDown.add(keyCode);
        
        // 关键：在帧末或下一帧开始时自动 release
        // 但不从 keysDown 移除
    }
}

// 每帧结束时
public void endFrame() {
    for (int key : pressedThisFrame) {
        sendKeyEvent(key, false);  // 自动 release
    }
    pressedThisFrame.clear();
    
    // keysDown 保持不变！
}
```

### 流程分析

**Step 7**: action[5]=8
- `press(E_KEY)` 被调用
  - `keysDown` 为空 → 发送 press 事件
  - `keysDown.add(E_KEY)`
  - 记录到 `pressedThisFrame`
- MC 收到 press → GUI 打开 ✅
- 帧结束：
  - 自动发送 release 事件
  - MC 收到 release → GUI toggle → 关闭
  - 但 `keysDown` 仍为 `{E_KEY}`

**Step 8**: action[5]=8
- `press(E_KEY)` 被调用
  - `keysDown.contains(E_KEY)` → **true**
  - **被忽略！**
- 没有任何事件
- GUI 保持关闭状态 ❌

这个模型**完全符合**用户的观察！

**Step 22-30 的间隔模式**：

**Step 22**: action[5]=8
- `press(E_KEY)` → 发送 press → GUI 打开 ✅
- `keysDown = {E_KEY}`
- 帧结束：release → GUI 关闭

**Step 23**: action[5]=0
- `release(E_KEY)` 被调用
  - `keysDown.contains(E_KEY)` → true
  - 发送 release 事件
  - **`keysDown.remove(E_KEY)`** ← 关键！
- `keysDown = {}`

**Step 24**: action[5]=8
- `press(E_KEY)` 被调用
  - `keysDown.contains(E_KEY)` → **false**（Step 23 已移除）
  - 发送 press 事件 → GUI 打开 ✅
  - `keysDown.add(E_KEY)`

这也完全符合！

---

## 结论

### 最可能的行为模型

1. **`press(keyCode)` 发送 press 事件**
   - 只有当 `!keysDown.contains(keyCode)` 时
   - 并将 keyCode 加入 `keysDown`

2. **每帧结束时，Malmo 自动发送 release 事件**
   - 针对所有在该帧内被 `press()` 的键
   - 但**不清空 `keysDown`**

3. **`release(keyCode)` 从 `keysDown` 中移除 keyCode**
   - 使得下次 `press(keyCode)` 可以再次触发

4. **MC 1.11.2 的 inventory GUI**
   - **press 事件**：toggle（打开/关闭）
   - **release 事件**：也 toggle？或者关闭？

### 解释用户观察

| Step | Action | keysDown 前 | press? | release? | GUI | keysDown 后 |
|------|--------|------------|--------|----------|-----|-------------|
| 7 | action[5]=8 | {} | ✅ 打开 | ✅ 关闭 | 显示* | {E} |
| 8 | action[5]=8 | {E} | ❌ 忽略 | ❌ 无 | 不显示 | {E} |
| 9 | action[5]=8 | {E} | ❌ 忽略 | ❌ 无 | 不显示 | {E} |
| 22 | action[5]=8 | {} | ✅ 打开 | ✅ 关闭 | 显示* | {E} |
| 23 | action[5]=0 | {E} | - | ✅ release | 不显示 | {} |
| 24 | action[5]=8 | {} | ✅ 打开 | ✅ 关闭 | 显示* | {E} |

*"显示"：用户在 press 事件后但 release 事件前观察到

---

## 验证方法

需要验证：
1. **Malmo 是否在每帧结束时自动 release？**
2. **MC 1.11.2 的 inventory 对 release 的响应是什么？**

### 实验 1：观察时机

在 `env.render()` 后加延迟，观察 GUI 是否在短时间内关闭：

```python
obs, reward, done, info = env.step(action)
env.render()
time.sleep(0.05)  # 等待 50ms
# 再次观察 GUI 状态
```

### 实验 2：Patch Malmo Java 代码

添加日志输出，记录每次 press/release 事件的时机。

---

## 对我们 Wrapper 的影响

如果上述模型正确，我们的 wrapper 策略是**正确的**：

```python
if current_inventory == 1 and not self._inventory_opened:
    self._inventory_opened = True
    # 让 inventory=1 通过 → press() → GUI 打开（但会在帧结束时 release）
else:
    # 删除 inventory 键 → 不调用 press() 也不调用 release()
    # keysDown 保持 {E_KEY}
    # 下次想再打开 GUI 时会失败（因为 keysDown 中已有）
    del minerl_action['inventory']
```

**问题**：这样的话，GUI 会在第一次打开后立即关闭（因为自动 release）！

**解决方案**：我们需要**持续发送 inventory=1**，但 Malmo 的 `keysDown` 去重会阻止重复 press。

**新策略**：
1. 第一次发送 inventory=1 → 打开
2. 紧接着发送 inventory=0 → 移除 keysDown
3. 下次发送 inventory=1 → 再次打开
4. 如此循环

但这又回到了"每帧都 toggle"的问题！

---

## 根本问题

**Malmo 的帧自动 release 机制**导致：
- 任何 press 事件在帧结束时都会自动 release
- 即使不显式调用 `release()`

这使得无法保持 inventory GUI 打开状态！

**唯一解决方案**：
1. **修改 Malmo Java 代码**，对 inventory 键禁用自动 release
2. **或者在 Python 层实现"假 GUI"**，不实际打开 MC 的 inventory


