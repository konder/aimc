# MineDojo Inventory 问题根本原因 - 最终确认

## 关键发现：每个 step 开始时都会调用大量 release！

### 日志分析

#### Step 1: action[5]=8

```
[12:59:29] 开始时：大量 release 调用（所有键）
[12:59:29] [FakeKeyboard.release] key=17, keysDown.contains=false, keysDown.size=0 (IGNORED)
[12:59:29] [FakeKeyboard.release] key=31, keysDown.contains=false, keysDown.size=0 (IGNORED)
... (10+ 个其他键)
[12:59:29] [FakeKeyboard.press] key=18, keysDown.contains=false, keysDown.size=0 ✅
[12:59:29] Pressed 18 ✅
```

**结果**：
- keysDown.size = 0 → 1
- Inventory GUI 打开 ✅

---

#### Step 2: action[5]=8 (连续第二次)

```
[12:59:30] 开始时：大量 release 调用（所有键，但没有 key=18!）
[12:59:30] [FakeKeyboard.release] key=17, keysDown.contains=false, keysDown.size=1 (IGNORED)
[12:59:30] [FakeKeyboard.release] key=31, keysDown.contains=false, keysDown.size=1 (IGNORED)
... (其他键)
[12:59:30] [FakeKeyboard.press] key=18, keysDown.contains=true, keysDown.size=1 ❌
[12:59:30] [FakeKeyboard.press] IGNORED - key already in keysDown
```

**关键发现**：
- keysDown.size 保持为 1（key=18 还在）
- press(18) 被忽略 ❌
- **没有自动 release(18)！**

---

#### Step 3: action[5]=8 (连续第三次)

```
[12:59:30] 开始时：大量 release 调用（所有键，但没有 key=18!）
[12:59:30] [FakeKeyboard.release] key=17, keysDown.contains=false, keysDown.size=1 (IGNORED)
... (其他键)
[12:59:30] [FakeKeyboard.press] key=18, keysDown.contains=true, keysDown.size=1 ❌
[12:59:30] [FakeKeyboard.press] IGNORED - key already in keysDown
```

**结果**：
- 同 Step 2
- press(18) 被忽略 ❌

---

#### Step 4: action[5]=0 (no-op，显式 release)

```
[12:59:31] 开始时：大量 release 调用
... (其他键)
[12:59:31] [FakeKeyboard.release] key=18, keysDown.contains=true, keysDown.size=1 ✅
[12:59:31] Released 18 ✅
[12:59:31] 调用栈:
  [1] com.microsoft.Malmo.Client.FakeKeyboard.release(FakeKeyboard.java:93)
  [2] com.microsoft.Malmo.MissionHandlers.CommandForKey.onExecute(CommandForKey.java:418)
  [3] com.microsoft.Malmo.MissionHandlers.CommandBase.execute(CommandBase.java:77)
```

**结果**：
- keysDown.size = 1 → 0
- 调用栈显示：**Python 显式发送 inventory=0，触发 CommandForKey.onExecute**
- **不是自动 release！**

---

#### Step 5: action[5]=8 (release 后再 press)

```
[12:59:31] 开始时：大量 release 调用
[12:59:31] [FakeKeyboard.release] key=17, keysDown.contains=false, keysDown.size=0 (IGNORED)
... (其他键)
[12:59:31] [FakeKeyboard.press] key=18, keysDown.contains=false, keysDown.size=0 ✅
[12:59:31] Pressed 18 ✅
```

**结果**：
- keysDown 已被 Step 4 清空
- press(18) 成功 ✅
- Inventory GUI 再次打开 ✅

---

## 核心结论

### ❌ 之前的假设是错误的！

**错误假设**：Malmo 在每帧结束时自动 release inventory 键

**真实情况**：
1. **Malmo 每个 step 开始时会 release 所有 "应该 release" 的键**
   - 这些键来自 Python 的 `action` 字典
   - 例如：forward=0, back=0, left=0... 都会触发 release
   
2. **但 inventory 键的行为不同！**
   - Step 1: action[5]=8 → inventory=1 → press(18)
   - Step 2: action[5]=8 → inventory=1 → press(18)（被忽略，因为 keysDown 中已有）
   - Step 4: action[5]=0 → inventory=0 → release(18)（显式 release）

3. **keysDown 不会自动清空**
   - press(18) 后，keysDown = {18}
   - 只有显式 release(18) 才会移除
   - **这与我们之前的假设完全相反！**

### ✅ 真正的问题

**为什么你观察到"连续 3 次 action[5]=8 只显示 1 次 GUI"？**

**答案**：
1. Step 1: press(18) 成功 → GUI 打开 ✅
2. Step 2: press(18) 被忽略（keysDown 中已有 18）→ 但此时 **每个 step 开始时的大量 release 调用可能包含了其他触发 GUI 关闭的机制**
3. 或者：**MC 1.11.2 的 inventory GUI 需要 "key release" 来保持打开状态？**

### 🔍 新的问题

为什么 Step 2/3 的 GUI 不显示？

**可能的原因**：
1. **MC 的 inventory GUI 需要键盘事件流**
   - press → GUI 打开
   - 持续有新的 press 事件 → GUI 保持打开
   - 没有新的 press 事件 → GUI 自动关闭？

2. **每个 step 开始时的大量 release 调用干扰了 GUI**
   - 虽然 release(18) 没有被调用
   - 但其他键的 release 可能触发了 GUI 状态刷新？

3. **MC 的渲染机制**
   - GUI 需要持续的"活跃"状态
   - press 事件被忽略 = 没有活跃状态 = GUI 关闭

---

## 解决方案评估

### 方案 A：Python Wrapper 删除 inventory 键（当前）

```python
if current_inventory == 1 and not self._inventory_opened:
    self._inventory_opened = True
    # 让 inventory=1 通过
else:
    del minerl_action['inventory']  # 删除 inventory 键
```

**问题**：删除键后，MineDojo 不会发送任何 inventory 命令，keysDown 保持 {18}，但 GUI 可能仍然关闭。

### 方案 B：持续发送 inventory=1？

不可行，因为 press(18) 在 keysDown 中已有 18 时会被忽略。

### 方案 C：交替发送 inventory=1 和 inventory=0

```python
if self._inventory_opened:
    if step_count % 2 == 0:
        minerl_action['inventory'] = 1
    else:
        minerl_action['inventory'] = 0
```

**问题**：会导致 GUI 不断 toggle，可能看到闪烁。

### 方案 D：修改 Malmo FakeKeyboard.press()

```java
public static void press(int key) {
    // 对于 inventory 键，总是发送 press 事件
    if (key == 18) {  // E 键
        add(new FakeKeyEvent(' ', key, true));
    } else if (!keysDown.contains(key)) {
        add(new FakeKeyEvent(' ', key, true));
    }
}
```

**效果**：每次 inventory=1 都会发送 press 事件，可能解决 GUI 关闭问题。

---

## 建议

1. **先测试当前 wrapper 的实际效果**
   - 运行 STEVE-1 评估
   - 观察 inventory 相关任务是否能完成

2. **如果确实需要 GUI 保持打开**
   - 实施方案 D：修改 FakeKeyboard.press()
   - 允许 inventory 键重复发送 press 事件

3. **或者接受现状**
   - 如果 STEVE-1 不需要 GUI 长时间打开
   - 只要能快速访问 inventory 即可


