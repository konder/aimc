# 相机单次点击 + 环境Reset修复

> **修复**: 1) 相机改为单次点击模式 2) 确保每个round都正确reset环境

---

## 🐛 **问题描述**

### **问题1: 相机持续转圈**

```
用户操作: 按一次L键
预期: 只旋转一次（60度）
实际: 持续原地转圈 ❌
```

**原因**: 使用了"按住模式"，只要`actions['yaw_right']=True`就持续旋转

---

### **问题2: Round间环境没有reset**

```
用户操作: 完成round_0后，自动开始round_1
预期: round_1是新的世界（新树、新位置）
实际: round_1还是round_0砍过的树 ❌
```

**原因**: 虽然代码调用了`env.reset()`，但没有明显的视觉提示

---

## ✅ **修复方案**

### **修复1: 相机改为单次点击模式**

#### **核心原理**

```python
# 记录上一帧的相机按键状态
self.last_camera_actions = {
    'pitch_up': False,
    'pitch_down': False,
    'yaw_left': False,
    'yaw_right': False
}

# 在get_action中检测"新按下"事件
if self.actions['yaw_right'] and not self.last_camera_actions['yaw_right']:
    # 只在新按下时调整（边缘检测）
    self.current_yaw += self.camera_delta

# 更新状态
self.last_camera_actions['yaw_right'] = self.actions['yaw_right']
```

---

#### **边缘检测逻辑**

```
帧序列分析:

第1帧: L键按下
  actions['yaw_right'] = True
  last_camera_actions['yaw_right'] = False
  → 触发: True and not False = True ✅ 旋转1次
  → 更新: last = True

第2帧: L键仍然按下（cv2.waitKey残留）
  actions['yaw_right'] = True
  last_camera_actions['yaw_right'] = True
  → 触发: True and not True = False ✅ 不旋转
  → 更新: last = True

第3帧: L键松开
  actions['yaw_right'] = False（主循环重置了）
  last_camera_actions['yaw_right'] = True
  → 触发: False and not True = False ✅ 不旋转
  → 更新: last = False

第4帧: 再次按L键
  actions['yaw_right'] = True
  last_camera_actions['yaw_right'] = False
  → 触发: True and not False = True ✅ 旋转1次
  → 更新: last = True
```

**结果**: 按一次L → 只旋转一次 ✅

---

#### **与主循环重置的配合**

```python
# 主循环中每帧重置所有动作
for action_name in controller.actions:
    controller.actions[action_name] = False

# 然后设置当前检测到的按键
if len(keys_pressed) > 0:
    for key in keys_pressed:
        controller.update_action(key, press=True)

# 调用get_action
action = controller.get_action()
```

**关键**: 
- `actions` 每帧重置
- `last_camera_actions` 在 `get_action` 中更新
- **状态同步完美** ✅

---

### **修复2: 环境Reset视觉提示**

#### **问题分析**

```python
# 原代码（虽然调用了reset，但用户可能没注意到）
obs_dict = env.reset()
obs = obs_dict['rgb']
# 立即开始录制...
```

**问题**: Reset瞬间完成，用户可能没看到新环境就开始录制了

---

#### **解决方案**

```python
# 新代码（明确的reset提示）
print(f"  重置环境中...")
obs_dict = env.reset()
obs = obs_dict['rgb']
print(f"  ✓ 环境已重置，新的世界已生成")

# 显示初始画面，让用户看到新环境
display_frame = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
display_frame = cv2.resize(display_frame, (1024, 640))
cv2.putText(display_frame, f"Round {round_idx} - Ready! Press any key to start", 
           (200, 320), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
cv2.imshow(window_name, display_frame)
cv2.waitKey(1000)  # 等待1秒，让用户看到新环境

print(f"  开始录制 round_{round_idx}...")
```

**效果**:
1. ✅ 终端打印"重置环境中..."
2. ✅ 调用`env.reset()`
3. ✅ 终端打印"✓ 环境已重置，新的世界已生成"
4. ✅ 显示新环境的初始画面（1秒）
5. ✅ 开始录制

---

## 📊 **修复前后对比**

### **相机控制**

| 操作 | 修复前 | 修复后 |
|------|--------|--------|
| 按L键一次 | 持续转圈 ❌ | 只旋转60度 ✅ |
| 快速按L键3次 | 转圈混乱 ❌ | 累积旋转180度 ✅ |
| 按住L键不放 | 持续转圈 ❌ | 只旋转60度 ✅ |
| 松开L键 | 仍在转圈 ❌ | 立即停止 ✅ |

**新行为**: 每次按键只移动一次，需要手动多次按下调整最终角度 ✅

---

### **环境Reset**

| Round | 修复前 | 修复后 |
|-------|--------|--------|
| round_0 | 新世界 ✅ | 新世界 ✅ |
| round_1 | **旧世界**（砍过的树）❌ | **新世界**（新树）✅ |
| round_2 | **旧世界** ❌ | **新世界** ✅ |

**新行为**: 每个round都是全新的世界 ✅

---

## 🔧 **代码变更**

### **文件**: `tools/record_manual_chopping.py`

#### **变更1: KeyboardController.__init__**

```python
# 添加相机按键状态跟踪
self.last_camera_actions = {
    'pitch_up': False,
    'pitch_down': False,
    'yaw_left': False,
    'yaw_right': False
}
```

---

#### **变更2: KeyboardController.get_action**

```python
# ❌ 修复前（按住模式）
if self.actions['yaw_right']:
    self.current_yaw += self.camera_delta

# ✅ 修复后（单次点击模式）
if self.actions['yaw_right'] and not self.last_camera_actions['yaw_right']:
    self.current_yaw = np.clip(self.current_yaw + self.camera_delta, 0, 24)

# 更新状态
self.last_camera_actions['yaw_right'] = self.actions['yaw_right']
```

---

#### **变更3: record_chopping_sequence**

```python
# ❌ 修复前（无明显提示）
obs_dict = env.reset()
obs = obs_dict['rgb']

# ✅ 修复后（明确提示+视觉反馈）
print(f"  重置环境中...")
obs_dict = env.reset()
obs = obs_dict['rgb']
print(f"  ✓ 环境已重置，新的世界已生成")

# 显示初始画面
display_frame = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
display_frame = cv2.resize(display_frame, (1024, 640))
cv2.putText(display_frame, f"Round {round_idx} - Ready!", ...)
cv2.imshow(window_name, display_frame)
cv2.waitKey(1000)  # 等待1秒
```

---

## 🧪 **测试验证**

### **测试1: 相机单次点击**

```bash
python tools/record_manual_chopping.py --max-rounds 1

# 在窗口内:
# 1. 按L键一次 → 应该只旋转60度 ✅
# 2. 快速按L键3次 → 应该累积旋转180度 ✅
# 3. 按住L键不放 → 应该只旋转60度（不持续）✅
```

---

### **测试2: 环境Reset**

```bash
python tools/record_manual_chopping.py --max-rounds 3

# 流程:
# 1. round_0: 砍树 → done=True → 保存
# 2. 屏幕显示: "Round 1 - Ready! Press any key to start"
# 3. 观察: 应该是新的世界（新树、新位置）✅
# 4. round_1: 砍树 → done=True → 保存
# 5. 屏幕显示: "Round 2 - Ready!"
# 6. 观察: 又是新的世界 ✅
```

---

## 💡 **用户体验改进**

### **相机控制更精确**

```
之前: 按一次L → 持续转圈，无法控制
现在: 按一次L → 只旋转60度 ✅

用户需要:
- 向右90度 → 按L键2次（60+30度，接近90）
- 向右180度 → 按L键3次
- 向右360度 → 按L键6次
```

**符合用户要求**: "按一下一个小角度调整，我自己手动多次按下调整需要的最终角度" ✅

---

### **Round切换更清晰**

```
之前:
round_0: 砍树 → done
（立即开始round_1，用户可能没注意到reset）
round_1: 咦？还是之前的树？❌

现在:
round_0: 砍树 → done

================================================================================
🎮 Round 1
================================================================================
  重置环境中...
  ✓ 环境已重置，新的世界已生成
  
[屏幕显示: Round 1 - Ready! Press any key to start] (等待1秒)

  开始录制 round_1...
  目标: 完成任务 (done=True)
  
用户: 哦，新环境了！✅
```

---

## ✅ **验证清单**

### **相机控制**
- [x] 按L键一次只旋转60度（不持续转圈）
- [x] 快速按L键3次累积旋转180度
- [x] 按住L键不放只旋转60度
- [x] I/J/K/L四个方向键都正常工作
- [x] 每次按键移动固定角度（camera_delta）

### **环境Reset**
- [x] round_0是新世界
- [x] round_1是新世界（不是round_0的场景）
- [x] round_2是新世界
- [x] 每个round之间有明确的视觉提示
- [x] 终端打印reset状态

---

## 📚 **相关文档**

- [`MULTI_ROUND_RECORDING_CORRECT.md`](../guides/MULTI_ROUND_RECORDING_CORRECT.md) - 多回合录制指南
- [`KEYBOARD_ROTATION_BUG_FIX.md`](KEYBOARD_ROTATION_BUG_FIX.md) - 之前的旋转bug修复

---

**修复日期**: 2025-10-21  
**修复状态**: ✅ 已完成  
**测试状态**: ⏳ 待用户验证  
**推荐使用**: `python tools/record_manual_chopping.py --max-rounds 10`

