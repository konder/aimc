# 按键控制更新

> **更新**: Q键和ESC键功能调整

---

## 🔄 **按键功能变更**

### **修改前**

| 按键 | 功能 |
|------|------|
| Q | 停止所有录制，退出程序 ❌ |
| ESC | 紧急退出，不保存 ❌ |

**问题**: 两个键功能太相似，都是退出

---

### **修改后** ✅

| 按键 | 功能 | 说明 |
|------|------|------|
| **Q** | 🔄 **重新录制当前回合** | 不保存当前回合数据，reset环境，重新录制round_N |
| **ESC** | ❌ **退出程序** | 不保存当前回合数据，直接退出整个程序 |
| **Done** | ✅ **自动保存并进入下一回合** | env返回done=True，自动保存round_N，开始round_N+1 |

---

## 📊 **使用场景**

### **场景1: 录制出错，想重新开始**

```
你在录制round_3时，发现开局位置不好

按Q键 → 重新录制round_3 ✅

流程:
1. 当前round_3数据不保存
2. 环境reset
3. 重新开始录制round_3
```

---

### **场景2: 不想继续了，直接退出**

```
你录制了5个回合，觉得够了

按ESC → 退出程序 ✅

流程:
1. 当前round_5数据不保存（如果未完成）
2. 退出程序
3. 已保存的round_0~4仍然存在
```

---

### **场景3: 正常完成任务**

```
你成功砍树获得木头

done=True → 自动保存并进入下一回合 ✅

流程:
1. round_3数据自动保存
2. 环境reset
3. 开始录制round_4
```

---

## 🔧 **技术实现**

### **核心逻辑**

```python
# 使用while循环代替for循环
round_idx = start_round
while round_idx < start_round + max_rounds:
    retry_current_round = False
    
    # 录制主循环
    while step_count < max_frames:
        if 'Q' pressed:
            retry_current_round = True
            break
        elif 'ESC' pressed:
            global_continue = False
            break
        
        if done:
            task_completed = True
            break
    
    # 检查是否需要重新录制
    if retry_current_round:
        continue  # 重新录制当前round_idx
    
    # 保存数据（只有done=True才保存）
    if task_completed:
        save_round()
        
    # 进入下一个round
    round_idx += 1
```

---

### **关键点**

1. ✅ **while循环**: 允许通过`continue`重新录制当前round
2. ✅ **retry_current_round标志**: 标记是否需要重新录制
3. ✅ **round_idx手动递增**: 只有在保存后才`+1`

---

## 🎮 **实际操作示例**

### **示例1: 重新录制**

```bash
python tools/record_manual_chopping.py --max-rounds 10

# round_0: 正在录制...
# 发现开局不好，按Q键

🔄 重新录制 round_0（用户按下Q）
   当前回合数据不保存，即将重置环境...
  准备重新录制 round_0...

================================================================================
🎮 Round 0
================================================================================
  重置环境中...
  ✓ 环境已重置，新的世界已生成
  开始录制 round_0...

# 重新开始录制round_0 ✅
```

---

### **示例2: 连续重新录制**

```bash
# round_0: 开局不好，按Q → 重录round_0
# round_0: 又不好，再按Q → 再重录round_0
# round_0: 这次好，砍树完成 → done=True，保存round_0

# round_1: 开始录制...
```

**可以多次按Q，直到满意为止！** ✅

---

### **示例3: 中途退出**

```bash
# round_0: 完成 ✅
# round_1: 完成 ✅
# round_2: 完成 ✅
# round_3: 正在录制...不想录了

按ESC → 退出程序

已保存的round:
  round_0: 234 帧
  round_1: 198 帧
  round_2: 267 帧

round_3: 不保存（未完成）
```

---

## 📋 **提示信息更新**

### **启动时提示**

```
================================================================================
🎬 多回合录制模式
================================================================================
  ✅ 完成任务(done=True) → 自动保存当前回合，进入下一回合
  🔄 按Q键 → 不保存当前回合，重新录制当前回合
  ❌ 按ESC → 不保存当前回合，退出程序
================================================================================
```

---

### **屏幕显示**

```
Round: 3 (目标: 9)
Completed: 3
Frame: 156/1000
Reward: 0.000
Total: 2.345
Status: Recording...

Q=retry | ESC=quit | Done=auto save&next
```

---

## ✅ **优势**

1. ✅ **Q键重新录制** - 不满意可以立即重录，不需要退出程序
2. ✅ **ESC退出** - 不想继续可以随时退出
3. ✅ **自动保存** - 完成任务自动保存，无需手动操作
4. ✅ **数据保护** - 只有done=True才保存，避免保存无效数据

---

## 🎯 **使用建议**

### **新手推荐**

```bash
# 慢慢来，不满意就按Q重录
python tools/record_manual_chopping.py --max-rounds 10 --max-frames 2000

# 每个round都尽量完成
# 不满意就按Q重录
# 录够了就按ESC退出
```

---

### **熟练玩家**

```bash
# 快速录制，追求效率
python tools/record_manual_chopping.py --max-rounds 10 --max-frames 1000

# 快速完成每个round
# 偶尔按Q重录开局不好的
```

---

**更新日期**: 2025-10-21  
**状态**: ✅ 已实现  
**推荐**: Q键重新录制功能非常实用！

