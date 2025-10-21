# 多回合录制正确实现

> **核心概念**: round_0, round_1, round_2... 每个round是一次完整的任务，只有done=True才保存

---

## 🎯 **正确的概念理解**

### **Round vs Episode**

```
❌ 错误理解（之前的实现）:
data/expert_demos/round_0/
├── episode_000/  # 第1个episode
├── episode_001/  # 第2个episode
...

✅ 正确理解（现在的实现）:
data/expert_demos/
├── round_0/      # 第1个round（一次完整任务）
│   ├── frame_00000.png
│   ├── frame_00001.png
│   ├── ...
│   └── metadata.txt
├── round_1/      # 第2个round
│   └── ...
├── round_2/      # 第3个round
│   └── ...
...
```

**理解**:
- **Round**: 一次完整的任务录制（从reset到done=True）
- **Episode**: MineDojo/Gym的术语，等同于round
- **本项目**: 使用round编号，每个round_N是一个独立的目录

---

## 🔑 **核心规则**

### **规则1: 只有done=True才保存**

```python
if done:
    task_completed = True
    print(f"\n🎉 round_{round_idx}: 任务完成！")
    
# 回合结束后
if task_completed and len(frames) > 0:
    # 保存round_N目录
    save_round_data()
else:
    print("未完成，不保存")
```

**行为**:
- ✅ **done=True**: 自动保存到`round_N/`
- ❌ **按Q键**: 不保存，直接退出
- ❌ **按ESC**: 不保存，直接退出
- ❌ **达到max_frames但done=False**: 不保存

---

### **规则2: 断点续录**

```bash
# 第一次: 录制round_0 ~ round_4（完成5个）
python tools/record_manual_chopping.py --max-rounds 10
# 实际完成5个后按Q退出

# 第二次: 从round_5继续
python tools/record_manual_chopping.py --start-round 5 --max-rounds 5
# 继续录制round_5 ~ round_9
```

**逻辑**:
```python
for round_idx in range(start_round, start_round + max_rounds):
    # 录制 round_idx
```

**示例**:
```
--start-round 0 --max-rounds 10
→ 录制 round_0, round_1, ..., round_9

--start-round 5 --max-rounds 10
→ 录制 round_5, round_6, ..., round_14

--start-round 10 --max-rounds 5
→ 录制 round_10, round_11, round_12, round_13, round_14
```

---

### **规则3: 已有数据检测**

```python
# 如果start_round=0且检测到已有round_*目录
existing_rounds = ['round_0', 'round_1', 'round_2']
print(f"检测到已有 {len(existing_rounds)} 个回合")
print(f"提示: 使用 --start-round {len(existing_rounds)} 可以继续录制")

response = input("是否删除所有已有数据并从头开始？(y/N): ")
if response.lower() == 'y':
    # 删除所有round_*
    shutil.rmtree(...)
else:
    print("❌ 取消录制")
    return
```

**保护机制**: 防止意外覆盖已有数据

---

## 📋 **使用示例**

### **场景1: 第一次录制（从头开始）**

```bash
# 录制10个回合
python tools/record_manual_chopping.py --max-rounds 10

# 流程:
# round_0: 砍树 → done=True → 保存
# round_1: 砍树 → done=True → 保存
# round_2: 砍树 → done=True → 保存
# ... (假设完成3个后按Q退出)

# 输出:
data/expert_demos/
├── round_0/
├── round_1/
├── round_2/
└── summary.txt
```

---

### **场景2: 断点续录**

```bash
# 从round_3继续录制
python tools/record_manual_chopping.py --start-round 3 --max-rounds 7

# 流程:
# round_3: 砍树 → done=True → 保存
# round_4: 砍树 → done=True → 保存
# ...
# round_9: 砍树 → done=True → 保存

# 最终输出:
data/expert_demos/
├── round_0/  # 之前录的
├── round_1/  # 之前录的
├── round_2/  # 之前录的
├── round_3/  # 新录的
├── round_4/  # 新录的
...
├── round_9/  # 新录的
└── summary.txt
```

---

### **场景3: 中途放弃某个round**

```bash
# 开始录制
python tools/record_manual_chopping.py --start-round 0 --max-rounds 10

# round_0: 正在录制...
# 突然发现开局位置不好，按Q退出

# 结果:
# round_0 不会被保存（因为done=False）
# 可以重新开始

# 重新运行
python tools/record_manual_chopping.py --start-round 0 --max-rounds 10
# round_0: 重新开始
```

---

### **场景4: 已有数据时的警告**

```bash
# 第一次运行
python tools/record_manual_chopping.py
# 完成 round_0, round_1, round_2

# 第二次运行（忘记加--start-round）
python tools/record_manual_chopping.py

# 输出:
⚠️  检测到已有 3 个回合: ['round_0', 'round_1', 'round_2']
提示: 使用 --start-round 3 可以继续录制

是否删除所有已有数据并从头开始？(y/N): 

# 选择N → 取消录制
# 选择y → 删除所有，重新开始
```

---

## 📊 **输出结构**

### **目录结构**

```
data/expert_demos/
├── round_0/
│   ├── frame_00000.png
│   ├── frame_00001.png
│   ├── ...
│   ├── frame_00234.png
│   └── metadata.txt
├── round_1/
│   ├── frame_00000.png
│   ├── ...
│   ├── frame_00198.png
│   └── metadata.txt
├── round_2/
│   └── ...
...
└── summary.txt
```

---

### **metadata.txt（每个round）**

```
Round: 0
Frames: 235
Total Reward: 2.345
Task Completed: True
Recording Time: 2025-10-21 16:30:45
```

---

### **summary.txt（全局）**

```
Total Completed Rounds: 3
Round Range: round_0 ~ round_2
Camera Delta: 4
Max Frames per Round: 1000
Recording Time: 2025-10-21 16:35:12

Saved Rounds:
  round_0: 235 frames
  round_1: 198 frames
  round_2: 267 frames
```

---

## 🎮 **实际操作流程**

### **第1步: 启动录制**

```bash
python tools/record_manual_chopping.py --max-rounds 10
```

### **第2步: 控制角色完成任务**

```
使用WASD移动，IJKL控制视角，F攻击

目标: 砍树获得木头（done=True）

屏幕显示:
Round: 0 (目标: 9)
Completed: 0
Frame: 156/1000
Reward: 0.000
Total: 2.345
Status: Recording...

Q/ESC=quit (no save) | Done=auto save
```

### **第3步: 任务完成自动保存**

```
🎉 round_0: 任务完成！已录制 234 帧
    物品变化: {'log': 1}

  💾 保存 round_0 数据...
  ✓ round_0 已保存: 234 帧 -> data/expert_demos/round_0

================================================================================
🎮 Round 1
================================================================================
  开始录制 round_1...
  目标: 完成任务 (done=True)
  提示: 按Q/ESC不会保存，只有done=True才会保存
```

### **第4步: 中途退出**

```
假设完成了3个round后不想继续:

按Q键 → 退出

================================================================================
📊 录制完成统计
================================================================================

✅ 成功完成回合数: 3
回合范围: round_0 ~ round_2

保存位置: data/expert_demos/

已保存的回合:
  round_0: 234 帧
  round_1: 198 帧
  round_2: 267 帧

✓ 统计信息已保存到: data/expert_demos/summary.txt

================================================================================
✅ 多回合录制完成！
================================================================================

继续录制提示:
  python tools/record_manual_chopping.py --start-round 3
```

### **第5步: 断点续录**

```bash
# 从round_3继续
python tools/record_manual_chopping.py --start-round 3 --max-rounds 7

# 继续录制round_3 ~ round_9
```

---

## ⚙️ **参数说明**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--base-dir` | str | `data/expert_demos` | 基础目录 |
| `--max-frames` | int | `1000` | 每回合最大帧数 |
| `--max-rounds` | int | `10` | 最大回合数 |
| `--start-round` | int | `0` | 起始回合编号 |
| `--camera-delta` | int | `4` | 相机转动增量 |

---

## ✅ **关键特性总结**

### **1. 只有done才保存** ✅

```python
if done:
    task_completed = True
    
if task_completed and len(frames) > 0:
    save_round()  # 保存
else:
    skip()  # 不保存
```

---

### **2. 断点续录** ✅

```bash
# 第一次
python tools/record_manual_chopping.py --max-rounds 10
# 完成3个后退出

# 第二次
python tools/record_manual_chopping.py --start-round 3 --max-rounds 7
# 继续录制round_3 ~ round_9
```

---

### **3. 数据保护** ✅

```python
if start_round == 0 and existing_rounds:
    response = input("是否删除所有已有数据？(y/N): ")
    if response != 'y':
        return  # 取消录制
```

---

### **4. 清晰的round目录** ✅

```
data/expert_demos/
├── round_0/
├── round_1/
├── round_2/
...
```

每个round是一个独立的目录，不是episode_000/episode_001...

---

## 📚 **相关文档**

- [`MULTI_EPISODE_AND_CAMERA_FIX.md`](../issues/MULTI_EPISODE_AND_CAMERA_FIX.md) - 修复记录
- [`DAGGER_QUICK_START.md`](DAGGER_QUICK_START.md) - DAgger快速开始

---

**最后更新**: 2025-10-21  
**状态**: ✅ 已正确实现  
**推荐使用**: `python tools/record_manual_chopping.py --max-rounds 10`

