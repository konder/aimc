# 多回合录制指南

> **功能**: 在一次运行中连续录制多个砍树回合，每次获得木头后自动保存并开始新回合

---

## 🎯 **核心功能**

### **自动回合管理**

```
回合1: 砍树 → 获得木头 ✓ → 自动保存 → Reset环境
回合2: 砍树 → 获得木头 ✓ → 自动保存 → Reset环境
回合3: 砍树 → 获得木头 ✓ → 自动保存 → Reset环境
...
回合10: 砍树 → 获得木头 ✓ → 自动保存 → 完成
```

**优点**:
- ✅ 无需反复启动程序
- ✅ 数据自动分类存储
- ✅ 环境自动随机化（增加数据多样性）
- ✅ 实时显示进度和统计

---

## 🚀 **快速开始**

### **基础用法**

```bash
# 默认录制10个回合，每回合最多1000帧
python tools/record_manual_chopping.py

# 输出结构:
data/expert_demos/round_0/
├── episode_000/
│   ├── frame_00000.png
│   ├── frame_00001.png
│   ├── ...
│   └── metadata.txt
├── episode_001/
│   ├── frame_00000.png
│   ├── ...
│   └── metadata.txt
├── episode_002/
│   └── ...
└── summary.txt  # 全局统计
```

---

### **自定义参数**

```bash
# 录制5个回合，每回合最多500帧
python tools/record_manual_chopping.py \
    --max-episodes 5 \
    --max-frames 500

# 使用自定义输出目录
python tools/record_manual_chopping.py \
    --output-dir data/my_demos \
    --max-episodes 10

# 调整相机灵敏度（更精细控制）
python tools/record_manual_chopping.py \
    --camera-delta 2 \
    --max-episodes 10
```

---

## 📋 **录制流程**

### **第1步: 启动程序**

```bash
python tools/record_manual_chopping.py --max-episodes 10
```

输出:
```
================================================================================
MineCLIP 砍树序列录制工具（多回合录制）
================================================================================

输出目录: data/expert_demos/round_0
每回合最大帧数: 1000
最大回合数: 10

[1/3] 创建MineDojo环境...
  任务: harvest_1_log_forest (森林中砍树)
  ✓ 环境创建成功
  动作空间: MultiDiscrete([3 3 4 25 25 8 244 36])

⚙️  相机设置: delta=4 (约60度/次)

[2/3] 开始多回合录制...

================================================================================
🎬 多回合录制模式
================================================================================
  获得木头后自动保存当前回合，并开始下一回合
  按Q键可随时停止所有录制
================================================================================

================================================================================
🎮 回合 1/10
================================================================================
  开始录制回合 1...
  目标: 获得一个木头（或达到1000帧）
```

---

### **第2步: 控制角色砍树**

**控制方式（键盘版）**:
- **W** - 前进
- **S** - 后退
- **A** - 左移
- **D** - 右移
- **Space** - 跳跃
- **I** - 视角向上
- **K** - 视角向下
- **J** - 视角向左
- **L** - 视角向右
- **F** - 攻击/砍树
- **Q** - 停止所有录制（立即结束）
- **ESC** - 紧急退出（不保存当前回合）

**屏幕显示**:
```
Episode: 1/10
Frame: 156/1000
Reward: 0.000
Total: 2.345
Status: In Progress

Press Q to stop all recording
```

---

### **第3步: 获得木头后自动进入下一回合**

当你砍树获得木头时:

```
🎉 回合1: 获得木头！已录制 234 帧

  💾 保存回合 1 数据...
  ✓ 回合 1 已保存: 234 帧 -> data/expert_demos/round_0/episode_000

================================================================================
🎮 回合 2/10
================================================================================
  开始录制回合 2...
  目标: 获得一个木头（或达到1000帧）
```

**自动操作**:
1. ✅ 保存当前回合所有帧
2. ✅ 保存当前回合元数据
3. ✅ Reset环境（新的随机种子）
4. ✅ 开始下一回合录制

---

### **第4步: 完成所有回合**

录制完10个回合后:

```
================================================================================
📊 录制完成统计
================================================================================

总回合数: 10
完成任务回合: 10/10
总帧数: 2456
平均帧数/回合: 245.6

各回合详情:
  回合 1: 234 帧, 奖励 2.345, ✓ 完成
  回合 2: 198 帧, 奖励 1.987, ✓ 完成
  回合 3: 267 帧, 奖励 2.678, ✓ 完成
  回合 4: 221 帧, 奖励 2.234, ✓ 完成
  回合 5: 189 帧, 奖励 1.890, ✓ 完成
  回合 6: 254 帧, 奖励 2.543, ✓ 完成
  回合 7: 211 帧, 奖励 2.112, ✓ 完成
  回合 8: 276 帧, 奖励 2.765, ✓ 完成
  回合 9: 303 帧, 奖励 3.034, ✓ 完成
  回合 10: 303 帧, 奖励 3.045, ✓ 完成

✓ 所有数据已保存到: data/expert_demos/round_0
✓ 统计信息已保存到: data/expert_demos/round_0/summary.txt

================================================================================
✅ 多回合录制完成！
================================================================================
```

---

## 📁 **输出文件结构**

```
data/expert_demos/round_0/
├── summary.txt              # 全局统计信息
├── episode_000/             # 第1回合
│   ├── frame_00000.png      # 第1帧
│   ├── frame_00001.png      # 第2帧
│   ├── ...
│   ├── frame_00233.png      # 最后一帧
│   └── metadata.txt         # 本回合元数据
├── episode_001/             # 第2回合
│   ├── frame_00000.png
│   ├── ...
│   └── metadata.txt
├── episode_002/             # 第3回合
│   └── ...
...
└── episode_009/             # 第10回合
    └── ...
```

---

### **summary.txt 内容**

```
Total Episodes: 10
Completed Episodes: 10
Total Frames: 2456
Average Frames per Episode: 245.6
Camera Delta: 4
Recording Time: 2025-10-21 15:30:45

Episode Details:
  Episode 0: 234 frames, reward 2.345, completed=True
  Episode 1: 198 frames, reward 1.987, completed=True
  Episode 2: 267 frames, reward 2.678, completed=True
  ...
```

---

### **metadata.txt 内容（每回合）**

```
Episode: 0
Frames: 234
Total Reward: 2.345
Task Completed: True
```

---

## ⚙️ **参数说明**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--output-dir` | str | `data/expert_demos/round_0` | 输出目录路径 |
| `--max-frames` | int | `1000` | 每回合最大帧数 |
| `--max-episodes` | int | `10` | 最大录制回合数 |
| `--camera-delta` | int | `4` | 相机转动增量（1-12） |

---

### **camera-delta 说明**

| 值 | 角度/次 | 用途 |
|----|---------|------|
| 1 | ~15° | 最精细控制 |
| 2 | ~30° | 精细控制（推荐新手） |
| 4 | ~60° | 平衡（默认） |
| 6 | ~90° | 快速转向 |
| 8 | ~120° | 极速转向 |
| 12 | ~180° | 最大角度 |

---

## 💡 **使用技巧**

### **1. 数据多样性**

```bash
# 录制多组数据，使用不同的round编号
python tools/record_manual_chopping.py --output-dir data/expert_demos/round_0 --max-episodes 10
python tools/record_manual_chopping.py --output-dir data/expert_demos/round_1 --max-episodes 10
python tools/record_manual_chopping.py --output-dir data/expert_demos/round_2 --max-episodes 10

# 最终获得30个回合的数据
```

---

### **2. 快速录制（熟练玩家）**

```bash
# 减少每回合帧数限制，快速完成10个回合
python tools/record_manual_chopping.py \
    --max-frames 500 \
    --max-episodes 10 \
    --camera-delta 6
```

---

### **3. 高质量录制（新手）**

```bash
# 增加帧数限制，使用精细相机控制
python tools/record_manual_chopping.py \
    --max-frames 2000 \
    --max-episodes 5 \
    --camera-delta 2
```

---

### **4. 中途停止**

如果你想提前结束（例如录了5个回合后觉得够了）:

1. **按Q键** - 停止所有录制，保存已完成的回合
2. 程序会自动保存统计信息并退出

```
⏸️  停止所有录制（用户按下Q）

  💾 保存回合 5 数据...
  ✓ 回合 5 已保存: 189 帧 -> data/expert_demos/round_0/episode_004

================================================================================
📊 录制完成统计
================================================================================

总回合数: 5  # 只录了5个，提前停止
完成任务回合: 5/5
...
```

---

## 🔍 **验证录制数据**

录制完成后，可以使用验证脚本分析数据质量:

```bash
# 方法1: 分析单个回合
python tools/verify_mineclip_16frames.py \
    --sequence-dir data/expert_demos/round_0/episode_000

# 方法2: 批量分析所有回合
for ep in data/expert_demos/round_0/episode_*/; do
    echo "Analyzing $ep"
    python tools/verify_mineclip_16frames.py --sequence-dir "$ep"
done
```

---

## ❓ **常见问题**

### **Q1: 如果我在某个回合中失败了怎么办？**

**A**: 两种情况:
1. **达到max_frames但没获得木头**: 自动保存当前回合（标记为未完成），然后开始下一回合
2. **你觉得这个回合质量不好**: 按Q键停止，删除最后一个episode目录，重新运行

---

### **Q2: 每个回合的环境是一样的吗？**

**A**: 不一样！环境使用`world_seed=None`（随机种子），每次reset后:
- ✅ 出生点不同
- ✅ 树木位置不同
- ✅ 地形略有差异

这增加了数据多样性，对训练有利 ✅

---

### **Q3: 可以调整录制回合数吗？**

**A**: 完全可以！

```bash
# 只录3个回合（快速测试）
python tools/record_manual_chopping.py --max-episodes 3

# 录20个回合（大规模数据收集）
python tools/record_manual_chopping.py --max-episodes 20
```

---

### **Q4: 如果某个回合卡住了怎么办？**

**A**: 如果你在某个回合中找不到树或遇到其他问题:
1. **按Q键** - 停止所有录制
2. 程序会保存当前已完成的回合
3. 未完成的回合会被标记为`task_completed=False`

---

### **Q5: 可以同时使用鼠标和键盘吗？**

**A**: 不能，目前是两个独立工具:
- `record_manual_chopping.py` - 键盘版
- `record_manual_chopping_mouse.py` - 鼠标版（暂不支持多回合）

**建议**: 如果你熟悉键盘控制，使用键盘版的多回合功能更方便。

---

## 🎯 **DAgger训练流程**

录制完专家数据后，可以用于DAgger训练:

```bash
# 第0轮: 使用专家数据训练初始策略
python src/training/train_bc.py \
    --data-dir data/expert_demos/round_0 \
    --output-model models/bc_round_0.zip

# 第1轮: 运行策略收集状态
python tools/run_policy_collect_states.py \
    --policy models/bc_round_0.zip \
    --output-dir data/states_round_1 \
    --episodes 10

# 第1轮: 手动标注状态
python tools/label_states.py \
    --states-dir data/states_round_1 \
    --output-dir data/expert_demos/round_1

# 第1轮: 合并数据重新训练
python src/training/train_bc.py \
    --data-dir data/expert_demos/round_0 data/expert_demos/round_1 \
    --output-model models/bc_round_1.zip

# 重复...
```

---

## 📚 **相关文档**

- [`CAMERA_CONTROL_FIX.md`](../issues/CAMERA_CONTROL_FIX.md) - 相机控制修复
- [`KEYBOARD_ROTATION_BUG_FIX.md`](../issues/KEYBOARD_ROTATION_BUG_FIX.md) - 旋转bug修复
- [`DAGGER_QUICK_START.md`](DAGGER_QUICK_START.md) - DAgger快速开始
- [`IMITATION_LEARNING_GUIDE.md`](IMITATION_LEARNING_GUIDE.md) - 模仿学习指南

---

**最后更新**: 2025-10-21  
**状态**: ✅ 已实现并测试  
**推荐用途**: DAgger专家数据收集

