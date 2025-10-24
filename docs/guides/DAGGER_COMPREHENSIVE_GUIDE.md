# DAgger 综合指南

> **完整教程**: 从零开始使用 DAgger 算法训练 Minecraft AI - 理论 + BC基线 + 迭代优化 + 多任务支持

---

## 📑 **目录**

1. [快速开始](#-快速开始)
2. [模仿学习理论背景](#-模仿学习理论背景)
3. [DAgger 核心概念](#-dagger-核心概念)
4. [BC 基线训练](#-bc-基线训练)
5. [录制工具使用](#-录制工具使用)
6. [DAgger 迭代优化](#-dagger-迭代优化)
7. [标注工具使用](#-标注工具使用)
8. [标注策略最佳实践](#-标注策略最佳实践)
9. [继续训练](#-继续训练)
10. [多任务支持](#-多任务支持)
11. [脚本使用详解](#-脚本使用详解)
12. [故障排查](#-故障排查)
13. [命令速查表](#-命令速查表)

---

## 🚀 **快速开始**

### **前置准备**

```bash
# 激活环境
conda activate minedojo

# 确认项目依赖
pip install -r requirements.txt
```

### **完整流程（3-5小时）**

```bash
# 一键运行完整工作流
bash scripts/run_dagger_workflow.sh \
    --task harvest_1_log \
    --num-episodes 10 \
    --iterations 3
```

**执行步骤**：
1. 录制 10 个专家演示（40-60分钟）
2. 训练 BC 基线（30-40分钟）
3. 评估 BC 成功率（10分钟）
4. DAgger 迭代 1-3（每轮 60-80分钟）
5. 显示完整训练历史

**预期成功率**: BC 60% → 迭代3后 85-90%

---

## 📚 **模仿学习理论背景**

### **核心思想**

通过人工录制专家演示数据，让智能体学习人类的行为策略。

### **为什么使用模仿学习？**

#### **相比纯RL的优势**

| 方面 | 纯RL（PPO） | 模仿学习 | BC+DAgger |
|------|-----------|---------|-----------|
| 首次成功 | ~50K steps | ~5K steps | **~2K steps** |
| 稳定成功率 | 100K steps | 20K steps | **10K steps** |
| 最终成功率 | 80-90% | 70-80% | **90-95%** |
| 训练时间 | 3-5小时 | 1小时 | **2-3小时** |
| 数据需求 | 无 | 10-20 episodes | **10-20 episodes** |

### **适用场景**

| 任务类型 | 适合程度 | 原因 |
|---------|---------|------|
| 🪵 砍树 | ⭐⭐⭐⭐⭐ | 简单、明确，易于演示 |
| 🏗️ 建造 | ⭐⭐⭐⭐⭐ | 需要特定序列，纯RL很难学会 |
| ⚔️ 战斗 | ⭐⭐⭐⭐ | 需要时机和策略 |
| 🌾 种植 | ⭐⭐⭐⭐ | 多步骤，顺序重要 |
| ⛏️ 挖矿 | ⭐⭐⭐ | 路径规划 |
| 🎣 随机探索 | ⭐⭐ | RL可能更适合 |

### **三种主要方法**

#### **方法1: 行为克隆（BC）**

```
专家演示 → 监督学习 → 策略模型
```

**优点**: 简单、快速  
**缺点**: 分布偏移问题

#### **方法2: DAgger（推荐）⭐**

```
专家演示 → BC → 收集失败 → 标注 → 聚合训练 → 循环
```

**优点**: 解决分布偏移  
**缺点**: 需要多次标注

#### **方法3: BC + RL 微调（最佳）⭐⭐⭐**

```
专家演示 → BC预训练 → PPO微调 → 最优策略
```

**优点**: 结合两者优势  
**缺点**: 两阶段训练

### **成功案例**

- **MineRL比赛**: 冠军队伍都使用了人类演示数据
- **OpenAI VPT**: 7万小时YouTube视频训练
- **DeepMind Gato**: 多任务模仿学习

---

## 🎯 **DAgger 核心概念**

### **什么是 DAgger？**

**DAgger** (Dataset Aggregation) 是改进版行为克隆算法，解决传统 BC 的"分布偏移"问题。

#### **问题：传统 BC 的分布偏移**

```
专家演示: s₀ → s₁ → s₂ → s₃  (专家轨迹)
学习策略: s₀ → s₁' → s₂'' → s₃'''  (略有偏差)
         ↑    ↑     ↑      ↑
      相同  略偏  更偏   完全偏离！
```

**问题**: 一旦偏离专家演示，策略会越来越差。

**训练分布 vs 测试分布**:
```python
# BC训练时
训练状态分布 = 专家访问的状态
P_train(s) = 只包含专家轨迹上的状态

# BC测试时
测试状态分布 = 学习策略访问的状态
P_test(s) = 包含学习策略偏离后的状态

# 问题: P_train ≠ P_test ⚠️
```

#### **DAgger 解决方案**

在策略访问的**新状态**上收集专家标注！

```
BC基线 (bc_baseline.zip) - 成功率: 60%
  ↓ 
  ↓ [收集失败状态] → [标注] → [重新训练]
  ↓
迭代1 (dagger_iter_1.zip) - 成功率: 75%
  ↓
  ↓ [收集失败状态] → [标注] → [重新训练]
  ↓
迭代2 (dagger_iter_2.zip) - 成功率: 85%
  ↓
  ↓ [收集失败状态] → [标注] → [重新训练]
  ↓
迭代3 (dagger_iter_3.zip) - 成功率: 92%
```

**DAgger解决**:
```python
# DAgger每轮迭代
P_train(s) 逐渐包含策略πᵢ访问的状态

# 经过多轮后
P_train(s) ≈ P_test(s)  ✅

# 结果: 策略在自己访问的状态上也有训练数据！
```

### **DAgger 算法流程**

```
┌─────────────────────────────────────────┐
│  阶段1: 初始训练                         │
│  用专家演示D₀训练初始策略π₁             │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│  阶段2: 策略执行 (第i轮)                 │
│  运行当前策略πᵢ，收集新轨迹              │
│  记录访问的状态 Sᵢ = {s₁, s₂, ...}      │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│  阶段3: 专家标注                         │
│  人工/专家对Sᵢ中的状态标注正确动作       │
│  Dᵢ = {(s, a*) | s ∈ Sᵢ}               │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│  阶段4: 数据聚合                         │
│  D_all = D₀ ∪ D₁ ∪ ... ∪ Dᵢ           │
│  用所有数据重新训练策略πᵢ₊₁              │
└─────────────────────────────────────────┘
              ↓
         重复2-4，直到收敛
```

### **关键原理**

#### **1. 每轮使用新模型收集数据**

```python
# 迭代1: 用BC基线收集
run_policy_collect_states(model="bc_baseline.zip")

# 迭代2: 用迭代1的模型收集
run_policy_collect_states(model="dagger_iter_1.zip")

# 迭代3: 用迭代2的模型收集
run_policy_collect_states(model="dagger_iter_2.zip")
```

**为什么？**
- BC基线会犯错误A、B、C
- 迭代1修正了A、B，但可能在新场景D犯错
- 迭代2修正了D，探索到新场景E...

#### **2. 数据是累积的**

```
BC训练:
  数据 = 专家演示（10个episodes）

迭代1训练:
  数据 = 专家演示 + iter_1标注

迭代2训练:
  数据 = 专家演示 + iter_1标注 + iter_2标注

迭代3训练:
  数据 = 专家演示 + iter_1标注 + iter_2标注 + iter_3标注
```

**重要**: 不需要合并模型文件，数据已经自动累积！

#### **3. 只需要最终模型**

```bash
# ✅ 正确：使用最新的模型
python tools/dagger/evaluate_policy.py \
    --model checkpoints/dagger/harvest_1_log/dagger_iter_3.zip

# ❌ 错误：不需要合并模型
# 模型不需要合并，只用最新的即可
```

### **理论保证**

DAgger的性能界限:
```
ε(π_dagger) ≤ ε_expert + O(T·ε_BC)
```

vs 传统BC:
```
ε(π_BC) ≤ ε_expert + O(T²·ε_BC)  # 注意是T²！
```

**结论**: DAgger的误差增长是**线性**的，BC是**二次**的！

---

## 📦 **BC 基线训练**

### **数据目录结构**

```
data/expert_demos/harvest_1_log/
├── episode_000/
│   ├── frame_00000.npy  # {'observation': obs, 'action': action}
│   ├── frame_00001.npy
│   └── ...
├── episode_001/
├── episode_002/
└── ...
```

**验证数据**:
```bash
ls -l data/expert_demos/harvest_1_log/episode_000/ | head
# 应该看到 frame_00000.npy, frame_00001.npy, ...
```

### **录制专家演示** ⏱️ 40-60分钟

```bash
# 使用脚本自动录制（推荐）
bash scripts/run_dagger_workflow.sh \
    --task harvest_1_log \
    --num-episodes 10 \
    --iterations 0  # 只录制，不训练

# 或手动录制（键盘控制）
python tools/dagger/record_manual_chopping.py \
    --max-frames 500 \
    --camera-delta 1

# 或使用Pygame鼠标控制（更自然）
bash scripts/run_minedojo_x86.sh python tools/dagger/record_manual_chopping_pygame.py \
    --base-dir data/expert_demos/harvest_1_log \
    --max-frames 1000 \
    --mouse-sensitivity 0.5
```

**录制建议**:
- 每个任务录制10-50次演示
- 保持演示质量（不要失误）
- 覆盖不同场景（森林、平原、不同时间）
- 每次演示包含完整的任务流程

### **训练 BC 模型** ⏱️ 30-40分钟

```bash
# 基础训练（推荐）
python src/training/train_bc.py \
    --data data/expert_demos/harvest_1_log/ \
    --output checkpoints/dagger/harvest_1_log/bc_baseline.zip \
    --epochs 50 \
    --learning-rate 3e-4 \
    --batch-size 64

# 快速测试（10分钟）
python src/training/train_bc.py \
    --data data/expert_demos/harvest_1_log/ \
    --output checkpoints/dagger/harvest_1_log/bc_test.zip \
    --epochs 10

# 完整训练（2小时）
python src/training/train_bc.py \
    --data data/expert_demos/harvest_1_log/ \
    --output checkpoints/dagger/harvest_1_log/bc_final.zip \
    --epochs 200 \
    --learning-rate 5e-4
```

**训练参数说明**:

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data` | - | 数据目录（必需） |
| `--output` | - | 输出模型路径（必需） |
| `--epochs` | 50 | 训练轮数 |
| `--batch-size` | 32 | 批次大小 |
| `--learning-rate` | 0.001 | 学习率 |
| `--test-split` | 0.2 | 测试集比例 |

**预期输出**:
```
从目录加载: data/expert_demos/harvest_1_log
  找到 10 个episode目录
  [episode_000] 加载 234 个帧...
    ✓ episode_000: 成功加载 234 帧
  ...

总计:
  观察: (2073, 160, 256, 3)
  动作: (2073, 8)

开始训练...
Epoch 1/30: Loss=2.345, Accuracy=0.234
Epoch 2/30: Loss=1.987, Accuracy=0.345
...
Epoch 30/30: Loss=0.876, Accuracy=0.678

✓ 模型已保存: checkpoints/dagger/harvest_1_log/bc_baseline.zip
```

### **评估 BC 基线** ⏱️ 10分钟

```bash
bash scripts/run_minedojo_x86.sh python tools/dagger/evaluate_policy.py \
    --model checkpoints/dagger/harvest_1_log/bc_baseline.zip \
    --episodes 20 \
    --task harvest_1_log
```

**预期输出**:
```
Episode 1/20 ✓ | 步数:234 | 奖励:  1.00
Episode 2/20 ✗ | 步数:1000 | 奖励:  0.00
...

评估结果
============================================================
成功率: 60.0% (12/20)
平均奖励: 0.60 ± 0.49
平均步数: 487 ± 312
============================================================
```

**决策点**:
- ✅ 成功率 ≥ 50% → 进入 DAgger 优化
- ❌ 成功率 < 50% → 增加专家演示或调整超参数

**数据需求评估**:

| 任务复杂度 | 演示次数 | 总帧数 | 训练时间 | 预期成功率 |
|-----------|---------|--------|---------|-----------|
| 简单（砍树）| 10-20次 | 5K-10K | 10分钟 | 50-70% |
| 中等（建造）| 30-50次 | 20K-30K | 30分钟 | 40-60% |
| 复杂（探险）| 50-100次 | 50K-100K | 1-2小时 | 30-50% |

---

## 🎮 **录制工具使用**

### **方法对比**

| 特性 | 键盘控制 (IJKL+F) | Pygame鼠标控制 ⭐ |
|------|------------------|-----------------|
| 视角转动 | 离散（固定角度） | ✅ 连续平滑 |
| 攻击操作 | F键 | ✅ 左键更自然 |
| 操作直觉 | 需要记忆按键 | ✅ 类似FPS游戏 |
| 精确度 | 低 | ✅ 高 |
| 学习曲线 | 陡峭 | ✅ 平缓 |
| 数据质量 | IDLE 28.5% | ✅ IDLE <20% |

### **键盘控制（OpenCV）**

#### **控制说明**

**移动控制**:
- `W` - 前进 ✅
- `S` - 后退 ✅
- `A` - 左移 ✅
- `D` - 右移 ✅
- `Space` - 跳跃 ✅

**相机控制（改用IJKL）**:
- `I` - 向上看 ✅
- `K` - 向下看 ✅
- `J` - 向左看 ✅
- `L` - 向右看 ✅

> **为什么改用IJKL？**  
> OpenCV的方向键捕获不稳定，改用字母键更可靠

**动作**:
- `F` - 攻击/砍树 ⭐

**系统**:
- `Q` - 停止并保存 ✅
- `ESC` - 紧急退出（不保存）

#### **快速开始**

```bash
# 激活环境
conda activate minedojo-x86

# 运行录制（会自动清理旧数据）
python tools/dagger/record_manual_chopping.py \
    --output-dir data/expert_demos/harvest_1_log
```

#### **操作技巧**

**找树**:
1. 用`W`前进
2. 用`J`/`L`左右转头
3. 用`I`/`K`调整俯仰角

**砍树**:
1. 靠近树（`W`前进）
2. 调整视角对准树干（`IJKL`）
3. **按住`F`键**连续攻击
4. 看到"🎉 获得木头！"

**完成**:
按`Q`保存并退出

### **Pygame鼠标控制（推荐）⭐**

#### **控制说明**

| 控制方式 | 功能 | 说明 |
|---------|------|------|
| 鼠标移动 | 转动视角 | 上下左右自由查看 |
| 鼠标左键 | 攻击/挖掘 | 砍树、挖掘方块 |
| W/A/S/D | 移动 | 前后左右移动 |
| Space | 跳跃 | 跳过障碍 |
| Q | 重试 | 重新录制当前episode |
| ESC | 退出 | 退出程序 |

#### **快速开始**

```bash
conda activate minedojo-x86

# 使用默认鼠标灵敏度
bash scripts/run_minedojo_x86.sh python tools/dagger/record_manual_chopping_pygame.py \
    --base-dir data/expert_demos/harvest_1_log \
    --max-frames 1000

# 调整鼠标灵敏度
bash scripts/run_minedojo_x86.sh python tools/dagger/record_manual_chopping_pygame.py \
    --base-dir data/expert_demos/harvest_1_log \
    --max-frames 1000 \
    --mouse-sensitivity 0.8
```

#### **参数说明**

| 参数 | 默认值 | 范围 | 说明 |
|------|--------|------|------|
| `--mouse-sensitivity` | 0.5 | 0.1-2.0 | 鼠标灵敏度 |
| `--base-dir` | `data/expert_demos` | - | 保存目录 |
| `--max-frames` | 1000 | 1-10000 | 最大帧数 |
| `--fps` | 20 | 1-60 | 录制帧率 |

#### **鼠标灵敏度调整**

| 场景 | 灵敏度 | 说明 |
|------|--------|------|
| 新手 | 0.3 | 慢速，精确控制 |
| 默认 | 0.5 | 平衡，推荐使用 |
| 熟练 | 0.8 | 快速反应 |
| 高手 | 1.2 | 极快，需要适应 |

#### **操作流程**

**找树阶段**:
1. **视角控制**: 移动鼠标环顾四周
2. **移动**: 按住W键向树靠近
3. **调整视角**: 鼠标移动，让树在屏幕中央

**砍树阶段**:
1. **瞄准**: 鼠标移动，瞄准树干
2. **攻击**: 点击鼠标左键开始挖掘
3. **持续攻击**: 连续点击左键直到树被砍倒
4. **收集**: 靠近掉落的木头自动拾取

#### **使用技巧**

**找树技巧**:
- **缓慢移动鼠标**: 环顾四周找树
- **小幅度调整**: 精确瞄准树干
- **配合W键**: 边走边找

**砍树技巧**:
- **瞄准中心**: 鼠标移动，让树在屏幕中央
- **连续点击**: 左键快速连点
- **保持视角**: 砍树时不要移动鼠标

**录制技巧**:
- **保持焦点**: pygame窗口必须在前台
- **平滑操作**: 避免鼠标突然大幅移动
- **适当休息**: 录制间隙可以松开鼠标

---

## 🔄 **DAgger 迭代优化**

### **单轮 DAgger 流程**

每轮 DAgger 包含 4 个步骤：

#### **步骤1: 收集失败状态** ⏱️ 10分钟

```bash
python tools/dagger/run_policy_collect_states.py \
    --model checkpoints/dagger/harvest_1_log/bc_baseline.zip \
    --episodes 20 \
    --output data/policy_states/harvest_1_log/iter_1/ \
    --save-failures-only \
    --task harvest_1_log
```

**输出**:
```
Episode 1/20 ✗ 失败 | 步数:456 | 奖励:  0.00
Episode 2/20 ✓ 成功 | 步数:234 | 奖励:  1.00
...

收集完成！
============================================================
总episode数: 20
成功: 8 (40.0%)
失败: 12 (60.0%)
保存episode数: 12  # 只保存失败的
总状态数: 5432
============================================================
```

#### **步骤2: 智能标注失败场景** ⏱️ 30-40分钟

```bash
python tools/dagger/label_states.py \
    --states data/policy_states/harvest_1_log/iter_1/ \
    --output data/expert_labels/harvest_1_log/iter_1.pkl \
    --smart-sampling \
    --failure-window 10
```

#### **步骤3: 聚合数据并训练** ⏱️ 30-40分钟

```bash
python src/training/train_dagger.py \
    --iteration 1 \
    --base-data data/expert_demos/harvest_1_log/ \
    --new-data data/expert_labels/harvest_1_log/iter_1.pkl \
    --output checkpoints/dagger/harvest_1_log/dagger_iter_1.zip \
    --epochs 30
```

**输出**:
```
数据聚合
============================================================
基础数据: 4523 样本
新标注: 612 样本
聚合后: 5135 样本
============================================================

[训练过程...]

✓ 模型已保存: checkpoints/dagger/harvest_1_log/dagger_iter_1.zip
✓ 聚合数据已保存: data/dagger/harvest_1_log/combined_iter_1.pkl
```

#### **步骤4: 评估改进** ⏱️ 10分钟

```bash
bash scripts/run_minedojo_x86.sh python tools/dagger/evaluate_policy.py \
    --model checkpoints/dagger/harvest_1_log/dagger_iter_1.zip \
    --episodes 20
```

**预期结果**:
```
成功率: 75.0% (15/20)  # 从60%提升到75% ✅
```

### **多轮迭代**

重复上述 4 个步骤，但使用新模型：

```bash
# 迭代2
python tools/dagger/run_policy_collect_states.py \
    --model checkpoints/dagger/harvest_1_log/dagger_iter_1.zip \
    --episodes 20 \
    --output data/policy_states/harvest_1_log/iter_2/

python tools/dagger/label_states.py \
    --states data/policy_states/harvest_1_log/iter_2/ \
    --output data/expert_labels/harvest_1_log/iter_2.pkl

python src/training/train_dagger.py \
    --iteration 2 \
    --base-data data/dagger/harvest_1_log/combined_iter_1.pkl \
    --new-data data/expert_labels/harvest_1_log/iter_2.pkl \
    --output checkpoints/dagger/harvest_1_log/dagger_iter_2.zip

# 预期: 75% → 85% ✅
```

---

## 🎨 **标注工具使用**

### **标注界面**

```
开始标注 (543 个状态)  # 智能采样后
============================================================
控制:
  W/S/A/D    - 前进/后退/左/右
  I/K/J/L    - 视角 上/下/左/右
  F          - 攻击（砍树）
  Q          - 前进+攻击
  P          - 保持策略（重要！）⭐
  N          - 跳过此状态
  Z          - 撤销上一个标注
  X/ESC      - 完成标注
============================================================

[显示失败场景的画面]
Progress: 1/543  |  Priority: HIGH
Episode: 3  |  Step: 445
Policy Action: [0 0 0 16 12 0 0 0]  
>>> 向下看

你的标注: 按 'I' (应该向上看寻找树)
  ✓ [1/543] 向上看 -> Camera Up
```

### **完整控制参考**

#### **移动控制**

| 按键 | 动作 | 数组 |
|------|------|------|
| W | 前进 | `[1, 0, 0, 12, 12, 0, 0, 0]` |
| S | 后退 | `[2, 0, 0, 12, 12, 0, 0, 0]` |
| A | 左移 | `[0, 1, 0, 12, 12, 0, 0, 0]` |
| D | 右移 | `[0, 2, 0, 12, 12, 0, 0, 0]` |

#### **视角控制**

| 按键 | 动作 | 数组 |
|------|------|------|
| I | 向上看 | `[0, 0, 0, 8, 12, 0, 0, 0]` |
| K | 向下看 | `[0, 0, 0, 16, 12, 0, 0, 0]` |
| J | 向左看 | `[0, 0, 0, 12, 8, 0, 0, 0]` |
| L | 向右看 | `[0, 0, 0, 12, 16, 0, 0, 0]` |

#### **动作**

| 按键 | 动作 | 数组 |
|------|------|------|
| F | 攻击（砍树） | `[0, 0, 0, 12, 12, 3, 0, 0]` |
| Space | 跳跃 | `[0, 0, 1, 12, 12, 0, 0, 0]` |

#### **组合动作**

| 按键 | 动作 | 数组 |
|------|------|------|
| Q | 前进+攻击 | `[1, 0, 0, 12, 12, 3, 0, 0]` |
| E | 向上看+攻击 | `[0, 0, 0, 8, 12, 3, 0, 0]` |
| R | 前进+跳跃 ⭐ | `[1, 0, 1, 12, 12, 0, 0, 0]` |
| T | 后退+跳跃 | `[2, 0, 1, 12, 12, 0, 0, 0]` |
| G | 前进+跳跃+攻击 | `[1, 0, 1, 12, 12, 3, 0, 0]` |

#### **特殊控制**

| 按键 | 功能 | 说明 |
|------|------|------|
| **P** | **保持策略动作** ⭐ | 认为当前策略是对的，不做修改 |
| N | 跳过此状态 | 不标注，直接跳过 |
| Z | 撤销上一个标注 | 最多撤销10步 |
| X / ESC | 完成标注 | 保存并退出 |

### **P键使用示例**

当你认为策略预测的动作是正确的时，按 **P** 键保持不变：

```
当前画面: 正对着树
策略动作: [1 0 0 12 12 3 0 0] (Forward + Attack)

你的判断: 策略动作正确！

按键: P

结果: ✓ [5/100] PASS (保持策略动作: Forward + Attack)
```

**何时使用P键**:
- ✅ 策略已经在正确地前进
- ✅ 策略正确地攻击树木
- ✅ 策略的视角调整合理
- ✅ 任何你认为策略做得对的情况

**优势**:
- 节省标注时间（不需要手动输入相同的动作）
- 标注速度提升50%+
- 减少误操作

### **实际使用示例**

#### **场景1: 策略表现良好**

```
Progress: 5/100 | Priority: HIGH
Episode: 2 | Step: 45
Policy Action: [1 0 0 12 12 3 0 0]
>>> Forward + Attack

画面: 正在向树前进并攻击
你的判断: ✅ 策略动作完全正确

按键: P

输出: ✓ [5/100] PASS (保持策略动作: Forward + Attack)
```

#### **场景2: 需要跳跃靠近树**

```
Progress: 15/100 | Priority: HIGH  
Episode: 3 | Step: 20
Policy Action: [1 0 0 12 12 0 0 0]
>>> Forward

画面: 前方有个小坡，需要跳跃
你的判断: ❌ 应该前进+跳跃

按键: R

输出: ✓ [15/100] 前进+跳跃 -> Forward + Jump
```

#### **场景3: 策略在IDLE，需要前进**

```
Progress: 25/100 | Priority: HIGH
Episode: 5 | Step: 10  
Policy Action: [0 0 0 12 12 0 0 0]
>>> IDLE

画面: 看到树了，但没有移动
你的判断: ❌ 应该前进

按键: W

输出: ✓ [25/100] 前进 -> Forward
```

### **标注技巧**

#### **技巧1: 善用P键**

**不好的做法**（标注慢）:
```
策略: Forward -> 你输入: W
策略: Forward -> 你输入: W  
策略: Attack -> 你输入: F
策略: Forward+Attack -> 你输入: Q
```

**好的做法**（标注快）:
```
策略: Forward -> 你输入: P ✓
策略: Forward -> 你输入: P ✓
策略: Attack -> 你输入: P ✓
策略: Forward+Attack -> 你输入: P ✓
```

**结果**: 速度提升3倍！

#### **技巧2: 关注"高优先级"状态**

```
Priority: HIGH  <- 失败前的关键决策，仔细标注！
Priority: LOW   <- 成功episode的随机采样，可以快速用P
```

#### **技巧3: 发现错误立即撤销**

```
✓ [10/100] 前进 -> Forward
✓ [11/100] 攻击 -> Attack
✓ [12/100] 前进 -> Forward  <- 哦不，应该是前进+跳跃！

按键: Z (撤销)
输出: ↶ 撤销标注 (剩余: 11)

按键: R (重新标注)
输出: ✓ [12/100] 前进+跳跃 -> Forward + Jump
```

#### **技巧4: 批量标注相似状态**

如果连续多个状态都是相同的错误：
```
状态1: 策略=IDLE, 应该=前进 -> W
状态2: 策略=IDLE, 应该=前进 -> W  
状态3: 策略=IDLE, 应该=前进 -> W
状态4: 策略=Forward, 正确 -> P
状态5: 策略=Forward, 正确 -> P
```

找到模式后，快速输入！

### **标注质量检查**

#### **标注速度对比**

| 方法 | 每个状态耗时 | 100个状态总时间 |
|------|------------|----------------|
| 之前（全手动） | ~5秒 | ~8.3分钟 |
| 现在（使用P键） | ~2秒 | ~3.3分钟 |
| 提升 | **60%** | **60%** |

#### **标注准确性**

- ✅ 立即看到标注的动作描述
- ✅ 发现错误可以马上撤销
- ✅ 减少输入错误（P键替代手动输入）

---

## 🎓 **标注策略最佳实践**

### **🚨 常见陷阱：过度标注视角调整**

#### **问题场景**

```
帧1: 看不到树，策略=前进
     你的直觉: 应该左转找树
     你标注: 视角左移 (L键)

帧2: (还是帧1的画面，因为动作还没执行)
     你的直觉: 继续左转
     你标注: 视角左移 (L键)

帧3-5: 连续标注视角左移...
```

**导致的问题**: 模型学会原地转圈，很少前进！

### **✅ 正确的标注策略**

#### **策略1: 视角调整 + 前进（推荐）⭐**

**核心思想**: **环视是短期行为，移动是主要策略**

```
帧1: 看不到树，策略=前进
     你的判断: 需要稍微左转找树，同时继续前进
     你标注: J (向左看)

帧2: (画面开始变化)
     你的判断: 已经开始左转了，继续前进找树
     你标注: W (前进)

帧3: (继续变化)
     你标注: W (前进)

帧4: 看到树了！
     你标注: L (向右看) 微调

帧5: 树在视野中心
     你标注: W (前进)
```

**关键点**:
- ✅ 视角调整**最多1-2帧**
- ✅ 立即切换回**前进**
- ✅ 主要策略是**前进+环视**，而不是**原地转圈**

#### **策略2: 使用P键（保持策略）⭐**

如果策略是**前进**，很多时候应该**保持前进**：

```
帧1: 看不到树，策略=前进
     你的判断: 虽然看不到树，但应该继续探索前进
     你标注: P (保持策略 - 前进)

帧2-5: 策略=前进
     你标注: P, P, P, P

帧6: 看到树了！策略=前进
     你标注: P
```

**何时使用P键**:
- ✅ 策略正在前进探索 → P
- ✅ 策略正在靠近树 → P
- ✅ 策略正在攻击树 → P
- ❌ 策略在原地转圈 → 改为W（前进）

#### **策略3: 跳过（N键）+ 关键帧标注**

对于**重复的相似画面**，只标注关键帧：

```
帧1: 看不到树，策略=前进
     你标注: J (向左看)

帧2-4: (画面变化很小，还在左转)
     你标注: N, N, N (跳过)

帧5: 看到树了！
     你标注: W (前进)
```

### **📋 标注黄金法则**

#### **法则1: 1-2-10规则**

- **1帧**: 视角调整（最多2帧）
- **2帧**: 过渡/等待
- **10帧**: 前进/攻击等主要动作

**比例**: 视角调整应该<20%，前进应该>60%

#### **法则2: 主动探索优先**

```
不确定哪里有树？
-> 答案：前进探索！（不是原地转圈）
```

**标注选择**:
- ✅ W (前进) - 主动探索
- ✅ P (保持策略前进)
- ❌ 连续 J/L (原地转圈找树)

#### **法则3: P键是你的好朋友**

如果策略正在做**合理的事情**（即使不是最优），使用P键：

```
策略=前进，但方向不是最优
-> 你标注: P (前进总比原地转圈好)

策略=前进+攻击，虽然树不在正中心
-> 你标注: P (大方向对了)
```

#### **法则4: 从任务目标反推**

**任务目标**: 砍树获得木头

**拆解**:
1. 找到树（10-30步）
2. 靠近树（5-15步）
3. 对准树（1-3步）
4. 攻击树（10-30步）

**标注分配**:
- 60-70%: 前进（W, Q）
- 20-30%: 攻击（F, Q）
- 5-10%: 视角调整（I/J/K/L）
- 0-5%: 跳跃（R, G）

### **📊 标注质量自检**

标注100个状态后，统计一下：

```
W (前进):          40次 (40%)
Q (前进+攻击):     15次 (15%)
F (攻击):          10次 (10%)
P (保持策略):      20次 (20%)
J/L (左右看):      8次  (8%)
I/K (上下看):      3次  (3%)
N (跳过):          4次  (4%)
```

**健康的分布** ✅:
- 前进相关（W+Q+R）: 50-70%
- 攻击相关（F+Q+G）: 20-40%
- 视角调整（I/J/K/L）: <15%
- 保持策略（P）: 20-40%

**不健康的分布** ❌:
- 视角调整 > 30%
- 前进 < 40%
- P键使用 < 10%（说明你在过度干预）

#### **连续性检查**

看你的标注序列：

**不好的序列** ❌:
```
L, L, L, L, L, W, L, L, L  # 过多视角调整
```

**好的序列** ✅:
```
L, W, W, P, P, W, Q, Q, P  # 主要是前进
```

#### **策略一致性**

如果你发现自己：
- 连续3帧以上标注**同样的视角调整** → ❌ 有问题
- 连续10帧以上标注**前进相关** → ✅ 正常
- 连续5帧以上标注**攻击相关** → ✅ 正常

### **📈 预期效果**

**正确标注后的模型行为**:

✅ **应该看到**:
- Agent大部分时间在前进
- 偶尔转头环视（1-2帧）
- 看到树后快速靠近
- 靠近后开始攻击

❌ **不应该看到**:
- 原地转圈超过2秒
- 长时间只调整视角不移动
- 来回摆头

### **🎯 学习曲线**

| DAgger轮次 | BC基线质量 | 你的标注策略 | 预期成功率 |
|-----------|-----------|------------|-----------|
| 第1轮 | IDLE 75% | 60%修正，40% P键 | 40-50% |
| 第2轮 | IDLE 30% | 40%修正，60% P键 | 60-70% |
| 第3轮 | IDLE 10% | 20%修正，80% P键 | 80-90% |

**关键**: P键使用率应该**随着迭代增加**！

---

## 🔁 **继续训练**

### **使用场景**

#### **场景1: 训练3轮后继续训练**

```bash
# 第一次：完成3轮DAgger
bash scripts/run_dagger_workflow.sh \
    --task harvest_1_log \
    --num-episodes 10 \
    --iterations 3

# 结果: dagger_iter_3.zip (成功率 85%)

# 继续训练2轮（迭代4-5）
bash scripts/run_dagger_workflow.sh \
    --task harvest_1_log \
    --continue-from checkpoints/dagger/harvest_1_log/dagger_iter_3.zip \
    --iterations 5

# 结果: dagger_iter_5.zip (成功率 92%)
```

#### **场景2: BC效果太差，跳过BC直接继续DAgger**

```bash
# 第一次：只训练了BC
bash scripts/run_dagger_workflow.sh \
    --task harvest_1_log \
    --num-episodes 10 \
    --iterations 0  # 只做BC，不做DAgger

# BC评估: 成功率只有 40%，太低了！

# 追加录制5个episodes
bash scripts/run_dagger_workflow.sh \
    --task harvest_1_log \
    --num-episodes 15 \
    --append-recording \
    --skip-bc

# 重新训练BC
python src/training/train_bc.py \
    --data data/expert_demos/harvest_1_log/ \
    --output checkpoints/dagger/harvest_1_log/bc_baseline.zip \
    --epochs 50

# BC评估: 成功率 65%，好多了！

# 从BC开始DAgger（迭代1-3）
bash scripts/run_dagger_workflow.sh \
    --task harvest_1_log \
    --continue-from checkpoints/dagger/harvest_1_log/bc_baseline.zip \
    --start-iteration 1 \
    --iterations 3
```

#### **场景3: 分多天训练**

```bash
# 第1天：录制 + BC + 迭代1
bash scripts/run_dagger_workflow.sh \
    --task harvest_1_log \
    --iterations 1

# 第2天：继续迭代2
bash scripts/run_dagger_workflow.sh \
    --task harvest_1_log \
    --continue-from checkpoints/dagger/harvest_1_log/dagger_iter_1.zip \
    --iterations 2

# 第3天：继续迭代3-5
bash scripts/run_dagger_workflow.sh \
    --task harvest_1_log \
    --continue-from checkpoints/dagger/harvest_1_log/dagger_iter_2.zip \
    --iterations 5
```

### **继续训练专用参数**

| 参数 | 说明 | 必需 |
|------|------|------|
| `--continue-from MODEL` | 从指定模型继续训练 | ✅ 是 |
| `--start-iteration N` | 从第N轮开始（可选，自动推断）| ❌ 否 |
| `--iterations N` | 总迭代轮数（包含已完成的）| ✅ 是 |

### **自动推断起始迭代**

```bash
# 自动推断（推荐）✅
bash scripts/run_dagger_workflow.sh \
    --task harvest_1_log \
    --continue-from checkpoints/dagger/harvest_1_log/dagger_iter_3.zip \
    --iterations 5
# 自动检测: 上一轮为 iter_3，从 iter_4 开始

# 手动指定 ✅
bash scripts/run_dagger_workflow.sh \
    --task harvest_1_log \
    --continue-from checkpoints/dagger/harvest_1_log/dagger_iter_3.zip \
    --start-iteration 4 \
    --iterations 5
```

### **最佳实践**

#### **1. 渐进式训练**

```bash
# 不推荐: 一次性训练10轮 ❌
bash scripts/run_dagger_workflow.sh --task harvest_1_log --iterations 10
# 问题: 标注10轮非常累，而且可能浪费（早期就收敛了）

# 推荐: 分批训练 ✅
# 第1批: 3轮
bash scripts/run_dagger_workflow.sh --task harvest_1_log --iterations 3
# 评估成功率: 90%，还有提升空间

# 第2批: 继续2轮
bash scripts/run_dagger_workflow.sh \
    --continue-from checkpoints/dagger/harvest_1_log/dagger_iter_3.zip \
    --iterations 5
# 评估成功率: 95%，已经很好了，停止训练
```

#### **2. 评估驱动**

每次继续训练前，先评估当前模型：

```bash
# 评估迭代3
bash scripts/run_minedojo_x86.sh python tools/dagger/evaluate_policy.py \
    --model checkpoints/dagger/harvest_1_log/dagger_iter_3.zip \
    --episodes 50

# 如果成功率 >= 95%: 停止训练，已经足够好了
# 如果成功率 < 95%: 继续训练
```

---

## 🌍 **多任务支持**

### **任务隔离**

每个任务有独立的数据和模型目录：

```
data/expert_demos/
├── harvest_1_log/          # 砍树任务
│   ├── episode_000/
│   └── ...
└── harvest_1_wool/         # 获取羊毛任务
    ├── episode_000/
    └── ...

checkpoints/dagger/
├── harvest_1_log/
│   ├── bc_baseline.zip
│   ├── dagger_iter_1.zip
│   └── ...
└── harvest_1_wool/
    ├── bc_baseline.zip
    └── ...
```

### **使用场景**

#### **场景1: 训练新任务**

```bash
# 完整流程：录制 → BC → DAgger
bash scripts/run_dagger_workflow.sh \
    --task harvest_1_log \
    --num-episodes 10 \
    --iterations 3
```

#### **场景2: 追加录制数据**

```bash
# 第一次录制了 3 个 episodes，发现不够
bash scripts/run_dagger_workflow.sh \
    --task harvest_1_log \
    --num-episodes 3

# 追加录制 7 个（共 10 个）
bash scripts/run_dagger_workflow.sh \
    --task harvest_1_log \
    --num-episodes 10 \
    --append-recording \
    --skip-bc  # 跳过BC训练，稍后重新训练
```

#### **场景3: 多任务并行训练**

```bash
# 任务1: 砍树
bash scripts/run_dagger_workflow.sh \
    --task harvest_1_log \
    --num-episodes 10

# 任务2: 获取羊毛
bash scripts/run_dagger_workflow.sh \
    --task harvest_1_wool \
    --num-episodes 10

# 任务3: 挖石头
bash scripts/run_dagger_workflow.sh \
    --task harvest_10_cobblestone \
    --num-episodes 10
```

### **支持的任务**

常用任务（参考 `docs/reference/MINEDOJO_TASKS_REFERENCE.md`）：
- `harvest_1_log` - 砍1棵树
- `harvest_10_log` - 砍10棵树
- `harvest_1_wool` - 获取1个羊毛
- `harvest_10_cobblestone` - 挖10个圆石

---

## 🛠️ **脚本使用详解**

### **基础用法**

```bash
# 运行完整工作流（默认3次迭代）
bash scripts/run_dagger_workflow.sh
```

**执行流程**：
1. 手动录制专家演示（10-15个episode）
2. 训练BC基线模型
3. 评估BC成功率
4. DAgger迭代1：收集失败 → 标注 → 训练 → 评估
5. DAgger迭代2：收集失败 → 标注 → 训练 → 评估
6. DAgger迭代3：收集失败 → 标注 → 训练 → 评估
7. 显示完整训练历史

**预计时间**: 3-5小时（取决于迭代次数和标注速度）

### **完整参数列表**

#### **任务配置**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--task` | `harvest_1_log` | MineDojo任务ID |
| `--method` | `dagger` | 训练方法 (dagger/ppo/hybrid) |
| `--device` | `cpu` | 训练设备 (auto/cpu/cuda/mps) |

#### **录制配置**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--num-episodes` | `10` | 录制专家演示数量 |
| `--mouse-sensitivity` | `0.15` | 鼠标灵敏度 (0.1-2.0) |
| `--max-frames` | `6000` | 每个episode最大帧数 |
| `--no-skip-idle` | `false` | 保存所有帧（包括IDLE） |
| `--append-recording` | `false` | 追加录制（不覆盖已有数据） |

#### **训练配置**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--bc-epochs` | `50` | BC训练轮数 |
| `--iterations` | `3` | DAgger迭代次数 |
| `--collect-episodes` | `20` | 每轮收集episode数 |
| `--eval-episodes` | `20` | 评估episode数 |

#### **跳过步骤**

| 参数 | 说明 |
|------|------|
| `--skip-recording` | 跳过手动录制（假设已有数据） |
| `--skip-bc` | 跳过BC训练（假设已有BC模型） |

#### **继续训练**

| 参数 | 说明 |
|------|------|
| `--continue-from MODEL` | 从指定模型继续DAgger训练 |
| `--start-iteration N` | 从第N轮DAgger开始 |

### **常用命令组合**

```bash
# 完整流程
bash scripts/run_dagger_workflow.sh \
    --task harvest_1_log \
    --num-episodes 10 \
    --iterations 3

# 跳过录制，从BC开始
bash scripts/run_dagger_workflow.sh \
    --task harvest_1_log \
    --skip-recording \
    --iterations 3

# 追加录制数据
bash scripts/run_dagger_workflow.sh \
    --task harvest_1_log \
    --num-episodes 20 \
    --append-recording \
    --iterations 0

# 只录制，不训练
bash scripts/run_dagger_workflow.sh \
    --task harvest_1_log \
    --num-episodes 10 \
    --iterations 0

# 只训练BC基线，不做DAgger
bash scripts/run_dagger_workflow.sh \
    --task harvest_1_log \
    --skip-recording \
    --iterations 0

# 继续更多轮DAgger
bash scripts/run_dagger_workflow.sh \
    --task harvest_1_log \
    --skip-recording \
    --skip-bc \
    --continue-from checkpoints/dagger/harvest_1_log/dagger_iter_3.zip \
    --iterations 5

# 重新训练BC（更多epochs）
bash scripts/run_dagger_workflow.sh \
    --task harvest_1_log \
    --skip-recording \
    --bc-epochs 100 \
    --iterations 3

# 快速测试
bash scripts/run_dagger_workflow.sh \
    --iterations 1 \
    --bc-epochs 10 \
    --collect-episodes 10 \
    --eval-episodes 10
```

### **输出文件结构**

```
data/expert_demos/harvest_1_log/     # 专家演示数据
├── episode_000/
├── episode_001/
└── ...

data/policy_states/harvest_1_log/    # 策略收集的状态
├── iter_1/
├── iter_2/
└── iter_3/

data/expert_labels/harvest_1_log/    # 标注数据
├── iter_1.pkl
├── iter_2.pkl
└── iter_3.pkl

data/dagger/harvest_1_log/           # 聚合数据
├── combined_iter_1.pkl
├── combined_iter_2.pkl
└── combined_iter_3.pkl

checkpoints/dagger/harvest_1_log/    # 模型检查点
├── bc_baseline.zip                  # BC基线模型
├── bc_baseline_eval_results.npy     # BC评估结果
├── dagger_iter_1.zip                # DAgger迭代1
├── dagger_iter_1_eval_results.npy
├── dagger_iter_2.zip                # DAgger迭代2
├── dagger_iter_2_eval_results.npy
└── dagger_iter_3.zip                # DAgger迭代3 (最终)
```

### **控制台输出示例**

```
============================================================================
阶段1: BC基线训练
============================================================================

训练参数:
  数据目录: data/expert_demos/harvest_1_log
  训练轮数: 50
  学习率: 0.0003
  批次大小: 64

[训练过程...]

✓ BC训练完成: checkpoints/dagger/harvest_1_log/bc_baseline.zip

============================================================================
阶段2: 评估BC基线
============================================================================

ℹ️  评估BC策略 (20 episodes)...
✓ BC基线成功率: 65.0%

============================================================================
阶段3: DAgger迭代 1/3
============================================================================

ℹ️  [1] 步骤1: 收集策略失败状态...
✓ 状态收集完成: data/policy_states/harvest_1_log/iter_1

ℹ️  [1] 步骤2: 智能标注失败场景...
[标注界面...]
✓ 标注完成: data/expert_labels/harvest_1_log/iter_1.pkl

ℹ️  [1] 步骤3: 聚合数据并训练DAgger模型...
✓ DAgger训练完成: checkpoints/dagger/harvest_1_log/dagger_iter_1.zip

ℹ️  [1] 步骤4: 评估迭代 1 策略...
✓ 迭代 1 成功率: 78.0%

[迭代2, 3...]

============================================================================
训练完成！
============================================================================

训练历史:
  BC基线:       65.0%
  DAgger迭代1:  78.0%
  DAgger迭代2:  85.0%
  DAgger迭代3:  91.0%

最终模型: checkpoints/dagger/harvest_1_log/dagger_iter_3.zip
```

---

## ⚠️ **故障排查**

### **录制相关问题**

#### **Q1: 键盘按键没反应？**

**原因**: OpenCV窗口失去焦点

**解决**:
- 确保点击了OpenCV窗口
- 尝试快速连续按键
- 检查终端是否有错误信息

#### **Q2: 相机不动？**

**原因**: 使用了错误的按键

**解决**:
- 使用IJKL，不是方向键
- 快速按键，不要长按

#### **Q3: 鼠标移动视角不动？**

**原因**: pygame窗口失去焦点

**解决**: 点击pygame窗口重新获得焦点

#### **Q4: 鼠标太灵敏/太迟钝？**

**解决**: 调整`--mouse-sensitivity`参数

```bash
# 太灵敏 → 降低
--mouse-sensitivity 0.3

# 太迟钝 → 提高
--mouse-sensitivity 0.8
```

### **BC训练相关问题**

#### **Q5: BC成功率太低（<50%）**

**方案1: 增加专家演示**
```bash
# 追加录制更多数据
bash scripts/run_dagger_workflow.sh \
    --task harvest_1_log \
    --num-episodes 20 \
    --append-recording \
    --iterations 0

# 重新训练BC
python src/training/train_bc.py \
    --data data/expert_demos/harvest_1_log/ \
    --output checkpoints/dagger/harvest_1_log/bc_baseline.zip \
    --epochs 50
```

**方案2: 调整BC参数**
```bash
bash scripts/run_dagger_workflow.sh \
    --task harvest_1_log \
    --bc-epochs 100 \
    --skip-recording
```

#### **Q6: 报错 "未加载到任何数据"**

**原因**: 数据目录结构不正确

**检查**:
```bash
ls data/expert_demos/harvest_1_log/
# 应该看到: episode_000/, episode_001/, ...

ls data/expert_demos/harvest_1_log/episode_000/
# 应该看到: frame_00000.npy, frame_00001.npy, ...
```

**解决**: 确保使用最新的录制工具录制数据

#### **Q7: 训练Loss不下降**

**可能原因**:
1. 数据量太少（<1000帧）
2. 数据质量差（随机操作）
3. 学习率过高

**解决**:
```bash
# 降低学习率
python src/training/train_bc.py \
    --data data/expert_demos/harvest_1_log/ \
    --output checkpoints/dagger/harvest_1_log/bc_baseline.zip \
    --epochs 50 \
    --learning-rate 0.0001

# 或增加数据量（录制更多episode）
```

#### **Q8: 训练Accuracy很低**

**原因**: BC训练的Accuracy是逐维度匹配的准确率

**解释**:
- MineDojo动作空间有8个维度
- 每个维度有多个可能值（如camera有25个值）
- 完全匹配所有8个维度很困难
- **Accuracy 0.3-0.5 是正常的**

**关键指标**: 在环境中评估实际表现（成功率）

### **DAgger迭代相关问题**

#### **Q9: DAgger迭代没有提升**

**检查标注质量**:
- 标注是否正确？
- 是否标注了足够的关键状态？
- 视角调整是否过多？（应该<20%）

**调整参数**:
```bash
bash scripts/run_dagger_workflow.sh \
    --collect-episodes 30 \      # 收集更多失败
    --skip-recording \
    --skip-bc
```

**手动标注更多**:
- 关闭智能采样，标注所有状态
- 修改脚本中的 SMART_SAMPLING=false

#### **Q10: 标注太慢了**

**解决方案**:
- 使用 `--smart-sampling`（只标注20-30%）
- 使用 `--failure-window 5`（只标注失败前5步）
- 使用组合键（'Q'=前进+攻击）
- 跳过不确定的状态（按'N'）
- 多使用P键（保持策略）

#### **Q11: 标注时退出了，如何继续？**

DAgger的标注是可以中断的：

```bash
# 重新运行标注工具，会从上次中断处继续
bash scripts/run_minedojo_x86.sh python tools/dagger/label_states.py \
    --states data/policy_states/harvest_1_log/iter_1 \
    --output data/expert_labels/harvest_1_log/iter_1.pkl \
    --smart-sampling
```

### **模型行为相关问题**

#### **Q12: 模型原地转圈**

**原因**: 标注时视角调整过多

**解决**:
1. 检查标注分布（视角调整应该<20%）
2. 重新标注，使用"前进优先"原则
3. 多使用P键（保持策略）

#### **Q13: 模型大部分时间IDLE**

**原因**: BC基线太差（IDLE > 70%）

**解决**:
```bash
# 补录数据到50+ episodes
bash scripts/run_dagger_workflow.sh \
    --task harvest_1_log \
    --num-episodes 50 \
    --append-recording \
    --iterations 0

# 重新训练BC
bash scripts/run_dagger_workflow.sh \
    --task harvest_1_log \
    --skip-recording \
    --bc-epochs 100 \
    --iterations 3
```

### **脚本运行相关问题**

#### **Q14: 脚本中断了，如何继续？**

```bash
# 如果在BC之前中断
bash scripts/run_dagger_workflow.sh

# 如果BC已完成，在DAgger迭代中中断
bash scripts/run_dagger_workflow.sh --skip-recording --skip-bc
```

#### **Q15: 未找到数据/模型**

| 问题 | 快速解决 |
|------|----------|
| 未找到专家演示数据 | 移除`--skip-recording`或手动录制 |
| BC模型不存在 | 移除`--skip-bc`或手动训练BC |
| 标注文件不存在 | 重新运行标注步骤 |

#### **Q16: 如何查看训练进度？**

```bash
# 查看当前有哪些模型
ls -lh checkpoints/dagger/harvest_1_log/

# 评估特定模型
bash scripts/run_minedojo_x86.sh python tools/dagger/evaluate_policy.py \
    --model checkpoints/dagger/harvest_1_log/dagger_iter_2.zip \
    --episodes 20
```

---

## 📋 **命令速查表**

### **快速参考**

```bash
# ✅ 完整流程（首次训练）
bash scripts/run_dagger_workflow.sh --task harvest_1_log --num-episodes 10 --iterations 3

# ✅ 跳过录制（已有数据）
bash scripts/run_dagger_workflow.sh --task harvest_1_log --skip-recording --iterations 3

# ✅ 追加录制数据
bash scripts/run_dagger_workflow.sh --task harvest_1_log --num-episodes 20 --append-recording --iterations 0

# ✅ 只录制，不训练
bash scripts/run_dagger_workflow.sh --task harvest_1_log --num-episodes 10 --iterations 0

# ✅ 只训练BC，不做DAgger
bash scripts/run_dagger_workflow.sh --task harvest_1_log --skip-recording --iterations 0

# ✅ 继续更多轮DAgger
bash scripts/run_dagger_workflow.sh --task harvest_1_log --continue-from checkpoints/dagger/harvest_1_log/dagger_iter_3.zip --iterations 5

# ✅ 评估模型
bash scripts/run_minedojo_x86.sh python tools/dagger/evaluate_policy.py --model checkpoints/dagger/harvest_1_log/bc_baseline.zip --episodes 20

# ✅ 重新训练BC（更多epochs）
bash scripts/run_dagger_workflow.sh --task harvest_1_log --skip-recording --bc-epochs 100 --iterations 0

# ✅ 使用鼠标录制
bash scripts/run_minedojo_x86.sh python tools/dagger/record_manual_chopping_pygame.py --base-dir data/expert_demos/harvest_1_log --mouse-sensitivity 0.5
```

### **参数速查表**

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--task` | 任务ID | `harvest_1_log` |
| `--num-episodes` | 录制数量 | `10` |
| `--iterations` | DAgger轮数 | `3` |
| `--bc-epochs` | BC训练轮数 | `50` |
| `--skip-recording` | 跳过录制 | `false` |
| `--skip-bc` | 跳过BC训练 | `false` |
| `--append-recording` | 追加录制 | `false` |
| `--continue-from` | 继续训练的模型 | - |
| `--start-iteration` | 起始迭代 | 自动推断 |
| `--mouse-sensitivity` | 鼠标灵敏度 | `0.15` |
| `--collect-episodes` | 每轮收集数 | `20` |
| `--eval-episodes` | 评估数量 | `20` |

### **故障速查**

| 问题 | 快速解决 |
|------|----------|
| 未找到数据 | 移除`--skip-recording`或手动录制 |
| BC模型不存在 | 移除`--skip-bc`或手动训练BC |
| IDLE > 70% | 补录到50+ episodes |
| 标注中断 | 重新运行相同命令会继续 |
| 模型原地转圈 | 检查标注分布，视角调整应<20% |
| 键盘没反应 | 点击OpenCV窗口获得焦点 |
| 鼠标不灵敏 | 调整`--mouse-sensitivity` |

---

## 📈 **性能预期**

### **成功率曲线**

```
成功率
  100% ┤
       │
   90% ┤                             ●─── 迭代3
       │                         ╱
   80% ┤                     ●───     迭代2
       │                 ╱
   70% ┤             ●───             迭代1
       │         ╱
   60% ┤─────●                        BC基线
       │
   50% ┤
       └─────┬─────┬─────┬─────┬─────→ 时间
          BC   迭代1  迭代2  迭代3  迭代4
```

### **时间预期**

| 阶段 | 成功率 | 时间 | IDLE占比 |
|------|--------|------|----------|
| 录制专家演示 | - | 40-60分钟 | - |
| BC基线 | 50-65% | 30-40分钟 | < 30% |
| DAgger迭代1 | 70-78% | 60-80分钟 | < 20% |
| DAgger迭代2 | 80-85% | 60-80分钟 | < 10% |
| DAgger迭代3 | 85-92% | 60-80分钟 | < 5% |

**总计**: 4-5小时达到90%+成功率

### **数据量预期**

| 轮次 | 数据量 | 标注时间 | 成功率 | 提升 |
|------|--------|---------|--------|------|
| 初始BC | 5K | 40分钟 | 60% | - |
| DAgger-1 | 7K | +30分钟 | 75% | +15% |
| DAgger-2 | 9K | +30分钟 | 85% | +10% |
| DAgger-3 | 11K | +20分钟 | 90% | +5% |
| DAgger-4 | 12K | +20分钟 | 92% | +2% |

---

## 🎯 **总结**

### **DAgger 核心优势**

1. ✅ 解决分布偏移问题
2. ✅ 性能优于纯BC
3. ✅ 理论保证（线性误差增长）
4. ✅ 适合Minecraft等复杂任务
5. ✅ 数据利用率高

### **关键要点**

1. **数据是累积的** - 每轮训练使用所有之前的数据
2. **只需要最终模型** - 不需要合并模型文件
3. **标注要高效** - 使用智能采样和P键
4. **前进优先** - 视角调整<20%，前进>60%
5. **渐进式训练** - 分批评估，不要一次训练太多轮
6. **使用鼠标录制** - 数据质量更高，更自然

### **推荐工作流**

```bash
# 第一次使用DAgger（推荐）
# 步骤1: 录制20个高质量演示（使用鼠标）
bash scripts/run_minedojo_x86.sh python tools/dagger/record_manual_chopping_pygame.py \
    --base-dir data/expert_demos/harvest_1_log \
    --mouse-sensitivity 0.5

# 步骤2: 训练BC + 3轮DAgger
bash scripts/run_dagger_workflow.sh \
    --task harvest_1_log \
    --skip-recording \
    --iterations 3

# 步骤3: 评估结果
bash scripts/run_minedojo_x86.sh python tools/dagger/evaluate_policy.py \
    --model checkpoints/dagger/harvest_1_log/dagger_iter_3.zip \
    --episodes 50
```

### **下一步**

1. ✅ 实现状态收集工具
2. ✅ 实现交互式标注工具
3. ✅ 运行第一轮DAgger
4. 评估效果决定是否继续
5. 迁移到其他任务
6. （可选）BC + PPO 微调

---

## 📚 **相关资源**

### **项目文档**
- `docs/guides/MULTI_EPISODE_RECORDING_GUIDE.md` - 多轮录制指南
- `docs/reference/MINEDOJO_TASKS_REFERENCE.md` - 任务列表
- `docs/reference/LABELING_KEYBOARD_REFERENCE.md` - 标注键盘参考

### **理论基础**
- **DAgger原论文**: Ross, S., Gordon, G., & Bagnell, D. (2011). "A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning" (AISTATS 2011)
- **论文链接**: https://arxiv.org/abs/1011.0686
- **OpenAI VPT**: https://arxiv.org/abs/2206.11795

### **代码库**
- **imitation库**: https://imitation.readthedocs.io/
- **MineRL比赛**: https://minerl.io/
- **MineRL数据集**: https://minerl.io/dataset/

---

**版本**: 3.0.0 (完整综合版)  
**创建日期**: 2025-10-24  
**最后更新**: 2025-10-24  

**祝训练顺利！** 🚀

---

> **提示**: 这是一个完整的一站式指南，包含了从理论到实践的所有内容。建议先阅读"快速开始"和"DAgger核心概念"，然后根据需要查阅具体章节。
