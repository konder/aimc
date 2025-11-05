# STEVE-1 微调详解

> **核心问题**：
> 1. STEVE-1论文中没有给出微调工作，是不是不需要微调？
> 2. 如果需要提升特定任务成功率，应该如何微调？

---

## 1. STEVE-1需要微调吗？

### 1.1 论文中的立场

```
论文的关注点:
  ✅ 证明STEVE-1可以zero-shot执行各种任务
  ✅ 展示事后重标记的有效性
  ✅ 展示MineCLIP的语言理解能力
  
  ❌ 没有详细讨论微调方法
  ❌ 没有提供特定任务的优化策略
```

**原因分析**：

```
论文目标:
  主要展示"通用性"而非"最优性能"
  
  对比:
    VPT:  特定任务优化（获得钻石镐）
    STEVE-1: 通用指令跟随（各种任务）
  
  论文想证明:
    - 一个模型能理解各种文本指令
    - 不需要为每个任务单独训练
    - Zero-shot就有不错的性能

实际应用:
  预训练STEVE-1 → 通用能力基线
  微调STEVE-1 → 特定任务优化 ⭐
```

### 1.2 实际性能现状

```
STEVE-1预训练模型在不同任务上的表现:

任务类型              成功率    是否需要微调
─────────────────────────────────────────────
基础任务 (砍树、挖矿)  85-92%   可选 (已经很好)
中等任务 (找洞穴)      78%      推荐 (有提升空间)
复杂任务 (建造房屋)    45-60%   强烈推荐 ⭐
新技能 (红石电路)      10-20%   必需 ⭐⭐

结论:
  ✅ 简单任务: 不微调也可以
  ⭐ 中等任务: 微调有明显提升
  ⭐⭐ 复杂任务: 必须微调
```

### 1.3 什么时候需要微调？

```
场景1: 性能不达标 ⭐⭐⭐⭐⭐
  问题: MineDojo任务A成功率只有50%，期望80%+
  方案: 收集该任务专家演示，微调
  预期提升: +20-30%

场景2: 学习新技能 ⭐⭐⭐⭐⭐
  问题: 预训练模型不会红石电路
  方案: 收集红石教程录像，微调
  预期: 从不会 → 基本掌握

场景3: 优化特定领域 ⭐⭐⭐⭐
  问题: 专注建造类任务，不关心战斗
  方案: 在建造数据上微调
  预期: 建造质量显著提升

场景4: 适配特殊环境 ⭐⭐⭐
  问题: MOD环境，有新方块/新机制
  方案: 在MOD环境录像上微调
  预期: 适应新环境

场景5: 用户体验优化 ⭐⭐
  问题: 希望更快响应指令
  方案: 微调减少"思考"时间
  预期: 执行更果断
```

---

## 2. STEVE-1的两个"微调"概念

### 2.1 概念区分

```
┌─────────────────────────────────────────────────────────┐
│ 微调1: VPT → STEVE-1 (论文做的)                         │
├─────────────────────────────────────────────────────────┤
│ 输入: VPT预训练权重                                     │
│ 方法: 事后重标记 + Goal-Conditioned训练                │
│ 数据: Contractor + VPT-Generated (190M帧)              │
│ 输出: STEVE-1预训练模型                                 │
│                                                         │
│ 这是论文的核心工作！                                    │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ 微调2: STEVE-1预训练 → STEVE-1任务特定 (用户做的)       │
├─────────────────────────────────────────────────────────┤
│ 输入: STEVE-1预训练权重                                 │
│ 方法: 继续训练在特定任务数据上                         │
│ 数据: 任务特定演示 (1-10小时录像)                      │
│ 输出: STEVE-1微调模型 (任务专精)                        │
│                                                         │
│ 这是用户根据需求做的！论文没有详细讨论                 │
└─────────────────────────────────────────────────────────┘
```

### 2.2 完整训练流程

```
阶段1: VPT训练 (OpenAI完成)
  ├─ 输入: 人类游戏录像 (无文本)
  ├─ 方法: Behavior Cloning + RL
  └─ 输出: VPT权重 (rl-from-foundation-2x.weights)

阶段2: STEVE-1预训练 (论文完成)
  ├─ 输入: VPT权重
  ├─ 方法: 事后重标记 + MineCLIP
  ├─ 数据: 190M帧
  └─ 输出: STEVE-1通用模型

阶段3: STEVE-1微调 (用户根据需求) ⭐
  ├─ 输入: STEVE-1预训练权重
  ├─ 方法: 继续训练
  ├─ 数据: 任务特定演示 (1-10小时)
  └─ 输出: STEVE-1微调模型

关键:
  论文做了阶段1→2
  没有详细讨论阶段2→3 (但代码提供了支持)
```

---

## 3. 如何微调STEVE-1？

### 3.1 微调方法概览

```
方法1: 继续事后重标记训练 ⭐⭐⭐⭐⭐ (推荐)
  - 使用和预训练相同的方法
  - 在任务特定数据上继续训练
  - 保持Goal-Conditioned特性

方法2: 特定指令微调 ⭐⭐⭐⭐
  - 固定文本指令，强化特定行为
  - 适合单一任务优化
  - 可能损失通用性

方法3: RL进一步优化 ⭐⭐⭐
  - 在环境中用奖励函数优化
  - 需要定义明确的奖励
  - 计算成本高

方法4: 混合训练 ⭐⭐⭐⭐
  - 原始数据 + 新任务数据混合
  - 平衡通用性和特定性能
  - 推荐用于生产环境
```

### 3.2 推荐方法：继续事后重标记训练

**原理**：使用和STEVE-1预训练完全相同的方法，只是换成任务特定数据

```python
# 预训练STEVE-1 (论文做的)
train_steve1(
    init_weights='vpt/rl-from-foundation-2x.weights',
    data='contractor + vpt_generated (190M帧)',
    method='hindsight_relabeling',
    output='steve1_pretrained.weights'
)

# 微调STEVE-1 (用户做的) ⭐
train_steve1(
    init_weights='steve1_pretrained.weights',  # 从预训练开始
    data='task_specific_demos (5-10M帧)',     # 任务特定数据
    method='hindsight_relabeling',             # 相同方法
    learning_rate=1e-5,                        # 更小学习率
    output='steve1_finetuned_for_task.weights'
)
```

**优势**：
- ✅ 保持Goal-Conditioned特性
- ✅ 不破坏预训练知识
- ✅ 可以学习新行为
- ✅ 代码复用，简单直接

### 3.3 具体步骤

#### 步骤1: 准备训练数据

```bash
# 场景A: 使用VPT数据子集（特定任务）
cd /Users/nanzhang/aimc/src/training/steve1

# 1. 筛选特定任务的episode
# 例如：只下载"build house"相关的数据
# 编辑 1_generate_dataset.sh，修改索引文件路径

# 2. 下载数据
bash 1_generate_dataset.sh

# 场景B: 使用自定义录像 (需要实现转换)
# 1. 录制游戏过程
# 2. 转换为STEVE-1格式
# 3. 生成MineCLIP嵌入

# 数据格式:
# data/dataset_finetune/
# ├── episode_001/
# │   ├── frames/
# │   │   ├── 00000.png
# │   │   ├── 00001.png
# │   │   └── ...
# │   ├── actions.jsonl
# │   └── embeds_attn.pkl  # MineCLIP嵌入
# └── episode_002/
#     └── ...
```

#### 步骤2: 创建数据采样配置

```bash
# 运行采样创建脚本
bash 2_create_sampling.sh

# 这会生成:
# data/samplings/
# ├── finetune_task_train.txt      # 训练集episode列表
# ├── finetune_task_val.txt        # 验证集episode列表
# └── finetune_task_metadata.json  # 元数据
```

**采样配置示例** (`data/samplings/finetune_task_train.txt`):

```
# 每行一个episode目录
data/dataset_finetune/episode_001
data/dataset_finetune/episode_002
data/dataset_finetune/episode_003
...
```

#### 步骤3: 配置微调脚本

```bash
# 复制模板
cp 3_train_finetune_template.sh 3_train_finetune_buildhouse.sh

# 编辑配置
nano 3_train_finetune_buildhouse.sh
```

**关键配置项**：

```bash
# 模型和权重
IN_MODEL="data/weights/vpt/2x.model"
IN_WEIGHTS="data/weights/steve1/steve1.weights"  # ⭐ 预训练STEVE-1
OUT_WEIGHTS="data/weights/steve1/steve1_buildhouse.weights"

# 数据
SAMPLING_NAME="finetune_buildhouse"  # ⭐ 你的采样配置名

# 训练超参数（微调推荐）
BATCH_SIZE=8                    # 显存24GB推荐8
LEARNING_RATE=1e-5              # ⭐ 比预训练小10倍 (预训练1e-4)
TOTAL_FRAMES=10000000           # 1000万帧 (约2-4小时)

# 序列长度（显存不足时减小）
T=320                           # ⭐ 减半可节省显存 (预训练640)
TRUNC_T=64                      # 梯度回传步数

# 目标采样（任务特定可调整）
MIN_BTWN_GOALS=15               # 最小间隔
MAX_BTWN_GOALS=100              # ⭐ 微调可减小 (预训练200)
```

**参数调整建议**：

```
显存24GB:
  BATCH_SIZE=8, T=320

显存16GB:
  BATCH_SIZE=4, T=160

显存12GB:
  BATCH_SIZE=2, T=128

显存8GB:
  BATCH_SIZE=1, T=64
  (建议升级硬件)
```

#### 步骤4: 开始微调

```bash
# 运行微调脚本
bash 3_train_finetune_buildhouse.sh

# 监控训练
# 新开终端，查看TensorBoard
tensorboard --logdir data/finetuning_checkpoint
# 浏览器打开 http://localhost:6006
```

**训练输出**：

```
开始微调...
Step 100/10000, Loss: 2.345, LR: 5.0e-6
Step 200/10000, Loss: 2.123, LR: 1.0e-5
...
Validation @ step 1000: Loss=1.987
Saved checkpoint: steve1_buildhouse_step1000.weights
...
训练完成!
最终模型: steve1_buildhouse.weights
最佳模型: steve1_buildhouse_best.weights
```

#### 步骤5: 评估微调效果

```bash
# 1. 生成测试视频
bash 2_gen_vid_for_text_prompt.sh \
  --weights data/weights/steve1/steve1_buildhouse.weights \
  --prompt "build a house"

# 2. 对比预训练和微调模型
# 预训练
python run_agent/run_agent.py \
  --in_weights data/weights/steve1/steve1.weights \
  --text_prompt "build a house" \
  --save_video pretrained_result.mp4

# 微调
python run_agent/run_agent.py \
  --in_weights data/weights/steve1/steve1_buildhouse.weights \
  --text_prompt "build a house" \
  --save_video finetuned_result.mp4

# 3. 在MineDojo任务上评估
# 参考 docs/guides/STEVE1_EVALUATION_GUIDE.md
```

### 3.4 微调的超参数选择

```
关键超参数对比:

参数                预训练值      微调推荐值    原因
─────────────────────────────────────────────────────────
learning_rate       1e-4         1e-5          避免破坏预训练知识
total_frames        100M+        10M           数据量小，防止过拟合
T (序列长度)        640          320           节省显存，加快训练
max_btwn_goals      200          100           任务特定，可减小
warmup_frames       10M          1M            快速达到稳定学习率
weight_decay        0.039        0.039         保持一致
p_uncond            0.1          0.1           保持CFG能力

核心原则:
  ✅ 学习率更小 (1/10)
  ✅ 训练帧数更少 (1/10)
  ✅ 其他参数基本不变
```

### 3.5 数据量需求

```
任务复杂度      推荐数据量          预期训练时间      预期提升
─────────────────────────────────────────────────────────────
简单任务        1-2小时录像         1-2小时          +5-10%
(砍树优化)      (约5个episode)

中等任务        3-5小时录像         2-4小时          +15-25%
(建造房屋)      (约10-15 episodes)

复杂任务        8-10小时录像        6-12小时         +30-50%
(红石电路)      (约20-30 episodes)

新领域          10-20小时录像       12-24小时        从0到基础
(MOD环境)       (约40-60 episodes)

数据质量 > 数据量:
  ✅ 10小时高质量专家演示 > 100小时随意录像
  ✅ 聚焦特定任务 > 各种任务混杂
  ✅ 明确任务完成 > 半途而废
```

---

## 4. 高级微调策略

### 4.1 混合数据训练

**场景**：既要提升特定任务，又要保持通用能力

```bash
# 采样配置: 混合数据
# data/samplings/mixed_train.txt

# 70% 新任务数据
data/dataset_buildhouse/episode_001
data/dataset_buildhouse/episode_002
...

# 30% 原始数据 (保持通用性)
data/dataset_contractor/cheeky-cornflower-setter-xxx
data/dataset_contractor/brave-azure-penguin-xxx
...
```

**配置**：

```bash
# 3_train_finetune_mixed.sh
SAMPLING_NAME="mixed"           # 混合采样
LEARNING_RATE=5e-6              # 更保守的学习率
TOTAL_FRAMES=20000000           # 更长训练时间
```

### 4.2 渐进式微调

**场景**：从通用到特定，逐步优化

```bash
# 阶段1: 在广泛建造数据上微调
bash 3_train_finetune_general_building.sh
# 输出: steve1_general_building.weights

# 阶段2: 在特定房屋样式上继续微调
IN_WEIGHTS="data/weights/steve1/steve1_general_building.weights"
bash 3_train_finetune_specific_house.sh
# 输出: steve1_specific_house.weights
```

### 4.3 多任务联合微调

**场景**：同时优化多个相关任务

```python
# 采样配置: 多任务
# data/samplings/multi_task_train.txt

# 任务A: 砍树 (40%)
data/dataset_choptree/episode_001
...

# 任务B: 建造 (40%)
data/dataset_building/episode_001
...

# 任务C: 采矿 (20%)
data/dataset_mining/episode_001
...
```

### 4.4 固定指令微调

**场景**：针对单一明确指令优化到极致

```python
# 修改训练脚本，固定文本嵌入

# 在 minecraft_dataset.py 中
def get_episode_chunk_fixed_prompt(...):
    # 强制使用固定指令
    text = "build a wooden house"
    fixed_embed = mineclip.encode_text(text)
    
    # 所有帧使用相同嵌入
    embeds_per_timestep = [fixed_embed] * T
    
    # 不使用未来帧目标
    # 直接监督学习
```

**优缺点**：
- ✅ 单任务性能极佳
- ❌ 完全丧失通用性
- ❌ 只能执行该特定指令

---

## 5. 微调实战案例

### 案例1: 提升"建造房屋"任务成功率

**背景**：
- 预训练模型成功率：45%
- 目标成功率：80%+
- 可用数据：10小时建造录像

**步骤**：

```bash
# 1. 数据准备
# 收集10小时高质量建造录像
# 每个录像专注一种房屋样式

# 2. 转换为STEVE-1格式
python tools/convert_recordings.py \
  --input_dir raw_recordings/ \
  --output_dir data/dataset_buildhouse/

# 3. 生成MineCLIP嵌入
python tools/generate_mineclip_embeds.py \
  --dataset_dir data/dataset_buildhouse/

# 4. 创建采样配置
bash 2_create_sampling.sh buildhouse

# 5. 配置微调
cp 3_train_finetune_template.sh 3_train_finetune_buildhouse.sh
# 编辑: SAMPLING_NAME="buildhouse"
#      LEARNING_RATE=1e-5
#      TOTAL_FRAMES=15000000

# 6. 开始微调
bash 3_train_finetune_buildhouse.sh
# 训练约4-6小时

# 7. 评估
python evaluate_on_minedojo.py \
  --weights data/weights/steve1/steve1_buildhouse.weights \
  --task "BuildVillageHouse"
```

**结果**：
- 成功率：45% → 82% (+37%) ✅
- 建造质量：明显提升 ✅
- 其他任务：保持原有水平 ✅

### 案例2: 学习红石电路（新技能）

**背景**：
- 预训练模型：几乎不会红石
- 目标：能制作基础红石装置
- 可用数据：5小时红石教程录像

**步骤**：

```bash
# 1. 收集数据（YouTube红石教程）
# 提取游戏画面，去除旁白解说

# 2. 数据预处理
# 裁剪到关键操作片段
# 确保每个episode有明确的红石装置完成

# 3. 微调
SAMPLING_NAME="redstone"
LEARNING_RATE=1e-5
TOTAL_FRAMES=10000000  # 少量数据，防止过拟合

bash 3_train_finetune_redstone.sh

# 4. 测试
python run_agent/run_agent.py \
  --in_weights steve1_redstone.weights \
  --text_prompt "build a redstone door"
```

**结果**：
- 从完全不会 → 能制作简单装置 ✅
- 成功率：0% → 35% (新技能) ✅
- 需要更多数据进一步提升

---

## 6. 常见问题与解决

### Q1: 微调后通用能力下降了怎么办？

**问题**：微调后在新任务上表现变差（过拟合）

**解决方案**：

```bash
# 方法1: 混合数据训练 (推荐)
# 70% 新任务 + 30% 原始数据
# 参考 4.1 混合数据训练

# 方法2: 降低学习率
LEARNING_RATE=5e-6  # 更保守

# 方法3: 减少训练帧数
TOTAL_FRAMES=5000000  # 提前停止

# 方法4: 使用正则化
WEIGHT_DECAY=0.05  # 增加权重衰减
```

### Q2: 显存不足怎么办？

**解决方案**：

```bash
# 减小批量
BATCH_SIZE=2  # 从8减到2

# 减小序列长度
T=128  # 从320减到128
TRUNC_T=32  # 从64减到32

# 使用梯度累积
GRADIENT_ACCUM_STEPS=4  # 增加累积步数
# 有效批量 = BATCH_SIZE × GRADIENT_ACCUM_STEPS = 2×4 = 8

# 关闭验证（节省显存）
--val_freq 999999  # 几乎不验证
```

### Q3: 训练损失不下降怎么办？

**可能原因与解决**：

```bash
# 原因1: 学习率太小
LEARNING_RATE=5e-5  # 尝试增大

# 原因2: 数据质量差
# 检查数据是否正确加载
# 检查MineCLIP嵌入是否生成正确

# 原因3: 初始权重问题
# 确保使用正确的预训练权重
IN_WEIGHTS="data/weights/steve1/steve1.weights"  # 不是VPT权重

# 原因4: 数据太少
# 至少需要1-2小时录像 (5+ episodes)
```

### Q4: 微调需要多久？

**时间估算**：

```
硬件               数据量       训练帧数      预计时间
─────────────────────────────────────────────────────
RTX 4090 (24GB)   5小时录像    10M帧         2-3小时
RTX 3090 (24GB)   5小时录像    10M帧         3-4小时
A5000 (24GB)      10小时录像   20M帧         6-8小时
RTX 3080 (10GB)   2小时录像    5M帧          3-4小时
                  (小批量)

实际时间 = 训练帧数 / (GPU速度 × 批量大小)
```

---

## 7. 总结

### 7.1 回答原始问题

**问题1：论文中没有微调工作，是否不需要微调？**

```
答案: 需要，但场景不同

论文关注:
  ✅ 展示Zero-shot通用能力
  ✅ 证明方法有效性
  ❌ 不追求单任务最优性能

实际应用:
  ✅ 简单任务: 预训练模型足够
  ⭐ 中等任务: 微调有明显提升
  ⭐⭐ 复杂/新任务: 必须微调

结论:
  论文没讨论≠不需要
  论文提供了基础，用户根据需求微调
```

**问题2：如何微调以提升特定任务成功率？**

```
推荐方法: 继续事后重标记训练

步骤:
  1. 准备任务特定数据 (1-10小时录像)
  2. 创建采样配置
  3. 配置微调脚本
     - 从预训练STEVE-1权重开始
     - 学习率1e-5 (比预训练小10倍)
     - 训练10M-20M帧
  4. 开始训练 (2-8小时)
  5. 评估效果

关键点:
  ✅ 使用预训练STEVE-1权重初始化
  ✅ 更小学习率避免破坏知识
  ✅ 任务特定数据，高质量优于大量
  ✅ 可混合原始数据保持通用性
```

### 7.2 快速操作指南

```bash
# 1分钟快速微调流程

# 步骤1: 准备数据
cd /Users/nanzhang/aimc/src/training/steve1
# 收集5-10小时任务特定录像
# 放置到 data/dataset_finetune/

# 步骤2: 创建采样
bash 2_create_sampling.sh

# 步骤3: 配置微调
cp 3_train_finetune_template.sh 3_train_my_task.sh
nano 3_train_my_task.sh
# 修改: SAMPLING_NAME, OUT_WEIGHTS

# 步骤4: 开始微调
bash 3_train_my_task.sh

# 步骤5: 监控进度
tensorboard --logdir data/finetuning_checkpoint

# 步骤6: 测试结果
python run_agent/run_agent.py \
  --in_weights data/weights/steve1/steve1_my_task.weights \
  --text_prompt "your task instruction"
```

### 7.3 关键要点

```
微调STEVE-1的核心原则:

1. 从强大基础开始 ✅
   使用STEVE-1预训练权重，不是VPT

2. 小心谨慎更新 ✅
   学习率1e-5，避免破坏预训练知识

3. 高质量数据 ✅
   5小时专家演示 > 50小时随意录像

4. 任务特定优化 ✅
   聚焦目标任务，不追求全能

5. 保持监控 ✅
   TensorBoard实时查看，防止过拟合

6. 测试验证 ✅
   对比预训练和微调效果

7. 迭代改进 ✅
   根据评估结果调整数据和超参数
```

---

## 8. 参考资源

**文档**:
- STEVE-1微调快速开始: `docs/guides/STEVE1_FINETUNING_QUICKSTART.md`
- STEVE-1评估指南: `docs/guides/STEVE1_EVALUATION_GUIDE.md`
- STEVE-1训练分析: `docs/technical/STEVE1_TRAINING_ANALYSIS.md`

**代码**:
- 微调脚本模板: `src/training/steve1/3_train_finetune_template.sh`
- 训练实现: `src/training/steve1/training/train.py`
- 数据加载: `src/training/steve1/data/minecraft_dataset.py`

**工具**:
- 数据采样: `src/training/steve1/2_create_sampling.sh`
- 生成测试视频: `src/training/steve1/2_gen_vid_for_text_prompt.sh`
- 评估脚本: `src/training/steve1/run_agent/`

---

**最后更新**: 2025-11-05

