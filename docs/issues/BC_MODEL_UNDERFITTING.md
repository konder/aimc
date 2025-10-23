# BC模型欠拟合问题诊断与解决方案

> **问题**: BC评估时角色几乎静止不动，虽然预测动作是"前进"  
> **状态**: 诊断完成，待验证解决方案  
> **日期**: 2025-10-23

---

## 📋 问题描述

### 症状
- BC模型评估时，持续预测动作 `[1 0 0 12 12 0 0 0]` (前进)
- 但游戏画面中角色几乎不动
- 偶尔在水面跳跃，大部分时间卡住

### 游戏环境
- 任务: `harvest_1_log`
- 生物群系: 沙漠
- 健康值: 满

---

## ✅ 已排除的问题

### 1. 数据质量 ✅

通过 `tools/analyze_bc_data.py` 分析：

```
总数据: 101 episodes, 765 frames
前进帧: 304 (39.7%)  ← 合理
静止帧: 0   (0.0%)   ← 优秀！
攻击帧: 398 (52.0%)  ← 合理
跳跃帧: 8   (1.0%)   ← 偏低但可接受
视角移动: 81 (10.6%) ← 合理
```

**结论:** 数据质量良好，使用了 `--skip-idle-frames`，没有静止帧问题。

---

### 2. 动作映射 ✅

```python
录制数据: [1 0 0 12 12 0 0 0] = 前进
模型预测: [1 0 0 12 12 0 0 0] = 前进
```

**结论:** 动作空间映射正确，不是格式问题。

---

## ⚠️ 真正的问题：BC模型欠拟合

### 核心原因

#### 1. 模型没有学会"视觉→动作"的映射

**观察:**
- 数据分布: 52%攻击，40%前进，11%视角，1%跳跃
- 模型预测: 100%前进

**问题:**
模型只学到了"前进是高频动作"的统计规律，而没有学会：
- 看到树 → 应该前进+攻击
- 没看到树 → 应该调整视角
- 靠近树 → 应该攻击
- 在水里 → 应该跳跃+前进

#### 2. 为什么"前进"但角色不动？

**Minecraft游戏机制:**
- 单纯"前进"遇到障碍会卡住
- 在水里，单纯前进移动很慢
- 需要组合动作才能有效移动：
  - 前进 + 跳跃（水里）
  - 前进 + 视角调整（绕过障碍）
  - 前进 + 攻击（清除树木）

**BC模型的问题:**
只学会输出单一动作"前进"，无法根据环境适应性地组合动作。

---

### BC训练参数分析

**当前配置:**
```python
epochs = 50
learning_rate = 3e-4
batch_size = 32
数据量: 765 frames
```

**问题:**
- **Epochs太少**: 对于pixel-based观察空间，50个epochs远远不够
- **数据量偏少**: 765帧对于学习复杂的视觉策略不足
- **没有处理类别不平衡**: 攻击52%，前进40%，但模型倾向输出前进

---

## 💡 解决方案

### 方案1: 增加训练轮数 ⭐ (立即尝试)

**理论依据:**
- 视觉策略学习需要更多迭代
- NatureCNN需要充分训练才能提取有效特征

**推荐配置:**
```bash
epochs = 200-500
learning_rate = 3e-4  (保持不变)
batch_size = 32       (保持不变)
```

**命令:**
```bash
python src/training/train_bc.py \
    --data-dir data/expert_demos/harvest_1_log \
    --output-path checkpoints/dagger/harvest_1_log/bc_baseline_200ep.zip \
    --epochs 200 \
    --batch-size 32 \
    --device mps
```

**预期:**
- 训练时间: 10-20分钟
- Loss应该持续下降
- 模型开始预测更多样化的动作

**评估:**
```bash
python tools/dagger/evaluate_policy.py \
    --model checkpoints/dagger/harvest_1_log/bc_baseline_200ep.zip \
    --episodes 10
```

---

### 方案2: 降低学习率 + 增加训练时间

**理论依据:**
- 更小的学习率避免过快收敛到局部最优
- 更精细的梯度更新

**推荐配置:**
```bash
epochs = 300
learning_rate = 1e-4  (降低)
batch_size = 32
```

**命令:**
```bash
python src/training/train_bc.py \
    --data-dir data/expert_demos/harvest_1_log \
    --output-path checkpoints/dagger/harvest_1_log/bc_baseline_lr1e4.zip \
    --epochs 300 \
    --batch-size 32 \
    --learning-rate 1e-4 \
    --device mps
```

---

### 方案3: 增加数据量 (最有效，但费时)

**理论依据:**
- BC是有监督学习，数据量是性能上限
- 需要覆盖更多场景和状态

**目标:**
```
当前: 101 episodes, 765 frames
目标: 200+ episodes, 2000+ frames
```

**需要覆盖的场景:**
- 不同角度看树（正面、侧面、远处、近处）
- 不同类型的树（Oak、Spruce、Birch等）
- 不同地形（平地、山坡、水边）
- 不同起始位置

**命令:**
```bash
bash scripts/run_dagger_workflow.sh \
    --task harvest_1_log \
    --num-episodes 100 \
    --append-recording
```

---

### 方案4: 使用DAgger迭代 ⭐⭐⭐ (核心方案)

**BC的根本局限:**
- 只能学习专家演示的状态分布
- 遇到未见过的状态（如卡在墙角）无法处理
- 存在"Covariate Shift"问题

**DAgger的优势:**
1. 让模型自己运行，收集它会遇到的状态
2. 专家对这些"难"状态进行标注
3. 模型学会从错误中恢复

**流程:**
```
BC模型运行 → 收集"卡住"的状态 → 人工标注正确动作 → 
重新训练 → 模型学会处理这些状态 → 重复
```

**命令:**
```bash
bash scripts/run_dagger_workflow.sh \
    --task harvest_1_log \
    --skip-recording \
    --skip-bc \
    --continue-from checkpoints/dagger/harvest_1_log/bc_baseline.zip \
    --start-iteration 1 \
    --iterations 3
```

**标注重点:**
- 模型一直前进撞墙 → 标注"调整视角"或"后退"
- 模型到树前不攻击 → 标注"攻击"
- 模型在水里不跳 → 标注"前进+跳跃"

---

## 🎯 推荐行动计划

### 阶段1: 快速验证（立即执行）

```bash
# 1. 重新训练BC (200 epochs)
python src/training/train_bc.py \
    --data-dir data/expert_demos/harvest_1_log \
    --output-path checkpoints/dagger/harvest_1_log/bc_baseline_200ep.zip \
    --epochs 200 \
    --device mps

# 2. 评估
python tools/dagger/evaluate_policy.py \
    --model checkpoints/dagger/harvest_1_log/bc_baseline_200ep.zip \
    --episodes 10
```

**判断标准:**
- ✅ 如果动作多样化，继续增加epochs到500
- ❌ 如果仍然单一动作，进入阶段2

---

### 阶段2: DAgger迭代（如果阶段1无效）

```bash
bash scripts/run_dagger_workflow.sh \
    --task harvest_1_log \
    --skip-recording \
    --skip-bc \
    --continue-from checkpoints/dagger/harvest_1_log/bc_baseline_200ep.zip \
    --start-iteration 1 \
    --iterations 5
```

**预期:**
- 迭代1: 收集大量"卡住"状态，成功率可能仍很低
- 迭代2-3: 开始学会基本动作组合
- 迭代4-5: 成功率显著提升

---

### 阶段3: 数据增强（长期方案）

```bash
# 继续录制更多episode
bash scripts/run_dagger_workflow.sh \
    --task harvest_1_log \
    --num-episodes 100 \
    --append-recording
```

然后从头重新训练BC和DAgger。

---

## 📊 为什么BC会学到"一直前进"？

### 从模型角度理解

**BC的训练目标:**
```
最小化: CrossEntropy(predicted_action, expert_action)
```

**如果模型视觉特征提取不足:**
```python
# 简化的伪代码
def policy(observation):
    # 视觉特征提取失败
    features = extract_features(obs)  # 提取的特征信息量不足
    
    # 无法根据画面区分场景，退化为统计猜测
    if random() < 0.4:
        return "forward"  # 40%概率
    elif random() < 0.52:
        return "attack"   # 52%概率
    ...
```

**实际情况:**
模型可能学到了一个"安全"的策略：
- "前进"在40%的时间是正确的
- 相比其他更复杂的动作组合，"前进"更容易学习
- 模型选择了最简单但次优的策略

### 解决方法

**增加训练强度:**
- 更多epochs → 强制模型学习更复杂的特征
- 更低学习率 → 更精细的学习

**增加数据多样性:**
- 更多场景 → 模型无法只靠"前进"应对所有情况
- DAgger → 强制模型面对它不擅长的状态

---

## 📈 预期效果对比

### 当前BC模型（50 epochs）
```
动作分布:
  前进: 95%
  其他: 5%

成功率: 0-5%
```

### 改进后BC模型（200 epochs）
```
动作分布:
  前进: 40-50%
  攻击: 30-40%
  视角: 10-15%
  跳跃: 5%

成功率: 10-20%
```

### DAgger迭代3轮后
```
动作分布:
  根据场景动态调整

成功率: 30-50%
```

---

## 🔧 调试工具

### 1. 分析BC训练数据
```bash
python tools/analyze_bc_data.py
```

### 2. 观察BC训练过程
```bash
# 训练时观察loss变化
# 如果loss<0.5后不再下降，说明已收敛
# 如果loss持续下降，说明需要更多epochs
```

### 3. 评估时打印详细动作
```bash
# 在evaluate_policy.py中已经打印了预测动作
# 观察动作的多样性
```

---

## 📚 相关文档

- `docs/guides/DAGGER_DETAILED_GUIDE.md` - DAgger详细指南
- `docs/guides/DAGGER_LABELING_STRATEGY.md` - DAgger标注策略
- `docs/technical/DAGGER_CNN_ARCHITECTURE.md` - BC/DAgger使用的CNN架构
- `tools/analyze_bc_data.py` - BC数据分析工具

---

## ✅ 验收标准

### BC训练成功的标志

**训练过程:**
- Loss从2-3降到0.5以下
- Loss曲线平滑下降
- 无NaN或爆炸

**评估效果:**
- 动作分布接近训练数据分布
- 至少50%的episodes中有攻击动作
- 至少30%的episodes中有视角调整
- 成功率 >10%

**如果不满足:**
继续增加epochs或进入DAgger迭代

---

**最后更新**: 2025-10-23  
**下次审查**: 验证方案1效果后更新

