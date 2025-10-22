# MineCLIP Prompt优化测试结果

**测试时间**: 2025-10-21  
**测试数据**: 272帧手动录制的砍树序列  
**测试方法**: 15种不同prompt的MineCLIP相似度对比

---

## 📊 **核心发现**

### **🎯 Top 5 最佳Prompt**

| 排名 | Prompt | 变化范围 | 变化百分比 | 评价 |
|------|--------|---------|-----------|------|
| 🥇 1 | **punching tree** | 0.0058 | **2.17%** | ⭐ 最佳 |
| 🥈 2 | **cutting wood** | 0.0054 | **1.99%** | ⭐ 很好 |
| 🥉 3 | **breaking tree** | 0.0049 | **1.88%** | ⭐ 很好 |
| 4 | **mining wood** | 0.0044 | **1.58%** | ✅ 良好 |
| 5 | **chopping a tree with hand** | 0.0043 | **1.57%** | ✅ 良好（当前使用）|

### **❌ 表现最差的Prompt**

| 排名 | Prompt | 变化范围 | 变化百分比 | 评价 |
|------|--------|---------|-----------|------|
| 15 | tree | 0.0013 | 0.47% | ❌ 太低 |
| 14 | oak tree | 0.0013 | 0.47% | ❌ 太低 |
| 13 | tree in front | 0.0013 | 0.47% | ❌ 太低 |
| 12 | a tree | 0.0013 | 0.49% | ❌ 太低 |
| 11 | facing a tree | 0.0015 | 0.53% | ❌ 太低 |

---

## 🔍 **深度分析**

### **关键发现1: 动作描述 >> 物体描述**

**动作类prompt表现最好**:
- `punching tree` (2.17%) ✅
- `cutting wood` (1.99%) ✅
- `breaking tree` (1.88%) ✅
- `mining wood` (1.58%) ✅

**物体类prompt表现最差**:
- `tree` (0.47%) ❌
- `oak tree` (0.47%) ❌
- `a tree` (0.49%) ❌

**结论**: MineCLIP更擅长识别"动作过程"而非"静态物体"

---

### **关键发现2: 简单动词 > 复杂描述**

| Prompt类型 | 示例 | 变化范围 | 结论 |
|-----------|------|---------|------|
| 简单动作 | `punching tree` | **2.17%** | ✅ 最好 |
| 简单动作 | `breaking tree` | **1.88%** | ✅ 很好 |
| 复杂描述 | `chopping a tree with hand` | 1.57% | ⚠️ 较差 |
| 物体名词 | `tree` | 0.47% | ❌ 最差 |

**结论**: 越简单、越直接的动作描述效果越好

---

### **关键发现3: 相似度变化范围仍然偏低**

**与理想目标对比**:

| 指标 | 最佳prompt | 最低要求 | 理想目标 | 达标? |
|------|-----------|---------|---------|------|
| 变化范围 | 2.17% | >1% | >5% | ⚠️ 勉强达标 |
| 绝对变化 | 0.0058 | >0.01 | >0.05 | ❌ 未达标 |

**结论**: 即使最佳prompt，变化范围仍然**低于理想值**

---

## 📈 **相似度曲线分析**

### **Top 5 Prompt的相似度变化趋势**

从图表可以看出：

1. **`punching tree` (红色, 2.17%)**:
   - 初始相似度: ~0.280
   - 峰值相似度: ~0.283
   - 趋势: 明显上升趋势（50-150帧）

2. **`cutting wood` (橙色, 1.99%)**:
   - 初始相似度: ~0.270
   - 峰值相似度: ~0.275
   - 趋势: 稳定上升

3. **`breaking tree` (绿色, 1.88%)**:
   - 初始相似度: ~0.261
   - 峰值相似度: ~0.265
   - 趋势: 中期上升，后期平稳

4. **`mining wood` (紫色, 1.58%)**:
   - 相似度较高（~0.270-0.271）
   - 变化相对平缓

5. **`chopping a tree with hand` (蓝色, 1.57%)**:
   - 当前使用的prompt
   - 表现中等偏下

**关键观察**: 所有prompt的相似度曲线都呈现"上升趋势"，说明MineCLIP确实能感知到任务进展，但变化幅度很小。

---

## ✅ **实用建议**

### **建议1: 更换为最佳Prompt** ⭐ 推荐

**立即行动**:
```python
# 在 src/training/train_get_wood.py 中修改
task_prompt = "punching tree"  # 从 "chopping a tree with hand" 改为这个
```

**预期效果**:
- MineCLIP奖励信号增强 **38%** (从1.57%提升到2.17%)
- 训练稳定性可能略有提升
- 但仍然**不足以作为主要奖励信号**

---

### **建议2: 采用混合奖励策略** ⭐⭐⭐ 强烈推荐

即使使用最佳prompt，2.17%的变化范围仍然太小。建议：

```python
# 混合奖励设计
dense_reward = 0

# MineCLIP奖励（降低权重）
mineclip_reward = similarity_diff * 10.0  # 从40.0降到10.0

# 环境事件奖励（提高权重）
if inventory.contains_wood:
    dense_reward += 100.0  # 收集到木头
elif is_attacking and facing_tree:
    dense_reward += 5.0    # 正在砍树
elif facing_tree and distance < 3.0:
    dense_reward += 1.0    # 靠近树木
elif can_see_tree:
    dense_reward += 0.1    # 看到树木

total_reward = sparse_reward + mineclip_reward + dense_reward
```

**预期效果**:
- 更明确的训练信号
- 更快的收敛速度
- 更高的最终性能

---

### **建议3: 尝试模仿学习** ⭐⭐⭐ 最佳长期方案

基于当前结果，**模仿学习可能是更好的选择**：

**理由**:
1. 你已经录制了272帧高质量演示数据
2. MineCLIP信号太弱（<2.5%），不足以单独使用
3. 模仿学习可以直接从演示中学习策略
4. 预期训练速度提升**5-10倍**

**快速验证**（3-4小时）:
1. 录制5-10次高质量演示
2. 实现简单BC训练
3. 对比效果

详见: [`docs/guides/IMITATION_LEARNING_GUIDE.md`](../guides/IMITATION_LEARNING_GUIDE.md)

---

## 🎯 **结论和决策**

### **MineCLIP单独使用的可行性评估**

| 方面 | 评分 | 说明 |
|------|------|------|
| 技术可行性 | ⭐⭐⭐⭐ | 实现正确，16帧视频模式有效 |
| 信号强度 | ⭐⭐ | 2.17%变化范围偏低 |
| 训练效果 | ⭐⭐ | 可以辅助，但不足以单独使用 |
| 通用性 | ⭐⭐⭐⭐ | 不需要手动设计奖励 |
| **总评** | **⭐⭐⭐** | **可用但不理想** |

---

### **推荐方案排序**

#### **🥇 方案1: 模仿学习 (BC + PPO)**
- **优势**: 最快、最稳定、最终性能最好
- **劣势**: 需要额外录制数据和实现
- **时间**: 3-4小时快速验证，15-20小时完整实施
- **预期**: 训练速度提升**5-10倍**

#### **🥈 方案2: 混合奖励 (MineCLIP + 环境事件)**
- **优势**: 立即可用，效果好于纯MineCLIP
- **劣势**: 需要手动设计环境事件奖励
- **时间**: 2-3小时实现
- **预期**: 训练速度提升**2-3倍**

#### **🥉 方案3: 优化后的MineCLIP**
- **优势**: 最简单，只需改prompt
- **劣势**: 效果提升有限（+38%信号强度）
- **时间**: 5分钟
- **预期**: 训练速度提升**10-20%**

---

## 📝 **立即可做的事**

### **最小改动（5分钟）**:

```bash
# 1. 更新prompt为最佳配置
vim src/training/train_get_wood.py
# 修改: task_prompt = "punching tree"

# 2. 重新训练10K steps测试效果
./scripts/train_get_wood.sh --total-steps 10000
```

### **推荐改动（2-3小时）**:

1. 实现混合奖励策略
2. 降低MineCLIP权重（40 → 10）
3. 添加环境事件奖励
4. 训练对比实验

### **长期方案（15-20小时）**:

1. 按照[模仿学习路线图](IMITATION_LEARNING_ROADMAP.md)实施
2. 录制20次高质量演示
3. 实现BC预训练
4. PPO微调优化

---

## 🔬 **额外实验建议**

### **实验1: 测试极端场景**

用3张图片测试MineCLIP的区分度：
- 全天空场景
- 全树木特写
- 半天空半树木

**预期**: 如果相似度差异<0.1，说明MineCLIP在Minecraft场景中区分度确实较低

### **实验2: 测试提示词组合**

尝试更具体的动作描述：
- `player punching tree trunk`
- `hand hitting wood block`
- `tree breaking animation`

**预期**: 可能进一步提升到2.5-3%

### **实验3: 对比不同pool_type**

测试`avg` pooling vs `attn` pooling:
```bash
# 修改模型配置测试
pool_type = "avg"  # 当前是 "attn.d2.nh8.glusw"
```

**预期**: 可能有5-10%差异

---

## 📚 **相关文档**

- **当前状态**: [`MINECLIP_STATUS_SUMMARY.md`](MINECLIP_STATUS_SUMMARY.md)
- **优化策略**: [`MINECLIP_OPTIMIZATION_STRATEGY.md`](MINECLIP_OPTIMIZATION_STRATEGY.md)
- **模仿学习**: [`IMITATION_LEARNING_ROADMAP.md`](IMITATION_LEARNING_ROADMAP.md)
- **完整指南**: [`../guides/IMITATION_LEARNING_GUIDE.md`](../guides/IMITATION_LEARNING_GUIDE.md)

---

**最后更新**: 2025-10-21  
**数据来源**: `logs/mineclip_optimization/prompt_optimization.png`  
**测试工具**: `tools/quick_optimize_mineclip.py`

