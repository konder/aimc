# MineCLIP 配置优化策略

> **当前状态**: MineCLIP相似度变化范围过小（只有0.35%），无法提供有效的训练信号
> 
> **目标**: 通过系统性测试找到最佳配置，使MineCLIP相似度变化范围达到5-10%以上

---

## 📊 **当前问题诊断**

###  **用户录制的砍树序列分析**

- ✅ **录制成功**: 272帧完整的寻找树木 → 接近树木 → 面对树木 → 攻击砍树序列
- ❌ **相似度变化太小**: 
  - 范围: 0.2682 - 0.2717
  - 变化: 只有0.0035 (0.35%)
  - 平均: 0.2702

### **问题表现**

从相似度曲线看：
- **前期（0-30帧）**: 寻找树木，相似度0.2682-0.2689（低）
- **中期（40-80帧）**: 面对树木并攻击，相似度上升到峰值0.2717（高）
- **后期（100-130帧）**: 继续攻击，相似度0.2698-0.2715（中）

**虽然有趋势，但变化幅度太小！**

### **可能的原因**

1. ❓ **Prompt问题**: `"chopping a tree with hand"` 太复杂或不够精确
2. ❓ **配置问题**: Resolution、Pool Type、Frame Sampling等参数不合适
3. ❓ **模型限制**: MineCLIP在这个简单任务上本身区分度就较低

---

## 🔧 **优化策略**

### **Phase 1: Prompt优化（正在进行）** ⏳

**测试的Prompt类型**:

| 类型 | 示例 | 假设 |
|------|------|------|
| 当前使用 | `chopping a tree with hand` | 基线对比 |
| 简单物体 | `tree`, `oak tree`, `tree trunk` | 更直接的视觉描述 |
| 视觉位置 | `looking at tree`, `tree in front` | 强调视觉中的位置 |
| 动作描述 | `breaking tree`, `punching tree`, `mining wood` | 强调动作过程 |
| 场景描述 | `forest`, `trees in minecraft` | 更宏观的场景 |

**期望**:
- 找到相似度变化范围最大的prompt（目标: >2%）
- 找到与砍树过程最匹配的描述方式

**工具**: `tools/quick_optimize_mineclip.py`

```bash
conda run -n minedojo-x86 python tools/quick_optimize_mineclip.py
```

**输出**:
- `logs/mineclip_optimization/prompt_optimization.png` - 对比图
- `logs/mineclip_optimization/prompt_results.txt` - 详细结果

---

### **Phase 2: 参数优化（待定）**

根据Phase 1的最佳prompt，测试其他参数：

1. **Pool Type**: `attn.d2.nh8.glusw` vs `avg`
2. **Frame Sampling**: stride=1 vs 2 vs 4
3. **Video Length**: 8帧 vs 16帧 vs 32帧

**工具**: `tools/optimize_mineclip_config.py`（完整版）

---

### **Phase 3: 混合策略（如果单独MineCLIP不足）**

如果优化后MineCLIP的相似度变化范围仍然<2%，考虑：

1. **MineCLIP + 环境事件奖励**:
   ```python
   # 示例
   dense_reward = 0.0
   
   # MineCLIP奖励（权重降低）
   dense_reward += similarity_diff * 0.5
   
   # 环境事件奖励（权重提高）
   if facing_tree:
       dense_reward += 0.1
   if attacking:
       dense_reward += 0.05
   if tree_health_decreasing:
       dense_reward += 0.2
   ```

2. **Curriculum Learning**:
   - 阶段1: 靠近树木（MineCLIP: "facing a tree"）
   - 阶段2: 攻击树木（MineCLIP: "breaking tree" + 环境奖励）
   - 阶段3: 收集木头（稀疏奖励为主）

---

## 📈 **成功标准**

### **最低要求**

| 指标 | 当前值 | 目标值 | 状态 |
|------|--------|--------|------|
| 相似度变化范围 | 0.0035 (0.35%) | >0.01 (>1%) | ❌ |
| 峰值与谷值差异 | 0.0035 | >0.02 | ❌ |
| 标准差 | ~0.001 | >0.005 | ❌ |

### **理想标准**

| 指标 | 目标值 |
|------|--------|
| 相似度变化范围 | >0.05 (>5%) |
| 峰值（砍树时） | >0.35 |
| 谷值（寻找树时） | <0.25 |
| 与任务进度相关性 | 明显正相关 |

---

## 🎯 **预期结果**

### **Scenario 1: 找到有效Prompt** ✅

如果某个prompt的相似度变化范围>2%:

```python
# 在train_get_wood.py中更新task_prompt
task_prompt = "最佳prompt"  # 例如: "tree trunk"
```

然后重新训练，观察:
- MineCLIP奖励是否有明显变化
- 智能体是否更快学会靠近树木
- 训练稳定性是否提高

### **Scenario 2: 所有Prompt效果都不佳** ⚠️

如果所有prompt的相似度变化范围都<1%:

**结论**: MineCLIP在"砍树"这个简单任务上的视觉区分度确实较低

**解决方案**:
1. 使用**混合奖励**（MineCLIP + 环境事件）
2. 调整MineCLIP权重降低（例如从40.0降到10.0）
3. 引入**课程学习**（分阶段训练）
4. 考虑使用**模仿学习**预训练

### **Scenario 3: MineCLIP根本无效** ❌

如果相似度几乎不变（<0.001）:

**可能原因**:
1. 模型权重加载有问题
2. 归一化参数错误
3. MineCLIP模型本身不适合这个任务

**验证方法**:
```bash
# 测试极端场景对比
python tools/verify_mineclip_16frames.py \
  --frames1 logs/test_frames/sky_only \
  --frames2 logs/test_frames/tree_close
```

期望: 相似度差异>0.1

---

## 📝 **当前行动**

1. ✅ **已完成**: 
   - 创建MineCLIP配置优化工具
   - 修复模型加载问题（state_dict提取、键名处理）
   - 创建简化版prompt测试工具

2. ⏳ **正在进行**:
   - 运行15个不同prompt的相似度测试
   - 生成对比图和详细结果

3. ⏸️ **等待结果**:
   - 分析哪个prompt效果最好
   - 决定下一步优化方向

---

## 🔍 **相关文件**

| 文件 | 说明 |
|------|------|
| `tools/record_manual_chopping.py` | 手动录制砍树序列工具 |
| `tools/verify_mineclip_16frames.py` | 验证MineCLIP相似度计算 |
| `tools/quick_optimize_mineclip.py` | 快速prompt优化工具 |
| `tools/optimize_mineclip_config.py` | 完整配置优化工具 |
| `logs/my_chopping/` | 用户录制的砍树序列（272帧）|
| `logs/mineclip_optimization/` | 优化测试结果输出目录 |
| `docs/MINEDOJO_ACTION_MAPPING.md` | MineDojo动作映射文档 |

---

**最后更新**: 2025-10-21

**下一步**: 等待prompt优化测试完成 → 分析结果 → 选择最佳配置或调整策略

