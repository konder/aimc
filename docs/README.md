# 📚 AIMC 文档中心

欢迎来到 AIMC 项目文档！文档已按功能分类组织，便于快速查找。

---

## 🎯 **快速开始 - 推荐阅读顺序**

### **想立即开始DAgger训练？**
1. 📖 **[DAGGER_QUICK_START.md](guides/DAGGER_QUICK_START.md)** ⭐⭐⭐ **强烈推荐！**
   - 完整的3-5小时训练流程
   - 从60%提升到90%成功率
   - 包含所有命令和预期输出

2. 📊 [DAGGER_VS_BC_COMPARISON.md](guides/DAGGER_VS_BC_COMPARISON.md)
   - 可视化理解DAgger优势
   
3. 📚 [DAGGER_DETAILED_GUIDE.md](guides/DAGGER_DETAILED_GUIDE.md)
   - 深入理解算法原理

---

## 🗂️ 目录结构

### 📊 **status/** - 当前状态和进展
实时更新的项目状态、问题分析和优化策略

| 文档 | 说明 | 优先级 |
|------|------|--------|
| [DAGGER_IMPLEMENTATION_PLAN.md](status/DAGGER_IMPLEMENTATION_PLAN.md) | DAgger详细实施计划 | ⭐⭐⭐ 🆕 |
| [IMITATION_LEARNING_ROADMAP.md](status/IMITATION_LEARNING_ROADMAP.md) | 模仿学习实施路线图 | ⭐⭐⭐ |
| [MINECLIP_STATUS_SUMMARY.md](status/MINECLIP_STATUS_SUMMARY.md) | MineCLIP当前状态总结 | ⭐⭐ |
| [MINECLIP_PROMPT_OPTIMIZATION_RESULTS.md](status/MINECLIP_PROMPT_OPTIMIZATION_RESULTS.md) | MineCLIP提示词优化结果 | ⭐⭐ |
| [MINECLIP_OPTIMIZATION_STRATEGY.md](status/MINECLIP_OPTIMIZATION_STRATEGY.md) | MineCLIP优化策略详解 | ⭐ |

---

### 📖 **guides/** - 操作指南
从入门到进阶的完整操作指南

| 文档 | 说明 | 适用人群 |
|------|------|----------|
| **DAgger 模仿学习** | | |
| [DAGGER_QUICK_START.md](guides/DAGGER_QUICK_START.md) | **DAgger快速开始（推荐！）** | ⭐⭐⭐ 🆕 |
| [DAGGER_LABELING_STRATEGY.md](guides/DAGGER_LABELING_STRATEGY.md) | **DAgger标注策略指南（必读！）** | ⭐⭐⭐ 🔥 |
| [DAGGER_DETAILED_GUIDE.md](guides/DAGGER_DETAILED_GUIDE.md) | DAgger算法详细实现指南 | ⭐⭐⭐ 🆕 |
| [DAGGER_VS_BC_COMPARISON.md](guides/DAGGER_VS_BC_COMPARISON.md) | DAgger vs BC 可视化对比 | ⭐⭐⭐ 🆕 |
| [IMITATION_LEARNING_GUIDE.md](guides/IMITATION_LEARNING_GUIDE.md) | 模仿学习完整指南 | ⭐⭐⭐ 🆕 |
| [LABEL_STATES_GUIDE.md](guides/LABEL_STATES_GUIDE.md) | 标注工具使用指南 | ⭐⭐ 🆕 |
| **录制和验证** | | |
| [RECORDING_CONTROLS.md](guides/RECORDING_CONTROLS.md) | 手动录制控制说明 | 🟢 新手 |
| [TEST_RECORDING.md](guides/TEST_RECORDING.md) | 录制测试指南 | 🟢 新手 |
| **MineCLIP** | | |
| [MINECLIP_CURRICULUM_LEARNING.md](guides/MINECLIP_CURRICULUM_LEARNING.md) | MineCLIP课程学习详解 | 🟡 中级 |
| [MINECLIP_SETUP_GUIDE.md](guides/MINECLIP_SETUP_GUIDE.md) | MineCLIP设置指南 | 🔴 高级 |
| **训练指南** | | |
| [CHECKPOINT_RESUME_GUIDE.md](guides/CHECKPOINT_RESUME_GUIDE.md) | 检查点恢复机制说明 | 🟡 中级 |
| [TRAINING_ACCELERATION_GUIDE.md](guides/TRAINING_ACCELERATION_GUIDE.md) | 训练加速完整指南 | 🟡 中级 |
| [GET_WOOD_TRAINING_GUIDE.md](guides/GET_WOOD_TRAINING_GUIDE.md) | 砍树任务训练指南 | 🟡 中级 |
| [TASKS_QUICK_START.md](guides/TASKS_QUICK_START.md) | 多任务快速开始 | 🟡 中级 |
| [QUICK_START_ACCELERATED_TRAINING.md](guides/QUICK_START_ACCELERATED_TRAINING.md) | 加速训练快速开始 | 🟡 中级 |
| [TENSORBOARD_中文指南.md](guides/TENSORBOARD_中文指南.md) | TensorBoard使用指南 | 🟢 新手 |

---

### 📚 **reference/** - 技术参考
技术细节和API参考文档

| 文档 | 说明 |
|------|------|
| [MINEDOJO_ACTION_MAPPING.md](reference/MINEDOJO_ACTION_MAPPING.md) | MineDojo动作空间映射详解（含functional=3验证）|
| [MINEDOJO_TASKS_REFERENCE.md](reference/MINEDOJO_TASKS_REFERENCE.md) | MineDojo所有任务参考 |

---

### 🔬 **technical/** - 深度技术解析
核心算法和设计原理的深入解释

| 文档 | 说明 | 难度 |
|------|------|------|
| [DAGGER_CNN_ARCHITECTURE.md](technical/DAGGER_CNN_ARCHITECTURE.md) | DAgger的CNN架构详解 (NatureCNN) | 🟡 中级 🆕 |
| [RGB_VS_GRAYSCALE_ANALYSIS.md](technical/RGB_VS_GRAYSCALE_ANALYSIS.md) | RGB vs 灰度图像训练分析 | 🟡 中级 🆕 |
| [MINECLIP_REWARD_DESIGN_EXPLAINED.md](technical/MINECLIP_REWARD_DESIGN_EXPLAINED.md) | MineCLIP差值奖励设计详解 | 🔴 高级 |

---

### 🐛 **issues/** - 问题记录
已知问题的诊断和解决方案

| 文档 | 问题 | 状态 |
|------|------|------|
| [EPISODE_TRACKING_FIX.md](issues/EPISODE_TRACKING_FIX.md) | 回合数追踪错误 | ✅ 已解决 |
| [CAMERA_CONTROL_ISSUE.md](issues/CAMERA_CONTROL_ISSUE.md) | 相机控制问题 | ✅ 已解决 |

---

### 📝 **summaries/** - 训练总结
历史训练记录和经验总结

| 文档 | 说明 |
|------|------|
| [TRAINING_HARVEST_PAPER.md](summaries/TRAINING_HARVEST_PAPER.md) | harvest_paper任务训练总结 |
| [ACCELERATED_TRAINING_SUMMARY.md](summaries/ACCELERATED_TRAINING_SUMMARY.md) | 加速训练总结 |

---

## 🚀 快速导航

### 我是新手，从哪里开始？
1. 📖 阅读 [`guides/DAGGER_QUICK_START.md`](guides/DAGGER_QUICK_START.md) ⭐ **最推荐！**
2. 🎓 学习 [`guides/DAGGER_LABELING_STRATEGY.md`](guides/DAGGER_LABELING_STRATEGY.md) 🔥 **必读！**
3. 🎮 学习 [`guides/RECORDING_CONTROLS.md`](guides/RECORDING_CONTROLS.md)
4. 🏃 尝试 [`guides/TEST_RECORDING.md`](guides/TEST_RECORDING.md)

### 我想了解DAgger
1. 🎯 快速上手: [`guides/DAGGER_QUICK_START.md`](guides/DAGGER_QUICK_START.md) ⭐
2. 🎓 **标注策略**: [`guides/DAGGER_LABELING_STRATEGY.md`](guides/DAGGER_LABELING_STRATEGY.md) 🔥 **必读！**
3. 📊 可视化对比: [`guides/DAGGER_VS_BC_COMPARISON.md`](guides/DAGGER_VS_BC_COMPARISON.md)
4. 📚 详细理论: [`guides/DAGGER_DETAILED_GUIDE.md`](guides/DAGGER_DETAILED_GUIDE.md)
5. 🛠️ 标注工具: [`guides/LABEL_STATES_GUIDE.md`](guides/LABEL_STATES_GUIDE.md)
6. 📋 实施计划: [`status/DAGGER_IMPLEMENTATION_PLAN.md`](status/DAGGER_IMPLEMENTATION_PLAN.md)

### 我想了解当前项目进展
1. 📊 查看 [`status/DAGGER_IMPLEMENTATION_PLAN.md`](status/DAGGER_IMPLEMENTATION_PLAN.md) ⭐ **最新状态**
2. 🔧 了解 [`status/MINECLIP_STATUS_SUMMARY.md`](status/MINECLIP_STATUS_SUMMARY.md)

### 我遇到了问题
1. 🐛 检查 [`issues/`](issues/) 目录看是否有相同问题
2. 📚 查阅 [`reference/MINEDOJO_ACTION_MAPPING.md`](reference/MINEDOJO_ACTION_MAPPING.md) 了解动作映射
3. 💬 如果是新问题，在项目issue中提出

### 我想优化训练效果
1. 🎯 **首选**: 使用DAgger - [`guides/DAGGER_QUICK_START.md`](guides/DAGGER_QUICK_START.md) ⭐
2. 📈 阅读 [`guides/TRAINING_ACCELERATION_GUIDE.md`](guides/TRAINING_ACCELERATION_GUIDE.md)
3. 🎓 学习 [`guides/MINECLIP_CURRICULUM_LEARNING.md`](guides/MINECLIP_CURRICULUM_LEARNING.md)
4. 💾 了解 [`guides/CHECKPOINT_RESUME_GUIDE.md`](guides/CHECKPOINT_RESUME_GUIDE.md)

---

## 📋 文档更新日志

### 2025-10-22 (Latest)
- 🔥 **新增DAgger标注策略指南**:
  - 添加 `DAGGER_LABELING_STRATEGY.md` - 避免原地转圈的标注策略 🔥 **必读！**
  - 详细解释"环视是短期行为，移动是主要策略"
  - 包含4个具体场景示例和质量自检方法
- 🎨 **新增RGB vs 灰度分析**:
  - 添加 `RGB_VS_GRAYSCALE_ANALYSIS.md` - 输入表示对比分析
  - Minecraft场景颜色重要性评估
  - harvest_1_log任务不建议灰度（预测损失15-25%）
  - MineCLIP兼容性分析
- 🧠 **新增DAgger CNN架构详解**:
  - 添加 `DAGGER_CNN_ARCHITECTURE.md` - 深入解析NatureCNN
  - 3层卷积+1层全连接，14.7M参数
  - 为什么CNN对Minecraft重要（空间不变性、层级特征）
  - 与MLP/ResNet对比，优化建议

### 2025-10-21
- ✅ **新增DAgger完整实现**:
  - 添加 `DAGGER_QUICK_START.md` - 快速开始指南 ⭐
  - 添加 `DAGGER_DETAILED_GUIDE.md` - 详细实现指南
  - 添加 `DAGGER_VS_BC_COMPARISON.md` - 可视化对比
  - 添加 `DAGGER_IMPLEMENTATION_PLAN.md` - 实施计划
- ✅ 添加 `MINECLIP_REWARD_DESIGN_EXPLAINED.md` - 奖励设计详解
- ✅ 添加 `MINECLIP_PROMPT_OPTIMIZATION_RESULTS.md` - 提示词优化结果
- ✅ 完成 `IMITATION_LEARNING_GUIDE.md` 和 `IMITATION_LEARNING_ROADMAP.md`
- ✅ 重组docs目录结构，按功能分类

### 之前
- 创建基础文档结构
- 添加训练指南和问题记录
- 添加MineCLIP相关文档

---

## 🤝 贡献文档

如果你想补充或改进文档：

1. 确定文档分类（status/guides/reference/issues/summaries/technical）
2. 遵循Markdown格式和命名规范
3. 在相应目录的README中添加链接
4. 提交PR并说明改进内容

---

**最后更新**: 2025-10-21  
**维护者**: AIMC Team  
**当前状态**: 🚀 DAgger实现完成，工具就绪，可开始训练！

