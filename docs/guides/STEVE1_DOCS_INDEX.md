# STEVE-1 文档导航

> 📚 完整的STEVE-1文档索引和学习路径
> 
> 最后更新: 2025-11-05

---

## 📖 文档结构总览

```
docs/
├── guides/          # 用户指南 (入门、使用、实战)
│   ├── 入门级      ⭐ 快速了解
│   ├── 进阶级      ⭐⭐ 深入学习
│   └── 实战级      ⭐⭐⭐ 动手实践
│
├── technical/       # 技术文档 (原理、实现、深入)
│   ├── 原理分析
│   ├── 实现细节
│   └── 进阶话题
│
├── reference/       # 参考文档 (速查、对比)
│
├── issues/          # 问题修复记录
│
└── summaries/       # 项目总结记录
```

---

## 🎯 核心文档 (按学习路径排序)

### 🌟 第一步：快速了解 (5-15分钟)

**1. STEVE1_QUICK_REFERENCE.md** ⭐ 必读
- 📄 位置: `docs/guides/`
- 📊 大小: 7.5KB
- 🎯 适合: 第一次接触STEVE-1
- 📝 内容:
  - 一页纸速查表
  - 核心概念和公式
  - 5分钟快速了解
  - 常见术语解释

**2. STEVE1_DOCS_INDEX.md** ⭐ 导航
- 📄 位置: `docs/guides/` (当前文档)
- 🎯 用途: 文档导航和学习路径

---

### 📚 第二步：理解原理 (30-60分钟)

**3. STEVE1_TRAINING_EXPLAINED.md** ⭐⭐ 核心必读
- 📄 位置: `docs/guides/`
- 📊 大小: 13KB
- 🎯 适合: 想深入理解训练原理
- 📝 内容:
  - 完整训练流程
  - 事后重标记详解
  - MineCLIP作用机制
  - 数据流程图解
  - BC vs RL对比
  - 为什么用未来帧作为目标
  - 常见误解澄清

**4. STEVE1_VPT_DATA_RELATIONSHIP.md** ⭐⭐ 数据关系
- 📄 位置: `docs/technical/`
- 📊 大小: 35KB
- 🎯 适合: 理解数据集和VPT关系
- 📝 内容:
  - Contractor数据在STEVE-1中的作用
  - VPT-Generated数据用途
  - Text-Video Pair数据集详解
  - 三个数据集的完整关系
  - 为什么需要独立评估集
  - 数据量对比和配比建议

**5. STEVE1_VPT_WEIGHTS_AND_CFG.md** ⭐⭐ 技术细节
- 📄 位置: `docs/technical/`
- 📊 大小: 19KB
- 🎯 适合: 了解VPT权重选择和CFG实现
- 📝 内容:
  - STEVE-1使用的VPT权重分支
  - 为什么选择rl-from-foundation
  - Classifier-Free Guidance原理
  - CFG实现位置和代码
  - CFG超参数调优

**6. SEQUENTIAL_POLICY_EVALUATION.md** ⭐⭐ 评估方法
- 📄 位置: `docs/technical/`
- 📊 大小: 29KB
- 🎯 适合: 理解序贯决策评估
- 📝 内容:
  - 为什么不能逐帧对比动作
  - Closed-loop vs Open-loop评估
  - 评估指标设计
  - 如何处理轨迹diverge
  - 专家数据在评估中的作用

---

### 🛠️ 第三步：动手实践 (1-2小时)

**7. STEVE1_SCRIPTS_USAGE_GUIDE.md** ⭐⭐⭐ 脚本使用
- 📄 位置: `docs/guides/`
- 📊 大小: 19KB
- 🎯 适合: 运行STEVE-1脚本
- 📝 内容:
  - 所有脚本详细说明
  - 参数配置
  - 使用示例
  - 常见错误处理

**8. STEVE1_EVALUATION_GUIDE.md** ⭐⭐⭐ 评估实战
- 📄 位置: `docs/guides/`
- 📊 大小: 17KB
- 🎯 适合: 评估模型性能
- 📝 内容:
  - 评估方法详解
  - MineDojo任务测试
  - 成功率计算
  - 对比分析

---

### 🎓 第四步：微调优化 (数小时-数天)

**9. STEVE1_FINETUNING_QUICKSTART.md** ⭐⭐⭐ 微调快速开始
- 📄 位置: `docs/guides/`
- 📊 大小: 15KB
- 🎯 适合: 快速开始微调
- 📝 内容:
  - 3步快速微调流程
  - 数据准备方法
  - 配置模板
  - 常见问题

**10. STEVE1_FINETUNING_EXPLAINED.md** ⭐⭐⭐ 微调详解
- 📄 位置: `docs/guides/`
- 📊 大小: 21KB
- 🎯 适合: 深入理解微调策略
- 📝 内容:
  - 是否需要微调的判断
  - 微调方法详解
  - 超参数选择
  - 高级微调策略
  - 实战案例分析

---

### 🚀 第五步：进阶话题 (按需阅读)

**11. STEVE1_ADVANCED_SOLUTIONS.md** ⭐⭐⭐ 进阶解决方案
- 📄 位置: `docs/guides/`
- 📊 大小: 14KB
- 🎯 适合: 解决特殊需求
- 📝 内容:
  - 如何支持中文语义
  - 从网络视频生成训练数据
  - IDM逆向动力学模型
  - 多语言MineCLIP方案
  - 视频动作预测方法

**12. STEVE1_TRAINING_ANALYSIS.md** ⭐⭐ 训练技术分析
- 📄 位置: `docs/technical/`
- 📊 大小: 23KB
- 🎯 适合: 深入技术细节
- 📝 内容:
  - 训练代码实现分析
  - 性能优化技巧
  - 高级技术话题

---

## 🗂️ 辅助文档（参考资料和历史记录）

### 参考资料 (reference/)

**BC_VS_RL_REFERENCE.md** ⭐⭐ 详细对比
- 📄 位置: `docs/reference/`
- 📊 大小: 15KB
- 🎯 用途: BC和RL的详细对比参考
- 📝 内容:
  - Behavior Cloning详解
  - Reinforcement Learning详解
  - 两者的核心区别
  - 实现细节对比
  - 适用场景分析

### 问题修复记录 (issues/)

**STEVE1_OFFLINE_SETUP.md**
- 📄 位置: `docs/issues/`
- 📊 大小: 7.0KB
- 🎯 用途: 离线环境配置问题修复
- 📝 内容: STEVE-1离线部署的解决方案

### 项目历史总结 (summaries/)

这些文档记录了项目的演进历史和文档创建过程：

**STEVE1_INTEGRATION_FIXES.md**
- 📄 位置: `docs/summaries/`
- 📊 大小: 7.9KB
- 🎯 用途: STEVE-1集成修复的历史记录
- 📝 内容: 集成过程中的问题和修复方案

**STEVE1_TRAINING_FINETUNING_SUMMARY.md**
- 📄 位置: `docs/summaries/`
- 📊 大小: 11KB
- 🎯 用途: 2025-10-31文档创建工作总结
- 📝 内容: 当时创建的文档列表和改进点

💡 **注意**: summaries/ 下的文档是历史记录，主要用于追溯项目演进，日常使用请参考guides/和technical/目录下的最新文档。

---

## 🎓 推荐学习路径

### 路径1: 新手入门 (1-2小时)

```
1. STEVE1_QUICK_REFERENCE.md (5分钟)
   快速了解核心概念
   ↓
2. STEVE1_TRAINING_EXPLAINED.md (30分钟)
   理解训练原理
   ↓
3. STEVE1_SCRIPTS_USAGE_GUIDE.md (30分钟)
   学习脚本使用
   ↓
4. 运行测试脚本
   bash src/training/steve1/2_gen_vid_for_text_prompt.sh
```

### 路径2: 深入研究 (1-2天)

```
1. STEVE1_TRAINING_EXPLAINED.md
   核心原理
   ↓
2. STEVE1_VPT_DATA_RELATIONSHIP.md
   数据关系
   ↓
3. STEVE1_VPT_WEIGHTS_AND_CFG.md
   技术细节
   ↓
4. SEQUENTIAL_POLICY_EVALUATION.md
   评估方法
   ↓
5. STEVE1_TRAINING_ANALYSIS.md
   实现分析
   ↓
6. 阅读源代码
   src/training/steve1/
```

### 路径3: 微调实战 (数小时-数天)

```
1. STEVE1_TRAINING_EXPLAINED.md
   理解原理
   ↓
2. STEVE1_FINETUNING_QUICKSTART.md
   快速开始
   ↓
3. 准备数据和微调
   bash src/training/steve1/3_train_finetune_template.sh
   ↓
4. STEVE1_EVALUATION_GUIDE.md
   评估效果
   ↓
5. STEVE1_FINETUNING_EXPLAINED.md (如有问题)
   深入优化策略
```

### 路径4: 进阶应用 (按需)

```
1. STEVE1_ADVANCED_SOLUTIONS.md
   进阶解决方案
   ↓
2. 实现自定义功能
   - 中文支持
   - 视频数据处理
   - 自定义任务
```

---

## 🔍 快速查找

### 想了解...

#### 基础概念
- **"STEVE-1是什么？"**
  → `STEVE1_QUICK_REFERENCE.md`

- **"MineCLIP如何工作？"**
  → `STEVE1_TRAINING_EXPLAINED.md` 第3节

- **"事后重标记是什么？"**
  → `STEVE1_TRAINING_EXPLAINED.md` 第2节

- **"为什么用未来帧不用当前帧？"**
  → `STEVE1_TRAINING_EXPLAINED.md` 第4节

#### 数据相关
- **"Contractor数据是什么？"**
  → `STEVE1_VPT_DATA_RELATIONSHIP.md` 第1节

- **"VPT-Generated数据用来做什么？"**
  → `STEVE1_VPT_DATA_RELATIONSHIP.md` 第2节

- **"为什么需要Text-Video Pair数据？"**
  → `STEVE1_VPT_DATA_RELATIONSHIP.md` 第7-8节

- **"如何准备训练数据？"**
  → `STEVE1_FINETUNING_QUICKSTART.md` 步骤1
  → `STEVE1_TRAINING_EXPLAINED.md` 第7.2节

#### 技术细节
- **"STEVE-1用哪个VPT权重？"**
  → `STEVE1_VPT_WEIGHTS_AND_CFG.md` 第1节

- **"Classifier-Free Guidance是什么？"**
  → `STEVE1_VPT_WEIGHTS_AND_CFG.md` 第2节

- **"如何评估序贯策略？"**
  → `SEQUENTIAL_POLICY_EVALUATION.md` 第2-3节

- **"为什么不能逐帧对比动作？"**
  → `SEQUENTIAL_POLICY_EVALUATION.md` 第3节

#### 实践操作
- **"如何运行训练脚本？"**
  → `STEVE1_SCRIPTS_USAGE_GUIDE.md`

- **"如何微调模型？"**
  → `STEVE1_FINETUNING_QUICKSTART.md` (快速)
  → `STEVE1_FINETUNING_EXPLAINED.md` (详细)

- **"如何评估性能？"**
  → `STEVE1_EVALUATION_GUIDE.md`

#### 进阶话题
- **"如何支持中文？"**
  → `STEVE1_ADVANCED_SOLUTIONS.md` 问题1

- **"如何从网络视频生成数据？"**
  → `STEVE1_ADVANCED_SOLUTIONS.md` 问题2

- **"BC和RL有什么区别？"**
  → `STEVE1_TRAINING_EXPLAINED.md` 第6节
  → `../reference/BC_VS_RL_REFERENCE.md`

---

## 📊 文档统计

### 完整清单

**总计：17个STEVE-1相关文档，约233KB**

```
核心文档 (12个，约176KB):
  guides/        8个  (用户指南)
  technical/     4个  (技术深入)

辅助文档 (5个，约57KB):
  reference/     1个  (参考资料)
  issues/        1个  (问题修复)
  summaries/     2个  (历史记录)
  根目录/        1个  (整理总结)
```

### 按类型分类

```
guides/ (用户指南): 8个文档
  ├─ 入门: 2个 (QUICK_REFERENCE, DOCS_INDEX)
  ├─ 原理: 1个 (TRAINING_EXPLAINED)
  ├─ 实战: 3个 (SCRIPTS, EVALUATION, FINETUNING_QUICKSTART)
  └─ 深入: 2个 (FINETUNING_EXPLAINED, ADVANCED_SOLUTIONS)

technical/ (技术文档): 4个文档
  ├─ 训练: 2个 (TRAINING_ANALYSIS, VPT_DATA_RELATIONSHIP)
  └─ 技术: 2个 (VPT_WEIGHTS_AND_CFG, SEQUENTIAL_POLICY_EVALUATION)

reference/ (参考资料): 1个文档
  └─ BC_VS_RL_REFERENCE (详细对比)

issues/ (问题修复): 1个文档
  └─ STEVE1_OFFLINE_SETUP (离线配置)

summaries/ (历史记录): 2个文档
  ├─ STEVE1_INTEGRATION_FIXES (集成修复历史)
  └─ STEVE1_TRAINING_FINETUNING_SUMMARY (2025-10-31总结)

根目录: 1个文档
  └─ STEVE1_DOCUMENTATION_SUMMARY (本次整理总结)
```

### 按难度分级

```
⭐ 入门级 (2个):
  - STEVE1_QUICK_REFERENCE.md
  - STEVE1_DOCS_INDEX.md

⭐⭐ 进阶级 (6个):
  - STEVE1_TRAINING_EXPLAINED.md
  - STEVE1_VPT_DATA_RELATIONSHIP.md
  - STEVE1_VPT_WEIGHTS_AND_CFG.md
  - SEQUENTIAL_POLICY_EVALUATION.md
  - STEVE1_TRAINING_ANALYSIS.md
  - STEVE1_EVALUATION_GUIDE.md

⭐⭐⭐ 实战级 (4个):
  - STEVE1_SCRIPTS_USAGE_GUIDE.md
  - STEVE1_FINETUNING_QUICKSTART.md
  - STEVE1_FINETUNING_EXPLAINED.md
  - STEVE1_ADVANCED_SOLUTIONS.md
```

### 按主题分类

```
训练原理 (4个):
  - STEVE1_TRAINING_EXPLAINED.md
  - STEVE1_TRAINING_ANALYSIS.md
  - STEVE1_VPT_DATA_RELATIONSHIP.md
  - STEVE1_VPT_WEIGHTS_AND_CFG.md

实践操作 (3个):
  - STEVE1_SCRIPTS_USAGE_GUIDE.md
  - STEVE1_EVALUATION_GUIDE.md
  - STEVE1_FINETUNING_QUICKSTART.md

微调优化 (2个):
  - STEVE1_FINETUNING_QUICKSTART.md
  - STEVE1_FINETUNING_EXPLAINED.md

进阶话题 (3个):
  - STEVE1_ADVANCED_SOLUTIONS.md
  - SEQUENTIAL_POLICY_EVALUATION.md
  - STEVE1_VPT_WEIGHTS_AND_CFG.md

快速参考 (1个):
  - STEVE1_QUICK_REFERENCE.md
```

---

## 🔄 文档更新历史

### 2025-11-05 (本次更新)

**新增文档**:
- ✅ `STEVE1_FINETUNING_EXPLAINED.md` (21KB)
  - 详细的微调指南
  - 实战案例分析
  
- ✅ `STEVE1_VPT_WEIGHTS_AND_CFG.md` (19KB)
  - VPT权重选择说明
  - CFG实现详解
  
- ✅ `STEVE1_VPT_DATA_RELATIONSHIP.md` (扩展到35KB)
  - 新增第7-8节：完整数据集构成
  - 新增评估数据集详解
  
- ✅ `SEQUENTIAL_POLICY_EVALUATION.md` (29KB)
  - 序贯策略评估方法
  - 解答轨迹diverge问题

**更新文档**:
- ✅ `STEVE1_DOCS_INDEX.md`
  - 更新文档列表
  - 优化学习路径
  - 添加快速查找索引

### 2025-11-05 之前

**第一批整理** (2025-11-05 早期):
- ✅ 合并5个训练相关文档 → `STEVE1_TRAINING_EXPLAINED.md`
- ✅ 删除重复文档 (BC_VS_RL_EXPLAINED等)
- ✅ 创建文档索引

**原始文档** (项目初期):
- STEVE1_EVALUATION_GUIDE.md
- STEVE1_FINETUNING_QUICKSTART.md
- STEVE1_SCRIPTS_USAGE_GUIDE.md
- STEVE1_TRAINING_ANALYSIS.md

---

## 💡 使用建议

### 📱 移动设备查看
所有文档都使用Markdown格式，在手机/平板上也能良好阅读。

### 🔖 收藏推荐
建议收藏以下3个文档：
1. `STEVE1_DOCS_INDEX.md` (导航)
2. `STEVE1_QUICK_REFERENCE.md` (速查)
3. `STEVE1_TRAINING_EXPLAINED.md` (原理)

### 🖨️ 打印友好
所有文档都可以直接打印或导出为PDF。

### 🔗 内部链接
文档之间有相互引用，方便跳转阅读。

---

## 📞 反馈与贡献

如果发现文档有误或需要补充，请：
1. 在项目issue中提出
2. 或直接修改文档并提交PR

---

**文档维护者**: AI Assistant  
**项目仓库**: `/Users/nanzhang/aimc`  
**最后更新**: 2025-11-05

---

## 🎯 快速开始

如果你是第一次接触STEVE-1：

```bash
# 1. 快速了解 (5分钟)
cat docs/guides/STEVE1_QUICK_REFERENCE.md

# 2. 理解原理 (30分钟)
cat docs/guides/STEVE1_TRAINING_EXPLAINED.md

# 3. 运行测试
cd src/training/steve1
bash 2_gen_vid_for_text_prompt.sh

# 4. 查看更多文档
cat docs/guides/STEVE1_DOCS_INDEX.md  # 本文档
```

祝你使用愉快！🚀
