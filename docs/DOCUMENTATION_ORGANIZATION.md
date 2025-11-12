# 文档组织结构说明

> **更新日期**: 2025-10-25  
> **维护者**: AIMC项目组

---

## 📁 目录结构

```
docs/
├── design/           # 设计文档 (1个)
│   └── UNIVERSAL_MINECLIP_STRATEGY.md
│
├── guides/           # 使用指南 (8个)
│   ├── CONFIG_YAML_SUPPORT.md
│   ├── DAGGER_COMPREHENSIVE_GUIDE.md
│   ├── DAGGER_WORKFLOW_SKIP_GUIDE.md
│   ├── GET_WOOD_CONFIG_GUIDE.md
│   ├── LABEL_STATES_SHORTCUTS_GUIDE.md
│   ├── MINECLIP_COMPREHENSIVE_GUIDE.md
│   ├── TASK_WRAPPERS_GUIDE.md
│   ├── TENSORBOARD_中文指南.md
│   └── WEB_COMPREHENSIVE_GUIDE.md
│
├── reference/        # 参考文档 (4个)
│   ├── KEYBOARD_REFERENCE.md
│   ├── MINECLIP_REWARD_REFERENCE.md
│   ├── MINEDOJO_ACTION_REFERENCE.md
│   └── MINEDOJO_TASKS_REFERENCE.md
│
├── technical/        # 技术文档 (1个)
│   └── DAGGER_CNN_ARCHITECTURE.md
│
├── summaries/        # 总结记录 (9个)
│   ├── ACCELERATED_TRAINING_SUMMARY.md
│   ├── ARCHITECTURE_REFACTORING_SUMMARY.md
│   ├── EVALUATION_AND_CONFIG_SUMMARY.md
│   ├── MINECLIP_16FRAMES_GUIDE.md
│   ├── README_16FRAMES.md
│   ├── REFACTORING_SUMMARY.md
│   ├── TRAINING_HARVEST_PAPER.md
│   ├── WEB_CONFIG_MANAGEMENT_IMPROVEMENTS.md
│   ├── WEB_IMPROVEMENTS_SUMMARY.md
│   └── WEB_RESTRUCTURE.md
│
├── issues/           # 问题修复 (7个)
│   ├── DAGGER_COMBINED_DATA_FIX.md
│   ├── DAGGER_DEVICE_PARAMETER_FIX.md
│   ├── DATA_NORMALIZATION_FIX_SUMMARY.md
│   ├── DATA_NORMALIZATION_INVESTIGATION.md
│   ├── INVENTORY_FORMAT_FIX_PLAN.md
│   ├── STEVE1_DTYPE_MISMATCH_FIX.md
│   └── STEVE1_4090_DTYPE_FIX_QUICKSTART.md
│
└── status/           # 状态追踪 (2个)
    ├── IMITATION_LEARNING_ROADMAP.md
    └── MINECLIP_STATUS_SUMMARY.md
```

**总计**: 32个文档，分类清晰，结构合理

---

## 📖 文档分类说明

### 1. design/ - 设计文档
**用途**: 架构设计、方案设计、策略设计

**包含文档**:
- `UNIVERSAL_MINECLIP_STRATEGY.md` - MineCLIP通用训练策略

**何时查阅**: 
- 了解系统架构设计思路
- 规划新功能或重构
- 学习设计决策背后的原因

---

### 2. guides/ - 使用指南
**用途**: 面向用户的操作手册和配置指南

**核心文档**:
- `WEB_COMPREHENSIVE_GUIDE.md` - **Web控制台完整指南** ⭐
- `DAGGER_COMPREHENSIVE_GUIDE.md` - **DAgger算法完整指南** ⭐
- `DAGGER_WORKFLOW_SKIP_GUIDE.md` - DAgger快捷操作
- `CONFIG_YAML_SUPPORT.md` - 配置文件使用
- `MINECLIP_COMPREHENSIVE_GUIDE.md` - MineCLIP奖励函数

**何时查阅**:
- 初次使用系统
- 学习具体功能的操作方法
- 配置任务参数
- 快速上手DAgger训练

---

### 3. reference/ - 参考文档
**用途**: 快速查询的API参考和参数说明

**包含文档**:
- `KEYBOARD_REFERENCE.md` - 键盘控制速查
- `MINECLIP_REWARD_REFERENCE.md` - MineCLIP奖励API
- `MINEDOJO_ACTION_REFERENCE.md` - MineDojo动作空间
- `MINEDOJO_TASKS_REFERENCE.md` - MineDojo任务列表

**何时查阅**:
- 查询具体参数或API
- 了解可用的动作和任务
- 快速查找配置选项

---

### 4. technical/ - 技术文档
**用途**: 深入的技术实现细节和算法原理

**包含文档**:
- `DAGGER_CNN_ARCHITECTURE.md` - DAgger CNN架构详解

**何时查阅**:
- 理解模型架构
- 优化训练性能
- 修改或扩展算法
- 深入学习技术细节

---

### 5. summaries/ - 总结记录
**用途**: 项目改进、重构、功能迭代的历史记录

**重要文档**:
- `ARCHITECTURE_REFACTORING_SUMMARY.md` - **架构重构总结** ⭐
- `WEB_IMPROVEMENTS_SUMMARY.md` - Web功能改进
- `WEB_RESTRUCTURE.md` - Web架构重构
- `REFACTORING_SUMMARY.md` - 代码重构记录

**何时查阅**:
- 了解项目演化历史
- 学习重构经验
- 追溯功能改进原因
- 版本变更记录

---

### 6. issues/ - 问题修复
**用途**: Bug修复记录和问题分析

**包含文档**:
- `DATA_NORMALIZATION_FIX_SUMMARY.md` - 数据归一化修复
- `DAGGER_COMBINED_DATA_FIX.md` - DAgger数据聚合修复
- `DAGGER_DEVICE_PARAMETER_FIX.md` - 设备参数修复
- `DATA_NORMALIZATION_INVESTIGATION.md` - 数据问题调查
- `INVENTORY_FORMAT_FIX_PLAN.md` - 库存格式修复计划
- `STEVE1_DTYPE_MISMATCH_FIX.md` - **STEVE-1 Dtype不匹配修复** ⭐
- `STEVE1_4090_DTYPE_FIX_QUICKSTART.md` - **4090 GPU快速修复指南** ⭐

**何时查阅**:
- 遇到类似问题
- 学习问题分析方法
- 了解已知问题和解决方案

---

### 7. status/ - 状态追踪
**用途**: 项目路线图、功能状态、进度报告

**包含文档**:
- `IMITATION_LEARNING_ROADMAP.md` - 模仿学习路线图
- `MINECLIP_STATUS_SUMMARY.md` - MineCLIP功能状态

**何时查阅**:
- 了解项目发展方向
- 查看功能完成状态
- 规划下一步工作

---

## 🎯 快速导航

### 新手入门推荐阅读顺序

1. **第一步**: `guides/WEB_COMPREHENSIVE_GUIDE.md`
   - 学习如何使用Web控制台

2. **第二步**: `guides/DAGGER_COMPREHENSIVE_GUIDE.md`
   - 理解DAgger算法和训练流程

3. **第三步**: `guides/CONFIG_YAML_SUPPORT.md`
   - 掌握配置文件的使用

4. **进阶**: `technical/DAGGER_CNN_ARCHITECTURE.md`
   - 深入了解模型架构

### 常见任务快速查找

| 任务 | 文档位置 |
|-----|---------|
| 使用Web控制台 | `guides/WEB_COMPREHENSIVE_GUIDE.md` |
| DAgger训练流程 | `guides/DAGGER_COMPREHENSIVE_GUIDE.md` |
| 配置任务参数 | `guides/CONFIG_YAML_SUPPORT.md` |
| 键盘控制说明 | `reference/KEYBOARD_REFERENCE.md` |
| 模型架构详解 | `technical/DAGGER_CNN_ARCHITECTURE.md` |
| 架构重构历史 | `summaries/ARCHITECTURE_REFACTORING_SUMMARY.md` |
| 4090 GPU修复 | `issues/STEVE1_4090_DTYPE_FIX_QUICKSTART.md` ⭐ |
| 问题修复记录 | `issues/` 目录 |
| 项目路线图 | `status/IMITATION_LEARNING_ROADMAP.md` |

---

## 📝 文档维护规范

### 创建新文档的规则

根据 `.cursorrules` 中定义的规则：

1. **design/** - 设计阶段文档
   - 命名：`*_STRATEGY.md`, `*_DESIGN.md`, `*_ARCHITECTURE.md`
   - 内容：系统设计、方案选型、架构规划

2. **guides/** - 用户操作指南
   - 命名：`*_GUIDE.md`, `*_COMPREHENSIVE_GUIDE.md`
   - 内容：使用教程、配置说明、快速开始

3. **reference/** - 快速参考
   - 命名：`*_REFERENCE.md`
   - 内容：API列表、参数说明、命令速查表

4. **technical/** - 技术实现
   - 命名：`*_ARCHITECTURE.md`, `*_IMPLEMENTATION.md`
   - 内容：算法原理、代码实现、性能优化

5. **summaries/** - 改进总结
   - 命名：`*_SUMMARY.md`, `*_REFACTORING_SUMMARY.md`
   - 内容：重构记录、功能改进、版本变更

6. **issues/** - 问题修复
   - 命名：`*_FIX.md`, `*_INVESTIGATION.md`
   - 内容：bug修复、问题分析、解决方案

7. **status/** - 状态追踪
   - 命名：`*_ROADMAP.md`, `*_STATUS.md`
   - 内容：路线图、进度报告、功能状态

### 文档创建原则

- ✅ 新增功能 → `guides/`（使用指南）+ `summaries/`（实现总结）
- ✅ 架构变更 → `design/`（设计文档）+ `summaries/`（变更总结）
- ✅ Bug修复 → `issues/`（修复记录）
- ✅ 状态更新 → `status/`（进度报告）
- ❌ **避免在docs根目录直接创建文档**
- ❌ **同类文档优先合并，避免重复**

### 文档命名规范

- 使用大写下划线命名：`FEATURE_NAME_TYPE.md`
- 类型后缀：`_GUIDE`, `_SUMMARY`, `_REFERENCE`, `_ARCHITECTURE`等
- 综合性文档使用：`*_COMPREHENSIVE_GUIDE.md`

---

## 🔄 最近的文档整理

### 2025-10-25 文档重组

**合并的文档**:

1. **Web相关** → `guides/WEB_COMPREHENSIVE_GUIDE.md`
   - 合并了：WEB_CONSOLE_GUIDE, WEB_TASK_MANAGEMENT_GUIDE, WEB_EVALUATE_STOP_FEATURE, WEB_TASK_CONFIG_MANAGEMENT

2. **架构重构** → `summaries/ARCHITECTURE_REFACTORING_SUMMARY.md`
   - 合并了：MIGRATION_SUMMARY, NEW_DIRECTORY_STRUCTURE, SCRIPT_SPLIT_IMPLEMENTATION, SCRIPT_SPLIT_SUMMARY, UNIFY_SCRIPT_CALLS

**移动的文档**:
- 所有docs根目录的文档已正确分类到子目录
- 配置相关文档移至guides/
- 总结类文档移至summaries/

**效果**:
- ✅ 文档数量从 35+ 减少到 30
- ✅ 消除了重复内容
- ✅ 结构更加清晰
- ✅ 易于查找和维护

---

## 🎓 贡献指南

### 添加新文档

1. 确定文档类型（设计/指南/参考/技术/总结/问题/状态）
2. 按照命名规范创建文档
3. 放置到对应的子目录
4. 在本文档中更新索引

### 更新现有文档

1. 保持文档的及时性和准确性
2. 重大更新需记录版本和日期
3. 涉及多个文档的更新需保持一致性

### 文档审查

定期审查文档：
- 检查是否有过时内容
- 合并重复或相似文档
- 更新链接和引用
- 改进文档结构

---

## 📚 外部资源

- **MineDojo官方文档**: https://minedojo.org
- **MineCLIP论文**: https://arxiv.org/abs/2206.08853
- **Stable-Baselines3文档**: https://stable-baselines3.readthedocs.io

---

**维护建议**: 
- 每月审查一次文档结构
- 及时合并重复内容
- 保持文档分类清晰
- 更新快速导航索引

**反馈**: 如有文档组织建议，请在项目仓库创建Issue

