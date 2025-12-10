# AIMC 文档索引

本文档提供项目文档的完整索引和分类说明。

## 📚 文档组织

```
docs/
├── guides/          # 使用指南（核心文档）
├── design/          # 设计文档
├── reference/       # 参考文档
├── technical/       # 技术文档
├── status/          # 项目状态
├── summaries/       # 实现总结（历史记录）
├── issues/          # 问题修复记录（历史记录）
└── archive/         # 归档文档（已停用功能）
    ├── dagger/      # DAgger 训练相关
    ├── vpt/         # VPT 独立评估相关
    ├── web/         # Web 控制台相关
    └── inventory/   # Inventory 系统相关
```

---

## ✅ 核心文档（当前有效）

### 部署与安装

| 文档 | 说明 |
|------|------|
| [DEPLOYMENT.md](guides/DEPLOYMENT.md) | **部署指南** - Docker/Linux/macOS 安装 |
| [MINERL_GUIDE.md](guides/MINERL_GUIDE.md) | MineRL 安装配置 |
| [MINERL_QUICKSTART.md](guides/MINERL_QUICKSTART.md) | MineRL 快速开始 |

### 评估系统

| 文档 | 说明 |
|------|------|
| [EVALUATION_FRAMEWORK_GUIDE.md](guides/EVALUATION_FRAMEWORK_GUIDE.md) | **评估框架指南** - 核心文档 |
| [STEVE1_EVALUATION_GUIDE.md](guides/STEVE1_EVALUATION_GUIDE.md) | STEVE-1 评估指南 |
| [STEVE1_EVALUATION_USAGE.md](guides/STEVE1_EVALUATION_USAGE.md) | 评估使用方法 |
| [DEEP_EVALUATION_METRICS_EXPLAINED.md](guides/DEEP_EVALUATION_METRICS_EXPLAINED.md) | 评估指标详解 |
| [STEVE1_DEEP_EVALUATION_GUIDE.md](guides/STEVE1_DEEP_EVALUATION_GUIDE.md) | 深度评估指南 |

### 任务配置

| 文档 | 说明 |
|------|------|
| [TASK_WRAPPERS_GUIDE.md](guides/TASK_WRAPPERS_GUIDE.md) | **任务配置指南** |
| [DYNAMIC_REWARD_CONFIG_GUIDE.md](guides/DYNAMIC_REWARD_CONFIG_GUIDE.md) | 动态奖励配置 |
| [FAILED_TASKS_FIX_GUIDE.md](guides/FAILED_TASKS_FIX_GUIDE.md) | 失败任务修复 |

### 样本录制

| 文档 | 说明 |
|------|------|
| [VISUAL_EMBED_16FRAMES_GUIDE.md](guides/VISUAL_EMBED_16FRAMES_GUIDE.md) | **视觉嵌入生成** |
| [EXTRACT_SUCCESS_VISUALS_QUICKSTART.md](guides/EXTRACT_SUCCESS_VISUALS_QUICKSTART.md) | 成功帧提取 |
| [INSTRUCTION_VIDEO_PAIRS_GUIDE.md](guides/INSTRUCTION_VIDEO_PAIRS_GUIDE.md) | 指令-视频配对 |

### STEVE-1 参考

| 文档 | 说明 |
|------|------|
| [STEVE1_DOCS_INDEX.md](guides/STEVE1_DOCS_INDEX.md) | STEVE-1 文档索引 |
| [STEVE1_QUICK_REFERENCE.md](guides/STEVE1_QUICK_REFERENCE.md) | 快速参考 |
| [STEVE1_ADVANCED_SOLUTIONS.md](guides/STEVE1_ADVANCED_SOLUTIONS.md) | 高级解决方案 |

### 设计文档

| 文档 | 说明 |
|------|------|
| [STEVE1_EVALUATION_FRAMEWORK_DESIGN.md](design/STEVE1_EVALUATION_FRAMEWORK_DESIGN.md) | 评估框架设计 |
| [EVALUATION_TASK_SYSTEM_ANALYSIS.md](design/EVALUATION_TASK_SYSTEM_ANALYSIS.md) | 任务系统分析 |
| [UNIVERSAL_MINECLIP_STRATEGY.md](design/UNIVERSAL_MINECLIP_STRATEGY.md) | MineCLIP 策略 |

---

## 📦 归档文档（历史参考）

以下文档已归档到 `archive/` 目录，记录了项目的开发历史。

**归档说明**: 详见 [archive/README.md](archive/README.md)

### DAgger 训练相关（`archive/dagger/`）

- `DAGGER_COMPREHENSIVE_GUIDE.md` - DAgger 训练指南
- 相关 `summaries/DAGGER_*.md` 文件

### VPT 独立评估（`archive/vpt/`）

- `VPT_ZERO_SHOT_QUICKSTART.md` - VPT 零样本指南
- 相关 `summaries/VPT_*.md` 文件

### Web 控制台（`archive/web/`）

- `WEB_COMPREHENSIVE_GUIDE.md` - Web 控制台指南
- 相关 `summaries/WEB_*.md` 文件

### Inventory 系统（`archive/inventory/`）

- MineDojo Inventory 实现文档

### 技术问题修复记录

`issues/` 目录包含各类技术问题的诊断和修复记录（保留供参考）：
- `*MINEDOJO*.md` - MineDojo 相关问题
- `*MINERL*.md` - MineRL 相关问题
- `MACOS_*.md` - macOS 兼容性问题
- `STEVE1_*.md` - STEVE-1 集成问题

---

## 🔗 快速链接

### 新手入门

1. [部署指南](guides/DEPLOYMENT.md) - 环境安装
2. [评估框架指南](guides/EVALUATION_FRAMEWORK_GUIDE.md) - 运行评估
3. [任务配置指南](guides/TASK_WRAPPERS_GUIDE.md) - 自定义任务

### 开发参考

1. [STEVE-1 文档索引](guides/STEVE1_DOCS_INDEX.md)
2. [评估指标详解](guides/DEEP_EVALUATION_METRICS_EXPLAINED.md)
3. [框架设计文档](design/STEVE1_EVALUATION_FRAMEWORK_DESIGN.md)

### 问题排查

1. [失败任务修复](guides/FAILED_TASKS_FIX_GUIDE.md)
2. [issues/](issues/) - 历史问题修复记录

---

## 📝 文档维护

### 文档命名规范

- **指南文档**: `*_GUIDE.md` 或 `*_QUICKSTART.md`
- **设计文档**: `*_DESIGN.md` 或 `*_STRATEGY.md`
- **总结文档**: `*_SUMMARY.md` 或 `*_IMPLEMENTATION.md`
- **问题文档**: `*_FIX.md` 或 `*_DIAGNOSIS.md`

### 文档更新原则

1. 新功能 → 添加到 `guides/`
2. 架构变更 → 更新 `design/`
3. Bug 修复 → 记录到 `issues/`
4. 功能完成 → 总结到 `summaries/`

