# 📦 归档文档目录

> **归档日期**: 2025-12-09

项目已进行重大方向调整，当前聚焦于 **STEVE-1 模型评估和训练改进**。以下功能模块的文档已归档：

---

## 📁 归档目录结构

```
archive/
├── dagger/           # DAgger 迭代训练相关
├── vpt/              # VPT 独立评估相关
├── web/              # Web 控制台相关
└── inventory/        # MineDojo Inventory 相关
```

---

## 🗂️ 归档内容说明

### 1. DAgger (`archive/dagger/`)

DAgger（Dataset Aggregation）算法相关文档，包含行为克隆和迭代训练。

**归档原因**: 项目转向 STEVE-1 模型，不再使用 DAgger 训练流程。

**主要文档**:
- `DAGGER_COMPREHENSIVE_GUIDE.md` - DAgger 完整指南
- 原路径：`docs/guides/DAGGER_*.md`, `docs/technical/DAGGER_*.md`

### 2. VPT (`archive/vpt/`)

VPT（Video Pre-Training）独立评估和训练文档。

**归档原因**: VPT 现在作为 STEVE-1 的基础模型使用，不再单独评估。

**主要文档**:
- `VPT_ZERO_SHOT_QUICKSTART.md` - VPT 零样本评估快速开始
- 原路径：`docs/guides/VPT_*.md`, `docs/summaries/VPT_*.md`

### 3. Web (`archive/web/`)

DAgger Web 控制台界面文档。

**归档原因**: Web 控制台功能暂停维护。

**主要文档**:
- `WEB_COMPREHENSIVE_GUIDE.md` - Web 控制台完整指南
- 原路径：`docs/guides/WEB_*.md`, `docs/summaries/WEB_*.md`

### 4. Inventory (`archive/inventory/`)

MineDojo 物品栏系统相关文档。

**归档原因**: 物品栏功能已稳定，暂不需要维护。

**主要文档**:
- `README.md` - 归档说明
- 原路径：`docs/summaries/INVENTORY_*.md`, `docs/summaries/MINEDOJO_INVENTORY_*.md`

---

## 🔍 如何查找原文档

如需查看完整历史文档，可使用 Git 历史：

```bash
# 查看文档历史版本
git log --oneline -- docs/guides/DAGGER_COMPREHENSIVE_GUIDE.md

# 恢复特定版本
git show <commit>:docs/guides/DAGGER_COMPREHENSIVE_GUIDE.md
```

---

## 📌 当前活跃文档

以下文档目录仍在活跃维护：

| 目录 | 说明 |
|------|------|
| `docs/guides/DEPLOYMENT.md` | 部署指南 |
| `docs/guides/EVALUATION_FRAMEWORK_GUIDE.md` | 评估框架指南 |
| `docs/guides/STEVE1_*.md` | STEVE-1 相关指南 |
| `docs/technical/STEVE1_*.md` | STEVE-1 技术文档 |

详细索引请参考：[DOCUMENTATION_INDEX.md](../DOCUMENTATION_INDEX.md)

