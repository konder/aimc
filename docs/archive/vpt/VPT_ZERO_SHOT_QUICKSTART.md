# VPT零样本评估快速开始

本指南介绍如何快速运行VPT零样本评估，测试预训练VPT模型在harvest_log任务上的表现。

## 前置条件

1. **VPT权重文件已下载**
   ```bash
   ls data/pretrained/vpt/rl-from-early-game-2x.weights
   ```

2. **MineDojo环境已安装**
   ```bash
   conda activate minedojo
   python -c "import minedojo; print('✓ MineDojo OK')"
   ```

## 快速启动

```bash
# 默认设置：评估10轮，自动检测设备
bash scripts/evaluate_vpt_zero_shot.sh

# 自定义评估轮数
bash scripts/evaluate_vpt_zero_shot.sh 20
```

---

## ⚠️ 归档说明

此文档已归档。项目当前聚焦于 **STEVE-1 模型评估和训练改进**，VPT 独立评估功能暂不维护。

**归档日期**: 2025-12-09
**原路径**: docs/guides/VPT_ZERO_SHOT_QUICKSTART.md

