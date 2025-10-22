# 🚀 快速开始 - MineCLIP 16帧视频模式

> **状态**: 代码已完成，可以开始录制验证 ✅  
> **目标**: 验证MineCLIP是否能正确识别砍树过程

---

## ⚡ **30秒快速开始**

```bash
# 1. 激活环境
conda activate minedojo-x86

# 2. 录制砍树过程（手动控制）
python tools/record_manual_chopping.py --output-dir logs/my_chopping

# 3. 验证MineCLIP效果
python tools/verify_mineclip_16frames.py --sequence-dir logs/my_chopping

# 4. 查看结果
cat logs/my_chopping/similarity_results.txt
```

---

## 🎮 **键盘控制（录制时）**

**移动**: `W`/`A`/`S`/`D`  
**转头**: 方向键 `↑`/`↓`/`←`/`→`  
**砍树**: `J`（攻击）⭐  
**跳跃**: `Space`  
**停止**: `Q`（保存并退出）

---

## 🎯 **核心思想**

你手动完成砍树时，MineCLIP相似度应该呈现：

```
寻找树(低) → 靠近树(上升) → 砍树中(最高⭐) → 收集(下降)
```

**如果MineCLIP配置正确，应该能识别出这个模式！**

---

## 📊 **判断标准**

验证后查看**相似度变化范围**：

| 范围 | 评估 | 建议 |
|------|------|------|
| > 0.05 | ✅ 优秀 | 立即用16帧模式训练 |
| 0.02-0.05 | ⚠️ 可用 | 尝试调整提示词 |
| < 0.02 | ❌ 效果差 | 考虑混合奖励方案 |

---

## 🔬 **录制技巧**

1. **完整流程** - 录制完整的寻找→靠近→砍树过程
2. **保持稳定** - 砍树时保持视角稳定，树干在画面中央
3. **连续攻击** - 按住`J`键直到树被砍断
4. **确认完成** - 看到"🎉 获得木头！"后按`Q`保存

---

## 💡 **常见问题**

**Q: 键盘没反应？**  
A: 点击OpenCV窗口使其获得焦点

**Q: 找不到树？**  
A: 使用方向键环顾四周，森林环境应该有树

**Q: 报错 AssertionError？**  
A: 已修复！使用`tools/record_manual_chopping.py`（8维正确action）

---

## 🚀 **如果验证效果好（变化范围 > 0.05）**

立即开始16帧视频模式训练：

```bash
./scripts/train_get_wood.sh test \
    --timesteps 10000 \
    --use-mineclip \
    --use-video-mode \
    --num-frames 16 \
    --compute-frequency 4
```

---

## 📖 **详细文档**

- `tools/README.md` - 工具使用说明
- `docs/guides/MANUAL_RECORDING_GUIDE.md` - 详细录制指南
- `docs/START_HERE.md` - 项目总体介绍

---

**现在开始录制！** 🎬

```bash
python tools/record_manual_chopping.py --output-dir logs/my_chopping
```

