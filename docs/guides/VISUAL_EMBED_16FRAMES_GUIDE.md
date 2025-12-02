# 视觉嵌入16帧视频模式说明

> 📚 **配套文档**: 
> - [完整工作流程](../../COMPLETE_VISUAL_EMBED_WORKFLOW.md)
> - [修复总结](../../VISUAL_EMBED_FIX_SUMMARY.md)

## 🎯 核心发现

根据[STEVE-1源代码](https://github.com/Shalev-Lifshitz/STEVE-1/blob/903b244796322f4d0073a8f62c05f51eac3aed52/steve1/utils/embed_utils.py#L8)确认：

**Prior VAE训练时对齐的是16帧视频的嵌入，而不是单帧图像嵌入！**

### 证据链

1. **Prior训练数据生成**：
```python
# STEVE-1训练数据对
text_embed = mineclip.encode_text("chop tree")  # [512]
video_frames = [16帧视频]  # [16, 3, 160, 256]
visual_embed = mineclip.encode_video(video_frames)  # [512] ← 16帧整体

# 训练对: (text_embed, visual_embed)
prior_model.train(text_embed → visual_embed)
```

2. **`get_prior_embed`返回值**：
```python
def get_prior_embed(text, mineclip, prior, device):
    text_embed = mineclip.encode_text(text)
    prior_embed = prior(text_embed)  # ← 对齐16帧视频嵌入空间
    return prior_embed
```

3. **评估时的正确比较**：
```python
# ✅ 正确：同一空间的比较
prior_embed = get_prior_embed("chop tree", ...)  # [512] 对齐16帧视频
visual_embed = mineclip.encode_video(success_frames_16)  # [512] 16帧视频
similarity = cosine_similarity(prior_embed, visual_embed)
```

---

## 🚀 快速开始

```bash
# 一键重新生成所有视觉嵌入
bash scripts/regenerate_visual_embeds.sh
```

---

## 📖 更多信息

- [完整工作流程](../../COMPLETE_VISUAL_EMBED_WORKFLOW.md) - 详细的3步流程
- [修复总结](../../VISUAL_EMBED_FIX_SUMMARY.md) - 问题分析和修复对比

---

**更新时间**: 2025-12-02  
**版本**: v2.0 (16帧视频模式)
