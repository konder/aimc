# STEVE-1 快速启动指南

## 环境验证

在运行 STEVE-1 之前，请先验证环境配置：

```bash
cd /Users/nanzhang/aimc
export HF_HOME="/Users/nanzhang/aimc/data/huggingface_cache"
export HF_HUB_OFFLINE=1
conda run -n minedojo-x86 python scripts/verify_steve1_setup.py
```

应该看到所有 ✅ 标记。

## 运行 STEVE-1

### 选项 1: 生成论文中的演示视频（推荐）

```bash
cd /Users/nanzhang/aimc
bash src/training/steve1/1_gen_paper_videos.sh
```

这将生成 13 个任务的演示视频，保存到 `data/generated_videos/paper_prompts/`

### 选项 2: 自定义文本提示

```bash
cd /Users/nanzhang/aimc
bash src/training/steve1/2_gen_vid_for_text_prompt.sh
```

修改脚本中的 `--text_prompt` 参数来自定义指令。

### 选项 3: 交互式会话

```bash
cd /Users/nanzhang/aimc
bash src/training/steve1/3_run_interactive_session.sh
```

## 已知问题

### Prior 模型文件不正确

当前的 `steve1_prior.pt` 文件不正确（是一个完整的 VPT 模型而不是 VAE Prior）。

**影响**: Text-to-behavior 功能可能无法正常工作。

**临时解决方案**: 使用视觉提示模式（不使用文本指令）。

**永久解决方案**: 从 STEVE-1 官方仓库获取正确的 prior 文件。

## 性能说明

- **设备**: Mac (CPU only)
- **速度**: CPU 推理较慢，每个视频可能需要数分钟到数十分钟
- **建议**: 
  - 先测试短视频（减少 `--gameplay_length`）
  - 生产环境建议使用配备 GPU 的服务器

## 输出

生成的视频将保存到：
- `data/generated_videos/paper_prompts/` - 论文演示视频
- `data/generated_videos/custom/` - 自定义提示视频

## 故障排除

### 错误: "Cannot find the requested files in the disk cache"

**解决**: 确保设置了 `HF_HOME` 环境变量：
```bash
export HF_HOME="/Users/nanzhang/aimc/data/huggingface_cache"
export HF_HUB_OFFLINE=1
```

### 错误: "AssertionError: Torch not compiled with CUDA enabled"

**解决**: 已修复！代码现在自动使用 CPU。

### MineRL 环境错误

这是已知的 MineRL 问题。脚本会自动重启并跳过已生成的视频。

## 更多信息

- [完整修复文档](../summaries/STEVE1_INTEGRATION_FIXES.md)
- [评估指南](../guides/STEVE1_EVALUATION_GUIDE.md)
- [模型下载](../reference/STEVE1_MODEL_DOWNLOAD_GUIDE.md)

