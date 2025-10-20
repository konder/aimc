# 离线训练设置指南

本指南介绍如何配置 MineCLIP 训练环境以支持离线使用，避免每次训练都访问 HuggingFace。

---

## 📦 需要下载的文件

### 1. MineCLIP 预训练模型（已完成）

MineCLIP 预训练权重已保存在 `data/mineclip/` 目录：

```bash
data/mineclip/
├── attn.pth    # attn 变体（推荐）
└── avg.pth     # avg 变体
```

### 2. CLIP Tokenizer（必需）

首次使用前需要下载 tokenizer：

```bash
# 在 minedojo-x86 环境中运行
python scripts/download_clip_tokenizer.py
```

下载成功后，文件会保存在 `data/clip_tokenizer/`：

```bash
data/clip_tokenizer/
├── tokenizer_config.json
├── special_tokens_map.json
├── vocab.json
└── merges.txt
```

---

## ✅ 验证离线设置

### 检查所有文件是否就绪：

```bash
# 检查 MineCLIP 模型
ls -lh data/mineclip/

# 检查 tokenizer
ls -lh data/clip_tokenizer/
```

### 预期输出：

```
data/mineclip/:
-rw-r--r--  1 user  staff   577M  attn.pth
-rw-r--r--  1 user  staff   577M  avg.pth

data/clip_tokenizer/:
-rw-r--r--  1 user  staff   0.7K  tokenizer_config.json
-rw-r--r--  1 user  staff   0.6K  special_tokens_map.json
-rw-r--r--  1 user  staff   512K  merges.txt
-rw-r--r--  1 user  staff   1.0M  vocab.json
```

---

## 🚀 离线训练

配置完成后，即使断网也可以正常训练：

```bash
# 使用 MineCLIP 训练（完全离线）
python src/training/train_get_wood.py --use-mineclip --total-timesteps 10000
```

### 训练输出会显示：

```
  MineCLIP 奖励包装器:
    任务描述: chop down a tree and collect one wood log
    模型变体: attn
    稀疏权重: 10.0
    MineCLIP权重: 0.1
    设备: mps
    正在加载 MineCLIP attn 模型...
    从 data/mineclip/attn.pth 加载权重...
    ✓ 权重加载成功
    使用本地 tokenizer: data/clip_tokenizer  ← 离线模式
    状态: ✓ MineCLIP 模型已加载
```

---

## 🔧 故障排除

### 问题 1: 仍然访问 HuggingFace

**症状**: 看到 "Retrying in Xs" 或连接 huggingface.co

**原因**: `data/clip_tokenizer/` 目录不存在或为空

**解决**:
```bash
python scripts/download_clip_tokenizer.py
```

### 问题 2: Tokenizer 下载失败

**症状**: `ConnectionResetError` 或 `Max retries exceeded`

**解决方案 1**: 使用国内镜像
```bash
# 设置 HuggingFace 镜像
export HF_ENDPOINT=https://hf-mirror.com
python scripts/download_clip_tokenizer.py
```

**解决方案 2**: 手动下载
从这个链接下载 tokenizer 文件：
https://huggingface.co/openai/clip-vit-base-patch16/tree/main

将以下文件保存到 `data/clip_tokenizer/`:
- `tokenizer_config.json`
- `special_tokens_map.json`
- `vocab.json`
- `merges.txt`

### 问题 3: MineCLIP 模型文件损坏

**症状**: `Error(s) in loading state_dict`

**解决**:
重新下载 MineCLIP 预训练模型：
```bash
# 从 GitHub Releases 下载
# https://github.com/MineDojo/MineCLIP/releases
```

---

## 📝 注意事项

1. **文件大小**: 
   - MineCLIP 模型: ~577MB × 2
   - Tokenizer: ~1.5MB
   - 总计约 1.2GB

2. **网络需求**:
   - 首次下载需要联网
   - 之后完全离线

3. **更新策略**:
   - MineCLIP 模型不会自动更新（手动下载）
   - Tokenizer 不会自动更新（除非删除本地文件）

4. **团队协作**:
   - `data/` 目录已在 `.gitignore` 中
   - 每个开发者需要独立下载这些文件
   - 可以通过内网共享加速团队部署

---

## 🎯 快速启动检查清单

- [ ] 下载 MineCLIP 模型 (`attn.pth`, `avg.pth`)
- [ ] 运行 `python scripts/download_clip_tokenizer.py`
- [ ] 验证 `data/mineclip/` 和 `data/clip_tokenizer/` 存在
- [ ] 运行测试训练 `python src/training/train_get_wood.py --use-mineclip --total-timesteps 100`
- [ ] 确认输出显示 "使用本地 tokenizer"

---

## 📚 相关文档

- [MineCLIP 设置指南](./MINECLIP_SETUP_GUIDE.md)
- [训练指南](./GET_WOOD_TRAINING_GUIDE.md)
- [故障排除](../FAQ.md)

