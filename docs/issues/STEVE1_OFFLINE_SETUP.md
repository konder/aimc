# STEVE-1 离线运行配置指南

> **问题**: 无法访问 Hugging Face Hub + Prior 模型文件错误  
> **状态**: ✅ 已解决

---

## 🐛 问题 1: 无法访问 Hugging Face Hub

### **错误信息**

```
MaxRetryError("SOCKSHTTPSConnectionPool(host='huggingface.co', port=443): 
Max retries exceeded with url: /openai/clip-vit-base-patch16/resolve/main/tokenizer_config.json
```

### **原因**

MineCLIP 尝试从 Hugging Face Hub 下载 `openai/clip-vit-base-patch16` 的 tokenizer 文件，但网络无法访问。

### **✅ 解决方案**

#### **步骤 1: 设置本地缓存**

运行设置脚本：

```bash
cd /Users/nanzhang/aimc
./scripts/setup_huggingface_cache.sh
```

该脚本会：
1. 创建本地 Hugging Face 缓存目录
2. 将已有的 tokenizer 文件复制到正确位置
3. 创建符合 Hugging Face 格式的缓存结构

**缓存位置**: `/Users/nanzhang/aimc/data/huggingface_cache/`

#### **步骤 2: 配置环境变量**

**方式 A - 永久配置** (推荐):

```bash
# 添加到 ~/.zshrc
cat >> ~/.zshrc << 'EOF'
# Hugging Face 本地缓存
export HF_HOME="/Users/nanzhang/aimc/data/huggingface_cache"
export TRANSFORMERS_CACHE="/Users/nanzhang/aimc/data/huggingface_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
EOF

# 重新加载配置
source ~/.zshrc
```

**方式 B - 临时配置**:

所有 STEVE-1 脚本已经自动设置这些环境变量，直接运行即可：

```bash
cd /Users/nanzhang/aimc/src/training/steve1
./1_gen_paper_videos.sh  # 已包含环境变量设置
```

#### **步骤 3: 验证**

```bash
# 检查缓存目录
ls -lh /Users/nanzhang/aimc/data/huggingface_cache/hub/models--openai--clip-vit-base-patch16/snapshots/main/

# 应该看到:
# - tokenizer_config.json
# - vocab.json
# - merges.txt
# - special_tokens_map.json
# - config.json
```

---

## 🐛 问题 2: Prior 模型加载失败

### **错误信息**

```
RuntimeError: Error(s) in loading state_dict for TranslatorVAE:
	Missing key(s) in state_dict: "encoder.0.weight", "encoder.0.bias", ...
	Unexpected key(s) in state_dict: "net.img_process.cnn.stacks.0.firstconv.layer.weight", ...
```

### **原因**

`/Users/nanzhang/aimc/data/weights/steve1/steve1_prior.pt` 文件（952MB）是一个完整的 **VPT 策略网络**，而不是应该的 **VAE Prior 模型**（应该只有几MB）。

### **诊断**

运行诊断脚本确认：

```bash
cd /Users/nanzhang/aimc
python src/training/steve1/diagnose_prior.py
```

**预期输出**:
```
文件大小: 952.0 MB
❌ 这是一个 VPT 策略网络，不是 VAE Prior！
```

### **✅ 解决方案**

#### **方案 A: 下载正确的 Prior 模型** (推荐)

从 STEVE-1 官方 GitHub 下载正确的 `steve1_prior.pt`:

```bash
cd /Users/nanzhang/aimc/data/weights/steve1

# 备份错误的文件
mv steve1_prior.pt steve1_prior.pt.backup_vpt_network

# 从官方下载（需要访问 GitHub）
# 链接: https://github.com/Shalev-Lifshitz/STEVE-1
# 运行其 download_weights.sh 脚本
```

**正确的文件特征**:
- 大小: < 50 MB
- 包含: `encoder.*` 和 `decoder.*` 层
- 不包含: `net.*`, `pi_head.*`, `value_head.*` 层

#### **方案 B: 只使用 Visual Prompts** (临时)

如果无法下载 Prior 模型，可以修改代码只使用 visual prompts：

**修改 `run_agent/run_agent.py`**:

```python
# 在 line 96-103，注释掉 text prompts
if args.custom_text_prompt is not None:
    # 暂时跳过 text prompts
    print("⚠️  Prior 模型不可用，跳过 text prompts")
    sys.exit(0)
else:
    # 只生成 visual prompt 视频
    visual_prompt_embeds = load_visual_prompt_embeds()
    generate_visual_prompt_videos(
        visual_prompt_embeds, args.in_model, args.in_weights,
        args.visual_cond_scale, args.gameplay_length, args.save_dirpath
    )
```

#### **方案 C: 训练自己的 Prior 模型**

如果有数据，可以训练 Prior:

```bash
cd /Users/nanzhang/aimc/src/training/steve1
./4_train_prior.sh
```

**需要**:
- Prior 训练数据集
- 充足的计算资源

---

## 📊 修改的文件

### **新增文件**

1. **`scripts/setup_huggingface_cache.sh`**
   - 自动设置本地 Hugging Face 缓存
   - 复制 tokenizer 文件到正确位置

2. **`data/huggingface_cache/`** (目录)
   - 本地缓存目录
   - 包含 CLIP tokenizer 文件

### **修改的脚本**

所有 STEVE-1 运行脚本已添加环境变量设置：

1. **`src/training/steve1/1_gen_paper_videos.sh`**
2. **`src/training/steve1/2_gen_vid_for_text_prompt.sh`**
3. **`src/training/steve1/3_run_interactive_session.sh`**

添加的内容：
```bash
# 设置 Hugging Face 缓存路径（避免在线下载）
export HF_HOME="$PROJECT_ROOT/data/huggingface_cache"
export TRANSFORMERS_CACHE="$PROJECT_ROOT/data/huggingface_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

---

## 🚀 使用指南

### **1. 一次性设置**

```bash
# 步骤 1: 运行缓存设置脚本
cd /Users/nanzhang/aimc
./scripts/setup_huggingface_cache.sh

# 步骤 2: 添加环境变量到 shell 配置（可选）
cat >> ~/.zshrc << 'EOF'
export HF_HOME="/Users/nanzhang/aimc/data/huggingface_cache"
export TRANSFORMERS_CACHE="/Users/nanzhang/aimc/data/huggingface_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
EOF
source ~/.zshrc
```

### **2. 运行 STEVE-1**

```bash
cd /Users/nanzhang/aimc/src/training/steve1

# 如果有正确的 Prior 模型
./2_gen_vid_for_text_prompt.sh

# 如果没有 Prior 模型（仅 visual prompts）
# 需要修改脚本或使用 visual_only 版本
```

---

## 🔍 验证配置

### **检查 Hugging Face 缓存**

```bash
# 验证缓存目录存在
ls -lh /Users/nanzhang/aimc/data/huggingface_cache/hub/models--openai--clip-vit-base-patch16/snapshots/main/

# 应该看到所有 tokenizer 文件
```

### **检查 Prior 模型**

```bash
# 检查文件大小
ls -lh /Users/nanzhang/aimc/data/weights/steve1/steve1_prior.pt

# 如果 > 100MB，说明是错误的文件
# 应该 < 50MB
```

### **测试运行**

```bash
cd /Users/nanzhang/aimc/src/training/steve1

# 测试是否还会尝试访问 huggingface.co
./2_gen_vid_for_text_prompt.sh 2>&1 | grep "huggingface.co"

# 如果没有输出，说明配置成功
```

---

## ⚠️ 已知问题

### **1. Prior 模型文件错误**

**状态**: ⚠️ 未解决（需要下载正确文件）

**临时方案**: 只使用 visual prompts

**永久方案**: 从 STEVE-1 官方下载正确的 `steve1_prior.pt`

### **2. 网络访问限制**

**状态**: ✅ 已解决（使用本地缓存）

**配置**: 所有脚本已自动设置离线模式

---

## 📚 相关文档

- **配置指南**: `docs/guides/STEVE1_PATH_CONFIGURATION_GUIDE.md`
- **评估指南**: `docs/guides/STEVE1_EVALUATION_GUIDE.md`
- **下载指南**: `docs/reference/STEVE1_MODEL_DOWNLOAD_GUIDE.md`

---

## 💡 总结

### **Hugging Face 问题**
✅ **已解决**: 使用本地缓存 + 离线模式

### **Prior 模型问题**
⚠️ **需要下载**: 当前文件是错误的 VPT 网络

**临时解决方案**: 使用 visual prompts  
**永久解决方案**: 下载正确的 Prior 模型（< 50MB）

---

**最后更新**: 2025-10-30  
**测试环境**: macOS, conda minedojo-x86, Python 3.9

