# STEVE-1 集成完整修复总结

## 修复时间
2024-10-30

## 问题概述
将 STEVE-1 代码集成到 AIMC 项目时遇到了以下问题：
1. Hugging Face 缓存结构不正确
2. 硬编码使用 CUDA 设备导致在 Mac 上无法运行
3. 权重文件路径不正确
4. Prior 模型文件错误

## 已完成的修复

### 1. Hugging Face CLIP Tokenizer 缓存结构修复 ✅

**问题**: 
- Tokenizer 尝试从 Hugging Face Hub 在线下载 `tokenizer_config.json`
- 错误: `LocalEntryNotFoundError: Cannot find the requested files in the disk cache`

**原因**:
- 缓存目录结构不符合 Hugging Face 官方规范
- 使用了 `snapshots/main/` 而不是 `snapshots/<commit_hash>/`
- 缺少 `refs/main` 文件和正确的符号链接

**解决方案**:
按照 Hugging Face 官方文档创建正确的缓存结构：

```
data/huggingface_cache/hub/
└── models--openai--clip-vit-base-patch16/
    ├── refs/
    │   └── main  (内容: commit hash)
    ├── blobs/
    │   ├── <hash1> (实际文件内容)
    │   ├── <hash2>
    │   └── ...
    └── snapshots/
        └── e6a30b0cf221a00a84c6e6bbd99e1cfb5b16b827/  (commit hash)
            ├── config.json -> ../../blobs/<hash>
            ├── tokenizer_config.json -> ../../blobs/<hash>
            ├── tokenizer.json -> ../../blobs/<hash>
            ├── vocab.json -> ../../blobs/<hash>
            ├── merges.txt -> ../../blobs/<hash>
            ├── special_tokens_map.json -> ../../blobs/<hash>
            └── preprocessor_config.json -> ../../blobs/<hash>
```

**使用的脚本**:
- `scripts/fix_hf_cache_correct_structure.py` - 创建正确的目录结构和 refs
- `scripts/create_symlinks.py` - 为 blobs 创建符号链接

**验证**:
```bash
export HF_HOME="/Users/nanzhang/aimc/data/huggingface_cache"
export HF_HUB_OFFLINE=1
python -c "from transformers import AutoTokenizer; \
    tokenizer = AutoTokenizer.from_pretrained('openai/clip-vit-base-patch16', local_files_only=True); \
    print('✅ 成功加载')"
```

### 2. CUDA/CPU 设备自动检测 ✅

**问题**:
- 代码硬编码 `device='cuda'`，在 Mac (无 NVIDIA GPU) 上报错
- 错误: `AssertionError: Torch not compiled with CUDA enabled`

**修复的文件**:

#### `src/training/steve1/config.py` (已正确配置)
```python
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
```

#### `src/training/steve1/utils/mineclip_agent_env_utils.py`
**修改前**:
```python
agent = MineRLConditionalAgent(env, device='cuda', ...)
```

**修改后**:
```python
agent = MineRLConditionalAgent(env, device=DEVICE, ...)
```

#### `src/training/steve1/evaluate.py`
**修改前**:
```python
agent = load_steve1_agent(..., device="cuda")
```

**修改后**:
```python
from src.training.steve1.config import DEVICE
agent = load_steve1_agent(..., device=DEVICE)
```

### 3. 权重文件路径修复 ✅

**修改的文件**:
- `src/training/steve1/config.py` - 使用 `DATA_DIR` 动态路径
- `src/training/steve1/run_agent/run_agent.py` - 更新默认参数路径
- `src/training/steve1/run_agent/paper_prompts.py` - 更新视觉提示路径
- 所有 `.sh` 脚本 - 使用 `$PROJECT_ROOT` 动态路径

**路径结构**:
```
/Users/nanzhang/aimc/data/
├── weights/
│   ├── vpt/
│   │   └── 2x.model
│   ├── mineclip/
│   │   └── attn.pth
│   └── steve1/
│       ├── steve1.weights
│       └── steve1_prior.pt
├── huggingface_cache/
│   └── hub/...
└── generated_videos/
    └── paper_prompts/
```

### 4. 环境变量配置 ✅

所有 STEVE-1 脚本现在都正确设置了以下环境变量：

```bash
export HF_HOME="$PROJECT_ROOT/data/huggingface_cache"
export HF_HUB_OFFLINE=1
```

## 验证工具

创建了 `scripts/verify_steve1_setup.py` 验证脚本，检查：
1. PyTorch 和 CUDA 可用性
2. 设备自动检测 (CPU/CUDA)
3. 权重文件是否存在
4. Hugging Face 缓存结构
5. Tokenizer 离线加载

**运行验证**:
```bash
cd /Users/nanzhang/aimc
export HF_HOME="/Users/nanzhang/aimc/data/huggingface_cache"
export HF_HUB_OFFLINE=1
conda run -n minedojo-x86 python scripts/verify_steve1_setup.py
```

**验证结果**:
```
✅ PyTorch 版本: 2.2.2
✅ CUDA 可用: False
ℹ️  将使用 CPU 运行
✅ 设备设置: cpu
✅ MineCLIP 权重文件存在
✅ Prior 权重文件存在
✅ 找到 1 个 Hugging Face snapshot
✅ Tokenizer 加载成功
✅ 词汇表大小: 49408
```

## 待解决问题

### Prior 模型文件不正确 ⚠️

**问题**:
- 当前的 `steve1_prior.pt` (952MB) 实际上是一个完整的 VPT 模型
- 包含 `net.*`, `pi_head.*`, `value_head.*` 等 VPT 模型键
- 应该是一个小型 VAE Prior 模型 (~10-50MB)

**临时解决方案**:
可以使用只基于视觉提示的模式，绕过 text-to-latent 的 VAE Prior

**永久解决方案**:
从 STEVE-1 官方仓库下载正确的 `steve1_prior.pt` 文件：
```bash
cd /Users/nanzhang/aimc/data/weights/steve1/
# 从官方仓库获取正确的 prior 文件
```

## 运行 STEVE-1

### 1. 生成论文中的演示视频
```bash
cd /Users/nanzhang/aimc
bash src/training/steve1/1_gen_paper_videos.sh
```

### 2. 使用自定义文本提示生成视频
```bash
cd /Users/nanzhang/aimc
bash src/training/steve1/2_gen_vid_for_text_prompt.sh
```

### 3. 交互式会话
```bash
cd /Users/nanzhang/aimc
bash src/training/steve1/3_run_interactive_session.sh
```

## 技术细节

### Hugging Face 缓存机制
参考: https://hugging-face.cn/docs/huggingface_hub/guides/manage-cache

Hugging Face 使用以下结构：
- `refs/` - 存储分支名到 commit hash 的映射
- `blobs/` - 存储实际文件内容（按 SHA256 哈希命名）
- `snapshots/<commit_hash>/` - 存储指向 blobs 的符号链接

这种设计允许：
1. 跨版本共享相同文件（节省空间）
2. 快速切换版本（只需改变符号链接）
3. 离线模式运行

### 设备自动检测
```python
import torch
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
```

在 Mac (x86/ARM) 上：
- 没有 NVIDIA GPU，`torch.cuda.is_available()` 返回 `False`
- 自动使用 CPU
- 可能较慢，但可以正常运行

在配备 NVIDIA GPU 的 Linux 服务器上：
- `torch.cuda.is_available()` 返回 `True`
- 自动使用 CUDA
- 显著加速推理速度

## 文件清单

### 修改的文件
- `src/training/steve1/config.py`
- `src/training/steve1/utils/mineclip_agent_env_utils.py`
- `src/training/steve1/evaluate.py`
- `src/training/steve1/run_agent/run_agent.py`
- `src/training/steve1/run_agent/paper_prompts.py`
- `src/training/steve1/*.sh` (所有脚本)

### 创建的工具脚本
- `scripts/fix_hf_cache_correct_structure.py` - 修复缓存结构
- `scripts/create_symlinks.py` - 创建符号链接
- `scripts/verify_steve1_setup.py` - 验证环境配置

### 创建的文档
- `docs/summaries/STEVE1_INTEGRATION_FIXES.md` (本文件)

## 相关文档

- [STEVE-1 评估指南](../guides/STEVE1_EVALUATION_GUIDE.md)
- [STEVE-1 模型下载指南](../reference/STEVE1_MODEL_DOWNLOAD_GUIDE.md)
- [MineCLIP 指令驱动代理](../technical/MINECLIP_INSTRUCTION_DRIVEN_AGENT.md)
- [VPT 模型参考](../reference/VPT_MODELS_REFERENCE.md)

## 下一步

1. **获取正确的 Prior 模型** - 从官方仓库下载正确的 `steve1_prior.pt`
2. **测试完整流程** - 运行 `1_gen_paper_videos.sh` 生成所有演示视频
3. **性能优化** - 如果 CPU 推理太慢，考虑：
   - 使用更小的模型
   - 减少 gameplay_length
   - 使用 GPU 服务器
4. **集成到训练流程** - 将 STEVE-1 作为 baseline 或数据生成器集成到现有训练流程

## 总结

✅ **Hugging Face 缓存**: 完全修复，可以离线加载 CLIP tokenizer  
✅ **设备检测**: 自动检测 CUDA/CPU，Mac 上可以正常运行  
✅ **权重路径**: 所有路径都使用动态配置，符合项目结构  
⚠️ **Prior 模型**: 需要下载正确的文件（当前文件不正确）  

当前状态：**可以运行，但 text-to-behavior 功能受限（Prior 模型不正确）**

