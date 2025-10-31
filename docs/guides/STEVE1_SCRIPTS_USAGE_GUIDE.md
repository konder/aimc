# STEVE-1 脚本使用指南

本文档详细说明 STEVE-1 训练和推理相关脚本的用途和使用方法。

## 📋 目录

- [前置准备](#前置准备)
- [脚本概览](#脚本概览)
- [阶段 1：数据准备](#阶段-1数据准备)
- [阶段 2：模型推理与评估](#阶段-2模型推理与评估)
- [阶段 3：模型训练](#阶段-3模型训练)
- [阶段 4：Prior 训练](#阶段-4prior-训练)
- [常见问题](#常见问题)

---

## 前置准备

### 环境要求
```bash
# 激活 MineDojo 环境
conda activate minedojo

# 确保模型权重已下载
# 参考: docs/reference/STEVE1_MODEL_DOWNLOAD_GUIDE.md
```

### 必需的模型权重
确保以下权重文件已准备好：
- `data/weights/vpt/2x.model` - VPT 基础模型架构
- `data/weights/steve1/steve1.weights` - STEVE-1 预训练权重
- `data/weights/steve1/steve1_prior.pt` - STEVE-1 Prior 权重
- `data/weights/vpt/rl-from-foundation-2x.weights` - （仅训练时需要）VPT RL 权重

### Hugging Face 离线模式
所有脚本已配置为离线模式，使用本地缓存：
```bash
export HF_HOME="$PROJECT_ROOT/data/huggingface_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

---

## 脚本概览

| 脚本名称 | 阶段 | 用途 | 执行时间 |
|---------|------|------|---------|
| `1_generate_dataset.sh` | 数据准备 | 下载并转换 VPT 数据集 | ~数小时 |
| `2_create_sampling.sh` | 数据准备 | 生成训练采样配置 | ~分钟 |
| `1_gen_paper_videos.sh` | 推理 | 生成论文演示视频 | ~小时级 |
| `2_gen_vid_for_text_prompt.sh` | 推理 | 为自定义提示生成视频 | ~分钟 |
| `3_run_interactive_session.sh` | 推理 | 交互式文本控制会话 | 持续运行 |
| `3_train.sh` | 训练 | 训练 STEVE-1 主模型 | ~天级 |
| `4_train_prior.sh` | 训练 | 训练 VAE Prior | ~小时级 |

---

## 阶段 1：数据准备

### 1.1 `1_generate_dataset.sh` - 下载并转换数据集

#### 用途
从 OpenAI VPT 公开数据集下载原始游戏录像，并转换为 STEVE-1 训练所需的格式。

#### 功能说明
1. **Contractor 数据集**：下载专业玩家录像（3 个索引：8.x, 9.x, 10.x）
2. **Mixed Agents 数据集**：生成混合智能体行为数据
3. 自动进行数据格式转换和预处理

#### 使用方法
```bash
cd src/training/steve1
bash 1_generate_dataset.sh
```

#### 重要参数配置
编辑脚本以调整以下参数：

```bash
# Contractor 数据集
OUTPUT_DIR_CONTRACTOR="$PROJECT_ROOT/data/dataset_contractor/"
N_EPISODES_CONTRACTOR=5  # 每个索引下载的 episode 数量（建议: 5-20）

# Mixed Agents 数据集
OUTPUT_DIR_MIXED_AGENTS="$PROJECT_ROOT/data/dataset_mixed_agents/"
N_EPISODES_MIXED_AGENTS=5  # 生成的 episode 数量（建议: 5-10）
```

#### 输出位置
- Contractor 数据：`data/dataset_contractor/`
- Mixed Agents 数据：`data/dataset_mixed_agents/`

#### 注意事项
- **首次使用必须运行**：需要准备训练数据
- **磁盘空间需求**：每个 episode 约 500MB-1GB
- **网络需求**：需要访问 OpenAI 数据存储（首次运行）
- **执行时间**：根据 episode 数量和网络速度，可能需要数小时

---

### 1.2 `2_create_sampling.sh` - 生成训练采样配置

#### 用途
创建训练时使用的数据采样策略配置文件，定义训练集和验证集的帧数分配。

#### 功能说明
- 生成 NeurIPS 标准采样配置
- 分配训练帧和验证帧数量
- 用于训练脚本的 `--sampling` 参数

#### 使用方法
```bash
cd src/training/steve1
bash 2_create_sampling.sh
```

#### 参数说明
```bash
--type neurips          # 采样类型（neurips 是标准配置）
--name neurips          # 配置名称
--output_dir data/samplings/  # 输出目录
--val_frames 10_000     # 验证集帧数
--train_frames 30_000   # 训练集帧数
```

#### 输出位置
- 采样配置：`data/samplings/neurips/`

#### 注意事项
- **必须在训练前运行**：生成 `3_train.sh` 所需的配置
- **路径问题**：脚本中的路径是相对路径，需要修正为绝对路径（见下方修复）

#### 已知问题与修复
**问题**：脚本使用相对路径，可能导致路径错误

**修复方案**：
```bash
# 修改后的脚本（使用绝对路径）
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../../.." && pwd )"

python "$SCRIPT_DIR/data/sampling/generate_sampling.py" \
--type neurips \
--name neurips \
--output_dir "$PROJECT_ROOT/data/samplings/" \
--val_frames 10_000 \
--train_frames 30_000
```

---

## 阶段 2：模型推理与评估

### 2.1 `1_gen_paper_videos.sh` - 生成论文演示视频

#### 用途
生成论文中展示的各种行为演示视频，用于验证模型能力和展示效果。

#### 功能说明
- 使用预训练的 STEVE-1 模型生成游戏行为视频
- 支持渲染窗口实时预览
- 自动重试机制（MineRL 不稳定时自动重启）
- 跳过已生成的视频（支持断点续传）

#### 使用方法
```bash
cd src/training/steve1
bash 1_gen_paper_videos.sh
```

#### 关键参数配置

**渲染开关**（实时预览游戏窗口）：
```bash
# 禁用渲染（更快，无窗口）
RENDER_FLAG=""

# 启用渲染（可以看到实时画面，速度较慢）
RENDER_FLAG="--render"
```

**生成参数**：
```bash
--text_cond_scale 6.0       # 文本条件强度（建议: 4.0-8.0）
--visual_cond_scale 7.0     # 视觉条件强度（建议: 5.0-9.0）
--gameplay_length 3000      # 游戏步数（1 步 ≈ 0.05 秒，3000 ≈ 2.5 分钟）
--custom_text_prompt "..."  # 自定义文本提示
```

#### 自定义文本提示示例
编辑脚本第 28 行：
```bash
--custom_text_prompt "build wooden house"    # 建造木屋
--custom_text_prompt "dig dirt"               # 挖土
--custom_text_prompt "chop tree"              # 砍树
--custom_text_prompt "swim in water"          # 游泳
```

#### 输出位置
- 视频保存：`data/generated_videos/paper_prompts/`
- 文件命名：`'<prompt_text>' - Text Prompt.mp4`

#### 自动重试机制
脚本包含无限循环和错误处理：
```bash
while [ $EXIT_STATUS -ne 0 ]; do
    echo "Encountered an error (likely internal MineRL error), restarting..."
    sleep 10
    $COMMAND
    EXIT_STATUS=$?
done
```

#### 注意事项
- **MineRL 不稳定**：可能随机崩溃，脚本会自动重启
- **渲染性能**：启用 `--render` 会显著降低生成速度（约 2-3 倍慢）
- **显卡需求**：需要 CUDA GPU，建议 8GB+ 显存
- **执行时间**：单个视频约 10-30 分钟（取决于长度和硬件）

---

### 2.2 `2_gen_vid_for_text_prompt.sh` - 为自定义文本生成视频

#### 用途
快速为单个自定义文本提示生成演示视频，用于测试和实验。

#### 与 `1_gen_paper_videos.sh` 的区别
| 特性 | `1_gen_paper_videos.sh` | `2_gen_vid_for_text_prompt.sh` |
|------|------------------------|-------------------------------|
| 用途 | 批量生成多个视频 | 快速测试单个提示 |
| 自动重试 | ✅ 有 | ❌ 无 |
| 游戏长度 | 3000 步（~2.5 分钟） | 1000 步（~50 秒） |
| 默认渲染 | 启用 | 禁用 |

#### 使用方法
```bash
cd src/training/steve1
bash 2_gen_vid_for_text_prompt.sh
```

#### 快速测试文本提示
编辑脚本第 27 行：
```bash
--custom_text_prompt "look at the sky"    # 看天空
--custom_text_prompt "find diamonds"      # 寻找钻石
--custom_text_prompt "build shelter"      # 建造庇护所
```

#### 输出位置
- 视频保存：`data/generated_videos/custom_text_prompt/`

#### 使用场景
- ✅ 快速验证模型对新提示的响应
- ✅ 调试文本条件参数
- ✅ 生成短视频演示
- ❌ 不适合批量生产视频（无自动重试）

---

### 2.3 `3_run_interactive_session.sh` - 交互式会话

#### 用途
启动交互式 Minecraft 会话，可以在游戏运行时动态输入文本提示，实时控制 AI 行为。

#### 功能说明
- **实时文本输入**：游戏运行时点击窗口暂停并输入新提示
- **连续控制**：AI 根据新提示调整行为
- **视频录制**：自动保存每次交互的视频
- **纯文本模式**：仅支持文本条件（不支持视觉提示）

#### 使用方法
```bash
cd src/training/steve1
bash 3_run_interactive_session.sh
```

#### 交互操作
1. 脚本启动后会打开 Minecraft 窗口
2. AI 开始执行默认行为
3. **点击游戏窗口** → 暂停游戏
4. **在终端输入文本提示** → 按 Enter
5. AI 继续执行，根据新提示调整行为
6. 重复步骤 3-5 进行多轮交互

#### 示例交互流程
```
开始 → "chop tree" → 观察 10 秒 → 暂停
      → "build crafting table" → 观察 15 秒 → 暂停
      → "explore cave" → 持续观察
```

#### 参数说明
```bash
--cond_scale 6.0  # 条件强度（文本影响力，建议: 4.0-8.0）
--output_video_dirpath  # 视频保存目录
```

#### 输出位置
- 视频保存：`data/generated_videos/interactive_videos/`
- 每次会话生成一个视频文件

#### 注意事项
- **必须启用渲染**：交互模式需要看到窗口
- **终端输入**：提示词在终端输入，不是在游戏窗口
- **点击窗口**：使用鼠标点击游戏窗口来触发输入模式
- **退出方式**：Ctrl+C 终止会话

#### 使用场景
- ✅ 演示 STEVE-1 的实时控制能力
- ✅ 探索不同文本提示的效果
- ✅ 制作交互式演示视频
- ✅ 调试模型响应

---

## 阶段 3：模型训练

### 3.1 `3_train.sh` - 训练 STEVE-1 主模型

#### 用途
使用准备好的数据集训练或微调 STEVE-1 模型，支持从预训练权重继续训练。

#### 功能说明
- 使用 Hugging Face Accelerate 进行分布式训练
- 支持混合精度训练（bf16）
- 自动检查点保存与恢复
- 定期验证和权重快照

#### 前置要求
必须先运行：
1. `1_generate_dataset.sh` - 准备训练数据
2. `2_create_sampling.sh` - 生成采样配置

#### 使用方法

**首次训练**：
```bash
cd src/training/steve1
bash 3_train.sh
```

**从检查点恢复训练**：
```bash
# 脚本会自动检测并恢复
# 如需从头开始，删除检查点目录：
rm -rf data/training_checkpoint/
bash 3_train.sh
```

#### 关键参数详解

**模型权重**：
```bash
--in_model data/weights/vpt/2x.model                           # VPT 架构定义
--in_weights data/weights/vpt/rl-from-foundation-2x.weights    # 初始权重
--out_weights data/weights/steve1/trained_with_script.weights  # 输出权重
```

**训练超参数**：
```bash
--T 640                        # 扩散步数（更高 = 更好质量，更慢）
--trunc_t 64                   # 截断步数（训练加速技巧）
--batch_size 12                # 每 GPU 批量大小（根据显存调整）
--gradient_accumulation_steps 4 # 梯度累积（有效批量 = 12 × 4 = 48）
--num_workers 4                # 数据加载并行数
--learning_rate 4e-5           # 学习率
--weight_decay 0.039428        # 权重衰减
```

**训练时长**：
```bash
--n_frames 100_000_000    # 总训练帧数（1 亿帧 ≈ 数天训练）
--warmup_frames 10_000_000 # 学习率预热帧数
```

**条件训练**：
```bash
--p_uncond 0.1            # 无条件训练概率（10% 用于 classifier-free guidance）
--min_btwn_goals 15       # 目标条件最小间隔帧
--max_btwn_goals 200      # 目标条件最大间隔帧
```

**验证与保存**：
```bash
--val_freq 1000                  # 每 1000 步验证一次
--val_freq_begin 100             # 前期每 100 步验证
--val_freq_switch_steps 500      # 500 步后切换到标准频率
--save_each_val False            # 是否每次验证都保存权重
--snapshot_every_n_frames 50_000_000  # 每 5000 万帧保存快照
```

**数据采样**：
```bash
--sampling neurips                       # 使用的采样配置名称
--sampling_dir data/samplings/           # 采样配置目录
```

#### 硬件需求
- **GPU**：至少 1 块 24GB+ 显存 GPU（如 RTX 3090/4090, A5000）
- **内存**：32GB+ 系统内存
- **磁盘**：100GB+ 用于检查点和数据集
- **估计时间**：完整训练约 3-7 天（取决于硬件）

#### 输出位置
- 训练权重：`data/weights/steve1/trained_with_script.weights`
- 检查点：`data/training_checkpoint/`
- TensorBoard 日志：`data/training_checkpoint/logs/`

#### 监控训练进度

**使用 TensorBoard**：
```bash
tensorboard --logdir data/training_checkpoint/logs/
# 浏览器打开 http://localhost:6006
```

**关键指标**：
- `loss/train` - 训练损失（应持续下降）
- `loss/val` - 验证损失（监控过拟合）
- `lr` - 学习率曲线

#### 调优建议

**显存不足**：
```bash
--batch_size 6                   # 减小批量大小
--gradient_accumulation_steps 8  # 增加梯度累积
```

**加快训练**：
```bash
--num_workers 8         # 增加数据加载线程（如果 CPU 足够）
--trunc_t 32            # 降低截断步数（质量略降）
```

**更好质量**：
```bash
--T 1280                # 增加扩散步数（训练更慢）
--batch_size 16         # 增大批量（如果显存允许）
```

#### 注意事项
- **检查点恢复**：脚本会自动从 `checkpoint_dir` 恢复，无需手动指定
- **权重文件大小**：约 1-2GB
- **训练稳定性**：使用 bf16 混合精度提高训练稳定性
- **数据集路径**：确保采样配置中的数据集路径正确

---

## 阶段 4：Prior 训练

### 4.1 `4_train_prior.sh` - 训练 VAE Prior

#### 用途
训练 VAE（变分自编码器）Prior 模型，用于将文本嵌入映射到 STEVE-1 的潜在空间。

#### 功能说明
- 学习文本到潜在表示的映射
- 提高文本条件的质量和一致性
- 可选步骤（使用预训练 Prior 即可）

#### 前置要求
需要准备 Prior 训练数据：
- 数据文件：`data/prior_dataset/data.pkl`
- 格式：包含文本-潜在对的 pickle 文件

#### 使用方法
```bash
cd src/training/steve1
bash 4_train_prior.sh
```

#### 参数说明
```bash
--data_path data/prior_dataset/data.pkl        # 输入数据
--output_path data/weights/vae/trained_vae.pt  # 输出权重
```

#### 输出位置
- Prior 权重：`data/weights/vae/trained_vae.pt`

#### 注意事项
- **已知问题**：脚本使用相对路径，需要修正为绝对路径
- **可选步骤**：如果只使用预训练模型，可跳过此步骤
- **数据准备**：Prior 数据集通常需要单独生成或下载
- **执行时间**：约数小时（取决于数据集大小）

#### 路径修复建议
```bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../../.." && pwd )"

python "$SCRIPT_DIR/data/text_alignment/vae_pipeline/train_vae.py" \
--data_path "$PROJECT_ROOT/data/prior_dataset/data.pkl" \
--output_path "$PROJECT_ROOT/data/weights/vae/trained_vae.pt"
```

---

## 常见问题

### Q1: 脚本报错 "No such file or directory"
**原因**：部分脚本使用相对路径，可能导致路径错误。

**解决方案**：
1. 确保在 `src/training/steve1/` 目录下运行脚本
2. 或修改脚本使用绝对路径（见各脚本的"修复方案"）

---

### Q2: MineRL 随机崩溃怎么办？
**原因**：MineRL/MineDojo 环境不稳定，可能随机退出。

**解决方案**：
- `1_gen_paper_videos.sh` 已内置自动重试机制
- 其他脚本可以手动重新运行（会跳过已完成部分）

---

### Q3: 显存不足 (CUDA out of memory)
**解决方案**：
```bash
# 训练脚本调整
--batch_size 6                   # 减小批量大小
--gradient_accumulation_steps 8  # 增加梯度累积

# 推理脚本调整
--gameplay_length 500            # 减少生成长度
```

---

### Q4: 如何验证模型是否正常工作？
**快速测试流程**：
```bash
# 1. 生成短视频测试
bash 2_gen_vid_for_text_prompt.sh

# 2. 检查输出视频
ls -lh data/generated_videos/custom_text_prompt/

# 3. 播放视频确认行为
# 使用 VLC 或其他播放器打开 .mp4 文件
```

---

### Q5: 训练时数据加载很慢
**原因**：磁盘 I/O 瓶颈或数据集未缓存。

**解决方案**：
```bash
--num_workers 8          # 增加数据加载线程
# 或将数据集复制到更快的 SSD
```

---

### Q6: 离线模式失败，报 Hugging Face 网络错误
**原因**：缺少必要的 Hugging Face 模型文件。

**解决方案**：
1. 检查缓存目录：`ls data/huggingface_cache/`
2. 参考下载指南：`docs/reference/STEVE1_MODEL_DOWNLOAD_GUIDE.md`
3. 确保已下载 `clip-vit-base-patch16` 模型

---

### Q7: 如何选择合适的文本提示？
**有效提示特征**：
- ✅ 具体动作：`chop tree`, `dig dirt`, `swim in water`
- ✅ 简短明确：2-5 个单词
- ❌ 避免复杂描述：`carefully navigate through the dark cave while avoiding monsters`
- ❌ 避免抽象概念：`be creative`, `play strategically`

**推荐提示示例**：
```
基础动作：walk, run, jump, swim, dig, chop
交互：open door, use crafting table, place block
探索：explore cave, climb mountain, find village
建造：build house, craft tools, make shelter
```

---

### Q8: 训练完成后如何使用新权重？
**替换推理脚本中的权重路径**：
```bash
# 例如修改 2_gen_vid_for_text_prompt.sh
--in_weights "$PROJECT_ROOT/data/weights/steve1/trained_with_script.weights"
```

---

## 🚀 推荐工作流

### 新手入门流程
```bash
# 1. 快速测试预训练模型（5 分钟）
bash 2_gen_vid_for_text_prompt.sh

# 2. 交互式体验（10 分钟）
bash 3_run_interactive_session.sh

# 3. 生成演示视频（30 分钟）
bash 1_gen_paper_videos.sh
```

### 完整训练流程
```bash
# 1. 数据准备（数小时）
bash 1_generate_dataset.sh
bash 2_create_sampling.sh

# 2. 开始训练（数天）
bash 3_train.sh

# 3. 测试新模型（30 分钟）
# 修改推理脚本使用新权重
bash 2_gen_vid_for_text_prompt.sh
```

### 调试与优化流程
```bash
# 1. 生成基线视频
bash 2_gen_vid_for_text_prompt.sh

# 2. 调整超参数
# 编辑 1_gen_paper_videos.sh 或 3_train.sh

# 3. 对比效果
# 比较不同参数下的视频质量
```

---

## 📚 相关文档

- **模型下载**：`docs/reference/STEVE1_MODEL_DOWNLOAD_GUIDE.md`
- **快速开始**：`docs/guides/STEVE1_QUICKSTART.md`
- **评估指南**：`docs/guides/STEVE1_EVALUATION_GUIDE.md`
- **路径配置**：`docs/guides/STEVE1_PATH_CONFIGURATION_GUIDE.md`
- **渲染功能**：`docs/guides/STEVE1_RENDER_FEATURE.md`

---

## 📝 脚本维护建议

### 建议修复的问题
1. **路径标准化**：将 `2_create_sampling.sh` 和 `4_train_prior.sh` 改用绝对路径
2. **错误处理**：为所有脚本添加错误检查和友好提示
3. **参数验证**：在脚本开始时验证必需文件是否存在
4. **日志输出**：将输出重定向到 `logs/steve1/` 目录

### 示例改进模板
```bash
#!/bin/bash
set -e  # 遇错即停

# 路径设置
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../../.." && pwd )"

# 参数验证
if [ ! -f "$PROJECT_ROOT/data/weights/steve1/steve1.weights" ]; then
    echo "错误: 权重文件不存在，请先下载模型"
    exit 1
fi

# 日志配置
LOG_DIR="$PROJECT_ROOT/logs/steve1"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/$(basename $0 .sh)_$(date +%Y%m%d_%H%M%S).log"

# 执行命令（带日志）
python ... 2>&1 | tee "$LOG_FILE"
```

---

**文档版本**: 1.0  
**最后更新**: 2025-10-31  
**维护者**: AIMC 项目组

