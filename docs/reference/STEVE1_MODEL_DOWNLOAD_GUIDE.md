# STEVE-1 模型下载指南

> **项目**: [STEVE-1: A Generative Model for Text-to-Behavior in Minecraft](https://github.com/Shalev-Lifshitz/STEVE-1)  
> **论文**: [arXiv:2306.00937](https://arxiv.org/abs/2306.00937)  
> **项目主页**: [sites.google.com/view/steve-1](https://sites.google.com/view/steve-1)

---

## 📦 模型权重获取方式

### ✅ **方法 1: 使用官方下载脚本（推荐）**

STEVE-1 项目在 GitHub 仓库中提供了 `download_weights.sh` 脚本，用于自动下载模型权重。

#### **步骤：**

```bash
# 1. 克隆 STEVE-1 仓库
cd /Users/nanzhang/aimc
git clone https://github.com/Shalev-Lifshitz/STEVE-1.git
cd STEVE-1

# 2. 安装依赖（如果需要）
pip install gdown

# 3. 运行下载脚本
chmod +x download_weights.sh
./download_weights.sh
```

#### **脚本说明：**

`download_weights.sh` 脚本会自动：
- 下载预训练的 VPT 模型权重
- 下载 STEVE-1 微调后的权重
- 下载 CVAE Prior 模型权重
- 下载训练数据集（如果需要）

**预计下载大小**：约 2-5 GB

---

### ⚠️ **Hugging Face 状态**

**结论**：目前 **Hugging Face 上没有 STEVE-1 的官方模型**。

根据搜索结果：
- ❌ Hugging Face Model Hub 未找到 `STEVE-1` 或 `steve1` 相关模型
- ❌ 作者未上传到 Hugging Face Spaces
- ✅ 唯一官方渠道是 GitHub 仓库

**原因分析**：
1. STEVE-1 基于 VPT + MineCLIP，模型结构复杂
2. 依赖 MineRL/MineDojo 环境，不便于 Hugging Face 部署
3. 作者选择通过 Google Drive + GitHub 分发

---

## 📂 模型文件结构

下载完成后，您应该会看到以下结构：

```
STEVE-1/
├── data/
│   ├── weights/
│   │   ├── vpt/                    # VPT 基础模型
│   │   │   ├── foundation-model-1x.model
│   │   │   └── foundation-model-1x.weights
│   │   ├── steve1/                 # STEVE-1 微调模型
│   │   │   └── steve1_weights.pt
│   │   └── prior/                  # CVAE Prior 模型
│   │       └── prior_weights.pt
│   └── datasets/                   # 训练数据集（可选）
└── download_weights.sh
```

---

## 🔍 模型权重详情

### **1. VPT 基础模型**

STEVE-1 基于 OpenAI 的 VPT（Video Pre-Training）模型。

**文件**：
- `foundation-model-1x.model` - 模型架构配置
- `foundation-model-1x.weights` - 预训练权重

**来源**：OpenAI VPT 官方发布  
**大小**：约 1.5 GB  
**用途**：作为 STEVE-1 的基础策略网络

**直接下载链接**（VPT 官方）：
```bash
# 如果 download_weights.sh 失败，可以手动下载
wget https://openaipublic.blob.core.windows.net/vpt/models/foundation-model-1x.model
wget https://openaipublic.blob.core.windows.net/vpt/models/foundation-model-1x.weights
```

---

### **2. STEVE-1 微调权重**

经过指令调优的 VPT 模型，能够理解文本和视觉指令。

**文件**：`steve1_weights.pt`  
**大小**：约 1-2 GB  
**训练方式**：
- 第一阶段：适配 VPT 到 MineCLIP 潜在空间
- 第二阶段：行为克隆 + 事后重标记（Hindsight Relabeling）

**特性**：
- ✅ 支持短期文本指令（"chop tree", "hunt cow"）
- ✅ 支持视觉指令（MineCLIP 图像编码）
- ✅ 在 12/13 早期游戏任务中成功

---

### **3. CVAE Prior 模型**

条件变分自编码器（CVAE），用于从文本生成 MineCLIP 潜在编码。

**文件**：`prior_weights.pt`  
**大小**：约 500 MB  
**用途**：将文本指令 → MineCLIP 潜在编码 → STEVE-1 行为

**工作流程**：
```
文本 "chop tree" 
  → CVAE Prior 
  → MineCLIP Latent Code [512维]
  → STEVE-1 Policy
  → Minecraft Actions
```

---

## 🚀 使用示例

### **运行 STEVE-1 Agent**

```bash
cd STEVE-1

# 1. 生成论文中的视频
./run_agent/1_gen_paper_videos.sh

# 2. 测试自定义文本指令
./run_agent/2_gen_vid_for_text_prompt.sh

# 3. 交互式会话（需要图形界面）
./run_agent/3_run_interactive_session.sh
```

### **Python 代码示例**

```python
from steve1 import STEVE1Agent

# 加载模型
agent = STEVE1Agent(
    vpt_weights="data/weights/vpt/foundation-model-1x.weights",
    steve1_weights="data/weights/steve1/steve1_weights.pt",
    prior_weights="data/weights/prior/prior_weights.pt"
)

# 使用文本指令
obs = env.reset()
for _ in range(1000):
    # 根据文本指令生成动作
    action = agent.predict(obs, text_prompt="chop tree")
    obs, reward, done, info = env.step(action)
    
    if done:
        break
```

---

## 🔧 训练自己的 STEVE-1

如果您想从头训练 STEVE-1（或在您的项目中集成）：

```bash
# 1. 生成游戏数据集
./train/1_generate_dataset.sh

# 2. 创建训练/验证分割
./train/2_create_sampling.sh

# 3. 训练 STEVE-1（适配 MineCLIP）
./train/3_train.sh

# 4. 训练 CVAE Prior
./train/4_train_prior.sh
```

**训练成本**：论文中提到仅需 **$60 计算费用**（使用预训练模型）

---

## 📊 STEVE-1 vs MineCLIP vs VPT 对比

| 特性 | VPT | MineCLIP | STEVE-1 |
|------|-----|----------|---------|
| **输入** | 视觉 | 视觉 + 文本 | 视觉 + 文本 |
| **输出** | 动作 | 相似度 | 动作 |
| **训练方式** | 行为克隆 | 对比学习 | VPT 微调 + Prior |
| **指令理解** | ❌ 无 | ✅ 理解但不执行 | ✅ 理解并执行 |
| **零样本迁移** | ❌ 无 | ✅ 有（评估） | ✅ 有（执行） |
| **训练成本** | 高（数百万数据） | 高（280万视频） | **低（$60）** |

**核心创新**：STEVE-1 结合了 VPT 的动作能力和 MineCLIP 的语言理解能力。

---

## 🔬 STEVE-1 的技术原理

### **训练流程**

```
阶段 1: 适配 MineCLIP 潜在空间
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
VPT 模型
  ↓
添加 MineCLIP 条件输入
  ↓
自监督行为克隆 + 事后重标记
  ↓
STEVE-1 基础模型（能理解 MineCLIP 编码）

阶段 2: 训练文本 → 潜在编码的 Prior
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
文本 "chop tree"
  ↓
CVAE Prior 模型
  ↓
MineCLIP Latent Code
  ↓
STEVE-1 执行
```

### **推理流程**

推理时:
━━━━━
文本指令 → Prior → MineCLIP编码 → STEVE-1 → 动作

```python
# 用户输入文本
text_prompt = "chop tree"

# 1. Prior 生成潜在编码
latent_code = prior_model.encode(text_prompt)  # [512]

# 2. STEVE-1 根据编码和观察生成动作
action = steve1_policy(
    visual_obs=current_frame,
    latent_goal=latent_code
)

# 3. 执行动作
env.step(action)
```

---

## 💡 与您的 AIMC 项目集成

### **方案 1: 直接使用 STEVE-1**

```python
# 在您的项目中集成 STEVE-1
from steve1 import STEVE1Agent
from minedojo import MinedojoEnv

env = MinedojoEnv(task_id="harvest_1_log")
agent = STEVE1Agent(weights_path="...")

# 用文本控制 Agent
obs = env.reset()
agent.play(env, instruction="chop tree with hand")
```

**优点**：
- ✅ 直接支持文本指令
- ✅ 已经训练好，无需重新训练
- ✅ 12/13 早期任务高成功率

**缺点**：
- ⚠️ 模型较大（3-4 GB）
- ⚠️ 依赖 VPT 架构
- ⚠️ 仅支持短期指令

---

### **方案 2: 借鉴 STEVE-1 的方法论**

**核心思想**：使用 MineCLIP 作为中间表示，避免大量文本标注。

```python
# 您可以实现类似的训练流程
# 1. 使用 VPT 作为基础策略
# 2. 添加 MineCLIP 条件
# 3. 事后重标记（用 MineCLIP 自动标注轨迹）
# 4. 训练 Prior（文本 → MineCLIP）
```

**适用场景**：
- 您想训练更长期的任务（如制作铁镐）
- 您有自己的游戏数据
- 您想定制模型架构

---

## 🎯 STEVE-1 vs 您的 MineCLIP 方法

| 方面 | 您的方法 | STEVE-1 |
|------|---------|---------|
| **奖励信号** | MineCLIP 相似度差值 | MineCLIP 潜在编码条件 |
| **训练方式** | 在线 RL（PPO） | 离线 BC + 事后重标记 |
| **基础模型** | 从头训练 | 基于 VPT |
| **数据需求** | 环境交互 | 预录游戏视频 |
| **优势** | 适应在线学习 | 训练成本低（$60） |
| **适用任务** | 单任务深度优化 | 多任务零样本迁移 |

**建议**：
- **如果目标是单个任务高性能**：您的 MineCLIP 奖励方法更合适
- **如果目标是多任务快速原型**：STEVE-1 更合适

---

## 📚 相关资源

### **官方链接**

- **GitHub 仓库**: https://github.com/Shalev-Lifshitz/STEVE-1
- **论文**: https://arxiv.org/abs/2306.00937
- **项目主页**: https://sites.google.com/view/steve-1

### **依赖模型**

- **VPT**: https://github.com/openai/Video-Pre-Training
- **MineCLIP**: https://github.com/MineDojo/MineCLIP
- **MineDojo**: https://minedojo.org

### **论文引用**

```bibtex
@article{lifshitz2023steve1,
  title={STEVE-1: A Generative Model for Text-to-Behavior in Minecraft}, 
  author={Shalev Lifshitz and Keiran Paster and Harris Chan and Jimmy Ba and Sheila McIlraith},
  year={2023},
  eprint={2306.00937},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}
```

---

## ⚡ 快速开始命令

```bash
# 一键下载并运行 STEVE-1
cd /Users/nanzhang/aimc
git clone https://github.com/Shalev-Lifshitz/STEVE-1.git
cd STEVE-1

# 安装依赖
pip install gdown tqdm accelerate==0.18.0 wandb
pip install minedojo git+https://github.com/MineDojo/MineCLIP
pip install git+https://github.com/minerllabs/minerl@v1.0.1
pip install gym==0.19 gym3 attrs opencv-python
pip install -e .

# 下载模型权重
chmod +x download_weights.sh
./download_weights.sh

# 测试运行
./run_agent/1_gen_paper_videos.sh
```

---

## ❓ 常见问题

### **Q1: 下载脚本失败怎么办？**

```bash
# 如果 gdown 失败，可能是网络问题
# 方案 1: 使用代理
export http_proxy=http://your-proxy:port
export https_proxy=http://your-proxy:port

# 方案 2: 手动从论文项目页面下载
# 访问 https://sites.google.com/view/steve-1
# 查找 "Downloads" 或 "Resources" 部分
```

### **Q2: 模型可以用于商业项目吗？**

**答**：需要检查许可证。STEVE-1 基于：
- VPT（OpenAI 发布，需查看许可）
- MineCLIP（学术研究许可）
- Minecraft™（微软知识产权）

**建议**：用于学术研究。商业用途需联系作者。

### **Q3: STEVE-1 能在我的项目中使用吗？**

**兼容性检查**：
- ✅ 您使用 MineDojo/MineRL 环境
- ✅ 您需要短期指令执行（<1分钟任务）
- ✅ 您的任务是早期游戏内容
- ⚠️ 您需要长期规划（STEVE-1 较弱）
- ⚠️ 您需要精确控制（STEVE-1 是生成模型）

---

## 🔍 下一步

1. **立即尝试**：
   ```bash
   cd /Users/nanzhang/aimc
   git clone https://github.com/Shalev-Lifshitz/STEVE-1.git
   cd STEVE-1 && ./download_weights.sh
   ```

2. **阅读论文**：理解 unCLIP 方法论和事后重标记技术

3. **集成到 AIMC**：评估是否适合您的任务需求

4. **对比测试**：STEVE-1 vs 您的 MineCLIP 奖励方法

---

**总结**：STEVE-1 的模型权重通过 GitHub 仓库的 `download_weights.sh` 脚本获取，目前不在 Hugging Face 上。这是一个基于 VPT + MineCLIP 的创新方法，特别适合需要文本指令控制的 Minecraft AI 任务。


