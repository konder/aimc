# AIMC 部署指南

本文档介绍如何在不同平台上部署 AIMC 项目。

## 目录

- [方案 A：Docker 部署（推荐）](#方案-a-docker-部署推荐)
- [方案 B：Linux 本地部署](#方案-b-linux-本地部署)
- [方案 C：macOS 本地部署](#方案-c-macos-本地部署)
- [模型权重下载](#模型权重下载)
- [验证安装](#验证安装)
- [常见问题](#常见问题)

---

## 方案 A：Docker 部署（推荐）

Docker 部署是最简单的方式，已包含所有依赖配置。

### 前置要求

- Docker 20.10+
- 至少 16GB 内存
- 至少 30GB 磁盘空间

### 构建镜像

```bash
# 克隆项目
git clone https://github.com/your-repo/aimc.git
cd aimc

# 构建 Docker 镜像（首次构建约 20-30 分钟）
docker build -t aimc:latest .
```

### 运行容器

```bash
# 基础运行（交互模式）
docker run -it --rm \
    -v $(pwd):/workspace \
    aimc:latest

# 带 GPU 支持（NVIDIA）
docker run -it --rm \
    --gpus all \
    -v $(pwd):/workspace \
    aimc:latest

# 带显示支持（用于渲染）
docker run -it --rm \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $(pwd):/workspace \
    aimc:latest
```

### 在容器中运行评估

```bash
# 容器内已自动激活 minedojo-x86 环境
cd /workspace

# 运行评估
scripts/run_evaluation.sh --task harvest_1_log --n-trials 1
```

---

## 方案 B：Linux 本地部署

### 系统要求

- Ubuntu 20.04 / 22.04 (推荐)
- Java JDK 8
- Python 3.9
- CUDA 11.x+ (可选，用于 GPU 加速)

### 1. 安装系统依赖

```bash
# 更新系统
sudo apt update && sudo apt upgrade -y

# 安装 Java 8
sudo apt install -y openjdk-8-jdk
export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64

# 安装其他依赖
sudo apt install -y \
    build-essential \
    git \
    wget \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    xvfb
```

### 2. 安装 Miniconda

```bash
# 下载 Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# 安装
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
rm Miniconda3-latest-Linux-x86_64.sh

# 初始化
$HOME/miniconda3/bin/conda init bash
source ~/.bashrc
```

### 3. 创建 Python 环境

```bash
# 创建环境
conda create -n minedojo-x86 python=3.9 -y
conda activate minedojo-x86

# 安装基础工具
pip install "pip<24.1" "setuptools<58" "wheel<0.38.0"
```

### 4. 安装 MineDojo

```bash
# 安装 MineDojo
pip install minedojo

# 修复 gym 版本
pip install --force-reinstall gym==0.21.0
```

### 5. 安装 MineRL（可选）

```bash
# 克隆 MineRL
cd /tmp
git clone https://github.com/minerllabs/minerl.git
cd minerl
git checkout v1.0.0

# 安装
pip install .
```

### 6. 安装 MineCLIP 和 STEVE-1

```bash
# 安装 MineCLIP
pip install git+https://github.com/MineDojo/MineCLIP

# 安装 STEVE-1
cd /opt
git clone https://github.com/Shalev-Lifshitz/STEVE-1.git steve1
cd steve1
pip install -e .
```

### 7. 安装项目依赖

```bash
# 回到项目目录
cd /path/to/aimc

# 安装依赖
pip install -r requirements.txt

# 修复版本冲突
pip install numpy==1.24.3 gym==0.21.0
```

### 8. 环境变量配置

```bash
# 添加到 ~/.bashrc
echo 'export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64' >> ~/.bashrc
echo 'export MINEDOJO_HEADLESS=1' >> ~/.bashrc
source ~/.bashrc
```

---

## 方案 C：macOS 本地部署

macOS 部署需要使用 x86_64 模式（即使在 Apple Silicon 上）。

### 系统要求

- macOS 12+ (Monterey 或更高)
- Homebrew
- Rosetta 2 (Apple Silicon)

### 1. 安装 Rosetta 2（Apple Silicon）

```bash
# 仅 Apple Silicon Mac 需要
softwareupdate --install-rosetta --agree-to-license
```

### 2. 安装系统依赖

```bash
# 安装 Homebrew（如果未安装）
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 安装 Java 8
brew install openjdk@8

# 配置 Java
sudo ln -sfn /opt/homebrew/opt/openjdk@8/libexec/openjdk.jdk /Library/Java/JavaVirtualMachines/openjdk-8.jdk
export JAVA_HOME=$(/usr/libexec/java_home -v 1.8)
```

### 3. 安装 Miniforge（x86_64 版本）

```bash
# 下载 x86_64 版本的 Miniforge
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-x86_64.sh

# 安装到指定目录
bash Miniforge3-MacOSX-x86_64.sh -b -p $HOME/miniforge-x86
rm Miniforge3-MacOSX-x86_64.sh

# 初始化
$HOME/miniforge-x86/bin/conda init zsh
source ~/.zshrc
```

### 4. 创建 x86_64 Python 环境

```bash
# 强制 x86_64 架构
arch -x86_64 /bin/zsh

# 创建环境
conda create -n minedojo-x86 python=3.9 -y
conda activate minedojo-x86
```

### 5. 安装依赖

```bash
# 保持在 x86_64 模式下
arch -x86_64 /bin/zsh
conda activate minedojo-x86

# 安装基础工具
pip install "pip<24.1" "setuptools<58" "wheel<0.38.0"

# 安装 MineDojo
pip install minedojo
pip install --force-reinstall gym==0.21.0

# 安装 MineCLIP
pip install git+https://github.com/MineDojo/MineCLIP

# 安装项目依赖
pip install -r requirements.txt
pip install numpy==1.24.3 gym==0.21.0
```

### 6. 安装 STEVE-1

```bash
cd /opt
git clone https://github.com/Shalev-Lifshitz/STEVE-1.git steve1
cd steve1
pip install -e .
```

### 7. 环境变量配置

```bash
# 添加到 ~/.zshrc
echo 'export JAVA_HOME=$(/usr/libexec/java_home -v 1.8)' >> ~/.zshrc
echo 'export MINEDOJO_HEADLESS=0' >> ~/.zshrc  # macOS 不需要 headless
source ~/.zshrc
```

### 8. 创建启动脚本

```bash
# 创建快捷启动脚本
cat > ~/start_aimc.sh << 'EOF'
#!/bin/zsh
arch -x86_64 /bin/zsh -c "
source $HOME/miniforge-x86/etc/profile.d/conda.sh
conda activate minedojo-x86
cd /path/to/aimc
exec /bin/zsh
"
EOF
chmod +x ~/start_aimc.sh
```

---

## 模型权重下载

### STEVE-1 权重

```bash
mkdir -p data/weights/steve1

# 下载 STEVE-1 权重（从 Hugging Face 或原项目）
# steve1.weights - Policy 模型权重
# steve1_prior.pt - Prior VAE 权重
# steve1_prior_goal_only_policy.pt - Goal-only Policy 权重（可选）
```

### VPT 权重

```bash
mkdir -p data/weights/vpt

# 下载 VPT 基础模型
# 2x.model - 模型架构
# 2x.weights - 模型权重
```

### MineCLIP 权重

```bash
mkdir -p data/weights/mineclip

# 下载 MineCLIP 权重
# attn.pth - 注意力模型权重
```

### 权重文件结构

```
data/weights/
├── steve1/
│   ├── steve1.weights
│   ├── steve1_prior.pt
│   └── steve1_prior_goal_only_policy.pt
├── vpt/
│   ├── 2x.model
│   └── 2x.weights
└── mineclip/
    └── attn.pth
```

---

## 验证安装

### 1. 验证 Python 环境

```bash
conda activate minedojo-x86
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"
python -c "import gym; print(f'Gym: {gym.__version__}')"
```

### 2. 验证 MineDojo

```bash
python -c "import minedojo; print('MineDojo OK')"
```

### 3. 验证 MineCLIP

```bash
python -c "from mineclip import MineCLIP; print('MineCLIP OK')"
```

### 4. 验证 STEVE-1

```bash
python -c "from steve1.VPT import lib as vpt_lib; print('STEVE-1 VPT OK')"
```

### 5. 运行完整测试

```bash
# 使用测试脚本（Docker 内置）
python docker/test_environments.py

# 或运行简单评估
scripts/run_evaluation.sh --task harvest_1_log --n-trials 1
```

---

## 常见问题

### Q: Java 版本不正确

```bash
# 检查 Java 版本
java -version

# 确保使用 Java 8
export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64  # Linux
export JAVA_HOME=$(/usr/libexec/java_home -v 1.8)    # macOS
```

### Q: Minecraft 启动失败（OpenGL 错误）

```bash
# Linux: 使用 xvfb
export MINEDOJO_HEADLESS=1

# macOS: 确保不在 headless 模式
export MINEDOJO_HEADLESS=0
```

### Q: NumPy 版本冲突

```bash
# 强制使用兼容版本
pip install numpy==1.24.3 --force-reinstall
```

### Q: Gym 版本冲突

```bash
# MineDojo/MineRL 需要 gym 0.21.0
pip install gym==0.21.0 --force-reinstall
```

### Q: macOS Apple Silicon 架构问题

```bash
# 确保在 x86_64 模式下运行
arch -x86_64 /bin/zsh
conda activate minedojo-x86
```

### Q: Docker 构建缓存

如果在企业网络环境中，需要提前准备 `docker/build_local_cache.zip`：
- 包含 Gradle 依赖缓存
- 包含 Maven 依赖缓存

---

## 下一步

- [评估框架指南](EVALUATION_FRAMEWORK_GUIDE.md) - 学习如何运行评估
- [任务配置指南](TASK_WRAPPERS_GUIDE.md) - 自定义评估任务
- [样本录制指南](VISUAL_EMBED_16FRAMES_GUIDE.md) - 录制训练数据

