# AutoDL 云GPU开发机部署指南

> **适用场景**: AutoDL、阿里云PAI-DSW、腾讯云等云GPU平台

## 目录
- [1. 快速部署（5分钟）](#1-快速部署5分钟)
- [2. 详细步骤](#2-详细步骤)
- [3. 常见问题](#3-常见问题)
- [4. 性能优化](#4-性能优化)

---

## 1. 快速部署（5分钟）

### 方式一：一键部署脚本（推荐）⭐

```bash
# 1. 克隆项目
cd ~
git clone https://github.com/your-repo/aimc.git
cd aimc

# 2. 运行一键部署脚本
bash scripts/autodl_setup.sh
```

### 方式二：手动部署

```bash
# 1. 激活 conda 环境
conda activate minedojo-x86  # 或你的环境名

# 2. 配置 pip 国内镜像（加速下载）
mkdir -p ~/.pip
cat > ~/.pip/pip.conf << 'EOF'
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple

[install]
trusted-host = pypi.tuna.tsinghua.edu.cn
EOF

# 3. 安装 PyTorch（根据你的 CUDA 版本选择）
# 查看 CUDA 版本
nvidia-smi  # 查看 CUDA Version 行

# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 或 CUDA 12.1
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 4. 验证 PyTorch 安装
python -c "import torch; print('PyTorch版本:', torch.__version__); print('CUDA可用:', torch.cuda.is_available()); print('GPU数量:', torch.cuda.device_count())"

# 5. 安装项目依赖
cd ~/aimc
pip install -r requirements.txt

# 6. 验证 MineDojo 安装
python tools/validate_install.py
```

---

## 2. 详细步骤

### 步骤 1：检查系统环境

```bash
# 检查 Python 版本（需要 3.9+）
python --version

# 检查 CUDA 版本
nvidia-smi

# 检查 Java 版本（MineDojo 需要）
java -version

# 如果 Java 未安装，安装 OpenJDK 8
sudo apt-get update
sudo apt-get install -y openjdk-8-jdk

# 设置 JAVA_HOME
export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64
echo 'export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64' >> ~/.bashrc
```

### 步骤 2：配置 Conda 环境

```bash
# 创建或激活 conda 环境
conda create -n minedojo python=3.9 -y
conda activate minedojo

# 将激活命令添加到 .bashrc（自动激活）
echo 'conda activate minedojo' >> ~/.bashrc
```

### 步骤 3：配置国内镜像（加速下载）

```bash
# 配置 pip 镜像（清华源）
mkdir -p ~/.pip
cat > ~/.pip/pip.conf << 'EOF'
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple

[install]
trusted-host = pypi.tuna.tsinghua.edu.cn
EOF

# 或使用阿里云镜像
# index-url = https://mirrors.aliyun.com/pypi/simple/
# trusted-host = mirrors.aliyun.com
```

### 步骤 4：安装 PyTorch（关键步骤）⭐

**重要**：不要使用 `conda install torch`，使用 pip 安装！

```bash
# 方式 1：自动检测 CUDA 版本并安装（推荐）
pip install torch torchvision torchaudio

# 方式 2：指定 CUDA 版本（更可靠）
# 先查看 CUDA 版本
nvidia-smi | grep "CUDA Version"

# 根据 CUDA 版本选择：
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 方式 3：使用国内镜像加速（如果 PyTorch 官方源慢）
pip install torch torchvision torchaudio -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**验证 PyTorch 安装**：

```bash
python << 'EOF'
import torch
print("=" * 50)
print("PyTorch 安装验证")
print("=" * 50)
print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 可用: {torch.cuda.is_available()}")
print(f"CUDA 版本: {torch.version.cuda}")
print(f"GPU 数量: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU 型号: {torch.cuda.get_device_name(0)}")
    print(f"GPU 内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
print("=" * 50)
EOF
```

**预期输出**：
```
==================================================
PyTorch 安装验证
==================================================
PyTorch 版本: 2.1.0+cu118
CUDA 可用: True
CUDA 版本: 11.8
GPU 数量: 1
GPU 型号: NVIDIA GeForce RTX 3090
GPU 内存: 24.0 GB
==================================================
```

### 步骤 5：安装 MineDojo 和项目依赖

```bash
# 克隆项目（如果还没有）
cd ~
git clone https://github.com/your-repo/aimc.git
cd aimc

# 安装旧版本的构建工具（MineDojo 需要）
pip install "pip<24.1" "setuptools<58" "wheel<0.38.0"

# 安装 NumPy（必须 < 2.0）
pip install "numpy>=1.21.0,<2.0"

# 安装 MineDojo
pip install minedojo

# 安装项目依赖
pip install -r requirements.txt
```

### 步骤 6：配置环境变量

```bash
# 设置无头模式（提升 20-40% 性能）
export JAVA_OPTS="-Djava.awt.headless=true"

# 添加到 .bashrc（永久生效）
echo 'export JAVA_OPTS="-Djava.awt.headless=true"' >> ~/.bashrc
```

### 步骤 7：验证安装

```bash
# 运行验证脚本
python tools/validate_install.py

# 预期输出：
# ✓ Python 版本正确: 3.9.x
# ✓ MineDojo 已安装
# ✓ PyTorch 已安装: 2.1.0+cu118
# ✓ CUDA 可用: True
# ✓ Java 可用: 1.8.0_xxx
# ✓ 环境创建成功
```

### 步骤 8：快速测试

```bash
# 运行 5 分钟快速测试
python src/training/train_get_wood.py \
    --timesteps 10000 \
    --use-mineclip \
    --device cuda

# 预期输出：
# 创建环境: harvest_1_log
# MineCLIP: 启用
# 设备: cuda
# [100步] ep_rew_mean: 0.05
# ...
```

---

## 3. 常见问题

### ❌ 问题 1：`conda install torch` 失败

**错误信息**：
```
PackagesNotFoundError: The following packages are not available from current channels:
  - torch
```

**原因**：清华 conda 镜像中没有 PyTorch 包（或版本不全）。

**解决方案**：**使用 pip 安装，不要使用 conda**！

```bash
# ✅ 正确方式
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# ❌ 错误方式
conda install torch  # 不要这样做！
```

---

### ❌ 问题 2：PyTorch 安装后 CUDA 不可用

**症状**：`torch.cuda.is_available()` 返回 `False`

**原因**：安装了 CPU 版本的 PyTorch，或 CUDA 版本不匹配。

**解决方案**：

```bash
# 1. 卸载现有 PyTorch
pip uninstall torch torchvision torchaudio

# 2. 检查 CUDA 版本
nvidia-smi | grep "CUDA Version"

# 3. 安装匹配的 PyTorch
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 4. 验证
python -c "import torch; print('CUDA可用:', torch.cuda.is_available())"
```

---

### ❌ 问题 3：MineDojo 安装失败

**错误信息**：
```
ERROR: No matching distribution found for minedojo
```

**解决方案**：

```bash
# 1. 升级 pip
pip install --upgrade pip

# 2. 安装旧版本构建工具
pip install "pip<24.1" "setuptools<58" "wheel<0.38.0"

# 3. 安装 NumPy < 2.0
pip install "numpy>=1.21.0,<2.0"

# 4. 重新安装 MineDojo
pip install minedojo
```

---

### ❌ 问题 4：环境创建失败

**错误信息**：
```
Exception: Could not find or load main class net.minecraft.launchwrapper.Launch
```

**原因**：Java 未正确配置或 Minecraft 编译失败。

**解决方案**：

```bash
# 1. 检查 Java
java -version

# 2. 安装 OpenJDK 8（如果未安装）
sudo apt-get install -y openjdk-8-jdk

# 3. 设置 JAVA_HOME
export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64
echo 'export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64' >> ~/.bashrc

# 4. 重新安装 MineDojo
pip uninstall minedojo
pip install minedojo

# 5. 验证
python -c "import minedojo; env = minedojo.make('harvest_1_log', image_size=(160, 256)); env.close(); print('✓ 环境创建成功')"
```

---

### ❌ 问题 5：内存不足 (OOM)

**错误信息**：
```
RuntimeError: CUDA out of memory
```

**解决方案**：

```bash
# 1. 减少并行环境数量
--n-envs 1

# 2. 减少批次大小
--batch-size 32

# 3. 使用更小的图像
--image-size 120 160

# 4. 检查 GPU 内存使用
nvidia-smi
```

---

### ❌ 问题 6：网络连接问题

**症状**：下载 PyTorch 或 MineDojo 时超时或失败。

**解决方案**：

```bash
# 方式 1：使用国内镜像
pip install torch torchvision torchaudio -i https://pypi.tuna.tsinghua.edu.cn/simple

# 方式 2：增加超时时间
pip install --timeout=1000 torch torchvision torchaudio

# 方式 3：使用 AutoDL 的代理（如果有）
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890
```

---

## 4. 性能优化

### 优化 1：启用混合精度训练

```bash
# 使用 FP16 混合精度（节省显存，加速 1.5-2x）
python src/training/train_get_wood.py \
    --use-mineclip \
    --device cuda \
    --fp16  # 如果支持
```

### 优化 2：增加并行环境

```bash
# 使用 4-8 个并行环境（需要更多显存）
python src/training/train_get_wood.py \
    --use-mineclip \
    --device cuda \
    --n-envs 4
```

### 优化 3：启用无头模式

```bash
# 无头模式可提升 20-40% 性能
export JAVA_OPTS="-Djava.awt.headless=true"
echo 'export JAVA_OPTS="-Djava.awt.headless=true"' >> ~/.bashrc
```

### 优化 4：使用 TensorBoard 监控

```bash
# 终端 1：启动训练
python src/training/train_get_wood.py --use-mineclip --device cuda

# 终端 2：启动 TensorBoard（在 AutoDL 上需要映射端口）
tensorboard --logdir logs/tensorboard --host 0.0.0.0 --port 6006

# 访问：http://你的实例IP:6006
# 或在 AutoDL 面板中查看自定义服务
```

---

## 5. AutoDL 特定配置

### AutoDL 端口映射

```bash
# 在 AutoDL 容器内启动 TensorBoard
tensorboard --logdir logs/tensorboard --host 0.0.0.0 --port 6006

# 在 AutoDL 面板中：
# 1. 进入"自定义服务"
# 2. 添加服务：端口 6006
# 3. 获取访问链接
```

### AutoDL 数据持久化

```bash
# AutoDL 的持久化目录
cd /root/autodl-tmp  # 数据持久化目录

# 克隆项目到持久化目录
git clone https://github.com/your-repo/aimc.git
cd aimc

# 将检查点和日志保存到持久化目录
python src/training/train_get_wood.py \
    --checkpoint-dir /root/autodl-tmp/checkpoints \
    --log-dir /root/autodl-tmp/logs
```

### AutoDL 性能基准

**RTX 3090 (24GB VRAM)**:
- 无头模式 + MineCLIP + CUDA: ~1200 步/分钟
- 200K 步训练: 约 1-1.5 小时
- 推荐配置: `--n-envs 4 --batch-size 128`

**RTX 4090 (24GB VRAM)**:
- 无头模式 + MineCLIP + CUDA: ~1800 步/分钟
- 200K 步训练: 约 1 小时
- 推荐配置: `--n-envs 8 --batch-size 256`

**A100 (40GB VRAM)**:
- 无头模式 + MineCLIP + CUDA: ~2500 步/分钟
- 200K 步训练: 约 45 分钟
- 推荐配置: `--n-envs 16 --batch-size 512`

---

## 6. 完整部署检查清单

### 安装前检查

- [ ] Python 3.9+ 已安装
- [ ] CUDA 驱动已安装（`nvidia-smi` 可用）
- [ ] Java 8+ 已安装
- [ ] 至少 20GB 可用磁盘空间

### 安装步骤

- [ ] 创建 conda 环境
- [ ] 配置 pip 国内镜像
- [ ] 使用 pip 安装 PyTorch（**不要用 conda**）
- [ ] 验证 PyTorch CUDA 可用
- [ ] 安装 MineDojo
- [ ] 安装项目依赖
- [ ] 设置环境变量（JAVA_HOME, JAVA_OPTS）

### 验证步骤

- [ ] `python -c "import torch; print(torch.cuda.is_available())"` 返回 `True`
- [ ] `python tools/validate_install.py` 全部通过
- [ ] 快速测试训练 10K 步成功

---

## 7. 快速命令速查

```bash
# 检查 CUDA
nvidia-smi

# 安装 PyTorch (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 验证 GPU
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

# 安装 MineDojo
pip install "numpy<2.0" minedojo

# 安装项目依赖
pip install -r requirements.txt

# 验证安装
python tools/validate_install.py

# 快速测试
python src/training/train_get_wood.py --timesteps 10000 --use-mineclip --device cuda

# 开始训练
python src/training/train_get_wood.py --use-mineclip --device cuda
```

---

## 8. 联系支持

如果遇到问题：
1. 查看 [常见问题 FAQ](../FAQ.md)
2. 查看 [GitHub Issues](https://github.com/your-repo/aimc/issues)
3. 发送日志到：konders@gmail.com

---

**祝部署成功！开始训练吧 🚀**

