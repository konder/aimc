#!/bin/bash
# AutoDL 云GPU开发机一键部署脚本
# 适用于: AutoDL, 阿里云PAI-DSW, 腾讯云等云GPU平台

set -e  # 遇到错误立即退出

echo "========================================"
echo "AIMC AutoDL 一键部署脚本"
echo "========================================"
echo ""

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 辅助函数
print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_info() {
    echo -e "ℹ $1"
}

# 1. 检查系统环境
echo ">>> 步骤 1/8: 检查系统环境"
echo ""

# 检查 Python
if command -v python &> /dev/null; then
    PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
    print_success "Python 版本: $PYTHON_VERSION"
else
    print_error "Python 未安装"
    exit 1
fi

# 检查 CUDA
if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
    GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -n 1)
    print_success "CUDA 版本: $CUDA_VERSION"
    print_success "GPU: $GPU_NAME ($GPU_MEMORY)"
else
    print_warning "CUDA 未检测到（将使用 CPU 训练）"
fi

# 检查 Java
if command -v java &> /dev/null; then
    JAVA_VERSION=$(java -version 2>&1 | awk -F '"' '/version/ {print $2}')
    print_success "Java 版本: $JAVA_VERSION"
else
    print_warning "Java 未安装，正在安装 OpenJDK 8..."
    
    # 检测系统类型
    if [ -f /etc/debian_version ]; then
        # Debian/Ubuntu
        sudo apt-get update -qq
        sudo apt-get install -y openjdk-8-jdk
    elif [ -f /etc/redhat-release ]; then
        # CentOS/RHEL
        sudo yum install -y java-1.8.0-openjdk
    else
        print_error "未知的 Linux 发行版，请手动安装 Java 8"
        exit 1
    fi
    
    # 设置 JAVA_HOME
    export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64
    echo 'export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64' >> ~/.bashrc
    print_success "Java 8 安装完成"
fi

echo ""

# 2. 创建或激活 conda 环境
echo ">>> 步骤 2/8: 配置 Conda 环境"
echo ""

CONDA_ENV_NAME="minedojo"

# 检查是否已经在 conda 环境中
if [ -n "$CONDA_DEFAULT_ENV" ]; then
    print_info "当前 Conda 环境: $CONDA_DEFAULT_ENV"
    
    # 如果不是目标环境，创建或激活目标环境
    if [ "$CONDA_DEFAULT_ENV" != "$CONDA_ENV_NAME" ]; then
        if conda env list | grep -q "^$CONDA_ENV_NAME "; then
            print_info "激活现有环境: $CONDA_ENV_NAME"
            eval "$(conda shell.bash hook)"
            conda activate $CONDA_ENV_NAME
        else
            print_info "创建新环境: $CONDA_ENV_NAME"
            conda create -n $CONDA_ENV_NAME python=3.9 -y
            eval "$(conda shell.bash hook)"
            conda activate $CONDA_ENV_NAME
        fi
    fi
else
    print_warning "未检测到 Conda 环境"
    print_info "请先运行: conda activate $CONDA_ENV_NAME"
    print_info "或手动创建环境: conda create -n $CONDA_ENV_NAME python=3.9"
    exit 1
fi

print_success "Conda 环境已配置: $CONDA_ENV_NAME"
echo ""

# 3. 配置 pip 镜像
echo ">>> 步骤 3/8: 配置 pip 国内镜像"
echo ""

mkdir -p ~/.pip
cat > ~/.pip/pip.conf << 'EOF'
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple

[install]
trusted-host = pypi.tuna.tsinghua.edu.cn
EOF

print_success "pip 镜像已配置: 清华源"
echo ""

# 4. 安装 PyTorch
echo ">>> 步骤 4/8: 安装 PyTorch"
echo ""

# 检查是否已安装 PyTorch
if python -c "import torch" &> /dev/null; then
    TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
    CUDA_AVAILABLE=$(python -c "import torch; print(torch.cuda.is_available())")
    print_info "PyTorch 已安装: $TORCH_VERSION"
    print_info "CUDA 可用: $CUDA_AVAILABLE"
    
    read -p "是否重新安装 PyTorch? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_info "跳过 PyTorch 安装"
    else
        pip uninstall -y torch torchvision torchaudio
        REINSTALL_TORCH=true
    fi
else
    REINSTALL_TORCH=true
fi

if [ "$REINSTALL_TORCH" = true ]; then
    print_info "正在安装 PyTorch..."
    
    # 检测 CUDA 版本并选择对应的 PyTorch
    if [ -n "$CUDA_VERSION" ]; then
        CUDA_MAJOR=$(echo $CUDA_VERSION | cut -d. -f1)
        CUDA_MINOR=$(echo $CUDA_VERSION | cut -d. -f2)
        
        if [ "$CUDA_MAJOR" -ge 12 ]; then
            print_info "检测到 CUDA $CUDA_VERSION，安装 PyTorch (CUDA 12.1)"
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        elif [ "$CUDA_MAJOR" -eq 11 ]; then
            print_info "检测到 CUDA $CUDA_VERSION，安装 PyTorch (CUDA 11.8)"
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        else
            print_warning "CUDA 版本较旧，安装 CPU 版本 PyTorch"
            pip install torch torchvision torchaudio
        fi
    else
        print_info "未检测到 CUDA，安装 CPU 版本 PyTorch"
        pip install torch torchvision torchaudio
    fi
    
    print_success "PyTorch 安装完成"
fi

# 验证 PyTorch
echo ""
print_info "验证 PyTorch 安装..."
python << 'EOF'
import torch
print("=" * 50)
print("PyTorch 安装验证")
print("=" * 50)
print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA 版本: {torch.version.cuda}")
    print(f"GPU 数量: {torch.cuda.device_count()}")
    print(f"GPU 型号: {torch.cuda.get_device_name(0)}")
    print(f"GPU 内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
print("=" * 50)
EOF

echo ""

# 5. 安装 MineDojo 依赖
echo ">>> 步骤 5/8: 安装 MineDojo 依赖"
echo ""

print_info "安装旧版本构建工具（MineDojo 需要）..."
pip install -q "pip<24.1" "setuptools<58" "wheel<0.38.0"
print_success "构建工具已安装"

print_info "安装 NumPy < 2.0 (MineDojo 要求)..."
pip install -q "numpy>=1.21.0,<2.0"
print_success "NumPy 已安装"

echo ""

# 6. 安装 MineDojo
echo ">>> 步骤 6/8: 安装 MineDojo"
echo ""

if python -c "import minedojo" &> /dev/null; then
    print_info "MineDojo 已安装"
    read -p "是否重新安装 MineDojo? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        pip uninstall -y minedojo
        pip install minedojo
        print_success "MineDojo 重新安装完成"
    else
        print_info "跳过 MineDojo 安装"
    fi
else
    print_info "正在安装 MineDojo..."
    pip install minedojo
    print_success "MineDojo 安装完成"
fi

echo ""

# 7. 安装项目依赖
echo ">>> 步骤 7/8: 安装项目依赖"
echo ""

if [ -f "requirements.txt" ]; then
    print_info "正在安装项目依赖..."
    pip install -q -r requirements.txt
    print_success "项目依赖安装完成"
else
    print_warning "未找到 requirements.txt"
fi

echo ""

# 8. 配置环境变量
echo ">>> 步骤 8/8: 配置环境变量"
echo ""

# 设置 JAVA_OPTS（无头模式）
if ! grep -q "JAVA_OPTS" ~/.bashrc; then
    echo 'export JAVA_OPTS="-Djava.awt.headless=true"' >> ~/.bashrc
    print_success "已添加 JAVA_OPTS 到 ~/.bashrc"
fi

# 设置 conda 自动激活
if ! grep -q "conda activate $CONDA_ENV_NAME" ~/.bashrc; then
    echo "conda activate $CONDA_ENV_NAME" >> ~/.bashrc
    print_success "已添加 conda 自动激活到 ~/.bashrc"
fi

export JAVA_OPTS="-Djava.awt.headless=true"
print_success "环境变量已配置"

echo ""
echo "========================================"
echo "部署完成！"
echo "========================================"
echo ""

# 9. 运行验证
echo ">>> 运行验证测试"
echo ""

if [ -f "tools/validate_install.py" ]; then
    print_info "运行验证脚本..."
    python tools/validate_install.py
else
    print_warning "未找到验证脚本: tools/validate_install.py"
    
    # 手动验证
    print_info "手动验证安装..."
    
    # 验证 MineDojo
    if python -c "import minedojo; print('✓ MineDojo 可用')" 2>/dev/null; then
        print_success "MineDojo 验证通过"
    else
        print_error "MineDojo 验证失败"
    fi
    
    # 验证 PyTorch
    if python -c "import torch; assert torch.cuda.is_available(), 'CUDA 不可用'; print('✓ PyTorch + CUDA 可用')" 2>/dev/null; then
        print_success "PyTorch + CUDA 验证通过"
    elif python -c "import torch; print('✓ PyTorch 可用 (CPU)')" 2>/dev/null; then
        print_warning "PyTorch 可用，但 CUDA 不可用（将使用 CPU）"
    else
        print_error "PyTorch 验证失败"
    fi
fi

echo ""
echo "========================================"
echo "下一步操作"
echo "========================================"
echo ""
echo "1. 快速测试（5-10 分钟）："
echo "   python src/training/train_get_wood.py --timesteps 10000 --use-mineclip --device cuda"
echo ""
echo "2. 开始训练（2-4 小时）："
echo "   python src/training/train_get_wood.py --use-mineclip --device cuda"
echo ""
echo "3. 启动 TensorBoard 监控："
echo "   tensorboard --logdir logs/tensorboard --host 0.0.0.0 --port 6006"
echo ""
echo "4. 查看文档："
echo "   cat docs/guides/AUTODL_DEPLOYMENT_GUIDE.md"
echo ""
echo "========================================"
echo "部署成功！祝训练顺利 🚀"
echo "========================================"

