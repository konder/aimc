# AIMC MineDojo Docker 镜像
# 基于 Ubuntu 20.04 + Miniconda + OpenJDK 8

FROM ubuntu:20.04

# 设置非交互模式
ENV DEBIAN_FRONTEND=noninteractive

# 如果在企业网络环境中，需要配置代理或使用国内镜像源（可选）
RUN sed -i 's|http://archive.ubuntu.com|http://mirrors.aliyun.com|g' /etc/apt/sources.list && \
    sed -i 's|http://security.ubuntu.com|http://mirrors.aliyun.com|g' /etc/apt/sources.list

# 更新系统并安装基础依赖
RUN apt-get update && \
    apt-get install -y \
    openjdk-8-jdk \
    wget \
    curl \
    git \
    build-essential \
    ca-certificates \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    xvfb \
    xserver-xephyr \
    vnc4server \
    python-opengl \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 安装 Miniconda
# 如果网络受限，请先手动下载 Miniconda3-latest-Linux-x86_64.sh 到 docker/ 目录
# 然后取消下面 COPY 行的注释，并注释掉 wget 行
# COPY Miniconda3-latest-Linux-x86_64.sh /tmp/miniconda.sh
RUN wget --no-check-certificate https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh

# 配置环境变量
ENV PATH="/opt/conda/bin:${PATH}"
ENV JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64
ENV MINEDOJO_HEADLESS=1

# 配置 conda 和 pip 使用国内镜像源（可选）
RUN conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/ && \
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/ && \
    conda config --set show_channel_urls yes && \
    mkdir -p /root/.pip && \
    echo "[global]" > /root/.pip/pip.conf && \
    echo "index-url = https://pypi.tuna.tsinghua.edu.cn/simple" >> /root/.pip/pip.conf && \
    echo "[install]" >> /root/.pip/pip.conf && \
    echo "trusted-host = pypi.tuna.tsinghua.edu.cn" >> /root/.pip/pip.conf

# 接受 Conda 服务条款并创建 minedojo-x86 环境
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    conda create -n minedojo-x86 python=3.9 -y && \
    conda init bash && \
    echo "conda activate minedojo-x86" >> /root/.bashrc

# 复制项目文件
COPY requirements.txt /tmp/requirements.txt

# 安装 Python 依赖
SHELL ["/bin/bash", "-c"]
RUN source /opt/conda/etc/profile.d/conda.sh && \
    conda activate minedojo-x86 && \
    pip install "pip<24.1" "setuptools<58" "wheel<0.38.0" && \
    pip install minedojo && \
    pip install -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

# 安装 MineCLIP (从 GitHub) - 可选，如果网络问题可以在容器内手动安装
RUN source /opt/conda/etc/profile.d/conda.sh && \
    conda activate minedojo-x86 && \
    pip install git+https://github.com/MineDojo/MineCLIP.git || echo "⚠️  MineCLIP 安装失败，可在容器内手动安装"

ENV MC_PATH=/opt/conda/envs/minedojo-x86/lib/python3.9/site-packages/minedojo/sim/Malmo/Minecraft
# ==================== MC 编译步骤 ====================
# 步骤 1: 克隆 MixinGradle hotfix（注意路径要与 patch 中的 Maven URL 一致）
RUN mkdir -p /opt && \
    cd /opt && \
    git clone https://github.com/verityw/MixinGradle-dcfaf61.git hotfix

# 步骤 2: 应用 MC 配置补丁（基于调试容器中的成功方案）
COPY docker/mc_config.patch /tmp/mc_config.patch

RUN cd $MC_PATH && \
    patch -p0 < /tmp/mc_config.patch && \
    rm /tmp/mc_config.patch && \
    ./gradlew shadowJar

RUN cd $MC_PATH && \
    mkdir -p ${MC_PATH}/run/gradle && \
    if [ -d "$HOME/.gradle/caches" ]; then \
        cp -r $HOME/.gradle/caches ${MC_PATH}/run/gradle/; \
    fi

# 设置工作目录
WORKDIR /workspace

# 默认激活 minedojo-x86 环境
ENTRYPOINT ["/bin/bash", "-c", "source /opt/conda/etc/profile.d/conda.sh && conda activate minedojo-x86 && exec /bin/bash"]
