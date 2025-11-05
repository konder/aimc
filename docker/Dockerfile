# AIMC（MineDojo/MineRL/MineClip/Steve1）环境

FROM ubuntu:20.04

# 设置非交互模式
ENV DEBIAN_FRONTEND=noninteractive

# [重要] 如果在企业网络环境中，需要配置代理或使用国内镜像源
# 取消下面的注释来使用阿里云镜像源
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
    zip \
    xvfb \
    x11-utils \
    x11-xserver-utils \
    vim \
    && rm -rf /var/lib/apt/lists/*

# 安装 Miniconda
RUN wget --no-check-certificate https://mirror.nju.edu.cn/github-release/conda-forge/miniforge/LatestRelease/Miniforge3-25.9.1-0-Linux-x86_64.sh -O /tmp/miniforge.sh && \
    bash /tmp/miniforge.sh -b -p /opt/conda && \
    rm /tmp/miniforge.sh

# 配置环境变量
ENV PATH="/opt/conda/bin:${PATH}"
ENV JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64
ENV MINEDOJO_HEADLESS=1

# 配置 conda 和 pip 使用国内镜像源（可选）
RUN conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/ && \
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/ && \
    conda config --set show_channel_urls yes

# 创建 minedojo-x86 环境
RUN conda create -n minedojo-x86 python=3.9 -y && \
    conda init bash && \
    echo "conda activate minedojo-x86" >> /root/.bashrc

# [重要] 如果在企业网络环境中，复制本地缓存文件
COPY docker/build_local_cache.zip /root/build_local_cache.zip
RUN unzip /root/build_local_cache.zip -d /root/

# 设置 SHELL 为 bash（重要！Docker 默认是 sh）
SHELL ["/bin/bash", "-c"]

# 创建 conda 环境激活脚本，方便后续使用
RUN echo '#!/bin/bash' > /opt/activate_env.sh && \
    echo 'source /opt/conda/etc/profile.d/conda.sh' >> /opt/activate_env.sh && \
    echo 'conda activate minedojo-x86' >> /opt/activate_env.sh && \
    chmod +x /opt/activate_env.sh

# ============================================================================
# 阶段 1: 准备依赖和补丁文件
# ============================================================================
# 准备 MixinGradle hotfix（MineDojo 编译依赖）
RUN mkdir -p /opt/hotfix/MixinGradle-dcfaf61/MixinGradle/dcfaf61
COPY docker/MixinGradle-dcfaf61.jar /opt/hotfix/MixinGradle-dcfaf61/MixinGradle/dcfaf61/
COPY docker/MixinGradle-dcfaf61.pom /opt/hotfix/MixinGradle-dcfaf61/MixinGradle/dcfaf61/

# 克隆 MineRL 源码（先下载，后面再安装）
RUN cd /tmp && \
    git clone https://github.com/minerllabs/minerl.git && \
    cd minerl && \
    git checkout v1.0.0

# 复制所有补丁文件到容器
COPY docker/minedojo_mc_config.patch /tmp/minedojo_mc_config.patch
COPY docker/minerl_mc_config.patch /tmp/minerl/minerl_mc_config.patch

# ============================================================================
# 阶段 2: 安装和配置 MineDojo
# ============================================================================
# 安装基础 Python 依赖和 MineDojo
RUN source /opt/activate_env.sh && \
    pip install "pip<24.1" "setuptools<58" "wheel<0.38.0" && \
    pip install minedojo && \
    pip install --force-reinstall gym==0.21.0

ENV MINE_DOJO_MC_PATH=/opt/conda/envs/minedojo-x86/lib/python3.9/site-packages/minedojo/sim/Malmo/Minecraft
# # 应用 MineDojo MC 配置补丁并编译
RUN cd $MINE_DOJO_MC_PATH && \
    patch -p0 < /tmp/minedojo_mc_config.patch && \
    ./gradlew shadowJar && \
    mkdir -p run/gradle && \
    cp -r /root/.gradle/caches run/gradle/

# ============================================================================
# 阶段 3: 准备 MineRL（应用补丁但暂不安装）
# ============================================================================
# 重要：MineRL 源码使用 CRLF 换行符，需要先转换为 Unix 格式（LF）
RUN cd /tmp/minerl && \
    find minerl scripts -name '*.py' -o -name '*.sh' | xargs sed -i 's/\r$//' && \
    patch -p0 < minerl_mc_config.patch

# ============================================================================
# 阶段 4: 安装项目依赖
# ============================================================================
WORKDIR /workspace

COPY requirements.txt /tmp/requirements.txt
RUN source /opt/activate_env.sh && \
    pip install -r /tmp/requirements.txt

RUN source /opt/activate_env.sh && \
    pip install git+https://github.com/MineDojo/MineCLIP
# ============================================================================
# 阶段 5: 安装 MineRL
# ============================================================================
# 注意: 如果 gym 0.21.0 导致 AttributeError，取消下面的注释强制使用 0.19.0
RUN source /opt/activate_env.sh && \
    cd /tmp/minerl && \
    pip install . && \
    cd /tmp && \
    rm -rf minerl /tmp/minerl_mc_config.patch /tmp/requirements.txt

# 修复 MineRL launchClient.sh - 在 Docker 无头模式下使用 xvfb-run
RUN sed -i 's#^java -Xmx\(.*\)$#[ -z "$DISPLAY" ] \&\& exec xvfb-run -a -e /dev/stdout -s '\''-screen 0 1400x900x24'\'' java -Xmx\1 || exec java -Xmx\1#' /opt/conda/envs/minedojo-x86/lib/python3.9/site-packages/minerl/MCP-Reborn/launchClient.sh

ENV MINE_RL_MC_PATH=/opt/conda/envs/minedojo-x86/lib/python3.9/site-packages/minerl/MCP-Reborn/
# ============================================================================
# 阶段 6: 环境验证
# ============================================================================
COPY docker/test_environments.py /workspace/

# ============================================================================
# 阶段 7: 解决核心包冲突
# ============================================================================
RUN source /opt/activate_env.sh && \
    pip install gymnasium=="1.1.1" && \
    pip install numpy=="1.24.3" && \
    pip install gym=="0.21.0"

# 默认激活 minedojo-x86 环境
ENTRYPOINT ["/bin/bash", "-c", "source /opt/conda/etc/profile.d/conda.sh && conda activate minedojo-x86 && exec /bin/bash"]

