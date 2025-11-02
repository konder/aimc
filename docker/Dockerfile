# AIMC MineDojo Docker 镜像 - 调试版本
# 只构建到 Python 依赖安装，不包含 MineCLIP 和 MC 编译
# 用于进入容器手动调试

FROM ubuntu:20.04

# 设置非交互模式
ENV DEBIAN_FRONTEND=noninteractive

# [重要] 如果在企业网络环境中，需要配置代理或使用国内镜像源
# 取消下面的注释来使用阿里云镜像源
#RUN sed -i 's|http://archive.ubuntu.com|http://mirrors.aliyun.com|g' /etc/apt/sources.list && \
#    sed -i 's|http://security.ubuntu.com|http://mirrors.aliyun.com|g' /etc/apt/sources.list

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
COPY build_local_cache.zip /root/build_local_cache.zip
RUN unzip /root/build_local_cache.zip -d /root/

# 安装 Python 依赖
SHELL ["/bin/bash", "-c"]
RUN source /opt/conda/etc/profile.d/conda.sh && \
    conda activate minedojo-x86 && \
    pip install "pip<24.1" "setuptools<58" "wheel<0.38.0" && \
    pip install minedojo

# ==================== Minedojo MC 编译步骤 ====================
# 步骤 1: 克隆 MixinGradle hotfix 并重组为标准 Maven 仓库结构
RUN mkdir -p /opt && \
    cd /opt && \
    git clone https://github.com/verityw/MixinGradle-dcfaf61.git hotfix-temp && \
    mkdir -p /opt/hotfix/MixinGradle-dcfaf61/MixinGradle/dcfaf61 && \
    cp /opt/hotfix-temp/MixinGradle/dcfaf61/MixinGradle-dcfaf61.jar /opt/hotfix/MixinGradle-dcfaf61/MixinGradle/dcfaf61/ && \
    echo '<?xml version="1.0" encoding="UTF-8"?>' > /opt/hotfix/MixinGradle-dcfaf61/MixinGradle/dcfaf61/MixinGradle-dcfaf61.pom && \
    echo '<project xmlns="http://maven.apache.org/POM/4.0.0">' >> /opt/hotfix/MixinGradle-dcfaf61/MixinGradle/dcfaf61/MixinGradle-dcfaf61.pom && \
    echo '  <modelVersion>4.0.0</modelVersion>' >> /opt/hotfix/MixinGradle-dcfaf61/MixinGradle/dcfaf61/MixinGradle-dcfaf61.pom && \
    echo '  <groupId>MixinGradle-dcfaf61</groupId>' >> /opt/hotfix/MixinGradle-dcfaf61/MixinGradle/dcfaf61/MixinGradle-dcfaf61.pom && \
    echo '  <artifactId>MixinGradle</artifactId>' >> /opt/hotfix/MixinGradle-dcfaf61/MixinGradle/dcfaf61/MixinGradle-dcfaf61.pom && \
    echo '  <version>dcfaf61</version>' >> /opt/hotfix/MixinGradle-dcfaf61/MixinGradle/dcfaf61/MixinGradle-dcfaf61.pom && \
    echo '</project>' >> /opt/hotfix/MixinGradle-dcfaf61/MixinGradle/dcfaf61/MixinGradle-dcfaf61.pom && \
    rm -rf /opt/hotfix-temp
ENV MINE_DOJO_MC_PATH=/opt/conda/envs/minedojo-x86/lib/python3.9/site-packages/minedojo/sim/Malmo/Minecraft

# 步骤 2: 应用 MC 配置补丁 并编译
COPY gradle-4.10.2-all.zip /opt/gradle-4.10.2-all.zip
COPY mc_config.patch /tmp/mc_config.patch
RUN cd $MINE_DOJO_MC_PATH && \
    patch -p0 < /tmp/mc_config.patch && \
    rm /tmp/mc_config.patch && \
    ./gradlew shadowJar

# 步骤 3: 修复MineDojo拉起MC的问题
RUN mkdir -p ${MINE_DOJO_MC_PATH}/run/gradle && \
    cp -r /root/.gradle/caches ${MINE_DOJO_MC_PATH}/run/gradle/

#RUN pip install git+https://github.com/MineDojo/MineCLIP

# 设置工作目录
WORKDIR /workspace

RUN git clone https://github.com/konder/aimc.git
RUN cd aimc && pip install -r requirements.txt

# 默认激活 minedojo-x86 环境
ENTRYPOINT ["/bin/bash", "-c", "source /opt/conda/etc/profile.d/conda.sh && conda activate minedojo-x86 && exec /bin/bash"]

