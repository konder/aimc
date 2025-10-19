# 基于 MineDojo 官方镜像，应用 MixinGradle 修复
FROM minedojo/minedojo:latest

LABEL maintainer="AIMC Project"
LABEL description="MineDojo with MixinGradle fix for Issue #113"

# 切换到 root 用户以安装软件
USER root

RUN apt-get update && apt-get install -y \
    git \
    vim

# MixinGradle 问题修复
RUN cd /root &&git clone https://github.com/verityw/MixinGradle-dcfaf61.git
RUN cd /opt/conda/lib/python3.9/site-packages/minedojo/sim/Malmo/Minecraft && \
    sed -i '/buildscript {/,/repositories {/ {/repositories {/a\        maven { url "file:///root" }}' build.gradle && \
    sed -i "s|com.github.SpongePowered:MixinGradle:dcfaf61|MixinGradle-dcfaf61:MixinGradle:dcfaf61|g" build.gradle && \
    sed -i "s|brandonhoughton:ForgeGradle|MineDojo:ForgeGradle|g" build.gradle && \
    sed -i "s|brandonhoughton:forgegradle|MineDojo:ForgeGradle|g" build.gradle && \
    sed -i "s|new File('src/main/resources/schemas.index')|new File(projectDir, 'src/main/resources/schemas.index')|g" build.gradle/

# 构建 JAR 和相关资源
RUN cd /opt/conda/lib/python3.9/site-packages/minedojo/sim/Malmo/Minecraft && \
    ./gradlew shadowJar

RUN cd /opt/conda/lib/python3.9/site-packages/minedojo/sim/Malmo/Minecraft && \
    mkdir -p run/gradle && \
    cp -r /root/.gradle/caches run/gradle/

# 设置工作目录
WORKDIR /workspace

# 添加标签说明
RUN echo "" && \
    echo "================================" && \
    echo "MineDojo" && \
    echo "✓ Ready to use!" && \
    echo "================================" && \
    echo ""

# 默认命令
CMD ["bash"]
