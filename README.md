#如何在ARM64上通过Rosetta 2部署minedojo

- 安装x86的jdk
```
arch -x86_64 brew install temurin@8
```
- 设置JAVA_HOME，用arch开启一个bash
```
export JAVA_HOME=/Library/Java/JavaVirtualMachines/temurin-8.jdk/Contents/Home/
source ~/.bash_profile
arch -x86_64 /bin/bash
```
- 创建minedojo-x86的python虚拟环境
```
conda create -n minedojo-x86 python=3.9 -y
conda activate minedojo-x86
```
- 安装minedojo前的国内代理（可选）
```
mkdir -p ~/.pip && \
    echo "[global]" > ~/.pip/pip.conf && \
    echo "index-url = https://pypi.tuna.tsinghua.edu.cn/simple" >> ~/.pip/pip.conf && \
    echo "[install]" >> ~/.pip/pip.conf && \
    echo "trusted-host = pypi.tuna.tsinghua.edu.cn" >> ~/.pip/pip.conf
```
- 安装minedojo
```
pip install "pip<24.1" "setuptools<58" "wheel<0.38.0"
pip install "numpy>=1.21.0,<2.0"
pip install minedojo
```
- 解决编译Minecraft的MixinGradle问题
```
mkdir /opt/MixinGradle
cd /opt/MixinGradle && git clone https://github.com/verityw/MixinGradle-dcfaf61.git
```
- 修复Malmo的编译Minecraft一系列问题
```
cd /usr/local/Caskroom/miniforge/base/envs/minedojo-x86/lib/python3.9/site-packages/minedojo/sim/Malmo/Minecraft
sed -i '/repositories {/a\        maven { url "file:///opt/hotfix" }' build.gradle
sed -i '4i\     maven { url "https://maven.aliyun.com/repository/public" }' build.gradle
sed -i '5i\     maven { url "https://maven.aliyun.com/repository/central" }' build.gradle
sed -i '6i\     maven { url "https://libraries.minecraft.net/" }' build.gradle
sed -i "s|com.github.SpongePowered:MixinGradle:dcfaf61|MixinGradle-dcfaf61:MixinGradle:dcfaf61|g" build.gradle
sed -i "s|brandonhoughton:ForgeGradle|MineDojo:ForgeGradle|g" build.gradle
sed -i "s|brandonhoughton:forgegradle|MineDojo:ForgeGradle|g" build.gradle
sed -i "s|new File('src/main/resources/schemas.index')|new File(projectDir, 'src/main/resources/schemas.index')|g" build.gradle
```
- 编译Minecraft前的代理（可选）
```
mkdir -p /root/.gradle
echo 'allprojects {\n\
    repositories {\n\
    maven { url "https://maven.aliyun.com/repository/public" }\n\
    maven { url "https://maven.aliyun.com/repository/central" }\n\
    maven { url "https://maven.aliyun.com/repository/gradle-plugin" }\n\
    maven { url "https://maven.aliyun.com/repository/spring" }\n\
    maven { url "https://maven.aliyun.com/repository/spring-plugin" }\n\
    maven { url "https://libraries.minecraft.net/" }\n\
    mavenCentral()\n\
    gradlePluginPortal()\n\
    mavenLocal()\n\
    }\n\
    }' > ~/.gradle/init.gradle
```
- 编译Mminecraft
```
/usr/local/Caskroom/miniforge/base/envs/minedojo-x86/lib/python3.9/site-packages/minedojo/sim/Malmo/Minecraft/gradlew shadowJar
mkdir /opt/MineDojo/minedojo/sim/Malmo/Minecraft/run/gradle && cp -r ~/.gradle/caches /opt/MineDojo/minedojo/sim/Malmo/Minecraft/run/gradle
```
- 如果有lwjgl问题，手动下载LWJGL-2.93库和修改launchClient.sh启用
    - 下载https://sf-west-interserver-1.dl.sourceforge.net/project/java-game-lib/Official%20Releases/LWJGL%202.9.3/lwjgl-2.9.3.zip?viasf=1
    - 修改/usr/local/Caskroom/miniforge/base/envs/minedojo-x86/lib/python3.9/site-packages/minedojo/sim/Malmo/Minecraft/launchClient.sh
    - 将启动命令改为java -Djava.library.path=/Users/nanzhang/lwjgl-2.9.3/native/macosx -Dorg.lwjgl.librarypath=/Users/nanzhang/lwjgl-2.9.3/native/macosx -Dfml.coreMods.load=com.microsoft.Malmo.OverclockingPlugin -Xmx2G -Dfile.encoding=UTF-8 -Duser.country=US -Duser.language=en -Duser.variant -jar ../build/libs/MalmoMod-0.37.0-fat.jar