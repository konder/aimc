安装完Minedojo后，需要手动编译MC，以下是注意点
1. 解决MixinGradle无法找到问题
```
sudo mkdir -p /opt/hotfix/
cd /opt/hotfix/
git clone https://github.com/verityw/MixinGradle-dcfaf61.git
```

2. 找到miniconda3的安装路径，进入Malmo的Minecraft目录
```
cd /path_of_miniconda3/envs/minedojo-x86/lib/python3.9/site-packages/minedojo/sim/Malmo/Minecraft
```

3. 找到/path_of_miniconda3/envs/minedojo-x86/lib/python3.9/site-packages/minedojo/sim/Malmo/Minecraft/gradle/wrapper/gradle-wrapper.properties，将distributionUrl=https\://services.gradle.org/distributions/gradle-4.10.2-all.zip修改为distributionUrl=https://mirrors.aliyun.com/gradle/gradle-4.10.2-all.zip

4. 修改/path_of_miniconda3/envs/minedojo-x86/lib/python3.9/site-packages/minedojo/sim/Malmo/Minecraft/build.gradle，找到buildscript->repositories->增加maven {"file:///opt/hotfix/"},        maven { url "https://maven.aliyun.com/repository/public" }, maven { url "https://maven.aliyun.com/repository/central" }, maven { url "https://libraries.minecraft.net/" }

5. 对比下https://github.com/MineDojo/MineDojo/blob/main/minedojo/sim/Malmo/Minecraft/build.gradle和/usr/local/Caskroom/miniforge/base/envs/minedojo-x86/lib/python3.9/site-packages/minedojo/sim/Malmo/Minecraft/build.gradle的差异，形成path（后者本机为准， file:///opt/hotfix/特殊）


6. 执行 ./gradlew shadowJar
7. 在/path_of_miniconda3/envs/minedojo-x86/lib/python3.9/site-packages/minedojo/sim/Malmo/Minecraft/run下创建gradle目录
8. copy ~/.gradle/cache 到 /path_of_miniconda3/envs/minedojo-x86/lib/python3.9/site-packages/minedojo/sim/Malmo/Minecraft/run/gradle