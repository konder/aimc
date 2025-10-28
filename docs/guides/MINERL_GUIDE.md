# MineRL 完整安装和使用指南

## 📋 概述

**最后更新**: 2025-10-28  
**环境**: macOS ARM64 + Rosetta 2 + x86  
**MineRL 版本**: 1.0.0  
**状态**: ✅ 完全可用（包括窗口显示）

---

## 📚 文档导航

### 本项目 MineRL 文档结构

```
aimc/
├── MINERL_QUICKSTART.md              ← 快速参考（推荐从这里开始）
└── docs/
    ├── guides/
    │   └── MINERL_GUIDE.md           ← 本文档（完整指南）
    └── summaries/
        └── MINERL_FINAL_SUCCESS.md   ← 项目总结和技术细节
```

### 按需求查找

| 需求 | 推荐文档 |
|------|---------|
| 🚀 **快速开始** | `MINERL_QUICKSTART.md` |
| 📖 **详细安装** | 本文档 |
| 🐛 **问题排查** | 本文档 → 故障排除章节 |
| 📝 **项目历史** | `docs/summaries/MINERL_FINAL_SUCCESS.md` |

---

## 🔑 核心要点（必读）

从所有调试过程中总结的最关键信息：

1. ⭐ **OpenCV 版本必须是 4.8.1.78**
   - OpenCV 4.11.0 在 macOS 上缺少 GUI 后端
   - 这是 `env.render()` 能否工作的关键

2. ⭐ **Java 参数必须有 -XstartOnFirstThread**
   - macOS 的 GLFW/OpenGL 要求
   - 需要修改 `MCP-Reborn/launchClient.sh`

3. ⭐ **清除 DISPLAY 变量（如果安装了 XQuartz）**
   - XQuartz 会干扰原生窗口显示
   - 在启动脚本中 `unset DISPLAY`

---

## 🎯 快速开始

### 完整安装步骤

```bash
# 1. 从 GitHub 克隆（PyPI 无 1.0.0 版本）
cd /tmp
git clone https://github.com/minerllabs/minerl.git
cd minerl
git checkout v1.0.0

# 2. 初始化子模块（重要！）
git submodule update --init --recursive

# 3. 修改启动脚本（macOS 必需）
# 编辑 MCP-Reborn/launchClient.sh
# 找到这行：
#   java -Xmx$maxMem -jar $fatjar --envPort=$port
# 改为：
#   java -XstartOnFirstThread -Xmx$maxMem -jar $fatjar --envPort=$port

# 4. 安装 MineRL
pip install -e .

# 5. 安装正确的 OpenCV 版本（关键！）
pip uninstall opencv-python -y
pip install opencv-python==4.8.1.78
```

### 验证安装

```bash
./scripts/run_minedojo_x86.sh python -c "
import gym
import minerl
import cv2

print('✓ MineRL installed')
print('✓ OpenCV:', cv2.__version__)
print('✓ Gym:', gym.__version__)

# 测试环境创建
env = gym.make('MineRLBasaltFindCave-v0')
print('✓ Environment created')
env.close()
print('✓ All systems OK')
"
```

### 基本使用

```python
import gym
import minerl

# 创建环境
env = gym.make("MineRLBasaltBuildVillageHouse-v0")

# 重置
obs = env.reset()

# 运行
for _ in range(100):
    action = env.action_space.noop()
    action['forward'] = 1
    obs, reward, done, info = env.step(action)
    
    # 显示窗口（已修复）
    env.render()
    
    if done:
        obs = env.reset()

env.close()
```

---

## 🔧 关键问题和解决方案

### 问题 1: GLFW 窗口崩溃 ✅

**错误信息**:
```
java.lang.IllegalStateException: GLFW windows may only be created on the main thread
and that thread must be the first thread in the process. Please run the JVM with
-XstartOnFirstThread.
```

**原因**: macOS 的 GLFW/OpenGL 要求窗口必须在主线程创建

**解决方案**:

修改 `MCP-Reborn/launchClient.sh`:
```bash
# 找到 java 启动命令行
# 将
java -Xmx$maxMem -jar $fatjar --envPort=$port

# 改为
java -XstartOnFirstThread -Xmx$maxMem -jar $fatjar --envPort=$port
```

---

### 问题 2: OpenCV 窗口不显示 ✅

**错误信息**:
```
cv2.error: OpenCV(4.11.0) :-1: error: (-5:Bad argument) in function 'imshow'
> Overload resolution failed:
>  - mat is not a numpy array, neither a scalar
>  - Expected Ptr<cv::cuda::GpuMat> for argument 'mat'
>  - Expected Ptr<cv::UMat> for argument 'mat'
```

**原因**: OpenCV 4.11.0 在 macOS 上缺少 GUI 后端支持（`可用后端: []`）

**解决方案**: 降级到稳定版本
```bash
pip uninstall opencv-python -y
pip install opencv-python==4.8.1.78
```

**验证修复**:
```python
import cv2
print(cv2.getBuildInformation())
# 应该看到 Cocoa 后端可用
```

---

### 问题 3: XQuartz DISPLAY 干扰 ✅

**症状**: 
- 设置了 `DISPLAY` 环境变量
- Minecraft 窗口无法正常显示
- 日志显示尝试使用 X11

**解决方案**: 在启动脚本中清除
```bash
# scripts/run_minedojo_x86.sh
unset DISPLAY
```

**原因**: 
- XQuartz 设置 `DISPLAY` 指向 X11 服务器
- LWJGL 会尝试使用 X11 而不是原生 Cocoa
- macOS 应该使用原生窗口系统

---

## 📦 依赖管理

### 核心依赖版本

```txt
# Python 环境
python==3.9

# 必需依赖
gym==0.19.0             # MineRL 1.0.0 要求
numpy==1.24.3           # 稳定版本
opencv-python==4.8.1.78 # ← 关键！必须是这个版本

# 系统依赖（macOS）
java==1.8               # Temurin 8
```

### 推荐的启动脚本

`scripts/run_minedojo_x86.sh`:
```bash
#!/bin/bash
# MineRL/MineDojo 通用启动脚本

# x86 架构切换（如果在 ARM64 Mac 上）
if [ "$(uname -m)" = "arm64" ]; then
    exec arch -x86_64 /bin/zsh "$0" "$@"
fi

# Java 环境配置
export JAVA_HOME=/Library/Java/JavaVirtualMachines/temurin-8.jdk/Contents/Home
export PATH="$JAVA_HOME/bin:$PATH"
export JAVA_OPTS="-XstartOnFirstThread -Xmx4G"

# 清除可能干扰的环境变量
unset DISPLAY

# 激活 conda 环境
eval "$(conda shell.bash hook)"
conda activate minedojo-x86

# 设置项目路径
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# 执行命令
exec "$@"
```

---

## 🎮 使用示例

### 示例 1: 基础训练循环

```python
import gym
import minerl

env = gym.make("MineRLBasaltFindCave-v0")
obs = env.reset()

for episode in range(10):
    done = False
    total_reward = 0
    step = 0
    
    while not done:
        # 你的策略（这里是简单的前进）
        action = env.action_space.noop()
        action['forward'] = 1
        
        obs, reward, done, info = env.step(action)
        total_reward += reward
        step += 1
    
    print(f"Episode {episode}: steps={step}, reward={total_reward}")
    obs = env.reset()

env.close()
```

### 示例 2: 带窗口显示的训练

```python
import gym
import minerl
import cv2

env = gym.make("MineRLBasaltBuildVillageHouse-v0")
obs = env.reset()

print("按 'q' 退出")

for _ in range(1000):
    action = env.action_space.noop()
    action['camera'] = [0, 3]  # 旋转视角
    action['forward'] = 1      # 前进
    
    obs, reward, done, info = env.step(action)
    
    # 方法 1: 使用 env.render()（推荐）
    env.render()
    
    # 方法 2: 手动显示（更灵活）
    # pov = obs['pov'][:, :, ::-1]  # RGB -> BGR
    # cv2.imshow("MineRL", pov)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break
    
    if done:
        print("任务完成，重置环境")
        obs = env.reset()

cv2.destroyAllWindows()
env.close()
```

### 示例 3: 保存视频

```python
import gym
import minerl
from PIL import Image
import os

env = gym.make("MineRLBasaltBuildVillageHouse-v0")
obs = env.reset()

# 创建输出目录
output_dir = "minerl_frames"
os.makedirs(output_dir, exist_ok=True)

print(f"保存帧到 {output_dir}/")

for i in range(500):
    action = env.action_space.noop()
    action['forward'] = 1
    action['camera'] = [0, 1]
    
    obs, reward, done, info = env.step(action)
    
    # 保存每一帧
    img = Image.fromarray(obs['pov'])
    img.save(f"{output_dir}/frame_{i:04d}.png")
    
    if i % 50 == 0:
        print(f"已保存 {i} 帧")
    
    if done:
        obs = env.reset()

env.close()

print(f"✓ 保存完成！共 {i+1} 帧")
print(f"合成视频命令:")
print(f"  ffmpeg -framerate 20 -i {output_dir}/frame_%04d.png output.mp4")
```

### 示例 4: 人类控制（键盘输入）

```python
import gym
import minerl
import cv2
import numpy as np

def get_keyboard_action(env):
    """根据键盘输入返回动作"""
    action = env.action_space.noop()
    
    key = cv2.waitKey(50) & 0xFF
    
    # WASD 移动
    if key == ord('w'):
        action['forward'] = 1
    elif key == ord('s'):
        action['back'] = 1
    elif key == ord('a'):
        action['left'] = 1
    elif key == ord('d'):
        action['right'] = 1
    
    # 跳跃和攻击
    if key == ord(' '):
        action['jump'] = 1
    if key == ord('j'):
        action['attack'] = 1
    
    # 退出
    if key == ord('q'):
        return None
    
    return action

env = gym.make("MineRLBasaltFindCave-v0")
obs = env.reset()

print("控制:")
print("  WASD - 移动")
print("  Space - 跳跃")
print("  J - 攻击")
print("  Q - 退出")

while True:
    action = get_keyboard_action(env)
    if action is None:
        break
    
    obs, reward, done, info = env.step(action)
    env.render()
    
    if done:
        obs = env.reset()

env.close()
```

---

## 🆚 MineRL vs MineDojo

### 功能对比

| 特性 | MineRL 1.0.0 | MineDojo |
|------|-------------|----------|
| **窗口显示** | ✅ 支持（需正确配置） | ✅ 原生支持 |
| **OpenCV 版本** | 4.8.1.78 | 4.8.1.78 |
| **BASALT 任务** | ✅ 内置 | ❌ 需自定义 |
| **安装难度** | ⭐⭐⭐⭐ | ⭐⭐ |
| **配置复杂度** | 高 | 低 |
| **文档完整度** | 中等 | 优秀 |
| **社区支持** | BASALT 竞赛 | 研究社区 |
| **适用场景** | 竞赛、BASALT | 研究、开发 |

### 使用建议

**选择 MineRL 1.0.0 的场景**:
- 🏆 参加 BASALT 竞赛
- 📊 需要 BASALT 任务的标准环境
- 🔬 复现 BASALT 相关论文

**选择 MineDojo 的场景**:
- 🎮 日常开发和调试
- 📚 学习强化学习
- ⚡ 快速原型开发
- 🔧 自定义任务

**两者可以共存**，使用同一个启动脚本 `run_minedojo_x86.sh`。

---

## 🔍 测试和验证

### 快速验证脚本

```bash
#!/bin/bash
# test_minerl.sh - 验证 MineRL 安装

echo "=== MineRL 安装验证 ==="

./scripts/run_minedojo_x86.sh python << 'EOF'
import sys
import gym
import minerl
import cv2
import numpy as np

print("\n1. 检查版本")
print(f"   Python: {sys.version.split()[0]}")
print(f"   OpenCV: {cv2.__version__}")
print(f"   Gym: {gym.__version__}")
print(f"   NumPy: {np.__version__}")

print("\n2. 检查环境变量")
import os
print(f"   JAVA_OPTS: {os.environ.get('JAVA_OPTS', 'NOT SET')}")
print(f"   DISPLAY: {os.environ.get('DISPLAY', 'NOT SET')}")

print("\n3. 测试环境创建")
try:
    env = gym.make('MineRLBasaltFindCave-v0')
    print("   ✓ 环境创建成功")
    env.close()
except Exception as e:
    print(f"   ✗ 错误: {e}")
    sys.exit(1)

print("\n4. 测试 OpenCV GUI")
try:
    test_img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.imshow("Test", test_img)
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    print("   ✓ OpenCV GUI 可用")
except Exception as e:
    print(f"   ✗ OpenCV GUI 错误: {e}")
    sys.exit(1)

print("\n✓ 所有测试通过！")
EOF
```

### 完整功能测试

```python
# test_minerl_full.py
import gym
import minerl
import cv2
import time

def test_minerl():
    """完整功能测试"""
    
    print("=== MineRL 完整测试 ===\n")
    
    # 1. 环境创建
    print("1. 创建环境...")
    env = gym.make("MineRLBasaltBuildVillageHouse-v0")
    print("   ✓ 环境创建成功")
    
    # 2. 重置
    print("\n2. 重置环境...")
    obs = env.reset()
    print(f"   ✓ 观测空间: {obs['pov'].shape}")
    
    # 3. 运行步骤
    print("\n3. 运行 20 步...")
    for i in range(20):
        action = env.action_space.noop()
        action['forward'] = 1
        action['camera'] = [0, 2]
        
        obs, reward, done, info = env.step(action)
        
        # 显示窗口
        env.render()
        
        if (i + 1) % 5 == 0:
            print(f"   步骤 {i+1}/20")
        
        if done:
            print("   任务完成，重置")
            obs = env.reset()
        
        time.sleep(0.05)
    
    # 4. 清理
    print("\n4. 清理...")
    env.close()
    cv2.destroyAllWindows()
    print("   ✓ 环境已关闭")
    
    print("\n✓ 所有测试通过！MineRL 工作正常")

if __name__ == "__main__":
    test_minerl()
```

---

## 🐛 故障排除

### 诊断清单

运行这个诊断脚本来检查配置：

```bash
./scripts/run_minedojo_x86.sh python << 'EOF'
import sys, os, cv2, gym

print("=== MineRL 诊断 ===\n")

# 1. 架构
import platform
print(f"1. 架构: {platform.machine()}")
if platform.machine() != "x86_64":
    print("   ⚠️  警告: 应该是 x86_64")

# 2. OpenCV
print(f"\n2. OpenCV: {cv2.__version__}")
if not cv2.__version__.startswith("4.8.1"):
    print("   ⚠️  警告: 推荐 4.8.1.78")

# 3. 环境变量
print(f"\n3. JAVA_OPTS: {os.environ.get('JAVA_OPTS', 'NOT SET')}")
if '-XstartOnFirstThread' not in os.environ.get('JAVA_OPTS', ''):
    print("   ⚠️  警告: 缺少 -XstartOnFirstThread")

print(f"4. DISPLAY: {os.environ.get('DISPLAY', 'NOT SET')}")
if 'DISPLAY' in os.environ:
    print("   ⚠️  警告: DISPLAY 应该被清除")

# 5. Gym
print(f"\n5. Gym: {gym.__version__}")
if gym.__version__ != "0.19.0":
    print("   ⚠️  警告: MineRL 1.0.0 要求 gym==0.19.0")

print("\n诊断完成")
EOF
```

### 常见问题

#### Q1: `env.render()` 失败，显示 "mat is not a numpy array"

**原因**: OpenCV 版本问题

**解决**:
```bash
pip install opencv-python==4.8.1.78 --force-reinstall
```

#### Q2: Minecraft 窗口不显示

**检查**:
```bash
# 检查 launchClient.sh
grep "XstartOnFirstThread" \
  $(python -c "import minerl; print(minerl.__path__[0])")/MCP-Reborn/launchClient.sh
```

**如果没有找到**，手动添加：
```bash
MINERL_PATH=$(python -c "import minerl; print(minerl.__path__[0])")
sed -i '' 's/java -Xmx/java -XstartOnFirstThread -Xmx/g' \
  "$MINERL_PATH/MCP-Reborn/launchClient.sh"
```

#### Q3: GLFW 错误崩溃

**确保**:
1. 在 x86 模式: `uname -m` 应该输出 `x86_64`
2. JAVA_OPTS 正确: `echo $JAVA_OPTS`
3. 使用启动脚本: `./scripts/run_minedojo_x86.sh`

#### Q4: gym 版本冲突

MineRL 1.0.0 需要 `gym==0.19.0`，可能与其他包冲突：

```bash
# 查看冲突
pip check

# 如果必须保留 gym 0.21（MineDojo 需要），考虑使用虚拟环境隔离
```

#### Q5: 仍然无法解决？

**替代方案**:
1. 使用 MineDojo（更简单，功能类似）
2. 使用 Matplotlib 可视化而不是窗口显示
3. 查看详细日志: `logs/mc_*.log`

---

## 📚 可用任务

### BASALT 竞赛任务

```python
# 4 个 BASALT 任务
'MineRLBasaltFindCave-v0'              # 找到洞穴
'MineRLBasaltMakeWaterfall-v0'         # 建造瀑布
'MineRLBasaltCreateVillageAnimalPen-v0'  # 建造动物圈
'MineRLBasaltBuildVillageHouse-v0'     # 建造村庄房屋
```

### 经典任务

```python
# 其他常用任务
'MineRLTreechop-v0'       # 砍树
'MineRLNavigate-v0'       # 导航到目标
'MineRLNavigateExtreme-v0'  # 极限导航
'MineRLObtainDiamond-v0'  # 获取钻石
'MineRLObtainIronPickaxe-v0'  # 获取铁镐
```

### 查看所有任务

```python
import gym
import minerl

# 列出所有 MineRL 环境
all_envs = [env_id for env_id in gym.envs.registry.keys() 
            if env_id.startswith('MineRL')]
print(f"共有 {len(all_envs)} 个 MineRL 环境")
for env_id in sorted(all_envs):
    print(f"  - {env_id}")
```

---

## 📝 最佳实践

### 1. 环境初始化

```python
import os
import gym
import minerl

# 推荐的环境配置
def create_minerl_env(task_name, render=True):
    """创建 MineRL 环境的推荐方式"""
    
    # 确保环境变量正确
    os.environ['JAVA_OPTS'] = '-XstartOnFirstThread -Xmx4G'
    if 'DISPLAY' in os.environ:
        del os.environ['DISPLAY']
    
    # 创建环境
    env = gym.make(task_name)
    
    return env

# 使用
env = create_minerl_env("MineRLBasaltFindCave-v0")
```

### 2. 错误处理

```python
import gym
import minerl

def safe_train():
    """带错误处理的训练循环"""
    env = None
    
    try:
        env = gym.make("MineRLBasaltFindCave-v0")
        obs = env.reset()
        
        for episode in range(10):
            done = False
            while not done:
                action = env.action_space.noop()
                action['forward'] = 1
                
                try:
                    obs, reward, done, info = env.step(action)
                    env.render()
                except Exception as e:
                    print(f"步骤错误: {e}")
                    break
            
            obs = env.reset()
            
    except KeyboardInterrupt:
        print("\n训练被用户中断")
    except Exception as e:
        print(f"错误: {e}")
    finally:
        if env is not None:
            env.close()
            print("环境已清理")

safe_train()
```

### 3. 性能优化

```python
import gym
import minerl
from concurrent.futures import ProcessPoolExecutor

def train_episode(episode_id):
    """单个 episode 的训练"""
    env = gym.make("MineRLBasaltFindCave-v0")
    obs = env.reset()
    
    total_reward = 0
    done = False
    
    while not done:
        action = env.action_space.noop()
        action['forward'] = 1
        obs, reward, done, info = env.step(action)
        total_reward += reward
    
    env.close()
    return episode_id, total_reward

# 并行训练（注意：不能显示窗口）
def parallel_train(num_episodes=10, num_workers=4):
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = executor.map(train_episode, range(num_episodes))
    
    for episode_id, reward in results:
        print(f"Episode {episode_id}: reward={reward}")

# 使用
# parallel_train()
```

### 4. 数据收集

```python
import gym
import minerl
import numpy as np
import pickle

def collect_demonstrations(task_name, num_episodes=10, output_file="demos.pkl"):
    """收集人类演示数据"""
    
    env = gym.make(task_name)
    demonstrations = []
    
    for episode in range(num_episodes):
        obs = env.reset()
        episode_data = {
            'observations': [],
            'actions': [],
            'rewards': []
        }
        
        done = False
        while not done:
            # 这里可以是人类输入或策略输出
            action = env.action_space.noop()
            action['forward'] = 1
            
            episode_data['observations'].append(obs)
            episode_data['actions'].append(action)
            
            obs, reward, done, info = env.step(action)
            episode_data['rewards'].append(reward)
            
            env.render()
        
        demonstrations.append(episode_data)
        print(f"收集 episode {episode+1}/{num_episodes}")
    
    env.close()
    
    # 保存数据
    with open(output_file, 'wb') as f:
        pickle.dump(demonstrations, f)
    
    print(f"✓ 数据已保存到 {output_file}")

# 使用
# collect_demonstrations("MineRLBasaltBuildVillageHouse-v0", num_episodes=5)
```

---

## 🎉 总结

### ✅ 最终配置

```
MineRL 1.0.0 成功配置
├─ OpenCV 4.8.1.78 ⭐ (关键)
├─ Java 8 + -XstartOnFirstThread ⭐
├─ unset DISPLAY ⭐
├─ gym==0.19.0
└─ numpy==1.24.3
```

### ✅ 功能状态

| 功能 | 状态 | 说明 |
|------|------|------|
| 安装 | ✅ | 从 GitHub 成功安装 |
| 运行 | ✅ | 程序稳定，无崩溃 |
| 窗口显示 | ✅ | **完美工作** |
| `env.render()` | ✅ | **完全可用** |
| BASALT 任务 | ✅ | 全部 4 个任务可用 |
| 训练 | ✅ | 可以正常训练 |

### ✅ 已解决的问题

- ✅ GLFW 窗口崩溃（-XstartOnFirstThread）
- ✅ OpenCV GUI 不工作（降级到 4.8.1.78）
- ✅ DISPLAY 变量干扰（unset DISPLAY）
- ✅ macOS ARM64 兼容性（Rosetta 2 + x86）
- ✅ 窗口显示问题（完整配置）

---

## 🔗 相关资源

### 官方资源
- **MineRL GitHub**: https://github.com/minerllabs/minerl
- **BASALT 竞赛**: https://www.aicrowd.com/challenges/neurips-2022-minerl-basalt-competition
- **MineRL 文档**: https://minerl.readthedocs.io/

### 本项目资源
- **快速参考**: `/MINERL_QUICKSTART.md`
- **项目总结**: `docs/summaries/MINERL_FINAL_SUCCESS.md`
- **启动脚本**: `scripts/run_minedojo_x86.sh`
- **MineDojo 文档**: `docs/guides/`

### 其他相关项目
- **MineDojo**: https://docs.minedojo.org/
- **VPT**: https://github.com/openai/Video-Pre-Training

---

**文档创建**: 2025-10-28  
**最后验证**: 2025-10-28  
**维护者**: AIMC Project  
**状态**: ✅ MineRL 1.0.0 完全可用，文档已整理完毕
