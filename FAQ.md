# 常见问题解答（FAQ）

## DAgger 训练相关

### Q1: 什么是 DAgger？和传统 RL 有什么区别？

**A**: DAgger (Dataset Aggregation) 是一种**迭代式模仿学习**算法。

**核心区别**:

| 特性 | 传统RL（PPO） | DAgger |
|------|-------------|--------|
| **数据来源** | 随机探索 | 人类演示 |
| **训练起点** | 随机策略 | 专家策略 |
| **首次成功** | 50K-200K步（几小时） | 5-10个演示（1小时录制） |
| **最终成功率** | 80-85% | **90-95%** |
| **鲁棒性** | 中等 | **高**（见过失败场景） |
| **调试难度** | 高（奖励函数设计） | 低（直观的人类演示） |

**工作流程**:
```
录制演示 → BC基线(60%) → 
迭代1(75%) → 迭代2(85%) → 迭代3(92%+)
```

详见：`docs/guides/DAGGER_COMPREHENSIVE_GUIDE.md`

---

### Q2: 需要录制多少专家演示？

**A**: 根据任务复杂度：

| 任务复杂度 | 演示数量 | 总帧数 | 录制时间 | BC成功率 |
|-----------|---------|--------|---------|---------|
| 简单（砍树）| **10-20次** | 5K-10K | 40-60分钟 | 50-70% |
| 中等（建造）| 30-50次 | 20K-30K | 2-3小时 | 40-60% |
| 复杂（探险）| 50-100次 | 50K-100K | 4-6小时 | 30-50% |

**关键点**:
- ✅ **数据质量 > 数量**: 保持一致的操作习惯
- ✅ **多样性很重要**: 覆盖不同场景（近/远距离、不同地形）
- ✅ **确保成功**: 每次演示都要完成任务
- ❌ **避免过度复杂**: 不要绕圈、多余跳跃等

**推荐**:
```bash
# 先录制 5 个测试质量
bash scripts/run_dagger_workflow.sh \
    --task harvest_1_log \
    --num-episodes 5 \
    --iterations 0

# 如果质量好，追加到 15-20 个
bash scripts/run_dagger_workflow.sh \
    --task harvest_1_log \
    --num-episodes 20 \
    --append-recording \
    --iterations 0
```

---

### Q3: 标注太慢怎么办？一次要标注几百个状态！

**A**: 使用**智能策略**，标注速度提升 **60%**：

#### **技巧1: 智能采样**

```bash
# 使用 --smart-sampling（只标注20-30%关键状态）
python tools/dagger/label_states.py \
    --states data/tasks/harvest_1_log/policy_states/iter_1/ \
    --output data/tasks/harvest_1_log/expert_labels/iter_1.pkl \
    --smart-sampling \
    --failure-window 5  # 只标注失败前5步
```

**效果**: 500个状态 → 只需标注100-150个

#### **技巧2: 善用P键** ⭐

| 按键 | 使用场景 | 耗时 |
|------|---------|------|
| **P** | 策略正确，保持不变 | ~1秒 |
| W/F等 | 需要修改动作 | ~3秒 |

**示例**:
```
状态1: 策略=前进 → P (策略对了，保持) ✓ 1秒
状态2: 策略=前进 → P ✓ 1秒
状态3: 策略=IDLE → W (应该前进) ✓ 3秒
状态4: 策略=前进+攻击 → P ✓ 1秒
状态5: 策略=前进+攻击 → P ✓ 1秒
```

**P键使用率**: 应该达到 30-60%

#### **技巧3: 跳过重复帧**

```
帧1: 向左看 → J ✓
帧2-4: (画面变化很小) → N, N, N (跳过)
帧5: 看到树了 → W (前进) ✓
```

#### **技巧4: 标注优先级**

| 优先级 | 场景 | 标注比例 |
|-------|------|---------|
| 🔴 高 | 失败前5步 | 100% |
| 🟡 中 | 偏离轨迹 | 50% |
| 🟢 低 | 正常执行 | 10% |

**标注速度对比**:
- 全手动标注: ~5秒/状态 → 500个状态 = **40分钟**
- 使用技巧: ~2秒/状态 → 150个状态 = **5分钟** ✅

---

### Q4: DAgger 迭代没有提升，还是60%成功率？

**A**: 检查以下几点：

#### **1. 标注质量检查**

```bash
# 统计你的标注分布
# 标注100个状态后，检查：

W (前进):          40次 (40%) ✅ 好
Q (前进+攻击):     15次 (15%) ✅ 好
F (攻击):          10次 (10%) ✅ 好
P (保持策略):      20次 (20%) ✅ 好
J/L (左右看):      12次 (12%) ⚠️ 有点多
I/K (上下看):      2次  (2%)  ✅ 好
N (跳过):          1次  (1%)  ✅ 好
```

**健康分布**:
- ✅ 前进相关（W+Q+R）: 50-70%
- ✅ 攻击相关（F+Q+G）: 20-40%
- ✅ 视角调整（I/J/K/L）: **<15%** ⭐ 关键
- ✅ 保持策略（P）: 20-40%

**不健康分布**:
- ❌ 视角调整 > 30% → 模型会原地转圈
- ❌ 前进 < 40% → 模型不知道要前进
- ❌ P键使用 < 10% → 说明策略质量很差或你过度干预

#### **2. 常见标注错误**

**错误1: 连续标注视角调整** ❌
```
帧1-5: L, L, L, L, L (连续向右看)
结果: 模型学会原地转圈
```

**正确做法** ✅:
```
帧1: L (向右看，1帧)
帧2-5: W, W, P, P (立即切换回前进)
结果: 模型学会前进+偶尔环视
```

**错误2: 不使用P键** ❌
```
策略: Forward → 你输入: W (重复输入)
策略: Forward → 你输入: W
策略: Attack → 你输入: F
```

**正确做法** ✅:
```
策略: Forward → P (保持)
策略: Forward → P (保持)
策略: Attack → P (保持)
```

#### **3. 调整参数**

```bash
# 收集更多失败场景
bash scripts/run_dagger_workflow.sh \
    --collect-episodes 30 \  # 从20增加到30
    --skip-recording \
    --skip-bc

# 增加BC训练轮数
--bc-epochs 100  # 从50增加到100
```

#### **4. 重新标注**

如果发现标注质量差，可以重新标注：

```bash
# 删除质量差的标注
rm data/tasks/harvest_1_log/expert_labels/iter_1.pkl

# 重新标注，使用新策略
python tools/dagger/label_states.py \
    --states data/tasks/harvest_1_log/policy_states/iter_1/ \
    --output data/tasks/harvest_1_log/expert_labels/iter_1.pkl \
    --smart-sampling \
    --failure-window 5
```

---

### Q5: 模型一直原地转圈，很少前进？

**A**: 这是**典型的标注问题** - 视角调整过多

#### **问题诊断**

回顾你的标注：
```
# 如果你经常这样标注：
看不到树 → J (向左看)
画面没变 → J (继续左看)
还是没变 → J (继续左看)
终于看到了 → W (前进)

# 结果：模型学到 "看不到树 = 一直转头"
```

#### **解决方案**

**标注原则**: **环视是短期行为（1-2帧），移动是主要策略（>60%）**

```
# 正确标注：
看不到树，策略=前进
→ J (向左看，只1帧！)

画面开始变化
→ W (立即切换回前进)

继续前进
→ W 或 P

看到树了
→ W (继续靠近)
```

**检查标注比例**:
```bash
# 如果你的标注中：
视角调整(I/J/K/L) > 30% → ❌ 太多了！重新标注
前进(W/Q/R) < 50% → ❌ 太少了！重新标注

视角调整 < 15% → ✅ 正常
前进 > 60% → ✅ 健康
```

#### **重新标注并重新训练**

```bash
# 1. 删除旧标注
rm data/tasks/harvest_1_log/expert_labels/iter_1.pkl

# 2. 重新标注（使用"前进优先"原则）
python tools/dagger/label_states.py \
    --states data/tasks/harvest_1_log/policy_states/iter_1/ \
    --output data/tasks/harvest_1_log/expert_labels/iter_1.pkl \
    --smart-sampling

# 3. 重新训练
python src/training/train_dagger.py \
    --iteration 1 \
    --base-data data/tasks/harvest_1_log/expert_demos/ \
    --new-data data/tasks/harvest_1_log/expert_labels/iter_1.pkl \
    --output data/tasks/harvest_1_log/checkpoints/dagger_iter_1.zip
```

---

### Q6: 鼠标录制和键盘录制哪个更好？

**A**: **强烈推荐鼠标录制** ⭐

**性能对比**:

| 特性 | 键盘控制 (I/J/K/L) | Pygame鼠标控制 |
|------|-------------------|--------------|
| 视角控制 | 离散（固定角度） | ✅ 连续平滑 |
| 攻击操作 | F键 | ✅ 鼠标左键（更自然） |
| 静态帧占比 | 28.5% ❌ | **<20%** ✅ |
| 多键检测 | 不支持 ❌ | ✅ W+左键同时 |
| FPS玩家友好 | 需要适应 | ✅ 立即上手 |
| 数据质量 | 中等 | **高（4-5倍提升）** ✅ |

**使用鼠标录制**:
```bash
bash scripts/run_minedojo_x86.sh python tools/dagger/record_manual_chopping_pygame.py \
    --base-dir data/tasks/harvest_1_log/expert_demos \
    --max-frames 1000 \
    --mouse-sensitivity 0.5
```

**控制说明**:
- 🖱️ 鼠标移动: 转动视角
- 🖱️ 鼠标左键: 攻击/挖掘
- ⌨️ W/A/S/D: 移动
- ⌨️ Space: 跳跃
- ⌨️ Q: 重试
- ⌨️ ESC: 退出

**鼠标灵敏度调整**:
- 新手: `0.3`（慢速，精确）
- 默认: `0.5`（推荐）
- 熟练: `0.8`（快速）

---

## 部署和环境相关

### Q7: Apple M 芯片如何部署？

**A**: 需要通过 Rosetta 2 运行 x86 版本的 MineDojo

#### **快速部署**

```bash
# 1. 安装 Rosetta 2
softwareupdate --install-rosetta --agree-to-license

# 2. 安装 x86 Java
arch -x86_64 brew install temurin@8
export JAVA_HOME=/Library/Java/JavaVirtualMachines/temurin-8.jdk/Contents/Home/

# 3. 在 x86 模式下创建环境
arch -x86_64 /bin/bash
conda create -n minedojo-x86 python=3.9 -y
conda activate minedojo-x86

# 4. 安装 MineDojo
pip install "pip<24.1" "setuptools<58" "wheel<0.38.0"
pip install "numpy>=1.21.0,<2.0"
pip install minedojo

# 5. 安装项目依赖
cd /path/to/aimc
pip install -r requirements.txt

# 6. 验证
python tools/validate_install.py
```

#### **便捷运行方式**

```bash
# 方法1: 使用项目脚本（推荐）
./scripts/run_minedojo_x86.sh python tools/validate_install.py
./scripts/run_minedojo_x86.sh bash scripts/run_dagger_workflow.sh --task harvest_1_log

# 方法2: 手动切换
arch -x86_64 /bin/bash
conda activate minedojo-x86
python tools/validate_install.py
```

#### **M 芯片注意事项**

- ✅ GPU加速: 使用 MPS（Metal Performance Shaders）
- ✅ 性能: M1/M2/M3 接近或超过中端 GPU
- ⚠️ 每次运行都需要 x86 模式
- ⚠️ 首次编译 Minecraft 需要10-30分钟

**详细步骤**: 参考 README.md 的"Apple M 芯片部署"章节

---

### Q8: Docker 部署有什么优势？

**A**: 环境隔离 + 一键部署

**优势**:
- ✅ 环境隔离（不影响主机）
- ✅ 一键部署（无需手动配置）
- ✅ 可复现性（环境完全一致）
- ✅ 易于分享（打包镜像）

**快速部署**:
```bash
# 1. 构建镜像
cd docker
docker build --platform linux/amd64 -t aimc-minedojo:latest .

# 2. 运行容器
docker run -it --rm \
  --platform linux/amd64 \
  -v $(pwd):/workspace \
  aimc-minedojo:latest

# 3. 在容器中运行
python tools/validate_install.py
bash scripts/run_dagger_workflow.sh --task harvest_1_log
```

**网络受限环境**: 参考 `docker/README.md` 获取离线部署方案

---

### Q9: 环境创建失败怎么办？

**A**: 检查以下几点：

#### **1. 检查 Java**

```bash
# 验证 Java 版本（需要 Java 8+）
java -version

# 如果没有，安装 Java
# Ubuntu/Debian
sudo apt-get install openjdk-8-jdk

# macOS (Intel)
brew install openjdk@8

# macOS (M芯片)
arch -x86_64 brew install temurin@8
```

#### **2. 设置环境变量**

```bash
# 设置 JAVA_HOME
export JAVA_HOME=/path/to/java

# 设置无头模式（提升性能）
export JAVA_OPTS="-Djava.awt.headless=true"

# 添加到 ~/.bashrc 或 ~/.zshrc
echo 'export JAVA_HOME=/path/to/java' >> ~/.zshrc
echo 'export JAVA_OPTS="-Djava.awt.headless=true"' >> ~/.zshrc
source ~/.zshrc
```

#### **3. 重新安装 MineDojo**

```bash
# 卸载
pip uninstall minedojo -y

# 重新安装
pip install minedojo

# 首次运行会自动下载和编译 Minecraft
python -c "import minedojo; env = minedojo.make('harvest_1_log'); env.reset(); env.close()"
```

---

### Q10: 如何安装 MineRL？

**A**: 根据需求选择安装方式

#### **PyPI vs Git 安装**

| 方式 | 版本 | 稳定性 | 使用场景 |
|------|------|--------|---------|
| PyPI | 0.4.4 | ✅ 稳定 | **生产环境（推荐）** |
| Git | master/dev | ⚠️ 开发中 | 开发/测试 |

#### **方式 1: 从 PyPI 安装（推荐）**

```bash
# 激活环境
conda activate minedojo

# 安装 MineRL 0.4.4（最新稳定版）
./scripts/install_minerl_patched.sh 0.4.4

# 验证安装
python tools/test_minerl_quick.py
```

#### **方式 2: 从 Git 安装（最新代码）**

```bash
# 安装 master 分支（最新）
./scripts/install_minerl_from_git.sh

# 或安装 dev 分支
./scripts/install_minerl_from_git.sh dev

# 验证安装
python tools/test_minerl_quick.py
```

#### **为什么需要特殊处理？**

直接 `pip install minerl` 会在临时目录编译 Minecraft，由于以下问题导致失败：
- ❌ Gradle 下载速度慢（国外源）
- ❌ Maven 仓库访问不稳定
- ❌ build.gradle 存在相对路径 bug

我们的脚本自动应用国内镜像补丁解决这些问题。

**脚本自动完成**:
- ✅ 下载 MineRL 源码
- ✅ 修改 Gradle 为阿里云镜像
- ✅ 修改 Maven 为国内镜像
- ✅ 修复相对路径 bug
- ✅ 安装到当前环境

**预计耗时**: 5-10 分钟

#### **手动安装（完全控制）**

```bash
# 1. 下载源码
pip download --no-deps --no-binary :all: minerl==0.4.4
tar -xzf minerl-0.4.4.tar.gz
cd minerl-0.4.4

# 2. 应用补丁
patch -p1 < /path/to/aimc/docker/minerl_build.patch

# 3. 安装
pip install .
```

#### **首次运行注意**

即使安装成功，首次运行时仍需构建 Minecraft（10-30分钟）：

```python
import minerl
env = minerl.make("MineRLTreechop-v0")  # 首次会触发构建
```

**故障排除**:
- Gradle 下载超时 → 检查补丁是否正确应用
- Maven 依赖失败 → 检查 build.gradle 仓库配置
- Java 内存不足 → `export GRADLE_OPTS="-Xmx4g"`

**详细文档**:
- 📖 完整指南: `docs/guides/MINERL_INSTALLATION_GUIDE.md`
- ⚡ 快速参考: `docs/reference/MINERL_INSTALLATION_REFERENCE.md`
- 🔧 Patch 文件: `docker/minerl_build.patch`

**MineRL vs MineDojo**:
- MineRL: 主要用于数据集处理和特定场景
- MineDojo: 本项目主要使用的环境
- 安装方式: MineRL 需要打补丁，MineDojo 可直接安装

---

### Q11: 内存不足怎么办？

**A**: 优化方案：

```bash
# 1. 减少并行环境（训练时）
python src/training/train_bc.py \
    --data data/tasks/harvest_1_log/expert_demos/ \
    --n-envs 1  # 默认可能是4

# 2. 减少批次大小
python src/training/train_bc.py \
    --data data/tasks/harvest_1_log/expert_demos/ \
    --batch-size 16  # 默认是32或64

# 3. 减少图像尺寸（影响性能）
# 修改环境创建参数
image_size=(120, 160)  # 默认是(160, 256)

# 4. 关闭不必要的程序
# 确保有至少 8GB 可用内存
```

---

### Q12: MineRL saves 目录占用大量空间怎么办？

**A**: MineRL 每次运行都会在 `MCP-Reborn/saves/` 目录生成世界存档，需要定期清理。

#### **问题说明**

**Saves 目录位置**:
```
/usr/local/Caskroom/miniforge/base/envs/minedojo-x86/lib/python3.9/site-packages/minerl/MCP-Reborn/saves/
```

**累积原因**:
- 每次 `env.reset()` 都可能创建新世界
- 每个世界 10-100+ MB
- 长期训练会产生数百个世界，占用 GB 级空间

#### **解决方案 1: 查看当前大小**

```bash
# 使用项目工具
bash scripts/clean_minerl_saves.sh --show-size

# 或手动检查
du -sh /usr/.../minerl/MCP-Reborn/saves
```

#### **解决方案 2: 手动清理**

```bash
# 预览会删除什么（不实际删除）
bash scripts/clean_minerl_saves.sh --dry-run

# 删除所有世界
bash scripts/clean_minerl_saves.sh

# 保留最新的 2 个世界
bash scripts/clean_minerl_saves.sh --keep-latest 2
```

#### **解决方案 3: 自动清理**

```bash
# 只在超过 500MB 时清理
bash scripts/clean_minerl_saves.sh --auto --threshold 500

# 在训练脚本开头添加这一行
```

#### **解决方案 4: Python 代码中自动清理**

```python
from src.utils.minerl_cleanup import auto_clean_if_needed

# 训练前检查清理
auto_clean_if_needed(threshold_mb=1000, keep_latest=2)

# 正常创建环境
env = gym.make('MineRLBasaltFindCave-v0')
# ...
```

#### **MCP-Reborn 和 Malmo 的关系**

- **MCP-Reborn**: MineRL 使用的 Minecraft 客户端（本项目当前使用）
  - 位置: `minerl/MCP-Reborn/`
  - 为 MineRL 框架提供标准化环境
  
- **Malmo**: MineDojo 使用的 Minecraft AI 平台（已安装但未使用）
  - 位置: `minedojo/sim/Malmo/Minecraft/`
  - 微软开发的官方 Minecraft AI Mod

两者都会生成 `saves/` 目录，当前项目使用 MineRL，所以需要清理 `MCP-Reborn/saves/`。

**详细文档**: `docs/guides/MINERL_SAVES_CLEANUP_GUIDE.md`

---

## 数据管理相关

### Q13: 如何追加录制更多数据？

**A**: 使用 `--append-recording` 参数

```bash
# 第一次录制了 10 个 episodes
bash scripts/run_dagger_workflow.sh \
    --task harvest_1_log \
    --num-episodes 10 \
    --iterations 3

# BC 成功率只有 40%，想补录到 20 个
bash scripts/run_dagger_workflow.sh \
    --task harvest_1_log \
    --num-episodes 20 \
    --append-recording \
    --skip-bc

# 数据变化:
# 录制前: episode_000 ~ episode_009 (10个)
# 录制后: episode_000 ~ episode_019 (20个)

# 重新训练 BC
python src/training/train_bc.py \
    --data data/tasks/harvest_1_log/expert_demos/ \
    --output data/tasks/harvest_1_log/checkpoints/bc_baseline.zip \
    --epochs 50
```

**注意**:
- ✅ `--append-recording` 会保留已有数据
- ✅ 从下一个编号开始录制
- ❌ 不使用该参数会提示是否覆盖

---

### Q14: 可以删除中间数据吗？

**A**: 可以，但要注意保留顺序

#### **可以删除的数据**

```bash
# 1. 删除 policy_states（收集的状态）
rm -rf data/tasks/harvest_1_log/policy_states/iter_1/
rm -rf data/tasks/harvest_1_log/policy_states/iter_2/

# 2. 删除中间模型（保留最新的）
rm data/tasks/harvest_1_log/checkpoints/dagger_iter_1.zip
rm data/tasks/harvest_1_log/checkpoints/dagger_iter_2.zip
# 保留: bc_baseline.zip, dagger_iter_3.zip

# 3. 删除评估结果
rm data/tasks/harvest_1_log/checkpoints/*_eval_results.npy
```

#### **不建议删除的数据**

```bash
# 1. 专家演示（BC训练需要）
data/tasks/harvest_1_log/expert_demos/

# 2. 标注数据（重新训练需要）
data/tasks/harvest_1_log/expert_labels/

# 3. 聚合数据（继续训练需要）
data/tasks/harvest_1_log/dagger/combined_iter_*.pkl

# 4. 最终模型
data/tasks/harvest_1_log/checkpoints/dagger_iter_3.zip
```

#### **完全清理一个任务**

```bash
# 删除特定任务的所有数据
rm -rf data/tasks/harvest_1_log/expert_demos/
rm -rf data/tasks/harvest_1_log/policy_states/
rm -rf data/tasks/harvest_1_log/expert_labels/
rm -rf data/tasks/harvest_1_log/dagger/
rm -rf data/tasks/harvest_1_log/checkpoints/
```

---

### Q15: 多任务的数据会互相干扰吗？

**A**: 不会，每个任务有独立的目录

**目录结构**:
```
data/tasks/
├── harvest_1_log/          # 任务1: 砍树
│   ├── expert_demos/       # 专家演示
│   │   ├── episode_000/
│   └── ...
└── harvest_1_wool/         # 任务2: 获取羊毛
    ├── episode_000/
    └── ...

data/tasks/harvest_1_log/
├── checkpoints/            # 任务1的模型
│   ├── bc_baseline.zip
│   └── ...
└── harvest_1_wool/         # 任务2的模型
    ├── bc_baseline.zip
    └── ...
```

**并行训练**:
```bash
# 同时训练多个任务（不同终端）
# 终端1
bash scripts/run_dagger_workflow.sh --task harvest_1_log --iterations 3

# 终端2
bash scripts/run_dagger_workflow.sh --task harvest_1_wool --iterations 3

# 数据和模型完全独立，互不影响
```

---

## 训练和评估相关

### Q16: 如何继续训练更多轮 DAgger？

**A**: 使用 `--continue-from` 参数

```bash
# 已经完成了3轮 DAgger
# 想再训练2轮（总共5轮）

bash scripts/run_dagger_workflow.sh \
    --task harvest_1_log \
    --continue-from data/tasks/harvest_1_log/checkpoints/dagger_iter_3.zip \
    --iterations 5  # 总轮数（不是新增轮数）

# 会自动：
# - 从 dagger_iter_3.zip 开始
# - 执行迭代 4 和 5
# - 生成 dagger_iter_4.zip 和 dagger_iter_5.zip
```

**自动推断起始迭代**:
```bash
# 不需要指定 --start-iteration
# 脚本会从文件名自动推断
data/tasks/harvest_1_log/checkpoints/dagger_iter_3.zip
→ 自动检测: 从 iter_4 开始
```

---

### Q17: 如何查看训练历史和对比模型？

**A**: 查看评估结果文件

```bash
# 1. 查看所有模型
ls -lh data/tasks/harvest_1_log/checkpoints/

# 2. 查看评估结果
python -c "
import numpy as np
results = np.load('data/tasks/harvest_1_log/checkpoints/bc_baseline_eval_results.npy', allow_pickle=True).item()
print(f'BC基线: {results[\"success_rate\"]*100:.1f}%')

results = np.load('data/tasks/harvest_1_log/checkpoints/dagger_iter_1_eval_results.npy', allow_pickle=True).item()
print(f'迭代1: {results[\"success_rate\"]*100:.1f}%')
"

# 3. 重新评估所有模型
for model in data/tasks/harvest_1_log/checkpoints/*.zip; do
    echo "评估: $model"
    bash scripts/run_minedojo_x86.sh python tools/dagger/evaluate_policy.py \
        --model "$model" \
        --episodes 20
done
```

**TensorBoard 可视化**:
```bash
# 启动 TensorBoard
tensorboard --logdir logs/tensorboard

# 查看不同模型的训练曲线对比
```

---

## 预训练模型相关

### Q18: 能否使用OpenAI的VPT模型作为预训练模型？

**A**: ✅ **完全可以！而且强烈推荐！**

VPT (Video Pre-Training) 是 OpenAI 专门为 Minecraft 开发的预训练模型，可以显著提升训练效率。

**核心优势**:

| 指标 | 从零训练 | VPT预训练 | 提升 |
|------|---------|----------|------|
| **专家数据需求** | 100回合 | **30-50回合** | -50% |
| **训练时间** | 3-5小时 | **1-2小时** | -60% |
| **BC基线成功率** | 60% | **75-80%** | +25% |
| **最终成功率** | 85-90% | **90-95%** | +8% |

**100个回合够用吗？**
- ✅ **绝对够用！甚至过量！**
- VPT微调通常只需 **10-50个回合**
- 100个回合可以分配：
  - 50个用于BC微调
  - 30个用于DAgger迭代1
  - 20个用于DAgger迭代2

**快速开始**:

```bash
# 1. 下载VPT模型（5分钟）
mkdir -p data/pretrained/vpt
cd data/pretrained/vpt
wget https://openaipublic.blob.core.windows.net/minecraft-rl/models/rl-from-early-game-2x.model
wget https://openaipublic.blob.core.windows.net/minecraft-rl/models/rl-from-early-game-2x.weights

# 2. 安装VPT库
pip install git+https://github.com/openai/Video-Pre-Training.git

# 3. 测试零样本性能（无需微调）
bash scripts/run_minedojo_x86.sh python tools/test_vpt_zero_shot.py \
    --model data/pretrained/vpt/rl-from-early-game-2x.model \
    --task harvest_1_log \
    --episodes 5

# 预期：20-40%成功率（相比从零的0%）

# 4. 使用现有专家数据微调
python src/training/train_bc_with_vpt.py \
    --vpt-model data/pretrained/vpt/rl-from-early-game-2x.model \
    --data data/tasks/harvest_1_log/expert_demos/ \
    --output data/tasks/harvest_1_log/checkpoints/vpt_finetuned.zip \
    --epochs 10

# 预期：75-80%成功率（相比BC的60%）
```

**推荐工作流**:

```
方案1: VPT + BC微调
  - 录制20-30个专家演示
  - 微调VPT（10-15分钟）
  - 成功率: 75-80%

方案2: VPT + BC + DAgger（最佳）⭐
  - 录制30-50个专家演示
  - 微调VPT → BC基线（75-80%）
  - 1-2轮DAgger迭代 → 90-95%
  - 总时间: 1-2小时（相比原来的3-5小时）
```

**为什么推荐VPT？**

1. ✅ **已掌握基础技能**: 移动、转视角、挖掘等
2. ✅ **探索效率高**: 知道如何导航，不会随机探索
3. ✅ **动作分布合理**: 接近人类玩家
4. ✅ **微调速度快**: 5-10倍加速
5. ✅ **成功率更高**: 基线提升15-20%

**详细文档**:
- 📖 **完整分析**: `docs/technical/VPT_INTEGRATION_ANALYSIS.md`
- 🚀 **快速开始**: `docs/guides/VPT_QUICKSTART_GUIDE.md`
- 💻 **示例代码**: `tmp/vpt_integration_example.py`

**VPT模型选择**:

| 模型 | 大小 | 性能 | 推荐场景 |
|------|------|------|---------|
| `rl-from-early-game-2x` | ~50MB | 高 | ✅ 砍树、挖矿等基础任务（推荐） |
| `rl-from-house-2x` | ~50MB | 中 | 房屋内任务 |
| `foundation-model-1x` | ~400MB | 最高 | 复杂任务、多技能组合 |

**状态**: VPT集成已在长期计划中，目前提供完整实施方案和示例代码

---

## 其他问题

### Q19: 支持哪些 MineDojo 任务？

**A**: 支持所有 MineDojo 程序化任务

**常用任务**:
```bash
# 采集类
harvest_1_log          # 获得1个原木
harvest_10_log         # 获得10个原木
harvest_1_wool         # 获得1个羊毛
harvest_milk           # 获得牛奶

# 挖掘类
harvest_10_cobblestone # 挖10个圆石
harvest_1_iron_ore     # 挖1个铁矿石

# 农业类
harvest_1_wheat        # 收获1个小麦

# 战斗类
combat_spider          # 击败蜘蛛
```

**查看所有任务**:
```bash
python -c "import minedojo; print('\n'.join(minedojo.tasks.ALL_PROGRAMMATIC_TASK_IDS[:30]))"
```

**训练新任务**:
```bash
# 只需修改 --task 参数
bash scripts/run_dagger_workflow.sh \
    --task harvest_1_wool \
    --num-episodes 10 \
    --iterations 3
```

---

### Q20: 在哪里获取更多帮助？

**A**: 

- 📖 **完整教程**: `docs/guides/DAGGER_COMPREHENSIVE_GUIDE.md` （强烈推荐）
- 📑 **任务参考**: `docs/reference/MINEDOJO_TASKS_REFERENCE.md`
- 🎮 **标注参考**: `docs/reference/LABELING_KEYBOARD_REFERENCE.md`
- 🔧 **诊断工具**: `python tools/validate_install.py`
- 💬 **GitHub Issues**: [提交问题](https://github.com/your-repo/aimc/issues)

---

## 快速参考

### 常用命令

```bash
# 完整 DAgger 训练
bash scripts/run_dagger_workflow.sh --task harvest_1_log --num-episodes 10 --iterations 3

# 跳过录制（已有数据）
bash scripts/run_dagger_workflow.sh --task harvest_1_log --skip-recording --iterations 3

# 追加录制
bash scripts/run_dagger_workflow.sh --task harvest_1_log --num-episodes 20 --append-recording --iterations 0

# 继续训练
bash scripts/run_dagger_workflow.sh --task harvest_1_log --continue-from data/tasks/harvest_1_log/checkpoints/dagger_iter_3.zip --iterations 5

# 评估模型
bash scripts/run_minedojo_x86.sh python tools/dagger/evaluate_policy.py --model data/tasks/harvest_1_log/checkpoints/dagger_iter_1.zip --episodes 20

# 验证安装
python tools/validate_install.py
```

### 故障速查

| 问题 | 快速解决 |
|------|----------|
| 未找到数据 | 移除`--skip-recording`或手动录制 |
| BC模型不存在 | 移除`--skip-bc`或手动训练BC |
| 成功率不提升 | 检查标注分布（视角<15%，前进>60%） |
| 模型原地转圈 | 重新标注，使用"前进优先"原则 |
| 标注太慢 | 使用P键 + 智能采样 |
| M芯片环境问题 | 使用 `./scripts/run_minedojo_x86.sh` |
| 内存不足 | 减少batch-size和n-envs |

---

**有其他问题？** 查看完整文档或运行诊断工具！

```bash
# 诊断工具
python tools/validate_install.py

# 完整教程
cat docs/guides/DAGGER_COMPREHENSIVE_GUIDE.md
```
