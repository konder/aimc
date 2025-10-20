# 快速开始：获得木头 MVP

最简单的MineDojo+MineCLIP训练示例，2-4小时完成！

---

## 🎯 任务说明

**目标**: 训练智能体学会砍树并获得1个木头

**使用技术**:
- ✅ MineDojo内置任务：`harvest_1_log`
- ✅ MineCLIP密集奖励：3-5倍训练加速
- ✅ PPO算法：成熟的强化学习算法

**预期结果**:
- 📊 成功率：80-90%
- ⏱️ 训练时间：2-4小时（使用MineCLIP）
- 💾 训练步数：200K步

---

## ⚡ 一分钟快速开始

```bash
# 1. 快速测试（5-10分钟，验证环境）
./scripts/train_get_wood.sh test --mineclip

# 2. 标准训练（2-4小时，获得可用模型）
./scripts/train_get_wood.sh --mineclip

# 3. 查看训练进度（另一个终端）
tensorboard --logdir logs/tensorboard

# 4. 评估模型
python scripts/evaluate_get_wood.py
```

---

## 📋 详细步骤

### 步骤1：环境准备

```bash
# 确保在项目目录
cd /Users/nanzhang/aimc

# 激活MineDojo环境
conda activate minedojo

# 验证安装
python -c "import minedojo; print('✓ MineDojo可用')"
```

### 步骤2：快速测试

```bash
# 运行快速测试（10K步，5-10分钟）
./scripts/train_get_wood.sh test --mineclip
```

**预期输出**:
```
========================================
MineDojo 获得木头训练
========================================
任务:       harvest_1_log (获得1个原木)
模式:       test
总步数:     10000
MineCLIP:   --use-mineclip
设备:       mps
========================================

创建环境: harvest_1_log (获得1个原木)
  图像尺寸: (160, 256)
  MineCLIP: 启用
  MineCLIP包装器:
    任务描述: chop down a tree and collect one wood log
    稀疏权重: 10.0
    MineCLIP权重: 0.1
    状态: ✓ 已启用

[1/4] 创建环境...
  ✓ 环境创建成功

[2/4] 创建PPO模型...
  ✓ 模型创建成功 (参数量: 2,543,619)

[3/4] 设置训练回调...
  ✓ 回调设置完成

[4/4] 开始训练...
========================================

[100步] ep_rew_mean: 0.05
[1000步] ep_rew_mean: 0.12
...
```

### 步骤3：标准训练

如果测试通过，开始标准训练：

```bash
# 标准训练（200K步，2-4小时）
./scripts/train_get_wood.sh --mineclip
```

### 步骤4：监控训练

在另一个终端：

```bash
# 启动TensorBoard
tensorboard --logdir logs/tensorboard

# 浏览器打开
# http://localhost:6006
```

**关键指标**:
- `rollout/ep_rew_mean` - 平均奖励（应该上升）
- `train/policy_loss` - 策略损失
- `rollout/ep_len_mean` - episode长度

### 步骤5：评估模型

训练完成后：

```bash
# 评估模型性能
python scripts/evaluate_get_wood.py

# 评估特定检查点
python scripts/evaluate_get_wood.py --model checkpoints/get_wood/get_wood_50000_steps.zip
```

**预期输出**:
```
========================================
MineDojo 获得木头模型评估
========================================
模型: checkpoints/get_wood/get_wood_final.zip
Episodes: 10
最大步数: 2000
========================================

[1/3] 加载模型...
  ✓ 模型加载成功

[2/3] 创建评估环境...
  ✓ 环境创建成功

[3/3] 运行 10 个评估episodes...

  Episode  1/10: ✓ 成功! (步数:  324, 奖励: 1.00)
  Episode  2/10: ✓ 成功! (步数:  156, 奖励: 1.00)
  Episode  3/10: ✗ 失败 (步数: 2000, 奖励: 0.00)
  ...

========================================
评估结果
========================================
总Episodes: 10
成功次数: 8
成功率: 80.0%

平均奖励: 0.800 ± 0.400
平均步数: 542.3 ± 612.1
成功时平均步数: 267.5 ± 143.2
最快成功: 156 步
最慢成功: 524 步
========================================

性能评级:
  良好 ⭐⭐⭐⭐
```

---

## 🔧 常用命令

### 训练相关

```bash
# 快速测试
./scripts/train_get_wood.sh test --mineclip

# 标准训练
./scripts/train_get_wood.sh --mineclip

# 长时间训练
./scripts/train_get_wood.sh long --mineclip

# 自定义步数
./scripts/train_get_wood.sh --timesteps 300000 --mineclip

# 使用GPU
./scripts/train_get_wood.sh --mineclip --device cuda

# 不使用MineCLIP（慢，不推荐）
./scripts/train_get_wood.sh
```

### 监控相关

```bash
# TensorBoard
tensorboard --logdir logs/tensorboard

# 实时日志
tail -f logs/training/training_*.log

# 查看检查点
ls -lh checkpoints/get_wood/
```

### 评估相关

```bash
# 评估最终模型
python scripts/evaluate_get_wood.py

# 评估特定模型
python scripts/evaluate_get_wood.py --model checkpoints/get_wood/get_wood_100000_steps.zip

# 更多episodes
python scripts/evaluate_get_wood.py --episodes 20
```

---

## 📊 预期时间线

### 使用MineCLIP（推荐）

| 步数 | 时间 | 里程碑 |
|------|------|--------|
| 10K | 5-10分钟 | 测试完成，验证环境 |
| 20-50K | 20-40分钟 | 首次成功获得木头 |
| 100K | 1-2小时 | 成功率约50% |
| 200K | 2-4小时 | 成功率约80%，可以使用 |
| 500K | 5-10小时 | 成功率约90%，性能优秀 |

### 不使用MineCLIP

| 步数 | 时间 | 里程碑 |
|------|------|--------|
| 100K | 1-3小时 | 可能还未成功 |
| 200K | 3-6小时 | 首次成功 |
| 500K | 8-16小时 | 成功率约60% |
| 1M+ | 16+小时 | 成功率约70-80% |

**结论**：MineCLIP加速约**3-5倍**！

---

## 💡 故障排除

### Q: 环境创建失败

```bash
# 检查Java
java -version  # 需要Java 8+

# 重新安装MineDojo
pip install --upgrade minedojo

# 设置无头模式
export JAVA_OPTS="-Djava.awt.headless=true"
```

### Q: MineCLIP不工作

```bash
# 确保首次使用时有网络（下载模型）
# 检查模型是否下载
ls -lh ~/.minedojo/models/

# 首次会自动下载（约250-350MB）
```

### Q: 训练太慢

```bash
# 1. 确保使用MineCLIP
./scripts/train_get_wood.sh --mineclip

# 2. 使用GPU（如果有）
./scripts/train_get_wood.sh --mineclip --device cuda

# 3. 减少图像尺寸（可能影响性能）
python src/training/train_get_wood.py --image-size 120 160 --use-mineclip
```

### Q: 模型不学习

```bash
# 检查TensorBoard中的ep_rew_mean
# 如果长时间为0：

# 1. 增加探索
python src/training/train_get_wood.py --use-mineclip --ent-coef 0.02

# 2. 训练更长时间
./scripts/train_get_wood.sh long --mineclip

# 3. 确保MineCLIP已启用
# 训练日志中应该看到 "MineCLIP: 启用"
```

---

## 📚 下一步

训练成功后，你可以：

### 1. 训练更多技能

```bash
# 编辑 src/training/train_get_wood.py
# 修改 task_id 为其他MineDojo内置任务：

task_id="harvest_8_log"     # 采集8个木头（更难）
task_id="harvest_1_milk"    # 采集牛奶
task_id="harvest_1_apple"   # 采集苹果
task_id="harvest_1_wheat"   # 采集小麦
```

### 2. 微调超参数

```bash
# 调整学习率
python src/training/train_get_wood.py --use-mineclip --learning-rate 0.0001

# 调整探索
python src/training/train_get_wood.py --use-mineclip --ent-coef 0.02

# 调整MineCLIP权重
# 编辑 train_get_wood.py 中的 mineclip_weight=0.1
```

### 3. 查看完整文档

- 📖 [获得木头训练指南](docs/guides/GET_WOOD_TRAINING_GUIDE.md) - 详细指南
- 🧠 [MineCLIP详解](docs/guides/MINECLIP_EXPLAINED.md) - 理解原理
- 💡 [高级训练方案](docs/guides/ADVANCED_TRAINING_SOLUTIONS.md) - 进阶技巧
- ❓ [常见问题FAQ](docs/FAQ.md) - 快速答疑

---

## ✅ 检查清单

开始训练前确认：

- [ ] MineDojo已安装：`python -c "import minedojo"`
- [ ] 项目目录正确：`cd /Users/nanzhang/aimc`
- [ ] 脚本有执行权限：`chmod +x scripts/*.sh`
- [ ] 有足够磁盘空间：至少10GB
- [ ] （可选）GPU可用：`python -c "import torch; print(torch.cuda.is_available())"`

训练中确认：

- [ ] MineCLIP已启用：日志中看到 "MineCLIP: 启用"
- [ ] TensorBoard运行：http://localhost:6006 可访问
- [ ] 奖励上升：`ep_rew_mean` 逐渐增加
- [ ] 定期保存：`checkpoints/get_wood/` 有新文件

训练后确认：

- [ ] 评估成功率 > 70%
- [ ] 模型文件存在：`checkpoints/get_wood/get_wood_final.zip`
- [ ] TensorBoard图表正常

---

## 🎉 成功标志

当你看到以下情况，说明训练成功：

1. **评估成功率 ≥ 80%**
2. **平均成功步数 < 500步**
3. **TensorBoard中ep_rew_mean稳定上升**
4. **模型能在测试中多次成功获得木头**

恭喜！你已经掌握了MineDojo+MineCLIP的基础训练流程！

---

## 📞 获取帮助

- 📖 完整文档：`docs/guides/GET_WOOD_TRAINING_GUIDE.md`
- ❓ 常见问题：`docs/FAQ.md`
- 🔧 诊断工具：`python scripts/validate_install.py`

---

**立即开始**：
```bash
./scripts/train_get_wood.sh test --mineclip
```

祝训练成功！🚀

