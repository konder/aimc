# Checkpoint恢复训练指南

> **功能**: 自动保存和恢复训练进度，支持累积训练  
> **版本**: 2025-10-21

---

## 🎯 **核心功能**

### **自动Checkpoint管理**

```bash
训练过程会自动保存checkpoint:
  ├─ 每10000步保存一次: get_wood_10000_steps.zip, get_wood_20000_steps.zip...
  ├─ 训练完成保存: get_wood_final.zip
  └─ 中断时保存: get_wood_interrupted.zip (Ctrl+C)
```

### **默认行为：自动恢复训练**

```bash
# 第1次运行（从头开始）
./scripts/train_get_wood.sh test --mineclip
→ 创建新模型
→ 训练0 → 10000步
→ 保存: get_wood_10000_steps.zip, get_wood_final.zip

# 第2次运行（自动恢复）
./scripts/train_get_wood.sh test --mineclip
→ 检测到: get_wood_10000_steps.zip
→ 加载模型，继续训练
→ 训练10000 → 20000步
→ 保存: get_wood_20000_steps.zip, get_wood_final.zip

# 第3次运行（继续累积）
./scripts/train_get_wood.sh test --mineclip
→ 检测到: get_wood_20000_steps.zip
→ 加载模型，继续训练
→ 训练20000 → 30000步
→ 保存: get_wood_30000_steps.zip, get_wood_final.zip
```

**累积训练进度：**
```
运行次数 | 总步数     | 模型状态
---------|-----------|----------
第1次    | 0→10000   | 新模型
第2次    | 10000→20000 | 继续训练 ✅
第3次    | 20000→30000 | 继续训练 ✅
第4次    | 30000→40000 | 继续训练 ✅
...
第10次   | 90000→100000 | 继续训练 ✅
```

---

## 🚀 **使用示例**

### **场景1：累积训练（推荐）**

```bash
# 运行10次，每次10000步，累积到100000步
for i in {1..10}; do
    echo "=== 第${i}次训练 ==="
    ./scripts/train_get_wood.sh test \
        --task-id harvest_1_log_forest \
        --mineclip --mineclip-weight 40.0 \
        --device cpu --headless
    
    echo "已完成 $((i * 10000)) 步"
    sleep 5
done

# 结果：
# - 模型参数累积更新
# - TensorBoard显示完整100000步的曲线
# - 每10000步有一个checkpoint备份
```

### **场景2：中断恢复**

```bash
# 开始训练
./scripts/train_get_wood.sh --mineclip

# 训练到5000步时，按 Ctrl+C 中断
# → 保存: get_wood_interrupted.zip

# 稍后恢复训练
./scripts/train_get_wood.sh --mineclip
# → 自动检测 get_wood_interrupted.zip
# → 从5000步继续训练
```

### **场景3：从头开始（清空历史）**

```bash
# 方法1：删除旧checkpoint
rm checkpoints/get_wood/*.zip

# 方法2：使用--no-resume参数
./scripts/train_get_wood.sh test --mineclip --no-resume

# 结果：
# → 忽略现有checkpoint
# → 创建全新模型
# → 从0步开始训练
```

---

## 🔍 **Checkpoint检测顺序**

脚本按以下顺序检测checkpoint：

```python
优先级（从高到低）:
1. get_wood_*_steps.zip（最新的步数checkpoint）
   例如: get_wood_30000_steps.zip, get_wood_20000_steps.zip
   → 选择修改时间最新的

2. get_wood_final.zip（最终模型）
   → 如果没有步数checkpoint，使用这个

3. get_wood_interrupted.zip（中断模型）
   → 如果前两者都没有，使用这个

4. 都没有 → 创建新模型
```

### **实际例子**

```bash
情况A：存在多个步数checkpoint
checkpoints/get_wood/
  ├─ get_wood_10000_steps.zip  (2025-10-20 10:00)
  ├─ get_wood_20000_steps.zip  (2025-10-20 12:00)
  └─ get_wood_30000_steps.zip  (2025-10-21 09:00) ← 最新
  
→ 加载: get_wood_30000_steps.zip

情况B：只有final模型
checkpoints/get_wood/
  └─ get_wood_final.zip
  
→ 加载: get_wood_final.zip

情况C：只有中断模型
checkpoints/get_wood/
  └─ get_wood_interrupted.zip
  
→ 加载: get_wood_interrupted.zip
```

---

## ⚙️ **参数说明**

### **Python脚本参数**

```bash
python src/training/train_get_wood.py \
    --resume                    # 自动恢复训练（默认启用）
    --no-resume                 # 强制从头开始
    --checkpoint-dir PATH       # checkpoint保存目录
    --save-freq 10000          # 每10000步保存一次
```

### **Shell脚本参数**

```bash
./scripts/train_get_wood.sh [模式] [选项]

选项:
  --no-resume    强制从头开始，不加载checkpoint
  
# 默认：自动恢复（不需要加参数）
```

---

## 📊 **TensorBoard累积显示**

### **累积训练的TensorBoard曲线**

```bash
启动TensorBoard:
tensorboard --logdir logs/tensorboard --port 6006

查看:
http://localhost:6006

显示内容:
- 完整的训练曲线（跨多次运行）
- ep_rew_mean: 回合奖励趋势
- loss: 损失函数变化
- policy_gradient_loss: 策略梯度

注意：
- 每次恢复训练都会在同一个图表中继续绘制
- 可以看到完整的训练历史
```

---

## 🛠️ **常见问题**

### **Q1: 如何确认checkpoint被加载了？**

```bash
运行时查看输出:

✅ 成功加载checkpoint:
  🔄 检测到checkpoint: get_wood_20000_steps.zip
  ✅ 从checkpoint恢复训练...
  ✓ 模型加载成功，继续训练

❌ 没有checkpoint（新训练）:
  🆕 创建新模型（从头开始）
```

### **Q2: checkpoint文件很大（173MB），正常吗？**

```bash
是的，完全正常！

checkpoint包含:
- 模型参数（CNN策略网络）: ~15M参数
- 优化器状态（Adam）: ~30M参数
- Value网络
- 其他训练状态

总计约173MB是标准大小
```

### **Q3: 可以从某个特定checkpoint恢复吗？**

```bash
方法1：删除其他checkpoint，只保留想要的
rm checkpoints/get_wood/get_wood_30000_steps.zip
# 保留: get_wood_20000_steps.zip
./scripts/train_get_wood.sh --mineclip

方法2：手动重命名
mv checkpoints/get_wood/get_wood_20000_steps.zip \
   checkpoints/get_wood/get_wood_latest_steps.zip
# 脚本会选择最新的
```

### **Q4: 如何清空所有历史，完全重新开始？**

```bash
方法1：删除checkpoint目录
rm -rf checkpoints/get_wood/*.zip

方法2：使用--no-resume
./scripts/train_get_wood.sh test --mineclip --no-resume

方法3：清空TensorBoard日志（可选）
rm -rf logs/tensorboard/PPO_*
```

### **Q5: 多次累积训练后，loss会继续下降吗？**

```bash
理论上：
- 前期（0-50000步）: loss快速下降
- 中期（50000-100000步）: loss缓慢下降
- 后期（100000+步）: loss趋于稳定

如果loss不再下降：
1. 可能已经收敛
2. 可以降低学习率：--learning-rate 0.0001
3. 可以调整MineCLIP权重
```

---

## 💡 **最佳实践**

### **推荐训练流程**

```bash
# 1. 快速测试（10000步）
./scripts/train_get_wood.sh test --mineclip --save-frames
# → 验证环境和参数

# 2. 累积训练（10次 × 10000步 = 100000步）
for i in {1..10}; do
    ./scripts/train_get_wood.sh test --mineclip --mineclip-weight 40.0
    echo "=== 已完成 $((i * 10000)) 步 ==="
done

# 3. 检查TensorBoard
tensorboard --logdir logs/tensorboard --port 6006

# 4. 继续训练（如果需要）
./scripts/train_get_wood.sh quick --mineclip  # 再50000步
```

### **Checkpoint管理**

```bash
# 定期备份重要checkpoint
cp checkpoints/get_wood/get_wood_50000_steps.zip \
   backups/get_wood_50000_$(date +%Y%m%d).zip

# 清理旧checkpoint（保留最新3个）
cd checkpoints/get_wood
ls -t get_wood_*_steps.zip | tail -n +4 | xargs rm -f
```

---

## 📈 **预期效果**

### **累积训练10次后**

```
总步数: 100,000
总回合: ~100
训练时间: ~20小时（分10次运行）

预期指标:
- ep_rew_mean: 0.1 → 5.0+ (稳步提升)
- ep_len_mean: 1000 → 200-500 (回合缩短，任务完成快)
- MineCLIP权重: 40.0 → ~20.0 (自动衰减)
- Agent能力: 随机探索 → 稳定完成任务

checkpoint文件:
- get_wood_10000_steps.zip
- get_wood_20000_steps.zip
- ...
- get_wood_100000_steps.zip
- get_wood_final.zip (最终模型)
```

---

## ✅ **总结**

**默认行为（推荐）：**
- ✅ 自动检测并加载最新checkpoint
- ✅ 累积训练，参数持续更新
- ✅ 多次运行自动衔接
- ✅ 中断恢复无缝继续

**特殊需求：**
- 🔄 从头开始：`--no-resume`
- 📁 指定目录：`--checkpoint-dir`
- 💾 调整保存频率：`--save-freq`

**你的场景（运行10次，每次10000步）：**
```bash
完全支持！✅
- 第1次：0 → 10000步（新模型）
- 第2次：10000 → 20000步（自动恢复）
- 第3次：20000 → 30000步（自动恢复）
- ...
- 第10次：90000 → 100000步（自动恢复）

模型参数会持续累积更新！
```

