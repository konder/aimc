# 设备支持和加速指南

本文档说明如何在不同设备上运行 MineDojo 训练，以及如何使用 GPU 加速。

---

## 支持的设备

| 设备 | 说明 | 相对速度 | 推荐使用场景 |
|------|------|----------|--------------|
| **CPU** | 所有平台都支持 | 1x | 测试、开发 |
| **CUDA** | NVIDIA GPU | 4-8x | 高性能训练 |
| **MPS** | Apple Silicon (M1/M2/M3) | 2-3x | Mac 用户训练 |

---

## 1. MPS 支持（Apple Silicon）

### 1.1 什么是 MPS？

**MPS (Metal Performance Shaders)** 是 Apple 为 M1/M2/M3 芯片提供的 GPU 加速框架。

**优势**：
- 🚀 比 CPU 快 **2-3 倍**
- 🔋 能效更高，MacBook 电池消耗更少
- 💻 无需额外硬件，M 系列 Mac 原生支持

### 1.2 检查 MPS 支持

```bash
# 运行设备检查工具
python scripts/check_device.py
```

输出示例：
```
======================================================================
PyTorch 环境检查
======================================================================
PyTorch 版本: 2.1.0
Python 版本: 3.9.18

======================================================================
设备可用性检查
======================================================================
✅ CPU: 始终可用
❌ CUDA: 不可用
✅ MPS: 可用 (Apple Silicon)
```

### 1.3 使用 MPS 训练

#### 方法1: 自动检测（推荐）

```bash
# 脚本会自动检测并使用 MPS
./scripts/train_harvest.sh
```

训练开始时会显示：
```
🍎 检测到 Apple Silicon，使用 MPS 加速
```

#### 方法2: 显式指定

```bash
# Python 直接指定
python src/training/train_harvest_paper.py --device mps
```

#### 方法3: 对比性能

```bash
# 使用 MPS
python src/training/train_harvest_paper.py --device mps

# 使用 CPU（对比）
python src/training/train_harvest_paper.py --device cpu
```

### 1.4 MPS 性能测试

在 M1 MacBook Pro (16GB) 上的实测数据：

**harvest_milk 任务，单环境**：
- CPU: 15-25 FPS，10K步耗时 8-10分钟
- MPS: 40-60 FPS，10K步耗时 3-5分钟
- **加速比**: 2.5x

**500K步完整训练**：
- CPU: 6-8小时
- MPS: 2.5-4小时
- **节省时间**: 3-4小时

### 1.5 MPS 注意事项

**内存限制**：
- 8GB M1: 建议单环境训练
- 16GB M1: 可以使用 2-4 个并行环境
- 32GB M2/M3: 可以使用 4-8 个并行环境

如果遇到内存不足：
```bash
# 减少批次大小
python src/training/train_harvest_paper.py \
    --device mps \
    --batch-size 32

# 减少图像尺寸
python src/training/train_harvest_paper.py \
    --device mps \
    --image-size 120 160
```

---

## 2. CUDA 支持（NVIDIA GPU）

### 2.1 检查 CUDA 支持

```bash
# 检查 CUDA 是否可用
python scripts/check_device.py
```

### 2.2 安装 CUDA 版本 PyTorch

```bash
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 2.3 使用 CUDA 训练

```bash
# 自动检测
./scripts/train_harvest.sh

# 或显式指定
python src/training/train_harvest_paper.py --device cuda
```

### 2.4 多 GPU 训练

```bash
# 使用多个并行环境
python src/training/train_harvest_paper.py \
    --device cuda \
    --n-envs 8
```

---

## 3. CPU 训练

### 3.1 何时使用 CPU

- 没有 GPU 的机器
- 快速测试和调试
- 对比性能基准

### 3.2 使用 CPU 训练

```bash
python src/training/train_harvest_paper.py --device cpu
```

### 3.3 优化 CPU 性能

```bash
# 设置线程数
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# 训练
python src/training/train_harvest_paper.py --device cpu
```

---

## 4. 设备选择建议

### 4.1 自动选择（推荐）

```bash
python src/training/train_harvest_paper.py --device auto
```

自动选择优先级：
1. CUDA（如果可用）
2. MPS（如果可用）
3. CPU（默认）

### 4.2 手动选择

根据你的硬件：

**Apple M1/M2/M3 Mac**:
```bash
--device mps  # 推荐
```

**NVIDIA GPU**:
```bash
--device cuda  # 推荐
```

**无 GPU**:
```bash
--device cpu
```

---

## 5. 性能对比表

### 5.1 训练速度（harvest_milk，单环境）

| 设备 | 硬件示例 | FPS | 10K步耗时 | 500K步耗时 |
|------|----------|-----|-----------|------------|
| CPU | Intel i7-10700 | 15-25 | 8-10 min | 6-8 h |
| MPS | M1 Pro 16GB | 40-60 | 3-5 min | 2.5-4 h |
| CUDA | GTX 1660 | 60-80 | 2-4 min | 2-3 h |
| CUDA | RTX 3070 | 100-150 | 1-2 min | 1-1.5 h |
| CUDA | RTX 4090 | 200-300 | 30-60 s | 30-45 min |

### 5.2 内存使用（harvest_milk，单环境）

| 设备 | 基础内存 | 推荐总内存 |
|------|----------|-----------|
| CPU | 2-3 GB | 8 GB+ |
| MPS | 3-4 GB | 16 GB+ |
| CUDA | 2-3 GB | 6 GB+ |

---

## 6. 故障排除

### 6.1 MPS 问题

#### 问题：MPS 不可用

```bash
# 检查 PyTorch 版本（需要 >= 1.12）
python -c "import torch; print(torch.__version__)"

# 升级 PyTorch
pip install --upgrade torch torchvision
```

#### 问题：MPS 训练出错

某些操作 MPS 可能不支持，回退到 CPU：
```bash
python src/training/train_harvest_paper.py --device cpu
```

#### 问题：内存不足

```bash
# 减少批次大小
--batch-size 32

# 减少图像尺寸
--image-size 120 160

# 单环境训练
--n-envs 1
```

### 6.2 CUDA 问题

#### 问题：CUDA out of memory

```bash
# 减少批次大小
--batch-size 32

# 减少并行环境
--n-envs 1
```

#### 问题：CUDA driver 版本不匹配

```bash
# 检查 CUDA 版本
nvidia-smi

# 安装对应版本的 PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### 6.3 CPU 问题

#### 问题：训练太慢

```bash
# 增加线程数
export OMP_NUM_THREADS=8

# 减少图像尺寸
--image-size 120 160

# 考虑使用 GPU
python scripts/check_device.py
```

---

## 7. 设备检查工具

### 7.1 运行检查

```bash
python scripts/check_device.py
```

### 7.2 检查内容

- PyTorch 版本
- CPU/CUDA/MPS 可用性
- 设备性能基准测试
- 模型创建测试
- 训练建议

### 7.3 示例输出

```
======================================================================
PyTorch 环境检查
======================================================================
PyTorch 版本: 2.1.0
Python 版本: 3.9.18

======================================================================
设备可用性检查
======================================================================
✅ CPU: 始终可用
❌ CUDA: 不可用
✅ MPS: 可用 (Apple Silicon)

======================================================================
设备性能测试
======================================================================
测试配置: 1000x1000 矩阵乘法, 100 次迭代

测试 CPU... ✓ 2.845秒 (70.23 GFLOPS)
测试 MPS... ✓ 0.982秒 (203.46 GFLOPS)

相对性能 (以CPU为基准):
  MPS   :  2.90x  ████████████████████████████
  CPU   :  1.00x  ██████████

======================================================================
训练建议
======================================================================
🍎 推荐使用 MPS (比 CPU 快 2-3 倍)
   python src/training/train_harvest_paper.py --device mps

自动检测设备:
   ./scripts/train_harvest.sh
   或
   python src/training/train_harvest_paper.py --device auto
```

---

## 8. 最佳实践

### 8.1 开发阶段

```bash
# 使用快速测试模式
./scripts/train_harvest.sh test

# 在 CPU 上快速迭代
python src/training/train_harvest_paper.py \
    --device cpu \
    --total-timesteps 10000
```

### 8.2 训练阶段

```bash
# 使用最快的可用设备
python src/training/train_harvest_paper.py --device auto

# 或明确指定
python src/training/train_harvest_paper.py --device mps  # Mac
python src/training/train_harvest_paper.py --device cuda # NVIDIA
```

### 8.3 生产阶段

```bash
# 长时间训练，使用多环境
python src/training/train_harvest_paper.py \
    --device cuda \
    --n-envs 8 \
    --total-timesteps 2000000
```

---

## 9. 性能优化建议

### 9.1 通用优化

1. **使用最快的可用设备**
   ```bash
   --device auto
   ```

2. **合理设置并行环境数**
   - CPU: 1-2 个
   - MPS: 1-4 个（取决于内存）
   - CUDA: 4-16 个（取决于显存）

3. **批次大小调优**
   - 更大的批次：更稳定，但内存需求高
   - 更小的批次：更快迭代，但可能不稳定

### 9.2 Mac 用户优化

```bash
# 推荐配置
python src/training/train_harvest_paper.py \
    --device mps \
    --n-envs 2 \
    --batch-size 64 \
    --image-size 160 256
```

### 9.3 高性能服务器优化

```bash
# 推荐配置
python src/training/train_harvest_paper.py \
    --device cuda \
    --n-envs 8 \
    --batch-size 128 \
    --image-size 160 256
```

---

## 总结

✅ **Mac 用户**: 使用 MPS，速度快 2-3 倍  
✅ **NVIDIA GPU 用户**: 使用 CUDA，速度快 4-8 倍  
✅ **自动检测**: `--device auto` 自动选择最优设备  
✅ **性能测试**: 使用 `python scripts/check_device.py`  

**开始你的加速训练之旅！** 🚀

