# MineDojo任务快速入门

## 📚 文档导航

### 完整任务参考
详细的任务清单、完成条件和奖励设计，请查看：
- **[MineDojo任务完整参考手册](../MINEDOJO_TASKS_REFERENCE.md)**

### 快速查找

#### 🎯 按难度选择任务

**新手推荐（⭐）**:
```python
"harvest_1_milk"          # 采集牛奶 - 最简单
"harvest_1_log"           # 采集木头
"harvest_1_dirt"          # 采集泥土
"combat_cow_forest_barehand"  # 空手打牛
```

**进阶任务（⭐⭐⭐）**:
```python
"harvest_1_paper"         # 制作纸张
"harvest_1_iron_ingot"    # 冶炼铁锭
"combat_zombie_forest_leather_armors_wooden_sword_shield"  # 战斗僵尸
"techtree_from_barehand_to_stone_sword"  # 制作石剑
```

**高级任务（⭐⭐⭐⭐⭐）**:
```python
"harvest_1_totem_of_undying"  # 获取不死图腾
"combat_enderman_plains_diamond_armors_diamond_sword_shield"  # 战斗末影人
"techtree_from_barehand_to_diamond_pickaxe"  # 从零制作钻石镐
"survival"                # 长期生存
```

#### 🏷️ 按类型选择任务

| 类型 | 任务数量 | 推荐场景 |
|------|---------|---------|
| **Harvest** (采集) | 895个 | 学习基础操作、资源收集 |
| **Combat** (战斗) | 462个 | 学习战斗策略、风险管理 |
| **TechTree** (科技树) | 213个 | 学习长期规划、复杂决策 |
| **Survival** (生存) | 2个 | 综合能力测试 |

## 🚀 快速使用

### 1. 列出所有可用任务

```python
from minedojo.tasks import ALL_PROGRAMMATIC_TASK_IDS

# 查看所有任务
print(f"总共有 {len(ALL_PROGRAMMATIC_TASK_IDS)} 个任务")

# 查看前10个任务
for task_id in ALL_PROGRAMMATIC_TASK_IDS[:10]:
    print(task_id)
```

### 2. 创建并运行任务

```python
import minedojo

# 创建环境
env = minedojo.make(
    task_id="harvest_1_milk",  # 任务ID
    image_size=(160, 256)       # 图像尺寸
)

# 重置环境
obs = env.reset()

# 运行
done = False
while not done:
    action = env.action_space.sample()  # 随机动作
    obs, reward, done, info = env.step(action)
    
    if reward > 0:
        print("任务完成！")
        break

env.close()
```

### 3. 查看任务描述

```python
import minedojo

# 创建环境
env = minedojo.make(task_id="harvest_1_paper")

# 如果任务有描述，会存储在环境中
if hasattr(env, 'task_prompt'):
    print(f"任务描述: {env.task_prompt}")

env.close()
```

## 📊 任务统计工具

项目提供了一个任务统计脚本：

```bash
# 运行任务列表工具
python scripts/list_minedojo_tasks.py
```

这会生成所有任务的摘要信息。

## 💡 训练技巧

### 从简单任务开始

建议按以下顺序训练：

1. **阶段1：基础采集**（学习基本操作）
   - `harvest_1_milk`
   - `harvest_1_log`
   - `harvest_1_dirt`

2. **阶段2：简单制作**（学习合成）
   - `harvest_1_stick`
   - `harvest_1_crafting_table`
   - `harvest_1_torch`

3. **阶段3：工具制作**（学习科技树）
   - `techtree_from_barehand_to_wooden_pickaxe`
   - `techtree_from_barehand_to_stone_sword`

4. **阶段4：战斗入门**（学习战斗）
   - `combat_cow_forest_barehand`
   - `combat_zombie_forest_leather_armors_wooden_sword_shield`

5. **阶段5：高级任务**（综合挑战）
   - `techtree_from_barehand_to_iron_sword`
   - `combat_enderman_*`
   - `survival`

### 课程学习策略

```python
# 定义课程学习任务序列
curriculum = [
    "harvest_1_milk",           # 难度1：基础采集
    "harvest_1_log",            # 难度1：基础采集
    "harvest_1_crafting_table", # 难度2：简单合成
    "harvest_1_stick",          # 难度2：简单合成
    "harvest_1_paper",          # 难度3：复杂合成
    "harvest_1_iron_ingot",     # 难度4：需要冶炼
]

# 训练循环
for task_id in curriculum:
    print(f"训练任务: {task_id}")
    env = minedojo.make(task_id=task_id)
    # ... 训练代码
    env.close()
```

## 🔧 常用任务模板

### 采集任务模板

```python
def train_harvest_task(task_id, max_steps=5000):
    """采集任务训练模板"""
    env = minedojo.make(task_id=task_id, image_size=(160, 256))
    obs = env.reset()
    
    for step in range(max_steps):
        action = agent.get_action(obs)
        obs, reward, done, info = env.step(action)
        
        if reward > 0:
            print(f"✓ 任务完成！步数：{step}")
            break
        
        if done:
            print(f"✗ 任务失败，步数：{step}")
            break
    
    env.close()
```

### 战斗任务模板

```python
def train_combat_task(task_id, max_steps=10000):
    """战斗任务训练模板"""
    env = minedojo.make(task_id=task_id, image_size=(160, 256))
    obs = env.reset()
    
    for step in range(max_steps):
        # 检查生命值
        health = obs['life_stats']['life'][0]
        if health < 5:
            # 低血量策略
            action = retreat_action()
        else:
            action = agent.get_action(obs)
        
        obs, reward, done, info = env.step(action)
        
        if reward > 0:
            print(f"✓ 击败目标！步数：{step}")
            break
    
    env.close()
```

### 科技树任务模板

```python
def train_techtree_task(task_id, max_steps=20000):
    """科技树任务训练模板"""
    env = minedojo.make(task_id=task_id, image_size=(160, 256))
    obs = env.reset()
    
    milestones = []  # 记录里程碑
    
    for step in range(max_steps):
        action = agent.get_action(obs)
        obs, reward, done, info = env.step(action)
        
        # 检查物品栏变化
        if step % 100 == 0:
            inventory = obs['inventory']['name']
            milestones.append(inventory)
        
        if reward > 0:
            print(f"✓ 科技树完成！步数：{step}")
            print(f"里程碑数量：{len(milestones)}")
            break
    
    env.close()
```

## 📈 评估指标

评估任务性能时，可以关注以下指标：

```python
def evaluate_task(task_id, num_episodes=10):
    """评估任务性能"""
    env = minedojo.make(task_id=task_id)
    
    results = {
        'success_rate': 0,
        'avg_steps': 0,
        'min_steps': float('inf'),
        'max_steps': 0
    }
    
    successes = 0
    total_steps = 0
    
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        steps = 0
        
        while not done and steps < 10000:
            action = agent.get_action(obs)
            obs, reward, done, info = env.step(action)
            steps += 1
            
            if reward > 0:
                successes += 1
                total_steps += steps
                results['min_steps'] = min(results['min_steps'], steps)
                results['max_steps'] = max(results['max_steps'], steps)
                break
    
    results['success_rate'] = successes / num_episodes
    results['avg_steps'] = total_steps / max(successes, 1)
    
    env.close()
    return results
```

## 🎓 学习资源

- **完整任务列表**: [MINEDOJO_TASKS_REFERENCE.md](../MINEDOJO_TASKS_REFERENCE.md)
- **训练指南**: [TRAINING_GUIDE.md](./TRAINING_GUIDE.md)
- **TensorBoard监控**: [TENSORBOARD_GUIDE.md](./TENSORBOARD_GUIDE.md)
- **MineDojo官方文档**: https://docs.minedojo.org

## ❓ 常见问题

### Q: 如何选择合适的任务进行训练？

**A**: 建议从简单的harvest任务开始，逐步增加难度。可以参考文档中的难度评级（⭐）。

### Q: 任务训练需要多长时间？

**A**: 
- 简单任务（⭐）: 1-10万步
- 中等任务（⭐⭐⭐）: 10-50万步
- 困难任务（⭐⭐⭐⭐⭐）: 50万步以上

### Q: 如何调整任务难度？

**A**: 
1. 选择不同数量要求（1个 vs 8个）
2. 选择有/无初始工具的版本
3. 战斗任务可以选择不同装备等级

### Q: 任务完成条件是什么？

**A**: 不同任务有不同条件：
- **Harvest**: 物品栏中有目标物品
- **Combat**: 击败目标生物
- **TechTree**: 拥有目标装备
- **Survival**: 存活尽可能长时间

---

**快速查询**: 使用 Ctrl+F 在[完整参考手册](../MINEDOJO_TASKS_REFERENCE.md)中搜索任务ID


