# STEVE-1评估器奖励计算重构总结

## 重构目标

将奖励计算逻辑从环境层（MineRLHarvestWrapper）移到评估层（steve1_evaluator），实现更清晰的职责分离。

## 核心改动

### 1. 奖励计算方法 (_calculate_custom_reward)

添加到 `steve1_evaluator.py` 中，在 `evaluate_task` 方法之后：

```python
def _calculate_custom_reward(
    self,
    obs: Dict[str, Any],
    prev_inventory: Dict[str, int],
    reward_config: List[Dict],
    reward_rule: str,
    item_completed: Dict[str, bool]
) -> Tuple[float, Dict[str, int], Dict[str, bool], bool]:
    """
    根据reward_config计算自定义奖励
    
    Args:
        obs: 当前观察
        prev_inventory: 上一步的库存
        reward_config: 奖励配置
        reward_rule: 完成规则 ("any", "all", "none")
        item_completed: 物品完成状态
        
    Returns:
        Tuple[reward, current_inventory, item_completed, task_done]
    """
    reward = 0.0
    task_done = False
    current_inventory = {}
    
    if 'inventory' not in obs:
        return reward, current_inventory, item_completed, task_done
    
    # 遍历所有配置的物品
    for cfg in reward_config:
        entity = cfg["entity"]
        target_amount = cfg["amount"]
        reward_value = cfg["reward"]
        
        # 获取当前数量
        current_count = 0
        if entity in obs['inventory']:
            count = obs['inventory'][entity]
            if hasattr(count, 'item'):
                count = count.item()
            current_count = int(count)
        
        current_inventory[entity] = current_count
        prev_count = prev_inventory.get(entity, 0)
        
        # 如果物品数量增加，给予增量奖励
        if current_count > prev_count:
            items_gained = current_count - prev_count
            reward += items_gained * (reward_value / target_amount)
        
        # 检查是否达到目标
        if current_count >= target_amount and not item_completed[entity]:
            item_completed[entity] = True
            logger.info(f"✅ 物品 '{entity}' 达到目标数量 {target_amount}，当前数量: {current_count}")
    
    # 检查任务完成条件
    if reward_rule == "any":
        if any(item_completed.values()):
            task_done = True
            total_reward = sum(
                cfg["reward"] 
                for cfg in reward_config 
                if item_completed[cfg["entity"]]
            )
            logger.info(f"✅ 任务完成！总奖励: {total_reward}")
    elif reward_rule == "all":
        if all(item_completed.values()):
            task_done = True
            total_reward = sum(cfg["reward"] for cfg in reward_config)
            logger.info(f"✅ 任务完成！所有物品都已达标，总奖励: {total_reward}")
    # reward_rule == "none" 时不检查完成条件
    
    return reward, current_inventory, item_completed, task_done
```

### 2. 修改 _run_single_trial

修改签名添加参数：
```python
def _run_single_trial(
    self,
    task_id: str,
    instruction: str,
    max_steps: int,
    trial_idx: int,
    reward_config: Optional[List[Dict]] = None,
    reward_rule: str = "any"
) -> TrialResult:
```

在循环中使用自定义奖励：
```python
# 初始化自定义奖励相关状态
if reward_config:
    item_inventory = {cfg["entity"]: 0 for cfg in reward_config}
    item_completed = {cfg["entity"]: False for cfg in reward_config}
else:
    item_inventory = {}
    item_completed = {}

while not done and steps < max_steps:
    # ... 获取动作并执行 ...
    obs, reward, done, info = self._env.step(action)
    
    # 如果有自定义奖励配置，计算额外奖励
    if reward_config:
        custom_reward, item_inventory, item_completed, custom_done = self._calculate_custom_reward(
            obs, item_inventory, reward_config, reward_rule, item_completed
        )
        reward += custom_reward  # 叠加自定义奖励
        if custom_done:
            done = True  # 任务提前完成
    
    total_reward += reward
    steps += 1
    # ...
```

### 3. 修改 eval_framework

在 `evaluate_single_task` 中读取 reward_config 和 reward_rule：

```python
# 从任务配置读取奖励配置
reward_config = task_config.get('reward_config')
reward_rule = task_config.get('reward_rule', 'any')

# 调用评估器时传递
result = self.evaluator.evaluate_task(
    task_id=task_id,
    language=language,
    n_trials=n_trials,
    max_steps=max_steps,
    instruction=instruction,
    reward_config=reward_config,
    reward_rule=reward_rule
)
```

### 4. 更新配置文件格式

`config/eval_tasks.yaml`:

```yaml
- task_id: "harvest_any_1_logs"
  category: "harvest"
  difficulty: "easy"
  description: "使用英文指令砍任意树木获取一块木头"
  env_name: "MineRLHarvestEnv-v0"
  en_instruction: "chop tree"
  reward_config:  # 奖励配置（不再嵌套在env_config中）
    - entity: "oak_log"
      amount: 1
      reward: 100
    - entity: "birch_log"
      amount: 1
      reward: 100
    - entity: "spruce_log"
      amount: 1
      reward: 100
    - entity: "dark_oak_log"
      amount: 1
      reward: 100
    - entity: "jungle_log"
      amount: 1
      reward: 100
    - entity: "acacia_log"
      amount: 1
      reward: 100
  reward_rule: "any"  # 任意一种达标即可（不再用task_complete_on_any）
  max_steps: 1000
```

## 优势

1. **职责分离清晰**：环境只负责基础功能，评估器负责奖励计算
2. **配置更简洁**：不需要env_config这一层嵌套
3. **命名更清晰**：reward_rule 比 task_complete_on_any 更直观
4. **易于扩展**：未来可以轻松添加更多 reward_rule 类型

## 配置对比

### 之前（env_config）
```yaml
env_config:
  reward_config:
    - entity: "oak_log"
      amount: 1
      reward: 100
  task_complete_on_any: true
```

### 现在（直接配置）
```yaml
reward_config:
  - entity: "oak_log"
    amount: 1
    reward: 100
reward_rule: "any"
```

更扁平、更清晰！

