#!/usr/bin/env python3
"""
测试通过 generator_options 指定 Biome 是否可行

目的：
1. 测试不同的 generator_options 格式
2. 检查 Minecraft 日志中的世界生成信息
3. 验证 Biome 是否被正确应用

使用方法：
    ./scripts/run_minedojo_x86.sh python scripts/test_biome_generator_options.py
"""

import gym
import time
import logging
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 测试配置
TEST_CASES = [
    {
        "name": "默认（无参数）",
        "generator_options": "",
        "description": "不指定任何参数，使用默认世界生成"
    },
    {
        "name": "JSON格式 - Plains",
        "generator_options": '{"biome":"minecraft:plains"}',
        "description": "尝试使用JSON格式指定Plains biome"
    },
    {
        "name": "JSON格式 - Mountains", 
        "generator_options": '{"biome":"minecraft:mountains"}',
        "description": "尝试使用JSON格式指定Mountains biome"
    },
    {
        "name": "biome前缀格式",
        "generator_options": "biome:minecraft:mountains",
        "description": "尝试使用特殊前缀格式"
    },
    {
        "name": "MC 1.12格式",
        "generator_options": '{"biome":4}',  # 4 = Mountains in MC 1.12
        "description": "尝试使用MC 1.12的biome ID格式"
    }
]


def test_generator_option(test_case, max_steps=50):
    """
    测试单个 generator_option 配置
    
    Args:
        test_case: 测试用例配置
        max_steps: 测试步数
    """
    logger.info("=" * 80)
    logger.info(f"测试: {test_case['name']}")
    logger.info(f"描述: {test_case['description']}")
    logger.info(f"generator_options: {test_case['generator_options']}")
    logger.info("=" * 80)
    
    try:
        # 导入MineRL环境
        from src.envs.minerl_harvest_default import register_minerl_harvest_default_env
        register_minerl_harvest_default_env()
        
        # 创建环境
        logger.info("创建环境...")
        env = gym.make(
            'MineRLHarvestDefaultEnv-v0',
            world_generator={
                'force_reset': True,
                'generator_options': test_case['generator_options']
            },
            reward_config=[
                {'entity': 'dirt', 'amount': 1, 'reward': 1}
            ],
            time_condition={'allow_passage_of_time': False},
            spawning_condition={'allow_spawning': True},
            max_episode_steps=max_steps
        )
        
        # 重置环境
        logger.info("重置环境（这会触发世界生成）...")
        obs = env.reset()
        logger.info("✓ 环境重置成功")
        
        # 运行几步
        logger.info(f"运行 {max_steps} 步...")
        for step in range(max_steps):
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            
            if step % 10 == 0:
                logger.info(f"  Step {step}/{max_steps}")
            
            if done:
                logger.info(f"  环境提前结束于步骤 {step}")
                break
        
        logger.info("✓ 测试完成")
        
        # 关闭环境
        env.close()
        logger.info("✓ 环境已关闭")
        
        # 提示查看日志
        import glob
        latest_log = max(glob.glob('/Users/nanzhang/aimc/logs/mc_*.log'), key=lambda x: Path(x).stat().st_mtime)
        logger.info("")
        logger.info("=" * 80)
        logger.info(f"请查看最新的Minecraft日志以验证世界生成配置:")
        logger.info(f"  {latest_log}")
        logger.info("")
        logger.info("建议的查看命令:")
        logger.info(f"  grep -i 'generator\\|biome\\|world' {latest_log} | tail -50")
        logger.info("=" * 80)
        logger.info("")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 测试失败: {e}", exc_info=True)
        return False
    
    finally:
        # 等待环境完全关闭
        time.sleep(2)


def main():
    """
    运行所有测试用例
    """
    logger.info("")
    logger.info("╔" + "=" * 78 + "╗")
    logger.info("║" + " " * 20 + "Biome Generator Options 测试" + " " * 29 + "║")
    logger.info("╚" + "=" * 78 + "╝")
    logger.info("")
    logger.info(f"将测试 {len(TEST_CASES)} 个配置")
    logger.info("")
    
    results = []
    
    for i, test_case in enumerate(TEST_CASES, 1):
        logger.info(f"\n{'=' * 80}")
        logger.info(f"测试 {i}/{len(TEST_CASES)}")
        logger.info(f"{'=' * 80}\n")
        
        success = test_generator_option(test_case)
        results.append({
            'name': test_case['name'],
            'success': success
        })
        
        if i < len(TEST_CASES):
            logger.info("\n等待5秒后继续下一个测试...\n")
            time.sleep(5)
    
    # 打印总结
    logger.info("")
    logger.info("╔" + "=" * 78 + "╗")
    logger.info("║" + " " * 30 + "测试总结" + " " * 40 + "║")
    logger.info("╚" + "=" * 78 + "╝")
    logger.info("")
    
    for i, result in enumerate(results, 1):
        status = "✅ 成功" if result['success'] else "❌ 失败"
        logger.info(f"{i}. {result['name']}: {status}")
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("下一步:")
    logger.info("  1. 查看Minecraft日志，搜索 'generator' 或 'biome' 关键字")
    logger.info("  2. 检查是否有错误或警告信息")
    logger.info("  3. 如果有效，更新 eval_tasks.yaml 使用工作的格式")
    logger.info("=" * 80)
    logger.info("")


if __name__ == '__main__':
    main()

