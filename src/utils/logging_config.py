"""
日志配置工具

提供统一的日志格式和过滤器
"""

import logging


class ShortModuleFormatter(logging.Formatter):
    """
    自定义日志格式化器，缩短模块名
    
    例如:
    - src.evaluation.steve1_evaluator → s.e.steve1_ev
    - src.envs.minerl_harvest_flatworld → s.en.m_flat
    - __main__ → main
    """
    
    # 模块名缩写映射
    MODULE_ABBREV = {
        'src.evaluation.steve1_evaluator': 's.ev.steve1',
        'src.evaluation.eval_framework': 's.ev.framework',
        'src.evaluation.task_loader': 's.ev.task_ld',
        'src.evaluation.report_generator': 's.ev.report',
        'src.envs.minerl_harvest_flatworld': 's.en.flat',
        'src.envs.minerl_harvest_default': 's.en.default',
        'src.utils.steve1_mineclip_agent_env_utils': 's.ut.env_utils',
        'src.utils.minerl_cleanup': 's.ut.cleanup',
        'src.translation.translator': 's.tr.translator',
        '__main__': 'main',
    }
    
    def format(self, record):
        """格式化日志记录，缩短模块名"""
        # 获取原始模块名
        original_name = record.name
        
        # 如果有完整匹配，使用映射
        if original_name in self.MODULE_ABBREV:
            record.name = self.MODULE_ABBREV[original_name]
        else:
            # 否则自动缩写
            record.name = self._abbreviate_name(original_name)
        
        # 使用父类格式化
        result = super().format(record)
        
        # 恢复原始名称（避免影响其他 handler）
        record.name = original_name
        
        return result
    
    def _abbreviate_name(self, name: str) -> str:
        """
        自动缩写模块名
        
        规则:
        - src.evaluation.xxx → s.ev.xxx
        - src.envs.xxx → s.en.xxx
        - src.utils.xxx → s.ut.xxx
        - 其他保持不变
        """
        parts = name.split('.')
        
        if len(parts) <= 2:
            return name
        
        # 缩写规则
        abbrev_parts = []
        for i, part in enumerate(parts):
            if i == len(parts) - 1:
                # 最后一个部分保留（或适当缩短）
                if len(part) > 12:
                    abbrev_parts.append(part[:10] + '..')
                else:
                    abbrev_parts.append(part)
            elif part == 'src':
                abbrev_parts.append('s')
            elif part == 'evaluation':
                abbrev_parts.append('ev')
            elif part == 'envs':
                abbrev_parts.append('en')
            elif part == 'utils':
                abbrev_parts.append('ut')
            elif part == 'translation':
                abbrev_parts.append('tr')
            elif len(part) > 8:
                abbrev_parts.append(part[:6])
            else:
                abbrev_parts.append(part)
        
        return '.'.join(abbrev_parts)


class ModuleFilter(logging.Filter):
    """
    日志过滤器，过滤掉不需要的模块
    """
    
    # 要过滤的模块名（黑名单）
    BLOCKED_MODULES = [
        'process_watcher',
        'minerl.env.malmo.instance',
    ]
    
    def filter(self, record):
        """
        过滤日志记录
        
        Returns:
            bool: True 表示允许，False 表示过滤掉
        """
        # 检查是否在黑名单中
        for blocked in self.BLOCKED_MODULES:
            if record.name.startswith(blocked):
                return False
        
        return True


def setup_logging(level=logging.INFO, format_string=None):
    """
    配置全局日志系统
    
    Args:
        level: 日志级别
        format_string: 日志格式字符串（可选）
    """
    # 默认格式
    if format_string is None:
        format_string = '%(asctime)s - %(name)-20s - %(levelname)-7s - %(message)s'
    
    # 获取根 logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # 清除现有的 handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # 创建控制台 handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    # 使用自定义格式化器
    formatter = ShortModuleFormatter(format_string)
    console_handler.setFormatter(formatter)
    
    # 添加过滤器
    console_handler.addFilter(ModuleFilter())
    
    # 添加 handler 到根 logger
    root_logger.addHandler(console_handler)


def setup_evaluation_logging():
    """
    为评估框架配置日志
    
    使用固定宽度的模块名，便于对齐
    """
    format_string = '%(asctime)s - %(name)-20s - %(levelname)-7s - %(message)s'
    setup_logging(level=logging.INFO, format_string=format_string)

