"""
中文翻译器
Chinese Translator - Translate Chinese instructions to English
"""

import json
from pathlib import Path
from typing import Dict, Optional
import logging


logger = logging.getLogger(__name__)


class ChineseTranslator:
    """中文到英文翻译器（支持术语词典优化）"""
    
    def __init__(
        self,
        term_dict_path: str = "data/chinese_terms.json",
        method: str = "term_dict",  # 'term_dict', 'baidu', 'openai'
        cache_translations: bool = True
    ):
        """
        初始化翻译器
        
        Args:
            term_dict_path: Minecraft术语词典路径
            method: 翻译方法
                - 'term_dict': 仅使用术语词典（最快，适合测试）
                - 'baidu': 百度翻译API（需要配置API key）
                - 'openai': OpenAI翻译（需要配置API key）
            cache_translations: 是否缓存翻译结果
        """
        self.method = method
        self.cache_translations = cache_translations
        self.cache: Dict[str, str] = {}
        
        # 加载术语词典
        self.term_dict = self._load_term_dict(term_dict_path)
        
        logger.info(f"翻译器初始化完成: method={method}, terms={len(self.term_dict)}")
    
    def _load_term_dict(self, dict_path: str) -> Dict[str, str]:
        """
        加载Minecraft术语词典
        
        Args:
            dict_path: 词典文件路径
            
        Returns:
            {中文: 英文} 词典
        """
        dict_file = Path(dict_path)
        
        if dict_file.exists():
            try:
                with open(dict_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 展平嵌套的词典结构
                term_dict = {}
                for key, value in data.items():
                    # 跳过元数据字段
                    if key.startswith('_'):
                        continue
                    
                    # 如果是嵌套字典（按类别分组）
                    if isinstance(value, dict):
                        for zh, en in value.items():
                            term_dict[zh] = en
                    # 如果是直接的键值对
                    elif isinstance(value, str):
                        term_dict[key] = value
                
                logger.info(f"成功加载术语词典: {len(term_dict)} 条")
                return term_dict
            except Exception as e:
                logger.warning(f"加载术语词典失败: {e}，使用空词典")
                return {}
        else:
            logger.warning(f"术语词典文件不存在: {dict_path}，使用空词典")
            return {}
    
    def translate(self, chinese_text: str) -> str:
        """
        翻译中文文本到英文
        
        Args:
            chinese_text: 中文文本
            
        Returns:
            英文翻译结果
        """
        # 1. 检查缓存
        if self.cache_translations and chinese_text in self.cache:
            logger.debug(f"使用缓存翻译: {chinese_text} -> {self.cache[chinese_text]}")
            return self.cache[chinese_text]
        
        # 2. 检查精确匹配术语词典
        if chinese_text in self.term_dict:
            translation = self.term_dict[chinese_text]
            logger.debug(f"术语词典精确匹配: {chinese_text} -> {translation}")
            
            if self.cache_translations:
                self.cache[chinese_text] = translation
            
            return translation
        
        # 3. 根据翻译方法执行翻译
        if self.method == "term_dict":
            # 仅使用术语词典（部分匹配）
            translation = self._translate_with_term_dict(chinese_text)
        elif self.method == "baidu":
            # 使用百度翻译API
            translation = self._translate_with_baidu(chinese_text)
        elif self.method == "openai":
            # 使用OpenAI翻译
            translation = self._translate_with_openai(chinese_text)
        else:
            raise ValueError(f"不支持的翻译方法: {self.method}")
        
        # 4. 缓存结果
        if self.cache_translations:
            self.cache[chinese_text] = translation
        
        logger.debug(f"翻译结果: {chinese_text} -> {translation}")
        return translation
    
    def _translate_with_term_dict(self, text: str) -> str:
        """
        使用术语词典翻译（部分匹配 + 直接返回）
        
        这是最简单的方法，适合：
        1. 快速测试
        2. 术语覆盖率高的场景
        3. 不想依赖外部API
        
        Args:
            text: 中文文本
            
        Returns:
            英文翻译（如果找不到匹配则返回原文）
        """
        # 尝试部分匹配（找文本中包含的术语）
        result = text
        for zh_term, en_term in self.term_dict.items():
            if zh_term in text:
                result = result.replace(zh_term, en_term)
        
        # 如果完全没有匹配，返回原文并警告
        if result == text:
            logger.warning(f"未找到匹配的术语: {text}，返回原文")
        
        return result
    
    def _translate_with_baidu(self, text: str) -> str:
        """
        使用百度翻译API
        
        需要配置:
            - BAIDU_APPID
            - BAIDU_SECRET_KEY
        
        Args:
            text: 中文文本
            
        Returns:
            英文翻译
        """
        # TODO: 实现百度翻译API调用
        # 这里先使用术语词典作为后备方案
        logger.warning("百度翻译API未实现，使用术语词典后备方案")
        return self._translate_with_term_dict(text)
    
    def _translate_with_openai(self, text: str) -> str:
        """
        使用OpenAI翻译
        
        需要配置:
            - OPENAI_API_KEY
        
        Args:
            text: 中文文本
            
        Returns:
            英文翻译
        """
        # TODO: 实现OpenAI翻译调用
        # 这里先使用术语词典作为后备方案
        logger.warning("OpenAI翻译未实现，使用术语词典后备方案")
        return self._translate_with_term_dict(text)
    
    def add_term(self, chinese: str, english: str):
        """
        添加术语到词典（运行时）
        
        Args:
            chinese: 中文术语
            english: 英文翻译
        """
        self.term_dict[chinese] = english
        logger.debug(f"添加术语: {chinese} -> {english}")
    
    def get_cache_stats(self) -> Dict[str, int]:
        """获取缓存统计"""
        return {
            'cache_size': len(self.cache),
            'term_dict_size': len(self.term_dict)
        }


if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(level=logging.DEBUG)
    
    # 测试翻译器
    translator = ChineseTranslator(method="term_dict")
    
    # 测试翻译
    test_cases = [
        "砍树",
        "采集泥土",
        "杀牛",
        "制作工作台",
        "未知的中文指令",
    ]
    
    print("\n翻译测试:")
    print("="*60)
    for chinese in test_cases:
        english = translator.translate(chinese)
        print(f"{chinese:15s} -> {english}")
    
    print("\n缓存统计:")
    print(translator.get_cache_stats())

