"""
策略加载器模块
支持动态加载和管理交易策略
"""

import os
import sys
import importlib.util
import inspect
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum


class StrategyType(Enum):
    """策略类型枚举"""
    SCAN = "scan"  # 扫描策略
    TRADE = "trade"  # 交易策略
    BACKTEST = "backtest"  # 回测策略


@dataclass
class StrategyInfo:
    """策略信息数据类"""
    name: str
    path: str
    type: StrategyType
    description: str = ""
    author: str = ""
    version: str = "1.0"
    config_schema: Dict[str, Any] = None
    module: Any = None

    def create_instance(self, config: Dict[str, Any] = None) -> Optional[Any]:
        """
        从策略信息创建策略实例
        
        Args:
            config: 策略配置字典
            
        Returns:
            策略实例对象
        """
        try:
            # 动态导入模块
            spec = importlib.util.spec_from_file_location(self.name, self.path)
            if spec is None or spec.loader is None:
                return None
            
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            try:
                spec.loader.exec_module(module)
            except Exception as e:
                del sys.modules[spec.name]
                print(f"[策略加载] {file_path} 执行失败: {e}")
                return None
            self.module = module
            
            # 自动查找策略类
            strategy_class = None
            for name, obj in inspect.getmembers(module, inspect.isclass):
                # 跳过导入的类
                if obj.__module__ == module.__name__:
                    # 优先查找包含 Strategy, Scanner 或 策略, 扫描器 的类名
                    if any(keyword in name for keyword in ['Strategy', 'Scanner', '策略', '扫描器']):
                        strategy_class = obj
                        break
            
            # 如果没有找到，返回第一个自定义类
            if not strategy_class:
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if obj.__module__ == module.__name__:
                        strategy_class = obj
                        break
            
            if strategy_class:
                return strategy_class(config or {})
            
            return None
        except Exception as e:
            print(f"实例化策略 {self.name} 失败: {e}")
            import traceback
            traceback.print_exc()
            return None


class StrategyLoader:
    """策略加载器"""

    def __init__(self, strategies_dir: str = None):
        """
        初始化策略加载器

        Args:
            strategies_dir: 策略目录路径，默认为项目根目录的 strategies 文件夹
        """
        self.strategies_dir = strategies_dir
        self.strategies: Dict[str, StrategyInfo] = {}
        self.loaded_modules: Dict[str, Any] = {}
        self._custom_paths: set = set()  # 通过文件对话框手动加载的路径集合

        if self.strategies_dir and not os.path.exists(self.strategies_dir):
            os.makedirs(self.strategies_dir)

    def discover_strategies(self) -> List[StrategyInfo]:
        """
        发现策略目录中的所有策略

        Returns:
            策略信息列表
        """
        if not self.strategies_dir:
            # 没有目录时直接返回已有策略（含手动加载的）
            return list(self.strategies.values())

        # 保留手动加载的自定义策略，清空目录扫描结果
        custom_saved = {name: info for name, info in self.strategies.items()
                        if info.path in self._custom_paths}
        self.strategies.clear()
        self.strategies.update(custom_saved)

        # 扫描策略目录
        for root, dirs, files in os.walk(self.strategies_dir):
            # 跳过 __pycache__ 等目录
            if '__pycache__' in root or root.startswith('.'):
                continue

            for file in files:
                if file.endswith('.py') and not file.startswith('_'):
                    file_path = os.path.join(root, file)
                    strategy_info = self._analyze_strategy_file(file_path)
                    if strategy_info:
                        self.strategies[strategy_info.name] = strategy_info

        return list(self.strategies.values())

    def _analyze_strategy_file(self, file_path: str) -> Optional[StrategyInfo]:
        """
        分析策略文件，提取策略信息

        Args:
            file_path: 策略文件路径

        Returns:
            策略信息对象，如果分析失败则返回 None
        """
        try:
            # 读取文件内容
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # 提取基本信息
            file_name = os.path.basename(file_path)
            strategy_name = os.path.splitext(file_name)[0]

            # 尝试从文档字符串中提取信息
            description = ""
            author = ""
            version = "1.0"

            if '"""' in content:
                doc_start = content.find('"""')
                doc_end = content.find('"""', doc_start + 3)
                if doc_end > doc_start:
                    docstring = content[doc_start + 3:doc_end].strip()
                    lines = docstring.split('\n')
                    description = lines[0].strip() if lines else ""

                    for line in lines:
                        if '作者:' in line or 'Author:' in line:
                            author = line.split(':', 1)[1].strip()
                        elif '版本:' in line or 'Version:' in line:
                            version = line.split(':', 1)[1].strip()

            # 判断策略类型
            strategy_type = StrategyType.TRADE
            if 'scan' in file_path.lower() or '扫描' in file_path:
                strategy_type = StrategyType.SCAN
            elif 'backtest' in file_path.lower() or '回测' in file_path:
                strategy_type = StrategyType.BACKTEST

            # 尝试提取配置模式
            config_schema = self._extract_config_schema(content)

            return StrategyInfo(
                name=strategy_name,
                path=file_path,
                type=strategy_type,
                description=description,
                author=author,
                version=version,
                config_schema=config_schema
            )

        except Exception as e:
            print(f"分析策略文件失败 {file_path}: {e}")
            return None

    def _extract_config_schema(self, content: str) -> Dict[str, Any]:
        """
        从策略代码中提取配置模式

        Args:
            content: 策略代码内容

        Returns:
            配置模式字典
        """
        schema = {}

        # 1. 尝试查找 CONFIG_SCHEMA 变量
        import re
        
        # 查找 CONFIG_SCHEMA = {...} 或 CONFIG_SCHEMA: Dict[...] = {...} (支持多行)
        config_match = re.search(
            r'CONFIG_SCHEMA(?:\s*:\s*[^=]+)?\s*=\s*(\{[\s\S]*?\n\})',
            content,
        )
        if config_match:
            try:
                schema_str = config_match.group(1)
                # 简单解析 Python 字典
                schema = eval(schema_str)
                if isinstance(schema, dict) and schema:
                    return schema
            except Exception as e:
                print(f"解析 CONFIG_SCHEMA 失败：{e}")
                pass

        # 2. 尝试查找类属性中的配置参数
        lines = content.split('\n')
        for line in lines:
            line = line.strip()

            # 查找常见的配置参数
            if 'stop_loss' in line and '=' in line:
                try:
                    value = float(line.split('=')[1].strip().rstrip(','))
                    schema['stop_loss'] = {'type': 'float', 'default': value, 'label': '止损百分比'}
                except:
                    pass

            if 'take_profit' in line and '=' in line:
                try:
                    value = float(line.split('=')[1].strip().rstrip(','))
                    schema['take_profit'] = {'type': 'float', 'default': value, 'label': '止盈百分比'}
                except:
                    pass

            if 'position_size' in line and '=' in line:
                try:
                    value = float(line.split('=')[1].strip().rstrip(','))
                    schema['position_size'] = {'type': 'float', 'default': value, 'label': '仓位比例'}
                except:
                    pass

            if 'rsi_period' in line and '=' in line:
                try:
                    value = int(line.split('=')[1].strip().rstrip(','))
                    schema['rsi_period'] = {'type': 'int', 'default': value, 'label': 'RSI 周期'}
                except:
                    pass

        return schema

    def load_strategy(self, strategy_name: str) -> Optional[Any]:
        """
        加载指定策略

        Args:
            strategy_name: 策略名称

        Returns:
            策略模块对象，如果加载失败则返回 None
        """
        if strategy_name not in self.strategies:
            print(f"策略 {strategy_name} 不存在")
            return None

        strategy_info = self.strategies[strategy_name]

        try:
            # 如果已经加载过，直接返回
            if strategy_name in self.loaded_modules:
                return self.loaded_modules[strategy_name]

            # 动态导入模块
            spec = importlib.util.spec_from_file_location(strategy_name, strategy_info.path)
            if spec is None or spec.loader is None:
                print(f"无法加载策略模块：{strategy_name}")
                return None

            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            try:
                spec.loader.exec_module(module)
            except Exception as e:
                del sys.modules[spec.name]
                print(f"[策略加载] {strategy_name} 执行失败: {e}")
                return None

            # 保存到已加载模块
            self.loaded_modules[strategy_name] = module
            strategy_info.module = module

            print(f"成功加载策略：{strategy_name}")
            return module

        except Exception as e:
            print(f"加载策略失败 {strategy_name}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def unload_strategy(self, strategy_name: str) -> bool:
        """
        卸载指定策略

        Args:
            strategy_name: 策略名称

        Returns:
            是否成功卸载
        """
        if strategy_name not in self.strategies:
            return False

        try:
            if strategy_name in self.loaded_modules:
                del self.loaded_modules[strategy_name]
            if strategy_name in sys.modules:
                del sys.modules[strategy_name]

            print(f"成功卸载策略：{strategy_name}")
            return True
        except Exception as e:
            print(f"卸载策略失败 {strategy_name}: {e}")
            return False

    def reload_strategy(self, strategy_name: str) -> Optional[Any]:
        """
        重新加载策略

        Args:
            strategy_name: 策略名称

        Returns:
            重新加载后的策略模块
        """
        self.unload_strategy(strategy_name)
        return self.load_strategy(strategy_name)

    def get_strategy_class(self, strategy_name: str, class_name: str = None) -> Optional[type]:
        """
        获取策略类

        Args:
            strategy_name: 策略名称
            class_name: 类名，如果为 None 则尝试自动查找

        Returns:
            策略类，如果未找到则返回 None
        """
        module = self.load_strategy(strategy_name)
        if module is None:
            return None

        try:
            # 如果指定了类名，直接获取
            if class_name:
                return getattr(module, class_name, None)

            # 自动查找策略类
            for name, obj in inspect.getmembers(module, inspect.isclass):
                # 跳过导入的类
                if obj.__module__ == module.__name__:
                    # 查找包含 Strategy 或 策略 的类名
                    if 'Strategy' in name or '策略' in name:
                        return obj

            # 如果没有找到，返回第一个自定义类
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if obj.__module__ == module.__name__:
                    return obj

            return None

        except Exception as e:
            print(f"获取策略类失败：{e}")
            return None

    def get_all_strategies(self) -> Dict[str, StrategyInfo]:
        """获取所有已发现的策略"""
        return self.strategies

    def get_strategies_by_type(self, strategy_type: StrategyType) -> List[StrategyInfo]:
        """
        按类型获取策略

        Args:
            strategy_type: 策略类型

        Returns:
            策略信息列表
        """
        return [info for info in self.strategies.values() if info.type == strategy_type]

    def load_custom_strategy(self, file_path: str) -> Optional[StrategyInfo]:
        """
        加载自定义策略文件

        Args:
            file_path: 策略文件路径

        Returns:
            策略信息对象
        """
        if not os.path.exists(file_path):
            print(f"文件不存在：{file_path}")
            return None

        strategy_info = self._analyze_strategy_file(file_path)
        if strategy_info:
            self.strategies[strategy_info.name] = strategy_info
            self._custom_paths.add(os.path.abspath(file_path))
        return strategy_info
