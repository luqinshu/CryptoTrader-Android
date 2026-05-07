"""
扫描策略基类
用于筛选符合条件的 OKX 合约交易对
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

from strategies._shared.validation import validate_price, validate_klines, check_nan_inf


class ScanResult(Enum):
    """扫描结果类型"""
    PASS = "通过"
    FAIL = "未通过"
    WARNING = "警告"


@dataclass
class ScannerSymbol:
    """交易对信息"""
    inst_id: str  # 交易对 ID
    last_price: float = 0.0  # 最新价
    volume_24h: float = 0.0  # 24h 成交量
    price_change_24h: float = 0.0  # 24h 涨跌幅
    high_24h: float = 0.0  # 24h 最高价
    low_24h: float = 0.0  # 24h 最低价
    open_interest: float = 0.0  # 持仓量
    extra_data: Dict[str, Any] = None  # 额外数据

    def __post_init__(self):
        if self.extra_data is None:
            self.extra_data = {}


@dataclass
class ScanCondition:
    """扫描条件"""
    name: str  # 条件名称
    description: str  # 条件描述
    field: str  # 要检查的字段
    operator: str  # 操作符: >, <, >=, <=, ==, between
    value: Any  # 比较值
    value2: Any = None  # 第二个值（用于 between）

    def check(self, symbol: ScannerSymbol) -> ScanResult:
        """
        检查交易对是否满足条件

        Args:
            symbol: 交易对信息

        Returns:
            扫描结果
        """
        try:
            # 获取字段值
            field_value = getattr(symbol, self.field, None)
            if field_value is None:
                field_value = symbol.extra_data.get(self.field)

            if field_value is None:
                return ScanResult.FAIL

            field_value = float(field_value)
            value = float(self.value)

            # 执行比较
            if self.operator == '>':
                return ScanResult.PASS if field_value > value else ScanResult.FAIL
            elif self.operator == '<':
                return ScanResult.PASS if field_value < value else ScanResult.FAIL
            elif self.operator == '>=':
                return ScanResult.PASS if field_value >= value else ScanResult.FAIL
            elif self.operator == '<=':
                return ScanResult.PASS if field_value <= value else ScanResult.FAIL
            elif self.operator == '==':
                return ScanResult.PASS if field_value == value else ScanResult.FAIL
            elif self.operator == 'between':
                value2 = float(self.value2)
                return ScanResult.PASS if value <= field_value <= value2 else ScanResult.FAIL
            else:
                return ScanResult.FAIL

        except Exception as e:
            print(f"[扫描条件] 检查失败 [{self.name}] field={self.field} op={self.operator}: {e}")
            return ScanResult.FAIL


class BaseScannerStrategy(ABC):
    """
    扫描策略基类
    
    所有扫描策略都应继承此类，实现 scan_symbol 方法
    """

    def __init__(self, config: Dict = None):
        """
        初始化扫描策略

        Args:
            config: 策略配置
        """
        self.config = config or {}
        self.conditions: List[ScanCondition] = []
        self._init_conditions()

    @abstractmethod
    def _init_conditions(self):
        """
        初始化扫描条件
        子类应在此方法中添加扫描条件
        """
        pass

    def add_condition(self, condition: ScanCondition):
        """
        添加扫描条件

        Args:
            condition: 扫描条件对象
        """
        self.conditions.append(condition)

    def scan_symbol(self, symbol: ScannerSymbol) -> Dict:
        """
        扫描单个交易对

        Args:
            symbol: 交易对信息

        Returns:
            扫描结果字典
        """
        result = {
            'symbol': symbol.inst_id,
            'passed': True,
            'conditions_met': 0,
            'conditions_total': len(self.conditions),
            'details': {},
            'score': 0.0
        }

        # 输入验证：价格有效性
        if not validate_price(symbol.last_price, label=symbol.inst_id):
            result['passed'] = False
            result['details']['_price_invalid'] = '价格无效(≤0 或 NaN/Inf)'
            return result

        # 输入验证：K 线数据有效性（如果提供了）
        klines = symbol.extra_data.get('klines') if symbol.extra_data else None
        if klines is not None:
            if isinstance(klines, dict):
                for tf, data in klines.items():
                    if not validate_klines(data, min_len=3, label=f"{symbol.inst_id}/{tf}"):
                        result['passed'] = False
                        result['details'][f'_klines_invalid_{tf}'] = f'{tf} K线数据无效'
                        return result
            elif not validate_klines(klines, min_len=3, label=symbol.inst_id):
                result['passed'] = False
                result['details']['_klines_invalid'] = 'K线数据无效'
                return result

        # 检查所有条件
        for condition in self.conditions:
            check_result = condition.check(symbol)
            result['details'][condition.name] = check_result.value

            if check_result == ScanResult.PASS:
                result['conditions_met'] += 1
                result['score'] += 1
            elif check_result == ScanResult.WARNING:
                result['score'] += 0.5

        # 计算得分率
        if result['conditions_total'] > 0:
            result['score'] = (result['conditions_met'] / result['conditions_total']) * 100

        # 只有全部条件都通过才算通过
        result['passed'] = result['conditions_met'] == result['conditions_total']

        # NaN/Inf 校验：确保 score 值安全
        result['score'] = check_nan_inf(result['score'], default=0.0)

        return result

    def get_config_schema(self) -> Dict:
        """
        获取配置模式
        子类可以重写此方法提供自定义配置界面
        """
        return {}


# 配置模式（供策略加载器使用）
SCANNER_CONFIG_SCHEMA = {
    'scan_interval': {'type': 'int', 'default': 60, 'label': '扫描间隔（秒）'},
    'min_volume_24h': {'type': 'float', 'default': 1000000, 'label': '最小 24h 成交量'},
    'max_price_change': {'type': 'float', 'default': 20.0, 'label': '最大涨跌幅 (%)'},
}
