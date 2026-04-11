"""
RSI 超买超卖扫描策略
筛选出 RSI 指标处于超买或超卖区域的合约交易对

作者：Crypto Trader
版本：1.0
"""

from src.scanner.base_scanner import (
    BaseScannerStrategy, ScannerSymbol, ScanCondition, ScanResult
)
from typing import Dict


class RSIScanner(BaseScannerStrategy):
    """
    RSI 超买超卖扫描器

    扫描逻辑:
    - RSI 低于超卖线 或 高于超买线
    - 24h 成交量大于阈值（确保流动性）
    """

    def _init_conditions(self):
        """初始化扫描条件"""
        # RSI 条件（需要额外数据中有 rsi 值）
        oversold = self.config.get('oversold', 30)
        overbought = self.config.get('overbought', 70)

        self.add_condition(ScanCondition(
            name="RSI 状态",
            description="RSI 处于超买或超卖区域",
            field="rsi",
            operator="between",
            value=0,
            value2=oversold
        ))
        # 注意：这个条件会在 scan_symbol 中被特殊处理

        # 成交量门槛
        self.add_condition(ScanCondition(
            name="成交量",
            description="24h 成交量 (USDT)",
            field="volume_24h",
            operator=">",
            value=self.config.get('min_volume_24h', 500000)
        ))

    def scan_symbol(self, symbol: ScannerSymbol) -> Dict:
        """
        扫描单个交易对（重写以支持 RSI 逻辑）
        """
        result = {
            'symbol': symbol.inst_id,
            'passed': False,
            'conditions_met': 0,
            'conditions_total': len(self.conditions),
            'details': {},
            'score': 0.0
        }

        # 获取 RSI 值
        rsi = symbol.extra_data.get('rsi', 50)
        result['details']['RSI 值'] = f"{rsi:.1f}"

        oversold = self.config.get('oversold', 30)
        overbought = self.config.get('overbought', 70)

        # 检查 RSI 是否超买或超卖
        if rsi < oversold:
            result['details']['RSI 状态'] = f"超卖 ({rsi:.1f} < {oversold})"
            result['conditions_met'] += 1
        elif rsi > overbought:
            result['details']['RSI 状态'] = f"超买 ({rsi:.1f} > {overbought})"
            result['conditions_met'] += 1
        else:
            result['details']['RSI 状态'] = f"正常 ({rsi:.1f})"

        # 检查成交量
        volume_cond = self.conditions[1] if len(self.conditions) > 1 else None
        if volume_cond:
            if symbol.volume_24h > volume_cond.value:
                result['details'][volume_cond.name] = "通过"
                result['conditions_met'] += 1
            else:
                result['details'][volume_cond.name] = "未通过"

        result['conditions_total'] = 2
        result['score'] = (result['conditions_met'] / result['conditions_total']) * 100
        result['passed'] = result['conditions_met'] == result['conditions_total']

        return result

    def get_config_schema(self) -> Dict:
        """获取配置模式"""
        return {
            'oversold': {
                'type': 'int',
                'default': 30,
                'label': '超卖线'
            },
            'overbought': {
                'type': 'int',
                'default': 70,
                'label': '超买线'
            },
            'min_volume_24h': {
                'type': 'float',
                'default': 500000,
                'label': '最小 24h 成交量 (USDT)'
            },
        }


# 策略配置模式
CONFIG_SCHEMA = {
    'oversold': {'type': 'int', 'default': 30, 'label': '超卖线'},
    'overbought': {'type': 'int', 'default': 70, 'label': '超买线'},
    'min_volume_24h': {'type': 'float', 'default': 500000, 'label': '最小 24h 成交量 (USDT)'},
}
