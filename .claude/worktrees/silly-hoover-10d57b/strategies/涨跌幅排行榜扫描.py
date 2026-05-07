"""
涨跌幅排行榜扫描策略
筛选24小时内涨幅最大和跌幅最大的交易对

作者：Crypto Trader
版本：1.0
"""

from src.scanner.base_scanner import (
    BaseScannerStrategy, ScannerSymbol, ScanCondition, ScanResult
)
from typing import Dict, List


class TopGainerLoserScanner(BaseScannerStrategy):
    """
    涨跌幅排行榜扫描器

    扫描逻辑:
    - 获取所有合约交易对的24h涨跌幅
    - 筛选涨幅最大的前N名
    - 筛选跌幅最大的前N名
    - 可选：过滤成交量过低的交易对
    """

    def _init_conditions(self):
        """初始化扫描条件"""
        # 基础成交量门槛（避免低流动性交易对）
        self.add_condition(ScanCondition(
            name="成交量",
            description="24h 成交量 (USDT)",
            field="volume_24h",
            operator=">",
            value=self.config.get('min_volume_24h', 100000)
        ))

    def scan_all_symbols(self, symbols_data: List[ScannerSymbol]) -> Dict:
        """
        扫描所有交易对并返回涨跌幅排行

        Args:
            symbols_data: 所有交易对的 ScannerSymbol 列表

        Returns:
            包含涨幅和跌幅排行榜的字典
        """
        # 应用成交量过滤
        min_volume = self.config.get('min_volume_24h', 100000)
        filtered_symbols = [
            s for s in symbols_data
            if s.volume_24h >= min_volume
        ]

        # 按涨跌幅排序
        sorted_by_gain = sorted(
            filtered_symbols,
            key=lambda s: s.price_change_24h,
            reverse=True  # 涨幅从大到小
        )

        # 获取前N名
        top_n = self.config.get('top_n', 10)
        
        top_gainers = sorted_by_gain[:top_n]  # 涨幅榜
        top_losers = sorted_by_gain[-top_n:]  # 跌幅榜（最后N个）
        top_losers.reverse()  # 跌幅从大到小（负值最大的在前）

        # 构建结果
        result = {
            'type': 'gainer_loser_ranking',
            'top_gainers': [self._format_symbol_result(s, rank+1) for rank, s in enumerate(top_gainers)],
            'top_losers': [self._format_symbol_result(s, rank+1) for rank, s in enumerate(top_losers)],
            'total_scanned': len(symbols_data),
            'total_filtered': len(filtered_symbols),
        }

        return result

    def _format_symbol_result(self, symbol: ScannerSymbol, rank: int) -> Dict:
        """
        格式化单个交易对的排行结果

        Args:
            symbol: 交易对信息
            rank: 排名

        Returns:
            格式化的结果字典
        """
        return {
            'rank': rank,
            'symbol': symbol.inst_id,
            'last_price': symbol.last_price,
            'price_change_24h': symbol.price_change_24h,
            'volume_24h': symbol.volume_24h,
            'high_24h': symbol.high_24h,
            'low_24h': symbol.low_24h,
            'extra_data': symbol.extra_data,
        }

    def scan_symbol(self, symbol: ScannerSymbol) -> Dict:
        """
        单个交易对扫描（兼容基类接口）
        
        注意：这个策略主要使用 scan_all_symbols 方法
        这个方法只是为了兼容性
        """
        # 检查成交量
        min_volume = self.config.get('min_volume_24h', 100000)
        passed = symbol.volume_24h >= min_volume

        return {
            'symbol': symbol.inst_id,
            'passed': passed,
            'conditions_met': 1 if passed else 0,
            'conditions_total': 1,
            'details': {
                '成交量': '通过' if passed else '未通过',
                '涨跌幅': f"{symbol.price_change_24h:.2f}%"
            },
            'score': 100.0 if passed else 0.0,
            'last_price': symbol.last_price,
            'volume_24h': symbol.volume_24h,
            'price_change_24h': symbol.price_change_24h,
            'high_24h': symbol.high_24h,
            'low_24h': symbol.low_24h,
        }

    def get_config_schema(self) -> Dict:
        """获取配置模式"""
        return {
            'top_n': {
                'type': 'int',
                'default': 10,
                'label': '显示前N名'
            },
            'min_volume_24h': {
                'type': 'float',
                'default': 100000,
                'label': '最小 24h 成交量 (USDT)'
            },
        }


# 策略配置模式
CONFIG_SCHEMA = {
    'top_n': {'type': 'int', 'default': 10, 'label': '显示前N名'},
    'min_volume_24h': {'type': 'float', 'default': 0, 'label': '最小 24h 成交量 (USDT)'},
}
