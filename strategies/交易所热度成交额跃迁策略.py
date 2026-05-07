"""
交易所热度/成交额突增策略

24H 成交额突然进入前排，同时短周期继续放量，是行情启动早期常见信号。
"""

from typing import Dict, List

import pandas as pd

from src.scanner.base_scanner import BaseScannerStrategy, ScanCondition, ScannerSymbol
from src.scanner.ranking import build_opportunity_profile, sort_scan_results
from strategies._shared.indicators import _to_df


class ExchangeHeatVolumeJumpScanner(BaseScannerStrategy):
    required_bars = ['1H']

    def _init_conditions(self):
        self.add_condition(ScanCondition("24H成交量", "过滤低流动性", "volume_24h", ">=", self.config.get('min_volume_24h', 8000000)))

    def scan_all_symbols(self, symbols: List[ScannerSymbol]) -> Dict:
        sorted_symbols = sorted(symbols, key=lambda s: float(s.volume_24h or 0.0), reverse=True)
        total = max(len(sorted_symbols), 1)
        rank_map = {s.inst_id: idx + 1 for idx, s in enumerate(sorted_symbols)}
        results = []
        for symbol in sorted_symbols:
            result = self.scan_symbol(symbol, rank_map.get(symbol.inst_id, total), total)
            if result.get('passed'):
                results.append(result)
        return {'type': 'exchange_heat_volume_jump_scan', 'all_opportunities': sort_scan_results(results)[:int(self.config.get('top_n', 30))]}

    def scan_symbol(self, symbol: ScannerSymbol, rank: int = 9999, total: int = 9999) -> Dict:
        try:
            h1 = _to_df(self._get_klines(symbol.extra_data.get('klines', {}), '1H'))
            if len(h1) < 80:
                return {'symbol': symbol.inst_id, 'passed': False, 'score': 0.0, 'direction': 'WAIT', 'details': {'状态': f'1H数据不足({len(h1)}/80)'}}
            top_pct = rank / max(total, 1) * 100
            vol_ratio = self._volume_ratio(h1)
            price_momentum = self._momentum(h1['c'], 12)
            close_position = self._range_position(h1.tail(24))

            score = 0.0
            signals = []
            if top_pct <= float(self.config.get('top_volume_percentile', 18.0)):
                score += 30
                signals.append(f"成交额排名进入前{top_pct:.1f}%")
            if vol_ratio >= float(self.config.get('min_intraday_volume_ratio', 1.8)):
                score += 26
                signals.append(f"1H成交额突增({vol_ratio:.2f}x)")
            if price_momentum >= float(self.config.get('min_price_momentum_pct', 1.5)):
                score += 20
                signals.append(f"价格同步启动({price_momentum:.2f}%)")
            elif close_position >= 60:
                score += 12
                signals.append("价格尚未大涨但承接较强")
            if close_position >= 55:
                score += 10
                signals.append(f"区间位置偏强({close_position:.0f})")

            direction = 'BUY' if score >= float(self.config.get('min_score', 76)) else 'WAIT'
            profile = build_opportunity_profile(
                base_score=score,
                direction=direction,
                volume_24h=symbol.volume_24h,
                factors={
                    'trend': 76 if price_momentum >= 1.5 else 58,
                    'trigger': 88 if direction == 'BUY' else 42,
                    'volume': max(0, 100 - top_pct),
                    'location': close_position,
                    'freshness': min(vol_ratio, 3.0) / 3.0 * 100,
                    'risk': 74,
                },
                signals=signals,
            )
            return {
                'symbol': symbol.inst_id,
                'passed': direction == 'BUY',
                'score': round(score, 2),
                'direction': direction,
                'signals': signals,
                'details': {
                    '评估': ' | '.join(signals) if signals else '暂无热度跃迁机会',
                    '成交额排名': str(rank),
                    '排名百分位': f'{top_pct:.1f}%',
                    '1H量比': f'{vol_ratio:.2f}x',
                    '12H动量': f'{price_momentum:.2f}%',
                    '收盘区间位置': f'{close_position:.0f}',
                },
                'last_price': symbol.last_price,
                'volume_24h': symbol.volume_24h,
                'price_change_24h': symbol.price_change_24h,
                'category': '热度跃迁',
                **profile,
            }
        except Exception as exc:
            return {'symbol': symbol.inst_id, 'passed': False, 'score': 0.0, 'direction': 'WAIT', 'details': {'状态': f'分析异常: {exc}'}}

    def _get_klines(self, klines_map: Dict[str, List], bar: str) -> List:
        return klines_map.get(bar) or klines_map.get(bar.lower()) or klines_map.get(bar.upper()) or []


    def _volume_ratio(self, df):
        avg = float(df['vol'].iloc[-49:-1].mean()) if len(df) >= 49 else 0.0
        return float(df['vol'].iloc[-1] / avg) if avg > 0 else 1.0

    def _momentum(self, close, lookback):
        if len(close) <= lookback or close.iloc[-(lookback + 1)] <= 0:
            return 0.0
        return (float(close.iloc[-1]) / float(close.iloc[-(lookback + 1)]) - 1.0) * 100

    def _range_position(self, df):
        high = float(df['h'].max())
        low = float(df['l'].min())
        close = float(df['c'].iloc[-1])
        return (close - low) / (high - low) * 100 if high > low else 50.0

    def get_config_schema(self) -> Dict:
        return {
            'min_score': {'type': 'int', 'default': 76, 'label': '最低通过分数'},
            'min_volume_24h': {'type': 'float', 'default': 8000000, 'label': '最小24H成交额'},
            'top_volume_percentile': {'type': 'float', 'default': 18.0, 'label': '成交额排名前%'},
            'min_intraday_volume_ratio': {'type': 'float', 'default': 1.8, 'label': '1H最小量比'},
            'top_n': {'type': 'int', 'default': 30, 'label': '保留数量'},
        }


STRATEGY_NAME  = "交易所热度成交额跃迁策略"
STRATEGY_TYPE  = "scan"
STRATEGY_CLASS = ExchangeHeatVolumeJumpScanner
BACKTEST_CLASS = ExchangeHeatVolumeJumpScanner
