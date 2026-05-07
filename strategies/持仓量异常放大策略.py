"""
持仓量异常放大策略

结合价格方向和持仓量/量能变化判断趋势增强、空头回补、主动打压或止损释放。
"""

from typing import Dict, List

import pandas as pd

from src.scanner.base_scanner import BaseScannerStrategy, ScanCondition, ScannerSymbol
from src.scanner.ranking import build_opportunity_profile
from strategies._shared.indicators import _to_df


class OpenInterestAnomalyScanner(BaseScannerStrategy):
    required_bars = ['4H', '1H']
    requires_derivative_metrics = True

    def _init_conditions(self):
        self.add_condition(ScanCondition("24H成交量", "过滤低流动性", "volume_24h", ">=", self.config.get('min_volume_24h', 12000000)))

    def scan_symbol(self, symbol: ScannerSymbol) -> Dict:
        try:
            klines = symbol.extra_data.get('klines', {})
            h1 = _to_df(self._get_klines(klines, '1H'))
            h4 = _to_df(self._get_klines(klines, '4H'))
            if len(h1) < 80 or len(h4) < 60:
                return self._fail(symbol, f"数据不足(1H={len(h1)},4H={len(h4)})")

            price_4h = self._momentum(h1['c'], 4)
            price_24h = float(symbol.price_change_24h or self._momentum(h1['c'], 24))
            oi_change = self._safe_float(symbol.extra_data.get('open_interest_change_24h'), None)
            oi_value = self._safe_float(symbol.extra_data.get('open_interest') or symbol.open_interest, 0.0)
            volume_ratio = self._volume_ratio(h1)
            # OKX 当前接口只有当前持仓量时，用放量作为“参与度异常”的保守代理。
            participation_change = oi_change if oi_change is not None else (volume_ratio - 1.0) * 18.0
            used_proxy = oi_change is None

            score = 0.0
            direction = 'WAIT'
            signals = []

            if participation_change >= float(self.config.get('min_participation_change_pct', 8.0)):
                score += 26
                signals.append(
                    f"{'量能代理' if used_proxy else '持仓量'}异常放大({participation_change:.2f}%)"
                )

            if price_24h > 2.0 and participation_change > 6:
                score += 30
                direction = 'BUY'
                signals.append("价格上涨 + 持仓/参与度上涨：趋势增强")
            elif price_24h > 2.0 and participation_change < -4:
                score += 16
                direction = 'WAIT'
                signals.append("价格上涨 + 持仓下降：疑似空头回补，持续性打折")
            elif price_24h < -2.0 and participation_change > 6:
                score += 30
                direction = 'SELL'
                signals.append("价格下跌 + 持仓/参与度上涨：空头主动打压")
            elif price_24h < -2.0 and participation_change < -4:
                score += 22
                direction = 'BUY'
                signals.append("价格下跌 + 持仓下降：止损释放，关注反弹")

            if abs(price_4h) >= 1.2:
                score += 10
                signals.append(f"4H方向确认({price_4h:.2f}%)")
            if volume_ratio >= 1.3:
                score += 12
                signals.append(f"成交活跃({volume_ratio:.2f}x)")
            if oi_value > 0:
                score += 6
                signals.append("持仓量数据有效")

            profile = build_opportunity_profile(
                base_score=score,
                direction=direction,
                volume_24h=symbol.volume_24h,
                factors={
                    'trend': min(100, 50 + abs(price_24h) * 5),
                    'trigger': 88 if direction != 'WAIT' else 42,
                    'volume': min(volume_ratio, 2.0) / 2.0 * 100,
                    'location': 70,
                    'freshness': min(abs(participation_change), 20) * 5,
                    'risk': 72 if not used_proxy else 58,
                },
                signals=signals,
            )
            return {
                'symbol': symbol.inst_id,
                'passed': score >= float(self.config.get('min_score', 76)) and direction != 'WAIT',
                'score': round(score, 2),
                'direction': direction,
                'signals': signals,
                'details': {
                    '评估': ' | '.join(signals) if signals else '暂无持仓量异常机会',
                    '24H涨跌': f'{price_24h:.2f}%',
                    '4H涨跌': f'{price_4h:.2f}%',
                    '参与度变化': f'{participation_change:.2f}%',
                    '当前持仓量': f'{oi_value:.2f}',
                    '量比': f'{volume_ratio:.2f}x',
                },
                'last_price': symbol.last_price,
                'volume_24h': symbol.volume_24h,
                'price_change_24h': symbol.price_change_24h,
                'category': '持仓量异常',
                **profile,
            }
        except Exception as exc:
            return self._fail(symbol, f"分析异常: {exc}")

    def _fail(self, symbol, reason):
        return {'symbol': symbol.inst_id, 'passed': False, 'score': 0.0, 'direction': 'WAIT', 'details': {'状态': reason}}

    def _safe_float(self, value, default=0.0):
        try:
            if value is None or value == '':
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

    def _get_klines(self, klines_map: Dict[str, List], bar: str) -> List:
        return klines_map.get(bar) or klines_map.get(bar.lower()) or klines_map.get(bar.upper()) or []



    def _momentum(self, close: pd.Series, lookback: int) -> float:
        if len(close) <= lookback or close.iloc[-(lookback + 1)] <= 0:
            return 0.0
        return (float(close.iloc[-1]) / float(close.iloc[-(lookback + 1)]) - 1.0) * 100

    def _volume_ratio(self, df: pd.DataFrame) -> float:
        avg = float(df['vol'].iloc[-21:-1].mean()) if len(df) >= 21 else 0.0
        return float(df['vol'].iloc[-1] / avg) if avg > 0 else 1.0

    def get_config_schema(self) -> Dict:
        return {
            'min_score': {'type': 'int', 'default': 76, 'label': '最低通过分数'},
            'min_volume_24h': {'type': 'float', 'default': 12000000, 'label': '最小24H成交额'},
            'min_participation_change_pct': {'type': 'float', 'default': 8.0, 'label': '最小持仓/参与度变化%'},
        }


STRATEGY_NAME  = "持仓量异常放大策略"
STRATEGY_TYPE  = "scan"
STRATEGY_CLASS = OpenInterestAnomalyScanner
BACKTEST_CLASS = OpenInterestAnomalyScanner
