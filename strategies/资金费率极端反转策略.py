"""
资金费率极端反转策略

资金费率极端偏多且价格滞涨，容易出现多杀多；资金费率极端偏空且价格拒绝下跌，容易出现空头回补。
"""

from typing import Dict, List

import pandas as pd

from src.scanner.base_scanner import BaseScannerStrategy, ScanCondition, ScannerSymbol
from src.scanner.ranking import build_opportunity_profile
from strategies._shared.indicators import _to_df


class ExtremeFundingReversalScanner(BaseScannerStrategy):
    required_bars = ['4H', '1H']
    requires_derivative_metrics = True

    def _init_conditions(self):
        self.add_condition(ScanCondition(
            name="24H成交量",
            description="过滤流动性不足标的",
            field="volume_24h",
            operator=">=",
            value=self.config.get('min_volume_24h', 15000000),
        ))

    def scan_symbol(self, symbol: ScannerSymbol) -> Dict:
        try:
            klines = symbol.extra_data.get('klines', {})
            h4 = _to_df(self._get_klines(klines, '4H'))
            h1 = _to_df(self._get_klines(klines, '1H'))
            if len(h4) < 80:
                return self._fail(symbol, f"4H数据不足({len(h4)}/80)")
            if len(h1) < 80:
                return self._fail(symbol, f"1H数据不足({len(h1)}/80)")

            funding_rate = float(symbol.extra_data.get('funding_rate', 0.0) or 0.0) * 100
            if funding_rate == 0:
                return self._fail(symbol, "缺少资金费率数据")

            pos_threshold = float(self.config.get('positive_funding_threshold_pct', 0.08))
            neg_threshold = -abs(float(self.config.get('negative_funding_threshold_pct', 0.08)))
            h4_momentum = self._momentum(h4['c'], 12)
            h1_momentum = self._momentum(h1['c'], 12)
            rsi = self._rsi(h1['c'])
            volume_ratio = self._volume_ratio(h1)
            upper_wick, lower_wick = self._wick_ratio(h1.iloc[-1])

            score = 0.0
            direction = 'WAIT'
            signals = []

            if funding_rate >= pos_threshold:
                score += 34
                signals.append(f"资金费率极端偏多({funding_rate:.4f}%)")
                if h4_momentum < float(self.config.get('max_stall_momentum_pct', 2.0)) and h1_momentum <= 0.8:
                    score += 24
                    signals.append("价格滞涨")
                if rsi >= float(self.config.get('overheated_rsi', 62)):
                    score += 12
                    signals.append(f"RSI偏热({rsi:.1f})")
                if upper_wick >= 0.35:
                    score += 10
                    signals.append("上影线抛压")
                if volume_ratio >= 1.2:
                    score += 10
                    signals.append(f"量能放大({volume_ratio:.2f}x)")
                direction = 'SELL' if score >= 70 else 'WAIT'

            elif funding_rate <= neg_threshold:
                score += 34
                signals.append(f"资金费率极端偏空({funding_rate:.4f}%)")
                if h4_momentum > -float(self.config.get('max_stall_momentum_pct', 2.0)) and h1_momentum >= -0.8:
                    score += 24
                    signals.append("价格拒绝下跌")
                if rsi <= float(self.config.get('oversold_rsi', 42)):
                    score += 12
                    signals.append(f"RSI偏冷({rsi:.1f})")
                if lower_wick >= 0.35:
                    score += 10
                    signals.append("下影线承接")
                if volume_ratio >= 1.2:
                    score += 10
                    signals.append(f"量能放大({volume_ratio:.2f}x)")
                direction = 'BUY' if score >= 70 else 'WAIT'

            profile = build_opportunity_profile(
                base_score=score,
                direction=direction,
                volume_24h=symbol.volume_24h,
                factors={
                    'trend': 55,
                    'trigger': 92 if direction != 'WAIT' else 38,
                    'volume': min(volume_ratio, 1.8) / 1.8 * 100,
                    'location': max(30, 100 - abs(h4_momentum) * 8),
                    'freshness': min(abs(funding_rate) / max(pos_threshold, abs(neg_threshold), 0.01), 1.8) * 55,
                    'risk': 78 if volume_ratio >= 1.0 else 55,
                },
                signals=signals,
            )
            return {
                'symbol': symbol.inst_id,
                'passed': score >= float(self.config.get('min_score', 78)) and direction != 'WAIT',
                'score': round(max(score, 0.0), 2),
                'direction': direction,
                'signals': signals,
                'details': {
                    '评估': ' | '.join(signals) if signals else '暂无资金费率极端反转机会',
                    '资金费率%': f'{funding_rate:.4f}',
                    '4H动量': f'{h4_momentum:.2f}%',
                    '1H动量': f'{h1_momentum:.2f}%',
                    'RSI': f'{rsi:.1f}',
                    '量比': f'{volume_ratio:.2f}x',
                },
                'last_price': symbol.last_price,
                'volume_24h': symbol.volume_24h,
                'price_change_24h': symbol.price_change_24h,
                'category': '资金费率反转',
                **profile,
            }
        except Exception as exc:
            return self._fail(symbol, f"分析异常: {exc}")

    def _fail(self, symbol, reason):
        return {'symbol': symbol.inst_id, 'passed': False, 'score': 0.0, 'direction': 'WAIT', 'details': {'状态': reason}}

    def _get_klines(self, klines_map: Dict[str, List], bar: str) -> List:
        return klines_map.get(bar) or klines_map.get(bar.lower()) or klines_map.get(bar.upper()) or []



    def _momentum(self, close: pd.Series, lookback: int) -> float:
        if len(close) <= lookback or close.iloc[-(lookback + 1)] <= 0:
            return 0.0
        return (float(close.iloc[-1]) / float(close.iloc[-(lookback + 1)]) - 1.0) * 100

    def _volume_ratio(self, df: pd.DataFrame) -> float:
        avg = float(df['vol'].iloc[-21:-1].mean()) if len(df) >= 21 else 0.0
        return float(df['vol'].iloc[-1] / avg) if avg > 0 else 1.0

    def _rsi(self, close: pd.Series, period: int = 14) -> float:
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
        if len(loss) == 0 or pd.isna(loss.iloc[-1]) or loss.iloc[-1] == 0:
            return 50.0
        return float(100 - (100 / (1 + gain.iloc[-1] / loss.iloc[-1])))

    def _wick_ratio(self, row) -> tuple:
        high, low, open_price, close = float(row['h']), float(row['l']), float(row['o']), float(row['c'])
        rng = max(high - low, 0.0)
        if rng <= 0:
            return 0.0, 0.0
        upper = max(high - max(open_price, close), 0.0) / rng
        lower = max(min(open_price, close) - low, 0.0) / rng
        return upper, lower

    def get_config_schema(self) -> Dict:
        return {
            'min_score': {'type': 'int', 'default': 78, 'label': '最低通过分数'},
            'min_volume_24h': {'type': 'float', 'default': 15000000, 'label': '最小24H成交额'},
            'positive_funding_threshold_pct': {'type': 'float', 'default': 0.08, 'label': '极端正资金费率%'},
            'negative_funding_threshold_pct': {'type': 'float', 'default': 0.08, 'label': '极端负资金费率%'},
        }


STRATEGY_NAME  = "资金费率极端反转策略"
STRATEGY_TYPE  = "scan"
STRATEGY_CLASS = ExtremeFundingReversalScanner
BACKTEST_CLASS = ExtremeFundingReversalScanner
