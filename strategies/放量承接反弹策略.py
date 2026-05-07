"""
成交量异常放大 + 价格不跌策略

大成交量砸不下去，代表承接强，后续容易反弹或继续走强。
"""

from typing import Dict, List

import pandas as pd

from src.scanner.base_scanner import BaseScannerStrategy, ScanCondition, ScannerSymbol
from src.scanner.ranking import build_opportunity_profile
from strategies._shared.indicators import _to_df


class VolumeAbsorptionReboundScanner(BaseScannerStrategy):
    required_bars = ['4H', '1H']

    def _init_conditions(self):
        self.add_condition(ScanCondition("24H成交量", "过滤低流动性", "volume_24h", ">=", self.config.get('min_volume_24h', 10000000)))

    def scan_symbol(self, symbol: ScannerSymbol) -> Dict:
        try:
            klines = symbol.extra_data.get('klines', {})
            h1 = _to_df(self._get_klines(klines, '1H'))
            h4 = _to_df(self._get_klines(klines, '4H'))
            if len(h1) < 100 or len(h4) < 60:
                return self._fail(symbol, f"数据不足(1H={len(h1)},4H={len(h4)})")

            vol_ratio = self._volume_ratio(h1)
            recent_drop = self._drawdown_from_high(h1, 24)
            close_position = self._range_position(h1.tail(12))
            lower_wick = self._lower_wick_ratio(h1.iloc[-1])
            h4_trend = self._momentum(h4['c'], 12)
            h1_recover = self._momentum(h1['c'], 4)
            rsi = self._rsi(h1['c'])

            score = 0.0
            signals = []
            if vol_ratio >= float(self.config.get('min_volume_spike_ratio', 2.0)):
                score += 30
                signals.append(f"成交量异常放大({vol_ratio:.2f}x)")
            if recent_drop <= float(self.config.get('max_recent_drop_pct', 4.5)):
                score += 20
                signals.append(f"放量但价格不跌({recent_drop:.2f}%)")
            if close_position >= 55:
                score += 16
                signals.append(f"收盘位于区间上半部({close_position:.0f})")
            if lower_wick >= 0.32:
                score += 12
                signals.append(f"下影线承接({lower_wick:.0%})")
            if h1_recover >= 0:
                score += 10
                signals.append("短线已止跌")
            if h4_trend >= -3:
                score += 8
            else:
                signals.append(f"4H仍偏弱({h4_trend:.2f}%)")
            if 35 <= rsi <= 62:
                score += 8
                signals.append(f"RSI承接区({rsi:.1f})")

            direction = 'BUY' if score >= float(self.config.get('min_score', 78)) else 'WAIT'
            profile = build_opportunity_profile(
                base_score=score,
                direction=direction,
                volume_24h=symbol.volume_24h,
                factors={
                    'trend': 72 if h4_trend >= -3 else 45,
                    'trigger': 90 if direction == 'BUY' else 40,
                    'volume': min(vol_ratio, 3.0) / 3.0 * 100,
                    'location': close_position,
                    'freshness': 88 if h1_recover >= 0 else 55,
                    'risk': 82 if recent_drop <= 4.5 else 55,
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
                    '评估': ' | '.join(signals) if signals else '暂无放量承接机会',
                    '量比': f'{vol_ratio:.2f}x',
                    '近24根回撤': f'{recent_drop:.2f}%',
                    '收盘区间位置': f'{close_position:.0f}',
                    '下影线': f'{lower_wick:.0%}',
                    'RSI': f'{rsi:.1f}',
                },
                'last_price': symbol.last_price,
                'volume_24h': symbol.volume_24h,
                'price_change_24h': symbol.price_change_24h,
                'category': '放量承接',
                **profile,
            }
        except Exception as exc:
            return self._fail(symbol, f"分析异常: {exc}")

    def _fail(self, symbol, reason):
        return {'symbol': symbol.inst_id, 'passed': False, 'score': 0.0, 'direction': 'WAIT', 'details': {'状态': reason}}

    def _get_klines(self, klines_map: Dict[str, List], bar: str) -> List:
        return klines_map.get(bar) or klines_map.get(bar.lower()) or klines_map.get(bar.upper()) or []



    def _volume_ratio(self, df):
        avg = float(df['vol'].iloc[-31:-1].mean()) if len(df) >= 31 else 0.0
        return float(df['vol'].iloc[-1] / avg) if avg > 0 else 1.0

    def _drawdown_from_high(self, df, lookback):
        recent = df.tail(lookback)
        high = float(recent['h'].max())
        close = float(recent['c'].iloc[-1])
        return (high - close) / high * 100 if high > 0 else 0.0

    def _range_position(self, df):
        high = float(df['h'].max())
        low = float(df['l'].min())
        close = float(df['c'].iloc[-1])
        return (close - low) / (high - low) * 100 if high > low else 50.0

    def _lower_wick_ratio(self, row):
        high, low, open_price, close = float(row['h']), float(row['l']), float(row['o']), float(row['c'])
        rng = high - low
        return max(min(open_price, close) - low, 0.0) / rng if rng > 0 else 0.0

    def _momentum(self, close, lookback):
        if len(close) <= lookback or close.iloc[-(lookback + 1)] <= 0:
            return 0.0
        return (float(close.iloc[-1]) / float(close.iloc[-(lookback + 1)]) - 1.0) * 100

    def _rsi(self, close, period=14):
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
        if len(loss) == 0 or pd.isna(loss.iloc[-1]) or loss.iloc[-1] == 0:
            return 50.0
        return float(100 - (100 / (1 + gain.iloc[-1] / loss.iloc[-1])))

    def get_config_schema(self) -> Dict:
        return {
            'min_score': {'type': 'int', 'default': 78, 'label': '最低通过分数'},
            'min_volume_24h': {'type': 'float', 'default': 10000000, 'label': '最小24H成交额'},
            'min_volume_spike_ratio': {'type': 'float', 'default': 2.0, 'label': '最小放量倍数'},
            'max_recent_drop_pct': {'type': 'float', 'default': 4.5, 'label': '最大近端回撤%'},
        }


STRATEGY_NAME  = "放量承接反弹策略"
STRATEGY_TYPE  = "scan"
STRATEGY_CLASS = VolumeAbsorptionReboundScanner
BACKTEST_CLASS = VolumeAbsorptionReboundScanner
