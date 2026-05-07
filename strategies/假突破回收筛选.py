"""
经典反转策略：假突破回收筛选

核心思想：
1. 价格短暂扫破 1H 关键支撑/压力，诱发追单或止损。
2. 随后收盘重新回到关键位内侧，形成“扫流动性后回收”。
3. 结合长影线、量能确认和 4H 波动过滤，筛出更值得跟踪的反转机会。
"""

from typing import Any, Dict, List

import pandas as pd

from src.scanner.base_scanner import BaseScannerStrategy, ScanCondition, ScannerSymbol
from src.scanner.ranking import build_opportunity_profile
from strategies._shared.indicators import _to_df


class FalseBreakoutReclaimScanner(BaseScannerStrategy):
    required_bars = ['4H', '1H']

    def _init_conditions(self):
        self.add_condition(ScanCondition(
            name="24H成交量",
            description="过滤流动性不足标的",
            field="volume_24h",
            operator=">=",
            value=self.config.get('min_volume_24h', 12000000),
        ))

    def scan_symbol(self, symbol: ScannerSymbol) -> Dict:
        klines_map = symbol.extra_data.get('klines', {})
        try:
            analysis = self._analyze(
                self._get_klines(klines_map, '4H'),
                self._get_klines(klines_map, '1H'),
                symbol.last_price,
            )
        except Exception as exc:
            return {
                'symbol': symbol.inst_id,
                'passed': False,
                'score': 0.0,
                'direction': 'WAIT',
                'details': {'状态': f'分析异常: {exc}'},
            }

        if not analysis['valid']:
            return {
                'symbol': symbol.inst_id,
                'passed': False,
                'score': 0.0,
                'direction': 'WAIT',
                'details': {'状态': analysis['reason']},
            }

        min_score = float(self.config.get('min_score', 80))
        profile = build_opportunity_profile(
            base_score=analysis['score'],
            direction=analysis['direction'],
            volume_24h=symbol.volume_24h,
            factors=analysis.get('ranking_factors', {}),
            signals=analysis['signals'],
        )
        return {
            'symbol': symbol.inst_id,
            'passed': analysis['score'] >= min_score and analysis['direction'] in {'BUY', 'SELL'},
            'score': round(analysis['score'], 2),
            'direction': analysis['direction'],
            'signals': analysis['signals'],
            'details': analysis['details'],
            'last_price': symbol.last_price,
            'volume_24h': symbol.volume_24h,
            'price_change_24h': symbol.price_change_24h,
            'category': '假突破回收',
            'ranking_factors': analysis.get('ranking_factors', {}),
            **profile,
        }

    def _analyze(self, h4_klines: List, h1_klines: List, last_price: float) -> Dict[str, Any]:
        h4 = _to_df(h4_klines)
        h1 = _to_df(h1_klines)
        lookback = int(self.config.get('lookback_window', 48))
        confirm_bars = int(self.config.get('confirm_bars', 3))
        min_bars = max(lookback + confirm_bars + 8, 80)

        if len(h4) < 70:
            return {'valid': False, 'reason': f'4H数据不足({len(h4)}/70)'}
        if len(h1) < min_bars:
            return {'valid': False, 'reason': f'1H数据不足({len(h1)}/{min_bars})'}

        recent = h1.tail(confirm_bars).copy().reset_index(drop=True)
        history_end = len(h1) - confirm_bars
        history_start = max(0, history_end - lookback)
        history = h1.iloc[history_start:history_end].copy()
        if len(history) < max(20, lookback // 2) or recent.empty:
            return {'valid': False, 'reason': '关键位样本不足'}

        support = float(history['l'].min())
        resistance = float(history['h'].max())
        price = float(last_price if last_price > 0 else h1['c'].iloc[-1])
        last_open = float(h1['o'].iloc[-1])
        last_close = float(h1['c'].iloc[-1])
        recent_low = float(recent['l'].min())
        recent_high = float(recent['h'].max())
        latest_low = float(h1['l'].iloc[-1])
        latest_high = float(h1['h'].iloc[-1])

        sweep_buffer = float(self.config.get('sweep_buffer_pct', 0.25)) / 100.0
        reclaim_buffer = float(self.config.get('reclaim_buffer_pct', 0.05)) / 100.0
        max_reclaim_distance = float(self.config.get('max_reclaim_distance_pct', 2.5))
        min_wick_ratio = float(self.config.get('min_wick_ratio', 0.42))
        min_volume_ratio = float(self.config.get('min_volume_ratio', 1.25))

        lower_wick_ratio, upper_wick_ratio = self._wick_ratios(h1.iloc[-1])
        volume_ratio = self._volume_ratio(h1)
        h4_rsi = self._rsi(h4['c'])
        h1_rsi = self._rsi(h1['c'])
        h4_ema20 = float(h4['c'].ewm(span=20, adjust=False).mean().iloc[-1])
        h4_ema50 = float(h4['c'].ewm(span=50, adjust=False).mean().iloc[-1])
        h4_slope = self._ema_slope_pct(h4['c'], 20, 6)
        atr_pct = self._atr_pct(h4)

        bull_sweep = support > 0 and recent_low < support * (1.0 - sweep_buffer)
        bull_reclaim = support > 0 and last_close > support * (1.0 + reclaim_buffer)
        bear_sweep = resistance > 0 and recent_high > resistance * (1.0 + sweep_buffer)
        bear_reclaim = resistance > 0 and last_close < resistance * (1.0 - reclaim_buffer)
        bullish_context = not (price < h4_ema20 < h4_ema50 and h4_slope < -1.2 and h4_rsi < 34)
        bearish_context = not (price > h4_ema20 > h4_ema50 and h4_slope > 1.2 and h4_rsi > 66)

        bull_distance = abs((last_close - support) / support * 100) if support > 0 else 999.0
        bear_distance = abs((last_close - resistance) / resistance * 100) if resistance > 0 else 999.0
        score = 0.0
        signals = []
        direction = 'WAIT'
        location_distance = min(bull_distance, bear_distance)

        if bull_sweep and bull_reclaim and bullish_context:
            score += 38
            direction = 'BUY'
            location_distance = bull_distance
            signals.append(f"假跌破支撑后回收({support:.6g})")
            if last_close > last_open:
                score += 10
                signals.append("收阳确认")
            if lower_wick_ratio >= min_wick_ratio or latest_low <= support * (1.0 - sweep_buffer):
                score += 14
                signals.append(f"下影线拒绝({lower_wick_ratio:.0%})")
            if 28 <= h1_rsi <= 58:
                score += 8
                signals.append(f"RSI处于回收反转区({h1_rsi:.1f})")

        if bear_sweep and bear_reclaim and bearish_context:
            score += 38
            direction = 'SELL'
            location_distance = bear_distance
            signals.append(f"假突破压力后回落({resistance:.6g})")
            if last_close < last_open:
                score += 10
                signals.append("收阴确认")
            if upper_wick_ratio >= min_wick_ratio or latest_high >= resistance * (1.0 + sweep_buffer):
                score += 14
                signals.append(f"上影线拒绝({upper_wick_ratio:.0%})")
            if 42 <= h1_rsi <= 72:
                score += 8
                signals.append(f"RSI处于回落反转区({h1_rsi:.1f})")

        if direction in {'BUY', 'SELL'} and volume_ratio >= min_volume_ratio:
            score += 14
            signals.append(f"扫盘量能确认({volume_ratio:.2f}x)")
        elif direction in {'BUY', 'SELL'}:
            score -= 8
            signals.append(f"量能不足({volume_ratio:.2f}x)")

        if direction in {'BUY', 'SELL'} and location_distance <= max_reclaim_distance:
            score += 10
            signals.append(f"回收后仍贴近关键位({location_distance:.2f}%)")
        elif direction in {'BUY', 'SELL'}:
            score -= 8
            signals.append(f"回收后离关键位偏远({location_distance:.2f}%)")

        max_atr = float(self.config.get('max_h4_atr_pct', 7.0))
        if atr_pct <= max_atr:
            score += 6
        else:
            score -= 10
            signals.append(f"4H波动过大({atr_pct:.2f}%)")

        if direction == 'BUY' and h4_rsi <= 62:
            score += 6
            signals.append(f"4H未明显过热({h4_rsi:.1f})")
        elif direction == 'SELL' and h4_rsi >= 38:
            score += 6
            signals.append(f"4H未明显过冷({h4_rsi:.1f})")

        if direction == 'WAIT':
            location_distance = min(
                abs((price - support) / support * 100) if support > 0 else 999.0,
                abs((price - resistance) / resistance * 100) if resistance > 0 else 999.0,
            )

        trigger_quality = 94.0 if direction in {'BUY', 'SELL'} else 28.0
        volume_quality = min(volume_ratio / max(min_volume_ratio, 0.1), 1.6) * 62.5
        location_quality = max(20.0, 100.0 - location_distance * 22.0)
        trend_quality = 72.0
        if direction == 'BUY' and price >= h4_ema20:
            trend_quality = 82.0
        elif direction == 'SELL' and price <= h4_ema20:
            trend_quality = 82.0

        return {
            'valid': True,
            'score': max(0.0, min(score, 100.0)),
            'direction': direction,
            'signals': signals,
            'ranking_factors': {
                'trend': trend_quality,
                'trigger': trigger_quality,
                'volume': volume_quality,
                'location': location_quality,
                'freshness': 96.0 if direction in {'BUY', 'SELL'} else 35.0,
                'risk': 88.0 if atr_pct <= max_atr and location_distance <= max_reclaim_distance else 55.0,
            },
            'details': {
                '评估': ' | '.join(signals) if signals else '暂无假突破回收机会',
                '关键支撑': f'{support:.6g}',
                '关键压力': f'{resistance:.6g}',
                '回收距离': f'{location_distance:.2f}%',
                '量比': f'{volume_ratio:.2f}x',
                '下影线占比': f'{lower_wick_ratio:.0%}',
                '上影线占比': f'{upper_wick_ratio:.0%}',
                '1H_RSI': f'{h1_rsi:.1f}',
                '4H_RSI': f'{h4_rsi:.1f}',
                '4H_ATR%': f'{atr_pct:.2f}%',
            },
        }

    def _wick_ratios(self, row: pd.Series) -> tuple:
        high = float(row.get('h', 0.0))
        low = float(row.get('l', 0.0))
        open_price = float(row.get('o', 0.0))
        close = float(row.get('c', 0.0))
        candle_range = max(high - low, 0.0)
        if candle_range <= 0:
            return 0.0, 0.0
        lower_wick = max(min(open_price, close) - low, 0.0)
        upper_wick = max(high - max(open_price, close), 0.0)
        return lower_wick / candle_range, upper_wick / candle_range

    def _ema_slope_pct(self, close: pd.Series, span: int, lookback: int) -> float:
        ema = close.ewm(span=span, adjust=False).mean()
        if len(ema) <= lookback:
            return 0.0
        base = float(ema.iloc[-(lookback + 1)])
        latest = float(ema.iloc[-1])
        return (latest / base - 1.0) * 100 if base > 0 else 0.0

    def _volume_ratio(self, df: pd.DataFrame) -> float:
        if len(df) < 21 or df['vol'].empty:
            return 1.0
        avg = float(df['vol'].iloc[-21:-1].mean())
        return float(df['vol'].iloc[-1] / avg) if avg > 0 else 1.0

    def _atr_pct(self, df: pd.DataFrame, period: int = 14) -> float:
        if len(df) < period + 1:
            return 0.0
        high = df['h']
        low = df['l']
        close = df['c']
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(period).mean().iloc[-1]
        if pd.isna(atr) or close.empty or close.iloc[-1] <= 0:
            return 0.0
        return float((atr / close.iloc[-1]) * 100)

    def _rsi(self, close: pd.Series, period: int = 14) -> float:
        if len(close) <= period:
            return 50.0
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
        if len(loss) == 0 or len(gain) == 0 or pd.isna(loss.iloc[-1]) or pd.isna(gain.iloc[-1]):
            return 50.0
        if loss.iloc[-1] == 0:
            return 100.0 if gain.iloc[-1] > 0 else 50.0
        rs = gain.iloc[-1] / loss.iloc[-1]
        return float(100 - (100 / (1 + rs)))

    def _get_klines(self, klines_map: Dict[str, List], bar: str) -> List:
        return klines_map.get(bar) or klines_map.get(bar.lower()) or klines_map.get(bar.upper()) or []



    def get_config_schema(self) -> Dict:
        return {
            'min_score': {'type': 'int', 'default': 80, 'label': '最低通过分数'},
            'min_volume_24h': {'type': 'float', 'default': 12000000, 'label': '最小24H成交额'},
            'lookback_window': {'type': 'int', 'default': 48, 'label': '关键位回看1H根数'},
            'confirm_bars': {'type': 'int', 'default': 3, 'label': '回收确认最近K线数'},
            'sweep_buffer_pct': {'type': 'float', 'default': 0.25, 'label': '扫破关键位幅度%'},
            'reclaim_buffer_pct': {'type': 'float', 'default': 0.05, 'label': '收回关键位幅度%'},
            'min_wick_ratio': {'type': 'float', 'default': 0.42, 'label': '最小影线占比'},
            'min_volume_ratio': {'type': 'float', 'default': 1.25, 'label': '最小扫盘量比'},
            'max_reclaim_distance_pct': {'type': 'float', 'default': 2.5, 'label': '最大回收距离%'},
            'max_h4_atr_pct': {'type': 'float', 'default': 7.0, 'label': '最大4H ATR%'},
        }


STRATEGY_NAME  = "假突破回收筛选"
STRATEGY_TYPE  = "scan"
STRATEGY_CLASS = FalseBreakoutReclaimScanner
BACKTEST_CLASS = FalseBreakoutReclaimScanner
