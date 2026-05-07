"""
经典突破策略：新高突破扫描

核心思想：
1. 先确认 1D / 4H 处于健康上升趋势，而不是震荡末端假突破。
2. 1H 必须刚突破近期关键高点，并伴随量能确认。
3. 过滤已经离均线太远、容易追高接力失败的标的。
"""

import pandas as pd
from typing import Any, Dict, List

from src.scanner.base_scanner import BaseScannerStrategy, ScanCondition, ScannerSymbol
from src.scanner.ranking import build_opportunity_profile
from strategies._shared.indicators import _to_df
from src.scanner.trend_quality import trend_quality_snapshot


class NewHighBreakoutScanner(BaseScannerStrategy):
    required_bars = ['1D', '4H', '1H']

    def _init_conditions(self):
        self.add_condition(ScanCondition(
            name="24H成交量",
            description="过滤流动性不足标的",
            field="volume_24h",
            operator=">=",
            value=self.config.get('min_volume_24h', 18000000),
        ))

    def scan_symbol(self, symbol: ScannerSymbol) -> Dict:
        klines_map = symbol.extra_data.get('klines', {})
        try:
            analysis = self._analyze(
                self._get_klines(klines_map, '1D'),
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

        min_score = float(self.config.get('min_score', 84))
        profile = build_opportunity_profile(
            base_score=analysis['score'],
            direction=analysis['direction'],
            volume_24h=symbol.volume_24h,
            factors=analysis.get('ranking_factors', {}),
            signals=analysis['signals'],
        )
        return {
            'symbol': symbol.inst_id,
            'passed': analysis['score'] >= min_score and analysis['direction'] == 'BUY',
            'score': round(analysis['score'], 2),
            'direction': analysis['direction'],
            'signals': analysis['signals'],
            'details': analysis['details'],
            'last_price': symbol.last_price,
            'volume_24h': symbol.volume_24h,
            'price_change_24h': symbol.price_change_24h,
            'ranking_factors': analysis.get('ranking_factors', {}),
            **profile,
        }

    def _analyze(self, d1_klines: List, h4_klines: List, h1_klines: List, last_price: float) -> Dict[str, Any]:
        d1 = _to_df(d1_klines)
        h4 = _to_df(h4_klines)
        h1 = _to_df(h1_klines)
        if len(d1) < 90:
            return {'valid': False, 'reason': f'日线数据不足({len(d1)}/90)'}
        if len(h4) < 120:
            return {'valid': False, 'reason': f'4H数据不足({len(h4)}/120)'}
        if len(h1) < 150:
            return {'valid': False, 'reason': f'1H数据不足({len(h1)}/150)'}

        score = 0.0
        signals = []

        price = float(last_price if last_price > 0 else h1['c'].iloc[-1])
        d1_close = d1['c']
        h4_close = h4['c']
        h1_close = h1['c']
        h1_vol = h1['vol']

        d1_ema20 = float(d1_close.ewm(span=20, adjust=False).mean().iloc[-1])
        d1_ema55 = float(d1_close.ewm(span=55, adjust=False).mean().iloc[-1])
        h4_ema20 = float(h4_close.ewm(span=20, adjust=False).mean().iloc[-1])
        h4_ema55 = float(h4_close.ewm(span=55, adjust=False).mean().iloc[-1])
        h1_ema20 = float(h1_close.ewm(span=20, adjust=False).mean().iloc[-1])

        d1_slope_pct = self._ema_slope_pct(d1_close, span=20, lookback=5)
        h4_slope_pct = self._ema_slope_pct(h4_close, span=20, lookback=5)
        h4_atr_pct = self._atr_pct(h4)
        h1_rsi = self._rsi(h1_close)
        trend_snapshot = trend_quality_snapshot(d1, h4, h1, price)
        trend_metrics = trend_snapshot.get('metrics', {})
        trend_long_score = float(trend_snapshot.get('long_score', 0.0))

        long_trend = (
            bool(trend_snapshot.get('long_ok'))
            and price > d1_ema20 > d1_ema55
            and price > h4_ema20 > h4_ema55
            and d1_slope_pct >= float(self.config.get('min_daily_slope_pct', 0.5))
            and h4_slope_pct >= float(self.config.get('min_h4_slope_pct', 0.8))
        )

        if long_trend:
            trend_bonus = min(34.0, 22.0 + max(trend_long_score - 68.0, 0.0) * 0.35)
            score += trend_bonus
            signals.append(f"多周期趋势质量通过({trend_long_score:.0f})")
        elif trend_long_score >= float(self.config.get('min_watch_trend_quality', 58.0)):
            score += 8
            signals.append(f"趋势质量观察级({trend_long_score:.0f})")

        breakout_window = int(self.config.get('breakout_window', 55))
        breakout_buffer_pct = float(self.config.get('breakout_buffer_pct', 0.2))
        breakout_slice = h1['h'].iloc[-(breakout_window + 1):-1] if len(h1) > breakout_window else h1['h'].iloc[:-1]
        breakout_level = float(breakout_slice.max()) if not breakout_slice.empty else price
        prev_close = float(h1_close.iloc[-2]) if len(h1_close) > 1 else price
        buffer_multiplier = 1.0 + breakout_buffer_pct / 100.0
        breakout_confirmed = prev_close <= breakout_level * buffer_multiplier and price > breakout_level * buffer_multiplier
        breakout_pct = ((price - breakout_level) / breakout_level * 100) if breakout_level > 0 else 0.0
        if breakout_confirmed:
            score += 28
            signals.append(f"1H突破近{breakout_window}根新高")

        recent_4h_high_slice = h4['h'].iloc[-31:-1] if len(h4) > 31 else h4['h'].iloc[:-1]
        daily_high_slice = d1['h'].iloc[-31:-1] if len(d1) > 31 else d1['h'].iloc[:-1]
        recent_4h_high = float(recent_4h_high_slice.max()) if not recent_4h_high_slice.empty else price
        daily_high = float(daily_high_slice.max()) if not daily_high_slice.empty else price
        if recent_4h_high > 0 and price > recent_4h_high * buffer_multiplier:
            score += 14
            signals.append("突破4H阶段高点")
        if daily_high > 0 and price > daily_high * (1.0 - 0.001):
            score += 10
            signals.append("接近日线阶段新高")

        tail_mean = float(h1_vol.tail(20).mean()) if not h1_vol.empty else 0.0
        volume_ratio = float(h1_vol.iloc[-1] / tail_mean) if tail_mean > 0 and not h1_vol.empty else 1.0
        if volume_ratio >= float(self.config.get('min_breakout_volume_ratio', 1.5)):
            score += 14
            signals.append(f"放量突破({volume_ratio:.2f}x)")

        if price > h1_ema20:
            score += 8
            signals.append("1H站稳短均线")

        if 58 <= h1_rsi <= float(self.config.get('max_breakout_rsi', 76)):
            score += 10
            signals.append(f"突破RSI健康({h1_rsi:.1f})")
        elif h1_rsi > float(self.config.get('max_breakout_rsi', 76)):
            score -= 6
            signals.append(f"突破后RSI偏热({h1_rsi:.1f})")

        extension_pct = abs((price - h4_ema20) / h4_ema20 * 100) if h4_ema20 > 0 else 999.0
        if extension_pct <= float(self.config.get('max_extension_pct', 5.0)):
            score += 8
        else:
            score -= 8
            signals.append(f"距离4H均线偏远({extension_pct:.2f}%)")

        if h4_atr_pct <= float(self.config.get('max_h4_atr_pct', 6.0)):
            score += 6
        else:
            score -= 5
            signals.append(f"4H波动过大({h4_atr_pct:.2f}%)")

        direction = 'BUY' if long_trend and breakout_confirmed and extension_pct <= float(self.config.get('max_extension_pct', 5.0)) else 'WAIT'

        trigger_quality = 95.0 if breakout_confirmed else 30.0
        trend_quality = trend_long_score
        volume_quality = min(volume_ratio / max(float(self.config.get('min_breakout_volume_ratio', 1.5)), 0.1), 1.6) * 62.5
        location_quality = max(20.0, 100.0 - max(extension_pct - 1.0, 0.0) * 17.0)
        freshness_quality = max(18.0, 100.0 - max(breakout_pct - 1.0, 0.0) * 22.0)

        return {
            'valid': True,
            'score': max(score, 0.0),
            'direction': direction,
            'signals': signals,
            'ranking_factors': {
                'trend': trend_quality,
                'trigger': trigger_quality,
                'volume': volume_quality,
                'location': location_quality,
                'freshness': freshness_quality,
                'risk': 88.0 if h4_atr_pct <= float(self.config.get('max_h4_atr_pct', 6.0)) else 56.0,
            },
            'details': {
                '评估': ' | '.join(signals) if signals else '暂无新高突破机会',
                '突破幅度': f'{breakout_pct:.2f}%',
                '量比': f'{volume_ratio:.2f}x',
                '1H_RSI': f'{h1_rsi:.1f}',
                '趋势质量': f'{trend_long_score:.1f}',
                '趋势诊断': str(trend_snapshot.get('reason', '')),
                'H4_ADX': f"{float(trend_metrics.get('h4_adx', 0.0)):.1f}",
                '趋势效率': f"{float(trend_metrics.get('h1_efficiency', 0.0)):.1f}",
                '延伸幅度': f'{extension_pct:.2f}%',
                '4H_ATR%': f'{h4_atr_pct:.2f}%',
            }
        }

    def _ema_slope_pct(self, close: pd.Series, span: int = 20, lookback: int = 5) -> float:
        ema = close.ewm(span=span, adjust=False).mean()
        if len(ema) <= lookback:
            return 0.0
        base = float(ema.iloc[-(lookback + 1)])
        latest = float(ema.iloc[-1])
        if base <= 0:
            return 0.0
        return (latest / base - 1.0) * 100

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
        atr_series = tr.rolling(period).mean()
        if atr_series.empty or pd.isna(atr_series.iloc[-1]) or close.empty or close.iloc[-1] <= 0:
            return 0.0
        return float((atr_series.iloc[-1] / close.iloc[-1]) * 100)

    def _rsi(self, close: pd.Series, period: int = 14) -> float:
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
        if len(loss) == 0 or len(gain) == 0 or pd.isna(loss.iloc[-1]):
            return 50.0
        if pd.isna(gain.iloc[-1]):
            return 50.0
        if loss.iloc[-1] == 0:
            return 100.0 if gain.iloc[-1] > 0 else 50.0
        rs = gain.iloc[-1] / loss.iloc[-1]
        return float(100 - (100 / (1 + rs)))

    def _get_klines(self, klines_map: Dict[str, List], bar: str) -> List:
        return klines_map.get(bar) or klines_map.get(bar.lower()) or klines_map.get(bar.upper()) or []



    def get_config_schema(self) -> Dict:
        return {
            'min_score': {'type': 'int', 'default': 84, 'label': '最低通过分数'},
            'min_volume_24h': {'type': 'float', 'default': 18000000, 'label': '最小24H成交额'},
            'breakout_window': {'type': 'int', 'default': 55, 'label': '突破观察窗口'},
            'breakout_buffer_pct': {'type': 'float', 'default': 0.2, 'label': '突破确认缓冲%'},
            'min_daily_slope_pct': {'type': 'float', 'default': 0.5, 'label': '最小日线斜率%'},
            'min_h4_slope_pct': {'type': 'float', 'default': 0.8, 'label': '最小4H斜率%'},
            'min_breakout_volume_ratio': {'type': 'float', 'default': 1.5, 'label': '突破最小量比'},
            'max_breakout_rsi': {'type': 'float', 'default': 76, 'label': '突破最大RSI'},
            'max_extension_pct': {'type': 'float', 'default': 5.0, 'label': '最大均线延伸%'},
            'max_h4_atr_pct': {'type': 'float', 'default': 6.0, 'label': '最大4H ATR%'},
            'min_watch_trend_quality': {'type': 'float', 'default': 58.0, 'label': '观察级趋势质量分'},
        }


STRATEGY_NAME  = "新高突破扫描"
STRATEGY_TYPE  = "scan"
STRATEGY_CLASS = NewHighBreakoutScanner
BACKTEST_CLASS = NewHighBreakoutScanner
