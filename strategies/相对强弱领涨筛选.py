"""
经典强弱策略：相对强弱领涨筛选

核心思想：
1. 不只看单个币是否上涨，而是看它是否明显强于全市场中位数。
2. 结合 1D / 4H / 1H 多周期结构，筛出抗跌、领涨、量能健康的强势标的。
3. 过滤已经过度延伸的追高标的，优先保留更适合继续跟踪的领涨机会。
"""

import logging
from statistics import median
from typing import Any, Dict, List

import pandas as pd

logger = logging.getLogger(__name__)

from src.scanner.base_scanner import BaseScannerStrategy, ScanCondition, ScannerSymbol
from src.scanner.ranking import build_opportunity_profile, sort_scan_results
from src.scanner.trend_quality import trend_quality_snapshot
from strategies._shared.indicators import _to_df


class RelativeStrengthLeaderScanner(BaseScannerStrategy):
    required_bars = ['1D', '4H', '1H']

    def _init_conditions(self):
        self.add_condition(ScanCondition(
            name="24H成交量",
            description="过滤流动性不足标的",
            field="volume_24h",
            operator=">=",
            value=self.config.get('min_volume_24h', 20000000),
        ))

    def scan_all_symbols(self, symbols: List[ScannerSymbol]) -> Dict:
        """批量扫描，用全市场中位数作为相对强弱基准。"""
        if not symbols:
            return {'type': 'relative_strength_scan', 'all_opportunities': []}

        market_changes = [
            float(item.price_change_24h or 0.0)
            for item in symbols
            if item.inst_id and item.inst_id.endswith('-USDT-SWAP')
        ]
        market_median_change = median(market_changes) if market_changes else 0.0
        market_top_cutoff = self._percentile(market_changes, 70) if market_changes else 0.0

        results = []
        for symbol in symbols:
            try:
                result = self.scan_symbol(symbol, market_median_change, market_top_cutoff)
                if result.get('passed'):
                    results.append(result)
            except Exception as exc:
                logger.error(f"[相对强弱领涨筛选] {symbol.inst_id} 分析失败: {exc}")

        results = sort_scan_results(results)
        top_n = int(self.config.get('top_n', 30))
        return {
            'type': 'relative_strength_scan',
            'all_opportunities': results[:top_n],
        }

    def scan_symbol(self, symbol: ScannerSymbol, market_median_change: float = 0.0, market_top_cutoff: float = 0.0) -> Dict:
        klines_map = symbol.extra_data.get('klines', {})
        try:
            analysis = self._analyze(
                self._get_klines(klines_map, '1D'),
                self._get_klines(klines_map, '4H'),
                self._get_klines(klines_map, '1H'),
                symbol.last_price,
                symbol.price_change_24h,
                market_median_change,
                market_top_cutoff,
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

        min_score = float(self.config.get('min_score', 82))
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
            'category': '相对强弱',
            'ranking_factors': analysis.get('ranking_factors', {}),
            **profile,
        }

    def _analyze(
        self,
        d1_klines: List,
        h4_klines: List,
        h1_klines: List,
        last_price: float,
        price_change_24h: float,
        market_median_change: float,
        market_top_cutoff: float,
    ) -> Dict[str, Any]:
        d1 = _to_df(d1_klines)
        h4 = _to_df(h4_klines)
        h1 = _to_df(h1_klines)
        if len(d1) < 80:
            return {'valid': False, 'reason': f'日线数据不足({len(d1)}/80)'}
        if len(h4) < 100:
            return {'valid': False, 'reason': f'4H数据不足({len(h4)}/100)'}
        if len(h1) < 120:
            return {'valid': False, 'reason': f'1H数据不足({len(h1)}/120)'}

        score = 0.0
        signals = []
        price = float(last_price if last_price > 0 else h1['c'].iloc[-1])

        h1_close = h1['c']
        h4_close = h4['c']
        d1_close = d1['c']

        d1_ema20 = float(d1_close.ewm(span=20, adjust=False).mean().iloc[-1])
        d1_ema50 = float(d1_close.ewm(span=50, adjust=False).mean().iloc[-1])
        h4_ema20 = float(h4_close.ewm(span=20, adjust=False).mean().iloc[-1])
        h4_ema50 = float(h4_close.ewm(span=50, adjust=False).mean().iloc[-1])
        h1_ema20 = float(h1_close.ewm(span=20, adjust=False).mean().iloc[-1])
        h1_ema50 = float(h1_close.ewm(span=50, adjust=False).mean().iloc[-1])

        rs_24h = float(price_change_24h or 0.0) - float(market_median_change or 0.0)
        if rs_24h >= float(self.config.get('min_relative_strength_24h', 3.0)):
            score += 24
            signals.append(f"24H强于市场中位数({rs_24h:.2f}%)")

        if float(price_change_24h or 0.0) >= float(market_top_cutoff or 0.0):
            score += 10
            signals.append("24H表现位于市场前列")

        h4_momentum = self._momentum_pct(h4_close, 18)
        h1_momentum = self._momentum_pct(h1_close, 24)
        if h4_momentum >= float(self.config.get('min_h4_momentum_pct', 4.0)):
            score += 16
            signals.append(f"4H动量强({h4_momentum:.2f}%)")
        if h1_momentum >= float(self.config.get('min_h1_momentum_pct', 2.5)):
            score += 12
            signals.append(f"1H继续领涨({h1_momentum:.2f}%)")

        trend_snapshot = trend_quality_snapshot(d1, h4, h1, price)
        trend_metrics = trend_snapshot.get('metrics', {})
        trend_long_score = float(trend_snapshot.get('long_score', 0.0))
        structure_ok = (
            bool(trend_snapshot.get('long_ok'))
            and price > d1_ema20 > d1_ema50
            and price > h4_ema20 > h4_ema50
            and price > h1_ema20 > h1_ema50
        )
        if structure_ok:
            trend_bonus = min(30.0, 18.0 + max(trend_long_score - 68.0, 0.0) * 0.35)
            score += trend_bonus
            signals.append(f"领涨趋势质量通过({trend_long_score:.0f})")
        elif trend_long_score >= float(self.config.get('min_watch_trend_quality', 58.0)):
            score += 6
            signals.append(f"趋势质量观察级({trend_long_score:.0f})")

        pullback_resilience = self._pullback_resilience(h4)
        if pullback_resilience >= float(self.config.get('min_resilience_score', 62.0)):
            score += 10
            signals.append(f"回撤韧性较强({pullback_resilience:.0f})")

        volume_ratio = self._volume_ratio(h1)
        if volume_ratio >= float(self.config.get('min_volume_ratio', 1.15)):
            score += 10
            signals.append(f"量能健康({volume_ratio:.2f}x)")

        h1_rsi = self._rsi(h1_close)
        if 55 <= h1_rsi <= float(self.config.get('max_rsi', 76.0)):
            score += 8
            signals.append(f"RSI强势但未过热({h1_rsi:.1f})")
        elif h1_rsi > float(self.config.get('max_rsi', 76.0)):
            score -= 8
            signals.append(f"RSI过热({h1_rsi:.1f})")

        extension_pct = abs((price - h4_ema20) / h4_ema20 * 100) if h4_ema20 > 0 else 999.0
        if extension_pct <= float(self.config.get('max_extension_pct', 6.0)):
            score += 8
        else:
            score -= 10
            signals.append(f"强势但延伸过大({extension_pct:.2f}%)")

        atr_pct = self._atr_pct(h4)
        if atr_pct <= float(self.config.get('max_h4_atr_pct', 7.0)):
            score += 6
        else:
            score -= 6
            signals.append(f"4H波动过大({atr_pct:.2f}%)")

        direction = 'BUY' if structure_ok and rs_24h >= float(self.config.get('min_relative_strength_24h', 3.0)) and extension_pct <= float(self.config.get('max_extension_pct', 6.0)) else 'WAIT'

        relative_quality = min(100.0, max(20.0, 55.0 + rs_24h * 6.0))
        trend_quality = trend_long_score
        trigger_quality = 86.0 if h1_momentum >= float(self.config.get('min_h1_momentum_pct', 2.5)) else 45.0
        volume_quality = min(volume_ratio / max(float(self.config.get('min_volume_ratio', 1.15)), 0.1), 1.6) * 62.5
        location_quality = max(20.0, 100.0 - max(extension_pct - 1.5, 0.0) * 14.0)

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
                'freshness': relative_quality,
                'risk': 88.0 if atr_pct <= float(self.config.get('max_h4_atr_pct', 7.0)) else 55.0,
            },
            'details': {
                '评估': ' | '.join(signals) if signals else '暂无相对强弱领涨机会',
                '相对市场强度': f'{rs_24h:.2f}%',
                '市场中位涨幅': f'{market_median_change:.2f}%',
                '4H动量': f'{h4_momentum:.2f}%',
                '1H动量': f'{h1_momentum:.2f}%',
                '趋势质量': f'{trend_long_score:.1f}',
                '趋势诊断': str(trend_snapshot.get('reason', '')),
                'H4_ADX': f"{float(trend_metrics.get('h4_adx', 0.0)):.1f}",
                '趋势效率': f"{float(trend_metrics.get('h1_efficiency', 0.0)):.1f}",
                '回撤韧性': f'{pullback_resilience:.0f}',
                '量比': f'{volume_ratio:.2f}x',
                'RSI': f'{h1_rsi:.1f}',
                '延伸幅度': f'{extension_pct:.2f}%',
                '4H_ATR%': f'{atr_pct:.2f}%',
            }
        }

    def _momentum_pct(self, close: pd.Series, lookback: int) -> float:
        if len(close) <= lookback:
            return 0.0
        base = float(close.iloc[-(lookback + 1)])
        latest = float(close.iloc[-1])
        return (latest / base - 1.0) * 100 if base > 0 else 0.0

    def _pullback_resilience(self, df: pd.DataFrame) -> float:
        if len(df) < 30:
            return 50.0
        recent = df.tail(24)
        high = float(recent['h'].max())
        low = float(recent['l'].min())
        close = float(recent['c'].iloc[-1])
        if high <= 0 or low <= 0 or high == low:
            return 50.0
        position_score = (close - low) / (high - low) * 100
        max_drop_pct = (high - low) / high * 100
        drop_penalty = max(0.0, max_drop_pct - 12.0) * 2.5
        return max(0.0, min(100.0, position_score - drop_penalty))

    def _volume_ratio(self, df: pd.DataFrame) -> float:
        if len(df) < 20 or df['vol'].empty:
            return 1.0
        avg = float(df['vol'].tail(20).mean())
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
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
        if len(loss) == 0 or len(gain) == 0 or pd.isna(loss.iloc[-1]) or pd.isna(gain.iloc[-1]):
            return 50.0
        if loss.iloc[-1] == 0:
            return 100.0 if gain.iloc[-1] > 0 else 50.0
        rs = gain.iloc[-1] / loss.iloc[-1]
        return float(100 - (100 / (1 + rs)))

    def _percentile(self, values: List[float], percentile: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(float(item) for item in values)
        idx = int(round((len(ordered) - 1) * percentile / 100.0))
        return ordered[max(0, min(idx, len(ordered) - 1))]

    def _get_klines(self, klines_map: Dict[str, List], bar: str) -> List:
        return klines_map.get(bar) or klines_map.get(bar.lower()) or klines_map.get(bar.upper()) or []



    def get_config_schema(self) -> Dict:
        return {
            'min_score': {'type': 'int', 'default': 82, 'label': '最低通过分数'},
            'top_n': {'type': 'int', 'default': 30, 'label': '最多输出数量'},
            'min_volume_24h': {'type': 'float', 'default': 20000000, 'label': '最小24H成交额'},
            'min_relative_strength_24h': {'type': 'float', 'default': 3.0, 'label': '最小相对市场强度%'},
            'min_h4_momentum_pct': {'type': 'float', 'default': 4.0, 'label': '最小4H动量%'},
            'min_h1_momentum_pct': {'type': 'float', 'default': 2.5, 'label': '最小1H动量%'},
            'min_resilience_score': {'type': 'float', 'default': 62.0, 'label': '最小回撤韧性分'},
            'min_volume_ratio': {'type': 'float', 'default': 1.15, 'label': '最小量比'},
            'max_rsi': {'type': 'float', 'default': 76.0, 'label': '最大RSI'},
            'max_extension_pct': {'type': 'float', 'default': 6.0, 'label': '最大4H均线延伸%'},
            'max_h4_atr_pct': {'type': 'float', 'default': 7.0, 'label': '最大4H ATR%'},
            'min_watch_trend_quality': {'type': 'float', 'default': 58.0, 'label': '观察级趋势质量分'},
        }


STRATEGY_NAME  = "相对强弱领涨筛选"
STRATEGY_TYPE  = "scan"
STRATEGY_CLASS = RelativeStrengthLeaderScanner
BACKTEST_CLASS = RelativeStrengthLeaderScanner
