"""
经典波动率策略：波动率收缩爆发扫描（精修版）

核心思想：
1. 先识别 4H / 1H 是否进入明显收敛和低波动阶段。
2. 等待 1H 突破压缩区并伴随量能放大。
3. 保留更像"收缩后刚选择方向"的机会，而不是已跑远的行情。

精修要点：
- 分数权重归一化为 100，移除两处负分惩罚（改为不加 bonus）。
- breakout_pct/extension bonus 仅在有突破时才可能得分，消除 WAIT
  状态虚拿 +8 的问题。
- 方向判定改用布尔 flag，并将量能要求内嵌到 direction 条件中；
  breakout 得分与 direction 判定解耦，消除"高分 WAIT"矛盾。
- 修复 prev_close 突破判定永久失效 bug（与其他策略一致）。
- ATR 改为 Wilder EMA 平滑（alpha=1/period），对近期波动更敏感。
- _atr_pct_mean 改为先按棒计算 ATR%，再取均值，消除 close 长度
  不匹配问题，且含义更准确。
- 布林带宽度直接用 4*std/ma，不再构造 upper/lower 再相减。
- compression_width_pct 改用中点归一化。
- extension_pct 改为相对压缩区中点衡量，与策略意图一致。
- volume_ratio 改用近 3 根峰值量 / 基线均量，避免突破后缩量漏信号。
- EMA span 统一为 21/55，与其他策略文件对齐。
- 工具函数提取为模块级，便于单元测试。
"""

from __future__ import annotations

import pandas as pd
from typing import Any, Dict, List, Tuple

from src.scanner.base_scanner import BaseScannerStrategy, ScanCondition, ScannerSymbol
from src.scanner.ranking import build_opportunity_profile
from strategies._shared.indicators import _check_df, _ema, _to_df

# ──────────────────────────────────────────────
# 各分段满分（合计 100）
# ──────────────────────────────────────────────
_W_ATR_COMPRESS  = 24  # 4H ATR 收缩
_W_BAND_COMPRESS = 20  # 1H 布林带收窄
_W_RANGE_WIDTH   = 16  # 1H 压缩区窄幅整理
_W_BREAKOUT      = 26  # 1H 方向爆发确认
_W_VOLUME        = 10  # 爆发量能 bonus
_W_FRESHNESS     =  4  # 突破初段 bonus（不扣分）
# 合计 = 24+20+16+26+10+4 = 100


class VolatilityCompressionBreakoutScanner(BaseScannerStrategy):
    required_bars = ['4H', '1H']

    def _init_conditions(self):
        self.add_condition(ScanCondition(
            name="24H成交量",
            description="过滤流动性不足标的",
            field="volume_24h",
            operator=">=",
            value=self.config.get('min_volume_24h', 15_000_000),
        ))

    # ──────────────────────────────────────────
    # 主入口
    # ──────────────────────────────────────────
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

        min_score = float(self.config.get('min_score', 70))
        profile = build_opportunity_profile(
            base_score=analysis['score'],
            direction=analysis['direction'],
            volume_24h=symbol.volume_24h,
            factors=analysis.get('ranking_factors', {}),
            signals=analysis['signals'],
        )
        return {
            'symbol': symbol.inst_id,
            'passed': (
                analysis['score'] >= min_score
                and analysis['direction'] in {'BUY', 'SELL'}
            ),
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

    # ──────────────────────────────────────────
    # 核心分析
    # ──────────────────────────────────────────
    def _analyze(self, h4_klines: List, h1_klines: List, last_price: float) -> Dict[str, Any]:
        h4 = _to_df(h4_klines)
        h1 = _to_df(h1_klines)

        _check_df(h4, '4H', 100)
        _check_df(h1, '1H', 140)

        score   = 0.0
        signals = []
        price   = float(last_price) if last_price > 0 else float(h1['c'].iloc[-1])

        # ── 均线（span 对齐为 21/55）──
        h4_ema21 = _ema(h4['c'], 21)
        h4_ema55 = _ema(h4['c'], 55)
        h1_ema21 = _ema(h1['c'], 21)

        # ── ATR 收缩：Wilder ATR%（当前 vs 基线均值）──
        atr_offset = int(self.config.get('atr_baseline_offset', 8))
        atr_window = int(self.config.get('atr_baseline_window', 20))
        h4_atr_now  = _atr_pct_current(h4, period=14)
        h4_atr_base = _atr_pct_baseline(h4, period=14, window=atr_window, offset=atr_offset)
        atr_compression_ratio = h4_atr_now / h4_atr_base if h4_atr_base > 0 else 1.0

        # ── 布林带收缩：当前带宽% vs 基线均值 ──
        bb_offset = int(self.config.get('bb_baseline_offset', 8))
        bb_window  = int(self.config.get('bb_baseline_window', 24))
        h1_band_now, h1_band_base = _bollinger_width_stats(
            h1['c'], period=20, window=bb_window, offset=bb_offset,
        )
        band_compression_ratio = h1_band_now / h1_band_base if h1_band_base > 0 else 1.0

        # ── 压缩区高低（排除当前棒，避免突破棒扩张参考区间）──
        range_window = int(self.config.get('compression_window', 24))
        ref_slice    = h1.tail(range_window + 1).iloc[:-1]  # 明确排除最后一棒
        zone_high    = float(ref_slice['h'].max()) if not ref_slice.empty else price
        zone_low     = float(ref_slice['l'].min()) if not ref_slice.empty else price
        zone_mid     = (zone_high + zone_low) / 2.0
        # 中点归一化宽度
        compression_width_pct = (
            (zone_high - zone_low) / zone_mid * 100
            if zone_mid > 0 else 999.0
        )

        # ── ① 4H ATR 收缩（满分 24）──
        max_atr_ratio = float(self.config.get('max_atr_compression_ratio', 0.82))
        if atr_compression_ratio <= max_atr_ratio:
            # 收缩越深得分越高
            atr_score = _W_ATR_COMPRESS * min(
                1.0, 0.55 + (max_atr_ratio - atr_compression_ratio) / max(max_atr_ratio - 0.3, 0.1) * 0.45
            )
            score += atr_score
            signals.append(f"4H ATR收缩({atr_compression_ratio:.2f} → +{atr_score:.0f}分)")
        else:
            signals.append(f"4H ATR未收缩({atr_compression_ratio:.2f}/{max_atr_ratio})")

        # ── ② 1H 布林带收窄（满分 20）──
        max_band_ratio = float(self.config.get('max_band_compression_ratio', 0.78))
        if band_compression_ratio <= max_band_ratio:
            band_score = _W_BAND_COMPRESS * min(
                1.0, 0.55 + (max_band_ratio - band_compression_ratio) / max(max_band_ratio - 0.3, 0.1) * 0.45
            )
            score += band_score
            signals.append(f"1H布林带收窄({band_compression_ratio:.2f} → +{band_score:.0f}分)")
        else:
            signals.append(f"1H布林带未收窄({band_compression_ratio:.2f}/{max_band_ratio})")

        # ── ③ 压缩区窄幅整理（满分 16）──
        max_cw = float(self.config.get('max_compression_width_pct', 5.5))
        if compression_width_pct <= max_cw:
            cw_score = _W_RANGE_WIDTH * min(
                1.0, 0.5 + (1.0 - compression_width_pct / max(max_cw, 0.1)) * 0.5
            )
            score += cw_score
            signals.append(f"1H压缩区窄幅整理({compression_width_pct:.2f}% → +{cw_score:.0f}分)")
        else:
            signals.append(f"压缩区过宽({compression_width_pct:.2f}%/{max_cw}%)")

        # ── 趋势背景 & 突破触发 ──
        bullish_context = price > h4_ema21 > h4_ema55
        bearish_context = price < h4_ema21 < h4_ema55

        buf           = float(self.config.get('breakout_buffer_pct', 0.18)) / 100.0
        upper_trigger = zone_high * (1.0 + buf)
        lower_trigger = zone_low  * (1.0 - buf)
        last_close    = float(h1['c'].iloc[-1])
        prev_close    = float(h1['c'].iloc[-2])

        # 修正：要求前一棒在扩展压缩区内，当前棒突破触发线
        # 不要求 prev <= trigger（那会在突破后每次都失效）
        prev_in_zone = (
            zone_low  * (1.0 - buf * 2) <= prev_close
            <= zone_high * (1.0 + buf * 2)
        )
        breakout_up   = bullish_context and prev_in_zone and last_close > upper_trigger and last_close > h1_ema21
        breakout_down = bearish_context and prev_in_zone and last_close < lower_trigger and last_close < h1_ema21

        # ── ④ 1H 方向爆发确认（满分 26）──
        if breakout_up:
            score += _W_BREAKOUT
            signals.append(f"1H向上爆发({last_close:.6g} > {upper_trigger:.6g})")
        elif breakout_down:
            score += _W_BREAKOUT
            signals.append(f"1H向下爆发({last_close:.6g} < {lower_trigger:.6g})")
        else:
            parts = []
            if not (bullish_context or bearish_context):
                parts.append("4H趋势背景不明")
            if not prev_in_zone:
                parts.append("前一棒不在压缩区")
            if last_close <= upper_trigger and last_close >= lower_trigger:
                parts.append("尚未触发突破")
            signals.append("未爆发: " + " | ".join(parts) if parts else "1H尚未爆发")

        # ── ⑤ 爆发量能 bonus（满分 10）──
        # 近 3 根峰值量 / 基线均量（排除近 range_window 根，即整理期）
        volume_ratio = _breakout_volume_ratio(
            h1['vol'], breakout_window=3, baseline_window=20,
        )
        min_vr = float(self.config.get('min_breakout_volume_ratio', 1.6))
        vol_ok = volume_ratio >= min_vr
        if vol_ok:
            vol_score = _W_VOLUME * min(1.0, 0.55 + (volume_ratio - min_vr) / max(min_vr, 0.1) * 0.3)
            score    += vol_score
            signals.append(f"爆发量能({volume_ratio:.2f}x → +{vol_score:.0f}分)")
        else:
            signals.append(f"量能不足({volume_ratio:.2f}x/{min_vr}x)")

        # ── ⑥ 突破初段 freshness bonus（满分 4，仅在有突破时生效）──
        breakout_pct = 0.0
        if breakout_up and zone_high > 0:
            breakout_pct = (last_close - zone_high) / zone_high * 100
        elif breakout_down and zone_low > 0:
            breakout_pct = (zone_low - last_close) / zone_low * 100

        max_bp = float(self.config.get('max_initial_breakout_pct', 3.0))
        if (breakout_up or breakout_down) and breakout_pct <= max_bp:
            fresh_score = _W_FRESHNESS * max(0.0, 1.0 - breakout_pct / max(max_bp, 0.1))
            score      += fresh_score
            if fresh_score > 0:
                signals.append(f"突破初段({breakout_pct:.2f}% → +{fresh_score:.0f}分)")
        elif breakout_pct > max_bp:
            signals.append(f"爆发后已跑远({breakout_pct:.2f}%，上限{max_bp}%)")

        # ── extension_pct：相对压缩区中点（而非 EMA21）──
        extension_pct = abs(last_close - zone_mid) / zone_mid * 100 if zone_mid > 0 else 0.0

        # ── 方向判定（布尔 flag；量能要求内嵌）──
        direction = 'WAIT'
        if breakout_up and vol_ok:
            direction = 'BUY'
        elif breakout_down and vol_ok:
            direction = 'SELL'

        return _build_result(
            valid=True, score=score, direction=direction, signals=signals,
            atr_compression_ratio=atr_compression_ratio,
            band_compression_ratio=band_compression_ratio,
            compression_width_pct=compression_width_pct,
            volume_ratio=volume_ratio, breakout_pct=breakout_pct,
            extension_pct=extension_pct,
            bullish_context=bullish_context, bearish_context=bearish_context,
        )

    # ──────────────────────────────────────────
    # 工具方法
    # ──────────────────────────────────────────
    def _get_klines(self, klines_map: Dict[str, List], bar: str) -> List:
        return (
            klines_map.get(bar)
            or klines_map.get(bar.lower())
            or klines_map.get(bar.upper())
            or []
        )



    def get_config_schema(self) -> Dict:
        return {
            'min_score':                  {'type': 'int',   'default': 70,         'label': '最低通过分数(0-100)'},
            'min_volume_24h':             {'type': 'float', 'default': 15_000_000, 'label': '最小24H成交额'},
            'compression_window':         {'type': 'int',   'default': 24,         'label': '压缩区观察窗口(1H根数)'},
            'atr_baseline_offset':        {'type': 'int',   'default': 8,          'label': 'ATR基线偏移量(排除近N根)'},
            'atr_baseline_window':        {'type': 'int',   'default': 20,         'label': 'ATR基线计算窗口'},
            'bb_baseline_offset':         {'type': 'int',   'default': 8,          'label': '布林带基线偏移量'},
            'bb_baseline_window':         {'type': 'int',   'default': 24,         'label': '布林带基线计算窗口'},
            'max_atr_compression_ratio':  {'type': 'float', 'default': 0.82,       'label': 'ATR收缩系数上限'},
            'max_band_compression_ratio': {'type': 'float', 'default': 0.78,       'label': '带宽收缩系数上限'},
            'max_compression_width_pct':  {'type': 'float', 'default': 5.5,        'label': '压缩区最大宽度%(中点归一化)'},
            'breakout_buffer_pct':        {'type': 'float', 'default': 0.18,       'label': '突破确认缓冲%'},
            'min_breakout_volume_ratio':  {'type': 'float', 'default': 1.6,        'label': '突破最小量比'},
            'max_initial_breakout_pct':   {'type': 'float', 'default': 3.0,        'label': '最大初段爆发幅度%'},
        }


# ══════════════════════════════════════════════
# 模块级工具函数（无 self 依赖，便于单元测试）
# ══════════════════════════════════════════════

def _atr_series_wilder(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Wilder ATR（EMA alpha=1/period）。
    比 rolling().mean() 对近期波动更敏感，更适合收缩检测。
    """
    if len(df) < period + 1:
        return pd.Series(dtype=float)
    prev_c = df['c'].shift(1)
    tr = pd.concat([
        df['h'] - df['l'],
        (df['h'] - prev_c).abs(),
        (df['l'] - prev_c).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def _atr_pct_current(df: pd.DataFrame, period: int = 14) -> float:
    """
    最新 ATR 占收盘价的百分比。
    逐棒计算 ATR%，含义比"ATR均值 / close均值"更准确。
    """
    atr = _atr_series_wilder(df, period)
    if atr.empty or df['c'].empty:
        return 0.0
    last_atr   = float(atr.iloc[-1])
    last_close = float(df['c'].iloc[-1])
    return last_atr / last_close * 100 if last_close > 0 else 0.0


def _atr_pct_baseline(
    df: pd.DataFrame,
    period: int = 14,
    window: int = 20,
    offset: int = 8,
) -> float:
    """
    基线 ATR%：取 [-(window+offset) : -offset] 区间的逐棒 ATR% 均值。
    先算每棒 ATR%，再取均值，避免 close 长度不对齐问题。
    """
    atr = _atr_series_wilder(df, period)
    if atr.empty or len(df) < period + window + offset:
        return 0.0
    # 对应 close 也按同样索引切片
    atr_slice   = atr.iloc[-(window + offset):-offset]
    close_slice = df['c'].iloc[-(window + offset):-offset]
    valid_mask  = (close_slice > 0) & atr_slice.notna()
    if valid_mask.sum() == 0:
        return 0.0
    pct_series = (atr_slice[valid_mask] / close_slice[valid_mask] * 100)
    return float(pct_series.mean())


def _bollinger_width_stats(
    close: pd.Series,
    period: int = 20,
    window: int = 24,
    offset: int = 8,
) -> Tuple[float, float]:
    """
    返回 (当前带宽%, 基线带宽%) 。
    带宽 = 4*std / ma（即 (upper-lower)/ma），ddof=1 与 TradingView 一致。
    """
    if len(close) < period + window + offset:
        return 0.0, 0.0
    ma  = close.rolling(period).mean()
    std = close.rolling(period).std(ddof=1)
    # 避免除以零
    width_pct = (4.0 * std / ma.replace(0.0, float('nan')) * 100).fillna(0.0)

    current = float(width_pct.iloc[-1]) if not pd.isna(width_pct.iloc[-1]) else 0.0
    base_slice = width_pct.iloc[-(window + offset):-offset].dropna()
    base = float(base_slice.mean()) if not base_slice.empty else 0.0
    return current, base


def _breakout_volume_ratio(
    vol: pd.Series,
    breakout_window: int = 3,
    baseline_window: int = 20,
) -> float:
    """
    近 breakout_window 根峰值量 / 近 baseline_window 根基线均量。
    基线窗口从 breakout_window 之前开始，不与突破窗口重叠。
    """
    total_needed = breakout_window + baseline_window
    if len(vol) < total_needed:
        return 1.0
    baseline_slice = vol.iloc[-(total_needed):-breakout_window]
    baseline_mean  = float(baseline_slice.mean())
    if baseline_mean <= 0:
        return 1.0
    peak_vol = float(vol.tail(breakout_window).max())
    return peak_vol / baseline_mean


def _build_result(
    *,
    valid: bool,
    score: float,
    direction: str,
    signals: List[str],
    atr_compression_ratio: float,
    band_compression_ratio: float,
    compression_width_pct: float,
    volume_ratio: float,
    breakout_pct: float,
    extension_pct: float,
    bullish_context: bool,
    bearish_context: bool,
    reason: str = '',
) -> Dict[str, Any]:
    """统一构造分析返回字典。"""
    compression_quality = max(25.0, 100.0 - max(compression_width_pct - 1.8, 0.0) * 12.0)
    freshness_quality   = max(22.0, 100.0 - max(breakout_pct - 0.8, 0.0) * 22.0)
    volume_quality      = min(volume_ratio / 1.6, 1.7) * 58.0
    trend_quality       = 84.0 if (bullish_context or bearish_context) else 50.0
    return {
        'valid':     valid,
        'reason':    reason,
        'score':     max(score, 0.0),
        'direction': direction,
        'signals':   signals,
        'ranking_factors': {
            'trend':     trend_quality,
            'trigger':   93.0 if direction in {'BUY', 'SELL'} else 28.0,
            'volume':    volume_quality,
            'location':  (compression_quality + max(20.0, 100.0 - extension_pct * 10.0)) / 2.0,
            'freshness': freshness_quality,
            'risk':      90.0 if atr_compression_ratio <= 0.82 else 58.0,
        },
        'details': {
            '评估':       ' | '.join(signals) if signals else '暂无波动率收缩爆发机会',
            'ATR收缩系数': f'{atr_compression_ratio:.2f}',
            '带宽收缩系数': f'{band_compression_ratio:.2f}',
            '压缩区宽度':  f'{compression_width_pct:.2f}%',
            '量比':        f'{volume_ratio:.2f}x',
            '突破幅度':    f'{breakout_pct:.2f}%',
            '延伸幅度':    f'{extension_pct:.2f}%',
        },
    }


STRATEGY_NAME  = "波动率收缩爆发扫描"
STRATEGY_TYPE  = "scan"
STRATEGY_CLASS = VolatilityCompressionBreakoutScanner
BACKTEST_CLASS = VolatilityCompressionBreakoutScanner
