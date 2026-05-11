"""
波段八策略组合扫描器 — 独立自包含版本

本文件将以下所有依赖内联，可不依赖任何外部子策略文件独立运行：
  1. strategies/_shared/indicators.py  — 共享技术指标
  2. 波段平台突破扫描_4.21_v2.py       — BreakoutSwingScanner
  3. 新高突破扫描.py                    — NewHighBreakoutScanner
  4. 单边趋势跟随扫描_4.21_v2.py        — DirectionalTrendFollowScanner
  5. 背离反转扫描_4.21_v2.py            — DivergenceReversalScanner
  6. 波段趋势回踩扫描_4.21_v3.py        — TrendPullbackSwingScanner
  7. 波段超跌反转扫描_4.21_v3.py        — OversoldReversalSwingScanner
  8. 波段缩量中继再启动扫描_4.21_v2.py  — ContinuationCompressionSwingScanner
  9. 趋势回踩二次启动筛选_4.21_v6.py    — TrendPullbackRestartScanner
  10. 量价背离扫描策略.py               — VolumePriceDivergenceScanner

加载方式：
  STRATEGY_CLASS = SwingEightStrategyComboScanner
  BACKTEST_CLASS = SwingEightStrategyComboScanner
"""

from __future__ import annotations

import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# 可选：扫描引擎基类（缺失时降级为 object）
# ══════════════════════════════════════════════════════════════════════════════

try:
    from src.scanner.base_scanner import BaseScannerStrategy, ScanCondition, ScannerSymbol
    from src.scanner.ranking import build_opportunity_profile
    _HAS_SCANNER_BASE = True
except ImportError:
    BaseScannerStrategy = object
    ScanCondition = None
    ScannerSymbol = None
    build_opportunity_profile = None
    _HAS_SCANNER_BASE = False

try:
    from src.scanner.strategy_lifecycle import apply_strategy_lifecycle_guard
    _HAS_LIFECYCLE = True
except ImportError:
    apply_strategy_lifecycle_guard = None
    _HAS_LIFECYCLE = False

# 子策略基类别名（独立运行时降级为 object）
_BASE_SCANNER_CLASS = BaseScannerStrategy if _HAS_SCANNER_BASE else object

# ══════════════════════════════════════════════════════════════════════════════
# 第一节：内联共享指标库（来自 strategies/_shared/indicators.py）
# ══════════════════════════════════════════════════════════════════════════════

def _check_df(df: pd.DataFrame, label: str, min_len: int) -> None:
    if len(df) < min_len:
        raise ValueError(f"{label}数据不足({len(df)}/{min_len})")


def _to_df(klines) -> pd.DataFrame:
    if not klines:
        return pd.DataFrame(columns=['ts', 'o', 'h', 'l', 'c', 'vol'])
    if isinstance(klines, pd.DataFrame):
        return klines
    valid = [r[:6] for r in klines if isinstance(r, (list, tuple)) and len(r) >= 6]
    if not valid:
        return pd.DataFrame(columns=['ts', 'o', 'h', 'l', 'c', 'vol'])
    df = pd.DataFrame(valid, columns=['ts', 'o', 'h', 'l', 'c', 'vol'])
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['ts', 'o', 'h', 'l', 'c'])
    df['vol'] = df['vol'].fillna(0.0)
    return df.sort_values('ts').drop_duplicates('ts', keep='last').reset_index(drop=True)


def _aggregate_bars(df: pd.DataFrame, gs: int) -> pd.DataFrame:
    usable = len(df) // gs * gs
    if usable <= 0:
        return pd.DataFrame(columns=["ts", "o", "h", "l", "c", "vol"])
    tail = df.tail(usable).reset_index(drop=True)
    g = tail.groupby(tail.index // gs)
    return pd.DataFrame({
        "ts": g["ts"].last(), "o": g["o"].first(),
        "h": g["h"].max(), "l": g["l"].min(),
        "c": g["c"].last(), "vol": g["vol"].sum(),
    }).reset_index(drop=True)


def _safe_float(value, default: float = 0.0) -> float:
    try:
        n = float(value)
    except (TypeError, ValueError):
        return default
    return n if np.isfinite(n) else default


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def _ema(s: pd.Series, span: int) -> float:
    return float(s.ewm(span=span, adjust=False).mean().iloc[-1])


def _efficiency_ratio(c: pd.Series, window: int = 20) -> float:
    if len(c) < window + 1:
        return 0.0
    wc = c.iloc[-(window + 1):]
    n_ = abs(float(wc.iloc[-1]) - float(wc.iloc[0]))
    p_ = float(wc.diff().abs().sum())
    return float(min(1.0, n_ / p_)) if p_ > 0 else 0.0


def _rsi_wilder(c: pd.Series, period: int = 14) -> float:
    if len(c) < period + 1:
        return 50.0
    d = c.diff().dropna()
    g = d.clip(lower=0)
    lo = (-d).clip(lower=0)
    ag = g.ewm(alpha=1 / period, adjust=False).mean().iloc[-1]
    al = lo.ewm(alpha=1 / period, adjust=False).mean().iloc[-1]
    if al == 0:
        return 100.0 if ag > 0 else 50.0
    return float(100 - 100 / (1 + ag / al))


def _macd(close: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
    e12 = close.ewm(span=12, adjust=False).mean()
    e26 = close.ewm(span=26, adjust=False).mean()
    line = e12 - e26
    sig = line.ewm(span=9, adjust=False).mean()
    return line, sig, (line - sig)


def _atr(df: pd.DataFrame, period: int = 14) -> float:
    if len(df) < period + 1:
        return float(df['h'].iloc[-1] - df['l'].iloc[-1]) or 1.0
    pc = df['c'].shift(1)
    tr = pd.concat([df['h'] - df['l'], (df['h'] - pc).abs(), (df['l'] - pc).abs()], axis=1).max(axis=1)
    return float(tr.ewm(alpha=1 / period, adjust=False).mean().iloc[-1]) or 1.0


def _adx(df: pd.DataFrame, period: int = 14) -> float:
    if len(df) < period * 2 + 1:
        return 0.0
    h, l, c = df['h'], df['l'], df['c']
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    um = h.diff()
    dm = -l.diff()
    pdm = ((um > dm) & (um > 0)).astype(float) * um.clip(lower=0)
    mdm = ((dm > um) & (dm > 0)).astype(float) * dm.clip(lower=0)
    atr_s = tr.ewm(alpha=1 / period, adjust=False).mean()
    pdi = 100 * pdm.ewm(alpha=1 / period, adjust=False).mean() / atr_s.replace(0, np.nan)
    mdi = 100 * mdm.ewm(alpha=1 / period, adjust=False).mean() / atr_s.replace(0, np.nan)
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    a = dx.ewm(alpha=1 / period, adjust=False).mean()
    v = a.iloc[-1]
    return float(v) if pd.notna(v) else 0.0


def _volume_ratio_adjusted(df: pd.DataFrame, window: int = 20) -> float:
    vol = df['vol']
    if len(vol) < window + 1:
        return 1.0
    baseline = float(vol.iloc[-(window + 1):-1].mean())
    if baseline <= 0:
        return 1.0
    cv = float(vol.iloc[-1])
    if len(df) >= 3:
        try:
            tl = float(df['ts'].iloc[-1])
            tp = float(df['ts'].iloc[-2])
            bi = tl - tp
            now_ms = pd.Timestamp.utcnow().value // 1_000_000
            tl_ms = tl * 1000 if tl < 1e12 else tl
            bi_ms = bi * 1000 if tl < 1e12 else bi
            el = now_ms - tl_ms
            if 0 < el < bi_ms:
                pr = el / bi_ms
                if pr >= 0.1:
                    cv = cv / pr
        except Exception:
            pass
    return float(cv / baseline)


def _volume_zscore(vol: pd.Series, window: int = 20) -> float:
    if len(vol) < window + 1:
        return 0.0
    bl = vol.iloc[-(window + 1):-1]
    bl = bl[bl > 0]
    if len(bl) < 5:
        return 0.0
    lb = np.log(bl.values)
    m = float(lb.mean())
    s = float(lb.std(ddof=0))
    if s == 0:
        return 0.0
    cur = float(vol.iloc[-1])
    if cur <= 0:
        return -3.0
    return float((np.log(cur) - m) / s)


def _local_trend_snapshot(d1: pd.DataFrame, h4: pd.DataFrame, h1: pd.DataFrame, price: float) -> Dict[str, Any]:
    d1e21 = _ema(d1['c'], 21); d1e55 = _ema(d1['c'], 55)
    h4e21 = _ema(h4['c'], 21); h4e55 = _ema(h4['c'], 55)
    h1e21 = _ema(h1['c'], 21); h1e55 = _ema(h1['c'], 55)
    pt = 40.0 / 3.0
    la = 0.0; sa = 0.0
    for p, e21, e55 in [(price, d1e21, d1e55), (price, h4e21, h4e55), (price, h1e21, h1e55)]:
        if p > e21 > e55: la += pt
        if p < e21 < e55: sa += pt
    h4adx = _adx(h4, 14)
    adxs = min(30.0, max(0.0, (h4adx - 15.0) / 25.0 * 30.0))
    h1er = _efficiency_ratio(h1['c'], 20)
    ers = min(20.0, h1er * 20.0)
    d1sp = abs(d1e21 - d1e55) / d1e55 * 100 if d1e55 > 0 else 0.0
    h4sp = abs(h4e21 - h4e55) / h4e55 * 100 if h4e55 > 0 else 0.0
    sps = min(10.0, (d1sp + h4sp) / 4.0 * 10.0)
    ls = la + adxs + ers + sps; ss = sa + adxs + ers + sps
    lok = ls >= 55.0 and h4adx >= 15.0; sok = ss >= 55.0 and h4adx >= 15.0
    rp = []
    if h4adx < 15.0: rp.append(f"H4_ADX弱({h4adx:.1f})")
    if la < pt * 2 and sa < pt * 2: rp.append("均线排列不统一")
    r = " | ".join(rp) if rp else "趋势结构健康"
    return {
        'long_ok': bool(lok), 'short_ok': bool(sok),
        'long_score': float(ls), 'short_score': float(ss), 'reason': r,
        'metrics': {
            'h4_adx': float(h4adx), 'h1_efficiency': float(h1er * 100),
            'd1_spread_pct': float(d1sp), 'h4_spread_pct': float(h4sp), 'reason': r,
        },
    }


def _latest_swing_levels(df: pd.DataFrame, left: int = 5, right: int = 5,
                         skip_recent: int = 3, max_lookback: int = 80) -> Tuple[Optional[float], Optional[float]]:
    if len(df) < left + right + skip_recent + 5:
        return None, None
    h = df['h'].values; l = df['l'].values
    end = len(df) - skip_recent; start = max(left, end - max_lookback)
    sh = None; sl_ = None
    for i in range(end - right - 1, start - 1, -1):
        if sh is None and h[i] == max(h[i - left:i + right + 1]): sh = float(h[i])
        if sl_ is None and l[i] == min(l[i - left:i + right + 1]): sl_ = float(l[i])
        if sh is not None and sl_ is not None: break
    return sh, sl_


def _calc_atr(high: pd.Series, low: pd.Series, close: pd.Series,
              span: int = 5, seg_start: int = 0, seg_end: int = -1) -> float:
    try:
        if seg_end <= seg_start: return 0.0
        pc = close.shift(1)
        tr = pd.concat([(high - low).abs(), (high - pc).abs(), (low - pc).abs()], axis=1).max(axis=1)
        seg_tr = tr.iloc[seg_start:seg_end]
        if len(seg_tr) == 0: return 0.0
        return float(seg_tr.ewm(alpha=1 / span, adjust=False).mean().iloc[-1] or 0.0)
    except Exception:
        return 0.0


def _calc_volume_delta(close: pd.Series, high: pd.Series, low: pd.Series, vol: pd.Series,
                       seg_start: int = 0, seg_end: int = -1) -> Tuple[float, float]:
    try:
        seg_c = close.iloc[seg_start:seg_end]; seg_h = high.iloc[seg_start:seg_end]
        seg_l = low.iloc[seg_start:seg_end]
        seg_v = vol.iloc[seg_start:seg_end] if len(vol) > seg_start else pd.Series(np.ones(len(seg_c)))
        if len(seg_c) == 0: return 0.0, 0.0
        spread = (seg_h.values - seg_l.values); spread = np.where(spread > 0, spread, 1.0)
        buy_ratio = np.clip((seg_c.values - seg_l.values) / spread, 0.0, 1.0)
        return float(np.sum(buy_ratio * seg_v.values)), float(np.sum((1.0 - buy_ratio) * seg_v.values))
    except Exception:
        return 0.0, 0.0


def _calc_vwap(close: pd.Series, high: pd.Series, low: pd.Series, vol: pd.Series,
               seg_start: int = 0, seg_end: int = -1) -> float:
    try:
        seg_c = close.iloc[seg_start:seg_end]; seg_h = high.iloc[seg_start:seg_end]
        seg_l = low.iloc[seg_start:seg_end]
        seg_v = vol.iloc[seg_start:seg_end] if len(vol) > seg_start else pd.Series(np.ones(len(seg_c)))
        if len(seg_c) == 0: return 0.0
        typical = (seg_h.values + seg_l.values + seg_c.values) / 3.0
        total_vol = float(np.sum(seg_v.values))
        if total_vol <= 0: return float(seg_c.iloc[-1])
        return float(np.sum(typical * seg_v.values) / total_vol)
    except Exception:
        return 0.0


# ══════════════════════════════════════════════════════════════════════════════
# 第二节：子策略 1 — 波段平台突破扫描（BreakoutSwingScanner）
# ══════════════════════════════════════════════════════════════════════════════

_BSS_W_PLATFORM   = 24
_BSS_W_CONTEXT    = 16
_BSS_W_BREAKOUT   = 20
_BSS_W_VOLUME     = 12
_BSS_W_MACD       =  8
_BSS_W_RSI        =  6
_BSS_W_EXTENSION  =  8
_BSS_W_FRESHNESS  =  6
_BSS_ADX_MIN      = 16.0

_BSS_DEFAULT_CONFIG = {
    'min_score': 68, 'min_volume_24h': 15_000_000,
    'platform_bars': 18, 'max_platform_width_pct': 9.0,
    'max_platform_width_atr': 3.5, 'min_inside_ratio': 0.60,
    'breakout_buffer_pct': 0.25, 'min_breakout_volume_ratio': 1.8,
    'max_extension_atr': 2.5, 'trim_pct': 0.10,
}


def _bss_inside_ratio(close, low, high):
    if close.empty or high <= low: return 0.0
    return float(close.between(low, high).mean())


def _bss_trimmed_platform(h4, bars, trim_pct):
    highs = h4['h'].tail(bars).values; lows = h4['l'].tail(bars).values
    n = len(highs); trim_count = max(1, int(n * trim_pct))
    sorted_h = np.sort(highs); sorted_l = np.sort(lows)
    p_high = float(sorted_h[-(trim_count + 1)]) if trim_count < n else float(sorted_h[-1])
    p_low = float(sorted_l[trim_count]) if trim_count < n else float(sorted_l[0])
    return p_high, p_low


def _bss_bb_bandwidth(close, period=20):
    if len(close) < period: return 10.0
    mid = close.rolling(period).mean().iloc[-1]
    std = close.rolling(period).std(ddof=1).iloc[-1]
    if not pd.notna(std) or mid <= 0: return 10.0
    return float(4 * std / mid * 100)


def _bss_breakout_volume_ratio_v2(vol, breakout_window=3, baseline_window=20):
    if len(vol) < baseline_window + breakout_window + 1: return 1.0
    baseline = float(vol.iloc[-(baseline_window + breakout_window):-breakout_window].mean())
    if baseline <= 0: return 1.0
    peak = float(vol.tail(breakout_window).max())
    return peak / baseline


def _bss_build_result(*, valid, score, direction, signals, range_width_pct, range_width_atr,
                      volume_ratio, vol_zscore, extension_atr, h4_adx, h4_ema_spread,
                      h1_rsi, bb_bandwidth, bullish_ctx, bearish_ctx, reason=''):
    pq = 100.0 if range_width_pct <= 4.5 else max(35.0, 100 - range_width_pct * 7)
    fq = max(20.0, 100 - max(extension_atr - 0.5, 0) * 25)
    return {
        'valid': valid, 'reason': reason, 'score': max(score, 0.0),
        'direction': direction, 'signals': signals,
        'ranking_factors': {
            'trend': 88.0 if (bullish_ctx or bearish_ctx) and h4_adx >= _BSS_ADX_MIN else 40.0,
            'trigger': 92.0 if direction in {'BUY', 'SELL'} else 25.0,
            'volume': min(volume_ratio / 1.8, 1.6) * 62.5,
            'location': (pq + fq) / 2.0, 'freshness': fq, 'risk': pq,
        },
        'details': {
            '评估': ' | '.join(signals) if signals else '暂无平台突破',
            '平台宽度%': f'{range_width_pct:.2f}%', '平台宽度ATR': f'{range_width_atr:.1f}',
            'BB带宽': f'{bb_bandwidth:.1f}%', '4H_ADX': f'{h4_adx:.1f}',
            '4H_EMA发散': f'{h4_ema_spread:+.2f}%', '量比': f'{volume_ratio:.2f}x',
            '量能Z分': f'{vol_zscore:+.2f}σ', '1H_RSI': f'{h1_rsi:.1f}', '延伸ATR': f'{extension_atr:.1f}',
        },
    }


def _bss_analyze_core(h4, h1, last_price, cfg):
    _check_df(h4, '4H', 80); _check_df(h1, '1H', 120)
    score = 0.0; signals = []
    last_close = float(last_price) if last_price and last_price > 0 else float(h1['c'].iloc[-1])
    prev_close = float(h1['c'].iloc[-2])
    ema21_4h = _ema(h4['c'], 21); ema55_4h = _ema(h4['c'], 55)
    h4_adx = _adx(h4, 14)
    h4_ema_spread = (ema21_4h - ema55_4h) / ema55_4h * 100 if ema55_4h > 0 else 0.0
    bullish_ctx = ema21_4h > ema55_4h and last_close > ema21_4h
    bearish_ctx = ema21_4h < ema55_4h and last_close < ema21_4h
    h4_atr = _atr(h4); h1_atr = _atr(h1)
    h1_rsi = _rsi_wilder(h1['c'])
    vol_ratio = _bss_breakout_volume_ratio_v2(h1['vol'])
    vol_zscore = _volume_zscore(h1['vol'])
    _, _, h1_macd_hist = _macd(h1['c'])
    pb = int(cfg.get('platform_bars', 18)); trim = float(cfg.get('trim_pct', 0.10))
    p_high, p_low = _bss_trimmed_platform(h4, pb, trim)
    midpoint = (p_high + p_low) / 2.0
    range_width_pct = (p_high - p_low) / midpoint * 100 if midpoint > 0 else 999.0
    range_width_atr = (p_high - p_low) / h4_atr if h4_atr > 0 else 999.0
    inside_ratio = _bss_inside_ratio(h4['c'].tail(pb), p_low, p_high)
    bb_bandwidth = _bss_bb_bandwidth(h4['c'], 20)
    max_w_pct = float(cfg.get('max_platform_width_pct', 9.0))
    max_w_atr = float(cfg.get('max_platform_width_atr', 3.5))
    min_inside = float(cfg.get('min_inside_ratio', 0.60))
    platform_tight = range_width_pct <= max_w_pct and range_width_atr <= max_w_atr
    platform_mature = inside_ratio >= min_inside
    if platform_tight and platform_mature:
        tightness = max(0.0, 1.0 - range_width_pct / max_w_pct) * 0.4
        maturity = min(1.0, (inside_ratio - min_inside) / max(1.0 - min_inside, 0.01)) * 0.3
        bb_bonus = min(0.3, max(0.0, (6.0 - bb_bandwidth) / 6.0 * 0.3))
        ps = _BSS_W_PLATFORM * min(1.0, 0.45 + tightness + maturity + bb_bonus)
        score += ps
        signals.append(f"平台收敛({range_width_pct:.1f}%/{range_width_atr:.1f}ATR, 充分{inside_ratio:.0%}, BB{bb_bandwidth:.1f}% → +{ps:.0f}分)")
    elif platform_tight:
        ps = _BSS_W_PLATFORM * 0.4; score += ps
        signals.append(f"平台窄但整理不足({inside_ratio:.0%} → +{ps:.0f}分)")
    else:
        signals.append(f"平台过宽({range_width_pct:.1f}%/{range_width_atr:.1f}ATR)")
        return _bss_build_result(valid=True, score=score, direction='WAIT', signals=signals,
            range_width_pct=range_width_pct, range_width_atr=range_width_atr,
            volume_ratio=0.0, vol_zscore=0.0, extension_atr=0.0,
            h4_adx=h4_adx, h4_ema_spread=h4_ema_spread, h1_rsi=h1_rsi,
            bb_bandwidth=bb_bandwidth, bullish_ctx=bullish_ctx, bearish_ctx=bearish_ctx)
    if (bullish_ctx or bearish_ctx) and h4_adx >= _BSS_ADX_MIN:
        base_ctx = 10.0; adx_bonus = min(3.0, max(0.0, (h4_adx - _BSS_ADX_MIN) / 10.0 * 3.0))
        spread_bonus = min(3.0, abs(h4_ema_spread) / 2.0 * 3.0)
        ctx_s = base_ctx + adx_bonus + spread_bonus; score += ctx_s
        dl = "多头" if bullish_ctx else "空头"
        signals.append(f"4H{dl}背景(ADX {h4_adx:.1f}, 发散{h4_ema_spread:+.1f}% → +{ctx_s:.0f}分)")
    elif bullish_ctx or bearish_ctx:
        score += 6.0; signals.append(f"4H趋势背景弱(ADX {h4_adx:.1f})")
    else:
        signals.append("4H趋势背景不明确")
    buf = float(cfg.get('breakout_buffer_pct', 0.25)) / 100.0
    pb_1h = pb * 4
    bk_high = float(h1['h'].tail(pb_1h).max()); bk_low = float(h1['l'].tail(pb_1h).min())
    trigger_up = bk_high * (1.0 + buf); trigger_down = bk_low * (1.0 - buf)
    prev_in_range = p_low * 0.995 <= prev_close <= p_high * 1.005
    breakout_up = bullish_ctx and prev_in_range and last_close > trigger_up
    breakout_down = bearish_ctx and prev_in_range and last_close < trigger_down
    last_o = float(h1['o'].iloc[-1]); last_h = float(h1['h'].iloc[-1]); last_l = float(h1['l'].iloc[-1])
    bar_range = last_h - last_l
    close_strength = abs(last_close - last_o) / bar_range if bar_range > 0 else 0.0
    if breakout_up: breakout_magnitude = (last_close - trigger_up) / h1_atr if h1_atr > 0 else 0.0
    elif breakout_down: breakout_magnitude = (trigger_down - last_close) / h1_atr if h1_atr > 0 else 0.0
    else: breakout_magnitude = 0.0
    if (breakout_up or breakout_down) and breakout_magnitude >= 0.2:
        base_bk = _BSS_W_BREAKOUT * 0.6
        strength_bonus = _BSS_W_BREAKOUT * 0.25 * min(1.0, close_strength / 0.7)
        magnitude_bonus = _BSS_W_BREAKOUT * 0.15 * min(1.0, breakout_magnitude / 1.0)
        bks = base_bk + strength_bonus + magnitude_bonus; score += bks
        arrow = "向上突破" if breakout_up else "向下跌破"
        signals.append(f"1H{arrow}(强度{close_strength:.2f}, 幅度{breakout_magnitude:.2f}ATR → +{bks:.0f}分)")
    elif breakout_up or breakout_down:
        bks = _BSS_W_BREAKOUT * 0.4; score += bks
        signals.append(f"突破幅度偏小({breakout_magnitude:.2f}ATR → +{bks:.0f}分)")
    else:
        if not prev_in_range: signals.append("上一根不在平台内，非有效突破")
        else: signals.append("1H尚未突破")
    min_vr = float(cfg.get('min_breakout_volume_ratio', 1.8))
    vrok = vol_ratio >= min_vr; vzok = vol_zscore >= 0.8
    if vrok and vzok:
        vs = _BSS_W_VOLUME * 0.9; score += vs
        signals.append(f"突破放量({vol_ratio:.2f}x, z={vol_zscore:+.2f} → +{vs:.0f}分)")
    elif vrok:
        vs = _BSS_W_VOLUME * 0.6; score += vs
        signals.append(f"量比达标({vol_ratio:.2f}x → +{vs:.0f}分)")
    elif vzok:
        vs = _BSS_W_VOLUME * 0.4; score += vs
        signals.append(f"放量显著(z={vol_zscore:+.2f} → +{vs:.0f}分)")
    else:
        signals.append(f"量能不足({vol_ratio:.2f}x)")
    macd_ok = False
    if len(h1_macd_hist) >= 2:
        mh_last = float(h1_macd_hist.iloc[-1]); mh_prev = float(h1_macd_hist.iloc[-2])
        if breakout_up and mh_last > 0 and mh_last > mh_prev: macd_ok = True
        elif breakout_down and mh_last < 0 and mh_last < mh_prev: macd_ok = True
    if macd_ok:
        score += _BSS_W_MACD; signals.append(f"MACD方向确认(+{_BSS_W_MACD}分)")
    elif (breakout_up or breakout_down) and len(h1_macd_hist) >= 1:
        mh = float(h1_macd_hist.iloc[-1])
        if (breakout_up and mh > 0) or (breakout_down and mh < 0): score += _BSS_W_MACD * 0.4
    rsi_ok = (breakout_up and 50 <= h1_rsi <= 72) or (breakout_down and 28 <= h1_rsi <= 50)
    if rsi_ok:
        score += _BSS_W_RSI; signals.append(f"RSI合理({h1_rsi:.1f} → +{_BSS_W_RSI}分)")
    elif (breakout_up and h1_rsi > 78) or (breakout_down and h1_rsi < 22):
        signals.append(f"RSI极端({'超买' if breakout_up else '超卖'}: {h1_rsi:.1f})")
    max_ext_atr = float(cfg.get('max_extension_atr', 2.5))
    if breakout_up: ext_atr = (last_close - p_high) / h4_atr if h4_atr > 0 else 0.0
    elif breakout_down: ext_atr = (p_low - last_close) / h4_atr if h4_atr > 0 else 0.0
    else: ext_atr = 0.0
    ext_atr = max(ext_atr, 0.0)
    if ext_atr <= max_ext_atr:
        es = _BSS_W_EXTENSION * max(0.0, 1.0 - ext_atr / max(max_ext_atr, 0.1))
        if es >= 2.0: score += es; signals.append(f"延伸适中({ext_atr:.1f}ATR → +{es:.0f}分)")
    else:
        signals.append(f"延伸过远({ext_atr:.1f}ATR)")
    freshness = 6.0 if breakout_magnitude <= 1.5 and (breakout_up or breakout_down) else 0.0
    if freshness > 0: score += freshness
    direction = 'WAIT'
    if breakout_up and vrok: direction = 'BUY'
    elif breakout_down and vrok: direction = 'SELL'
    return _bss_build_result(valid=True, score=score, direction=direction, signals=signals,
        range_width_pct=range_width_pct, range_width_atr=range_width_atr,
        volume_ratio=vol_ratio, vol_zscore=vol_zscore, extension_atr=ext_atr,
        h4_adx=h4_adx, h4_ema_spread=h4_ema_spread, h1_rsi=h1_rsi,
        bb_bandwidth=bb_bandwidth, bullish_ctx=bullish_ctx, bearish_ctx=bearish_ctx)


class BreakoutSwingScanner(BaseScannerStrategy if _HAS_SCANNER_BASE else object):
    required_bars = ['4H', '1H']
    name = "波段平台突破扫描"
    description = "4H 窄幅整理 → 1H 放量突破 + ADX/MACD/收盘强度确认"
    strategy_type = "scan"

    def __init__(self, config=None):
        self.config = {**_BSS_DEFAULT_CONFIG, **(config or {})}
        if _HAS_SCANNER_BASE and hasattr(super(), '__init__'):
            try: super().__init__(config or {})
            except Exception: pass

    def _init_conditions(self):
        if ScanCondition is None: return
        self.add_condition(ScanCondition(name="24H成交量", description="过滤流动性不足标的",
            field="volume_24h", operator=">=", value=self.config.get('min_volume_24h', 15_000_000)))

    def scan_symbol(self, symbol) -> Dict:
        km = symbol.extra_data.get('klines', {})
        try:
            h4 = _to_df(self._get_klines(km, '4H')); h1 = _to_df(self._get_klines(km, '1H'))
            analysis = _bss_analyze_core(h4, h1, getattr(symbol, 'last_price', 0.0), self.config)
        except Exception as exc:
            return {'symbol': getattr(symbol, 'inst_id', ''), 'passed': False, 'score': 0.0, 'direction': 'WAIT', 'details': {'状态': f'分析异常: {exc}'}}
        if not analysis['valid']:
            return {'symbol': getattr(symbol, 'inst_id', ''), 'passed': False, 'score': 0.0, 'direction': 'WAIT', 'details': {'状态': analysis.get('reason', '')}}
        ms = float(self.config.get('min_score', 68))
        passed = analysis['score'] >= ms and analysis['direction'] in {'BUY', 'SELL'}
        result = {'symbol': getattr(symbol, 'inst_id', ''), 'passed': passed, 'score': round(analysis['score'], 2),
                  'direction': analysis['direction'], 'signals': analysis['signals'], 'details': analysis['details'],
                  'last_price': getattr(symbol, 'last_price', 0.0), 'volume_24h': getattr(symbol, 'volume_24h', 0.0),
                  'price_change_24h': getattr(symbol, 'price_change_24h', 0.0), 'category': '波段平台突破',
                  'ranking_factors': analysis.get('ranking_factors', {})}
        if build_opportunity_profile:
            try: result.update(build_opportunity_profile(base_score=analysis['score'], direction=analysis['direction'],
                volume_24h=getattr(symbol, 'volume_24h', 0.0), factors=analysis.get('ranking_factors', {}), signals=analysis['signals']))
            except Exception: pass
        return result

    def _get_klines(self, km, bar):
        return km.get(bar) or km.get(bar.lower()) or km.get(bar.upper()) or []

    def get_config_schema(self):
        return {k: {'type': 'float', 'default': v, 'label': k} for k, v in _BSS_DEFAULT_CONFIG.items()}


# ══════════════════════════════════════════════════════════════════════════════
# 第三节：子策略 2 — 新高突破扫描（NewHighBreakoutScanner）
# ══════════════════════════════════════════════════════════════════════════════

_NHB_DEFAULT_CONFIG = {
    'min_score': 84, 'min_volume_24h': 18_000_000,
    'breakout_window': 55, 'breakout_buffer_pct': 0.2,
    'min_daily_slope_pct': 0.5, 'min_h4_slope_pct': 0.8,
    'min_breakout_volume_ratio': 1.5, 'max_breakout_rsi': 76,
    'max_extension_pct': 5.0, 'max_h4_atr_pct': 6.0,
    'max_extension_atr': 3.0,           # ATR 归一化的"距 4H EMA20"上限
    'min_watch_trend_quality': 58.0,
}


def _nhb_ema_slope_pct(close, span=20, lookback=5):
    ema = close.ewm(span=span, adjust=False).mean()
    if len(ema) <= lookback: return 0.0
    base = float(ema.iloc[-(lookback + 1)]); latest = float(ema.iloc[-1])
    return (latest / base - 1.0) * 100 if base > 0 else 0.0


def _nhb_atr_pct(df, period=14):
    if len(df) < period + 1: return 0.0
    h = df['h']; l = df['l']; c = df['c']; pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    atr_series = tr.rolling(period).mean()
    if atr_series.empty or pd.isna(atr_series.iloc[-1]) or c.empty or c.iloc[-1] <= 0: return 0.0
    return float((atr_series.iloc[-1] / c.iloc[-1]) * 100)


def _nhb_rsi(close, period=14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    if len(loss) == 0 or len(gain) == 0 or pd.isna(loss.iloc[-1]) or pd.isna(gain.iloc[-1]): return 50.0
    if loss.iloc[-1] == 0: return 100.0 if gain.iloc[-1] > 0 else 50.0
    return float(100 - (100 / (1 + gain.iloc[-1] / loss.iloc[-1])))


class NewHighBreakoutScanner(BaseScannerStrategy if _HAS_SCANNER_BASE else object):
    required_bars = ['1D', '4H', '1H']
    name = "新高突破扫描"
    description = "多周期趋势确认 + 1H 创近期新高 + 放量突破"
    strategy_type = "scan"

    def __init__(self, config=None):
        self.config = {**_NHB_DEFAULT_CONFIG, **(config or {})}
        if _HAS_SCANNER_BASE and hasattr(super(), '__init__'):
            try: super().__init__(config or {})
            except Exception: pass

    def _init_conditions(self):
        if ScanCondition is None: return
        self.add_condition(ScanCondition(name="24H成交量", description="过滤流动性不足标的",
            field="volume_24h", operator=">=", value=self.config.get('min_volume_24h', 18_000_000)))

    def scan_symbol(self, symbol) -> Dict:
        klines_map = symbol.extra_data.get('klines', {})
        try:
            analysis = self._analyze(
                self._get_klines(klines_map, '1D'),
                self._get_klines(klines_map, '4H'),
                self._get_klines(klines_map, '1H'),
                getattr(symbol, 'last_price', 0.0))
        except Exception as exc:
            return {'symbol': getattr(symbol, 'inst_id', ''), 'passed': False, 'score': 0.0, 'direction': 'WAIT', 'details': {'状态': f'分析异常: {exc}'}}
        if not analysis['valid']:
            return {'symbol': getattr(symbol, 'inst_id', ''), 'passed': False, 'score': 0.0, 'direction': 'WAIT', 'details': {'状态': analysis['reason']}}
        min_score = float(self.config.get('min_score', 84))
        result = {
            'symbol': getattr(symbol, 'inst_id', ''),
            'passed': analysis['score'] >= min_score and analysis['direction'] == 'BUY',
            'score': round(analysis['score'], 2), 'direction': analysis['direction'],
            'signals': analysis['signals'], 'details': analysis['details'],
            'last_price': getattr(symbol, 'last_price', 0.0), 'volume_24h': getattr(symbol, 'volume_24h', 0.0),
            'price_change_24h': getattr(symbol, 'price_change_24h', 0.0),
            'ranking_factors': analysis.get('ranking_factors', {}),
        }
        if build_opportunity_profile:
            try: result.update(build_opportunity_profile(base_score=analysis['score'], direction=analysis['direction'],
                volume_24h=getattr(symbol, 'volume_24h', 0.0), factors=analysis.get('ranking_factors', {}), signals=analysis['signals']))
            except Exception: pass
        return result

    def _analyze(self, d1_klines, h4_klines, h1_klines, last_price):
        d1 = _to_df(d1_klines); h4 = _to_df(h4_klines); h1 = _to_df(h1_klines)
        if len(d1) < 90: return {'valid': False, 'reason': f'日线数据不足({len(d1)}/90)'}
        if len(h4) < 120: return {'valid': False, 'reason': f'4H数据不足({len(h4)}/120)'}
        if len(h1) < 150: return {'valid': False, 'reason': f'1H数据不足({len(h1)}/150)'}
        score = 0.0; signals = []
        price = float(last_price if last_price > 0 else h1['c'].iloc[-1])
        d1_close = d1['c']; h4_close = h4['c']; h1_close = h1['c']; h1_vol = h1['vol']
        d1_ema20 = float(d1_close.ewm(span=20, adjust=False).mean().iloc[-1])
        d1_ema55 = float(d1_close.ewm(span=55, adjust=False).mean().iloc[-1])
        h4_ema20 = float(h4_close.ewm(span=20, adjust=False).mean().iloc[-1])
        h4_ema55 = float(h4_close.ewm(span=55, adjust=False).mean().iloc[-1])
        h1_ema20 = float(h1_close.ewm(span=20, adjust=False).mean().iloc[-1])
        d1_slope_pct = _nhb_ema_slope_pct(d1_close, span=20, lookback=5)
        h4_slope_pct = _nhb_ema_slope_pct(h4_close, span=20, lookback=5)
        h4_atr_pct = _nhb_atr_pct(h4)
        h1_rsi = _nhb_rsi(h1_close)
        trend_snapshot = _local_trend_snapshot(d1, h4, h1, price)
        trend_metrics = trend_snapshot.get('metrics', {})
        trend_long_score = float(trend_snapshot.get('long_score', 0.0))
        long_trend = (bool(trend_snapshot.get('long_ok'))
            and price > d1_ema20 > d1_ema55 and price > h4_ema20 > h4_ema55
            and d1_slope_pct >= float(self.config.get('min_daily_slope_pct', 0.5))
            and h4_slope_pct >= float(self.config.get('min_h4_slope_pct', 0.8)))
        if long_trend:
            trend_bonus = min(34.0, 22.0 + max(trend_long_score - 68.0, 0.0) * 0.35)
            score += trend_bonus; signals.append(f"多周期趋势质量通过({trend_long_score:.0f})")
        elif trend_long_score >= float(self.config.get('min_watch_trend_quality', 58.0)):
            score += 8; signals.append(f"趋势质量观察级({trend_long_score:.0f})")
        breakout_window = int(self.config.get('breakout_window', 55))
        breakout_buffer_pct = float(self.config.get('breakout_buffer_pct', 0.2))
        breakout_slice = h1['h'].iloc[-(breakout_window + 1):-1] if len(h1) > breakout_window else h1['h'].iloc[:-1]
        breakout_level = float(breakout_slice.max()) if not breakout_slice.empty else price
        prev_close = float(h1_close.iloc[-2]) if len(h1_close) > 1 else price
        buffer_multiplier = 1.0 + breakout_buffer_pct / 100.0
        breakout_confirmed = prev_close <= breakout_level * buffer_multiplier and price > breakout_level * buffer_multiplier
        breakout_pct = ((price - breakout_level) / breakout_level * 100) if breakout_level > 0 else 0.0
        if breakout_confirmed:
            score += 28; signals.append(f"1H突破近{breakout_window}根新高")
        recent_4h_high = float(h4['h'].iloc[-31:-1].max()) if len(h4) > 31 else price
        daily_high = float(d1['h'].iloc[-31:-1].max()) if len(d1) > 31 else price
        if recent_4h_high > 0 and price > recent_4h_high * buffer_multiplier:
            score += 14; signals.append("突破4H阶段高点")
        if daily_high > 0 and price > daily_high * (1.0 - 0.001):
            score += 10; signals.append("接近日线阶段新高")
        tail_mean = float(h1_vol.tail(20).mean()) if not h1_vol.empty else 0.0
        volume_ratio = float(h1_vol.iloc[-1] / tail_mean) if tail_mean > 0 and not h1_vol.empty else 1.0
        if volume_ratio >= float(self.config.get('min_breakout_volume_ratio', 1.5)):
            score += 14; signals.append(f"放量突破({volume_ratio:.2f}x)")
        if price > h1_ema20:
            score += 8; signals.append("1H站稳短均线")
        if 58 <= h1_rsi <= float(self.config.get('max_breakout_rsi', 76)):
            score += 10; signals.append(f"突破RSI健康({h1_rsi:.1f})")
        elif h1_rsi > float(self.config.get('max_breakout_rsi', 76)):
            score -= 6; signals.append(f"突破后RSI偏热({h1_rsi:.1f})")
        extension_pct = abs((price - h4_ema20) / h4_ema20 * 100) if h4_ema20 > 0 else 999.0
        h4_atr_val = _atr(h4)
        extension_atr = abs(price - h4_ema20) / h4_atr_val if h4_atr_val > 0 else 999.0
        max_ext_pct = float(self.config.get('max_extension_pct', 5.0))
        max_ext_atr = float(self.config.get('max_extension_atr', 3.0))
        # 双约束（任一过远即扣分），ATR 化更鲁棒
        if extension_pct <= max_ext_pct and extension_atr <= max_ext_atr:
            score += 8
        else:
            score -= 8
            signals.append(f"距离4H均线偏远({extension_pct:.2f}%/{extension_atr:.2f}ATR)")
        if h4_atr_pct <= float(self.config.get('max_h4_atr_pct', 6.0)): score += 6
        else: score -= 5; signals.append(f"4H波动过大({h4_atr_pct:.2f}%)")
        direction = 'BUY' if (long_trend and breakout_confirmed
                              and extension_pct <= max_ext_pct
                              and extension_atr <= max_ext_atr) else 'WAIT'
        return {
            'valid': True, 'score': max(score, 0.0), 'direction': direction, 'signals': signals,
            'ranking_factors': {
                'trend': trend_long_score, 'trigger': 95.0 if breakout_confirmed else 30.0,
                'volume': min(volume_ratio / max(float(self.config.get('min_breakout_volume_ratio', 1.5)), 0.1), 1.6) * 62.5,
                'location': max(20.0, 100.0 - max(extension_pct - 1.0, 0.0) * 17.0),
                'freshness': max(18.0, 100.0 - max(breakout_pct - 1.0, 0.0) * 22.0),
                'risk': 88.0 if h4_atr_pct <= float(self.config.get('max_h4_atr_pct', 6.0)) else 56.0,
            },
            'details': {
                '评估': ' | '.join(signals) if signals else '暂无新高突破机会',
                '突破幅度': f'{breakout_pct:.2f}%', '量比': f'{volume_ratio:.2f}x',
                '1H_RSI': f'{h1_rsi:.1f}', '趋势质量': f'{trend_long_score:.1f}',
                '趋势诊断': str(trend_snapshot.get('reason', '')),
                'H4_ADX': f"{float(trend_metrics.get('h4_adx', 0.0)):.1f}",
                '趋势效率': f"{float(trend_metrics.get('h1_efficiency', 0.0)):.1f}",
                '延伸幅度': f'{extension_pct:.2f}%', '4H_ATR%': f'{h4_atr_pct:.2f}%',
            }
        }

    def _get_klines(self, klines_map, bar):
        return klines_map.get(bar) or klines_map.get(bar.lower()) or klines_map.get(bar.upper()) or []

    def get_config_schema(self):
        return {k: {'type': 'float', 'default': v, 'label': k} for k, v in _NHB_DEFAULT_CONFIG.items()}


# ══════════════════════════════════════════════════════════════════════════════
# 第四节：子策略 3 — 单边趋势跟随扫描（DirectionalTrendFollowScanner）
# ══════════════════════════════════════════════════════════════════════════════

_DTF_W_TREND       = 28
_DTF_W_MOMENTUM    = 16
_DTF_W_H1_ALIGN    = 12
_DTF_W_ACCEL       = 12
_DTF_W_BREAKOUT    =  8
_DTF_W_VOLUME      =  8
_DTF_W_RSI         =  6
_DTF_W_EXTENSION   =  6
_DTF_W_VOLATILITY  =  4
_DTF_W_3M          =  6
_DTF_ADX_MIN       = 18.0
_DTF_EMA_SPREAD_MIN = 0.20

_DTF_DEFAULT_CONFIG = {
    'min_score': 70, 'min_volume_24h': 18_000_000,
    'min_daily_slope_pct': 0.4, 'min_h4_slope_pct': 0.5,
    'min_breakout_pct': 0.8, 'min_volume_ratio': 1.2,
    'max_extension_atr': 3.5,
    'require_3m_stabilize': True,
    'm3_stabilize_window': 30, 'm3_stabilize_confirm_bars': 5,
    'm3_ema_span': 8, 'm3_slope_lookback': 4,
}

try:
    from src.scanner.trend_quality import trend_quality_snapshot as _external_trend_snapshot
except ImportError:
    _external_trend_snapshot = None


def _dtf_trend_snapshot(d1, h4, h1, price):
    if _external_trend_snapshot:
        try: return _external_trend_snapshot(d1, h4, h1, price)
        except Exception: pass
    return _local_trend_snapshot(d1, h4, h1, price)


def _dtf_ema_slope_pct(c, span, lb):
    e = c.ewm(span=span, adjust=False).mean()
    if len(e) <= lb: return 0.0
    b = float(e.iloc[-(lb + 1)]); l = float(e.iloc[-1])
    return (l / b - 1) * 100 if b > 0 else 0.0


def _dtf_atr_pct(df, period=14):
    a = _atr(df, period); lc = float(df['c'].iloc[-1])
    return float(a / lc * 100) if lc > 0 else 0.0


def _dtf_trend_breakout_pct(df, bull):
    if bull is None or len(df) < 26: return 0.0
    lc = float(df['c'].iloc[-1]); ref = df.iloc[-25:-1]
    if bull:
        rh = float(ref['h'].max())
        return max((lc - rh) / rh * 100, 0.0) if rh > 0 else 0.0
    else:
        rl = float(ref['l'].min())
        return max((rl - lc) / rl * 100, 0.0) if rl > 0 else 0.0


def _dtf_m3_stabilize_default(reason):
    return {'valid': False, 'passed': False, 'quality': 30.0, 'reason': reason,
            'crossover_ok': False, 'ema_slope_ok': False, 'stabilize_ratio': 0.0,
            'last_ema8': 0.0, 'ema8_slope_pct': 0.0}


def _dtf_three_min_stabilize_core(df, *, bull_bias, cfg):
    window = int(cfg.get('m3_stabilize_window', 30))
    confirm_bars = int(cfg.get('m3_stabilize_confirm_bars', 5))
    ema_span = int(cfg.get('m3_ema_span', 8))
    slope_lb = int(cfg.get('m3_slope_lookback', 4))
    if len(df) < ema_span + slope_lb + 2:
        return _dtf_m3_stabilize_default('3m数据不足')
    ema8_s = df['c'].ewm(span=ema_span, adjust=False).mean()
    rc = df['c'].tail(window + confirm_bars).reset_index(drop=True)
    re = ema8_s.tail(window + confirm_bars).reset_index(drop=True)
    n = len(rc)
    if n < 3: return _dtf_m3_stabilize_default('3m样本不足')
    lc = float(rc.iloc[-1]); pc = float(rc.iloc[-2])
    le = float(re.iloc[-1]); pe = float(re.iloc[-2])
    slope_ok = ((bull_bias and le > float(re.iloc[-slope_lb - 1]))
                or (not bull_bias and le < float(re.iloc[-slope_lb - 1])))
    if bull_bias:
        cross_ok = pc <= pe and lc > le
        above_now = lc > le
    else:
        cross_ok = pc >= pe and lc < le
        above_now = lc < le
    cs = rc.tail(confirm_bars); ce = re.tail(confirm_bars)
    sr = float((cs > ce).mean()) if bull_bias else float((cs < ce).mean())
    if cross_ok and slope_ok:
        q = min(100, 65 + sr * 25 + (10 if sr >= 0.8 else 0))
        lb = "多头" if bull_bias else "空头"
        return {'valid': True, 'passed': True, 'quality': q,
                'reason': f"3m{lb}回调企稳EMA{ema_span}(持稳{sr:.0%})",
                'crossover_ok': True, 'ema_slope_ok': True, 'stabilize_ratio': sr,
                'last_ema8': le, 'ema8_slope_pct': (le / float(re.iloc[-slope_lb - 1]) - 1) * 100}
    elif above_now and slope_ok and sr >= 0.6:
        q = 55 + sr * 20
        lb = "多头" if bull_bias else "空头"
        return {'valid': True, 'passed': True, 'quality': q,
                'reason': f"3m{lb}持稳EMA{ema_span}(持稳{sr:.0%}，斜率正确)",
                'crossover_ok': False, 'ema_slope_ok': True, 'stabilize_ratio': sr,
                'last_ema8': le, 'ema8_slope_pct': (le / float(re.iloc[-slope_lb - 1]) - 1) * 100}
    elif not slope_ok:
        return {'valid': True, 'passed': False, 'quality': 20.0,
                'reason': f"3m EMA{ema_span}斜率方向不符",
                'crossover_ok': cross_ok, 'ema_slope_ok': False, 'stabilize_ratio': sr,
                'last_ema8': le, 'ema8_slope_pct': (le / float(re.iloc[-slope_lb - 1]) - 1) * 100}
    else:
        lb = "多头" if bull_bias else "空头"
        return {'valid': True, 'passed': False, 'quality': 30.0,
                'reason': f"3m价格在EMA{ema_span}{'下方' if bull_bias else '上方'}",
                'crossover_ok': False, 'ema_slope_ok': slope_ok, 'stabilize_ratio': sr,
                'last_ema8': le, 'ema8_slope_pct': (le / float(re.iloc[-slope_lb - 1]) - 1) * 100}


def _dtf_build_result(*, valid, score, direction, signals, d1_slope, h4_slope, vol_ratio, vol_zscore,
                      h1_rsi, extension_atr, breakout_pct, h4_atr_pct, d1_adx, h4_adx,
                      d1_ema_spread, h4_ema_spread, bullish, bearish,
                      trend_long_score, trend_short_score, trend_metrics, m3_stab, reason=''):
    tq = trend_long_score if bullish else trend_short_score if bearish else max(trend_long_score, trend_short_score)
    trigger_q = 88.0 if breakout_pct >= 0.8 else 45.0
    vol_q = min(vol_ratio / 1.2, 1.6) * 62.5
    loc_q = max(20, 100 - max(extension_atr - 1, 0) * 18)
    fresh_q = 92.0 if direction in {'BUY', 'SELL'} and breakout_pct <= 3.5 else 68.0 if direction in {'BUY', 'SELL'} else 30.0
    return {
        'valid': valid, 'reason': reason, 'score': max(score, 0.0), 'direction': direction, 'signals': signals,
        'ranking_factors': {'trend': tq, 'trigger': trigger_q, 'volume': vol_q, 'location': loc_q,
                            'freshness': fresh_q, 'risk': 88.0 if h4_atr_pct <= 6 else 55.0},
        'details': {
            '评估': ' | '.join(signals) if signals else '暂无单边趋势跟随机会',
            '日线斜率': f'{d1_slope:.2f}%', '4H斜率': f'{h4_slope:.2f}%',
            '日线ADX': f'{d1_adx:.1f}', '4H_ADX': f'{h4_adx:.1f}',
            '日线EMA发散': f'{d1_ema_spread:+.2f}%', '4H_EMA发散': f'{h4_ema_spread:+.2f}%',
            '量比': f'{vol_ratio:.2f}x', '量能Z分': f'{vol_zscore:+.2f}σ',
            '1H_RSI': f'{h1_rsi:.1f}', '延伸(ATR倍)': f'{extension_atr:.1f}',
            '4H_ATR%': f'{h4_atr_pct:.2f}%',
            '趋势质量': str(trend_metrics.get('reason', '-') or '-'),
            'H4_ADX_内置': f"{float(trend_metrics.get('h4_adx', 0)):.1f}",
            '趋势效率': f"{float(trend_metrics.get('h1_efficiency', 0)):.1f}",
            '3m企稳确认': '通过' if m3_stab.get('passed') else '未通过',
            '3m企稳说明': str(m3_stab.get('reason', '-')),
            '3m EMA8': f"{float(m3_stab.get('last_ema8', 0)):.8g}",
            '3m EMA8斜率': f"{float(m3_stab.get('ema8_slope_pct', 0)):.2f}%",
            '3m持稳比例': f"{float(m3_stab.get('stabilize_ratio', 0)):.0%}",
        },
    }


def _dtf_analyze_core(d1, h4, h1, m3, last_price, cfg):
    _check_df(d1, '日线', 90); _check_df(h4, '4H', 120); _check_df(h1, '1H', 150)
    require_3m = bool(cfg.get('require_3m_stabilize', True))
    min_3m_bars = int(cfg.get('m3_stabilize_window', 30)) + 10
    # 3m 数据缺失时，自动降级为不强制 3m 校验，而非直接抛异常
    if require_3m and len(m3) < min_3m_bars:
        require_3m = False
    score = 0.0; signals = []
    price = float(last_price) if last_price and last_price > 0 else float(h1['c'].iloc[-1])
    d1_ema21 = _ema(d1['c'], 21); d1_ema55 = _ema(d1['c'], 55)
    h4_ema21 = _ema(h4['c'], 21); h4_ema55 = _ema(h4['c'], 55)
    h1_ema21 = _ema(h1['c'], 21); h1_ema55 = _ema(h1['c'], 55)
    d1_slope = _dtf_ema_slope_pct(d1['c'], 21, 6)
    h4_slope = _dtf_ema_slope_pct(h4['c'], 21, 6)
    h1_rsi = _rsi_wilder(h1['c'])
    h4_atr = _atr(h4); h4_atr_pct = _dtf_atr_pct(h4)
    vol_ratio = _volume_ratio_adjusted(h1)
    vol_zscore = _volume_zscore(h1['vol'])
    d1_adx = _adx(d1, 14); h4_adx = _adx(h4, 14)
    extension_atr = abs(price - h4_ema21) / h4_atr if h4_atr > 0 else 0.0
    d1_ema_spread = (d1_ema21 - d1_ema55) / d1_ema55 * 100 if d1_ema55 > 0 else 0.0
    h4_ema_spread = (h4_ema21 - h4_ema55) / h4_ema55 * 100 if h4_ema55 > 0 else 0.0
    _, _, h1_macd_hist = _macd(h1['c'])
    _, _, h4_macd_hist = _macd(h4['c'])
    trend_snap = _dtf_trend_snapshot(d1, h4, h1, price)
    trend_metrics = trend_snap.get('metrics', {})
    trend_long_score = float(trend_snap.get('long_score', 0) or 0)
    trend_short_score = float(trend_snap.get('short_score', 0) or 0)
    _m3_def = _dtf_m3_stabilize_default('趋势未确认')
    min_d1_slope = float(cfg.get('min_daily_slope_pct', 0.4))
    min_h4_slope = float(cfg.get('min_h4_slope_pct', 0.5))
    bullish = (bool(trend_snap.get('long_ok'))
               and price > d1_ema21 > d1_ema55 and price > h4_ema21 > h4_ema55
               and d1_slope > min_d1_slope and h4_slope > min_h4_slope
               and d1_adx >= _DTF_ADX_MIN and h4_adx >= _DTF_ADX_MIN
               and d1_ema_spread >= _DTF_EMA_SPREAD_MIN and h4_ema_spread >= _DTF_EMA_SPREAD_MIN)
    bearish = (bool(trend_snap.get('short_ok'))
               and price < d1_ema21 < d1_ema55 and price < h4_ema21 < h4_ema55
               and d1_slope < -min_d1_slope and h4_slope < -min_h4_slope
               and d1_adx >= _DTF_ADX_MIN and h4_adx >= _DTF_ADX_MIN
               and d1_ema_spread <= -_DTF_EMA_SPREAD_MIN and h4_ema_spread <= -_DTF_EMA_SPREAD_MIN)
    if bullish or bearish:
        base_t = 20.0
        spread_abs = (abs(d1_ema_spread) + abs(h4_ema_spread)) / 2.0
        spread_bonus = min(4.0, spread_abs / 2.0 * 4.0)
        adx_avg = (d1_adx + h4_adx) / 2.0
        adx_bonus = min(4.0, max(0.0, (adx_avg - _DTF_ADX_MIN) / 12.0 * 4.0))
        ts_ = base_t + spread_bonus + adx_bonus; score += ts_
        dl = "多头" if bullish else "空头"
        ref = trend_long_score if bullish else trend_short_score
        signals.append(f"{dl}趋势通过(质量{ref:.0f}, ADX {adx_avg:.1f}, 发散{spread_abs:.2f}% → +{ts_:.1f}分)")
    else:
        rp = []
        if not (bool(trend_snap.get('long_ok')) or bool(trend_snap.get('short_ok'))):
            rp.append(str(trend_snap.get('reason', '趋势质量不足')))
        if d1_adx < _DTF_ADX_MIN: rp.append(f"日线ADX不足({d1_adx:.1f})")
        if h4_adx < _DTF_ADX_MIN: rp.append(f"4H ADX不足({h4_adx:.1f})")
        if abs(d1_slope) < min_d1_slope: rp.append(f"日线斜率不足({d1_slope:.2f}%)")
        if abs(h4_slope) < min_h4_slope: rp.append(f"4H斜率不足({h4_slope:.2f}%)")
        signals.append("趋势未确认: " + " | ".join(rp) if rp else "趋势未确认")
        return _dtf_build_result(valid=True, score=score, direction='WAIT', signals=signals,
            d1_slope=d1_slope, h4_slope=h4_slope, vol_ratio=vol_ratio, vol_zscore=vol_zscore,
            h1_rsi=h1_rsi, extension_atr=extension_atr, breakout_pct=0.0,
            h4_atr_pct=h4_atr_pct, d1_adx=d1_adx, h4_adx=h4_adx,
            d1_ema_spread=d1_ema_spread, h4_ema_spread=h4_ema_spread,
            bullish=bullish, bearish=bearish,
            trend_long_score=trend_long_score, trend_short_score=trend_short_score,
            trend_metrics=trend_metrics, m3_stab=_m3_def)
    slope_score = min(8.0, 4.0 + abs(d1_slope + h4_slope) / 4.0 * 4.0); score += slope_score
    macd_expanding = False
    if len(h4_macd_hist) >= 3:
        mh = h4_macd_hist.values[-3:]
        if bullish: macd_expanding = mh[-1] > mh[-2] > mh[-3] and mh[-1] > 0
        else: macd_expanding = mh[-1] < mh[-2] < mh[-3] and mh[-1] < 0
    if macd_expanding:
        score += 8.0; signals.append("MACD持续扩张(+8分)")
    elif len(h4_macd_hist) >= 2:
        last_mh = float(h4_macd_hist.iloc[-1])
        if (bullish and last_mh > 0) or (bearish and last_mh < 0):
            score += 4.0; signals.append("MACD方向正确(+4分)")
    signals.append(f"动量(斜率{d1_slope:.2f}%/{h4_slope:.2f}% → +{slope_score:.0f}分)")
    h1_align_bull = bullish and price > h1_ema21 > h1_ema55
    h1_align_bear = bearish and price < h1_ema21 < h1_ema55
    if h1_align_bull or h1_align_bear:
        h1_ema21_s = h1['c'].ewm(span=21, adjust=False).mean()
        h1_ema55_s = h1['c'].ewm(span=55, adjust=False).mean()
        recent_c = h1['c'].tail(10).values
        r21 = h1_ema21_s.tail(10).values; r55 = h1_ema55_s.tail(10).values
        if bullish: stability = float(((recent_c > r21) & (r21 > r55)).mean())
        else: stability = float(((recent_c < r21) & (r21 < r55)).mean())
        align_score = _DTF_W_H1_ALIGN * (0.6 + stability * 0.4); score += align_score
        signals.append(f"1H顺势{'多' if bullish else '空'}头排列(稳定{stability:.0%} → +{align_score:.1f}分)")
    else:
        signals.append("1H EMA排列尚未顺势")
    accel_score = 0.0
    breakout_pct = _dtf_trend_breakout_pct(h1, bull=bullish)
    if breakout_pct > 0.3: accel_score += 4.0
    if len(h1) >= 23:
        recent_vols = h1['vol'].iloc[-3:].values
        baseline_vol = float(h1['vol'].iloc[-23:-3].mean())
        if baseline_vol > 0 and max(recent_vols) / baseline_vol >= 1.5: accel_score += 4.0
    if len(h1_macd_hist) >= 3:
        mh1 = h1_macd_hist.values[-3:]
        if (bullish and mh1[-1] > mh1[-2] > mh1[-3]) or (bearish and mh1[-1] < mh1[-2] < mh1[-3]):
            accel_score += 4.0
    if accel_score > 0: score += accel_score; signals.append(f"趋势加速信号(+{accel_score:.0f}分)")
    else: signals.append("无新加速迹象")
    min_bp = float(cfg.get('min_breakout_pct', 0.8))
    if breakout_pct >= min_bp:
        bp_s = _DTF_W_BREAKOUT * min(1.0, 0.6 + (breakout_pct - min_bp) / max(min_bp * 2, 0.1) * 0.4)
        score += bp_s; signals.append(f"1H创新段({breakout_pct:.2f}% → +{bp_s:.0f}分)")
    min_vr = float(cfg.get('min_volume_ratio', 1.2))
    vrok = vol_ratio >= min_vr; vzok = vol_zscore >= 0.5
    if vrok and vzok:
        vs = _DTF_W_VOLUME * 0.85; score += vs; signals.append(f"量能强({vol_ratio:.2f}x, z={vol_zscore:+.2f} → +{vs:.0f}分)")
    elif vrok or vzok:
        vs = _DTF_W_VOLUME * 0.5; score += vs; signals.append(f"量能部分({vol_ratio:.2f}x, z={vol_zscore:+.2f} → +{vs:.0f}分)")
    else:
        signals.append(f"量能不足({vol_ratio:.2f}x)")
    rsi_ok = (bullish and 55 <= h1_rsi <= 72) or (bearish and 28 <= h1_rsi <= 45)
    if rsi_ok: score += _DTF_W_RSI; signals.append(f"RSI健康({h1_rsi:.1f} → +{_DTF_W_RSI}分)")
    elif (bullish and h1_rsi > 80) or (bearish and h1_rsi < 20):
        signals.append(f"RSI极端({'超买' if bullish else '超卖'}: {h1_rsi:.1f})")
    max_ext = float(cfg.get('max_extension_atr', 3.5))
    if extension_atr <= max_ext:
        es = _DTF_W_EXTENSION * max(0.0, 1.0 - extension_atr / max(max_ext, 0.1))
        if es >= 1.5: score += es; signals.append(f"延伸适中({extension_atr:.1f}ATR → +{es:.0f}分)")
    else:
        signals.append(f"延伸偏大({extension_atr:.1f}ATR)")
    if 1.5 <= h4_atr_pct <= 6.0:
        score += _DTF_W_VOLATILITY; signals.append(f"波动率合理({h4_atr_pct:.2f}% → +{_DTF_W_VOLATILITY}分)")
    m3_stab = (_dtf_three_min_stabilize_core(m3, bull_bias=bullish, cfg=cfg)
               if len(m3) >= min_3m_bars else _dtf_m3_stabilize_default('3m数据不足'))
    if m3_stab.get('passed'): score += _DTF_W_3M; signals.append(str(m3_stab.get('reason')))
    elif m3_stab.get('valid'): signals.append(f"3m观察：{m3_stab.get('reason')}")
    has_accel = accel_score > 0
    m3_ok = bool(m3_stab.get('passed')) or not require_3m
    direction = 'WAIT'
    if h1_align_bull and has_accel and m3_ok: direction = 'BUY'
    elif h1_align_bear and has_accel and m3_ok: direction = 'SELL'
    return _dtf_build_result(valid=True, score=score, direction=direction, signals=signals,
        d1_slope=d1_slope, h4_slope=h4_slope, vol_ratio=vol_ratio, vol_zscore=vol_zscore,
        h1_rsi=h1_rsi, extension_atr=extension_atr, breakout_pct=breakout_pct,
        h4_atr_pct=h4_atr_pct, d1_adx=d1_adx, h4_adx=h4_adx,
        d1_ema_spread=d1_ema_spread, h4_ema_spread=h4_ema_spread,
        bullish=bullish, bearish=bearish,
        trend_long_score=trend_long_score, trend_short_score=trend_short_score,
        trend_metrics=trend_metrics, m3_stab=m3_stab)


class DirectionalTrendFollowScanner(BaseScannerStrategy if _HAS_SCANNER_BASE else object):
    required_bars = ['1D', '4H', '1H', '3m']
    name = "单边趋势跟随扫描"
    description = "多周期单边趋势 + ADX/发散度/MACD 确认 + 趋势加速信号 + 3m 企稳"
    strategy_type = "scan"

    def __init__(self, config=None):
        self.config = {**_DTF_DEFAULT_CONFIG, **(config or {})}
        if _HAS_SCANNER_BASE and hasattr(super(), '__init__'):
            try: super().__init__(config or {})
            except Exception: pass

    def _init_conditions(self):
        if ScanCondition is None: return
        self.add_condition(ScanCondition(name="24H成交量", description="过滤流动性不足标的",
            field="volume_24h", operator=">=", value=self.config.get('min_volume_24h', 18_000_000)))

    def scan_symbol(self, symbol) -> Dict:
        km = symbol.extra_data.get('klines', {})
        try:
            d1 = _to_df(self._get_klines(km, '1D')); h4 = _to_df(self._get_klines(km, '4H'))
            h1 = _to_df(self._get_klines(km, '1H')); m3 = _to_df(self._get_klines(km, '3m'))
            analysis = _dtf_analyze_core(d1, h4, h1, m3, getattr(symbol, 'last_price', 0.0), self.config)
        except Exception as exc:
            return {'symbol': getattr(symbol, 'inst_id', ''), 'passed': False, 'score': 0.0, 'direction': 'WAIT', 'details': {'状态': f'分析异常: {exc}'}}
        if not analysis['valid']:
            return {'symbol': getattr(symbol, 'inst_id', ''), 'passed': False, 'score': 0.0, 'direction': 'WAIT', 'details': {'状态': analysis.get('reason', '')}}
        ms = float(self.config.get('min_score', 70))
        passed = analysis['score'] >= ms and analysis['direction'] in {'BUY', 'SELL'}
        result = {'symbol': getattr(symbol, 'inst_id', ''), 'passed': passed, 'score': round(analysis['score'], 2),
                  'direction': analysis['direction'], 'signals': analysis['signals'], 'details': analysis['details'],
                  'last_price': getattr(symbol, 'last_price', 0.0), 'volume_24h': getattr(symbol, 'volume_24h', 0.0),
                  'price_change_24h': getattr(symbol, 'price_change_24h', 0.0), 'category': '单边趋势跟随',
                  'ranking_factors': analysis.get('ranking_factors', {})}
        if build_opportunity_profile:
            try: result.update(build_opportunity_profile(base_score=analysis['score'], direction=analysis['direction'],
                volume_24h=getattr(symbol, 'volume_24h', 0.0), factors=analysis.get('ranking_factors', {}), signals=analysis['signals']))
            except Exception: pass
        return result

    def _get_klines(self, km, bar):
        return km.get(bar) or km.get(bar.lower()) or km.get(bar.upper()) or []

    def get_config_schema(self):
        return {k: {'type': 'float', 'default': v, 'label': k} for k, v in _DTF_DEFAULT_CONFIG.items()}


# ══════════════════════════════════════════════════════════════════════════════
# 第五节：子策略 4 — 背离反转扫描（DivergenceReversalScanner）
# ══════════════════════════════════════════════════════════════════════════════

_DR_W_DIVERGENCE   = 30
_DR_W_CANDLE       = 14
_DR_W_VOLUME       = 12
_DR_W_RSI_POS      = 10
_DR_W_LOCATION     =  8
_DR_W_MACD_CROSS   =  8
_DR_W_H4_DIV       =  6
_DR_W_FRESHNESS    =  6
_DR_W_VOL_ZSCORE   =  6
_DR_W_3M           =  6

_DR_DEFAULT_CONFIG = {
    'min_score': 65, 'min_volume_24h': 12_000_000,
    'max_h4_rsi_for_buy': 42, 'min_h4_rsi_for_sell': 58,
    'min_volume_ratio': 1.15, 'max_reversal_range_pct': 4.0,
    'divergence_window': 48, 'min_pivot_separation': 8,
    'recover_atr_multiple': 0.3,
    'require_3m_divergence': True,
    'm3_reversal_window': 80, 'm3_neckline_bars': 10,
    'm3_min_pullback_bars': 3, 'm3_breakout_buffer_pct': 0.12,
    'm3_origin_tolerance_pct': 0.18,
}


def _dr_rsi_series(close, period=14):
    if len(close) < period + 1: return pd.Series([50.0] * len(close), index=close.index)
    delta = close.diff(); gain = delta.clip(lower=0); loss = (-delta).clip(lower=0)
    ag = gain.ewm(alpha=1 / period, adjust=False).mean()
    al = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = ag / al.replace(0, float('nan'))
    return (100 - 100 / (1 + rs)).fillna(50.0)


def _dr_rsi_scalar(close, period=14):
    s = _dr_rsi_series(close, period)
    return float(s.iloc[-1]) if len(s) else 50.0


def _dr_find_pivots(series, find_min, window, min_sep, *, left: int = 3, right: int = 2):
    """
    左右窗口 pivot：第 i 根需在 [i-left, i+right] 中为局部极值。
    返回最近的两个 pivot 索引（升序），找不到则降级为旧的首/末半 idxmin/max。
    """
    if len(series) < min_sep + 2: return []
    s = series.tail(window).reset_index(drop=True)
    n = len(s)
    if n < min_sep + left + right + 2: return []
    vals = s.values
    pivots = []
    for i in range(left, n - right):
        seg = vals[i - left:i + right + 1]
        v = vals[i]
        if find_min:
            if v == seg.min() and (vals[i - 1] > v or vals[i + 1] > v):
                pivots.append(i)
        else:
            if v == seg.max() and (vals[i - 1] < v or vals[i + 1] < v):
                pivots.append(i)
    # 保留最近两个，且间隔 ≥ min_sep
    if len(pivots) >= 2:
        p2 = pivots[-1]
        p1 = None
        for p in reversed(pivots[:-1]):
            if p2 - p >= min_sep:
                p1 = p; break
        if p1 is not None:
            return [p1, p2]
    # 降级回旧实现（保持向后兼容）
    half = max(min_sep + 1, n // 2)
    if find_min:
        p2 = int(s.iloc[half:].idxmin()); p1_end = max(0, p2 - min_sep)
        if p1_end <= 0: return [p2]
        p1 = int(s.iloc[:p1_end].idxmin())
    else:
        p2 = int(s.iloc[half:].idxmax()); p1_end = max(0, p2 - min_sep)
        if p1_end <= 0: return [p2]
        p1 = int(s.iloc[:p1_end].idxmax())
    if p2 <= p1: return [p2]
    return [p1, p2]


def _dr_bullish_divergence(df, rsi_series, macd_hist, window, min_sep, recover_threshold):
    try:
        if len(df) < window or len(rsi_series) < window or len(macd_hist) < window: return False, {}
        pl = df['l'].tail(window).reset_index(drop=True); cl = df['c'].tail(window).reset_index(drop=True)
        rw = rsi_series.tail(window).reset_index(drop=True); mw = macd_hist.tail(window).reset_index(drop=True)
        pvs = _dr_find_pivots(pl, find_min=True, window=window, min_sep=min_sep)
        if len(pvs) < 2: return False, {}
        p1, p2 = pvs[0], pvs[1]
        if p2 <= p1: return False, {}
        pll = float(pl.iloc[p2]) < float(pl.iloc[p1]) * 0.998
        rg = float(rw.iloc[p2]) - float(rw.iloc[p1]); rhl = rg > 2.5
        mhl = float(mw.iloc[p2]) > float(mw.iloc[p1])
        cr = float(cl.iloc[-1]) >= float(cl.iloc[p2]) + recover_threshold
        passed = pll and rhl and mhl and cr
        return passed, ({'rsi_gap': abs(rg), 'p1': p1, 'p2': p2} if passed else {})
    except Exception: return False, {}


def _dr_bearish_divergence(df, rsi_series, macd_hist, window, min_sep, recover_threshold):
    try:
        if len(df) < window or len(rsi_series) < window or len(macd_hist) < window: return False, {}
        ph = df['h'].tail(window).reset_index(drop=True); cl = df['c'].tail(window).reset_index(drop=True)
        rw = rsi_series.tail(window).reset_index(drop=True); mw = macd_hist.tail(window).reset_index(drop=True)
        pvs = _dr_find_pivots(ph, find_min=False, window=window, min_sep=min_sep)
        if len(pvs) < 2: return False, {}
        p1, p2 = pvs[0], pvs[1]
        if p2 <= p1: return False, {}
        phh = float(ph.iloc[p2]) > float(ph.iloc[p1]) * 1.002
        rg = float(rw.iloc[p1]) - float(rw.iloc[p2]); rlh = rg > 2.5
        mlh = float(mw.iloc[p2]) < float(mw.iloc[p1])
        cw = float(cl.iloc[-1]) <= float(cl.iloc[p2]) - recover_threshold
        passed = phh and rlh and mlh and cw
        return passed, ({'rsi_gap': abs(rg), 'p1': p1, 'p2': p2} if passed else {})
    except Exception: return False, {}


def _dr_reversal_range_pct(df, price, is_bull):
    if len(df) < 25 or price <= 0: return 0.0
    if is_bull:
        ref = float(df['l'].tail(25).min())
        return (price - ref) / ref * 100 if ref > 0 else 0.0
    else:
        ref = float(df['h'].tail(25).max())
        return (ref - price) / ref * 100 if ref > 0 else 0.0


def _dr_m3_default_result(reason, origin_price=0.0):
    return {'valid': False, 'passed': False, 'quality': 35.0, 'reason': reason, 'phase': 0,
            'bull_bias': True, 'origin_price': origin_price, 'breakout_level': 0.0,
            'phase1_extreme': 0.0, 'pb_extreme': 0.0, 'confirmation_level': 0.0,
            'phase3_close': 0.0, 'breakout_pct': 0.0, 'continuation_pct': 0.0, 'pullback_depth_pct': 0.0}


def _dr_m3_result_failed(reason, *, origin_price, breakout_level, pb_extreme, phase1_extreme, bull_bias):
    bp = abs(phase1_extreme - breakout_level) / breakout_level * 100 if breakout_level > 0 else 0.0
    if bull_bias: dp = (pb_extreme - origin_price) / origin_price * 100 if origin_price > 0 else -999
    else: dp = (origin_price - pb_extreme) / origin_price * 100 if origin_price > 0 else -999
    return {'valid': True, 'passed': False, 'quality': 15.0, 'reason': reason, 'phase': 2, 'bull_bias': bull_bias,
            'origin_price': origin_price, 'breakout_level': breakout_level, 'phase1_extreme': phase1_extreme,
            'pb_extreme': pb_extreme, 'confirmation_level': phase1_extreme, 'phase3_close': 0.0,
            'breakout_pct': bp, 'continuation_pct': 0.0, 'pullback_depth_pct': dp}


def _dr_three_min_reversal_core(df, *, bull_bias, cfg):
    window = int(cfg.get('m3_reversal_window', 80)); neckline_bars = int(cfg.get('m3_neckline_bars', 10))
    origin_tol = float(cfg.get('m3_origin_tolerance_pct', 0.18)); breakout_buf = float(cfg.get('m3_breakout_buffer_pct', 0.12))
    min_pb_bars = int(cfg.get('m3_min_pullback_bars', 3)); min_bars = neckline_bars + min_pb_bars + 6
    if len(df) < max(window, min_bars): return _dr_m3_default_result(f'3m数据不足({len(df)}/{max(window, min_bars)})')
    recent = df.tail(window).reset_index(drop=True); n = len(recent)
    ose = max(neckline_bars + min_pb_bars + 4, n // 2)
    if bull_bias: op = int(recent.iloc[:ose]['l'].idxmin()); opr = float(recent['l'].iloc[op])
    else: op = int(recent.iloc[:ose]['h'].idxmax()); opr = float(recent['h'].iloc[op])
    ns_ = op + 1; ne = min(ns_ + neckline_bars, n - min_pb_bars - 3)
    if ne <= ns_ + 1: return _dr_m3_default_result('3m颈线样本不足', origin_price=opr)
    neck = recent.iloc[ns_:ne]
    if bull_bias:
        bl = float(neck['h'].max())
        if not pd.notna(bl) or bl <= 0: return _dr_m3_default_result('3m颈线异常', origin_price=opr)
        tl = bl * (1 + breakout_buf / 100); fl = opr * (1 - origin_tol / 100)
    else:
        bl = float(neck['l'].min())
        if not pd.notna(bl) or bl <= 0: return _dr_m3_default_result('3m颈线异常(空)', origin_price=opr)
        tl = bl * (1 - breakout_buf / 100); fl = opr * (1 + origin_tol / 100)
    phase = 0; p1e = opr; cl_ = tl; pbe = float('inf') if bull_bias else float('-inf'); pbc = 0; p3c = 0.0
    for i in range(ne, n):
        bc = float(recent['c'].iloc[i]); bh = float(recent['h'].iloc[i]); bll = float(recent['l'].iloc[i])
        if bull_bias:
            if phase == 0:
                if bc > tl: phase = 1; p1e = bh; cl_ = bh
            elif phase == 1:
                if bc > tl:
                    if bh > p1e: p1e = bh; cl_ = bh
                else:
                    if bc < fl: return _dr_m3_result_failed('Phase2收盘跌破', origin_price=opr, breakout_level=bl, pb_extreme=bc, phase1_extreme=p1e, bull_bias=True)
                    phase = 2; pbe = bll; pbc = 1
            elif phase == 2:
                pbe = min(pbe, bll); pbc += 1
                if bc < fl: return _dr_m3_result_failed('Phase2收盘跌破', origin_price=opr, breakout_level=bl, pb_extreme=pbe, phase1_extreme=p1e, bull_bias=True)
                if pbc >= min_pb_bars and bc > cl_: phase = 3; p3c = bc; break
        else:
            if phase == 0:
                if bc < tl: phase = 1; p1e = bll; cl_ = bll
            elif phase == 1:
                if bc < tl:
                    if bll < p1e: p1e = bll; cl_ = bll
                else:
                    if bc > fl: return _dr_m3_result_failed('Phase2收盘超起点', origin_price=opr, breakout_level=bl, pb_extreme=bc, phase1_extreme=p1e, bull_bias=False)
                    phase = 2; pbe = bh; pbc = 1
            elif phase == 2:
                pbe = max(pbe, bh); pbc += 1
                if bc > fl: return _dr_m3_result_failed('Phase2收盘超起点', origin_price=opr, breakout_level=bl, pb_extreme=pbe, phase1_extreme=p1e, bull_bias=False)
                if pbc >= min_pb_bars and bc < cl_: phase = 3; p3c = bc; break
    lc = float(recent['c'].iloc[-1])
    if phase == 3:
        if bull_bias: dp = (pbe - opr) / opr * 100; cp = (p3c - cl_) / cl_ * 100; held = lc > p3c
        else: dp = (opr - pbe) / opr * 100; cp = (cl_ - p3c) / cl_ * 100; held = lc < p3c
        q = min(100, 60 + min(dp + 2, 4) * 5 + min(cp, 3) * 4 + (8 if held else 0))
        lb = "多头" if bull_bias else "空头"
        return {'valid': True, 'passed': True, 'quality': q,
                'reason': f"3m{lb}反转三段确认：突破→回踩({dp:+.2f}%)→继续(+{cp:.2f}%)",
                'phase': 3, 'bull_bias': bull_bias, 'origin_price': opr, 'breakout_level': bl,
                'phase1_extreme': p1e, 'pb_extreme': pbe, 'confirmation_level': cl_, 'phase3_close': p3c,
                'breakout_pct': abs(p1e - bl) / bl * 100, 'continuation_pct': cp, 'pullback_depth_pct': dp}
    elif phase == 2:
        if bull_bias: dp = (pbe - opr) / opr * 100; fh = pbe >= fl
        else: dp = (opr - pbe) / opr * 100; fh = pbe <= fl
        q = 35 + 25 + (20 if fh else 0)
        r = f"3m已突破并回踩({dp:+.2f}%)，等继续" if fh else f"3m回踩越过起点({dp:+.2f}%)"
        return {'valid': True, 'passed': False, 'quality': float(q), 'reason': r, 'phase': 2, 'bull_bias': bull_bias,
                'origin_price': opr, 'breakout_level': bl, 'phase1_extreme': p1e, 'pb_extreme': pbe,
                'confirmation_level': cl_, 'phase3_close': 0.0,
                'breakout_pct': abs(p1e - bl) / bl * 100, 'continuation_pct': 0.0, 'pullback_depth_pct': dp}
    elif phase == 1:
        lb = "多头" if bull_bias else "空头"
        return {'valid': True, 'passed': False, 'quality': 60.0, 'reason': f"3m{lb}已突破颈线，等回踩",
                'phase': 1, 'bull_bias': bull_bias, 'origin_price': opr, 'breakout_level': bl,
                'phase1_extreme': p1e, 'pb_extreme': float('nan'), 'confirmation_level': cl_, 'phase3_close': 0.0,
                'breakout_pct': abs(p1e - bl) / bl * 100, 'continuation_pct': 0.0, 'pullback_depth_pct': 0.0}
    else:
        return _dr_m3_default_result('3m尚未完成初次突破', origin_price=opr)


def _dr_build_result(*, valid, score, direction, signals, h4_rsi, h1_rsi, vol_ratio, vol_zscore,
                     reversal_range_pct, bull_div, bear_div, m3_confirm, reason=''):
    if bull_div: eq = min(100, max(20, (42 - h4_rsi) * 4 + 55))
    elif bear_div: eq = min(100, max(20, (h4_rsi - 58) * 4 + 55))
    else: eq = 30.0
    m3p = bool(m3_confirm.get('passed'))
    return {
        'valid': valid, 'reason': reason, 'score': max(score, 0.0),
        'direction': direction, 'signals': signals,
        'ranking_factors': {
            'trend': eq, 'trigger': 90.0 if direction in {'BUY', 'SELL'} else 40.0,
            'volume': min(vol_ratio / 1.15, 1.6) * 62.5,
            'location': max(20, 100 - reversal_range_pct * 16),
            'freshness': 90.0 if direction in {'BUY', 'SELL'} else 35.0,
            'risk': 86.0 if m3p else 65.0 if m3_confirm.get('valid') else 45.0,
        },
        'details': {
            '评估': ' | '.join(signals) if signals else '暂无背离反转机会',
            '4H_RSI': f'{h4_rsi:.1f}', '1H_RSI': f'{h1_rsi:.1f}',
            '量比': f'{vol_ratio:.2f}x', '量能Z分': f'{vol_zscore:+.2f}σ',
            '位置偏离': f'{reversal_range_pct:.2f}%',
            '3m结构确认': '通过' if m3p else '未通过',
            '3m当前阶段': f"Phase {m3_confirm.get('phase', 0)}",
            '3m结构说明': str(m3_confirm.get('reason', '-')),
            '3m起点价': f"{float(m3_confirm.get('origin_price', 0)):.8g}",
            '3m颈线突破位': f"{float(m3_confirm.get('breakout_level', 0)):.8g}",
            '3mPhase1极值': f"{float(m3_confirm.get('phase1_extreme', 0)):.8g}",
            '3m回踩极值': f"{float(m3_confirm.get('pb_extreme', 0) or 0):.8g}",
            '3m回踩深度': f"{float(m3_confirm.get('pullback_depth_pct', 0)):.2f}%",
            '3m继续突破幅度': f"{float(m3_confirm.get('continuation_pct', 0)):.2f}%",
        },
    }


def _dr_analyze_core(h4, h1, m3, last_price, cfg):
    _check_df(h4, '4H', 90); _check_df(h1, '1H', 120)
    require_3m = bool(cfg.get('require_3m_divergence', True))
    min_3m_bars = max(int(cfg.get('m3_reversal_window', 80)),
                      int(cfg.get('m3_neckline_bars', 10)) + int(cfg.get('m3_min_pullback_bars', 3)) + 6)
    if require_3m and len(m3) < min_3m_bars:
        require_3m = False  # 3m 数据缺失时降级
    score = 0.0; signals = []
    price = float(last_price) if last_price and last_price > 0 else float(h1['c'].iloc[-1])
    h4_rsi = _dr_rsi_scalar(h4['c']); h1_rsi_s = _dr_rsi_series(h1['c'])
    h1_rsi = float(h1_rsi_s.iloc[-1]) if len(h1_rsi_s) else 50.0
    _, _, macd_hist = _macd(h1['c']); h1_atr = _atr(h1)
    vol_ratio = _volume_ratio_adjusted(h1); vol_zscore = _volume_zscore(h1['vol'])
    h4_rsi_s = _dr_rsi_series(h4['c'])
    _m3ph = _dr_m3_default_result('背离未确认')
    dw = int(cfg.get('divergence_window', 48)); ms = int(cfg.get('min_pivot_separation', 8))
    rt = h1_atr * float(cfg.get('recover_atr_multiple', 0.3))
    bull_div, bull_meta = _dr_bullish_divergence(h1, h1_rsi_s, macd_hist, window=dw, min_sep=ms, recover_threshold=rt)
    bear_div, bear_meta = _dr_bearish_divergence(h1, h1_rsi_s, macd_hist, window=dw, min_sep=ms, recover_threshold=rt)
    max_h4_rsi_buy = float(cfg.get('max_h4_rsi_for_buy', 42)); min_h4_rsi_sell = float(cfg.get('min_h4_rsi_for_sell', 58))
    h4_exh_bull = h4_rsi <= max_h4_rsi_buy; h4_exh_bear = h4_rsi >= min_h4_rsi_sell
    is_bull = bull_div and h4_exh_bull; is_bear = bear_div and h4_exh_bear
    if is_bull:
        rsi_gap = bull_meta.get('rsi_gap', 2.5)
        ds = _DR_W_DIVERGENCE * min(1.0, 0.65 + (rsi_gap - 2.5) / 15.0 * 0.35); score += ds
        signals.append(f"1H底背离+4H弱末端(RSI背离{rsi_gap:.1f}→+{ds:.0f}分)")
    elif is_bear:
        rsi_gap = bear_meta.get('rsi_gap', 2.5)
        ds = _DR_W_DIVERGENCE * min(1.0, 0.65 + (rsi_gap - 2.5) / 15.0 * 0.35); score += ds
        signals.append(f"1H顶背离+4H强末端(RSI背离{rsi_gap:.1f}→+{ds:.0f}分)")
    elif bull_div and not h4_exh_bull:
        signals.append(f"1H底背离但4H RSI偏高({h4_rsi:.1f})")
        return _dr_build_result(valid=True, score=score, direction='WAIT', signals=signals,
            h4_rsi=h4_rsi, h1_rsi=h1_rsi, vol_ratio=vol_ratio, vol_zscore=vol_zscore,
            reversal_range_pct=0.0, bull_div=bull_div, bear_div=bear_div, m3_confirm=_m3ph)
    elif bear_div and not h4_exh_bear:
        signals.append(f"1H顶背离但4H RSI偏低({h4_rsi:.1f})")
        return _dr_build_result(valid=True, score=score, direction='WAIT', signals=signals,
            h4_rsi=h4_rsi, h1_rsi=h1_rsi, vol_ratio=vol_ratio, vol_zscore=vol_zscore,
            reversal_range_pct=0.0, bull_div=bull_div, bear_div=bear_div, m3_confirm=_m3ph)
    else:
        signals.append("未检测到有效背离结构")
        return _dr_build_result(valid=True, score=score, direction='WAIT', signals=signals,
            h4_rsi=h4_rsi, h1_rsi=h1_rsi, vol_ratio=vol_ratio, vol_zscore=vol_zscore,
            reversal_range_pct=0.0, bull_div=False, bear_div=False, m3_confirm=_m3ph)
    meta = bull_meta if is_bull else bear_meta
    p2_dist = dw - meta.get('p2', dw)
    if p2_dist <= 8: fs = _DR_W_FRESHNESS; signals.append(f"背离新鲜(p2距今{p2_dist}根 → +{fs}分)")
    elif p2_dist <= 16: fs = _DR_W_FRESHNESS * 0.5; signals.append(f"背离较新(p2距今{p2_dist}根 → +{fs:.0f}分)")
    else: fs = 0.0; signals.append(f"背离较旧(p2距今{p2_dist}根)")
    score += fs
    macd_crossed = False
    if len(macd_hist) >= 3:
        mh = macd_hist.iloc[-3:].values
        if is_bull: macd_crossed = mh[-2] < 0 and mh[-1] > 0
        else: macd_crossed = mh[-2] > 0 and mh[-1] < 0
    if macd_crossed:
        score += _DR_W_MACD_CROSS; signals.append(f"MACD直方图{'正穿零' if is_bull else '负穿零'}确认(+{_DR_W_MACD_CROSS}分)")
    elif len(macd_hist) >= 2:
        if (is_bull and float(macd_hist.iloc[-1]) > float(macd_hist.iloc[-2])) or \
           (is_bear and float(macd_hist.iloc[-1]) < float(macd_hist.iloc[-2])):
            mc = _DR_W_MACD_CROSS * 0.4; score += mc; signals.append(f"MACD方向趋势正确(+{mc:.0f}分)")
    last_c = float(h1['c'].iloc[-1]); last_o = float(h1['o'].iloc[-1])
    last_h = float(h1['h'].iloc[-1]); last_l = float(h1['l'].iloc[-1])
    prev_c = float(h1['c'].iloc[-2]); prev_o = float(h1['o'].iloc[-2])
    bar_body = abs(last_c - last_o); bar_range = last_h - last_l
    body_atr_ratio = bar_body / h1_atr if h1_atr > 0 else 0.0
    candle_bull_ok = is_bull and last_c > last_o; candle_bear_ok = is_bear and last_c < last_o
    if candle_bull_ok or candle_bear_ok:
        base_candle = _DR_W_CANDLE * 0.65
        strength = min(1.0, max(0.0, body_atr_ratio - 0.3) / 0.7)
        engulf = (last_c > prev_o if candle_bull_ok else last_c < prev_o)
        quality_bonus = _DR_W_CANDLE * 0.35 * (strength * 0.6 + (1.0 if engulf else 0.0) * 0.4)
        cs = base_candle + quality_bonus; score += cs
        label = "阳" if candle_bull_ok else "阴"
        signals.append(f"1H反转{label}线(实体/ATR={body_atr_ratio:.2f},{'吞没' if engulf else '未吞没'} → +{cs:.1f}分)")
    else:
        signals.append("1H K线方向未确认")
    min_vr = float(cfg.get('min_volume_ratio', 1.15))
    if vol_ratio >= min_vr:
        vs = _DR_W_VOLUME * min(1.0, 0.6 + (vol_ratio - min_vr) / max(min_vr, 0.1) * 0.3); score += vs
        signals.append(f"量能确认({vol_ratio:.2f}x → +{vs:.0f}分)")
    else:
        signals.append(f"量能不足({vol_ratio:.2f}x)")
    if vol_zscore >= 0.8:
        zs = _DR_W_VOL_ZSCORE * min(1.0, vol_zscore / 2.0); score += zs
        signals.append(f"放量显著(z={vol_zscore:+.2f} → +{zs:.1f}分)")
    elif vol_zscore >= 0.0:
        score += _DR_W_VOL_ZSCORE * 0.3
    rsi_pos_bull = is_bull and h1_rsi <= 52; rsi_pos_bear = is_bear and h1_rsi >= 48
    if rsi_pos_bull or rsi_pos_bear:
        score += _DR_W_RSI_POS; signals.append(f"RSI反转位置合理({h1_rsi:.1f} → +{_DR_W_RSI_POS}分)")
    rrp = _dr_reversal_range_pct(h1, price, is_bull=is_bull)
    max_range = float(cfg.get('max_reversal_range_pct', 4.0))
    if rrp <= max_range:
        ls = _DR_W_LOCATION * max(0.0, 1.0 - rrp / max(max_range, 0.1))
        if ls >= _DR_W_LOCATION * 0.3: score += ls; signals.append(f"反转位置合理({rrp:.2f}% → +{ls:.1f}分)")
    else:
        signals.append(f"反转已偏离({rrp:.2f}%)")
    h4_div_ok = False
    if len(h4_rsi_s) >= 30 and len(h4) >= 30:
        h4dw = min(30, len(h4))
        h4_lows = h4['l'].tail(h4dw).reset_index(drop=True)
        h4_highs = h4['h'].tail(h4dw).reset_index(drop=True)
        h4_rsi_w = h4_rsi_s.tail(h4dw).reset_index(drop=True)
        if is_bull:
            pvs = _dr_find_pivots(h4_lows, find_min=True, window=h4dw, min_sep=5)
            if len(pvs) >= 2:
                pp1, pp2 = pvs[0], pvs[1]
                if float(h4_lows.iloc[pp2]) < float(h4_lows.iloc[pp1]) and \
                   float(h4_rsi_w.iloc[pp2]) > float(h4_rsi_w.iloc[pp1]): h4_div_ok = True
        else:
            pvs = _dr_find_pivots(h4_highs, find_min=False, window=h4dw, min_sep=5)
            if len(pvs) >= 2:
                pp1, pp2 = pvs[0], pvs[1]
                if float(h4_highs.iloc[pp2]) > float(h4_highs.iloc[pp1]) and \
                   float(h4_rsi_w.iloc[pp2]) < float(h4_rsi_w.iloc[pp1]): h4_div_ok = True
    if h4_div_ok: score += _DR_W_H4_DIV; signals.append(f"4H RSI也呈背离(+{_DR_W_H4_DIV}分)")
    if candle_bull_ok or candle_bear_ok:
        m3_confirm = (_dr_three_min_reversal_core(m3, bull_bias=is_bull, cfg=cfg)
                      if len(m3) >= min_3m_bars else _dr_m3_default_result('3m数据不足'))
    else:
        m3_confirm = _dr_m3_default_result('K线未确认，跳过3m')
    if m3_confirm.get('passed'): score += _DR_W_3M; signals.append(str(m3_confirm.get('reason')))
    elif m3_confirm.get('valid') and (candle_bull_ok or candle_bear_ok):
        signals.append(f"3m观察(Phase {m3_confirm.get('phase', 0)})：{m3_confirm.get('reason')}")
    m3_ok = bool(m3_confirm.get('passed')) or not require_3m; vol_ok = vol_ratio >= min_vr
    direction = 'WAIT'
    if candle_bull_ok and m3_ok and vol_ok: direction = 'BUY'
    elif candle_bear_ok and m3_ok and vol_ok: direction = 'SELL'
    return _dr_build_result(valid=True, score=score, direction=direction, signals=signals,
        h4_rsi=h4_rsi, h1_rsi=h1_rsi, vol_ratio=vol_ratio, vol_zscore=vol_zscore,
        reversal_range_pct=rrp, bull_div=is_bull, bear_div=is_bear, m3_confirm=m3_confirm)


class DivergenceReversalScanner(BaseScannerStrategy if _HAS_SCANNER_BASE else object):
    required_bars = ['4H', '1H', '3m']
    name = "背离反转扫描"
    description = "4H 衰竭 + 1H RSI/MACD 背离 + 反转 K 线 + 3m 三段式确认"
    strategy_type = "scan"

    def __init__(self, config=None):
        self.config = {**_DR_DEFAULT_CONFIG, **(config or {})}
        if _HAS_SCANNER_BASE and hasattr(super(), '__init__'):
            try: super().__init__(config or {})
            except Exception: pass

    def _init_conditions(self):
        if ScanCondition is None: return
        self.add_condition(ScanCondition(name="24H成交量", description="过滤流动性不足标的",
            field="volume_24h", operator=">=", value=self.config.get('min_volume_24h', 12_000_000)))

    def scan_symbol(self, symbol) -> Dict:
        km = symbol.extra_data.get('klines', {})
        try:
            h4 = _to_df(self._get_klines(km, '4H')); h1 = _to_df(self._get_klines(km, '1H'))
            m3 = _to_df(self._get_klines(km, '3m'))
            analysis = _dr_analyze_core(h4, h1, m3, getattr(symbol, 'last_price', 0.0), self.config)
        except Exception as exc:
            return {'symbol': getattr(symbol, 'inst_id', ''), 'passed': False, 'score': 0.0, 'direction': 'WAIT', 'details': {'状态': f'分析异常: {exc}'}}
        if not analysis['valid']:
            return {'symbol': getattr(symbol, 'inst_id', ''), 'passed': False, 'score': 0.0, 'direction': 'WAIT', 'details': {'状态': analysis.get('reason', '')}}
        ms = float(self.config.get('min_score', 65))
        passed = analysis['score'] >= ms and analysis['direction'] in {'BUY', 'SELL'}
        result = {'symbol': getattr(symbol, 'inst_id', ''), 'passed': passed, 'score': round(analysis['score'], 2),
                  'direction': analysis['direction'], 'signals': analysis['signals'], 'details': analysis['details'],
                  'last_price': getattr(symbol, 'last_price', 0.0), 'volume_24h': getattr(symbol, 'volume_24h', 0.0),
                  'price_change_24h': getattr(symbol, 'price_change_24h', 0.0), 'category': '背离反转',
                  'ranking_factors': analysis.get('ranking_factors', {})}
        if build_opportunity_profile:
            try: result.update(build_opportunity_profile(base_score=analysis['score'], direction=analysis['direction'],
                volume_24h=getattr(symbol, 'volume_24h', 0.0), factors=analysis.get('ranking_factors', {}), signals=analysis['signals']))
            except Exception: pass
        return result

    def _get_klines(self, km, bar):
        return km.get(bar) or km.get(bar.lower()) or km.get(bar.upper()) or []

    def get_config_schema(self):
        return {k: {'type': 'float', 'default': v, 'label': k} for k, v in _DR_DEFAULT_CONFIG.items()}


# ══════════════════════════════════════════════════════════════════════════════
# 第六节：波段趋势回踩扫描 v3（TrendPullbackSwingScanner）
# ══════════════════════════════════════════════════════════════════════════════

_TPB_W_TREND       = 30
_TPB_W_PULLBACK    = 20
_TPB_W_RETEST_TIME =  5
_TPB_W_KEYLEVEL    =  8
_TPB_W_CONFIRM     = 17
_TPB_W_VOLUME      = 12
_TPB_W_RSI         =  8

_TPB_ADX_MIN_TREND     = 18.0
_TPB_EMA_SPREAD_MIN    = 0.20
_TPB_RETEST_LOOKBACK   = 12
_TPB_RETEST_MIN_TOUCHES = 2

_TPB_DEFAULT_CONFIG = {
    'min_score':                68,
    'min_volume_24h':           12_000_000,
    'max_pullback_pct':         3.2,
    'max_pullback_atr':         1.5,
    'min_confirm_volume_ratio': 1.3,
    'min_volume_zscore':        0.8,
    'require_3m_continuation':  True,
    'm3_window':                60,
    'm3_neckline_bars':         8,
    'm3_min_pullback_bars':     3,
    'm3_breakout_buffer_pct':   0.10,
    'm3_origin_tolerance_pct':  0.15,
}


class TrendPullbackSwingScanner(_BASE_SCANNER_CLASS):
    required_bars = ['1D', '4H', '1H', '3m']
    name        = "波段趋势回踩扫描"
    description = "1D/4H 波段趋势 → ATR 归一化回踩 → 1H 穿越确认 → 3m 三段式结构"
    strategy_type = "scan"

    def __init__(self, config=None):
        self.config = {**_TPB_DEFAULT_CONFIG, **(config or {})}
        if _HAS_SCANNER_BASE and hasattr(super(), '__init__'):
            try:
                super().__init__(config or {})
            except Exception:
                pass

    def _init_conditions(self):
        if ScanCondition is None:
            return
        self.add_condition(ScanCondition(
            name="24H成交量", description="过滤流动性不足标的",
            field="volume_24h", operator=">=",
            value=self.config.get('min_volume_24h', 12_000_000),
        ))

    def scan_symbol(self, symbol) -> dict:
        klines_map = symbol.extra_data.get('klines', {})
        try:
            d1 = _to_df(self._get_klines(klines_map, '1D'))
            h4 = _to_df(self._get_klines(klines_map, '4H'))
            h1 = _to_df(self._get_klines(klines_map, '1H'))
            m3 = _to_df(self._get_klines(klines_map, '3m'))
            analysis = _tpb_analyze_core(d1, h4, h1, m3, getattr(symbol, 'last_price', 0.0), self.config)
        except Exception as exc:
            return {'symbol': getattr(symbol, 'inst_id', ''), 'passed': False, 'score': 0.0,
                    'direction': 'WAIT', 'details': {'状态': f'分析异常: {exc}'}}
        if not analysis['valid']:
            return {'symbol': getattr(symbol, 'inst_id', ''), 'passed': False, 'score': 0.0,
                    'direction': 'WAIT', 'details': {'状态': analysis.get('reason', '')}}
        min_score = float(self.config.get('min_score', 68))
        passed = analysis['score'] >= min_score and analysis['direction'] in {'BUY', 'SELL'}
        result = {
            'symbol': getattr(symbol, 'inst_id', ''), 'passed': passed,
            'score': round(analysis['score'], 2),
            'direction': analysis['direction'],
            'signals': analysis['signals'],
            'details': analysis['details'],
            'last_price': getattr(symbol, 'last_price', 0.0),
            'volume_24h': getattr(symbol, 'volume_24h', 0.0),
            'price_change_24h': getattr(symbol, 'price_change_24h', 0.0),
            'category': '波段趋势回踩',
            'ranking_factors': analysis.get('ranking_factors', {}),
        }
        if build_opportunity_profile:
            try:
                result.update(build_opportunity_profile(
                    base_score=analysis['score'], direction=analysis['direction'],
                    volume_24h=getattr(symbol, 'volume_24h', 0.0),
                    factors=analysis.get('ranking_factors', {}),
                    signals=analysis['signals']))
            except Exception:
                pass
        return result

    def _get_klines(self, klines_map, bar):
        return klines_map.get(bar) or klines_map.get(bar.lower()) or klines_map.get(bar.upper()) or []

    def get_config_schema(self):
        return {
            'min_score':                 {'type': 'int',   'default': 68,         'label': '最低通过分数(0-100)'},
            'min_volume_24h':            {'type': 'float', 'default': 12_000_000, 'label': '最小24H成交额'},
            'max_pullback_pct':          {'type': 'float', 'default': 3.2,        'label': '最大回踩距离%（价格）'},
            'max_pullback_atr':          {'type': 'float', 'default': 1.5,        'label': '最大回踩距离（ATR倍数）'},
            'min_confirm_volume_ratio':  {'type': 'float', 'default': 1.3,        'label': '确认最小量比'},
            'min_volume_zscore':         {'type': 'float', 'default': 0.8,        'label': '启动量 z-score 下限'},
            'require_3m_continuation':   {'type': 'bool',  'default': True,       'label': '要求3m三段式趋势延续确认'},
            'm3_window':                 {'type': 'int',   'default': 60,         'label': '3m观察窗口(根数)'},
            'm3_neckline_bars':          {'type': 'int',   'default': 8,          'label': '3m颈线样本数'},
            'm3_min_pullback_bars':      {'type': 'int',   'default': 3,          'label': '3m最少回踩根数'},
            'm3_breakout_buffer_pct':    {'type': 'float', 'default': 0.10,       'label': '3m突破缓冲%'},
            'm3_origin_tolerance_pct':   {'type': 'float', 'default': 0.15,       'label': '3m起点容忍偏离%'},
        }


def _tpb_analyze_core(d1, h4, h1, m3, last_price, cfg):
    _check_df(d1, '日线', 90)
    _check_df(h4, '4H', 90)
    _check_df(h1, '1H', 120)
    require_3m = bool(cfg.get('require_3m_continuation', True))
    min_3m_bars = max(int(cfg.get('m3_window', 60)),
                      int(cfg.get('m3_neckline_bars', 8)) + int(cfg.get('m3_min_pullback_bars', 3)) + 6)
    if require_3m and len(m3) < min_3m_bars:
        require_3m = False  # 3m 数据缺失时降级
    price = float(last_price) if last_price and last_price > 0 else float(h1['c'].iloc[-1])
    score = 0.0; signals = []
    d1_ema21 = _ema(d1['c'], 21); d1_ema55 = _ema(d1['c'], 55)
    h4_ema21 = _ema(h4['c'], 21); h4_ema55 = _ema(h4['c'], 55)
    h1_ema21 = _ema(h1['c'], 21); h1_ema55 = _ema(h1['c'], 55)
    h1_last_close = float(h1['c'].iloc[-1]); h1_prev_close = float(h1['c'].iloc[-2])
    h1_rsi     = _rsi_wilder(h1['c'])
    h4_atr     = _atr(h4)
    vol_ratio  = _volume_ratio_adjusted(h1)
    vol_zscore = _volume_zscore(h1['vol'])
    d1_adx     = _adx(d1, 14)
    h4_adx     = _adx(h4, 14)
    d1_ema_spread = (d1_ema21 - d1_ema55) / d1_ema55 * 100 if d1_ema55 > 0 else 0.0
    h4_ema_spread = (h4_ema21 - h4_ema55) / h4_ema55 * 100 if h4_ema55 > 0 else 0.0
    trend_snap = _local_trend_snapshot(d1, h4, h1, price)
    trend_metrics = trend_snap.get('metrics', {})
    trend_long_score = float(trend_snap.get('long_score', 0) or 0)
    trend_short_score = float(trend_snap.get('short_score', 0) or 0)
    _m3ph = _tpb_m3_default('趋势未确认')
    bullish = (
        bool(trend_snap.get('long_ok'))
        and price > d1_ema21 > d1_ema55
        and price > h4_ema21 > h4_ema55
        and d1_adx >= _TPB_ADX_MIN_TREND and h4_adx >= _TPB_ADX_MIN_TREND
        and d1_ema_spread >= _TPB_EMA_SPREAD_MIN and h4_ema_spread >= _TPB_EMA_SPREAD_MIN
    )
    bearish = (
        bool(trend_snap.get('short_ok'))
        and price < d1_ema21 < d1_ema55
        and price < h4_ema21 < h4_ema55
        and d1_adx >= _TPB_ADX_MIN_TREND and h4_adx >= _TPB_ADX_MIN_TREND
        and d1_ema_spread <= -_TPB_EMA_SPREAD_MIN and h4_ema_spread <= -_TPB_EMA_SPREAD_MIN
    )
    # ① 趋势（30 分）
    if bullish or bearish:
        spread_abs = (abs(d1_ema_spread) + abs(h4_ema_spread)) / 2.0
        spread_bonus = min(4.0, spread_abs / 2.0 * 4.0)
        adx_avg = (d1_adx + h4_adx) / 2.0
        adx_bonus = min(4.0, max(0.0, (adx_avg - _TPB_ADX_MIN_TREND) / 12.0 * 4.0))
        ts_ = 22.0 + spread_bonus + adx_bonus; score += ts_
        dl = "多头" if bullish else "空头"
        rs = trend_long_score if bullish else trend_short_score
        signals.append(f"{dl}趋势通过(质量{rs:.0f}, ADX {adx_avg:.1f}, 发散{spread_abs:.2f}% → +{ts_:.1f}分)")
    else:
        rp = []
        if d1_adx < _TPB_ADX_MIN_TREND: rp.append(f"日线ADX不足({d1_adx:.1f})")
        if h4_adx < _TPB_ADX_MIN_TREND: rp.append(f"4H ADX不足({h4_adx:.1f})")
        if abs(d1_ema_spread) < _TPB_EMA_SPREAD_MIN: rp.append(f"日线EMA发散不足({d1_ema_spread:.2f}%)")
        if abs(h4_ema_spread) < _TPB_EMA_SPREAD_MIN: rp.append(f"4H EMA发散不足({h4_ema_spread:.2f}%)")
        signals.append("趋势未确认: " + " | ".join(rp) if rp else "趋势未确认")
        return _tpb_build_result(valid=True, score=score, direction='WAIT', signals=signals,
            pullback_pct=0.0, pb_atr=0.0, vol_ratio=vol_ratio, vol_zscore=vol_zscore,
            close_strength=0.0, h1_rsi=h1_rsi, d1_adx=d1_adx, h4_adx=h4_adx,
            d1_ema_spread=d1_ema_spread, h4_ema_spread=h4_ema_spread,
            trend_snap=trend_snap, trend_metrics=trend_metrics, m3_result=_m3ph)
    # ② 方向性回踩（20 分）
    raw_distance = (price - h4_ema21) / h4_ema21 * 100 if h4_ema21 > 0 else 999.0
    pb_atr = abs(price - h4_ema21) / h4_atr if h4_atr > 0 else 999.0
    max_pb = float(cfg.get('max_pullback_pct', 3.2)); max_pb_atr = float(cfg.get('max_pullback_atr', 1.5))
    if bullish:
        pullback_ok = 0.0 <= raw_distance <= max_pb and pb_atr <= max_pb_atr
        pullback_depth = abs(raw_distance); direction_mismatch = raw_distance < 0
    else:
        pullback_ok = -max_pb <= raw_distance <= 0.0 and pb_atr <= max_pb_atr
        pullback_depth = abs(raw_distance); direction_mismatch = raw_distance > 0
    if pullback_ok:
        pb_score = _TPB_W_PULLBACK * (1 - 0.65 * max(pullback_depth / max_pb, pb_atr / max_pb_atr))
        pb_score = round(max(pb_score, 5.0), 1); score += pb_score
        signals.append(f"方向性回踩({pullback_depth:.2f}%, {pb_atr:.2f}ATR → +{pb_score:.1f}分)")
    elif direction_mismatch:
        signals.append(f"价格已穿越EMA21至反向侧({raw_distance:+.2f}%)")
    else:
        signals.append(f"回踩过远({pullback_depth:.2f}%, {pb_atr:.2f}ATR)")
    # ③ 回踩时间真实性（5 分）
    h1_ema21_series = h1['c'].ewm(span=21, adjust=False).mean()
    rw = h1.tail(_TPB_RETEST_LOOKBACK); ew = h1_ema21_series.tail(_TPB_RETEST_LOOKBACK)
    touch_count = int((rw['l'].values <= ew.values).sum()) if bullish else int((rw['h'].values >= ew.values).sum())
    if touch_count >= _TPB_RETEST_MIN_TOUCHES:
        rt = _TPB_W_RETEST_TIME * min(1.0, touch_count / 5.0); score += rt
        signals.append(f"回踩真实(近{_TPB_RETEST_LOOKBACK}根触碰{touch_count}根 → +{rt:.1f}分)")
    else:
        signals.append(f"回踩过程不足(仅{touch_count}根触及)")
    # ④ 4H swing 关键位 bonus（8 分）
    sh, sl = _latest_swing_levels(h4, left=5, right=5, skip_recent=3, max_lookback=80)
    retest_quality = 0.0; max_kl = 2.2
    if bullish and sh and sh > 0 and sh < price:
        retest_quality = (price - sh) / sh * 100
        if retest_quality <= max_kl:
            kls = _TPB_W_KEYLEVEL * max(0.5, 1.0 - retest_quality / max_kl)
            score += kls; signals.append(f"回踩前swing高({retest_quality:.2f}% → +{kls:.1f}分)")
    elif bearish and sl and sl > 0 and sl > price:
        retest_quality = (sl - price) / sl * 100
        if retest_quality <= max_kl:
            kls = _TPB_W_KEYLEVEL * max(0.5, 1.0 - retest_quality / max_kl)
            score += kls; signals.append(f"反抽前swing低({retest_quality:.2f}% → +{kls:.1f}分)")
    # ⑤ 1H 穿越确认 + 收盘强度（17 分）
    confirm_bull = (bullish and h1_prev_close <= h1_ema21 and h1_last_close > h1_ema21 and h1_ema21 > h1_ema55)
    confirm_bear = (bearish and h1_prev_close >= h1_ema21 and h1_last_close < h1_ema21 and h1_ema21 < h1_ema55)
    last_h1 = h1.iloc[-1]; bar_range = float(last_h1['h']) - float(last_h1['l'])
    if bar_range > 0:
        close_strength = ((float(last_h1['c']) - float(last_h1['l'])) / bar_range if bullish
                          else (float(last_h1['h']) - float(last_h1['c'])) / bar_range)
    else:
        close_strength = 0.5
    if confirm_bull or confirm_bear:
        base_c = _TPB_W_CONFIRM * 0.7
        strength_bonus = _TPB_W_CONFIRM * 0.3 * max(0.0, (close_strength - 0.5) / 0.5)
        cs = base_c + strength_bonus; score += cs
        arrow = "站上" if confirm_bull else "跌破"
        signals.append(f"1H收盘{arrow}EMA21(强度{close_strength:.2f} → +{cs:.1f}分)")
    else:
        signals.append("1H穿越未触发")
    # ⑥ 量能（12 分）
    min_vr = float(cfg.get('min_confirm_volume_ratio', 1.3)); min_zs = float(cfg.get('min_volume_zscore', 0.8))
    vrok = vol_ratio >= min_vr; vzok = vol_zscore >= min_zs
    if vrok and vzok:
        rc = 0.55 + min(1.0, (vol_ratio - min_vr) / max(min_vr, 0.1) * 0.25) * 0.45
        zc = 0.55 + min(1.0, (vol_zscore - min_zs) / 1.5) * 0.45
        vs = _TPB_W_VOLUME * (rc + zc) / 2.0; score += vs
        signals.append(f"量能强({vol_ratio:.2f}x, z={vol_zscore:+.2f} → +{vs:.1f}分)")
    elif vrok or vzok:
        vs = _TPB_W_VOLUME * 0.45; score += vs
        signals.append(f"量能部分({vol_ratio:.2f}x, z={vol_zscore:+.2f} → +{vs:.1f}分)")
    else:
        signals.append(f"量能不足({vol_ratio:.2f}x, z={vol_zscore:+.2f})")
    # ⑦ RSI 健康度 bonus（8 分）
    rsi_ok = (bullish and 45 <= h1_rsi <= 68) or (bearish and 32 <= h1_rsi <= 55)
    if rsi_ok:
        score += _TPB_W_RSI; signals.append(f"RSI健康({'多' if bullish else '空'}头区间, {h1_rsi:.1f})")
    elif (bullish and h1_rsi > 75) or (bearish and h1_rsi < 25):
        signals.append(f"RSI警示({'超买' if bullish else '超卖'}: {h1_rsi:.1f})")
    # ⑧ 3m 三段式结构
    if confirm_bull or confirm_bear:
        m3_result = (_tpb_three_min_continuation(m3, bullish=bullish, cfg=cfg)
                     if len(m3) >= min_3m_bars else _tpb_m3_default('3m数据不足'))
    else:
        m3_result = _tpb_m3_default('1H穿越未触发，跳过3m')
    if m3_result.get('passed'):
        signals.append(str(m3_result.get('reason')))
    elif m3_result.get('valid') and (confirm_bull or confirm_bear):
        signals.append(f"3m观察(Phase {m3_result.get('phase', 0)})：{m3_result.get('reason')}")
    m3_ok = bool(m3_result.get('passed')) or not require_3m
    direction = 'WAIT'
    if confirm_bull and pullback_ok and m3_ok: direction = 'BUY'
    elif confirm_bear and pullback_ok and m3_ok: direction = 'SELL'
    return _tpb_build_result(valid=True, score=score, direction=direction, signals=signals,
        pullback_pct=pullback_depth if pullback_ok or direction_mismatch else abs(raw_distance),
        pb_atr=pb_atr, vol_ratio=vol_ratio, vol_zscore=vol_zscore, close_strength=close_strength,
        h1_rsi=h1_rsi, d1_adx=d1_adx, h4_adx=h4_adx,
        d1_ema_spread=d1_ema_spread, h4_ema_spread=h4_ema_spread,
        trend_snap=trend_snap, trend_metrics=trend_metrics, m3_result=m3_result)


def _tpb_three_min_continuation(df, *, bullish, cfg):
    window = int(cfg.get('m3_window', 60)); neckline_bars = int(cfg.get('m3_neckline_bars', 8))
    origin_tol = float(cfg.get('m3_origin_tolerance_pct', 0.15)); breakout_buf = float(cfg.get('m3_breakout_buffer_pct', 0.10))
    min_pb_bars = int(cfg.get('m3_min_pullback_bars', 3)); min_bars = neckline_bars + min_pb_bars + 6
    if len(df) < max(window, min_bars): return _tpb_m3_default(f'3m数据不足({len(df)}/{max(window, min_bars)})')
    recent = df.tail(window).reset_index(drop=True); n = len(recent)
    ose = max(neckline_bars + min_pb_bars + 4, n // 2)
    if bullish: op = int(recent.iloc[:ose]['l'].idxmin()); opr = float(recent['l'].iloc[op])
    else: op = int(recent.iloc[:ose]['h'].idxmax()); opr = float(recent['h'].iloc[op])
    ns_ = op + 1; ne = min(ns_ + neckline_bars, n - min_pb_bars - 3)
    if ne <= ns_ + 1: return _tpb_m3_default('3m颈线样本不足', origin_price=opr)
    neck = recent.iloc[ns_:ne]
    if bullish:
        bl = float(neck['h'].max())
        if not pd.notna(bl) or bl <= 0: return _tpb_m3_default('3m颈线异常', origin_price=opr)
        tl = bl * (1.0 + breakout_buf / 100.0); fl = opr * (1.0 - origin_tol / 100.0)
    else:
        bl = float(neck['l'].min())
        if not pd.notna(bl) or bl <= 0: return _tpb_m3_default('3m颈线异常(空)', origin_price=opr)
        tl = bl * (1.0 - breakout_buf / 100.0); fl = opr * (1.0 + origin_tol / 100.0)
    phase = 0; p1e = opr; cl_ = tl; pbe = float('inf') if bullish else float('-inf'); pbc = 0; p3c = 0.0
    for i in range(ne, n):
        bc = float(recent['c'].iloc[i]); bh = float(recent['h'].iloc[i]); bll = float(recent['l'].iloc[i])
        if bullish:
            if phase == 0:
                if bc > tl: phase = 1; p1e = bh; cl_ = bh
            elif phase == 1:
                if bc > tl:
                    if bh > p1e: p1e = bh; cl_ = bh
                else:
                    if bc < fl: return {'valid': True, 'passed': False, 'quality': 15.0, 'reason': 'Phase2收盘跌破起点', 'phase': 2, 'origin_price': opr, 'breakout_level': bl, 'phase1_extreme': p1e, 'pb_extreme': bc, 'confirmation_level': cl_, 'phase3_close': 0.0, 'pb_depth_pct': (bc - opr) / opr * 100, 'continuation_pct': 0.0}
                    phase = 2; pbe = bll; pbc = 1
            elif phase == 2:
                pbe = min(pbe, bll); pbc += 1
                if bc < fl: return {'valid': True, 'passed': False, 'quality': 15.0, 'reason': 'Phase2收盘跌破起点', 'phase': 2, 'origin_price': opr, 'breakout_level': bl, 'phase1_extreme': p1e, 'pb_extreme': pbe, 'confirmation_level': cl_, 'phase3_close': 0.0, 'pb_depth_pct': (pbe - opr) / opr * 100, 'continuation_pct': 0.0}
                if pbc >= min_pb_bars and bc > cl_: phase = 3; p3c = bc; break
        else:
            if phase == 0:
                if bc < tl: phase = 1; p1e = bll; cl_ = bll
            elif phase == 1:
                if bc < tl:
                    if bll < p1e: p1e = bll; cl_ = bll
                else:
                    if bc > fl: return {'valid': True, 'passed': False, 'quality': 15.0, 'reason': 'Phase2收盘超起点', 'phase': 2, 'origin_price': opr, 'breakout_level': bl, 'phase1_extreme': p1e, 'pb_extreme': bc, 'confirmation_level': cl_, 'phase3_close': 0.0, 'pb_depth_pct': (opr - bc) / opr * 100, 'continuation_pct': 0.0}
                    phase = 2; pbe = bh; pbc = 1
            elif phase == 2:
                pbe = max(pbe, bh); pbc += 1
                if bc > fl: return {'valid': True, 'passed': False, 'quality': 15.0, 'reason': 'Phase2收盘超起点', 'phase': 2, 'origin_price': opr, 'breakout_level': bl, 'phase1_extreme': p1e, 'pb_extreme': pbe, 'confirmation_level': cl_, 'phase3_close': 0.0, 'pb_depth_pct': (opr - pbe) / opr * 100, 'continuation_pct': 0.0}
                if pbc >= min_pb_bars and bc < cl_: phase = 3; p3c = bc; break
    lc = float(recent['c'].iloc[-1])
    if phase == 3:
        if bullish: dp = (pbe - opr) / opr * 100; cp = (p3c - cl_) / cl_ * 100; held = lc > p3c
        else: dp = (opr - pbe) / opr * 100; cp = (cl_ - p3c) / cl_ * 100; held = lc < p3c
        q = min(100.0, 60 + min(dp + 2, 4) * 5 + min(cp, 3) * 4 + (8 if held else 0))
        lb = "多头" if bullish else "空头"
        return {'valid': True, 'passed': True, 'quality': q, 'reason': f"3m{lb}三段延续：突破→回踩({dp:+.2f}%)→继续(+{cp:.2f}%)",
                'phase': 3, 'origin_price': opr, 'breakout_level': bl, 'phase1_extreme': p1e,
                'pb_extreme': pbe, 'confirmation_level': cl_, 'phase3_close': p3c, 'pb_depth_pct': dp, 'continuation_pct': cp}
    elif phase == 2:
        if bullish: dp = (pbe - opr) / opr * 100; fh = pbe >= fl
        else: dp = (opr - pbe) / opr * 100; fh = pbe <= fl
        q = 35 + 25 + (20 if fh else 0)
        r = f"3m已突破并回踩({dp:+.2f}%)，等待继续" if fh else f"3m回踩越过起点({dp:+.2f}%)"
        return {'valid': True, 'passed': False, 'quality': float(q), 'reason': r, 'phase': 2,
                'origin_price': opr, 'breakout_level': bl, 'phase1_extreme': p1e, 'pb_extreme': pbe,
                'confirmation_level': cl_, 'phase3_close': 0.0, 'pb_depth_pct': dp, 'continuation_pct': 0.0}
    elif phase == 1:
        lb = "多头" if bullish else "空头"
        return {'valid': True, 'passed': False, 'quality': 60.0, 'reason': f"3m{lb}已突破颈线，等回踩(极值={p1e:.6g})",
                'phase': 1, 'origin_price': opr, 'breakout_level': bl, 'phase1_extreme': p1e,
                'pb_extreme': float('nan'), 'confirmation_level': cl_, 'phase3_close': 0.0, 'pb_depth_pct': 0.0, 'continuation_pct': 0.0}
    else:
        return _tpb_m3_default(f'3m尚未完成初次突破', origin_price=opr)


def _tpb_m3_default(reason, origin_price=0.0):
    return {'valid': False, 'passed': False, 'quality': 35.0, 'reason': reason,
            'phase': 0, 'origin_price': origin_price, 'breakout_level': 0.0,
            'phase1_extreme': 0.0, 'pb_extreme': 0.0, 'confirmation_level': 0.0,
            'phase3_close': 0.0, 'pb_depth_pct': 0.0, 'continuation_pct': 0.0}


def _tpb_build_result(*, valid, score, direction, signals, pullback_pct, pb_atr,
                      vol_ratio, vol_zscore, close_strength, h1_rsi,
                      d1_adx, h4_adx, d1_ema_spread, h4_ema_spread,
                      trend_snap, trend_metrics, m3_result, reason=''):
    bull = direction == 'BUY'; bear = direction == 'SELL'
    tq = float(trend_snap.get('long_score' if (bull or not bear) else 'short_score', 25.0) or 25.0)
    m3p = bool(m3_result.get('passed'))
    lq = max(20.0, 100.0 - pullback_pct * 15.0)
    vq = min(vol_ratio / 1.3, 1.6) * 62.5
    fq = (96.0 if direction in {'BUY', 'SELL'} and m3p and pullback_pct <= 2.0
          else 88.0 if direction in {'BUY', 'SELL'} and m3p
          else 70.0 if direction in {'BUY', 'SELL'} else 35.0)
    return {
        'valid': valid, 'reason': reason, 'score': max(score, 0.0),
        'direction': direction, 'signals': signals,
        'ranking_factors': {
            'trend': tq, 'trigger': 90.0 if direction in {'BUY', 'SELL'} else 30.0,
            'volume': vq, 'location': lq, 'freshness': fq,
            'risk': 85.0 if 0.4 <= pullback_pct <= 3.2 else 55.0,
        },
        'details': {
            '评估': ' | '.join(signals) if signals else '暂无回踩确认机会',
            '回踩距离': f'{pullback_pct:.2f}%', 'ATR倍数': f'{pb_atr:.2f}',
            '量比': f'{vol_ratio:.2f}x', '量能Z分': f'{vol_zscore:+.2f}σ',
            '启动收盘强度': f'{close_strength:.2f}', '1H_RSI': f'{h1_rsi:.1f}',
            '日线ADX': f'{d1_adx:.1f}', '4H_ADX': f'{h4_adx:.1f}',
            '日线EMA发散': f'{d1_ema_spread:+.2f}%', '4H_EMA发散': f'{h4_ema_spread:+.2f}%',
            '趋势质量': trend_snap.get('reason', '-'),
            '3m结构确认': '通过' if m3p else '未通过',
            '3m当前阶段': f"Phase {m3_result.get('phase', 0)}",
            '3m结构说明': str(m3_result.get('reason', '-')),
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
# 第七节：波段超跌反转扫描 v3（OversoldReversalSwingScanner）
# ══════════════════════════════════════════════════════════════════════════════

_ORS_DEFAULT_CONFIG = {
    'min_score': 65, 'min_volume_24h': 10_000_000,
    'max_h4_rsi': 35, 'max_h1_rsi': 30,
    'min_recent_drop_pct': 12, 'min_reversal_volume_ratio': 1.5,
    'require_3m_reversal_retest': True,
    'm3_reversal_window': 80, 'm3_neckline_bars': 10,
    'm3_min_pullback_bars': 3, 'm3_breakout_buffer_pct': 0.12,
    'm3_origin_tolerance_pct': 0.18,
    'trend_rsi_min': 40, 'trend_rsi_max': 58, 'trend_adx_min': 18,
}


class OversoldReversalSwingScanner(_BASE_SCANNER_CLASS):
    required_bars = ['4H', '1H', '3m']
    name = "波段超跌反转扫描"
    description = "超跌反弹 + 趋势初段双模式：极端超跌猜底 / RSI 恢复+EMA 站上+MACD 金叉 = 趋势确认"
    strategy_type = "scan"

    def __init__(self, config=None):
        self.config = {**_ORS_DEFAULT_CONFIG, **(config or {})}
        if _HAS_SCANNER_BASE and hasattr(super(), '__init__'):
            try: super().__init__(config or {})
            except Exception: pass

    def _init_conditions(self):
        if ScanCondition is None: return
        self.add_condition(ScanCondition(name="24H成交量", description="过滤流动性不足标的",
            field="volume_24h", operator=">=", value=self.config.get('min_volume_24h', 10_000_000)))

    def scan_symbol(self, symbol) -> dict:
        km = symbol.extra_data.get('klines', {})
        try:
            h4 = _to_df(self._get_klines(km, '4H'))
            h1 = _to_df(self._get_klines(km, '1H'))
            m3 = _to_df(self._get_klines(km, '3m'))
            analysis = _ors_analyze_core(h4, h1, m3, getattr(symbol, 'last_price', 0.0), self.config)
        except Exception as exc:
            return {'symbol': getattr(symbol, 'inst_id', ''), 'passed': False, 'score': 0.0,
                    'direction': 'WAIT', 'details': {'状态': f'分析异常: {exc}'}}
        if not analysis['valid']:
            return {'symbol': getattr(symbol, 'inst_id', ''), 'passed': False, 'score': 0.0,
                    'direction': 'WAIT', 'details': {'状态': analysis.get('reason', '')}}
        ms = float(self.config.get('min_score', 65))
        passed = analysis['score'] >= ms and analysis['direction'] == 'BUY'
        result = {
            'symbol': getattr(symbol, 'inst_id', ''), 'passed': passed,
            'score': round(analysis['score'], 2), 'direction': analysis['direction'],
            'signals': analysis['signals'], 'details': analysis['details'],
            'last_price': getattr(symbol, 'last_price', 0.0), 'volume_24h': getattr(symbol, 'volume_24h', 0.0),
            'price_change_24h': getattr(symbol, 'price_change_24h', 0.0),
            'category': '波段超跌反转', 'ranking_factors': analysis.get('ranking_factors', {}),
        }
        if build_opportunity_profile:
            try:
                result.update(build_opportunity_profile(
                    base_score=analysis['score'], direction=analysis['direction'],
                    volume_24h=getattr(symbol, 'volume_24h', 0.0),
                    factors=analysis.get('ranking_factors', {}), signals=analysis['signals']))
            except Exception: pass
        return result

    def _get_klines(self, km, bar):
        return km.get(bar) or km.get(bar.lower()) or km.get(bar.upper()) or []

    def get_config_schema(self):
        return {
            'min_score':                  {'type': 'int',   'default': 65,        'label': '最低通过分数'},
            'min_volume_24h':             {'type': 'float', 'default': 10_000_000,'label': '最小24H成交额'},
            'max_h4_rsi':                 {'type': 'float', 'default': 35,        'label': '超跌模式: 4H最大RSI'},
            'max_h1_rsi':                 {'type': 'float', 'default': 30,        'label': '超跌模式: 1H最大RSI'},
            'min_recent_drop_pct':        {'type': 'float', 'default': 12,        'label': '超跌模式: 近20根最小跌幅%'},
            'min_reversal_volume_ratio':  {'type': 'float', 'default': 1.5,       'label': '反转最小量比'},
            'trend_rsi_min':              {'type': 'float', 'default': 40,        'label': '趋势模式: RSI 下限'},
            'trend_rsi_max':              {'type': 'float', 'default': 58,        'label': '趋势模式: RSI 上限'},
            'trend_adx_min':              {'type': 'float', 'default': 18,        'label': '趋势模式: ADX 下限'},
            'require_3m_reversal_retest': {'type': 'bool',  'default': True,      'label': '要求3m三段式确认'},
        }


def _ors_analyze_core(h4, h1, m3, last_price, cfg):
    _check_df(h4, '4H', 90); _check_df(h1, '1H', 120)
    require_3m = bool(cfg.get('require_3m_reversal_retest', True))
    min_3m_bars = max(int(cfg.get('m3_reversal_window', 80)),
                      int(cfg.get('m3_neckline_bars', 10)) + int(cfg.get('m3_min_pullback_bars', 3)) + 6)
    if require_3m and len(m3) < min_3m_bars:
        require_3m = False  # 3m 数据缺失时降级
    price = float(last_price) if last_price and last_price > 0 else float(h1['c'].iloc[-1])
    h4_rsi = _rsi_wilder(h4['c']); h1_rsi = _rsi_wilder(h1['c'])
    bb_lower = _ors_bb_lower(h1['c'], 20, 2.0)
    vol_ratio = _volume_ratio_adjusted(h1); vol_zscore = _volume_zscore(h1['vol'])
    candle_quality, candle_ok = _ors_reversal_candle_quality(h1)
    h1_ema8 = _ema(h1['c'], 8); h1_ema21 = _ema(h1['c'], 21)
    _, _, macd_hist = _macd(h1['c']); macd_line, macd_sig, _ = _macd(h1['c'])
    h4_adx = _adx(h4, 14)
    recent_high_close = float(h4['c'].tail(20).max())
    recent_drop_pct = (recent_high_close - price) / recent_high_close * 100 if recent_high_close > 0 else 0.0
    m3_retest = (_ors_three_min_reversal(m3, cfg=cfg) if len(m3) >= min_3m_bars
                 else _ors_m3_default('3m数据不足'))
    score_a, signals_a, dir_a = _ors_score_oversold(
        h4_rsi, h1_rsi, bb_lower, price, recent_drop_pct, candle_quality, candle_ok,
        vol_ratio, vol_zscore, macd_hist, h1_ema8, h1_ema21, None, m3_retest, require_3m, cfg)
    score_b, signals_b, dir_b = _ors_score_trend_init(
        h4_rsi, h1_rsi, h4_adx, price, h1_ema8, h1_ema21,
        macd_line, macd_sig, macd_hist, candle_quality, candle_ok,
        vol_ratio, vol_zscore, bb_lower, None, h1, m3_retest, require_3m, cfg)
    if score_a >= score_b:
        score, signals, direction, mode = score_a, signals_a, dir_a, "超跌反转"
    else:
        score, signals, direction, mode = score_b, signals_b, dir_b, "趋势初段"
    signals.insert(0, f"[{mode}模式]")
    m3p = bool(m3_retest.get('passed'))
    oq = min(100, max(20, (35 - h4_rsi) * 4 + 55)) if h4_rsi <= 35 else 40.0
    return {
        'valid': True, 'reason': '', 'score': max(score, 0.0), 'direction': direction, 'signals': signals,
        'ranking_factors': {
            'trend': oq if mode == "超跌反转" else min(100, max(40, h4_adx * 2.5)),
            'trigger': 90.0 if direction == 'BUY' else 28.0,
            'volume': min(vol_ratio / 1.5, 1.6) * 62.5,
            'location': 92.0 if h4_rsi <= 25 else 72.0 if h4_rsi <= 35 else 55.0,
            'freshness': 94.0 if direction == 'BUY' and m3p else 72.0 if direction == 'BUY' else 32.0,
            'risk': 86.0 if m3p else 55.0,
        },
        'details': {
            '评估': ' | '.join(signals) if signals else '暂无超跌反转机会', '模式': mode,
            '4H_RSI': f'{h4_rsi:.1f}', '1H_RSI': f'{h1_rsi:.1f}', '4H_ADX': f'{h4_adx:.1f}',
            '近期跌幅': f'{recent_drop_pct:.2f}%', '量比': f'{vol_ratio:.2f}x',
            '量能Z分': f'{vol_zscore:+.2f}σ', 'EMA8': f'{h1_ema8:.4f}', 'EMA21': f'{h1_ema21:.4f}',
            '3m结构确认': '通过' if m3p else '未通过',
            '3m当前阶段': f"Phase {m3_retest.get('phase', 0)}",
            '3m结构说明': str(m3_retest.get('reason', '-')),
        },
    }


def _ors_score_oversold(h4_rsi, h1_rsi, bb_lower, price, recent_drop_pct,
                        candle_quality, candle_ok, vol_ratio, vol_zscore,
                        macd_hist, h1_ema8, h1_ema21, h1_atr, m3_retest, require_3m, cfg):
    score = 0.0; signals = []
    max_h4_rsi = float(cfg.get('max_h4_rsi', 35)); max_h1_rsi = float(cfg.get('max_h1_rsi', 30))
    min_vr = float(cfg.get('min_reversal_volume_ratio', 1.5))
    h4_rsi_ok = h4_rsi <= max_h4_rsi
    if not h4_rsi_ok:
        signals.append(f"4H RSI未达超跌({h4_rsi:.1f}/{max_h4_rsi})"); return score, signals, 'WAIT'
    h4s = 20.0 * min(1.0, (max_h4_rsi - h4_rsi) / max(max_h4_rsi - 10, 1) + 0.5)
    score += h4s; signals.append(f"4H超跌({h4_rsi:.1f} → +{h4s:.0f}分)")
    if h1_rsi <= max_h1_rsi:
        h1s = 12.0 * min(1.0, (max_h1_rsi - h1_rsi) / max(max_h1_rsi - 10, 1) + 0.5)
        score += h1s; signals.append(f"1H深度超跌({h1_rsi:.1f} → +{h1s:.0f}分)")
    else:
        signals.append(f"1H RSI偏高({h1_rsi:.1f})")
    if price <= bb_lower * 1.005:
        bbs = 10.0 if price <= bb_lower else 5.5
        score += bbs; signals.append(f"{'跌破' if price<=bb_lower else '贴近'}布林下轨(+{bbs:.0f}分)")
    min_drop = float(cfg.get('min_recent_drop_pct', 12))
    if recent_drop_pct >= min_drop:
        ds = min(10.0, 10.0 * recent_drop_pct / max(min_drop, 1) * 0.75)
        score += ds; signals.append(f"跌幅充分({recent_drop_pct:.1f}% → +{ds:.0f}分)")
    drop_speed = recent_drop_pct / 20.0
    if drop_speed >= 2.0: score += 6.0; signals.append(f"急跌(速度{drop_speed:.1f}%/根 → +6分)")
    elif drop_speed >= 1.0: score += 3.0
    if candle_ok:
        cs = 14.0 * candle_quality; score += cs; signals.append(f"反转K线({candle_quality:.0%} → +{cs:.0f}分)")
    else:
        signals.append("未见反转K线")
    if vol_ratio >= min_vr:
        vs = min(10.0, 10.0 * vol_ratio / min_vr / 2.0); score += vs
        signals.append(f"反转量能({vol_ratio:.2f}x → +{vs:.0f}分)")
    if vol_zscore >= 0.8: score += min(6.0, 6.0 * vol_zscore / 2.0)
    if len(macd_hist) >= 2 and float(macd_hist.iloc[-1]) > float(macd_hist.iloc[-2]):
        score += 6.0; signals.append("MACD动量恢复(+6分)")
    if price > h1_ema8 and h1_ema8 > h1_ema21:
        score += 6.0; signals.append("已站上短期均线(+6分)")
    if m3_retest.get('passed'): score += 6.0; signals.append(str(m3_retest.get('reason')))
    elif m3_retest.get('valid'): signals.append(f"3m观察：{m3_retest.get('reason')}")
    m3_ok = bool(m3_retest.get('passed')) or not require_3m
    vol_ok = vol_ratio >= min_vr
    direction = 'BUY' if candle_ok and h4_rsi_ok and m3_ok and vol_ok else 'WAIT'
    return score, signals, direction


def _ors_score_trend_init(h4_rsi, h1_rsi, h4_adx, price, h1_ema8, h1_ema21,
                          macd_line, macd_sig, macd_hist, candle_quality, candle_ok,
                          vol_ratio, vol_zscore, bb_lower, h1_atr, h1,
                          m3_retest, require_3m, cfg):
    score = 0.0; signals = []
    trend_rsi_min = float(cfg.get('trend_rsi_min', 40)); trend_rsi_max = float(cfg.get('trend_rsi_max', 58))
    trend_adx_min = float(cfg.get('trend_adx_min', 18)); min_vr = float(cfg.get('min_reversal_volume_ratio', 1.5))
    rsi_recovery = trend_rsi_min <= h4_rsi <= trend_rsi_max
    if not rsi_recovery:
        signals.append(f"RSI不在恢复区间({h4_rsi:.1f}，需{trend_rsi_min}-{trend_rsi_max})")
        return score, signals, 'WAIT'
    ema_trend = price > h1_ema8 and h1_ema8 > h1_ema21
    if ema_trend:
        spread = (h1_ema8 - h1_ema21) / h1_ema21 * 100 if h1_ema21 > 0 else 0.0
        es = 16.0 + min(6.0, spread / 0.5 * 3.0); score += es
        signals.append(f"EMA趋势形成(发散{spread:.2f}% → +{es:.0f}分)")
    else:
        signals.append("EMA趋势未形成"); return score, signals, 'WAIT'
    macd_golden = False; macd_above_zero = False
    if len(macd_line) >= 2 and len(macd_sig) >= 2:
        prev_diff = float(macd_line.iloc[-2]) - float(macd_sig.iloc[-2])
        curr_diff = float(macd_line.iloc[-1]) - float(macd_sig.iloc[-1])
        macd_golden = prev_diff <= 0 and curr_diff > 0
        macd_above_zero = float(macd_hist.iloc[-1]) > 0 if len(macd_hist) >= 1 else False
    if macd_golden: score += 14.0; signals.append("MACD金叉确认(+14分)")
    elif macd_above_zero: score += 7.0; signals.append("MACD在零轴上方(+7分)")
    elif len(macd_hist) >= 2 and float(macd_hist.iloc[-1]) > float(macd_hist.iloc[-2]):
        score += 4.0; signals.append("MACD动量恢复(+4分)")
    if h4_adx >= trend_adx_min:
        adx_s = min(12.0, 8.0 + (h4_adx - trend_adx_min) / 10.0 * 4.0)
        score += adx_s; signals.append(f"ADX趋势确认({h4_adx:.1f} → +{adx_s:.0f}分)")
    else:
        signals.append(f"ADX偏弱({h4_adx:.1f})")
    if candle_ok: cs = 14.0 * candle_quality; score += cs; signals.append(f"趋势K线({candle_quality:.0%} → +{cs:.0f}分)")
    if vol_ratio >= min_vr * 0.8:
        vs = min(10.0, 10.0 * vol_ratio / min_vr); score += vs; signals.append(f"量能({vol_ratio:.2f}x → +{vs:.0f}分)")
    rsi_center = (trend_rsi_min + trend_rsi_max) / 2
    rsi_quality = 1.0 - abs(h1_rsi - rsi_center) / ((trend_rsi_max - trend_rsi_min) / 2)
    rsi_s = 10.0 * max(0.0, rsi_quality)
    if rsi_s >= 3.0: score += rsi_s; signals.append(f"RSI恢复健康({h1_rsi:.1f} → +{rsi_s:.0f}分)")
    if vol_zscore >= 0.5: score += min(6.0, 6.0 * vol_zscore / 1.5)
    if _ors_obv_trend_up(h1): score += 6.0; signals.append("OBV趋势翻转(+6分)")
    if _ors_bb_expanding(h1['c']) and price > bb_lower * 1.02: score += 6.0; signals.append("布林带扩张(+6分)")
    if m3_retest.get('passed'): score += 6.0; signals.append(str(m3_retest.get('reason')))
    elif m3_retest.get('valid'): signals.append(f"3m观察：{m3_retest.get('reason')}")
    m3_ok = bool(m3_retest.get('passed')) or not require_3m
    direction = 'BUY' if ema_trend and m3_ok and candle_ok else 'WAIT'
    return score, signals, direction


def _ors_three_min_reversal(df, *, cfg):
    window = int(cfg.get('m3_reversal_window', 80)); neckline_bars = int(cfg.get('m3_neckline_bars', 10))
    origin_tol = float(cfg.get('m3_origin_tolerance_pct', 0.18)); breakout_buf = float(cfg.get('m3_breakout_buffer_pct', 0.12))
    min_pb_bars = int(cfg.get('m3_min_pullback_bars', 3)); min_bars = neckline_bars + min_pb_bars + 6
    if len(df) < max(window, min_bars): return _ors_m3_default(f'3m数据不足({len(df)}/{max(window, min_bars)})')
    recent = df.tail(window).reset_index(drop=True); n = len(recent)
    ose = max(neckline_bars + min_pb_bars + 4, n // 2)
    op = int(recent.iloc[:ose]['l'].idxmin()); opr = float(recent['l'].iloc[op])
    ns_ = op + 1; ne = min(ns_ + neckline_bars, n - min_pb_bars - 3)
    if ne <= ns_ + 1: return _ors_m3_default('3m颈线样本不足', origin_price=opr)
    neck = recent.iloc[ns_:ne]; bl = float(neck['h'].max())
    if not pd.notna(bl) or bl <= 0: return _ors_m3_default('3m颈线异常', origin_price=opr)
    tl = bl * (1 + breakout_buf / 100); fl = opr * (1 - origin_tol / 100)
    phase = 0; p1h = 0.0; cl_ = tl; pbl = float('inf'); pbc = 0; p3c = 0.0
    for i in range(ne, n):
        bc = float(recent['c'].iloc[i]); bh = float(recent['h'].iloc[i]); bll = float(recent['l'].iloc[i])
        if phase == 0:
            if bc > tl: phase = 1; p1h = bh; cl_ = bh
        elif phase == 1:
            if bc > tl:
                if bh > p1h: p1h = bh; cl_ = bh
            else:
                if bc < fl: return _ors_m3_failed('Phase2收盘跌破', opr, bl, bc, p1h)
                phase = 2; pbl = bll; pbc = 1
        elif phase == 2:
            pbl = min(pbl, bll); pbc += 1
            if bc < fl: return _ors_m3_failed('Phase2收盘跌破', opr, bl, pbl, p1h)
            if pbc >= min_pb_bars and bc > cl_: phase = 3; p3c = bc; break
    lc = float(recent['c'].iloc[-1])
    if phase == 3:
        dp = (pbl - opr) / opr * 100; cp = (p3c - cl_) / cl_ * 100; held = lc > p3c
        q = min(100, 60 + min(dp + 2, 4) * 5 + min(cp, 3) * 4 + (8 if held else 0))
        return {'valid': True, 'passed': True, 'quality': q,
                'reason': f"3m三段确认：突破→回踩({dp:+.2f}%)→继续(+{cp:.2f}%)",
                'phase': 3, 'origin_price': opr, 'breakout_level': bl, 'phase1_high': p1h,
                'pullback_low': pbl, 'confirmation_level': cl_, 'phase3_close': p3c,
                'continuation_pct': cp, 'pullback_depth_pct': dp}
    elif phase == 2:
        dp = (pbl - opr) / opr * 100; fh = pbl >= fl; q = 35 + 25 + (20 if fh else 0)
        r = f"3m已突破并回踩({dp:+.2f}%)，等继续" if fh else f"3m回踩跌破起点({dp:+.2f}%)"
        return {'valid': True, 'passed': False, 'quality': float(q), 'reason': r, 'phase': 2,
                'origin_price': opr, 'breakout_level': bl, 'phase1_high': p1h, 'pullback_low': pbl,
                'confirmation_level': cl_, 'phase3_close': 0.0, 'continuation_pct': 0.0, 'pullback_depth_pct': dp}
    elif phase == 1:
        return {'valid': True, 'passed': False, 'quality': 60.0,
                'reason': f"3m已突破颈线，等回踩(高点={p1h:.6g})", 'phase': 1,
                'origin_price': opr, 'breakout_level': bl, 'phase1_high': p1h, 'pullback_low': float('nan'),
                'confirmation_level': cl_, 'phase3_close': 0.0, 'continuation_pct': 0.0, 'pullback_depth_pct': 0.0}
    else:
        return _ors_m3_default('3m尚未完成初次突破', origin_price=opr)


def _ors_m3_default(reason, origin_price=0.0):
    return {'valid': False, 'passed': False, 'quality': 35.0, 'reason': reason, 'phase': 0,
            'origin_price': origin_price, 'breakout_level': 0.0, 'phase1_high': 0.0,
            'pullback_low': 0.0, 'confirmation_level': 0.0, 'phase3_close': 0.0,
            'continuation_pct': 0.0, 'pullback_depth_pct': 0.0}

def _ors_m3_failed(reason, opr, bl, pullback_low, phase1_high):
    return {'valid': True, 'passed': False, 'quality': 15.0, 'reason': reason, 'phase': 2,
            'origin_price': opr, 'breakout_level': bl, 'phase1_high': phase1_high,
            'pullback_low': pullback_low, 'confirmation_level': phase1_high, 'phase3_close': 0.0,
            'continuation_pct': 0.0,
            'pullback_depth_pct': (pullback_low - opr) / opr * 100 if opr > 0 else -999}

def _ors_bb_lower(close, period=20, width=2.0):
    if len(close) < period: return float(close.iloc[-1])
    mid = close.rolling(period).mean().iloc[-1]; std = close.rolling(period).std(ddof=1).iloc[-1]
    return float(mid - width * std) if pd.notna(std) else float(mid)

def _ors_bb_expanding(close, period=20):
    if len(close) < period + 3: return False
    mid = close.rolling(period).mean(); std = close.rolling(period).std(ddof=1)
    bw = (std / mid * 100).dropna()
    if len(bw) < 3: return False
    return float(bw.iloc[-1]) > float(bw.iloc[-2]) > float(bw.iloc[-3])

def _ors_obv_trend_up(df, short=5, long=13):
    if len(df) < long + 2: return False
    obv = (df['vol'] * np.sign(df['c'].diff().fillna(0))).cumsum()
    obv_short = obv.ewm(span=short, adjust=False).mean(); obv_long = obv.ewm(span=long, adjust=False).mean()
    prev = float(obv_short.iloc[-2]) - float(obv_long.iloc[-2])
    curr = float(obv_short.iloc[-1]) - float(obv_long.iloc[-1])
    return prev <= 0 and curr > 0

def _ors_reversal_candle_quality(h1):
    if len(h1) < 2: return 0.0, False
    o = float(h1['o'].iloc[-1]); h = float(h1['h'].iloc[-1])
    l = float(h1['l'].iloc[-1]); c = float(h1['c'].iloc[-1])
    prev_c = float(h1['c'].iloc[-2]); prev_o = float(h1['o'].iloc[-2])
    fr = h - l
    if fr <= 0: return 0.0, False
    body = c - o; ls = o - l; br = body / fr
    is_bull = body > 0; close_above = c > prev_c; body_ok = br >= 0.30
    if not (is_bull and close_above and body_ok): return max(0.0, br * 0.4), False
    engulf = c > prev_o; sr = ls / max(abs(body), fr * 0.01)
    quality = min(1.0, br * 1.2 + min(sr, 2.0) * 0.15 + (0.1 if engulf else 0.0))
    return quality, True


# ══════════════════════════════════════════════════════════════════════════════
# 第八节：波段缩量中继再启动扫描 v2（ContinuationCompressionSwingScanner）
# ══════════════════════════════════════════════════════════════════════════════

_CCS_ADX_MIN = 18.0

_CCS_DEFAULT_CONFIG = {
    'min_score': 70, 'min_volume_24h': 18_000_000,
    'runup_lookback_bars': 40, 'min_prior_runup_pct': 12.0,
    'consolidation_bars': 12, 'max_base_width_pct': 4.8,
    'max_base_width_atr': 2.5,
    'max_contraction_ratio': 0.82, 'breakout_buffer_pct': 0.2,
    'min_breakout_volume_ratio': 1.6, 'max_extension_atr': 2.0,
}


class ContinuationCompressionSwingScanner(_BASE_SCANNER_CLASS):
    required_bars = ['4H', '1H']
    name = "波段缩量中继再启动扫描"
    description = "4H 强趋势 → 1H 缩量整理 → 放量突破二次启动 + ADX/MACD/收盘强度确认"
    strategy_type = "scan"

    def __init__(self, config=None):
        self.config = {**_CCS_DEFAULT_CONFIG, **(config or {})}
        if _HAS_SCANNER_BASE and hasattr(super(), '__init__'):
            try: super().__init__(config or {})
            except Exception: pass

    def _init_conditions(self):
        if ScanCondition is None: return
        self.add_condition(ScanCondition(name="24H成交量", description="过滤流动性不足标的",
            field="volume_24h", operator=">=", value=self.config.get('min_volume_24h', 18_000_000)))

    def scan_symbol(self, symbol) -> dict:
        km = symbol.extra_data.get('klines', {})
        try:
            h4 = _to_df(self._get_klines(km, '4H'))
            h1 = _to_df(self._get_klines(km, '1H'))
            analysis = _ccs_analyze_core(h4, h1, getattr(symbol, 'last_price', 0.0), self.config)
        except Exception as exc:
            return {'symbol': getattr(symbol, 'inst_id', ''), 'passed': False, 'score': 0.0,
                    'direction': 'WAIT', 'details': {'状态': f'分析异常: {exc}'}}
        if not analysis['valid']:
            return {'symbol': getattr(symbol, 'inst_id', ''), 'passed': False, 'score': 0.0,
                    'direction': 'WAIT', 'details': {'状态': analysis.get('reason', '')}}
        ms = float(self.config.get('min_score', 70))
        passed = analysis['score'] >= ms and analysis['direction'] in {'BUY', 'SELL'}
        result = {
            'symbol': getattr(symbol, 'inst_id', ''), 'passed': passed,
            'score': round(analysis['score'], 2), 'direction': analysis['direction'],
            'signals': analysis['signals'], 'details': analysis['details'],
            'last_price': getattr(symbol, 'last_price', 0.0), 'volume_24h': getattr(symbol, 'volume_24h', 0.0),
            'price_change_24h': getattr(symbol, 'price_change_24h', 0.0),
            'category': '缩量中继再启动', 'ranking_factors': analysis.get('ranking_factors', {}),
        }
        if build_opportunity_profile:
            try:
                result.update(build_opportunity_profile(
                    base_score=analysis['score'], direction=analysis['direction'],
                    volume_24h=getattr(symbol, 'volume_24h', 0.0),
                    factors=analysis.get('ranking_factors', {}), signals=analysis['signals']))
            except Exception: pass
        return result

    def _get_klines(self, km, bar):
        return km.get(bar) or km.get(bar.lower()) or km.get(bar.upper()) or []

    def get_config_schema(self):
        return {
            'min_score':                 {'type': 'int',   'default': 70,        'label': '最低通过分数'},
            'min_volume_24h':            {'type': 'float', 'default': 18_000_000,'label': '最小24H成交额'},
            'runup_lookback_bars':       {'type': 'int',   'default': 40,        'label': '前段涨/跌幅回望(4H根)'},
            'min_prior_runup_pct':       {'type': 'float', 'default': 12.0,      'label': '前段最小涨/跌幅%'},
            'consolidation_bars':        {'type': 'int',   'default': 12,        'label': '整理期窗口(1H根)'},
            'max_base_width_pct':        {'type': 'float', 'default': 4.8,       'label': '平台最大宽度%'},
            'max_base_width_atr':        {'type': 'float', 'default': 2.5,       'label': '平台最大宽度(ATR)'},
            'max_contraction_ratio':     {'type': 'float', 'default': 0.82,      'label': '缩量系数上限'},
            'breakout_buffer_pct':       {'type': 'float', 'default': 0.2,       'label': '突破缓冲%'},
            'min_breakout_volume_ratio': {'type': 'float', 'default': 1.6,       'label': '启动最小量比'},
            'max_extension_atr':         {'type': 'float', 'default': 2.0,       'label': '最大延伸(ATR)'},
        }


def _ccs_analyze_core(h4, h1, last_price, cfg):
    _check_df(h4, '4H', 90); _check_df(h1, '1H', 120)
    score = 0.0; signals = []
    price = float(last_price) if last_price and last_price > 0 else float(h1['c'].iloc[-1])
    ema21_4h = _ema(h4['c'], 21); ema55_4h = _ema(h4['c'], 55); ema21_1h = _ema(h1['c'], 21)
    h4_adx = _adx(h4, 14); h4_atr = _atr(h4); h1_atr = _atr(h1)
    h4_ema_spread = (ema21_4h - ema55_4h) / ema55_4h * 100 if ema55_4h > 0 else 0.0
    h1_rsi = _rsi_wilder(h1['c']); _, _, h1_macd_hist = _macd(h1['c'])
    vol_zscore = _volume_zscore(h1['vol'])
    runup_lb = int(cfg.get('runup_lookback_bars', 40)); min_runup = float(cfg.get('min_prior_runup_pct', 12))
    swing_low = float(h4['c'].tail(runup_lb).min()); swing_high = float(h4['c'].tail(runup_lb).max())
    runup_pct = (price - swing_low) / swing_low * 100 if swing_low > 0 else 0.0
    rundown_pct = (swing_high - price) / swing_high * 100 if swing_high > 0 else 0.0
    bullish = (price > ema21_4h > ema55_4h and price > ema55_4h
               and runup_pct >= min_runup and h4_adx >= _CCS_ADX_MIN and h4_ema_spread >= 0.20)
    bearish = (price < ema21_4h < ema55_4h and price < ema55_4h
               and rundown_pct >= min_runup and h4_adx >= _CCS_ADX_MIN and h4_ema_spread <= -0.20)
    is_bull = bullish
    trend_runup = runup_pct if bullish else rundown_pct if bearish else 0.0
    # ① 趋势质量（26 分）
    if bullish or bearish:
        adx_bonus = min(4.0, max(0.0, (h4_adx - _CCS_ADX_MIN) / 12.0 * 4.0))
        spread_bonus = min(4.0, abs(h4_ema_spread) / 2.0 * 4.0)
        ts_ = 18.0 + adx_bonus + spread_bonus; score += ts_
        dl = "多头" if bullish else "空头"
        signals.append(f"4H{dl}强趋势(前段{'涨' if bullish else '跌'}{trend_runup:.1f}%, ADX {h4_adx:.1f} → +{ts_:.0f}分)")
    else:
        rp = []
        if not (price > ema21_4h > ema55_4h or price < ema21_4h < ema55_4h): rp.append("EMA排列不符")
        if runup_pct < min_runup and rundown_pct < min_runup: rp.append(f"前段涨/跌幅不足({max(runup_pct, rundown_pct):.1f}%)")
        if h4_adx < _CCS_ADX_MIN: rp.append(f"ADX不足({h4_adx:.1f})")
        signals.append("趋势不符: " + " | ".join(rp))
        return {'valid': True, 'reason': '', 'score': score, 'direction': 'WAIT', 'signals': signals,
                'ranking_factors': {'trend': 28.0, 'trigger': 30.0, 'volume': 0.0, 'location': 50.0, 'freshness': 25.0, 'risk': 54.0},
                'details': {'评估': ' | '.join(signals), '方向': 'N/A', '4H_ADX': f'{h4_adx:.1f}', '4H_EMA发散': f'{h4_ema_spread:+.2f}%'}}
    # ② 平台收窄（18 分）
    cb = int(cfg.get('consolidation_bars', 12))
    base_slice = h1.tail(cb)
    base_high = float(base_slice['h'].max()); base_low = float(base_slice['l'].min())
    base_mid = (base_high + base_low) / 2.0
    base_width_pct = (base_high - base_low) / base_mid * 100 if base_mid > 0 else 999.0
    base_width_atr = (base_high - base_low) / h1_atr if h1_atr > 0 else 999.0
    max_w_pct = float(cfg.get('max_base_width_pct', 4.8)); max_w_atr = float(cfg.get('max_base_width_atr', 2.5))
    tight = base_width_pct <= max_w_pct and base_width_atr <= max_w_atr
    if tight:
        tightness_pct = max(0.0, 1.0 - base_width_pct / max_w_pct) * 0.5
        tightness_atr = max(0.0, 1.0 - base_width_atr / max_w_atr) * 0.5
        ps = 18 * (0.5 + tightness_pct + tightness_atr); score += ps
        signals.append(f"整理平台收窄({base_width_pct:.1f}%/{base_width_atr:.1f}ATR → +{ps:.0f}分)")
    else:
        signals.append(f"整理区间过宽({base_width_pct:.1f}%/{base_width_atr:.1f}ATR)")
    # ③ 缩量（14 分）
    baseline_start = cb + 20
    prior_slice = h1['vol'].tail(baseline_start).head(20)
    prior_vol = float(prior_slice.median()) if not prior_slice.empty else 0.0
    consol_slice = h1['vol'].tail(cb + 1).iloc[:-1]
    consol_vol = float(consol_slice.mean()) if not consol_slice.empty else prior_vol
    breakout_bar_vol = float(h1['vol'].iloc[-1])
    contraction_ratio = consol_vol / prior_vol if prior_vol > 0 else 1.0
    breakout_vol_ratio = breakout_bar_vol / prior_vol if prior_vol > 0 else 1.0
    max_cr = float(cfg.get('max_contraction_ratio', 0.82))
    if contraction_ratio <= max_cr:
        cs = 14 * min(1.0, 0.5 + (max_cr - contraction_ratio) / max(max_cr - 0.3, 0.1) * 0.5)
        score += cs; signals.append(f"缩量({contraction_ratio:.2f}x → +{cs:.0f}分)")
    else:
        signals.append(f"未缩量({contraction_ratio:.2f}x)")
    # ④ 突破确认 + 收盘强度（18 分）
    buf = float(cfg.get('breakout_buffer_pct', 0.2)) / 100.0
    ref_slice = h1.tail(cb + 1).iloc[:-1]
    last_close = float(h1['c'].iloc[-1]); prev_close = float(h1['c'].iloc[-2])
    if is_bull:
        breakout_level = float(ref_slice['h'].max()) if not ref_slice.empty else base_high
        trigger = breakout_level * (1.0 + buf)
        prev_in_range = base_low * 0.995 <= prev_close <= base_high * 1.005
        breakout_ok = prev_in_range and last_close > trigger and last_close > ema21_1h
        breakout_magnitude = (last_close - trigger) / h1_atr if h1_atr > 0 else 0.0
    else:
        breakout_level = float(ref_slice['l'].min()) if not ref_slice.empty else base_low
        trigger = breakout_level * (1.0 - buf)
        prev_in_range = base_low * 0.995 <= prev_close <= base_high * 1.005
        breakout_ok = prev_in_range and last_close < trigger and last_close < ema21_1h
        breakout_magnitude = (trigger - last_close) / h1_atr if h1_atr > 0 else 0.0
    last_o = float(h1['o'].iloc[-1]); last_h = float(h1['h'].iloc[-1]); last_l = float(h1['l'].iloc[-1])
    bar_range = last_h - last_l
    close_strength = abs(last_close - last_o) / bar_range if bar_range > 0 else 0.0
    if breakout_ok and breakout_magnitude >= 0.15:
        bks = 18 * 0.6 + 18 * 0.25 * min(1.0, close_strength / 0.6) + 18 * 0.15 * min(1.0, breakout_magnitude / 0.8)
        score += bks; arrow = "向上突破" if is_bull else "向下跌破"
        signals.append(f"1H{arrow}(强度{close_strength:.2f}, 幅度{breakout_magnitude:.2f}ATR → +{bks:.0f}分)")
    elif breakout_ok:
        bks = 18 * 0.4; score += bks; signals.append(f"突破幅度偏小({breakout_magnitude:.2f}ATR → +{bks:.0f}分)")
    else:
        signals.append("尚未二次启动")
    # ⑤ 量能（10 分）
    min_vr = float(cfg.get('min_breakout_volume_ratio', 1.6))
    vrok = breakout_vol_ratio >= min_vr; vzok = vol_zscore >= 0.8
    if vrok and vzok:
        vs = 10 * 0.9; score += vs; signals.append(f"启动放量({breakout_vol_ratio:.2f}x, z={vol_zscore:+.2f} → +{vs:.0f}分)")
    elif vrok:
        vs = 10 * 0.6; score += vs; signals.append(f"量比达标({breakout_vol_ratio:.2f}x → +{vs:.0f}分)")
    elif vzok:
        score += 10 * 0.4
    else:
        signals.append(f"量能不足({breakout_vol_ratio:.2f}x)")
    # ⑥ MACD 方向（6 分）
    macd_ok = False
    if len(h1_macd_hist) >= 2:
        mh = float(h1_macd_hist.iloc[-1]); mp = float(h1_macd_hist.iloc[-2])
        if is_bull and mh > 0 and mh > mp: macd_ok = True
        elif not is_bull and mh < 0 and mh < mp: macd_ok = True
    if macd_ok: score += 6; signals.append(f"MACD方向确认(+6分)")
    elif breakout_ok and len(h1_macd_hist) >= 1:
        mh = float(h1_macd_hist.iloc[-1])
        if (is_bull and mh > 0) or (not is_bull and mh < 0): score += 6 * 0.4
    # ⑦ RSI 合理（4 分 bonus）
    rsi_ok = (is_bull and 50 <= h1_rsi <= 75) or (not is_bull and 25 <= h1_rsi <= 50)
    if rsi_ok: score += 4; signals.append(f"RSI合理({h1_rsi:.1f} → +4分)")
    elif (is_bull and h1_rsi > 82) or (not is_bull and h1_rsi < 18):
        signals.append(f"RSI极端({'超买' if is_bull else '超卖'}: {h1_rsi:.1f})")
    # ⑧ 延伸适中（4 分 bonus）
    max_ext = float(cfg.get('max_extension_atr', 2.0))
    ext_atr = ((last_close - base_low) / h4_atr if is_bull else (base_high - last_close) / h4_atr) if h4_atr > 0 else 0.0
    ext_atr = max(ext_atr, 0.0)
    if ext_atr <= max_ext:
        es = 4 * max(0.0, 1.0 - ext_atr / max(max_ext, 0.1))
        if es >= 1.0: score += es; signals.append(f"延伸适中({ext_atr:.1f}ATR → +{es:.0f}分)")
    else:
        signals.append(f"延伸偏大({ext_atr:.1f}ATR)")
    direction = 'WAIT'
    if breakout_ok and vrok: direction = 'BUY' if is_bull else 'SELL'
    lq = max(25, 100 - base_width_pct * 10); fq = max(25, 100 - max(ext_atr - 0.5, 0) * 25)
    vq = min(breakout_vol_ratio / 1.6, 1.7) * 58
    return {
        'valid': True, 'reason': '', 'score': max(score, 0.0), 'direction': direction, 'signals': signals,
        'ranking_factors': {
            'trend': 94.0 if h4_adx >= _CCS_ADX_MIN else 28.0,
            'trigger': 92.0 if direction in {'BUY', 'SELL'} else 30.0,
            'volume': vq, 'location': lq, 'freshness': fq,
            'risk': 88.0 if contraction_ratio <= 0.82 else 54.0,
        },
        'details': {
            '评估': ' | '.join(signals) if signals else '暂无缩量中继机会',
            '方向': '多头' if is_bull else '空头', '前段涨/跌幅': f'{trend_runup:.2f}%',
            '平台宽度%': f'{base_width_pct:.2f}%', '平台宽度ATR': f'{base_width_atr:.1f}',
            '缩量系数': f'{contraction_ratio:.2f}', '启动量比': f'{breakout_vol_ratio:.2f}x',
            '量能Z分': f'{vol_zscore:+.2f}σ', '4H_ADX': f'{h4_adx:.1f}',
            '4H_EMA发散': f'{h4_ema_spread:+.2f}%', '1H_RSI': f'{h1_rsi:.1f}', '延伸ATR': f'{ext_atr:.1f}',
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
# 第九节：量价背离扫描策略（VolumePriceDivergenceScanner）
# ══════════════════════════════════════════════════════════════════════════════

_VPD_CONFIG_SCHEMA = {
    "min_volume_24h":           {"type": "float", "default": 2_000_000,  "label": "最小24H成交额"},
    "min_score":                {"type": "float", "default": 58.0,       "label": "最低输出分数"},
    "top_n":                    {"type": "int",   "default": 15,         "label": "最多输出信号数"},
    "allow_short":              {"type": "bool",  "default": True,       "label": "允许空头"},
    "divergence_lookback_1h":   {"type": "int",   "default": 20,         "label": "1H背离检测回溯根数"},
    "divergence_lookback_4h":   {"type": "int",   "default": 16,         "label": "4H背离检测回溯根数"},
    "vol_decline_threshold":    {"type": "float", "default": 0.70,       "label": "量能递减阈值"},
    "price_rise_threshold":     {"type": "float", "default": 1.5,        "label": "价格上涨阈值%"},
    "vol_surge_ratio":          {"type": "float", "default": 1.60,       "label": "底背离放量倍数"},
    "price_drop_threshold":     {"type": "float", "default": -2.5,       "label": "价格下跌阈值%"},
    "climax_vol_ratio":         {"type": "float", "default": 3.0,        "label": "量能高潮单根量/均量"},
    "climax_price_range_pct":   {"type": "float", "default": 3.0,        "label": "量能高潮当天涨跌<此%"},
    "weak_rally_bars":          {"type": "int",   "default": 5,          "label": "缩量反弹检测连续根数"},
    "weak_rally_vol_decline":   {"type": "float", "default": 0.75,       "label": "缩量反弹量递减比例"},
    "weak_rally_price_rise":    {"type": "float", "default": 2.0,        "label": "缩量反弹累计涨幅≥此%"},
}
_VPD_DEFAULT_CONFIG = {k: v["default"] for k, v in _VPD_CONFIG_SCHEMA.items()}


class VolumePriceDivergenceScanner(_BASE_SCANNER_CLASS):
    required_bars = ["4H", "1H"]
    strategy_type = "scan"
    name = "量价背离扫描策略"
    description = "检测顶底背离/量能高潮/缩量反弹四类反转前兆"

    def __init__(self, config=None):
        self.config = {**_VPD_DEFAULT_CONFIG, **(config or {})}
        if _HAS_SCANNER_BASE and hasattr(super(), "__init__"):
            try: super().__init__(self.config)
            except Exception: pass

    def _init_conditions(self): pass

    def get_config_schema(self): return dict(_VPD_CONFIG_SCHEMA)

    def scan_symbol(self, symbol) -> dict:
        inst_id = getattr(symbol, "inst_id", "")
        price = float(getattr(symbol, "last_price", 0) or 0)
        vol24 = float(getattr(symbol, "volume_24h", 0) or 0)
        chg24 = float(getattr(symbol, "price_change_24h", 0) or 0)
        base = {"symbol": inst_id, "passed": False, "score": 0.0, "direction": "WAIT",
                "signals": [], "details": {}, "last_price": price, "volume_24h": vol24,
                "price_change_24h": chg24, "factor_scores": {}, "ranking_factors": {}}
        if vol24 < float(self.config.get("min_volume_24h", 2_000_000)):
            return {**base, "details": {"跳过原因": "成交额不足"}}
        h1 = self._get_klines(symbol, "1H"); h4 = self._get_klines(symbol, "4H")
        if len(h1) < 40 or len(h4) < 24:
            return {**base, "details": {"跳过原因": f"K线不足(H1:{len(h1)}/4H:{len(h4)})"}}
        h1_c, h1_v = self._extract_cv(h1); h4_c, h4_v = self._extract_cv(h4)
        results = []
        bear = self._detect_bearish_divergence(h1_c, h1_v, h4_c, h4_v)
        if bear["detected"]: results.append(("顶背离", bear, "SELL", "📉"))
        bull = self._detect_bullish_divergence(h1_c, h1_v, h4_c, h4_v)
        if bull["detected"]: results.append(("底背离", bull, "BUY", "📈"))
        if bool(self.config.get("allow_short", True)):
            climax = self._detect_volume_climax(h1_c, h1_v, h4_c, h4_v, chg24)
            if climax["detected"]:
                direction = "SELL" if climax.get("bias", "neutral") == "bear" else "BUY"
                results.append(("量能高潮", climax, direction, "📉" if direction == "SELL" else "📈"))
        weak = self._detect_weak_rally(h1_c, h1_v)
        if weak["detected"]: results.append(("缩量反弹", weak, "SELL", "⚠"))
        if not results: return {**base, "details": {"状态": "未检测到量价背离信号"}}
        best_type, best_r, direction, emoji = max(results, key=lambda x: x[1]["score"])
        score = best_r["score"]; passed = score >= float(self.config.get("min_score", 58.0))
        category = f"{emoji} {best_type}"
        signals = [f"{category} · {score:.1f}分"] + best_r.get("messages", [])[:4]
        ranking_factors = {
            "trend": score * 0.7, "trigger": score, "volume": best_r.get("vol_score", score * 0.8),
            "location": best_r.get("location_score", 50), "freshness": best_r.get("freshness_score", 60),
            "risk": 100 - score * 0.3,
        }
        return {**base, "passed": passed, "score": score, "opportunity_score": score,
                "direction": direction, "category": category, "signals": signals,
                "factor_scores": {"divergence_score": score}, "ranking_factors": ranking_factors,
                "details": {"机会类型": category, "背离类型": best_type,
                            "评分": f"{score:.1f}", "方向": "空头" if direction == "SELL" else "多头",
                            **best_r.get("diag", {})}}

    def _detect_bearish_divergence(self, h1_c, h1_v, h4_c, h4_v):
        lb = int(self.config["divergence_lookback_1h"]); lb4 = int(self.config["divergence_lookback_4h"])
        vol_th = float(self.config["vol_decline_threshold"]); price_th = float(self.config["price_rise_threshold"])
        half1 = max(4, lb // 2)
        if len(h1_c) < lb: return {"detected": False}
        p1, p2 = h1_c[-lb:-half1], h1_c[-half1:]; v1, v2 = h1_v[-lb:-half1], h1_v[-half1:]
        h1_div = (np.mean(p2) > np.mean(p1) * (1 + price_th / 100) and np.mean(v2) < np.mean(v1) * vol_th)
        half4 = max(3, lb4 // 2); h4_div = False
        if len(h4_c) >= lb4:
            p4_1, p4_2 = h4_c[-lb4:-half4], h4_c[-half4:]; v4_1, v4_2 = h4_v[-lb4:-half4], h4_v[-half4:]
            h4_div = (np.mean(p4_2) > np.mean(p4_1) * (1 + price_th / 200) and np.mean(v4_2) < np.mean(v4_1) * vol_th)
        score = 0; messages = []
        if h1_div:
            score += 45; vol_drop = (1 - np.mean(v2) / max(np.mean(v1), 1)) * 100
            messages.append(f"1H顶背离: HH(+{(np.mean(p2)/np.mean(p1)-1)*100:.1f}%) + 量缩({vol_drop:.0f}%)")
        if h4_div: score += 25; messages.append("4H顶背离确认")
        if h1_div: score += 10
        return {"detected": h1_div and score >= 50, "score": min(100, score + 20),
                "vol_score": min(100, score * 0.8), "messages": messages,
                "diag": {"1H顶背离": str(h1_div), "4H顶背离": str(h4_div),
                         "1H后半均量vs前半": f"{np.mean(v2)/max(np.mean(v1),1):.2f}x"}}

    def _detect_bullish_divergence(self, h1_c, h1_v, h4_c, h4_v):
        lb = int(self.config["divergence_lookback_1h"]); vol_th = float(self.config["vol_surge_ratio"])
        price_th = float(self.config["price_drop_threshold"]); half1 = max(4, lb // 2)
        if len(h1_c) < lb: return {"detected": False}
        p1, p2 = h1_c[-lb:-half1], h1_c[-half1:]; v1, v2 = h1_v[-lb:-half1], h1_v[-half1:]
        h1_div = (np.mean(p2) < np.mean(p1) * (1 + price_th / 100) and np.mean(v2) > np.mean(v1) * vol_th)
        score = 0; messages = []
        if h1_div:
            score += 45; vol_surge = (np.mean(v2) / max(np.mean(v1), 1) - 1) * 100
            messages.append(f"1H底背离: LL({(np.mean(p2)/np.mean(p1)-1)*100:.1f}%) + 放量(+{vol_surge:.0f}%)")
        if len(h1_c) >= 4 and h1_c[-1] > h1_c[-3]: score += 20; messages.append("价格止跌回升确认")
        return {"detected": h1_div and score >= 50, "score": min(100, score + 20),
                "vol_score": min(100, score * 0.9), "messages": messages,
                "diag": {"1H底背离": str(h1_div), "止跌": str(h1_c[-1] > h1_c[-3]) if len(h1_c) >= 3 else "N/A"}}

    def _detect_volume_climax(self, h1_c, h1_v, h4_c, h4_v, chg24):
        climax_vol = float(self.config["climax_vol_ratio"]); climax_range = float(self.config["climax_price_range_pct"])
        if len(h1_v) < 20 or len(h1_c) < 20: return {"detected": False}
        avg_vol = np.mean(h1_v[-20:-1]); last_vol = h1_v[-1]; vol_ratio = last_vol / max(avg_vol, 1)
        if vol_ratio < climax_vol: return {"detected": False}
        last_move = abs(h1_c[-1] / h1_c[-2] - 1) * 100 if len(h1_c) >= 2 else 0
        if last_move >= climax_range: return {"detected": False}
        bias = "bear" if chg24 > 3 and last_move < 2 else "neutral"
        return {"detected": True, "score": min(95, 55 + vol_ratio * 5), "vol_score": min(100, vol_ratio * 20),
                "location_score": 100 - last_move * 15, "freshness_score": 85, "bias": bias,
                "messages": [f"量能高潮: 单根量{vol_ratio:.1f}x均量 + 涨幅仅{last_move:.1f}%(未突破)", f"24H涨跌{chg24:+.1f}%"],
                "diag": {"量比": f"{vol_ratio:.1f}x", "单根涨跌": f"{last_move:.1f}%", "24H": f"{chg24:+.1f}%"}}

    def _detect_weak_rally(self, h1_c, h1_v):
        n = int(self.config["weak_rally_bars"]); vol_dec = float(self.config["weak_rally_vol_decline"])
        price_rise = float(self.config["weak_rally_price_rise"])
        if len(h1_c) < n + 3 or len(h1_v) < n + 3: return {"detected": False}
        recent_c = h1_c[-n:]; recent_v = h1_v[-n:]
        up_count = sum(1 for i in range(1, n) if recent_c[i] > recent_c[i-1])
        if up_count < n * 0.6: return {"detected": False}
        cum_rise = (recent_c[-1] / recent_c[0] - 1) * 100
        if cum_rise < price_rise: return {"detected": False}
        half = max(2, n // 2); v1 = np.mean(recent_v[:half]); v2 = np.mean(recent_v[-half:])
        if v2 >= v1 * vol_dec: return {"detected": False}
        return {"detected": True, "score": min(90, 50 + cum_rise * 5 + (1 - v2 / max(v1, 1)) * 30),
                "vol_score": min(90, (1 - v2 / max(v1, 1)) * 80),
                "messages": [f"缩量反弹: {n}根涨{cum_rise:.1f}% + 量缩{(1-v2/max(v1,1))*100:.0f}%"],
                "diag": {"连续上涨根数": f"{up_count}/{n}", "累计涨幅": f"{cum_rise:.1f}%",
                         "量比(后/前)": f"{v2/max(v1,1):.2f}x"}}

    @staticmethod
    def _get_klines(symbol, tf):
        try:
            ed = getattr(symbol, "extra_data", {}) or {}
            km = ed.get("klines", {}) if isinstance(ed, dict) else {}
            rows = km.get(tf) or []
            return [r for r in rows if isinstance(r, (list, tuple)) and len(r) >= 6]
        except Exception: return []

    @staticmethod
    def _extract_cv(rows):
        closes, volumes = [], []
        for r in rows:
            try:
                c = float(r[4]); v = float(r[5])
                if c > 0 and v >= 0: closes.append(c); volumes.append(v)
            except Exception: pass
        return closes, volumes


# ══════════════════════════════════════════════════════════════════════════════
# 第十节：趋势回踩二次启动筛选 v6（TrendPullbackRestartScanner）
# ══════════════════════════════════════════════════════════════════════════════

_TPR_W_TREND        = 32
_TPR_W_PULLBACK     = 18
_TPR_W_RETEST_TIME  =  6
_TPR_W_KEYLEVEL     = 10
_TPR_W_RESTART      = 20
_TPR_W_VOLUME       = 14
_TPR_W_3M           =  6

_TPR_ADX_MIN_TREND      = 18.0
_TPR_EMA_SPREAD_MIN     = 0.25
_TPR_RETEST_LOOKBACK    = 12
_TPR_RETEST_MIN_TOUCHES = 2

_TPR_DEFAULT_CONFIG = {
    'min_score':                 72,
    'min_volume_24h':             15_000_000,
    'max_pullback_distance_pct':   3.5,
    'keylevel_lookback_bars':      80,
    'max_key_level_retest_pct':    2.2,
    'min_restart_volume_ratio':    1.25,
    'min_volume_zscore':           0.8,
    'max_buy_rsi':                 68.0,
    'min_sell_rsi':                32.0,
    'max_h4_atr_pct':              6.5,
    'require_3m_restart':          True,
    'm3_swing_window':             60,
    'm3_neckline_bars':             8,
    'm3_min_pullback_bars':         3,
    'm3_breakout_buffer_pct':      0.12,
    'm3_origin_tolerance_pct':     0.15,
}


class TrendPullbackRestartScanner(_BASE_SCANNER_CLASS):
    required_bars = ['1D', '4H', '1H', '3m']
    name        = "趋势回踩二次启动筛选"
    description = "1D/4H 趋势确认 → 1H 回踩 EMA21 → 放量穿越 → 3m 三段式结构，多周期共振筛选"
    strategy_type = "scan"

    def __init__(self, config=None):
        self.config = {**_TPR_DEFAULT_CONFIG, **(config or {})}
        if _HAS_SCANNER_BASE and hasattr(super(), '__init__'):
            try: super().__init__(config or {})
            except Exception: pass

    def _init_conditions(self):
        if ScanCondition is None: return
        self.add_condition(ScanCondition(name="24H成交量", description="过滤流动性不足标的",
            field="volume_24h", operator=">=", value=self.config.get('min_volume_24h', 15_000_000)))

    def scan_symbol(self, symbol) -> dict:
        klines_map = symbol.extra_data.get('klines', {})
        try:
            d1 = _to_df(self._get_klines(klines_map, '1D'))
            h4 = _to_df(self._get_klines(klines_map, '4H'))
            h1 = _to_df(self._get_klines(klines_map, '1H'))
            m3 = _to_df(self._get_klines(klines_map, '3m'))
            analysis = _tpr_analyze_core(d1, h4, h1, m3, getattr(symbol, 'last_price', 0.0), self.config)
        except Exception as exc:
            return {'symbol': getattr(symbol, 'inst_id', ''), 'passed': False, 'score': 0.0,
                    'direction': 'WAIT', 'details': {'状态': f'分析异常: {exc}'}}
        if not analysis['valid']:
            return {'symbol': getattr(symbol, 'inst_id', ''), 'passed': False, 'score': 0.0,
                    'direction': 'WAIT', 'details': {'状态': analysis.get('reason', '')}}
        min_score = float(self.config.get('min_score', 72))
        passed = analysis['score'] >= min_score and analysis['direction'] in {'BUY', 'SELL'}
        result = {
            'symbol': getattr(symbol, 'inst_id', ''), 'passed': passed,
            'score': round(analysis['score'], 2), 'direction': analysis['direction'],
            'signals': analysis['signals'], 'details': analysis['details'],
            'last_price': getattr(symbol, 'last_price', 0.0), 'volume_24h': getattr(symbol, 'volume_24h', 0.0),
            'price_change_24h': getattr(symbol, 'price_change_24h', 0.0),
            'category': '趋势回踩二次启动', 'ranking_factors': analysis.get('ranking_factors', {}),
        }
        if build_opportunity_profile:
            try:
                result.update(build_opportunity_profile(
                    base_score=analysis['score'], direction=analysis['direction'],
                    volume_24h=getattr(symbol, 'volume_24h', 0.0),
                    factors=analysis.get('ranking_factors', {}), signals=analysis['signals']))
            except Exception: pass
        return result

    def _get_klines(self, klines_map, bar):
        return klines_map.get(bar) or klines_map.get(bar.lower()) or klines_map.get(bar.upper()) or []

    def get_config_schema(self):
        return {
            'min_score':                 {'type': 'int',   'default': 72,         'label': '最低通过分数(0-100)'},
            'min_volume_24h':            {'type': 'float', 'default': 15_000_000, 'label': '最小24H成交额'},
            'max_pullback_distance_pct': {'type': 'float', 'default': 3.5,        'label': '最大回踩4H EMA21距离%'},
            'min_restart_volume_ratio':  {'type': 'float', 'default': 1.25,       'label': '二次启动最小量比'},
            'min_volume_zscore':         {'type': 'float', 'default': 0.8,        'label': '启动量 z-score 下限'},
            'max_key_level_retest_pct':  {'type': 'float', 'default': 2.2,        'label': '关键位回测容差%'},
            'require_3m_restart':        {'type': 'bool',  'default': True,       'label': '要求3m三段式确认'},
        }


def _tpr_ema_slope_pct(c, span, lb):
    e = c.ewm(span=span, adjust=False).mean()
    if len(e) <= lb: return 0.0
    b = float(e.iloc[-(lb + 1)]); l = float(e.iloc[-1])
    return (l / b - 1.0) * 100 if b > 0 else 0.0

def _tpr_atr_pct(df, period=14):
    if len(df) < period + 1: return 0.0
    pc = df['c'].shift(1)
    tr = pd.concat([df['h'] - df['l'], (df['h'] - pc).abs(), (df['l'] - pc).abs()], axis=1).max(axis=1)
    a = tr.ewm(alpha=1 / period, adjust=False).mean().iloc[-1]
    lc = float(df['c'].iloc[-1])
    if pd.isna(a) or lc <= 0: return 0.0
    return float(a / lc * 100)

def _tpr_m3_default(reason, origin_price=0.0):
    return {'valid': False, 'passed': False, 'quality': 35.0, 'reason': reason, 'phase': 0,
            'bull_bias': True, 'origin_price': origin_price, 'breakout_level': 0.0,
            'phase1_extreme': 0.0, 'pullback_extreme': 0.0, 'confirmation_level': 0.0,
            'phase3_close': 0.0, 'breakout_pct': 0.0, 'continuation_pct': 0.0, 'pullback_depth_pct': 0.0}

def _tpr_m3_failed(reason, *, origin_price, breakout_level, pullback_extreme, phase1_extreme, bull_bias):
    bp = abs(phase1_extreme - breakout_level) / breakout_level * 100 if breakout_level > 0 else 0.0
    dp = (pullback_extreme - origin_price) / origin_price * 100 if origin_price > 0 else -999.0
    return {'valid': True, 'passed': False, 'quality': 15.0, 'reason': reason, 'phase': 2,
            'bull_bias': bull_bias, 'origin_price': origin_price, 'breakout_level': breakout_level,
            'phase1_extreme': phase1_extreme, 'pullback_extreme': pullback_extreme,
            'confirmation_level': phase1_extreme, 'phase3_close': 0.0,
            'breakout_pct': bp, 'continuation_pct': 0.0, 'pullback_depth_pct': dp}


def _tpr_three_min_restart(df, *, bull_bias, cfg):
    window = int(cfg.get('m3_swing_window', 60)); neckline_bars = int(cfg.get('m3_neckline_bars', 8))
    origin_tol = float(cfg.get('m3_origin_tolerance_pct', 0.15)); breakout_buf = float(cfg.get('m3_breakout_buffer_pct', 0.12))
    min_pb_bars = int(cfg.get('m3_min_pullback_bars', 3)); min_bars = neckline_bars + min_pb_bars + 6
    if len(df) < max(window, min_bars): return _tpr_m3_default(f'3m数据不足({len(df)}/{max(window, min_bars)})')
    recent = df.tail(window).reset_index(drop=True); n = len(recent)
    ose = max(neckline_bars + min_pb_bars + 4, n // 2)
    if bull_bias: op = int(recent.iloc[:ose]['l'].idxmin()); opr = float(recent['l'].iloc[op])
    else: op = int(recent.iloc[:ose]['h'].idxmax()); opr = float(recent['h'].iloc[op])
    ns_ = op + 1; ne = min(ns_ + neckline_bars, n - min_pb_bars - 3)
    if ne <= ns_ + 1: return _tpr_m3_default(f'3m颈线样本不足', origin_price=opr)
    neck = recent.iloc[ns_:ne]
    if bull_bias:
        bl = float(neck['h'].max())
        if not pd.notna(bl) or bl <= 0: return _tpr_m3_default('3m颈线计算异常', origin_price=opr)
        tl = bl * (1.0 + breakout_buf / 100.0); fl = opr * (1.0 - origin_tol / 100.0)
    else:
        bl = float(neck['l'].min())
        if not pd.notna(bl) or bl <= 0: return _tpr_m3_default('3m颈线计算异常(空头)', origin_price=opr)
        tl = bl * (1.0 - breakout_buf / 100.0); fl = opr * (1.0 + origin_tol / 100.0)
    phase = 0; p1e = opr; cl_ = tl; pbe = float('inf') if bull_bias else float('-inf'); pbc = 0; p3c = 0.0
    for i in range(ne, n):
        bc = float(recent['c'].iloc[i]); bh = float(recent['h'].iloc[i]); bll = float(recent['l'].iloc[i])
        if bull_bias:
            if phase == 0:
                if bc > tl: phase = 1; p1e = bh; cl_ = bh
            elif phase == 1:
                if bc > tl:
                    if bh > p1e: p1e = bh; cl_ = bh
                else:
                    if bc < fl: return _tpr_m3_failed(f'Phase2首棒收盘跌破起点', origin_price=opr, breakout_level=bl, pullback_extreme=bc, phase1_extreme=p1e, bull_bias=True)
                    phase = 2; pbe = bll; pbc = 1
            elif phase == 2:
                pbe = min(pbe, bll); pbc += 1
                if bc < fl: return _tpr_m3_failed(f'Phase2收盘跌破起点', origin_price=opr, breakout_level=bl, pullback_extreme=pbe, phase1_extreme=p1e, bull_bias=True)
                if pbc >= min_pb_bars and bc > cl_: phase = 3; p3c = bc; break
        else:
            if phase == 0:
                if bc < tl: phase = 1; p1e = bll; cl_ = bll
            elif phase == 1:
                if bc < tl:
                    if bll < p1e: p1e = bll; cl_ = bll
                else:
                    if bc > fl: return _tpr_m3_failed(f'Phase2首棒收盘反抽超起点', origin_price=opr, breakout_level=bl, pullback_extreme=bc, phase1_extreme=p1e, bull_bias=False)
                    phase = 2; pbe = bh; pbc = 1
            elif phase == 2:
                pbe = max(pbe, bh); pbc += 1
                if bc > fl: return _tpr_m3_failed(f'Phase2收盘反抽超起点', origin_price=opr, breakout_level=bl, pullback_extreme=pbe, phase1_extreme=p1e, bull_bias=False)
                if pbc >= min_pb_bars and bc < cl_: phase = 3; p3c = bc; break
    lc = float(recent['c'].iloc[-1])
    if phase == 3:
        if bull_bias: dp = (pbe - opr) / opr * 100; cp = (p3c - cl_) / cl_ * 100; held = lc > p3c
        else: dp = (opr - pbe) / opr * 100; cp = (cl_ - p3c) / cl_ * 100; held = lc < p3c
        q = min(100.0, 60.0 + min(dp + 2.0, 4.0) * 5.0 + min(cp, 3.0) * 4.0 + (8.0 if held else 0.0))
        lb = "多头" if bull_bias else "空头"
        return {'valid': True, 'passed': True, 'quality': q,
                'reason': f"3m{lb}三段确认：突破→回踩({dp:+.2f}%)→继续突破(+{cp:.2f}%)",
                'phase': 3, 'bull_bias': bull_bias, 'origin_price': opr, 'breakout_level': bl,
                'phase1_extreme': p1e, 'pullback_extreme': pbe, 'confirmation_level': cl_,
                'phase3_close': p3c, 'breakout_pct': abs(p1e - bl) / bl * 100,
                'continuation_pct': cp, 'pullback_depth_pct': dp}
    elif phase == 2:
        if bull_bias: dp = (pbe - opr) / opr * 100; fh = pbe >= fl
        else: dp = (opr - pbe) / opr * 100; fh = pbe <= fl
        q = 35.0 + 25.0 + (20.0 if fh else 0.0)
        r = f"3m已突破并回踩({dp:+.2f}%)，等待继续突破确认" if fh else f"3m回踩越过起点({dp:+.2f}%)"
        return {'valid': True, 'passed': False, 'quality': float(q), 'reason': r, 'phase': 2,
                'bull_bias': bull_bias, 'origin_price': opr, 'breakout_level': bl,
                'phase1_extreme': p1e, 'pullback_extreme': pbe, 'confirmation_level': cl_,
                'phase3_close': 0.0, 'breakout_pct': abs(p1e - bl) / bl * 100,
                'continuation_pct': 0.0, 'pullback_depth_pct': dp}
    elif phase == 1:
        lb = "多头" if bull_bias else "空头"
        return {'valid': True, 'passed': False, 'quality': 60.0,
                'reason': f"3m{lb}已突破颈线，等待回踩确认(Phase1极值={p1e:.6g})",
                'phase': 1, 'bull_bias': bull_bias, 'origin_price': opr, 'breakout_level': bl,
                'phase1_extreme': p1e, 'pullback_extreme': float('nan'), 'confirmation_level': cl_,
                'phase3_close': 0.0, 'breakout_pct': abs(p1e - bl) / bl * 100,
                'continuation_pct': 0.0, 'pullback_depth_pct': 0.0}
    else:
        return _tpr_m3_default(f'3m尚未完成初次{"向上" if bull_bias else "向下"}突破', origin_price=opr)


def _tpr_analyze_core(d1, h4, h1, m3, last_price, cfg):
    _check_df(d1, '日线', 90); _check_df(h4, '4H', 110); _check_df(h1, '1H', 140)
    require_3m = bool(cfg.get('require_3m_restart', True))
    min_3m_bars = max(int(cfg.get('m3_swing_window', 60)),
                      int(cfg.get('m3_neckline_bars', 8)) + int(cfg.get('m3_min_pullback_bars', 3)) + 6)
    if require_3m and len(m3) < min_3m_bars:
        require_3m = False  # 3m 数据缺失时降级
    score = 0.0; signals = []
    price = float(last_price) if last_price and last_price > 0 else float(h1['c'].iloc[-1])
    d1_ema21 = _ema(d1['c'], 21); d1_ema55 = _ema(d1['c'], 55)
    h4_ema21 = _ema(h4['c'], 21); h4_ema55 = _ema(h4['c'], 55)
    h1_ema21 = _ema(h1['c'], 21); h1_ema55 = _ema(h1['c'], 55)
    h1_last_close = float(h1['c'].iloc[-1]); h1_prev_close = float(h1['c'].iloc[-2])
    d1_slope = _tpr_ema_slope_pct(d1['c'], 21, 6); h4_slope = _tpr_ema_slope_pct(h4['c'], 21, 6)
    h1_rsi = _rsi_wilder(h1['c']); h4_atr_pct = _tpr_atr_pct(h4)
    vol_ratio = _volume_ratio_adjusted(h1); vol_zscore = _volume_zscore(h1['vol'])
    d1_adx = _adx(d1, 14); h4_adx = _adx(h4, 14)
    d1_ema_spread = (d1_ema21 - d1_ema55) / d1_ema55 * 100 if d1_ema55 > 0 else 0.0
    h4_ema_spread = (h4_ema21 - h4_ema55) / h4_ema55 * 100 if h4_ema55 > 0 else 0.0
    _m3ph = _tpr_m3_default('趋势未确认，跳过3m')
    trend_snap = _local_trend_snapshot(d1, h4, h1, price)
    trend_metrics = trend_snap.get('metrics', {})
    trend_long_score = float(trend_snap.get('long_score', 0.0) or 0.0)
    trend_short_score = float(trend_snap.get('short_score', 0.0) or 0.0)
    bullish_trend = (
        bool(trend_snap.get('long_ok'))
        and price > d1_ema21 > d1_ema55 and price > h4_ema21 > h4_ema55
        and d1_slope > 0.4 and h4_slope > 0.6
        and d1_adx >= _TPR_ADX_MIN_TREND and h4_adx >= _TPR_ADX_MIN_TREND
        and d1_ema_spread >= _TPR_EMA_SPREAD_MIN and h4_ema_spread >= _TPR_EMA_SPREAD_MIN)
    bearish_trend = (
        bool(trend_snap.get('short_ok'))
        and price < d1_ema21 < d1_ema55 and price < h4_ema21 < h4_ema55
        and d1_slope < -0.4 and h4_slope < -0.6
        and d1_adx >= _TPR_ADX_MIN_TREND and h4_adx >= _TPR_ADX_MIN_TREND
        and d1_ema_spread <= -_TPR_EMA_SPREAD_MIN and h4_ema_spread <= -_TPR_EMA_SPREAD_MIN)
    # ① 趋势（32 分）
    if bullish_trend or bearish_trend:
        spread_abs = (abs(d1_ema_spread) + abs(h4_ema_spread)) / 2.0
        spread_bonus = min(5.0, spread_abs / 2.0 * 5.0)
        adx_avg = (d1_adx + h4_adx) / 2.0
        adx_bonus = min(3.0, max(0.0, (adx_avg - _TPR_ADX_MIN_TREND) / 12.0 * 3.0))
        ts_ = 24.0 + spread_bonus + adx_bonus; score += ts_
        dl = "多头" if bullish_trend else "空头"
        rs = trend_long_score if bullish_trend else trend_short_score
        signals.append(f"{dl}趋势通过(质量{rs:.0f}, ADX {adx_avg:.1f}, 发散{spread_abs:.2f}% → +{ts_:.1f}分)")
    else:
        rp = []
        if d1_adx < _TPR_ADX_MIN_TREND: rp.append(f"日线ADX不足({d1_adx:.1f})")
        if h4_adx < _TPR_ADX_MIN_TREND: rp.append(f"4H ADX不足({h4_adx:.1f})")
        if abs(d1_ema_spread) < _TPR_EMA_SPREAD_MIN: rp.append(f"日线EMA发散不足({d1_ema_spread:.2f}%)")
        if abs(h4_ema_spread) < _TPR_EMA_SPREAD_MIN: rp.append(f"4H EMA发散不足({h4_ema_spread:.2f}%)")
        if not (d1_slope > 0.4 or d1_slope < -0.4): rp.append(f"日线斜率不足({d1_slope:.2f}%)")
        if not (h4_slope > 0.6 or h4_slope < -0.6): rp.append(f"4H斜率不足({h4_slope:.2f}%)")
        signals.append("趋势未确认: " + " | ".join(rp) if rp else "趋势未确认")
        tq = max(trend_long_score, trend_short_score)
        return {'valid': True, 'reason': '', 'score': score, 'direction': 'WAIT', 'signals': signals,
                'ranking_factors': {'trend': tq, 'trigger': 32.0, 'volume': 0.0, 'location': 50.0, 'freshness': 34.0, 'risk': 55.0},
                'details': {'评估': ' | '.join(signals), '日线ADX': f'{d1_adx:.1f}', '4H_ADX': f'{h4_adx:.1f}',
                            '日线EMA发散': f'{d1_ema_spread:+.2f}%', '4H_EMA发散': f'{h4_ema_spread:+.2f}%'}}
    # ② 方向性回踩（18 分）
    raw_distance = (price - h4_ema21) / h4_ema21 * 100 if h4_ema21 > 0 else 999.0
    max_pb = float(cfg.get('max_pullback_distance_pct', 3.5))
    if bullish_trend: pullback_ok = 0.0 <= raw_distance <= max_pb; pullback_depth = abs(raw_distance); direction_mismatch = raw_distance < 0
    else: pullback_ok = -max_pb <= raw_distance <= 0.0; pullback_depth = abs(raw_distance); direction_mismatch = raw_distance > 0
    if pullback_ok:
        pb_score = _TPR_W_PULLBACK * min(1.0, 0.55 + (1.0 - pullback_depth / max(max_pb, 0.1)) * 0.45)
        score += pb_score; signals.append(f"方向性回踩4H EMA21({pullback_depth:.2f}% → +{pb_score:.1f}分)")
    elif direction_mismatch: signals.append(f"价格已穿越EMA21至反向侧({raw_distance:+.2f}%)")
    else: signals.append(f"距均线过远({pullback_depth:.2f}%/{max_pb}%)")
    # ③ 回踩时间（6 分）
    h1_ema21_s = h1['c'].ewm(span=21, adjust=False).mean()
    rw = h1.tail(_TPR_RETEST_LOOKBACK); ew = h1_ema21_s.tail(_TPR_RETEST_LOOKBACK)
    tc = int((rw['l'].values <= ew.values).sum()) if bullish_trend else int((rw['h'].values >= ew.values).sum())
    if tc >= _TPR_RETEST_MIN_TOUCHES:
        rt = _TPR_W_RETEST_TIME * min(1.0, tc / 5.0); score += rt
        signals.append(f"回踩过程真实(近{_TPR_RETEST_LOOKBACK}根触碰{tc}根 → +{rt:.1f}分)")
    else: signals.append(f"未见真实回踩过程(仅{tc}根触及/需{_TPR_RETEST_MIN_TOUCHES})")
    # ④ 关键位（10 分）
    sh, sl = _latest_swing_levels(h4, left=5, right=5, skip_recent=3, max_lookback=80)
    retest_quality = 0.0; max_kl = float(cfg.get('max_key_level_retest_pct', 2.2))
    if bullish_trend and sh and sh > 0 and price > 0 and sh < price:
        retest_quality = (price - sh) / sh * 100
        if retest_quality <= max_kl:
            kls = _TPR_W_KEYLEVEL * max(0.5, 1.0 - retest_quality / max(max_kl, 0.1))
            score += kls; signals.append(f"回踩前swing高支撑({retest_quality:.2f}% → +{kls:.1f}分)")
    elif bearish_trend and sl and sl > 0 and price > 0 and sl > price:
        retest_quality = (sl - price) / sl * 100
        if retest_quality <= max_kl:
            kls = _TPR_W_KEYLEVEL * max(0.5, 1.0 - retest_quality / max(max_kl, 0.1))
            score += kls; signals.append(f"反抽前swing低压力({retest_quality:.2f}% → +{kls:.1f}分)")
    # ⑤ 重启 + 收盘强度（20 分）
    restart_up = bullish_trend and h1_prev_close <= h1_ema21 and h1_last_close > h1_ema21 and h1_ema21 > h1_ema55
    restart_down = bearish_trend and h1_prev_close >= h1_ema21 and h1_last_close < h1_ema21 and h1_ema21 < h1_ema55
    last_h1 = h1.iloc[-1]; br = float(last_h1['h']) - float(last_h1['l'])
    if br > 0:
        close_strength = ((float(last_h1['c']) - float(last_h1['l'])) / br if bullish_trend
                          else (float(last_h1['h']) - float(last_h1['c'])) / br)
    else: close_strength = 0.5
    if restart_up or restart_down:
        b = _TPR_W_RESTART * 0.7; sb = _TPR_W_RESTART * 0.3 * max(0.0, (close_strength - 0.5) / 0.5)
        rs_ = b + sb; score += rs_
        arr = "站上" if restart_up else "跌破"
        signals.append(f"1H收盘{arr}EMA21(收盘强度{close_strength:.2f} → +{rs_:.1f}分)")
    else:
        signals.append("1H尚未二次启动")
    # ⑥ 量能（14 分）
    min_vr = float(cfg.get('min_restart_volume_ratio', 1.25)); min_zs = float(cfg.get('min_volume_zscore', 0.8))
    vrok = vol_ratio >= min_vr; vzok = vol_zscore >= min_zs
    if vrok and vzok:
        rc = 0.55 + min(1.0, (vol_ratio - min_vr) / max(min_vr, 0.1) * 0.25) * 0.45
        zc = 0.55 + min(1.0, (vol_zscore - min_zs) / 1.5) * 0.45
        vs = _TPR_W_VOLUME * (rc + zc) / 2.0; score += vs
        signals.append(f"启动量能强({vol_ratio:.2f}x, z={vol_zscore:+.2f} → +{vs:.1f}分)")
    elif vrok or vzok:
        vs = _TPR_W_VOLUME * 0.45; score += vs
        signals.append(f"量能部分满足({vol_ratio:.2f}x, z={vol_zscore:+.2f} → +{vs:.1f}分)")
    else:
        signals.append(f"量能不足(量比{vol_ratio:.2f}x/需{min_vr}x)")
    # RSI / ATR 警示
    if (bullish_trend and h1_rsi > float(cfg.get('max_buy_rsi', 68.0))) or (bearish_trend and h1_rsi < float(cfg.get('min_sell_rsi', 32.0))):
        signals.append(f"RSI警示({'超买' if bullish_trend else '超卖'}: {h1_rsi:.1f})")
    if h4_atr_pct > float(cfg.get('max_h4_atr_pct', 6.5)): signals.append(f"4H波动偏大({h4_atr_pct:.2f}%)")
    # ⑦ 3m（bonus 6 分）
    if restart_up or restart_down:
        m3_confirm = (_tpr_three_min_restart(m3, bull_bias=bullish_trend, cfg=cfg) if len(m3) >= min_3m_bars
                      else _tpr_m3_default('3m数据不足'))
    else:
        m3_confirm = _tpr_m3_default('1H未穿越，跳过3m')
    if m3_confirm.get('passed'): score += _TPR_W_3M; signals.append(str(m3_confirm.get('reason')))
    elif m3_confirm.get('valid') and (restart_up or restart_down):
        signals.append(f"3m结构观察：{m3_confirm.get('reason')}")
    direction = 'WAIT'
    if restart_up and pullback_ok and (not require_3m or bool(m3_confirm.get('passed'))): direction = 'BUY'
    elif restart_down and pullback_ok and (not require_3m or bool(m3_confirm.get('passed'))): direction = 'SELL'
    tq = trend_long_score if bullish_trend else trend_short_score if bearish_trend else max(trend_long_score, trend_short_score)
    lq = max(20.0, 100.0 - abs(pullback_depth - 1.2) * 20.0); vq = min(vol_ratio / 1.25, 1.6) * 62.5
    fq = 94.0 if direction in {'BUY', 'SELL'} and pullback_depth <= 2.2 else 68.0 if direction in {'BUY', 'SELL'} else 34.0
    return {
        'valid': True, 'reason': '', 'score': max(score, 0.0), 'direction': direction, 'signals': signals,
        'ranking_factors': {'trend': tq, 'trigger': 92.0 if direction in {'BUY', 'SELL'} else 32.0,
                            'volume': vq, 'location': lq, 'freshness': fq,
                            'risk': 88.0 if h4_atr_pct <= 6.5 else 55.0},
        'details': {
            '评估': ' | '.join(signals) if signals else '暂无趋势回踩二次启动机会',
            '日线斜率': f'{d1_slope:.2f}%', '4H斜率': f'{h4_slope:.2f}%',
            '日线ADX': f'{d1_adx:.1f}', '4H_ADX': f'{h4_adx:.1f}',
            '日线EMA发散': f'{d1_ema_spread:+.2f}%', '4H_EMA发散': f'{h4_ema_spread:+.2f}%',
            '回踩距离': f'{pullback_depth:.2f}%', '关键位回测': f'{retest_quality:.2f}%',
            '量比': f'{vol_ratio:.2f}x', '量能Z分': f'{vol_zscore:+.2f}σ',
            '启动收盘强度': f'{close_strength:.2f}', '1H_RSI': f'{h1_rsi:.1f}', '4H_ATR%': f'{h4_atr_pct:.2f}%',
            '3m结构确认': '通过' if m3_confirm.get('passed') else '未通过',
            '3m当前阶段': f"Phase {m3_confirm.get('phase', 0)}",
            '3m结构说明': str(m3_confirm.get('reason', '-')),
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
# 增强节 A：共用工具函数（波动率压缩 / OBV / VWAP / 支撑汇聚 / K 线形态 / 风险）
# ══════════════════════════════════════════════════════════════════════════════

def _bb_bandwidth_series(close, n=20, k=2.0):
    """布林带带宽序列 (upper - lower) / mid。"""
    mid = close.rolling(n).mean()
    std = close.rolling(n).std(ddof=0)
    up = mid + k * std; lo = mid - k * std
    bw = (up - lo) / mid.replace(0, np.nan)
    return bw

def _donchian_bounds(df, n=20):
    """Donchian 通道，不含当前根。返回 (上轨, 下轨)。"""
    up = df['h'].rolling(n).max().shift(1)
    lo = df['l'].rolling(n).min().shift(1)
    return up, lo

def _vwap_rolling(df, n=48):
    """滚动 VWAP（近 n 根）。"""
    tp = (df['h'] + df['l'] + df['c']) / 3.0
    pv = (tp * df['vol']).rolling(n).sum()
    vs = df['vol'].rolling(n).sum().replace(0, np.nan)
    return pv / vs

def _obv_series(df):
    """OBV 序列。"""
    sign = np.sign(df['c'].diff().fillna(0.0))
    return (sign * df['vol']).cumsum()

def _hidden_bull_divergence(close, rsi, lookback=6):
    """隐藏多头背离：价格创近 lookback 新低，但 RSI 未创新低。"""
    if len(close) < lookback + 1 or len(rsi) < lookback + 1: return False
    window_c = close.iloc[-lookback:]; window_r = rsi.iloc[-lookback:]
    return float(window_c.iloc[-1]) <= float(window_c.min()) and float(window_r.iloc[-1]) > float(window_r.min()) * 1.002

def _hidden_bear_divergence(close, rsi, lookback=6):
    """隐藏空头背离：价格创新高但 RSI 未创新高。"""
    if len(close) < lookback + 1 or len(rsi) < lookback + 1: return False
    window_c = close.iloc[-lookback:]; window_r = rsi.iloc[-lookback:]
    return float(window_c.iloc[-1]) >= float(window_c.max()) and float(window_r.iloc[-1]) < float(window_r.max()) * 0.998

def _bullish_engulfing(df):
    """看涨吞没形态。"""
    if len(df) < 2: return False
    o1, c1 = float(df['o'].iloc[-2]), float(df['c'].iloc[-2])
    o2, c2 = float(df['o'].iloc[-1]), float(df['c'].iloc[-1])
    return c1 < o1 and c2 > o2 and o2 <= c1 and c2 >= o1

def _bearish_engulfing(df):
    if len(df) < 2: return False
    o1, c1 = float(df['o'].iloc[-2]), float(df['c'].iloc[-2])
    o2, c2 = float(df['o'].iloc[-1]), float(df['c'].iloc[-1])
    return c1 > o1 and c2 < o2 and o2 >= c1 and c2 <= o1

def _hammer_like(df, bull=True):
    """锤子线（bull=True）或倒锤/射击之星（bull=False）。"""
    if len(df) < 1: return False
    o, h, l, c = (float(df[k].iloc[-1]) for k in ('o','h','l','c'))
    rng = h - l
    if rng <= 0: return False
    body = abs(c - o); lower_sh = min(o, c) - l; upper_sh = h - max(o, c)
    if bull:
        return body / rng <= 0.35 and lower_sh >= body * 2.0 and upper_sh <= body * 0.6 and c >= o
    else:
        return body / rng <= 0.35 and upper_sh >= body * 2.0 and lower_sh <= body * 0.6 and c <= o

def _support_confluence(price, levels, tolerance_pct=0.8, side: str = 'any'):
    """
    统计 price 附近 tolerance_pct% 内的支撑/压力位汇聚，并按方向过滤：
      side='bull' → 只算价格 ≥ level - 小容差（价格在支撑上方或贴近）
      side='bear' → 只算价格 ≤ level + 小容差（价格在压力下方或贴近）
      side='any'  → 不过滤方向
    返回 (命中数, 命中列表[(name, level)])。
    """
    hits = []
    for name, v in levels.items():
        if v is None or not np.isfinite(v) or v <= 0: continue
        dist_pct = abs(price - v) / price * 100
        if dist_pct > tolerance_pct: continue
        if side == 'bull' and price < v * (1 - tolerance_pct / 100.0 * 0.5):
            # 价格已显著跌破该位，不能算"回踩企稳"的有效支撑
            continue
        if side == 'bear' and price > v * (1 + tolerance_pct / 100.0 * 0.5):
            continue
        hits.append((name, v))
    return len(hits), hits

def _fib_levels_from_swing(df, lookback=40):
    """从近 lookback 根取 swing high/low，返回 0.382/0.5/0.618 回撤位。"""
    if len(df) < lookback: return {}
    win = df.tail(lookback)
    hi = float(win['h'].max()); lo = float(win['l'].min())
    rng = hi - lo
    if rng <= 0: return {}
    return {'fib382': hi - rng * 0.382, 'fib500': hi - rng * 0.500, 'fib618': hi - rng * 0.618}

def _compute_risk_params(df_h1, direction, atr_value=None, rr_multiples=(1.5, 2.5)):
    """
    基于 ATR 与最近 swing 的入场/止损/目标参数。
    返回 {entry, stop_loss, target_1, target_2, rr_1, rr_2, atr}.
    """
    if direction not in {'BUY', 'SELL'} or df_h1 is None or len(df_h1) < 30:
        return {}
    entry = float(df_h1['c'].iloc[-1])
    atr = float(atr_value) if atr_value and atr_value > 0 else _atr(df_h1, 14)
    if atr <= 0: return {}
    recent = df_h1.tail(20)
    if direction == 'BUY':
        swing_low = float(recent['l'].min())
        stop = min(swing_low * 0.998, entry - atr * 1.2)
        risk = entry - stop
        if risk <= 0: return {}
        t1 = entry + risk * rr_multiples[0]
        t2 = entry + risk * rr_multiples[1]
    else:
        swing_high = float(recent['h'].max())
        stop = max(swing_high * 1.002, entry + atr * 1.2)
        risk = stop - entry
        if risk <= 0: return {}
        t1 = entry - risk * rr_multiples[0]
        t2 = entry - risk * rr_multiples[1]
    return {
        'entry': round(entry, 8), 'stop_loss': round(stop, 8),
        'target_1': round(t1, 8), 'target_2': round(t2, 8),
        'rr_1': float(rr_multiples[0]), 'rr_2': float(rr_multiples[1]),
        'atr': round(atr, 8), 'risk_pct': round(risk / entry * 100, 3),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 增强节 B：市场状态检测器（RegimeDetector）
# ══════════════════════════════════════════════════════════════════════════════

class RegimeDetector:
    """
    用 ADX + 布林带宽判定当前市场状态：
      - 'trend'    : ADX ≥ 22 → 适合跟随/回踩/缩量中继
      - 'range'    : ADX < 18 且 BBW 低分位 → 适合突破/背离
      - 'volatile' : BBW 高分位 → 降权或跳过
      - 'neutral'  : 其他
    """

    @staticmethod
    def detect(h4_df, h1_df):
        try:
            adx4 = _adx(h4_df, 14)
            bw_h1 = _bb_bandwidth_series(h1_df['c'], 20, 2.0).dropna()
            if len(bw_h1) < 50:
                return {'regime': 'neutral', 'adx_4h': adx4, 'bbw_rank': 0.5}
            cur_bw = float(bw_h1.iloc[-1])
            bw_rank = float((bw_h1.tail(50) < cur_bw).mean())  # 0~1
            if bw_rank >= 0.85:
                regime = 'volatile'
            elif adx4 >= 22.0:
                regime = 'trend'
            elif adx4 < 18.0 and bw_rank <= 0.35:
                regime = 'range'
            else:
                regime = 'neutral'
            return {'regime': regime, 'adx_4h': round(adx4, 2), 'bbw_rank': round(bw_rank, 3)}
        except Exception:
            return {'regime': 'neutral', 'adx_4h': 0.0, 'bbw_rank': 0.5}


# ══════════════════════════════════════════════════════════════════════════════
# 增强节 C：1H 早期突破扫描（EarlyBreakoutScanner）
# 核心逻辑：波动率压缩 → Donchian 突破 → OBV 领先 → 收盘强度 → 量能爆发
# ══════════════════════════════════════════════════════════════════════════════

_EBS_DEFAULT_CONFIG = {
    'min_score': 68,
    'min_volume_24h': 10_000_000,
    'bbw_lookback': 50,             # BBW 分位回溯
    'bbw_rank_max': 0.30,           # BBW 当前需 ≤ 过去 50 根 30 分位
    'donchian_n': 20,               # Donchian 通道长度
    'obv_lead_lookback': 20,        # OBV 领先回望
    'close_strength_min': 0.60,     # 收盘强度下限
    'vol_burst_vs_max': 1.10,       # 突破根量能 ≥ 前 20 根最大 × 1.1
    'max_24h_change_pct': 18.0,     # 24h 涨幅过滤，防追末段
    'min_h4_trend_align': 0.30,     # 4H EMA21 相对 EMA55 发散下限（同向）
}


class EarlyBreakoutScanner(_BASE_SCANNER_CLASS):
    required_bars = ['4H', '1H']
    name = "1H早期趋势突破扫描"
    description = "波动率压缩 + Donchian 突破 + OBV 领先 + 量能爆发，捕捉早期趋势启动"
    strategy_type = "scan"

    def __init__(self, config=None):
        self.config = {**_EBS_DEFAULT_CONFIG, **(config or {})}
        if _HAS_SCANNER_BASE and hasattr(super(), '__init__'):
            try: super().__init__(config or {})
            except Exception: pass

    def _init_conditions(self):
        if ScanCondition is None: return
        self.add_condition(ScanCondition(name="24H成交量", description="流动性过滤",
            field="volume_24h", operator=">=", value=self.config.get('min_volume_24h', 10_000_000)))

    def _get_klines(self, km, bar):
        return km.get(bar) or km.get(bar.lower()) or km.get(bar.upper()) or []

    def get_config_schema(self):
        return {
            'min_score':           {'type':'int',  'default':68,   'label':'最低通过分数'},
            'min_volume_24h':      {'type':'float','default':10_000_000, 'label':'最小24H成交额'},
            'bbw_rank_max':        {'type':'float','default':0.30, 'label':'BBW分位上限(越低越压缩)'},
            'donchian_n':          {'type':'int',  'default':20,   'label':'Donchian通道长度'},
            'close_strength_min':  {'type':'float','default':0.60, 'label':'收盘强度下限'},
            'vol_burst_vs_max':    {'type':'float','default':1.10, 'label':'突破根/前20根最大量'},
            'max_24h_change_pct':  {'type':'float','default':18.0, 'label':'24H涨幅上限%'},
        }

    def scan_symbol(self, symbol) -> dict:
        km = symbol.extra_data.get('klines', {})
        try:
            h4 = _to_df(self._get_klines(km, '4H'))
            h1 = _to_df(self._get_klines(km, '1H'))
            analysis = _ebs_analyze_core(h4, h1, symbol, self.config)
        except Exception as exc:
            return {'symbol': getattr(symbol, 'inst_id', ''), 'passed': False, 'score': 0.0,
                    'direction': 'WAIT', 'details': {'状态': f'分析异常: {exc}'}}
        if not analysis['valid']:
            return {'symbol': getattr(symbol, 'inst_id', ''), 'passed': False, 'score': 0.0,
                    'direction': 'WAIT', 'details': {'状态': analysis.get('reason', '')}}
        ms = float(self.config.get('min_score', 68))
        passed = analysis['score'] >= ms and analysis['direction'] in {'BUY', 'SELL'}
        risk = _compute_risk_params(h1, analysis['direction']) if passed else {}
        result = {
            'symbol': getattr(symbol, 'inst_id', ''), 'passed': passed,
            'score': round(analysis['score'], 2), 'direction': analysis['direction'],
            'signals': analysis['signals'], 'details': analysis['details'],
            'last_price': getattr(symbol, 'last_price', 0.0), 'volume_24h': getattr(symbol, 'volume_24h', 0.0),
            'price_change_24h': getattr(symbol, 'price_change_24h', 0.0),
            'category': '早期趋势突破', 'ranking_factors': analysis.get('ranking_factors', {}),
            'risk': risk,
        }
        if build_opportunity_profile:
            try:
                result.update(build_opportunity_profile(
                    base_score=analysis['score'], direction=analysis['direction'],
                    volume_24h=getattr(symbol, 'volume_24h', 0.0),
                    factors=analysis.get('ranking_factors', {}), signals=analysis['signals']))
            except Exception: pass
        return result


def _ebs_analyze_core(h4, h1, symbol, cfg):
    _check_df(h4, '4H', 60); _check_df(h1, '1H', 80)
    signals = []; score = 0.0
    chg24 = float(getattr(symbol, 'price_change_24h', 0) or 0)
    # — 趋势方向同步（4H EMA21 vs EMA55）
    ema21_4h = _ema(h4['c'], 21); ema55_4h = _ema(h4['c'], 55)
    spread_4h = (ema21_4h - ema55_4h) / ema55_4h * 100 if ema55_4h > 0 else 0.0
    min_align = float(cfg.get('min_h4_trend_align', 0.30))
    bull_align = spread_4h >= min_align; bear_align = spread_4h <= -min_align
    # — BBW 压缩
    bw = _bb_bandwidth_series(h1['c'], 20, 2.0).dropna()
    lookback = int(cfg.get('bbw_lookback', 50))
    if len(bw) < lookback:
        return {'valid': True, 'reason': 'BBW样本不足', 'score': 0.0, 'direction': 'WAIT',
                'signals': ['BBW样本不足'], 'details': {'状态': 'BBW样本不足'}, 'ranking_factors': {}}
    cur_bw = float(bw.iloc[-1]); bw_rank = float((bw.tail(lookback) < cur_bw).mean())
    bw_rank_max = float(cfg.get('bbw_rank_max', 0.30))
    # 用最近 5 根中的最低分位代表"压缩态"，当前 BBW 是否抬头判定"释放中"
    bw_rank_recent_min = float(min(
        (bw.tail(lookback).rank(pct=True).iloc[-k] for k in range(1, 6)),
        default=bw_rank))
    bw_expanding = len(bw) >= 4 and float(bw.iloc[-1]) > float(bw.iloc[-3]) > 0
    compressed = bw_rank_recent_min <= bw_rank_max  # 近期处于压缩
    if compressed:
        cs_ = 22.0 * (1.0 - bw_rank_recent_min / max(bw_rank_max, 0.05)); score += cs_
        signals.append(f"波动率压缩(近期最低分位{bw_rank_recent_min*100:.0f}% → +{cs_:.1f}分)")
    else:
        signals.append(f"未压缩(近期最低分位{bw_rank_recent_min*100:.0f}%)")
    if bw_expanding:
        score += 6.0; signals.append("BBW 抬头(压缩释放) → +6分")
    # — Donchian 突破
    dn = int(cfg.get('donchian_n', 20))
    dc_up, dc_lo = _donchian_bounds(h1, dn)
    last_c = float(h1['c'].iloc[-1]); last_o = float(h1['o'].iloc[-1])
    last_h = float(h1['h'].iloc[-1]); last_l = float(h1['l'].iloc[-1])
    up_level = float(dc_up.iloc[-1]) if not pd.isna(dc_up.iloc[-1]) else float('inf')
    lo_level = float(dc_lo.iloc[-1]) if not pd.isna(dc_lo.iloc[-1]) else 0.0
    # 前一根需仍在通道内，避免追陈旧延伸
    prev_c = float(h1['c'].iloc[-2]) if len(h1) >= 2 else last_c
    prev_up = float(dc_up.iloc[-2]) if len(dc_up) >= 2 and not pd.isna(dc_up.iloc[-2]) else up_level
    prev_lo = float(dc_lo.iloc[-2]) if len(dc_lo) >= 2 and not pd.isna(dc_lo.iloc[-2]) else lo_level
    fresh_up = prev_c <= prev_up * 1.001
    fresh_dn = prev_c >= prev_lo * 0.999
    breakout_up = last_c > up_level and bull_align and fresh_up
    breakout_dn = last_c < lo_level and bear_align and fresh_dn
    direction = 'BUY' if breakout_up else 'SELL' if breakout_dn else 'WAIT'
    if direction != 'WAIT':
        score += 24.0; signals.append(f"Donchian{dn}{'上突' if breakout_up else '下破'}(前根在通道内) → +24分")
    elif (last_c > up_level and bull_align and not fresh_up) or (last_c < lo_level and bear_align and not fresh_dn):
        signals.append("突破已延伸数根，跳过(非新鲜启动)")
    else:
        signals.append("未突破Donchian通道")
    # — 收盘强度
    rng = last_h - last_l
    if rng > 0:
        if breakout_up: cs_raw = (last_c - last_l) / rng
        elif breakout_dn: cs_raw = (last_h - last_c) / rng
        else: cs_raw = 0.5
    else:
        cs_raw = 0.5
    min_cs = float(cfg.get('close_strength_min', 0.60))
    if direction != 'WAIT' and cs_raw >= min_cs:
        ss = 14.0 * min(1.0, (cs_raw - min_cs) / (1 - min_cs) * 0.5 + 0.5)
        score += ss; signals.append(f"收盘强度{cs_raw:.2f} → +{ss:.1f}分")
    elif direction != 'WAIT':
        signals.append(f"收盘强度不足({cs_raw:.2f})")
    # — OBV 领先
    obv = _obv_series(h1)
    obv_lb = int(cfg.get('obv_lead_lookback', 20))
    if len(obv) >= obv_lb + 2:
        prev_obv_max = float(obv.iloc[-(obv_lb+1):-1].max())
        prev_obv_min = float(obv.iloc[-(obv_lb+1):-1].min())
        cur_obv = float(obv.iloc[-1]); prev_obv = float(obv.iloc[-2])
        # 用绝对范围 0.5% 作为容差，避免对负 OBV 用乘性容差翻向
        obv_band = max(abs(prev_obv_max), abs(prev_obv_min), 1.0) * 0.005
        obv_lead_up = direction == 'BUY' and prev_obv >= prev_obv_max - obv_band
        obv_lead_dn = direction == 'SELL' and prev_obv <= prev_obv_min + obv_band
        confirm_up = direction == 'BUY' and cur_obv > prev_obv_max
        confirm_dn = direction == 'SELL' and cur_obv < prev_obv_min
        if obv_lead_up or obv_lead_dn: score += 12.0; signals.append("OBV领先(前根已近新高/低) → +12分")
        if confirm_up or confirm_dn: score += 6.0; signals.append("OBV同步确认 → +6分")
    # — 量能爆发
    vol_last = float(h1['vol'].iloc[-1]); vol_prev_max = float(h1['vol'].iloc[-21:-1].max())
    vol_burst_th = float(cfg.get('vol_burst_vs_max', 1.10))
    if vol_prev_max > 0 and vol_last >= vol_prev_max * vol_burst_th:
        ratio = vol_last / vol_prev_max
        vs = 14.0 * min(1.0, (ratio - 1.0) / 1.0 + 0.6)
        score += vs; signals.append(f"量能爆发({ratio:.2f}x近20根最大 → +{vs:.1f}分)")
    elif direction != 'WAIT':
        signals.append(f"量能未爆发({vol_last/max(vol_prev_max,1):.2f}x)")
    # — 追末段过滤
    max_chg = float(cfg.get('max_24h_change_pct', 18.0))
    if direction == 'BUY' and chg24 > max_chg:
        score -= 10.0; signals.append(f"24H已涨{chg24:+.1f}% → -10分(追末段警示)")
    if direction == 'SELL' and chg24 < -max_chg:
        score -= 10.0; signals.append(f"24H已跌{chg24:+.1f}% → -10分(追末段警示)")
    # — 方向对齐 bonus（4H 对齐强度）
    align_abs = abs(spread_4h)
    if direction != 'WAIT' and align_abs >= min_align:
        ab = min(8.0, align_abs / 2.0 * 8.0); score += ab
        signals.append(f"4H方向对齐(发散{align_abs:.2f}% → +{ab:.1f}分)")
    # 最终 direction 收敛
    if not compressed: direction = 'WAIT'  # 必须压缩
    if direction != 'WAIT' and cs_raw < min_cs: direction = 'WAIT'  # 必须有强度
    rf = {
        'trend': min(100.0, 50 + align_abs * 15),
        'trigger': 92.0 if direction in {'BUY','SELL'} else 30.0,
        'volume': min(100.0, (vol_last / max(vol_prev_max, 1)) * 55) if vol_prev_max else 40.0,
        'location': max(20.0, 100.0 - bw_rank * 100),
        'freshness': 90.0 if compressed and direction != 'WAIT' else 45.0,
        'risk': 82.0 if direction != 'WAIT' else 50.0,
    }
    return {
        'valid': True, 'reason': '', 'score': max(score, 0.0),
        'direction': direction, 'signals': signals, 'ranking_factors': rf,
        'details': {
            '评估': ' | '.join(signals), 'BBW分位': f'{bw_rank*100:.0f}%',
            'Donchian上轨': f'{up_level:.6g}', 'Donchian下轨': f'{lo_level:.6g}',
            '收盘强度': f'{cs_raw:.2f}',
            '量能比(对近20根最大)': f'{vol_last/max(vol_prev_max,1):.2f}x',
            '4H_EMA发散': f'{spread_4h:+.2f}%', '24H涨跌': f'{chg24:+.2f}%',
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
# 增强节 D：1H 回调企稳扫描（PullbackStabilizationScanner）
# 核心逻辑：支撑区汇聚 + 缩量止跌 + RSI 隐藏背离 + 反包 K 线 + 3m 三段式确认
# ══════════════════════════════════════════════════════════════════════════════

_PSS_DEFAULT_CONFIG = {
    'min_score': 68,
    'min_volume_24h': 10_000_000,
    'confluence_tolerance_pct': 0.80,   # 支撑区汇聚容差
    'min_confluence_hits': 2,           # 至少 2 个位置汇聚
    'contraction_max_ratio': 0.85,      # 回踩末段量 / 整体回踩中位量
    'hidden_div_lookback': 6,
    'require_3m': False,                # 3m 确认作为 bonus，非硬要求
    'require_h4_trend': True,           # 4H 趋势仍存
    'h4_adx_min': 15.0,                 # 4H ADX 软门槛（比原策略 18 低）
    'max_pullback_depth_atr': 3.0,      # 最大回踩深度（ATR 倍数，过深= 趋势破坏）
    'min_pullback_depth_atr': 0.6,      # 最小回踩深度（ATR 倍数，过浅= 没真回调）
}


class PullbackStabilizationScanner(_BASE_SCANNER_CLASS):
    required_bars = ['4H', '1H', '3m']
    name = "1H回调企稳扫描"
    description = "支撑汇聚 + 缩量止跌 + 隐藏背离 + 反包 K 线 + 3m 触发，捕捉趋势回调买点"
    strategy_type = "scan"

    def __init__(self, config=None):
        self.config = {**_PSS_DEFAULT_CONFIG, **(config or {})}
        if _HAS_SCANNER_BASE and hasattr(super(), '__init__'):
            try: super().__init__(config or {})
            except Exception: pass

    def _init_conditions(self):
        if ScanCondition is None: return
        self.add_condition(ScanCondition(name="24H成交量", description="流动性过滤",
            field="volume_24h", operator=">=", value=self.config.get('min_volume_24h', 10_000_000)))

    def _get_klines(self, km, bar):
        return km.get(bar) or km.get(bar.lower()) or km.get(bar.upper()) or []

    def get_config_schema(self):
        return {
            'min_score':                {'type':'int',  'default':68,   'label':'最低通过分数'},
            'min_volume_24h':           {'type':'float','default':10_000_000, 'label':'最小24H成交额'},
            'confluence_tolerance_pct': {'type':'float','default':0.80, 'label':'支撑区汇聚容差%'},
            'min_confluence_hits':      {'type':'int',  'default':2,    'label':'最少汇聚位数'},
            'contraction_max_ratio':    {'type':'float','default':0.85, 'label':'末段/整体缩量比'},
            'require_3m':               {'type':'bool', 'default':False,'label':'强制3m确认'},
            'h4_adx_min':               {'type':'float','default':15.0, 'label':'4H ADX软门槛'},
        }

    def scan_symbol(self, symbol) -> dict:
        km = symbol.extra_data.get('klines', {})
        try:
            h4 = _to_df(self._get_klines(km, '4H'))
            h1 = _to_df(self._get_klines(km, '1H'))
            m3_raw = self._get_klines(km, '3m')
            m3 = _to_df(m3_raw) if m3_raw else None
            analysis = _pss_analyze_core(h4, h1, m3, getattr(symbol, 'last_price', 0.0), self.config)
        except Exception as exc:
            return {'symbol': getattr(symbol, 'inst_id', ''), 'passed': False, 'score': 0.0,
                    'direction': 'WAIT', 'details': {'状态': f'分析异常: {exc}'}}
        if not analysis['valid']:
            return {'symbol': getattr(symbol, 'inst_id', ''), 'passed': False, 'score': 0.0,
                    'direction': 'WAIT', 'details': {'状态': analysis.get('reason', '')}}
        ms = float(self.config.get('min_score', 68))
        passed = analysis['score'] >= ms and analysis['direction'] in {'BUY', 'SELL'}
        risk = _compute_risk_params(h1, analysis['direction']) if passed else {}
        result = {
            'symbol': getattr(symbol, 'inst_id', ''), 'passed': passed,
            'score': round(analysis['score'], 2), 'direction': analysis['direction'],
            'signals': analysis['signals'], 'details': analysis['details'],
            'last_price': getattr(symbol, 'last_price', 0.0), 'volume_24h': getattr(symbol, 'volume_24h', 0.0),
            'price_change_24h': getattr(symbol, 'price_change_24h', 0.0),
            'category': '回调企稳', 'ranking_factors': analysis.get('ranking_factors', {}),
            'risk': risk,
        }
        if build_opportunity_profile:
            try:
                result.update(build_opportunity_profile(
                    base_score=analysis['score'], direction=analysis['direction'],
                    volume_24h=getattr(symbol, 'volume_24h', 0.0),
                    factors=analysis.get('ranking_factors', {}), signals=analysis['signals']))
            except Exception: pass
        return result


def _pss_analyze_core(h4, h1, m3, last_price, cfg):
    _check_df(h4, '4H', 60); _check_df(h1, '1H', 60)
    signals = []; score = 0.0
    price = float(last_price) if last_price and last_price > 0 else float(h1['c'].iloc[-1])
    # — 4H 趋势软门槛 + 用 _local_trend_snapshot 复合打分（与其他模块一致）
    ema21_4h = _ema(h4['c'], 21); ema55_4h = _ema(h4['c'], 55)
    spread_4h = (ema21_4h - ema55_4h) / ema55_4h * 100 if ema55_4h > 0 else 0.0
    h4_adx = _adx(h4, 14); h1_atr = _atr(h1)
    h4_adx_min = float(cfg.get('h4_adx_min', 15.0))
    # _local_trend_snapshot 需要 1D，若缺失则用 4H 复用做近似
    try:
        d1_for_snap = _aggregate_bars(h4, 6) if len(h4) >= 60 else h4
        snap = _local_trend_snapshot(d1_for_snap, h4, h1, price)
    except Exception:
        snap = {'long_score': 50.0, 'short_score': 50.0, 'long_ok': False, 'short_ok': False, 'metrics': {}, 'reason': ''}
    long_score = float(snap.get('long_score', 0.0) or 0.0)
    short_score = float(snap.get('short_score', 0.0) or 0.0)
    bull_trend = spread_4h > 0.25 and h4_adx >= h4_adx_min and long_score >= short_score
    bear_trend = spread_4h < -0.25 and h4_adx >= h4_adx_min and short_score > long_score
    if not (bull_trend or bear_trend):
        return {'valid': True, 'reason': '', 'score': 0.0, 'direction': 'WAIT',
                'signals': [f'4H趋势弱(spread={spread_4h:+.2f}%, ADX={h4_adx:.1f}, 趋势分L{long_score:.0f}/S{short_score:.0f})'],
                'details': {'状态': '无有效趋势', '趋势分': f'L{long_score:.0f}/S{short_score:.0f}'},
                'ranking_factors': {'trend':30,'trigger':0,'volume':0,'location':50,'freshness':30,'risk':55}}
    is_bull = bull_trend
    tref = long_score if is_bull else short_score
    signals.append(f"4H{'多头' if is_bull else '空头'}趋势(质量{tref:.0f}, spread={spread_4h:+.2f}%, ADX={h4_adx:.1f})")
    # 趋势基础分按质量缩放
    score += min(18.0, 8.0 + max(0.0, (tref - 50.0) / 50.0 * 10.0))
    # — 支撑/压力位汇聚（EMA21 / EMA55 / swing / fib / VWAP）
    ema21_1h = _ema(h1['c'], 21); ema55_1h = _ema(h1['c'], 55)
    sh, sl = _latest_swing_levels(h4, left=5, right=5, skip_recent=3, max_lookback=80)
    fibs = _fib_levels_from_swing(h1, lookback=40)
    vwap = _vwap_rolling(h1, 48)
    vwap_val = float(vwap.iloc[-1]) if len(vwap.dropna()) else None
    if is_bull:
        levels = {'1H_EMA21': ema21_1h, '1H_EMA55': ema55_1h,
                  'swing_low': sl, 'fib382': fibs.get('fib382'), 'fib500': fibs.get('fib500'),
                  'fib618': fibs.get('fib618'), 'vwap': vwap_val}
    else:
        levels = {'1H_EMA21': ema21_1h, '1H_EMA55': ema55_1h,
                  'swing_high': sh, 'fib382': fibs.get('fib382'), 'fib500': fibs.get('fib500'),
                  'fib618': fibs.get('fib618'), 'vwap': vwap_val}
    tol = float(cfg.get('confluence_tolerance_pct', 0.80))
    hits_n, hits = _support_confluence(price, levels, tol, side=('bull' if is_bull else 'bear'))
    min_hits = int(cfg.get('min_confluence_hits', 2))
    if hits_n >= min_hits:
        cs_ = min(22.0, 8.0 + hits_n * 5.0); score += cs_
        sig_names = ','.join(n for n, _ in hits[:4])
        signals.append(f"支撑汇聚{hits_n}位[{sig_names}] → +{cs_:.1f}分")
    else:
        signals.append(f"支撑汇聚不足({hits_n}位)")
    # — 回踩深度（不能过深）
    recent_hi = float(h1['h'].tail(20).max()); recent_lo = float(h1['l'].tail(20).min())
    if is_bull:
        pullback_atr = (recent_hi - price) / h1_atr if h1_atr > 0 else 0.0
    else:
        pullback_atr = (price - recent_lo) / h1_atr if h1_atr > 0 else 0.0
    max_pb = float(cfg.get('max_pullback_depth_atr', 3.0))
    min_pb = float(cfg.get('min_pullback_depth_atr', 0.6))
    if pullback_atr > max_pb:
        score -= 8.0; signals.append(f"回踩过深({pullback_atr:.2f}ATR>{max_pb}) → -8分")
    elif pullback_atr < min_pb:
        # 真回调必须有实际幅度：没回调过就不是"企稳"
        score -= 6.0; signals.append(f"未见真实回踩({pullback_atr:.2f}ATR<{min_pb}) → -6分")
        _no_real_pullback = True
    else:
        _no_real_pullback = False
    # — 缩量止跌
    if is_bull:
        # 找最近 swing high 之后的回踩段
        swing_hi_idx = int(h1['h'].tail(20).idxmax())
        pb_slice = h1.loc[swing_hi_idx:]
    else:
        swing_lo_idx = int(h1['l'].tail(20).idxmin())
        pb_slice = h1.loc[swing_lo_idx:]
    if len(pb_slice) >= 4:
        whole_med = float(pb_slice['vol'].median())
        recent3_med = float(pb_slice['vol'].tail(3).median())
        ratio = recent3_med / whole_med if whole_med > 0 else 1.0
        max_r = float(cfg.get('contraction_max_ratio', 0.85))
        if ratio <= max_r:
            vs = 16.0 * min(1.0, (max_r - ratio) / max(max_r - 0.3, 0.1) * 0.6 + 0.4)
            score += vs; signals.append(f"缩量止跌({ratio:.2f}x → +{vs:.1f}分)")
        else:
            signals.append(f"未见缩量({ratio:.2f}x)")
    # — RSI 隐藏背离（用完整 RSI 序列）
    delta = h1['c'].diff()
    gain = delta.where(delta > 0, 0.0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0.0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan); rsi_full = 100 - 100 / (1 + rs)
    lb = int(cfg.get('hidden_div_lookback', 6))
    if is_bull and _hidden_bull_divergence(h1['c'], rsi_full, lb):
        score += 16.0; signals.append("RSI隐藏多头背离 → +16分")
    elif not is_bull and _hidden_bear_divergence(h1['c'], rsi_full, lb):
        score += 16.0; signals.append("RSI隐藏空头背离 → +16分")
    # — 反包 / 锤子 K 线
    if is_bull and (_bullish_engulfing(h1) or _hammer_like(h1, bull=True)):
        score += 14.0; signals.append("看涨反包/锤子线 → +14分")
    elif (not is_bull) and (_bearish_engulfing(h1) or _hammer_like(h1, bull=False)):
        score += 14.0; signals.append("看跌反包/射击之星 → +14分")
    # — MACD 动量转折 bonus（柱体连升/连降，允许仍为负/正）
    _, _, hist = _macd(h1['c'])
    if len(hist) >= 3:
        hm2, hm1, h0 = float(hist.iloc[-3]), float(hist.iloc[-2]), float(hist.iloc[-1])
        turn_up = hm2 < hm1 < h0
        turn_dn = hm2 > hm1 > h0
        if is_bull and turn_up: score += 6.0; signals.append("MACD柱体连升 → +6分")
        elif (not is_bull) and turn_dn: score += 6.0; signals.append("MACD柱体连降 → +6分")
    # — 3m 三段式确认（作为 bonus；若开启 require_3m 则必需）
    m3_ok = False; m3_reason = '未检测'
    if m3 is not None and len(m3) >= 30:
        try:
            m3_res = _tpr_three_min_restart(m3, bull_bias=is_bull, cfg={'m3_swing_window':60,'m3_neckline_bars':8,
                                                                         'm3_min_pullback_bars':3,'m3_breakout_buffer_pct':0.12,
                                                                         'm3_origin_tolerance_pct':0.15})
            m3_ok = bool(m3_res.get('passed')); m3_reason = str(m3_res.get('reason', ''))
        except Exception as e:
            m3_reason = f'3m异常:{e}'
    if m3_ok: score += 10.0; signals.append(f"3m三段确认 → +10分")
    elif bool(cfg.get('require_3m', False)):
        signals.append(f"3m未确认({m3_reason})")
    # — 方向收敛
    direction = 'WAIT'
    must_pass = hits_n >= min_hits and not _no_real_pullback
    if bool(cfg.get('require_3m', False)): must_pass = must_pass and m3_ok
    if must_pass:
        direction = 'BUY' if is_bull else 'SELL'
    rf = {
        'trend': min(100.0, 40 + abs(spread_4h) * 15 + (h4_adx - 10) * 1.5),
        'trigger': 88.0 if direction in {'BUY','SELL'} else 32.0,
        'volume': 75.0,
        'location': min(100.0, 40 + hits_n * 18),
        'freshness': 80.0 if direction in {'BUY','SELL'} else 45.0,
        'risk': 85.0 if pullback_atr <= max_pb else 45.0,
    }
    return {
        'valid': True, 'reason': '', 'score': max(score, 0.0),
        'direction': direction, 'signals': signals, 'ranking_factors': rf,
        'details': {
            '评估': ' | '.join(signals),
            '趋势方向': '多头' if is_bull else '空头',
            '4H_EMA发散': f'{spread_4h:+.2f}%', '4H_ADX': f'{h4_adx:.1f}',
            '支撑汇聚位数': hits_n, '汇聚明细': [n for n,_ in hits],
            '回踩深度ATR': f'{pullback_atr:.2f}', '3m确认': '通过' if m3_ok else m3_reason,
        },
    }


def _rsi_wilder_series_last(s):
    """占位：保留供未来使用。"""
    return np.nan


# ══════════════════════════════════════════════════════════════════════════════
# 最终节：组合扫描器主类（ComboSwingScanner）
# ══════════════════════════════════════════════════════════════════════════════

# 子策略注册表（name → class）
_STRATEGY_REGISTRY = {
    'breakout':          BreakoutSwingScanner,
    'new_high':          NewHighBreakoutScanner,
    'trend_follow':      DirectionalTrendFollowScanner,
    'divergence':        DivergenceReversalScanner,
    'trend_pullback':    TrendPullbackSwingScanner,
    'oversold_reversal': OversoldReversalSwingScanner,
    'compression':       ContinuationCompressionSwingScanner,
    'vol_price_div':     VolumePriceDivergenceScanner,
    'restart':           TrendPullbackRestartScanner,
    'early_breakout':    EarlyBreakoutScanner,
    'pullback_stab':     PullbackStabilizationScanner,
}

# Regime → 推荐子策略权重（相对权重，>1 加强，<1 降权，0 禁用）
_REGIME_WEIGHTS = {
    'trend':    {'trend_follow':1.2, 'trend_pullback':1.2, 'restart':1.2, 'pullback_stab':1.3,
                 'compression':1.1, 'early_breakout':0.9, 'breakout':0.9, 'new_high':1.0,
                 'divergence':0.7, 'oversold_reversal':0.7, 'vol_price_div':0.8},
    'range':   {'breakout':1.3, 'early_breakout':1.3, 'divergence':1.2, 'oversold_reversal':1.1,
                'vol_price_div':1.1, 'new_high':1.0, 'compression':1.0, 'trend_follow':0.6,
                'trend_pullback':0.7, 'restart':0.7, 'pullback_stab':0.8},
    'volatile':{'divergence':1.1, 'oversold_reversal':1.0, 'vol_price_div':1.1,
                'early_breakout':0.5, 'breakout':0.6, 'new_high':0.6, 'trend_follow':0.5,
                'trend_pullback':0.6, 'restart':0.5, 'pullback_stab':0.6, 'compression':0.6},
    'neutral': {k: 1.0 for k in ['trend_follow','trend_pullback','restart','pullback_stab',
                                 'compression','early_breakout','breakout','new_high',
                                 'divergence','oversold_reversal','vol_price_div']},
}

_ALL_STRATEGY_NAMES = list(_STRATEGY_REGISTRY.keys())


class ComboSwingScanner:
    """
    组合扫描器：并行调用所有（或指定子集）子策略，汇总 passed 结果。

    用法示例：
        scanner = ComboSwingScanner(strategies=['breakout', 'new_high'], config={})
        result  = scanner.scan_symbol(symbol)
        results = scanner.scan_all(symbols)
    """
    name          = "波段八策略组合扫描器"
    description   = "平台突破 / 新高突破 / 趋势跟随 / 背离反转 / 趋势回踩 / 超跌反转 / 缩量中继 / 量价背离 / 二次启动"
    strategy_type = "scan"

    def __init__(self, strategies=None, config=None, use_regime=True, max_workers: int = 8):
        cfg = config or {}
        if strategies is None:
            strategies = _ALL_STRATEGY_NAMES
        self._scanners = {}
        for key in strategies:
            cls = _STRATEGY_REGISTRY.get(key)
            if cls is None:
                raise ValueError(f"未知子策略 key: {key!r}，可用: {_ALL_STRATEGY_NAMES}")
            sub_cfg = cfg.get(key, cfg)
            inst = cls(sub_cfg)
            # 若运行环境提供 lifecycle guard，统一应用一次
            if _HAS_LIFECYCLE and apply_strategy_lifecycle_guard is not None:
                try: apply_strategy_lifecycle_guard(inst)
                except Exception: pass
            self._scanners[key] = inst
        self.use_regime = bool(use_regime)
        self.max_workers = max(1, int(max_workers))

    def _detect_regime(self, symbol):
        """检测市场状态。失败返回 neutral。"""
        try:
            km = symbol.extra_data.get('klines', {}) if isinstance(symbol.extra_data, dict) else {}
            h4 = _to_df(km.get('4H') or km.get('4h') or [])
            h1 = _to_df(km.get('1H') or km.get('1h') or [])
            if len(h4) < 40 or len(h1) < 60:
                return {'regime': 'neutral', 'adx_4h': 0.0, 'bbw_rank': 0.5}
            return RegimeDetector.detect(h4, h1)
        except Exception:
            return {'regime': 'neutral', 'adx_4h': 0.0, 'bbw_rank': 0.5}

    def scan_symbol(self, symbol) -> dict:
        """对单个 symbol 运行所有子策略，返回汇总结果（带 regime 权重 + risk 参数）。"""
        regime_info = self._detect_regime(symbol) if self.use_regime else {'regime': 'neutral'}
        regime = regime_info['regime']
        weights = _REGIME_WEIGHTS.get(regime, _REGIME_WEIGHTS['neutral'])

        all_results = {}; best_score = -1.0; best_key = None; passed_list = []
        for key, scanner in self._scanners.items():
            try:
                r = scanner.scan_symbol(symbol)
            except Exception as exc:
                r = {'passed': False, 'score': 0.0, 'direction': 'WAIT',
                     'details': {'状态': f'{key} 异常: {exc}'}, 'signals': []}
            # regime 加权
            w = float(weights.get(key, 1.0))
            r['score_raw'] = float(r.get('score', 0) or 0)
            r['score_weighted'] = r['score_raw'] * w
            r['regime_weight'] = w
            all_results[key] = r
            if r.get('passed') and w > 0.5:
                passed_list.append(key)
            if r['score_weighted'] > best_score:
                best_score = r['score_weighted']; best_key = key

        if not passed_list:
            return {
                'symbol': getattr(symbol, 'inst_id', ''),
                'passed': False, 'score': best_score, 'direction': 'WAIT', 'signals': [],
                'category': '未通过任何策略', 'regime': regime_info,
                'details': {'通过策略': '无', '最高分策略': best_key,
                            '最高分(加权)': f'{best_score:.2f}', '市场状态': regime},
                'sub_results': all_results,
            }

        best_passed_key = max(passed_list, key=lambda k: all_results[k]['score_weighted'])
        best_r = all_results[best_passed_key]
        # 合成多策略 signals（最多 3 条顶级策略）
        top3 = sorted(passed_list, key=lambda k: all_results[k]['score_weighted'], reverse=True)[:3]
        consensus_dirs = [all_results[k].get('direction') for k in top3]
        # 真正按多数投票（≥2 同向）取共识，否则回退到最高分策略方向
        from collections import Counter as _Counter
        _dir_cnt = _Counter(d for d in consensus_dirs if d in {'BUY', 'SELL'})
        if _dir_cnt and _dir_cnt.most_common(1)[0][1] >= 2:
            consensus = _dir_cnt.most_common(1)[0][0]
        else:
            consensus = best_r.get('direction')
        # risk：优先用子策略已计算，否则从 1H K 线现算
        risk = best_r.get('risk') or {}
        if not risk:
            try:
                km = symbol.extra_data.get('klines', {})
                h1 = _to_df(km.get('1H') or km.get('1h') or [])
                risk = _compute_risk_params(h1, consensus) if len(h1) >= 30 else {}
            except Exception:
                risk = {}
        return {
            'symbol':        getattr(symbol, 'inst_id', ''),
            'passed':        True,
            'score':         round(best_r['score_weighted'], 2),
            'score_raw':     round(best_r['score_raw'], 2),
            'direction':     consensus,
            'signals':       best_r.get('signals', []),
            'category':      best_r.get('category', best_passed_key),
            'details':       best_r.get('details', {}),
            'ranking_factors': best_r.get('ranking_factors', {}),
            'risk':          risk,
            'regime':        regime_info,
            'last_price':    getattr(symbol, 'last_price', 0),
            'volume_24h':    getattr(symbol, 'volume_24h', 0),
            'price_change_24h': getattr(symbol, 'price_change_24h', 0),
            'passed_strategies': passed_list,
            'best_strategy': best_passed_key,
            'consensus_count': consensus_dirs.count(consensus),
            'sub_results':   all_results,
        }

    def scan_all(self, symbols, top_n=30) -> dict:
        """批量扫描（并行），返回 passed 结果列表（按分数降序）。"""
        passed = []
        total = len(symbols)
        if self.max_workers > 1 and total > 1:
            with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
                futs = {ex.submit(self.scan_symbol, s): s for s in symbols}
                for fu in as_completed(futs):
                    try:
                        r = fu.result()
                    except Exception:
                        continue
                    if r.get('passed'):
                        passed.append(r)
        else:
            for sym in symbols:
                try:
                    r = self.scan_symbol(sym)
                except Exception:
                    continue
                if r.get('passed'):
                    passed.append(r)
        passed.sort(key=lambda x: float(x.get('score', 0) or 0), reverse=True)
        return {
            'type':             'combo_swing',
            'all_opportunities': passed[:top_n],
            'total_passed':     len(passed),
            'total_scanned':    total,
            'strategies_used':  list(self._scanners.keys()),
        }

    def get_config_schema(self) -> dict:
        schema = {}
        for key, scanner in self._scanners.items():
            if hasattr(scanner, 'get_config_schema'):
                schema[key] = scanner.get_config_schema()
        return schema


# ══════════════════════════════════════════════════════════════════════════════
# 模块级导出
# ══════════════════════════════════════════════════════════════════════════════
STRATEGY_NAME  = "波段八策略组合扫描器"
STRATEGY_TYPE  = "scan"
STRATEGY_CLASS = ComboSwingScanner

__all__ = [
    # 组合扫描器（主入口）
    'ComboSwingScanner',
    # 各子策略
    'BreakoutSwingScanner',
    'NewHighBreakoutScanner',
    'DirectionalTrendFollowScanner',
    'DivergenceReversalScanner',
    'TrendPullbackSwingScanner',
    'OversoldReversalSwingScanner',
    'ContinuationCompressionSwingScanner',
    'VolumePriceDivergenceScanner',
    'TrendPullbackRestartScanner',
    'EarlyBreakoutScanner',
    'PullbackStabilizationScanner',
    # 增强模块
    'RegimeDetector',
    # 注册表
    '_STRATEGY_REGISTRY',
    '_ALL_STRATEGY_NAMES',
    '_REGIME_WEIGHTS',
    # 模块元信息
    'STRATEGY_NAME', 'STRATEGY_TYPE', 'STRATEGY_CLASS',
]
