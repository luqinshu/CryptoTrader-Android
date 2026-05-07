"""
多周期趋势质量评分工具。

比单纯 EMA 排列更稳健：同时评估均线结构、斜率、ADX、价格结构、
趋势效率和 ATR 延伸度，用于降低震荡市/假突破里的趋势误判。
"""

from typing import Dict

import pandas as pd


def _clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, float(value)))


def ema_slope_pct(close: pd.Series, span: int = 20, lookback: int = 6) -> float:
    if close is None or len(close) <= span + lookback:
        return 0.0
    ema = close.ewm(span=span, adjust=False).mean()
    base = float(ema.iloc[-(lookback + 1)])
    latest = float(ema.iloc[-1])
    return ((latest / base) - 1.0) * 100.0 if base > 0 else 0.0


def _col(df: pd.DataFrame, *names: str):
    """兼容 'h'/'l'/'c' 和 'high'/'low'/'close' 两种列名。"""
    for n in names:
        if n in df.columns:
            return df[n]
    raise KeyError(f"DataFrame 缺少列 {names[0]}，可用列: {list(df.columns)}")


def atr_pct(df: pd.DataFrame, period: int = 14) -> float:
    if df is None or len(df) < period + 1:
        return 0.0
    high = _col(df, 'h', 'high')
    low = _col(df, 'l', 'low')
    close = _col(df, 'c', 'close')
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(period).mean().iloc[-1]
    last = float(close.iloc[-1])
    return float(atr / last * 100.0) if last > 0 and pd.notna(atr) else 0.0


def adx(df: pd.DataFrame, period: int = 14) -> float:
    if df is None or len(df) < period * 2 + 2:
        return 0.0
    high = _col(df, 'h', 'high').astype(float)
    low = _col(df, 'l', 'low').astype(float)
    close = _col(df, 'c', 'close').astype(float)
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    plus_di = 100 * plus_dm.rolling(period).mean() / atr.replace(0, pd.NA)
    minus_di = 100 * minus_dm.rolling(period).mean() / atr.replace(0, pd.NA)
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, pd.NA)) * 100
    value = dx.rolling(period).mean().iloc[-1]
    return float(value) if pd.notna(value) else 0.0


def structure_score(df: pd.DataFrame, direction: str, window: int = 8) -> float:
    if df is None or len(df) < window * 2 + 2:
        return 50.0
    recent = df.tail(window)
    previous = df.iloc[-window * 2:-window]
    recent_high = float(_col(recent, 'h', 'high').max())
    recent_low = float(_col(recent, 'l', 'low').min())
    previous_high = float(_col(previous, 'h', 'high').max())
    previous_low = float(_col(previous, 'l', 'low').min())
    if direction == 'BUY':
        score = 50.0
        score += 25.0 if recent_high > previous_high else -18.0
        score += 25.0 if recent_low > previous_low else -18.0
        return _clamp(score)
    if direction == 'SELL':
        score = 50.0
        score += 25.0 if recent_low < previous_low else -18.0
        score += 25.0 if recent_high < previous_high else -18.0
        return _clamp(score)
    return 50.0


def trend_efficiency(close: pd.Series, lookback: int = 24) -> float:
    if close is None or len(close) <= lookback:
        return 0.0
    section = close.tail(lookback + 1).astype(float)
    net = abs(float(section.iloc[-1] - section.iloc[0]))
    path = float(section.diff().abs().sum())
    return _clamp((net / path) * 100.0 if path > 0 else 0.0)


def trend_quality_snapshot(d1: pd.DataFrame, h4: pd.DataFrame, h1: pd.DataFrame, price: float) -> Dict[str, object]:
    """返回多空两侧趋势质量评分和关键诊断。"""
    result = {
        'long_score': 0.0,
        'short_score': 0.0,
        'long_ok': False,
        'short_ok': False,
        'reason': '数据不足',
        'metrics': {},
    }
    if any(df is None or df.empty for df in (d1, h4, h1)):
        return result
    if len(d1) < 60 or len(h4) < 80 or len(h1) < 80:
        return result

    d1_close = _col(d1, 'c', 'close').astype(float)
    h4_close = _col(h4, 'c', 'close').astype(float)
    h1_close = _col(h1, 'c', 'close').astype(float)
    price = float(price if price > 0 else h1_close.iloc[-1])

    d1_ema20 = float(d1_close.ewm(span=20, adjust=False).mean().iloc[-1])
    d1_ema50 = float(d1_close.ewm(span=50, adjust=False).mean().iloc[-1])
    h4_ema20 = float(h4_close.ewm(span=20, adjust=False).mean().iloc[-1])
    h4_ema50 = float(h4_close.ewm(span=50, adjust=False).mean().iloc[-1])
    h1_ema20 = float(h1_close.ewm(span=20, adjust=False).mean().iloc[-1])
    h1_ema50 = float(h1_close.ewm(span=50, adjust=False).mean().iloc[-1])

    d1_slope = ema_slope_pct(d1_close, 20, 6)
    h4_slope = ema_slope_pct(h4_close, 20, 6)
    h1_slope = ema_slope_pct(h1_close, 20, 6)
    h4_adx = adx(h4)
    h4_atr = atr_pct(h4)
    h1_efficiency = trend_efficiency(h1_close, 24)
    extension_pct = abs((price - h4_ema20) / h4_ema20 * 100.0) if h4_ema20 > 0 else 999.0

    long_alignment = price > d1_ema20 > d1_ema50 and price > h4_ema20 > h4_ema50 and price > h1_ema20 > h1_ema50
    short_alignment = price < d1_ema20 < d1_ema50 and price < h4_ema20 < h4_ema50 and price < h1_ema20 < h1_ema50
    long_slope = d1_slope > 0.25 and h4_slope > 0.35 and h1_slope > 0.10
    short_slope = d1_slope < -0.25 and h4_slope < -0.35 and h1_slope < -0.10

    long_structure = structure_score(h4, 'BUY') * 0.55 + structure_score(h1, 'BUY') * 0.45
    short_structure = structure_score(h4, 'SELL') * 0.55 + structure_score(h1, 'SELL') * 0.45
    adx_score = _clamp((h4_adx - 12.0) * 4.0)
    efficiency_score = h1_efficiency
    extension_score = _clamp(100.0 - max(extension_pct - 1.2, 0.0) * 16.0)
    atr_score = _clamp(100.0 - max(h4_atr - 2.0, 0.0) * 10.0)

    long_score = (
        (100.0 if long_alignment else 25.0) * 0.28
        + (100.0 if long_slope else 30.0) * 0.20
        + long_structure * 0.18
        + adx_score * 0.12
        + efficiency_score * 0.10
        + extension_score * 0.07
        + atr_score * 0.05
    )
    short_score = (
        (100.0 if short_alignment else 25.0) * 0.28
        + (100.0 if short_slope else 30.0) * 0.20
        + short_structure * 0.18
        + adx_score * 0.12
        + efficiency_score * 0.10
        + extension_score * 0.07
        + atr_score * 0.05
    )

    result.update({
        'long_score': _clamp(long_score),
        'short_score': _clamp(short_score),
        'long_ok': long_alignment and long_slope and long_score >= 68.0,
        'short_ok': short_alignment and short_slope and short_score >= 68.0,
        'reason': (
            f"趋势质量 多{long_score:.0f}/空{short_score:.0f} | "
            f"ADX{h4_adx:.1f} 效率{h1_efficiency:.0f} 延伸{extension_pct:.2f}% ATR{h4_atr:.2f}%"
        ),
        'metrics': {
            'd1_slope_pct': d1_slope,
            'h4_slope_pct': h4_slope,
            'h1_slope_pct': h1_slope,
            'h4_adx': h4_adx,
            'h4_atr_pct': h4_atr,
            'h1_efficiency': h1_efficiency,
            'extension_pct': extension_pct,
            'long_structure': long_structure,
            'short_structure': short_structure,
        },
    })
    return result
