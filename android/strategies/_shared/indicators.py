"""
共享技术指标库 —— 所有策略文件统一引用此模块。

消除 30 个策略文件中重复的 _to_df/_ema/_rsi/_atr/_adx/_macd/_volume_zscore 等 ~18 个函数。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════
# 数据预处理
# ═══════════════════════════════════════════════

def _check_df(df: pd.DataFrame, label: str, min_len: int) -> None:
    """检验 DataFrame 长度是否达标，不达标抛出 ValueError。"""
    if len(df) < min_len:
        raise ValueError(f"{label}数据不足({len(df)}/{min_len})")


def _to_df(klines) -> pd.DataFrame:
    """将 K 线列表或 DataFrame 统一转为标准 DataFrame。

    Columns: ['ts', 'o', 'h', 'l', 'c', 'vol']
    """
    _EMPTY_COLS = ['ts', 'o', 'h', 'l', 'c', 'vol']
    
    def _empty_df():
        """ARM-safe empty DataFrame: avoids pd.DataFrame(columns=...) which
        triggers init_dict → construct_1d_arraylike_from_scalar crash."""
        try:
            # try creating from numpy empty array first
            arr = np.empty((0, 6))
            df = pd.DataFrame(arr)
            df.columns = _EMPTY_COLS
            return df
        except Exception:
            # fallback: create a 1-row dummy then truncate
            df = pd.DataFrame(np.zeros((1, 6)), columns=_EMPTY_COLS)
            return df.iloc[0:0]
    
    if not klines:
        return _empty_df()
    if isinstance(klines, pd.DataFrame):
        return klines
    valid = [r[:6] for r in klines if isinstance(r, (list, tuple)) and len(r) >= 6]
    if not valid:
        return _empty_df()
    try:
        df = pd.DataFrame(valid, columns=['ts', 'o', 'h', 'l', 'c', 'vol'])
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna(subset=['ts', 'o', 'h', 'l', 'c'])
        df['vol'] = df['vol'].fillna(0.0)
    except Exception:
        # fallback for ARM numpy dtype issues
        arr = np.array(valid, dtype=np.float64)
        df = pd.DataFrame(arr, columns=['ts', 'o', 'h', 'l', 'c', 'vol'])
        df = df.dropna(subset=['ts', 'o', 'h', 'l', 'c'])
        df['vol'] = df['vol'].fillna(0.0)
    return df.sort_values('ts').drop_duplicates('ts', keep='last').reset_index(drop=True)


def _aggregate_bars(df: pd.DataFrame, gs: int) -> pd.DataFrame:
    """将细粒度 K 线聚合成粗粒度 K 线（如 1m→5m）。

    Args:
        df: 原始 K 线 DataFrame（columns: ts,o,h,l,c,vol）
        gs: 聚合粒度（根数）
    """
    usable = len(df) // gs * gs
    if usable <= 0:
        return pd.DataFrame(columns=["ts", "o", "h", "l", "c", "vol"])
    tail = df.tail(usable).reset_index(drop=True)
    g = tail.groupby(tail.index // gs)
    return pd.DataFrame({
        "ts": g["ts"].last(),
        "o": g["o"].first(),
        "h": g["h"].max(),
        "l": g["l"].min(),
        "c": g["c"].last(),
        "vol": g["vol"].sum(),
    }).reset_index(drop=True)


# ═══════════════════════════════════════════════
# 基础数学工具
# ═══════════════════════════════════════════════

def _safe_float(value, default: float = 0.0) -> float:
    """安全地转为 float，NaN/Inf/TypeError 返回 default。"""
    try:
        n = float(value)
    except (TypeError, ValueError):
        return default
    return n if np.isfinite(n) else default


def _clamp(value: float, low: float, high: float) -> float:
    """将值限制在 [low, high] 范围内。"""
    return max(low, min(high, float(value)))


def _pct_change(series: pd.Series, bars: int) -> float:
    """计算最近 bars 根 K 线的涨跌幅。"""
    if len(series) <= bars or bars <= 0:
        return 0.0
    b = float(series.iloc[-(bars + 1)])
    l = float(series.iloc[-1])
    return l / b - 1.0 if b > 0 else 0.0


def _cfg_float(config: Dict[str, Any], key: str, default: float) -> float:
    """从配置字典中读取 float 类型参数，缺失时返回默认值。"""
    if key not in config or config.get(key) is None:
        return float(default)
    return _safe_float(config.get(key), float(default))


def _cfg_int(config: Dict[str, Any], key: str, default: int) -> int:
    """从配置字典中读取 int 类型参数，缺失时返回默认值。"""
    if key not in config or config.get(key) is None:
        return int(default)
    try:
        return int(float(config.get(key)))
    except (TypeError, ValueError):
        return int(default)


# ═══════════════════════════════════════════════
# 统计函数
# ═══════════════════════════════════════════════

def _robust_zscore(series: pd.Series) -> pd.Series:
    """鲁棒 z-score（中位数/MAD），截断到 [-3, 3]。"""
    vals = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    med = vals.median()
    mad = (vals - med).abs().median()
    if pd.notna(mad) and mad > 1e-12:
        z = 0.6745 * (vals - med) / mad
    else:
        std = vals.std(ddof=0)
        z = (vals - vals.mean()) / std if pd.notna(std) and std > 1e-12 else vals * 0.0
    return z.clip(-3.0, 3.0)


# ═══════════════════════════════════════════════
# 均线 / 趋势指标
# ═══════════════════════════════════════════════

def _ema(s: pd.Series, span: int) -> float:
    """返回 EMA 的最新值。"""
    return float(s.ewm(span=span, adjust=False).mean().iloc[-1])


def _efficiency_ratio(c: pd.Series, window: int = 20) -> float:
    """计算价格效率比（净位移 / 总路径）。

    值越接近 1 表示趋势越平滑，接近 0 表示噪音多。
    """
    if len(c) < window + 1:
        return 0.0
    wc = c.iloc[-(window + 1):]
    n_ = abs(float(wc.iloc[-1]) - float(wc.iloc[0]))
    p_ = float(wc.diff().abs().sum())
    return float(min(1.0, n_ / p_)) if p_ > 0 else 0.0


def _measure_trend_age(close: pd.Series, fast: int, slow: int, direction: float) -> int:
    """统计 EMA 快慢线连续同向排列的 K 线根数。

    Args:
        close: 收盘价序列
        fast: 快线周期
        slow: 慢线周期
        direction: >=0 看多，<0 看空
    """
    if len(close) < slow + 3:
        return 0
    ema_f = close.ewm(span=fast, adjust=False).mean().values
    ema_s = close.ewm(span=slow, adjust=False).mean().values
    cond = (ema_f > ema_s) if direction >= 0 else (ema_f < ema_s)
    age = 0
    for i in range(len(cond) - 1, max(len(cond) - 100, -1), -1):
        if cond[i]:
            age += 1
        else:
            break
    return age


# ═══════════════════════════════════════════════
# 震荡指标
# ═══════════════════════════════════════════════

def _rsi_wilder(c: pd.Series, period: int = 14) -> float:
    """Wilder 平滑 RSI，返回最新值。"""
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

# Alias for strategies that use _rsi instead of _rsi_wilder
_rsi = _rsi_wilder

def _macd_mom(c: pd.Series) -> float:
    """MACD momentum (histogram) latest value"""
    diff, dea, hist = _macd(c)
    if len(hist) == 0:
        return 0.0
    return float(hist.iloc[-1])

def _macd(close: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """计算 MACD，返回 (DIF, DEA, 柱状线)。"""
    e12 = close.ewm(span=12, adjust=False).mean()
    e26 = close.ewm(span=26, adjust=False).mean()
    line = e12 - e26
    sig = line.ewm(span=9, adjust=False).mean()
    return line, sig, (line - sig)


# ═══════════════════════════════════════════════
# 波动率 / 趋势强度
# ═══════════════════════════════════════════════

def _atr(df: pd.DataFrame, period: int = 14) -> float:
    """Wilder 平滑 ATR，返回最新值。"""
    if len(df) < period + 1:
        return float(df['h'].iloc[-1] - df['l'].iloc[-1]) or 1.0
    pc = df['c'].shift(1)
    tr = pd.concat([
        df['h'] - df['l'],
        (df['h'] - pc).abs(),
        (df['l'] - pc).abs(),
    ], axis=1).max(axis=1)
    return float(tr.ewm(alpha=1 / period, adjust=False).mean().iloc[-1]) or 1.0


def _adx(df: pd.DataFrame, period: int = 14) -> float:
    """Wilder 平滑 ADX，返回最新值。数据不足返回 0.0。"""
    if len(df) < period * 2 + 1:
        return 0.0
    h, l, c = df['h'], df['l'], df['c']
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    um = h.diff()
    dm = -l.diff()
    pdm = um.clip(lower=0).where((um > dm) & (um > 0), 0.0)
    mdm = dm.clip(lower=0).where((dm > um) & (dm > 0), 0.0)
    atr_s = tr.ewm(alpha=1 / period, adjust=False).mean()
    pdi = 100 * pdm.ewm(alpha=1 / period, adjust=False).mean() / atr_s.replace(0, np.nan)
    mdi = 100 * mdm.ewm(alpha=1 / period, adjust=False).mean() / atr_s.replace(0, np.nan)
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    a = dx.ewm(alpha=1 / period, adjust=False).mean()
    v = a.iloc[-1]
    return float(v) if pd.notna(v) else 0.0


# ═══════════════════════════════════════════════
# 成交量指标
# ═══════════════════════════════════════════════

def _volume_ratio_adjusted(df: pd.DataFrame, window: int = 20) -> float:
    """计算成交量相对比率，并对未收盘 bar 做进度修正。"""
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
            if tl < 1e12:
                tl_ms = tl * 1000
                bi_ms = bi * 1000
            else:
                tl_ms = tl
                bi_ms = bi
            el = now_ms - tl_ms
            if 0 < el < bi_ms:
                pr = el / bi_ms
                if pr >= 0.1:
                    cv = cv / pr
        except Exception:
            pass
    return float(cv / baseline)


def _volume_zscore(vol: pd.Series, window: int = 20) -> float:
    """对数成交量 z-score（最近 window 根 vs 前 window 根）。"""
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


# ═══════════════════════════════════════════════
# 趋势快照
# ═══════════════════════════════════════════════

def _local_trend_snapshot(
    d1: pd.DataFrame, h4: pd.DataFrame, h1: pd.DataFrame, price: float
) -> Dict[str, Any]:
    """综合日线/4H/1H 多周期趋势评分。

    Returns:
        {'long_ok', 'short_ok', 'long_score', 'short_score', 'reason', 'metrics'}
    """
    d1e21 = _ema(d1['c'], 21)
    d1e55 = _ema(d1['c'], 55)
    h4e21 = _ema(h4['c'], 21)
    h4e55 = _ema(h4['c'], 55)
    h1e21 = _ema(h1['c'], 21)
    h1e55 = _ema(h1['c'], 55)
    pt = 40.0 / 3.0
    la = 0.0
    sa = 0.0
    for p, e21, e55 in [(price, d1e21, d1e55), (price, h4e21, h4e55), (price, h1e21, h1e55)]:
        if p > e21 > e55:
            la += pt
        if p < e21 < e55:
            sa += pt
    h4adx = _adx(h4, 14)
    adxs = min(30.0, max(0.0, (h4adx - 15.0) / 25.0 * 30.0))
    h1er = _efficiency_ratio(h1['c'], 20)
    ers = min(20.0, h1er * 20.0)
    d1sp = abs(d1e21 - d1e55) / d1e55 * 100 if d1e55 > 0 else 0.0
    h4sp = abs(h4e21 - h4e55) / h4e55 * 100 if h4e55 > 0 else 0.0
    sps = min(10.0, (d1sp + h4sp) / 4.0 * 10.0)
    ls = la + adxs + ers + sps
    ss = sa + adxs + ers + sps
    lok = ls >= 55.0 and h4adx >= 15.0
    sok = ss >= 55.0 and h4adx >= 15.0
    rp = []
    if h4adx < 15.0:
        rp.append(f"H4_ADX弱({h4adx:.1f})")
    if la < pt * 2 and sa < pt * 2:
        rp.append("均线排列不统一")
    r = " | ".join(rp) if rp else "趋势结构健康"
    return {
        'long_ok': bool(lok), 'short_ok': bool(sok),
        'long_score': float(ls), 'short_score': float(ss), 'reason': r,
        'metrics': {
            'h4_adx': float(h4adx), 'h1_efficiency': float(h1er * 100),
            'd1_spread_pct': float(d1sp), 'h4_spread_pct': float(h4sp), 'reason': r,
        },
    }


def _latest_swing_levels(
    df: pd.DataFrame, left: int = 5, right: int = 5,
    skip_recent: int = 3, max_lookback: int = 80
) -> Tuple[Optional[float], Optional[float]]:
    """找出最近的波段高点和低点。

    Returns: (swing_high, swing_low) — None 表示未找到。
    """
    if len(df) < left + right + skip_recent + 5:
        return None, None
    h = df['h'].values
    l = df['l'].values
    end = len(df) - skip_recent
    start = max(left, end - max_lookback)
    sh = None
    sl_ = None
    for i in range(end - right - 1, start - 1, -1):
        if sh is None and h[i] == max(h[i - left:i + right + 1]):
            sh = float(h[i])
        if sl_ is None and l[i] == min(l[i - left:i + right + 1]):
            sl_ = float(l[i])
        if sh is not None and sl_ is not None:
            break
    return sh, sl_


# ═══════════════════════════════════════════════
# 微观回调/企稳检测
# ═══════════════════════════════════════════════

def _calc_atr(
    high: pd.Series, low: pd.Series, close: pd.Series,
    span: int = 5, seg_start: int = 0, seg_end: int = -1
) -> float:
    """计算指定窗口内的平均真实波幅。"""
    try:
        if seg_end <= seg_start:
            return 0.0
        pc = close.shift(1)
        tr = pd.concat([
            (high - low).abs(),
            (high - pc).abs(),
            (low - pc).abs(),
        ], axis=1).max(axis=1)
        seg_tr = tr.iloc[seg_start:seg_end]
        if len(seg_tr) == 0:
            return 0.0
        return float(seg_tr.ewm(alpha=1 / span, adjust=False).mean().iloc[-1] or 0.0)
    except Exception:
        return 0.0


def _calc_volume_delta(
    close: pd.Series, high: pd.Series, low: pd.Series, vol: pd.Series,
    seg_start: int = 0, seg_end: int = -1
) -> Tuple[float, float]:
    """近似计算买卖力量（无 tick 数据时的替代方案）。

    Returns: (buy_power, sell_power)
    """
    try:
        seg_c = close.iloc[seg_start:seg_end]
        seg_h = high.iloc[seg_start:seg_end]
        seg_l = low.iloc[seg_start:seg_end]
        seg_v = vol.iloc[seg_start:seg_end] if len(vol) > seg_start else pd.Series(np.ones(len(seg_c)))
        if len(seg_c) == 0:
            return 0.0, 0.0
        spread = (seg_h.values - seg_l.values)
        spread = np.where(spread > 0, spread, 1.0)
        buy_ratio = (seg_c.values - seg_l.values) / spread
        buy_ratio = np.clip(buy_ratio, 0.0, 1.0)
        buy_power = float(np.sum(buy_ratio * seg_v.values))
        sell_power = float(np.sum((1.0 - buy_ratio) * seg_v.values))
        return buy_power, sell_power
    except Exception:
        return 0.0, 0.0


def _calc_vwap(
    close: pd.Series, high: pd.Series, low: pd.Series, vol: pd.Series,
    seg_start: int = 0, seg_end: int = -1
) -> float:
    """计算指定窗口的 VWAP（成交量加权均价）。"""
    try:
        seg_c = close.iloc[seg_start:seg_end]
        seg_h = high.iloc[seg_start:seg_end]
        seg_l = low.iloc[seg_start:seg_end]
        seg_v = vol.iloc[seg_start:seg_end] if len(vol) > seg_start else pd.Series(np.ones(len(seg_c)))
        if len(seg_c) == 0:
            return 0.0
        typical = (seg_h.values + seg_l.values + seg_c.values) / 3.0
        total_vol = float(np.sum(seg_v.values))
        if total_vol <= 0:
            return float(seg_c.iloc[-1])
        return float(np.sum(typical * seg_v.values) / total_vol)
    except Exception:
        return 0.0


def _micro_pullback_continuation(
    m3: pd.DataFrame, trend_hint: float, config: Dict[str, Any]
) -> Dict[str, Any]:
    """3 分钟微观回调企稳检测（Gate A 微观确认）。

    Args:
        m3: 3 分钟 K 线 DataFrame
        trend_hint: 大周期趋势方向（>0 多头，<0 空头）
        config: 策略配置字典

    Returns:
        {'confirmed', 'score', 'state', 'reason', 'pullback_pct',
         'impulse_pct', 'staleness_bars', 'timeliness_score', ...}
    """
    _default = lambda reason: {
        "confirmed": False, "score": 0.0, "state": "未通过",
        "reason": reason, "pullback_pct": 0.0, "impulse_pct": 0.0,
        "staleness_bars": 0, "timeliness_score": 0.0,
        "atr_squeeze": False, "volume_delta": False, "vwap_aligned": False,
        "micro_indicators": "无",
    }
    if m3 is None or len(m3) < 36:
        return _default("3m数据不足")
    if abs(float(trend_hint)) < 0.12:
        return _default("大周期趋势不够明确")

    n = len(m3)
    stab_bars = max(2, int(config.get("m3_stabilization_bars", 4) or 4))
    min_pb = float(config.get("m3_pullback_min_pct", 0.50) or 0.50)
    max_pb = float(config.get("m3_pullback_max_pct", 2.20) or 2.20)
    max_stale = int(config.get("max_m3_staleness_bars", 15) or 15)
    min_impulse = float(config.get("m3_min_impulse_pct", 0.65) or 0.65)
    vol_min_ratio = float(config.get("vol_continuation_min_ratio", 0.78) or 0.78)
    require_freshness = bool(config.get("require_m3_freshness", True))

    close = m3["c"].astype(float)
    high = m3["h"].astype(float)
    low = m3["l"].astype(float)
    vol = m3["vol"].astype(float) if "vol" in m3.columns else pd.Series(np.ones(n), index=m3.index)
    ema8 = close.ewm(span=8, adjust=False).mean()
    ema21 = close.ewm(span=21, adjust=False).mean()

    seg_stab_end = n
    seg_stab_start = max(0, n - stab_bars)
    seg_pb_end = seg_stab_start
    seg_pb_start = max(0, seg_pb_end - max(6, stab_bars + 2))
    seg_imp_end = seg_pb_start
    seg_imp_start = max(0, seg_imp_end - 20)

    if seg_imp_end <= seg_imp_start or seg_pb_end <= seg_pb_start:
        return _default("m3窗口分段不足")

    if trend_hint > 0:
        impulse_low = float(low.iloc[seg_imp_start:seg_imp_end].min())
        impulse_high = float(high.iloc[seg_imp_start:seg_imp_end].max())
        pullback_low = float(low.iloc[seg_pb_start:seg_pb_end].min())
        pb_idx_rel = int(low.iloc[seg_pb_start:seg_pb_end].idxmin()) - seg_pb_start
        staleness_bars = (n - 1) - (seg_pb_start + pb_idx_rel)
        impulse_pct = (impulse_high / max(impulse_low, 1e-9) - 1.0) * 100.0
        pullback_pct = (impulse_high / max(pullback_low, 1e-9) - 1.0) * 100.0
        retrace = pullback_pct / max(impulse_pct, 1e-9)

        stab_close = close.iloc[seg_stab_start:seg_stab_end]
        stab_low = low.iloc[seg_stab_start:seg_stab_end]
        ema8_stab = ema8.iloc[seg_stab_start:seg_stab_end]
        ema21_stab = ema21.iloc[seg_stab_start:seg_stab_end]
        ema_ok = (float(ema8_stab.iloc[-1]) >= float(ema21_stab.iloc[-1]) * 0.997)
        price_above_ema = float(stab_close.iloc[-1]) > float(ema8_stab.iloc[-1])
        no_new_low = float(stab_low.min()) >= pullback_low * 0.995
        stab_direction_ok = float(stab_close.iloc[-1]) >= float(stab_close.iloc[0]) * 0.998
        stabilized = price_above_ema and ema_ok and no_new_low and stab_direction_ok
    else:
        impulse_high = float(high.iloc[seg_imp_start:seg_imp_end].max())
        impulse_low = float(low.iloc[seg_imp_start:seg_imp_end].min())
        pullback_high = float(high.iloc[seg_pb_start:seg_pb_end].max())
        pb_idx_rel = int(high.iloc[seg_pb_start:seg_pb_end].idxmax()) - seg_pb_start
        staleness_bars = (n - 1) - (seg_pb_start + pb_idx_rel)
        impulse_pct = (impulse_high / max(impulse_low, 1e-9) - 1.0) * 100.0
        pullback_pct = (pullback_high / max(impulse_low, 1e-9) - 1.0) * 100.0
        retrace = pullback_pct / max(impulse_pct, 1e-9)

        stab_close = close.iloc[seg_stab_start:seg_stab_end]
        stab_high = high.iloc[seg_stab_start:seg_stab_end]
        ema8_stab = ema8.iloc[seg_stab_start:seg_stab_end]
        ema21_stab = ema21.iloc[seg_stab_start:seg_stab_end]
        ema_ok = (float(ema8_stab.iloc[-1]) <= float(ema21_stab.iloc[-1]) * 1.003)
        price_below_ema = float(stab_close.iloc[-1]) < float(ema8_stab.iloc[-1])
        no_new_high = float(stab_high.max()) <= pullback_high * 1.005
        stab_direction_ok = float(stab_close.iloc[-1]) <= float(stab_close.iloc[0]) * 1.002
        stabilized = price_below_ema and ema_ok and no_new_high and stab_direction_ok

    recent_vol = float(vol.iloc[seg_stab_start:seg_stab_end].mean())
    base_vol = float(vol.iloc[max(0, seg_stab_start - 18):seg_stab_start].mean()) if seg_stab_start >= 6 else float(vol.mean())
    vol_ok = recent_vol >= base_vol * vol_min_ratio if base_vol > 0 else True
    pullback_ok = min_pb <= pullback_pct <= max_pb
    impulse_ok = impulse_pct >= min_impulse
    retrace_ok = 0.15 <= retrace <= 0.85

    atr_squeeze_ok = False
    atr_squeeze_score = 0.0
    if bool(config.get("enable_atr_squeeze_check", True)):
        atr_recent = _calc_atr(high, low, close, span=5, seg_start=seg_stab_start, seg_end=seg_stab_end)
        atr_prior = _calc_atr(high, low, close, span=5,
                              seg_start=max(0, seg_stab_start - 12),
                              seg_end=max(0, seg_stab_start))
        if atr_prior > 0 and atr_recent > 0:
            squeeze_ratio = atr_recent / atr_prior
            atr_threshold = float(config.get("atr_squeeze_ratio", 0.55) or 0.55)
            atr_squeeze_ok = squeeze_ratio <= atr_threshold
            atr_squeeze_score = _clamp(1.0 - squeeze_ratio / atr_threshold, 0.0, 1.0)

    delta_ok = False
    delta_score = 0.0
    if bool(config.get("enable_volume_delta_check", True)):
        buy_power, sell_power = _calc_volume_delta(close, high, low, vol,
                                                    seg_start=seg_stab_start, seg_end=seg_stab_end)
        total_power = buy_power + sell_power
        if total_power > 0:
            delta_ratio = buy_power / max(sell_power, 1e-9)
            delta_min = float(config.get("volume_delta_min_ratio", 1.15) or 1.15)
            if trend_hint > 0:
                delta_ok = delta_ratio >= delta_min
                delta_score = _clamp(delta_ratio / max(delta_min * 2, 1.0), 0.0, 1.0)
            else:
                delta_ok = (1.0 / max(delta_ratio, 1e-9)) >= delta_min
                delta_score = _clamp((1.0 / max(delta_ratio, 1e-9)) / max(delta_min * 2, 1.0), 0.0, 1.0)

    vwap_ok = False
    vwap_score = 0.0
    if bool(config.get("enable_vwap_alignment_check", True)):
        vwap_val = _calc_vwap(close, high, low, vol, seg_start=seg_stab_start, seg_end=seg_stab_end)
        if vwap_val > 0:
            latest_close = float(close.iloc[-1])
            if trend_hint > 0:
                vwap_ok = latest_close > vwap_val
                vwap_score = _clamp((latest_close / max(vwap_val, 1e-9) - 1.0) * 100.0, 0.0, 1.0)
            else:
                vwap_ok = latest_close < vwap_val
                vwap_score = _clamp((1.0 - latest_close / max(vwap_val, 1e-9)) * 100.0, 0.0, 1.0)

    freshness_ok = staleness_bars <= max_stale
    timeliness_score = _clamp(1.0 - staleness_bars / max(max_stale, 1), -1.0, 1.0)

    if require_freshness:
        confirmed = bool(stabilized and vol_ok and pullback_ok and impulse_ok and retrace_ok and freshness_ok)
    else:
        confirmed = bool(stabilized and vol_ok and pullback_ok and impulse_ok and retrace_ok)

    state = ("回调完成" if confirmed else
             "企稳中" if pullback_ok and impulse_ok and (stabilized or vol_ok) else
             "已过期" if not freshness_ok else "未通过")

    imp_norm = _clamp(impulse_pct / max(max_pb * 1.5, 1.0), 0.0, 1.0)
    pb_norm = _clamp(pullback_pct / max(max_pb, 1.0), 0.0, 1.0)
    score = _clamp(
        imp_norm * 0.4 + pb_norm * 0.4
        + (0.5 if stabilized else -0.3)
        + (0.3 if vol_ok else -0.2)
        + (0.2 if freshness_ok else -0.4)
        + atr_squeeze_score * 0.25
        + delta_score * 0.20
        + vwap_score * 0.15
        - 1.0,
        -2.0, 2.0,
    )

    reason_parts = []
    if not impulse_ok:
        reason_parts.append("原趋势脉冲不足")
    if not pullback_ok:
        reason_parts.append(f"回调幅度不在{min_pb:.1f}~{max_pb:.1f}%内")
    if not retrace_ok:
        reason_parts.append(f"回调比例{retrace:.0%}失衡")
    if not stabilized:
        reason_parts.append("企稳结构不足")
    if not vol_ok:
        reason_parts.append("量能续航不足")
    if not freshness_ok and require_freshness:
        reason_parts.append(f"回调过旧({staleness_bars}根/{max_stale}根上限)")
    extra_parts = []
    if atr_squeeze_ok:
        extra_parts.append("ATR收缩蓄力")
    if delta_ok:
        extra_parts.append("净买入占优" if trend_hint > 0 else "净卖出占优")
    if vwap_ok:
        extra_parts.append("VWAP支撑确认" if trend_hint > 0 else "VWAP压制确认")

    return {
        "confirmed": confirmed, "state": state, "score": float(score),
        "reason": "通过" if confirmed else "，".join(reason_parts) or "未通过",
        "pullback_pct": round(float(pullback_pct), 4),
        "impulse_pct": round(float(impulse_pct), 4),
        "staleness_bars": staleness_bars,
        "timeliness_score": float(timeliness_score),
        "atr_squeeze": atr_squeeze_ok,
        "volume_delta": delta_ok,
        "vwap_aligned": vwap_ok,
        "micro_indicators": " | ".join(extra_parts) if extra_parts else "无",
    }


# ═══════════════════════════════════════════════
# 辅助数据类
# ═══════════════════════════════════════════════

class _MinimalSymbol:
    """轻量级 symbol 代理，用于策略内部传递交易对信息。"""

    __slots__ = ('inst_id', 'extra_data')

    def __init__(self, inst_id: str, extra_data: Optional[Dict[str, Any]] = None):
        self.inst_id = inst_id
        self.extra_data = extra_data or {}
