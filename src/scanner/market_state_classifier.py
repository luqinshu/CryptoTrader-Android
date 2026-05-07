"""
市场状态自适应分类器 v1.0
==========================
基于 BTC 多周期指标自动判断当前市场处于四种状态之一：

  trending  — 强趋势市（ADX>25 + EMA排列清晰 + 波动率适中）
  range     — 震荡市（ADX<20 + BB带宽窄 + 价格在区间内）
  volatile  — 高波动市（ATR飙升 + BB带宽扩张 + ADX跳升）
  neutral   — 无明确特征（数据不足或过渡期）

用途：
  1. 扫描引擎启动时自动检测，注入到所有策略 config 中
  2. 策略根据 market_state 切换参数预设（trending_params / range_params 等）
  3. 回测报告中标注各交易所处的市场状态，用于事后分析
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# 状态枚举
STATE_TRENDING  = "trending"
STATE_RANGE     = "range"
STATE_VOLATILE  = "volatile"
STATE_NEUTRAL   = "neutral"
VALID_STATES    = {STATE_TRENDING, STATE_RANGE, STATE_VOLATILE, STATE_NEUTRAL}


def _safe_mean(values: List[float]) -> float:
    cleaned = [float(v) for v in values if np.isfinite(v)]
    return float(np.mean(cleaned)) if cleaned else 0.0


def _safe_ema(values: List[float], span: int) -> float:
    if len(values) < span:
        return values[-1] if values else 0.0
    k = 2.0 / (span + 1.0)
    ema = float(np.mean(values[:span]))
    for v in values[span:]:
        ema = float(v) * k + ema * (1.0 - k)
    return ema


def classify_market_state(
    klines_4h: List[List],
    klines_1h: List[List],
    klines_1d: List[List],
    btc_change_24h: float = 0.0,
) -> Tuple[str, float, Dict[str, Any]]:
    """
    主分类函数。

    Args:
        klines_4h: BTC 4H K线 [ts, o, h, l, c, vol]
        klines_1h: BTC 1H K线
        klines_1d: BTC 1D K线
        btc_change_24h: BTC 24H涨跌幅%

    Returns:
        (state: str, confidence: float 0~1, diagnostics: dict)
    """
    metrics: Dict[str, float] = {}
    diagnostics: Dict[str, Any] = {"state": STATE_NEUTRAL, "confidence": 0.0}

    # ── 数据完整性检查 ──────────────────────────────────────────────────
    min_4h = 40
    min_1h = 60
    if not klines_4h or len(klines_4h) < min_4h:
        return STATE_NEUTRAL, 0.0, {"reason": f"4H数据不足({len(klines_4h)}<{min_4h})"}
    if not klines_1h or len(klines_1h) < min_1h:
        return STATE_NEUTRAL, 0.0, {"reason": f"1H数据不足({len(klines_1h)}<{min_1h})"}

    # ── 提取 OHLCV ──────────────────────────────────────────────────────
    def _extract(rows, col_idx):
        return [float(r[col_idx]) for r in rows if len(r) > col_idx and float(r[col_idx]) > 0]

    h4_c = _extract(klines_4h, 4)
    h4_h = _extract(klines_4h, 2)
    h4_l = _extract(klines_4h, 3)
    h4_v = _extract(klines_4h, 5)

    h1_c = _extract(klines_1h, 4)
    h1_h = _extract(klines_1h, 2)
    h1_l = _extract(klines_1h, 3)

    d1_c = _extract(klines_1d, 4) if klines_1d else h4_c

    # ── 指标计算 ────────────────────────────────────────────────────────

    # ADX (简化版: 只用4H)
    adx_val = _compute_fast_adx(h4_c, h4_h, h4_l, period=14)
    metrics["adx"] = adx_val

    # ATR%
    atr_val = _compute_atr(h4_c, h4_h, h4_l, period=14)
    if h4_c:
        atr_pct = atr_val / h4_c[-1] * 100
    else:
        atr_pct = 0
    metrics["atr_pct"] = atr_pct

    # BB带宽（4H）
    if len(h4_c) >= 20:
        bb_mid = float(np.mean(h4_c[-20:]))
        bb_std = float(np.std(h4_c[-20:], ddof=1)) if len(h4_c[-20:]) > 1 else 0.0
        bb_width_pct = (bb_std * 4.0 / max(abs(bb_mid), 1e-9)) * 100.0
    else:
        bb_width_pct = 99.0
    metrics["bb_width_pct"] = bb_width_pct

    # BB带宽最近变化（扩张 vs 收缩）
    if len(h4_c) >= 40:
        bb_mid_old = float(np.mean(h4_c[-40:-20]))
        bb_std_old = float(np.std(h4_c[-40:-20], ddof=1)) if len(h4_c[-40:-20]) > 1 else 0
        bb_width_old = (bb_std_old * 4.0 / max(abs(bb_mid_old), 1e-9)) * 100.0
        bb_expanding = bb_width_pct > bb_width_old * 1.15
    else:
        bb_expanding = False
    metrics["bb_expanding"] = 1.0 if bb_expanding else 0.0

    # 趋势一致性（4H: 最近10根中有几根收涨）
    if len(h4_c) >= 11:
        trend_consistency = sum(1 for i in range(1, 11) if h4_c[-i] > h4_c[-i-1]) / 10.0
    else:
        trend_consistency = 0.5
    metrics["trend_consistency"] = trend_consistency

    # EMA排列 (1H: 12/26/50)
    ema12_1h = _safe_ema(h1_c, 12)
    ema26_1h = _safe_ema(h1_c, 26)
    ema50_1h = _safe_ema(h1_c, 50) if len(h1_c) >= 50 else ema26_1h
    h1_bullish = float(ema12_1h > ema26_1h > ema50_1h)
    h1_bearish = float(ema12_1h < ema26_1h < ema50_1h)
    metrics["h1_bullish"] = h1_bullish
    metrics["h1_bearish"] = h1_bearish

    # 日线 EMA 排列
    if len(d1_c) >= 60:
        ema20_d1 = _safe_ema(d1_c, 20)
        ema50_d1 = _safe_ema(d1_c, 50)
        ema120_d1 = _safe_ema(d1_c, 120) if len(d1_c) >= 120 else ema50_d1
        d1_bullish = float(ema20_d1 > ema50_d1 > ema120_d1)
        d1_bearish = float(ema20_d1 < ema50_d1 < ema120_d1)
    else:
        d1_bullish = 0.0
        d1_bearish = 0.0
    metrics["d1_bullish"] = d1_bullish
    metrics["d1_bearish"] = d1_bearish

    # 1H 波动率（对数收益率标准差 * sqrt(24)）
    if len(h1_c) >= 24:
        log_rets = np.diff(np.log(np.array(h1_c[-25:])))
        log_rets = log_rets[np.isfinite(log_rets)]
        rv_1h = float(np.std(log_rets) * np.sqrt(24) * 100.0) if len(log_rets) > 0 else 0.0
    else:
        rv_1h = 0.0
    metrics["rv_1h"] = rv_1h

    # ── 状态判定逻辑 ────────────────────────────────────────────────────

    # volatile: 波动率飙升 + BB扩张 + 趋势一致性低
    volatile_score = (
        (1.0 if atr_pct > 4.5 else 0.0) * 0.30 +
        (1.0 if rv_1h > 7.0 else 0.0) * 0.25 +
        (1.0 if bb_expanding else 0.0) * 0.25 +
        (1.0 if trend_consistency < 0.45 else 0.0) * 0.20
    )

    # trending: ADX高 + EMA排列清晰 + 趋势一致性高 + 波动率适中
    trending_score = (
        _clamp((adx_val - 20.0) / 20.0, 0.0, 1.0) * 0.25 +
        max(h1_bullish, h1_bearish) * 0.20 +
        max(d1_bullish, d1_bearish) * 0.20 +
        _clamp(trend_consistency - 0.45, 0.0, 0.5) * 2.0 * 0.20 +
        _clamp(1.0 - atr_pct / 8.0, 0.0, 1.0) * 0.15
    )

    # range: ADX低 + BB窄 + 趋势一致性中性 + EMA无排列
    range_score = (
        _clamp(1.0 - adx_val / 25.0, 0.0, 1.0) * 0.30 +
        _clamp(1.0 - bb_width_pct / 8.0, 0.0, 1.0) * 0.25 +
        _clamp(1.0 - abs(trend_consistency - 0.5) * 4.0, 0.0, 1.0) * 0.20 +
        _clamp(1.0 - rv_1h / 6.0, 0.0, 1.0) * 0.15 +
        (1.0 - max(h1_bullish, h1_bearish)) * 0.10
    )

    # 选最高分的状态
    scores = {
        STATE_TRENDING: trending_score,
        STATE_RANGE: range_score,
        STATE_VOLATILE: volatile_score,
    }
    best_state = max(scores, key=scores.get)
    best_score = scores[best_state]

    # 最低置信阈值
    if best_score < 0.35:
        best_state = STATE_NEUTRAL
        best_score = 0.0

    diagnostics = {
        "state": best_state,
        "confidence": round(best_score, 3),
        "scores": {k: round(v, 3) for k, v in scores.items()},
        "metrics": {k: round(v, 3) if isinstance(v, float) else v for k, v in metrics.items()},
        "btc_24h": round(btc_change_24h, 2),
    }

    return best_state, best_score, diagnostics


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(v)))


def _compute_atr(closes, highs, lows, period=14):
    if len(closes) < 2:
        return 0.0
    trs = []
    for i in range(1, min(len(closes), period + 20)):
        idx = -i
        a_idx = -(i + 1)
        tr = max(
            highs[idx] - lows[idx],
            abs(highs[idx] - closes[a_idx]),
            abs(lows[idx] - closes[a_idx]),
        )
        trs.append(tr)
    if not trs:
        return 0.0
    return float(np.mean(trs[-min(period, len(trs)):]))


def _compute_fast_adx(closes, highs, lows, period=14):
    """快速ADX（简化版，只算最后值）。数据不足返回 0.0（非 15.0，避免被误判为弱趋势）。"""
    n = len(closes)
    if n < period * 2 + 5:
        return 0.0
    tr_list, pdm_list, ndm_list = [], [], []
    for i in range(1, n):
        tr = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
        up = max(highs[i] - highs[i-1], 0.0)
        dn = max(lows[i-1] - lows[i], 0.0)
        if up > dn: dn = 0.0
        elif dn > up: up = 0.0
        else: up = dn = 0.0
        tr_list.append(tr); pdm_list.append(up); ndm_list.append(dn)

    def _wilder(vals, p):
        if len(vals) < p: return []
        s = [sum(vals[:p])]
        for v in vals[p:]:
            s.append(s[-1] - s[-1]/p + v)
        return s

    atr_s = _wilder(tr_list, period)
    pdm_s = _wilder(pdm_list, period)
    ndm_s = _wilder(ndm_list, period)
    if not atr_s or not pdm_s or not ndm_s:
        return 0.0

    dx_vals = []
    for atr_v, pdm_v, ndm_v in zip(atr_s, pdm_s, ndm_s):
        if atr_v <= 0: continue
        dp = pdm_v / atr_v * 100; dm = ndm_v / atr_v * 100
        den = dp + dm
        dx_vals.append(abs(dp - dm) / den * 100 if den > 0 else 0)

    if len(dx_vals) < period:
        return 0.0

    adx = sum(dx_vals[:period]) / period
    for v in dx_vals[period:]:
        adx = (adx * (period - 1) + v) / period
    return round(adx, 2)
