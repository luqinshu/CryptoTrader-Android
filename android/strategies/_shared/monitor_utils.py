"""
持仓反转监控 — 纯 Python 技术指标（无 pandas/numpy 依赖，ARM 安全）。

从 position_reversal_monitor.py 提取并适配为 Kivy/Android 可用。
"""

from __future__ import annotations

from typing import Any, List, Optional


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_candles(rows: List[List[Any]]) -> List[dict]:
    """将 OKX kline row 列表解析为统一的 K 线字典列表。

    OKX row 格式: [ts(ms), open, high, low, close, vol_quote, ...]
    返回: [{'ts': int, 'o': float, 'h': float, 'l': float, 'c': float, 'v': float}, ...]
    时间升序（最早在前），与 OKX 返回的降序相反。
    """
    candles: List[dict] = []
    for row in reversed(rows or []):
        if len(row) < 6:
            continue
        ts = int(safe_float(row[0]))
        o = safe_float(row[1])
        h = safe_float(row[2])
        l_val = safe_float(row[3])
        c_val = safe_float(row[4])
        v = safe_float(row[5])
        if ts <= 0:
            continue
        candles.append({'ts': ts, 'o': o, 'h': h, 'l': l_val, 'c': c_val, 'v': v})
    return candles


def ema(values: List[float], period: int) -> float:
    """计算 EMA 最新值。"""
    if not values or period <= 0:
        return 0.0
    if len(values) < period:
        return sum(values) / max(len(values), 1)
    multiplier = 2.0 / (period + 1)
    result = sum(values[:period]) / period
    for v in values[period:]:
        result = (v - result) * multiplier + result
    return result


def atr(candles: List[dict], period: int = 14) -> float:
    """计算 ATR 最新值。"""
    if len(candles) < 2:
        return 0.0
    trs: List[float] = []
    for i in range(1, len(candles)):
        cur = candles[i]
        prv = candles[i - 1]
        tr = max(
            cur['h'] - cur['l'],
            abs(cur['h'] - prv['c']),
            abs(cur['l'] - prv['c']),
        )
        trs.append(tr)
    if not trs:
        return 0.0
    if len(trs) < period:
        return sum(trs) / len(trs)
    window = trs[-period:]
    return sum(window) / len(window)


def rolling_atr_rank(candles: List[dict], period: int = 14, rank_window: int = 30) -> float:
    """ATR 历史分位 — 当前 ATR 在最近 rank_window 根 K 线中的排名百分比。"""
    if len(candles) < period + 2:
        return 0.0
    atr_series: List[float] = []
    for end in range(period + 1, len(candles) + 1):
        atr_series.append(atr(candles[:end], period))
    sample = atr_series[-rank_window:] if len(atr_series) >= rank_window else atr_series
    cur = sample[-1] if sample else 0.0
    if not sample or cur <= 0:
        return 0.0
    lower_or_equal = sum(1 for v in sample if v <= cur)
    return lower_or_equal / len(sample)


def calc_vwap(candles: List[dict]) -> float:
    """计算 VWAP。"""
    pv_sum = 0.0
    vol_sum = 0.0
    for c in candles:
        typical = (c['h'] + c['l'] + c['c']) / 3.0
        pv_sum += typical * c['v']
        vol_sum += c['v']
    return pv_sum / vol_sum if vol_sum > 0 else 0.0


def completed_window(candles: List[dict], lookback: int) -> List[dict]:
    """获取已完成 K 线的回看窗口（排除当前未闭合的 K 线）。"""
    if len(candles) <= 1:
        return []
    return candles[:-1][-lookback:]


def latest_complete_close(candles: List[dict]) -> float:
    """最近一根已完成 K 线的收盘价。"""
    if len(candles) >= 2:
        return candles[-2]['c']
    return candles[-1]['c'] if candles else 0.0


def closes(candles: List[dict]) -> List[float]:
    """提取收盘价序列。"""
    return [c['c'] for c in candles]


def volumes(candles: List[dict]) -> List[float]:
    """提取成交量序列。"""
    return [c['v'] for c in candles]
