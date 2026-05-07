"""
波动率分层池 v1.0
==================
按品种的 ATR% 分为三档，每档使用不同的策略参数。

原理：
  低波动品种 (BTC/ETH/BNB)  — ATR < 3%  — 保守参数: 小回调即触发、紧止损
  中波动品种 (SOL/AVAX/DOGE) — ATR 3~8% — 标准参数: 平衡
  高波动品种 (PEPE/BONK)      — ATR > 8%  — 进攻参数: 宽回调容忍、宽止损

ATR 计算：
  基于 4H K线（至少 24 根），用 Wilder EMA 方式计算，再除以当前价格得到 ATR%

分层参数覆盖 (每池预设):
  - pullback range (min/max)  — 回调幅度容忍范围
  - stop_loss/atr_mult       — 止损 ATR 倍数
  - position_pct              — 建议仓位比例
  - volume_confirm_ratio      — 量能确认要求
  - stabilization_bars        — 企稳确认根数
  - resumption_bars           — 回升确认根数
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ── 分层定义 ────────────────────────────────────────────────────────────────

POOL_LOW    = "low_vol"       # ATR < 3%
POOL_MEDIUM = "medium_vol"    # ATR 3~8%
POOL_HIGH   = "high_vol"      # ATR > 8%

POOL_LABELS = {
    POOL_LOW: "低波动保守",
    POOL_MEDIUM: "中波动标准",
    POOL_HIGH: "高波动进攻",
}

# 每池的参数覆盖 (策略通用键名)
POOL_PARAMS: Dict[str, Dict[str, Any]] = {
    POOL_LOW: {
        "m3_pullback_min_pct": 0.08,
        "m3_pullback_max_pct": 1.20,
        "m3_stabilization_bars": 3,
        "m3_resumption_bars": 4,
        "volume_confirm_ratio": 0.65,
        "stop_atr_multiplier": 1.5,
        "target_atr_multiplier": 2.5,
        "position_pct_hint": 0.15,
        "h1_pullback_min_pct": 1.0,
        "h1_pullback_max_pct": 5.0,
        "h1_max_dryup_ratio": 0.70,
        "h1_stab_pos_min": 0.35,
    },
    POOL_MEDIUM: {
        "m3_pullback_min_pct": 0.15,
        "m3_pullback_max_pct": 2.50,
        "m3_stabilization_bars": 2,
        "m3_resumption_bars": 3,
        "volume_confirm_ratio": 0.50,
        "stop_atr_multiplier": 1.8,
        "target_atr_multiplier": 3.0,
        "position_pct_hint": 0.10,
        "h1_pullback_min_pct": 1.8,
        "h1_pullback_max_pct": 8.0,
        "h1_max_dryup_ratio": 0.78,
        "h1_stab_pos_min": 0.28,
    },
    POOL_HIGH: {
        "m3_pullback_min_pct": 0.10,
        "m3_pullback_max_pct": 4.50,
        "m3_stabilization_bars": 2,
        "m3_resumption_bars": 3,
        "volume_confirm_ratio": 0.35,
        "stop_atr_multiplier": 2.5,
        "target_atr_multiplier": 4.0,
        "position_pct_hint": 0.05,
        "h1_pullback_min_pct": 1.2,
        "h1_pullback_max_pct": 12.0,
        "h1_max_dryup_ratio": 0.85,
        "h1_stab_pos_min": 0.22,
    },
}

# ATR 阈值
LOW_THRESHOLD  = 3.0   # ATR% < 3 → 低波动
HIGH_THRESHOLD = 8.0   # ATR% > 8 → 高波动


def compute_atr_pct(klines: List[List], period: int = 14) -> float:
    """
    从原始 K 线计算 ATR%。

    Args:
        klines: [[ts, o, h, l, c, vol], ...]
        period: ATR 周期

    Returns:
        ATR% (0~100), 计算失败返回 4.0(默认中波动)
    """
    if not klines or len(klines) < period + 2:
        return 4.0

    try:
        trs = []
        for i in range(1, min(len(klines), period + 20)):
            idx = -i
            prev_idx = -(i + 1)
            h = float(klines[idx][2])
            l = float(klines[idx][3])
            pc = float(klines[prev_idx][4])
            if h <= 0 or l <= 0 or pc <= 0:
                continue
            tr = max(h - l, abs(h - pc), abs(l - pc))
            trs.append(tr)

        if not trs:
            return 4.0

        # Wilder EMA ATR
        atr = float(np.mean(trs[:period])) if len(trs) >= period else float(np.mean(trs))
        for tr in trs[period:]:
            atr = (atr * (period - 1) + tr) / period

        last_close = float(klines[-1][4])
        if last_close <= 0:
            return 4.0

        return round(atr / last_close * 100.0, 2)
    except Exception:
        return 4.0


def classify_pool(atr_pct: float) -> str:
    """根据 ATR% 返回池名"""
    if atr_pct < LOW_THRESHOLD:
        return POOL_LOW
    if atr_pct > HIGH_THRESHOLD:
        return POOL_HIGH
    return POOL_MEDIUM


def classify_symbol(klines_4h: List[List]) -> Tuple[str, float]:
    """
    对一个品种分类。

    Args:
        klines_4h: 4H K线数据

    Returns:
        (pool_name, atr_pct)
    """
    atr_pct = compute_atr_pct(klines_4h, period=14)
    pool = classify_pool(atr_pct)
    return pool, atr_pct


def classify_all_symbols(
    symbols: List[Any],
    klines_4h_map: Dict[str, List[List]],
) -> Dict[str, Tuple[str, float]]:
    """
    批量分类所有品种。

    Args:
        symbols: ScannerSymbol 列表
        klines_4h_map: {inst_id: 4H_klines}

    Returns:
        {inst_id: (pool_name, atr_pct)}
    """
    result = {}
    for sym in symbols:
        inst_id = str(getattr(sym, "inst_id", ""))
        klines = klines_4h_map.get(inst_id, [])
        if not klines:
            result[inst_id] = (POOL_MEDIUM, 4.0)
            continue
        pool, atr_pct = classify_symbol(klines)
        result[inst_id] = (pool, atr_pct)
    return result


def get_pool_params(pool: str, strategy_keys: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    获取指定池的参数覆盖。

    Args:
        pool: 池名 (low_vol/medium_vol/high_vol)
        strategy_keys: 只返回这些键 (None = 全部)

    Returns:
        参数 dict
    """
    params = POOL_PARAMS.get(pool, POOL_PARAMS[POOL_MEDIUM])
    if strategy_keys:
        return {k: v for k, v in params.items() if k in strategy_keys}
    return dict(params)


def get_position_pct_hint(pool: str) -> float:
    """获取建议仓位比例"""
    return POOL_PARAMS.get(pool, POOL_PARAMS[POOL_MEDIUM]).get("position_pct_hint", 0.10)
