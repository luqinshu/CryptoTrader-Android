"""
基于风险预算的仓位计算器 v1.0
=============================
核心原理：让每笔交易的风险敞口相等，而不是让每笔交易的仓位相等。

问题：
  传统固定仓位: BTC 10%仓位 + PEPE 10%仓位
  → BTC 一天波动 2% = 资本波动 0.2%
  → PEPE 一天波动 40% = 资本波动 4%    (20倍风险差！)

解决：
  风险预算: 每笔交易最多亏损资本的 1%
  → BTC ATR=2%, 止损=1.8×ATR=3.6% → 仓位=1%/3.6%=27.8%
  → PEPE ATR=40%, 止损=2.5×ATR=100%→仓位=1%/100%=1%
  (自动均衡！)

公式:
  Position Value = Risk Amount / Stop Distance %
  Position Size  = Position Value / Entry Price

支持两种止损模式:
  1. ATR动态止损 (默认): Stop% = ATR% × multiplier
  2. Kelly公式 (可选): 胜率已知时最优仓位

集成:
  任何策略在输出信号时调用 calculate_position()
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np


# ── 默认配置 ────────────────────────────────────────────────────────────────

DEFAULT_RISK_PCT     = 0.01    # 每笔风险 = 总资本的 1%
DEFAULT_ATR_MULT     = 1.8     # ATR 止损倍数
DEFAULT_MAX_POSITION = 0.50    # 单笔最高仓位 50%
DEFAULT_MIN_POSITION = 0.005   # 单笔最低仓位 0.5%
DEFAULT_LEVERAGE_MAX = 5       # 最大杠杆


# ── 核心计算 ────────────────────────────────────────────────────────────────

def calculate_position(
    capital: float,
    entry_price: float,
    atr_pct: float,
    risk_pct: float = DEFAULT_RISK_PCT,
    atr_multiplier: float = DEFAULT_ATR_MULT,
    max_position_pct: float = DEFAULT_MAX_POSITION,
    min_position_pct: float = DEFAULT_MIN_POSITION,
    max_leverage: int = DEFAULT_LEVERAGE_MAX,
    win_rate: Optional[float] = None,
) -> Dict[str, Any]:
    """
    计算基于风险预算的建议仓位。

    Args:
        capital:            总资金 (USDT)
        entry_price:        入场价
        atr_pct:            ATR% (如 2.5 表示 2.5% 的日波动率)
        risk_pct:           每笔最大风险占比 (如 0.01 = 1%)
        atr_multiplier:     ATR止损倍数 (如 1.8)
        max_position_pct:   最大仓位占比上限
        min_position_pct:   最小仓位占比下限 (低于此值不交易)
        max_leverage:       最大杠杆倍数
        win_rate:           历史胜率 (0~1), 用于 Kelly 优化

    Returns:
        {
            "position_usdt":     仓位 USDT 金额,
            "position_pct":      仓位占资本%,
            "position_size":     合约张数 (USDT本位),
            "risk_amount_usdt":  风险金额,
            "stop_loss_price":   止损价,
            "stop_loss_pct":     止损幅度%,
            "leverage":          实际杠杆,
            "kelly_optimal_pct": Kelly最优% (if win_rate),
            "acceptable":        是否可交易 (position_pct >= min),
        }
    """
    if entry_price <= 0 or atr_pct <= 0 or capital <= 0:
        return _invalid_result()

    # ── ATR 止损 ──────────────────────────────────────────────────────────
    stop_pct = (atr_pct / 100.0) * atr_multiplier

    # ── 风险金额 ──────────────────────────────────────────────────────────
    risk_amount = capital * risk_pct

    # ── 仓位计算 ──────────────────────────────────────────────────────────
    if stop_pct <= 0:
        return _invalid_result()

    # Position Value = Risk / Stop%
    position_value = risk_amount / stop_pct
    position_pct = position_value / capital
    position_size = position_value / entry_price

    # ── 上限约束 ──────────────────────────────────────────────────────────
    position_pct = min(position_pct, max_position_pct)
    position_value = capital * position_pct
    position_size = position_value / entry_price

    # ── 杠杆 ─────────────────────────────────────────────────────────────
    leverage = position_pct  # USDT本位: 1x 杠杆 = 100% 仓位

    # ── 止损价 ────────────────────────────────────────────────────────────
    stop_loss_price = entry_price * (1 - stop_pct)

    # ── Kelly 公式 (可选) ────────────────────────────────────────────────
    kelly_pct = None
    if win_rate is not None and 0 < win_rate < 1:
        # Kelly: f* = win_rate - (1-win_rate) / (avg_win/avg_loss)
        # 简化: 假设盈亏比 = 1.5 (ATR止盈3× / 止损2×)
        avg_win_loss_ratio = 1.5
        kelly_raw = win_rate - (1 - win_rate) / avg_win_loss_ratio
        # Half-Kelly 更稳健
        kelly_pct = max(0, kelly_raw * 0.5)
        # 用 Kelly 建议替代 ATR 仓位 (取较小值)
        if kelly_pct < position_pct:
            position_pct = kelly_pct
            position_value = capital * position_pct
            position_size = position_value / entry_price

    # ── 下限 ──────────────────────────────────────────────────────────────
    acceptable = position_pct >= min_position_pct

    return {
        "position_usdt": round(position_value, 2),
        "position_pct": round(position_pct * 100, 2),
        "position_size": round(position_size, 6),
        "risk_amount_usdt": round(risk_amount, 2),
        "stop_loss_price": round(stop_loss_price, 4),
        "stop_loss_pct": round(stop_pct * 100, 2),
        "leverage": round(leverage, 1),
        "kelly_optimal_pct": round(kelly_pct * 100, 2) if kelly_pct is not None else None,
        "acceptable": acceptable,
        "formula": "atr" if kelly_pct is None else "kelly_capped",
    }


def calculate_batch_positions(
    signals: list,
    capital: float,
    total_risk_pct: float = 0.04,  # 总风险上限 4%
) -> list:
    """
    为一批信号批量计算仓位，确保总风险不超上限。

    原理：
      如果 risk_pct=1%, 最多同时持有 4 个品种（4%总风险）
      超过 4 个时按信号分数排序，取前 4 个
    """
    max_signals = int(total_risk_pct / DEFAULT_RISK_PCT)
    signals_sorted = sorted(signals, key=lambda s: s.get("score", 0), reverse=True)

    results = []
    cumulative_risk = 0.0

    for sig in signals_sorted[:max_signals]:
        pos = calculate_position(
            capital=capital,
            entry_price=sig.get("entry_price", sig.get("last_price", 0)),
            atr_pct=sig.get("atr_pct", sig.get("vol_atr_pct", 4.0)),
            risk_pct=DEFAULT_RISK_PCT,
            win_rate=sig.get("win_rate"),
        )
        if pos["acceptable"]:
            pos["symbol"] = sig.get("symbol", "")
            pos["score"] = sig.get("score", 0)
            results.append(pos)
            cumulative_risk += DEFAULT_RISK_PCT

    return results


def _invalid_result() -> Dict[str, Any]:
    return {
        "position_usdt": 0, "position_pct": 0, "position_size": 0,
        "risk_amount_usdt": 0, "stop_loss_price": 0, "stop_loss_pct": 0,
        "leverage": 0, "kelly_optimal_pct": None, "acceptable": False,
        "formula": "invalid",
    }


# ── 便捷函数 ────────────────────────────────────────────────────────────────

def size_by_atr(
    capital: float, entry: float, atr_pct: float,
    risk_pct: float = 0.01, atr_mult: float = 1.8,
) -> Tuple[float, float]:
    """
    快速计算: (仓位USDT, 仓位%)
    """
    pos = calculate_position(capital, entry, atr_pct, risk_pct, atr_mult)
    return pos["position_usdt"], pos["position_pct"]


def suggest_stop(entry: float, atr_pct: float, atr_mult: float = 1.8) -> Tuple[float, float]:
    """
    快速计算: (止损价, 止损%)
    """
    stop_pct = (atr_pct / 100.0) * atr_mult
    return round(entry * (1 - stop_pct), 4), round(stop_pct * 100, 2)
