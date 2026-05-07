#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DRL元学习小时趋势启动扫描策略

设计目标：
1) 着重捕捉 1H 级别趋势启动；
2) 兼容程序扫描与回测入口；
3) 将 DQN/Double-DQN/Dueling、A2C、SAC、元学习、多时间框架思想映射到可运行的因子评分框架。
"""

from __future__ import annotations

from math import exp, log10, sqrt
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from strategies._shared.indicators import (
    _to_df, _aggregate_bars, _micro_pullback_continuation, _rsi_wilder as _rsi,
    _pct_change, _safe_float, _cfg_float, _clamp,
)

try:
    from src.scanner.base_scanner import BaseScannerStrategy, ScanCondition
    from src.scanner.ranking import build_opportunity_profile
    _HAS_SCANNER_BASE = True
except Exception:
    BaseScannerStrategy = object
    ScanCondition = None
    build_opportunity_profile = None
    _HAS_SCANNER_BASE = False


CONFIG_SCHEMA = {
    "min_score": {"type": "float", "default": 74.0, "label": "最低扫描分数"},
    "backtest_min_score": {"type": "float", "default": 70.0, "label": "回测最低入场分数"},
    "min_volume_24h": {"type": "float", "default": 8_000_000.0, "label": "最小24H成交额"},
    "top_n": {"type": "int", "default": 20, "label": "最多输出数量"},
    "allow_short": {"type": "bool", "default": True, "label": "允许空头"},
    "min_abs_edge": {"type": "float", "default": 0.22, "label": "最小优势"},
    "position_size": {"type": "float", "default": 0.10, "label": "回测仓位比例"},
    "entropy_alpha": {"type": "float", "default": 0.18, "label": "SAC熵权重"},
    "q_temperature": {"type": "float", "default": 0.85, "label": "Q softmax温度"},
    "double_q_blend": {"type": "float", "default": 0.36, "label": "Double-Q目标网络混合"},
    "meta_adapt_strength": {"type": "float", "default": 0.40, "label": "元学习适配强度"},
    "risk_penalty_strength": {"type": "float", "default": 0.70, "label": "风险惩罚强度"},
    "max_atr_pct": {"type": "float", "default": 8.2, "label": "最大ATR%"},
    "hourly_start_momentum_bps": {"type": "float", "default": 45.0, "label": "1H趋势启动最小动量bps"},
    "hourly_start_breakout_bps": {"type": "float", "default": 20.0, "label": "1H突破压强最小bps"},
    "hourly_start_volume_impulse": {"type": "float", "default": 1.18, "label": "1H启动最小量能脉冲"},
    "require_m3_pullback_confirmation": {"type": "bool", "default": True, "label": "要求3分钟回调企稳续势"},
    "m3_pullback_min_pct": {"type": "float", "default": 0.50, "label": "3分钟最小回调幅度%"},
    "m3_pullback_max_pct": {"type": "float", "default": 2.20, "label": "3分钟最大回调幅度%"},
    "m3_stabilization_bars": {"type": "int", "default": 4, "label": "3分钟企稳确认根数"},
}

_DEFAULT_CONFIG = {k: v["default"] for k, v in CONFIG_SCHEMA.items()}


class DRLMetaHourlyTrendStartScannerStrategy(BaseScannerStrategy if _HAS_SCANNER_BASE else object):
    """小时级趋势启动扫描器。"""

    required_bars = ["3m", "15m", "1H", "4H", "1D"]
    requires_derivative_metrics = True
    requires_on_chain_metrics = True
    name = "DRL元学习小时趋势启动扫描策略"
    description = "DQN/Double/Dueling + A2C + SAC + 元学习 + 多时间框架融合"
    strategy_type = "scan"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        merged = {**_DEFAULT_CONFIG, **(config or {})}
        self.config = merged
        self.last_analysis: Dict[str, Dict[str, Any]] = {}
        if _HAS_SCANNER_BASE and hasattr(super(), "__init__"):
            try:
                super().__init__(merged)
                self.config = merged
            except Exception:
                self.config = merged

    def _init_conditions(self):
        if ScanCondition is None or not hasattr(self, "add_condition"):
            return
        self.add_condition(ScanCondition(
            name="24H成交额",
            description="过滤流动性不足的交易对",
            field="volume_24h",
            operator=">=",
            value=self.config.get("min_volume_24h", 8_000_000.0),
        ))

    def get_config_schema(self) -> Dict[str, Any]:
        return dict(CONFIG_SCHEMA)

    def scan_symbol(self, symbol) -> Dict[str, Any]:
        snap = _build_snapshot(symbol, self.config)
        if not snap["valid"]:
            return _failed_result(symbol, snap["reason"])
        result = _score_snapshot(snap, self.config)
        self.last_analysis[str(getattr(symbol, "inst_id", ""))] = result
        return result

    def scan_all_symbols(self, symbols: List[Any]) -> Dict[str, Any]:
        min_volume = _cfg_float(self.config, "min_volume_24h", 8_000_000.0)
        candidates: List[Dict[str, Any]] = []
        for symbol in symbols:
            if float(getattr(symbol, "volume_24h", 0.0) or 0.0) < min_volume:
                continue
            result = self.scan_symbol(symbol)
            if result.get("passed"):
                candidates.append(result)

        candidates.sort(key=lambda x: _result_sort_key(x), reverse=True)
        top_n = int(self.config.get("top_n", 20) or 20)
        return {
            "type": "drl_meta_hourly_trend_start",
            "all_opportunities": candidates[:top_n],
        }

    def generate_signal(self, data, *args, **kwargs):
        if isinstance(data, (list, tuple)):
            data = {"klines_map": {"1H": list(data)}}
        if not isinstance(data, dict) or not (data.get("klines_map") or data.get("klines")):
            return None
        if not data.get("klines_map"):
            data = {**data, "klines_map": {"1H": data.get("klines") or []}}

        cfg = dict(self.config)
        cfg["min_score"] = _cfg_float(cfg, "backtest_min_score", _cfg_float(cfg, "min_score", 70.0))
        symbol = _symbol_from_backtest_data(data, cfg)
        result = _score_snapshot(_build_snapshot(symbol, cfg), cfg)
        if not result.get("passed"):
            return None
        direction = str(result.get("direction", "WAIT")).upper()
        if direction not in {"BUY", "SELL"}:
            return None
        return {
            "action": "BUY" if direction == "BUY" else "SHORT",
            "position_size": _cfg_float(cfg, "position_size", 0.10),
            "entry_price": float(result.get("last_price", 0.0) or 0.0),
            "reason": f"{result.get('category')} | 评分 {float(result.get('score', 0.0)):.1f}",
            "score": float(result.get("opportunity_score", result.get("score", 0.0)) or 0.0),
            "raw_result": result,
        }

    def reset_backtest_state(self):
        self.last_analysis.clear()


def _build_snapshot(symbol, config: Dict[str, Any]) -> Dict[str, Any]:
    inst_id = str(getattr(symbol, "inst_id", "") or "")
    extra = getattr(symbol, "extra_data", {}) or {}
    klines = extra.get("klines", {}) or {}

    m3 = _to_df(_get_klines(klines, "3m"))
    if m3.empty:
        m3 = _to_df(_get_klines(klines, "3M"))
    if m3.empty:
        m1 = _to_df(_get_klines(klines, "1m"))
        if len(m1) >= 120:
            m3 = _aggregate_bars(m1, 3)
    m15 = _to_df(_get_klines(klines, "15m"))
    h1 = _to_df(_get_klines(klines, "1H"))
    h4 = _to_df(_get_klines(klines, "4H"))
    d1 = _to_df(_get_klines(klines, "1D"))

    if h1.empty and len(m15) >= 16:
        h1 = _aggregate_bars(m15, 4)
    if h4.empty and len(h1) >= 12:
        h4 = _aggregate_bars(h1, 4)
    if d1.empty and len(h4) >= 24:
        d1 = _aggregate_bars(h4, 6)
    if len(h1) < 45:
        return {"valid": False, "symbol": inst_id, "reason": f"K线不足(1H={len(h1)})"}

    price = float(getattr(symbol, "last_price", 0.0) or h1["c"].iloc[-1])
    volume_24h = float(getattr(symbol, "volume_24h", 0.0) or (h1["c"] * h1["vol"]).tail(24).sum())
    change_24h = float(getattr(symbol, "price_change_24h", 0.0) or _pct_change(h1["c"], min(len(h1) - 1, 24)) * 100.0)
    funding_rate = _safe_float(extra.get("funding_rate"), 0.0) * 100.0
    oi_change = _safe_float(extra.get("open_interest_change_pct"), 0.0)
    on_chain = extra.get("on_chain", {}) if isinstance(extra.get("on_chain", {}), dict) else {}
    social = extra.get("social", {}) if isinstance(extra.get("social", {}), dict) else {}
    llm = extra.get("llm_factors", {}) if isinstance(extra.get("llm_factors", {}), dict) else {}

    m15_mom = _pct_change(m15["c"], 4) * 10_000 if len(m15) > 6 else _pct_change(h1["c"], 1) * 10_000
    h1_mom = _pct_change(h1["c"], 4) * 10_000
    h4_mom = _pct_change(h4["c"], 3) * 10_000 if len(h4) > 5 else 0.0
    d1_mom = _pct_change(d1["c"], 3) * 10_000 if len(d1) > 5 else 0.0
    trend_h1 = _ema_trend_score(h1["c"], 12, 34)
    trend_h4 = _ema_trend_score(h4["c"], 8, 21) if len(h4) > 25 else trend_h1 * 0.7
    trend_d1 = _ema_trend_score(d1["c"], 5, 13) if len(d1) > 16 else trend_h4 * 0.6
    trend_alignment = _clamp((trend_h1 * 0.45 + trend_h4 * 0.35 + trend_d1 * 0.20), -2.0, 2.0)
    micro_confirm = _micro_pullback_continuation(m3, trend_alignment, config)
    if bool(config.get("require_m3_pullback_confirmation", True)) and not micro_confirm["confirmed"]:
        return {"valid": False, "symbol": inst_id, "reason": f"3分钟回调续势未确认: {micro_confirm['reason']}"}
    rsi_1h = _rsi(h1["c"], 14)
    adx_1h = _adx_like(h1, 14)
    adx_4h = _adx_like(h4, 14) if len(h4) >= 25 else adx_1h * 0.7
    atr_pct = _atr_pct(h1, 14)
    realized_vol = _realized_vol_pct(h1["c"], 24)
    volume_impulse = _volume_impulse(h1, 24)
    breakout_bps = _breakout_pressure_bps(h1, 24)

    # CNN-like 多时间框架局部形态特征（通过固定卷积核提取）
    conv_m15 = _conv_feature(m15["c"] if not m15.empty else h1["c"], [1.0, 0.0, -1.0])
    conv_h1 = _conv_feature(h1["c"], [1.0, 0.0, -1.0])
    conv_h4 = _conv_feature(h4["c"] if len(h4) > 5 else h1["c"], [1.0, -2.0, 1.0])
    cnn_multi_tf = _clamp(conv_m15 * 0.35 + conv_h1 * 0.45 + conv_h4 * 0.20, -2.0, 2.0)

    whale_flow = _safe_float(on_chain.get("whale_flow"), 0.0)
    exchange_netflow = _safe_float(on_chain.get("exchange_netflow"), 0.0)
    active_addr_z = _safe_float(on_chain.get("active_addresses_z"), 0.0)
    llm_sent = _blend([
        _safe_float(llm.get("sentiment"), np.nan),
        _safe_float(llm.get("narrative_strength"), np.nan),
        _safe_float(social.get("sentiment"), np.nan),
        _safe_float(extra.get("news_sentiment"), np.nan),
    ], [0.35, 0.30, 0.25, 0.10])

    factors = {
        "momentum_1h": _clamp(h1_mom / 180.0, -2.5, 2.5),
        "momentum_4h": _clamp(h4_mom / 280.0, -2.5, 2.5),
        "momentum_1d": _clamp(d1_mom / 380.0, -2.5, 2.5),
        "trend_alignment": trend_alignment,
        "m3_pullback_score": micro_confirm["score"],
        "breakout": _clamp(breakout_bps / 120.0, -2.5, 2.5),
        "volume_impulse": _clamp(volume_impulse - 1.0, -1.5, 2.5),
        "adx_strength": _clamp((adx_1h - 18.0) / 18.0, -1.5, 2.0),
        "funding_contra": _clamp(-funding_rate / 0.08, -2.0, 2.0),
        "oi_confirmation": _clamp(oi_change / 8.0, -2.0, 2.0) * _direction_sign(h1_mom, trend_alignment),
        "cnn_multi_tf": cnn_multi_tf,
        "on_chain_accumulation": _clamp(whale_flow - exchange_netflow * 0.65 + active_addr_z * 0.25, -2.0, 2.0),
        "llm_sentiment": _clamp(llm_sent, -2.0, 2.0),
        "risk_vol": _clamp((realized_vol - 6.0) / 4.0, -1.5, 2.5),
        "risk_atr": _clamp((atr_pct - 3.0) / 3.0, -1.5, 2.5),
        "liquidity": _clamp((log10(max(volume_24h, 1.0)) - 6.2) / 2.0, -1.0, 2.0),
    }

    meta_feedback = extra.get("strategy_feedback", {}) if isinstance(extra.get("strategy_feedback"), dict) else {}
    regime = _detect_regime(trend_alignment, adx_1h, realized_vol, h1_mom)

    return {
        "valid": True,
        "reason": "",
        "symbol": inst_id,
        "last_price": price,
        "volume_24h": volume_24h,
        "price_change_24h": change_24h,
        "funding_rate_pct": funding_rate,
        "open_interest_change_pct": oi_change,
        "atr_pct": atr_pct,
        "realized_vol_pct": realized_vol,
        "rsi_1h": rsi_1h,
        "adx_1h": adx_1h,
        "adx_4h": adx_4h,
        "m15_mom_bps": m15_mom,
        "m3_pullback_confirmed": micro_confirm["confirmed"],
        "m3_structure_state": micro_confirm["state"],
        "m3_pullback_reason": micro_confirm["reason"],
        "m3_pullback_pct": micro_confirm["pullback_pct"],
        "m3_impulse_pct": micro_confirm["impulse_pct"],
        "h1_mom_bps": h1_mom,
        "h4_mom_bps": h4_mom,
        "d1_mom_bps": d1_mom,
        "breakout_bps": breakout_bps,
        "volume_impulse": volume_impulse,
        "regime": regime,
        "meta_feedback": meta_feedback,
        "factors": factors,
    }


def _score_snapshot(snap: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    if not snap.get("valid", True) or "factors" not in snap:
        return {"passed": False, "reason": snap.get("reason", "快照无效")}
    factors = snap["factors"]
    meta_weights = _meta_regime_weights(snap["regime"], snap["meta_feedback"], config)

    # Dueling DQN: V(s) + A(s,a)
    state_value = _clamp(
        0.24 * factors["liquidity"]
        + 0.22 * factors["trend_alignment"]
        + 0.12 * factors["volume_impulse"]
        - 0.20 * factors["risk_vol"]
        - 0.18 * factors["risk_atr"],
        -2.0,
        2.0,
    )

    long_adv = _clamp(
        meta_weights["trend"] * factors["trend_alignment"]
        + meta_weights["momentum"] * factors["momentum_1h"]
        + 0.20 * factors["breakout"]
        + 0.12 * factors["cnn_multi_tf"]
        + 0.10 * factors["oi_confirmation"]
        + 0.08 * factors["on_chain_accumulation"]
        + 0.06 * factors["llm_sentiment"]
        - 0.10 * max(factors["risk_vol"], 0.0),
        -3.0,
        3.0,
    )

    short_adv = _clamp(
        meta_weights["trend"] * (-factors["trend_alignment"])
        + meta_weights["momentum"] * (-factors["momentum_1h"])
        + 0.20 * (-factors["breakout"])
        + 0.12 * (-factors["cnn_multi_tf"])
        + 0.10 * (-factors["oi_confirmation"])
        + 0.06 * (-factors["llm_sentiment"])
        + 0.05 * max(factors["risk_vol"], 0.0),
        -3.0,
        3.0,
    )

    wait_adv = _clamp(0.30 * (factors["risk_vol"] + factors["risk_atr"]) - 0.18 * abs(factors["trend_alignment"]), -2.5, 2.5)
    mean_adv = (long_adv + short_adv + wait_adv) / 3.0
    q_long_online = state_value + (long_adv - mean_adv)
    q_short_online = state_value + (short_adv - mean_adv)
    q_wait_online = state_value + (wait_adv - mean_adv)

    # Double DQN: 使用慢速目标值减少过估计
    target_scale = _cfg_float(config, "double_q_blend", 0.36)
    q_long_target = _clamp(0.62 * factors["momentum_4h"] + 0.38 * factors["trend_alignment"], -2.8, 2.8)
    q_short_target = _clamp(-0.62 * factors["momentum_4h"] - 0.38 * factors["trend_alignment"], -2.8, 2.8)
    q_wait_target = _clamp(0.50 * (factors["risk_vol"] + factors["risk_atr"]) - 0.12 * abs(factors["momentum_4h"]), -2.8, 2.8)
    q_long = (1.0 - target_scale) * q_long_online + target_scale * q_long_target
    q_short = (1.0 - target_scale) * q_short_online + target_scale * q_short_target
    q_wait = (1.0 - target_scale) * q_wait_online + target_scale * q_wait_target

    # A2C: Advantage = Q - V
    adv_long = q_long - state_value
    adv_short = q_short - state_value
    adv_wait = q_wait - state_value

    # SAC: 最大熵目标
    probs = _softmax([q_long, q_short, q_wait], _cfg_float(config, "q_temperature", 0.85))
    entropy = _normalized_entropy(probs)
    sac_alpha = _cfg_float(config, "entropy_alpha", 0.18)
    objective_long = q_long + sac_alpha * entropy
    objective_short = q_short + sac_alpha * entropy
    objective_wait = q_wait + sac_alpha * entropy

    objectives = {"BUY": objective_long, "SELL": objective_short, "WAIT": objective_wait}
    direction = max(objectives, key=objectives.get)
    best = float(objectives[direction])
    second = sorted(objectives.values(), reverse=True)[1]
    confidence = _clamp(best - second, 0.0, 3.0)

    allow_short = bool(config.get("allow_short", True))
    if direction == "SELL" and not allow_short:
        direction = "WAIT"

    startup_gate = _hourly_startup_gate(snap, direction, config)
    risk_penalty = _cfg_float(config, "risk_penalty_strength", 0.70) * max(0.0, factors["risk_vol"] + factors["risk_atr"] * 0.8)
    raw_edge = best - objective_wait if direction in {"BUY", "SELL"} else 0.0
    edge = _clamp(raw_edge * (0.70 + 0.30 * confidence) - risk_penalty * 0.20, -2.8, 2.8)
    score = _edge_to_score(edge, confidence, startup_gate, snap)

    passed = (
        direction in {"BUY", "SELL"}
        and startup_gate
        and score >= _cfg_float(config, "min_score", 74.0)
        and abs(edge) >= _cfg_float(config, "min_abs_edge", 0.22)
        and snap["atr_pct"] <= _cfg_float(config, "max_atr_pct", 8.2)
    )

    category = "DRL小时趋势多头启动" if direction == "BUY" else "DRL小时趋势空头启动" if direction == "SELL" else "DRL小时趋势观察"
    signals = [
        f"{category} 评分 {score:.1f}",
        f"Q(L/S/W)=({q_long:+.2f},{q_short:+.2f},{q_wait:+.2f}) | A2C优势({adv_long:+.2f},{adv_short:+.2f},{adv_wait:+.2f})",
        f"SAC熵={entropy:.2f} 置信度={confidence:.2f} 边际={edge:+.3f}",
        f"1H动量={snap['h1_mom_bps']:+.1f}bps 4H动量={snap['h4_mom_bps']:+.1f}bps 突破压强={snap['breakout_bps']:+.1f}bps",
        f"量能脉冲={snap['volume_impulse']:.2f}x ADX1H={snap['adx_1h']:.1f} ATR%={snap['atr_pct']:.2f} Regime={snap['regime']}",
    ]

    ranking_factors = {
        "trend": _clamp(50 + factors["trend_alignment"] * 24 + factors["momentum_4h"] * 8, 0, 100),
        "trigger": _clamp(50 + factors["breakout"] * 18 + factors["momentum_1h"] * 18 + confidence * 10, 0, 100),
        "volume": _clamp(50 + factors["volume_impulse"] * 26 + factors["liquidity"] * 12, 0, 100),
        "location": _clamp(55 + (50.0 - abs(snap["rsi_1h"] - 55.0)) * 0.6, 10, 95),
        "freshness": _clamp(50 + snap["m15_mom_bps"] / 6.0 + snap["h1_mom_bps"] / 10.0, 0, 95),
        "risk": _clamp(84 - max(factors["risk_vol"], 0.0) * 18 - max(factors["risk_atr"], 0.0) * 15, 10, 95),
    }

    result = {
        "symbol": snap["symbol"],
        "passed": passed,
        "score": round(float(score), 2),
        "direction": direction if direction in {"BUY", "SELL"} else "WAIT",
        "signals": signals,
        "category": category,
        "strategy_category": category,
        "last_price": snap["last_price"],
        "volume_24h": snap["volume_24h"],
        "price_change_24h": snap["price_change_24h"],
        "ranking_factors": ranking_factors,
        "metrics": {
            "alpha_edge": round(float(edge), 6),
            "raw_edge": round(float(raw_edge), 6),
            "confidence": round(float(confidence), 6),
            "entropy": round(float(entropy), 6),
            "q_long": round(float(q_long), 6),
            "q_short": round(float(q_short), 6),
            "q_wait": round(float(q_wait), 6),
            "state_value": round(float(state_value), 6),
            "adv_long": round(float(adv_long), 6),
            "adv_short": round(float(adv_short), 6),
            "atr_pct": round(float(snap["atr_pct"]), 6),
            "realized_vol_pct": round(float(snap["realized_vol_pct"]), 6),
            "regime": snap["regime"],
        },
        "factor_scores": {k: round(float(v), 6) for k, v in factors.items()},
        "details": {
            "机会类型": category,
            "评估": " | ".join(signals),
            "综合边际": f"{edge:+.3f}",
            "置信度": f"{confidence:.2f}",
            "熵": f"{entropy:.2f}",
            "1H动量bps": f"{snap['h1_mom_bps']:+.1f}",
            "突破压强bps": f"{snap['breakout_bps']:+.1f}",
            "ADX1H": f"{snap['adx_1h']:.1f}",
            "ATR%": f"{snap['atr_pct']:.2f}",
            "资金费率%": f"{snap['funding_rate_pct']:+.4f}",
            "OI变化%": f"{snap['open_interest_change_pct']:+.2f}",
            "启动门槛通过": "是" if startup_gate else "否",
            "3分钟回调确认": "是" if snap.get("m3_pullback_confirmed") else "否",
            "3分钟结构": str(snap.get("m3_structure_state", "-")),
            "3分钟回调幅度%": f"{float(snap.get('m3_pullback_pct', 0.0)):.2f}",
            "3分钟原趋势脉冲%": f"{float(snap.get('m3_impulse_pct', 0.0)):.2f}",
            "3分钟回调结论": str(snap.get("m3_pullback_reason", "-")),
        },
    }
    if build_opportunity_profile:
        try:
            result.update(build_opportunity_profile(score, result["direction"], snap["volume_24h"], ranking_factors, signals))
        except Exception:
            pass
    return result


def _hourly_startup_gate(snap: Dict[str, Any], direction: str, config: Dict[str, Any]) -> bool:
    h1_mom_min = _cfg_float(config, "hourly_start_momentum_bps", 45.0)
    breakout_min = _cfg_float(config, "hourly_start_breakout_bps", 20.0)
    vol_min = _cfg_float(config, "hourly_start_volume_impulse", 1.18)
    if direction == "BUY":
        return (
            snap["h1_mom_bps"] >= h1_mom_min
            and snap["breakout_bps"] >= breakout_min
            and snap["volume_impulse"] >= vol_min
            and snap["adx_1h"] >= 17.0
        )
    if direction == "SELL":
        return (
            snap["h1_mom_bps"] <= -h1_mom_min
            and snap["breakout_bps"] <= -breakout_min
            and snap["volume_impulse"] >= vol_min
            and snap["adx_1h"] >= 17.0
        )
    return False


def _meta_regime_weights(regime: str, feedback: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, float]:
    base = {
        "trend": 0.42,
        "momentum": 0.38,
        "reversal": 0.20,
    }
    if regime == "TREND":
        base = {"trend": 0.48, "momentum": 0.40, "reversal": 0.12}
    elif regime == "RANGE":
        base = {"trend": 0.30, "momentum": 0.24, "reversal": 0.46}
    elif regime == "VOLATILE":
        base = {"trend": 0.36, "momentum": 0.28, "reversal": 0.36}

    # Learning-Based Linear Balancer: 依据历史反馈做线性再平衡
    trend_wr = _safe_float(feedback.get("trend_win_rate"), 0.52)
    range_wr = _safe_float(feedback.get("range_win_rate"), 0.48)
    adapt = _cfg_float(config, "meta_adapt_strength", 0.40)
    delta = _clamp((trend_wr - range_wr) * 2.0, -0.30, 0.30) * adapt
    base["trend"] = max(0.05, base["trend"] + delta)
    base["momentum"] = max(0.05, base["momentum"] + delta * 0.7)
    base["reversal"] = max(0.05, base["reversal"] - delta * 1.7)
    total = base["trend"] + base["momentum"] + base["reversal"]
    return {k: v / total for k, v in base.items()}


def _detect_regime(trend_alignment: float, adx_1h: float, realized_vol_pct: float, h1_mom_bps: float) -> str:
    if adx_1h >= 22.0 and abs(trend_alignment) >= 0.45 and abs(h1_mom_bps) >= 35.0:
        return "TREND"
    if realized_vol_pct >= 8.0:
        return "VOLATILE"
    return "RANGE"


def _edge_to_score(edge: float, confidence: float, startup_gate: bool, snap: Dict[str, Any]) -> float:
    base = 34.0 + abs(edge) * 11.5 + confidence * 4.5
    quality = 0.0
    quality += max(0.0, snap["volume_impulse"] - 1.0) * 3.0
    quality += max(0.0, (snap["adx_1h"] - 16.0) / 10.0) * 2.8
    quality += max(0.0, 1.0 - snap["atr_pct"] / 10.0) * 2.5
    gate_bonus = 1.5 if startup_gate else -9.0
    return _clamp(base + quality + gate_bonus, 0.0, 100.0)


def _result_sort_key(item: Dict[str, Any]) -> Tuple[float, float, float]:
    return (
        float(item.get("opportunity_score", item.get("score", 0.0)) or 0.0),
        float(item.get("score", 0.0) or 0.0),
        float(item.get("volume_24h", 0.0) or 0.0),
    )


def _get_klines(klines_map: Dict[str, List], bar: str) -> List:
    aliases = {
        "15m": ["15m", "15M"],
        "1H": ["1H", "1h", "60m", "60M"],
        "4H": ["4H", "4h", "240m", "240M"],
        "1D": ["1D", "1d", "D", "day"],
    }
    for key in aliases.get(bar, [bar, bar.lower(), bar.upper()]):
        if key in klines_map and klines_map.get(key):
            return klines_map.get(key)
    return []


def _ema_trend_score(close: pd.Series, fast: int, slow: int) -> float:
    if len(close) < slow + 3:
        return 0.0
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    slope = float(ema_fast.diff().tail(3).mean() or 0.0)
    spread = (float(ema_fast.iloc[-1]) / max(float(ema_slow.iloc[-1]), 1e-9) - 1.0) * 100.0
    score = spread * 2.2 + slope / max(float(close.iloc[-1]), 1e-9) * 10_000 * 0.8
    return _clamp(score / 30.0, -2.0, 2.0)


def _atr_pct(df: pd.DataFrame, period: int = 14) -> float:
    if len(df) < period + 2:
        return 0.0
    prev_close = df["c"].shift(1)
    tr = pd.concat([(df["h"] - df["l"]).abs(), (df["h"] - prev_close).abs(), (df["l"] - prev_close).abs()], axis=1).max(axis=1)
    atr = float(tr.ewm(alpha=1 / period, adjust=False).mean().iloc[-1] or 0.0)
    price = float(df["c"].iloc[-1] or 0.0)
    return atr / price * 100.0 if price > 0 else 0.0


def _adx_like(df: pd.DataFrame, period: int = 14) -> float:
    if len(df) < period + 5:
        return 15.0
    high = df["h"]
    low = df["l"]
    close = df["c"]
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    prev_close = close.shift(1)
    tr = pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False).mean().replace(0, np.nan)
    plus_di = 100.0 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / atr
    minus_di = 100.0 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / atr
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100.0
    adx = dx.ewm(alpha=1 / period, adjust=False).mean()
    val = float(adx.iloc[-1]) if np.isfinite(adx.iloc[-1]) else 15.0
    return _clamp(val, 5.0, 60.0)


def _realized_vol_pct(close: pd.Series, window: int = 24) -> float:
    returns = close.pct_change().dropna().tail(window)
    if returns.empty:
        return 0.0
    return float(returns.std(ddof=0) * sqrt(max(len(returns), 1)) * 100.0)


def _volume_impulse(df: pd.DataFrame, window: int = 24) -> float:
    if len(df) < window + 3:
        return 1.0
    base = float(df["vol"].iloc[-(window + 1):-1].median() or 0.0)
    latest = float(df["vol"].tail(3).mean() or 0.0)
    return latest / base if base > 0 else 1.0


def _breakout_pressure_bps(df: pd.DataFrame, window: int = 24) -> float:
    if len(df) < window + 2:
        return 0.0
    high_ref = float(df["h"].iloc[-(window + 1):-1].max() or 0.0)
    low_ref = float(df["l"].iloc[-(window + 1):-1].min() or 0.0)
    close = float(df["c"].iloc[-1] or 0.0)
    if close <= 0:
        return 0.0
    up = (close / max(high_ref, 1e-9) - 1.0) * 10_000
    down = (close / max(low_ref, 1e-9) - 1.0) * 10_000
    return _clamp(up if abs(up) >= abs(down) else down, -600.0, 600.0)


def _conv_feature(series: pd.Series, kernel: List[float]) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna().values.astype(float)
    k = np.array(kernel, dtype=float)
    if len(values) < len(k) + 2:
        return 0.0
    norm = np.maximum(np.abs(values[-len(k):]).mean(), 1e-9)
    conv = float(np.dot(values[-len(k):], k) / norm)
    return _clamp(conv, -3.0, 3.0)


def _softmax(values: List[float], temperature: float) -> List[float]:
    t = max(float(temperature), 1e-6)
    arr = np.array(values, dtype=float) / t
    arr = arr - np.max(arr)
    exps = np.exp(arr)
    denom = float(np.sum(exps) or 1.0)
    return (exps / denom).tolist()


def _normalized_entropy(probs: List[float]) -> float:
    eps = 1e-12
    n = max(len(probs), 1)
    ent = -sum(float(p) * np.log(float(p) + eps) for p in probs)
    return _clamp(ent / np.log(n + eps), 0.0, 1.0)


def _symbol_from_backtest_data(data: Dict[str, Any], config: Dict[str, Any]):
    klines_map = data.get("klines_map") or {}
    h1 = _to_df(_get_klines(klines_map, "1H") or _get_klines(klines_map, "15m") or data.get("klines") or [])
    last_price = float(h1["c"].iloc[-1]) if not h1.empty else 0.0
    volume = float((h1["c"] * h1["vol"]).tail(48).sum()) if not h1.empty else 0.0
    extra = {
        "klines": klines_map,
        "funding_rate": data.get("funding_rate", 0.0),
        "open_interest_change_pct": data.get("open_interest_change_pct", 0.0),
        "on_chain": data.get("on_chain", {}),
        "social": data.get("social", {}),
        "llm_factors": data.get("llm_factors", {}),
        "news_sentiment": data.get("news_sentiment", 0.0),
        "strategy_feedback": data.get("strategy_feedback", {}),
    }
    return _MinimalSymbol(
        inst_id=str(config.get("inst_id", "BACKTEST-SYMBOL") or "BACKTEST-SYMBOL"),
        last_price=last_price,
        volume_24h=volume,
        price_change_24h=_pct_change(h1["c"], min(len(h1) - 1, 24)) * 100.0 if not h1.empty else 0.0,
        extra_data=extra,
    )


class _MinimalSymbol:
    def __init__(self, inst_id: str, last_price: float, volume_24h: float, price_change_24h: float, extra_data: Dict[str, Any]):
        self.inst_id = inst_id
        self.last_price = last_price
        self.volume_24h = volume_24h
        self.price_change_24h = price_change_24h
        self.high_24h = 0.0
        self.low_24h = 0.0
        self.open_interest = 0.0
        self.extra_data = extra_data


def _failed_result(symbol, reason: str) -> Dict[str, Any]:
    return {
        "symbol": str(getattr(symbol, "inst_id", "") or ""),
        "passed": False,
        "score": 0.0,
        "direction": "WAIT",
        "signals": [],
        "details": {"状态": reason},
        "metrics": {},
    }


def _blend(values: List[float], weights: List[float]) -> float:
    total = 0.0
    denom = 0.0
    for value, weight in zip(values, weights):
        if np.isfinite(value):
            total += float(value) * float(weight)
            denom += float(weight)
    return total / denom if denom > 0 else 0.0


def _direction_sign(primary: float, fallback: float) -> float:
    if abs(float(primary or 0.0)) > 1e-9:
        return 1.0 if primary > 0 else -1.0
    if abs(float(fallback or 0.0)) > 1e-9:
        return 1.0 if fallback > 0 else -1.0
    return 1.0


STRATEGY_NAME = "DRL元学习小时趋势启动扫描策略"
STRATEGY_TYPE = "scan"
STRATEGY_CLASS = DRLMetaHourlyTrendStartScannerStrategy
BACKTEST_CLASS = DRLMetaHourlyTrendStartScannerStrategy
