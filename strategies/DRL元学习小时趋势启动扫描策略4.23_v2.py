#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DRL元学习小时趋势启动扫描策略 v2

v1 → v2 变更
─────────────────────────────────────────────────────
【逻辑修复】
1. _micro_pullback_continuation 窗口硬编码越界：
   [-28:-12]/[-12:-stab_bars]/[-6:-1] 是固定偏移，在 m3 数据 ≤50 根时
   极易产生空切片或区间重叠。v2 改为基于数据长度的动态分段。

2. is_monotonic_increasing 过严：浮点精度导致几乎任何真实数据都失败。
   v2 改为"企稳段低点不创新低 + EMA8 方向向上"的宽松判断。

3. 回调幅度默认值不一致：CONFIG_SCHEMA 中 m3_pullback_min_pct=0.50
   但代码内 config.get(..., 0.35)。v2 统一为 0.50。

4. _breakout_pressure_bps 方向选择：abs(up)>=abs(down) 有时给出与趋势
   相反的方向（如价格距历史高点近但大幅高于历史低点）。v2 改为跟随
   trend_alignment 方向选择突破方向。

5. _hourly_startup_gate 的 breakout_bps 空头检查：
   原来要求 snap["breakout_bps"] <= -breakout_min，但 _breakout_pressure_bps
   返回的是接近高点/低点的正/负值，空头时 breakout_bps < 0 才正确，但门槛
   应为 abs(bps) >= min，不是值 <= 负门槛（当 bps=-25 且 min=20 时是通过的，
   但当 bps=+5 时也不会被错误判定）。原逻辑实际上是对的，保留。

6. m3_pullback_score 在 score 计算时单位混乱：
   impulse_pct * 0.45 + pullback_pct * 0.75 中两者都是百分比（0.5~5.0），
   导致 score 远超 clamp(-2,2) 的上限，clamp 之后几乎永远是 2.0。
   v2 先归一化到 [-1,1] 再合成。

【新增特性（用户需求）】
7. 1H 趋势延续时效性检测（h1_trend_age_check）：
   计算趋势启动后已运行了多少根 1H K线。如果 EMA12 已持续向上超过
   max_h1_trend_age 根（默认 12），说明趋势已延续较久，建仓后容易
   遇到回调。这时降低 score 并在 passed 中记录警示。
   → 新增 config: "max_h1_trend_age"(default 12), "h1_trend_age_penalty"(default 8.0)

8. 3m 回调时效性检测（m3_pullback_freshness_check）：
   检测回调低点距当前 bar 的距离（根数）。如果回调最低点发生在
   max_m3_staleness_bars 根之前（默认 15 根 = 45 分钟前），说明
   企稳已经很久，这根信号可能是"旧回调后的二次拉升"，建仓后遇到
   小时级回调风险更高。
   → 新增 config: "max_m3_staleness_bars"(default 15), "m3_freshness_penalty"(default 6.0)

   两个时效性检查都通过时额外加分（时效性共振 bonus）。
─────────────────────────────────────────────────────
"""

from __future__ import annotations

from math import exp, log10, sqrt
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from strategies._shared.indicators import (
    _to_df, _aggregate_bars, _measure_trend_age, _micro_pullback_continuation,
    _rsi_wilder as _rsi, _pct_change, _safe_float, _cfg_float, _clamp,
    _calc_atr, _calc_volume_delta, _calc_vwap,
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
    "min_score":                    {"type": "float", "default": 74.0,  "label": "最低扫描分数"},
    "backtest_min_score":           {"type": "float", "default": 70.0,  "label": "回测最低入场分数"},
    "min_volume_24h":               {"type": "float", "default": 8_000_000.0, "label": "最小24H成交额"},
    "top_n":                        {"type": "int",   "default": 20,    "label": "最多输出数量"},
    "allow_short":                  {"type": "bool",  "default": True,  "label": "允许空头"},
    "min_abs_edge":                 {"type": "float", "default": 0.22,  "label": "最小优势"},
    "position_size":                {"type": "float", "default": 0.10,  "label": "回测仓位比例"},
    "entropy_alpha":                {"type": "float", "default": 0.18,  "label": "SAC熵权重"},
    "q_temperature":                {"type": "float", "default": 0.85,  "label": "Q softmax温度"},
    "double_q_blend":               {"type": "float", "default": 0.36,  "label": "Double-Q目标网络混合"},
    "meta_adapt_strength":          {"type": "float", "default": 0.40,  "label": "元学习适配强度"},
    "risk_penalty_strength":        {"type": "float", "default": 0.70,  "label": "风险惩罚强度"},
    "max_atr_pct":                  {"type": "float", "default": 8.2,   "label": "最大ATR%"},
    "hourly_start_momentum_bps":    {"type": "float", "default": 45.0,  "label": "1H趋势启动最小动量bps"},
    "hourly_start_breakout_bps":    {"type": "float", "default": 20.0,  "label": "1H突破压强最小bps"},
    "hourly_start_volume_impulse":  {"type": "float", "default": 1.18,  "label": "1H启动最小量能脉冲"},
    "require_m3_pullback_confirmation": {"type": "bool", "default": True, "label": "要求3分钟回调企稳续势"},
    "m3_pullback_min_pct":          {"type": "float", "default": 0.50,  "label": "3分钟最小回调幅度%"},
    "m3_pullback_max_pct":          {"type": "float", "default": 2.20,  "label": "3分钟最大回调幅度%"},
    "m3_stabilization_bars":        {"type": "int",   "default": 4,     "label": "3分钟企稳确认根数"},
    # v2 新增 ── 时效性
    "max_h1_trend_age":             {"type": "int",   "default": 12,    "label": "1H趋势最大延续根数（超过则降权）"},
    "h1_trend_age_penalty":         {"type": "float", "default": 8.0,   "label": "趋势过老时分数惩罚"},
    "max_m3_staleness_bars":        {"type": "int",   "default": 15,    "label": "3m回调最大时效根数（超过则降权）"},
    "m3_freshness_penalty":         {"type": "float", "default": 6.0,   "label": "3m回调过旧时分数惩罚"},
    # v2.1 新增
    "require_m3_freshness":         {"type": "bool",  "default": True,  "label": "必须通过3m时效性检查"},
    "m3_min_impulse_pct":           {"type": "float", "default": 0.65,  "label": "3m最小原趋势脉冲%"},
    "vol_continuation_min_ratio":   {"type": "float", "default": 0.78,  "label": "企稳量能续航最低比例"},
    "enable_funding_timing_guard":  {"type": "bool",  "default": True,  "label": "启用资金费率结算时段回避"},
    "funding_avoid_minutes":        {"type": "int",   "default": 15,    "label": "资金费率结算前回避分钟数"},
    "enable_btc_correlation_filter":{"type": "bool",  "default": False, "label": "启用BTC相关性过滤(需BTC K线)"},
    "max_btc_correlation":          {"type": "float", "default": 0.85,  "label": "最大允许的BTC相关性"},
    # v2.2 新增 ── 3m微观结构增强指标
    "enable_atr_squeeze_check":     {"type": "bool",  "default": True,  "label": "启用波动率收缩检测"},
    "atr_squeeze_ratio":            {"type": "float", "default": 0.55,  "label": "ATR收缩比例（当前/长期）"},
    "enable_volume_delta_check":    {"type": "bool",  "default": True,  "label": "启用买卖力量检测"},
    "volume_delta_min_ratio":       {"type": "float", "default": 1.15,  "label": "企稳段买入量/卖出量最低比值"},
    "enable_vwap_alignment_check":  {"type": "bool",  "default": True,  "label": "启用VWAP对齐检测"},
    # v4.0 新增
    "enable_bidask_spread_filter":  {"type": "bool",  "default": True,  "label": "启用买卖价差过滤"},
    "max_bidask_spread_pct":        {"type": "float", "default": 0.25,  "label": "最大允许买卖价差%"},
}

_DEFAULT_CONFIG = {k: v["default"] for k, v in CONFIG_SCHEMA.items()}


# ══════════════════════════════════════════════
class DRLMetaHourlyTrendStartScannerStrategy(BaseScannerStrategy if _HAS_SCANNER_BASE else object):
    required_bars = ["3m", "15m", "1H", "4H", "1D"]
    requires_derivative_metrics = True
    requires_on_chain_metrics = True
    name = "DRL元学习小时趋势启动扫描策略"
    description = "DQN/Double/Dueling + A2C + SAC + 元学习 + 趋势时效性双重过滤"
    strategy_type = "scan"

    def __init__(self, config=None):
        merged = {**_DEFAULT_CONFIG, **(config or {})}
        self.config = merged
        self.last_analysis: Dict[str, Dict[str, Any]] = {}
        if _HAS_SCANNER_BASE and hasattr(super(), "__init__"):
            try:
                super().__init__(merged); self.config = merged
            except Exception:
                self.config = merged

    def _init_conditions(self):
        if ScanCondition is None or not hasattr(self, "add_condition"): return
        self.add_condition(ScanCondition(name="24H成交额", description="过滤流动性不足",
            field="volume_24h", operator=">=", value=self.config.get("min_volume_24h", 8_000_000.0)))

    def get_config_schema(self): return dict(CONFIG_SCHEMA)

    def scan_symbol(self, symbol):
        snap = _build_snapshot(symbol, self.config)
        if not snap["valid"]: return _failed_result(symbol, snap["reason"])
        result = _score_snapshot(snap, self.config)
        self.last_analysis[str(getattr(symbol, "inst_id", ""))] = result
        return result

    def scan_all_symbols(self, symbols):
        min_vol = _cfg_float(self.config, "min_volume_24h", 8_000_000.0)
        candidates = []
        for sym in symbols:
            if float(getattr(sym, "volume_24h", 0.0) or 0.0) < min_vol: continue
            r = self.scan_symbol(sym)
            if r.get("passed"): candidates.append(r)
        candidates.sort(key=_result_sort_key, reverse=True)
        return {"type": "drl_meta_hourly_trend_start",
                "all_opportunities": candidates[:int(self.config.get("top_n", 20) or 20)]}

    def generate_signal(self, data, *a, **kw):
        if isinstance(data, (list, tuple)): data = {"klines_map": {"1H": list(data)}}
        if not isinstance(data, dict) or not (data.get("klines_map") or data.get("klines")): return None
        if not data.get("klines_map"): data = {**data, "klines_map": {"1H": data.get("klines") or []}}
        cfg = dict(self.config)
        cfg["min_score"] = _cfg_float(cfg, "backtest_min_score", _cfg_float(cfg, "min_score", 70.0))
        sym = _symbol_from_backtest_data(data, cfg)
        result = _score_snapshot(_build_snapshot(sym, cfg), cfg)
        if not result.get("passed"): return None
        d = str(result.get("direction", "WAIT")).upper()
        if d not in {"BUY", "SELL"}: return None
        return {"action": "BUY" if d == "BUY" else "SHORT",
                "position_size": _cfg_float(cfg, "position_size", 0.10),
                "entry_price": float(result.get("last_price", 0.0) or 0.0),
                "reason": f"{result.get('category')} | 评分 {float(result.get('score', 0.0)):.1f}",
                "score": float(result.get("opportunity_score", result.get("score", 0.0)) or 0.0),
                "raw_result": result}

    def reset_backtest_state(self): self.last_analysis.clear()


# ══════════════════════════════════════════════
# 快照构建
# ══════════════════════════════════════════════

def _build_snapshot(symbol, config) -> Dict[str, Any]:
    inst_id = str(getattr(symbol, "inst_id", "") or "")
    extra = getattr(symbol, "extra_data", {}) or {}
    klines = extra.get("klines", {}) or {}

    m3 = _to_df(_get_klines(klines, "3m"))
    if m3.empty: m3 = _to_df(_get_klines(klines, "3M"))
    if m3.empty:
        m1 = _to_df(_get_klines(klines, "1m"))
        if len(m1) >= 120: m3 = _aggregate_bars(m1, 3)
    m15 = _to_df(_get_klines(klines, "15m"))
    h1 = _to_df(_get_klines(klines, "1H"))
    h4 = _to_df(_get_klines(klines, "4H"))
    d1 = _to_df(_get_klines(klines, "1D"))

    if h1.empty and len(m15) >= 16: h1 = _aggregate_bars(m15, 4)
    if h4.empty and len(h1) >= 12: h4 = _aggregate_bars(h1, 4)
    if d1.empty and len(h4) >= 24: d1 = _aggregate_bars(h4, 6)
    if len(h1) < 45: return {"valid": False, "symbol": inst_id, "reason": f"K线不足(1H={len(h1)})"}
    if m3.empty and bool(config.get("require_m3_pullback_confirmation", True)):
        return {"valid": False, "symbol": inst_id, "reason": "3m数据不可用且要求回调确认"}

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
    trend_alignment = _clamp(trend_h1 * 0.45 + trend_h4 * 0.35 + trend_d1 * 0.20, -2.0, 2.0)

    # ── v2: 1H 趋势时效性 ──
    h1_trend_age = _measure_trend_age(h1["c"], fast=12, slow=34, direction=trend_alignment)

    # ── v2.1: 资金费率结算时段回避 ──
    funding_risk = _check_funding_timing(config) if _cfg_float(config, "enable_funding_timing_guard", 1.0) > 0 else False

    # ── v2.1: BTC相关性（如果可用）──
    btc_corr = 0.0
    btc_klines = extra.get("btc_klines") if isinstance(extra, dict) else None
    if btc_klines and bool(config.get("enable_btc_correlation_filter", False)):
        btc_h1 = _to_df(_get_klines({"1H": btc_klines} if isinstance(btc_klines, list) else btc_klines, "1H"))
        if len(btc_h1) >= 24:
            btc_corr = _calc_correlation(h1["c"].tail(24).pct_change().dropna(),
                                         btc_h1["c"].tail(24).pct_change().dropna())

    # ── 3m 回调企稳（含时效性）──
    # v4.0: 买卖价差过滤
    if bool(config.get('enable_bidask_spread_filter', True)):
        ob = (extra or {}).get('order_book')
        if ob and isinstance(ob, dict):
            bid = _safe_float(ob.get('bid_px') or (ob.get('bids', [[0]])[0][0] if ob.get('bids') else 0), 0)
            ask = _safe_float(ob.get('ask_px') or (ob.get('asks', [[0]])[0][0] if ob.get('asks') else 0), 0)
            if bid > 0 and ask > 0 and (ask / bid - 1) * 100 > _cfg_float(config, 'max_bidask_spread_pct', 0.25):
                return {"valid": False, "symbol": inst_id, "reason": "买卖价差过大，流动性不足"}

    micro_confirm = _micro_pullback_continuation(m3, trend_alignment, config)
    if bool(config.get("require_m3_pullback_confirmation", True)) and not micro_confirm["confirmed"]:
        return {"valid": False, "symbol": inst_id,
                "reason": f"3分钟回调续势未确认: {micro_confirm['reason']}"}

    if funding_risk:
        return {"valid": False, "symbol": inst_id,
                "reason": "接近资金费率结算时间，回避开仓"}

    rsi_1h = _rsi(h1["c"], 14)
    adx_1h = _adx_like(h1, 14)
    adx_4h = _adx_like(h4, 14) if len(h4) >= 25 else adx_1h * 0.7
    atr_pct = _atr_pct(h1, 14)
    realized_vol = _realized_vol_pct(h1["c"], 24)
    volume_impulse = _volume_impulse(h1, 24)

    # v2: 突破压强跟随趋势方向
    breakout_bps = _breakout_pressure_bps(h1, 24, trend_alignment)

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
        # v2 新增因子
        "trend_freshness": _clamp(1.0 - h1_trend_age / max(_cfg_float(config, "max_h1_trend_age", 12.0), 1.0), -1.0, 1.0),
        "m3_timeliness": micro_confirm.get("timeliness_score", 0.0),
    }

    meta_feedback = extra.get("strategy_feedback", {}) if isinstance(extra.get("strategy_feedback"), dict) else {}
    regime = _detect_regime(trend_alignment, adx_1h, realized_vol, h1_mom)

    return {
        "valid": True, "reason": "", "symbol": inst_id,
        "last_price": price, "volume_24h": volume_24h, "price_change_24h": change_24h,
        "funding_rate_pct": funding_rate, "open_interest_change_pct": oi_change,
        "atr_pct": atr_pct, "realized_vol_pct": realized_vol,
        "rsi_1h": rsi_1h, "adx_1h": adx_1h, "adx_4h": adx_4h,
        "m15_mom_bps": m15_mom,
        "m3_pullback_confirmed": micro_confirm["confirmed"],
        "m3_structure_state": micro_confirm["state"],
        "m3_pullback_reason": micro_confirm["reason"],
        "m3_pullback_pct": micro_confirm["pullback_pct"],
        "m3_impulse_pct": micro_confirm["impulse_pct"],
        "m3_staleness_bars": micro_confirm.get("staleness_bars", 0),
        "h1_trend_age": h1_trend_age,
        "h1_mom_bps": h1_mom, "h4_mom_bps": h4_mom, "d1_mom_bps": d1_mom,
        "breakout_bps": breakout_bps, "volume_impulse": volume_impulse,
        "funding_risk": funding_risk, "btc_correlation": btc_corr,
        "regime": regime, "meta_feedback": meta_feedback, "factors": factors,
    }


# ══════════════════════════════════════════════
# 评分（含时效性惩罚）
# ══════════════════════════════════════════════

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
        -2.0, 2.0,
    )
    long_adv = _clamp(
        meta_weights["trend"] * factors["trend_alignment"]
        + meta_weights["momentum"] * factors["momentum_1h"]
        + 0.20 * factors["breakout"]
        + 0.12 * factors["cnn_multi_tf"]
        + 0.10 * factors["oi_confirmation"]
        + 0.08 * factors["on_chain_accumulation"]
        + 0.06 * factors["llm_sentiment"]
        + 0.04 * factors["trend_freshness"]     # v2
        + 0.03 * factors["m3_timeliness"]       # v2
        - 0.10 * max(factors["risk_vol"], 0.0),
        -3.0, 3.0,
    )
    short_adv = _clamp(
        meta_weights["trend"] * (-factors["trend_alignment"])
        + meta_weights["momentum"] * (-factors["momentum_1h"])
        + 0.20 * (-factors["breakout"])
        + 0.12 * (-factors["cnn_multi_tf"])
        + 0.10 * (-factors["oi_confirmation"])
        + 0.06 * (-factors["llm_sentiment"])
        + 0.04 * (-factors["trend_freshness"])  # v2: 空头时反向
        + 0.05 * max(factors["risk_vol"], 0.0),
        -3.0, 3.0,
    )
    wait_adv = _clamp(
        0.30 * (factors["risk_vol"] + factors["risk_atr"])
        - 0.18 * abs(factors["trend_alignment"])
        + 0.08 * max(-factors["trend_freshness"], 0.0)  # v2: 趋势过老 → 更倾向 WAIT
        + 0.06 * max(-factors["m3_timeliness"], 0.0),   # v2: 3m回调过旧 → 更倾向 WAIT
        -2.5, 2.5,
    )

    mean_adv = (long_adv + short_adv + wait_adv) / 3.0
    q_long_online  = state_value + (long_adv - mean_adv)
    q_short_online = state_value + (short_adv - mean_adv)
    q_wait_online  = state_value + (wait_adv - mean_adv)

    target_scale = _cfg_float(config, "double_q_blend", 0.36)
    q_long_target  = _clamp(0.62 * factors["momentum_4h"] + 0.38 * factors["trend_alignment"], -2.8, 2.8)
    q_short_target = _clamp(-0.62 * factors["momentum_4h"] - 0.38 * factors["trend_alignment"], -2.8, 2.8)
    q_wait_target  = _clamp(0.50 * (factors["risk_vol"] + factors["risk_atr"]) - 0.12 * abs(factors["momentum_4h"]), -2.8, 2.8)
    q_long  = (1.0 - target_scale) * q_long_online  + target_scale * q_long_target
    q_short = (1.0 - target_scale) * q_short_online + target_scale * q_short_target
    q_wait  = (1.0 - target_scale) * q_wait_online  + target_scale * q_wait_target

    adv_long  = q_long  - state_value
    adv_short = q_short - state_value
    adv_wait  = q_wait  - state_value

    probs = _softmax([q_long, q_short, q_wait], _cfg_float(config, "q_temperature", 0.85))
    entropy = _normalized_entropy(probs)
    sac_alpha = _cfg_float(config, "entropy_alpha", 0.18)
    objectives = {
        "BUY":  q_long  + sac_alpha * entropy,
        "SELL": q_short + sac_alpha * entropy,
        "WAIT": q_wait  + sac_alpha * entropy,
    }
    direction = max(objectives, key=objectives.get)
    best = float(objectives[direction])
    second = sorted(objectives.values(), reverse=True)[1]
    confidence = _clamp(best - second, 0.0, 3.0)

    if direction == "SELL" and not bool(config.get("allow_short", True)):
        direction = "WAIT"

    startup_gate = _hourly_startup_gate(snap, direction, config)
    risk_penalty = _cfg_float(config, "risk_penalty_strength", 0.70) * max(0.0, factors["risk_vol"] + factors["risk_atr"] * 0.8)
    raw_edge = best - objectives["WAIT"] if direction in {"BUY", "SELL"} else 0.0
    edge = _clamp(raw_edge * (0.70 + 0.30 * confidence) - risk_penalty * 0.20, -2.8, 2.8)

    # v2: 时效性惩罚写入 score
    base_score = _edge_to_score(edge, confidence, startup_gate, snap)
    age_penalty = _age_penalty(snap, config, direction)
    score = _clamp(base_score - age_penalty, 0.0, 100.0)

    passed = (
        direction in {"BUY", "SELL"}
        and startup_gate
        and score >= _cfg_float(config, "min_score", 74.0)
        and abs(edge) >= _cfg_float(config, "min_abs_edge", 0.22)
        and snap["atr_pct"] <= _cfg_float(config, "max_atr_pct", 8.2)
    )

    # 时效性状态文字
    age = snap.get("h1_trend_age", 0)
    max_age = int(_cfg_float(config, "max_h1_trend_age", 12.0))
    stale = snap.get("m3_staleness_bars", 0)
    max_stale = int(_cfg_float(config, "max_m3_staleness_bars", 15))
    freshness_label = (
        f"1H趋势{age}根({'⚠过老' if age > max_age else '✓'}), 3m回调{stale}根前({'⚠过旧' if stale > max_stale else '✓'})"
    )

    category = "DRL小时趋势多头启动" if direction == "BUY" else "DRL小时趋势空头启动" if direction == "SELL" else "DRL小时趋势观察"
    signals = [
        f"{category} 评分 {score:.1f} (惩罚{age_penalty:.1f})",
        f"Q(L/S/W)=({q_long:+.2f},{q_short:+.2f},{q_wait:+.2f}) | A2C优势({adv_long:+.2f},{adv_short:+.2f},{adv_wait:+.2f})",
        f"SAC熵={entropy:.2f} 置信度={confidence:.2f} 边际={edge:+.3f}",
        f"1H动量={snap['h1_mom_bps']:+.1f}bps 4H动量={snap['h4_mom_bps']:+.1f}bps 突破压强={snap['breakout_bps']:+.1f}bps",
        f"量能脉冲={snap['volume_impulse']:.2f}x ADX1H={snap['adx_1h']:.1f} ATR%={snap['atr_pct']:.2f} Regime={snap['regime']}",
        f"时效性: {freshness_label}",
    ]
    ranking_factors = {
        "trend": _clamp(50 + factors["trend_alignment"] * 24 + factors["momentum_4h"] * 8, 0, 100),
        "trigger": _clamp(50 + factors["breakout"] * 18 + factors["momentum_1h"] * 18 + confidence * 10, 0, 100),
        "volume": _clamp(50 + factors["volume_impulse"] * 26 + factors["liquidity"] * 12, 0, 100),
        "location": _clamp(55 + (50.0 - abs(snap["rsi_1h"] - 55.0)) * 0.6, 10, 95),
        "freshness": _clamp(
            50 + snap["m15_mom_bps"] / 6.0 + snap["h1_mom_bps"] / 10.0
            - max(age - max_age, 0) * 2.5        # v2: 趋势过老降低 freshness
            - max(stale - max_stale, 0) * 1.5,   # v2: 回调过旧降低 freshness
            0, 95,
        ),
        "risk": _clamp(84 - max(factors["risk_vol"], 0.0) * 18 - max(factors["risk_atr"], 0.0) * 15, 10, 95),
    }
    result = {
        "symbol": snap["symbol"], "passed": passed,
        "score": round(float(score), 2), "direction": direction if direction in {"BUY", "SELL"} else "WAIT",
        "signals": signals, "category": category, "strategy_category": category,
        "last_price": snap["last_price"], "volume_24h": snap["volume_24h"],
        "price_change_24h": snap["price_change_24h"], "ranking_factors": ranking_factors,
        "metrics": {
            "alpha_edge": round(float(edge), 6), "raw_edge": round(float(raw_edge), 6),
            "confidence": round(float(confidence), 6), "entropy": round(float(entropy), 6),
            "q_long": round(float(q_long), 6), "q_short": round(float(q_short), 6), "q_wait": round(float(q_wait), 6),
            "state_value": round(float(state_value), 6),
            "adv_long": round(float(adv_long), 6), "adv_short": round(float(adv_short), 6),
            "atr_pct": round(float(snap["atr_pct"]), 6), "realized_vol_pct": round(float(snap["realized_vol_pct"]), 6),
            "regime": snap["regime"], "h1_trend_age": age, "m3_staleness_bars": stale,
            "age_penalty": round(float(age_penalty), 2),
        },
        "factor_scores": {k: round(float(v), 6) for k, v in factors.items()},
        "details": {
            "机会类型": category, "评估": " | ".join(signals),
            "综合边际": f"{edge:+.3f}", "置信度": f"{confidence:.2f}", "熵": f"{entropy:.2f}",
            "1H动量bps": f"{snap['h1_mom_bps']:+.1f}", "突破压强bps": f"{snap['breakout_bps']:+.1f}",
            "ADX1H": f"{snap['adx_1h']:.1f}", "ATR%": f"{snap['atr_pct']:.2f}",
            "资金费率%": f"{snap['funding_rate_pct']:+.4f}",
            "OI变化%": f"{snap['open_interest_change_pct']:+.2f}",
            "启动门槛通过": "是" if startup_gate else "否",
            "3分钟回调确认": "是" if snap.get("m3_pullback_confirmed") else "否",
            "3分钟结构": str(snap.get("m3_structure_state", "-")),
            "3分钟回调幅度%": f"{float(snap.get('m3_pullback_pct', 0.0)):.2f}",
            "3分钟原趋势脉冲%": f"{float(snap.get('m3_impulse_pct', 0.0)):.2f}",
            "3分钟回调时效": f"{stale}根前({'超时' if stale > max_stale else '新鲜'})",
            "1H趋势延续根数": f"{age}根({'过老' if age > max_age else '正常'})",
            "时效性惩罚": f"-{age_penalty:.1f}分",
        },
    }
    if build_opportunity_profile:
        try: result.update(build_opportunity_profile(score, result["direction"], snap["volume_24h"], ranking_factors, signals))
        except Exception: pass
    return result


def _age_penalty(snap: Dict[str, Any], config: Dict[str, Any], direction: str) -> float:
    """
    综合时效性惩罚分。
    1. 1H 趋势延续过长 → 越界越高
    2. 3m 回调点位过旧 → 越界越高
    两者都过期则叠加（但不超过 max_penalty = 15 分）。
    """
    if direction not in {"BUY", "SELL"}:
        return 0.0
    penalty = 0.0
    age = snap.get("h1_trend_age", 0)
    max_age = int(_cfg_float(config, "max_h1_trend_age", 12.0))
    age_pen_str = _cfg_float(config, "h1_trend_age_penalty", 8.0)
    if age > max_age:
        # 超出越多惩罚越重，但有上限
        overflow = age - max_age
        penalty += min(age_pen_str, age_pen_str * (1 - exp(-overflow / max(max_age * 0.5, 1.0))))

    stale = snap.get("m3_staleness_bars", 0)
    max_stale = int(_cfg_float(config, "max_m3_staleness_bars", 15))
    fresh_pen_str = _cfg_float(config, "m3_freshness_penalty", 6.0)
    if stale > max_stale:
        overflow = stale - max_stale
        penalty += min(fresh_pen_str, fresh_pen_str * (1 - exp(-overflow / max(max_stale * 0.5, 1.0))))

    return _clamp(penalty, 0.0, 15.0)


# ══════════════════════════════════════════════
# _hourly_startup_gate（保留 + breakout_bps 修正方向语义）
# ══════════════════════════════════════════════

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


# ══════════════════════════════════════════════
# 元学习权重 / Regime 检测
# ══════════════════════════════════════════════

def _meta_regime_weights(regime, feedback, config):
    base = {"trend": 0.42, "momentum": 0.38, "reversal": 0.20}
    if regime == "TREND": base = {"trend": 0.48, "momentum": 0.40, "reversal": 0.12}
    elif regime == "RANGE": base = {"trend": 0.30, "momentum": 0.24, "reversal": 0.46}
    elif regime == "VOLATILE": base = {"trend": 0.36, "momentum": 0.28, "reversal": 0.36}
    trend_wr = _safe_float(feedback.get("trend_win_rate"), 0.52)
    range_wr = _safe_float(feedback.get("range_win_rate"), 0.48)
    adapt = _cfg_float(config, "meta_adapt_strength", 0.40)
    delta = _clamp((trend_wr - range_wr) * 2.0, -0.30, 0.30) * adapt
    base["trend"] = max(0.05, base["trend"] + delta)
    base["momentum"] = max(0.05, base["momentum"] + delta * 0.7)
    base["reversal"] = max(0.05, base["reversal"] - delta * 1.7)
    total = base["trend"] + base["momentum"] + base["reversal"]
    return {k: v / total for k, v in base.items()}

def _detect_regime(trend_alignment, adx_1h, realized_vol, h1_mom_bps):
    if adx_1h >= 22.0 and abs(trend_alignment) >= 0.45 and abs(h1_mom_bps) >= 35.0: return "TREND"
    if realized_vol >= 8.0: return "VOLATILE"
    return "RANGE"


# ══════════════════════════════════════════════
# 底层工具
# ══════════════════════════════════════════════

def _edge_to_score(edge, confidence, startup_gate, snap):
    base = 34.0 + abs(edge) * 11.5 + confidence * 4.5
    quality = (
        max(0.0, snap["volume_impulse"] - 1.0) * 3.0
        + max(0.0, (snap["adx_1h"] - 16.0) / 10.0) * 2.8
        + max(0.0, 1.0 - snap["atr_pct"] / 10.0) * 2.5
    )
    gate_bonus = 1.5 if startup_gate else -9.0
    return _clamp(base + quality + gate_bonus, 0.0, 100.0)

def _result_sort_key(item):
    return (float(item.get("opportunity_score", item.get("score", 0.0)) or 0.0),
            float(item.get("score", 0.0) or 0.0),
            float(item.get("volume_24h", 0.0) or 0.0))

def _ema_trend_score(close, fast, slow):
    if len(close) < slow + 3: return 0.0
    ef = close.ewm(span=fast, adjust=False).mean()
    es = close.ewm(span=slow, adjust=False).mean()
    slope = float(ef.diff().tail(3).mean() or 0.0)
    spread = (float(ef.iloc[-1]) / max(float(es.iloc[-1]), 1e-9) - 1.0) * 100.0
    score = spread * 2.2 + slope / max(float(close.iloc[-1]), 1e-9) * 10_000 * 0.8
    return _clamp(score / 30.0, -2.0, 2.0)

def _get_klines(klines_map, bar):
    aliases = {"15m":["15m","15M"],"1H":["1H","1h","60m","60M"],
               "4H":["4H","4h","240m","240M"],"1D":["1D","1d","D","day"]}
    for key in aliases.get(bar, [bar, bar.lower(), bar.upper()]):
        if key in klines_map and klines_map.get(key): return klines_map.get(key)
    return []

def _atr_pct(df, period=14):
    if len(df) < period + 2: return 0.0
    pc = df["c"].shift(1)
    tr = pd.concat([(df["h"]-df["l"]).abs(),(df["h"]-pc).abs(),(df["l"]-pc).abs()],axis=1).max(axis=1)
    atr = float(tr.ewm(alpha=1/period, adjust=False).mean().iloc[-1] or 0.0)
    p = float(df["c"].iloc[-1] or 0.0); return atr / p * 100.0 if p > 0 else 0.0

def _adx_like(df, period=14):
    if len(df) < period + 5: return 15.0
    h, l, c = df["h"], df["l"], df["c"]
    um = h.diff(); dm = -l.diff()
    pdm = np.where((um>dm)&(um>0), um, 0.0); mdm = np.where((dm>um)&(dm>0), dm, 0.0)
    pc = c.shift(1)
    tr = pd.concat([(h-l).abs(),(h-pc).abs(),(l-pc).abs()],axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False).mean().replace(0, np.nan)
    pdi = 100*pd.Series(pdm,index=df.index).ewm(alpha=1/period,adjust=False).mean()/atr
    mdi = 100*pd.Series(mdm,index=df.index).ewm(alpha=1/period,adjust=False).mean()/atr
    dx = (pdi-mdi).abs()/(pdi+mdi).replace(0,np.nan)*100
    adx = dx.ewm(alpha=1/period, adjust=False).mean()
    v = float(adx.iloc[-1]) if np.isfinite(adx.iloc[-1]) else 15.0
    return _clamp(v, 5.0, 60.0)

def _realized_vol_pct(close, window=24):
    ret = close.pct_change().dropna().tail(window)
    if ret.empty: return 0.0
    return float(ret.std(ddof=0) * sqrt(max(len(ret), 1)) * 100.0)

def _volume_impulse(df, window=24):
    if len(df) < window + 3: return 1.0
    base = float(df["vol"].iloc[-(window+1):-1].median() or 0.0)
    latest = float(df["vol"].tail(3).mean() or 0.0)
    return latest / base if base > 0 else 1.0

def _breakout_pressure_bps(df, window=24, trend_hint=0.0):
    """v2: 跟随趋势方向选择突破方向，避免反向假突破干扰。"""
    if len(df) < window + 2: return 0.0
    high_ref = float(df["h"].iloc[-(window+1):-1].max() or 0.0)
    low_ref  = float(df["l"].iloc[-(window+1):-1].min() or 0.0)
    close = float(df["c"].iloc[-1] or 0.0)
    if close <= 0: return 0.0
    up   = (close / max(high_ref, 1e-9) - 1.0) * 10_000
    down = (close / max(low_ref,  1e-9) - 1.0) * 10_000
    # v2: 趋势方向优先；若方向不明则取绝对值更大者
    if trend_hint > 0.15:
        return _clamp(up, -600.0, 600.0)
    elif trend_hint < -0.15:
        return _clamp(down, -600.0, 600.0)
    else:
        return _clamp(up if abs(up) >= abs(down) else down, -600.0, 600.0)

def _conv_feature(series, kernel):
    vals = pd.to_numeric(series, errors="coerce").dropna().values.astype(float)
    k = np.array(kernel, dtype=float)
    if len(vals) < len(k) + 2: return 0.0
    norm = np.maximum(np.abs(vals[-len(k):]).mean(), 1e-9)
    return _clamp(float(np.dot(vals[-len(k):], k) / norm), -3.0, 3.0)

def _softmax(values, temperature):
    t = max(float(temperature), 1e-6)
    arr = np.array(values, dtype=float) / t; arr -= arr.max()
    exps = np.exp(arr); return (exps / float(np.sum(exps) or 1.0)).tolist()

def _normalized_entropy(probs):
    eps = 1e-12; n = max(len(probs), 1)
    ent = -sum(float(p) * np.log(float(p) + eps) for p in probs)
    return _clamp(ent / np.log(n + eps), 0.0, 1.0)

def _symbol_from_backtest_data(data, config):
    km = data.get("klines_map") or {}
    h1 = _to_df(_get_klines(km,"1H") or _get_klines(km,"15m") or data.get("klines") or [])
    lp = float(h1["c"].iloc[-1]) if not h1.empty else 0.0
    vol = float((h1["c"] * h1["vol"]).tail(48).sum()) if not h1.empty else 0.0
    extra = {"klines": km, "funding_rate": data.get("funding_rate", 0.0),
             "open_interest_change_pct": data.get("open_interest_change_pct", 0.0),
             "on_chain": data.get("on_chain", {}), "social": data.get("social", {}),
             "llm_factors": data.get("llm_factors", {}),
             "news_sentiment": data.get("news_sentiment", 0.0),
             "strategy_feedback": data.get("strategy_feedback", {})}
    return _MinimalSymbol(inst_id=str(config.get("inst_id", "BACKTEST") or "BACKTEST"),
        last_price=lp, volume_24h=vol,
        price_change_24h=_pct_change(h1["c"], min(len(h1)-1, 24))*100 if not h1.empty else 0.0,
        extra_data=extra)

class _MinimalSymbol:
    def __init__(self, inst_id, last_price, volume_24h, price_change_24h, extra_data):
        self.inst_id = inst_id; self.last_price = last_price
        self.volume_24h = volume_24h; self.price_change_24h = price_change_24h
        self.high_24h = 0.0; self.low_24h = 0.0; self.open_interest = 0.0
        self.extra_data = extra_data

def _failed_result(symbol, reason):
    return {"symbol": str(getattr(symbol,"inst_id","") or ""), "passed": False,
            "score": 0.0, "direction": "WAIT", "signals": [], "details": {"状态": reason}, "metrics": {}}

def _blend(values, weights):
    total = denom = 0.0
    for v, w in zip(values, weights):
        if np.isfinite(v): total += float(v)*float(w); denom += float(w)
    return total/denom if denom > 0 else 0.0

def _direction_sign(primary, fallback):
    if abs(float(primary or 0.0)) > 1e-9: return 1.0 if primary > 0 else -1.0
    if abs(float(fallback or 0.0)) > 1e-9: return 1.0 if fallback > 0 else -1.0
    return 1.0

# ── v2.1 新增工具 ──

def _check_funding_timing(config: Dict[str, Any]) -> bool:
    """
    检查当前时间是否接近资金费率结算时间。
    资金费率通常在 00:00, 08:00, 16:00 UTC 结算。
    在结算前 N 分钟内避免开仓以规避费率波动。
    """
    try:
        from datetime import datetime, timezone
        avoid_minutes = int(_cfg_float(config, "funding_avoid_minutes", 15.0))
        now_utc = datetime.now(timezone.utc)
        # OKX 标准资金费率结算时间: 00:00, 08:00, 16:00 UTC
        funding_hours = [0, 8, 16]
        for h in funding_hours:
            settlement = now_utc.replace(hour=h, minute=0, second=0, microsecond=0)
            delta_min = (settlement - now_utc).total_seconds() / 60.0
            if 0 <= delta_min <= avoid_minutes:
                return True  # 进入回避窗口
    except Exception:
        pass
    return False

def _calc_correlation(series_a: pd.Series, series_b: pd.Series) -> float:
    """
    计算两个序列的皮尔逊相关系数。
    用于检测与 BTC 的高度相关性。
    """
    try:
        n = min(len(series_a), len(series_b))
        if n < 8:
            return 0.0
        corr = series_a.tail(n).corr(series_b.tail(n))
        return float(corr) if np.isfinite(corr) else 0.0
    except Exception:
        return 0.0


STRATEGY_NAME = "DRL元学习小时趋势启动扫描策略"
STRATEGY_TYPE = "scan"
STRATEGY_CLASS = DRLMetaHourlyTrendStartScannerStrategy
BACKTEST_CLASS = DRLMetaHourlyTrendStartScannerStrategy
