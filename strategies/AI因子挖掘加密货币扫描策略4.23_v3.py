#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI因子挖掘加密货币交易对扫描策略 v3

v2 → v3 变更
─────────────────────────────────────────────────────
【逻辑错误修复】
1. _micro_pullback_continuation 窗口硬编码越界
   [-28:-12]/[-12:-stab_bars]/[-6:-1] 是固定偏移，m3 数据量刚好
   满足最小值 36 根时极易产生空切片或区间重叠。
   v3 改为基于数据长度的动态分段（与 DRL 策略保持一致）。

2. is_monotonic_increasing/decreasing 过严
   真实浮点数据几乎永远不满足严格单调，导致企稳永远失败。
   v3 改为"企稳段低点不创新低 + EMA 方向正确 + 大方向收盘向上"。

3. m3 score 单位混乱
   impulse_pct * 0.45 + pullback_pct * 0.75 两者都是 0.5~5% 的百分比，
   相加后结果 >> 2.0，clamp(-2,2) 导致分数几乎永远是满分。
   v3 先归一化到 [0,1] 再合成。

4. m3_pullback_min_pct 默认值不一致
   CONFIG_SCHEMA 写 0.50，但代码内 config.get(..., 0.35)。
   v3 统一为 0.50。

5. generate_signal 每次都实例化新的 self.__class__(cfg)
   → 外层实例的 self.config 不生效，会丢失外部 inject 的配置。
   v3 改为复用 self.scan_symbol(symbol)。

6. _apply_factor_correlation_control 传入了 weights 参数，但
   v2 已经将其改为使用 weights 权重选 anchor，与旧版行为不同。
   实际上正确，但函数签名在调用处不匹配（旧版不传 weights）。
   v3 统一调用。

【新增：趋势时效性双重过滤（用户需求）】
7. 1H 趋势延续时效检测：
   计算 EMA12 > EMA34 已持续了多少根 1H K 线。超过
   max_h1_trend_age 根（默认 12）时进行惩罚，避免"老趋势
   末端建仓"。

8. 3m 回调时效检测：
   检测回调最低点（多头）或最高点（空头）距当前 bar 的根数。
   超过 max_m3_staleness_bars（默认 15 根 = 45 分钟）时惩罚，
   避免"很久前企稳完的回调"触发建仓。

两项都在正常范围内时给予 +bonus_freshness_score（默认 +0.06）
的 edge 加分，鼓励"刚好在新鲜启动点"的信号。
─────────────────────────────────────────────────────
"""

from __future__ import annotations

from math import exp, log10, sqrt
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from strategies._shared.indicators import (
    _to_df, _aggregate_bars, _efficiency_ratio, _rsi_wilder as _rsi,
    _robust_zscore, _measure_trend_age, _micro_pullback_continuation,
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
    "min_score":                    {"type": "float", "default": 72.0,        "label": "最低扫描分数"},
    "backtest_min_score":           {"type": "float", "default": 68.0,        "label": "回测最低入场分数"},
    "min_volume_24h":               {"type": "float", "default": 5_000_000.0, "label": "最小24H成交额"},
    "top_n":                        {"type": "int",   "default": 20,          "label": "最多输出数量"},
    "allow_short":                  {"type": "bool",  "default": True,        "label": "允许空头"},
    "use_dynamic_ic_weights":       {"type": "bool",  "default": False,       "label": "[不稳定] 启用Rolling IC动态权重"},
    "ic_weight_blend":              {"type": "float", "default": 0.55,        "label": "IC权重混合比例"},
    "max_factor_weight":            {"type": "float", "default": 0.24,        "label": "单因子最大权重"},
    "min_abs_edge":                 {"type": "float", "default": 0.24,        "label": "最小截面优势"},
    "correlation_penalty":          {"type": "float", "default": 0.08,        "label": "同质因子惩罚"},
    "correlation_threshold":        {"type": "float", "default": 0.85,        "label": "因子正交化相关阈值"},
    "deduplicate_base_asset":       {"type": "bool",  "default": True,        "label": "同Base资产只保留最高分"},
    "enable_mfin_interactions":     {"type": "bool",  "default": False,       "label": "[不稳定] 启用非线性交互项"},
    "enable_early_trend_factors":   {"type": "bool",  "default": True,        "label": "启用小时级早启动/转折因子"},
    "early_trend_min_trigger":      {"type": "float", "default": 0.18,        "label": "早启动最低触发强度"},
    "enable_llm_factors":           {"type": "bool",  "default": False,       "label": "[不稳定] 启用LLM/新闻/社交因子(需外部数据)"},
    "enable_on_chain":              {"type": "bool",  "default": True,        "label": "启用链上因子"},
    "risk_penalty_strength":        {"type": "float", "default": 0.75,        "label": "风险惩罚强度"},
    "max_atr_pct":                  {"type": "float", "default": 8.0,         "label": "最大ATR%"},
    "position_size":                {"type": "float", "default": 0.10,        "label": "回测仓位比例"},
    "require_m3_pullback_confirmation": {"type": "bool", "default": True,     "label": "要求3分钟回调企稳续势"},
    "m3_pullback_min_pct":          {"type": "float", "default": 0.50,        "label": "3分钟最小回调幅度%"},
    "m3_pullback_max_pct":          {"type": "float", "default": 2.20,        "label": "3分钟最大回调幅度%"},
    "m3_stabilization_bars":        {"type": "int",   "default": 4,           "label": "3分钟企稳确认根数"},
    # v3 新增 — 时效性
    "max_h1_trend_age":             {"type": "int",   "default": 12,          "label": "1H趋势最大延续根数（超过则惩罚）"},
    "h1_trend_age_penalty":         {"type": "float", "default": 0.10,        "label": "趋势过老时 edge 惩罚量"},
    "max_m3_staleness_bars":        {"type": "int",   "default": 15,          "label": "3m回调最大时效根数（超过则惩罚）"},
    "m3_freshness_penalty":         {"type": "float", "default": 0.08,        "label": "3m回调过旧时 edge 惩罚量"},
    "bonus_freshness_score":        {"type": "float", "default": 0.06,        "label": "两项时效均通过时 edge 加分"},
}

_DEFAULT_CONFIG = {k: v["default"] for k, v in CONFIG_SCHEMA.items()}

_BASE_WEIGHTS = {
    "momentum": 0.13, "trend": 0.12, "reversal": 0.07, "low_vol": 0.08,
    "liquidity": 0.08, "volume_impulse": 0.07, "funding_contra": 0.06,
    "oi_confirmation": 0.06, "on_chain_accumulation": 0.11,
    "network_value": 0.07, "llm_sentiment": 0.09, "event_momentum": 0.04,
    "developer_activity": 0.02, "early_trend_trigger": 0.08,
    "ema_compression_breakout": 0.03, "rsi_midline_turn": 0.025,
    "macd_hist_turn": 0.025, "donchian_breakout": 0.025,
    "volume_price_confirm": 0.025,
}

_INTERACTION_TRANSFERS = {
    "ai_trend_interaction": (0.04, ("trend", "llm_sentiment")),
    "accumulation_breakout": (0.04, ("on_chain_accumulation", "volume_impulse")),
    "quality_momentum": (0.03, ("momentum", "low_vol")),
    "fresh_breakout_confirmation": (0.035, ("early_trend_trigger", "volume_impulse")),
}

VERSION = "3.0"


# ══════════════════════════════════════════════
class AIAutomatedAlphaCryptoScannerStrategy(BaseScannerStrategy if _HAS_SCANNER_BASE else object):
    required_bars = ["3m", "15m", "1H", "4H", "1D"]
    requires_derivative_metrics = True
    requires_on_chain_metrics = True
    name = "AI因子挖掘加密货币扫描策略"
    description = "Chain-of-Alpha v3: 稳定的技术/链上多因子截面扫描（IC/交互/LLM默认关闭）"
    strategy_type = "scan"

    def __init__(self, config=None):
        merged = {**_DEFAULT_CONFIG, **(config or {})}
        self.config = merged
        self.last_factor_weights = dict(_BASE_WEIGHTS)
        self.last_analysis: Dict[str, Any] = {}
        if _HAS_SCANNER_BASE and hasattr(super(), "__init__"):
            try:
                super().__init__(merged); self.config = merged
            except Exception:
                self.config = merged

    def _init_conditions(self):
        if ScanCondition is None or not hasattr(self, "add_condition"): return
        self.add_condition(ScanCondition(name="24H成交额", description="过滤成交额不足",
            field="volume_24h", operator=">=", value=self.config.get("min_volume_24h", 5_000_000.0)))

    def get_config_schema(self): return dict(CONFIG_SCHEMA)

    def scan_symbol(self, symbol):
        snap = _build_snapshot(symbol, self.config)
        if not snap["valid"]: return _failed_result(symbol, snap["reason"])
        weights = _resolve_weights([symbol], self.config)
        self.last_factor_weights = weights
        result = _score_single_snapshot(snap, weights, self.config)
        self.last_analysis[getattr(symbol, "inst_id", "")] = result
        return result

    def scan_all_symbols(self, symbols):
        min_volume = float(self.config.get("min_volume_24h", 5_000_000.0) or 0.0)
        snapshots = []; source_symbols = []
        for sym in symbols:
            if float(getattr(sym, "volume_24h", 0.0) or 0.0) < min_volume: continue
            snap = _build_snapshot(sym, self.config)
            if snap["valid"]: snapshots.append(snap); source_symbols.append(sym)
        if not snapshots: return {"type": "ai_factor_mining", "all_opportunities": []}

        weights = _resolve_weights(source_symbols, self.config)
        ff = pd.DataFrame([s["factors"] for s in snapshots], index=[s["symbol"] for s in snapshots])
        z = ff.apply(_robust_zscore, axis=0).fillna(0.0)
        z = _apply_factor_correlation_control(z, weights, self.config)

        edge_series = pd.Series(0.0, index=z.index)
        for name, weight in weights.items():
            if name in z.columns: edge_series = edge_series + z[name] * float(weight)

        results = []
        for snap in snapshots:
            edge = float(edge_series.get(snap["symbol"], 0.0))
            sf = z.loc[snap["symbol"]].to_dict() if snap["symbol"] in z.index else {}
            r = _build_result_from_edge(snap, edge, weights, self.config, len(snapshots), sf)
            if r.get("passed"): results.append(r)

        results = _mmr_select(results, int(self.config.get("top_n", 20) or 20),
                              bool(self.config.get("deduplicate_base_asset", True)))
        self.last_factor_weights = weights
        return {"type": "ai_factor_mining", "all_opportunities": results}

    def generate_signal(self, data, *args, **kwargs):
        if isinstance(data, (list, tuple)): data = {"klines_map": {"1H": list(data)}}
        if not isinstance(data, dict) or not (data.get("klines_map") or data.get("klines")): return None
        if not data.get("klines_map"): data = {**data, "klines_map": {"1H": data.get("klines") or []}}
        cfg = dict(self.config)
        cfg["min_score"] = _cfg_float(cfg, "backtest_min_score", _cfg_float(cfg, "min_score", 68.0))
        # v3 修复: 复用 self.scan_symbol 而非重新实例化，保留外部注入配置
        self.config = cfg
        result = self.scan_symbol(_symbol_from_backtest_data(data, cfg))
        self.last_analysis["BACKTEST"] = result
        if not result.get("passed"): return None
        d = str(result.get("direction", "WAIT")).upper()
        if d not in {"BUY", "SELL"}: return None
        return {"action": "BUY" if d == "BUY" else "SHORT",
                "position_size": float(cfg.get("position_size", 0.10) or 0.10),
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
    if len(h1) < 35: return {"valid": False, "symbol": inst_id, "reason": f"K线不足(1H={len(h1)})"}

    price = float(getattr(symbol, "last_price", 0.0) or h1["c"].iloc[-1])
    volume_24h = float(getattr(symbol, "volume_24h", 0.0) or (h1["c"] * h1["vol"]).tail(24).sum())
    price_change_24h = float(getattr(symbol, "price_change_24h", 0.0) or _pct_change(h1["c"], min(len(h1)-1, 24))*100.0)
    on_chain = extra.get("on_chain", {}) if isinstance(extra.get("on_chain", {}), dict) else {}
    llm = extra.get("llm_factors", {}) if isinstance(extra.get("llm_factors", {}), dict) else {}
    social = extra.get("social", {}) if isinstance(extra.get("social", {}), dict) else {}

    rsi_1h = _rsi(h1["c"], 14)
    atr_pct = _atr_pct(h4 if len(h4) >= 20 else h1, 14)
    realized_vol = _realized_vol_pct(h1["c"], 24)
    trend = _trend_quality(h1, h4, d1)

    # v3: 1H 趋势时效
    h1_trend_age = _measure_trend_age(h1["c"], fast=12, slow=34, direction=trend)

    # 3m 企稳（含时效性）
    micro_confirm = _micro_pullback_continuation(m3, trend, config)
    if bool(config.get("require_m3_pullback_confirmation", True)) and not micro_confirm["confirmed"]:
        return {"valid": False, "symbol": inst_id,
                "reason": f"3分钟回调续势未确认: {micro_confirm['reason']}"}

    momentum = _blend([
        _pct_change(h1["c"], 6) * 100.0,
        _pct_change(h4["c"], 6) * 100.0 if len(h4) > 7 else 0.0,
        _pct_change(d1["c"], 5) * 100.0 if len(d1) > 6 else 0.0,
    ], [0.45, 0.35, 0.20])
    reversal = _clamp((50.0 - rsi_1h) / 22.0, -1.5, 1.5) - _clamp(_pct_change(h1["c"], 3) * 18.0, -1.0, 1.0)
    liquidity = _clamp((log10(max(volume_24h, 1.0)) - 6.0) / 2.0, -1.0, 1.5)
    volume_impulse = _volume_impulse(h1)
    funding = _safe_float(extra.get("funding_rate"), 0.0) * 100.0
    oi_change = _safe_float(extra.get("open_interest_change_pct"), 0.0)

    on_chain_score = _on_chain_accumulation(on_chain) if bool(config.get("enable_on_chain", True)) else 0.0
    network_value = _network_value_score(on_chain) if bool(config.get("enable_on_chain", True)) else 0.0
    llm_sentiment = _llm_sentiment_score(extra, llm, social) if bool(config.get("enable_llm_factors", True)) else 0.0
    event_momentum = _event_score(llm, social) if bool(config.get("enable_llm_factors", True)) else 0.0
    developer_activity = _developer_activity_score(llm, social) if bool(config.get("enable_llm_factors", True)) else 0.0
    risk_score = _risk_warning_score(extra, llm, atr_pct, realized_vol)
    early = _early_trend_features(h1) if bool(config.get("enable_early_trend_factors", True)) else _empty_early_trend_features()

    factors = {
        "momentum": _clamp(momentum / 9.0, -2.0, 2.0),
        "trend": trend,
        "reversal": reversal,
        "low_vol": _clamp((5.5 - realized_vol) / 4.0, -1.5, 1.5),
        "liquidity": liquidity,
        "volume_impulse": _clamp(volume_impulse - 1.0, -1.0, 2.0),
        "funding_contra": _clamp(-funding / 0.08, -1.5, 1.5),
        "oi_confirmation": _clamp(oi_change / 8.0, -1.5, 1.5) * _direction_sign(momentum, trend),
        "on_chain_accumulation": on_chain_score,
        "network_value": network_value,
        "llm_sentiment": llm_sentiment,
        "event_momentum": event_momentum,
        "developer_activity": developer_activity,
        "early_trend_trigger": early["trigger"],
        "ema_compression_breakout": early["ema_compression_breakout"],
        "rsi_midline_turn": early["rsi_midline_turn"],
        "macd_hist_turn": early["macd_hist_turn"],
        "donchian_breakout": early["donchian_breakout"],
        "volume_price_confirm": early["volume_price_confirm"],
        "m3_pullback_score": micro_confirm["score"],
    }
    if bool(config.get("enable_mfin_interactions", True)):
        factors["ai_trend_interaction"] = _clamp(factors["trend"] * max(factors["llm_sentiment"], 0.0), -1.5, 1.5)
        factors["accumulation_breakout"] = _clamp(max(factors["on_chain_accumulation"], 0.0) * max(factors["volume_impulse"], 0.0), -1.5, 1.5)
        factors["quality_momentum"] = _clamp(factors["momentum"] * max(factors["low_vol"], 0.0), -1.5, 1.5)
        factors["fresh_breakout_confirmation"] = _clamp(
            factors["early_trend_trigger"] * max(abs(factors["volume_impulse"]), 0.0), -1.5, 1.5)

    return {
        "valid": True, "reason": "", "symbol": inst_id,
        "last_price": price, "volume_24h": volume_24h, "price_change_24h": price_change_24h,
        "atr_pct": atr_pct, "realized_vol": realized_vol, "rsi_1h": rsi_1h,
        "funding_rate_pct": funding, "open_interest_change_pct": oi_change,
        "risk_warning": risk_score,
        "h1_trend_age": h1_trend_age,
        "m3_pullback_confirmed": micro_confirm["confirmed"],
        "m3_structure_state": micro_confirm["state"],
        "m3_pullback_reason": micro_confirm["reason"],
        "m3_pullback_pct": micro_confirm["pullback_pct"],
        "m3_impulse_pct": micro_confirm["impulse_pct"],
        "m3_staleness_bars": micro_confirm.get("staleness_bars", 0),
        "early_trend": early, "factors": factors, "extra": extra,
    }


def _timeliness_edge_adjustment(snap: Dict[str, Any], config: Dict[str, Any], direction: str) -> float:
    """
    计算时效性对 edge 的净调整量（正=加分，负=惩罚）。
    两项都新鲜 → +bonus
    趋势过老 → -age_penalty（指数缓增，越界越多越重）
    3m 过旧 → -freshness_penalty
    """
    if direction not in {"BUY", "SELL"}: return 0.0

    age = snap.get("h1_trend_age", 0)
    max_age = int(_cfg_float(config, "max_h1_trend_age", 12.0))
    age_pen = _cfg_float(config, "h1_trend_age_penalty", 0.10)
    stale = snap.get("m3_staleness_bars", 0)
    max_stale = int(_cfg_float(config, "max_m3_staleness_bars", 15))
    fresh_pen = _cfg_float(config, "m3_freshness_penalty", 0.08)
    bonus = _cfg_float(config, "bonus_freshness_score", 0.06)

    adj = 0.0
    age_ok = age <= max_age
    stale_ok = stale <= max_stale

    if not age_ok:
        overflow = age - max_age
        adj -= age_pen * (1 - exp(-overflow / max(max_age * 0.5, 1.0)))
    if not stale_ok:
        overflow = stale - max_stale
        adj -= fresh_pen * (1 - exp(-overflow / max(max_stale * 0.5, 1.0)))
    if age_ok and stale_ok:
        adj += bonus

    return _clamp(adj, -0.20, 0.10)


# ══════════════════════════════════════════════
# 评分
# ══════════════════════════════════════════════

def _score_single_snapshot(snap, weights, config):
    sf = _single_asset_scoring_factors(snap, weights)
    edge = _weighted_edge(sf, weights)
    return _build_result_from_edge(snap, edge, weights, config, 1, sf)

def _single_asset_scoring_factors(snap, weights):
    scales = {
        "momentum":0.75,"trend":0.65,"reversal":0.85,"low_vol":0.75,"liquidity":0.80,
        "volume_impulse":0.65,"funding_contra":0.70,"oi_confirmation":0.70,
        "on_chain_accumulation":0.85,"network_value":0.80,"llm_sentiment":0.85,
        "event_momentum":0.85,"developer_activity":0.85,"ai_trend_interaction":0.65,
        "accumulation_breakout":0.65,"quality_momentum":0.65,"early_trend_trigger":0.75,
        "ema_compression_breakout":0.80,"rsi_midline_turn":0.85,"macd_hist_turn":0.85,
        "donchian_breakout":0.80,"volume_price_confirm":0.80,"fresh_breakout_confirmation":0.70,
    }
    raw = snap.get("factors", {}); scoring = {}
    for name in weights:
        if name not in raw: continue
        scale = max(scales.get(name, 0.80), 1e-9)
        scoring[name] = _clamp(float(raw.get(name, 0.0)) / scale, -3.0, 3.0)
    return scoring

def _weighted_edge(factors, weights):
    return float(sum(float(factors.get(n, 0.0)) * float(w) for n, w in weights.items()))

def _build_result_from_edge(snap, edge, weights, config, universe_size, scoring_factors=None):
    scoring_factors = scoring_factors or snap["factors"]
    risk_penalty = _cfg_float(config, "risk_penalty_strength", 0.75) * max(float(snap.get("risk_warning", 0.0)), 0.0)
    atr_penalty = max(float(snap.get("atr_pct", 0.0)) - _cfg_float(config, "max_atr_pct", 8.0), 0.0) * 0.08
    adjusted_edge = float(edge) - risk_penalty * np.sign(edge) - atr_penalty * np.sign(edge)
    early_trigger = float(snap["factors"].get("early_trend_trigger", 0.0) or 0.0)
    min_early = _cfg_float(config, "early_trend_min_trigger", 0.18)
    if abs(early_trigger) >= min_early and np.sign(early_trigger) == np.sign(adjusted_edge or early_trigger):
        adjusted_edge += np.sign(early_trigger) * min(abs(early_trigger) * 0.08, 0.16)

    allow_short = bool(config.get("allow_short", True))
    direction = "BUY" if adjusted_edge >= 0 else "SELL"
    if direction == "SELL" and not allow_short: direction = "WAIT"

    # v3: 时效性调整
    time_adj = _timeliness_edge_adjustment(snap, config, direction)
    adjusted_edge += time_adj

    score = _edge_to_score(abs(adjusted_edge), snap)
    passed = (
        direction in {"BUY", "SELL"}
        and score >= _cfg_float(config, "min_score", 72.0)
        and abs(adjusted_edge) >= _cfg_float(config, "min_abs_edge", 0.24)
    )

    # 时效性状态文字
    age = snap.get("h1_trend_age", 0)
    max_age = int(_cfg_float(config, "max_h1_trend_age", 12.0))
    stale = snap.get("m3_staleness_bars", 0)
    max_stale = int(_cfg_float(config, "max_m3_staleness_bars", 15))
    freshness_label = (
        f"1H趋势{age}根({'⚠过老' if age > max_age else '✓新鲜'}), "
        f"3m回调{stale}根前({'⚠过旧' if stale > max_stale else '✓新鲜'}), "
        f"时效调整{time_adj:+.3f}"
    )

    category = "AI因子多头机会" if direction=="BUY" else "AI因子空头机会" if direction=="SELL" else "AI因子观察"
    top_factors = _top_factor_reasons(scoring_factors, weights, direction)
    signals = [
        f"{category} 评分 {score:.1f}",
        f"综合Alpha {adjusted_edge:+.3f} / 原始 {edge:+.3f} / 风险惩罚 {risk_penalty:.2f}",
        f"趋势 {snap['factors'].get('trend',0):+.2f} / 动量 {snap['factors'].get('momentum',0):+.2f} / LLM {snap['factors'].get('llm_sentiment',0):+.2f}",
        f"早启 {snap['factors'].get('early_trend_trigger',0):+.2f} / EMA {snap['factors'].get('ema_compression_breakout',0):+.2f} / 突破 {snap['factors'].get('donchian_breakout',0):+.2f}",
        f"链上 {snap['factors'].get('on_chain_accumulation',0):+.2f} / 量能 {snap['factors'].get('volume_impulse',0):+.2f} / 低波 {snap['factors'].get('low_vol',0):+.2f}",
        f"时效: {freshness_label}",
        "；".join(top_factors),
    ]
    ranking_factors = {
        "trend": _clamp(50 + snap["factors"].get("trend", 0.0)*28, 0, 100),
        "trigger": _clamp(50 + abs(adjusted_edge)*45, 0, 100),
        "volume": _clamp(50 + snap["factors"].get("volume_impulse",0.0)*25 + snap["factors"].get("liquidity",0.0)*18, 0, 100),
        "location": _clamp(60 + snap["factors"].get("reversal",0.0)*12, 20, 95),
        "freshness": _clamp(
            55 + snap["factors"].get("event_momentum",0.0)*25 + snap["factors"].get("llm_sentiment",0.0)*12
            - max(age - max_age, 0) * 2.0 - max(stale - max_stale, 0) * 1.5,
            20, 96),
        "risk": _clamp(82 - snap.get("risk_warning",0.0)*30 - max(snap.get("atr_pct",0.0)-5.0,0.0)*4, 10, 95),
    }
    result = {
        "symbol": snap["symbol"], "passed": passed,
        "score": round(score, 2), "direction": direction,
        "signals": signals, "category": category, "strategy_category": category,
        "last_price": snap["last_price"], "volume_24h": snap["volume_24h"],
        "price_change_24h": snap["price_change_24h"], "ranking_factors": ranking_factors,
        "metrics": {
            "alpha_edge": round(adjusted_edge, 6), "raw_edge": round(edge, 6),
            "atr_pct": round(float(snap.get("atr_pct", 0.0)), 4),
            "realized_vol_pct": round(float(snap.get("realized_vol", 0.0)), 4),
            "rsi_1h": round(float(snap.get("rsi_1h", 0.0)), 4),
            "funding_rate_pct": round(float(snap.get("funding_rate_pct", 0.0)), 6),
            "open_interest_change_pct": round(float(snap.get("open_interest_change_pct", 0.0)), 4),
            "risk_warning": round(float(snap.get("risk_warning", 0.0)), 4),
            "early_trend_trigger": round(early_trigger, 6),
            "h1_trend_age": age, "m3_staleness_bars": stale,
            "timeliness_adj": round(time_adj, 4),
            "universe_size": universe_size,
        },
        "factor_scores": {k: round(float(v), 6) for k, v in snap["factors"].items()},
        "scoring_factor_scores": {k: round(float(v), 6) for k, v in scoring_factors.items()},
        "factor_weights": {k: round(float(v), 6) for k, v in weights.items()},
        "details": {
            "机会类型": category, "评估": " | ".join(signals),
            "综合Alpha": f"{adjusted_edge:+.3f}", "原始Alpha": f"{edge:+.3f}",
            "风险惩罚": f"{risk_penalty:.2f}", "时效调整": f"{time_adj:+.4f}",
            "ATR%": f"{float(snap.get('atr_pct',0.0)):.2f}",
            "1H RSI": f"{float(snap.get('rsi_1h',0.0)):.1f}",
            "资金费率%": f"{float(snap.get('funding_rate_pct',0.0)):+.4f}",
            "OI变化%": f"{float(snap.get('open_interest_change_pct',0.0)):+.2f}",
            "早启动触发": f"{early_trigger:+.2f}",
            "EMA收敛突破": f"{float(snap['factors'].get('ema_compression_breakout',0.0)):+.2f}",
            "RSI中轴转折": f"{float(snap['factors'].get('rsi_midline_turn',0.0)):+.2f}",
            "MACD柱体拐头": f"{float(snap['factors'].get('macd_hist_turn',0.0)):+.2f}",
            "唐奇安突破": f"{float(snap['factors'].get('donchian_breakout',0.0)):+.2f}",
            "量价确认": f"{float(snap['factors'].get('volume_price_confirm',0.0)):+.2f}",
            "3分钟回调确认": "是" if snap.get("m3_pullback_confirmed") else "否",
            "3分钟结构": str(snap.get("m3_structure_state", "-")),
            "3分钟回调幅度%": f"{float(snap.get('m3_pullback_pct',0.0)):.2f}",
            "3分钟原趋势脉冲%": f"{float(snap.get('m3_impulse_pct',0.0)):.2f}",
            "3分钟回调时效": f"{stale}根前({'超时' if stale > max_stale else '新鲜'})",
            "1H趋势延续根数": f"{age}根({'过老' if age > max_age else '正常'})",
            "时效状态": freshness_label,
        },
    }
    if build_opportunity_profile:
        try: result.update(build_opportunity_profile(score, direction, snap["volume_24h"], ranking_factors, signals))
        except Exception: pass
    return result


# ══════════════════════════════════════════════
# 权重管理
# ══════════════════════════════════════════════

def _resolve_weights(symbols, config):
    weights = dict(_BASE_WEIGHTS)
    if not bool(config.get("enable_llm_factors", True)):
        for n in ("llm_sentiment","event_momentum","developer_activity"): weights[n] = 0.0
    if not bool(config.get("enable_on_chain", True)):
        for n in ("on_chain_accumulation","network_value"): weights[n] = 0.0
    if not bool(config.get("enable_early_trend_factors", True)):
        for n in ("early_trend_trigger","ema_compression_breakout","rsi_midline_turn",
                  "macd_hist_turn","donchian_breakout","volume_price_confirm","fresh_breakout_confirmation"):
            weights[n] = 0.0
    if bool(config.get("enable_mfin_interactions", True)):
        weights = _add_interaction_weights(weights)
    if bool(config.get("use_dynamic_ic_weights", True)):
        ic = _collect_ic(symbols)
        if ic:
            dynamic = {n: max(_clamp(float(ic.get(n,0.0)),-0.18,0.18),0.0) for n in weights}
            if sum(dynamic.values()) > 0:
                dynamic = _normalize_weights(dynamic, _cfg_float(config,"max_factor_weight",0.24))
                blend = _clamp(_cfg_float(config,"ic_weight_blend",0.55),0.0,1.0)
                weights = {n: weights.get(n,0.0)*(1-blend)+dynamic.get(n,0.0)*blend for n in weights}
    return _normalize_weights(weights, _cfg_float(config,"max_factor_weight",0.24))

def _add_interaction_weights(weights):
    adjusted = dict(weights)
    for interaction, (amount, parents) in _INTERACTION_TRANSFERS.items():
        if any(float(adjusted.get(p,0.0)) <= 0.0 for p in parents):
            adjusted[interaction] = 0.0; continue
        parent_share = amount / max(len(parents), 1); funded = 0.0
        for p in parents:
            available = max(float(adjusted.get(p,0.0)),0.0)
            take = min(parent_share, available)
            adjusted[p] = available - take; funded += take
        adjusted[interaction] = adjusted.get(interaction,0.0) + funded
    return adjusted

def _collect_ic(symbols):
    values = {}
    for sym in symbols:
        extra = getattr(sym,"extra_data",{}) or {}
        for key in ("factor_ic","rolling_ic"):
            data = extra.get(key)
            if isinstance(data, dict):
                for name, value in data.items():
                    f = _safe_float(value, np.nan)
                    if np.isfinite(f): values.setdefault(str(name),[]).append(float(f))
    return {name: float(np.nanmean(vals)) for name, vals in values.items() if vals}

def _normalize_weights(weights, max_weight):
    cleaned = {n: max(float(v),0.0) for n,v in weights.items()}
    total = sum(cleaned.values())
    if total <= 0: cleaned = dict(_BASE_WEIGHTS); total = sum(cleaned.values())
    normalized = {n: v/total for n,v in cleaned.items()}
    capped = {n: min(v, max_weight) for n,v in normalized.items()}
    total = sum(capped.values())
    return {n: (v/total if total > 0 else 0.0) for n,v in capped.items()}

def _apply_factor_correlation_control(z, weights, config):
    if z.empty or z.shape[1] < 3: return z
    penalty_strength = _clamp(_cfg_float(config,"correlation_penalty",0.18),0.0,0.8)
    threshold = _clamp(_cfg_float(config,"correlation_threshold",0.78),0.30,0.98)
    if penalty_strength <= 0: return z
    corr = z.corr().abs().fillna(0.0); adjusted = z.copy()
    pairs = []
    cols = list(adjusted.columns)
    for i, left in enumerate(cols):
        for right in cols[i+1:]:
            cv = float(corr.loc[left, right])
            if cv >= threshold: pairs.append((cv, left, right))
    for cv, left, right in sorted(pairs, reverse=True):
        lw = abs(float(weights.get(left,0.0))); rw = abs(float(weights.get(right,0.0)))
        anchor, target = (left, right) if lw >= rw else (right, left)
        as_ = adjusted[anchor].replace([np.inf,-np.inf],np.nan).fillna(0.0)
        ts_ = adjusted[target].replace([np.inf,-np.inf],np.nan).fillna(0.0)
        ac = as_ - float(as_.mean()); tc = ts_ - float(ts_.mean())
        var = float(ac.var(ddof=0) or 0.0)
        if var <= 1e-12: continue
        beta = float((tc * ac).mean() / var); residual = tc - beta * ac
        if float(residual.std(ddof=0) or 0.0) <= 1e-10: residual = pd.Series(0.0, index=ts_.index)
        extra_str = (float(cv) - threshold) / max(1.0 - threshold, 1e-9)
        eff_str = _clamp(max(penalty_strength, extra_str), 0.0, 1.0)
        new_t = ts_ * (1 - eff_str) + residual * eff_str
        if float(new_t.std(ddof=0) or 0.0) <= 1e-10: new_t = pd.Series(0.0, index=ts_.index)
        adjusted[target] = new_t
    return adjusted

def _mmr_select(results, top_n, deduplicate_base_asset=True):
    ordered = sorted(results, key=lambda x: float(x.get("opportunity_score",x.get("score",0.0)) or 0.0), reverse=True)
    selected = []; seen = set()
    for item in ordered:
        base = _base_asset_key(str(item.get("symbol","")))
        if deduplicate_base_asset and base in seen: continue
        selected.append(item); seen.add(base)
        if len(selected) >= top_n: break
    return selected

def _base_asset_key(symbol):
    return str(symbol or "").replace("/","-").replace("_","-").upper().split("-")[0]


# ══════════════════════════════════════════════
# 工具函数（与 v2 保持一致）
# ══════════════════════════════════════════════

def _failed_result(symbol, reason):
    return {"symbol": str(getattr(symbol,"inst_id","") or ""), "passed": False,
            "score": 0.0, "direction": "WAIT", "signals": [],
            "details": {"状态": reason}, "metrics": {}}

def _edge_to_score(abs_edge, snap):
    quality = (min(max(snap["factors"].get("liquidity",0.0),0.0),1.2)*3.0
             + min(max(snap["factors"].get("volume_impulse",0.0),0.0),1.2)*2.5
             + min(max(snap["factors"].get("low_vol",0.0),0.0),1.2)*2.0)
    return _clamp(50.0 + abs_edge*58.0 + quality, 0.0, 100.0)

def _top_factor_reasons(factors, weights, direction):
    sign = 1.0 if direction=="BUY" else -1.0
    contribs = [(n, float(s)*sign*float(weights.get(n,0.0))) for n,s in factors.items()]
    names = {"momentum":"动量","trend":"趋势质量","reversal":"反转位置","low_vol":"低波质量",
             "liquidity":"流动性","volume_impulse":"量能脉冲","funding_contra":"资金费率反身性",
             "oi_confirmation":"OI确认","on_chain_accumulation":"链上积累","network_value":"网络估值",
             "llm_sentiment":"LLM/新闻情绪","event_momentum":"事件动量","developer_activity":"开发活跃",
             "ai_trend_interaction":"AI情绪趋势共振","accumulation_breakout":"链上放量共振",
             "quality_momentum":"低波动量共振","early_trend_trigger":"小时早启触发",
             "ema_compression_breakout":"EMA收敛突破","rsi_midline_turn":"RSI中轴转折",
             "macd_hist_turn":"MACD柱体拐头","donchian_breakout":"唐奇安突破",
             "volume_price_confirm":"量价确认","fresh_breakout_confirmation":"早启放量确认"}
    return [f"{names.get(n,n)}({v:+.3f})" for n,v in sorted(contribs,key=lambda x:x[1],reverse=True)[:5]]

def _get_klines(klines_map, bar):
    aliases = {"1H":["1H","1h","60m","60M"],"4H":["4H","4h","240m","240M"],
               "1D":["1D","1d","D","day"],"15m":["15m","15M"]}
    for key in aliases.get(bar,[bar,bar.lower(),bar.upper()]):
        if key in klines_map and klines_map.get(key): return klines_map.get(key)
    return []

def _atr_pct(df, period=14):
    if len(df) < period+2: return 0.0
    pc = df["c"].shift(1)
    tr = pd.concat([(df["h"]-df["l"]).abs(),(df["h"]-pc).abs(),(df["l"]-pc).abs()],axis=1).max(axis=1)
    atr = float(tr.ewm(alpha=1/period,adjust=False).mean().iloc[-1] or 0.0)
    p = float(df["c"].iloc[-1] or 0.0); return atr/p*100.0 if p > 0 else 0.0

def _realized_vol_pct(close, window=24):
    ret = close.pct_change().dropna().tail(window)
    if ret.empty: return 0.0
    return float(ret.std(ddof=0)*sqrt(max(len(ret),1))*100.0)

def _trend_quality(h1, h4, d1):
    score = 0.0
    for df, fast, slow, w in [(h1,12,34,0.45),(h4,8,21,0.35),(d1,5,13,0.20)]:
        if len(df) < slow+2: continue
        ef = df["c"].ewm(span=fast,adjust=False).mean()
        es = df["c"].ewm(span=slow,adjust=False).mean()
        slope = float(ef.diff().tail(3).mean() or 0.0)
        d = 1.0 if ef.iloc[-1]>es.iloc[-1] and slope>0 else -1.0 if ef.iloc[-1]<es.iloc[-1] and slope<0 else 0.0
        score += d*w*(0.65+0.35*_efficiency_ratio(df["c"],min(20,len(df)-1)))
    return _clamp(score,-1.5,1.5)

def _volume_impulse(df, window=24):
    if len(df) < window+2: return 1.0
    base = float(df["vol"].iloc[-(window+1):-1].median() or 0.0)
    latest = float(df["vol"].tail(3).mean() or 0.0)
    return latest/base if base>0 else 1.0

def _on_chain_accumulation(oc):
    wf=_safe_float(oc.get("whale_flow"),0.0); en=_safe_float(oc.get("exchange_netflow"),0.0); sf=_safe_float(oc.get("stablecoin_flow"),0.0)
    return _clamp(_clamp(wf,-2,2)-_clamp(en,-2,2)*0.65+_clamp(sf,-2,2)*0.35,-2,2)

def _network_value_score(oc):
    active=_safe_float(oc.get("active_addresses_z"),np.nan)
    if not np.isfinite(active):
        active=_safe_float(oc.get("active_addresses"),0.0); active=_clamp((log10(max(active,1.0))-4.0)/2.0,-1.5,1.5)
    nvt=_safe_float(oc.get("nvt_signal_z"),np.nan)
    if not np.isfinite(nvt):
        nvt=_safe_float(oc.get("nvt_signal"),0.0); nvt=_clamp((60.0-nvt)/45.0,-1.5,1.5) if nvt>0 else 0.0
    mvrv=_safe_float(oc.get("mvrv_z"),0.0)
    return _clamp(active*0.45+nvt*0.35+_clamp(-mvrv/2,-1.5,1.5)*0.20,-2,2)

def _llm_sentiment_score(extra, llm, social):
    vals=[_safe_float(llm.get("sentiment"),np.nan),_safe_float(llm.get("narrative_strength"),np.nan),
          _safe_float(social.get("sentiment"),np.nan),_safe_float(extra.get("news_sentiment"),np.nan)]
    clean=[v for v in vals if np.isfinite(v)]; return 0.0 if not clean else _clamp(float(np.mean(clean)),-2,2)

def _event_score(llm, social):
    gs=_safe_float(social.get("galaxy_score"),np.nan); gc=gs/50.0-1.0 if np.isfinite(gs) else np.nan
    vals=[_safe_float(llm.get("event_score"),np.nan),_safe_float(llm.get("announcement_score"),np.nan),
          _safe_float(social.get("social_volume_z"),np.nan),gc]
    clean=[v for v in vals if np.isfinite(v)]; return 0.0 if not clean else _clamp(float(np.mean(clean)),-2,2)

def _developer_activity_score(llm, social):
    vals=[_safe_float(llm.get("github_activity"),np.nan),_safe_float(llm.get("dev_momentum"),np.nan),
          _safe_float(social.get("developer_activity"),np.nan)]
    clean=[v for v in vals if np.isfinite(v)]; return 0.0 if not clean else _clamp(float(np.mean(clean)),-2,2)

def _risk_warning_score(extra, llm, atr_pct, realized_vol):
    warnings=[_safe_float(llm.get("risk_warning"),0.0),_safe_float(extra.get("risk_warning"),0.0),
              max(atr_pct-7.0,0.0)/4.0,max(realized_vol-8.0,0.0)/5.0]
    return _clamp(float(np.nanmean(warnings)),0.0,2.5)

def _empty_early_trend_features():
    return {"trigger":0.0,"early_trend_trigger":0.0,"ema_compression_breakout":0.0,
            "rsi_midline_turn":0.0,"macd_hist_turn":0.0,"donchian_breakout":0.0,"volume_price_confirm":0.0}

def _early_trend_features(h1):
    if h1 is None or len(h1) < 58: return _empty_early_trend_features()
    close=h1["c"].astype(float); high=h1["h"].astype(float)
    low=h1["l"].astype(float); vol=h1["vol"].astype(float)
    price=float(close.iloc[-1])
    if price<=0: return _empty_early_trend_features()
    ef8=close.ewm(span=8,adjust=False).mean(); ef21=close.ewm(span=21,adjust=False).mean(); ef55=close.ewm(span=55,adjust=False).mean()
    prev_spread=float((ef21.iloc[-7]-ef55.iloc[-7])/max(abs(ef55.iloc[-7]),1e-9)*100.0)
    cur_fs=float((ef8.iloc[-1]-ef21.iloc[-1])/price*100.0)
    cur_ss=float((ef21.iloc[-1]-ef55.iloc[-1])/price*100.0)
    spread_delta=cur_ss-prev_spread; compression=1.0-_clamp(abs(prev_spread)/2.2,0.0,1.0)
    ema_c=_clamp((cur_fs*0.9+spread_delta*0.7)*(0.65+compression*0.55),-2.0,2.0)
    ph=float(high.iloc[-21:-1].max()); pl=float(low.iloc[-21:-1].min())
    ub=(price/ph-1.0)*100.0 if ph>0 else 0.0; db=(pl/price-1.0)*100.0 if pl>0 else 0.0
    donchian=_clamp(ub/0.9,0.0,2.0)-_clamp(db/0.9,0.0,2.0)
    rsi_s=_rsi_series_wilder(close,14)
    rsi_now=float(rsi_s.iloc[-1]) if not rsi_s.empty else 50.0
    rsi_prev=float(rsi_s.iloc[-4]) if len(rsi_s)>=4 else 50.0
    rsi_mid=_clamp((rsi_now-50.0)/16.0+(rsi_now-rsi_prev)/12.0,-2.0,2.0)
    mh=_macd_hist_series(close)
    if len(mh)>=5:
        hn=float(mh.iloc[-1])/price*100.0; hp=float(mh.iloc[-4])/price*100.0
        macd_turn=_clamp(hn*8.0+(hn-hp)*3.0,-2.0,2.0)
    else: macd_turn=0.0
    tr=pd.concat([(high-low).abs(),(high-close.shift(1)).abs(),(low-close.shift(1)).abs()],axis=1).max(axis=1)
    brange=float(tr.iloc[-25:-1].median() or 0.0); rr=float(tr.tail(3).mean()/brange) if brange>0 else 1.0
    bvol=float(vol.iloc[-25:-1].median() or 0.0); vr=float(vol.tail(3).mean()/bvol) if bvol>0 else 1.0
    cs=_close_strength(h1,3); pd_=1.0 if close.iloc[-1]>=close.iloc[-4] else -1.0
    vp=pd_*_clamp((rr-1.0)*0.7+(vr-1.0)*0.6+(cs-0.5)*1.2,-2.0,2.0)
    trigger=_clamp(ema_c*0.30+donchian*0.24+rsi_mid*0.18+macd_turn*0.16+vp*0.12,-2.5,2.5)
    return {"trigger":trigger,"early_trend_trigger":trigger,"ema_compression_breakout":ema_c,
            "rsi_midline_turn":rsi_mid,"macd_hist_turn":macd_turn,"donchian_breakout":donchian,"volume_price_confirm":vp}

def _rsi_series_wilder(close, period=14):
    if len(close)<period+2: return pd.Series([50.0]*len(close),index=close.index)
    delta=close.diff(); gain=delta.clip(lower=0).ewm(alpha=1/period,adjust=False).mean()
    loss=(-delta.clip(upper=0)).ewm(alpha=1/period,adjust=False).mean()
    rs=gain/loss.replace(0,np.nan)
    return (100.0-100.0/(1.0+rs)).replace([np.inf,-np.inf],np.nan).fillna(50.0).clip(0.0,100.0)

def _macd_hist_series(close):
    if len(close)<35: return pd.Series(dtype=float)
    fast=close.ewm(span=12,adjust=False).mean(); slow=close.ewm(span=26,adjust=False).mean()
    dif=fast-slow; dea=dif.ewm(span=9,adjust=False).mean(); return dif-dea

def _close_strength(df, window=3):
    if len(df)<window: return 0.5
    tail=df.tail(window); rng=(tail["h"]-tail["l"]).replace(0,np.nan)
    s=(tail["c"]-tail["l"])/rng; return float(s.mean()) if not s.isna().all() else 0.5

def _symbol_from_backtest_data(data, config):
    km=data.get("klines_map") or {}
    h1=_to_df(_get_klines(km,"1H") or _get_klines(km,"15m") or data.get("klines") or [])
    price=float(h1["c"].iloc[-1]) if not h1.empty else 0.0
    volume=float((h1["c"]*h1["vol"]).tail(48).sum()) if not h1.empty else 0.0
    extra={"klines":km,"funding_rate":data.get("funding_rate",0.0),"open_interest_change_pct":data.get("open_interest_change_pct",0.0),
           "on_chain":data.get("on_chain",{}),"llm_factors":data.get("llm_factors",{}),"social":data.get("social",{}),"news_sentiment":data.get("news_sentiment",0.0)}
    return _MinimalSymbol(inst_id=str(config.get("inst_id","BACKTEST") or "BACKTEST"),last_price=price,volume_24h=volume,
        price_change_24h=_pct_change(h1["c"],min(len(h1)-1,24))*100.0 if not h1.empty else 0.0,extra_data=extra)

def _blend(values,weights):
    total=denom=0.0
    for v,w in zip(values,weights):
        if np.isfinite(v):total+=float(v)*float(w);denom+=float(w)
    return total/denom if denom>0 else 0.0

def _direction_sign(momentum,trend):
    if abs(float(momentum or 0.0))>1e-9: return 1.0 if momentum>0 else -1.0
    if abs(float(trend or 0.0))>1e-9: return 1.0 if trend>0 else -1.0
    return 1.0


STRATEGY_NAME = "AI因子挖掘加密货币扫描策略"
STRATEGY_TYPE = "scan"
STRATEGY_CLASS = AIAutomatedAlphaCryptoScannerStrategy
BACKTEST_CLASS = AIAutomatedAlphaCryptoScannerStrategy
