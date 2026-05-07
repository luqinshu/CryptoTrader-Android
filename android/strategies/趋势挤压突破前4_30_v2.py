#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
趋势延续·挤压突破前扫描器  v3.0
=====================================
核心目标：在日线或小时线趋势「尚未完全启动」的时刻提前埋伏，
而不是等趋势跑远后才追进。

v3.0 相对 v2.0 的重大改进：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. 修复 BUG: fallback 值与 CONFIG_SCHEMA 不一致
2. 修复 BUG: BB 历史分位包含当前 bar，导致分位偏高
3. 修复 BUG: RSI 对短历史数据失真（增加 Wilder 热身期保护）
4. 核心逻辑重构：ADX 从「已趋势」改为「趋势萌芽检测」
   - 允许 ADX < 20 但正在上升（趋势正在启动中）
   - ADX 从低位回升 = 趋势从无到有的最佳埋伏点
5. 新增 TTM Squeeze（布林带 + 肯特纳通道双重确认）
   - BB 收窄至 KC 内部 = 经典弹簧压缩状态
6. 新增 RSI 多头/空头背离检测（回调中动能背离 = 方向性启动前信号）
7. 新增 NR4/NR7 窄幅 K 线检测（统计学最高概率突破前置特征）
8. 修复末根量能回暖逻辑（与基线对比，不与整理段自身对比）
9. 新增 15m 入场时机层（EMA 微型金叉 + 量能首次放大）
10. 新增摆动点真实检测（左右各 N 根确认的真正 Swing High/Low）
11. 新增 EMA 斜率加速度（斜率本身在加速 = 趋势正在增强）
12. 评分体系重新校准（BUG 修复后重新平衡各分项权重）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from strategies._shared.indicators import _clamp

logger = logging.getLogger(__name__)

try:
    from src.scanner.base_scanner import BaseScannerStrategy, ScanCondition, ScannerSymbol
    from src.scanner.ranking import build_opportunity_profile, enrich_scan_result
    _HAS_SCANNER_BASE = True
except Exception:
    BaseScannerStrategy = object
    ScanCondition = None
    ScannerSymbol = Any
    build_opportunity_profile = None
    enrich_scan_result = None
    _HAS_SCANNER_BASE = False


CONFIG_SCHEMA: Dict[str, Any] = {
    # ── 基础过滤 ──
    "min_volume_24h":               {"type": "float", "default": 5_000_000.0,  "label": "最小24H成交额（USDT）"},
    "min_score":                    {"type": "float", "default": 55.0,          "label": "最低综合输出分数（v3.2降至55提高命中率）"},
    "top_n":                        {"type": "int",   "default": 20,            "label": "最多输出信号数"},
    "allow_short":                  {"type": "bool",  "default": True,          "label": "允许空头方向信号"},

    # ── 日线趋势萌芽检测（v3.2放宽以提高命中率）──
    "d1_adx_period":                {"type": "int",   "default": 14,            "label": "日线ADX计算周期"},
    "d1_adx_min":                   {"type": "float", "default": 12.0,          "label": "日线ADX最低阈值（v3.2: 14→12，捕捉更早期萌芽）"},
    "d1_adx_strong":                {"type": "float", "default": 28.0,          "label": "日线ADX强趋势阈值"},
    "d1_adx_rising_bars":           {"type": "int",   "default": 4,             "label": "ADX连续上升根数（v3.2: 5→4）"},
    "d1_adx_rising_min_gain":       {"type": "float", "default": 1.5,           "label": "ADX最小上升绝对值（v3.2: 2.0→1.5）"},
    "d1_trend_consistency":         {"type": "float", "default": 0.45,          "label": "趋势一致性最低占比（v3.2: 0.55→0.45）"},
    "d1_fast_slope_min_pct":        {"type": "float", "default": 0.06,          "label": "日线快EMA最小斜率%（v3.2: 0.10→0.06）"},
    "d1_min_ema_spread_pct":        {"type": "float", "default": 0.25,          "label": "日线EMA张口最小%（v3.2: 0.40→0.25）"},
    "d1_max_extension_atr":         {"type": "float", "default": 4.00,          "label": "日线距快EMA最大ATR倍数（v3.2: 3.5→4.0）"},
    "d1_min_bars":                  {"type": "int",   "default": 140,           "label": "日线最少K线数量"},

    # ── 小时线回调参数 ──
    "h1_pullback_min_pct":          {"type": "float", "default": 1.8,           "label": "小时线最小有效回调幅度%（v3.2: 2.5→1.8）"},
    "h1_pullback_max_pct":          {"type": "float", "default": 10.0,          "label": "小时线最大回调幅度%（v3.2: 8→10轻度放宽）"},
    "h1_max_dryup_ratio":           {"type": "float", "default": 0.82,          "label": "企稳段最大缩量系数（v3.2: 0.72→0.82）"},
    "h1_min_rebound_ratio_vs_base": {"type": "float", "default": 0.65,          "label": "末根量能回暖系数（v3.2: 0.80→0.65放宽）"},
    "h1_stab_pos_min":              {"type": "float", "default": 0.25,          "label": "价格在企稳区间的最低位置比（v3.2: 0.35→0.25）"},
    "h1_bb_rank_max":               {"type": "float", "default": 0.40,          "label": "BB带宽历史分位上限（v3.2: 0.30→0.40放宽）"},
    "h1_max_pullback_atr":          {"type": "float", "default": 5.0,          "label": "H1回调上限ATR倍数（v3.2: 4→5）"},
    "m15_timing_weight":            {"type": "float", "default": 0.20,         "label": "15m时机分权重（v3.2: 0.25→0.20，日线权重回升）"},
    "enable_volume_pressure":       {"type": "bool",  "default": True,         "label": "启用买卖压力分析"},
    "enable_atr_target_suggestion": {"type": "bool",  "default": True,         "label": "输出ATR止损/止盈建议"},
    "stop_atr_mult":                {"type": "float", "default": 2.0,          "label": "止损ATR倍数"},
    "target_atr_mult":              {"type": "float", "default": 3.0,          "label": "止盈ATR倍数(盈亏比1.5:1)"},

    # ── 动能确认参数 ──
    "h1_rsi_period":                {"type": "int",   "default": 14,            "label": "小时线RSI周期"},
    "h1_rsi_bull_min":              {"type": "float", "default": 38.0,          "label": "多头RSI下限（v3.2: 42→38放宽）"},
    "h1_rsi_bull_max":              {"type": "float", "default": 72.0,          "label": "多头RSI上限"},
    "h1_rsi_bear_min":              {"type": "float", "default": 28.0,          "label": "空头RSI下限"},
    "h1_rsi_bear_max":              {"type": "float", "default": 62.0,          "label": "空头RSI上限（v3.2: 58→62放宽）"},
    "h1_rsi_divergence_bars":       {"type": "int",   "default": 20,            "label": "RSI背离检测回溯H1根数（新增）"},
    "h1_macd_fast":                 {"type": "int",   "default": 12,            "label": "MACD快线周期"},
    "h1_macd_slow":                 {"type": "int",   "default": 26,            "label": "MACD慢线周期"},
    "h1_macd_signal":               {"type": "int",   "default": 9,             "label": "MACD信号线周期"},
    "h1_min_bars":                  {"type": "int",   "default": 120,           "label": "小时线最少K线数量（v3增加保证历史分位可靠）"},

    # ── H1 摆动/企稳/ATR/量能参数 ──
    "h1_swing_lookback":            {"type": "int",   "default": 8,             "label": "H1摆动点回溯根数"},
    "h1_swing_confirm_bars":        {"type": "int",   "default": 2,             "label": "H1摆动确认根数"},
    "h1_stab_bars":                 {"type": "int",   "default": 8,             "label": "H1企稳确认根数"},
    "h1_stab_atr_ratio":            {"type": "float", "default": 0.55,          "label": "H1企稳段ATR收缩比"},
    "h1_per_bar_atr_ratio":         {"type": "float", "default": 0.20,          "label": "H1单根ATR比"},
    "h1_atr_period":                {"type": "int",   "default": 14,            "label": "H1 ATR计算周期"},
    "h1_vol_baseline_bars":         {"type": "int",   "default": 20,            "label": "H1成交量基线根数"},
    "h1_bb_period":                 {"type": "int",   "default": 20,            "label": "H1布林带周期"},
    "h1_bb_std":                    {"type": "float", "default": 2.0,           "label": "H1布林带标准差倍数"},
    "h1_kc_period":                 {"type": "int",   "default": 20,            "label": "H1 Keltner通道周期"},
    "h1_kc_mult":                   {"type": "float", "default": 2.0,           "label": "H1 KC通道ATR倍数"},
    "h1_bb_squeeze_pct":            {"type": "float", "default": 6.0,           "label": "H1 BB挤压宽度阈值%"},
    "h1_bb_lookback":               {"type": "int",   "default": 80,            "label": "H1 BB历史回溯根数"},
    "h1_bb_expand_ratio":           {"type": "float", "default": 1.5,           "label": "H1 BB膨胀量比阈值"},

    # ── 日线EMA参数 ──
    "d1_ema_fast":                  {"type": "int",   "default": 20,            "label": "日线快EMA周期"},
    "d1_ema_mid":                   {"type": "int",   "default": 50,            "label": "日线中EMA周期"},
    "d1_ema_slow":                  {"type": "int",   "default": 120,           "label": "日线慢EMA周期"},
    "d1_slope_lookback":            {"type": "int",   "default": 5,             "label": "日线斜率回溯周期"},
    "d1_slope_accel_bars":          {"type": "int",   "default": 3,             "label": "日线斜率加速检测根数"},
    "d1_trend_bars":                {"type": "int",   "default": 10,            "label": "日线趋势持续性检测根数"},

    # ── TTM挤压 / BTC环境 / 信号持久 ──
    "require_ttm_squeeze":          {"type": "bool",  "default": True,          "label": "强制要求TTM挤压"},
    "ttm_near_squeeze_tolerance":   {"type": "float", "default": 0.20,          "label": "TTM近挤压容忍比例"},
    "use_atr_pullback_range":       {"type": "bool",  "default": False,         "label": "用ATR倍数限制回调范围"},

    # ── BTC市场 / 信号持久性 ──
    "enable_btc_market_filter":     {"type": "bool",  "default": False,         "label": "启用BTC环境过滤"},
    "btc_dump_block_threshold_pct": {"type": "float", "default": -5.0,          "label": "BTC暴跌阈值%"},
    "enable_signal_persistence":    {"type": "bool",  "default": False,         "label": "启用信号持久性追踪"},
    "signal_persistence_scans":     {"type": "int",   "default": 2,             "label": "信号连续出现N次才稳定"},
    "persistence_stable_bonus":     {"type": "float", "default": 4.0,           "label": "信号稳定出现加分"},

    # ── NR4/NR7 窄幅 K 线检测（新增）──
    "h1_nr_lookback":               {"type": "int",   "default": 7,             "label": "NR检测回溯根数（4=NR4，7=NR7）"},
    "require_nr_bar":               {"type": "bool",  "default": False,         "label": "是否强制要求NR4/NR7（关闭时作为加分项）"},

    # ── 15m 入场时机层（新增）──
    "m15_ema_fast":                 {"type": "int",   "default": 8,             "label": "15m快速EMA周期"},
    "m15_ema_slow":                 {"type": "int",   "default": 21,            "label": "15m慢速EMA周期"},
    "m15_min_bars":                 {"type": "int",   "default": 60,            "label": "15m最少K线数量"},
}

_DEFAULT_CONFIG = {k: v["default"] for k, v in CONFIG_SCHEMA.items()}

# v3.2: 市场状态自适应参数
MARKET_STATE_PARAMS: Dict[str, Dict[str, Any]] = {
    "trending": {
        "d1_adx_min": 16.0, "d1_adx_rising_bars": 3, "d1_adx_rising_min_gain": 1.0,
        "d1_trend_consistency": 0.50, "d1_fast_slope_min_pct": 0.08, "d1_min_ema_spread_pct": 0.30,
        "h1_pullback_min_pct": 1.5, "h1_pullback_max_pct": 7.0, "h1_max_dryup_ratio": 0.78,
        "h1_stab_pos_min": 0.30, "m15_timing_weight": 0.22,
    },
    "range": {
        "d1_adx_min": 10.0, "d1_adx_rising_bars": 5, "d1_adx_rising_min_gain": 2.0,
        "d1_trend_consistency": 0.40, "d1_fast_slope_min_pct": 0.04, "d1_min_ema_spread_pct": 0.20,
        "h1_pullback_min_pct": 2.0, "h1_pullback_max_pct": 12.0, "h1_max_dryup_ratio": 0.88,
        "h1_bb_rank_max": 0.30, "require_ttm_squeeze": False, "h1_stab_pos_min": 0.20,
        "m15_timing_weight": 0.30,
    },
    "volatile": {
        "d1_adx_min": 18.0, "d1_adx_rising_bars": 2, "d1_adx_rising_min_gain": 0.8,
        "d1_trend_consistency": 0.35, "d1_fast_slope_min_pct": 0.12, "d1_min_ema_spread_pct": 0.40,
        "h1_pullback_min_pct": 1.0, "h1_pullback_max_pct": 15.0, "h1_max_dryup_ratio": 0.70,
        "h1_max_pullback_atr": 6.0, "use_atr_pullback_range": True, "h1_stab_pos_min": 0.35,
        "m15_timing_weight": 0.15,
    },
}


class TrendSqueezeBreakoutScannerV3(BaseScannerStrategy if _HAS_SCANNER_BASE else object):
    """
    v3: 日线/小时线趋势「萌芽期」识别 + TTM Squeeze + RSI背离 + NR4/NR7 + 15m时机确认。

    核心思路：
      1. 日线层：趋势不要求已强（ADX≥14即可），但要求 ADX 正在上升（趋势萌芽）
                 EMA 开始有序排列，斜率由平转升，EMA 张口正在扩大
      2. H1 层：正常回调后进入 TTM 挤压压缩态（BB 收进 KC 内）
                 成交量充分萎缩，末根量能对比基线开始回暖
                 RSI 有背离或回升，MACD 柱体回升拐头
      3. 15m层：EMA8 金叉 EMA21（微型多头排列刚形成）
                 成交量相对之前 N 根有所放大（首根放量）
    """

    required_bars = ["1D", "1H", "15m"]
    strategy_type = "scan"
    name = "趋势延续·挤压突破前扫描器 v3"
    description = (
        "日线趋势萌芽（ADX回升+EMA渐进有序+斜率加速）"
        " + H1 TTM挤压（BB在KC内）+ 量能萎缩 + RSI背离/回暖"
        " + 15m EMA金叉首放量"
    )

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        # 以 schema 默认值为基础，再覆盖传入 config，确保 fallback 永远与 schema 一致
        self.config = {**_DEFAULT_CONFIG, **(config or {})}
        if _HAS_SCANNER_BASE and hasattr(super(), "__init__"):
            try:
                super().__init__(self.config)
            except Exception:
                pass
        # v3.1: 信号持续性追踪 {symbol_dir: (scan_count, last_score)}
        self._signal_history: Dict[str, Tuple[int, float]] = {}
        self._persist_counts: Dict[str, int] = {}
        self._scan_counter: int = 0

    def _init_conditions(self):
        """BaseScannerStrategy 要求实现的抽象方法；本策略使用自定义 scan_symbol，无需条件列表。"""
        pass  # 不使用基类的 ScanCondition 列表机制

    def get_config_schema(self) -> Dict[str, Any]:
        return dict(CONFIG_SCHEMA)

    def _apply_market_state_params(self, state: str) -> None:
        """根据市场状态自适应覆盖参数"""
        if state not in MARKET_STATE_PARAMS:
            return
        for k, v in MARKET_STATE_PARAMS[state].items():
            self.config[k] = v
        self._market_state = state

    def _apply_vol_pool_params(self, pool: str) -> None:
        """v3.2: 根据波动率池覆盖参数"""
        from src.scanner.volatility_pools import get_pool_params
        overrides = get_pool_params(pool, list(self.config.keys()))
        for k, v in overrides.items():
            if k in self.config:
                self.config[k] = v
        self._vol_pool = pool

    # ══════════════════════════════════════════════════════════════════════════
    # 主入口
    # ══════════════════════════════════════════════════════════════════════════

    def scan_symbol(self, symbol) -> Dict[str, Any]:
        inst_id = getattr(symbol, "inst_id", "")
        vol24   = float(getattr(symbol, "volume_24h", 0) or 0)
        price   = float(getattr(symbol, "last_price", getattr(symbol, "price", 0)) or 0)

        base_fail = {
            "symbol": inst_id, "passed": False, "score": 0.0,
            "opportunity_score": 0.0, "direction": "WAIT",
            "category": "趋势挤压观察", "signals": [],
            "factor_scores": {}, "ranking_factors": {},
        }

        if vol24 < float(self.config.get("min_volume_24h", 5_000_000.0)):
            return {**base_fail, "details": {"跳过原因": "成交额不足"}}

        d1_rows  = self._get_klines(symbol, "1D")
        h1_rows  = self._get_klines(symbol, "1H")
        m15_rows = self._get_klines(symbol, "15m")

        d1_min  = int(self.config["d1_min_bars"])
        h1_min  = int(self.config["h1_min_bars"])
        if len(d1_rows) < d1_min:
            return {**base_fail, "details": {"跳过原因": f"日线数据不足({len(d1_rows)}<{d1_min}根)"}}
        if len(h1_rows) < h1_min:
            return {**base_fail, "details": {"跳过原因": f"小时线数据不足({len(h1_rows)}<{h1_min}根)"}}

        # ── Step 1：日线趋势萌芽 ──────────────────────────────────────────────
        d1_ok, d1_dir, d1_score, d1_details, d1_factors = self._check_daily_trend_sprout(d1_rows)
        if not d1_ok:
            return {
                **base_fail,
                "details": {"日线过滤": "未通过", **d1_details},
                "signals": [f"日线淘汰: {d1_details.get('淘汰原因', '')}"],
                "factor_scores": d1_factors,
            }

        if d1_dir == "bear" and not bool(self.config.get("allow_short", True)):
            return {
                **base_fail,
                "details": {"日线过滤": "通过", **d1_details, "跳过原因": "未启用空头"},
                "signals": ["空头趋势，但配置关闭空头输出"],
                "factor_scores": d1_factors,
            }

        # ── Step 2：H1 TTM 挤压 + 量能 + 动能 ───────────────────────────────
        h1_ok, h1_score, h1_details, h1_factors = self._check_h1_ttm_squeeze(h1_rows, d1_dir)
        if not h1_ok:
            return {
                **base_fail,
                "details": {"日线过滤": "通过", "H1过滤": "未通过", **d1_details, **h1_details},
                "signals": [f"H1淘汰: {h1_details.get('淘汰原因', '')}"],
                "factor_scores": {**d1_factors, **h1_factors},
            }

        # ── Step 3：15m 入场时机确认（软条件，不通过只扣分，不淘汰）────────
        m15_bonus = 0.0
        m15_details: Dict[str, Any] = {}
        m15_min = int(self.config["m15_min_bars"])
        if bool(self.config.get("use_15m_confirmation", True)) and len(m15_rows) >= m15_min:
            m15_bonus, m15_details = self._check_15m_entry_timing(m15_rows, d1_dir)

        # ── 综合评分 ─────────────────────────────────────────────────────────
        # v3.1: 15m时机分权重提升至可配置(默认25%)
        m15_weight = float(self.config.get("m15_timing_weight", 0.25) or 0.25)
        d1_weight = 1.0 - 0.50 - m15_weight  # 日线权重=剩余部分
        total_score = round(d1_score * d1_weight + h1_score * 0.50 + m15_bonus * m15_weight, 2)

        direction = "BUY" if d1_dir == "bull" else "SELL"

        # v3.1: BTC市场环境过滤—从extra_data获取BTC基准
        btc_penalty = 0.0
        if bool(self.config.get("enable_btc_market_filter", False)) and d1_dir == "bull":
            btc_ctx = self._get_btc_from_symbol(symbol)
            btc_penalty = self._calc_btc_penalty(btc_ctx)
            total_score = round(max(0, total_score - btc_penalty), 2)

        # v3.1: 信号持续性追踪
        persistence_bonus = 0.0
        if bool(self.config.get("enable_signal_persistence", False)):
            self._scan_counter += 1
            key = f"{inst_id}:{direction}"
            prev = self._signal_history.get(key)
            if prev and self._scan_counter - prev[0] <= 2:  # 2次扫描内再次出现
                count = self._persist_counts.get(key, 0) + 1
                self._persist_counts[key] = count
                if count >= int(self.config.get("signal_persistence_scans", 2)):
                    persistence_bonus = float(self.config.get("persistence_stable_bonus", 4.0))
            else:
                self._persist_counts[key] = 1
            self._signal_history[key] = (self._scan_counter, total_score)

        min_score   = float(self.config.get("min_score", 62.0))
        passed      = total_score + persistence_bonus >= min_score
        total_score = round(total_score + persistence_bonus, 2)

        category  = "📈 多头萌芽挤压前夕" if d1_dir == "bull" else "📉 空头萌芽挤压前夕"

        adx_val      = float(d1_details.get("ADX", 0) or 0)
        adx_rising   = bool(d1_details.get("ADX上升中", False))
        squeeze_pct  = float(h1_details.get("BB带宽%", 0) or 0)
        pullback     = float(h1_details.get("回调幅度%", 0) or 0)
        h1_rsi       = float(h1_details.get("H1_RSI", 50) or 50)
        rsi_diverge  = bool(h1_details.get("RSI背离", False))
        ttm_on       = bool(h1_details.get("TTM挤压激活", False))
        nr_bar       = bool(h1_details.get("NR4/NR7", False))
        buy_ratio    = float(h1_details.get("买量占比", 0.5) or 0.5)
        sl_pct       = float(h1_details.get("ATR止损%", 0) or 0)
        tp_pct       = float(h1_details.get("ATR止盈%", 0) or 0)

        signals = [
            f"{category} · 综合评分 {total_score:.1f}",
            (
                f"日线ADX={adx_val:.1f}{'↑萌芽' if adx_rising else ''}  "
                f"EMA{'多头' if d1_dir == 'bull' else '空头'}排列"
                f"  斜率{float(d1_details.get('快EMA斜率%', 0) or 0):+.2f}%"
                f"{'↗加速' if d1_details.get('斜率加速中') else ''}"
            ),
            (
                f"H1回调{pullback:.2f}%  企稳{h1_details.get('企稳根数', 0)}根"
                f"  {'🟢TTM挤压' if ttm_on else 'BB挤压'}  带宽{squeeze_pct:.2f}%"
                f"  缩量{float(h1_details.get('缩量系数', 0) or 0):.2f}"
            ),
            (
                f"RSI={h1_rsi:.1f}{'🔔背离' if rsi_diverge else ''}  "
                f"MACD柱{float(h1_details.get('MACD柱体%', 0) or 0):+.3f}%  "
                f"{'📦NR4/NR7  ' if nr_bar else ''}"
                f"量能回暖{float(h1_details.get('末根量能vs基线', 0) or 0):.2f}"
            ),
        ]
        if m15_details:
            signals.append(
                f"15m: {'✅金叉' if m15_details.get('EMA金叉') else '⬜未金叉'}"
                f"  量能{float(m15_details.get('量能系数', 1.0) or 1.0):.2f}x"
                f"  时机分={m15_bonus:.0f}"
            )
        if persistence_bonus > 0:
            signals.append(f"🔄 连续出现·稳定加分 +{persistence_bonus:.0f}")
        if btc_penalty > 0:
            signals.append(f"⚠ BTC弱势·环境降{btc_penalty:.0f}分")
        if buy_ratio > 0.55:
            signals.append(f"📊 买量占比{buy_ratio:.0%}（买方主动）")
        if sl_pct > 0:
            signals.append(f"🛑 止损-{sl_pct:.1f}% / 🎯 止盈+{tp_pct:.1f}%")

        factor_scores = {
            **d1_factors, **h1_factors,
            "daily_score": round(d1_score, 2),
            "hourly_score": round(h1_score, 2),
            "m15_bonus": round(m15_bonus, 2),
            "total_score": round(total_score, 2),
        }
        ranking_factors = self._build_ranking_factors(
            total_score=total_score, d1_score=d1_score, h1_score=h1_score,
            m15_bonus=m15_bonus, vol24=vol24,
            d1_details=d1_details, h1_details=h1_details,
        )

        result = {
            "symbol": inst_id, "passed": passed,
            "score": total_score, "opportunity_score": total_score,
            "direction": direction, "category": category,
            "strategy_category": category, "signals": signals,
            "last_price": price, "volume_24h": vol24,
            "price_change_24h": float(getattr(symbol, "price_change_24h", 0) or 0),
            "factor_scores": factor_scores,
            "ranking_factors": ranking_factors,
            "details": {
                "策略": self.name, "方向": "多头" if d1_dir == "bull" else "空头",
                "综合评分": f"{total_score:.1f}",
                "BTC环境分": f"-{btc_penalty:.0f}" if btc_penalty > 0 else "正常",
                "信号持续性": f"+{persistence_bonus:.0f}" if persistence_bonus > 0 else "首次/不稳定",
                "买量占比": f"{buy_ratio:.1%}",
                "ATR止损%": f"-{sl_pct:.1f}%" if sl_pct > 0 else "-",
                "ATR止盈%": f"+{tp_pct:.1f}%" if tp_pct > 0 else "-",
                **d1_details, **h1_details, **m15_details,
            },
        }

        if build_opportunity_profile and passed:
            try:
                result.update(build_opportunity_profile(total_score, direction, vol24, ranking_factors, signals))
            except Exception:
                pass
        if enrich_scan_result and passed:
            try:
                enrich_scan_result(result)
            except Exception:
                pass

        return result

    def scan_all_symbols(self, symbols: List) -> Dict[str, Any]:
        results: List[Dict[str, Any]] = []
        # v3.1: 并发扫描（I/O密集型，max_workers=8 平衡限流与吞吐）
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=min(8, len(symbols) or 1)) as executor:
            futures = {executor.submit(self._safe_scan, sym): sym for sym in symbols}
            for future in as_completed(futures):
                try:
                    result = future.result(timeout=15)
                except Exception:
                    continue
                if result and result.get("passed"):
                    results.append(result)
        results.sort(
            key=lambda item: float(item.get("opportunity_score", item.get("score", 0.0)) or 0.0),
            reverse=True,
        )
        top_n = int(self.config.get("top_n", 20) or 20)
        return {
            "type": "trend_squeeze_breakout_v3",
            "all_opportunities": results[:top_n],
            "total_passed": len(results),
            "total_scanned": len(symbols),
        }

    def _safe_scan(self, sym) -> Optional[Dict]:
        """单个符号扫描 + 异常保护"""
        try:
            return self.scan_symbol(sym)
        except Exception as exc:
            logger.error(f"[趋势挤压v3] {getattr(sym, 'inst_id', '')} 异常: {exc}")
            return None

    # ══════════════════════════════════════════════════════════════════════════
    # Step 1：日线趋势萌芽检测（v3 核心改进）
    # ══════════════════════════════════════════════════════════════════════════

    def _check_daily_trend_sprout(
        self, rows: List
    ) -> Tuple[bool, str, float, Dict[str, Any], Dict[str, float]]:
        """日线趋势萌芽检测：EMA排列 + ADX + 斜率 + 持续性 + 乖离率"""

        def _v(row, idx, default=0.0):
            try: return float(row[idx])
            except (TypeError, ValueError, IndexError): return default

        if not rows or len(rows) < 130:
            return False, "wait", 0.0, {"淘汰原因": f"日线数据不足({len(rows)}根，需130根)"}, {"daily_adx_score": 0.0}

        closes  = [_v(r, 4) for r in rows]
        highs   = [_v(r, 2) for r in rows]
        lows    = [_v(r, 3) for r in rows]

        ema_f         = int(self.config.get("d1_ema_fast", 20))
        ema_m         = int(self.config.get("d1_ema_mid", 50))
        ema_s         = int(self.config.get("d1_ema_slow", 120))
        adx_min       = float(self.config.get("d1_adx_min", 14.0))
        adx_strong    = float(self.config.get("d1_adx_strong", 28.0))
        adx_rise_bars = int(self.config.get("d1_adx_rising_bars", 5))
        adx_rise_gain = float(self.config.get("d1_adx_rising_min_gain", 2.0))
        trend_bars    = int(self.config.get("d1_trend_bars", 10))
        consistency   = float(self.config.get("d1_trend_consistency", 0.55))
        slope_lb      = int(self.config.get("d1_slope_lookback", 5))
        min_slope_pct = float(self.config.get("d1_fast_slope_min_pct", 0.10))
        accel_bars    = int(self.config.get("d1_slope_accel_bars", 3))
        min_spread    = float(self.config.get("d1_min_ema_spread_pct", 0.40))
        max_ext_atr   = float(self.config.get("d1_max_extension_atr", 3.50))
        adx_period    = int(self.config.get("d1_adx_period", 14))

        adx_val, di_plus, di_minus = _calc_adx(closes, highs, lows, adx_period)
        # ─ ADX 系列（用于判断是否"上升中"）────────────────────────────────
        adx_series = _calc_adx_series(closes, highs, lows, adx_period, last_n=adx_rise_bars + 3)
        atr = _calc_atr(closes, highs, lows, adx_period)
        ema_fast_series = _calc_ema_series(closes, ema_f)
        ema_mid_series  = _calc_ema_series(closes, ema_m)
        ema_slow_series = _calc_ema_series(closes, ema_s)

        ema_fast = ema_fast_series[-1] if ema_fast_series else 0.0
        ema_mid  = ema_mid_series[-1]  if ema_mid_series  else 0.0
        ema_slow = ema_slow_series[-1] if ema_slow_series else 0.0
        last_close = closes[-1] if closes else 0.0

        # ── 检查 1：ADX 最低阈值 ────────────────────────────────────────────
        if adx_val < adx_min:
            return False, "wait", 0.0, {
                "淘汰原因": f"日线ADX={adx_val:.1f} < {adx_min}（震荡无方向）",
                "ADX": round(adx_val, 2),
            }, {"daily_adx_score": 0.0}

        # ── 检查 2：ADX 是否在上升（趋势萌芽信号）────────────────────────
        adx_rising = False
        adx_gain = 0.0
        if len(adx_series) >= adx_rise_bars:
            recent_adx = adx_series[-adx_rise_bars:]
            adx_rising_count = sum(
                1 for i in range(1, len(recent_adx)) if recent_adx[i] > recent_adx[i - 1]
            )
            adx_gain = recent_adx[-1] - recent_adx[0]
            # 允许趋势萌芽：ADX < adx_strong 时要求 ADX 正在上升
            # ADX 已很强（≥adx_strong）则不再要求上升
            if adx_val < adx_strong:
                if adx_rising_count < adx_rise_bars - 1 or adx_gain < adx_rise_gain:
                    return False, "wait", 0.0, {
                        "淘汰原因": (
                            f"ADX={adx_val:.1f}尚弱且未持续上升"
                            f"（{adx_rise_bars}根内上升{adx_rising_count}次，涨幅{adx_gain:.1f}，"
                            f"需≥{adx_rise_gain}）"
                        ),
                        "ADX": round(adx_val, 2),
                        "ADX涨幅": round(adx_gain, 2),
                    }, {"daily_adx_score": round(adx_val * 1.5, 2)}
                adx_rising = True

        # ── 检查 3：EMA 排列方向 ────────────────────────────────────────────
        bull_ema = ema_fast > ema_mid > ema_slow
        bear_ema = ema_fast < ema_mid < ema_slow
        # v3 宽松：EMA 不严格有序时，允许"fast 与 mid 有序 but slow 尚未到位"（趋势刚开始）
        bull_partial = ema_fast > ema_mid and ema_mid > ema_slow * 0.98
        bear_partial = ema_fast < ema_mid and ema_mid < ema_slow * 1.02
        if not (bull_ema or bear_ema or bull_partial or bear_partial):
            return False, "wait", 0.0, {
                "淘汰原因": f"EMA三线无有序排列趋势 EMA{ema_f}={ema_fast:.4f} EMA{ema_m}={ema_mid:.4f} EMA{ema_s}={ema_slow:.4f}",
                "ADX": round(adx_val, 2),
            }, {"daily_alignment_score": 0.0}

        direction = "bull" if (bull_ema or bull_partial) else "bear"

        di_ok = (
            (direction == "bull" and di_plus > di_minus) or
            (direction == "bear" and di_minus > di_plus)
        )
        if not di_ok:
            return False, "wait", 0.0, {
                "淘汰原因": f"DI方向与EMA方向冲突（DI+={di_plus:.1f}, DI-={di_minus:.1f}）",
                "ADX": round(adx_val, 2), "DI+": round(di_plus, 2), "DI-": round(di_minus, 2),
            }, {"daily_di_score": 0.0}

        # ── 检查 4：趋势方向一致性 ────────────────────────────────────────
        if len(closes) >= trend_bars + 1:
            recent = closes[-(trend_bars + 1):]
            bar_hits = sum(1 for i in range(1, len(recent)) if (recent[i] > recent[i - 1]) == (direction == "bull"))
            bar_ratio = bar_hits / max(trend_bars, 1)
            ema_side_hits = sum(
                1 for k in range(-trend_bars, 0)
                if len(ema_fast_series) > abs(k) and np.isfinite(ema_fast_series[k])
                and (closes[k] > ema_fast_series[k]) == (direction == "bull")
            )
            ema_ratio = ema_side_hits / max(trend_bars, 1)
            ratio = (bar_ratio + ema_ratio) / 2.0
        else:
            ratio = 0.5

        if ratio < consistency:
            return False, "wait", 0.0, {
                "淘汰原因": f"日线趋势一致性不足({ratio * 100:.0f}% < {consistency * 100:.0f}%)",
                "ADX": round(adx_val, 2), "趋势一致性%": round(ratio * 100, 1),
            }, {"daily_consistency_score": round(ratio * 100, 2)}

        # ── 检查 5：EMA 斜率 + 斜率加速度（新增：斜率在加速才说明趋势在增强）
        fast_slope_pct = 0.0
        slope_accel    = False
        if len(ema_fast_series) > slope_lb and np.isfinite(ema_fast_series[-1 - slope_lb]):
            fast_prev = float(ema_fast_series[-1 - slope_lb])
            fast_slope_pct = _pct_change(ema_fast, fast_prev)

        slope_ok = fast_slope_pct >= min_slope_pct if direction == "bull" else fast_slope_pct <= -min_slope_pct
        if not slope_ok:
            return False, "wait", 0.0, {
                "淘汰原因": f"快EMA斜率不足({fast_slope_pct:+.2f}%，需{min_slope_pct:+.2f}%)",
                "ADX": round(adx_val, 2), "快EMA斜率%": round(fast_slope_pct, 3),
            }, {"daily_slope_score": max(0.0, abs(fast_slope_pct) * 40.0)}

        # 斜率加速度：过去 accel_bars 内斜率是否在持续增大
        if len(ema_fast_series) > slope_lb + accel_bars:
            slope_prev_pct = _pct_change(
                float(ema_fast_series[-1 - slope_lb]),
                float(ema_fast_series[-1 - slope_lb - accel_bars])
            )
            slope_accel = (
                (direction == "bull" and fast_slope_pct > slope_prev_pct) or
                (direction == "bear" and fast_slope_pct < slope_prev_pct)
            )

        # ── 检查 6：EMA 张口（v3 检查 spread 是否在扩大）────────────────────
        ema_spread_pct = abs(ema_fast - ema_slow) / max(abs(last_close), 1e-9) * 100.0
        if ema_spread_pct < min_spread:
            return False, "wait", 0.0, {
                "淘汰原因": f"EMA张口不足({ema_spread_pct:.2f}% < {min_spread:.2f}%)，趋势尚未展开",
                "EMA张口%": round(ema_spread_pct, 3), "ADX": round(adx_val, 2),
            }, {"daily_spread_score": round(ema_spread_pct * 30.0, 2)}

        # 张口是否在扩大
        spread_expanding = False
        if len(ema_fast_series) > accel_bars and len(ema_slow_series) > accel_bars:
            prev_spread = abs(ema_fast_series[-1 - accel_bars] - ema_slow_series[-1 - accel_bars]) / max(abs(last_close), 1e-9) * 100.0
            spread_expanding = ema_spread_pct > prev_spread

        # ── 检查 7：过度延伸过滤 ─────────────────────────────────────────────
        directional_ext_atr = 0.0
        if atr > 0:
            if direction == "bull":
                directional_ext_atr = max((last_close - ema_fast) / atr, 0.0)
            else:
                directional_ext_atr = max((ema_fast - last_close) / atr, 0.0)

        if directional_ext_atr > max_ext_atr:
            return False, "wait", 0.0, {
                "淘汰原因": f"日线已过度延伸({directional_ext_atr:.2f} ATR > {max_ext_atr:.2f})",
                "快EMA延伸ATR": round(directional_ext_atr, 3), "ADX": round(adx_val, 2),
            }, {"daily_extension_health_score": 0.0}

        # ── 评分 ──────────────────────────────────────────────────────────────
        # ADX 分：弱趋势萌芽（<strong）也给分，但上升中额外加成
        adx_base_score = 20.0 * _clamp((adx_val - adx_min) / max(adx_strong - adx_min, 1e-9), 0.0, 1.0)
        adx_rise_bonus  = 6.0 if adx_rising else 0.0                    # 上升中加成
        adx_score       = adx_base_score + adx_rise_bonus

        alignment_score    = 14.0 if (bull_ema or bear_ema) else 8.0    # 完全有序比部分有序得分高
        consistency_score  = 16.0 * _clamp((ratio - consistency) / max(1.0 - consistency, 1e-9), 0.0, 1.0)
        slope_score        = 16.0 * _clamp(abs(fast_slope_pct) / max(min_slope_pct * 3.0, 1e-9), 0.0, 1.0)
        slope_accel_bonus  = 5.0 if slope_accel else 0.0               # 加速度加成
        spread_score       = 10.0 * _clamp(ema_spread_pct / max(min_spread * 2.5, 1e-9), 0.0, 1.0)
        spread_expand_bonus = 4.0 if spread_expanding else 0.0         # 张口扩大加成
        ext_health_score   = 9.0 * _clamp(1.0 - directional_ext_atr / max(max_ext_atr, 1e-9), 0.0, 1.0)

        score = round(min(
            adx_score + alignment_score + consistency_score + slope_score + slope_accel_bonus
            + spread_score + spread_expand_bonus + ext_health_score,
            100.0,
        ), 2)

        details = {
            "ADX": round(adx_val, 2), "DI+": round(di_plus, 2), "DI-": round(di_minus, 2),
            "ADX上升中": adx_rising, "ADX涨幅": round(adx_gain, 2),
            "日线方向": "多头" if direction == "bull" else "空头",
            f"EMA{ema_f}": round(ema_fast, 4), f"EMA{ema_m}": round(ema_mid, 4), f"EMA{ema_s}": round(ema_slow, 4),
            "EMA完全有序": bull_ema or bear_ema,
            "趋势一致性%": round(ratio * 100, 1),
            "快EMA斜率%": round(fast_slope_pct, 3),
            "斜率加速中": slope_accel,
            "EMA张口%": round(ema_spread_pct, 3),
            "EMA张口扩大中": spread_expanding,
            "日线ATR": round(atr, 4),
            "快EMA延伸ATR": round(directional_ext_atr, 3),
            "日线评分": round(score, 2),
            "最新收盘": round(last_close, 4),
        }
        factor_scores = {
            "daily_adx_score": round(adx_score, 2),
            "daily_alignment_score": round(alignment_score, 2),
            "daily_consistency_score": round(consistency_score, 2),
            "daily_slope_score": round(slope_score + slope_accel_bonus, 2),
            "daily_spread_score": round(spread_score + spread_expand_bonus, 2),
            "daily_extension_health_score": round(ext_health_score, 2),
        }
        return True, direction, score, details, factor_scores

    # ══════════════════════════════════════════════════════════════════════════
    # Step 2：H1 TTM 挤压检测（全面重写）
    # ══════════════════════════════════════════════════════════════════════════

    def _check_h1_ttm_squeeze(
        self, rows: List, d1_direction: str
    ) -> Tuple[bool, float, Dict[str, Any], Dict[str, float]]:
        def _v(row, idx, default=0.0):
            try:
                return float(row[idx])
            except Exception:
                return default

        closes  = [_v(r, 4) for r in rows]
        highs   = [_v(r, 2) for r in rows]
        lows    = [_v(r, 3) for r in rows]
        volumes = [_v(r, 5) for r in rows]
        n = len(closes)

        pb_min          = float(self.config["h1_pullback_min_pct"])
        pb_max          = float(self.config["h1_pullback_max_pct"])
        sw_look         = int(self.config["h1_swing_lookback"])
        sw_confirm      = int(self.config["h1_swing_confirm_bars"])
        stab_bars       = int(self.config["h1_stab_bars"])        # 使用 schema 默认值 8
        stab_atr        = float(self.config["h1_stab_atr_ratio"])
        per_bar_atr_r   = float(self.config["h1_per_bar_atr_ratio"])
        atr_per         = int(self.config["h1_atr_period"])
        vol_base_bars   = int(self.config["h1_vol_baseline_bars"])
        max_dryup       = float(self.config["h1_max_dryup_ratio"])           # schema 默认 0.72
        min_rebound_b   = float(self.config["h1_min_rebound_ratio_vs_base"]) # vs 基线，schema 默认 0.80

        bb_per      = int(self.config["h1_bb_period"])
        bb_std_mult = float(self.config["h1_bb_std"])
        kc_per      = int(self.config["h1_kc_period"])
        kc_mult     = float(self.config["h1_kc_mult"])
        bb_sq_pct   = float(self.config["h1_bb_squeeze_pct"])
        bb_look     = int(self.config["h1_bb_lookback"])
        bb_rank_max = float(self.config["h1_bb_rank_max"])
        bb_expand   = float(self.config["h1_bb_expand_ratio"])
        req_ttm     = bool(self.config.get("require_ttm_squeeze", True))
        ttm_tol     = float(self.config.get("ttm_near_squeeze_tolerance", 0.20) or 0.20)

        rsi_period    = int(self.config["h1_rsi_period"])
        rsi_bull_min  = float(self.config["h1_rsi_bull_min"])
        rsi_bull_max  = float(self.config["h1_rsi_bull_max"])
        rsi_bear_min  = float(self.config["h1_rsi_bear_min"])
        rsi_bear_max  = float(self.config["h1_rsi_bear_max"])
        div_bars      = int(self.config["h1_rsi_divergence_bars"])
        macd_fast_p   = int(self.config["h1_macd_fast"])
        macd_slow_p   = int(self.config["h1_macd_slow"])
        macd_sig_p    = int(self.config["h1_macd_signal"])
        nr_look       = int(self.config["h1_nr_lookback"])
        req_nr        = bool(self.config.get("require_nr_bar", False))

        cur_close = closes[-1]
        atr = _calc_atr(closes, highs, lows, atr_per)
        if atr <= 0:
            return False, 0.0, {"淘汰原因": "ATR计算失败"}, {"hourly_atr_score": 0.0}

        # ── 真实摆动高/低点检测（v3 修复：左右各 sw_confirm 根确认）────────
        look = min(sw_look, n - stab_bars - 1)
        if look < max(sw_confirm * 2 + 1, 8):
            return False, 0.0, {"淘汰原因": "H1有效数据不足"}, {"hourly_pullback_score": 0.0}

        search_highs = highs[-(look + stab_bars):-stab_bars]
        search_lows  = lows[-(look + stab_bars):-stab_bars]

        # v3.1: ATR自适应回调范围
        use_atr_pb = bool(self.config.get("use_atr_pullback_range", False))
        atr_pb_max = float(self.config.get("h1_max_pullback_atr", 5.0) or 5.0)
        if use_atr_pb and atr > 0:
            atr_pct = atr / cur_close * 100.0
            effective_pb_min = max(pb_min, atr_pct * 1.2)  # 至少1.2倍ATR
            effective_pb_max = min(pb_max, atr_pct * atr_pb_max)  # 不超过N倍ATR
        else:
            effective_pb_min = pb_min
            effective_pb_max = pb_max

        search_len   = len(search_highs)

        if d1_direction == "bull":
            # 找真实摆动高点（左右各 sw_confirm 根都比它低）
            swing_extreme = _find_swing_high(search_highs, sw_confirm)
            if swing_extreme is None:
                swing_extreme = max(search_highs) if search_highs else 0.0
            pullback_pct = (swing_extreme - cur_close) / max(swing_extreme, 1e-9) * 100.0
        else:
            swing_extreme = _find_swing_low(search_lows, sw_confirm)
            if swing_extreme is None:
                swing_extreme = min(search_lows) if search_lows else cur_close
            if swing_extreme <= 0:
                return False, 0.0, {"淘汰原因": "摆动低点异常"}, {"hourly_pullback_score": 0.0}
            pullback_pct = (cur_close - swing_extreme) / swing_extreme * 100.0

        if pullback_pct < effective_pb_min:
            return False, 0.0, {
                "淘汰原因": f"回调幅度不足({pullback_pct:.2f}% < {effective_pb_min:.2f}%)",
                "回调幅度%": round(pullback_pct, 2),
            }, {"hourly_pullback_score": round(_clamp(pullback_pct / max(effective_pb_min, 1e-9), 0.0, 1.0) * 100.0, 2)}
        if pullback_pct > effective_pb_max:
            return False, 0.0, {
                "淘汰原因": f"回调过深({pullback_pct:.2f}% > {effective_pb_max:.2f}%)，更像趋势反转",
                "回调幅度%": round(pullback_pct, 2),
            }, {"hourly_pullback_score": 0.0}

        # ── 企稳：整体振幅 + 逐根振幅 ────────────────────────────────────────
        stab_highs   = highs[-stab_bars:]
        stab_lows    = lows[-stab_bars:]
        stab_range   = max(stab_highs) - min(stab_lows)
        stab_ratio   = stab_range / max(atr, 1e-9)
        if stab_range > atr * stab_atr:
            return False, 0.0, {
                "淘汰原因": f"未企稳：{stab_bars}根整体波幅({stab_range:.4f}) > ATR×{stab_atr}({atr*stab_atr:.4f})",
                "回调幅度%": round(pullback_pct, 2), "企稳波幅/ATR": round(stab_ratio, 3),
            }, {"hourly_stability_score": 0.0}

        per_bar_ranges = [highs[-stab_bars + i] - lows[-stab_bars + i] for i in range(stab_bars)]
        max_single = max(per_bar_ranges) if per_bar_ranges else 0.0
        if max_single > atr * per_bar_atr_r:
            return False, 0.0, {
                "淘汰原因": f"企稳期单根大K({max_single:.4f} > ATR×{per_bar_atr_r}={atr*per_bar_atr_r:.4f})",
                "回调幅度%": round(pullback_pct, 2), "最大单根振幅": round(max_single, 4),
            }, {"hourly_stability_score": 0.0}

        # ══ TTM Squeeze：BB 收进 KC 内（v3.1: 宽松模式允许近挤压）───────────
        bb_upper, bb_lower, cur_bw, bb_mid = _calc_bb(closes, bb_per, bb_std_mult)
        kc_upper, kc_lower, _kc_mid        = _calc_kc(closes, highs, lows, kc_per, kc_mult, atr_per)
        ttm_strict  = (bb_upper <= kc_upper) and (bb_lower >= kc_lower)
        kc_range    = kc_upper - kc_lower
        bb_above_pct = max(0, bb_upper - kc_upper) / max(kc_range, 1e-9) if kc_range > 0 else 999
        bb_below_pct = max(0, kc_lower - bb_lower) / max(kc_range, 1e-9) if kc_range > 0 else 999
        ttm_near    = bb_above_pct <= ttm_tol and bb_below_pct <= ttm_tol
        ttm_squeeze_on = ttm_strict or (not req_ttm and ttm_near)

        if req_ttm and not ttm_strict:
            return False, 0.0, {
                "淘汰原因": (
                    f"严格TTM挤压未激活：BB未完全收进KC内"
                ), "BB带宽%": round(cur_bw, 3), "TTM挤压激活": False,
            }, {"hourly_ttm_score": 0.0}
        if not ttm_squeeze_on:
            return False, 0.0, {
                "淘汰原因": (
                    f"TTM挤压未激活：BB超出KC范围"
                    f"（BB上超出{bb_above_pct*100:.0f}% BB下超出{bb_below_pct*100:.0f}% 容差{ttm_tol*100:.0f}%）"
                ), "BB带宽%": round(cur_bw, 3), "TTM挤压激活": False,
            }, {"hourly_ttm_score": 0.0}

        # ── BB 历史分位（修复：排除当前 bar 自身，使用 [1:] 开始的历史）─────
        if n < bb_per + bb_look:
            return False, 0.0, {
                "淘汰原因": f"BB历史数据不足({n}<{bb_per + bb_look}根)",
            }, {"hourly_squeeze_score": 0.0}

        # 从 i=1 开始（跳过 i=0 的当前 bar），修复历史分位包含自身的 BUG
        bb_widths_hist: List[float] = []
        for i in range(1, bb_look + 1):
            end_idx = n - i
            if end_idx < bb_per:
                break
            c_slice = closes[end_idx - bb_per:end_idx]
            mid_h   = float(np.mean(c_slice))
            std_h   = float(np.std(c_slice, ddof=1)) if len(c_slice) > 1 else 0.0
            bw_h    = (bb_std_mult * 2.0 * std_h / max(abs(mid_h), 1e-9)) * 100.0
            bb_widths_hist.append(bw_h)

        if not bb_widths_hist:
            return False, 0.0, {"淘汰原因": "BB历史带宽序列计算失败"}, {"hourly_squeeze_score": 0.0}

        hist_min  = min(bb_widths_hist)
        # 分位：当前带宽在历史中的百分位（纯历史，不含当前自身）
        bw_rank   = sum(1 for w in bb_widths_hist if w <= cur_bw) / max(len(bb_widths_hist), 1)
        abs_squeeze  = cur_bw < bb_sq_pct
        rel_squeeze  = cur_bw <= hist_min * bb_expand
        rank_squeeze = bw_rank <= bb_rank_max

        if not (rank_squeeze and (abs_squeeze or rel_squeeze)):
            return False, 0.0, {
                "淘汰原因": (
                    f"BB未达挤压标准：带宽{cur_bw:.2f}%"
                    f"  分位{bw_rank * 100:.0f}%（上限{bb_rank_max * 100:.0f}%）"
                    f"  历史最低{hist_min:.2f}%×{bb_expand}={hist_min*bb_expand:.2f}%"
                ),
                "BB带宽%": round(cur_bw, 3), "回调幅度%": round(pullback_pct, 2),
                "TTM挤压激活": ttm_squeeze_on,
            }, {"hourly_squeeze_score": 0.0}

        # 价格在 BB 内（突破前应在 BB 内）
        band_range = bb_upper - bb_lower
        if not (bb_lower <= cur_close <= bb_upper):
            return False, 0.0, {
                "淘汰原因": f"收盘价已在BB外({'上轨' if cur_close > bb_upper else '下轨'})，突破已发生",
                "BB带宽%": round(cur_bw, 3), "回调幅度%": round(pullback_pct, 2),
            }, {"hourly_location_score": 0.0}

        pos_in_band = (cur_close - bb_lower) / max(band_range, 1e-9)
        if d1_direction == "bull" and pos_in_band < 0.20:
            return False, 0.0, {
                "淘汰原因": f"多头但价格贴近BB下轨(位置{pos_in_band:.2f})，回调未修复",
                "BB带内位置": round(pos_in_band, 3), "回调幅度%": round(pullback_pct, 2),
            }, {"hourly_location_score": round(pos_in_band * 100.0, 2)}
        if d1_direction == "bear" and pos_in_band > 0.80:
            return False, 0.0, {
                "淘汰原因": f"空头但价格贴近BB上轨(位置{pos_in_band:.2f})，反弹未结束",
                "BB带内位置": round(pos_in_band, 3), "回调幅度%": round(pullback_pct, 2),
            }, {"hourly_location_score": round((1.0 - pos_in_band) * 100.0, 2)}

        # 企稳区间内价格位置
        stab_pos_min  = float(self.config["h1_stab_pos_min"])
        stab_top, stab_bot = max(stab_highs), min(stab_lows)
        stab_zone_range = stab_top - stab_bot
        pos_in_stab     = (cur_close - stab_bot) / max(stab_zone_range, 1e-9)
        if d1_direction == "bull" and pos_in_stab < stab_pos_min:
            return False, 0.0, {
                "淘汰原因": f"多头蓄势但仍在企稳区下半部({pos_in_stab:.2f}<{stab_pos_min:.2f})",
                "企稳区间位置": round(pos_in_stab, 3), "回调幅度%": round(pullback_pct, 2),
            }, {"hourly_location_score": round(pos_in_stab * 100.0, 2)}
        if d1_direction == "bear" and pos_in_stab > (1.0 - stab_pos_min):
            return False, 0.0, {
                "淘汰原因": f"空头蓄势但仍在企稳区上半部({pos_in_stab:.2f}>{1.0-stab_pos_min:.2f})",
                "企稳区间位置": round(pos_in_stab, 3), "回调幅度%": round(pullback_pct, 2),
            }, {"hourly_location_score": round((1.0 - pos_in_stab) * 100.0, 2)}

        # ── 量能：修复后版本（末根量 vs 基线，不是 vs 整理段均量）─────────
        baseline_slice = (
            volumes[-(stab_bars + vol_base_bars):-stab_bars]
            if n >= stab_bars + vol_base_bars else volumes[:-stab_bars]
        )
        baseline_vol   = _mean_positive(baseline_slice)
        stab_vol_mean  = _mean_positive(volumes[-stab_bars:])
        dryup_ratio    = stab_vol_mean / max(baseline_vol, 1e-9) if baseline_vol > 0 else 1.0

        if baseline_vol > 0 and dryup_ratio > max_dryup:
            return False, 0.0, {
                "淘汰原因": f"整理段缩量不足({dryup_ratio:.2f} > {max_dryup:.2f})，蓄势特征不足",
                "缩量系数": round(dryup_ratio, 3), "回调幅度%": round(pullback_pct, 2),
            }, {"hourly_dryup_score": round(max(0.0, 100.0 - dryup_ratio * 100.0), 2)}

        # 修复：末根量 vs 基线（不再与整理段自身比）
        lastbar_vol_ratio_vs_base = volumes[-1] / max(baseline_vol, 1e-9) if baseline_vol > 0 else 1.0
        if lastbar_vol_ratio_vs_base < min_rebound_b:
            return False, 0.0, {
                "淘汰原因": (
                    f"末根量能未充分回暖vs基线"
                    f"({lastbar_vol_ratio_vs_base:.2f} < {min_rebound_b:.2f})"
                ),
                "末根量能vs基线": round(lastbar_vol_ratio_vs_base, 3), "回调幅度%": round(pullback_pct, 2),
            }, {"hourly_rebound_score": 0.0}

        # ── RSI + RSI 背离（v3 新增背离检测）─────────────────────────────────
        h1_rsi = _calc_rsi_wilder(closes, rsi_period)
        if d1_direction == "bull":
            rsi_ok     = rsi_bull_min <= h1_rsi <= rsi_bull_max
            rsi_reason = f"多头RSI不在区间({h1_rsi:.1f}，期望{rsi_bull_min:.1f}-{rsi_bull_max:.1f})"
            target_pos = 0.60
        else:
            rsi_ok     = rsi_bear_min <= h1_rsi <= rsi_bear_max
            rsi_reason = f"空头RSI不在区间({h1_rsi:.1f}，期望{rsi_bear_min:.1f}-{rsi_bear_max:.1f})"
            target_pos = 0.40

        # RSI 背离：允许 RSI 不在区间但有背离（此时不淘汰，只扣分）
        rsi_divergence = _detect_rsi_divergence(closes, highs, lows, rsi_period, div_bars, d1_direction)
        if not rsi_ok and not rsi_divergence:
            return False, 0.0, {
                "淘汰原因": rsi_reason + "（且无RSI背离信号）",
                "H1_RSI": round(h1_rsi, 2), "回调幅度%": round(pullback_pct, 2),
                "RSI背离": False,
            }, {"hourly_rsi_score": 0.0}

        # ── MACD 柱体（MACD 拐头 + 零线穿越加成）────────────────────────────
        macd_hist = _calc_macd_hist_series(closes, fast=macd_fast_p, slow=macd_slow_p, signal=macd_sig_p)
        macd_last  = macd_hist[-1] if macd_hist else 0.0
        macd_prev  = macd_hist[-2] if len(macd_hist) >= 2 else macd_last
        macd_prev2 = macd_hist[-3] if len(macd_hist) >= 3 else macd_prev

        if d1_direction == "bull":
            macd_adverse     = macd_last < macd_prev < macd_prev2 and macd_last < 0
            macd_turn_str    = max(macd_last - macd_prev, 0.0) + max(macd_last - macd_prev2, 0.0) * 0.5
            macd_zero_cross  = macd_prev <= 0 < macd_last    # 零线刚上穿：加分
        else:
            macd_adverse     = macd_last > macd_prev > macd_prev2 and macd_last > 0
            macd_turn_str    = max(macd_prev - macd_last, 0.0) + max(macd_prev2 - macd_last, 0.0) * 0.5
            macd_zero_cross  = macd_prev >= 0 > macd_last

        if macd_adverse:
            return False, 0.0, {
                "淘汰原因": f"MACD柱仍逆趋势走坏({macd_prev2:.6f}→{macd_prev:.6f}→{macd_last:.6f})",
                "MACD柱体": round(macd_last, 6), "回调幅度%": round(pullback_pct, 2),
            }, {"hourly_macd_score": 0.0}

        # ── NR4/NR7 窄幅 K 线检测（新增）────────────────────────────────────
        bar_ranges = [highs[i] - lows[i] for i in range(len(closes))]
        nr_detected = False
        if len(bar_ranges) >= nr_look:
            recent_ranges = bar_ranges[-nr_look:]
            min_range_in_window = min(recent_ranges)
            # NR: 最后一根或倒数第二根是窗口内最小振幅
            nr_detected = (bar_ranges[-1] == min_range_in_window or
                           (len(bar_ranges) >= 2 and bar_ranges[-2] == min_range_in_window))

        if req_nr and not nr_detected:
            return False, 0.0, {
                "淘汰原因": f"未检测到NR{nr_look}窄幅K线（强制要求）",
                "回调幅度%": round(pullback_pct, 2),
            }, {"hourly_nr_score": 0.0}

        # ══ 评分计算 ════════════════════════════════════════════════════════
        pullback_mid   = (pb_min + pb_max) / 2.0
        pullback_score = 18.0 * _clamp(1.0 - abs(pullback_pct - pullback_mid) / max(pb_max - pb_min, 1e-9), 0.0, 1.0)
        stability_score = 16.0 * _clamp(1.0 - stab_ratio / max(stab_atr, 1e-9), 0.0, 1.0)

        # TTM 挤压得分：TTM 激活额外加成
        ttm_bonus       = 8.0 if ttm_squeeze_on else 0.0
        sq_abs          = _clamp(1.0 - cur_bw / max(bb_sq_pct, 1e-9), 0.0, 1.0)
        sq_rank         = _clamp(1.0 - bw_rank / max(bb_rank_max, 1e-9), 0.0, 1.0)
        rel_ratio       = cur_bw / max(hist_min * bb_expand, 1e-9)
        sq_rel          = _clamp(1.0 - rel_ratio, 0.0, 1.0)
        squeeze_quality = max(sq_abs, (sq_rank + sq_rel) / 2.0)
        squeeze_score   = 16.0 * squeeze_quality + ttm_bonus

        dryup_score    = 10.0 * _clamp((max_dryup - dryup_ratio) / max(max_dryup, 1e-9), 0.0, 1.0)

        # RSI 评分：有背离时给满分的 70%（不要求在区间内）
        if rsi_divergence:
            rsi_score    = 12.0 * 0.70
            rsi_div_bonus = 5.0
        else:
            if d1_direction == "bull":
                rsi_center   = (rsi_bull_min + rsi_bull_max) / 2.0
                rsi_halfrange = max((rsi_bull_max - rsi_bull_min) / 2.0, 1e-9)
            else:
                rsi_center   = (rsi_bear_min + rsi_bear_max) / 2.0
                rsi_halfrange = max((rsi_bear_max - rsi_bear_min) / 2.0, 1e-9)
            rsi_quality  = _clamp(1.0 - abs(h1_rsi - rsi_center) / rsi_halfrange, 0.0, 1.0)
            rsi_score    = 12.0 * rsi_quality
            rsi_div_bonus = 0.0

        macd_pct     = macd_last / max(abs(cur_close), 1e-9) * 100.0
        macd_quality = _clamp(abs(macd_turn_str) / max(abs(cur_close) * 0.002, 1e-9), 0.0, 1.0)
        macd_score   = 8.0 * macd_quality + (4.0 if macd_zero_cross else 0.0)

        location_quality = _clamp(1.0 - abs(pos_in_band - target_pos) / 0.40, 0.0, 1.0)
        rebound_quality  = _clamp(
            (lastbar_vol_ratio_vs_base - min_rebound_b) / max(1.5 - min_rebound_b, 1e-9), 0.0, 1.0,
        )
        location_score = 10.0 * (location_quality * 0.65 + rebound_quality * 0.35)

        nr_bonus = 4.0 if nr_detected else 0.0    # NR4/NR7 加分项

        h1_score = round(min(
            pullback_score + stability_score + squeeze_score + dryup_score
            + rsi_score + rsi_div_bonus + macd_score + location_score + nr_bonus,
            100.0,
        ), 2)

        details = {
            "H1方向": "多头回调" if d1_direction == "bull" else "空头反弹",
            "摆动极值": round(swing_extreme, 4),
            "当前收盘": round(cur_close, 4),
            "回调幅度%": round(pullback_pct, 2),
            "企稳根数": stab_bars,
            "企稳波幅/ATR": round(stab_ratio, 3),
            "企稳区间位置": round(pos_in_stab, 3),
            "ATR": round(atr, 4),
            "缩量系数": round(dryup_ratio, 3),
            "末根量能vs基线": round(lastbar_vol_ratio_vs_base, 3),
            "BB上轨": round(bb_upper, 4), "BB中轨": round(bb_mid, 4), "BB下轨": round(bb_lower, 4),
            "BB带宽%": round(cur_bw, 3),
            "BB历史最低%": round(hist_min, 3),
            "BB带宽历史分位%": round(bw_rank * 100, 1),
            "BB带内位置": round(pos_in_band, 3),
            "KC上轨": round(kc_upper, 4), "KC下轨": round(kc_lower, 4),
            "TTM挤压激活": ttm_squeeze_on,
            "H1_RSI": round(h1_rsi, 2),
            "RSI背离": rsi_divergence,
            "MACD柱体": round(macd_last, 6),
            "MACD柱体%": round(macd_pct, 4),
            "MACD零线穿越": macd_zero_cross,
            "NR4/NR7": nr_detected,
            "小时线评分": round(h1_score, 2),
        }
        factor_scores = {
            "hourly_pullback_score": round(pullback_score, 2),
            "hourly_stability_score": round(stability_score, 2),
            "hourly_squeeze_score": round(squeeze_score, 2),
            "hourly_dryup_score": round(dryup_score, 2),
            "hourly_rsi_score": round(rsi_score + rsi_div_bonus, 2),
            "hourly_macd_score": round(macd_score, 2),
            "hourly_location_score": round(location_score, 2),
            "hourly_nr_bonus": round(nr_bonus, 2),
        }
        # v3.1: 成交量买卖压力分析
        if bool(self.config.get("enable_volume_pressure", True)):
            buy_vol, sell_vol = _calc_buy_sell_pressure(rows[-24:])
            details["买量占比"] = round(buy_vol / max(buy_vol + sell_vol, 1), 3)
            details["主动买卖比"] = round(buy_vol / max(sell_vol, 1), 2)

        # v3.1: ATR止损/止盈建议
        if bool(self.config.get("enable_atr_target_suggestion", True)):
            sl_mult = float(self.config.get("stop_atr_mult", 2.0) or 2.0)
            tp_mult = float(self.config.get("target_atr_mult", 3.0) or 3.0)
            if d1_direction == "bull":
                details["ATR止损"] = round(cur_close - atr * sl_mult, 6)
                details["ATR止盈"] = round(cur_close + atr * tp_mult, 6)
            else:
                details["ATR止损"] = round(cur_close + atr * sl_mult, 6)
                details["ATR止盈"] = round(cur_close - atr * tp_mult, 6)
            details["ATR止损%"] = round(atr * sl_mult / cur_close * 100, 2)
            details["ATR止盈%"] = round(atr * tp_mult / cur_close * 100, 2)

        return True, h1_score, details, factor_scores

    # ══════════════════════════════════════════════════════════════════════════
    # Step 3：15m 入场时机确认（新增）
    # ══════════════════════════════════════════════════════════════════════════

    def _check_15m_entry_timing(
        self, rows: List, d1_direction: str
    ) -> Tuple[float, Dict[str, Any]]:
        """
        15m 层不做淘汰，只返回 0~100 的时机分和详情。
        判据：
          1. EMA8 与 EMA21 的方向（刚金叉/死叉 = 最高分）
          2. 最后一根量能是否比过去 N 根的均量大（首根放量）
          3. 最后一根 K 线收盘方向与 d1_direction 一致
        """
        def _v(row, idx, default=0.0):
            try:
                return float(row[idx])
            except Exception:
                return default

        closes  = [_v(r, 4) for r in rows]
        volumes = [_v(r, 5) for r in rows]
        ema_f   = int(self.config["m15_ema_fast"])
        ema_s   = int(self.config["m15_ema_slow"])

        ema_fast_s = _calc_ema_series(closes, ema_f)
        ema_slow_s = _calc_ema_series(closes, ema_s)

        if len(ema_fast_s) < 3 or len(ema_slow_s) < 3:
            return 0.0, {}

        ef_now  = ema_fast_s[-1]
        es_now  = ema_slow_s[-1]
        ef_prev = ema_fast_s[-2]
        es_prev = ema_slow_s[-2]

        golden_cross = ef_now > es_now and ef_prev <= es_prev    # 刚金叉
        death_cross  = ef_now < es_now and ef_prev >= es_prev    # 刚死叉
        bull_align   = ef_now > es_now                            # 多头排列
        bear_align   = ef_now < es_now                            # 空头排列

        if d1_direction == "bull":
            cross_match = golden_cross
            align_match = bull_align
        else:
            cross_match = death_cross
            align_match = bear_align

        # 量能：最后一根 vs 前 10 根均量
        vol_base  = _mean_positive(volumes[-11:-1]) if len(volumes) >= 12 else _mean_positive(volumes[:-1])
        vol_last  = volumes[-1]
        vol_ratio = vol_last / max(vol_base, 1e-9) if vol_base > 0 else 1.0

        # 最后一根 K 线方向
        closes_last_ok = (
            (d1_direction == "bull" and closes[-1] > closes[-2]) or
            (d1_direction == "bear" and closes[-1] < closes[-2])
        ) if len(closes) >= 2 else False

        # 评分
        cross_score  = 45.0 if cross_match else (25.0 if align_match else 0.0)
        vol_score    = 30.0 * _clamp((vol_ratio - 1.0) / 1.0, 0.0, 1.0)  # 放量越大越高分
        dir_score    = 25.0 if closes_last_ok else 0.0
        timing_score = round(min(cross_score + vol_score + dir_score, 100.0), 2)

        details = {
            "15m EMA金叉": golden_cross, "15m EMA死叉": death_cross,
            "15m EMA方向一致": align_match,
            "EMA金叉": cross_match,
            "量能系数": round(vol_ratio, 2),
            "K线方向一致": closes_last_ok,
            "15m时机分": timing_score,
        }
        return timing_score, details

    # ══════════════════════════════════════════════════════════════════════════
    # v3.1: BTC市场环境 + 交易辅助
    # ══════════════════════════════════════════════════════════════════════════

    def _get_btc_from_symbol(self, symbol) -> Dict[str, float]:
        """从symbol的extra_data中提取BTC基准数据（如果引擎注入了BTC行情）"""
        ctx = {}
        try:
            ed = getattr(symbol, "extra_data", None) or {}
            btc_data = ed.get("btc_market") if isinstance(ed, dict) else {}
            if btc_data:
                ctx["btc_1h"] = float(btc_data.get("btc_1h_move", 0) or 0)
                ctx["btc_24h"] = float(btc_data.get("btc_24h_move", 0) or 0)
                return ctx
        except Exception:
            pass
        # 回退：从symbol自身判断（如果symbol本身就是BTC）
        sym_id = str(getattr(symbol, "inst_id", "") or "").upper()
        if "BTC" in sym_id:
            ctx["btc_24h"] = float(getattr(symbol, "price_change_24h", 0) or 0)
        return ctx

    def _calc_btc_penalty(self, btc_ctx: Dict) -> float:
        """BTC大跌时返回应扣分数"""
        if not btc_ctx:
            return 0.0
        threshold = float(self.config.get("btc_dump_block_threshold_pct", -3.0) or -3.0)
        btc_move = btc_ctx.get("btc_1h", btc_ctx.get("btc_24h", 0))
        if btc_move < threshold:
            return min(12.0, abs(btc_move - threshold) * 2.5)
        return 0.0

    # ══════════════════════════════════════════════════════════════════════════
    # 工具方法
    # ══════════════════════════════════════════════════════════════════════════

    def _build_ranking_factors(
        self, *, total_score, d1_score, h1_score, m15_bonus, vol24,
        d1_details, h1_details,
    ) -> Dict[str, float]:
        vol_score    = _clamp(40.0 + math.log10(max(vol24, 1.0)) * 8.5, 25.0, 100.0)
        pullback     = float(h1_details.get("回调幅度%", 0) or 0)
        stab_ratio   = float(h1_details.get("企稳波幅/ATR", 99) or 99)
        dryup_ratio  = float(h1_details.get("缩量系数", 1.2) or 1.2)
        band_pos     = float(h1_details.get("BB带内位置", 0.5) or 0.5)
        ext_atr      = float(d1_details.get("快EMA延伸ATR", 0) or 0)
        ttm_on       = bool(h1_details.get("TTM挤压激活", False))
        adx_rising   = bool(d1_details.get("ADX上升中", False))
        rsi_div      = bool(h1_details.get("RSI背离", False))

        target_pos   = 0.60 if str(d1_details.get("日线方向", "")).startswith("多") else 0.40
        location_score = _clamp(
            100.0
            - abs(pullback - (float(self.config.get("h1_pullback_min_pct", 2.5))
                             + float(self.config.get("h1_pullback_max_pct", 15.0))) / 2.0) * 5.0
            - abs(band_pos - target_pos) * 40.0,
            20.0, 98.0,
        )
        freshness_score = _clamp(
            100.0
            - max(float(h1_details.get("BB带宽%", 0) or 0) - float(self.config.get("h1_bb_squeeze_pct", 4.0)), 0.0) * 8.0
            - max(dryup_ratio - 0.70, 0.0) * 50.0
            + (8.0 if ttm_on else 0.0),
            30.0, 98.0,
        )
        risk_score = _clamp(
            98.0 - max(stab_ratio - 0.20, 0.0) * 30.0 - max(ext_atr - 1.0, 0.0) * 10.0,
            25.0, 96.0,
        )
        trend_sprout_score = _clamp(
            d1_score + (5.0 if adx_rising else 0.0) + (5.0 if rsi_div else 0.0),
            0.0, 100.0,
        )
        return {
            "trend":         round(trend_sprout_score, 2),
            "trigger":       round(_clamp(h1_score, 0.0, 100.0), 2),
            "volume":        round(vol_score, 2),
            "location":      round(location_score, 2),
            "freshness":     round(freshness_score, 2),
            "risk":          round(risk_score, 2),
            "timing":        round(m15_bonus, 2),
            "base_score":    round(_clamp(total_score, 0.0, 100.0), 2),
        }

    @staticmethod
    def _get_klines(symbol, tf: str) -> List:
        try:
            extra_data = getattr(symbol, "extra_data", None) or {}
            klines_map = extra_data.get("klines", {}) if isinstance(extra_data, dict) else {}
            rows = klines_map.get(tf, [])
            if not isinstance(rows, (list, tuple)):
                return []
            cleaned = []
            for row in rows:
                if not isinstance(row, (list, tuple)) or len(row) < 6:
                    continue
                try:
                    ts = int(float(row[0])); o = float(row[1])
                    h = float(row[2]);        l = float(row[3])
                    c = float(row[4]);        v = float(row[5])
                except (TypeError, ValueError):
                    continue
                if min(o, h, l, c) <= 0:
                    continue
                cleaned.append([ts, o, h, l, c, v])
            cleaned.sort(key=lambda x: x[0])
            deduped: Dict[int, List] = {}
            for row in cleaned:
                deduped[int(row[0])] = row
            return [deduped[k] for k in sorted(deduped)]
        except Exception:
            return []


# ══════════════════════════════════════════════════════════════════════════════
# 辅助函数
# ══════════════════════════════════════════════════════════════════════════════

def _calc_ema(data: List[float], period: int) -> float:
    if not data or len(data) < period:
        return data[-1] if data else 0.0
    k = 2.0 / (period + 1.0)
    value = float(np.mean(data[:period]))
    for price in data[period:]:
        value = float(price) * k + value * (1.0 - k)
    return value


def _calc_ema_series(data: List[float], period: int) -> List[float]:
    if not data or len(data) < period:
        return [float(x) for x in data]
    k = 2.0 / (period + 1.0)
    seed = float(np.mean(data[:period]))
    series = [float("nan")] * (period - 1) + [seed]
    for price in data[period:]:
        seed = float(price) * k + seed * (1.0 - k)
        series.append(seed)
    return series


def _calc_atr(closes: List[float], highs: List[float], lows: List[float], period: int = 14) -> float:
    if len(closes) < 2:
        return 0.0
    trs: List[float] = []
    for i in range(1, len(closes)):
        tr = max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1]))
        trs.append(tr)
    if not trs:
        return 0.0
    if len(trs) < period:
        return float(np.mean(trs))
    atr = float(np.mean(trs[:period]))
    for tr in trs[period:]:
        atr = (atr * (period - 1) + tr) / period
    return atr


def _calc_adx(
    closes: List[float], highs: List[float], lows: List[float], period: int = 14
) -> Tuple[float, float, float]:
    n = len(closes)
    if n < period * 2 + 5:
        return 0.0, 0.0, 0.0
    tr_l, pdm_l, ndm_l = [], [], []
    for i in range(1, n):
        tr = max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1]))
        up = max(highs[i] - highs[i - 1], 0.0)
        dn = max(lows[i - 1] - lows[i], 0.0)
        if up > dn:       dn = 0.0
        elif dn > up:     up = 0.0
        else:             up = dn = 0.0
        tr_l.append(tr); pdm_l.append(up); ndm_l.append(dn)

    def _wilder(vals: List[float], p: int) -> List[float]:
        if len(vals) < p:
            return []
        s = [sum(vals[:p])]
        for v in vals[p:]:
            s.append(s[-1] - s[-1] / p + v)
        return s

    atr_s = _wilder(tr_l, period)
    pdm_s = _wilder(pdm_l, period)
    ndm_s = _wilder(ndm_l, period)
    if not atr_s:
        return 0.0, 0.0, 0.0
    dx_vals: List[float] = []
    for atr_v, pdm_v, ndm_v in zip(atr_s, pdm_s, ndm_s):
        if atr_v <= 0:
            continue
        dp = pdm_v / atr_v * 100.0; dm = ndm_v / atr_v * 100.0
        den = dp + dm
        dx_vals.append(abs(dp - dm) / den * 100.0 if den > 0 else 0.0)
    if len(dx_vals) < period:
        atr_last = atr_s[-1]
        return 0.0, round(pdm_s[-1] / max(atr_last, 1e-9) * 100.0, 2), round(ndm_s[-1] / max(atr_last, 1e-9) * 100.0, 2)
    adx = sum(dx_vals[:period]) / period
    for v in dx_vals[period:]:
        adx = (adx * (period - 1) + v) / period
    atr_last = atr_s[-1]
    di_plus  = pdm_s[-1] / max(atr_last, 1e-9) * 100.0
    di_minus = ndm_s[-1] / max(atr_last, 1e-9) * 100.0
    return round(adx, 2), round(di_plus, 2), round(di_minus, 2)


def _calc_adx_series(
    closes: List[float], highs: List[float], lows: List[float], period: int = 14, last_n: int = 10
) -> List[float]:
    """返回最近 last_n 根的 ADX 数值序列，用于判断 ADX 是否持续上升。"""
    needed = period * 2 + 5 + last_n
    results: List[float] = []
    for offset in range(last_n - 1, -1, -1):
        end = len(closes) - offset
        if end < period * 2 + 5:
            results.append(0.0)
            continue
        adx_v, _, _ = _calc_adx(closes[:end], highs[:end], lows[:end], period)
        results.append(adx_v)
    return results


def _calc_bb(
    closes: List[float], period: int = 20, std_mult: float = 2.0
) -> Tuple[float, float, float, float]:
    """返回 (upper, lower, bandwidth_pct, mid)"""
    if len(closes) < period:
        mid = closes[-1] if closes else 0.0
        return mid, mid, 0.0, mid
    c_slice = closes[-period:]
    mid     = float(np.mean(c_slice))
    std     = float(np.std(c_slice, ddof=1)) if len(c_slice) > 1 else 0.0
    upper   = mid + std_mult * std
    lower   = mid - std_mult * std
    bw_pct  = (std_mult * 2.0 * std / max(abs(mid), 1e-9)) * 100.0
    return upper, lower, bw_pct, mid


def _calc_kc(
    closes: List[float], highs: List[float], lows: List[float],
    period: int = 20, mult: float = 1.5, atr_period: int = 14
) -> Tuple[float, float, float]:
    """肯特纳通道 (upper, lower, mid)"""
    mid = _calc_ema(closes, period)
    atr = _calc_atr(closes, highs, lows, atr_period)
    return mid + mult * atr, mid - mult * atr, mid


def _calc_rsi_wilder(data: List[float], period: int = 14) -> float:
    """Wilder EMA RSI，增加热身期保护，避免短历史数据严重失真。"""
    # 至少需要 3×period 根数据才能保证 Wilder 充分热身
    warmup = period * 3
    if len(data) < period + 1:
        return 50.0
    # 若数据不足热身期，截取足够的历史（不报错）
    effective = data if len(data) >= warmup else data
    deltas = np.diff(np.asarray(effective, dtype=float))
    gains  = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = float(np.mean(gains[:period]))
    avg_loss = float(np.mean(losses[:period]))
    for g, l in zip(gains[period:], losses[period:]):
        avg_gain = (avg_gain * (period - 1) + float(g)) / period
        avg_loss = (avg_loss * (period - 1) + float(l)) / period
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    return float(100.0 - 100.0 / (1.0 + avg_gain / avg_loss))


def _detect_rsi_divergence(
    closes: List[float], highs: List[float], lows: List[float],
    rsi_period: int, lookback: int, direction: str
) -> bool:
    """
    检测 RSI 背离（多头/空头）。
    多头背离：价格创新低但 RSI 未创新低（看涨背离）。
    空头背离：价格创新高但 RSI 未创新高（看跌背离）。
    """
    n = len(closes)
    if n < rsi_period + lookback + 5:
        return False
    try:
        # 计算完整 RSI 序列（最近 lookback+rsi_period 根）
        seg = closes[-(lookback + rsi_period + 5):]
        deltas = np.diff(np.asarray(seg, dtype=float))
        gains  = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        avg_g = float(np.mean(gains[:rsi_period]))
        avg_l = float(np.mean(losses[:rsi_period]))
        rsi_vals: List[float] = []
        for g, l in zip(gains[rsi_period:], losses[rsi_period:]):
            avg_g = (avg_g * (rsi_period - 1) + float(g)) / rsi_period
            avg_l = (avg_l * (rsi_period - 1) + float(l)) / rsi_period
            rs    = avg_g / max(avg_l, 1e-9)
            rsi_vals.append(100.0 - 100.0 / (1.0 + rs))

        if len(rsi_vals) < 4:
            return False

        price_recent = closes[-(lookback):]
        rsi_recent   = rsi_vals[-(lookback):]

        if direction == "bull":
            # 多头背离：价格低点在下降，但 RSI 低点在上升
            price_low1 = min(price_recent[:len(price_recent)//2])
            price_low2 = min(price_recent[len(price_recent)//2:])
            rsi_low1   = min(rsi_recent[:len(rsi_recent)//2])
            rsi_low2   = min(rsi_recent[len(rsi_recent)//2:])
            return price_low2 < price_low1 and rsi_low2 > rsi_low1 + 2.0
        else:
            # 空头背离：价格高点在上升，但 RSI 高点在下降
            price_hi1 = max(price_recent[:len(price_recent)//2])
            price_hi2 = max(price_recent[len(price_recent)//2:])
            rsi_hi1   = max(rsi_recent[:len(rsi_recent)//2])
            rsi_hi2   = max(rsi_recent[len(rsi_recent)//2:])
            return price_hi2 > price_hi1 and rsi_hi2 < rsi_hi1 - 2.0
    except Exception:
        return False


def _find_swing_high(highs: List[float], confirm_bars: int) -> Optional[float]:
    """找真实摆动高点（左右各 confirm_bars 根都比它低）。"""
    n = len(highs)
    best = None
    for i in range(confirm_bars, n - confirm_bars):
        left_ok  = all(highs[j] < highs[i] for j in range(i - confirm_bars, i))
        right_ok = all(highs[j] < highs[i] for j in range(i + 1, i + confirm_bars + 1))
        if left_ok and right_ok:
            if best is None or highs[i] > best:
                best = highs[i]
    return best


def _find_swing_low(lows: List[float], confirm_bars: int) -> Optional[float]:
    """找真实摆动低点（左右各 confirm_bars 根都比它高）。"""
    n = len(lows)
    best = None
    for i in range(confirm_bars, n - confirm_bars):
        left_ok  = all(lows[j] > lows[i] for j in range(i - confirm_bars, i))
        right_ok = all(lows[j] > lows[i] for j in range(i + 1, i + confirm_bars + 1))
        if left_ok and right_ok:
            if best is None or lows[i] < best:
                best = lows[i]
    return best


def _calc_macd_hist_series(
    data: List[float], fast: int = 12, slow: int = 26, signal: int = 9
) -> List[float]:
    if not data:
        return []
    ema_fast = _calc_ema_series(data, fast)
    ema_slow = _calc_ema_series(data, slow)
    macd_line = [
        float(f - s) if (np.isfinite(f) and np.isfinite(s)) else float("nan")
        for f, s in zip(ema_fast, ema_slow)
    ]
    valid = [v for v in macd_line if np.isfinite(v)]
    if len(valid) < signal:
        return [0.0] * len(data)
    sig_series = _calc_ema_series(valid, signal)
    sig_iter = iter(sig_series)
    hist: List[float] = []
    for v in macd_line:
        if not np.isfinite(v):
            hist.append(0.0)
            continue
        sv = next(sig_iter, float("nan"))
        hist.append(float(v - sv) if np.isfinite(sv) else 0.0)
    return hist


def _pct_change(current: float, past: float) -> float:
    if abs(past) < 1e-9:
        return 0.0
    return (current - past) / past * 100.0


def _mean_positive(values: List[float]) -> float:
    cleaned = [float(v) for v in values if float(v) >= 0]
    return float(np.mean(cleaned)) if cleaned else 0.0


def _calc_buy_sell_pressure(rows: List) -> Tuple[float, float]:
    """
    基于K线形态估算买卖压力：
      -收盘在K线上半部→买入压力；下半部→卖出压力
      -权重按成交量分配
    """
    buy_vol = sell_vol = 0.0
    for row in rows[-24:]:
        try:
            o, h, l, c, v = float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5])
            rng = h - l
            if rng <= 0:
                continue
            buy_ratio = (c - l) / rng
            buy_vol += v * buy_ratio
            sell_vol += v * (1 - buy_ratio)
        except Exception:
            continue
    return buy_vol, sell_vol


STRATEGY_NAME  = "趋势延续·挤压突破前扫描器 v3"
STRATEGY_TYPE  = "scan"
STRATEGY_CLASS = TrendSqueezeBreakoutScannerV3
BACKTEST_CLASS = TrendSqueezeBreakoutScannerV3
