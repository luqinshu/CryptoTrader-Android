#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI截面五引擎组合扫描器 v2

一次扫描同时运行五个互补策略并合成多引擎共振：
  1. 截面多因子加密货币扫描4_23_v3   — 全市场截面 z-score 排序 + 链上 + 正交化 + IC半衰期
  2. AI因子挖掘加密货币扫描策略4_23_v3 — Chain-of-Alpha: LLM/社交/链上/技术多因子
  3. DRL元学习小时趋势启动扫描策略4_23_v2 — DQN/A2C/SAC + 元学习 + 小时趋势启动
  4. XGBoost截面排序策略4.27_v2        — 梯度提升排序
  5. AI订单流动量突破组合策略4.27_v2    — 订单流异常检测

v1 → v2 改动
─────────────────────────────────────────────────────
1. CHILDREN 更新到最新文件名 (_23_v3 / _23_v2)。
2. 文件查找逻辑优先 __file__ 同级，再搜索 strategies/ 子目录，
   避免路径找不到而跳过所有子策略。
3. CONFIG_SCHEMA 补全三个子策略新增的时效性参数
   (max_h1_trend_age / h1_trend_age_penalty / max_m3_staleness_bars /
    m3_freshness_penalty / bonus_freshness_score)，使组合层可以
   统一控制并正确透传给子策略。
4. _child_config 统一透传时效性参数到各子策略。
5. 修复 v1 中 _sync_config 每次 on_bar 重建 config 导致截面子策略
   动态 IC 权重缓存被清空的问题（与截面策略 v3.1 的修复一致）。
6. 截面子策略加速模式增加 require_m3_pullback_confirmation=False
   选项（快速扫描时可关闭，精确扫描时保留）。
7. 共振结果新增 h1_trend_age / m3_staleness 汇总字段。
─────────────────────────────────────────────────────
"""

from __future__ import annotations

import importlib.util
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

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


CONFIG_SCHEMA = {
    # ── 基础 ──
    "min_volume_24h":               {"type": "float", "default": 8_000_000.0, "label": "组合最小24H成交额"},
    "min_score":                    {"type": "float", "default": 70.0,        "label": "子策略最低扫描分数"},
    "backtest_min_score":           {"type": "float", "default": 60.0,        "label": "回测最低入场分数"},
    "top_n":                        {"type": "int",   "default": 12,          "label": "组合最多输出"},
    "top_n_per_strategy":           {"type": "int",   "default": 6,           "label": "每个子策略最多输出"},
    "include_individual_results":   {"type": "bool",  "default": True,        "label": "保留子策略单独结果"},
    "include_consensus_results":    {"type": "bool",  "default": True,        "label": "输出多策略共振结果"},
    "dedupe_by_symbol":             {"type": "bool",  "default": True,        "label": "按交易对去重仅保留最高分"},
    "allow_short":                  {"type": "bool",  "default": True,        "label": "允许空头"},
    "max_atr_pct":                  {"type": "float", "default": 8.0,         "label": "最大ATR%"},
    # ── 共振 ──
    "min_consensus_engines":        {"type": "int",   "default": 2,           "label": "最少共振引擎数"},
    "consensus_bonus":              {"type": "float", "default": 8.5,         "label": "双引擎共振加分"},
    "triple_consensus_bonus":       {"type": "float", "default": 13.0,        "label": "三引擎共振加分"},
    "direction_conflict_penalty":   {"type": "float", "default": 8.0,         "label": "方向冲突降分"},
    # ── AI因子挖掘子策略 ──
    "min_abs_edge":                 {"type": "float", "default": 0.24,        "label": "AI策略最小优势"},
    "use_dynamic_ic_weights":       {"type": "bool",  "default": True,        "label": "启用动态IC权重"},
    "ic_weight_blend":              {"type": "float", "default": 0.55,        "label": "IC权重混合比例"},
    "enable_mfin_interactions":     {"type": "bool",  "default": True,        "label": "AI策略启用交互项"},
    "enable_llm_factors":           {"type": "bool",  "default": True,        "label": "AI策略启用LLM/社交因子"},
    "enable_on_chain":              {"type": "bool",  "default": True,        "label": "启用链上因子"},
    # ── 截面多因子子策略 ──
    "use_orthogonalization":        {"type": "bool",  "default": True,        "label": "截面策略启用因子正交化"},
    # ── 早启动因子（DRL+AI共用）──
    "enable_early_trend_factors":   {"type": "bool",  "default": True,        "label": "启用小时级早启动/转折因子"},
    "early_trend_min_trigger":      {"type": "float", "default": 0.18,        "label": "早启动最低触发强度"},
    # ── 3m 回调企稳 ──
    "require_m3_pullback_confirmation": {"type": "bool",  "default": True,   "label": "要求3分钟回调企稳续势"},
    "m3_pullback_min_pct":          {"type": "float", "default": 0.20,        "label": "3分钟最小回调幅度%"},
    "m3_pullback_max_pct":          {"type": "float", "default": 3.00,        "label": "3分钟最大回调幅度%"},
    "m3_stabilization_bars":        {"type": "int",   "default": 2,           "label": "3分钟企稳确认根数"},
    "require_m3_freshness":         {"type": "bool",  "default": False,       "label": "必须通过3m时效性检查"},
    "m3_min_impulse_pct":           {"type": "float", "default": 0.65,        "label": "3m最小原趋势脉冲%"},
    "vol_continuation_min_ratio":   {"type": "float", "default": 0.78,        "label": "企稳量能续航最低比例"},
    # ── v3.2 微观结构 ──
    "enable_atr_squeeze_check":     {"type": "bool",  "default": True,        "label": "启用ATR收缩检测"},
    "atr_squeeze_ratio":            {"type": "float", "default": 0.55,        "label": "ATR收缩比例"},
    "enable_volume_delta_check":    {"type": "bool",  "default": True,        "label": "启用买卖力量检测"},
    "volume_delta_min_ratio":       {"type": "float", "default": 1.15,        "label": "买卖量最低比值"},
    "enable_vwap_alignment_check":  {"type": "bool",  "default": True,        "label": "启用VWAP对齐检测"},
    # ── 时效性过滤（v2 新增，三子策略共用）──
    "max_h1_trend_age":             {"type": "int",   "default": 12,          "label": "1H趋势最大延续根数（超过则惩罚）"},
    "h1_trend_age_penalty":         {"type": "float", "default": 8.0,         "label": "趋势过老分数惩罚"},
    "max_m3_staleness_bars":        {"type": "int",   "default": 15,          "label": "3m回调最大时效根数"},
    "m3_freshness_penalty":         {"type": "float", "default": 6.0,         "label": "3m回调过旧分数惩罚"},
    "bonus_freshness_score":        {"type": "float", "default": 3.0,         "label": "两项时效均通过时加分"},
    # ── 硬性3m回调企稳过滤 ──
    "m3_hard_filter":               {"type": "bool",  "default": True,        "label": "硬性3m回调企稳过滤（不通过则丢弃）"},
    "m3_soft_filter_mode":          {"type": "bool",  "default": False,       "label": "软过滤模式（不过滤仅降权，可发现更多机会）"},
    "m3_impulse_lookback_bars":     {"type": "int",   "default": 15,          "label": "3m回调检测回溯K线数"},
    "m3_no_break_tolerance_pct":    {"type": "float", "default": 0.30,        "label": "回调不得跌破突破点的容差%"},
    # ── 硬性小时线趋势延续过滤 ──
    "h1_trend_hard_filter":         {"type": "bool",  "default": True,        "label": "硬性1H趋势延续过滤（EMA金叉/死叉判断）"},
    "h1_ema_fast":                  {"type": "int",   "default": 12,          "label": "1H快线EMA周期"},
    "h1_ema_slow":                  {"type": "int",   "default": 26,          "label": "1H慢线EMA周期"},
    # ── 性能 / 模式 ──
    "position_size":                {"type": "float", "default": 0.02,        "label": "回测仓位比例"},
    "use_pilot_add_system":         {"type": "bool",  "default": True,        "label": "启用1%试仓+10%加仓系统"},
    "mode": {
        "type": "select", "default": "normal", "label": "扫描模式",
        "options": [{"label": "常规", "value": "normal"}, {"label": "超严", "value": "ultra"}],
    },
    "ultra_strict_mode":            {"type": "bool",  "default": False,       "label": "超严模式(目标6-9条)"},
    "ultra_target_top_n":           {"type": "int",   "default": 9,           "label": "超严模式最大输出"},
    "parallel_child_engines":       {"type": "bool",  "default": False,       "label": "并行运行子引擎"},
    "fast_scan_mode":               {"type": "bool",  "default": False,       "label": "启用快速候选池"},
    "max_scan_symbols":             {"type": "int",   "default": 200,         "label": "快速候选池上限"},
    "drl_candidate_cap":            {"type": "int",   "default": 180,         "label": "DRL子引擎候选池上限"},
    "profile_child_timing":         {"type": "bool",  "default": True,        "label": "输出子引擎耗时"},
    "accelerate_cross_section_child": {"type": "bool","default": True,        "label": "组合内启用截面子引擎加速"},
    "accel_disable_m3":             {"type": "bool",  "default": False,       "label": "加速模式下关闭3m企稳检查"},
    # v4.0 新增
    "enable_state_conditional_weights": {"type":"bool","default":True,       "label": "启用市场状态条件权重"},
    # v4.1 新增：评分归一化 + 引擎动态权重 + 信号持续时长 + 绩效追踪
    "enable_score_normalization":     {"type": "bool",  "default": True,        "label": "启用跨引擎评分z-score归一化"},
    "enable_engine_track_record":     {"type": "bool",  "default": True,        "label": "启用引擎历史胜率动态权重"},
    "engine_weight_decay":            {"type": "float", "default": 0.85,        "label": "引擎权重衰减因子(0.7-1)"},
    "min_signal_persistence_bars":   {"type": "int",   "default": 0,           "label": "信号最少持续3m根数(防闪信号，0=关闭，扫描器重建时不应使用)"},
    "prefilter_3m_before_consensus": {"type": "bool",  "default": False,       "label": "3m回调预过滤(子引擎结果先在组合层过3m再参与共识，与m3_hard_filter双重过滤，建议关闭)"},
    "max_track_record_entries":      {"type": "int",   "default": 50,          "label": "绩效追踪最大记录数"},
    # v4.2 交易员视角新增 ──────────────────────────────────────────────
    "enable_btc_correlation_filter": {"type": "bool",  "default": True,        "label": "启用BTC相关性过滤(BTC暴跌时降权山寨多头)"},
    "btc_dump_threshold_pct":        {"type": "float", "default": -2.5,        "label": "BTC 1H跌幅阈值%(低于此值降权山寨多头)"},
    "btc_dump_penalty":              {"type": "float", "default": 12.0,        "label": "BTC暴跌时山寨多头降分"},
    "enable_funding_filter":         {"type": "bool",  "default": True,        "label": "启用资金费率极端检测"},
    "funding_extreme_threshold":     {"type": "float", "default": 0.10,        "label": "资金费率极端阈值%(>此值降权多头)"},
    "funding_penalty":               {"type": "float", "default": 8.0,         "label": "极端费率降分"},
    "enable_confluence_scoring":    {"type": "bool",  "default": True,        "label": "启用1H/4H/日线多周期共振加分"},
    "confluence_bonus_max":          {"type": "float", "default": 6.0,         "label": "多周期共振最高加分"},
    "enable_atr_stop_suggestion":    {"type": "bool",  "default": True,        "label": "输出ATR止损建议位"},
    "stop_atr_multiplier":           {"type": "float", "default": 1.8,         "label": "止损ATR倍数"},
    "enable_volume_quality_check":   {"type": "bool",  "default": True,        "label": "启用成交量质量检测(过滤刷量)"},
    "vol_conc_ratio_threshold":      {"type": "float", "default": 0.60,        "label": "成交量集中度阈值(>60%视为可疑刷量)"},
    "enable_btc_relative_strength":  {"type": "bool",  "default": True,        "label": "启用相对BTC强弱过滤"},
    "btc_rs_min_ratio":              {"type": "float", "default": -0.3,        "label": "相对BTC最小涨跌幅比(低于此值标记弱势)"},
    # v4.3 交易员视角：1H小时线突破检测 ─────────────────────────────────
    "enable_h1_breakout_detect":     {"type": "bool",  "default": True,        "label": "启用1H突破检测(及时捕获小时线起涨点)"},
    "h1_breakout_lookback":          {"type": "int",   "default": 12,          "label": "1H突破回溯K线数(突破点=N根内最高)"},
    "h1_breakout_vol_ratio":         {"type": "float", "default": 1.35,        "label": "1H突破放量倍数(突破K线量/均量)"},
    "h1_breakout_close_position":    {"type": "float", "default": 0.65,        "label": "1H突破收盘位置(收在K线高位%确认强势)"},
    "h1_breakout_bonus":             {"type": "float", "default": 8.0,         "label": "1H突破检测通过加分"},
    "h1_early_breakout_bonus":      {"type": "float", "default": 4.0,         "label": "1H早期突破(未放量/未确认)加分"},
    "enable_h1_squeeze_detect":     {"type": "bool",  "default": True,        "label": "启用1H波动率压缩检测(布林带收窄→突破前兆)"},
    "h1_squeeze_bb_period":          {"type": "int",   "default": 20,          "label": "布林带周期"},
    "h1_squeeze_bb_width_percentile": {"type": "float","default": 0.20,       "label": "布林带宽历史分位(<此值视为压缩)"},
    "h1_squeeze_bonus":              {"type": "float", "default": 4.0,         "label": "波动压缩预突破加分"},
    "enable_h1_structure_score":    {"type": "bool",  "default": True,        "label": "启用1H结构评分(HH/HL/支撑测试)"},
    "h1_structure_lookback":         {"type": "int",   "default": 24,          "label": "1H结构回溯K线数"},
    "h1_structure_bonus_max":        {"type": "float", "default": 5.0,         "label": "1H结构优秀最高加分"},
    "h1_breakout_relax_3m":         {"type": "bool",  "default": True,        "label": "1H突破时放宽3m回调要求"},
}

_DEFAULT_CONFIG = {key: spec["default"] for key, spec in CONFIG_SCHEMA.items()}


# ══════════════════════════════════════════════════════════════════════════════
class AICrossSectionDualFactorComboScanner(BaseScannerStrategy if _HAS_SCANNER_BASE else object):
    """截面多因子 + AI因子挖掘 + DRL小时趋势启动 + XGBoost + 订单流 五引擎组合。"""

    required_bars = ["3m", "15m", "1H", "4H", "1D"]
    requires_derivative_metrics = True
    requires_on_chain_metrics = True
    name = "AI截面五引擎组合扫描器"
    description = "五引擎共振：截面多因子 + AI因子挖掘 + DRL小时趋势 + XGBoost排序 + 订单流动量"
    strategy_type = "scan"

    CHILDREN = [
        ("截面多因子",      "截面多因子加密货币扫描4.23_v3.py",      "ACrossSectionalMultiFactorScannerStrategy", 970),
        ("AI因子挖掘",      "AI因子挖掘加密货币扫描策略4.23_v3.py",  "AIAutomatedAlphaCryptoScannerStrategy",     960),
        ("DRL小时趋势启动", "DRL元学习小时趋势启动扫描策略4.23_v2.py","DRLMetaHourlyTrendStartScannerStrategy",    965),
        ("XGBoost截面排序", "XGBoost截面排序策略4.27_v2.py",         "XGBoostCrossSectionalRanker",               955),
        ("AI订单流动量",    "AI订单流动量突破组合策略4.27_v2.py",    "AIOrderflowMomentumBreakoutScanner",        945),
    ]

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = {**_DEFAULT_CONFIG, **(config or {})}
        self._apply_runtime_modes()
        self.child_strategies: List[Tuple[str, Any, int]] = []
        self.last_child_results: Dict[str, List[Dict[str, Any]]] = {}
        self.last_child_timing: Dict[str, float] = {}
        self._synced = False  # v2: 防止 _sync_config 重复清空子策略缓存
        # v4.1: 引擎动态权重追踪
        self._engine_track: Dict[str, Dict[str, List[float]]] = {}  # {engine_name: {"BUY": [pnls], "SELL": [pnls]}}
        # v4.1: 信号持续时长追踪 {symbol_direction: last_seen_scan_idx}
        self._signal_persistence: Dict[str, int] = {}
        self._scan_counter: int = 0
        # v4.1: 历史绩效缓存 {symbol: [(direction, score, result_24h), ...]}
        self._performance_cache: Dict[str, List[Tuple[str, float, Optional[float]]]] = {}
        self._max_perf_entries: int = int(self.config.get("max_track_record_entries", 50) or 50)
        if _HAS_SCANNER_BASE and hasattr(super(), "__init__"):
            try:
                super().__init__(self.config)
            except Exception:
                pass
        self.child_strategies = self._build_child_strategies()

    def _init_conditions(self):
        if ScanCondition is None or not hasattr(self, "add_condition"):
            return
        self.add_condition(ScanCondition(
            name="组合流动性", description="过滤成交额不足的交易对",
            field="volume_24h", operator=">=",
            value=self.config.get("min_volume_24h", 8_000_000.0),
        ))

    def get_config_schema(self) -> Dict[str, Any]:
        return dict(CONFIG_SCHEMA)

    # ── 单标的扫描 ──────────────────────────────────────────────────────────
    def scan_symbol(self, symbol) -> Dict[str, Any]:
        child_results = []
        for strategy_name, strategy, priority in self.child_strategies:
            try:
                result = strategy.scan_symbol(symbol)
            except Exception as exc:
                logger.error(f"[组合] {strategy_name} -> {getattr(symbol,'inst_id','')} 失败: {exc}")
                continue
            normalized = self._annotate_child_result(result, strategy_name, priority)
            if normalized.get("passed"):
                child_results.append(normalized)

        consensus = self._build_consensus_result(getattr(symbol, "inst_id", ""), child_results)
        candidates = ([consensus] if consensus else []) + child_results
        if candidates:
            candidates.sort(key=_result_sort_key, reverse=True)
            best = self._apply_m3_hard_filter(candidates[0], symbol)
            return best
        return {
            "symbol": getattr(symbol, "inst_id", ""),
            "passed": False, "score": 0.0, "direction": "WAIT",
            "category": "AI截面组合观察",
            "details": {"状态": "三个子策略均未触发"},
        }

    # ── 批量扫描 ─────────────────────────────────────────────────────────────
    def scan_all_symbols(self, symbols: List) -> Dict[str, Any]:
        top_n_per = int(self.config.get("top_n_per_strategy", 6) or 6)
        scan_symbols = self._select_scan_universe(symbols)
        # 构建 inst_id → symbol 对象的查找表（用于硬性3m过滤）
        sym_lookup: Dict[str, Any] = {
            str(getattr(s, "inst_id", "") or ""): s for s in scan_symbols
        }
        all_child: List[Dict[str, Any]] = []
        by_strategy: Dict[str, List[Dict[str, Any]]] = {}
        timing: Dict[str, float] = {}

        if bool(self.config.get("parallel_child_engines", False)) and len(self.child_strategies) > 1:
            with ThreadPoolExecutor(max_workers=min(5, len(self.child_strategies))) as exc:
                futures = {
                    exc.submit(self._timed_child_scan, sn, st, pri, scan_symbols): (sn, pri)
                    for sn, st, pri in self.child_strategies
                }
                for future in as_completed(futures):
                    sn, _ = futures[future]
                    try:
                        results, elapsed = future.result()
                    except Exception as e:
                        logger.error(f"[组合] {sn} 并行扫描失败: {e}")
                        results, elapsed = [], 0.0
                    timing[sn] = round(float(elapsed), 3)
                    results.sort(key=_result_sort_key, reverse=True)
                    by_strategy[sn] = results[:top_n_per]
                    all_child.extend(results[:top_n_per])
        else:
            for sn, st, pri in self.child_strategies:
                results, elapsed = self._timed_child_scan(sn, st, pri, scan_symbols)
                timing[sn] = round(float(elapsed), 3)
                results.sort(key=_result_sort_key, reverse=True)
                by_strategy[sn] = results[:top_n_per]
                all_child.extend(results[:top_n_per])

        consensus_results = []
        if bool(self.config.get("include_consensus_results", True)):
            consensus_results = self._build_consensus_results(all_child)

        output: List[Dict[str, Any]] = []
        if bool(self.config.get("include_consensus_results", True)):
            output.extend(consensus_results)
        if bool(self.config.get("include_individual_results", True)):
            output.extend(all_child)

        output = _dedupe_results(output, dedupe_by_symbol=bool(self.config.get("dedupe_by_symbol", True)))
        output.sort(key=_result_sort_key, reverse=True)
        top_n = int(self.config.get("top_n", 12) or 12)
        # ── 3m/1H 质量检查 ──────────────────────────────────────────
        # 共识结果（多引擎共振）：跳过硬过滤，仅做软降权
        # 个体结果：执行硬过滤（不通过则淘汰）
        hard_filter = bool(self.config.get("m3_hard_filter", True))
        soft_only = bool(self.config.get("m3_soft_filter_mode", False))
        filtered = []
        relax_for_h1 = bool(self.config.get("h1_breakout_relax_3m", True))
        for item in output:
            sym_id = str(item.get("symbol", ""))
            sym_obj = sym_lookup.get(sym_id)
            if not sym_obj:
                if item.get("passed"):
                    filtered.append(item)
                continue
            # v4.3: 检查1H是否已突破 — 若是则放宽3m过滤
            h1_breakout_confirmed = False
            if relax_for_h1:
                is_bo, bo_conf, _ = self._h1_breakout_detect(sym_obj)
                h1_breakout_confirmed = is_bo and bo_conf >= 0.67
            relax_m3 = h1_breakout_confirmed
            if relax_m3:
                item["_relax_3m"] = True
                item.setdefault("details", {})["1H突破"] = "1H已确认突破，3m过滤放宽"
            is_consensus = "共振" in str(item.get("category", item.get("source_strategy", "")))
            if is_consensus:
                # 共识结果：仅软检查，不淘汰
                check_ok, check_reason = self._m3_pullback_hard_check(
                    self._get_m3_klines(sym_obj),
                    str(item.get("direction", "WAIT")).upper(),
                    relax=relax_m3
                )
                h1_ok, h1_reason = (True, "") if not bool(self.config.get("h1_trend_hard_filter", True)) else \
                    self._h1_trend_check(self._get_h1_klines(sym_obj), str(item.get("direction", "WAIT")).upper())
                details = dict(item.get("details") or {})
                details["3m硬性过滤"] = f"{'通过' if check_ok else '未通过(共识免淘汰)'}: {check_reason}"
                details["1H趋势过滤"] = f"{'通过' if h1_ok else '未通过(共识免淘汰)'}: {h1_reason}"
                item["details"] = details
                if not check_ok:
                    item["score"] = round(max(0, float(item.get("score", 0) or 0) - 5.0), 2)
                if not h1_ok:
                    item["score"] = round(max(0, float(item.get("score", 0) or 0) - 3.0), 2)
                filtered.append(item)
            elif hard_filter and not soft_only:
                filtered_item = self._apply_m3_hard_filter(item, sym_obj)
                if filtered_item.get("passed"):
                    filtered.append(filtered_item)
                else:
                    logger.info(f"[3m硬性过滤] 淘汰 {sym_id}: {(filtered_item.get('details') or {}).get('3m硬性过滤','')}")
            else:
                # 软模式或不启用硬过滤：全部保留
                if item.get("passed"):
                    filtered.append(item)
        final = filtered[:top_n]
        # v4.1: 信号持续时长过滤（去除首次闪现的信号）
        if bool(self.config.get("min_signal_persistence_bars", 0) or 0) > 0:
            final = self._apply_persistence_filter(final)
        # v4.2: 交易员视角过滤器（BTC环境/费率/共振/ATR止损/量质/相对强弱）
        final = self._apply_trading_filters(final, scan_symbols)
        self.last_child_results = by_strategy
        self.last_child_timing = timing
        logger.info(f"[组合] 扫描完成: 子策略产出 {sum(len(v) for v in by_strategy.values())} 条 → "
              f"共识 {len(consensus_results)} 条 → 过滤后 {len(final)} 条 (扫描 {len(scan_symbols)} 个品种)")
        return {
            "type": "ai_cross_section_triple_engine_combo",
            "all_opportunities": final,
            "child_counts": {n: len(v) for n, v in by_strategy.items()},
            "consensus_count": len(consensus_results),
            "dedupe_by_symbol": bool(self.config.get("dedupe_by_symbol", True)),
            "ultra_strict_mode": bool(self.config.get("ultra_strict_mode", False)),
            "scanned_symbols": len(scan_symbols),
            "input_symbols": len(symbols),
            "parallel_child_engines": bool(self.config.get("parallel_child_engines", False)),
            "child_timing_sec": timing if bool(self.config.get("profile_child_timing", True)) else {},
        }

    # ── 回测信号 ─────────────────────────────────────────────────────────────
    def generate_signal(self, data, *args, **kwargs):
        state_mode = self._normalize_state_mode(kwargs.pop("state_mode", None))
        inferred_state = self._infer_market_state_from_data(data)
        active_state = inferred_state if state_mode == "auto" else state_mode

        signals = []
        for sn, st, pri in self.child_strategies:
            if not hasattr(st, "generate_signal"):
                continue
            try:
                # 透传市场状态给子策略（如果子策略支持）
                child_kw = dict(kwargs)
                child_kw.setdefault("market_state", active_state)
                sig = st.generate_signal(data, *args, **child_kw)
            except Exception as e:
                logger.error(f"[组合] {sn} 回测信号失败: {e}")
                continue
            if not sig:
                continue
            norm = dict(sig)
            norm["source_strategy"] = sn
            norm["_priority"] = pri
            norm["_action_norm"] = "SHORT" if str(norm.get("action", "")).upper() == "SELL" else str(norm.get("action", "")).upper()
            base = float(norm.get("score", 0.0) or 0.0)
            bonus, mult = self._state_adjustment(sn, active_state)
            norm["score"] = max(0.0, min(100.0, (base + bonus) * mult))
            norm["_base_score"] = base
            signals.append(norm)

        if not signals:
            return None
        signals.sort(key=lambda x: (float(x.get("score", 0) or 0), int(x.get("_priority", 0) or 0)), reverse=True)

        min_consensus = int(self.config.get("min_consensus_engines", 2) or 2)
        pool = [s for s in signals if s.get("_action_norm") in {"BUY", "SHORT"}]
        counts: Dict[str, int] = {}
        for s in pool:
            a = str(s.get("_action_norm", ""))
            counts[a] = counts.get(a, 0) + 1

        if counts:
            best_action = max(counts, key=counts.get)
            support = counts.get(best_action, 0)
        else:
            best_action, support = "", 0

        if best_action in {"BUY", "SHORT"} and support >= min_consensus:
            aligned = [s for s in pool if s.get("_action_norm") == best_action]
            best = max(aligned, key=lambda s: float(s.get("score", 0) or 0))
            base_avg = sum(float(s.get("score", 0) or 0) for s in aligned) / len(aligned)

            # ── 共振加分（上限封顶，与 _build_consensus_result 一致）──
            if support >= 5:
                rb = min(12.0, float(self.config.get("triple_consensus_bonus", 13.0) or 13.0) * 0.90)
                rl = "五引擎同向最强共振"
            elif support >= 4:
                rb = min(10.0, float(self.config.get("triple_consensus_bonus", 13.0) or 13.0) * 0.75)
                rl = "四引擎同向强共振"
            elif support >= 3:
                rb = min(8.0, float(self.config.get("triple_consensus_bonus", 13.0) or 13.0) * 0.60)
                rl = "三引擎强共振"
            elif len(pool) > support:
                rb = min(5.0, float(self.config.get("consensus_bonus", 8.5) or 8.5) * 0.55)
                rl = "多引擎多数共振"
            else:
                rb = min(4.0, float(self.config.get("consensus_bonus", 8.5) or 8.5) * 0.45)
                rl = "双引擎同向共振"

            # ── 信息比率：引擎分数标准差越小 → 共识越强 ──
            aligned_scores = [float(s.get("score", 0) or 0) for s in aligned]
            score_std = float(np.std(aligned_scores)) if len(aligned_scores) >= 2 else 10.0
            if score_std > 1e-6:
                ir = base_avg / score_std
                ir_bonus = min(5.0, max(0.0, np.log1p(max(ir - 4, 0)) * 2.0))
            else:
                ir_bonus = 0.0

            # ── 争议惩罚 ──
            disagree_count = len(pool) - support
            cp = float(self.config.get("direction_conflict_penalty", 8.0) or 8.0) * (disagree_count / max(len(pool), 1)) * 0.5

            score = min(100.0, max(0.0, base_avg + rb + ir_bonus - cp))
            use_pilot = bool(self.config.get("use_pilot_add_system", True))
            ps = float(self.config.get("position_size", 0.02) or 0.02)
            result = {
                "action": best_action,
                "entry_price": float(best.get("entry_price", 0.0) or 0.0),
                "reason": f"{rl} | {score:.1f} | " + " / ".join(s["source_strategy"] for s in aligned)
                          + f" | 状态={active_state} | IR+{ir_bonus:.1f}-分歧{disagree_count}",
                "score": score, "raw_signals": signals,
                "consensus_engines": support, "state_mode": state_mode,
                "market_state": active_state, "inferred_market_state": inferred_state,
                "strategy_gates": {"h4_trend", "h1_trend", "rsi", "m3_pullback", "volume", "d1_trend", "entry_rule"},
                "timeframe_bias": f"multi_engine_{best_action}",
            }
            if not use_pilot:
                result["position_size"] = ps
            return result

        best = signals[0]
        min_score = float(self.config.get("backtest_min_score", 60.0) or 60.0)
        if float(best.get("score", 0) or 0) < min_score:
            return None
        best = dict(best)
        best.setdefault("strategy_gates", {"h4_trend", "h1_trend", "rsi", "m3_pullback", "volume"})
        best.update({"state_mode": state_mode, "market_state": active_state, "inferred_market_state": inferred_state})
        return best

    def reset_backtest_state(self):
        for _, st, _ in self.child_strategies:
            if hasattr(st, "reset_backtest_state"):
                try: st.reset_backtest_state()
                except Exception: pass
        self.last_child_results.clear()
        self._synced = False

    # ── 内部方法 ─────────────────────────────────────────────────────────────
    def _apply_runtime_modes(self) -> None:
        mode = str(self.config.get("mode", "normal")).strip().lower()
        ultra = bool(self.config.get("ultra_strict_mode", False)) or mode in {"1","ultra","ultra_strict","strict","超严"}
        if not ultra:
            self.config["ultra_strict_mode"] = False
            self.config["mode"] = "normal"
            return
        self.config["ultra_strict_mode"] = True
        self.config["mode"] = "ultra"
        tn = max(6, min(9, int(self.config.get("ultra_target_top_n", 9) or 9)))
        self.config["min_score"] = max(float(self.config.get("min_score", 83.0) or 83.0), 86.0)
        self.config["backtest_min_score"] = max(float(self.config.get("backtest_min_score", 75.0) or 75.0), 78.0)
        self.config["top_n"] = min(int(self.config.get("top_n", tn) or tn), tn)
        self.config["top_n_per_strategy"] = min(int(self.config.get("top_n_per_strategy", 6) or 6), 5)
        self.config["min_consensus_engines"] = max(int(self.config.get("min_consensus_engines", 2) or 2), 2)
        self.config["direction_conflict_penalty"] = max(float(self.config.get("direction_conflict_penalty", 8.0) or 8.0), 9.0)
        self.config["include_consensus_results"] = True
        self.config["dedupe_by_symbol"] = True

    def _select_scan_universe(self, symbols: List) -> List:
        if not bool(self.config.get("fast_scan_mode", True)):
            return list(symbols)
        limit = int(self.config.get("max_scan_symbols", 200) or 200)
        if limit <= 0 or len(symbols) <= limit:
            return list(symbols)
        return sorted(symbols, key=self._scan_universe_score, reverse=True)[:limit]

    def _scan_universe_score(self, symbol) -> float:
        vol = _safe_number(getattr(symbol, "volume_24h", 0.0))
        chg = abs(_safe_number(getattr(symbol, "price_change_24h", 0.0)))
        klines = getattr(symbol, "extra_data", {}).get("klines", {}) if getattr(symbol, "extra_data", None) else {}
        kb = sum(1.0 for bar in ("15m","1H","4H","1D") if klines.get(bar)) * 3.0
        h1m = _recent_kline_move_pct(klines.get("1H"))
        m15m = _recent_kline_move_pct(klines.get("15m"))
        # 动量质量分：趋势性波动 > 震荡噪音（用效率比衡量）
        h1_rows = klines.get("1H") or []
        momentum_quality = 0.0
        if len(h1_rows) >= 8:
            try:
                closes = [float(r[4]) for r in h1_rows[-8:] if float(r[4]) > 0]
                if len(closes) >= 6:
                    net_move = abs(closes[-1] / max(closes[0], 1e-9) - 1.0) * 100.0
                    path_sum = sum(abs(d) for d in pd.Series(closes).pct_change().dropna()) * 100.0
                    if path_sum > 0:
                        er = min(1.0, net_move / path_sum)  # 效率比：越接近1=趋势越干净
                        momentum_quality = er * 10.0
            except Exception: pass
        # 用 symbol hash 做 micro-tiebreaker，确保同分时结果稳定
        sym_hash = float(hash(str(getattr(symbol, "inst_id", ""))) % 1000) / 10000.0
        return (min(np.log10(max(vol,1.0)),11.0)*8 + min(chg,40.0)*1.8
                + min(abs(h1m),30.0)*2.4 + min(abs(m15m),20.0)*1.2
                + kb + momentum_quality + sym_hash)

    def _select_child_symbols(self, strategy_name: str, symbols: List) -> List:
        if strategy_name != "DRL小时趋势启动":
            return list(symbols)
        cap = int(self.config.get("drl_candidate_cap", 180) or 180)
        if cap <= 0 or len(symbols) <= cap:
            return list(symbols)
        return sorted(symbols, key=self._drl_candidate_score, reverse=True)[:cap]

    def _drl_candidate_score(self, symbol) -> float:
        vol = _safe_number(getattr(symbol, "volume_24h", 0.0))
        chg = abs(_safe_number(getattr(symbol, "price_change_24h", 0.0)))
        klines = (getattr(symbol, "extra_data", {}) or {}).get("klines", {})
        h1m = _recent_kline_move_pct(klines.get("1H"), 6)
        m15m = _recent_kline_move_pct(klines.get("15m"), 8)
        return min(np.log10(max(vol,1.0)),11.0)*8 + min(chg,40.0)*2.5 + min(abs(h1m),30.0)*3 + min(abs(m15m),20.0)*1.5

    def _timed_child_scan(self, sn, st, pri, symbols):
        child_syms = self._select_child_symbols(sn, symbols)
        t0 = time.perf_counter()
        results = self._run_child_scan(sn, st, pri, child_syms)
        return results, time.perf_counter() - t0

    def _run_child_scan(self, sn, st, pri, symbols):
        results = []
        sym_lookup = {str(getattr(s, "inst_id", "") or ""): s for s in symbols}
        try:
            if hasattr(st, "scan_all_symbols") and callable(st.scan_all_symbols):
                batch = st.scan_all_symbols(symbols)
                for item in (batch.get("all_opportunities", []) if isinstance(batch, dict) else []):
                    norm = self._annotate_child_result(item, sn, pri)
                    if norm.get("passed"):
                        results.append(norm)
                return results
        except Exception as e:
            logger.warning(f"[组合] {sn} 批量扫描失败，降级逐个: {e}")
        for sym in symbols:
            try:
                item = st.scan_symbol(sym)
            except Exception as e:
                logger.error(f"[组合] {sn} -> {getattr(sym,'inst_id','')} 失败: {e}")
                continue
            norm = self._annotate_child_result(item, sn, pri)
            if norm.get("passed"):
                results.append(norm)
        # v4.1: 子引擎结果在进入共识前先过3m预过滤
        if bool(self.config.get("prefilter_3m_before_consensus", True)) and results:
            filtered_results = []
            for r in results:
                sym_id = str(r.get("symbol", ""))
                sym_obj = sym_lookup.get(sym_id)
                if not sym_obj:
                    filtered_results.append(r)
                    continue
                r_dir = str(r.get("direction", "WAIT")).upper()
                if r_dir not in {"BUY", "SELL"}:
                    filtered_results.append(r)
                    continue
                m3_rows = self._get_m3_klines(sym_obj)
                m3_ok, m3_reason = self._m3_pullback_hard_check(m3_rows, r_dir)
                if m3_ok:
                    r["details"] = dict(r.get("details") or {})
                    r["details"]["3m预过滤"] = f"通过: {m3_reason}"
                    filtered_results.append(r)
            return filtered_results
        return results

    def _annotate_child_result(self, result, sn, priority):
        norm = dict(result or {})
        if not norm:
            return norm
        norm["source_strategy"] = sn
        norm["category"] = f"{sn} | {norm.get('category', norm.get('strategy_category', '扫描机会'))}"
        norm["strategy_category"] = norm["category"]
        norm["group_sort_score"] = priority
        details = norm.get("details")
        if isinstance(details, dict):
            details.setdefault("来源策略", sn)
            details.setdefault("机会类型", norm["category"])
        signals = list(norm.get("signals", []) or [])
        if not signals:
            signals = [norm["category"]]
        elif sn not in str(signals[0]):
            signals[0] = f"{sn}: {signals[0]}"
        norm["signals"] = signals
        if enrich_scan_result:
            try: enrich_scan_result(norm)
            except Exception: pass
        return norm

    def _build_consensus_results(self, child_results):
        by_sym: Dict[str, List] = {}
        for item in child_results:
            by_sym.setdefault(str(item.get("symbol", "")), []).append(item)
        return [r for r in (self._build_consensus_result(sym, items) for sym, items in by_sym.items()) if r]

    def _build_consensus_result(self, symbol: str, items: List) -> Optional[Dict[str, Any]]:
        min_eng = int(self.config.get("min_consensus_engines", 2) or 2)
        if len(items) < min_eng:
            return None
        strategies = {item.get("source_strategy") for item in items}
        if len(strategies) < min_eng:
            return None
        dirs = [str(item.get("direction","WAIT")).upper() for item in items if str(item.get("direction","WAIT")).upper() in {"BUY","SELL"}]
        if not dirs:
            return None
        dc = {d: dirs.count(d) for d in {"BUY","SELL"}}
        direction = max(dc, key=dc.get)
        support = int(dc.get(direction, 0))
        disagree = max(0, len(dirs) - support)
        if support < min_eng:
            return None

        # ── v4.4: 统一市场状态推断 ──────────────────────────────────
        active_state = self._infer_consensus_state(items)
        # 提取时效性数据（用于半衰期衰减）
        max_trend_age = 0
        max_staleness = 0
        for it in items:
            d = it.get("details") if isinstance(it.get("details"), dict) else {}
            age_v = _safe_number(d.get("1H趋势延续根数", "0"), 0)
            stale_v = _safe_number(d.get("3分钟时效(根)", "0"), 0)
            if age_v > max_trend_age: max_trend_age = int(age_v)
            if stale_v > max_staleness: max_staleness = int(stale_v)

        # ── 基础评分：引擎动态胜率加权均值 ─────────────────────────
        child_scores = []
        if bool(self.config.get("enable_engine_track_record", True)):
            eng_weights = {}
            for it in items:
                sn = str(it.get("source_strategy", ""))
                d = str(it.get("direction", "WAIT")).upper()
                eng_weights[sn] = self._get_engine_weight(sn, d)
            weighted = sum(
                float(it.get("score", 0) or 0) * eng_weights.get(it.get("source_strategy", ""), 1.0)
                for it in items
            )
            weight_sum = sum(eng_weights.get(it.get("source_strategy", ""), 1.0) for it in items)
            if weight_sum > 0:
                base_score = weighted / weight_sum
            else:
                base_score = sum(float(it.get("score", 0) or 0) for it in items) / len(items)
            # 同时收集加权后分数用于 IR 计算
            for it in items:
                sn = str(it.get("source_strategy", ""))
                w = eng_weights.get(sn, 1.0)
                child_scores.append(float(it.get("score", 0) or 0) * w)
        else:
            raw_scores = [float(it.get("score", 0) or 0) for it in items]
            base_score = sum(raw_scores) / len(items)
            child_scores = list(raw_scores)

        opp_score = base_score * 0.98

        # ── v4.4: 统一市场状态调整（仅一套权重，取代旧的 STATE_WEIGHTS + _state_adjustment）──
        if bool(self.config.get("enable_state_conditional_weights", True)):
            total_bonus = 0.0
            total_mult = 1.0
            count = 0
            for it in items:
                sn = str(it.get("source_strategy", ""))
                bonus, mult = self._state_adjustment(sn, active_state)
                total_bonus += bonus
                total_mult += mult
                count += 1
            avg_bonus = total_bonus / max(count, 1)
            avg_mult = total_mult / max(count, 1)
            base_score = base_score + avg_bonus * 0.5  # 削弱奖金影响力（避免与共振 bonus 叠加过猛）
            base_score = base_score * (1.0 + (avg_mult - 1.0) * 0.3)
            opp_score = base_score * 0.97

        # ── v4.4: 趋势半衰期衰减（老趋势信号降权） ────────────────
        if max_trend_age > 0:
            age_limit = int(self.config.get("max_h1_trend_age", 12) or 12)
            half_life = max(age_limit * 0.5, 4.0)
            decay = 2.0 ** (-max_trend_age / half_life)  # e.g., age=12, half=6 → 0.25
            # 新旧混合：decay 越低信号越不可靠
            freshness_factor = 0.3 + 0.7 * decay  # 保底 0.3
            base_score *= freshness_factor
            opp_score *= freshness_factor

        # ── v4.4: 信息比率排名（引擎共识度） ───────────────────────
        consensus_ir_bonus = 0.0
        if len(child_scores) >= 2:
            score_std = float(np.std(child_scores))
            if score_std > 1e-6:
                ir = base_score / score_std  # 高均值 + 低方差 = 引擎高度一致
                # 将 IR 映射到 bonus：IR=5 → +2, IR=10 → +5, IR=20 → +8
                consensus_ir_bonus = min(8.0, max(0.0, np.log1p(max(ir - 3, 0)) * 2.5))
                base_score += consensus_ir_bonus
                opp_score += consensus_ir_bonus * 0.8

        # ── v4.4: 引擎共振加分（上限封顶，防止通胀） ─────────
        # 加分规则：支持引擎越多加分越高，但有上限
        if support >= 5:
            rb = min(15.0, float(self.config.get("triple_consensus_bonus", 13.0) or 13.0) * 1.12)
            signal_head = "五引擎同向强共振"; category = "五引擎强共振"; gss = 1400
        elif support >= 4:
            rb = min(13.0, float(self.config.get("triple_consensus_bonus", 13.0) or 13.0))
            signal_head = "四引擎同向强共振"; category = "四引擎强共振"; gss = 1370
        elif support >= 3:
            rb = min(11.0, float(self.config.get("triple_consensus_bonus", 13.0) or 13.0) * 0.85)
            signal_head = "三引擎同向强共振"; category = "三引擎强共振"; gss = 1350
        elif disagree == 0:
            rb = min(7.0, float(self.config.get("consensus_bonus", 8.5) or 8.5) * 0.82)
            signal_head = "双引擎同向共振"; category = "双引擎共振"; gss = 1240
        else:
            rb = min(5.0, float(self.config.get("consensus_bonus", 8.5) or 8.5) * 0.55)
            signal_head = "多引擎多数共振(含分歧)"; category = "多引擎多数共振"; gss = 1180

        # 方向分歧惩罚（按分歧比例，每 20% 分歧 —1 分）
        disagreement_ratio = disagree / max(len(dirs), 1)
        cp = float(self.config.get("direction_conflict_penalty", 8.0) or 8.0) * disagreement_ratio * 0.8

        # ── 最终评分：基础分 + 共振加分 + IR bonus − 分歧惩罚 ──
        raw_score = base_score + rb - cp
        # 加分项总计封顶（防止 trading_filters 后续再加分导致通胀）
        score = min(100.0, max(0.0, raw_score))
        opp_score = min(100.0, max(0.0, opp_score + rb - cp))
        passed = score >= float(self.config.get("min_score", 83.0) or 83.0)
        # 记录评分构成供诊断
        score_detail = {
            "base": round(float(base_score), 2),
            "resonance_bonus": round(float(rb), 2),
            "conflict_penalty": round(float(cp), 2),
            "ir_bonus": round(float(consensus_ir_bonus), 2),
            "state": active_state,
            "trend_age": max_trend_age,
        }

        src_names = [str(item.get("source_strategy","")) for item in items]
        lp = _first_number(items, "last_price")
        v24 = _first_number(items, "volume_24h")
        pc24 = _first_number(items, "price_change_24h")

        # v2: 汇总时效性信息
        age_infos = []
        stale_infos = []
        # v3.2: 汇总微观结构信息
        micro_infos = []
        for item in items:
            d = item.get("details") if isinstance(item.get("details"), dict) else {}
            age_s = str(d.get("1H趋势延续根数","")).strip()
            stale_s = str(d.get("3分钟时效(根)","")).strip()
            micro_s = str(d.get("3m微观指标", d.get("微观指标", ""))).strip()
            if age_s: age_infos.append(f"{item.get('source_strategy')}={age_s}")
            if stale_s: stale_infos.append(f"{item.get('source_strategy')}={stale_s}")
            if micro_s: micro_infos.append(f"{item.get('source_strategy')}:{micro_s}")

        m3_states = []
        for item in items:
            d = item.get("details") if isinstance(item.get("details"), dict) else {}
            st = str(d.get("3分钟结构","") or "").strip()
            if st: m3_states.append(f"{item.get('source_strategy')}={st}")

        signals = [
            f"{signal_head} {score:.1f}",
            f"同向支持引擎: {support}/{len(items)}",
            "来源: " + " / ".join(src_names),
            "子策略评分: " + " / ".join(f"{item.get('source_strategy')}={float(item.get('score',0) or 0):.1f}" for item in items),
        ]
        if m3_states: signals.append("3m结构: " + " / ".join(m3_states))
        if age_infos: signals.append("1H趋势时效: " + " / ".join(age_infos))
        if micro_infos: signals.append("3m微观: " + " / ".join(micro_infos))  # v3.2

        result = {
            "symbol": symbol, "passed": passed,
            "score": round(score, 2), "opportunity_score": round(opp_score, 2),
            "direction": direction, "signals": signals,
            "category": category, "strategy_category": category,
            "source_strategy": "多引擎共振",
            "group_sort_score": gss,
            "last_price": lp, "volume_24h": v24, "price_change_24h": pc24,
            "child_results": items, "consensus_engines": support,
            "details": {
                "机会类型": category,
                "来源策略": " / ".join(src_names),
                "主方向": direction,
                "同向支持引擎": f"{support}/{len(items)}",
                "方向分歧数": str(disagree),
                "评分构成": f"基础{score_detail['base']:.1f}+共振{score_detail['resonance_bonus']:.1f}"
                            f"+IR{score_detail['ir_bonus']:.1f}-分歧{score_detail['conflict_penalty']:.1f}"
                            f" | 状态={score_detail['state']} 时效={score_detail['trend_age']}根",
                "子策略评分": " / ".join(f"{item.get('source_strategy')}={float(item.get('score',0) or 0):.1f}" for item in items),
                "3分钟结构": " / ".join(m3_states) if m3_states else "-",
                "1H趋势时效": " / ".join(age_infos) if age_infos else "-",   # v2
                "3m回调时效": " / ".join(stale_infos) if stale_infos else "-", # v2
                "3m微观指标": " / ".join(micro_infos) if micro_infos else "-", # v3.2
                "评估": " | ".join(signals),
            },
            "ranking_factors": _merge_ranking_factors(items, score),
        }
        if build_opportunity_profile:
            try: result.update(build_opportunity_profile(score, direction, v24, result["ranking_factors"], signals))
            except Exception: pass
        # v4.1: 历史绩效标签
        perf_label = self._get_signal_performance_label(symbol, direction)
        if perf_label != "无历史记录":
            result.setdefault("details", {})
            result["details"]["历史绩效"] = perf_label

        return result if result.get("passed") else None

    # ── v4.1: 信号持久度过滤 ──────────────────────────────────────────────────
    def _apply_persistence_filter(self, results: List[Dict]) -> List[Dict]:
        """过滤闪现信号：同一币种同方向需在连续N次扫描中出现"""
        min_bars = max(1, int(self.config.get("min_signal_persistence_bars", 3) or 3))
        self._scan_counter += 1
        passed = []
        for item in results:
            sym = str(item.get("symbol", ""))
            d = str(item.get("direction", "WAIT")).upper()
            key = f"{sym}:{d}"
            last_seen = self._signal_persistence.get(key, -999)
            if self._scan_counter - last_seen <= 1:
                # 连续出现：累计计数
                streak = self._signal_persistence.get(f"{key}:streak", 0) + 1
                self._signal_persistence[f"{key}:streak"] = streak
                self._signal_persistence[key] = self._scan_counter
                if streak >= min_bars:
                    passed.append(item)
                    item.setdefault("_persistence_streak", streak)
            else:
                # 首次出现或中断：重置计数
                self._signal_persistence[f"{key}:streak"] = 1
                self._signal_persistence[key] = self._scan_counter
                if min_bars <= 1:
                    passed.append(item)
                    item.setdefault("_persistence_streak", 1)
                else:
                    logger.debug(f"[持续检查] {sym} {d} 首次出现(需连续{min_bars}次扫描确认)")
        return passed

    def report_trade_outcome(self, source_strategy: str, direction: str, pnl: float):
        """外部（回测引擎/实盘）调用：报告一笔交易的实际结果。

        Args:
            source_strategy: 子策略名称（如"DRL小时趋势启动"）
            direction: "BUY" 或 "SELL"
            pnl: 实际收益率（小数，如 0.03 = 3%）
        """
        if not bool(self.config.get("enable_engine_track_record", True)):
            return
        d = str(direction).upper()
        if d == "SELL": d = "SHORT"
        self._record_engine_performance(source_strategy, d, float(pnl))
        if d in {"BUY", "SHORT"}:
            ws = [self._get_engine_weight(sn, d)
                  for sn, _, _ in self.child_strategies]
            logger.debug(f"[引擎权重] {source_strategy} {d} PnL={pnl:+.3f} → 胜率权重: "
                        f"{' / '.join(f'{sn}={w:.2f}' for (sn,_,_), w in zip(self.child_strategies, ws))}")

    def _record_performance_for_scan_results(self, results: List[Dict]) -> None:
        """批量初始化引擎追踪条目（PnL=0.0，由 report_trade_outcome 更新）"""
        for item in results:
            child_results = item.get("child_results") if isinstance(item.get("child_results"), list) else []
            for cr in child_results:
                sn = str(cr.get("source_strategy", ""))
                cd = str(cr.get("direction", "WAIT")).upper()
                if sn and cd in {"BUY", "SELL"}:
                    # 只在首次出现时初始化，避免覆盖已有数据
                    track = self._engine_track.setdefault(sn, {}).setdefault(cd, [])
                    if not track:
                        pass  # 等 report_trade_outcome 来填充

    # ═══════════════════════════════════════════════════════════════════════════
    # v4.2 交易员视角 — 风险/市场环境过滤器
    # ═══════════════════════════════════════════════════════════════════════════

    # ── 1. BTC 相关性过滤 ────────────────────────────────────────────────────
    def _get_btc_context(self, symbols: List) -> Dict[str, float]:
        """从扫描品种列表中提取 BTC 基准数据"""
        btc = next((s for s in symbols if str(getattr(s, "inst_id", "")).upper() in
                     {"BTC-USDT-SWAP", "BTC-USDT"}), None)
        if not btc:
            return {}
        klines = (getattr(btc, "extra_data", {}) or {}).get("klines", {})
        h1 = klines.get("1H") or klines.get("1h") or []
        h4 = klines.get("4H") or klines.get("4h") or []
        d1 = klines.get("1D") or klines.get("1d") or []
        ctx = {}
        # BTC 1H 涨跌
        if len(h1) >= 2:
            try:
                ctx["btc_1h_move"] = (float(h1[-1][4]) / float(h1[-2][4]) - 1.0) * 100
            except: pass
        # BTC 4H EMA 状态
        if len(h4) >= 50:
            try:
                closes = [float(r[4]) for r in h4 if float(r[4]) > 0]
                ema20 = pd.Series(closes).ewm(span=20, adjust=False).mean().iloc[-1]
                ema50 = pd.Series(closes).ewm(span=50, adjust=False).mean().iloc[-1]
                ctx["btc_4h_bullish"] = float(ema20 > ema50)
            except: pass
        # BTC 24h 涨跌 (from ticker)
        ctx["btc_24h_move"] = _safe_number(getattr(btc, "price_change_24h", 0))
        return ctx

    def _apply_btc_correlation_filter(self, results: List[Dict], btc_ctx: Dict) -> List[Dict]:
        """BTC暴跌时降权山寨币多头信号"""
        if not bool(self.config.get("enable_btc_correlation_filter", True)) or not btc_ctx:
            return results
        threshold = float(self.config.get("btc_dump_threshold_pct", -2.5) or -2.5)
        penalty = float(self.config.get("btc_dump_penalty", 12.0) or 12.0)
        btc_move = btc_ctx.get("btc_1h_move", btc_ctx.get("btc_24h_move", 0))
        btc_4h_bull = btc_ctx.get("btc_4h_bullish", 1.0)
        is_dump = btc_move < threshold
        for item in results:
            sym = str(item.get("symbol", "")).upper()
            if "BTC" in sym:
                continue  # BTC自身不受影响
            direction = str(item.get("direction", "WAIT")).upper()
            if is_dump and direction in {"BUY", "LONG"}:
                item["score"] = round(max(0, float(item.get("score", 0) or 0) - penalty), 2)
                item["_btc_penalty"] = True
                d = item.setdefault("details", {})
                d["BTC环境"] = f"⚠ BTC 1H跌{btc_move:.1f}%，山寨多头降{penalty}分"
            elif not is_dump and btc_4h_bull:
                # BTC 稳定或上涨：不加分也不扣分，仅标注
                d = item.setdefault("details", {})
                d["BTC环境"] = f"✓ BTC 4H多头确认，山寨多头环境安全"
        return results

    # ── 2. 资金费率过滤 ──────────────────────────────────────────────────────
    def _apply_funding_filter(self, results: List[Dict], symbols: List) -> List[Dict]:
        """极端资金费率时降权同向信号（多头拥挤→降权多头，空头拥挤→降权空头）"""
        if not bool(self.config.get("enable_funding_filter", True)):
            return results
        threshold = float(self.config.get("funding_extreme_threshold", 0.10) or 0.10) / 100.0
        penalty = float(self.config.get("funding_penalty", 8.0) or 8.0)
        for item in results:
            sym_id = str(item.get("symbol", ""))
            sym = next((s for s in symbols if str(getattr(s, "inst_id", "")) == sym_id), None)
            if not sym:
                continue
            funding = _safe_number(
                (getattr(sym, "extra_data", {}) or {}).get("funding_rate", 0)
            )
            if abs(funding) < threshold:
                continue
            direction = str(item.get("direction", "WAIT")).upper()
            d = item.setdefault("details", {})
            if funding > threshold and direction in {"BUY", "LONG"}:
                item["score"] = round(max(0, float(item.get("score", 0) or 0) - penalty), 2)
                d["资金费率"] = f"⚠ 多头拥挤({funding*100:.3f}%)，降{penalty}分"
            elif funding < -threshold and direction in {"SELL", "SHORT"}:
                item["score"] = round(max(0, float(item.get("score", 0) or 0) - penalty), 2)
                d["资金费率"] = f"⚠ 空头拥挤({funding*100:.3f}%)，降{penalty}分"
            else:
                d["资金费率"] = f"正常({funding*100:.3f}%)"
        return results

    # ── 3. 多周期趋势共振评分 ───────────────────────────────────────────────
    def _get_symbol_klines_indicators(self, symbol, bars=("1H","4H","1D")) -> Dict[str, float]:
        """提取单个品种的多周期EMA/ADX指标"""
        klines = (getattr(symbol, "extra_data", {}) or {}).get("klines", {})
        ind = {}
        for bar in bars:
            rows = klines.get(bar) or []
            if len(rows) < 26:
                continue
            try:
                closes = [float(r[4]) for r in rows if float(r[4]) > 0]
                if len(closes) < 26:
                    continue
                s = pd.Series(closes)
                ema12 = float(s.ewm(span=12, adjust=False).mean().iloc[-1])
                ema26 = float(s.ewm(span=26, adjust=False).mean().iloc[-1])
                ind[f"{bar}_bullish"] = 1.0 if ema12 > ema26 else 0.0
                ind[f"{bar}_gap_pct"] = abs(ema12 - ema26) / max(ema26, 1e-9) * 100
                # 简版ADX
                if len(closes) >= 28:
                    tr_vals = [max(closes[i]-closes[i-1], closes[i-1]-closes[i], 0)
                               for i in range(1, len(closes))]
                    if len(tr_vals) >= 14:
                        atr14 = float(pd.Series(tr_vals).rolling(14).mean().iloc[-1])
                        last_px = closes[-1]
                        if atr14 > 0 and last_px > 0:
                            ind[f"{bar}_adx_pct"] = atr14 / last_px * 100
            except: pass
        return ind

    def _apply_confluence_scoring(self, results: List[Dict], symbols: List) -> List[Dict]:
        """多周期EMA金叉/死叉共振加分：根据方向对称处理"""
        if not bool(self.config.get("enable_confluence_scoring", True)):
            return results
        max_bonus = float(self.config.get("confluence_bonus_max", 6.0) or 6.0)
        sym_map = {str(getattr(s, "inst_id", "")): s for s in symbols}
        for item in results:
            sym_id = str(item.get("symbol", ""))
            sym = sym_map.get(sym_id)
            if not sym:
                continue
            direction = str(item.get("direction", "WAIT")).upper()
            ind = self._get_symbol_klines_indicators(sym)
            if not ind:
                continue
            # 根据方向计算共振：多头=金叉数，空头=死叉数
            bullish = [ind.get(f"{b}_bullish", 0) for b in ("1H","4H","1D")]
            if direction in {"SELL", "SHORT"}:
                aligned_count = sum(1 for v in bullish if v == 0.0)  # 死叉=EMA12<EMA26
            else:
                aligned_count = sum(1 for v in bullish if v == 1.0)  # 金叉
            if aligned_count >= 3:
                bonus = max_bonus
                label = "★★★ 1H/4H/日线三周期共振"
            elif aligned_count >= 2:
                bonus = max_bonus * 0.60
                label = "★★ 双周期共振"
            else:
                bonus = 0
                label = f"★ {aligned_count}周期对齐"
            if bonus > 0:
                item["score"] = round(min(100, float(item.get("score", 0) or 0) + bonus), 2)
            d = item.setdefault("details", {})
            gaps = " | ".join(f"{b}={ind.get(b+'_gap_pct',0):.1f}%" for b in ("1H","4H","1D") if f"{b}_gap_pct" in ind)
            d["周期共振"] = f"{label} ({'金叉' if direction in {'BUY','LONG'} else '死叉'}) (+{bonus:.1f}分) [{gaps}]"

        return results

    # ── 4. ATR 动态止损建议 ──────────────────────────────────────────────────
    def _get_atr(self, rows: List, period: int = 14) -> float:
        """从原始K线列表计算 ATR"""
        if not rows or len(rows) < period + 2:
            return 0.0
        trs = []
        for i in range(1, min(len(rows), period + 12)):
            try:
                h, l, pc = float(rows[-i][2]), float(rows[-i][3]), float(rows[-i-1][4])
                trs.append(max(h - l, abs(h - pc), abs(l - pc)))
            except: pass
        if not trs:
            return 0.0
        return float(pd.Series(trs).ewm(span=period, adjust=False).mean().iloc[-1])

    def _apply_atr_stop_suggestion(self, results: List[Dict], symbols: List) -> List[Dict]:
        """为每条信号附加基于15m ATR的动态止损建议"""
        if not bool(self.config.get("enable_atr_stop_suggestion", True)):
            return results
        mult = float(self.config.get("stop_atr_multiplier", 1.8) or 1.8)
        sym_map = {str(getattr(s, "inst_id", "")): s for s in symbols}
        for item in results:
            sym_id = str(item.get("symbol", ""))
            sym = sym_map.get(sym_id)
            if not sym:
                continue
            klines = (getattr(sym, "extra_data", {}) or {}).get("klines", {})
            m15 = klines.get("15m") or klines.get("15M") or []
            atr = self._get_atr(m15)
            if atr <= 0:
                rows_3m = klines.get("3m") or []
                atr = self._get_atr(rows_3m, 14) * 2.0  # 3m ATR ×2 近似15m ATR
            if atr <= 0:
                continue
            lp = _safe_number(getattr(sym, "last_price", item.get("last_price", 0)))
            if lp <= 0:
                continue
            direction = str(item.get("direction", "WAIT")).upper()
            sl_price = lp - atr * mult if direction in {"BUY", "LONG"} else lp + atr * mult
            sl_pct = atr * mult / lp * 100
            d = item.setdefault("details", {})
            d["ATR止损建议"] = f"{sl_price:.6g} ({sl_pct:.2f}% | {mult}×ATR={atr:.6g})"
        return results

    # ── 5. 成交量质量 ────────────────────────────────────────────────────────
    def _apply_volume_quality_check(self, results: List[Dict], symbols: List) -> List[Dict]:
        """检测刷量：超过60%成交量集中在单根K线 → 降权"""
        if not bool(self.config.get("enable_volume_quality_check", True)):
            return results
        threshold = float(self.config.get("vol_conc_ratio_threshold", 0.60) or 0.60)
        sym_map = {str(getattr(s, "inst_id", "")): s for s in symbols}
        for item in results:
            sym_id = str(item.get("symbol", ""))
            sym = sym_map.get(sym_id)
            if not sym:
                continue
            klines = (getattr(sym, "extra_data", {}) or {}).get("klines", {})
            h1 = klines.get("1H") or klines.get("1h") or []
            if len(h1) < 12:
                continue
            try:
                vols = []
                for r in h1[-12:]:
                    v = float(r[5]) if len(r) > 5 else 0
                    if v > 0: vols.append(v)
                if len(vols) >= 8:
                    max_vol = max(vols)
                    total = sum(vols)
                    conc = max_vol / max(total, 1e-9)
                    d = item.setdefault("details", {})
                    if conc > threshold:
                        item["score"] = round(max(0, float(item.get("score", 0) or 0) - 4.0), 2)
                        d["成交量质量"] = f"⚠ 集中度{conc:.0%}(>{threshold:.0%})，疑似刷量，降4分"
                    else:
                        d["成交量质量"] = f"✓ 正常(集中度{conc:.0%})"
            except: pass
        return results

    # ── 6. 相对BTC强弱 ──────────────────────────────────────────────────────
    def _apply_btc_relative_strength(self, results: List[Dict], symbols: List, btc_ctx: Dict) -> List[Dict]:
        """山寨弱于BTC时标记或降权"""
        if not bool(self.config.get("enable_btc_relative_strength", True)) or not btc_ctx:
            return results
        min_ratio = float(self.config.get("btc_rs_min_ratio", -0.3) or -0.3)
        btc_24h = btc_ctx.get("btc_24h_move", 0)
        sym_map = {str(getattr(s, "inst_id", "")): s for s in symbols}
        for item in results:
            sym_id = str(item.get("symbol", ""))
            if "BTC" in sym_id.upper():
                continue
            sym = sym_map.get(sym_id)
            if not sym:
                continue
            alt_24h = _safe_number(getattr(sym, "price_change_24h", 0))
            if abs(btc_24h) < 0.5:
                continue  # BTC 静止时不做比较
            rs = alt_24h - btc_24h  # 山寨涨跌 - BTC涨跌
            d = item.setdefault("details", {})
            direction = str(item.get("direction", "WAIT")).upper()
            if direction in {"BUY", "LONG"} and rs < min_ratio:
                penalty = min(8.0, abs(rs - min_ratio) * 2.0)
                item["score"] = round(max(0, float(item.get("score", 0) or 0) - penalty), 2)
                d["相对BTC"] = f"⚠ BTC+{btc_24h:.1f}% vs 山寨{alt_24h:+.1f}%(差{rs:.1f}%)，弱于大盘降{penalty:.1f}分"
            elif direction in {"SELL", "SHORT"} and rs > -min_ratio:
                d["相对BTC"] = f"山寨{alt_24h:+.1f}% vs BTC+{btc_24h:.1f}%(差{rs:.1f}%)，相对强势，注意空头风险"
            else:
                d["相对BTC"] = f"山寨{alt_24h:+.1f}% vs BTC+{btc_24h:.1f}%(差{rs:.1f}%)"

        return results

    # ═══════════════════════════════════════════════════════════════════════════
    # v4.3 交易员视角 — 1H 小时线突破检测
    # ═══════════════════════════════════════════════════════════════════════════

    def _h1_breakout_detect(self, symbol) -> Tuple[bool, float, str]:
        """
        检测1H级别突破：
          ① 当前收盘突破最近 N 根1H K线最高点
          ② 突破K线成交量显著放大(>1.35x均量)
          ③ K线收盘在K线上半部(强势收盘)
        返回: (是否突破, 置信度0~1, 诊断描述)
        """
        if not bool(self.config.get("enable_h1_breakout_detect", True)):
            return False, 0.0, "未启用"
        rows = self._get_h1_klines(symbol)
        lookback = int(self.config.get("h1_breakout_lookback", 12) or 12)
        vol_ratio = float(self.config.get("h1_breakout_vol_ratio", 1.35) or 1.35)
        close_pos = float(self.config.get("h1_breakout_close_position", 0.65) or 0.65)
        if len(rows) < lookback + 3:
            return False, 0.0, f"1H数据不足({len(rows)}根，需{lookback+3})"

        def rv(row, idx):
            try: return float(row[idx])
            except: return 0.0

        # 最近一根K线
        cur_open, cur_high, cur_low, cur_close, cur_vol = rv(rows[-1],1), rv(rows[-1],2), rv(rows[-1],3), rv(rows[-1],4), rv(rows[-1],5)

        # 前 lookback 根K线(不含最新一根)的最高点
        prev_rows = rows[-lookback-1:-1]
        prev_high = max((rv(r, 2) for r in prev_rows), default=0.0)
        if prev_high <= 0:
            return False, 0.0, "无有效历史高点"

        # ① 突破检测
        breakthrough = cur_close > prev_high * 1.001 and cur_high > prev_high
        if not breakthrough:
            return False, 0.0, f"收盘{cur_close:.6g}未突破{lookback}根高点{prev_high:.6g}"

        # ② 放量检测
        prev_vols = [rv(r, 5) for r in prev_rows[-8:] if rv(r, 5) > 0]
        avg_vol = sum(prev_vols) / len(prev_vols) if prev_vols else cur_vol
        vol_surge = cur_vol >= avg_vol * vol_ratio

        # ③ 强势收盘
        candle_range = cur_high - cur_low
        if candle_range > 0:
            position = (cur_close - cur_low) / candle_range
        else:
            position = 0.5
        strong_close = position >= close_pos and cur_close >= cur_open

        # 综合判断
        checks = [breakthrough, vol_surge, strong_close]
        passed = sum(checks)
        conf = passed / 3.0

        details = (
            f"突破{lookback}根高点{prev_high:.6g}→{cur_close:.6g}(+{(cur_close/prev_high-1)*100:.2f}%)"
            f" | 量{cur_vol:.0f}vs均{avg_vol:.0f}({cur_vol/avg_vol:.1f}x)"
            f" | 收盘位{position:.0%}"
            f" | 评分{passed}/3"
        )
        return True, conf, details

    def _h1_squeeze_detect(self, symbol) -> Tuple[bool, float, str]:
        """
        布林带压缩检测：带宽处于历史低位 → 即将突破
        """
        if not bool(self.config.get("enable_h1_squeeze_detect", True)):
            return False, 0.0, "未启用"
        rows = self._get_h1_klines(symbol)
        period = int(self.config.get("h1_squeeze_bb_period", 20) or 20)
        pctile = float(self.config.get("h1_squeeze_bb_width_percentile", 0.20) or 0.20)
        if len(rows) < period * 3:
            return False, 0.0, f"1H数据不足({len(rows)}根，需{period*3})"

        try:
            closes = []
            for r in rows[-period*3:]:
                c = float(r[4]) if float(r[4]) > 0 else None
                if c: closes.append(c)
            if len(closes) < period * 2:
                return False, 0.0, "有效收盘不足"

            s = pd.Series(closes)
            ma = s.rolling(period).mean()
            std = s.rolling(period).std()
            bbw = (std * 2) / ma  # 布林带宽

            cur_bbw = float(bbw.iloc[-1])
            if pd.isna(cur_bbw) or cur_bbw <= 0:
                return False, 0.0, "带宽计算异常"

            # 计算带宽在历史中的分位
            hist_bbw = bbw.dropna().values
            if len(hist_bbw) < 10:
                return False, 0.0, "带宽历史不足"
            rank = np.sum(hist_bbw <= cur_bbw) / len(hist_bbw)
            is_squeeze = rank <= pctile

            if not is_squeeze:
                return False, 0.0, f"带宽{cur_bbw:.4f}，历史分位{rank:.0%}，未压缩"

            # 方向判断：价格在均线上方=多头突破概率更高
            last_close = closes[-1]
            last_ma = float(ma.iloc[-1])
            bullish_bias = last_close > last_ma

            return True, 0.65 if bullish_bias else 0.45, (
                f"布林带宽压缩({cur_bbw:.4f}, 分位{rank:.1%})"
                f" | 价格在MA{'上' if bullish_bias else '下'}方"
            )
        except Exception as e:
            return False, 0.0, f"计算异常:{e}"

    def _h1_structure_score(self, symbol, direction: str) -> Tuple[float, str]:
        """
        1H 技术结构评分（0~100）:
          - 更高高点 + 更高低点 (上升趋势)
          - 多次测试支撑不破
          - 均线多头排列
        """
        if not bool(self.config.get("enable_h1_structure_score", True)):
            return 50.0, "未启用"
        rows = self._get_h1_klines(symbol)
        lookback = int(self.config.get("h1_structure_lookback", 24) or 24)
        if len(rows) < lookback + 5:
            return 50.0, f"1H数据不足({len(rows)}根)"
        direction_up = str(direction).upper() in {"BUY", "LONG"}

        try:
            closes = [float(r[4]) for r in rows[-lookback:] if float(r[4]) > 0]
            highs  = [float(r[2]) for r in rows[-lookback:] if float(r[2]) > 0]
            lows   = [float(r[3]) for r in rows[-lookback:] if float(r[3]) > 0]
            if len(closes) < 8:
                return 50.0, "有效K线不足"

            score = 50.0
            parts = []

            # ① HH/HL 结构（分成两段比较）
            half = len(highs) // 2
            if half >= 4:
                h1, h2 = highs[:half], highs[half:]
                l1, l2 = lows[:half], lows[half:]
                if direction_up:
                    if max(h2) > max(h1): score += 15; parts.append("更高高点✓")
                    else: parts.append("HH未确认")
                    if min(l2) > min(l1): score += 15; parts.append("更高低点✓")
                    else: parts.append("HL未确认")
                else:
                    if min(l2) < min(l1): score += 15; parts.append("更低低点✓")
                    else: parts.append("LL未确认")
                    if max(h2) < max(h1): score += 15; parts.append("更低高点✓")
                    else: parts.append("LH未确认")

            # ② 均线排列
            if len(closes) >= 50:
                s_closes = pd.Series(closes)
                ema12 = float(s_closes.ewm(span=12, adjust=False).mean().iloc[-1])
                ema26 = float(s_closes.ewm(span=26, adjust=False).mean().iloc[-1])
                ema50 = float(s_closes.ewm(span=50, adjust=False).mean().iloc[-1]) if len(closes) >= 50 else ema26
                if direction_up:
                    aligned = closes[-1] > ema12 > ema26
                    if aligned: score += 12; parts.append("EMA多头排列✓")
                    else: parts.append("EMA排列未完成")
                else:
                    aligned = closes[-1] < ema12 < ema26
                    if aligned: score += 12; parts.append("EMA空头排列✓")
                    else: parts.append("EMA排列未完成")

            # ③ 支撑/压力测试次数
            if direction_up:
                # 找最近低点，统计回踩不破次数
                recent_low = min(lows[-8:])
                bounce_count = sum(1 for l in lows[-12:] if abs(l - recent_low) / recent_low < 0.015)
                if bounce_count >= 3:
                    score += 8; parts.append(f"支撑测试{bounce_count}次✓")
                else:
                    parts.append(f"支撑测试{bounce_count}次")
            else:
                recent_high = max(highs[-8:])
                reject_count = sum(1 for h in highs[-12:] if abs(h - recent_high) / recent_high < 0.015)
                if reject_count >= 3:
                    score += 8; parts.append(f"压力测试{reject_count}次✓")
                else:
                    parts.append(f"压力测试{reject_count}次")

            return min(100.0, score), " | ".join(parts)
        except Exception as e:
            return 50.0, f"结构计算异常:{e}"

    def _apply_h1_breakout_scoring(self, results: List[Dict], symbols: List) -> List[Dict]:
        """
        统一应用1H突破检测：通过加分、预突破加分、结构评分
        分为 "确认突破" 和 "早期信号" 两个等级
        """
        if not bool(self.config.get("enable_h1_breakout_detect", True)):
            return results
        sym_map = {str(getattr(s, "inst_id", "")): s for s in symbols}
        breakout_bonus = float(self.config.get("h1_breakout_bonus", 8.0) or 8.0)
        early_bonus = float(self.config.get("h1_early_breakout_bonus", 4.0) or 4.0)
        squeeze_bonus = float(self.config.get("h1_squeeze_bonus", 4.0) or 4.0)
        structure_max = float(self.config.get("h1_structure_bonus_max", 5.0) or 5.0)

        for item in results:
            sym_id = str(item.get("symbol", ""))
            sym = sym_map.get(sym_id)
            if not sym:
                continue
            direction = str(item.get("direction", "WAIT")).upper()
            d = item.setdefault("details", {})

            # ① 突破检测
            is_breakout, conf, bk_detail = self._h1_breakout_detect(sym)
            if is_breakout:
                if conf >= 0.67:  # 2/3以上=确认突破
                    item["score"] = round(min(100, float(item.get("score", 0) or 0) + breakout_bonus), 2)
                    d["1H突破"] = f"✓ 确认突破(置信{conf:.0%}) +{breakout_bonus}分 | {bk_detail}"
                    item["_h1_breakout_level"] = "confirmed"
                else:
                    item["score"] = round(min(100, float(item.get("score", 0) or 0) + early_bonus), 2)
                    d["1H突破"] = f"▲ 早期突破(置信{conf:.0%}) +{early_bonus}分 | {bk_detail}"
                    item["_h1_breakout_level"] = "early"
            elif conf > 0:  # 突破已发生但检查未全通过
                # 放量不足或收盘位置不够：仍给一点分作为预突破信号
                item["score"] = round(min(100, float(item.get("score", 0) or 0) + 2.0), 2)
                d["1H突破"] = f"△ 潜在突破(待确认) +2分 | {bk_detail}"
                item["_h1_breakout_level"] = "potential"

            # ② 波动压缩检测
            is_squeeze, sq_conf, sq_detail = self._h1_squeeze_detect(sym)
            if is_squeeze:
                item["score"] = round(min(100, float(item.get("score", 0) or 0) + squeeze_bonus * sq_conf), 2)
                d["波动压缩"] = f"✓ 布林带压缩(置信{sq_conf:.0%}) +{squeeze_bonus*sq_conf:.1f}分 | {sq_detail}"
                # 压缩+突破=最强信号
                if is_breakout:
                    item["score"] = round(min(100, float(item.get("score", 0) or 0) + 2.0), 2)
                    d["波动压缩"] = d.get("波动压缩","") + " [+共振加成2分]"

            # ③ 结构评分
            struct_score, struct_detail = self._h1_structure_score(sym, direction)
            if struct_score > 50:
                bonus = (struct_score - 50) / 50 * structure_max
                item["score"] = round(min(100, float(item.get("score", 0) or 0) + bonus), 2)
                d["1H结构"] = f"评分{struct_score:.0f}/100 +{bonus:.1f}分 | {struct_detail}"

            # ④ 如果1H突破已确认，放宽3m过滤要求
            if item.get("_h1_breakout_level") in ("confirmed",):
                item["_relax_3m"] = True
                d["3m过滤"] = "1H已确认突破，3m要求放宽"

        return results

    # ── v4.4: 共识层市场状态推断（统一用 _state_adjustment，不再用硬编码 STATE_WEIGHTS）──
    def _infer_consensus_state(self, items: List[Dict]) -> str:
        """从子策略结果中提取市场状态，无标记时返回 neutral"""
        for it in items:
            st = str(it.get("market_state", it.get("state_mode", ""))).strip().lower()
            if st in {"trend", "range", "volatile"}:
                return st
        return "neutral"

    # ── 统一应用所有交易视角过滤器 ──────────────────────────────────────────
    def _apply_trading_filters(self, results: List[Dict], symbols: List) -> List[Dict]:
        """按顺序应用所有v4.2交易视角过滤器（v4.4: 加分上限保护）"""
        btc_ctx = self._get_btc_context(symbols)
        # 记录每个结果的原始分，后续加分项总上限 15 分
        orig_scores = {id(r): float(r.get("score", 0) or 0) for r in results}
        results = self._apply_h1_breakout_scoring(results, symbols)
        results = self._apply_btc_correlation_filter(results, btc_ctx)
        results = self._apply_funding_filter(results, symbols)
        results = self._apply_confluence_scoring(results, symbols)
        results = self._apply_atr_stop_suggestion(results, symbols)
        results = self._apply_volume_quality_check(results, symbols)
        results = self._apply_btc_relative_strength(results, symbols, btc_ctx)
        # 加分封顶：总分最多比原始分高 18 分
        # 扣分下限：总分最多比原始分低 20 分（防止多项惩罚叠加淹没高质量信号）
        MAX_ADD = 18.0
        MAX_DED = 20.0
        for r in results:
            orig = orig_scores.get(id(r), float(r.get("score", 0) or 0))
            current = float(r.get("score", 0) or 0)
            if current > orig + MAX_ADD:
                r["score"] = round(orig + MAX_ADD, 2)
                r.setdefault("details", {})["加分封顶"] = f"原始{orig:.1f}→封顶{orig+MAX_ADD:.1f}"
            elif current < orig - MAX_DED:
                r["score"] = round(orig - MAX_DED, 2)
                r.setdefault("details", {})["扣分下限"] = f"原始{orig:.1f}→下限{orig-MAX_DED:.1f}(惩罚已封顶)"
        return results

    # ── 硬性过滤：辅助工具 ──────────────────────────────────────────────────
    def _get_m3_klines(self, symbol) -> List:
        """从 symbol.extra_data 中取出3m K线列表，格式：[ts, open, high, low, close, vol, ...]"""
        try:
            ed = getattr(symbol, "extra_data", None) or {}
            klines = ed.get("klines", {}) if isinstance(ed, dict) else {}
            rows = klines.get("3m") or klines.get("3M") or []
            return rows if isinstance(rows, (list, tuple)) else []
        except Exception:
            return []

    def _get_h1_klines(self, symbol) -> List:
        """从 symbol.extra_data 中取出1H K线列表。"""
        try:
            ed = getattr(symbol, "extra_data", None) or {}
            klines = ed.get("klines", {}) if isinstance(ed, dict) else {}
            rows = (klines.get("1H") or klines.get("1h")
                    or klines.get("60m") or klines.get("60M") or [])
            return rows if isinstance(rows, (list, tuple)) else []
        except Exception:
            return []

    # ── 硬性检查 1：3m 回调、不跌破突破点、企稳 ─────────────────────────────
    def _m3_pullback_hard_check(self, m3_rows: List, direction: str, relax: bool = False) -> Tuple[bool, str]:
        """
        三合一硬性检测（LONG 示例，SHORT 镜像）：
          ① 3m 存在真实脉冲高点（lookback 内）
          ② 当前回调幅度在 [min_pct, max_pct] 区间内
          ③ 企稳阶段（最后 stab_bars 根）最低收盘价不低于脉冲起点（突破点）
             —— 即"不跌破原突破点"（允许 m3_no_break_tolerance_pct 容差）
          ④ 企稳形态：区间紧缩 OR 末棒方向正确（阳线/阴线）
        SHORT 镜像：脉冲低点 → 反弹 → 不突破下跌起点 → 区间/末棒
        """
        lookback     = max(8, int(self.config.get("m3_impulse_lookback_bars", 15) or 15))
        min_pct      = float(self.config.get("m3_pullback_min_pct",      0.50) or 0.50)
        max_pct      = float(self.config.get("m3_pullback_max_pct",      2.20) or 2.20)
        stab_bars    = max(2, int(self.config.get("m3_stabilization_bars", 4) or 4))
        no_brk_tol   = float(self.config.get("m3_no_break_tolerance_pct", 0.30) or 0.30) / 100.0

        # v4.3: 1H突破已确认时放宽3m要求
        if relax:
            min_pct *= 0.5      # 最低回调从0.5%→0.25%
            max_pct *= 1.5      # 最高回调从2.2%→3.3%
            no_brk_tol *= 2.0   # 突破点容差翻倍

        if not m3_rows or len(m3_rows) < lookback + stab_bars:
            return False, f"3m数据不足({len(m3_rows)}根，需{lookback + stab_bars}根)"

        rows = list(m3_rows)[-(lookback + stab_bars):]

        def _val(row, idx, default=0.0):
            try: return float(row[idx])
            except (TypeError, ValueError, IndexError): return default

        # 诊断用数据摘要
        try:
            data_range = f"最新={_val(rows[-1],4):.6g} 最高={max(_val(r,2) for r in rows):.6g} 最低={min(_val(r,3) for r in rows):.6g}"
        except Exception:
            data_range = "解析失败"

        direction_up = str(direction).upper() in {"BUY", "LONG"}
        body_rows = rows[:lookback]   # 脉冲主体段
        tail_rows = rows[lookback:]   # 企稳候选段

        if direction_up:
            # ① 找脉冲高点及其位置
            peak_high = max((_val(r, 2) for r in body_rows), default=0.0)
            if peak_high <= 0:
                return False, f"3m无有效脉冲高点 [{data_range}]"

            peak_idx = max(range(len(body_rows)),
                           key=lambda i: _val(body_rows[i], 2))

            # ③ 突破起点 = 脉冲高点出现前最低收盘（脉冲的起跳平台）
            pre_peak_closes = [_val(body_rows[i], 4) for i in range(peak_idx)]
            breakout_base   = min((c for c in pre_peak_closes if c > 0), default=0.0)

            # ② 当前企稳段末棒收盘价
            cur_close = _val(tail_rows[-1], 4) if tail_rows else 0.0
            if cur_close <= 0:
                return False, "3m末棒价格无效"

            pullback_pct = (peak_high - cur_close) / peak_high * 100.0
            if pullback_pct < min_pct:
                return False, f"3m回调幅度不足({pullback_pct:.2f}%<{min_pct}%)"
            if pullback_pct > max_pct:
                return False, f"3m回调过深({pullback_pct:.2f}%>{max_pct}%，可能趋势反转)"

            # ③ 企稳段任意收盘不得低于突破起点（含容差）
            if breakout_base > 0:
                floor = breakout_base * (1.0 - no_brk_tol)
                tail_closes = [_val(r, 4) for r in tail_rows]
                min_tail_close = min((c for c in tail_closes if c > 0), default=cur_close)
                if min_tail_close < floor:
                    depth = (floor - min_tail_close) / floor * 100
                    return False, (
                        f"3m回调跌破突破起点"
                        f"(最低收盘{min_tail_close:.4f} < 突破点{breakout_base:.4f}"
                        f"-容差，低{depth:.2f}%)")

            # ④ 企稳形态
            tail_highs  = [_val(r, 2) for r in tail_rows]
            tail_lows   = [_val(r, 3) for r in tail_rows]
            range_pct   = (max(tail_highs) - min(tail_lows)) / max(peak_high, 1e-9) * 100.0
            last_bull   = _val(tail_rows[-1], 4) >= _val(tail_rows[-1], 1)
            tight_range = range_pct <= max_pct * 0.60
            if not (tight_range or last_bull):
                return False, (
                    f"3m未企稳(波动{range_pct:.2f}%，"
                    f"末棒{'阳' if last_bull else '阴'}，需紧缩或阳线收盘)")

            stab_desc = "区间紧缩" if tight_range else "末棒阳线"
            base_desc  = f"，突破点{breakout_base:.4f}" if breakout_base > 0 else ""
            return True, (
                f"3m多头回调企稳✓ 回调{pullback_pct:.2f}%{base_desc}，{stab_desc}")

        else:  # SHORT 镜像
            # ① 找脉冲低点
            trough_low = min(
                (_val(r, 3) for r in body_rows if _val(r, 3) > 0), default=0.0)
            if trough_low <= 0:
                return False, "3m无有效脉冲低点"

            trough_idx = min(range(len(body_rows)),
                             key=lambda i: _val(body_rows[i], 3) or float("inf"))

            # ③ 突破起点 = 脉冲低点出现前最高收盘（下跌起点的压力位）
            pre_trough_closes = [_val(body_rows[i], 4) for i in range(trough_idx)]
            breakout_base     = max((c for c in pre_trough_closes if c > 0), default=0.0)

            cur_close = _val(tail_rows[-1], 4) if tail_rows else 0.0
            if cur_close <= 0:
                return False, "3m末棒价格无效"

            bounce_pct = (cur_close - trough_low) / trough_low * 100.0
            if bounce_pct < min_pct:
                return False, f"3m反弹幅度不足({bounce_pct:.2f}%<{min_pct}%)"
            if bounce_pct > max_pct:
                return False, f"3m反弹过高({bounce_pct:.2f}%>{max_pct}%，可能趋势反转)"

            # ③ 企稳段任意收盘不得高于突破起点（含容差）
            if breakout_base > 0:
                ceiling = breakout_base * (1.0 + no_brk_tol)
                tail_closes = [_val(r, 4) for r in tail_rows]
                max_tail_close = max((c for c in tail_closes if c > 0), default=cur_close)
                if max_tail_close > ceiling:
                    height = (max_tail_close - ceiling) / ceiling * 100
                    return False, (
                        f"3m反弹突破下跌起点"
                        f"(最高收盘{max_tail_close:.4f} > 压力点{breakout_base:.4f}"
                        f"+容差，高{height:.2f}%)")

            # ④ 企稳形态
            tail_highs  = [_val(r, 2) for r in tail_rows]
            tail_lows   = [_val(r, 3) for r in tail_rows]
            range_pct   = (max(tail_highs) - min(tail_lows)) / max(cur_close, 1e-9) * 100.0
            last_bear   = _val(tail_rows[-1], 4) <= _val(tail_rows[-1], 1)
            tight_range = range_pct <= max_pct * 0.60
            if not (tight_range or last_bear):
                return False, (
                    f"3m未企稳(波动{range_pct:.2f}%，"
                    f"末棒{'阴' if last_bear else '阳'}，需紧缩或阴线收盘)")

            stab_desc = "区间紧缩" if tight_range else "末棒阴线"
            base_desc  = f"，压力点{breakout_base:.4f}" if breakout_base > 0 else ""
            return True, (
                f"3m空头反弹企稳✓ 反弹{bounce_pct:.2f}%{base_desc}，{stab_desc}")

    # ── 硬性检查 2：小时线趋势延续（EMA 排列） ───────────────────────────────
    def _h1_trend_check(self, h1_rows: List, direction: str) -> Tuple[bool, str]:
        """
        验证 H1 趋势仍在延续，不允许已出现趋势反转。
        用 EMA(fast) vs EMA(slow) 判断：
          LONG  → EMA(fast) > EMA(slow)，且最近2根H1收盘未连续低于EMA(fast)
          SHORT → EMA(fast) < EMA(slow)，且最近2根H1收盘未连续高于EMA(fast)
        数据不足时宽松处理（返回True），不因数据缺失误杀。
        """
        fast_span = max(3, int(self.config.get("h1_ema_fast", 12) or 12))
        slow_span = max(fast_span + 1, int(self.config.get("h1_ema_slow", 26) or 26))
        min_bars  = slow_span + 5

        if not h1_rows or len(h1_rows) < min_bars:
            return True, f"1H数据不足({len(h1_rows)}根)，跳过趋势验证"

        closes: List[float] = []
        for row in h1_rows:
            try:
                c = float(row[4])
                if np.isfinite(c) and c > 0:
                    closes.append(c)
            except (TypeError, ValueError, IndexError):
                pass

        if len(closes) < min_bars:
            return True, "1H有效收盘数据不足，跳过趋势验证"

        # 计算最终 EMA（Wilder 式指数平滑）
        def _ema_final(data: List[float], span: int) -> float:
            k, val = 2.0 / (span + 1), data[0]
            for v in data[1:]:
                val = v * k + val * (1.0 - k)
            return val

        ema_fast = _ema_final(closes, fast_span)
        ema_slow = _ema_final(closes, slow_span)
        last2    = closes[-2:]  # 最近两根 H1 收盘

        direction_up = str(direction).upper() in {"BUY", "LONG"}

        if direction_up:
            if ema_fast < ema_slow:
                gap = (ema_slow - ema_fast) / max(ema_slow, 1e-9) * 100
                return False, (
                    f"1H趋势已转空(EMA{fast_span} < EMA{slow_span}，"
                    f"偏离{gap:.2f}%)，多头信号无效")
            # 检测连续反转迹象：最近2根H1全部收盘低于快线EMA
            if len(last2) >= 2 and all(c < ema_fast for c in last2):
                return False, (
                    f"1H多头趋势减弱：最近2根收盘({last2[-1]:.4f})均低于EMA{fast_span}"
                    f"({ema_fast:.4f})，趋势动能不足")
            gap = (ema_fast - ema_slow) / max(ema_slow, 1e-9) * 100
            return True, f"1H多头趋势延续✓ EMA{fast_span}>{slow_span}，领先{gap:.2f}%"
        else:
            if ema_fast > ema_slow:
                gap = (ema_fast - ema_slow) / max(ema_fast, 1e-9) * 100
                return False, (
                    f"1H趋势已转多(EMA{fast_span} > EMA{slow_span}，"
                    f"偏离{gap:.2f}%)，空头信号无效")
            if len(last2) >= 2 and all(c > ema_fast for c in last2):
                return False, (
                    f"1H空头趋势减弱：最近2根收盘({last2[-1]:.4f})均高于EMA{fast_span}"
                    f"({ema_fast:.4f})，趋势动能不足")
            gap = (ema_slow - ema_fast) / max(ema_slow, 1e-9) * 100
            return True, f"1H空头趋势延续✓ EMA{fast_span}<{slow_span}，领先{gap:.2f}%"

    # ── 综合应用两项硬性过滤 ─────────────────────────────────────────────────
    def _apply_m3_hard_filter(self, result: Dict[str, Any], symbol) -> Dict[str, Any]:
        """
        同时执行两项硬性过滤，任一不通过则 passed=False：
          1. 3m 回调 + 不跌破突破点 + 企稳
          2. 1H 趋势仍在延续（EMA 排列）
        """
        if not bool(self.config.get("m3_hard_filter", True)):
            return result
        if not result.get("passed"):
            return result
        direction = str(result.get("direction", "WAIT")).upper()
        if direction not in {"BUY", "SELL"}:
            return result

        m3_rows = self._get_m3_klines(symbol)
        h1_rows = self._get_h1_klines(symbol)

        relax_m3 = bool(result.get("_relax_3m", False))
        m3_ok, m3_reason = self._m3_pullback_hard_check(m3_rows, direction, relax=relax_m3)
        if relax_m3 and not m3_ok:
            m3_reason += " (1H已突破，3m放宽后仍不通过)"
        h1_ok, h1_reason = (
            self._h1_trend_check(h1_rows, direction)
            if bool(self.config.get("h1_trend_hard_filter", True))
            else (True, "1H过滤已关闭")
        )

        result = dict(result)
        details = dict(result.get("details") or {})

        if m3_ok and h1_ok:
            details["3m硬性过滤"] = f"通过: {m3_reason}"
            details["1H趋势过滤"] = f"通过: {h1_reason}"
            result["details"] = details
            return result

        # 任一不通过
        result["passed"] = False
        fail_parts = []
        if not m3_ok:
            details["3m硬性过滤"] = f"未通过: {m3_reason}"
            fail_parts.append(f"[3m] {m3_reason}")
        else:
            details["3m硬性过滤"] = f"通过: {m3_reason}"
        if not h1_ok:
            details["1H趋势过滤"] = f"未通过: {h1_reason}"
            fail_parts.append(f"[1H] {h1_reason}")
        else:
            details["1H趋势过滤"] = f"通过: {h1_reason}"

        result["details"] = details
        sigs = list(result.get("signals") or [])
        sigs.append("[硬性过滤淘汰] " + " | ".join(fail_parts))
        result["signals"] = sigs
        return result

    # ── 子策略加载（v2: 多路径查找）─────────────────────────────────────────
    def _build_child_strategies(self):
        strategies = []
        search_dirs = self._strategy_search_dirs()
        for sn, filename, class_name, priority in self.CHILDREN:
            module_path = self._find_strategy_file(filename, search_dirs)
            if module_path is None:
                logger.warning(f"[组合] 跳过 {sn}: 在以下路径均未找到 {filename}\n  {[str(d) for d in search_dirs]}")
                continue
            spec = importlib.util.spec_from_file_location(f"ai_combo_{class_name}", str(module_path))
            if spec is None or spec.loader is None:
                logger.warning(f"[组合] 跳过 {sn}: spec 创建失败")
                continue
            module = importlib.util.module_from_spec(spec)
            try:
                sys.modules[spec.name] = module
                spec.loader.exec_module(module)
            except Exception as e:
                logger.error(f"[组合] 跳过 {sn}: 加载失败 {e}")
                continue
            strategy_class = (getattr(module, class_name, None)
                              or getattr(module, "STRATEGY_CLASS", None))
            if strategy_class is None:
                logger.warning(f"[组合] 跳过 {sn}: 缺少策略类 {class_name}")
                continue
            try:
                child_cfg = self._child_config(sn)
                strategies.append((sn, strategy_class(child_cfg), priority))
                logger.info(f"[组合] 已加载 {sn} ({module_path.name})")
            except Exception as e:
                logger.error(f"[组合] 跳过 {sn}: 初始化失败 {e}")
        return strategies

    def _strategy_search_dirs(self) -> List[Path]:
        """按优先级返回搜索目录列表。"""
        dirs = []
        # 1. 当前文件同级目录
        try:
            dirs.append(Path(__file__).resolve().parent)
        except NameError:
            dirs.append(Path.cwd())
        # 2. strategies/ 子目录
        for base in list(dirs):
            candidate = base / "strategies"
            if candidate.is_dir():
                dirs.append(candidate)
        # 3. 当前工作目录
        cwd = Path.cwd()
        if cwd not in dirs:
            dirs.append(cwd)
        if (cwd / "strategies").is_dir():
            dirs.append(cwd / "strategies")
        return dirs

    def _find_strategy_file(self, filename: str, search_dirs: List[Path]) -> Optional[Path]:
        """在所有搜索目录中找到第一个匹配的文件。"""
        for d in search_dirs:
            p = d / filename
            if p.exists():
                return p
        return None

    # 子策略允许透传的键白名单（避免组合层 70+ 参数污染子策略命名空间）
    _CHILD_SAFE_KEYS = {
        "min_volume_24h", "min_score", "position_size", "allow_short", "max_atr_pct",
        # 时效性
        "max_h1_trend_age", "h1_trend_age_penalty", "max_m3_staleness_bars",
        "m3_freshness_penalty", "bonus_freshness_score",
        # 3m 回调企稳
        "require_m3_pullback_confirmation", "m3_pullback_min_pct", "m3_pullback_max_pct",
        "m3_stabilization_bars", "require_m3_freshness", "m3_min_impulse_pct",
        "vol_continuation_min_ratio",
        # 微观结构
        "enable_atr_squeeze_check", "atr_squeeze_ratio",
        "enable_volume_delta_check", "volume_delta_min_ratio",
        "enable_vwap_alignment_check",
        # 截面加速
        "use_dynamic_ic_weights", "ic_weight_blend",
        "use_orthogonalization", "enable_mfin_interactions",
        "enable_llm_factors", "enable_on_chain",
        "enable_early_trend_factors", "early_trend_min_trigger",
    }

    def _child_config(self, strategy_name: str) -> Dict[str, Any]:
        """
        仅透传白名单内的安全参数给子策略，避免参数名冲突。
        截面子策略加速模式时可选关闭 m3 检查。
        """
        cfg = {k: self.config[k] for k in self._CHILD_SAFE_KEYS if k in self.config}
        if strategy_name == "截面多因子" and bool(self.config.get("accelerate_cross_section_child", True)):
            cfg["use_dynamic_ic_weights"] = False
            cfg["ic_weight_blend"] = 0.0
            if bool(self.config.get("accel_disable_m3", False)):
                cfg["require_m3_pullback_confirmation"] = False
        return cfg

    # ── 市场状态推断 ──────────────────────────────────────────────────────────
    def _normalize_scores_cross_sectional(self, items: List[Dict]) -> None:
        """跨引擎 z-score 归一化：让不同评分体系的引擎结果可比"""
        if not items or not bool(self.config.get("enable_score_normalization", True)):
            return
        scores = [float(it.get("score", 0) or 0) for it in items]
        mean_s = np.mean(scores) if scores else 50.0
        std_s = max(np.std(scores), 1e-6)
        for it in items:
            raw = float(it.get("score", 0) or 0)
            norm = (raw - mean_s) / std_s * 15.0 + 50.0
            it["_norm_score"] = round(max(0.0, min(100.0, norm)), 2)

    def _get_engine_weight(self, engine_name: str, direction: str) -> float:
        """获取引擎当前胜率权重（0.5~1.5），无历史记录时返回 1.0"""
        if not bool(self.config.get("enable_engine_track_record", True)):
            return 1.0
        track = self._engine_track.get(engine_name, {}).get(direction, [])
        if len(track) < 5:
            return 1.0
        wins = sum(1 for p in track if p > 0)
        wr = wins / len(track)
        return max(0.5, min(1.5, 0.5 + wr))

    def _record_engine_performance(self, engine_name: str, direction: str, pnl: float) -> None:
        """记录引擎信号的实际绩效，expire 旧记录"""
        if not bool(self.config.get("enable_engine_track_record", True)):
            return
        decay = float(self.config.get("engine_weight_decay", 0.85) or 0.85)
        track = self._engine_track.setdefault(engine_name, {}).setdefault(direction, [])
        track.append(pnl)
        # 保持最新 N 条
        while len(track) > self._max_perf_entries:
            track.pop(0)

    def _record_signal_performance(self, symbol: str, direction: str, score: float,
                                    pnl: Optional[float]) -> None:
        """记录单个币种的信号→PnL，供前端展示历史绩效"""
        cache = self._performance_cache.setdefault(symbol, [])
        cache.append((direction, score, pnl))
        while len(cache) > self._max_perf_entries:
            cache.pop(0)

    def _get_signal_performance_label(self, symbol: str, direction: str) -> str:
        """生成历史绩效标签文字"""
        cache = self._performance_cache.get(symbol, [])
        if not cache:
            return "无历史记录"
        recent = [r for r in cache[-10:] if r[0] == direction]
        if not recent:
            return "无同向历史记录"
        wins = sum(1 for r in recent if (r[2] or 0) > 0)
        pnls = [r[2] for r in recent if r[2] is not None]
        avg_pnl = sum(pnls) / len(pnls) * 100 if pnls else 0.0
        return f"近{len(recent)}次同向: {wins}赢{len(recent)-wins}亏, 均{avg_pnl:+.2f}%"
    def _state_adjustment(self, sn: str, state: str) -> Tuple[float, float]:
        table = {
            "trend":    {"DRL小时趋势启动": (2.8,1.03), "AI因子挖掘": (1.5,1.01), "截面多因子": (0.8,1.00),
                         "XGBoost截面排序": (1.2,1.01), "AI订单流动量": (1.0,1.00)},
            "range":    {"截面多因子": (2.1,1.02),       "AI因子挖掘": (1.2,1.01), "DRL小时趋势启动": (-1.8,0.98),
                         "XGBoost截面排序": (1.5,1.01), "AI订单流动量": (0.5,1.00)},
            "volatile": {"AI因子挖掘": (0.9,1.00),       "截面多因子": (0.6,1.00), "DRL小时趋势启动": (-1.2,0.97),
                         "AI订单流动量": (1.2,1.01), "XGBoost截面排序": (0.7,1.00)},
            "neutral":  {"DRL小时趋势启动": (0.4,1.00),  "AI因子挖掘": (0.4,1.00), "截面多因子": (0.4,1.00),
                         "XGBoost截面排序": (0.4,1.00), "AI订单流动量": (0.4,1.00)},
        }
        return table.get(str(state).lower(), {}).get(sn, (0.0, 1.0))

    def _normalize_state_mode(self, state_mode) -> str:
        mode = str(state_mode or "auto").strip().lower()
        return mode if mode in {"auto","trend","range","volatile","neutral"} else "auto"

    def _infer_market_state_from_data(self, data) -> str:
        if isinstance(data, dict):
            direct = str(data.get("market_state", data.get("state_mode",""))).strip().lower()
            if direct in {"trend","range","volatile","neutral"}:
                return direct
        closes = self._extract_closes_from_backtest_data(data)
        if len(closes) < 40:
            return "neutral"
        series = pd.Series(closes, dtype=float).replace([np.inf,-np.inf], np.nan).dropna()
        if len(series) < 40:
            return "neutral"
        fast = float(series.ewm(span=12, adjust=False).mean().iloc[-1])
        slow = float(series.ewm(span=34, adjust=False).mean().iloc[-1])
        last = max(abs(float(series.iloc[-1])), 1e-9)
        gap_pct = abs(fast - slow) / last * 100.0
        lb = min(24, len(series) - 1)
        dir_move = abs((float(series.iloc[-1]) / max(float(series.iloc[-lb-1]),1e-9) - 1.0)*100.0) if lb > 0 else 0.0
        rv = float(np.log(series).diff().dropna().tail(min(24,len(series)-1)).std() * np.sqrt(24) * 100.0)
        if rv >= 6.2: return "volatile"
        if gap_pct >= 1.25 and dir_move >= 1.6: return "trend"
        if gap_pct <= 0.85 and dir_move <= 1.2 and rv <= 5.4: return "range"
        return "neutral"

    def _extract_closes_from_backtest_data(self, data) -> List[float]:
        rows: List = []
        if isinstance(data, dict):
            km = data.get("klines_map") if isinstance(data.get("klines_map"), dict) else {}
            for key in ("1H","1h","60m","60M","15m","15M"):
                if key in km and isinstance(km.get(key), list) and km.get(key):
                    rows = list(km.get(key) or []); break
            if not rows and isinstance(data.get("klines"), list): rows = list(data.get("klines") or [])
            if not rows and isinstance(data.get("bars"), list): rows = list(data.get("bars") or [])
        elif isinstance(data, (list, tuple)): rows = list(data)
        closes = []
        for row in rows:
            val = row.get("c", row.get("close")) if isinstance(row, dict) else (row[4] if isinstance(row,(list,tuple)) and len(row)>=5 else row)
            try:
                c = float(val)
                if np.isfinite(c): closes.append(c)
            except (TypeError, ValueError): pass
        return closes

    # ── 回测对比 ──────────────────────────────────────────────────────────────
    def run_state_backtest_compare(self, dataset, *a, **kw) -> Dict[str, Any]:
        if isinstance(dataset, dict) and isinstance(dataset.get("samples"), list):
            samples = list(dataset.get("samples") or [])
        elif isinstance(dataset, (list,tuple)): samples = list(dataset)
        else: samples = [dataset]
        samples = [s for s in samples if s is not None]
        modes = ["auto","trend","range","volatile"]
        compare = {m: self._simulate_state_mode(samples, m, *a, **kw) for m in modes}
        best_mode = max(compare, key=lambda m: (
            float(compare[m].get("total_return_pct") or -1e9) if compare[m].get("total_return_pct") is not None
            else float(compare[m].get("avg_signal_score", 0) or 0)
        ))
        return {"type":"state_mode_backtest_compare","sample_count":len(samples),"compare":compare,"best_mode":best_mode}

    def _simulate_state_mode(self, samples, mode, *a, **kw):
        trades=wins=losses=0; score_sum=0.0; consensus_hits=0; rets=[]
        for sample in samples:
            sig = self.generate_signal(sample, *a, state_mode=mode, **kw)
            if not sig: continue
            if str(sig.get("action","")).upper() not in {"BUY","SHORT"}: continue
            trades += 1
            score_sum += float(sig.get("score",0) or 0)
            if int(sig.get("consensus_engines",0) or 0) >= int(self.config.get("min_consensus_engines",2) or 2):
                consensus_hits += 1
            ret = self._extract_realized_return(sample, sig)
            if ret is None: continue
            rets.append(ret)
            if ret > 0: wins += 1
            elif ret < 0: losses += 1
        avg = score_sum/trades if trades else 0.0
        wr = (wins/(wins+losses)*100) if (wins+losses) else None
        if rets:
            eq=peak=1.0; mdd=0.0
            for r in rets:
                eq*=(1+r); peak=max(peak,eq)
                mdd=max(mdd,(peak-eq)/peak if peak>0 else 0)
            tr=(eq-1)*100
        else: tr=mdd=None
        return {"mode":mode,"trades":trades,"wins":wins,"losses":losses,
                "win_rate_pct":round(float(wr),2) if wr is not None else None,
                "avg_signal_score":round(float(avg),2),
                "consensus_hits":consensus_hits,
                "consensus_ratio_pct":round(consensus_hits/trades*100,2) if trades else 0.0,
                "total_return_pct":round(float(tr),2) if tr is not None else None,
                "max_drawdown_pct":round(float(mdd)*100,2) if mdd is not None else None}

    def _extract_realized_return(self, sample, signal) -> Optional[float]:
        if not isinstance(sample, dict): return None
        candidates = [sample.get(k) for k in ("future_return","future_return_1h","next_return","label_return","target_return","ret_1h")]
        labels = sample.get("labels") if isinstance(sample.get("labels"), dict) else {}
        candidates += [labels.get(k) for k in ("future_return","future_return_1h","next_return","ret_1h")]
        ret = None
        for v in candidates:
            try:
                n = float(v)
                if np.isfinite(n): ret = n; break
            except (TypeError, ValueError): pass
        if ret is None: return None
        # 格式归一化：如果返回值为整数>100（如120=120%）或小数>10（如12.0=1200%），除以100
        # 如果值在 (2.0, 100.0] 之间，很可能是百分数格式
        # 如果值 <= 2.0，视为小数格式
        if abs(ret) > 100.0:
            ret /= 100.0
        elif abs(ret) > 2.0:
            # 50/50 猜测：2.0~100 区间的值可以是 2.5% 也可以是 250%
            # 用 magnitude check：如果值 > 10 大概率是百分数（单期回报10%以上罕见）
            if abs(ret) > 10.0:
                ret /= 100.0
        if str(signal.get("action","")).upper() in {"SHORT","SELL"}: ret = -ret
        return max(-0.99, min(10.0, ret))  # 放宽上下限到 -99%/+1000%


# ══════════════════════════════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════════════════════════════

def _result_sort_key(item: Dict[str, Any]) -> Tuple[float, float, float, float, str]:
    return (
        float(item.get("group_sort_score", 0) or 0),
        float(item.get("opportunity_score", item.get("score", 0)) or 0),
        float(item.get("score", 0) or 0),
        float(item.get("volume_24h", 0) or 0),
        str(item.get("symbol", "")),  # 确定性 tiebreaker
    )

def _dedupe_results(results: List[Dict[str, Any]], dedupe_by_symbol: bool = True) -> List[Dict[str, Any]]:
    best: Dict = {}
    for item in results:
        sym = str(item.get("symbol",""))
        key = sym if dedupe_by_symbol else (sym, str(item.get("source_strategy","")))
        ex = best.get(key)
        if ex is None:
            best[key] = item
            continue
        new_key = _result_sort_key(item)
        old_key = _result_sort_key(ex)
        if new_key > old_key:
            best[key] = item
        elif new_key == old_key:
            # 同分时：优先保留共识结果（多引擎共振 > 单引擎个体）
            is_new_consensus = "共振" in str(item.get("category", ""))
            is_old_consensus = "共振" in str(ex.get("category", ""))
            if is_new_consensus and not is_old_consensus:
                best[key] = item
    return list(best.values())

def _safe_number(value, default: float = 0.0) -> float:
    try:
        n = float(value)
    except (TypeError, ValueError):
        return default
    return n if np.isfinite(n) else default

def _recent_kline_move_pct(rows, lookback: int = 6) -> float:
    if not rows or not isinstance(rows, (list,tuple)) or len(rows) < 2:
        return 0.0
    try:
        end = float(rows[-1][4]); start = float(rows[max(0,len(rows)-lookback-1)][4])
    except (TypeError, ValueError, IndexError):
        return 0.0
    if not np.isfinite(end) or not np.isfinite(start) or start <= 0:
        return 0.0
    return (end/start - 1.0)*100.0

def _first_number(items: List[Dict], key: str) -> float:
    for item in items:
        try:
            v = float(item.get(key, 0) or 0)
            if v: return v
        except (TypeError, ValueError): pass
    return 0.0

def _merge_ranking_factors(items: List[Dict], fallback: float) -> Dict[str, float]:
    keys = ["trend","trigger","volume","location","freshness","risk"]
    merged = {}
    for key in keys:
        vals = []
        for item in items:
            f = item.get("ranking_factors") or {}
            try: vals.append(float(f.get(key, fallback)))
            except (TypeError, ValueError): pass
        merged[key] = sum(vals)/len(vals) if vals else fallback
    return merged


STRATEGY_NAME = "AI截面五引擎组合扫描器"
STRATEGY_TYPE = "scan"
STRATEGY_CLASS = AICrossSectionDualFactorComboScanner
BACKTEST_CLASS = AICrossSectionDualFactorComboScanner
