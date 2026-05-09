#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
趋势延续·挤压突破前扫描器  v4.1
=====================================
核心目标：在日线或小时线趋势「尚未完全启动」的时刻提前埋伏，
而不是等趋势跑远后才追进。

v4.1 相对 v3.0 的重大改进（在 v3.0 基础上累积）：
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
    "min_score":                    {"type": "float", "default": 60.0,          "label": "最低综合输出分数（v4.4: 52→60，减少假信号）"},
    "top_n":                        {"type": "int",   "default": 6,             "label": "最多输出信号数（v4.4: 12→6，只保留最优信号）"},
    "min_d1_score":                 {"type": "float", "default": 45.0,          "label": "日线子分最低门槛（低于此直接淘汰，保证日线趋势质量）"},
    "min_h1_score":                 {"type": "float", "default": 40.0,          "label": "H1子分最低门槛（低于此直接淘汰，保证挤压+量能质量）"},
    "allow_short":                  {"type": "bool",  "default": True,          "label": "允许空头方向信号"},

    # ── 日线趋势萌芽检测（v3.3进一步放宽以提高命中率）──
    "d1_adx_period":                {"type": "int",   "default": 14,            "label": "日线ADX计算周期"},
    "d1_adx_min":                   {"type": "float", "default": 17.0,          "label": "日线ADX最低阈值（v4.4: 15→17，进一步过滤无方向性震荡）"},
    "d1_adx_strong":                {"type": "float", "default": 28.0,          "label": "日线ADX强趋势阈值（达此值不再检查是否上升）"},
    "d1_adx_rising_bars":           {"type": "int",   "default": 4,             "label": "ADX连续上升根数（v3.2: 5→4）"},
    "d1_adx_rising_min_gain":       {"type": "float", "default": 2.0,           "label": "ADX最小上升绝对值（v4.1: 1.0→2.0，1点在随机波动范围内无意义）"},
    "d1_trend_consistency":         {"type": "float", "default": 0.60,          "label": "趋势一致性最低占比（v4.4: 0.55→0.60，60%以上方向性更可靠）"},
    "d1_fast_slope_min_pct":        {"type": "float", "default": 0.06,          "label": "日线快EMA最小斜率%（v3.2: 0.10→0.06）"},
    "d1_min_ema_spread_pct":        {"type": "float", "default": 0.15,          "label": "日线EMA张口最小%（v3.3: 0.25→0.15，低波动市场更易通过）"},
    "d1_max_extension_atr":         {"type": "float", "default": 4.00,          "label": "日线距快EMA最大ATR倍数（v3.2: 3.5→4.0）"},
    "d1_min_bars":                  {"type": "int",   "default": 100,           "label": "日线最少K线数量（引擎limit=200，140太严）"},

    # ── 小时线回调参数 ──
    "h1_pullback_min_pct":          {"type": "float", "default": 1.5,           "label": "小时线最小有效回调幅度%（v4.1: 0.5→1.5，0.5%是噪声无法识别真实回调）"},
    "h1_pullback_max_pct":          {"type": "float", "default": 10.0,          "label": "小时线最大回调幅度%（v3.2: 8→10轻度放宽）"},
    "h1_max_dryup_ratio":           {"type": "float", "default": 0.65,          "label": "企稳段最大缩量系数（v4.4: 0.72→0.65，要求更充分的量能收缩）"},
    "h1_min_rebound_ratio_vs_base": {"type": "float", "default": 0.60,          "label": "末根量能回暖系数（v4.4: 0.50→0.60，末根需达基线60%才算真实回暖）"},
    "h1_min_rebound_hard_check":    {"type": "bool",  "default": True,           "label": "末根量能硬性检查（v4.4新增：末根量能低于回暖系数时直接拒绝）"},
    "h1_stab_pos_min":              {"type": "float", "default": 0.35,          "label": "价格在企稳区间的最低位置比（v4.1: 0.20→0.35，多头需回升至上35%才有效）"},
    "h1_bb_rank_max":               {"type": "float", "default": 0.45,          "label": "BB带宽历史分位上限（v4.1: 0.72→0.45，72分位是偏宽非挤压；真挤压在低分位）"},
    "h1_max_pullback_atr":          {"type": "float", "default": 5.0,          "label": "H1回调上限ATR倍数（v3.2: 4→5）"},
    "m15_timing_weight":            {"type": "float", "default": 0.30,         "label": "15m时机分权重（v4.0: 0.20→0.30，H1权重=0.70-此值）"},
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
    "h1_swing_lookback":            {"type": "int",   "default": 20,            "label": "H1摆动点回溯根数（8太少，找不到真实摆动高点）"},
    "h1_swing_confirm_bars":        {"type": "int",   "default": 2,             "label": "H1摆动确认根数"},
    "h1_stab_bars":                 {"type": "int",   "default": 8,             "label": "H1企稳确认根数"},
    "h1_stab_atr_ratio":            {"type": "float", "default": 4.50,          "label": "H1企稳段总振幅/ATR上限（v3.3: 3.0→4.5，整理期允许更宽范围）"},
    "h1_per_bar_atr_ratio":         {"type": "float", "default": 1.20,          "label": "H1单根ATR比（v3.3: 0.85→1.2，ATR是均值单根超ATR很正常）"},
    "h1_atr_period":                {"type": "int",   "default": 14,            "label": "H1 ATR计算周期"},
    "h1_vol_baseline_bars":         {"type": "int",   "default": 20,            "label": "H1成交量基线根数"},
    "h1_bb_period":                 {"type": "int",   "default": 20,            "label": "H1布林带周期"},
    "h1_bb_std":                    {"type": "float", "default": 2.0,           "label": "H1布林带标准差倍数"},
    "h1_kc_period":                 {"type": "int",   "default": 20,            "label": "H1 Keltner通道周期"},
    "h1_kc_mult":                   {"type": "float", "default": 2.0,           "label": "H1 KC通道ATR倍数"},
    "h1_bb_squeeze_pct":            {"type": "float", "default": 9.0,           "label": "H1 BB挤压宽度绝对阈值%（v3.3: 6→9，中等波动市场更易达标）"},
    "h1_bb_lookback":               {"type": "int",   "default": 80,            "label": "H1 BB历史回溯根数"},
    "h1_bb_expand_ratio":           {"type": "float", "default": 1.3,           "label": "H1 BB相对历史最低倍数（v4.1: 2.0→1.3，2x太松；历史低×1.3更接近真实挤压）"},

    # ── 日线EMA参数 ──
    "d1_ema_fast":                  {"type": "int",   "default": 20,            "label": "日线快EMA周期"},
    "d1_ema_mid":                   {"type": "int",   "default": 50,            "label": "日线中EMA周期"},
    "d1_ema_slow":                  {"type": "int",   "default": 120,           "label": "日线慢EMA周期"},
    "d1_slope_lookback":            {"type": "int",   "default": 5,             "label": "日线斜率回溯周期"},
    "d1_slope_accel_bars":          {"type": "int",   "default": 3,             "label": "日线斜率加速检测根数"},
    "d1_trend_bars":                {"type": "int",   "default": 10,            "label": "日线趋势持续性检测根数"},

    # ── TTM挤压 / BTC环境 / 信号持久 ──
    "require_ttm_squeeze":          {"type": "bool",  "default": False,         "label": "强制要求严格TTM挤压（关闭=近挤压也算）"},
    "ttm_near_squeeze_tolerance":   {"type": "float", "default": 0.05,          "label": "TTM近挤压容忍比例（v4.4: 0.10→0.05，进一步收紧近挤压判定）"},
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

    # ── H1 快速EMA（新增，用于企稳期价格位置验证）──
    "h1_ema_fast":                  {"type": "int",   "default": 21,            "label": "H1快速EMA周期（验证价格是否贴近EMA支撑，避免深度偏离）"},

    # ── 挤压持续时间（新增：弹簧压缩越久弹力越大）──
    "h1_squeeze_duration_bonus_per_bar": {"type": "float", "default": 1.5,    "label": "每根挤压K线加分（持续挤压=能量积累，越久弹力越大）"},
    "h1_squeeze_duration_max_bonus":     {"type": "float", "default": 15.0,   "label": "挤压持续时间最大加分上限"},

    # ── Fibonacci回调位验证（新增：关键Fib位企稳准确率显著更高）──
    "h1_fib_tolerance_pct":         {"type": "float", "default": 2.5,          "label": "Fibonacci回调位容差%（±2.5%视为命中）"},
    "h1_fib_lookback":              {"type": "int",   "default": 60,            "label": "Fibonacci基础点位回溯根数（H1）"},

    # ── 日线MACD（新增：D1层缺少动能方向确认）──
    "d1_macd_fast":                 {"type": "int",   "default": 12,            "label": "日线MACD快线周期"},
    "d1_macd_slow":                 {"type": "int",   "default": 26,            "label": "日线MACD慢线周期"},
    "d1_macd_signal":               {"type": "int",   "default": 9,             "label": "日线MACD信号线周期"},

    # ── 15m StochRSI（新增：精准捕捉超卖反转时机）──
    "m15_stoch_rsi_period":         {"type": "int",   "default": 14,            "label": "15m StochRSI RSI周期"},
    "m15_stoch_lookback":           {"type": "int",   "default": 14,            "label": "15m StochRSI 随机振荡回溯周期"},
    "m15_stoch_smooth":             {"type": "int",   "default": 3,             "label": "15m StochRSI K值平滑周期"},

    # ── 15m 入场时机层（新增）──
    "m15_ema_fast":                 {"type": "int",   "default": 8,             "label": "15m快速EMA周期"},
    "m15_ema_slow":                 {"type": "int",   "default": 21,            "label": "15m慢速EMA周期"},
    "m15_min_bars":                 {"type": "int",   "default": 60,            "label": "15m最少K线数量"},
    "use_15m_confirmation":         {"type": "bool",  "default": True,          "label": "启用15m时机确认（关闭=不计算15m评分，仅做参考）"},
    "m15_min_timing_score":         {"type": "float", "default": 20.0,          "label": "15m时机最低分软门槛（低于此值按比例扣分，不硬拒）"},

    # ── VWAP偏离度（B级新增：机构成本线支撑验证）──
    "h1_vwap_lookback":             {"type": "int",   "default": 50,            "label": "H1 VWAP计算回溯根数（典型价×成交量加权）"},
    "h1_vwap_max_dev_pct":          {"type": "float", "default": 5.0,           "label": "VWAP最大允许偏离%（超出此值为深度偏离，软惩罚）"},
    "h1_vwap_near_pct":             {"type": "float", "default": 2.0,           "label": "VWAP命中容差%（±2%内=机构成本线支撑，满分加成）"},

    # ── D1日线K线形态（B级新增：企稳区高质量止跌确认）──
    "enable_d1_patterns":           {"type": "bool",  "default": True,          "label": "启用日线K线形态检测（锤子线/吞没线/十字星）"},
    "d1_pattern_bonus":             {"type": "float", "default": 10.0,          "label": "日线K线形态最大加分"},

    # ── 15m RSI动量（B级新增：RSI从低位回升方向确认）──
    "m15_rsi_period":               {"type": "int",   "default": 14,            "label": "15m RSI计算周期"},
    "m15_rsi_oversold":             {"type": "float", "default": 35.0,          "label": "15m RSI超卖阈值（低于此值后回升为强信号）"},

    # ── 加密资金费率（B级新增：过滤多/空头过拥挤场景）──
    "enable_funding_rate_filter":   {"type": "bool",  "default": True,          "label": "启用资金费率过滤（需引擎注入funding_rate）"},
    "funding_rate_bull_max":        {"type": "float", "default": 0.05,          "label": "多头最大资金费率%/8h（超过=多头过拥挤）"},
    "funding_rate_bear_min":        {"type": "float", "default": -0.05,         "label": "空头最小资金费率%/8h（低于=空头过拥挤）"},
    "funding_rate_penalty":         {"type": "float", "default": 8.0,           "label": "资金费率过拥挤最大扣分"},

    # ── Volume Profile POC（C级新增：成交量集中区=真实支撑）──
    "h1_vp_lookback":               {"type": "int",   "default": 50,            "label": "Volume Profile回溯根数"},
    "h1_vp_bins":                   {"type": "int",   "default": 20,            "label": "Volume Profile价格分桶数"},
    "h1_vp_tolerance_pct":          {"type": "float", "default": 2.0,           "label": "VP POC命中容差%（±2%内=在成交量集中区）"},

    # ── 持仓量 OI 变化（C级新增：OI+价格同向验证趋势真实性）──
    "enable_oi_filter":             {"type": "bool",  "default": True,          "label": "启用OI变化验证（需引擎注入 oi_change_pct）"},
    "oi_confirm_bonus":             {"type": "float", "default": 5.0,           "label": "OI趋势确认加分（OI增+价格同向）"},
    "oi_diverge_penalty":           {"type": "float", "default": 5.0,           "label": "OI趋势背离扣分（OI减+价格上涨=虚涨）"},

    # ── 板块相对强度（C级新增：板块排名过低的币种剔除）──
    "enable_sector_filter":         {"type": "bool",  "default": False,         "label": "启用板块相对强度过滤（需引擎注入 sector_rank_pct）"},
    "sector_min_rank_pct":          {"type": "float", "default": 0.40,          "label": "板块内最低相对强度排名（0.4=需在板块前60%）"},

    # ── v4.2 新增：H4趋势共振确认（多周期合一进一步过滤臲信号）──
    "enable_h4_resonance":          {"type": "bool",  "default": True,          "label": "v4.2启用H4趋势共振确认（H4与D1方向一致加分，逆趋扣分）"},
    "h4_resonance_bonus":           {"type": "float", "default": 8.0,           "label": "H4共振加分（EMA方向一致）"},
    "h4_conflict_penalty":          {"type": "float", "default": 10.0,          "label": "H4逆向扣分（EMA方向相反=硬拒风险）"},
    "h4_conflict_hard_block":       {"type": "bool",  "default": False,         "label": "H4逆趋硬拒（v4.5默认关闭=H4转折滞后于H1是常态，硬拒过严，改为软扣分）"},
    "h4_ema_fast":                  {"type": "int",   "default": 20,            "label": "H4快线EMA周期"},
    "h4_ema_slow":                  {"type": "int",   "default": 50,            "label": "H4慢线EMA周期"},
    "h4_min_bars":                  {"type": "int",   "default": 60,            "label": "H4最少K线数量"},

    # ── v4.2 新增：企稳段收敛型弹笧（振幅在收窄＝能量积累更多）──
    "enable_convergence_bonus":     {"type": "bool",  "default": True,          "label": "v4.2启用收敛型弹笧加分（企稳段振幅逐根收窄=弹笧能量积累）"},
    "convergence_max_bonus":        {"type": "float", "default": 8.0,           "label": "收敛型弹笧最大加分"},

    # ── v4.2 新增：D1趋势年龄过滤（ADX运行过长=中后期，非萌芽）──
    "d1_adx_max_age_bars":          {"type": "int",   "default": 60,            "label": "v4.2 D1 ADX连续＞阸28最大允许根数（60根=2周远超趋势中后期）"},
    "d1_adx_age_penalty_start":     {"type": "int",   "default": 30,           "label": "ADX超过该根数开始扣分（30根=1个月开始趋势老化惩罚）"},
    "d1_adx_age_penalty_max":       {"type": "float", "default": 12.0,          "label": "ADX趋势年龄最大扣分"},

    # ── v4.2 新增：15m MACD 动能入场确认──
    "use_15m_macd_confirm":         {"type": "bool",  "default": True,          "label": "v4.2 15m MACD方向确认（MACD柱与趋势一致加分）"},
    "m15_macd_cross_bonus":         {"type": "float", "default": 10.0,          "label": "15m MACD金叉/死叉加分"},
    "m15_macd_align_bonus":         {"type": "float", "default": 5.0,           "label": "15m MACD柱方向一致加分"},

    # ── v4.5 P1-1 双通路（萌芽 vs 挤压再启动）──────────────────────────────
    "dual_path_enabled":            {"type": "bool",  "default": True,          "label": "v4.5启用双通路：A萌芽通路+B挤压再启动通路（解决物理互斥矛盾）"},
    "path_a_min_h1_score":          {"type": "float", "default": 35.0,          "label": "A萌芽通路H1分门槛（萌芽期BB通常已扩张，门槛低于通路B）"},
    "path_b_min_h1_score":          {"type": "float", "default": 45.0,          "label": "B挤压通路H1分门槛（要求严格TTM挤压+Fib/VWAP命中）"},
    "path_a_min_total":             {"type": "float", "default": 58.0,          "label": "A萌芽通路总分门槛"},
    "path_b_min_total":             {"type": "float", "default": 62.0,          "label": "B挤压通路总分门槛"},
    "path_a_required_volume_boost": {"type": "float", "default": 1.10,          "label": "A萌芽通路要求15m末根量能/均量（≥1.10x）"},

    # ── v4.5 P1-2 H4多维度共振分（替代二值方向判定）──────────────────────
    "h4_resonance_v2_enabled":      {"type": "bool",  "default": True,          "label": "v4.5 H4多维度共振分（方向40%+ADX 30%+斜率30%）替代旧二值判定"},
    "h4_resonance_v2_max":          {"type": "float", "default": 12.0,          "label": "H4共振v2最大加分"},
    "h4_resonance_v2_hard_floor":   {"type": "float", "default": 0.20,          "label": "H4共振v2硬拒下限（共振分<0.20=H4严重逆向，硬拒）"},
    "h4_adx_period":                {"type": "int",   "default": 14,            "label": "H4 ADX周期"},
    "h4_slope_lookback":            {"type": "int",   "default": 5,             "label": "H4快EMA斜率回溯根数"},

    # ── v4.5 P1-3 15m 价格突破企稳上沿（突破前一刻特征）──────────────────
    "m15_breakout_imminent_enabled":{"type": "bool",  "default": True,          "label": "v4.5启用15m突破前一刻特征（接近H1企稳上沿+量能放大）"},
    "m15_breakout_proximity_pct":   {"type": "float", "default": 0.4,           "label": "接近企稳上沿容差%（≤此值视为即将突破）"},
    "m15_breakout_vol_min":         {"type": "float", "default": 1.30,          "label": "突破前量能最低倍数（×均量）"},
    "m15_breakout_max_bonus":       {"type": "float", "default": 12.0,          "label": "突破前一刻最大加分"},

    # ── v4.5 P1-4 z-score 趋势加速度（统一替代 ext_atr 硬拒+mature_pullback）
    "zscore_health_enabled":        {"type": "bool",  "default": True,          "label": "v4.5启用z-score统一评估趋势加速度（替代ext_atr硬拒+mature冲突）"},
    "zscore_atr_period":            {"type": "int",   "default": 14,            "label": "z-score计算用ATR周期"},
    "zscore_optimal_max":           {"type": "float", "default": 0.5,           "label": "z<此值=最优早期入场（满分）"},
    "zscore_emerging_max":          {"type": "float", "default": 1.5,           "label": "z=此值=萌芽期（线性扣分）"},
    "zscore_extended_max":          {"type": "float", "default": 2.5,           "label": "z=此值=已透支（最大扣分）"},
    "zscore_max_bonus":             {"type": "float", "default": 10.0,          "label": "z-score最大加分（早期入场）"},
    "zscore_max_penalty":           {"type": "float", "default": 15.0,          "label": "z-score最大扣分（已透支）"},

    # ── v4.5 P2-1 加分项加权聚合（替代简单相加）─────────────────────────
    "score_aggregation_mode":       {
        "type": "select",
        "default": "weighted",
        "options": [
            {"label": "weighted（加权平均，v4.5新版，鉴别度更高）", "value": "weighted"},
            {"label": "sum（简单相加，兼容旧版）",                  "value": "sum"},
        ],
        "label": "评分聚合模式（v4.5新增）",
    },

    # ── v4.5 P2-2 动态止损止盈（联动 swing/Fib/POC/BB）───────────────────
    "dynamic_sl_tp_enabled":        {"type": "bool",  "default": True,          "label": "v4.5启用动态止损止盈（基于关键支撑阻力位）"},
    "min_rr_ratio":                 {"type": "float", "default": 1.5,           "label": "最小盈亏比（动态计算后<此值=放弃信号）"},

    # ── v4.5 P2-3 共振分（D1×H4×H1 几何平均）────────────────────────────
    "resonance_score_enabled":      {"type": "bool",  "default": True,          "label": "v4.5启用三周期共振分（几何平均，比线性相加鉴别度高）"},
    "resonance_score_weight":       {"type": "float", "default": 0.20,          "label": "共振分占总分权重"},

    # ── v4.5 P2-4 仓位建议（信号置信度×波动率自适应）──────────────────────
    "position_advice_enabled":      {"type": "bool",  "default": True,          "label": "v4.5启用建议仓位输出（基础风险×置信度/ATR%）"},
    "position_base_risk_pct":       {"type": "float", "default": 1.0,           "label": "基础风险敞口%（账户净值的此比例承担单笔最大损失）"},

    # ── v4.3 新增（8项修复）──────────────────────────────────────────────────
    # #1/#3 摆动点修复相关
    "h1_swing_stale_bars":          {"type": "int",   "default": 12,            "label": "#3 摆动点时效阈值（超过此根数开始扣分，默认12根=12小时）"},
    "h1_swing_stale_penalty":       {"type": "float", "default": 8.0,           "label": "#3 摆动点时效最大扣分"},
    # #4 成熟趋势次级回调
    "enable_mature_pullback_bonus": {"type": "bool",  "default": True,          "label": "#4 启用成熟趋势次级回调加分（ADX强且价格回踩快EMA）"},
    "mature_pullback_ema_tol_pct":  {"type": "float", "default": 1.5,           "label": "#4 回踩快EMA的容差%（±1.5%内认定为精确回踩）"},
    "mature_pullback_max_bonus":    {"type": "float", "default": 15.0,          "label": "#4 成熟趋势次级回调最大加分"},
    # #6 量能形态分类
    "enable_vol_pattern_bonus":     {"type": "bool",  "default": True,          "label": "#6 启用量能形态分类加分（末根初放量特征）"},
    "vol_pattern_quiet_max":        {"type": "float", "default": 0.55,          "label": "#6 整理段内根量能阈值（低于此×基线=深度缩量）"},
    "vol_pattern_tick_min":         {"type": "float", "default": 0.75,          "label": "#6 末根初放量下限（高于此×基线算回暖）"},
    "vol_pattern_tick_max":         {"type": "float", "default": 1.50,          "label": "#6 末根放量上限（超过此×基线疑似假突破）"},
    "vol_pattern_max_bonus":        {"type": "float", "default": 8.0,           "label": "#6 量能最优形态最大加分"},
}

_DEFAULT_CONFIG = {k: v["default"] for k, v in CONFIG_SCHEMA.items()}

# v3.2: 市场状态自适应参数
MARKET_STATE_PARAMS: Dict[str, Dict[str, Any]] = {
    "trending": {
        # 趋势市场：ADX已在上行周期，放宽萌芽ADX要求，但收紧一致性/量能/多周期
        # v4.5: h1_pullback_min_pct 2.0→1.5，趋势市场回调本就较浅，2.0%过滤太多合理入场点
        "d1_adx_min": 16.0, "d1_adx_strong": 22.0,
        "d1_adx_rising_bars": 3, "d1_adx_rising_min_gain": 1.5,
        "d1_trend_consistency": 0.55, "d1_fast_slope_min_pct": 0.05, "d1_min_ema_spread_pct": 0.15,
        "h1_pullback_min_pct": 1.5, "h1_pullback_max_pct": 10.0, "h1_max_dryup_ratio": 0.68,
        "h1_stab_pos_min": 0.35, "m15_timing_weight": 0.25,
        "h1_stab_atr_ratio": 4.50, "h1_per_bar_atr_ratio": 1.20, "h1_swing_lookback": 20,
        "h1_bb_rank_max": 0.45,
        "h1_bb_expand_ratio": 1.4,
        "ttm_near_squeeze_tolerance": 0.06,
    },
    "range": {
        # 震荡市：更严格的ADX要求，TTM挤压更重要
        "d1_adx_min": 15.0, "d1_adx_rising_bars": 5, "d1_adx_rising_min_gain": 2.5,
        "d1_trend_consistency": 0.60, "d1_fast_slope_min_pct": 0.04, "d1_min_ema_spread_pct": 0.15,
        "h1_pullback_min_pct": 1.5, "h1_pullback_max_pct": 12.0, "h1_max_dryup_ratio": 0.65,
        "h1_bb_rank_max": 0.35, "require_ttm_squeeze": True, "h1_stab_pos_min": 0.40,
        "h1_per_bar_atr_ratio": 1.20, "h1_stab_atr_ratio": 4.00,
        "m15_timing_weight": 0.30, "h1_bb_expand_ratio": 1.2,
        "ttm_near_squeeze_tolerance": 0.08,
    },
    "volatile": {
        # 高波动市：更高ADX门槛，更宽回调范围，更严格缩量
        "d1_adx_min": 20.0, "d1_adx_rising_bars": 2, "d1_adx_rising_min_gain": 1.5,
        "d1_trend_consistency": 0.55, "d1_fast_slope_min_pct": 0.12, "d1_min_ema_spread_pct": 0.40,
        "h1_pullback_min_pct": 2.0, "h1_pullback_max_pct": 15.0, "h1_max_dryup_ratio": 0.60,
        "h1_max_pullback_atr": 6.0, "use_atr_pullback_range": True, "h1_stab_pos_min": 0.40,
        "m15_timing_weight": 0.15, "h1_bb_expand_ratio": 1.2,
        "ttm_near_squeeze_tolerance": 0.08,
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

    required_bars = ["1D", "1H", "15m", "4H"]
    # 各周期所需根数上限（引擎若支持 required_bars_limits 则按此分别设置 limit）
    # OKX 单次最多 300 根；1D 取 300 确保最老合约也能拿满 100+ 根历史
    required_bars_limits = {"1D": 300, "4H": 200, "1H": 200, "15m": 100}
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
        """根据市场状态自适应覆盖参数。
        置信度 < 60% 时为弱判定，跳过对 H1 回调/企稳门槛的收紧覆盖，
        避免边缘市场状态把过多合理入场对滤掉。
        """
        if state not in MARKET_STATE_PARAMS:
            return
        confidence = float(self.config.get("market_state_conf", 1.0))
        # 这些参数在低置信度时不应覆盖（会过度收紧，导致0信号）
        _h1_strict_keys = {
            "h1_pullback_min_pct", "h1_pullback_max_pct",
            "h1_stab_atr_ratio", "h1_per_bar_atr_ratio",
            "h1_max_dryup_ratio", "h1_stab_pos_min",
            "h1_bb_rank_max", "ttm_near_squeeze_tolerance",
        }
        for k, v in MARKET_STATE_PARAMS[state].items():
            if confidence < 0.60 and k in _h1_strict_keys:
                continue  # 弱判定：保留默认 H1 参数，不收紧
            self.config[k] = v
        self._market_state = state

    def _apply_vol_pool_params(self, pool: str) -> None:
        """v3.2: 根据波动率池覆盖参数"""
        try:
            from src.scanner.volatility_pools import get_pool_params
            overrides = get_pool_params(pool, list(self.config.keys()))
            for k, v in overrides.items():
                if k in self.config:
                    self.config[k] = v
        except Exception:
            pass  # 波动率池模块不可用时静默跳过，不影响主逻辑
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

        # ── C3: 板块相对强度预过滤（最轻量快速拒绝，无需拉K线）─────────────
        if bool(self.config.get("enable_sector_filter", False)):
            try:
                extra_d = getattr(symbol, "extra_data", {}) or {}
                sector_rank = extra_d.get("sector_rank_pct") if isinstance(extra_d, dict) else None
                if sector_rank is not None:
                    sector_rank = float(sector_rank)
                    sector_min  = float(self.config.get("sector_min_rank_pct", 0.40))
                    if sector_rank < sector_min:
                        return {
                            **base_fail,
                            "details": {
                                "跳过原因": (
                                    f"板块相对强度不足({sector_rank:.0%} < {sector_min:.0%})"
                                    f"，该币在同板块中排名过低"
                                ),
                            },
                        }
            except Exception:
                pass

        d1_rows  = self._get_klines(symbol, "1D")
        h1_rows  = self._get_klines(symbol, "1H")
        h4_rows  = self._get_klines(symbol, "4H")   # v4.2: H4共振确认
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

        # v4.4: 日线子分硬门槛（通过定性检查但评分过低则淘汰）
        min_d1_score = float(self.config.get("min_d1_score", 45.0))
        if d1_score < min_d1_score:
            return {
                **base_fail,
                "details": {"日线过滤": "通过但评分不足", **d1_details,
                            "淘汰原因": f"日线评分{d1_score:.1f} < 门槛{min_d1_score:.0f}"},
                "signals": [f"日线评分不足({d1_score:.1f}<{min_d1_score:.0f})"],
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
        h1_ok, h1_score, h1_details, h1_factors = self._check_h1_ttm_squeeze(h1_rows, d1_dir, h4_rows)
        if not h1_ok:
            return {
                **base_fail,
                "details": {"日线过滤": "通过", "H1过滤": "未通过", **d1_details, **h1_details},
                "signals": [f"H1淘汰: {h1_details.get('淘汰原因', '')}"],
                "factor_scores": {**d1_factors, **h1_factors},
            }

        # P1-1 双通路：根据 H1 识别的 signal_path 应用差异化 H1 子分门槛
        #   A 通路（萌芽期）：BB 未严格挤压，门槛较低（35），但要求 15m 量能放大
        #   B 通路（挤压再启动）：严格 TTM 挤压，门槛较高（45）
        signal_path = h1_details.get("信号通路", "B-挤压再启动")
        dual_path_on = bool(self.config.get("dual_path_enabled", True))
        if dual_path_on and signal_path == "A-萌芽期":
            min_h1_score = float(self.config.get("path_a_min_h1_score", 35.0))
        else:
            min_h1_score = float(self.config.get("min_h1_score", 40.0)) if not dual_path_on \
                           else float(self.config.get("path_b_min_h1_score", 45.0))
        if h1_score < min_h1_score:
            return {
                **base_fail,
                "details": {"日线过滤": "通过", "H1过滤": "通过但评分不足", **d1_details, **h1_details,
                            "淘汰原因": f"[{signal_path}] H1评分{h1_score:.1f} < 门槛{min_h1_score:.0f}"},
                "signals": [f"[{signal_path}] H1评分不足({h1_score:.1f}<{min_h1_score:.0f})"],
                "factor_scores": {**d1_factors, **h1_factors},
            }

        # ── Step 3：15m 入场时机确认（软条件，不通过只扣分，不淘汰）────────
        m15_bonus = 0.0
        m15_details: Dict[str, Any] = {}
        m15_min = int(self.config["m15_min_bars"])
        if bool(self.config.get("use_15m_confirmation", True)) and len(m15_rows) >= m15_min:
            # P1-3: 传递 H1 企稳上沿/下沿 用于"突破前一刻"判断
            h1_stab_top = float(h1_details.get("企稳上沿", 0) or 0)
            h1_stab_bot = float(h1_details.get("企稳下沿", 0) or 0)
            m15_bonus, m15_details = self._check_15m_entry_timing(
                m15_rows, d1_dir, h1_stab_top=h1_stab_top, h1_stab_bot=h1_stab_bot,
            )

        # ── 综合评分 ─────────────────────────────────────────────────────────
        # v4.0: 权重调整：15m=0.30（↑），H1=0.70-15m权重（↓），D1=剩余（通常0.30）
        m15_weight = float(self.config.get("m15_timing_weight", 0.30))
        h1_weight  = max(0.30, 0.70 - m15_weight)
        d1_weight  = max(0.0, 1.0 - h1_weight - m15_weight)

        # P2-3: 三周期共振分（D1×H4×H1 几何平均，比线性相加鉴别度高）
        # 共振分：把三个层级的评分都归一到 [0,1]，求几何平均后乘 100
        # 几何平均的特性：任何一个层级很弱（接近 0）会显著拉低最终分数 → 真正"全部共振"才高分
        resonance_score = 0.0
        if bool(self.config.get("resonance_score_enabled", True)):
            d1_norm = _clamp(d1_score / 100.0, 0.05, 1.0)        # 下限 0.05 避免 0 导致几何平均归零
            h1_norm = _clamp(h1_score / 100.0, 0.05, 1.0)
            m15_norm = _clamp(m15_bonus / 100.0, 0.05, 1.0) if (
                bool(self.config.get("use_15m_confirmation", True)) and len(m15_rows) >= m15_min
            ) else 0.5  # 没有 m15 数据时用中性 0.5（不影响共振分）
            # H4 共振 v2 分（0~1）作为额外乘子
            h4_v2_norm = float(h1_details.get("H4共振v2分", 0.5) or 0.5)
            h4_v2_norm = _clamp(h4_v2_norm, 0.05, 1.0)
            # 几何平均
            geo_mean = (d1_norm * h1_norm * m15_norm * h4_v2_norm) ** 0.25
            resonance_score = round(geo_mean * 100.0, 2)

        # P2-1: 评分聚合模式
        #   sum 模式（旧）：简单加权求和，加分项触顶 100 后丢失鉴别度
        #   weighted 模式（新）：基础分 70% + logistic 锦上添花 30%（边际递减）
        agg_mode = str(self.config.get("score_aggregation_mode", "weighted")).lower()
        # 把共振分按权重融入基础分
        res_w = float(self.config.get("resonance_score_weight", 0.20))
        base_score = (d1_score * d1_weight + h1_score * h1_weight + m15_bonus * m15_weight) * (1.0 - res_w) \
                     + resonance_score * res_w
        if agg_mode == "weighted":
            # 基础分：3 个层级的加权平均（已是 0~100）
            base_score_capped = _clamp(base_score, 0.0, 100.0)
            # 锦上添花的"溢出"部分：原始 base_score 超过 100 的部分，按 logistic 边际递减
            #   x' = 100 + 30 × (1 - exp(-(x-100)/15))   when x>100
            #   即超过 100 后，再加 30 分但边际衰减
            if base_score > 100.0:
                overflow = base_score - 100.0
                margin   = 30.0 * (1.0 - math.exp(-overflow / 15.0))
                total_score = round(_clamp(base_score_capped + margin * 0.3, 0.0, 100.0), 2)
            else:
                total_score = round(base_score_capped, 2)
        else:
            # 旧 sum 模式（兼容）
            total_score = round(base_score, 2)

        # P0-B8 修复：原惩罚系数 0.5 太弱（最大仅 3 分），起不到过滤作用。
        # 改为 1.0 倍 + 极低分(<10)叠加额外惩罚，让 15m 时机分真正影响综合排名。
        m15_soft_min = float(self.config.get("m15_min_timing_score", 20.0))
        m15_data_ok  = bool(self.config.get("use_15m_confirmation", True)) and len(m15_rows) >= m15_min
        if m15_data_ok and m15_bonus < m15_soft_min:
            # 主惩罚：(soft_min - m15_bonus) × 时机权重 × 1.0（原 0.5）
            m15_penalty  = (m15_soft_min - m15_bonus) * m15_weight * 1.0
            # 额外惩罚：m15_bonus < 10 时再扣 (10 - m15_bonus) × 0.6（极低时机分扣得更狠）
            if m15_bonus < 10.0:
                m15_penalty += (10.0 - m15_bonus) * 0.6
            m15_penalty  = round(m15_penalty, 2)
            total_score  = max(0.0, round(total_score - m15_penalty, 2))

        direction = "BUY" if d1_dir == "bull" else "SELL"

        # ── v4.2: D1 ADX趋势年龄惩罚（ADX运行过长=中后期，非萌芽期）───────────
        adx_age_penalty = 0.0
        adx_age_bars    = 0
        adx_max_age     = int(self.config.get("d1_adx_max_age_bars", 60))
        adx_age_start   = int(self.config.get("d1_adx_age_penalty_start", 30))
        adx_age_max_pts = float(self.config.get("d1_adx_age_penalty_max", 12.0))
        adx_strong_val  = float(self.config.get("d1_adx_strong", 28.0))
        if len(d1_rows) >= 10:
            # #8 fix: 复用 _check_daily_trend_sprout 已计算结果：
            # d1_details["ADX"] 是当前值；用 _calc_adx_series_fast 只做
            # 一次完整 Wilder 序列推算（比 last_n 次逐偏移重算快 ~last_n 倍）
            _d1_c = [float(r[4]) for r in d1_rows]
            _d1_h = [float(r[2]) for r in d1_rows]
            _d1_l = [float(r[3]) for r in d1_rows]
            _adx_period = int(self.config.get("d1_adx_period", 14))
            age_look     = min(adx_max_age + 5, len(d1_rows) // 2)
            adx_age_series = _calc_adx_series_fast(_d1_c, _d1_h, _d1_l, _adx_period, age_look)
            for av in reversed(adx_age_series):
                if av >= adx_strong_val:
                    adx_age_bars += 1
                else:
                    break
            if adx_age_bars >= adx_max_age:
                # 超过硬限：趋势路途已非常老化，直接淘汰
                return {
                    **base_fail,
                    "details": {
                        "跳过原因": f"D1 ADX连续{adx_age_bars}根>{adx_strong_val}，趋势已进入中后期（>{adx_max_age}根）",
                        "ADX年龄根数": adx_age_bars,
                    },
                    "signals": [f"D1 ADX年龄过长({adx_age_bars}根)，跑势中后期不适在此最早入场"],
                    "factor_scores": d1_factors,
                }
            elif adx_age_bars > adx_age_start:
                # 部分老化：按比例扣分
                age_ratio       = (adx_age_bars - adx_age_start) / max(adx_max_age - adx_age_start, 1)
                adx_age_penalty = round(adx_age_max_pts * _clamp(age_ratio, 0.0, 1.0), 2)
                total_score     = round(max(0.0, total_score - adx_age_penalty), 2)

        # v3.1: BTC市场环境过滤—从 extra_data 获取BTC基准
        btc_penalty = 0.0
        if bool(self.config.get("enable_btc_market_filter", False)) and d1_dir == "bull":
            btc_ctx = self._get_btc_from_symbol(symbol)
            btc_penalty = self._calc_btc_penalty(btc_ctx)
            total_score = round(max(0, total_score - btc_penalty), 2)

        # ── B4: 资金费率过滤（过拥挤场景降分）────────────────────────────────
        funding_penalty = 0.0
        funding_rate    = 0.0
        if bool(self.config.get("enable_funding_rate_filter", True)):
            try:
                extra_d      = getattr(symbol, "extra_data", {}) or {}
                funding_rate = float((extra_d.get("funding_rate") if isinstance(extra_d, dict) else None) or 0)
                if funding_rate != 0:
                    fr_bull_max  = float(self.config.get("funding_rate_bull_max",  0.05))
                    fr_bear_min  = float(self.config.get("funding_rate_bear_min", -0.05))
                    fr_max_pts   = float(self.config.get("funding_rate_penalty",   8.0))
                    if d1_dir == "bull" and funding_rate > fr_bull_max:
                        excess          = (funding_rate - fr_bull_max) / max(fr_bull_max, 1e-9)
                        funding_penalty = min(fr_max_pts * (1.0 + excess), fr_max_pts * 2.0)
                    elif d1_dir == "bear" and funding_rate < fr_bear_min:
                        excess          = (fr_bear_min - funding_rate) / max(abs(fr_bear_min), 1e-9)
                        funding_penalty = min(fr_max_pts * (1.0 + excess), fr_max_pts * 2.0)
                    funding_penalty = round(funding_penalty, 2)
                    total_score = round(max(0.0, total_score - funding_penalty), 2)
            except Exception:
                pass

        # ── C2: OI变化验证（OI+价格同向=趋势真实；OI减+价格涨=虚涨）─────────
        oi_adj   = 0.0
        oi_state = "未获取"
        if bool(self.config.get("enable_oi_filter", True)):
            try:
                extra_d    = getattr(symbol, "extra_data", {}) or {}
                oi_chg_pct = float((extra_d.get("oi_change_pct") if isinstance(extra_d, dict) else None) or 0)
                price_chg  = float(getattr(symbol, "price_change_24h", 0) or 0)
                if oi_chg_pct != 0:
                    oi_confirm = (
                        (d1_dir == "bull" and oi_chg_pct > 0 and price_chg > 0) or
                        (d1_dir == "bear" and oi_chg_pct > 0 and price_chg < 0)
                    )
                    oi_diverge = (
                        (d1_dir == "bull" and oi_chg_pct < 0 and price_chg > 0) or
                        (d1_dir == "bear" and oi_chg_pct < 0 and price_chg < 0)
                    )
                    if oi_confirm:
                        oi_adj   = float(self.config.get("oi_confirm_bonus",  5.0))
                        oi_state = f"OI同向确认(+{oi_adj:.1f})"
                    elif oi_diverge:
                        oi_adj   = -float(self.config.get("oi_diverge_penalty", 5.0))
                        oi_state = f"OI虚涨背离({oi_adj:.1f})"
                    else:
                        oi_state = "OI中性"
                    total_score = round(max(0.0, total_score + oi_adj), 2)
            except Exception:
                pass

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

        # P1-1 双通路总分门槛 + A 通路要求 15m 量能放大
        if dual_path_on:
            if signal_path == "A-萌芽期":
                # A 通路：要求 15m 量能放大（≥1.10x 均量），否则强制扣 6 分
                vol_ratio_15m = float(m15_details.get("量能系数", 1.0) or 1.0)
                vol_min       = float(self.config.get("path_a_required_volume_boost", 1.10))
                if vol_ratio_15m < vol_min:
                    total_score = max(0.0, round(total_score - 6.0, 2))
                min_score = float(self.config.get("path_a_min_total", 58.0))
            else:
                min_score = float(self.config.get("path_b_min_total", 62.0))
        else:
            # P0-B1: fallback 与 CONFIG_SCHEMA 默认值统一为 60.0
            min_score = float(self.config.get("min_score", 60.0))
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
            f"{category} · {signal_path} · 综合评分 {total_score:.1f}",
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
        h4_res = h1_details.get("H4共振状态", "")
        if h4_res:
            signals.append(f"📐 H4: {h4_res}")
        if adx_age_penalty > 0:
            signals.append(f"⏳ ADX趋势年龄{adx_age_bars}根·老化扣分-{adx_age_penalty:.1f}")
        if btc_penalty > 0:
            signals.append(f"⚠ BTC弱势·环境降{btc_penalty:.0f}分")
        if funding_penalty > 0:
            signals.append(f"💸 资金费率过拥挤({funding_rate:+.3f}%)·降{funding_penalty:.0f}分")
        if oi_adj != 0:
            signals.append(f"📈 {oi_state}")
        if h1_details.get("VWAP加分", 0) > 0:
            signals.append(
                f"🏦 VWAP偏离{float(h1_details.get('VWAP偏离%',0)):+.2f}%"
                f"  VP_POC偏离{float(h1_details.get('VP_POC偏离%',0)):+.2f}%"
            )
        d1_pat = d1_details.get("D1形态", "无")
        if d1_pat and d1_pat != "无":
            signals.append(f"🕯 D1形态:{d1_pat}(+{d1_details.get('D1形态加分',0):.0f}分)")
        if buy_ratio > 0.55:
            signals.append(f"📊 买量占比{buy_ratio:.0%}（买方主动）")
        if sl_pct > 0:
            signals.append(f"🛑 止损-{sl_pct:.1f}% / 🎯 止盈+{tp_pct:.1f}%")
        # P2-2: 动态止损止盈输出
        dyn_sl_pct = float(h1_details.get("动态止损%", 0) or 0)
        dyn_tp_pct = float(h1_details.get("动态止盈%", 0) or 0)
        dyn_rr     = float(h1_details.get("动态盈亏比", 0) or 0)
        if dyn_sl_pct > 0 and dyn_tp_pct > 0:
            signals.append(
                f"📐 动态: 止损-{dyn_sl_pct:.1f}% / 止盈+{dyn_tp_pct:.1f}%"
                f" ({h1_details.get('动态止盈来源','')}, RR={dyn_rr:.2f})"
            )
        # P2-4: 仓位建议
        if position_pct > 0:
            signals.append(f"💰 建议仓位 {position_pct:.2f}%（基础风险×置信度/止损）")
        # P2-3: 共振分提示
        if resonance_score > 0:
            signals.append(f"🌊 三周期共振分 {resonance_score:.1f}/100")

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

        # ── P2-4: 建议仓位（信号置信度 × 波动率自适应）─────────────────
        # 公式：position_pct = base_risk × confidence / sl_pct
        #   base_risk = 账户净值的此比例承担最大损失（默认 1%）
        #   confidence = total_score / 100，表示信号置信度
        #   sl_pct = 止损距离百分比（来自动态止损或 ATR 止损）
        #   ADX 年龄惩罚：年龄越大，仓位越小（在 30 根后线性递减到 50%）
        position_pct = 0.0
        if bool(self.config.get("position_advice_enabled", True)):
            try:
                base_risk     = float(self.config.get("position_base_risk_pct", 1.0)) / 100.0
                confidence    = _clamp(total_score / 100.0, 0.0, 1.0)
                sl_pct_use    = float(h1_details.get("动态止损%", h1_details.get("ATR止损%", 0)) or 0)
                if sl_pct_use > 0:
                    pos_raw = base_risk * confidence / (sl_pct_use / 100.0)
                    # ADX 年龄系数：30 根 → 1.0，60 根 → 0.5
                    age_factor = 1.0 - 0.5 * _clamp((adx_age_bars - 30) / 30.0, 0.0, 1.0)
                    pos_raw   *= age_factor
                    # 限制最大仓位（不超过 20%）
                    position_pct = round(_clamp(pos_raw * 100.0, 0.0, 20.0), 2)
            except Exception:
                pass

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
                "信号通路": signal_path,
                "综合评分": f"{total_score:.1f}",
                "三周期共振分": f"{resonance_score:.1f}",
                "BTC环境分": f"-{btc_penalty:.0f}" if btc_penalty > 0 else "正常",
                "资金费率%": round(funding_rate, 4),
                "资金费率扣分": f"-{funding_penalty:.1f}" if funding_penalty > 0 else "无",
                "OI变化状态": oi_state,
                "信号持续性": f"+{persistence_bonus:.0f}" if persistence_bonus > 0 else "首次/不稳定",
                "买量占比": f"{buy_ratio:.1%}",
                "ATR止损%": f"-{sl_pct:.1f}%" if sl_pct > 0 else "-",
                "ATR止盈%": f"+{tp_pct:.1f}%" if tp_pct > 0 else "-",
                "建议仓位%": f"{position_pct:.2f}%" if position_pct > 0 else "-",
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
        fail_stats: Dict[str, int] = {}
        step_stats = {"成交额不足": 0, "数据不足": 0, "Step1日线": 0, "Step2_H1": 0, "通过": 0, "其他": 0}
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=min(8, len(symbols) or 1)) as executor:
            futures = {executor.submit(self._safe_scan, sym): sym for sym in symbols}
            for future in as_completed(futures):
                try:
                    result = future.result(timeout=15)
                except Exception as e:
                    fail_stats[f"[future异常]{type(e).__name__}"] = fail_stats.get(f"[future异常]{type(e).__name__}", 0) + 1
                    step_stats["其他"] += 1
                    continue
                if result is None:
                    fail_stats["[None结果]"] = fail_stats.get("[None结果]", 0) + 1
                    step_stats["其他"] += 1
                    continue
                if result.get("passed"):
                    results.append(result)
                    step_stats["通过"] += 1
                else:
                    details = result.get("details", {})
                    # 步骤级分类
                    if "跳过原因" in details:
                        skip = details["跳过原因"]
                        if "成交额" in str(skip):
                            step_stats["成交额不足"] += 1
                        else:
                            step_stats["数据不足"] += 1
                    elif "日线过滤" in details:
                        if details.get("日线过滤") == "未通过":
                            step_stats["Step1日线"] += 1
                        else:
                            step_stats["Step2_H1"] += 1
                    else:
                        step_stats["其他"] += 1
                    # 原因提取
                    reason = (
                        details.get("淘汰原因") or
                        details.get("跳过原因") or
                        next(
                            (v for v in details.values()
                             if isinstance(v, str) and len(v) > 3 and v not in ("未通过", "通过")),
                            None
                        ) or
                        f"details_keys={list(details.keys())[:3]}"
                    )
                    key = str(reason)[:45]
                    fail_stats[key] = fail_stats.get(key, 0) + 1
        results.sort(
            key=lambda item: float(item.get("opportunity_score", item.get("score", 0.0)) or 0.0),
            reverse=True,
        )
        total_accounted = len(results) + sum(fail_stats.values())
        top = sorted(fail_stats.items(), key=lambda x: x[1], reverse=True)[:15]
        print(f"[趋势挤压诊断] 通过{len(results)}/{len(symbols)} 统计{total_accounted} 步骤分布:{step_stats}")
        print(f"[趋势挤压诊断] 失败TOP15: { {k:v for k,v in top} }")
        top_n = int(self.config.get("top_n", 12))
        # #7 fix: 附加诊断信息到返回字典，UI层可直接读取展示
        return {
            "type": "trend_squeeze_breakout_v3",
            "all_opportunities": results[:top_n],
            "total_passed": len(results),
            "total_scanned": len(symbols),
            "diagnostics": {
                "step_stats":     step_stats,
                "fail_top15":     {k: v for k, v in top},
                "total_accounted": total_accounted,
                "pass_rate_pct":  round(len(results) / max(len(symbols), 1) * 100, 1),
            },
        }

    def _safe_scan(self, sym) -> Optional[Dict]:
        """单个符号扫描 + 异常保护"""
        try:
            return self.scan_symbol(sym)
        except Exception as exc:
            exc_key = f"[异常]{type(exc).__name__}: {str(exc)[:50]}"
            logger.warning(f"[趋势挤压v3] {getattr(sym, 'inst_id', '')} 异常: {exc}")
            return {"passed": False, "details": {"淘汰原因": exc_key}}

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

        d1_min_bars = int(self.config.get("d1_min_bars", 100))
        min_required = max(d1_min_bars, 100)
        if not rows or len(rows) < min_required:
            return False, "wait", 0.0, {"淘汰原因": f"日线数据不足({len(rows)}根，需{min_required}根)"}, {"daily_adx_score": 0.0}

        opens   = [_v(r, 1) for r in rows]
        highs   = [_v(r, 2) for r in rows]
        lows    = [_v(r, 3) for r in rows]
        closes  = [_v(r, 4) for r in rows]

        ema_f         = int(self.config.get("d1_ema_fast", 20))
        ema_m         = int(self.config.get("d1_ema_mid", 50))
        ema_s         = int(self.config.get("d1_ema_slow", 120))
        adx_min       = float(self.config.get("d1_adx_min", 15.0))
        adx_strong    = float(self.config.get("d1_adx_strong", 28.0))
        adx_rise_bars = int(self.config.get("d1_adx_rising_bars", 4))
        adx_rise_gain = float(self.config.get("d1_adx_rising_min_gain", 2.0))
        trend_bars    = int(self.config.get("d1_trend_bars", 10))
        consistency   = float(self.config.get("d1_trend_consistency", 0.55))
        slope_lb      = int(self.config.get("d1_slope_lookback", 5))
        min_slope_pct = float(self.config.get("d1_fast_slope_min_pct", 0.06))
        accel_bars    = int(self.config.get("d1_slope_accel_bars", 3))
        min_spread    = float(self.config.get("d1_min_ema_spread_pct", 0.15))
        max_ext_atr   = float(self.config.get("d1_max_extension_atr", 4.00))
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
                # v4.1 Bug修复：原 < adx_rise_bars-1 要求100%连续上升（3/3 or 4/4），
                # ADX自身有1-2点波动，导致大量有效萌芽被误拒；
                # 改为允许1次偶发回落：需要 >= max(1, adx_rise_bars-2) 次上升
                min_rises = max(1, adx_rise_bars - 2)
                if adx_rising_count < min_rises or adx_gain < adx_rise_gain:
                    return False, "wait", 0.0, {
                        "淘汰原因": (
                            f"ADX={adx_val:.1f}尚弱且未持续上升"
                            f"（{adx_rise_bars}根内上升{adx_rising_count}次，需≥{min_rises}次，"
                            f"涨幅{adx_gain:.1f}，需≥{adx_rise_gain}）"
                        ),
                        "ADX": round(adx_val, 2),
                        "ADX涨幅": round(adx_gain, 2),
                    }, {"daily_adx_score": round(adx_val * 1.5, 2)}
                adx_rising = True

        # ── 检查 3：EMA 排列方向 ────────────────────────────────────────────
        bull_ema = ema_fast > ema_mid > ema_slow      # 完全多头排列
        bear_ema = ema_fast < ema_mid < ema_slow      # 完全空头排列
        # 萌芽期：只要 fast 穿越 mid 即可，slow 尚未跟上是正常现象
        # 用 DI+/DI- 作为辅助方向裁定（避免错误识别方向）
        bull_partial = ema_fast > ema_mid and di_plus > di_minus
        bear_partial = ema_fast < ema_mid and di_minus > di_plus

        if not (bull_ema or bear_ema or bull_partial or bear_partial):
            return False, "wait", 0.0, {
                "淘汰原因": f"EMA无方向信号：EMA{ema_f}={'↑' if ema_fast>ema_mid else '↓'}EMA{ema_m} DI+={di_plus:.1f} DI-={di_minus:.1f}",
                "ADX": round(adx_val, 2),
            }, {"daily_alignment_score": 0.0}

        direction = "bull" if (bull_ema or bull_partial) else "bear"

        # ── 检查 4：趋势方向一致性 ────────────────────────────────────────
        if len(closes) >= trend_bars + 1:
            recent = closes[-(trend_bars + 1):]
            # #2 fix: 只统计移动幅度超过 ATR×0.3 的"有效方向根"，排除横盘噪声
            noise_threshold = atr * 0.3
            bar_hits = sum(
                1 for i in range(1, len(recent))
                if abs(recent[i] - recent[i - 1]) > noise_threshold
                and (recent[i] > recent[i - 1]) == (direction == "bull")
            )
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

        # v4.1 Bug修复：consistency参数原先被读取但从未用于门控或评分（写死0.20）
        # 现在用consistency作为真正的软硬双门槛：
        #   ratio < 0.30 (hard floor) → 直接淘汰
        #   0.30 ≤ ratio < consistency → 允许通过但评分按实际比例折扣
        #   ratio ≥ consistency → 正常评分（线性满分）
        hard_consistency_floor = 0.30  # 低于30%方向极度混乱，无论配置如何均淘汰
        if ratio < hard_consistency_floor:
            return False, "wait", 0.0, {
                "淘汰原因": f"日线趋势一致性极低({ratio * 100:.0f}% < {hard_consistency_floor*100:.0f}%，方向极度混乱)",
                "ADX": round(adx_val, 2), "趋势一致性%": round(ratio * 100, 1),
            }, {"daily_consistency_score": round(ratio * 100, 2)}

        # ── 检查 5：EMA 斜率 + 斜率加速度（新增：斜率在加速才说明趋势在增强）
        fast_slope_pct = 0.0
        slope_accel    = False
        if len(ema_fast_series) > slope_lb and np.isfinite(ema_fast_series[-1 - slope_lb]):
            fast_prev = float(ema_fast_series[-1 - slope_lb])
            fast_slope_pct = _pct_change(ema_fast, fast_prev)

        slope_ok = fast_slope_pct >= min_slope_pct if direction == "bull" else fast_slope_pct <= -min_slope_pct
        # v4.0: 软硬双门槛 — 斜率明显反向(<40%阈值)才硬拒，避免遗漏ADX强劲的早期趋势
        # slope_ok=False但未触硬拒线：评分中体现斜率不足，不再直接淘汰
        slope_hard_min = min_slope_pct * 0.40
        slope_hard_ok = (fast_slope_pct >= slope_hard_min) if direction == "bull" else (fast_slope_pct <= -slope_hard_min)
        if not slope_hard_ok:
            hard_line_str = f"+{slope_hard_min:.2f}%" if direction == "bull" else f"-{slope_hard_min:.2f}%"
            return False, "wait", 0.0, {
                "淘汰原因": f"快EMA斜率反向或极度平坦({fast_slope_pct:+.2f}%，硬拒线{hard_line_str})",
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

        # P1-4: 极端透支才硬拒（>1.5×max_ext_atr，约 6 ATR），其余进入 z-score 软评分
        if directional_ext_atr > max_ext_atr * 1.5:
            return False, "wait", 0.0, {
                "淘汰原因": f"日线已严重透支({directional_ext_atr:.2f} ATR > {max_ext_atr*1.5:.2f}，硬拒线)",
                "快EMA延伸ATR": round(directional_ext_atr, 3), "ADX": round(adx_val, 2),
            }, {"daily_extension_health_score": 0.0}

        # ── P1-4: z-score 趋势加速度统一评估（替代 ext_atr 硬拒+mature冲突）
        # z = (price - ema_fast) / atr 方向化（多头>0 表示在 EMA 上方，越高越透支）
        # z < zscore_optimal_max (0.5)  → 早期最佳入场（满分加分）
        # z < zscore_emerging_max (1.5) → 萌芽期（线性递减加分）
        # z < zscore_extended_max (2.5) → 透支区（线性扣分）
        # z >= extended_max             → 重度透支（最大扣分，但不硬拒）
        zscore_adj = 0.0
        z_label    = "中性"
        if bool(self.config.get("zscore_health_enabled", True)) and atr > 0:
            zscore_signed = ((last_close - ema_fast) / atr) if direction == "bull" \
                            else ((ema_fast - last_close) / atr)
            z_optimal = float(self.config.get("zscore_optimal_max",  0.5))
            z_emerge  = float(self.config.get("zscore_emerging_max", 1.5))
            z_extend  = float(self.config.get("zscore_extended_max", 2.5))
            max_bonus = float(self.config.get("zscore_max_bonus",   10.0))
            max_pen   = float(self.config.get("zscore_max_penalty", 15.0))
            if zscore_signed <= z_optimal:
                # 早期或回调到 EMA 内：满分加分
                zscore_adj = max_bonus
                z_label    = f"早期(z={zscore_signed:.2f}≤{z_optimal})"
            elif zscore_signed <= z_emerge:
                # 萌芽期：线性递减
                ratio = (zscore_signed - z_optimal) / max(z_emerge - z_optimal, 1e-9)
                zscore_adj = max_bonus * (1.0 - ratio)
                z_label    = f"萌芽(z={zscore_signed:.2f})"
            elif zscore_signed <= z_extend:
                # 透支区：线性扣分
                ratio = (zscore_signed - z_emerge) / max(z_extend - z_emerge, 1e-9)
                zscore_adj = -max_pen * ratio
                z_label    = f"透支(z={zscore_signed:.2f})"
            else:
                # 重度透支
                zscore_adj = -max_pen
                z_label    = f"重度透支(z={zscore_signed:.2f})"
            zscore_adj = round(zscore_adj, 2)

        # ── 新增：日线MACD方向确认（D1层原本完全缺少动能方向信号）────────────
        d1_macd_f = int(self.config.get("d1_macd_fast", 12))
        d1_macd_s = int(self.config.get("d1_macd_slow", 26))
        d1_macd_g = int(self.config.get("d1_macd_signal", 9))
        d1_macd_series  = _calc_macd_hist_series(closes, fast=d1_macd_f, slow=d1_macd_s, signal=d1_macd_g)
        d1_macd_last    = d1_macd_series[-1] if d1_macd_series else 0.0
        d1_macd_prev    = d1_macd_series[-2] if len(d1_macd_series) >= 2 else d1_macd_last
        d1_macd_aligned = (
            (direction == "bull" and d1_macd_last >= 0) or
            (direction == "bear" and d1_macd_last <= 0)
        )
        d1_macd_turning = (
            (direction == "bull" and d1_macd_last > d1_macd_prev) or
            (direction == "bear" and d1_macd_last < d1_macd_prev)
        )
        d1_macd_zero_cross = (
            (direction == "bull" and d1_macd_prev <= 0 < d1_macd_last) or
            (direction == "bear" and d1_macd_prev >= 0 > d1_macd_last)
        )
        # D1 MACD 不作硬门槛，仅影响评分（日线动能是锦上添花，不是必要条件）
        d1_macd_bonus = 0.0
        if d1_macd_aligned:     d1_macd_bonus += 5.0   # 方向一致
        if d1_macd_turning:     d1_macd_bonus += 4.0   # 正在拐头
        if d1_macd_zero_cross:  d1_macd_bonus += 6.0   # 零线穿越最强

        # ── #4 fix: 成熟趋势次级回调专项加分 ──────────────────────────────
        # P0-B9 双模式说明：
        #   策略名为「萌芽期」，但允许两种入场场景：
        #     A 模式：D1 趋势萌芽（ADX 17~strong 且上升中）→ 主入场场景
        #     B 模式：D1 趋势已成型（ADX≥strong）但回调到快EMA → 次级回调入场（本加分项）
        #   B 模式 + ADX 年龄惩罚共同确保不在「中后期已透支」的趋势顶部入场。
        #   B 模式入场严格依赖「价格紧贴 EMA + 量能配合」，不做萌芽 ADX 检查。
        mature_pullback_bonus = 0.0
        is_mature_pullback_mode = False
        if bool(self.config.get("enable_mature_pullback_bonus", True)):
            near_ema_tol = float(self.config.get("mature_pullback_ema_tol_pct", 1.5))
            if adx_val >= adx_strong and ema_fast > 0:
                ema_dev_pct = abs(last_close - ema_fast) / ema_fast * 100.0
                if ema_dev_pct <= near_ema_tol:
                    max_mab = float(self.config.get("mature_pullback_max_bonus", 15.0))
                    # 越贴近EMA分越高（偏离0%=满分，偏离near_ema_tol%=0分）
                    mature_pullback_bonus = round(max_mab * (1.0 - ema_dev_pct / near_ema_tol), 2)
                    is_mature_pullback_mode = True

        # ── 评分 ──────────────────────────────────────────────────────────────
        # ADX 分：弱趋势萌芽（<strong）也给分，但上升中额外加成
        adx_base_score = 20.0 * _clamp((adx_val - adx_min) / max(adx_strong - adx_min, 1e-9), 0.0, 1.0)
        adx_rise_bonus  = 6.0 if adx_rising else 0.0                    # 上升中加成
        adx_score       = adx_base_score + adx_rise_bonus

        alignment_score    = 14.0 if (bull_ema or bear_ema) else 8.0    # 完全有序比部分有序得分高
        # v4.1 Bug修复：一致性评分现在真正使用consistency配置值
        # 从hard_floor到consistency线性满分，低于consistency但>hard_floor按比例给分
        consistency_score  = 16.0 * _clamp(
            (ratio - hard_consistency_floor) / max(consistency - hard_consistency_floor, 1e-9),
            0.0, 1.0,
        )
        slope_score        = 16.0 * _clamp(abs(fast_slope_pct) / max(min_slope_pct * 3.0, 1e-9), 0.0, 1.0)
        slope_accel_bonus  = 5.0 if slope_accel else 0.0               # 加速度加成
        spread_score       = 10.0 * _clamp(ema_spread_pct / max(min_spread * 2.5, 1e-9), 0.0, 1.0)
        spread_expand_bonus = 4.0 if spread_expanding else 0.0         # 张口扩大加成
        ext_health_score   = 9.0 * _clamp(1.0 - directional_ext_atr / max(max_ext_atr, 1e-9), 0.0, 1.0)

        # ── D1 K线形态检测（锤子线/吞没线/十字星 = 企稳高质量止跌信号）───────
        d1_pattern_name = ""
        d1_pattern_bonus = 0.0
        if bool(self.config.get("enable_d1_patterns", True)) and len(closes) >= 2:
            max_pattern_pts = float(self.config.get("d1_pattern_bonus", 10.0))
            pat_name, pat_raw = _detect_d1_patterns(opens, highs, lows, closes, direction)
            d1_pattern_name  = pat_name
            d1_pattern_bonus = round(min(pat_raw, max_pattern_pts), 2)

        score = round(min(
            adx_score + alignment_score + consistency_score + slope_score + slope_accel_bonus
            + spread_score + spread_expand_bonus + ext_health_score + d1_macd_bonus
            + d1_pattern_bonus + mature_pullback_bonus
            + zscore_adj,                              # P1-4: z-score 加/扣分
            100.0,
        ), 2)
        score = max(0.0, score)

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
            "D1_MACD方向一致": d1_macd_aligned,
            "D1_MACD拐头": d1_macd_turning,
            "D1_MACD零线穿越": d1_macd_zero_cross,
            "D1_MACD加分": round(d1_macd_bonus, 1),
            "D1形态": d1_pattern_name if d1_pattern_name else "无",
            "D1形态加分": round(d1_pattern_bonus, 2),
            "成熟趋势次级回调加分": round(mature_pullback_bonus, 2),
            "入场模式": "B-成熟回调" if is_mature_pullback_mode else "A-萌芽期",
            "z-score": z_label,
            "z-score调整": round(zscore_adj, 2),
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
            "daily_macd_bonus": round(d1_macd_bonus, 2),
            "daily_pattern_bonus": round(d1_pattern_bonus, 2),
            "daily_mature_pullback_bonus": round(mature_pullback_bonus, 2),
            "daily_zscore_adj": round(zscore_adj, 2),
        }
        return True, direction, score, details, factor_scores

    # ══════════════════════════════════════════════════════════════════════════
    # Step 2：H1 TTM 挤压检测（全面重写）
    # ══════════════════════════════════════════════════════════════════════════

    def _check_h1_ttm_squeeze(
        self, rows: List, d1_direction: str, h4_rows: Optional[List] = None
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
        min_rebound_b   = float(self.config["h1_min_rebound_ratio_vs_base"]) # vs 基线，schema 默认 0.50

        bb_per      = int(self.config["h1_bb_period"])
        bb_std_mult = float(self.config["h1_bb_std"])
        kc_per      = int(self.config["h1_kc_period"])
        kc_mult     = float(self.config["h1_kc_mult"])
        bb_sq_pct   = float(self.config["h1_bb_squeeze_pct"])
        bb_look     = int(self.config["h1_bb_lookback"])
        bb_rank_max = float(self.config["h1_bb_rank_max"])
        bb_expand   = float(self.config["h1_bb_expand_ratio"])
        req_ttm     = bool(self.config.get("require_ttm_squeeze", False))
        ttm_tol     = float(self.config.get("ttm_near_squeeze_tolerance", 0.10))

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
        atr_pb_max = float(self.config.get("h1_max_pullback_atr", 5.0))
        if use_atr_pb and atr > 0:
            atr_pct = atr / cur_close * 100.0
            effective_pb_min = max(pb_min, atr_pct * 1.2)  # 至少1.2倍ATR
            effective_pb_max = min(pb_max, atr_pct * atr_pb_max)  # 不超过N倍ATR
        else:
            effective_pb_min = pb_min
            effective_pb_max = pb_max

        # #1 fix: 回退策略改为局部极值（而非全段极值）。局部极值 = 最近11根内的最高/最低点。
        # #3 fix: 摆动点时效性记录，超过 sw_stale_bars 根进行降分。
        sw_stale_bars  = int(self.config.get("h1_swing_stale_bars", 12))   # 摆动点超过此根数认为过期
        sw_stale_pen   = float(self.config.get("h1_swing_stale_penalty", 8.0))  # 摆动点过期扣分
        swing_staleness_penalty = 0.0
        swing_age = None   # 摆动点距当前的根数

        # stab 窗口内的极值（最近 stab_bars 根，原先被排除在 search 窗口之外）
        # 这些 K 线极有可能包含真正的近期高/低点，必须一并纳入参考
        stab_window_highs = list(highs[-stab_bars:]) if stab_bars > 0 else []
        stab_window_lows  = list(lows[-stab_bars:])  if stab_bars > 0 else []

        # P0-B2/B3 修复：swing_age 必须是「相对当前 bar 的根数」，不是子序列内偏移。
        # search 窗口对应索引 [n-look-stab_bars, n-stab_bars-1]，stab 窗口对应 [n-stab_bars, n-1]。
        # 因此：search 内的 idx 距当前 bar = (look - 1 - idx) + stab_bars
        #       stab   内的 idx 距当前 bar = (stab_bars - 1 - idx)
        if d1_direction == "bull":
            swing_extreme = _find_swing_high(search_highs, sw_confirm)
            swing_age_in_search = _find_swing_high_age(search_highs, sw_confirm)
            swing_age = (swing_age_in_search + stab_bars) if swing_age_in_search is not None else None
            if swing_extreme is None:
                local_n = min(11, len(search_highs))
                if search_highs:
                    sub = search_highs[-local_n:]
                    swing_extreme = max(sub)
                    inner_idx     = sub.index(swing_extreme)        # 0..local_n-1（局部窗口内偏移）
                    swing_age     = (local_n - 1 - inner_idx) + stab_bars
                else:
                    swing_extreme = 0.0
                    swing_age     = None
            # v3.2 fix: 纳入被 stab_bars 窗口排除的近期高点——但 swing_age 须用「距当前」算法
            if stab_window_highs:
                stab_max = max(stab_window_highs)
                if stab_max > (swing_extreme or 0.0):
                    swing_extreme = stab_max
                    inner_idx = stab_window_highs.index(stab_max)   # 0=stab 窗口最早；stab_bars-1=最新
                    swing_age = stab_bars - 1 - inner_idx           # P0-B2: 距当前 bar 的根数
            pullback_pct = (swing_extreme - cur_close) / max(swing_extreme, 1e-9) * 100.0
        else:
            swing_extreme = _find_swing_low(search_lows, sw_confirm)
            swing_age_in_search = _find_swing_low_age(search_lows, sw_confirm)
            swing_age = (swing_age_in_search + stab_bars) if swing_age_in_search is not None else None
            if swing_extreme is None:
                local_n = min(11, len(search_lows))
                if search_lows:
                    sub = search_lows[-local_n:]
                    swing_extreme = min(sub)
                    inner_idx     = sub.index(swing_extreme)
                    swing_age     = (local_n - 1 - inner_idx) + stab_bars     # P0-B3: 距当前 bar
                else:
                    swing_extreme = cur_close
                    swing_age     = None
            # v3.2 fix: 纳入被 stab_bars 窗口排除的近期低点
            if stab_window_lows:
                stab_min = min(stab_window_lows)
                if stab_min < (swing_extreme if swing_extreme > 0 else float("inf")):
                    swing_extreme = stab_min
                    inner_idx = stab_window_lows.index(stab_min)
                    swing_age = stab_bars - 1 - inner_idx           # P0-B2: 距当前 bar
            if swing_extreme <= 0:
                return False, 0.0, {"淘汰原因": "摆动低点异常"}, {"hourly_pullback_score": 0.0}
            pullback_pct = (cur_close - swing_extreme) / swing_extreme * 100.0

        # #3 fix: 摆动点时效性惩罚（摆动点趋旧，进入整理段的时间趋長，信号延迟）
        if swing_age is not None and swing_age > sw_stale_bars:
            stale_ratio              = (swing_age - sw_stale_bars) / max(sw_stale_bars, 1)
            swing_staleness_penalty  = round(min(sw_stale_pen * _clamp(stale_ratio, 0.0, 1.0), sw_stale_pen), 2)

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

        # ── Fibonacci 回调位验证（关键Fib位企稳=成功率显著更高）─────────────
        # P0-B4 修复：Fib 基准必须用「swing_high 之前的相邻摆动低点」（多头），而非全窗口最低点。
        # 用全窗口最低点会导致 Fib 范围过大、回调位飘移，命中率显著降低。
        # 实现：以 swing_age 为锚点，在 [swing_age, swing_age+fib_look] 区间内反向查找相邻 swing_low。
        fib_tol  = float(self.config.get("h1_fib_tolerance_pct", 2.5))
        fib_look = int(self.config.get("h1_fib_lookback", 60))
        fib_bonus = 0.0
        fib_hit_level = ""
        if fib_look >= 10 and n >= fib_look + stab_bars and swing_age is not None:
            # 摆动点在序列里的索引：swing_idx = n - 1 - swing_age
            swing_idx = max(0, n - 1 - swing_age)
            # Fib 基准搜索区间：swing 点之前的 fib_look 根
            fib_base_start = max(0, swing_idx - fib_look)
            fib_base_end   = swing_idx                           # 含 swing 点之前一根
            seg_lows       = lows[fib_base_start:fib_base_end]
            seg_highs      = highs[fib_base_start:fib_base_end]
            if d1_direction == "bull" and seg_lows:
                # 多头 Fib：取 swing_high 之前的「相邻摆动低点」
                # 优先：在 seg 内用 _find_swing_low（左右各2根确认），找不到再回退到 seg 最低点
                adj_swing_low = _find_swing_low(seg_lows, max(2, sw_confirm))
                if adj_swing_low is None:
                    adj_swing_low = min(seg_lows)
                fib_base_price = adj_swing_low
                fib_top_price  = swing_extreme
                fib_range      = fib_top_price - fib_base_price
                if fib_range > 0:
                    # 回调 Fib 位 = 高点 - range × ratio
                    fib_candidates = [
                        ("61.8%", fib_top_price - fib_range * 0.618, 12.0),
                        ("50.0%", fib_top_price - fib_range * 0.500, 10.0),
                        ("38.2%", fib_top_price - fib_range * 0.382,  8.0),
                    ]
                    for lvl_name, lvl_price, lvl_score in fib_candidates:
                        if lvl_price > 0:
                            dev_pct = abs(cur_close - lvl_price) / max(lvl_price, 1e-9) * 100.0
                            if dev_pct <= fib_tol and lvl_score > fib_bonus:
                                fib_bonus     = lvl_score
                                fib_hit_level = lvl_name
            elif seg_highs:
                # 空头 Fib：取 swing_low 之前的「相邻摆动高点」
                adj_swing_high = _find_swing_high(seg_highs, max(2, sw_confirm))
                if adj_swing_high is None:
                    adj_swing_high = max(seg_highs)
                fib_top_price  = adj_swing_high
                fib_base_price = swing_extreme
                fib_range      = fib_top_price - fib_base_price
                if fib_range > 0:
                    # 反弹 Fib 位 = 低点 + range × ratio
                    fib_candidates = [
                        ("61.8%", fib_base_price + fib_range * 0.618, 12.0),
                        ("50.0%", fib_base_price + fib_range * 0.500, 10.0),
                        ("38.2%", fib_base_price + fib_range * 0.382,  8.0),
                    ]
                    for lvl_name, lvl_price, lvl_score in fib_candidates:
                        if lvl_price > 0:
                            dev_pct = abs(cur_close - lvl_price) / max(lvl_price, 1e-9) * 100.0
                            if dev_pct <= fib_tol and lvl_score > fib_bonus:
                                fib_bonus     = lvl_score
                                fib_hit_level = lvl_name

        # ── 企稳：整体振幅 + 逐根振幅 ────────────────────────────────────────
        stab_highs   = highs[-stab_bars:]
        stab_lows    = lows[-stab_bars:]
        stab_range   = max(stab_highs) - min(stab_lows)
        stab_ratio   = stab_range / max(atr, 1e-9)
        # 同样只在ATR有意义时做绝对振幅过滤
        if atr > cur_close * 0.00001 and stab_range > atr * stab_atr:
            return False, 0.0, {
                "淘汰原因": f"未企稳：{stab_bars}根整体波幅({stab_range:.4f}) > ATR×{stab_atr}({atr*stab_atr:.4f})",
                "回调幅度%": round(pullback_pct, 2), "企稳波幅/ATR": round(stab_ratio, 3),
            }, {"hourly_stability_score": 0.0}

        per_bar_ranges = [highs[-stab_bars + i] - lows[-stab_bars + i] for i in range(stab_bars)]
        max_single = max(per_bar_ranges) if per_bar_ranges else 0.0
        # 只有当ATR有实际意义时才检查（ATR < 0.001% of price 时浮点精度不可靠，跳过）
        atr_meaningful = atr > cur_close * 0.00001
        if atr_meaningful and max_single / atr > per_bar_atr_r:
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

        # ── 挤压持续时间计数（弹簧压缩越久，爆发力越大）─────────────────
        # #5 fix: 阈值从 hist_min*bb_expand*1.2（依赖极值不稳定）
        #         改为 历史分位p25（更稳健地定义"处于挤压状态"）
        squeeze_threshold_hist = float(np.percentile(bb_widths_hist, 25)) if len(bb_widths_hist) >= 4 else hist_min * bb_expand * 1.2
        squeeze_duration = 0
        for bw_h in bb_widths_hist:   # bb_widths_hist[0] = 最近1根前的BB宽
            if bw_h <= squeeze_threshold_hist:
                squeeze_duration += 1
            else:
                break  # 连续性中断，停止计数
        dur_bonus_per = float(self.config.get("h1_squeeze_duration_bonus_per_bar", 1.5))
        dur_bonus_max = float(self.config.get("h1_squeeze_duration_max_bonus", 15.0))
        squeeze_duration_bonus = min(squeeze_duration * dur_bonus_per, dur_bonus_max)

        # P1-1 双通路判定：
        #   B 通路：严格挤压（rank+abs/rel 满足）→ 信号路径"挤压再启动"
        #   A 通路：未严格挤压 但 BB 仍处于历史中下分位（≤0.65）→ 信号路径"萌芽期"
        #          A 通路要求 15m 量能放大 + 价格在企稳区上半部 + RSI 健康（在后续逻辑中补充）
        path_b_squeeze_strict = (rank_squeeze and (abs_squeeze or rel_squeeze))
        path_a_emerging_zone  = (bw_rank <= 0.65)   # BB 不必极度收窄，但需在中下分位
        dual_path_on = bool(self.config.get("dual_path_enabled", True))
        if not path_b_squeeze_strict and not (dual_path_on and path_a_emerging_zone):
            return False, 0.0, {
                "淘汰原因": (
                    f"BB未达任何通路标准：带宽{cur_bw:.2f}%"
                    f"  分位{bw_rank * 100:.0f}%（B通路上限{bb_rank_max * 100:.0f}%/A通路上限65%）"
                    f"  历史最低{hist_min:.2f}%×{bb_expand}={hist_min*bb_expand:.2f}%"
                ),
                "BB带宽%": round(cur_bw, 3), "回调幅度%": round(pullback_pct, 2),
                "TTM挤压激活": ttm_squeeze_on,
            }, {"hourly_squeeze_score": 0.0}
        # 标识当前命中通路（B 优先，B 不满足时降级到 A）
        signal_path = "B-挤压再启动" if path_b_squeeze_strict else "A-萌芽期"

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

        # v4.0: H1快速EMA价格位置验证（pos_in_stab只看区间，不看EMA距离）
        # 深度偏离EMA（>5%）说明价格实质上未回到支撑位，硬拒；<5%偏离作为软评分
        h1_ema_fast_per = int(self.config.get("h1_ema_fast", 21))
        h1_ema_fast_s   = _calc_ema_series(closes, h1_ema_fast_per)
        h1_ema_fast_val = (
            h1_ema_fast_s[-1]
            if (h1_ema_fast_s and np.isfinite(h1_ema_fast_s[-1]))
            else cur_close
        )
        ema_gap_pct = (cur_close - h1_ema_fast_val) / max(h1_ema_fast_val, 1e-9) * 100.0
        if d1_direction == "bull" and ema_gap_pct < -5.0:
            return False, 0.0, {
                "淘汰原因": f"多头回调未恢复：价格深在H1 EMA{h1_ema_fast_per}下方({ema_gap_pct:.2f}%，硬拒<-5%)",
                "H1_EMA偏离%": round(ema_gap_pct, 2),
                "回调幅度%": round(pullback_pct, 2),
            }, {"hourly_location_score": round(pos_in_stab * 100.0, 2)}
        if d1_direction == "bear" and ema_gap_pct > 5.0:
            return False, 0.0, {
                "淘汰原因": f"空头反弹未结束：价格深在H1 EMA{h1_ema_fast_per}上方({ema_gap_pct:.2f}%，硬拒>+5%)",
                "H1_EMA偏离%": round(ema_gap_pct, 2),
                "回调幅度%": round(pullback_pct, 2),
            }, {"hourly_location_score": round((1.0 - pos_in_stab) * 100.0, 2)}
        # EMA接近度得分：-5%→0分, 0%→0.5分, +5%→1分（线性插值）
        ema_proximity_score = _clamp((ema_gap_pct + 5.0) / 10.0, 0.0, 1.0)

        # ── VWAP偏离度（机构成本线支撑验证）────────────────────────────────
        vwap_look     = int(self.config.get("h1_vwap_lookback", 50))
        vwap_max_dev  = float(self.config.get("h1_vwap_max_dev_pct", 5.0))
        vwap_near_pct = float(self.config.get("h1_vwap_near_pct", 2.0))
        vwap_val      = _calc_vwap(closes, highs, lows, volumes, vwap_look)
        vwap_dev_pct  = (cur_close - vwap_val) / max(vwap_val, 1e-9) * 100.0 if vwap_val > 0 else 0.0
        # 多头：价格应贴近或略高于VWAP（回调到机构成本线才是高质量支撑）
        # 空头：价格应贴近或略低于VWAP
        # 双向：±vwap_near_pct内=满加成；±vwap_max_dev内=线性渐减；超出=0分（不硬拒）
        vwap_abs_dev  = abs(vwap_dev_pct)
        if vwap_abs_dev <= vwap_near_pct:
            vwap_bonus = 10.0                          # 命中机构成本线：满分
        elif vwap_abs_dev <= vwap_max_dev:
            vwap_bonus = 10.0 * (1.0 - (vwap_abs_dev - vwap_near_pct) / max(vwap_max_dev - vwap_near_pct, 1e-9))
        else:
            vwap_bonus = 0.0

        # ── Volume Profile POC（成交量最密集区 = 真实支撑/阻力）────────────
        vp_look      = int(self.config.get("h1_vp_lookback", 50))
        vp_bins_n    = int(self.config.get("h1_vp_bins", 20))
        vp_tol_pct   = float(self.config.get("h1_vp_tolerance_pct", 2.0))
        vp_poc       = _calc_vp_poc(closes, highs, lows, volumes, vp_look, vp_bins_n)
        vp_dev_pct   = abs(cur_close - vp_poc) / max(vp_poc, 1e-9) * 100.0 if vp_poc > 0 else 999.0
        if vp_dev_pct <= vp_tol_pct:
            vp_bonus = 10.0                                # 命中POC：历史多空主战场
        elif vp_dev_pct <= vp_tol_pct * 2.5:
            vp_bonus = 10.0 * (1.0 - (vp_dev_pct - vp_tol_pct) / max(vp_tol_pct * 1.5, 1e-9))
        else:
            vp_bonus = 0.0

        # ── 量能：修复后版本（末根量 vs 基线，不是 vs 整理段均量）─────────
        baseline_slice = (
            volumes[-(stab_bars + vol_base_bars):-stab_bars]
            if n >= stab_bars + vol_base_bars else volumes[:-stab_bars]
        )
        # v4.0: 基线改用中位数（均值对pump/dump极值敏感，中位数更稳健）
        baseline_vol   = _median_positive(baseline_slice)
        stab_vol_mean  = _mean_positive(volumes[-stab_bars:])
        dryup_ratio    = stab_vol_mean / max(baseline_vol, 1e-9) if baseline_vol > 0 else 1.0

        if baseline_vol > 0 and dryup_ratio > max_dryup:
            return False, 0.0, {
                "淘汰原因": f"整理段缩量不足({dryup_ratio:.2f} > {max_dryup:.2f})，蓄势特征不足",
                "缩量系数": round(dryup_ratio, 3), "回调幅度%": round(pullback_pct, 2),
            }, {"hourly_dryup_score": round(max(0.0, 100.0 - dryup_ratio * 100.0), 2)}

        # 修复：末根量 vs 基线（不再与整理段自身比）
        lastbar_vol_ratio_vs_base = volumes[-1] / max(baseline_vol, 1e-9) if baseline_vol > 0 else 1.0

        # ── #6 fix: 量能形态分类，识别"末根初放量"最优形态 ──────────────
        # 最优形态：整理段其余根均在0.55倍基线以下，末根出现0.75~1.5倍基线的初步放量
        # 这是"能量积累完成，主力开始试探"的典型特征
        vol_pattern_bonus = 0.0
        vol_pattern_tag   = ""
        if bool(self.config.get("enable_vol_pattern_bonus", True)) and baseline_vol > 0 and stab_bars >= 4:
            inner_vols      = volumes[-(stab_bars):-1]   # 整理段中除末根外的量
            inner_max_ratio = max(v / baseline_vol for v in inner_vols) if inner_vols else 1.0
            last_ratio      = lastbar_vol_ratio_vs_base
            quiet_threshold = float(self.config.get("vol_pattern_quiet_max", 0.55))
            first_tick_lo   = float(self.config.get("vol_pattern_tick_min",  0.75))
            first_tick_hi   = float(self.config.get("vol_pattern_tick_max",  1.50))
            if inner_max_ratio < quiet_threshold and first_tick_lo <= last_ratio <= first_tick_hi:
                vol_pattern_bonus = float(self.config.get("vol_pattern_max_bonus", 8.0))
                vol_pattern_tag   = f"末根初放量({last_ratio:.2f}x, 内段最大{inner_max_ratio:.2f}x)"
            elif inner_max_ratio < quiet_threshold and last_ratio > first_tick_hi:
                # 末根放量过猛：可能是假突破后下跌，轻微扣分
                vol_pattern_bonus = -3.0
                vol_pattern_tag   = f"末根量过大({last_ratio:.2f}x>={first_tick_hi}x)"
            elif inner_max_ratio < quiet_threshold:
                vol_pattern_tag   = f"持续缩量未回暖({last_ratio:.2f}x)"
            else:
                vol_pattern_tag   = f"量能形态一般"

        # v4.4: 末根量能回暖硬性检查（h1_min_rebound_hard_check=True 时精确阈值拒绝）
        rebound_hard = bool(self.config.get("h1_min_rebound_hard_check", True))
        if rebound_hard:
            if lastbar_vol_ratio_vs_base < min_rebound_b:
                return False, 0.0, {
                    "淘汰原因": (
                        f"末根量能未达回暖阈值"
                        f"({lastbar_vol_ratio_vs_base:.2f} < {min_rebound_b:.2f}×基线)"
                    ),
                    "末根量能vs基线": round(lastbar_vol_ratio_vs_base, 3), "回调幅度%": round(pullback_pct, 2),
                }, {"hourly_rebound_score": 0.0}
        else:
            # 软模式：仅极度萎靡才硬拒
            if lastbar_vol_ratio_vs_base < min_rebound_b * 0.5:
                return False, 0.0, {
                    "淘汰原因": (
                        f"末根量能极度萎靡vs基线"
                        f"({lastbar_vol_ratio_vs_base:.2f} < {min_rebound_b*0.5:.2f})"
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
        macd_prev3 = macd_hist[-4] if len(macd_hist) >= 4 else macd_prev2
        macd_prev4 = macd_hist[-5] if len(macd_hist) >= 5 else macd_prev3

        if d1_direction == "bull":
            # P0-B6: 5根连续走坏几乎不会触发（萌芽期 MACD 起伏大）。改为软扣分：
            #   3根连续走坏 → 扣 4 分；4根 → 扣 8 分；5根 → 扣 12 分（不再硬拒）
            macd_3_adverse   = (macd_last < macd_prev < macd_prev2)
            macd_4_adverse   = macd_3_adverse and (macd_prev2 < macd_prev3)
            macd_5_adverse   = macd_4_adverse and (macd_prev3 < macd_prev4) and (macd_last < 0)
            macd_turn_str    = max(macd_last - macd_prev, 0.0) + max(macd_last - macd_prev2, 0.0) * 0.5
            macd_zero_cross  = macd_prev <= 0 < macd_last    # 零线刚上穿：加分
        else:
            macd_3_adverse   = (macd_last > macd_prev > macd_prev2)
            macd_4_adverse   = macd_3_adverse and (macd_prev2 > macd_prev3)
            macd_5_adverse   = macd_4_adverse and (macd_prev3 > macd_prev4) and (macd_last > 0)
            macd_turn_str    = max(macd_prev - macd_last, 0.0) + max(macd_prev2 - macd_last, 0.0) * 0.5
            macd_zero_cross  = macd_prev >= 0 > macd_last

        # P0-B6: MACD 走坏改为软扣分（不再硬拒，避免萌芽期短暂回调误判）
        macd_adverse_penalty = 0.0
        if macd_5_adverse:
            macd_adverse_penalty = 12.0
        elif macd_4_adverse:
            macd_adverse_penalty = 8.0
        elif macd_3_adverse:
            macd_adverse_penalty = 4.0

        # ── 回调评分 ──────────────────────────────────────────────────────────
        pullback_quality = _clamp(
            (pullback_pct - effective_pb_min) / max(effective_pb_max - effective_pb_min, 1e-9),
            0.0, 1.0,
        )
        pullback_score = 14.0 * pullback_quality

        # ── 企稳评分 ──────────────────────────────────────────────────────────
        stability_score = 12.0 * _clamp(
            1.0 - stab_ratio / max(stab_atr, 1e-9), 0.0, 1.0
        )

        # ── v4.5 P1-2: H4 多维度共振分（方向 40% + ADX 强度 30% + 斜率 30%）─────
        # 原二值判定（仅 EMA 方向）鉴别度太低；新版用 0~1 连续分数：
        #   方向：H4 EMA fast vs slow 与 D1 方向是否一致（贡献 0 或 1.0）
        #   ADX 强度：H4 ADX 归一化到 [0,1]（≥30 = 满分）
        #   斜率：H4 快EMA 斜率与方向一致性 + 强度
        # 综合 res_v2 ∈ [0,1]；< hard_floor=0.20 → 硬拒（视为严重逆向）
        enable_h4_v2 = bool(self.config.get("h4_resonance_v2_enabled", True))
        enable_h4_resonance = bool(self.config.get("enable_h4_resonance", True))
        h4_resonance_bonus  = 0.0
        h4_res_state        = "未获取"
        h4_res_v2_score     = 0.5   # 默认中性
        if (enable_h4_v2 or enable_h4_resonance) and h4_rows and len(h4_rows) >= int(self.config.get("h4_min_bars", 60)):
            try:
                h4_closes = [_v(r, 4) for r in h4_rows]
                h4_highs  = [_v(r, 2) for r in h4_rows]
                h4_lows   = [_v(r, 3) for r in h4_rows]
                h4_ef = int(self.config.get("h4_ema_fast", 20))
                h4_es = int(self.config.get("h4_ema_slow", 50))
                h4_ema_f_s = _calc_ema_series(h4_closes, h4_ef)
                h4_ema_s_s = _calc_ema_series(h4_closes, h4_es)
                h4_ef_val  = h4_ema_f_s[-1] if h4_ema_f_s and np.isfinite(h4_ema_f_s[-1]) else None
                h4_es_val  = h4_ema_s_s[-1] if h4_ema_s_s and np.isfinite(h4_ema_s_s[-1]) else None

                if enable_h4_v2 and h4_ef_val is not None and h4_es_val is not None:
                    # 1) 方向分（0/1）
                    h4_dir_match = (d1_direction == "bull" and h4_ef_val > h4_es_val) or \
                                   (d1_direction == "bear" and h4_ef_val < h4_es_val)
                    dir_sub = 1.0 if h4_dir_match else 0.0
                    # 2) ADX 强度分（0~1）
                    h4_adx_val, _, _ = _calc_adx(h4_closes, h4_highs, h4_lows,
                                                  int(self.config.get("h4_adx_period", 14)))
                    adx_sub = _clamp(h4_adx_val / 30.0, 0.0, 1.0)
                    # 3) 斜率方向 + 强度分（0~1）
                    slope_lb = int(self.config.get("h4_slope_lookback", 5))
                    slope_sub = 0.0
                    if len(h4_ema_f_s) > slope_lb and np.isfinite(h4_ema_f_s[-1 - slope_lb]):
                        h4_slope_pct = _pct_change(h4_ef_val, float(h4_ema_f_s[-1 - slope_lb]))
                        slope_dir_ok = (d1_direction == "bull" and h4_slope_pct > 0) or \
                                       (d1_direction == "bear" and h4_slope_pct < 0)
                        # 斜率方向一致 + 强度归一（0.5%/根 = 满分）
                        slope_sub = (0.6 if slope_dir_ok else 0.0) + 0.4 * _clamp(abs(h4_slope_pct) / 0.5, 0.0, 1.0)
                        slope_sub = _clamp(slope_sub, 0.0, 1.0)
                    # 加权综合：方向 40% + ADX 30% + 斜率 30%
                    h4_res_v2_score = dir_sub * 0.40 + adx_sub * 0.30 + slope_sub * 0.30
                    # 硬拒：共振分严重低 → H4 严重逆向且无强度
                    hard_floor = float(self.config.get("h4_resonance_v2_hard_floor", 0.20))
                    if h4_res_v2_score < hard_floor and bool(self.config.get("h4_conflict_hard_block", False)):
                        return False, 0.0, {
                            "淘汰原因": f"H4共振v2分{h4_res_v2_score:.2f}<{hard_floor:.2f}（H4严重逆向+无强度）",
                            "回调幅度%": round(pullback_pct, 2),
                        }, {"hourly_h4_bonus": 0.0}
                    # 加分：以 0.5 为零点的线性函数 [-max, +max]
                    h4_max = float(self.config.get("h4_resonance_v2_max", 12.0))
                    h4_resonance_bonus = round((h4_res_v2_score - 0.5) * 2.0 * h4_max, 2)
                    h4_res_state = (
                        f"共振v2={h4_res_v2_score:.2f} "
                        f"(方向{dir_sub:.0f}/ADX{adx_sub:.2f}/斜率{slope_sub:.2f})"
                    )
                elif enable_h4_resonance and h4_ef_val is not None and h4_es_val is not None:
                    # 旧路径（兼容）
                    h4_aligned   = (d1_direction == "bull" and h4_ef_val > h4_es_val) or \
                                   (d1_direction == "bear" and h4_ef_val < h4_es_val)
                    h4_conflicted = not h4_aligned
                    if h4_conflicted:
                        if bool(self.config.get("h4_conflict_hard_block", False)):
                            return False, 0.0, {
                                "淘汰原因": "H4方向与D1逆向（h4_conflict_hard_block=True）",
                                "回调幅度%": round(pullback_pct, 2),
                            }, {"hourly_h4_bonus": 0.0}
                        h4_resonance_bonus = -float(self.config.get("h4_conflict_penalty", 10.0))
                        h4_res_state       = "冲突(H4与D1反向)"
                    else:
                        h4_resonance_bonus = float(self.config.get("h4_resonance_bonus", 8.0))
                        h4_res_state       = "共振(H4与D1一致)"
            except Exception:
                pass

        # ── v4.2: 企稳段收敛型弹簧加分 ───────────────────────────────────────
        enable_convergence_bonus = bool(self.config.get("enable_convergence_bonus", True))
        convergence_bonus = 0.0
        is_converging     = False
        if enable_convergence_bonus and stab_bars >= 4:
            half = stab_bars // 2
            first_half_avg  = float(np.mean(per_bar_ranges[:half]))  if half > 0 else 0.0
            second_half_avg = float(np.mean(per_bar_ranges[half:]))  if half > 0 else 0.0
            if first_half_avg > 0 and second_half_avg < first_half_avg:
                is_converging     = True
                conv_ratio        = 1.0 - second_half_avg / first_half_avg  # 0=无收窄 1=完全收窄
                convergence_bonus = round(
                    float(self.config.get("convergence_max_bonus", 8.0)) * _clamp(conv_ratio, 0.0, 1.0), 2
                )

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

        # ── NR4/NR7 窄幅K线检测（严格：末根必须是窗口内最小振幅）────────────
        nr_detected = _detect_nr_bar(highs, lows, nr_look)
        if req_nr and not nr_detected:
            return False, 0.0, {
                "淘汰原因": f"未检测到NR{nr_look}窄幅K线（require_nr_bar=True）",
                "回调幅度%": round(pullback_pct, 2),
            }, {"hourly_nr_bonus": 0.0}

        nr_bonus      = 4.0 if nr_detected else 0.0    # NR4/NR7 加分项
        # v4.0: EMA接近度得分（-5%~0偏离:0~4分, 0~+5%:4~8分）
        ema_pos_score = 8.0 * ema_proximity_score

        h1_score = round(min(
            pullback_score + stability_score + squeeze_score + dryup_score
            + rsi_score + rsi_div_bonus + macd_score + location_score + nr_bonus + ema_pos_score
            + squeeze_duration_bonus + fib_bonus + vwap_bonus + vp_bonus
            + h4_resonance_bonus + convergence_bonus
            + vol_pattern_bonus                    # #6 量能形态加/扣分
            - swing_staleness_penalty              # #3 摆动点时效惩罚
            - macd_adverse_penalty,                # P0-B6: MACD 软扣分替代硬拒
            100.0,
        ), 2)
        h1_score = max(h1_score, 0.0)

        details = {
            "H1方向": "多头回调" if d1_direction == "bull" else "空头反弹",
            "摆动极值": round(swing_extreme, 4),
            "当前收盘": round(cur_close, 4),
            "回调幅度%": round(pullback_pct, 2),
            "企稳根数": stab_bars,
            "企稳波幅/ATR": round(stab_ratio, 3),
            "企稳区间位置": round(pos_in_stab, 3),
            "企稳上沿": round(stab_top, 6),
            "企稳下沿": round(stab_bot, 6),
            "ATR": round(atr, 4),
            "缩量系数": round(dryup_ratio, 3),
            "末根量能vs基线": round(lastbar_vol_ratio_vs_base, 3),
            "BB上轨": round(bb_upper, 4), "BB中轨": round(bb_mid, 4), "BB下轨": round(bb_lower, 4),
            "BB带宽%": round(cur_bw, 3),
            "BB历史最低%": round(hist_min, 3),
            "BB带宽历史分位%": round(bw_rank * 100, 1),
            "BB带内位置": round(pos_in_band, 3),
            "H1_EMA快线": round(h1_ema_fast_val, 4),
            "H1_EMA偏离%": round(ema_gap_pct, 2),
            "KC上轨": round(kc_upper, 4), "KC下轨": round(kc_lower, 4),
            "TTM挤压激活": ttm_squeeze_on,
            "H1_RSI": round(h1_rsi, 2),
            "RSI背离": rsi_divergence,
            "MACD柱体": round(macd_last, 6),
            "MACD柱体%": round(macd_pct, 4),
            "MACD零线穿越": macd_zero_cross,
            "NR4/NR7": nr_detected,
            "挤压持续根数": squeeze_duration,
            "挤压持续加分": round(squeeze_duration_bonus, 2),
            "Fib命中位": fib_hit_level if fib_hit_level else "未命中",
            "Fib加分": round(fib_bonus, 2),
            "VWAP": round(vwap_val, 4),
            "VWAP偏离%": round(vwap_dev_pct, 2),
            "VWAP加分": round(vwap_bonus, 2),
            "VP_POC": round(vp_poc, 4),
            "VP_POC偏离%": round(vp_dev_pct, 2),
            "VP_POC加分": round(vp_bonus, 2),
            "H4共振状态": h4_res_state,
            "H4共振加分": round(h4_resonance_bonus, 2),
            "H4共振v2分": round(h4_res_v2_score, 3),
            "企稳收敛型": is_converging,
            "企稳收敛加分": round(convergence_bonus, 2),
            "摆动点距当前根数": swing_age if swing_age is not None else -1,
            "摆动点时效扣分": round(swing_staleness_penalty, 2),
            "量能形态": vol_pattern_tag,
            "量能形态加分": round(vol_pattern_bonus, 2),
            "MACD逆向扣分": round(macd_adverse_penalty, 2),
            "信号通路": signal_path,
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
            "hourly_ema_pos_score": round(ema_pos_score, 2),
            "hourly_squeeze_duration_bonus": round(squeeze_duration_bonus, 2),
            "hourly_fib_bonus": round(fib_bonus, 2),
            "hourly_vwap_bonus": round(vwap_bonus, 2),
            "hourly_vp_bonus": round(vp_bonus, 2),
            "hourly_h4_bonus": round(h4_resonance_bonus, 2),
            "hourly_convergence_bonus": round(convergence_bonus, 2),
            "hourly_swing_stale_penalty": round(swing_staleness_penalty, 2),
            "hourly_vol_pattern_bonus": round(vol_pattern_bonus, 2),
        }
        # v3.1: 成交量买卖压力分析
        if bool(self.config.get("enable_volume_pressure", True)):
            buy_vol, sell_vol = _calc_buy_sell_pressure(rows[-24:])
            details["买量占比"] = round(buy_vol / max(buy_vol + sell_vol, 1), 3)
            details["主动买卖比"] = round(buy_vol / max(sell_vol, 1), 2)

        # v3.1: ATR 止损/止盈建议（基础版本，兼容保留）
        if bool(self.config.get("enable_atr_target_suggestion", True)):
            sl_mult = float(self.config.get("stop_atr_mult", 2.0))
            tp_mult = float(self.config.get("target_atr_mult", 3.0))
            if d1_direction == "bull":
                details["ATR止损"] = round(cur_close - atr * sl_mult, 6)
                details["ATR止盈"] = round(cur_close + atr * tp_mult, 6)
            else:
                details["ATR止损"] = round(cur_close + atr * sl_mult, 6)
                details["ATR止盈"] = round(cur_close - atr * tp_mult, 6)
            details["ATR止损%"] = round(atr * sl_mult / cur_close * 100, 2)
            details["ATR止盈%"] = round(atr * tp_mult / cur_close * 100, 2)

        # ── P2-2: 动态止损止盈（联动 H1 swing / Fib / VP_POC / BB）─────────
        # 多头：止损取 max(swing_low, cur_close - 1.5×ATR)（更紧的）
        #       止盈取 min(BB上轨, fib_top_price, VP_POC*1.05) 中最近且有效的目标
        # 空头：止损 min(swing_high, cur_close + 1.5×ATR)
        #       止盈 max(BB下轨, fib_base_price, VP_POC*0.95)
        if bool(self.config.get("dynamic_sl_tp_enabled", True)):
            try:
                if d1_direction == "bull":
                    # 止损候选
                    sl_swing  = swing_extreme - atr * 0.5 if swing_extreme > 0 else 0.0
                    sl_recent = min(stab_lows) - atr * 0.3 if stab_lows else 0.0
                    sl_atr    = cur_close - atr * 1.5
                    sl_dyn    = max(sl_swing, sl_recent, sl_atr)  # 最紧（最高）的止损
                    # 止盈候选：取所有高于 cur_close 的合理目标，选最近的
                    tp_candidates = []
                    if bb_upper > cur_close:                 tp_candidates.append(("BB上轨", bb_upper))
                    if vp_poc > 0 and vp_poc * 1.05 > cur_close:
                        tp_candidates.append(("VP_POC×1.05", vp_poc * 1.05))
                    if vwap_val > 0 and vwap_val * 1.03 > cur_close:
                        tp_candidates.append(("VWAP×1.03", vwap_val * 1.03))
                    # ATR×3 作为兜底
                    tp_candidates.append(("ATR×3", cur_close + atr * 3.0))
                    tp_dyn_name, tp_dyn = min(tp_candidates, key=lambda x: x[1])
                    risk   = cur_close - sl_dyn
                    reward = tp_dyn - cur_close
                else:
                    sl_swing  = swing_extreme + atr * 0.5 if swing_extreme > 0 else 0.0
                    sl_recent = max(stab_highs) + atr * 0.3 if stab_highs else 0.0
                    sl_atr    = cur_close + atr * 1.5
                    sl_dyn    = min(sl_swing, sl_recent, sl_atr) if sl_swing > 0 else min(sl_recent, sl_atr)
                    tp_candidates = []
                    if bb_lower > 0 and bb_lower < cur_close:
                        tp_candidates.append(("BB下轨", bb_lower))
                    if vp_poc > 0 and vp_poc * 0.95 < cur_close:
                        tp_candidates.append(("VP_POC×0.95", vp_poc * 0.95))
                    if vwap_val > 0 and vwap_val * 0.97 < cur_close:
                        tp_candidates.append(("VWAP×0.97", vwap_val * 0.97))
                    tp_candidates.append(("ATR×3", cur_close - atr * 3.0))
                    tp_dyn_name, tp_dyn = max(tp_candidates, key=lambda x: x[1])
                    risk   = sl_dyn - cur_close
                    reward = cur_close - tp_dyn
                rr = reward / risk if risk > 0 else 0.0
                details["动态止损"]    = round(sl_dyn, 6)
                details["动态止盈"]    = round(tp_dyn, 6)
                details["动态止盈来源"] = tp_dyn_name
                details["动态止损%"]   = round(abs(cur_close - sl_dyn) / cur_close * 100.0, 2)
                details["动态止盈%"]   = round(abs(tp_dyn - cur_close) / cur_close * 100.0, 2)
                details["动态盈亏比"]  = round(rr, 2)
                # 盈亏比过低的信号扣分（不硬拒）
                min_rr = float(self.config.get("min_rr_ratio", 1.5))
                if 0 < rr < min_rr:
                    rr_pen = round((min_rr - rr) * 6.0, 2)
                    h1_score = max(0.0, h1_score - rr_pen)
                    details["盈亏比扣分"] = rr_pen
                    details["小时线评分"] = round(h1_score, 2)
            except Exception:
                pass

        return True, h1_score, details, factor_scores

    # ══════════════════════════════════════════════════════════════════════════
    # Step 3：15m 入场时机确认（新增）
    # ══════════════════════════════════════════════════════════════════════════

    def _check_15m_entry_timing(
        self, rows: List, d1_direction: str,
        h1_stab_top: float = 0.0, h1_stab_bot: float = 0.0,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        15m 层不做淘汰，只返回 0~100 的时机分和详情。
        判据（v4.5 加入"突破前一刻"特征）：
          1. EMA8/EMA21 金叉/死叉方向（25分）
          2. 末根量能放大 vs 均量（20分）
          3. 末根K线收盘方向一致（15分）
          4. StochRSI/RSI 反转动量（取大去重，25分）
          5. 15m MACD 方向确认（10分）
          6. P1-3: 价格接近H1企稳上沿/下沿 + 量能放大 = 突破前一刻（最大12分）
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

        # ── 15m RSI动量：从低位回升确认方向 ─────────────────────────────────
        m15_rsi_per      = int(self.config.get("m15_rsi_period", 14))
        m15_rsi_oversold = float(self.config.get("m15_rsi_oversold", 35.0))
        rsi_momentum_score = 0.0
        rsi_momentum_ok    = False
        if len(closes) >= m15_rsi_per + 3:
            rsi_now  = _calc_rsi_wilder(closes,       m15_rsi_per)
            rsi_prev = _calc_rsi_wilder(closes[:-1],  m15_rsi_per)
            rsi_prev2= _calc_rsi_wilder(closes[:-2],  m15_rsi_per)
            if d1_direction == "bull":
                # 多头：RSI从超卖区（<m15_rsi_oversold）开始回升
                rsi_rising = rsi_now > rsi_prev >= rsi_prev2   # 连续回升
                if rsi_prev2 < m15_rsi_oversold and rsi_rising:
                    rsi_momentum_score = 15.0                   # 超卖区启动：满分
                    rsi_momentum_ok    = True
                elif rsi_now > rsi_prev:                        # 仅单步回升：半分
                    rsi_momentum_score = 7.0
                    rsi_momentum_ok    = rsi_prev < m15_rsi_oversold + 10
            else:
                # 空头：RSI从超买区（>100-m15_rsi_oversold）开始回落
                overbought = 100.0 - m15_rsi_oversold
                rsi_falling = rsi_now < rsi_prev <= rsi_prev2
                if rsi_prev2 > overbought and rsi_falling:
                    rsi_momentum_score = 15.0
                    rsi_momentum_ok    = True
                elif rsi_now < rsi_prev:
                    rsi_momentum_score = 7.0
                    rsi_momentum_ok    = rsi_prev > overbought - 10

        # ── StochRSI %K：超卖(<20)多头、超买(>80)空头 = 反转时机最佳────────
        stoch_rsi_period = int(self.config.get("m15_stoch_rsi_period", 14))
        stoch_lookback   = int(self.config.get("m15_stoch_lookback", 14))
        stoch_smooth     = int(self.config.get("m15_stoch_smooth", 3))
        stoch_k = _calc_stoch_rsi_k(closes, stoch_rsi_period, stoch_lookback, stoch_smooth)
        stoch_ok = False
        stoch_score = 0.0
        if stoch_k >= 0:                                      # -1 = 数据不足
            if d1_direction == "bull":
                # 多头：StochRSI从超卖区反弹（K<20=满分，20~40=半分，>40=不加分）
                if stoch_k < 20:
                    stoch_score = 20.0
                    stoch_ok    = True
                elif stoch_k < 40:
                    stoch_score = 20.0 * (1.0 - (stoch_k - 20.0) / 20.0)
                    stoch_ok    = stoch_score >= 10.0
            else:
                # 空头：StochRSI从超买区回落（K>80=满分，60~80=半分，<60=不加分）
                if stoch_k > 80:
                    stoch_score = 20.0
                    stoch_ok    = True
                elif stoch_k > 60:
                    stoch_score = 20.0 * ((stoch_k - 60.0) / 20.0)
                    stoch_ok    = stoch_score >= 10.0

        # \u2500\u2500 v4.2: 15m MACD\u52a8\u80fd\u786e\u8ba4\uff08\u6bd4EMA\u91d1\u53c9\u66f4\u7075\u654f\u7684\u5165\u573a\u4fe1\u53f7\uff09\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
        macd_conf_score = 0.0
        macd_conf_tag   = ""
        if bool(self.config.get("use_15m_macd_confirm", True)) and len(closes) >= 40:
            m15_macd_hist = _calc_macd_hist_series(closes, fast=12, slow=26, signal=9)
            if len(m15_macd_hist) >= 3:
                mh_last = m15_macd_hist[-1]
                mh_prev = m15_macd_hist[-2]
                mh_prev2= m15_macd_hist[-3]
                # \u91d1\u53c9/\u6b7b\u53c9\uff1a\u4e0a\u6839\u623f\u4e0b\u6839\u7a7f\u96f6
                m15_macd_golden = mh_prev <= 0 < mh_last
                m15_macd_death  = mh_prev >= 0 > mh_last
                # \u65b9\u5411\u4e00\u81f4\uff1a\u67f1\u4e0e\u8d8b\u52bf\u540c\u5411\u4e14\u6b63\u5728\u53d1\u5c55
                m15_macd_bull_align = mh_last > 0 and mh_last > mh_prev
                m15_macd_bear_align = mh_last < 0 and mh_last < mh_prev
                cross_bonus = float(self.config.get("m15_macd_cross_bonus", 10.0))
                align_bonus = float(self.config.get("m15_macd_align_bonus",  5.0))
                if d1_direction == "bull":
                    if m15_macd_golden:
                        macd_conf_score = cross_bonus
                        macd_conf_tag   = f"15m MACD\u91d1\u53c9(+{cross_bonus:.0f})"
                    elif m15_macd_bull_align:
                        macd_conf_score = align_bonus
                        macd_conf_tag   = f"15m MACD\u591a\u5934(+{align_bonus:.0f})"
                else:
                    if m15_macd_death:
                        macd_conf_score = cross_bonus
                        macd_conf_tag   = f"15m MACD\u6b7b\u53c9(+{cross_bonus:.0f})"
                    elif m15_macd_bear_align:
                        macd_conf_score = align_bonus
                        macd_conf_tag   = f"15m MACD\u7a7a\u5934(+{align_bonus:.0f})"

        # P0-B7 \u4fee\u590d\uff1aRSI \u56de\u5347\u4e0e StochRSI \u8d85\u5356\u5f80\u5f80\u540c\u65f6\u53d1\u751f\uff08\u540c\u6e90\u4fe1\u53f7\uff09\uff0c\u539f\u6765\u76f4\u63a5\u76f8\u52a0\u4f1a\u53cc\u91cd\u8ba1\u5206\u3002
        # \u6539\u4e3a\u300c\u540c\u6e90\u53bb\u91cd\u300d\uff1a\u53d6\u4e24\u8005\u6700\u5927\u503c\u518d\u4e58 1.0\uff1b\u7f3a\u4e00\u65f6\u53e6\u4e00\u8005\u6253 0.75 \u6298\u6263\u3002
        cross_score  = 25.0 if cross_match else (15.0 if align_match else 0.0)
        vol_score    = 20.0 * _clamp((vol_ratio - 1.0) / 1.0, 0.0, 1.0)
        dir_score    = 15.0 if closes_last_ok else 0.0
        # \u540c\u6e90\u53bb\u91cd\uff1a\u4e24\u4e2a\u4fe1\u53f7\u90fd\u89e6\u53d1\u65f6\u53d6\u5927\u503c\uff1b\u53ea\u6709\u4e00\u4e2a\u65f6\u964d\u6743 0.75
        if rsi_momentum_score > 0 and stoch_score > 0:
            momentum_score = max(rsi_momentum_score, stoch_score)
        else:
            momentum_score = max(rsi_momentum_score, stoch_score) * 0.75
        momentum_score = momentum_score * 1.25

        # \u2500\u2500 P1-3: "\u7a81\u7834\u524d\u4e00\u523b"\u7279\u5f81 = \u63a5\u8fd1H1\u4f01\u7a33\u4e0a\u6cbf/\u4e0b\u6cbf + \u91cf\u80fd\u653e\u5927 \u2500\u2500\u2500\u2500\u2500\u2500
        # \u4ef7\u683c\u7d27\u8d34\u4f01\u7a33\u4e0a\u6cbf\uff08\u591a\u5934\uff09\u6216\u4e0b\u6cbf\uff08\u7a7a\u5934\uff09\uff0c\u4e14 15m \u672b\u6839\u91cf\u80fd \u2265 \u5747\u91cf\u00d71.30 \u2192 \u5373\u5c06\u7a81\u7834
        breakout_imminent_score = 0.0
        breakout_imminent_tag   = ""
        if (bool(self.config.get("m15_breakout_imminent_enabled", True))
                and h1_stab_top > 0 and h1_stab_bot > 0):
            cur_close_15m = closes[-1] if closes else 0.0
            prox_pct = float(self.config.get("m15_breakout_proximity_pct", 0.4))
            vol_min  = float(self.config.get("m15_breakout_vol_min", 1.30))
            max_b    = float(self.config.get("m15_breakout_max_bonus", 12.0))
            if d1_direction == "bull":
                # \u8ddd\u4f01\u7a33\u4e0a\u6cbf\u7684%\uff08\u8d1f=\u5df2\u8d85\u8fc7\u4e0a\u6cbf\uff0c\u6b63=\u63a5\u8fd1\u672a\u5230\uff09
                gap_to_top = (h1_stab_top - cur_close_15m) / max(h1_stab_top, 1e-9) * 100.0
                if -prox_pct <= gap_to_top <= prox_pct and vol_ratio >= vol_min:
                    # \u8d8a\u63a5\u8fd1\u4e0a\u6cbf\u3001\u91cf\u80fd\u8d8a\u8db3\uff0c\u5206\u6570\u8d8a\u9ad8
                    proximity_quality = 1.0 - abs(gap_to_top) / prox_pct
                    vol_quality       = _clamp((vol_ratio - vol_min) / max(vol_min * 0.5, 1e-9), 0.0, 1.0)
                    breakout_imminent_score = round(max_b * (proximity_quality * 0.6 + vol_quality * 0.4), 2)
                    breakout_imminent_tag   = f"\u63a5\u8fd1\u4e0a\u6cbf({gap_to_top:+.2f}%)+\u91cf\u80fd{vol_ratio:.2f}x"
            else:
                # \u7a7a\u5934\uff1a\u8ddd\u4f01\u7a33\u4e0b\u6cbf
                gap_to_bot = (cur_close_15m - h1_stab_bot) / max(h1_stab_bot, 1e-9) * 100.0
                if -prox_pct <= gap_to_bot <= prox_pct and vol_ratio >= vol_min:
                    proximity_quality = 1.0 - abs(gap_to_bot) / prox_pct
                    vol_quality       = _clamp((vol_ratio - vol_min) / max(vol_min * 0.5, 1e-9), 0.0, 1.0)
                    breakout_imminent_score = round(max_b * (proximity_quality * 0.6 + vol_quality * 0.4), 2)
                    breakout_imminent_tag   = f"\u63a5\u8fd1\u4e0b\u6cbf({gap_to_bot:+.2f}%)+\u91cf\u80fd{vol_ratio:.2f}x"

        timing_score = round(min(
            cross_score + vol_score + dir_score + momentum_score + macd_conf_score
            + breakout_imminent_score,
            100.0,
        ), 2)

        details = {
            "15m EMA\u91d1\u53c9": golden_cross, "15m EMA\u6b7b\u53c9": death_cross,
            "15m EMA\u65b9\u5411\u4e00\u81f4": align_match,
            "EMA\u91d1\u53c9": cross_match,
            "\u91cf\u80fd\u7cfb\u6570": round(vol_ratio, 2),
            "K\u7ebf\u65b9\u5411\u4e00\u81f4": closes_last_ok,
            "15m_RSI\u52a8\u91cf\u4fe1\u53f7": rsi_momentum_ok,
            "15m_RSI\u52a8\u91cf\u5f97\u5206": round(rsi_momentum_score, 2),
            "15m_StochRSI_K": stoch_k if stoch_k >= 0 else None,
            "15m_StochRSI\u8d85\u5356\u4fe1\u53f7": stoch_ok,
            "15m_Stoch\u5f97\u5206": round(stoch_score, 2),
            "15m_MACD\u786e\u8ba4": macd_conf_tag,
            "15m_MACD\u5f97\u5206": round(macd_conf_score, 2),
            "15m\u7a81\u7834\u524d\u4e00\u523b": breakout_imminent_tag,
            "15m\u7a81\u7834\u524d\u52a0\u5206": round(breakout_imminent_score, 2),
            "15m\u65f6\u673a\u5206": timing_score,
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
        threshold = float(self.config.get("btc_dump_block_threshold_pct", -5.0))
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
            - abs(pullback - (float(self.config.get("h1_pullback_min_pct", 1.5))
                             + float(self.config.get("h1_pullback_max_pct", 10.0))) / 2.0) * 5.0
            - abs(band_pos - target_pos) * 40.0,
            20.0, 98.0,
        )
        freshness_score = _clamp(
            100.0
            - max(float(h1_details.get("BB带宽%", 0) or 0) - float(self.config.get("h1_bb_squeeze_pct", 9.0)), 0.0) * 8.0
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
    results: List[float] = []
    for offset in range(last_n - 1, -1, -1):
        end = len(closes) - offset
        if end < period * 2 + 5:
            results.append(0.0)
            continue
        adx_v, _, _ = _calc_adx(closes[:end], highs[:end], lows[:end], period)
        results.append(adx_v)
    return results


def _calc_adx_series_fast(
    closes: List[float], highs: List[float], lows: List[float], period: int = 14, last_n: int = 65
) -> List[float]:
    """
    #8 fix: O(n) 单次完整 Wilder 推算，返回最近 last_n 根的 ADX 序列。
    比 _calc_adx_series（逐偏移重算 last_n 次）快约 last_n 倍。
    """
    n = len(closes)
    min_len = period * 2 + 5
    if n < min_len:
        return [0.0] * min(last_n, n)
    try:
        hi  = np.asarray(highs,  dtype=float)
        lo  = np.asarray(lows,   dtype=float)
        cl  = np.asarray(closes, dtype=float)
        # True Range
        prev_cl = np.concatenate(([cl[0]], cl[:-1]))
        tr  = np.maximum(hi - lo, np.maximum(np.abs(hi - prev_cl), np.abs(lo - prev_cl)))
        # Directional movement
        up_move   = hi[1:] - hi[:-1]
        down_move = lo[:-1] - lo[1:]
        pdm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        ndm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        tr1 = tr[1:]

        # Wilder smoothing in one pass
        atr_s   = np.zeros(len(tr1))
        pdm_s   = np.zeros(len(tr1))
        ndm_s   = np.zeros(len(tr1))
        atr_s[0]  = float(np.sum(tr1[:period]))
        pdm_s[0]  = float(np.sum(pdm[:period]))
        ndm_s[0]  = float(np.sum(ndm[:period]))
        for i in range(1, len(tr1)):
            atr_s[i] = atr_s[i - 1] - atr_s[i - 1] / period + tr1[i]
            pdm_s[i] = pdm_s[i - 1] - pdm_s[i - 1] / period + pdm[i]
            ndm_s[i] = ndm_s[i - 1] - ndm_s[i - 1] / period + ndm[i]

        with np.errstate(divide="ignore", invalid="ignore"):
            di_p = np.where(atr_s > 0, pdm_s / atr_s * 100.0, 0.0)
            di_n = np.where(atr_s > 0, ndm_s / atr_s * 100.0, 0.0)
            dx   = np.where((di_p + di_n) > 0, np.abs(di_p - di_n) / (di_p + di_n) * 100.0, 0.0)

        # ADX = Wilder-smoothed DX, starting from index period-1
        adx_series: List[float] = []
        if len(dx) < period:
            return [0.0] * min(last_n, n)
        adx_val = float(np.mean(dx[:period]))
        adx_series.append(adx_val)
        for i in range(period, len(dx)):
            adx_val = (adx_val * (period - 1) + dx[i]) / period
            adx_series.append(adx_val)

        return adx_series[-last_n:] if len(adx_series) >= last_n else adx_series
    except Exception:
        return [0.0] * min(last_n, n)


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
    # 至少需要 period+1 根才能计算第一个 delta；全量历史越长 Wilder 越准确，不做截断
    if len(data) < period + 1:
        return 50.0
    deltas = np.diff(np.asarray(data, dtype=float))
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
    v4.2 \u91cd\u5199\uff1a\u57fa\u4e8e\u771f\u5b9e\u6446\u52a8\u70b9\u7684RSI\u80cc\u79bb\u68c0\u6d4b\uff08\u539f\u4e8c\u5206\u6cd5\u6781\u4e0d\u53ef\u9760\uff09\u3002

    \u7b97\u6cd5\uff1a
      1. \u8ba1\u7b97\u8fc7\u53bb lookback \u6839\u5185\u6bcf\u6839\u7684RSI\u5e8f\u5217
      2. \u627e\u5230\u8fd1\u671f2\u4e2a\u771f\u5b9e\u6446\u52a8\u4f4e/\u9ad8\u70b9\uff08\u5de6\u53f3\u5404 confirm_bars \u6839\u786e\u8ba4\uff09
      3. \u6bd4\u8f83\u8fd9\u4e24\u4e2a\u6446\u52a8\u70b9\u7684\u4ef7\u683c\u548cRSI\uff1a
         \u591a\u5934\u80cc\u79bb\uff1a\u4ef7\u683c Low2 < Low1 \u4e14 RSI_Low2 > RSI_Low1+\u9608\u503c
         \u7a7a\u5934\u80cc\u79bb\uff1a\u4ef7\u683c High2 > High1 \u4e14 RSI_High2 < RSI_High1-\u9608\u503c
    \u8fd4\u56de True/False\u3002
    """
    n = len(closes)
    min_needed = rsi_period + lookback + 10
    if n < min_needed:
        return False
    try:
        # \u2460 \u8ba1\u7b97\u5b8c\u6574RSI\u5e8f\u5217\uff08lookback+rsi_period+5\u6839\u4ee5\u4fdd\u8bc1Wilder\u70ed\u8eab\u671f\uff09
        seg_len = lookback + rsi_period + 10
        seg     = closes[-seg_len:]
        deltas  = np.diff(np.asarray(seg, dtype=float))
        gains   = np.where(deltas > 0, deltas, 0.0)
        losses  = np.where(deltas < 0, -deltas, 0.0)
        avg_g   = float(np.mean(gains[:rsi_period]))
        avg_l   = float(np.mean(losses[:rsi_period]))
        rsi_vals: List[float] = []
        for g, l in zip(gains[rsi_period:], losses[rsi_period:]):
            avg_g = (avg_g * (rsi_period - 1) + float(g)) / rsi_period
            avg_l = (avg_l * (rsi_period - 1) + float(l)) / rsi_period
            rs    = avg_g / max(avg_l, 1e-9)
            rsi_vals.append(100.0 - 100.0 / (1.0 + rs))

        # \u5bf9\u9f50\u5230\u6700\u8fd1 lookback \u6839
        price_seg = closes[-lookback:]
        lows_seg  = lows[-lookback:]
        highs_seg = highs[-lookback:]
        rsi_seg   = rsi_vals[-lookback:] if len(rsi_vals) >= lookback else rsi_vals

        if len(rsi_seg) < 8:
            return False

        confirm = 2   # \u6446\u52a8\u70b9\u786e\u8ba4\u6839\u6570

        if direction == "bull":
            # \u591a\u5934\u80cc\u79bb\uff1a\u627e\u76845\u4e2a\u771f\u5b9e\u6446\u52a8\u4f4e\u70b9\uff0c\u5224\u65ad\u4ef7\u683c\u65b0\u4f4e\u4f46RSI\u672a\u65b0\u4f4e
            swing_lows: List[Tuple[int, float, float]] = []  # (idx, price_low, rsi_val)
            for i in range(confirm, len(lows_seg) - confirm):
                if all(lows_seg[j] > lows_seg[i] for j in range(i - confirm, i)) and \
                   all(lows_seg[j] > lows_seg[i] for j in range(i + 1, i + confirm + 1)):
                    rsi_at_i = rsi_seg[i] if i < len(rsi_seg) else 50.0
                    swing_lows.append((i, lows_seg[i], rsi_at_i))

            if len(swing_lows) >= 2:
                # \u53d6\u6700\u8fd1\u4e24\u4e2a\u6446\u52a8\u4f4e\u70b9
                sl1, sl2 = swing_lows[-2], swing_lows[-1]
                price_lower = sl2[1] < sl1[1]           # \u4ef7\u683c\u521b\u65b0\u4f4e
                rsi_higher  = sl2[2] > sl1[2] + 2.0     # RSI\u672a\u521b\u65b0\u4f4e\uff08\u9608\u5dee\u81f3\u5c112\u70b9\uff09
                return price_lower and rsi_higher

        else:
            # \u7a7a\u5934\u80cc\u79bb\uff1a\u627e\u771f\u5b9e\u6446\u52a8\u9ad8\u70b9\uff0c\u5224\u65ad\u4ef7\u683c\u65b0\u9ad8\u4f46RSI\u672a\u65b0\u9ad8
            swing_highs: List[Tuple[int, float, float]] = []
            for i in range(confirm, len(highs_seg) - confirm):
                if all(highs_seg[j] < highs_seg[i] for j in range(i - confirm, i)) and \
                   all(highs_seg[j] < highs_seg[i] for j in range(i + 1, i + confirm + 1)):
                    rsi_at_i = rsi_seg[i] if i < len(rsi_seg) else 50.0
                    swing_highs.append((i, highs_seg[i], rsi_at_i))

            if len(swing_highs) >= 2:
                sh1, sh2 = swing_highs[-2], swing_highs[-1]
                price_higher = sh2[1] > sh1[1]           # \u4ef7\u683c\u521b\u65b0\u9ad8
                rsi_lower    = sh2[2] < sh1[2] - 2.0     # RSI\u672a\u521b\u65b0\u9ad8
                return price_higher and rsi_lower

        return False
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


def _find_swing_high_age(highs: List[float], confirm_bars: int) -> Optional[int]:
    """返回最近摆动高点距列表末尾的距离（根数）。"""
    n = len(highs)
    best_val = None
    best_idx = None
    for i in range(confirm_bars, n - confirm_bars):
        left_ok  = all(highs[j] < highs[i] for j in range(i - confirm_bars, i))
        right_ok = all(highs[j] < highs[i] for j in range(i + 1, i + confirm_bars + 1))
        if left_ok and right_ok:
            if best_val is None or highs[i] > best_val:
                best_val = highs[i]
                best_idx = i
    return (n - 1 - best_idx) if best_idx is not None else None


def _find_swing_low_age(lows: List[float], confirm_bars: int) -> Optional[int]:
    """返回最近摆动低点距列表末尾的距离（根数）。"""
    n = len(lows)
    best_val = None
    best_idx = None
    for i in range(confirm_bars, n - confirm_bars):
        left_ok  = all(lows[j] > lows[i] for j in range(i - confirm_bars, i))
        right_ok = all(lows[j] > lows[i] for j in range(i + 1, i + confirm_bars + 1))
        if left_ok and right_ok:
            if best_val is None or lows[i] < best_val:
                best_val = lows[i]
                best_idx = i
    return (n - 1 - best_idx) if best_idx is not None else None


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


def _median_positive(values: List[float]) -> float:
    """正值中位数，减少pump/dump极值对成交量基线的影响。"""
    cleaned = [float(v) for v in values if float(v) >= 0]
    return float(np.median(cleaned)) if cleaned else 0.0


def _calc_stoch_rsi_k(
    closes: List[float],
    rsi_period: int = 14,
    stoch_period: int = 14,
    smooth_k: int = 3,
) -> float:
    """
    StochRSI %K（平滑后）= 随机振荡 × RSI，比普通RSI更灵敏捕捉超卖/超买反转。
    算法：
      1. Wilder RSI序列（逐根计算）
      2. 对RSI序列做stoch_period窗口的随机振荡：raw = (RSI - min) / (max - min) × 100
      3. 对raw序列做smooth_k期SMA平滑得到%K
    返回：0~100 的%K值，数据不足时返回 -1.0。
    """
    min_len = rsi_period + stoch_period + smooth_k + 5
    if len(closes) < min_len:
        return -1.0

    # 1. 逐根Wilder RSI序列
    rsi_series: List[float] = []
    prev_avg_gain: Optional[float] = None
    prev_avg_loss: Optional[float] = None
    for i in range(1, len(closes)):
        delta = closes[i] - closes[i - 1]
        gain  = max(delta, 0.0)
        loss  = max(-delta, 0.0)
        if prev_avg_gain is None:
            if i < rsi_period:
                continue
            # 初始均值：前rsi_period根的SMA
            gains  = [max(closes[j] - closes[j - 1], 0.0) for j in range(i - rsi_period + 1, i + 1)]
            losses = [max(closes[j - 1] - closes[j], 0.0) for j in range(i - rsi_period + 1, i + 1)]
            prev_avg_gain = sum(gains) / rsi_period
            prev_avg_loss = sum(losses) / rsi_period
        else:
            prev_avg_gain = (prev_avg_gain * (rsi_period - 1) + gain) / rsi_period
            prev_avg_loss = (prev_avg_loss * (rsi_period - 1) + loss) / rsi_period
        if prev_avg_loss < 1e-12:
            rsi_series.append(100.0)
        else:
            rs = prev_avg_gain / prev_avg_loss
            rsi_series.append(100.0 - 100.0 / (1.0 + rs))

    if len(rsi_series) < stoch_period + smooth_k:
        return -1.0

    # 2. 随机振荡序列
    raw_stoch: List[float] = []
    for i in range(stoch_period - 1, len(rsi_series)):
        window  = rsi_series[i - stoch_period + 1 : i + 1]
        lo, hi  = min(window), max(window)
        if hi - lo < 1e-9:
            raw_stoch.append(50.0)          # 极度平稳时返回中值
        else:
            raw_stoch.append((rsi_series[i] - lo) / (hi - lo) * 100.0)

    if len(raw_stoch) < smooth_k:
        return -1.0

    # 3. smooth_k期SMA平滑
    k_val = sum(raw_stoch[-smooth_k:]) / smooth_k
    return round(float(k_val), 2)


def _calc_vwap(
    closes: List[float], highs: List[float], lows: List[float],
    volumes: List[float], lookback: int = 50,
) -> float:
    """
    VWAP（成交量加权平均价）= Σ(典型价×成交量) / Σ成交量。
    典型价 = (H + L + C) / 3。
    代表机构的平均持仓成本，是重要的动态支撑/阻力位。
    """
    n = min(lookback, len(closes))
    if n < 2:
        return closes[-1] if closes else 0.0
    tp_v_sum = 0.0
    v_sum     = 0.0
    for i in range(len(closes) - n, len(closes)):
        tp       = (highs[i] + lows[i] + closes[i]) / 3.0
        vol      = max(volumes[i], 0.0)
        tp_v_sum += tp * vol
        v_sum    += vol
    return tp_v_sum / max(v_sum, 1e-9)


def _calc_vp_poc(
    closes: List[float], highs: List[float], lows: List[float],
    volumes: List[float], lookback: int = 50, bins: int = 20,
) -> float:
    """
    Volume Profile POC（Point of Control）= 成交量最密集的价格区间中点。
    价格在POC附近 = 历史上多空争夺最激烈的真实支撑/阻力区。
    返回POC价格，数据不足或异常时返回0.0。
    """
    n = min(lookback, len(closes))
    if n < 5 or bins < 2:
        return 0.0
    lo = min(lows[len(lows) - n:])
    hi = max(highs[len(highs) - n:])
    if hi <= lo or (hi - lo) < 1e-9:
        return closes[-1] if closes else 0.0
    bin_size = (hi - lo) / bins
    vol_bins  = [0.0] * bins
    for i in range(len(closes) - n, len(closes)):
        typ_price = (highs[i] + lows[i] + closes[i]) / 3.0
        b_idx     = min(int((typ_price - lo) / bin_size), bins - 1)
        vol_bins[b_idx] += max(volumes[i], 0.0)
    max_bin = vol_bins.index(max(vol_bins))
    return lo + (max_bin + 0.5) * bin_size


def _detect_d1_patterns(
    opens: List[float], highs: List[float], lows: List[float],
    closes: List[float], direction: str,
) -> Tuple[str, float]:
    """
    检测日线K线形态，返回 (形态名称, 加分)。
    支持：
      - 锤子线/倒锤子线 Hammer（多头止跌信号）
      - 看涨吞没线 Bullish Engulfing
      - 十字星 Doji（方向不明但企稳）
    空头方向对应：吊颈线 / 看跌吞没线。
    """
    if len(closes) < 2:
        return "", 0.0
    o0, h0, l0, c0 = opens[-1], highs[-1], lows[-1], closes[-1]   # 最新K线
    o1, h1, l1, c1 = opens[-2], highs[-2], lows[-2], closes[-2]   # 前一K线

    body0    = abs(c0 - o0)
    rng0     = h0 - l0
    if rng0 < 1e-9:
        return "", 0.0
    lower_wick0  = min(o0, c0) - l0          # 下影线
    upper_wick0  = h0 - max(o0, c0)          # 上影线
    body_ratio0  = body0 / rng0              # 实体占比

    if direction == "bull":
        # ── 看涨吞没线：当前阳线实体完全包住前一阴线实体 ──
        if c0 > o0 and c1 < o1:             # 当前阳、前一阴
            if o0 <= c1 and c0 >= o1:       # 完全吞没
                return "看涨吞没", 10.0
        # ── 锤子线：下影线 ≥ 实体2倍，上影线 ≤ 实体0.5倍，收阳 ──
        if (lower_wick0 >= body0 * 2.0
                and upper_wick0 <= body0 * 0.5
                and body_ratio0 >= 0.10
                and c0 >= o0):
            return "锤子线", 8.0
        # ── 十字星：实体极小（<5%振幅）且处在支撑区 ──
        if body_ratio0 < 0.05 and lower_wick0 > upper_wick0:
            return "多头十字星", 5.0
    else:
        # ── 看跌吞没线 ──
        if c0 < o0 and c1 > o1:
            if o0 >= c1 and c0 <= o1:
                return "看跌吞没", 10.0
        # ── 吊颈线：上影线 ≥ 实体2倍，下影线 ≤ 实体0.5倍，收阴 ──
        if (upper_wick0 >= body0 * 2.0
                and lower_wick0 <= body0 * 0.5
                and body_ratio0 >= 0.10
                and c0 <= o0):
            return "吊颈线", 8.0
        # ── 空头十字星 ──
        if body_ratio0 < 0.05 and upper_wick0 > lower_wick0:
            return "空头十字星", 5.0

    return "", 0.0


def _detect_nr_bar(highs: List[float], lows: List[float], lookback: int = 7) -> bool:
    """
    NR4/NR7 窄幅K线检测（严格版）：末根振幅必须是最近 lookback 根中最小的。
    lookback=4 → NR4；lookback=7 → NR7（默认）。
    """
    n = len(highs)
    if n < lookback:
        return False
    window_highs = highs[-lookback:]
    window_lows  = lows[-lookback:]
    ranges = [window_highs[i] - window_lows[i] for i in range(lookback)]
    if not ranges or ranges[-1] <= 0:
        return False
    return ranges[-1] == min(ranges)


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
