#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
小月期货多周期布林趋势转折扫描策略 v1.1

v1.1 修复:
  · Step3 起爆点强制要求 MACD 金叉/死叉 + 价格站稳（非二选一）
  · Step2 RSI 可选硬门槛（默认开启，`h1_rsi_gate_enabled`）
  · Step1 增加 ADX 趋势确认 + BBW 收缩检测
  · BTC 环境熔断增加 h1 fallback（修复死代码）
  · 可选 4H 大周期回退（新上市山寨币）
  · 斜率阈值 ATR 自适应
  · hist_strength 改用百分比而非绝对 magic number
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from strategies._shared.indicators import _to_df, _atr, _adx, _rsi_wilder, _clamp

logger = logging.getLogger(__name__)

try:
    from src.scanner.base_scanner import BaseScannerStrategy, ScanCondition
    _HAS_BASE = True
except ImportError:
    BaseScannerStrategy = object; ScanCondition = None; _HAS_BASE = False


CONFIG_SCHEMA: Dict[str, Any] = {
    "preset": {
        "type": "select",
        "default": "custom",
        "label": "参数模板",
        "options": [
            {"label": "自定义", "value": "custom"},
            {"label": "BTC/ETH 主流币稳健版", "value": "major_conservative"},
            {"label": "山寨币趋势进攻版", "value": "altcoin_aggressive"},
        ],
    },
    "min_volume_24h":           {"type": "float", "default": 3_000_000,   "label": "最小24H成交额(USDT)"},
    "min_score":                {"type": "float", "default": 65.0,         "label": "最低输出分数"},
    "top_n":                    {"type": "int",   "default": 12,           "label": "最多输出"},
    "allow_short":              {"type": "bool",  "default": True,          "label": "允许做空"},
    "use_4h_fallback":          {"type": "bool",  "default": True,          "label": "D1不足时用4H做大周期"},

    "boll_period":              {"type": "int",   "default": 20,           "label": "布林轨周期"},
    "boll_std_mult":            {"type": "float", "default": 2.0,          "label": "布林轨标准差"},
    "slope_lookback":           {"type": "int",   "default": 8,            "label": "中轨斜率回溯根数"},
    "slope_min_angle":          {"type": "float", "default": 0.15,         "label": "中轨最小斜率(度)"},
    "slope_atr_scale":          {"type": "bool",  "default": False,        "label": "斜率阈值ATR自适应"},
    "adx_min_trend":            {"type": "float", "default": 16.0,         "label": "ADX最低趋势确认"},
    "adx_period":               {"type": "int",   "default": 14,           "label": "ADX计算周期"},
    "bbw_squeeze_bonus":        {"type": "float", "default": 10.0,         "label": "BBW收缩加分"},

    "mid_band_proximity_atr":   {"type": "float", "default": 1.2,          "label": "中轨接近度(×ATR)"},
    "mid_band_proximity_floor":  {"type": "float", "default": 3.5,          "label": "中轨接近度保底%"},
    "h1_rsi_period":            {"type": "int",   "default": 14,           "label": "1H RSI周期"},
    "h1_rsi_oversold_long":    {"type": "float", "default": 55.0,         "label": "多头RSI上限(回踩中轨时RSI应低于此值)"},
    "h1_rsi_overbought_short":  {"type": "float", "default": 45.0,         "label": "空头RSI下限(回踩中轨时RSI应高于此值)"},
    "h1_rsi_gate_enabled":      {"type": "bool",  "default": False,        "label": "RSI硬门槛(不满足则拒绝)"},

    "m15_macd_fast":            {"type": "int",   "default": 12,           "label": "15m MACD快线"},
    "m15_macd_slow":            {"type": "int",   "default": 26,           "label": "15m MACD慢线"},
    "m15_macd_signal":          {"type": "int",   "default": 9,            "label": "15m MACD信号线"},
    "m15_macd_required":        {"type": "bool",  "default": True,         "label": "强制要求MACD金叉/死叉"},
    "m15_macd_cross_lookback":  {"type": "int",   "default": 10,           "label": "15m MACD交叉回看根数"},
    "m15_vol_surge_ratio":      {"type": "float", "default": 1.2,          "label": "15m放量比"},
    "m15_vol_baseline":         {"type": "int",   "default": 10,           "label": "15m量基线根数"},
    "m15_volume_required":      {"type": "bool",  "default": False,        "label": "强制要求15m放量(软条件)"},
    "m15_price_required":       {"type": "bool",  "default": True,         "label": "强制要求价格站稳中轨"},

    "stop_atr_mult":            {"type": "float", "default": 1.8,          "label": "止损ATR倍数"},
    "take_profit_atr_mult":     {"type": "float", "default": 3.0,          "label": "止盈ATR倍数"},
    "min_stop_loss_pct":        {"type": "float", "default": 0.8,          "label": "最小止损百分比"},
    "max_stop_loss_pct":        {"type": "float", "default": 6.0,          "label": "最大止损百分比"},
    "trail_activate_atr_mult":  {"type": "float", "default": 1.5,          "label": "追踪激活(×ATR)"},
    "trail_distance_atr_mult":  {"type": "float", "default": 0.8,          "label": "追踪距离(×ATR)"},
    "btc_dump_threshold_pct":   {"type": "float", "default": -3.0,         "label": "BTC暴跌阈值%"},
    "btc_pump_threshold_pct":   {"type": "float", "default": 3.0,          "label": "BTC暴涨阈值%"},
    "btc_dump_penalty":         {"type": "float", "default": 10.0,         "label": "BTC暴跌降分"},
    "funding_extreme_pct":      {"type": "float", "default": 0.15,         "label": "极端费率%"},
    "funding_penalty":          {"type": "float", "default": 6.0,          "label": "极端费率降分"},
    "position_size":            {"type": "float", "default": 0.05,         "label": "基准仓位比例"},

    # v1.2 新增
    "m15_macd_converge_lookback": {"type": "int",  "default": 4,           "label": "15m MACD收敛检测回看根数"},
    "m15_ema_fast_cross":       {"type": "bool",  "default": True,         "label": "15m EMA5/10快叉作为备选触发"},
    "h1_rsi_soft_penalty":      {"type": "float", "default": 12.0,         "label": "H1 RSI不满足时软降分(RSI门槛关闭时生效)"},
    "d1_slope_accel_bonus":     {"type": "float", "default": 10.0,         "label": "D1中轨斜率加速加分"},

    # v1.3 新增
    "h4_resonance_enabled":     {"type": "bool",  "default": True,         "label": "H4多周期共振确认"},
    "h4_resonance_bonus":       {"type": "float", "default": 12.0,         "label": "H4共振加分"},
    "h4_diverge_penalty":       {"type": "float", "default": 15.0,         "label": "H4反向惩罚分"},
    "m15_bbw_squeeze_bonus":    {"type": "float", "default": 15.0,         "label": "15m BB挤压扩张加分"},
    "m15_bbw_squeeze_lookback": {"type": "int",   "default": 10,           "label": "15m BB挤压检测回看根数"},
    "candle_quality_enabled":   {"type": "bool",  "default": True,         "label": "K线形态质量评分"},
    "candle_quality_weight":    {"type": "float", "default": 0.10,         "label": "K线形态评分权重(0~0.2)"},
    "rsi_diverge_enabled":      {"type": "bool",  "default": True,         "label": "RSI动量背离过滤"},
    "rsi_diverge_penalty":      {"type": "float", "default": 20.0,         "label": "RSI背离降分"},
    "oi_confirm_enabled":       {"type": "bool",  "default": True,         "label": "OI持仓量同向确认"},
    "oi_confirm_bonus":         {"type": "float", "default": 8.0,          "label": "OI同向加分"},
    "oi_diverge_penalty":       {"type": "float", "default": 8.0,          "label": "OI背离降分"},
    "oi_min_change_pct":        {"type": "float", "default": 2.0,          "label": "OI最小变化%触发判断"},

    # v1.4 新增
    "h4_hard_filter_enabled":   {"type": "bool",  "default": True,         "label": "H4明确反向时硬拒绝"},
    "h4_hard_filter_threshold": {"type": "float", "default": -10.0,        "label": "H4共振分低于此值时拒绝(负数)"},
    "h1_rsi_filter_enabled":    {"type": "bool",  "default": True,         "label": "H1 RSI动量区间过滤"},
    "h1_rsi_bull_min":          {"type": "float", "default": 35.0,         "label": "多头H1 RSI最低值"},
    "h1_rsi_bull_max":          {"type": "float", "default": 75.0,         "label": "多头H1 RSI最高值"},
    "h1_rsi_bear_min":          {"type": "float", "default": 25.0,         "label": "空头H1 RSI最低值(RSI需低于此)"},
    "h1_rsi_bear_max":          {"type": "float", "default": 65.0,         "label": "空头H1 RSI最高值"},
    "m15_bbw_overextend_ratio": {"type": "float", "default": 1.30,         "label": "15m BBW过度扩张倍数阈值"},
    "m15_bbw_overextend_pen":   {"type": "float", "default": 12.0,         "label": "15m BBW过度扩张惩罚分"},

    # v1.5 新增
    "d1_rsi_filter_enabled":    {"type": "bool",  "default": True,         "label": "D1 RSI健康区间过滤"},
    "d1_rsi_bull_min":          {"type": "float", "default": 40.0,         "label": "多头D1 RSI最低值"},
    "d1_rsi_bull_max":          {"type": "float", "default": 80.0,         "label": "多头D1 RSI最高值(超买拒绝)"},
    "d1_rsi_bear_min":          {"type": "float", "default": 20.0,         "label": "空头D1 RSI最低值"},
    "d1_rsi_bear_max":          {"type": "float", "default": 60.0,         "label": "空头D1 RSI最高值"},
    "h1_vol_trend_enabled":     {"type": "bool",  "default": True,         "label": "H1量能趋势验证"},
    "h1_vol_trend_ratio":       {"type": "float", "default": 0.80,         "label": "近5根H1均量/前10根均量最低比值"},
    "m15_hist_consec_enabled":  {"type": "bool",  "default": True,         "label": "15m MACD柱体连续强化过滤"},
    "h1_pullback_zone_enabled": {"type": "bool",  "default": True,         "label": "H1回踩质量区间精细化"},
    "h1_pullback_zone_above_max": {"type": "float", "default": 3.0,        "label": "多头H1收盘高于中轨最大%(过高=错过)"},
    "h1_pullback_zone_below_max": {"type": "float", "default": 0.5,        "label": "多头H1收盘低于中轨最大%(过低=仍跌)"},

    # ── v1.6 P1-1 H1 回踩参考改 D1 EMA20 ────────────────────────────────────
    "h1_use_d1_ema_anchor":     {"type": "bool",  "default": True,         "label": "v1.6 H1回踩参考D1 EMA20(替代H1中轨独立判断)"},
    "h1_d1_anchor_max_atr":     {"type": "float", "default": 1.5,          "label": "H1距D1 EMA锚最大ATR倍数(超出=已远离趋势线)"},

    # ── v1.6 P1-2 swing 摆动点 + Fibonacci 回踩位 ───────────────────────────
    "swing_detection_enabled":  {"type": "bool",  "default": True,         "label": "v1.6启用真实摆动点检测(左右各N根确认)"},
    "swing_confirm_bars":       {"type": "int",   "default": 2,            "label": "摆动点左右确认根数"},
    "swing_lookback":           {"type": "int",   "default": 30,           "label": "swing搜索回溯H1根数"},
    "fib_validate_enabled":     {"type": "bool",  "default": True,         "label": "v1.6启用Fibonacci回踩位验证"},
    "fib_tolerance_pct":        {"type": "float", "default": 2.0,          "label": "Fib位容差%(命中加分)"},
    "fib_max_bonus":            {"type": "float", "default": 12.0,         "label": "Fib命中最大加分(61.8=12, 50=10, 38.2=8)"},

    # ── v1.6 P1-3 量能趋势按回踩段例外 ────────────────────────────────────
    "h1_vol_trend_pullback_relaxed": {"type": "bool", "default": True,     "label": "v1.6 回踩期(价格<中轨)放宽量能门槛"},
    "h1_vol_trend_pullback_factor":  {"type": "float","default": 0.75,     "label": "回踩期量能门槛打折系数(0.80×0.75=0.60)"},

    # ── v1.6 P1-4 触发等级降级改扣分 ────────────────────────────────────
    "trig_consec_fail_penalty": {"type": "float", "default": 5.0,          "label": "v1.6 trig_level=2 时柱体未连续的扣分(替代降级)"},

    # ── v1.6 P1-5 市场状态自适应权重 ────────────────────────────────────
    "market_state_weights_enabled": {"type": "bool", "default": True,      "label": "v1.6启用市场状态自适应评分权重"},

    # ── v1.6 P2-1 H4 共振 v2 多维度评分 ────────────────────────────────────
    "h4_resonance_v2_enabled":  {"type": "bool",  "default": True,         "label": "v1.6 H4共振v2(方向40%+斜率强度30%+MACD柱30%)"},
    "h4_resonance_v2_max":      {"type": "float", "default": 12.0,         "label": "H4共振v2最大加分"},
    "h4_resonance_v2_hard_floor": {"type": "float","default": 0.20,        "label": "H4共振v2硬拒下限(<此值=H4严重逆向)"},

    # ── v1.6 P2-3 VWAP 偏离度 ────────────────────────────────────
    "vwap_validate_enabled":    {"type": "bool",  "default": True,         "label": "v1.6启用VWAP机构成本支撑验证"},
    "vwap_lookback":            {"type": "int",   "default": 50,           "label": "VWAP计算回溯H1根数"},
    "vwap_max_dev_pct":         {"type": "float", "default": 4.0,          "label": "VWAP最大允许偏离%(超出=深度偏离)"},
    "vwap_near_pct":            {"type": "float", "default": 1.5,          "label": "VWAP命中容差%(±此内=机构成本支撑)"},
    "vwap_max_bonus":            {"type": "float","default": 8.0,          "label": "VWAP命中最大加分"},

    # ── v1.6 P2-4 信号持久性 ────────────────────────────────────
    "persistence_enabled":      {"type": "bool",  "default": True,         "label": "v1.6启用信号持久性追踪"},
    "persistence_min_count":    {"type": "int",   "default": 2,            "label": "连续N次扫描出现才稳定"},
    "persistence_bonus":        {"type": "float", "default": 4.0,          "label": "稳定信号加分"},

    # ── v1.6 P2-5 Volume Profile POC ────────────────────────────────────
    "vp_poc_validate_enabled":  {"type": "bool",  "default": True,         "label": "v1.6启用VP_POC成交量集中区验证"},
    "vp_poc_lookback":          {"type": "int",   "default": 50,           "label": "VP_POC回溯H1根数"},
    "vp_poc_bins":              {"type": "int",   "default": 20,           "label": "VP_POC价格分桶数"},
    "vp_poc_tolerance_pct":     {"type": "float", "default": 2.0,          "label": "VP_POC命中容差%"},
    "vp_poc_max_bonus":         {"type": "float", "default": 8.0,          "label": "VP_POC命中最大加分"},

    # ── v1.6 P2-6 多根K线连续蓄势 ────────────────────────────────────
    "candle_consec_enabled":    {"type": "bool",  "default": True,         "label": "v1.6启用多根K线连续蓄势识别"},
    "candle_consec_bonus":      {"type": "float", "default": 6.0,          "label": "连续3根同向K线最大加分"},

    # ── v1.6 P3-1 动态止损止盈 ────────────────────────────────────
    "dynamic_sl_tp_enabled":    {"type": "bool",  "default": True,         "label": "v1.6启用动态止损止盈(联动swing/Fib/BB)"},
    "min_rr_ratio":             {"type": "float", "default": 1.5,          "label": "最小盈亏比(<此值软扣分)"},
    "rr_too_low_penalty":       {"type": "float", "default": 6.0,          "label": "盈亏比过低扣分"},

    # ── v1.6 P3-7 BTC 环境 ATR 自适应 ────────────────────────────────────
    "btc_env_atr_adaptive":     {"type": "bool",  "default": True,         "label": "v1.6 BTC环境阈值随BTC ATR自适应"},
    "btc_env_atr_mult":         {"type": "float", "default": 1.0,          "label": "BTC ATR倍数(阈值=BTC ATR%×此值)"},

    # ── v1.6 P4-5 z-score 趋势加速度 ────────────────────────────────────
    "zscore_health_enabled":    {"type": "bool",  "default": True,         "label": "v1.6启用z-score趋势加速度评估"},
    "zscore_atr_period":        {"type": "int",   "default": 14,           "label": "z-score计算ATR周期"},
    "zscore_optimal_max":       {"type": "float", "default": 0.5,          "label": "z<此值=最优早期入场(满分加分)"},
    "zscore_emerging_max":      {"type": "float", "default": 1.5,          "label": "z=此值=萌芽期(线性递减)"},
    "zscore_extended_max":      {"type": "float", "default": 2.5,          "label": "z=此值=已透支(最大扣分)"},
    "zscore_max_bonus":         {"type": "float", "default": 8.0,          "label": "z-score最大加分"},
    "zscore_max_penalty":       {"type": "float", "default": 12.0,         "label": "z-score最大扣分"},
}

_DEFAULTS = {k: v["default"] for k, v in CONFIG_SCHEMA.items()}

PRESET_CONFIGS: Dict[str, Dict[str, Any]] = {
    "custom": {},
    "major_conservative": {
        "allow_short": False,
        "min_volume_24h": 20_000_000,
        "min_score": 70.0,
        "adx_min_trend": 18.0,
        "slope_min_angle": 0.08,
        "h1_rsi_oversold_long": 58.0,
        "h1_rsi_overbought_short": 42.0,
        "mid_band_proximity_floor": 2.5,
        "m15_macd_cross_lookback": 4,
        "m15_vol_surge_ratio": 1.45,
        "m15_volume_required": True,
        "stop_atr_mult": 1.6,
        "take_profit_atr_mult": 2.8,
        "min_stop_loss_pct": 0.7,
        "max_stop_loss_pct": 4.5,
        "position_size": 0.035,
        "btc_dump_threshold_pct": -2.5,
        "btc_pump_threshold_pct": 2.8,
    },
    "altcoin_aggressive": {
        "allow_short": True,
        "min_volume_24h": 5_000_000,
        "min_score": 60.0,
        "use_4h_fallback": True,
        "adx_min_trend": 14.0,
        "slope_min_angle": 0.06,
        "h1_rsi_oversold_long": 60.0,
        "h1_rsi_overbought_short": 40.0,
        "mid_band_proximity_floor": 3.5,
        "m15_macd_cross_lookback": 5,
        "m15_vol_surge_ratio": 1.2,
        "m15_volume_required": True,
        "stop_atr_mult": 1.9,
        "take_profit_atr_mult": 3.2,
        "min_stop_loss_pct": 0.9,
        "max_stop_loss_pct": 6.5,
        "position_size": 0.05,
        "btc_dump_threshold_pct": -3.5,
        "btc_pump_threshold_pct": 3.5,
    },
}

BATCH_PRESET_LABEL = "批量回测小月布林趋势模板"


def _bollinger(df: pd.DataFrame, period: int = 20, std_m: float = 2.0) -> pd.DataFrame:
    df = df.copy()
    df["boll_mid"] = df["c"].rolling(period).mean()
    df["boll_std"] = df["c"].rolling(period).std()
    df["boll_up"] = df["boll_mid"] + std_m * df["boll_std"]
    df["boll_down"] = df["boll_mid"] - std_m * df["boll_std"]
    return df

def _slope_angle(series: pd.Series, lookback: int) -> float:
    if len(series) < lookback: return 0.0
    y = series.iloc[-lookback:].values
    x = np.arange(lookback)
    sl = np.polyfit(x, y, 1)[0]
    return float(np.degrees(np.arctan(sl / max(np.mean(y), 1e-9))))

# ─── v1.6 新增辅助函数 ────────────────────────────────────────────────────

def _find_swing_high(highs: list, confirm: int = 2) -> Optional[Tuple[int, float]]:
    """真实摆动高点：左右各 confirm 根都比它低。返回 (idx_in_list, value)。"""
    n = len(highs)
    best_idx, best_val = None, None
    for i in range(confirm, n - confirm):
        if all(highs[j] < highs[i] for j in range(i - confirm, i)) and \
           all(highs[j] < highs[i] for j in range(i + 1, i + confirm + 1)):
            if best_val is None or highs[i] > best_val:
                best_val, best_idx = highs[i], i
    return (best_idx, best_val) if best_val is not None else None


def _find_swing_low(lows: list, confirm: int = 2) -> Optional[Tuple[int, float]]:
    """真实摆动低点：左右各 confirm 根都比它高。返回 (idx_in_list, value)。"""
    n = len(lows)
    best_idx, best_val = None, None
    for i in range(confirm, n - confirm):
        if all(lows[j] > lows[i] for j in range(i - confirm, i)) and \
           all(lows[j] > lows[i] for j in range(i + 1, i + confirm + 1)):
            if best_val is None or lows[i] < best_val:
                best_val, best_idx = lows[i], i
    return (best_idx, best_val) if best_val is not None else None


def _calc_vwap(closes: list, highs: list, lows: list, vols: list, lookback: int = 50) -> float:
    """VWAP = Σ(典型价×成交量) / Σ成交量。代表机构成本线。"""
    n = min(lookback, len(closes))
    if n < 2:
        return float(closes[-1]) if closes else 0.0
    tp_v_sum = v_sum = 0.0
    for i in range(len(closes) - n, len(closes)):
        tp = (highs[i] + lows[i] + closes[i]) / 3.0
        v  = max(float(vols[i]), 0.0)
        tp_v_sum += tp * v
        v_sum    += v
    return tp_v_sum / max(v_sum, 1e-9)


def _calc_vp_poc(closes: list, highs: list, lows: list, vols: list,
                 lookback: int = 50, bins: int = 20) -> float:
    """Volume Profile POC = 成交量最密集的价格区间中点。"""
    n = min(lookback, len(closes))
    if n < 5 or bins < 2:
        return 0.0
    seg_lo = min(lows[len(lows) - n:])
    seg_hi = max(highs[len(highs) - n:])
    if seg_hi <= seg_lo or (seg_hi - seg_lo) < 1e-9:
        return float(closes[-1]) if closes else 0.0
    bin_size = (seg_hi - seg_lo) / bins
    vol_bins = [0.0] * bins
    for i in range(len(closes) - n, len(closes)):
        tp = (highs[i] + lows[i] + closes[i]) / 3.0
        b_idx = min(int((tp - seg_lo) / bin_size), bins - 1)
        vol_bins[b_idx] += max(float(vols[i]), 0.0)
    max_bin = vol_bins.index(max(vol_bins))
    return seg_lo + (max_bin + 0.5) * bin_size


def _detect_market_state(d1_atr_pct: float, d1_adx: float, d1_slope_deg: float) -> str:
    """根据 D1 指标自动识别市场状态。"""
    if d1_atr_pct >= 4.5:
        return "volatile"
    if d1_adx >= 25 and abs(d1_slope_deg) >= 0.30:
        return "trending"
    if d1_adx < 18 and abs(d1_slope_deg) < 0.20:
        return "range"
    return "neutral"


def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    """统一列名：回测引擎用 timestamp/open/high/low/close/volume → ts/o/h/l/c/vol"""
    if df is None or len(df) == 0:
        return df
    rename = {"timestamp":"ts","open":"o","high":"h","low":"l","close":"c","volume":"vol"}
    existing = {k: v for k, v in rename.items() if k in df.columns and v not in df.columns}
    if existing:
        df = df.rename(columns=existing)
    # 确保 c 列存在
    if "c" not in df.columns and "close" in df.columns:
        df["c"] = df["close"]
    if "h" not in df.columns and "high" in df.columns:
        df["h"] = df["high"]
    if "l" not in df.columns and "low" in df.columns:
        df["l"] = df["low"]
    if "vol" not in df.columns and "volume" in df.columns:
        df["vol"] = df["volume"]
    return df


class XiaoYueBollMacdScanner(BaseScannerStrategy if _HAS_BASE else object):
    """小月期货多周期布林趋势转折扫描器"""
    required_bars = ["1D", "4H", "1H", "15m"]
    required_bars_limits = {"1D": 80, "4H": 80, "1H": 120, "15m": 80}
    name = "小月期货多周期布林趋势转折"
    description = "D1布林定势→H4共振→1H回踩中轨→15m MACD/BB起爆 | ATR止损+追踪"

    def __init__(self, config: Optional[Dict] = None):
        merged = {**_DEFAULTS, **(config or {})}
        preset_key = str(merged.get("preset", "custom") or "custom")
        preset_values = PRESET_CONFIGS.get(preset_key, {})
        if preset_values:
            merged = {**_DEFAULTS, **preset_values, **(config or {})}
        self.config = merged
        if _HAS_BASE and hasattr(super(), "__init__"):
            try: super().__init__(self.config)
            except Exception: pass
        # P2-4: 信号持久性追踪
        self._signal_history: Dict[str, Tuple[int, float]] = {}     # key=symbol:dir → (last_scan_id, score)
        self._persist_counts: Dict[str, int] = {}
        self._scan_counter: int = 0

    def _init_conditions(self):
        if ScanCondition is None: return
        self.add_condition(ScanCondition(
            name="流动性", field="volume_24h", operator=">=",
            value=self.config["min_volume_24h"]))

    def get_config_schema(self) -> Dict: return dict(CONFIG_SCHEMA)

    def scan_symbol(self, symbol) -> Dict:
        inst_id = getattr(symbol, "inst_id", "")
        ed = getattr(symbol, "extra_data", {}) or {}
        klines = ed.get("klines", {}) or {}
        lp = float(getattr(symbol, "last_price", 0) or 0)

        def _safe_kline(d, *keys):
            for k in keys:
                v = d.get(k)
                if v is not None and (isinstance(v, pd.DataFrame) or (isinstance(v, (list, tuple)) and len(v) > 0)):
                    return v
            return []

        d1_raw = _safe_kline(klines, "1D", "1d")
        h4_raw = _safe_kline(klines, "4H", "4h")
        h1_raw = _safe_kline(klines, "1H", "1h")
        m15_raw = _safe_kline(klines, "15m", "15M")
        d1 = _normalize_cols(d1_raw if isinstance(d1_raw, pd.DataFrame) else _to_df(d1_raw))
        h4 = _normalize_cols(h4_raw if isinstance(h4_raw, pd.DataFrame) else _to_df(h4_raw))
        h1 = _normalize_cols(h1_raw if isinstance(h1_raw, pd.DataFrame) else _to_df(h1_raw))
        m15 = _normalize_cols(m15_raw if isinstance(m15_raw, pd.DataFrame) else _to_df(m15_raw))
        use_4h = bool(self.config.get("use_4h_fallback", True))
        if len(d1) >= 22:
            big_df, big_label = d1, "D1"
        elif use_4h and len(h4) >= 40:
            big_df, big_label = h4, "4H"
        else:
            big_df, big_label = d1, "D1"

        # 截断到计算所需的最小窗口（大幅加速回测）
        big_df = big_df.tail(80) if len(big_df) > 80 else big_df
        h1 = h1.tail(120) if len(h1) > 120 else h1
        m15 = m15.tail(80) if len(m15) > 80 else m15

        fail = lambda r: {"symbol":inst_id,"passed":False,"score":0,"direction":"WAIT","category":"小月期货","details":{"状态":r}}
        min_big = 22 if big_label == "D1" else 40
        if lp <= 0: return fail("缺少最新价")
        if len(big_df) < min_big or len(h1) < 25 or len(m15) < 30:
            return fail(f"数据不足(大周期{len(big_df)}/{min_big},1H{len(h1)}/25,15m{len(m15)}/30)")

        d1_ok, d1_dir, d1_sc, d1_det, atr_pct = self._step1_daily_trend(big_df, big_label)
        if not d1_ok: return fail(f"{big_label}: {d1_det}")

        # 改进E：D1 RSI 健康区间过滤（超买/未确认趋势拒绝）
        if bool(self.config.get("d1_rsi_filter_enabled", True)) and len(big_df) >= 15:
            try:
                d1_rsi_val = _rsi_wilder(big_df["c"], 14)
                if d1_dir == "bull":
                    d1_rsi_lo = float(self.config.get("d1_rsi_bull_min", 40.0))
                    d1_rsi_hi = float(self.config.get("d1_rsi_bull_max", 80.0))
                    if not (d1_rsi_lo <= d1_rsi_val <= d1_rsi_hi):
                        return fail(f"D1 RSI{d1_rsi_val:.1f}超出多头健康区间[{d1_rsi_lo:.0f},{d1_rsi_hi:.0f}]")
                else:
                    d1_rsi_lo = float(self.config.get("d1_rsi_bear_min", 20.0))
                    d1_rsi_hi = float(self.config.get("d1_rsi_bear_max", 60.0))
                    if not (d1_rsi_lo <= d1_rsi_val <= d1_rsi_hi):
                        return fail(f"D1 RSI{d1_rsi_val:.1f}超出空头健康区间[{d1_rsi_lo:.0f},{d1_rsi_hi:.0f}]")
            except Exception:
                pass

        # Step 1b: H4 共振确认
        h4_bonus, h4_det = self._step1b_h4_resonance(h4, d1_dir)
        # 改进A：H4明确反向时硬拒绝（兼容 v2 的 -999 标记）
        if bool(self.config.get("h4_hard_filter_enabled", True)):
            h4_hard_thr = float(self.config.get("h4_hard_filter_threshold", -10.0))
            if h4_bonus <= -999.0 or h4_bonus <= h4_hard_thr:
                return fail(f"H4反向共振拒绝({h4_det})")
        # P2-1: v2 路径的负值要钳制到合理范围避免影响后续累加
        if h4_bonus <= -999.0:
            h4_bonus = float(self.config.get("h4_diverge_penalty", 15.0)) * -1.0

        h1_ok, h1_sc, h1_det, h1_atr = self._step2_h1_pullback(h1, d1_dir)
        if not h1_ok: return fail(f"1H: {h1_det}")

        # 改进B：H1 RSI 动量区间过滤
        if bool(self.config.get("h1_rsi_filter_enabled", True)) and len(h1) >= 15:
            try:
                h1_rsi_val = _rsi_wilder(h1["c"], 14)
                if d1_dir == "bull":
                    rsi_lo = float(self.config.get("h1_rsi_bull_min", 35.0))
                    rsi_hi = float(self.config.get("h1_rsi_bull_max", 75.0))
                    if not (rsi_lo <= h1_rsi_val <= rsi_hi):
                        return fail(f"H1 RSI{h1_rsi_val:.1f}超出多头区间[{rsi_lo:.0f},{rsi_hi:.0f}]")
                else:
                    rsi_lo = float(self.config.get("h1_rsi_bear_min", 25.0))
                    rsi_hi = float(self.config.get("h1_rsi_bear_max", 65.0))
                    if not (rsi_lo <= h1_rsi_val <= rsi_hi):
                        return fail(f"H1 RSI{h1_rsi_val:.1f}超出空头区间[{rsi_lo:.0f},{rsi_hi:.0f}]")
            except Exception:
                pass

        # 改进F：H1 量能趋势验证（近5根均量 ≥ 前10根均量×阈值）
        # P1-3: 回踩期(价格<中轨)放宽门槛，避免误杀真实回踩信号
        if bool(self.config.get("h1_vol_trend_enabled", True)) and len(h1) >= 16:
            try:
                vol_series = h1["vol"]
                recent5_avg = float(vol_series.iloc[-5:].mean())
                prior10_avg = float(vol_series.iloc[-15:-5].mean())
                vol_trend_ratio = float(self.config.get("h1_vol_trend_ratio", 0.80))
                # P1-3: 检测当前是否处于回踩期
                in_pullback = False
                if bool(self.config.get("h1_vol_trend_pullback_relaxed", True)):
                    try:
                        h1_b = _bollinger(h1, int(self.config["boll_period"]), float(self.config["boll_std_mult"]))
                        h1_mid = float(h1_b["boll_mid"].iloc[-1])
                        h1_close = float(h1["c"].iloc[-1])
                        if d1_dir == "bull":
                            in_pullback = h1_close < h1_mid       # 多头回踩 = 价格低于中轨
                        else:
                            in_pullback = h1_close > h1_mid       # 空头反弹 = 价格高于中轨
                    except Exception:
                        in_pullback = False
                effective_ratio = vol_trend_ratio
                if in_pullback:
                    pb_factor = float(self.config.get("h1_vol_trend_pullback_factor", 0.75))
                    effective_ratio = vol_trend_ratio * pb_factor    # 0.80×0.75 = 0.60
                if prior10_avg > 0 and recent5_avg < prior10_avg * effective_ratio:
                    pb_tag = "[回踩段宽松]" if in_pullback else ""
                    return fail(f"H1量能持续性不足{pb_tag}(近5均={recent5_avg:.0f} < {effective_ratio:.0%}×前10均={prior10_avg:.0f})")
            except Exception:
                pass

        # P1-1: H1 回踩参考 D1 EMA20（解决"H1中轨与D1趋势线脱钩"问题）
        # H1 价格距 D1 EMA20 的 ATR 倍数 > h1_d1_anchor_max_atr → 已远离趋势线，拒绝
        if bool(self.config.get("h1_use_d1_ema_anchor", True)) and len(big_df) >= 22:
            try:
                d1_ema_period = int(self.config.get("boll_period", 20))
                d1_ema20 = float(big_df["c"].ewm(span=d1_ema_period, adjust=False).mean().iloc[-1])
                h1_close = float(h1["c"].iloc[-1])
                h1_atr_v = float(_atr(h1, 14))
                d1_anchor_max = float(self.config.get("h1_d1_anchor_max_atr", 1.5))
                if h1_atr_v > 0 and d1_ema20 > 0:
                    dist_atr_mult = abs(h1_close - d1_ema20) / h1_atr_v
                    # 多头：价格应接近或略高于 D1 EMA20；超出 ×ATR 距离 = 远离趋势线
                    # 空头：价格应接近或略低于 D1 EMA20
                    direction_align = (
                        (d1_dir == "bull" and h1_close >= d1_ema20 - h1_atr_v * 0.5) or
                        (d1_dir == "bear" and h1_close <= d1_ema20 + h1_atr_v * 0.5)
                    )
                    if (dist_atr_mult > d1_anchor_max) or (not direction_align):
                        return fail(
                            f"H1距D1 EMA{d1_ema_period}过远({dist_atr_mult:.2f}×ATR > {d1_anchor_max:.2f}"
                            f" 或方向不符) 已脱离趋势线"
                        )
            except Exception:
                pass

        # 改进H：H1 回踩质量精细化（价格与中轨的距离区间）
        if bool(self.config.get("h1_pullback_zone_enabled", True)) and len(h1) >= 1:
            try:
                h1_b = _bollinger(h1, int(self.config["boll_period"]), float(self.config["boll_std_mult"]))
                h1_mid = float(h1_b["boll_mid"].iloc[-1])
                h1_close = float(h1["c"].iloc[-1])
                if h1_mid > 0:
                    h1_dist_pct = (h1_close - h1_mid) / h1_mid * 100  # 正=在中轨上方
                    above_max = float(self.config.get("h1_pullback_zone_above_max", 3.0))
                    below_max = float(self.config.get("h1_pullback_zone_below_max", 0.5))
                    if d1_dir == "bull":
                        if h1_dist_pct > above_max:
                            return fail(f"H1回踩位置偏高({h1_dist_pct:.1f}%>+{above_max:.1f}%，已反弹过多)")
                        if h1_dist_pct < -below_max:
                            return fail(f"H1价格仍低于中轨({h1_dist_pct:.1f}%<-{below_max:.1f}%，回调未企稳)")
                    else:  # bear
                        if h1_dist_pct < -above_max:
                            return fail(f"H1回踩位置偏低({h1_dist_pct:.1f}%<-{above_max:.1f}%，已反弹过多)")
                        if h1_dist_pct > below_max:
                            return fail(f"H1价格仍高于中轨({h1_dist_pct:.1f}%>+{below_max:.1f}%，回调未企稳)")
            except Exception:
                pass

        if h1_atr > atr_pct: atr_pct = h1_atr

        m15_ok, m15_sc, m15_det = self._step3_m15_entry(m15, d1_dir)
        if not m15_ok: return fail(f"15m: {m15_det}")

        # P1-5: 市场状态自适应权重
        d1_metrics = getattr(self, "_last_d1_metrics", {})
        market_state = "neutral"
        if bool(self.config.get("market_state_weights_enabled", True)) and d1_metrics:
            market_state = _detect_market_state(
                d1_metrics.get("atr_pct", 2.0),
                d1_metrics.get("adx", 18.0),
                d1_metrics.get("slope_deg", 0.0),
            )
        # 各市场状态的 (D1, H1, 15m) 权重
        weights_map = {
            "trending": (0.20, 0.25, 0.55),     # 趋势市重起爆
            "range":    (0.35, 0.35, 0.30),     # 震荡市重多周期共振
            "volatile": (0.20, 0.30, 0.50),     # 高波动重 H1+15m
            "neutral":  (0.25, 0.30, 0.45),     # 默认（原 D1 25% + H1 30% + 15m 45%）
        }
        d_w, h_w, m_w = weights_map.get(market_state, weights_map["neutral"])

        # 基础评分（权重市场状态自适应）
        score = round(_clamp(d1_sc * d_w + h1_sc * h_w + m15_sc * m_w, 0, 100), 2)
        score = _clamp(score + h4_bonus, 0, 100)

        # Step 4: K 线形态质量评分（15m 最近一根）
        candle_bonus = 0.0
        candle_det = ""
        if bool(self.config.get("candle_quality_enabled", True)) and len(m15) >= 2:
            cq_raw, candle_det = self._score_candle_quality(m15, d1_dir)
            cq_weight = _clamp(float(self.config.get("candle_quality_weight", 0.10)), 0, 0.20)
            candle_bonus = cq_raw * cq_weight   # 最大贡献 = 50 * 0.10 = 5 分
            score = _clamp(score + candle_bonus, 0, 100)

        # ── P2-6: 多根 K 线连续蓄势（15m 末 3 根同向 + 量能递增）───────────
        consec_bonus = 0.0
        consec_det   = ""
        if bool(self.config.get("candle_consec_enabled", True)) and len(m15) >= 3:
            try:
                m15_closes = m15["c"].tolist()
                m15_opens  = m15["o"].tolist() if "o" in m15.columns else m15["c"].tolist()
                m15_vols   = m15["vol"].tolist()
                last3_dir  = []
                for i in [-3, -2, -1]:
                    o = float(m15_opens[i]); c = float(m15_closes[i])
                    last3_dir.append("up" if c > o else "down" if c < o else "flat")
                vol_increasing = float(m15_vols[-1]) >= float(m15_vols[-2]) >= float(m15_vols[-3]) * 0.95
                want = "up" if d1_dir == "bull" else "down"
                same_dir_count = sum(1 for d in last3_dir if d == want)
                if same_dir_count >= 2 and vol_increasing:
                    max_b = float(self.config.get("candle_consec_bonus", 6.0))
                    consec_bonus = max_b * (same_dir_count / 3.0)
                    score = _clamp(score + consec_bonus, 0, 100)
                    consec_det = f"末3根{same_dir_count}/3{want} 量能递增 +{consec_bonus:.1f}"
            except Exception:
                pass

        # ── P1-2: swing 摆动点 + Fibonacci 回踩位验证（基于 H1）───────────
        fib_bonus = 0.0
        fib_det = ""
        swing_info: Dict[str, Any] = {}
        if bool(self.config.get("swing_detection_enabled", True)) and len(h1) >= 10:
            try:
                sw_confirm = int(self.config.get("swing_confirm_bars", 2))
                sw_look    = int(self.config.get("swing_lookback", 30))
                seg_n      = min(sw_look, len(h1))
                highs_seg  = h1["h"].iloc[-seg_n:].tolist()
                lows_seg   = h1["l"].iloc[-seg_n:].tolist()
                cur_close  = float(h1["c"].iloc[-1])
                if d1_dir == "bull":
                    sh = _find_swing_high(highs_seg, sw_confirm)
                    sl = _find_swing_low(lows_seg, sw_confirm)
                    if sh and sl and sh[1] > sl[1]:
                        swing_high = sh[1]; swing_low = sl[1]
                        fib_range = swing_high - swing_low
                        if fib_range > 0 and bool(self.config.get("fib_validate_enabled", True)):
                            tol  = float(self.config.get("fib_tolerance_pct", 2.0))
                            maxb = float(self.config.get("fib_max_bonus", 12.0))
                            for lvl_name, ratio, weight in [("61.8%", 0.618, 1.0), ("50.0%", 0.500, 0.83), ("38.2%", 0.382, 0.67)]:
                                fib_price = swing_high - fib_range * ratio
                                if fib_price > 0:
                                    dev = abs(cur_close - fib_price) / fib_price * 100
                                    if dev <= tol:
                                        fib_bonus = max(fib_bonus, maxb * weight)
                                        fib_det = f"Fib{lvl_name}命中(dev{dev:.2f}%) +{fib_bonus:.1f}"
                        swing_info = {"swing_high": swing_high, "swing_low": swing_low}
                else:
                    sh = _find_swing_high(highs_seg, sw_confirm)
                    sl = _find_swing_low(lows_seg, sw_confirm)
                    if sh and sl and sh[1] > sl[1]:
                        swing_high = sh[1]; swing_low = sl[1]
                        fib_range = swing_high - swing_low
                        if fib_range > 0 and bool(self.config.get("fib_validate_enabled", True)):
                            tol  = float(self.config.get("fib_tolerance_pct", 2.0))
                            maxb = float(self.config.get("fib_max_bonus", 12.0))
                            for lvl_name, ratio, weight in [("61.8%", 0.618, 1.0), ("50.0%", 0.500, 0.83), ("38.2%", 0.382, 0.67)]:
                                fib_price = swing_low + fib_range * ratio
                                if fib_price > 0:
                                    dev = abs(cur_close - fib_price) / fib_price * 100
                                    if dev <= tol:
                                        fib_bonus = max(fib_bonus, maxb * weight)
                                        fib_det = f"Fib{lvl_name}命中(dev{dev:.2f}%) +{fib_bonus:.1f}"
                        swing_info = {"swing_high": swing_high, "swing_low": swing_low}
                if fib_bonus > 0:
                    score = _clamp(score + fib_bonus, 0, 100)
            except Exception:
                pass

        # ── P2-3: VWAP 机构成本支撑验证（基于 H1）─────────────────────────
        vwap_bonus = 0.0
        vwap_det = ""
        if bool(self.config.get("vwap_validate_enabled", True)) and len(h1) >= 10:
            try:
                vwap_lb   = int(self.config.get("vwap_lookback", 50))
                vwap_near = float(self.config.get("vwap_near_pct", 1.5))
                vwap_max  = float(self.config.get("vwap_max_dev_pct", 4.0))
                vwap_maxb = float(self.config.get("vwap_max_bonus", 8.0))
                v_closes = h1["c"].tolist()
                v_highs  = h1["h"].tolist()
                v_lows   = h1["l"].tolist()
                v_vols   = h1["vol"].tolist()
                vwap_val = _calc_vwap(v_closes, v_highs, v_lows, v_vols, vwap_lb)
                cur_close = float(v_closes[-1])
                vwap_dev = abs(cur_close - vwap_val) / max(vwap_val, 1e-9) * 100
                if vwap_dev <= vwap_near:
                    vwap_bonus = vwap_maxb
                elif vwap_dev <= vwap_max:
                    vwap_bonus = vwap_maxb * (1.0 - (vwap_dev - vwap_near) / max(vwap_max - vwap_near, 1e-9))
                if vwap_bonus > 0:
                    score = _clamp(score + vwap_bonus, 0, 100)
                    vwap_det = f"VWAP偏离{vwap_dev:.2f}% +{vwap_bonus:.1f}"
            except Exception:
                pass

        # ── P2-5: Volume Profile POC 验证（基于 H1）─────────────────────────
        vp_bonus = 0.0
        vp_det = ""
        if bool(self.config.get("vp_poc_validate_enabled", True)) and len(h1) >= 10:
            try:
                vp_lb  = int(self.config.get("vp_poc_lookback", 50))
                vp_bn  = int(self.config.get("vp_poc_bins", 20))
                vp_tol = float(self.config.get("vp_poc_tolerance_pct", 2.0))
                vp_max = float(self.config.get("vp_poc_max_bonus", 8.0))
                p_closes = h1["c"].tolist()
                p_highs  = h1["h"].tolist()
                p_lows   = h1["l"].tolist()
                p_vols   = h1["vol"].tolist()
                vp_poc_val = _calc_vp_poc(p_closes, p_highs, p_lows, p_vols, vp_lb, vp_bn)
                if vp_poc_val > 0:
                    cur_close = float(p_closes[-1])
                    vp_dev = abs(cur_close - vp_poc_val) / vp_poc_val * 100
                    if vp_dev <= vp_tol:
                        vp_bonus = vp_max
                    elif vp_dev <= vp_tol * 2.5:
                        vp_bonus = vp_max * (1.0 - (vp_dev - vp_tol) / max(vp_tol * 1.5, 1e-9))
                    if vp_bonus > 0:
                        score = _clamp(score + vp_bonus, 0, 100)
                        vp_det = f"VP_POC偏离{vp_dev:.2f}% +{vp_bonus:.1f}"
            except Exception:
                pass

        # ── P4-5: z-score 趋势加速度评估（基于 D1）─────────────────────────
        zscore_adj = 0.0
        z_label = ""
        if bool(self.config.get("zscore_health_enabled", True)) and len(big_df) >= 22:
            try:
                z_period   = int(self.config.get("boll_period", 20))
                d1_ema_z   = float(big_df["c"].ewm(span=z_period, adjust=False).mean().iloc[-1])
                d1_atr_v   = float(_atr(big_df, int(self.config.get("zscore_atr_period", 14))))
                d1_close   = float(big_df["c"].iloc[-1])
                if d1_atr_v > 0 and d1_ema_z > 0:
                    z_signed = ((d1_close - d1_ema_z) / d1_atr_v) if d1_dir == "bull" \
                               else ((d1_ema_z - d1_close) / d1_atr_v)
                    z_opt   = float(self.config.get("zscore_optimal_max", 0.5))
                    z_emrg  = float(self.config.get("zscore_emerging_max", 1.5))
                    z_extd  = float(self.config.get("zscore_extended_max", 2.5))
                    z_max_b = float(self.config.get("zscore_max_bonus", 8.0))
                    z_max_p = float(self.config.get("zscore_max_penalty", 12.0))
                    if z_signed <= z_opt:
                        zscore_adj = z_max_b
                        z_label = f"早期(z={z_signed:.2f})"
                    elif z_signed <= z_emrg:
                        ratio = (z_signed - z_opt) / max(z_emrg - z_opt, 1e-9)
                        zscore_adj = z_max_b * (1.0 - ratio)
                        z_label = f"萌芽(z={z_signed:.2f})"
                    elif z_signed <= z_extd:
                        ratio = (z_signed - z_emrg) / max(z_extd - z_emrg, 1e-9)
                        zscore_adj = -z_max_p * ratio
                        z_label = f"透支(z={z_signed:.2f})"
                    else:
                        zscore_adj = -z_max_p
                        z_label = f"重度透支(z={z_signed:.2f})"
                    score = _clamp(score + zscore_adj, 0, 100)
            except Exception:
                pass

        score, details, risk_cfg = self._apply_risk(
            score, inst_id, ed, d1_dir, atr_pct, lp, d1_det, h1_det, m15_det,
            swing_info=swing_info, h1=h1, market_state=market_state)

        # Step 5: RSI 背离过滤（在 _apply_risk 之后，直接修改 score）
        if bool(self.config.get("rsi_diverge_enabled", True)) and len(big_df) >= 20:
            div_penalty, div_det = self._check_rsi_divergence(big_df, d1_dir)
            if div_penalty > 0:
                score = max(0.0, score - div_penalty)
                details["RSI背离"] = div_det

        # H4 共振细节写入
        if h4_det:
            details["H4共振"] = h4_det
        if candle_det:
            details["K线质量"] = candle_det
        # P2-6: 多根连续蓄势
        if consec_det:
            details["连续蓄势"] = consec_det
        # P1-2: Fib 命中
        if fib_det:
            details["Fib命中"] = fib_det
        # P2-3: VWAP
        if vwap_det:
            details["VWAP"] = vwap_det
        # P2-5: VP_POC
        if vp_det:
            details["VP_POC"] = vp_det
        # P4-5: z-score
        if z_label:
            details["z-score"] = z_label + (f" {zscore_adj:+.1f}分" if abs(zscore_adj) > 0.01 else "")

        # ── P2-4: 信号持久性追踪（连续 N 次扫描出现 → 加分）─────────────
        persistence_bonus = 0.0
        if bool(self.config.get("persistence_enabled", True)):
            self._scan_counter += 1
            dir_str = "BUY" if d1_dir == "bull" else "SELL"
            key = f"{inst_id}:{dir_str}"
            prev = self._signal_history.get(key)
            if prev and self._scan_counter - prev[0] <= 2:   # 连续 2 次扫描内
                count = self._persist_counts.get(key, 0) + 1
                self._persist_counts[key] = count
                min_cnt = int(self.config.get("persistence_min_count", 2))
                if count >= min_cnt:
                    persistence_bonus = float(self.config.get("persistence_bonus", 4.0))
                    details["信号持久性"] = f"✓ 连续{count}次出现 +{persistence_bonus:.0f}"
            else:
                self._persist_counts[key] = 1
            self._signal_history[key] = (self._scan_counter, score)
            score = _clamp(score + persistence_bonus, 0, 100)

        if score < self.config["min_score"]: return fail(f"评分不足({score:.1f})")

        dir_cn = "多头" if d1_dir == "bull" else "空头"
        # 信号摘要（v1.6 增强）
        sig_lines = [
            f"{dir_cn} {score:.1f}分 [{market_state}]",
            f"D1布林{'上' if d1_dir=='bull' else '下'}轨→H4共振→1H回踩→15m起爆",
        ]
        if fib_det:        sig_lines.append(f"📐 {fib_det}")
        if vwap_det:       sig_lines.append(f"🏦 {vwap_det}")
        if vp_det:         sig_lines.append(f"📊 {vp_det}")
        if z_label:        sig_lines.append(f"⚡ z-score {z_label}")
        if consec_det:     sig_lines.append(f"🕯 {consec_det}")
        if persistence_bonus > 0:
            sig_lines.append(f"🔄 信号稳定+{persistence_bonus:.0f}")
        # P3-1 动态止损止盈
        sig_lines.append(
            f"🛑 动态止损-{risk_cfg['dynamic_stop_loss_pct']:.2f}% / "
            f"🎯 止盈+{risk_cfg['dynamic_take_profit_pct']:.2f}% "
            f"({risk_cfg['dynamic_tp_source']}, RR={risk_cfg['dynamic_rr_ratio']:.2f})"
        )
        sig_lines.append(f"💰 建议仓位 {risk_cfg['position_pct']:.2f}%")

        return {
            "symbol": inst_id, "passed": True, "score": round(score, 2),
            "opportunity_score": round(score, 2),
            "direction": "BUY" if d1_dir == "bull" else "SELL",
            "category": "小月期货布林趋势转折",
            "signals": sig_lines,
            "last_price": lp,
            "volume_24h": float(getattr(symbol, "volume_24h", 0) or 0),
            "details": details,
            # P3-4: 全部 risk_cfg 字段透出（含 dynamic_*、trail_*、position_pct）
            "stop_loss_pct": risk_cfg["stop_loss_pct"],
            "take_profit_pct": risk_cfg["take_profit_pct"],
            "dynamic_stop_loss_pct": risk_cfg["dynamic_stop_loss_pct"],
            "dynamic_take_profit_pct": risk_cfg["dynamic_take_profit_pct"],
            "dynamic_rr_ratio": risk_cfg["dynamic_rr_ratio"],
            "trail_activate_pct": risk_cfg["trail_activate_pct"],
            "trail_distance_pct": risk_cfg["trail_distance_pct"],
            "position_pct": risk_cfg["position_pct"],
            "market_state": market_state,
            "ranking_factors": {
                "trend": d1_sc, "trigger": m15_sc, "volume": m15_sc * 0.8,
                "location": h1_sc, "freshness": 50,
                "risk": _clamp(100 - atr_pct * 5, 0, 100),
            },
        }

    # ── 改进一：H4 多周期共振 ────────────────────────────────────────────────
    def _step1b_h4_resonance(self, h4: pd.DataFrame, direction: str) -> Tuple[float, str]:
        """
        检测 H4 周期与 D1 方向是否共振。
        返回 (score_delta, detail_str)：正值=加分，负值=惩罚。
        v1.6 改进：优先使用 v2 多维度评分（方向 40%+斜率强度 30%+MACD 柱强度 30%）。
        """
        if not bool(self.config.get("h4_resonance_enabled", True)):
            return 0.0, ""
        # P2-1: H4 共振 v2 多维度评分（替代二值方向判定）
        if bool(self.config.get("h4_resonance_v2_enabled", True)):
            return self._h4_resonance_v2(h4, direction)
        bonus   = float(self.config.get("h4_resonance_bonus",  12.0))
        penalty = float(self.config.get("h4_diverge_penalty",  15.0))
        if h4 is None or len(h4) < 30:
            return 0.0, "H4数据不足，跳过共振"
        try:
            period = int(self.config["boll_period"])
            std_m  = float(self.config["boll_std_mult"])
            b4 = _bollinger(h4, period, std_m)
            last4 = b4.iloc[-1]
            mid4  = float(last4["boll_mid"])
            c4    = float(last4["c"])
            # H4 MACD
            mf = int(self.config["m15_macd_fast"])   # 复用 MACD 参数
            ms_p = int(self.config["m15_macd_slow"])
            msig = int(self.config["m15_macd_signal"])
            ef4 = h4["c"].ewm(span=mf,   adjust=False).mean()
            es4 = h4["c"].ewm(span=ms_p, adjust=False).mean()
            ml4 = ef4 - es4
            msl4 = ml4.ewm(span=msig, adjust=False).mean()
            hist4 = float((ml4 - msl4).iloc[-1])
            mid_slope4 = _slope_angle(b4["boll_mid"], int(self.config["slope_lookback"]))

            # 判断 H4 方向
            h4_bull = c4 > mid4 and mid_slope4 > 0 and hist4 > 0
            h4_bear = c4 < mid4 and mid_slope4 < 0 and hist4 < 0

            if direction == "bull":
                if h4_bull:
                    return bonus, f"H4共振多头 斜率{mid_slope4:.1f}° MACD柱正"
                elif h4_bear:
                    return -penalty, f"⚠H4反向偏空 斜率{mid_slope4:.1f}°"
                else:
                    return 0.0, f"H4中性 斜率{mid_slope4:.1f}°"
            else:  # bear
                if h4_bear:
                    return bonus, f"H4共振空头 斜率{mid_slope4:.1f}° MACD柱负"
                elif h4_bull:
                    return -penalty, f"⚠H4反向偏多 斜率{mid_slope4:.1f}°"
                else:
                    return 0.0, f"H4中性 斜率{mid_slope4:.1f}°"
        except Exception:
            return 0.0, "H4共振计算异常"

    # ── P2-1: H4 共振 v2 多维度评分 ────────────────────────────────────────
    def _h4_resonance_v2(self, h4: pd.DataFrame, direction: str) -> Tuple[float, str]:
        """
        H4 共振 v2：方向 40% + 斜率强度 30% + MACD 柱强度 30%，归一化到 [0,1]。
        v2 score < hard_floor (0.20) 且开启 hard_filter → 返回硬拒标记（外层判断）。
        加分映射：以 0.5 为零点的线性函数 [-max, +max]。
        """
        max_b = float(self.config.get("h4_resonance_v2_max", 12.0))
        if h4 is None or len(h4) < 30:
            return 0.0, "H4数据不足，跳过共振v2"
        try:
            period = int(self.config["boll_period"])
            std_m  = float(self.config["boll_std_mult"])
            b4 = _bollinger(h4, period, std_m)
            last4 = b4.iloc[-1]
            mid4  = float(last4["boll_mid"])
            c4    = float(last4["c"])
            slope_deg = _slope_angle(b4["boll_mid"], int(self.config["slope_lookback"]))
            # MACD
            mf = int(self.config["m15_macd_fast"])
            ms = int(self.config["m15_macd_slow"])
            msig = int(self.config["m15_macd_signal"])
            ef4 = h4["c"].ewm(span=mf, adjust=False).mean()
            es4 = h4["c"].ewm(span=ms, adjust=False).mean()
            ml4 = ef4 - es4
            msl4 = ml4.ewm(span=msig, adjust=False).mean()
            hist4 = float((ml4 - msl4).iloc[-1])

            # 1) 方向分（0/1）
            if direction == "bull":
                dir_match = c4 > mid4
            else:
                dir_match = c4 < mid4
            dir_sub = 1.0 if dir_match else 0.0

            # 2) 斜率强度分（0~1，归一化到 0.6° = 满分）
            slope_aligned = (direction == "bull" and slope_deg > 0) or \
                            (direction == "bear" and slope_deg < 0)
            slope_intensity = _clamp(abs(slope_deg) / 0.6, 0.0, 1.0)
            slope_sub = (0.5 if slope_aligned else 0.0) + 0.5 * slope_intensity
            slope_sub = _clamp(slope_sub, 0.0, 1.0)

            # 3) MACD 柱方向 + 强度（0~1，归一化到价格 0.3% = 满分强度）
            hist_aligned = (direction == "bull" and hist4 > 0) or \
                           (direction == "bear" and hist4 < 0)
            hist_intensity = _clamp(abs(hist4) / max(c4 * 0.003, 1e-9), 0.0, 1.0)
            macd_sub = (0.5 if hist_aligned else 0.0) + 0.5 * hist_intensity
            macd_sub = _clamp(macd_sub, 0.0, 1.0)

            # 综合：方向 40% + 斜率 30% + MACD 30%
            v2_score = dir_sub * 0.40 + slope_sub * 0.30 + macd_sub * 0.30
            self._last_h4_v2_score = v2_score   # 用于 P3-1 等下游使用

            # 硬拒（仅当 hard_filter_enabled 同时 v2 < hard_floor）
            hard_floor = float(self.config.get("h4_resonance_v2_hard_floor", 0.20))
            if (bool(self.config.get("h4_hard_filter_enabled", True))
                    and v2_score < hard_floor):
                # 通过返回极大负值让上层硬拒
                return -999.0, f"H4共振v2={v2_score:.2f}<{hard_floor:.2f} 严重逆向"

            # 加分映射：以 0.5 为零点的线性函数 [-max, +max]
            adj = round((v2_score - 0.5) * 2.0 * max_b, 2)
            return adj, f"H4共振v2={v2_score:.2f}(方向{dir_sub:.0f}/斜率{slope_sub:.2f}/MACD{macd_sub:.2f})"
        except Exception:
            return 0.0, "H4共振v2计算异常"

    # ── 改进三：K 线形态质量评分 ─────────────────────────────────────────────
    def _score_candle_quality(self, df: pd.DataFrame, direction: str) -> Tuple[float, str]:
        """
        评估 15m 最新一根 K 线的形态质量（0~50 分）。
        同时检测吞没形态（额外加分）。
        """
        if len(df) < 2:
            return 0.0, ""
        try:
            last = df.iloc[-1]; prev = df.iloc[-2]
            o  = float(last.get("o",  last.get("open",  0)))
            h  = float(last.get("h",  last.get("high",  0)))
            l  = float(last.get("l",  last.get("low",   0)))
            c  = float(last.get("c",  last.get("close", 0)))
            rng = max(h - l, 1e-9)
            body = abs(c - o)
            body_ratio  = body / rng
            upper_wick  = (h - max(c, o)) / rng
            lower_wick  = (min(c, o) - l) / rng

            if direction == "bull":
                # 多头：大实体阳线 + 下影（买盘）+ 无长上影（无抛压）
                quality = body_ratio * 40 + lower_wick * 20 - upper_wick * 15
                # 锤子线：下影 >= 2×实体，上影短，今根必须收阳
                hammer = lower_wick >= 2 * body_ratio and upper_wick < 0.1 and c >= o
                # P0-X1 修复：吞没阳线必须满足
                #   ① 今根阳线 (c > o)
                #   ② 前根阴线 (pc < po)
                #   ③ 今根实体完全包住前根实体 (c > po AND o < pc)
                # 原代码缺 ① 会把十字星/弱反弹误判为吞没
                po = float(prev.get("o", prev.get("open", 0)))
                pc = float(prev.get("c", prev.get("close", 0)))
                engulf = (c > o) and (pc < po) and (c > po) and (o < pc)
                tag = ("锤子线✓" if hammer else "") + ("吞没✓" if engulf else "")
                if hammer:  quality += 15
                if engulf:  quality += 20
            else:  # bear
                quality = body_ratio * 40 + upper_wick * 20 - lower_wick * 15
                # 射击之星：上影 >= 2×实体，下影短，今根必须收阴
                shooting = upper_wick >= 2 * body_ratio and lower_wick < 0.1 and c <= o
                po = float(prev.get("o", prev.get("open", 0)))
                pc = float(prev.get("c", prev.get("close", 0)))
                # P0-X1 修复：吞没阴线必须 ① 今根阴线 ② 前根阳线 ③ 实体包住
                engulf = (c < o) and (pc > po) and (c < po) and (o > pc)
                tag = ("射击之星✓" if shooting else "") + ("吞没✓" if engulf else "")
                if shooting: quality += 15
                if engulf:   quality += 20

            quality = _clamp(quality, 0, 50)
            label = tag if tag else f"实体{body_ratio:.0%}"
            return quality, f"形态{label} 质量{quality:.0f}/50"
        except Exception:
            return 0.0, ""

    # ── 改进四：RSI 动量背离检测（P0-X2: O(n²) → O(n)）──────────────────────
    def _check_rsi_divergence(self, df: pd.DataFrame, direction: str) -> Tuple[float, str]:
        """
        检测价格与 RSI 的顶/底背离。
        返回 (penalty, detail)。无背离时返回 (0, "")。
        P0-X2 修复：原实现每次循环都重算整段 RSI（O(n²)），20 根窗口要算 20 次 RSI。
                    改为一次性 EMA 算完，复杂度降到 O(n)。
        """
        penalty = float(self.config.get("rsi_diverge_penalty", 20.0))
        try:
            if len(df) < 20:
                return 0.0, ""
            rsi_period = int(self.config.get("h1_rsi_period", 14))
            closes = df["c"]
            # P0-X2: 一次性 Wilder RSI 序列（用 ewm alpha=1/period 等价于 Wilder 平滑）
            deltas  = closes.diff()
            gains   = deltas.where(deltas > 0, 0.0)
            losses  = (-deltas).where(deltas < 0, 0.0)
            avg_g   = gains.ewm(alpha=1.0 / rsi_period, adjust=False).mean()
            avg_l   = losses.ewm(alpha=1.0 / rsi_period, adjust=False).mean()
            rs      = avg_g / avg_l.replace(0, 1e-9)
            rsi_full = (100.0 - 100.0 / (1.0 + rs)).fillna(50.0)
            window  = min(20, len(closes))
            rsi_all = rsi_full.iloc[-window:].tolist()
            prices  = closes.iloc[-window:].tolist()

            # 比较最近 5 根与前 10 根的极值
            recent_price_max = max(prices[-5:])
            prior_price_max  = max(prices[-15:-5])
            recent_rsi_max   = max(rsi_all[-5:])
            prior_rsi_max    = max(rsi_all[-15:-5])
            recent_price_min = min(prices[-5:])
            prior_price_min  = min(prices[-15:-5])
            recent_rsi_min   = min(rsi_all[-5:])
            prior_rsi_min    = min(rsi_all[-15:-5])

            if direction == "bull":
                # 顶背离：价格新高但 RSI 未新高（多头警示反转）
                if recent_price_max > prior_price_max * 1.005 and recent_rsi_max < prior_rsi_max - 3:
                    return penalty, f"顶背离⚠ 价格↑{(recent_price_max/prior_price_max-1)*100:.1f}% RSI↓{prior_rsi_max-recent_rsi_max:.1f}"
            else:  # bear
                # 底背离：价格新低但 RSI 未新低（空头警示反转）
                if recent_price_min < prior_price_min * 0.995 and recent_rsi_min > prior_rsi_min + 3:
                    return penalty, f"底背离⚠ 价格↓ RSI↑{recent_rsi_min-prior_rsi_min:.1f}"
            return 0.0, ""
        except Exception:
            return 0.0, ""

    def _step1_daily_trend(self, df: pd.DataFrame, label: str = "D1") -> Tuple[bool, str, float, str, float]:
        period = int(self.config["boll_period"]); std_m = float(self.config["boll_std_mult"])
        sl_lb = int(self.config["slope_lookback"]); min_ang = float(self.config["slope_min_angle"])
        b = _bollinger(df, period, std_m)
        last = b.iloc[-1]; mid_slope = _slope_angle(b["boll_mid"], sl_lb)
        c, mid = float(last["c"]), float(last["boll_mid"])

        # ATR 自适应斜率阈值
        atr_pct = float(_atr(df, 14) / max(c, 1e-9) * 100) if c > 0 else 2.0
        eff_min_angle = min_ang
        if bool(self.config.get("slope_atr_scale", False)):
            eff_min_angle = max(min_ang, atr_pct * 0.03)  # 高 ATR 时提高斜率门槛

        if c > mid and mid_slope > eff_min_angle: direction, cn = "bull", "多头"
        elif c < mid and mid_slope < -eff_min_angle: direction, cn = "bear", "空头"
        else: return False, "neutral", 0, f"无趋势(收盘{c:.4g}vs中轨{mid:.4g} 斜率{mid_slope:.1f}°<{eff_min_angle:.1f}°)", atr_pct

        # ADX 趋势确认（使用共享库，内置数据量保护）
        try:
            adx = _adx(df, int(self.config["adx_period"]))
        except Exception: adx = 0.0
        adx_min = float(self.config["adx_min_trend"])
        if adx < adx_min and label != "4H":
            return False, "neutral", 0, f"ADX={adx:.1f}<{adx_min}趋势弱", atr_pct

        # BBW 收缩检测
        bbw_series = (b["boll_up"] - b["boll_down"]) / b["boll_mid"] * 100
        cur_bbw = float(bbw_series.iloc[-1])
        avg_bbw = float(bbw_series.iloc[-sl_lb:].mean())
        squeeze = cur_bbw < avg_bbw * 0.92  # 带宽低于近期均值=收缩中

        # 改进4: 中轨斜率加速检测（近半段斜率比全段更陡 = 趋势在增强）
        slope_accel = False
        accel_bonus = float(self.config.get("d1_slope_accel_bonus", 10.0))
        half_lb = max(2, sl_lb // 2)
        if len(b) >= sl_lb + 1:
            slope_half = _slope_angle(b["boll_mid"], half_lb)
            if direction == "bull":
                slope_accel = slope_half > mid_slope * 1.10 and slope_half > eff_min_angle
            else:
                slope_accel = slope_half < mid_slope * 1.10 and slope_half < -eff_min_angle

        # 评分
        score = _clamp(abs(mid_slope) / max(eff_min_angle * 4, 0.5), 0, 1) * 50
        # P0-X3 修复：原 range(min(sl_lb, len(b)-1))，当 sl_lb=8 且 len(b)=8 时 i=0 → 索引 -8 越界
        # 改为 sl_lb-1 上限确保 -(sl_lb-i) 永远在 [-(sl_lb-1), -1] 内
        recent_check_n = min(sl_lb - 1, len(b) - 1)
        recent_ok = all((float(b["c"].iloc[-(sl_lb-i)]) > float(b["boll_mid"].iloc[-(sl_lb-i)]))
                        if direction == "bull" else
                        (float(b["c"].iloc[-(sl_lb-i)]) < float(b["boll_mid"].iloc[-(sl_lb-i)]))
                        for i in range(recent_check_n)) if recent_check_n > 0 else False
        if recent_ok: score += 15
        if squeeze: score += self.config["bbw_squeeze_bonus"]
        if slope_accel: score += accel_bonus
        detail = (f"{label}{cn} 斜率{mid_slope:.1f}°"
                  f"{' 加速↑' if slope_accel else ''}"
                  f" ADX={adx:.1f}"
                  f"{' 收缩' if squeeze else ''}"
                  f" {'持续' if recent_ok else ''}")
        # P1-5: 把 ADX/斜率/ATR 缓存到 self 用于市场状态自适应
        self._last_d1_metrics = {"adx": float(adx), "slope_deg": float(mid_slope), "atr_pct": float(atr_pct)}
        return True, direction, min(100, score), detail, atr_pct

    def _step2_h1_pullback(self, df: pd.DataFrame, direction: str) -> Tuple[bool, float, str, float]:
        period = int(self.config["boll_period"]); std_m = float(self.config["boll_std_mult"])
        prox_a = float(self.config["mid_band_proximity_atr"]); prox_f = float(self.config["mid_band_proximity_floor"])
        b = _bollinger(df, period, std_m); last = b.iloc[-1]
        mid = float(last["boll_mid"]); close = float(last["c"])
        up = float(last["boll_up"]); dn = float(last["boll_down"])
        if mid <= 0: return False, 0, "中轨无效", 2.0
        atr_pct = float(_atr(df, 14) / close * 100) if close > 0 else 2.0
        dist_pct = abs(close - mid) / mid * 100
        max_d = max(prox_a * atr_pct, prox_f)
        # 价格必须在带内（未穿越布林带）
        if direction == "bull" and close >= up:
            return False, 0, f"价格已穿越布林上轨，追高风险大(close={close:.4g}>up={up:.4g})", atr_pct
        if direction == "bear" and close <= dn:
            return False, 0, f"价格已穿越布林下轨，追空风险大(close={close:.4g}<dn={dn:.4g})", atr_pct
        # 距中轨超过允许范围才拒绝
        if dist_pct >= max_d: return False, 0, f"距中轨{dist_pct:.1f}%>{max_d:.1f}%", atr_pct
        rsi = _rsi_wilder(df["c"], int(self.config["h1_rsi_period"]))
        # 回踩中轨时RSI方向：多头RSI应低于超买区(偏弱)，空头RSI应高于超卖区(偏强)
        rsi_ok = (rsi < self.config["h1_rsi_oversold_long"]) if direction == "bull" else (rsi > self.config["h1_rsi_overbought_short"])
        rsi_gate = bool(self.config.get("h1_rsi_gate_enabled", False))
        # 改进3: 硬门槛模式=直接拒绝；软门槛模式=降分放行
        if rsi_gate and not rsi_ok:
            return False, 0, f"RSI={rsi:.1f} 不在回踩区({'需<{}'.format(self.config['h1_rsi_oversold_long']) if direction=='bull' else '需>{}'.format(self.config['h1_rsi_overbought_short'])})", atr_pct
        rsi_soft_penalty = float(self.config.get("h1_rsi_soft_penalty", 12.0)) if not rsi_ok else 0.0

        # 评分：距中轨越近分越高，带内位置分，RSI加分
        score = _clamp(1 - dist_pct / max(max_d, 1e-9), 0, 1) * 50
        band_width = max(up - dn, 1e-9)
        band_pos = (close - dn) / band_width  # 0=下轨 1=上轨
        # 多头：带内中下段分数高（回踩到中轨0.3~0.6为优）；空头相反
        if direction == "bull":
            band_score = _clamp(1 - abs(band_pos - 0.45) / 0.45, 0, 1) * 20
        else:
            band_score = _clamp(1 - abs(band_pos - 0.55) / 0.45, 0, 1) * 20
        score += band_score
        if rsi_ok:
            score += 20
        else:
            score -= rsi_soft_penalty  # RSI不满足：软降分而非拒绝
        if (direction == "bull" and close > mid) or (direction == "bear" and close < mid): score += 10
        rsi_tag = "✓" if rsi_ok else f"⚠{rsi:.0f}"
        return True, min(100, max(0, score)), f"距中轨{dist_pct:.1f}%(max{max_d:.1f}%) RSI={rsi:.1f}{rsi_tag} 带位{band_pos:.2f}", atr_pct

    def _step3_m15_entry(self, df: pd.DataFrame, direction: str) -> Tuple[bool, float, str]:
        period = int(self.config["boll_period"]); std_m = float(self.config["boll_std_mult"])
        mf = int(self.config["m15_macd_fast"]); ms = int(self.config["m15_macd_slow"])
        msig = int(self.config["m15_macd_signal"])
        b = _bollinger(df, period, std_m); last = b.iloc[-1]
        mid_last, close_last = float(last["boll_mid"]), float(last["c"])
        price_sig = close_last > mid_last if direction == "bull" else close_last < mid_last

        # ── MACD 计算 ───────────────────────────────────────────────────────
        ef = df["c"].ewm(span=mf, adjust=False).mean()
        es = df["c"].ewm(span=ms, adjust=False).mean()
        ml = ef - es
        msl = ml.ewm(span=msig, adjust=False).mean()
        mc = float(ml.iloc[-1]); mp = float(ml.iloc[-2]) if len(ml) >= 2 else mc
        sc = float(msl.iloc[-1]); sp = float(msl.iloc[-2]) if len(msl) >= 2 else sc
        hist_now = float((ml - msl).iloc[-1])

        # 当根交叉
        macd_c_now = (mc > sc and mp <= sp) if direction == "bull" else (mc < sc and mp >= sp)

        # 回看N根内是否有交叉（改进5：lookback默认已扩大为10）
        lookback = max(1, int(self.config.get("m15_macd_cross_lookback", 10) or 10))
        macd_c_recent = False
        if len(ml) >= 2:
            max_lb = min(lookback, len(ml) - 2)
            for offset in range(1, max_lb + 1):
                ci = len(ml) - 1 - offset; pi = ci - 1
                cm = float(ml.iloc[ci]); cs_ = float(msl.iloc[ci])
                pm = float(ml.iloc[pi]); ps_ = float(msl.iloc[pi])
                if direction == "bull":
                    if cm > cs_ and pm <= ps_: macd_c_recent = True; break
                else:
                    if cm < cs_ and pm >= ps_: macd_c_recent = True; break

        hist_confirm = hist_now > 0 if direction == "bull" else hist_now < 0
        macd_c = macd_c_now or (macd_c_recent and hist_confirm)

        # ── 改进1: MACD 收敛预判 ─────────────────────────────────────────────
        # 柱体向零轴方向连续收缩（负柱体绝对值连续缩小/正柱体绝对值连续缩小）
        # → 预示即将发生金叉/死叉，作为弱触发信号
        converging = False
        conv_lb = max(3, int(self.config.get("m15_macd_converge_lookback", 4)))
        if len(ml) >= conv_lb + 1:
            hist_series = ml - msl
            hv = [float(hist_series.iloc[-(conv_lb - i)]) for i in range(conv_lb)]
            if direction == "bull":
                # 负柱体连续向0靠近（值从大负到小负）
                converging = all(hv[j] < 0 for j in range(conv_lb)) and \
                             all(hv[j + 1] > hv[j] for j in range(conv_lb - 1))
            else:
                # 正柱体连续向0靠近（值从大正到小正）
                converging = all(hv[j] > 0 for j in range(conv_lb)) and \
                             all(hv[j + 1] < hv[j] for j in range(conv_lb - 1))

        # ── 改进2: EMA5/10 快叉作为备选触发 ────────────────────────────────
        ema_fast_cross = False
        if bool(self.config.get("m15_ema_fast_cross", True)) and len(df) >= 11:
            ema5  = df["c"].ewm(span=5,  adjust=False).mean()
            ema10 = df["c"].ewm(span=10, adjust=False).mean()
            if direction == "bull":
                ema_fast_cross = (float(ema5.iloc[-1]) > float(ema10.iloc[-1]) and
                                  float(ema5.iloc[-2]) <= float(ema10.iloc[-2]))
            else:
                ema_fast_cross = (float(ema5.iloc[-1]) < float(ema10.iloc[-1]) and
                                  float(ema5.iloc[-2]) >= float(ema10.iloc[-2]))

        # ── MACD 趋势持续（已在正确侧且价格站稳）────────────────────────────
        macd_trend_ok = hist_confirm and price_sig

        # ── 改进G：15m MACD 柱体连续强化检测 ────────────────────────────────
        # 要求最近2根柱体绝对值均在扩大（排除单根反弹叉后萎缩的假叉）
        hist_consec_ok = True   # 默认通过，仅在启用且数据充足时校验
        if bool(self.config.get("m15_hist_consec_enabled", True)) and len(ml) >= 3:
            hist_s = ml - msl
            h_now  = float(hist_s.iloc[-1])
            h_prev = float(hist_s.iloc[-2])
            h_prev2= float(hist_s.iloc[-3])
            if direction == "bull":
                # 多头：hist 须连续≥0 且最近两根绝对值递增
                hist_consec_ok = (h_now >= 0 and h_prev >= 0 and abs(h_now) >= abs(h_prev))
            else:
                # 空头：hist 须连续≤0 且最近两根绝对值递增
                hist_consec_ok = (h_now <= 0 and h_prev <= 0 and abs(h_now) >= abs(h_prev))

        # 综合触发等级（决定评分和放行条件）：
        #   level 3 = macd_c（正式交叉）           → 满分
        #   level 2 = macd_trend_ok（趋势持续）    → 中分
        #   level 1 = converging（收敛预判）        → 低分，需 price_sig
        #   level 1 = ema_fast_cross（快叉）        → 低分，需 price_sig
        trig_level = 0
        trig_note  = "无触发"
        if macd_c:
            trig_level = 3
            trig_note  = "当根交叉" if macd_c_now else "近期交叉延续"
        elif macd_trend_ok:
            trig_level = 2
            trig_note  = "趋势持续"
        elif price_sig and converging:
            trig_level = 1
            trig_note  = "MACD收敛中"
        elif price_sig and ema_fast_cross:
            trig_level = 1
            trig_note  = "EMA5/10快叉"

        # ── 量能 ────────────────────────────────────────────────────────────
        vr = float(self.config["m15_vol_surge_ratio"])
        vb = int(self.config["m15_vol_baseline"])
        cv = float(df["vol"].iloc[-1])
        bv = float(df["vol"].iloc[-vb - 1:-1].mean()) if len(df) >= vb + 2 else cv
        vol_ok = cv > bv * vr if bv > 0 else True

        req_macd  = bool(self.config.get("m15_macd_required", True))
        req_price = bool(self.config.get("m15_price_required", True))
        req_volume = bool(self.config.get("m15_volume_required", True))

        # ── 强制检查 ─────────────────────────────────────────────────────────
        if req_price and req_macd:
            if not price_sig:
                return False, 0, f"价格未站稳中轨(触发:{trig_note})"
            if trig_level == 0:
                return False, 0, f"无有效触发信号(价格{'✓' if price_sig else '✗'}/MACD未收敛)"
        elif req_price and not req_macd:
            if not price_sig: return False, 0, "价格未站稳中轨"
        elif req_macd and not req_price:
            if trig_level == 0: return False, 0, "MACD无触发信号"
        else:
            if trig_level == 0 and not price_sig: return False, 0, "无起爆信号"

        if req_volume and not vol_ok:
            return False, 0, f"15m未放量(cv={cv:.0f} < {vr:.2f}×{bv:.0f})"

        # 改进G：MACD 柱体连续强化（level 3 交叉时强制要求，level 2 时扣分但不降级）
        # P1-4: 原代码 trig_level=2 时降级到 1（22→14分，损失 8 分）。改为扣固定分但保留 level，影响更精准。
        consec_penalty_pending = 0.0
        if not hist_consec_ok:
            if trig_level == 3:
                return False, 0, "15m MACD柱体方向未连续(假叉过滤)"
            elif trig_level == 2:
                consec_penalty_pending = float(self.config.get("trig_consec_fail_penalty", 5.0))
                trig_note += " 柱体未连续-{:.0f}".format(consec_penalty_pending)

        # ── 改进二：15m BB 挤压扩张检测 ─────────────────────────────────────
        bbw_lb = int(self.config.get("m15_bbw_squeeze_lookback", 10))
        bbw_series = (b["boll_up"] - b["boll_down"]) / b["boll_mid"].replace(0, 1e-9)
        bbw_now  = float(bbw_series.iloc[-1])
        bbw_avg  = float(bbw_series.iloc[-bbw_lb:].mean()) if len(bbw_series) >= bbw_lb else bbw_now
        bb_squeeze   = bbw_now < bbw_avg * 0.80          # 带宽低于均值80% = 仍在挤压
        bb_expanding = len(bbw_series) >= 2 and bbw_now > float(bbw_series.iloc[-2])  # 本根已开始扩张

        # ── 评分 ─────────────────────────────────────────────────────────────
        score = 0
        if price_sig: score += 30
        if   trig_level == 3: score += 35          # 正式 MACD 交叉
        elif trig_level == 2: score += 22          # 趋势持续（无新叉）
        elif trig_level == 1: score += 14          # 收敛预判 / EMA快叉（保守给分）
        if vol_ok: score += 20
        # MACD 柱体强度（占中轨百分比，上限12分）
        mh = ml - msl
        hist_pct = abs(float(mh.iloc[-1])) / max(mid_last, 1e-9) * 100
        score += _clamp(hist_pct / 0.5, 0, 1) * 12
        # BB 挤压扩张奖励
        m15_bbw_bonus = float(self.config.get("m15_bbw_squeeze_bonus", 15.0))
        if bb_expanding and bb_squeeze:
            score += m15_bbw_bonus        # 挤压中刚开始扩张 = 最佳时机
        elif bb_expanding:
            score += m15_bbw_bonus * 0.5  # 已扩张中，给半分

        # 改进C：BBW 已过度扩张（行情已走太远）惩罚
        overextend_ratio = float(self.config.get("m15_bbw_overextend_ratio", 1.30))
        overextend_pen = float(self.config.get("m15_bbw_overextend_pen", 12.0))
        bb_overextended = (bb_expanding and not bb_squeeze and bbw_now > bbw_avg * overextend_ratio)
        if bb_overextended:
            score -= overextend_pen

        # P1-4: 应用柱体未连续扣分（保留 trig_level）
        if consec_penalty_pending > 0:
            score -= consec_penalty_pending

        bb_note = ("挤压扩张✓" if (bb_squeeze and bb_expanding) else
                   ("过度扩张⚠" if bb_overextended else
                    ("扩张中" if bb_expanding else ("挤压中" if bb_squeeze else ""))))
        ct = "金叉" if direction == "bull" else "死叉"
        return True, min(100, score), (
            f"价格{'✓' if price_sig else '✗'} "
            f"MACD{'✓' if trig_level>=2 else '~' if trig_level==1 else '✗'}({trig_note}){ct} "
            f"放量{'✓' if vol_ok else '✗'}"
            + (f" BB{bb_note}" if bb_note else "")
        )

    def _apply_risk(self, score, inst_id, ed, direction, atr_pct, lp, bd, hd, md,
                    swing_info: Optional[Dict[str, float]] = None,
                    h1: Optional[pd.DataFrame] = None,
                    market_state: str = "neutral"):
        d = {"大周期": bd, "1H回踩": hd, "15m起爆": md, "ATR%": f"{atr_pct:.1f}%",
             "市场状态": market_state}
        f_ = float(ed.get("funding_rate", 0) or 0); ep = self.config["funding_extreme_pct"] / 100.0
        if abs(f_) > ep and ((f_ > ep and direction == "bull") or (f_ < -ep and direction == "bear")):
            score -= self.config["funding_penalty"]; d["资金费率"] = f"⚠ 拥挤({f_*100:.2f}%)"

        # P3-7: BTC 环境阈值 ATR 自适应
        bc = float(ed.get("btc_1h_pct", ed.get("btc_context", {}).get("btc_1h_pct", 0)) or 0) if isinstance(ed, dict) else 0
        btc_atr = float(ed.get("btc_atr_pct", 0) or 0) if isinstance(ed, dict) else 0
        if bool(self.config.get("btc_env_atr_adaptive", True)) and btc_atr > 0:
            atr_mult = float(self.config.get("btc_env_atr_mult", 1.0))
            dump_thr = -btc_atr * atr_mult
            pump_thr = btc_atr * atr_mult
        else:
            dump_thr = self.config["btc_dump_threshold_pct"]
            pump_thr = float(self.config.get("btc_pump_threshold_pct", 3.0))
        if bc < dump_thr and direction == "bull":
            score -= self.config["btc_dump_penalty"]; d["BTC环境"] = f"⚠ BTC跌{bc:.1f}%(阈值{dump_thr:.1f}%)"
        if bc > pump_thr and direction == "bear":
            score -= self.config["btc_dump_penalty"]; d["BTC环境"] = f"⚠ BTC涨{bc:.1f}%(阈值{pump_thr:.1f}%)"

        # ── ATR 基础止损止盈（兼容保留） ──
        sm = self.config["stop_atr_mult"]; tm = self.config["take_profit_atr_mult"]
        raw_sp = atr_pct * sm
        sp = _clamp(
            raw_sp,
            float(self.config.get("min_stop_loss_pct", 0.8)),
            float(self.config.get("max_stop_loss_pct", 6.0)),
        )
        sl = lp * (1 - sp / 100) if direction == "bull" else lp * (1 + sp / 100)
        tp_pct = sp * tm / max(sm, 1e-9)
        tp = lp * (1 + tp_pct / 100) if direction == "bull" else lp * (1 - tp_pct / 100)
        # P0-X5: 仅当真实发生钳制时才标注"钳制"，避免误导
        sp_clamped = abs(raw_sp - sp) > 1e-6
        d["ATR止损"] = (
            f"{sl:.6g}(ATR{sm}×={raw_sp:.2f}%"
            + (f"→钳制{sp:.2f}%" if sp_clamped else "")
            + ")"
        )
        d["ATR止盈"] = f"{tp:.6g}(盈亏比{tm/max(sm,1e-9):.2f})"

        # ── P3-1: 动态止损止盈（联动 swing/Fib/BB） ──
        dyn_sl_pct = sp; dyn_tp_pct = tp_pct; dyn_rr = tm / max(sm, 1e-9); dyn_tp_src = "ATR"
        if bool(self.config.get("dynamic_sl_tp_enabled", True)) and h1 is not None and len(h1) >= 5:
            try:
                h1_atr_v = float(_atr(h1, 14))
                period = int(self.config.get("boll_period", 20))
                std_m  = float(self.config.get("boll_std_mult", 2.0))
                h1_b   = _bollinger(h1, period, std_m)
                bb_up  = float(h1_b["boll_up"].iloc[-1])
                bb_dn  = float(h1_b["boll_down"].iloc[-1])
                swing_high = float(swing_info.get("swing_high", 0)) if swing_info else 0.0
                swing_low  = float(swing_info.get("swing_low",  0)) if swing_info else 0.0
                if direction == "bull":
                    # 止损候选（多头）
                    sl_atr   = lp - h1_atr_v * 1.5
                    sl_swing = (swing_low - h1_atr_v * 0.3) if swing_low > 0 else 0.0
                    sl_dyn   = max(sl_atr, sl_swing) if sl_swing > 0 else sl_atr
                    sl_dyn   = min(sl_dyn, lp * 0.998)             # 不能高于当前价
                    # 止盈候选（多头）
                    tp_cands = []
                    if bb_up > lp:                  tp_cands.append(("BB上轨", bb_up))
                    if swing_high > lp:             tp_cands.append(("swing_high", swing_high - h1_atr_v * 0.2))
                    tp_cands.append(("ATR×3", lp + h1_atr_v * 3.0))
                    tp_dyn_src, tp_dyn = min(tp_cands, key=lambda x: x[1])
                    risk   = lp - sl_dyn
                    reward = tp_dyn - lp
                else:
                    sl_atr   = lp + h1_atr_v * 1.5
                    sl_swing = (swing_high + h1_atr_v * 0.3) if swing_high > 0 else 0.0
                    sl_dyn   = min(sl_atr, sl_swing) if sl_swing > 0 else sl_atr
                    sl_dyn   = max(sl_dyn, lp * 1.002)
                    tp_cands = []
                    if bb_dn < lp:                  tp_cands.append(("BB下轨", bb_dn))
                    if 0 < swing_low < lp:          tp_cands.append(("swing_low", swing_low + h1_atr_v * 0.2))
                    tp_cands.append(("ATR×3", lp - h1_atr_v * 3.0))
                    tp_dyn_src, tp_dyn = max(tp_cands, key=lambda x: x[1])
                    risk   = sl_dyn - lp
                    reward = lp - tp_dyn
                if risk > 0 and reward > 0:
                    dyn_sl_pct = abs(lp - sl_dyn) / lp * 100
                    dyn_tp_pct = abs(tp_dyn - lp) / lp * 100
                    dyn_rr     = reward / risk
                    dyn_tp_src = tp_dyn_src
                    d["动态止损"] = f"{sl_dyn:.6g} (-{dyn_sl_pct:.2f}%)"
                    d["动态止盈"] = f"{tp_dyn:.6g} (+{dyn_tp_pct:.2f}%, 来源={tp_dyn_src})"
                    d["动态盈亏比"] = f"{dyn_rr:.2f}"
                    # P3-2: 盈亏比过低软扣分
                    min_rr = float(self.config.get("min_rr_ratio", 1.5))
                    if 0 < dyn_rr < min_rr:
                        rr_pen = float(self.config.get("rr_too_low_penalty", 6.0))
                        score = max(0.0, score - rr_pen * (1.0 - dyn_rr / min_rr))
                        d["盈亏比警告"] = f"⚠ {dyn_rr:.2f}<{min_rr:.2f} 扣分"
            except Exception:
                pass

        # ── P3-4: 追踪止损输出字段（不只是文本） ──
        ta = atr_pct * self.config["trail_activate_atr_mult"]
        td_ = atr_pct * self.config["trail_distance_atr_mult"]
        d["追踪止损"] = f"浮盈>{ta:.1f}%激活 距离{td_:.1f}%"
        # ── P4-3: 仓位建议（数字字段 + 文本两种） ──
        bs = self.config["position_size"]; sc_ = min(1.0, 3.0 / max(atr_pct, 1.0))
        # P4-3: 信号置信度 × 波动率自适应（替代纯 base_size 缩放）
        confidence = _clamp(score / 100.0, 0.0, 1.0)
        # 风险公式：position_pct = base_risk(1%) × confidence / sl_pct
        base_risk = 0.01
        sl_pct_use = max(dyn_sl_pct, 0.5)   # 至少 0.5% 止损（防爆仓）
        position_advice = _clamp(base_risk * confidence / (sl_pct_use / 100.0) * 100, 0.0, 20.0)
        position_advice = round(min(position_advice, bs * sc_ * 100, 10.0), 2)
        d["建议仓位"] = f"{position_advice:.2f}%"
        # ── 改进五：OI 持仓量同向确认 ────────────────────────────────────────
        if bool(self.config.get("oi_confirm_enabled", True)):
            oi_chg = float(ed.get("oi_change_pct", 0) or 0)
            oi_min = float(self.config.get("oi_min_change_pct", 2.0))
            oi_bonus = float(self.config.get("oi_confirm_bonus", 8.0))
            oi_pen = float(self.config.get("oi_diverge_penalty", 8.0))
            if abs(oi_chg) >= oi_min:
                if (direction == "bull" and oi_chg > 0) or (direction == "bear" and oi_chg < 0):
                    score += oi_bonus
                    d["OI确认"] = f"OI{oi_chg:+.1f}% 量价同向✓"
                else:
                    score -= oi_pen
                    d["OI警告"] = f"OI{oi_chg:+.1f}% 背离⚠"
        risk_cfg = {
            "stop_loss_pct":      round(float(sp), 4),
            "take_profit_pct":    round(float(tp_pct), 4),
            # P3-1: 动态止损止盈
            "dynamic_stop_loss_pct":   round(float(dyn_sl_pct), 4),
            "dynamic_take_profit_pct": round(float(dyn_tp_pct), 4),
            "dynamic_rr_ratio":   round(float(dyn_rr), 4),
            "dynamic_tp_source":  dyn_tp_src,
            # P3-4: 追踪止损字段化
            "trail_activate_pct": round(float(ta), 4),
            "trail_distance_pct": round(float(td_), 4),
            # P4-3: 仓位建议
            "position_pct":       round(float(position_advice), 4),
        }
        return round(max(0, score), 2), d, risk_cfg

    def scan_all_symbols(self, symbols):
        # P3-3: ThreadPoolExecutor 并行扫描（max 8 线程）
        from concurrent.futures import ThreadPoolExecutor, as_completed
        res = []
        fail_stats: Dict[str, int] = {}
        # P3-6: 失败原因结构化分类
        step_stats = {"D1": 0, "H4": 0, "H1": 0, "15m": 0, "评分不足": 0, "数据不足": 0, "异常": 0}

        def _classify_fail(reason: str) -> str:
            r = str(reason)
            if "数据不足" in r or "缺少最新价" in r:  return "数据不足"
            if r.startswith("D1") or r.startswith("4H"):  return "D1"
            if "H4" in r:                              return "H4"
            if r.startswith("1H") or "H1" in r:        return "H1"
            if r.startswith("15m"):                    return "15m"
            if "评分不足" in r:                         return "评分不足"
            return "异常"

        max_workers = min(8, max(1, len(symbols)))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self._safe_scan, s): s for s in symbols}
            for future in as_completed(futures):
                try:
                    r = future.result(timeout=15)
                except Exception as e:
                    fail_stats[f"[future异常]{type(e).__name__}"] = fail_stats.get(f"[future异常]{type(e).__name__}", 0) + 1
                    step_stats["异常"] += 1
                    continue
                if r is None:
                    step_stats["异常"] += 1
                    continue
                if r.get("passed"):
                    res.append(r)
                else:
                    reason = str(r.get("details", {}).get("状态", "unknown"))
                    key = reason[:30]
                    fail_stats[key] = fail_stats.get(key, 0) + 1
                    step_stats[_classify_fail(reason)] += 1

        res.sort(key=lambda r: float(r.get("score", 0) or 0), reverse=True)
        top = sorted(fail_stats.items(), key=lambda x: x[1], reverse=True)[:15]
        print(f"[小月布林诊断] 通过{len(res)}/{len(symbols)} 步骤分布:{step_stats}")
        print(f"[小月布林诊断] 失败TOP15: { {k:v for k,v in top} }")
        return {
            "type": "xiaoyue",
            "all_opportunities": res[:int(self.config.get("top_n", 12))],
            "scanned_symbols": len(symbols),
            "diagnostics": {
                "step_stats": step_stats,
                "fail_top15": {k: v for k, v in top},
                "pass_rate_pct": round(len(res) / max(len(symbols), 1) * 100, 1),
            },
        }

    def _safe_scan(self, sym):
        """单个符号扫描 + 异常保护"""
        try:
            return self.scan_symbol(sym)
        except Exception as exc:
            inst_id = getattr(sym, 'inst_id', '')
            print(f"[小月布林] {inst_id} 扫描异常: {exc}")
            return {"passed": False, "details": {"状态": f"[异常]{type(exc).__name__}"}}

    def generate_signal(self, data, *a, **kw):
        km = data.get("klines_map", {}) or {}
        # 内存保护：截断到最小计算窗口（4GB机器回测关键优化）
        if isinstance(km, dict):
            limits = {"1D":50,"1d":50,"4H":60,"4h":60,"1H":120,"1h":120,"15m":80,"15M":80,"3m":60,"3M":60}
            for k, v in list(km.items()):
                cap = limits.get(k, 120)
                if isinstance(v, (pd.DataFrame, list, tuple)) and len(v) > cap:
                    if isinstance(v, pd.DataFrame):
                        km[k] = v.tail(cap)
                    elif isinstance(v, (list, tuple)):    # P0-X4: tuple 一并支持
                        km[k] = list(v)[-cap:]

        class S: pass
        s = S(); s.inst_id = str(data.get("inst_id", data.get("symbol", "BT")))
        s.last_price = float(data.get("last_price", data.get("close", 0)) or 0)
        s.volume_24h = float(data.get("volume_24h", 0) or 0); s.extra_data = {"klines": km}
        r = self.scan_symbol(s)
        if not r.get("passed"): return None
        direction = str(r.get("direction", "") or "").upper()
        action = "BUY" if direction == "BUY" else "SHORT"
        return {
            "action": action,
            "score": r.get("score"),
            "reason": (r.get("signals") or [""])[0],
            "details": r.get("details", {}),
            "take_profit_pct": r.get("take_profit_pct"),
            "stop_loss_pct": r.get("stop_loss_pct"),
        }


STRATEGY_NAME = "小月期货多周期布林趋势转折"
STRATEGY_TYPE = "scan"
STRATEGY_CLASS = XiaoYueBollMacdScanner
BACKTEST_CLASS = XiaoYueBollMacdScanner
