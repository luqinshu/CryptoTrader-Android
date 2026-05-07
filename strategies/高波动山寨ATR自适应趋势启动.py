#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高波动山寨币 ATR 自适应趋势启动扫描策略 v1.1

专为 ATR% > 3% 的山寨币设计。核心理念：
  · 用 ATR 倍数替代固定百分比（ATR=3%时门槛 2.4%回调，ATR=15%时门槛 12%回调）
  · 用仓位缩放替代品种过滤（高波动=小仓位，而非直接过滤）
  · 用紧追踪止损让利润奔跑（0.5×ATR，比现有策略更紧）
  · 用单通道 1H→15m→3m 替代多引擎共识（速度优先）

v1.1 改进:
  · Layer1 EMA 分级检测：EMA12>EMA26 即通过（早期信号），价格>EMA12 为满分
  · 指标缓存：ADX/ATR/BB 同品种同周期 60 秒内复用
  · 可选 4H 多周期上下文：4H 趋势同向时加分（非硬过滤）
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from strategies._shared.indicators import _to_df, _atr, _rsi_wilder, _clamp, _ema, _cfg_float, _cfg_int

logger = logging.getLogger(__name__)

# ══ 指标缓存（同品种同周期 60s TTL，线程安全）══
import threading as _threading
_INDICATOR_CACHE: Dict[str, Tuple[float, Any]] = {}
_CACHE_LOCK = _threading.Lock()
_CACHE_TTL = 60.0

def _cached(key: str, compute_fn, *args):
    now = __import__('time').time()
    with _CACHE_LOCK:
        entry = _INDICATOR_CACHE.get(key)
        if entry and now - entry[0] < _CACHE_TTL:
            return entry[1]
    result = compute_fn(*args)
    with _CACHE_LOCK:
        _INDICATOR_CACHE[key] = (now, result)
        if len(_INDICATOR_CACHE) > 500:
            expired = [k for k, (t, _) in _INDICATOR_CACHE.items() if now - t > _CACHE_TTL * 2]
            for k in expired: _INDICATOR_CACHE.pop(k, None)
    return result

try:
    from src.scanner.base_scanner import BaseScannerStrategy, ScanCondition, ScannerSymbol
    _HAS_BASE = True
except ImportError:
    BaseScannerStrategy = object
    ScanCondition = None
    ScannerSymbol = None
    _HAS_BASE = False


CONFIG_SCHEMA: Dict[str, Any] = {
    # ── 流动性 ──
    "min_volume_24h":           {"type": "float", "default": 5_000_000,   "label": "最小24H成交额(USDT)"},
    "min_score":                {"type": "float", "default": 62.0,         "label": "最低输出分数"},
    "top_n":                    {"type": "int",   "default": 15,           "label": "最多输出信号数"},

    # ── Layer 1: 1H 趋势萌芽 ──
    "h1_adx_period":            {"type": "int",   "default": 14,           "label": "1H ADX周期"},
    "h1_adx_min":               {"type": "float", "default": 14.0,         "label": "1H ADX最低阈值"},
    "h1_adx_rising_bars":       {"type": "int",   "default": 3,            "label": "ADX连续上升根数"},
    "h1_ema_fast":              {"type": "int",   "default": 12,           "label": "1H快EMA"},
    "h1_ema_slow":              {"type": "int",   "default": 26,           "label": "1H慢EMA"},
    "h1_ema_strict_mode":       {"type": "bool",  "default": False,        "label": "EMA严格模式(需price>EMA12>EMA26全满足)"},
    "h1_rsi_period":            {"type": "int",   "default": 14,           "label": "1H RSI周期"},
    "h1_rsi_min_long":          {"type": "float", "default": 35.0,         "label": "多头RSI下限"},
    "h1_rsi_max_long":          {"type": "float", "default": 75.0,         "label": "多头RSI上限"},
    "h1_rsi_min_short":         {"type": "float", "default": 25.0,         "label": "空头RSI下限"},
    "h1_rsi_max_short":         {"type": "float", "default": 65.0,         "label": "空头RSI上限"},

    # ── 可选: 4H 多周期上下文 ──
    "enable_mtf_context":       {"type": "bool",  "default": True,         "label": "启用4H多周期上下文加分"},
    "h4_ema_fast":              {"type": "int",   "default": 20,           "label": "4H快EMA"},
    "h4_ema_slow":              {"type": "int",   "default": 50,           "label": "4H慢EMA"},
    "h4_mtf_bonus_max":         {"type": "float", "default": 8.0,          "label": "4H同向最高加分"},

    # ── Layer 2: 15m 挤压蓄力 ──
    "m15_bb_period":            {"type": "int",   "default": 20,           "label": "15m BB周期"},
    "m15_bb_width_percentile":  {"type": "float", "default": 0.25,         "label": "BB带宽历史分位上限"},
    "m15_vol_dryup_ratio":      {"type": "float", "default": 0.65,         "label": "缩量系数(当前/基线)"},
    "m15_vol_baseline":         {"type": "int",   "default": 20,           "label": "成交量基线根数"},
    "m15_range_atr_mult":       {"type": "float", "default": 1.3,          "label": "5根K线振幅上限(×ATR)"},
    "m15_stable_bars":          {"type": "int",   "default": 5,            "label": "价格稳定检测根数"},

    # ── Layer 3: 3m 入场触发 (ATR自适应) ──
    "m3_pullback_atr_min":      {"type": "float", "default": 0.8,          "label": "最小回调(×ATR)"},
    "m3_pullback_atr_max":      {"type": "float", "default": 3.5,          "label": "最大回调(×ATR)"},
    "m3_stabilize_bars":        {"type": "int",   "default": 3,            "label": "企稳确认根数"},
    "m3_vol_surge_ratio":       {"type": "float", "default": 1.5,          "label": "突破量比(当前/近5根)"},
    "m3_ema_fast":              {"type": "int",   "default": 8,            "label": "3m快EMA"},
    "m3_ema_slow":              {"type": "int",   "default": 21,           "label": "3m慢EMA"},
    "m3_min_atr_pct":           {"type": "float", "default": 2.0,          "label": "3m最低ATR%(低于此视为横盘)"},
    "m3_nr_lookback":           {"type": "int",   "default": 5,            "label": "NR检测回溯根数(NR5)"},
    "m3_require_nr":            {"type": "bool",  "default": False,        "label": "强制要求NR窄幅"},
    "m3_vwap_deviation_max":    {"type": "float", "default": 1.5,          "label": "VWAP最大偏离%(×ATR)"},

    # ── 风控 ──
    "stop_atr_mult":            {"type": "float", "default": 1.5,          "label": "止损ATR倍数"},
    "risk_per_trade_pct":       {"type": "float", "default": 0.5,          "label": "单笔风险%(账户)"},
    "trail_activate_atr_mult":  {"type": "float", "default": 1.0,          "label": "追踪激活浮盈(×ATR)"},
    "trail_distance_atr_mult":  {"type": "float", "default": 0.5,          "label": "追踪距离(×ATR)"},
    "btc_dump_threshold_pct":   {"type": "float", "default": -3.0,         "label": "BTC暴跌熔断阈值%"},
    "btc_dump_penalty":         {"type": "float", "default": 12.0,         "label": "BTC暴跌对山寨多头降分"},
    "funding_extreme_pct":      {"type": "float", "default": 0.15,         "label": "极端资金费率%"},
    "funding_penalty":          {"type": "float", "default": 6.0,          "label": "极端费率降分"},
    "oi_confirm_enabled":       {"type": "bool",  "default": True,         "label": "启用OI+价格方向确认"},
    "oi_confirm_weight":        {"type": "float", "default": 4.0,          "label": "OI确认升降分幅度"},
    "position_size":            {"type": "float", "default": 0.05,         "label": "基准仓位比例"},
    "max_position_pct":         {"type": "float", "default": 0.10,         "label": "最大仓位上限"},
    "allow_short":              {"type": "bool",  "default": True,          "label": "允许做空"},
    "take_profit_atr_mult":     {"type": "float", "default": 2.5,          "label": "止盈ATR倍数(盈亏比~1.7)"},
}

_DEFAULTS = {k: v["default"] for k, v in CONFIG_SCHEMA.items()}


class VolatileAltcoinATRTrendScanner(BaseScannerStrategy if _HAS_BASE else object):
    """高波动山寨币 ATR 自适应趋势启动扫描器。"""

    required_bars = ["1H", "15m", "3m"]
    strategy_type = "scan"
    name = "高波动山寨ATR自适应趋势启动"
    description = "1H→15m→3m ATR全自适应流水线 | 0.5%风险/笔 | 0.5×ATR紧追踪 | BTC熔断保护"

    def __init__(self, config: Optional[Dict] = None):
        self.config = {**_DEFAULTS, **(config or {})}
        if _HAS_BASE and hasattr(super(), "__init__"):
            try: super().__init__(self.config)
            except Exception: pass

    def _init_conditions(self):
        if ScanCondition is None:
            return
        self.add_condition(ScanCondition(
            name="流动性", field="volume_24h", operator=">=",
            value=self.config["min_volume_24h"],
        ))

    def get_config_schema(self) -> Dict:
        return dict(CONFIG_SCHEMA)

    # ══════════════════════════════════════════════════
    # 单标的扫描
    # ══════════════════════════════════════════════════

    def scan_symbol(self, symbol) -> Dict:
        inst_id = getattr(symbol, "inst_id", "")
        ed = getattr(symbol, "extra_data", {}) or {}
        klines = ed.get("klines", {}) or {}

        def _safe_kline(d, *keys):
            for k in keys:
                v = d.get(k)
                if v is not None and (not isinstance(v, (list, tuple)) or len(v) > 0):
                    return v
            return []

        h1 = _to_df(_safe_kline(klines, "1H", "1h"))
        m15 = _to_df(_safe_kline(klines, "15m", "15M"))
        m3 = _to_df(_safe_kline(klines, "3m", "3M"))
        h4 = _to_df(_safe_kline(klines, "4H", "4h"))
        lp = float(getattr(symbol, "last_price", 0) or 0)

        fail = lambda reason: {
            "symbol": inst_id, "passed": False, "score": 0.0,
            "direction": "WAIT", "category": "ATR自适应扫描",
            "details": {"状态": reason},
        }

        if lp <= 0:
            return fail("缺少最新价")
        if len(h1) < 50 or len(m15) < 40 or len(m3) < 40:
            return fail(f"数据不足(H1={len(h1)}/50, 15m={len(m15)}/40, 3m={len(m3)}/40)")

        # ── Layer 1: 1H 趋势萌芽 ────────────────────
        h1_ok, h1_score, h1_dir, h1_detail, atr_pct = self._layer1_h1_trend_sprout(h1, inst_id)
        if not h1_ok:
            return fail(f"1H未通过: {h1_detail}")

        # ── Layer 2: 15m 挤压蓄力 ────────────────────
        m15_ok, m15_score, m15_detail = self._layer2_m15_squeeze(m15, h1_dir, atr_pct)
        if not m15_ok:
            return fail(f"15m未通过: {m15_detail}")

        # ── Layer 3: 3m 入场触发 ────────────────────
        m3_ok, m3_score, m3_detail, m3_extra = self._layer3_m3_entry(
            m3, h1_dir, atr_pct, lp
        )
        if not m3_ok:
            return fail(f"3m未通过: {m3_detail}")

        # ── 综合评分 ──────────────────────────────────
        base_score = h1_score * 0.35 + m15_score * 0.25 + m3_score * 0.40

        # 初始化 details（4H上下文 + 风控调整都会写入）
        details: Dict[str, str] = {
            "ATR%": f"{atr_pct:.1f}%",
            "1H方向": h1_dir,
            "1H评分": f"{h1_score:.0f}",
            "15m评分": f"{m15_score:.0f}",
            "3m评分": f"{m3_score:.0f}",
            "回调幅度": f"{m3_extra.get('pullback_pct', 0):.1f}%",
            "企稳": "✓" if m3_extra.get("stable") else "✗",
            "EMA金叉": "✓" if m3_extra.get("golden_cross") else "✗",
            "放量": "✓" if m3_extra.get("vol_surge") else "✗",
            "NR窄幅": "✓" if m3_extra.get("nr_ok") else "✗",
            "VWAP": "✓" if m3_extra.get("vwap_ok") else "✗",
        }

        # ── 4H 多周期上下文加分（可选，非硬过滤）─────────
        mtf_bonus = 0.0
        if bool(self.config.get("enable_mtf_context", True)) and len(h4) >= 30:
            try:
                h4_ema_f = h4["c"].ewm(span=int(self.config["h4_ema_fast"]), adjust=False).mean()
                h4_ema_s = h4["c"].ewm(span=int(self.config["h4_ema_slow"]), adjust=False).mean()
                h4_bull = float(h4_ema_f.iloc[-1]) > float(h4_ema_s.iloc[-1])
                if (h1_dir == "bull" and h4_bull) or (h1_dir == "bear" and not h4_bull):
                    gap = abs(float(h4_ema_f.iloc[-1]) - float(h4_ema_s.iloc[-1])) / max(float(h4_ema_s.iloc[-1]), 1e-9) * 100
                    mtf_bonus = min(self.config["h4_mtf_bonus_max"], gap * 1.5)
                    details["4H上下文"] = f"✓ 同向共振 +{mtf_bonus:.1f}分(4H EMA张开{gap:.1f}%)"
            except Exception:
                pass

        score = round(_clamp(base_score + mtf_bonus, 0, 100), 2)

        # ── 风控调整 ──────────────────────────────────
        score, details = self._apply_risk_adjustments(
            score, inst_id, ed, h1_dir, atr_pct, lp, details, h1
        )

        if score < self.config["min_score"]:
            return fail(f"评分不足({score:.1f}<{self.config['min_score']})")

        return {
            "symbol": inst_id, "passed": True,
            "score": round(score, 2),
            "opportunity_score": round(score, 2),
            "direction": "BUY" if h1_dir == "bull" else "SELL",
            "category": "高波动ATR自适应趋势启动",
            "signals": [
                f"{'多头' if h1_dir == 'bull' else '空头'}趋势启动 {score:.1f}分",
                f"ATR={atr_pct:.1f}% ADX上升 15m挤压 3m{'回调企稳' if h1_dir == 'bull' else '反弹企稳'}",
            ],
            "last_price": lp, "volume_24h": float(getattr(symbol, "volume_24h", 0) or 0),
            "details": details,
            "ranking_factors": {
                "trend": h1_score,
                "trigger": m3_score,
                "volume": float(m3_extra.get("vol_surge", 0)) * 50,
                "location": float(m3_extra.get("vwap_ok", 0)) * 50 + 50,
                "freshness": m15_score * 0.5 + (8 if m3_extra.get("nr_ok") else 0),
                "risk": _clamp(100 - atr_pct * 5, 0, 100),
            },
        }

    # ══════════════════════════════════════════════════
    # Layer 1: 1H 趋势萌芽检测
    # ══════════════════════════════════════════════════

    def _layer1_h1_trend_sprout(
        self, h1: pd.DataFrame, inst_id: str = ""
    ) -> Tuple[bool, float, str, str, float]:
        """1H ADX上升 + EMA排列 + RSI区间 → 趋势萌芽确认"""
        c, h, l = h1["c"], h1["h"], h1["l"]
        n = len(c)
        if n < 50:
            return False, 0, "neutral", "数据不足", 2.0

        # ATR%
        atr_pct = float(_atr(h1, 14) / c.iloc[-1] * 100) if c.iloc[-1] > 0 else 2.0
        atr_pct = max(atr_pct, 0.5)

        # ADX + DI（使用缓存避免同品种重复计算）
        cache_key = f"adx_{inst_id}_1H"
        adx, pdi, mdi = _cached(cache_key, _calc_adx, h1, int(self.config["h1_adx_period"]))
        rising = self._adx_rising(h1, int(self.config["h1_adx_period"]),
                                  int(self.config["h1_adx_rising_bars"]),
                                  adx_cache=(adx, pdi, mdi))

        if adx < self.config["h1_adx_min"]:
            return False, 0, "neutral", f"ADX={adx:.1f}<{self.config['h1_adx_min']}", atr_pct

        # 方向判断：EMA12>EMA26 即通过（早期信号），价格在 EMA12 上方为满分
        ema_f = c.ewm(span=int(self.config["h1_ema_fast"]), adjust=False).mean()
        ema_s = c.ewm(span=int(self.config["h1_ema_slow"]), adjust=False).mean()
        rsi = _rsi_wilder(c, int(self.config["h1_rsi_period"]))
        ema_f_last = float(ema_f.iloc[-1])
        ema_s_last = float(ema_s.iloc[-1])
        close_last = float(c.iloc[-1])

        strict = bool(self.config.get("h1_ema_strict_mode", False))
        if pdi > mdi and ema_f_last > ema_s_last:
            direction = "bull"
            price_above = close_last > ema_f_last
            if strict and not price_above:
                return False, 0, "bull", f"EMA严格模式: 价格{close_last:.5g}≤EMA12{ema_f_last:.5g}", atr_pct
            rsi_ok = self.config["h1_rsi_min_long"] <= rsi <= self.config["h1_rsi_max_long"]
        elif mdi > pdi and ema_f_last < ema_s_last:
            direction = "bear"
            price_below = close_last < ema_f_last
            if strict and not price_below:
                return False, 0, "bear", f"EMA严格模式: 价格{close_last:.5g}≥EMA12{ema_f_last:.5g}", atr_pct
            rsi_ok = self.config["h1_rsi_min_short"] <= rsi <= self.config["h1_rsi_max_short"]
        else:
            return False, 0, "neutral", "趋势方向不明确", atr_pct

        if not rsi_ok:
            return False, 0, direction, f"RSI={rsi:.1f}不在区间", atr_pct

        # 评分（EMA 分级：price>EMA12 满分，仅 EMA12>EMA26 降权。新增 MACD 柱动量）
        adx_score = _clamp((adx - 12) / 18, 0, 1) * 35
        rising_bonus = 15 if rising else 0
        ema_spread = abs(ema_f_last - ema_s_last) / max(ema_s_last, 1e-9) * 100
        ema_score = _clamp(ema_spread, 0, 5) * 6
        if (direction == "bull" and close_last > ema_f_last) or \
           (direction == "bear" and close_last < ema_f_last):
            ema_score += 10
        # MACD 柱状线动量（连续 N 根递增 = 趋势加速）
        macd_bonus = 0
        try:
            macd_line = ema_f - ema_s
            macd_signal = macd_line.ewm(span=9, adjust=False).mean()
            macd_hist = macd_line - macd_signal
            if len(macd_hist) >= 3:
                last3 = macd_hist.iloc[-3:].values
                if direction == "bull" and last3[-1] > 0 and all(last3[i] >= last3[i-1] for i in range(1,3)):
                    macd_bonus = 8  # 多头 MACD 柱连续 3 根递增
                elif direction == "bear" and last3[-1] < 0 and all(last3[i] <= last3[i-1] for i in range(1,3)):
                    macd_bonus = 8  # 空头 MACD 柱连续 3 根递减
        except Exception:
            pass
        rsi_score = _clamp(1 - abs(rsi - 55) / 35, 0, 1) * 5
        score = min(100, adx_score + rising_bonus + ema_score + rsi_score + macd_bonus)

        direction_cn = "bullish" if direction == "bull" else "bearish"
        return True, score, direction, f"ADX={adx:.1f}{'↑' if rising else ''} {direction_cn} RSI={rsi:.1f}", atr_pct

    # ══════════════════════════════════════════════════
    # Layer 2: 15m 挤压蓄力
    # ══════════════════════════════════════════════════

    def _layer2_m15_squeeze(
        self, m15: pd.DataFrame, direction: str, atr_pct: float
    ) -> Tuple[bool, float, str]:
        """15m BB带宽压缩 + 缩量 + 价格稳定 → 蓄力确认"""
        c, h, l, v = m15["c"], m15["h"], m15["l"], m15["vol"]
        if len(c) < 30:
            return False, 0, "15m数据不足"

        # BB 带宽分位
        bb_period = int(self.config["m15_bb_period"])
        ma = c.rolling(bb_period).mean()
        std = c.rolling(bb_period).std()
        bbw = (std * 2) / ma
        cur_bbw = float(bbw.iloc[-1])
        hist_bbw = bbw.dropna().values
        if len(hist_bbw) < 10:
            return False, 0, "BB历史不足"
        rank = float(np.sum(hist_bbw <= cur_bbw) / len(hist_bbw))
        squeeze = rank <= self.config["m15_bb_width_percentile"]

        # 缩量
        vol_base = float(v.iloc[-self.config["m15_vol_baseline"]-1:-1].mean())
        vol_recent = float(v.iloc[-5:].mean())
        dryup = vol_recent < vol_base * self.config["m15_vol_dryup_ratio"] if vol_base > 0 else False

        # 价格稳定（5根K线振幅 < 1.3×ATR）— 用 15m 自身的 ATR% 而非 1H 的 atr_pct
        m15_atr = float(_atr(m15, 14))
        m15_atr_pct = m15_atr / max(float(c.iloc[-1]), 1e-9) * 100
        recent_range = (h.iloc[-5:].max() - l.iloc[-5:].min()) / max(c.iloc[-1], 1e-9) * 100
        stable = recent_range < max(m15_atr_pct, atr_pct * 0.5) * self.config["m15_range_atr_mult"]

        if not squeeze and not dryup:
            return False, 0, f"挤压不足(分位{rank:.0%})且未缩量"
        if not stable:
            return False, 0, f"价格波动过大({recent_range:.1f}%>{max(m15_atr_pct,atr_pct*0.5)*self.config['m15_range_atr_mult']:.1f}%)"

        score = 0
        if squeeze: score += 40 * (1 - rank)
        if dryup: score += 30
        if stable: score += 30
        return True, min(100, score), f"分位{rank:.0%} {'缩量' if dryup else ''} {'稳定' if stable else ''}"

    # ══════════════════════════════════════════════════
    # Layer 3: 3m 入场触发
    # ══════════════════════════════════════════════════

    def _layer3_m3_entry(
        self, m3: pd.DataFrame, direction: str, atr_pct: float, last_price: float
    ) -> Tuple[bool, float, str, Dict]:
        """3m ATR自适应回调 + 企稳 + 放量 + EMA金叉 → 入场"""
        c, h, l, v = m3["c"], m3["h"], m3["l"], m3["vol"]
        n = len(c)
        min_bars = 20
        if n < min_bars:
            return False, 0, f"3m数据不足({n}<{min_bars})", {}

        # ATR 自适应回调门槛
        pullback_min = max(self.config["m3_pullback_atr_min"] * atr_pct, self.config["m3_min_atr_pct"] * 0.5)
        pullback_max = self.config["m3_pullback_atr_max"] * atr_pct
        stab_bars = int(self.config["m3_stabilize_bars"])
        vol_surge = self.config["m3_vol_surge_ratio"]

        ema_fast = c.ewm(span=int(self.config["m3_ema_fast"]), adjust=False).mean()
        ema_slow = c.ewm(span=int(self.config["m3_ema_slow"]), adjust=False).mean()

        cur_close = float(c.iloc[-1])
        cur_vol = float(v.iloc[-1])
        base_vol = float(v.iloc[-6:-1].mean()) if len(v) >= 6 else cur_vol

        if direction == "bull":
            # 找近 20 根内的局部高点和其后的回调（使用位置索引避免 index offset bug）
            search_end = n - stab_bars
            lookback = min(20, search_end)
            # 使用 .iloc 获取无歧义的位置子集
            seg_close = c.iloc[-lookback:search_end].reset_index(drop=True)
            if len(seg_close) < 3:
                return False, 0, "回调段不足", {}
            peak_rel_idx = int(seg_close.idxmax())
            peak = float(seg_close.iloc[peak_rel_idx])
            # 回调低点（peak之后）
            if peak_rel_idx < len(seg_close) - 1:
                pullback_low = float(seg_close.iloc[peak_rel_idx:].min())
            else:
                pullback_low = peak
            pullback_pct = (peak - pullback_low) / max(peak, 1e-9) * 100

            if pullback_pct < pullback_min:
                return False, 0, f"回调幅度不足({pullback_pct:.1f}%<{pullback_min:.1f}%)", {}
            if pullback_pct > pullback_max:
                return False, 0, f"回调过深({pullback_pct:.1f}%>{pullback_max:.1f}%)", {}

            # 企稳: 最后 stab_bars 根收盘 > EMA8
            stable = all(c.iloc[-stab_bars:].values >= ema_fast.iloc[-stab_bars:].values * 0.997)
            # EMA 金叉
            golden_cross = float(ema_fast.iloc[-1]) > float(ema_slow.iloc[-1])

        else:  # bear
            search_end = n - stab_bars
            lookback = min(20, search_end)
            seg_close = c.iloc[-lookback:search_end].reset_index(drop=True)
            if len(seg_close) < 3:
                return False, 0, "反弹段不足", {}
            trough_rel_idx = int(seg_close.idxmin())
            trough = float(seg_close.iloc[trough_rel_idx])
            if trough_rel_idx < len(seg_close) - 1:
                bounce_high = float(seg_close.iloc[trough_rel_idx:].max())
            else:
                bounce_high = trough
            bounce_pct = (bounce_high - trough) / max(trough, 1e-9) * 100

            if bounce_pct < pullback_min:
                return False, 0, f"反弹幅度不足({bounce_pct:.1f}%<{pullback_min:.1f}%)", {}
            if bounce_pct > pullback_max:
                return False, 0, f"反弹过高({bounce_pct:.1f}%>{pullback_max:.1f}%)", {}

            stable = all(c.iloc[-stab_bars:].values <= ema_fast.iloc[-stab_bars:].values * 1.003)
            golden_cross = float(ema_fast.iloc[-1]) < float(ema_slow.iloc[-1])

        if not stable:
            return False, 0, f"企稳不足(近{stab_bars}根未保持{'EMA8上方' if direction=='bull' else 'EMA8下方'})", {}

        # 放量
        vol_ok = cur_vol >= base_vol * vol_surge if base_vol > 0 else True

        # ── NR 窄幅检测 ────────────────────────────
        nr_ok = True
        nr_bonus = 0
        nr_look = int(self.config.get("m3_nr_lookback", 5))
        if nr_look >= 3 and n >= nr_look + 2:
            try:
                nr_ranges = [(float(h.iloc[-i]) - float(l.iloc[-i])) / max(float(c.iloc[-i]), 1e-9)
                             for i in range(1, nr_look + 1)]
                prev_ranges = [(float(h.iloc[-(i+1)]) - float(l.iloc[-(i+1)])) / max(float(c.iloc[-(i+1)]), 1e-9)
                               for i in range(1, nr_look + 1)]
                if nr_ranges and prev_ranges:
                    nr_ok = max(nr_ranges) <= min(prev_ranges + [1.0])
                    if nr_ok: nr_bonus = 8
                if self.config.get("m3_require_nr") and not nr_ok:
                    return False, 0, f"NR{nr_look}未满足(近{nr_look}根非全局最窄)", {}
            except Exception:
                pass

        # ── VWAP 偏离检查 ────────────────────────────
        vwap_ok = True
        vwap_bonus = 0
        vwap_max = float(self.config.get("m3_vwap_deviation_max", 1.5))
        try:
            typical = (h + l + c) / 3
            vwap_series = (typical * v).cumsum() / v.cumsum()
            vwap_val = float(vwap_series.iloc[-1])
            vwap_dev = abs(cur_close - vwap_val) / max(vwap_val, 1e-9) * 100
            vwap_max_pct = atr_pct * vwap_max
            vwap_ok = vwap_dev < vwap_max_pct
            if vwap_ok and vwap_dev < vwap_max_pct * 0.5:
                vwap_bonus = 6
        except Exception:
            pass

        # 评分（回调质量 + 企稳 + 金叉 + 放量 + NR + VWAP）
        pb_pct_val = pullback_pct if direction == "bull" else bounce_pct
        ideal_pb = pullback_min * 2.0
        pb_score = _clamp(pb_pct_val / max(ideal_pb, 1e-9), 0, 1) * 42
        stable_score = 22 if stable else 0
        cross_score = 12 if golden_cross else 0
        vol_score = 8 if vol_ok else 0
        score = min(100, pb_score + stable_score + cross_score + vol_score + nr_bonus + vwap_bonus)

        extra = {
            "pullback_pct": round(pb_pct_val, 2),
            "stable": stable,
            "golden_cross": golden_cross,
            "vol_surge": vol_ok,
            "nr_ok": nr_ok,
            "vwap_ok": vwap_ok,
        }
        detail = f"{'回调' if direction=='bull' else '反弹'}{pb_pct_val:.1f}%({'企稳' if stable else ''} {'金叉' if golden_cross else ''} {'放量' if vol_ok else ''} {'NR' if nr_ok else ''} {'VWAP' if vwap_ok else ''})"
        return True, score, detail, extra

    # ══════════════════════════════════════════════════
    # 风控调整
    # ══════════════════════════════════════════════════

    def _apply_risk_adjustments(
        self, score: float, inst_id: str, ed: Dict,
        direction: str, atr_pct: float, lp: float,
        details: Dict[str, str], h1: pd.DataFrame = None
    ) -> Tuple[float, Dict]:
        """综合风控评分调整：费率 + BTC + OI确认 + 仓位 + 止损 + 追踪"""
        funding = float(ed.get("funding_rate", 0) or 0)

        # ── 资金费率 ──
        funding = float(ed.get("funding_rate", 0) or 0)
        ext_pct = self.config["funding_extreme_pct"] / 100.0
        if abs(funding) > ext_pct:
            penalty = self.config["funding_penalty"]
            if funding > ext_pct and direction == "bull":
                score -= penalty
                details["资金费率"] = f"⚠ 多头拥挤({funding*100:.2f}%) -{penalty}"
            elif funding < -ext_pct and direction == "bear":
                score -= penalty
                details["资金费率"] = f"⚠ 空头拥挤({funding*100:.2f}%) -{penalty}"
            else:
                details["资金费率"] = f"极端({funding*100:.2f}%)方向匹配"

        # ── BTC 环境（从 extra_data 提取，若无则从 1H 价格变化推断）─────
        btc_1h = 0.0
        btc_ctx = ed.get("btc_context", {}) or {}
        btc_raw = btc_ctx.get("btc_1h_pct")
        if btc_raw is not None:
            try: btc_1h = float(btc_raw)
            except Exception: pass
        # 从 h1 中也可以推算 BTC 情绪：如果价格明显下跌但 Layer1 通过（说明是山寨独立行情）
        if btc_1h == 0.0 and len(h1) >= 2:
            try: btc_1h = (float(h1["c"].iloc[-1]) / float(h1["c"].iloc[-2]) - 1) * 100
            except Exception: pass
        if btc_1h < self.config["btc_dump_threshold_pct"] and direction == "bull":
            penalty = self.config["btc_dump_penalty"]
            score -= penalty
            details["BTC环境"] = f"⚠ BTC跌{btc_1h:.1f}%<{self.config['btc_dump_threshold_pct']}% -{penalty}"

        # ── OI 确认（价格 + OI 变化组合信号）─────────
        if bool(self.config.get("oi_confirm_enabled", True)):
            oi_4h = float(ed.get("oi_change_4h", 0) or 0)
            oi_24h = float(ed.get("oi_change_24h", 0) or 0)
            oi_pct = oi_4h if abs(oi_4h) > 0 else oi_24h
            if abs(oi_pct) > 0.5:
                weight = self.config["oi_confirm_weight"]
                price_1h_pct = 0.0
                try:
                    if hasattr(h1, 'iloc') and len(h1) >= 2:
                        price_1h_pct = (float(h1["c"].iloc[-1]) / float(h1["c"].iloc[-2]) - 1) * 100
                except Exception: pass
                # Price↑ + OI↑ = 多头加仓 (bullish confirm)
                # Price↑ + OI↓ = 空头回补 (caution)
                if direction == "bull":
                    if price_1h_pct > 0 and oi_pct > 1.0:
                        score += weight; details["OI确认"] = f"✓ OI+{oi_pct:.1f}% 价格+{price_1h_pct:.1f}% 多头加仓"
                    elif price_1h_pct > 0 and oi_pct < -2.0:
                        score -= weight*0.5; details["OI确认"] = f"⚠ OI{oi_pct:.1f}% 价格+{price_1h_pct:.1f}% 疑似空头回补"
                elif direction == "bear":
                    if price_1h_pct < 0 and oi_pct > 1.0:
                        score += weight; details["OI确认"] = f"✓ OI+{oi_pct:.1f}% 价格{price_1h_pct:.1f}% 空头加仓"
                    elif price_1h_pct < 0 and oi_pct < -2.0:
                        score -= weight*0.5; details["OI确认"] = f"⚠ OI{oi_pct:.1f}% 价格{price_1h_pct:.1f}% 疑似多头减仓"

        # ── ATR 自适应仓位 ──
        base_size = self.config["position_size"]
        scale = min(1.0, 3.0 / max(atr_pct, 1.0))
        scaled_pos = min(base_size * scale, self.config["max_position_pct"])
        details["建议仓位"] = f"{scaled_pos*100:.1f}%({atr_pct:.1f}%ATR自适应)"

        # ── ATR 止损 ──
        stop_mult = self.config["stop_atr_mult"]
        stop_pct = atr_pct * stop_mult
        if direction == "bull":
            sl = lp * (1 - stop_pct / 100)
            tp = lp * (1 + stop_pct * self.config["take_profit_atr_mult"] / stop_mult / 100)
        else:
            sl = lp * (1 + stop_pct / 100)
            tp = lp * (1 - stop_pct * self.config["take_profit_atr_mult"] / stop_mult / 100)
        details["ATR止损"] = f"{sl:.6g} (ATR{stop_mult}×={stop_pct:.1f}%)"
        details["ATR止盈"] = f"{tp:.6g} (盈亏比~{self.config['take_profit_atr_mult']/stop_mult:.1f})"

        # ── 追踪止损 ──
        trail_act = atr_pct * self.config["trail_activate_atr_mult"]
        trail_dist = atr_pct * self.config["trail_distance_atr_mult"]
        details["追踪止损"] = f"浮盈>{trail_act:.1f}%激活, 距离{trail_dist:.1f}%"

        return round(max(0, score), 2), details

    # ══════════════════════════════════════════════════
    # 批量扫描（复用 scan_symbol）
    # ══════════════════════════════════════════════════

    def scan_all_symbols(self, symbols: List) -> Dict:
        """批量扫描 — 逐标的调用 scan_symbol 并汇总。"""
        results = []
        for sym in symbols:
            try:
                r = self.scan_symbol(sym)
                if r.get("passed"):
                    results.append(r)
            except Exception:
                continue
        results.sort(key=lambda r: float(r.get("score", 0) or 0), reverse=True)
        top_n = int(self.config.get("top_n", 15) or 15)
        return {
            "type": "volatile_altcoin_atr",
            "all_opportunities": results[:top_n],
            "scanned_symbols": len(symbols),
        }

    # ══════════════════════════════════════════════════
    # 回测信号
    # ══════════════════════════════════════════════════

    def generate_signal(self, data, *args, **kwargs):
        """回测信号生成 — 包装 scan_symbol 兼容回测引擎。"""
        klines_map = data.get("klines_map", {}) or {}
        h1 = _to_df(klines_map.get("1H") or klines_map.get("1h") or data.get("hourly_df", []))
        m15 = _to_df(klines_map.get("15m") or klines_map.get("15M") or [])
        m3 = _to_df(klines_map.get("3m") or klines_map.get("3M") or [])

        class FakeSymbol:
            pass
        sym = FakeSymbol()
        sym.inst_id = str(data.get("inst_id", data.get("symbol", "BACKTEST")))
        sym.last_price = float(data.get("last_price", data.get("close", 0)) or 0)
        sym.volume_24h = float(data.get("volume_24h", 0) or 0)
        sym.extra_data = {"klines": {"1H": h1, "15m": m15, "3m": m3}}

        result = self.scan_symbol(sym)
        if not result.get("passed"):
            return None
        return {
            "action": result.get("direction", "WAIT"),
            "score": result.get("score", 0),
            "reason": result.get("signals", [""])[0] if result.get("signals") else "",
            "details": result.get("details", {}),
            "position_size": result.get("details", {}).get("建议仓位", "5%"),
        }

    def _adx_rising(self, df: pd.DataFrame, period: int, bars: int,
                     adx_cache: Optional[Tuple[float, float, float]] = None) -> bool:
        """检测 ADX 是否连续上升 N 根。
        如果提供了 adx_cache=(adx_val, pdi, mdi)，则复用已计算的 DI 序列，避免双重计算。
        """
        if len(df) < period * 2 + bars:
            return False
        try:
            c, h, l = df["c"], df["h"], df["l"]
            pc = c.shift(1)
            tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
            um, dm = h.diff(), -l.diff()
            pdm = ((um > dm) & (um > 0)).astype(float) * um.clip(lower=0)
            mdm = ((dm > um) & (dm > 0)).astype(float) * dm.clip(lower=0)
            atr_s = tr.ewm(alpha=1/period, adjust=False).mean()
            pdi = 100 * pdm.ewm(alpha=1/period, adjust=False).mean() / atr_s.replace(0, np.nan)
            mdi = 100 * mdm.ewm(alpha=1/period, adjust=False).mean() / atr_s.replace(0, np.nan)
            dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
            adx_vals = dx.ewm(alpha=1/period, adjust=False).mean().tail(bars + 1)
            if len(adx_vals) < bars + 1:
                return False
            return all(adx_vals.iloc[i] > adx_vals.iloc[i - 1] for i in range(1, bars + 1))
        except Exception:
            return False


# ══════════════════════════════════════════════════
# 独立工具函数
# ══════════════════════════════════════════════════

def _calc_adx(df: pd.DataFrame, period: int = 14) -> Tuple[float, float, float]:
    """计算 ADX, +DI, -DI"""
    if len(df) < period * 2 + 1:
        return 0.0, 0.0, 0.0
    c, h, l = df["c"], df["h"], df["l"]
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    um, dm = h.diff(), -l.diff()
    pdm = ((um > dm) & (um > 0)).astype(float) * um.clip(lower=0)
    mdm = ((dm > um) & (dm > 0)).astype(float) * dm.clip(lower=0)
    atr_s = tr.ewm(alpha=1/period, adjust=False).mean()
    pdi = 100 * pdm.ewm(alpha=1/period, adjust=False).mean() / atr_s.replace(0, np.nan)
    mdi = 100 * mdm.ewm(alpha=1/period, adjust=False).mean() / atr_s.replace(0, np.nan)
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    adx = dx.ewm(alpha=1/period, adjust=False).mean()
    v = adx.iloc[-1]
    return (float(v) if pd.notna(v) else 0.0,
            float(pdi.iloc[-1]) if pd.notna(pdi.iloc[-1]) else 0.0,
            float(mdi.iloc[-1]) if pd.notna(mdi.iloc[-1]) else 0.0)


STRATEGY_NAME = "高波动山寨ATR自适应趋势启动"
STRATEGY_TYPE = "scan"
STRATEGY_CLASS = VolatileAltcoinATRTrendScanner
BACKTEST_CLASS = VolatileAltcoinATRTrendScanner
