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

from strategies._shared.indicators import _to_df, _atr, _rsi_wilder, _clamp

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
    "min_score":                {"type": "float", "default": 60.0,         "label": "最低输出分数"},
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

    "mid_band_proximity_atr":   {"type": "float", "default": 0.35,         "label": "中轨接近度(×ATR)"},
    "mid_band_proximity_floor":  {"type": "float", "default": 0.8,         "label": "中轨接近度保底%"},
    "h1_rsi_period":            {"type": "int",   "default": 14,           "label": "1H RSI周期"},
    "h1_rsi_oversold_long":    {"type": "float", "default": 45.0,         "label": "多头RSI超卖区"},
    "h1_rsi_overbought_short":  {"type": "float", "default": 55.0,         "label": "空头RSI超买区"},
    "h1_rsi_gate_enabled":      {"type": "bool",  "default": True,         "label": "RSI硬门槛(不满足则拒绝)"},

    "m15_macd_fast":            {"type": "int",   "default": 12,           "label": "15m MACD快线"},
    "m15_macd_slow":            {"type": "int",   "default": 26,           "label": "15m MACD慢线"},
    "m15_macd_signal":          {"type": "int",   "default": 9,            "label": "15m MACD信号线"},
    "m15_macd_required":        {"type": "bool",  "default": True,         "label": "强制要求MACD金叉/死叉"},
    "m15_macd_cross_lookback":  {"type": "int",   "default": 3,            "label": "15m MACD交叉回看根数"},
    "m15_vol_surge_ratio":      {"type": "float", "default": 1.3,          "label": "15m放量比"},
    "m15_vol_baseline":         {"type": "int",   "default": 10,           "label": "15m量基线根数"},
    "m15_volume_required":      {"type": "bool",  "default": True,         "label": "强制要求15m放量"},
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
}

_DEFAULTS = {k: v["default"] for k, v in CONFIG_SCHEMA.items()}

PRESET_CONFIGS: Dict[str, Dict[str, Any]] = {
    "custom": {},
    "major_conservative": {
        "allow_short": False,
        "min_volume_24h": 20_000_000,
        "min_score": 66.0,
        "adx_min_trend": 18.0,
        "slope_min_angle": 0.08,
        "h1_rsi_oversold_long": 50.0,
        "h1_rsi_overbought_short": 58.0,
        "mid_band_proximity_floor": 1.20,
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
        "h1_rsi_oversold_long": 52.0,
        "h1_rsi_overbought_short": 53.0,
        "mid_band_proximity_floor": 1.35,
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
    required_bars = ["1D", "1H", "15m"]
    name = "小月期货多周期布林趋势转折"
    description = "D1布林定势→1H回踩中轨→15m MACD金叉+放量起爆 | ATR止损+追踪"

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
        big_df = big_df.tail(50) if len(big_df) > 50 else big_df
        h1 = h1.tail(120) if len(h1) > 120 else h1
        m15 = m15.tail(80) if len(m15) > 80 else m15

        fail = lambda r: {"symbol":inst_id,"passed":False,"score":0,"direction":"WAIT","category":"小月期货","details":{"状态":r}}
        min_big = 22 if big_label == "D1" else 40
        if lp <= 0: return fail("缺少最新价")
        if len(big_df) < min_big or len(h1) < 25 or len(m15) < 30:
            return fail(f"数据不足(大周期{len(big_df)}/{min_big},1H{len(h1)}/25,15m{len(m15)}/30)")

        d1_ok, d1_dir, d1_sc, d1_det, atr_pct = self._step1_daily_trend(big_df, big_label)
        if not d1_ok: return fail(f"{big_label}: {d1_det}")
        h1_ok, h1_sc, h1_det, h1_atr = self._step2_h1_pullback(h1, d1_dir)
        if not h1_ok: return fail(f"1H: {h1_det}")
        if h1_atr > atr_pct: atr_pct = h1_atr  # 取更保守的 ATR
        m15_ok, m15_sc, m15_det = self._step3_m15_entry(m15, d1_dir)
        if not m15_ok: return fail(f"15m: {m15_det}")

        score = round(_clamp(d1_sc * 0.30 + h1_sc * 0.30 + m15_sc * 0.40, 0, 100), 2)
        score, details, risk_cfg = self._apply_risk(score, inst_id, ed, d1_dir, atr_pct, lp, d1_det, h1_det, m15_det)
        if score < self.config["min_score"]: return fail(f"评分不足({score:.1f})")

        return {
            "symbol": inst_id, "passed": True, "score": round(score, 2),
            "opportunity_score": round(score, 2),
            "direction": "BUY" if d1_dir == "bull" else "SELL",
            "category": "小月期货布林趋势转折",
            "signals": [f"{'多头' if d1_dir=='bull' else '空头'} {score:.1f}分",
                        f"D1布林{'上' if d1_dir=='bull' else '下'}轨→1H回踩→15m起爆"],
            "last_price": lp,
            "volume_24h": float(getattr(symbol,"volume_24h",0) or 0),
            "details": details,
            "stop_loss_pct": risk_cfg["stop_loss_pct"],
            "take_profit_pct": risk_cfg["take_profit_pct"],
            "ranking_factors": {"trend": d1_sc, "trigger": m15_sc, "volume": m15_sc*0.8,
                                "location": h1_sc, "freshness": 50, "risk": _clamp(100-atr_pct*5,0,100)},
        }

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

        # ADX 趋势确认
        try:
            c2, h2, l2 = df["c"], df["h"], df["l"]
            pc = c2.shift(1)
            tr = pd.concat([h2-l2, (h2-pc).abs(), (l2-pc).abs()], axis=1).max(axis=1)
            um, dm = h2.diff(), -l2.diff()
            pdm = ((um>dm)&(um>0)).astype(float)*um.clip(lower=0)
            mdm = ((dm>um)&(dm>0)).astype(float)*dm.clip(lower=0)
            ap = int(self.config["adx_period"])
            atr_s = tr.ewm(alpha=1/ap, adjust=False).mean()
            pdi = 100*pdm.ewm(alpha=1/ap, adjust=False).mean()/atr_s.replace(0,np.nan)
            mdi = 100*mdm.ewm(alpha=1/ap, adjust=False).mean()/atr_s.replace(0,np.nan)
            dx = 100*(pdi-mdi).abs()/(pdi+mdi).replace(0,np.nan)
            adx = float(dx.ewm(alpha=1/ap, adjust=False).mean().iloc[-1]) if pd.notna(dx.iloc[-1]) else 0
        except Exception: adx = 15.0
        adx_min = float(self.config["adx_min_trend"])
        if adx < adx_min and label != "4H":
            return False, "neutral", 0, f"ADX={adx:.1f}<{adx_min}趋势弱", atr_pct

        # BBW 收缩检测
        bbw_series = (b["boll_up"] - b["boll_down"]) / b["boll_mid"] * 100
        cur_bbw = float(bbw_series.iloc[-1])
        avg_bbw = float(bbw_series.iloc[-sl_lb:].mean())
        squeeze = cur_bbw < avg_bbw * 0.92  # 带宽低于近期均值=收缩中

        # 评分
        score = _clamp(abs(mid_slope) / max(eff_min_angle * 4, 0.5), 0, 1) * 50
        recent_ok = all((float(b["c"].iloc[-(sl_lb-i)]) > float(b["boll_mid"].iloc[-(sl_lb-i)]))
                        if direction == "bull" else
                        (float(b["c"].iloc[-(sl_lb-i)]) < float(b["boll_mid"].iloc[-(sl_lb-i)]))
                        for i in range(min(sl_lb, len(b)-1)))
        if recent_ok: score += 15
        if squeeze: score += self.config["bbw_squeeze_bonus"]
        detail = f"{label}{cn} 斜率{mid_slope:.1f}° ADX={adx:.1f}{' 收缩' if squeeze else ''} {'持续' if recent_ok else ''}"
        return True, direction, min(100, score), detail, atr_pct

    def _step2_h1_pullback(self, df: pd.DataFrame, direction: str) -> Tuple[bool, float, str, float]:
        period = int(self.config["boll_period"]); std_m = float(self.config["boll_std_mult"])
        prox_a = float(self.config["mid_band_proximity_atr"]); prox_f = float(self.config["mid_band_proximity_floor"])
        b = _bollinger(df, period, std_m); last = b.iloc[-1]
        mid = float(last["boll_mid"]); close = float(last["c"])
        if mid <= 0: return False, 0, "中轨无效", 2.0
        atr_pct = float(_atr(df, 14) / close * 100) if close > 0 else 2.0
        dist_pct = abs(close - mid) / mid * 100; max_d = max(prox_a * atr_pct, prox_f)
        if dist_pct >= max_d: return False, 0, f"距中轨{dist_pct:.1f}%>{max_d:.1f}%", atr_pct
        rsi = _rsi_wilder(df["c"], int(self.config["h1_rsi_period"]))
        rsi_ok = (rsi < self.config["h1_rsi_oversold_long"]) if direction == "bull" else (rsi > self.config["h1_rsi_overbought_short"])
        # RSI 硬门槛
        if bool(self.config.get("h1_rsi_gate_enabled", True)) and not rsi_ok:
            return False, 0, f"RSI={rsi:.1f} 不在回踩区({'超卖' if direction=='bull' else '超买'})", atr_pct
        score = _clamp(1 - dist_pct / max(max_d, 1e-9), 0, 1) * 60
        if rsi_ok: score += 20
        if (direction == "bull" and close > mid) or (direction == "bear" and close < mid): score += 15
        return True, min(100, score), f"距中轨{dist_pct:.1f}% RSI={rsi:.1f}", atr_pct

    def _step3_m15_entry(self, df: pd.DataFrame, direction: str) -> Tuple[bool, float, str]:
        period = int(self.config["boll_period"]); std_m = float(self.config["boll_std_mult"])
        mf = int(self.config["m15_macd_fast"]); ms = int(self.config["m15_macd_slow"])
        msig = int(self.config["m15_macd_signal"])
        b = _bollinger(df, period, std_m); last = b.iloc[-1]
        mid_last, close_last = float(last["boll_mid"]), float(last["c"])
        price_sig = close_last > mid_last if direction == "bull" else close_last < mid_last
        ef = df["c"].ewm(span=mf, adjust=False).mean(); es = df["c"].ewm(span=ms, adjust=False).mean()
        ml = ef - es; msl = ml.ewm(span=msig, adjust=False).mean()
        mc = float(ml.iloc[-1]); mp = float(ml.iloc[-2]) if len(ml) >= 2 else mc
        sc = float(msl.iloc[-1]); sp = float(msl.iloc[-2]) if len(msl) >= 2 else sc
        macd_c_now = (mc > sc and mp <= sp) if direction == "bull" else (mc < sc and mp >= sp)
        lookback = max(1, int(self.config.get("m15_macd_cross_lookback", 3) or 3))
        macd_c_recent = False
        if len(ml) >= 2:
            max_lb = min(lookback, len(ml) - 1)
            for offset in range(max_lb):
                cur_idx = len(ml) - 1 - offset
                prev_idx = cur_idx - 1
                cur_main = float(ml.iloc[cur_idx]); cur_signal = float(msl.iloc[cur_idx])
                prev_main = float(ml.iloc[prev_idx]); prev_signal = float(msl.iloc[prev_idx])
                if direction == "bull":
                    if cur_main > cur_signal and prev_main <= prev_signal:
                        macd_c_recent = True
                        break
                else:
                    if cur_main < cur_signal and prev_main >= prev_signal:
                        macd_c_recent = True
                        break
        hist_now = float((ml - msl).iloc[-1])
        hist_confirm = hist_now > 0 if direction == "bull" else hist_now < 0
        macd_c = macd_c_now or (macd_c_recent and hist_confirm)
        vr = float(self.config["m15_vol_surge_ratio"]); vb = int(self.config["m15_vol_baseline"])
        cv = float(df["vol"].iloc[-1]); bv = float(df["vol"].iloc[-vb-1:-1].mean()) if len(df) >= vb+2 else cv
        vol_ok = cv > bv * vr if bv > 0 else True
        req_macd = bool(self.config.get("m15_macd_required", True))
        req_price = bool(self.config.get("m15_price_required", True))
        req_volume = bool(self.config.get("m15_volume_required", True))
        # 强制检查：根据配置决定是否需要 MACD + 价格
        if req_price and req_macd:
            if not price_sig or not macd_c:
                return False, 0, f"需价格站稳中轨+MACD交叉({'✓' if price_sig else '✗'}/{'✓' if macd_c else '✗'})"
        elif req_price and not req_macd:
            if not price_sig: return False, 0, "价格未站稳中轨"
        elif req_macd and not req_price:
            if not macd_c: return False, 0, "MACD未交叉"
        else:
            if not price_sig and not macd_c: return False, 0, "无起爆信号"
        if req_volume and not vol_ok:
            return False, 0, f"15m未放量(cv={cv:.0f} < {vr:.2f}×{bv:.0f})"
        score = 0
        if price_sig: score += 30
        if macd_c: score += 35
        if vol_ok: score += 20
        # hist_strength 改为价格百分比（而非绝对 magic number）
        mh = ml - msl
        hist_pct = abs(float(mh.iloc[-1])) / max(mid_last, 1e-9) * 100
        score += _clamp(hist_pct / 0.5, 0, 1) * 12  # hist占中轨0.5%以上满分
        ct = "金叉" if direction == "bull" else "死叉"
        macd_note = "当根交叉" if macd_c_now else ("近期交叉延续" if macd_c else "未交叉")
        return True, min(100, score), f"价格{'✓' if price_sig else '✗'} MACD{'✓' if macd_c else '✗'}({macd_note}){ct} 放量{'✓' if vol_ok else '✗'}"

    def _apply_risk(self, score, inst_id, ed, direction, atr_pct, lp, bd, hd, md):
        d = {"大周期": bd, "1H回踩": hd, "15m起爆": md, "ATR%": f"{atr_pct:.1f}%"}
        f_ = float(ed.get("funding_rate", 0) or 0); ep = self.config["funding_extreme_pct"] / 100.0
        if abs(f_) > ep and ((f_ > ep and direction == "bull") or (f_ < -ep and direction == "bear")):
            score -= self.config["funding_penalty"]; d["资金费率"] = f"⚠ 拥挤({f_*100:.2f}%)"
        bc = float(ed.get("btc_1h_pct", ed.get("btc_context", {}).get("btc_1h_pct", 0)) or 0) if isinstance(ed, dict) else 0
        if bc < self.config["btc_dump_threshold_pct"] and direction == "bull":
            score -= self.config["btc_dump_penalty"]; d["BTC环境"] = f"⚠ BTC跌{bc:.1f}%"
        if bc > float(self.config.get("btc_pump_threshold_pct", 3.0)) and direction == "bear":
            score -= self.config["btc_dump_penalty"]; d["BTC环境"] = f"⚠ BTC涨{bc:.1f}%"
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
        d["ATR止损"] = f"{sl:.6g}(ATR{sm}×={raw_sp:.1f}%→钳制{sp:.1f}%)"
        d["ATR止盈"] = f"{tp:.6g}(盈亏比{tm/sm:.1f})"
        ta = atr_pct * self.config["trail_activate_atr_mult"]
        td_ = atr_pct * self.config["trail_distance_atr_mult"]
        d["追踪止损"] = f"浮盈>{ta:.1f}%激活 距离{td_:.1f}%"
        bs = self.config["position_size"]; sc_ = min(1.0, 3.0 / max(atr_pct, 1.0))
        d["建议仓位"] = f"{min(bs * sc_, 0.10)*100:.1f}%"
        risk_cfg = {
            "stop_loss_pct": round(float(sp), 4),
            "take_profit_pct": round(float(tp_pct), 4),
        }
        return round(max(0, score), 2), d, risk_cfg

    def scan_all_symbols(self, symbols):
        res = []; 
        for s in symbols:
            try:
                r = self.scan_symbol(s)
                if r.get("passed"): res.append(r)
            except Exception as e:
                logger.debug(f"[小月布林] {getattr(s,'inst_id','')} 扫描异常: {e}")
                continue
        res.sort(key=lambda r: float(r.get("score", 0) or 0), reverse=True)
        return {"type": "xiaoyue", "all_opportunities": res[:int(self.config.get("top_n", 12))], "scanned_symbols": len(symbols)}

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
                    elif isinstance(v, list):
                        km[k] = v[-cap:]

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
