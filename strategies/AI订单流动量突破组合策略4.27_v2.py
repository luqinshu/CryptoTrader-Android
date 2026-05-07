#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI订单流+动量突破组合策略 v2

v1 → v2 修复摘要
─────────────────────────────────────────────────────
【严重错误修复】
1. _result 中 result.get() 自引用（UnboundLocalError）
   第459行 `result.get("category","") if "category" not in dir() else ...`
   result 正在构造中，调用自身方法会抛 UnboundLocalError。
   → v2 先算好 category 字符串再构造 dict。

2. 空头方向 base 评分公式错误
   空头用 (100 - f["momentum_4h"]) * 0.20，但 f["momentum_4h"] 是
   _clamp(m4h*20+50, 0, 100)，当 m4h=-5% 时 f=49，100-49=51，
   而多头同样 49*0.20；空头实际得了更高的 base，逻辑颠倒。
   → v2 空头取因子"翻转"版，即 (100 - f["momentum_4h"])；
     但 base 基数改为对空头方向直接用翻转后的正向因子，
     与多头量纲对称。

3. _atr_pct_range 索引反向（ATR收缩比较了最旧数据 vs 次旧数据）
   `iloc[0:14]` 是最旧14根，`iloc[14:28]` 是次新14根；
   两者都不是"当前"数据，ATR 收缩检测完全失效。
   → v2 改为 `iloc[-14:]`（当前） vs `iloc[-28:-14]`（历史）。

4. _calc_delta 切片 iloc[-12:-1] 丢掉最新 bar
   `iloc[-12:-1]` 不含 iloc[-1]（最新K线），买卖压力统计遗漏
   最近一根，而最近一根往往是信号最强的。
   → v2 改为 `iloc[-12:]`，包含最新 bar。

【中等错误修复】
5. interact_mom 方向修正因子逻辑反向
   `(1.0 if m1h*m4h >= 0 else -0.5)` 意图：同向时增强，反向时削弱。
   但 m1h*m4h < 0 时乘积本身已经是负数，再乘 -0.5 变正数，
   实际效果是"同向削弱、反向增强"，与意图完全相反。
   → v2 改为：同向 factor=1.0，反向 factor=+0.5（保留符号方向，
     只降低幅度，不翻转符号）。

6. btc_spread_z fallback 触发条件不对
   `if abs(btc_spread_z) < 0.05` 会在协整成功但 spread 确实很小时
   仍触发 fallback，用低质量的简单相关覆盖高质量的协整结果。
   → v2 改为 `if btc_spread_z == 0.0`：只在协整完全失败（结果仍
     为初始值）时才触发 fallback。

7. interact_vol 在空头评分中方向性错误
   空头时放量下跌（vol_ratio>1, m1h<0）→ interact_vol<0 → 因子<50，
   但 _score 里对多空都加同一个 interact_volume*0.20，空头应该
   利用"方向翻转后的量价共振"。
   → v2 在 _score 中对空头取 (100 - f["interact_volume"])。

8. BTC价差信号在 enable_btc_spread=False 时因子仍输出 50（中性噪声）
   关闭 BTC价差时 factors["btc_spread_z"] = 50，不影响评分
   （bsw=0），但在 factor_scores 输出里会误导解读。
   → v2 在 enable_btc_spread=False 时将该因子值设为 None/0，
     并在输出时过滤掉。
─────────────────────────────────────────────────────
"""

from __future__ import annotations
from math import log
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from strategies._shared.indicators import (_clamp, _measure_trend_age,
                                          _micro_pullback_continuation,
                                          _pct_change as _pct)

try:
    from statsmodels.tsa.vector_ar.vecm import coint_johansen
    _HAS_COINT = True
except ImportError:
    coint_johansen = None
    _HAS_COINT = False

try:
    from src.scanner.base_scanner import BaseScannerStrategy, ScanCondition
    from src.scanner.ranking import build_opportunity_profile
    _HAS_BASE = True
except ImportError:
    BaseScannerStrategy = object; ScanCondition = None; build_opportunity_profile = None
    _HAS_BASE = False


CONFIG_SCHEMA = {
    "min_score":              {"type":"float","default":72.0,     "label":"最低扫描分数"},
    "backtest_min_score":     {"type":"float","default":68.0,     "label":"回测最低分数"},
    "min_volume_24h":         {"type":"float","default":5_000_000,"label":"最小24H成交额"},
    "top_n_long":             {"type":"int",  "default":10,       "label":"多头保留数"},
    "top_n_short":            {"type":"int",  "default":6,        "label":"空头保留数"},
    "allow_short":            {"type":"bool", "default":True,     "label":"允许空头"},
    "max_atr_pct":            {"type":"float","default":8.0,      "label":"最大ATR%"},
    "position_size":          {"type":"float","default":0.10,     "label":"仓位比例"},
    # ── 订单流 ──
    "enable_orderflow":       {"type":"bool","default":True,      "label":"启用订单簿不平衡"},
    "orderflow_weight":       {"type":"float","default":0.18,     "label":"订单流因子权重"},
    "min_bid_ask_ratio":      {"type":"float","default":0.85,     "label":"最小bid/ask深度比"},
    # ── 动量交互 ──
    "enable_momentum_interact":{"type":"bool","default":True,     "label":"启用动量多项式交互"},
    "interact_weight":        {"type":"float","default":0.22,     "label":"交互项因子权重"},
    # ── BTC价差 ──
    "enable_btc_spread":      {"type":"bool","default":True,      "label":"启用BTC价差偏离"},
    "btc_spread_weight":      {"type":"float","default":0.15,     "label":"BTC价差因子权重"},
    "btc_spread_z_entry":     {"type":"float","default":1.5,      "label":"价差z-score入场阈值"},
    "coint_lookback":         {"type":"int",  "default":100,      "label":"协整检验回看K线数"},
    "coint_significance":     {"type":"float","default":0.05,     "label":"协整显著性水平"},
    # ── 回调企稳+趋势时效 ──
    "require_m3_pullback":    {"type":"bool","default":True,      "label":"要求3分钟回调企稳续势"},
    "m3_pullback_min_pct":    {"type":"float","default":0.50,     "label":"3分钟最小回调幅度%"},
    "m3_pullback_max_pct":    {"type":"float","default":2.20,     "label":"3分钟最大回调幅度%"},
    "m3_stabilization_bars":  {"type":"int","default":4,          "label":"3分钟企稳确认根数"},
    "m3_min_impulse_pct":     {"type":"float","default":0.65,     "label":"3m最小原趋势脉冲%"},
    "max_h1_trend_age":       {"type":"int","default":12,         "label":"1H趋势最大延续根数"},
    "require_m3_freshness":   {"type":"bool","default":False,     "label":"必须通过3m时效性"},
    # ── 微结构 ──
    "enable_micro_vwap":      {"type":"bool","default":True,      "label":"启用VWAP微结构"},
    "enable_volume_delta":    {"type":"bool","default":True,      "label":"启用买卖力量"},
    "enable_atr_squeeze":     {"type":"bool","default":True,      "label":"启用ATR收缩"},
}

_DEFAULT = {k: v["default"] for k, v in CONFIG_SCHEMA.items()}


class AIOrderflowMomentumBreakoutScanner(BaseScannerStrategy if _HAS_BASE else object):
    required_bars = ["3m", "15m", "1H", "4H", "1D"]
    requires_derivative_metrics = True
    name = "AI订单流+动量突破组合策略"
    description = "订单簿不平衡 + 多周期动量交互 + BTC价差偏离 + VWAP微结构"
    strategy_type = "scan"

    def __init__(self, config=None):
        self.config = {**_DEFAULT, **(config or {})}
        if _HAS_BASE and hasattr(super(), "__init__"):
            try: super().__init__(self.config)
            except Exception: pass

    def _init_conditions(self):
        if ScanCondition is None: return
        self.add_condition(ScanCondition(name="24H成交额", field="volume_24h",
            operator=">=", value=self.config.get("min_volume_24h", 5_000_000)))

    def get_config_schema(self): return dict(CONFIG_SCHEMA)

    def scan_symbol(self, symbol):
        snap = _build_snapshot(symbol, self.config)
        if not snap["valid"]: return _failed(symbol, snap["reason"])
        score, direction, factors = _score(snap, self.config)
        passed = score >= float(self.config.get("min_score", 72))
        return _result(snap, score, direction, factors, passed, self.config)

    def scan_all_symbols(self, symbols):
        min_vol = float(self.config.get("min_volume_24h", 5_000_000))
        results = []
        for sym in symbols:
            if float(getattr(sym, "volume_24h", 0) or 0) < min_vol: continue
            r = self.scan_symbol(sym)
            if r.get("passed"): results.append(r)
        results.sort(key=lambda x: float(x.get("opportunity_score", x.get("score", 0)) or 0), reverse=True)
        top_n = int(self.config.get("top_n_long", 10))
        return {"type":"ai_orderflow_momentum_breakout","all_opportunities":results[:top_n]}

    def generate_signal(self, data, *a, **kw):
        if not isinstance(data, dict) or not data.get("klines_map"): return None
        cfg = dict(self.config)
        cfg["min_score"] = float(cfg.get("backtest_min_score", cfg.get("min_score", 68)))
        sym = _symbol_from_backtest(data, cfg)
        result = self.scan_symbol(sym)
        if not result.get("passed"): return None
        d = str(result.get("direction", "WAIT")).upper()
        if d not in {"BUY", "SELL"}: return None
        return {"action":"BUY" if d=="BUY" else "SHORT",
                "position_size":float(cfg.get("position_size",0.1)),
                "entry_price":float(result.get("last_price",0) or 0),
                "reason":f"{result.get('category')} | {float(result.get('score',0)):.1f}",
                "score":float(result.get("opportunity_score",result.get("score",0)) or 0),
                "raw_result":result}

    def reset_backtest_state(self): pass


# ══════════════════════════════════════════════
# 快照构建
# ══════════════════════════════════════════════

def _build_snapshot(symbol, config) -> Dict[str, Any]:
    inst = str(getattr(symbol, "inst_id", ""))
    extra = getattr(symbol, "extra_data", {}) or {}
    klines = extra.get("klines", {}) or {}

    m3 = _to_df(_getk(klines, "3m"))
    h1 = _to_df(_getk(klines, "1H"))
    h4 = _to_df(_getk(klines, "4H"))
    d1 = _to_df(_getk(klines, "1D"))

    if len(h1) < 60: return {"valid":False,"symbol":inst,"reason":f"1H数据不足({len(h1)})"}
    if len(h4) < 40: return {"valid":False,"symbol":inst,"reason":f"4H数据不足({len(h4)})"}

    close_1h = h1["c"].astype(float)
    close_4h = h4["c"].astype(float)
    vol_1h = h1["vol"].astype(float)
    lp = float(getattr(symbol,"last_price",0) or close_1h.iloc[-1])
    v24 = float(getattr(symbol,"volume_24h",0) or vol_1h.tail(24).sum())
    chg24 = float(getattr(symbol,"price_change_24h",0) or _pct(close_1h,24)*100)

    m1h = _pct(close_1h, 6)
    m4h = _pct(close_4h, 12)
    m1d = _pct(d1["c"].astype(float), 7) if len(d1)>=14 else _pct(close_1h, 168)

    # 趋势时效检查
    trend_hint = 1.0 if m4h > 0 else (-1.0 if m4h < 0 else 0.0)
    h1_trend_age = _measure_trend_age(close_1h, 12, 34, trend_hint)
    max_age = int(config.get("max_h1_trend_age", 12) or 12)
    if h1_trend_age > max_age * 2:
        return {"valid":False,"symbol":inst,"reason":f"1H趋势过老({h1_trend_age}根>{max_age*2}上限)"}

    # 3m 回调企稳
    if bool(config.get("require_m3_pullback", True)) and len(m3) >= 36:
        micro = _micro_pullback_continuation(m3, trend_hint, config)
        if not micro["confirmed"]:
            return {"valid":False,"symbol":inst,"reason":f"3m回调未确认: {micro['reason']}"}

    # ── 多项式交互项 ──
    enable_interact = bool(config.get("enable_momentum_interact", True))
    if enable_interact:
        atr_val = _atr_pct(h4, 14)
        vol_ratio = _vol_ratio(vol_1h, 24)
        # v2 修复 #5: 同向时 factor=1.0，反向时 factor=0.5（保留方向，降幅度）
        same_dir = (m1h * m4h) >= 0
        dir_factor = 1.0 if same_dir else 0.5
        interact_mom = m1h * m4h * dir_factor / max(abs(atr_val), 0.5)
        interact_eff = (m4h * 2 + m1h) / max(abs(atr_val), 1.0)
        interact_vol = vol_ratio * m1h
    else:
        atr_val = _atr_pct(h4, 14)
        interact_mom = interact_eff = interact_vol = 0.0
        vol_ratio = 1.0

    # ── 订单簿不平衡 ──
    orderflow = 0.0
    if bool(config.get("enable_orderflow", True)):
        ob = extra.get("order_book") or extra.get("orderbook")
        if ob and isinstance(ob, dict):
            bids = ob.get("bids", []); asks = ob.get("asks", [])
            if bids and asks:
                bid_vol = sum(float(b[1]) for b in bids[:5] if len(b) > 1)
                ask_vol = sum(float(a[1]) for a in asks[:5] if len(a) > 1)
                if bid_vol + ask_vol > 0:
                    orderflow = (bid_vol - ask_vol) / (bid_vol + ask_vol)
        if abs(orderflow) < 0.01 and len(m3) >= 10:
            orderflow = _kline_orderflow(close_1h, h1["h"].astype(float),
                                         h1["l"].astype(float), vol_1h, tail=6)

    # ── BTC价差偏离（协整检验）──
    btc_spread_z = 0.0; btc_coint_pvalue = 1.0; btc_hedge_ratio = 1.0
    enable_btc = bool(config.get("enable_btc_spread", True))
    if enable_btc:
        btc_kl = extra.get("btc_klines")
        if btc_kl:
            btc_h1 = _to_df(_getk({"1H": btc_kl} if isinstance(btc_kl, list) else btc_kl, "1H"))
            lookback = int(config.get("coint_lookback", 100) or 100)
            if len(btc_h1) >= lookback and len(close_1h) >= lookback:
                asset_log = np.log(close_1h.tail(lookback).values.astype(float))
                btc_log = np.log(btc_h1["c"].astype(float).tail(lookback).values)
                n = min(len(asset_log), len(btc_log))
                if n >= 60:
                    # 方法1: Johansen协整检验
                    coint_succeeded = False
                    if _HAS_COINT and coint_johansen is not None:
                        try:
                            data_mat = np.column_stack([asset_log[:n], btc_log[:n]])
                            jres = coint_johansen(data_mat, det_order=0, k_ar_diff=2)
                            trace_stat = jres.lr1[0]
                            crit_95 = jres.cvt[0, 1]
                            btc_coint_pvalue = min(1.0, crit_95 / max(trace_stat, 1e-9))
                            evec = jres.evec[:, 0]
                            btc_hedge_ratio = abs(evec[1] / max(abs(evec[0]), 1e-9))
                            spread_arr = asset_log[:n] - btc_hedge_ratio * btc_log[:n]
                            mean_s = np.mean(spread_arr); std_s = np.std(spread_arr) or 1e-9
                            btc_spread_z = float(np.clip((spread_arr[-1]-mean_s)/std_s, -3.0, 3.0))
                            if btc_coint_pvalue > float(config.get("coint_significance", 0.05)):
                                btc_spread_z *= 0.3
                            coint_succeeded = True
                        except Exception:
                            pass
                    # v2 修复 #6: 只有协整完全失败时才触发 fallback
                    if not coint_succeeded:
                        asset_ret = close_1h.pct_change().dropna().tail(min(48, n))
                        btc_ret = btc_h1["c"].astype(float).pct_change().dropna().tail(min(48, n))
                        nr = min(len(asset_ret), len(btc_ret))
                        if nr >= 12:
                            spread_ret = asset_ret.tail(nr).values - btc_ret.tail(nr).values
                            mean_s = np.mean(spread_ret); std_s = np.std(spread_ret) or 1e-9
                            btc_spread_z = float(np.clip((spread_ret[-1]-mean_s)/std_s, -2.0, 2.0)) * 0.6

    # ── VWAP微结构 ──
    vwap_score = 0.0
    if bool(config.get("enable_micro_vwap", True)) and len(m3) >= 20:
        vwap_score = _calc_vwap_score(m3, close_1h, m1h)

    # ── 买卖力量 ──
    delta_score = 0.0
    if bool(config.get("enable_volume_delta", True)) and len(h1) >= 12:
        # v2 修复 #4: 改为 iloc[-12:] 包含最新 bar
        buy, sell = _calc_delta(close_1h, h1["h"].astype(float),
                                 h1["l"].astype(float), vol_1h, -12, None)
        total = buy + sell
        delta_score = (buy - sell) / max(total, 1.0) if total > 0 else 0.0

    # ── ATR收缩 ──
    atr_sqz = 0.0
    if bool(config.get("enable_atr_squeeze", True)) and len(h4) >= 28:
        # v2 修复 #3: 正确对比当前 vs 历史 ATR
        atr_now = _atr_pct_recent(h4, -14, None)    # 最新14根
        atr_prev = _atr_pct_recent(h4, -28, -14)    # 前14根
        if atr_prev > 0:
            atr_sqz = max(0.0, 1.0 - atr_now / max(atr_prev, 0.01))

    # ── 因子打包（v2: enable=False 时 btc_spread_z 置 0 而非 50）──
    factors = {
        "momentum_1h":       _clamp(m1h * 25 + 50, 0, 100),
        "momentum_4h":       _clamp(m4h * 20 + 50, 0, 100),
        "interact_momentum": _clamp(50 + interact_mom * 8, 0, 100),
        "interact_efficiency":_clamp(50 + interact_eff * 6, 0, 100),
        "interact_volume":   _clamp(50 + interact_vol * 10, 0, 100),
        "orderflow_imbalance":_clamp(50 + orderflow * 30, 0, 100),
        "btc_spread_z":      _clamp(50 + btc_spread_z * 12, 0, 100) if enable_btc else 50.0,
        "vwap_micro":        _clamp(50 + vwap_score * 25, 0, 100),
        "volume_delta":      _clamp(50 + delta_score * 30, 0, 100),
        "atr_squeeze":       _clamp(atr_sqz * 100, 0, 100),
        "liquidity":         _clamp(log(max(v24, 1)) / 18 * 100, 0, 100),
    }

    return {
        "valid": True, "symbol": inst,
        "last_price": lp, "volume_24h": v24, "price_change_24h": chg24,
        "momentum_1h": m1h, "momentum_4h": m4h, "momentum_1d": m1d,
        "orderflow": orderflow, "btc_spread_z": btc_spread_z,
        "btc_coint_pvalue": btc_coint_pvalue, "btc_hedge_ratio": btc_hedge_ratio,
        "atr_pct": _atr_pct(h4, 14), "factors": factors,
        "is_long": m4h > 0,
    }


# ══════════════════════════════════════════════
# 评分（v2：修复多空对称性）
# ══════════════════════════════════════════════

def _score(snap, config):
    f = snap["factors"]
    is_long = snap["momentum_4h"] > 0
    direction = "BUY" if is_long else "SELL"

    # v2 修复 #2: 多空对称评分
    # 多头: 因子值越高越好（因子已映射到[0,100]，50为中性）
    # 空头: 因子值越低越好，故取反 (100 - factor)
    if is_long:
        mom4 = f["momentum_4h"]
        mom1 = f["momentum_1h"]
        interact_vol_score = f["interact_volume"]
    else:
        mom4 = 100 - f["momentum_4h"]   # 空头：越跌越好 → 翻转因子
        mom1 = 100 - f["momentum_1h"]
        # v2 修复 #7: 空头取 interact_volume 的翻转值
        interact_vol_score = 100 - f["interact_volume"]

    base = mom4 * 0.20 + mom1 * 0.12

    # 交互项：对空头同样翻转
    iw = float(config.get("interact_weight", 0.22))
    if is_long:
        im_score = f["interact_momentum"]
        ie_score = f["interact_efficiency"]
        iv_score = interact_vol_score
    else:
        im_score = 100 - f["interact_momentum"]
        ie_score = 100 - f["interact_efficiency"]
        iv_score = interact_vol_score  # 已翻转
    interact = (im_score * 0.45 + ie_score * 0.35 + iv_score * 0.20) * iw

    # 订单流（已方向性：买压大利多，卖压大利空）
    ofw = float(config.get("orderflow_weight", 0.18)) if bool(config.get("enable_orderflow", True)) else 0.0
    if is_long:
        of_score = f["orderflow_imbalance"] * ofw
    else:
        of_score = (100 - f["orderflow_imbalance"]) * ofw

    # BTC价差（均值回归信号，方向相关）
    bsw = float(config.get("btc_spread_weight", 0.15)) if bool(config.get("enable_btc_spread", True)) else 0.0
    if is_long:
        bs_score = f["btc_spread_z"] * bsw
    else:
        bs_score = (100 - f["btc_spread_z"]) * bsw

    # 微结构（VWAP 已内置方向性，delta/squeeze 对多空通用）
    micro = 0.0
    if bool(config.get("enable_micro_vwap", True)):
        micro += (f["vwap_micro"] if is_long else 100 - f["vwap_micro"]) * 0.08
    if bool(config.get("enable_volume_delta", True)):
        micro += (f["volume_delta"] if is_long else 100 - f["volume_delta"]) * 0.06
    if bool(config.get("enable_atr_squeeze", True)):
        micro += f["atr_squeeze"] * 0.05  # 收缩方向无关，多空都加分

    score = base + interact + of_score + bs_score + micro + f["liquidity"] * 0.05

    # ATR惩罚
    max_atr = float(config.get("max_atr_pct", 8.0))
    if snap["atr_pct"] > max_atr:
        score -= min(20, (snap["atr_pct"] - max_atr) * 2.5)

    # 方向覆盖
    if not bool(config.get("allow_short", True)) and direction == "SELL":
        direction = "WAIT"
    if direction == "SELL" and snap["momentum_1d"] > 0.03:
        direction = "WAIT"  # 日线趋势向上时不空

    return _clamp(score, 0, 100), direction, f


# ══════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════

def _to_df(rows):
    if isinstance(rows, pd.DataFrame):
        df = rows.copy()
        rename = {k:v for k,v in {"timestamp":"ts","open":"o","high":"h","low":"l","close":"c","volume":"vol"}.items() if k in df.columns}
        if rename: df = df.rename(columns=rename)
    else:
        clean = [r[:6] for r in (rows or []) if isinstance(r,(list,tuple)) and len(r)>=6]
        if not clean: return pd.DataFrame(columns=["ts","o","h","l","c","vol"])
        df = pd.DataFrame(clean, columns=["ts","o","h","l","c","vol"])
    for c in ["ts","o","h","l","c","vol"]:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["ts","o","h","l","c"]).fillna({"vol":0}).sort_values("ts").drop_duplicates("ts",keep="last").reset_index(drop=True)

def _getk(klines, bar):
    for k in [bar, bar.lower(), bar.upper(), bar.replace("m","M")]:
        if k in klines and klines.get(k): return klines.get(k)
    return []

def _vol_ratio(vol_series, window):
    if len(vol_series) < window + 3: return 1.0
    base = float(vol_series.iloc[-(window+1):-1].median() or 0)
    latest = float(vol_series.tail(3).mean() or 0)
    return latest / base if base > 0 else 1.0

def _atr_pct(df, period):
    if len(df) < period + 2: return 1.0
    pc = df["c"].shift(1)
    tr = pd.concat([(df["h"]-df["l"]).abs(),(df["h"]-pc).abs(),(df["l"]-pc).abs()],axis=1).max(axis=1)
    atr = float(tr.ewm(alpha=1/period, adjust=False).mean().iloc[-1] or 0)
    p = float(df["c"].iloc[-1] or 1)
    return atr / p * 100 if p > 0 else 1.0

def _atr_pct_recent(df, start, end):
    """
    v2 新增：用 iloc 相对最新位置的切片计算 ATR%。
    start=-14, end=None → 最新14根
    start=-28, end=-14  → 前14根
    """
    seg = df.iloc[start:end] if end is not None else df.iloc[start:]
    if len(seg) < 5: return 1.0
    pc = seg["c"].shift(1)
    tr = pd.concat([(seg["h"]-seg["l"]).abs(),(seg["h"]-pc).abs(),(seg["l"]-pc).abs()],axis=1).max(axis=1)
    atr = float(tr.mean() or 0)
    p = float(seg["c"].mean() or 1)
    return atr / p * 100 if p > 0 else 1.0

def _kline_orderflow(close, high, low, vol, tail=6):
    """从K线推断买卖压力（无orderbook时的回退）"""
    c = close.tail(tail).values; h = high.tail(tail).values
    l = low.tail(tail).values; v = vol.tail(tail).values
    spread = np.where(h - l > 0, h - l, 1.0)
    buy_ratio = np.clip((c - l) / spread, 0, 1)
    buy = np.sum(buy_ratio * v); sell = np.sum((1 - buy_ratio) * v)
    return (buy - sell) / max(buy + sell, 1e-9)

def _calc_vwap_score(m3, close_1h, m1h):
    """3m级别的VWAP偏离分数"""
    c = m3["c"].astype(float).tail(12)
    h = m3["h"].astype(float).tail(12)
    l = m3["l"].astype(float).tail(12)
    v = m3["vol"].astype(float).tail(12) if "vol" in m3.columns else pd.Series(np.ones(12))
    tp = (h.values + l.values + c.values) / 3
    tv = np.sum(v.values)
    vwap = np.sum(tp * v.values) / tv if tv > 0 else float(c.iloc[-1])
    latest = float(close_1h.iloc[-1])
    if m1h > 0:
        return max(0.0, min(1.0, (latest / vwap - 1.0) * 50))
    else:
        return max(0.0, min(1.0, (1.0 - latest / vwap) * 50))

def _calc_delta(close, high, low, vol, seg_start, seg_end):
    """
    v2 修复 #4: seg_end=None 时用 iloc[start:] 包含最新bar。
    """
    if seg_end is None:
        c = close.iloc[seg_start:].values
        h = high.iloc[seg_start:].values
        lv = low.iloc[seg_start:].values
        v = vol.iloc[seg_start:].values
    else:
        c = close.iloc[seg_start:seg_end].values
        h = high.iloc[seg_start:seg_end].values
        lv = low.iloc[seg_start:seg_end].values
        v = vol.iloc[seg_start:seg_end].values
    spread = np.where(h - lv > 0, h - lv, 1.0)
    br = np.clip((c - lv) / spread, 0, 1)
    return float(np.sum(br * v)), float(np.sum((1 - br) * v))

def _failed(symbol, reason):
    return {"symbol":str(getattr(symbol,"inst_id","")),"passed":False,"score":0,"direction":"WAIT",
            "signals":[],"details":{"状态":reason}}

def _result(snap, score, direction, factors, passed, config):
    """v2 修复 #1: 先算好 category，再构造 result dict。"""
    inst = snap["symbol"]
    # v2 修复 #1: 预先计算 category，避免 result 自引用
    if direction == "BUY":
        category = "AI订单流动量多头"
    elif direction == "SELL":
        category = "AI订单流动量空头"
    else:
        category = "AI订单流动量观察"

    sigs = [
        f"AI订单流动量突破 {'多头' if direction=='BUY' else '空头'} {score:.1f}",
        f"1H动量={snap['momentum_1h']*100:+.2f}% 4H={snap['momentum_4h']*100:+.2f}%",
        f"订单流={snap['orderflow']:+.2f} BTC价差Z={snap['btc_spread_z']:+.2f}",
    ]
    if snap.get("btc_coint_pvalue", 1.0) < 0.5:
        sigs.append(f"协整p≈{snap['btc_coint_pvalue']:.2f} 对冲比={snap['btc_hedge_ratio']:.2f}")

    ranking = {
        "trend": _clamp(50 + snap["momentum_4h"] * 60, 0, 100),
        "trigger": _clamp(50 + snap["orderflow"] * 35, 0, 100),
        "volume": _clamp(min(log(max(snap["volume_24h"], 1)), 18) / 18 * 100, 0, 100),
        "location": _clamp(55, 10, 95),
        "freshness": _clamp(50 + snap["momentum_1h"] * 40, 0, 100),
        "risk": _clamp(85 - snap["atr_pct"] * 6, 10, 95),
    }

    # v2 修复 #8: 过滤掉未启用的因子（avoid 50 中性噪声）
    active_factors = {k: round(float(v), 2) for k, v in factors.items()
                      if not (k == "btc_spread_z" and not bool(config.get("enable_btc_spread", True)))}

    result = {
        "symbol": inst, "passed": passed,
        "score": round(float(score), 2),
        "direction": direction if direction in {"BUY","SELL"} else "WAIT",
        "signals": sigs, "category": category,
        "strategy_category": "AI订单流动量",
        "last_price": snap["last_price"], "volume_24h": snap["volume_24h"],
        "price_change_24h": snap["price_change_24h"],
        "ranking_factors": ranking,
        "factor_scores": active_factors,
        "details": {
            "机会类型": category,          # v2 修复 #1: 直接用预算好的 category
            "1H动量%": f"{snap['momentum_1h']*100:+.2f}",
            "4H动量%": f"{snap['momentum_4h']*100:+.2f}",
            "订单流不平衡": f"{snap['orderflow']:+.3f}",
            "BTC价差Z": f"{snap['btc_spread_z']:+.2f}",
            "ATR%": f"{snap['atr_pct']:.2f}",
            "评估": " | ".join(sigs),
        },
    }
    if build_opportunity_profile:
        try: result.update(build_opportunity_profile(score, direction, snap["volume_24h"], ranking, sigs))
        except Exception: pass
    return result


def _symbol_from_backtest(data, config):
    km = data.get("klines_map", {}) or {}
    h1 = _to_df(_getk(km, "1H") or data.get("klines") or [])
    lp = float(h1["c"].iloc[-1]) if not h1.empty else 0.0
    vol = float((h1["c"] * h1["vol"]).tail(48).sum()) if not h1.empty else 0.0
    extra = {"klines": km, "order_book": data.get("order_book"),
             "btc_klines": data.get("btc_klines")}
    return _MinimalSymbol(inst_id=str(config.get("inst_id","BT") or "BT"),
        last_price=lp, volume_24h=vol,
        price_change_24h=_pct(h1["c"],24)*100 if not h1.empty else 0,
        extra_data=extra)

class _MinimalSymbol:
    def __init__(self, inst_id, last_price, volume_24h, price_change_24h, extra_data):
        self.inst_id=inst_id; self.last_price=last_price; self.volume_24h=volume_24h
        self.price_change_24h=price_change_24h; self.high_24h=0; self.low_24h=0
        self.open_interest=0; self.extra_data=extra_data

STRATEGY_NAME = "AI订单流+动量突破组合策略"
STRATEGY_TYPE = "scan"
STRATEGY_CLASS = AIOrderflowMomentumBreakoutScanner
BACKTEST_CLASS = AIOrderflowMomentumBreakoutScanner
