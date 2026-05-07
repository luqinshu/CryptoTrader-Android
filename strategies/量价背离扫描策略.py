#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
量价背离扫描策略 v1.0
=====================
检测价格与成交量之间的背离信号，作为趋势反转的早期预警。

四种背离模式：
  1. 顶背离 — 价格创新高但成交量递减 → SELL
  2. 底背离 — 价格创新低但成交量递增 → BUY（恐慌性抛售接近尾声）
  3. 量能高潮 — 单根K线成交量极端放大但价格未突破区间 → 反转预警
  4. 缩量反弹 — 价格连续上涨但量能持续萎缩 → 假突破

检测窗口：基于 1H 和 4H K线数据
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

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
    "min_volume_24h":           {"type": "float", "default": 2_000_000, "label": "最小24H成交额"},
    "min_score":                {"type": "float", "default": 58.0,      "label": "最低输出分数"},
    "top_n":                    {"type": "int",   "default": 15,        "label": "最多输出信号数"},
    "allow_short":              {"type": "bool",  "default": True,      "label": "允许空头"},
    # 顶背离
    "divergence_lookback_1h":   {"type": "int",   "default": 20,        "label": "1H背离检测回溯根数"},
    "divergence_lookback_4h":   {"type": "int",   "default": 16,        "label": "4H背离检测回溯根数"},
    "vol_decline_threshold":    {"type": "float", "default": 0.70,      "label": "量能递减阈值(后段/前段<此值)"},
    "price_rise_threshold":     {"type": "float", "default": 1.5,       "label": "价格上涨阈值%(确认HH)"},
    # 底背离
    "vol_surge_ratio":          {"type": "float", "default": 1.60,      "label": "底背离放量倍数(后段量/前段量)"},
    "price_drop_threshold":     {"type": "float", "default": -2.5,      "label": "价格下跌阈值%(确认LL)"},
    # 量能高潮
    "climax_vol_ratio":         {"type": "float", "default": 3.0,       "label": "量能高潮: 单根量/均量≥此倍"},
    "climax_price_range_pct":   {"type": "float", "default": 3.0,       "label": "量能高潮: 当天涨跌<此%视为未突破"},
    # 缩量反弹
    "weak_rally_bars":          {"type": "int",   "default": 5,         "label": "缩量反弹检测连续根数"},
    "weak_rally_vol_decline":   {"type": "float", "default": 0.75,      "label": "缩量反弹: 量递减比例"},
    "weak_rally_price_rise":    {"type": "float", "default": 2.0,       "label": "缩量反弹: 累计涨幅≥此%"},
}

_DEFAULT_CONFIG = {k: v["default"] for k, v in CONFIG_SCHEMA.items()}


class VolumePriceDivergenceScanner(BaseScannerStrategy if _HAS_SCANNER_BASE else object):
    required_bars = ["4H", "1H"]
    strategy_type = "scan"
    name = "量价背离扫描策略"
    description = "检测顶底背离/量能高潮/缩量反弹四类反转前兆"

    def __init__(self, config: Dict = None):
        self.config = {**_DEFAULT_CONFIG, **(config or {})}
        if _HAS_SCANNER_BASE and hasattr(super(), "__init__"):
            try: super().__init__(self.config)
            except Exception: pass

    def _init_conditions(self): pass

    def get_config_schema(self): return dict(CONFIG_SCHEMA)

    # ══════════════════════════════════════════════════════════════════════════

    def scan_symbol(self, symbol) -> Dict[str, Any]:
        inst_id = getattr(symbol, "inst_id", "")
        price = float(getattr(symbol, "last_price", 0) or 0)
        vol24 = float(getattr(symbol, "volume_24h", 0) or 0)
        chg24 = float(getattr(symbol, "price_change_24h", 0) or 0)

        base = {
            "symbol": inst_id, "passed": False, "score": 0.0,
            "direction": "WAIT", "signals": [], "details": {},
            "last_price": price, "volume_24h": vol24, "price_change_24h": chg24,
            "factor_scores": {}, "ranking_factors": {},
        }

        if vol24 < float(self.config.get("min_volume_24h", 2_000_000)):
            return {**base, "details": {"跳过原因": "成交额不足"}}

        h1 = self._get_klines(symbol, "1H")
        h4 = self._get_klines(symbol, "4H")

        if len(h1) < 40 or len(h4) < 24:
            return {**base, "details": {"跳过原因": f"K线不足(H1:{len(h1)}/4H:{len(h4)})"}}

        h1_c, h1_v = self._extract_cv(h1)
        h4_c, h4_v = self._extract_cv(h4)

        results = []

        # ① 顶背离
        bear = self._detect_bearish_divergence(h1_c, h1_v, h4_c, h4_v)
        if bear["detected"]:
            results.append(("顶背离", bear, "SELL", "📉"))

        # ② 底背离
        bull = self._detect_bullish_divergence(h1_c, h1_v, h4_c, h4_v)
        if bull["detected"]:
            results.append(("底背离", bull, "BUY", "📈"))

        # ③ 量能高潮
        if bool(self.config.get("allow_short", True)):
            climax = self._detect_volume_climax(h1_c, h1_v, h4_c, h4_v, chg24)
            if climax["detected"]:
                direction = "SELL" if climax.get("bias", "neutral") == "bear" else "BUY"
                emoji = "📉" if direction == "SELL" else "📈"
                results.append(("量能高潮", climax, direction, emoji))

        # ④ 缩量反弹
        weak = self._detect_weak_rally(h1_c, h1_v)
        if weak["detected"]:
            results.append(("缩量反弹", weak, "SELL", "⚠"))

        if not results:
            return {**base, "details": {"状态": "未检测到量价背离信号"}}

        # 取最高分
        best_type, best_r, direction, emoji = max(results, key=lambda x: x[1]["score"])
        score = best_r["score"]
        passed = score >= float(self.config.get("min_score", 58.0))
        category = f"{emoji} {best_type}"

        signals = [f"{category} · {score:.1f}分"]
        for key_msg in best_r.get("messages", [])[:4]:
            signals.append(key_msg)

        ranking_factors = {
            "trend": score * 0.7, "trigger": score,
            "volume": best_r.get("vol_score", score * 0.8),
            "location": best_r.get("location_score", 50),
            "freshness": best_r.get("freshness_score", 60),
            "risk": 100 - score * 0.3,
        }

        result = {
            **base,
            "passed": passed, "score": score, "opportunity_score": score,
            "direction": direction, "category": category,
            "signals": signals, "factor_scores": {"divergence_score": score},
            "ranking_factors": ranking_factors,
            "details": {
                "机会类型": category, "背离类型": best_type,
                "评分": f"{score:.1f}", "方向": "空头" if direction == "SELL" else "多头",
                **best_r.get("diag", {}),
            },
        }

        if passed and enrich_scan_result:
            try: enrich_scan_result(result)
            except Exception: pass

        return result

    # ══════════════════════════════════════════════════════════════════════════
    # 四种背离检测
    # ══════════════════════════════════════════════════════════════════════════

    def _detect_bearish_divergence(self, h1_c, h1_v, h4_c, h4_v) -> Dict:
        """顶背离: 价格创新高 + 量递减"""
        lb = int(self.config["divergence_lookback_1h"])
        lb4 = int(self.config["divergence_lookback_4h"])
        vol_th = float(self.config["vol_decline_threshold"])
        price_th = float(self.config["price_rise_threshold"])

        half1 = max(4, lb // 2)
        if len(h1_c) < lb:
            return {"detected": False}

        # 1H: 后半段平均价格 > 前半段 (HH) 且后半段均量 < 前半段均量 (量缩)
        p1, p2 = h1_c[-lb:-half1], h1_c[-half1:]
        v1, v2 = h1_v[-lb:-half1], h1_v[-half1:]
        h1_div = (
            np.mean(p2) > np.mean(p1) * (1 + price_th / 100)
            and np.mean(v2) < np.mean(v1) * vol_th
        )

        # 4H: 同逻辑确认
        half4 = max(3, lb4 // 2)
        h4_div = False
        if len(h4_c) >= lb4:
            p4_1, p4_2 = h4_c[-lb4:-half4], h4_c[-half4:]
            v4_1, v4_2 = h4_v[-lb4:-half4], h4_v[-half4:]
            h4_div = (
                np.mean(p4_2) > np.mean(p4_1) * (1 + price_th / 200)
                and np.mean(v4_2) < np.mean(v4_1) * vol_th
            )

        confirmed = h1_div
        score = 0
        messages = []
        if h1_div:
            score += 45
            vol_drop = (1 - np.mean(v2) / max(np.mean(v1), 1)) * 100
            messages.append(f"1H顶背离: HH(+{(np.mean(p2)/np.mean(p1)-1)*100:.1f}%) + 量缩({vol_drop:.0f}%)")
        if h4_div:
            score += 25
            messages.append(f"4H顶背离确认")
        if confirmed:
            score += 10

        return {
            "detected": confirmed and score >= 50,
            "score": min(100, score + 20),
            "vol_score": min(100, score * 0.8),
            "messages": messages,
            "diag": {
                "1H顶背离": str(h1_div),
                "4H顶背离": str(h4_div),
                "1H后半均量vs前半": f"{np.mean(v2)/max(np.mean(v1),1):.2f}x",
            },
        }

    def _detect_bullish_divergence(self, h1_c, h1_v, h4_c, h4_v) -> Dict:
        """底背离: 价格创新低 + 量递增(恐慌抛售→吸筹)"""
        lb = int(self.config["divergence_lookback_1h"])
        vol_th = float(self.config["vol_surge_ratio"])
        price_th = float(self.config["price_drop_threshold"])

        half1 = max(4, lb // 2)
        if len(h1_c) < lb:
            return {"detected": False}

        p1, p2 = h1_c[-lb:-half1], h1_c[-half1:]
        v1, v2 = h1_v[-lb:-half1], h1_v[-half1:]
        h1_div = (
            np.mean(p2) < np.mean(p1) * (1 + price_th / 100)
            and np.mean(v2) > np.mean(v1) * vol_th
        )

        score = 0
        messages = []
        if h1_div:
            score += 45
            vol_surge = (np.mean(v2) / max(np.mean(v1), 1) - 1) * 100
            messages.append(f"1H底背离: LL({(np.mean(p2)/np.mean(p1)-1)*100:.1f}%) + 放量(+{vol_surge:.0f}%)")

        # 确认: 价格最近3根止跌回升
        if len(h1_c) >= 4 and h1_c[-1] > h1_c[-3]:
            score += 20
            messages.append("价格止跌回升确认")

        return {
            "detected": h1_div and score >= 50,
            "score": min(100, score + 20),
            "vol_score": min(100, score * 0.9),
            "messages": messages,
            "diag": {"1H底背离": str(h1_div), "止跌": str(h1_c[-1] > h1_c[-3]) if len(h1_c) >= 3 else "N/A"},
        }

    def _detect_volume_climax(self, h1_c, h1_v, h4_c, h4_v, chg24) -> Dict:
        """量能高潮: 单根量远超均量 + 价格未有效突破"""
        climax_vol = float(self.config["climax_vol_ratio"])
        climax_range = float(self.config["climax_price_range_pct"])

        if len(h1_v) < 20 or len(h1_c) < 20:
            return {"detected": False}

        avg_vol = np.mean(h1_v[-20:-1])
        last_vol = h1_v[-1]
        vol_ratio = last_vol / max(avg_vol, 1)

        if vol_ratio < climax_vol:
            return {"detected": False}

        # 最后一根K线涨跌幅
        if len(h1_c) >= 2:
            last_move = abs(h1_c[-1] / h1_c[-2] - 1) * 100
        else:
            last_move = 0

        stuck = last_move < climax_range
        if not stuck:
            return {"detected": False}

        # 判断方向倾向
        bias = "bear" if chg24 > 3 and last_move < 2 else "neutral"

        return {
            "detected": True,
            "score": min(95, 55 + vol_ratio * 5),
            "vol_score": min(100, vol_ratio * 20),
            "location_score": 100 - last_move * 15,
            "freshness_score": 85,
            "bias": bias,
            "messages": [
                f"量能高潮: 单根量{vol_ratio:.1f}x均量 + 涨幅仅{last_move:.1f}%(未突破)",
                f"24H涨跌{chg24:+.1f}%",
            ],
            "diag": {"量比": f"{vol_ratio:.1f}x", "单根涨跌": f"{last_move:.1f}%", "24H": f"{chg24:+.1f}%"},
        }

    def _detect_weak_rally(self, h1_c, h1_v) -> Dict:
        """缩量反弹: 连续N根上涨 + 量递减"""
        n = int(self.config["weak_rally_bars"])
        vol_dec = float(self.config["weak_rally_vol_decline"])
        price_rise = float(self.config["weak_rally_price_rise"])

        if len(h1_c) < n + 3 or len(h1_v) < n + 3:
            return {"detected": False}

        recent_c = h1_c[-n:]
        recent_v = h1_v[-n:]

        # 连续N根中有 ≥ ceil(N*0.6) 根收涨
        up_count = sum(1 for i in range(1, n) if recent_c[i] > recent_c[i-1])
        if up_count < n * 0.6:
            return {"detected": False}

        # 累计涨幅
        cum_rise = (recent_c[-1] / recent_c[0] - 1) * 100
        if cum_rise < price_rise:
            return {"detected": False}

        # 量能递减: 后半段均量 < 前半段 × vol_dec
        half = max(2, n // 2)
        v1 = np.mean(recent_v[:half])
        v2 = np.mean(recent_v[-half:])
        if v2 >= v1 * vol_dec:
            return {"detected": False}

        return {
            "detected": True,
            "score": min(90, 50 + cum_rise * 5 + (1 - v2 / max(v1, 1)) * 30),
            "vol_score": min(90, (1 - v2 / max(v1, 1)) * 80),
            "messages": [
                f"缩量反弹: {n}根涨{cum_rise:.1f}% + 量缩{(1-v2/max(v1,1))*100:.0f}%",
                f"后半量/前半量={v2/max(v1,1):.2f}x < {vol_dec}",
            ],
            "diag": {
                "连续上涨根数": f"{up_count}/{n}",
                "累计涨幅": f"{cum_rise:.1f}%",
                "量比(后/前)": f"{v2/max(v1,1):.2f}x",
            },
        }

    # ══════════════════════════════════════════════════════════════════════════
    # 工具
    # ══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _get_klines(symbol, tf):
        try:
            ed = getattr(symbol, "extra_data", {}) or {}
            km = ed.get("klines", {}) if isinstance(ed, dict) else {}
            rows = km.get(tf) or []
            return [r for r in rows if isinstance(r, (list, tuple)) and len(r) >= 6]
        except Exception:
            return []

    @staticmethod
    def _extract_cv(rows):
        closes, volumes = [], []
        for r in rows:
            try:
                c = float(r[4]); v = float(r[5])
                if c > 0 and v >= 0:
                    closes.append(c); volumes.append(v)
            except Exception: pass
        return closes, volumes

    def scan_all_symbols(self, symbols: List) -> Dict:
        results = []
        for sym in symbols:
            try:
                r = self.scan_symbol(sym)
            except Exception: continue
            if r.get("passed"): results.append(r)
        results.sort(key=lambda x: float(x.get("score", 0) or 0), reverse=True)
        return {
            "type": "volume_price_divergence",
            "all_opportunities": results[:int(self.config.get("top_n", 15))],
            "total_passed": len(results), "total_scanned": len(symbols),
        }


STRATEGY_NAME  = "量价背离扫描策略"
STRATEGY_TYPE  = "scan"
STRATEGY_CLASS = VolumePriceDivergenceScanner
BACKTEST_CLASS = VolumePriceDivergenceScanner
