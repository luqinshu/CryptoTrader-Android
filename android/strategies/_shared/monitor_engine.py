"""
持仓反转监控引擎 — Kivy/Android 适配版。

从 position_reversal_monitor.py 提取核心算法，适配为线程安全的 Kivy 集成。

用法:
    monitor = PositionReversalMonitor(client, config)
    alerts = monitor.run_cycle()  # 在 bg 线程中调用
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from strategies._shared.monitor_utils import (
    atr,
    calc_vwap,
    closes,
    completed_window,
    ema,
    latest_complete_close,
    parse_candles,
    rolling_atr_rank,
    safe_float,
    volumes,
)

# ── 告警等级 ──────────────────────────────────────────────────────────────────
LEVEL_ORDER = ["NORMAL", "OBSERVE", "WARN", "ALERT", "CRITICAL"]

# ── 默认配置 ──────────────────────────────────────────────────────────────────
DEFAULT_MONITOR_CONFIG: Dict[str, Any] = {
    "loop_seconds": 20,
    "cooldown_seconds": 300,
    "persist_cycles_warn": 3,
    "volatility_profiles": {
        "low": {"atr_multiplier": 1.0},
        "medium": {"atr_multiplier": 1.2},
        "high": {"atr_multiplier": 1.5},
    },
    "thresholds": {
        "ret_5m_atr": {"observe": 1.2, "warn": 1.5},
        "ret_15m_atr": {"observe": 1.5, "warn": 1.8},
        "vwap_dev": {"observe": -0.008, "warn": -0.012},
        "vol_ratio": {"observe": 1.5, "warn": 1.8},
        "oi_change_5m": {"observe": -0.015, "warn": -0.03},
        "depth_imbalance": {"observe": 0.8, "warn": 0.6},
        "atr_rank": {"observe": 0.7, "warn": 0.85},
    },
    "derivatives": {
        "funding_delta_warn": -0.00005,
        "basis_delta_warn": -0.001,
    },
    "weights": {
        "break_15m": 30,
        "ret_15m_atr": 15,
        "trend_against_position": 15,
        "vwap_against": 10,
        "vol_ratio": 10,
        "oi_against": 10,
        "funding_or_basis_weak": 5,
        "orderbook_against": 5,
        "break_1h_bonus": 10,
        "atr_rank_bonus": 5,
    },
    "score_levels": {
        "observe_min": 40,
        "warn_min": 60,
        "alert_min": 75,
        "critical_min": 85,
    },
    "actions": {
        "NORMAL": "HOLD",
        "OBSERVE": "STOP_ADDING",
        "WARN": "TIGHTEN_STOP_AND_REDUCE_20",
        "ALERT": "REDUCE_30_TO_50",
        "CRITICAL": "STOP_LOSS_OR_HARD_REDUCE",
    },
}


def _level_rank(level: str) -> int:
    return LEVEL_ORDER.index(level) if level in LEVEL_ORDER else 0


def _first_data(payload: Dict[str, Any]) -> Dict[str, Any]:
    if payload.get("code") != "0" or not payload.get("data"):
        raise RuntimeError(payload.get("msg") or "OKX 返回空数据")
    return payload["data"][0]


class PositionReversalMonitor:
    """持仓反转监控器 — Kivy 适配版。"""

    def __init__(self, client: Any, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            client: OKXClient 实例
            config: 监控配置，为 None 时使用默认配置
        """
        self.client = client
        user_cfg = config or {}
        self.cfg = DEFAULT_MONITOR_CONFIG.copy()
        _deep_update(self.cfg, user_cfg)
        self._state: Dict[str, Dict[str, float]] = {}
        self._cooldowns: Dict[str, float] = {}
        self._persist_counts: Dict[str, int] = {}
        self.positions_summary: List[Dict[str, Any]] = []

    # ── 公开 API ──────────────────────────────────────────────────────────────

    def run_cycle(self) -> Dict[str, Any]:
        """执行一轮监控。返回 {'evaluations': [...], 'alerts': [...]}。"""
        positions = self._fetch_positions()
        if not positions:
            self.positions_summary = []
            return {"evaluations": [], "alerts": [], "positions": []}

        evals: List[Dict[str, Any]] = []
        alerts: List[Dict[str, Any]] = []

        for pos in positions:
            try:
                evaluation = self._evaluate_position(pos)
                evals.append(evaluation)
                if self._should_alert(evaluation):
                    self._record_alert(evaluation)
                    alerts.append(evaluation)
            except Exception:
                pass

        self.positions_summary = evals
        return {"evaluations": evals, "alerts": alerts, "positions": positions}

    # ── 持仓获取 ──────────────────────────────────────────────────────────────

    def _fetch_positions(self) -> List[Dict[str, Any]]:
        resp = self.client.get_positions()
        if resp.get("code") != "0":
            return []
        positions: List[Dict[str, Any]] = []
        for row in resp.get("data", []):
            qty = safe_float(row.get("pos"))
            if abs(qty) <= 0:
                continue
            pos_side = str(row.get("posSide", "")).lower()
            if pos_side in ("long", "short"):
                side = pos_side.upper()
            else:
                side = "LONG" if qty > 0 else "SHORT"
            positions.append({
                "inst_id": str(row.get("instId", "")).upper().strip(),
                "side": side,
                "entry_price": safe_float(row.get("avgPx")),
                "size": abs(qty),
                "lever": safe_float(row.get("lever"), 1.0),
                "liq_px": safe_float(row.get("liqPx")),
            })
        return positions

    # ── 市场数据 ──────────────────────────────────────────────────────────────

    def _fetch_market_bundle(self, inst_id: str) -> Dict[str, Any]:
        ticker = _first_data(self.client.get_ticker(inst_id))
        price = safe_float(ticker.get("last"))

        k5m_raw = self.client.get_kline(inst_id, bar="5m", limit=96).get("data", [])
        k15m_raw = self.client.get_kline(inst_id, bar="15m", limit=96).get("data", [])
        k1h_raw = self.client.get_kline(inst_id, bar="1H", limit=96).get("data", [])
        c5m = parse_candles(k5m_raw)
        c15m = parse_candles(k15m_raw)
        c1h = parse_candles(k1h_raw)

        # 数据量检查
        if len(c5m) < 20 or len(c15m) < 50 or len(c1h) < 20:
            raise RuntimeError(f"K线数据不足 (5m:{len(c5m)} 15m:{len(c15m)} 1h:{len(c1h)})")

        # 盘口
        book_data = _first_data(self.client.get_order_book(inst_id, limit=5))
        bids = book_data.get("bids", [])[:5]
        asks = book_data.get("asks", [])[:5]
        bid_depth = sum(safe_float(item[1]) for item in bids if len(item) >= 2)
        ask_depth = sum(safe_float(item[1]) for item in asks if len(item) >= 2)
        depth_imbalance = bid_depth / ask_depth if ask_depth > 0 else 999.0

        # 资金费率 / OI
        funding_rate = 0.0
        oi_value = 0.0
        if inst_id.endswith("-SWAP"):
            try:
                funding_rate = safe_float(_first_data(self.client.get_funding_rate(inst_id)).get("fundingRate"))
            except Exception:
                pass
            try:
                oi_p = _first_data(self.client.get_open_interest(inst_id, instType="SWAP"))
                oi_value = safe_float(oi_p.get("oiUsd") or oi_p.get("oi"))
            except Exception:
                pass

        return {
            "price": price,
            "candles_5m": c5m,
            "candles_15m": c15m,
            "candles_1h": c1h,
            "depth_imbalance": depth_imbalance,
            "funding_rate": funding_rate,
            "oi_value": oi_value,
        }

    # ── 评估逻辑 ──────────────────────────────────────────────────────────────

    def _evaluate_position(self, pos: Dict[str, Any]) -> Dict[str, Any]:
        inst_id = pos["inst_id"]
        side = pos["side"]
        entry_price = pos["entry_price"]

        bundle = self._fetch_market_bundle(inst_id)
        price = bundle["price"]
        c5m = bundle["candles_5m"]
        c15m = bundle["candles_15m"]
        c1h = bundle["candles_1h"]
        depth_imbalance = bundle["depth_imbalance"]
        funding_rate = bundle["funding_rate"]
        oi_value = bundle["oi_value"]

        # ── 波动计算 ──
        atr_5m = atr(c5m, 14)
        atr_15m = atr(c15m, 14)
        close_5m = latest_complete_close(c5m)
        close_15m = latest_complete_close(c15m)
        ret_5m_atr = abs(price - close_5m) / atr_5m if atr_5m > 0 else 0.0
        ret_15m_atr = abs(price - close_15m) / atr_15m if atr_15m > 0 else 0.0

        # ── EMA 趋势 ──
        closes_15m = closes(c15m)
        ema20 = ema(closes_15m, 20)
        ema60 = ema(closes_15m, 60)

        # ── 结构破位 ──
        struct_15m = completed_window(c15m, 4)
        struct_1h = completed_window(c1h, 4)
        if side == "LONG":
            break_15m = price < min(c["l"] for c in struct_15m) if struct_15m else False
            break_1h = price < min(c["l"] for c in struct_1h) if struct_1h else False
            trend_against = ema20 < ema60
        else:
            break_15m = price > max(c["h"] for c in struct_15m) if struct_15m else False
            break_1h = price > max(c["h"] for c in struct_1h) if struct_1h else False
            trend_against = ema20 > ema60

        # ── VWAP 偏离 ──
        vwap_window = c5m[-96:] if len(c5m) >= 96 else c5m
        vwap_val = calc_vwap(vwap_window)
        vwap_dev = (price - vwap_val) / vwap_val if vwap_val > 0 else 0.0

        # ── 量能比 ──
        vol_series = volumes(c15m)
        recent_vol = vol_series[-1] if vol_series else 0.0
        prev_vols = vol_series[-21:-1] if len(vol_series) >= 21 else vol_series[:-1]
        avg_vol = sum(prev_vols) / len(prev_vols) if prev_vols else 0.0
        vol_ratio = recent_vol / avg_vol if avg_vol > 0 else 0.0

        # ── OI / Funding 变化 ──
        state = self._get_state(inst_id)
        prev_oi = state.get("prev_oi", 0.0)
        prev_funding = state.get("prev_funding", 0.0)
        oi_change_5m = (oi_value - prev_oi) / prev_oi if prev_oi > 0 else 0.0
        funding_delta = funding_rate - prev_funding

        # ── ATR 分位 ──
        atr_rank_val = rolling_atr_rank(c15m, 14, 30)

        # ── 方向性判断 ──
        vol_profile = "medium"
        if side == "LONG":
            vwap_against = vwap_dev <= self._threshold("vwap_dev", "warn", vol_profile)
            book_against = depth_imbalance <= self._threshold("depth_imbalance", "warn", vol_profile)
            oi_against = oi_change_5m <= self._threshold("oi_change_5m", "warn", vol_profile)
            funding_weak = funding_delta <= self.cfg["derivatives"]["funding_delta_warn"]
            pnl_pct = (price - entry_price) / max(entry_price, 1e-9)
        else:
            vwap_against = vwap_dev >= abs(self._threshold("vwap_dev", "warn", vol_profile))
            book_thr = self._threshold("depth_imbalance", "warn", vol_profile)
            book_against = depth_imbalance >= (1.0 / max(book_thr, 0.01))
            oi_against = oi_change_5m <= self._threshold("oi_change_5m", "warn", vol_profile)
            funding_weak = funding_delta >= abs(self.cfg["derivatives"]["funding_delta_warn"])
            pnl_pct = (entry_price - price) / max(entry_price, 1e-9)

        # ── 风险标记 ──
        risk_flags = [
            ret_5m_atr >= self._threshold("ret_5m_atr", "observe", vol_profile),
            ret_15m_atr >= self._threshold("ret_15m_atr", "observe", vol_profile),
            break_15m,
            trend_against,
            vwap_against,
            vol_ratio >= self._threshold("vol_ratio", "observe", vol_profile),
            book_against,
            oi_against,
            funding_weak,
        ]
        if sum(1 for f in risk_flags if f) >= 2:
            self._persist_counts[inst_id] = self._persist_counts.get(inst_id, 0) + 1
        else:
            self._persist_counts[inst_id] = 0
        persist_count = self._persist_counts.get(inst_id, 0)

        # ── 加权评分 ──
        w = self.cfg["weights"]
        score = 0
        if break_15m:
            score += w["break_15m"]
        if ret_15m_atr >= self._threshold("ret_15m_atr", "warn", vol_profile):
            score += w["ret_15m_atr"]
        if trend_against:
            score += w["trend_against_position"]
        if vwap_against:
            score += w["vwap_against"]
        if vol_ratio >= self._threshold("vol_ratio", "warn", vol_profile):
            score += w["vol_ratio"]
        if oi_against:
            score += w["oi_against"]
        if funding_weak:
            score += w["funding_or_basis_weak"]
        if book_against:
            score += w["orderbook_against"]
        if break_1h:
            score += w["break_1h_bonus"]
        if atr_rank_val >= self._threshold("atr_rank", "warn", vol_profile):
            score += w["atr_rank_bonus"]
        score = min(score, 100)

        # ── 告警等级 ──
        lvls = self.cfg["score_levels"]
        if score >= lvls["critical_min"]:
            level = "CRITICAL"
        elif score >= lvls["alert_min"]:
            level = "ALERT"
        elif score >= lvls["warn_min"]:
            level = "WARN"
        elif score >= lvls["observe_min"]:
            level = "OBSERVE"
        else:
            level = "NORMAL"

        # PnL 止损覆盖
        if pnl_pct <= -0.03 or (break_1h and trend_against and persist_count >= self.cfg.get("persist_cycles_warn", 3)):
            level = "CRITICAL"
        elif pnl_pct <= -0.015 and _level_rank(level) < _level_rank("WARN"):
            level = "WARN"

        # ── 原因描述 ──
        reasons = []
        if ret_15m_atr >= self._threshold("ret_15m_atr", "observe", vol_profile):
            reasons.append(f"15m波动={ret_15m_atr:.1f}ATR")
        if break_15m:
            reasons.append("15m破位")
        if break_1h:
            reasons.append("1h破位")
        if trend_against:
            reasons.append("EMA趋势反向")
        if vwap_against:
            reasons.append(f"VWAP偏离={vwap_dev:+.2%}")
        if vol_ratio >= self._threshold("vol_ratio", "observe", vol_profile):
            reasons.append(f"量能={vol_ratio:.1f}x")
        if oi_against:
            reasons.append(f"OI变化={oi_change_5m:+.2%}")
        funding_or_basis = False
        if funding_weak:
            reasons.append(f"资金费率Δ={funding_delta:+.5f}")
            funding_or_basis = True
        if book_against:
            reasons.append(f"盘口比={depth_imbalance:.2f}")

        # 更新状态
        state["prev_oi"] = oi_value
        state["prev_funding"] = funding_rate

        return {
            "inst_id": inst_id,
            "side": side,
            "price": price,
            "entry_price": entry_price,
            "pnl_pct": pnl_pct,
            "ret_5m_atr": round(ret_5m_atr, 2),
            "ret_15m_atr": round(ret_15m_atr, 2),
            "break_15m": break_15m,
            "break_1h": break_1h,
            "trend_against": trend_against,
            "vwap_dev": round(vwap_dev, 4),
            "vol_ratio": round(vol_ratio, 2),
            "oi_change_5m": round(oi_change_5m, 4),
            "funding_delta": round(funding_delta, 6),
            "depth_imbalance": round(depth_imbalance, 2),
            "atr_rank": round(atr_rank_val, 2),
            "persist_count": persist_count,
            "reversal_score": score,
            "alert_level": level,
            "action": self.cfg["actions"].get(level, "HOLD"),
            "reason": " | ".join(reasons) if reasons else "无异常",
        }

    # ── 告警管理 ──────────────────────────────────────────────────────────────

    def _should_alert(self, ev: Dict[str, Any]) -> bool:
        inst_id = ev["inst_id"]
        level = ev["alert_level"]
        if _level_rank(level) <= _level_rank("NORMAL"):
            return False
        cooldown_end = self._cooldowns.get(inst_id, 0)
        now = time.time()
        if now < cooldown_end:
            return False
        return True

    def _record_alert(self, ev: Dict[str, Any]) -> None:
        self._cooldowns[ev["inst_id"]] = time.time() + self.cfg.get("cooldown_seconds", 300)

    def clear_alert(self, inst_id: str) -> None:
        """手动清除冷却，允许立即再次告警。"""
        self._cooldowns.pop(inst_id, None)

    def reset(self) -> None:
        """重置所有状态。"""
        self._state.clear()
        self._cooldowns.clear()
        self._persist_counts.clear()

    # ── 内部工具 ──────────────────────────────────────────────────────────────

    def _get_state(self, inst_id: str) -> Dict[str, float]:
        if inst_id not in self._state:
            self._state[inst_id] = {"prev_oi": 0.0, "prev_funding": 0.0}
        return self._state[inst_id]

    def _threshold(self, key: str, level: str, vol_profile: str = "medium") -> float:
        base = self.cfg.get("thresholds", {}).get(key, {}).get(level, 0.0)
        atr_mult = self.cfg.get("volatility_profiles", {}).get(vol_profile, {}).get("atr_multiplier", 1.0)
        return base * atr_mult


# ── 工具函数 ─────────────────────────────────────────────────────────────────

def _deep_update(target: Dict, source: Dict) -> None:
    """递归合并 source 到 target。"""
    for key, value in source.items():
        if key in target and isinstance(target[key], dict) and isinstance(value, dict):
            _deep_update(target[key], value)
        else:
            target[key] = value
