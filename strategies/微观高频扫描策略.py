#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
微观高频扫描策略

程序加载版：用短周期 K 线和可选盘口/成交/清算数据，扫描 5-30 分钟级别的短线机会。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from strategies._shared.indicators import (
    _to_df, _aggregate_bars, _pct_change, _safe_float, _clamp,
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
    "min_score": {"type": "float", "default": 72.0, "label": "最低扫描分数"},
    "min_volume_24h": {"type": "float", "default": 8_000_000.0, "label": "最小24H成交额"},
    "top_n": {"type": "int", "default": 12, "label": "最多输出数量"},
    "allow_short": {"type": "bool", "default": True, "label": "允许空头"},
    "max_spread_bps": {"type": "float", "default": 12.0, "label": "最大估算价差bps"},
    "max_slippage_bps": {"type": "float", "default": 18.0, "label": "最大估算滑点bps"},
    "trade_size_usd": {"type": "float", "default": 15_000.0, "label": "估算单笔金额USDT"},
    "ofi_window": {"type": "int", "default": 18, "label": "OFI窗口(1m根数)"},
    "min_ofi_persistence": {"type": "float", "default": 0.58, "label": "OFI持续比例"},
    "min_volume_impulse": {"type": "float", "default": 1.05, "label": "最小量能脉冲"},
    "max_vpin": {"type": "float", "default": 0.90, "label": "最大毒性成交"},
    "liquidation_near_pct": {"type": "float", "default": 1.2, "label": "清算簇接近距离%"},
    "liquidation_hot_density": {"type": "float", "default": 0.30, "label": "清算簇热度阈值"},
    "position_size": {"type": "float", "default": 0.08, "label": "回测仓位比例"},
}

_DEFAULT_CONFIG = {key: spec["default"] for key, spec in CONFIG_SCHEMA.items()}


class AMicrostructureHFScannerStrategy(BaseScannerStrategy if _HAS_SCANNER_BASE else object):
    """短线微观结构扫描器。"""

    required_bars = ["1m", "3m", "5m", "15m"]
    requires_derivative_metrics = True
    name = "微观高频扫描策略"
    description = "OFI/CVD/OBI/Microprice/VPIN/清算簇/执行质量"
    strategy_type = "scan"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        merged = {**_DEFAULT_CONFIG, **(config or {})}
        self.last_metrics: Dict[str, Dict[str, float]] = {}
        if _HAS_SCANNER_BASE and hasattr(super(), "__init__"):
            try:
                super().__init__(merged)
                self.config = merged
            except Exception:
                self.config = merged
        else:
            self.config = merged

    def _init_conditions(self):
        if ScanCondition is None or not hasattr(self, "add_condition"):
            return
        self.add_condition(ScanCondition(
            name="24H成交额",
            description="过滤成交额不足的交易对",
            field="volume_24h",
            operator=">=",
            value=self.config.get("min_volume_24h", 8_000_000.0),
        ))

    def get_config_schema(self) -> Dict[str, Any]:
        return dict(CONFIG_SCHEMA)

    def scan_symbol(self, symbol) -> Dict[str, Any]:
        result = _analyze_symbol(symbol, self.config)
        self.last_metrics[getattr(symbol, "inst_id", "")] = result.get("metrics", {})
        return result

    def scan_all_symbols(self, symbols: List[Any]) -> Dict[str, Any]:
        min_volume = float(self.config.get("min_volume_24h", 8_000_000.0) or 0.0)
        opportunities = []
        for symbol in symbols:
            if float(getattr(symbol, "volume_24h", 0.0) or 0.0) < min_volume:
                continue
            result = self.scan_symbol(symbol)
            if result.get("passed"):
                opportunities.append(result)
        opportunities.sort(
            key=lambda item: (
                float(item.get("opportunity_score", item.get("score", 0.0)) or 0.0),
                float(item.get("volume_24h", 0.0) or 0.0),
            ),
            reverse=True,
        )
        return {
            "type": "microstructure_hf",
            "all_opportunities": opportunities[:int(self.config.get("top_n", 12) or 12)],
        }

    def generate_signal(self, data, *args, **kwargs):
        if isinstance(data, (list, tuple)):
            data = {"klines_map": {"1m": list(data)}}
        if not isinstance(data, dict) or not (data.get("klines_map") or data.get("klines")):
            return None
        if not data.get("klines_map"):
            data = {**data, "klines_map": {"1m": data.get("klines") or []}}
        symbol = _symbol_from_backtest_data(data, self.config)
        result = _analyze_symbol(symbol, self.config)
        if not result.get("passed"):
            return None
        direction = str(result.get("direction", "WAIT")).upper()
        if direction not in {"BUY", "SELL"}:
            return None
        return {
            "action": "BUY" if direction == "BUY" else "SHORT",
            "position_size": float(self.config.get("position_size", 0.08) or 0.08),
            "entry_price": float(result.get("last_price", 0.0) or 0.0),
            "reason": f"{result.get('category')} | 评分 {float(result.get('score', 0.0)):.1f}",
            "score": float(result.get("opportunity_score", result.get("score", 0.0)) or 0.0),
            "raw_result": result,
        }

    def reset_backtest_state(self):
        self.last_metrics.clear()


def _analyze_symbol(symbol, config: Dict[str, Any]) -> Dict[str, Any]:
    inst_id = str(getattr(symbol, "inst_id", "") or "")
    extra = getattr(symbol, "extra_data", {}) or {}
    klines = extra.get("klines", {}) or {}

    m1 = _to_df(_get_klines(klines, "1m"))
    m3 = _to_df(_get_klines(klines, "3m"))
    m5 = _to_df(_get_klines(klines, "5m"))
    m15 = _to_df(_get_klines(klines, "15m"))
    if len(m1) < 30 and len(m3) >= 20:
        m1 = _expand_to_1m_proxy(m3, 3)
    if len(m1) < 30 and len(m5) >= 12:
        m1 = _expand_to_1m_proxy(m5, 5)
    if len(m1) < 30:
        return _failed(inst_id, f"K线不足(1m={len(m1)},3m={len(m3)},5m={len(m5)})")
    if m5.empty:
        m5 = _aggregate_bars(m1, 5)
    if m15.empty:
        m15 = _aggregate_bars(m1, 15)

    price = float(getattr(symbol, "last_price", 0.0) or m1["c"].iloc[-1])
    volume_24h = float(getattr(symbol, "volume_24h", 0.0) or (m1["c"] * m1["vol"]).tail(1440).sum())
    price_change_24h = float(getattr(symbol, "price_change_24h", 0.0) or _pct_change(m1["c"], min(len(m1) - 1, 240)) * 100)

    orderbook = _extract_orderbook(extra)
    spread_bps = _spread_bps(orderbook, m1, price)
    est_depth = _estimated_depth_usd(orderbook, symbol, m1, price)
    slippage_bps = _estimate_slippage_bps(float(config.get("trade_size_usd", 15_000.0) or 15_000.0), est_depth)
    if spread_bps > float(config.get("max_spread_bps", 12.0) or 12.0):
        return _failed(inst_id, f"价差过大({spread_bps:.2f}bps)")
    if slippage_bps > float(config.get("max_slippage_bps", 18.0) or 18.0):
        return _failed(inst_id, f"估算滑点过高({slippage_bps:.2f}bps)")

    signed_volume = _signed_volume(m1)
    ofi = _ofi_proxy(m1, orderbook)
    ofi_window = int(config.get("ofi_window", 18) or 18)
    funding_rate = _safe_float(extra.get("funding_rate"), 0.0) * 100.0
    oi_value = _safe_float(getattr(symbol, "open_interest", 0.0) or extra.get("open_interest"), 0.0)
    liq = _liquidation_signal(extra, price, oi_value)

    metrics = {
        "ofi_z": _last_zscore(ofi, ofi_window),
        "ofi_persistence": _persistence(ofi, ofi_window),
        "obi": _order_book_imbalance(orderbook),
        "close_pressure": _close_pressure(m1),
        "microprice_edge_bps": _microprice_edge_bps(orderbook, price),
        "trade_imbalance": _trade_imbalance(extra, signed_volume),
        "cvd_slope": _cvd_slope(signed_volume),
        "vpin_like": _vpin_like(signed_volume),
        "depth_growth": _depth_growth(m1),
        "momentum_1m_bps": _pct_bps(m1["c"], 3),
        "momentum_5m_bps": _pct_bps(m1["c"], 12),
        "momentum_15m_bps": _pct_bps(m1["c"], 24),
        "realized_vol_bps": _realized_vol_bps(m1["c"], 20),
        "volume_impulse": _volume_ratio(m1["vol"], 20),
        "vwap_dev_bps": _vwap_deviation_bps(m1),
        "trend_alignment": _trend_alignment(m1, m5, m15),
        "liquidity_score": _liquidity_score(volume_24h, est_depth, spread_bps, slippage_bps),
        "spread_bps": spread_bps,
        "slippage_bps": slippage_bps,
        "funding_rate_pct": funding_rate,
        "open_interest_usd": oi_value,
        "open_interest_change_pct": _safe_float(extra.get("open_interest_change_pct"), 0.0),
        "liq_distance_pct": liq["distance_pct"],
        "liq_density": liq["density_ratio"],
        "liq_side": liq["side"],
    }
    if abs(metrics["microprice_edge_bps"]) < 0.01:
        metrics["microprice_edge_bps"] = metrics["close_pressure"] * 3.0

    long_score, short_score, long_reasons, short_reasons = _score_metrics(metrics, config)
    if not bool(config.get("allow_short", True)):
        short_score = 0.0
    direction = "BUY" if long_score >= short_score else "SELL"
    score = max(long_score, short_score)
    reasons = long_reasons if direction == "BUY" else short_reasons
    min_score = float(config.get("min_score", 72.0) or 72.0)
    passed = score >= min_score
    category = "微观高频多头" if direction == "BUY" else "微观高频空头"
    signals = _signal_text(direction, score, metrics, reasons)
    ranking_factors = {
        "trend": _clamp(50 + metrics["trend_alignment"] * 50, 0, 100),
        "trigger": _clamp(50 + abs(metrics["ofi_z"]) * 18 + abs(metrics["trade_imbalance"]) * 30, 0, 100),
        "volume": _clamp(metrics["volume_impulse"] / 1.8 * 100, 0, 100),
        "location": _clamp(75 - abs(metrics["vwap_dev_bps"]) * 0.08, 20, 95),
        "freshness": _clamp(55 + abs(metrics["cvd_slope"]) * 2 + abs(metrics["momentum_5m_bps"]) * 0.08, 30, 96),
        "risk": _clamp(metrics["liquidity_score"] - max(metrics["vpin_like"] - 0.65, 0) * 22, 20, 95),
    }

    result = {
        "symbol": inst_id,
        "passed": passed,
        "score": round(float(score), 2),
        "direction": direction,
        "signals": signals,
        "category": category,
        "strategy_category": category,
        "last_price": price,
        "volume_24h": volume_24h,
        "price_change_24h": price_change_24h,
        "ranking_factors": ranking_factors,
        "metrics": {k: (round(v, 6) if isinstance(v, float) else v) for k, v in metrics.items()},
        "details": {
            "机会类型": category,
            "评估": " | ".join(signals),
            "OFI_Z": f"{metrics['ofi_z']:+.2f}",
            "OFI持续": f"{metrics['ofi_persistence']:.0%}",
            "OBI": f"{metrics['obi']:+.2f}",
            "收盘压力": f"{metrics['close_pressure']:+.2f}",
            "Microprice边际": f"{metrics['microprice_edge_bps']:+.2f}bps",
            "主动成交失衡": f"{metrics['trade_imbalance']:+.2f}",
            "CVD斜率": f"{metrics['cvd_slope']:+.2f}",
            "VPIN毒性": f"{metrics['vpin_like']:.2f}",
            "深度增长": f"{metrics['depth_growth']:+.1%}",
            "1m动量": f"{metrics['momentum_1m_bps']:+.1f}bps",
            "5m动量": f"{metrics['momentum_5m_bps']:+.1f}bps",
            "15m动量": f"{metrics['momentum_15m_bps']:+.1f}bps",
            "量能脉冲": f"{metrics['volume_impulse']:.2f}x",
            "VWAP偏离": f"{metrics['vwap_dev_bps']:+.1f}bps",
            "趋势一致": f"{metrics['trend_alignment']:+.2f}",
            "价差": f"{spread_bps:.2f}bps",
            "估算滑点": f"{slippage_bps:.2f}bps",
            "资金费率%": f"{funding_rate:+.4f}%",
            "OI变化%": f"{metrics['open_interest_change_pct']:+.2f}",
            "清算簇距离": f"{liq['distance_pct']:.2f}%",
            "清算簇密度": f"{liq['density_ratio']:.2%}",
            "清算簇方向": str(liq["side"]),
            "失效条件": _invalidation_text(direction, price, m1, spread_bps),
        },
    }
    if build_opportunity_profile:
        try:
            result.update(build_opportunity_profile(score, direction, volume_24h, ranking_factors, signals))
        except Exception:
            pass
    return result


def _score_metrics(metrics: Dict[str, Any], config: Dict[str, Any]):
    long_score = 0.0
    short_score = 0.0
    long_reasons: List[str] = []
    short_reasons: List[str] = []
    min_persist = float(config.get("min_ofi_persistence", 0.58) or 0.58)

    ofi_z = float(metrics["ofi_z"])
    persistence = float(metrics["ofi_persistence"])
    if ofi_z > 0.65 and persistence >= min_persist:
        add = min(24.0, 9.0 + ofi_z * 5.5)
        long_score += add
        long_reasons.append(f"OFI持续正向(z={ofi_z:.2f},持续{persistence:.0%})")
    elif ofi_z < -0.65 and persistence >= min_persist:
        add = min(24.0, 9.0 + abs(ofi_z) * 5.5)
        short_score += add
        short_reasons.append(f"OFI持续负向(z={ofi_z:.2f},持续{persistence:.0%})")

    pressure = max(float(metrics["obi"]), float(metrics["close_pressure"]))
    sell_pressure = min(float(metrics["obi"]), float(metrics["close_pressure"]))
    if pressure > 0.12:
        add = min(16.0, 6.0 + pressure * 26.0)
        long_score += add
        long_reasons.append(f"买盘/收盘压力占优({pressure:+.2f})")
    elif sell_pressure < -0.12:
        add = min(16.0, 6.0 + abs(sell_pressure) * 26.0)
        short_score += add
        short_reasons.append(f"卖盘/收盘压力占优({sell_pressure:+.2f})")

    micro = float(metrics["microprice_edge_bps"])
    if micro > 0.55:
        long_score += min(12.0, 4.0 + micro * 1.5)
        long_reasons.append(f"Microprice向上({micro:+.2f}bps)")
    elif micro < -0.55:
        short_score += min(12.0, 4.0 + abs(micro) * 1.5)
        short_reasons.append(f"Microprice向下({micro:+.2f}bps)")

    trade_imb = float(metrics["trade_imbalance"])
    if trade_imb > 0.12:
        long_score += min(13.0, 5.5 + trade_imb * 21.0)
        long_reasons.append(f"主动买成交占优({trade_imb:+.2f})")
    elif trade_imb < -0.12:
        short_score += min(13.0, 5.5 + abs(trade_imb) * 21.0)
        short_reasons.append(f"主动卖成交占优({trade_imb:+.2f})")

    cvd = float(metrics["cvd_slope"])
    if cvd > 1.2:
        long_score += min(10.0, 3.0 + cvd * 1.6)
        long_reasons.append(f"CVD斜率转强({cvd:+.2f})")
    elif cvd < -1.2:
        short_score += min(10.0, 3.0 + abs(cvd) * 1.6)
        short_reasons.append(f"CVD斜率转弱({cvd:+.2f})")

    mom5 = float(metrics["momentum_5m_bps"])
    mom15 = float(metrics["momentum_15m_bps"])
    trend = float(metrics["trend_alignment"])
    if max(mom5, mom15) > 12.0 and trend >= 0:
        add = min(12.0, 3.0 + max(mom5, mom15) / 18.0 + trend * 4.0)
        long_score += add
        long_reasons.append(f"短窗动量顺势({mom5:+.1f}/{mom15:+.1f}bps)")
    elif min(mom5, mom15) < -12.0 and trend <= 0:
        add = min(12.0, 3.0 + abs(min(mom5, mom15)) / 18.0 + abs(trend) * 4.0)
        short_score += add
        short_reasons.append(f"短窗动量顺势({mom5:+.1f}/{mom15:+.1f}bps)")

    vol_impulse = float(metrics["volume_impulse"])
    if vol_impulse >= float(config.get("min_volume_impulse", 1.05) or 1.05):
        bonus = min(8.0, (vol_impulse - 1.0) * 7.0)
        if long_score >= short_score:
            long_score += bonus
            long_reasons.append(f"量能脉冲确认({vol_impulse:.2f}x)")
        else:
            short_score += bonus
            short_reasons.append(f"量能脉冲确认({vol_impulse:.2f}x)")

    depth_growth = float(metrics["depth_growth"])
    if depth_growth > 0.06:
        if long_score >= short_score:
            long_score += 5.0
            long_reasons.append(f"流动性扩张({depth_growth:+.1%})")
        else:
            short_score += 5.0
            short_reasons.append(f"流动性扩张({depth_growth:+.1%})")

    if float(metrics["liq_distance_pct"]) <= float(config.get("liquidation_near_pct", 1.2) or 1.2):
        if float(metrics["liq_density"]) >= float(config.get("liquidation_hot_density", 0.30) or 0.30):
            if metrics["liq_side"] == "short":
                long_score += 12.0
                long_reasons.append("上方空头清算簇接近")
            elif metrics["liq_side"] == "long":
                short_score += 12.0
                short_reasons.append("下方多头清算簇接近")

    funding = float(metrics["funding_rate_pct"])
    oi_change = float(metrics["open_interest_change_pct"])
    if funding > 0.06 and oi_change > 0:
        short_score += 5.0
        short_reasons.append("资金费率/OI过热")
    elif funding < -0.04 and oi_change > 0:
        long_score += 5.0
        long_reasons.append("资金费率偏冷且OI扩张")

    execution_bonus = _clamp((float(metrics["liquidity_score"]) - 55.0) / 45.0 * 8.0, 0.0, 8.0)
    long_score += execution_bonus
    short_score += execution_bonus

    long_alignment = _direction_alignment_count(metrics, 1.0, config)
    short_alignment = _direction_alignment_count(metrics, -1.0, config)
    if long_alignment >= 4:
        long_score += min(14.0, (long_alignment - 3) * 3.5)
        long_reasons.append(f"多指标同向共振({long_alignment}/8)")
    if short_alignment >= 4:
        short_score += min(14.0, (short_alignment - 3) * 3.5)
        short_reasons.append(f"多指标同向共振({short_alignment}/8)")

    vpin = float(metrics["vpin_like"])
    max_vpin = float(config.get("max_vpin", 0.90) or 0.90)
    if vpin > max_vpin:
        penalty = min(18.0, 8.0 + (vpin - max_vpin) * 45.0)
        if long_alignment >= 5 or short_alignment >= 5:
            penalty *= 0.45
        long_score -= penalty
        short_score -= penalty
    elif vpin > 0.70:
        long_score -= 4.0
        short_score -= 4.0
    return max(long_score, 0.0), max(short_score, 0.0), long_reasons, short_reasons


def _direction_alignment_count(metrics: Dict[str, Any], sign: float, config: Dict[str, Any]) -> int:
    checks = [
        float(metrics["ofi_z"]) * sign > 0.65,
        max(float(metrics["obi"]) * sign, float(metrics["close_pressure"]) * sign) > 0.18,
        float(metrics["microprice_edge_bps"]) * sign > 0.55,
        float(metrics["trade_imbalance"]) * sign > 0.12,
        float(metrics["cvd_slope"]) * sign > 1.20,
        max(float(metrics["momentum_5m_bps"]) * sign, float(metrics["momentum_15m_bps"]) * sign) > 12.0,
        float(metrics["trend_alignment"]) * sign > 0.35,
        float(metrics["volume_impulse"]) >= float(config.get("min_volume_impulse", 1.05) or 1.05),
    ]
    return int(sum(bool(item) for item in checks))


def _signal_text(direction: str, score: float, metrics: Dict[str, Any], reasons: List[str]) -> List[str]:
    side = "多头" if direction == "BUY" else "空头"
    top = reasons[:4] if reasons else ["执行质量尚可"]
    return [
        f"微观{side}评分 {score:.1f}",
        f"OFI {metrics['ofi_z']:+.2f} / 主动成交 {metrics['trade_imbalance']:+.2f} / CVD {metrics['cvd_slope']:+.2f}",
        f"动量 {metrics['momentum_5m_bps']:+.1f}bps / 量能 {metrics['volume_impulse']:.2f}x / VPIN {metrics['vpin_like']:.2f}",
        f"执行质量: 价差 {metrics['spread_bps']:.2f}bps, 滑点 {metrics['slippage_bps']:.2f}bps",
        "；".join(top),
    ]


def _failed(symbol: str, reason: str) -> Dict[str, Any]:
    return {"symbol": symbol, "passed": False, "score": 0.0, "direction": "WAIT", "signals": [], "details": {"状态": reason}, "metrics": {}}


def _get_klines(klines_map: Dict[str, List], bar: str) -> List:
    return klines_map.get(bar) or klines_map.get(bar.lower()) or klines_map.get(bar.upper()) or []


def _expand_to_1m_proxy(df: pd.DataFrame, factor: int) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        for i in range(factor):
            item = row.copy()
            item["ts"] = float(row["ts"]) + i * 60_000
            item["vol"] = float(row["vol"]) / factor
            rows.append(item)
    return pd.DataFrame(rows).reset_index(drop=True) if rows else df


def _extract_orderbook(extra: Dict[str, Any]) -> Dict[str, float]:
    book = extra.get("orderbook") or extra.get("book") or extra.get("order_book") or {}
    if not isinstance(book, dict):
        return {}
    bid_price = _safe_float(book.get("bid_price") or book.get("bid") or _first_level(book.get("bids"), 0), 0.0)
    ask_price = _safe_float(book.get("ask_price") or book.get("ask") or _first_level(book.get("asks"), 0), 0.0)
    bid_size = _safe_float(book.get("bid_size") or book.get("bid_volume") or _first_level(book.get("bids"), 1), 0.0)
    ask_size = _safe_float(book.get("ask_size") or book.get("ask_volume") or _first_level(book.get("asks"), 1), 0.0)
    return {
        "bid_price": bid_price,
        "ask_price": ask_price,
        "bid_size": bid_size,
        "ask_size": ask_size,
        "bid_depth_usd": _safe_float(book.get("bid_depth_usd"), bid_price * bid_size if bid_price > 0 else 0.0),
        "ask_depth_usd": _safe_float(book.get("ask_depth_usd"), ask_price * ask_size if ask_price > 0 else 0.0),
    }


def _first_level(levels: Any, idx: int) -> float:
    try:
        if levels and len(levels) > 0 and len(levels[0]) > idx:
            return float(levels[0][idx])
    except Exception:
        return 0.0
    return 0.0


def _spread_bps(orderbook: Dict[str, float], df: pd.DataFrame, price: float) -> float:
    bid = orderbook.get("bid_price", 0.0)
    ask = orderbook.get("ask_price", 0.0)
    if bid > 0 and ask > bid:
        return (ask - bid) / ((ask + bid) / 2.0) * 10_000
    latest_range = (df["h"].tail(8) - df["l"].tail(8)) / df["c"].tail(8).replace(0, np.nan) * 10_000
    estimate = float(latest_range.median()) if not latest_range.dropna().empty else 4.0
    return _clamp(estimate * 0.18, 0.5, 20.0)


def _estimated_depth_usd(orderbook: Dict[str, float], symbol, df: pd.DataFrame, price: float) -> float:
    depth = orderbook.get("bid_depth_usd", 0.0) + orderbook.get("ask_depth_usd", 0.0)
    if depth > 0:
        return depth
    volume_24h = float(getattr(symbol, "volume_24h", 0.0) or 0.0)
    recent_turnover = float((df["c"] * df["vol"]).tail(20).median() or 0.0)
    return max(volume_24h * 0.18, recent_turnover * 80.0, price * max(float(df["vol"].tail(20).median() or 0.0), 1.0) * 40.0)


def _estimate_slippage_bps(trade_size_usd: float, depth_usd: float) -> float:
    if depth_usd <= 0:
        return 999.0
    return _clamp((trade_size_usd / depth_usd) * 10_000 * 0.12, 0.0, 999.0)


def _ofi_proxy(df: pd.DataFrame, orderbook: Dict[str, float]) -> pd.Series:
    close_location = ((df["c"] - df["l"]) / (df["h"] - df["l"]).replace(0, np.nan) - 0.5).fillna(0.0)
    pressure = close_location * df["vol"] + np.sign(df["c"].diff().fillna(0.0)) * df["vol"] * 0.35
    if orderbook:
        pressure.iloc[-1] += _order_book_imbalance(orderbook) * max(float(df["vol"].tail(20).median() or 0.0), 1.0)
    return pressure


def _signed_volume(df: pd.DataFrame) -> pd.Series:
    body = df["c"] - df["o"]
    close_location = ((df["c"] - df["l"]) / (df["h"] - df["l"]).replace(0, np.nan) - 0.5).fillna(0.0)
    sign = np.where(body > 0, 1.0, np.where(body < 0, -1.0, np.sign(close_location)))
    return pd.Series(sign, index=df.index) * df["vol"]


def _last_zscore(series: pd.Series, window: int) -> float:
    tail = pd.to_numeric(series, errors="coerce").tail(max(window, 5))
    if len(tail) < 5:
        return 0.0
    std = float(tail.std(ddof=0) or 0.0)
    if std <= 1e-12:
        return 0.0
    return _clamp((float(tail.iloc[-1]) - float(tail.mean())) / std, -4.0, 4.0)


def _persistence(series: pd.Series, window: int) -> float:
    tail = pd.to_numeric(series, errors="coerce").tail(max(window, 5))
    if tail.empty:
        return 0.0
    last_sign = np.sign(float(tail.iloc[-1]))
    if last_sign == 0:
        return 0.0
    return float((np.sign(tail) == last_sign).mean())


def _order_book_imbalance(orderbook: Dict[str, float]) -> float:
    bid = float(orderbook.get("bid_depth_usd", 0.0) or 0.0)
    ask = float(orderbook.get("ask_depth_usd", 0.0) or 0.0)
    total = bid + ask
    return 0.0 if total <= 0 else _clamp((bid - ask) / total, -1.0, 1.0)


def _microprice_edge_bps(orderbook: Dict[str, float], price: float) -> float:
    bid_price = float(orderbook.get("bid_price", 0.0) or 0.0)
    ask_price = float(orderbook.get("ask_price", 0.0) or 0.0)
    bid_size = float(orderbook.get("bid_size", 0.0) or 0.0)
    ask_size = float(orderbook.get("ask_size", 0.0) or 0.0)
    denom = bid_size + ask_size
    if denom <= 0 or bid_price <= 0 or ask_price <= 0 or price <= 0:
        return 0.0
    microprice = (ask_price * bid_size + bid_price * ask_size) / denom
    return (microprice / price - 1.0) * 10_000


def _close_pressure(df: pd.DataFrame, window: int = 8) -> float:
    tail = df.tail(window)
    rng = (tail["h"] - tail["l"]).replace(0, np.nan)
    pressure = ((tail["c"] - tail["l"]) / rng - 0.5).dropna()
    return float(pressure.mean() * 2.0) if not pressure.empty else 0.0


def _trade_imbalance(extra: Dict[str, Any], signed_volume: pd.Series) -> float:
    buy = _safe_float(extra.get("taker_buy_volume"), np.nan)
    sell = _safe_float(extra.get("taker_sell_volume"), np.nan)
    if np.isfinite(buy) and np.isfinite(sell) and buy + sell > 0:
        return _clamp((buy - sell) / (buy + sell), -1.0, 1.0)
    tail = signed_volume.tail(12)
    denom = float(tail.abs().sum() or 0.0)
    return 0.0 if denom <= 0 else _clamp(float(tail.sum()) / denom, -1.0, 1.0)


def _vpin_like(signed_volume: pd.Series, window: int = 24) -> float:
    tail = signed_volume.tail(window)
    denom = float(tail.abs().sum() or 0.0)
    return 0.0 if denom <= 0 else _clamp(abs(float(tail.sum())) / denom, 0.0, 1.0)


def _cvd_slope(signed_volume: pd.Series) -> float:
    tail = signed_volume.tail(24)
    if len(tail) < 6:
        return 0.0
    cvd = tail.cumsum()
    slope = float(np.polyfit(np.arange(len(cvd), dtype=float), cvd.values.astype(float), 1)[0])
    denom = max(float(tail.abs().mean() or 0.0), 1.0)
    return _clamp(slope / denom * 10.0, -30.0, 30.0)


def _depth_growth(df: pd.DataFrame) -> float:
    turnover = df["c"] * df["vol"]
    if len(turnover) < 12:
        return 0.0
    recent = float(turnover.tail(4).mean() or 0.0)
    base = float(turnover.iloc[-12:-4].mean() or 0.0)
    return 0.0 if base <= 0 else _clamp(recent / base - 1.0, -1.0, 3.0)


def _realized_vol_bps(close: pd.Series, window: int = 20) -> float:
    returns = close.pct_change().dropna().tail(window)
    return float(returns.std(ddof=0) * np.sqrt(max(len(returns), 1)) * 10_000) if not returns.empty else 0.0


def _volume_ratio(vol: pd.Series, window: int = 20) -> float:
    if len(vol) < window + 1:
        return 1.0
    base = float(vol.iloc[-(window + 1):-1].mean() or 0.0)
    return float(vol.iloc[-1] / base) if base > 0 else 1.0


def _vwap_deviation_bps(df: pd.DataFrame, window: int = 30) -> float:
    tail = df.tail(window)
    vol = float(tail["vol"].sum() or 0.0)
    if vol <= 0:
        return 0.0
    vwap = float((tail["c"] * tail["vol"]).sum() / vol)
    price = float(tail["c"].iloc[-1])
    return (price / vwap - 1.0) * 10_000 if vwap > 0 else 0.0


def _trend_alignment(m1: pd.DataFrame, m5: pd.DataFrame, m15: pd.DataFrame) -> float:
    score = 0.0
    for df, fast, slow, weight in [(m1, 8, 21, 0.35), (m5, 8, 21, 0.35), (m15, 5, 13, 0.30)]:
        if len(df) < slow + 2:
            continue
        ema_fast = df["c"].ewm(span=fast, adjust=False).mean()
        ema_slow = df["c"].ewm(span=slow, adjust=False).mean()
        slope = float(ema_fast.diff().tail(3).mean() or 0.0)
        direction = 1.0 if ema_fast.iloc[-1] > ema_slow.iloc[-1] and slope > 0 else -1.0 if ema_fast.iloc[-1] < ema_slow.iloc[-1] and slope < 0 else 0.0
        score += direction * weight
    return _clamp(score, -1.0, 1.0)


def _liquidity_score(volume_24h: float, depth_usd: float, spread_bps: float, slippage_bps: float) -> float:
    volume_part = _clamp((np.log10(max(volume_24h, 1.0)) - 6.0) / 2.2 * 45.0, 0.0, 45.0)
    depth_part = _clamp((np.log10(max(depth_usd, 1.0)) - 4.5) / 2.0 * 35.0, 0.0, 35.0)
    execution_part = _clamp(30.0 - spread_bps * 1.1 - slippage_bps * 0.7, 0.0, 30.0)
    return _clamp(volume_part + depth_part + execution_part, 0.0, 100.0)


def _liquidation_signal(extra: Dict[str, Any], price: float, oi_value: float) -> Dict[str, Any]:
    clusters = extra.get("liquidation_clusters") or extra.get("liquidations") or []
    parsed = []
    for item in clusters:
        if isinstance(item, dict):
            level = _safe_float(item.get("level") or item.get("price"), 0.0)
            density = _safe_float(item.get("density") or item.get("density_usd") or item.get("notional"), 0.0)
            side = str(item.get("side", "none"))
        elif isinstance(item, (list, tuple)) and len(item) >= 3:
            level = _safe_float(item[0], 0.0)
            density = _safe_float(item[1], 0.0)
            side = str(item[2])
        else:
            continue
        if level > 0 and density > 0:
            parsed.append((level, density, side))
    if not parsed or price <= 0:
        return {"distance_pct": 999.0, "density_ratio": 0.0, "side": "none", "level": 0.0}
    level, density, side = min(parsed, key=lambda row: abs(row[0] - price))
    denom = oi_value if oi_value > 0 else max(density, 1.0)
    return {"distance_pct": abs(level - price) / price * 100, "density_ratio": density / denom, "side": side, "level": level}


def _invalidation_text(direction: str, price: float, df: pd.DataFrame, spread_bps: float) -> str:
    vol_bps = max(_realized_vol_bps(df["c"], 20), 5.0)
    invalid_bps = max(vol_bps * 1.25, spread_bps * 3.0, 12.0)
    level = price * (1 - invalid_bps / 10_000) if direction == "BUY" else price * (1 + invalid_bps / 10_000)
    return f"OFI连续2根反向或价格触及 {level:.8g}"


def _pct_bps(series: pd.Series, bars: int) -> float:
    return _pct_change(series, bars) * 10_000


class _MinimalSymbol:
    def __init__(self, inst_id: str, last_price: float, volume_24h: float, price_change_24h: float, extra_data: Dict[str, Any]):
        self.inst_id = inst_id
        self.last_price = last_price
        self.volume_24h = volume_24h
        self.price_change_24h = price_change_24h
        self.high_24h = 0.0
        self.low_24h = 0.0
        self.open_interest = 0.0
        self.extra_data = extra_data


def _symbol_from_backtest_data(data: Dict[str, Any], config: Dict[str, Any]):
    klines_map = data.get("klines_map") or {}
    m1 = _to_df(_get_klines(klines_map, "1m") or _get_klines(klines_map, "1H") or data.get("klines") or [])
    price = float(m1["c"].iloc[-1]) if not m1.empty else 0.0
    return _MinimalSymbol(
        inst_id=str(config.get("inst_id", "BACKTEST-SYMBOL") or "BACKTEST-SYMBOL"),
        last_price=price,
        volume_24h=float((m1["c"] * m1["vol"]).tail(240).sum()) if not m1.empty else 0.0,
        price_change_24h=_pct_change(m1["c"], min(len(m1) - 1, 240)) * 100 if not m1.empty else 0.0,
        extra_data={"klines": klines_map},
    )


STRATEGY_NAME = "微观高频扫描策略"
STRATEGY_TYPE = "scan"
STRATEGY_CLASS = AMicrostructureHFScannerStrategy
BACKTEST_CLASS = AMicrostructureHFScannerStrategy
