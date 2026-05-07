"""
Strategy lifecycle guard for combo scanners.

Each scan strategy has a market premise where it works best and a set of
failure conditions where the same signal becomes low quality.  This module
keeps those rules centralized so the combo scanner can down-rank or block
signals whose operating environment has already deteriorated.
"""

from __future__ import annotations

import re
from typing import Callable, Dict, List, Tuple

from src.scanner.ranking import build_opportunity_profile


ACTIVE_DIRECTIONS = {"BUY", "SELL", "LONG", "SHORT"}


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _detail_float(details: Dict, key: str, default: float = 0.0) -> float:
    if not isinstance(details, dict):
        return default
    raw = details.get(key)
    if raw is None:
        return default
    if isinstance(raw, (int, float)):
        return float(raw)
    match = re.search(r"-?\d+(?:\.\d+)?", str(raw))
    return float(match.group(0)) if match else default


def _factor(result: Dict, key: str, default: float = 0.0) -> float:
    factors = result.get("ranking_factors") or {}
    return _safe_float(factors.get(key), default)


def _score_to_level(score: float) -> str:
    if score >= 92:
        return "S"
    if score >= 84:
        return "A"
    if score >= 76:
        return "B"
    if score >= 68:
        return "C"
    return "D"


def _add_check(
    premises: List[str],
    failures: List[str],
    warnings: List[str],
    ok: bool,
    label: str,
    fail_label: str | None = None,
    hard: bool = False,
) -> None:
    if ok:
        premises.append(label)
    elif hard:
        failures.append(fail_label or label)
    else:
        warnings.append(fail_label or label)


def _breakout_start(result: Dict, details: Dict, config: Dict) -> Tuple[List[str], List[str], List[str]]:
    premises, failures, warnings = [], [], []
    width = _detail_float(details, "平台宽度", 999.0)
    volume = _detail_float(details, "量比", 0.0)
    extension = _detail_float(details, "延伸幅度", 999.0)
    max_width = _safe_float(config.get("lifecycle_breakout_max_platform_width_pct"), 9.0)
    min_volume = _safe_float(config.get("lifecycle_breakout_min_volume_ratio"), 1.45)
    max_extension = _safe_float(config.get("lifecycle_breakout_max_extension_pct"), 5.0)

    _add_check(premises, failures, warnings, width <= max_width, f"平台宽度合格({width:.2f}%)", f"平台过宽({width:.2f}%)", hard=width > max_width * 1.25)
    _add_check(premises, failures, warnings, volume >= min_volume, f"突破量能合格({volume:.2f}x)", f"突破量能不足({volume:.2f}x)")
    _add_check(premises, failures, warnings, extension <= max_extension, f"突破初段未跑远({extension:.2f}%)", f"突破后延伸过大({extension:.2f}%)", hard=True)
    _add_check(premises, failures, warnings, _factor(result, "trigger") >= 60.0, "突破触发确认", "突破触发质量不足", hard=True)
    return premises, failures, warnings


def _new_high(result: Dict, details: Dict, config: Dict) -> Tuple[List[str], List[str], List[str]]:
    premises, failures, warnings = [], [], []
    trend = _detail_float(details, "趋势质量", _factor(result, "trend"))
    adx = _detail_float(details, "H4_ADX", 0.0)
    efficiency = _detail_float(details, "趋势效率", 0.0)
    extension = _detail_float(details, "延伸幅度", 999.0)
    rsi = _detail_float(details, "1H_RSI", 50.0)
    atr = _detail_float(details, "4H_ATR%", 0.0)

    min_trend = _safe_float(config.get("lifecycle_new_high_min_trend_quality"), 68.0)
    min_adx = _safe_float(config.get("lifecycle_new_high_min_adx"), 18.0)
    min_efficiency = _safe_float(config.get("lifecycle_new_high_min_efficiency"), 22.0)
    max_extension = _safe_float(config.get("lifecycle_new_high_max_extension_pct"), 5.0)
    max_rsi = _safe_float(config.get("lifecycle_new_high_max_rsi"), 79.0)
    max_atr = _safe_float(config.get("lifecycle_new_high_max_atr_pct"), 6.5)

    _add_check(premises, failures, warnings, trend >= min_trend, f"趋势质量合格({trend:.0f})", f"趋势质量不足({trend:.0f})", hard=True)
    _add_check(premises, failures, warnings, adx >= min_adx, f"ADX支持突破({adx:.1f})", f"ADX偏弱({adx:.1f})")
    _add_check(premises, failures, warnings, efficiency >= min_efficiency, f"趋势效率合格({efficiency:.0f})", f"趋势效率不足({efficiency:.0f})")
    _add_check(premises, failures, warnings, extension <= max_extension, f"新高初段未过度延伸({extension:.2f}%)", f"新高已过度延伸({extension:.2f}%)", hard=True)
    _add_check(premises, failures, warnings, rsi <= max_rsi, f"RSI未极端过热({rsi:.1f})", f"RSI极端过热({rsi:.1f})", hard=rsi > max_rsi + 4)
    _add_check(premises, failures, warnings, atr <= max_atr, f"波动风险可控({atr:.2f}%)", f"4H波动过大({atr:.2f}%)")
    return premises, failures, warnings


def _directional_trend(result: Dict, details: Dict, config: Dict) -> Tuple[List[str], List[str], List[str]]:
    premises, failures, warnings = [], [], []
    trend = _factor(result, "trend", _detail_float(details, "趋势质量", 0.0))
    adx = _detail_float(details, "H4_ADX", 0.0)
    efficiency = _detail_float(details, "趋势效率", 0.0)
    extension = _detail_float(details, "延伸幅度", 999.0)
    atr = _detail_float(details, "4H_ATR%", 0.0)

    min_trend = _safe_float(config.get("lifecycle_directional_min_trend_quality"), 70.0)
    min_adx = _safe_float(config.get("lifecycle_directional_min_adx"), 20.0)
    min_efficiency = _safe_float(config.get("lifecycle_directional_min_efficiency"), 25.0)
    max_extension = _safe_float(config.get("lifecycle_directional_max_extension_pct"), 4.8)
    max_atr = _safe_float(config.get("lifecycle_directional_max_atr_pct"), 6.5)

    _add_check(premises, failures, warnings, trend >= min_trend, f"单边趋势质量合格({trend:.0f})", f"趋势强度不足({trend:.0f})", hard=True)
    _add_check(premises, failures, warnings, adx >= min_adx, f"ADX确认单边({adx:.1f})", f"ADX不足以支持单边({adx:.1f})")
    _add_check(premises, failures, warnings, efficiency >= min_efficiency, f"趋势效率合格({efficiency:.0f})", f"趋势推进效率低({efficiency:.0f})")
    _add_check(premises, failures, warnings, extension <= max_extension, f"顺势位置未跑远({extension:.2f}%)", f"顺势追价过远({extension:.2f}%)", hard=True)
    _add_check(premises, failures, warnings, atr <= max_atr, f"ATR风险可控({atr:.2f}%)", f"波动过大不适合追随({atr:.2f}%)")
    return premises, failures, warnings


def _trend_pullback(result: Dict, details: Dict, config: Dict) -> Tuple[List[str], List[str], List[str]]:
    premises, failures, warnings = [], [], []
    trend = _factor(result, "trend", _detail_float(details, "趋势质量", 0.0))
    adx = _detail_float(details, "H4_ADX", 0.0)
    pullback = _detail_float(details, "回踩距离", 999.0)
    volume = _detail_float(details, "量比", 0.0)

    min_trend = _safe_float(config.get("lifecycle_pullback_min_trend_quality"), 68.0)
    min_adx = _safe_float(config.get("lifecycle_pullback_min_adx"), 16.0)
    min_pullback = _safe_float(config.get("lifecycle_pullback_min_distance_pct"), 0.25)
    max_pullback = _safe_float(config.get("lifecycle_pullback_max_distance_pct"), 3.5)
    min_volume = _safe_float(config.get("lifecycle_pullback_min_volume_ratio"), 1.05)

    _add_check(premises, failures, warnings, trend >= min_trend, f"主趋势仍有效({trend:.0f})", f"主趋势质量不足({trend:.0f})", hard=True)
    _add_check(premises, failures, warnings, adx >= min_adx, f"趋势强度支持回踩({adx:.1f})", f"ADX偏弱，可能是震荡回踩({adx:.1f})")
    _add_check(premises, failures, warnings, min_pullback <= pullback <= max_pullback, f"回踩深度合格({pullback:.2f}%)", f"回踩过浅或过深({pullback:.2f}%)", hard=pullback > max_pullback)
    _add_check(premises, failures, warnings, volume >= min_volume, f"回踩确认量能合格({volume:.2f}x)", f"回踩确认量能不足({volume:.2f}x)")
    _add_check(premises, failures, warnings, _factor(result, "trigger") >= 60.0, "二次启动触发有效", "二次启动触发不足", hard=True)
    return premises, failures, warnings


def _divergence_reversal(result: Dict, details: Dict, config: Dict) -> Tuple[List[str], List[str], List[str]]:
    premises, failures, warnings = [], [], []
    direction = str(result.get("direction") or "WAIT").upper()
    h4_rsi = _detail_float(details, "4H_RSI", 50.0)
    h1_rsi = _detail_float(details, "1H_RSI", 50.0)
    volume = _detail_float(details, "量比", 0.0)
    range_pct = _detail_float(details, "位置偏离", 999.0)
    max_range = _safe_float(config.get("lifecycle_divergence_max_range_pct"), 4.2)
    min_volume = _safe_float(config.get("lifecycle_divergence_min_volume_ratio"), 1.05)

    exhausted = h4_rsi <= 44.0 if direction in {"BUY", "LONG"} else h4_rsi >= 56.0
    rsi_confirm = h1_rsi <= 54.0 if direction in {"BUY", "LONG"} else h1_rsi >= 46.0
    _add_check(premises, failures, warnings, exhausted, f"4H衰竭区确认({h4_rsi:.1f})", f"4H未进入反转衰竭区({h4_rsi:.1f})", hard=True)
    _add_check(premises, failures, warnings, rsi_confirm, f"1H背离位置合理({h1_rsi:.1f})", f"1H背离位置不充分({h1_rsi:.1f})")
    _add_check(premises, failures, warnings, range_pct <= max_range, f"反转位置未偏离({range_pct:.2f}%)", f"反转位置已偏离({range_pct:.2f}%)", hard=True)
    _add_check(premises, failures, warnings, volume >= min_volume, f"反转确认量能合格({volume:.2f}x)", f"反转确认量能不足({volume:.2f}x)")
    _add_check(premises, failures, warnings, _factor(result, "trigger") >= 65.0, "背离触发确认", "背离触发质量不足", hard=True)
    return premises, failures, warnings


def _oversold_reversal(result: Dict, details: Dict, config: Dict) -> Tuple[List[str], List[str], List[str]]:
    premises, failures, warnings = [], [], []
    h4_rsi = _detail_float(details, "4H_RSI", 50.0)
    h1_rsi = _detail_float(details, "1H_RSI", 50.0)
    drop = _detail_float(details, "近期跌幅", 0.0)
    volume = _detail_float(details, "量比", 0.0)
    max_h4_rsi = _safe_float(config.get("lifecycle_oversold_max_h4_rsi"), 38.0)
    max_h1_rsi = _safe_float(config.get("lifecycle_oversold_max_h1_rsi"), 34.0)
    min_drop = _safe_float(config.get("lifecycle_oversold_min_drop_pct"), 10.0)
    min_volume = _safe_float(config.get("lifecycle_oversold_min_volume_ratio"), 1.15)

    _add_check(premises, failures, warnings, h4_rsi <= max_h4_rsi, f"4H超跌充分({h4_rsi:.1f})", f"4H超跌不充分({h4_rsi:.1f})", hard=h4_rsi > max_h4_rsi + 5)
    _add_check(premises, failures, warnings, h1_rsi <= max_h1_rsi, f"1H超跌充分({h1_rsi:.1f})", f"1H超跌不充分({h1_rsi:.1f})")
    _add_check(premises, failures, warnings, drop >= min_drop, f"下跌释放充分({drop:.2f}%)", f"下跌释放不足({drop:.2f}%)", hard=drop < min_drop * 0.7)
    _add_check(premises, failures, warnings, volume >= min_volume, f"反转量能合格({volume:.2f}x)", f"反转量能不足({volume:.2f}x)")
    _add_check(premises, failures, warnings, _factor(result, "trigger") >= 60.0, "反转K线触发确认", "反转K线触发不足", hard=True)
    return premises, failures, warnings


def _volatility_compression(result: Dict, details: Dict, config: Dict) -> Tuple[List[str], List[str], List[str]]:
    premises, failures, warnings = [], [], []
    atr_ratio = _detail_float(details, "ATR收缩系数", 1.0)
    band_ratio = _detail_float(details, "带宽收缩系数", 1.0)
    width = _detail_float(details, "压缩区宽度", 999.0)
    volume = _detail_float(details, "量比", 0.0)
    breakout = _detail_float(details, "突破幅度", 999.0)
    extension = _detail_float(details, "延伸幅度", 999.0)

    max_atr_ratio = _safe_float(config.get("lifecycle_compression_max_atr_ratio"), 0.88)
    max_band_ratio = _safe_float(config.get("lifecycle_compression_max_band_ratio"), 0.86)
    max_width = _safe_float(config.get("lifecycle_compression_max_width_pct"), 6.0)
    min_volume = _safe_float(config.get("lifecycle_compression_min_volume_ratio"), 1.25)
    max_breakout = _safe_float(config.get("lifecycle_compression_max_breakout_pct"), 3.2)
    max_extension = _safe_float(config.get("lifecycle_compression_max_extension_pct"), 5.2)

    compressed = atr_ratio <= max_atr_ratio or band_ratio <= max_band_ratio
    _add_check(premises, failures, warnings, compressed, f"至少一个波动收缩维度有效(ATR{atr_ratio:.2f}/带宽{band_ratio:.2f})", f"波动未真正收缩(ATR{atr_ratio:.2f}/带宽{band_ratio:.2f})", hard=atr_ratio > 1.0 and band_ratio > 1.0)
    _add_check(premises, failures, warnings, width <= max_width, f"压缩区宽度合格({width:.2f}%)", f"压缩区过宽({width:.2f}%)", hard=width > max_width * 1.25)
    _add_check(premises, failures, warnings, volume >= min_volume, f"爆发量能合格({volume:.2f}x)", f"爆发量能不足({volume:.2f}x)")
    _add_check(premises, failures, warnings, breakout <= max_breakout, f"爆发仍在初段({breakout:.2f}%)", f"爆发后已跑远({breakout:.2f}%)", hard=breakout > max_breakout * 1.5)
    _add_check(premises, failures, warnings, extension <= max_extension, f"均线延伸可控({extension:.2f}%)", f"均线延伸过大({extension:.2f}%)", hard=True)
    return premises, failures, warnings


def _continuation_restart(result: Dict, details: Dict, config: Dict) -> Tuple[List[str], List[str], List[str]]:
    premises, failures, warnings = [], [], []
    runup = _detail_float(details, "前段涨幅", 0.0)
    width = _detail_float(details, "平台宽度", 999.0)
    contraction = _detail_float(details, "缩量系数", 1.0)
    volume = _detail_float(details, "启动量比", 0.0)
    extension = _detail_float(details, "延伸幅度", 999.0)

    min_runup = _safe_float(config.get("lifecycle_continuation_min_runup_pct"), 12.0)
    max_width = _safe_float(config.get("lifecycle_continuation_max_base_width_pct"), 5.0)
    max_contraction = _safe_float(config.get("lifecycle_continuation_max_contraction_ratio"), 0.88)
    min_volume = _safe_float(config.get("lifecycle_continuation_min_volume_ratio"), 1.35)
    max_extension = _safe_float(config.get("lifecycle_continuation_max_extension_pct"), 5.8)

    _add_check(premises, failures, warnings, runup >= min_runup, f"前段趋势涨幅充分({runup:.2f}%)", f"前段趋势不足({runup:.2f}%)", hard=runup < min_runup * 0.7)
    _add_check(premises, failures, warnings, width <= max_width, f"中继平台宽度合格({width:.2f}%)", f"中继平台过宽({width:.2f}%)", hard=width > max_width * 1.3)
    _add_check(premises, failures, warnings, contraction <= max_contraction, f"整理期缩量有效({contraction:.2f})", f"整理期未缩量({contraction:.2f})", hard=contraction > 1.0)
    _add_check(premises, failures, warnings, volume >= min_volume, f"二次启动量能合格({volume:.2f}x)", f"二次启动量能不足({volume:.2f}x)")
    _add_check(premises, failures, warnings, extension <= max_extension, f"二次启动位置可控({extension:.2f}%)", f"二次启动已过度延伸({extension:.2f}%)", hard=True)
    _add_check(premises, failures, warnings, _factor(result, "trigger") >= 60.0, "二次启动触发确认", "二次启动触发不足", hard=True)
    return premises, failures, warnings


def _false_breakout_reclaim(result: Dict, details: Dict, config: Dict) -> Tuple[List[str], List[str], List[str]]:
    premises, failures, warnings = [], [], []
    direction = str(result.get("direction") or "WAIT").upper()
    reclaim_distance = _detail_float(details, "回收距离", 999.0)
    volume = _detail_float(details, "量比", 0.0)
    lower_wick = _detail_float(details, "下影线占比", 0.0)
    upper_wick = _detail_float(details, "上影线占比", 0.0)
    h1_rsi = _detail_float(details, "1H_RSI", 50.0)
    h4_atr = _detail_float(details, "4H_ATR%", 0.0)
    max_distance = _safe_float(config.get("lifecycle_false_breakout_max_reclaim_distance_pct"), 2.6)
    min_volume = _safe_float(config.get("lifecycle_false_breakout_min_volume_ratio"), 1.15)
    min_wick = _safe_float(config.get("lifecycle_false_breakout_min_wick_ratio_pct"), 35.0)
    max_atr = _safe_float(config.get("lifecycle_false_breakout_max_atr_pct"), 7.5)

    wick_ok = lower_wick >= min_wick if direction in {"BUY", "LONG"} else upper_wick >= min_wick
    rsi_ok = 24.0 <= h1_rsi <= 62.0 if direction in {"BUY", "LONG"} else 38.0 <= h1_rsi <= 76.0
    _add_check(premises, failures, warnings, reclaim_distance <= max_distance, f"回收后仍贴近关键位({reclaim_distance:.2f}%)", f"回收后离关键位过远({reclaim_distance:.2f}%)", hard=True)
    _add_check(premises, failures, warnings, wick_ok, f"扫流动性影线有效({max(lower_wick, upper_wick):.0f}%)", f"扫盘拒绝影线不足({max(lower_wick, upper_wick):.0f}%)")
    _add_check(premises, failures, warnings, volume >= min_volume, f"扫盘量能合格({volume:.2f}x)", f"扫盘量能不足({volume:.2f}x)")
    _add_check(premises, failures, warnings, rsi_ok, f"RSI处于回收区({h1_rsi:.1f})", f"RSI位置不支持回收({h1_rsi:.1f})")
    _add_check(premises, failures, warnings, h4_atr <= max_atr, f"4H波动可控({h4_atr:.2f}%)", f"4H波动过大，假突破容易失真({h4_atr:.2f}%)")
    _add_check(premises, failures, warnings, _factor(result, "trigger") >= 65.0, "回收触发确认", "回收触发质量不足", hard=True)
    return premises, failures, warnings


def _relative_strength_leader(result: Dict, details: Dict, config: Dict) -> Tuple[List[str], List[str], List[str]]:
    premises, failures, warnings = [], [], []
    rs = _detail_float(details, "相对市场强度", 0.0)
    trend = _detail_float(details, "趋势质量", _factor(result, "trend", 0.0))
    h4_momentum = _detail_float(details, "4H动量", 0.0)
    h1_momentum = _detail_float(details, "1H动量", 0.0)
    extension = _detail_float(details, "延伸幅度", 999.0)
    rsi = _detail_float(details, "RSI", 50.0)
    min_rs = _safe_float(config.get("lifecycle_relative_min_strength_pct"), 3.0)
    min_trend = _safe_float(config.get("lifecycle_relative_min_trend_quality"), 66.0)
    max_extension = _safe_float(config.get("lifecycle_relative_max_extension_pct"), 6.2)
    max_rsi = _safe_float(config.get("lifecycle_relative_max_rsi"), 78.0)

    _add_check(premises, failures, warnings, rs >= min_rs, f"相对强弱显著({rs:.2f}%)", f"相对强弱不足({rs:.2f}%)", hard=True)
    _add_check(premises, failures, warnings, trend >= min_trend, f"领涨趋势质量合格({trend:.0f})", f"领涨趋势质量不足({trend:.0f})", hard=True)
    _add_check(premises, failures, warnings, h4_momentum > 0 and h1_momentum > 0, f"4H/1H动量共振({h4_momentum:.2f}%/{h1_momentum:.2f}%)", f"短中周期动量未共振({h4_momentum:.2f}%/{h1_momentum:.2f}%)")
    _add_check(premises, failures, warnings, extension <= max_extension, f"领涨未过度追高({extension:.2f}%)", f"领涨已过度延伸({extension:.2f}%)", hard=True)
    _add_check(premises, failures, warnings, rsi <= max_rsi, f"RSI未极端过热({rsi:.1f})", f"RSI极端过热({rsi:.1f})", hard=rsi > max_rsi + 4)
    return premises, failures, warnings


def _funding_reversal(result: Dict, details: Dict, config: Dict) -> Tuple[List[str], List[str], List[str]]:
    premises, failures, warnings = [], [], []
    direction = str(result.get("direction") or "WAIT").upper()
    funding = _detail_float(details, "资金费率%", 0.0)
    h4_momentum = _detail_float(details, "4H动量", 0.0)
    h1_momentum = _detail_float(details, "1H动量", 0.0)
    rsi = _detail_float(details, "RSI", 50.0)
    volume = _detail_float(details, "量比", 0.0)
    threshold = _safe_float(config.get("lifecycle_funding_extreme_threshold_pct"), 0.08)
    stall_limit = _safe_float(config.get("lifecycle_funding_stall_momentum_pct"), 2.0)
    min_volume = _safe_float(config.get("lifecycle_funding_min_volume_ratio"), 1.05)

    if direction in {"SELL", "SHORT"}:
        funding_ok = funding >= threshold
        stall_ok = h4_momentum <= stall_limit and h1_momentum <= 1.0
        rsi_ok = rsi >= 58.0
        crowd_label = "多头拥挤"
    else:
        funding_ok = funding <= -threshold
        stall_ok = h4_momentum >= -stall_limit and h1_momentum >= -1.0
        rsi_ok = rsi <= 46.0
        crowd_label = "空头拥挤"

    _add_check(premises, failures, warnings, funding_ok, f"资金费率极端{crowd_label}({funding:.4f}%)", f"资金费率不够极端({funding:.4f}%)", hard=True)
    _add_check(premises, failures, warnings, stall_ok, f"价格对拥挤方向拒绝延续({h4_momentum:.2f}%/{h1_momentum:.2f}%)", f"价格仍顺拥挤方向加速({h4_momentum:.2f}%/{h1_momentum:.2f}%)", hard=True)
    _add_check(premises, failures, warnings, rsi_ok, f"RSI支持反向挤压({rsi:.1f})", f"RSI未显示反向挤压({rsi:.1f})")
    _add_check(premises, failures, warnings, volume >= min_volume, f"反转参与量合格({volume:.2f}x)", f"反转参与量不足({volume:.2f}x)")
    return premises, failures, warnings


def _open_interest_anomaly(result: Dict, details: Dict, config: Dict) -> Tuple[List[str], List[str], List[str]]:
    premises, failures, warnings = [], [], []
    direction = str(result.get("direction") or "WAIT").upper()
    price_24h = _detail_float(details, "24H涨跌", 0.0)
    price_4h = _detail_float(details, "4H涨跌", 0.0)
    participation = _detail_float(details, "参与度变化", 0.0)
    oi_value = _detail_float(details, "当前持仓量", 0.0)
    volume = _detail_float(details, "量比", 0.0)
    min_participation = _safe_float(config.get("lifecycle_oi_min_participation_change_pct"), 7.0)
    min_volume = _safe_float(config.get("lifecycle_oi_min_volume_ratio"), 1.15)

    participation_ok = abs(participation) >= min_participation
    direction_ok = (direction in {"BUY", "LONG"} and (price_24h > 1.5 or price_4h > 0.8)) or (direction in {"SELL", "SHORT"} and (price_24h < -1.5 or price_4h < -0.8))
    release_reversal = direction in {"BUY", "LONG"} and price_24h < -2.0 and participation < -4.0
    _add_check(premises, failures, warnings, participation_ok, f"持仓/参与度变化显著({participation:.2f}%)", f"持仓/参与度变化不足({participation:.2f}%)", hard=True)
    _add_check(premises, failures, warnings, direction_ok or release_reversal, f"价格方向与参与度解释一致({price_24h:.2f}%/{price_4h:.2f}%)", f"价格方向与持仓解释不一致({price_24h:.2f}%/{price_4h:.2f}%)", hard=True)
    _add_check(premises, failures, warnings, volume >= min_volume, f"成交活跃确认({volume:.2f}x)", f"成交活跃度不足({volume:.2f}x)")
    _add_check(premises, failures, warnings, oi_value > 0, f"真实持仓量数据有效({oi_value:.2f})", "缺少真实持仓量，当前为量能代理")
    return premises, failures, warnings


def _volume_absorption(result: Dict, details: Dict, config: Dict) -> Tuple[List[str], List[str], List[str]]:
    premises, failures, warnings = [], [], []
    volume = _detail_float(details, "量比", 0.0)
    drawdown = _detail_float(details, "近24根回撤", 999.0)
    close_position = _detail_float(details, "收盘区间位置", 0.0)
    lower_wick = _detail_float(details, "下影线", 0.0)
    rsi = _detail_float(details, "RSI", 50.0)
    min_volume = _safe_float(config.get("lifecycle_absorption_min_volume_ratio"), 1.8)
    max_drawdown = _safe_float(config.get("lifecycle_absorption_max_drawdown_pct"), 5.0)
    min_close_position = _safe_float(config.get("lifecycle_absorption_min_close_position"), 52.0)
    min_wick = _safe_float(config.get("lifecycle_absorption_min_lower_wick_pct"), 25.0)

    _add_check(premises, failures, warnings, volume >= min_volume, f"异常放量有效({volume:.2f}x)", f"放量不够异常({volume:.2f}x)", hard=True)
    _add_check(premises, failures, warnings, drawdown <= max_drawdown, f"放量但价格抗跌({drawdown:.2f}%)", f"放量后价格仍明显下跌({drawdown:.2f}%)", hard=True)
    _add_check(premises, failures, warnings, close_position >= min_close_position, f"收盘位置显示承接({close_position:.0f})", f"收盘位置偏弱({close_position:.0f})")
    _add_check(premises, failures, warnings, lower_wick >= min_wick, f"下影线承接有效({lower_wick:.0f}%)", f"下影线承接不足({lower_wick:.0f}%)")
    _add_check(premises, failures, warnings, 32.0 <= rsi <= 66.0, f"RSI位于承接区({rsi:.1f})", f"RSI不在承接区({rsi:.1f})")
    return premises, failures, warnings


def _btc_eth_lead_lag(result: Dict, details: Dict, config: Dict) -> Tuple[List[str], List[str], List[str]]:
    premises, failures, warnings = [], [], []
    direction = str(result.get("direction") or "WAIT").upper()
    anchor = _detail_float(details, "BTC_ETH基准涨幅", 0.0)
    target = _detail_float(details, "目标24H涨幅", 0.0)
    relative = _detail_float(details, "相对强弱", 0.0)
    min_anchor = _safe_float(config.get("lifecycle_leadlag_min_anchor_move_pct"), 1.5)
    min_relative = _safe_float(config.get("lifecycle_leadlag_min_relative_pct"), 2.2)

    anchor_ok = abs(anchor) >= min_anchor
    if direction in {"BUY", "LONG"}:
        relation_ok = relative >= min_relative
    elif direction in {"SELL", "SHORT"}:
        relation_ok = relative <= -min_relative
    else:
        relation_ok = False

    _add_check(premises, failures, warnings, anchor_ok, f"BTC/ETH牵引足够明显({anchor:.2f}%)", f"BTC/ETH基准波动过小({anchor:.2f}%)", hard=True)
    _add_check(premises, failures, warnings, relation_ok, f"目标相对强弱明确({relative:.2f}%)", f"目标相对强弱不够明确({relative:.2f}%)", hard=True)
    _add_check(premises, failures, warnings, abs(target) >= 2.0, f"目标自身波动有效({target:.2f}%)", f"目标自身波动不足({target:.2f}%)")
    return premises, failures, warnings


def _exchange_heat_jump(result: Dict, details: Dict, config: Dict) -> Tuple[List[str], List[str], List[str]]:
    premises, failures, warnings = [], [], []
    rank_pct = _detail_float(details, "排名百分位", 100.0)
    volume = _detail_float(details, "1H量比", 0.0)
    momentum = _detail_float(details, "12H动量", 0.0)
    close_position = _detail_float(details, "收盘区间位置", 50.0)
    max_rank_pct = _safe_float(config.get("lifecycle_heat_max_rank_percentile"), 20.0)
    min_volume = _safe_float(config.get("lifecycle_heat_min_volume_ratio"), 1.6)
    min_momentum = _safe_float(config.get("lifecycle_heat_min_momentum_pct"), 1.2)

    _add_check(premises, failures, warnings, rank_pct <= max_rank_pct, f"成交额排名进入前排({rank_pct:.1f}%)", f"成交额排名不够靠前({rank_pct:.1f}%)", hard=True)
    _add_check(premises, failures, warnings, volume >= min_volume, f"短周期热度跃迁({volume:.2f}x)", f"短周期热度不足({volume:.2f}x)", hard=True)
    _add_check(premises, failures, warnings, momentum >= min_momentum or close_position >= 62.0, f"价格同步响应({momentum:.2f}%/位置{close_position:.0f})", f"只有热度没有价格响应({momentum:.2f}%/位置{close_position:.0f})")
    return premises, failures, warnings


PROFILE_RULES: Dict[str, Callable[[Dict, Dict, Dict], Tuple[List[str], List[str], List[str]]]] = {
    "突破启动": _breakout_start,
    "新高突破": _new_high,
    "单边趋势": _directional_trend,
    "趋势回踩": _trend_pullback,
    "趋势回踩二次启动": _trend_pullback,
    "背离反转": _divergence_reversal,
    "超跌反转": _oversold_reversal,
    "波动率收缩爆发": _volatility_compression,
    "中继再启动": _continuation_restart,
    "假突破回收": _false_breakout_reclaim,
    "相对强弱": _relative_strength_leader,
    "资金费率反转": _funding_reversal,
    "持仓量异常": _open_interest_anomaly,
    "放量承接": _volume_absorption,
    "BTC/ETH牵引": _btc_eth_lead_lag,
    "热度跃迁": _exchange_heat_jump,
}


def apply_strategy_lifecycle_guard(result: Dict, category: str, config: Dict | None = None) -> Dict:
    """Attach premise/failure diagnostics and down-rank invalid strategy states."""
    normalized = dict(result or {})
    config = config or {}
    if config.get("enable_strategy_lifecycle_guard", True) is False:
        return normalized

    details = normalized.get("details")
    if not isinstance(details, dict):
        details = {"状态": str(details or "")}
        normalized["details"] = details

    rule = PROFILE_RULES.get(category)
    if rule is None:
        return normalized

    premises, failures, warnings = rule(normalized, details, config)
    direction = str(normalized.get("direction") or "WAIT").upper()
    if direction not in ACTIVE_DIRECTIONS:
        warnings.append("未出现可执行方向")

    warning_penalty = min(len(warnings) * _safe_float(config.get("lifecycle_warning_penalty"), 4.0), 18.0)
    failure_cap = _safe_float(config.get("lifecycle_failure_score_cap"), 67.0)
    score = _safe_float(normalized.get("score"), 0.0)
    opportunity_score = _safe_float(normalized.get("opportunity_score"), score)

    if failures:
        normalized["passed"] = False
        normalized["score"] = round(min(score, failure_cap), 2)
        normalized["opportunity_score"] = round(min(opportunity_score, failure_cap), 2)
        normalized["opportunity_level"] = _score_to_level(_safe_float(normalized.get("opportunity_score")))
        details["策略状态"] = "失效/禁止入选"
    else:
        adjusted_score = max(0.0, score - warning_penalty)
        normalized["score"] = round(adjusted_score, 2)
        normalized["opportunity_score"] = round(max(0.0, opportunity_score - warning_penalty * 0.55), 2)
        normalized["opportunity_level"] = _score_to_level(_safe_float(normalized.get("opportunity_score")))
        details["策略状态"] = "前提通过" if not warnings else "前提基本通过/降权"

    profile = build_opportunity_profile(
        base_score=normalized.get("score", 0.0),
        direction=normalized.get("direction", "WAIT"),
        volume_24h=normalized.get("volume_24h", 0.0),
        factors=normalized.get("ranking_factors", {}),
        signals=normalized.get("signals", []),
    )
    if failures:
        profile["opportunity_score"] = round(min(_safe_float(profile.get("opportunity_score")), failure_cap), 2)
        profile["opportunity_level"] = _score_to_level(_safe_float(profile.get("opportunity_score")))
    elif warning_penalty > 0:
        profile["opportunity_score"] = round(max(0.0, _safe_float(profile.get("opportunity_score")) - warning_penalty * 0.55), 2)
        profile["opportunity_level"] = _score_to_level(_safe_float(profile.get("opportunity_score")))
    normalized.update(profile)

    details["发挥前提"] = "；".join(premises) if premises else "未满足核心前提"
    details["失效条件"] = "；".join(failures) if failures else "未触发硬失效"
    details["降权提示"] = "；".join(warnings) if warnings else "无"
    normalized["lifecycle"] = {
        "category": category,
        "premises": premises,
        "failures": failures,
        "warnings": warnings,
        "status": details["策略状态"],
    }
    if failures:
        normalized["priority_reason"] = f"{category}失效: {'；'.join(failures[:2])}"
    elif warnings:
        normalized["priority_reason"] = f"{category}降权: {'；'.join(warnings[:2])}"
    return normalized
