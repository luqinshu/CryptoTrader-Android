"""
扫描结果统一机会评分与排序工具。
"""

from typing import Dict, Iterable, List, Optional


ACTIVE_DIRECTIONS = {"BUY", "SELL", "LONG", "SHORT"}


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def build_opportunity_profile(
    base_score: float,
    direction: str = "WAIT",
    volume_24h: float = 0.0,
    factors: Optional[Dict[str, float]] = None,
    signals: Optional[Iterable[str]] = None,
) -> Dict[str, object]:
    """
    将不同策略的原始分数转成统一的“重大机会优先级”评分。
    各策略只需提供基础分和若干质量因子即可。
    """
    normalized_base = max(0.0, min(_safe_float(base_score), 100.0))
    normalized_direction = str(direction or "WAIT").upper()
    normalized_factors = factors or {}
    signal_list = [str(item) for item in (signals or []) if item]

    trend_score = max(0.0, min(_safe_float(normalized_factors.get("trend"), normalized_base), 100.0))
    trigger_score = max(0.0, min(_safe_float(normalized_factors.get("trigger"), normalized_base), 100.0))
    volume_score = max(0.0, min(_safe_float(normalized_factors.get("volume"), 50.0), 100.0))
    location_score = max(0.0, min(_safe_float(normalized_factors.get("location"), normalized_base), 100.0))
    freshness_score = max(0.0, min(_safe_float(normalized_factors.get("freshness"), 55.0), 100.0))
    risk_score = max(0.0, min(_safe_float(normalized_factors.get("risk"), location_score), 100.0))

    liquidity_score = min(max(_safe_float(volume_24h), 0.0) / 20000000.0, 1.0) * 100.0

    weighted_score = (
        normalized_base * 0.36
        + trend_score * 0.16
        + trigger_score * 0.20
        + volume_score * 0.08
        + location_score * 0.10
        + freshness_score * 0.05
        + risk_score * 0.02
        + liquidity_score * 0.03
    )

    if normalized_direction not in ACTIVE_DIRECTIONS:
        weighted_score = min(weighted_score, 69.0)

    if trigger_score < 55:
        weighted_score = min(weighted_score, 76.0)
    if volume_score < 35:
        weighted_score -= 4.0

    opportunity_score = max(0.0, min(weighted_score, 100.0))
    if opportunity_score >= 92:
        level = "S"
    elif opportunity_score >= 84:
        level = "A"
    elif opportunity_score >= 76:
        level = "B"
    elif opportunity_score >= 68:
        level = "C"
    else:
        level = "D"

    priority_reason = " | ".join(signal_list[:3]) if signal_list else f"基础评分 {normalized_base:.1f}"

    return {
        "opportunity_score": round(opportunity_score, 2),
        "opportunity_level": level,
        "priority_reason": priority_reason,
    }


def enrich_scan_result(result: Optional[Dict]) -> Optional[Dict]:
    if not isinstance(result, dict):
        return result

    if "opportunity_score" in result and "opportunity_level" in result:
        return result

    profile = build_opportunity_profile(
        base_score=result.get("score", 0.0),
        direction=result.get("direction") or result.get("side", "WAIT"),
        volume_24h=result.get("volume_24h", 0.0),
        factors=result.get("ranking_factors") or result.get("factors") or {},
        signals=result.get("signals", []),
    )
    result.update(profile)
    return result


def scan_result_sort_key(item: Dict) -> tuple:
    enrich_scan_result(item)
    group_sort_score = _safe_float(item.get("group_sort_score"), -1.0)
    return (
        group_sort_score,
        _safe_float(item.get("opportunity_score"), _safe_float(item.get("score"), 0.0)),
        _safe_float(item.get("score"), 0.0),
        _safe_float(item.get("volume_24h"), 0.0),
        -abs(_safe_float(item.get("price_change_24h"), item.get("change_24h", 0.0))),
    )


def sort_scan_results(results: Iterable[Dict]) -> List[Dict]:
    enriched = [item for item in results if item]
    return sorted(enriched, key=scan_result_sort_key, reverse=True)
