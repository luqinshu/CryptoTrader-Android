"""
截面排序引擎 v1.0
==================
将全市场扫描结果从"通过/不通过"二元判断升级为"全市场相对排名"。

核心思路：
  不是问 "这个币好不好"（绝对值），
  而是问 "这个币在全市场中排第几"（相对值）。

流程：
  1. 输入: 所有品种的扫描结果 (passed=True/False + 多因子评分)
  2. 每个因子做截面标准化 (winsorize 极值 → z-score → [0,100])
  3. 加权合成 composite_score
  4. 按 composite_score 排序
  5. 输出: top N 做多候选, bottom N 做空候选, 各品种的百分位排名

因子体系 (6维度):
  trend_quality   — 趋势质量 (ADX + EMA排列 + 斜率)
  momentum        — 动量 (短期涨跌 + RSI位置 + MACD)
  volume_quality  — 量能质量 (24h成交额 + 放量比例 + 缩量系数)
  volatility      — 波动率适配 (ATR% 并非越低越好, 适中最优)
  liquidity       — 流动性 (24h成交额绝对值的对数)
  btcrelative     — 相对BTC强弱 (山寨涨跌 - BTC涨跌, 多空不同向)
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ── 因子定义 ────────────────────────────────────────────────────────────────
FACTOR_DEFS = {
    "trend_quality": {
        "weight": 0.22,
        "label": "趋势质量",
        "polarity": 1,
        "extract": lambda r: _safe_number(
            (r.get("ranking_factors", {}) or {}).get("trend", r.get("factor_scores", {}).get("daily_adx_score", r.get("score", 50))),
            default=50,
        ),
    },
    "momentum": {
        "weight": 0.20,
        "label": "动量",
        "polarity": 1,
        "extract": lambda r: _safe_number(
            (r.get("ranking_factors", {}) or {}).get("trigger",
                r.get("factor_scores", {}).get("hourly_macd_score", r.get("score", 50))),
            default=50,
        ),
    },
    "volume_quality": {
        "weight": 0.16,
        "label": "量能质量",
        "polarity": 1,
        "extract": lambda r: _safe_number(
            (r.get("ranking_factors", {}) or {}).get("volume",
                r.get("factor_scores", {}).get("hourly_dryup_score",
                _safe_number(r.get("volume_24h", 0)) / 1e7 * 50)),
            default=50,
        ),
    },
    "volatility": {
        "weight": 0.16,
        "label": "波动适配",
        "polarity": -1,
        "extract": lambda r: _safe_number(
            (r.get("details", {}) or {}).get("4H_ATR%",
                (r.get("details", {}) or {}).get("日线ATR",
                (r.get("details", {}) or {}).get("ATR", 4.0))),
            default=4.0,
        ),
    },
    "liquidity": {
        "weight": 0.14,
        "label": "流动性",
        "polarity": 1,
        "extract": lambda r: min(_safe_number(r.get("volume_24h", 1e6), default=1e6), 1e11),
    },
    "btc_relative": {
        "weight": 0.12,
        "label": "相对BTC",
        "polarity": 1,
        "extract": lambda r: _compute_btc_relative(r),
    },
}


def _safe_number(v, default=0.0):
    try:
        s = str(v).replace('%', '').strip()
        return float(s)
    except (TypeError, ValueError):
        return default


def _compute_btc_relative(result: Dict) -> float:
    """计算相对BTC强弱: 山寨24H涨跌 - BTC24H涨跌, 多空方向校正"""
    alt_chg = _safe_number(result.get("price_change_24h", result.get("change_24h", 0)))
    btc_chg = _safe_number((result.get("details", {}) or {}).get("BTC24H%",
               result.get("btc_24h_move", 0)))
    direction = str(result.get("direction", "WAIT")).upper()

    # 多头信号: 山寨>BTC = 好; 空头信号: 山寨<BTC = 好
    raw_rs = alt_chg - btc_chg
    if direction in ("SELL", "SHORT"):
        raw_rs = -raw_rs
    return raw_rs


def winsorize(values: List[float], lower_pct: float = 0.02, upper_pct: float = 0.98) -> List[float]:
    """缩尾处理: 把极值拉到分位数边界"""
    if len(values) < 5:
        return list(values)
    arr = np.array(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 3:
        return list(values)
    lo = np.quantile(arr, lower_pct)
    hi = np.quantile(arr, upper_pct)
    return [max(lo, min(hi, float(v))) if np.isfinite(float(v)) else lo for v in values]


def robust_zscore(values: List[float]) -> List[float]:
    """稳健 z-score: 用中位数和 MAD 而非均值/标准差。
    返回与输入等长的列表，NaN 值返回 0.0（保持索引对齐）。"""
    n = len(values)
    if n < 3:
        return [0.0] * n
    arr = np.array(values, dtype=float)
    fin_mask = np.isfinite(arr)
    arr_clean = arr[fin_mask]
    if len(arr_clean) < 3:
        return [0.0] * n
    med = np.median(arr_clean)
    mad = np.median(np.abs(arr_clean - med))
    if mad < 1e-9:
        return [0.0] * n
    result = [0.0] * n
    for i in range(n):
        if fin_mask[i]:
            result[i] = (float(arr[i]) - med) / (mad * 1.4826)
    return result


def normalize_to_0_100(zscores: List[float]) -> List[float]:
    """将 z-score 映射到 0~100 区间"""
    if not zscores:
        return []
    lo, hi = min(zscores), max(zscores)
    if hi - lo < 1e-9:
        return [50.0] * len(zscores)
    return [max(0.0, min(100.0, (z - lo) / (hi - lo) * 100.0)) for z in zscores]


def cross_sectional_rank(
    results: List[Dict],
    factor_weights: Optional[Dict[str, float]] = None,
    min_samples: int = 10,
) -> List[Dict]:
    """
    主入口: 对全市场扫描结果进行截面排序。

    Args:
        results:       scan_all_symbols 输出的 all_opportunities 列表
        factor_weights: 自定义因子权重 (None=使用默认)
        min_samples:   最少品种数 (不足则退回原始排序)

    Returns:
        排序后的 results，每项新增:
          - composite_score:  综合截面评分 (0~100)
          - percentile:       全市场排名百分位
          - factor_scores:    各因子得分 {factor_name: 0~100}
          - long_candidate:   top 25% = True
          - short_candidate:  bottom 25% = True
    """
    if len(results) < min_samples:
        # 样本不足，降级为原始分数排序
        results.sort(key=lambda r: float(r.get("score", 0) or 0), reverse=True)
        for i, r in enumerate(results):
            r["composite_score"] = float(r.get("score", 50) or 50)
            r["percentile"] = round((1 - i / max(len(results), 1)) * 100, 1)
            r["cross_sectional"] = {"status": "fallback", "reason": f"样本不足({len(results)}<{min_samples})"}
        return results

    # ── Step 1: 提取各因子的原始值 ─────────────────────────────────────────
    weights = factor_weights or {name: f["weight"] for name, f in FACTOR_DEFS.items()}
    raw_factors: Dict[str, List[float]] = {}
    for name, fdef in FACTOR_DEFS.items():
        if name not in weights:
            continue
        raw_factors[name] = [fdef["extract"](r) for r in results]

    # ── Step 2: Winsorize → 稳健 z-score → 归一化到 0~100 ────────────────
    normalized: Dict[str, List[float]] = {}
    for name, raw_vals in raw_factors.items():
        polarity = FACTOR_DEFS[name]["polarity"]
        winsor = winsorize(raw_vals)
        zs = robust_zscore(winsor)
        # 极性调整: 负极性因子(如波动率)反转方向
        if polarity < 0:
            zs = [-z for z in zs]
        n100 = normalize_to_0_100(zs)
        # 补回可能因 NaN 截断的缺失值
        while len(n100) < len(results):
            n100.append(50.0)
        normalized[name] = n100[:len(results)]

    # ── Step 3: 加权合成 composite_score ───────────────────────────────────
    weight_sum = sum(weights.get(name, 0) for name in normalized)
    if weight_sum <= 0:
        weight_sum = 1.0

    for i, r in enumerate(results):
        weighted = 0.0
        factor_detail = {}
        for name in normalized:
            w = weights.get(name, 0)
            s = normalized[name][i]
            weighted += s * w
            factor_detail[name] = {
                "score": round(s, 1),
                "raw": round(raw_factors[name][i], 3) if i < len(raw_factors[name]) else 0,
                "label": FACTOR_DEFS[name]["label"],
            }
        composite = round(weighted / weight_sum, 1)
        r["composite_score"] = composite
        r["factor_detail"] = factor_detail

    # ── Step 4: 排序 + 百分位 ──────────────────────────────────────────────
    results.sort(key=lambda r: float(r.get("composite_score", 0) or 0), reverse=True)
    n = len(results)
    top_cutoff = max(1, int(n * 0.25))
    bot_cutoff = max(1, int(n * 0.25))

    for i, r in enumerate(results):
        r["rank"] = i + 1
        r["percentile"] = round((1 - i / n) * 100, 1)
        r["long_candidate"] = i < top_cutoff
        r["short_candidate"] = i >= n - bot_cutoff
        # 统计显著性标签
        if r["composite_score"] >= 80:
            r["confidence_label"] = "⭐⭐⭐ 高置信"
        elif r["composite_score"] >= 65:
            r["confidence_label"] = "⭐⭐ 中置信"
        elif r["composite_score"] >= 50:
            r["confidence_label"] = "⭐ 低置信"
        else:
            r["confidence_label"] = "观察"

    return results


def generate_long_short_pairs(
    ranked_results: List[Dict],
    top_n: int = 8,
    bottom_n: int = 8,
    min_composite: float = 55.0,
) -> Dict[str, Any]:
    """
    从排名结果中生成做多/做空对。

    Returns:
        {"longs": [...], "shorts": [...], "pairs": [...], "summary": str}
    """
    longs = [
        {
            "symbol": r.get("symbol", ""),
            "composite_score": r.get("composite_score", 0),
            "percentile": r.get("percentile", 0),
            "direction": r.get("direction", "N/A"),
            "category": r.get("category", ""),
            "top_factors": _top_factors(r),
        }
        for r in ranked_results
        if r.get("long_candidate") and r.get("composite_score", 0) >= min_composite
    ][:top_n]

    shorts = [
        {
            "symbol": r.get("symbol", ""),
            "composite_score": r.get("composite_score", 0),
            "percentile": r.get("percentile", 0),
            "direction": r.get("direction", "N/A"),
            "category": r.get("category", ""),
            "top_factors": _top_factors(r),
        }
        for r in ranked_results
        if r.get("short_candidate") and r.get("composite_score", 0) <= 100 - min_composite
    ][-bottom_n:][::-1]  # 最差的在最后，反转

    # 配对: 每个 long 配一个 short (如果方向相反)
    pairs = []
    for lo, sh in zip(longs, shorts):
        if lo["direction"] != sh["direction"] and lo["direction"] in ("BUY", "LONG"):
            pairs.append({
                "long": lo["symbol"],
                "short": sh["symbol"],
                "spread_score": round(lo["composite_score"] - sh["composite_score"], 1),
            })

    return {
        "longs": longs,
        "shorts": shorts,
        "pairs": pairs,
        "summary": (
            f"做多候选 {len(longs)} 个 | 做空候选 {len(shorts)} 个 | "
            f"配对 {len(pairs)} 组"
        ),
    }


def _top_factors(result: Dict, n: int = 3) -> List[str]:
    """提取最强的 N 个因子名"""
    fd = result.get("factor_detail", {})
    if not fd:
        return []
    sorted_f = sorted(fd.items(), key=lambda x: x[1].get("score", 0), reverse=True)
    return [f"{info['label']}({info['score']:.0f})" for name, info in sorted_f[:n]]
