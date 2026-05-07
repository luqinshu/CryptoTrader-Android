"""
多策略共振仪表盘 v1.0
======================
当一个交易对同时被多个扫描策略选中时，信号置信度远高于单策略信号。

核心逻辑：
  1. 按 (symbol, direction) 分组
  2. 统计每个组被多少策略选中
  3. 根据共识深度加分
  4. 标注共振等级

共振加分规则：
  单一引擎         +0    (基准)
  双引擎同向       +8    ★★
  三引擎同向       +14   ★★★
  四引擎同向       +20   ★★★★
  五引擎及以上     +25   ★★★★★

方向分歧处理：
  同一品种 BUY 和 SELL 同时出现 → 降权 (多空分歧, 信号不纯粹)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple


def _safe_float(v, default=0.0):
    try: return float(v)
    except: return default


def compute_resonance(
    results: List[Dict],
    direction_field: str = "direction",
    strategy_field: str = "category",
    symbol_field: str = "symbol",
) -> List[Dict]:
    """
    计算多策略共振并应用到每个结果的 score 和元数据。

    Args:
        results: 扫描结果列表
        direction_field: 结果中表示方向的字段名
        strategy_field: 结果中表示策略名的字段名
        symbol_field: 结果中表示品种的字段名

    Returns:
        处理后的 results (in-place 修改 + 按共振分数重新排序)
    """
    if len(results) < 2:
        for r in results:
            r["resonance_depth"] = 1
            r["resonance_bonus"] = 0.0
            r["resonance_engines"] = [r.get(strategy_field, "?")]
            r["resonance_label"] = "单引擎"
        return results

    # ── 按 (symbol, direction) 分组 ─────────────────────────────────────────
    groups: Dict[Tuple[str, str], List[Dict]] = {}
    for r in results:
        sym = str(r.get(symbol_field, ""))
        d = str(r.get(direction_field, "WAIT")).upper()
        if d not in ("BUY", "SELL", "LONG", "SHORT"):
            continue
        d = "BUY" if d in ("BUY", "LONG") else "SELL"
        key = (sym, d)
        groups.setdefault(key, []).append(r)

    # ── 检测方向冲突 ───────────────────────────────────────────────────────
    sym_dirs: Dict[str, Set[str]] = {}
    for r in results:
        sym = str(r.get(symbol_field, ""))
        d = str(r.get(direction_field, "WAIT")).upper()
        if d in ("BUY", "LONG", "SELL", "SHORT"):
            d = "BUY" if d in ("BUY", "LONG") else "SELL"
            sym_dirs.setdefault(sym, set()).add(d)

    conflict_symbols = {s for s, dirs in sym_dirs.items() if len(dirs) > 1}

    # ── 计算共振 ───────────────────────────────────────────────────────────
    for (sym, d), items in groups.items():
        depth = len(items)
        unique_strategies = list(dict.fromkeys(
            str(it.get(strategy_field, "?")) for it in items
        ))
        has_conflict = sym in conflict_symbols

        # 共振加分
        if depth >= 5:
            bonus = 25.0
            label = "★★★★★ 五引擎共振"
        elif depth >= 4:
            bonus = 20.0
            label = "★★★★ 四引擎共振"
        elif depth >= 3:
            bonus = 14.0
            label = "★★★ 三引擎共振"
        elif depth >= 2:
            bonus = 8.0
            label = "★★ 双引擎共振"
        else:
            bonus = 0.0
            label = "单引擎"

        # 方向冲突惩罚
        if has_conflict:
            bonus *= 0.5
            label += " ⚠方向分歧"

        # 应用到组内每个结果
        for r in items:
            old_score = _safe_float(r.get("score", 50))
            r["resonance_depth"] = depth
            r["resonance_bonus"] = round(bonus, 1)
            r["resonance_engines"] = unique_strategies
            r["resonance_label"] = label
            r["resonance_conflict"] = has_conflict
            r["score"] = round(min(100, old_score + bonus), 1)
            # 同时提升 composite_score
            if "composite_score" in r:
                r["composite_score"] = round(min(100, _safe_float(r["composite_score"]) + bonus * 0.7), 1)
            # 提升 opportunity_score
            if "opportunity_score" in r:
                r["opportunity_score"] = round(min(100, _safe_float(r["opportunity_score"]) + bonus * 0.8), 1)
            # 更新 details
            d_details = r.setdefault("details", {})
            d_details["共振引擎"] = " + ".join(unique_strategies)
            d_details["共振深度"] = str(depth)
            d_details["共振等级"] = label

    # ── 按共振调整后的分数排序 ─────────────────────────────────────────────
    results.sort(
        key=lambda r: (
            _safe_float(r.get("resonance_depth", 0)),   # 共振深度优先
            _safe_float(r.get("score", 0)),              # 分数次之
        ),
        reverse=True,
    )

    return results


def build_resonance_summary(results: List[Dict]) -> Dict[str, Any]:
    """
    生成共振摘要 — 供 UI 仪表盘展示。
    """
    depths = [r.get("resonance_depth", 1) for r in results]
    conflicts = sum(1 for r in results if r.get("resonance_conflict", False))

    # Top resonating symbols
    seen = set()
    top_resonance = []
    for r in results:
        sym = r.get("symbol", "")
        if sym in seen:
            continue
        seen.add(sym)
        depth = r.get("resonance_depth", 1)
        engines = r.get("resonance_engines", [])
        top_resonance.append({
            "symbol": sym,
            "depth": depth,
            "engines": engines,
            "score": r.get("score", 0),
            "direction": r.get("direction", "WAIT"),
        })

    return {
        "total_results": len(results),
        "max_resonance_depth": max(depths) if depths else 1,
        "avg_resonance_depth": round(sum(depths) / max(len(depths), 1), 2),
        "direction_conflicts": conflicts,
        "depth_distribution": {
            "1_engine": sum(1 for d in depths if d == 1),
            "2_engines": sum(1 for d in depths if d == 2),
            "3_engines": sum(1 for d in depths if d == 3),
            "4plus_engines": sum(1 for d in depths if d >= 4),
        },
        "top_resonance": top_resonance[:10],
    }
