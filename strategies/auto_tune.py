#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
策略参数自动优化器 v1.0
=======================
使用网格搜索对策略参数进行自动优化，支持样本外验证。

功能：
  1. 定义参数搜索空间 → 网格搜索组合
  2. 每种组合回测 → 收集指标
  3. 样本外验证: 训练期优化 / 测试期验证
  4. 多目标评分: Sharpe + 胜率 + 盈亏比 + 总收益 的加权组合
  5. 输出 Top N 参数组合及完整性能报告

调用方式：
  from strategies.auto_tune import tune_strategy
  results = tune_strategy(
      strategy_class, param_grid, inst_id, okx_client,
      train_start, train_end, test_start, test_end
  )
"""

from __future__ import annotations

import itertools
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── 数据结构 ────────────────────────────────────────────────────────────────

@dataclass
class TuneTrial:
    params: Dict[str, Any]
    train_metrics: Dict[str, float]
    test_metrics: Optional[Dict[str, float]] = None
    composite_score: float = 0.0
    rank: int = 0

    def to_dict(self):
        return {
            "params": self.params,
            "train": self.train_metrics,
            "test": self.test_metrics,
            "composite_score": round(self.composite_score, 2),
            "rank": self.rank,
        }


# ── 默认参数搜索空间 (可被策略覆盖) ─────────────────────────────────────────

DEFAULT_PARAM_GRID: Dict[str, List[Any]] = {
    # 三分钟策略常用参数
    "m3_pullback_min_pct": [0.08, 0.12, 0.15, 0.20, 0.25],
    "m3_pullback_max_pct": [1.0, 1.5, 2.0, 2.5, 3.5],
    "m3_stabilization_bars": [2, 3],
    "m3_resumption_bars": [2, 3, 4],
    "volume_confirm_ratio": [0.35, 0.50, 0.65],
    "h1_rsi_min_long": [35, 40, 45],
    "h1_rsi_max_long": [65, 70, 75],
}

# 常用指标权重
DEFAULT_METRIC_WEIGHTS = {
    "sharpe_ratio": 0.25,
    "win_rate": 0.20,
    "profit_factor": 0.20,
    "total_return": 0.20,
    "max_drawdown": -0.15,  # 负权=越小越好
}


# ── 评分函数 ────────────────────────────────────────────────────────────────

def compute_composite_score(
    metrics: Dict[str, float],
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """
    多目标加权评分。

    指标标准化: 用 rank-based 归一化 (不是 min-max，防止离群值主导)
    """
    return _safe_score(metrics, weights or DEFAULT_METRIC_WEIGHTS)


def _safe_score(metrics, weights):
    """单组指标的直接评分"""
    score = 0.0
    w_sum = 0.0
    for key, w in weights.items():
        v = metrics.get(key, 0)
        if v is None or not np.isfinite(v):
            continue
        score += v * w
        w_sum += abs(w)
    return score / max(w_sum, 1e-9)


def rank_normalize(trials: List[TuneTrial], weights: Dict[str, float]) -> List[TuneTrial]:
    """
    对多个 trial 做 rank-based 归一化后计算 composite_score。

    原理：
      每个指标按 rank 映射到 [0, 100]
      正权指标：rank 越高分越高
      负权指标（如最大回撤）：rank 越高分越低
    """
    n = len(trials)
    if n <= 1:
        for t in trials:
            t.composite_score = _safe_score(t.train_metrics, weights)
        return trials

    for key in weights:
        # 收集所有 trial 的该指标值
        vals = []
        for t in trials:
            v = t.train_metrics.get(key)
            if v is not None and np.isfinite(v):
                vals.append((t, v))

        if not vals:
            continue

        # 排序 → rank
        vals.sort(key=lambda x: x[1])
        for rank_idx, (trial, _) in enumerate(vals):
            rank_score = (rank_idx / (len(vals) - 1)) * 100.0 if len(vals) > 1 else 50.0
            # 负权: 反转 rank
            w = weights[key]
            if w < 0:
                rank_score = 100.0 - rank_score
            trial.train_metrics[f"{key}_rank"] = round(rank_score, 1)

    # 用 rank 值计算 composite
    for t in trials:
        score = 0.0
        w_sum = 0.0
        for key, w in weights.items():
            rank_key = f"{key}_rank"
            v = t.train_metrics.get(rank_key)
            if v is None:
                continue
            score += v * abs(w)
            w_sum += abs(w)
        t.composite_score = round(score / max(w_sum, 1e-9), 2) if w_sum > 0 else 0.0

    trials.sort(key=lambda t: t.composite_score, reverse=True)
    for i, t in enumerate(trials, 1):
        t.rank = i

    return trials


# ── 主优化函数 ──────────────────────────────────────────────────────────────

def tune_strategy(
    strategy_class,
    param_grid: Dict[str, List[Any]],
    inst_id: str,
    okx_client,
    train_start: str,
    train_end: str,
    test_start: Optional[str] = None,
    test_end: Optional[str] = None,
    metric_weights: Optional[Dict[str, float]] = None,
    progress_callback: Optional[Callable] = None,
    max_trials: int = 100,
) -> List[TuneTrial]:
    """
    主优化入口。

    Args:
        strategy_class:  策略类 (不是实例)
        param_grid:      参数搜索空间 {key: [values]}
        inst_id:         交易对
        okx_client:      OKX 客户端
        train_start/end: 训练期
        test_start/end:  测试期 (None=跳过)
        metric_weights:  指标权重
        progress_callback: (current, total, best_score)
        max_trials:      最多试验数 (超限则截断)

    Returns:
        Top N TuneTrial 列表 (按 composite_score 降序)
    """
    weights = metric_weights or DEFAULT_METRIC_WEIGHTS

    # 生成参数组合
    keys = list(param_grid.keys())
    all_combos = list(itertools.product(*[param_grid[k] for k in keys]))
    total = len(all_combos)

    if total > max_trials:
        # 随机抽样
        np.random.seed(42)
        indices = np.random.choice(total, max_trials, replace=False)
        all_combos = [all_combos[i] for i in sorted(indices)]
        total = max_trials

    logger.info(f"[自动优化] {inst_id}: {len(keys)}参数 × {total}组合")

    trials: List[TuneTrial] = []

    for idx, combo in enumerate(all_combos):
        params = dict(zip(keys, combo))
        strategy_config = {k: v for k, v in params.items()}

        # ── 训练期回测 ──────────────────────────────────────────────────
        train_metrics = _run_single_backtest(
            strategy_class, inst_id, okx_client,
            train_start, train_end, strategy_config,
        )
        if not train_metrics:
            continue

        trial = TuneTrial(params=params, train_metrics=train_metrics)

        # ── 测试期回测 ──────────────────────────────────────────────────
        if test_start and test_end:
            test_metrics = _run_single_backtest(
                strategy_class, inst_id, okx_client,
                test_start, test_end, strategy_config,
            )
            trial.test_metrics = test_metrics

        trials.append(trial)

        if progress_callback and idx % max(1, total // 20) == 0:
            progress_callback(idx + 1, total, 0)

    # ── 评分 + 排名 ─────────────────────────────────────────────────────
    trials = rank_normalize(trials, weights)

    if progress_callback:
        progress_callback(total, total, trials[0].composite_score if trials else 0)

    return trials[:50]  # 返回 top 50


def _run_single_backtest(strategy_class, inst_id, okx_client, start, end, config) -> Optional[Dict]:
    """执行单次回测并提取指标"""
    try:
        from src.backtest.engine import BacktestEngine
        engine = BacktestEngine(okx_client, initial_capital=10000.0)
        result = engine.run(
            strategy_class(), inst_id=inst_id,
            start_date=start, end_date=end,
            config=config, bar=config.get("bar", "3m"),
        )
        if result is None:
            return None

        return {
            "sharpe_ratio": float(getattr(result, "sharpe_ratio", 0) or 0),
            "win_rate": float(getattr(result, "stats", {}).get("win_rate", 0) or 0),
            "profit_factor": float(getattr(result, "stats", {}).get("profit_factor", 0) or 0),
            "total_return": float(getattr(result, "total_return", 0) or 0),
            "max_drawdown": float(getattr(result, "max_drawdown", 0) or 0),
            "total_trades": int(getattr(result, "total_trades", 0) or 0),
            "final_capital": float(getattr(result, "final_capital", 0) or 0),
        }
    except Exception as e:
        logger.error(f"[自动优化] 回测异常: {e}")
        return None


def quick_tune(
    strategy_class,
    param_grid: Dict[str, List[Any]],
    inst_id: str = "BTC-USDT-SWAP",
    okx_client=None,
    train_start: str = "2025-01-01",
    train_end: str = "2025-06-01",
    test_start: str = "2025-06-01",
    test_end: str = "2025-09-01",
) -> List[TuneTrial]:
    """
    快速优化（使用 OKX 实时数据，无需单币种数据库）。

    要求: okx_client 已初始化且有网络连接。
    """
    if okx_client is None:
        from src.api.okx_client import OKXClient
        okx_client = OKXClient()

    return tune_strategy(
        strategy_class=strategy_class,
        param_grid=param_grid,
        inst_id=inst_id,
        okx_client=okx_client,
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
    )


def print_top_trials(trials: List[TuneTrial], n: int = 10):
    """打印 Top N 参数组合"""
    logger.info(f"\n{'='*80}")
    logger.info(f"  Top {n} 参数组合")
    logger.info(f"{'='*80}")
    for t in trials[:n]:
        logger.info(f"\n  #{t.rank}  composite={t.composite_score}")
        logger.info(f"  参数: {t.params}")
        train = t.train_metrics
        logger.info(f"  训练: Sharpe={train.get('sharpe_ratio',0):.2f} "
              f"胜率={train.get('win_rate',0)*100:.0f}% "
              f"盈亏比={train.get('profit_factor',0):.2f} "
              f"收益={train.get('total_return',0):+.2f}% "
              f"回撤={train.get('max_drawdown',0):.2f}% "
              f"交易{train.get('total_trades',0)}笔")
        if t.test_metrics:
            test = t.test_metrics
            logger.info(f"  测试: Sharpe={test.get('sharpe_ratio',0):.2f} "
                  f"胜率={test.get('win_rate',0)*100:.0f}% "
                  f"盈亏比={test.get('profit_factor',0):.2f} "
                  f"收益={test.get('total_return',0):+.2f}%")
    logger.info(f"{'='*80}\n")
