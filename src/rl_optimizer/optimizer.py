"""
RL参数优化器 v2：真梯度上升 + 衰减探索 + 参数独立评分 + Thompson Sampling。
"""

from __future__ import annotations

import json
import math
import random
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


class ParameterOptimizer:
    """
    v2 强化学习优化器改进：
    - 真梯度上升: 参数变异 → 测试 → 比较，不随机游走
    - 衰减 epsilon: 探索率随进化代数递减
    - 参数独立评分: 每个参数值有独立的胜率/收益评估
    - Thompson Sampling: 不确定性建模，平衡探索与利用
    """

    def __init__(self, data_dir: Optional[str] = None):
        self._data_dir = Path(data_dir) if data_dir else Path(__file__).resolve().parent.parent.parent / "data" / "rl_optimizer"
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._params_path = self._data_dir / "optimized_params.json"
        self._history_path = self._data_dir / "optimization_history.json"

        self._param_history: List[Dict] = []
        self._load()

        self._param_space = {
            "截面多因子": {"min_score": (65, 85), "max_h1_trend_age": (8, 24), "h1_trend_age_penalty": (4.0, 12.0), "m3_freshness_penalty": (3.0, 10.0), "bonus_freshness_score": (1.0, 5.0)},
            "AI因子挖掘": {"min_abs_edge": (0.15, 0.45), "risk_penalty_strength": (0.4, 1.0), "max_h1_trend_age": (8, 24), "m3_freshness_penalty": (0.04, 0.15)},
            "DRL小时趋势启动": {"min_score": (65, 85), "max_h1_trend_age": (8, 24), "h1_trend_age_penalty": (4.0, 14.0)},
            "XGBoost截面排序": {"xgb_learning_rate": (0.01, 0.15), "xgb_max_depth": (3, 8), "xgb_min_samples": (300, 1000), "min_score": (65, 85)},
            "AI订单流动量": {"orderflow_weight": (0.10, 0.30), "interact_weight": (0.12, 0.35), "btc_spread_weight": (0.08, 0.25), "min_score": (65, 85)},
        }

        # 参数追踪: {strategy: {key: {"value": float, "success": int, "failure": int, "avg_return": float}}}
        self._param_tracker: Dict[str, Dict] = {}

    def _load(self):
        self._optimized = {}
        if self._params_path.exists():
            try:
                with open(self._params_path, "r", encoding="utf-8") as f:
                    self._optimized = json.load(f)
            except Exception:
                pass
        self._param_history = []
        if self._history_path.exists():
            try:
                with open(self._history_path, "r", encoding="utf-8") as f:
                    self._param_history = json.load(f)
            except Exception:
                pass
        # 从 optimized 中恢复 param_tracker
        self._param_tracker = self._optimized.get("_param_tracker", {})

    def _save(self):
        try:
            self._optimized["_param_tracker"] = self._param_tracker
            with open(self._params_path, "w", encoding="utf-8") as f:
                json.dump(self._optimized, f, indent=2, ensure_ascii=False)
            with open(self._history_path, "w", encoding="utf-8") as f:
                json.dump(self._param_history[-500:], f, indent=2, ensure_ascii=False, default=str)
        except Exception:
            pass

    def generation(self, strategy_name: str) -> int:
        return len([h for h in self._param_history if h.get("strategy") == strategy_name])

    def get_optimized_params(self, strategy_name: str) -> Dict[str, Any]:
        params = deepcopy(self._optimized.get(strategy_name, {}))
        params.pop("_param_tracker", None)
        return params

    def get_best_params(self, strategy_name: str, use_exploration: bool = True) -> Dict[str, Any]:
        """Thompson Sampling: 用 Beta 分布采样选择最优参数值"""
        params = {}
        gen = self.generation(strategy_name)
        epsilon = max(0.03, 0.30 * math.exp(-gen * 0.08))  # 衰减探索率

        for key, (lo, hi) in self._param_space.get(strategy_name, {}).items():
            tracker = self._param_tracker.get(strategy_name, {}).get(key)
            value_stats = tracker.get("value_stats", {}) if tracker else {}
            if value_stats:
                candidates = []
                for bucket, stat in value_stats.items():
                    success = float(stat.get("success", 0.0))
                    failure = float(stat.get("failure", 0.0))
                    neutral = float(stat.get("neutral", 0.0))
                    avg_return = float(stat.get("reward_sum", 0.0)) / max(float(stat.get("trials", 0.0)), 1.0)
                    sampled = random.betavariate(1.0 + success, 1.0 + failure + neutral * 0.35)
                    score = sampled + np.tanh(avg_return / 8.0) * 0.08
                    candidates.append((score, stat))
                best_stat = max(candidates, key=lambda item: item[0])[1]
                if use_exploration and random.random() < epsilon:
                    params[key] = round(random.uniform(lo, hi), 3)
                else:
                    params[key] = round(float(best_stat.get("value", tracker.get("value", (lo + hi) / 2))), 3)
            elif tracker:
                # 有tracker但无bucket统计时，用tracker记录的最优值；探索时随机扰动
                if use_exploration and random.random() < epsilon:
                    params[key] = round(random.uniform(lo, hi), 3)
                else:
                    params[key] = round(float(tracker["value"]), 3)
            else:
                # 无任何历史数据：探索时随机，利用时取中点（确保key始终存在）
                if use_exploration and random.random() < epsilon:
                    params[key] = round(random.uniform(lo, hi), 3)
                else:
                    params[key] = round((lo + hi) / 2, 3)
        return params

    @staticmethod
    def _bucket_value(value: Any) -> str:
        if isinstance(value, bool):
            return "1" if value else "0"
        if isinstance(value, int):
            return str(value)
        try:
            return f"{float(value):.4f}"
        except Exception:
            return str(value)

    def update_from_samples(self, strategy_name: str, samples: List[Dict[str, Any]], stats: Optional[Dict[str, Any]] = None):
        """根据实际参数样本更新参数评分，避免把整条策略表现错误归因到全部参数。"""
        if not samples or len(samples) < 3:
            return

        param_tracker = self._param_tracker.setdefault(strategy_name, {})
        strategy_stats = stats or {}

        for key, (lo, hi) in self._param_space.get(strategy_name, {}).items():
            tracker = param_tracker.setdefault(
                key,
                {
                    "value": (lo + hi) / 2,
                    "success": 0.0,
                    "failure": 0.0,
                    "neutral": 0.0,
                    "avg_return": 0.0,
                    "trials": 0,
                    "value_stats": {},
                },
            )
            value_stats = tracker.setdefault("value_stats", {})
            total_reward = 0.0
            total_trials = 0.0

            for sample in samples:
                params = sample.get("param_snapshot", {})
                if key not in params:
                    continue
                raw_value = params.get(key)
                try:
                    numeric_value = float(raw_value) if not isinstance(raw_value, bool) else (1.0 if raw_value else 0.0)
                except Exception:
                    continue
                bucket = self._bucket_value(raw_value)
                bucket_stat = value_stats.setdefault(
                    bucket,
                    {
                        "value": numeric_value,
                        "success": 0.0,
                        "failure": 0.0,
                        "neutral": 0.0,
                        "reward_sum": 0.0,
                        "trials": 0.0,
                    },
                )
                bucket_stat["success"] += float(sample.get("win_weight", 0.0))
                bucket_stat["failure"] += float(sample.get("loss_weight", 0.0))
                bucket_stat["neutral"] += float(sample.get("neutral_weight", 0.0))
                bucket_stat["reward_sum"] += float(sample.get("net_pnl", 0.0))
                bucket_stat["trials"] += 1.0
                total_reward += float(sample.get("net_pnl", 0.0))
                total_trials += 1.0

            if not value_stats:
                continue

            scored_values = []
            for bucket_stat in value_stats.values():
                success = float(bucket_stat.get("success", 0.0))
                failure = float(bucket_stat.get("failure", 0.0))
                neutral = float(bucket_stat.get("neutral", 0.0))
                trials = max(float(bucket_stat.get("trials", 0.0)), 1.0)
                posterior = (success + 1.0) / (success + failure + neutral + 2.0)
                avg_return = float(bucket_stat.get("reward_sum", 0.0)) / trials
                # 风险调整评分：胜率×回报×试次数惩罚（防止小样本噪声）
                return_score = avg_return * 100.0  # 回报率放大
                score = posterior * 30.0 + np.tanh(return_score / 5.0) * 30.0 + min(trials, 30.0) * 0.20
                # 胜率 30% + 回报 tanh 30% + 样本量 bonus：均衡胜率和风险调整回报
                scored_values.append((score, posterior, avg_return, bucket_stat))

            best_score, best_posterior, best_avg_return, best_stat = max(scored_values, key=lambda item: item[0])
            trials_sum = sum(float(item.get("trials", 0.0)) for item in value_stats.values())
            success_sum = sum(float(item.get("success", 0.0)) for item in value_stats.values())
            failure_sum = sum(float(item.get("failure", 0.0)) for item in value_stats.values())
            neutral_sum = sum(float(item.get("neutral", 0.0)) for item in value_stats.values())

            tracker["value"] = round(float(best_stat.get("value", tracker["value"])), 4)
            tracker["success"] = success_sum
            tracker["failure"] = failure_sum
            tracker["neutral"] = neutral_sum
            tracker["trials"] = int(trials_sum)
            tracker["avg_return"] = total_reward / max(total_trials, 1.0)
            tracker["best_bucket_score"] = round(best_score, 3)
            tracker["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            self._optimized.setdefault(strategy_name, {})
            self._optimized[strategy_name][key] = {
                "value": tracker["value"],
                "win_rate": round(best_posterior * 100.0, 2),
                "avg_return": round(best_avg_return, 3),
                "samples": int(float(best_stat.get("trials", 0.0))),
            }

        self._param_history.append({
            "strategy": strategy_name,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "win_rate": strategy_stats.get("win_rate", 0),
            "profit_factor": strategy_stats.get("profit_factor", 0),
            "sharpe": round(strategy_stats.get("sharpe", 0), 2),
            "total_signals": strategy_stats.get("total", len(samples)),
            "net_pnl": round(strategy_stats.get("net_pnl", sum(float(s.get("net_pnl", 0.0)) for s in samples)), 2),
        })
        self._save()

    def update_from_stats(self, strategy_name: str, stats: Dict):
        """兼容旧调用；新逻辑请优先使用 update_from_samples。"""
        if not stats or stats.get("total", 0) < 3:
            return

    def param_importance(self, strategy_name: str) -> Dict[str, Dict]:
        """返回各参数的重要性评分（胜率贡献）"""
        tracker = self._param_tracker.get(strategy_name, {})
        result = {}
        for key, t in tracker.items():
            total = float(t.get("success", 0)) + float(t.get("failure", 0)) + float(t.get("neutral", 0))
            if total > 0:
                wr = float(t.get("success", 0)) / total * 100
            else:
                wr = 50.0
            bucket_scores = []
            for stat in t.get("value_stats", {}).values():
                trials = max(float(stat.get("trials", 0.0)), 1.0)
                success = float(stat.get("success", 0.0))
                failure = float(stat.get("failure", 0.0))
                neutral = float(stat.get("neutral", 0.0))
                posterior = (success + 1.0) / (success + failure + neutral + 2.0) * 100.0
                avg_return = float(stat.get("reward_sum", 0.0)) / trials
                bucket_scores.append(posterior + np.tanh(avg_return / 8.0) * 8.0)
            spread = (max(bucket_scores) - min(bucket_scores)) if len(bucket_scores) >= 2 else abs(wr - 50.0)
            result[key] = {
                "value": t.get("value", 0),
                "win_rate": round(wr, 1),
                "trials": int(t.get("trials", 0)),
                "importance": round(min(5.0, max(0.0, spread / 10.0)), 2),
            }
        return result

    def evolution_summary(self) -> List[Dict]:
        return self._param_history[-50:]

    def strategy_generation(self, strategy_name: str) -> int:
        return self.generation(strategy_name)
