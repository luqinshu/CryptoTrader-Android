"""
信号追踪器 v2：多时间窗口验证 + 波动率归一化 + 贝叶斯胜率估计。
"""

from __future__ import annotations

import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


class SignalTracker:
    """追踪扫描信号，多时间窗口对比实际走势验证"""

    # 多时间窗口验证点：(小时, 权重)
    VALIDATION_WINDOWS = [(2, 0.25), (6, 0.35), (24, 0.25), (72, 0.15)]

    def __init__(self, data_dir: Optional[str] = None):
        self._data_dir = Path(data_dir) if data_dir else Path(__file__).resolve().parent.parent.parent / "data" / "rl_signals"
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._signals_path = self._data_dir / "signal_history.json"
        self._signals: List[Dict] = []
        self._stats_cache: Dict[str, Any] = {}  # "strategy:days" -> stats
        self._samples_cache: Dict[str, Any] = {}
        self._cache_dirty = True
        self._load()

    def _load(self):
        if self._signals_path.exists():
            try:
                with open(self._signals_path, "r", encoding="utf-8") as f:
                    self._signals = json.load(f)
            except Exception:
                self._signals = []
        self._invalidate_cache()

    def _save(self):
        try:
            with open(self._signals_path, "w", encoding="utf-8") as f:
                json.dump(self._signals[-8000:], f, indent=2, ensure_ascii=False, default=str)
        except Exception:
            pass

    def _invalidate_cache(self):
        self._cache_dirty = True
        self._stats_cache.clear()
        self._samples_cache.clear()

    def record_signal(self, signal: Dict[str, Any]):
        now = time.time()
        strategy = signal.get("strategy_name") or signal.get("category", "unknown")
        entry = {
            "symbol": signal.get("symbol", ""),
            "direction": signal.get("direction", "WAIT"),
            "score": float(signal.get("score", 0) or 0),
            "entry_price": float(signal.get("last_price", 0) or 0),
            "strategy": strategy,
            "param_snapshot": dict(signal.get("param_snapshot", {}) or {}),
            "strategy_code_hash": str(signal.get("strategy_code_hash", "") or ""),
            "strategy_source_file": str(signal.get("strategy_source_file", "") or ""),
            "strategy_source_path": str(signal.get("strategy_source_path", "") or ""),
            "category": signal.get("category", ""),
            "timestamp": now,
            "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "validations": {},  # {window_hours: {"price": x, "pnl": x, "outcome": "win"/"loss"/"neutral"}}
            "details": {
                k: signal.get(k) for k in ("signals", "opportunity_score", "factor_scores", "details")
                if signal.get(k) is not None
            },
        }
        for existing in reversed(self._signals):
            if (existing.get("symbol") == entry["symbol"]
                    and existing.get("direction") == entry["direction"]
                    and existing.get("strategy") == entry["strategy"]
                    and now - existing.get("timestamp", 0) < 300):
                return
        self._signals.append(entry)
        self._invalidate_cache()
        self._save()

    def validate_outstanding(self, okx_client=None, atr_cache: Dict[str, float] = None) -> int:
        """多时间窗口验证：2h/6h/24h/72h 分别评估"""
        now = time.time()
        updated = 0
        for sig in self._signals:
            elapsed_h = (now - sig.get("timestamp", 0)) / 3600.0
            windows = [(h, w) for h, w in self.VALIDATION_WINDOWS if elapsed_h >= h and str(h) not in sig.get("validations", {})]
            if not windows:
                continue

            entry_price = sig["entry_price"]
            direction = sig["direction"]
            if entry_price <= 0 or direction not in ("BUY", "LONG", "SELL", "SHORT"):
                sig.setdefault("validations", {})
                if not sig["validations"]:
                    sig["validations"]["invalid"] = {"outcome": "invalid", "pnl": 0}
                    updated += 1
                continue

            if not okx_client:
                continue

            try:
                ticker = okx_client.get_ticker(sig["symbol"])
                if ticker.get("code") != "0" or not ticker.get("data"):
                    continue
                current_price = float(ticker["data"][0]["last"])
            except Exception:
                continue

            sig.setdefault("validations", {})

            for h, _ in windows:
                if direction in ("BUY", "LONG"):
                    pnl = (current_price - entry_price) / entry_price * 100
                else:
                    pnl = (entry_price - current_price) / entry_price * 100

                # 波动率归一化阈值
                atr = atr_cache.get(sig["symbol"], 2.0) if atr_cache else 2.0
                threshold = max(atr * 0.25, 0.3)

                outcome = "win" if pnl > threshold else ("loss" if pnl < -threshold else "neutral")
                sig["validations"][str(h)] = {"price": current_price, "pnl": round(pnl, 2), "outcome": outcome}
                updated += 1

        if updated > 0:
            self._invalidate_cache()
            self._save()
        return updated

    def _summarize_signal(self, signal: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        vals = signal.get("validations", {})
        if not vals:
            return None

        total_weight = 0.0
        win_weight = 0.0
        loss_weight = 0.0
        neutral_weight = 0.0
        weighted_pnls: List[float] = []
        raw_pnls: List[float] = []

        for h_str, (_, weight) in [(str(int(h)), (h, w)) for h, w in self.VALIDATION_WINDOWS]:
            v = vals.get(h_str)
            if not isinstance(v, dict):
                continue
            pnl = float(v.get("pnl", 0) or 0)
            outcome = str(v.get("outcome", "neutral"))
            total_weight += weight
            weighted_pnls.append(pnl * weight)
            raw_pnls.append(pnl)
            if outcome == "win":
                win_weight += weight
            elif outcome == "loss":
                loss_weight += weight
            else:
                neutral_weight += weight

        if total_weight <= 0:
            return None

        return {
            "symbol": signal.get("symbol", ""),
            "timestamp": signal.get("timestamp", 0),
            "datetime": signal.get("datetime", ""),
            "strategy": signal.get("strategy", ""),
            "score": float(signal.get("score", 0) or 0),
            "param_snapshot": dict(signal.get("param_snapshot", {}) or {}),
            "strategy_code_hash": str(signal.get("strategy_code_hash", "") or ""),
            "strategy_source_file": str(signal.get("strategy_source_file", "") or ""),
            "total_weight": total_weight,
            "win_weight": win_weight,
            "loss_weight": loss_weight,
            "neutral_weight": neutral_weight,
            "net_pnl": sum(weighted_pnls),
            "avg_pnl": float(np.mean(raw_pnls)) if raw_pnls else 0.0,
            "win_rate": win_weight / total_weight * 100.0,
            "validated_windows": len(raw_pnls),
            "raw_pnls": raw_pnls,
        }

    def strategy_stats(self, strategy_name: Optional[str] = None, days: int = 30, min_signals: int = 5) -> Dict:
        """贝叶斯加权绩效统计（多窗口综合评分），带缓存"""
        cache_key = f"{strategy_name or '__all__'}:{days}"
        if not self._cache_dirty and cache_key in self._stats_cache:
            return self._stats_cache[cache_key]

        cutoff = time.time() - days * 86400
        candidates = [s for s in self._signals if s.get("timestamp", 0) > cutoff]
        if strategy_name:
            candidates = [s for s in candidates if s.get("strategy") == strategy_name]

        if len(candidates) < min_signals:
            return {"total": len(candidates)}

        # 贝叶斯先验: 假设 50% 胜率 + 5 个虚拟样本
        alpha, beta_prior = 2.5, 2.5
        total_wins = 0.0
        total_count = 0.0
        weighted_pnls: List[float] = []
        raw_win_pnls: List[float] = []
        raw_loss_pnls: List[float] = []

        for s in candidates:
            summary = self._summarize_signal(s)
            if not summary:
                continue
            total_count += summary["total_weight"]
            total_wins += summary["win_weight"]
            weighted_pnls.append(summary["net_pnl"])
            for pnl in summary["raw_pnls"]:
                if pnl > 0:
                    raw_win_pnls.append(pnl)
                elif pnl < 0:
                    raw_loss_pnls.append(pnl)

        if total_count < 3:
            return {"total": len(candidates)}

        # 贝叶斯平滑胜率
        bayesian_win_rate = (total_wins + alpha) / (total_count + alpha + beta_prior) * 100

        avg_win = np.mean(raw_win_pnls) if raw_win_pnls else 0.0
        avg_loss = abs(np.mean(raw_loss_pnls)) if raw_loss_pnls else 0.0
        net = sum(weighted_pnls)

        # 夏普比率近似 (日化)
        returns = [s.get("validations", {}).get("24", {}).get("pnl", 0) for s in candidates]
        returns_clean = [r for r in returns if r != 0]
        sharpe = (np.mean(returns_clean) / max(np.std(returns_clean, ddof=1), 0.01) * np.sqrt(365 / 24)
                    if len(returns_clean) > 3 else 0.0)

        result = {
            "total": len(candidates),
            "validated_count": int(total_count),
            "win_rate": round(bayesian_win_rate, 1),
            "raw_win_rate": round(total_wins / total_count * 100 if total_count else 0, 1),
            "avg_win_pct": round(avg_win, 2),
            "avg_loss_pct": round(avg_loss, 2),
            "profit_factor": round(sum(p for p in weighted_pnls if p > 0) / max(abs(sum(p for p in weighted_pnls if p < 0)), 1e-9), 2),
            "net_pnl": round(net, 2),
            "sharpe": round(sharpe, 2),
            "period_days": days,
        }
        self._stats_cache[cache_key] = result
        return result

    def strategy_learning_samples(self, strategy_name: Optional[str] = None, days: int = 30) -> List[Dict[str, Any]]:
        cache_key = f"samples:{strategy_name or '__all__'}:{days}"
        if not self._cache_dirty and cache_key in self._samples_cache:
            return self._samples_cache[cache_key]
        cutoff = time.time() - days * 86400
        samples: List[Dict[str, Any]] = []
        for signal in self._signals:
            if signal.get("timestamp", 0) <= cutoff:
                continue
            if strategy_name and signal.get("strategy") != strategy_name:
                continue
            summary = self._summarize_signal(signal)
            if not summary:
                continue
            if not summary.get("param_snapshot"):
                continue
            samples.append(summary)
        self._samples_cache[cache_key] = samples
        return samples

    def strategy_version_stats(self, strategy_name: str, days: int = 30) -> Dict[str, Dict[str, float]]:
        stats: Dict[str, Dict[str, float]] = {}
        for sample in self.strategy_learning_samples(strategy_name, days):
            code_hash = str(sample.get("strategy_code_hash", "") or "")
            if not code_hash:
                continue
            item = stats.setdefault(code_hash, {"count": 0.0, "signal_count": 0, "wins": 0.0, "net_pnl": 0.0})
            item["count"] += sample.get("total_weight", 0.0)
            item["signal_count"] += 1
            item["wins"] += sample.get("win_weight", 0.0)
            item["net_pnl"] += sample.get("net_pnl", 0.0)

        result: Dict[str, Dict[str, float]] = {}
        for code_hash, item in stats.items():
            count = float(item.get("count", 0.0))
            signal_count = int(item.get("signal_count", 0))
            if signal_count < 2:
                continue
            result[code_hash] = {
                "sample_weight": round(count, 2),
                "signal_count": signal_count,
                "win_rate": round(item["wins"] / count * 100.0, 2) if count > 0 else 0.0,
                "net_pnl": round(item["net_pnl"], 2),
            }
        return result

    def strategy_recent_signals(self, strategy_name: str, limit: int = 20) -> List[Dict]:
        signals = [s for s in reversed(self._signals) if s.get("strategy") == strategy_name][:limit]
        return signals

    def all_strategies(self) -> List[str]:
        return list(set(s.get("strategy", "unknown") for s in self._signals))

    def score_vs_winrate(self, strategy_name: Optional[str] = None, days: int = 30) -> List[Dict]:
        """评分区间 vs 胜率（用于发现最优分数阈值）"""
        bins = [(i, i + 5) for i in range(50, 100, 5)]
        result = []
        for lo, hi in bins:
            sigs = [s for s in self._signals
                    if lo <= float(s.get("score", 0)) < hi
                    and s.get("timestamp", 0) > time.time() - days * 86400]
            if strategy_name:
                sigs = [s for s in sigs if s.get("strategy") == strategy_name]
            if not sigs:
                continue
            wins = sum(1 for s in sigs for v in s.get("validations", {}).values()
                       if isinstance(v, dict) and v.get("outcome") == "win")
            total = sum(1 for s in sigs for v in s.get("validations", {}).values()
                        if isinstance(v, dict) and v.get("outcome") in ("win", "loss"))
            if total > 0:
                result.append({"score_range": f"{lo}-{hi}", "signals": len(sigs), "win_rate": round(wins / total * 100, 1)})
        return result
