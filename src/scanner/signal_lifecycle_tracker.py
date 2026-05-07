"""
信号生命周期追踪器 v1.0
========================
闭环学习：记录每个扫描信号 → 24H后结算 → 反馈到后续信号评分。

核心功能：
  1. record_signal()     — 信号产生时立即记录
  2. resolve_outcomes()  — 定期检查，用当前价格结算PnL
  3. get_track_record()  — 查询 (品种,方向) 的历史表现
  4. apply_signal_adjustment() — 根据历史胜率调整信号分数

衰减机制：
  - 最近 10 条记录权重 1.0
  - 每往前 10 条衰减 ×0.75
  - 最多保留 100 条/品种

数据存储：
  - 内存 Dict + 可选 JSON 持久化（跨程序重启）
"""

from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ── 数据结构 ────────────────────────────────────────────────────────────────

class SignalRecord:
    __slots__ = ("symbol", "direction", "entry_price", "signal_score",
                 "created_at", "resolved_at", "exit_price", "pnl_pct", "resolved")
    def __init__(self, symbol, direction, entry_price, signal_score):
        self.symbol = symbol
        self.direction = direction
        self.entry_price = entry_price
        self.signal_score = signal_score
        self.created_at = time.time()
        self.resolved_at = 0.0
        self.exit_price = 0.0
        self.pnl_pct = 0.0
        self.resolved = False

    def to_dict(self):
        return {
            "symbol": self.symbol, "direction": self.direction,
            "entry": self.entry_price, "score": self.signal_score,
            "created": self.created_at, "resolved_at": self.resolved_at,
            "exit": self.exit_price, "pnl_pct": self.pnl_pct,
            "resolved": self.resolved,
        }

    @classmethod
    def from_dict(cls, d):
        r = cls(d["symbol"], d["direction"], d["entry"], d["score"])
        r.created_at = d["created"]
        r.resolved_at = d.get("resolved_at", 0)
        r.exit_price = d.get("exit", 0)
        r.pnl_pct = d.get("pnl_pct", 0)
        r.resolved = d.get("resolved", False)
        return r


class SignalLifecycleTracker:
    """全局单例：信号生命周期管理（线程安全）"""

    _instance: Optional["SignalLifecycleTracker"] = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, persist_path: str = ""):
        if hasattr(self, "_initialized"):
            return
        self._initialized = True
        self._records: Dict[str, List[SignalRecord]] = {}  # key = "SYMBOL:DIR"
        self._resolve_window_hours: float = 24.0
        self._max_records_per_key: int = 100
        self._decay_base: float = 0.75
        self._persist_path = persist_path or str(
            Path(__file__).resolve().parent.parent.parent / "data" / "signal_lifecycle.json"
        )
        self._load()

    # ── 公共接口 ───────────────────────────────────────────────────────────

    def record_signal(self, symbol: str, direction: str, entry_price: float,
                      signal_score: float) -> None:
        """信号产生时调用"""
        key = self._key(symbol, direction)
        if key not in self._records:
            self._records[key] = []
        rec = SignalRecord(symbol, direction, entry_price, signal_score)
        self._records[key].append(rec)
        # 保留最近 N 条
        if len(self._records[key]) > self._max_records_per_key:
            self._records[key] = self._records[key][-self._max_records_per_key:]

    def resolve_outcomes(self, symbol: str, direction: str,
                         current_price: float) -> int:
        """
        结算所有未解决的信号。
        多头: PnL = (current - entry) / entry
        空头: PnL = (entry - current) / entry
        返回本次结算的数量。
        """
        key = self._key(symbol, direction)
        if key not in self._records:
            return 0
        now = time.time()
        resolved_count = 0
        for rec in self._records[key]:
            if rec.resolved:
                continue
            age_hours = (now - rec.created_at) / 3600.0
            if age_hours < self._resolve_window_hours:
                continue
            rec.resolved = True
            rec.resolved_at = now
            rec.exit_price = current_price
            if direction.upper() in ("BUY", "LONG"):
                rec.pnl_pct = (current_price - rec.entry_price) / rec.entry_price * 100
            else:
                rec.pnl_pct = (rec.entry_price - current_price) / rec.entry_price * 100
            resolved_count += 1
        return resolved_count

    def resolve_all(self, price_map: Dict[str, float]) -> Dict[str, int]:
        """批量结算: {symbol: current_price} → 返回各key结算数"""
        result = {}
        for key in list(self._records.keys()):
            parts = key.rsplit(":", 1)
            if len(parts) != 2:
                continue
            symbol, direction = parts
            price = price_map.get(symbol)
            if price is None or not np.isfinite(price):
                continue
            n = self.resolve_outcomes(symbol, direction, price)
            if n > 0:
                result[key] = n
        return result

    def get_track_record(self, symbol: str, direction: str) -> Dict[str, Any]:
        """
        查询 (品种, 方向) 的历史表现。
        返回:
          total_signals, resolved_count, win_rate (0~1),
          avg_pnl_pct, weighted_win_rate (衰减), recent_pnls (近10次)
        """
        key = self._key(symbol, direction)
        records = self._records.get(key, [])
        resolved = [r for r in records if r.resolved]
        total = len(records)

        if not resolved:
            return {
                "total_signals": total, "resolved_count": 0,
                "win_rate": 0.5, "avg_pnl_pct": 0.0,
                "weighted_win_rate": 0.5, "recent_pnls": [],
                "status": "insufficient_data",
            }

        wins = sum(1 for r in resolved if r.pnl_pct > 0)
        win_rate = wins / len(resolved)
        avg_pnl = sum(r.pnl_pct for r in resolved) / len(resolved)

        # 指数衰减加权胜率: 最近10条权重1.0, 每往前10条×0.75
        sorted_resolved = sorted(resolved, key=lambda r: r.resolved_at, reverse=True)
        weighted_wins = 0.0
        weighted_total = 0.0
        for i, r in enumerate(sorted_resolved):
            bucket = i // 10
            weight = self._decay_base ** bucket
            weighted_total += weight
            if r.pnl_pct > 0:
                weighted_wins += weight
        weighted_win_rate = weighted_wins / max(weighted_total, 1e-9)

        recent_pnls = [round(r.pnl_pct, 2) for r in sorted_resolved[:10]]

        return {
            "total_signals": total,
            "resolved_count": len(resolved),
            "pending_count": total - len(resolved),
            "win_rate": round(win_rate, 3),
            "avg_pnl_pct": round(avg_pnl, 2),
            "weighted_win_rate": round(weighted_win_rate, 3),
            "recent_pnls": recent_pnls,
            "status": "tracked",
        }

    def apply_signal_adjustment(self, symbol: str, direction: str,
                                 base_score: float) -> Tuple[float, Dict]:
        """
        根据历史表现调整信号分数。
        - 胜率>60% → 加分(max +8)
        - 胜率<35% → 降分(max -10)
        - 数据不足 → 不变
        返回: (adjusted_score, adjustment_detail)
        """
        record = self.get_track_record(symbol, direction)
        if record["status"] == "insufficient_data" or record["resolved_count"] < 3:
            return base_score, {"adjusted": False, "reason": "数据不足(需≥3次结算)"}

        wr = record["weighted_win_rate"]
        avg_pnl = record["avg_pnl_pct"]

        if wr >= 0.60 and avg_pnl > 0:
            bonus = min(8.0, (wr - 0.50) * 20 + avg_pnl * 0.5)
            adjusted = min(100, base_score + bonus)
            detail = {"adjusted": True, "bonus": round(bonus, 1),
                      "reason": f"历史胜率{wr:.0%} 均PnL{avg_pnl:+.1f}% +{bonus:.1f}分"}
        elif wr < 0.35:
            penalty = min(10.0, (0.50 - wr) * 20 + abs(avg_pnl) * 0.5)
            adjusted = max(0, base_score - penalty)
            detail = {"adjusted": True, "penalty": round(penalty, 1),
                      "reason": f"历史胜率{wr:.0%} 均PnL{avg_pnl:+.1f}% -{penalty:.1f}分"}
        else:
            adjusted = base_score
            detail = {"adjusted": False, "reason": f"历史胜率{wr:.0%} 在正常范围, 不调整"}

        return round(adjusted, 1), detail

    def get_summary(self) -> Dict[str, Any]:
        """全局统计摘要"""
        total_signals = sum(len(v) for v in self._records.values())
        total_resolved = sum(1 for v in self._records.values() for r in v if r.resolved)
        all_pnls = [r.pnl_pct for v in self._records.values() for r in v if r.resolved]
        return {
            "total_keys_tracked": len(self._records),
            "total_signals": total_signals,
            "total_resolved": total_resolved,
            "global_win_rate": round(sum(1 for p in all_pnls if p > 0) / max(len(all_pnls), 1), 3) if all_pnls else 0.0,
            "global_avg_pnl_pct": round(sum(all_pnls) / max(len(all_pnls), 1), 2) if all_pnls else 0.0,
        }

    # ── 持久化 ──────────────────────────────────────────────────────────────

    def save(self) -> None:
        """保存到 JSON 文件"""
        try:
            data = {}
            for key, records in self._records.items():
                data[key] = [r.to_dict() for r in records]
            os.makedirs(os.path.dirname(self._persist_path), exist_ok=True)
            with open(self._persist_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[生命周期追踪] 保存失败: {e}")

    def _load(self) -> None:
        """从 JSON 文件加载"""
        if not os.path.exists(self._persist_path):
            return
        try:
            with open(self._persist_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for key, recs in data.items():
                self._records[key] = [SignalRecord.from_dict(r) for r in recs]
            print(f"[生命周期追踪] 加载 {len(data)} 个品种×方向的历史记录")
        except Exception as e:
            print(f"[生命周期追踪] 加载失败: {e}")

    # ── 内部 ────────────────────────────────────────────────────────────────

    @staticmethod
    def _key(symbol: str, direction: str) -> str:
        d = str(direction).upper()
        d = "BUY" if d in ("BUY", "LONG") else ("SELL" if d in ("SELL", "SHORT") else d)
        return f"{symbol}:{d}"


# ── 全局便捷函数 ────────────────────────────────────────────────────────────

_tracker: Optional[SignalLifecycleTracker] = None


def get_tracker(persist_path: str = "") -> SignalLifecycleTracker:
    global _tracker
    if _tracker is None:
        _tracker = SignalLifecycleTracker(persist_path)
    return _tracker


def record_signal(symbol: str, direction: str, entry_price: float,
                  signal_score: float) -> None:
    get_tracker().record_signal(symbol, direction, entry_price, signal_score)


def resolve_outcomes(symbol: str, direction: str, current_price: float) -> int:
    return get_tracker().resolve_outcomes(symbol, direction, current_price)


def get_track_record(symbol: str, direction: str) -> Dict[str, Any]:
    return get_tracker().get_track_record(symbol, direction)
