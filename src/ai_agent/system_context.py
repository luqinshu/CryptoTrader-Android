"""
系统上下文采集器：从各模块收集实时状态，构建 AI 可读的系统快照。
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from typing import Any, Dict, List, Optional


class SystemContext:
    """采集交易系统各模块状态，生成结构化上下文"""

    def __init__(self):
        self.trade_executor = None
        self.scanner = None
        self.trade_pool = None
        self.monitor_pool = None
        self.tracker = None
        self.timeframe_tracker = None
        self.optimizer = None
        self.mutator = None

    def snapshot(self) -> Dict[str, Any]:
        """生成完整系统快照"""
        return {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "balance": self._get_balance(),
            "positions": self._get_positions(),
            "pending_orders": self._get_pending_orders(),
            "scanner_status": self._get_scanner_status(),
            "trade_pool": self._get_trade_pool(),
            "monitor_pool": self._get_monitor_pool(),
            "signal_tracker": self._get_signal_summary(),
            "timeframe_accuracy": self._get_timeframe_accuracy(),
            "mutations": self._get_mutation_summary(),
        }

    def brief_snapshot(self) -> str:
        """生成供 AI prompt 使用的简短系统状态摘要"""
        parts = []
        parts.append("(以下为系统自动采集的策略运行统计，可供分析参考)")

        bal = self._get_balance()
        parts.append(f"策略资金: {bal:.2f} USDT" if bal else "策略资金: 获取中")

        positions = self._get_positions()
        if positions:
            pos_summary = []
            for p in positions[:5]:
                pos_summary.append(f"{p.get('instId','')}({p.get('side','')})浮盈{p.get('upl','0')}U")
            parts.append(f"当前策略涉及币种({len(positions)}): " + ", ".join(pos_summary))
        else:
            parts.append("策略涉及币种: 0")

        scanner = self._get_scanner_status()
        if scanner.get("last_scan_time"):
            parts.append(f"扫描更新: {scanner['last_scan_time']}，信号数: {scanner.get('signals_found', 0)}")

        pool = self._get_trade_pool()
        if pool:
            parts.append(f"交易对池容量: {pool.get('count', 0)}")

        monitor = self._get_monitor_pool()
        if monitor:
            parts.append(f"监控池: {monitor.get('count', 0)}对, 告警: {monitor.get('alerts', 0)}次")

        signals = self._get_signal_summary()
        if signals:
            parts.append(f"信号统计: {signals.get('total', 0)}条, 平均正确率 {signals.get('avg_win_rate', 0):.1f}%")

        return "\n".join(parts)

    # ─── 各模块采集 ─────────────────────────────────────────────────

    def _get_balance(self) -> float:
        if not self.trade_executor:
            return 0.0
        try:
            return float(self.trade_executor.get_usdt_balance() or 0)
        except Exception:
            return 0.0

    def _get_positions(self) -> List[Dict]:
        if not self.trade_executor:
            return []
        try:
            pos = self.trade_executor.get_positions()
            if isinstance(pos, dict):
                result = []
                for inst_id, info in pos.items():
                    result.append({
                        "instId": inst_id,
                        "side": str(getattr(info, 'side', '')),
                        "pos": float(getattr(info, 'size', 0) or 0),
                        "entry": float(getattr(info, 'entry_price', 0) or 0),
                        "curr": float(getattr(info, 'current_price', 0) or 0),
                        "upl": float(getattr(info, 'unrealized_pnl', 0) or 0),
                        "upl_pct": float(getattr(info, 'pnl_percent', 0) or 0),
                    })
                return result
            return []
        except Exception:
            return []

    def _get_pending_orders(self) -> List[Dict]:
        if not self.trade_executor:
            return []
        try:
            orders = self.trade_executor.get_pending_orders()
            return [{
                "instId": o.get("instId", ""),
                "side": o.get("side", ""),
                "sz": o.get("sz", ""),
                "px": o.get("px", ""),
                "state": o.get("state", ""),
            } for o in (orders or [])]
        except Exception:
            return []

    def _get_scanner_status(self) -> Dict:
        if not self.scanner:
            return {}
        try:
            status = {}
            if hasattr(self.scanner, 'last_scan_time'):
                status["last_scan_time"] = str(getattr(self.scanner, 'last_scan_time', ''))
            if hasattr(self.scanner, 'last_scan_signals'):
                signals = getattr(self.scanner, 'last_scan_signals', [])
                status["signals_found"] = len(signals)
                status["signals"] = [
                    {"symbol": s.get("symbol", ""), "score": round(float(s.get("score", 0) or 0), 1),
                     "direction": s.get("direction", "")}
                    for s in (signals or [])[:5]
                ]
            return status
        except Exception:
            return {}

    def _get_trade_pool(self) -> Dict:
        if not self.trade_pool:
            return {}
        try:
            items = []
            if hasattr(self.trade_pool, 'get_items'):
                items = self.trade_pool.get_items()
            elif hasattr(self.trade_pool, 'items'):
                items = self.trade_pool.items
            return {"count": len(items)}
        except Exception:
            return {}

    def _get_monitor_pool(self) -> Dict:
        if not self.monitor_pool:
            return {}
        try:
            items = []
            alerts = 0
            if hasattr(self.monitor_pool, 'get_active'):
                items = self.monitor_pool.get_active()
            if hasattr(self.monitor_pool, 'alert_count'):
                alerts = self.monitor_pool.alert_count
            return {"count": len(items) if items else 0, "alerts": alerts}
        except Exception:
            return {}

    def _get_signal_summary(self) -> Dict:
        if not self.tracker:
            return {}
        try:
            strategies = self.tracker.all_strategies()
            total = sum(
                self.tracker.strategy_stats(s, days=7).get("total", 0)
                for s in strategies
            )
            wrs = [
                self.tracker.strategy_stats(s, days=7).get("win_rate", 0)
                for s in strategies if self.tracker.strategy_stats(s, days=7).get("total", 0) > 0
            ]
            return {
                "total": total,
                "avg_win_rate": sum(wrs) / len(wrs) if wrs else 0,
                "strategies": len(strategies),
            }
        except Exception:
            return {}

    def _get_timeframe_accuracy(self) -> Dict:
        if not self.timeframe_tracker:
            return {}
        try:
            return self.timeframe_tracker.accuracy_by_strategy(days=7)
        except Exception:
            return {}

    def _get_mutation_summary(self) -> Dict:
        if not self.mutator:
            return {}
        try:
            entries = self.mutator.all_evolution_entries()
            return {"total_mutations": len(entries)}
        except Exception:
            return {}
