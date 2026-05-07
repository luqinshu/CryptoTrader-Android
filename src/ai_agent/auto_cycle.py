"""
自主进化循环控制器：自动化 分析→决策→应用→验证→学习 全流程。
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from src.qt_compat import QThread, Signal



class AutoCycleController(QThread):
    """
    自主进化循环。

    流程（每次周期）：
    1. 检查策略健康度 → 低于阈值触发分析
    2. AI 分析 → 生成代码修改建议
    3. 评分过滤 → 仅保留高置信度(>70%)低风险变更
    4. 安全应用 → 备份 + 写入
    5. 等待验证 → N 分钟后检查胜率变化
    6. 优胜劣汰 → 变好保留，变差自动回退

    配置:
    - cycle_interval_min: 周期间隔（分钟）
    - min_confidence: 自动应用的最低置信度
    - verify_wait_signals: 应用后等待多少条新信号再验证
    - max_drawdown_after_apply: 应用后最大允许回撤
    - auto_apply_enabled: 是否自动应用（关闭则仅分析+建议）
    """

    log_signal = Signal(str, str)       # (message, level)
    status_signal = Signal(str)         # status text
    analysis_requested = Signal(str)    # strategy_name → 请求分析
    change_applied = Signal(str, list)  # strategy_name, changes
    change_rolled_back = Signal(str, str)  # strategy_name, reason

    def __init__(
        self,
        tracker=None,
        advisor=None,
        code_applier=None,
        parser=None,
        cycle_interval_min: int = 60,
        min_confidence: float = 70.0,
        verify_wait_signals: int = 10,
        max_drawdown_after_apply: float = 5.0,
        auto_apply_enabled: bool = True,
    ):
        super().__init__()
        self.tracker = tracker
        self.advisor = advisor            # StrategyAdvisor (has LLM client)
        self.code_applier = code_applier  # SafeCodeApplier
        self.parser = parser              # CodeDiffParser
        self.cycle_interval_min = cycle_interval_min
        self.min_confidence = min_confidence
        self.verify_wait_signals = verify_wait_signals
        self.max_drawdown_after_apply = max_drawdown_after_apply
        self.auto_apply_enabled = auto_apply_enabled

        self._stop_flag = False
        self._pending_verifications: Dict[str, Dict] = {}
        self._change_history: List[Dict] = []
        self._load_history()

    def _history_path(self) -> Path:
        return Path(__file__).resolve().parent.parent.parent / "data" / "ai_cycle_history.json"

    def _load_history(self):
        p = self._history_path()
        if p.exists():
            try:
                self._change_history = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                self._change_history = []

    def _save_history(self):
        p = self._history_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        try:
            p.write_text(
                json.dumps(self._change_history[-500:], indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )
        except Exception:
            pass

    def stop(self):
        self._stop_flag = True

    def run(self):
        self.log_signal.emit("🔄 自主进化循环已启动", "INFO")
        self.status_signal.emit("🔄 自主进化运行中")

        cycle = 0
        while not self._stop_flag:
            try:
                self._run_cycle(cycle)
            except Exception as e:
                self.log_signal.emit(f"⚠️ 进化循环异常: {e}", "ERROR")
            cycle += 1
            # 等待周期间隔（每 5 秒检查停止标志）
            for _ in range(self.cycle_interval_min * 12):
                if self._stop_flag:
                    break
                self.msleep(5000)

        self.status_signal.emit("⏹ 自主进化已停止")
        self.log_signal.emit("🔄 自主进化循环已停止", "INFO")

    def _run_cycle(self, cycle: int):
        self.log_signal.emit(f"--- 进化周期 #{cycle + 1} ---", "INFO")

        # === 第 1 步：检查策略健康度（含验证待处理变更）===
        strategies = self.tracker.all_strategies() if self.tracker else []
        for name in strategies:
            stats = self.tracker.strategy_stats(name, days=7)
            wr = stats.get("win_rate", 0)
            total = stats.get("total", 0)

            if total < 10:
                continue

            # 验证之前应用的变更（用当前stats，避免二次查询）
            pending = self._pending_verifications.get(name)
            if pending:
                signals_since = total - pending.get("signal_count_at_apply", 0)
                if signals_since >= self.verify_wait_signals:
                    self._verify_change(name, pending, stats)
                    continue  # 本轮已验证，跳过分析触发

            # 健康度低于 50% → 触发分析（有advisor则直接分析，否则发信号给UI）
            if wr < 50.0:
                self.log_signal.emit(f"⚠️ {name} 胜率仅 {wr:.1f}%，低于 50% 阈值，请求 AI 分析", "WARNING")
                if self.advisor and self.auto_apply_enabled:
                    self._direct_analyze_and_apply(name, stats)
                else:
                    self.analysis_requested.emit(name)

    def _direct_analyze_and_apply(self, strategy_name: str, stats: Dict):
        """直接调用advisor分析并应用变更（无需UI参与）"""
        if not self.advisor or not self.code_applier or not self.parser:
            return
        try:
            result = self.advisor.analyze_strategy(strategy_name, days=7)
            if not result or result.get("error"):
                self.log_signal.emit(f"  ❌ AI分析失败: {result.get('error', '无响应') if result else '无响应'}", "ERROR")
                return
            advice = result.get("advice", "")
            fname = self.advisor._find_strategy_file(strategy_name)
            if not fname:
                return
            changes = self.parser.parse_analysis(advice, target_file=fname)
            if not changes:
                self.log_signal.emit(f"  📊 {strategy_name} AI未提取到可用修改建议", "INFO")
                return
            # 置信度过滤
            approved = [c for c in changes if c.confidence >= self.min_confidence / 100.0]
            if not approved:
                self.log_signal.emit(f"  ⏸ 所有变更置信度低于{self.min_confidence:.0f}%，跳过", "INFO")
                return
            success, msg, applied = self.code_applier.apply_changes(fname, approved)
            if success:
                self.on_changes_applied(strategy_name, applied, stats)
                self.change_applied.emit(strategy_name, applied)
                self.log_signal.emit(f"  ✅ {len(applied)}项修改已应用，等待验证", "SUCCESS")
            else:
                self.log_signal.emit(f"  ❌ 应用失败: {msg}", "ERROR")
        except Exception as e:
            self.log_signal.emit(f"  ⚠️ 直接分析异常: {e}", "ERROR")

    def _verify_change(self, strategy_name: str, pending: Dict, current_stats: Dict):
        """验证一项变更的实际效果"""
        before_wr = pending.get("win_rate_before", 0)
        current_wr = current_stats.get("win_rate", 0)
        change_desc = pending.get("description", "")

        delta = current_wr - before_wr
        if delta > 2.0:
            self.log_signal.emit(
                f"✅ {strategy_name} 变更有效: 胜率 {before_wr:.1f}% → {current_wr:.1f}% (+{delta:.1f}%)",
                "SUCCESS",
            )
            self._record_result(strategy_name, change_desc, "improved", delta)
            del self._pending_verifications[strategy_name]
        elif delta < -self.max_drawdown_after_apply:
            self.log_signal.emit(
                f"❌ {strategy_name} 变更后胜率下降: {before_wr:.1f}% → {current_wr:.1f}% ({delta:.1f}%)，自动回退",
                "ERROR",
            )
            if self.code_applier:
                success, msg = self.code_applier.rollback(strategy_name)
                self.change_rolled_back.emit(strategy_name, msg)
                self.log_signal.emit(f"  ⏪ 已回退: {msg}", "INFO")
            self._record_result(strategy_name, change_desc, "rolled_back", delta)
            del self._pending_verifications[strategy_name]
        else:
            self.log_signal.emit(
                f"📊 {strategy_name} 变更后表现接近 ({delta:+.1f}%)，继续观察",
                "INFO",
            )

    def _record_result(self, strategy: str, description: str, outcome: str, delta: float):
        entry = {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "strategy": strategy,
            "description": description,
            "outcome": outcome,
            "delta": round(delta, 1),
        }
        self._change_history.append(entry)
        self._save_history()

    def on_changes_applied(self, strategy_name: str, changes: List, current_stats: Dict):
        """当变更被应用后调用，注册待验证"""
        self._pending_verifications[strategy_name] = {
            "applied_at": time.time(),
            "signal_count_at_apply": current_stats.get("total", 0),
            "win_rate_before": current_stats.get("win_rate", 0),
            "description": "; ".join(
                getattr(c, "description", str(c)) for c in changes[:3]
            ),
        }
        self.log_signal.emit(
            f"📝 {strategy_name} 变更已注册验证（等待 {self.verify_wait_signals} 条新信号后评估）",
            "INFO",
        )

    def get_health_report(self) -> List[Dict]:
        """返回所有策略的健康报告"""
        if not self.tracker:
            return []
        report = []
        for name in self.tracker.all_strategies():
            stats = self.tracker.strategy_stats(name, days=7)
            wr = stats.get("win_rate", 0)
            total = stats.get("total", 0)
            pending = name in self._pending_verifications
            report.append({
                "strategy": name,
                "win_rate_7d": wr,
                "signals_7d": total,
                "pending_verify": pending,
                "status": "healthy" if wr >= 50 else "critical",
            })
        return report

    def get_change_history(self) -> List[Dict]:
        return self._change_history[-100:]
