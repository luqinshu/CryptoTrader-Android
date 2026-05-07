"""
自主进化引擎：让 AI 持续优化自己的扫描策略分析方式。

进化循环：
  1. 健康扫描  → 找出表现最差的策略
  2. 元学习    → AI 分析历史建议效果，提炼规律（self_reflect）
  3. 深度进化  → 用元洞察驱动更精准的参数优化（evolve_scanning_params）
  4. 安全应用  → 备份 + 语法检查 + 写入
  5. 验证等待  → 收集足够信号后评估效果
  6. 反馈学习  → 将结果写回进化记忆，供下一轮 self_reflect 使用
"""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from src.ai_agent.code_parser import CodeDiffParser
from src.ai_agent.code_applier import SafeCodeApplier


class EvolutionMemory:
    """持久化进化记忆：历史建议 + 效果 + 元学习洞察"""

    def __init__(self, data_dir: Optional[Path] = None):
        self._dir = data_dir or (Path(__file__).resolve().parent.parent.parent / "data" / "evolution")
        self._dir.mkdir(parents=True, exist_ok=True)
        self._history_path = self._dir / "evolution_history.json"
        self._insights_path = self._dir / "meta_insights.json"
        self._history: List[Dict] = []
        self._insights: List[Dict] = []
        self._load()

    def _load(self):
        for path, attr in [(self._history_path, "_history"), (self._insights_path, "_insights")]:
            if path.exists():
                try:
                    setattr(self, attr, json.loads(path.read_text(encoding="utf-8")))
                except Exception:
                    setattr(self, attr, [])

    def _save(self):
        try:
            self._history_path.write_text(
                json.dumps(self._history[-500:], indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )
            self._insights_path.write_text(
                json.dumps(self._insights[-50:], indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )
        except Exception:
            pass

    def record_change(
        self,
        strategy: str,
        description: str,
        outcome: str,  # "improved" / "rolled_back" / "observing" / "no_change"
        delta: float,
        mode: str = "standard",  # "standard" / "evolution"
        meta_insight_used: str = "",
    ):
        self._history.append({
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "strategy": strategy,
            "description": description,
            "outcome": outcome,
            "delta": round(delta, 2),
            "mode": mode,
            "meta_insight_used": meta_insight_used,
        })
        self._save()

    def record_insight(self, insight: Dict):
        self._insights.append({
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            **insight,
        })
        self._save()

    def get_history(self, n: int = 50) -> List[Dict]:
        return self._history[-n:]

    def get_latest_insight(self) -> str:
        if not self._insights:
            return ""
        latest = self._insights[-1]
        return latest.get("meta_insight", "")

    def get_next_focus(self) -> List[Dict]:
        if not self._insights:
            return []
        return self._insights[-1].get("next_focus", [])

    def success_rate_by_mode(self) -> Dict[str, float]:
        from collections import defaultdict
        counts: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
        for h in self._history:
            mode = h.get("mode", "standard")
            if h.get("outcome") == "improved":
                counts[mode][0] += 1
            counts[mode][1] += 1
        return {
            mode: round(wins / total * 100, 1) if total else 0.0
            for mode, (wins, total) in counts.items()
        }


class SelfEvolver:
    """
    自主进化引擎。

    可作为独立线程运行，也可被 AutoCycleController 调用。

    核心特性：
    - 元学习：AI 分析自身历史建议的成效，提炼"什么有效/什么无效"
    - 深度进化：用成功/失败信号分布驱动更精准的参数搜索
    - 进化记忆：持久化历史和洞察，每轮循环越来越聪明
    - 自适应节奏：连续失败时放缓节奏，成功时加快频率
    """

    def __init__(
        self,
        advisor,          # StrategyAdvisor
        tracker,          # 信号追踪器
        code_applier: Optional[SafeCodeApplier] = None,
        cycle_interval_min: int = 60,
        verify_wait_signals: int = 10,
        max_drawdown_allowed: float = 5.0,
        min_signals_to_analyze: int = 5,
        reflect_every_n_cycles: int = 3,
        log_callback: Optional[Callable[[str, str], None]] = None,
    ):
        self.advisor = advisor
        self.tracker = tracker
        self.code_applier = code_applier or SafeCodeApplier()
        self.parser = CodeDiffParser
        self.cycle_interval_min = cycle_interval_min
        self.verify_wait_signals = verify_wait_signals
        self.max_drawdown_allowed = max_drawdown_allowed
        self.min_signals_to_analyze = min_signals_to_analyze
        self.reflect_every_n_cycles = reflect_every_n_cycles
        self._log = log_callback or (lambda msg, lvl: None)

        self.memory = EvolutionMemory()
        self._pending: Dict[str, Dict] = {}  # strategy → {before_wr, signal_count, ...}
        self._cycle = 0
        self._stop_flag = False
        self._thread: Optional[threading.Thread] = None

        # 自适应节奏：连续失败计数
        self._consecutive_failures = 0
        self._consecutive_successes = 0

    # ─── 启动 / 停止 ─────────────────────────────────────────────────

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop_flag = False
        self._thread = threading.Thread(target=self._loop, daemon=True, name="SelfEvolver")
        self._thread.start()
        self._log("🧬 自主进化引擎已启动", "INFO")

    def stop(self):
        self._stop_flag = True
        self._log("⏹ 自主进化引擎停止中...", "INFO")

    def is_running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    # ─── 主循环 ──────────────────────────────────────────────────────

    def _loop(self):
        while not self._stop_flag:
            try:
                self._run_cycle()
            except Exception as e:
                self._log(f"⚠️ 进化循环异常: {e}", "ERROR")
                self._consecutive_failures += 1

            # 自适应等待：连续失败则延长间隔
            wait_min = self.cycle_interval_min
            if self._consecutive_failures >= 3:
                wait_min = min(wait_min * 2, 240)
                self._log(f"⏳ 连续失败，本轮等待 {wait_min} 分钟", "WARNING")
            elif self._consecutive_successes >= 2:
                wait_min = max(wait_min // 2, 15)

            deadline = time.time() + wait_min * 60
            while time.time() < deadline and not self._stop_flag:
                time.sleep(5)

        self._log("🧬 自主进化引擎已停止", "INFO")

    def _run_cycle(self):
        self._cycle += 1
        self._log(f"--- 进化周期 #{self._cycle} ---", "INFO")

        # 第 1 步：验证待处理变更
        self._verify_pending()

        # 第 2 步：每 N 轮做一次元反思
        meta_insight = self.memory.get_latest_insight()
        if self._cycle % self.reflect_every_n_cycles == 0:
            meta_insight = self._do_self_reflect() or meta_insight

        # 第 3 步：确定本轮优化目标
        targets = self._pick_targets(meta_insight)

        # 第 4 步：对每个目标执行深度进化
        for target in targets:
            if self._stop_flag:
                break
            self._evolve_one(target["strategy"], meta_insight, mode=target.get("mode", "evolution"))

    # ─── 元反思 ──────────────────────────────────────────────────────

    def _do_self_reflect(self) -> str:
        self._log("🔍 AI 元反思：分析历史建议效果...", "INFO")
        history = self.memory.get_history(50)
        if len(history) < 3:
            return ""
        try:
            insight = self.advisor.self_reflect(history, days=7)
        except Exception as e:
            self._log(f"  ❌ 元反思失败: {e}", "ERROR")
            return ""
        if not insight:
            return ""
        self.memory.record_insight(insight)
        meta = insight.get("meta_insight", "")
        effective = insight.get("effective_patterns", [])
        harmful = insight.get("harmful_patterns", [])
        self._log(f"  💡 元洞察: {meta}", "INFO")
        if effective:
            self._log(f"  ✅ 有效模式: {'; '.join(effective[:2])}", "SUCCESS")
        if harmful:
            self._log(f"  ❌ 无效模式: {'; '.join(harmful[:2])}", "WARNING")
        return meta

    # ─── 目标选取 ─────────────────────────────────────────────────────

    def _pick_targets(self, meta_insight: str) -> List[Dict]:
        """综合健康状态 + 元反思建议，选出本轮优化目标"""
        strategies = self.tracker.all_strategies() if self.tracker else []
        if not strategies:
            return []

        # 元反思指定的高优先级目标
        priority_set = {
            f["strategy"] for f in self.memory.get_next_focus()
            if f.get("priority") == "high"
        }

        # 收集健康数据
        candidates = []
        for name in strategies:
            if name in self._pending:
                continue  # 有待验证变更，本轮跳过
            stats = self.tracker.strategy_stats(name, days=7)
            wr = stats.get("win_rate", 0)
            total = stats.get("total", 0)
            if total < self.min_signals_to_analyze:
                continue
            score = (100 - wr) + (20 if name in priority_set else 0)
            mode = "evolution" if name in priority_set else "standard"
            candidates.append({"strategy": name, "score": score, "win_rate": wr, "mode": mode})

        # 按综合评分降序，取前 3
        candidates.sort(key=lambda x: -x["score"])
        chosen = candidates[:3]
        for c in chosen:
            self._log(f"  🎯 目标: {c['strategy']} 胜率{c['win_rate']:.1f}% mode={c['mode']}", "INFO")
        return chosen

    # ─── 单策略进化 ───────────────────────────────────────────────────

    def _evolve_one(self, strategy_name: str, meta_insight: str, mode: str = "evolution"):
        self._log(f"🧬 进化 [{mode}] → {strategy_name}", "INFO")
        stats = self.tracker.strategy_stats(strategy_name, days=7)

        try:
            if mode == "evolution":
                result = self.advisor.evolve_scanning_params(
                    strategy_name, days=14, meta_insight=meta_insight
                )
            else:
                result = self.advisor.analyze_strategy(strategy_name, days=7)
        except Exception as e:
            self._log(f"  ❌ AI分析异常: {e}", "ERROR")
            self._consecutive_failures += 1
            return

        if not result or result.get("error"):
            self._log(f"  ❌ AI分析失败: {result.get('error', '') if result else 'None'}", "ERROR")
            self._consecutive_failures += 1
            return

        advice = result.get("advice", "")
        fname = self.advisor._find_strategy_file(strategy_name)
        if not fname:
            self._log(f"  ⚠️ 未找到 {strategy_name} 对应文件，跳过", "WARNING")
            return

        changes = self.parser.parse_analysis(advice, target_file=fname)
        if not changes:
            self._log(f"  📊 未提取到可用修改建议", "INFO")
            self.memory.record_change(strategy_name, "无修改", "no_change", 0.0, mode, meta_insight)
            return

        self._log(f"  📝 提取到 {len(changes)} 项修改", "INFO")
        success, msg, applied = self.code_applier.apply_changes(fname, changes)

        if success and applied:
            self._register_pending(strategy_name, applied, stats, mode, meta_insight)
            self._consecutive_failures = 0
            self._consecutive_successes += 1
            self._log(f"  ✅ {len(applied)}项已应用，等待 {self.verify_wait_signals} 条信号后验证", "SUCCESS")
        else:
            self._log(f"  ❌ 应用失败: {msg[:100]}", "ERROR")
            self.memory.record_change(strategy_name, str(applied)[:100], "apply_failed", 0.0, mode, meta_insight)
            self._consecutive_failures += 1
            self._consecutive_successes = 0

    # ─── 待验证队列 ───────────────────────────────────────────────────

    def _register_pending(
        self,
        strategy: str,
        applied: List,
        stats: Dict,
        mode: str,
        meta_insight: str,
    ):
        self._pending[strategy] = {
            "applied_at": time.time(),
            "signal_count_at_apply": stats.get("total", 0),
            "win_rate_before": stats.get("win_rate", 0),
            "description": "; ".join(str(a) for a in applied[:3]),
            "mode": mode,
            "meta_insight_used": meta_insight,
            "file": self.advisor._find_strategy_file(strategy),
        }

    def _verify_pending(self):
        for strategy, pending in list(self._pending.items()):
            stats = self.tracker.strategy_stats(strategy, days=7)
            total = stats.get("total", 0)
            since = total - pending.get("signal_count_at_apply", 0)
            if since < self.verify_wait_signals:
                continue

            before_wr = pending.get("win_rate_before", 0)
            current_wr = stats.get("win_rate", 0)
            delta = current_wr - before_wr
            mode = pending.get("mode", "standard")
            meta_insight = pending.get("meta_insight_used", "")
            desc = pending.get("description", "")

            if delta > 2.0:
                self._log(
                    f"✅ {strategy} 胜率 {before_wr:.1f}%→{current_wr:.1f}% (+{delta:.1f}%)",
                    "SUCCESS",
                )
                self.memory.record_change(strategy, desc, "improved", delta, mode, meta_insight)
                self._consecutive_failures = 0
                self._consecutive_successes += 1
            elif delta < -self.max_drawdown_allowed:
                self._log(
                    f"❌ {strategy} 胜率下降 {delta:.1f}%，自动回退",
                    "ERROR",
                )
                fname = pending.get("file", "")
                if fname:
                    ok, msg = self.code_applier.rollback(fname)
                    self._log(f"  ⏪ {msg}", "INFO")
                self.memory.record_change(strategy, desc, "rolled_back", delta, mode, meta_insight)
                self._consecutive_failures += 1
                self._consecutive_successes = 0
            else:
                self._log(
                    f"📊 {strategy} 变更后 {delta:+.1f}%，继续观察",
                    "INFO",
                )
                # 信号量翻倍后再判断
                if since >= self.verify_wait_signals * 2:
                    self.memory.record_change(strategy, desc, "observing", delta, mode, meta_insight)
                    del self._pending[strategy]
                continue

            del self._pending[strategy]

    # ─── 状态查询 ─────────────────────────────────────────────────────

    def get_status(self) -> Dict:
        return {
            "cycle": self._cycle,
            "running": self.is_running(),
            "pending_verifications": list(self._pending.keys()),
            "consecutive_failures": self._consecutive_failures,
            "consecutive_successes": self._consecutive_successes,
            "success_rate_by_mode": self.memory.success_rate_by_mode(),
            "latest_insight": self.memory.get_latest_insight(),
        }

    def get_history(self, n: int = 20) -> List[Dict]:
        return self.memory.get_history(n)

    def manual_reflect(self) -> Optional[str]:
        """手动触发一次元反思，返回洞察摘要"""
        insight = self._do_self_reflect()
        return insight if insight else "（历史数据不足，无法反思）"
