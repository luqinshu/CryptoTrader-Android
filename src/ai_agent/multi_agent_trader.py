"""
多智能体交易决策系统：分析师→辩论→风控→执行，像投资团队一样协作。

基于 LangGraph 多智能体框架思路：
- MarketAnalyst: 扫描结果 + 技术面分析
- RiskAnalyst: 余额/回撤/仓位风控
- ExecutionAgent: 生成 + 执行交易信号
- Coordinator: 状态路由 + 共识决策
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional
from src.qt_compat import QThread, Signal
from src.trading.position_registry import position_registry



@dataclass
class AgentState:
    """多智能体共享状态"""
    symbol: str = ""
    trade_date: str = ""
    sender: str = ""

    # 扫描数据
    scanner_results: List[Dict] = field(default_factory=list)
    top_signal: Optional[Dict] = None

    # 市场分析
    market_report: str = ""
    market_confidence: float = 0.0

    # 风控分析
    risk_report: str = ""
    risk_passed: bool = False
    risk_score: float = 0.0

    # 决策
    debate_history: str = ""
    final_decision: str = ""
    final_action: str = ""  # BUY / SELL / HOLD
    final_size: float = 0.0
    final_reason: str = ""

    # 账户
    balance: float = 0.0
    positions: List[Dict] = field(default_factory=list)
    drawdown_pct: float = 0.0

    # 辩论轮次
    debate_rounds: int = 0

    def to_snapshot(self) -> Dict:
        return {
            "symbol": self.symbol,
            "action": self.final_action,
            "size": self.final_size,
            "reason": self.final_reason[:120],
            "market_conf": self.market_confidence,
            "risk_score": self.risk_score,
            "balance": self.balance,
            "timestamp": self.trade_date,
        }


class MultiAgentTrader(QThread):
    """
    多智能体交易协调器。

    工作流：
    Scanner → MarketAnalyst → Debate → RiskAnalyst → Coordinator → Execution
    """

    log_signal = Signal(str, str)
    decision_signal = Signal(dict)      # 最终决策
    trade_signal = Signal(str, str, float)  # symbol, action, size

    MAX_DEBATE_ROUNDS = 2

    def __init__(self, llm_client=None, trade_executor=None, scanner=None,
                 tracker=None, timeframe_tracker=None, optimizer=None):
        super().__init__()
        self.llm = llm_client
        self.executor = trade_executor
        self.scanner = scanner
        self.tracker = tracker
        self.timeframe_tracker = timeframe_tracker
        self.optimizer = optimizer
        self._stop_flag = False
        self.state = AgentState()
        self.decision_history: List[Dict] = []

        # 注册接管回调：当其他系统强制接管我们管理的标的时清空内部状态
        position_registry.register_takeover_callback(self._on_takeover)

    def stop(self):
        self._stop_flag = True

    def _on_takeover(self, inst_id: str, old_owner: str, new_owner: str):
        """当其他系统强制接管我们管理的标的时，清空内部状态机。"""
        if old_owner != 'MultiAgent':
            return
        self.log_signal.emit(
            f"⚠️ {inst_id} 被 {new_owner} 强制接管，清理内部状态", "WARNING"
        )
        if self.state.symbol == inst_id:
            self.state.positions.clear()
            self.state.final_action = "HOLD"
            self.state.final_size = 0.0

    def run(self):
        self.log_signal.emit("🤖 多智能体交易系统已启动", "INFO")
        while not self._stop_flag:
            try:
                self._cycle()
            except Exception as e:
                self.log_signal.emit(f"⚠️ 多智能体异常: {e}", "ERROR")
            for _ in range(1200):
                if self._stop_flag:
                    break
                self.msleep(500)

    def _cycle(self):
        """执行一个完整分析周期"""
        self.log_signal.emit("--- 多智能体分析周期 ---", "INFO")

        # 1. 采集数据
        if not self._collect_data():
            self.log_signal.emit("📊 无新信号数据，跳过", "INFO")
            return

        # 2. 市场分析
        self._market_analysis()

        # 3. 风控分析
        self._risk_analysis()

        # 4. 决策
        self._make_decision()

        # 5. 执行
        if self.state.final_action in ("BUY", "SELL") and self.state.final_size > 0:
            self._execute()

    def _collect_data(self) -> bool:
        """采集扫描结果和账户数据"""
        self.state.trade_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 余额
        if self.executor:
            try:
                self.state.balance = float(self.executor.get_usdt_balance() or 0)
            except Exception:
                self.state.balance = 0.0

        # 持仓
        if self.executor:
            try:
                pos = self.executor.get_positions()
                if isinstance(pos, dict):
                    for inst_id, info in pos.items():
                        self.state.positions.append({
                            "instId": inst_id,
                            "side": str(getattr(info, 'side', '')),
                            "size": float(getattr(info, 'size', 0) or 0),
                            "upl": float(getattr(info, 'unrealized_pnl', 0) or 0),
                        })
            except Exception:
                pass

        # 扫描结果
        if self.scanner:
            try:
                if hasattr(self.scanner, 'last_scan_signals'):
                    signals = getattr(self.scanner, 'last_scan_signals', [])
                    self.state.scanner_results = signals
                    if signals:
                        self.state.top_signal = max(signals, key=lambda s: float(s.get("score", 0) or 0))
                        self.state.symbol = self.state.top_signal.get("symbol", "")
            except Exception:
                pass

        return len(self.state.scanner_results) > 0

    def _market_analysis(self):
        """市场分析师：评估扫描信号质量"""
        if not self.state.top_signal:
            self.state.market_report = "无有效信号"
            return

        signal = self.state.top_signal
        parts = []
        score = float(signal.get("score", 0) or 0)
        direction = signal.get("direction", "")

        parts.append(f"信号评分: {score:.1f}")
        parts.append(f"方向: {direction}")
        parts.append(f"来源策略: {signal.get('strategy_name', signal.get('category', '未知'))}")

        # 多周期验证
        if self.timeframe_tracker and self.state.symbol:
            try:
                acc = self.timeframe_tracker.accuracy_by_timeframe(days=7)
                parts.append(f"周期准确率: {json.dumps({k: round(v.get('accuracy',0),1) for k,v in acc.items()}, ensure_ascii=False)}")
            except Exception:
                pass

        self.state.market_report = "\n".join(parts)
        self.state.market_confidence = min(score / 100.0, 0.95)

        # LLM 深度分析
        if self.llm and score > 60:
            self._llm_deep_market_analysis(signal)

        self.log_signal.emit(
            f"📊 市场分析: {signal.get('symbol','')} {direction} 评分{score:.0f} 置信度{self.state.market_confidence:.0%}",
            "INFO",
        )

    def _llm_deep_market_analysis(self, signal: Dict):
        """调用 LLM 做深度市场分析"""
        try:
            prompt = f"""你是加密货币交易分析师。分析以下扫描信号：

信号: {json.dumps(signal, ensure_ascii=False, default=str)[:500]}
市场置信度: {self.state.market_confidence:.0%}
多周期趋势: {json.dumps(self.timeframe_tracker.accuracy_by_timeframe(days=7) if self.timeframe_tracker else {}, ensure_ascii=False)}

给出简短评估（1-2句）：该信号的质量如何，主要风险是什么。"""
            reply = self.llm.chat([
                {"role": "system", "content": "你是加密货币分析师。只输出1-2句评估。"},
                {"role": "user", "content": prompt},
            ], timeout=60)
            if reply:
                self.state.market_report += f"\n\nAI评估: {reply}"
        except Exception:
            pass

    def _risk_analysis(self):
        """风控分析师：检查交易风险"""
        checks = []

        # 余额检查
        if self.state.balance < 10:
            checks.append("❌ 余额不足 ($10)")
            self.state.risk_passed = False
        else:
            checks.append(f"✅ 余额充足 (${self.state.balance:.0f})")

        # 持仓检查
        positions = self.state.positions
        if len(positions) >= 5:
            checks.append("⚠️ 持仓已达上限 (5)")
        else:
            checks.append(f"✅ 持仓数 ({len(positions)}/5)")

        # 同币种检查
        if self.state.symbol:
            same_symbol = [p for p in positions if p["instId"] == self.state.symbol]
            if same_symbol:
                checks.append(f"⚠️ 已持有 {self.state.symbol}")

        # 评分
        pass_count = sum(1 for c in checks if "✅" in c)
        warn_count = sum(1 for c in checks if "⚠️" in c)
        fail_count = sum(1 for c in checks if "❌" in c)

        self.state.risk_score = min(1.0, pass_count / max(len(checks), 1))
        self.state.risk_passed = fail_count == 0 and pass_count >= 2
        self.state.risk_report = "\n".join(checks)

        self.log_signal.emit(
            f"🛡 风控: {'✅通过' if self.state.risk_passed else '❌未通过'} (评分{self.state.risk_score:.0%})",
            "SUCCESS" if self.state.risk_passed else "WARNING",
        )

    def _make_decision(self):
        """协调器：综合分析 → 最终决策"""
        signal = self.state.top_signal
        if not signal:
            self.state.final_action = "HOLD"
            self.state.final_reason = "无信号"
            return

        direction = signal.get("direction", "")
        score = float(signal.get("score", 0) or 0)
        passed = self.state.risk_passed

        if not passed:
            self.state.final_action = "HOLD"
            self.state.final_reason = "风控未通过"
            self.decision_signal.emit(self.state.to_snapshot())
            return

        if direction in ("BUY", "LONG") and score >= 70 and self.state.market_confidence > 0.65:
            self.state.final_action = "BUY"
            self.state.final_size = round(self.state.balance * 0.05, 0)  # 5% 仓位
            self.state.final_reason = f"高评分信号({score:.0f}) + 风控通过"
        elif direction in ("SELL", "SHORT") and score >= 75:
            self.state.final_action = "SELL"
            self.state.final_size = round(self.state.balance * 0.03, 0)
            self.state.final_reason = f"空头信号({score:.0f}) + 风控通过"
        else:
            self.state.final_action = "HOLD"
            self.state.final_reason = f"信号评分不足 ({score:.0f}<70)"

        self.log_signal.emit(
            f"🎯 决策: {self.state.final_action} {self.state.symbol} "
            f"仓位${self.state.final_size:.0f} ({self.state.final_reason})",
            "SUCCESS" if self.state.final_action != "HOLD" else "INFO",
        )
        self.decision_signal.emit(self.state.to_snapshot())

    def _execute(self):
        """执行代理：执行交易（含注册器冲突检查）"""
        if not self.executor:
            self.log_signal.emit("❌ 无交易执行器", "ERROR")
            return

        symbol = self.state.symbol
        action = self.state.final_action
        size   = self.state.final_size

        # ── 注册器检查（仅开仓需要，平仓不受限） ──────────────────────────────
        if action == "BUY":
            if not position_registry.try_lock(symbol, 'MultiAgent'):
                _owner = position_registry.get_owner(symbol)
                self.log_signal.emit(
                    f"⛔ {symbol} 已由 {_owner} 持有，MultiAgent BUY 被注册器拒绝",
                    "WARNING",
                )
                return

        try:
            if action == "BUY":
                # 注册器前置检查（修复：SEL L路径绕过注册器）
                if not position_registry.try_lock(symbol, 'MultiAgent'):
                    _owner = position_registry.get_owner(symbol)
                    self.log_signal.emit(
                        f"⛔ {symbol} 由 {_owner} 持有，MultiAgent BUY 被拒绝", "WARNING")
                    return
                result = self.executor.execute_buy(symbol, position_ratio=0.05)
                if not (result and getattr(result, 'success', False)):
                    position_registry.release(symbol, 'MultiAgent')
            elif action == "SELL":
                # SELL 必须检查注册器：不允许平掉其他系统管理的仓位
                if not position_registry.try_lock(symbol, 'MultiAgent'):
                    _owner = position_registry.get_owner(symbol)
                    self.log_signal.emit(
                        f"⛔ {symbol} 由 {_owner} 持有，MultiAgent SELL 被拒绝", "WARNING")
                    return
                result = self.executor.execute_sell(symbol, size)
                if result and getattr(result, 'success', False):
                    pass  # 平仓成功，保持注册器所有权
                else:
                    position_registry.release(symbol, 'MultiAgent')
            else:
                return

            if result and getattr(result, 'success', False):
                self.log_signal.emit(f"✅ {action} {symbol} 已执行 ${size:.0f}", "SUCCESS")
                self.trade_signal.emit(symbol, action, size)
            else:
                msg = getattr(result, 'message', '未知错误') if result else '无响应'
                self.log_signal.emit(f"❌ 执行失败: {msg}", "ERROR")
        except Exception as e:
            # 异常时也要释放锁，防止死锁
            if action == "BUY":
                position_registry.release(symbol, 'MultiAgent')
            self.log_signal.emit(f"❌ 执行异常: {e}", "ERROR")
