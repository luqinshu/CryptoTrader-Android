"""
模拟交易策略运行器

继承 StrategyRunner，将 trade_executor 替换为 PaperTradeEngine。
在每个主循环周期额外执行：
  1. 更新所有模拟持仓的实时价格
  2. 手动检查止盈/止损（交易所不会代劳）
  3. 会话结束时保存最终报告并广播汇总
"""

import time
from typing import Dict, Optional, Any

from src.qt_compat import Signal
from src.strategy.runner import StrategyRunner, AutoTradeCampaign
from src.trading.paper_engine import PaperTradeEngine


class PaperStrategyRunner(StrategyRunner):
    """
    模拟交易策略运行器。

    使用 PaperTradeEngine 取代真实 TradeExecutor，
    其余开仓 / 加仓 / 风控离场逻辑完全复用父类实现。
    """

    # 额外信号：推送模拟持仓实时状态给 UI
    paper_state_signal = Signal(dict)   # {'positions': [...], 'summary': {...}}

    def __init__(
        self,
        strategy_instance,
        inst_id: str,
        okx_client,
        paper_engine: PaperTradeEngine,
        config: Dict = None,
    ):
        # 父类需要 trade_executor，直接传入 paper_engine（同接口）
        super().__init__(
            strategy_instance=strategy_instance,
            inst_id=inst_id,
            okx_client=okx_client,
            trade_executor=paper_engine,
            config=config,
        )
        self._paper = paper_engine

    # ── 覆盖主循环，加入模拟特有逻辑 ────────────────────────────────────────

    def run(self):
        self._running = True
        self._stop_flag = False
        self.log_signal.emit("[模拟交易] 策略启动，资金不与真实账户挂钩", "INFO")
        self.log_signal.emit(
            f"模拟初始资金：{self._paper.initial_capital:,.2f} USDT，"
            f"手续费={self._paper._fee_rate * 100:.3f}%，"
            f"滑点={self._paper._slip_rate * 100:.3f}%",
            "INFO"
        )

        try:
            if not hasattr(self.strategy, 'generate_signal'):
                self.log_signal.emit("策略缺少 generate_signal 方法", "ERROR")
                self.finished.emit()
                return

            self.log_signal.emit(f"[模拟] 开始监控 {self.inst_id}", "INFO")

            while not self._stop_flag:
                try:
                    # 1. 更新持仓浮动价格
                    self._paper.update_prices()

                    # 2. 手动检查止盈 / 止损（真实交易所会代劳，模拟需要自己做）
                    self._check_paper_tp_sl()

                    # 3. 获取 K 线，运行策略信号 → 开 / 平仓
                    klines = self._get_klines()
                    if klines:
                        raw_signal = self.strategy.generate_signal(klines)
                        self._process_auto_trade_cycle(raw_signal, klines)

                    # 4. 广播模拟状态给 UI
                    self._emit_paper_state()

                    time.sleep(max(self._config_int('auto_loop_interval_seconds', 5), 1))

                except Exception as e:
                    self.log_signal.emit(f"[模拟] 执行错误：{str(e)}", "ERROR")
                    time.sleep(5)

        except Exception as e:
            self.log_signal.emit(f"[模拟] 策略异常：{str(e)}", "ERROR")
            self.error_signal.emit(str(e))

        finally:
            self._running = False
            path = self._paper.save_final()
            self.log_signal.emit(f"[模拟] 会话已保存：{path}", "INFO")
            self._emit_paper_state()
            self.finished.emit()

    # ── 模拟专属：TP/SL 手动检查 ────────────────────────────────────────────

    def _check_paper_tp_sl(self):
        for inst_id in list(self._paper._positions.keys()):
            reason = self._paper.check_tp_sl(inst_id)
            if not reason:
                continue
            pos = self._paper._positions.get(inst_id)
            if not pos:
                continue
            self.log_signal.emit(f"[模拟] {inst_id} {reason}，自动平仓", "WARNING")
            result = self._paper.execute_stop_loss(inst_id, exit_reason=reason)
            if result.success:
                self.log_signal.emit(f"[模拟] {result.message}", "SUCCESS")
                self.trade_signal.emit(
                    "SELL" if pos.direction == 'LONG' else "COVER",
                    inst_id, pos.current_price, pos.size,
                )
                self._campaign = None
                self._pending_signal = None
            else:
                self.log_signal.emit(f"[模拟] 自动平仓失败：{result.message}", "ERROR")

    # ── 模拟专属：推送状态 ───────────────────────────────────────────────────

    def _emit_paper_state(self):
        try:
            self.paper_state_signal.emit({
                'positions': self._paper.get_open_positions_info(),
                'summary': self._paper.get_summary(),
                'trades': [vars(t) for t in self._paper._trades],
            })
        except Exception:
            pass
