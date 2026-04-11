"""
策略运行器
负责执行策略逻辑并生成交易信号
"""

import threading
import time
from typing import Dict, Optional, Callable
from dataclasses import dataclass
from enum import Enum

from PyQt5.QtCore import QObject, pyqtSignal


class SignalAction(Enum):
    """信号动作"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    STOP_LOSS = "STOP_LOSS"


@dataclass
class TradeSignal:
    """交易信号"""
    action: SignalAction
    inst_id: str
    price: float
    size: float
    confidence: float = 0.0
    reason: str = ""


class StrategyRunner(QObject):
    """策略运行器 - PyQt 信号版本"""

    # 信号
    log_signal = pyqtSignal(str, str)  # 日志信号
    trade_signal = pyqtSignal(str, str, float, float)  # 交易信号 (action, inst_id, price, size)
    finished = pyqtSignal()
    error_signal = pyqtSignal(str)

    def __init__(self, strategy_instance, inst_id: str, okx_client, trade_executor,
                 config: Dict = None):
        """
        初始化策略运行器

        Args:
            strategy_instance: 策略实例
            inst_id: 交易对 ID
            okx_client: OKX 客户端
            trade_executor: 交易执行器
            config: 策略配置
        """
        super().__init__()
        self.strategy = strategy_instance
        self.inst_id = inst_id
        self.okx_client = okx_client
        self.trade_executor = trade_executor
        self.config = config or {}
        self._stop_flag = False
        self._running = False

    def run(self):
        """运行策略主循环"""
        self._running = True
        self._stop_flag = False
        self.log_signal.emit("策略启动", "INFO")

        try:
            # 检查策略是否有 generate_signal 方法
            if not hasattr(self.strategy, 'generate_signal'):
                self.log_signal.emit("策略缺少 generate_signal 方法", "ERROR")
                self.finished.emit()
                return

            self.log_signal.emit(f"开始监控 {self.inst_id}", "INFO")

            # 主循环
            while not self._stop_flag:
                try:
                    # 获取市场数据
                    klines = self._get_klines()
                    if not klines:
                        time.sleep(5)
                        continue

                    # 生成交易信号
                    signal = self.strategy.generate_signal(klines)

                    if signal and signal.get('action') != 'HOLD':
                        self._handle_signal(signal)

                    time.sleep(1)  # 1 秒循环

                except Exception as e:
                    self.log_signal.emit(f"策略执行错误：{str(e)}", "ERROR")
                    time.sleep(5)

        except Exception as e:
            self.log_signal.emit(f"策略异常：{str(e)}", "ERROR")
            self.error_signal.emit(str(e))

        finally:
            self._running = False
            self.finished.emit()

    def stop(self):
        """停止策略"""
        self.log_signal.emit("正在停止策略...", "WARNING")
        self._stop_flag = True

    def _get_klines(self) -> Optional[Dict]:
        """获取 K 线数据"""
        try:
            # 获取多周期 K 线
            daily = self.okx_client.get_kline(self.inst_id, bar="1D", limit=100)
            hourly = self.okx_client.get_kline(self.inst_id, bar="1H", limit=500)
            m15 = self.okx_client.get_kline(self.inst_id, bar="15m", limit=200)

            if daily.get('code') != '0' or hourly.get('code') != '0':
                return None

            return {
                'daily': daily.get('data', []),
                'hourly': hourly.get('data', []),
                'm15': m15.get('data', [])
            }
        except Exception as e:
            self.log_signal.emit(f"获取 K 线失败：{str(e)}", "ERROR")
            return None

    def _handle_signal(self, signal: Dict):
        """处理交易信号"""
        try:
            action = signal.get('action', '')
            entry_price = signal.get('entry_price', 0)
            position_size = signal.get('position_size', 0.01)

            if not entry_price:
                # 获取当前价格
                ticker = self.okx_client.get_ticker(self.inst_id)
                if ticker and 'data' in ticker and len(ticker['data']) > 0:
                    entry_price = float(ticker['data'][0].get('last', 0))

            if action in ['BUY', 'BUY_LONG']:
                self.log_signal.emit(f"生成买入信号 @ {entry_price}", "TRADE")
                self.trade_signal.emit("BUY", self.inst_id, entry_price, position_size)

                # 执行买入
                result = self.trade_executor.execute_buy(
                    self.inst_id,
                    position_ratio=position_size
                )
                if result.success:
                    self.log_signal.emit(f"买入成功：{result.filled_size}", "SUCCESS")
                else:
                    self.log_signal.emit(f"买入失败：{result.message}", "ERROR")

            elif action in ['SELL', 'EXIT_LONG']:
                self.log_signal.emit(f"生成卖出信号 @ {entry_price}", "TRADE")
                self.trade_signal.emit("SELL", self.inst_id, entry_price, position_size)

                # 执行卖出
                positions = self.trade_executor.get_positions(self.inst_id)
                if self.inst_id in positions:
                    result = self.trade_executor.execute_sell(
                        self.inst_id,
                        positions[self.inst_id].size
                    )
                    if result.success:
                        self.log_signal.emit(f"卖出成功：{result.filled_size}", "SUCCESS")
                    else:
                        self.log_signal.emit(f"卖出失败：{result.message}", "ERROR")

            elif action in ['SHORT', 'SELL_SHORT']:
                self.log_signal.emit(f"生成做空信号 @ {entry_price}", "TRADE")
                self.trade_signal.emit("SHORT", self.inst_id, entry_price, position_size)

            elif action in ['COVER', 'EXIT_SHORT']:
                self.log_signal.emit(f"生成平空信号 @ {entry_price}", "TRADE")
                self.trade_signal.emit("COVER", self.inst_id, entry_price, position_size)

            elif action == 'STOP_LOSS':
                self.log_signal.emit(f"触发止损 @ {entry_price}", "WARNING")
                self.trade_signal.emit("STOP_LOSS", self.inst_id, entry_price, position_size)

        except Exception as e:
            self.log_signal.emit(f"处理信号失败：{str(e)}", "ERROR")


class SimpleStrategyRunner(QObject):
    """简单策略运行器 - 用于没有 generate_signal 方法的策略"""

    log_signal = pyqtSignal(str, str)
    trade_signal = pyqtSignal(str, str, float, float)
    finished = pyqtSignal()

    def __init__(self, strategy_instance, inst_id: str, okx_client, trade_executor,
                 config: Dict = None, interval: int = 60):
        """
        初始化简单策略运行器

        Args:
            strategy_instance: 策略实例
            inst_id: 交易对 ID
            okx_client: OKX 客户端
            trade_executor: 交易执行器
            config: 策略配置
            interval: 检查间隔 (秒)
        """
        super().__init__()
        self.strategy = strategy_instance
        self.inst_id = inst_id
        self.okx_client = okx_client
        self.trade_executor = trade_executor
        self.config = config or {}
        self.interval = interval
        self._stop_flag = False

    def run(self):
        """运行策略"""
        self.log_signal.emit("简单策略启动", "INFO")
        self.log_signal.emit(f"检查间隔：{self.interval}秒", "INFO")

        try:
            while not self._stop_flag:
                # 调用策略的 check 方法（如果有）
                if hasattr(self.strategy, 'check'):
                    result = self.strategy.check(self.inst_id)
                    if result:
                        self._handle_result(result)

                time.sleep(self.interval)

        except Exception as e:
            self.log_signal.emit(f"策略错误：{str(e)}", "ERROR")

        self.finished.emit()

    def stop(self):
        """停止策略"""
        self._stop_flag = True

    def _handle_result(self, result: Dict):
        """处理策略结果"""
        action = result.get('action', '')
        price = result.get('price', 0)
        size = result.get('size', 0)

        if action == 'buy':
            self.trade_signal.emit("BUY", self.inst_id, price, size)
            self.log_signal.emit(f"执行买入：{size} @ {price}", "TRADE")
        elif action == 'sell':
            self.trade_signal.emit("SELL", self.inst_id, price, size)
            self.log_signal.emit(f"执行卖出：{size} @ {price}", "TRADE")
