"""
风险管理模块 —— 统一止损强制执行 + 日亏损熔断 + 最大回撤保护。

所有交易策略和 ExecutionEngine 共用的风险控制层。
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class RiskLimits:
    """风险参数配置"""

    # 单笔止损（ATR 倍数）
    stop_loss_atr_mult: float = 2.0

    # 单品种最大敞口（占账户权益比例）
    max_exposure_per_symbol: float = 0.25

    # 最大并发持仓数
    max_concurrent_positions: int = 8

    # 日亏损熔断（占初始权益比例，0 表示不启用）
    daily_loss_limit_pct: float = 5.0

    # 最大回撤熔断（占峰值权益比例，0 表示不启用）
    max_drawdown_limit_pct: float = 15.0

    # 单日最大交易次数（0 表示不限制）
    max_daily_trades: int = 50

    # 最小持仓时间（秒，防止过度交易）
    min_hold_seconds: int = 30

    # 开盘/收盘保护（UTC 时间）
    avoid_open_minutes: int = 5       # 避免开盘后 N 分钟内交易
    avoid_close_minutes: int = 5      # 避免收盘前 N 分钟内交易
    candle_close_utc: int = 0         # 日线收盘 UTC 小时（0=00:00）


@dataclass
class DailyStats:
    """当日交易统计"""
    date: str = ""
    total_pnl: float = 0.0
    trade_count: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    gross_profit: float = 0.0
    gross_loss: float = 0.0


class RiskGuard:
    """风险守卫 —— 所有策略和交易引擎共用。

    用法:
        guard = RiskGuard(RiskLimits(daily_loss_limit_pct=5.0))
        guard.set_initial_equity(10000.0)

        # 每笔交易前
        if not guard.can_open_position(inst_id, equity):
            return  # 被熔断或超限

        # 计算止损
        sl_price = guard.calc_stop_loss(entry_price, atr_value, direction='long')

        # 记录交易结果
        guard.record_trade(pnl)

        # 检查是否需要熔断
        if guard.is_circuit_breaker_active():
            logger.warning("风险熔断已触发！")
    """

    def __init__(self, limits: Optional[RiskLimits] = None):
        self.limits = limits or RiskLimits()
        self._lock = threading.RLock()

        # 账户级别状态
        self._initial_equity: float = 0.0
        self._peak_equity: float = 0.0

        # 当日统计
        today = datetime.utcnow().strftime("%Y-%m-%d")
        self._daily_stats = DailyStats(date=today)

        # 熔断标志
        self._circuit_breaker: bool = False
        self._circuit_reason: str = ""

        # 当前持仓追踪（inst_id -> 持仓信息）
        self._positions: Dict[str, Dict] = {}

    # ── 初始化 ──────────────────────────────────

    def set_initial_equity(self, equity: float):
        """设置初始权益（策略启动时调用一次）。"""
        with self._lock:
            if self._initial_equity <= 0:
                self._initial_equity = equity
                self._peak_equity = equity
                logger.info(f"风险守卫初始化: 初始权益={equity:.2f}")

    def update_equity(self, equity: float):
        """更新当前权益（每个周期调用）。"""
        with self._lock:
            if equity > self._peak_equity:
                self._peak_equity = equity

    # ── 止损计算 ──────────────────────────────

    @staticmethod
    def calc_stop_loss(
        entry_price: float,
        atr_value: float,
        direction: str = "long",
        atr_mult: Optional[float] = None,
    ) -> float:
        """基于 ATR 计算止损价格。

        Args:
            entry_price: 入场价
            atr_value: 当前 ATR 值
            direction: 'long' 或 'short'
            atr_mult: ATR 倍数（None 则用默认 2.0）

        Returns:
            止损价格
        """
        mult = atr_mult if atr_mult is not None else 2.0
        if direction == "long":
            return entry_price - atr_value * mult
        else:
            return entry_price + atr_value * mult

    @staticmethod
    def calc_position_size(
        equity: float,
        entry_price: float,
        atr_value: float,
        risk_per_trade_pct: float = 1.0,
        atr_mult: float = 2.0,
        leverage: float = 1.0,
    ) -> float:
        """基于风险百分比计算仓位大小。

        size = (equity * risk_pct) / (atr * atr_mult) / leverage

        Args:
            equity: 账户权益
            entry_price: 入场价
            atr_value: ATR
            risk_per_trade_pct: 单笔风险比例（如 1.0 = 1%）
            atr_mult: ATR 止损倍数
            leverage: 杠杆倍数

        Returns:
            建议仓位大小（合约张数）
        """
        if equity <= 0 or entry_price <= 0 or atr_value <= 0:
            return 0.0
        risk_amount = equity * (risk_per_trade_pct / 100.0)
        stop_distance = atr_value * atr_mult
        if stop_distance <= 0:
            return 0.0
        size = risk_amount / stop_distance / leverage
        return size

    # ── 持仓管理 ──────────────────────────────

    def can_open_position(self, inst_id: str, equity: float) -> Tuple[bool, str]:
        """判断是否允许开新仓。

        Returns:
            (是否允许, 拒绝原因)
        """
        with self._lock:
            # 熔断检查
            if self._circuit_breaker:
                return False, f"熔断中: {self._circuit_reason}"

            # 回撤检查
            if self.limits.max_drawdown_limit_pct > 0 and self._peak_equity > 0:
                dd_pct = (self._peak_equity - equity) / self._peak_equity * 100.0
                if dd_pct >= self.limits.max_drawdown_limit_pct:
                    self._activate_circuit(f"最大回撤超过{self.limits.max_drawdown_limit_pct}% (当前{dd_pct:.2f}%)")
                    return False, self._circuit_reason

            # 日亏损熔断
            if self.limits.daily_loss_limit_pct > 0 and self._initial_equity > 0:
                loss_pct = (-self._daily_stats.total_pnl) / self._initial_equity * 100.0
                if loss_pct >= self.limits.daily_loss_limit_pct:
                    self._activate_circuit(f"日亏损超过{self.limits.daily_loss_limit_pct}%")
                    return False, self._circuit_reason

            # 并发持仓上限
            if len(self._positions) >= self.limits.max_concurrent_positions:
                return False, f"已达最大并发持仓数({self.limits.max_concurrent_positions})"

            # 单品种敞口
            if inst_id in self._positions:
                symbol_exposure = self._positions[inst_id].get("notional", 0) / max(equity, 1.0)
                if symbol_exposure >= self.limits.max_exposure_per_symbol:
                    return False, f"{inst_id}敞口已达上限({self.limits.max_exposure_per_symbol*100:.0f}%)"

            # 日交易次数上限
            if (self.limits.max_daily_trades > 0 and
                    self._daily_stats.trade_count >= self.limits.max_daily_trades):
                return False, f"已达日交易次数上限({self.limits.max_daily_trades})"

            # 开盘/收盘保护
            timing_ok, timing_reason = self._check_timing()
            if not timing_ok:
                return False, timing_reason

            return True, ""

    def register_position(self, inst_id: str, entry_price: float, notional: float,
                          sl_price: float, direction: str):
        """注册新持仓。"""
        with self._lock:
            self._positions[inst_id] = {
                "entry_price": entry_price,
                "sl_price": sl_price,
                "notional": notional,
                "direction": direction,
                "entry_time": time.time(),
            }

    def check_stop_loss(self, inst_id: str, current_price: float) -> Tuple[bool, str]:
        """检查是否触发止损。

        Returns:
            (是否触发, 原因)
        """
        with self._lock:
            pos = self._positions.get(inst_id)
            if not pos:
                return False, ""

            # 最小持仓时间保护
            hold_seconds = time.time() - pos.get("entry_time", 0)
            if hold_seconds < self.limits.min_hold_seconds:
                return False, ""

            sl = pos["sl_price"]
            direction = pos["direction"]

            if direction == "long" and current_price <= sl:
                return True, f"多头止损触发 ({current_price:.4f} <= {sl:.4f})"
            elif direction == "short" and current_price >= sl:
                return True, f"空头止损触发 ({current_price:.4f} >= {sl:.4f})"

            return False, ""

    def unregister_position(self, inst_id: str):
        """注销持仓。"""
        with self._lock:
            self._positions.pop(inst_id, None)

    # ── 交易记录 ──────────────────────────────

    def record_trade(self, pnl: float):
        """记录一笔已完成的交易结果。"""
        with self._lock:
            self._check_day_rollover()
            self._daily_stats.trade_count += 1
            self._daily_stats.total_pnl += pnl
            if pnl > 0:
                self._daily_stats.winning_trades += 1
                self._daily_stats.gross_profit += pnl
            else:
                self._daily_stats.losing_trades += 1
                self._daily_stats.gross_loss += abs(pnl)

    # ── 查询接口 ──────────────────────────────

    def is_circuit_breaker_active(self) -> bool:
        return self._circuit_breaker

    def get_circuit_reason(self) -> str:
        return self._circuit_reason

    def get_daily_stats(self) -> DailyStats:
        with self._lock:
            self._check_day_rollover()
            return self._daily_stats

    def get_active_positions(self) -> Dict[str, Dict]:
        with self._lock:
            return dict(self._positions)

    def reset_daily(self):
        """重置当日统计（手动/新一天开始）。"""
        with self._lock:
            today = datetime.utcnow().strftime("%Y-%m-%d")
            self._daily_stats = DailyStats(date=today)
            self._circuit_breaker = False
            self._circuit_reason = ""
            logger.info("当日风险统计已重置")

    def emergency_stop(self, reason: str = "手动紧急停止"):
        """紧急停止所有交易。"""
        with self._lock:
            self._circuit_breaker = True
            self._circuit_reason = reason
            logger.critical(f"紧急熔断触发: {reason}")

    # ── 内部方法 ──────────────────────────────

    def _activate_circuit(self, reason: str):
        self._circuit_breaker = True
        self._circuit_reason = reason
        logger.critical(f"风险熔断触发: {reason}")

    def _check_day_rollover(self):
        """检查是否跨日，自动重置日统计。"""
        today = datetime.utcnow().strftime("%Y-%m-%d")
        if self._daily_stats.date != today:
            self._daily_stats = DailyStats(date=today)

    def _check_timing(self) -> Tuple[bool, str]:
        """检查当前时间是否在交易保护窗口内。"""
        now = datetime.utcnow()
        minute = now.minute + now.hour * 60

        # 开盘保护
        open_start = self.limits.candle_close_utc * 60
        open_end = open_start + self.limits.avoid_open_minutes
        if open_start <= minute < open_end:
            return False, "开盘保护窗口内"

        # 收盘保护（往前推）
        close_start = open_start - self.limits.avoid_close_minutes
        if close_start <= minute < open_start and open_start > 0:
            return False, "收盘保护窗口内"

        return True, ""
