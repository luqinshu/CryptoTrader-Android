"""
回测引擎模块
支持策略历史数据回测和绩效分析
"""

import numpy as np
import pandas as pd
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.scanner.base_scanner import ScannerSymbol
from src.strategy.runner import StrategyRunner
from src.trading.executor import OrderResult, PositionInfo, PositionSide


class TradeDirection(Enum):
    """交易方向"""
    LONG = "long"
    SHORT = "short"


@dataclass
class Trade:
    """交易记录"""
    entry_time: datetime
    exit_time: Optional[datetime]
    direction: TradeDirection
    entry_price: float
    exit_price: Optional[float]
    size: float
    raw_entry_price: float = 0.0
    raw_exit_price: float = 0.0
    pnl: float = 0.0
    pnl_percent: float = 0.0
    entry_reason: str = ""
    exit_reason: str = ""
    fees: float = 0.0
    slippage_cost: float = 0.0
    funding_cost: float = 0.0


@dataclass
class BacktestResult:
    """回测结果"""
    # 基本信息
    strategy_name: str
    inst_id: str
    start_date: datetime
    end_date: datetime
    initial_capital: float
    
    # 绩效指标
    total_return: float = 0.0  # 总收益率
    annual_return: float = 0.0  # 年化收益率
    max_drawdown: float = 0.0  # 最大回撤
    sharpe_ratio: float = 0.0  # 夏普比率
    sortino_ratio: float = 0.0  # 索提诺比率
    win_rate: float = 0.0  # 胜率
    profit_factor: float = 0.0  # 盈亏比
    total_trades: int = 0  # 总交易次数
    winning_trades: int = 0  # 盈利交易数
    losing_trades: int = 0  # 亏损交易数
    avg_win: float = 0.0  # 平均盈利
    avg_loss: float = 0.0  # 平均亏损
    avg_trade_duration: float = 0.0  # 平均持仓时间 (小时)
    live_readiness_score: float = 0.0  # 实盘准入评分
    max_consecutive_losses: int = 0  # 最大连续亏损次数
    
    # 风险指标（扩展）
    calmar_ratio: float = 0.0       # 卡玛比率 = 年化收益 / 最大回撤
    var_95: float = 0.0             # VaR-95%：5th 百分位单笔亏损率
    recovery_factor: float = 0.0   # 恢复因子 = 总收益 / 最大回撤
    ulcer_index: float = 0.0       # 溃疡指数：回撤深度均方根

    # 最终资金
    final_capital: float = 0.0
    total_pnl: float = 0.0

    # 交易记录
    trades: List[Trade] = field(default_factory=list)
    
    # 权益曲线数据
    equity_curve: List[Tuple[datetime, float]] = field(default_factory=list)
    
    # 详细统计
    stats: Dict[str, Any] = field(default_factory=dict)


class BacktestEngine:
    """回测引擎"""

    def __init__(self, initial_capital: float = 10000.0):
        """
        初始化回测引擎

        Args:
            initial_capital: 初始资金
        """
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.position = 0.0
        self.position_price = 0.0
        self.position_direction = None
        self.trades: List[Trade] = []
        self.equity_curve: List[Tuple[datetime, float]] = []
        self.current_trade: Optional[Trade] = None
        self.cost_config: Dict[str, float] = {}
        self._equity_sample_every = 1
        self._equity_counter = 0
        self._peak_equity = initial_capital
        self._max_drawdown_pct = 0.0

    def reset(self):
        """重置引擎状态"""
        self.capital = self.initial_capital
        self.position = 0.0
        self.position_price = 0.0
        self.position_direction = None
        self.trades = []
        self.equity_curve = []
        self.current_trade = None
        self.cost_config = {}
        self._equity_sample_every = 1
        self._equity_counter = 0
        self._peak_equity = self.initial_capital
        self._max_drawdown_pct = 0.0

    def configure_equity_sampling(self, expected_points: int, max_points: int = 3000):
        """根据预估回放长度配置权益曲线采样频率。"""
        if expected_points <= 0 or max_points <= 0:
            self._equity_sample_every = 1
            return
        self._equity_sample_every = max(1, int(expected_points / max_points))

    def set_cost_config(self, config: Dict[str, float]):
        """设置回测成本模型"""
        self.cost_config = config or {}

    def _cost_rate(self, key: str, default: float = 0.0) -> float:
        try:
            return float(self.cost_config.get(key, default) or 0.0) / 100.0
        except (TypeError, ValueError):
            return default

    def _apply_entry_costs(self, price: float, direction: TradeDirection) -> Tuple[float, float, float]:
        slippage = self._cost_rate('slippage_pct')
        impact = self._cost_rate('market_impact_pct')
        adjusted = price * (1 + slippage + impact) if direction == TradeDirection.LONG else price * (1 - slippage - impact)
        return adjusted, abs(adjusted - price), 0.0

    def _apply_exit_costs(self, price: float, direction: TradeDirection) -> Tuple[float, float]:
        slippage = self._cost_rate('slippage_pct')
        impact = self._cost_rate('market_impact_pct')
        adjusted = price * (1 - slippage - impact) if direction == TradeDirection.LONG else price * (1 + slippage + impact)
        return adjusted, abs(adjusted - price)

    def _fee(self, notional: float) -> float:
        return abs(notional) * self._cost_rate('fee_pct', 0.05)

    def _funding_cost(self, trade: Trade, exit_time: datetime) -> float:
        if not trade.entry_time:
            return 0.0
        funding_rate = self._cost_rate('funding_rate_8h_pct')
        if funding_rate <= 0:
            return 0.0
        hours = max((exit_time - trade.entry_time).total_seconds() / 3600.0, 0.0)
        intervals = hours / 8.0
        notional = abs(trade.entry_price * trade.size)
        # 简化为持仓方向都支付资金费，偏保守。
        return notional * funding_rate * intervals

    def execute_buy(self, timestamp: datetime, price: float, size: float,
                    position_ratio: float = None, entry_reason: str = "") -> bool:
        """
        执行买入

        Args:
            timestamp: 时间戳
            price: 价格
            size: 数量 (如果为 None，则根据 position_ratio 计算)
            position_ratio: 仓位比例

        Returns:
            是否成功
        """
        if self.position != 0 and self.position_direction != TradeDirection.LONG:
            return False

        # 计算可买入数量
        if size is None:
            ratio = position_ratio or 1.0
            available_capital = self.capital * ratio
            size = available_capital / price

        raw_price = price
        price, slippage_cost, _ = self._apply_entry_costs(price, TradeDirection.LONG)
        cost = size * price
        fee = self._fee(cost)
        if cost > self.capital:
            size = max(0, self.capital - fee) / price if price > 0 else 0
            if size <= 0:
                return False
            cost = size * price
            fee = self._fee(cost)
        if cost + fee > self.capital:
            return False

        self.capital -= cost + fee
        if self.position > 0 and self.position_direction == TradeDirection.LONG and self.current_trade:
            prev_size = self.position
            total_size = prev_size + size
            self.position_price = ((prev_size * self.position_price) + (size * price)) / total_size
            self.position = total_size
            self.current_trade.entry_price = self.position_price
            self.current_trade.size = total_size
            self.current_trade.fees += fee
            self.current_trade.slippage_cost += slippage_cost * size
            existing_reason = self.current_trade.entry_reason or ""
            self.current_trade.entry_reason = f"{existing_reason} / {entry_reason}".strip(" /")
            setattr(self.current_trade, "_entry_notional", float(getattr(self.current_trade, "_entry_notional", prev_size * self.position_price)) + cost)
        else:
            self.position = size
            self.position_price = price
            self.position_direction = TradeDirection.LONG

            # 记录交易
            self.current_trade = Trade(
                entry_time=timestamp,
                exit_time=None,
                direction=TradeDirection.LONG,
                entry_price=price,
                exit_price=None,
                size=size,
                raw_entry_price=raw_price,
                entry_reason=entry_reason,
                fees=fee,
                slippage_cost=slippage_cost * size
            )
            setattr(self.current_trade, "_entry_notional", cost)

        return True

    def execute_sell(self, timestamp: datetime, price: float, size: float = None,
                     exit_reason: str = "") -> bool:
        """
        执行卖出

        Args:
            timestamp: 时间戳
            price: 价格
            size: 数量 (如果为 None，则全部卖出)

        Returns:
            是否成功
        """
        if self.position <= 0 or self.position_direction != TradeDirection.LONG:
            return False

        if size is None or size >= self.position:
            size = self.position

        # 计算盈亏
        raw_price = price
        price, exit_slippage_cost = self._apply_exit_costs(price, TradeDirection.LONG)
        gross_pnl = (price - self.position_price) * size
        exit_fee = self._fee(size * price)
        funding_cost = self._funding_cost(self.current_trade, timestamp) if self.current_trade else 0.0
        pnl = gross_pnl - exit_fee - funding_cost
        pnl_percent = (price - self.position_price) / self.position_price * 100

        # 更新资金
        self.capital += size * price - exit_fee - funding_cost
        self.position -= size

        # 记录交易
        if self.current_trade:
            self.current_trade.pnl += pnl
            self.current_trade.fees += exit_fee
            self.current_trade.slippage_cost += exit_slippage_cost * size
            self.current_trade.funding_cost += funding_cost
            if self.position <= 0:
                self.current_trade.exit_time = timestamp
                self.current_trade.exit_price = price
                self.current_trade.raw_exit_price = raw_price
                entry_notional = float(getattr(self.current_trade, "_entry_notional", max(self.current_trade.entry_price * max(self.current_trade.size, 1e-9), 1e-9)))
                self.current_trade.pnl_percent = (self.current_trade.pnl / entry_notional * 100.0) if entry_notional > 0 else pnl_percent
                self.current_trade.exit_reason = exit_reason or "止盈/止损/信号"
                self.trades.append(self.current_trade)
                self.current_trade = None

        if self.position <= 0:
            self.position_direction = None

        return True

    def execute_short(self, timestamp: datetime, price: float, size: float,
                      position_ratio: float = None, entry_reason: str = "") -> bool:
        """
        执行做空

        Args:
            timestamp: 时间戳
            price: 价格
            size: 数量
            position_ratio: 仓位比例

        Returns:
            是否成功
        """
        if self.position != 0 and self.position_direction != TradeDirection.SHORT:
            return False

        if size is None:
            ratio = position_ratio or 1.0
            available_capital = self.capital * ratio
            size = available_capital / price

        raw_price = price
        price, slippage_cost, _ = self._apply_entry_costs(price, TradeDirection.SHORT)
        notional = size * price
        fee = self._fee(notional)
        # SHORT 做空：收到卖出所得（notional），扣除手续费
        self.capital += notional - fee
        if self.position > 0 and self.position_direction == TradeDirection.SHORT and self.current_trade:
            prev_size = self.position
            total_size = prev_size + size
            self.position_price = ((prev_size * self.position_price) + (size * price)) / total_size
            self.position = total_size
            self.current_trade.entry_price = self.position_price
            self.current_trade.size = total_size
            self.current_trade.fees += fee
            self.current_trade.slippage_cost += slippage_cost * size
            existing_reason = self.current_trade.entry_reason or ""
            self.current_trade.entry_reason = f"{existing_reason} / {entry_reason}".strip(" /")
            setattr(self.current_trade, "_entry_notional", float(getattr(self.current_trade, "_entry_notional", prev_size * self.position_price)) + notional)
        else:
            self.position = size
            self.position_price = price
            self.position_direction = TradeDirection.SHORT

            # 记录交易
            self.current_trade = Trade(
                entry_time=timestamp,
                exit_time=None,
                direction=TradeDirection.SHORT,
                entry_price=price,
                exit_price=None,
                size=size,
                raw_entry_price=raw_price,
                entry_reason=entry_reason,
                fees=fee,
                slippage_cost=slippage_cost * size
            )
            setattr(self.current_trade, "_entry_notional", notional)

        return True

    def execute_cover(self, timestamp: datetime, price: float, size: float = None,
                      exit_reason: str = "") -> bool:
        """
        执行平空

        Args:
            timestamp: 时间戳
            price: 价格
            size: 数量

        Returns:
            是否成功
        """
        if self.position <= 0 or self.position_direction != TradeDirection.SHORT:
            return False

        if size is None or size >= self.position:
            size = self.position

        # 计算盈亏 (做空：价格下跌盈利)
        raw_price = price
        price, exit_slippage_cost = self._apply_exit_costs(price, TradeDirection.SHORT)
        gross_pnl = (self.position_price - price) * size
        exit_fee = self._fee(size * price)
        funding_cost = 0.0
        if self.current_trade and self.position <= size:
            funding_cost = self._funding_cost(self.current_trade, timestamp)
        pnl = gross_pnl - exit_fee - funding_cost
        pnl_percent = (self.position_price - price) / self.position_price * 100

        # 更新资金：平空需回购（支付 buyback + 费用）
        buyback = size * price
        self.capital -= buyback + exit_fee + funding_cost
        self.position -= size

        # 记录交易
        if self.current_trade:
            self.current_trade.pnl += pnl
            self.current_trade.fees += exit_fee
            self.current_trade.slippage_cost += exit_slippage_cost * size
            self.current_trade.funding_cost += funding_cost
            if self.position <= 0:
                self.current_trade.exit_time = timestamp
                self.current_trade.exit_price = price
                self.current_trade.raw_exit_price = raw_price
                entry_notional = float(getattr(self.current_trade, "_entry_notional", max(self.current_trade.entry_price * max(self.current_trade.size, 1e-9), 1e-9)))
                self.current_trade.pnl_percent = (self.current_trade.pnl / entry_notional * 100.0) if entry_notional > 0 else pnl_percent
                self.current_trade.exit_reason = exit_reason or "止盈/止损/信号"
                self.trades.append(self.current_trade)
                self.current_trade = None

        if self.position <= 0:
            self.position_direction = None

        return True

    def update_equity(self, timestamp: datetime, current_price: float, force_record: bool = False):
        """更新权益曲线"""
        if self.position > 0:
            if self.position_direction == TradeDirection.LONG:
                equity = self.capital + self.position * current_price
            else:
                # SHORT: capital 已含做空所得 notional，正确公式 = capital - 未实现浮动
                equity = self.capital - self.position * current_price
        else:
            equity = self.capital

        self._equity_counter += 1
        if equity > self._peak_equity:
            self._peak_equity = equity
        if self._peak_equity > 0:
            drawdown_pct = (self._peak_equity - equity) / self._peak_equity * 100.0
            if drawdown_pct > self._max_drawdown_pct:
                self._max_drawdown_pct = drawdown_pct

        if force_record or self._equity_counter == 1 or self._equity_counter % self._equity_sample_every == 0:
            self.equity_curve.append((timestamp, equity))
        return equity

    def get_current_equity(self, current_price: float) -> float:
        """获取当前权益"""
        if self.position > 0:
            if self.position_direction == TradeDirection.LONG:
                return self.capital + self.position * current_price
            else:
                return self.capital - self.position * current_price
        return self.capital


class _BacktestOkxView:
    """供 StrategyRunner 读取的轻量行情视图。"""

    def __init__(self):
        self.current_price = 0.0

    def set_price(self, price: float):
        self.current_price = float(price or 0.0)

    def get_ticker(self, inst_id: str) -> Dict[str, Any]:
        return {
            'code': '0',
            'data': [{'last': str(self.current_price)}],
        }


class _BacktestTradeExecutorAdapter:
    """把 BacktestEngine 适配成 StrategyRunner 需要的执行接口。"""

    def __init__(self, engine: BacktestEngine, inst_id: str):
        self.engine = engine
        self.inst_id = inst_id
        self.current_time: Optional[datetime] = None
        self.current_price: float = 0.0
        self.default_leverage: int = 1
        self._pending_exit_reason: str = ""

    def set_market(self, timestamp: datetime, price: float, leverage: int = 1):
        self.current_time = timestamp
        self.current_price = float(price or 0.0)
        self.default_leverage = max(int(leverage or 1), 1)

    def set_next_exit_reason(self, reason: str):
        self._pending_exit_reason = str(reason or "")

    def get_usdt_balance(self) -> float:
        return float(self.engine.get_current_equity(self.current_price))

    def get_positions(self, inst_id: str = None) -> Dict[str, PositionInfo]:
        if inst_id and inst_id != self.inst_id:
            return {}
        if self.engine.position <= 0 or self.engine.position_direction is None:
            return {}
        entry = float(self.engine.position_price or 0.0)
        cur = float(self.current_price or entry)
        size = float(self.engine.position or 0.0)
        if self.engine.position_direction == TradeDirection.LONG:
            side = PositionSide.LONG
            upnl = (cur - entry) * size
            pnl_pct = ((cur - entry) / entry * 100.0) if entry > 0 else 0.0
        else:
            side = PositionSide.SHORT
            upnl = (entry - cur) * size
            pnl_pct = ((entry - cur) / entry * 100.0) if entry > 0 else 0.0
        return {
            self.inst_id: PositionInfo(
                inst_id=self.inst_id,
                side=side,
                size=size,
                entry_price=entry,
                current_price=cur,
                unrealized_pnl=upnl,
                pnl_percent=pnl_pct,
                leverage=float(self.default_leverage),
                notional_usd=abs(size * cur),
            )
        }

    def execute_entry(
        self,
        inst_id: str,
        direction: str,
        usdt_amount: float,
        leverage: int = 1,
        tp_pct: float = 0.05,
        sl_pct: float = 0.03,
        order_type: str = "market",
        price: Optional[float] = None,
        reason: str = "",
    ) -> OrderResult:
        if inst_id != self.inst_id or self.current_time is None:
            return OrderResult(False, message="回测执行器未就绪")
        px = float(price or self.current_price or 0.0)
        if px <= 0:
            return OrderResult(False, message="回测价格无效")
        lev = max(int(leverage or 1), 1)
        self.default_leverage = lev
        size = (float(usdt_amount or 0.0) * lev) / px
        if size <= 0:
            return OrderResult(False, message="回测下单数量无效")
        normalized = str(direction or "").upper()
        if normalized in {"BUY", "LONG"}:
            ok = self.engine.execute_buy(self.current_time, px, size, entry_reason=reason)
        elif normalized in {"SELL", "SHORT"}:
            ok = self.engine.execute_short(self.current_time, px, size, entry_reason=reason)
        else:
            return OrderResult(False, message=f"不支持的回测方向: {direction}")
        return OrderResult(ok, message="ok" if ok else "state_rejected", filled_size=size, filled_price=px)

    def execute_sell(self, inst_id: str, quantity: float, exit_reason: str = "") -> OrderResult:
        if inst_id != self.inst_id or self.current_time is None:
            return OrderResult(False, message="回测执行器未就绪")
        reason = exit_reason or self._pending_exit_reason or "止盈/止损/信号"
        self._pending_exit_reason = ""
        ok = self.engine.execute_sell(self.current_time, self.current_price, quantity, exit_reason=reason)
        return OrderResult(ok, message="ok" if ok else "close_rejected", filled_size=quantity, filled_price=self.current_price)

    def execute_cover(self, inst_id: str, quantity: float, exit_reason: str = "") -> OrderResult:
        if inst_id != self.inst_id or self.current_time is None:
            return OrderResult(False, message="回测执行器未就绪")
        reason = exit_reason or self._pending_exit_reason or "止盈/止损/信号"
        self._pending_exit_reason = ""
        ok = self.engine.execute_cover(self.current_time, self.current_price, quantity, exit_reason=reason)
        return OrderResult(ok, message="ok" if ok else "close_rejected", filled_size=quantity, filled_price=self.current_price)

    def close_position(self, inst_id: str, exit_reason: str = "") -> OrderResult:
        positions = self.get_positions(inst_id)
        pos = positions.get(inst_id)
        if not pos:
            return OrderResult(False, message="无回测持仓")
        if pos.side == PositionSide.LONG:
            return self.execute_sell(inst_id, pos.size, exit_reason=exit_reason)
        return self.execute_cover(inst_id, pos.size, exit_reason=exit_reason)

    def execute_stop_loss(self, inst_id: str, exit_reason: str = "") -> OrderResult:
        return self.close_position(inst_id, exit_reason=exit_reason)

    def close_position_partial(self, inst_id: str, ratio: float) -> OrderResult:
        positions = self.get_positions(inst_id)
        pos = positions.get(inst_id)
        if not pos:
            return OrderResult(False, message="无回测持仓")
        qty = pos.size * float(ratio or 0.0)
        if qty <= 0:
            return OrderResult(False, message="部分平仓数量无效")
        if pos.side == PositionSide.LONG:
            return self.execute_sell(inst_id, qty, exit_reason="分批减仓")
        return self.execute_cover(inst_id, qty, exit_reason="分批减仓")


class _StateMachineBacktestRunner(StrategyRunner):
    """回测专用状态机运行器，直接复用 StrategyRunner 逻辑。"""

    def __init__(self, strategy_instance, inst_id: str, okx_view, trade_executor, config: Dict = None):
        super().__init__(strategy_instance, inst_id, okx_view, trade_executor, config or {})
        self._clock_ts: float = 0.0
        self.metrics: Dict[str, Any] = {
            'pilot_trade_count': 0,
            'pilot_win_count': 0,
            'add_on_trade_count': 0,
            'add_on_win_count': 0,
            'first_principle_trigger_count': 0,
            'campaign_close_count': 0,
        }

    def _now_ts(self) -> float:
        return self._clock_ts or time.time()

    def set_clock(self, timestamp: datetime):
        self._clock_ts = timestamp.timestamp()

    def step(self, timestamp: datetime, raw_signal: Dict[str, Any], klines: Dict[str, Any]):
        self.set_clock(timestamp)
        self._process_auto_trade_cycle(raw_signal, klines)
        # 兜底：即使 market snapshot 无效，用执行器价格直接检查止损和超时
        self._failsafe_exit_check(timestamp, klines)

    def _failsafe_exit_check(self, timestamp: datetime, klines: Dict[str, Any]):
        """不依赖 market snapshot 的出场检查（止损 → 移动止盈 → 超时）。"""
        if not self._campaign:
            return
        pos = self.trade_executor.get_positions(self.inst_id).get(self.inst_id)
        if not pos:
            self._campaign = None
            return
        px = float(getattr(pos, 'current_price', 0) or 0)
        entry = float(getattr(pos, 'entry_price', 0) or 0)
        side = getattr(getattr(pos, 'side', None), 'name', '').upper()
        cost_line = self._campaign.stage1_cost_line
        campaign = self._campaign

        # 补算缺失的止损线
        if not cost_line or cost_line <= 0:
            if entry > 0 and px > 0:
                cost_line = self._compute_cost_line(entry, side,
                    self._rows_to_df(klines.get('m3', [])))
                campaign.stage1_cost_line = cost_line

        # ── ① ATR 止损线 ──────────────────────────────────────────
        if (side == 'LONG' and px <= cost_line > 0) or \
           (side == 'SHORT' and px >= cost_line > 0):
            self._close_position(
                f"ATR止损：{side} px={px:.2f} 触发止损线={cost_line:.2f}")
            return

        # ── ② 移动止盈（时间衰减：持仓越久，回撤容忍度越低）───────
        hold_h = (self._now_ts() - campaign.opened_at) / 3600
        base_trail = self._config_float('trail_stop_pct', 3.0) / 100.0
        # 时间衰减：12h→3%, 24h→2%, 36h→1.2%, 48h→0.8%
        if hold_h < 12:
            trail_pct = base_trail
        elif hold_h < 24:
            trail_pct = max(base_trail * 0.65, 0.015)
        elif hold_h < 36:
            trail_pct = max(base_trail * 0.40, 0.010)
        else:
            trail_pct = max(base_trail * 0.25, 0.006)

        if side == 'LONG':
            if px > (campaign.peak_profit_price or entry):
                campaign.peak_profit_price = px
            peak = campaign.peak_profit_price or px
            if peak > entry and (peak - px) / peak >= trail_pct:
                self._close_position(
                    f"移动止盈(衰减{trail_pct*100:.1f}%)：LONG 峰值{peak:.2f}"
                    f"回撤{(peak-px)/peak*100:.2f}%触发 持仓{hold_h:.1f}h")
                return
        else:
            trough = campaign.peak_profit_price
            if not trough or px < trough:
                campaign.peak_profit_price = px
                trough = px
            if entry > trough and (px - trough) / trough >= trail_pct:
                self._close_position(
                    f"移动止盈(衰减{trail_pct*100:.1f}%)：SHORT 谷值{trough:.2f}"
                    f"反弹{(px-trough)/trough*100:.2f}%触发 持仓{hold_h:.1f}h")
                return

        # ── ③ 超时强平（最后防线，时限放宽到72h）─────────────────
        max_hours = self._config_float('max_hold_hours', 72.0)
        if max_hours > 0 and hold_h >= max_hours:
            pnl_pct = ((px - entry) / entry * 100) if side == 'LONG' else \
                      ((entry - px) / entry * 100)
            self._close_position(
                f"超时强平：{side} 持仓{hold_h:.1f}h PnL={pnl_pct:+.2f}%")

    def _try_open_pilot_position(self, market: Dict):
        before = self._campaign is not None
        super()._try_open_pilot_position(market)
        if not before and self._campaign is not None:
            self.metrics['pilot_trade_count'] += 1

    def _process_campaign_add_on(self, market: Dict):
        before = bool(self._campaign and self._campaign.stage2_filled)
        super()._process_campaign_add_on(market)
        after = bool(self._campaign and self._campaign.stage2_filled)
        if not before and after:
            self.metrics['add_on_trade_count'] += 1

    def _close_position(self, reason: str):
        had_campaign = self._campaign is not None
        had_stage2 = bool(self._campaign and self._campaign.stage2_filled)
        if '第一原则' in str(reason or ''):
            self.metrics['first_principle_trigger_count'] += 1
        if hasattr(self.trade_executor, 'set_next_exit_reason'):
            self.trade_executor.set_next_exit_reason(reason)
        super()._close_position(reason)
        if had_campaign:
            self.metrics['campaign_close_count'] += 1
            if self.trade_executor.engine.trades:
                last_trade = self.trade_executor.engine.trades[-1]
                if float(getattr(last_trade, 'pnl', 0.0) or 0.0) > 0:
                    self.metrics['pilot_win_count'] += 1
                    if had_stage2:
                        self.metrics['add_on_win_count'] += 1


class BacktestAnalyzer:
    """回测结果分析器"""

    @staticmethod
    def _contains_any(text: str, keywords: List[str]) -> bool:
        hay = str(text or "")
        return any(keyword in hay for keyword in keywords)

    @staticmethod
    def analyze(result: BacktestResult) -> BacktestResult:
        """
        分析回测结果，计算各项指标

        Args:
            result: 回测结果

        Returns:
            包含完整指标的回测结果
        """
        if not result.trades:
            return result

        # 计算基础统计
        trades_df = pd.DataFrame([{
            'pnl': t.pnl,
            'pnl_percent': t.pnl_percent,
            'duration': (t.exit_time - t.entry_time).total_seconds() / 3600  # 小时
        } for t in result.trades if t.exit_time])

        if len(trades_df) == 0:
            return result

        # 胜率
        winning = trades_df[trades_df['pnl'] > 0]
        losing = trades_df[trades_df['pnl'] <= 0]

        result.winning_trades = len(winning)
        result.losing_trades = len(losing)
        result.win_rate = len(winning) / len(trades_df) * 100 if len(trades_df) > 0 else 0

        # 平均盈亏
        result.avg_win = winning['pnl'].mean() if len(winning) > 0 else 0
        result.avg_loss = abs(losing['pnl'].mean()) if len(losing) > 0 else 0

        # 盈亏比
        result.profit_factor = result.avg_win / result.avg_loss if result.avg_loss > 0 else 0
        total_profit = winning['pnl'].sum() if len(winning) > 0 else 0.0
        total_loss = abs(losing['pnl'].sum()) if len(losing) > 0 else 0.0
        if total_loss > 0:
            result.profit_factor = total_profit / total_loss

        # 平均持仓时间
        result.avg_trade_duration = trades_df['duration'].mean()

        # 总收益
        result.total_pnl = sum(t.pnl for t in result.trades if t.exit_time)
        result.total_return = (result.final_capital - result.initial_capital) / result.initial_capital * 100

        # 年化收益率
        days = (result.end_date - result.start_date).days
        if days > 0:
            result.annual_return = ((1 + result.total_return / 100) ** (365 / days) - 1) * 100

        # 夏普比率 (假设无风险利率为 0)
        if len(trades_df) > 1:
            returns = trades_df['pnl_percent']
            result.sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std() if returns.std() > 0 else 0

            # 索提诺比率
            downside_returns = returns[returns < 0]
            if len(downside_returns) > 0:
                result.sortino_ratio = np.sqrt(252) * returns.mean() / downside_returns.std() if downside_returns.std() > 0 else 0

        # 最大回撤
        if result.max_drawdown <= 0 and result.equity_curve:
            equity_series = pd.Series([e[1] for e in result.equity_curve])
            running_max = equity_series.cummax()
            drawdown = (equity_series - running_max) / running_max * 100
            result.max_drawdown = abs(drawdown.min())

        if not trades_df.empty:
            best_trade = trades_df.loc[trades_df['pnl'].idxmax()]
            worst_trade = trades_df.loc[trades_df['pnl'].idxmin()]
            max_loss_streak = 0
            current_streak = 0
            for pnl in trades_df['pnl']:
                if pnl <= 0:
                    current_streak += 1
                    max_loss_streak = max(max_loss_streak, current_streak)
                else:
                    current_streak = 0
            result.max_consecutive_losses = max_loss_streak
            total_fees = sum(float(getattr(t, 'fees', 0.0) or 0.0) for t in result.trades)
            total_slippage = sum(float(getattr(t, 'slippage_cost', 0.0) or 0.0) for t in result.trades)
            total_funding = sum(float(getattr(t, 'funding_cost', 0.0) or 0.0) for t in result.trades)
            result.stats.update({
                'best_trade_pnl': float(best_trade['pnl']),
                'best_trade_pct': float(best_trade['pnl_percent']),
                'worst_trade_pnl': float(worst_trade['pnl']),
                'worst_trade_pct': float(worst_trade['pnl_percent']),
                'total_fees': total_fees,
                'total_slippage_cost': total_slippage,
                'total_funding_cost': total_funding,
            })

            if result.stats.get('backtest_mode') != 'state_machine':
                pilot_trades = [
                    t for t in result.trades
                    if t.exit_time and BacktestAnalyzer._contains_any(
                        getattr(t, 'entry_reason', ''),
                        ['1%试仓', '试仓', 'pilot']
                    )
                ]
                addon_trades = [
                    t for t in result.trades
                    if t.exit_time and BacktestAnalyzer._contains_any(
                        getattr(t, 'entry_reason', ''),
                        ['10%加仓', '加仓', 'add-on', 'addon']
                    )
                ]
                first_principle_exits = [
                    t for t in result.trades
                    if t.exit_time and BacktestAnalyzer._contains_any(
                        getattr(t, 'exit_reason', ''),
                        ['第一原则', '成本线', 'ATR止损线', '止损线']
                    )
                ]
                pilot_wins = [t for t in pilot_trades if float(getattr(t, 'pnl', 0.0) or 0.0) > 0]
                addon_wins = [t for t in addon_trades if float(getattr(t, 'pnl', 0.0) or 0.0) > 0]
                result.stats.update({
                    'pilot_trade_count': len(pilot_trades),
                    'pilot_win_count': len(pilot_wins),
                    'pilot_success_rate': (len(pilot_wins) / len(pilot_trades) * 100.0) if pilot_trades else 0.0,
                    'add_on_trade_count': len(addon_trades),
                    'add_on_win_count': len(addon_wins),
                    'add_on_success_rate': (len(addon_wins) / len(addon_trades) * 100.0) if addon_trades else 0.0,
                    'first_principle_trigger_count': len(first_principle_exits),
                })

            for days in (7, 30, 90):
                cutoff = result.end_date - timedelta(days=days)
                window_trades = [t for t in result.trades if t.exit_time and t.exit_time >= cutoff]
                window_pnl = sum(float(t.pnl) for t in window_trades)
                result.stats[f'return_{days}d'] = (window_pnl / result.initial_capital * 100) if result.initial_capital > 0 else 0.0
                result.stats[f'trades_{days}d'] = len(window_trades)
            # Calmar 比率
            if result.max_drawdown > 0:
                result.calmar_ratio = round(result.annual_return / result.max_drawdown, 4)

            # VaR-95%（5th 百分位单笔亏损率）
            if len(trades_df) >= 2:
                result.var_95 = round(float(np.percentile(trades_df['pnl_percent'], 5)), 4)

            # 恢复因子
            if result.max_drawdown > 0:
                result.recovery_factor = round(result.total_return / result.max_drawdown, 4)

            # 溃疡指数（equity curve 回撤的均方根）
            if result.equity_curve and len(result.equity_curve) >= 2:
                eq = np.array([e[1] for e in result.equity_curve], dtype=float)
                running_peak = np.maximum.accumulate(eq)
                dd_pct = np.where(running_peak > 0, (running_peak - eq) / running_peak * 100.0, 0.0)
                result.ulcer_index = round(float(np.sqrt(np.mean(dd_pct ** 2))), 4)

            result.live_readiness_score = BacktestAnalyzer.calculate_live_readiness_score(result)

        return result

    @staticmethod
    def calculate_live_readiness_score(result: BacktestResult) -> float:
        """按收益、回撤、胜率、盈亏比、夏普、卡玛、连亏和交易数量合成实盘准入评分"""
        trade_score = min(result.total_trades / 30.0, 1.0) * 100.0
        return_score = max(0.0, min(100.0, 50.0 + result.total_return * 2.0))
        drawdown_score = max(0.0, 100.0 - result.max_drawdown * 5.0)
        win_score = max(0.0, min(100.0, result.win_rate))
        pf_score = max(0.0, min(100.0, result.profit_factor * 35.0))
        sharpe_score = max(0.0, min(100.0, 50.0 + result.sharpe_ratio * 18.0))
        # 卡玛比率：0 = 0分，≥3 = 100分
        calmar_score = max(0.0, min(100.0, result.calmar_ratio / 3.0 * 100.0))
        loss_streak_score = max(0.0, 100.0 - result.max_consecutive_losses * 12.0)
        score = (
            return_score * 0.15
            + drawdown_score * 0.18
            + win_score * 0.12
            + pf_score * 0.14
            + sharpe_score * 0.12
            + calmar_score * 0.10
            + loss_streak_score * 0.11
            + trade_score * 0.08
        )
        return round(max(0.0, min(100.0, score)), 2)

    @staticmethod
    def generate_report(result: BacktestResult) -> str:
        """
        生成回测报告

        Args:
            result: 回测结果

        Returns:
            报告文本
        """
        data_sources = result.stats.get('data_sources', {})
        data_ranges = result.stats.get('data_ranges', {})
        bars_loaded = result.stats.get('bars_loaded', {})
        signal_counts = result.stats.get('signal_counts', {})

        data_lines = []
        for bar_name in result.stats.get('requested_bars', []):
            source = data_sources.get(bar_name, 'missing')
            count = bars_loaded.get(bar_name, 0)
            date_range = data_ranges.get(bar_name, {})
            range_text = f"{date_range.get('start', '-') } -> {date_range.get('end', '-')}"
            data_lines.append(f"{bar_name}: {count} 根 | 来源={source} | 区间={range_text}")
        if not data_lines:
            data_lines.append("未记录数据加载明细")

        signal_lines = [f"{action}: {count}" for action, count in sorted(signal_counts.items())]
        if not signal_lines:
            signal_lines.append("无信号记录")

        trade_lines = []
        for idx, trade in enumerate(result.trades, start=1):
            if not trade.exit_time:
                continue
            direction = "LONG" if trade.direction == TradeDirection.LONG else "SHORT"
            trade_lines.append(
                f"{idx}. {direction} | 开仓 {trade.entry_time.strftime('%Y-%m-%d %H:%M')} "
                f"市场价 {getattr(trade, 'raw_entry_price', trade.entry_price):.4f} / 成交价 {trade.entry_price:.4f} "
                f"| 平仓 {trade.exit_time.strftime('%Y-%m-%d %H:%M')} "
                f"触发价 {getattr(trade, 'raw_exit_price', trade.exit_price):.4f} / 成交价 {trade.exit_price:.4f} "
                f"| 数量 {trade.size:.6f} | 盈亏 {trade.pnl:.2f} USDT ({trade.pnl_percent:.2f}%) "
                f"| 入场原因: {trade.entry_reason or '-'} | 出场原因: {trade.exit_reason or '-'}"
            )
        if not trade_lines:
            trade_lines.append("本次回测无完整成交记录")

        report = f"""
========================================
           回测结果报告
========================================

策略名称：{result.strategy_name}
交易对：{result.inst_id}
回测区间：{result.start_date.strftime('%Y-%m-%d')} 至 {result.end_date.strftime('%Y-%m-%d')}

----------------------------------------
资金信息
----------------------------------------
初始资金：{result.initial_capital:,.2f} USDT
最终资金：{result.final_capital:,.2f} USDT
总盈亏：{result.total_pnl:,.2f} USDT
总收益率：{result.total_return:.2f}%
年化收益率：{result.annual_return:.2f}%

----------------------------------------
风险指标
----------------------------------------
最大回撤：{result.max_drawdown:.2f}%
夏普比率：{result.sharpe_ratio:.2f}
索提诺比率：{result.sortino_ratio:.2f}
卡玛比率：{result.calmar_ratio:.2f}
VaR-95%（单笔最坏）：{result.var_95:.2f}%
恢复因子：{result.recovery_factor:.2f}
溃疡指数：{result.ulcer_index:.2f}%
实盘准入评分：{result.live_readiness_score:.2f}
最大连续亏损：{result.max_consecutive_losses}

----------------------------------------
交易统计
----------------------------------------
总交易次数：{result.total_trades}
盈利交易：{result.winning_trades}
亏损交易：{result.losing_trades}
胜率：{result.win_rate:.2f}%
盈亏比：{result.profit_factor:.2f}
平均盈利：{result.avg_win:,.2f} USDT
平均亏损：{result.avg_loss:,.2f} USDT
平均持仓时间：{result.avg_trade_duration:.1f} 小时
最佳单笔：{result.stats.get('best_trade_pnl', 0.0):,.2f} USDT ({result.stats.get('best_trade_pct', 0.0):.2f}%)
最差单笔：{result.stats.get('worst_trade_pnl', 0.0):,.2f} USDT ({result.stats.get('worst_trade_pct', 0.0):.2f}%)
试仓次数：{result.stats.get('pilot_trade_count', 0)}
试仓成功率：{result.stats.get('pilot_success_rate', 0.0):.2f}%
二次加仓次数：{result.stats.get('add_on_trade_count', 0)}
二次加仓成功率：{result.stats.get('add_on_success_rate', 0.0):.2f}%
第一原则触发次数：{result.stats.get('first_principle_trigger_count', 0)}
手续费成本：{result.stats.get('total_fees', 0.0):,.2f} USDT
滑点/冲击成本：{result.stats.get('total_slippage_cost', 0.0):,.2f} USDT
资金费率成本：{result.stats.get('total_funding_cost', 0.0):,.2f} USDT

----------------------------------------
数据验证
----------------------------------------
驱动周期：{result.stats.get('driver_bar', '-')}
{chr(10).join(data_lines)}

----------------------------------------
信号统计
----------------------------------------
{chr(10).join(signal_lines)}

----------------------------------------
交易明细
----------------------------------------
{chr(10).join(trade_lines)}

========================================
"""
        return report


class Backtester:
    """回测器 - 整合引擎和分析器"""

    def __init__(self, okx_client=None, initial_capital: float = 10000.0):
        """
        初始化回测器

        Args:
            okx_client: OKX 客户端 (用于获取历史数据)
            initial_capital: 初始资金
        """
        self.okx_client = okx_client
        self.data_manager = None
        self.initial_capital = initial_capital
        self.engine = BacktestEngine(initial_capital)
        self.analyzer = BacktestAnalyzer()
        self._signal_window = 500
        self._historical_cache: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self._scanner_signal_cache: Dict[str, Dict[str, Any]] = {}

    _INDICATOR_COLUMNS = (
        'ema_8', 'ema_12', 'ema_13', 'ema_20', 'ema_21', 'ema_26', 'ema_60',
        'rsi_14', 'atr_14', 'macd', 'macd_signal', 'macd_hist', 'vol_sma_20',
    )

    def _get_cached_rows(self, inst_id: str, bar: str, start_ms: int, end_ms: int) -> Optional[List[List[str]]]:
        cache = self._historical_cache.get((inst_id, bar))
        if not cache:
            return None
        if cache.get('start_ms', 0) > start_ms or cache.get('end_ms', 0) < end_ms:
            return None
        rows = cache.get('rows') or []
        if not rows:
            return None
        return [row for row in rows if start_ms <= int(row[0]) <= end_ms]

    def _store_cached_rows(self, inst_id: str, bar: str, start_ms: int, end_ms: int, rows: List[List[str]]) -> None:
        if not rows:
            return
        key = (inst_id, bar)
        existing = self._historical_cache.get(key)
        if existing:
            if existing.get('start_ms', 0) <= start_ms and existing.get('end_ms', 0) >= end_ms:
                return
            if start_ms <= existing.get('start_ms', 0) and end_ms >= existing.get('end_ms', 0):
                self._historical_cache[key] = {
                    'start_ms': start_ms,
                    'end_ms': end_ms,
                    'rows': rows,
                }
                return
        self._historical_cache[key] = {
            'start_ms': start_ms,
            'end_ms': end_ms,
            'rows': rows,
        }

    def run_backtest(self, strategy, inst_id: str, start_date: str, end_date: str,
                     bar: str = "1H", config: Dict = None,
                     progress_callback: Optional[Callable[[int, str], None]] = None,
                     should_stop: Optional[Callable[[], bool]] = None,
                     should_pause: Optional[Callable[[], bool]] = None) -> BacktestResult:
        """
        运行回测

        Args:
            strategy: 策略实例
            inst_id: 交易对 ID
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            bar: K 线周期 (1m, 5m, 15m, 1H, 4H, 1D)
            config: 策略配置

        Returns:
            回测结果
        """
        self.engine.reset()
        self._scanner_signal_cache = {}
        config = dict(config or {})
        config.setdefault('inst_id', inst_id)
        state_machine_mode = self._should_use_state_machine_backtest(strategy, config)
        self.engine.set_cost_config({
            'fee_pct': config.get('fee_pct', 0.05),
            'slippage_pct': config.get('slippage_pct', 0.03),
            'funding_rate_8h_pct': config.get('funding_rate_8h_pct', 0.0),
            'market_impact_pct': config.get('market_impact_pct', 0.02),
        })
        if hasattr(strategy, 'reset_backtest_state'):
            strategy.reset_backtest_state()
        if progress_callback:
            progress_callback(5, "解析回测周期")
        requested_start = datetime.strptime(start_date, "%Y-%m-%d")
        requested_end = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)

        # 获取历史数据
        bars = self._resolve_bars(bar, strategy)
        if state_machine_mode:
            bars = self._augment_state_machine_bars(bars)
        klines_map, fetch_meta = self._fetch_klines_map(
            inst_id, bars, start_date, end_date,
            progress_callback=progress_callback,
            should_stop=should_stop
        )
        if not klines_map:
            return BacktestResult(
                strategy_name=strategy.__class__.__name__,
                inst_id=inst_id,
                start_date=datetime.now(),
                end_date=datetime.now(),
                initial_capital=self.initial_capital
            )

        driver_bar = self._resolve_driver_bar(bar, bars, strategy, config)
        if state_machine_mode and '3m' in klines_map:
            driver_bar = '3m'
        driver_klines = klines_map.get(driver_bar, [])
        if not driver_klines:
            driver_bar = next(iter(klines_map.keys()))
            driver_klines = klines_map.get(driver_bar, [])
        if not driver_klines:
            return BacktestResult(
                strategy_name=strategy.__class__.__name__,
                inst_id=inst_id,
                start_date=datetime.now(),
                end_date=datetime.now(),
                initial_capital=self.initial_capital
            )

        # 解析 K 线数据
        df_map = {bar_name: self._parse_klines(bar_klines) for bar_name, bar_klines in klines_map.items() if bar_klines}
        df = df_map.get(driver_bar)
        if df is None or df.empty:
            return BacktestResult(
                strategy_name=strategy.__class__.__name__,
                inst_id=inst_id,
                start_date=datetime.now(),
                end_date=datetime.now(),
                initial_capital=self.initial_capital
            )

        # 初始化策略
        if config:
            strategy.config = config
        if progress_callback:
            progress_callback(20, f"已载入 {len(df)} 根 {driver_bar} K线（回测驱动周期）")

        signal_context = self._build_signal_context(df_map, driver_bar)
        driver_timestamps = signal_context['driver_timestamps']
        driver_timestamps_ns = signal_context['driver_timestamps_ns']
        driver_closes = signal_context['driver_closes']
        driver_highs = signal_context['driver_highs']
        driver_lows = signal_context['driver_lows']

        if state_machine_mode:
            return self._run_state_machine_backtest(
                strategy=strategy,
                inst_id=inst_id,
                requested_start=requested_start,
                requested_end=requested_end,
                df=df,
                df_map=df_map,
                driver_bar=driver_bar,
                signal_context=signal_context,
                driver_timestamps=driver_timestamps,
                driver_timestamps_ns=driver_timestamps_ns,
                driver_closes=driver_closes,
                fetch_meta=fetch_meta,
                bars=bars,
                config=config,
                progress_callback=progress_callback,
                should_stop=should_stop,
                should_pause=should_pause,
            )

        # 回测主循环
        result = BacktestResult(
            strategy_name=strategy.__class__.__name__,
            inst_id=inst_id,
            start_date=max(df['timestamp'].iloc[0], requested_start),
            end_date=min(df['timestamp'].iloc[-1], requested_end),
            initial_capital=self.initial_capital
        )
        result.stats.update({
            'data_sources': fetch_meta.get('sources', {}),
            'data_ranges': fetch_meta.get('ranges', {}),
            'bars_loaded': fetch_meta.get('counts', {}),
            'requested_bars': bars,
            'requested_driver_bar': bar,
            'driver_bar': driver_bar,
        })
        signal_counts: Dict[str, int] = {}

        start_index = df['timestamp'].searchsorted(requested_start, side='left')
        if start_index >= len(df) - 1:
            result.final_capital = self.initial_capital
            return result

        loop_total = max((len(df) - 1) - start_index, 1)
        self.engine.configure_equity_sampling(loop_total)

        for i in range(start_index, len(df) - 1):
            if should_stop and should_stop():
                break
            while should_pause and should_pause():
                time.sleep(0.2)
                if should_stop and should_stop():
                    break
            if should_stop and should_stop():
                break

            timestamp = driver_timestamps[i]
            current_ts_ns = driver_timestamps_ns[i]
            current_price = driver_closes[i]
            row = {
                'high': driver_highs[i],
                'low': driver_lows[i],
                'close': current_price,
            }

            # 更新权益曲线
            self.engine.update_equity(timestamp, current_price)

            # 资金熔断：回撤超过阈值时强制平仓并终止回测
            _dd_stop = float(config.get('max_drawdown_stop_pct', 0.0) or 0.0)
            if _dd_stop > 0 and self.engine._max_drawdown_pct >= _dd_stop:
                if self.engine.position > 0:
                    if self.engine.position_direction == TradeDirection.LONG:
                        self.engine.execute_sell(timestamp, current_price, exit_reason="资金熔断强制平仓")
                    else:
                        self.engine.execute_cover(timestamp, current_price, exit_reason="资金熔断强制平仓")
                signal_counts['CIRCUIT_BREAKER'] = signal_counts.get('CIRCUIT_BREAKER', 0) + 1
                break

            # 扫描类策略通常只负责入场，回测端统一补上止盈/止损退出。
            risk_exit = self._check_risk_exit(row, config)
            if risk_exit:
                action = risk_exit.get('action', '')
                reason = risk_exit.get('reason', '')
                exit_price = risk_exit.get('price', current_price)
                signal_counts[action] = signal_counts.get(action, 0) + 1
                if action == 'EXIT_LONG' and self.engine.position_direction == TradeDirection.LONG:
                    self.engine.execute_sell(timestamp, exit_price, exit_reason=reason)
                    continue
                if action == 'EXIT_SHORT' and self.engine.position_direction == TradeDirection.SHORT:
                    self.engine.execute_cover(timestamp, exit_price, exit_reason=reason)
                    continue

            # 生成交易信号
            signal = self._generate_signal(
                strategy,
                df_map,
                driver_bar,
                i,
                inst_id=inst_id,
                signal_context=signal_context,
                current_ts_ns=current_ts_ns,
                current_ts=timestamp,
                current_price=current_price,
                config=config,
            )

            if signal:
                action = signal.get('action', '')
                position_size = signal.get('position_size', 0.1)
                entry_price = signal.get('entry_price', current_price)
                signal_reason = signal.get('reason', '')
                limit_miss_prob = float(config.get('limit_miss_probability_pct', 0.0) or 0.0) / 100.0
                if action in ['BUY', 'BUY_LONG', 'SHORT', 'SELL_SHORT'] and limit_miss_prob > 0:
                    import hashlib as _hashlib
                    _h = int(_hashlib.md5(str(int(timestamp.timestamp() * 1000)).encode()).hexdigest()[:8], 16)
                    deterministic_bucket = (_h % 10000) / 10000.0
                    if deterministic_bucket < limit_miss_prob:
                        signal_counts['LIMIT_MISSED'] = signal_counts.get('LIMIT_MISSED', 0) + 1
                        continue
                if action:
                    signal_counts[action] = signal_counts.get(action, 0) + 1

                if action in ['BUY', 'BUY_LONG'] and self.engine.position == 0:
                    self.engine.execute_buy(timestamp, entry_price, None, position_size, entry_reason=signal_reason)

                elif action in ['SELL', 'EXIT_LONG'] and self.engine.position > 0:
                    self.engine.execute_sell(timestamp, current_price, exit_reason=signal_reason or action)

                elif action in ['SHORT', 'SELL_SHORT'] and self.engine.position == 0:
                    self.engine.execute_short(timestamp, entry_price, None, position_size, entry_reason=signal_reason)

                elif action in ['COVER', 'EXIT_SHORT'] and self.engine.position > 0:
                    self.engine.execute_cover(timestamp, current_price, exit_reason=signal_reason or action)

                elif action == 'STOP_LOSS':
                    if self.engine.position_direction == TradeDirection.LONG:
                        self.engine.execute_sell(timestamp, current_price, exit_reason=signal_reason or "止损")
                    else:
                        self.engine.execute_cover(timestamp, current_price, exit_reason=signal_reason or "止损")

            processed = i - start_index + 1
            if progress_callback and (processed == 1 or processed % max(loop_total // 20, 1) == 0):
                pct = 20 + int(75 * processed / loop_total)
                progress_callback(pct, f"回放中 {processed}/{loop_total}")

        # 结束时平仓
        if len(df) > 0:
            final_price = df['close'].iloc[-1]
            final_time = df['timestamp'].iloc[-1]
            if self.engine.position > 0:
                if self.engine.position_direction == TradeDirection.LONG:
                    self.engine.execute_sell(final_time, final_price, exit_reason="回测结束强制平仓")
                else:
                    self.engine.execute_cover(final_time, final_price, exit_reason="回测结束强制平仓")
            self.engine.update_equity(final_time, final_price, force_record=True)

        # 填充结果
        result.trades = self.engine.trades.copy()
        result.equity_curve = self.engine.equity_curve.copy()
        result.total_trades = len(result.trades)
        result.final_capital = self.engine.get_current_equity(df['close'].iloc[-1])
        result.max_drawdown = self.engine._max_drawdown_pct
        result.stats['signal_counts'] = signal_counts
        result.stats['equity_sample_every'] = self.engine._equity_sample_every
        result.stats['cost_model'] = {
            'fee_pct': config.get('fee_pct', 0.05),
            'slippage_pct': config.get('slippage_pct', 0.03),
            'funding_rate_8h_pct': config.get('funding_rate_8h_pct', 0.0),
            'limit_miss_probability_pct': config.get('limit_miss_probability_pct', 0.0),
            'market_impact_pct': config.get('market_impact_pct', 0.02),
        }

        # 分析结果
        result = self.analyzer.analyze(result)

        return result

    def _should_use_state_machine_backtest(self, strategy, config: Dict) -> bool:
        mode = str(config.get('backtest_mode', '') or '').strip().lower()
        if mode == 'state_machine':
            return True
        preferred = str(getattr(strategy, 'preferred_backtest_mode', '') or '').strip().lower()
        return preferred == 'state_machine'

    def _augment_state_machine_bars(self, bars: List[str]) -> List[str]:
        ordered = list(dict.fromkeys(list(bars) + ['3m', '15m', '1H', '4H', '1D']))
        return ordered

    def _build_klines_window(self, signal_context: Dict[str, Any], current_ts_ns: int, window: int = 500) -> Dict[str, List]:
        klines_map: Dict[str, List] = {}
        context_bars = (signal_context or {}).get('bars', {})
        for bar_name, bar_cache in context_bars.items():
            timestamps_ns = bar_cache['timestamps_ns']
            cursor = int(bar_cache.get('cursor', 0))
            total = len(timestamps_ns)
            while cursor < total and timestamps_ns[cursor] <= current_ts_ns:
                cursor += 1
            bar_cache['cursor'] = cursor
            if cursor <= 0:
                continue
            start = max(0, cursor - window)
            klines_map[bar_name] = bar_cache['rows'][start:cursor]
        return klines_map

    def _state_machine_window_size(self, bar_name: str, runner: "_StateMachineBacktestRunner") -> int:
        if runner._campaign is None and runner._pending_signal is None:
            return self._signal_window
        compact = {
            '3m': 96,
            '15m': 64,
            '1H': 96,
            '4H': 72,
            '1D': 40,
        }
        return compact.get(bar_name, min(self._signal_window, 96))

    def _build_state_machine_klines_window(
        self,
        signal_context: Dict[str, Any],
        current_ts_ns: int,
        runner: "_StateMachineBacktestRunner",
    ) -> Dict[str, List]:
        klines_map: Dict[str, List] = {}
        context_bars = (signal_context or {}).get('bars', {})
        for bar_name, bar_cache in context_bars.items():
            timestamps_ns = bar_cache['timestamps_ns']
            cursor = int(bar_cache.get('cursor', 0))
            total = len(timestamps_ns)
            while cursor < total and timestamps_ns[cursor] <= current_ts_ns:
                cursor += 1
            bar_cache['cursor'] = cursor
            if cursor <= 0:
                continue
            window = self._state_machine_window_size(bar_name, runner)
            start = max(0, cursor - window)
            klines_map[bar_name] = bar_cache['rows'][start:cursor]
        return klines_map

    def _should_generate_state_machine_signal(self, strategy, runner: "_StateMachineBacktestRunner", config: Dict[str, Any]) -> bool:
        if getattr(strategy, 'requires_continuous_signals', False):
            return True
        if not bool(config.get('state_machine_stage_aware_signal_gating', True)):
            return True
        return runner._campaign is None and runner._pending_signal is None

    def _run_state_machine_backtest(
        self,
        strategy,
        inst_id: str,
        requested_start: datetime,
        requested_end: datetime,
        df: pd.DataFrame,
        df_map: Dict[str, pd.DataFrame],
        driver_bar: str,
        signal_context: Dict[str, Any],
        driver_timestamps: List[datetime],
        driver_timestamps_ns: np.ndarray,
        driver_closes: np.ndarray,
        fetch_meta: Dict[str, Dict[str, Any]],
        bars: List[str],
        config: Dict,
        progress_callback: Optional[Callable[[int, str], None]] = None,
        should_stop: Optional[Callable[[], bool]] = None,
        should_pause: Optional[Callable[[], bool]] = None,
    ) -> BacktestResult:
        okx_view = _BacktestOkxView()
        executor = _BacktestTradeExecutorAdapter(self.engine, inst_id)
        runner_config = dict(config)
        runner_config['_system_name'] = f"BacktestStateMachine:{inst_id}"
        runner = _StateMachineBacktestRunner(strategy, inst_id, okx_view, executor, runner_config)

        result = BacktestResult(
            strategy_name=strategy.__class__.__name__,
            inst_id=inst_id,
            start_date=max(df['timestamp'].iloc[0], requested_start),
            end_date=min(df['timestamp'].iloc[-1], requested_end),
            initial_capital=self.initial_capital
        )
        result.stats.update({
            'data_sources': fetch_meta.get('sources', {}),
            'data_ranges': fetch_meta.get('ranges', {}),
            'bars_loaded': fetch_meta.get('counts', {}),
            'requested_bars': bars,
            'requested_driver_bar': driver_bar,
            'driver_bar': driver_bar,
            'backtest_mode': 'state_machine',
        })
        signal_counts: Dict[str, int] = {}

        start_index = df['timestamp'].searchsorted(requested_start, side='left')
        if start_index >= len(df) - 1:
            result.final_capital = self.initial_capital
            return result

        loop_total = max((len(df) - 1) - start_index, 1)
        self.engine.configure_equity_sampling(loop_total)

        for i in range(start_index, len(df) - 1):
            if should_stop and should_stop():
                break
            while should_pause and should_pause():
                time.sleep(0.2)
                if should_stop and should_stop():
                    break
            if should_stop and should_stop():
                break

            timestamp = driver_timestamps[i]
            current_ts_ns = driver_timestamps_ns[i]
            current_price = float(driver_closes[i])
            okx_view.set_price(current_price)
            executor.set_market(timestamp, current_price, leverage=int(config.get('leverage', 1) or 1))
            self.engine.update_equity(timestamp, current_price)

            klines_window = self._build_state_machine_klines_window(signal_context, current_ts_ns, runner)
            raw_signal = None
            if self._should_generate_state_machine_signal(strategy, runner, config):
                try:
                    # 状态机模式：入场信号由策略负责，出场由状态机内部处理
                    raw_signal = (strategy.generate_signal(klines_window, skip_exit_checks=True)
                                  if hasattr(strategy, 'generate_signal') else None)
                except Exception:
                    raw_signal = None
            action = str((raw_signal or {}).get('action', '') or '')
            if action:
                signal_counts[action] = signal_counts.get(action, 0) + 1

            runner.step(timestamp, raw_signal or {}, {
                'daily': klines_window.get('1D', []),
                '4h': klines_window.get('4H', []),
                'hourly': klines_window.get('1H', []),
                'm15': klines_window.get('15m', []),
                'm3': klines_window.get('3m', []),
            })

            processed = i - start_index + 1
            if progress_callback and (processed == 1 or processed % max(loop_total // 20, 1) == 0 or processed == loop_total):
                pct = 20 + int(75 * processed / loop_total)
                progress_callback(pct, f"状态机回放中 {processed}/{loop_total}")

        if progress_callback:
            progress_callback(96, f"回放完成 {loop_total}/{loop_total}，正在收尾平仓...")

        if len(df) > 0:
            final_price = float(df['close'].iloc[-1])
            final_time = df['timestamp'].iloc[-1]
            okx_view.set_price(final_price)
            executor.set_market(final_time, final_price, leverage=int(config.get('leverage', 1) or 1))
            if self.engine.position > 0:
                runner.set_clock(final_time)
                runner._close_position("回测结束强制平仓")
            self.engine.update_equity(final_time, final_price, force_record=True)

        if progress_callback:
            progress_callback(97, "正在统计回测结果...")

        result.trades = self.engine.trades.copy()
        result.equity_curve = self.engine.equity_curve.copy()
        result.total_trades = len(result.trades)
        result.final_capital = self.engine.get_current_equity(float(df['close'].iloc[-1]))
        result.max_drawdown = self.engine._max_drawdown_pct
        result.stats['signal_counts'] = signal_counts
        result.stats['equity_sample_every'] = self.engine._equity_sample_every
        result.stats['cost_model'] = {
            'fee_pct': config.get('fee_pct', 0.05),
            'slippage_pct': config.get('slippage_pct', 0.03),
            'funding_rate_8h_pct': config.get('funding_rate_8h_pct', 0.0),
            'limit_miss_probability_pct': config.get('limit_miss_probability_pct', 0.0),
            'market_impact_pct': config.get('market_impact_pct', 0.02),
        }
        result.stats.update(runner.metrics)

        if progress_callback:
            progress_callback(98, "正在分析回测指标...")

        result = self.analyzer.analyze(result)

        if progress_callback:
            progress_callback(99, "回测完成，正在渲染结果...")

        return result

    def _get_warmup_delta(self, bar: str) -> timedelta:
        warmup_map = {
            '1m': timedelta(days=2),
            '3m': timedelta(days=4),
            '5m': timedelta(days=6),
            '15m': timedelta(days=12),
            '30m': timedelta(days=20),
            '1H': timedelta(days=30),
            '2H': timedelta(days=45),
            '4H': timedelta(days=90),
            '1D': timedelta(days=240),
            '1W': timedelta(days=720),
        }
        return warmup_map.get(bar, timedelta(days=30))

    def _resolve_bars(self, bar: str, strategy) -> List[str]:
        """解析回测所需周期列表"""
        if "全周期" in str(bar):
            return ["1m", "3m", "5m", "15m", "30m", "1H", "2H", "4H", "1D"]
        if hasattr(strategy, 'required_bars') and bar in getattr(strategy, 'required_bars', []):
            return list(dict.fromkeys(getattr(strategy, 'required_bars')))
        return [bar]

    def _resolve_driver_bar(self, requested_bar: str, bars: List[str], strategy, config: Optional[Dict[str, Any]] = None) -> str:
        """选择回测驱动周期。

        默认优先尊重用户在回测界面选择的周期，避免多周期扫描策略因为包含低周期数据，
        被隐式降级成超高频回放，导致回测速度远慢于用户预期。
        """
        bar_rank = {'1m': 1, '3m': 3, '5m': 5, '15m': 15, '30m': 30, '1H': 60, '2H': 120, '4H': 240, '1D': 1440, '1W': 10080}
        required = getattr(strategy, 'required_bars', []) if hasattr(strategy, 'required_bars') else []
        candidates = [bar for bar in required if bar in bars] or bars
        prefer_requested = bool((config or {}).get('respect_selected_bar_as_driver', True))
        if (
            prefer_requested
            and requested_bar
            and "全周期" not in str(requested_bar)
            and requested_bar in candidates
        ):
            return requested_bar
        return sorted(candidates, key=lambda item: bar_rank.get(item, 999999))[0]

    def _fetch_klines_map(self, inst_id: str, bars: List[str], start_date: str,
                          end_date: str,
                          progress_callback: Optional[Callable[[int, str], None]] = None,
                          should_stop: Optional[Callable[[], bool]] = None) -> Tuple[Dict[str, List], Dict[str, Dict[str, Any]]]:
        """获取一个或多个周期的 K 线数据"""
        results: Dict[str, List] = {}
        meta = {'sources': {}, 'ranges': {}, 'counts': {}}
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)
        end_ms = int(end_dt.timestamp() * 1000)

        try:
            for idx, bar in enumerate(bars):
                if should_stop and should_stop():
                    break
                fetch_start_dt = start_dt - self._get_warmup_delta(bar)
                start_ms = int(fetch_start_dt.timestamp() * 1000)
                cached_rows = self._get_cached_rows(inst_id, bar, start_ms, end_ms)
                if cached_rows:
                    results[bar] = cached_rows
                    meta['sources'][bar] = 'memory-cache'
                    meta['counts'][bar] = len(cached_rows)
                    meta['ranges'][bar] = {
                        'start': datetime.fromtimestamp(int(cached_rows[0][0]) / 1000).strftime('%Y-%m-%d %H:%M'),
                        'end': datetime.fromtimestamp(int(cached_rows[-1][0]) / 1000).strftime('%Y-%m-%d %H:%M'),
                    }
                    if progress_callback:
                        pct = 5 + int(15 * (idx + 1) / max(len(bars), 1))
                        progress_callback(pct, f"命中内存缓存 {bar} 数据 ({len(cached_rows)} 根)")
                    continue

                local_rows = self._load_local_klines(inst_id, bar, start_ms, end_ms)
                if local_rows:
                    results[bar] = local_rows
                    self._store_cached_rows(inst_id, bar, start_ms, end_ms, local_rows)
                    meta['sources'][bar] = 'database'
                    meta['counts'][bar] = len(local_rows)
                    meta['ranges'][bar] = {
                        'start': datetime.fromtimestamp(int(local_rows[0][0]) / 1000).strftime('%Y-%m-%d %H:%M'),
                        'end': datetime.fromtimestamp(int(local_rows[-1][0]) / 1000).strftime('%Y-%m-%d %H:%M'),
                    }
                    if progress_callback:
                        pct = 5 + int(15 * (idx + 1) / max(len(bars), 1))
                        progress_callback(pct, f"读取本地 {bar} 数据完成 ({len(local_rows)} 根)")
                    continue

                if not self.okx_client:
                    continue

                all_rows: List[List[Any]] = []
                seen_ts = set()
                cursor = None

                # 按日期范围动态计算最大翻页数，不再硬限 4000 根
                _bar_ms_map = {
                    "1m": 60_000, "3m": 180_000, "5m": 300_000,
                    "15m": 900_000, "30m": 1_800_000,
                    "1H": 3_600_000, "2H": 7_200_000, "4H": 14_400_000,
                    "6H": 21_600_000, "12H": 43_200_000,
                    "1D": 86_400_000, "1W": 604_800_000,
                }
                bar_ms = _bar_ms_map.get(bar, 3_600_000)
                range_ms = max(end_ms - start_ms, bar_ms)
                needed_bars = range_ms // bar_ms + 300  # +300 预热缓冲
                max_pages = min(500, int(needed_bars // 100) + 5)  # 单周期最多 50000 根

                for _ in range(max_pages):
                    if should_stop and should_stop():
                        break
                    result = self.okx_client.get_history_kline(inst_id, bar=bar, limit=100, after=cursor)
                    if not result or result.get('code') != '0' or not result.get('data'):
                        break

                    batch = result.get('data', [])
                    added = 0
                    oldest_ts = None
                    for row in batch:
                        try:
                            ts = int(row[0])
                        except Exception:
                            continue
                        oldest_ts = ts if oldest_ts is None else min(oldest_ts, ts)
                        if ts in seen_ts:
                            continue
                        seen_ts.add(ts)
                        if start_ms <= ts <= end_ms:
                            all_rows.append(row)
                            added += 1

                    if oldest_ts is None or oldest_ts <= start_ms:
                        break
                    cursor = str(oldest_ts)
                    if added == 0 and oldest_ts > end_ms:
                        continue

                if all_rows:
                    all_rows.sort(key=lambda x: int(x[0]))
                    results[bar] = all_rows
                    self._store_cached_rows(inst_id, bar, start_ms, end_ms, all_rows)
                    meta['sources'][bar] = 'okx'
                    meta['counts'][bar] = len(all_rows)
                    meta['ranges'][bar] = {
                        'start': datetime.fromtimestamp(int(all_rows[0][0]) / 1000).strftime('%Y-%m-%d %H:%M'),
                        'end': datetime.fromtimestamp(int(all_rows[-1][0]) / 1000).strftime('%Y-%m-%d %H:%M'),
                    }

                if progress_callback:
                    pct = 5 + int(15 * (idx + 1) / max(len(bars), 1))
                    source = meta['sources'].get(bar, 'missing')
                    progress_callback(pct, f"获取 {bar} 历史K线完成 ({len(results.get(bar, []))} 根, {source})")
        except Exception as e:
            print(f"获取 K 线失败：{e}")

        return results, meta

    def _load_local_klines(self, inst_id: str, bar: str, start_ms: int, end_ms: int) -> List[List[str]]:
        """优先从本地数据库加载 K 线数据"""
        if not self.data_manager:
            return []

        try:
            df = self.data_manager.load_klines(inst_id, bar, start_ts=start_ms, end_ts=end_ms)
        except Exception:
            return []

        if df is None or df.empty:
            return []

        df = df.copy()
        column_map = {
            'o': 'open',
            'h': 'high',
            'l': 'low',
            'c': 'close',
            'v': 'volume',
            'vol': 'volume',
        }
        df = df.rename(columns=column_map)
        required_cols = ['ts', 'open', 'high', 'low', 'close', 'volume']
        if any(col not in df.columns for col in required_cols):
            return []

        for col in required_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna(subset=required_cols).sort_values('ts').drop_duplicates(subset='ts', keep='last')
        if df.empty:
            return []

        ts_values = df['ts'].astype('int64').astype(str).tolist()
        open_values = df['open'].astype(str).tolist()
        high_values = df['high'].astype(str).tolist()
        low_values = df['low'].astype(str).tolist()
        close_values = df['close'].astype(str).tolist()
        volume_values = df['volume'].astype(str).tolist()
        return [
            [ts_values[idx], open_values[idx], high_values[idx], low_values[idx], close_values[idx], volume_values[idx], '', '', '']
            for idx in range(len(df))
        ]

    def _parse_klines(self, klines: List) -> pd.DataFrame:
        """解析 K 线数据"""
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'volCcy', 'volCcyQuote', 'confirm'
        ])

        # 转换时间戳
        df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')

        # 转换价格数据
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)

        df = df.sort_values('timestamp').drop_duplicates(subset='timestamp', keep='last').reset_index(drop=True)
        return self._precompute_common_indicators(df)

    def _precompute_common_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """预计算多策略常用指标列，减少回测主循环内重复计算。"""
        if df is None or df.empty:
            return df
        out = df.copy()
        close = out['close']
        high = out['high']
        low = out['low']
        volume = out['volume']

        for span in (8, 12, 13, 20, 21, 26, 60):
            out[f'ema_{span}'] = close.ewm(span=span, adjust=False).mean()

        delta = close.diff()
        gain = delta.clip(lower=0.0)
        loss = (-delta.clip(upper=0.0))
        avg_gain = gain.rolling(14, min_periods=14).mean()
        avg_loss = loss.rolling(14, min_periods=14).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        out['rsi_14'] = (100.0 - (100.0 / (1.0 + rs))).bfill().fillna(50.0)

        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        out['atr_14'] = tr.rolling(14, min_periods=1).mean().bfill()

        ema12 = out['ema_12']
        ema26 = out['ema_26']
        out['macd'] = ema12 - ema26
        out['macd_signal'] = out['macd'].ewm(span=9, adjust=False).mean()
        out['macd_hist'] = out['macd'] - out['macd_signal']
        out['vol_sma_20'] = volume.rolling(20, min_periods=1).mean()
        return out

    def _df_to_okx_rows(self, df: pd.DataFrame) -> List[List[str]]:
        """将 DataFrame 转为策略兼容的 OKX K 线列表格式"""
        if df is None or df.empty:
            return []
        ts_values = (df['timestamp'].astype('int64') // 10**6).astype('int64').tolist()
        open_values = df['open'].astype(str).tolist()
        high_values = df['high'].astype(str).tolist()
        low_values = df['low'].astype(str).tolist()
        close_values = df['close'].astype(str).tolist()
        volume_values = df['volume'].astype(str).tolist()
        return [
            [ts_values[idx], open_values[idx], high_values[idx], low_values[idx], close_values[idx], volume_values[idx], '', '', '']
            for idx in range(len(ts_values))
        ]

    def _build_signal_context(self, df_map: Dict[str, pd.DataFrame], driver_bar: str) -> Dict[str, Any]:
        driver_df = df_map.get(driver_bar)
        if driver_df is None or driver_df.empty:
            return {}

        bars: Dict[str, Dict[str, Any]] = {}
        for bar_name, bar_df in df_map.items():
            if bar_df is None or bar_df.empty:
                continue
            bars[bar_name] = {
                'df': bar_df,
                'timestamps_ns': bar_df['timestamp'].astype('int64').to_numpy(),
                'rows': self._df_to_okx_rows(bar_df),
                'cursor': 0,
                'indicator_cols': {col: bar_df[col].to_numpy(dtype=float) for col in self._INDICATOR_COLUMNS if col in bar_df.columns},
            }

        return {
            'driver_df': driver_df,
            'driver_timestamps': driver_df['timestamp'].tolist(),
            'driver_timestamps_ns': driver_df['timestamp'].astype('int64').to_numpy(),
            'driver_closes': driver_df['close'].to_numpy(dtype=float),
            'driver_highs': driver_df['high'].to_numpy(dtype=float),
            'driver_lows': driver_df['low'].to_numpy(dtype=float),
            'bars': bars,
        }

    def _quote_volume_24h(self, df: pd.DataFrame, current_ts: datetime) -> float:
        """用历史 K 线估算近 24H 成交额，供扫描策略做流动性过滤。"""
        if df is None or df.empty:
            return 0.0
        start_ts = current_ts - timedelta(hours=24)
        recent = df[(df['timestamp'] >= start_ts) & (df['timestamp'] <= current_ts)]
        if recent.empty:
            recent = df[df['timestamp'] <= current_ts].tail(24)
        return float((recent['volume'] * recent['close']).sum()) if not recent.empty else 0.0

    def _build_scanner_symbol_from_history(
        self,
        inst_id: str,
        base_history: pd.DataFrame,
        current_price: float,
        klines_map: Dict[str, List],
        indicator_map: Optional[Dict[str, Dict[str, List[float]]]] = None,
    ) -> ScannerSymbol:
        """把回测历史切片转换成扫描器需要的 ScannerSymbol。"""
        history = base_history.tail(48) if base_history is not None and not base_history.empty else pd.DataFrame()
        high_24h = float(history['high'].max()) if not history.empty else float(current_price)
        low_24h = float(history['low'].min()) if not history.empty else float(current_price)
        open_24h = float(history['open'].iloc[0]) if not history.empty else float(current_price)
        price_change_24h = ((current_price - open_24h) / open_24h * 100.0) if open_24h > 0 else 0.0
        volume_24h = float((history['volume'] * history['close']).sum()) if not history.empty else 0.0
        return ScannerSymbol(
            inst_id=inst_id,
            last_price=float(current_price),
            volume_24h=volume_24h,
            price_change_24h=price_change_24h,
            high_24h=high_24h,
            low_24h=low_24h,
            open_interest=0.0,
            extra_data={'klines': klines_map, 'indicator_map': indicator_map or {}},
        )

    def _build_indicator_window(self, signal_context: Dict[str, Any], window: int = 500) -> Dict[str, Dict[str, List[float]]]:
        indicator_map: Dict[str, Dict[str, List[float]]] = {}
        context_bars = (signal_context or {}).get('bars', {})
        for bar_name, bar_cache in context_bars.items():
            cursor = int(bar_cache.get('cursor', 0))
            if cursor <= 0:
                continue
            start = max(0, cursor - window)
            cols = bar_cache.get('indicator_cols') or {}
            if not cols:
                continue
            indicator_map[bar_name] = {
                col: values[start:cursor].tolist()
                for col, values in cols.items()
            }
        return indicator_map

    def _scanner_result_to_signal(self, result: Dict, current_price: float, config: Dict) -> Optional[Dict]:
        """将扫描策略结果转换为回测引擎可执行的交易信号。"""
        if not result or not result.get('passed'):
            return None
        direction = str(result.get('direction', 'WAIT') or 'WAIT').upper()
        if direction in {'BUY', 'LONG'}:
            action = 'BUY'
        elif direction in {'SELL', 'SHORT'}:
            action = 'SHORT'
        else:
            return None
        score = float(result.get('opportunity_score', result.get('score', 0.0)) or 0.0)
        min_score = float(config.get('backtest_min_score', config.get('min_score', 0.0)) or 0.0)
        if min_score > 0 and score < min_score:
            return None
        signals = result.get('signals') or []
        reason_parts = [
            str(result.get('category') or result.get('strategy_category') or '扫描策略'),
            f"评分 {score:.1f}",
        ]
        if signals:
            reason_parts.append(str(signals[0]))
        return {
            'action': action,
            'position_size': float(config.get('position_size', config.get('backtest_position_size', 0.1)) or 0.1),
            'entry_price': current_price,
            'reason': " | ".join(reason_parts),
            'score': score,
            'raw_result': result,
        }

    def _config_pct(self, config: Dict, keys: List[str], default: float) -> float:
        for key in keys:
            if key in config:
                try:
                    return float(config.get(key) or 0.0)
                except (TypeError, ValueError):
                    return default
        return default

    def _check_risk_exit(self, row: pd.Series, config: Dict) -> Optional[Dict]:
        """按固定止盈/止损模拟扫描策略持仓退出。"""
        if self.engine.position <= 0 or self.engine.position_direction is None:
            return None
        tp_pct = self._config_pct(config, ['take_profit_pct', 'tp_percent', 'take_profit'], 5.0) / 100.0
        sl_pct = self._config_pct(config, ['stop_loss_pct', 'sl_percent', 'stop_loss'], 3.0) / 100.0
        if tp_pct <= 0 and sl_pct <= 0:
            return None
        high = float(row.get('high', row.get('close', 0.0)) or 0.0)
        low = float(row.get('low', row.get('close', 0.0)) or 0.0)
        entry = float(self.engine.position_price or 0.0)
        if entry <= 0:
            return None
        conservative = bool(config.get('conservative_same_bar_exit', True))
        if self.engine.position_direction == TradeDirection.LONG:
            tp_price = entry * (1 + tp_pct)
            sl_price = entry * (1 - sl_pct)
            hit_tp = tp_pct > 0 and high >= tp_price
            hit_sl = sl_pct > 0 and low <= sl_price
            if hit_tp and hit_sl:
                return {'action': 'EXIT_LONG', 'price': sl_price if conservative else tp_price, 'reason': '同K线触发止盈止损，保守按止损' if conservative else '同K线触发止盈止损，按止盈'}
            if hit_sl:
                return {'action': 'EXIT_LONG', 'price': sl_price, 'reason': f'固定止损 {sl_pct * 100:.2f}%'}
            if hit_tp:
                return {'action': 'EXIT_LONG', 'price': tp_price, 'reason': f'固定止盈 {tp_pct * 100:.2f}%'}
        if self.engine.position_direction == TradeDirection.SHORT:
            tp_price = entry * (1 - tp_pct)
            sl_price = entry * (1 + sl_pct)
            hit_tp = tp_pct > 0 and low <= tp_price
            hit_sl = sl_pct > 0 and high >= sl_price
            if hit_tp and hit_sl:
                return {'action': 'EXIT_SHORT', 'price': sl_price if conservative else tp_price, 'reason': '同K线触发止盈止损，保守按止损' if conservative else '同K线触发止盈止损，按止盈'}
            if hit_sl:
                return {'action': 'EXIT_SHORT', 'price': sl_price, 'reason': f'固定止损 {sl_pct * 100:.2f}%'}
            if hit_tp:
                return {'action': 'EXIT_SHORT', 'price': tp_price, 'reason': f'固定止盈 {tp_pct * 100:.2f}%'}
        return None

    def _generate_signal(
        self,
        strategy,
        df_map: Dict[str, pd.DataFrame],
        driver_bar: str,
        index: int,
        inst_id: str = "",
        signal_context: Optional[Dict[str, Any]] = None,
        current_ts_ns: Optional[int] = None,
        current_ts: Optional[datetime] = None,
        current_price: float = 0.0,
        config: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict]:
        """生成交易信号"""
        try:
            driver_df = (signal_context or {}).get('driver_df') if signal_context else None
            if driver_df is None:
                driver_df = df_map.get(driver_bar)
            if driver_df is None or driver_df.empty or index >= len(driver_df):
                return None

            if current_ts is None:
                current_ts = driver_df.iloc[index]['timestamp']
            if current_ts_ns is None:
                current_ts_ns = int(pd.Timestamp(current_ts).value)

            window = self._signal_window
            current_slice = None

            klines_map = {}
            context_bars = (signal_context or {}).get('bars', {})
            for bar_name, bar_cache in context_bars.items():
                timestamps_ns = bar_cache['timestamps_ns']
                cursor = int(bar_cache.get('cursor', 0))
                total = len(timestamps_ns)
                while cursor < total and timestamps_ns[cursor] <= current_ts_ns:
                    cursor += 1
                bar_cache['cursor'] = cursor
                if cursor <= 0:
                    continue
                start = max(0, cursor - window)
                klines_map[bar_name] = bar_cache['rows'][start:cursor]

            if not klines_map:
                for bar_name, bar_df in df_map.items():
                    sliced = bar_df.iloc[max(0, index + 1 - window):index+1] if bar_name == driver_bar else bar_df[bar_df['timestamp'] <= current_ts].tail(window)
                    if not sliced.empty:
                        klines_map[bar_name] = self._df_to_okx_rows(sliced)

            # 准备多种数据格式以兼容不同策略
            data = {
                'hourly': klines_map.get(driver_bar, []),
                'hourly_df': None,
                'klines': klines_map.get(driver_bar, []),
                'klines_map': klines_map,
                'indicator_map': self._build_indicator_window(signal_context or {}, window=window),
            }

            if hasattr(strategy, 'generate_signal'):
                current_slice = driver_df.iloc[max(0, index + 1 - window):index+1]
                data['hourly_df'] = current_slice
                return strategy.generate_signal(data)
            elif hasattr(strategy, 'check'):
                current_slice = driver_df.iloc[max(0, index + 1 - window):index+1]
                data['hourly_df'] = current_slice
                return strategy.check(data)
            elif hasattr(strategy, 'scan_symbol'):
                cache_key = str(inst_id or getattr(strategy, '__class__', type(strategy)).__name__)
                stride = int((config or {}).get('scanner_signal_stride_bars', 1) or 1)
                # 有持仓时不跳过 — 让 execute_buy/execute_sell 自行判断方向冲突
                cached = self._scanner_signal_cache.get(cache_key)
                if cached and stride > 1 and (index - int(cached.get('last_index', -10**9))) < stride:
                    return cached.get('signal')
                strategy_config = getattr(strategy, 'config', {}) or {}
                base_df = df_map.get('1H') or df_map.get(driver_bar) or next(iter(df_map.values()))
                base_cache = context_bars.get('1H') or context_bars.get(driver_bar)
                if base_cache is not None:
                    base_cursor = int(base_cache.get('cursor', 0))
                    base_history = base_df.iloc[max(0, base_cursor - 48):base_cursor] if base_cursor > 0 else base_df.iloc[0:0]
                else:
                    base_history = base_df[base_df['timestamp'] <= current_ts].tail(48)
                symbol = self._build_scanner_symbol_from_history(
                    inst_id or str(strategy_config.get('inst_id', '') or ''),
                    base_history,
                    float(current_price or driver_df.iloc[index]['close']),
                    klines_map,
                    indicator_map=data.get('indicator_map'),
                )
                # 回测配置里没有 inst_id 传给策略对象时，用结果展示交易对不影响交易逻辑。
                if not symbol.inst_id:
                    symbol.inst_id = "BACKTEST-SYMBOL"
                result = strategy.scan_symbol(symbol)
                signal = self._scanner_result_to_signal(result, float(current_price or driver_df.iloc[index]['close']), strategy_config)
                self._scanner_signal_cache[cache_key] = {
                    'last_index': index,
                    'signal': signal,
                }
                return signal

        except Exception as e:
            # 静默失败，不输出大量日志
            pass

        return None
