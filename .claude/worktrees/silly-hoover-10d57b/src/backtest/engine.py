"""
回测引擎模块
支持策略历史数据回测和绩效分析
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import threading


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
    pnl: float = 0.0
    pnl_percent: float = 0.0
    exit_reason: str = ""


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

    def reset(self):
        """重置引擎状态"""
        self.capital = self.initial_capital
        self.position = 0.0
        self.position_price = 0.0
        self.position_direction = None
        self.trades = []
        self.equity_curve = []
        self.current_trade = None

    def execute_buy(self, timestamp: datetime, price: float, size: float, 
                    position_ratio: float = None) -> bool:
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
        if self.position != 0:
            return False

        # 计算可买入数量
        if size is None:
            ratio = position_ratio or 1.0
            available_capital = self.capital * ratio
            size = available_capital / price

        cost = size * price
        if cost > self.capital:
            size = self.capital / price
            cost = size * price

        self.position = size
        self.position_price = price
        self.position_direction = TradeDirection.LONG
        self.capital -= cost

        # 记录交易
        self.current_trade = Trade(
            entry_time=timestamp,
            exit_time=None,
            direction=TradeDirection.LONG,
            entry_price=price,
            exit_price=None,
            size=size
        )

        return True

    def execute_sell(self, timestamp: datetime, price: float, size: float = None) -> bool:
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
        pnl = (price - self.position_price) * size
        pnl_percent = (price - self.position_price) / self.position_price * 100

        # 更新资金
        self.capital += size * price
        self.position -= size

        # 记录交易
        if self.current_trade:
            self.current_trade.exit_time = timestamp
            self.current_trade.exit_price = price
            self.current_trade.pnl = pnl
            self.current_trade.pnl_percent = pnl_percent
            self.current_trade.exit_reason = "止盈/止损/信号"
            self.trades.append(self.current_trade)
            self.current_trade = None

        if self.position <= 0:
            self.position_direction = None

        return True

    def execute_short(self, timestamp: datetime, price: float, size: float,
                      position_ratio: float = None) -> bool:
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
        if self.position != 0:
            return False

        if size is None:
            ratio = position_ratio or 1.0
            available_capital = self.capital * ratio
            size = available_capital / price

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
            size=size
        )

        return True

    def execute_cover(self, timestamp: datetime, price: float, size: float = None) -> bool:
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
        pnl = (self.position_price - price) * size
        pnl_percent = (self.position_price - price) / self.position_price * 100

        # 更新资金
        self.capital += pnl
        self.position -= size

        # 记录交易
        if self.current_trade:
            self.current_trade.exit_time = timestamp
            self.current_trade.exit_price = price
            self.current_trade.pnl = pnl
            self.current_trade.pnl_percent = pnl_percent
            self.current_trade.exit_reason = "止盈/止损/信号"
            self.trades.append(self.current_trade)
            self.current_trade = None

        if self.position <= 0:
            self.position_direction = None

        return True

    def update_equity(self, timestamp: datetime, current_price: float):
        """更新权益曲线"""
        if self.position > 0:
            if self.position_direction == TradeDirection.LONG:
                equity = self.capital + self.position * current_price
            else:
                equity = self.capital + (self.position_price - current_price) * self.position
        else:
            equity = self.capital

        self.equity_curve.append((timestamp, equity))
        return equity

    def get_current_equity(self, current_price: float) -> float:
        """获取当前权益"""
        if self.position > 0:
            if self.position_direction == TradeDirection.LONG:
                return self.capital + self.position * current_price
            else:
                return self.capital + (self.position_price - current_price) * self.position
        return self.capital


class BacktestAnalyzer:
    """回测结果分析器"""

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
        if result.equity_curve:
            equity_series = pd.Series([e[1] for e in result.equity_curve])
            running_max = equity_series.cummax()
            drawdown = (equity_series - running_max) / running_max * 100
            result.max_drawdown = abs(drawdown.min())

        return result

    @staticmethod
    def generate_report(result: BacktestResult) -> str:
        """
        生成回测报告

        Args:
            result: 回测结果

        Returns:
            报告文本
        """
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
        self.initial_capital = initial_capital
        self.engine = BacktestEngine(initial_capital)
        self.analyzer = BacktestAnalyzer()

    def run_backtest(self, strategy, inst_id: str, start_date: str, end_date: str,
                     bar: str = "1H", config: Dict = None) -> BacktestResult:
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

        # 获取历史数据
        klines = self._fetch_klines(inst_id, bar, start_date, end_date)
        if not klines or len(klines) == 0:
            return BacktestResult(
                strategy_name=strategy.__class__.__name__,
                inst_id=inst_id,
                start_date=datetime.now(),
                end_date=datetime.now(),
                initial_capital=self.initial_capital
            )

        # 解析 K 线数据
        df = self._parse_klines(klines)

        # 初始化策略
        if config:
            strategy.config = config

        # 回测主循环
        result = BacktestResult(
            strategy_name=strategy.__class__.__name__,
            inst_id=inst_id,
            start_date=df['timestamp'].iloc[0],
            end_date=df['timestamp'].iloc[-1],
            initial_capital=self.initial_capital
        )

        for i in range(len(df) - 1):
            row = df.iloc[i]
            timestamp = row['timestamp']
            current_price = row['close']

            # 更新权益曲线
            self.engine.update_equity(timestamp, current_price)

            # 生成交易信号
            signal = self._generate_signal(strategy, df, i)

            if signal:
                action = signal.get('action', '')
                position_size = signal.get('position_size', 0.1)
                entry_price = signal.get('entry_price', current_price)

                if action in ['BUY', 'BUY_LONG'] and self.engine.position == 0:
                    self.engine.execute_buy(timestamp, entry_price, None, position_size)

                elif action in ['SELL', 'EXIT_LONG'] and self.engine.position > 0:
                    self.engine.execute_sell(timestamp, current_price)

                elif action in ['SHORT', 'SELL_SHORT'] and self.engine.position == 0:
                    self.engine.execute_short(timestamp, entry_price, None, position_size)

                elif action in ['COVER', 'EXIT_SHORT'] and self.engine.position > 0:
                    self.engine.execute_cover(timestamp, current_price)

                elif action == 'STOP_LOSS':
                    if self.engine.position_direction == TradeDirection.LONG:
                        self.engine.execute_sell(timestamp, current_price)
                    else:
                        self.engine.execute_cover(timestamp, current_price)

        # 结束时平仓
        if len(df) > 0:
            final_price = df['close'].iloc[-1]
            final_time = df['timestamp'].iloc[-1]
            if self.engine.position > 0:
                if self.engine.position_direction == TradeDirection.LONG:
                    self.engine.execute_sell(final_time, final_price)
                else:
                    self.engine.execute_cover(final_time, final_price)

        # 填充结果
        result.trades = self.engine.trades.copy()
        result.equity_curve = self.engine.equity_curve.copy()
        result.total_trades = len(result.trades)
        result.final_capital = self.engine.get_current_equity(df['close'].iloc[-1])

        # 分析结果
        result = self.analyzer.analyze(result)

        return result

    def _fetch_klines(self, inst_id: str, bar: str, start_date: str, 
                      end_date: str) -> List:
        """获取 K 线数据"""
        if not self.okx_client:
            return []

        try:
            result = self.okx_client.get_kline(inst_id, bar=bar, limit=300)
            if result and result.get('code') == '0':
                return result.get('data', [])
        except Exception as e:
            print(f"获取 K 线失败：{e}")

        return []

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

        return df.sort_values('timestamp').reset_index(drop=True)

    def _generate_signal(self, strategy, df: pd.DataFrame, index: int) -> Optional[Dict]:
        """生成交易信号"""
        try:
            # 准备数据 - 转换为 OKX API 格式（列表格式）
            hourly_data = df.iloc[:index+1].tail(500)
            
            # 转换为 OKX API 格式列表
            hourly_list = []
            for _, row in hourly_data.iterrows():
                hourly_list.append([
                    int(row['timestamp'].timestamp() * 1000),  # 毫秒时间戳
                    str(row['open']),
                    str(row['high']),
                    str(row['low']),
                    str(row['close']),
                    str(row['volume']),
                    '', '', ''
                ])
            
            # 准备多种数据格式以兼容不同策略
            data = {
                'hourly': hourly_list,  # OKX API 列表格式
                'hourly_df': hourly_data,  # DataFrame 格式
                'klines': hourly_list,  # 通用 klines 格式
            }

            if hasattr(strategy, 'generate_signal'):
                return strategy.generate_signal(data)
            elif hasattr(strategy, 'check'):
                return strategy.check(data)

        except Exception as e:
            # 静默失败，不输出大量日志
            pass

        return None
