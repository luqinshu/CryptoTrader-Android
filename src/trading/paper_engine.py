"""
模拟交易引擎

完全复现实盘交易所有功能，但不向交易所发送真实订单。
接口与 TradeExecutor 对齐，可直接替换 StrategyRunner 中的 trade_executor。
"""

import json
import os
import time
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any


# ────────────────────────────────────────────────────────────────────────────
# 数据结构
# ────────────────────────────────────────────────────────────────────────────

class _Side:
    """兼容 TradeExecutor Position.side 接口"""
    def __init__(self, name: str):
        self.name = name

    def __str__(self):
        return self.name


@dataclass
class PaperPosition:
    """模拟持仓"""
    inst_id: str
    direction: str          # 'LONG' | 'SHORT'
    entry_price: float
    size: float             # 合约数量（USDT 名义价值 / 实际入场价）
    usdt_amount: float      # 投入本金（不含杠杆）
    leverage: int
    tp_price: float = 0.0
    sl_price: float = 0.0
    entry_time: str = ""
    entry_reason: str = ""
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    margin: float = 0.0     # 占用保证金

    @property
    def side(self) -> _Side:
        return _Side(self.direction)


@dataclass
class PaperTrade:
    """单笔模拟成交记录"""
    session_id: str
    strategy_name: str
    inst_id: str
    direction: str
    entry_time: str
    exit_time: str
    entry_price: float
    exit_price: float
    size: float
    usdt_amount: float
    leverage: int
    pnl: float
    pnl_pct: float
    fee: float
    slippage_cost: float
    entry_reason: str
    exit_reason: str


class PaperTradeResult:
    """模拟 TradeExecutor 返回值"""
    def __init__(self, success: bool, message: str = "",
                 order_id: str = "", filled_size: float = 0.0, pnl: float = 0.0):
        self.success = success
        self.message = message
        self.order_id = order_id
        self.filled_size = filled_size
        self.pnl = pnl


# ────────────────────────────────────────────────────────────────────────────
# 核心引擎
# ────────────────────────────────────────────────────────────────────────────

class PaperTradeEngine:
    """
    模拟交易引擎。

    与 TradeExecutor 同接口，可直接传入 StrategyRunner / PaperStrategyRunner。
    所有"成交"均在本地模拟，费用 / 滑点 / 资金费率全部扣除，
    结果实时保存到 JSON 报告供事后复盘。
    """

    REPORTS_DIR = Path(__file__).parent.parent / "reports" / "paper_trades"

    def __init__(
        self,
        okx_client,
        initial_capital: float = 10_000.0,
        fee_pct: float = 0.05,
        slippage_pct: float = 0.03,
        market_impact_pct: float = 0.02,
        funding_rate_8h_pct: float = 0.01,
        strategy_name: str = "",
        session_id: str = "",
    ):
        self.okx_client = okx_client
        self.initial_capital = initial_capital
        self.balance = initial_capital           # 可用余额（含浮盈浮亏变化）
        self._fee_rate = fee_pct / 100.0
        self._slip_rate = slippage_pct / 100.0
        self._impact_rate = market_impact_pct / 100.0
        self._funding_rate = funding_rate_8h_pct / 100.0
        self.strategy_name = strategy_name
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.start_time = datetime.now().isoformat()

        self._positions: Dict[str, PaperPosition] = {}
        self._trades: List[PaperTrade] = []
        self._order_counter = 0
        self._state_lock = threading.RLock()

        # 异步节流写盘：避免每次成交都 json.dump 整个 trade list
        self._dirty = False
        self._save_stop_event = threading.Event()
        self._save_thread = threading.Thread(target=self._save_loop, daemon=True)
        self._save_thread.start()

        self.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── 接口：余额 / 持仓 ────────────────────────────────────────────────────

    def get_usdt_balance(self) -> float:
        with self._state_lock:
            return self.balance

    def get_positions(self, inst_id: str = None) -> Dict:
        with self._state_lock:
            if inst_id:
                pos = self._positions.get(inst_id)
                return {inst_id: pos} if pos else {}
            return dict(self._positions)

    # ── 接口：开仓 ───────────────────────────────────────────────────────────

    def execute_entry(
        self,
        inst_id: str,
        direction: str,
        usdt_amount: float = 0.0,
        leverage: int = 1,
        tp_pct: float = 0.05,
        sl_pct: float = 0.03,
        order_type: str = "market",
        price: float = None,
        reason: str = "",
        **kwargs,
    ) -> PaperTradeResult:
        market_price = self._market_price(inst_id)
        if market_price <= 0:
            return PaperTradeResult(False, f"无法获取 {inst_id} 市价")

        exec_price = (price if order_type == "limit" and price and price > 0
                      else market_price)
        exec_price = self._with_slippage(exec_price, direction, entry=True)
        leverage = max(int(leverage or 1), 1)
        notional = usdt_amount * leverage
        fee = notional * self._fee_rate
        total_cost = usdt_amount + fee

        with self._state_lock:
            if inst_id in self._positions:
                return PaperTradeResult(False, "已有持仓，不允许重复开仓")

            if total_cost > self.balance:
                return PaperTradeResult(
                    False,
                    f"模拟余额不足：需要 {total_cost:.2f} USDT，当前 {self.balance:.2f} USDT"
                )

            size = notional / exec_price
            self.balance -= total_cost

            if direction == 'LONG':
                tp_price = exec_price * (1 + tp_pct)
                sl_price = exec_price * (1 - sl_pct)
            else:
                tp_price = exec_price * (1 - tp_pct)
                sl_price = exec_price * (1 + sl_pct)

            self._order_counter += 1
            order_id = f"PAPER-{self.session_id}-{self._order_counter:04d}"
            pos = PaperPosition(
                inst_id=inst_id,
                direction=direction,
                entry_price=exec_price,
                size=size,
                usdt_amount=usdt_amount,
                leverage=leverage,
                tp_price=tp_price,
                sl_price=sl_price,
                entry_time=datetime.now().isoformat(),
                entry_reason=reason,
                margin=usdt_amount,
                current_price=exec_price,
            )
            self._positions[inst_id] = pos
            self._save()
        return PaperTradeResult(
            True,
            f"[模拟] {direction} 开仓 {inst_id} @ {exec_price:.6f}，"
            f"保证金={usdt_amount:.2f}，数量={size:.6f}，名义={notional:.2f} USDT，手续费={fee:.4f} USDT",
            order_id=order_id,
            filled_size=size,
        )

    # ── 接口：平仓 ───────────────────────────────────────────────────────────

    def execute_sell(self, inst_id: str, size: float = None,
                     exit_reason: str = "策略平多") -> PaperTradeResult:
        return self._close(inst_id, 'LONG', exit_reason)

    def execute_cover(self, inst_id: str, size: float = None,
                      exit_reason: str = "策略平空") -> PaperTradeResult:
        return self._close(inst_id, 'SHORT', exit_reason)

    def execute_stop_loss(self, inst_id: str,
                          exit_reason: str = "止损") -> PaperTradeResult:
        with self._state_lock:
            pos = self._positions.get(inst_id)
        if not pos:
            return PaperTradeResult(False, "无持仓")
        return self._close(inst_id, pos.direction, exit_reason)

    def _close(self, inst_id: str, direction: str, reason: str) -> PaperTradeResult:
        with self._state_lock:
            pos = self._positions.get(inst_id)
        if not pos:
            return PaperTradeResult(False, "无持仓")

        market_price = self._market_price(inst_id)
        if market_price <= 0:
            market_price = pos.current_price or pos.entry_price

        exec_price = self._with_slippage(market_price, direction, entry=False)

        # 资金费用估算（按持仓时间/8h 计次）
        entry_ts = self._parse_ts(pos.entry_time)
        hold_hours = (time.time() - entry_ts) / 3600.0 if entry_ts else 0.0
        funding_intervals = hold_hours / 8.0
        notional = pos.size * exec_price
        funding_cost = notional * self._funding_rate * funding_intervals

        exit_fee = notional * self._fee_rate

        if direction == 'LONG':
            gross_pnl = (exec_price - pos.entry_price) * pos.size
        else:
            gross_pnl = (pos.entry_price - exec_price) * pos.size

        pnl = gross_pnl - exit_fee - funding_cost
        pnl_pct = pnl / pos.usdt_amount * 100 if pos.usdt_amount > 0 else 0.0

        with self._state_lock:
            current = self._positions.get(inst_id)
            if not current:
                return PaperTradeResult(False, "无持仓")
            self.balance += current.usdt_amount + pnl   # 归还本金 + 净盈亏

            trade = PaperTrade(
                session_id=self.session_id,
                strategy_name=self.strategy_name,
                inst_id=inst_id,
                direction=direction,
                entry_time=current.entry_time,
                exit_time=datetime.now().isoformat(),
                entry_price=current.entry_price,
                exit_price=exec_price,
                size=current.size,
                usdt_amount=current.usdt_amount,
                leverage=current.leverage,
                pnl=round(pnl, 6),
                pnl_pct=round(pnl_pct, 4),
                fee=round(exit_fee, 6),
                slippage_cost=round(abs(exec_price - market_price) * current.size, 6),
                entry_reason=current.entry_reason,
                exit_reason=reason,
            )
            self._trades.append(trade)
            del self._positions[inst_id]
            self._save()
        return PaperTradeResult(
            True,
            f"[模拟] 平仓 {inst_id} @ {exec_price:.6f}，"
            f"盈亏={pnl:+.2f} USDT ({pnl_pct:+.2f}%)",
            pnl=pnl,
        )

    def _liquidate(self, inst_id: str, pos, price: float):
        """强平：亏损超过保证金 90% 时强制平仓。"""
        direction = pos.direction
        loss_ratio = abs(pos.unrealized_pnl) / max(pos.usdt_amount, 1e-9) * 100
        # 强平后余额 = 原余额 - 亏损（保证金已全部损失）
        self.balance -= pos.usdt_amount
        trade = PaperTrade(
            session_id=self.session_id, strategy_name=self.strategy_name,
            inst_id=inst_id, direction=direction,
            entry_time=pos.entry_time,
            exit_time=datetime.now().isoformat(),
            entry_price=pos.entry_price, exit_price=price,
            size=pos.size, usdt_amount=pos.usdt_amount, leverage=pos.leverage,
            pnl=round(-pos.usdt_amount * 0.90, 6),
            pnl_pct=round(-90.0, 4),
            fee=0.0, exit_reason=f"强平(亏损{loss_ratio:.1f}%)",
        )
        self._trades.append(trade)
        del self._positions[inst_id]
        self._mark_dirty()

    # ── 持仓更新（每轮主循环调用）───────────────────────────────────────────

    def update_prices(self):
        """更新所有持仓的当前价格、浮动盈亏，并检查强平。"""
        with self._state_lock:
            positions = list(self._positions.items())
        for inst_id, pos in positions:
            price = self._market_price(inst_id)
            if price > 0:
                with self._state_lock:
                    current = self._positions.get(inst_id)
                    if not current:
                        continue
                    current.current_price = price
                    if current.direction == 'LONG':
                        current.unrealized_pnl = (price - current.entry_price) * current.size
                    else:
                        current.unrealized_pnl = (current.entry_price - price) * current.size
                    # 强平检查：浮动亏损超过保证金的 90% 触发强平
                    if current.unrealized_pnl < 0 and current.usdt_amount > 0:
                        loss_ratio = abs(current.unrealized_pnl) / current.usdt_amount
                        if loss_ratio >= 0.90:
                            self._liquidate(inst_id, current, price)

    def close_position_partial(self, inst_id: str, ratio: float = 0.5) -> PaperTradeResult:
        """模拟分批平仓（按比例减仓）"""
        with self._state_lock:
            pos = self._positions.get(inst_id)
        if not pos:
            return PaperTradeResult(False, "无持仓")
        if ratio <= 0 or ratio >= 1.0:
            return PaperTradeResult(False, f"减仓比例无效: {ratio}")

        market_price = self._market_price(inst_id)
        if market_price <= 0:
            market_price = pos.current_price or pos.entry_price

        exec_price = self._with_slippage(market_price, pos.direction, entry=False)
        close_size = pos.size * ratio
        close_usdt = pos.usdt_amount * ratio

        if pos.direction == 'LONG':
            gross_pnl = (exec_price - pos.entry_price) * close_size
        else:
            gross_pnl = (pos.entry_price - exec_price) * close_size

        fee = close_size * exec_price * self._fee_rate
        pnl = gross_pnl - fee

        with self._state_lock:
            current = self._positions.get(inst_id)
            if not current:
                return PaperTradeResult(False, "无持仓")
            self.balance += close_usdt + pnl
            current.size -= close_size
            current.usdt_amount -= close_usdt
            current.margin -= close_usdt

        return PaperTradeResult(
            True,
            f"[模拟] 减仓 {ratio*100:.0f}% {inst_id} @ {exec_price:.6f}，盈亏={pnl:+.4f} USDT",
            pnl=pnl,
        )

    def check_tp_sl(self, inst_id: str) -> Optional[str]:
        """检查止盈止损是否触发，返回原因字符串或 None"""
        with self._state_lock:
            pos = self._positions.get(inst_id)
            if not pos or pos.current_price <= 0:
                return None
            price = pos.current_price
            if pos.direction == 'LONG':
                if pos.sl_price > 0 and price <= pos.sl_price:
                    return f"止损触发 @ {price:.6f}（止损线 {pos.sl_price:.6f}）"
                if pos.tp_price > 0 and price >= pos.tp_price:
                    return f"止盈触发 @ {price:.6f}（止盈线 {pos.tp_price:.6f}）"
            else:
                if pos.sl_price > 0 and price >= pos.sl_price:
                    return f"止损触发 @ {price:.6f}（止损线 {pos.sl_price:.6f}）"
                if pos.tp_price > 0 and price <= pos.tp_price:
                    return f"止盈触发 @ {price:.6f}（止盈线 {pos.tp_price:.6f}）"
            return None

    # ── 统计 / 报告 ──────────────────────────────────────────────────────────

    def get_summary(self) -> Dict[str, Any]:
        with self._state_lock:
            trades = list(self._trades)
            balance = self.balance
        n = len(trades)
        if n == 0:
            return {
                'total_trades': 0, 'win_trades': 0, 'lose_trades': 0,
                'win_rate': 0.0, 'total_pnl': 0.0, 'total_return': 0.0,
                'balance': balance, 'initial_capital': self.initial_capital,
                'avg_pnl': 0.0, 'max_profit': 0.0, 'max_loss': 0.0,
                'total_fees': 0.0, 'total_slippage': 0.0,
            }
        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]
        total_pnl = sum(t.pnl for t in trades)
        return {
            'total_trades': n,
            'win_trades': len(wins),
            'lose_trades': len(losses),
            'win_rate': round(len(wins) / n * 100, 2),
            'total_pnl': round(total_pnl, 4),
            'total_return': round((balance - self.initial_capital) / self.initial_capital * 100, 4),
            'balance': round(balance, 4),
            'initial_capital': self.initial_capital,
            'avg_pnl': round(total_pnl / n, 4),
            'max_profit': round(max(t.pnl for t in trades), 4),
            'max_loss': round(min(t.pnl for t in trades), 4),
            'total_fees': round(sum(t.fee for t in trades), 4),
            'total_slippage': round(sum(t.slippage_cost for t in trades), 4),
        }

    def get_open_positions_info(self) -> List[Dict]:
        with self._state_lock:
            positions = list(self._positions.items())
        result = []
        for inst_id, pos in positions:
            result.append({
                'inst_id': inst_id,
                'direction': pos.direction,
                'entry_price': pos.entry_price,
                'current_price': pos.current_price,
                'size': pos.size,
                'usdt_amount': pos.usdt_amount,
                'leverage': pos.leverage,
                'unrealized_pnl': pos.unrealized_pnl,
                'margin': pos.margin,
                'tp_price': pos.tp_price,
                'sl_price': pos.sl_price,
                'entry_time': pos.entry_time,
                'entry_reason': pos.entry_reason,
            })
        return result

    def _save(self):
        """标记脏数据（由后台线程节流写盘，避免每次成交都同步 I/O）。"""
        self._dirty = True

    def _save_loop(self):
        """后台线程：每秒检查脏标记，有变更时异步写 JSON 报告。"""
        while not self._save_stop_event.wait(1.0):
            if not self._dirty:
                continue
            self._dirty = False
            path = self.REPORTS_DIR / f"session_{self.session_id}.json"
            # 数据拷贝在锁内、json.dump 在锁外，避免持有锁时阻塞 I/O
            with self._state_lock:
                trades = [asdict(t) for t in self._trades]
                balance = self.balance
            data = {
                'session_id': self.session_id,
                'strategy_name': self.strategy_name,
                'start_time': self.start_time,
                'initial_capital': self.initial_capital,
                'balance': balance,
                'fee_pct': self._fee_rate * 100,
                'slippage_pct': self._slip_rate * 100,
                'funding_rate_8h_pct': self._funding_rate * 100,
                'trades': trades,
                'summary': self.get_summary(),
                'saved_at': datetime.now().isoformat(),
            }
            try:
                tmp_path = path.with_suffix('.tmp')
                with open(tmp_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                os.replace(tmp_path, path)  # 原子替换，防进程崩溃半截文件
            except Exception as e:
                print(f"[PaperEngine] 保存报告失败: {e}")

    def save_final(self):
        """会话结束时保存最终报告（强制立即写盘并停止后台线程）。"""
        self._save_stop_event.set()
        self._dirty = True
        # 最后一次同步写盘，确保数据完整落盘
        path = self.REPORTS_DIR / f"session_{self.session_id}.json"
        with self._state_lock:
            trades = [asdict(t) for t in self._trades]
            balance = self.balance
        data = {
            'session_id': self.session_id,
            'strategy_name': self.strategy_name,
            'start_time': self.start_time,
            'initial_capital': self.initial_capital,
            'balance': balance,
            'fee_pct': self._fee_rate * 100,
            'slippage_pct': self._slip_rate * 100,
            'funding_rate_8h_pct': self._funding_rate * 100,
            'trades': trades,
            'summary': self.get_summary(),
            'saved_at': datetime.now().isoformat(),
        }
        try:
            tmp_path = path.with_suffix('.tmp')
            with open(tmp_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, path)
        except Exception as e:
            print(f"[PaperEngine] 保存最终报告失败: {e}")
        return str(path)

    # ── 历史会话 ─────────────────────────────────────────────────────────────

    @classmethod
    def list_sessions(cls) -> List[Dict]:
        """列出所有历史模拟交易会话（按时间倒序）"""
        if not cls.REPORTS_DIR.exists():
            return []
        sessions = []
        for f in sorted(cls.REPORTS_DIR.glob("session_*.json"), reverse=True):
            try:
                with open(f, encoding='utf-8') as fp:
                    d = json.load(fp)
                summary = d.get('summary', {})
                sessions.append({
                    'file': str(f),
                    'session_id': d.get('session_id', ''),
                    'strategy_name': d.get('strategy_name', '-'),
                    'start_time': d.get('start_time', '-'),
                    'saved_at': d.get('saved_at', '-'),
                    'initial_capital': d.get('initial_capital', 0),
                    'balance': summary.get('balance', 0),
                    'total_trades': summary.get('total_trades', 0),
                    'win_rate': summary.get('win_rate', 0.0),
                    'total_pnl': summary.get('total_pnl', 0.0),
                    'total_return': summary.get('total_return', 0.0),
                })
            except Exception:
                pass
        return sessions

    @classmethod
    def load_session(cls, file_path: str) -> Dict:
        """加载单个历史会话全量数据"""
        with open(file_path, encoding='utf-8') as f:
            return json.load(f)

    # ── 工具 ─────────────────────────────────────────────────────────────────

    def _market_price(self, inst_id: str) -> float:
        try:
            ticker = self.okx_client.get_ticker(inst_id)
            if ticker and 'data' in ticker and ticker['data']:
                return float(ticker['data'][0].get('last', 0) or 0)
        except Exception:
            pass
        return 0.0

    def _with_slippage(self, price: float, direction: str, entry: bool) -> float:
        cost = self._slip_rate + self._impact_rate
        if entry and direction == 'LONG':
            return price * (1 + cost)
        if entry and direction == 'SHORT':
            return price * (1 - cost)
        if not entry and direction == 'LONG':
            return price * (1 - cost)
        return price * (1 + cost)   # exit SHORT

    @staticmethod
    def _parse_ts(iso_str: str) -> float:
        try:
            return datetime.fromisoformat(iso_str).timestamp()
        except Exception:
            return 0.0
