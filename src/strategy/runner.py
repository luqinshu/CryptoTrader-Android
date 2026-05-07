"""
策略运行器
负责执行策略逻辑并生成交易信号。
"""

import time
from typing import Dict, Optional, List, Any
from dataclasses import dataclass, field
from enum import Enum

import pandas as pd

from src.qt_compat import QObject, Signal
from src.trading.entry_rule_guard import evaluate_entry_rule_from_frames
from src.trading.position_registry import position_registry


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


@dataclass
class PendingAutoSignal:
    direction: str
    signal_price: float
    reason: str
    created_at: float
    tp_pct_override: Optional[float] = None
    sl_pct_override: Optional[float] = None
    raw_signal: Dict[str, Any] = field(default_factory=dict)
    strategy_gates: set = field(default_factory=set)


@dataclass
class AutoTradeCampaign:
    direction: str
    first_signal_price: float
    stage1_entry_price: float = 0.0
    stage1_cost_line: float = 0.0
    stage2_entry_price: float = 0.0
    highest_since_stage1: float = 0.0
    lowest_since_stage1: float = 0.0
    stage2_armed: bool = False
    stage2_filled: bool = False
    total_allocated_usdt: float = 0.0
    opened_at: float = 0.0
    last_reason: str = ""
    peak_profit_price: float = 0.0   # 移动止盈/Stage1浮盈保护基准价
    tp_pct_override: Optional[float] = None
    sl_pct_override: Optional[float] = None
    # ── P1 新增字段 ─────────────────────────────────────────────────────────
    partial_exit_done: bool = False  # Stage2后是否已执行分批减仓（锁定50%利润）
    stage1_trail_armed: bool = False # Stage1期间浮盈保护是否已激活
    stage1_atr_buffer_pct: float = 0.0  # Stage1 时的 ATR 缓冲百分比，Stage2 复用锁定
    # ── Stage3：趋势确认后的第三次金字塔加仓 ─────────────────────────────
    stage3_armed: bool = False       # Stage2完成后，趋势再涨 N% 且 MACD 确认时置 True
    stage3_filled: bool = False      # Stage3 加仓已执行
    stage3_entry_price: float = 0.0  # Stage3 开仓价


class StrategyRunner(QObject):
    """自动交易策略运行器。

    第一原则：
    任何开仓、加仓、持仓都不得违反总开仓成本线（开仓线 + 手续费 + 滑点/冲击）。
    只要跌破或上破该风险线，立即离场，不允许继续持有。
    """

    # 信号
    log_signal = Signal(str, str)  # 日志信号
    trade_signal = Signal(str, str, float, float)  # 交易信号 (action, inst_id, price, size)
    state_signal = Signal(dict)  # 自动交易状态
    finished = Signal()
    error_signal = Signal(str)

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
        self._pending_signal: Optional[PendingAutoSignal] = None
        self._campaign: Optional[AutoTradeCampaign] = None
        self._last_state: Dict[str, Any] = {}
        self._consecutive_losses: int = 0   # 连续亏损计数，触发熔断
        self._trade_history: List[bool] = []  # 近期交易盈亏记录（True=盈利）
        self._close_retry_count: int = 0
        self._consecutive_kline_fails: int = 0   # 连续 K 线获取失败次数
        self._kline_fail_limit: int = self._config_int('kline_fail_limit', 10)  # 连续失败上限
        self._last_kline_fail_code: str = '0'     # 最近一次 K 线失败原因码
        self._position_opened_at: float = 0.0      # 持仓开仓时间戳（独立于 campaign，防 campaign 提前清空后失去持有时间参照）

        # 注册接管回调：当其他系统强制接管我们的 inst_id 时清空内部状态
        position_registry.register_takeover_callback(self._on_takeover)

    def run(self):
        """运行自动交易主循环。"""
        self._running = True
        self._stop_flag = False

        # ── 孤儿仓位恢复：启动时检测是否已有交易所持仓 ──────────────────
        self._recover_orphan_position()

        try:
            while not self._stop_flag:
                try:
                    klines = self._get_klines()
                    if not klines:
                        self._consecutive_kline_fails += 1
                        fail_code = getattr(self, '_last_kline_fail_code', '-1')

                        # 限流超时用更长的退避时间，让令牌桶有时间恢复
                        if fail_code == '-2':
                            backoff = min(2 ** (self._consecutive_kline_fails - 1), 30)
                            self._interruptible_sleep(backoff)
                        else:
                            self._interruptible_sleep(5)

                        # 连续失败达到上限 → 告警并自停
                        if self._consecutive_kline_fails >= self._kline_fail_limit:
                            self.log_signal.emit(
                                f"[{self.inst_id}] 连续 {self._consecutive_kline_fails} 次"
                                f"获取 K 线失败（最后错误码={fail_code}），自动暂停交易",
                                "ERROR",
                            )
                            self.error_signal.emit(
                                f"连续{self._consecutive_kline_fails}次K线获取失败"
                            )
                            self._stop_flag = True
                        continue

                    # K 线获取成功 → 重置失败计数器
                    self._consecutive_kline_fails = 0

                    raw_signal = self.strategy.generate_signal(klines)
                    self._process_auto_trade_cycle(raw_signal, klines)
                    self._interruptible_sleep(
                        max(self._config_int('auto_loop_interval_seconds', 5), 1))

                except Exception as e:
                    self.log_signal.emit(f"自动交易执行错误：{str(e)}", "ERROR")
                    self._interruptible_sleep(5)

        except Exception as e:
            self.log_signal.emit(f"自动交易策略异常：{str(e)}", "ERROR")
            self.error_signal.emit(str(e))

        finally:
            self._running = False
            self.finished.emit()

    def stop(self):
        """停止策略"""
        self.log_signal.emit("正在停止自动交易策略...", "WARNING")
        self._stop_flag = True
        position_registry.unregister_takeover_callback(self._on_takeover)

    def _interruptible_sleep(self, seconds: float, step: float = 0.1):
        """分片 sleep，每 step 秒检查 _stop_flag，避免 QThread 退出前卡死。"""
        steps = max(1, int(seconds / step))
        for _ in range(steps):
            if self._stop_flag:
                return
            time.sleep(step)

    def _now_ts(self) -> float:
        """统一时间源，便于回测模式复用状态机逻辑。"""
        return time.time()

    def _get_klines(self) -> Optional[Dict]:
        """获取 K 线数据。

        返回 None 表示核心周期（3m/1H）获取失败。辅助周期（4H/15m/daily）
        失败时返回部分数据，不影响主流程。

        通过 _last_kline_fail_code 暴露失败原因：
          '-2' = 限流超时（需退避），'-3' = 网络超时，'-1' = 其他错误。
        """
        try:
            daily = self.okx_client.get_kline(self.inst_id, bar="1D", limit=100)
            h4 = self.okx_client.get_kline(self.inst_id, bar="4H", limit=240)
            hourly = self.okx_client.get_kline(self.inst_id, bar="1H", limit=500)
            m15 = self.okx_client.get_kline(self.inst_id, bar="15m", limit=200)
            m3 = self.okx_client.get_kline(self.inst_id, bar="3m", limit=240)

            # 核心周期检查：3m 和 1H 必须成功
            m3_ok = m3 and m3.get('code') == '0'
            h1_ok = hourly and hourly.get('code') == '0'
            if not m3_ok or not h1_ok:
                # 记录最严重的错误码用于退避策略
                for resp in [m3, hourly]:
                    code = resp.get('code', '-1') if resp else '-1'
                    if code != '0':
                        self._last_kline_fail_code = code
                        self.log_signal.emit(
                            f"核心 K 线获取失败 (3m={m3_ok} 1H={h1_ok}, code={code})", "WARNING"
                        )
                        return None

            # 辅助周期：失败不阻塞，仅记录警告
            failed_timeframes = []
            for name, resp in [("4H", h4), ("15m", m15), ("daily", daily)]:
                code = resp.get('code', '-1') if resp else '-1'
                if code != '0':
                    failed_timeframes.append(name)
            if failed_timeframes:
                self.log_signal.emit(
                    f"辅助 K 线获取失败：{', '.join(failed_timeframes)}，继续使用已有数据",
                    "INFO",
                )

            self._last_kline_fail_code = '0'
            return {
                'daily': daily.get('data', []) if daily and daily.get('code') == '0' else [],
                '4h': h4.get('data', []) if h4 and h4.get('code') == '0' else [],
                'hourly': hourly.get('data', []),
                'm15': m15.get('data', []) if m15 and m15.get('code') == '0' else [],
                'm3': m3.get('data', []),
            }
        except Exception as e:
            self.log_signal.emit(f"获取 K 线失败：{str(e)}", "ERROR")
            self._last_kline_fail_code = '-1'
            return None

    def _process_auto_trade_cycle(self, raw_signal: Dict, klines: Dict):
        # ① 清理超时待命信号
        self._check_pending_signal_expiry()

        # ② 先注册入场信号（避免因市场数据暂时不足而丢弃信号）
        if (raw_signal and raw_signal.get('action') not in {'', 'HOLD', None}
                and self._campaign is None and self._pending_signal is None):
            self._register_pending_signal(raw_signal, {})

        market = self._build_market_snapshot(klines)
        if not market.get('valid'):
            if self._campaign:
                # 有持仓时，仅跳过建仓信号，仍然检查退出条件
                self.log_signal.emit(
                    f"自动交易：市场数据暂时不足({market.get('reason','')})，继续检查持仓风控", "WARNING")
                # 仍然执行关键退出检查（使用最近缓存的 market 数据）
                if hasattr(self, '_last_valid_market') and self._last_valid_market:
                    self._check_max_hold_time()
                    self._enforce_first_principle(self._last_valid_market)
            else:
                self.log_signal.emit(
                    f"自动交易等待：{market.get('reason', '市场数据不足')}", "INFO")
            return
        # 缓存最新有效 market 快照（供数据缺失时使用）
        self._last_valid_market = market

        if self._campaign:
            # ③ 超时强制平仓
            if self._check_max_hold_time():
                return
            # ④ 第一原则：ATR止损线守护
            if self._enforce_first_principle(market):
                return
            # ⑤ Stage2 移动止盈（加仓后激活）
            if self._check_trailing_stop(market):
                return
            # ⑥ Stage1 浮盈保护（加仓前激活，防止浮盈回吐）
            if self._check_stage1_trail_stop(market):
                return
            # ⑦ H1 趋势转折离场
            if self._check_hourly_reversal_exit(market):
                return
            # ⑧ 分批减仓（Stage2完成且浮盈达标时平掉50%）
            self._check_partial_exit(market)

        # 消费退出信号（market 有效时才执行）
        if raw_signal and raw_signal.get('action') not in {'', 'HOLD', None}:
            if self._consume_exit_signal(raw_signal, market):
                return

        if self._campaign:
            self._process_campaign_add_on(market)
            # Stage3：Stage2 加仓完成后进一步趋势确认则金字塔追加
            if self._campaign and self._campaign.stage2_filled:
                self._process_campaign_stage3(market)
            return

        if self._pending_signal:
            self._try_open_pilot_position(market)

    def _consume_exit_signal(self, signal: Dict, market: Dict) -> bool:
        action = str(signal.get('action', '') or '').upper()
        positions = self.trade_executor.get_positions(self.inst_id)
        pos = positions.get(self.inst_id)
        if not pos:
            return False
        if action in {'SELL', 'EXIT_LONG'} and getattr(pos.side, 'name', '').upper() == 'LONG':
            self._close_position(f"策略给出多头离场信号：{signal.get('reason', action)}")
            return True
        if action in {'COVER', 'EXIT_SHORT'} and getattr(pos.side, 'name', '').upper() == 'SHORT':
            self._close_position(f"策略给出空头离场信号：{signal.get('reason', action)}")
            return True
        return False

    def _register_pending_signal(self, signal: Dict, market: Dict):
        action = str(signal.get('action', '') or '').upper()
        direction = ""
        if action in {'BUY', 'BUY_LONG', 'LONG'}:
            direction = 'LONG'
        elif action in {'SHORT', 'SELL_SHORT'}:
            direction = 'SHORT'
        else:
            return
        if direction == 'SHORT' and not self._config_bool('allow_short', True):
            self.log_signal.emit("检测到做空信号，但自动交易已关闭做空执行", "WARNING")
            return

        positions = self.trade_executor.get_positions(self.inst_id)
        if self.inst_id in positions:
            return

        price = float(signal.get('entry_price')
                      or market.get('latest_price', 0)
                      or 0)
        reason = str(signal.get('reason', '') or action)
        # 策略已通过的 Gate 列表（策略自行声明，Runner 据此跳过重复检查）
        gates = set(signal.get('strategy_gates') or [])
        tp_override = signal.get('take_profit_pct', signal.get('tp_pct'))
        sl_override = signal.get('stop_loss_pct', signal.get('sl_pct'))
        if self._pending_signal and self._pending_signal.direction == direction:
            self._pending_signal.signal_price = price
            self._pending_signal.reason = reason
            self._pending_signal.created_at = self._now_ts()
            self._pending_signal.tp_pct_override = float(tp_override) if tp_override not in (None, "") else None
            self._pending_signal.sl_pct_override = float(sl_override) if sl_override not in (None, "") else None
            self._pending_signal.raw_signal = dict(signal)
            self._pending_signal.strategy_gates = gates
            return
        if self._pending_signal and self._pending_signal.direction != direction:
            self.log_signal.emit(
                f"方向冲突：当前待执行 {self._pending_signal.direction} 信号将被新 {direction} 信号覆盖",
                "WARNING",
            )
        self._pending_signal = PendingAutoSignal(
            direction=direction,
            signal_price=price,
            reason=reason,
            created_at=self._now_ts(),
            tp_pct_override=float(tp_override) if tp_override not in (None, "") else None,
            sl_pct_override=float(sl_override) if sl_override not in (None, "") else None,
            raw_signal=dict(signal),
            strategy_gates=gates,
        )
        self.log_signal.emit(
            f"捕获 {direction} 开仓信号，进入 3m 回调企稳等待队列：{reason}",
            "TRADE",
        )

    def _try_open_pilot_position(self, market: Dict):
        pending = self._pending_signal
        if not pending:
            return

        # ── 风控门槛 1：连续亏损熔断 + 窗口胜率熔断 ─────────────────────
        # 连续亏损默认 5 次（原 3 次太激进，胜率55%时9%概率连亏3次属正常噪声）
        max_cl = self._config_int('max_consecutive_losses', 5)
        if self._consecutive_losses >= max_cl:
            self.log_signal.emit(
                f"连续亏损熔断（{self._consecutive_losses}次 ≥ {max_cl}），暂停开仓。请手动检查后重启策略。",
                "ERROR",
            )
            self._stop_flag = True
            return
        # 窗口胜率熔断：近 10 笔胜率 < 30% 时熔断（策略可能已失效）
        _win_window = self._config_int('loss_rate_window', 10)
        _min_win_rate = self._config_float('min_recent_win_rate', 30.0) / 100.0
        if len(self._trade_history) >= _win_window:
            _recent = self._trade_history[-_win_window:]
            _cur_rate = sum(_recent) / len(_recent)
            if _cur_rate < _min_win_rate:
                self.log_signal.emit(
                    f"窗口胜率熔断：近 {_win_window} 笔胜率 {_cur_rate*100:.0f}%"
                    f" < 最低要求 {_min_win_rate*100:.0f}%，暂停开仓",
                    "ERROR",
                )
                self._stop_flag = True
                return

        # 策略已通过的 Gate 列表（跳过重复检查，避免双层过滤导致入场过少）
        _sg = pending.strategy_gates

        # ── 风控门槛 2：4H主趋势过滤（主方向） ───────────────────────────
        if 'h4_trend' not in _sg:
            h4_df = market.get('h4_df')
            if not self._h4_trend_allows(h4_df, pending.direction):
                self.log_signal.emit(
                    f"4H主趋势过滤：{pending.direction} 信号与4H主趋势不一致，跳过本次入场",
                    "WARNING",
                )
                return

        # ── 风控门槛 3：日线趋势过滤（顺势交易） ─────────────────────────
        if 'd1_trend' not in _sg:
            d1_df = market.get('d1_df')
            if not self._d1_trend_allows(d1_df, pending.direction):
                self.log_signal.emit(
                    f"日线趋势过滤：{pending.direction} 信号与日线趋势相反，跳过本次入场",
                    "WARNING",
                )
                return

        # ── 风控门槛 4：H1 趋势延续 ────────────────────────────────────────
        if 'h1_trend' not in _sg:
            if not self._hourly_trend_continues(market['h1_df'], pending.direction):
                self.log_signal.emit("H1 趋势已不再延续，取消本次待开仓信号", "WARNING")
                self._pending_signal = None
                return

        # ── 硬性原则：必须出现3m明显回调且不跌破原突破点，同时H1原趋势继续 ───
        entry_guard: Dict[str, Any] = {}
        if 'entry_rule' not in _sg:
            entry_guard = evaluate_entry_rule_from_frames(
                market['m3_df'],
                market['h1_df'],
                pending.direction,
                config=self.config,
            )
            if not entry_guard.get('ok'):
                self.log_signal.emit(
                    f"硬性开仓原则未满足：{entry_guard.get('reason', '3m/H1检查失败')}，继续等待",
                    "INFO",
                )
                return

        # ── 风控门槛 5：RSI 极值过滤（避免追高杀低） ──────────────────────
        if 'rsi' not in _sg:
            if not self._rsi_not_extreme(market['h1_df'], pending.direction):
                self.log_signal.emit(
                    f"RSI 极值过滤：H1 RSI 已过热/过冷，{pending.direction} 入场条件不满足，等待回归",
                    "WARNING",
                )
                return

        # ── 风控门槛 6：3m 回调企稳 ────────────────────────────────────────
        pullback = {}
        if 'm3_pullback' not in _sg:
            pullback = self._detect_pullback_stabilization(market['m3_df'], pending.direction)
            if not pullback.get('ready'):
                return

        # ── 风控门槛 7：成交量确认（避免阴跌/无量反弹） ────────────────────
        if 'volume' not in _sg:
            if not self._volume_confirms(market['m3_df'], pending.direction):
                self.log_signal.emit("成交量过低，不满足入场条件，继续等待", "WARNING")
                return

        # ── 风控门槛 8：4H ADX 趋势强度 ─────────────────────────────────────
        # ADX < 20 表示当前处于震荡区间，趋势跟随策略在此环境下胜率显著下降。
        # 不硬性拦截（扫描信号已通过严格过滤），但仅在 ADX 过低时减少仓位。
        h4_adx = float(market.get('h4_adx', 0) or 0)
        _min_adx = self._config_float('h4_min_adx', 20.0)
        if h4_adx > 0 and h4_adx < _min_adx:
            self.log_signal.emit(
                f"4H ADX={h4_adx:.1f} < {_min_adx:.0f}（震荡区间），降低仓位至 50% 入场",
                "WARNING",
            )
            # 注入仓位缩减标志，_resolve_entry_amount 或上层可读取
            self.config['_adx_scale_factor'] = 0.5
        else:
            self.config.pop('_adx_scale_factor', None)

        # ── 风控门槛 9：H1 MACD Histogram 方向确认 ──────────────────────────
        # MACD 柱（histogram）方向与交易方向一致 → 短期动能支持入场。
        # 反向时不拦截，但记录警告，供 Stage2 加仓时参考。
        h1_macd_hist = float(market.get('h1_macd_hist', 0) or 0)
        if h1_macd_hist != 0:
            if pending.direction == 'LONG' and h1_macd_hist < 0:
                self.log_signal.emit(
                    f"H1 MACD Histogram={h1_macd_hist:.4f}（空头动能），多头入场谨慎",
                    "WARNING",
                )
                # 若 ADX 同时也低，叠加两个弱信号则放弃本次入场
                if h4_adx > 0 and h4_adx < _min_adx:
                    self.log_signal.emit(
                        "H1 MACD 反向 + 4H ADX 不足，双重弱信号，取消本次入场", "WARNING"
                    )
                    self.config.pop('_adx_scale_factor', None)
                    return
            elif pending.direction == 'SHORT' and h1_macd_hist > 0:
                self.log_signal.emit(
                    f"H1 MACD Histogram={h1_macd_hist:.4f}（多头动能），空头入场谨慎",
                    "WARNING",
                )
                if h4_adx > 0 and h4_adx < _min_adx:
                    self.log_signal.emit(
                        "H1 MACD 反向 + 4H ADX 不足，双重弱信号，取消本次入场", "WARNING"
                    )
                    self.config.pop('_adx_scale_factor', None)
                    return

        # ── 注册器检查：防止与其他系统产生持仓冲突 ─────────────────────────
        _sys = self.config.get('_system_name', 'StrategyRunner')
        if not position_registry.try_lock(self.inst_id, _sys):
            _owner = position_registry.get_owner(self.inst_id)
            self.log_signal.emit(
                f"[注册器] {self.inst_id} 已由 {_owner} 持有，"
                f"本次 {_sys} 入场被拒绝，等待下一轮",
                "WARNING",
            )
            return

        amount = self._resolve_entry_amount(stage='pilot')
        if amount <= 0:
            position_registry.release(self.inst_id, _sys)
            self.log_signal.emit("自动交易资金池不足，无法执行 1% 试仓", "ERROR")
            return
        result = self._open_position(
            direction=pending.direction,
            usdt_amount=amount,
            reason=f"1%试仓 | {pending.reason} | {entry_guard.get('reason', pullback.get('reason', ''))}",
            tp_pct_override=pending.tp_pct_override,
            sl_pct_override=pending.sl_pct_override,
        )
        if not result.success:
            position_registry.release(self.inst_id, _sys)
            self.log_signal.emit(f"1%试仓失败：{result.message}", "ERROR")
            return

        latest_price = market['latest_price']
        # 传入 m3_df，使用 ATR 动态止损线替代原来的 0.1% 固定缓冲
        cost_line = self._compute_cost_line(latest_price, pending.direction, market.get('m3_df'))
        # 存储实际 ATR 缓冲百分比供阶段2复用（修复：之前用硬编码 2%）
        atr_buffer_pct = abs(cost_line - latest_price) / latest_price * 100 if latest_price > 0 else 2.0
        self._campaign = AutoTradeCampaign(
            direction=pending.direction,
            first_signal_price=pending.signal_price,
            stage1_entry_price=latest_price,
            stage1_cost_line=cost_line,
            highest_since_stage1=latest_price,
            lowest_since_stage1=latest_price,
            total_allocated_usdt=amount,
            opened_at=self._now_ts(),
            last_reason=pending.reason,
            peak_profit_price=latest_price,
            tp_pct_override=pending.tp_pct_override,
            sl_pct_override=pending.sl_pct_override,
        )
        self._campaign.stage1_atr_buffer_pct = atr_buffer_pct
        self._position_opened_at = self._now_ts()
        self._pending_signal = None
        self.log_signal.emit(
            f"自动交易 1%试仓成功：{pending.direction} @ {latest_price:.6f}，"
            f"ATR止损线={cost_line:.6f}（距开仓价"
            f" {abs(cost_line - latest_price) / latest_price * 100:.2f}%），"
            f"后续只在第二次 3m 回调企稳且不跌破止损线时允许加仓。",
            "SUCCESS",
        )

    def _process_campaign_add_on(self, market: Dict):
        if not self._campaign:
            return
        campaign = self._campaign
        price = market['latest_price']
        if campaign.direction == 'LONG':
            campaign.highest_since_stage1 = max(campaign.highest_since_stage1 or price, price)
            campaign.lowest_since_stage1 = min(campaign.lowest_since_stage1 or price, price)
        else:
            campaign.highest_since_stage1 = max(campaign.highest_since_stage1 or price, price)
            campaign.lowest_since_stage1 = min(campaign.lowest_since_stage1 or price, price)

        if campaign.stage2_filled:
            return

        if not campaign.stage2_armed:
            if self._detect_stage1_continuation(market['m3_df'], campaign):
                # ── 试仓亏损保护：若试仓曾显著亏损，禁止二次加仓 ──────
                if self._trial_was_in_loss(campaign):
                    self.log_signal.emit(
                        "试仓曾处于亏损状态（最大回撤超过阈值），放弃本次二次加仓，保留试仓继续观察",
                        "WARNING",
                    )
                    return
                campaign.stage2_armed = True
                self.log_signal.emit("3m 首段延续已确认，开始等待第二次回调企稳后再加仓 10%", "INFO")
            return

        second_pullback = self._detect_second_pullback(market['m3_df'], campaign)
        if second_pullback.get('breach_entry_line'):
            self._close_position("第二次 3m 回调跌破首笔开仓风险线，强制离场")
            return
        if not second_pullback.get('ready'):
            return
        if not self._hourly_trend_continues(market['h1_df'], campaign.direction):
            campaign.stage2_armed = False
            self.log_signal.emit("第二次加仓前检查发现 H1 趋势暂不延续，放弃本次加仓但保留试仓继续观察", "WARNING")
            return

        amount = self._resolve_entry_amount(stage='add')
        if amount <= 0:
            self.log_signal.emit("自动交易资金池不足，无法执行 10% 加仓", "ERROR")
            return
        result = self._open_position(
            direction=campaign.direction,
            usdt_amount=amount,
            reason=f"10%加仓 | 第二次3m回调企稳 | {campaign.last_reason}",
            tp_pct_override=campaign.tp_pct_override,
            sl_pct_override=campaign.sl_pct_override,
        )
        if not result.success:
            self.log_signal.emit(f"10%加仓失败：{result.message}", "ERROR")
            return

        campaign.stage2_filled = True
        campaign.total_allocated_usdt += amount
        campaign.stage2_entry_price = market['latest_price']
        # 修复：不重置 peak_profit_price，保留阶段1已跟踪的最高价
        # campaign.peak_profit_price = market['latest_price']  # ← 移除：否则丢失阶段1浮盈

        # ── Stage2 后用锁定 ATR 缓冲区刷新止损线（trailing only）────────────
        # 使用 Stage1 时刻锁定的 ATR 缓冲百分比，防止 3m ATR 在加仓后膨胀
        # 导致止损线不合理放大。同时只允许止损线向有利方向移动（trailing）。
        try:
            new_pos = self.trade_executor.get_positions(self.inst_id).get(self.inst_id)
            new_avg_px = float(getattr(new_pos, 'entry_price', 0) or 0)
            if new_avg_px > 0:
                # ── BUGFIX: stage1_atr_buffer_pct 以百分比存储（如 2.0 代表 2%），
                # 此处需除以 100 转换为小数；回退值同样用百分比再除以 100
                locked_pct = (getattr(campaign, 'stage1_atr_buffer_pct', 0.0) or 2.0) / 100.0
                direction = str(campaign.direction).upper()
                if direction == 'LONG':
                    new_cost_line = new_avg_px * (1.0 - locked_pct)
                    # Trailing only：止损线只能上移（收紧），不允许下移（放大）
                    if new_cost_line <= campaign.stage1_cost_line:
                        new_cost_line = campaign.stage1_cost_line
                else:
                    new_cost_line = new_avg_px * (1.0 + locked_pct)
                    # Trailing only：止损线只能下移（收紧），不允许上移（放大）
                    if new_cost_line >= campaign.stage1_cost_line:
                        new_cost_line = campaign.stage1_cost_line

                campaign.stage1_cost_line = new_cost_line
                self.log_signal.emit(
                    f"Stage2加仓后止损线已更新：{new_cost_line:.6f}"
                    f"（新均价 {new_avg_px:.6f}，锁定ATR缓冲 {locked_pct*100:.2f}%，"
                    f"距均价 {abs(new_cost_line - new_avg_px) / new_avg_px * 100:.2f}%）",
                    "INFO",
                )
        except Exception:
            pass

        self.log_signal.emit(
            f"自动交易 10%加仓成功：{campaign.direction} @ {market['latest_price']:.6f}，"
            f"已启动 ATR 自适应移动止盈，后续按第一原则 + H1反转 + 移动止盈管理离场。"
            f"若趋势继续延伸将触发 Stage3 金字塔加仓。",
            "SUCCESS",
        )

    def _process_campaign_stage3(self, market: Dict):
        """
        Stage3 金字塔加仓（Stage2 完成后趋势强劲延续时执行）。

        触发条件（全部满足）：
          1. stage2_filled=True，stage3_filled=False
          2. 当前价相对 Stage2 开仓价再上涨 ≥ stage3_trigger_pct（默认 2%）
          3. H1 MACD Histogram 方向与交易方向一致（短期动能支持）
          4. 4H ADX > h4_min_adx（默认 20，确认趋势仍在延续而非反转）
          5. 资金池有足够余额（stage3_position_pct，默认 15%）

        效果：在强趋势中追加 15% 保证金，最终仓位 2%+10%+15%=27%；
               同时将 ATR 止损线 trailing 上移至 Stage3 开仓价基础。
        """
        if not self._campaign:
            return
        campaign = self._campaign
        if not campaign.stage2_filled or campaign.stage3_filled:
            return

        price     = market['latest_price']
        s2_entry  = campaign.stage2_entry_price
        direction = str(campaign.direction).upper()

        if s2_entry <= 0:
            return

        # 条件1：价格相对 Stage2 开仓价继续上涨 / 下跌（方向性）
        trigger_pct = self._config_float('stage3_trigger_pct', 2.0) / 100.0
        if direction == 'LONG':
            gain_since_s2 = (price - s2_entry) / s2_entry
        else:
            gain_since_s2 = (s2_entry - price) / s2_entry

        if gain_since_s2 < trigger_pct:
            return

        # 条件2：H1 MACD Histogram 方向确认
        h1_macd_hist = float(market.get('h1_macd_hist', 0) or 0)
        if h1_macd_hist != 0:
            if direction == 'LONG' and h1_macd_hist < 0:
                self.log_signal.emit(
                    f"Stage3待触发：价格涨幅 {gain_since_s2*100:.2f}% 已达阈值，"
                    f"但 H1 MACD Histogram 反向（{h1_macd_hist:.4f}），等待确认",
                    "INFO",
                )
                return
            if direction == 'SHORT' and h1_macd_hist > 0:
                self.log_signal.emit(
                    f"Stage3待触发：价格跌幅 {gain_since_s2*100:.2f}% 已达阈值，"
                    f"但 H1 MACD Histogram 反向（{h1_macd_hist:.4f}），等待确认",
                    "INFO",
                )
                return

        # 条件3：4H ADX 确认趋势强度 > 20
        h4_adx = float(market.get('h4_adx', 0) or 0)
        _min_adx = self._config_float('h4_min_adx', 20.0)
        if h4_adx > 0 and h4_adx < _min_adx:
            self.log_signal.emit(
                f"Stage3待触发：4H ADX={h4_adx:.1f} < {_min_adx:.0f}，趋势强度不足，暂缓加仓",
                "INFO",
            )
            return

        # 条件满足，执行 Stage3 加仓
        amount = self._resolve_entry_amount(stage='stage3')
        if amount <= 0:
            self.log_signal.emit("Stage3加仓：资金池不足，跳过", "WARNING")
            return

        result = self._open_position(
            direction=direction,
            usdt_amount=amount,
            reason=f"Stage3金字塔加仓 | 趋势延伸{gain_since_s2*100:.1f}%"
                   f" | ADX={h4_adx:.1f} | MACD确认 | {campaign.last_reason}",
        )
        if not result.success:
            self.log_signal.emit(f"Stage3加仓失败：{result.message}", "ERROR")
            return

        campaign.stage3_filled = True
        campaign.stage3_entry_price = price
        campaign.total_allocated_usdt += amount

        # 更新止损线（trailing only，向有利方向收紧）
        try:
            locked_pct = (getattr(campaign, 'stage1_atr_buffer_pct', 0.0) or 2.0) / 100.0
            if direction == 'LONG':
                new_cost = price * (1.0 - locked_pct)
                if new_cost > campaign.stage1_cost_line:
                    campaign.stage1_cost_line = new_cost
            else:
                new_cost = price * (1.0 + locked_pct)
                if new_cost < campaign.stage1_cost_line:
                    campaign.stage1_cost_line = new_cost
        except Exception:
            pass

        self.log_signal.emit(
            f"🚀 Stage3金字塔加仓成功：{direction} @ {price:.6f}，"
            f"追加 {amount:.2f} USDT（总保证金 {campaign.total_allocated_usdt:.2f} USDT），"
            f"ADX={h4_adx:.1f} 趋势确认，止损线已收紧至 {campaign.stage1_cost_line:.6f}",
            "SUCCESS",
        )

    def _build_market_snapshot(self, klines: Dict) -> Dict[str, Any]:
        m3_df = self._rows_to_df(klines.get('m3', []))
        h1_df = self._rows_to_df(klines.get('hourly', []))
        h4_df = self._rows_to_df(klines.get('4h', []))
        d1_df = self._rows_to_df(klines.get('daily', []))
        if m3_df.empty or h1_df.empty or h4_df.empty or len(m3_df) < 30 or len(h1_df) < 60 or len(h4_df) < 40:
            return {'valid': False, 'reason': '3m/H1/4H 数据不足'}
        latest_price = float(m3_df['close'].iloc[-1])

        # ── 注入 ATR% 供追踪止损自适应（基于 H4 ATR）────────────────────────
        atr_pct = 0.0
        try:
            if len(h4_df) >= 15:
                h4_atr = float(self._atr(h4_df, 14).iloc[-1])
                h4_close = float(h4_df['close'].iloc[-1])
                if h4_close > 0 and h4_atr > 0:
                    atr_pct = h4_atr / h4_close * 100.0
        except Exception:
            pass

        # ── 注入 H1 MACD Histogram 最新值 ──────────────────────────────────
        h1_macd_hist = 0.0
        try:
            if len(h1_df) >= 35:
                h1_macd_hist = float(self._macd_hist(h1_df['close']).iloc[-1])
        except Exception:
            pass

        # ── 注入 4H ADX（趋势强度）──────────────────────────────────────────
        h4_adx = 0.0
        try:
            if len(h4_df) >= 30:
                h4_adx = self._adx(h4_df, 14)
        except Exception:
            pass

        return {
            'valid': True,
            'm3_df': m3_df,
            'h1_df': h1_df,
            'h4_df': h4_df,
            'd1_df': d1_df,
            'latest_price': latest_price,
            'atr_pct': atr_pct,       # H4 ATR% — 供追踪止损自适应
            'h1_macd_hist': h1_macd_hist,  # H1 MACD Histogram 方向确认
            'h4_adx': h4_adx,         # 4H ADX 趋势强度
        }

    def _rows_to_df(self, rows: List) -> pd.DataFrame:
        if not rows:
            return pd.DataFrame()
        try:
            # OKX V5 K线返回列数可能变化（通常9列），动态适配而非硬编码
            _EXPECTED_COLS = ['ts', 'open', 'high', 'low', 'close', 'volume']
            _EXTRA_COLS = [f'_x{i}' for i in range(20)]  # 预留扩展列
            first_row_len = len(rows[0]) if rows else 0
            if first_row_len <= len(_EXPECTED_COLS):
                col_names = _EXPECTED_COLS[:first_row_len]
            else:
                col_names = _EXPECTED_COLS + _EXTRA_COLS[:first_row_len - len(_EXPECTED_COLS)]
            df = pd.DataFrame(rows, columns=col_names)
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            df['ts'] = pd.to_numeric(df['ts'], errors='coerce')
            df = df.dropna(subset=['ts', 'open', 'high', 'low', 'close']).copy()
            df['timestamp'] = pd.to_datetime(df['ts'].astype('int64'), unit='ms')
            df = df.sort_values('timestamp').drop_duplicates(subset='timestamp', keep='last').reset_index(drop=True)
            return df
        except Exception:
            return pd.DataFrame()

    def _hourly_trend_continues(self, h1_df: pd.DataFrame, direction: str) -> bool:
        if h1_df is None or h1_df.empty or len(h1_df) < 30:
            return False
        close = h1_df['close']
        ema_fast = close.ewm(span=self._config_int('h1_fast_ema', 12), adjust=False).mean()
        ema_slow = close.ewm(span=self._config_int('h1_slow_ema', 26), adjust=False).mean()
        fast = float(ema_fast.iloc[-1])
        slow = float(ema_slow.iloc[-1])
        # 使用10根K线的相对斜率（比4根更抗噪），要求斜率绝对值 > 0.02%
        if len(ema_fast) >= 10:
            base = float(ema_fast.iloc[-10]) or 1.0
            slope_pct = (float(ema_fast.iloc[-1]) - base) / base
        else:
            slope_pct = 0.0
        slope_threshold = 0.0002   # 0.02% 最小斜率要求
        last_close = float(close.iloc[-1])
        if direction == 'LONG':
            return fast > slow and slope_pct > slope_threshold and last_close >= slow
        return fast < slow and slope_pct < -slope_threshold and last_close <= slow

    def _detect_pullback_stabilization(self, m3_df: pd.DataFrame, direction: str) -> Dict[str, Any]:
        if m3_df is None or m3_df.empty or len(m3_df) < 24:
            return {'ready': False, 'reason': '3m数据不足'}
        df = m3_df.tail(48).reset_index(drop=True)
        closes = df['close']
        lows = df['low']
        highs = df['high']
        ema_fast = closes.ewm(span=self._config_int('m3_fast_ema', 8), adjust=False).mean()
        ema_mid = closes.ewm(span=self._config_int('m3_mid_ema', 13), adjust=False).mean()
        stab_bars = self._config_int('m3_stabilization_bars', 3)
        min_pb = self._config_float('m3_pullback_min_pct', 0.35)
        max_pb = self._config_float('m3_pullback_max_pct', 2.50)

        if direction == 'LONG':
            peak_idx_raw = closes.iloc[:-stab_bars].idxmax()
            if pd.isna(peak_idx_raw):
                return {'ready': False, 'reason': 'price data unavailable'}
            peak_idx = int(peak_idx_raw)
            peak = float(highs.iloc[peak_idx])
            pb_slice = df.iloc[peak_idx + 1:]
            if pb_slice.empty:
                return {'ready': False, 'reason': '3m 尚未出现回调'}
            trough = float(pb_slice['low'].min())
            trough_idx = int(pb_slice['low'].idxmin())
            pullback_pct = (peak - trough) / peak * 100 if peak > 0 else 0.0
            stabilized = (
                trough_idx <= len(df) - stab_bars
                and bool((closes.tail(stab_bars) >= ema_fast.tail(stab_bars)).all())
                and bool((closes.tail(stab_bars) >= ema_mid.tail(stab_bars)).all())
                and float(closes.iloc[-1]) >= float(closes.iloc[-stab_bars])
            )
        else:
            peak_idx = int(closes.iloc[:-stab_bars].idxmin())
            peak = float(lows.iloc[peak_idx])
            pb_slice = df.iloc[peak_idx + 1:]
            if pb_slice.empty:
                return {'ready': False, 'reason': '3m 尚未出现回调'}
            trough = float(pb_slice['high'].max())
            trough_idx = int(pb_slice['high'].idxmax())
            pullback_pct = (trough - peak) / peak * 100 if peak > 0 else 0.0
            stabilized = (
                trough_idx <= len(df) - stab_bars
                and bool((closes.tail(stab_bars) <= ema_fast.tail(stab_bars)).all())
                and bool((closes.tail(stab_bars) <= ema_mid.tail(stab_bars)).all())
                and float(closes.iloc[-1]) <= float(closes.iloc[-stab_bars])
            )

        ready = min_pb <= pullback_pct <= max_pb and stabilized
        reason = f"3m回调{pullback_pct:.2f}% + {'企稳完成' if stabilized else '等待企稳'}"
        return {'ready': ready, 'pullback_pct': pullback_pct, 'reason': reason}

    def _detect_stage1_continuation(self, m3_df: pd.DataFrame, campaign: AutoTradeCampaign) -> bool:
        if m3_df is None or m3_df.empty or len(m3_df) < 10:
            return False
        closes = m3_df['close'].tail(8)
        latest = float(closes.iloc[-1])
        breakout_buffer = self._config_float('m3_breakout_buffer_pct', 0.15) / 100.0
        if campaign.direction == 'LONG':
            return latest > max(float(closes.iloc[:-1].max()), campaign.stage1_entry_price * (1.0 + breakout_buffer))
        return latest < min(float(closes.iloc[:-1].min()), campaign.stage1_entry_price * (1.0 - breakout_buffer))

    def _trial_was_in_loss(self, campaign: AutoTradeCampaign) -> bool:
        """试仓是否曾处于显著亏损：防止亏损试仓后仍触发二次加仓"""
        loss_pct = self._config_float('trial_loss_block_pct', 0.30) / 100.0
        if campaign.direction == 'LONG':
            if campaign.lowest_since_stage1 and campaign.lowest_since_stage1 > 0:
                trial_drawdown = (campaign.stage1_entry_price - campaign.lowest_since_stage1) / campaign.stage1_entry_price
                return trial_drawdown > loss_pct
            return False
        else:
            if campaign.highest_since_stage1 and campaign.highest_since_stage1 > 0:
                trial_drawdown = (campaign.highest_since_stage1 - campaign.stage1_entry_price) / campaign.stage1_entry_price
                return trial_drawdown > loss_pct
            return False

    def _detect_second_pullback(self, m3_df: pd.DataFrame, campaign: AutoTradeCampaign) -> Dict[str, Any]:
        if m3_df is None or m3_df.empty or len(m3_df) < 20:
            return {'ready': False, 'breach_entry_line': False}
        df = m3_df.tail(36).reset_index(drop=True)
        closes = df['close']
        lows = df['low']
        highs = df['high']
        ema_fast = closes.ewm(span=self._config_int('m3_fast_ema', 8), adjust=False).mean()
        ema_mid = closes.ewm(span=self._config_int('m3_mid_ema', 13), adjust=False).mean()
        stab_bars = self._config_int('m3_stabilization_bars', 3)
        min_pb = self._config_float('m3_pullback_min_pct', 0.35)
        max_pb = self._config_float('m3_pullback_max_pct', 2.50)

        if campaign.direction == 'LONG':
            reference = max(campaign.highest_since_stage1, float(highs.max()))
            pullback_low = float(lows.min())
            pullback_pct = (reference - pullback_low) / reference * 100 if reference > 0 else 0.0
            breach = bool((closes.tail(2) <= campaign.stage1_cost_line).all())
            stabilized = bool((closes.tail(stab_bars) >= ema_fast.tail(stab_bars)).all()) and bool((closes.tail(stab_bars) >= ema_mid.tail(stab_bars)).all())
        else:
            reference = min(campaign.lowest_since_stage1, float(lows.min()))
            pullback_high = float(highs.max())
            pullback_pct = (pullback_high - reference) / reference * 100 if reference > 0 else 0.0
            breach = bool((closes.tail(2) >= campaign.stage1_cost_line).all())
            stabilized = bool((closes.tail(stab_bars) <= ema_fast.tail(stab_bars)).all()) and bool((closes.tail(stab_bars) <= ema_mid.tail(stab_bars)).all())

        ready = min_pb <= pullback_pct <= max_pb and stabilized and not breach
        return {'ready': ready, 'breach_entry_line': breach}

    def _enforce_first_principle(self, market: Dict[str, Any]) -> bool:
        """
        第一原则：价格触及建仓时确定的 ATR 止损线则立即离场。
        """
        if not self._campaign:
            return False
        positions = self.trade_executor.get_positions(self.inst_id)
        pos = positions.get(self.inst_id)
        if not pos:
            self._on_position_gone()
            self._campaign = None
            return False

        side       = getattr(pos.side, 'name', str(pos.side)).upper()
        latest     = float(market.get('latest_price',
                            getattr(pos, 'current_price', 0) or 0))
        cost_line  = self._campaign.stage1_cost_line

        # 止损线缺失时从入场价重新计算
        if not cost_line or cost_line <= 0:
            entry = float(getattr(pos, 'entry_price', 0) or 0)
            if entry > 0:
                cost_line = self._compute_cost_line(
                    entry, side, market.get('m3_df'))
                self._campaign.stage1_cost_line = cost_line
            else:
                return False

        if side == 'LONG' and latest <= cost_line:
            self._close_position(
                f"第一原则触发：当前价 {latest:.6f} 跌破止损线 {cost_line:.6f}"
            )
            return True
        if side == 'SHORT' and latest >= cost_line:
            self._close_position(
                f"第一原则触发：当前价 {latest:.6f} 上破止损线 {cost_line:.6f}"
            )
            return True
        return False

    def _check_hourly_reversal_exit(self, market: Dict[str, Any]) -> bool:
        positions = self.trade_executor.get_positions(self.inst_id)
        pos = positions.get(self.inst_id)
        if not pos:
            return False
        # 最小持有时间检查：优先取 campaign.opened_at，退而取 runner 记录的独立开仓时间
        min_hold_hours = self._config_float('h1_reversal_min_hold_hours', 6.0)
        if self._campaign and self._campaign.opened_at:
            hold_secs = self._now_ts() - self._campaign.opened_at
        elif self._position_opened_at > 0:
            hold_secs = self._now_ts() - self._position_opened_at
        else:
            hold_secs = float('inf')
        if hold_secs < min_hold_hours * 3600:
            return False
        direction = getattr(pos.side, 'name', str(pos.side)).upper()
        reason = self._detect_hourly_reversal(market['h1_df'], direction)
        if not reason:
            return False
        self._close_position(f"H1 趋势转折离场：{reason}")
        return True

    def _detect_hourly_reversal(self, h1_df: pd.DataFrame, direction: str) -> Optional[str]:
        if h1_df is None or h1_df.empty or len(h1_df) < 40:
            return None
        df = h1_df.tail(64).reset_index(drop=True)
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume'] if 'volume' in df.columns else None
        ema26 = close.ewm(span=26, adjust=False).mean()
        last_close = float(close.iloc[-1])
        threshold = self._config_float('h1_large_pullback_pct', 5.00)
        rsi = self._rsi(close, 14)
        macd_hist = self._macd_hist(close)

        if direction == 'LONG':
            recent_peak = float(high.tail(24).max())
            pullback_pct = (recent_peak - last_close) / recent_peak * 100 if recent_peak > 0 else 0.0
            if pullback_pct >= threshold and last_close < float(ema26.iloc[-1]):
                return f"小时线大幅回调 {pullback_pct:.2f}%"
            if last_close < float(ema26.iloc[-1]) * 0.999 and self._detect_head_shoulders_top(close, volume):
                return "头肩顶"
            if last_close < float(ema26.iloc[-1]) * 0.999 and float(rsi.iloc[-1]) < 48 and self._detect_rounding_top(close):
                return "圆弧顶"
            if last_close < float(ema26.iloc[-1]) * 0.999 and float(rsi.iloc[-1]) < 50 and self._detect_bearish_divergence(close, rsi, macd_hist):
                return "顶背离"
        else:
            recent_low = float(low.tail(24).min())
            rebound_pct = (last_close - recent_low) / recent_low * 100 if recent_low > 0 else 0.0
            if rebound_pct >= threshold and last_close > float(ema26.iloc[-1]):
                return f"小时线大幅反抽 {rebound_pct:.2f}%"
            if last_close > float(ema26.iloc[-1]) * 1.001 and float(rsi.iloc[-1]) > 52 and self._detect_rounding_bottom(close):
                return "圆弧底反转"
            if last_close > float(ema26.iloc[-1]) * 1.001 and float(rsi.iloc[-1]) > 50 and self._detect_bullish_divergence(close, rsi, macd_hist):
                return "底背离"
        return None

    def _detect_head_shoulders_top(self, close: pd.Series, volume: Optional[pd.Series] = None) -> bool:
        """
        头肩顶检测（增强版）：
        - 自适应窗口（15~21 根 K 线扫描）
        - 要求头部高于两肩至少 0.5%
        - 跌破颈线确认
        - 成交量递减验证（头部成交量 < 左肩成交量，增加可靠性）
        """
        for window in (15, 18, 21):
            values = close.tail(window).tolist()
            if len(values) < window:
                continue
            # 自适应三段划分
            seg = window // 3
            left = max(values[1:seg])
            head = max(values[seg:seg*2])
            right = max(values[seg*2:window-1])
            neckline = min(values[seg-1], values[seg*2-1])

            if not (head > left > neckline and head > right > neckline):
                continue
            # 头部须高于两肩至少 0.8%（防止横盘噪音误判）
            min_prominence = 0.008
            if head <= 0 or (head - left) / head < min_prominence or (head - right) / head < min_prominence:
                continue
            # 跌破颈线
            if values[-1] >= neckline:
                continue
            # 成交量递减确认（可选，无 volume 数据时跳过此检查）
            if volume is not None and len(volume) >= window:
                vol_values = volume.tail(window).tolist()
                vol_left = sum(vol_values[1:seg]) / max(seg - 1, 1)
                vol_head = sum(vol_values[seg:seg*2]) / max(seg, 1)
                # 头部成交量应 ≤ 左肩成交量（经典量价背离）
                if vol_head > vol_left * 1.2:
                    continue  # 头部放量不符合经典头肩顶特征
            return True
        return False

    def _detect_rounding_top(self, close: pd.Series) -> bool:
        values = close.tail(8).tolist()
        if len(values) < 8:
            return False
        mid = values[3]
        return values[0] < values[1] < values[2] < mid and values[4] < mid and values[5] < values[4] and values[6] < values[5] and values[7] < values[6]

    def _detect_rounding_bottom(self, close: pd.Series) -> bool:
        values = close.tail(8).tolist()
        if len(values) < 8:
            return False
        mid = values[3]
        return values[0] > values[1] > values[2] > mid and values[4] > mid and values[5] > values[4] and values[6] > values[5] and values[7] > values[6]

    def _detect_bearish_divergence(self, close: pd.Series, rsi: pd.Series, macd_hist: pd.Series) -> bool:
        if len(close) < 24:
            return False
        c1 = float(close.iloc[-18:-9].max())
        c2 = float(close.iloc[-9:].max())
        r1 = float(rsi.iloc[-18:-9].max())
        r2 = float(rsi.iloc[-9:].max())
        m1 = float(macd_hist.iloc[-18:-9].max())
        m2 = float(macd_hist.iloc[-9:].max())
        # 要求价格新高幅度 > 0.3% 且 RSI+MACD 同时背离（更严格）
        price_new_high = c2 > c1 * 1.003
        return price_new_high and r2 < r1 and m2 < m1

    def _detect_bullish_divergence(self, close: pd.Series, rsi: pd.Series, macd_hist: pd.Series) -> bool:
        if len(close) < 24:
            return False
        c1 = float(close.iloc[-18:-9].min())
        c2 = float(close.iloc[-9:].min())
        r1 = float(rsi.iloc[-18:-9].min())
        r2 = float(rsi.iloc[-9:].min())
        m1 = float(macd_hist.iloc[-18:-9].min())
        m2 = float(macd_hist.iloc[-9:].min())
        price_new_low = c2 < c1 * 0.997
        return price_new_low and r2 > r1 and m2 > m1

    def _rsi(self, close: pd.Series, period: int = 14) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / loss.replace(0, pd.NA)
        return (100 - (100 / (1 + rs))).bfill().fillna(50.0)

    def _macd_hist(self, close: pd.Series) -> pd.Series:
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        return macd - signal

    def _open_position(
        self,
        direction: str,
        usdt_amount: float,
        reason: str,
        tp_pct_override: Optional[float] = None,
        sl_pct_override: Optional[float] = None,
    ):
        latest_price = 0.0
        ticker = self.okx_client.get_ticker(self.inst_id)
        if ticker and 'data' in ticker and ticker['data']:
            latest_price = float(ticker['data'][0].get('last', 0) or 0)
        self.trade_signal.emit("BUY" if direction == "LONG" else "SHORT", self.inst_id, latest_price, usdt_amount)
        self.log_signal.emit(reason, "TRADE")
        tp_pct = (
            float(tp_pct_override)
            if tp_pct_override not in (None, "")
            else self._config_float('take_profit_pct', 5.0)
        ) / 100.0
        sl_pct = (
            float(sl_pct_override)
            if sl_pct_override not in (None, "")
            else self._config_float('stop_loss_pct', 3.0)
        ) / 100.0
        return self.trade_executor.execute_entry(
            self.inst_id,
            direction,
            usdt_amount=usdt_amount,
            leverage=self._config_int('leverage', 3),
            tp_pct=tp_pct,
            sl_pct=sl_pct,
            order_type="market" if self._config_bool('prefer_market_order', True) else "limit",
            price=None,
        )

    def _close_position(self, reason: str):
        positions = self.trade_executor.get_positions(self.inst_id)
        pos = positions.get(self.inst_id)
        if not pos:
            self._campaign = None
            self._position_opened_at = 0.0
            self._close_retry_count = 0
            # 平仓时即使持仓不存在也释放注册器（防止残留锁）
            _sys = self.config.get('_system_name', 'StrategyRunner')
            position_registry.release(self.inst_id, _sys)
            return
        side_name = getattr(pos.side, 'name', str(pos.side)).upper()
        cur_price = float(pos.current_price or 0)
        self.trade_signal.emit("SELL" if side_name == "LONG" else "COVER", self.inst_id, cur_price, float(pos.size or 0))
        self.log_signal.emit(reason, "WARNING")
        if side_name == "LONG":
            result = self.trade_executor.execute_sell(self.inst_id, pos.size)
        else:
            result = self.trade_executor.execute_cover(self.inst_id, pos.size)
        if result.success:
            realized_pnl = self._compute_realized_pnl(pos, result)
            self._close_retry_count = 0
            # 记录交易结果到历史窗口（保留最近 20 笔）
            _win = realized_pnl >= 0
            self._trade_history.append(_win)
            if len(self._trade_history) > 20:
                self._trade_history.pop(0)

            if _win:
                self._consecutive_losses = 0
                self.log_signal.emit(f"自动交易离场成功（盈利 {realized_pnl:+.2f} USDT）", "SUCCESS")
            else:
                self._consecutive_losses += 1
                _recent = self._trade_history[-10:] if len(self._trade_history) >= 10 else self._trade_history
                _win_rate = sum(_recent) / len(_recent) * 100 if _recent else 50
                self.log_signal.emit(
                    f"自动交易离场（亏损 {realized_pnl:+.2f} USDT，"
                    f"连续亏损 {self._consecutive_losses} 次，"
                    f"近期{len(_recent)}笔胜率 {_win_rate:.0f}%）",
                    "WARNING",
                )
            self._on_close_success(reason, pos, result, realized_pnl)
            self._campaign = None
            self._position_opened_at = 0.0
            self._pending_signal = None
            # 平仓完成后释放注册器，允许其他系统接管该标的
            _sys = self.config.get('_system_name', 'StrategyRunner')
            position_registry.release(self.inst_id, _sys)
        else:
            self._close_retry_count += 1
            self.log_signal.emit(f"自动交易离场失败：{result.message}", "ERROR")
            self._on_close_failure(reason, pos, result, self._close_retry_count)
            max_retries = self._config_int('max_close_retries', 5)
            if self._close_retry_count >= max_retries:
                self.log_signal.emit(f"平仓 {self._close_retry_count} 次失败，释放注册器以防死锁", "ERROR")
                _sys = self.config.get('_system_name', 'StrategyRunner')
                position_registry.release(self.inst_id, _sys)
                self._close_retry_count = 0  # 重置计数器让后续周期可以重试

    def _compute_realized_pnl(self, pos, result) -> float:
        pnl_from_result = getattr(result, 'pnl', None)
        if pnl_from_result not in (None, ""):
            try:
                return float(pnl_from_result)
            except (TypeError, ValueError):
                pass

        size = float(getattr(pos, 'size', 0) or 0)
        entry_price = float(getattr(pos, 'entry_price', 0) or 0)
        exit_price = float(getattr(result, 'filled_price', 0) or 0) or float(getattr(pos, 'current_price', 0) or 0)
        if size <= 0 or entry_price <= 0 or exit_price <= 0:
            return float(getattr(pos, 'unrealized_pnl', 0) or 0)

        side_name = getattr(getattr(pos, 'side', None), 'name', str(getattr(pos, 'side', ''))).upper()
        if side_name == "LONG":
            gross_pnl = (exit_price - entry_price) * size
        else:
            gross_pnl = (entry_price - exit_price) * size

        entry_fee = 0.0
        exit_fee = 0.0
        if hasattr(self.trade_executor, "estimate_fee_for_notional"):
            try:
                entry_fee = float(self.trade_executor.estimate_fee_for_notional(entry_price * size))
                exit_fee = float(self.trade_executor.estimate_fee_for_notional(exit_price * size))
            except Exception:
                entry_fee = 0.0
                exit_fee = 0.0
        return gross_pnl - entry_fee - exit_fee

    def _on_close_success(self, reason: str, pos, result, realized_pnl: float):
        """平仓成功钩子，供子类扩展。"""

    def _on_close_failure(self, reason: str, pos, result, retry_count: int):
        """平仓失败钩子，供子类扩展。"""

    def _on_position_gone(self):
        """
        持仓被外部清除时的钩子（强平 / 手动平仓 / TP/SL 触发）。

        默认行为：释放注册器（防止残留锁）。
        子类（如 ScanCampaignWorker）可覆盖以额外释放资金池。
        """
        _sys = self.config.get('_system_name', 'StrategyRunner')
        position_registry.release(self.inst_id, _sys)
        self.log_signal.emit(
            f"[{self.inst_id}] 检测到持仓被外部清除，已释放注册器", "WARNING"
        )

    def _on_takeover(self, inst_id: str, old_owner: str, new_owner: str):
        """当其他系统强制接管我们管理的 inst_id 时，清空内部状态但不平仓。"""
        if inst_id != self.inst_id:
            return
        self._campaign = None
        self._pending_signal = None
        self._close_retry_count = 0
        if hasattr(self, 'log_signal'):
            self.log_signal.emit(
                f"[{self.inst_id}] ⚠️ 被 {new_owner} 强制接管（原持有者={old_owner}），"
                "内部 campaign 已清空，仓位由接管方管理",
                "WARNING",
            )

    def _config_float(self, key: str, default: float) -> float:
        try:
            value = self.config.get(key, default)
            if value is None or value == "":
                value = default
            return float(value)
        except (TypeError, ValueError):
            return default

    def _config_int(self, key: str, default: int) -> int:
        try:
            value = self.config.get(key, default)
            if value is None or value == "":
                value = default
            return int(value)
        except (TypeError, ValueError):
            return default

    def _config_bool(self, key: str, default: bool) -> bool:
        value = self.config.get(key, default)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y", "on"}
        return bool(value)

    def _resolve_entry_amount(self, position_size: float = 0.0, stage: str = "pilot") -> float:
        pool_total = self._config_float('auto_trading_capital', 1000.0)
        if stage == "pilot":
            # 默认从 1% 提升至 2%（Kelly 最优区间；原 1% 资金利用率极低）
            ratio = self._config_float('pilot_position_pct', 0.02)
        elif stage == "add":
            ratio = self._config_float('add_position_pct', 0.10)
        elif stage == "stage3":
            ratio = self._config_float('stage3_position_pct', 0.15)
        else:
            ratio = self._config_float('position_size', 0.1)
        chosen = pool_total * ratio
        # ADX 缩减因子：震荡市自动降低仓位
        adx_scale = float(self.config.get('_adx_scale_factor', 1.0) or 1.0)
        if adx_scale != 1.0:
            chosen *= adx_scale
        if position_size and position_size > 1:
            chosen = min(chosen, float(position_size))
        try:
            exchange_available = float(self.trade_executor.get_usdt_balance() or 0.0)
        except Exception:
            exchange_available = 0.0
        return max(min(chosen, exchange_available), 0.0)

    def _compute_cost_line(self, entry_price: float, direction: str,
                            m3_df: Optional[pd.DataFrame] = None) -> float:
        """
        计算止损基准线（第一原则触发线）。

        ── 原逻辑（已废弃）──────────────────────────────────────────────────
        cost_line = entry_price × (1 + fee+slippage+impact ≈ 0.1%)
        多头止损线在开仓价 *上方*，任何轻微回落即触发 → 假止损率极高。

        ── 新逻辑 ───────────────────────────────────────────────────────────
        使用 ATR 动态缓冲（默认 1.5 × 14周期3m ATR）设置止损距离：
          • 多头：cost_line = entry_price × (1 - atr_buffer)  → 止损在下方
          • 空头：cost_line = entry_price × (1 + atr_buffer)  → 止损在上方
        当 ATR 数据不足时，保底使用 stop_loss_floor_pct（默认 1.5%）。
        """
        direction = str(direction).upper()

        # 保底止损距离（防止 ATR 极小时止损过紧）
        floor_pct = self._config_float('stop_loss_floor_pct', 2.0) / 100.0
        # ATR 倍数（可通过配置调整灵敏度）
        atr_mult  = self._config_float('cost_line_atr_multiplier', 2.0)
        # 止损上限（防止极端 ATR 导致止损过远）
        cap_pct   = self._config_float('stop_loss_cap_pct', 5.0) / 100.0

        atr_buffer = 0.0
        if m3_df is not None and not m3_df.empty and len(m3_df) >= 15:
            try:
                atr = float(self._atr(m3_df, 14).iloc[-1])
                if entry_price > 0 and atr > 0:
                    atr_buffer = (atr * atr_mult) / entry_price
            except Exception:
                pass

        total_buffer = min(max(floor_pct, atr_buffer), cap_pct)

        if direction == 'LONG':
            return entry_price * (1.0 - total_buffer)   # 止损在开仓价下方
        return entry_price * (1.0 + total_buffer)         # 止损在开仓价上方

    # ══════════════════════════════════════════════════════════════════════════
    # 新增风控 & 核心指标方法
    # ══════════════════════════════════════════════════════════════════════════

    def _h4_trend_allows(self, h4_df: pd.DataFrame, direction: str) -> bool:
        """
        4H 主趋势过滤。

        作为 3m 自动交易的主方向锚点：
        - LONG: 4H EMA快线 > EMA慢线，且快线斜率向上
        - SHORT: 4H EMA快线 < EMA慢线，且快线斜率向下
        """
        if h4_df is None or h4_df.empty or len(h4_df) < 40:
            return True
        close = h4_df['close']
        fast_span = self._config_int('h4_fast_ema', 20)
        slow_span = self._config_int('h4_slow_ema', 60)
        ema_fast = close.ewm(span=fast_span, adjust=False).mean()
        ema_slow = close.ewm(span=slow_span, adjust=False).mean()
        last_fast = float(ema_fast.iloc[-1])
        last_slow = float(ema_slow.iloc[-1])
        last_close = float(close.iloc[-1])
        lookback = min(6, len(ema_fast) - 1)
        if lookback <= 0:
            return True
        base = float(ema_fast.iloc[-1 - lookback]) or 1.0
        slope_pct = (last_fast - base) / base
        tolerance = 0.003
        if direction == 'LONG':
            return last_fast > last_slow and slope_pct > 0 and last_close >= last_fast * (1 - tolerance)
        return last_fast < last_slow and slope_pct < 0 and last_close <= last_fast * (1 + tolerance)

    def _d1_trend_allows(self, d1_df: pd.DataFrame, direction: str) -> bool:
        """
        日线趋势过滤（顺势交易核心规则）。
        LONG  → 日线收盘需在 EMA50 上方，且 EMA20 ≥ EMA50（多头排列）。
        SHORT → 日线收盘需在 EMA50 下方，且 EMA20 ≤ EMA50（空头排列）。
        数据不足时放行（不拦截），避免因数据缺失误判。
        """
        if d1_df is None or d1_df.empty or len(d1_df) < 50:
            return True
        close = d1_df['close']
        ema20 = close.ewm(span=20, adjust=False).mean()
        ema50 = close.ewm(span=50, adjust=False).mean()
        last_close = float(close.iloc[-1])
        last_ema20 = float(ema20.iloc[-1])
        last_ema50 = float(ema50.iloc[-1])
        tolerance = 0.01   # 1% 容差，避免临界频繁翻转
        if direction == 'LONG':
            return last_close > last_ema50 * (1 - tolerance) and last_ema20 >= last_ema50 * (1 - tolerance)
        return last_close < last_ema50 * (1 + tolerance) and last_ema20 <= last_ema50 * (1 + tolerance)

    def _rsi_not_extreme(self, h1_df: pd.DataFrame, direction: str) -> bool:
        """
        RSI 极值过滤：避免在超买区追多、超卖区追空。
        LONG  → H1 RSI 须 ≤ 75（未超买）。
        SHORT → H1 RSI 须 ≥ 25（未超卖）。
        """
        if h1_df is None or h1_df.empty or len(h1_df) < 20:
            return True
        rsi = self._rsi(h1_df['close'], 14)
        cur_rsi = float(rsi.iloc[-1])
        overbought = self._config_float('rsi_overbought', 75.0)
        oversold   = self._config_float('rsi_oversold',   25.0)
        if direction == 'LONG':
            return cur_rsi <= overbought
        return cur_rsi >= oversold

    def _volume_confirms(self, m3_df: pd.DataFrame, direction: str) -> bool:
        """
        成交量确认：入场 K 线的量须 ≥ 近 20 根均量的 80%。
        防止在无量行情（操纵、假突破）中入场。
        """
        if m3_df is None or m3_df.empty or len(m3_df) < 22:
            return True
        vol = m3_df['volume']
        avg_vol = float(vol.tail(21).iloc[:-1].mean())   # 过去 20 根均量（不含当前）
        cur_vol = float(vol.iloc[-1])
        if avg_vol <= 0:
            return True
        threshold = self._config_float('volume_confirm_ratio', 0.80)
        return cur_vol >= avg_vol * threshold

    def _atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Average True Range（平均真实波幅）"""
        high = df['high']
        low  = df['low']
        prev_close = df['close'].shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low  - prev_close).abs(),
        ], axis=1).max(axis=1)
        return tr.ewm(span=period, adjust=False).mean()

    def _adx(self, df: pd.DataFrame, period: int = 14) -> float:
        """
        Average Directional Index（平均趋向指数）。
        ADX > 25 = 明确趋势；ADX < 20 = 震荡整理；ADX > 40 = 强趋势。
        返回最新一根 ADX 值（浮点数，0-100）。
        """
        try:
            high  = df['high'].astype(float)
            low   = df['low'].astype(float)
            close = df['close'].astype(float)
            n = len(df)
            if n < period + 5:
                return 0.0
            # +DM / -DM
            up_move   = high.diff()
            down_move = -low.diff()
            plus_dm   = pd.Series(0.0, index=df.index)
            minus_dm  = pd.Series(0.0, index=df.index)
            cond_plus  = (up_move > down_move) & (up_move > 0)
            cond_minus = (down_move > up_move) & (down_move > 0)
            plus_dm[cond_plus]   = up_move[cond_plus]
            minus_dm[cond_minus] = down_move[cond_minus]
            # True Range
            prev_close = close.shift(1)
            tr = pd.concat([
                high - low,
                (high - prev_close).abs(),
                (low  - prev_close).abs(),
            ], axis=1).max(axis=1)
            # Wilder smoothing (EMA with alpha=1/period)
            atr_w   = tr.ewm(alpha=1.0/period, adjust=False).mean()
            plus_di  = 100 * plus_dm.ewm(alpha=1.0/period,  adjust=False).mean() / atr_w.replace(0, 1e-9)
            minus_di = 100 * minus_dm.ewm(alpha=1.0/period, adjust=False).mean() / atr_w.replace(0, 1e-9)
            dx_denom = (plus_di + minus_di).replace(0, 1e-9)
            dx = 100 * (plus_di - minus_di).abs() / dx_denom
            adx = dx.ewm(alpha=1.0/period, adjust=False).mean()
            return float(adx.iloc[-1])
        except Exception:
            return 0.0

    def _check_stage1_trail_stop(self, market: Dict[str, Any]) -> bool:
        """
        Stage1 浮盈保护（stage2 未完成时有效）。

        场景：1% 试仓建立后价格快速上涨，但从未满足第二次企稳条件，
        行情反转后原来的浮盈全部吐出才离场——损失机会成本。

        逻辑：
          1. 价格顺方向移动 ≥ stage1_trail_activate_pct（默认 3.0%）→ 保护启动
          2. 激活后持续追踪峰值（多头=最高价，空头=最低价）
          3. 价格从峰值回撤 ≥ 峰值涨幅 × stage1_trail_ratio（默认 50%）→ 全平离场
        """
        if not self._campaign or self._campaign.stage2_filled:
            return False

        campaign    = self._campaign
        price       = market['latest_price']
        entry       = campaign.stage1_entry_price
        if entry <= 0:
            return False

        activate_pct = self._config_float('stage1_trail_activate_pct', 2.0) / 100.0
        trail_ratio  = self._config_float('stage1_trail_ratio', 0.5)   # 峰值涨幅回撤比例

        # 动态风险覆盖：ATR 自适应激活阈值
        dynamic_risk = self.config.get('_dynamic_risk')
        if dynamic_risk and hasattr(dynamic_risk, 'trailing_stop_params'):
            try:
                atr_pct = market.get('atr_pct', 2.0)
                if atr_pct > 0:
                    snap = type('MockSnap', (), {'atr_pct': atr_pct})()
                    trail = dynamic_risk.trailing_stop_params(snap)
                    if trail.get('enabled'):
                        activate_pct = trail['activate_pct'] / 100.0
                        trail_ratio = 0.40  # 动态 ATR 下用更紧的回撤比例
            except Exception:
                pass

        if campaign.direction == 'LONG':
            gain_pct = (price - entry) / entry
            if gain_pct >= activate_pct:
                campaign.stage1_trail_armed = True
                if price > (campaign.peak_profit_price or 0):
                    campaign.peak_profit_price = price

            if not campaign.stage1_trail_armed:
                return False

            peak = campaign.peak_profit_price or price
            peak_gain = peak - entry
            if peak_gain > 0 and (peak - price) >= peak_gain * trail_ratio:
                self._close_position(
                    f"Stage1浮盈保护触发：从峰值 {peak:.6f} 回撤"
                    f" {(peak-price)/peak*100:.2f}%（峰值涨幅"
                    f" {peak_gain/entry*100:.2f}% 的 {trail_ratio*100:.0f}%），已离场锁定收益"
                )
                return True

        else:  # SHORT
            gain_pct = (entry - price) / entry
            if gain_pct >= activate_pct:
                campaign.stage1_trail_armed = True
                trough = campaign.peak_profit_price
                if not trough or price < trough:
                    campaign.peak_profit_price = price

            if not campaign.stage1_trail_armed:
                return False

            trough = campaign.peak_profit_price or price
            trough_gain = entry - trough
            if trough_gain > 0 and (price - trough) >= trough_gain * trail_ratio:
                self._close_position(
                    f"Stage1浮盈保护触发：从谷值 {trough:.6f} 反弹"
                    f" {(price-trough)/trough*100:.2f}%（谷值跌幅"
                    f" {trough_gain/entry*100:.2f}% 的 {trail_ratio*100:.0f}%），已离场锁定收益"
                )
                return True

        return False

    def _check_partial_exit(self, market: Dict[str, Any]) -> bool:
        """
        分批减仓（Stage2 完成后，浮盈达标时平掉 50% 锁定利润）。

        条件：
          - stage2_filled = True（已加仓到 11% 仓位）
          - partial_exit_done = False（本 Campaign 尚未执行过分批减仓）
          - 持仓浮盈 ≥ partial_exit_trigger_pct（默认 2.5%）

        效果：锁定一半利润，剩余 50% 仓位继续由移动止盈管理。
        返回 False 表示不终止 Campaign（仍继续持仓）。
        """
        if not self._campaign:
            return False
        campaign = self._campaign
        if not campaign.stage2_filled or campaign.partial_exit_done:
            return False

        positions = self.trade_executor.get_positions(self.inst_id)
        pos = positions.get(self.inst_id)
        if not pos:
            return False

        pnl_pct      = float(getattr(pos, 'pnl_percent', 0) or 0)   # 已是 %
        trigger_pct  = self._config_float('partial_exit_trigger_pct', 2.5)
        if pnl_pct < trigger_pct:
            return False

        result = self.trade_executor.close_position_partial(self.inst_id, ratio=0.5)
        if result and result.success:
            campaign.partial_exit_done = True
            # 更新资金记录（已减仓一半）
            campaign.total_allocated_usdt *= 0.5
            self.log_signal.emit(
                f"✅ 分批减仓成功：浮盈 {pnl_pct:.2f}% ≥ 阈值 {trigger_pct:.1f}%，"
                f"已平掉 50% 仓位锁定利润，剩余 50% 继续由移动止盈管理",
                "SUCCESS",
            )
        else:
            msg = getattr(result, 'message', '未知') if result else '执行器无响应'
            self.log_signal.emit(f"⚠️ 分批减仓失败：{msg}", "WARNING")

        return False  # 不终止 Campaign

    def _check_trailing_stop(self, market: Dict[str, Any]) -> bool:
        """
        移动止盈（Stage2 加仓后激活）。
        使用 ATR 自适应追踪距离：trail_distance = H4 ATR × trail_atr_mult
        低波动时追踪更紧，高波动时自动放宽，避免被噪声触发。
        """
        if not self._campaign or not self._campaign.stage2_filled:
            return False
        campaign = self._campaign
        price      = market['latest_price']

        # ── ATR 自适应追踪止损距离 ────────────────────────────────────────
        # 优先使用 H4 ATR%（已在 market snapshot 里计算好）；
        # 回退到固定 trail_stop_pct 配置值（向上调整默认值为 3%，原 2% 在大波动上太紧）
        atr_pct = float(market.get('atr_pct', 0) or 0)
        if atr_pct > 0:
            atr_mult      = self._config_float('trail_atr_mult', 2.0)
            floor_pct     = self._config_float('trail_stop_floor_pct', 1.5) / 100.0
            ceiling_pct   = self._config_float('trail_stop_ceiling_pct', 8.0) / 100.0
            trail_pct     = min(max(atr_pct * atr_mult / 100.0, floor_pct), ceiling_pct)
        else:
            trail_pct = self._config_float('trail_stop_pct', 3.0) / 100.0

        if campaign.direction == 'LONG':
            # 更新最高价
            if price > (campaign.peak_profit_price or price):
                campaign.peak_profit_price = price
            peak = campaign.peak_profit_price or price
            if peak > 0 and price <= peak * (1.0 - trail_pct):
                drawdown = (peak - price) / peak * 100
                self._close_position(
                    f"移动止盈触发：价格 {price:.6f} 从峰值 {peak:.6f} 回撤 {drawdown:.2f}%"
                )
                return True
        else:
            # 更新最低价
            if campaign.peak_profit_price <= 0 or price < campaign.peak_profit_price:
                campaign.peak_profit_price = price
            trough = campaign.peak_profit_price or price
            if trough > 0 and price >= trough * (1.0 + trail_pct):
                rebound = (price - trough) / trough * 100
                self._close_position(
                    f"移动止盈触发：价格 {price:.6f} 从谷值 {trough:.6f} 反弹 {rebound:.2f}%"
                )
                return True
        return False

    def _check_max_hold_time(self) -> bool:
        """
        超时强制平仓：持仓时间超过 max_hold_hours（默认 48h）后强制离场。
        防止隔夜/周末极端行情下无止损保护。
        """
        if not self._campaign:
            return False
        max_hours  = self._config_float('max_hold_hours', 48.0)
        hold_secs  = self._now_ts() - (self._campaign.opened_at or self._now_ts())
        if hold_secs >= max_hours * 3600:
            self._close_position(
                f"超时强制离场：持仓已达 {hold_secs / 3600:.1f}h，超过上限 {max_hours:.0f}h"
            )
            return True
        return False

    def _check_pending_signal_expiry(self):
        """
        待命信号超时清除。
        默认 4 小时内未达到入场条件，放弃信号（防止追踪过时行情）。
        """
        if not self._pending_signal:
            return
        expiry_hours = self._config_float('pending_signal_expiry_hours', 4.0)
        age_hours    = (self._now_ts() - self._pending_signal.created_at) / 3600
        if age_hours > expiry_hours:
            self.log_signal.emit(
                f"待命信号超时（{age_hours:.1f}h > {expiry_hours:.0f}h），已清除：{self._pending_signal.reason[:60]}",
                "WARNING",
            )
            self._pending_signal = None


class SimpleStrategyRunner(QObject):
    """简单策略运行器 - 用于没有 generate_signal 方法的策略"""

    log_signal = Signal(str, str)
    trade_signal = Signal(str, str, float, float)
    finished = Signal()

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

    def _recover_orphan_position(self):
        """启动时检测交易所是否已有本策略管理的持仓（崩溃恢复/重启恢复）。

        如果发现持仓，重建最小 campaign 状态以确保风控（ATR止损/H1反转/超时）正常运作。
        """
        try:
            positions = self.trade_executor.get_positions(self.inst_id)
            pos = positions.get(self.inst_id)
            if pos is None:
                return
            size = float(getattr(pos, 'size', 0) or 0)
            entry = float(getattr(pos, 'entry_price', 0) or 0)
            if size <= 0 or entry <= 0:
                return
            side = str(getattr(pos, 'side', '') or getattr(pos, 'direction', '')).upper()
            if side in ('LONG', 'BUY'):
                direction = 'LONG'
            elif side in ('SHORT', 'SELL'):
                direction = 'SHORT'
            else:
                return

            self._campaign = AutoTradeCampaign(
                direction=direction,
                first_signal_price=entry,
                stage1_entry_price=entry,
                stage1_cost_line=0.0,
                highest_since_stage1=entry,
                lowest_since_stage1=entry,
                total_allocated_usdt=float(getattr(pos, 'margin', size * entry * 0.1) or size * entry * 0.1),
                opened_at=self._now_ts(),
                last_reason='系统恢复-重新接管已有持仓',
                peak_profit_price=entry,
            )
            self._position_opened_at = self._now_ts()
            self.log_signal.emit(
                f"[{self.inst_id}] 🔄 检测到已有持仓 size={size} @ {entry:.6g} ({direction})，"
                f"已重建风控（止损/追踪/H1反转/超时保护已激活）",
                "WARNING",
            )
        except Exception as e:
            self.log_signal.emit(f"[{self.inst_id}] 孤儿仓位恢复失败: {e}", "ERROR")

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
