"""
扫描驱动自动交易编排器

流程：
  扫描页 → scan_results_ready 信号 → ScanDrivenAutoTrader.on_scan_results()
    → 筛选合格交易对 → 为每对启动 ScanCampaignWorker
    → 等待 H1 趋势确认 + 3m 回调企稳 → 开仓 → 管理持仓

支持实盘（TradeExecutor）和模拟（PaperTradeEngine），接口统一，切换只需换 trade_executor。
"""

import time
import threading
from typing import Dict, Optional, Tuple

from src.qt_compat import QObject, QThread, Signal
from src.strategy.runner import StrategyRunner, PendingAutoSignal, AutoTradeCampaign
from src.trading.entry_rule_guard import evaluate_entry_rule_from_klines
from src.trading.executor import PositionSide
from src.trading.position_registry import position_registry
from src.trading.risk_manager import RiskGuard, RiskLimits
from src.trading.risk_budget_sizer import calculate_position
from src.trading.dynamic_risk_controller import DynamicRiskController, DynamicRiskConfig, MarketRiskSnapshot


# ────────────────────────────────────────────────────────────────────────────
# 中央资金池
# ────────────────────────────────────────────────────────────────────────────

class CapitalPool:
    """
    线程安全资金池管理器。

    解决问题：多个 ScanCampaignWorker 并发时，各自独立调用
    get_usdt_balance() 会看到相同余额，叠加下单超过预算上限。

    机制：
      • reserve(amount)  → 预留保证金，失败返回 False（不阻塞）
      • release(amount)  → 释放保证金（平仓/止损后调用）
      • reset(new_total) → 动态更新资金池上限（余额变动后同步）
    """

    def __init__(self, total_usdt: float) -> None:
        self._lock     = threading.Lock()
        self._total    = max(0.0, float(total_usdt))
        self._session_start_total = self._total
        self._realized_pnl = 0.0
        self._reserved = 0.0   # 已被 Worker 预留的保证金（含待开仓+已持仓）

    def reserve(self, amount: float) -> bool:
        """预留保证金 amount USDT；资金不足时返回 False（不抛出异常）。"""
        if amount <= 0:
            return True
        with self._lock:
            if self._reserved + amount > self._total:
                return False
            self._reserved += amount
            return True

    def release(self, amount: float) -> None:
        """释放之前预留的保证金。"""
        if amount <= 0:
            return
        with self._lock:
            self._reserved = max(0.0, self._reserved - amount)

    def reset(self, new_total: float) -> None:
        """更新资金池总量（如外部充值/回款后调用）。"""
        with self._lock:
            self._total = max(0.0, float(new_total))
            self._session_start_total = self._total
            self._realized_pnl = 0.0

    def apply_realized_pnl(self, pnl: float, configured_cap: Optional[float] = None) -> None:
        """按会话已实现盈亏更新资金池上限。"""
        with self._lock:
            self._realized_pnl += float(pnl or 0.0)
            new_total = self._session_start_total + self._realized_pnl
            if configured_cap is not None:
                new_total = min(new_total, max(0.0, float(configured_cap)))
            self._total = max(0.0, new_total)

    def tighten_total(self, candidate_total: float) -> None:
        """仅允许向下收紧资金池上限，禁止因外部余额波动回升。"""
        with self._lock:
            self._total = max(0.0, min(self._total, float(candidate_total or 0.0)))

    @property
    def available(self) -> float:
        with self._lock:
            return max(0.0, self._total - self._reserved)

    @property
    def reserved(self) -> float:
        with self._lock:
            return self._reserved

    @property
    def total(self) -> float:
        with self._lock:
            return self._total

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "total": self._total,
                "reserved": round(self._reserved, 2),
                "available": round(max(0.0, self._total - self._reserved), 2),
                "realized_pnl": round(self._realized_pnl, 2),
            }


# ────────────────────────────────────────────────────────────────────────────
# 单对自动交易工作器
# ────────────────────────────────────────────────────────────────────────────

class ScanCampaignWorker(StrategyRunner):
    """
    扫描驱动的单对自动交易工作器。

    复用 StrategyRunner 的所有风控和建仓逻辑：
      - H1 趋势确认
      - 3m 回调企稳
      - 1% 试仓 → 10% 加仓
      - 第一原则熔断
      - H1 趋势反转离场

    入场信号由扫描结果提供，不需要策略自身再次生成信号。
    """

    campaign_state_signal = Signal(str, dict)   # (inst_id, state_dict)

    def __init__(self, candidate: dict, okx_client, trade_executor, config: dict = None):
        class _DummyStrategy:
            """哑策略：仅占位，信号已由扫描器提供。"""
            def generate_signal(self, klines):
                return None

        super().__init__(
            strategy_instance=_DummyStrategy(),
            inst_id=candidate['inst_id'],
            okx_client=okx_client,
            trade_executor=trade_executor,
            config=config or {},
        )
        self._candidate = candidate
        self._reusable = False   # TP/SL 触发后置 True，保留 worker 等待新信号
        # 信号有效期（默认 4 小时）
        self._signal_expiry = time.time() + float((config or {}).get('signal_expiry_hours', 4) or 4) * 3600

        # ── 扫描驱动模式专属默认参数 ────────────────────────────────────────
        # 只用 setdefault，不覆盖用户在 config 里的显式设置。
        #
        # H1 反转最小持仓期：父类默认 6h 太长，扫描驱动短线模式改为 1.5h
        self.config.setdefault('h1_reversal_min_hold_hours', 1.5)
        # 试仓比例从 1% 提升至 2%（Kelly 最优区间，提高资金利用率）
        self.config.setdefault('pilot_position_pct', 0.02)
        # 连续亏损熔断从 3 次放宽至 5 次（胜率 55% 时连亏 3 次属正常噪声）
        self.config.setdefault('max_consecutive_losses', 5)
        # ATR 自适应追踪止损参数（替代固定 2%，随波动率自动调整）
        self.config.setdefault('trail_atr_mult', 2.0)
        self.config.setdefault('trail_stop_floor_pct', 1.5)
        self.config.setdefault('trail_stop_ceiling_pct', 8.0)
        # Stage3 金字塔加仓默认比例（Stage2完成后趋势延续时再加 15%）
        self.config.setdefault('stage3_position_pct', 0.15)
        self.config.setdefault('stage3_trigger_pct', 2.0)   # Stage2完成后再涨 2% 触发

        direction = self._normalize_direction(candidate.get('direction', ''))
        if direction:
            reason = candidate.get('reason', '')
            score_txt = f"评分:{float(candidate.get('score', 0)):.1f}"
            self._pending_signal = PendingAutoSignal(
                direction=direction,
                signal_price=float(candidate.get('entry_price', 0) or 0),
                reason=f"[扫描] {score_txt} | {reason}",
                created_at=time.time(),
                raw_signal=dict(candidate),
            )
        else:
            self._pending_signal = None
        self._last_gate_reason = "等待3m回调企稳与H1趋势延续确认"

    # ── 资金池：覆盖父类资金计算，接入 CapitalPool ────────────────────────

    def _resolve_entry_amount(self, position_size: float = 0.0,
                               stage: str = 'pilot') -> float:
        """
        覆盖 StrategyRunner._resolve_entry_amount。

        如果 config 中存在 '_capital_pool'（CapitalPool 实例），则从资金池预留；
        否则回落到父类逻辑（兼容独立运行场景）。
        """
        pool: Optional[CapitalPool] = self.config.get('_capital_pool')
        if pool is None:
            return super()._resolve_entry_amount(position_size, stage)

        pool_total = pool.total
        if stage == 'pilot':
            ratio = self._config_float('pilot_position_pct', 0.01)
        elif stage == 'add':
            ratio = self._config_float('add_position_pct', 0.10)
        else:
            ratio = self._config_float('position_size', 0.1)

        amount = pool_total * ratio
        if position_size > 1:
            amount = min(amount, float(position_size))

        # 检查交易所实际可用余额（上限保障）
        try:
            exchange_available = float(self.trade_executor.get_usdt_balance() or 0.0)
        except Exception:
            exchange_available = 0.0
        amount = min(amount, exchange_available)
        amount = max(amount, 0.0)

        if amount <= 0:
            return 0.0

        # ── CRITICAL-2 FIX：最小订单金额保护 ────────────────────────────────
        # OKX 合约最低保证金通常 20 USDT；低于限额时下单被拒，
        # 但 campaign/registry/pool 已占用资源 → 状态机损坏
        min_usdt = float(self.config.get('min_order_usdt', 20.0))
        if amount < min_usdt:
            self.log_signal.emit(
                f"[资金池] {self.inst_id} 计算保证金 {amount:.2f} USDT"
                f" 低于最小订单限制 {min_usdt:.0f} USDT，跳过开仓"
                f"（可通过 min_order_usdt 配置调整）",
                "WARNING",
            )
            return 0.0

        # 向资金池预留保证金；名义价值由执行器按 leverage 放大
        if not pool.reserve(amount):
            leverage = max(self._config_int('leverage', 3), 1)
            self.log_signal.emit(
                f"[资金池] {self.inst_id} 资金池可用 {pool.available:.0f} USDT，"
                f"需要保证金 {amount:.0f} USDT（目标名义 {amount * leverage:.0f} USDT），预留失败，跳过开仓",
                "WARNING",
            )
            return 0.0

        leverage = max(self._config_int('leverage', 3), 1)
        self.log_signal.emit(
            f"[资金池] {self.inst_id} 已预留保证金 {amount:.2f} USDT"
            f"（目标名义 {amount * leverage:.2f} USDT）"
            f"（池可用剩余 {pool.available:.2f} USDT）",
            "INFO",
        )
        return amount

    def _close_position(self, reason: str):
        """
        覆盖 StrategyRunner._close_position。

        平仓成功后将当前 campaign 已分配资金归还资金池，
        并根据交易所实际余额同步资金池总量，避免盈亏漂移。
        """
        super()._close_position(reason)

    def _on_close_success(self, reason: str, pos, result, realized_pnl: float):
        pool: Optional[CapitalPool] = self.config.get('_capital_pool')
        allocated = float(getattr(self._campaign, 'total_allocated_usdt', 0) or 0)
        if pool and allocated > 0:
            pool.release(allocated)
            pool_cfg = float(self.config.get('auto_trading_capital', 1000.0) or 1000.0)
            pool.apply_realized_pnl(realized_pnl, configured_cap=pool_cfg)
            # ── HIGH-3 FIX：平仓后同步真实余额 ──────────────────────────────
            # apply_realized_pnl 基于代码内部计算，不含资金费率（每8小时）等交易所扣款。
            # 长期运行后 pool.total 会高估可用资金，用真实余额做单向收紧（tighten_total）。
            try:
                real_balance = float(self.trade_executor.get_usdt_balance() or 0.0)
                if real_balance > 0:
                    pool.tighten_total(real_balance)
            except Exception:
                pass
            self.log_signal.emit(
                f"[资金池] {self.inst_id} 平仓后释放保证金 {allocated:.2f} USDT，"
                f"已实现盈亏 {realized_pnl:+.2f} USDT（池可用 {pool.available:.2f} USDT）",
                "INFO",
            )

    def _on_close_failure(self, reason: str, pos, result, retry_count: int):
        self.log_signal.emit(
            f"[资金池] {self.inst_id} 平仓失败，保留保证金占用与注册器锁定，等待重试（第 {retry_count} 次）",
            "WARNING",
        )

    def _on_position_gone(self):
        """
        覆盖 StrategyRunner._on_position_gone。

        持仓被外部清除（强平 / 交易所 TP/SL / 手动平仓）时，
        除释放注册器外还需归还资金池预留额度，防止资金池永久泄漏。
        """
        pool: Optional[CapitalPool] = self.config.get('_capital_pool')
        allocated = float(getattr(self._campaign, 'total_allocated_usdt', 0) or 0)
        # 先释放注册器（父类默认行为）
        super()._on_position_gone()
        # 再归还资金池并同步真实余额
        if pool and allocated > 0:
            pool.release(allocated)
            try:
                real_balance = float(self.trade_executor.get_usdt_balance() or 0.0)
                if real_balance > 0:
                    pool.tighten_total(real_balance)
            except Exception:
                pass
            self.log_signal.emit(
                f"[资金池] {self.inst_id} 外部平仓后释放 {allocated:.2f} USDT"
                f"（池可用恢复至 {pool.available:.2f} USDT）",
                "WARNING",
            )

    # ── 主循环（覆盖父类，跳过 generate_signal）────────────────────────────

    def run(self):
        self._running = True
        self._stop_flag = False

        # ── CRITICAL-3 FIX：孤儿仓位恢复 ────────────────────────────────────
        # 程序崩溃/重启后，交易所可能有残留持仓但本地 _campaign=None。
        # 调用父类的孤儿仓位恢复，重建最小 campaign 以确保 ATR止损/H1反转/
        # 超时保护正常运作。必须在 _pending_signal 检查之前执行。
        self._recover_orphan_position()

        if not self._pending_signal and not self._campaign:
            self.log_signal.emit(
                f"[扫描驱动] {self.inst_id} 方向无法识别，跳过", "WARNING"
            )
            self.finished.emit()
            return

        self.log_signal.emit(
            f"[扫描驱动] {self.inst_id} 工作器启动"
            f"  方向={self._pending_signal.direction}"
            f"  有效期至 {time.strftime('%H:%M', time.localtime(self._signal_expiry))}"
            f"  {self._pending_signal.reason[:60]}",
            "TRADE",
        )

        try:
            while not self._stop_flag:
                klines = self._get_klines()
                if not klines:
                    self._interruptible_sleep(5)
                    continue

                market = self._build_market_snapshot(klines)
                if not market.get('valid'):
                    self._interruptible_sleep(10)
                    continue

                # 模拟模式：手动检查 TP/SL（交易所不会代劳）
                self._check_paper_tp_sl()
                if self._stop_flag:
                    break

                if self._campaign:
                    # ── 有持仓：全套风控检查（按优先级排序）────────────────
                    if self._check_max_hold_time():              # ① 超时强平
                        break
                    if self._enforce_first_principle(market):    # ② ATR止损线守护
                        break
                    if self._check_trailing_stop(market):        # ③ Stage2移动止盈
                        break
                    if self._check_stage1_trail_stop(market):   # ④ Stage1浮盈保护
                        break
                    if self._check_hourly_reversal_exit(market): # ⑤ H1趋势反转
                        break
                    self._check_partial_exit(market)             # ⑥ 分批减仓（不break）
                    self._process_campaign_add_on(market)
                    # ⑦ Stage3 金字塔加仓（Stage2后趋势强劲延续时追加 15%）
                    if self._campaign and self._campaign.stage2_filled:
                        self._process_campaign_stage3(market)

                elif self._pending_signal:
                    # ── 等待入场：信号超时检查 ──────────────────────────
                    if time.time() > self._signal_expiry:
                        self._last_gate_reason = "信号等待超时，已撤销本轮待命"
                        self.log_signal.emit(
                            f"[扫描驱动] {self.inst_id} 信号等待超时，"
                            "标记为可复用，等待下一轮扫描信号", "WARNING"
                        )
                        self._pending_signal = None
                        self._reusable = True
                        self._interruptible_sleep(5)
                        continue
                    # 使用轻量入场门槛（扫描已做过策略过滤）
                    self._try_open_pilot_position_scan(market)

                elif self._reusable:
                    # TP/SL 已触发或信号已过期，等待新信号推送
                    self._emit_campaign_state(market)
                    self._interruptible_sleep(5)
                    continue

                else:
                    # 无待命信号、无持仓、非复用 → 正常结束
                    break

                self._emit_campaign_state(market)
                self._interruptible_sleep(
                    max(self._config_int('auto_loop_interval_seconds', 5), 1))

        except Exception as e:
            import traceback
            self.log_signal.emit(
                f"[扫描驱动] {self.inst_id} 异常：{e}\n{traceback.format_exc()[:300]}",
                "ERROR",
            )
        finally:
            self._running = False
            self.finished.emit()

    # ── 轻量入场门槛（专为扫描驱动设计）────────────────────────────────────
    #
    # 扫描策略已做过全市场筛选，此处只做最后安全卡控，不重复严格过滤：
    #   Gate 1：连续亏损熔断（强制，避免策略失效后连续爆仓）
    #   Gate 2：H1 趋势方向未反转（EMA12 vs EMA26，1% 容差）
    #   Gate 3：RSI 未到极端区域（80/20，比标准版宽松）
    #   Gate 4：成交量不低于均量 60%（比标准版 80% 宽松）
    # ──────────────────────────────────────────────────────────────────────────

    def _try_open_pilot_position_scan(self, market: dict):
        """扫描驱动专用轻量入场逻辑（替代父类 _try_open_pilot_position）。"""
        pending = self._pending_signal
        if not pending:
            return

        inst = self.inst_id
        direction = pending.direction

        # ── Gate 1：连续亏损熔断 + 窗口胜率熔断 ────────────────────────────
        max_cl = self._config_int('max_consecutive_losses', 5)  # 默认提高到 5
        if self._consecutive_losses >= max_cl:
            self._last_gate_reason = f"连续亏损熔断：{self._consecutive_losses} 次"
            self.log_signal.emit(
                f"[扫描驱动] {inst} 连续亏损熔断（{self._consecutive_losses}次），暂停开仓",
                "ERROR",
            )
            self._stop_flag = True
            return
        # 窗口胜率检查
        _win_window = self._config_int('loss_rate_window', 10)
        _min_wr = self._config_float('min_recent_win_rate', 30.0) / 100.0
        if len(self._trade_history) >= _win_window:
            _recent = self._trade_history[-_win_window:]
            _cur_wr = sum(_recent) / len(_recent)
            if _cur_wr < _min_wr:
                self._last_gate_reason = f"近{_win_window}笔胜率{_cur_wr*100:.0f}%过低"
                self.log_signal.emit(
                    f"[扫描驱动] {inst} 窗口胜率熔断（{_cur_wr*100:.0f}%<{_min_wr*100:.0f}%），暂停",
                    "ERROR",
                )
                self._stop_flag = True
                return

        h1_df = market.get('h1_df')
        m3_df = market.get('m3_df')
        latest_price = market.get('latest_price', 0.0)

        # ── Gate 2：H1 趋势方向基本检查（未发生反转）──────────────────────
        if h1_df is not None and not h1_df.empty and len(h1_df) >= 30:
            ema12 = float(h1_df['close'].ewm(span=12, adjust=False).mean().iloc[-1])
            ema26 = float(h1_df['close'].ewm(span=26, adjust=False).mean().iloc[-1])
            tolerance = 0.01   # 1% 容差
            if direction == 'LONG' and ema12 < ema26 * (1 - tolerance):
                self._last_gate_reason = "H1趋势已反转，LONG待命失效"
                self.log_signal.emit(
                    f"[扫描驱动] {inst} H1 趋势已反转（EMA12 < EMA26），信号失效，撤销待命",
                    "WARNING",
                )
                self._pending_signal = None
                return
            if direction == 'SHORT' and ema12 > ema26 * (1 + tolerance):
                self._last_gate_reason = "H1趋势已反转，SHORT待命失效"
                self.log_signal.emit(
                    f"[扫描驱动] {inst} H1 趋势已反转（EMA12 > EMA26），信号失效，撤销待命",
                    "WARNING",
                )
                self._pending_signal = None
                return

        # ── Gate 3：RSI 未到极端区（80/20，宽松阈值）──────────────────────
        if h1_df is not None and not h1_df.empty and len(h1_df) >= 20:
            rsi_series = self._rsi(h1_df['close'], 14)
            cur_rsi = float(rsi_series.iloc[-1])
            ob = self._config_float('rsi_overbought_scan', 80.0)   # 扫描模式更宽松
            os_ = self._config_float('rsi_oversold_scan', 20.0)
            if direction == 'LONG' and cur_rsi > ob:
                self._last_gate_reason = f"H1 RSI超买 {cur_rsi:.1f}，继续等待回落"
                self.log_signal.emit(
                    f"[扫描驱动] {inst} RSI 极端超买 {cur_rsi:.1f}>{ob:.0f}，等待回落",
                    "WARNING",
                )
                return
            if direction == 'SHORT' and cur_rsi < os_:
                self._last_gate_reason = f"H1 RSI超卖 {cur_rsi:.1f}，继续等待回升"
                self.log_signal.emit(
                    f"[扫描驱动] {inst} RSI 极端超卖 {cur_rsi:.1f}<{os_:.0f}，等待回升",
                    "WARNING",
                )
                return

        # ── Gate 4：成交量（60% 均量阈值，比标准版 80% 宽松）──────────────
        if m3_df is not None and not m3_df.empty and len(m3_df) >= 22:
            vol = m3_df['volume']
            avg_vol = float(vol.tail(21).iloc[:-1].mean())
            cur_vol = float(vol.iloc[-1])
            vol_threshold = self._config_float('volume_confirm_ratio_scan', 0.60)
            if avg_vol > 0 and cur_vol < avg_vol * vol_threshold:
                self._last_gate_reason = (
                    f"3m成交量不足：{cur_vol:.0f} < 均量{avg_vol:.0f}×{vol_threshold:.2f}"
                )
                self.log_signal.emit(
                    f"[扫描驱动] {inst} 成交量 {cur_vol:.0f} < 均量 {avg_vol:.0f} 的 {vol_threshold*100:.0f}%，等待放量",
                    "WARNING",
                )
                return

        # ── Gate 5：3m 回调企稳实时确认 ─────────────────────────────────────
        #
        # 扫描策略的 Gate A 是截面静态检查（扫描时刻的历史形态），
        # 此处做入场时刻的实时验证，确保当前 K 线仍处于企稳状态，
        # 避免在企稳后又二次下跌时仍按旧信号入场。
        if m3_df is not None and not m3_df.empty and len(m3_df) >= 24:
            pullback = self._detect_pullback_stabilization(m3_df, direction)
            if not pullback.get('ready'):
                self._last_gate_reason = f"3m企稳未就绪：{pullback.get('reason', '继续等待')}"
                self.log_signal.emit(
                    f"[扫描驱动] {inst} Gate5 3m企稳未就绪"
                    f"（{pullback.get('reason', '')}），继续等待",
                    "INFO",
                )
                return

        # ── Gate 6：DRC OI + 资金费率方向确认 ──────────────────────────────
        # DynamicRiskController 基于 OI 变化 + 资金费率拐点做趋势质量评估。
        # 仅在 DRC 实例存在（由 ScanDrivenAutoTrader 注入 config）时生效。
        _drc = self.config.get('_dynamic_risk')
        if _drc is not None:
            try:
                from src.trading.dynamic_risk_controller import MarketRiskSnapshot
                _snap = MarketRiskSnapshot(
                    symbol=inst,
                    funding_rate=float(self._candidate.get('funding_rate', 0) or 0),
                    funding_rate_1d_ago=float(self._candidate.get('funding_24h_ago', 0) or 0),
                    oi_change_24h=float(self._candidate.get('oi_change_24h', 0) or 0),
                    oi_change_4h=float(self._candidate.get('oi_change_4h', 0) or 0),
                    price_change_1h=float(self._candidate.get('price_change_1h', 0) or 0),
                    atr_pct=float(market.get('atr_pct', 2.0) or 2.0),
                )
                # BTC 暴跌熔断
                _halt, _halt_reason = _drc.btc_crash_halt(_snap)
                if _halt and direction == 'LONG':
                    self._last_gate_reason = f"BTC暴跌熔断：{_halt_reason}"
                    self.log_signal.emit(
                        f"[扫描驱动] {inst} Gate6 BTC暴跌熔断多头：{_halt_reason}", "ERROR"
                    )
                    return
                # 资金费率极端拥挤过滤
                _funding = _drc.funding_direction_signal(_snap)
                if _funding.get('weight', 0) >= 0.6:
                    if _funding['signal'] == 'bearish' and direction == 'LONG':
                        self._last_gate_reason = f"资金费率多头极度拥挤：{_funding['detail']}"
                        self.log_signal.emit(
                            f"[扫描驱动] {inst} Gate6 {_funding['detail']}，放弃多头入场",
                            "WARNING",
                        )
                        return
                    elif _funding['signal'] == 'bullish' and direction == 'SHORT':
                        self._last_gate_reason = f"资金费率空头极度拥挤：{_funding['detail']}"
                        self.log_signal.emit(
                            f"[扫描驱动] {inst} Gate6 {_funding['detail']}，放弃空头入场",
                            "WARNING",
                        )
                        return
                # OI 趋势质量（仅做日志，不拦截）
                _oi = _drc.oi_trend_confirm(_snap, direction.lower())
                if _oi.get('weight', 0) < -0.2:
                    self.log_signal.emit(
                        f"[扫描驱动] {inst} OI质量警告：{_oi.get('detail','')}，降低期望收益",
                        "WARNING",
                    )
            except Exception:
                pass  # DRC 检查失败不影响主流程

        # ── 注册器检查：防止与其他 AI 系统产生持仓冲突 ───────────────────────
        if not position_registry.try_lock(inst, 'ScanDriven'):
            _owner = position_registry.get_owner(inst)
            self._last_gate_reason = f"注册器占用：已被 {_owner} 持有"
            self.log_signal.emit(
                f"[扫描驱动] {inst} 已由 {_owner} 持有，跳过本次入场",
                "WARNING",
            )
            return

        # ── 所有门槛通过，执行 1% 试仓 ──────────────────────────────────────
        amount = self._resolve_entry_amount(stage='pilot')
        if amount <= 0:
            position_registry.release(inst, 'ScanDriven')
            self._last_gate_reason = "自动交易资金池不足，无法执行1%试仓"
            self.log_signal.emit(
                f"[扫描驱动] {inst} 自动交易资金池不足，无法执行 1% 试仓",
                "ERROR",
            )
            return

        self.log_signal.emit(
            f"[扫描驱动] {inst} 全部 Gate 通过  "
            f"方向={direction}  当前价={latest_price:.6f}  金额={amount:.2f} USDT  "
            f"原因={pending.reason[:80]}",
            "TRADE",
        )

        result = self._open_position(
            direction=direction,
            usdt_amount=amount,
            reason=f"1%试仓[扫描驱动] | {pending.reason}",
        )
        if not result.success:
            position_registry.release(inst, 'ScanDriven')
            self._last_gate_reason = f"1%试仓失败：{result.message}"
            self.log_signal.emit(f"[扫描驱动] {inst} 1%试仓失败：{result.message}", "ERROR")
            return

        # 使用 ATR 动态止损线（传入 m3_df），锁定 buffer 供 Stage2 复用
        cost_line = self._compute_cost_line(latest_price, direction, m3_df)
        stage1_buffer_pct = (abs(cost_line - latest_price) / latest_price
                             if latest_price > 0 else 0.02)
        self._campaign = AutoTradeCampaign(
            direction=direction,
            first_signal_price=pending.signal_price,
            stage1_entry_price=latest_price,
            stage1_cost_line=cost_line,
            stage1_atr_buffer_pct=stage1_buffer_pct,
            highest_since_stage1=latest_price,
            lowest_since_stage1=latest_price,
            total_allocated_usdt=amount,
            opened_at=time.time(),
            last_reason=pending.reason,
            peak_profit_price=latest_price,
        )
        self._pending_signal = None
        self._last_gate_reason = "已通过全部过滤并完成1%试仓"
        _dist_pct = abs(cost_line - latest_price) / latest_price * 100 if latest_price else 0
        self.log_signal.emit(
            f"[扫描驱动] ✅ {inst} 1%试仓成功！"
            f"  {direction} @ {latest_price:.6f}"
            f"  ATR止损线={cost_line:.6f}（距开仓价 {_dist_pct:.2f}%）"
            f"  后续等待第二次企稳后加仓 10%",
            "SUCCESS",
        )

    # ── 模拟 TP/SL 检查（仅 PaperTradeEngine 有此接口）────────────────────

    def _check_paper_tp_sl(self):
        executor = self.trade_executor
        if not (hasattr(executor, 'check_tp_sl') and hasattr(executor, 'update_prices')):
            return
        try:
            executor.update_prices()
            pos = executor.get_positions(self.inst_id).get(self.inst_id)
            if not pos:
                return
            reason = executor.check_tp_sl(self.inst_id)
            if not reason:
                return
            self.log_signal.emit(f"[模拟] {self.inst_id} {reason}，自动平仓", "WARNING")
            res = executor.execute_stop_loss(self.inst_id, exit_reason=reason)
            if res.success:
                self.log_signal.emit(f"[模拟] {self.inst_id} 平仓成功：{res.message}", "SUCCESS")
                self.trade_signal.emit(
                    "SELL" if getattr(pos, 'direction', '') == 'LONG' else "COVER",
                    self.inst_id,
                    float(getattr(pos, 'current_price', 0)),
                    float(getattr(pos, 'size', 0)),
                )
                # ⚠️ 仅平仓成功后才归还资金池（修复：防止止损失败仍释放资金）
                _pool: Optional[CapitalPool] = self.config.get('_capital_pool')
                _alloc = float(getattr(self._campaign, 'total_allocated_usdt', 0) or 0)
                _entry_px = float(getattr(pos, 'entry_price', 0) or 0)
                _size = float(getattr(pos, 'size', 0) or 0)
                _exit_px = float(getattr(pos, 'current_price', 0) or 0)
                _dir = str(getattr(pos, 'direction', '') or '').upper()
                if _entry_px > 0 and _size > 0:
                    if _dir == 'LONG':
                        _realized = (_exit_px - _entry_px) * _size
                    elif _dir == 'SHORT':
                        _realized = (_entry_px - _exit_px) * _size
                    else:
                        _realized = 0.0
                else:
                    _realized = 0.0
                if _pool and _realized != 0:
                    _pool.apply_realized_pnl(_realized)
                self._campaign = None
                self._pending_signal = None
                self._reusable = True
                if _pool and _alloc > 0:
                    _pool.release(_alloc)
                    self.log_signal.emit(
                        f"[资金池] {self.inst_id} 模拟TP/SL后释放 {_alloc:.2f} USDT"
                        f"{'，结算盈亏 ' + str(round(_realized, 4)) + ' USDT' if _realized != 0 else ''}",
                        "INFO",
                    )
            else:
                self.log_signal.emit(
                    f"[模拟] {self.inst_id} ⚠️ TP/SL 平仓失败({res.message})，保留仓位与资金池",
                    "WARNING",
                )
        except Exception as e:
            self.log_signal.emit(f"[模拟] TP/SL 检查异常：{e}", "ERROR")

    # ── 状态广播 ──────────────────────────────────────────────────────────

    def _emit_campaign_state(self, market: dict):
        pos = self.trade_executor.get_positions(self.inst_id).get(self.inst_id)
        state: dict = {
            'inst_id': self.inst_id,
            'direction': self._candidate.get('direction', ''),
            'score': float(self._candidate.get('score', 0)),
            'stage': '已开仓' if self._campaign else '等待入场',
            'current_price': market.get('latest_price', 0),
            'scan_reason': self._candidate.get('reason', ''),
            'gate_reason': str(getattr(self, '_last_gate_reason', '') or ''),
            'entry_price': 0.0,
            'unrealized_pnl': 0.0,
            'stage2': bool(self._campaign and self._campaign.stage2_filled) if self._campaign else False,
        }
        if pos:
            state['entry_price'] = float(getattr(pos, 'entry_price', 0))
            state['unrealized_pnl'] = float(getattr(pos, 'unrealized_pnl', 0))
        self.campaign_state_signal.emit(self.inst_id, state)

    # ── 信号复用 ─────────────────────────────────────────────────────────

    def is_idle(self) -> bool:
        """返回 True 表示 worker 无持仓、无待处理信号，可接收新信号。"""
        return (self._campaign is None
                and self._pending_signal is None
                and self._reusable)

    def receive_signal(self, candidate: dict):
        """推送新的扫描候选信号给正在等待的 worker。"""
        direction = self._normalize_direction(candidate.get('direction', ''))
        if not direction:
            return
        self._candidate = candidate
        reason = candidate.get('reason', '')
        score_txt = f"评分:{float(candidate.get('score', 0)):.1f}"
        self._pending_signal = PendingAutoSignal(
            direction=direction,
            signal_price=float(candidate.get('entry_price', 0) or 0),
            reason=f"[扫描] {score_txt} | {reason}",
            created_at=time.time(),
            raw_signal=dict(candidate),
        )
        self._signal_expiry = time.time() + float(
            (self.config or {}).get('signal_expiry_hours', 4) or 4
        ) * 3600
        self._reusable = False  # 收到新信号，退出复用等待模式
        self._last_gate_reason = f"收到新信号，等待3m回调企稳与H1趋势延续确认：{reason}"
        self.log_signal.emit(
            f"[扫描驱动] {self.inst_id} 复用已有 worker，接收新信号: {direction}",
            "INFO",
        )

    # ── 工具 ─────────────────────────────────────────────────────────────

    @staticmethod
    def _normalize_direction(d: str) -> str:
        d = str(d or '').upper().strip()
        if d in {'BUY', 'LONG', 'BUY_LONG'}:
            return 'LONG'
        if d in {'SELL', 'SHORT', 'SELL_SHORT'}:
            return 'SHORT'
        return ''


# ────────────────────────────────────────────────────────────────────────────
# 编排器
# ────────────────────────────────────────────────────────────────────────────

class ScanDrivenAutoTrader(QObject):
    """
    扫描驱动自动交易编排器。

    职责：
    - 接收扫描结果（scan_results_ready 信号）
    - 按最低评分、最大并发持仓数过滤
    - 为每个合格交易对启动一个 ScanCampaignWorker
    - 汇聚所有 worker 的状态并推送给 UI

    实盘 / 模拟：只需替换传入的 trade_executor。
    """

    log_signal = Signal(str, str)
    state_updated = Signal(list)           # list[dict] 全部 campaign 状态
    position_opened = Signal(str, str)     # (inst_id, direction)
    position_closed = Signal(str)          # inst_id
    conflict_signal = Signal(str, str, str)  # (inst_id, new_direction, existing_direction) 方向冲突通知
    worker_finished_signal = Signal(str)

    def __init__(self, okx_client, trade_executor, config: dict = None):
        super().__init__()
        self.okx_client = okx_client
        self.trade_executor = trade_executor
        self.config = config or {}
        self._workers: Dict[str, Tuple[QThread, ScanCampaignWorker]] = {}
        self._states: Dict[str, dict] = {}
        self._workers_lock = threading.RLock()
        self._session_start_balance: float = 0.0   # 会话初始资金，用于回撤保护
        self._equity_breaker_tripped: bool = False  # 是否已触发资金回撤熔断
        self._global_consecutive_losses: int = 0     # 全局连续亏损计数（跨 worker）
        # 僵尸 worker 线程列表：stop 后若线程未在 wait 时间内退出，暂存于此保持
        # Python 引用，防止 GC 在 OS 线程仍运行时销毁 QThread → SIGABRT。
        self._zombie_threads: list = []
        # 中央资金池：统一管理扫描驱动交易的可用资金，防止多 Worker 超额占用
        pool_usdt = float(self.config.get('auto_trading_capital', 1000.0) or 1000.0)
        self._capital_pool = CapitalPool(pool_usdt)
        # 风险守卫：日亏损熔断 + 最大回撤 + 并发上限 + 开盘保护
        limits = RiskLimits(
            daily_loss_limit_pct=float(self.config.get('max_daily_loss_pct', 5.0) or 5.0),
            max_drawdown_limit_pct=float(self.config.get('max_equity_drawdown_pct', 15.0) or 15.0),
            max_concurrent_positions=int(self.config.get('max_concurrent_positions', 3) or 3),
            max_daily_trades=int(self.config.get('max_daily_trades', 50) or 50),
            max_exposure_per_symbol=float(self.config.get('max_exposure_per_symbol', 0.25) or 0.25),
        )
        self.risk_guard = RiskGuard(limits)
        # 动态风险控制器 (ATR自适应止损 + 追踪止损 + 费率拐点 + OI确认 + BTC熔断 + 波动率仓位缩放)
        dynamic_cfg = DynamicRiskConfig(
            atr_stop_mult=float(self.config.get('atr_stop_mult', 2.5) or 2.5),
            trail_enabled=bool(self.config.get('trail_enabled', True)),
            trail_activate_atr_mult=float(self.config.get('trail_activate_atr', 1.5) or 1.5),
            trail_distance_atr_mult=float(self.config.get('trail_distance_atr', 2.0) or 2.0),
            btc_crash_halt_pct=float(self.config.get('btc_crash_halt_pct', -5.0) or -5.0),
            vol_scale_enabled=bool(self.config.get('vol_scale_enabled', True)),
        )
        self.dynamic_risk = DynamicRiskController(dynamic_cfg)
        try:
            bal = float(self.trade_executor.get_usdt_balance() or 0.0)
            if bal > 0:
                self.risk_guard.set_initial_equity(bal)
        except Exception:
            pass
        self.worker_finished_signal.connect(self._on_worker_finished)
        self._init_session_balance()

        # 注册接管回调：当其他系统强制接管我们管理的标的时停止对应 worker
        position_registry.register_takeover_callback(self._on_takeover)

    # ── 接收扫描结果 ──────────────────────────────────────────────────────

    def _init_session_balance(self):
        """记录会话开始时的资金余额，用于回撤保护基准；同时初始化资金池上限。"""
        try:
            bal = float(self.trade_executor.get_usdt_balance() or 0.0)
            if bal > 0:
                self._session_start_balance = bal
                # 资金池上限取：配置值 vs 实际余额 的较小值（防止配置超过实际余额）
                pool_cfg = float(self.config.get('auto_trading_capital', 1000.0) or 1000.0)
                self._capital_pool.reset(min(pool_cfg, bal))
        except Exception:
            pass

    def _check_equity_drawdown(self) -> bool:
        """
        资金回撤熔断：从会话权益峰值计算回撤，超过阈值则停止所有新开仓。

        使用峰值回撤（而非从初始余额回撤），避免开仓瞬间的保证金占用 +
        手续费 + 滑点导致 totalEq 短期下降被误判为真实回撤。
        默认阈值 15%（可通过 max_equity_drawdown_pct 配置）。
        """
        if self._session_start_balance <= 0:
            return False
        max_dd_pct = float(self.config.get('max_equity_drawdown_pct', 15.0) or 15.0) / 100.0
        try:
            cur_bal = float(self.trade_executor.get_usdt_balance() or 0.0)
        except Exception:
            # 查询失败时，若熔断已触发则维持；否则放行
            return self._equity_breaker_tripped

        if cur_bal <= 0:
            return self._equity_breaker_tripped

        # 追踪会话权益峰值
        if not hasattr(self, '_session_peak_equity'):
            self._session_peak_equity = self._session_start_balance
        if cur_bal > self._session_peak_equity:
            self._session_peak_equity = cur_bal

        # 从峰值回撤（而非从初始值），过滤开仓瞬间的临时占用波动
        if self._session_peak_equity <= 0:
            return False

        drawdown = (self._session_peak_equity - cur_bal) / self._session_peak_equity

        # ── MEDIUM-1 FIX：熔断自动恢复 ────────────────────────────────────
        # 当余额从熔断时的低点回升，回撤收窄至 recovery_threshold 以内时自动解除。
        # recovery_threshold 须在两个分支都可见，故提前定义。
        recovery_threshold = float(self.config.get('equity_breaker_recovery_pct', 5.0) or 5.0) / 100.0

        if self._equity_breaker_tripped:
            if drawdown <= recovery_threshold:
                self._equity_breaker_tripped = False
                self._session_peak_equity = cur_bal   # 重置峰值
                self.log_signal.emit(
                    f"✅ 资金回撤熔断已自动解除：当前权益 {cur_bal:.2f} USDT，"
                    f"相对峰值回撤已收窄至 {drawdown*100:.1f}%"
                    f"（恢复阈值 {recovery_threshold*100:.0f}%），"
                    f"允许恢复自动开仓。",
                    "INFO",
                )
                return False
            return True

        if drawdown >= max_dd_pct:
            self._equity_breaker_tripped = True
            self.log_signal.emit(
                f"⚠️ 资金回撤熔断：当前权益 {cur_bal:.2f} USDT，"
                f"相对会话峰值 {self._session_peak_equity:.2f} USDT "
                f"回撤 {drawdown*100:.1f}%（阈值 {max_dd_pct*100:.0f}%）。"
                f"已暂停所有新开仓；余额恢复至峰值的"
                f" {(1-recovery_threshold)*100:.0f}% 以上时自动解除。",
                "ERROR",
            )
            return True
        return False

    def on_monitor_signal(self, sig_dict: dict):
        """
        接收来自 MonitorWorker.high_quality_signal 的高质量信号。

        MonitorEngine 产生的信号格式与 on_scan_results 期望的列表元素格式兼容：
          { inst_id, symbol, direction, score, opportunity_score,
            reason, entry_price, source, timestamp }

        直接包装成单元素列表转发给 on_scan_results，复用全部过滤/启动逻辑。
        """
        if not sig_dict:
            return
        self.log_signal.emit(
            f"[监控池→自动交易] 收到高质量信号：{sig_dict.get('symbol','')} "
            f"{sig_dict.get('direction','')} 评分={sig_dict.get('score',0):.0f}",
            "INFO",
        )
        self.on_scan_results([sig_dict])

    def on_scan_results(self, results: list):
        """扫描结果到达时筛选并启动 worker。"""
        if not results:
            return

        # ── 全局资金回撤熔断检查 ────────────────────────────────────────────
        if self._check_equity_drawdown():
            return

        # ── RiskGuard 风险检查（日亏损/回撤/超限/开盘保护） ─────────────────
        try:
            cur_bal = float(self.trade_executor.get_usdt_balance() or 0.0)
            self.risk_guard.update_equity(cur_bal)
        except Exception:
            pass
        if self.risk_guard.is_circuit_breaker_active():
            self.log_signal.emit(
                f"[风险守卫] 熔断触发: {self.risk_guard.get_circuit_reason()}，停止新开仓",
                "ERROR",
            )
            return

        # ── 全局连续亏损熔断 ──────────────────────────────────────────────────
        max_consecutive = int(self.config.get('max_consecutive_losses', 3) or 3)
        if self._global_consecutive_losses >= max_consecutive:
            self.log_signal.emit(
                f"[风控] 全局连续亏损熔断（{self._global_consecutive_losses}次），暂停新开仓",
                "ERROR",
            )
            return

        # ── 动态风险：BTC 暴跌熔断（所有山寨暂停新多头）─────────────────────
        btc_ctx = self._extract_btc_context_from_results(results)
        if btc_ctx:
            self.dynamic_risk.update_btc_crash_status(btc_ctx.get('btc_1h_pct', 0))
            is_halted, halt_reason = self.dynamic_risk.btc_crash_halt(
                MarketRiskSnapshot(symbol="GLOBAL", btc_1h_pct=btc_ctx.get('btc_1h_pct', 0))
            )
            if is_halted:
                self.log_signal.emit(f"[动态风控] {halt_reason}", "ERROR")

        # 默认评分阈值降为 50（扫描策略已做过主要过滤，不需要二次高门槛）
        min_score = float(self.config.get('min_auto_score', 50.0) or 50.0)
        max_pos = int(self.config.get('max_concurrent_positions', 3) or 3)
        allow_short = bool(self.config.get('allow_short', True))

        with self._workers_lock:
            running_count = sum(1 for t, _ in self._workers.values() if t.isRunning())

        self.log_signal.emit(
            f"[扫描驱动] 收到 {len(results)} 条扫描结果，"
            f"筛选阈值: 评分≥{min_score:.0f}  最大持仓={max_pos}  "
            f"当前监控中={running_count}",
            "INFO",
        )

        launched = 0
        for res in results:
            if running_count >= max_pos:
                self.log_signal.emit(
                    f"[扫描驱动] 已达最大监控数 {max_pos}，剩余结果跳过", "INFO"
                )
                break

            inst_id = str(res.get('symbol', res.get('inst_id', '')) or '').strip()
            if not inst_id:
                continue

            score = float(res.get('opportunity_score', res.get('score', 0)) or 0)
            if score < min_score:
                self.log_signal.emit(
                    f"[扫描驱动] {inst_id} 评分 {score:.1f} < {min_score:.0f}，跳过",
                    "INFO",
                )
                continue

            direction = ScanCampaignWorker._normalize_direction(
                str(res.get('direction', res.get('side', '')) or '')
            )
            if not direction:
                self.log_signal.emit(
                    f"[扫描驱动] {inst_id} 方向字段为空/不识别"
                    f"（direction={res.get('direction')} side={res.get('side')}），跳过",
                    "WARNING",
                )
                continue
            if direction == 'SHORT' and not allow_short:
                self.log_signal.emit(
                    f"[扫描驱动] {inst_id} SHORT 信号，但已禁止做空，跳过", "INFO"
                )
                continue

            # ── RiskGuard: 检查是否允许开新仓 ──
            try:
                cur_bal = float(self.trade_executor.get_usdt_balance() or 0.0)
            except Exception:
                cur_bal = 0.0
            can_open, open_reason = self.risk_guard.can_open_position(inst_id, cur_bal)
            if not can_open:
                self.log_signal.emit(
                    f"[风险守卫] {inst_id} 不允许开仓: {open_reason}", "WARNING"
                )
                continue

            klines_map = dict(res.get('klines_map', {}) or {})
            if klines_map:
                guard = evaluate_entry_rule_from_klines(
                    klines_map,
                    direction,
                    self.config,
                )
                if not guard.get('ok'):
                    self.log_signal.emit(
                        f"[扫描驱动] {inst_id} 未通过3m/H1硬性开仓原则：{guard.get('reason', '未知原因')}，跳过",
                        "INFO",
                    )
                    continue

            # ── 规则：检查该交易对是否已有持仓（手动 / 自动均算）────────────
            existing_dir = self._get_existing_position_direction(inst_id)
            if existing_dir:
                if existing_dir == direction:
                    # 同向：已有持仓，不重复开仓
                    self.log_signal.emit(
                        f"[扫描驱动] {inst_id} 已有 {existing_dir} 持仓，跳过重复开仓",
                        "INFO",
                    )
                    continue
                else:
                    # 反向：发出冲突通知并立即平仓
                    self.log_signal.emit(
                        f"[扫描驱动] ⚠️ {inst_id} 扫描信号 {direction} 与现有 {existing_dir} 持仓方向相反！"
                        f"正在通知并平仓...",
                        "WARNING",
                    )
                    self.conflict_signal.emit(inst_id, direction, existing_dir)
                    self._close_conflicting_position(inst_id)
                    # 本轮不立即建新仓，等下次扫描确认入场
                    continue

            # ── 已在监控中 → 若 worker 空闲则复用，否则跳过 ─────────────────
            with self._workers_lock:
                existing_entry = self._workers.get(inst_id)
            if existing_entry and existing_entry[0].isRunning():
                _, worker = existing_entry
                if worker.is_idle():
                    self.log_signal.emit(
                        f"[扫描驱动] {inst_id} worker 空闲中，推送新信号复用",
                        "INFO",
                    )
                    worker.receive_signal(candidate)
                    launched += 1
                    continue
                self.log_signal.emit(
                    f"[扫描驱动] {inst_id} 已在监控中，跳过重复启动", "INFO"
                )
                continue

            # 统计当前实际持仓数（包括实盘和模拟）
            active_positions = self._count_active_positions()
            if active_positions >= max_pos:
                self.log_signal.emit(
                    f"[扫描驱动] {inst_id} 符合条件但已达最大持仓数 {max_pos}（当前={active_positions}），跳过",
                    "INFO",
                )
                continue

            signals_list = res.get('signals', [])
            reason = (', '.join(signals_list[:3]) if isinstance(signals_list, list)
                      else str(res.get('reason', res.get('priority_reason', ''))))

            candidate = {
                'inst_id': inst_id,
                'direction': direction,
                'score': score,
                'reason': reason,
                'entry_price': float(res.get('entry_price', 0) or 0),
            }
            self._launch_worker(candidate)
            running_count += 1
            launched += 1

        if launched > 0:
            self.log_signal.emit(
                f"[扫描驱动] 本轮新启动 {launched} 个工作器，当前监控中 {running_count} 个",
                "SUCCESS",
            )
        else:
            self.log_signal.emit(
                f"[扫描驱动] 本轮无新交易对符合条件（共过滤 {len(results)} 条结果）",
                "INFO",
            )

    def _get_existing_position_direction(self, inst_id: str) -> str:
        """
        查询该交易对是否已有持仓（手动交易 / 自动交易 / 扫描驱动 均算）。

        返回 'LONG' / 'SHORT'，无持仓返回空字符串。
        优先以交易所实际持仓为准；模拟引擎从 _positions 读取。
        """
        try:
            pos_map = self.trade_executor.get_positions(inst_id)
            pos = pos_map.get(inst_id) if pos_map else None
            if pos and getattr(pos, 'size', 0) and float(pos.size or 0) > 0:
                side = getattr(pos, 'side', None)
                if side == PositionSide.LONG:
                    return 'LONG'
                if side == PositionSide.SHORT:
                    return 'SHORT'
        except Exception as e:
            print(f"[扫描驱动] 查询 {inst_id} 持仓失败: {e}")
        return ''

    def _close_conflicting_position(self, inst_id: str):
        """
        平掉与扫描信号方向相反的持仓：
        1. 无条件释放注册器（不论持有者是谁）
        2. 先停止该交易对的扫描驱动 worker（如有）
        3. 再通过 trade_executor 执行平仓
        """
        # 强制接管注册器（反向信号说明原持仓已失效，不论哪个系统建立的）
        # force_takeover 会通知原 owner 清理内部状态机，防止状态错乱
        position_registry.force_takeover(inst_id, 'ScanDriven')

        # 停止该 inst_id 的 worker（避免 worker 在平仓后继续管理已不存在的持仓）
        with self._workers_lock:
            entry = self._workers.get(inst_id)
        if entry:
            try:
                thread, worker = entry
                worker.stop()
                thread.quit()
                self._safe_discard_worker(thread, wait_ms=500)
            except Exception:
                pass
            with self._workers_lock:
                self._workers.pop(inst_id, None)

        # 执行平仓
        try:
            result = self.trade_executor.execute_stop_loss(inst_id)
            if result.success:
                self.log_signal.emit(
                    f"[扫描驱动] ✅ {inst_id} 反向信号平仓成功：{result.message}",
                    "SUCCESS",
                )
            else:
                self.log_signal.emit(
                    f"[扫描驱动] ❌ {inst_id} 反向信号平仓失败：{result.message}",
                    "ERROR",
                )
        except Exception as e:
            self.log_signal.emit(
                f"[扫描驱动] ❌ {inst_id} 平仓异常：{e}",
                "ERROR",
            )

    def _on_takeover(self, inst_id: str, old_owner: str, new_owner: str):
        """当其他系统强制接管我们的标的时，停止对应 worker 并释放资金池。"""
        if old_owner != 'ScanDriven':
            return
        self.log_signal.emit(
            f"[扫描驱动] ⚠️ {inst_id} 被 {new_owner} 强制接管，停止本地 worker",
            "WARNING",
        )
        with self._workers_lock:
            entry = self._workers.get(inst_id)
        if entry:
            thread, worker = entry
            # 释放 worker 占用的资金池
            if hasattr(worker, '_campaign') and worker._campaign:
                _pool: Optional[CapitalPool] = self.config.get('_capital_pool')
                _alloc = float(getattr(worker._campaign, 'total_allocated_usdt', 0) or 0)
                if _pool and _alloc > 0:
                    _pool.release(_alloc)
                    self.log_signal.emit(
                        f"[资金池] {inst_id} 被接管后释放 {_alloc:.2f} USDT",
                        "INFO",
                    )
            worker.stop()
            thread.quit()
            self._safe_discard_worker(thread, wait_ms=500)
            with self._workers_lock:
                self._workers.pop(inst_id, None)
            self._states.pop(inst_id, None)
            self.position_closed.emit(inst_id)

    def _count_active_positions(self) -> int:
        """统计当前有实际持仓（campaign 已开仓）的 worker 数量。"""
        count = 0
        with self._workers_lock:
            workers = list(self._workers.values())
        for _, worker in workers:
            if worker._campaign is not None:
                count += 1
        return count

    def _launch_worker(self, candidate: dict):
        inst_id = candidate['inst_id']

        # ── 注册器前置检查：防止与其他系统（TradingAssistant/MultiAgent）冲突 ──
        # 注意：此时仅预占位（监控中），真正开仓时 _try_open_pilot_position_scan
        # 会再次确认并锁定；若 try_lock 失败说明已被其他 AI 系统占用。
        if not position_registry.try_lock(inst_id, 'ScanDriven'):
            _owner = position_registry.get_owner(inst_id)
            self.log_signal.emit(
                f"[扫描驱动] {inst_id} 注册器已被 {_owner} 占用，跳过启动 Worker",
                "WARNING",
            )
            return

        worker_config = dict(self.config)
        worker_config['_capital_pool'] = self._capital_pool
        worker_config['_dynamic_risk'] = self.dynamic_risk  # 注入动态风控

        # ── 风险预算仓位计算 ──────────────────────────────────────────────
        try:
            pool_avail = self._capital_pool.available
            entry_px = float(candidate.get('entry_price', 0) or 0)
            # 从 klines_map 估算 ATR%
            atr_pct = _estimate_atr_pct_from_candidate(candidate)
            if pool_avail > 0 and entry_px > 0 and atr_pct > 0:
                sizing = calculate_position(
                    capital=pool_avail, entry_price=entry_px,
                    atr_pct=atr_pct,
                    risk_pct=float(self.config.get('auto_risk_per_trade_pct', 0.01) or 0.01),
                    max_position_pct=float(self.config.get('max_position_pct', 0.50) or 0.50),
                )
                if sizing.get('acceptable'):
                    candidate['_risk_budget_size'] = sizing
                    worker_config['_risk_budget_size'] = sizing
                    self.log_signal.emit(
                        f"[风控] {inst_id} 风险预算仓位={sizing['position_usdt']:.0f}USDT"
                        f"({sizing['position_pct']*100:.1f}%) ATR%={atr_pct:.1f}%",
                        "INFO",
                    )
        except Exception:
            pass

        worker = ScanCampaignWorker(
            candidate=candidate,
            okx_client=self.okx_client,
            trade_executor=self.trade_executor,
            config=worker_config,
        )
        thread = QThread()
        worker.moveToThread(thread)

        worker.log_signal.connect(self.log_signal)
        worker.campaign_state_signal.connect(self._on_worker_state)

        # worker.finished 是自定义 Signal()，从 run() 内部 emit。
        # 只用它通知线程事件循环退出（quit），不在这里做任何 Python 引用操作。
        worker.finished.connect(thread.quit)

        # ⚠️ 关键：所有需要"丢弃 Python 引用"的操作必须挂到 thread.finished（Qt 内置
        # 信号，OS 线程完全退出后才发出），而不是自定义 worker.finished（那时 run()
        # 尚未 return，线程仍在运行）。
        #
        # 之前写法：
        #   worker.finished.connect(lambda _id: self.worker_finished_signal.emit(_id))
        # → _on_worker_finished 在 run() 内发出信号时立即被主线程调用
        # → _workers.pop(inst_id) 丢弃 (thread, worker) 的最后 Python 引用
        # → Python GC 在主线程销毁 QThread，而 OS 线程仍在运行 → SIGABRT
        thread.finished.connect(lambda _id=inst_id: self.worker_finished_signal.emit(_id))
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.started.connect(worker.run)

        with self._workers_lock:
            self._workers[inst_id] = (thread, worker)
        thread.start()

        self.log_signal.emit(
            f"[扫描驱动] ▶ 启动 {inst_id} {candidate['direction']} 工作器"
            f"  评分={candidate['score']:.1f}  {candidate['reason'][:50]}",
            "TRADE",
        )

    # ── worker 回调 ──────────────────────────────────────────────────────

    def _on_worker_state(self, inst_id: str, state: dict):
        prev = self._states.get(inst_id, {})
        self._states[inst_id] = state
        # 检测开仓/平仓事件
        if not prev.get('stage') and state.get('stage') == '已开仓':
            self.position_opened.emit(inst_id, state.get('direction', ''))
        self.state_updated.emit(list(self._states.values()))

    def _on_worker_finished(self, inst_id: str):
        if inst_id in self._states:
            # 更新全局连续亏损计数
            pnl = float(self._states[inst_id].get('realized_pnl', 0) or 0)
            if pnl < 0:
                self._global_consecutive_losses += 1
                self.log_signal.emit(
                    f"[风控] {inst_id} 亏损 {pnl:+.2f}，全局连续亏损 {self._global_consecutive_losses} 次",
                    "WARNING",
                )
            else:
                self._global_consecutive_losses = 0
            self._states[inst_id]['stage'] = '已完成'
        with self._workers_lock:
            self._workers.pop(inst_id, None)
        # Worker 退出即释放注册器，允许其他系统或下一轮扫描重新接管该标的
        position_registry.release(inst_id, 'ScanDriven')
        self.state_updated.emit(list(self._states.values()))
        self.log_signal.emit(f"[扫描驱动] ■ {inst_id} 工作器退出，注册器已释放", "INFO")
        self.position_closed.emit(inst_id)

    def _safe_discard_worker(self, thread: QThread, wait_ms: int = 500):
        """
        安全丢弃 worker 线程引用。
        若线程在 wait_ms 内未退出，暂存僵尸列表保持 Python 引用，
        防止 GC 在 OS 线程仍运行时销毁 QThread 对象（→ SIGABRT）。
        """
        if thread is None:
            return
        if thread.isRunning():
            thread.wait(wait_ms)
        if thread.isRunning():
            # 仍在运行，放入僵尸列表
            if thread not in self._zombie_threads:
                self._zombie_threads.append(thread)
                thread.finished.connect(
                    lambda t=thread: (
                        self._zombie_threads.remove(t)
                        if t in self._zombie_threads else None
                    )
                )

    # ── 控制 ─────────────────────────────────────────────────────────────

    def stop_all(self):
        """停止所有工作器并批量释放注册器。"""
        with self._workers_lock:
            workers = list(self._workers.values())
        for thread, worker in workers:
            try:
                worker.stop()
                thread.quit()
                self._safe_discard_worker(thread, wait_ms=500)
            except Exception:
                pass
        with self._workers_lock:
            self._workers.clear()
        released = position_registry.release_by_system('ScanDriven')
        self.log_signal.emit(
            f"[扫描驱动] 所有工作器已停止，注册器释放 {released} 个标的",
            "WARNING",
        )

    def stop_one(self, inst_id: str):
        """手动停止指定交易对的工作器。"""
        with self._workers_lock:
            entry = self._workers.get(inst_id)
        if not entry:
            return
        thread, worker = entry
        try:
            worker.stop()
            thread.quit()
            self._safe_discard_worker(thread, wait_ms=500)
        except Exception:
            pass
        with self._workers_lock:
            self._workers.pop(inst_id, None)
        self.log_signal.emit(f"[扫描驱动] 已手动停止 {inst_id}", "WARNING")

    @property
    def is_active(self) -> bool:
        with self._workers_lock:
            workers = list(self._workers.values())
        return any(t.isRunning() for t, _ in workers)

    def _extract_btc_context_from_results(self, results: list) -> Optional[Dict]:
        """从扫描结果中提取 BTC 1H 涨跌（如果存在）"""
        for r in results:
            d = r.get("details", {}) if isinstance(r.get("details"), dict) else {}
            btc_env = d.get("BTC环境", "")
            if "BTC" in btc_env:
                try:
                    import re
                    m = re.search(r'([+-]?\d+\.?\d*)%', btc_env)
                    if m:
                        return {"btc_1h_pct": float(m.group(1))}
                except Exception:
                    pass
        return None

    def active_summary(self) -> dict:
        total = len(self._workers)
        with_pos = self._count_active_positions()
        return {
            'total_monitoring': total,
            'with_position':    with_pos,
            'capital_pool':     self._capital_pool.snapshot(),
        }


def _estimate_atr_pct_from_candidate(candidate: dict) -> float:
    """从候选信号中估算 ATR%（优先 details.ATR% → 无则用 score 推断）"""
    details = candidate.get('details', {}) if isinstance(candidate.get('details'), dict) else {}
    atr_str = str(details.get('ATR%', details.get('atr_pct', '')))
    try:
        atr_pct = float(atr_str.replace('%', '').strip())
        if atr_pct > 0:
            return atr_pct
    except (ValueError, AttributeError):
        pass
    # Fallback: 从 score 反推（高分次品种波动大，低分稳定币波动小）
    score = float(candidate.get('score', 50))
    return 0.5 + (score / 100.0) * 4.5  # 1%~5% ATR 范围
