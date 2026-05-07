"""
风控守卫引擎：全方位监控交易风险，自动弹窗告警。

监控维度：
  • 持仓风险  —— 浮亏/杠杆/集中度/爆仓距离
  • 资金风险  —— 余额骤降/可用率不足/总名义值超限
  • 市场风险  —— 资金费率极端/价格剧烈波动/趋势反转
  • 策略健康  —— 胜率骤降/信号异常/连续亏损
  • 系统健康  —— API 延迟/数据断流/线程异常
"""

from __future__ import annotations

import time
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import json

try:
    from PySide6.QtCore import QThread, Signal
    _QT_AVAILABLE = True
except ImportError:
    _QT_AVAILABLE = False
    QThread = object
    def Signal(*a): return None


# ─── 告警数据结构 ─────────────────────────────────────────────────────────────

LEVEL_INFO     = "INFO"
LEVEL_WARNING  = "WARNING"
LEVEL_CRITICAL = "CRITICAL"
LEVEL_DANGER   = "DANGER"   # 最高级，强制弹窗，需确认


@dataclass
class RiskAlert:
    level: str              # INFO / WARNING / CRITICAL / DANGER
    category: str           # POSITION / CAPITAL / MARKET / STRATEGY / SYSTEM
    title: str
    message: str
    detail: str = ""        # 详细数据/建议操作
    symbol: str = ""        # 相关交易对（可空）
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    auto_close_sec: int = 0 # 0=不自动关闭；>0 秒后自动关闭（仅 INFO）
    suggested_action: str = ""


# ─── 风控规则配置 ──────────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    # 持仓风险
    "max_single_loss_pct":     -8.0,   # 单仓浮亏超此%→CRITICAL
    "max_total_loss_pct":      -5.0,   # 总持仓合计浮亏超此%→DANGER
    "max_leverage":            15.0,   # 杠杆超此→WARNING
    "max_leverage_danger":     20.0,   # 杠杆超此→CRITICAL
    "liq_distance_pct_warn":   10.0,   # 距爆仓价不足此%→WARNING
    "liq_distance_pct_crit":   5.0,    # 距爆仓价不足此%→CRITICAL
    "max_position_concentration": 0.6, # 单一方向占总名义值比例上限

    # 资金风险
    "min_available_ratio":     0.15,   # 可用余额占总权益低于此→WARNING
    "min_available_ratio_crit": 0.05,  # 可用余额占总权益低于此→CRITICAL
    "balance_drop_pct_warn":   5.0,    # 余额单周期跌幅%→WARNING
    "balance_drop_pct_crit":   10.0,   # 余额单周期跌幅%→CRITICAL
    "max_total_notional_ratio": 3.0,   # 总名义值/余额超此→WARNING

    # 市场风险
    "extreme_funding_rate":    0.003,  # 资金费率绝对值超此→WARNING
    "extreme_funding_danger":  0.006,  # 资金费率绝对值超此→CRITICAL
    "price_spike_pct":         8.0,    # 持仓品种 1h 波动超此%→WARNING

    # 策略健康
    "min_win_rate_warn":       40.0,   # 7日胜率低于此→WARNING
    "min_win_rate_crit":       30.0,   # 7日胜率低于此→CRITICAL
    "max_consecutive_losses":  5,      # 连续亏损次数→WARNING
    "signal_silence_hours":    6,      # 策略超过此小时无信号→WARNING

    # 系统健康
    "api_latency_warn_ms":     2000,   # API 延迟超此→WARNING
    "api_latency_crit_ms":     5000,   # API 延迟超此→CRITICAL
    "check_interval_sec":      30,     # 巡检间隔（秒）

    # ── P1 新增：DANGER 级别告警自动平仓 ────────────────────────────────────
    # 默认关闭（False）。开启后，凡触发 DANGER 告警（总持仓浮亏超限等）
    # 风控守卫将自动平掉所有持仓，无需人工干预。
    # 建议仅在无人值守且风险承受极低的场景开启。
    "auto_close_on_danger":    False,

    # DANGER 同类告警自动平仓冷却时间（秒），防止同一告警触发多次平仓
    "danger_close_cooldown_sec": 120,
}


# ─── 风控守卫主类 ──────────────────────────────────────────────────────────────

class RiskGuard(QThread if _QT_AVAILABLE else object):
    """
    风控守卫后台线程。

    每隔 check_interval_sec 秒执行全面巡检，发现风险时：
      - INFO/WARNING  → 写入告警日志，通过 alert_signal 通知 UI 显示横幅
      - CRITICAL      → 触发弹窗，需用户点击确认
      - DANGER        → 触发强制弹窗 + 声音（建议立即操作）
    """

    if _QT_AVAILABLE:
        alert_signal   = Signal(object)   # RiskAlert
        metrics_signal = Signal(dict)     # 实时指标快照（供 UI Dashboard 刷新）

    def __init__(
        self,
        okx_client=None,
        executor=None,
        tracker=None,
        config: Optional[Dict] = None,
        log_callback: Optional[Callable[[str, str], None]] = None,
    ):
        if _QT_AVAILABLE:
            super().__init__()
        self.okx_client = okx_client
        self.executor   = executor    # TradeExecutor 实例
        self.tracker    = tracker     # 信号追踪器
        self.cfg        = {**DEFAULT_CONFIG, **(config or {})}
        self._log       = log_callback or (lambda msg, lvl: None)
        self._stop_flag = False

        # 状态记忆（用于检测变化）
        self._last_balance: float = 0.0
        self._last_check_ts: float = 0.0
        self._alert_history: List[Dict] = []
        self._suppressed: Dict[str, float] = {}  # category+title → 上次告警时间（防刷屏）
        self._suppress_sec = 300   # 同类告警至少间隔 5 分钟
        self._last_danger_close_ts: float = 0.0  # 上次 DANGER 自动平仓时间戳（防重复）

        self._data_dir = Path(__file__).resolve().parent.parent.parent / "data"
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._alert_log_path = self._data_dir / "risk_alerts.json"
        self._load_history()

    # ─── 启动/停止 ────────────────────────────────────────────────────────────

    def stop(self):
        self._stop_flag = True

    def run(self):
        self._log("🛡️ 风控守卫已启动", "INFO")
        interval = max(10, int(self.cfg.get("check_interval_sec", 30)))
        while not self._stop_flag:
            try:
                self._run_all_checks()
            except Exception as e:
                self._log(f"⚠️ 风控巡检异常: {e}", "ERROR")
            for _ in range(interval * 2):
                if self._stop_flag:
                    break
                time.sleep(0.5)
        self._log("🛡️ 风控守卫已停止", "INFO")

    # ─── 全量巡检 ─────────────────────────────────────────────────────────────

    def _run_all_checks(self):
        alerts = []
        metrics = {}

        # 1. 持仓风险
        pos_alerts, pos_metrics = self._check_positions()
        alerts.extend(pos_alerts)
        metrics["positions"] = pos_metrics

        # 2. 资金风险
        cap_alerts, cap_metrics = self._check_capital()
        alerts.extend(cap_alerts)
        metrics["capital"] = cap_metrics

        # 3. 市场风险（依赖持仓列表）
        if pos_metrics.get("positions"):
            mkt_alerts = self._check_market(pos_metrics["positions"])
            alerts.extend(mkt_alerts)

        # 4. 策略健康
        strat_alerts, strat_metrics = self._check_strategies()
        alerts.extend(strat_alerts)
        metrics["strategies"] = strat_metrics

        # 5. 系统健康
        sys_alerts, sys_metrics = self._check_system()
        alerts.extend(sys_alerts)
        metrics["system"] = sys_metrics

        metrics["timestamp"] = datetime.now().strftime("%H:%M:%S")
        metrics["alert_count_total"] = len(self._alert_history)

        # 发送指标快照
        if _QT_AVAILABLE:
            self.metrics_signal.emit(metrics)

        # 发送告警
        for alert in alerts:
            self._dispatch_alert(alert)

    # ─── 持仓风险 ──────────────────────────────────────────────────────────────

    def _check_positions(self):
        alerts = []
        metrics = {"positions": [], "total_pnl_pct": 0.0, "max_loss_pct": 0.0}
        if not self.executor:
            return alerts, metrics
        try:
            positions = self.executor.get_active_positions()
            metrics["positions"] = [
                {
                    "inst_id": p.inst_id, "side": p.side.value,
                    "pnl_pct": round(p.pnl_percent, 2),
                    "leverage": p.leverage,
                    "notional_usd": round(p.notional_usd, 1),
                }
                for p in positions
            ]
            if not positions:
                return alerts, metrics

            total_pnl = sum(p.unrealized_pnl for p in positions)
            total_notional = sum(p.notional_usd for p in positions) or 1

            balance = self.executor.get_usdt_balance() or 1
            total_pnl_pct = total_pnl / balance * 100
            metrics["total_pnl_pct"] = round(total_pnl_pct, 2)

            long_notional  = sum(p.notional_usd for p in positions if p.side.value == "long")
            short_notional = sum(p.notional_usd for p in positions if p.side.value == "short")
            concentration  = max(long_notional, short_notional) / total_notional
            metrics["concentration"] = round(concentration, 2)

            if concentration > self.cfg["max_position_concentration"]:
                alerts.append(RiskAlert(
                    level=LEVEL_WARNING, category="POSITION",
                    title="持仓方向集中",
                    message=f"单一方向占比 {concentration:.0%}，超过 {self.cfg['max_position_concentration']:.0%} 上限",
                    detail=f"多头:{long_notional:.0f}$ 空头:{short_notional:.0f}$",
                    suggested_action="建议适当分散持仓方向，降低方向风险",
                ))

            for p in positions:
                pnl = p.pnl_percent
                metrics["max_loss_pct"] = min(metrics["max_loss_pct"], pnl)

                # 单仓浮亏
                if pnl <= self.cfg["max_single_loss_pct"]:
                    lvl = LEVEL_CRITICAL if pnl <= self.cfg["max_single_loss_pct"] * 1.5 else LEVEL_WARNING
                    alerts.append(RiskAlert(
                        level=lvl, category="POSITION", symbol=p.inst_id,
                        title=f"单仓浮亏过大",
                        message=f"{p.inst_id} 浮亏 {pnl:.2f}%（{p.side.value.upper()}）",
                        detail=f"未实现亏损: {p.unrealized_pnl:.2f} USDT | 开仓价: {p.entry_price:.4f}",
                        suggested_action="考虑止损或减仓以控制风险",
                    ))

                # 杠杆过高
                if p.leverage >= self.cfg["max_leverage_danger"]:
                    alerts.append(RiskAlert(
                        level=LEVEL_CRITICAL, category="POSITION", symbol=p.inst_id,
                        title="杠杆倍数危险",
                        message=f"{p.inst_id} 当前杠杆 {p.leverage:.0f}x，已超过危险阈值 {self.cfg['max_leverage_danger']:.0f}x",
                        suggested_action="立即降低杠杆倍数，避免爆仓",
                    ))
                elif p.leverage >= self.cfg["max_leverage"]:
                    alerts.append(RiskAlert(
                        level=LEVEL_WARNING, category="POSITION", symbol=p.inst_id,
                        title="杠杆倍数偏高",
                        message=f"{p.inst_id} 杠杆 {p.leverage:.0f}x",
                        suggested_action="建议将杠杆控制在 10x 以内",
                    ))

            # 总浮亏
            if total_pnl_pct <= self.cfg["max_total_loss_pct"]:
                lvl = LEVEL_DANGER if total_pnl_pct <= self.cfg["max_total_loss_pct"] * 2 else LEVEL_CRITICAL
                alerts.append(RiskAlert(
                    level=lvl, category="POSITION",
                    title="总持仓浮亏超限",
                    message=f"所有持仓合计浮亏 {total_pnl_pct:.2f}%，已触及风控红线",
                    detail=f"总浮亏: {total_pnl:.2f} USDT | 持仓数: {len(positions)}",
                    suggested_action="⚠️ 建议立即评估是否全部或部分平仓止损",
                ))

        except Exception as e:
            self._log(f"持仓检查异常: {e}", "ERROR")
        return alerts, metrics

    # ─── 资金风险 ──────────────────────────────────────────────────────────────

    def _check_capital(self):
        alerts = []
        metrics = {"balance": 0.0, "available_ratio": 1.0}
        if not self.executor:
            return alerts, metrics
        try:
            balance = self.executor.get_usdt_balance()
            if balance <= 0:
                return alerts, metrics
            metrics["balance"] = round(balance, 2)

            # 余额骤降检测
            if self._last_balance > 0:
                drop_pct = (self._last_balance - balance) / self._last_balance * 100
                if drop_pct >= self.cfg["balance_drop_pct_crit"]:
                    alerts.append(RiskAlert(
                        level=LEVEL_CRITICAL, category="CAPITAL",
                        title="账户余额骤降",
                        message=f"余额从 {self._last_balance:.2f} 降至 {balance:.2f} USDT（-{drop_pct:.1f}%）",
                        suggested_action="检查是否有异常交易或爆仓",
                    ))
                elif drop_pct >= self.cfg["balance_drop_pct_warn"]:
                    alerts.append(RiskAlert(
                        level=LEVEL_WARNING, category="CAPITAL",
                        title="账户余额下降",
                        message=f"本轮余额下降 {drop_pct:.1f}%（{self._last_balance:.2f} → {balance:.2f} USDT）",
                    ))
            self._last_balance = balance

            # 总名义值/余额比率
            if self.executor:
                try:
                    total_notional = self.executor.get_total_position_notional()
                    ratio = total_notional / balance if balance > 0 else 0
                    metrics["notional_ratio"] = round(ratio, 2)
                    if ratio > self.cfg["max_total_notional_ratio"]:
                        alerts.append(RiskAlert(
                            level=LEVEL_WARNING, category="CAPITAL",
                            title="总仓位名义值过高",
                            message=f"总名义值 {total_notional:.0f} USDT = 余额的 {ratio:.1f}x",
                            suggested_action="建议减仓，控制整体风险敞口",
                        ))
                except Exception:
                    pass

        except Exception as e:
            self._log(f"资金检查异常: {e}", "ERROR")
        return alerts, metrics

    # ─── 市场风险 ──────────────────────────────────────────────────────────────

    def _check_market(self, positions: List[Dict]):
        alerts = []
        if not self.okx_client or not positions:
            return alerts
        try:
            for pos in positions:
                inst_id = pos.get("inst_id", "")
                if not inst_id or not inst_id.endswith("-SWAP"):
                    continue

                # 资金费率
                try:
                    fr = self.okx_client.get_funding_rate(inst_id)
                    if fr and fr.get("code") == "0" and fr.get("data"):
                        rate = abs(float(fr["data"][0].get("fundingRate", 0) or 0))
                        if rate >= self.cfg["extreme_funding_danger"]:
                            alerts.append(RiskAlert(
                                level=LEVEL_CRITICAL, category="MARKET", symbol=inst_id,
                                title="资金费率极端异常",
                                message=f"{inst_id} 资金费率 {rate*100:.4f}%，极端偏离",
                                detail="极端资金费率意味着市场情绪极度偏向单边",
                                suggested_action="评估持仓方向是否与主流方向相反，考虑平仓规避",
                            ))
                        elif rate >= self.cfg["extreme_funding_rate"]:
                            alerts.append(RiskAlert(
                                level=LEVEL_WARNING, category="MARKET", symbol=inst_id,
                                title="资金费率偏高",
                                message=f"{inst_id} 资金费率 {rate*100:.4f}%",
                                auto_close_sec=15,
                            ))
                except Exception:
                    pass

                # 价格剧烈波动（通过 24h 高低幅）
                try:
                    ticker = self.okx_client.get_ticker(inst_id)
                    if ticker and ticker.get("code") == "0" and ticker.get("data"):
                        d = ticker["data"][0]
                        high = float(d.get("high24h") or 0)
                        low  = float(d.get("low24h") or 1)
                        if low > 0:
                            swing = (high - low) / low * 100
                            if swing > self.cfg["price_spike_pct"] * 2:
                                alerts.append(RiskAlert(
                                    level=LEVEL_WARNING, category="MARKET", symbol=inst_id,
                                    title="价格剧烈波动",
                                    message=f"{inst_id} 24h 振幅 {swing:.1f}%",
                                    suggested_action="高波动环境下建议收紧止损或缩小仓位",
                                    auto_close_sec=20,
                                ))
                except Exception:
                    pass

        except Exception as e:
            self._log(f"市场检查异常: {e}", "ERROR")
        return alerts

    # ─── 策略健康 ──────────────────────────────────────────────────────────────

    def _check_strategies(self):
        alerts = []
        metrics = {}
        if not self.tracker:
            return alerts, metrics
        try:
            strategies = self.tracker.all_strategies() or []
            for name in strategies:
                stats = self.tracker.strategy_stats(name, days=7)
                wr    = stats.get("win_rate", 50)
                total = stats.get("total", 0)
                if total < 5:
                    continue
                metrics[name] = {"win_rate": wr, "total": total}

                if wr <= self.cfg["min_win_rate_crit"]:
                    alerts.append(RiskAlert(
                        level=LEVEL_CRITICAL, category="STRATEGY",
                        title="策略胜率严重下滑",
                        message=f"策略「{name}」近7日胜率仅 {wr:.1f}%，低于 {self.cfg['min_win_rate_crit']}% 红线",
                        detail=f"信号总数: {total}",
                        suggested_action="建议暂停该策略，触发 AI 分析并优化参数",
                    ))
                elif wr <= self.cfg["min_win_rate_warn"]:
                    alerts.append(RiskAlert(
                        level=LEVEL_WARNING, category="STRATEGY",
                        title="策略胜率偏低",
                        message=f"策略「{name}」近7日胜率 {wr:.1f}%",
                        suggested_action="建议启动 AI 分析优化",
                        auto_close_sec=30,
                    ))

                # 连续亏损检测
                try:
                    recent = self.tracker.strategy_recent_signals(name, limit=10)
                    consecutive = 0
                    for sig in reversed(recent):
                        if sig.get("validated") and float(sig.get("net_pnl", 0) or 0) < 0:
                            consecutive += 1
                        else:
                            break
                    if consecutive >= self.cfg["max_consecutive_losses"]:
                        alerts.append(RiskAlert(
                            level=LEVEL_WARNING, category="STRATEGY",
                            title="策略连续亏损",
                            message=f"策略「{name}」已连续亏损 {consecutive} 次",
                            suggested_action="建议暂停策略，等待市场条件改善",
                        ))
                except Exception:
                    pass

        except Exception as e:
            self._log(f"策略健康检查异常: {e}", "ERROR")
        return alerts, metrics

    # ─── 系统健康 ──────────────────────────────────────────────────────────────

    def _check_system(self):
        alerts = []
        metrics = {}
        if not self.okx_client:
            return alerts, metrics
        try:
            t0 = time.time()
            result = self.okx_client.get_tickers(instType="SWAP")
            latency_ms = int((time.time() - t0) * 1000)
            metrics["api_latency_ms"] = latency_ms
            metrics["api_ok"] = result.get("code") == "0"

            if not metrics["api_ok"]:
                alerts.append(RiskAlert(
                    level=LEVEL_CRITICAL, category="SYSTEM",
                    title="OKX API 连接异常",
                    message=f"行情接口返回错误: {result.get('msg', '未知')}",
                    suggested_action="检查网络连接和 API 密钥配置",
                ))
            elif latency_ms >= self.cfg["api_latency_crit_ms"]:
                alerts.append(RiskAlert(
                    level=LEVEL_CRITICAL, category="SYSTEM",
                    title="API 响应严重迟缓",
                    message=f"行情接口延迟 {latency_ms}ms，超过 {self.cfg['api_latency_crit_ms']}ms",
                    suggested_action="检查网络状况，考虑切换代理",
                ))
            elif latency_ms >= self.cfg["api_latency_warn_ms"]:
                alerts.append(RiskAlert(
                    level=LEVEL_WARNING, category="SYSTEM",
                    title="API 响应偏慢",
                    message=f"行情接口延迟 {latency_ms}ms",
                    auto_close_sec=10,
                ))

            import psutil
            mem = psutil.virtual_memory()
            metrics["memory_used_pct"] = round(mem.percent, 1)
            if mem.percent > 90:
                alerts.append(RiskAlert(
                    level=LEVEL_WARNING, category="SYSTEM",
                    title="系统内存不足",
                    message=f"内存占用 {mem.percent:.0f}%",
                    suggested_action="建议关闭不必要的程序释放内存",
                ))

        except ImportError:
            pass
        except Exception as e:
            self._log(f"系统健康检查异常: {e}", "ERROR")
        return alerts, metrics

    # ─── 告警分发 ──────────────────────────────────────────────────────────────

    def _dispatch_alert(self, alert: RiskAlert):
        """防重复 + 发送告警 + DANGER 自动平仓（可选）"""
        key = f"{alert.category}:{alert.title}"
        now = time.time()
        last = self._suppressed.get(key, 0)

        # DANGER 告警每次都发；其他同类间隔 suppress_sec
        if alert.level != LEVEL_DANGER and now - last < self._suppress_sec:
            return

        self._suppressed[key] = now
        self._alert_history.append({
            "time": alert.timestamp, "level": alert.level,
            "category": alert.category, "title": alert.title,
            "message": alert.message,
        })
        self._save_history()
        self._log(f"[{alert.level}] {alert.title}: {alert.message}", alert.level)

        if _QT_AVAILABLE:
            self.alert_signal.emit(alert)

        # ── DANGER 级别：可选自动平仓 ────────────────────────────────────────
        if alert.level == LEVEL_DANGER and self.cfg.get("auto_close_on_danger", False):
            cooldown = float(self.cfg.get("danger_close_cooldown_sec", 120))
            if now - self._last_danger_close_ts < cooldown:
                self._log(
                    f"[风控守卫] DANGER 自动平仓冷却中（距上次 {now-self._last_danger_close_ts:.0f}s），跳过",
                    "WARNING",
                )
                return
            self._last_danger_close_ts = now
            self._execute_danger_close_all(alert)

    # ─── DANGER 自动平仓 ───────────────────────────────────────────────────────

    def _execute_danger_close_all(self, trigger_alert: RiskAlert):
        """
        DANGER 告警触发时自动平掉所有持仓。

        设计原则：
          • 全量平仓，不区分盈亏（极端风险下保本优先）
          • 每个持仓单独调用 execute_stop_loss，失败单独记录不影响其他
          • 平仓结果通过 _log 输出，同时追加到 alert_history

        只有 auto_close_on_danger=True 时才会被调用。
        """
        if not self.executor:
            self._log("[风控守卫] DANGER 自动平仓：无交易执行器，跳过", "ERROR")
            return

        try:
            positions = self.executor.get_active_positions()
        except Exception as e:
            self._log(f"[风控守卫] DANGER 自动平仓：获取持仓失败 {e}", "ERROR")
            return

        if not positions:
            self._log("[风控守卫] DANGER 自动平仓：当前无持仓，无需操作", "INFO")
            return

        self._log(
            f"[风控守卫] ⚠️ DANGER 自动平仓启动："
            f"触发原因={trigger_alert.title}，"
            f"共 {len(positions)} 个持仓将被平掉",
            "CRITICAL",
        )

        success_count = 0
        fail_count    = 0
        for pos in positions:
            inst_id = pos.inst_id
            try:
                result = self.executor.execute_stop_loss(inst_id)
                if result and result.success:
                    success_count += 1
                    self._log(
                        f"[风控守卫] ✅ {inst_id} 自动平仓成功"
                        f"（持仓方向={pos.side.value}，浮亏={pos.pnl_percent:.2f}%）",
                        "SUCCESS",
                    )
                    # 释放注册器（如果引入了 position_registry）
                    try:
                        from src.trading.position_registry import position_registry
                        position_registry.force_takeover(inst_id, 'RiskGuard')
                    except Exception:
                        pass
                else:
                    fail_count += 1
                    msg = getattr(result, 'message', '未知') if result else '执行器无响应'
                    self._log(f"[风控守卫] ❌ {inst_id} 自动平仓失败：{msg}", "ERROR")
            except Exception as e:
                fail_count += 1
                self._log(f"[风控守卫] ❌ {inst_id} 自动平仓异常：{e}", "ERROR")

        summary = (
            f"[风控守卫] DANGER 自动平仓完成：成功 {success_count} 个"
            f"{'，失败 ' + str(fail_count) + ' 个（请手动检查）' if fail_count else ''}"
        )
        self._log(summary, "WARNING" if fail_count else "SUCCESS")
        self._alert_history.append({
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "level": LEVEL_DANGER,
            "category": "AUTO_CLOSE",
            "title": "DANGER 自动平仓",
            "message": summary,
        })
        self._save_history()

    # ─── 持久化 ────────────────────────────────────────────────────────────────

    def _load_history(self):
        if self._alert_log_path.exists():
            try:
                self._alert_history = json.loads(
                    self._alert_log_path.read_text(encoding="utf-8")
                )[-200:]
            except Exception:
                self._alert_history = []

    def _save_history(self):
        try:
            self._alert_log_path.write_text(
                json.dumps(self._alert_history[-200:], indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )
        except Exception:
            pass

    # ─── 手动触发 ──────────────────────────────────────────────────────────────

    def run_once(self) -> List[RiskAlert]:
        """手动触发一次全量巡检，返回告警列表（不发信号）"""
        collected = []
        original_dispatch = self._dispatch_alert
        def capture(alert):
            collected.append(alert)
            original_dispatch(alert)
        self._dispatch_alert = capture
        try:
            self._run_all_checks()
        finally:
            self._dispatch_alert = original_dispatch
        return collected

    def get_history(self, n: int = 50) -> List[Dict]:
        return self._alert_history[-n:]

    def update_config(self, new_cfg: Dict):
        self.cfg.update(new_cfg)
