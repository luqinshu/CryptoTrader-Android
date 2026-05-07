"""
3分钟K线实时监控智能体：剧烈变动告警 + 企稳买入信号。
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from PySide6.QtCore import QThread, Signal


class KLineMonitor(QThread):
    """
    每 3 分钟拉取监控池中交易对的 3m K线，
    检测剧烈变动和企稳买入信号，弹窗提醒。
    """

    alert_signal = Signal(str, str, str)  # symbol, type, message
    buy_signal = Signal(str, float, str)  # symbol, price, reason

    # 阈值
    VIOLENT_CHANGE_PCT = 2.5       # 单根3m K线涨/跌幅 > 2.5% → 剧烈变动
    SPIKE_VOLUME_RATIO = 2.0       # 成交量 > 均量2倍 → 放量
    STABILIZE_MAX_ATR = 0.8        # 企稳: ATR < 0.8%
    STABILIZE_MIN_BARS = 3         # 企稳: 连续3根小波动K线
    STABILIZE_PULLBACK_PCT = 1.5   # 回撤幅度 > 1.5% 后企稳 → 买入信号

    def __init__(self, okx_client=None, monitor_pool=None, symbols: List[str] = None):
        super().__init__()
        self.okx = okx_client
        self.monitor_pool = monitor_pool
        self._symbols = symbols or []
        self._stop_flag = False
        self._last_alert: Dict[str, float] = {}  # symbol → 上次告警时间，避免重复

    def stop(self):
        self._stop_flag = True

    def run(self):
        while not self._stop_flag:
            try:
                self._check_all()
            except Exception as e:
                self.alert_signal.emit("系统", "error", f"监控异常: {e}")
            for _ in range(180):
                if self._stop_flag:
                    break
                self.msleep(1000)

    def _check_all(self):
        symbols = self._get_symbols()
        if not symbols:
            return

        for symbol in symbols[:8]:  # 最多8个，防止API超限
            try:
                klines = self._fetch_3m_klines(symbol, limit=20)
                if not klines or len(klines) < 10:
                    continue
                self._analyze(symbol, klines)
            except Exception:
                continue

    def _get_symbols(self) -> List[str]:
        if self._symbols:
            return self._symbols
        if self.monitor_pool:
            try:
                if hasattr(self.monitor_pool, 'get_active'):
                    items = self.monitor_pool.get_active()
                    return [getattr(i, 'inst_id', '') for i in items if getattr(i, 'inst_id', '')]
                if hasattr(self.monitor_pool, 'items'):
                    return [getattr(i, 'inst_id', '') for i in self.monitor_pool.items if getattr(i, 'inst_id', '')]
            except Exception:
                pass
        return []

    def _fetch_3m_klines(self, symbol: str, limit: int = 20) -> Optional[List[List]]:
        if not self.okx:
            return None
        try:
            resp = self.okx.get_history_kline(symbol, bar="3m", limit=limit)
            if resp.get("code") == "0" and resp.get("data"):
                return [list(row) for row in reversed(resp["data"])]
        except Exception:
            pass
        return None

    def _analyze(self, symbol: str, klines: List[List]):
        if len(klines) < 5:
            return

        closes = [float(k[4]) for k in klines]
        highs = [float(k[2]) for k in klines]
        lows = [float(k[3]) for k in klines]
        volumes = [float(k[5]) for k in klines]
        current = closes[-1]
        prev_close = closes[-2]

        # ── 剧烈变动检测 ──
        change_pct = (current / prev_close - 1) * 100
        avg_vol = sum(volumes[-6:-1]) / 5 if len(volumes) >= 6 else volumes[-1]

        # 条件1: 单根K线大幅涨跌
        if abs(change_pct) >= self.VIOLENT_CHANGE_PCT:
            direction = "急涨" if change_pct > 0 else "急跌"
            vol_ratio = volumes[-1] / max(avg_vol, 1e-9)
            vol_note = f"，放量{vol_ratio:.1f}x" if vol_ratio > self.SPIKE_VOLUME_RATIO else ""

            if self._should_alert(symbol):
                msg = f"{direction} {abs(change_pct):.2f}%{vol_note}\n价格: {current:.4f}\n时间: {datetime.now().strftime('%H:%M:%S')}"
                self.alert_signal.emit(symbol, "violent", msg)

        # 条件2: 放量但不涨 → 可能变盘
        vol_ratio = volumes[-1] / max(avg_vol, 1e-9)
        if vol_ratio > self.SPIKE_VOLUME_RATIO * 1.5 and abs(change_pct) < 0.5:
            if self._should_alert(symbol):
                msg = f"放量{vol_ratio:.1f}x但滞涨(变盘预警)\n价格: {current:.4f}"
                self.alert_signal.emit(symbol, "diverge", msg)

        # ── 企稳买入检测 ──
        if len(klines) >= self.STABILIZE_MIN_BARS + 3:
            # 先是回撤
            recent_high = max(highs[-8:-3]) if len(highs) >= 8 else max(highs[:-1])
            pullback = (recent_high / current - 1) * 100

            if pullback >= self.STABILIZE_PULLBACK_PCT:
                # 检查最后N根是否企稳
                last_bars = klines[-self.STABILIZE_MIN_BARS:]
                ranges = [(float(k[2]) - float(k[3])) / float(k[4]) * 100 for k in last_bars]
                avg_range = sum(ranges) / len(ranges) if ranges else 999

                if avg_range <= self.STABILIZE_MAX_ATR:
                    # 确认不是下降趋势
                    ema5 = self._ema(closes, 5)
                    if current >= ema5[-1] * 0.995:
                        if self._should_alert(symbol):
                            reason = (
                                f"回撤{pullback:.2f}%后企稳\n"
                                f"波动率: {avg_range:.2f}% (ATR以内)\n"
                                f"价格: {current:.4f}"
                            )
                            self.buy_signal.emit(symbol, current, reason)
                            self.alert_signal.emit(symbol, "buy_ready", reason)

    def _should_alert(self, symbol: str) -> bool:
        """同一币种30分钟内不重复告警"""
        now = time.time()
        last = self._last_alert.get(symbol, 0)
        if now - last < 1800:
            return False
        self._last_alert[symbol] = now
        return True

    @staticmethod
    def _ema(data, period):
        if len(data) < period:
            return [data[-1]] * len(data)
        alpha = 2.0 / (period + 1)
        result = [sum(data[:period]) / period]
        for v in data[period:]:
            result.append(alpha * v + (1 - alpha) * result[-1])
        return [result[0]] * (period - 1) + result
