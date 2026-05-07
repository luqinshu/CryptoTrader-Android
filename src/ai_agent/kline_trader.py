"""
K线 AI 交易智能体：读 K 线 → 技术分析 → 决策 → 自动交易。
"""

from __future__ import annotations

import json
import math
import time
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from src.qt_compat import QThread, Signal



class KLineAnalyzer:
    """K线技术分析引擎：趋势 / 动量 / 波动 / 形态 / 信号评分"""

    @staticmethod
    def analyze(klines: List[List]) -> Dict[str, Any]:
        """
        分析 K 线数据，返回完整诊断。
        klines: [[ts, open, high, low, close, vol], ...] 最新在最后
        """
        if len(klines) < 30:
            return {"error": "K线不足（需≥30根）", "signal": "HOLD", "confidence": 0}

        closes = [float(k[4]) for k in klines]
        highs = [float(k[2]) for k in klines]
        lows = [float(k[3]) for k in klines]
        volumes = [float(k[5]) for k in klines]
        current = closes[-1]

        half = len(closes) // 2
        result = {}

        # ── EMA 趋势 ──
        ema12 = KLineAnalyzer._ema(closes, 12)
        ema26 = KLineAnalyzer._ema(closes, 26)
        ema50 = KLineAnalyzer._ema(closes, 50)
        trend = "ranging"
        if current > ema12[-1] > ema26[-1] > ema50[-1]:
            trend = "strong_up"
        elif current > ema12[-1] > ema26[-1]:
            trend = "up"
        elif current < ema12[-1] < ema26[-1] < ema50[-1]:
            trend = "strong_down"
        elif current < ema12[-1] < ema26[-1]:
            trend = "down"
        result["trend"] = trend
        result["ema12"] = round(ema12[-1], 4)
        result["ema26"] = round(ema26[-1], 4)
        result["ema50"] = round(ema50[-1], 4)

        # ── 动量 MACD ──
        dif = [ema12[i] - ema26[i] for i in range(len(ema12))]
        dea = KLineAnalyzer._ema(dif, 9)
        macd_bar = 2 * (dif[-1] - dea[-1])
        result["macd"] = round(macd_bar, 4)
        result["macd_signal"] = "bullish" if dif[-1] > dea[-1] else "bearish"

        # ── RSI ──
        rsi = KLineAnalyzer._rsi(closes, 14)
        result["rsi"] = round(rsi, 1)
        result["rsi_state"] = "oversold" if rsi < 30 else ("overbought" if rsi > 70 else "neutral")

        # ── 布林带 ──
        bb = KLineAnalyzer._bollinger(closes, 20, 2)
        bb_width = (bb["upper"] - bb["lower"]) / bb["mid"] * 100
        bb_pos = (current - bb["lower"]) / max(bb["upper"] - bb["lower"], 1e-9) * 100
        result["bb_upper"] = round(bb["upper"], 4)
        result["bb_mid"] = round(bb["mid"], 4)
        result["bb_lower"] = round(bb["lower"], 4)
        result["bb_position"] = round(bb_pos, 1)
        result["bb_width"] = round(bb_width, 1)

        # ── 成交量分析 ──
        vol_ma20 = sum(volumes[-20:]) / 20
        vol_current = volumes[-1]
        vol_ratio = vol_current / max(vol_ma20, 1e-9)
        result["volume_ratio"] = round(vol_ratio, 2)
        result["volume_signal"] = "high" if vol_ratio > 1.5 else ("low" if vol_ratio < 0.5 else "normal")

        # ── 波动率 ATR ──
        atr = KLineAnalyzer._atr(highs, lows, closes, 14)
        atr_pct = atr / current * 100
        result["atr"] = round(atr, 4)
        result["atr_pct"] = round(atr_pct, 2)

        # ── 近 N 根涨跌幅 ──
        result["change_1h"] = round((current / closes[-2] - 1) * 100, 2) if len(closes) >= 2 else 0
        result["change_6h"] = round((current / closes[-6] - 1) * 100, 2) if len(closes) >= 6 else 0
        result["change_24h"] = round((current / closes[-24] - 1) * 100, 2) if len(closes) >= 24 else 0

        # ── 近期最高最低 ──
        high_20 = max(highs[-20:])
        low_20 = min(lows[-20:])
        result["high_20"] = round(high_20, 4)
        result["low_20"] = round(low_20, 4)
        result["dist_from_high"] = round((current / high_20 - 1) * 100, 2)
        result["dist_from_low"] = round((current / low_20 - 1) * 100, 2)

        # ── 综合评分 ──
        score = 50.0
        if trend in ("strong_up", "up"):
            score += 15
        elif trend in ("strong_down", "down"):
            score -= 15
        if result["macd_signal"] == "bullish":
            score += 10
        else:
            score -= 10
        if rsi < 30:
            score += 10  # 超卖反弹
        elif rsi > 70:
            score -= 10
        if vol_ratio > 1.5 and trend == "up":
            score += 5
        score = max(0, min(100, score))
        result["score"] = round(score, 1)

        # ── 交易信号 ──
        signal = "HOLD"
        if trend in ("strong_up", "up") and rsi < 65 and score >= 60:
            signal = "BUY"
        elif trend in ("strong_down", "down") and rsi > 35 and score <= 40:
            signal = "SELL"
        elif rsi < 25 and score >= 45:
            signal = "BUY"  # 超卖反弹
        elif rsi > 75 and score <= 50:
            signal = "SELL"  # 超买回落

        result["signal"] = signal
        result["confidence"] = abs(score - 50) / 50.0
        result["current_price"] = round(current, 4)

        return result

    @staticmethod
    def _ema(data, period):
        if len(data) < period:
            return [data[-1]] * len(data)
        alpha = 2.0 / (period + 1)
        result = [sum(data[:period]) / period]
        for v in data[period:]:
            result.append(alpha * v + (1 - alpha) * result[-1])
        return [result[0]] * (period - 1) + result

    @staticmethod
    def _rsi(closes, period=14):
        if len(closes) < period + 1:
            return 50.0
        gains = []
        losses = []
        for i in range(1, len(closes)):
            delta = closes[i] - closes[i - 1]
            gains.append(delta if delta > 0 else 0)
            losses.append(-delta if delta < 0 else 0)
        avg_gain = sum(gains[-period:]) / period if gains else 0.0
        avg_loss = sum(losses[-period:]) / period if losses else 0.0
        if avg_gain == 0 and avg_loss == 0:
            return 50.0  # 无波动 → 中性
        if avg_loss == 0:
            return 100.0
        # Wilder 平滑：用 ewm(alpha=1/period) 替代 SMA
        import pandas as pd
        g_s = pd.Series(gains[-period*2:])
        l_s = pd.Series(losses[-period*2:])
        avg_gain = float(g_s.ewm(alpha=1/period, adjust=False).mean().iloc[-1]) if len(g_s) > 0 else avg_gain
        avg_loss = float(l_s.ewm(alpha=1/period, adjust=False).mean().iloc[-1]) if len(l_s) > 0 else avg_loss
        if avg_loss == 0:
            return 100.0 if avg_gain > 0 else 50.0
        rs = avg_gain / avg_loss
        return 100.0 - 100.0 / (1.0 + rs)

    @staticmethod
    def _bollinger(closes, period=20, std_dev=2):
        if len(closes) < period:
            return {"upper": closes[-1], "mid": closes[-1], "lower": closes[-1]}
        recent = closes[-period:]
        mid = sum(recent) / period
        variance = sum((x - mid) ** 2 for x in recent) / period
        std = math.sqrt(variance)
        return {"upper": mid + std_dev * std, "mid": mid, "lower": mid - std_dev * std}

    @staticmethod
    def _atr(highs, lows, closes, period=14):
        if len(closes) < period + 1:
            return 0.0
        tr = []
        for i in range(1, len(closes)):
            tr.append(max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1])))
        return sum(tr[-period:]) / period


class KLineTrader(QThread):
    """
    K线 AI 交易智能体。

    循环（每 5 分钟）：
    1. 拉指定交易对的 K 线数据
    2. 技术分析（趋势/MACD/RSI/布林/成交量/ATR）
    3. 综合评分 → 生成 BUY/SELL/HOLD 信号
    4. 发送给 LLM 二次确认（如果已配置）
    5. 自动执行交易
    """

    log_signal = Signal(str, str)
    analysis_signal = Signal(dict)      # K 线分析结果
    trade_signal = Signal(str, str, float)  # symbol, side, size

    def __init__(self, okx_client=None, llm_client=None, trade_executor=None,
                 symbol: str = "BTC-USDT", bar: str = "5m", interval_sec: int = 300):
        super().__init__()
        self.okx = okx_client
        self.llm = llm_client
        self.executor = trade_executor
        self.symbol = symbol
        self.bar = bar
        self.interval_sec = interval_sec
        self._stop_flag = False
        self._position_size_pct = 0.05  # 5% per trade

    def stop(self):
        self._stop_flag = True

    def run(self):
        self.log_signal.emit(f"📊 K线智能体启动 [{self.symbol} {self.bar}]", "INFO")
        while not self._stop_flag:
            try:
                self._cycle()
            except Exception as e:
                self.log_signal.emit(f"⚠️ K线分析异常: {e}", "ERROR")
            for _ in range(self.interval_sec):
                if self._stop_flag:
                    break
                self.msleep(1000)

    def _cycle(self):
        # 1. 拉 K 线
        klines = self._fetch_klines(limit=80)
        if not klines:
            self.log_signal.emit(f"📊 {self.symbol}: 无 K 线数据", "INFO")
            return

        # 2. 技术分析
        result = KLineAnalyzer.analyze(klines)
        if "error" in result:
            self.log_signal.emit(f"📊 {self.symbol}: {result['error']}", "INFO")
            return

        self.analysis_signal.emit(result)

        signal = result["signal"]
        score = result["score"]
        conf = result["confidence"]

        self.log_signal.emit(
            f"📊 {self.symbol} | 趋势:{result['trend']} | RSI:{result['rsi']} | "
            f"MACD:{result['macd_signal']} | 评分:{score:.0f} | 信号:{signal} | 价格:{result['current_price']}",
            "SUCCESS" if signal != "HOLD" else "INFO",
        )

        if signal == "HOLD":
            return

        # 3. 风控检查
        if not self.executor:
            self.log_signal.emit(f"🤖 {symbol} {signal} (无执行器，仅分析)", "INFO")
            return

        try:
            balance = self.executor.get_usdt_balance()
        except Exception:
            balance = 0.0

        if balance < 10:
            self.log_signal.emit(f"⚠️ 余额不足 ${balance:.0f}，跳过执行", "WARNING")
            return

        # 4. LLM 二次确认
        if self.llm and conf > 0.3:
            approved = self._llm_confirm(result)
            if not approved:
                self.log_signal.emit(f"🤖 LLM 否决 {signal} 信号", "INFO")
                return

        # 5. 执行
        size = balance * self._position_size_pct
        try:
            if signal == "BUY":
                r = self.executor.execute_buy(self.symbol, position_ratio=self._position_size_pct)
            elif signal == "SELL":
                r = self.executor.execute_sell(self.symbol, size)
            else:
                return
            ok = getattr(r, 'success', False) if r else False
            if ok:
                self.log_signal.emit(f"✅ 已{signal} {self.symbol} ${size:.0f}", "SUCCESS")
                self.trade_signal.emit(self.symbol, signal, size)
            else:
                msg = getattr(r, 'message', '') if r else ''
                self.log_signal.emit(f"❌ 执行失败: {msg}", "ERROR")
        except Exception as e:
            self.log_signal.emit(f"❌ 执行异常: {e}", "ERROR")

    def _fetch_klines(self, limit=80) -> Optional[List[List]]:
        if not self.okx:
            return None
        try:
            resp = self.okx.get_history_kline(self.symbol, bar=self.bar, limit=limit)
            if resp.get("code") == "0" and resp.get("data"):
                return [list(row) for row in reversed(resp["data"])]
        except Exception as e:
            self.log_signal.emit(f"❌ 获取K线失败: {e}", "ERROR")
        return None

    def _llm_confirm(self, analysis: Dict) -> bool:
        prompt = f"""你是加密货币交易确认助手。评估这个交易信号：

币种: {self.symbol}
周期: {self.bar}
当前价: {analysis['current_price']}
信号: {analysis['signal']}
趋势: {analysis['trend']}
RSI: {analysis['rsi']}
MACD: {analysis['macd_signal']}
布林位置: {analysis['bb_position']}%
成交量比: {analysis['volume_ratio']}x
置信度: {analysis['confidence']:.0%}

仅回复 YES 或 NO（是否同意执行这个信号）。"""
        try:
            reply = self.llm.chat([
                {"role": "system", "content": "只回复 YES 或 NO，不要其他内容。"},
                {"role": "user", "content": prompt},
            ], timeout=30)
            return reply and "YES" in reply.upper()
        except Exception:
            # LLM 不可用时拒绝交易（fail-close），而非默认同意
            return False
