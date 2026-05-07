"""
动态风险控制器 v1.0

解决加密货币大波动环境下的风控缺口：
  1. 资金费率作为方向信号（拐点检测，非简单阈值过滤）
  2. OI（未平仓合约）+ 价格组合趋势确认
  3. ATR 自适应止损（而非固定百分比）
  4. 波动率仓位缩放（高波动 = 小仓位，非直接过滤）
  5. 追踪止损 ATR 倍数（让利润奔跑）
  6. 波动率自适应 Gate 阈值
  7. BTC 暴跌全局熔断
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DynamicRiskConfig:
    """动态风险参数"""
    # ── ATR 自适应止损 ──
    atr_stop_mult: float = 2.5          # 基准 ATR 倍数
    atr_stop_floor_pct: float = 1.5     # 保底止损% (最低不低于此, 防止 ATR 极小)
    atr_stop_ceiling_pct: float = 15.0  # 止损上限% (极端波动保护)

    # ── 追踪止损 ──
    trail_enabled: bool = True
    trail_activate_atr_mult: float = 1.5  # 浮盈 ≥ N×ATR 后激活
    trail_distance_atr_mult: float = 2.0  # 追踪距离 = N×ATR

    # ── 波动率仓位缩放 ──
    vol_scale_enabled: bool = True
    base_vol_pct: float = 3.0           # 基准波动率%（BTC 日波动率）
    max_scale_ratio: float = 0.20       # 最高波动时仓位缩放下限(相对基准)

    # ── 资金费率 ──
    funding_reversal_threshold: float = 0.08  # 费率绝对值 > 此值视为极端
    funding_trend_weight: float = 0.15        # 费率趋势在信号评分中的权重

    # ── OI 趋势确认 ──
    oi_confirmation_enabled: bool = True

    # ── BTC 市场环境熔断 ──
    btc_crash_halt_pct: float = -5.0    # BTC 1H 跌幅超过此值 → 暂停冒货新开多
    btc_crash_halt_minutes: int = 30    # 熔断持续时间


@dataclass
class MarketRiskSnapshot:
    """单次扫描的市场风险快照"""
    symbol: str = ""
    btc_1h_pct: float = 0.0
    btc_4h_bullish: float = 1.0
    alt_24h_pct: float = 0.0
    atr_pct: float = 2.0           # 当前品种 ATR%（4H 或 1H）
    funding_rate: float = 0.0      # 当前资金费率（小数）
    funding_rate_1d_ago: float = 0.0  # 24H 前资金费率
    oi_change_24h: float = 0.0     # 24H OI 变化率
    oi_change_4h: float = 0.0      # 4H OI 变化率
    price_change_1h: float = 0.0   # 1H 价格变化


class DynamicRiskController:
    """动态风险控制器 —— 集中管理所有自适应风控逻辑。

    用法:
        ctrl = DynamicRiskController()
        snap = ctrl.build_snapshot(inst_id, klines_map, extra_data)
        adjusted_stop = ctrl.adaptive_stop_loss(entry_price, snap)
        scaled_size = ctrl.volatility_scaled_position(base_size, snap)
        trail_params = ctrl.trailing_stop_params(snap)
        funding_signal = ctrl.funding_direction_signal(snap)
        oi_confirm = ctrl.oi_trend_confirm(snap, direction)
        if ctrl.btc_crash_halt(snap):
            return  # 拒绝新开仓
    """

    def __init__(self, config: Optional[DynamicRiskConfig] = None):
        self.config = config or DynamicRiskConfig()
        self._btc_crash_until: float = 0.0

    # ═══════════════════════════════════════════════
    # 1. 快照构建
    # ═══════════════════════════════════════════════

    def build_snapshot(
        self, inst_id: str, klines_map: Dict[str, Any],
        extra_data: Optional[Dict] = None, btc_klines: Optional[Dict] = None
    ) -> MarketRiskSnapshot:
        """从 K 线数据构建完整风险快照"""
        snap = MarketRiskSnapshot(symbol=inst_id)

        ed = extra_data or {}

        # ATR%（从 4H 或 1H 计算）
        h4 = klines_map.get("4H") or klines_map.get("4h") or []
        h1 = klines_map.get("1H") or klines_map.get("1h") or []
        rows = h4 if len(h4) >= 15 else h1
        if len(rows) >= 15:
            snap.atr_pct = self._calc_atr_pct(rows)

        # 资金费率
        snap.funding_rate = float(ed.get("funding_rate", 0) or 0)
        snap.funding_rate_1d_ago = float(ed.get("funding_24h_ago", 0) or 0)

        # OI 变化
        snap.oi_change_24h = float(ed.get("oi_change_24h", 0) or 0)
        snap.oi_change_4h = float(ed.get("oi_change_4h", 0) or 0)

        # 价格变化
        snap.alt_24h_pct = float(ed.get("price_change_24h", 0) or 0)
        if len(h1) >= 2:
            try:
                snap.price_change_1h = (float(h1[-1][4]) / float(h1[-2][4]) - 1) * 100
            except Exception:
                pass

        # BTC 环境
        if btc_klines:
            snap.btc_1h_pct = self._calc_btc_1h_pct(btc_klines)
            snap.btc_4h_bullish = self._calc_ema_bullish(btc_klines, '4H')

        return snap

    # ═══════════════════════════════════════════════
    # 2. ATR 自适应止损
    # ═══════════════════════════════════════════════

    def adaptive_stop_loss(
        self, entry_price: float, snap: MarketRiskSnapshot, direction: str = "long"
    ) -> Tuple[float, float]:
        """返回 (止损价, 止损距离%)。
        公式: stop_distance% = clamp(ATR% × mult, floor, ceiling)
        """
        atr_pct = max(snap.atr_pct, 0.3)
        stop_pct = atr_pct * self.config.atr_stop_mult
        stop_pct = max(self.config.atr_stop_floor_pct, stop_pct)
        stop_pct = min(self.config.atr_stop_ceiling_pct, stop_pct)

        if direction == "long":
            sl_price = entry_price * (1 - stop_pct / 100.0)
        else:
            sl_price = entry_price * (1 + stop_pct / 100.0)
        return sl_price, stop_pct

    # ═══════════════════════════════════════════════
    # 3. 波动率仓位缩放
    # ═══════════════════════════════════════════════

    def volatility_scaled_position(
        self, base_size: float, snap: MarketRiskSnapshot
    ) -> float:
        """
        size = base_size × (base_vol / actual_vol)
        ATR=3% 基准 → 满仓；ATR=15% → 仓位缩到 20%
        """
        if not self.config.vol_scale_enabled:
            return base_size
        base_vol = self.config.base_vol_pct
        actual_vol = max(snap.atr_pct, base_vol * 0.5)
        scale = base_vol / actual_vol
        scale = max(self.config.max_scale_ratio, min(1.0, scale))
        return base_size * scale

    # ═══════════════════════════════════════════════
    # 4. 追踪止损参数
    # ═══════════════════════════════════════════════

    def trailing_stop_params(
        self, snap: MarketRiskSnapshot
    ) -> Dict[str, Any]:
        """返回追踪止损配置：{enabled, activate_pct, trail_pct}"""
        atr = max(snap.atr_pct, 0.5)
        return {
            "enabled": self.config.trail_enabled,
            "activate_pct": atr * self.config.trail_activate_atr_mult,
            "trail_pct": atr * self.config.trail_distance_atr_mult,
        }

    # ═══════════════════════════════════════════════
    # 5. 资金费率方向信号（拐点检测）
    # ═══════════════════════════════════════════════

    def funding_direction_signal(
        self, snap: MarketRiskSnapshot
    ) -> Dict[str, Any]:
        """
        费率不仅是过滤器，更是方向信号：
        - 费率从正转负拐点 → 空头拥挤解除，多头机会
        - 费率从负转正拐点 → 多头过热，即将回调
        返回 {"signal": "bullish"/"bearish"/"neutral", "weight": 0~1}
        """
        current = snap.funding_rate
        prior = snap.funding_rate_1d_ago

        result = {"signal": "neutral", "weight": 0.0, "detail": ""}
        if abs(current) > self.config.funding_reversal_threshold:
            if current > self.config.funding_reversal_threshold:
                result = {"signal": "bearish", "weight": 0.6, "detail": f"多头极度拥挤(费率{current*100:.2f}%)"}
            else:
                result = {"signal": "bullish", "weight": 0.6, "detail": f"空头极度拥挤(费率{current*100:.2f}%)"}

        # 拐点检测：方向翻转
        if abs(prior) > 0.01:
            if prior > 0.02 and current < -0.01:
                result = {"signal": "bullish", "weight": 0.8, "detail": f"费率正转负拐点({prior*100:.1f}%→{current*100:.1f}%)"}
            elif prior < -0.02 and current > 0.01:
                result = {"signal": "bearish", "weight": 0.8, "detail": f"费率负转正拐点({prior*100:.1f}%→{current*100:.1f}%)"}

        return result

    # ═══════════════════════════════════════════════
    # 6. OI + 价格趋势确认
    # ═══════════════════════════════════════════════

    def oi_trend_confirm(
        self, snap: MarketRiskSnapshot, direction: str = "long"
    ) -> Dict[str, Any]:
        """
        OI + 价格组合信号：
        Price↑ + OI↑ → 多头加仓，趋势确认 (bullish confirm)
        Price↓ + OI↑ → 空头加仓，下跌延续 (bearish confirm)
        Price↑ + OI↓ → 空头回补，反弹末端 (bullish caution)
        Price↓ + OI↓ → 多头平仓，底部信号 (bearish caution)
        """
        if not self.config.oi_confirmation_enabled:
            return {"signal": "neutral", "weight": 0.0}

        price_up = snap.price_change_1h > 0
        oi_up = snap.oi_change_4h > 1.0  # 4H OI 变化 > 1% = 显著

        if price_up and oi_up:
            result = {"signal": "bullish_confirm", "weight": 0.7, "detail": "OI+价格同步上升(多头加仓)"}
        elif price_up and not oi_up:
            result = {"signal": "bullish_caution", "weight": -0.3, "detail": "价格上涨但OI下降(空头回补，非真正需求)"}
        elif not price_up and oi_up:
            result = {"signal": "bearish_confirm", "weight": 0.7, "detail": "OI上升+价格下跌(空头加仓)"}
        elif not price_up and not oi_up:
            result = {"signal": "bearish_caution", "weight": -0.2, "detail": "OI下跌+价格下跌(多头减仓，接近底部)"}
        else:
            result = {"signal": "neutral", "weight": 0.0}

        if direction == "short":
            # 镜像翻转
            if "bullish" in result["signal"]:
                result["signal"] = result["signal"].replace("bullish", "bearish")
                result["weight"] = -result["weight"]
            elif "bearish" in result["signal"]:
                result["signal"] = result["signal"].replace("bearish", "bullish")
                result["weight"] = -result["weight"]

        return result

    # ═══════════════════════════════════════════════
    # 7. BTC 暴跌全局熔断
    # ═══════════════════════════════════════════════

    def btc_crash_halt(self, snap: MarketRiskSnapshot) -> Tuple[bool, str]:
        """BTC 1H 暴跌超过阈值 → 熔断所有山寨币多头"""
        if snap.btc_1h_pct > self.config.btc_crash_halt_pct:
            return False, ""

        # 已触发熔断且未过期
        now = time.time()
        if self._btc_crash_until > now:
            remaining = int((self._btc_crash_until - now) / 60)
            return True, f"BTC暴跌熔断中(剩余{remaining}分钟)"

        # 新触发
        if "BTC" not in snap.symbol.upper():
            self._btc_crash_until = now + self.config.btc_crash_halt_minutes * 60
            return True, f"BTC 1H暴跌{snap.btc_1h_pct:.1f}%，暂停山寨多头{self.config.btc_crash_halt_minutes}分钟"

        return False, ""

    def update_btc_crash_status(self, btc_1h_pct: float) -> None:
        """被动更新熔断状态（例如从 ticker 获取 BTC 价格）"""
        if btc_1h_pct <= self.config.btc_crash_halt_pct:
            if self._btc_crash_until <= time.time():
                self._btc_crash_until = time.time() + self.config.btc_crash_halt_minutes * 60
                logger.warning(f"BTC暴跌熔断触发: 1H={btc_1h_pct:.1f}%，持续{self.config.btc_crash_halt_minutes}分钟")
        else:
            if self._btc_crash_until and btc_1h_pct > -2.0:
                self._btc_crash_until = 0.0
                logger.info("BTC暴跌熔断解除")

    # ═══════════════════════════════════════════════
    # 8. 综合评分调整（一次调用获取所有调整）
    # ═══════════════════════════════════════════════

    def adjust_score_and_size(
        self, base_score: float, base_size: float,
        entry_price: float, direction: str,
        snap: MarketRiskSnapshot
    ) -> Dict[str, Any]:
        """一次调用完成所有风险调整，返回调整后的分数、仓位、止损"""
        # 资金费率调整
        funding = self.funding_direction_signal(snap)
        score_adjust = 0.0
        if funding["signal"] == "bullish" and direction == "long":
            score_adjust += funding["weight"] * 5.0
        elif funding["signal"] == "bearish" and direction == "long":
            score_adjust -= funding["weight"] * 5.0
        elif funding["signal"] == "bearish" and direction == "short":
            score_adjust += funding["weight"] * 5.0
        elif funding["signal"] == "bullish" and direction == "short":
            score_adjust -= funding["weight"] * 5.0

        # OI 趋势确认
        oi = self.oi_trend_confirm(snap, direction)
        score_adjust += oi["weight"] * 3.0

        adjusted_score = max(0.0, min(100.0, base_score + score_adjust))

        # ATR 自适应止损
        sl_price, sl_pct = self.adaptive_stop_loss(entry_price, snap, direction)

        # 波动率仓位缩放
        scaled_size = self.volatility_scaled_position(base_size, snap)

        # 追踪止损
        trail = self.trailing_stop_params(snap)

        return {
            "adjusted_score": round(adjusted_score, 2),
            "score_adjustment": round(score_adjust, 2),
            "scaled_position_size": round(scaled_size, 4),
            "stop_loss_price": round(sl_price, 6),
            "stop_loss_pct": round(sl_pct, 2),
            "trailing_stop": trail,
            "funding_signal": funding,
            "oi_signal": oi,
            "atr_pct": snap.atr_pct,
        }

    # ═══════════════════════════════════════════════
    # 内部工具方法
    # ═══════════════════════════════════════════════

    def _calc_atr_pct(self, rows: List) -> float:
        """从 K 线数据列表计算 ATR%"""
        try:
            if len(rows) < 15:
                return 2.0
            trs = []
            for i in range(1, min(len(rows), 20)):
                h = float(rows[-i][2])
                l = float(rows[-i][3])
                pc = float(rows[-i-1][4])
                trs.append(max(h - l, abs(h - pc), abs(l - pc)))
            if not trs:
                return 2.0
            avg_tr = float(np.mean(trs))
            last_close = float(rows[-1][4])
            if last_close <= 0:
                return 2.0
            return (avg_tr / last_close) * 100.0
        except Exception:
            return 2.0

    def _calc_btc_1h_pct(self, btc_klines: Dict) -> float:
        try:
            h1 = btc_klines.get("1H") or btc_klines.get("1h") or []
            if len(h1) >= 2:
                return (float(h1[-1][4]) / float(h1[-2][4]) - 1) * 100
        except Exception:
            pass
        return 0.0

    def _calc_ema_bullish(self, btc_klines: Dict, bar: str) -> float:
        try:
            rows = btc_klines.get(bar) or []
            if len(rows) < 50:
                return 1.0
            closes = [float(r[4]) for r in rows if float(r[4]) > 0]
            if len(closes) < 50:
                return 1.0
            import pandas as pd
            s = pd.Series(closes)
            ema20 = s.ewm(span=20, adjust=False).mean().iloc[-1]
            ema50 = s.ewm(span=50, adjust=False).mean().iloc[-1]
            return 1.0 if ema20 > ema50 else 0.0
        except Exception:
            return 1.0
