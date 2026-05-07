#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
三分钟多周期回调企稳策略

设计目标：
1. 4H 决定主趋势方向
2. 1H 确认波段是否仍在健康延续
3. 3m 只负责寻找回调企稳后的入场 setup
4. 真正的 1% 试仓 / 10% 加仓 / 第一原则止损由 StrategyRunner 状态机执行

该策略不直接做复杂仓位管理，只输出：
- BUY / SHORT: 候选 setup 已形成
- HOLD: 继续等待
- EXIT_LONG / EXIT_SHORT: 主趋势明显失效
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


CONFIG_SCHEMA: Dict[str, Dict[str, Any]] = {
    "preset": {
        "type": "select",
        "default": "custom",
        "label": "参数模板",
        "options": [
            {"label": "自定义", "value": "custom"},
            {"label": "BTC/ETH 保守多头版", "value": "btc_eth_conservative_long"},
            {"label": "BTC/ETH 自适应宽松版", "value": "btc_eth_relaxed"},
            {"label": "山寨币进攻版", "value": "altcoin_aggressive"},
        ],
    },
    "h4_fast_ema": {"type": "int", "default": 20, "label": "4H快线EMA"},
    "h4_slow_ema": {"type": "int", "default": 60, "label": "4H慢线EMA"},
    "h1_fast_ema": {"type": "int", "default": 12, "label": "1H快线EMA"},
    "h1_slow_ema": {"type": "int", "default": 26, "label": "1H慢线EMA"},
    "h1_rsi_min_long": {"type": "float", "default": 35.0, "label": "1H做多最低RSI"},
    "h1_rsi_max_long": {"type": "float", "default": 78.0, "label": "1H做多最高RSI"},
    "h1_rsi_min_short": {"type": "float", "default": 22.0, "label": "1H做空最低RSI"},
    "h1_rsi_max_short": {"type": "float", "default": 63.0, "label": "1H做空最高RSI"},
    "m3_fast_ema": {"type": "int", "default": 8, "label": "3m快线EMA"},
    "m3_mid_ema": {"type": "int", "default": 13, "label": "3m中线EMA"},
    "m3_slow_ema": {"type": "int", "default": 21, "label": "3m慢线EMA"},
    "m3_pullback_min_pct": {"type": "float", "default": 0.15, "label": "3m最小回调幅度%"},
    "m3_pullback_max_pct": {"type": "float", "default": 3.50, "label": "3m最大回调幅度%"},
    "m3_stabilization_bars": {"type": "int", "default": 2, "label": "3m企稳确认根数"},
    "m3_resumption_bars": {"type": "int", "default": 3, "label": "3m回升确认根数"},
    "m3_breakout_buffer_pct": {"type": "float", "default": 0.12, "label": "3m突破缓冲%"},
    "volume_confirm_ratio": {"type": "float", "default": 0.40, "label": "3m量能确认倍数"},
    "h1_max_stale_hours": {"type": "float", "default": 6.0, "label": "1H上下文最大滞后小时"},
    "h4_max_stale_hours": {"type": "float", "default": 24.0, "label": "4H上下文最大滞后小时"},
    "allow_short": {"type": "bool", "default": True, "label": "允许做空"},
    "h4_spread_min_pct": {"type": "float", "default": 0.15, "label": "4H EMA最小扩散%"},
    "h4_slope_min_pct": {"type": "float", "default": 0.06, "label": "4H EMA最小斜率%"},
    "h1_spread_min_pct": {"type": "float", "default": 0.07, "label": "1H EMA最小扩散%"},
    "h1_slope_min_pct": {"type": "float", "default": 0.03, "label": "1H EMA最小斜率%"},
    "atr_adaptive_thresholds": {"type": "bool", "default": True, "label": "ATR自适应阈值"},
}

DEFAULT_CONFIG = {key: spec["default"] for key, spec in CONFIG_SCHEMA.items()}

PRESET_CONFIGS: Dict[str, Dict[str, Any]] = {
    "custom": {},
    "btc_eth_conservative_long": {
        "allow_short": True,
        "h4_fast_ema": 20,
        "h4_slow_ema": 60,
        "h1_fast_ema": 12,
        "h1_slow_ema": 26,
        "h1_rsi_min_long": 48.0,
        "h1_rsi_max_long": 66.0,
        "h1_rsi_min_short": 35.0,
        "h1_rsi_max_short": 52.0,
        "m3_pullback_min_pct": 0.35,
        "m3_pullback_max_pct": 1.20,
        "m3_stabilization_bars": 3,
        "m3_breakout_buffer_pct": 0.20,
        "volume_confirm_ratio": 0.90,
        "h1_max_stale_hours": 2.0,
        "h4_max_stale_hours": 8.0,
    },
    "btc_eth_relaxed": {
        "allow_short": True,
        "h4_fast_ema": 18,
        "h4_slow_ema": 55,
        "h1_fast_ema": 10,
        "h1_slow_ema": 24,
        "h1_rsi_min_long": 40.0,
        "h1_rsi_max_long": 72.0,
        "h1_rsi_min_short": 28.0,
        "h1_rsi_max_short": 60.0,
        "m3_pullback_min_pct": 0.20,
        "m3_pullback_max_pct": 1.50,
        "m3_stabilization_bars": 2,
        "m3_breakout_buffer_pct": 0.15,
        "volume_confirm_ratio": 0.65,
        "h1_max_stale_hours": 4.0,
        "h4_max_stale_hours": 16.0,
        "h4_spread_min_pct": 0.15,
        "h4_slope_min_pct": 0.08,
        "h1_spread_min_pct": 0.08,
        "h1_slope_min_pct": 0.04,
        "atr_adaptive_thresholds": True,
        "trail_stop_pct": 2.5,
        "stage1_trail_activate_pct": 1.5,
        "stage1_trail_ratio": 0.35,
        "max_hold_hours": 72.0,
        "cost_line_atr_multiplier": 1.5,
    },
    "altcoin_aggressive": {
        "allow_short": True,
        "h4_fast_ema": 18,
        "h4_slow_ema": 55,
        "h1_fast_ema": 10,
        "h1_slow_ema": 24,
        "h1_rsi_min_long": 43.0,
        "h1_rsi_max_long": 72.0,
        "h1_rsi_min_short": 32.0,
        "h1_rsi_max_short": 58.0,
        "m3_pullback_min_pct": 0.25,
        "m3_pullback_max_pct": 1.60,
        "m3_stabilization_bars": 2,
        "m3_breakout_buffer_pct": 0.15,
        "volume_confirm_ratio": 0.70,
        "h1_max_stale_hours": 4.0,
        "h4_max_stale_hours": 16.0,
    },
}

BATCH_PRESET_LABEL = "批量回测三分钟策略模板"

# v4.3: 市场状态自适应参数 — 不同行情用不同参数
MARKET_STATE_PARAMS: Dict[str, Dict[str, Any]] = {
    "trending": {
        # 趋势市：收紧入场，放宽止损 — 趋势中的回调更可靠
        "m3_pullback_min_pct": 0.12,
        "m3_pullback_max_pct": 2.50,
        "m3_stabilization_bars": 3,
        "m3_resumption_bars": 3,
        "volume_confirm_ratio": 0.50,
        "h1_rsi_min_long": 42.0,
        "h1_rsi_max_long": 72.0,
        "h1_rsi_min_short": 28.0,
        "h1_rsi_max_short": 58.0,
        "h4_spread_min_pct": 0.20,
        "h4_slope_min_pct": 0.08,
        "h1_spread_min_pct": 0.10,
        "h1_slope_min_pct": 0.05,
    },
    "range": {
        # 震荡市：放宽入场，收紧止损 — 防假突破
        "m3_pullback_min_pct": 0.25,
        "m3_pullback_max_pct": 1.50,
        "m3_stabilization_bars": 2,
        "m3_resumption_bars": 4,
        "volume_confirm_ratio": 0.70,
        "h1_rsi_min_long": 35.0,
        "h1_rsi_max_long": 65.0,
        "h1_rsi_min_short": 35.0,
        "h1_rsi_max_short": 65.0,
        "h4_spread_min_pct": 0.10,
        "h4_slope_min_pct": 0.04,
        "h1_spread_min_pct": 0.05,
        "h1_slope_min_pct": 0.02,
    },
    "volatile": {
        # 高波动市：大幅放宽所有阈值 — ATR自适应扛波动
        "m3_pullback_min_pct": 0.10,
        "m3_pullback_max_pct": 4.50,
        "m3_stabilization_bars": 2,
        "m3_resumption_bars": 3,
        "volume_confirm_ratio": 0.35,
        "h1_rsi_min_long": 38.0,
        "h1_rsi_max_long": 76.0,
        "h1_rsi_min_short": 24.0,
        "h1_rsi_max_short": 62.0,
        "h4_spread_min_pct": 0.12,
        "h4_slope_min_pct": 0.05,
        "h1_spread_min_pct": 0.07,
        "h1_slope_min_pct": 0.03,
        "atr_adaptive_thresholds": True,
    },
}

REPORT_TUNING_GUIDE: Dict[str, str] = {
    "trade_count_too_low": "若总交易次数过低，优先放宽 m3_pullback_max_pct、volume_confirm_ratio 或 h1_rsi 区间，而不是先放松 4H 主趋势。",
    "holds_too_long": "若平均持仓时间过长或大量靠回测结束强平，优先收紧 max_hold_hours、h1_reversal_min_hold_hours、trail_stop_pct。",
    "short_bias_bad": "若亏损主要来自空头，先关闭 allow_short，或收紧 h1_rsi_max_short、m3_pullback_max_pct、4H 空头 spread/slope 条件。",
    "drawdown_high": "若最大回撤偏高，优先收紧 volume_confirm_ratio、m3_breakout_buffer_pct，并缩小 pilot/add 仓位比例。",
}


class ThreeMinuteMultiTimeframePullbackStrategy:
    name = "三分钟多周期回调企稳策略"
    description = "4H定方向，1H验延续，3m等回调企稳后输出 setup"
    strategy_type = "trade"
    required_bars = ["3m", "1H", "4H"]
    preferred_backtest_mode = "state_machine"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = dict(DEFAULT_CONFIG)
        self.config.update(config or {})
        preset_key = str(self.config.get("preset", "custom") or "custom")
        preset_values = PRESET_CONFIGS.get(preset_key, {})
        if preset_values:
            merged = dict(DEFAULT_CONFIG)
            merged.update(preset_values)
            merged.update(config or {})
            self.config = merged

    def get_config_schema(self) -> Dict[str, Dict[str, Any]]:
        return dict(CONFIG_SCHEMA)

    # v4.3: 市场状态自适应 — 引擎检测到状态变化时自动切换参数
    def _apply_market_state_params(self, state: str) -> None:
        """根据市场状态覆盖策略参数（由扫描引擎在每次扫描前调用）"""
        if state not in MARKET_STATE_PARAMS:
            return
        overrides = MARKET_STATE_PARAMS[state]
        for k, v in overrides.items():
            self.config[k] = v
        self._market_state = state

    def _apply_vol_pool_params(self, pool: str) -> None:
        """v4.7: 根据波动率池覆盖参数"""
        from src.scanner.volatility_pools import get_pool_params
        overrides = get_pool_params(pool, list(self.config.keys()))
        for k, v in overrides.items():
            if k in self.config:
                self.config[k] = v
        self._vol_pool = pool

    # ══════════════════════════════════════════════════════════════════════════
    # 多币种扫描接口（v4.3 新增）
    # ══════════════════════════════════════════════════════════════════════════

    def scan_symbol(self, symbol) -> Dict[str, Any]:
        """单币种扫描：从 ScannerSymbol 提取 klines，调用 generate_signal"""
        inst_id = getattr(symbol, "inst_id", "")
        price = float(getattr(symbol, "last_price", 0) or 0)

        base = {
            "symbol": inst_id, "passed": False, "score": 0.0,
            "direction": "WAIT", "signals": [], "details": {},
        }

        ed = (getattr(symbol, "extra_data", {}) or {}) if hasattr(symbol, "extra_data") else {}
        klines_map = ed.get("klines", {}) if isinstance(ed, dict) else {}

        h4_data = klines_map.get("4H") or klines_map.get("4h") or []
        h1_data = klines_map.get("1H") or klines_map.get("1h") or klines_map.get("hourly") or []
        m3_data = klines_map.get("3m") or klines_map.get("3M") or []

        if len(h4_data) < 80 or len(h1_data) < 60 or len(m3_data) < 48:
            return {**base, "details": {"跳过原因": f"数据不足(4H:{len(h4_data)}/1H:{len(h1_data)}/3m:{len(m3_data)})"}}

        signal = self.generate_signal({"4h": h4_data, "1H": h1_data, "3m": m3_data}, skip_exit_checks=True)
        if not signal or signal.get("action") == "HOLD":
            return {**base, "details": {"状态": signal.get("reason", "未触发") if signal else "无信号"}}

        direction_map = {"BUY": "BUY", "SHORT": "SELL"}
        direction = direction_map.get(signal.get("action", ""), "WAIT")
        conf = float(signal.get("confidence", 0) or 0)
        score = round(conf * 100, 1)

        return {
            "symbol": inst_id,
            "passed": True,
            "score": score,
            "opportunity_score": score,
            "direction": direction,
            "signals": [signal.get("reason", "")],
            "details": {
                "机会类型": "三分钟回调企稳",
                "入场价": signal.get("entry_price", 0),
                "置信度": f"{conf:.0%}",
                "方向": signal.get("timeframe_bias", ""),
                "建议止损": signal.get("suggested_stop", "-"),
                "止损%": f"-{signal.get('suggested_stop_pct', 0)}%",
                "原因": signal.get("reason", ""),
            },
            "last_price": price,
            "volume_24h": float(getattr(symbol, "volume_24h", 0) or 0),
        }

    def scan_all_symbols(self, symbols: List) -> Dict[str, Any]:
        """批量扫描：并发调用 scan_symbol，返回通过的结果"""
        results: List[Dict] = []
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=min(16, len(symbols) or 1)) as executor:
            futures = {executor.submit(self._safe_scan_symbol, s): s for s in symbols}
            for future in as_completed(futures):
                try:
                    r = future.result(timeout=20)
                except Exception:
                    continue
                if r and r.get("passed"):
                    results.append(r)
        results.sort(key=lambda x: float(x.get("score", 0) or 0), reverse=True)
        top_n = min(len(results), 30)
        return {
            "type": "three_min_pullback_scan",
            "all_opportunities": results[:top_n],
            "total_passed": len(results),
            "total_scanned": len(symbols),
        }

    def _safe_scan_symbol(self, symbol) -> Optional[Dict]:
        try:
            return self.scan_symbol(symbol)
        except Exception:
            return None

    def generate_signal(self, data, *args, **kwargs):
        h4_df = self._rows_to_df(data.get("4h") or data.get("4H") or data.get("h4") or [])
        h1_df = self._rows_to_df(data.get("hourly") or data.get("1H") or data.get("h1") or [])
        m3_df = self._rows_to_df(data.get("m3") or data.get("3m") or [])

        if len(h4_df) < 80 or len(h1_df) < 80 or len(m3_df) < 48:
            return self._hold("多周期数据不足")

        freshness = self._context_freshness_ok(m3_df, h1_df, h4_df)
        if not freshness["ok"]:
            return self._hold(freshness["reason"])

        long_bias = self._h4_trend_bias(h4_df, "LONG")
        short_bias = self._h4_trend_bias(h4_df, "SHORT")

        # 策略已验证的 Gate：Runner 跳过重复检查。含 d1_trend/entry_rule
        # 避免 Runner 内 _d1_trend_allows 和 evaluate_entry_rule_from_frames
        # 用日线EMA死叉或回调幅度重复过滤已通过4H/1H/3m共振的入场信号
        _gates = {"h4_trend", "h1_trend", "rsi", "m3_pullback", "volume",
                  "d1_trend", "entry_rule"}

        # ── 优先入场：先找 setup，找到直接返回 ──────────────────────────
        if long_bias and self._h1_trend_ok(h1_df, "LONG"):
            if self._h1_exhaustion(h1_df, "LONG"):
                return self._hold("1H多头力竭(RSI极端/MACD衰减)，暂停开多")
            if self._h4_exhaustion(h4_df, "LONG"):
                return self._hold("4H多头力竭(大周期顶部)，暂停开多")
            setup = self._m3_pullback_setup(m3_df, "LONG")
            if setup["ready"]:
                signal = {
                    "action": "BUY",
                    "entry_price": float(m3_df["close"].iloc[-1]),
                    "reason": (
                        f"4H多头 + 1H延续 + 3m回调企稳"
                        f" | 回调{setup['pullback_pct']:.2f}%"
                        f" | 反弹{setup.get('bounce_ratio', 0):.0%}"
                        f" | {setup['detail']}"
                    ),
                    "confidence": setup["confidence"],
                    "timeframe_bias": "4H/1H LONG",
                    "strategy_gates": _gates,
                }
                signal = self._add_atr_stop_suggestion(signal, m3_df, "LONG")
                signal = self._add_risk_budget_sizing(signal, m3_df)
                return signal
        if short_bias and bool(self.config.get("allow_short", False)) and self._h1_trend_ok(h1_df, "SHORT"):
            if self._h1_exhaustion(h1_df, "SHORT"):
                return self._hold("1H空头力竭(RSI极端/MACD衰减)，暂停开空")
            if self._h4_exhaustion(h4_df, "SHORT"):
                return self._hold("4H空头力竭(大周期底部)，暂停开空")
            setup = self._m3_pullback_setup(m3_df, "SHORT")
            if setup["ready"]:
                signal = {
                    "action": "SHORT",
                    "entry_price": float(m3_df["close"].iloc[-1]),
                    "reason": (
                        f"4H空头 + 1H延续 + 3m反抽企稳"
                        f" | 回抽{setup['pullback_pct']:.2f}%"
                        f" | {setup['detail']}"
                    ),
                    "confidence": setup["confidence"],
                    "timeframe_bias": "4H/1H SHORT",
                    "strategy_gates": _gates,
                }
                signal = self._add_atr_stop_suggestion(signal, m3_df, "SHORT")
                signal = self._add_risk_budget_sizing(signal, m3_df)
                return signal

        # ── 无入场 setup 才检查是否需要退出 ──────────────────────────
        skip_exit = bool(kwargs.get("skip_exit_checks", False))
        if not skip_exit:
            if self._should_exit(h4_df, h1_df, m3_df, "LONG"):
                return {
                    "action": "EXIT_LONG",
                    "entry_price": float(m3_df["close"].iloc[-1]),
                    "reason": "4H/1H/3m 多头共振失效，先退出等待",
                    "confidence": 0.75,
                }
            if bool(self.config.get("allow_short", False)) and self._should_exit(h4_df, h1_df, m3_df, "SHORT"):
                return {
                    "action": "EXIT_SHORT",
                    "entry_price": float(m3_df["close"].iloc[-1]),
                    "reason": "4H/1H/3m 空头共振失效，先退出等待",
                    "confidence": 0.75,
                }

        return self._hold("未形成4H/1H/3m共振 setup")

    def _add_atr_stop_suggestion(self, signal: Dict, m3_df: pd.DataFrame, direction: str) -> Dict:
        """为信号附加基于3m ATR的动态止损建议"""
        try:
            atr_series = self._atr(m3_df, 14)
            atr_val = float(atr_series.iloc[-1])
            entry = signal.get("entry_price", 0)
            if atr_val > 0 and entry > 0:
                mult = 1.8
                sl = entry - atr_val * mult if direction == "LONG" else entry + atr_val * mult
                sl_pct = atr_val * mult / entry * 100
                signal["suggested_stop"] = round(sl, 4)
                signal["suggested_stop_pct"] = round(sl_pct, 2)
                signal["reason"] += f" | 止损{sl:.2f}(-{sl_pct:.1f}%)"
        except Exception:
            pass
        return signal

    def _add_risk_budget_sizing(self, signal: Dict, m3_df: pd.DataFrame) -> Dict:
        """v4.8: 基于风险预算计算建议仓位"""
        try:
            from src.trading.risk_budget_sizer import calculate_position
            atr_series = self._atr(m3_df, 14)
            atr_val = float(atr_series.iloc[-1])
            entry = signal.get("entry_price", 0)
            if atr_val > 0 and entry > 0:
                atr_pct = atr_val / entry * 100
                pos = calculate_position(
                    capital=10000.0, entry_price=entry, atr_pct=atr_pct,
                    risk_pct=0.01, atr_multiplier=1.8,
                )
                signal["suggested_position_pct"] = pos["position_pct"]
                signal["suggested_position_usdt"] = pos["position_usdt"]
                signal["suggested_stop_loss"] = pos["stop_loss_price"]
                signal["reason"] += (
                    f" | 仓位{pos['position_pct']:.1f}%"
                    f"(${pos['position_usdt']:.0f})"
                    f" 风险{pos['risk_amount_usdt']:.0f}"
                )
        except Exception:
            pass
        return signal

    def _hold(self, reason: str) -> Dict[str, Any]:
        return {"action": "HOLD", "reason": reason, "confidence": 0.0}

    def _rows_to_df(self, rows: List) -> pd.DataFrame:
        if not rows:
            return pd.DataFrame()
        base_cols = ["ts", "open", "high", "low", "close", "volume"]
        extra = [f"_x{i}" for i in range(20)]
        width = len(rows[0]) if rows else 0
        cols = base_cols[:width] if width <= len(base_cols) else base_cols + extra[: width - len(base_cols)]
        df = pd.DataFrame(rows, columns=cols)
        for col in ("open", "high", "low", "close", "volume"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df["ts"] = pd.to_numeric(df["ts"], errors="coerce")
        df = df.dropna(subset=["ts", "open", "high", "low", "close"]).copy()
        df["timestamp"] = pd.to_datetime(df["ts"].astype("int64"), unit="ms")
        return df.sort_values("timestamp").drop_duplicates(subset="timestamp", keep="last").reset_index(drop=True)

    def _ema(self, close: pd.Series, span: int) -> pd.Series:
        return close.ewm(span=max(int(span), 1), adjust=False).mean()

    def _context_freshness_ok(self, m3_df: pd.DataFrame, h1_df: pd.DataFrame, h4_df: pd.DataFrame) -> Dict[str, Any]:
        latest_ts = pd.Timestamp(m3_df["timestamp"].iloc[-1])
        h1_last = pd.Timestamp(h1_df["timestamp"].iloc[-1])
        h4_last = pd.Timestamp(h4_df["timestamp"].iloc[-1])
        h1_age_hours = max((latest_ts - h1_last).total_seconds() / 3600.0, 0.0)
        h4_age_hours = max((latest_ts - h4_last).total_seconds() / 3600.0, 0.0)
        max_h1 = float(self.config.get("h1_max_stale_hours", 3.0))
        max_h4 = float(self.config.get("h4_max_stale_hours", 12.0))
        if h1_age_hours > max_h1:
            return {"ok": False, "reason": f"1H 上下文过旧 ({h1_age_hours:.1f}h > {max_h1:.1f}h)"}
        if h4_age_hours > max_h4:
            return {"ok": False, "reason": f"4H 上下文过旧 ({h4_age_hours:.1f}h > {max_h4:.1f}h)"}
        return {"ok": True, "reason": ""}

    def _rsi(self, close: pd.Series, period: int = 14) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / loss.replace(0, pd.NA)
        return (100 - (100 / (1 + rs))).bfill().fillna(50.0)

    def _h4_trend_bias(self, h4_df: pd.DataFrame, direction: str) -> bool:
        if h4_df.empty or len(h4_df) < 80:
            return False
        close = h4_df["close"]
        fast = self._ema(close, int(self.config.get("h4_fast_ema", 20)))
        slow = self._ema(close, int(self.config.get("h4_slow_ema", 60)))
        slope_lookback = min(6, len(fast) - 1)
        if slope_lookback <= 0:
            return False
        slope = (float(fast.iloc[-1]) - float(fast.iloc[-1 - slope_lookback])) / max(float(fast.iloc[-1 - slope_lookback]), 1e-9)
        last_close = float(close.iloc[-1])
        last_fast = float(fast.iloc[-1])
        last_slow = float(slow.iloc[-1])
        spread_pct = (last_fast - last_slow) / max(last_slow, 1e-9)

        atr_factor = self._atr_scale(h4_df, 14)
        min_spread = float(self.config.get("h4_spread_min_pct", 0.25)) / 100.0 * atr_factor
        min_slope = float(self.config.get("h4_slope_min_pct", 0.12)) / 100.0 * atr_factor

        if direction == "LONG":
            return (
                spread_pct >= min_spread
                and slope >= min_slope
                and last_close >= last_fast * 0.992
            )
        return (
            spread_pct <= -min_spread
            and slope <= -min_slope
            and last_close <= last_fast * 1.008
        )

    def _h4_exhaustion(self, h4_df: pd.DataFrame, direction: str) -> bool:
        """4H级别力竭检测：防止在大周期顶部追多/底部追空"""
        if h4_df.empty or len(h4_df) < 60:
            return False
        close = h4_df["close"]
        rsi = self._rsi(close, 14)
        cur_rsi = float(rsi.iloc[-1])
        macd = self._ema(close, 12) - self._ema(close, 26)
        macd_signal = macd.ewm(span=9, adjust=False).mean()
        macd_hist = macd - macd_signal
        cur_hist = float(macd_hist.iloc[-1])
        lookback = min(30, len(close) - 1)
        if direction == "LONG":
            rsi_high = cur_rsi > 72.0
            hist_declining = all(float(macd_hist.iloc[-(i+1)]) <= float(macd_hist.iloc[-(i+2)])
                                 for i in range(3) if len(macd_hist) >= 5)
            recent_high_idx = close.iloc[-lookback:].idxmax()
            rsi_at_high = float(rsi.iloc[recent_high_idx])
            divergence = (float(close.iloc[-1]) >= float(close.iloc[recent_high_idx]) * 0.995
                          and cur_rsi < rsi_at_high - 1.5)
            return rsi_high and (hist_declining or divergence)
        else:
            rsi_low = cur_rsi < 28.0
            hist_rising = all(float(macd_hist.iloc[-(i+1)]) >= float(macd_hist.iloc[-(i+2)])
                              for i in range(3) if len(macd_hist) >= 5)
            recent_low_idx = close.iloc[-lookback:].idxmin()
            rsi_at_low = float(rsi.iloc[recent_low_idx])
            divergence = (float(close.iloc[-1]) <= float(close.iloc[recent_low_idx]) * 1.005
                          and cur_rsi > rsi_at_low + 1.5)
            return rsi_low and (hist_rising or divergence)

    def _h1_exhaustion(self, h1_df: pd.DataFrame, direction: str) -> bool:
        """趋势力竭检测：RSI极端 + MACD柱连续衰减 + 背离 → 避免追在末端"""
        if h1_df.empty or len(h1_df) < 40:
            return False
        close = h1_df["close"]
        rsi = self._rsi(close, 14)
        macd = self._ema(close, 12) - self._ema(close, 26)
        macd_signal = macd.ewm(span=9, adjust=False).mean()
        macd_hist = macd - macd_signal
        cur_rsi = float(rsi.iloc[-1])
        hist_values = [float(macd_hist.iloc[-(i + 1)]) for i in range(5)]
        cur_hist = float(macd_hist.iloc[-1])
        prev_hist = float(macd_hist.iloc[-2])
        if direction == "LONG":
            # RSI > 65 且 MACD柱连续衰减 或 MACD从正转负
            rsi_extreme = cur_rsi > 65.0
            hist_declining = all(hist_values[i] <= hist_values[i + 1] for i in range(1))
            hist_cross_below_zero = cur_hist < 0 and prev_hist >= 0
            # 价格创新高但RSI未创新高 → 顶背离（扩大检测窗口到20根）
            lookback = min(20, len(close) - 1)
            recent_high_idx = close.iloc[-lookback:].idxmax()
            rsi_at_high = float(rsi.iloc[recent_high_idx])
            divergence = (float(close.iloc[-1]) >= float(close.iloc[recent_high_idx]) * 0.995
                          and cur_rsi < rsi_at_high - 2.0)
            return rsi_extreme and (hist_declining or hist_cross_below_zero or divergence)
        else:
            rsi_extreme = cur_rsi < 35.0
            hist_rising = all(hist_values[i] >= hist_values[i + 1] for i in range(1))
            hist_cross_above_zero = cur_hist > 0 and prev_hist <= 0
            lookback = min(20, len(close) - 1)
            recent_low_idx = close.iloc[-lookback:].idxmin()
            rsi_at_low = float(rsi.iloc[recent_low_idx])
            divergence = (float(close.iloc[-1]) <= float(close.iloc[recent_low_idx]) * 1.005
                          and cur_rsi > rsi_at_low + 2.0)
            return rsi_extreme and (hist_rising or hist_cross_above_zero or divergence)

    def _h1_trend_ok(self, h1_df: pd.DataFrame, direction: str) -> bool:
        if h1_df.empty or len(h1_df) < 60:
            return False
        close = h1_df["close"]
        fast = self._ema(close, int(self.config.get("h1_fast_ema", 12)))
        slow = self._ema(close, int(self.config.get("h1_slow_ema", 26)))
        rsi = self._rsi(close, 14)
        macd = self._ema(close, 12) - self._ema(close, 26)
        macd_signal = macd.ewm(span=9, adjust=False).mean()
        macd_hist = macd - macd_signal
        last_close = float(close.iloc[-1])
        last_fast = float(fast.iloc[-1])
        last_slow = float(slow.iloc[-1])
        slope_lookback = min(8, len(fast) - 1)
        slope = (float(fast.iloc[-1]) - float(fast.iloc[-1 - slope_lookback])) / max(float(fast.iloc[-1 - slope_lookback]), 1e-9)
        cur_rsi = float(rsi.iloc[-1])
        spread_pct = (last_fast - last_slow) / max(last_slow, 1e-9)
        cur_hist = float(macd_hist.iloc[-1])
        hist_mean = float(macd_hist.tail(6).mean())

        atr_factor = self._atr_scale(h1_df, 14)
        min_spread = float(self.config.get("h1_spread_min_pct", 0.10)) / 100.0 * atr_factor
        min_slope = float(self.config.get("h1_slope_min_pct", 0.05)) / 100.0 * atr_factor

        if direction == "LONG":
            hist_rising = cur_hist > float(macd_hist.iloc[-3]) and cur_hist > -0.0002 * max(last_close, 1.0)
            return (
                spread_pct >= min_spread
                and slope >= min_slope
                and last_close >= last_slow * 0.999
                and hist_rising
                and float(self.config.get("h1_rsi_min_long", 35.0)) <= cur_rsi <= float(self.config.get("h1_rsi_max_long", 78.0))
            )
        hist_falling = cur_hist < float(macd_hist.iloc[-3]) and cur_hist < 0.0002 * max(last_close, 1.0)
        return (
            spread_pct <= -min_spread
            and slope <= -min_slope
            and last_close <= last_slow * 1.001
            and hist_falling
            and float(self.config.get("h1_rsi_min_short", 22.0)) <= cur_rsi <= float(self.config.get("h1_rsi_max_short", 63.0))
        )

    def _h1_reversal_exit(self, h1_df: pd.DataFrame, direction: str) -> bool:
        if h1_df.empty or len(h1_df) < 32:
            return False
        close = h1_df["close"]
        fast = self._ema(close, int(self.config.get("h1_fast_ema", 12)))
        slow = self._ema(close, int(self.config.get("h1_slow_ema", 26)))
        confirm_bars = 3
        if len(fast) < confirm_bars or len(slow) < confirm_bars:
            return False
        cross_threshold = 0.002
        if direction == "LONG":
            cross_count = 0
            for i in range(confirm_bars):
                idx = -(i + 1)
                f_val = float(fast.iloc[idx])
                s_val = float(slow.iloc[idx])
                if f_val < s_val * (1 - cross_threshold):
                    cross_count += 1
            if cross_count < 2:
                return False
            return float(close.iloc[-1]) < float(slow.iloc[-1])
        else:
            cross_count = 0
            for i in range(confirm_bars):
                idx = -(i + 1)
                f_val = float(fast.iloc[idx])
                s_val = float(slow.iloc[idx])
                if f_val > s_val * (1 + cross_threshold):
                    cross_count += 1
            if cross_count < 2:
                return False
            return float(close.iloc[-1]) > float(slow.iloc[-1])

    def _m3_momentum_reversal(self, m3_df: pd.DataFrame, direction: str) -> bool:
        if m3_df.empty or len(m3_df) < 24:
            return False
        close = m3_df["close"]
        fast = self._ema(close, int(self.config.get("m3_fast_ema", 8)))
        mid = self._ema(close, int(self.config.get("m3_mid_ema", 13)))
        slow = self._ema(close, int(self.config.get("m3_slow_ema", 21)))
        lookback = 3
        if direction == "LONG":
            cross_count = int((fast.tail(lookback) < mid.tail(lookback)).sum())
            price_below_slow = float(close.iloc[-1]) < float(slow.iloc[-1]) * 0.998
            return cross_count >= 2 and price_below_slow
        cross_count = int((fast.tail(lookback) > mid.tail(lookback)).sum())
        price_above_slow = float(close.iloc[-1]) > float(slow.iloc[-1]) * 1.002
        return cross_count >= 2 and price_above_slow

    def _should_exit(self, h4_df: pd.DataFrame, h1_df: pd.DataFrame, m3_df: pd.DataFrame, direction: str) -> bool:
        h4_bias_ok = self._h4_trend_bias(h4_df, direction)
        h1_ok = self._h1_trend_ok(h1_df, direction)
        h1_reversal = self._h1_reversal_exit(h1_df, direction)
        m3_reversal = self._m3_momentum_reversal(m3_df, direction)
        if direction == "LONG":
            return ((not h4_bias_ok) and (h1_reversal or m3_reversal)) or (h1_reversal and m3_reversal) or ((not h1_ok) and m3_reversal)
        return ((not h4_bias_ok) and (h1_reversal or m3_reversal)) or (h1_reversal and m3_reversal) or ((not h1_ok) and m3_reversal)

    def _find_local_swing(self, series: pd.Series, direction: str, order: int = 5) -> int:
        """找最近的局部 swing high/low 拐点索引（替代全局极值）。"""
        values = series.values
        if len(values) < order * 2 + 1:
            fallback = series.idxmax() if direction == "LONG" else series.idxmin()
            if pd.isna(fallback):
                return len(values) - 1
            return int(fallback)
        if direction == "LONG":
            # 找局部高点
            for i in range(len(values) - order - 1, order - 1, -1):
                window = values[max(0, i - order): i + order + 1]
                if values[i] == window.max():
                    return i
            fallback = series.idxmax()
            return int(fallback) if not pd.isna(fallback) else len(values) - 1
        else:
            for i in range(len(values) - order - 1, order - 1, -1):
                window = values[max(0, i - order): i + order + 1]
                if values[i] == window.min():
                    return i
            fallback = series.idxmin()
            return int(fallback) if not pd.isna(fallback) else 0

    def _atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Average True Range"""
        h, l, c = df["high"], df["low"], df["close"].shift(1)
        tr = pd.concat([h - l, (h - c).abs(), (l - c).abs()], axis=1).max(axis=1)
        return tr.ewm(span=period, adjust=False).mean()

    def _atr_scale(self, df: pd.DataFrame, period: int = 14) -> float:
        """返回 ATR% 缩放因子 (0.5~2.5)，用于自适应阈值调整"""
        if not bool(self.config.get("atr_adaptive_thresholds", True)):
            return 1.0
        atr = self._atr(df, period)
        if atr.empty or len(atr) < period:
            return 1.0
        price = df["close"].iloc[-1]
        if price <= 0:
            return 1.0
        atr_pct = float(atr.iloc[-1]) / price * 100
        return max(0.5, min(1.5, 4.0 * atr_pct / 3.0))

    def _m3_resumption_ok(self, df: pd.DataFrame, direction: str) -> Dict[str, Any]:
        """3m 回升确认：低点上移 + EMA拐头 + 突破盘整 → 试仓成功前提"""
        res_bars = max(int(self.config.get("m3_resumption_bars", 3)), 2)
        if len(df) < res_bars + 5:
            return {"ok": False, "score": 0.0, "detail": "数据不足"}
        close = df["close"]
        low = df["low"]
        high = df["high"]
        fast = self._ema(close, int(self.config.get("m3_fast_ema", 8)))
        mid = self._ema(close, int(self.config.get("m3_mid_ema", 13)))

        if direction == "LONG":
            # ① 微观上行：最近 res_bars 根低点逐步抬高
            lows = [float(low.iloc[-(i + 1)]) for i in range(res_bars)]
            higher_lows = all(lows[i] > lows[i + 1] for i in range(res_bars - 1))
            # ② fast EMA 拐头向上
            ema_slope = float(fast.iloc[-1]) > float(fast.iloc[-res_bars]) * 1.0002
            # ③ 突破盘整：收盘在最近 (res_bars+2) 根高点之上
            breakout_level = float(high.tail(res_bars + 2).max())
            breakout = float(close.iloc[-1]) >= breakout_level * 0.999
            # ④ fast EMA 在 mid EMA 之上（趋势排列恢复）
            ema_aligned = float(fast.iloc[-1]) >= float(mid.iloc[-1]) * 0.999
            # ⑤ 收盘在 fast EMA 之上
            above_fast = float(close.iloc[-1]) >= float(fast.iloc[-1]) * 0.999
        else:
            highs = [float(high.iloc[-(i + 1)]) for i in range(res_bars)]
            higher_lows = all(highs[i] < highs[i + 1] for i in range(res_bars - 1))
            ema_slope = float(fast.iloc[-1]) < float(fast.iloc[-res_bars]) * 0.9998
            breakout_level = float(low.tail(res_bars + 2).min())
            breakout = float(close.iloc[-1]) <= breakout_level * 1.001
            ema_aligned = float(fast.iloc[-1]) <= float(mid.iloc[-1]) * 1.001
            above_fast = float(close.iloc[-1]) <= float(fast.iloc[-1]) * 1.001

        # 至少满足 4/5，且 breakout 为强制条件
        checks = [higher_lows, ema_slope, breakout, ema_aligned, above_fast]
        passed = sum(checks)
        ok = passed >= 4 and breakout

        detail = (
            f"低点上移={higher_lows} EMA拐头={ema_slope} "
            f"突破盘整={breakout} EMA排列={ema_aligned} 站上快线={above_fast}"
        )
        return {"ok": ok, "score": passed / 5.0 * 0.85, "detail": detail}

    def _m3_pullback_setup(self, m3_df: pd.DataFrame, direction: str) -> Dict[str, Any]:
        if m3_df.empty or len(m3_df) < 36:
            return {"ready": False, "pullback_pct": 0.0, "detail": "3m数据不足", "confidence": 0.0}
        df = m3_df.tail(48).reset_index(drop=True)
        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"] if "volume" in df.columns else pd.Series([0] * len(df))
        fast = self._ema(close, int(self.config.get("m3_fast_ema", 8)))
        mid = self._ema(close, int(self.config.get("m3_mid_ema", 13)))
        slow = self._ema(close, int(self.config.get("m3_slow_ema", 21)))
        rsi = self._rsi(close, 14)
        atr = self._atr(df, 14)
        stab_bars = max(int(self.config.get("m3_stabilization_bars", 3)), 2)
        min_pb = float(self.config.get("m3_pullback_min_pct", 0.15))
        max_pb = float(self.config.get("m3_pullback_max_pct", 2.50))
        vol_ratio = float(self.config.get("volume_confirm_ratio", 0.60))
        use_atr_adaptive = bool(self.config.get("atr_adaptive_thresholds", True))

        # ── ATR 自适应回调幅度 ──────────────────────────────────────────
        recent_atr = float(atr.tail(stab_bars).mean()) if len(atr) >= stab_bars else 0.0
        prior_atr = float(atr.iloc[-stab_bars - 8:-stab_bars].mean()) if len(atr) >= stab_bars + 8 else recent_atr
        vol_contracting = prior_atr <= 0 or recent_atr <= prior_atr * 1.05
        last_price = float(close.iloc[-1])
        if use_atr_adaptive and recent_atr > 0:
            atr_pct = recent_atr / last_price * 100
            scale = max(0.6, min(2.5, atr_pct * 1.8))
            effective_min_pb = min_pb * scale
            effective_max_pb = max_pb * scale
        else:
            atr_pct = 0.0  # v4.4: 防止未定义
            effective_min_pb = min_pb
            effective_max_pb = max_pb

        # ── 局部 swing high/low 找回调锚点 ──────────────────────────────
        search_slice = close.iloc[:-stab_bars]
        if direction == "LONG":
            anchor_idx = self._find_local_swing(search_slice, "LONG", order=5)
            anchor = float(high.iloc[anchor_idx])
            pb_slice = df.iloc[anchor_idx + 1:]
            if pb_slice.empty:
                return {"ready": False, "pullback_pct": 0.0, "detail": "尚未出现回调", "confidence": 0.0}
            trough = float(pb_slice["low"].min())
            pullback_pct = (anchor - trough) / anchor * 100 if anchor > 0 else 0.0

            # ── 反弹比例（假企稳过滤） ────────────────────────────────
            pullback_distance = anchor - trough
            bounce_distance = last_price - trough
            bounce_ratio = bounce_distance / pullback_distance if pullback_distance > 0 else 0.0
            bounce_ok = bounce_ratio >= 0.25

            # ── 企稳判定（比例制替代全或无） ──────────────────────────
            bars_above_fast = int((close.tail(stab_bars) >= fast.tail(stab_bars)).sum())
            bars_above_mid = int((close.tail(stab_bars) >= mid.tail(stab_bars)).sum())
            stabilized_ratio = (bars_above_fast + bars_above_mid) / (stab_bars * 2)
            rsi_bottom = 35.0 if not use_atr_adaptive else max(30.0, 45.0 - atr_pct * 3.0)
            rsi_floor_ok = float(rsi.iloc[-1]) >= rsi_bottom
            stabilized = stabilized_ratio >= 0.50 and float(close.iloc[-1]) >= float(mid.iloc[-1]) * 0.998 and rsi_floor_ok

            # ── 延续判定（MACD柱+RSI斜率+价格结构+蜡烛形态） ─────────
            above_fast_ema = float(close.iloc[-1]) >= float(fast.iloc[-1]) * 0.998
            macd_3m = fast - slow
            macd_signal_3m = macd_3m.ewm(span=9, adjust=False).mean()
            macd_hist = macd_3m - macd_signal_3m
            hist_turning_up = float(macd_hist.iloc[-1]) > float(macd_hist.iloc[-2]) and float(macd_hist.iloc[-1]) >= 0
            hist_positive = float(macd_hist.iloc[-1]) > 0
            rsi_rising = float(rsi.iloc[-1]) >= float(rsi.iloc[-2])
            price_higher = float(close.iloc[-1]) >= float(close.iloc[-3])
            # 蜡烛形态：收盘在上半部 + 阳线
            candle_range = float(high.iloc[-1]) - float(low.iloc[-1])
            candle_position = (float(close.iloc[-1]) - float(low.iloc[-1])) / candle_range if candle_range > 0 else 0.5
            bullish_candle = candle_position >= 0.40 and float(close.iloc[-1]) >= float(df["open"].iloc[-1])
            # 低价拒测：最近 stab_bars 低点离 trough 有安全距离
            recent_low = float(low.tail(stab_bars).min())
            retest_rejected = recent_low > trough * 1.0015
            # 至少 1/3 条件满足
            momentum_signals = sum([rsi_rising, price_higher, vol_contracting])
            continuation = (
                above_fast_ema
                and (hist_turning_up or hist_positive)
                and momentum_signals >= 1
                and (bullish_candle or retest_rejected)
            )
        else:
            anchor_idx = self._find_local_swing(search_slice, "SHORT", order=5)
            anchor = float(low.iloc[anchor_idx])
            pb_slice = df.iloc[anchor_idx + 1:]
            if pb_slice.empty:
                return {"ready": False, "pullback_pct": 0.0, "detail": "尚未出现反抽", "confidence": 0.0}
            rebound = float(pb_slice["high"].max())
            pullback_pct = (rebound - anchor) / anchor * 100 if anchor > 0 else 0.0

            rebound_distance = rebound - anchor
            retreat_distance = rebound - last_price
            bounce_ratio = retreat_distance / rebound_distance if rebound_distance > 0 else 0.0
            bounce_ok = bounce_ratio >= 0.25

            bars_below_fast = int((close.tail(stab_bars) <= fast.tail(stab_bars)).sum())
            bars_below_mid = int((close.tail(stab_bars) <= mid.tail(stab_bars)).sum())
            stabilized_ratio = (bars_below_fast + bars_below_mid) / (stab_bars * 2)
            rsi_top = 65.0 if not use_atr_adaptive else min(70.0, 55.0 + atr_pct * 3.0)
            rsi_ceil_ok = float(rsi.iloc[-1]) <= rsi_top
            stabilized = stabilized_ratio >= 0.50 and float(close.iloc[-1]) <= float(mid.iloc[-1]) * 1.002 and rsi_ceil_ok

            below_fast_ema = float(close.iloc[-1]) <= float(fast.iloc[-1]) * 1.002
            macd_3m = fast - slow
            macd_signal_3m = macd_3m.ewm(span=9, adjust=False).mean()
            macd_hist = macd_3m - macd_signal_3m
            hist_turning_down = float(macd_hist.iloc[-1]) < float(macd_hist.iloc[-2]) and float(macd_hist.iloc[-1]) <= 0
            hist_negative = float(macd_hist.iloc[-1]) < 0
            rsi_falling = float(rsi.iloc[-1]) <= float(rsi.iloc[-2])
            price_lower = float(close.iloc[-1]) <= float(close.iloc[-3])
            candle_range = float(high.iloc[-1]) - float(low.iloc[-1])
            candle_position = (float(high.iloc[-1]) - float(close.iloc[-1])) / candle_range if candle_range > 0 else 0.5
            bearish_candle = candle_position >= 0.40 and float(close.iloc[-1]) <= float(df["open"].iloc[-1])
            recent_high = float(high.tail(stab_bars).max())
            retest_rejected = recent_high < rebound * 0.9985
            momentum_signals = sum([rsi_falling, price_lower, vol_contracting])
            continuation = (
                below_fast_ema
                and (hist_turning_down or hist_negative)
                and momentum_signals >= 1
                and (bearish_candle or retest_rejected)
            )

        # ── 量能确认（放宽+趋势判断） ──────────────────────────────────
        avg_vol = float(volume.tail(21).iloc[:-1].mean()) if len(volume) >= 21 else 0.0
        cur_vol = float(volume.iloc[-1])
        vol_ok = avg_vol <= 0 or cur_vol >= avg_vol * vol_ratio
        vol_multiple = cur_vol / avg_vol if avg_vol > 0 else 1.0
        vol_rising_trend = (
            float(volume.tail(5).mean()) >= float(volume.tail(10).head(5).mean()) * 0.95
            if len(volume) >= 10 else True
        )

        # ── EMA 序列堆叠（比例制） ─────────────────────────────────────
        # 只需要大多数 bar 符合趋势排列即可
        stack_ratio = (
            int((fast.tail(stab_bars) >= mid.tail(stab_bars)).sum()) / stab_bars
            if direction == "LONG"
            else int((fast.tail(stab_bars) <= mid.tail(stab_bars)).sum()) / stab_bars
        )
        trend_stack_ok = stack_ratio >= 0.6

        slow_filter_ok = (
            float(close.iloc[-1]) >= float(slow.iloc[-1])
            if direction == "LONG"
            else float(close.iloc[-1]) <= float(slow.iloc[-1])
        )

        # ── 3m 回升确认（试仓成功率核心） ──────────────────────────────
        resumption = self._m3_resumption_ok(df, direction)

        # ── 综合入场判定 ──────────────────────────────────────────────
        ready = (
            effective_min_pb <= pullback_pct <= effective_max_pb
            and bounce_ok
            and stabilized
            and continuation
            and vol_ok
            and trend_stack_ok
            and resumption["ok"]
        )

        # ── 信心评分 ──────────────────────────────────────────────────
        confidence = 0.0
        if ready:
            base = 0.50
            # 反弹比例得分（30%~70% 最优）
            bounce_score = max(0, 0.08 - abs(bounce_ratio - 0.45) * 0.12)
            # 回调处于最优区段 (20%~80% of range) 得分最高
            pb_range = max(effective_max_pb - effective_min_pb, 1e-6)
            pb_position = (pullback_pct - effective_min_pb) / pb_range
            pb_score = max(0, 0.08 - abs(pb_position - 0.45) * 0.12)
            # 成交量得分
            vol_score = min(vol_multiple / 1.8, 1.0) * 0.06
            vol_trend_bonus = 0.04 if vol_rising_trend else 0.0
            # EMA 堆叠得分（按比例线性）
            stack_score = stack_ratio * 0.06
            # RSI 得分（偏中性）
            cur_rsi = float(rsi.iloc[-1])
            rsi_ideal = 55 if direction == "LONG" else 45
            rsi_score = max(0.0, 1.0 - abs(cur_rsi - rsi_ideal) / 25.0) * 0.05
            # MACD 柱得分
            macd_score = 0.06 if (
                (direction == "LONG" and hist_turning_up)
                or (direction == "SHORT" and hist_turning_down)
            ) else 0.02
            # 波动率收窄加分
            contract_score = 0.04 if vol_contracting else 0.0
            # 慢线过滤加分
            slow_bonus = 0.03 if slow_filter_ok else 0.0
            # 蜡烛形态加分 (v4.4: 改为条件加分)
            candle_score = 0.06 if (direction == "LONG" and bullish_candle) or (direction == "SHORT" and bearish_candle) else 0.0
            # 拒测加分 (v4.4: 改为条件加分)
            retest_score = 0.04 if retest_rejected else 0.0
            # 回升确认得分
            resume_score = resumption["score"]
            confidence = min(0.95, base + bounce_score + pb_score + vol_score + vol_trend_bonus + stack_score + rsi_score + macd_score + contract_score + slow_bonus + candle_score + retest_score + resume_score)

        detail = (
            f"企稳={stabilized}({stabilized_ratio:.0%}) 延续={continuation} "
            f"回升={resumption['ok']} 反弹={bounce_ratio:.0%} 量能={vol_ok}({vol_multiple:.1f}x) "
            f"EMA序列={trend_stack_ok}({stack_ratio:.0%}) "
            f"慢线过滤={slow_filter_ok} ATR收窄={vol_contracting} "
            f"回调{effective_min_pb:.2f}%~{effective_max_pb:.2f}% | {resumption['detail']}"
        )
        return {
            "ready": ready,
            "pullback_pct": pullback_pct,
            "bounce_ratio": bounce_ratio,
            "detail": detail,
            "confidence": confidence,
        }


STRATEGY_NAME = "三分钟多周期回调企稳策略"
STRATEGY_TYPE = "trade"
STRATEGY_CLASS = ThreeMinuteMultiTimeframePullbackStrategy
BACKTEST_CLASS = ThreeMinuteMultiTimeframePullbackStrategy
