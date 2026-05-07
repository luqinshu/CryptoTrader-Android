#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
三分钟多周期回调企稳策略专项验证脚本

验证目标：
1. 新策略文件可正常加载，并暴露配置 schema
2. 在合成多周期数据下 generate_signal() 能稳定返回结构化结果
3. BacktestAnalyzer 能正确统计：
   - 试仓次数 / 试仓成功率
   - 二次加仓次数 / 二次加仓成功率
   - 第一原则触发次数
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STRATEGY_PATH = PROJECT_ROOT / "strategies" / "三分钟多周期回调企稳策略.py"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest.engine import (
    BacktestAnalyzer,
    BacktestEngine,
    BacktestResult,
    Trade,
    TradeDirection,
    _BacktestOkxView,
    _BacktestTradeExecutorAdapter,
    _StateMachineBacktestRunner,
)
from src.trading.position_registry import position_registry


def load_strategy_class():
    spec = importlib.util.spec_from_file_location("three_minute_strategy", STRATEGY_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.ThreeMinuteMultiTimeframePullbackStrategy


def build_rows(start: float, step_ms: int, count: int, drift: float, wave: list[float]):
    rows = []
    price = start
    ts = 1700000000000
    for i in range(count):
        price = price * (1 + drift + wave[i % len(wave)])
        op = price * 0.999
        cl = price
        hi = max(op, cl) * 1.002
        lo = min(op, cl) * 0.998
        vol = 1000 + i * 7
        rows.append([str(ts), f"{op:.6f}", f"{hi:.6f}", f"{lo:.6f}", f"{cl:.6f}", f"{vol:.4f}", "0", "0", "0"])
        ts += step_ms
    return rows


def verify_strategy_signal():
    StrategyClass = load_strategy_class()
    strategy = StrategyClass({})
    schema = strategy.get_config_schema()
    assert "h4_fast_ema" in schema
    assert "m3_pullback_max_pct" in schema

    h4 = build_rows(100.0, 4 * 60 * 60 * 1000, 120, 0.0018, [0.0004, -0.0001, 0.0003, 0.0])
    h1 = build_rows(120.0, 60 * 60 * 1000, 140, 0.0010, [0.0005, -0.0002, 0.0004, 0.0])
    m3 = build_rows(140.0, 3 * 60 * 1000, 80, 0.0002, [0.0012, 0.0009, -0.0015, -0.0009, 0.0013, 0.0010, 0.0006, 0.0004])

    signal = strategy.generate_signal({"4h": h4, "hourly": h1, "m3": m3})
    assert isinstance(signal, dict)
    assert "action" in signal
    assert signal["action"] in {"BUY", "SHORT", "HOLD", "EXIT_LONG", "EXIT_SHORT"}


def verify_backtest_metrics():
    start = datetime(2026, 4, 1)
    result = BacktestResult(
        strategy_name="三分钟多周期回调企稳策略",
        inst_id="BTC-USDT-SWAP",
        start_date=start,
        end_date=start + timedelta(days=10),
        initial_capital=10000.0,
        final_capital=10320.0,
    )
    result.trades = [
        Trade(
            entry_time=start,
            exit_time=start + timedelta(hours=2),
            direction=TradeDirection.LONG,
            entry_price=100.0,
            exit_price=103.0,
            size=10.0,
            pnl=30.0,
            pnl_percent=3.0,
            entry_reason="1%试仓 | 4H多头 + 1H延续 + 3m回调企稳",
            exit_reason="移动止盈",
        ),
        Trade(
            entry_time=start + timedelta(hours=3),
            exit_time=start + timedelta(hours=6),
            direction=TradeDirection.LONG,
            entry_price=104.0,
            exit_price=102.0,
            size=10.0,
            pnl=-20.0,
            pnl_percent=-1.92,
            entry_reason="10%加仓 | 第二次3m回调企稳",
            exit_reason="第一原则触发：跌破ATR止损线",
        ),
        Trade(
            entry_time=start + timedelta(hours=7),
            exit_time=start + timedelta(hours=10),
            direction=TradeDirection.SHORT,
            entry_price=101.0,
            exit_price=98.0,
            size=8.0,
            pnl=24.0,
            pnl_percent=2.97,
            entry_reason="10%加仓 | 第二次3m回调企稳",
            exit_reason="H1 趋势转折离场",
        ),
    ]
    result.total_trades = len(result.trades)
    result.total_pnl = sum(t.pnl for t in result.trades)
    result.equity_curve = [
        (start, 10000.0),
        (start + timedelta(days=1), 10060.0),
        (start + timedelta(days=2), 10020.0),
        (start + timedelta(days=3), 10320.0),
    ]

    analyzed = BacktestAnalyzer.analyze(result)
    assert analyzed.stats["pilot_trade_count"] == 1
    assert analyzed.stats["pilot_win_count"] == 1
    assert round(analyzed.stats["pilot_success_rate"], 2) == 100.00
    assert analyzed.stats["add_on_trade_count"] == 2
    assert analyzed.stats["add_on_win_count"] == 1
    assert round(analyzed.stats["add_on_success_rate"], 2) == 50.00
    assert analyzed.stats["first_principle_trigger_count"] == 1


def rows_from_closes(closes: list[float], step_ms: int = 180000, start_ts: int = 1700000000000):
    rows = []
    ts = start_ts
    prev = closes[0]
    for close in closes:
        op = prev
        hi = max(op, close) * 1.0015
        lo = min(op, close) * 0.9985
        rows.append([str(ts), f"{op:.6f}", f"{hi:.6f}", f"{lo:.6f}", f"{close:.6f}", "1500.0", "0", "0", "0"])
        prev = close
        ts += step_ms
    return rows


def build_trend_rows(start: float, count: int, step_ms: int, drift: float):
    closes = []
    price = start
    for i in range(count):
        price *= (1.0 + drift + (0.00015 if i % 3 == 0 else -0.00005))
        closes.append(round(price, 6))
    return rows_from_closes(closes, step_ms=step_ms)


def verify_state_machine_pilot_and_addon():
    class AlwaysLongSetupStrategy:
        def generate_signal(self, klines):
            latest = float(klines["m3"][-1][4])
            return {"action": "BUY", "entry_price": latest, "reason": "测试多头setup"}

    common_h4 = build_trend_rows(100.0, 60, 4 * 60 * 60 * 1000, 0.0020)
    common_h1 = build_trend_rows(120.0, 90, 60 * 60 * 1000, 0.0010)
    common_d1 = build_trend_rows(90.0, 40, 24 * 60 * 60 * 1000, 0.0012)

    m3_pilot = rows_from_closes([
        100.0, 100.1, 100.2, 100.3, 100.4, 100.5, 100.6, 100.7,
        100.8, 100.9, 101.0, 101.1, 101.2, 101.3, 101.4, 101.5,
        101.6, 101.7, 101.8, 101.9, 102.0, 102.1, 102.2, 102.3,
        102.4, 102.5, 102.6, 102.7, 102.8, 102.9, 103.0, 103.1,
        103.2, 103.4, 103.6, 103.8, 104.0, 104.2, 104.4, 104.6,
        104.8, 105.0, 104.0, 104.2, 104.4, 104.6, 104.8, 105.0,
    ])
    m3_cont = rows_from_closes([
        104.2, 104.3, 104.4, 104.5, 104.6, 104.7, 104.8, 104.9,
        105.0, 105.1, 105.15, 105.2, 105.25, 105.3, 105.35, 105.4,
        105.45, 105.5, 105.55, 105.6, 105.65, 105.7, 105.72, 105.74,
        105.76, 105.78, 105.80, 105.82, 105.84, 105.86, 105.88, 105.90,
        105.92, 105.94, 105.96, 105.98, 106.00, 106.02, 106.04, 106.06,
        106.08, 106.10, 106.12, 106.14, 106.16, 106.18, 106.20, 106.25,
    ], start_ts=1700000000000 + 48 * 180000)
    m3_add = rows_from_closes([
        105.1, 105.2, 105.3, 105.4, 105.5, 105.6, 105.7, 105.8,
        105.9, 106.0, 106.1, 106.2, 106.25, 106.3, 106.32, 106.34,
        106.36, 106.38, 106.40, 106.42, 106.44, 106.46, 106.48, 106.50,
        106.52, 106.54, 106.56, 106.58, 106.60, 106.62, 106.64, 106.66,
        106.68, 106.70, 106.72, 106.74, 106.76, 106.78, 105.50, 105.60,
        105.70, 105.80, 105.90, 106.00, 106.10, 106.20, 106.30, 106.40,
    ], start_ts=1700000000000 + 96 * 180000)

    engine = BacktestEngine(initial_capital=10000.0)
    okx_view = _BacktestOkxView()
    executor = _BacktestTradeExecutorAdapter(engine, "BTC-USDT-SWAP")
    runner = _StateMachineBacktestRunner(
        AlwaysLongSetupStrategy(),
        "BTC-USDT-SWAP",
        okx_view,
        executor,
        {
            "_system_name": "StateMachineTest",
            "allow_short": False,
            "auto_trading_capital": 10000.0,
            "pilot_position_pct": 0.01,
            "add_position_pct": 0.10,
            "leverage": 1,
            "m3_pullback_min_pct": 0.35,
            "m3_pullback_max_pct": 1.60,
            "m3_breakout_hold_buffer_pct": 1.50,
            "m3_stabilization_bars": 3,
            "stop_loss_floor_pct": 0.80,
            "cost_line_atr_multiplier": 1.20,
            "volume_confirm_ratio": 0.50,
            "rsi_oversold": 0.0,
            "stage1_trail_activate_pct": 99.0,
            "trail_stop_pct": 99.0,
            "partial_exit_trigger_pct": 99.0,
        },
    )
    runner._check_hourly_reversal_exit = lambda market: False

    try:
        t1 = datetime(2026, 4, 1, 0, 0)
        okx_view.set_price(float(m3_pilot[-1][4]))
        executor.set_market(t1, float(m3_pilot[-1][4]), leverage=1)
        runner.step(t1, {"action": "BUY", "entry_price": float(m3_pilot[-1][4]), "reason": "测试多头setup"}, {
            "daily": common_d1,
            "4h": common_h4,
            "hourly": common_h1,
            "m15": m3_pilot,
            "m3": m3_pilot,
        })
        assert runner.metrics["pilot_trade_count"] == 1
        assert runner._campaign is not None
        assert runner._campaign.stage2_filled is False
        assert engine.current_trade is not None
        assert engine.position > 0

        t2 = t1 + timedelta(minutes=3)
        okx_view.set_price(float(m3_cont[-1][4]))
        executor.set_market(t2, float(m3_cont[-1][4]), leverage=1)
        runner.step(t2, {"action": "BUY", "entry_price": float(m3_cont[-1][4]), "reason": "测试多头setup"}, {
            "daily": common_d1,
            "4h": common_h4,
            "hourly": common_h1,
            "m15": m3_cont,
            "m3": m3_cont,
        })
        assert runner._campaign is not None
        assert runner._campaign.stage2_armed is True
        assert runner.metrics["add_on_trade_count"] == 0

        t3 = t2 + timedelta(minutes=3)
        okx_view.set_price(float(m3_add[-1][4]))
        executor.set_market(t3, float(m3_add[-1][4]), leverage=1)
        runner.step(t3, {"action": "BUY", "entry_price": float(m3_add[-1][4]), "reason": "测试多头setup"}, {
            "daily": common_d1,
            "4h": common_h4,
            "hourly": common_h1,
            "m15": m3_add,
            "m3": m3_add,
        })
        assert runner.metrics["add_on_trade_count"] == 1
        assert runner._campaign is not None
        assert runner._campaign.stage2_filled is True
        assert engine.current_trade is not None
        assert engine.position > 0
    finally:
        position_registry.release("BTC-USDT-SWAP", "StateMachineTest")


def verify_state_machine_first_principle_exit():
    class AlwaysLongSetupStrategy:
        def generate_signal(self, klines):
            latest = float(klines["m3"][-1][4])
            return {"action": "BUY", "entry_price": latest, "reason": "测试第一原则"}

    common_h4 = build_trend_rows(100.0, 60, 4 * 60 * 60 * 1000, 0.0020)
    common_h1 = build_trend_rows(120.0, 90, 60 * 60 * 60 * 1000, 0.0010)
    common_d1 = build_trend_rows(90.0, 40, 24 * 60 * 60 * 1000, 0.0012)

    m3_entry = rows_from_closes([
        100.0, 100.1, 100.2, 100.3, 100.4, 100.5, 100.6, 100.7,
        100.8, 100.9, 101.0, 101.1, 101.2, 101.3, 101.4, 101.5,
        101.6, 101.7, 101.8, 101.9, 102.0, 102.1, 102.2, 102.3,
        102.4, 102.5, 102.6, 102.7, 102.8, 102.9, 103.0, 103.1,
        103.2, 103.4, 103.6, 103.8, 104.0, 104.2, 104.4, 104.6,
        104.8, 105.0, 104.0, 104.2, 104.4, 104.6, 104.8, 105.0,
    ])
    # 最后一根价格明显跌破 stage1_cost_line(约 104.16)，必须触发第一原则离场
    m3_breach = rows_from_closes([
        104.8, 104.7, 104.6, 104.5, 104.4, 104.3, 104.2, 104.1,
        104.0, 103.9, 103.8, 103.7, 103.6, 103.5, 103.4, 103.3,
        103.2, 103.1, 103.0, 102.9, 102.8, 102.7, 102.6, 102.5,
        102.4, 102.3, 102.2, 102.1, 102.0, 101.9, 101.8, 101.7,
        101.6, 101.5, 101.4, 101.3, 101.2, 101.1, 101.0, 100.9,
        100.8, 100.7, 100.6, 100.5, 100.4, 100.3, 100.2, 100.1,
    ], start_ts=1700000000000 + 48 * 180000)

    engine = BacktestEngine(initial_capital=10000.0)
    okx_view = _BacktestOkxView()
    executor = _BacktestTradeExecutorAdapter(engine, "ETH-USDT-SWAP")
    runner = _StateMachineBacktestRunner(
        AlwaysLongSetupStrategy(),
        "ETH-USDT-SWAP",
        okx_view,
        executor,
        {
            "_system_name": "FirstPrincipleTest",
            "allow_short": False,
            "auto_trading_capital": 10000.0,
            "pilot_position_pct": 0.01,
            "add_position_pct": 0.10,
            "leverage": 1,
            "m3_pullback_min_pct": 0.35,
            "m3_pullback_max_pct": 1.60,
            "m3_breakout_hold_buffer_pct": 1.50,
            "m3_stabilization_bars": 3,
            "stop_loss_floor_pct": 0.80,
            "cost_line_atr_multiplier": 1.20,
            "volume_confirm_ratio": 0.50,
            "rsi_oversold": 0.0,
            "stage1_trail_activate_pct": 99.0,
            "trail_stop_pct": 99.0,
            "partial_exit_trigger_pct": 99.0,
        },
    )
    runner._check_hourly_reversal_exit = lambda market: False

    try:
        t1 = datetime(2026, 4, 2, 0, 0)
        okx_view.set_price(float(m3_entry[-1][4]))
        executor.set_market(t1, float(m3_entry[-1][4]), leverage=1)
        runner.step(t1, {"action": "BUY", "entry_price": float(m3_entry[-1][4]), "reason": "测试第一原则"}, {
            "daily": common_d1,
            "4h": common_h4,
            "hourly": common_h1,
            "m15": m3_entry,
            "m3": m3_entry,
        })
        assert runner.metrics["pilot_trade_count"] == 1
        assert runner._campaign is not None
        assert engine.position > 0

        t2 = t1 + timedelta(minutes=3)
        okx_view.set_price(float(m3_breach[-1][4]))
        executor.set_market(t2, float(m3_breach[-1][4]), leverage=1)
        runner.step(t2, {"action": "BUY", "entry_price": float(m3_breach[-1][4]), "reason": "测试第一原则"}, {
            "daily": common_d1,
            "4h": common_h4,
            "hourly": common_h1,
            "m15": m3_breach,
            "m3": m3_breach,
        })
        assert runner.metrics["first_principle_trigger_count"] == 1
        assert runner.metrics["campaign_close_count"] == 1
        assert runner._campaign is None
        assert engine.position == 0
        assert len(engine.trades) == 1
        assert "第一原则" in engine.trades[-1].exit_reason
    finally:
        position_registry.release("ETH-USDT-SWAP", "FirstPrincipleTest")


def build_reversal_h1_rows(start: float, count: int, step_ms: int):
    closes = []
    price = start
    for i in range(count):
        if i < count - 12:
            price *= 1.0012
        else:
            price *= 0.9935
        closes.append(round(price, 6))
    return rows_from_closes(closes, step_ms=step_ms)


def build_short_trend_rows(start: float, count: int, step_ms: int, drift: float):
    closes = []
    price = start
    for i in range(count):
        price *= (1.0 - drift + (-0.00015 if i % 3 == 0 else 0.00005))
        closes.append(round(price, 6))
    return rows_from_closes(closes, step_ms=step_ms)


def build_short_reversal_h1_rows(start: float, count: int, step_ms: int):
    closes = []
    price = start
    for i in range(count):
        if i < count - 12:
            price *= 0.9988
        else:
            price *= 1.0065
        closes.append(round(price, 6))
    return rows_from_closes(closes, step_ms=step_ms)


def verify_state_machine_h1_reversal_exit():
    class AlwaysLongSetupStrategy:
        def generate_signal(self, klines):
            latest = float(klines["m3"][-1][4])
            return {"action": "BUY", "entry_price": latest, "reason": "测试H1转折离场"}

    common_h4 = build_trend_rows(100.0, 60, 4 * 60 * 60 * 1000, 0.0020)
    common_d1 = build_trend_rows(90.0, 40, 24 * 60 * 60 * 1000, 0.0012)
    healthy_h1 = build_trend_rows(120.0, 90, 60 * 60 * 60 * 1000, 0.0010)
    reversal_h1 = build_reversal_h1_rows(130.0, 90, 60 * 60 * 60 * 1000)

    m3_entry = rows_from_closes([
        100.0, 100.1, 100.2, 100.3, 100.4, 100.5, 100.6, 100.7,
        100.8, 100.9, 101.0, 101.1, 101.2, 101.3, 101.4, 101.5,
        101.6, 101.7, 101.8, 101.9, 102.0, 102.1, 102.2, 102.3,
        102.4, 102.5, 102.6, 102.7, 102.8, 102.9, 103.0, 103.1,
        103.2, 103.4, 103.6, 103.8, 104.0, 104.2, 104.4, 104.6,
        104.8, 105.0, 104.0, 104.2, 104.4, 104.6, 104.8, 105.0,
    ])

    # 价格仍高于第一原则成本线，但 H1 已经明显转弱，应该走趋势转折离场而不是成本线止损
    m3_hold = rows_from_closes([
        105.2, 105.3, 105.4, 105.5, 105.6, 105.7, 105.8, 105.9,
        106.0, 106.1, 106.2, 106.3, 106.4, 106.5, 106.6, 106.7,
        106.8, 106.9, 107.0, 107.1, 107.2, 107.3, 107.4, 107.5,
        107.6, 107.7, 107.8, 107.9, 108.0, 108.1, 108.2, 108.3,
        108.4, 108.5, 108.6, 108.7, 108.8, 108.9, 109.0, 109.1,
        109.2, 109.1, 109.0, 108.9, 108.8, 108.7, 108.6, 108.5,
    ], start_ts=1700000000000 + 48 * 180000)

    engine = BacktestEngine(initial_capital=10000.0)
    okx_view = _BacktestOkxView()
    executor = _BacktestTradeExecutorAdapter(engine, "SOL-USDT-SWAP")
    runner = _StateMachineBacktestRunner(
        AlwaysLongSetupStrategy(),
        "SOL-USDT-SWAP",
        okx_view,
        executor,
        {
            "_system_name": "H1ReversalTest",
            "allow_short": False,
            "auto_trading_capital": 10000.0,
            "pilot_position_pct": 0.01,
            "add_position_pct": 0.10,
            "leverage": 1,
            "m3_pullback_min_pct": 0.35,
            "m3_pullback_max_pct": 1.60,
            "m3_breakout_hold_buffer_pct": 1.50,
            "m3_stabilization_bars": 3,
            "stop_loss_floor_pct": 0.80,
            "cost_line_atr_multiplier": 1.20,
            "volume_confirm_ratio": 0.50,
            "rsi_oversold": 0.0,
            "stage1_trail_activate_pct": 99.0,
            "trail_stop_pct": 99.0,
            "partial_exit_trigger_pct": 99.0,
            "h1_large_pullback_pct": 2.0,
            "h1_reversal_min_hold_hours": 0.0,
        },
    )

    try:
        t1 = datetime(2026, 4, 3, 0, 0)
        okx_view.set_price(float(m3_entry[-1][4]))
        executor.set_market(t1, float(m3_entry[-1][4]), leverage=1)
        runner.step(t1, {"action": "BUY", "entry_price": float(m3_entry[-1][4]), "reason": "测试H1转折离场"}, {
            "daily": common_d1,
            "4h": common_h4,
            "hourly": healthy_h1,
            "m15": m3_entry,
            "m3": m3_entry,
        })
        assert runner.metrics["pilot_trade_count"] == 1
        assert runner._campaign is not None
        assert engine.position > 0

        t2 = t1 + timedelta(minutes=3)
        okx_view.set_price(float(m3_hold[-1][4]))
        executor.set_market(t2, float(m3_hold[-1][4]), leverage=1)
        runner.step(t2, {"action": "BUY", "entry_price": float(m3_hold[-1][4]), "reason": "测试H1转折离场"}, {
            "daily": common_d1,
            "4h": common_h4,
            "hourly": reversal_h1,
            "m15": m3_hold,
            "m3": m3_hold,
        })
        assert runner._campaign is None
        assert engine.position == 0
        assert len(engine.trades) == 1
        assert "H1 趋势转折离场" in engine.trades[-1].exit_reason
        assert runner.metrics["first_principle_trigger_count"] == 0
    finally:
        position_registry.release("SOL-USDT-SWAP", "H1ReversalTest")


def verify_state_machine_short_first_principle_exit():
    class AlwaysShortSetupStrategy:
        def generate_signal(self, klines):
            latest = float(klines["m3"][-1][4])
            return {"action": "SHORT", "entry_price": latest, "reason": "测试空头第一原则"}

    common_h4 = build_short_trend_rows(200.0, 60, 4 * 60 * 60 * 1000, 0.0018)
    common_h1 = build_short_trend_rows(180.0, 90, 60 * 60 * 60 * 1000, 0.0010)
    common_d1 = build_short_trend_rows(220.0, 40, 24 * 60 * 60 * 1000, 0.0012)

    m3_entry = rows_from_closes([
        110.0, 109.9, 109.8, 109.7, 109.6, 109.5, 109.4, 109.3,
        109.2, 109.1, 109.0, 108.9, 108.8, 108.7, 108.6, 108.5,
        108.4, 108.3, 108.2, 108.1, 108.0, 107.9, 107.8, 107.7,
        107.6, 107.5, 107.4, 107.3, 107.2, 107.1, 107.0, 106.9,
        106.8, 106.6, 106.4, 106.2, 106.0, 105.8, 105.6, 105.4,
        105.2, 105.0, 106.0, 105.8, 105.6, 105.4, 105.2, 105.0,
    ])
    # 最后一根价格明显上破空头成本线(约105.84)，必须触发第一原则离场
    m3_breach = rows_from_closes([
        105.2, 105.3, 105.4, 105.5, 105.6, 105.7, 105.8, 105.9,
        106.0, 106.1, 106.2, 106.3, 106.4, 106.5, 106.6, 106.7,
        106.8, 106.9, 107.0, 107.1, 107.2, 107.3, 107.4, 107.5,
        107.6, 107.7, 107.8, 107.9, 108.0, 108.1, 108.2, 108.3,
        108.4, 108.5, 108.6, 108.7, 108.8, 108.9, 109.0, 109.1,
        109.2, 109.3, 109.4, 109.5, 109.6, 109.7, 109.8, 109.9,
    ], start_ts=1700000000000 + 48 * 180000)

    engine = BacktestEngine(initial_capital=10000.0)
    okx_view = _BacktestOkxView()
    executor = _BacktestTradeExecutorAdapter(engine, "XRP-USDT-SWAP")
    runner = _StateMachineBacktestRunner(
        AlwaysShortSetupStrategy(),
        "XRP-USDT-SWAP",
        okx_view,
        executor,
        {
            "_system_name": "ShortFirstPrincipleTest",
            "allow_short": True,
            "auto_trading_capital": 10000.0,
            "pilot_position_pct": 0.01,
            "add_position_pct": 0.10,
            "leverage": 1,
            "m3_pullback_min_pct": 0.35,
            "m3_pullback_max_pct": 1.60,
            "m3_breakout_hold_buffer_pct": 1.50,
            "m3_stabilization_bars": 3,
            "stop_loss_floor_pct": 0.80,
            "cost_line_atr_multiplier": 1.20,
            "volume_confirm_ratio": 0.50,
            "rsi_oversold": 0.0,
            "stage1_trail_activate_pct": 99.0,
            "trail_stop_pct": 99.0,
            "partial_exit_trigger_pct": 99.0,
        },
    )
    runner._check_hourly_reversal_exit = lambda market: False

    try:
        t1 = datetime(2026, 4, 4, 0, 0)
        okx_view.set_price(float(m3_entry[-1][4]))
        executor.set_market(t1, float(m3_entry[-1][4]), leverage=1)
        runner.step(t1, {"action": "SHORT", "entry_price": float(m3_entry[-1][4]), "reason": "测试空头第一原则"}, {
            "daily": common_d1,
            "4h": common_h4,
            "hourly": common_h1,
            "m15": m3_entry,
            "m3": m3_entry,
        })
        assert runner.metrics["pilot_trade_count"] == 1
        assert runner._campaign is not None
        assert engine.position > 0

        t2 = t1 + timedelta(minutes=3)
        okx_view.set_price(float(m3_breach[-1][4]))
        executor.set_market(t2, float(m3_breach[-1][4]), leverage=1)
        runner.step(t2, {"action": "SHORT", "entry_price": float(m3_breach[-1][4]), "reason": "测试空头第一原则"}, {
            "daily": common_d1,
            "4h": common_h4,
            "hourly": common_h1,
            "m15": m3_breach,
            "m3": m3_breach,
        })
        assert runner.metrics["first_principle_trigger_count"] == 1
        assert runner.metrics["campaign_close_count"] == 1
        assert runner._campaign is None
        assert engine.position == 0
        assert len(engine.trades) == 1
        assert "第一原则" in engine.trades[-1].exit_reason
    finally:
        position_registry.release("XRP-USDT-SWAP", "ShortFirstPrincipleTest")


def verify_state_machine_short_h1_reversal_exit():
    class AlwaysShortSetupStrategy:
        def generate_signal(self, klines):
            latest = float(klines["m3"][-1][4])
            return {"action": "SHORT", "entry_price": latest, "reason": "测试空头H1转折离场"}

    common_h4 = build_short_trend_rows(200.0, 60, 4 * 60 * 60 * 1000, 0.0018)
    common_d1 = build_short_trend_rows(220.0, 40, 24 * 60 * 60 * 1000, 0.0012)
    healthy_h1 = build_short_trend_rows(180.0, 90, 60 * 60 * 60 * 1000, 0.0010)
    reversal_h1 = build_short_reversal_h1_rows(170.0, 90, 60 * 60 * 60 * 1000)

    m3_entry = rows_from_closes([
        110.0, 109.9, 109.8, 109.7, 109.6, 109.5, 109.4, 109.3,
        109.2, 109.1, 109.0, 108.9, 108.8, 108.7, 108.6, 108.5,
        108.4, 108.3, 108.2, 108.1, 108.0, 107.9, 107.8, 107.7,
        107.6, 107.5, 107.4, 107.3, 107.2, 107.1, 107.0, 106.9,
        106.8, 106.6, 106.4, 106.2, 106.0, 105.8, 105.6, 105.4,
        105.2, 105.0, 106.0, 105.8, 105.6, 105.4, 105.2, 105.0,
    ])
    # 价格仍低于空头成本线，但 H1 已明显转强，应该走 H1 趋势转折离场
    m3_hold = rows_from_closes([
        104.8, 104.7, 104.6, 104.5, 104.4, 104.3, 104.2, 104.1,
        104.0, 103.9, 103.8, 103.7, 103.6, 103.5, 103.4, 103.3,
        103.2, 103.1, 103.0, 102.9, 102.8, 102.7, 102.6, 102.5,
        102.4, 102.3, 102.2, 102.1, 102.0, 101.9, 101.8, 101.7,
        101.6, 101.5, 101.4, 101.3, 101.2, 101.1, 101.0, 100.9,
        100.8, 100.9, 101.0, 101.1, 101.2, 101.3, 101.4, 101.5,
    ], start_ts=1700000000000 + 48 * 180000)

    engine = BacktestEngine(initial_capital=10000.0)
    okx_view = _BacktestOkxView()
    executor = _BacktestTradeExecutorAdapter(engine, "DOGE-USDT-SWAP")
    runner = _StateMachineBacktestRunner(
        AlwaysShortSetupStrategy(),
        "DOGE-USDT-SWAP",
        okx_view,
        executor,
        {
            "_system_name": "ShortH1ReversalTest",
            "allow_short": True,
            "auto_trading_capital": 10000.0,
            "pilot_position_pct": 0.01,
            "add_position_pct": 0.10,
            "leverage": 1,
            "m3_pullback_min_pct": 0.35,
            "m3_pullback_max_pct": 1.60,
            "m3_breakout_hold_buffer_pct": 1.50,
            "m3_stabilization_bars": 3,
            "stop_loss_floor_pct": 0.80,
            "cost_line_atr_multiplier": 1.20,
            "volume_confirm_ratio": 0.50,
            "rsi_oversold": 0.0,
            "stage1_trail_activate_pct": 99.0,
            "trail_stop_pct": 99.0,
            "partial_exit_trigger_pct": 99.0,
            "h1_large_pullback_pct": 2.0,
            "h1_reversal_min_hold_hours": 0.0,
        },
    )

    try:
        t1 = datetime(2026, 4, 5, 0, 0)
        okx_view.set_price(float(m3_entry[-1][4]))
        executor.set_market(t1, float(m3_entry[-1][4]), leverage=1)
        runner.step(t1, {"action": "SHORT", "entry_price": float(m3_entry[-1][4]), "reason": "测试空头H1转折离场"}, {
            "daily": common_d1,
            "4h": common_h4,
            "hourly": healthy_h1,
            "m15": m3_entry,
            "m3": m3_entry,
        })
        assert runner.metrics["pilot_trade_count"] == 1
        assert runner._campaign is not None
        assert engine.position > 0

        t2 = t1 + timedelta(minutes=3)
        okx_view.set_price(float(m3_hold[-1][4]))
        executor.set_market(t2, float(m3_hold[-1][4]), leverage=1)
        runner.step(t2, {"action": "SHORT", "entry_price": float(m3_hold[-1][4]), "reason": "测试空头H1转折离场"}, {
            "daily": common_d1,
            "4h": common_h4,
            "hourly": reversal_h1,
            "m15": m3_hold,
            "m3": m3_hold,
        })
        assert runner._campaign is None
        assert engine.position == 0
        assert len(engine.trades) == 1
        assert "H1 趋势转折离场" in engine.trades[-1].exit_reason
        assert runner.metrics["first_principle_trigger_count"] == 0
    finally:
        position_registry.release("DOGE-USDT-SWAP", "ShortH1ReversalTest")


def main():
    verify_strategy_signal()
    verify_backtest_metrics()
    verify_state_machine_pilot_and_addon()
    verify_state_machine_first_principle_exit()
    verify_state_machine_h1_reversal_exit()
    verify_state_machine_short_first_principle_exit()
    verify_state_machine_short_h1_reversal_exit()
    print("three-minute strategy backtest verification passed")


if __name__ == "__main__":
    main()
