#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.strategy.runner import StrategyRunner, AutoTradeCampaign
from src.trading.executor import OrderResult, PositionInfo, PositionSide, TradeExecutor
from src.trading.position_registry import position_registry
from src.trading.scan_auto_trader import CapitalPool


class FakeOkxClient:
    def __init__(self):
        self.last_order = None

    def get_ticker(self, inst_id: str):
        return {"code": "0", "data": [{"last": "100"}]}

    def get_account_config(self):
        return {"code": "0", "data": [{"posMode": "long_short_mode", "acctLv": "2"}]}

    def get_order(self, inst_id: str, order_id: str):
        return {"code": "0", "data": [{"state": "filled", "avgPx": "100"}]}

    def _request(self, method: str, path: str, params=None, data=None):
        if path == "/api/v5/public/instruments":
            return {
                "code": "0",
                "data": [{
                    "instId": (params or {}).get("instId"),
                    "ctVal": "1",
                    "lotSz": "0.1",
                    "minSz": "0.1",
                    "tickSz": "0.1",
                }]
            }
        if path == "/api/v5/account/set-leverage":
            return {"code": "0", "data": [{"lever": str((data or {}).get("lever", ""))}]}
        if path == "/api/v5/trade/order":
            self.last_order = dict(data or {})
            return {"code": "0", "data": [{"ordId": "TP-ORDER"}]}
        raise AssertionError(f"Unexpected request: {method} {path}")


class DummyStrategy:
    def generate_signal(self, klines):
        return {"action": "HOLD"}


class FailingCloseExecutor:
    def __init__(self, pos: PositionInfo):
        self.pos = pos

    def get_positions(self, inst_id: str = None):
        return {self.pos.inst_id: self.pos}

    def execute_sell(self, inst_id: str, quantity: float):
        return OrderResult(False, message="simulated close failure")

    def execute_cover(self, inst_id: str, quantity: float):
        return OrderResult(False, message="simulated close failure")


class SuccessfulCloseExecutor:
    def __init__(self, pos: PositionInfo, filled_price: float):
        self.pos = pos
        self.filled_price = filled_price

    def get_positions(self, inst_id: str = None):
        return {self.pos.inst_id: self.pos}

    def execute_sell(self, inst_id: str, quantity: float):
        return OrderResult(True, message="ok", filled_size=quantity, filled_price=self.filled_price)

    def execute_cover(self, inst_id: str, quantity: float):
        return OrderResult(True, message="ok", filled_size=quantity, filled_price=self.filled_price)


def make_position(inst_id: str, side: PositionSide, entry: float, current: float, size: float) -> PositionInfo:
    return PositionInfo(
        inst_id=inst_id,
        side=side,
        size=size,
        entry_price=entry,
        current_price=current,
        unrealized_pnl=(current - entry) * size if side == PositionSide.LONG else (entry - current) * size,
        pnl_percent=0.0,
        leverage=1,
        notional_usd=current * size,
    )


def verify_limit_tp_order_price_uses_fill_buffer():
    client = FakeOkxClient()
    executor = TradeExecutor(client)
    result = executor.place_smart_order(
        "BTC-USDT-SWAP",
        side="buy",
        pos_side="long",
        usdt_amount=100.0,
        leverage=5,
        order_type="limit",
        price=100.0,
        tp_pct=0.05,
        sl_pct=0.03,
        tgt_type="limit",
    )
    assert result.success
    assert client.last_order is not None
    # tp_price = 105, limit buffer = 105 * 0.999 = 104.895
    assert client.last_order["tpTriggerPx"] == "105"
    assert client.last_order["tpOrdPx"] == "104.895"


def verify_close_failure_keeps_campaign_and_lock():
    inst_id = "FAIL-USDT-SWAP"
    system_name = "CloseFailureSafety"
    pos = make_position(inst_id, PositionSide.LONG, 100.0, 101.0, 1.0)
    runner = StrategyRunner(DummyStrategy(), inst_id, okx_client=None, trade_executor=FailingCloseExecutor(pos), config={"_system_name": system_name})
    runner._campaign = AutoTradeCampaign(
        direction="LONG",
        first_signal_price=100.0,
        stage1_entry_price=100.0,
        stage1_cost_line=99.0,
        stage2_entry_price=0.0,
        highest_since_stage1=101.0,
        lowest_since_stage1=100.0,
        total_allocated_usdt=100.0,
        opened_at=1.0,
        last_reason="test",
    )
    runner._pending_signal = object()
    assert position_registry.try_lock(inst_id, system_name)
    try:
        runner._close_position("force close fail")
        assert runner._campaign is not None
        assert runner._pending_signal is not None
        assert runner._close_retry_count == 1
        assert position_registry.get_owner(inst_id) == system_name
    finally:
        position_registry.release(inst_id, system_name)


def verify_consecutive_losses_uses_realized_pnl():
    inst_id = "REALIZED-USDT-SWAP"
    system_name = "RealizedLossSafety"
    # 浮盈 +1，但真实成交价 99，实际亏损 -1（未扣费前）
    pos = make_position(inst_id, PositionSide.LONG, 100.0, 101.0, 1.0)
    runner = StrategyRunner(DummyStrategy(), inst_id, okx_client=None, trade_executor=SuccessfulCloseExecutor(pos, filled_price=99.0), config={"_system_name": system_name})
    runner._campaign = AutoTradeCampaign(
        direction="LONG",
        first_signal_price=100.0,
        stage1_entry_price=100.0,
        stage1_cost_line=99.0,
        stage2_entry_price=0.0,
        highest_since_stage1=101.0,
        lowest_since_stage1=100.0,
        total_allocated_usdt=100.0,
        opened_at=1.0,
        last_reason="test",
    )
    assert position_registry.try_lock(inst_id, system_name)
    runner._close_position("realized loss close")
    assert runner._campaign is None
    assert runner._consecutive_losses == 1
    assert position_registry.get_owner(inst_id) in ("", None)


def verify_capital_pool_only_moves_with_session_realized_pnl():
    pool = CapitalPool(1000.0)
    assert pool.reserve(100.0) is True
    pool.apply_realized_pnl(-200.0, configured_cap=1000.0)
    snap = pool.snapshot()
    assert snap["total"] == 800.0
    assert snap["available"] == 700.0
    pool.tighten_total(600.0)
    snap = pool.snapshot()
    assert snap["total"] == 600.0
    assert snap["available"] == 500.0
    pool.tighten_total(900.0)
    snap = pool.snapshot()
    assert snap["total"] == 600.0
    assert snap["available"] == 500.0


def main():
    verify_limit_tp_order_price_uses_fill_buffer()
    verify_close_failure_keeps_campaign_and_lock()
    verify_consecutive_losses_uses_realized_pnl()
    verify_capital_pool_only_moves_with_session_realized_pnl()
    print("trade safety fixes verification passed")


if __name__ == "__main__":
    main()
