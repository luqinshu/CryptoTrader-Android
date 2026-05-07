#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.trading.executor import TradeExecutor
from src.trading.paper_engine import PaperTradeEngine
from src.trading.scan_auto_trader import CapitalPool


class FakeOkxClient:
    def __init__(self):
        self.last_order = None
        self.last_leverage_request = None

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
            self.last_leverage_request = dict(data or {})
            return {"code": "0", "data": [{"lever": str((data or {}).get("lever", ""))}]}
        if path == "/api/v5/trade/order":
            self.last_order = dict(data or {})
            return {"code": "0", "data": [{"ordId": "TEST-ORDER"}]}
        raise AssertionError(f"Unexpected request: {method} {path}")


def verify_executor_uses_notional_for_swap_size():
    client = FakeOkxClient()
    executor = TradeExecutor(client)
    result = executor.place_smart_order(
        "BTC-USDT-SWAP",
        side="buy",
        pos_side="long",
        usdt_amount=100.0,
        leverage=5,
        order_type="market",
    )
    assert result.success
    assert client.last_leverage_request is not None
    assert client.last_leverage_request["lever"] == "5"
    assert client.last_order is not None
    # 名义价值 = 100 * 5 = 500；价格 = 100；ctVal = 1 => sz = 5
    assert client.last_order["sz"] == "5.0"


def verify_paper_engine_matches_real_notional_sizing():
    client = FakeOkxClient()
    paper = PaperTradeEngine(
        client,
        initial_capital=10000.0,
        fee_pct=0.0,
        slippage_pct=0.0,
        market_impact_pct=0.0,
        funding_rate_8h_pct=0.0,
    )
    result = paper.execute_entry(
        "BTC-USDT-SWAP",
        "LONG",
        usdt_amount=100.0,
        leverage=5,
        tp_pct=0.05,
        sl_pct=0.03,
    )
    assert result.success
    pos = paper._positions["BTC-USDT-SWAP"]
    assert round(pos.size, 8) == 5.0
    assert pos.margin == 100.0
    assert pos.usdt_amount == 100.0


def verify_capital_pool_tracks_margin_not_notional():
    pool = CapitalPool(1000.0)
    assert pool.reserve(100.0) is True
    assert round(pool.available, 8) == 900.0
    pool.release(100.0)
    assert round(pool.available, 8) == 1000.0


def main():
    verify_executor_uses_notional_for_swap_size()
    verify_paper_engine_matches_real_notional_sizing()
    verify_capital_pool_tracks_margin_not_notional()
    print("leverage/notional consistency verification passed")


if __name__ == "__main__":
    main()
