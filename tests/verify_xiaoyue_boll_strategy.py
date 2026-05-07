#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

STRATEGY_PATH = PROJECT_ROOT / "strategies" / "小月期货多周期布林趋势转折.py"

from src.strategy.runner import StrategyRunner
from src.scanner.base_scanner import ScannerSymbol
from src.trading.executor import OrderResult


def _load_module():
    spec = importlib.util.spec_from_file_location("xiaoyue_verify", STRATEGY_PATH)
    assert spec and spec.loader, "策略模块加载失败"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeOkxClient:
    def get_ticker(self, inst_id: str):
        return {"code": "0", "data": [{"last": "100"}]}


class CaptureExecutor:
    def __init__(self):
        self.last_entry = None

    def get_positions(self, inst_id: str = None):
        return {}

    def execute_entry(
        self,
        inst_id: str,
        direction: str,
        usdt_amount: float,
        leverage: int,
        tp_pct: float,
        sl_pct: float,
        order_type: str,
        price=None,
    ):
        self.last_entry = {
            "inst_id": inst_id,
            "direction": direction,
            "usdt_amount": usdt_amount,
            "leverage": leverage,
            "tp_pct": tp_pct,
            "sl_pct": sl_pct,
            "order_type": order_type,
            "price": price,
        }
        return OrderResult(True, message="ok")


class DummyStrategy:
    def generate_signal(self, klines):
        return {"action": "HOLD"}


def verify_presets_load():
    module = _load_module()
    strategy = module.XiaoYueBollMacdScanner({"preset": "major_conservative"})
    assert strategy.config["allow_short"] is False
    assert float(strategy.config["min_volume_24h"]) >= 20_000_000
    assert getattr(module, "BATCH_PRESET_LABEL", "") == "批量回测小月布林趋势模板"
    assert "altcoin_aggressive" in getattr(module, "PRESET_CONFIGS", {})
    schema = strategy.get_config_schema()
    assert "preset" in schema
    assert schema["preset"]["default"] == "custom"


def verify_generate_signal_maps_bear_to_short_and_exports_risk():
    module = _load_module()
    strategy = module.XiaoYueBollMacdScanner()

    def fake_scan_symbol(_symbol):
        return {
            "passed": True,
            "direction": "SELL",
            "score": 88.5,
            "signals": ["空头 88.5分"],
            "details": {"ATR止损": "95", "ATR止盈": "88"},
            "stop_loss_pct": 2.4,
            "take_profit_pct": 4.0,
        }

    strategy.scan_symbol = fake_scan_symbol
    signal = strategy.generate_signal({
        "inst_id": "BTC-USDT-SWAP",
        "last_price": 100.0,
        "volume_24h": 50_000_000,
        "klines_map": {},
    })
    assert signal is not None
    assert signal["action"] == "SHORT"
    assert float(signal["stop_loss_pct"]) == 2.4
    assert float(signal["take_profit_pct"]) == 4.0


def verify_runner_passes_strategy_tp_sl_override_to_executor():
    executor = CaptureExecutor()
    runner = StrategyRunner(
        DummyStrategy(),
        inst_id="BTC-USDT-SWAP",
        okx_client=FakeOkxClient(),
        trade_executor=executor,
        config={"take_profit_pct": 5.0, "stop_loss_pct": 3.0, "leverage": 3},
    )
    signal = {
        "action": "SHORT",
        "entry_price": 100.0,
        "reason": "策略空头测试",
        "take_profit_pct": 4.2,
        "stop_loss_pct": 2.1,
    }
    runner._register_pending_signal(signal, {"latest_price": 100.0})
    pending = runner._pending_signal
    assert pending is not None
    assert float(pending.tp_pct_override) == 4.2
    assert float(pending.sl_pct_override) == 2.1

    result = runner._open_position(
        direction="SHORT",
        usdt_amount=100.0,
        reason="test",
        tp_pct_override=pending.tp_pct_override,
        sl_pct_override=pending.sl_pct_override,
    )
    assert result.success
    assert executor.last_entry is not None
    assert executor.last_entry["direction"] == "SHORT"
    assert abs(float(executor.last_entry["tp_pct"]) - 0.042) < 1e-9
    assert abs(float(executor.last_entry["sl_pct"]) - 0.021) < 1e-9


def _rows_from_closes(closes, start_ts=1700000000000, step_ms=3600000, volume_base=1000.0):
    rows = []
    prev = float(closes[0])
    for idx, close in enumerate(closes):
        close = float(close)
        open_price = prev
        high = max(open_price, close) + 0.35
        low = min(open_price, close) - 0.35
        volume = volume_base + idx * 3
        rows.append([start_ts + idx * step_ms, open_price, high, low, close, volume])
        prev = close
    return rows


def _make_bull_symbol():
    daily = _rows_from_closes(
        [100 + i * 1.4 for i in range(55)],
        step_ms=24 * 60 * 60 * 1000,
        volume_base=5000,
    )
    h1 = _rows_from_closes(
        [
            130, 131, 132, 133, 134, 135, 136, 137, 138, 139,
            140, 141, 142, 143, 144, 145, 146, 147, 148, 149,
            150, 151, 152, 153, 154, 155, 156, 157, 158, 159,
            160, 159.2, 158.7, 158.4, 158.2, 158.1, 158.3, 158.5,
            158.8, 159.1, 159.4, 159.7,
        ],
        volume_base=2200,
    )
    m15 = _rows_from_closes(
        [
            159.0, 158.8, 158.6, 158.5, 158.4, 158.3, 158.2, 158.1, 158.0, 157.9,
            157.8, 157.7, 157.6, 157.55, 157.5, 157.45, 157.4, 157.5, 157.7, 158.0,
            158.4, 158.9, 159.5, 160.2, 161.0, 161.4, 161.2, 160.8, 160.1, 159.3,
            158.6, 158.0, 157.5, 157.2, 157.0, 165.5,
        ],
        step_ms=15 * 60 * 1000,
        volume_base=800,
    )
    m15[-1][5] = 2200.0
    return ScannerSymbol(
        inst_id="BULL-USDT-SWAP",
        last_price=float(m15[-1][4]),
        volume_24h=50_000_000,
        extra_data={"klines": {"1D": daily, "1H": h1, "15m": m15}},
    )


def _make_bear_symbol():
    daily = _rows_from_closes(
        [200 - i * 1.5 for i in range(55)],
        step_ms=24 * 60 * 60 * 1000,
        volume_base=5200,
    )
    h1 = _rows_from_closes(
        [
            180, 179, 178, 177, 176, 175, 174, 173, 172, 171,
            170, 169, 168, 167, 166, 165, 164, 163, 162, 161,
            160, 159, 158, 157, 156, 155, 154, 153, 152, 151,
            150, 150.8, 151.3, 151.6, 151.8, 151.9, 151.7, 151.5,
            151.2, 150.9, 150.6, 150.3,
        ],
        volume_base=2200,
    )
    m15 = _rows_from_closes(
        [
            151.0, 151.2, 151.4, 151.5, 151.6, 151.7, 151.8, 151.9, 152.0, 152.1,
            152.2, 152.3, 152.4, 152.45, 152.5, 152.55, 152.6, 152.5, 152.3, 152.0,
            151.6, 151.1, 150.5, 149.8, 149.0, 148.6, 148.8, 149.2, 149.9, 150.7,
            151.4, 152.0, 152.5, 152.8, 153.0, 144.5,
        ],
        step_ms=15 * 60 * 1000,
        volume_base=800,
    )
    m15[-1][5] = 2200.0
    return ScannerSymbol(
        inst_id="BEAR-USDT-SWAP",
        last_price=float(m15[-1][4]),
        volume_24h=42_000_000,
        extra_data={"klines": {"1D": daily, "1H": h1, "15m": m15}},
    )


def verify_scan_symbol_real_buy_and_short_paths():
    module = _load_module()
    scanner = module.XiaoYueBollMacdScanner({
        "min_score": 40.0,
        "h1_rsi_gate_enabled": False,
        "m15_volume_required": True,
        "allow_short": True,
        "mid_band_proximity_floor": 2.0,
        "mid_band_proximity_atr": 0.8,
        "m15_macd_fast": 6,
        "m15_macd_slow": 13,
        "m15_macd_signal": 4,
    })
    bull = scanner.scan_symbol(_make_bull_symbol())
    bear = scanner.scan_symbol(_make_bear_symbol())
    assert bull["passed"] is True, bull
    assert bull["direction"] == "BUY", bull
    assert float(bull["stop_loss_pct"]) > 0
    assert float(bull["take_profit_pct"]) > float(bull["stop_loss_pct"])
    assert bear["passed"] is True, bear
    assert bear["direction"] == "SELL", bear
    assert float(bear["stop_loss_pct"]) > 0
    assert float(bear["take_profit_pct"]) > float(bear["stop_loss_pct"])


def main():
    verify_presets_load()
    verify_generate_signal_maps_bear_to_short_and_exports_risk()
    verify_runner_passes_strategy_tp_sl_override_to_executor()
    verify_scan_symbol_real_buy_and_short_paths()
    print("xiaoyue boll strategy verification passed")


if __name__ == "__main__":
    main()
