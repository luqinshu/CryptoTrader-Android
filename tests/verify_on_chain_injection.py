"""
验证链上数据源能注入 ScannerSymbol.extra_data['on_chain']。

只使用本地临时 JSON，不请求外网。
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from src.data.on_chain_provider import OnChainDataProvider
from src.scanner.base_scanner import ScannerSymbol
from src.scanner.engine import ScanEngine


class DummyOKX:
    pass


def main():
    os.environ.pop("NANSEN_API_KEY", None)
    os.environ.pop("CRYPTOQUANT_API_KEY", None)
    os.environ.pop("CRYPTOQUANT_DATA_URL", None)

    payload = {
        "BTC": {
            "whale_flow": 0.012,
            "exchange_netflow": -1250,
            "active_addresses": 850000,
            "nvt_ratio": 42.5,
        },
        "ETH-USDT-SWAP": {
            "whale_netflow": 0.008,
            "exchange_net_flow": -900,
            "activeAddresses": 620000,
            "nvt": 36.0,
        },
    }

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "on_chain.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        os.environ["ON_CHAIN_DATA_SOURCE"] = "json"
        os.environ["ON_CHAIN_DATA_PATH"] = str(path)

        provider = OnChainDataProvider()
        data = provider.fetch_many(["BTC-USDT-SWAP", "ETH-USDT-SWAP", "SOL-USDT-SWAP"])
        assert data["BTC-USDT-SWAP"]["whale_flow"] == 0.012
        assert data["ETH-USDT-SWAP"]["exchange_netflow"] == -900.0
        assert "SOL-USDT-SWAP" not in data

        symbols = [
            ScannerSymbol(inst_id="BTC-USDT-SWAP"),
            ScannerSymbol(inst_id="ETH-USDT-SWAP"),
            ScannerSymbol(inst_id="SOL-USDT-SWAP"),
        ]
        engine = ScanEngine(DummyOKX())
        engine.enrich_on_chain_metrics(symbols)

        assert symbols[0].extra_data["on_chain"]["active_addresses"] == 850000.0
        assert symbols[1].extra_data["on_chain"]["nvt_ratio"] == 36.0
        assert "on_chain" not in symbols[2].extra_data

    os.environ["ON_CHAIN_DATA_SOURCE"] = "nansen"
    os.environ["NANSEN_API_KEY"] = "test-key"
    provider = OnChainDataProvider()
    captured = {}
    def fake_post(url, body, auth_style="auto"):
        captured["body"] = body
        return {
            "data": [
                {"symbol": "BTC", "buy_volume": 300.0, "sell_volume": 100.0, "unique_buyers": 42},
                {"token": {"symbol": "ETH"}, "buy_volume": 50.0, "sell_volume": 150.0},
            ]
        }
    provider._post_json = fake_post
    nansen_data = provider.fetch_many(["BTC-USDT-SWAP", "ETH-USDT-SWAP"])
    assert captured["body"]["filters"]["only_smart_money"] is True
    assert round(nansen_data["BTC-USDT-SWAP"]["whale_flow"], 4) == 0.5
    assert round(nansen_data["ETH-USDT-SWAP"]["whale_flow"], 4) == -0.5

    os.environ["ON_CHAIN_DATA_SOURCE"] = "cryptoquant"
    os.environ["CRYPTOQUANT_API_KEY"] = "test-key"
    os.environ["CRYPTOQUANT_DATA_URL"] = "https://example.invalid/{asset}"
    provider = OnChainDataProvider()
    calls = []
    def fake_request(url, auth_style="auto"):
        calls.append((url, auth_style))
        asset = url.rsplit("/", 1)[-1]
        return {"asset": asset, "whale_flow": 0.01, "exchange_netflow": -10, "active_addresses": 100, "nvt_ratio": 20}
    provider._request_json = fake_request
    cq_data = provider.fetch_many(["BTC-USDT-SWAP"])
    assert calls[0][1] == "bearer"
    assert cq_data["BTC-USDT-SWAP"]["exchange_netflow"] == -10.0

    os.environ.pop("CRYPTOQUANT_DATA_URL", None)
    provider = OnChainDataProvider()
    provider._request_json = lambda url, auth_style="auto": {
        "data": [
            {"path": "/v1/btc/market-data/price-ohlcv"},
            {"endpoint": "/v1/btc/exchange-flows/netflow"},
        ]
    }
    endpoints = provider.discover_cryptoquant_endpoints()
    assert "/v1/btc/market-data/price-ohlcv" in endpoints
    assert "/v1/btc/exchange-flows/netflow" in endpoints

    print("链上数据注入验证通过")


if __name__ == "__main__":
    main()
