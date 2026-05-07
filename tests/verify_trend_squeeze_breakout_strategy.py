"""
本地验证“趋势延续·挤压突破前扫描器”的关键路径。

这个脚本只使用合成 K 线，不请求外网：
1. 验证新参数与策略类能正常加载
2. 验证 K 线清洗会排序、去重、过滤坏值
3. 验证默认参数下能拦截“日线过度延伸”
4. 验证宽松参数下主路径能输出 BUY，并且 ranking_factors 为 0-100 尺度
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

STRATEGY_PATH = PROJECT_ROOT / "strategies" / "趋势延续挤压突破前扫描器4.28_v1.py"

from src.scanner.base_scanner import ScannerSymbol


def _load_module():
    spec = importlib.util.spec_from_file_location("trend_squeeze_breakout_verify", STRATEGY_PATH)
    assert spec and spec.loader, "策略模块加载失败"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_daily(overextended: bool = False):
    rows = []
    ts = 1700000000000
    price = 100.0
    for i in range(170):
        step = 0.34 + (i % 6) * 0.02
        if i > 150:
            step = (0.18 + (i % 3) * 0.015) if not overextended else (0.95 + (i % 2) * 0.08)
        open_price = price - step * 0.45
        close = price + step
        high = close + step * 0.82
        low = open_price - step * 0.62
        volume = 1_000_000 + i * 1200
        rows.append([ts + i * 86400000, open_price, high, low, close, volume])
        price = close
    return rows


def _make_hourly_candidate():
    rows = []
    ts = 1700000000000
    price = 150.0
    for i in range(100):
        if i < 55:
            step = 0.20 + (i % 4) * 0.01
            open_price = price
            close = price + step
            high = close + 0.18
            low = open_price - 0.14
            volume = 98000 - i * 320
        elif i < 71:
            step = 0.34 + (i % 3) * 0.01
            open_price = price
            close = price + step
            high = close + 0.14
            low = open_price - 0.12
            volume = 71000 - (i - 55) * 650
        elif i < 87:
            step = -0.40 + (i % 3) * 0.02
            open_price = price
            close = price + step
            high = max(open_price, close) + 0.10
            low = min(open_price, close) - 0.15
            volume = 50000 - (i - 71) * 480
        else:
            base = 159.6 + (i - 87) * 0.018
            open_price = base - 0.014
            close = base + (0.006 if i % 2 == 0 else -0.003)
            high = max(open_price, close) + 0.012
            low = min(open_price, close) - 0.011
            volume = 25000 - (i - 87) * 140
            if i == 99:
                volume = 24500
        rows.append([ts + i * 3600000, open_price, high, low, close, max(volume, 15000)])
        price = close
    return rows


def _make_symbol(*, overextended: bool = False):
    h1_rows = _make_hourly_candidate()
    return ScannerSymbol(
        inst_id="TEST-USDT-SWAP",
        last_price=float(h1_rows[-1][4]),
        volume_24h=30_000_000,
        price_change_24h=2.3,
        extra_data={
            "klines": {
                "1D": _make_daily(overextended=overextended),
                "1H": h1_rows,
            }
        },
    )


def main():
    module = _load_module()
    strategy_cls = module.TrendSqueezeBreakoutScanner

    default_scanner = strategy_cls()
    schema = default_scanner.get_config_schema()
    assert "d1_max_extension_atr" in schema
    assert "h1_rsi_bull_min" in schema
    assert "h1_max_dryup_ratio" in schema

    dirty_symbol = ScannerSymbol(
        inst_id="DIRTY-USDT-SWAP",
        extra_data={
            "klines": {
                "1H": [
                    [3, 1, 1.2, 0.9, 1.1, 10],
                    [2, 1, 1.1, 0.8, 1.0, 9],
                    [2, 1.5, 1.6, 1.4, 1.55, 11],  # duplicate ts, should keep latest
                    [4, -1, 1.0, 0.7, 0.8, 8],     # bad row, should be filtered
                ]
            }
        },
    )
    cleaned = strategy_cls._get_klines(dirty_symbol, "1H")
    assert [row[0] for row in cleaned] == [2, 3]
    assert cleaned[0][4] == 1.55

    overextended_result = default_scanner.scan_symbol(_make_symbol(overextended=True))
    assert overextended_result["passed"] is False
    assert "过度延伸" in str(overextended_result.get("details", {}).get("淘汰原因", ""))

    permissive_scanner = strategy_cls({
        "h1_pullback_min_pct": 2.8,
        "h1_bb_squeeze_pct": 3.7,
        "h1_rsi_bull_min": 22.0,
        "h1_max_dryup_ratio": 1.0,
        "min_score": 55.0,
    })
    candidate_result = permissive_scanner.scan_symbol(_make_symbol(overextended=False))
    assert candidate_result["passed"] is True, candidate_result
    assert candidate_result["direction"] == "BUY", candidate_result
    assert float(candidate_result["score"]) >= 55.0
    assert float(candidate_result["opportunity_score"]) >= float(candidate_result["score"])

    ranking = candidate_result.get("ranking_factors", {})
    for key in ("trend", "trigger", "volume", "location", "freshness", "risk"):
        assert key in ranking, ranking
        assert 0.0 <= float(ranking[key]) <= 100.0, (key, ranking)
    assert float(ranking["trend"]) > 1.0
    assert float(ranking["volume"]) > 1.0

    print("趋势延续·挤压突破前扫描器验证通过")
    print(f"默认参数过度延伸拦截: {overextended_result['details'].get('淘汰原因')}")
    print(
        "宽松参数正例输出: "
        f"direction={candidate_result['direction']}, "
        f"score={float(candidate_result['score']):.2f}, "
        f"opportunity={float(candidate_result['opportunity_score']):.2f}"
    )
    print(
        "统一评分尺度: "
        f"trend={float(ranking['trend']):.2f}, "
        f"trigger={float(ranking['trigger']):.2f}, "
        f"volume={float(ranking['volume']):.2f}"
    )


if __name__ == "__main__":
    main()
