"""
本地验证“波段八策略组合扫描器”接入自动交易的关键安全路径。

这个脚本只使用 mock 行情和 mock 执行器：
- 不连接 OKX
- 不读取真实账户
- 不发送真实订单
"""

import time
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.strategy.loader import StrategyLoader
from src.ui.quant_trade_window import AutoTraderWorker


class FakeOKX:
    def get_kline(self, inst_id, bar="1H", limit=200):
        rows = []
        base = 100.0
        step = {"1D": 0.45, "4H": 0.22, "1H": 0.10, "15m": 0.03, "5m": 0.01}.get(bar, 0.1)
        now = int(time.time() * 1000)
        for i in range(limit):
            idx = limit - i
            close = base + idx * step
            open_price = close - step * 0.45
            high = close + step * 1.8
            low = open_price - step * 1.4
            volume = 100000 + idx * 100
            rows.append([
                str(now - i * 60000),
                str(open_price),
                str(high),
                str(low),
                str(close),
                str(volume),
                str(volume * close),
            ])
        return {"code": "0", "data": rows}

    def get_ticker(self, inst_id):
        return {
            "code": "0",
            "data": [{
                "instId": inst_id,
                "last": "120",
                "open24h": "110",
                "high24h": "121",
                "low24h": "108",
                "volCcyQuote": "20000000",
            }],
        }

    def get_tickers(self, instType="SWAP"):
        return {
            "code": "0",
            "data": [{
                "instId": "TEST-USDT-SWAP",
                "last": "120",
                "open24h": "110",
                "high24h": "121",
                "low24h": "108",
                "volCcyQuote": "20000000",
            }],
        }

    def get_order_book(self, inst_id, limit=1):
        return {"code": "0", "data": [{"bids": [["119.99", "10"]], "asks": [["120.01", "10"]]}]}


class FakeExecutor:
    def __init__(self):
        self.entry_calls = []

    def get_positions(self, inst_id=None):
        return {}

    def get_usdt_balance(self):
        return 10000.0

    def estimate_position_notional(self, pos):
        return 0.0

    def execute_entry(self, *args, **kwargs):
        self.entry_calls.append((args, kwargs))
        raise AssertionError("scan_only 模式不允许发送真实订单")


def main():
    project_root = Path(__file__).resolve().parents[1]
    loader = StrategyLoader(str(project_root / "strategies"))
    discovered = loader.discover_strategies()
    strategy_name = "波段八策略组合扫描器"
    assert any(item.name == strategy_name for item in discovered), "策略加载器未发现波段八策略组合扫描器"

    strategy_class = loader.get_strategy_class(strategy_name)
    assert strategy_class is not None, "策略加载器未找到组合扫描器策略类"
    strategy = strategy_class({"min_volume_24h": 1000000, "top_n_per_group": 5})
    assert len(strategy.child_strategies) == 8, f"子策略数量异常: {len(strategy.child_strategies)}"

    config = {
        "inst_id": "TEST-USDT-SWAP",
        "strategy_name": strategy_name,
        "trade_mode": "scan_only",
        "auto_initial_capital": 1000.0,
        "max_total_amount": 1000.0,
        "per_trade_amount": 100.0,
        "enable_risk_based_sizing": True,
        "risk_per_trade_pct": 1.0,
        "sl_percent": 3.0,
        "tp_percent": 5.0,
        "enable_multi_timeframe_score": True,
        "enable_market_filter": False,
        "enable_market_breadth_filter": False,
        "enable_market_environment": False,
        "min_signal_score": 0.0,
        "min_auto_opportunity_level": "C",
        "position_ownership_path": "/tmp/nonexistent_position_ownership_state.json",
        "runtime_state_path": "/tmp/auto_trade_runtime_state_safety_test.json",
        "strategy_health_path": "/tmp/strategy_health_state_safety_test.json",
    }
    worker = AutoTraderWorker(strategy, FakeOKX(), FakeExecutor(), config)

    analysis = worker._analyze_candidate(
        "TEST-USDT-SWAP",
        ticker={
            "instId": "TEST-USDT-SWAP",
            "last": "120",
            "open24h": "110",
            "high24h": "121",
            "low24h": "108",
            "volCcyQuote": "20000000",
        },
    )
    assert "action" in analysis and "score" in analysis and "signals" in analysis, analysis

    amount, sizing = worker._risk_adjusted_trade_amount(
        current_equity=10000.0,
        per_trade_amount=500.0,
        remaining_budget=1000.0,
        sl_pct=0.03,
        size_multiplier=1.0,
    )
    assert amount <= 333.34, (amount, sizing)
    assert sizing.get("risk_base") == 1000.0, sizing

    candidate = {
        "inst_id": "TEST-USDT-SWAP",
        "action": "BUY",
        "score": 99,
        "opportunity_level": "S",
        "signals": ["safety-test"],
    }
    result, confirmed = worker._execute_auto_entry(candidate, 100.0, 3, 0.05, 0.03)
    assert result is None and confirmed is False
    assert worker.trade_executor.entry_calls == []

    print("波段八策略自动交易安全路径验证通过")
    print(f"子策略数量: {len(strategy.child_strategies)}")
    print(f"分析输出: action={analysis['action']}, score={float(analysis['score']):.2f}")
    print(f"自动资金舱风险定额: amount={amount:.2f}, risk_base={sizing.get('risk_base'):.2f}")
    print("只扫描模式已确认不会调用 execute_entry")


if __name__ == "__main__":
    main()
