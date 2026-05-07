"""
BTC/ETH 牵引联动策略

判断目标币相对 BTC/ETH 的领涨、抗跌、滞涨和风险跟跌状态。
"""

from statistics import median
from typing import Dict, List

from src.scanner.base_scanner import BaseScannerStrategy, ScanCondition, ScannerSymbol
from src.scanner.ranking import build_opportunity_profile, sort_scan_results


class BTCETHLeadLagScanner(BaseScannerStrategy):
    required_bars = ['4H', '1H']

    def _init_conditions(self):
        self.add_condition(ScanCondition("24H成交量", "过滤低流动性", "volume_24h", ">=", self.config.get('min_volume_24h', 12000000)))

    def scan_all_symbols(self, symbols: List[ScannerSymbol]) -> Dict:
        btc = next((s for s in symbols if s.inst_id == 'BTC-USDT-SWAP'), None)
        eth = next((s for s in symbols if s.inst_id == 'ETH-USDT-SWAP'), None)
        market_anchor = median([s.price_change_24h for s in [btc, eth] if s]) if (btc or eth) else 0.0
        results = []
        for symbol in symbols:
            if symbol.inst_id in {'BTC-USDT-SWAP', 'ETH-USDT-SWAP'}:
                continue
            result = self.scan_symbol(symbol, market_anchor)
            if result.get('passed'):
                results.append(result)
        return {'type': 'btc_eth_lead_lag_scan', 'all_opportunities': sort_scan_results(results)[:int(self.config.get('top_n', 30))]}

    def scan_symbol(self, symbol: ScannerSymbol, market_anchor_change: float = 0.0) -> Dict:
        try:
            relative = float(symbol.price_change_24h or 0.0) - float(market_anchor_change or 0.0)
            score = 0.0
            direction = 'WAIT'
            signals = []

            if market_anchor_change >= 2.0 and relative >= float(self.config.get('min_lead_strength_pct', 2.5)):
                score += 46
                direction = 'BUY'
                signals.append(f"BTC/ETH强，目标更强：领涨({relative:.2f}%)")
            elif market_anchor_change >= 2.0 and relative <= -2.0:
                score += 20
                signals.append(f"BTC/ETH强但目标滞涨({relative:.2f}%)")
            elif market_anchor_change <= -2.0 and relative >= 2.0:
                score += 42
                direction = 'BUY'
                signals.append(f"BTC/ETH弱，目标抗跌({relative:.2f}%)")
            elif market_anchor_change <= -2.0 and relative <= -2.5:
                score += 34
                direction = 'SELL'
                signals.append(f"BTC/ETH跌，目标跟跌更猛({relative:.2f}%)")
            else:
                signals.append(f"联动不明显({relative:.2f}%)")

            if symbol.volume_24h >= float(self.config.get('strong_volume_24h', 30000000)):
                score += 14
                signals.append("成交额活跃")
            if abs(symbol.price_change_24h) >= 3.0:
                score += 10
                signals.append("24H波动足够")

            profile = build_opportunity_profile(
                base_score=score,
                direction=direction,
                volume_24h=symbol.volume_24h,
                factors={
                    'trend': 82 if direction == 'BUY' else 72 if direction == 'SELL' else 45,
                    'trigger': 88 if direction != 'WAIT' else 40,
                    'volume': min(symbol.volume_24h / 50000000, 1.0) * 100,
                    'location': min(abs(relative) * 18, 100),
                    'freshness': 80,
                    'risk': 76 if direction == 'BUY' else 62,
                },
                signals=signals,
            )
            return {
                'symbol': symbol.inst_id,
                'passed': score >= float(self.config.get('min_score', 70)) and direction != 'WAIT',
                'score': round(score, 2),
                'direction': direction,
                'signals': signals,
                'details': {
                    '评估': ' | '.join(signals),
                    'BTC_ETH基准涨幅': f'{market_anchor_change:.2f}%',
                    '目标24H涨幅': f'{symbol.price_change_24h:.2f}%',
                    '相对强弱': f'{relative:.2f}%',
                },
                'last_price': symbol.last_price,
                'volume_24h': symbol.volume_24h,
                'price_change_24h': symbol.price_change_24h,
                'category': 'BTC/ETH牵引',
                **profile,
            }
        except Exception as exc:
            return {'symbol': symbol.inst_id, 'passed': False, 'score': 0.0, 'direction': 'WAIT', 'details': {'状态': f'分析异常: {exc}'}}

    def get_config_schema(self) -> Dict:
        return {
            'min_score': {'type': 'int', 'default': 70, 'label': '最低通过分数'},
            'min_volume_24h': {'type': 'float', 'default': 12000000, 'label': '最小24H成交额'},
            'min_lead_strength_pct': {'type': 'float', 'default': 2.5, 'label': '最小相对领涨幅度%'},
            'top_n': {'type': 'int', 'default': 30, 'label': '保留数量'},
        }


STRATEGY_NAME  = "BTC_ETH牵引联动策略"
STRATEGY_TYPE  = "scan"
STRATEGY_CLASS = BTCETHLeadLagScanner
BACKTEST_CLASS = BTCETHLeadLagScanner
