"""
单边大趋势捕捉扫描策略 - 早期发现 V5.0 (Institutional Edge 版)
专注于识别机构吸筹结束、多周期共振爆发的第一个波段

作者：Crypto Trader
版本：5.0 - Early Bird Ultra (改进版)
"""

import pandas as pd
import numpy as np
import ta
from src.scanner.base_scanner import (
    BaseScannerStrategy, ScannerSymbol, ScanCondition
)
from typing import Dict, List, Tuple
from datetime import datetime


class UnilateralTrendScannerEarly(BaseScannerStrategy):
    """
    单边大趋势扫描器 - 早期发现版 V5.0
    
    核心升级：
    1. 趋势共振：5m 启动必须得到 15m EMA 走势的初步认可。
    2. 成本线锚定：引入 VWAP，确保价格正在夺回/跌破机构平均持有成本。
    3. 动能过滤：使用 ADX 确保趋势具有持续潜力，非脉冲式噪音。
    4. 能量爆发：结合布林带带宽标准差，识别真正的波动率收缩。
    5. 风控完善：内置止损止盈建议，降低回撤风险。
    """

    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.default_stop_loss = self.config.get('default_stop_loss', 0.02)
        self.default_take_profit = self.config.get('default_take_profit', 0.06)
        self.min_holding_hours = self.config.get('min_holding_hours', 2)
        self.max_holding_hours = self.config.get('max_holding_hours', 24)

    def _init_conditions(self):
        """初始化扫描条件"""
        self.add_condition(ScanCondition(
            name="基础流动性",
            description="24h 成交量需 > 1M USDT",
            field="volume_24h",
            operator=">",
            value=self.config.get('min_volume_24h', 1000000)
        ))

    def scan_all_symbols(self, symbols_data: List[ScannerSymbol]) -> Dict:
        """全市场扫描"""
        config = {
            'min_volume_24h': self.config.get('min_volume_24h', 1000000),
            'min_score': self.config.get('min_score', 60),
            'top_n': self.config.get('top_n', 30),
        }

        filtered_symbols = [s for s in symbols_data if s.volume_24h >= config['min_volume_24h']]
        results = []

        for symbol in filtered_symbols:
            try:
                long_a = self._analyze_early_ultra(symbol, "LONG")
                short_a = self._analyze_early_ultra(symbol, "SHORT")
                
                analysis = long_a if long_a['score'] >= short_a['score'] else short_a
                if analysis['score'] >= config['min_score']:
                    results.append(analysis)
            except Exception:
                continue

        results.sort(key=lambda x: x['score'], reverse=True)

        return {
            'type': 'early_bird_ultra_v5',
            'scan_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_opportunities': len(results),
            'all_opportunities': results[:config['top_n']],
            'note': 'V5.0 - 整合VWAP成本线、ADX强度、15m趋势共振及智能风控',
        }

    def _analyze_early_ultra(self, symbol: ScannerSymbol, direction: str) -> Dict:
        """极速早期分析逻辑 V5.0"""
        score = 0
        signals = []
        
        klines_5m = self._to_df(symbol.extra_data.get('klines', {}).get('5m', []))
        klines_15m = self._to_df(symbol.extra_data.get('klines', {}).get('15m', []))
        
        if klines_5m.empty or len(klines_5m) < 30:
            return {'score': 0, 'direction': direction}

        if klines_15m.empty or len(klines_15m) < 30:
            return {'score': 0, 'direction': direction}

        tp = (klines_5m['h'] + klines_5m['l'] + klines_5m['c']) / 3
        vwap = (tp * klines_5m['v']).cumsum() / klines_5m['v'].cumsum()
        curr_price = klines_5m['c'].iloc[-1]
        curr_vwap = vwap.iloc[-1]
        
        if (direction == "LONG" and curr_price > curr_vwap) or \
           (direction == "SHORT" and curr_price < curr_vwap):
            score += 25
            signals.append("🏛️ 价格站上机构成本线 (VWAP) (+25)")

        ema15_9 = ta.trend.ema_indicator(klines_15m['c'], window=9)
        ema15_21 = ta.trend.ema_indicator(klines_15m['c'], window=21)
        if direction == "LONG":
            if klines_15m['c'].iloc[-1] > ema15_21.iloc[-1] and ema15_9.iloc[-1] > ema15_9.iloc[-2]:
                score += 20
                signals.append("📈 15m 趋势初步顺向确认 (+20)")
        else:
            if klines_15m['c'].iloc[-1] < ema15_21.iloc[-1] and ema15_9.iloc[-1] < ema15_9.iloc[-2]:
                score += 20
                signals.append("📉 15m 趋势初步顺向确认 (+20)")

        adx_obj = ta.trend.ADXIndicator(klines_5m['h'], klines_5m['l'], klines_5m['c'], window=14)
        adx = adx_obj.adx().iloc[-1]
        if adx > 25 and adx > adx_obj.adx().iloc[-2]:
            score += 20
            signals.append(f"⚡ 趋势动能正在增强 (ADX={adx:.1f}) (+20)")

        rsi = ta.momentum.RSIIndicator(klines_5m['c'], window=14).rsi()
        curr_rsi = rsi.iloc[-1]
        if direction == "LONG" and rsi.iloc[-10:-1].min() < 40 and curr_rsi > 50:
            score += 20
            signals.append(f"🔄 RSI 超卖强势回升 ({curr_rsi:.1f}) (+20)")
        elif direction == "SHORT" and rsi.iloc[-10:-1].max() > 60 and curr_rsi < 50:
            score += 20
            signals.append(f"🔄 RSI 超买强势回落 ({curr_rsi:.1f}) (+20)")

        bb = ta.volatility.BollingerBands(klines_5m['c'], window=20)
        bw = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg() * 100
        if bw.iloc[-1] < bw.rolling(100).mean().iloc[-1] * 0.75:
            score += 15
            signals.append(f"💥 波动率极限收缩 ({bw.iloc[-1]:.1f}%) (+15)")

        stop_loss = self._calculate_stop_loss(curr_price, direction)
        take_profit = self._calculate_take_profit(curr_price, direction)
        
        return {
            'symbol': symbol.inst_id,
            'direction': direction,
            'score': score,
            'rating': "🚀 黄金启动点" if score >= 85 else ("🔥 早期强势" if score >= 70 else "👀 潜力观察"),
            'signals': signals,
            'last_price': curr_price,
            'volume_24h': symbol.volume_24h,
            'price_change_24h': symbol.price_change_24h,
            'risk_management': {
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'stop_loss_pct': self.default_stop_loss,
                'take_profit_pct': self.default_take_profit,
                'min_holding_hours': self.min_holding_hours,
                'max_holding_hours': self.max_holding_hours,
            }
        }

    def _calculate_stop_loss(self, price: float, direction: str) -> float:
        if direction == "LONG":
            return price * (1 - self.default_stop_loss)
        else:
            return price * (1 + self.default_stop_loss)

    def _calculate_take_profit(self, price: float, direction: str) -> float:
        if direction == "LONG":
            return price * (1 + self.default_take_profit)
        else:
            return price * (1 - self.default_take_profit)

    def _to_df(self, klines: List) -> pd.DataFrame:
        if not klines or len(klines) < 20: return pd.DataFrame()
        df = pd.DataFrame(klines, columns=['ts', 'o', 'h', 'l', 'c', 'v', 'v2', 'v3', 'f'])
        for col in ['o', 'h', 'l', 'c', 'v']: df[col] = df[col].astype(float)
        return df

    def scan_symbol(self, symbol: ScannerSymbol) -> Dict:
        """兼容扫描接口"""
        long_a = self._analyze_early_ultra(symbol, "LONG")
        short_a = self._analyze_early_ultra(symbol, "SHORT")
        analysis = long_a if long_a['score'] >= short_a['score'] else short_a
        
        return {
            'symbol': symbol.inst_id,
            'passed': analysis['score'] >= self.config.get('min_score', 60),
            'score': analysis['score'],
            'direction': analysis['direction'],
            'rating': analysis['rating'],
            'signals': analysis['signals'],
            'last_price': symbol.last_price,
            'volume_24h': symbol.volume_24h,
            'price_change_24h': symbol.price_change_24h,
            'risk_management': analysis.get('risk_management', {}),
        }

    def get_config_schema(self) -> Dict:
        return {
            'min_volume_24h': {'type': 'float', 'default': 1000000, 'label': '最小24h成交量'},
            'min_score': {'type': 'int', 'default': 60, 'label': '最低通过分数'},
            'top_n': {'type': 'int', 'default': 30, 'label': '显示结果数'},
            'default_stop_loss': {'type': 'float', 'default': 0.02, 'label': '默认止损比例'},
            'default_take_profit': {'type': 'float', 'default': 0.06, 'label': '默认止盈比例'},
            'min_holding_hours': {'type': 'int', 'default': 2, 'label': '最小持仓时长(小时)'},
            'max_holding_hours': {'type': 'int', 'default': 24, 'label': '最大持仓时长(小时)'},
        }

CONFIG_SCHEMA = {
    'min_volume_24h': {'type': 'float', 'default': 1000000, 'label': '最小24h成交量'},
    'min_score': {'type': 'int', 'default': 60, 'label': '最低通过分数'},
    'top_n': {'type': 'int', 'default': 30, 'label': '显示结果数'},
    'default_stop_loss': {'type': 'float', 'default': 0.02, 'label': '默认止损比例'},
    'default_take_profit': {'type': 'float', 'default': 0.06, 'label': '默认止盈比例'},
    'min_holding_hours': {'type': 'int', 'default': 2, 'label': '最小持仓时长(小时)'},
    'max_holding_hours': {'type': 'int', 'default': 24, 'label': '最大持仓时长(小时)'},
}
