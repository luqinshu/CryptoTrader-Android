"""
三周期共振趋势扫描策略 (V3.5 强化实战版)
逻辑：1D 趋势斜率 + 1H MACD 挤压变盘 + 3m VWAP 确认

作者：Crypto Trader
版本：3.5
"""

import pandas as pd
import numpy as np
import ta
from typing import Dict, List, Tuple
from src.scanner.base_scanner import (
    BaseScannerStrategy, ScannerSymbol, ScanCondition
)


class TripleTimeframeTrendScanner(BaseScannerStrategy):
    """
    三周期共振趋势扫描器 V3.5
    
    核心升级：
    1. 趋势斜率检测：确保大级别不是横盘。
    2. MACD 变盘检测：在小时线挤压期寻找动能反转。
    3. VWAP 价格共振：确保 3m 级别的突破具有机构资金支撑。
    """

    def _init_conditions(self):
        """初始化扫描条件，防止加载报错"""
        self.add_condition(ScanCondition(
            name="基础流动性",
            description="24h 成交量需大于 1M USDT",
            field="volume_24h",
            operator=">",
            value=self.config.get('min_volume_24h', 1000000)
        ))

    def scan_symbol(self, symbol: ScannerSymbol) -> Dict:
        """扫描单个交易对"""
        result = {
            'symbol': symbol.inst_id,
            'passed': False,
            'score': 0.0,
            'direction': 'NEUTRAL',
            'signals': [],
            'last_price': symbol.last_price,
            'volume_24h': symbol.volume_24h,
            'price_change_24h': symbol.price_change_24h,
        }

        try:
            klines = symbol.extra_data.get('klines', {})
            df_d = self._to_df(klines.get('1D', []))
            df_h = self._to_df(klines.get('1H', []))
            df_3m = self._to_df(klines.get('3m', []))

            if df_d.empty or df_h.empty or df_3m.empty:
                return result

            # 1. 1D 趋势斜率分析 (40分)
            d_score, d_dir, d_sigs = self._analyze_daily_slope(df_d)
            if d_score < 20: return result
            
            result['direction'] = d_dir
            result['score'] += d_score
            result['signals'].extend(d_sigs)

            # 2. 1H MACD 变盘 + 挤压分析 (30分)
            h_score, h_sigs = self._analyze_hourly_momentum(df_h, d_dir)
            result['score'] += h_score
            result['signals'].extend(h_sigs)

            # 3. 3m VWAP + StochRSI 确认 (30分)
            m_score, m_sigs = self._analyze_3min_pro(df_3m, d_dir)
            result['score'] += m_score
            result['signals'].extend(m_sigs)

            # 综合判定
            min_score = self.config.get('min_score', 75)
            result['passed'] = result['score'] >= min_score
            result['rating'] = "🔥 完美共振" if result['score'] >= 90 else ("📈 动能启动" if result['score'] >= 75 else "⏸️ 蓄势")

        except Exception as e:
            result['error'] = str(e)

        return result

    def _to_df(self, klines: List) -> pd.DataFrame:
        if not klines or len(klines) < 30: return pd.DataFrame()
        df = pd.DataFrame(klines, columns=['ts', 'o', 'h', 'l', 'c', 'v', 'v2', 'v3', 'f'])
        for col in ['o', 'h', 'l', 'c', 'v']: df[col] = df[col].astype(float)
        return df

    def _analyze_daily_slope(self, df: pd.DataFrame) -> Tuple[float, str, List]:
        """1D 级别：检测均线斜率，确保大环境不是死水一潭"""
        score = 0
        direction = "NEUTRAL"
        sigs = []
        
        ema20 = ta.trend.ema_indicator(df['c'], 20)
        # 计算 EMA20 最近 5 天的斜率 (百分比变化)
        slope = (ema20.iloc[-1] - ema20.iloc[-5]) / ema20.iloc[-5] * 100
        rsi = ta.momentum.RSIIndicator(df['c']).rsi().iloc[-1]
        
        # 多头：均线向上 且 RSI 在 52 以上
        if slope > 0.1 and rsi > 52:
            direction = "LONG"
            score = 40 if slope > 0.3 else 30
            sigs.append(f"1D: 多头加速 (斜率{slope:.2f}%)")
        # 空头：均线向下 且 RSI 在 48 以下
        elif slope < -0.1 and rsi < 48:
            direction = "SHORT"
            score = 40 if slope < -0.3 else 30
            sigs.append(f"1D: 空头加速 (斜率{slope:.2f}%)")
            
        return score, direction, sigs

    def _analyze_hourly_momentum(self, df: pd.DataFrame, direction: str) -> Tuple[float, List]:
        """1H 级别：挤压 + MACD 动能确认"""
        score = 0
        sigs = []
        
        # 波动率挤压
        bb = ta.volatility.BollingerBands(df['c'], window=20)
        width = (bb.bollinger_hband().iloc[-1] - bb.bollinger_lband().iloc[-1]) / bb.bollinger_mavg().iloc[-1] * 100
        
        # MACD 确认
        macd = ta.trend.MACD(df['c'])
        macd_hist = macd.macd_diff()
        
        if width < 4.0:
            score += 15
            sigs.append(f"1H: 波动率低位挤压 ({width:.1f}%)")
            
        # 检查 MACD 柱状图是否顺向且放大
        if direction == "LONG" and macd_hist.iloc[-1] > 0 and macd_hist.iloc[-1] > macd_hist.iloc[-2]:
            score += 15
            sigs.append("1H: MACD 动能增强")
        elif direction == "SHORT" and macd_hist.iloc[-1] < 0 and macd_hist.iloc[-1] < macd_hist.iloc[-2]:
            score += 15
            sigs.append("1H: MACD 动能减弱")
            
        return score, sigs

    def _analyze_3min_pro(self, df: pd.DataFrame, direction: str) -> Tuple[float, List]:
        """3m 级别：StochRSI + VWAP + 回调不破位确认"""
        score = 0
        sigs = []
        
        # 1. 简易 VWAP 计算
        tp = (df['h'] + df['l'] + df['c']) / 3
        vwap = (tp * df['v']).cumsum() / df['v'].cumsum()
        
        stoch_rsi = ta.momentum.StochRSIIndicator(df['c'])
        k = stoch_rsi.stochrsi_k().iloc[-1]
        
        curr_price = df['c'].iloc[-1]
        curr_vwap = vwap.iloc[-1]
        
        if direction == "LONG":
            if curr_price > curr_vwap:
                score += 15
                sigs.append("3m: 站稳 VWAP 机构价线")
            if k < 0.3: # 低位
                score += 15
                sigs.append("3m: StochRSI 超卖回升点")
        else:
            if curr_price < curr_vwap:
                score += 15
                sigs.append("3m: 跌破 VWAP 机构价线")
            if k > 0.7: # 高位
                score += 15
                sigs.append("3m: StochRSI 超买回落点")

        # 2. 核心新增：回调不破位检测 (上涨后的支撑确认)
        if len(df) >= 15:
            recent_l = df['l'].iloc[-15:].values
            recent_h = df['h'].iloc[-15:].values
            
            if direction == "LONG":
                # 寻找最近波段的起点（最低点）
                origin_idx = np.argmin(recent_l)
                origin_price = recent_l[origin_idx]
                # 确保起点之后有明显的拉升 (>0.3%)
                if origin_idx < 14:
                    max_after = np.max(recent_h[origin_idx:])
                    if max_after > origin_price * 1.003:
                        # 检查从最高点回落以来，是否从未跌破起点
                        pullback_low = np.min(recent_l[origin_idx:])
                        if pullback_low >= origin_price:
                            score += 15
                            sigs.append("3m: 上涨回调未破位 (强支撑)")
            else:
                # 做空逻辑：下跌反弹未破原点
                origin_idx = np.argmax(recent_h)
                origin_price = recent_h[origin_idx]
                if origin_idx < 14:
                    min_after = np.min(recent_l[origin_idx:])
                    if min_after < origin_price * 0.997:
                        pullback_high = np.max(recent_h[origin_idx:])
                        if pullback_high <= origin_price:
                            score += 15
                            sigs.append("3m: 下跌反弹未破位 (强压力)")
                
        return score, sigs

    def get_config_schema(self) -> Dict:
        return {
            'min_score': {'type': 'int', 'default': 75, 'label': '最低通过评分'},
            'min_volume_24h': {'type': 'float', 'default': 1000000, 'label': '最小24h成交量'}
        }

# 配置文件
CONFIG_SCHEMA = {
    'min_score': {'type': 'int', 'default': 75, 'label': '最低通过评分'},
    'min_volume_24h': {'type': 'float', 'default': 1000000, 'label': '最小24h成交量'},
}
