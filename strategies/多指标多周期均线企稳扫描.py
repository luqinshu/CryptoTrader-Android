"""
多指标趋势验证扫描策略（专业版）
日线三种经典指标验证上涨趋势 + 小时线 MACD 转正 + 多周期均线聚集确认企稳

作者：Crypto Trader
版本：3.0 专业版
"""

import pandas as pd
import numpy as np
import ta
from typing import Dict, List, Tuple
from src.scanner.base_scanner import BaseScannerStrategy, ScannerSymbol


class MultiIndicatorTrendScanner(BaseScannerStrategy):
    """
    多指标趋势验证扫描器（专业 V2.0 版）
    
    核心理念：在趋势确认的基础上，寻找波动率收缩(VCP)后的低风险爆发点。
    """

    def _init_conditions(self):
        """初始化扫描条件"""
        pass

    def scan_symbol(self, symbol: ScannerSymbol) -> Dict:
        """扫描单个交易对"""
        result = {
            'symbol': symbol.inst_id,
            'passed': False,
            'score': 0.0,
            'signals': [],
            'last_price': symbol.last_price,
            'volume_24h': symbol.volume_24h,
            'price_change_24h': symbol.price_change_24h,
        }

        try:
            klines = symbol.extra_data.get('klines', {})
            df_d = self._to_df(klines.get('1D', []))
            df_h = self._to_df(klines.get('1H', []))
            df_15m = self._to_df(klines.get('15m', []))

            if df_d.empty or df_h.empty:
                return result

            # 1. 大周期趋势背景 (30分) - ADX + MA排列
            bg_score, bg_sigs = self._analyze_background(df_d)
            result['score'] += bg_score
            result['signals'].extend(bg_sigs)

            # 2. 波动率收缩 (VCP) 检测 (30分) - 核心增强
            vcp_score, vcp_sigs = self._analyze_vcp(df_h)
            result['score'] += vcp_score
            result['signals'].extend(vcp_sigs)

            # 3. 成交量枯竭与异动 (20分)
            vol_score, vol_sigs = self._analyze_volume_footprint(df_h)
            result['score'] += vol_score
            result['signals'].extend(vol_sigs)

            # 4. 短期企稳与口袋买点 (20分)
            setup_score, setup_sigs = self._analyze_setup(df_15m)
            result['score'] += setup_score
            result['signals'].extend(setup_sigs)

            # 最终判断
            result['passed'] = result['score'] >= 75
            result['rating'] = "💎 完美基底" if result['score'] >= 85 else ("📈 趋势企稳" if result['score'] >= 65 else "⏸️ 震荡")

        except Exception as e:
            result['error'] = str(e)

        return result

    def _to_df(self, klines: List) -> pd.DataFrame:
        if not klines or len(klines) < 30: return pd.DataFrame()
        df = pd.DataFrame(klines, columns=['ts', 'o', 'h', 'l', 'c', 'v', 'v2', 'v3', 'f'])
        for col in ['o', 'h', 'l', 'c', 'v']: df[col] = df[col].astype(float)
        return df

    def _analyze_background(self, df: pd.DataFrame) -> Tuple[float, List]:
        score = 0
        sigs = []
        # ADX 强度
        adx = ta.trend.ADXIndicator(df['h'], df['l'], df['c']).adx().iloc[-1]
        if adx > 25:
            score += 15
            sigs.append(f"背景: 强趋势 (ADX={adx:.1f})")
        
        # 均线多头
        ma20 = ta.trend.sma_indicator(df['c'], 20).iloc[-1]
        ma50 = ta.trend.sma_indicator(df['c'], 50).iloc[-1]
        if df['c'].iloc[-1] > ma20 > ma50:
            score += 15
            sigs.append("背景: 均线多头排列")
        return score, sigs

    def _analyze_vcp(self, df: pd.DataFrame) -> Tuple[float, List]:
        """波动率收缩检测"""
        score = 0
        sigs = []
        # 计算 ATR 波动率比例
        atr = ta.volatility.AverageTrueRange(df['h'], df['l'], df['c'], window=14).average_true_range()
        atr_ratio = (atr / df['c']) * 100
        
        curr_atr = atr_ratio.iloc[-1]
        prev_atr = atr_ratio.iloc[-10:-1].mean()
        
        if curr_atr < prev_atr * 0.8:
            score += 30
            sigs.append(f"VCP: 波动率显著收缩 (ATR下降)")
        elif curr_atr < prev_atr:
            score += 15
            sigs.append("VCP: 波动率正在收敛")
        return score, sigs

    def _analyze_volume_footprint(self, df: pd.DataFrame) -> Tuple[float, List]:
        """成交量足迹：枯竭 vs 异动"""
        score = 0
        sigs = []
        recent_v = df['v'].iloc[-5:].mean()
        prev_v = df['v'].iloc[-20:-5].mean()
        
        # 成交量枯竭 (筹码锁定)
        if recent_v < prev_v * 0.7:
            score += 10
            sigs.append("成交量: 显著枯竭 (洗盘彻底)")
        
        # 口袋买点潜质 (缩量回调后的首个放量)
        if df['v'].iloc[-1] > df['v'].iloc[-5:].mean() * 1.5 and df['c'].iloc[-1] > df['o'].iloc[-1]:
            score += 10
            sigs.append("成交量: 出现攻击性量能")
        return score, sigs

    def _analyze_setup(self, df: pd.DataFrame) -> Tuple[float, List]:
        """短期企稳信号"""
        score = 0
        sigs = []
        # RSI 处于健康区间
        rsi = ta.momentum.RSIIndicator(df['c']).rsi().iloc[-1]
        if 50 < rsi < 65:
            score += 10
            sigs.append(f"短期: RSI健康({rsi:.0f})")
        
        # 价格站稳均线
        ma10 = ta.trend.sma_indicator(df['c'], 10).iloc[-1]
        if df['c'].iloc[-1] > ma10:
            score += 10
            sigs.append("短期: 价格站稳MA10")
        return score, sigs

    def get_config_schema(self) -> Dict:
        """获取配置模式"""
        return {
            'min_volume_24h': {
                'type': 'float',
                'default': 1000000,
                'label': '最小24h成交量 (USDT)'
            },
            'hourly_ma_threshold': {
                'type': 'float',
                'default': 2.0,
                'label': '小时线均线聚集阈值 (%)'
            },
            'min15_ma_threshold': {
                'type': 'float',
                'default': 1.5,
                'label': '15 分钟均线聚集阈值 (%)'
            },
            'min_daily_score': {
                'type': 'float',
                'default': 0.65,
                'label': '日线最低趋势评分'
            },
        }


# 策略配置模式
CONFIG_SCHEMA = {
    'min_volume_24h': {'type': 'float', 'default': 1000000, 'label': '最小24h成交量 (USDT)'},
    'hourly_ma_threshold': {'type': 'float', 'default': 2.0, 'label': '小时线均线聚集阈值 (%)'},
    'min15_ma_threshold': {'type': 'float', 'default': 1.5, 'label': '15 分钟均线聚集阈值 (%)'},
    'min_daily_score': {'type': 'float', 'default': 0.65, 'label': '日线最低趋势评分'},
}
