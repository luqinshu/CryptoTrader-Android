"""
OKX 小时线波段共振扫描策略 (V4.0 Elite 顶级版)
核心逻辑：
1. 1D: EMA 20/50/200 结构化趋势 + ADX 趋势质量校验
2. 1H: Squeeze Pro (BB < Keltner) + 斐波那契黄金分割 + 隐藏背离确认
3. 3m: 突破回调 + 机构吸收特征 (Lower Wicks) + 成交量枯竭 (Dry-up)

作者：Crypto Trader
版本：4.0
"""

import pandas as pd
import numpy as np
import ta
from typing import Dict, List, Tuple
from datetime import datetime
from src.scanner.base_scanner import (
    BaseScannerStrategy, ScannerSymbol, ScanCondition
)


class OKXHourSwingScanner(BaseScannerStrategy):
    """
    针对 OKX 设计的顶级波段扫描器 - 寻找机构级共振点
    """

    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.min_adx = self.config.get('min_adx', 25)
        self.default_stop_loss = self.config.get('default_stop_loss', 0.03)
        self.default_take_profit = self.config.get('default_take_profit', 0.10)

    def scan_all_symbols(self, symbols_data: List[ScannerSymbol]) -> Dict:
        """全市场扫描"""
        config = {
            'min_volume_24h': self.config.get('min_volume_24h', 10000000),
            'min_score': self.config.get('min_score', 80),
            'top_n': self.config.get('top_n', 30),
        }
        filtered_symbols = [s for s in symbols_data if s.volume_24h >= config['min_volume_24h']]
        results = []
        for symbol in filtered_symbols:
            try:
                result = self.scan_symbol(symbol)
                if result.get('passed', False):
                    results.append(result)
            except Exception:
                continue
        results.sort(key=lambda x: x['score'], reverse=True)
        return {
            'type': 'hour_swing_v4',
            'scan_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_opportunities': len(results),
            'all_opportunities': results[:config['top_n']],
            'note': 'V4.0 Elite - 多周期共振 + ATR动态风控',
        }

    def _init_conditions(self):
        self.add_condition(ScanCondition(
            name="顶级活跃度",
            description="24h 成交量 > 10M USDT，锁定高流动性标的",
            field="volume_24h",
            operator=">",
            value=self.config.get('min_volume_24h', 10000000)
        ))

    def scan_symbol(self, symbol: ScannerSymbol) -> Dict:
        result = {
            'symbol': symbol.inst_id,
            'passed': False,
            'score': 0.0,
            'direction': 'NEUTRAL',
            'signals': [],
            'last_price': symbol.last_price,
            'volume_24h': symbol.volume_24h,
            'risk_management': {}
        }

        try:
            klines = symbol.extra_data.get('klines', {})
            df_d = self._to_df(klines.get('1D', []), min_len=200)
            df_h = self._to_df(klines.get('1H', []), min_len=100)
            df_3m = self._to_df(klines.get('3m', []), min_len=30)

            if df_d.empty or df_h.empty or df_3m.empty:
                return result

            # 1. 1D 级别：趋势成色校验 (30分)
            d_score, d_dir, d_sigs = self._check_1d_elite(df_d)
            if d_dir == "NEUTRAL": return result
            result['direction'] = d_dir
            result['score'] += d_score
            result['signals'].extend(d_sigs)

            # 2. 1H 级别：Squeeze Pro + 隐藏背离 (40分)
            h_score, h_sigs = self._check_1h_elite(df_h, d_dir)
            if h_score < 20: return result 
            result['score'] += h_score
            result['signals'].extend(h_sigs)

            # 3. 3m 级别：吸收特征 + 缩量确认 (30分)
            m_score, m_sigs = self._check_3m_elite(df_3m, d_dir)
            result['score'] += m_score
            result['signals'].extend(m_sigs)

            # 最终判断
            min_score = self.config.get('min_score', 80)
            result['passed'] = result['score'] >= min_score
            result['rating'] = "💎 顶级机构共振" if result['score'] >= 90 else ("📈 结构企稳" if result['score'] >= 80 else "⏸️ 蓄势")

            # 动态风险管理 (基于 ATR)
            curr_price = df_3m['c'].iloc[-1]
            atr_h = ta.volatility.AverageTrueRange(df_h['h'], df_h['l'], df_h['c'], window=14).average_true_range().iloc[-1]
            result['risk_management'] = self._calculate_dynamic_risk(curr_price, d_dir, atr_h)

        except Exception as e:
            result['error'] = str(e)

        return result

    def _check_1d_elite(self, df: pd.DataFrame) -> Tuple[float, str, List]:
        """1D: 均线系统 + ADX 趋势强度"""
        ema20 = ta.trend.ema_indicator(df['c'], 20).iloc[-1]
        ema50 = ta.trend.ema_indicator(df['c'], 50).iloc[-1]
        ema200 = ta.trend.ema_indicator(df['c'], 200).iloc[-1]
        
        adx_obj = ta.trend.ADXIndicator(df['h'], df['l'], df['c'], window=14)
        adx = adx_obj.adx().iloc[-1]
        pdi = adx_obj.adx_pos().iloc[-1]
        mdi = adx_obj.adx_neg().iloc[-1]
        
        if ema20 > ema50 > ema200 and adx > self.min_adx and pdi > mdi:
            return 30, "LONG", [f"1D: 强势多头结构 (ADX={adx:.1f})"]
        elif ema20 < ema50 < ema200 and adx > self.min_adx and mdi > pdi:
            return 30, "SHORT", [f"1D: 强势空头结构 (ADX={adx:.1f})"]
        return 0, "NEUTRAL", []

    def _check_1h_elite(self, df: pd.DataFrame, direction: str) -> Tuple[float, List]:
        """1H: Squeeze Pro + 斐波那契 + 隐藏背离"""
        score = 0
        sigs = []
        
        # Squeeze Pro
        bb = ta.volatility.BollingerBands(df['c'], window=20)
        kc_h = ta.volatility.keltner_channel_hband(df['h'], df['l'], df['c'], window=20)
        kc_l = ta.volatility.keltner_channel_lband(df['h'], df['l'], df['c'], window=20)
        if bb.bollinger_hband().iloc[-1] < kc_h.iloc[-1] and bb.bollinger_lband().iloc[-1] > kc_l.iloc[-1]:
            score += 20
            sigs.append("1H: Squeeze Pro 能量挤压")

        # 斐波那契 0.5-0.618 区间
        recent_h = df['h'].iloc[-30:].max()
        recent_l = df['l'].iloc[-30:].min()
        curr_c = df['c'].iloc[-1]
        price_range = recent_h - recent_l
        
        # 隐藏背离检测 (RSI)
        rsi = ta.momentum.RSIIndicator(df['c'], window=14).rsi()
        if direction == "LONG":
            fib_618 = recent_h - price_range * 0.65
            fib_382 = recent_h - price_range * 0.35
            if fib_618 <= curr_c <= fib_382:
                score += 10
                sigs.append("1H: 黄金位回踩企稳")
            # 隐藏看涨背离：价格低点抬高，但RSI低点降低
            if df['l'].iloc[-1] > df['l'].iloc[-20:-5].min() and rsi.iloc[-1] < rsi.iloc[-20:-5].min():
                score += 10
                sigs.append("1H: 发现隐藏看涨背离")
        else:
            if recent_l + price_range * 0.35 <= curr_c <= recent_l + price_range * 0.65:
                score += 10
                sigs.append("1H: 斐波那契反弹遇阻")
            if df['h'].iloc[-1] < df['h'].iloc[-20:-5].max() and rsi.iloc[-1] > rsi.iloc[-20:-5].max():
                score += 10
                sigs.append("1H: 发现隐藏看跌背离")
                
        return score, sigs

    def _check_3m_elite(self, df: pd.DataFrame, direction: str) -> Tuple[float, List]:
        """3m: 突破确认 + 影线吸收 + 缩量回调"""
        score = 0
        sigs = []
        
        if len(df) < 20:
            return 0, []
        
        ema20 = ta.trend.ema_indicator(df['c'], 20)
        v_ema = df['v'].rolling(10).mean()
        
        if direction == "LONG":
            if df['c'].iloc[-1] > ema20.iloc[-1] and df['v'].iloc[-1] > v_ema.iloc[-1] * 1.5:
                score += 10
                sigs.append("3m: 动能放量突破")
            
            recent = df.iloc[-5:].copy()
            lower_wicks = (np.minimum(recent['c'], recent['o']) - recent['l']).sum()
            bodies = abs(recent['c'] - recent['o']).sum()
            if bodies > 0 and lower_wicks > bodies * 0.8:
                score += 10
                sigs.append("3m: 底部机构吸收迹象")
            
            avg_vol_recent = df['v'].iloc[-10:].mean()
            avg_vol_earlier = df['v'].iloc[-20:-10].mean()
            if avg_vol_recent < avg_vol_earlier * 0.7:
                score += 10
                sigs.append("3m: 缩量蓄势确认")
        else:
            if df['c'].iloc[-1] < ema20.iloc[-1] and df['v'].iloc[-1] > v_ema.iloc[-1] * 1.5:
                score += 10
                sigs.append("3m: 动能下破确认")
            
            recent = df.iloc[-5:].copy()
            upper_wicks = (recent['h'] - np.maximum(recent['c'], recent['o'])).sum()
            bodies = abs(recent['c'] - recent['o']).sum()
            if bodies > 0 and upper_wicks > bodies * 0.8:
                score += 10
                sigs.append("3m: 顶部机构派发迹象")
            
            avg_vol_recent = df['v'].iloc[-10:].mean()
            avg_vol_earlier = df['v'].iloc[-20:-10].mean()
            if avg_vol_recent < avg_vol_earlier * 0.7:
                score += 10
                sigs.append("3m: 缩量整理确认")
                
        return score, sigs

    def _calculate_dynamic_risk(self, price: float, direction: str, atr: float) -> Dict:
        # 基于 ATR 的 1.5 倍作为动态止损
        sl_dist = atr * 1.5
        if direction == "LONG":
            sl = price - sl_dist
            tp = price + (sl_dist * 2.5) # 盈亏比 1:2.5
        else:
            sl = price + sl_dist
            tp = price - (sl_dist * 2.5)
            
        return {
            'stop_loss': round(sl, 8),
            'take_profit': round(tp, 8),
            'sl_distance_pct': round((sl_dist/price)*100, 2),
            'rr_ratio': 2.5
        }

    def _to_df(self, klines: List, min_len: int = 50) -> pd.DataFrame:
        if not klines or len(klines) < min_len: return pd.DataFrame()
        df = pd.DataFrame(klines, columns=['ts', 'o', 'h', 'l', 'c', 'v', 'v2', 'v3', 'f'])
        for col in ['o', 'h', 'l', 'c', 'v']: df[col] = df[col].astype(float)
        return df

    def get_config_schema(self) -> Dict:
        return {
            'min_score': {'type': 'int', 'default': 80, 'label': '最低通过评分'},
            'min_volume_24h': {'type': 'float', 'default': 10000000, 'label': '最小24h成交量'},
            'min_adx': {'type': 'int', 'default': 25, 'label': '最小1D趋势强度(ADX)'},
            'default_stop_loss': {'type': 'float', 'default': 0.03, 'label': '默认止损比例'},
            'default_take_profit': {'type': 'float', 'default': 0.10, 'label': '默认止盈比例'},
            'top_n': {'type': 'int', 'default': 30, 'label': '显示结果数'},
        }

CONFIG_SCHEMA = {
    'min_score': {'type': 'int', 'default': 80, 'label': '最低通过评分'},
    'min_volume_24h': {'type': 'float', 'default': 10000000, 'label': '最小24h成交量'},
    'min_adx': {'type': 'int', 'default': 25, 'label': '最小1D趋势强度(ADX)'},
    'default_stop_loss': {'type': 'float', 'default': 0.03, 'label': '默认止损比例'},
    'default_take_profit': {'type': 'float', 'default': 0.10, 'label': '默认止盈比例'},
    'top_n': {'type': 'int', 'default': 30, 'label': '显示结果数'},
}
