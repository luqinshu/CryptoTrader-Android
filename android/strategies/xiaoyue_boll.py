"""
小月期货多周期布林趋势转折扫描策略 (Android 简化版)

核心逻辑：
D1布林定势 → 1H回踩中轨 → 15m MACD金叉+放量起爆 | ATR止损+追踪
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime
from src.scanner.base_scanner import (
    BaseScannerStrategy, ScannerSymbol, ScanCondition
)

class XiaoYueBollMacdScanner(BaseScannerStrategy):
    """
    小月期货多周期布林趋势转折扫描器 (Android 版)
    """

    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.default_stop_loss = self.config.get('default_stop_loss', 0.025)
        self.default_take_profit = self.config.get('default_take_profit', 0.075)

    def scan_all_symbols(self, symbols_data: List[ScannerSymbol]) -> Dict:
        """全市场扫描"""
        config = {
            'min_volume_24h': self.config.get('min_volume_24h', 5000000),
            'min_score': self.config.get('min_score', 60),
            'top_n': self.config.get('top_n', 20),
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
            'type': 'xiaoyue_boll',
            'scan_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_opportunities': len(results),
            'all_opportunities': results[:config['top_n']],
            'note': '小月期货 - 多周期布林趋势转折',
        }

    def _init_conditions(self):
        self.add_condition(ScanCondition(
            name="活跃度",
            description="24h 成交量 > 5M USDT",
            field="volume_24h",
            operator=">",
            value=self.config.get('min_volume_24h', 5000000)
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
            d1_raw = klines.get('1D', []) or klines.get('1d', [])
            h1_raw = klines.get('1H', []) or klines.get('1h', [])
            m15_raw = klines.get('15m', []) or klines.get('15M', [])
            
            if len(d1_raw) < 25 or len(h1_raw) < 30 or len(m15_raw) < 35:
                result['signals'].append('数据不足，跳过')
                return result

            d1_df = self._klines_to_df(d1_raw)
            h1_df = self._klines_to_df(h1_raw)
            m15_df = self._klines_to_df(m15_raw)

            if d1_df is None or h1_df is None or m15_df is None:
                return result

            d1_ok, d1_dir, d1_score, atr_pct = self._step1_daily_trend(d1_df)
            if not d1_ok:
                result['signals'].append(f'大周期无趋势')
                return result

            h1_ok, h1_score = self._step2_h1_pullback(h1_df, d1_dir)
            if not h1_ok:
                result['signals'].append(f'1H未回踩')
                return result

            m15_ok, m15_score = self._step3_m15_entry(m15_df, d1_dir)
            if not m15_ok:
                result['signals'].append(f'15m无起爆信号')
                return result

            score = d1_score * 0.35 + h1_score * 0.30 + m15_score * 0.35
            result['score'] = min(100.0, max(0.0, score))

            if result['score'] >= 60:
                result['passed'] = True
                result['direction'] = 'BUY' if d1_dir == 'bull' else 'SELL'
                result['signals'] = [
                    f"小月布林 {'多头' if d1_dir=='bull' else '空头'}",
                    f"评分: {result['score']:.0f}",
                    f"D1趋势 ✓ 1H回踩 ✓ 15m起爆 ✓"
                ]

                self._apply_risk_management(result, atr_pct, symbol.last_price, d1_dir)

        except Exception as e:
            result['signals'].append(f'错误: {str(e)}')

        return result

    def _step1_daily_trend(self, df: pd.DataFrame) -> Tuple[bool, str, float, float]:
        """D1: 布林趋势确认"""
        period = 20
        std_mult = 2.0
        
        df['boll_mid'] = df['c'].rolling(period).mean()
        df['boll_std'] = df['c'].rolling(period).std()
        df['boll_up'] = df['boll_mid'] + std_mult * df['boll_std']
        df['boll_down'] = df['boll_mid'] - std_mult * df['boll_std']
        
        last_close = float(df['c'].iloc[-1])
        last_mid = float(df['boll_mid'].iloc[-1])
        
        tr = pd.concat([
            df['h'] - df['l'],
            (df['h'] - df['c'].shift(1)).abs(),
            (df['l'] - df['c'].shift(1)).abs()
        ], axis=1).max(axis=1)
        atr = tr.rolling(14).mean().iloc[-1]
        atr_pct = float(atr / last_close * 100) if last_close > 0 else 2.0
        
        lookback = 8
        if len(df) > lookback:
            y = df['boll_mid'].iloc[-lookback:].values
            x = np.arange(lookback)
            mid_slope = float(np.polyfit(x, y, 1)[0])
            slope_angle = float(np.degrees(np.arctan(mid_slope / max(np.mean(y), 1e-9))))
        else:
            slope_angle = 0.0
        
        direction = 'neutral'
        if last_close > last_mid and slope_angle > 0.12:
            direction = 'bull'
        elif last_close < last_mid and slope_angle < -0.12:
            direction = 'bear'
        
        if direction == 'neutral':
            return False, direction, 0, atr_pct
        
        score = min(100.0, abs(slope_angle) / 0.5 * 50)
        return True, direction, score, atr_pct

    def _step2_h1_pullback(self, df: pd.DataFrame, direction: str) -> Tuple[bool, float]:
        """1H: 回踩中轨确认"""
        period = 20
        std_mult = 2.0
        
        df['boll_mid'] = df['c'].rolling(period).mean()
        df['boll_std'] = df['c'].rolling(period).std()
        
        last_close = float(df['c'].iloc[-1])
        last_mid = float(df['boll_mid'].iloc[-1])
        
        dist_pct = abs(last_close - last_mid) / last_mid * 100 if last_mid > 0 else 100
        
        if dist_pct > 2.5:
            return False, 0
        
        delta = df['c'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        last_rsi = float(rsi.iloc[-1])
        
        rsi_ok = False
        if direction == 'bull' and last_rsi < 50:
            rsi_ok = True
        elif direction == 'bear' and last_rsi > 50:
            rsi_ok = True
        
        score = min(100.0, (1 - dist_pct / 3) * 70)
        if rsi_ok:
            score += 20
        
        return True, score

    def _step3_m15_entry(self, df: pd.DataFrame, direction: str) -> Tuple[bool, float]:
        """15m: MACD+放量起爆"""
        period = 20
        std_mult = 2.0
        
        df['boll_mid'] = df['c'].rolling(period).mean()
        
        fast = df['c'].ewm(span=12, adjust=False).mean()
        slow = df['c'].ewm(span=26, adjust=False).mean()
        macd = fast - slow
        signal = macd.ewm(span=9, adjust=False).mean()
        
        last_close = float(df['c'].iloc[-1])
        last_mid = float(df['boll_mid'].iloc[-1])
        
        price_ok = False
        if direction == 'bull' and last_close > last_mid:
            price_ok = True
        elif direction == 'bear' and last_close < last_mid:
            price_ok = True
        
        macd_cross = False
        if direction == 'bull' and float(macd.iloc[-1]) > float(signal.iloc[-1]) and float(macd.iloc[-2]) <= float(signal.iloc[-2]):
            macd_cross = True
        elif direction == 'bear' and float(macd.iloc[-1]) < float(signal.iloc[-1]) and float(macd.iloc[-2]) >= float(signal.iloc[-2]):
            macd_cross = True
        
        if len(df) > 12:
            current_vol = float(df['vol'].iloc[-1])
            avg_vol = float(df['vol'].iloc[-12:-1].mean())
            vol_ok = current_vol > avg_vol * 1.2
        else:
            vol_ok = False
        
        score = 0
        if price_ok:
            score += 35
        if macd_cross:
            score += 40
        if vol_ok:
            score += 20
        
        if score >= 50:
            return True, score
        
        return False, 0

    def _apply_risk_management(self, result: Dict, atr_pct: float, last_price: float, direction: str):
        """ATR 动态风控"""
        stop_loss_pct = min(5.0, max(1.0, atr_pct * 1.8))
        take_profit_pct = stop_loss_pct * 2.8
        
        result['risk_management'] = {
            'stop_loss': last_price * (1 - stop_loss_pct / 100) if direction == 'bull' else last_price * (1 + stop_loss_pct / 100),
            'take_profit': last_price * (1 + take_profit_pct / 100) if direction == 'bull' else last_price * (1 - take_profit_pct / 100),
            'stop_loss_pct': stop_loss_pct,
            'take_profit_pct': take_profit_pct,
            'rr_ratio': take_profit_pct / stop_loss_pct,
            'suggested_position': min(0.1, 2.5 / max(atr_pct, 0.5))
        }

    def _klines_to_df(self, klines_data: List) -> pd.DataFrame:
        """K线数据转换为DataFrame"""
        if not klines_data or len(klines_data) < 10:
            return None
        
        try:
            df = pd.DataFrame(klines_data)
            if len(df.columns) < 6:
                return None
            
            if len(df.columns) >= 6:
                df = df.iloc[:, [0, 1, 2, 3, 4, 5]].copy()
                df.columns = ['ts', 'o', 'h', 'l', 'c', 'vol']
            
            df['ts'] = pd.to_numeric(df['ts'], errors='coerce')
            df['o'] = pd.to_numeric(df['o'], errors='coerce')
            df['h'] = pd.to_numeric(df['h'], errors='coerce')
            df['l'] = pd.to_numeric(df['l'], errors='coerce')
            df['c'] = pd.to_numeric(df['c'], errors='coerce')
            df['vol'] = pd.to_numeric(df['vol'], errors='coerce')
            
            df = df.dropna(subset=['o', 'h', 'l', 'c']).reset_index(drop=True)
            return df if len(df) >= 20 else None
            
        except Exception:
            return None
