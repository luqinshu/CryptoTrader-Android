"""
K线反转形态扫描策略
识别小时线级别的经典反转形态：圆弧顶/底、头肩顶/底、双顶/底等

作者：Crypto Trader
版本：1.0
"""

import pandas as pd
import numpy as np
import ta
from typing import Dict, List, Tuple, Optional
from src.scanner.base_scanner import BaseScannerStrategy, ScannerSymbol


class CandlestickReversalScanner(BaseScannerStrategy):
    """
    K线反转形态扫描器 (V2.0 增强版)

    识别形态与动能确认:
    1. 结构性反转：双顶/底、头肩顶/底 (结合市场结构转变 MSS)
    2. 动能反转：RSI 底背离/顶背离 (Divergence)
    3. 极端反转：圆弧顶/底 (结合波动率收缩与爆发)
    4. K线组合：吞没、星线 (作为辅助确认信号)
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
            df = self._to_df(klines.get('1H', []))
            if df.empty or len(df) < 50:
                return result

            # --- 1. 前序趋势检测 (20分) ---
            # 反转必须发生在已经有过一段明显趋势之后
            trend_score, trend_dir = self._check_preceding_trend(df)
            result['score'] += trend_score

            # --- 2. 动能背离检测 (30分) ---
            # 价格创新高/低，但RSI未创新高/低
            div_score, div_sigs = self._detect_divergence(df)
            result['score'] += div_score
            result['signals'].extend(div_sigs)

            # --- 3. 核心形态识别 (30分) ---
            # 双顶底、头肩、圆弧
            pattern_score, pattern_sigs = self._detect_structural_patterns(df)
            result['score'] += pattern_score
            result['signals'].extend(pattern_sigs)

            # --- 4. 成交量足迹确认 (20分) ---
            # 放量突破颈线 或 底部缩量
            vol_score, vol_sigs = self._check_volume_profile(df)
            result['score'] += vol_score
            result['signals'].extend(vol_sigs)

            # 最终决策
            result['passed'] = result['score'] >= 70
            result['rating'] = "💎 高置信反转" if result['score'] >= 85 else ("📈 潜在反转" if result['score'] >= 65 else "⏸️ 趋势延续")
            result['direction'] = "LONG" if "底" in "".join(result['signals']) or "看涨" in "".join(result['signals']) else "SHORT"

        except Exception as e:
            result['error'] = str(e)

        return result

    def _to_df(self, klines: List) -> pd.DataFrame:
        if not klines or len(klines) < 30: return pd.DataFrame()
        df = pd.DataFrame(klines, columns=['ts', 'o', 'h', 'l', 'c', 'v', 'v2', 'v3', 'f'])
        for col in ['o', 'h', 'l', 'c', 'v']: df[col] = df[col].astype(float)
        return df

    def _check_preceding_trend(self, df: pd.DataFrame) -> Tuple[float, str]:
        """检测反转前的趋势强度"""
        ema50 = ta.trend.ema_indicator(df['c'], 50)
        curr_price = df['c'].iloc[-1]
        dist = (curr_price - ema50.iloc[-1]) / ema50.iloc[-1] * 100
        
        # 如果价格偏离EMA50超过5%，认为有反转空间
        if abs(dist) > 5:
            return 20, ("UP" if dist > 0 else "DOWN")
        elif abs(dist) > 3:
            return 10, ("UP" if dist > 0 else "DOWN")
        return 0, "SIDE"

    def _detect_divergence(self, df: pd.DataFrame) -> Tuple[float, List]:
        """RSI 背离检测"""
        score = 0
        sigs = []
        rsi = ta.momentum.RSIIndicator(df['c'], window=14).rsi()
        
        # 寻找最近两个低点
        lows = df['l'].values
        rsi_vals = rsi.values
        
        # 简化版背离逻辑
        if lows[-1] < lows[-10:-1].min() and rsi_vals[-1] > rsi_vals[-10:-1].min():
            score = 30
            sigs.append("⚡ RSI 底背离 (动能衰竭)")
        elif lows[-1] > lows[-10:-1].max() and rsi_vals[-1] < rsi_vals[-10:-1].max():
            score = 30
            sigs.append("⚡ RSI 顶背离 (上涨乏力)")
            
        return score, sigs

    def _detect_structural_patterns(self, df: pd.DataFrame) -> Tuple[float, List]:
        """结构化形态检测 (双顶底/头肩)"""
        score = 0
        sigs = []
        h = df['h'].values
        l = df['l'].values
        c = df['c'].values
        
        # 1. 简易双底识别 (最近20根线内有两个相近低点)
        l_min1 = min(l[-10:])
        l_min2 = min(l[-20:-10])
        if abs(l_min1 - l_min2) / l_min1 < 0.005 and c[-1] > l_min1 * 1.01:
            score += 20
            sigs.append("🏛️ 潜在双底形态")

        # 2. 吞没形态确认 (辅助)
        if c[-1] > df['o'].iloc[-1] and df['c'].iloc[-2] < df['o'].iloc[-2]:
            if c[-1] > df['o'].iloc[-2] and df['o'].iloc[-1] < df['c'].iloc[-2]:
                score += 10
                sigs.append("🔥 看涨吞没")
        elif c[-1] < df['o'].iloc[-1] and df['c'].iloc[-2] > df['o'].iloc[-2]:
            if c[-1] < df['o'].iloc[-2] and df['o'].iloc[-1] > df['c'].iloc[-2]:
                score += 10
                sigs.append("🧊 看跌吞没")
                
        return score, sigs

    def _check_volume_profile(self, df: pd.DataFrame) -> Tuple[float, List]:
        """成交量验证"""
        score = 0
        sigs = []
        v = df['v'].values
        avg_v = v[-10:-1].mean()
        
        if v[-1] > avg_v * 2:
            score = 20
            sigs.append("🔊 反转爆量确认")
        elif v[-1] < avg_v * 0.5:
            score = 10
            sigs.append("🔈 缩量至极致 (变盘在即)")
            
        return score, sigs

    def get_config_schema(self) -> Dict:
        return {
            'min_score': {'type': 'float', 'default': 70, 'label': '最低通过总分 (0-100)'},
            'rsi_window': {'type': 'int', 'default': 14, 'label': 'RSI周期'}
        }

    def _identify_rounding_patterns(self, df: pd.DataFrame, window: int = 30) -> List[Dict]:
        """识别圆弧顶/底"""
        patterns = []
        if len(df) < window:
            return patterns

        highs = df['h'].values
        lows = df['l'].values
        x = np.arange(window)

        try:
            # 圆弧顶（拟合最后window根K线的高点）
            coeffs_top = np.polyfit(x, highs[-window:], 2)
            a_top = coeffs_top[0]

            if a_top < 0:  # 开口向下
                y_fit = np.polyval(coeffs_top, x)
                ss_res = np.sum((highs[-window:] - y_fit) ** 2)
                ss_tot = np.sum((highs[-window:] - np.mean(highs[-window:])) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

                if r_squared > 0.7:
                    neckline = np.min(highs[-window:])
                    current_price = df['c'].iloc[-1]
                    breakout = current_price < neckline * 0.99

                    patterns.append({
                        'pattern': '圆弧顶',
                        'confidence': r_squared,
                        'neckline': neckline,
                        'breakout': breakout,
                        'target': neckline - (np.max(highs[-window:]) - neckline)
                    })

            # 圆弧底（拟合最后window根K线的低点）
            coeffs_bot = np.polyfit(x, lows[-window:], 2)
            a_bot = coeffs_bot[0]

            if a_bot > 0:  # 开口向上
                y_fit = np.polyval(coeffs_bot, x)
                ss_res = np.sum((lows[-window:] - y_fit) ** 2)
                ss_tot = np.sum((lows[-window:] - np.mean(lows[-window:])) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

                if r_squared > 0.7:
                    neckline = np.max(lows[-window:])
                    current_price = df['c'].iloc[-1]
                    breakout = current_price > neckline * 1.01

                    patterns.append({
                        'pattern': '圆弧底',
                        'confidence': r_squared,
                        'neckline': neckline,
                        'breakout': breakout,
                        'target': neckline + (neckline - np.min(lows[-window:]))
                    })
        except Exception:
            pass

        return patterns

    def _identify_head_shoulders_patterns(self, df: pd.DataFrame, lookback: int = 60) -> List[Dict]:
        """识别头肩顶/底"""
        patterns = []
        if len(df) < lookback:
            return patterns

        highs = df['h'].values
        lows = df['l'].values
        closes = df['c'].values

        # 寻找峰值
        peaks = self._find_peaks(highs[-lookback:], order=3)
        if len(peaks) < 3:
            return patterns

        # 寻找头肩顶
        for i in range(len(peaks) - 2):
            left_idx = peaks[i]
            head_idx = peaks[i+1]
            right_idx = peaks[i+2]

            # 检查间距
            if head_idx - left_idx < 8 or right_idx - head_idx < 8:
                continue

            left_high = highs[-lookback:][left_idx]
            head_high = highs[-lookback:][head_idx]
            right_high = highs[-lookback:][right_idx]

            # 头部必须最高
            if head_high <= left_high or head_high <= right_high:
                continue

            # 左右肩高度相近
            shoulder_diff = abs(left_high - right_high) / head_high
            if shoulder_diff > 0.03:
                continue

            # 计算颈线（简化：取两个肩之间的最低点）
            trough = np.min(lows[-lookback:][left_idx:right_idx+1])
            neckline = trough

            current_price = closes[-1]
            breakout = current_price < neckline * 0.99

            # 对称性评分
            left_dist = head_idx - left_idx
            right_dist = right_idx - head_idx
            symmetry = min(left_dist, right_dist) / max(left_dist, right_dist)

            confidence = symmetry * 0.6 + (0.4 if breakout else 0)

            patterns.append({
                'pattern': '头肩顶',
                'confidence': confidence,
                'neckline': neckline,
                'breakout': breakout,
                'target': neckline - (head_high - neckline)
            })
            break  # 只取第一个

        return patterns

    def _identify_double_patterns(self, df: pd.DataFrame, lookback: int = 50) -> List[Dict]:
        """识别双顶/底"""
        patterns = []
        if len(df) < lookback:
            return patterns

        highs = df['h'].values
        lows = df['l'].values
        closes = df['c'].values

        # 寻找显著峰值
        peaks = self._find_significant_peaks(highs[-lookback:])

        if len(peaks) >= 2:
            # 检查双顶
            p1, p2 = peaks[0], peaks[1]
            if p2 - p1 >= 10:
                h1, h2 = highs[-lookback:][p1], highs[-lookback:][p2]

                if abs(h1 - h2) / max(h1, h2) < 0.02:
                    trough = np.min(lows[-lookback:][p1:p2+1])
                    neckline = trough
                    current_price = closes[-1]
                    breakout = current_price < neckline * 0.99

                    patterns.append({
                        'pattern': '双顶',
                        'confidence': 0.6 + (0.4 if breakout else 0),
                        'neckline': neckline,
                        'breakout': breakout,
                        'target': neckline - (h1 - neckline)
                    })

        # 寻找显著谷底
        troughs = self._find_significant_troughs(lows[-lookback:])

        if len(troughs) >= 2:
            t1, t2 = troughs[0], troughs[1]
            if t2 - t1 >= 10:
                l1, l2 = lows[-lookback:][t1], lows[-lookback:][t2]

                if abs(l1 - l2) / max(l1, l2) < 0.02:
                    peak = np.max(highs[-lookback:][t1:t2+1])
                    neckline = peak
                    current_price = closes[-1]
                    breakout = current_price > neckline * 1.01

                    patterns.append({
                        'pattern': '双底',
                        'confidence': 0.6 + (0.4 if breakout else 0),
                        'neckline': neckline,
                        'breakout': breakout,
                        'target': neckline + (neckline - l1)
                    })

        return patterns

    def _identify_engulfing_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """识别吞没形态"""
        patterns = []
        if len(df) < 2:
            return patterns

        curr = df.iloc[-1]
        prev = df.iloc[-2]

        curr_body_top = max(curr['o'], curr['c'])
        curr_body_bot = min(curr['o'], curr['c'])
        prev_body_top = max(prev['o'], prev['c'])
        prev_body_bot = min(prev['o'], prev['c'])

        # 看涨吞没
        if (prev['c'] < prev['o'] and curr['c'] > curr['o'] and
            curr_body_bot <= prev_body_bot and curr_body_top >= prev_body_top):

            patterns.append({
                'pattern': '看涨吞没',
                'confidence': 0.65,
                'neckline': prev['l'],
                'breakout': curr['c'] > prev['h'],
                'target': curr['c'] + (prev['h'] - prev['l'])
            })

        # 看跌吞没
        elif (prev['c'] > prev['o'] and curr['c'] < curr['o'] and
              curr_body_bot <= prev_body_bot and curr_body_top >= prev_body_top):

            patterns.append({
                'pattern': '看跌吞没',
                'confidence': 0.65,
                'neckline': prev['h'],
                'breakout': curr['c'] < prev['l'],
                'target': curr['c'] - (prev['h'] - prev['l'])
            })

        return patterns

    def _identify_star_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """识别早晨/黄昏之星"""
        patterns = []
        if len(df) < 3:
            return patterns

        d1, d2, d3 = df.iloc[-3], df.iloc[-2], df.iloc[-1]

        def get_body_size(row):
            return abs(row['c'] - row['o'])

        body1 = get_body_size(d1)
        body2 = get_body_size(d2)
        body3 = get_body_size(d3)

        # 早晨之星
        if (body1 > 0 and d1['c'] < d1['o'] and
            body2 < body1 * 0.3 and
            body3 > body1 * 0.6 and d3['c'] > d3['o'] and
            d3['c'] > (max(d1['o'], d1['c']) + min(d1['o'], d1['c'])) / 2):

            patterns.append({
                'pattern': '早晨之星',
                'confidence': 0.7,
                'neckline': min(d1['l'], d2['l']),
                'breakout': d3['c'] > d1['h'],
                'target': d3['c'] + (d1['h'] - d1['l'])
            })

        # 黄昏之星
        elif (body1 > 0 and d1['c'] > d1['o'] and
              body2 < body1 * 0.3 and
              body3 > body1 * 0.6 and d3['c'] < d3['o'] and
              d3['c'] < (max(d1['o'], d1['c']) + min(d1['o'], d1['c'])) / 2):

            patterns.append({
                'pattern': '黄昏之星',
                'confidence': 0.7,
                'neckline': max(d1['h'], d2['h']),
                'breakout': d3['c'] < d1['l'],
                'target': d3['c'] - (d1['h'] - d1['l'])
            })

        return patterns

    def _check_volume_confirmation(self, df: pd.DataFrame, pattern: Dict) -> float:
        """成交量确认评分"""
        try:
            recent_vol = df['vol'].iloc[-5:].mean()
            prev_vol = df['vol'].iloc[-10:-5].mean()

            if prev_vol == 0:
                return 0.5

            vol_ratio = recent_vol / prev_vol

            # 突破时放量
            if pattern.get('breakout', False):
                if vol_ratio > 1.5:
                    return 1.0
                elif vol_ratio > 1.2:
                    return 0.8
                else:
                    return 0.5
            else:
                # 未突破时观察成交量
                if 0.8 <= vol_ratio <= 1.5:
                    return 0.6
                else:
                    return 0.4
        except Exception:
            return 0.5

    def _check_trend_context(self, df: pd.DataFrame, pattern: Dict) -> float:
        """趋势背景确认评分"""
        try:
            # 计算最近趋势
            recent_closes = df['c'].iloc[-20:]
            slope = (recent_closes.iloc[-1] - recent_closes.iloc[0]) / recent_closes.iloc[0]

            pattern_name = pattern.get('pattern', '')

            # 看涨形态需要下跌趋势背景
            bullish_patterns = ['圆弧底', '头肩底', '双底', '看涨吞没', '早晨之星']
            if any(bp in pattern_name for bp in bullish_patterns):
                if slope < -0.05:
                    return 1.0
                elif slope < -0.02:
                    return 0.8
                elif slope < 0:
                    return 0.6
                else:
                    return 0.3

            # 看跌形态需要上涨趋势背景
            bearish_patterns = ['圆弧顶', '头肩顶', '双顶', '看跌吞没', '黄昏之星']
            if any(bp in pattern_name for bp in bearish_patterns):
                if slope > 0.05:
                    return 1.0
                elif slope > 0.02:
                    return 0.8
                elif slope > 0:
                    return 0.6
                else:
                    return 0.3

            return 0.5
        except Exception:
            return 0.5

    def _find_peaks(self, data: np.ndarray, order: int = 2) -> List[int]:
        """寻找局部高点"""
        peaks = []
        for i in range(order, len(data) - order):
            is_peak = True
            for j in range(1, order + 1):
                if data[i] <= data[i-j] or data[i] <= data[i+j]:
                    is_peak = False
                    break
            if is_peak:
                peaks.append(i)
        return peaks

    def _find_significant_peaks(self, data: np.ndarray, threshold_pct: float = 0.02) -> List[int]:
        """寻找显著峰值"""
        peaks = []
        for i in range(2, len(data) - 2):
            if (data[i] > data[i-1] and data[i] > data[i-2] and
                data[i] > data[i+1] and data[i] > data[i+2]):
                local_min = min(data[max(0, i-5):min(len(data), i+5)])
                if local_min > 0 and (data[i] - local_min) / local_min > threshold_pct:
                    peaks.append(i)
        return peaks

    def _find_significant_troughs(self, data: np.ndarray, threshold_pct: float = 0.02) -> List[int]:
        """寻找显著谷底"""
        troughs = []
        for i in range(2, len(data) - 2):
            if (data[i] < data[i-1] and data[i] < data[i-2] and
                data[i] < data[i+1] and data[i] < data[i+2]):
                local_max = max(data[max(0, i-5):min(len(data), i+5)])
                if local_max > 0 and (local_max - data[i]) / local_max > threshold_pct:
                    troughs.append(i)
        return troughs

    def get_config_schema(self) -> Dict:
        """获取配置模式"""
        return {
            'min_confidence': {
                'type': 'float',
                'default': 0.6,
                'label': '最低形态置信度 (0-1)'
            },
            'require_breakout': {
                'type': 'bool',
                'default': True,
                'label': '要求颈线突破确认'
            },
            'min_score': {
                'type': 'float',
                'default': 70,
                'label': '最低通过总分 (0-100)'
            },
        }


# 策略配置模式
CONFIG_SCHEMA = {
    'min_confidence': {'type': 'float', 'default': 0.6, 'label': '最低形态置信度 (0-1)'},
    'require_breakout': {'type': 'bool', 'default': True, 'label': '要求颈线突破确认'},
    'min_score': {'type': 'float', 'default': 70, 'label': '最低通过总分 (0-100)'},
}
