"""
多指标趋势验证扫描策略（严格版）
日线三种经典指标验证上涨趋势 + 多周期均线聚集确认企稳

作者：Crypto Trader
版本：3.0 严格版
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from src.scanner.base_scanner import BaseScannerStrategy, ScannerSymbol


class MultiIndicatorTrendScanner(BaseScannerStrategy):
    """
    多指标趋势验证扫描器（严格版）

    扫描逻辑:
    1. 日线级别：三种经典指标验证上涨趋势
       - MA 均线系统（多头排列 + 价格位置 + 长期趋势）
       - MACD 动能指标（金叉/零轴上方/柱状图放量）
       - RSI 相对强弱（理想趋势区域 55-70）
    
    2. 小时线：多周期均线聚集（MA5/MA10/MA20 严格收敛 + 价格企稳）
    
    3. 15 分钟线：多周期均线聚集确认企稳 + 价格行为确认 + 成交量配合

    优化点:
    - 提高通过门槛：总分 >= 85 分（原来 70 分）
    - 收紧均线聚集阈值
    - 增加成交量增长要求
    - 增加 24h 最低成交量过滤
    """

    def _init_conditions(self):
        """初始化扫描条件"""
        pass

    def scan_symbol(self, symbol: ScannerSymbol) -> Dict:
        """扫描单个交易对"""
        result = {
            'symbol': symbol.inst_id,
            'passed': False,
            'conditions_met': 0,
            'conditions_total': 3,
            'details': {},
            'score': 0.0,
            'last_price': symbol.last_price,
            'volume_24h': symbol.volume_24h,
            'price_change_24h': symbol.price_change_24h,
            'high_24h': symbol.high_24h,
            'low_24h': symbol.low_24h,
            'trend_strength': 0.0,
            'convergence_score': 0.0,
        }

        try:
            # 过滤：24h 成交量过低直接跳过
            min_volume = self.config.get('min_volume_24h', 5000000)  # 默认 500 万 USDT
            if symbol.volume_24h < min_volume:
                result['details']['成交量'] = f"不足{min_volume/1000000:.0f}M"
                result['score'] = 0.0
                return result

            klines = symbol.extra_data.get('klines', {})
            daily_klines = klines.get('1D', [])
            hourly_klines = klines.get('1H', [])
            min15_klines = klines.get('15m', [])

            # 1. 日线三指标验证上涨趋势
            daily_score, daily_info = self._check_daily_trend(daily_klines)
            result['details']['日线趋势'] = daily_info
            result['trend_strength'] = daily_score
            
            # 严格门槛：日线评分需 >= 0.7（原来 0.5）
            if daily_score < 0.7:
                result['score'] = daily_score * 33
                return result
            
            result['conditions_met'] += 1

            # 2. 小时线均线聚集
            hourly_score, hourly_info = self._check_hourly_ma_convergence(hourly_klines)
            result['details']['小时线聚集'] = hourly_info
            result['convergence_score'] = max(result['convergence_score'], hourly_score)
            
            # 严格门槛：小时线评分需 >= 0.6（原来 0.4）
            if hourly_score < 0.6:
                result['score'] = 33 + hourly_score * 33
                return result
            
            result['conditions_met'] += 1

            # 3. 15 分钟线均线聚集确认
            min15_score, min15_info = self._check_15min_ma_convergence(min15_klines)
            result['details']['15 分钟企稳'] = min15_info
            
            # 严格门槛：15 分钟评分需 >= 0.6（原来 0.4）
            if min15_score < 0.6:
                result['score'] = 66 + min15_score * 34
                return result
            
            result['conditions_met'] += 1

            # 计算总分
            result['score'] = (daily_score * 33) + (hourly_score * 33) + (min15_score * 34)
            result['convergence_score'] = (hourly_score + min15_score) / 2

            # 判断是否通过（严格标准）
            min_total_score = self.config.get('min_total_score', 85)  # 默认 85 分
            
            if result['score'] >= min_total_score and daily_score >= 0.7 and hourly_score >= 0.6 and min15_score >= 0.6:
                result['passed'] = True

        except Exception as e:
            result['details']['错误'] = str(e)

        return result

    def _check_daily_trend(self, klines: List) -> Tuple[float, str]:
        """日线级别三指标验证上涨趋势（评分制 - 严格版）"""
        try:
            if not klines or len(klines) < 30:
                return 0.0, "数据不足"

            df = pd.DataFrame(klines, columns=[
                'ts', 'o', 'h', 'l', 'c', 'vol', 'volCcy', 'volCcyQuote', 'confirm'
            ])
            for col in ['o', 'h', 'l', 'c', 'vol']:
                df[col] = df[col].astype(float)

            score = 0.0
            details = []

            # === 指标 1: MA 多头排列 (40 分) ===
            ma5 = df['c'].iloc[-5:].mean()
            ma10 = df['c'].iloc[-10:].mean()
            ma20 = df['c'].iloc[-20:].mean()
            ma60 = df['c'].iloc[-60:].mean() if len(df) >= 60 else ma20
            
            current_price = df['c'].iloc[-1]
            
            # 严格：只有完美多头排列才能得高分
            if ma5 > ma10 > ma20 > ma60:
                score += 0.4
                details.append("MA 完美多头")
            elif ma5 > ma10 > ma20:
                score += 0.25  # 降低分数
                details.append("MA 标准多头")
            elif ma5 > ma10:
                score += 0.1
                details.append("MA 弱多")
            else:
                details.append("MA 空")

            # 价格必须在 MA20 上方
            if current_price > ma20 * 1.02:  # 至少高出 2%
                score += 0.15
                details.append("价>MA20(+2%)")
            elif current_price > ma20:
                score += 0.05
                details.append("价>MA20")
            
            # === 指标 2: MACD 动能 (30 分) ===
            ema12 = df['c'].ewm(span=12, adjust=False).mean()
            ema26 = df['c'].ewm(span=26, adjust=False).mean()
            dif = ema12 - ema26
            dea = dif.ewm(span=9, adjust=False).mean()
            macd_hist = (dif - dea) * 2
            
            current_dif = dif.iloc[-1]
            current_dea = dea.iloc[-1]
            current_hist = macd_hist.iloc[-1]
            
            # 严格：必须金叉且柱状图正
            if current_dif > current_dea and current_hist > 0:
                score += 0.3
                details.append("MACD 金叉+")
            elif current_dif > 0 and current_dea > 0:
                score += 0.15
                details.append("MACD 零轴上")
            elif current_hist > 0:
                score += 0.05
                details.append("MACD 柱+")
            else:
                details.append("MACD 弱")

            # MACD 柱状图必须递增
            if len(macd_hist) >= 3:
                recent_hist = macd_hist.iloc[-3:]
                if all(recent_hist.iloc[i] < recent_hist.iloc[i+1] for i in range(2)):
                    score += 0.1  # 增加分数
                    details.append("动能增强")

            # === 指标 3: RSI 趋势区域 (30 分) ===
            delta = df['c'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
            rs = gain / loss.replace(0, np.nan)
            rsi = 100 - (100 / (1 + rs))
            current_rsi = rsi.iloc[-1]
            
            if pd.isna(current_rsi):
                current_rsi = 50

            # 严格：只给理想区域高分
            if 55 <= current_rsi <= 70:
                score += 0.3
                details.append(f"RSI{current_rsi:.0f}理想")
            elif 70 < current_rsi < 75:  # 稍微放宽上限
                score += 0.1
                details.append(f"RSI{current_rsi:.0f}偏强")
            elif 50 <= current_rsi < 55:
                score += 0.05
                details.append(f"RSI{current_rsi:.0f}弱")
            else:
                details.append(f"RSI{current_rsi:.0f}不符")

            return min(score, 1.0), "/".join(details)

        except Exception as e:
            return 0.0, f"异常:{e}"

    def _check_hourly_ma_convergence(self, klines: List) -> Tuple[float, str]:
        """小时线多周期均线聚集（收敛）- 评分制（严格版）"""
        try:
            if not klines or len(klines) < 30:
                return 0.0, "数据不足"

            df = pd.DataFrame(klines, columns=[
                'ts', 'o', 'h', 'l', 'c', 'vol', 'volCcy', 'volCcyQuote', 'confirm'
            ])
            for col in ['o', 'h', 'l', 'c', 'vol']:
                df[col] = df[col].astype(float)

            ma5 = df['c'].iloc[-5:].mean()
            ma10 = df['c'].iloc[-10:].mean()
            ma20 = df['c'].iloc[-20:].mean()
            
            current_price = df['c'].iloc[-1]

            mas = [ma5, ma10, ma20]
            max_ma = max(mas)
            min_ma = min(mas)
            avg_ma = sum(mas) / len(mas)
            spread_pct = ((max_ma - min_ma) / min_ma) * 100

            # 严格：降低阈值
            threshold = self.config.get('hourly_ma_threshold', 1.2)  # 原来 2.0
            
            if spread_pct <= threshold * 0.5:
                conv_score = 0.7
            elif spread_pct <= threshold:
                conv_score = 0.5 + 0.2 * (1 - spread_pct / threshold)
            elif spread_pct <= threshold * 1.5:
                conv_score = 0.3 * (1 - (spread_pct - threshold) / (threshold * 0.5))
            else:
                conv_score = 0.0

            # 价格必须在均线组上方
            if current_price > max_ma:
                price_score = 0.3
            elif current_price > avg_ma:
                price_score = 0.15
            elif current_price > min_ma:
                price_score = 0.05
            else:
                price_score = 0.0

            total_score = conv_score + price_score
            
            if total_score >= 0.7:
                return total_score, f"聚集{spread_pct:.1f}%"
            else:
                return total_score, f"发散{spread_pct:.1f}%"

        except Exception as e:
            return 0.0, f"异常:{e}"

    def _check_15min_ma_convergence(self, klines: List) -> Tuple[float, str]:
        """15 分钟线多周期均线聚集确认企稳 - 评分制（严格版）"""
        try:
            if not klines or len(klines) < 30:
                return 0.0, "数据不足"

            df = pd.DataFrame(klines, columns=[
                'ts', 'o', 'h', 'l', 'c', 'vol', 'volCcy', 'volCcyQuote', 'confirm'
            ])
            for col in ['o', 'h', 'l', 'c', 'vol']:
                df[col] = df[col].astype(float)

            ma5 = df['c'].iloc[-5:].mean()
            ma10 = df['c'].iloc[-10:].mean()
            ma20 = df['c'].iloc[-20:].mean()
            
            current_price = df['c'].iloc[-1]

            mas = [ma5, ma10, ma20]
            max_ma = max(mas)
            min_ma = min(mas)
            spread_pct = ((max_ma - min_ma) / min_ma) * 100
            
            # 严格：降低阈值
            threshold = self.config.get('min15_ma_threshold', 0.8)  # 原来 1.5
            
            if spread_pct <= threshold * 0.5:
                conv_score = 0.5
            elif spread_pct <= threshold:
                conv_score = 0.3 + 0.2 * (1 - spread_pct / threshold)
            elif spread_pct <= threshold * 1.5:
                conv_score = 0.15 * (1 - (spread_pct - threshold) / (threshold * 0.5))
            else:
                conv_score = 0.0

            # 价格波动必须更小
            recent_closes = df['c'].iloc[-5:]
            price_range = (recent_closes.max() - recent_closes.min()) / recent_closes.min() * 100
            
            if price_range < 0.3:  # 严格：原来 0.5
                stab_score = 0.3
            elif price_range < 0.6:  # 严格：原来 1.0
                stab_score = 0.2
            elif price_range < 1.0:
                stab_score = 0.1
            else:
                stab_score = 0.0

            # 成交量必须温和放大
            recent_vol = df['vol'].iloc[-5:].mean()
            prev_vol = df['vol'].iloc[-10:-5].mean()
            vol_ratio = recent_vol / prev_vol if prev_vol > 0 else 1
            
            # 严格：要求在 0.9-1.3 之间（原来 0.8-1.5）
            if 0.9 <= vol_ratio <= 1.3:
                vol_score = 0.2
            elif 0.7 <= vol_ratio < 0.9 or 1.3 < vol_ratio <= 1.8:
                vol_score = 0.1
            else:
                vol_score = 0.0

            total_score = conv_score + stab_score + vol_score
            
            if total_score >= 0.7:
                return total_score, f"企稳{spread_pct:.1f}%"
            else:
                return total_score, f"未稳{spread_pct:.1f}%"

        except Exception as e:
            return 0.0, f"异常:{e}"

    def get_config_schema(self) -> Dict:
        """获取配置模式"""
        return {
            'hourly_ma_threshold': {
                'type': 'float',
                'default': 1.2,
                'label': '小时线均线聚集阈值 (%)'
            },
            'min15_ma_threshold': {
                'type': 'float',
                'default': 0.8,
                'label': '15 分钟均线聚集阈值 (%)'
            },
            'min_total_score': {
                'type': 'float',
                'default': 85,
                'label': '最低通过总分 (0-100)'
            },
            'min_volume_24h': {
                'type': 'float',
                'default': 5000000,
                'label': '最低 24h 成交量 (USDT)'
            },
        }


# 策略配置模式（严格标准）
CONFIG_SCHEMA = {
    'hourly_ma_threshold': {'type': 'float', 'default': 1.2, 'label': '小时线均线聚集阈值 (%)'},
    'min15_ma_threshold': {'type': 'float', 'default': 0.8, 'label': '15 分钟均线聚集阈值 (%)'},
    'min_total_score': {'type': 'float', 'default': 85, 'label': '最低通过总分 (0-100)'},
    'min_volume_24h': {'type': 'float', 'default': 5000000, 'label': '最低 24h 成交量 (USDT)'},
}
