"""
专业波段扫描策略 V4.0 - 机构版
Pro Swing Trading Scanner - Institutional Grade

策略设计理念（基于全网最佳实践）:
1. 趋势跟踪 + 均值共振入场
2. 多指标过滤（MA + RSI + MACD + Volume）
3. 严格的风控与仓位管理评估

作者: AI Assistant
版本: 4.0 Scanner
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from src.scanner.base_scanner import BaseScannerStrategy, ScannerSymbol


class ProSwingScanner(BaseScannerStrategy):
    """
    专业波段扫描器 V4.0

    扫描逻辑（共振入场条件）:
    1. 趋势确认：价格站上 50MA（或 20MA 短期趋势）
    2. RSI 均值回归：RSI(14) 处于 30-60 区间（非超买，有上涨空间）
    3. MACD 动能：MACD 柱状图 > 0 或 金叉初期
    4. 成交量：当前成交量 > 20 日均量 * 1.2（资金进场）

    评分制:
    - 完美共振 (85-100 分)
    - 强共振 (70-85 分)
    - 弱共振 (50-70 分)
    - 不共振 (<50 分)
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
            'conditions_total': 4,
            'details': {},
            'score': 0.0,
            'last_price': symbol.last_price,
            'volume_24h': symbol.volume_24h,
            'price_change_24h': symbol.price_change_24h,
            'high_24h': symbol.high_24h,
            'low_24h': symbol.low_24h,
        }

        try:
            # 获取日线数据 (需要从 extra_data 获取)
            klines = symbol.extra_data.get('klines', {})
            daily_klines = klines.get('1D', [])

            if not daily_klines or len(daily_klines) < 60:
                result['details']['状态'] = "数据不足 (<60根)"
                return result

            # 解析 K 线数据
            df = pd.DataFrame(daily_klines, columns=[
                'ts', 'o', 'h', 'l', 'c', 'vol', 'volCcy', 'volCcyQuote', 'confirm'
            ])
            for col in ['o', 'h', 'l', 'c', 'vol']:
                df[col] = df[col].astype(float)

            score = 0.0
            details = []

            # --- 1. 趋势指标 (30 分) ---
            ma50 = df['c'].iloc[-50:].mean()
            ma20 = df['c'].iloc[-20:].mean()
            current_price = df['c'].iloc[-1]

            if current_price > ma50:
                score += 30
                details.append("趋势强 (价>MA50)")
            elif current_price > ma20:
                score += 15
                details.append("趋势中 (价>MA20)")
            else:
                details.append("趋势弱 (价<MA20)")

            # --- 2. RSI 指标 (25 分) ---
            delta = df['c'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = (-delta).where(delta < 0, 0).rolling(window=14).mean()
            rs = gain / loss.replace(0, np.nan)
            rsi = 100 - (100 / (1 + rs))
            current_rsi = rsi.iloc[-1]

            if pd.isna(current_rsi):
                current_rsi = 50

            # 理想区间：30-60（有上涨空间且未超买）
            if 30 <= current_rsi <= 60:
                score += 25
                details.append(f"RSI理想 ({current_rsi:.1f})")
            elif 60 < current_rsi < 70:
                score += 10
                details.append(f"RSI偏强 ({current_rsi:.1f})")
            elif current_rsi < 30:
                score += 15  # 超卖可能反弹，给中等分
                details.append(f"RSI超卖 ({current_rsi:.1f})")
            else:
                details.append(f"RSI超买 ({current_rsi:.1f})")

            # --- 3. MACD 动能 (25 分) ---
            ema12 = df['c'].ewm(span=12, adjust=False).mean()
            ema26 = df['c'].ewm(span=26, adjust=False).mean()
            dif = ema12 - ema26
            dea = dif.ewm(span=9, adjust=False).mean()
            macd_hist = (dif - dea) * 2
            
            current_hist = macd_hist.iloc[-1]
            
            # 柱状图转正或放大
            if current_hist > 0:
                score += 25
                details.append("MACD金叉+")
            elif current_hist > 0 and len(macd_hist) >= 2:
                if macd_hist.iloc[-1] > macd_hist.iloc[-2]:
                    score += 15
                    details.append("MACD改善")
            else:
                details.append("MACD弱势")

            # --- 4. 成交量 (20 分) ---
            vol_ma20 = df['vol'].iloc[-20:].mean()
            current_vol = df['vol'].iloc[-1]

            threshold = self.config.get('volume_threshold', 1.2)
            if current_vol > vol_ma20 * threshold:
                score += 20
                details.append(f"放量 ({current_vol/vol_ma20:.1f}x)")
            elif current_vol > vol_ma20:
                score += 10
                details.append("平量")
            else:
                details.append("缩量")

            # --- 最终评分与通过判断 ---
            result['score'] = score
            result['details']['评估'] = "/".join(details)
            result['details']['RSI'] = f"{current_rsi:.1f}"

            # 设置通过门槛
            min_score = self.config.get('min_score', 75)
            if score >= min_score:
                result['passed'] = True

        except Exception as e:
            result['details']['错误'] = str(e)

        return result

    def get_config_schema(self) -> Dict:
        """获取配置模式"""
        return {
            'volume_threshold': {
                'type': 'float',
                'default': 1.2,
                'label': '成交量放大倍数 (相对于MA20)'
            },
            'min_score': {
                'type': 'int',
                'default': 75,
                'label': '最低通过分数 (0-100)'
            },
        }


# 策略配置模式
CONFIG_SCHEMA = {
    'volume_threshold': {'type': 'float', 'default': 1.2, 'label': '成交量放大倍数'},
    'min_score': {'type': 'int', 'default': 75, 'label': '最低通过分数'},
}
