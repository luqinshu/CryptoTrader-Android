"""
多周期突破回调扫描策略（优化版）
日线突破 → 小时线回调企稳 → 3分钟线企稳确认

作者：Crypto Trader
版本：2.0
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List
from src.scanner.base_scanner import (
    BaseScannerStrategy, ScannerSymbol, ScanCondition, ScanResult
)


class MultiTimeframeBreakoutScanner(BaseScannerStrategy):
    """
    多周期突破回调扫描器（优化版）

    扫描逻辑:
    1. 日线：价格突破近期高点（N日突破）
    2. 小时线：突破后回调，但在支撑位企稳（不跌破关键支撑）
    3. 3分钟线：企稳确认（价格开始反弹，成交量放大）

    三个条件都满足时，在列表中显示该交易对
    """

    def _init_conditions(self):
        """初始化扫描条件"""
        pass

    def scan_symbol(self, symbol: ScannerSymbol) -> Dict:
        """扫描单个交易对（多周期分析）"""
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
        }

        try:
            klines = symbol.extra_data.get('klines', {})
            daily_klines = klines.get('1D', [])
            hourly_klines = klines.get('1H', [])
            min3_klines = klines.get('3m', [])

            # 1. 检查日线突破
            daily_passed, daily_info = self._check_daily_breakout(daily_klines)
            result['details']['日线突破'] = daily_info
            if daily_passed:
                result['conditions_met'] += 1
            else:
                return result

            # 2. 检查小时线回调企稳
            hourly_passed, hourly_info = self._check_hourly_pullback(hourly_klines)
            result['details']['小时线企稳'] = hourly_info
            if hourly_passed:
                result['conditions_met'] += 1
            else:
                return result

            # 3. 检查3分钟线企稳确认
            min3_passed, min3_info = self._check_3min_confirmation(min3_klines)
            result['details']['3分钟企稳'] = min3_info
            if min3_passed:
                result['conditions_met'] += 1
            else:
                return result

            result['passed'] = True
            result['score'] = 100.0

        except Exception as e:
            result['details']['错误'] = str(e)

        return result

    def _check_daily_breakout(self, klines: List) -> tuple:
        """
        检查日线突破

        条件：
        - 当前价格突破近 N 日最高点（更宽松）
        - 突破幅度可以很小甚至为 0（只要在高位）
        """
        try:
            if not klines or len(klines) < 30:
                return False, "数据不足(<30根)"

            df = pd.DataFrame(klines, columns=[
                'ts', 'o', 'h', 'l', 'c', 'vol', 'volCcy', 'volCcyQuote', 'confirm'
            ])
            for col in ['o', 'h', 'l', 'c', 'vol']:
                df[col] = df[col].astype(float)

            breakout_days = self.config.get('breakout_days', 20)
            breakout_threshold = self.config.get('breakout_threshold', 0.5)  # 降低到 0.5%

            # 计算近 N 日最高价（不包括最后一根 K 线）
            if len(df) < breakout_days + 2:
                return False, f"数据不足(< {breakout_days + 2}根)"

            recent_high = df['h'].iloc[-breakout_days-1:-1].max()
            current_price = df['c'].iloc[-1]
            breakout_pct = ((current_price - recent_high) / recent_high) * 100

            # 优化：突破幅度 >= 0.5% 就算通过
            if current_price >= recent_high and breakout_pct >= breakout_threshold:
                return True, f"突破 {breakout_pct:.1f}% (>{breakout_threshold}%)"
            else:
                return False, f"未突破 ({breakout_pct:.1f}%)"

        except Exception as e:
            return False, f"异常: {e}"

    def _check_hourly_pullback(self, klines: List) -> tuple:
        """
        检查小时线回调企稳（优化版）

        条件更宽松：
        - 价格从高点回调 1.5% 以上即可（原来 3%）
        - 企稳 K 线数减少到 3 根（原来 6 根）
        - 成交量条件放宽
        """
        try:
            if not klines or len(klines) < 24:
                return False, "数据不足(<24根)"

            df = pd.DataFrame(klines, columns=[
                'ts', 'o', 'h', 'l', 'c', 'vol', 'volCcy', 'volCcyQuote', 'confirm'
            ])
            for col in ['o', 'h', 'l', 'c', 'vol']:
                df[col] = df[col].astype(float)

            # 优化参数：更宽松
            pullback_pct = self.config.get('pullback_pct', 1.5)  # 降低到 1.5%
            stabilization_bars = self.config.get('stabilization_bars', 3)  # 减少到 3 根

            # 找到 24 小时内最高点
            recent_high = df['h'].iloc[-24:].max()
            current_price = df['c'].iloc[-1]

            # 计算从高点回调幅度
            pullback = ((recent_high - current_price) / recent_high) * 100

            # 检查是否有回调（不要太宽松）
            if pullback < pullback_pct:
                return False, f"回调不足 {pullback:.1f}%(需>{pullback_pct}%)"

            # 检查是否企稳：最近 N 根 K 线不再创新低
            # 优化：检查最近 stabilization_bars 根 K 线的低点是否在抬升
            if len(df) >= stabilization_bars + 1:
                recent_lows = df['l'].iloc[-stabilization_bars:]
                lowest_point = recent_lows.min()
                last_low = df['l'].iloc[-1]

                # 只要最后 K 线的低点接近最低点就算企稳（放宽条件）
                stabilization_threshold = lowest_point * 1.02  # 允许 2% 误差
                if last_low > stabilization_threshold:
                    # 未企稳：还在创新低
                    return False, f"未企稳(新低)"

            # 优化：成交量条件放宽
            if len(df) >= 12:
                recent_vol = df['vol'].iloc[-6:].mean()
                prev_vol = df['vol'].iloc[-12:-6].mean()
                vol_ratio = recent_vol / prev_vol if prev_vol > 0 else 1

                # 放宽到 2 倍
                if vol_ratio > 2.0:
                    return False, f"放量下跌(量比 {vol_ratio:.2f})"

            return True, f"企稳(回调{pullback:.1f}%)"

        except Exception as e:
            return False, f"异常: {e}"

    def _check_3min_confirmation(self, klines: List) -> tuple:
        """
        检查3分钟线企稳确认（优化版）

        条件更宽松：
        - 最小反弹百分比降低到 0.2%（原来 0.5%）
        - 确认 K 线数减少到 5（原来 10）
        - 均线条件放宽
        """
        try:
            if not klines or len(klines) < 30:
                return False, "数据不足(<30根)"

            df = pd.DataFrame(klines, columns=[
                'ts', 'o', 'h', 'l', 'c', 'vol', 'volCcy', 'volCcyQuote', 'confirm'
            ])
            for col in ['o', 'h', 'l', 'c', 'vol']:
                df[col] = df[col].astype(float)

            # 优化参数：更宽松
            confirm_bars = self.config.get('confirm_bars', 5)  # 减少到 5 根
            min_rebound_pct = self.config.get('min_rebound_pct', 0.2)  # 降低到 0.2%

            if len(df) < confirm_bars + 5:
                return False, f"数据不足"

            # 检查价格是否开始反弹
            recent_closes = df['c'].iloc[-confirm_bars:]
            first_close = recent_closes.iloc[0]
            last_close = recent_closes.iloc[-1]

            rebound_pct = ((last_close - first_close) / first_close) * 100

            if rebound_pct < min_rebound_pct:
                return False, f"反弹不足 {rebound_pct:.2f}%(需>{min_rebound_pct}%)"

            # 检查均线是否向上（短期均线多头排列）
            ma5 = df['c'].iloc[-5:].mean()
            ma10 = df['c'].iloc[-10:].mean()
            ma_bullish = ma5 >= ma10  # 放宽：大于等于

            if ma_bullish:
                return True, f"确认反弹 +{rebound_pct:.2f}%(均线多)"
            else:
                # 即使均线不理想，只要有反弹也勉强通过（进一步优化）
                if rebound_pct >= min_rebound_pct * 2:
                    return True, f"反弹 +{rebound_pct:.2f}%(弱)"
                return False, f"均线空(ma5={ma5:.2f}<ma10={ma10:.2f})"

        except Exception as e:
            return False, f"异常: {e}"

    def get_config_schema(self) -> Dict:
        """获取配置模式"""
        return {
            'breakout_days': {
                'type': 'int',
                'default': 20,
                'label': '突破天数（日线）'
            },
            'breakout_threshold': {
                'type': 'float',
                'default': 0.5,
                'label': '突破幅度阈值 (%)'
            },
            'pullback_pct': {
                'type': 'float',
                'default': 1.5,
                'label': '回调百分比 (%)'
            },
            'stabilization_bars': {
                'type': 'int',
                'default': 3,
                'label': '企稳 K 线数（小时线）'
            },
            'confirm_bars': {
                'type': 'int',
                'default': 5,
                'label': '确认 K 线数（3分钟线）'
            },
            'min_rebound_pct': {
                'type': 'float',
                'default': 0.2,
                'label': '最小反弹百分比 (%)'
            },
        }


# 策略配置模式（优化后的默认值）
CONFIG_SCHEMA = {
    'breakout_days': {'type': 'int', 'default': 20, 'label': '突破天数（日线）'},
    'breakout_threshold': {'type': 'float', 'default': 0.5, 'label': '突破幅度阈值 (%)'},
    'pullback_pct': {'type': 'float', 'default': 1.5, 'label': '回调百分比 (%)'},
    'stabilization_bars': {'type': 'int', 'default': 3, 'label': '企稳 K 线数（小时线）'},
    'confirm_bars': {'type': 'int', 'default': 5, 'label': '确认 K 线数（3分钟线）'},
    'min_rebound_pct': {'type': 'float', 'default': 0.2, 'label': '最小反弹百分比 (%)'},
}
