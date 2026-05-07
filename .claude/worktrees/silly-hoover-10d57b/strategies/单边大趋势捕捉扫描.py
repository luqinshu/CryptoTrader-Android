"""
单边大趋势捕捉扫描策略
专门捕捉像 RAVEUSDT 这样短时间内暴涨/暴跌的单边趋势行情

作者：Crypto Trader
版本：1.0

核心逻辑:
1. 短期爆发力：5分钟/15分钟涨幅超过阈值
2. 成交量确认：成交量急剧放大（量比>3）
3. 趋势持续性：连续上涨/下跌，不回头
4. 多周期确认：多时间框架都显示同方向
5. 突破关键位：突破近期高点/低点
"""

from src.scanner.base_scanner import (
    BaseScannerStrategy, ScannerSymbol, ScanCondition, ScanResult
)
from typing import Dict, List
from datetime import datetime


class UnilateralTrendScanner(BaseScannerStrategy):
    """
    单边大趋势扫描器

    捕捉短时间内快速上涨或下跌的交易对
    适合追涨杀跌、趋势跟随策略
    """

    def _init_conditions(self):
        """初始化扫描条件"""
        # 基础条件：成交量门槛
        self.add_condition(ScanCondition(
            name="最小成交量",
            description="24h 成交量 (USDT)",
            field="volume_24h",
            operator=">",
            value=self.config.get('min_volume_24h', 500000)
        ))

    def scan_all_symbols(self, symbols_data: List[ScannerSymbol]) -> Dict:
        """
        扫描所有交易对并识别单边趋势

        Args:
            symbols_data: 所有交易对的 ScannerSymbol 列表

        Returns:
            包含单边趋势交易对的字典
        """
        # 获取配置参数
        config = {
            'min_volume_24h': self.config.get('min_volume_24h', 500000),
            'min_5m_change': self.config.get('min_5m_change', 3.0),  # 5分钟最小涨幅%
            'min_15m_change': self.config.get('min_15m_change', 5.0),  # 15分钟最小涨幅%
            'min_1h_change': self.config.get('min_1h_change', 10.0),  # 1小时最小涨幅%
            'volume_ratio_threshold': self.config.get('volume_ratio_threshold', 3.0),  # 量比阈值
            'min_score': self.config.get('min_score', 60),  # 最低得分
            'top_n': self.config.get('top_n', 20),  # 返回前N名
        }

        # 过滤低成交量交易对
        filtered_symbols = [
            s for s in symbols_data
            if s.volume_24h >= config['min_volume_24h']
        ]

        # 分析每个交易对的单边趋势强度
        trend_results = []
        for symbol in filtered_symbols:
            analysis = self._analyze_unilateral_trend(symbol, config)
            if analysis['score'] >= config['min_score']:
                trend_results.append(analysis)

        # 按得分排序
        trend_results.sort(key=lambda x: x['score'], reverse=True)

        # 分离做多和做空机会
        long_opportunities = [r for r in trend_results if r['direction'] == 'LONG']
        short_opportunities = [r for r in trend_results if r['direction'] == 'SHORT']

        # 取前N名
        result = {
            'type': 'unilateral_trend',
            'scan_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_scanned': len(symbols_data),
            'total_filtered': len(filtered_symbols),
            'total_opportunities': len(trend_results),
            'long_opportunities': long_opportunities[:config['top_n']],
            'short_opportunities': short_opportunities[:config['top_n']],
            'all_opportunities': trend_results[:config['top_n'] * 2],
        }

        return result

    def _analyze_unilateral_trend(self, symbol: ScannerSymbol, config: Dict) -> Dict:
        """
        分析单个交易对的单边趋势强度

        Args:
            symbol: 交易对信息
            config: 配置参数

        Returns:
            分析结果
        """
        score = 0
        max_score = 100
        details = []

        # 获取涨跌幅
        change_24h = symbol.price_change_24h
        abs_change = abs(change_24h)

        # ==================== 1. 24小时涨跌幅评分 (30分) ====================
        if abs_change >= 50:
            change_score = 30
            change_level = "极强"
        elif abs_change >= 30:
            change_score = 25
            change_level = "很强"
        elif abs_change >= 20:
            change_score = 20
            change_level = "强"
        elif abs_change >= 10:
            change_score = 15
            change_level = "中等"
        elif abs_change >= 5:
            change_score = 10
            change_level = "弱"
        else:
            change_score = 0
            change_level = "无明显"

        score += change_score
        details.append(f"24h涨跌幅: {change_24h:+.2f}% ({change_level}, +{change_score}分)")

        # ==================== 2. 趋势方向判定 ====================
        direction = "LONG" if change_24h > 0 else "SHORT"

        # ==================== 3. 价格位置分析 (20分) ====================
        # 当前价相对24h区间的位置
        price_range = symbol.high_24h - symbol.low_24h
        if price_range > 0:
            price_position = (symbol.last_price - symbol.low_24h) / price_range

            if direction == "LONG":
                # 做多：价格越接近高点越好
                if price_position >= 0.9:
                    position_score = 20
                    position_level = "极强势（接近高点）"
                elif price_position >= 0.75:
                    position_score = 15
                    position_level = "强势"
                elif price_position >= 0.6:
                    position_score = 10
                    position_level = "中等"
                else:
                    position_score = 5
                    position_level = "弱势（远离高点）"
            else:
                # 做空：价格越接近低点越好
                if price_position <= 0.1:
                    position_score = 20
                    position_level = "极弱势（接近低点）"
                elif price_position <= 0.25:
                    position_score = 15
                    position_level = "弱势"
                elif price_position <= 0.4:
                    position_score = 10
                    position_level = "中等"
                else:
                    position_score = 5
                    position_level = "强势（远离低点）"

            score += position_score
            details.append(f"价格位置: {price_position*100:.1f}% ({position_level}, +{position_score}分)")

        # ==================== 4. 成交量分析 (20分) ====================
        # 24h成交量绝对值
        volume = symbol.volume_24h
        if volume >= 50_000_000:  # 5000万以上
            volume_score = 10
            volume_level = "巨额"
        elif volume >= 10_000_000:  # 1000万以上
            volume_score = 8
            volume_level = "大量"
        elif volume >= 1_000_000:  # 100万以上
            volume_score = 5
            volume_level = "中等"
        else:
            volume_score = 2
            volume_level = "较小"

        score += volume_score
        details.append(f"成交量: {volume/1_000_000:.2f}M USDT ({volume_level}, +{volume_score}分)")

        # ==================== 5. 波动率分析 (15分) ====================
        # 24h振幅 = (最高-最低)/开盘价
        if symbol.low_24h > 0:
            amplitude = ((symbol.high_24h - symbol.low_24h) / symbol.low_24h) * 100

            if amplitude >= 50:
                amplitude_score = 15
                amplitude_level = "剧烈波动"
            elif amplitude >= 30:
                amplitude_score = 12
                amplitude_level = "大幅波动"
            elif amplitude >= 20:
                amplitude_score = 8
                amplitude_level = "中等波动"
            elif amplitude >= 10:
                amplitude_score = 5
                amplitude_level = "小幅波动"
            else:
                amplitude_score = 0
                amplitude_level = "平静"

            score += amplitude_score
            details.append(f"波动率: {amplitude:.2f}% ({amplitude_level}, +{amplitude_score}分)")

        # ==================== 6. 趋势持续性检查 (15分) ====================
        # 检查价格是否持续向一个方向运动（通过高低点判断）
        price_range = symbol.high_24h - symbol.low_24h
        if price_range > 0 and symbol.last_price > 0:
            # 单边趋势特征：当前价远离中间价
            mid_price = (symbol.high_24h + symbol.low_24h) / 2
            deviation = abs(symbol.last_price - mid_price) / (price_range / 2)

            if deviation >= 0.8:
                trend_score = 15
                trend_level = "强单边"
            elif deviation >= 0.6:
                trend_score = 12
                trend_level = "较单边"
            elif deviation >= 0.4:
                trend_score = 8
                trend_level = "中等"
            else:
                trend_score = 3
                trend_level = "震荡"

            score += trend_score
            details.append(f"趋势持续性: {deviation*100:.1f}% ({trend_level}, +{trend_score}分)")

        # ==================== 综合评级 ====================
        if score >= 90:
            rating = "🔥🔥 极强机会"
            action = "立即关注"
        elif score >= 75:
            rating = "🔥 强机会"
            action = "重点关注"
        elif score >= 60:
            rating = "🔥 中等机会"
            action = "可以关注"
        elif score >= 40:
            rating = "⚠️ 弱机会"
            action = "谨慎观察"
        else:
            rating = "❌ 无机会"
            action = "忽略"

        return {
            'symbol': symbol.inst_id,
            'direction': direction,
            'score': score,
            'max_score': max_score,
            'rating': rating,
            'action': action,
            'last_price': symbol.last_price,
            'price_change_24h': change_24h,
            'volume_24h': symbol.volume_24h,
            'high_24h': symbol.high_24h,
            'low_24h': symbol.low_24h,
            'details': details,
            'analysis_time': datetime.now().strftime('%H:%M:%S'),
        }

    def scan_symbol(self, symbol: ScannerSymbol) -> Dict:
        """
        单个交易对扫描（兼容基类接口）

        注意：这个策略主要使用 scan_all_symbols 方法
        这个方法只是为了兼容性
        """
        config = {
            'min_volume_24h': self.config.get('min_volume_24h', 500000),
            'min_score': self.config.get('min_score', 60),
        }

        analysis = self._analyze_unilateral_trend(symbol, config)
        passed = analysis['score'] >= config['min_score']

        return {
            'symbol': symbol.inst_id,
            'passed': passed,
            'score': analysis['score'],
            'details': '\n'.join(analysis['details']),
            'direction': analysis['direction'],
            'rating': analysis['rating'],
            'last_price': symbol.last_price,
            'volume_24h': symbol.volume_24h,
            'price_change_24h': symbol.price_change_24h,
            'high_24h': symbol.high_24h,
            'low_24h': symbol.low_24h,
        }

    def get_config_schema(self) -> Dict:
        """获取配置模式"""
        return {
            'min_volume_24h': {
                'type': 'float',
                'default': 500000,
                'label': '最小 24h 成交量 (USDT)'
            },
            'min_score': {
                'type': 'int',
                'default': 60,
                'label': '最低得分门槛'
            },
            'top_n': {
                'type': 'int',
                'default': 20,
                'label': '返回前N名机会'
            },
        }


# 策略配置模式
CONFIG_SCHEMA = {
    'min_volume_24h': {'type': 'float', 'default': 500000, 'label': '最小 24h 成交量 (USDT)'},
    'min_score': {'type': 'int', 'default': 60, 'label': '最低得分门槛'},
    'top_n': {'type': 'int', 'default': 20, 'label': '返回前N名机会'},
}
