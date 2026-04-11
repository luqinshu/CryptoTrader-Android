"""
单边大趋势捕捉扫描策略 - 高级版
利用多时间框架K线数据,精确捕捉短期爆发性行情

作者：Crypto Trader
版本：2.0

核心逻辑:
1. 【短期爆发】5分钟/15分钟K线出现大幅上涨/下跌
2. 【量价齐升】成交量比前N根K线平均值放大N倍
3. 【趋势加速】近期K线涨幅递增(加速上涨)
4. 【突破确认】突破近期高点/低点
5. 【多周期共振】1H/4H/1D 同方向
6. 【连续性】连续多根K线同向(不回头)

适用场景:
- 捕捉像 RAVEUSDT 24h暴涨123% 的行情
- 追涨杀跌、趋势跟随
- 短线快速进出
"""

from src.scanner.base_scanner import (
    BaseScannerStrategy, ScannerSymbol, ScanCondition, ScanResult
)
from typing import Dict, List
from datetime import datetime


class UnilateralTrendScannerAdvanced(BaseScannerStrategy):
    """
    单边大趋势扫描器 - 高级版

    使用多时间框架K线分析,精准捕捉爆发性单边行情
    """

    def _init_conditions(self):
        """初始化扫描条件"""
        self.add_condition(ScanCondition(
            name="最小成交量",
            description="24h 成交量 (USDT)",
            field="volume_24h",
            operator=">",
            value=self.config.get('min_volume_24h', 1000000)
        ))

    def scan_all_symbols(self, symbols_data: List[ScannerSymbol]) -> Dict:
        """
        扫描所有交易对并识别单边趋势 (V2.0 高级增强版)
        """
        config = {
            'min_volume_24h': self.config.get('min_volume_24h', 1000000),
            'min_score': self.config.get('min_score', 65),
            'top_n': self.config.get('top_n', 20),
        }

        filtered_symbols = [s for s in symbols_data if s.volume_24h >= config['min_volume_24h']]

        trend_results = []
        for symbol in filtered_symbols:
            try:
                # 独立分析两个方向，不再依赖24h涨跌幅定死方向
                long_a = self._analyze_trend_by_direction(symbol, "LONG")
                short_a = self._analyze_trend_by_direction(symbol, "SHORT")
                
                analysis = long_a if long_a['score'] >= short_a['score'] else short_a
                
                if analysis['score'] >= config['min_score']:
                    trend_results.append(analysis)
            except Exception:
                continue

        trend_results.sort(key=lambda x: x['score'], reverse=True)

        return {
            'type': 'unilateral_trend_advanced_v2',
            'scan_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_opportunities': len(trend_results),
            'all_opportunities': trend_results[:config['top_n']],
            'note': '高级增强版 - 基于动能加速、突破质量及买卖盘强度分析',
        }

    def _analyze_trend_by_direction(self, symbol: ScannerSymbol, direction: str) -> Dict:
        """多维度分析趋势强度"""
        score = 0
        signals = []
        
        klines_5m = symbol.extra_data.get('klines', {}).get('5m', [])
        klines_15m = symbol.extra_data.get('klines', {}).get('15m', [])
        klines_1h = symbol.extra_data.get('klines', {}).get('1h', [])

        if not klines_5m or not klines_15m:
            return {'score': 0, 'direction': direction}

        # 1. 价格动能与加速度 (35分)
        m_score = 0
        c5m = self._get_change(klines_5m, 5)
        if (direction == "LONG" and c5m >= 1.5) or (direction == "SHORT" and c5m <= -1.5):
            val = min(20, int(abs(c5m) * 6))
            m_score += val
            signals.append(f"🚀 5m动能: {c5m:+.2f}% (+{val})")
            
            # 加速度检测 (最近两根对比)
            body0 = abs(float(klines_5m[0][4]) - float(klines_5m[0][1]))
            body1 = abs(float(klines_5m[1][4]) - float(klines_5m[1][1]))
            if body0 > body1 * 1.8:
                m_score += 15
                signals.append(f"⚡ 动能垂直加速 (+15)")
        score += m_score

        # 2. 突破质量与持续性 (25分)
        b_score = 0
        if len(klines_1h) >= 20:
            highs = [float(k[2]) for k in klines_1h[1:21]]
            lows = [float(k[3]) for k in klines_1h[1:21]]
            curr = symbol.last_price
            
            if direction == "LONG" and curr > max(highs):
                b_score += 25
                signals.append(f"💥 突破H1区间高点 (+25)")
            elif direction == "SHORT" and curr < min(lows):
                b_score += 25
                signals.append(f"📉 跌破H1区间低点 (+25)")
        score += b_score

        # 3. 成交量Delta/买压 (20分)
        v_score = 0
        recent_v = [float(k[5]) for k in klines_5m[:3]]
        avg_v = sum([float(k[5]) for k in klines_5m[3:13]]) / 10
        if recent_v[0] > avg_v * 3:
            v_score += 20
            signals.append(f"🔊 爆量支撑: {(recent_v[0]/avg_v):.1f}x (+20)")
        elif recent_v[0] > avg_v * 1.5:
            v_score += 10
            signals.append(f"🔈 量能放大: {(recent_v[0]/avg_v):.1f}x (+10)")
        score += v_score

        # 4. 多周期对齐 (20分)
        a_score = 0
        c15m = self._get_change(klines_15m, 4)
        c1h = self._get_change(klines_1h, 4)
        
        aligned = 0
        if (direction == "LONG" and c15m > 0 and c1h > 0) or \
           (direction == "SHORT" and c15m < 0 and c1h < 0):
            a_score = 20
            signals.append(f"🎯 三周期趋势完美共振 (+20)")
        elif (direction == "LONG" and c15m > 0) or (direction == "SHORT" and c15m < 0):
            a_score = 10
            signals.append(f"🎯 5m/15m双周期共振 (+10)")
        score += a_score

        # 5. 风险预警：动能衰竭/超买预防 (减分项)
        if score > 50:
            # 如果5分钟K线留下了长影线 (影线 > 实体 * 1.5)
            k0 = klines_5m[0]
            body = abs(float(k0[4]) - float(k0[1]))
            shadow = (float(k0[2]) - float(k0[4])) if direction == "LONG" else (float(k0[4]) - float(k0[3]))
            if shadow > body * 1.5 and body > 0:
                score -= 20
                signals.append(f"⚠️ 警惕：上方抛压/假突破 (影线长)")

        return {
            'symbol': symbol.inst_id,
            'direction': direction,
            'score': score,
            'rating': "🚀超级爆发" if score >= 85 else ("🔥强势" if score >= 70 else "👀关注"),
            'signals': signals,
            'last_price': symbol.last_price,
            'volume_24h': symbol.volume_24h,
            'price_change_24h': symbol.price_change_24h,
        }

    def _get_change(self, klines, n):
        if len(klines) < n: return 0
        start = float(klines[n-1][4])
        end = float(klines[0][4])
        return ((end - start) / start) * 100 if start != 0 else 0

    def _count_consecutive_bars(self, klines: List, max_bars: int = 10) -> int:
        """
        计算连续同向K线数量

        Args:
            klines: K线数据列表
            max_bars: 最多检查的K线数量

        Returns:
            连续同向K线数量
        """
        if len(klines) < 2:
            return 0

        closes = [float(k[4]) for k in klines[:max_bars]]
        opens = [float(k[1]) for k in klines[:max_bars]]

        # 判断第一根K线方向
        if closes[0] >= opens[0]:
            direction = 1  # 阳线
        else:
            direction = -1  # 阴线

        consecutive = 1
        for i in range(1, len(closes)):
            if closes[i] >= opens[i]:
                current_dir = 1
            else:
                current_dir = -1

            if current_dir == direction:
                consecutive += 1
            else:
                break

        return consecutive

    def scan_symbol(self, symbol: ScannerSymbol) -> Dict:
        """
        单个交易对扫描（兼容基类接口）
        """
        config = {
            'min_volume_24h': self.config.get('min_volume_24h', 1000000),
            'min_score': self.config.get('min_score', 65),
        }

        try:
            analysis = self._analyze_unilateral_trend_advanced(symbol, config)
            passed = analysis['score'] >= config['min_score']
        except Exception as e:
            analysis = {
                'score': 0,
                'direction': 'LONG' if symbol.price_change_24h > 0 else 'SHORT',
                'rating': '❌ 分析失败',
                'action': str(e),
                'signals': [],
                'details': [],
            }
            passed = False

        return {
            'symbol': symbol.inst_id,
            'passed': passed,
            'score': analysis.get('score', 0),
            'details': '\n'.join(analysis.get('details', [])),
            'direction': analysis.get('direction', 'LONG'),
            'rating': analysis.get('rating', 'N/A'),
            'action': analysis.get('action', 'N/A'),
            'signals': analysis.get('signals', []),
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
                'default': 1000000,
                'label': '最小 24h 成交量 (USDT)'
            },
            'min_5m_change': {
                'type': 'float',
                'default': 2.0,
                'label': '5分钟最小涨幅 (%)'
            },
            'min_15m_change': {
                'type': 'float',
                'default': 5.0,
                'label': '15分钟最小涨幅 (%)'
            },
            'min_1h_change': {
                'type': 'float',
                'default': 8.0,
                'label': '1小时最小涨幅 (%)'
            },
            'volume_ratio_5m': {
                'type': 'float',
                'default': 3.0,
                'label': '5分钟量比阈值'
            },
            'volume_ratio_15m': {
                'type': 'float',
                'default': 2.5,
                'label': '15分钟量比阈值'
            },
            'consecutive_bars': {
                'type': 'int',
                'default': 3,
                'label': '连续同向K线数'
            },
            'breakout_lookback': {
                'type': 'int',
                'default': 20,
                'label': '突破检查周期数'
            },
            'min_score': {
                'type': 'int',
                'default': 65,
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
    'min_volume_24h': {'type': 'float', 'default': 1000000, 'label': '最小 24h 成交量 (USDT)'},
    'min_score': {'type': 'int', 'default': 65, 'label': '最低得分门槛'},
    'top_n': {'type': 'int', 'default': 20, 'label': '返回前N名机会'},
}
