"""
成交量异动扫描策略 (优化版)
筛选出近 2 小时内成交量异常放大的合约交易对

作者：Crypto Trader
版本：2.0 优化版
"""

from src.scanner.base_scanner import BaseScannerStrategy, ScannerSymbol
from typing import Dict, List, Tuple
import pandas as pd


class VolumeSpikeScanner(BaseScannerStrategy):
    """
    成交量异动扫描器 (优化版)
    
    扫描逻辑:
    1. 使用 1H K 线数据，计算最近 2 小时的成交量。
    2. 计算成交量异动比率 = (近 2H 成交量) / (过去 24H 平均每小时成交量)。
    3. 过滤：异动比率需超过阈值（默认 3 倍）。
    4. 过滤：绝对成交量需超过最小值（确保有流动性）。
    5. 过滤：价格涨幅在合理区间（捕捉启动点，避免接盘）。
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
        }

        try:
            # 获取 1H K 线数据
            klines_1h = symbol.extra_data.get('klines', {}).get('1H', [])
            if not klines_1h or len(klines_1h) < 26:
                result['details']['状态'] = "1H 数据不足"
                return result

            # 解析 K 线
            # OKX 1H K 线字段: ts, o, h, l, c, vol, ...
            df = pd.DataFrame(klines_1h, columns=['ts', 'o', 'h', 'l', 'c', 'vol', 'volCcy', 'volCcyQuote', 'confirm'])
            df['vol'] = df['vol'].astype(float)
            df['c'] = df['c'].astype(float)
            
            # 1. 计算近 2H 成交量 (最后 2 根 K 线)
            vol_last_2h = df['vol'].iloc[-2:].sum()
            
            # 2. 计算过去 24H 平均每小时成交量 (过去 24 根 K 线，含当前)
            vol_avg_24h = df['vol'].iloc[-24:].mean()

            if vol_avg_24h == 0:
                return result

            # 3. 计算异动比率 (Spike Ratio)
            spike_ratio = vol_last_2h / (vol_avg_24h * 2) # 对比 2 小时的平均量
            # 解释：vol_avg_24h 是单小时平均，乘以 2 代表 2 小时的基准量
            
            result['details']['异动比率'] = f"{spike_ratio:.2f}x"
            result['details']['近 2H 量'] = f"{vol_last_2h/1000000:.2f}M"

            # --- 条件 1: 异动比率 ---
            spike_threshold = self.config.get('spike_ratio_threshold', 2.5)
            if spike_ratio >= spike_threshold:
                result['conditions_met'] += 1
                result['details']['比率状态'] = f"达标 (>{spike_threshold}x)"
            else:
                result['details']['比率状态'] = f"未达标 (<{spike_threshold}x)"
                return result

            # --- 条件 2: 绝对流动性过滤 ---
            min_abs_volume = self.config.get('min_abs_volume_2h', 500000) # 默认 50 万 USDT
            if vol_last_2h >= min_abs_volume:
                result['conditions_met'] += 1
                result['details']['流动性'] = "充足"
            else:
                result['details']['流动性'] = "不足"
                return result

            # --- 条件 3: 价格行为过滤 ---
            # 检查近 2H 的涨幅，确保是启动而不是已经飞完了
            price_start = df['c'].iloc[-3] # 2 小时前的收盘价
            price_current = df['c'].iloc[-1] # 当前价格
            price_change_2h = ((price_current - price_start) / price_start) * 100
            
            max_price_change = self.config.get('max_price_change_2h', 15.0) # 默认 2 小时内涨幅不超过 15%
            
            result['details']['2H 涨幅'] = f"{price_change_2h:.2f}%"

            # 涨幅为正且未过度透支 (或者允许微跌但在放量)
            # 这里我们设定：如果放量，通常伴随上涨。我们捕捉涨幅在 0% ~ 15% 之间的。
            # 这样可以过滤掉那些已经涨了 50% 的庄股，也能过滤掉无量阴跌的。
            if 0 <= price_change_2h <= max_price_change:
                result['conditions_met'] += 1
                result['details']['涨幅状态'] = "合理启动"
            elif price_change_2h > max_price_change:
                result['details']['涨幅状态'] = "过高/可能见顶"
                return result
            else:
                # 放量下跌？通常不是好的做多异动，但也可能是主力出逃。
                # 这里我们默认过滤掉跌幅过大的。
                if price_change_2h < -5.0:
                    result['details']['涨幅状态'] = "放量下跌"
                    return result
                else:
                    # 微跌但放量，可能是洗盘，算通过
                    result['conditions_met'] += 1
                    result['details']['涨幅状态'] = "洗盘/蓄势"

            # --- 计算得分 ---
            # 基础分 60，每满足一个条件加分，比率越高加分越多
            result['score'] = 60 + (spike_ratio * 5)
            if result['conditions_met'] == 3:
                result['passed'] = True

        except Exception as e:
            result['details']['错误'] = str(e)

        return result

    def get_config_schema(self) -> Dict:
        """获取配置模式"""
        return {
            'spike_ratio_threshold': {
                'type': 'float',
                'default': 2.5,
                'label': '成交量异动倍数阈值 (相对均值)'
            },
            'min_abs_volume_2h': {
                'type': 'float',
                'default': 500000,
                'label': '近 2H 最低成交量 (USDT)'
            },
            'max_price_change_2h': {
                'type': 'float',
                'default': 15.0,
                'label': '近 2H 最大允许涨幅 (%)'
            },
        }


# 策略配置模式
CONFIG_SCHEMA = {
    'spike_ratio_threshold': {'type': 'float', 'default': 2.5, 'label': '成交量异动倍数阈值'},
    'min_abs_volume_2h': {'type': 'float', 'default': 500000, 'label': '近 2H 最低成交量 (USDT)'},
    'max_price_change_2h': {'type': 'float', 'default': 15.0, 'label': '近 2H 最大允许涨幅 (%)'},
}
