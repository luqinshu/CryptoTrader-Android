"""
重点监控池引擎 - 实时监测 OKX 交易对趋势突破/回调/企稳信号
"""

import time
import hashlib
import json
import threading
from datetime import datetime
from typing import Dict, List, Optional, Set
from collections import deque
from src.qt_compat import QObject, QThread, Qt, Signal



# ── 信号类型枚举 ──
class SignalType:
    TREND_BREAKOUT = "趋势突破"          # 价格突破关键阻力/支撑 + 放量
    DEEP_PULLBACK = "大幅回调"           # 从近期高点深度回撤但主趋势未破
    STABILIZATION_BREAKOUT = "企稳突破"  # 小时级别缩量横盘后放量突破
    VOLUME_SURGE = "放量异动"            # 成交量突然异常放大
    MOMENTUM_DIVERGENCE = "动量背离"     # RSI/价格背离


# ── 信号数据结构 ──
class MonitorSignal:
    """监控信号"""
    __slots__ = (
        'inst_id', 'signal_type', 'direction', 'score', 'message',
        'price', 'timestamp', 'details', 'signal_id'
    )

    def __init__(self, inst_id: str, signal_type: str, direction: str,
                 score: float, message: str, price: float = 0.0,
                 details: Dict = None):
        self.inst_id = inst_id
        self.signal_type = signal_type
        self.direction = direction  # 'BUY' / 'SHORT' / 'NEUTRAL'
        self.score = score
        self.message = message
        self.price = price
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.details = details or {}
        self.signal_id = self._generate_id()

    def _generate_id(self) -> str:
        # 使用6分钟滑动窗口（而非5分钟硬桶），避免边界附近的漏去重
        bucket = int(time.time() // 360)  # 6分钟窗口
        raw = f"{self.inst_id}:{self.signal_type}:{self.direction}:{bucket}"
        return hashlib.sha1(raw.encode()).hexdigest()[:12]

    def to_dict(self) -> Dict:
        return {
            'inst_id': self.inst_id,
            'signal_type': self.signal_type,
            'direction': self.direction,
            'score': round(self.score, 1),
            'message': self.message,
            'price': self.price,
            'timestamp': self.timestamp,
            'signal_id': self.signal_id,
            'details': self.details,
        }


# ── K线数据缓存 ──
class KlineCache:
    """K线数据环形缓存（含过期时间）"""
    __slots__ = ('bar', 'max_len', 'data', 'last_fetch_ts', 'cache_ttl_sec')

    def __init__(self, bar: str, max_len: int = 200, ttl_sec: int = 120):
        self.bar = bar
        self.max_len = max_len
        self.data: deque = deque(maxlen=max_len)
        self.last_fetch_ts = 0.0
        self.cache_ttl_sec = ttl_sec

    def is_expired(self) -> bool:
        """缓存是否过期（超过TTL未刷新）"""
        return time.time() - self.last_fetch_ts > self.cache_ttl_sec

    def add(self, candles: List[List]):
        """添加K线数据（倒序，最新在前）"""
        self.last_fetch_ts = time.time()
        for row in reversed(candles):
            if len(row) < 6:
                continue
            ts = float(row[0]) if row[0] else 0
            # 避免重复时间戳
            if self.data and abs(self.data[-1][0] - ts) < 1:
                continue
            self.data.append((
                ts,
                float(row[1] or 0),  # open
                float(row[2] or 0),  # high
                float(row[3] or 0),  # low
                float(row[4] or 0),  # close
                float(row[5] or 0),  # vol
            ))

    def closes(self) -> List[float]:
        return [c[4] for c in self.data]

    def highs(self) -> List[float]:
        return [c[2] for c in self.data]

    def lows(self) -> List[float]:
        return [c[3] for c in self.data]

    def volumes(self) -> List[float]:
        return [c[5] for c in self.data]

    def last(self):
        return self.data[-1] if self.data else None

    def __len__(self):
        return len(self.data)


# ── 监控引擎 ──
class MonitorEngine(QObject):
    """重点监控池引擎 - 实时监测并发出信号"""

    # UI 信号
    signal_detected = Signal(object)       # MonitorSignal
    status_update = Signal(str, str)       # (inst_id, status_text)
    log_message = Signal(str, str)         # (message, level)
    pair_data_updated = Signal(str, dict)  # (inst_id, data_dict)

    # ── 高质量信号桥接信号（评分 ≥ high_signal_threshold 时发射）────────────
    # 接收方：ScanDrivenAutoTrader.on_monitor_signal(sig_dict)
    # sig_dict 格式与 on_scan_results 兼容：
    #   { inst_id, symbol, direction, score, reason, entry_price, source }
    high_quality_signal = Signal(dict)

    def __init__(self, okx_client):
        super().__init__()
        self.okx_client = okx_client
        self._stop_flag = False
        self._pairs_lock = threading.Lock()
        self._monitored_pairs: Dict[str, Dict] = {}  # inst_id -> config (protected by _pairs_lock)
        self._klines: Dict[str, Dict[str, KlineCache]] = {}  # inst_id -> bar -> cache
        self._recent_signals: deque = deque(maxlen=200)  # 最近信号防重
        self._signal_ids: Set[str] = set()
        self._last_check: Dict[str, float] = {}  # inst_id -> last check timestamp

        # 默认监测参数
        self.config = {
            'check_interval_sec': 60,         # 检查间隔
            'trend_breakout_pct': 2.5,         # 突破幅度阈值 %
            'trend_breakout_volume_ratio': 1.5, # 突破放量倍数
            'deep_pullback_pct': 5.0,          # 深度回调阈值 %
            'deep_pullback_lookback_bars': 24,  # 回调回看K线数
            'stabilization_max_range_pct': 1.2, # 企稳最大振幅 %
            'stabilization_min_bars': 6,        # 企稳最少横盘K线数
            'stabilization_breakout_volume_ratio': 1.8,  # 企稳突破放量
            'volume_surge_ratio': 3.0,          # 放量异动倍数
            'rsi_oversold': 30,                 # RSI超卖
            'rsi_overbought': 70,               # RSI超买
            'rsi_divergence_lookback': 20,      # RSI背离回看
            'min_signal_score': 60,             # 最低信号分
            'signal_cooldown_min': 15,          # 同品种同类型信号冷却(分钟)
            'required_bars': ['1D', '1H', '4H', '15m'],
            'high_signal_threshold': 80,        # 触发 high_quality_signal 的最低分（桥接自动交易）
        }

    # ── 交易对管理 ──
    def add_pair(self, inst_id: str, config: Dict = None):
        """添加监测交易对"""
        inst_id = str(inst_id).upper().strip()
        if not inst_id:
            return
        with self._pairs_lock:
            if inst_id in self._monitored_pairs:
                self.log_message.emit(f"{inst_id} 已在监控池中", "WARNING")
                return
            self._monitored_pairs[inst_id] = config or {}
            self._klines[inst_id] = {}
            self._last_check[inst_id] = 0.0
        self.log_message.emit(f"✅ {inst_id} 已加入重点监控池", "SUCCESS")

    def remove_pair(self, inst_id: str):
        """移除监测交易对"""
        inst_id = str(inst_id).upper().strip()
        with self._pairs_lock:
            if inst_id in self._monitored_pairs:
                del self._monitored_pairs[inst_id]
                self._klines.pop(inst_id, None)
                self._last_check.pop(inst_id, None)
                self.log_message.emit(f"🗑️ {inst_id} 已从监控池移除", "INFO")

    def set_pairs(self, pairs: List[str]):
        """批量设置监控交易对"""
        with self._pairs_lock:
            current = set(self._monitored_pairs.keys())
        new = {p.upper().strip() for p in pairs if p.strip()}
        # 移除旧交易对
        for inst_id in current - new:
            self.remove_pair(inst_id)
        # 添加新交易对
        for inst_id in new - current:
            self.add_pair(inst_id)

    def get_pairs(self) -> List[str]:
        with self._pairs_lock:
            return sorted(self._monitored_pairs.keys())

    def get_pair_config(self, inst_id: str) -> Dict:
        with self._pairs_lock:
            return self._monitored_pairs.get(inst_id.upper(), {})

    def update_config(self, updates: Dict):
        """更新监控参数"""
        for k, v in updates.items():
            if k in self.config:
                self.config[k] = v

    # ── K线数据获取 ──
    def _fetch_klines(self, inst_id: str, bar: str, limit: int = 80) -> List:
        """获取K线数据"""
        try:
            result = self.okx_client.get_kline(inst_id, bar=bar, limit=limit)
            if result and result.get('code') == '0' and result.get('data'):
                return result['data']
        except Exception as e:
            self.log_message.emit(f"{inst_id} {bar} K线获取失败: {e}", "WARNING")
        return []

    def _ensure_klines(self, inst_id: str):
        """确保K线缓存已填充（过期自动刷新）"""
        for bar in self.config['required_bars']:
            if bar not in self._klines.get(inst_id, {}):
                self._klines.setdefault(inst_id, {})[bar] = KlineCache(bar)

            cache = self._klines[inst_id][bar]
            # 缓存未过期或已有足够数据则跳过
            if not cache.is_expired() and len(cache) > 0:
                continue
            candles = self._fetch_klines(inst_id, bar, limit=80)
            if candles:
                cache.add(candles)

    # ── 技术指标计算 ──
    @staticmethod
    def _calc_rsi(closes: List[float], period: int = 14) -> float:
        """Wilder EMA 方法 RSI（标准实现，比简单均值更准确）"""
        if len(closes) < period + 1:
            return 50.0
        deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]
        gains = [max(d, 0.0) for d in deltas]
        losses = [abs(min(d, 0.0)) for d in deltas]
        # 初始平均
        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period
        # Wilder EMA 平滑（后续数据）
        for i in range(period, len(gains)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            return 100.0 if avg_gain > 0 else 50.0
        return 100 - (100 / (1 + avg_gain / avg_loss))

    @staticmethod
    def _calc_ema(closes: List[float], period: int) -> float:
        if len(closes) < period:
            return sum(closes) / len(closes) if closes else 0.0
        multiplier = 2 / (period + 1)
        ema = sum(closes[:period]) / period
        for price in closes[period:]:
            ema = (price - ema) * multiplier + ema
        return ema

    @staticmethod
    def _calc_ema_series(closes: List[float], period: int) -> List[float]:
        """返回完整 EMA 序列"""
        if not closes:
            return []
        k = 2 / (period + 1)
        if len(closes) < period:
            avg = sum(closes) / len(closes)
            return [avg] * len(closes)
        result = [sum(closes[:period]) / period]
        for v in closes[period:]:
            result.append(v * k + result[-1] * (1 - k))
        return [result[0]] * (period - 1) + result

    @staticmethod
    def _calc_macd_full(closes: List[float]) -> dict:
        """
        标准 MACD(12,26,9) 完整计算。
        返回 {macd, signal, hist, cross('golden'/'death'/'none'), hist_growing, above_zero}
        """
        if len(closes) < 35:
            return {'macd': 0.0, 'signal': 0.0, 'hist': 0.0,
                    'cross': 'none', 'hist_growing': False, 'above_zero': False}
        k12, k26, k9 = 2/13, 2/27, 2/10
        # EMA12 系列
        ema12 = [sum(closes[:12]) / 12]
        for v in closes[12:]:
            ema12.append(v * k12 + ema12[-1] * (1 - k12))
        # EMA26 系列
        ema26 = [sum(closes[:26]) / 26]
        for v in closes[26:]:
            ema26.append(v * k26 + ema26[-1] * (1 - k26))
        # MACD 线（对齐：ema12 第14根起 vs ema26 第0根起）
        offset = 26 - 12
        macd_line = [ema12[i + offset] - ema26[i] for i in range(len(ema26))]
        if len(macd_line) < 9:
            return {'macd': macd_line[-1], 'signal': 0.0, 'hist': 0.0,
                    'cross': 'none', 'hist_growing': False, 'above_zero': macd_line[-1] > 0}
        # 信号线 = MACD 的 EMA9
        sig = [sum(macd_line[:9]) / 9]
        for v in macd_line[9:]:
            sig.append(v * k9 + sig[-1] * (1 - k9))
        macd_val, sig_val = macd_line[-1], sig[-1]
        hist = macd_val - sig_val
        prev_hist = macd_line[-2] - sig[-2] if len(macd_line) >= 2 and len(sig) >= 2 else hist
        cross = 'none'
        if prev_hist <= 0 and hist > 0:
            cross = 'golden'    # 金叉
        elif prev_hist >= 0 and hist < 0:
            cross = 'death'     # 死叉
        return {
            'macd': round(macd_val, 8),
            'signal': round(sig_val, 8),
            'hist': round(hist, 8),
            'cross': cross,
            'hist_growing': abs(hist) > abs(prev_hist),   # 柱子是否扩张（动量增强）
            'above_zero': macd_val > 0,                   # MACD 线在零轴上方
        }

    @staticmethod
    def _calc_bb_position(closes: List[float], period: int = 20, mult: float = 2.0) -> dict:
        """
        布林带位置分析。
        返回 {bandwidth(%), position(0~1), squeeze, above_upper, below_lower}
        position: 0=处于下轨, 1=处于上轨, 0.5=中轨
        squeeze: 带宽收窄（volatility squeeze，通常是大行情前兆）
        """
        if len(closes) < period:
            return {'bandwidth': 0.0, 'position': 0.5, 'squeeze': False,
                    'above_upper': False, 'below_lower': False, 'upper': 0.0, 'lower': 0.0}
        recent = closes[-period:]
        mid = sum(recent) / period
        std = (sum((x - mid)**2 for x in recent) / period) ** 0.5
        upper = mid + mult * std
        lower = mid - mult * std
        last = closes[-1]
        bw = (upper - lower) / mid * 100 if mid > 0 else 0.0
        pos = (last - lower) / (upper - lower) if (upper - lower) > 0 else 0.5
        return {
            'upper': round(upper, 8),
            'middle': round(mid, 8),
            'lower': round(lower, 8),
            'bandwidth': round(bw, 2),
            'position': round(pos, 3),
            'squeeze': bw < 3.5,       # 带宽 < 3.5% 视为收窄蓄势
            'above_upper': last > upper,
            'below_lower': last < lower,
        }

    @staticmethod
    def _calc_atr(highs: List[float], lows: List[float], closes: List[float],
                  period: int = 14) -> float:
        n = min(len(highs), len(lows), len(closes))
        if n < period + 1:
            return 0.0
        tr_values = []
        for i in range(1, n):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1])
            )
            tr_values.append(tr)
        return sum(tr_values[-period:]) / period

    def _d1_trend_label(self, inst_id: str) -> str:
        """
        从日线K线缓存计算趋势标签。
        返回 'bull'（多头）/ 'bear'（空头）/ 'sideways'（横盘）/ 'unknown'（数据不足）
        判断条件：日线 EMA20 vs EMA50 位置 + 收盘价位置
        """
        d1 = self._klines.get(inst_id, {}).get('1D')
        if not d1 or len(d1) < 50:
            return 'unknown'
        closes = d1.closes()
        ema20 = self._calc_ema(closes, 20)
        ema50 = self._calc_ema(closes, 50)
        last = closes[-1]
        tol = 0.01   # 1% 容差防止临界频繁翻转
        if ema20 > ema50 * (1 + tol) and last > ema50:
            return 'bull'
        if ema20 < ema50 * (1 - tol) and last < ema50:
            return 'bear'
        return 'sideways'

    def _change_pct(self, old: float, new: float) -> float:
        return (new / old - 1.0) * 100.0 if old > 0 else 0.0

    # ── 信号检测 ──
    def _check_trend_breakout(self, inst_id: str, h1: KlineCache,
                              h4: KlineCache) -> List[MonitorSignal]:
        """检测趋势突破信号"""
        signals = []
        if len(h1) < 40 or len(h4) < 20:
            return signals

        closes_h1 = h1.closes()
        closes_h4 = h4.closes()
        volumes_h1 = h1.volumes()
        highs_h4 = h4.highs()
        lows_h4 = h4.lows()

        cfg = self.config
        last_close = closes_h1[-1]
        last_vol = volumes_h1[-1]

        # 计算4H级别阻力/支撑（近20根高/低点）
        h4_resistance = max(highs_h4[-20:])
        h4_support = min(lows_h4[-20:])
        h4_range = h4_resistance - h4_support

        if h4_range <= 0:
            return signals

        # 计算均量
        avg_vol = sum(volumes_h1[-21:-1]) / 20 if len(volumes_h1) > 20 else last_vol
        if avg_vol <= 0:
            return signals
        vol_ratio = last_vol / avg_vol

        # 判断方向
        ema_20 = self._calc_ema(closes_h1, 20)
        price_vs_ema = self._change_pct(ema_20, last_close)

        # 向上突破
        breakout_up = (last_close > h4_resistance * (1 - cfg['trend_breakout_pct'] / 200))
        if breakout_up and vol_ratio >= cfg['trend_breakout_volume_ratio'] and price_vs_ema > 0:
            score = min(100, 60 + vol_ratio * 10 + abs(price_vs_ema) * 3)
            signals.append(MonitorSignal(
                inst_id=inst_id, signal_type=SignalType.TREND_BREAKOUT,
                direction='BUY', score=score, price=last_close,
                message=f"向上突破4H阻力 {h4_resistance:.4f} | 放量{vol_ratio:.1f}x | EMA上方{price_vs_ema:.1f}%",
                details={
                    'resistance': h4_resistance, 'support': h4_support,
                    'volume_ratio': round(vol_ratio, 2),
                    'price_vs_ema_pct': round(price_vs_ema, 2),
                    'h4_range': round(h4_range, 4),
                }
            ))

        # 向下突破
        breakout_down = (last_close < h4_support * (1 + cfg['trend_breakout_pct'] / 200))
        if breakout_down and vol_ratio >= cfg['trend_breakout_volume_ratio'] and price_vs_ema < 0:
            score = min(100, 60 + vol_ratio * 10 + abs(price_vs_ema) * 3)
            signals.append(MonitorSignal(
                inst_id=inst_id, signal_type=SignalType.TREND_BREAKOUT,
                direction='SHORT', score=score, price=last_close,
                message=f"向下跌破4H支撑 {h4_support:.4f} | 放量{vol_ratio:.1f}x | EMA下方{abs(price_vs_ema):.1f}%",
                details={
                    'resistance': h4_resistance, 'support': h4_support,
                    'volume_ratio': round(vol_ratio, 2),
                    'price_vs_ema_pct': round(price_vs_ema, 2),
                    'h4_range': round(h4_range, 4),
                }
            ))

        return signals

    def _check_deep_pullback(self, inst_id: str, h1: KlineCache,
                             h4: KlineCache) -> List[MonitorSignal]:
        """检测大幅回调信号"""
        signals = []
        if len(h1) < 30 or len(h4) < 20:
            return signals

        closes_h1 = h1.closes()
        closes_h4 = h4.closes()
        highs_h1 = h1.highs()
        cfg = self.config

        last_close = closes_h1[-1]
        lookback = cfg['deep_pullback_lookback_bars']

        # 找近期高点
        recent_high = max(highs_h1[-lookback:])
        drawdown_pct = abs(self._change_pct(recent_high, last_close))

        if drawdown_pct < cfg['deep_pullback_pct']:
            return signals

        # 判断主趋势是否仍完好（4H级别EMA）
        ema_20_h4 = self._calc_ema(closes_h4, 20)
        ema_50_h4 = self._calc_ema(closes_h4, 50)

        # 牛市回调（主趋势向上，当前回调）
        if ema_20_h4 > ema_50_h4 and last_close > ema_50_h4:
            rsi = self._calc_rsi(closes_h1, 14)
            if rsi < cfg['rsi_oversold'] + 10:  # RSI <= 40 超卖区域
                score = min(100, 55 + drawdown_pct * 4 + (50 - rsi) * 0.5)
                signals.append(MonitorSignal(
                    inst_id=inst_id, signal_type=SignalType.DEEP_PULLBACK,
                    direction='BUY', score=score, price=last_close,
                    message=f"牛市深度回调 {drawdown_pct:.1f}% | RSI={rsi:.0f} | 4H趋势仍向上",
                    details={
                        'drawdown_pct': round(drawdown_pct, 2),
                        'recent_high': recent_high,
                        'rsi': round(rsi, 1),
                        'ema_20': round(ema_20_h4, 4),
                        'ema_50': round(ema_50_h4, 4),
                    }
                ))

        # 熊市反弹（主趋势向下，当前反弹）
        if ema_20_h4 < ema_50_h4 and last_close < ema_50_h4:
            rsi = self._calc_rsi(closes_h1, 14)
            if rsi > cfg['rsi_overbought'] - 10:  # RSI >= 60
                score = min(100, 55 + drawdown_pct * 4 + (rsi - 50) * 0.5)
                signals.append(MonitorSignal(
                    inst_id=inst_id, signal_type=SignalType.DEEP_PULLBACK,
                    direction='SHORT', score=score, price=last_close,
                    message=f"熊市反弹 {drawdown_pct:.1f}% | RSI={rsi:.0f} | 4H趋势仍向下",
                    details={
                        'drawdown_pct': round(drawdown_pct, 2),
                        'recent_high': recent_high,
                        'rsi': round(rsi, 1),
                        'ema_20': round(ema_20_h4, 4),
                        'ema_50': round(ema_50_h4, 4),
                    }
                ))

        return signals

    def _check_stabilization_breakout(self, inst_id: str, h1: KlineCache,
                                      m15: KlineCache) -> List[MonitorSignal]:
        """检测企稳后突破信号"""
        signals = []
        if len(h1) < 30 or len(m15) < 20:
            return signals

        cfg = self.config
        closes_h1 = h1.closes()
        highs_h1 = h1.highs()
        lows_h1 = h1.lows()
        volumes_h1 = h1.volumes()
        closes_m15 = m15.closes()
        volumes_m15 = m15.volumes()

        min_bars = cfg['stabilization_min_bars']
        max_range_pct = cfg['stabilization_max_range_pct']

        # 检查最近N根H1是否横盘企稳
        if len(closes_h1) < min_bars + 2:
            return signals

        recent_closes = closes_h1[-min_bars:]
        recent_high = max(recent_closes)
        recent_low = min(recent_closes)
        consolidation_range = self._change_pct(recent_low, recent_high)

        if consolidation_range > max_range_pct:
            return signals  # 振幅过大，非企稳状态

        # 确认前面有趋势（波动大于横盘）
        prior_closes = closes_h1[-min_bars * 3:-min_bars]
        if len(prior_closes) < min_bars:
            return signals
        prior_range = self._change_pct(min(prior_closes), max(prior_closes))
        if prior_range <= consolidation_range * 1.5:
            return signals  # 之前波动不足

        # 检查是否出现突破
        last_close = closes_h1[-1]
        breakout_high = max(recent_closes[:-1])  # 除最新外的最高
        breakout_low = min(recent_closes[:-1])   # 除最新外的最低

        avg_vol_h1 = sum(volumes_h1[-min_bars * 2:-1]) / (min_bars * 2 - 1) if len(volumes_h1) > min_bars * 2 else volumes_h1[-1]
        last_vol_h1 = volumes_h1[-1]
        vol_ratio_h1 = last_vol_h1 / avg_vol_h1 if avg_vol_h1 > 0 else 1.0

        # 也检查15m级别的放量
        if len(volumes_m15) >= 5:
            avg_vol_m15 = sum(volumes_m15[-6:-1]) / 5
            last_vol_m15 = volumes_m15[-1]
            vol_ratio_m15 = last_vol_m15 / avg_vol_m15 if avg_vol_m15 > 0 else 1.0
        else:
            vol_ratio_m15 = 1.0

        vol_boost = max(vol_ratio_h1, vol_ratio_m15)
        vol_threshold = cfg['stabilization_breakout_volume_ratio']

        # 向上突破
        if last_close > breakout_high and vol_boost >= vol_threshold:
            score = min(100, 55 + vol_boost * 10 + abs(self._change_pct(breakout_high, last_close)) * 5)
            signals.append(MonitorSignal(
                inst_id=inst_id, signal_type=SignalType.STABILIZATION_BREAKOUT,
                direction='BUY', score=score, price=last_close,
                message=f"H1企稳后向上突破 | 横盘振幅{consolidation_range:.1f}% | 放量{vol_boost:.1f}x",
                details={
                    'consolidation_range_pct': round(consolidation_range, 2),
                    'prior_range_pct': round(prior_range, 2),
                    'breakout_level': breakout_high,
                    'volume_ratio': round(vol_boost, 2),
                }
            ))

        # 向下突破
        if last_close < breakout_low and vol_boost >= vol_threshold:
            score = min(100, 55 + vol_boost * 10 + abs(self._change_pct(breakout_low, last_close)) * 5)
            signals.append(MonitorSignal(
                inst_id=inst_id, signal_type=SignalType.STABILIZATION_BREAKOUT,
                direction='SHORT', score=score, price=last_close,
                message=f"H1企稳后向下突破 | 横盘振幅{consolidation_range:.1f}% | 放量{vol_boost:.1f}x",
                details={
                    'consolidation_range_pct': round(consolidation_range, 2),
                    'prior_range_pct': round(prior_range, 2),
                    'breakout_level': breakout_low,
                    'volume_ratio': round(vol_boost, 2),
                }
            ))

        return signals

    def _check_volume_surge(self, inst_id: str, h1: KlineCache) -> List[MonitorSignal]:
        """检测成交量异动"""
        signals = []
        if len(h1) < 30:
            return signals

        volumes = h1.volumes()
        closes = h1.closes()
        last_vol = volumes[-1]
        last_close = closes[-1]

        # 近20根均值（排除最新）
        avg_vol = sum(volumes[-21:-1]) / 20 if len(volumes) > 20 else last_vol
        if avg_vol <= 0:
            return signals
        vol_ratio = last_vol / avg_vol

        if vol_ratio >= self.config['volume_surge_ratio']:
            # 判断涨跌
            if len(closes) >= 2:
                change = self._change_pct(closes[-2], last_close)
                direction = 'BUY' if change > 0 else 'SHORT'
            else:
                direction = 'NEUTRAL'
                change = 0.0

            score = min(100, 50 + vol_ratio * 8)
            signals.append(MonitorSignal(
                inst_id=inst_id, signal_type=SignalType.VOLUME_SURGE,
                direction=direction, score=score, price=last_close,
                message=f"成交量异常放大 {vol_ratio:.1f}x | 涨跌{change:+.2f}%",
                details={
                    'volume_ratio': round(vol_ratio, 2),
                    'avg_volume': round(avg_vol, 0),
                    'last_volume': round(last_vol, 0),
                    'price_change_pct': round(change, 2),
                }
            ))
        return signals

    def _check_momentum_divergence(self, inst_id: str, h1: KlineCache) -> List[MonitorSignal]:
        """检测RSI动量背离"""
        signals = []
        if len(h1) < 30:
            return signals

        closes = h1.closes()
        lookback = self.config['rsi_divergence_lookback']
        rsi_period = 14

        # 需要至少 lookback + rsi_period 根K线
        if len(closes) < lookback + rsi_period:
            return signals

        # 计算当前和之前RSI
        current_rsi = self._calc_rsi(closes, rsi_period)
        prior_closes = closes[:-lookback]
        prior_rsi = self._calc_rsi(prior_closes, rsi_period)

        current_high = max(closes[-lookback:])
        prior_high = max(prior_closes[-lookback:])
        current_low = min(closes[-lookback:])
        prior_low = min(prior_closes[-lookback:])

        # 顶背离：价格新高但RSI下降
        if current_high > prior_high and current_rsi < prior_rsi - 5:
            score = min(100, 60 + (prior_rsi - current_rsi) * 2)
            signals.append(MonitorSignal(
                inst_id=inst_id, signal_type=SignalType.MOMENTUM_DIVERGENCE,
                direction='SHORT', score=score, price=closes[-1],
                message=f"顶背离 | 价格新高但RSI {prior_rsi:.0f}→{current_rsi:.0f}",
                details={
                    'prior_rsi': round(prior_rsi, 1),
                    'current_rsi': round(current_rsi, 1),
                    'prior_high': prior_high,
                    'current_high': current_high,
                }
            ))

        # 底背离：价格新低但RSI上升
        if current_low < prior_low and current_rsi > prior_rsi + 5:
            score = min(100, 60 + (current_rsi - prior_rsi) * 2)
            signals.append(MonitorSignal(
                inst_id=inst_id, signal_type=SignalType.MOMENTUM_DIVERGENCE,
                direction='BUY', score=score, price=closes[-1],
                message=f"底背离 | 价格新低但RSI {prior_rsi:.0f}→{current_rsi:.0f}",
                details={
                    'prior_rsi': round(prior_rsi, 1),
                    'current_rsi': round(current_rsi, 1),
                    'prior_low': prior_low,
                    'current_low': current_low,
                }
            ))

        return signals

    # ── 主检测循环 ──
    def _deduplicate_signal(self, signal: MonitorSignal) -> bool:
        """信号去重 - 同品种同类型在冷却期内不重复"""
        now = time.time()
        cooldown = self.config['signal_cooldown_min'] * 60

        # 清理过期信号ID
        expire = now - cooldown * 2
        to_remove = []
        for s in self._recent_signals:
            try:
                sts = datetime.strptime(s.timestamp, "%Y-%m-%d %H:%M:%S").timestamp()
                if sts < expire:
                    to_remove.append(s)
            except:
                to_remove.append(s)
        for s in to_remove:
            if s in self._recent_signals:
                self._recent_signals.remove(s)
            self._signal_ids.discard(s.signal_id)

        # 检查是否重复
        if signal.signal_id in self._signal_ids:
            return True

        for s in self._recent_signals:
            if (s.inst_id == signal.inst_id
                    and s.signal_type == signal.signal_type
                    and s.direction == signal.direction):
                try:
                    sts = datetime.strptime(s.timestamp, "%Y-%m-%d %H:%M:%S").timestamp()
                    if now - sts < cooldown:
                        return True
                except:
                    pass

        self._recent_signals.append(signal)
        self._signal_ids.add(signal.signal_id)
        return False

    def check_all(self) -> List[MonitorSignal]:
        """检查所有监控交易对，返回检测到的信号列表"""
        all_signals = []
        now = time.time()
        min_interval = max(15, self.config['check_interval_sec'])

        with self._pairs_lock:
            pairs_snapshot = list(self._monitored_pairs.keys())

        for inst_id in pairs_snapshot:
            # 检查间隔
            last_check = self._last_check.get(inst_id, 0)
            if now - last_check < min_interval:
                continue
            self._last_check[inst_id] = now

            try:
                # 获取K线
                self._ensure_klines(inst_id)
                klines = self._klines.get(inst_id, {})

                h1 = klines.get('1H')
                h4 = klines.get('4H')
                m15 = klines.get('15m')

                if not h1 or len(h1) < 20:
                    self.status_update.emit(inst_id, "数据不足")
                    continue

                # 更新最新一根K线
                latest_candles = self._fetch_klines(inst_id, '1H', limit=2)
                if latest_candles:
                    h1.add(latest_candles)

                # 运行各检测器
                signals = []
                if h4 and len(h4) >= 10:
                    signals.extend(self._check_trend_breakout(inst_id, h1, h4))
                    signals.extend(self._check_deep_pullback(inst_id, h1, h4))

                if m15 and len(m15) >= 10:
                    signals.extend(self._check_stabilization_breakout(inst_id, h1, m15))

                signals.extend(self._check_volume_surge(inst_id, h1))
                signals.extend(self._check_momentum_divergence(inst_id, h1))

                # 过滤低分 + 去重
                min_score = self.config['min_signal_score']
                high_threshold = float(self.config.get('high_signal_threshold', 80))
                for sig in signals:
                    if sig.score >= min_score and not self._deduplicate_signal(sig):
                        all_signals.append(sig)
                        self.signal_detected.emit(sig)
                        # ── 高质量信号桥接：评分 ≥ 阈值时发射，供自动交易系统接收 ──
                        if sig.score >= high_threshold and sig.direction in ('BUY', 'SHORT'):
                            self.high_quality_signal.emit({
                                'inst_id':          sig.inst_id,
                                'symbol':           sig.inst_id,
                                'direction':        sig.direction,
                                'score':            sig.score,
                                'opportunity_score': sig.score,
                                'reason':           f"[监控池] {sig.signal_type}: {sig.message}",
                                'entry_price':      sig.price,
                                'source':           'monitor_pool',
                                'timestamp':        sig.timestamp,
                            })

                # 更新状态 + 发送完整多维指标
                last = h1.last()
                if last:
                    price = last[4]
                    closes_h1 = h1.closes()
                    vols_h1 = h1.volumes()

                    rsi = self._calc_rsi(closes_h1, 14)
                    vols_avg = sum(vols_h1[-21:-1]) / 20 if len(vols_h1) > 20 and sum(vols_h1[-21:-1]) > 0 else 1.0
                    vol_ratio = round(vols_h1[-1] / vols_avg, 2) if vols_avg > 0 else 1.0

                    # ── 新增指标 ──
                    macd_data = self._calc_macd_full(closes_h1)
                    bb_data   = self._calc_bb_position(closes_h1, period=20)
                    d1_trend  = self._d1_trend_label(inst_id)

                    # MACD 状态文字
                    macd_sym = ('⚡金叉' if macd_data['cross'] == 'golden' else
                                '💀死叉' if macd_data['cross'] == 'death'  else
                                '↑' if macd_data['hist'] > 0 else '↓')

                    self.pair_data_updated.emit(inst_id, {
                        'price':        price,
                        'rsi':          round(rsi, 1),
                        'volume_ratio': vol_ratio,
                        'macd':         macd_data,
                        'bb':           bb_data,
                        'd1_trend':     d1_trend,   # 'bull'/'bear'/'sideways'/'unknown'
                    })
                    self.status_update.emit(
                        inst_id,
                        f"✓ ${price:.4f}  RSI{rsi:.0f}  MACD{macd_sym}  "
                        f"BB{bb_data['position']:.2f}  D1:{'🟢' if d1_trend=='bull' else '🔴' if d1_trend=='bear' else '➖'}"
                    )

            except Exception as e:
                self.status_update.emit(inst_id, f"错误: {e}")
                self.log_message.emit(f"{inst_id} 检测异常: {e}", "ERROR")

        return all_signals

    # ── 线程控制 ──
    def stop(self):
        self._stop_flag = True

    def is_running(self) -> bool:
        return not self._stop_flag


# ── 监控工作线程 ──


class MonitorWorker(QThread):
    """后台监控工作线程"""
    signal_detected = Signal(object)
    status_update = Signal(str, str)
    log_message = Signal(str, str)
    pair_data_updated = Signal(str, dict)
    # 高质量信号桥接（转发自 MonitorEngine.high_quality_signal）
    high_quality_signal = Signal(dict)

    def __init__(self, okx_client, config: Dict = None, pairs: List[str] = None):
        super().__init__()
        self._okx_client = okx_client
        self._config = config or {}
        self._pending_pairs = pairs or []
        self._engine: MonitorEngine = None

    def set_pairs(self, pairs: List[str]):
        if self._engine:
            self._engine.set_pairs(pairs)

    def add_pair(self, inst_id: str):
        if self._engine:
            self._engine.add_pair(inst_id)

    def remove_pair(self, inst_id: str):
        if self._engine:
            self._engine.remove_pair(inst_id)

    def get_pairs(self) -> List[str]:
        if self._engine:
            return self._engine.get_pairs()
        return []

    def update_config(self, config: Dict):
        if self._engine:
            self._engine.update_config(config)

    def stop(self):
        if self._engine:
            self._engine.stop()

    def run(self):
        """在工作线程中创建引擎，确保线程亲和性正确"""
        self._engine = MonitorEngine(self._okx_client)
        if self._config:
            self._engine.update_config(self._config)

        # 显式使用 QueuedConnection 确保跨线程信号安全
        self._engine.signal_detected.connect(self.signal_detected, Qt.QueuedConnection)
        self._engine.status_update.connect(self.status_update, Qt.QueuedConnection)
        self._engine.log_message.connect(self.log_message, Qt.QueuedConnection)
        self._engine.pair_data_updated.connect(self.pair_data_updated, Qt.QueuedConnection)
        self._engine.high_quality_signal.connect(self.high_quality_signal, Qt.QueuedConnection)

        self._engine.log_message.emit("🔍 重点监控池已启动", "SUCCESS")

        # 应用初始交易对列表
        if self._pending_pairs:
            self._engine.set_pairs(self._pending_pairs)

        interval = max(15, self._engine.config['check_interval_sec'])

        while self._engine.is_running():
            try:
                self._engine.check_all()
            except Exception as e:
                self._engine.log_message.emit(f"监控循环异常: {e}", "ERROR")

            # 分段睡眠，便于及时停止
            for _ in range(interval):
                if not self._engine.is_running():
                    break
                self.msleep(1000)  # 使用 msleep 替代 time.sleep，更安全

        self._engine.log_message.emit("⏹ 重点监控池已停止", "INFO")
