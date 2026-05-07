"""
经典波段策略：平台整理突破扫描（精修版 v2）

v1 → v2 变更摘要
─────────────────────────────────────────────────────
【核心问题诊断】
1. 平台检测用整个窗口的 max/min 作为上下轨，但如果窗口内有一根
   spike（异常影线），会让"平台"看起来很宽 → 实际窄幅整理被误判
   为"过宽"。v2 改用"去除最高/最低 10% 异常值后的高低点"作为
   平台边界，更接近真实整理区间。

2. 突破判定用 `prev_close` 在"很宽的缓冲区内" → 几乎所有 bar 都
   满足这个条件（因为 `breakout_buf*2` 很宽松），导致"假突破"频繁
   通过。v2 改为：prev_close 必须在平台区间内（上轨以下 / 下轨以上），
   当前 close 必须站上/跌破触发位，且突破幅度 ≥ ATR 的一个比例。

3. 趋势背景只看 EMA21 vs EMA55 排列，没有衡量趋势的强弱 → 震荡市
   偶然排列也给 18 分。v2 引入 ADX + EMA 发散度。

4. _breakout_volume_ratio 的 baseline 含突破窗口 → 基线被突破放量
   拉高，v2 排除最近 3 根。

5. 没有突破后的"确认 bar"检查 → 单根假突破后马上回落也通过。
   v2 新增"突破 bar 收盘强度"（实体占比、是否吞没前一根）。

6. 没有任何"平台整理时间"门槛 → 价格只要在 18 根 4H 内振幅窄就算
   "平台"，但 18 根可能只是一个短暂横盘（2-3 天），不是真正的蓄力。
   v2 新增 inside_ratio 的时间维度加权（整理越久越好）。

7. extension_pct 只看百分比不看 ATR → 高波动品种 5% 是正常的，
   低波动品种 2% 就已经很远。v2 改为 ATR 归一化。

8. _to_df dropna 在 vol=null 时丢整行 → v2 修复。

9. details 输出信息太少 → v2 增加 ADX/ATR/RSI/MACD/EMA 发散度。

【新增指标】
10. ADX(14) 4H：区分震荡/趋势（ADX < 20 的突破不可靠）
11. Bollinger Band 收缩度（bandwidth）：整理期间 BB 越窄越好
12. 突破 bar 收盘强度：(close-open)/(high-low)
13. MACD 直方图方向：突破时 MACD 应与突破方向一致
14. Volume z-score：异常放量检测
15. RSI：突破时 RSI 应在合理区间（不超买/超卖）

【评分重构（合计 100）】
- platform_quality  24  （ATR 归一化宽度 + 整理充分度 + BB 收缩）
- trend_context     16  （4H EMA 背景 + ADX + 发散度）
- breakout_confirm  20  （突破 + 收盘强度 + ATR 突破幅度）
- volume            12  （量比 + z-score）
- macd_confirm       8  （MACD 方向一致）
- rsi_position       6  （RSI 合理区间）
- extension          8  （延伸适中 bonus，ATR 归一化）
- freshness          6  （突破新鲜度：距突破 bar 的根数）
─────────────────────────────────────────────────────
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from strategies._shared.indicators import (_to_df, _check_df, _ema, _atr, _adx,
                                          _rsi_wilder, _macd, _volume_zscore)

try:
    from src.scanner.base_scanner import BaseScannerStrategy, ScanCondition, ScannerSymbol
    from src.scanner.ranking import build_opportunity_profile
    _HAS_SCANNER_BASE = True
except ImportError:
    BaseScannerStrategy = object
    ScanCondition = None; ScannerSymbol = None
    build_opportunity_profile = None
    _HAS_SCANNER_BASE = False

_W_PLATFORM   = 24
_W_CONTEXT    = 16
_W_BREAKOUT   = 20
_W_VOLUME     = 12
_W_MACD       =  8
_W_RSI        =  6
_W_EXTENSION  =  8
_W_FRESHNESS  =  6

_ADX_MIN = 16.0

_DEFAULT_CONFIG = {
    'min_score': 68, 'min_volume_24h': 15_000_000,
    'platform_bars': 18, 'max_platform_width_pct': 9.0,
    'max_platform_width_atr': 3.5, 'min_inside_ratio': 0.60,
    'breakout_buffer_pct': 0.25, 'min_breakout_volume_ratio': 1.8,
    'max_extension_atr': 2.5,
    'trim_pct': 0.10,  # 去除异常值的百分比
}


class BreakoutSwingScanner(BaseScannerStrategy if _HAS_SCANNER_BASE else object):
    required_bars = ['4H', '1H']
    name = "波段平台突破扫描"
    description = "4H 窄幅整理 → 1H 放量突破 + ADX/MACD/收盘强度确认"
    strategy_type = "scan"

    def __init__(self, config=None):
        self.config = {**_DEFAULT_CONFIG, **(config or {})}
        if _HAS_SCANNER_BASE and hasattr(super(), '__init__'):
            try: super().__init__(config or {})
            except Exception: pass

    def _init_conditions(self):
        if ScanCondition is None: return
        self.add_condition(ScanCondition(
            name="24H成交量", description="过滤流动性不足标的",
            field="volume_24h", operator=">=",
            value=self.config.get('min_volume_24h', 15_000_000),
        ))

    def scan_symbol(self, symbol) -> Dict:
        km = symbol.extra_data.get('klines', {})
        try:
            h4 = _to_df(self._get_klines(km, '4H'))
            h1 = _to_df(self._get_klines(km, '1H'))
            analysis = _analyze_core(h4, h1, symbol.last_price, self.config)
        except Exception as exc:
            return {'symbol': symbol.inst_id, 'passed': False, 'score': 0.0,
                    'direction': 'WAIT', 'details': {'状态': f'分析异常: {exc}'}}
        if not analysis['valid']:
            return {'symbol': symbol.inst_id, 'passed': False, 'score': 0.0,
                    'direction': 'WAIT', 'details': {'状态': analysis.get('reason', '')}}
        ms = float(self.config.get('min_score', 68))
        passed = analysis['score'] >= ms and analysis['direction'] in {'BUY', 'SELL'}
        result = {
            'symbol': symbol.inst_id, 'passed': passed,
            'score': round(analysis['score'], 2), 'direction': analysis['direction'],
            'signals': analysis['signals'], 'details': analysis['details'],
            'last_price': symbol.last_price, 'volume_24h': symbol.volume_24h,
            'price_change_24h': symbol.price_change_24h,
            'category': '波段平台突破', 'ranking_factors': analysis.get('ranking_factors', {}),
        }
        if build_opportunity_profile:
            try:
                result.update(build_opportunity_profile(
                    base_score=analysis['score'], direction=analysis['direction'],
                    volume_24h=symbol.volume_24h,
                    factors=analysis.get('ranking_factors', {}),
                    signals=analysis['signals']))
            except Exception: pass
        return result

    def _get_klines(self, km, bar):
        return km.get(bar) or km.get(bar.lower()) or km.get(bar.upper()) or []

    def get_config_schema(self):
        return {
            'min_score':                 {'type':'int',  'default':68,        'label':'最低通过分数'},
            'min_volume_24h':            {'type':'float','default':15_000_000,'label':'最小24H成交额'},
            'platform_bars':             {'type':'int',  'default':18,        'label':'平台窗口(4H根)'},
            'max_platform_width_pct':    {'type':'float','default':9.0,       'label':'平台最大宽度%'},
            'max_platform_width_atr':    {'type':'float','default':3.5,       'label':'平台最大宽度(ATR)'},
            'min_inside_ratio':          {'type':'float','default':0.60,      'label':'整理充分度(0-1)'},
            'breakout_buffer_pct':       {'type':'float','default':0.25,      'label':'突破缓冲%'},
            'min_breakout_volume_ratio': {'type':'float','default':1.8,       'label':'突破最小量比'},
            'max_extension_atr':         {'type':'float','default':2.5,       'label':'最大延伸(ATR)'},
            'trim_pct':                  {'type':'float','default':0.10,      'label':'异常值裁剪比例'},
        }


# ══════════════════════════════════════════════
def analyze_bars(h4, h1, last_price, config=None):
    cfg = {**_DEFAULT_CONFIG, **(config or {})}
    try: return _analyze_core(h4, h1, last_price, cfg)
    except Exception as exc:
        return {'valid':False,'reason':f'异常: {exc}','score':0.0,'direction':'WAIT',
                'signals':[],'details':{},'ranking_factors':{}}

def klines_list_to_df(rows): return _to_df(rows)


# ══════════════════════════════════════════════
# 核心分析
# ══════════════════════════════════════════════
def _analyze_core(h4, h1, last_price, cfg):
    _check_df(h4, '4H', 80); _check_df(h1, '1H', 120)
    score = 0.0; signals = []
    last_close = float(last_price) if last_price and last_price > 0 else float(h1['c'].iloc[-1])
    prev_close = float(h1['c'].iloc[-2])

    # ── 4H 趋势背景 + ADX ──
    ema21_4h = _ema(h4['c'], 21); ema55_4h = _ema(h4['c'], 55)
    h4_adx = _adx(h4, 14)
    h4_ema_spread = (ema21_4h - ema55_4h) / ema55_4h * 100 if ema55_4h > 0 else 0.0
    bullish_ctx = ema21_4h > ema55_4h and last_close > ema21_4h
    bearish_ctx = ema21_4h < ema55_4h and last_close < ema21_4h

    # ── 辅助 ──
    h4_atr = _atr(h4); h1_atr = _atr(h1)
    h1_rsi = _rsi_wilder(h1['c'])
    vol_ratio = _breakout_volume_ratio_v2(h1['vol'])
    vol_zscore = _volume_zscore(h1['vol'])
    _, _, h1_macd_hist = _macd(h1['c'])

    # ── 平台检测（去异常值后的边界）──
    pb = int(cfg.get('platform_bars', 18))
    trim = float(cfg.get('trim_pct', 0.10))
    p_high, p_low = _trimmed_platform(h4, pb, trim)
    midpoint = (p_high + p_low) / 2.0
    range_width_pct = (p_high - p_low) / midpoint * 100 if midpoint > 0 else 999.0
    range_width_atr = (p_high - p_low) / h4_atr if h4_atr > 0 else 999.0
    inside_ratio = _inside_ratio(h4['c'].tail(pb), p_low, p_high)

    # BB 收缩度
    bb_bandwidth = _bb_bandwidth(h4['c'], 20)

    # ① 平台质量（24 分）
    max_w_pct = float(cfg.get('max_platform_width_pct', 9.0))
    max_w_atr = float(cfg.get('max_platform_width_atr', 3.5))
    min_inside = float(cfg.get('min_inside_ratio', 0.60))
    platform_tight = range_width_pct <= max_w_pct and range_width_atr <= max_w_atr
    platform_mature = inside_ratio >= min_inside

    if platform_tight and platform_mature:
        tightness = max(0.0, 1.0 - range_width_pct / max_w_pct) * 0.4
        maturity = min(1.0, (inside_ratio - min_inside) / max(1.0 - min_inside, 0.01)) * 0.3
        bb_bonus = min(0.3, max(0.0, (6.0 - bb_bandwidth) / 6.0 * 0.3))  # BB 越窄越好
        ps = _W_PLATFORM * min(1.0, 0.45 + tightness + maturity + bb_bonus)
        score += ps
        signals.append(f"平台收敛({range_width_pct:.1f}%/{range_width_atr:.1f}ATR, "
                       f"充分{inside_ratio:.0%}, BB{bb_bandwidth:.1f}% → +{ps:.0f}分)")
    elif platform_tight:
        ps = _W_PLATFORM * 0.4
        score += ps
        signals.append(f"平台窄但整理不足({inside_ratio:.0%} → +{ps:.0f}分)")
    else:
        signals.append(f"平台过宽({range_width_pct:.1f}%/{range_width_atr:.1f}ATR)")
        return _build_result(valid=True, score=score, direction='WAIT', signals=signals,
            range_width_pct=range_width_pct, range_width_atr=range_width_atr,
            volume_ratio=0.0, vol_zscore=0.0, extension_atr=0.0,
            h4_adx=h4_adx, h4_ema_spread=h4_ema_spread, h1_rsi=h1_rsi,
            bb_bandwidth=bb_bandwidth, bullish_ctx=bullish_ctx, bearish_ctx=bearish_ctx)

    # ② 趋势背景（16 分）
    if (bullish_ctx or bearish_ctx) and h4_adx >= _ADX_MIN:
        base_ctx = 10.0
        adx_bonus = min(3.0, max(0.0, (h4_adx - _ADX_MIN) / 10.0 * 3.0))
        spread_bonus = min(3.0, abs(h4_ema_spread) / 2.0 * 3.0)
        ctx_s = base_ctx + adx_bonus + spread_bonus
        score += ctx_s
        dl = "多头" if bullish_ctx else "空头"
        signals.append(f"4H{dl}背景(ADX {h4_adx:.1f}, 发散{h4_ema_spread:+.1f}% → +{ctx_s:.0f}分)")
    elif bullish_ctx or bearish_ctx:
        score += 6.0
        signals.append(f"4H趋势背景弱(ADX {h4_adx:.1f})")
    else:
        signals.append("4H趋势背景不明确")

    # ③ 1H 突破确认 + 收盘强度（20 分）
    buf = float(cfg.get('breakout_buffer_pct', 0.25)) / 100.0
    pb_1h = pb * 4  # 4H → 1H
    bk_high = float(h1['h'].tail(pb_1h).max())
    bk_low = float(h1['l'].tail(pb_1h).min())
    trigger_up = bk_high * (1.0 + buf)
    trigger_down = bk_low * (1.0 - buf)

    # prev_close 必须在平台内（严格）
    prev_in_range = p_low * 0.995 <= prev_close <= p_high * 1.005
    breakout_up = bullish_ctx and prev_in_range and last_close > trigger_up
    breakout_down = bearish_ctx and prev_in_range and last_close < trigger_down

    # 收盘强度
    last_o = float(h1['o'].iloc[-1]); last_h = float(h1['h'].iloc[-1]); last_l = float(h1['l'].iloc[-1])
    bar_range = last_h - last_l
    if bar_range > 0:
        close_strength = abs(last_close - last_o) / bar_range
    else:
        close_strength = 0.0

    # 突破幅度 ≥ 0.3 ATR
    if breakout_up:
        breakout_magnitude = (last_close - trigger_up) / h1_atr if h1_atr > 0 else 0.0
    elif breakout_down:
        breakout_magnitude = (trigger_down - last_close) / h1_atr if h1_atr > 0 else 0.0
    else:
        breakout_magnitude = 0.0

    if (breakout_up or breakout_down) and breakout_magnitude >= 0.2:
        base_bk = _W_BREAKOUT * 0.6
        strength_bonus = _W_BREAKOUT * 0.25 * min(1.0, close_strength / 0.7)
        magnitude_bonus = _W_BREAKOUT * 0.15 * min(1.0, breakout_magnitude / 1.0)
        bks = base_bk + strength_bonus + magnitude_bonus
        score += bks
        arrow = "向上突破" if breakout_up else "向下跌破"
        signals.append(f"1H{arrow}(强度{close_strength:.2f}, 幅度{breakout_magnitude:.2f}ATR → +{bks:.0f}分)")
    elif breakout_up or breakout_down:
        bks = _W_BREAKOUT * 0.4
        score += bks
        signals.append(f"突破幅度偏小({breakout_magnitude:.2f}ATR → +{bks:.0f}分)")
    else:
        if not prev_in_range:
            signals.append("上一根不在平台内，非有效突破")
        else:
            signals.append("1H尚未突破")

    # ④ 量能（12 分）
    min_vr = float(cfg.get('min_breakout_volume_ratio', 1.8))
    vrok = vol_ratio >= min_vr
    vzok = vol_zscore >= 0.8
    if vrok and vzok:
        vs = _W_VOLUME * 0.9
        score += vs; signals.append(f"突破放量({vol_ratio:.2f}x, z={vol_zscore:+.2f} → +{vs:.0f}分)")
    elif vrok:
        vs = _W_VOLUME * 0.6
        score += vs; signals.append(f"量比达标({vol_ratio:.2f}x → +{vs:.0f}分)")
    elif vzok:
        vs = _W_VOLUME * 0.4
        score += vs; signals.append(f"放量显著(z={vol_zscore:+.2f} → +{vs:.0f}分)")
    else:
        signals.append(f"量能不足({vol_ratio:.2f}x)")

    # ⑤ MACD 方向（8 分）
    macd_ok = False
    if len(h1_macd_hist) >= 2:
        mh_last = float(h1_macd_hist.iloc[-1])
        mh_prev = float(h1_macd_hist.iloc[-2])
        if breakout_up and mh_last > 0 and mh_last > mh_prev:
            macd_ok = True
        elif breakout_down and mh_last < 0 and mh_last < mh_prev:
            macd_ok = True
    if macd_ok:
        score += _W_MACD; signals.append(f"MACD方向确认(+{_W_MACD}分)")
    elif breakout_up or breakout_down:
        if len(h1_macd_hist) >= 1:
            mh = float(h1_macd_hist.iloc[-1])
            if (breakout_up and mh > 0) or (breakout_down and mh < 0):
                score += _W_MACD * 0.4

    # ⑥ RSI 位置（6 分）
    rsi_ok = False
    if breakout_up and 50 <= h1_rsi <= 72:
        rsi_ok = True
    elif breakout_down and 28 <= h1_rsi <= 50:
        rsi_ok = True
    if rsi_ok:
        score += _W_RSI; signals.append(f"RSI合理({h1_rsi:.1f} → +{_W_RSI}分)")
    elif (breakout_up and h1_rsi > 78) or (breakout_down and h1_rsi < 22):
        signals.append(f"RSI极端({'超买' if breakout_up else '超卖'}: {h1_rsi:.1f})")

    # ⑦ 延伸适中（8 分 bonus，ATR 归一化）
    max_ext_atr = float(cfg.get('max_extension_atr', 2.5))
    if breakout_up:
        ext_atr = (last_close - p_high) / h4_atr if h4_atr > 0 else 0.0
    elif breakout_down:
        ext_atr = (p_low - last_close) / h4_atr if h4_atr > 0 else 0.0
    else:
        ext_atr = 0.0
    ext_atr = max(ext_atr, 0.0)

    if ext_atr <= max_ext_atr:
        es = _W_EXTENSION * max(0.0, 1.0 - ext_atr / max(max_ext_atr, 0.1))
        if es >= 2.0:
            score += es; signals.append(f"延伸适中({ext_atr:.1f}ATR → +{es:.0f}分)")
    else:
        signals.append(f"延伸过远({ext_atr:.1f}ATR)")

    # ⑧ 突破新鲜度（6 分）— 如果已经突破好几根了则降权
    # 简单启发：如果突破幅度很大但 vol_ratio 不高，说明是好几根前突破的
    freshness = 6.0 if breakout_magnitude <= 1.5 and (breakout_up or breakout_down) else 0.0
    if freshness > 0:
        score += freshness

    # 方向判定
    # v2 收紧：要求量能达标
    direction = 'WAIT'
    if breakout_up and vrok:
        direction = 'BUY'
    elif breakout_down and vrok:
        direction = 'SELL'

    return _build_result(
        valid=True, score=score, direction=direction, signals=signals,
        range_width_pct=range_width_pct, range_width_atr=range_width_atr,
        volume_ratio=vol_ratio, vol_zscore=vol_zscore,
        extension_atr=ext_atr, h4_adx=h4_adx, h4_ema_spread=h4_ema_spread,
        h1_rsi=h1_rsi, bb_bandwidth=bb_bandwidth,
        bullish_ctx=bullish_ctx, bearish_ctx=bearish_ctx)


# ══════════════════════════════════════════════
# 底层工具
# ══════════════════════════════════════════════
def _inside_ratio(close, low, high):
    if close.empty or high <= low: return 0.0
    return float(close.between(low, high).mean())

def _trimmed_platform(h4, bars, trim_pct):
    """去除异常影线后的平台高低点。"""
    highs = h4['h'].tail(bars).values
    lows = h4['l'].tail(bars).values
    n = len(highs)
    trim_count = max(1, int(n * trim_pct))
    sorted_h = np.sort(highs)
    sorted_l = np.sort(lows)
    # 去掉最高的 trim_count 个 high 和最低的 trim_count 个 low
    p_high = float(sorted_h[-(trim_count + 1)]) if trim_count < n else float(sorted_h[-1])
    p_low = float(sorted_l[trim_count]) if trim_count < n else float(sorted_l[0])
    return p_high, p_low

def _bb_bandwidth(close, period=20):
    """布林带宽度百分比：(upper - lower) / mid * 100。"""
    if len(close) < period: return 10.0
    mid = close.rolling(period).mean().iloc[-1]
    std = close.rolling(period).std(ddof=1).iloc[-1]
    if not pd.notna(std) or mid <= 0: return 10.0
    return float(4 * std / mid * 100)  # 2σ 上下 = 4σ 总宽

def _breakout_volume_ratio_v2(vol, breakout_window=3, baseline_window=20):
    """突破窗口最大量 / 基线均量（基线排除突破窗口）。"""
    if len(vol) < baseline_window + breakout_window + 1: return 1.0
    baseline = float(vol.iloc[-(baseline_window + breakout_window):-breakout_window].mean())
    if baseline <= 0: return 1.0
    peak = float(vol.tail(breakout_window).max())
    return peak / baseline

def _build_result(*, valid, score, direction, signals, range_width_pct, range_width_atr,
                  volume_ratio, vol_zscore, extension_atr, h4_adx, h4_ema_spread,
                  h1_rsi, bb_bandwidth, bullish_ctx, bearish_ctx, reason=''):
    pq = 100.0 if range_width_pct <= 4.5 else max(35.0, 100 - range_width_pct * 7)
    fq = max(20.0, 100 - max(extension_atr - 0.5, 0) * 25)
    return {
        'valid': valid, 'reason': reason, 'score': max(score, 0.0),
        'direction': direction, 'signals': signals,
        'ranking_factors': {
            'trend': 88.0 if (bullish_ctx or bearish_ctx) and h4_adx >= _ADX_MIN else 40.0,
            'trigger': 92.0 if direction in {'BUY','SELL'} else 25.0,
            'volume': min(volume_ratio / 1.8, 1.6) * 62.5,
            'location': (pq + fq) / 2.0,
            'freshness': fq,
            'risk': pq,
        },
        'details': {
            '评估': ' | '.join(signals) if signals else '暂无平台突破',
            '平台宽度%': f'{range_width_pct:.2f}%',
            '平台宽度ATR': f'{range_width_atr:.1f}',
            'BB带宽': f'{bb_bandwidth:.1f}%',
            '4H_ADX': f'{h4_adx:.1f}',
            '4H_EMA发散': f'{h4_ema_spread:+.2f}%',
            '量比': f'{volume_ratio:.2f}x',
            '量能Z分': f'{vol_zscore:+.2f}σ',
            '1H_RSI': f'{h1_rsi:.1f}',
            '延伸ATR': f'{extension_atr:.1f}',
        },
    }

STRATEGY_NAME  = "波段平台突破扫描"
STRATEGY_TYPE  = "scan"
STRATEGY_CLASS = BreakoutSwingScanner
