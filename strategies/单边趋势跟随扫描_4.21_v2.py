"""
经典趋势策略：单边趋势跟随扫描（精修版 v2）

v1 → v2 变更摘要
─────────────────────────────────────────────────────
【核心问题诊断】
v1 作为"单边趋势跟随"策略，有几个根本性的设计缺陷导致它
**抓不到真正的趋势行情**：

1. 斜率门槛过高（日线 0.8%，4H 1.0%）：6 根 EMA21 涨 0.8% 意味着
   日线级别年化 50%+ 的趋势，只有极端行情才满足。正常的月级上升
   趋势（如 BTC 从 6 万涨到 7 万）斜率只有 ~0.3-0.5%。
   v2 降低到 0.4%/0.5%，并引入 ADX 替代纯斜率来判定"是否在趋势中"。

2. 趋势判定只看 EMA 排列+斜率，不区分"趋势"和"震荡中偶然排列对齐"
   → v2 引入 ADX ≥ 18 + EMA 发散度，与其他优化后的策略一致。

3. 方向判定只看 1H EMA 排列（h1_ema21 > h1_ema55），没有时效性
   → 一个品种如果一直涨，每次扫描都给 BUY，没有"新信号"的概念。
   v2 新增"趋势加速确认"：要求近期出现过新的加速迹象（突破新高、
   量能放大、MACD 扩张等），避免在老趋势末段反复给信号。

4. 3m 企稳只看 EMA8 穿越，没有和 1H 级别的趋势动态结合
   → v2 保留 EMA8 企稳但放宽为不强制要求"刚穿越那根"
   （趋势跟随不一定有回踩，可能只是短暂横盘后继续）。

5. _volume_ratio 虽然已修但没有做未收盘 bar 的进度修正 → v2 修复。
6. _to_df dropna 在 vol=null 时丢整行 → v2 修复。
7. min_score=74 且基础满分=88，实际只需 84% 通过率，门槛不高不低
   但权重分配失衡（斜率独占 24 分 = 27%，信息量不够大）→ v2 重构。

【评分重构（合计 100 + 6 bonus）】
- trend_quality  28  （多周期 EMA 排列 + ADX + 发散度 + 趋势效率）
- momentum       16  （日线/4H 斜率 + MACD 扩张程度）
- h1_alignment   12  （1H EMA 顺势排列 + 价格在 EMA 上/下方稳定性）
- acceleration   12  （趋势加速信号：新高/量能放大/MACD 创新高，新增）
- breakout        8  （1H 顺势创新段）
- volume          8  （量比 + z-score）
- rsi             6  （RSI 健康区间 bonus）
- extension       6  （延伸适中 bonus）
- volatility      4  （ATR 合理区间 bonus，新增）
- 3m             +6  （企稳 bonus）
─────────────────────────────────────────────────────
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from strategies._shared.indicators import (_to_df, _check_df, _ema, _rsi_wilder, _atr, _adx,
                                          _efficiency_ratio, _macd, _volume_ratio_adjusted,
                                          _volume_zscore, _local_trend_snapshot)

try:
    from src.scanner.base_scanner import BaseScannerStrategy, ScanCondition, ScannerSymbol
    from src.scanner.ranking import build_opportunity_profile
    from src.scanner.trend_quality import trend_quality_snapshot as _external_trend_snapshot
    _HAS_SCANNER_BASE = True
except ImportError:
    BaseScannerStrategy = object
    ScanCondition = None; ScannerSymbol = None
    build_opportunity_profile = None; _external_trend_snapshot = None
    _HAS_SCANNER_BASE = False

_W_TREND       = 28
_W_MOMENTUM    = 16
_W_H1_ALIGN    = 12
_W_ACCEL       = 12  # 新增
_W_BREAKOUT    =  8
_W_VOLUME      =  8
_W_RSI         =  6
_W_EXTENSION   =  6
_W_VOLATILITY  =  4  # 新增
_W_3M          =  6

_ADX_MIN       = 18.0
_EMA_SPREAD_MIN = 0.20

_DEFAULT_CONFIG = {
    'min_score': 70, 'min_volume_24h': 18_000_000,
    'min_daily_slope_pct': 0.4, 'min_h4_slope_pct': 0.5,
    'min_breakout_pct': 0.8, 'min_volume_ratio': 1.2,
    'max_extension_atr': 3.5,
    'require_3m_stabilize': True,
    'm3_stabilize_window': 30, 'm3_stabilize_confirm_bars': 5,
    'm3_ema_span': 8, 'm3_slope_lookback': 4,
}


class DirectionalTrendFollowScanner(BaseScannerStrategy if _HAS_SCANNER_BASE else object):
    required_bars = ['1D', '4H', '1H', '3m']
    name = "单边趋势跟随扫描"
    description = "多周期单边趋势 + ADX/发散度/MACD 确认 + 趋势加速信号 + 3m 企稳"
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
            value=self.config.get('min_volume_24h', 18_000_000),
        ))

    def scan_symbol(self, symbol) -> Dict:
        km = symbol.extra_data.get('klines', {})
        try:
            d1 = _to_df(self._get_klines(km, '1D'))
            h4 = _to_df(self._get_klines(km, '4H'))
            h1 = _to_df(self._get_klines(km, '1H'))
            m3 = _to_df(self._get_klines(km, '3m'))
            analysis = _analyze_core(d1, h4, h1, m3, symbol.last_price, self.config)
        except Exception as exc:
            return {'symbol': symbol.inst_id, 'passed': False, 'score': 0.0,
                    'direction': 'WAIT', 'details': {'状态': f'分析异常: {exc}'}}
        if not analysis['valid']:
            return {'symbol': symbol.inst_id, 'passed': False, 'score': 0.0,
                    'direction': 'WAIT', 'details': {'状态': analysis.get('reason', '')}}
        ms = float(self.config.get('min_score', 70))
        passed = analysis['score'] >= ms and analysis['direction'] in {'BUY', 'SELL'}
        result = {
            'symbol': symbol.inst_id, 'passed': passed,
            'score': round(analysis['score'], 2), 'direction': analysis['direction'],
            'signals': analysis['signals'], 'details': analysis['details'],
            'last_price': symbol.last_price, 'volume_24h': symbol.volume_24h,
            'price_change_24h': symbol.price_change_24h,
            'category': '单边趋势跟随', 'ranking_factors': analysis.get('ranking_factors', {}),
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
            'min_score':                 {'type':'int',  'default':70,        'label':'最低通过分数'},
            'min_volume_24h':            {'type':'float','default':18_000_000,'label':'最小24H成交额'},
            'min_daily_slope_pct':       {'type':'float','default':0.4,       'label':'最小日线EMA21斜率%'},
            'min_h4_slope_pct':          {'type':'float','default':0.5,       'label':'最小4H EMA21斜率%'},
            'min_breakout_pct':          {'type':'float','default':0.8,       'label':'最小1H顺势新段%'},
            'min_volume_ratio':          {'type':'float','default':1.2,       'label':'最小确认量比'},
            'max_extension_atr':         {'type':'float','default':3.5,       'label':'最大延伸ATR倍数'},
            'require_3m_stabilize':      {'type':'bool', 'default':True,      'label':'要求3m回调企稳确认'},
            'm3_stabilize_window':       {'type':'int',  'default':30,        'label':'3m观察窗口(根数)'},
            'm3_stabilize_confirm_bars': {'type':'int',  'default':5,         'label':'3m企稳确认根数'},
            'm3_ema_span':               {'type':'int',  'default':8,         'label':'3m快速EMA周期'},
            'm3_slope_lookback':         {'type':'int',  'default':4,         'label':'3m EMA斜率回望'},
        }


# ══════════════════════════════════════════════
# 纯函数 API
# ══════════════════════════════════════════════
def analyze_bars(d1, h4, h1, m3, last_price, config=None):
    cfg = {**_DEFAULT_CONFIG, **(config or {})}
    try: return _analyze_core(d1, h4, h1, m3, last_price, cfg)
    except Exception as exc:
        return {'valid':False,'reason':f'异常: {exc}','score':0.0,'direction':'WAIT',
                'signals':[],'details':{},'ranking_factors':{}}

def klines_list_to_df(rows): return _to_df(rows)


# ══════════════════════════════════════════════
# 核心分析
# ══════════════════════════════════════════════
def _analyze_core(d1, h4, h1, m3, last_price, cfg):
    _check_df(d1, '日线', 90); _check_df(h4, '4H', 120); _check_df(h1, '1H', 150)
    require_3m = bool(cfg.get('require_3m_stabilize', True))
    min_3m_bars = int(cfg.get('m3_stabilize_window', 30)) + 10
    if require_3m and len(m3) < min_3m_bars:
        _check_df(m3, '3m', min_3m_bars)

    score = 0.0; signals = []
    price = float(last_price) if last_price and last_price > 0 else float(h1['c'].iloc[-1])

    # 均线
    d1_ema21 = _ema(d1['c'], 21); d1_ema55 = _ema(d1['c'], 55)
    h4_ema21 = _ema(h4['c'], 21); h4_ema55 = _ema(h4['c'], 55)
    h1_ema21 = _ema(h1['c'], 21); h1_ema55 = _ema(h1['c'], 55)

    # 斜率
    d1_slope = _ema_slope_pct(d1['c'], 21, 6)
    h4_slope = _ema_slope_pct(h4['c'], 21, 6)

    # 辅助
    h1_rsi = _rsi_wilder(h1['c'])
    h4_atr = _atr(h4); h4_atr_pct = _atr_pct(h4)
    vol_ratio = _volume_ratio_adjusted(h1)
    vol_zscore = _volume_zscore(h1['vol'])
    d1_adx = _adx(d1, 14); h4_adx = _adx(h4, 14)
    extension_atr = abs(price - h4_ema21) / h4_atr if h4_atr > 0 else 0.0

    d1_ema_spread = (d1_ema21 - d1_ema55) / d1_ema55 * 100 if d1_ema55 > 0 else 0.0
    h4_ema_spread = (h4_ema21 - h4_ema55) / h4_ema55 * 100 if h4_ema55 > 0 else 0.0

    # MACD
    _, _, h1_macd_hist = _macd(h1['c'])
    _, _, h4_macd_hist = _macd(h4['c'])

    # 趋势快照
    trend_snap = _trend_snapshot(d1, h4, h1, price)
    trend_metrics = trend_snap.get('metrics', {})
    trend_long_score = float(trend_snap.get('long_score', 0) or 0)
    trend_short_score = float(trend_snap.get('short_score', 0) or 0)

    _m3_def = _m3_stabilize_default('趋势未确认')

    min_d1_slope = float(cfg.get('min_daily_slope_pct', 0.4))
    min_h4_slope = float(cfg.get('min_h4_slope_pct', 0.5))

    # 趋势方向（含 ADX + 发散度 + 降低后的斜率门槛）
    bullish = (
        bool(trend_snap.get('long_ok'))
        and price > d1_ema21 > d1_ema55 and price > h4_ema21 > h4_ema55
        and d1_slope > min_d1_slope and h4_slope > min_h4_slope
        and d1_adx >= _ADX_MIN and h4_adx >= _ADX_MIN
        and d1_ema_spread >= _EMA_SPREAD_MIN and h4_ema_spread >= _EMA_SPREAD_MIN
    )
    bearish = (
        bool(trend_snap.get('short_ok'))
        and price < d1_ema21 < d1_ema55 and price < h4_ema21 < h4_ema55
        and d1_slope < -min_d1_slope and h4_slope < -min_h4_slope
        and d1_adx >= _ADX_MIN and h4_adx >= _ADX_MIN
        and d1_ema_spread <= -_EMA_SPREAD_MIN and h4_ema_spread <= -_EMA_SPREAD_MIN
    )

    # ① 趋势质量（28 分）
    if bullish or bearish:
        base_t = 20.0
        spread_abs = (abs(d1_ema_spread) + abs(h4_ema_spread)) / 2.0
        spread_bonus = min(4.0, spread_abs / 2.0 * 4.0)
        adx_avg = (d1_adx + h4_adx) / 2.0
        adx_bonus = min(4.0, max(0.0, (adx_avg - _ADX_MIN) / 12.0 * 4.0))
        ts_ = base_t + spread_bonus + adx_bonus
        score += ts_
        dl = "多头" if bullish else "空头"
        ref = trend_long_score if bullish else trend_short_score
        signals.append(f"{dl}趋势通过(质量{ref:.0f}, ADX {adx_avg:.1f}, 发散{spread_abs:.2f}% → +{ts_:.1f}分)")
    else:
        rp = []
        if not (bool(trend_snap.get('long_ok')) or bool(trend_snap.get('short_ok'))):
            rp.append(str(trend_snap.get('reason', '趋势质量不足')))
        if d1_adx < _ADX_MIN: rp.append(f"日线ADX不足({d1_adx:.1f})")
        if h4_adx < _ADX_MIN: rp.append(f"4H ADX不足({h4_adx:.1f})")
        if abs(d1_slope) < min_d1_slope: rp.append(f"日线斜率不足({d1_slope:.2f}%)")
        if abs(h4_slope) < min_h4_slope: rp.append(f"4H斜率不足({h4_slope:.2f}%)")
        signals.append("趋势未确认: " + " | ".join(rp) if rp else "趋势未确认")
        return _build_result(
            valid=True, score=score, direction='WAIT', signals=signals,
            d1_slope=d1_slope, h4_slope=h4_slope, vol_ratio=vol_ratio, vol_zscore=vol_zscore,
            h1_rsi=h1_rsi, extension_atr=extension_atr, breakout_pct=0.0,
            h4_atr_pct=h4_atr_pct, d1_adx=d1_adx, h4_adx=h4_adx,
            d1_ema_spread=d1_ema_spread, h4_ema_spread=h4_ema_spread,
            bullish=bullish, bearish=bearish,
            trend_long_score=trend_long_score, trend_short_score=trend_short_score,
            trend_metrics=trend_metrics, m3_stab=_m3_def)

    # ② 动量（16 分）— 斜率 + MACD 扩张
    slope_score = min(8.0, 4.0 + abs(d1_slope + h4_slope) / 4.0 * 4.0)
    score += slope_score

    macd_expanding = False
    if len(h4_macd_hist) >= 3:
        mh = h4_macd_hist.values[-3:]
        if bullish:
            macd_expanding = mh[-1] > mh[-2] > mh[-3] and mh[-1] > 0
        else:
            macd_expanding = mh[-1] < mh[-2] < mh[-3] and mh[-1] < 0
    if macd_expanding:
        score += 8.0
        signals.append(f"MACD持续扩张(+8分)")
    elif len(h4_macd_hist) >= 2:
        last_mh = float(h4_macd_hist.iloc[-1])
        if (bullish and last_mh > 0) or (bearish and last_mh < 0):
            score += 4.0
            signals.append("MACD方向正确(+4分)")
    signals.append(f"动量(斜率{d1_slope:.2f}%/{h4_slope:.2f}% → +{slope_score:.0f}分)")

    # ③ 1H EMA 顺势排列（12 分）
    h1_align_bull = bullish and price > h1_ema21 > h1_ema55
    h1_align_bear = bearish and price < h1_ema21 < h1_ema55
    if h1_align_bull or h1_align_bear:
        # 排列稳定性：近 10 根 1H 有多少根维持正确排列
        h1_ema21_s = h1['c'].ewm(span=21, adjust=False).mean()
        h1_ema55_s = h1['c'].ewm(span=55, adjust=False).mean()
        recent_c = h1['c'].tail(10).values
        r21 = h1_ema21_s.tail(10).values; r55 = h1_ema55_s.tail(10).values
        if bullish:
            stability = float(((recent_c > r21) & (r21 > r55)).mean())
        else:
            stability = float(((recent_c < r21) & (r21 < r55)).mean())
        align_score = _W_H1_ALIGN * (0.6 + stability * 0.4)
        score += align_score
        signals.append(f"1H顺势{'多' if bullish else '空'}头排列(稳定{stability:.0%} → +{align_score:.1f}分)")
    else:
        signals.append("1H EMA排列尚未顺势")

    # ④ 趋势加速信号（12 分，新增）— 防止老趋势反复给信号
    accel_score = 0.0
    # 近 6 根 1H 有没有创 24 根新高/新低
    breakout_pct = _trend_breakout_pct(h1, bull=bullish)
    if breakout_pct > 0.3:
        accel_score += 4.0
    # 近 3 根 vol 有没有放量（至少 1 根 > 1.5x）
    if len(h1) >= 23:
        recent_vols = h1['vol'].iloc[-3:].values
        baseline_vol = float(h1['vol'].iloc[-23:-3].mean())
        if baseline_vol > 0 and max(recent_vols) / baseline_vol >= 1.5:
            accel_score += 4.0
    # MACD hist 在 1H 上近 3 根递增/递减
    if len(h1_macd_hist) >= 3:
        mh1 = h1_macd_hist.values[-3:]
        if (bullish and mh1[-1] > mh1[-2] > mh1[-3]) or \
           (bearish and mh1[-1] < mh1[-2] < mh1[-3]):
            accel_score += 4.0
    if accel_score > 0:
        score += accel_score
        signals.append(f"趋势加速信号(+{accel_score:.0f}分)")
    else:
        signals.append("无新加速迹象")

    # ⑤ 1H 顺势突破（8 分）
    min_bp = float(cfg.get('min_breakout_pct', 0.8))
    if breakout_pct >= min_bp:
        bp_s = _W_BREAKOUT * min(1.0, 0.6 + (breakout_pct - min_bp) / max(min_bp * 2, 0.1) * 0.4)
        score += bp_s
        signals.append(f"1H创新段({breakout_pct:.2f}% → +{bp_s:.0f}分)")

    # ⑥ 量能（8 分）— 量比 + z-score
    min_vr = float(cfg.get('min_volume_ratio', 1.2))
    vrok = vol_ratio >= min_vr; vzok = vol_zscore >= 0.5
    if vrok and vzok:
        vs = _W_VOLUME * 0.85
        score += vs; signals.append(f"量能强({vol_ratio:.2f}x, z={vol_zscore:+.2f} → +{vs:.0f}分)")
    elif vrok or vzok:
        vs = _W_VOLUME * 0.5
        score += vs; signals.append(f"量能部分({vol_ratio:.2f}x, z={vol_zscore:+.2f} → +{vs:.0f}分)")
    else:
        signals.append(f"量能不足({vol_ratio:.2f}x)")

    # ⑦ RSI 健康（6 分 bonus）
    rsi_ok = (bullish and 55 <= h1_rsi <= 72) or (bearish and 28 <= h1_rsi <= 45)
    if rsi_ok:
        score += _W_RSI
        signals.append(f"RSI健康({h1_rsi:.1f} → +{_W_RSI}分)")
    elif (bullish and h1_rsi > 80) or (bearish and h1_rsi < 20):
        signals.append(f"RSI极端({'超买' if bullish else '超卖'}: {h1_rsi:.1f})")

    # ⑧ 延伸适中（6 分 bonus）
    max_ext = float(cfg.get('max_extension_atr', 3.5))
    if extension_atr <= max_ext:
        es = _W_EXTENSION * max(0.0, 1.0 - extension_atr / max(max_ext, 0.1))
        if es >= 1.5:
            score += es; signals.append(f"延伸适中({extension_atr:.1f}ATR → +{es:.0f}分)")
    else:
        signals.append(f"延伸偏大({extension_atr:.1f}ATR)")

    # ⑨ 波动率合理（4 分 bonus，新增）
    if 1.5 <= h4_atr_pct <= 6.0:
        score += _W_VOLATILITY
        signals.append(f"波动率合理({h4_atr_pct:.2f}% → +{_W_VOLATILITY}分)")

    # ⑩ 3m 企稳（+6 bonus）
    m3_stab = (
        _three_min_stabilize_core(m3, bull_bias=bullish, cfg=cfg)
        if len(m3) >= min_3m_bars
        else _m3_stabilize_default('3m数据不足'))
    if m3_stab.get('passed'):
        score += _W_3M; signals.append(str(m3_stab.get('reason')))
    elif m3_stab.get('valid'):
        signals.append(f"3m观察：{m3_stab.get('reason')}")

    # 方向判定
    # v2：要求 1H 排列 + 趋势加速至少有一项
    has_accel = accel_score > 0
    m3_ok = bool(m3_stab.get('passed')) or not require_3m
    direction = 'WAIT'
    if h1_align_bull and has_accel and m3_ok:
        direction = 'BUY'
    elif h1_align_bear and has_accel and m3_ok:
        direction = 'SELL'

    return _build_result(
        valid=True, score=score, direction=direction, signals=signals,
        d1_slope=d1_slope, h4_slope=h4_slope, vol_ratio=vol_ratio, vol_zscore=vol_zscore,
        h1_rsi=h1_rsi, extension_atr=extension_atr, breakout_pct=breakout_pct,
        h4_atr_pct=h4_atr_pct, d1_adx=d1_adx, h4_adx=h4_adx,
        d1_ema_spread=d1_ema_spread, h4_ema_spread=h4_ema_spread,
        bullish=bullish, bearish=bearish,
        trend_long_score=trend_long_score, trend_short_score=trend_short_score,
        trend_metrics=trend_metrics, m3_stab=m3_stab)


# ══════════════════════════════════════════════
# 3m 企稳（放宽：趋势跟随允许"已在正确侧持稳"也通过）
# ══════════════════════════════════════════════
def _three_min_stabilize_core(df, *, bull_bias, cfg):
    window = int(cfg.get('m3_stabilize_window', 30))
    confirm_bars = int(cfg.get('m3_stabilize_confirm_bars', 5))
    ema_span = int(cfg.get('m3_ema_span', 8))
    slope_lb = int(cfg.get('m3_slope_lookback', 4))
    if len(df) < ema_span + slope_lb + 2:
        return _m3_stabilize_default(f'3m数据不足')
    ema8_s = df['c'].ewm(span=ema_span, adjust=False).mean()
    rc = df['c'].tail(window + confirm_bars).reset_index(drop=True)
    re = ema8_s.tail(window + confirm_bars).reset_index(drop=True)
    n = len(rc)
    if n < 3: return _m3_stabilize_default('3m样本不足')
    lc = float(rc.iloc[-1]); pc = float(rc.iloc[-2])
    le = float(re.iloc[-1]); pe = float(re.iloc[-2])
    slope_ok = ((bull_bias and le > float(re.iloc[-slope_lb-1]))
                or (not bull_bias and le < float(re.iloc[-slope_lb-1])))
    if bull_bias:
        cross_ok = pc <= pe and lc > le
        above_now = lc > le
    else:
        cross_ok = pc >= pe and lc < le
        above_now = lc < le
    cs = rc.tail(confirm_bars); ce = re.tail(confirm_bars)
    sr = float((cs > ce).mean()) if bull_bias else float((cs < ce).mean())

    if cross_ok and slope_ok:
        q = min(100, 65 + sr * 25 + (10 if sr >= 0.8 else 0))
        lb = "多头" if bull_bias else "空头"
        return {'valid':True,'passed':True,'quality':q,
                'reason':f"3m{lb}回调企稳EMA{ema_span}(持稳{sr:.0%})",
                'crossover_ok':True,'ema_slope_ok':True,'stabilize_ratio':sr,
                'last_ema8':le,'ema8_slope_pct':(le/float(re.iloc[-slope_lb-1])-1)*100}
    elif above_now and slope_ok and sr >= 0.6:
        # v2 放宽：趋势跟随中"已经持稳"也可以通过（不需要刚好看到穿越那根）
        q = 55 + sr * 20
        lb = "多头" if bull_bias else "空头"
        return {'valid':True,'passed':True,'quality':q,
                'reason':f"3m{lb}持稳EMA{ema_span}(持稳{sr:.0%}，斜率正确)",
                'crossover_ok':False,'ema_slope_ok':True,'stabilize_ratio':sr,
                'last_ema8':le,'ema8_slope_pct':(le/float(re.iloc[-slope_lb-1])-1)*100}
    elif not slope_ok:
        return {'valid':True,'passed':False,'quality':20.0,
                'reason':f"3m EMA{ema_span}斜率方向不符",
                'crossover_ok':cross_ok,'ema_slope_ok':False,'stabilize_ratio':sr,
                'last_ema8':le,'ema8_slope_pct':(le/float(re.iloc[-slope_lb-1])-1)*100}
    else:
        lb = "多头" if bull_bias else "空头"
        return {'valid':True,'passed':False,'quality':30.0,
                'reason':f"3m价格在EMA{ema_span}{'下方' if bull_bias else '上方'}",
                'crossover_ok':False,'ema_slope_ok':slope_ok,'stabilize_ratio':sr,
                'last_ema8':le,'ema8_slope_pct':(le/float(re.iloc[-slope_lb-1])-1)*100}


# ══════════════════════════════════════════════
# 趋势快照 / 底层工具
# ══════════════════════════════════════════════
def _trend_snapshot(d1, h4, h1, price):
    if _external_trend_snapshot:
        try: return _external_trend_snapshot(d1, h4, h1, price)
        except Exception: pass
    return _local_trend_snapshot(d1, h4, h1, price)

def _ema_slope_pct(c,span,lb):
    e=c.ewm(span=span,adjust=False).mean()
    if len(e)<=lb:return 0.0
    b=float(e.iloc[-(lb+1)]);l=float(e.iloc[-1])
    return(l/b-1)*100 if b>0 else 0.0
def _atr_pct(df,period=14):
    a=_atr(df,period);lc=float(df['c'].iloc[-1])
    return float(a/lc*100) if lc>0 else 0.0
def _trend_breakout_pct(df,bull):
    if bull is None or len(df)<26:return 0.0
    lc=float(df['c'].iloc[-1]);ref=df.iloc[-25:-1]
    if bull:
        rh=float(ref['h'].max())
        return max((lc-rh)/rh*100,0.0) if rh>0 else 0.0
    else:
        rl=float(ref['l'].min())
        return max((rl-lc)/rl*100,0.0) if rl>0 else 0.0
def _m3_stabilize_default(reason):
    return{'valid':False,'passed':False,'quality':30.0,'reason':reason,
           'crossover_ok':False,'ema_slope_ok':False,'stabilize_ratio':0.0,
           'last_ema8':0.0,'ema8_slope_pct':0.0}
def _build_result(*,valid,score,direction,signals,d1_slope,h4_slope,vol_ratio,vol_zscore,
                  h1_rsi,extension_atr,breakout_pct,h4_atr_pct,d1_adx,h4_adx,
                  d1_ema_spread,h4_ema_spread,bullish,bearish,
                  trend_long_score,trend_short_score,trend_metrics,m3_stab,reason=''):
    tq=trend_long_score if bullish else trend_short_score if bearish else max(trend_long_score,trend_short_score)
    trigger_q=88.0 if breakout_pct>=0.8 else 45.0
    vol_q=min(vol_ratio/1.2,1.6)*62.5
    loc_q=max(20,100-max(extension_atr-1,0)*18)
    fresh_q=92.0 if direction in{'BUY','SELL'} and breakout_pct<=3.5 else 68.0 if direction in{'BUY','SELL'} else 30.0
    return{
        'valid':valid,'reason':reason,'score':max(score,0.0),'direction':direction,'signals':signals,
        'ranking_factors':{'trend':tq,'trigger':trigger_q,'volume':vol_q,'location':loc_q,
            'freshness':fresh_q,'risk':88.0 if h4_atr_pct<=6 else 55.0},
        'details':{
            '评估':' | '.join(signals) if signals else '暂无单边趋势跟随机会',
            '日线斜率':f'{d1_slope:.2f}%','4H斜率':f'{h4_slope:.2f}%',
            '日线ADX':f'{d1_adx:.1f}','4H_ADX':f'{h4_adx:.1f}',
            '日线EMA发散':f'{d1_ema_spread:+.2f}%','4H_EMA发散':f'{h4_ema_spread:+.2f}%',
            '量比':f'{vol_ratio:.2f}x','量能Z分':f'{vol_zscore:+.2f}σ',
            '1H_RSI':f'{h1_rsi:.1f}','延伸(ATR倍)':f'{extension_atr:.1f}',
            '4H_ATR%':f'{h4_atr_pct:.2f}%',
            '趋势质量':str(trend_metrics.get('reason','-') or '-'),
            'H4_ADX_内置':f"{float(trend_metrics.get('h4_adx',0)):.1f}",
            '趋势效率':f"{float(trend_metrics.get('h1_efficiency',0)):.1f}",
            '3m企稳确认':'通过' if m3_stab.get('passed') else '未通过',
            '3m企稳说明':str(m3_stab.get('reason','-')),
            '3m EMA8':f"{float(m3_stab.get('last_ema8',0)):.8g}",
            '3m EMA8斜率':f"{float(m3_stab.get('ema8_slope_pct',0)):.2f}%",
            '3m持稳比例':f"{float(m3_stab.get('stabilize_ratio',0)):.0%}",
        },
    }

STRATEGY_NAME  = "单边趋势跟随扫描"
STRATEGY_TYPE  = "scan"
STRATEGY_CLASS = DirectionalTrendFollowScanner
