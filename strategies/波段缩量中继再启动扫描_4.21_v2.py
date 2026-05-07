"""
经典波段策略：缩量中继再启动扫描（精修版 v2）

v1 → v2 变更摘要
─────────────────────────────────────────────────────
【核心问题诊断】
1. 只做 BUY 不做 SELL：缩量中继在空头趋势中同样成立（放量续跌），
   v1 只看多头。v2 增加空头中继支持。

2. 趋势判定只看 EMA 排列 + runup_pct，没有 ADX → 震荡市偶然满足
   EMA 排列且近 40 根有 12% 振幅也通过。v2 引入 ADX ≥ 18。

3. prev_inside 判定用 `breakout_buf*2` 太宽 → 几乎所有 bar 都算
   "在整理区内"。v2 收紧为 prev_close 必须在 base_low ~ base_high
   的 ±0.5% 内。

4. 整理区"收窄"只看振幅百分比，没看 ATR → 高波动品种 4.8% 可能
   只是正常波动，不是真正收窄。v2 新增 ATR 归一化宽度。

5. 缩量检测用 consol_vol / prior_vol，但 prior_vol 可能包含前一段
   缩量尾部 → 基线偏低导致缩量系数接近 1。v2 改为用 prior 20 根
   的"中位数"而非均值，更抗异常值。

6. 没有 MACD 确认 → 放量突破但 MACD 方向反的也通过。v2 新增。

7. 没有 RSI 检查 → RSI > 85 的极度超买追涨也通过。v2 新增警示。

8. 没有突破 bar 的收盘强度检查 → 十字星假突破也通过。v2 新增。

9. _to_df dropna 在 vol=null 时丢整行 → v2 修复。

10. 方向判定不要求量能达标 → v2 收紧。

【评分重构（合计 100）】
- trend         26  （4H 强趋势 + ADX + EMA 发散度）
- platform      18  （整理平台收窄 + ATR 归一化）
- contraction   14  （缩量程度）
- breakout      18  （突破 + 收盘强度 + 突破幅度/ATR）
- volume        10  （启动量比 + z-score）
- macd           6  （MACD 方向确认，新增）
- rsi            4  （RSI 合理区间 bonus，新增）
- extension      4  （延伸适中 bonus，ATR 归一化）
─────────────────────────────────────────────────────
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from strategies._shared.indicators import (
    _check_df, _to_df, _ema, _atr, _adx, _rsi_wilder, _macd, _volume_zscore,
)

try:
    from src.scanner.base_scanner import BaseScannerStrategy, ScanCondition, ScannerSymbol
    from src.scanner.ranking import build_opportunity_profile
    _HAS_SCANNER_BASE = True
except ImportError:
    BaseScannerStrategy = object
    ScanCondition = None; ScannerSymbol = None
    build_opportunity_profile = None
    _HAS_SCANNER_BASE = False

_W_TREND       = 26
_W_PLATFORM    = 18
_W_CONTRACTION = 14
_W_BREAKOUT    = 18
_W_VOLUME      = 10
_W_MACD        =  6
_W_RSI         =  4
_W_EXTENSION   =  4

_ADX_MIN = 18.0

_DEFAULT_CONFIG = {
    'min_score': 70, 'min_volume_24h': 18_000_000,
    'runup_lookback_bars': 40, 'min_prior_runup_pct': 12.0,
    'consolidation_bars': 12, 'max_base_width_pct': 4.8,
    'max_base_width_atr': 2.5,
    'max_contraction_ratio': 0.82, 'breakout_buffer_pct': 0.2,
    'min_breakout_volume_ratio': 1.6, 'max_extension_atr': 2.0,
}


class ContinuationCompressionSwingScanner(BaseScannerStrategy if _HAS_SCANNER_BASE else object):
    required_bars = ['4H', '1H']
    name = "波段缩量中继再启动扫描"
    description = "4H 强趋势 → 1H 缩量整理 → 放量突破二次启动 + ADX/MACD/收盘强度确认"
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
            h4 = _to_df(self._get_klines(km, '4H'))
            h1 = _to_df(self._get_klines(km, '1H'))
            analysis = _analyze_core(h4, h1, symbol.last_price, self.config)
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
            'category': '缩量中继再启动', 'ranking_factors': analysis.get('ranking_factors', {}),
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
            'runup_lookback_bars':       {'type':'int',  'default':40,        'label':'前段涨/跌幅回望(4H根)'},
            'min_prior_runup_pct':       {'type':'float','default':12.0,      'label':'前段最小涨/跌幅%'},
            'consolidation_bars':        {'type':'int',  'default':12,        'label':'整理期窗口(1H根)'},
            'max_base_width_pct':        {'type':'float','default':4.8,       'label':'平台最大宽度%'},
            'max_base_width_atr':        {'type':'float','default':2.5,       'label':'平台最大宽度(ATR)'},
            'max_contraction_ratio':     {'type':'float','default':0.82,      'label':'缩量系数上限'},
            'breakout_buffer_pct':       {'type':'float','default':0.2,       'label':'突破缓冲%'},
            'min_breakout_volume_ratio': {'type':'float','default':1.6,       'label':'启动最小量比'},
            'max_extension_atr':         {'type':'float','default':2.0,       'label':'最大延伸(ATR)'},
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
    _check_df(h4, '4H', 90); _check_df(h1, '1H', 120)
    score = 0.0; signals = []
    price = float(last_price) if last_price and last_price > 0 else float(h1['c'].iloc[-1])

    # 均线
    ema21_4h = _ema(h4['c'], 21); ema55_4h = _ema(h4['c'], 55)
    ema21_1h = _ema(h1['c'], 21)
    h4_adx = _adx(h4, 14)
    h4_atr = _atr(h4); h1_atr = _atr(h1)
    h4_ema_spread = (ema21_4h - ema55_4h) / ema55_4h * 100 if ema55_4h > 0 else 0.0
    h1_rsi = _rsi_wilder(h1['c'])
    _, _, h1_macd_hist = _macd(h1['c'])
    vol_zscore = _volume_zscore(h1['vol'])

    # 前段主升/主跌
    runup_lb = int(cfg.get('runup_lookback_bars', 40))
    min_runup = float(cfg.get('min_prior_runup_pct', 12))
    swing_low = float(h4['c'].tail(runup_lb).min())
    swing_high = float(h4['c'].tail(runup_lb).max())
    runup_pct = (price - swing_low) / swing_low * 100 if swing_low > 0 else 0.0
    rundown_pct = (swing_high - price) / swing_high * 100 if swing_high > 0 else 0.0

    # 方向判定
    bullish = (price > ema21_4h > ema55_4h and price > ema55_4h
               and runup_pct >= min_runup and h4_adx >= _ADX_MIN
               and h4_ema_spread >= 0.20)
    bearish = (price < ema21_4h < ema55_4h and price < ema55_4h
               and rundown_pct >= min_runup and h4_adx >= _ADX_MIN
               and h4_ema_spread <= -0.20)

    is_bull = bullish  # 最终方向
    trend_runup = runup_pct if bullish else rundown_pct if bearish else 0.0

    # ① 趋势质量（26 分）
    if bullish or bearish:
        base_t = 18.0
        adx_bonus = min(4.0, max(0.0, (h4_adx - _ADX_MIN) / 12.0 * 4.0))
        spread_bonus = min(4.0, abs(h4_ema_spread) / 2.0 * 4.0)
        ts_ = base_t + adx_bonus + spread_bonus
        score += ts_
        dl = "多头" if bullish else "空头"
        signals.append(f"4H{dl}强趋势(前段{'涨' if bullish else '跌'}{trend_runup:.1f}%, "
                       f"ADX {h4_adx:.1f} → +{ts_:.0f}分)")
    else:
        rp = []
        if not (price > ema21_4h > ema55_4h or price < ema21_4h < ema55_4h):
            rp.append("EMA排列不符")
        if runup_pct < min_runup and rundown_pct < min_runup:
            rp.append(f"前段涨/跌幅不足({max(runup_pct, rundown_pct):.1f}%)")
        if h4_adx < _ADX_MIN:
            rp.append(f"ADX不足({h4_adx:.1f})")
        signals.append("趋势不符: " + " | ".join(rp))
        return _build_result(
            valid=True, score=score, direction='WAIT', signals=signals,
            runup_pct=trend_runup, base_width_pct=0.0, base_width_atr=0.0,
            contraction_ratio=1.0, breakout_vol_ratio=0.0,
            extension_atr=0.0, h4_adx=h4_adx, h4_ema_spread=h4_ema_spread,
            h1_rsi=h1_rsi, vol_zscore=vol_zscore, strong_trend=False, is_bull=True)

    # 整理平台
    cb = int(cfg.get('consolidation_bars', 12))
    base_slice = h1.tail(cb)
    base_high = float(base_slice['h'].max()); base_low = float(base_slice['l'].min())
    base_mid = (base_high + base_low) / 2.0
    base_width_pct = (base_high - base_low) / base_mid * 100 if base_mid > 0 else 999.0
    base_width_atr = (base_high - base_low) / h1_atr if h1_atr > 0 else 999.0

    # ② 平台收窄（18 分）
    max_w_pct = float(cfg.get('max_base_width_pct', 4.8))
    max_w_atr = float(cfg.get('max_base_width_atr', 2.5))
    tight = base_width_pct <= max_w_pct and base_width_atr <= max_w_atr
    if tight:
        tightness_pct = max(0.0, 1.0 - base_width_pct / max_w_pct) * 0.5
        tightness_atr = max(0.0, 1.0 - base_width_atr / max_w_atr) * 0.5
        ps = _W_PLATFORM * (0.5 + tightness_pct + tightness_atr)
        score += ps
        signals.append(f"整理平台收窄({base_width_pct:.1f}%/{base_width_atr:.1f}ATR → +{ps:.0f}分)")
    else:
        signals.append(f"整理区间过宽({base_width_pct:.1f}%/{base_width_atr:.1f}ATR)")

    # 量能分离
    baseline_start = cb + 20
    prior_slice = h1['vol'].tail(baseline_start).head(20)
    prior_vol = float(prior_slice.median()) if not prior_slice.empty else 0.0  # v2: 中位数
    consol_slice = h1['vol'].tail(cb + 1).iloc[:-1]
    consol_vol = float(consol_slice.mean()) if not consol_slice.empty else prior_vol
    breakout_bar_vol = float(h1['vol'].iloc[-1])
    contraction_ratio = consol_vol / prior_vol if prior_vol > 0 else 1.0
    breakout_vol_ratio = breakout_bar_vol / prior_vol if prior_vol > 0 else 1.0

    # ③ 缩量（14 分）
    max_cr = float(cfg.get('max_contraction_ratio', 0.82))
    if contraction_ratio <= max_cr:
        cs = _W_CONTRACTION * min(1.0, 0.5 + (max_cr - contraction_ratio) / max(max_cr - 0.3, 0.1) * 0.5)
        score += cs
        signals.append(f"缩量({contraction_ratio:.2f}x → +{cs:.0f}分)")
    else:
        signals.append(f"未缩量({contraction_ratio:.2f}x)")

    # ④ 突破确认 + 收盘强度（18 分）
    buf = float(cfg.get('breakout_buffer_pct', 0.2)) / 100.0
    ref_slice = h1.tail(cb + 1).iloc[:-1]
    last_close = float(h1['c'].iloc[-1]); prev_close = float(h1['c'].iloc[-2])

    if is_bull:
        breakout_level = float(ref_slice['h'].max()) if not ref_slice.empty else base_high
        trigger = breakout_level * (1.0 + buf)
        prev_in_range = base_low * 0.995 <= prev_close <= base_high * 1.005
        breakout_ok = prev_in_range and last_close > trigger and last_close > ema21_1h
        breakout_magnitude = (last_close - trigger) / h1_atr if h1_atr > 0 else 0.0
    else:
        breakout_level = float(ref_slice['l'].min()) if not ref_slice.empty else base_low
        trigger = breakout_level * (1.0 - buf)
        prev_in_range = base_low * 0.995 <= prev_close <= base_high * 1.005
        breakout_ok = prev_in_range and last_close < trigger and last_close < ema21_1h
        breakout_magnitude = (trigger - last_close) / h1_atr if h1_atr > 0 else 0.0

    # 收盘强度
    last_o = float(h1['o'].iloc[-1]); last_h = float(h1['h'].iloc[-1]); last_l = float(h1['l'].iloc[-1])
    bar_range = last_h - last_l
    if bar_range > 0:
        close_strength = abs(last_close - last_o) / bar_range
    else:
        close_strength = 0.0

    if breakout_ok and breakout_magnitude >= 0.15:
        base_bk = _W_BREAKOUT * 0.6
        strength_bonus = _W_BREAKOUT * 0.25 * min(1.0, close_strength / 0.6)
        mag_bonus = _W_BREAKOUT * 0.15 * min(1.0, breakout_magnitude / 0.8)
        bks = base_bk + strength_bonus + mag_bonus
        score += bks
        arrow = "向上突破" if is_bull else "向下跌破"
        signals.append(f"1H{arrow}(强度{close_strength:.2f}, 幅度{breakout_magnitude:.2f}ATR → +{bks:.0f}分)")
    elif breakout_ok:
        bks = _W_BREAKOUT * 0.4
        score += bks
        signals.append(f"突破幅度偏小({breakout_magnitude:.2f}ATR → +{bks:.0f}分)")
    else:
        rp = []
        if not prev_in_range: rp.append("前一棒不在整理区")
        if is_bull and last_close <= trigger: rp.append(f"未突破触发位({trigger:.4g})")
        elif not is_bull and last_close >= trigger: rp.append(f"未跌破触发位({trigger:.4g})")
        signals.append("未启动: " + " | ".join(rp) if rp else "尚未二次启动")

    # ⑤ 量能（10 分）
    min_vr = float(cfg.get('min_breakout_volume_ratio', 1.6))
    vrok = breakout_vol_ratio >= min_vr
    vzok = vol_zscore >= 0.8
    if vrok and vzok:
        vs = _W_VOLUME * 0.9
        score += vs; signals.append(f"启动放量({breakout_vol_ratio:.2f}x, z={vol_zscore:+.2f} → +{vs:.0f}分)")
    elif vrok:
        vs = _W_VOLUME * 0.6
        score += vs; signals.append(f"量比达标({breakout_vol_ratio:.2f}x → +{vs:.0f}分)")
    elif vzok:
        vs = _W_VOLUME * 0.4
        score += vs
    else:
        signals.append(f"量能不足({breakout_vol_ratio:.2f}x)")

    # ⑥ MACD 方向（6 分）
    macd_ok = False
    if len(h1_macd_hist) >= 2:
        mh = float(h1_macd_hist.iloc[-1]); mp = float(h1_macd_hist.iloc[-2])
        if is_bull and mh > 0 and mh > mp: macd_ok = True
        elif not is_bull and mh < 0 and mh < mp: macd_ok = True
    if macd_ok:
        score += _W_MACD; signals.append(f"MACD方向确认(+{_W_MACD}分)")
    elif breakout_ok and len(h1_macd_hist) >= 1:
        mh = float(h1_macd_hist.iloc[-1])
        if (is_bull and mh > 0) or (not is_bull and mh < 0):
            score += _W_MACD * 0.4

    # ⑦ RSI 合理（4 分 bonus）
    rsi_ok = False
    if is_bull and 50 <= h1_rsi <= 75: rsi_ok = True
    elif not is_bull and 25 <= h1_rsi <= 50: rsi_ok = True
    if rsi_ok:
        score += _W_RSI; signals.append(f"RSI合理({h1_rsi:.1f} → +{_W_RSI}分)")
    elif (is_bull and h1_rsi > 82) or (not is_bull and h1_rsi < 18):
        signals.append(f"RSI极端({'超买' if is_bull else '超卖'}: {h1_rsi:.1f})")

    # ⑧ 延伸适中（4 分 bonus，ATR 归一化）
    max_ext = float(cfg.get('max_extension_atr', 2.0))
    if is_bull:
        ext_atr = (last_close - base_low) / h4_atr if h4_atr > 0 else 0.0
    else:
        ext_atr = (base_high - last_close) / h4_atr if h4_atr > 0 else 0.0
    ext_atr = max(ext_atr, 0.0)
    if ext_atr <= max_ext:
        es = _W_EXTENSION * max(0.0, 1.0 - ext_atr / max(max_ext, 0.1))
        if es >= 1.0:
            score += es; signals.append(f"延伸适中({ext_atr:.1f}ATR → +{es:.0f}分)")
    else:
        signals.append(f"延伸偏大({ext_atr:.1f}ATR)")

    # 方向判定 — v2 收紧：要求量能达标
    direction = 'WAIT'
    if breakout_ok and vrok:
        direction = 'BUY' if is_bull else 'SELL'

    return _build_result(
        valid=True, score=score, direction=direction, signals=signals,
        runup_pct=trend_runup, base_width_pct=base_width_pct, base_width_atr=base_width_atr,
        contraction_ratio=contraction_ratio, breakout_vol_ratio=breakout_vol_ratio,
        extension_atr=ext_atr, h4_adx=h4_adx, h4_ema_spread=h4_ema_spread,
        h1_rsi=h1_rsi, vol_zscore=vol_zscore, strong_trend=True, is_bull=is_bull)


# ══════════════════════════════════════════════
# 底层工具
# ══════════════════════════════════════════════
def _build_result(*, valid, score, direction, signals, runup_pct, base_width_pct, base_width_atr,
                  contraction_ratio, breakout_vol_ratio, extension_atr, h4_adx, h4_ema_spread,
                  h1_rsi, vol_zscore, strong_trend, is_bull, reason=''):
    lq = max(25, 100 - base_width_pct * 10); fq = max(25, 100 - max(extension_atr - 0.5, 0) * 25)
    vq = min(breakout_vol_ratio / 1.6, 1.7) * 58
    return {
        'valid': valid, 'reason': reason, 'score': max(score, 0.0),
        'direction': direction, 'signals': signals,
        'ranking_factors': {
            'trend': 94.0 if strong_trend and h4_adx >= _ADX_MIN else 28.0,
            'trigger': 92.0 if direction in {'BUY','SELL'} else 30.0,
            'volume': vq, 'location': lq, 'freshness': fq,
            'risk': 88.0 if contraction_ratio <= 0.82 else 54.0,
        },
        'details': {
            '评估': ' | '.join(signals) if signals else '暂无缩量中继机会',
            '方向': '多头' if is_bull else '空头',
            '前段涨/跌幅': f'{runup_pct:.2f}%',
            '平台宽度%': f'{base_width_pct:.2f}%',
            '平台宽度ATR': f'{base_width_atr:.1f}',
            '缩量系数': f'{contraction_ratio:.2f}',
            '启动量比': f'{breakout_vol_ratio:.2f}x',
            '量能Z分': f'{vol_zscore:+.2f}σ',
            '4H_ADX': f'{h4_adx:.1f}',
            '4H_EMA发散': f'{h4_ema_spread:+.2f}%',
            '1H_RSI': f'{h1_rsi:.1f}',
            '延伸ATR': f'{extension_atr:.1f}',
        },
    }

STRATEGY_NAME  = "波段缩量中继再启动扫描"
STRATEGY_TYPE  = "scan"
STRATEGY_CLASS = ContinuationCompressionSwingScanner
BACKTEST_CLASS = ContinuationCompressionSwingScanner
