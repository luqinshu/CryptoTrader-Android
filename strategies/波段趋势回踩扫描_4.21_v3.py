"""
经典波段策略：趋势回踩确认扫描（精修版 v3）

v2 → v3 变更摘要
─────────────────────────────────────────────────────
【逻辑修复】
1. 回踩方向性缺失：v2 用 abs() 算回踩距离，上升趋势中价格已跌破均线到
   下方也会被判为"回踩到位"。v3 区分方向：多头只认 price ≥ EMA21 且贴近；
   空头只认 price ≤ EMA21 且贴近。穿到对侧视为趋势破坏。
2. 回踩真实性缺失：v2 没有验证"是否真的发生过回踩过程"。价格一直在均线
   远上方的品种也可能偶然某一根 close 接近均线就拿到 25 分。v3 新增近 N
   根 1H 至少 k 根触及过 EMA21 的时间维度校验。
3. _volume_ratio 包含当前 bar：v2 用 `vol.tail(window).mean()` 计算基线，
   把"正在形成的 bar"也算进均值了。v3 改为 `vol.iloc[-(window+1):-1].mean()`
   严格排除当前 bar，且加了未收盘 bar 的按进度外推修正。
4. 3m 状态机 Phase 2 用 bar_l / bar_h 做硬裁决：3m 级别噪声极大，
   一根影线就 kill 信号。v3 改用 close 做判断，容忍影线。
5. 3m 起点搜索窗口用前 2/3 导致状态机迭代空间不足（与趋势策略同问题）。
   v3 改为前 1/2。
6. _to_df 的 dropna() 在 vol 偶尔为 null 时整行丢弃，v3 改为仅对
   价格列 dropna，vol 补 0。
7. 趋势判定没有区分"趋势"和"震荡"：v2 只看 EMA 排列 + trend_quality，
   在震荡市 EMA 排列也可能偶然成立。v3 加入 ADX + EMA 发散度硬门槛。

【新增指标】
8.  ADX(14)：日线、4H 各一个。< 18 视为震荡，直接 WAIT。
9.  EMA 发散度 (ema21-ema55)/ema55 百分比。
10. 收盘强度 (c-l)/(h-l)：1H 启动 bar 的买/卖盘决心，纳入确认评分。
11. 成交量 z-score（log 空间）：与量比双重门槛，低波动标的不再被假突破。
12. 4H swing high/low（局部极值）：替代固定 lookback 窗口 max/min 作为
    "前高/前低"关键位。新增为独立 bonus（+8 分）。

【评分重构（总分仍 100，3m 为通过门槛不加分）】
- trend        30  （含 EMA 排列 + ADX + 发散度）
- pullback     20  （方向性 + ATR 归一化）
- retest_time   5  （回踩时间真实性，新增）
- keylevel      8  （4H swing 前高/前低 bonus，新增）
- confirm      17  （1H 穿越 + 收盘强度）
- volume       12  （量比 + z-score）
- rsi           8  （RSI 健康度 bonus）
─────────────────────────────────────────────────────
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from strategies._shared.indicators import (_to_df, _check_df, _ema, _rsi_wilder, _atr, _adx,
                                          _efficiency_ratio, _volume_ratio_adjusted,
                                          _volume_zscore, _latest_swing_levels,
                                          _local_trend_snapshot)

try:
    from src.scanner.base_scanner import BaseScannerStrategy, ScanCondition, ScannerSymbol
    from src.scanner.ranking import build_opportunity_profile
    from src.scanner.trend_quality import trend_quality_snapshot as _external_trend_snapshot
    _HAS_SCANNER_BASE = True
except ImportError:
    BaseScannerStrategy = object
    ScanCondition = None
    ScannerSymbol = None
    build_opportunity_profile = None
    _external_trend_snapshot = None
    _HAS_SCANNER_BASE = False

# ──────────────────────────────────────────────
# 评分权重（合计 100；3m 为通过门槛不加分）
# ──────────────────────────────────────────────
_W_TREND       = 30
_W_PULLBACK    = 20
_W_RETEST_TIME =  5   # 新增
_W_KEYLEVEL    =  8   # 新增
_W_CONFIRM     = 17
_W_VOLUME      = 12
_W_RSI         =  8

_ADX_MIN_TREND     = 18.0
_EMA_SPREAD_MIN    = 0.20
_RETEST_LOOKBACK   = 12
_RETEST_MIN_TOUCHES = 2

_DEFAULT_CONFIG: Dict[str, Any] = {
    'min_score':                68,
    'min_volume_24h':           12_000_000,
    'max_pullback_pct':         3.2,
    'max_pullback_atr':         1.5,
    'min_confirm_volume_ratio': 1.3,
    'min_volume_zscore':        0.8,
    'require_3m_continuation':  True,
    'm3_window':                60,
    'm3_neckline_bars':         8,
    'm3_min_pullback_bars':     3,
    'm3_breakout_buffer_pct':   0.10,
    'm3_origin_tolerance_pct':  0.15,
}


# ══════════════════════════════════════════════
# 扫描器
# ══════════════════════════════════════════════

class TrendPullbackSwingScanner(BaseScannerStrategy if _HAS_SCANNER_BASE else object):
    required_bars = ['1D', '4H', '1H', '3m']
    name        = "波段趋势回踩扫描"
    description = "1D/4H 波段趋势 → ATR 归一化回踩 → 1H 穿越确认 → 3m 三段式结构"
    strategy_type = "scan"

    def __init__(self, config=None):
        self.config = {**_DEFAULT_CONFIG, **(config or {})}
        if _HAS_SCANNER_BASE and hasattr(super(), '__init__'):
            try:
                super().__init__(config or {})
            except Exception:
                pass

    def _init_conditions(self):
        if ScanCondition is None:
            return
        self.add_condition(ScanCondition(
            name="24H成交量", description="过滤流动性不足标的",
            field="volume_24h", operator=">=",
            value=self.config.get('min_volume_24h', 12_000_000),
        ))

    def scan_symbol(self, symbol) -> Dict:
        klines_map = symbol.extra_data.get('klines', {})
        try:
            d1 = _to_df(self._get_klines(klines_map, '1D'))
            h4 = _to_df(self._get_klines(klines_map, '4H'))
            h1 = _to_df(self._get_klines(klines_map, '1H'))
            m3 = _to_df(self._get_klines(klines_map, '3m'))
            analysis = _analyze_core(d1, h4, h1, m3, symbol.last_price, self.config)
        except Exception as exc:
            return {
                'symbol': symbol.inst_id, 'passed': False, 'score': 0.0,
                'direction': 'WAIT', 'details': {'状态': f'分析异常: {exc}'},
            }
        if not analysis['valid']:
            return {
                'symbol': symbol.inst_id, 'passed': False, 'score': 0.0,
                'direction': 'WAIT', 'details': {'状态': analysis.get('reason', '')},
            }
        min_score = float(self.config.get('min_score', 68))
        passed = analysis['score'] >= min_score and analysis['direction'] in {'BUY', 'SELL'}
        result = {
            'symbol': symbol.inst_id, 'passed': passed,
            'score': round(analysis['score'], 2),
            'direction': analysis['direction'],
            'signals': analysis['signals'],
            'details': analysis['details'],
            'last_price': symbol.last_price,
            'volume_24h': symbol.volume_24h,
            'price_change_24h': symbol.price_change_24h,
            'category': '波段趋势回踩',
            'ranking_factors': analysis.get('ranking_factors', {}),
        }
        if build_opportunity_profile:
            try:
                profile = build_opportunity_profile(
                    base_score=analysis['score'], direction=analysis['direction'],
                    volume_24h=symbol.volume_24h,
                    factors=analysis.get('ranking_factors', {}),
                    signals=analysis['signals'],
                )
                result.update(profile)
            except Exception:
                pass
        return result

    def _get_klines(self, klines_map, bar):
        return klines_map.get(bar) or klines_map.get(bar.lower()) or klines_map.get(bar.upper()) or []

    def get_config_schema(self):
        return {
            'min_score':                 {'type': 'int',   'default': 68,         'label': '最低通过分数(0-100)'},
            'min_volume_24h':            {'type': 'float', 'default': 12_000_000, 'label': '最小24H成交额'},
            'max_pullback_pct':          {'type': 'float', 'default': 3.2,        'label': '最大回踩距离%（价格）'},
            'max_pullback_atr':          {'type': 'float', 'default': 1.5,        'label': '最大回踩距离（ATR倍数）'},
            'min_confirm_volume_ratio':  {'type': 'float', 'default': 1.3,        'label': '确认最小量比'},
            'min_volume_zscore':         {'type': 'float', 'default': 0.8,        'label': '启动量 z-score 下限'},
            'require_3m_continuation':   {'type': 'bool',  'default': True,       'label': '要求3m三段式趋势延续确认'},
            'm3_window':                 {'type': 'int',   'default': 60,         'label': '3m观察窗口(根数)'},
            'm3_neckline_bars':          {'type': 'int',   'default': 8,          'label': '3m颈线样本数'},
            'm3_min_pullback_bars':      {'type': 'int',   'default': 3,          'label': '3m最少回踩根数'},
            'm3_breakout_buffer_pct':    {'type': 'float', 'default': 0.10,       'label': '3m突破缓冲%'},
            'm3_origin_tolerance_pct':   {'type': 'float', 'default': 0.15,       'label': '3m起点容忍偏离%'},
        }


# ══════════════════════════════════════════════
# 纯函数公开 API
# ══════════════════════════════════════════════

def analyze_bars(d1, h4, h1, m3, last_price, config=None):
    cfg = {**_DEFAULT_CONFIG, **(config or {})}
    try:
        return _analyze_core(d1, h4, h1, m3, last_price, cfg)
    except Exception as exc:
        return {'valid': False, 'reason': f'分析异常: {exc}', 'score': 0.0,
                'direction': 'WAIT', 'signals': [], 'details': {}, 'ranking_factors': {}}

def klines_list_to_df(rows):
    return _to_df(rows)


# ══════════════════════════════════════════════
# 核心分析
# ══════════════════════════════════════════════

def _analyze_core(d1, h4, h1, m3, last_price, cfg):
    _check_df(d1, '日线', 90)
    _check_df(h4, '4H', 90)
    _check_df(h1, '1H', 120)

    require_3m = bool(cfg.get('require_3m_continuation', True))
    min_3m_bars = max(
        int(cfg.get('m3_window', 60)),
        int(cfg.get('m3_neckline_bars', 8)) + int(cfg.get('m3_min_pullback_bars', 3)) + 6,
    )
    if require_3m and len(m3) < min_3m_bars:
        _check_df(m3, '3m', min_3m_bars)

    price = float(last_price) if last_price and last_price > 0 else float(h1['c'].iloc[-1])
    score = 0.0
    signals = []

    # 均线
    d1_ema21 = _ema(d1['c'], 21); d1_ema55 = _ema(d1['c'], 55)
    h4_ema21 = _ema(h4['c'], 21); h4_ema55 = _ema(h4['c'], 55)
    h1_ema21 = _ema(h1['c'], 21); h1_ema55 = _ema(h1['c'], 55)
    h1_last_close = float(h1['c'].iloc[-1])
    h1_prev_close = float(h1['c'].iloc[-2])

    # 辅助指标
    h1_rsi     = _rsi_wilder(h1['c'])
    h4_atr     = _atr(h4)
    vol_ratio  = _volume_ratio_adjusted(h1)
    vol_zscore = _volume_zscore(h1['vol'])
    d1_adx     = _adx(d1, 14)
    h4_adx     = _adx(h4, 14)

    d1_ema_spread = (d1_ema21 - d1_ema55) / d1_ema55 * 100 if d1_ema55 > 0 else 0.0
    h4_ema_spread = (h4_ema21 - h4_ema55) / h4_ema55 * 100 if h4_ema55 > 0 else 0.0

    # 趋势快照
    trend_snap = _trend_snapshot(d1, h4, h1, price)
    trend_metrics = trend_snap.get('metrics', {})
    trend_long_score = float(trend_snap.get('long_score', 0) or 0)
    trend_short_score = float(trend_snap.get('short_score', 0) or 0)

    _m3ph = _m3_default_result('趋势未确认')

    # 趋势方向（含 ADX + 发散度门槛）
    bullish = (
        bool(trend_snap.get('long_ok'))
        and price > d1_ema21 > d1_ema55
        and price > h4_ema21 > h4_ema55
        and d1_adx >= _ADX_MIN_TREND and h4_adx >= _ADX_MIN_TREND
        and d1_ema_spread >= _EMA_SPREAD_MIN and h4_ema_spread >= _EMA_SPREAD_MIN
    )
    bearish = (
        bool(trend_snap.get('short_ok'))
        and price < d1_ema21 < d1_ema55
        and price < h4_ema21 < h4_ema55
        and d1_adx >= _ADX_MIN_TREND and h4_adx >= _ADX_MIN_TREND
        and d1_ema_spread <= -_EMA_SPREAD_MIN and h4_ema_spread <= -_EMA_SPREAD_MIN
    )

    # ① 趋势（30 分）
    if bullish or bearish:
        base_t = 22.0
        spread_abs = (abs(d1_ema_spread) + abs(h4_ema_spread)) / 2.0
        spread_bonus = min(4.0, spread_abs / 2.0 * 4.0)
        adx_avg = (d1_adx + h4_adx) / 2.0
        adx_bonus = min(4.0, max(0.0, (adx_avg - _ADX_MIN_TREND) / 12.0 * 4.0))
        ts_ = base_t + spread_bonus + adx_bonus
        score += ts_
        dl = "多头" if bullish else "空头"
        rs = trend_long_score if bullish else trend_short_score
        signals.append(f"{dl}趋势通过(质量{rs:.0f}, ADX {adx_avg:.1f}, 发散{spread_abs:.2f}% → +{ts_:.1f}分)")
    else:
        rp = []
        if not (bool(trend_snap.get('long_ok')) or bool(trend_snap.get('short_ok'))):
            rp.append(str(trend_snap.get('reason', '趋势质量不足')))
        if d1_adx < _ADX_MIN_TREND: rp.append(f"日线ADX不足({d1_adx:.1f})")
        if h4_adx < _ADX_MIN_TREND: rp.append(f"4H ADX不足({h4_adx:.1f})")
        if abs(d1_ema_spread) < _EMA_SPREAD_MIN: rp.append(f"日线EMA发散不足({d1_ema_spread:.2f}%)")
        if abs(h4_ema_spread) < _EMA_SPREAD_MIN: rp.append(f"4H EMA发散不足({h4_ema_spread:.2f}%)")
        signals.append("趋势未确认: " + " | ".join(rp) if rp else "趋势未确认")
        return _build_result(valid=True, score=score, direction='WAIT', signals=signals,
            pullback_pct=0.0, pb_atr=0.0, vol_ratio=vol_ratio, vol_zscore=vol_zscore,
            close_strength=0.0, h1_rsi=h1_rsi, d1_adx=d1_adx, h4_adx=h4_adx,
            d1_ema_spread=d1_ema_spread, h4_ema_spread=h4_ema_spread,
            trend_snap=trend_snap, trend_metrics=trend_metrics, m3_result=_m3ph)

    # ② 方向性回踩 + ATR 归一化（20 分）
    raw_distance = (price - h4_ema21) / h4_ema21 * 100 if h4_ema21 > 0 else 999.0
    pb_atr = abs(price - h4_ema21) / h4_atr if h4_atr > 0 else 999.0
    max_pb = float(cfg.get('max_pullback_pct', 3.2))
    max_pb_atr = float(cfg.get('max_pullback_atr', 1.5))

    if bullish:
        pullback_ok = 0.0 <= raw_distance <= max_pb and pb_atr <= max_pb_atr
        pullback_depth = abs(raw_distance)
        direction_mismatch = raw_distance < 0
    else:
        pullback_ok = -max_pb <= raw_distance <= 0.0 and pb_atr <= max_pb_atr
        pullback_depth = abs(raw_distance)
        direction_mismatch = raw_distance > 0

    if pullback_ok:
        pb_score = _pullback_score(pullback_depth, pb_atr, max_pb, max_pb_atr)
        score += pb_score
        signals.append(f"方向性回踩({pullback_depth:.2f}%, {pb_atr:.2f}ATR → +{pb_score:.1f}分)")
    elif direction_mismatch:
        signals.append(f"价格已穿越EMA21至反向侧({raw_distance:+.2f}%)，非健康回踩")
    else:
        signals.append(f"回踩过远({pullback_depth:.2f}%, {pb_atr:.2f}ATR)")

    # ③ 回踩时间真实性（5 分）
    h1_ema21_series = h1['c'].ewm(span=21, adjust=False).mean()
    rw = h1.tail(_RETEST_LOOKBACK)
    ew = h1_ema21_series.tail(_RETEST_LOOKBACK)
    if bullish:
        touch_count = int((rw['l'].values <= ew.values).sum())
    else:
        touch_count = int((rw['h'].values >= ew.values).sum())

    if touch_count >= _RETEST_MIN_TOUCHES:
        rt = _W_RETEST_TIME * min(1.0, touch_count / 5.0)
        score += rt
        signals.append(f"回踩真实(近{_RETEST_LOOKBACK}根触碰{touch_count}根 → +{rt:.1f}分)")
    else:
        signals.append(f"回踩过程不足(仅{touch_count}根触及)")

    # ④ 4H swing 关键位 bonus（8 分）
    sh, sl = _latest_swing_levels(h4, left=5, right=5, skip_recent=3, max_lookback=80)
    retest_quality = 0.0
    max_kl = 2.2
    if bullish and sh and sh > 0 and sh < price:
        retest_quality = (price - sh) / sh * 100
        if retest_quality <= max_kl:
            kls = _W_KEYLEVEL * max(0.5, 1.0 - retest_quality / max_kl)
            score += kls
            signals.append(f"回踩前swing高({retest_quality:.2f}% → +{kls:.1f}分)")
    elif bearish and sl and sl > 0 and sl > price:
        retest_quality = (sl - price) / sl * 100
        if retest_quality <= max_kl:
            kls = _W_KEYLEVEL * max(0.5, 1.0 - retest_quality / max_kl)
            score += kls
            signals.append(f"反抽前swing低({retest_quality:.2f}% → +{kls:.1f}分)")

    # ⑤ 1H 穿越确认 + 收盘强度（17 分）
    confirm_bull = (bullish and h1_prev_close <= h1_ema21
                    and h1_last_close > h1_ema21 and h1_ema21 > h1_ema55)
    confirm_bear = (bearish and h1_prev_close >= h1_ema21
                    and h1_last_close < h1_ema21 and h1_ema21 < h1_ema55)

    last_h1 = h1.iloc[-1]
    bar_range = float(last_h1['h']) - float(last_h1['l'])
    if bar_range > 0:
        close_strength = ((float(last_h1['c']) - float(last_h1['l'])) / bar_range
                          if bullish else
                          (float(last_h1['h']) - float(last_h1['c'])) / bar_range)
    else:
        close_strength = 0.5

    if confirm_bull or confirm_bear:
        base_c = _W_CONFIRM * 0.7
        strength_bonus = _W_CONFIRM * 0.3 * max(0.0, (close_strength - 0.5) / 0.5)
        cs = base_c + strength_bonus
        score += cs
        arrow = "站上" if confirm_bull else "跌破"
        signals.append(f"1H收盘{arrow}EMA21(强度{close_strength:.2f} → +{cs:.1f}分)")
    else:
        parts = []
        cc = ((h1_prev_close <= h1_ema21 and h1_last_close > h1_ema21) if bullish
              else (h1_prev_close >= h1_ema21 and h1_last_close < h1_ema21))
        if not cc:
            parts.append(f"穿越未触发(prev={h1_prev_close:.4g},last={h1_last_close:.4g},ema={h1_ema21:.4g})")
        co = (h1_ema21 > h1_ema55) if bullish else (h1_ema21 < h1_ema55)
        if not co:
            parts.append("1H EMA排列不支持")
        signals.append("未确认穿越: " + " | ".join(parts) if parts else "1H穿越未触发")

    # ⑥ 量能（12 分）— 量比 + z-score 双重
    min_vr = float(cfg.get('min_confirm_volume_ratio', 1.3))
    min_zs = float(cfg.get('min_volume_zscore', 0.8))
    vrok = vol_ratio >= min_vr
    vzok = vol_zscore >= min_zs
    if vrok and vzok:
        rc = 0.55 + min(1.0, (vol_ratio - min_vr) / max(min_vr, 0.1) * 0.25) * 0.45
        zc = 0.55 + min(1.0, (vol_zscore - min_zs) / 1.5) * 0.45
        vs = _W_VOLUME * (rc + zc) / 2.0
        score += vs
        signals.append(f"量能强({vol_ratio:.2f}x, z={vol_zscore:+.2f} → +{vs:.1f}分)")
    elif vrok or vzok:
        vs = _W_VOLUME * 0.45
        score += vs
        signals.append(f"量能部分({vol_ratio:.2f}x, z={vol_zscore:+.2f} → +{vs:.1f}分)")
    else:
        signals.append(f"量能不足({vol_ratio:.2f}x, z={vol_zscore:+.2f})")

    # ⑦ RSI 健康度 bonus（8 分）
    rsi_ok = (bullish and 45 <= h1_rsi <= 68) or (bearish and 32 <= h1_rsi <= 55)
    if rsi_ok:
        score += _W_RSI
        signals.append(f"RSI健康({'多' if bullish else '空'}头区间, {h1_rsi:.1f})")
    elif (bullish and h1_rsi > 75) or (bearish and h1_rsi < 25):
        signals.append(f"RSI警示({'超买' if bullish else '超卖'}: {h1_rsi:.1f})")

    # ⑧ 3m 三段式结构
    if confirm_bull or confirm_bear:
        m3_result = (
            _three_min_continuation_core(m3, bullish=bullish, cfg=cfg)
            if len(m3) >= min_3m_bars
            else _m3_default_result('3m数据不足')
        )
    else:
        m3_result = _m3_default_result('1H穿越未触发，跳过3m')

    if m3_result.get('passed'):
        signals.append(str(m3_result.get('reason')))
    elif m3_result.get('valid') and (confirm_bull or confirm_bear):
        signals.append(f"3m观察(Phase {m3_result.get('phase', 0)})：{m3_result.get('reason')}")

    # 方向判定
    m3_ok = bool(m3_result.get('passed')) or not require_3m
    direction = 'WAIT'
    if confirm_bull and pullback_ok and m3_ok:
        direction = 'BUY'
    elif confirm_bear and pullback_ok and m3_ok:
        direction = 'SELL'

    return _build_result(
        valid=True, score=score, direction=direction, signals=signals,
        pullback_pct=pullback_depth if pullback_ok or direction_mismatch else abs(raw_distance),
        pb_atr=pb_atr, vol_ratio=vol_ratio, vol_zscore=vol_zscore,
        close_strength=close_strength, h1_rsi=h1_rsi,
        d1_adx=d1_adx, h4_adx=h4_adx,
        d1_ema_spread=d1_ema_spread, h4_ema_spread=h4_ema_spread,
        trend_snap=trend_snap, trend_metrics=trend_metrics, m3_result=m3_result,
    )


# ══════════════════════════════════════════════
# 3m 状态机（纯函数版，v3 改用 close 做硬裁决）
# ══════════════════════════════════════════════

def _three_min_continuation_core(df, *, bullish, cfg):
    window = int(cfg.get('m3_window', 60))
    neckline_bars = int(cfg.get('m3_neckline_bars', 8))
    origin_tol = float(cfg.get('m3_origin_tolerance_pct', 0.15))
    breakout_buf = float(cfg.get('m3_breakout_buffer_pct', 0.10))
    min_pb_bars = int(cfg.get('m3_min_pullback_bars', 3))
    min_bars = neckline_bars + min_pb_bars + 6
    if len(df) < max(window, min_bars):
        return _m3_default_result(f'3m数据不足({len(df)}/{max(window, min_bars)})')
    recent = df.tail(window).reset_index(drop=True)
    n = len(recent)
    # 起点搜索前 1/2（修复 v2 的 2/3 窗口问题）
    ose = max(neckline_bars + min_pb_bars + 4, n // 2)
    if bullish:
        op = int(recent.iloc[:ose]['l'].idxmin())
        opr = float(recent['l'].iloc[op])
    else:
        op = int(recent.iloc[:ose]['h'].idxmax())
        opr = float(recent['h'].iloc[op])
    ns_ = op + 1
    ne = min(ns_ + neckline_bars, n - min_pb_bars - 3)
    if ne <= ns_ + 1:
        return _m3_default_result('3m颈线样本不足', origin_price=opr)
    neck = recent.iloc[ns_:ne]
    if bullish:
        bl = float(neck['h'].max())
        if not pd.notna(bl) or bl <= 0:
            return _m3_default_result('3m颈线异常', origin_price=opr)
        tl = bl * (1.0 + breakout_buf / 100.0)
        fl = opr * (1.0 - origin_tol / 100.0)
    else:
        bl = float(neck['l'].min())
        if not pd.notna(bl) or bl <= 0:
            return _m3_default_result('3m颈线异常(空)', origin_price=opr)
        tl = bl * (1.0 - breakout_buf / 100.0)
        fl = opr * (1.0 + origin_tol / 100.0)
    phase = 0; p1e = opr; cl_ = tl
    pbe = float('inf') if bullish else float('-inf')
    pbc = 0; p3c = 0.0
    for i in range(ne, n):
        bc = float(recent['c'].iloc[i])
        bh = float(recent['h'].iloc[i])
        bll = float(recent['l'].iloc[i])
        if bullish:
            if phase == 0:
                if bc > tl: phase = 1; p1e = bh; cl_ = bh
            elif phase == 1:
                if bc > tl:
                    if bh > p1e: p1e = bh; cl_ = bh
                else:
                    if bc < fl:
                        return _m3_result_failed(f'Phase2首棒收盘跌破起点', origin_price=opr,
                            breakout_level=bl, phase1_extreme=p1e, pb_extreme=bc, bullish=True)
                    phase = 2; pbe = bll; pbc = 1
            elif phase == 2:
                pbe = min(pbe, bll); pbc += 1
                if bc < fl:
                    return _m3_result_failed(f'Phase2收盘跌破起点', origin_price=opr,
                        breakout_level=bl, phase1_extreme=p1e, pb_extreme=pbe, bullish=True)
                if pbc >= min_pb_bars and bc > cl_:
                    phase = 3; p3c = bc; break
        else:
            if phase == 0:
                if bc < tl: phase = 1; p1e = bll; cl_ = bll
            elif phase == 1:
                if bc < tl:
                    if bll < p1e: p1e = bll; cl_ = bll
                else:
                    if bc > fl:
                        return _m3_result_failed(f'Phase2首棒收盘超起点', origin_price=opr,
                            breakout_level=bl, phase1_extreme=p1e, pb_extreme=bc, bullish=False)
                    phase = 2; pbe = bh; pbc = 1
            elif phase == 2:
                pbe = max(pbe, bh); pbc += 1
                if bc > fl:
                    return _m3_result_failed(f'Phase2收盘超起点', origin_price=opr,
                        breakout_level=bl, phase1_extreme=p1e, pb_extreme=pbe, bullish=False)
                if pbc >= min_pb_bars and bc < cl_:
                    phase = 3; p3c = bc; break
    lc = float(recent['c'].iloc[-1])
    if phase == 3:
        if bullish:
            dp = (pbe - opr) / opr * 100; cp = (p3c - cl_) / cl_ * 100; held = lc > p3c
        else:
            dp = (opr - pbe) / opr * 100; cp = (cl_ - p3c) / cl_ * 100; held = lc < p3c
        q = min(100.0, 60 + min(dp + 2, 4) * 5 + min(cp, 3) * 4 + (8 if held else 0))
        lb = "多头" if bullish else "空头"
        return {'valid': True, 'passed': True, 'quality': q,
                'reason': f"3m{lb}三段延续：突破→回踩({dp:+.2f}%)→继续(+{cp:.2f}%)",
                'phase': 3, 'origin_price': opr, 'breakout_level': bl,
                'phase1_extreme': p1e, 'pb_extreme': pbe, 'confirmation_level': cl_,
                'phase3_close': p3c, 'pb_depth_pct': dp, 'continuation_pct': cp}
    elif phase == 2:
        if bullish: dp = (pbe - opr) / opr * 100; fh = pbe >= fl
        else: dp = (opr - pbe) / opr * 100; fh = pbe <= fl
        q = 35 + 25 + (20 if fh else 0)
        r = f"3m已突破并回踩({dp:+.2f}%)，等待继续" if fh else f"3m回踩越过起点({dp:+.2f}%)"
        return {'valid': True, 'passed': False, 'quality': float(q), 'reason': r,
                'phase': 2, 'origin_price': opr, 'breakout_level': bl,
                'phase1_extreme': p1e, 'pb_extreme': pbe, 'confirmation_level': cl_,
                'phase3_close': 0.0, 'pb_depth_pct': dp, 'continuation_pct': 0.0}
    elif phase == 1:
        lb = "多头" if bullish else "空头"
        return {'valid': True, 'passed': False, 'quality': 60.0,
                'reason': f"3m{lb}已突破颈线，等回踩(极值={p1e:.6g})",
                'phase': 1, 'origin_price': opr, 'breakout_level': bl,
                'phase1_extreme': p1e, 'pb_extreme': float('nan'),
                'confirmation_level': cl_, 'phase3_close': 0.0,
                'pb_depth_pct': 0.0, 'continuation_pct': 0.0}
    else:
        return _m3_default_result(f'3m尚未完成初次突破', origin_price=opr)


# ══════════════════════════════════════════════
# 趋势快照
# ══════════════════════════════════════════════

def _trend_snapshot(d1, h4, h1, price):
    if _external_trend_snapshot:
        try: return _external_trend_snapshot(d1, h4, h1, price)
        except Exception: pass
    return _local_trend_snapshot(d1, h4, h1, price)


# ══════════════════════════════════════════════
# 底层工具
# ══════════════════════════════════════════════

def _ema_slope_pct(c, span, lb):
    e = c.ewm(span=span, adjust=False).mean()
    if len(e) <= lb: return 0.0
    b = float(e.iloc[-(lb + 1)]); l = float(e.iloc[-1])
    return (l / b - 1.0) * 100 if b > 0 else 0.0

def _pullback_score(pb_pct, pb_atr, max_pct, max_atr):
    norm_pct = pb_pct / max_pct
    norm_atr = pb_atr / max_atr
    worst = max(norm_pct, norm_atr)
    sc = _W_PULLBACK * (1 - 0.65 * worst)
    return round(max(sc, 5.0), 1)

def _m3_default_result(reason, origin_price=0.0):
    return {'valid': False, 'passed': False, 'quality': 35.0, 'reason': reason,
            'phase': 0, 'origin_price': origin_price, 'breakout_level': 0.0,
            'phase1_extreme': 0.0, 'pb_extreme': 0.0, 'confirmation_level': 0.0,
            'phase3_close': 0.0, 'pb_depth_pct': 0.0, 'continuation_pct': 0.0}

def _m3_result_failed(reason, *, origin_price, breakout_level, phase1_extreme, pb_extreme, bullish):
    if bullish: dp = (pb_extreme - origin_price) / origin_price * 100 if origin_price > 0 else -999
    else: dp = (origin_price - pb_extreme) / origin_price * 100 if origin_price > 0 else -999
    return {'valid': True, 'passed': False, 'quality': 15.0, 'reason': reason,
            'phase': 2, 'origin_price': origin_price, 'breakout_level': breakout_level,
            'phase1_extreme': phase1_extreme, 'pb_extreme': pb_extreme,
            'confirmation_level': phase1_extreme, 'phase3_close': 0.0,
            'pb_depth_pct': dp, 'continuation_pct': 0.0}

def _build_result(*, valid, score, direction, signals, pullback_pct, pb_atr,
                  vol_ratio, vol_zscore, close_strength, h1_rsi,
                  d1_adx, h4_adx, d1_ema_spread, h4_ema_spread,
                  trend_snap, trend_metrics, m3_result, reason=''):
    bull = direction == 'BUY'; bear = direction == 'SELL'
    tq = float(trend_snap.get('long_score' if (bull or not bear) else 'short_score', 25.0) or 25.0)
    m3p = bool(m3_result.get('passed'))
    lq = max(20.0, 100.0 - pullback_pct * 15.0)
    vq = min(vol_ratio / 1.3, 1.6) * 62.5
    fq = (96.0 if direction in {'BUY', 'SELL'} and m3p and pullback_pct <= 2.0
          else 88.0 if direction in {'BUY', 'SELL'} and m3p
          else 70.0 if direction in {'BUY', 'SELL'} else 35.0)
    return {
        'valid': valid, 'reason': reason, 'score': max(score, 0.0),
        'direction': direction, 'signals': signals,
        'ranking_factors': {
            'trend': tq, 'trigger': 90.0 if direction in {'BUY', 'SELL'} else 30.0,
            'volume': vq, 'location': lq, 'freshness': fq,
            'risk': 85.0 if 0.4 <= pullback_pct <= 3.2 else 55.0,
        },
        'details': {
            '评估':           ' | '.join(signals) if signals else '暂无回踩确认机会',
            '回踩距离':       f'{pullback_pct:.2f}%',
            'ATR倍数':        f'{pb_atr:.2f}',
            '量比':           f'{vol_ratio:.2f}x',
            '量能Z分':        f'{vol_zscore:+.2f}σ',
            '启动收盘强度':   f'{close_strength:.2f}',
            '1H_RSI':         f'{h1_rsi:.1f}',
            '日线ADX':        f'{d1_adx:.1f}',
            '4H_ADX':         f'{h4_adx:.1f}',
            '日线EMA发散':    f'{d1_ema_spread:+.2f}%',
            '4H_EMA发散':     f'{h4_ema_spread:+.2f}%',
            '趋势质量':       trend_snap.get('reason', '-'),
            'H4_ADX_内置':    f"{float(trend_metrics.get('h4_adx', 0.0)):.1f}",
            '趋势效率':       f"{float(trend_metrics.get('h1_efficiency', 0.0)):.1f}",
            '3m结构确认':     '通过' if m3p else '未通过',
            '3m当前阶段':     f"Phase {m3_result.get('phase', 0)}",
            '3m结构说明':     str(m3_result.get('reason', '-')),
            '3m起点':         f"{float(m3_result.get('origin_price', 0)):.8g}",
            '3m颈线突破位':   f"{float(m3_result.get('breakout_level', 0)):.8g}",
            '3mPhase1极值':   f"{float(m3_result.get('phase1_extreme', 0)):.8g}",
            '3m回踩极值':     f"{float(m3_result.get('pb_extreme', 0) or 0):.8g}",
            '3m回踩深度':     f"{float(m3_result.get('pb_depth_pct', 0)):.2f}%",
            '3m继续突破幅度': f"{float(m3_result.get('continuation_pct', 0)):.2f}%",
        },
    }


# ══════════════════════════════════════════════
# 模块导出
# ══════════════════════════════════════════════
STRATEGY_NAME  = "波段趋势回踩扫描"
STRATEGY_TYPE  = "scan"
STRATEGY_CLASS = TrendPullbackSwingScanner
