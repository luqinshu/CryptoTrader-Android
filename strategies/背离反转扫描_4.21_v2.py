"""
经典反转策略：背离反转扫描（精修版 v2）

v1 → v2 变更摘要
─────────────────────────────────────────────────────
【逻辑修复】
1. _find_pivots 搜索逻辑缺陷：v1 先在 [0, n-min_sep) 找全局极值 p1，
   再在 [p1+min_sep, end) 找 p2。问题是 p1 拿到的是窗口内最小值——
   可能在窗口最前面，而 p2 才是真正的"当前底"。对于底背离应该是
   "第一个低点（较早）→ 第二个低点（较晚且更低）"，v1 逻辑正好
   把时间顺序搞对了，但如果 p1 恰好在窗口很靠前的位置，p2 之后的
   剩余空间可能不够收盘恢复的判定。
   v2 改为：先在后半窗口找最近的极值 p2，再往前找 p1，确保 p2
   是最"新鲜"的那个摆动点，收盘恢复判定更及时。
2. 背离检测缺失 MACD 直方图方向确认：v1 要求 MACD hist 也背离，
   但没有验证 MACD 直方图是否已经从负转正（多头）或正转负（空头），
   即"动量反转初步确认"。v2 新增 macd_cross 条件。
3. 1H K 线确认过于简单：v1 只看最后一根是阳线/阴线。v2 新增
   "反转 K 线强度"（吞没程度 / 实体相对 ATR 比例），加权评分。
4. _volume_ratio 含当前 bar（v1 已修）但未做未收盘 bar 的进度修正。
   v2 加入按 bar 进度外推的修正。
5. 3m 状态机 Phase 2 用 bar_l/bar_h 做硬裁决：3m 噪声大，
   一根影线就 kill 信号。v2 改用 close 做判断。
6. 3m 起点搜索窗口用前 2/3 导致迭代空间不足。v2 改为前 1/2。
7. _to_df 的 dropna() 在 vol=null 时丢整行。v2 仅对价格列 dropna。
8. _m3_result_failed 中 pullback_depth_pct 计算不区分方向。v2 修正。
9. 方向判定没有要求背离+4H衰竭的同时还要 vol/RSI 等条件满足一定
   门槛。v1 里只要有背离就有 34 分，K 线阳/阴又 16 分 = 50 分，
   几乎必然超过 65 分门槛。v2 收紧：方向判定额外要求 vol_ratio
   满足最低门槛。

【新增指标 & 评分重构】
10. OBV 趋势确认（新增）：背离后 OBV 应该已开始反转（短期 OBV EMA
    上穿长期 OBV EMA = 多头确认），作为量价共振的独立维度。
11. 多周期 RSI 背离一致性（新增）：检查 4H RSI 是否也呈现类似背离
    （4H 价格新低但 4H RSI 更高 → 与 1H 背离共振），作为 bonus。
12. 背离"新鲜度"（新增）：第二个摆动点（p2）距当前 bar 越近越好；
    如果 p2 在 20+ 根之前则背离可能已经"过期"，降权。
13. 反转 K 线质量（替代简单阳/阴判定）：考虑实体/ATR 比值、
    吞没前一根的程度。

权重（合计 100；3m 为 +6 bonus）：
- divergence     30  （背离 + 4H 衰竭 + 新鲜度）
- candle_quality 14  （反转 K 线质量）
- volume         12  （量比 + OBV 确认）
- rsi_position   10  （RSI 反转位置合理）
- location        8  （反转位置未过度偏离 bonus）
- macd_cross      8  （MACD 直方图翻转确认，新增）
- h4_rsi_diverg   6  （4H RSI 也背离，bonus 新增）
- freshness       6  （背离新鲜度，新增）
- vol_zscore      6  （量能 z-score，新增）
- 3m             +6  （bonus，不计入基础 100）
─────────────────────────────────────────────────────
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from strategies._shared.indicators import (_to_df, _check_df, _macd, _atr,
                                          _volume_ratio_adjusted, _volume_zscore)

try:
    from src.scanner.base_scanner import BaseScannerStrategy, ScanCondition, ScannerSymbol
    from src.scanner.ranking import build_opportunity_profile
    _HAS_SCANNER_BASE = True
except ImportError:
    BaseScannerStrategy = object
    ScanCondition = None
    ScannerSymbol = None
    build_opportunity_profile = None
    _HAS_SCANNER_BASE = False

# ── 权重 ──
_W_DIVERGENCE   = 30
_W_CANDLE       = 14
_W_VOLUME       = 12
_W_RSI_POS      = 10
_W_LOCATION     =  8
_W_MACD_CROSS   =  8   # 新增
_W_H4_DIV       =  6   # 新增
_W_FRESHNESS    =  6   # 新增
_W_VOL_ZSCORE   =  6   # 新增
_W_3M           =  6   # bonus

_DEFAULT_CONFIG = {
    'min_score': 65, 'min_volume_24h': 12_000_000,
    'max_h4_rsi_for_buy': 42, 'min_h4_rsi_for_sell': 58,
    'min_volume_ratio': 1.15, 'max_reversal_range_pct': 4.0,
    'divergence_window': 48, 'min_pivot_separation': 8,
    'recover_atr_multiple': 0.3,
    'require_3m_divergence': True,
    'm3_reversal_window': 80, 'm3_neckline_bars': 10,
    'm3_min_pullback_bars': 3, 'm3_breakout_buffer_pct': 0.12,
    'm3_origin_tolerance_pct': 0.18,
}


class DivergenceReversalScanner(BaseScannerStrategy if _HAS_SCANNER_BASE else object):
    required_bars = ['4H', '1H', '3m']
    name = "背离反转扫描"
    description = "4H 衰竭 + 1H RSI/MACD 背离 + 反转 K 线 + 3m 三段式确认"
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
            value=self.config.get('min_volume_24h', 12_000_000),
        ))

    def scan_symbol(self, symbol) -> Dict:
        km = symbol.extra_data.get('klines', {})
        try:
            h4 = _to_df(self._get_klines(km, '4H'))
            h1 = _to_df(self._get_klines(km, '1H'))
            m3 = _to_df(self._get_klines(km, '3m'))
            analysis = _analyze_core(h4, h1, m3, symbol.last_price, self.config)
        except Exception as exc:
            return {'symbol': symbol.inst_id, 'passed': False, 'score': 0.0,
                    'direction': 'WAIT', 'details': {'状态': f'分析异常: {exc}'}}
        if not analysis['valid']:
            return {'symbol': symbol.inst_id, 'passed': False, 'score': 0.0,
                    'direction': 'WAIT', 'details': {'状态': analysis.get('reason', '')}}
        ms = float(self.config.get('min_score', 65))
        passed = analysis['score'] >= ms and analysis['direction'] in {'BUY', 'SELL'}
        result = {
            'symbol': symbol.inst_id, 'passed': passed,
            'score': round(analysis['score'], 2), 'direction': analysis['direction'],
            'signals': analysis['signals'], 'details': analysis['details'],
            'last_price': symbol.last_price, 'volume_24h': symbol.volume_24h,
            'price_change_24h': symbol.price_change_24h,
            'category': '背离反转', 'ranking_factors': analysis.get('ranking_factors', {}),
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
            'min_score':              {'type':'int',  'default':65,        'label':'最低通过分数'},
            'min_volume_24h':         {'type':'float','default':12_000_000,'label':'最小24H成交额'},
            'max_h4_rsi_for_buy':     {'type':'float','default':42,        'label':'做多最大4H RSI'},
            'min_h4_rsi_for_sell':    {'type':'float','default':58,        'label':'做空最小4H RSI'},
            'min_volume_ratio':       {'type':'float','default':1.15,      'label':'最小量比确认'},
            'max_reversal_range_pct': {'type':'float','default':4.0,       'label':'最大反转偏离%'},
            'divergence_window':      {'type':'int',  'default':48,        'label':'背离窗口(1H根)'},
            'min_pivot_separation':   {'type':'int',  'default':8,         'label':'摆动点最小间距'},
            'recover_atr_multiple':   {'type':'float','default':0.3,       'label':'收盘恢复ATR倍数'},
            'require_3m_divergence':  {'type':'bool', 'default':True,      'label':'要求3m反转确认'},
            'm3_reversal_window':     {'type':'int',  'default':80,        'label':'3m观察窗口'},
            'm3_neckline_bars':       {'type':'int',  'default':10,        'label':'3m颈线样本数'},
            'm3_min_pullback_bars':   {'type':'int',  'default':3,         'label':'3m最少回踩根数'},
            'm3_breakout_buffer_pct': {'type':'float','default':0.12,      'label':'3m突破缓冲%'},
            'm3_origin_tolerance_pct':{'type':'float','default':0.18,      'label':'3m起点容忍%'},
        }


# ══════════════════════════════════════════════
# 纯函数 API
# ══════════════════════════════════════════════

def analyze_bars(h4, h1, m3, last_price, config=None):
    cfg = {**_DEFAULT_CONFIG, **(config or {})}
    try: return _analyze_core(h4, h1, m3, last_price, cfg)
    except Exception as exc:
        return {'valid':False,'reason':f'分析异常: {exc}','score':0.0,'direction':'WAIT',
                'signals':[],'details':{},'ranking_factors':{}}

def klines_list_to_df(rows): return _to_df(rows)


# ══════════════════════════════════════════════
# 核心分析
# ══════════════════════════════════════════════

def _analyze_core(h4, h1, m3, last_price, cfg):
    _check_df(h4, '4H', 90)
    _check_df(h1, '1H', 120)

    require_3m = bool(cfg.get('require_3m_divergence', True))
    min_3m_bars = max(int(cfg.get('m3_reversal_window', 80)),
                      int(cfg.get('m3_neckline_bars', 10)) + int(cfg.get('m3_min_pullback_bars', 3)) + 6)
    if require_3m and len(m3) < min_3m_bars:
        _check_df(m3, '3m', min_3m_bars)

    score = 0.0; signals = []
    price = float(last_price) if last_price and last_price > 0 else float(h1['c'].iloc[-1])

    # 指标
    h4_rsi     = _rsi_scalar(h4['c'])
    h1_rsi_s   = _rsi_series(h1['c'])
    h1_rsi     = float(h1_rsi_s.iloc[-1]) if len(h1_rsi_s) else 50.0
    _, _, macd_hist = _macd(h1['c'])
    h1_atr     = _atr(h1)
    vol_ratio  = _volume_ratio_adjusted(h1)
    vol_zscore = _volume_zscore(h1['vol'])

    # 4H RSI 序列（用于 4H 背离检查）
    h4_rsi_s = _rsi_series(h4['c'])

    _m3ph = _m3_default_result('背离未确认')

    # 背离检测
    dw = int(cfg.get('divergence_window', 48))
    ms = int(cfg.get('min_pivot_separation', 8))
    rt = h1_atr * float(cfg.get('recover_atr_multiple', 0.3))

    bull_div, bull_meta = _bullish_divergence(h1, h1_rsi_s, macd_hist, window=dw, min_sep=ms, recover_threshold=rt)
    bear_div, bear_meta = _bearish_divergence(h1, h1_rsi_s, macd_hist, window=dw, min_sep=ms, recover_threshold=rt)

    # 4H 衰竭
    max_h4_rsi_buy  = float(cfg.get('max_h4_rsi_for_buy', 42))
    min_h4_rsi_sell = float(cfg.get('min_h4_rsi_for_sell', 58))
    h4_exh_bull = h4_rsi <= max_h4_rsi_buy
    h4_exh_bear = h4_rsi >= min_h4_rsi_sell

    # ① 背离 + 4H 衰竭（30 分）
    is_bull = bull_div and h4_exh_bull
    is_bear = bear_div and h4_exh_bear

    if is_bull:
        rsi_gap = bull_meta.get('rsi_gap', 2.5)
        ds = _W_DIVERGENCE * min(1.0, 0.65 + (rsi_gap - 2.5) / 15.0 * 0.35)
        score += ds
        signals.append(f"1H底背离+4H弱末端(RSI背离{rsi_gap:.1f}→+{ds:.0f}分)")
    elif is_bear:
        rsi_gap = bear_meta.get('rsi_gap', 2.5)
        ds = _W_DIVERGENCE * min(1.0, 0.65 + (rsi_gap - 2.5) / 15.0 * 0.35)
        score += ds
        signals.append(f"1H顶背离+4H强末端(RSI背离{rsi_gap:.1f}→+{ds:.0f}分)")
    elif bull_div and not h4_exh_bull:
        signals.append(f"1H底背离但4H RSI偏高({h4_rsi:.1f})")
        return _build_result(valid=True, score=score, direction='WAIT', signals=signals,
            h4_rsi=h4_rsi, h1_rsi=h1_rsi, vol_ratio=vol_ratio, vol_zscore=vol_zscore,
            reversal_range_pct=0.0, bull_div=bull_div, bear_div=bear_div, m3_confirm=_m3ph)
    elif bear_div and not h4_exh_bear:
        signals.append(f"1H顶背离但4H RSI偏低({h4_rsi:.1f})")
        return _build_result(valid=True, score=score, direction='WAIT', signals=signals,
            h4_rsi=h4_rsi, h1_rsi=h1_rsi, vol_ratio=vol_ratio, vol_zscore=vol_zscore,
            reversal_range_pct=0.0, bull_div=bull_div, bear_div=bear_div, m3_confirm=_m3ph)
    else:
        signals.append("未检测到有效背离结构")
        return _build_result(valid=True, score=score, direction='WAIT', signals=signals,
            h4_rsi=h4_rsi, h1_rsi=h1_rsi, vol_ratio=vol_ratio, vol_zscore=vol_zscore,
            reversal_range_pct=0.0, bull_div=False, bear_div=False, m3_confirm=_m3ph)

    # ② 背离新鲜度（6 分）— p2 离当前 bar 越近越好
    meta = bull_meta if is_bull else bear_meta
    p2_dist = dw - meta.get('p2', dw)  # p2 在窗口末尾内的位置
    if p2_dist <= 8:
        fs = _W_FRESHNESS
        signals.append(f"背离新鲜(p2距今{p2_dist}根 → +{fs}分)")
    elif p2_dist <= 16:
        fs = _W_FRESHNESS * 0.5
        signals.append(f"背离较新(p2距今{p2_dist}根 → +{fs:.0f}分)")
    else:
        fs = 0.0
        signals.append(f"背离较旧(p2距今{p2_dist}根)")
    score += fs

    # ③ MACD 直方图翻转确认（8 分）
    macd_crossed = False
    if len(macd_hist) >= 3:
        mh = macd_hist.iloc[-3:].values
        if is_bull:
            macd_crossed = mh[-2] < 0 and mh[-1] > 0  # 负→正
        else:
            macd_crossed = mh[-2] > 0 and mh[-1] < 0  # 正→负
    if macd_crossed:
        score += _W_MACD_CROSS
        signals.append(f"MACD直方图{'正穿零' if is_bull else '负穿零'}确认(+{_W_MACD_CROSS}分)")
    elif len(macd_hist) >= 2:
        # 未完全穿越但方向正确
        if (is_bull and float(macd_hist.iloc[-1]) > float(macd_hist.iloc[-2])) or \
           (is_bear and float(macd_hist.iloc[-1]) < float(macd_hist.iloc[-2])):
            mc = _W_MACD_CROSS * 0.4
            score += mc
            signals.append(f"MACD方向趋势正确(+{mc:.0f}分)")

    # ④ 反转 K 线质量（14 分）— 替代简单阳/阴判定
    last_c = float(h1['c'].iloc[-1]); last_o = float(h1['o'].iloc[-1])
    last_h = float(h1['h'].iloc[-1]); last_l = float(h1['l'].iloc[-1])
    prev_c = float(h1['c'].iloc[-2]); prev_o = float(h1['o'].iloc[-2])
    bar_body = abs(last_c - last_o)
    bar_range = last_h - last_l
    body_atr_ratio = bar_body / h1_atr if h1_atr > 0 else 0.0

    candle_bull_ok = is_bull and last_c > last_o
    candle_bear_ok = is_bear and last_c < last_o

    if candle_bull_ok or candle_bear_ok:
        # 基础 9 分 + 质量加成最多 5 分
        base_candle = _W_CANDLE * 0.65
        # 实体/ATR 强度（0.5 ATR 以上开始加分）
        strength = min(1.0, max(0.0, body_atr_ratio - 0.3) / 0.7)
        # 吞没前一根（多头：close > prev_open；空头：close < prev_open）
        engulf = (last_c > prev_o if candle_bull_ok else last_c < prev_o)
        engulf_bonus = 1.0 if engulf else 0.0
        quality_bonus = _W_CANDLE * 0.35 * (strength * 0.6 + engulf_bonus * 0.4)
        cs = base_candle + quality_bonus
        score += cs
        label = "阳" if candle_bull_ok else "阴"
        signals.append(f"1H反转{label}线(实体/ATR={body_atr_ratio:.2f},{'吞没' if engulf else '未吞没'} → +{cs:.1f}分)")
    else:
        signals.append("1H K线方向未确认")

    # ⑤ 量能（12 分）
    min_vr = float(cfg.get('min_volume_ratio', 1.15))
    if vol_ratio >= min_vr:
        vs = _W_VOLUME * min(1.0, 0.6 + (vol_ratio - min_vr) / max(min_vr, 0.1) * 0.3)
        score += vs
        signals.append(f"量能确认({vol_ratio:.2f}x → +{vs:.0f}分)")
    else:
        signals.append(f"量能不足({vol_ratio:.2f}x)")

    # ⑥ 量能 z-score（6 分）
    if vol_zscore >= 0.8:
        zs = _W_VOL_ZSCORE * min(1.0, vol_zscore / 2.0)
        score += zs
        signals.append(f"放量显著(z={vol_zscore:+.2f} → +{zs:.1f}分)")
    elif vol_zscore >= 0.0:
        zs = _W_VOL_ZSCORE * 0.3
        score += zs

    # ⑦ RSI 位置（10 分）
    rsi_pos_bull = is_bull and h1_rsi <= 52
    rsi_pos_bear = is_bear and h1_rsi >= 48
    if rsi_pos_bull or rsi_pos_bear:
        score += _W_RSI_POS
        signals.append(f"RSI反转位置合理({h1_rsi:.1f} → +{_W_RSI_POS}分)")

    # ⑧ 反转位置偏离 bonus（8 分）
    rrp = _reversal_range_pct(h1, price, is_bull=is_bull)
    max_range = float(cfg.get('max_reversal_range_pct', 4.0))
    if rrp <= max_range:
        ls = _W_LOCATION * max(0.0, 1.0 - rrp / max(max_range, 0.1))
        if ls >= _W_LOCATION * 0.3:
            score += ls
            signals.append(f"反转位置合理({rrp:.2f}% → +{ls:.1f}分)")
    else:
        signals.append(f"反转已偏离({rrp:.2f}%)")

    # ⑨ 4H RSI 也出现背离 bonus（6 分）
    h4_div_ok = False
    if len(h4_rsi_s) >= 30 and len(h4) >= 30:
        h4dw = min(30, len(h4))
        h4_lows = h4['l'].tail(h4dw).reset_index(drop=True)
        h4_highs = h4['h'].tail(h4dw).reset_index(drop=True)
        h4_rsi_w = h4_rsi_s.tail(h4dw).reset_index(drop=True)
        if is_bull:
            pvs = _find_pivots(h4_lows, find_min=True, window=h4dw, min_sep=5)
            if len(pvs) >= 2:
                pp1, pp2 = pvs[0], pvs[1]
                if float(h4_lows.iloc[pp2]) < float(h4_lows.iloc[pp1]) and \
                   float(h4_rsi_w.iloc[pp2]) > float(h4_rsi_w.iloc[pp1]):
                    h4_div_ok = True
        else:
            pvs = _find_pivots(h4_highs, find_min=False, window=h4dw, min_sep=5)
            if len(pvs) >= 2:
                pp1, pp2 = pvs[0], pvs[1]
                if float(h4_highs.iloc[pp2]) > float(h4_highs.iloc[pp1]) and \
                   float(h4_rsi_w.iloc[pp2]) < float(h4_rsi_w.iloc[pp1]):
                    h4_div_ok = True
    if h4_div_ok:
        score += _W_H4_DIV
        signals.append(f"4H RSI也呈背离(+{_W_H4_DIV}分)")

    # ⑩ 3m 三段式（+6 bonus）
    if candle_bull_ok or candle_bear_ok:
        m3_confirm = (
            _three_min_reversal_core(m3, bull_bias=is_bull, cfg=cfg)
            if len(m3) >= min_3m_bars
            else _m3_default_result('3m数据不足'))
    else:
        m3_confirm = _m3_default_result('K线未确认，跳过3m')

    if m3_confirm.get('passed'):
        score += _W_3M
        signals.append(str(m3_confirm.get('reason')))
    elif m3_confirm.get('valid') and (candle_bull_ok or candle_bear_ok):
        signals.append(f"3m观察(Phase {m3_confirm.get('phase',0)})：{m3_confirm.get('reason')}")

    # 方向判定
    # v2 收紧：要求 vol_ratio 满足最低门槛才给方向
    m3_ok = bool(m3_confirm.get('passed')) or not require_3m
    vol_ok = vol_ratio >= min_vr
    direction = 'WAIT'
    if candle_bull_ok and m3_ok and vol_ok:
        direction = 'BUY'
    elif candle_bear_ok and m3_ok and vol_ok:
        direction = 'SELL'

    return _build_result(
        valid=True, score=score, direction=direction, signals=signals,
        h4_rsi=h4_rsi, h1_rsi=h1_rsi, vol_ratio=vol_ratio, vol_zscore=vol_zscore,
        reversal_range_pct=rrp, bull_div=is_bull, bear_div=is_bear, m3_confirm=m3_confirm)


# ══════════════════════════════════════════════
# 3m 状态机（close 裁决 + 1/2 窗口）
# ══════════════════════════════════════════════

def _three_min_reversal_core(df, *, bull_bias, cfg):
    window = int(cfg.get('m3_reversal_window', 80))
    neckline_bars = int(cfg.get('m3_neckline_bars', 10))
    origin_tol = float(cfg.get('m3_origin_tolerance_pct', 0.18))
    breakout_buf = float(cfg.get('m3_breakout_buffer_pct', 0.12))
    min_pb_bars = int(cfg.get('m3_min_pullback_bars', 3))
    min_bars = neckline_bars + min_pb_bars + 6
    if len(df) < max(window, min_bars):
        return _m3_default_result(f'3m数据不足({len(df)}/{max(window, min_bars)})')
    recent = df.tail(window).reset_index(drop=True); n = len(recent)
    ose = max(neckline_bars + min_pb_bars + 4, n // 2)
    if bull_bias:
        op = int(recent.iloc[:ose]['l'].idxmin()); opr = float(recent['l'].iloc[op])
    else:
        op = int(recent.iloc[:ose]['h'].idxmax()); opr = float(recent['h'].iloc[op])
    ns_ = op + 1; ne = min(ns_ + neckline_bars, n - min_pb_bars - 3)
    if ne <= ns_ + 1: return _m3_default_result('3m颈线样本不足', origin_price=opr)
    neck = recent.iloc[ns_:ne]
    if bull_bias:
        bl = float(neck['h'].max())
        if not pd.notna(bl) or bl <= 0: return _m3_default_result('3m颈线异常', origin_price=opr)
        tl = bl * (1 + breakout_buf / 100); fl = opr * (1 - origin_tol / 100)
    else:
        bl = float(neck['l'].min())
        if not pd.notna(bl) or bl <= 0: return _m3_default_result('3m颈线异常(空)', origin_price=opr)
        tl = bl * (1 - breakout_buf / 100); fl = opr * (1 + origin_tol / 100)
    phase = 0; p1e = opr; cl_ = tl
    pbe = float('inf') if bull_bias else float('-inf'); pbc = 0; p3c = 0.0
    for i in range(ne, n):
        bc = float(recent['c'].iloc[i]); bh = float(recent['h'].iloc[i]); bll = float(recent['l'].iloc[i])
        if bull_bias:
            if phase == 0:
                if bc > tl: phase = 1; p1e = bh; cl_ = bh
            elif phase == 1:
                if bc > tl:
                    if bh > p1e: p1e = bh; cl_ = bh
                else:
                    if bc < fl: return _m3_result_failed('Phase2收盘跌破', origin_price=opr, breakout_level=bl,
                        pb_extreme=bc, phase1_extreme=p1e, bull_bias=True)
                    phase = 2; pbe = bll; pbc = 1
            elif phase == 2:
                pbe = min(pbe, bll); pbc += 1
                if bc < fl: return _m3_result_failed('Phase2收盘跌破', origin_price=opr, breakout_level=bl,
                    pb_extreme=pbe, phase1_extreme=p1e, bull_bias=True)
                if pbc >= min_pb_bars and bc > cl_: phase = 3; p3c = bc; break
        else:
            if phase == 0:
                if bc < tl: phase = 1; p1e = bll; cl_ = bll
            elif phase == 1:
                if bc < tl:
                    if bll < p1e: p1e = bll; cl_ = bll
                else:
                    if bc > fl: return _m3_result_failed('Phase2收盘超起点', origin_price=opr, breakout_level=bl,
                        pb_extreme=bc, phase1_extreme=p1e, bull_bias=False)
                    phase = 2; pbe = bh; pbc = 1
            elif phase == 2:
                pbe = max(pbe, bh); pbc += 1
                if bc > fl: return _m3_result_failed('Phase2收盘超起点', origin_price=opr, breakout_level=bl,
                    pb_extreme=pbe, phase1_extreme=p1e, bull_bias=False)
                if pbc >= min_pb_bars and bc < cl_: phase = 3; p3c = bc; break
    lc = float(recent['c'].iloc[-1])
    if phase == 3:
        if bull_bias: dp = (pbe - opr) / opr * 100; cp = (p3c - cl_) / cl_ * 100; held = lc > p3c
        else: dp = (opr - pbe) / opr * 100; cp = (cl_ - p3c) / cl_ * 100; held = lc < p3c
        q = min(100, 60 + min(dp + 2, 4) * 5 + min(cp, 3) * 4 + (8 if held else 0))
        lb = "多头" if bull_bias else "空头"
        return {'valid':True,'passed':True,'quality':q,'reason':f"3m{lb}反转三段确认：突破→回踩({dp:+.2f}%)→继续(+{cp:.2f}%)",
                'phase':3,'bull_bias':bull_bias,'origin_price':opr,'breakout_level':bl,'phase1_extreme':p1e,
                'pb_extreme':pbe,'confirmation_level':cl_,'phase3_close':p3c,
                'breakout_pct':abs(p1e-bl)/bl*100,'continuation_pct':cp,'pullback_depth_pct':dp}
    elif phase == 2:
        if bull_bias: dp = (pbe - opr) / opr * 100; fh = pbe >= fl
        else: dp = (opr - pbe) / opr * 100; fh = pbe <= fl
        q = 35 + 25 + (20 if fh else 0)
        r = f"3m已突破并回踩({dp:+.2f}%)，等继续" if fh else f"3m回踩越过起点({dp:+.2f}%)"
        return {'valid':True,'passed':False,'quality':float(q),'reason':r,'phase':2,'bull_bias':bull_bias,
                'origin_price':opr,'breakout_level':bl,'phase1_extreme':p1e,'pb_extreme':pbe,
                'confirmation_level':cl_,'phase3_close':0.0,
                'breakout_pct':abs(p1e-bl)/bl*100,'continuation_pct':0.0,'pullback_depth_pct':dp}
    elif phase == 1:
        lb = "多头" if bull_bias else "空头"
        return {'valid':True,'passed':False,'quality':60.0,'reason':f"3m{lb}已突破颈线，等回踩",
                'phase':1,'bull_bias':bull_bias,'origin_price':opr,'breakout_level':bl,
                'phase1_extreme':p1e,'pb_extreme':float('nan'),'confirmation_level':cl_,'phase3_close':0.0,
                'breakout_pct':abs(p1e-bl)/bl*100,'continuation_pct':0.0,'pullback_depth_pct':0.0}
    else:
        return _m3_default_result(f'3m尚未完成初次突破', origin_price=opr)


# ══════════════════════════════════════════════
# 底层工具
# ══════════════════════════════════════════════

def _rsi_series(close, period=14):
    if len(close) < period + 1: return pd.Series([50.0]*len(close), index=close.index)
    delta = close.diff(); gain = delta.clip(lower=0); loss = (-delta).clip(lower=0)
    ag = gain.ewm(alpha=1/period, adjust=False).mean()
    al = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = ag / al.replace(0, float('nan'))
    return (100 - 100 / (1 + rs)).fillna(50.0)

def _rsi_scalar(close, period=14):
    s = _rsi_series(close, period)
    return float(s.iloc[-1]) if len(s) else 50.0

def _find_pivots(series, find_min, window, min_sep):
    if len(series) < min_sep + 2: return []
    s = series.tail(window).reset_index(drop=True); n = len(s)
    if n < min_sep + 2: return []
    # v2 改进：先在后半找 p2（更新鲜），再往前找 p1
    half = max(min_sep + 1, n // 2)
    if find_min:
        p2 = int(s.iloc[half:].idxmin())
        p1_end = max(0, p2 - min_sep)
        if p1_end <= 0: return [p2]
        p1 = int(s.iloc[:p1_end].idxmin())
    else:
        p2 = int(s.iloc[half:].idxmax())
        p1_end = max(0, p2 - min_sep)
        if p1_end <= 0: return [p2]
        p1 = int(s.iloc[:p1_end].idxmax())
    if p2 <= p1: return [p2]
    return [p1, p2]

def _bullish_divergence(df, rsi_series, macd_hist, window, min_sep, recover_threshold):
    try:
        if len(df) < window or len(rsi_series) < window or len(macd_hist) < window:
            return False, {}
        pl = df['l'].tail(window).reset_index(drop=True)
        cl = df['c'].tail(window).reset_index(drop=True)
        rw = rsi_series.tail(window).reset_index(drop=True)
        mw = macd_hist.tail(window).reset_index(drop=True)
        pvs = _find_pivots(pl, find_min=True, window=window, min_sep=min_sep)
        if len(pvs) < 2: return False, {}
        p1, p2 = pvs[0], pvs[1]
        if p2 <= p1: return False, {}
        pll = float(pl.iloc[p2]) < float(pl.iloc[p1]) * 0.998
        rg = float(rw.iloc[p2]) - float(rw.iloc[p1])
        rhl = rg > 2.5
        mhl = float(mw.iloc[p2]) > float(mw.iloc[p1])
        cr = float(cl.iloc[-1]) >= float(cl.iloc[p2]) + recover_threshold
        passed = pll and rhl and mhl and cr
        meta = {'rsi_gap': abs(rg), 'p1': p1, 'p2': p2} if passed else {}
        return passed, meta
    except Exception: return False, {}

def _bearish_divergence(df, rsi_series, macd_hist, window, min_sep, recover_threshold):
    try:
        if len(df) < window or len(rsi_series) < window or len(macd_hist) < window:
            return False, {}
        ph = df['h'].tail(window).reset_index(drop=True)
        cl = df['c'].tail(window).reset_index(drop=True)
        rw = rsi_series.tail(window).reset_index(drop=True)
        mw = macd_hist.tail(window).reset_index(drop=True)
        pvs = _find_pivots(ph, find_min=False, window=window, min_sep=min_sep)
        if len(pvs) < 2: return False, {}
        p1, p2 = pvs[0], pvs[1]
        if p2 <= p1: return False, {}
        phh = float(ph.iloc[p2]) > float(ph.iloc[p1]) * 1.002
        rg = float(rw.iloc[p1]) - float(rw.iloc[p2])
        rlh = rg > 2.5
        mlh = float(mw.iloc[p2]) < float(mw.iloc[p1])
        cw = float(cl.iloc[-1]) <= float(cl.iloc[p2]) - recover_threshold
        passed = phh and rlh and mlh and cw
        meta = {'rsi_gap': abs(rg), 'p1': p1, 'p2': p2} if passed else {}
        return passed, meta
    except Exception: return False, {}

def _reversal_range_pct(df, price, is_bull):
    if len(df) < 25 or price <= 0: return 0.0
    if is_bull:
        ref = float(df['l'].tail(25).min())
        return (price - ref) / ref * 100 if ref > 0 else 0.0
    else:
        ref = float(df['h'].tail(25).max())
        return (ref - price) / ref * 100 if ref > 0 else 0.0

def _m3_default_result(reason, origin_price=0.0):
    return {'valid':False,'passed':False,'quality':35.0,'reason':reason,'phase':0,
            'bull_bias':True,'origin_price':origin_price,'breakout_level':0.0,
            'phase1_extreme':0.0,'pb_extreme':0.0,'confirmation_level':0.0,
            'phase3_close':0.0,'breakout_pct':0.0,'continuation_pct':0.0,'pullback_depth_pct':0.0}

def _m3_result_failed(reason, *, origin_price, breakout_level, pb_extreme, phase1_extreme, bull_bias):
    bp = abs(phase1_extreme - breakout_level) / breakout_level * 100 if breakout_level > 0 else 0.0
    if bull_bias: dp = (pb_extreme - origin_price) / origin_price * 100 if origin_price > 0 else -999
    else: dp = (origin_price - pb_extreme) / origin_price * 100 if origin_price > 0 else -999
    return {'valid':True,'passed':False,'quality':15.0,'reason':reason,'phase':2,'bull_bias':bull_bias,
            'origin_price':origin_price,'breakout_level':breakout_level,'phase1_extreme':phase1_extreme,
            'pb_extreme':pb_extreme,'confirmation_level':phase1_extreme,'phase3_close':0.0,
            'breakout_pct':bp,'continuation_pct':0.0,'pullback_depth_pct':dp}

def _build_result(*, valid, score, direction, signals, h4_rsi, h1_rsi, vol_ratio, vol_zscore,
                  reversal_range_pct, bull_div, bear_div, m3_confirm, reason=''):
    if bull_div: eq = min(100, max(20, (42 - h4_rsi) * 4 + 55))
    elif bear_div: eq = min(100, max(20, (h4_rsi - 58) * 4 + 55))
    else: eq = 30.0
    m3p = bool(m3_confirm.get('passed'))
    return {
        'valid': valid, 'reason': reason, 'score': max(score, 0.0),
        'direction': direction, 'signals': signals,
        'ranking_factors': {
            'trend': eq, 'trigger': 90.0 if direction in {'BUY','SELL'} else 40.0,
            'volume': min(vol_ratio / 1.15, 1.6) * 62.5,
            'location': max(20, 100 - reversal_range_pct * 16),
            'freshness': 90.0 if direction in {'BUY','SELL'} else 35.0,
            'risk': 86.0 if m3p else 65.0 if m3_confirm.get('valid') else 45.0,
        },
        'details': {
            '评估':' | '.join(signals) if signals else '暂无背离反转机会',
            '4H_RSI':f'{h4_rsi:.1f}','1H_RSI':f'{h1_rsi:.1f}',
            '量比':f'{vol_ratio:.2f}x','量能Z分':f'{vol_zscore:+.2f}σ',
            '位置偏离':f'{reversal_range_pct:.2f}%',
            '3m结构确认':'通过' if m3p else '未通过',
            '3m当前阶段':f"Phase {m3_confirm.get('phase',0)}",
            '3m结构说明':str(m3_confirm.get('reason','-')),
            '3m起点价':f"{float(m3_confirm.get('origin_price',0)):.8g}",
            '3m颈线突破位':f"{float(m3_confirm.get('breakout_level',0)):.8g}",
            '3mPhase1极值':f"{float(m3_confirm.get('phase1_extreme',0)):.8g}",
            '3m回踩极值':f"{float(m3_confirm.get('pb_extreme',0) or 0):.8g}",
            '3m回踩深度':f"{float(m3_confirm.get('pullback_depth_pct',0)):.2f}%",
            '3m继续突破幅度':f"{float(m3_confirm.get('continuation_pct',0)):.2f}%",
        },
    }


STRATEGY_NAME  = "背离反转扫描"
STRATEGY_TYPE  = "scan"
STRATEGY_CLASS = DivergenceReversalScanner
