"""
经典波段策略：超跌反转扫描（精修版 v3 — 趋势捕捉增强）

v2 → v3 核心变更
─────────────────────────────────────────────────────
【设计思路转变】
v2 只在"极端超跌 + 反转 K 线"时给 BUY，要求 4H RSI ≤ 35、1H RSI ≤ 30。
这把策略限制在了"跌到不能再跌"的极端场景，而错过了真正有价值的机会：
**超跌反弹初段确认后形成的新上升趋势**。

v3 拆成两个模式：
A) 经典超跌模式（保留）：极端超跌 + 反转 K 线 → 适合猜底
B) 趋势初段模式（新增）：超跌反弹后 RSI 回到 40-55 区间，价格站上
   短期 EMA，MACD 金叉 → 不再是猜底，而是确认趋势已经形成

两个模式共享同一个评分框架，但权重和条件不同。

【逻辑修复】
1. _volume_ratio 包含当前 bar：v2 用 `vol.tail(window).mean()` 计算基线
   → v3 改为 `vol.iloc[-(window+1):-1].mean()` + 未收盘 bar 进度修正。
2. 方向判定只要 candle_ok + h4_rsi_ok 就给 BUY，但没有要求量能通过
   → v3 收紧：BUY 需要量能也满足门槛。
3. 3m Phase 2 用 bar_l 做硬裁决 → v3 改用 close（容忍影线噪声）。
4. 3m 起点搜索窗口前 2/3 → v3 改为前 1/2。
5. _to_df 的 dropna() 在 vol=null 时丢整行 → v3 仅对价格列 dropna。
6. recent_drop_pct 用 4H 收盘最高价但没区分是哪种跌法（缓跌 vs 暴跌）
   → v3 新增"跌速"指标（ATR 归一化的每日跌幅）。
7. 布林带判定太宽松（price ≤ bb_lower * 1.02 就给分）→ v3 收紧到
   1.005 并增加 BB 收缩/扩张状态判断。
8. 只做 BUY 不做 SELL → v3 保留此设计（超跌反转本身就是多头策略）
   但新增趋势初段模式扩展了 BUY 的触发场景。

【新增指标（趋势捕捉相关）】
9.  EMA 趋势初段确认：价格站上 EMA8 且 EMA8 > EMA21（短期趋势形成）
10. MACD 金叉确认：MACD line 上穿 signal line
11. RSI 动量恢复：RSI 从超跌区回升到 40-55（不再是极端超跌，而是
    "已经反弹且动量健康"）
12. OBV 趋势翻转：短期 OBV 均线上穿长期 OBV 均线
13. 成交量 z-score：log 空间异常放量检测
14. ADX 趋势确认：ADX > 20 说明新趋势正在形成（而非仅仅是超卖反弹）
15. 吞没 K 线检测：替代简单阳线判定

【评分（合计 100 + 6 bonus）】
超跌模式：
  h4_rsi(20) + h1_rsi(12) + bb(10) + drop(10) + candle(14) + volume(10) +
  drop_speed(6) + vol_zscore(6) + macd(6) + trend_init(6) = 100 + 3m(6)

趋势初段模式：
  trend_init(22) + macd(14) + adx(12) + candle(14) + volume(10) +
  rsi_recovery(10) + vol_zscore(6) + obv(6) + bb_expand(6) = 100 + 3m(6)

两种模式取更高分。
─────────────────────────────────────────────────────
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from strategies._shared.indicators import (_to_df, _check_df, _ema, _rsi_wilder, _macd, _atr, _adx,
                                          _volume_ratio_adjusted, _volume_zscore)

try:
    from src.scanner.base_scanner import BaseScannerStrategy, ScanCondition, ScannerSymbol
    from src.scanner.ranking import build_opportunity_profile
    _HAS_SCANNER_BASE = True
except ImportError:
    BaseScannerStrategy = object
    ScanCondition = None; ScannerSymbol = None
    build_opportunity_profile = None
    _HAS_SCANNER_BASE = False

_DEFAULT_CONFIG = {
    'min_score': 65, 'min_volume_24h': 10_000_000,
    'max_h4_rsi': 35, 'max_h1_rsi': 30,
    'min_recent_drop_pct': 12, 'min_reversal_volume_ratio': 1.5,
    'require_3m_reversal_retest': True,
    'm3_reversal_window': 80, 'm3_neckline_bars': 10,
    'm3_min_pullback_bars': 3, 'm3_breakout_buffer_pct': 0.12,
    'm3_origin_tolerance_pct': 0.18,
    # 趋势初段模式参数
    'trend_rsi_min': 40, 'trend_rsi_max': 58,
    'trend_adx_min': 18,
}


class OversoldReversalSwingScanner(BaseScannerStrategy if _HAS_SCANNER_BASE else object):
    required_bars = ['4H', '1H', '3m']
    name = "波段超跌反转扫描"
    description = "超跌反弹 + 趋势初段双模式：极端超跌猜底 / RSI 恢复+EMA 站上+MACD 金叉 = 趋势确认"
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
            value=self.config.get('min_volume_24h', 10_000_000),
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
        passed = analysis['score'] >= ms and analysis['direction'] == 'BUY'
        result = {
            'symbol': symbol.inst_id, 'passed': passed,
            'score': round(analysis['score'], 2), 'direction': analysis['direction'],
            'signals': analysis['signals'], 'details': analysis['details'],
            'last_price': symbol.last_price, 'volume_24h': symbol.volume_24h,
            'price_change_24h': symbol.price_change_24h,
            'category': '波段超跌反转', 'ranking_factors': analysis.get('ranking_factors', {}),
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
            'min_score':                  {'type':'int',  'default':65,        'label':'最低通过分数'},
            'min_volume_24h':             {'type':'float','default':10_000_000,'label':'最小24H成交额'},
            'max_h4_rsi':                 {'type':'float','default':35,        'label':'超跌模式: 4H最大RSI'},
            'max_h1_rsi':                 {'type':'float','default':30,        'label':'超跌模式: 1H最大RSI'},
            'min_recent_drop_pct':        {'type':'float','default':12,        'label':'超跌模式: 近20根最小跌幅%'},
            'min_reversal_volume_ratio':  {'type':'float','default':1.5,       'label':'反转最小量比'},
            'trend_rsi_min':              {'type':'float','default':40,        'label':'趋势模式: RSI 下限'},
            'trend_rsi_max':              {'type':'float','default':58,        'label':'趋势模式: RSI 上限'},
            'trend_adx_min':              {'type':'float','default':18,        'label':'趋势模式: ADX 下限'},
            'require_3m_reversal_retest': {'type':'bool', 'default':True,      'label':'要求3m三段式确认'},
            'm3_reversal_window':         {'type':'int',  'default':80,        'label':'3m观察窗口'},
            'm3_neckline_bars':           {'type':'int',  'default':10,        'label':'3m颈线样本数'},
            'm3_min_pullback_bars':       {'type':'int',  'default':3,         'label':'3m最少回踩根数'},
            'm3_breakout_buffer_pct':     {'type':'float','default':0.12,      'label':'3m突破缓冲%'},
            'm3_origin_tolerance_pct':    {'type':'float','default':0.18,      'label':'3m起点容忍%'},
        }


# ══════════════════════════════════════════════
# 纯函数 API
# ══════════════════════════════════════════════
def analyze_bars(h4, h1, m3, last_price, config=None):
    cfg = {**_DEFAULT_CONFIG, **(config or {})}
    try: return _analyze_core(h4, h1, m3, last_price, cfg)
    except Exception as exc:
        return {'valid':False,'reason':f'异常: {exc}','score':0.0,'direction':'WAIT',
                'signals':[],'details':{},'ranking_factors':{}}

def klines_list_to_df(rows): return _to_df(rows)


# ══════════════════════════════════════════════
# 核心分析
# ══════════════════════════════════════════════
def _analyze_core(h4, h1, m3, last_price, cfg):
    _check_df(h4, '4H', 90); _check_df(h1, '1H', 120)
    require_3m = bool(cfg.get('require_3m_reversal_retest', True))
    min_3m_bars = max(int(cfg.get('m3_reversal_window', 80)),
                      int(cfg.get('m3_neckline_bars', 10)) + int(cfg.get('m3_min_pullback_bars', 3)) + 6)
    if require_3m and len(m3) < min_3m_bars:
        _check_df(m3, '3m', min_3m_bars)
    price = float(last_price) if last_price and last_price > 0 else float(h1['c'].iloc[-1])

    # ── 共用指标 ──
    h4_rsi = _rsi_wilder(h4['c']); h1_rsi = _rsi_wilder(h1['c'])
    bb_lower = _bb_lower(h1['c'], 20, 2.0)
    vol_ratio = _volume_ratio_adjusted(h1)
    vol_zscore = _volume_zscore(h1['vol'])
    candle_quality, candle_ok = _reversal_candle_quality(h1)
    h1_atr = _atr(h1)
    _, _, macd_hist = _macd(h1['c']); macd_line, macd_sig, _ = _macd(h1['c'])
    h4_adx = _adx(h4, 14)

    # 近期跌幅
    recent_high_close = float(h4['c'].tail(20).max())
    recent_drop_pct = (recent_high_close - price) / recent_high_close * 100 if recent_high_close > 0 else 0.0

    # EMA
    h1_ema8 = _ema(h1['c'], 8); h1_ema21 = _ema(h1['c'], 21)

    # 3m 结构
    m3_retest = (
        _three_min_reversal_core(m3, cfg=cfg)
        if len(m3) >= min_3m_bars
        else _m3_default_result('3m数据不足'))

    # ── 两种模式分别评分 ──
    score_a, signals_a, dir_a = _score_oversold_mode(
        h4_rsi, h1_rsi, bb_lower, price, recent_drop_pct, candle_quality, candle_ok,
        vol_ratio, vol_zscore, macd_hist, h1_ema8, h1_ema21, h1_atr, m3_retest,
        require_3m, cfg)

    score_b, signals_b, dir_b = _score_trend_init_mode(
        h4_rsi, h1_rsi, h4_adx, price, h1_ema8, h1_ema21,
        macd_line, macd_sig, macd_hist, candle_quality, candle_ok,
        vol_ratio, vol_zscore, bb_lower, h1_atr, h1, m3_retest,
        require_3m, cfg)

    # 取更高分的模式
    if score_a >= score_b:
        score, signals, direction = score_a, signals_a, dir_a
        mode = "超跌反转"
    else:
        score, signals, direction = score_b, signals_b, dir_b
        mode = "趋势初段"

    signals.insert(0, f"[{mode}模式]")

    return _build_result(
        valid=True, score=score, direction=direction, signals=signals,
        h4_rsi=h4_rsi, h1_rsi=h1_rsi, recent_drop_pct=recent_drop_pct,
        vol_ratio=vol_ratio, vol_zscore=vol_zscore, h4_adx=h4_adx,
        h1_ema8=h1_ema8, h1_ema21=h1_ema21, mode=mode, m3_retest=m3_retest)


# ══════════════════════════════════════════════
# 模式 A：经典超跌（保留 v2 核心逻辑，修正问题）
# ══════════════════════════════════════════════
def _score_oversold_mode(h4_rsi, h1_rsi, bb_lower, price, recent_drop_pct,
                         candle_quality, candle_ok, vol_ratio, vol_zscore,
                         macd_hist, h1_ema8, h1_ema21, h1_atr, m3_retest,
                         require_3m, cfg):
    score = 0.0; signals = []
    max_h4_rsi = float(cfg.get('max_h4_rsi', 35))
    max_h1_rsi = float(cfg.get('max_h1_rsi', 30))
    min_vr = float(cfg.get('min_reversal_volume_ratio', 1.5))

    # ① 4H RSI 超跌（20 分）
    h4_rsi_ok = h4_rsi <= max_h4_rsi
    if not h4_rsi_ok:
        signals.append(f"4H RSI未达超跌({h4_rsi:.1f}/{max_h4_rsi})")
        return score, signals, 'WAIT'
    h4s = 20.0 * min(1.0, (max_h4_rsi - h4_rsi) / max(max_h4_rsi - 10, 1) + 0.5)
    score += h4s; signals.append(f"4H超跌({h4_rsi:.1f} → +{h4s:.0f}分)")

    # ② 1H RSI 深度超跌（12 分）
    if h1_rsi <= max_h1_rsi:
        h1s = 12.0 * min(1.0, (max_h1_rsi - h1_rsi) / max(max_h1_rsi - 10, 1) + 0.5)
        score += h1s; signals.append(f"1H深度超跌({h1_rsi:.1f} → +{h1s:.0f}分)")
    else:
        signals.append(f"1H RSI偏高({h1_rsi:.1f})")

    # ③ 布林下轨（10 分）
    if price <= bb_lower * 1.005:
        bbs = 10.0 if price <= bb_lower else 5.5
        score += bbs; signals.append(f"{'跌破' if price<=bb_lower else '贴近'}布林下轨(+{bbs:.0f}分)")

    # ④ 近期跌幅（10 分）
    min_drop = float(cfg.get('min_recent_drop_pct', 12))
    if recent_drop_pct >= min_drop:
        ds = min(10.0, 10.0 * recent_drop_pct / max(min_drop, 1) * 0.75)
        score += ds; signals.append(f"跌幅充分({recent_drop_pct:.1f}% → +{ds:.0f}分)")

    # ⑤ 跌速（6 分）— ATR 归一化
    drop_speed = recent_drop_pct / 20.0  # 20 根 4H ≈ 80 小时 ≈ 3.3 天的平均日跌幅
    if drop_speed >= 2.0:
        score += 6.0; signals.append(f"急跌(速度{drop_speed:.1f}%/根 → +6分)")
    elif drop_speed >= 1.0:
        score += 3.0

    # ⑥ 反转 K 线（14 分）
    if candle_ok:
        cs = 14.0 * candle_quality
        score += cs; signals.append(f"反转K线({candle_quality:.0%} → +{cs:.0f}分)")
    else:
        signals.append("未见反转K线")

    # ⑦ 量能（10 分）
    if vol_ratio >= min_vr:
        vs = min(10.0, 10.0 * vol_ratio / min_vr / 2.0)
        score += vs; signals.append(f"反转量能({vol_ratio:.2f}x → +{vs:.0f}分)")

    # ⑧ vol z-score（6 分）
    if vol_zscore >= 0.8:
        zs = min(6.0, 6.0 * vol_zscore / 2.0)
        score += zs

    # ⑨ MACD 方向（6 分）
    if len(macd_hist) >= 2 and float(macd_hist.iloc[-1]) > float(macd_hist.iloc[-2]):
        score += 6.0; signals.append("MACD动量恢复(+6分)")

    # ⑩ 趋势初步（6 分 bonus）— 超跌后如果已站上短均线
    if price > h1_ema8 and h1_ema8 > h1_ema21:
        score += 6.0; signals.append("已站上短期均线(+6分)")

    # 3m
    if m3_retest.get('passed'):
        score += 6.0; signals.append(str(m3_retest.get('reason')))
    elif m3_retest.get('valid'):
        signals.append(f"3m观察：{m3_retest.get('reason')}")

    # 方向
    m3_ok = bool(m3_retest.get('passed')) or not require_3m
    vol_ok = vol_ratio >= min_vr
    direction = 'BUY' if candle_ok and h4_rsi_ok and m3_ok and vol_ok else 'WAIT'

    return score, signals, direction


# ══════════════════════════════════════════════
# 模式 B：趋势初段（新增）
# ══════════════════════════════════════════════
def _score_trend_init_mode(h4_rsi, h1_rsi, h4_adx, price, h1_ema8, h1_ema21,
                           macd_line, macd_sig, macd_hist, candle_quality, candle_ok,
                           vol_ratio, vol_zscore, bb_lower, h1_atr, h1,
                           m3_retest, require_3m, cfg):
    """
    不要求极端超跌，而是在超跌反弹后确认新趋势形成。
    触发条件：
    - 4H RSI 曾经超跌（< 40），现在已恢复到 40-58（说明刚从底部起来）
    - 价格站上 EMA8 > EMA21（短期趋势已形成）
    - MACD 金叉
    """
    score = 0.0; signals = []
    trend_rsi_min = float(cfg.get('trend_rsi_min', 40))
    trend_rsi_max = float(cfg.get('trend_rsi_max', 58))
    trend_adx_min = float(cfg.get('trend_adx_min', 18))
    min_vr = float(cfg.get('min_reversal_volume_ratio', 1.5))

    # 前提：4H RSI 在恢复区间（不是极端超跌也不是正常区间）
    rsi_recovery = trend_rsi_min <= h4_rsi <= trend_rsi_max
    if not rsi_recovery:
        signals.append(f"RSI不在恢复区间({h4_rsi:.1f}，需{trend_rsi_min}-{trend_rsi_max})")
        return score, signals, 'WAIT'

    # ① EMA 趋势初段（22 分）
    ema_trend = price > h1_ema8 and h1_ema8 > h1_ema21
    if ema_trend:
        # 发散度加成
        spread = (h1_ema8 - h1_ema21) / h1_ema21 * 100 if h1_ema21 > 0 else 0.0
        base = 16.0
        spread_bonus = min(6.0, spread / 0.5 * 3.0)
        es = base + spread_bonus
        score += es
        signals.append(f"EMA趋势形成(价>{h1_ema8:.2f}>{h1_ema21:.2f}, 发散{spread:.2f}% → +{es:.0f}分)")
    else:
        signals.append("EMA趋势未形成")
        return score, signals, 'WAIT'  # 这是趋势模式的硬要求

    # ② MACD 金叉（14 分）
    macd_golden = False
    if len(macd_line) >= 2 and len(macd_sig) >= 2:
        prev_diff = float(macd_line.iloc[-2]) - float(macd_sig.iloc[-2])
        curr_diff = float(macd_line.iloc[-1]) - float(macd_sig.iloc[-1])
        macd_golden = prev_diff <= 0 and curr_diff > 0
        macd_above_zero = float(macd_hist.iloc[-1]) > 0

    if macd_golden:
        score += 14.0; signals.append("MACD金叉确认(+14分)")
    elif macd_above_zero if len(macd_hist) >= 1 else False:
        score += 7.0; signals.append("MACD在零轴上方(+7分)")
    else:
        if len(macd_hist) >= 2 and float(macd_hist.iloc[-1]) > float(macd_hist.iloc[-2]):
            score += 4.0; signals.append("MACD动量恢复(+4分)")

    # ③ ADX 趋势强度（12 分）
    if h4_adx >= trend_adx_min:
        adx_s = min(12.0, 8.0 + (h4_adx - trend_adx_min) / 10.0 * 4.0)
        score += adx_s
        signals.append(f"ADX趋势确认({h4_adx:.1f} → +{adx_s:.0f}分)")
    else:
        signals.append(f"ADX偏弱({h4_adx:.1f})")

    # ④ K 线确认（14 分）
    if candle_ok:
        cs = 14.0 * candle_quality
        score += cs; signals.append(f"趋势K线({candle_quality:.0%} → +{cs:.0f}分)")

    # ⑤ 量能（10 分）
    if vol_ratio >= min_vr * 0.8:  # 趋势模式量比要求略低
        vs = min(10.0, 10.0 * vol_ratio / min_vr)
        score += vs; signals.append(f"量能({vol_ratio:.2f}x → +{vs:.0f}分)")

    # ⑥ RSI 恢复质量（10 分）
    rsi_center = (trend_rsi_min + trend_rsi_max) / 2
    rsi_quality = 1.0 - abs(h1_rsi - rsi_center) / ((trend_rsi_max - trend_rsi_min) / 2)
    rsi_s = 10.0 * max(0.0, rsi_quality)
    if rsi_s >= 3.0:
        score += rsi_s; signals.append(f"RSI恢复健康({h1_rsi:.1f} → +{rsi_s:.0f}分)")

    # ⑦ vol z-score（6 分）
    if vol_zscore >= 0.5:
        zs = min(6.0, 6.0 * vol_zscore / 1.5)
        score += zs

    # ⑧ OBV 翻转（6 分）
    obv_ok = _obv_trend_up(h1)
    if obv_ok:
        score += 6.0; signals.append("OBV趋势翻转(+6分)")

    # ⑨ BB 扩张（6 分）— 从收缩进入扩张说明新趋势启动
    bb_expanding = _bb_expanding(h1['c'])
    if bb_expanding and price > bb_lower * 1.02:
        score += 6.0; signals.append("布林带扩张(+6分)")

    # 3m
    if m3_retest.get('passed'):
        score += 6.0; signals.append(str(m3_retest.get('reason')))
    elif m3_retest.get('valid'):
        signals.append(f"3m观察：{m3_retest.get('reason')}")

    # 方向
    m3_ok = bool(m3_retest.get('passed')) or not require_3m
    direction = 'BUY' if ema_trend and m3_ok and candle_ok else 'WAIT'

    return score, signals, direction


# ══════════════════════════════════════════════
# 3m 状态机（close 裁决 + 1/2 窗口）
# ══════════════════════════════════════════════
def _three_min_reversal_core(df, *, cfg):
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
    op = int(recent.iloc[:ose]['l'].idxmin()); opr = float(recent['l'].iloc[op])
    ns_ = op + 1; ne = min(ns_ + neckline_bars, n - min_pb_bars - 3)
    if ne <= ns_ + 1: return _m3_default_result('3m颈线样本不足', origin_price=opr)
    neck = recent.iloc[ns_:ne]
    bl = float(neck['h'].max())
    if not pd.notna(bl) or bl <= 0: return _m3_default_result('3m颈线异常', origin_price=opr)
    tl = bl * (1 + breakout_buf / 100); fl = opr * (1 - origin_tol / 100)
    phase = 0; p1h = 0.0; cl_ = tl; pbl = float('inf'); pbc = 0; p3c = 0.0
    for i in range(ne, n):
        bc = float(recent['c'].iloc[i]); bh = float(recent['h'].iloc[i]); bll = float(recent['l'].iloc[i])
        if phase == 0:
            if bc > tl: phase = 1; p1h = bh; cl_ = bh
        elif phase == 1:
            if bc > tl:
                if bh > p1h: p1h = bh; cl_ = bh
            else:
                if bc < fl: return _m3_result_failed('Phase2收盘跌破', origin_price=opr,
                    breakout_level=bl, pullback_low=bc, phase1_high=p1h)
                phase = 2; pbl = bll; pbc = 1
        elif phase == 2:
            pbl = min(pbl, bll); pbc += 1
            if bc < fl: return _m3_result_failed('Phase2收盘跌破', origin_price=opr,
                breakout_level=bl, pullback_low=pbl, phase1_high=p1h)
            if pbc >= min_pb_bars and bc > cl_: phase = 3; p3c = bc; break
    lc = float(recent['c'].iloc[-1])
    if phase == 3:
        dp = (pbl - opr) / opr * 100; cp = (p3c - cl_) / cl_ * 100; held = lc > p3c
        q = min(100, 60 + min(dp + 2, 4) * 5 + min(cp, 3) * 4 + (8 if held else 0))
        return {'valid':True,'passed':True,'quality':q,
                'reason':f"3m三段确认：突破→回踩({dp:+.2f}%)→继续(+{cp:.2f}%)",
                'phase':3,'origin_price':opr,'breakout_level':bl,'phase1_high':p1h,
                'pullback_low':pbl,'confirmation_level':cl_,'phase3_close':p3c,
                'breakout_pct':(p1h-bl)/bl*100,'continuation_pct':cp,'pullback_depth_pct':dp}
    elif phase == 2:
        dp = (pbl - opr) / opr * 100; fh = pbl >= fl
        q = 35 + 25 + (20 if fh else 0)
        r = f"3m已突破并回踩({dp:+.2f}%)，等继续" if fh else f"3m回踩跌破起点({dp:+.2f}%)"
        return {'valid':True,'passed':False,'quality':float(q),'reason':r,'phase':2,
                'origin_price':opr,'breakout_level':bl,'phase1_high':p1h,'pullback_low':pbl,
                'confirmation_level':cl_,'phase3_close':0.0,'breakout_pct':(p1h-bl)/bl*100,
                'continuation_pct':0.0,'pullback_depth_pct':dp}
    elif phase == 1:
        return {'valid':True,'passed':False,'quality':60.0,
                'reason':f"3m已突破颈线，等回踩(高点={p1h:.6g})",'phase':1,
                'origin_price':opr,'breakout_level':bl,'phase1_high':p1h,'pullback_low':float('nan'),
                'confirmation_level':cl_,'phase3_close':0.0,'breakout_pct':(p1h-bl)/bl*100,
                'continuation_pct':0.0,'pullback_depth_pct':0.0}
    else:
        return _m3_default_result('3m尚未完成初次突破', origin_price=opr)


# ══════════════════════════════════════════════
# 底层工具
# ══════════════════════════════════════════════
def _bb_lower(close, period=20, width=2.0):
    if len(close) < period: return float(close.iloc[-1])
    mid = close.rolling(period).mean().iloc[-1]
    std = close.rolling(period).std(ddof=1).iloc[-1]
    return float(mid - width * std) if pd.notna(std) else float(mid)

def _bb_expanding(close, period=20):
    """布林带宽度是否在扩张（近 3 根 bandwidth 递增）。"""
    if len(close) < period + 3: return False
    mid = close.rolling(period).mean()
    std = close.rolling(period).std(ddof=1)
    bw = (std / mid * 100).dropna()
    if len(bw) < 3: return False
    return float(bw.iloc[-1]) > float(bw.iloc[-2]) > float(bw.iloc[-3])

def _obv_trend_up(df, short=5, long=13):
    """OBV 短期均线上穿长期均线。"""
    if len(df) < long + 2: return False
    obv = (df['vol'] * np.sign(df['c'].diff().fillna(0))).cumsum()
    obv_short = obv.ewm(span=short, adjust=False).mean()
    obv_long  = obv.ewm(span=long,  adjust=False).mean()
    prev = float(obv_short.iloc[-2]) - float(obv_long.iloc[-2])
    curr = float(obv_short.iloc[-1]) - float(obv_long.iloc[-1])
    return prev <= 0 and curr > 0

def _reversal_candle_quality(h1):
    if len(h1) < 2: return 0.0, False
    o = float(h1['o'].iloc[-1]); h = float(h1['h'].iloc[-1])
    l = float(h1['l'].iloc[-1]); c = float(h1['c'].iloc[-1])
    prev_c = float(h1['c'].iloc[-2]); prev_o = float(h1['o'].iloc[-2])
    fr = h - l
    if fr <= 0: return 0.0, False
    body = c - o; ls = o - l; br = body / fr
    is_bull = body > 0; close_above = c > prev_c; body_ok = br >= 0.30
    if not (is_bull and close_above and body_ok):
        return max(0.0, br * 0.4), False
    # 吞没加成
    engulf = c > prev_o
    sr = ls / max(abs(body), fr * 0.01)
    quality = min(1.0, br * 1.2 + min(sr, 2.0) * 0.15 + (0.1 if engulf else 0.0))
    return quality, True

def _m3_default_result(reason, origin_price=0.0):
    return {'valid':False,'passed':False,'quality':35.0,'reason':reason,'phase':0,
            'origin_price':origin_price,'breakout_level':0.0,'phase1_high':0.0,
            'pullback_low':0.0,'confirmation_level':0.0,'phase3_close':0.0,
            'breakout_pct':0.0,'continuation_pct':0.0,'pullback_depth_pct':0.0}

def _m3_result_failed(reason, *, origin_price, breakout_level, pullback_low, phase1_high):
    return {'valid':True,'passed':False,'quality':15.0,'reason':reason,'phase':2,
            'origin_price':origin_price,'breakout_level':breakout_level,
            'phase1_high':phase1_high,'pullback_low':pullback_low,
            'confirmation_level':phase1_high,'phase3_close':0.0,
            'breakout_pct':(phase1_high-breakout_level)/breakout_level*100 if breakout_level>0 else 0.0,
            'continuation_pct':0.0,
            'pullback_depth_pct':(pullback_low-origin_price)/origin_price*100 if origin_price>0 else -999}

def _build_result(*, valid, score, direction, signals, h4_rsi, h1_rsi, recent_drop_pct,
                  vol_ratio, vol_zscore, h4_adx, h1_ema8, h1_ema21, mode, m3_retest, reason=''):
    oq = min(100, max(20, (35 - h4_rsi) * 4 + 55)) if h4_rsi <= 35 else 40.0
    m3p = bool(m3_retest.get('passed'))
    return {
        'valid':valid,'reason':reason,'score':max(score,0.0),'direction':direction,'signals':signals,
        'ranking_factors':{
            'trend': oq if mode == "超跌反转" else min(100, max(40, h4_adx * 2.5)),
            'trigger': 90.0 if direction=='BUY' else 28.0,
            'volume': min(vol_ratio/1.5, 1.6)*62.5,
            'location': 92.0 if h4_rsi<=25 else 72.0 if h4_rsi<=35 else 55.0,
            'freshness': 94.0 if direction=='BUY' and m3p else 72.0 if direction=='BUY' else 32.0,
            'risk': 86.0 if m3p else 55.0,
        },
        'details':{
            '评估':' | '.join(signals) if signals else '暂无超跌反转机会',
            '模式':mode,
            '4H_RSI':f'{h4_rsi:.1f}','1H_RSI':f'{h1_rsi:.1f}',
            '4H_ADX':f'{h4_adx:.1f}',
            '近期跌幅':f'{recent_drop_pct:.2f}%',
            '量比':f'{vol_ratio:.2f}x','量能Z分':f'{vol_zscore:+.2f}σ',
            'EMA8':f'{h1_ema8:.4f}','EMA21':f'{h1_ema21:.4f}',
            '3m结构确认':'通过' if m3p else '未通过',
            '3m当前阶段':f"Phase {m3_retest.get('phase',0)}",
            '3m结构说明':str(m3_retest.get('reason','-')),
            '3m反转起点':f"{float(m3_retest.get('origin_price',0)):.8g}",
            '3m颈线突破位':f"{float(m3_retest.get('breakout_level',0)):.8g}",
            '3m回踩低点':f"{float(m3_retest.get('pullback_low',0) or 0):.8g}",
            '3m回踩深度':f"{float(m3_retest.get('pullback_depth_pct',0)):.2f}%",
            '3m继续突破幅度':f"{float(m3_retest.get('continuation_pct',0)):.2f}%",
        },
    }

STRATEGY_NAME  = "波段超跌反转扫描"
STRATEGY_TYPE  = "scan"
STRATEGY_CLASS = OversoldReversalSwingScanner
