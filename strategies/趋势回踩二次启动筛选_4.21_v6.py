"""
经典趋势策略：趋势回踩二次启动筛选（精修版 v6 — 扫描 + 回测双入口）

本文件同时支持三种使用场景
─────────────────────────────────────────────────────
A) 实盘扫描（ScannerPage 驱动）
   入口类: TrendPullbackRestartScanner(BaseScannerStrategy)
   - required_bars = ['1D', '4H', '1H', '3m']
   - scan_symbol(symbol) → dict
   - ScannerPage 按 required_bars 自动下载 4 个周期 K 线
   - STRATEGY_TYPE = "scan" 让 StrategyLoader 识别为扫描策略

B) main.py Backtester 回测（BacktestThread 驱动）
   入口类: TrendPullbackBacktestStrategy
   - __init__(config) 空配置实例化
   - on_bar(bar) / generate_signal(bar) / next(bar) 等按根推进
   - 内部自动从 1H 聚合出 4H / 1D
   - 3m 自动跳过（require_3m_restart=False）
   - 需要 ≥ 120 天 1H 数据才会出信号（90 天才够 D1）
   - UI K 线周期必须选 1H

C) 独立纯函数回测（Jupyter / 脚本）
   - analyze_bars(d1, h4, h1, m3, last_price, config) → dict
   - backtest_walk_forward(d1, h4, h1, m3, config, ...) → DataFrame
   - klines_list_to_df(rows) → DataFrame

StrategyLoader 识别约定
─────────────────────────────────────────────────────
STRATEGY_NAME = "趋势回踩二次启动筛选"
STRATEGY_TYPE = "scan"   ← ScannerPage 靠这个发现文件

由于 main.py 的回测页面（第 732 行 get_strategy_class）通常取
模块内第一个策略类或 STRATEGY_CLASS 常量，此处导出两个类并设置：
  STRATEGY_CLASS      = TrendPullbackRestartScanner  （扫描 — 默认）
  BACKTEST_CLASS      = TrendPullbackBacktestStrategy （回测 — 备用）

若 StrategyLoader 只找 BaseScannerStrategy 的子类 → 自动选到扫描器。
若回测页面也用 get_strategy_class → 会拿到扫描器（回测引擎跑不动
scan_symbol，应在 BacktestPage 中直接用 BACKTEST_CLASS）。

★ 实际项目中最稳妥的做法：
  strategies/
    趋势回踩二次启动筛选.py         ← 本文件（扫描，STRATEGY_TYPE="scan"）
    趋势回踩二次启动_回测.py        ← 小文件, from 本文件 import
                                       TrendPullbackBacktestStrategy
                                       STRATEGY_TYPE="backtest"
  这样两条路径完全独立且 StrategyLoader 不会混淆。
  如果你不想拆文件，用本文件即可，回测侧按下面说明手动指定类。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from strategies._shared.indicators import (
    _check_df, _ema, _rsi_wilder, _adx, _efficiency_ratio,
    _volume_ratio_adjusted, _volume_zscore, _latest_swing_levels,
    _local_trend_snapshot, _to_df,
)

# ── 可选：Scanner 基类（仅实盘扫描需要）──
try:
    from src.scanner.base_scanner import BaseScannerStrategy, ScanCondition, ScannerSymbol
    from src.scanner.ranking import build_opportunity_profile
    from src.scanner.trend_quality import trend_quality_snapshot as _external_trend_snapshot
    _HAS_SCANNER_BASE = True
except ImportError:
    BaseScannerStrategy = object  # type: ignore
    ScanCondition       = None    # type: ignore
    ScannerSymbol       = None    # type: ignore
    build_opportunity_profile = None  # type: ignore
    _external_trend_snapshot  = None  # type: ignore
    _HAS_SCANNER_BASE = False


# ══════════════════════════════════════════════
# 评分权重 + 阈值
# ══════════════════════════════════════════════
_W_TREND        = 32
_W_PULLBACK     = 18
_W_RETEST_TIME  =  6
_W_KEYLEVEL     = 10
_W_RESTART      = 20
_W_VOLUME       = 14
_W_3M           =  6

_ADX_MIN_TREND      = 18.0
_EMA_SPREAD_MIN     = 0.25
_RETEST_LOOKBACK    = 12
_RETEST_MIN_TOUCHES = 2

_DEFAULT_CONFIG: Dict[str, Any] = {
    'min_score':                 72,
    'min_volume_24h':             15_000_000,
    'max_pullback_distance_pct':   3.5,
    'keylevel_lookback_bars':      80,
    'max_key_level_retest_pct':    2.2,
    'min_restart_volume_ratio':    1.25,
    'min_volume_zscore':           0.8,
    'max_buy_rsi':                 68.0,
    'min_sell_rsi':                32.0,
    'max_h4_atr_pct':              6.5,
    'require_3m_restart':          True,
    'm3_swing_window':             60,
    'm3_neckline_bars':             8,
    'm3_min_pullback_bars':         3,
    'm3_breakout_buffer_pct':      0.12,
    'm3_origin_tolerance_pct':     0.15,
}


# ══════════════════════════════════════════════
# A) 实盘扫描器（ScannerPage 入口）
# ══════════════════════════════════════════════

class TrendPullbackRestartScanner(BaseScannerStrategy if _HAS_SCANNER_BASE else object):
    """
    ScannerPage 使用的扫描器。
    required_bars 告诉扫描引擎需要下载 4 个周期的 K 线，
    scan_symbol 逐个交易对评估并返回筛选结果。
    """
    required_bars = ['1D', '4H', '1H', '3m']

    # 策略元信息（ScannerPage / StrategyLoader 可能读取）
    name        = "趋势回踩二次启动筛选"
    description = "1D/4H 趋势确认 → 1H 回踩 EMA21 → 放量穿越 → 3m 三段式结构，多周期共振筛选"
    author      = "refined_v6"
    version     = "6.0"
    strategy_type = "scan"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config: Dict[str, Any] = {**_DEFAULT_CONFIG, **(config or {})}
        if _HAS_SCANNER_BASE and hasattr(super(), '__init__'):
            try:
                super().__init__(config or {})
            except Exception:
                pass

    def _init_conditions(self):
        if ScanCondition is None:
            return
        self.add_condition(ScanCondition(
            name="24H成交量",
            description="过滤流动性不足标的",
            field="volume_24h",
            operator=">=",
            value=self.config.get('min_volume_24h', 15_000_000),
        ))

    def scan_symbol(self, symbol) -> Dict:
        """
        ScannerPage 对每个合约对调用此方法。
        symbol: ScannerSymbol 对象
          .inst_id        → 'BTC-USDT-SWAP' 等
          .last_price     → float
          .volume_24h     → float
          .price_change_24h → float
          .extra_data     → {'klines': {'1D': [[ts,o,h,l,c,vol,...], ...], '4H': ..., '1H': ..., '3m': ...}}
        """
        klines_map = symbol.extra_data.get('klines', {})
        try:
            d1 = klines_list_to_df(self._get_klines(klines_map, '1D'))
            h4 = klines_list_to_df(self._get_klines(klines_map, '4H'))
            h1 = klines_list_to_df(self._get_klines(klines_map, '1H'))
            m3 = klines_list_to_df(self._get_klines(klines_map, '3m'))
            cfg = {**_DEFAULT_CONFIG, **(self.config or {})}
            analysis = _analyze_core(d1, h4, h1, m3, symbol.last_price, cfg)
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

        min_score = float(self.config.get('min_score', 72))
        passed = (
            analysis['score'] >= min_score
            and analysis['direction'] in {'BUY', 'SELL'}
        )

        result = {
            'symbol':           symbol.inst_id,
            'passed':           passed,
            'score':            round(analysis['score'], 2),
            'direction':        analysis['direction'],
            'signals':          analysis['signals'],
            'details':          analysis['details'],
            'last_price':       symbol.last_price,
            'volume_24h':       symbol.volume_24h,
            'price_change_24h': symbol.price_change_24h,
            'category':         '趋势回踩二次启动',
            'ranking_factors':  analysis.get('ranking_factors', {}),
        }

        # build_opportunity_profile 在扫描环境中可用
        if build_opportunity_profile is not None:
            try:
                profile = build_opportunity_profile(
                    base_score=analysis['score'],
                    direction=analysis['direction'],
                    volume_24h=symbol.volume_24h,
                    factors=analysis.get('ranking_factors', {}),
                    signals=analysis['signals'],
                )
                result.update(profile)
            except Exception:
                pass

        return result

    def _get_klines(self, klines_map: Dict[str, List], bar: str) -> List:
        return (
            klines_map.get(bar)
            or klines_map.get(bar.lower())
            or klines_map.get(bar.upper())
            or []
        )

    def get_config_schema(self) -> Dict:
        return {
            'min_score':                 {'type': 'int',   'default': 72,         'label': '最低通过分数(0-100)'},
            'min_volume_24h':            {'type': 'float', 'default': 15_000_000, 'label': '最小24H成交额'},
            'max_pullback_distance_pct': {'type': 'float', 'default': 3.5,        'label': '最大回踩4H EMA21距离%'},
            'keylevel_lookback_bars':    {'type': 'int',   'default': 80,         'label': '4H swing 回望窗口'},
            'max_key_level_retest_pct':  {'type': 'float', 'default': 2.2,        'label': '关键位回测容差%'},
            'min_restart_volume_ratio':  {'type': 'float', 'default': 1.25,       'label': '二次启动最小量比'},
            'min_volume_zscore':         {'type': 'float', 'default': 0.8,        'label': '启动量 z-score 下限'},
            'max_buy_rsi':               {'type': 'float', 'default': 68.0,       'label': '做多最大RSI(警示用)'},
            'min_sell_rsi':              {'type': 'float', 'default': 32.0,       'label': '做空最小RSI(警示用)'},
            'max_h4_atr_pct':            {'type': 'float', 'default': 6.5,        'label': '最大4H ATR%(警示用)'},
            'require_3m_restart':        {'type': 'bool',  'default': True,       'label': '要求3m三段式二次启动确认'},
            'm3_swing_window':           {'type': 'int',   'default': 60,         'label': '3m观察窗口(根数)'},
            'm3_neckline_bars':          {'type': 'int',   'default': 8,          'label': '3m颈线样本数'},
            'm3_min_pullback_bars':      {'type': 'int',   'default': 3,          'label': '3m最少回踩根数'},
            'm3_breakout_buffer_pct':    {'type': 'float', 'default': 0.12,       'label': '3m突破缓冲%'},
            'm3_origin_tolerance_pct':   {'type': 'float', 'default': 0.15,       'label': '3m起点容忍偏离%'},
        }


# ══════════════════════════════════════════════
# B) main.py Backtester 适配器（回测入口）
# ══════════════════════════════════════════════

class TrendPullbackBacktestStrategy:
    """
    单周期（1H）驱动的回测适配器。
    main.py Backtester: strategy = TrendPullbackBacktestStrategy({})
    然后逐根 on_bar(bar) 驱动。
    内部自动聚合 1H → 4H → 1D，3m 跳过。

    ★ 最低数据需求：2160+ 根 1H（~90 天），推荐 ≥ 120 天。
    ★ UI K 线周期必须选 1H。
    """
    bar_timeframe  = '1H'
    required_bars  = ['1H']
    name           = "趋势回踩二次启动(回测)"
    description    = "单周期适配：只需 1H 数据，内部自动聚合 4H/1D。3m 不参与。最少 120 天。"
    author         = "refined_v6"
    version        = "6.0"
    strategy_type  = "backtest"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config: Dict[str, Any] = {**_DEFAULT_CONFIG, **(config or {})}
        self.config['require_3m_restart'] = False
        self._h1_bars: List[Dict[str, float]] = []
        self._last_h1_ts: Optional[int] = None
        self.last_analysis: Dict[str, Any] = {}
        self.signal_log: List[Dict[str, Any]] = []

    # ─── 通用入口：适配不同 Backtester 命名 ───
    def on_bar(self, bar, *a, **kw):       return self._handle_bar(bar)
    def generate_signal(self, bar, *a, **kw): return self._handle_bar(bar)
    def next(self, bar, *a, **kw):         return self._handle_bar(bar)
    def process(self, bar, *a, **kw):      return self._handle_bar(bar)
    def step(self, bar, *a, **kw):         return self._handle_bar(bar)
    def analyze(self, bar, *a, **kw):      return self._handle_bar(bar)

    def _handle_bar(self, bar) -> str:
        parsed = _parse_bar(bar)
        if parsed is None:
            return 'WAIT'
        ts = parsed['ts']
        if self._last_h1_ts is not None and ts <= self._last_h1_ts:
            if self._h1_bars and self._h1_bars[-1]['ts'] == ts:
                self._h1_bars[-1] = parsed
            else:
                return 'WAIT'
        else:
            self._h1_bars.append(parsed)
            self._last_h1_ts = ts

        if len(self._h1_bars) < 160:
            return 'WAIT'

        h1_df = pd.DataFrame(self._h1_bars)
        h4_df = _aggregate_bars(h1_df, group_size=4)
        d1_df = _aggregate_bars(h1_df, group_size=24)

        if len(d1_df) < 90 or len(h4_df) < 110:
            return 'WAIT'

        m3_df = pd.DataFrame(columns=['ts', 'o', 'h', 'l', 'c', 'vol'])

        try:
            result = _analyze_core(d1_df, h4_df, h1_df, m3_df,
                                   float(parsed['c']), self.config)
        except Exception:
            self.last_analysis = {'valid': False, 'direction': 'WAIT', 'score': 0.0}
            return 'WAIT'

        self.last_analysis = result
        min_score = float(self.config.get('min_score', 72))
        passed = (result.get('valid') and result.get('score', 0) >= min_score
                  and result.get('direction') in {'BUY', 'SELL'})
        direction = result.get('direction', 'WAIT') if passed else 'WAIT'

        if direction in {'BUY', 'SELL'}:
            self.signal_log.append({
                'ts': ts, 'price': float(parsed['c']),
                'direction': direction, 'score': float(result.get('score', 0)),
                'signals': result.get('signals', []),
            })
        return direction

    def reset(self):
        self._h1_bars.clear()
        self._last_h1_ts = None
        self.last_analysis = {}
        self.signal_log.clear()

    def get_config_schema(self) -> Dict:
        return {
            'min_score':                 {'type': 'int',   'default': 72,   'label': '最低通过分数(0-100)'},
            'max_pullback_distance_pct': {'type': 'float', 'default': 3.5,  'label': '最大回踩4H EMA21距离%'},
            'min_restart_volume_ratio':  {'type': 'float', 'default': 1.25, 'label': '二次启动最小量比'},
            'min_volume_zscore':         {'type': 'float', 'default': 0.8,  'label': '启动量 z-score 下限'},
            'max_key_level_retest_pct':  {'type': 'float', 'default': 2.2,  'label': '关键位回测容差%'},
        }


# ══════════════════════════════════════════════
# C) 纯函数公开 API（独立回测 / Jupyter）
# ══════════════════════════════════════════════

def analyze_bars(d1, h4, h1, m3, last_price, config=None):
    """单点评估。参数同 v4/v5。"""
    cfg = {**_DEFAULT_CONFIG, **(config or {})}
    try:
        return _analyze_core(d1, h4, h1, m3, last_price, cfg)
    except Exception as exc:
        return {
            'valid': False, 'reason': f'分析异常: {exc}', 'score': 0.0,
            'direction': 'WAIT', 'passed': False, 'signals': [],
            'details': {'状态': f'分析异常: {exc}'}, 'ranking_factors': {},
        }


def klines_list_to_df(rows):
    """交易所 list-of-list → DataFrame。"""
    return _to_df(rows)


def backtest_walk_forward(d1, h4, h1, m3, config=None, *, start_idx=None,
                          end_idx=None, stride=1, min_score=72.0,
                          emit_only_passed=True, verbose=False):
    """独立多周期 walk-forward 回测。参数同 v4/v5。"""
    cfg = {**_DEFAULT_CONFIG, **(config or {})}; cfg['min_score'] = min_score
    for name, df in [('d1',d1),('h4',h4),('h1',h1),('m3',m3)]:
        for col in ('ts','o','h','l','c','vol'):
            if col not in df.columns:
                raise ValueError(f"{name} 缺少必要列 '{col}'")
    d1=d1.sort_values('ts').reset_index(drop=True)
    h4=h4.sort_values('ts').reset_index(drop=True)
    h1=h1.sort_values('ts').reset_index(drop=True)
    m3=m3.sort_values('ts').reset_index(drop=True)
    d1_ts=d1['ts'].values; h4_ts=h4['ts'].values; m3_ts=m3['ts'].values; h1_ts=h1['ts'].values
    if start_idx is None: start_idx=140
    if end_idx is None: end_idx=len(h1)
    start_idx=max(start_idx,140); end_idx=min(end_idx,len(h1))
    records=[]
    for i in range(start_idx,end_idx,stride):
        t=h1_ts[i]
        d1_end=int(np.searchsorted(d1_ts,t,side='right'))
        h4_end=int(np.searchsorted(h4_ts,t,side='right'))
        m3_end=int(np.searchsorted(m3_ts,t,side='right'))
        lp=float(h1['c'].iloc[i])
        try:
            r=_analyze_core(d1.iloc[:d1_end],h4.iloc[:h4_end],h1.iloc[:i+1],m3.iloc[:m3_end],lp,cfg)
        except Exception: continue
        if not r.get('valid'): continue
        passed=r['score']>=min_score and r['direction'] in {'BUY','SELL'}
        r['passed']=passed
        if emit_only_passed and not passed: continue
        det=r.get('details',{})
        records.append({
            'ts_1h':int(t),
            'dt_utc':pd.to_datetime(int(t),unit='ms',utc=True) if t>1e12 else pd.to_datetime(int(t),unit='s',utc=True),
            'price':lp,'direction':r['direction'],'score':round(r['score'],2),'passed':passed,
            'pullback_depth':_parse_pct(det.get('回踩距离')),
            'vol_ratio':_parse_x(det.get('量比')),
            'vol_zscore':_parse_sigma(det.get('量能Z分')),
            'close_strength':_parse_float(det.get('启动收盘强度')),
            'h1_rsi':_parse_float(det.get('1H_RSI')),
            'd1_adx':_parse_float(det.get('日线ADX')),
            'h4_adx':_parse_float(det.get('4H_ADX')),
            'signals':' ; '.join(r.get('signals',[])),
        })
    df=pd.DataFrame(records)
    if not df.empty: df=df.sort_values('ts_1h').reset_index(drop=True)
    return df


# ══════════════════════════════════════════════
# 核心分析引擎（_analyze_core）
# ══════════════════════════════════════════════

def _analyze_core(d1, h4, h1, m3, last_price, cfg):
    _check_df(d1,'日线',90); _check_df(h4,'4H',110); _check_df(h1,'1H',140)
    require_3m=bool(cfg.get('require_3m_restart',True))
    min_3m_bars=max(int(cfg.get('m3_swing_window',60)),int(cfg.get('m3_neckline_bars',8))+int(cfg.get('m3_min_pullback_bars',3))+6)
    if require_3m and len(m3)<min_3m_bars: _check_df(m3,'3m',min_3m_bars)
    score=0.0; signals=[]; price=float(last_price) if last_price and last_price>0 else float(h1['c'].iloc[-1])
    d1_ema21=_ema(d1['c'],21);d1_ema55=_ema(d1['c'],55)
    h4_ema21=_ema(h4['c'],21);h4_ema55=_ema(h4['c'],55)
    h1_ema21=_ema(h1['c'],21);h1_ema55=_ema(h1['c'],55)
    h1_last_close=float(h1['c'].iloc[-1]);h1_prev_close=float(h1['c'].iloc[-2])
    d1_slope=_ema_slope_pct(d1['c'],21,6);h4_slope=_ema_slope_pct(h4['c'],21,6)
    h1_rsi=_rsi_wilder(h1['c']);h4_atr_pct=_atr_pct(h4)
    vol_ratio=_volume_ratio_adjusted(h1);vol_zscore=_volume_zscore(h1['vol'])
    d1_adx=_adx(d1,14);h4_adx=_adx(h4,14)
    d1_ema_spread=(d1_ema21-d1_ema55)/d1_ema55*100 if d1_ema55>0 else 0.0
    h4_ema_spread=(h4_ema21-h4_ema55)/h4_ema55*100 if h4_ema55>0 else 0.0
    _m3ph=_m3_default_result('趋势未确认，跳过3m')
    trend_snap=_trend_snapshot(d1,h4,h1,price)
    trend_metrics=trend_snap.get('metrics',{})
    trend_long_score=float(trend_snap.get('long_score',0.0) or 0.0)
    trend_short_score=float(trend_snap.get('short_score',0.0) or 0.0)
    bullish_trend=(
        bool(trend_snap.get('long_ok'))
        and price>d1_ema21>d1_ema55 and price>h4_ema21>h4_ema55
        and d1_slope>0.4 and h4_slope>0.6
        and d1_adx>=_ADX_MIN_TREND and h4_adx>=_ADX_MIN_TREND
        and d1_ema_spread>=_EMA_SPREAD_MIN and h4_ema_spread>=_EMA_SPREAD_MIN)
    bearish_trend=(
        bool(trend_snap.get('short_ok'))
        and price<d1_ema21<d1_ema55 and price<h4_ema21<h4_ema55
        and d1_slope<-0.4 and h4_slope<-0.6
        and d1_adx>=_ADX_MIN_TREND and h4_adx>=_ADX_MIN_TREND
        and d1_ema_spread<=-_EMA_SPREAD_MIN and h4_ema_spread<=-_EMA_SPREAD_MIN)
    # ① 趋势（32 分）
    if bullish_trend or bearish_trend:
        base_t=24.0
        spread_abs=(abs(d1_ema_spread)+abs(h4_ema_spread))/2.0
        spread_bonus=min(5.0,spread_abs/2.0*5.0)
        adx_avg=(d1_adx+h4_adx)/2.0
        adx_bonus=min(3.0,max(0.0,(adx_avg-_ADX_MIN_TREND)/12.0*3.0))
        ts_=base_t+spread_bonus+adx_bonus; score+=ts_
        dl="多头" if bullish_trend else "空头"
        rs=trend_long_score if bullish_trend else trend_short_score
        signals.append(f"{dl}趋势通过(质量{rs:.0f}, ADX {adx_avg:.1f}, 发散{spread_abs:.2f}% → +{ts_:.1f}分)")
    else:
        rp=[]
        if not(bool(trend_snap.get('long_ok')) or bool(trend_snap.get('short_ok'))): rp.append(str(trend_snap.get('reason','趋势质量不足')))
        if d1_adx<_ADX_MIN_TREND: rp.append(f"日线ADX不足({d1_adx:.1f})")
        if h4_adx<_ADX_MIN_TREND: rp.append(f"4H ADX不足({h4_adx:.1f})")
        if abs(d1_ema_spread)<_EMA_SPREAD_MIN: rp.append(f"日线EMA发散不足({d1_ema_spread:.2f}%)")
        if abs(h4_ema_spread)<_EMA_SPREAD_MIN: rp.append(f"4H EMA发散不足({h4_ema_spread:.2f}%)")
        if not(d1_slope>0.4 or d1_slope<-0.4): rp.append(f"日线斜率不足({d1_slope:.2f}%)")
        if not(h4_slope>0.6 or h4_slope<-0.6): rp.append(f"4H斜率不足({h4_slope:.2f}%)")
        signals.append("趋势未确认: "+" | ".join(rp) if rp else "趋势未确认")
        return _build_result(valid=True,score=score,direction='WAIT',signals=signals,pullback_depth=0.0,retest_quality=0.0,vol_ratio=vol_ratio,vol_zscore=vol_zscore,close_strength=0.0,h1_rsi=h1_rsi,h4_atr_pct=h4_atr_pct,d1_adx=d1_adx,h4_adx=h4_adx,d1_ema_spread=d1_ema_spread,h4_ema_spread=h4_ema_spread,trend_long_score=trend_long_score,trend_short_score=trend_short_score,bullish_trend=bullish_trend,bearish_trend=bearish_trend,d1_slope=d1_slope,h4_slope=h4_slope,trend_metrics=trend_metrics,m3_confirm=_m3ph)
    # ② 方向性回踩（18 分）
    raw_distance=(price-h4_ema21)/h4_ema21*100 if h4_ema21>0 else 999.0
    max_pb=float(cfg.get('max_pullback_distance_pct',3.5))
    if bullish_trend: pullback_ok=0.0<=raw_distance<=max_pb; pullback_depth=abs(raw_distance); direction_mismatch=raw_distance<0
    else: pullback_ok=-max_pb<=raw_distance<=0.0; pullback_depth=abs(raw_distance); direction_mismatch=raw_distance>0
    if pullback_ok:
        pb_score=_W_PULLBACK*min(1.0,0.55+(1.0-pullback_depth/max(max_pb,0.1))*0.45); score+=pb_score
        signals.append(f"方向性回踩4H EMA21({pullback_depth:.2f}% → +{pb_score:.1f}分)")
    elif direction_mismatch: signals.append(f"价格已穿越EMA21至反向侧({raw_distance:+.2f}%)，非健康回踩")
    else: signals.append(f"距均线过远({pullback_depth:.2f}%/{max_pb}%)")
    # ③ 回踩时间（6 分）
    h1_ema21_s=h1['c'].ewm(span=21,adjust=False).mean()
    rw=h1.tail(_RETEST_LOOKBACK); ew=h1_ema21_s.tail(_RETEST_LOOKBACK)
    tc=int((rw['l'].values<=ew.values).sum()) if bullish_trend else int((rw['h'].values>=ew.values).sum())
    if tc>=_RETEST_MIN_TOUCHES:
        rt=_W_RETEST_TIME*min(1.0,tc/5.0); score+=rt
        signals.append(f"回踩过程真实(近{_RETEST_LOOKBACK}根触碰{tc}根 → +{rt:.1f}分)")
    else: signals.append(f"未见真实回踩过程(仅{tc}根触及/需{_RETEST_MIN_TOUCHES})")
    # ④ 关键位（10 分）
    sh,sl=_latest_swing_levels(h4,left=5,right=5,skip_recent=3,max_lookback=80)
    retest_quality=0.0; max_kl=float(cfg.get('max_key_level_retest_pct',2.2))
    if bullish_trend and sh and sh>0 and price>0 and sh<price:
        retest_quality=(price-sh)/sh*100
        if retest_quality<=max_kl: kls=_W_KEYLEVEL*max(0.5,1.0-retest_quality/max(max_kl,0.1)); score+=kls; signals.append(f"回踩前swing高支撑({retest_quality:.2f}% → +{kls:.1f}分)")
    elif bearish_trend and sl and sl>0 and price>0 and sl>price:
        retest_quality=(sl-price)/sl*100
        if retest_quality<=max_kl: kls=_W_KEYLEVEL*max(0.5,1.0-retest_quality/max(max_kl,0.1)); score+=kls; signals.append(f"反抽前swing低压力({retest_quality:.2f}% → +{kls:.1f}分)")
    # ⑤ 重启 + 收盘强度（20 分）
    restart_up=bullish_trend and h1_prev_close<=h1_ema21 and h1_last_close>h1_ema21 and h1_ema21>h1_ema55
    restart_down=bearish_trend and h1_prev_close>=h1_ema21 and h1_last_close<h1_ema21 and h1_ema21<h1_ema55
    last_h1=h1.iloc[-1]; br=float(last_h1['h'])-float(last_h1['l'])
    if br>0: close_strength=((float(last_h1['c'])-float(last_h1['l']))/br if bullish_trend else (float(last_h1['h'])-float(last_h1['c']))/br)
    else: close_strength=0.5
    if restart_up or restart_down:
        b=_W_RESTART*0.7; sb=_W_RESTART*0.3*max(0.0,(close_strength-0.5)/0.5); rs_=b+sb; score+=rs_
        arr="站上" if restart_up else "跌破"
        signals.append(f"1H收盘{arr}EMA21(收盘强度{close_strength:.2f} → +{rs_:.1f}分)")
    else:
        pts=[]
        cc=((h1_prev_close<=h1_ema21 and h1_last_close>h1_ema21) if bullish_trend else (h1_prev_close>=h1_ema21 and h1_last_close<h1_ema21))
        if not cc: pts.append(f"EMA21穿越未触发(prev={h1_prev_close:.4g},last={h1_last_close:.4g},ema={h1_ema21:.4g})")
        co=(h1_ema21>h1_ema55) if bullish_trend else (h1_ema21<h1_ema55)
        if not co: pts.append("1H EMA排列不支持")
        signals.append("未二次启动: "+" | ".join(pts) if pts else "1H尚未二次启动")
    # ⑥ 量能（14 分）
    min_vr=float(cfg.get('min_restart_volume_ratio',1.25)); min_zs=float(cfg.get('min_volume_zscore',0.8))
    vrok=vol_ratio>=min_vr; vzok=vol_zscore>=min_zs
    if vrok and vzok:
        rc=0.55+min(1.0,(vol_ratio-min_vr)/max(min_vr,0.1)*0.25)*0.45
        zc=0.55+min(1.0,(vol_zscore-min_zs)/1.5)*0.45
        vs=_W_VOLUME*(rc+zc)/2.0; score+=vs
        signals.append(f"启动量能强({vol_ratio:.2f}x, z={vol_zscore:+.2f} → +{vs:.1f}分)")
    elif vrok or vzok:
        vs=_W_VOLUME*0.45; score+=vs
        signals.append(f"量能部分满足({vol_ratio:.2f}x, z={vol_zscore:+.2f} → +{vs:.1f}分)")
    else:
        signals.append(f"量能不足(量比{vol_ratio:.2f}x/需{min_vr}x, z={vol_zscore:+.2f}/需{min_zs:+.2f})")
    # RSI / ATR 警示
    max_buy_rsi=float(cfg.get('max_buy_rsi',68.0)); min_sell_rsi=float(cfg.get('min_sell_rsi',32.0))
    if(bullish_trend and h1_rsi>max_buy_rsi)or(bearish_trend and h1_rsi<min_sell_rsi):
        signals.append(f"RSI警示({'超买' if bullish_trend else '超卖'}: {h1_rsi:.1f})")
    if h4_atr_pct>float(cfg.get('max_h4_atr_pct',6.5)):
        signals.append(f"4H波动偏大({h4_atr_pct:.2f}%)")
    # ⑦ 3m（bonus 6 分）
    m3_bias=bullish_trend
    if restart_up or restart_down:
        m3_confirm=(_three_min_restart_core(m3,bull_bias=m3_bias,cfg=cfg) if len(m3)>=min_3m_bars else _m3_default_result('3m数据不足，跳过结构确认'))
    else: m3_confirm=_m3_default_result('1H未穿越，跳过3m结构')
    if m3_confirm.get('passed'): score+=_W_3M; signals.append(str(m3_confirm.get('reason')))
    elif m3_confirm.get('valid') and(restart_up or restart_down): signals.append(f"3m结构观察：{m3_confirm.get('reason')}")
    # 方向判定
    direction='WAIT'
    if restart_up and pullback_ok and(not require_3m or bool(m3_confirm.get('passed'))): direction='BUY'
    elif restart_down and pullback_ok and(not require_3m or bool(m3_confirm.get('passed'))): direction='SELL'
    return _build_result(valid=True,score=score,direction=direction,signals=signals,pullback_depth=pullback_depth,retest_quality=retest_quality,vol_ratio=vol_ratio,vol_zscore=vol_zscore,close_strength=close_strength,h1_rsi=h1_rsi,h4_atr_pct=h4_atr_pct,d1_adx=d1_adx,h4_adx=h4_adx,d1_ema_spread=d1_ema_spread,h4_ema_spread=h4_ema_spread,trend_long_score=trend_long_score,trend_short_score=trend_short_score,bullish_trend=bullish_trend,bearish_trend=bearish_trend,d1_slope=d1_slope,h4_slope=h4_slope,trend_metrics=trend_metrics,m3_confirm=m3_confirm)


# ══════════════════════════════════════════════
# 3m 状态机 / 趋势快照 / 底层工具
# ══════════════════════════════════════════════

def _three_min_restart_core(df,*,bull_bias,cfg):
    window=int(cfg.get('m3_swing_window',60));neckline_bars=int(cfg.get('m3_neckline_bars',8))
    origin_tol=float(cfg.get('m3_origin_tolerance_pct',0.15));breakout_buf=float(cfg.get('m3_breakout_buffer_pct',0.12))
    min_pb_bars=int(cfg.get('m3_min_pullback_bars',3))
    min_bars=neckline_bars+min_pb_bars+6
    if len(df)<max(window,min_bars): return _m3_default_result(f'3m数据不足({len(df)}/{max(window,min_bars)})')
    recent=df.tail(window).reset_index(drop=True); n=len(recent)
    ose=max(neckline_bars+min_pb_bars+4,n//2)
    if bull_bias: op=int(recent.iloc[:ose]['l'].idxmin()); opr=float(recent['l'].iloc[op])
    else: op=int(recent.iloc[:ose]['h'].idxmax()); opr=float(recent['h'].iloc[op])
    ns_=op+1; ne=min(ns_+neckline_bars,n-min_pb_bars-3)
    if ne<=ns_+1: return _m3_default_result(f'3m颈线样本不足({"多头" if bull_bias else "空头"})',origin_price=opr)
    neck=recent.iloc[ns_:ne]
    if bull_bias:
        bl=float(neck['h'].max())
        if not pd.notna(bl) or bl<=0: return _m3_default_result('3m颈线计算异常',origin_price=opr)
        tl=bl*(1.0+breakout_buf/100.0); fl=opr*(1.0-origin_tol/100.0)
    else:
        bl=float(neck['l'].min())
        if not pd.notna(bl) or bl<=0: return _m3_default_result('3m颈线计算异常(空头)',origin_price=opr)
        tl=bl*(1.0-breakout_buf/100.0); fl=opr*(1.0+origin_tol/100.0)
    phase=0;p1e=opr;cl_=tl;pbe=float('inf') if bull_bias else float('-inf');pbc=0;p3c=0.0
    for i in range(ne,n):
        bc=float(recent['c'].iloc[i]);bh=float(recent['h'].iloc[i]);bll=float(recent['l'].iloc[i])
        if bull_bias:
            if phase==0:
                if bc>tl: phase=1;p1e=bh;cl_=bh
            elif phase==1:
                if bc>tl:
                    if bh>p1e: p1e=bh;cl_=bh
                else:
                    if bc<fl: return _m3_result_failed(f'Phase2首棒收盘跌破起点(close={bc:.6g})',origin_price=opr,breakout_level=bl,pullback_extreme=bc,phase1_extreme=p1e,bull_bias=True)
                    phase=2;pbe=bll;pbc=1
            elif phase==2:
                pbe=min(pbe,bll);pbc+=1
                if bc<fl: return _m3_result_failed(f'Phase2收盘跌破起点(close至{bc:.6g})',origin_price=opr,breakout_level=bl,pullback_extreme=pbe,phase1_extreme=p1e,bull_bias=True)
                if pbc>=min_pb_bars and bc>cl_: phase=3;p3c=bc;break
        else:
            if phase==0:
                if bc<tl: phase=1;p1e=bll;cl_=bll
            elif phase==1:
                if bc<tl:
                    if bll<p1e: p1e=bll;cl_=bll
                else:
                    if bc>fl: return _m3_result_failed(f'Phase2首棒收盘反抽超起点(close={bc:.6g})',origin_price=opr,breakout_level=bl,pullback_extreme=bc,phase1_extreme=p1e,bull_bias=False)
                    phase=2;pbe=bh;pbc=1
            elif phase==2:
                pbe=max(pbe,bh);pbc+=1
                if bc>fl: return _m3_result_failed(f'Phase2收盘反抽超起点(close至{bc:.6g})',origin_price=opr,breakout_level=bl,pullback_extreme=pbe,phase1_extreme=p1e,bull_bias=False)
                if pbc>=min_pb_bars and bc<cl_: phase=3;p3c=bc;break
    lc=float(recent['c'].iloc[-1])
    if phase==3:
        if bull_bias: dp=(pbe-opr)/opr*100; cp=(p3c-cl_)/cl_*100; held=lc>p3c
        else: dp=(opr-pbe)/opr*100; cp=(cl_-p3c)/cl_*100; held=lc<p3c
        q=min(100.0,60.0+min(dp+2.0,4.0)*5.0+min(cp,3.0)*4.0+(8.0 if held else 0.0))
        lb="多头" if bull_bias else "空头"
        return {'valid':True,'passed':True,'quality':q,'reason':f"3m{lb}三段确认：突破→回踩({dp:+.2f}%)→继续突破(+{cp:.2f}%)",'phase':3,'bull_bias':bull_bias,'origin_price':opr,'breakout_level':bl,'phase1_extreme':p1e,'pullback_extreme':pbe,'confirmation_level':cl_,'phase3_close':p3c,'breakout_pct':abs(p1e-bl)/bl*100,'continuation_pct':cp,'pullback_depth_pct':dp}
    elif phase==2:
        if bull_bias: dp=(pbe-opr)/opr*100; fh=pbe>=fl
        else: dp=(opr-pbe)/opr*100; fh=pbe<=fl
        q=35.0+25.0+(20.0 if fh else 0.0)
        r=f"3m已突破并回踩({dp:+.2f}%)，等待继续突破确认" if fh else f"3m回踩越过起点({dp:+.2f}%)"
        return {'valid':True,'passed':False,'quality':float(q),'reason':r,'phase':2,'bull_bias':bull_bias,'origin_price':opr,'breakout_level':bl,'phase1_extreme':p1e,'pullback_extreme':pbe,'confirmation_level':cl_,'phase3_close':0.0,'breakout_pct':abs(p1e-bl)/bl*100,'continuation_pct':0.0,'pullback_depth_pct':dp}
    elif phase==1:
        lb="多头" if bull_bias else "空头"
        return {'valid':True,'passed':False,'quality':60.0,'reason':f"3m{lb}已突破颈线，等待回踩确认(Phase1极值={p1e:.6g})",'phase':1,'bull_bias':bull_bias,'origin_price':opr,'breakout_level':bl,'phase1_extreme':p1e,'pullback_extreme':float('nan'),'confirmation_level':cl_,'phase3_close':0.0,'breakout_pct':abs(p1e-bl)/bl*100,'continuation_pct':0.0,'pullback_depth_pct':0.0}
    else:
        return _m3_default_result(f'3m尚未完成初次{"向上" if bull_bias else "向下"}突破',origin_price=opr)

def _trend_snapshot(d1,h4,h1,price):
    if _external_trend_snapshot:
        try: return _external_trend_snapshot(d1,h4,h1,price)
        except Exception: pass
    return _local_trend_snapshot(d1,h4,h1,price)

# ── 底层工具 ──
def _ema_slope_pct(c,span,lb):
    e=c.ewm(span=span,adjust=False).mean()
    if len(e)<=lb: return 0.0
    b=float(e.iloc[-(lb+1)]); l=float(e.iloc[-1])
    return (l/b-1.0)*100 if b>0 else 0.0
def _atr_pct(df,period=14):
    if len(df)<period+1: return 0.0
    pc=df['c'].shift(1)
    tr=pd.concat([df['h']-df['l'],(df['h']-pc).abs(),(df['l']-pc).abs()],axis=1).max(axis=1)
    a=tr.ewm(alpha=1/period,adjust=False).mean().iloc[-1]
    lc=float(df['c'].iloc[-1])
    if pd.isna(a) or lc<=0: return 0.0
    return float(a/lc*100)
def _m3_default_result(reason,origin_price=0.0):
    return {'valid':False,'passed':False,'quality':35.0,'reason':reason,'phase':0,'bull_bias':True,'origin_price':origin_price,'breakout_level':0.0,'phase1_extreme':0.0,'pullback_extreme':0.0,'confirmation_level':0.0,'phase3_close':0.0,'breakout_pct':0.0,'continuation_pct':0.0,'pullback_depth_pct':0.0}
def _m3_result_failed(reason,*,origin_price,breakout_level,pullback_extreme,phase1_extreme,bull_bias):
    bp=abs(phase1_extreme-breakout_level)/breakout_level*100 if breakout_level>0 else 0.0
    dp=(pullback_extreme-origin_price)/origin_price*100 if origin_price>0 else -999.0
    return {'valid':True,'passed':False,'quality':15.0,'reason':reason,'phase':2,'bull_bias':bull_bias,'origin_price':origin_price,'breakout_level':breakout_level,'phase1_extreme':phase1_extreme,'pullback_extreme':pullback_extreme,'confirmation_level':phase1_extreme,'phase3_close':0.0,'breakout_pct':bp,'continuation_pct':0.0,'pullback_depth_pct':dp}
def _build_result(*,valid,score,direction,signals,pullback_depth,retest_quality,vol_ratio,vol_zscore,close_strength,h1_rsi,h4_atr_pct,d1_adx,h4_adx,d1_ema_spread,h4_ema_spread,trend_long_score,trend_short_score,bullish_trend,bearish_trend,d1_slope,h4_slope,trend_metrics,m3_confirm,reason=''):
    tq=trend_long_score if bullish_trend else trend_short_score if bearish_trend else max(trend_long_score,trend_short_score)
    lq=max(20.0,100.0-abs(pullback_depth-1.2)*20.0); vq=min(vol_ratio/1.25,1.6)*62.5
    fq=94.0 if direction in{'BUY','SELL'} and pullback_depth<=2.2 else 68.0 if direction in{'BUY','SELL'} else 34.0
    return {'valid':valid,'reason':reason,'score':max(score,0.0),'direction':direction,'signals':signals,
        'ranking_factors':{'trend':tq,'trigger':92.0 if direction in{'BUY','SELL'} else 32.0,'volume':vq,'location':lq,'freshness':fq,'risk':88.0 if h4_atr_pct<=6.5 else 55.0},
        'details':{'评估':' | '.join(signals) if signals else '暂无趋势回踩二次启动机会','日线斜率':f'{d1_slope:.2f}%','4H斜率':f'{h4_slope:.2f}%','日线ADX':f'{d1_adx:.1f}','4H_ADX':f'{h4_adx:.1f}','日线EMA发散':f'{d1_ema_spread:+.2f}%','4H_EMA发散':f'{h4_ema_spread:+.2f}%','趋势质量':f'{tq:.1f}','趋势诊断':str(trend_metrics.get('reason','') or ''),'H4_ADX_内置':f"{float(trend_metrics.get('h4_adx',0.0)):.1f}",'趋势效率':f"{float(trend_metrics.get('h1_efficiency',0.0)):.1f}",'回踩距离':f'{pullback_depth:.2f}%','关键位回测':f'{retest_quality:.2f}%','量比':f'{vol_ratio:.2f}x','量能Z分':f'{vol_zscore:+.2f}σ','启动收盘强度':f'{close_strength:.2f}','1H_RSI':f'{h1_rsi:.1f}','4H_ATR%':f'{h4_atr_pct:.2f}%','3m结构确认':'通过' if m3_confirm.get('passed') else '未通过','3m当前阶段':f"Phase {m3_confirm.get('phase',0)}",'3m结构说明':str(m3_confirm.get('reason','-')),'_m3_raw':m3_confirm}}
def _aggregate_bars(h1,group_size):
    n=len(h1)
    if n<group_size: return pd.DataFrame(columns=h1.columns)
    usable=(n//group_size)*group_size
    h1t=h1.iloc[n-usable:].reset_index(drop=True)
    g=h1t.groupby(h1t.index//group_size)
    return pd.DataFrame({'ts':g['ts'].first(),'o':g['o'].first(),'h':g['h'].max(),'l':g['l'].min(),'c':g['c'].last(),'vol':g['vol'].sum()}).reset_index(drop=True)
def _parse_bar(bar):
    try:
        if isinstance(bar,dict):
            gv=lambda *ks: next((bar[k] for k in ks if k in bar),None)
            ts=gv('ts','timestamp','time');o=gv('o','open');h=gv('h','high');l=gv('l','low');c=gv('c','close');vol=gv('vol','volume')
        elif isinstance(bar,(list,tuple)) and len(bar)>=6: ts,o,h,l,c,vol=bar[0],bar[1],bar[2],bar[3],bar[4],bar[5]
        elif isinstance(bar,pd.Series):
            gv=lambda *ks: next((bar[k] for k in ks if k in bar.index),None)
            ts=gv('ts','timestamp','time');o=gv('o','open');h=gv('h','high');l=gv('l','low');c=gv('c','close');vol=gv('vol','volume')
        else:
            gv=lambda *ks: next((getattr(bar,k) for k in ks if hasattr(bar,k)),None)
            ts=gv('ts','timestamp','time');o=gv('o','open');h=gv('h','high');l=gv('l','low');c=gv('c','close');vol=gv('vol','volume')
        if None in(ts,o,h,l,c): return None
        return {'ts':int(pd.Timestamp(ts).value//1_000_000) if not isinstance(ts,(int,float)) else int(ts),'o':float(o),'h':float(h),'l':float(l),'c':float(c),'vol':float(vol) if vol is not None else 0.0}
    except Exception: return None
def _parse_float(v):
    if v is None: return 0.0
    try: return float(str(v).strip())
    except: return 0.0
def _parse_pct(v):
    if v is None: return 0.0
    try: return float(str(v).strip().rstrip('%'))
    except: return 0.0
def _parse_x(v):
    if v is None: return 0.0
    try: return float(str(v).strip().rstrip('x').rstrip('X'))
    except: return 0.0
def _parse_sigma(v):
    if v is None: return 0.0
    try: return float(str(v).strip().rstrip('σ'))
    except: return 0.0
def _safe_get_m3(r):
    raw=r.get('details',{}).get('_m3_raw')
    return raw if isinstance(raw,dict) else {}


# ══════════════════════════════════════════════
# 模块级导出
# ══════════════════════════════════════════════
STRATEGY_NAME  = "趋势回踩二次启动筛选"
STRATEGY_TYPE  = "scan"      # ← ScannerPage 靠这个发现此文件
STRATEGY_CLASS = TrendPullbackRestartScanner  # ← 默认策略类 = 扫描器
BACKTEST_CLASS = TrendPullbackBacktestStrategy  # ← 回测用的适配类

__all__ = [
    'TrendPullbackRestartScanner',
    'TrendPullbackBacktestStrategy',
    'analyze_bars', 'backtest_walk_forward', 'klines_list_to_df',
    '_local_trend_snapshot', '_DEFAULT_CONFIG',
    'STRATEGY_NAME', 'STRATEGY_TYPE', 'STRATEGY_CLASS', 'BACKTEST_CLASS',
]
