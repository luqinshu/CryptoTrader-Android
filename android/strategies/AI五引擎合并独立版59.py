#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI五引擎合并独立版

将五个子策略类与组合扫描器合并为单一文件，无需外部文件加载。

子策略：
  1. ACrossSectionalMultiFactorScannerStrategy  — 截面多因子
  2. AIAutomatedAlphaCryptoScannerStrategy      — AI因子挖掘
  3. DRLMetaHourlyTrendStartScannerStrategy     — DRL小时趋势启动
  4. XGBoostCrossSectionalRanker               — XGBoost截面排序
  5. AIOrderflowMomentumBreakoutScanner        — AI订单流动量

组合器：
  AICrossSectionDualFactorComboScanner         — 五引擎组合
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from math import exp, log, log2, log10, sqrt
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from src.scanner.base_scanner import BaseScannerStrategy, ScanCondition, ScannerSymbol
    from src.scanner.ranking import build_opportunity_profile, enrich_scan_result
    _HAS_SCANNER_BASE = True
except Exception:
    BaseScannerStrategy = object
    ScanCondition = None
    ScannerSymbol = Any
    build_opportunity_profile = None
    enrich_scan_result = None
    _HAS_SCANNER_BASE = False

try:
    from src.scanner.base_scanner import BaseScannerStrategy as _BaseScan2
    _HAS_BASE = _HAS_SCANNER_BASE
except Exception:
    _HAS_BASE = _HAS_SCANNER_BASE

from strategies._shared.indicators import (
    _to_df, _aggregate_bars, _ema, _rsi_wilder, _adx, _efficiency_ratio,
    _volume_zscore, _robust_zscore, _measure_trend_age, _micro_pullback_continuation,
    _safe_float, _clamp, _calc_atr, _calc_volume_delta, _calc_vwap,
    _pct_change, _cfg_float,
)

# v6.0 合并文件遗留 bug 修复：子策略 1459/2227/3270/3271/3305 行使用 _rsi（无前缀），
# 但合并文件只导入了 _rsi_wilder。补一个别名让子策略代码可以直接调用 _rsi。
_rsi = _rsi_wilder



# ════════════════════════════════════════════════════════════════════════════
# 子策略 1 — 截面多因子加密货币扫描
# ════════════════════════════════════════════════════════════════════════════

_cs1_CONFIG_SCHEMA = {
    'min_score':              {'type':'float','default':72.0,      'label':'最低扫描分数'},
    'backtest_min_score':     {'type':'float','default':70.0,      'label':'回测最低入场分数'},
    'min_volume_24h':         {'type':'float','default':8_000_000, 'label':'最小24H成交额'},
    'top_n_long':             {'type':'int',  'default':12,        'label':'截面多头保留'},
    'top_n_short':            {'type':'int',  'default':8,         'label':'截面空头保留'},
    'allow_short':            {'type':'bool', 'default':True,      'label':'允许空头'},
    'min_abs_edge':           {'type':'float','default':0.38,      'label':'最小绝对优势'},
    'max_atr_pct':            {'type':'float','default':7.5,       'label':'最大4H ATR%'},
    'use_dynamic_ic_weights': {'type':'bool', 'default':True,      'label':'启用动态IC权重'},
    'ic_lookback_points':     {'type':'int',  'default':42,        'label':'IC回看截面数'},
    'ic_forward_bars':        {'type':'int',  'default':6,         'label':'IC未来收益窗口'},
    'ic_ewm_span':            {'type':'int',  'default':12,        'label':'IC EWMA周期'},
    'ic_min_assets':          {'type':'int',  'default':35,        'label':'IC最少截面资产'},
    'ic_weight_blend':        {'type':'float','default':0.62,      'label':'动态权重混合比'},
    'max_dynamic_factor_weight':{'type':'float','default':0.28,    'label':'单因子权重上限'},
    'ic_half_life_points':    {'type':'int',  'default':15,        'label':'IC半衰期(截面数)'},  # v3
    'use_orthogonalization':  {'type':'bool', 'default':True,      'label':'启用因子正交化'},     # v3
    'enable_early_trend_factors': {'type':'bool','default':True,   'label':'启用小时级早启动/转折因子'},
    'early_trend_min_trigger': {'type':'float','default':0.18,      'label':'早启动最低触发强度'},
    'require_m3_pullback_confirmation': {'type':'bool','default':True,'label':'要求3分钟回调企稳续势'},
    'm3_pullback_min_pct': {'type':'float','default':0.50,'label':'3分钟最小回调幅度%'},
    'm3_pullback_max_pct': {'type':'float','default':2.20,'label':'3分钟最大回调幅度%'},
    'm3_stabilization_bars': {'type':'int','default':4,'label':'3分钟企稳确认根数'},
    'max_m3_staleness_bars':   {'type':'int',  'default':15,        'label':'3分钟回调最大时效(根数)'},  # v3
    # v3.1 新增 — 趋势时效性
    'max_h1_trend_age':        {'type':'int',  'default':12,        'label':'1H趋势最大延续根数（超过则惩罚）'},
    'h1_trend_age_penalty':    {'type':'float','default':8.0,       'label':'趋势过老时score惩罚量'},
    'm3_freshness_penalty':    {'type':'float','default':6.0,       'label':'3m回调过旧时score惩罚量'},
    'bonus_freshness_score':   {'type':'float','default':3.0,       'label':'两项时效均通过时score加分'},
    # v3.2 新增 ── 3m微观结构增强
    'm3_min_impulse_pct':      {'type':'float','default':0.65,     'label':'3m最小原趋势脉冲%'},
    'vol_continuation_min_ratio':{'type':'float','default':0.78,   'label':'企稳量能续航最低比例'},
    'require_m3_freshness':    {'type':'bool', 'default':False,    'label':'必须通过3m时效性检查'},
    'enable_atr_squeeze_check':{'type':'bool', 'default':True,     'label':'启用ATR收缩检测'},
    'atr_squeeze_ratio':       {'type':'float','default':0.55,     'label':'ATR收缩比例（当前/长期）'},
    'enable_volume_delta_check':{'type':'bool','default':True,     'label':'启用买卖力量检测'},
    'volume_delta_min_ratio':  {'type':'float','default':1.15,     'label':'企稳段买入量/卖出量最低比值'},
    'enable_vwap_alignment_check':{'type':'bool','default':True,   'label':'启用VWAP对齐检测'},
    # v4.0 新增
    'enable_bidask_spread_filter':{'type':'bool','default':True,   'label':'启用买卖价差过滤'},
    'max_bidask_spread_pct':     {'type':'float','default':0.25,  'label':'最大允许买卖价差%'},
    # v4.1 趋势早期捕捉
    'h1_trend_age_hard_limit':   {'type':'int',  'default':20,     'label':'1H趋势硬过滤上限（超过直接排除，0=不限）'},
    'require_early_trend_entry': {'type':'bool', 'default':False,  'label':'只接受趋势早期信号（trend_age<=max_h1_trend_age）'},
    'early_trend_edge_discount': {'type':'float','default':0.06,   'label':'早期趋势降低min_abs_edge的折扣量'},
    'position_size':          {'type':'float','default':0.1,       'label':'回测仓位比例'},
    'take_profit_pct':        {'type':'float','default':5.5,       'label':'止盈%'},
    'stop_loss_pct':          {'type':'float','default':3.2,       'label':'止损%'},
}
_cs1__DEFAULT_CONFIG = {k: v['default'] for k, v in _cs1_CONFIG_SCHEMA.items()}


def _backtest_config(config: Dict[str, Any]) -> Dict[str, Any]:
    cfg = dict(config or {})
    cfg['min_score'] = float(cfg.get('backtest_min_score', cfg.get('min_score', 70.0)) or 70.0)
    return cfg


def _scan_symbol_with_config(symbol, config: Dict[str, Any]) -> Dict:
    snap = _cs1__build_snapshot(symbol, config)
    if not snap.valid:
        return _cs1__failed_result(symbol, snap.reason)
    score, direction, edge, fs = _single_asset_score(snap, config)
    ms = float(config.get('min_score', 72.0))
    passed = score >= ms and direction in {'BUY','SELL'}
    return _build_scan_result(snapshot=snap,score=score,direction=direction,edge=edge,
        factor_scores=fs,passed=passed,rank=None,universe_size=None,
        category="截面多因子",config=config)


def _signal_from_scan_result(result: Dict[str, Any], config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not result or not result.get('passed'):
        return None
    direction = result.get('direction')
    if direction not in {'BUY', 'SELL'}:
        return None
    return {
        'action': 'BUY' if direction == 'BUY' else 'SHORT',
        'position_size': float(config.get('position_size', 0.1)),
        'entry_price': float(result.get('last_price', 0) or 0),
        'reason': f"{result.get('category')} | {float(result.get('score',0)):.1f}",
        'score': float(result.get('opportunity_score', result.get('score',0)) or 0),
        'raw_result': result,
    }


# ══════════════════════════════════════════════
# 因子快照
# ══════════════════════════════════════════════
class FactorSnapshot:
    __slots__ = (
        'symbol','valid','reason','last_price','volume_24h','price_change_24h',
        'momentum_1h','momentum_4h','momentum_1d','short_reversal',
        'trend_quality','realized_vol','atr_pct','liquidity','volume_impulse',
        'funding_rate','oi_heat','rsi_1h','rsi_4h','direction_bias','factors',
        'macd_momentum','bb_percentb','vol_zscore','close_strength',
        'efficiency_ratio','rsi_alignment',
        # v3 新增
        'whale_flow','exchange_netflow','active_addresses','nvt_signal',
        'momentum_decay','momentum_acceleration',
        'early_trend_trigger','ema_compression_breakout','rsi_midline_turn',
        'macd_hist_turn','donchian_breakout','volume_price_confirm',
        'm3_pullback_confirmed','m3_structure_state','m3_pullback_reason','m3_pullback_pct','m3_impulse_pct',
        'm3_staleness_bars',  # v3fix: 回调时效性
        'h1_trend_age',       # v3.1: 1H趋势延续根数
        'ema_cross_signal',   # v4.1: 1H EMA金叉/死叉信号
    )
    def __init__(self, **kw):
        for s in self.__slots__:
            setattr(self, s, kw.get(s, 0.0 if s not in ('factors','valid','reason','symbol','direction_bias') else
                                     ({} if s=='factors' else (True if s=='valid' else ('' if s in('reason','symbol','direction_bias') else 0.0)))))
        if not isinstance(self.factors, dict): self.factors = {}
        self.valid = bool(self.valid)
        self.reason = str(self.reason or '')
        self.symbol = str(self.symbol or '')
        self.direction_bias = str(self.direction_bias or 'WAIT')


# ══════════════════════════════════════════════
# Scanner
# ══════════════════════════════════════════════
class ACrossSectionalMultiFactorScannerStrategy(BaseScannerStrategy if _HAS_SCANNER_BASE else object):
    required_bars = ['3m','1D','4H','1H']
    requires_derivative_metrics = True
    requires_on_chain_metrics = True
    name = "截面多因子加密货币扫描"
    description = "v3: 链上因子 + 动量衰减 + 因子正交化 + IC半衰期"
    strategy_type = "scan"

    def __init__(self, config=None):
        self.config = {**_cs1__DEFAULT_CONFIG, **(config or {})}
        self.last_factor_weights = _factor_weights()
        self.last_ic_snapshot = {}
        if _HAS_SCANNER_BASE and hasattr(super(),'__init__'):
            try: super().__init__(self.config)
            except Exception: pass

    def _init_conditions(self):
        if ScanCondition is None: return
        self.add_condition(ScanCondition(name="24H成交额",description="过滤流动性不足",
            field="volume_24h",operator=">=",value=self.config.get('min_volume_24h',8_000_000)))

    def scan_symbol(self, symbol) -> Dict:
        return _scan_symbol_with_config(symbol, self.config)

    def generate_signal(self, data, *a, **kw):
        if not isinstance(data, dict) or not data.get('klines_map'):
            return None
        cfg = _backtest_config(self.config)
        sym = _cs1__symbol_from_backtest_data(data, cfg)
        result = _scan_symbol_with_config(sym, cfg)
        self.last_analysis = result
        return _signal_from_scan_result(result, cfg)

    def reset_backtest_state(self):
        self.last_analysis = {}

    def scan_all_symbols(self, symbols: List) -> Dict:
        min_vol = float(self.config.get('min_volume_24h',8_000_000))
        snap_map: Dict[str,FactorSnapshot] = {}
        valid_syms = []
        for sym in symbols:
            if float(getattr(sym,'volume_24h',0) or 0) < min_vol: continue
            snap = _cs1__build_snapshot(sym, self.config)
            if snap.valid:
                snap_map[snap.symbol] = snap
                valid_syms.append(sym)
        if len(snap_map) < 5:
            return {'type':'cross_section_multi_factor','all_opportunities':[]}

        snaps = list(snap_map.values())
        factor_frame = pd.DataFrame([s.factors for s in snaps], index=[s.symbol for s in snaps])

        # v3: 因子正交化
        if bool(self.config.get('use_orthogonalization', True)):
            factor_frame = _orthogonalize_factors(factor_frame)

        z = factor_frame.apply(_robust_zscore, axis=0).fillna(0.0)

        weights, ic_snap = _resolve_factor_weights(valid_syms, self.config)
        weights = _adapt_weights_to_factor_frame(weights, factor_frame)
        self.last_factor_weights = weights
        self.last_ic_snapshot = ic_snap

        long_edge = sum(z[n]*w for n,w in weights.items() if n in z.columns)
        results = []
        min_edge = float(self.config.get('min_abs_edge',0.38))
        min_score = float(self.config.get('min_score',72))
        allow_short = bool(self.config.get('allow_short',True))
        usize = len(snap_map)
        hard_atr = float(self.config.get('max_atr_pct',7.5))

        ranked_long = long_edge.sort_values(ascending=False)
        for rank, sn in enumerate(ranked_long.head(int(self.config.get('top_n_long',12))).index, 1):
            snap = snap_map.get(sn)
            if not snap or snap.atr_pct > hard_atr: continue
            early_ok = snap.early_trend_trigger >= float(self.config.get('early_trend_min_trigger',0.18))
            ema_cross_ok = abs(_safe_float(getattr(snap,'ema_cross_signal',0.0),0.0)) >= 0.5
            if snap.momentum_4h <= 0 and snap.momentum_1d <= 2 and not early_ok and not ema_cross_ok: continue
            # v3: 动量衰减过滤
            if snap.momentum_decay < -3.0 and not early_ok and not ema_cross_ok: continue
            # v4.1: 硬年龄上限（超过直接排除，避免在老趋势末端建仓）
            hard_limit = int(_safe_float(self.config.get('h1_trend_age_hard_limit', 20), 20))
            if hard_limit > 0 and snap.h1_trend_age > hard_limit: continue
            # v4.1: 只要早期趋势模式
            max_age = int(_safe_float(self.config.get('max_h1_trend_age', 12), 12))
            if bool(self.config.get('require_early_trend_entry', False)) and snap.h1_trend_age > max_age: continue
            # v3.1: 两项都严重超标则直接跳过
            max_stale = int(_safe_float(self.config.get('max_m3_staleness_bars', 15), 15))
            if snap.h1_trend_age > max_age * 2 and snap.m3_staleness_bars > max_stale: continue
            edge = float(long_edge.loc[sn])
            score = _cs1__edge_to_score(edge, snap)
            # v4.1: 早期趋势（trend_age 在正常范围内或刚金叉）降低 min_edge 门槛
            is_early = snap.h1_trend_age <= max_age or ema_cross_ok
            discount = float(self.config.get('early_trend_edge_discount', 0.06)) if is_early else 0.0
            if edge < min_edge - discount or score < min_score: continue
            results.append(_build_scan_result(snapshot=snap,score=score,direction='BUY',edge=edge,
                factor_scores=z.loc[sn].to_dict(),passed=True,rank=rank,universe_size=usize,
                category="截面多因子多头",config=self.config,weights=weights,ic_snapshot=ic_snap))

        if allow_short:
            short_edge = -long_edge
            for rank, sn in enumerate(short_edge.sort_values(ascending=False).head(int(self.config.get('top_n_short',8))).index, 1):
                snap = snap_map.get(sn)
                if not snap or snap.atr_pct > hard_atr: continue
                early_ok = snap.early_trend_trigger <= -float(self.config.get('early_trend_min_trigger',0.18))
                ema_cross_ok_s = _safe_float(getattr(snap,'ema_cross_signal',0.0),0.0) <= -0.5
                if snap.momentum_4h >= 0 and snap.momentum_1d >= -3 and not early_ok and not ema_cross_ok_s: continue
                if snap.momentum_decay > 3.0 and not early_ok and not ema_cross_ok_s: continue
                # v4.1: 硬年龄上限
                hard_limit_s = int(_safe_float(self.config.get('h1_trend_age_hard_limit', 20), 20))
                if hard_limit_s > 0 and snap.h1_trend_age > hard_limit_s: continue
                max_age_s = int(_safe_float(self.config.get('max_h1_trend_age', 12), 12))
                if bool(self.config.get('require_early_trend_entry', False)) and snap.h1_trend_age > max_age_s: continue
                max_stale_s = int(_safe_float(self.config.get('max_m3_staleness_bars', 15), 15))
                if snap.h1_trend_age > max_age_s * 2 and snap.m3_staleness_bars > max_stale_s: continue
                edge = float(short_edge.loc[sn])
                score = _cs1__edge_to_score(edge, snap)
                is_early_s = snap.h1_trend_age <= max_age_s or ema_cross_ok_s
                discount_s = float(self.config.get('early_trend_edge_discount', 0.06)) if is_early_s else 0.0
                if edge < min_edge - discount_s or score < min_score: continue
                results.append(_build_scan_result(snapshot=snap,score=score,direction='SELL',edge=edge,
                    factor_scores=z.loc[sn].to_dict(),passed=True,rank=rank,universe_size=usize,
                    category="截面多因子空头",config=self.config,weights=weights,ic_snapshot=ic_snap))

        results.sort(key=lambda r:(float(r.get('opportunity_score',r.get('score',0)) or 0),
                                   float(r.get('volume_24h',0) or 0)), reverse=True)
        return {'type':'cross_section_multi_factor','all_opportunities':results}

    def get_config_schema(self) -> Dict: return dict(_cs1_CONFIG_SCHEMA)


# ══════════════════════════════════════════════
# 回测适配器
# ══════════════════════════════════════════════
class ZCrossSectionalMultiFactorBacktestStrategy:
    required_bars = ['1D','4H','1H']
    name = "截面多因子加密货币扫描(回测)"
    strategy_type = "backtest"
    def __init__(self, config=None):
        self.config = _backtest_config({**_cs1__DEFAULT_CONFIG, **(config or {})})
        self._bars = []
        self.last_analysis = {}
        self._scanner = ACrossSectionalMultiFactorScannerStrategy(self.config)
        self._synced = False  # v3.1: 只同步一次，避免反复覆盖 scanner 缓存
    def on_bar(self, bar, *a, **kw): return self._handle_bar(bar)
    def generate_signal(self, data, *a, **kw):
        if isinstance(data, dict) and data.get('klines_map'):
            self._ensure_sync()
            sym = _cs1__symbol_from_backtest_data(data, self.config)
            r = self._scanner.scan_symbol(sym)
            self.last_analysis = r
            return _signal_from_scan_result(r, self.config)
        return self._handle_bar(data)
    def next(self, bar, *a, **kw): return self._handle_bar(bar)
    def process(self, bar, *a, **kw): return self._handle_bar(bar)
    def step(self, bar, *a, **kw): return self._handle_bar(bar)
    def _handle_bar(self, bar):
        self._ensure_sync()
        parsed = _parse_bar(bar)
        if not parsed: return 'WAIT'
        self._bars.append(parsed)
        if len(self._bars) < 200: return 'WAIT'
        h1 = pd.DataFrame(self._bars)
        h4 = _aggregate_bars(h1, 4); d1 = _aggregate_bars(h1, 24)
        kl = {'1H':_df_to_rows(h1.tail(260)),'4H':_df_to_rows(h4.tail(220)),'1D':_df_to_rows(d1.tail(180))}
        sym = _cs1__MinimalSymbol(inst_id=str(self.config.get('inst_id','BT') or 'BT'),
            last_price=float(parsed['c']),volume_24h=float(h1['vol'].tail(24).sum()),
            price_change_24h=_pct_change(h1['c'],24),extra_data={'klines':kl})
        r = self._scanner.scan_symbol(sym)
        self.last_analysis = r
        if not r.get('passed'): return 'WAIT'
        return 'BUY' if r.get('direction')=='BUY' else 'SELL'
    def reset(self): self._bars.clear(); self.last_analysis = {}; self._synced = False
    def reset_backtest_state(self): self.reset()
    def _ensure_sync(self):
        # v3.1 修复: 只在首次调用时同步一次，避免每次 on_bar 重建 config
        # 导致 scanner.last_factor_weights 等动态权重缓存被清空
        if not self._synced:
            self.config = _backtest_config({**_cs1__DEFAULT_CONFIG, **(self.config or {})})
            self._scanner.config = self.config
            self._synced = True
    def get_config_schema(self): return dict(_cs1_CONFIG_SCHEMA)


class _cs1__MinimalSymbol:
    __slots__ = ('inst_id','last_price','volume_24h','price_change_24h',
                 'high_24h','low_24h','open_interest','extra_data')
    def __init__(self, **kw):
        for s in self.__slots__:
            setattr(self, s, kw.get(s, 0.0 if s != 'extra_data' else {}))
        if not isinstance(self.inst_id, str): self.inst_id = str(self.inst_id or '')
        if not isinstance(self.extra_data, dict): self.extra_data = {}


# ══════════════════════════════════════════════
# A) 链上数据因子提取
# ══════════════════════════════════════════════

def _extract_on_chain_factors(symbol) -> Dict[str, float]:
    """
    从 symbol.extra_data.get('on_chain', {}) 提取链上因子。
    数据源约定（由调用方注入）：
      whale_flow: 大户净流入（归一化为市值百分比），正=鲸鱼买入
      exchange_netflow: 交易所净流入，负=提币看涨
      active_addresses: 24h 活跃地址数
      nvt_ratio: NVT 比率 (market_cap / on_chain_tx_volume)
    没有数据时返回 nan（z-score 会变 0，不影响其他因子）。
    """
    oc = getattr(symbol, 'extra_data', {}).get('on_chain', {}) or {}
    whale = _safe_float(oc.get('whale_flow'), np.nan)
    exch_nf = _safe_float(oc.get('exchange_netflow'), np.nan)
    active = _safe_float(oc.get('active_addresses'), np.nan)
    nvt = _safe_float(oc.get('nvt_ratio'), np.nan)
    return {
        'whale_flow': whale,
        'exchange_netflow': -exch_nf if not np.isnan(exch_nf) else np.nan,  # 负流入=看涨→取反使高值=看涨
        'active_addresses': log(max(active, 1.0)) if not np.isnan(active) and active > 0 else np.nan,
        'nvt_signal': -nvt if not np.isnan(nvt) else np.nan,  # NVT 低=低估→取反使高值=低估
    }


# ══════════════════════════════════════════════
# B) 多周期动量衰减检测
# ══════════════════════════════════════════════

def _momentum_decay(m1h: float, m4h: float, m1d: float) -> float:
    """
    动量衰减 = 短周期动量(归一化) - 长周期动量(归一化)。
    正值 = 短周期加速（健康）；负值 = 短周期衰减（趋势转弱）。
    三个周期的量纲不同，按典型范围归一化：
      1H 6根 ≈ ±3%，4H 12根 ≈ ±8%，1D 7根 ≈ ±15%
    """
    norm_1h = m1h / 3.0 if abs(m1h) < 50 else np.sign(m1h)
    norm_4h = m4h / 8.0 if abs(m4h) < 50 else np.sign(m4h)
    norm_1d = m1d / 15.0 if abs(m1d) < 50 else np.sign(m1d)
    return (norm_1h * 0.5 + norm_4h * 0.3) - norm_1d * 0.8


def _momentum_acceleration(h1_close: pd.Series, period: int = 6) -> float:
    """
    动量二阶导：当前 N 根涨幅 vs 前 N 根涨幅的差。
    正=加速上涨（或减速下跌），负=减速上涨（或加速下跌）。
    """
    if len(h1_close) < period * 2 + 1:
        return 0.0
    current_mom = _pct_change(h1_close, period)
    # 从倒数第 period+1 根往前再看 period 根
    prev_slice = h1_close.iloc[:-(period)]
    if len(prev_slice) < period + 1:
        return 0.0
    prev_mom = _pct_change(prev_slice, period)
    return current_mom - prev_mom


def _timeliness_score_adjustment(snap, config, direction: str) -> float:
    """
    根据趋势时效性和 3m 回调时效性计算 score 的净调整量。

    规则：
    - 1H 趋势延续根数超过 max_h1_trend_age → 惩罚（指数递增，越老越重）
    - 3m 回调距今超过 max_m3_staleness_bars → 惩罚
    - 两项都在范围内 → 加分（鼓励"新鲜启动点"）

    返回值直接加到 score 上（负=惩罚，正=加分）。
    """
    if direction not in {'BUY', 'SELL'}:
        return 0.0

    age = getattr(snap, 'h1_trend_age', 0)
    max_age = int(_safe_float(config.get('max_h1_trend_age', 12), 12))
    age_pen = _safe_float(config.get('h1_trend_age_penalty', 8.0), 8.0)

    stale = getattr(snap, 'm3_staleness_bars', 0)
    max_stale = int(_safe_float(config.get('max_m3_staleness_bars', 15), 15))
    fresh_pen = _safe_float(config.get('m3_freshness_penalty', 6.0), 6.0)

    bonus = _safe_float(config.get('bonus_freshness_score', 3.0), 3.0)

    adj = 0.0
    age_ok = age <= max_age
    stale_ok = stale <= max_stale

    if not age_ok:
        # 指数递增惩罚：刚越界时较轻，越老越重，但有上限
        overflow = age - max_age
        adj -= age_pen * (1.0 - exp(-overflow / max(max_age * 0.5, 1.0)))

    if not stale_ok:
        overflow = stale - max_stale
        adj -= fresh_pen * (1.0 - exp(-overflow / max(max_stale * 0.5, 1.0)))

    if age_ok and stale_ok:
        adj += bonus

    return _clamp(adj, -15.0, 8.0)


def _cs1__empty_early_trend_features() -> Dict[str, float]:
    return {
        'trigger': 0.0,
        'early_trend_trigger': 0.0,
        'ema_compression_breakout': 0.0,
        'rsi_midline_turn': 0.0,
        'macd_hist_turn': 0.0,
        'donchian_breakout': 0.0,
        'volume_price_confirm': 0.0,
    }


def _cs1__early_trend_features(h1: pd.DataFrame) -> Dict[str, float]:
    """
    小时级趋势早启动/转折因子。
    设计重点是捕捉“均线刚从收敛区扩散 + 价格刚突破短箱体 + RSI/MACD 拐头 + 量价确认”，
    尽量避免只在趋势已经拉开很远后才给分。
    """
    if h1 is None or len(h1) < 58:
        return _cs1__empty_early_trend_features()
    close = h1['c'].astype(float)
    high = h1['h'].astype(float)
    low = h1['l'].astype(float)
    vol = h1['vol'].astype(float)
    price = float(close.iloc[-1])
    if price <= 0:
        return _cs1__empty_early_trend_features()

    ema8 = close.ewm(span=8, adjust=False).mean()
    ema21 = close.ewm(span=21, adjust=False).mean()
    ema55 = close.ewm(span=55, adjust=False).mean()
    prev_spread = float((ema21.iloc[-7] - ema55.iloc[-7]) / max(abs(ema55.iloc[-7]), 1e-9) * 100.0)
    cur_fast_spread = float((ema8.iloc[-1] - ema21.iloc[-1]) / price * 100.0)
    cur_slow_spread = float((ema21.iloc[-1] - ema55.iloc[-1]) / price * 100.0)
    spread_delta = cur_slow_spread - prev_spread
    compression = 1.0 - _clamp(abs(prev_spread) / 2.2, 0.0, 1.0)
    ema_component = _clamp((cur_fast_spread * 0.9 + spread_delta * 0.7) * (0.65 + compression * 0.55), -2.0, 2.0)

    prev_high = float(high.iloc[-21:-1].max())
    prev_low = float(low.iloc[-21:-1].min())
    up_break = (price / prev_high - 1.0) * 100.0 if prev_high > 0 else 0.0
    down_break = (prev_low / price - 1.0) * 100.0 if prev_low > 0 else 0.0
    donchian = _clamp(up_break / 0.9, 0.0, 2.0) - _clamp(down_break / 0.9, 0.0, 2.0)

    rsi_series = _cs1__rsi_series_wilder(close, 14)
    rsi_now = float(rsi_series.iloc[-1]) if not rsi_series.empty else 50.0
    rsi_prev = float(rsi_series.iloc[-4]) if len(rsi_series) >= 4 else 50.0
    rsi_mid = _clamp((rsi_now - 50.0) / 16.0 + (rsi_now - rsi_prev) / 12.0, -2.0, 2.0)

    macd_hist = _cs1__macd_hist_series(close)
    if len(macd_hist) >= 5:
        hist_now = float(macd_hist.iloc[-1]) / price * 100.0
        hist_prev = float(macd_hist.iloc[-4]) / price * 100.0
        hist_slope = (hist_now - hist_prev) * 3.0
        macd_turn = _clamp(hist_now * 8.0 + hist_slope, -2.0, 2.0)
    else:
        macd_turn = 0.0

    tr = pd.concat([
        (high - low).abs(),
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    base_range = float(tr.iloc[-25:-1].median() or 0.0)
    range_ratio = float(tr.tail(3).mean() / base_range) if base_range > 0 else 1.0
    base_vol = float(vol.iloc[-25:-1].median() or 0.0)
    volume_ratio = float(vol.tail(3).mean() / base_vol) if base_vol > 0 else 1.0
    close_strength = _avg_close_strength(h1, 3)
    price_dir = 1.0 if close.iloc[-1] >= close.iloc[-4] else -1.0
    volume_price = price_dir * _clamp((range_ratio - 1.0) * 0.7 + (volume_ratio - 1.0) * 0.6 + (close_strength - 0.5) * 1.2, -2.0, 2.0)

    # v4.1: 检测 1H EMA12/34 是否刚发生金叉/死叉（趋势初始信号）
    ema_cross = _cs1__ema_just_crossed(close, fast=12, slow=34, lookback=6)
    # 金叉/死叉信号赋予最高权重 0.20，其余因子按比例压缩
    trigger = _clamp(
        ema_component * 0.24 + donchian * 0.20 + rsi_mid * 0.15 + macd_turn * 0.13
        + volume_price * 0.10 + ema_cross * 0.20,
        -2.5,
        2.5,
    )
    return {
        'trigger': trigger,
        'early_trend_trigger': trigger,
        'ema_compression_breakout': ema_component,
        'rsi_midline_turn': rsi_mid,
        'macd_hist_turn': macd_turn,
        'donchian_breakout': donchian,
        'volume_price_confirm': volume_price,
        'ema_cross_signal': ema_cross,  # v4.1: 金叉/死叉信号
    }


def _cs1__ema_just_crossed(close: pd.Series, fast: int = 12, slow: int = 34, lookback: int = 6) -> float:
    """
    检测 EMA 是否在最近 lookback 根内刚发生金叉/死叉。
    返回: +1.0=刚发生金叉(看多), -1.0=刚发生死叉(看空), 0.0=无最近交叉。
    交叉越近（越新），绝对值越大（最大1.0，越旧越衰减到0.1）。
    """
    if len(close) < slow + lookback + 2:
        return 0.0
    ema_f = close.ewm(span=fast, adjust=False).mean()
    ema_s = close.ewm(span=slow, adjust=False).mean()
    diff = ema_f - ema_s
    for i in range(1, min(lookback + 1, len(diff))):
        cur = float(diff.iloc[-i])
        prev = float(diff.iloc[-i - 1])
        if prev <= 0 < cur:   # 金叉
            return 1.0 * (1.0 - (i - 1) * 0.12)   # 越近权重越高
        if prev >= 0 > cur:   # 死叉
            return -1.0 * (1.0 - (i - 1) * 0.12)
    return 0.0


def _cs1__rsi_series_wilder(close: pd.Series, period: int = 14) -> pd.Series:
    if len(close) < period + 2:
        return pd.Series([50.0] * len(close), index=close.index)
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100.0 - 100.0 / (1.0 + rs)).replace([np.inf, -np.inf], np.nan).fillna(50.0).clip(0.0, 100.0)


def _cs1__macd_hist_series(close: pd.Series) -> pd.Series:
    if len(close) < 35:
        return pd.Series(dtype=float)
    fast = close.ewm(span=12, adjust=False).mean()
    slow = close.ewm(span=26, adjust=False).mean()
    dif = fast - slow
    dea = dif.ewm(span=9, adjust=False).mean()
    return dif - dea


# ══════════════════════════════════════════════
# C) 因子正交化 (Gram-Schmidt)
# ══════════════════════════════════════════════

def _orthogonalize_factors(factor_frame: pd.DataFrame) -> pd.DataFrame:
    """
    v4.0: 对称正交化 — 用协方差矩阵 Cholesky 白化替代 Gram-Schmidt。
    之前 Gram-Schmidt 不对称（锚因子保留全部方差，从属因子只保留残差），
    导致锚因子在截面上占主导。对称正交化让同组所有因子平等分担信息。
    """
    df = factor_frame.copy()
    ortho_groups = [
        ['momentum_4h', 'momentum_1h', 'momentum_1d'],
        ['trend_quality', 'efficiency_ratio'],
        ['volume_impulse', 'vol_zscore'],
    ]
    for group in ortho_groups:
        cols = [c for c in group if c in df.columns]
        if len(cols) < 2:
            continue
        sub = df[cols].values.astype(float)
        # 用均值填充 NaN（白化需要完整矩阵）
        col_means = np.nanmean(sub, axis=0)
        sub_filled = np.where(np.isnan(sub), col_means, sub)
        # 协方差矩阵
        cov = np.cov(sub_filled, rowvar=False)
        # 正则化：防止奇异矩阵
        cov += np.eye(len(cols)) * 1e-8 * np.trace(cov)
        try:
            # Cholesky 分解：cov = L @ L^T
            L = np.linalg.cholesky(cov)
            # 白化矩阵：W = L^{-1}，使得 W @ cov @ W^T = I
            W = np.linalg.inv(L)
            whitened = sub_filled @ W.T
        except np.linalg.LinAlgError:
            # Cholesky 失败（数值不稳定），退化为 SVD 白化
            U, S, Vt = np.linalg.svd(sub_filled - col_means, full_matrices=False)
            S_inv = np.diag(1.0 / np.maximum(S, 1e-8))
            whitened = U @ S_inv @ Vt
        # 还原原始量纲（保持方差量级大致不变）
        orig_std = np.nanstd(sub_filled, axis=0)
        whitened_std = np.nanstd(whitened, axis=0)
        scale = np.where(whitened_std > 1e-12, orig_std / whitened_std, 1.0)
        whitened = whitened * scale
        for i, col in enumerate(cols):
            df[col] = whitened[:, i]
    return df

def _adapt_weights_to_factor_frame(weights: Dict[str, float], factor_frame: pd.DataFrame) -> Dict[str, float]:
    """
    v4.0: 因子拥挤度检测 — 截面标准差过小的因子说明被过度拥挤，
    区分度下降（每个人都用同样的因子），应降低权重。
    """
    usable = {}
    for name, weight in weights.items():
        if name not in factor_frame.columns:
            continue
        values = pd.to_numeric(factor_frame[name], errors='coerce').replace([np.inf, -np.inf], np.nan).dropna()
        if len(values) < 5:
            continue
        std_val = float(values.std(ddof=0) or 0.0)
        if std_val <= 1e-12:
            continue
        # v4.0: 拥挤度检测 — 截面标准差相对均值过小 = 因子拥挤
        mean_abs = float(values.abs().mean() or 1e-9)
        cv = std_val / mean_abs  # 变异系数
        crowd_penalty = min(1.0, max(0.3, cv * 3.0))  # CV<0.17时权重降到0.3x
        usable[name] = weight * crowd_penalty
    if not usable:
        return _cs1__normalize_weights(weights)
    return _cs1__normalize_weights(usable)


# ══════════════════════════════════════════════
# D) IC 半衰期衰减
# ══════════════════════════════════════════════

def _half_life_weights(n_points: int, half_life: int) -> np.ndarray:
    """
    生成指数衰减权重，距今最远的点权重最低。
    w[i] = 2^(-i/half_life)，i=0 是最新，i=n-1 是最旧。
    注意：_rolling_ic_snapshot 中 IC 序列是按时间顺序 append 的
    （最旧在前），所以需要反转。
    """
    if n_points <= 0 or half_life <= 0:
        return np.ones(max(n_points, 1))
    indices = np.arange(n_points)
    # 最新 = index n-1 → decay=0，最旧 = index 0 → decay 最大
    decay = np.exp(-np.log(2) * (n_points - 1 - indices) / half_life)
    return decay / decay.sum() * n_points  # 归一化使均值=1（不改变量纲）


# ══════════════════════════════════════════════
# 因子快照构建
# ══════════════════════════════════════════════

def _cs1__build_snapshot(symbol, config) -> FactorSnapshot:
    klines = getattr(symbol, 'extra_data', {}).get('klines', {}) or {}
    m3 = _to_df(_cs1__get_klines(klines, '3m'))
    if m3.empty:
        m3 = _to_df(_cs1__get_klines(klines, '3M'))
    if m3.empty:
        m1 = _to_df(_cs1__get_klines(klines, '1m'))
        if len(m1) >= 120:
            m3 = _aggregate_bars(m1, 3)
    d1 = _to_df(_cs1__get_klines(klines, '1D'))
    h4 = _to_df(_cs1__get_klines(klines, '4H'))
    h1 = _to_df(_cs1__get_klines(klines, '1H'))
    inst = str(getattr(symbol, 'inst_id', ''))
    if len(h1) < 80 or len(h4) < 60:
        return FactorSnapshot(symbol=inst,valid=False,reason=f'K线不足(1H={len(h1)},4H={len(h4)})',direction_bias='WAIT')

    lp = float(getattr(symbol,'last_price',0) or h1['c'].iloc[-1])
    v24 = float(getattr(symbol,'volume_24h',0) or h1['vol'].tail(24).sum())
    pc24 = float(getattr(symbol,'price_change_24h',0) or _pct_change(h1['c'],24))

    m1h = _pct_change(h1['c'],6); m4h = _pct_change(h4['c'],12)
    m1d = _pct_change(d1['c'],7) if len(d1)>=12 else _pct_change(h1['c'],168)
    sr = -_pct_change(h1['c'],3)

    # v4.0: BTC beta 中性化 — 去除BTC带动的系统性涨跌
    btc_beta = 0.0
    if bool(config.get('enable_btc_correlation_filter', False)):
        btc_data = (getattr(symbol, 'extra_data', {}) or {}).get('btc_klines')
        if btc_data:
            btc_h1 = _to_df(_cs1__get_klines({'1H': btc_data} if isinstance(btc_data, list) else btc_data, '1H'))
            if len(btc_h1) >= 24:
                asset_ret = h1['c'].pct_change().dropna().tail(24)
                btc_ret = btc_h1['c'].pct_change().dropna().tail(24)
                if len(asset_ret) >= 12 and len(btc_ret) >= 12:
                    n = min(len(asset_ret), len(btc_ret))
                    cov = np.cov(asset_ret.tail(n), btc_ret.tail(n))[0, 1]
                    btc_var = np.var(btc_ret.tail(n))
                    if btc_var > 1e-12:
                        btc_beta = cov / btc_var
                        btc_mom = _pct_change(btc_h1['c'], 6) if len(btc_h1) >= 7 else 0.0
                        # 去除 BTC beta 贡献
                        m1h -= btc_beta * btc_mom * 0.9
                        m4h -= btc_beta * btc_mom * 0.8
                        m1d -= btc_beta * btc_mom * 0.7

    ema21 = _ema(h4['c'],21); ema55 = _ema(h4['c'],55)
    spread = (ema21-ema55)/ema55*100 if ema55>0 else 0
    slope = _ema_slope_pct(h4['c'],21,6); adx = _adx(h4)
    tq = _cs1__trend_quality(lp,ema21,ema55,spread,slope,adx)
    rv = _cs1__realized_vol_pct(h1['c'],48); atr = _cs1__atr_pct(h4)
    liq = log(max(v24,1.0)); vi = _volume_ratio(h1['vol'],24)
    vz = _volume_zscore(h1['vol'])
    fr = _safe_float(getattr(symbol,'extra_data',{}).get('funding_rate'),0)*100
    oi_v = _safe_float(getattr(symbol,'open_interest',0),0)
    oi_h = log(max(oi_v,1.0)) if oi_v>0 else 0.0
    r1h = _rsi_wilder(h1['c']); r4h = _rsi_wilder(h4['c'])
    mm = _macd_momentum(h1['c']); bb = _bb_percentb(h1['c'])
    cs = _avg_close_strength(h1,6); er = _efficiency_ratio(h4['c'],20)
    ra = _rsi_alignment(r1h, r4h)

    # v3: 动量衰减 + 加速
    mdecay = _momentum_decay(m1h, m4h, m1d)
    maccel = _momentum_acceleration(h1['c'], 6)
    early = _cs1__early_trend_features(h1) if bool(config.get('enable_early_trend_factors', True)) else _cs1__empty_early_trend_features()

    # v3: 链上因子
    oc = _extract_on_chain_factors(symbol)

    db = ('BUY' if ema21>=ema55 and m4h>=0 else 'SELL' if ema21<ema55 and m4h<0 else 'WAIT')
    trend_hint = 1.0 if db == 'BUY' else -1.0 if db == 'SELL' else 0.0

    # v3.1: 1H 趋势时效性（在 db 确定后计算）
    h1_trend_age = _measure_trend_age(h1['c'], fast=12, slow=34, direction=trend_hint)

    # v4.0: 买卖价差过滤 — 价差过大说明流动性差，3m信号质量低
    if bool(config.get('enable_bidask_spread_filter', True)):
        ob = (getattr(symbol, 'extra_data', {}) or {}).get('order_book')
        if ob and isinstance(ob, dict):
            bid = _safe_float(ob.get('bid_px') or ob.get('bids', [[0]])[0][0], 0)
            ask = _safe_float(ob.get('ask_px') or ob.get('asks', [[0]])[0][0], 0)
            if bid > 0 and ask > 0:
                spread_pct = (ask / bid - 1.0) * 100.0
                if spread_pct > float(config.get('max_bidask_spread_pct', 0.25) or 0.25):
                    return FactorSnapshot(symbol=inst, valid=False,
                        reason=f"买卖价差过大({spread_pct:.2f}%)，流动性不足", direction_bias='WAIT')

    micro_confirm = _micro_pullback_continuation(m3, trend_hint, config)
    if bool(config.get('require_m3_pullback_confirmation', True)) and not micro_confirm['confirmed']:
        return FactorSnapshot(symbol=inst, valid=False, reason=f"3分钟回调续势未确认: {micro_confirm['reason']}", direction_bias='WAIT')
    fc = -abs(fr); fcn = -fr

    factors = {
        'momentum_1h':m1h,'momentum_4h':m4h,'momentum_1d':m1d,'short_reversal':sr,
        'trend_quality':tq,'low_volatility':-rv,'liquidity':liq,'volume_impulse':vi,
        'funding_carry':fc,'funding_contrarian':fcn,'oi_heat':oi_h,
        'macd_momentum':mm,'bb_percentb':bb,'vol_zscore':vz,'close_strength':cs,
        'efficiency_ratio':er,'rsi_alignment':ra,
        # v3
        'momentum_decay':mdecay,'momentum_acceleration':maccel,
        'early_trend_trigger':early['trigger'],
        'ema_compression_breakout':early['ema_compression_breakout'],
        'rsi_midline_turn':early['rsi_midline_turn'],
        'macd_hist_turn':early['macd_hist_turn'],
        'donchian_breakout':early['donchian_breakout'],
        'volume_price_confirm':early['volume_price_confirm'],
        'ema_cross_signal':early.get('ema_cross_signal', 0.0),  # v4.1
        **oc,  # whale_flow, exchange_netflow, active_addresses, nvt_signal
    }

    return FactorSnapshot(
        symbol=inst,valid=True,reason='',last_price=lp,volume_24h=v24,
        price_change_24h=pc24,momentum_1h=m1h,momentum_4h=m4h,momentum_1d=m1d,
        short_reversal=sr,trend_quality=tq,realized_vol=rv,atr_pct=atr,
        liquidity=liq,volume_impulse=vi,funding_rate=fr,oi_heat=oi_h,
        rsi_1h=r1h,rsi_4h=r4h,direction_bias=db,factors=factors,
        macd_momentum=mm,bb_percentb=bb,vol_zscore=vz,close_strength=cs,
        efficiency_ratio=er,rsi_alignment=ra,
        whale_flow=_safe_float(oc.get('whale_flow'),0),
        exchange_netflow=_safe_float(oc.get('exchange_netflow'),0),
        active_addresses=_safe_float(oc.get('active_addresses'),0),
        nvt_signal=_safe_float(oc.get('nvt_signal'),0),
        momentum_decay=mdecay,momentum_acceleration=maccel,
        early_trend_trigger=early['trigger'],
        ema_compression_breakout=early['ema_compression_breakout'],
        rsi_midline_turn=early['rsi_midline_turn'],
        macd_hist_turn=early['macd_hist_turn'],
        donchian_breakout=early['donchian_breakout'],
        volume_price_confirm=early['volume_price_confirm'],
        m3_pullback_confirmed=micro_confirm['confirmed'],
        m3_structure_state=micro_confirm['state'],
        m3_pullback_reason=micro_confirm['reason'],
        m3_pullback_pct=micro_confirm['pullback_pct'],
        m3_impulse_pct=micro_confirm['impulse_pct'],
        m3_staleness_bars=micro_confirm.get('staleness_bars', 0),
        h1_trend_age=h1_trend_age,  # v3.1
        ema_cross_signal=early.get('ema_cross_signal', 0.0),  # v4.1
    )


# ══════════════════════════════════════════════
# 单标的评分
# ══════════════════════════════════════════════
def _single_asset_score(snap, config):
    max_atr = float(config.get('max_atr_pct',7.5))
    trend_p   = _clamp(snap.trend_quality,0,100)*0.24
    mom_raw   = snap.momentum_4h*2+snap.momentum_1d*0.7+snap.momentum_1h
    mom_p     = _clamp(mom_raw*3+40,0,100)*0.18
    lowvol_p  = _clamp(100-snap.realized_vol*8,0,100)*0.08
    vol_p     = _clamp(snap.volume_impulse/1.4*100,0,100)*0.08
    fund_p    = _clamp(100-abs(snap.funding_rate)*450,0,100)*0.05
    liq_p     = _clamp((snap.liquidity-14)/4*100,0,100)*0.07
    macd_p    = _clamp(50+snap.macd_momentum*20,0,100)*0.06
    bb_p      = _clamp(snap.bb_percentb*100,0,100)*0.03
    eff_p     = _clamp(snap.efficiency_ratio*100,0,100)*0.04
    rsia_p    = _clamp(snap.rsi_alignment,0,100)*0.02
    sr_p      = _clamp(50+snap.short_reversal*15,0,100)*0.03
    oi_p      = _clamp(50+snap.oi_heat*5,0,100)*0.02
    decay_p   = _clamp(50+snap.momentum_decay*6,0,100)*0.06
    accel_p   = _clamp(50+snap.momentum_acceleration*5,0,100)*0.04
    # v4.1: early_p 权重从 0.07 → 0.12，turn_p 保持
    early_p   = _clamp(50+snap.early_trend_trigger*22,0,100)*0.12
    turn_p    = _clamp(50+snap.rsi_midline_turn*18+snap.macd_hist_turn*16,0,100)*0.04
    whale_p   = _clamp(50+_safe_float(snap.whale_flow,0)*200,0,100)*0.03
    nvt_p     = _clamp(50+_safe_float(snap.nvt_signal,0)*10,0,100)*0.02
    # v4.1: EMA 金叉/死叉加分（以方向计）
    ema_cross = _safe_float(getattr(snap, 'ema_cross_signal', 0.0), 0.0)
    cross_p   = _clamp(50+ema_cross*40,0,100)*0.08

    score = (trend_p+mom_p+lowvol_p+vol_p+fund_p+liq_p+macd_p+bb_p+eff_p+rsia_p+
             sr_p+oi_p+decay_p+accel_p+early_p+turn_p+whale_p+nvt_p+cross_p)

    if snap.atr_pct>max_atr: score -= min(28,(snap.atr_pct-max_atr)*3.2)
    if snap.rsi_1h>78: score -= 5
    elif snap.rsi_1h<22: score -= 5
    if snap.momentum_decay < -2.0: score -= 4

    # v4.1: 方向判断——金叉/死叉信号可直接确定方向（趋势刚开始）
    if snap.atr_pct > max_atr:
        direction = 'WAIT'; edge = 0
    elif ema_cross >= 0.7:   # 最近1~2根内金叉，优先确定多头方向
        direction = 'BUY'; edge = score / 100
    elif ema_cross <= -0.7 and bool(config.get('allow_short', True)):
        direction = 'SELL'; edge = score / 100
    elif (snap.direction_bias=='BUY' and mom_raw>0) or snap.early_trend_trigger >= float(config.get('early_trend_min_trigger',0.18)):
        direction='BUY'; edge=score/100
    elif ((snap.direction_bias=='SELL' and mom_raw<0) or snap.early_trend_trigger <= -float(config.get('early_trend_min_trigger',0.18))) and bool(config.get('allow_short',True)):
        direction='SELL'; edge=score/100
    else:
        direction='WAIT'; edge=0

    # v3.1: 时效性调整
    time_adj = _timeliness_score_adjustment(snap, config, direction)
    score = _clamp(score + time_adj, 0.0, 100.0)

    fs = {'trend':trend_p,'momentum':mom_p,'low_vol':lowvol_p,'volume':vol_p,
          'funding':fund_p,'liquidity':liq_p,'macd':macd_p,'bb':bb_p,
          'efficiency':eff_p,'rsi_align':rsia_p,'short_reversal':sr_p,'oi_heat':oi_p,
          'decay':decay_p,'accel':accel_p,
          'early_trend':early_p,'turn':turn_p,'whale':whale_p,'nvt':nvt_p,
          'ema_cross':cross_p}
    return round(_clamp(score,0,100),2), direction, edge, fs


# ══════════════════════════════════════════════
# 因子权重 + Rolling IC + 半衰期
# ══════════════════════════════════════════════

def _factor_weights():
    return {
        # v4.1: 动量因子整体降权（避免趋势运行越久分越高）
        'momentum_1h':0.05,'momentum_4h':0.10,'momentum_1d':0.08,
        'short_reversal':0.02,'trend_quality':0.12,'low_volatility':0.06,
        'liquidity':0.06,'volume_impulse':0.06,'funding_carry':0.02,
        'funding_contrarian':0.02,'oi_heat':0.02,
        'macd_momentum':0.04,'bb_percentb':0.02,'vol_zscore':0.02,
        'close_strength':0.02,'efficiency_ratio':0.02,'rsi_alignment':0.01,
        # v3
        'momentum_decay':0.05,'momentum_acceleration':0.03,
        # v4.1: 早期趋势因子大幅提权（捕捉趋势初始启动）
        'early_trend_trigger':0.12,'ema_compression_breakout':0.05,
        'rsi_midline_turn':0.04,'macd_hist_turn':0.04,
        'donchian_breakout':0.04,'volume_price_confirm':0.03,
        'ema_cross_signal':0.06,  # v4.1: 1H EMA金叉/死叉
        'whale_flow':0.03,'exchange_netflow':0.02,'active_addresses':0.02,'nvt_signal':0.02,
    }

def _resolve_factor_weights(symbols, config):
    base = _cs1__normalize_weights(_factor_weights())
    if not bool(config.get('use_dynamic_ic_weights',True)): return base, {}
    ic_snap = _rolling_ic_snapshot(symbols, config)
    if not ic_snap: return base, {}
    hist = set(ic_snap.keys())
    dyn = {n: base.get(n,0)*max(float(ic_snap.get(n,0) or 0),0) for n in hist}
    if sum(dyn.values())<=1e-12: return base, ic_snap
    ht = sum(base.get(n,0) for n in hist)
    hd = _cs1__normalize_weights(dyn)
    blend = _clamp(float(config.get('ic_weight_blend',0.62)),0,1)
    mixed = dict(base)
    for n in hist: mixed[n] = base.get(n,0)*(1-blend) + hd.get(n,0)*ht*blend
    capped = _cap_weights(_cs1__normalize_weights(mixed), float(config.get('max_dynamic_factor_weight',0.28)))
    return capped, ic_snap

def _rolling_ic_snapshot(symbols, config):
    lookback = max(8,int(config.get('ic_lookback_points',42)))
    fwd = max(1,int(config.get('ic_forward_bars',6)))
    min_assets = max(8,int(config.get('ic_min_assets',35)))
    ewm_span = max(2,int(config.get('ic_ewm_span',12)))
    half_life = max(1,int(config.get('ic_half_life_points',15)))
    use_ortho = bool(config.get('use_orthogonalization', True))  # v3fix: IC也正交化
    fnames = list(_factor_weights().keys())
    ic_series = {n:[] for n in fnames}

    prepared = []
    for sym in symbols:
        kl = getattr(sym,'extra_data',{}).get('klines',{}) or {}
        h1 = _to_df(_cs1__get_klines(kl,'1H'))
        if len(h1) >= lookback+fwd+80: prepared.append((sym,h1))
    if len(prepared) < min_assets: return {}
    min_len = min(len(h1_) for _,h1_ in prepared)
    max_pts = min(lookback, min_len-fwd-80)
    if max_pts < 8: return {}

    for step in range(max_pts, 0, -1):
        rows = []
        sym_indices = []
        for sym, h1_ in prepared:
            pos = len(h1_)-fwd-step
            if pos<80 or pos+fwd>=len(h1_): continue
            factors = _historical_h1_factors(sym, h1_, pos)
            entry = float(h1_['c'].iloc[pos])
            exit_p = float(h1_['c'].iloc[pos+fwd])
            if entry<=0: continue
            factors['forward_return'] = exit_p/entry-1
            rows.append(factors)
            sym_indices.append(str(getattr(sym,'inst_id','')))
        if len(rows)<min_assets: continue
        frame = pd.DataFrame(rows, index=sym_indices)
        # v3fix: 对历史截面也做正交化，使IC权重与实盘因子空间一致
        if use_ortho:
            frame = _orthogonalize_factors(frame)
        fwd_ret = frame['forward_return']
        for n in fnames:
            if n not in frame.columns: continue
            vals = pd.to_numeric(frame[n], errors='coerce')
            valid = pd.concat([vals, fwd_ret], axis=1).dropna()
            if len(valid) < min_assets:
                continue
            if valid.iloc[:, 0].std(ddof=0) <= 1e-12 or valid.iloc[:, 1].std(ddof=0) <= 1e-12:
                continue
            corr = valid.iloc[:, 0].rank().corr(valid.iloc[:, 1].rank())
            if pd.notna(corr): ic_series[n].append(float(corr))

    latest = {}
    for n, vals in ic_series.items():
        if len(vals) < 4: continue
        s = pd.Series(vals)
        hl_weights = _half_life_weights(len(s), half_life)
        weighted = s * hl_weights
        latest[n] = float(weighted.ewm(span=ewm_span, adjust=False).mean().iloc[-1])
    return latest


def _historical_h1_factors(symbol, h1, pos):
    hist = h1.iloc[:pos+1]; close = hist['c']
    ema21=_ema(close,21);ema55=_ema(close,55);price=float(close.iloc[-1])
    spread = (ema21-ema55)/ema55*100 if ema55>0 else 0
    slope = _ema_slope_pct(close,21,6); adx_v = _adx(hist.tail(120))
    m1h=_pct_change(close,6)
    m4h=_pct_change(close,48)    # v3fix: 12×4H = 48×1H，与实盘一致
    m1d=_pct_change(close,168)
    # v3fix: 用1H-6/1H-48近似1H/4H RSI，替代硬编码50.0
    rsi1h = _rsi_wilder(close, 14)
    rsi4h_approx = _rsi_wilder(close, 56) if len(close) >= 58 else 50.0
    ra = _rsi_alignment(rsi1h, rsi4h_approx)
    return {
        'momentum_1h':m1h,'momentum_4h':m4h,'momentum_1d':m1d,
        'short_reversal':-_pct_change(close,3),
        'trend_quality':_cs1__trend_quality(price,ema21,ema55,spread,slope,adx_v),
        'low_volatility':-_cs1__realized_vol_pct(close,48),
        'liquidity':log(max(float(getattr(symbol,'volume_24h',0) or hist['vol'].tail(24).sum()),1.0)),
        'volume_impulse':_volume_ratio(hist['vol'],24),
        'macd_momentum':_macd_momentum(close),'bb_percentb':_bb_percentb(close),
        'vol_zscore':_volume_zscore(hist['vol']),'close_strength':_avg_close_strength(hist,6),
        'efficiency_ratio':_efficiency_ratio(close,20),'rsi_alignment':ra,
        'momentum_decay':_momentum_decay(m1h,m4h,m1d),
        'momentum_acceleration':_momentum_acceleration(close,6),
        **_cs1__early_trend_features(hist),
        'funding_carry':0,'funding_contrarian':0,'oi_heat':0,
        'whale_flow':np.nan,'exchange_netflow':np.nan,'active_addresses':np.nan,'nvt_signal':np.nan,
    }


# ══════════════════════════════════════════════
# 结果构造
# ══════════════════════════════════════════════
def _build_scan_result(snapshot,score,direction,edge,factor_scores,passed,rank,universe_size,
                       category,config,weights=None,ic_snapshot=None):
    signals = _factor_signal_text(snapshot,direction,edge,factor_scores,rank,universe_size)
    rf = {
        'trend':_clamp(snapshot.trend_quality,0,100),
        'trigger':88.0 if direction in{'BUY','SELL'} else 30.0,
        'volume':_clamp(snapshot.volume_impulse/1.5*100,0,100),
        'location':_clamp(100-abs(snapshot.rsi_1h-50)*1.4,20,100),
        'freshness':_clamp(58+abs(edge)*24,30,96),
        'risk':_clamp(100-snapshot.atr_pct*7,25,92),
    }
    result = {
        'symbol':snapshot.symbol,'passed':passed,'score':round(score,2),
        'direction':direction,'signals':signals,'category':category,
        'strategy_category':category,'last_price':snapshot.last_price,
        'volume_24h':snapshot.volume_24h,'price_change_24h':snapshot.price_change_24h,
        'ranking_factors':rf,
        'details':{
            '机会类型':category,'评估':' | '.join(signals),
            '截面排名':f"{rank}/{universe_size}" if rank and universe_size else '-',
            '综合优势':f"{edge:+.2f}",
            '1H动量':f"{snapshot.momentum_1h:+.2f}%",'4H动量':f"{snapshot.momentum_4h:+.2f}%",
            '7D动量':f"{snapshot.momentum_1d:+.2f}%",'短期反转':f"{snapshot.short_reversal:+.2f}",
            '趋势质量':f"{snapshot.trend_quality:.1f}",
            '1H_RSI':f"{snapshot.rsi_1h:.1f}",'4H_RSI':f"{snapshot.rsi_4h:.1f}",
            'ATR%':f"{snapshot.atr_pct:.2f}%",'量比':f"{snapshot.volume_impulse:.2f}x",
            '量能Z分':f"{snapshot.vol_zscore:+.2f}σ",'MACD动量':f"{snapshot.macd_momentum:+.2f}",
            'BB%b':f"{snapshot.bb_percentb:.2f}",'趋势效率':f"{snapshot.efficiency_ratio:.2f}",
            '动量衰减':f"{snapshot.momentum_decay:+.2f}",
            '动量加速':f"{snapshot.momentum_acceleration:+.2f}",
            '早启动触发':f"{snapshot.early_trend_trigger:+.2f}",
            'EMA收敛突破':f"{snapshot.ema_compression_breakout:+.2f}",
            'RSI中轴转折':f"{snapshot.rsi_midline_turn:+.2f}",
            'MACD柱体拐头':f"{snapshot.macd_hist_turn:+.2f}",
            '唐奇安突破':f"{snapshot.donchian_breakout:+.2f}",
            '量价确认':f"{snapshot.volume_price_confirm:+.2f}",
            '3分钟回调确认':'是' if snapshot.m3_pullback_confirmed else '否',
            '3分钟结构':str(snapshot.m3_structure_state or '-'),
            '3分钟回调幅度%':f"{snapshot.m3_pullback_pct:.2f}",
            '3分钟原趋势脉冲%':f"{snapshot.m3_impulse_pct:.2f}",
            '3分钟回调结论':str(snapshot.m3_pullback_reason or '-'),
            '3分钟时效(根)':str(snapshot.m3_staleness_bars),  # v3fix
            '1H趋势延续根数':f"{snapshot.h1_trend_age}根({'⚠过老' if snapshot.h1_trend_age > int(_safe_float(config.get('max_h1_trend_age',12),12)) else '✓'})",  # v3.1
            '鲸鱼流入':f"{snapshot.whale_flow:+.4f}",
            '交易所净流':f"{snapshot.exchange_netflow:+.4f}",
            '活跃地址':f"{snapshot.active_addresses:.2f}",
            'NVT信号':f"{snapshot.nvt_signal:+.2f}",
            '资金费率%':f"{snapshot.funding_rate:+.4f}%",'持仓热度':f"{snapshot.oi_heat:.2f}",
            '因子z分':_format_factor_scores(factor_scores),
            '动态权重':_format_factor_scores(weights or _factor_weights()),
            'RollingIC':_format_factor_scores(ic_snapshot or {}),
        },
    }
    if build_opportunity_profile:
        try: result.update(build_opportunity_profile(base_score=score,direction=direction,
                volume_24h=snapshot.volume_24h,factors=rf,signals=signals))
        except Exception: pass
    return result

def _cs1__failed_result(symbol, reason):
    return {'symbol':str(getattr(symbol,'inst_id','')),'passed':False,'score':0.0,
            'direction':'WAIT','details':{'状态':reason or '数据不足'}}

def _factor_signal_text(snap,direction,edge,factor_scores,rank,usize):
    side = '多头' if direction=='BUY' else '空头' if direction=='SELL' else '观察'
    rt = f"截面{side}排名#{rank}/{usize}" if rank and usize else f"{side}综合评分"
    top = sorted(factor_scores.items(),key=lambda x:abs(float(x[1] or 0)),reverse=True)[:3]
    ft = ', '.join(f"{_factor_label(k)} {float(v):+.2f}" for k,v in top)
    staleness = f" 时效{snap.m3_staleness_bars}根" if snap.m3_staleness_bars else ''
    age_label = f" 趋势{snap.h1_trend_age}根" if snap.h1_trend_age else ''
    return [
        f"{rt}(优势{edge:+.2f})",
        f"动量(1H{snap.momentum_1h:+.1f}% 4H{snap.momentum_4h:+.1f}% 7D{snap.momentum_1d:+.1f}% 衰减{snap.momentum_decay:+.1f})",
        f"早启(触发{snap.early_trend_trigger:+.1f} EMA{snap.ema_compression_breakout:+.1f} RSI{snap.rsi_midline_turn:+.1f} MACD{snap.macd_hist_turn:+.1f})",
        f"趋势(质量{snap.trend_quality:.0f} 效率{snap.efficiency_ratio:.2f} ATR{snap.atr_pct:.1f}%)",
        f"量能(比{snap.volume_impulse:.1f}x z={snap.vol_zscore:+.1f} MACD{snap.macd_momentum:+.1f})",
        f"链上(鲸鱼{snap.whale_flow:+.3f} 交易所{snap.exchange_netflow:+.3f} NVT{snap.nvt_signal:+.1f})",
        f"3m(回调{snap.m3_pullback_pct:.1f}% 脉冲{snap.m3_impulse_pct:.1f}%{staleness})",
        f"时效({age_label} 3m{staleness})",  # v3.1
        f"主导: {ft}",
    ]


# ══════════════════════════════════════════════
# 底层工具（与 v2 一致 + v3 新增）
# ══════════════════════════════════════════════
def _cs1__normalize_weights(w):
    c = {}
    for n, v in w.items():
        fv = _safe_float(v, 0.0)
        c[n] = max(fv, 0.0) if np.isfinite(fv) else 0.0
    t=sum(c.values())
    return {n:v/t for n,v in c.items()} if t>1e-12 else dict(_factor_weights())

def _cap_weights(w, cap):
    cap=_clamp(cap,0.05,1.0); w=_cs1__normalize_weights(w)
    capped={};rem=[];rt=0;ct=0
    for n,v in w.items():
        if v>cap: capped[n]=cap; ct+=cap
        else: rem.append(n); rt+=v
    if not rem or ct>=1: return _cs1__normalize_weights(capped)
    budget=1-ct
    for n in rem: capped[n]=w[n]/rt*budget if rt>0 else budget/len(rem)
    return _cs1__normalize_weights(capped)

def _cs1__edge_to_score(edge, snap):
    score = 58+min(max(edge,0),2.2)*16
    if snap.volume_impulse>=1.4: score+=2
    if snap.atr_pct<=5: score+=2
    if abs(snap.funding_rate)<=0.07: score+=1.5    # v3fix: funding_rate已*100，0.07≈0.07%费率
    if snap.efficiency_ratio>=0.3: score+=1.5
    if snap.macd_momentum>0.5: score+=1
    if snap.momentum_decay>0.5: score+=1
    score -= min(18, max(snap.atr_pct-7.5,0)*2.4)
    return round(_clamp(score,0,100),2)

def _cs1__get_klines(km, bar):
    return km.get(bar) or km.get(bar.lower()) or km.get(bar.upper()) or []
def _pct_change(s, bars):
    if len(s)<=bars: return 0.0
    b=float(s.iloc[-(bars+1)]); l=float(s.iloc[-1])
    return (l/b-1)*100 if b>0 else 0.0
def _ema_slope_pct(s, span, lb):
    if len(s)<=lb+span: return 0.0
    e=s.ewm(span=span,adjust=False).mean()
    b=float(e.iloc[-(lb+1)]); l=float(e.iloc[-1])
    return (l/b-1)*100 if b>0 else 0.0
def _cs1__trend_quality(price,e21,e55,spread,slope,adx):
    bull=price>e21>e55; bear=price<e21<e55; sc=30.0
    if bull or bear: sc+=28
    sc+=_clamp(abs(spread)/2.8*18,0,18); sc+=_clamp(abs(slope)/2*14,0,14)
    sc+=_clamp((adx-12)/18*10,0,10)
    return round(_clamp(sc,0,100),2)
def _cs1__realized_vol_pct(close,window=48):
    if len(close)<3: return 0.0
    ret=close.pct_change().dropna().tail(window)
    return float(ret.std(ddof=0)*np.sqrt(max(len(ret),1))*100) if not ret.empty else 0.0
def _cs1__atr_pct(df,period=14):
    if len(df)<period+1: return 0.0
    pc=df['c'].shift(1)
    tr=pd.concat([df['h']-df['l'],(df['h']-pc).abs(),(df['l']-pc).abs()],axis=1).max(axis=1)
    atr=tr.ewm(alpha=1/period,adjust=False).mean().iloc[-1]
    c=float(df['c'].iloc[-1])
    return float(atr/c*100) if c>0 and pd.notna(atr) else 0.0
def _volume_ratio(vol,window=24):
    if len(vol)<window+1: return 1.0
    bl=float(vol.iloc[-(window+1):-1].mean())
    return float(vol.iloc[-1]/bl) if bl>0 else 1.0
def _macd_momentum(close):
    if len(close)<35: return 0.0
    e12=close.ewm(span=12,adjust=False).mean(); e26=close.ewm(span=26,adjust=False).mean()
    hist=e12-e26-(e12-e26).ewm(span=9,adjust=False).mean()
    if len(hist)<3: return 0.0
    lc=float(close.iloc[-1])
    return float(np.mean(hist.iloc[-3:].values)/lc*100) if lc>0 else 0.0
def _bb_percentb(close,period=20,width=2.0):
    if len(close)<period: return 0.5
    mid=close.rolling(period).mean().iloc[-1]
    std=close.rolling(period).std(ddof=1).iloc[-1]
    if not pd.notna(std) or std<=0: return 0.5
    u=mid+width*std; l=mid-width*std; p=float(close.iloc[-1])
    return float((p-l)/(u-l)) if u>l else 0.5
def _avg_close_strength(df,window=6):
    if len(df)<window: return 0.5
    r=df.tail(window); rng=r['h']-r['l']; rng=rng.replace(0,np.nan)
    s=(r['c']-r['l'])/rng
    return float(s.mean()) if not s.isna().all() else 0.5
def _rsi_alignment(r1h,r4h):
    b1=(r1h-50)/50; b4=(r4h-50)/50
    return _clamp(50+b1*b4*50,0,100)
def _parse_bar(bar):
    if isinstance(bar,dict):
        gv=lambda *ks:next((bar[k] for k in ks if k in bar),None)
        ts=gv('ts','timestamp','time');o=gv('o','open');h=gv('h','high');l=gv('l','low');c=gv('c','close');v=gv('vol','volume')
    elif isinstance(bar,pd.Series):
        gv=lambda *ks:next((bar[k] for k in ks if k in bar.index),None)
        ts=gv('ts','timestamp','time');o=gv('o','open');h=gv('h','high');l=gv('l','low');c=gv('c','close');v=gv('vol','volume')
    elif isinstance(bar,(list,tuple)) and len(bar)>=6: ts,o,h,l,c,v=bar[:6]
    else:
        try:
            gv=lambda *ks:next((getattr(bar,k) for k in ks if hasattr(bar,k)),None)
            ts=gv('ts','timestamp');o=gv('o','open');h=gv('h','high');l=gv('l','low');c=gv('c','close');v=gv('vol','volume')
        except Exception: return None
    try: return {'ts':float(ts),'o':float(o),'h':float(h),'l':float(l),'c':float(c),'vol':float(v or 0)}
    except (TypeError,ValueError): return None
def _df_to_rows(df): return df[['ts','o','h','l','c','vol']].astype(float).values.tolist()
def _cs1__symbol_from_backtest_data(data, config):
    km=data.get('klines_map') or {}; h1_rows=_cs1__get_klines(km,'1H') or data.get('klines') or []
    h1=_to_df(h1_rows); lp=float(h1['c'].iloc[-1]) if not h1.empty else 0
    return _cs1__MinimalSymbol(inst_id=str(config.get('inst_id','BT') or 'BT'),last_price=lp,
        volume_24h=float(h1['vol'].tail(24).sum()) if not h1.empty else 0,
        price_change_24h=_pct_change(h1['c'],24) if not h1.empty else 0,extra_data={'klines':km})
def _factor_label(n):
    return {'momentum_1h':'1H动量','momentum_4h':'4H动量','momentum_1d':'7D动量',
            'short_reversal':'短反转','trend_quality':'趋势','low_volatility':'低波',
            'liquidity':'流动性','volume_impulse':'量能','funding_carry':'资金成本',
            'funding_contrarian':'资金反转','oi_heat':'持仓热度','macd_momentum':'MACD',
            'bb_percentb':'BB%b','vol_zscore':'量Z','close_strength':'收盘强','efficiency_ratio':'效率',
            'rsi_alignment':'RSI同','momentum_decay':'衰减','momentum_acceleration':'加速',
            'early_trend_trigger':'早启','ema_compression_breakout':'EMA启',
            'rsi_midline_turn':'RSI转','macd_hist_turn':'MACD转',
            'donchian_breakout':'突破','volume_price_confirm':'量价',
            'whale_flow':'鲸鱼','exchange_netflow':'交易所流','active_addresses':'活跃地址',
            'nvt_signal':'NVT'}.get(n,n)
def _format_factor_scores(scores):
    if not scores: return '-'
    cleaned = [(k, _safe_float(v, 0.0)) for k, v in scores.items()]
    top=sorted(cleaned,key=lambda x:abs(x[1]),reverse=True)[:6]
    return '；'.join(f"{_factor_label(k)} {v:+.2f}" for k,v in top)

# ── 纯函数 API ──
def analyze_bars(h4,h1,last_price,config=None):
    cfg={**_cs1__DEFAULT_CONFIG,**(config or {})}
    d1=_aggregate_bars(h1 if isinstance(h1,pd.DataFrame) else _to_df(h1),24)
    h4_=h4 if isinstance(h4,pd.DataFrame) else _to_df(h4)
    h1_=h1 if isinstance(h1,pd.DataFrame) else _to_df(h1)
    kl={'1H':_df_to_rows(h1_),'4H':_df_to_rows(h4_),'1D':_df_to_rows(d1)}
    sym=_cs1__MinimalSymbol(inst_id='ANALYZE',last_price=float(last_price),
        volume_24h=float(h1_['vol'].tail(24).sum()) if not h1_.empty else 0,
        price_change_24h=_pct_change(h1_['c'],24) if not h1_.empty else 0,
        extra_data={'klines':kl})
    snap=_cs1__build_snapshot(sym,cfg)
    if not snap.valid: return {'valid':False,'reason':snap.reason,'score':0,'direction':'WAIT'}
    score,direction,edge,fs=_single_asset_score(snap,cfg)
    return {'valid':True,'score':score,'direction':direction,'edge':edge,'factor_scores':fs}
def klines_list_to_df(rows): return _to_df(rows)

# ════════════════════════════════════════════════════════════════════════════
# 子策略 2 — AI因子挖掘加密货币扫描策略
# ════════════════════════════════════════════════════════════════════════════

_ai2_CONFIG_SCHEMA = {
    "min_score":                    {"type": "float", "default": 72.0,        "label": "最低扫描分数"},
    "backtest_min_score":           {"type": "float", "default": 68.0,        "label": "回测最低入场分数"},
    "min_volume_24h":               {"type": "float", "default": 5_000_000.0, "label": "最小24H成交额"},
    "top_n":                        {"type": "int",   "default": 20,          "label": "最多输出数量"},
    "allow_short":                  {"type": "bool",  "default": True,        "label": "允许空头"},
    "use_dynamic_ic_weights":       {"type": "bool",  "default": False,       "label": "[不稳定] 启用Rolling IC动态权重"},
    "ic_weight_blend":              {"type": "float", "default": 0.55,        "label": "IC权重混合比例"},
    "max_factor_weight":            {"type": "float", "default": 0.24,        "label": "单因子最大权重"},
    "min_abs_edge":                 {"type": "float", "default": 0.24,        "label": "最小截面优势"},
    "correlation_penalty":          {"type": "float", "default": 0.08,        "label": "同质因子惩罚"},
    "correlation_threshold":        {"type": "float", "default": 0.85,        "label": "因子正交化相关阈值"},
    "deduplicate_base_asset":       {"type": "bool",  "default": True,        "label": "同Base资产只保留最高分"},
    "enable_mfin_interactions":     {"type": "bool",  "default": False,       "label": "[不稳定] 启用非线性交互项"},
    "enable_early_trend_factors":   {"type": "bool",  "default": True,        "label": "启用小时级早启动/转折因子"},
    "early_trend_min_trigger":      {"type": "float", "default": 0.18,        "label": "早启动最低触发强度"},
    "enable_llm_factors":           {"type": "bool",  "default": False,       "label": "[不稳定] 启用LLM/新闻/社交因子(需外部数据)"},
    "enable_on_chain":              {"type": "bool",  "default": True,        "label": "启用链上因子"},
    "risk_penalty_strength":        {"type": "float", "default": 0.75,        "label": "风险惩罚强度"},
    "max_atr_pct":                  {"type": "float", "default": 8.0,         "label": "最大ATR%"},
    "position_size":                {"type": "float", "default": 0.10,        "label": "回测仓位比例"},
    "require_m3_pullback_confirmation": {"type": "bool", "default": True,     "label": "要求3分钟回调企稳续势"},
    "m3_pullback_min_pct":          {"type": "float", "default": 0.50,        "label": "3分钟最小回调幅度%"},
    "m3_pullback_max_pct":          {"type": "float", "default": 2.20,        "label": "3分钟最大回调幅度%"},
    "m3_stabilization_bars":        {"type": "int",   "default": 4,           "label": "3分钟企稳确认根数"},
    # v3 新增 — 时效性
    "max_h1_trend_age":             {"type": "int",   "default": 12,          "label": "1H趋势最大延续根数（超过则惩罚）"},
    "h1_trend_age_penalty":         {"type": "float", "default": 0.10,        "label": "趋势过老时 edge 惩罚量"},
    "max_m3_staleness_bars":        {"type": "int",   "default": 15,          "label": "3m回调最大时效根数（超过则惩罚）"},
    "m3_freshness_penalty":         {"type": "float", "default": 0.08,        "label": "3m回调过旧时 edge 惩罚量"},
    "bonus_freshness_score":        {"type": "float", "default": 0.06,        "label": "两项时效均通过时 edge 加分"},
    # v4.1 趋势早期捕捉
    "h1_trend_age_hard_limit":      {"type": "int",   "default": 20,          "label": "1H趋势硬过滤上限（超过直接排除，0=不限）"},
    "require_early_trend_entry":    {"type": "bool",  "default": False,       "label": "只接受趋势早期信号（trend_age<=max_h1_trend_age）"},
    "early_trend_edge_discount":    {"type": "float", "default": 0.06,        "label": "早期趋势降低min_abs_edge的折扣量"},
}

_ai2__DEFAULT_CONFIG = {k: v["default"] for k, v in _ai2_CONFIG_SCHEMA.items()}

_BASE_WEIGHTS = {
    # v4.1: 降低动量权重（避免老趋势得高分），提升早期特征权重
    "momentum": 0.09, "trend": 0.12, "reversal": 0.07, "low_vol": 0.08,
    "liquidity": 0.08, "volume_impulse": 0.06, "funding_contra": 0.05,
    "oi_confirmation": 0.05, "on_chain_accumulation": 0.09,
    "network_value": 0.06, "llm_sentiment": 0.07, "event_momentum": 0.03,
    "developer_activity": 0.02, "early_trend_trigger": 0.13,
    "ema_compression_breakout": 0.04, "rsi_midline_turn": 0.035,
    "macd_hist_turn": 0.035, "donchian_breakout": 0.035,
    "volume_price_confirm": 0.03,
    "ema_cross_signal": 0.06,  # v4.1: 1H EMA金叉/死叉信号
}

_INTERACTION_TRANSFERS = {
    "ai_trend_interaction": (0.04, ("trend", "llm_sentiment")),
    "accumulation_breakout": (0.04, ("on_chain_accumulation", "volume_impulse")),
    "quality_momentum": (0.03, ("momentum", "low_vol")),
    "fresh_breakout_confirmation": (0.035, ("early_trend_trigger", "volume_impulse")),
}

VERSION = "3.0"


# ══════════════════════════════════════════════
class AIAutomatedAlphaCryptoScannerStrategy(BaseScannerStrategy if _HAS_SCANNER_BASE else object):
    required_bars = ["3m", "15m", "1H", "4H", "1D"]
    requires_derivative_metrics = True
    requires_on_chain_metrics = True
    name = "AI因子挖掘加密货币扫描策略"
    description = "Chain-of-Alpha v3: 稳定的技术/链上多因子截面扫描（IC/交互/LLM默认关闭）"
    strategy_type = "scan"

    def __init__(self, config=None):
        merged = {**_ai2__DEFAULT_CONFIG, **(config or {})}
        self.config = merged
        self.last_factor_weights = dict(_BASE_WEIGHTS)
        self.last_analysis: Dict[str, Any] = {}
        if _HAS_SCANNER_BASE and hasattr(super(), "__init__"):
            try:
                super().__init__(merged); self.config = merged
            except Exception:
                self.config = merged

    def _init_conditions(self):
        if ScanCondition is None or not hasattr(self, "add_condition"): return
        self.add_condition(ScanCondition(name="24H成交额", description="过滤成交额不足",
            field="volume_24h", operator=">=", value=self.config.get("min_volume_24h", 5_000_000.0)))

    def get_config_schema(self): return dict(_ai2_CONFIG_SCHEMA)

    def scan_symbol(self, symbol):
        snap = _ai2__build_snapshot(symbol, self.config)
        if not snap["valid"]: return _ai2__failed_result(symbol, snap["reason"])
        weights = _resolve_weights([symbol], self.config)
        self.last_factor_weights = weights
        result = _score_single_snapshot(snap, weights, self.config)
        self.last_analysis[getattr(symbol, "inst_id", "")] = result
        return result

    def scan_all_symbols(self, symbols):
        min_volume = float(self.config.get("min_volume_24h", 5_000_000.0) or 0.0)
        snapshots = []; source_symbols = []
        for sym in symbols:
            if float(getattr(sym, "volume_24h", 0.0) or 0.0) < min_volume: continue
            snap = _ai2__build_snapshot(sym, self.config)
            if snap["valid"]: snapshots.append(snap); source_symbols.append(sym)
        if not snapshots: return {"type": "ai_factor_mining", "all_opportunities": []}

        weights = _resolve_weights(source_symbols, self.config)
        ff = pd.DataFrame([s["factors"] for s in snapshots], index=[s["symbol"] for s in snapshots])
        z = ff.apply(_robust_zscore, axis=0).fillna(0.0)
        z = _apply_factor_correlation_control(z, weights, self.config)

        edge_series = pd.Series(0.0, index=z.index)
        for name, weight in weights.items():
            if name in z.columns: edge_series = edge_series + z[name] * float(weight)

        results = []
        for snap in snapshots:
            edge = float(edge_series.get(snap["symbol"], 0.0))
            sf = z.loc[snap["symbol"]].to_dict() if snap["symbol"] in z.index else {}
            r = _build_result_from_edge(snap, edge, weights, self.config, len(snapshots), sf)
            if r.get("passed"): results.append(r)

        results = _mmr_select(results, int(self.config.get("top_n", 20) or 20),
                              bool(self.config.get("deduplicate_base_asset", True)))
        self.last_factor_weights = weights
        return {"type": "ai_factor_mining", "all_opportunities": results}

    def generate_signal(self, data, *args, **kwargs):
        if isinstance(data, (list, tuple)): data = {"klines_map": {"1H": list(data)}}
        if not isinstance(data, dict) or not (data.get("klines_map") or data.get("klines")): return None
        if not data.get("klines_map"): data = {**data, "klines_map": {"1H": data.get("klines") or []}}
        cfg = dict(self.config)
        cfg["min_score"] = _cfg_float(cfg, "backtest_min_score", _cfg_float(cfg, "min_score", 68.0))
        # v3 修复: 复用 self.scan_symbol 而非重新实例化，保留外部注入配置
        self.config = cfg
        result = self.scan_symbol(_ai2__symbol_from_backtest_data(data, cfg))
        self.last_analysis["BACKTEST"] = result
        if not result.get("passed"): return None
        d = str(result.get("direction", "WAIT")).upper()
        if d not in {"BUY", "SELL"}: return None
        return {"action": "BUY" if d == "BUY" else "SHORT",
                "position_size": float(cfg.get("position_size", 0.10) or 0.10),
                "entry_price": float(result.get("last_price", 0.0) or 0.0),
                "reason": f"{result.get('category')} | 评分 {float(result.get('score', 0.0)):.1f}",
                "score": float(result.get("opportunity_score", result.get("score", 0.0)) or 0.0),
                "raw_result": result}

    def reset_backtest_state(self): self.last_analysis.clear()


# ══════════════════════════════════════════════
# 快照构建
# ══════════════════════════════════════════════

def _ai2__build_snapshot(symbol, config) -> Dict[str, Any]:
    inst_id = str(getattr(symbol, "inst_id", "") or "")
    extra = getattr(symbol, "extra_data", {}) or {}
    klines = extra.get("klines", {}) or {}
    m3 = _to_df(_ai2__get_klines(klines, "3m"))
    if m3.empty: m3 = _to_df(_ai2__get_klines(klines, "3M"))
    if m3.empty:
        m1 = _to_df(_ai2__get_klines(klines, "1m"))
        if len(m1) >= 120: m3 = _aggregate_bars(m1, 3)
    m15 = _to_df(_ai2__get_klines(klines, "15m"))
    h1 = _to_df(_ai2__get_klines(klines, "1H"))
    h4 = _to_df(_ai2__get_klines(klines, "4H"))
    d1 = _to_df(_ai2__get_klines(klines, "1D"))

    if h1.empty and len(m15) >= 16: h1 = _aggregate_bars(m15, 4)
    if h4.empty and len(h1) >= 12: h4 = _aggregate_bars(h1, 4)
    if d1.empty and len(h4) >= 24: d1 = _aggregate_bars(h4, 6)
    if len(h1) < 35: return {"valid": False, "symbol": inst_id, "reason": f"K线不足(1H={len(h1)})"}

    price = float(getattr(symbol, "last_price", 0.0) or h1["c"].iloc[-1])
    volume_24h = float(getattr(symbol, "volume_24h", 0.0) or (h1["c"] * h1["vol"]).tail(24).sum())
    price_change_24h = float(getattr(symbol, "price_change_24h", 0.0) or _pct_change(h1["c"], min(len(h1)-1, 24))*100.0)
    on_chain = extra.get("on_chain", {}) if isinstance(extra.get("on_chain", {}), dict) else {}
    llm = extra.get("llm_factors", {}) if isinstance(extra.get("llm_factors", {}), dict) else {}
    social = extra.get("social", {}) if isinstance(extra.get("social", {}), dict) else {}

    rsi_1h = _rsi(h1["c"], 14)
    atr_pct = _ai2__atr_pct(h4 if len(h4) >= 20 else h1, 14)
    realized_vol = _ai2__realized_vol_pct(h1["c"], 24)
    trend = _ai2__trend_quality(h1, h4, d1)

    # v3: 1H 趋势时效
    h1_trend_age = _measure_trend_age(h1["c"], fast=12, slow=34, direction=trend)

    # 3m 企稳（含时效性）
    micro_confirm = _micro_pullback_continuation(m3, trend, config)
    if bool(config.get("require_m3_pullback_confirmation", True)) and not micro_confirm["confirmed"]:
        return {"valid": False, "symbol": inst_id,
                "reason": f"3分钟回调续势未确认: {micro_confirm['reason']}"}

    momentum = _ai2__blend([
        _pct_change(h1["c"], 6) * 100.0,
        _pct_change(h4["c"], 6) * 100.0 if len(h4) > 7 else 0.0,
        _pct_change(d1["c"], 5) * 100.0 if len(d1) > 6 else 0.0,
    ], [0.45, 0.35, 0.20])
    reversal = _clamp((50.0 - rsi_1h) / 22.0, -1.5, 1.5) - _clamp(_pct_change(h1["c"], 3) * 18.0, -1.0, 1.0)
    liquidity = _clamp((log10(max(volume_24h, 1.0)) - 6.0) / 2.0, -1.0, 1.5)
    volume_impulse = _ai2__volume_impulse(h1)
    funding = _safe_float(extra.get("funding_rate"), 0.0) * 100.0
    oi_change = _safe_float(extra.get("open_interest_change_pct"), 0.0)

    on_chain_score = _on_chain_accumulation(on_chain) if bool(config.get("enable_on_chain", True)) else 0.0
    network_value = _network_value_score(on_chain) if bool(config.get("enable_on_chain", True)) else 0.0
    llm_sentiment = _llm_sentiment_score(extra, llm, social) if bool(config.get("enable_llm_factors", True)) else 0.0
    event_momentum = _event_score(llm, social) if bool(config.get("enable_llm_factors", True)) else 0.0
    developer_activity = _developer_activity_score(llm, social) if bool(config.get("enable_llm_factors", True)) else 0.0
    risk_score = _risk_warning_score(extra, llm, atr_pct, realized_vol)
    early = _ai2__early_trend_features(h1) if bool(config.get("enable_early_trend_factors", True)) else _ai2__empty_early_trend_features()

    # v4.1: 硬年龄上限过滤（趋势运行过久直接排除）
    hard_limit = int(_cfg_float(config, "h1_trend_age_hard_limit", 20))
    if hard_limit > 0 and h1_trend_age > hard_limit:
        return {"valid": False, "symbol": inst_id,
                "reason": f"1H趋势已运行{h1_trend_age}根，超过硬过滤上限{hard_limit}根"}
    # v4.1: require_early_trend_entry 严格模式
    max_age_cfg = int(_cfg_float(config, "max_h1_trend_age", 12))
    if bool(config.get("require_early_trend_entry", False)) and h1_trend_age > max_age_cfg:
        return {"valid": False, "symbol": inst_id,
                "reason": f"1H趋势已运行{h1_trend_age}根，超过早期入场上限{max_age_cfg}根"}

    factors = {
        "momentum": _clamp(momentum / 9.0, -2.0, 2.0),
        "trend": trend,
        "reversal": reversal,
        "low_vol": _clamp((5.5 - realized_vol) / 4.0, -1.5, 1.5),
        "liquidity": liquidity,
        "volume_impulse": _clamp(volume_impulse - 1.0, -1.0, 2.0),
        "funding_contra": _clamp(-funding / 0.08, -1.5, 1.5),
        "oi_confirmation": _clamp(oi_change / 8.0, -1.5, 1.5) * _ai2__direction_sign(momentum, trend),
        "on_chain_accumulation": on_chain_score,
        "network_value": network_value,
        "llm_sentiment": llm_sentiment,
        "event_momentum": event_momentum,
        "developer_activity": developer_activity,
        "early_trend_trigger": early["trigger"],
        "ema_compression_breakout": early["ema_compression_breakout"],
        "rsi_midline_turn": early["rsi_midline_turn"],
        "macd_hist_turn": early["macd_hist_turn"],
        "donchian_breakout": early["donchian_breakout"],
        "volume_price_confirm": early["volume_price_confirm"],
        "ema_cross_signal": early.get("ema_cross_signal", 0.0),  # v4.1
        "m3_pullback_score": micro_confirm["score"],
    }
    if bool(config.get("enable_mfin_interactions", True)):
        factors["ai_trend_interaction"] = _clamp(factors["trend"] * max(factors["llm_sentiment"], 0.0), -1.5, 1.5)
        factors["accumulation_breakout"] = _clamp(max(factors["on_chain_accumulation"], 0.0) * max(factors["volume_impulse"], 0.0), -1.5, 1.5)
        factors["quality_momentum"] = _clamp(factors["momentum"] * max(factors["low_vol"], 0.0), -1.5, 1.5)
        factors["fresh_breakout_confirmation"] = _clamp(
            factors["early_trend_trigger"] * max(abs(factors["volume_impulse"]), 0.0), -1.5, 1.5)

    return {
        "valid": True, "reason": "", "symbol": inst_id,
        "last_price": price, "volume_24h": volume_24h, "price_change_24h": price_change_24h,
        "atr_pct": atr_pct, "realized_vol": realized_vol, "rsi_1h": rsi_1h,
        "funding_rate_pct": funding, "open_interest_change_pct": oi_change,
        "risk_warning": risk_score,
        "h1_trend_age": h1_trend_age,
        "m3_pullback_confirmed": micro_confirm["confirmed"],
        "m3_structure_state": micro_confirm["state"],
        "m3_pullback_reason": micro_confirm["reason"],
        "m3_pullback_pct": micro_confirm["pullback_pct"],
        "m3_impulse_pct": micro_confirm["impulse_pct"],
        "m3_staleness_bars": micro_confirm.get("staleness_bars", 0),
        "early_trend": early, "factors": factors, "extra": extra,
    }


def _timeliness_edge_adjustment(snap: Dict[str, Any], config: Dict[str, Any], direction: str) -> float:
    """
    计算时效性对 edge 的净调整量（正=加分，负=惩罚）。
    两项都新鲜 → +bonus
    趋势过老 → -age_penalty（指数缓增，越界越多越重）
    3m 过旧 → -freshness_penalty
    """
    if direction not in {"BUY", "SELL"}: return 0.0

    age = snap.get("h1_trend_age", 0)
    max_age = int(_cfg_float(config, "max_h1_trend_age", 12.0))
    age_pen = _cfg_float(config, "h1_trend_age_penalty", 0.10)
    stale = snap.get("m3_staleness_bars", 0)
    max_stale = int(_cfg_float(config, "max_m3_staleness_bars", 15))
    fresh_pen = _cfg_float(config, "m3_freshness_penalty", 0.08)
    bonus = _cfg_float(config, "bonus_freshness_score", 0.06)

    adj = 0.0
    age_ok = age <= max_age
    stale_ok = stale <= max_stale

    if not age_ok:
        overflow = age - max_age
        adj -= age_pen * (1 - exp(-overflow / max(max_age * 0.5, 1.0)))
    if not stale_ok:
        overflow = stale - max_stale
        adj -= fresh_pen * (1 - exp(-overflow / max(max_stale * 0.5, 1.0)))
    if age_ok and stale_ok:
        adj += bonus

    return _clamp(adj, -0.20, 0.10)


# ══════════════════════════════════════════════
# 评分
# ══════════════════════════════════════════════

def _score_single_snapshot(snap, weights, config):
    sf = _single_asset_scoring_factors(snap, weights)
    edge = _weighted_edge(sf, weights)
    return _build_result_from_edge(snap, edge, weights, config, 1, sf)

def _single_asset_scoring_factors(snap, weights):
    scales = {
        "momentum":0.75,"trend":0.65,"reversal":0.85,"low_vol":0.75,"liquidity":0.80,
        "volume_impulse":0.65,"funding_contra":0.70,"oi_confirmation":0.70,
        "on_chain_accumulation":0.85,"network_value":0.80,"llm_sentiment":0.85,
        "event_momentum":0.85,"developer_activity":0.85,"ai_trend_interaction":0.65,
        "accumulation_breakout":0.65,"quality_momentum":0.65,"early_trend_trigger":0.75,
        "ema_compression_breakout":0.80,"rsi_midline_turn":0.85,"macd_hist_turn":0.85,
        "donchian_breakout":0.80,"volume_price_confirm":0.80,"fresh_breakout_confirmation":0.70,
        "ema_cross_signal":0.80,
    }
    raw = snap.get("factors", {}); scoring = {}
    for name in weights:
        if name not in raw: continue
        scale = max(scales.get(name, 0.80), 1e-9)
        scoring[name] = _clamp(float(raw.get(name, 0.0)) / scale, -3.0, 3.0)
    return scoring

def _weighted_edge(factors, weights):
    return float(sum(float(factors.get(n, 0.0)) * float(w) for n, w in weights.items()))

def _build_result_from_edge(snap, edge, weights, config, universe_size, scoring_factors=None):
    scoring_factors = scoring_factors or snap["factors"]
    risk_penalty = _cfg_float(config, "risk_penalty_strength", 0.75) * max(float(snap.get("risk_warning", 0.0)), 0.0)
    atr_penalty = max(float(snap.get("atr_pct", 0.0)) - _cfg_float(config, "max_atr_pct", 8.0), 0.0) * 0.08
    adjusted_edge = float(edge) - risk_penalty * np.sign(edge) - atr_penalty * np.sign(edge)
    early_trigger = float(snap["factors"].get("early_trend_trigger", 0.0) or 0.0)
    min_early = _cfg_float(config, "early_trend_min_trigger", 0.18)
    if abs(early_trigger) >= min_early and np.sign(early_trigger) == np.sign(adjusted_edge or early_trigger):
        adjusted_edge += np.sign(early_trigger) * min(abs(early_trigger) * 0.08, 0.16)

    allow_short = bool(config.get("allow_short", True))

    # v4.1: 极新鲜的 EMA 金叉/死叉信号可强制确定方向
    ema_cross = float(snap["factors"].get("ema_cross_signal", 0.0) or 0.0)
    if ema_cross >= 0.7:
        direction = "BUY"
    elif ema_cross <= -0.7 and allow_short:
        direction = "SELL"
    else:
        direction = "BUY" if adjusted_edge >= 0 else "SELL"
    if direction == "SELL" and not allow_short: direction = "WAIT"

    # v3: 时效性调整
    time_adj = _timeliness_edge_adjustment(snap, config, direction)
    adjusted_edge += time_adj

    score = _ai2__edge_to_score(abs(adjusted_edge), snap)

    # v4.1: 早期趋势降低 min_abs_edge 门槛
    age = snap.get("h1_trend_age", 0)
    max_age = int(_cfg_float(config, "max_h1_trend_age", 12.0))
    is_early = age <= max_age or abs(ema_cross) >= 0.5
    discount = _cfg_float(config, "early_trend_edge_discount", 0.06) if is_early else 0.0
    passed = (
        direction in {"BUY", "SELL"}
        and score >= _cfg_float(config, "min_score", 72.0)
        and abs(adjusted_edge) >= _cfg_float(config, "min_abs_edge", 0.24) - discount
    )

    # 时效性状态文字
    age = snap.get("h1_trend_age", 0)
    max_age = int(_cfg_float(config, "max_h1_trend_age", 12.0))
    stale = snap.get("m3_staleness_bars", 0)
    max_stale = int(_cfg_float(config, "max_m3_staleness_bars", 15))
    freshness_label = (
        f"1H趋势{age}根({'⚠过老' if age > max_age else '✓新鲜'}), "
        f"3m回调{stale}根前({'⚠过旧' if stale > max_stale else '✓新鲜'}), "
        f"时效调整{time_adj:+.3f}"
    )

    category = "AI因子多头机会" if direction=="BUY" else "AI因子空头机会" if direction=="SELL" else "AI因子观察"
    top_factors = _top_factor_reasons(scoring_factors, weights, direction)
    signals = [
        f"{category} 评分 {score:.1f}",
        f"综合Alpha {adjusted_edge:+.3f} / 原始 {edge:+.3f} / 风险惩罚 {risk_penalty:.2f}",
        f"趋势 {snap['factors'].get('trend',0):+.2f} / 动量 {snap['factors'].get('momentum',0):+.2f} / LLM {snap['factors'].get('llm_sentiment',0):+.2f}",
        f"早启 {snap['factors'].get('early_trend_trigger',0):+.2f} / EMA {snap['factors'].get('ema_compression_breakout',0):+.2f} / 突破 {snap['factors'].get('donchian_breakout',0):+.2f}",
        f"链上 {snap['factors'].get('on_chain_accumulation',0):+.2f} / 量能 {snap['factors'].get('volume_impulse',0):+.2f} / 低波 {snap['factors'].get('low_vol',0):+.2f}",
        f"时效: {freshness_label}",
        "；".join(top_factors),
    ]
    ranking_factors = {
        "trend": _clamp(50 + snap["factors"].get("trend", 0.0)*28, 0, 100),
        "trigger": _clamp(50 + abs(adjusted_edge)*45, 0, 100),
        "volume": _clamp(50 + snap["factors"].get("volume_impulse",0.0)*25 + snap["factors"].get("liquidity",0.0)*18, 0, 100),
        "location": _clamp(60 + snap["factors"].get("reversal",0.0)*12, 20, 95),
        "freshness": _clamp(
            55 + snap["factors"].get("event_momentum",0.0)*25 + snap["factors"].get("llm_sentiment",0.0)*12
            - max(age - max_age, 0) * 2.0 - max(stale - max_stale, 0) * 1.5,
            20, 96),
        "risk": _clamp(82 - snap.get("risk_warning",0.0)*30 - max(snap.get("atr_pct",0.0)-5.0,0.0)*4, 10, 95),
    }
    result = {
        "symbol": snap["symbol"], "passed": passed,
        "score": round(score, 2), "direction": direction,
        "signals": signals, "category": category, "strategy_category": category,
        "last_price": snap["last_price"], "volume_24h": snap["volume_24h"],
        "price_change_24h": snap["price_change_24h"], "ranking_factors": ranking_factors,
        "metrics": {
            "alpha_edge": round(adjusted_edge, 6), "raw_edge": round(edge, 6),
            "atr_pct": round(float(snap.get("atr_pct", 0.0)), 4),
            "realized_vol_pct": round(float(snap.get("realized_vol", 0.0)), 4),
            "rsi_1h": round(float(snap.get("rsi_1h", 0.0)), 4),
            "funding_rate_pct": round(float(snap.get("funding_rate_pct", 0.0)), 6),
            "open_interest_change_pct": round(float(snap.get("open_interest_change_pct", 0.0)), 4),
            "risk_warning": round(float(snap.get("risk_warning", 0.0)), 4),
            "early_trend_trigger": round(early_trigger, 6),
            "h1_trend_age": age, "m3_staleness_bars": stale,
            "timeliness_adj": round(time_adj, 4),
            "universe_size": universe_size,
        },
        "factor_scores": {k: round(float(v), 6) for k, v in snap["factors"].items()},
        "scoring_factor_scores": {k: round(float(v), 6) for k, v in scoring_factors.items()},
        "factor_weights": {k: round(float(v), 6) for k, v in weights.items()},
        "details": {
            "机会类型": category, "评估": " | ".join(signals),
            "综合Alpha": f"{adjusted_edge:+.3f}", "原始Alpha": f"{edge:+.3f}",
            "风险惩罚": f"{risk_penalty:.2f}", "时效调整": f"{time_adj:+.4f}",
            "ATR%": f"{float(snap.get('atr_pct',0.0)):.2f}",
            "1H RSI": f"{float(snap.get('rsi_1h',0.0)):.1f}",
            "资金费率%": f"{float(snap.get('funding_rate_pct',0.0)):+.4f}",
            "OI变化%": f"{float(snap.get('open_interest_change_pct',0.0)):+.2f}",
            "早启动触发": f"{early_trigger:+.2f}",
            "EMA收敛突破": f"{float(snap['factors'].get('ema_compression_breakout',0.0)):+.2f}",
            "RSI中轴转折": f"{float(snap['factors'].get('rsi_midline_turn',0.0)):+.2f}",
            "MACD柱体拐头": f"{float(snap['factors'].get('macd_hist_turn',0.0)):+.2f}",
            "唐奇安突破": f"{float(snap['factors'].get('donchian_breakout',0.0)):+.2f}",
            "量价确认": f"{float(snap['factors'].get('volume_price_confirm',0.0)):+.2f}",
            "3分钟回调确认": "是" if snap.get("m3_pullback_confirmed") else "否",
            "3分钟结构": str(snap.get("m3_structure_state", "-")),
            "3分钟回调幅度%": f"{float(snap.get('m3_pullback_pct',0.0)):.2f}",
            "3分钟原趋势脉冲%": f"{float(snap.get('m3_impulse_pct',0.0)):.2f}",
            "3分钟回调时效": f"{stale}根前({'超时' if stale > max_stale else '新鲜'})",
            "1H趋势延续根数": f"{age}根({'过老' if age > max_age else '正常'})",
            "时效状态": freshness_label,
        },
    }
    if build_opportunity_profile:
        try: result.update(build_opportunity_profile(score, direction, snap["volume_24h"], ranking_factors, signals))
        except Exception: pass
    return result


# ══════════════════════════════════════════════
# 权重管理
# ══════════════════════════════════════════════

def _resolve_weights(symbols, config):
    weights = dict(_BASE_WEIGHTS)
    if not bool(config.get("enable_llm_factors", True)):
        for n in ("llm_sentiment","event_momentum","developer_activity"): weights[n] = 0.0
    if not bool(config.get("enable_on_chain", True)):
        for n in ("on_chain_accumulation","network_value"): weights[n] = 0.0
    if not bool(config.get("enable_early_trend_factors", True)):
        for n in ("early_trend_trigger","ema_compression_breakout","rsi_midline_turn",
                  "macd_hist_turn","donchian_breakout","volume_price_confirm","fresh_breakout_confirmation"):
            weights[n] = 0.0
    if bool(config.get("enable_mfin_interactions", True)):
        weights = _add_interaction_weights(weights)
    if bool(config.get("use_dynamic_ic_weights", True)):
        ic = _collect_ic(symbols)
        if ic:
            dynamic = {n: max(_clamp(float(ic.get(n,0.0)),-0.18,0.18),0.0) for n in weights}
            if sum(dynamic.values()) > 0:
                dynamic = _ai2__normalize_weights(dynamic, _cfg_float(config,"max_factor_weight",0.24))
                blend = _clamp(_cfg_float(config,"ic_weight_blend",0.55),0.0,1.0)
                weights = {n: weights.get(n,0.0)*(1-blend)+dynamic.get(n,0.0)*blend for n in weights}
    return _ai2__normalize_weights(weights, _cfg_float(config,"max_factor_weight",0.24))

def _add_interaction_weights(weights):
    adjusted = dict(weights)
    for interaction, (amount, parents) in _INTERACTION_TRANSFERS.items():
        if any(float(adjusted.get(p,0.0)) <= 0.0 for p in parents):
            adjusted[interaction] = 0.0; continue
        parent_share = amount / max(len(parents), 1); funded = 0.0
        for p in parents:
            available = max(float(adjusted.get(p,0.0)),0.0)
            take = min(parent_share, available)
            adjusted[p] = available - take; funded += take
        adjusted[interaction] = adjusted.get(interaction,0.0) + funded
    return adjusted

def _collect_ic(symbols):
    values = {}
    for sym in symbols:
        extra = getattr(sym,"extra_data",{}) or {}
        for key in ("factor_ic","rolling_ic"):
            data = extra.get(key)
            if isinstance(data, dict):
                for name, value in data.items():
                    f = _safe_float(value, np.nan)
                    if np.isfinite(f): values.setdefault(str(name),[]).append(float(f))
    return {name: float(np.nanmean(vals)) for name, vals in values.items() if vals}

def _ai2__normalize_weights(weights, max_weight):
    cleaned = {n: max(float(v),0.0) for n,v in weights.items()}
    total = sum(cleaned.values())
    if total <= 0: cleaned = dict(_BASE_WEIGHTS); total = sum(cleaned.values())
    normalized = {n: v/total for n,v in cleaned.items()}
    capped = {n: min(v, max_weight) for n,v in normalized.items()}
    total = sum(capped.values())
    return {n: (v/total if total > 0 else 0.0) for n,v in capped.items()}

def _apply_factor_correlation_control(z, weights, config):
    if z.empty or z.shape[1] < 3: return z
    penalty_strength = _clamp(_cfg_float(config,"correlation_penalty",0.18),0.0,0.8)
    threshold = _clamp(_cfg_float(config,"correlation_threshold",0.78),0.30,0.98)
    if penalty_strength <= 0: return z
    corr = z.corr().abs().fillna(0.0); adjusted = z.copy()
    pairs = []
    cols = list(adjusted.columns)
    for i, left in enumerate(cols):
        for right in cols[i+1:]:
            cv = float(corr.loc[left, right])
            if cv >= threshold: pairs.append((cv, left, right))
    for cv, left, right in sorted(pairs, reverse=True):
        lw = abs(float(weights.get(left,0.0))); rw = abs(float(weights.get(right,0.0)))
        anchor, target = (left, right) if lw >= rw else (right, left)
        as_ = adjusted[anchor].replace([np.inf,-np.inf],np.nan).fillna(0.0)
        ts_ = adjusted[target].replace([np.inf,-np.inf],np.nan).fillna(0.0)
        ac = as_ - float(as_.mean()); tc = ts_ - float(ts_.mean())
        var = float(ac.var(ddof=0) or 0.0)
        if var <= 1e-12: continue
        beta = float((tc * ac).mean() / var); residual = tc - beta * ac
        if float(residual.std(ddof=0) or 0.0) <= 1e-10: residual = pd.Series(0.0, index=ts_.index)
        extra_str = (float(cv) - threshold) / max(1.0 - threshold, 1e-9)
        eff_str = _clamp(max(penalty_strength, extra_str), 0.0, 1.0)
        new_t = ts_ * (1 - eff_str) + residual * eff_str
        if float(new_t.std(ddof=0) or 0.0) <= 1e-10: new_t = pd.Series(0.0, index=ts_.index)
        adjusted[target] = new_t
    return adjusted

def _mmr_select(results, top_n, deduplicate_base_asset=True):
    ordered = sorted(results, key=lambda x: float(x.get("opportunity_score",x.get("score",0.0)) or 0.0), reverse=True)
    selected = []; seen = set()
    for item in ordered:
        base = _base_asset_key(str(item.get("symbol","")))
        if deduplicate_base_asset and base in seen: continue
        selected.append(item); seen.add(base)
        if len(selected) >= top_n: break
    return selected

def _base_asset_key(symbol):
    return str(symbol or "").replace("/","-").replace("_","-").upper().split("-")[0]


# ══════════════════════════════════════════════
# 工具函数（与 v2 保持一致）
# ══════════════════════════════════════════════

def _ai2__failed_result(symbol, reason):
    return {"symbol": str(getattr(symbol,"inst_id","") or ""), "passed": False,
            "score": 0.0, "direction": "WAIT", "signals": [],
            "details": {"状态": reason}, "metrics": {}}

def _ai2__edge_to_score(abs_edge, snap):
    quality = (min(max(snap["factors"].get("liquidity",0.0),0.0),1.2)*3.0
             + min(max(snap["factors"].get("volume_impulse",0.0),0.0),1.2)*2.5
             + min(max(snap["factors"].get("low_vol",0.0),0.0),1.2)*2.0)
    return _clamp(50.0 + abs_edge*58.0 + quality, 0.0, 100.0)

def _top_factor_reasons(factors, weights, direction):
    sign = 1.0 if direction=="BUY" else -1.0
    contribs = [(n, float(s)*sign*float(weights.get(n,0.0))) for n,s in factors.items()]
    names = {"momentum":"动量","trend":"趋势质量","reversal":"反转位置","low_vol":"低波质量",
             "liquidity":"流动性","volume_impulse":"量能脉冲","funding_contra":"资金费率反身性",
             "oi_confirmation":"OI确认","on_chain_accumulation":"链上积累","network_value":"网络估值",
             "llm_sentiment":"LLM/新闻情绪","event_momentum":"事件动量","developer_activity":"开发活跃",
             "ai_trend_interaction":"AI情绪趋势共振","accumulation_breakout":"链上放量共振",
             "quality_momentum":"低波动量共振","early_trend_trigger":"小时早启触发",
             "ema_compression_breakout":"EMA收敛突破","rsi_midline_turn":"RSI中轴转折",
             "macd_hist_turn":"MACD柱体拐头","donchian_breakout":"唐奇安突破",
             "volume_price_confirm":"量价确认","fresh_breakout_confirmation":"早启放量确认"}
    return [f"{names.get(n,n)}({v:+.3f})" for n,v in sorted(contribs,key=lambda x:x[1],reverse=True)[:5]]

def _ai2__get_klines(klines_map, bar):
    aliases = {"1H":["1H","1h","60m","60M"],"4H":["4H","4h","240m","240M"],
               "1D":["1D","1d","D","day"],"15m":["15m","15M"]}
    for key in aliases.get(bar,[bar,bar.lower(),bar.upper()]):
        if key in klines_map and klines_map.get(key): return klines_map.get(key)
    return []

def _ai2__atr_pct(df, period=14):
    if len(df) < period+2: return 0.0
    pc = df["c"].shift(1)
    tr = pd.concat([(df["h"]-df["l"]).abs(),(df["h"]-pc).abs(),(df["l"]-pc).abs()],axis=1).max(axis=1)
    atr = float(tr.ewm(alpha=1/period,adjust=False).mean().iloc[-1] or 0.0)
    p = float(df["c"].iloc[-1] or 0.0); return atr/p*100.0 if p > 0 else 0.0

def _ai2__realized_vol_pct(close, window=24):
    ret = close.pct_change().dropna().tail(window)
    if ret.empty: return 0.0
    return float(ret.std(ddof=0)*sqrt(max(len(ret),1))*100.0)

def _ai2__trend_quality(h1, h4, d1):
    score = 0.0
    for df, fast, slow, w in [(h1,12,34,0.45),(h4,8,21,0.35),(d1,5,13,0.20)]:
        if len(df) < slow+2: continue
        ef = df["c"].ewm(span=fast,adjust=False).mean()
        es = df["c"].ewm(span=slow,adjust=False).mean()
        slope = float(ef.diff().tail(3).mean() or 0.0)
        d = 1.0 if ef.iloc[-1]>es.iloc[-1] and slope>0 else -1.0 if ef.iloc[-1]<es.iloc[-1] and slope<0 else 0.0
        score += d*w*(0.65+0.35*_efficiency_ratio(df["c"],min(20,len(df)-1)))
    return _clamp(score,-1.5,1.5)

def _ai2__volume_impulse(df, window=24):
    if len(df) < window+2: return 1.0
    base = float(df["vol"].iloc[-(window+1):-1].median() or 0.0)
    latest = float(df["vol"].tail(3).mean() or 0.0)
    return latest/base if base>0 else 1.0

def _on_chain_accumulation(oc):
    wf=_safe_float(oc.get("whale_flow"),0.0); en=_safe_float(oc.get("exchange_netflow"),0.0); sf=_safe_float(oc.get("stablecoin_flow"),0.0)
    return _clamp(_clamp(wf,-2,2)-_clamp(en,-2,2)*0.65+_clamp(sf,-2,2)*0.35,-2,2)

def _network_value_score(oc):
    active=_safe_float(oc.get("active_addresses_z"),np.nan)
    if not np.isfinite(active):
        active=_safe_float(oc.get("active_addresses"),0.0); active=_clamp((log10(max(active,1.0))-4.0)/2.0,-1.5,1.5)
    nvt=_safe_float(oc.get("nvt_signal_z"),np.nan)
    if not np.isfinite(nvt):
        nvt=_safe_float(oc.get("nvt_signal"),0.0); nvt=_clamp((60.0-nvt)/45.0,-1.5,1.5) if nvt>0 else 0.0
    mvrv=_safe_float(oc.get("mvrv_z"),0.0)
    return _clamp(active*0.45+nvt*0.35+_clamp(-mvrv/2,-1.5,1.5)*0.20,-2,2)

def _llm_sentiment_score(extra, llm, social):
    vals=[_safe_float(llm.get("sentiment"),np.nan),_safe_float(llm.get("narrative_strength"),np.nan),
          _safe_float(social.get("sentiment"),np.nan),_safe_float(extra.get("news_sentiment"),np.nan)]
    clean=[v for v in vals if np.isfinite(v)]; return 0.0 if not clean else _clamp(float(np.mean(clean)),-2,2)

def _event_score(llm, social):
    gs=_safe_float(social.get("galaxy_score"),np.nan); gc=gs/50.0-1.0 if np.isfinite(gs) else np.nan
    vals=[_safe_float(llm.get("event_score"),np.nan),_safe_float(llm.get("announcement_score"),np.nan),
          _safe_float(social.get("social_volume_z"),np.nan),gc]
    clean=[v for v in vals if np.isfinite(v)]; return 0.0 if not clean else _clamp(float(np.mean(clean)),-2,2)

def _developer_activity_score(llm, social):
    vals=[_safe_float(llm.get("github_activity"),np.nan),_safe_float(llm.get("dev_momentum"),np.nan),
          _safe_float(social.get("developer_activity"),np.nan)]
    clean=[v for v in vals if np.isfinite(v)]; return 0.0 if not clean else _clamp(float(np.mean(clean)),-2,2)

def _risk_warning_score(extra, llm, atr_pct, realized_vol):
    warnings=[_safe_float(llm.get("risk_warning"),0.0),_safe_float(extra.get("risk_warning"),0.0),
              max(atr_pct-7.0,0.0)/4.0,max(realized_vol-8.0,0.0)/5.0]
    return _clamp(float(np.nanmean(warnings)),0.0,2.5)

def _ai2__empty_early_trend_features():
    return {"trigger":0.0,"early_trend_trigger":0.0,"ema_compression_breakout":0.0,
            "rsi_midline_turn":0.0,"macd_hist_turn":0.0,"donchian_breakout":0.0,
            "volume_price_confirm":0.0,"ema_cross_signal":0.0}

def _ai2__early_trend_features(h1):
    if h1 is None or len(h1) < 58: return _ai2__empty_early_trend_features()
    close=h1["c"].astype(float); high=h1["h"].astype(float)
    low=h1["l"].astype(float); vol=h1["vol"].astype(float)
    price=float(close.iloc[-1])
    if price<=0: return _ai2__empty_early_trend_features()
    ef8=close.ewm(span=8,adjust=False).mean(); ef21=close.ewm(span=21,adjust=False).mean(); ef55=close.ewm(span=55,adjust=False).mean()
    prev_spread=float((ef21.iloc[-7]-ef55.iloc[-7])/max(abs(ef55.iloc[-7]),1e-9)*100.0)
    cur_fs=float((ef8.iloc[-1]-ef21.iloc[-1])/price*100.0)
    cur_ss=float((ef21.iloc[-1]-ef55.iloc[-1])/price*100.0)
    spread_delta=cur_ss-prev_spread; compression=1.0-_clamp(abs(prev_spread)/2.2,0.0,1.0)
    ema_c=_clamp((cur_fs*0.9+spread_delta*0.7)*(0.65+compression*0.55),-2.0,2.0)
    ph=float(high.iloc[-21:-1].max()); pl=float(low.iloc[-21:-1].min())
    ub=(price/ph-1.0)*100.0 if ph>0 else 0.0; db=(pl/price-1.0)*100.0 if pl>0 else 0.0
    donchian=_clamp(ub/0.9,0.0,2.0)-_clamp(db/0.9,0.0,2.0)
    rsi_s=_ai2__rsi_series_wilder(close,14)
    rsi_now=float(rsi_s.iloc[-1]) if not rsi_s.empty else 50.0
    rsi_prev=float(rsi_s.iloc[-4]) if len(rsi_s)>=4 else 50.0
    rsi_mid=_clamp((rsi_now-50.0)/16.0+(rsi_now-rsi_prev)/12.0,-2.0,2.0)
    mh=_ai2__macd_hist_series(close)
    if len(mh)>=5:
        hn=float(mh.iloc[-1])/price*100.0; hp=float(mh.iloc[-4])/price*100.0
        macd_turn=_clamp(hn*8.0+(hn-hp)*3.0,-2.0,2.0)
    else: macd_turn=0.0
    tr=pd.concat([(high-low).abs(),(high-close.shift(1)).abs(),(low-close.shift(1)).abs()],axis=1).max(axis=1)
    brange=float(tr.iloc[-25:-1].median() or 0.0); rr=float(tr.tail(3).mean()/brange) if brange>0 else 1.0
    bvol=float(vol.iloc[-25:-1].median() or 0.0); vr=float(vol.tail(3).mean()/bvol) if bvol>0 else 1.0
    cs=_ai2__close_strength(h1,3); pd_=1.0 if close.iloc[-1]>=close.iloc[-4] else -1.0
    vp=pd_*_clamp((rr-1.0)*0.7+(vr-1.0)*0.6+(cs-0.5)*1.2,-2.0,2.0)
    # v4.1: 检测 1H EMA12/34 金叉/死叉
    ema_cross = _ai2__ema_just_crossed(close, fast=12, slow=34, lookback=6)
    trigger=_clamp(ema_c*0.24+donchian*0.20+rsi_mid*0.15+macd_turn*0.13+vp*0.10+ema_cross*0.20,-2.5,2.5)
    return {"trigger":trigger,"early_trend_trigger":trigger,"ema_compression_breakout":ema_c,
            "rsi_midline_turn":rsi_mid,"macd_hist_turn":macd_turn,"donchian_breakout":donchian,
            "volume_price_confirm":vp,"ema_cross_signal":ema_cross}

def _ai2__ema_just_crossed(close, fast=12, slow=34, lookback=6):
    """v4.1: 检测 EMA 是否在最近 lookback 根内刚发生金叉/死叉。+1~0=金叉, -1~0=死叉"""
    if len(close) < slow + lookback + 2:
        return 0.0
    ema_f = close.ewm(span=fast, adjust=False).mean()
    ema_s = close.ewm(span=slow, adjust=False).mean()
    diff = ema_f - ema_s
    for i in range(1, min(lookback + 1, len(diff))):
        cur = float(diff.iloc[-i]); prev = float(diff.iloc[-i - 1])
        if prev <= 0 < cur: return 1.0 * (1.0 - (i - 1) * 0.12)
        if prev >= 0 > cur: return -1.0 * (1.0 - (i - 1) * 0.12)
    return 0.0


def _ai2__rsi_series_wilder(close, period=14):
    if len(close)<period+2: return pd.Series([50.0]*len(close),index=close.index)
    delta=close.diff(); gain=delta.clip(lower=0).ewm(alpha=1/period,adjust=False).mean()
    loss=(-delta.clip(upper=0)).ewm(alpha=1/period,adjust=False).mean()
    rs=gain/loss.replace(0,np.nan)
    return (100.0-100.0/(1.0+rs)).replace([np.inf,-np.inf],np.nan).fillna(50.0).clip(0.0,100.0)

def _ai2__macd_hist_series(close):
    if len(close)<35: return pd.Series(dtype=float)
    fast=close.ewm(span=12,adjust=False).mean(); slow=close.ewm(span=26,adjust=False).mean()
    dif=fast-slow; dea=dif.ewm(span=9,adjust=False).mean(); return dif-dea

def _ai2__close_strength(df, window=3):
    if len(df)<window: return 0.5
    tail=df.tail(window); rng=(tail["h"]-tail["l"]).replace(0,np.nan)
    s=(tail["c"]-tail["l"])/rng; return float(s.mean()) if not s.isna().all() else 0.5

def _ai2__symbol_from_backtest_data(data, config):
    km=data.get("klines_map") or {}
    h1=_to_df(_ai2__get_klines(km,"1H") or _ai2__get_klines(km,"15m") or data.get("klines") or [])
    price=float(h1["c"].iloc[-1]) if not h1.empty else 0.0
    volume=float((h1["c"]*h1["vol"]).tail(48).sum()) if not h1.empty else 0.0
    extra={"klines":km,"funding_rate":data.get("funding_rate",0.0),"open_interest_change_pct":data.get("open_interest_change_pct",0.0),
           "on_chain":data.get("on_chain",{}),"llm_factors":data.get("llm_factors",{}),"social":data.get("social",{}),"news_sentiment":data.get("news_sentiment",0.0)}
    return _MinimalSymbol(inst_id=str(config.get("inst_id","BACKTEST") or "BACKTEST"),last_price=price,volume_24h=volume,
        price_change_24h=_pct_change(h1["c"],min(len(h1)-1,24))*100.0 if not h1.empty else 0.0,extra_data=extra)

def _ai2__blend(values,weights):
    total=denom=0.0
    for v,w in zip(values,weights):
        if np.isfinite(v):total+=float(v)*float(w);denom+=float(w)
    return total/denom if denom>0 else 0.0

def _ai2__direction_sign(momentum,trend):
    if abs(float(momentum or 0.0))>1e-9: return 1.0 if momentum>0 else -1.0
    if abs(float(trend or 0.0))>1e-9: return 1.0 if trend>0 else -1.0
    return 1.0

# ════════════════════════════════════════════════════════════════════════════
# 子策略 3 — DRL元学习小时趋势启动扫描策略
# ════════════════════════════════════════════════════════════════════════════

_drl3_CONFIG_SCHEMA = {
    "min_score":                    {"type": "float", "default": 74.0,  "label": "最低扫描分数"},
    "backtest_min_score":           {"type": "float", "default": 70.0,  "label": "回测最低入场分数"},
    "min_volume_24h":               {"type": "float", "default": 8_000_000.0, "label": "最小24H成交额"},
    "top_n":                        {"type": "int",   "default": 20,    "label": "最多输出数量"},
    "allow_short":                  {"type": "bool",  "default": True,  "label": "允许空头"},
    "min_abs_edge":                 {"type": "float", "default": 0.22,  "label": "最小优势"},
    "position_size":                {"type": "float", "default": 0.10,  "label": "回测仓位比例"},
    "entropy_alpha":                {"type": "float", "default": 0.18,  "label": "SAC熵权重"},
    "q_temperature":                {"type": "float", "default": 0.85,  "label": "Q softmax温度"},
    "double_q_blend":               {"type": "float", "default": 0.36,  "label": "Double-Q目标网络混合"},
    "meta_adapt_strength":          {"type": "float", "default": 0.40,  "label": "元学习适配强度"},
    "risk_penalty_strength":        {"type": "float", "default": 0.70,  "label": "风险惩罚强度"},
    "max_atr_pct":                  {"type": "float", "default": 8.2,   "label": "最大ATR%"},
    "hourly_start_momentum_bps":    {"type": "float", "default": 45.0,  "label": "1H趋势启动最小动量bps"},
    "hourly_start_breakout_bps":    {"type": "float", "default": 20.0,  "label": "1H突破压强最小bps"},
    "hourly_start_volume_impulse":  {"type": "float", "default": 1.18,  "label": "1H启动最小量能脉冲"},
    "require_m3_pullback_confirmation": {"type": "bool", "default": True, "label": "要求3分钟回调企稳续势"},
    "m3_pullback_min_pct":          {"type": "float", "default": 0.50,  "label": "3分钟最小回调幅度%"},
    "m3_pullback_max_pct":          {"type": "float", "default": 2.20,  "label": "3分钟最大回调幅度%"},
    "m3_stabilization_bars":        {"type": "int",   "default": 4,     "label": "3分钟企稳确认根数"},
    # v2 新增 ── 时效性
    "max_h1_trend_age":             {"type": "int",   "default": 12,    "label": "1H趋势最大延续根数（超过则降权）"},
    "h1_trend_age_penalty":         {"type": "float", "default": 8.0,   "label": "趋势过老时分数惩罚"},
    "max_m3_staleness_bars":        {"type": "int",   "default": 15,    "label": "3m回调最大时效根数（超过则降权）"},
    "m3_freshness_penalty":         {"type": "float", "default": 6.0,   "label": "3m回调过旧时分数惩罚"},
    # v2.1 新增
    "require_m3_freshness":         {"type": "bool",  "default": True,  "label": "必须通过3m时效性检查"},
    "m3_min_impulse_pct":           {"type": "float", "default": 0.65,  "label": "3m最小原趋势脉冲%"},
    "vol_continuation_min_ratio":   {"type": "float", "default": 0.78,  "label": "企稳量能续航最低比例"},
    "enable_funding_timing_guard":  {"type": "bool",  "default": True,  "label": "启用资金费率结算时段回避"},
    "funding_avoid_minutes":        {"type": "int",   "default": 15,    "label": "资金费率结算前回避分钟数"},
    "enable_btc_correlation_filter":{"type": "bool",  "default": False, "label": "启用BTC相关性过滤(需BTC K线)"},
    "max_btc_correlation":          {"type": "float", "default": 0.85,  "label": "最大允许的BTC相关性"},
    # v2.2 新增 ── 3m微观结构增强指标
    "enable_atr_squeeze_check":     {"type": "bool",  "default": True,  "label": "启用波动率收缩检测"},
    "atr_squeeze_ratio":            {"type": "float", "default": 0.55,  "label": "ATR收缩比例（当前/长期）"},
    "enable_volume_delta_check":    {"type": "bool",  "default": True,  "label": "启用买卖力量检测"},
    "volume_delta_min_ratio":       {"type": "float", "default": 1.15,  "label": "企稳段买入量/卖出量最低比值"},
    "enable_vwap_alignment_check":  {"type": "bool",  "default": True,  "label": "启用VWAP对齐检测"},
    # v4.0 新增
    "enable_bidask_spread_filter":  {"type": "bool",  "default": True,  "label": "启用买卖价差过滤"},
    "max_bidask_spread_pct":        {"type": "float", "default": 0.25,  "label": "最大允许买卖价差%"},
}

_drl3__DEFAULT_CONFIG = {k: v["default"] for k, v in _drl3_CONFIG_SCHEMA.items()}


# ══════════════════════════════════════════════
class DRLMetaHourlyTrendStartScannerStrategy(BaseScannerStrategy if _HAS_SCANNER_BASE else object):
    required_bars = ["3m", "15m", "1H", "4H", "1D"]
    requires_derivative_metrics = True
    requires_on_chain_metrics = True
    name = "DRL元学习小时趋势启动扫描策略"
    description = "DQN/Double/Dueling + A2C + SAC + 元学习 + 趋势时效性双重过滤"
    strategy_type = "scan"

    def __init__(self, config=None):
        merged = {**_drl3__DEFAULT_CONFIG, **(config or {})}
        self.config = merged
        self.last_analysis: Dict[str, Dict[str, Any]] = {}
        if _HAS_SCANNER_BASE and hasattr(super(), "__init__"):
            try:
                super().__init__(merged); self.config = merged
            except Exception:
                self.config = merged

    def _init_conditions(self):
        if ScanCondition is None or not hasattr(self, "add_condition"): return
        self.add_condition(ScanCondition(name="24H成交额", description="过滤流动性不足",
            field="volume_24h", operator=">=", value=self.config.get("min_volume_24h", 8_000_000.0)))

    def get_config_schema(self): return dict(_drl3_CONFIG_SCHEMA)

    def scan_symbol(self, symbol):
        snap = _drl3__build_snapshot(symbol, self.config)
        if not snap["valid"]: return _drl3__failed_result(symbol, snap["reason"])
        result = _score_snapshot(snap, self.config)
        self.last_analysis[str(getattr(symbol, "inst_id", ""))] = result
        return result

    def scan_all_symbols(self, symbols):
        min_vol = _cfg_float(self.config, "min_volume_24h", 8_000_000.0)
        candidates = []
        for sym in symbols:
            if float(getattr(sym, "volume_24h", 0.0) or 0.0) < min_vol: continue
            r = self.scan_symbol(sym)
            if r.get("passed"): candidates.append(r)
        candidates.sort(key=_drl3__result_sort_key, reverse=True)
        return {"type": "drl_meta_hourly_trend_start",
                "all_opportunities": candidates[:int(self.config.get("top_n", 20) or 20)]}

    def generate_signal(self, data, *a, **kw):
        if isinstance(data, (list, tuple)): data = {"klines_map": {"1H": list(data)}}
        if not isinstance(data, dict) or not (data.get("klines_map") or data.get("klines")): return None
        if not data.get("klines_map"): data = {**data, "klines_map": {"1H": data.get("klines") or []}}
        cfg = dict(self.config)
        cfg["min_score"] = _cfg_float(cfg, "backtest_min_score", _cfg_float(cfg, "min_score", 70.0))
        sym = _drl3__symbol_from_backtest_data(data, cfg)
        result = _score_snapshot(_drl3__build_snapshot(sym, cfg), cfg)
        if not result.get("passed"): return None
        d = str(result.get("direction", "WAIT")).upper()
        if d not in {"BUY", "SELL"}: return None
        return {"action": "BUY" if d == "BUY" else "SHORT",
                "position_size": _cfg_float(cfg, "position_size", 0.10),
                "entry_price": float(result.get("last_price", 0.0) or 0.0),
                "reason": f"{result.get('category')} | 评分 {float(result.get('score', 0.0)):.1f}",
                "score": float(result.get("opportunity_score", result.get("score", 0.0)) or 0.0),
                "raw_result": result}

    def reset_backtest_state(self): self.last_analysis.clear()


# ══════════════════════════════════════════════
# 快照构建
# ══════════════════════════════════════════════

def _drl3__build_snapshot(symbol, config) -> Dict[str, Any]:
    inst_id = str(getattr(symbol, "inst_id", "") or "")
    extra = getattr(symbol, "extra_data", {}) or {}
    klines = extra.get("klines", {}) or {}

    m3 = _to_df(_drl3__get_klines(klines, "3m"))
    if m3.empty: m3 = _to_df(_drl3__get_klines(klines, "3M"))
    if m3.empty:
        m1 = _to_df(_drl3__get_klines(klines, "1m"))
        if len(m1) >= 120: m3 = _aggregate_bars(m1, 3)
    m15 = _to_df(_drl3__get_klines(klines, "15m"))
    h1 = _to_df(_drl3__get_klines(klines, "1H"))
    h4 = _to_df(_drl3__get_klines(klines, "4H"))
    d1 = _to_df(_drl3__get_klines(klines, "1D"))

    if h1.empty and len(m15) >= 16: h1 = _aggregate_bars(m15, 4)
    if h4.empty and len(h1) >= 12: h4 = _aggregate_bars(h1, 4)
    if d1.empty and len(h4) >= 24: d1 = _aggregate_bars(h4, 6)
    if len(h1) < 45: return {"valid": False, "symbol": inst_id, "reason": f"K线不足(1H={len(h1)})"}
    if m3.empty and bool(config.get("require_m3_pullback_confirmation", True)):
        return {"valid": False, "symbol": inst_id, "reason": "3m数据不可用且要求回调确认"}

    price = float(getattr(symbol, "last_price", 0.0) or h1["c"].iloc[-1])
    volume_24h = float(getattr(symbol, "volume_24h", 0.0) or (h1["c"] * h1["vol"]).tail(24).sum())
    change_24h = float(getattr(symbol, "price_change_24h", 0.0) or _pct_change(h1["c"], min(len(h1) - 1, 24)) * 100.0)
    funding_rate = _safe_float(extra.get("funding_rate"), 0.0) * 100.0
    oi_change = _safe_float(extra.get("open_interest_change_pct"), 0.0)
    on_chain = extra.get("on_chain", {}) if isinstance(extra.get("on_chain", {}), dict) else {}
    social = extra.get("social", {}) if isinstance(extra.get("social", {}), dict) else {}
    llm = extra.get("llm_factors", {}) if isinstance(extra.get("llm_factors", {}), dict) else {}

    m15_mom = _pct_change(m15["c"], 4) * 10_000 if len(m15) > 6 else _pct_change(h1["c"], 1) * 10_000
    h1_mom = _pct_change(h1["c"], 4) * 10_000
    h4_mom = _pct_change(h4["c"], 3) * 10_000 if len(h4) > 5 else 0.0
    d1_mom = _pct_change(d1["c"], 3) * 10_000 if len(d1) > 5 else 0.0
    trend_h1 = _ema_trend_score(h1["c"], 12, 34)
    trend_h4 = _ema_trend_score(h4["c"], 8, 21) if len(h4) > 25 else trend_h1 * 0.7
    trend_d1 = _ema_trend_score(d1["c"], 5, 13) if len(d1) > 16 else trend_h4 * 0.6
    trend_alignment = _clamp(trend_h1 * 0.45 + trend_h4 * 0.35 + trend_d1 * 0.20, -2.0, 2.0)

    # ── v2: 1H 趋势时效性 ──
    h1_trend_age = _measure_trend_age(h1["c"], fast=12, slow=34, direction=trend_alignment)

    # ── v2.1: 资金费率结算时段回避 ──
    funding_risk = _check_funding_timing(config) if _cfg_float(config, "enable_funding_timing_guard", 1.0) > 0 else False

    # ── v2.1: BTC相关性（如果可用）──
    btc_corr = 0.0
    btc_klines = extra.get("btc_klines") if isinstance(extra, dict) else None
    if btc_klines and bool(config.get("enable_btc_correlation_filter", False)):
        btc_h1 = _to_df(_drl3__get_klines({"1H": btc_klines} if isinstance(btc_klines, list) else btc_klines, "1H"))
        if len(btc_h1) >= 24:
            btc_corr = _calc_correlation(h1["c"].tail(24).pct_change().dropna(),
                                         btc_h1["c"].tail(24).pct_change().dropna())

    # ── 3m 回调企稳（含时效性）──
    # v4.0: 买卖价差过滤
    if bool(config.get('enable_bidask_spread_filter', True)):
        ob = (extra or {}).get('order_book')
        if ob and isinstance(ob, dict):
            bid = _safe_float(ob.get('bid_px') or (ob.get('bids', [[0]])[0][0] if ob.get('bids') else 0), 0)
            ask = _safe_float(ob.get('ask_px') or (ob.get('asks', [[0]])[0][0] if ob.get('asks') else 0), 0)
            if bid > 0 and ask > 0 and (ask / bid - 1) * 100 > _cfg_float(config, 'max_bidask_spread_pct', 0.25):
                return {"valid": False, "symbol": inst_id, "reason": "买卖价差过大，流动性不足"}

    micro_confirm = _micro_pullback_continuation(m3, trend_alignment, config)
    if bool(config.get("require_m3_pullback_confirmation", True)) and not micro_confirm["confirmed"]:
        return {"valid": False, "symbol": inst_id,
                "reason": f"3分钟回调续势未确认: {micro_confirm['reason']}"}

    if funding_risk:
        return {"valid": False, "symbol": inst_id,
                "reason": "接近资金费率结算时间，回避开仓"}

    rsi_1h = _rsi(h1["c"], 14)
    adx_1h = _adx_like(h1, 14)
    adx_4h = _adx_like(h4, 14) if len(h4) >= 25 else adx_1h * 0.7
    atr_pct = _drl3__atr_pct(h1, 14)
    realized_vol = _drl3__realized_vol_pct(h1["c"], 24)
    volume_impulse = _drl3__volume_impulse(h1, 24)

    # v2: 突破压强跟随趋势方向
    breakout_bps = _breakout_pressure_bps(h1, 24, trend_alignment)

    conv_m15 = _conv_feature(m15["c"] if not m15.empty else h1["c"], [1.0, 0.0, -1.0])
    conv_h1 = _conv_feature(h1["c"], [1.0, 0.0, -1.0])
    conv_h4 = _conv_feature(h4["c"] if len(h4) > 5 else h1["c"], [1.0, -2.0, 1.0])
    cnn_multi_tf = _clamp(conv_m15 * 0.35 + conv_h1 * 0.45 + conv_h4 * 0.20, -2.0, 2.0)

    whale_flow = _safe_float(on_chain.get("whale_flow"), 0.0)
    exchange_netflow = _safe_float(on_chain.get("exchange_netflow"), 0.0)
    active_addr_z = _safe_float(on_chain.get("active_addresses_z"), 0.0)
    llm_sent = _drl3__blend([
        _safe_float(llm.get("sentiment"), np.nan),
        _safe_float(llm.get("narrative_strength"), np.nan),
        _safe_float(social.get("sentiment"), np.nan),
        _safe_float(extra.get("news_sentiment"), np.nan),
    ], [0.35, 0.30, 0.25, 0.10])

    factors = {
        "momentum_1h": _clamp(h1_mom / 180.0, -2.5, 2.5),
        "momentum_4h": _clamp(h4_mom / 280.0, -2.5, 2.5),
        "momentum_1d": _clamp(d1_mom / 380.0, -2.5, 2.5),
        "trend_alignment": trend_alignment,
        "m3_pullback_score": micro_confirm["score"],
        "breakout": _clamp(breakout_bps / 120.0, -2.5, 2.5),
        "volume_impulse": _clamp(volume_impulse - 1.0, -1.5, 2.5),
        "adx_strength": _clamp((adx_1h - 18.0) / 18.0, -1.5, 2.0),
        "funding_contra": _clamp(-funding_rate / 0.08, -2.0, 2.0),
        "oi_confirmation": _clamp(oi_change / 8.0, -2.0, 2.0) * _drl3__direction_sign(h1_mom, trend_alignment),
        "cnn_multi_tf": cnn_multi_tf,
        "on_chain_accumulation": _clamp(whale_flow - exchange_netflow * 0.65 + active_addr_z * 0.25, -2.0, 2.0),
        "llm_sentiment": _clamp(llm_sent, -2.0, 2.0),
        "risk_vol": _clamp((realized_vol - 6.0) / 4.0, -1.5, 2.5),
        "risk_atr": _clamp((atr_pct - 3.0) / 3.0, -1.5, 2.5),
        "liquidity": _clamp((log10(max(volume_24h, 1.0)) - 6.2) / 2.0, -1.0, 2.0),
        # v2 新增因子
        "trend_freshness": _clamp(1.0 - h1_trend_age / max(_cfg_float(config, "max_h1_trend_age", 12.0), 1.0), -1.0, 1.0),
        "m3_timeliness": micro_confirm.get("timeliness_score", 0.0),
    }

    meta_feedback = extra.get("strategy_feedback", {}) if isinstance(extra.get("strategy_feedback"), dict) else {}
    regime = _detect_regime(trend_alignment, adx_1h, realized_vol, h1_mom)

    return {
        "valid": True, "reason": "", "symbol": inst_id,
        "last_price": price, "volume_24h": volume_24h, "price_change_24h": change_24h,
        "funding_rate_pct": funding_rate, "open_interest_change_pct": oi_change,
        "atr_pct": atr_pct, "realized_vol_pct": realized_vol,
        "rsi_1h": rsi_1h, "adx_1h": adx_1h, "adx_4h": adx_4h,
        "m15_mom_bps": m15_mom,
        "m3_pullback_confirmed": micro_confirm["confirmed"],
        "m3_structure_state": micro_confirm["state"],
        "m3_pullback_reason": micro_confirm["reason"],
        "m3_pullback_pct": micro_confirm["pullback_pct"],
        "m3_impulse_pct": micro_confirm["impulse_pct"],
        "m3_staleness_bars": micro_confirm.get("staleness_bars", 0),
        "h1_trend_age": h1_trend_age,
        "h1_mom_bps": h1_mom, "h4_mom_bps": h4_mom, "d1_mom_bps": d1_mom,
        "breakout_bps": breakout_bps, "volume_impulse": volume_impulse,
        "funding_risk": funding_risk, "btc_correlation": btc_corr,
        "regime": regime, "meta_feedback": meta_feedback, "factors": factors,
    }


# ══════════════════════════════════════════════
# 评分（含时效性惩罚）
# ══════════════════════════════════════════════

def _score_snapshot(snap: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    if not snap.get("valid", True) or "factors" not in snap:
        return {"passed": False, "reason": snap.get("reason", "快照无效")}
    factors = snap["factors"]
    meta_weights = _meta_regime_weights(snap["regime"], snap["meta_feedback"], config)

    # Dueling DQN: V(s) + A(s,a)
    state_value = _clamp(
        0.24 * factors["liquidity"]
        + 0.22 * factors["trend_alignment"]
        + 0.12 * factors["volume_impulse"]
        - 0.20 * factors["risk_vol"]
        - 0.18 * factors["risk_atr"],
        -2.0, 2.0,
    )
    long_adv = _clamp(
        meta_weights["trend"] * factors["trend_alignment"]
        + meta_weights["momentum"] * factors["momentum_1h"]
        + 0.20 * factors["breakout"]
        + 0.12 * factors["cnn_multi_tf"]
        + 0.10 * factors["oi_confirmation"]
        + 0.08 * factors["on_chain_accumulation"]
        + 0.06 * factors["llm_sentiment"]
        + 0.04 * factors["trend_freshness"]     # v2
        + 0.03 * factors["m3_timeliness"]       # v2
        - 0.10 * max(factors["risk_vol"], 0.0),
        -3.0, 3.0,
    )
    short_adv = _clamp(
        meta_weights["trend"] * (-factors["trend_alignment"])
        + meta_weights["momentum"] * (-factors["momentum_1h"])
        + 0.20 * (-factors["breakout"])
        + 0.12 * (-factors["cnn_multi_tf"])
        + 0.10 * (-factors["oi_confirmation"])
        + 0.06 * (-factors["llm_sentiment"])
        + 0.04 * (-factors["trend_freshness"])  # v2: 空头时反向
        + 0.05 * max(factors["risk_vol"], 0.0),
        -3.0, 3.0,
    )
    wait_adv = _clamp(
        0.30 * (factors["risk_vol"] + factors["risk_atr"])
        - 0.18 * abs(factors["trend_alignment"])
        + 0.08 * max(-factors["trend_freshness"], 0.0)  # v2: 趋势过老 → 更倾向 WAIT
        + 0.06 * max(-factors["m3_timeliness"], 0.0),   # v2: 3m回调过旧 → 更倾向 WAIT
        -2.5, 2.5,
    )

    mean_adv = (long_adv + short_adv + wait_adv) / 3.0
    q_long_online  = state_value + (long_adv - mean_adv)
    q_short_online = state_value + (short_adv - mean_adv)
    q_wait_online  = state_value + (wait_adv - mean_adv)

    target_scale = _cfg_float(config, "double_q_blend", 0.36)
    q_long_target  = _clamp(0.62 * factors["momentum_4h"] + 0.38 * factors["trend_alignment"], -2.8, 2.8)
    q_short_target = _clamp(-0.62 * factors["momentum_4h"] - 0.38 * factors["trend_alignment"], -2.8, 2.8)
    q_wait_target  = _clamp(0.50 * (factors["risk_vol"] + factors["risk_atr"]) - 0.12 * abs(factors["momentum_4h"]), -2.8, 2.8)
    q_long  = (1.0 - target_scale) * q_long_online  + target_scale * q_long_target
    q_short = (1.0 - target_scale) * q_short_online + target_scale * q_short_target
    q_wait  = (1.0 - target_scale) * q_wait_online  + target_scale * q_wait_target

    adv_long  = q_long  - state_value
    adv_short = q_short - state_value
    adv_wait  = q_wait  - state_value

    probs = _softmax([q_long, q_short, q_wait], _cfg_float(config, "q_temperature", 0.85))
    entropy = _normalized_entropy(probs)
    sac_alpha = _cfg_float(config, "entropy_alpha", 0.18)
    objectives = {
        "BUY":  q_long  + sac_alpha * entropy,
        "SELL": q_short + sac_alpha * entropy,
        "WAIT": q_wait  + sac_alpha * entropy,
    }
    direction = max(objectives, key=objectives.get)
    best = float(objectives[direction])
    second = sorted(objectives.values(), reverse=True)[1]
    confidence = _clamp(best - second, 0.0, 3.0)

    if direction == "SELL" and not bool(config.get("allow_short", True)):
        direction = "WAIT"

    startup_gate = _hourly_startup_gate(snap, direction, config)
    risk_penalty = _cfg_float(config, "risk_penalty_strength", 0.70) * max(0.0, factors["risk_vol"] + factors["risk_atr"] * 0.8)
    raw_edge = best - objectives["WAIT"] if direction in {"BUY", "SELL"} else 0.0
    edge = _clamp(raw_edge * (0.70 + 0.30 * confidence) - risk_penalty * 0.20, -2.8, 2.8)

    # v2: 时效性惩罚写入 score
    base_score = _drl3__edge_to_score(edge, confidence, startup_gate, snap)
    age_penalty = _age_penalty(snap, config, direction)
    score = _clamp(base_score - age_penalty, 0.0, 100.0)

    passed = (
        direction in {"BUY", "SELL"}
        and startup_gate
        and score >= _cfg_float(config, "min_score", 74.0)
        and abs(edge) >= _cfg_float(config, "min_abs_edge", 0.22)
        and snap["atr_pct"] <= _cfg_float(config, "max_atr_pct", 8.2)
    )

    # 时效性状态文字
    age = snap.get("h1_trend_age", 0)
    max_age = int(_cfg_float(config, "max_h1_trend_age", 12.0))
    stale = snap.get("m3_staleness_bars", 0)
    max_stale = int(_cfg_float(config, "max_m3_staleness_bars", 15))
    freshness_label = (
        f"1H趋势{age}根({'⚠过老' if age > max_age else '✓'}), 3m回调{stale}根前({'⚠过旧' if stale > max_stale else '✓'})"
    )

    category = "DRL小时趋势多头启动" if direction == "BUY" else "DRL小时趋势空头启动" if direction == "SELL" else "DRL小时趋势观察"
    signals = [
        f"{category} 评分 {score:.1f} (惩罚{age_penalty:.1f})",
        f"Q(L/S/W)=({q_long:+.2f},{q_short:+.2f},{q_wait:+.2f}) | A2C优势({adv_long:+.2f},{adv_short:+.2f},{adv_wait:+.2f})",
        f"SAC熵={entropy:.2f} 置信度={confidence:.2f} 边际={edge:+.3f}",
        f"1H动量={snap['h1_mom_bps']:+.1f}bps 4H动量={snap['h4_mom_bps']:+.1f}bps 突破压强={snap['breakout_bps']:+.1f}bps",
        f"量能脉冲={snap['volume_impulse']:.2f}x ADX1H={snap['adx_1h']:.1f} ATR%={snap['atr_pct']:.2f} Regime={snap['regime']}",
        f"时效性: {freshness_label}",
    ]
    ranking_factors = {
        "trend": _clamp(50 + factors["trend_alignment"] * 24 + factors["momentum_4h"] * 8, 0, 100),
        "trigger": _clamp(50 + factors["breakout"] * 18 + factors["momentum_1h"] * 18 + confidence * 10, 0, 100),
        "volume": _clamp(50 + factors["volume_impulse"] * 26 + factors["liquidity"] * 12, 0, 100),
        "location": _clamp(55 + (50.0 - abs(snap["rsi_1h"] - 55.0)) * 0.6, 10, 95),
        "freshness": _clamp(
            50 + snap["m15_mom_bps"] / 6.0 + snap["h1_mom_bps"] / 10.0
            - max(age - max_age, 0) * 2.5        # v2: 趋势过老降低 freshness
            - max(stale - max_stale, 0) * 1.5,   # v2: 回调过旧降低 freshness
            0, 95,
        ),
        "risk": _clamp(84 - max(factors["risk_vol"], 0.0) * 18 - max(factors["risk_atr"], 0.0) * 15, 10, 95),
    }
    result = {
        "symbol": snap["symbol"], "passed": passed,
        "score": round(float(score), 2), "direction": direction if direction in {"BUY", "SELL"} else "WAIT",
        "signals": signals, "category": category, "strategy_category": category,
        "last_price": snap["last_price"], "volume_24h": snap["volume_24h"],
        "price_change_24h": snap["price_change_24h"], "ranking_factors": ranking_factors,
        "metrics": {
            "alpha_edge": round(float(edge), 6), "raw_edge": round(float(raw_edge), 6),
            "confidence": round(float(confidence), 6), "entropy": round(float(entropy), 6),
            "q_long": round(float(q_long), 6), "q_short": round(float(q_short), 6), "q_wait": round(float(q_wait), 6),
            "state_value": round(float(state_value), 6),
            "adv_long": round(float(adv_long), 6), "adv_short": round(float(adv_short), 6),
            "atr_pct": round(float(snap["atr_pct"]), 6), "realized_vol_pct": round(float(snap["realized_vol_pct"]), 6),
            "regime": snap["regime"], "h1_trend_age": age, "m3_staleness_bars": stale,
            "age_penalty": round(float(age_penalty), 2),
        },
        "factor_scores": {k: round(float(v), 6) for k, v in factors.items()},
        "details": {
            "机会类型": category, "评估": " | ".join(signals),
            "综合边际": f"{edge:+.3f}", "置信度": f"{confidence:.2f}", "熵": f"{entropy:.2f}",
            "1H动量bps": f"{snap['h1_mom_bps']:+.1f}", "突破压强bps": f"{snap['breakout_bps']:+.1f}",
            "ADX1H": f"{snap['adx_1h']:.1f}", "ATR%": f"{snap['atr_pct']:.2f}",
            "资金费率%": f"{snap['funding_rate_pct']:+.4f}",
            "OI变化%": f"{snap['open_interest_change_pct']:+.2f}",
            "启动门槛通过": "是" if startup_gate else "否",
            "3分钟回调确认": "是" if snap.get("m3_pullback_confirmed") else "否",
            "3分钟结构": str(snap.get("m3_structure_state", "-")),
            "3分钟回调幅度%": f"{float(snap.get('m3_pullback_pct', 0.0)):.2f}",
            "3分钟原趋势脉冲%": f"{float(snap.get('m3_impulse_pct', 0.0)):.2f}",
            "3分钟回调时效": f"{stale}根前({'超时' if stale > max_stale else '新鲜'})",
            "1H趋势延续根数": f"{age}根({'过老' if age > max_age else '正常'})",
            "时效性惩罚": f"-{age_penalty:.1f}分",
        },
    }
    if build_opportunity_profile:
        try: result.update(build_opportunity_profile(score, result["direction"], snap["volume_24h"], ranking_factors, signals))
        except Exception: pass
    return result


def _age_penalty(snap: Dict[str, Any], config: Dict[str, Any], direction: str) -> float:
    """
    综合时效性惩罚分。
    1. 1H 趋势延续过长 → 越界越高
    2. 3m 回调点位过旧 → 越界越高
    两者都过期则叠加（但不超过 max_penalty = 15 分）。
    """
    if direction not in {"BUY", "SELL"}:
        return 0.0
    penalty = 0.0
    age = snap.get("h1_trend_age", 0)
    max_age = int(_cfg_float(config, "max_h1_trend_age", 12.0))
    age_pen_str = _cfg_float(config, "h1_trend_age_penalty", 8.0)
    if age > max_age:
        # 超出越多惩罚越重，但有上限
        overflow = age - max_age
        penalty += min(age_pen_str, age_pen_str * (1 - exp(-overflow / max(max_age * 0.5, 1.0))))

    stale = snap.get("m3_staleness_bars", 0)
    max_stale = int(_cfg_float(config, "max_m3_staleness_bars", 15))
    fresh_pen_str = _cfg_float(config, "m3_freshness_penalty", 6.0)
    if stale > max_stale:
        overflow = stale - max_stale
        penalty += min(fresh_pen_str, fresh_pen_str * (1 - exp(-overflow / max(max_stale * 0.5, 1.0))))

    return _clamp(penalty, 0.0, 15.0)


# ══════════════════════════════════════════════
# _hourly_startup_gate（保留 + breakout_bps 修正方向语义）
# ══════════════════════════════════════════════

def _hourly_startup_gate(snap: Dict[str, Any], direction: str, config: Dict[str, Any]) -> bool:
    h1_mom_min = _cfg_float(config, "hourly_start_momentum_bps", 45.0)
    breakout_min = _cfg_float(config, "hourly_start_breakout_bps", 20.0)
    vol_min = _cfg_float(config, "hourly_start_volume_impulse", 1.18)
    if direction == "BUY":
        return (
            snap["h1_mom_bps"] >= h1_mom_min
            and snap["breakout_bps"] >= breakout_min
            and snap["volume_impulse"] >= vol_min
            and snap["adx_1h"] >= 17.0
        )
    if direction == "SELL":
        return (
            snap["h1_mom_bps"] <= -h1_mom_min
            and snap["breakout_bps"] <= -breakout_min
            and snap["volume_impulse"] >= vol_min
            and snap["adx_1h"] >= 17.0
        )
    return False


# ══════════════════════════════════════════════
# 元学习权重 / Regime 检测
# ══════════════════════════════════════════════

def _meta_regime_weights(regime, feedback, config):
    base = {"trend": 0.42, "momentum": 0.38, "reversal": 0.20}
    if regime == "TREND": base = {"trend": 0.48, "momentum": 0.40, "reversal": 0.12}
    elif regime == "RANGE": base = {"trend": 0.30, "momentum": 0.24, "reversal": 0.46}
    elif regime == "VOLATILE": base = {"trend": 0.36, "momentum": 0.28, "reversal": 0.36}
    trend_wr = _safe_float(feedback.get("trend_win_rate"), 0.52)
    range_wr = _safe_float(feedback.get("range_win_rate"), 0.48)
    adapt = _cfg_float(config, "meta_adapt_strength", 0.40)
    delta = _clamp((trend_wr - range_wr) * 2.0, -0.30, 0.30) * adapt
    base["trend"] = max(0.05, base["trend"] + delta)
    base["momentum"] = max(0.05, base["momentum"] + delta * 0.7)
    base["reversal"] = max(0.05, base["reversal"] - delta * 1.7)
    total = base["trend"] + base["momentum"] + base["reversal"]
    return {k: v / total for k, v in base.items()}

def _detect_regime(trend_alignment, adx_1h, realized_vol, h1_mom_bps):
    if adx_1h >= 22.0 and abs(trend_alignment) >= 0.45 and abs(h1_mom_bps) >= 35.0: return "TREND"
    if realized_vol >= 8.0: return "VOLATILE"
    return "RANGE"


# ══════════════════════════════════════════════
# 底层工具
# ══════════════════════════════════════════════

def _drl3__edge_to_score(edge, confidence, startup_gate, snap):
    base = 34.0 + abs(edge) * 11.5 + confidence * 4.5
    quality = (
        max(0.0, snap["volume_impulse"] - 1.0) * 3.0
        + max(0.0, (snap["adx_1h"] - 16.0) / 10.0) * 2.8
        + max(0.0, 1.0 - snap["atr_pct"] / 10.0) * 2.5
    )
    gate_bonus = 1.5 if startup_gate else -9.0
    return _clamp(base + quality + gate_bonus, 0.0, 100.0)

def _drl3__result_sort_key(item):
    return (float(item.get("opportunity_score", item.get("score", 0.0)) or 0.0),
            float(item.get("score", 0.0) or 0.0),
            float(item.get("volume_24h", 0.0) or 0.0))

def _ema_trend_score(close, fast, slow):
    if len(close) < slow + 3: return 0.0
    ef = close.ewm(span=fast, adjust=False).mean()
    es = close.ewm(span=slow, adjust=False).mean()
    slope = float(ef.diff().tail(3).mean() or 0.0)
    spread = (float(ef.iloc[-1]) / max(float(es.iloc[-1]), 1e-9) - 1.0) * 100.0
    score = spread * 2.2 + slope / max(float(close.iloc[-1]), 1e-9) * 10_000 * 0.8
    return _clamp(score / 30.0, -2.0, 2.0)

def _drl3__get_klines(klines_map, bar):
    aliases = {"15m":["15m","15M"],"1H":["1H","1h","60m","60M"],
               "4H":["4H","4h","240m","240M"],"1D":["1D","1d","D","day"]}
    for key in aliases.get(bar, [bar, bar.lower(), bar.upper()]):
        if key in klines_map and klines_map.get(key): return klines_map.get(key)
    return []

def _drl3__atr_pct(df, period=14):
    if len(df) < period + 2: return 0.0
    pc = df["c"].shift(1)
    tr = pd.concat([(df["h"]-df["l"]).abs(),(df["h"]-pc).abs(),(df["l"]-pc).abs()],axis=1).max(axis=1)
    atr = float(tr.ewm(alpha=1/period, adjust=False).mean().iloc[-1] or 0.0)
    p = float(df["c"].iloc[-1] or 0.0); return atr / p * 100.0 if p > 0 else 0.0

def _adx_like(df, period=14):
    if len(df) < period + 5: return 15.0
    h, l, c = df["h"], df["l"], df["c"]
    um = h.diff(); dm = -l.diff()
    pdm = np.where((um>dm)&(um>0), um, 0.0); mdm = np.where((dm>um)&(dm>0), dm, 0.0)
    pc = c.shift(1)
    tr = pd.concat([(h-l).abs(),(h-pc).abs(),(l-pc).abs()],axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False).mean().replace(0, np.nan)
    pdi = 100*pd.Series(pdm,index=df.index).ewm(alpha=1/period,adjust=False).mean()/atr
    mdi = 100*pd.Series(mdm,index=df.index).ewm(alpha=1/period,adjust=False).mean()/atr
    dx = (pdi-mdi).abs()/(pdi+mdi).replace(0,np.nan)*100
    adx = dx.ewm(alpha=1/period, adjust=False).mean()
    v = float(adx.iloc[-1]) if np.isfinite(adx.iloc[-1]) else 15.0
    return _clamp(v, 5.0, 60.0)

def _drl3__realized_vol_pct(close, window=24):
    ret = close.pct_change().dropna().tail(window)
    if ret.empty: return 0.0
    return float(ret.std(ddof=0) * sqrt(max(len(ret), 1)) * 100.0)

def _drl3__volume_impulse(df, window=24):
    if len(df) < window + 3: return 1.0
    base = float(df["vol"].iloc[-(window+1):-1].median() or 0.0)
    latest = float(df["vol"].tail(3).mean() or 0.0)
    return latest / base if base > 0 else 1.0

def _breakout_pressure_bps(df, window=24, trend_hint=0.0):
    """v2: 跟随趋势方向选择突破方向，避免反向假突破干扰。"""
    if len(df) < window + 2: return 0.0
    high_ref = float(df["h"].iloc[-(window+1):-1].max() or 0.0)
    low_ref  = float(df["l"].iloc[-(window+1):-1].min() or 0.0)
    close = float(df["c"].iloc[-1] or 0.0)
    if close <= 0: return 0.0
    up   = (close / max(high_ref, 1e-9) - 1.0) * 10_000
    down = (close / max(low_ref,  1e-9) - 1.0) * 10_000
    # v2: 趋势方向优先；若方向不明则取绝对值更大者
    if trend_hint > 0.15:
        return _clamp(up, -600.0, 600.0)
    elif trend_hint < -0.15:
        return _clamp(down, -600.0, 600.0)
    else:
        return _clamp(up if abs(up) >= abs(down) else down, -600.0, 600.0)

def _conv_feature(series, kernel):
    vals = pd.to_numeric(series, errors="coerce").dropna().values.astype(float)
    k = np.array(kernel, dtype=float)
    if len(vals) < len(k) + 2: return 0.0
    norm = np.maximum(np.abs(vals[-len(k):]).mean(), 1e-9)
    return _clamp(float(np.dot(vals[-len(k):], k) / norm), -3.0, 3.0)

def _softmax(values, temperature):
    t = max(float(temperature), 1e-6)
    arr = np.array(values, dtype=float) / t; arr -= arr.max()
    exps = np.exp(arr); return (exps / float(np.sum(exps) or 1.0)).tolist()

def _normalized_entropy(probs):
    eps = 1e-12; n = max(len(probs), 1)
    ent = -sum(float(p) * np.log(float(p) + eps) for p in probs)
    return _clamp(ent / np.log(n + eps), 0.0, 1.0)

def _drl3__symbol_from_backtest_data(data, config):
    km = data.get("klines_map") or {}
    h1 = _to_df(_drl3__get_klines(km,"1H") or _drl3__get_klines(km,"15m") or data.get("klines") or [])
    lp = float(h1["c"].iloc[-1]) if not h1.empty else 0.0
    vol = float((h1["c"] * h1["vol"]).tail(48).sum()) if not h1.empty else 0.0
    extra = {"klines": km, "funding_rate": data.get("funding_rate", 0.0),
             "open_interest_change_pct": data.get("open_interest_change_pct", 0.0),
             "on_chain": data.get("on_chain", {}), "social": data.get("social", {}),
             "llm_factors": data.get("llm_factors", {}),
             "news_sentiment": data.get("news_sentiment", 0.0),
             "strategy_feedback": data.get("strategy_feedback", {})}
    return _drl3__MinimalSymbol(inst_id=str(config.get("inst_id", "BACKTEST") or "BACKTEST"),
        last_price=lp, volume_24h=vol,
        price_change_24h=_pct_change(h1["c"], min(len(h1)-1, 24))*100 if not h1.empty else 0.0,
        extra_data=extra)

class _drl3__MinimalSymbol:
    def __init__(self, inst_id, last_price, volume_24h, price_change_24h, extra_data):
        self.inst_id = inst_id; self.last_price = last_price
        self.volume_24h = volume_24h; self.price_change_24h = price_change_24h
        self.high_24h = 0.0; self.low_24h = 0.0; self.open_interest = 0.0
        self.extra_data = extra_data

def _drl3__failed_result(symbol, reason):
    return {"symbol": str(getattr(symbol,"inst_id","") or ""), "passed": False,
            "score": 0.0, "direction": "WAIT", "signals": [], "details": {"状态": reason}, "metrics": {}}

def _drl3__blend(values, weights):
    total = denom = 0.0
    for v, w in zip(values, weights):
        if np.isfinite(v): total += float(v)*float(w); denom += float(w)
    return total/denom if denom > 0 else 0.0

def _drl3__direction_sign(primary, fallback):
    if abs(float(primary or 0.0)) > 1e-9: return 1.0 if primary > 0 else -1.0
    if abs(float(fallback or 0.0)) > 1e-9: return 1.0 if fallback > 0 else -1.0
    return 1.0

# ── v2.1 新增工具 ──

def _check_funding_timing(config: Dict[str, Any]) -> bool:
    """
    检查当前时间是否接近资金费率结算时间。
    资金费率通常在 00:00, 08:00, 16:00 UTC 结算。
    在结算前 N 分钟内避免开仓以规避费率波动。
    """
    try:
        from datetime import datetime, timezone
        avoid_minutes = int(_cfg_float(config, "funding_avoid_minutes", 15.0))
        now_utc = datetime.now(timezone.utc)
        # OKX 标准资金费率结算时间: 00:00, 08:00, 16:00 UTC
        funding_hours = [0, 8, 16]
        for h in funding_hours:
            settlement = now_utc.replace(hour=h, minute=0, second=0, microsecond=0)
            delta_min = (settlement - now_utc).total_seconds() / 60.0
            if 0 <= delta_min <= avoid_minutes:
                return True  # 进入回避窗口
    except Exception:
        pass
    return False

def _calc_correlation(series_a: pd.Series, series_b: pd.Series) -> float:
    """
    计算两个序列的皮尔逊相关系数。
    用于检测与 BTC 的高度相关性。
    """
    try:
        n = min(len(series_a), len(series_b))
        if n < 8:
            return 0.0
        corr = series_a.tail(n).corr(series_b.tail(n))
        return float(corr) if np.isfinite(corr) else 0.0
    except Exception:
        return 0.0

# ════════════════════════════════════════════════════════════════════════════
# 子策略 4 — XGBoost截面排序策略
# ════════════════════════════════════════════════════════════════════════════

logger = logging.getLogger(__name__)

try:
    import xgboost as xgb
    _HAS_XGB = True
except Exception:
    xgb = None
    _HAS_XGB = False

try:
    from src.scanner.base_scanner import BaseScannerStrategy, ScanCondition
    from src.scanner.ranking import build_opportunity_profile
    _HAS_BASE = True
except ImportError:
    BaseScannerStrategy = object; ScanCondition = None; build_opportunity_profile = None
    _HAS_BASE = False


_xgb4_CONFIG_SCHEMA = {
    "min_score":              {"type":"float","default":72.0,     "label":"最低扫描分数"},
    "backtest_min_score":     {"type":"float","default":68.0,     "label":"回测最低分数"},
    "min_volume_24h":         {"type":"float","default":5_000_000,"label":"最小24H成交额"},
    "top_n":                  {"type":"int",  "default":15,       "label":"最多输出数量"},
    "allow_short":            {"type":"bool", "default":True,     "label":"允许空头"},
    "max_atr_pct":            {"type":"float","default":8.0,      "label":"最大ATR%"},
    "position_size":          {"type":"float","default":0.10,     "label":"仓位比例"},
    # ── XGBoost训练 ──
    "xgb_max_depth":          {"type":"int",  "default":5,        "label":"树最大深度"},
    "xgb_n_estimators":       {"type":"int",  "default":120,      "label":"树数量"},
    "xgb_learning_rate":      {"type":"float","default":0.05,     "label":"学习率"},
    "xgb_subsample":          {"type":"float","default":0.75,     "label":"样本采样比例"},
    "xgb_colsample_bytree":   {"type":"float","default":0.75,     "label":"特征采样比例"},
    "xgb_reg_alpha":          {"type":"float","default":1.0,      "label":"L1正则化"},
    "xgb_reg_lambda":         {"type":"float","default":2.0,      "label":"L2正则化"},
    "xgb_min_child_weight":   {"type":"int",  "default":3,        "label":"最小叶子权重"},
    # ── 在线学习 ──
    "xgb_retrain_hours":      {"type":"int",  "default":24,       "label":"重新训练间隔(小时)"},
    "xgb_min_samples":        {"type":"int",  "default":500,      "label":"最少训练样本数"},
    "xgb_fallback_weight":    {"type":"float","default":0.30,     "label":"未训练时线性权重混合"},
    "xgb_trained_weight":     {"type":"float","default":0.75,     "label":"已训练时模型权重混合"},
    "xgb_label_delay_bars":   {"type":"int",  "default":24,       "label":"标签延迟(1H根数,即实现收益窗口)"},
    # ── 回调企稳+趋势时效 ──
    "require_m3_pullback":    {"type":"bool","default":True,      "label":"要求3分钟回调企稳续势"},
    "m3_pullback_min_pct":    {"type":"float","default":0.50,     "label":"3分钟最小回调幅度%"},
    "m3_pullback_max_pct":    {"type":"float","default":2.20,     "label":"3分钟最大回调幅度%"},
    "m3_stabilization_bars":  {"type":"int","default":4,          "label":"3分钟企稳确认根数"},
    "m3_min_impulse_pct":     {"type":"float","default":0.65,     "label":"3m最小原趋势脉冲%"},
    "max_h1_trend_age":       {"type":"int","default":12,         "label":"1H趋势最大延续根数"},
}

_DEFAULT = {k: v["default"] for k, v in _xgb4_CONFIG_SCHEMA.items()}

_FACTOR_NAMES = [
    "momentum_1h", "momentum_4h", "momentum_1d", "short_reversal",
    "trend_quality", "low_volatility", "liquidity", "volume_impulse",
    "funding_carry", "oi_heat", "macd_momentum", "bb_percentb",
    "vol_zscore", "close_strength", "efficiency_ratio", "rsi_alignment",
    "momentum_decay", "momentum_acceleration", "early_trend_trigger",
]


class XGBoostCrossSectionalRanker(BaseScannerStrategy if _HAS_BASE else object):
    required_bars = ["3m", "15m", "1H", "4H", "1D"]
    requires_derivative_metrics = True
    name = "XGBoost截面排序策略"
    description = "非线性: XGBoost Ranker 替代线性因子加权，自动发现交互效应，截面 z-score 归一化"
    strategy_type = "scan"

    def __init__(self, config=None):
        self.config = {**_DEFAULT, **(config or {})}
        self._model: Optional[object] = None
        self._last_train_time: float = 0.0
        # v2: 样本含 entry_price + entry_time，用于延迟标签
        self._pending_samples: List[Dict] = []   # 未产生标签的样本
        self._labeled_samples: List[Dict] = []   # 已有真实标签的样本
        self._linear_weights = _default_linear_weights()
        if _HAS_BASE and hasattr(super(), "__init__"):
            try: super().__init__(self.config)
            except Exception: pass

    def _init_conditions(self):
        if ScanCondition is None: return
        self.add_condition(ScanCondition(name="24H成交额", field="volume_24h",
            operator=">=", value=self.config.get("min_volume_24h", 5_000_000)))

    def get_config_schema(self): return dict(_xgb4_CONFIG_SCHEMA)

    # ── 单标的评分（不走截面 z-score，用于 scan_symbol 模式）──────────────
    def scan_symbol(self, symbol):
        snap = _xgb4__build_snapshot(symbol, self.config)
        if not snap["valid"]: return _xgb4__failed(symbol, snap["reason"])
        # 非截面模式：对因子做自归一化后线性评分
        f = snap["factors"]
        linear_score = _linear_score_from_factors(f, self._linear_weights)
        xgb_score = linear_score
        if self._model is not None and _HAS_XGB:
            try:
                X = pd.DataFrame([{k: float(f.get(k, 0)) for k in _FACTOR_NAMES}])
                pred = float(self._model.predict(X)[0])
                xgb_score = _clamp(pred * 15 + 50, 0, 100)
            except Exception:
                pass
        blend = float(self.config.get("xgb_trained_weight", 0.75)) if self._model else float(self.config.get("xgb_fallback_weight", 0.30))
        score = linear_score * (1 - blend) + xgb_score * blend
        if snap["atr_pct"] > float(self.config.get("max_atr_pct", 8.0)):
            score -= min(20, (snap["atr_pct"] - float(self.config.get("max_atr_pct", 8.0))) * 2.5)
        score = _clamp(score, 0, 100)

        direction = "BUY" if snap["momentum_4h"] > 0 else "SELL"
        if not bool(self.config.get("allow_short", True)) and direction == "SELL":
            direction = "WAIT"

        passed = score >= float(self.config.get("min_score", 72))
        # 积累待标注样本
        self._record_pending_sample(snap)
        return _xgb4__result(snap, score, direction, f, passed, self.config)

    # ── 批量截面扫描（核心：截面 z-score + XGBoost）────────────────────────
    def scan_all_symbols(self, symbols):
        # 1. 更新延迟标签
        self._update_labels(symbols)
        # 2. 触发训练
        self._maybe_train()

        min_vol = float(self.config.get("min_volume_24h", 5_000_000))
        snaps = []
        for sym in symbols:
            if float(getattr(sym, "volume_24h", 0) or 0) < min_vol: continue
            snap = _xgb4__build_snapshot(sym, self.config)
            if snap["valid"]:
                snaps.append(snap)
                self._record_pending_sample(snap)

        if not snaps:
            return {"type":"xgboost_cross_section_ranker","all_opportunities":[],
                    "model_trained":self._model is not None,"training_samples":len(self._labeled_samples)}

        # 3. 截面因子矩阵 z-score 归一化
        factor_frame = pd.DataFrame([s["factors"] for s in snaps], index=[s["symbol"] for s in snaps])
        z = factor_frame.apply(_robust_zscore, axis=0).fillna(0.0)

        # 4. 截面 edge 评分
        results = []
        for snap in snaps:
            sym_name = snap["symbol"]
            edge = _cross_section_edge(z.loc[sym_name].to_dict(), self._linear_weights, self._model)
            score = _clamp(edge * 16 + 58, 0, 100)  # edge∈[-3,3] → score∈[10,106] → clip
            if snap["atr_pct"] > float(self.config.get("max_atr_pct", 8.0)):
                score -= min(20, (snap["atr_pct"] - float(self.config.get("max_atr_pct", 8.0))) * 2.5)
            score = _clamp(score, 0, 100)

            direction = "BUY" if snap["momentum_4h"] > 0 else "SELL"
            if not bool(self.config.get("allow_short", True)) and direction == "SELL":
                direction = "WAIT"

            passed = score >= float(self.config.get("min_score", 72))
            if passed:
                results.append(_xgb4__result(snap, score, direction, z.loc[sym_name].to_dict(), passed, self.config))

        results.sort(key=lambda x: float(x.get("opportunity_score", x.get("score", 0)) or 0), reverse=True)
        top_n = int(self.config.get("top_n", 15))
        return {
            "type": "xgboost_cross_section_ranker",
            "all_opportunities": results[:top_n],
            "model_trained": self._model is not None,
            "training_samples": len(self._labeled_samples),
            "pending_labels": len(self._pending_samples),
        }

    # ── 延迟标签机制 ────────────────────────────────────────────────────────
    def _record_pending_sample(self, snap):
        """记录待标注样本（含入场价和时间）。"""
        if len(self._pending_samples) + len(self._labeled_samples) >= 5000:
            return
        self._pending_samples.append({
            "symbol": snap["symbol"],
            "factors": snap["factors"].copy(),
            "entry_price": snap["last_price"],
            "entry_time": time.time(),
            # 弱标签兜底：在真实收益可用前使用
            "weak_label": float(snap.get("momentum_1d", 0)),
        })

    def _update_labels(self, symbols):
        """
        v2 延迟标签：当前价格 vs 入场价，计算 N 根后的实现收益。
        只有超过 xgb_label_delay_bars 个小时后才贴标签。
        """
        delay_bars = int(self.config.get("xgb_label_delay_bars", 24))
        delay_secs = delay_bars * 3600
        now = time.time()
        # 建立当前价格字典
        price_map: Dict[str, float] = {}
        for sym in symbols:
            try: price_map[str(getattr(sym, "inst_id", ""))] = float(getattr(sym, "last_price", 0) or 0)
            except Exception: pass

        still_pending = []
        for sample in self._pending_samples:
            elapsed = now - float(sample.get("entry_time", now))
            sym = str(sample.get("symbol", ""))
            current_price = price_map.get(sym, 0.0)
            if elapsed >= delay_secs and current_price > 0:
                entry = float(sample.get("entry_price", 0) or 0)
                if entry > 0:
                    sample["realized_return"] = (current_price / entry - 1.0) * 100.0
                else:
                    sample["realized_return"] = sample.get("weak_label", 0.0)
                self._labeled_samples.append(sample)
            else:
                still_pending.append(sample)
        self._pending_samples = still_pending
        # 保留最近 2000 个已标注样本
        if len(self._labeled_samples) > 2000:
            self._labeled_samples = self._labeled_samples[-2000:]

    # ── 训练 ────────────────────────────────────────────────────────────────
    def _maybe_train(self):
        now = time.time()
        min_samples = int(self.config.get("xgb_min_samples", 500))
        retrain_hours = int(self.config.get("xgb_retrain_hours", 24))
        # 优先用已标注样本；不足时用弱标签样本补充
        usable = self._labeled_samples or self._pending_samples
        if len(usable) < min_samples:
            return
        if self._model is not None and (now - self._last_train_time) < retrain_hours * 3600:
            return
        try:
            self._train_model(usable)
            self._last_train_time = now
        except Exception as e:
            logger.error(f"[XGBoost] 训练失败: {e}")

    def _train_model(self, usable_samples):
        """v2: 修复 groups/qid 不一致 + 训练样本总和对齐"""
        if not _HAS_XGB:
            return
        samples = usable_samples[-2000:]
        n = len(samples)
        if n < 50: return

        X = pd.DataFrame([s["factors"] for s in samples])
        for col in _FACTOR_NAMES:
            if col not in X.columns:
                X[col] = 0.0
        X = X[_FACTOR_NAMES].fillna(0.0)

        # v2: 用实际标签（实现收益），回退到弱标签
        y = np.array([
            float(s.get("realized_return", s.get("weak_label", s.get("momentum_1d", 0))))
            for s in samples
        ])

        # v2 修复 #1: 截面按时间分组（每组约10条，但用余数确保总和=n）
        group_size = max(5, n // 10)
        n_full_groups = n // group_size
        remainder = n - n_full_groups * group_size
        groups = [group_size] * n_full_groups
        if remainder > 0:
            groups.append(remainder)  # 补全最后不满的组
        assert sum(groups) == n, f"groups和不等于样本数: {sum(groups)} != {n}"

        # v2 修复 #2: 只传 qid（不同时传 groups），避免 API 冲突
        qid = np.repeat(np.arange(len(groups)), groups)
        assert len(qid) == n

        params = {
            "objective": "rank:pairwise",
            "max_depth": int(self.config.get("xgb_max_depth", 5)),
            "learning_rate": float(self.config.get("xgb_learning_rate", 0.05)),
            "n_estimators": int(self.config.get("xgb_n_estimators", 120)),
            "subsample": float(self.config.get("xgb_subsample", 0.75)),
            "colsample_bytree": float(self.config.get("xgb_colsample_bytree", 0.75)),
            "reg_alpha": float(self.config.get("xgb_reg_alpha", 1.0)),
            "reg_lambda": float(self.config.get("xgb_reg_lambda", 2.0)),
            "min_child_weight": int(self.config.get("xgb_min_child_weight", 3)),
            "verbosity": 0,
            "random_state": 42,
        }
        model = xgb.XGBRanker(**params)
        # v2 修复 #2: 统一使用 qid 参数
        model.fit(X, y, qid=qid)
        self._model = model

        try:
            imp = model.get_booster().get_score(importance_type="gain")
            top5 = sorted(imp.items(), key=lambda x: x[1], reverse=True)[:5]
            label_src = "实际收益" if self._labeled_samples else "弱标签(动量)"
            logger.info(f"[XGBoost] 训练完成: 样本={n}, 标签={label_src}, 特征TOP5: {[(k,round(v,1)) for k,v in top5]}")
        except Exception:
            pass

    # ── 回测接口 ─────────────────────────────────────────────────────────────
    def generate_signal(self, data, *a, **kw):
        if not isinstance(data, dict) or not data.get("klines_map"): return None
        cfg = dict(self.config)
        cfg["min_score"] = float(cfg.get("backtest_min_score", cfg.get("min_score", 68)))
        sym = _xgb4__symbol_from_backtest(data, cfg)
        result = self.scan_symbol(sym)
        if not result.get("passed"): return None
        d = str(result.get("direction", "WAIT")).upper()
        if d not in {"BUY", "SELL"}: return None
        return {"action": "BUY" if d=="BUY" else "SHORT",
                "position_size": float(cfg.get("position_size", 0.1)),
                "entry_price": float(result.get("last_price", 0) or 0),
                "reason": f"XGBoost | {float(result.get('score', 0)):.1f}",
                "score": float(result.get("opportunity_score", result.get("score", 0)) or 0),
                "raw_result": result}

    def reset_backtest_state(self):
        self._pending_samples.clear()
        self._labeled_samples.clear()
        self._model = None
        self._last_train_time = 0.0


# ══════════════════════════════════════════════
# 截面评分
# ══════════════════════════════════════════════

def _cross_section_edge(z_factors: Dict[str, float], weights: Dict[str, float], model) -> float:
    """
    截面 edge：使用 z-score 归一化后的因子。
    有模型时用 XGBoost；无模型时用线性加权。
    """
    if model is not None and _HAS_XGB:
        try:
            X = pd.DataFrame([{k: float(z_factors.get(k, 0)) for k in _FACTOR_NAMES}])
            pred = float(model.predict(X)[0])
            return _clamp(pred, -3.0, 3.0)
        except Exception:
            pass
    # 线性回退
    return _clamp(sum(float(z_factors.get(n, 0)) * float(w) for n, w in weights.items()), -3.0, 3.0)


def _linear_score_from_factors(factors: Dict[str, float], weights: Dict[str, float]) -> float:
    """
    v2: 单标的线性评分，先对各因子做粗略归一化再加权。
    避免 momentum_1h（~2.0）和 trend_quality（~80.0）量纲混乱。
    """
    # 粗略归一化参数（典型范围的一半作为 scale）
    scale = {
        "momentum_1h": 5.0, "momentum_4h": 10.0, "momentum_1d": 15.0,
        "short_reversal": 5.0, "trend_quality": 50.0, "low_volatility": 10.0,
        "liquidity": 3.0, "volume_impulse": 1.5, "funding_carry": 0.1,
        "oi_heat": 3.0, "macd_momentum": 0.5, "bb_percentb": 0.5,
        "vol_zscore": 2.0, "close_strength": 0.5, "efficiency_ratio": 0.5,
        "rsi_alignment": 25.0, "momentum_decay": 1.0, "momentum_acceleration": 3.0,
        "early_trend_trigger": 2.0,
    }
    edge = 0.0
    for name, w in weights.items():
        if name not in factors: continue
        sc = max(scale.get(name, 1.0), 1e-9)
        normed = _clamp(float(factors[name]) / sc, -3.0, 3.0)
        edge += normed * float(w)
    return _clamp(edge * 16 + 50, 0, 100)  # edge∈[-3,3] → [2, 98]


# ══════════════════════════════════════════════
# 因子快照
# ══════════════════════════════════════════════

def _xgb4__build_snapshot(symbol, config) -> Dict[str, Any]:
    inst = str(getattr(symbol, "inst_id", ""))
    extra = getattr(symbol, "extra_data", {}) or {}
    klines = extra.get("klines", {}) or {}
    h1 = _to_df(_xgb4__getk(klines, "1H"))
    h4 = _to_df(_xgb4__getk(klines, "4H"))
    d1 = _to_df(_xgb4__getk(klines, "1D"))
    m3 = _to_df(_xgb4__getk(klines, "3m"))
    if len(h1) < 60 or len(h4) < 40:
        return {"valid": False, "symbol": inst, "reason": f"数据不足(1H={len(h1)},4H={len(h4)})"}

    close_1h = h1["c"].astype(float)
    close_4h = h4["c"].astype(float)
    vol_1h = h1["vol"].astype(float)
    lp = float(getattr(symbol, "last_price", 0) or close_1h.iloc[-1])
    v24 = float(getattr(symbol, "volume_24h", 0) or vol_1h.tail(24).sum())
    chg24 = float(getattr(symbol, "price_change_24h", 0) or _pct(close_1h, 24) * 100)

    m1h = _pct(close_1h, 6)   # 小数（比如 0.02 = 2%）
    m4h = _pct(close_4h, 12)
    m1d = _pct(d1["c"].astype(float), 7) if len(d1) >= 14 else _pct(close_1h, 168)

    # 趋势时效检查
    trend_hint = 1.0 if m4h > 0 else (-1.0 if m4h < 0 else 0.0)
    h1_trend_age = _measure_trend_age(close_1h, 12, 34, trend_hint)
    max_age = int(config.get("max_h1_trend_age", 12) or 12)
    if h1_trend_age > max_age * 2:
        return {"valid": False, "symbol": inst, "reason": f"1H趋势过老({h1_trend_age}根)"}

    # 3m 回调企稳检查
    if bool(config.get("require_m3_pullback", True)) and len(m3) >= 36:
        micro = _micro_pullback_check(m3, trend_hint, config)
        if not micro["confirmed"]:
            return {"valid": False, "symbol": inst, "reason": f"3m回调未确认: {micro['reason']}"}

    sr = -_pct(close_1h, 3)
    tq = _xgb4__trend_quality(h4)
    rv = _realized_vol(close_1h, 48)
    atr = _xgb4__atr_pct(h4, 14)
    liq = log(max(v24, 1.0))
    vi = _xgb4__vol_ratio(vol_1h, 24)
    vz = _vol_zscore(vol_1h)   # v2: 已修正排除当前bar
    fr = float((extra.get("funding_rate") or 0)) * 100
    oi = log(max(float(getattr(symbol, "open_interest", 0) or 1), 1))
    mm = _macd_mom(close_1h)
    bb = _bb_pctb(close_1h)    # v2: 已修正为标准 [0,1]
    cs = _xgb4__close_strength(h1, 6)
    er = _eff_ratio(close_4h, 20)
    ra = _rsi_align(close_1h, close_4h)
    mdecay = _mom_decay(m1h, m4h, m1d)   # v2: 统一在函数内乘100
    maccel = _mom_accel(close_1h, 6)
    early = _early_trigger(h1) if len(h1) >= 58 else 0.0  # v2: rsi_c 允许负值

    factors = {
        # 动量值保留小数形式（截面 z-score 归一化后量纲无关）
        "momentum_1h": m1h * 100,    # 百分比形式
        "momentum_4h": m4h * 100,
        "momentum_1d": m1d * 100,
        "short_reversal": sr * 100,
        "trend_quality": tq,          # 0-100
        "low_volatility": -rv,        # 负波动率 → 高值=低波
        "liquidity": liq,             # log(volume)
        "volume_impulse": vi,         # 比值 ~1.0-3.0
        "funding_carry": -abs(fr),    # 负绝对值 → 高值=低费率
        "oi_heat": oi,
        "macd_momentum": mm,
        "bb_percentb": bb,            # v2: [0,1]
        "vol_zscore": vz,
        "close_strength": cs,
        "efficiency_ratio": er,
        "rsi_alignment": ra,
        "momentum_decay": mdecay,
        "momentum_acceleration": maccel,
        "early_trend_trigger": early,
    }

    return {
        "valid": True, "symbol": inst,
        "last_price": lp, "volume_24h": v24, "price_change_24h": chg24,
        "momentum_1d": m1d, "momentum_4h": m4h, "momentum_1h": m1h,
        "atr_pct": atr, "factors": factors,
    }


# ══════════════════════════════════════════════
# 线性权重
# ══════════════════════════════════════════════

def _default_linear_weights():
    return {
        "momentum_1h": 0.07, "momentum_4h": 0.14, "momentum_1d": 0.11,
        "short_reversal": 0.02, "trend_quality": 0.12, "low_volatility": 0.06,
        "liquidity": 0.06, "volume_impulse": 0.06, "funding_carry": 0.02,
        "oi_heat": 0.02, "macd_momentum": 0.04, "bb_percentb": 0.02,
        "vol_zscore": 0.02, "close_strength": 0.02, "efficiency_ratio": 0.02,
        "rsi_alignment": 0.01, "momentum_decay": 0.05, "momentum_acceleration": 0.03,
        "early_trend_trigger": 0.07,
    }


# ══════════════════════════════════════════════
# 工具函数（修复版）
# ══════════════════════════════════════════════

def _xgb4__getk(klines, bar):
    for k in [bar, bar.lower(), bar.upper()]:
        if k in klines and klines.get(k): return klines[k]
    return []

def _pct(s, bars):
    if len(s) <= bars or bars <= 0: return 0.0
    return float(s.iloc[-1]) / max(float(s.iloc[-(bars+1)]), 1e-9) - 1

def _xgb4__vol_ratio(v, w):
    if len(v) < w + 3: return 1.0
    return float(v.tail(3).mean()) / max(float(v.iloc[-(w+1):-1].median() or 0), 1e-9)

def _vol_zscore(v):
    """v2: 排除当前 bar 的基线计算。"""
    if len(v) < 25: return 0.0
    baseline = v.iloc[-25:-1]  # 排除最后一根（当前bar）
    m = float(baseline.mean())
    s = float(baseline.std(ddof=0) or 1)
    return (float(v.iloc[-1]) - m) / s

def _xgb4__atr_pct(df, p):
    if len(df) < p + 2: return 1.0
    pc = df["c"].shift(1)
    tr = pd.concat([(df["h"]-df["l"]).abs(), (df["h"]-pc).abs(), (df["l"]-pc).abs()], axis=1).max(axis=1)
    a = float(tr.ewm(alpha=1/p, adjust=False).mean().iloc[-1] or 0)
    return a / float(df["c"].iloc[-1] or 1) * 100

def _realized_vol(c, w):
    r = c.pct_change().dropna().tail(w)
    return float(r.std(ddof=0) * np.sqrt(len(r)) * 100) if len(r) > 0 else 0

def _xgb4__trend_quality(h4):
    if len(h4) < 56: return 0.0
    c = h4["c"].astype(float)
    e21 = c.ewm(span=21, adjust=False).mean()
    e55 = c.ewm(span=55, adjust=False).mean()
    sp = (float(e21.iloc[-1]) / max(float(e55.iloc[-1]), 1e-9) - 1) * 100
    sl = float(e21.diff().tail(6).mean() or 0) / max(float(c.iloc[-1]), 1e-9) * 10000
    adx = _adx(h4)
    return _clamp(sp * 2.5 + sl * 0.8 + adx * 0.15, 0, 100)

def _macd_mom(c):
    if len(c) < 35: return 0.0
    f = c.ewm(span=12, adjust=False).mean()
    s = c.ewm(span=26, adjust=False).mean()
    d = f - s; dea = d.ewm(span=9, adjust=False).mean(); hist = d - dea
    return float((hist.iloc[-1] - hist.iloc[-3]) / max(abs(c.iloc[-1]), 1e-9) * 100)

def _bb_pctb(c):
    """v2 修正: 标准 %b = (price - lower) / (upper - lower)，范围 [0,1]。"""
    if len(c) < 21: return 0.5
    m = c.rolling(20).mean()
    s = c.rolling(20).std(ddof=1)
    upper = m + 2 * s; lower = m - 2 * s
    band = (upper - lower).iloc[-1]
    if band <= 0: return 0.5
    return float(_clamp((c.iloc[-1] - lower.iloc[-1]) / band, 0.0, 1.0))

def _xgb4__close_strength(df, n):
    c = df["c"].astype(float); h = df["h"].astype(float); l = df["l"].astype(float)
    vals = [(float(c.iloc[-i]) - float(l.iloc[-i])) / (float(h.iloc[-i]) - float(l.iloc[-i]) + 1e-9)
            for i in range(1, n+1)]
    return sum(vals) / n

def _eff_ratio(c, w):
    if len(c) < w + 1: return 0.5
    return abs(float(c.iloc[-1]) - float(c.iloc[-w])) / (c.diff().abs().tail(w).sum() + 1e-9)

def _rsi_align(c1, c4):
    if len(c1) < 15: return 50.0
    r1 = _rsi(c1, 14)
    r2 = _rsi(c4, 14) if len(c4) >= 15 else r1
    return 50 + (r1 - r2) * 0.5

def _mom_decay(m1, m4, md):
    """
    v2: 统一在函数内将小数形式的动量转换为百分比，再归一化。
    m1/m4/md 均为 _pct() 的原始返回（小数，如 0.02 = 2%）。
    """
    m1p = m1 * 100; m4p = m4 * 100; mdp = md * 100
    n1 = _clamp(m1p / 3.0, -1, 1)
    n4 = _clamp(m4p / 8.0, -1, 1)
    nd = _clamp(mdp / 15.0, -1, 1)
    return (n1 * 0.4 + n4 * 0.3) - nd * 0.7

def _mom_accel(c, p=6):
    if len(c) < p * 2 + 2: return 0.0
    cur = _pct(c, p)
    prev = _pct(c.iloc[:-(p)], p)
    return cur - prev

def _early_trigger(h1):
    if len(h1) < 58: return 0.0
    c = h1["c"].astype(float); h = h1["h"].astype(float)
    l = h1["l"].astype(float); v = h1["vol"].astype(float)
    e8 = c.ewm(span=8, adjust=False).mean()
    e21 = c.ewm(span=21, adjust=False).mean()
    e55 = c.ewm(span=55, adjust=False).mean()
    sp = float((e21.iloc[-1] - e55.iloc[-1]) / max(abs(e55.iloc[-1]), 1e-9) * 100)
    compression = 1 - _clamp(abs(sp) / 2.2, 0, 1)
    ema_c = _clamp(float((e8.iloc[-1]-e21.iloc[-1]) / max(float(c.iloc[-1]), 1e-9)*90 + sp*0.7) * (0.65+compression*0.55), -2, 2)
    ph = float(h.iloc[-21:-1].max()); pl = float(l.iloc[-21:-1].min())
    up = (float(c.iloc[-1]) / max(ph, 1e-9) - 1) * 100
    dn = (1 - float(c.iloc[-1]) / max(pl, 1e-9)) * 100
    don = _clamp(up/0.9, 0, 2) - _clamp(dn/0.9, 0, 2)
    r = _rsi(c); m = _macd_mom(c)
    # v2 修正: rsi_c 允许 [-1,1]，保留空头信号
    rsi_c = _clamp((r - 50) / 16, -1, 1)
    macd_c = _clamp(m / 5, -1, 1)
    v_ratio = _xgb4__vol_ratio(v, 24); cs = _xgb4__close_strength(h1, 3)
    vp = (1 if c.iloc[-1] >= c.iloc[-4] else -1) * _clamp((v_ratio-1)*0.7 + (cs-0.5)*1.2, -2, 2)
    return _clamp(ema_c*0.35 + don*0.25 + rsi_c*0.18 + macd_c*0.12 + vp*0.10, -2.5, 2.5)

def _micro_pullback_check(m3, trend_hint, config):
    if m3 is None or len(m3) < 36: return {"confirmed": False, "reason": "3m数据不足"}
    if abs(float(trend_hint)) < 0.10: return {"confirmed": False, "reason": "趋势不够明确"}
    n = len(m3)
    sb = max(2, int(config.get("m3_stabilization_bars", 4) or 4))
    mpb = float(config.get("m3_pullback_min_pct", 0.5) or 0.5)
    xpb = float(config.get("m3_pullback_max_pct", 2.2) or 2.2)
    mi = float(config.get("m3_min_impulse_pct", 0.65) or 0.65)
    c = m3["c"].astype(float); h = m3["h"].astype(float); l = m3["l"].astype(float)
    v = m3["vol"].astype(float) if "vol" in m3.columns else pd.Series(np.ones(n))
    e8 = c.ewm(span=8, adjust=False).mean(); e21 = c.ewm(span=21, adjust=False).mean()
    # 动态分段
    ss, se = max(0, n-sb), n
    ps, pe = max(0, se-max(6, sb+2)), se
    i1, i2 = max(0, pe-20), pe
    if i2 <= i1 or pe <= ps: return {"confirmed": False, "reason": "窗口分段不足"}
    if trend_hint > 0:
        il = float(l.iloc[i1:i2].min()); ih = float(h.iloc[i1:i2].max()); pl = float(l.iloc[ps:pe].min())
        ip = (ih/max(il,1e-9)-1)*100; pp = (ih/max(pl,1e-9)-1)*100; rt = pp/max(ip,1e-9)
        sc = c.iloc[ss:se]; sl = l.iloc[ss:se]; e8s = e8.iloc[ss:se]; e21s = e21.iloc[ss:se]
        ema_ok = float(e8s.iloc[-1]) >= float(e21s.iloc[-1]) * 0.997
        pa = float(sc.iloc[-1]) > float(e8s.iloc[-1])
        nl = float(sl.min()) >= pl * 0.995
        do = float(sc.iloc[-1]) >= float(sc.iloc[0]) * 0.998
        stabilized = pa and ema_ok and nl and do
    else:
        ih = float(h.iloc[i1:i2].max()); il = float(l.iloc[i1:i2].min()); ph = float(h.iloc[ps:pe].max())
        ip = (ih/max(il,1e-9)-1)*100; pp = (ph/max(il,1e-9)-1)*100; rt = pp/max(ip,1e-9)
        sc = c.iloc[ss:se]; sh = h.iloc[ss:se]; e8s = e8.iloc[ss:se]; e21s = e21.iloc[ss:se]
        ema_ok = float(e8s.iloc[-1]) <= float(e21s.iloc[-1]) * 1.003
        pb = float(sc.iloc[-1]) < float(e8s.iloc[-1])
        nh = float(sh.max()) <= ph * 1.005
        do = float(sc.iloc[-1]) <= float(sc.iloc[0]) * 1.002
        stabilized = pb and ema_ok and nh and do
    rv = float(v.iloc[ss:se].mean())
    bv = float(v.iloc[max(0,ss-18):ss].mean()) if ss >= 6 else float(v.mean())
    vo = rv >= bv * 0.78 if bv > 0 else True
    po = mpb <= pp <= xpb; io = ip >= mi; ro = 0.15 <= rt <= 0.85
    confirmed = bool(stabilized and vo and po and io and ro)
    rp = []
    if not io: rp.append("脉冲不足")
    if not po: rp.append("回调幅度超范围")
    if not ro: rp.append("回调比失衡")
    if not stabilized: rp.append("企稳不足")
    if not vo: rp.append("量能不足")
    return {"confirmed": confirmed, "reason": "通过" if confirmed else "，".join(rp) or "未通过"}


# ══════════════════════════════════════════════
# 结果构造 / 回测辅助
# ══════════════════════════════════════════════

def _xgb4__failed(symbol, reason):
    return {"symbol": str(getattr(symbol, "inst_id", "")), "passed": False,
            "score": 0, "direction": "WAIT", "signals": [], "details": {"状态": reason}}

def _xgb4__result(snap, score, direction, fs, passed, config):
    inst = snap["symbol"]
    sigs = [f"XGBoost截面排序 {'多头' if direction=='BUY' else '空头'} {score:.1f}"]
    ranking = {
        "trend": _clamp(50 + snap["momentum_4h"] * 40, 0, 100),
        "trigger": 88.0 if direction in {"BUY","SELL"} else 30.0,
        "volume": _clamp(log(max(snap["volume_24h"], 1)) / 18 * 100, 0, 100),
        "location": 55.0,
        "freshness": _clamp(50 + snap["momentum_1h"] * 30, 0, 100),
        "risk": _clamp(85 - snap["atr_pct"] * 6, 10, 95),
    }
    result = {
        "symbol": inst, "passed": passed,
        "score": round(float(score), 2),
        "direction": direction if direction in {"BUY","SELL"} else "WAIT",
        "signals": sigs,
        "category": f"XGBoost截面{'多头' if direction=='BUY' else '空头' if direction=='SELL' else '观察'}",
        "strategy_category": "XGBoost截面排序",
        "last_price": snap["last_price"],
        "volume_24h": snap["volume_24h"],
        "price_change_24h": snap["price_change_24h"],
        "ranking_factors": ranking,
        "factor_scores": {k: round(float(v), 4) for k, v in fs.items()},
        "details": {
            "机会类型": f"XGBoost截面{'多头' if direction=='BUY' else '空头' if direction=='SELL' else '观察'}",
            "评估": " | ".join(sigs),
        },
    }
    if build_opportunity_profile:
        try: result.update(build_opportunity_profile(score, direction, snap["volume_24h"], ranking, sigs))
        except Exception: pass
    return result

def _xgb4__symbol_from_backtest(data, config):
    km = data.get("klines_map", {}) or {}
    h1 = _to_df(_xgb4__getk(km, "1H") or data.get("klines") or [])
    lp = float(h1["c"].iloc[-1]) if not h1.empty else 0
    vol = float((h1["c"] * h1["vol"]).tail(48).sum()) if not h1.empty else 0
    extra = {"klines": km, "funding_rate": data.get("funding_rate", 0)}
    return _xgb4__MinimalSymbol(inst_id=str(config.get("inst_id", "BT")),
        last_price=lp, volume_24h=vol,
        price_change_24h=_pct(h1["c"], 24)*100 if not h1.empty else 0,
        extra_data=extra)

class _xgb4__MinimalSymbol:
    def __init__(self, inst_id, last_price, volume_24h, price_change_24h, extra_data):
        self.inst_id = inst_id; self.last_price = last_price
        self.volume_24h = volume_24h; self.price_change_24h = price_change_24h
        self.high_24h = 0; self.low_24h = 0; self.open_interest = 0
        self.extra_data = extra_data

# ════════════════════════════════════════════════════════════════════════════
# 子策略 5 — AI订单流动量突破组合策略
# ════════════════════════════════════════════════════════════════════════════

_of5_CONFIG_SCHEMA = {
    "min_score":              {"type":"float","default":72.0,     "label":"最低扫描分数"},
    "backtest_min_score":     {"type":"float","default":68.0,     "label":"回测最低分数"},
    "min_volume_24h":         {"type":"float","default":5_000_000,"label":"最小24H成交额"},
    "top_n_long":             {"type":"int",  "default":10,       "label":"多头保留数"},
    "top_n_short":            {"type":"int",  "default":6,        "label":"空头保留数"},
    "allow_short":            {"type":"bool", "default":True,     "label":"允许空头"},
    "max_atr_pct":            {"type":"float","default":8.0,      "label":"最大ATR%"},
    "position_size":          {"type":"float","default":0.10,     "label":"仓位比例"},
    # ── 订单流 ──
    "enable_orderflow":       {"type":"bool","default":True,      "label":"启用订单簿不平衡"},
    "orderflow_weight":       {"type":"float","default":0.18,     "label":"订单流因子权重"},
    "min_bid_ask_ratio":      {"type":"float","default":0.85,     "label":"最小bid/ask深度比"},
    # ── 动量交互 ──
    "enable_momentum_interact":{"type":"bool","default":True,     "label":"启用动量多项式交互"},
    "interact_weight":        {"type":"float","default":0.22,     "label":"交互项因子权重"},
    # ── BTC价差 ──
    "enable_btc_spread":      {"type":"bool","default":True,      "label":"启用BTC价差偏离"},
    "btc_spread_weight":      {"type":"float","default":0.15,     "label":"BTC价差因子权重"},
    "btc_spread_z_entry":     {"type":"float","default":1.5,      "label":"价差z-score入场阈值"},
    "coint_lookback":         {"type":"int",  "default":100,      "label":"协整检验回看K线数"},
    "coint_significance":     {"type":"float","default":0.05,     "label":"协整显著性水平"},
    # ── 回调企稳+趋势时效 ──
    "require_m3_pullback":    {"type":"bool","default":True,      "label":"要求3分钟回调企稳续势"},
    "m3_pullback_min_pct":    {"type":"float","default":0.50,     "label":"3分钟最小回调幅度%"},
    "m3_pullback_max_pct":    {"type":"float","default":2.20,     "label":"3分钟最大回调幅度%"},
    "m3_stabilization_bars":  {"type":"int","default":4,          "label":"3分钟企稳确认根数"},
    "m3_min_impulse_pct":     {"type":"float","default":0.65,     "label":"3m最小原趋势脉冲%"},
    "max_h1_trend_age":       {"type":"int","default":12,         "label":"1H趋势最大延续根数"},
    "require_m3_freshness":   {"type":"bool","default":False,     "label":"必须通过3m时效性"},
    # ── 微结构 ──
    "enable_micro_vwap":      {"type":"bool","default":True,      "label":"启用VWAP微结构"},
    "enable_volume_delta":    {"type":"bool","default":True,      "label":"启用买卖力量"},
    "enable_atr_squeeze":     {"type":"bool","default":True,      "label":"启用ATR收缩"},
}

_DEFAULT = {k: v["default"] for k, v in _of5_CONFIG_SCHEMA.items()}


class AIOrderflowMomentumBreakoutScanner(BaseScannerStrategy if _HAS_BASE else object):
    required_bars = ["3m", "15m", "1H", "4H", "1D"]
    requires_derivative_metrics = True
    name = "AI订单流+动量突破组合策略"
    description = "订单簿不平衡 + 多周期动量交互 + BTC价差偏离 + VWAP微结构"
    strategy_type = "scan"

    def __init__(self, config=None):
        self.config = {**_DEFAULT, **(config or {})}
        if _HAS_BASE and hasattr(super(), "__init__"):
            try: super().__init__(self.config)
            except Exception: pass

    def _init_conditions(self):
        if ScanCondition is None: return
        self.add_condition(ScanCondition(name="24H成交额", field="volume_24h",
            operator=">=", value=self.config.get("min_volume_24h", 5_000_000)))

    def get_config_schema(self): return dict(_of5_CONFIG_SCHEMA)

    def scan_symbol(self, symbol):
        snap = _of5__build_snapshot(symbol, self.config)
        if not snap["valid"]: return _of5__failed(symbol, snap["reason"])
        score, direction, factors = _score(snap, self.config)
        passed = score >= float(self.config.get("min_score", 72))
        return _of5__result(snap, score, direction, factors, passed, self.config)

    def scan_all_symbols(self, symbols):
        min_vol = float(self.config.get("min_volume_24h", 5_000_000))
        results = []
        for sym in symbols:
            if float(getattr(sym, "volume_24h", 0) or 0) < min_vol: continue
            r = self.scan_symbol(sym)
            if r.get("passed"): results.append(r)
        results.sort(key=lambda x: float(x.get("opportunity_score", x.get("score", 0)) or 0), reverse=True)
        top_n = int(self.config.get("top_n_long", 10))
        return {"type":"ai_orderflow_momentum_breakout","all_opportunities":results[:top_n]}

    def generate_signal(self, data, *a, **kw):
        if not isinstance(data, dict) or not data.get("klines_map"): return None
        cfg = dict(self.config)
        cfg["min_score"] = float(cfg.get("backtest_min_score", cfg.get("min_score", 68)))
        sym = _of5__symbol_from_backtest(data, cfg)
        result = self.scan_symbol(sym)
        if not result.get("passed"): return None
        d = str(result.get("direction", "WAIT")).upper()
        if d not in {"BUY", "SELL"}: return None
        return {"action":"BUY" if d=="BUY" else "SHORT",
                "position_size":float(cfg.get("position_size",0.1)),
                "entry_price":float(result.get("last_price",0) or 0),
                "reason":f"{result.get('category')} | {float(result.get('score',0)):.1f}",
                "score":float(result.get("opportunity_score",result.get("score",0)) or 0),
                "raw_result":result}

    def reset_backtest_state(self): pass


# ══════════════════════════════════════════════
# 快照构建
# ══════════════════════════════════════════════

def _of5__build_snapshot(symbol, config) -> Dict[str, Any]:
    inst = str(getattr(symbol, "inst_id", ""))
    extra = getattr(symbol, "extra_data", {}) or {}
    klines = extra.get("klines", {}) or {}

    m3 = _to_df(_of5__getk(klines, "3m"))
    h1 = _to_df(_of5__getk(klines, "1H"))
    h4 = _to_df(_of5__getk(klines, "4H"))
    d1 = _to_df(_of5__getk(klines, "1D"))

    if len(h1) < 60: return {"valid":False,"symbol":inst,"reason":f"1H数据不足({len(h1)})"}
    if len(h4) < 40: return {"valid":False,"symbol":inst,"reason":f"4H数据不足({len(h4)})"}

    close_1h = h1["c"].astype(float)
    close_4h = h4["c"].astype(float)
    vol_1h = h1["vol"].astype(float)
    lp = float(getattr(symbol,"last_price",0) or close_1h.iloc[-1])
    v24 = float(getattr(symbol,"volume_24h",0) or vol_1h.tail(24).sum())
    chg24 = float(getattr(symbol,"price_change_24h",0) or _pct(close_1h,24)*100)

    m1h = _pct(close_1h, 6)
    m4h = _pct(close_4h, 12)
    m1d = _pct(d1["c"].astype(float), 7) if len(d1)>=14 else _pct(close_1h, 168)

    # 趋势时效检查
    trend_hint = 1.0 if m4h > 0 else (-1.0 if m4h < 0 else 0.0)
    h1_trend_age = _measure_trend_age(close_1h, 12, 34, trend_hint)
    max_age = int(config.get("max_h1_trend_age", 12) or 12)
    if h1_trend_age > max_age * 2:
        return {"valid":False,"symbol":inst,"reason":f"1H趋势过老({h1_trend_age}根>{max_age*2}上限)"}

    # 3m 回调企稳
    if bool(config.get("require_m3_pullback", True)) and len(m3) >= 36:
        micro = _micro_pullback_continuation(m3, trend_hint, config)
        if not micro["confirmed"]:
            return {"valid":False,"symbol":inst,"reason":f"3m回调未确认: {micro['reason']}"}

    # ── 多项式交互项 ──
    enable_interact = bool(config.get("enable_momentum_interact", True))
    if enable_interact:
        atr_val = _of5__atr_pct(h4, 14)
        vol_ratio = _of5__vol_ratio(vol_1h, 24)
        # v2 修复 #5: 同向时 factor=1.0，反向时 factor=0.5（保留方向，降幅度）
        same_dir = (m1h * m4h) >= 0
        dir_factor = 1.0 if same_dir else 0.5
        interact_mom = m1h * m4h * dir_factor / max(abs(atr_val), 0.5)
        interact_eff = (m4h * 2 + m1h) / max(abs(atr_val), 1.0)
        interact_vol = vol_ratio * m1h
    else:
        atr_val = _of5__atr_pct(h4, 14)
        interact_mom = interact_eff = interact_vol = 0.0
        vol_ratio = 1.0

    # ── 订单簿不平衡 ──
    orderflow = 0.0
    if bool(config.get("enable_orderflow", True)):
        ob = extra.get("order_book") or extra.get("orderbook")
        if ob and isinstance(ob, dict):
            bids = ob.get("bids", []); asks = ob.get("asks", [])
            if bids and asks:
                bid_vol = sum(float(b[1]) for b in bids[:5] if len(b) > 1)
                ask_vol = sum(float(a[1]) for a in asks[:5] if len(a) > 1)
                if bid_vol + ask_vol > 0:
                    orderflow = (bid_vol - ask_vol) / (bid_vol + ask_vol)
        if abs(orderflow) < 0.01 and len(m3) >= 10:
            orderflow = _kline_orderflow(close_1h, h1["h"].astype(float),
                                         h1["l"].astype(float), vol_1h, tail=6)

    # ── BTC价差偏离（协整检验）──
    btc_spread_z = 0.0; btc_coint_pvalue = 1.0; btc_hedge_ratio = 1.0
    enable_btc = bool(config.get("enable_btc_spread", True))
    if enable_btc:
        btc_kl = extra.get("btc_klines")
        if btc_kl:
            btc_h1 = _to_df(_of5__getk({"1H": btc_kl} if isinstance(btc_kl, list) else btc_kl, "1H"))
            lookback = int(config.get("coint_lookback", 100) or 100)
            if len(btc_h1) >= lookback and len(close_1h) >= lookback:
                asset_log = np.log(close_1h.tail(lookback).values.astype(float))
                btc_log = np.log(btc_h1["c"].astype(float).tail(lookback).values)
                n = min(len(asset_log), len(btc_log))
                if n >= 60:
                    # 方法1: Johansen协整检验
                    coint_succeeded = False
                    if _HAS_COINT and coint_johansen is not None:
                        try:
                            data_mat = np.column_stack([asset_log[:n], btc_log[:n]])
                            jres = coint_johansen(data_mat, det_order=0, k_ar_diff=2)
                            trace_stat = jres.lr1[0]
                            crit_95 = jres.cvt[0, 1]
                            btc_coint_pvalue = min(1.0, crit_95 / max(trace_stat, 1e-9))
                            evec = jres.evec[:, 0]
                            btc_hedge_ratio = abs(evec[1] / max(abs(evec[0]), 1e-9))
                            spread_arr = asset_log[:n] - btc_hedge_ratio * btc_log[:n]
                            mean_s = np.mean(spread_arr); std_s = np.std(spread_arr) or 1e-9
                            btc_spread_z = float(np.clip((spread_arr[-1]-mean_s)/std_s, -3.0, 3.0))
                            if btc_coint_pvalue > float(config.get("coint_significance", 0.05)):
                                btc_spread_z *= 0.3
                            coint_succeeded = True
                        except Exception:
                            pass
                    # v2 修复 #6: 只有协整完全失败时才触发 fallback
                    if not coint_succeeded:
                        asset_ret = close_1h.pct_change().dropna().tail(min(48, n))
                        btc_ret = btc_h1["c"].astype(float).pct_change().dropna().tail(min(48, n))
                        nr = min(len(asset_ret), len(btc_ret))
                        if nr >= 12:
                            spread_ret = asset_ret.tail(nr).values - btc_ret.tail(nr).values
                            mean_s = np.mean(spread_ret); std_s = np.std(spread_ret) or 1e-9
                            btc_spread_z = float(np.clip((spread_ret[-1]-mean_s)/std_s, -2.0, 2.0)) * 0.6

    # ── VWAP微结构 ──
    vwap_score = 0.0
    if bool(config.get("enable_micro_vwap", True)) and len(m3) >= 20:
        vwap_score = _calc_vwap_score(m3, close_1h, m1h)

    # ── 买卖力量 ──
    delta_score = 0.0
    if bool(config.get("enable_volume_delta", True)) and len(h1) >= 12:
        # v2 修复 #4: 改为 iloc[-12:] 包含最新 bar
        buy, sell = _calc_delta(close_1h, h1["h"].astype(float),
                                 h1["l"].astype(float), vol_1h, -12, None)
        total = buy + sell
        delta_score = (buy - sell) / max(total, 1.0) if total > 0 else 0.0

    # ── ATR收缩 ──
    atr_sqz = 0.0
    if bool(config.get("enable_atr_squeeze", True)) and len(h4) >= 28:
        # v2 修复 #3: 正确对比当前 vs 历史 ATR
        atr_now = _atr_pct_recent(h4, -14, None)    # 最新14根
        atr_prev = _atr_pct_recent(h4, -28, -14)    # 前14根
        if atr_prev > 0:
            atr_sqz = max(0.0, 1.0 - atr_now / max(atr_prev, 0.01))

    # ── 因子打包（v2: enable=False 时 btc_spread_z 置 0 而非 50）──
    factors = {
        "momentum_1h":       _clamp(m1h * 25 + 50, 0, 100),
        "momentum_4h":       _clamp(m4h * 20 + 50, 0, 100),
        "interact_momentum": _clamp(50 + interact_mom * 8, 0, 100),
        "interact_efficiency":_clamp(50 + interact_eff * 6, 0, 100),
        "interact_volume":   _clamp(50 + interact_vol * 10, 0, 100),
        "orderflow_imbalance":_clamp(50 + orderflow * 30, 0, 100),
        "btc_spread_z":      _clamp(50 + btc_spread_z * 12, 0, 100) if enable_btc else 50.0,
        "vwap_micro":        _clamp(50 + vwap_score * 25, 0, 100),
        "volume_delta":      _clamp(50 + delta_score * 30, 0, 100),
        "atr_squeeze":       _clamp(atr_sqz * 100, 0, 100),
        "liquidity":         _clamp(log(max(v24, 1)) / 18 * 100, 0, 100),
    }

    return {
        "valid": True, "symbol": inst,
        "last_price": lp, "volume_24h": v24, "price_change_24h": chg24,
        "momentum_1h": m1h, "momentum_4h": m4h, "momentum_1d": m1d,
        "orderflow": orderflow, "btc_spread_z": btc_spread_z,
        "btc_coint_pvalue": btc_coint_pvalue, "btc_hedge_ratio": btc_hedge_ratio,
        "atr_pct": _of5__atr_pct(h4, 14), "factors": factors,
        "is_long": m4h > 0,
    }


# ══════════════════════════════════════════════
# 评分（v2：修复多空对称性）
# ══════════════════════════════════════════════

def _score(snap, config):
    f = snap["factors"]
    is_long = snap["momentum_4h"] > 0
    direction = "BUY" if is_long else "SELL"

    # v2 修复 #2: 多空对称评分
    # 多头: 因子值越高越好（因子已映射到[0,100]，50为中性）
    # 空头: 因子值越低越好，故取反 (100 - factor)
    if is_long:
        mom4 = f["momentum_4h"]
        mom1 = f["momentum_1h"]
        interact_vol_score = f["interact_volume"]
    else:
        mom4 = 100 - f["momentum_4h"]   # 空头：越跌越好 → 翻转因子
        mom1 = 100 - f["momentum_1h"]
        # v2 修复 #7: 空头取 interact_volume 的翻转值
        interact_vol_score = 100 - f["interact_volume"]

    base = mom4 * 0.20 + mom1 * 0.12

    # 交互项：对空头同样翻转
    iw = float(config.get("interact_weight", 0.22))
    if is_long:
        im_score = f["interact_momentum"]
        ie_score = f["interact_efficiency"]
        iv_score = interact_vol_score
    else:
        im_score = 100 - f["interact_momentum"]
        ie_score = 100 - f["interact_efficiency"]
        iv_score = interact_vol_score  # 已翻转
    interact = (im_score * 0.45 + ie_score * 0.35 + iv_score * 0.20) * iw

    # 订单流（已方向性：买压大利多，卖压大利空）
    ofw = float(config.get("orderflow_weight", 0.18)) if bool(config.get("enable_orderflow", True)) else 0.0
    if is_long:
        of_score = f["orderflow_imbalance"] * ofw
    else:
        of_score = (100 - f["orderflow_imbalance"]) * ofw

    # BTC价差（均值回归信号，方向相关）
    bsw = float(config.get("btc_spread_weight", 0.15)) if bool(config.get("enable_btc_spread", True)) else 0.0
    if is_long:
        bs_score = f["btc_spread_z"] * bsw
    else:
        bs_score = (100 - f["btc_spread_z"]) * bsw

    # 微结构（VWAP 已内置方向性，delta/squeeze 对多空通用）
    micro = 0.0
    if bool(config.get("enable_micro_vwap", True)):
        micro += (f["vwap_micro"] if is_long else 100 - f["vwap_micro"]) * 0.08
    if bool(config.get("enable_volume_delta", True)):
        micro += (f["volume_delta"] if is_long else 100 - f["volume_delta"]) * 0.06
    if bool(config.get("enable_atr_squeeze", True)):
        micro += f["atr_squeeze"] * 0.05  # 收缩方向无关，多空都加分

    score = base + interact + of_score + bs_score + micro + f["liquidity"] * 0.05

    # ATR惩罚
    max_atr = float(config.get("max_atr_pct", 8.0))
    if snap["atr_pct"] > max_atr:
        score -= min(20, (snap["atr_pct"] - max_atr) * 2.5)

    # 方向覆盖
    if not bool(config.get("allow_short", True)) and direction == "SELL":
        direction = "WAIT"
    if direction == "SELL" and snap["momentum_1d"] > 0.03:
        direction = "WAIT"  # 日线趋势向上时不空

    return _clamp(score, 0, 100), direction, f


# ══════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════

def _to_df(rows):
    if isinstance(rows, pd.DataFrame):
        df = rows.copy()
        rename = {k:v for k,v in {"timestamp":"ts","open":"o","high":"h","low":"l","close":"c","volume":"vol"}.items() if k in df.columns}
        if rename: df = df.rename(columns=rename)
    else:
        clean = [r[:6] for r in (rows or []) if isinstance(r,(list,tuple)) and len(r)>=6]
        if not clean: return pd.DataFrame(columns=["ts","o","h","l","c","vol"])
        try:
            df = pd.DataFrame(clean, columns=["ts","o","h","l","c","vol"])
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        except Exception:
            arr = np.array(clean, dtype=np.float64)
            df = pd.DataFrame(arr, columns=["ts","o","h","l","c","vol"])
    for c in ["ts","o","h","l","c","vol"]:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["ts","o","h","l","c"]).fillna({"vol":0}).sort_values("ts").drop_duplicates("ts",keep="last").reset_index(drop=True)

def _of5__getk(klines, bar):
    for k in [bar, bar.lower(), bar.upper(), bar.replace("m","M")]:
        if k in klines and klines.get(k): return klines.get(k)
    return []

def _of5__vol_ratio(vol_series, window):
    if len(vol_series) < window + 3: return 1.0
    base = float(vol_series.iloc[-(window+1):-1].median() or 0)
    latest = float(vol_series.tail(3).mean() or 0)
    return latest / base if base > 0 else 1.0

def _of5__atr_pct(df, period):
    if len(df) < period + 2: return 1.0
    pc = df["c"].shift(1)
    tr = pd.concat([(df["h"]-df["l"]).abs(),(df["h"]-pc).abs(),(df["l"]-pc).abs()],axis=1).max(axis=1)
    atr = float(tr.ewm(alpha=1/period, adjust=False).mean().iloc[-1] or 0)
    p = float(df["c"].iloc[-1] or 1)
    return atr / p * 100 if p > 0 else 1.0

def _atr_pct_recent(df, start, end):
    """
    v2 新增：用 iloc 相对最新位置的切片计算 ATR%。
    start=-14, end=None → 最新14根
    start=-28, end=-14  → 前14根
    """
    seg = df.iloc[start:end] if end is not None else df.iloc[start:]
    if len(seg) < 5: return 1.0
    pc = seg["c"].shift(1)
    tr = pd.concat([(seg["h"]-seg["l"]).abs(),(seg["h"]-pc).abs(),(seg["l"]-pc).abs()],axis=1).max(axis=1)
    atr = float(tr.mean() or 0)
    p = float(seg["c"].mean() or 1)
    return atr / p * 100 if p > 0 else 1.0

def _kline_orderflow(close, high, low, vol, tail=6):
    """从K线推断买卖压力（无orderbook时的回退）"""
    c = close.tail(tail).values; h = high.tail(tail).values
    l = low.tail(tail).values; v = vol.tail(tail).values
    spread = np.where(h - l > 0, h - l, 1.0)
    buy_ratio = np.clip((c - l) / spread, 0, 1)
    buy = np.sum(buy_ratio * v); sell = np.sum((1 - buy_ratio) * v)
    return (buy - sell) / max(buy + sell, 1e-9)

def _calc_vwap_score(m3, close_1h, m1h):
    """3m级别的VWAP偏离分数"""
    c = m3["c"].astype(float).tail(12)
    h = m3["h"].astype(float).tail(12)
    l = m3["l"].astype(float).tail(12)
    v = m3["vol"].astype(float).tail(12) if "vol" in m3.columns else pd.Series(np.ones(12))
    tp = (h.values + l.values + c.values) / 3
    tv = np.sum(v.values)
    vwap = np.sum(tp * v.values) / tv if tv > 0 else float(c.iloc[-1])
    latest = float(close_1h.iloc[-1])
    if m1h > 0:
        return max(0.0, min(1.0, (latest / vwap - 1.0) * 50))
    else:
        return max(0.0, min(1.0, (1.0 - latest / vwap) * 50))

def _calc_delta(close, high, low, vol, seg_start, seg_end):
    """
    v2 修复 #4: seg_end=None 时用 iloc[start:] 包含最新bar。
    """
    if seg_end is None:
        c = close.iloc[seg_start:].values
        h = high.iloc[seg_start:].values
        lv = low.iloc[seg_start:].values
        v = vol.iloc[seg_start:].values
    else:
        c = close.iloc[seg_start:seg_end].values
        h = high.iloc[seg_start:seg_end].values
        lv = low.iloc[seg_start:seg_end].values
        v = vol.iloc[seg_start:seg_end].values
    spread = np.where(h - lv > 0, h - lv, 1.0)
    br = np.clip((c - lv) / spread, 0, 1)
    return float(np.sum(br * v)), float(np.sum((1 - br) * v))

def _of5__failed(symbol, reason):
    return {"symbol":str(getattr(symbol,"inst_id","")),"passed":False,"score":0,"direction":"WAIT",
            "signals":[],"details":{"状态":reason}}

def _of5__result(snap, score, direction, factors, passed, config):
    """v2 修复 #1: 先算好 category，再构造 result dict。"""
    inst = snap["symbol"]
    # v2 修复 #1: 预先计算 category，避免 result 自引用
    if direction == "BUY":
        category = "AI订单流动量多头"
    elif direction == "SELL":
        category = "AI订单流动量空头"
    else:
        category = "AI订单流动量观察"

    sigs = [
        f"AI订单流动量突破 {'多头' if direction=='BUY' else '空头'} {score:.1f}",
        f"1H动量={snap['momentum_1h']*100:+.2f}% 4H={snap['momentum_4h']*100:+.2f}%",
        f"订单流={snap['orderflow']:+.2f} BTC价差Z={snap['btc_spread_z']:+.2f}",
    ]
    if snap.get("btc_coint_pvalue", 1.0) < 0.5:
        sigs.append(f"协整p≈{snap['btc_coint_pvalue']:.2f} 对冲比={snap['btc_hedge_ratio']:.2f}")

    ranking = {
        "trend": _clamp(50 + snap["momentum_4h"] * 60, 0, 100),
        "trigger": _clamp(50 + snap["orderflow"] * 35, 0, 100),
        "volume": _clamp(min(log(max(snap["volume_24h"], 1)), 18) / 18 * 100, 0, 100),
        "location": _clamp(55, 10, 95),
        "freshness": _clamp(50 + snap["momentum_1h"] * 40, 0, 100),
        "risk": _clamp(85 - snap["atr_pct"] * 6, 10, 95),
    }

    # v2 修复 #8: 过滤掉未启用的因子（avoid 50 中性噪声）
    active_factors = {k: round(float(v), 2) for k, v in factors.items()
                      if not (k == "btc_spread_z" and not bool(config.get("enable_btc_spread", True)))}

    result = {
        "symbol": inst, "passed": passed,
        "score": round(float(score), 2),
        "direction": direction if direction in {"BUY","SELL"} else "WAIT",
        "signals": sigs, "category": category,
        "strategy_category": "AI订单流动量",
        "last_price": snap["last_price"], "volume_24h": snap["volume_24h"],
        "price_change_24h": snap["price_change_24h"],
        "ranking_factors": ranking,
        "factor_scores": active_factors,
        "details": {
            "机会类型": category,          # v2 修复 #1: 直接用预算好的 category
            "1H动量%": f"{snap['momentum_1h']*100:+.2f}",
            "4H动量%": f"{snap['momentum_4h']*100:+.2f}",
            "订单流不平衡": f"{snap['orderflow']:+.3f}",
            "BTC价差Z": f"{snap['btc_spread_z']:+.2f}",
            "ATR%": f"{snap['atr_pct']:.2f}",
            "评估": " | ".join(sigs),
        },
    }
    if build_opportunity_profile:
        try: result.update(build_opportunity_profile(score, direction, snap["volume_24h"], ranking, sigs))
        except Exception: pass
    return result


def _of5__symbol_from_backtest(data, config):
    km = data.get("klines_map", {}) or {}
    h1 = _to_df(_of5__getk(km, "1H") or data.get("klines") or [])
    lp = float(h1["c"].iloc[-1]) if not h1.empty else 0.0
    vol = float((h1["c"] * h1["vol"]).tail(48).sum()) if not h1.empty else 0.0
    extra = {"klines": km, "order_book": data.get("order_book"),
             "btc_klines": data.get("btc_klines")}
    return _of5__MinimalSymbol(inst_id=str(config.get("inst_id","BT") or "BT"),
        last_price=lp, volume_24h=vol,
        price_change_24h=_pct(h1["c"],24)*100 if not h1.empty else 0,
        extra_data=extra)

class _of5__MinimalSymbol:
    def __init__(self, inst_id, last_price, volume_24h, price_change_24h, extra_data):
        self.inst_id=inst_id; self.last_price=last_price; self.volume_24h=volume_24h
        self.price_change_24h=price_change_24h; self.high_24h=0; self.low_24h=0
        self.open_interest=0; self.extra_data=extra_data

# ════════════════════════════════════════════════════════════════════════════
# 组合扫描器 — AI截面五引擎组合扫描器
# ════════════════════════════════════════════════════════════════════════════

CONFIG_SCHEMA = {
    # ── 基础 ──
    "min_volume_24h":               {"type": "float", "default": 5_000_000.0, "label": "组合最小24H成交额（v2.1: 8M→5M，更多币种纳入）"},
    "min_score":                    {"type": "float", "default": 68.0,        "label": "子策略最低扫描分数（共识信号用此门槛）"},
    "backtest_min_score":           {"type": "float", "default": 68.0,        "label": "回测最低入场分数"},
    "single_engine_min_score":      {"type": "float", "default": 78.0,        "label": "单引擎信号更高门槛（非共振信号需超此分才保留）"},
    "top_n":                        {"type": "int",   "default": 9,           "label": "组合最多输出（收紧为9条）"},
    "top_n_per_strategy":           {"type": "int",   "default": 6,           "label": "每个子策略最多输出"},
    "include_individual_results":   {"type": "bool",  "default": True,        "label": "保留子策略单独结果"},
    "include_consensus_results":    {"type": "bool",  "default": True,        "label": "输出多策略共振结果"},
    "dedupe_by_symbol":             {"type": "bool",  "default": True,        "label": "按交易对去重仅保留最高分"},
    "allow_short":                  {"type": "bool",  "default": True,        "label": "允许空头"},
    "max_atr_pct":                  {"type": "float", "default": 8.0,         "label": "最大ATR%"},
    # ── 共振 ──
    "min_consensus_engines":        {"type": "int",   "default": 2,           "label": "最少共振引擎数"},
    "consensus_bonus":              {"type": "float", "default": 8.5,         "label": "双引擎共振加分"},
    "triple_consensus_bonus":       {"type": "float", "default": 13.0,        "label": "三引擎共振加分"},
    "direction_conflict_penalty":   {"type": "float", "default": 8.0,         "label": "方向冲突降分"},
    # ── AI因子挖掘子策略 ──
    "min_abs_edge":                 {"type": "float", "default": 0.18,        "label": "AI策略最小优势（v2.1: 0.24→0.18，更多信号通过）"},
    "use_dynamic_ic_weights":       {"type": "bool",  "default": True,        "label": "启用动态IC权重"},
    "ic_weight_blend":              {"type": "float", "default": 0.55,        "label": "IC权重混合比例"},
    "enable_mfin_interactions":     {"type": "bool",  "default": True,        "label": "AI策略启用交互项"},
    "enable_llm_factors":           {"type": "bool",  "default": True,        "label": "AI策略启用LLM/社交因子"},
    "enable_on_chain":              {"type": "bool",  "default": True,        "label": "启用链上因子"},
    # ── 截面多因子子策略 ──
    "use_orthogonalization":        {"type": "bool",  "default": True,        "label": "截面策略启用因子正交化"},
    # ── 早启动因子（DRL+AI共用）──
    "enable_early_trend_factors":   {"type": "bool",  "default": True,        "label": "启用小时级早启动/转折因子"},
    "early_trend_min_trigger":      {"type": "float", "default": 0.12,        "label": "早启动最低触发强度（v2.1: 0.18→0.12）"},
    # ── 3m 回调企稳 ──
    "require_m3_pullback_confirmation": {"type": "bool",  "default": True,   "label": "要求3分钟回调企稳续势"},
    "m3_pullback_min_pct":          {"type": "float", "default": 0.10,        "label": "3分钟最小回调幅度%（v2.1: 0.20→0.10）"},
    "m3_pullback_max_pct":          {"type": "float", "default": 5.00,        "label": "3分钟最大回调幅度%（v2.1: 3.00→5.00，允许更深的回调整理）"},
    "m3_stabilization_bars":        {"type": "int",   "default": 2,           "label": "3分钟企稳确认根数"},
    "require_m3_freshness":         {"type": "bool",  "default": False,       "label": "必须通过3m时效性检查"},
    "m3_min_impulse_pct":           {"type": "float", "default": 0.45,        "label": "3m最小原趋势脉冲%（v2.1: 0.65→0.45）"},
    "vol_continuation_min_ratio":   {"type": "float", "default": 0.60,        "label": "企稳量能续航最低比例（v2.1: 0.78→0.60）"},
    # ── v3.2 微观结构 ──
    "enable_atr_squeeze_check":     {"type": "bool",  "default": True,        "label": "启用ATR收缩检测"},
    "atr_squeeze_ratio":            {"type": "float", "default": 0.42,        "label": "ATR收缩比例（v2.1: 0.55→0.42）"},
    "enable_volume_delta_check":    {"type": "bool",  "default": True,        "label": "启用买卖力量检测"},
    "volume_delta_min_ratio":       {"type": "float", "default": 1.05,        "label": "买卖量最低比值（v2.1: 1.15→1.05）"},
    "enable_vwap_alignment_check":  {"type": "bool",  "default": True,        "label": "启用VWAP对齐检测"},
    # ── 时效性过滤（v2 新增，三子策略共用）──
    "max_h1_trend_age":             {"type": "int",   "default": 8,           "label": "1H趋势最大延续根数（超过则惩罚，v5.0: 12→8）"},
    "h1_trend_age_penalty":         {"type": "float", "default": 10.0,        "label": "趋势过老分数惩罚（v5.0: 8→10）"},
    "max_m3_staleness_bars":        {"type": "int",   "default": 15,          "label": "3m回调最大时效根数"},
    "m3_freshness_penalty":         {"type": "float", "default": 6.0,         "label": "3m回调过旧分数惩罚"},
    "bonus_freshness_score":        {"type": "float", "default": 3.0,         "label": "两项时效均通过时加分"},
    # ── 硬性3m回调企稳过滤 ──
    "m3_hard_filter":               {"type": "bool",  "default": True,        "label": "硬性3m回调企稳过滤（不通过则丢弃）"},
    "m3_soft_filter_mode":          {"type": "bool",  "default": True,        "label": "软过滤模式（v2.1: 默认True，不过滤仅降权，发现更多机会）"},
    "m3_impulse_lookback_bars":     {"type": "int",   "default": 15,          "label": "3m回调检测回溯K线数"},
    "m3_no_break_tolerance_pct":    {"type": "float", "default": 0.60,        "label": "回调不得跌破突破点的容差%（v2.1: 0.30→0.60，更宽松）"},
    # ── 硬性小时线趋势延续过滤 ──
    "h1_trend_hard_filter":         {"type": "bool",  "default": False,       "label": "硬性1H趋势延续过滤（v2.1: 默认False，不做EMA强制拦截）"},
    "h1_ema_fast":                  {"type": "int",   "default": 12,          "label": "1H快线EMA周期"},
    "h1_ema_slow":                  {"type": "int",   "default": 26,          "label": "1H慢线EMA周期"},
    # ── 性能 / 模式 ──
    "position_size":                {"type": "float", "default": 0.02,        "label": "回测仓位比例"},
    "use_pilot_add_system":         {"type": "bool",  "default": True,        "label": "启用1%试仓+10%加仓系统"},
    "mode": {
        "type": "select", "default": "normal", "label": "扫描模式",
        "options": [{"label": "常规", "value": "normal"}, {"label": "超严", "value": "ultra"}],
    },
    "ultra_strict_mode":            {"type": "bool",  "default": False,       "label": "超严模式(目标6-9条)"},
    "ultra_target_top_n":           {"type": "int",   "default": 9,           "label": "超严模式最大输出"},
    "parallel_child_engines":       {"type": "bool",  "default": True,        "label": "并行运行子引擎(v6.0: False→True，5×速度提升)"},
    "fast_scan_mode":               {"type": "bool",  "default": False,       "label": "启用快速候选池"},
    "max_scan_symbols":             {"type": "int",   "default": 200,         "label": "快速候选池上限"},
    "drl_candidate_cap":            {"type": "int",   "default": 180,         "label": "DRL子引擎候选池上限"},
    "profile_child_timing":         {"type": "bool",  "default": True,        "label": "输出子引擎耗时"},
    "accelerate_cross_section_child": {"type": "bool","default": True,        "label": "组合内启用截面子引擎加速"},
    "accel_disable_m3":             {"type": "bool",  "default": False,       "label": "加速模式下关闭3m企稳检查"},
    # v4.0 新增
    "enable_state_conditional_weights": {"type":"bool","default":True,       "label": "启用市场状态条件权重"},
    # v4.1 新增：评分归一化 + 引擎动态权重 + 信号持续时长 + 绩效追踪
    "enable_score_normalization":     {"type": "bool",  "default": True,        "label": "启用跨引擎评分z-score归一化"},
    "enable_engine_track_record":     {"type": "bool",  "default": True,        "label": "启用引擎历史胜率动态权重"},
    "engine_weight_decay":            {"type": "float", "default": 0.85,        "label": "引擎权重衰减因子(0.7-1)"},
    "min_signal_persistence_bars":   {"type": "int",   "default": 0,           "label": "信号最少持续3m根数(防闪信号，0=关闭，扫描器重建时不应使用)"},
    "prefilter_3m_before_consensus": {"type": "bool",  "default": False,       "label": "3m回调预过滤(子引擎结果先在组合层过3m再参与共识，与m3_hard_filter双重过滤，建议关闭)"},
    "max_track_record_entries":      {"type": "int",   "default": 50,          "label": "绩效追踪最大记录数"},
    # v4.2 交易员视角新增 ──────────────────────────────────────────────
    "enable_btc_correlation_filter": {"type": "bool",  "default": True,        "label": "启用BTC相关性过滤(BTC暴跌时降权山寨多头)"},
    "btc_dump_threshold_pct":        {"type": "float", "default": -2.5,        "label": "BTC 1H跌幅阈值%(低于此值降权山寨多头)"},
    "btc_dump_penalty":              {"type": "float", "default": 12.0,        "label": "BTC暴跌时山寨多头降分"},
    "enable_funding_filter":         {"type": "bool",  "default": True,        "label": "启用资金费率极端检测"},
    "funding_extreme_threshold":     {"type": "float", "default": 0.10,        "label": "资金费率极端阈值%(>此值降权多头)"},
    "funding_penalty":               {"type": "float", "default": 8.0,         "label": "极端费率降分"},
    "enable_confluence_scoring":    {"type": "bool",  "default": True,        "label": "启用1H/4H/日线多周期共振加分"},
    "confluence_bonus_max":          {"type": "float", "default": 6.0,         "label": "多周期共振最高加分"},
    "enable_atr_stop_suggestion":    {"type": "bool",  "default": True,        "label": "输出ATR止损建议位"},
    "stop_atr_multiplier":           {"type": "float", "default": 1.8,         "label": "止损ATR倍数"},
    "enable_volume_quality_check":   {"type": "bool",  "default": True,        "label": "启用成交量质量检测(过滤刷量)"},
    "vol_conc_ratio_threshold":      {"type": "float", "default": 0.60,        "label": "成交量集中度阈值(>60%视为可疑刷量)"},
    "enable_btc_relative_strength":  {"type": "bool",  "default": True,        "label": "启用相对BTC强弱过滤"},
    "btc_rs_min_ratio":              {"type": "float", "default": -0.3,        "label": "相对BTC最小涨跌幅比(低于此值标记弱势)"},
    # v4.3 交易员视角：1H小时线突破检测 ─────────────────────────────────
    "enable_h1_breakout_detect":     {"type": "bool",  "default": True,        "label": "启用1H突破检测(及时捕获小时线起涨点)"},
    "h1_breakout_lookback":          {"type": "int",   "default": 12,          "label": "1H突破回溯K线数(突破点=N根内最高)"},
    "h1_breakout_vol_ratio":         {"type": "float", "default": 1.35,        "label": "1H突破放量倍数(突破K线量/均量)"},
    "h1_breakout_close_position":    {"type": "float", "default": 0.65,        "label": "1H突破收盘位置(收在K线高位%确认强势)"},
    "h1_breakout_bonus":             {"type": "float", "default": 8.0,         "label": "1H突破检测通过加分"},
    "h1_early_breakout_bonus":      {"type": "float", "default": 4.0,         "label": "1H早期突破(未放量/未确认)加分"},
    "enable_h1_squeeze_detect":     {"type": "bool",  "default": True,        "label": "启用1H波动率压缩检测(布林带收窄→突破前兆)"},
    "h1_squeeze_bb_period":          {"type": "int",   "default": 20,          "label": "布林带周期"},
    "h1_squeeze_bb_width_percentile": {"type": "float","default": 0.30,       "label": "布林带宽历史分位(<此值视为压缩)（v2.1: 0.20→0.30）"},
    "h1_squeeze_bonus":              {"type": "float", "default": 4.0,         "label": "波动压缩预突破加分"},
    "enable_h1_structure_score":    {"type": "bool",  "default": True,        "label": "启用1H结构评分(HH/HL/支撑测试)"},
    "h1_structure_lookback":         {"type": "int",   "default": 24,          "label": "1H结构回溯K线数"},
    "h1_structure_bonus_max":        {"type": "float", "default": 5.0,         "label": "1H结构优秀最高加分"},
    "h1_breakout_relax_3m":         {"type": "bool",  "default": True,        "label": "1H突破时放宽3m回调要求"},

    # ── v5.0 早启动/回调再启动优化 ──────────────────────────────────────────
    "h1_entry_timing_enabled":      {"type": "bool",  "default": True,        "label": "启用H1入场时机评分（早启动/回调再启动）"},
    "h1_fresh_cross_bars":          {"type": "int",   "default": 6,           "label": "EMA金叉/死叉视为刚发生的最大H1根数"},
    "h1_fresh_cross_bonus":         {"type": "float", "default": 12.0,        "label": "EMA刚交叉加分（趋势刚启动）"},
    "h1_pullback_reentry_enabled":  {"type": "bool",  "default": True,        "label": "启用H1回调再启动检测"},
    "h1_pullback_reentry_bonus":    {"type": "float", "default": 10.0,        "label": "H1回调后再启动加分"},
    "h1_ema_gap_hard_limit_pct":    {"type": "float", "default": 8.0,         "label": "H1 EMA12/34间距超此%时视为趋势过老（硬限制）"},
    "h1_ema_gap_penalty":           {"type": "float", "default": 15.0,        "label": "趋势过老时惩罚分（EMA间距过大）"},
    "h1_trend_age_hard_limit":      {"type": "int",   "default": 16,          "label": "H1趋势年龄硬上限（超过直接拒绝，0=不限）"},

    # ── v6.0 P1-1 子策略按风格分组共识 ───────────────────────────────────
    "style_grouped_consensus":      {"type": "bool",  "default": True,        "label": "v6.0 启用风格分组共识(3因子模型算1票，避免假共振)"},
    "min_style_groups_consensus":   {"type": "int",   "default": 2,           "label": "最少需要N个不同风格组同向(替代min_consensus_engines)"},

    # ── v6.0 P1-3 z-score 排名门槛 ────────────────────────────────────
    "use_zscore_ranking":           {"type": "bool",  "default": True,        "label": "v6.0 启用z-score候选池排名(替代绝对分数门槛)"},
    "min_zscore_threshold":         {"type": "float", "default": 0.5,         "label": "最低z-score(0.5≈前30%, 1.04≈前15%)"},

    # ── v6.0 P2-1 突破放量百分位 ──────────────────────────────────────
    "h1_breakout_vol_use_percentile": {"type": "bool","default": True,        "label": "v6.0 突破放量改用历史分位(替代固定倍数)"},
    "h1_breakout_vol_percentile":   {"type": "float", "default": 0.75,        "label": "成交量历史分位阈值(0.75=前25%)"},

    # ── v6.0 P2-2 H4 方向硬过滤 ───────────────────────────────────────
    "h4_direction_hard_filter":     {"type": "bool",  "default": True,        "label": "v6.0 启用H4方向硬过滤(H4反向直接淘汰)"},
    "h4_filter_min_confidence":     {"type": "float", "default": 0.4,         "label": "H4反向硬拒最低置信度(EMA gap>此值才硬拒)"},

    # ── v6.0 P2-5 BTC 熔断机制 ────────────────────────────────────────
    "btc_circuit_breaker_enabled":  {"type": "bool",  "default": True,        "label": "v6.0 启用BTC剧烈波动熔断"},
    "btc_circuit_threshold_pct":    {"type": "float", "default": 5.0,         "label": "BTC 1H |涨跌|超此值时全面熔断"},
}

_DEFAULT_CONFIG = {key: spec["default"] for key, spec in CONFIG_SCHEMA.items()}


# ══════════════════════════════════════════════════════════════════════════════
class AICrossSectionDualFactorComboScanner(BaseScannerStrategy if _HAS_SCANNER_BASE else object):
    """截面多因子 + AI因子挖掘 + DRL小时趋势启动 + XGBoost + 订单流 五引擎组合。"""

    required_bars = ["3m", "15m", "1H", "4H", "1D"]
    requires_derivative_metrics = True
    requires_on_chain_metrics = True
    name = "AI截面五引擎组合扫描器"
    description = "五引擎共振：截面多因子 + AI因子挖掘 + DRL小时趋势 + XGBoost排序 + 订单流动量"
    strategy_type = "scan"

    CHILDREN = [
        ("截面多因子",      "截面多因子加密货币扫描4.23_v3.py",      "ACrossSectionalMultiFactorScannerStrategy", 970),
        ("AI因子挖掘",      "AI因子挖掘加密货币扫描策略4.23_v3.py",  "AIAutomatedAlphaCryptoScannerStrategy",     960),
        ("DRL小时趋势启动", "DRL元学习小时趋势启动扫描策略4.23_v2.py","DRLMetaHourlyTrendStartScannerStrategy",    965),
        ("XGBoost截面排序", "XGBoost截面排序策略4.27_v2.py",         "XGBoostCrossSectionalRanker",               955),
        ("AI订单流动量",    "AI订单流动量突破组合策略4.27_v2.py",    "AIOrderflowMomentumBreakoutScanner",        945),
    ]

    # P1-1: 子策略风格分组（解决"3 个因子模型同向算 3 票"的假象共振问题）
    # 真正独立的风格只有 3 类：因子（横截面）/ 动量（时序）/ 订单流（微观）
    ENGINE_STYLES = {
        "factor":    {"截面多因子", "AI因子挖掘", "XGBoost截面排序"},  # 横截面因子模型
        "momentum":  {"DRL小时趋势启动"},                                 # 时序动量
        "orderflow": {"AI订单流动量"},                                   # 订单流微观
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = {**_DEFAULT_CONFIG, **(config or {})}
        self._apply_runtime_modes()
        self.child_strategies: List[Tuple[str, Any, int]] = []
        self.last_child_results: Dict[str, List[Dict[str, Any]]] = {}
        self.last_child_timing: Dict[str, float] = {}
        self._synced = False  # v2: 防止 _sync_config 重复清空子策略缓存
        # v4.1: 引擎动态权重追踪
        self._engine_track: Dict[str, Dict[str, List[float]]] = {}  # {engine_name: {"BUY": [pnls], "SELL": [pnls]}}
        # v4.1: 信号持续时长追踪 {symbol_direction: last_seen_scan_idx}
        self._signal_persistence: Dict[str, int] = {}
        self._scan_counter: int = 0
        # v4.1: 历史绩效缓存 {symbol: [(direction, score, result_24h), ...]}
        self._performance_cache: Dict[str, List[Tuple[str, float, Optional[float]]]] = {}
        self._max_perf_entries: int = int(self.config.get("max_track_record_entries", 50) or 50)
        # P1-2: H1/4H 指标缓存 {symbol_id: {closes/highs/lows/ema12/ema21/ema26/ema34/ema50/atr14}}
        # 每次 scan_all_symbols 末尾清空，确保扫描内一致 + 不跨次累积内存
        self._symbol_indicator_cache: Dict[str, Dict[str, Any]] = {}
        if _HAS_SCANNER_BASE and hasattr(super(), "__init__"):
            try:
                super().__init__(self.config)
            except Exception:
                pass
        self.child_strategies = self._build_child_strategies()

    # ── P1-1: 风格分组工具 ──────────────────────────────────────────────────
    def _get_engine_style(self, engine_name: str) -> str:
        """返回引擎所属风格组：factor / momentum / orderflow / other"""
        for style, members in self.ENGINE_STYLES.items():
            if engine_name in members:
                return style
        return "other"

    def _count_style_groups(self, items: List[Dict], direction: str) -> Tuple[int, Dict[str, int]]:
        """统计 items 中有多少个**不同风格组**支持给定方向。
        返回 (风格组数, {style: 引擎数})。
        """
        style_counts: Dict[str, int] = {}
        for item in items:
            if str(item.get("direction", "")).upper() != direction:
                continue
            sn = str(item.get("source_strategy", ""))
            style = self._get_engine_style(sn)
            style_counts[style] = style_counts.get(style, 0) + 1
        return len(style_counts), style_counts

    # ── P1-2: H1 指标缓存 ──────────────────────────────────────────────────
    def _get_h1_indicators(self, symbol) -> Optional[Dict[str, Any]]:
        """获取并缓存 H1 EMA/ATR/收盘序列，避免在多个过滤器中重复计算。"""
        sym_id = str(getattr(symbol, "inst_id", ""))
        if not sym_id:
            return None
        if sym_id in self._symbol_indicator_cache:
            return self._symbol_indicator_cache[sym_id]
        rows = self._get_h1_klines(symbol)
        if len(rows) < 26:
            self._symbol_indicator_cache[sym_id] = None
            return None
        try:
            closes = [float(r[4]) for r in rows if float(r[4]) > 0]
            highs  = [float(r[2]) for r in rows if float(r[2]) > 0]
            lows   = [float(r[3]) for r in rows if float(r[3]) > 0]
            vols   = [float(r[5]) if len(r) > 5 else 0.0 for r in rows]
            if len(closes) < 26:
                self._symbol_indicator_cache[sym_id] = None
                return None
            s = pd.Series(closes)
            ema12 = s.ewm(span=12, adjust=False).mean()
            ema21 = s.ewm(span=21, adjust=False).mean()
            ema26 = s.ewm(span=26, adjust=False).mean()
            ema34 = s.ewm(span=34, adjust=False).mean()
            ema50 = s.ewm(span=50, adjust=False).mean() if len(closes) >= 50 else ema26
            # ATR(14)
            atr14 = 0.0
            if len(closes) >= 16:
                trs = [max(highs[i]-lows[i], abs(highs[i]-closes[i-1]), abs(lows[i]-closes[i-1]))
                       for i in range(-14, 0)]
                atr14 = float(np.mean(trs))
            cache = {
                "rows": rows, "closes": closes, "highs": highs, "lows": lows, "vols": vols,
                "ema12": ema12, "ema21": ema21, "ema26": ema26, "ema34": ema34, "ema50": ema50,
                "atr14": atr14,
                "atr_pct": (atr14 / max(closes[-1], 1e-9) * 100.0) if atr14 > 0 else 0.0,
            }
            self._symbol_indicator_cache[sym_id] = cache
            return cache
        except Exception:
            self._symbol_indicator_cache[sym_id] = None
            return None

    def _init_conditions(self):
        if ScanCondition is None or not hasattr(self, "add_condition"):
            return
        self.add_condition(ScanCondition(
            name="组合流动性", description="过滤成交额不足的交易对",
            field="volume_24h", operator=">=",
            value=self.config.get("min_volume_24h", 5_000_000.0),
        ))

    def get_config_schema(self) -> Dict[str, Any]:
        return dict(CONFIG_SCHEMA)

    # ── 单标的扫描 ──────────────────────────────────────────────────────────
    def scan_symbol(self, symbol) -> Dict[str, Any]:
        child_results = []
        for strategy_name, strategy, priority in self.child_strategies:
            try:
                result = strategy.scan_symbol(symbol)
            except Exception as exc:
                logger.error(f"[组合] {strategy_name} -> {getattr(symbol,'inst_id','')} 失败: {exc}")
                continue
            normalized = self._annotate_child_result(result, strategy_name, priority)
            if normalized.get("passed"):
                child_results.append(normalized)

        consensus = self._build_consensus_result(getattr(symbol, "inst_id", ""), child_results)
        candidates = ([consensus] if consensus else []) + child_results
        if candidates:
            candidates.sort(key=_result_sort_key, reverse=True)
            best = self._apply_m3_hard_filter(candidates[0], symbol)
            return best
        return {
            "symbol": getattr(symbol, "inst_id", ""),
            "passed": False, "score": 0.0, "direction": "WAIT",
            "category": "AI截面组合观察",
            "details": {"状态": "三个子策略均未触发"},
        }

    # ── 批量扫描 ─────────────────────────────────────────────────────────────
    def scan_all_symbols(self, symbols: List) -> Dict[str, Any]:
        top_n_per = int(self.config.get("top_n_per_strategy", 6) or 6)
        scan_symbols = self._select_scan_universe(symbols)
        # 构建 inst_id → symbol 对象的查找表（用于硬性3m过滤）
        sym_lookup: Dict[str, Any] = {
            str(getattr(s, "inst_id", "") or ""): s for s in scan_symbols
        }
        all_child: List[Dict[str, Any]] = []
        by_strategy: Dict[str, List[Dict[str, Any]]] = {}
        timing: Dict[str, float] = {}

        if bool(self.config.get("parallel_child_engines", False)) and len(self.child_strategies) > 1:
            with ThreadPoolExecutor(max_workers=min(5, len(self.child_strategies))) as exc:
                futures = {
                    exc.submit(self._timed_child_scan, sn, st, pri, scan_symbols): (sn, pri)
                    for sn, st, pri in self.child_strategies
                }
                for future in as_completed(futures):
                    sn, _ = futures[future]
                    try:
                        results, elapsed = future.result()
                    except Exception as e:
                        logger.error(f"[组合] {sn} 并行扫描失败: {e}")
                        results, elapsed = [], 0.0
                    timing[sn] = round(float(elapsed), 3)
                    results.sort(key=_result_sort_key, reverse=True)
                    by_strategy[sn] = results[:top_n_per]
                    all_child.extend(results[:top_n_per])
        else:
            for sn, st, pri in self.child_strategies:
                results, elapsed = self._timed_child_scan(sn, st, pri, scan_symbols)
                timing[sn] = round(float(elapsed), 3)
                results.sort(key=_result_sort_key, reverse=True)
                by_strategy[sn] = results[:top_n_per]
                all_child.extend(results[:top_n_per])

        consensus_results = []
        if bool(self.config.get("include_consensus_results", True)):
            consensus_results = self._build_consensus_results(all_child)

        output: List[Dict[str, Any]] = []
        if bool(self.config.get("include_consensus_results", True)):
            output.extend(consensus_results)
        if bool(self.config.get("include_individual_results", True)):
            output.extend(all_child)

        output = _dedupe_results(output, dedupe_by_symbol=bool(self.config.get("dedupe_by_symbol", True)))
        output.sort(key=_result_sort_key, reverse=True)
        top_n = int(self.config.get("top_n", 12) or 12)

        # ── P1-3: z-score 排名门槛（候选池前 N% 而非绝对分数）──────────
        # 解决评分膨胀（基础 60+加分 18 = 78，与真正 86 难区分）
        # zscore >= 0.5 ≈ 前 30%，zscore >= 1.04 ≈ 前 15%
        if bool(self.config.get("use_zscore_ranking", True)) and len(output) >= 5:
            scores = [float(r.get("score", 0) or 0) for r in output]
            mean_s = float(np.mean(scores))
            std_s = max(float(np.std(scores)), 1e-6)
            min_z = float(self.config.get("min_zscore_threshold", 0.5))
            for r in output:
                z = (float(r.get("score", 0) or 0) - mean_s) / std_s
                r["_zscore"] = round(z, 3)
                r.setdefault("details", {})["z-score"] = f"{z:+.2f} (μ={mean_s:.1f} σ={std_s:.2f})"
        # ── 3m/1H 质量检查 ──────────────────────────────────────────
        # 共识结果（多引擎共振）：跳过硬过滤，仅做软降权
        # 个体结果：执行硬过滤（不通过则淘汰）
        hard_filter = bool(self.config.get("m3_hard_filter", True))
        soft_only = bool(self.config.get("m3_soft_filter_mode", False))
        global_min = float(self.config.get("min_score", 68.0) or 68.0)
        single_min = float(self.config.get("single_engine_min_score", 78.0) or 78.0)
        filtered = []
        relax_for_h1 = bool(self.config.get("h1_breakout_relax_3m", True))
        # P1-3: z-score 排名过滤（与绝对分数门槛并行检查）
        zscore_on = bool(self.config.get("use_zscore_ranking", True)) and len(output) >= 5
        z_threshold = float(self.config.get("min_zscore_threshold", 0.5))
        for item in output:
            sym_id = str(item.get("symbol", ""))
            sym_obj = sym_lookup.get(sym_id)
            # P1-3: z-score 排名预筛（共识结果免筛，避免高质量信号被均值统计淹没）
            if zscore_on and "共振" not in str(item.get("category", "")):
                z = float(item.get("_zscore", 0))
                if z < z_threshold:
                    logger.debug(f"[z-score] 淘汰 {sym_id}: z={z:.2f}<{z_threshold}")
                    continue
            if not sym_obj:
                if item.get("passed"):
                    filtered.append(item)
                continue
            # v4.3: 检查1H是否已突破 — 若是则放宽3m过滤
            h1_breakout_confirmed = False
            if relax_for_h1:
                is_bo, bo_conf, _ = self._h1_breakout_detect(sym_obj)
                h1_breakout_confirmed = is_bo and bo_conf >= 0.67
            relax_m3 = h1_breakout_confirmed
            if relax_m3:
                item["_relax_3m"] = True
                item.setdefault("details", {})["1H突破"] = "1H已确认突破，3m过滤放宽"
            is_consensus = "共振" in str(item.get("category", item.get("source_strategy", "")))
            if is_consensus:
                # 共识结果：仅软检查，不淘汰
                check_ok, check_reason = self._m3_pullback_hard_check(
                    self._get_m3_klines(sym_obj),
                    str(item.get("direction", "WAIT")).upper(),
                    relax=relax_m3
                )
                h1_ok, h1_reason = (True, "") if not bool(self.config.get("h1_trend_hard_filter", True)) else \
                    self._h1_trend_check(self._get_h1_klines(sym_obj), str(item.get("direction", "WAIT")).upper())
                details = dict(item.get("details") or {})
                details["3m硬性过滤"] = f"{'通过' if check_ok else '未通过(共识免淘汰)'}: {check_reason}"
                details["1H趋势过滤"] = f"{'通过' if h1_ok else '未通过(共识免淘汰)'}: {h1_reason}"
                item["details"] = details
                if not check_ok:
                    item["score"] = round(max(0, float(item.get("score", 0) or 0) - 5.0), 2)
                if not h1_ok:
                    item["score"] = round(max(0, float(item.get("score", 0) or 0) - 3.0), 2)
                filtered.append(item)
            elif hard_filter and not soft_only:
                # 单引擎信号额外分数门槛：需超过 single_engine_min_score
                item_score = float(item.get("score", 0) or 0)
                if item_score < global_min:
                    logger.info(f"[分数门槛] 淘汰 {sym_id}: 评分{item_score:.1f} < {global_min:.0f}")
                    continue
                if item_score < single_min:
                    logger.info(f"[单引擎门槛] 淘汰 {sym_id}: 单引擎评分{item_score:.1f} < {single_min:.0f}")
                    continue
                filtered_item = self._apply_m3_hard_filter(item, sym_obj)
                if filtered_item.get("passed"):
                    filtered.append(filtered_item)
                else:
                    logger.info(f"[3m硬性过滤] 淘汰 {sym_id}: {(filtered_item.get('details') or {}).get('3m硬性过滤','')}")
            else:
                # 软模式或不启用硬过滤：应用分数门槛后保留
                item_score = float(item.get("score", 0) or 0)
                if item_score < global_min:
                    logger.info(f"[分数门槛] 淘汰 {sym_id}: 评分{item_score:.1f} < {global_min:.0f}")
                    continue
                if not is_consensus and item_score < single_min:
                    logger.info(f"[单引擎门槛] 淘汰 {sym_id}: 单引擎评分{item_score:.1f} < {single_min:.0f}")
                    continue
                if item.get("passed"):
                    filtered.append(item)
        final = filtered[:top_n]
        # v4.1: 信号持续时长过滤（去除首次闪现的信号）
        if bool(self.config.get("min_signal_persistence_bars", 0) or 0) > 0:
            final = self._apply_persistence_filter(final)
        # v4.2: 交易员视角过滤器（BTC环境/费率/共振/ATR止损/量质/相对强弱）
        final = self._apply_trading_filters(final, scan_symbols)
        self.last_child_results = by_strategy
        self.last_child_timing = timing
        child_total = sum(len(v) for v in by_strategy.values())
        child_breakdown = " | ".join(f"{n}={len(v)}" for n, v in by_strategy.items())
        pre_filter_count = len(output)
        logger.info(f"[组合] 扫描完成: 子策略产出 {child_total} 条 → "
              f"共识 {len(consensus_results)} 条 → 过滤后 {len(final)} 条 (扫描 {len(scan_symbols)} 个品种)")
        print(f"[AI五引擎诊断] 品种={len(scan_symbols)} 子策略产出={child_total}({child_breakdown}) "
              f"共识={len(consensus_results)} 去重前={pre_filter_count} 3m/1H过滤后={len(filtered)} 最终={len(final)}")
        result_payload = {
            "type": "ai_cross_section_triple_engine_combo",
            "all_opportunities": final,
            "child_counts": {n: len(v) for n, v in by_strategy.items()},
            "consensus_count": len(consensus_results),
            "dedupe_by_symbol": bool(self.config.get("dedupe_by_symbol", True)),
            "ultra_strict_mode": bool(self.config.get("ultra_strict_mode", False)),
            "scanned_symbols": len(scan_symbols),
            "input_symbols": len(symbols),
            "parallel_child_engines": bool(self.config.get("parallel_child_engines", False)),
            "child_timing_sec": timing if bool(self.config.get("profile_child_timing", True)) else {},
            "failed_engines": dict(getattr(self, "_failed_engines", {})),   # P1-5: 暴露失败子策略
            "mode": str(self.config.get("mode", "normal")),                 # P1-4: 模式
        }
        # P1-2 + P3-4: 扫描结束清空 H1 指标缓存（避免下一次扫描使用旧数据）
        self._symbol_indicator_cache.clear()
        return result_payload

    # ── 回测信号 ─────────────────────────────────────────────────────────────
    def generate_signal(self, data, *args, **kwargs):
        state_mode = self._normalize_state_mode(kwargs.pop("state_mode", None))
        inferred_state = self._infer_market_state_from_data(data)
        active_state = inferred_state if state_mode == "auto" else state_mode

        signals = []
        for sn, st, pri in self.child_strategies:
            if not hasattr(st, "generate_signal"):
                continue
            try:
                # 透传市场状态给子策略（如果子策略支持）
                child_kw = dict(kwargs)
                child_kw.setdefault("market_state", active_state)
                sig = st.generate_signal(data, *args, **child_kw)
            except Exception as e:
                logger.error(f"[组合] {sn} 回测信号失败: {e}")
                continue
            if not sig:
                continue
            norm = dict(sig)
            norm["source_strategy"] = sn
            norm["_priority"] = pri
            norm["_action_norm"] = "SHORT" if str(norm.get("action", "")).upper() == "SELL" else str(norm.get("action", "")).upper()
            base = float(norm.get("score", 0.0) or 0.0)
            bonus, mult = self._state_adjustment(sn, active_state)
            norm["score"] = max(0.0, min(100.0, (base + bonus) * mult))
            norm["_base_score"] = base
            signals.append(norm)

        if not signals:
            return None
        signals.sort(key=lambda x: (float(x.get("score", 0) or 0), int(x.get("_priority", 0) or 0)), reverse=True)

        min_consensus = int(self.config.get("min_consensus_engines", 2) or 2)
        pool = [s for s in signals if s.get("_action_norm") in {"BUY", "SHORT"}]
        counts: Dict[str, int] = {}
        for s in pool:
            a = str(s.get("_action_norm", ""))
            counts[a] = counts.get(a, 0) + 1

        if counts:
            best_action = max(counts, key=counts.get)
            support = counts.get(best_action, 0)
        else:
            best_action, support = "", 0

        if best_action in {"BUY", "SHORT"} and support >= min_consensus:
            aligned = [s for s in pool if s.get("_action_norm") == best_action]
            best = max(aligned, key=lambda s: float(s.get("score", 0) or 0))
            base_avg = sum(float(s.get("score", 0) or 0) for s in aligned) / len(aligned)

            # ── 共振加分（上限封顶，与 _build_consensus_result 一致）──
            if support >= 5:
                rb = min(12.0, float(self.config.get("triple_consensus_bonus", 13.0) or 13.0) * 0.90)
                rl = "五引擎同向最强共振"
            elif support >= 4:
                rb = min(10.0, float(self.config.get("triple_consensus_bonus", 13.0) or 13.0) * 0.75)
                rl = "四引擎同向强共振"
            elif support >= 3:
                rb = min(8.0, float(self.config.get("triple_consensus_bonus", 13.0) or 13.0) * 0.60)
                rl = "三引擎强共振"
            elif len(pool) > support:
                rb = min(5.0, float(self.config.get("consensus_bonus", 8.5) or 8.5) * 0.55)
                rl = "多引擎多数共振"
            else:
                rb = min(4.0, float(self.config.get("consensus_bonus", 8.5) or 8.5) * 0.45)
                rl = "双引擎同向共振"

            # ── 信息比率：引擎分数标准差越小 → 共识越强 ──
            aligned_scores = [float(s.get("score", 0) or 0) for s in aligned]
            score_std = float(np.std(aligned_scores)) if len(aligned_scores) >= 2 else 10.0
            if score_std > 1e-6:
                ir = base_avg / score_std
                ir_bonus = min(5.0, max(0.0, np.log1p(max(ir - 4, 0)) * 2.0))
            else:
                ir_bonus = 0.0

            # ── 争议惩罚 ──
            disagree_count = len(pool) - support
            cp = float(self.config.get("direction_conflict_penalty", 8.0) or 8.0) * (disagree_count / max(len(pool), 1)) * 0.5

            score = min(100.0, max(0.0, base_avg + rb + ir_bonus - cp))
            use_pilot = bool(self.config.get("use_pilot_add_system", True))
            ps = float(self.config.get("position_size", 0.02) or 0.02)
            result = {
                "action": best_action,
                "entry_price": float(best.get("entry_price", 0.0) or 0.0),
                "reason": f"{rl} | {score:.1f} | " + " / ".join(s["source_strategy"] for s in aligned)
                          + f" | 状态={active_state} | IR+{ir_bonus:.1f}-分歧{disagree_count}",
                "score": score, "raw_signals": signals,
                "consensus_engines": support, "state_mode": state_mode,
                "market_state": active_state, "inferred_market_state": inferred_state,
                "strategy_gates": {"h4_trend", "h1_trend", "rsi", "m3_pullback", "volume", "d1_trend", "entry_rule"},
                "timeframe_bias": f"multi_engine_{best_action}",
            }
            if not use_pilot:
                result["position_size"] = ps
            return result

        best = signals[0]
        min_score = float(self.config.get("backtest_min_score", 60.0) or 60.0)
        if float(best.get("score", 0) or 0) < min_score:
            return None
        best = dict(best)
        best.setdefault("strategy_gates", {"h4_trend", "h1_trend", "rsi", "m3_pullback", "volume"})
        best.update({"state_mode": state_mode, "market_state": active_state, "inferred_market_state": inferred_state})
        return best

    def reset_backtest_state(self):
        for _, st, _ in self.child_strategies:
            if hasattr(st, "reset_backtest_state"):
                try: st.reset_backtest_state()
                except Exception: pass
        self.last_child_results.clear()
        self.last_child_timing.clear()
        self._synced = False
        # P0-B2 修复：原代码不清理这些状态，回测多次跑会跨次累积污染
        self._engine_track.clear()
        self._signal_persistence.clear()
        self._performance_cache.clear()
        self._scan_counter = 0
        # P1-2 缓存清理（详见后续）
        if hasattr(self, "_symbol_indicator_cache"):
            self._symbol_indicator_cache.clear()

    # ── 内部方法 ─────────────────────────────────────────────────────────────
    def _apply_runtime_modes(self) -> None:
        # P1-4 修复：统一 mode 字段，废弃 ultra_strict_mode bool（仍兼容旧配置文件）
        mode = str(self.config.get("mode", "normal")).strip().lower()
        # 兼容：ultra_strict_mode=True 等价于 mode="ultra"
        if bool(self.config.get("ultra_strict_mode", False)) or mode in {"1","ultra","ultra_strict","strict","超严"}:
            mode = "ultra"
        # 规范化
        if mode not in {"normal", "ultra"}:
            mode = "normal"
        self.config["mode"] = mode
        # 同步旧字段（向后兼容）
        self.config["ultra_strict_mode"] = (mode == "ultra")
        if mode == "normal":
            return
        # ultra 模式参数收紧
        tn = max(6, min(9, int(self.config.get("ultra_target_top_n", 9) or 9)))
        self.config["min_score"] = max(float(self.config.get("min_score", 83.0) or 83.0), 86.0)
        self.config["backtest_min_score"] = max(float(self.config.get("backtest_min_score", 75.0) or 75.0), 78.0)
        self.config["top_n"] = min(int(self.config.get("top_n", tn) or tn), tn)
        self.config["top_n_per_strategy"] = min(int(self.config.get("top_n_per_strategy", 6) or 6), 5)
        self.config["min_consensus_engines"] = max(int(self.config.get("min_consensus_engines", 2) or 2), 2)
        self.config["direction_conflict_penalty"] = max(float(self.config.get("direction_conflict_penalty", 8.0) or 8.0), 9.0)
        self.config["include_consensus_results"] = True
        self.config["dedupe_by_symbol"] = True
        # P1-3: ultra 模式 z-score 提升到 1.04（前 15%）
        if bool(self.config.get("use_zscore_ranking", True)):
            self.config["min_zscore_threshold"] = max(
                float(self.config.get("min_zscore_threshold", 0.5)), 1.04)
        # P1-1: ultra 模式至少 3 个不同风格组同向
        if bool(self.config.get("style_grouped_consensus", True)):
            self.config["min_style_groups_consensus"] = max(
                int(self.config.get("min_style_groups_consensus", 2)), 3)

    def _select_scan_universe(self, symbols: List) -> List:
        if not bool(self.config.get("fast_scan_mode", False)):
            return list(symbols)
        limit = int(self.config.get("max_scan_symbols", 200) or 200)
        if limit <= 0 or len(symbols) <= limit:
            return list(symbols)
        return sorted(symbols, key=self._scan_universe_score, reverse=True)[:limit]

    def _scan_universe_score(self, symbol) -> float:
        vol = _safe_number(getattr(symbol, "volume_24h", 0.0))
        chg = abs(_safe_number(getattr(symbol, "price_change_24h", 0.0)))
        klines = getattr(symbol, "extra_data", {}).get("klines", {}) if getattr(symbol, "extra_data", None) else {}
        kb = sum(1.0 for bar in ("15m","1H","4H","1D") if klines.get(bar)) * 3.0
        h1m = _recent_kline_move_pct(klines.get("1H"))
        m15m = _recent_kline_move_pct(klines.get("15m"))
        # 动量质量分：趋势性波动 > 震荡噪音（用效率比衡量）
        h1_rows = klines.get("1H") or []
        momentum_quality = 0.0
        if len(h1_rows) >= 8:
            try:
                closes = [float(r[4]) for r in h1_rows[-8:] if float(r[4]) > 0]
                if len(closes) >= 6:
                    net_move = abs(closes[-1] / max(closes[0], 1e-9) - 1.0) * 100.0
                    path_sum = sum(abs(d) for d in pd.Series(closes).pct_change().dropna()) * 100.0
                    if path_sum > 0:
                        er = min(1.0, net_move / path_sum)  # 效率比：越接近1=趋势越干净
                        momentum_quality = er * 10.0
            except Exception: pass
        # 用 symbol hash 做 micro-tiebreaker，确保同分时结果稳定
        sym_hash = float(hash(str(getattr(symbol, "inst_id", ""))) % 1000) / 10000.0
        return (min(np.log10(max(vol,1.0)),11.0)*8 + min(chg,40.0)*1.8
                + min(abs(h1m),30.0)*2.4 + min(abs(m15m),20.0)*1.2
                + kb + momentum_quality + sym_hash)

    def _select_child_symbols(self, strategy_name: str, symbols: List) -> List:
        if strategy_name != "DRL小时趋势启动":
            return list(symbols)
        cap = int(self.config.get("drl_candidate_cap", 180) or 180)
        if cap <= 0 or len(symbols) <= cap:
            return list(symbols)
        return sorted(symbols, key=self._drl_candidate_score, reverse=True)[:cap]

    def _drl_candidate_score(self, symbol) -> float:
        vol = _safe_number(getattr(symbol, "volume_24h", 0.0))
        chg = abs(_safe_number(getattr(symbol, "price_change_24h", 0.0)))
        klines = (getattr(symbol, "extra_data", {}) or {}).get("klines", {})
        h1m = _recent_kline_move_pct(klines.get("1H"), 6)
        m15m = _recent_kline_move_pct(klines.get("15m"), 8)
        return min(np.log10(max(vol,1.0)),11.0)*8 + min(chg,40.0)*2.5 + min(abs(h1m),30.0)*3 + min(abs(m15m),20.0)*1.5

    def _timed_child_scan(self, sn, st, pri, symbols):
        child_syms = self._select_child_symbols(sn, symbols)
        t0 = time.perf_counter()
        results = self._run_child_scan(sn, st, pri, child_syms)
        return results, time.perf_counter() - t0

    def _run_child_scan(self, sn, st, pri, symbols):
        results = []
        sym_lookup = {str(getattr(s, "inst_id", "") or ""): s for s in symbols}
        try:
            if hasattr(st, "scan_all_symbols") and callable(st.scan_all_symbols):
                batch = st.scan_all_symbols(symbols)
                for item in (batch.get("all_opportunities", []) if isinstance(batch, dict) else []):
                    norm = self._annotate_child_result(item, sn, pri)
                    if norm.get("passed"):
                        results.append(norm)
                return results
        except Exception as e:
            logger.warning(f"[组合] {sn} 批量扫描失败，降级逐个: {e}")
        for sym in symbols:
            try:
                item = st.scan_symbol(sym)
            except Exception as e:
                logger.error(f"[组合] {sn} -> {getattr(sym,'inst_id','')} 失败: {e}")
                continue
            norm = self._annotate_child_result(item, sn, pri)
            if norm.get("passed"):
                results.append(norm)
        # v4.1: 子引擎结果在进入共识前先过3m预过滤
        if bool(self.config.get("prefilter_3m_before_consensus", False)) and results:
            filtered_results = []
            for r in results:
                sym_id = str(r.get("symbol", ""))
                sym_obj = sym_lookup.get(sym_id)
                if not sym_obj:
                    filtered_results.append(r)
                    continue
                r_dir = str(r.get("direction", "WAIT")).upper()
                if r_dir not in {"BUY", "SELL"}:
                    filtered_results.append(r)
                    continue
                m3_rows = self._get_m3_klines(sym_obj)
                m3_ok, m3_reason = self._m3_pullback_hard_check(m3_rows, r_dir)
                if m3_ok:
                    r["details"] = dict(r.get("details") or {})
                    r["details"]["3m预过滤"] = f"通过: {m3_reason}"
                    filtered_results.append(r)
            return filtered_results
        return results

    def _annotate_child_result(self, result, sn, priority):
        norm = dict(result or {})
        if not norm:
            return norm
        norm["source_strategy"] = sn
        norm["category"] = f"{sn} | {norm.get('category', norm.get('strategy_category', '扫描机会'))}"
        norm["strategy_category"] = norm["category"]
        norm["group_sort_score"] = priority
        details = norm.get("details")
        if isinstance(details, dict):
            details.setdefault("来源策略", sn)
            details.setdefault("机会类型", norm["category"])
        signals = list(norm.get("signals", []) or [])
        if not signals:
            signals = [norm["category"]]
        elif sn not in str(signals[0]):
            signals[0] = f"{sn}: {signals[0]}"
        norm["signals"] = signals
        if enrich_scan_result:
            try: enrich_scan_result(norm)
            except Exception: pass
        return norm

    def _build_consensus_results(self, child_results):
        by_sym: Dict[str, List] = {}
        for item in child_results:
            by_sym.setdefault(str(item.get("symbol", "")), []).append(item)
        return [r for r in (self._build_consensus_result(sym, items) for sym, items in by_sym.items()) if r]

    def _build_consensus_result(self, symbol: str, items: List) -> Optional[Dict[str, Any]]:
        min_eng = int(self.config.get("min_consensus_engines", 2) or 2)
        if len(items) < min_eng:
            return None
        strategies = {item.get("source_strategy") for item in items}
        if len(strategies) < min_eng:
            return None
        dirs = [str(item.get("direction","WAIT")).upper() for item in items if str(item.get("direction","WAIT")).upper() in {"BUY","SELL"}]
        if not dirs:
            return None
        dc = {d: dirs.count(d) for d in {"BUY","SELL"}}
        direction = max(dc, key=dc.get)
        support = int(dc.get(direction, 0))
        disagree = max(0, len(dirs) - support)
        if support < min_eng:
            return None

        # ── P1-1: 风格分组共识检查 ────────────────────────────────────
        # 当前 5 引擎中 3 个是因子模型（截面/AI挖掘/XGBoost），高度相关。
        # 即使 3 个都同向也只算 1 个独立信号。改为按风格组投票：
        #   factor / momentum / orderflow 是 3 类完全独立的方法
        #   要求至少 N 个不同风格组同向（默认 2）
        style_grouped_on = bool(self.config.get("style_grouped_consensus", True))
        min_style_groups = int(self.config.get("min_style_groups_consensus", 2) or 2)
        n_styles, style_counts = self._count_style_groups(items, direction)
        if style_grouped_on and n_styles < min_style_groups:
            # 同风格内多次同向 ≠ 真共识
            logger.debug(f"[共识] {symbol} {direction} 仅 {n_styles} 个风格组({style_counts})，需≥{min_style_groups}")
            return None

        # ── v4.4: 统一市场状态推断 ──────────────────────────────────
        active_state = self._infer_consensus_state(items)
        # 提取时效性数据（用于半衰期衰减）
        max_trend_age = 0
        max_staleness = 0
        for it in items:
            d = it.get("details") if isinstance(it.get("details"), dict) else {}
            age_v = _safe_number(d.get("1H趋势延续根数", "0"), 0)
            stale_v = _safe_number(d.get("3分钟时效(根)", "0"), 0)
            if age_v > max_trend_age: max_trend_age = int(age_v)
            if stale_v > max_staleness: max_staleness = int(stale_v)

        # ── 基础评分：引擎动态胜率加权均值 ─────────────────────────
        child_scores = []
        if bool(self.config.get("enable_engine_track_record", True)):
            eng_weights = {}
            for it in items:
                sn = str(it.get("source_strategy", ""))
                d = str(it.get("direction", "WAIT")).upper()
                eng_weights[sn] = self._get_engine_weight(sn, d)
            weighted = sum(
                float(it.get("score", 0) or 0) * eng_weights.get(it.get("source_strategy", ""), 1.0)
                for it in items
            )
            weight_sum = sum(eng_weights.get(it.get("source_strategy", ""), 1.0) for it in items)
            if weight_sum > 0:
                base_score = weighted / weight_sum
            else:
                base_score = sum(float(it.get("score", 0) or 0) for it in items) / len(items)
            # 同时收集加权后分数用于 IR 计算
            for it in items:
                sn = str(it.get("source_strategy", ""))
                w = eng_weights.get(sn, 1.0)
                child_scores.append(float(it.get("score", 0) or 0) * w)
        else:
            raw_scores = [float(it.get("score", 0) or 0) for it in items]
            base_score = sum(raw_scores) / len(items)
            child_scores = list(raw_scores)

        opp_score = base_score * 0.98

        # ── v4.4: 统一市场状态调整（仅一套权重，取代旧的 STATE_WEIGHTS + _state_adjustment）──
        if bool(self.config.get("enable_state_conditional_weights", True)):
            total_bonus = 0.0
            total_mult = 1.0
            count = 0
            for it in items:
                sn = str(it.get("source_strategy", ""))
                bonus, mult = self._state_adjustment(sn, active_state)
                total_bonus += bonus
                total_mult += mult
                count += 1
            avg_bonus = total_bonus / max(count, 1)
            avg_mult = total_mult / max(count, 1)
            base_score = base_score + avg_bonus * 0.5  # 削弱奖金影响力（避免与共振 bonus 叠加过猛）
            base_score = base_score * (1.0 + (avg_mult - 1.0) * 0.3)
            opp_score = base_score * 0.97

        # ── v5.0: 趋势半衰期衰减（更激进，着重惩罚老趋势） ──────────
        if max_trend_age > 0:
            age_limit = int(self.config.get("max_h1_trend_age", 8) or 8)
            # v5.0: 半衰期缩短到 0.35×age_limit，更快衰减
            half_life = max(age_limit * 0.35, 2.5)
            decay = 2.0 ** (-max_trend_age / half_life)
            # v5.0: 保底 0.2（原来0.3），老信号降权更强
            freshness_factor = 0.2 + 0.8 * decay
            base_score *= freshness_factor
            opp_score *= freshness_factor

        # ── v4.4: 信息比率排名（引擎共识度） ───────────────────────
        # P0-B10 修复：原代码 ≥2 引擎就计算 IR，但样本=2时 std 极小 → IR 极大 → 总满分 8
        # 改为 ≥3 引擎才算 IR；2 引擎时仅看均值不奖励
        consensus_ir_bonus = 0.0
        if len(child_scores) >= 3:
            score_std = float(np.std(child_scores))
            if score_std > 1e-6:
                ir = base_score / score_std  # 高均值 + 低方差 = 引擎高度一致
                # 将 IR 映射到 bonus：IR=5 → +2, IR=10 → +5, IR=20 → +8
                consensus_ir_bonus = min(8.0, max(0.0, np.log1p(max(ir - 3, 0)) * 2.5))
                base_score += consensus_ir_bonus
                opp_score += consensus_ir_bonus * 0.8

        # ── v4.4: 引擎共振加分（上限封顶，防止通胀） ─────────
        # 加分规则：支持引擎越多加分越高，但有上限
        if support >= 5:
            rb = min(15.0, float(self.config.get("triple_consensus_bonus", 13.0) or 13.0) * 1.12)
            signal_head = "五引擎同向强共振"; category = "五引擎强共振"; gss = 1400
        elif support >= 4:
            rb = min(13.0, float(self.config.get("triple_consensus_bonus", 13.0) or 13.0))
            signal_head = "四引擎同向强共振"; category = "四引擎强共振"; gss = 1370
        elif support >= 3:
            rb = min(11.0, float(self.config.get("triple_consensus_bonus", 13.0) or 13.0) * 0.85)
            signal_head = "三引擎同向强共振"; category = "三引擎强共振"; gss = 1350
        elif disagree == 0:
            rb = min(7.0, float(self.config.get("consensus_bonus", 8.5) or 8.5) * 0.82)
            signal_head = "双引擎同向共振"; category = "双引擎共振"; gss = 1240
        else:
            rb = min(5.0, float(self.config.get("consensus_bonus", 8.5) or 8.5) * 0.55)
            signal_head = "多引擎多数共振(含分歧)"; category = "多引擎多数共振"; gss = 1180

        # 方向分歧惩罚
        # P0-B9 修复：原 cp = penalty × ratio × 0.8，分歧 2/5 时仅扣 2.6 分太弱
        # 改为：基础扣 4.0，每多 1 个反向引擎再扣 3.0
        if disagree > 0:
            cp = 4.0 + (disagree - 1) * 3.0
            # 仍按可配置上限封顶
            cp = min(cp, float(self.config.get("direction_conflict_penalty", 8.0) or 8.0) * 2.0)
        else:
            cp = 0.0
        disagreement_ratio = disagree / max(len(dirs), 1)

        # ── 最终评分：基础分 + 共振加分 + IR bonus − 分歧惩罚 ──
        raw_score = base_score + rb - cp
        # 加分项总计封顶（防止 trading_filters 后续再加分导致通胀）
        score = min(100.0, max(0.0, raw_score))
        opp_score = min(100.0, max(0.0, opp_score + rb - cp))
        passed = score >= float(self.config.get("min_score", 68.0) or 68.0)
        # 记录评分构成供诊断
        score_detail = {
            "base": round(float(base_score), 2),
            "resonance_bonus": round(float(rb), 2),
            "conflict_penalty": round(float(cp), 2),
            "ir_bonus": round(float(consensus_ir_bonus), 2),
            "state": active_state,
            "trend_age": max_trend_age,
        }

        src_names = [str(item.get("source_strategy","")) for item in items]
        lp = _first_number(items, "last_price")
        v24 = _first_number(items, "volume_24h")
        pc24 = _first_number(items, "price_change_24h")

        # v2: 汇总时效性信息
        age_infos = []
        stale_infos = []
        # v3.2: 汇总微观结构信息
        micro_infos = []
        for item in items:
            d = item.get("details") if isinstance(item.get("details"), dict) else {}
            age_s = str(d.get("1H趋势延续根数","")).strip()
            stale_s = str(d.get("3分钟时效(根)","")).strip()
            micro_s = str(d.get("3m微观指标", d.get("微观指标", ""))).strip()
            if age_s: age_infos.append(f"{item.get('source_strategy')}={age_s}")
            if stale_s: stale_infos.append(f"{item.get('source_strategy')}={stale_s}")
            if micro_s: micro_infos.append(f"{item.get('source_strategy')}:{micro_s}")

        m3_states = []
        for item in items:
            d = item.get("details") if isinstance(item.get("details"), dict) else {}
            st = str(d.get("3分钟结构","") or "").strip()
            if st: m3_states.append(f"{item.get('source_strategy')}={st}")

        signals = [
            f"{signal_head} {score:.1f}",
            f"同向支持引擎: {support}/{len(items)}",
            "来源: " + " / ".join(src_names),
            "子策略评分: " + " / ".join(f"{item.get('source_strategy')}={float(item.get('score',0) or 0):.1f}" for item in items),
        ]
        if m3_states: signals.append("3m结构: " + " / ".join(m3_states))
        if age_infos: signals.append("1H趋势时效: " + " / ".join(age_infos))
        if micro_infos: signals.append("3m微观: " + " / ".join(micro_infos))  # v3.2

        result = {
            "symbol": symbol, "passed": passed,
            "score": round(score, 2), "opportunity_score": round(opp_score, 2),
            "direction": direction, "signals": signals,
            "category": category, "strategy_category": category,
            "source_strategy": "多引擎共振",
            "group_sort_score": gss,
            "last_price": lp, "volume_24h": v24, "price_change_24h": pc24,
            "child_results": items, "consensus_engines": support,
            "details": {
                "机会类型": category,
                "来源策略": " / ".join(src_names),
                "主方向": direction,
                "同向支持引擎": f"{support}/{len(items)}",
                "方向分歧数": str(disagree),
                "评分构成": f"基础{score_detail['base']:.1f}+共振{score_detail['resonance_bonus']:.1f}"
                            f"+IR{score_detail['ir_bonus']:.1f}-分歧{score_detail['conflict_penalty']:.1f}"
                            f" | 状态={score_detail['state']} 时效={score_detail['trend_age']}根",
                "风格分组": f"{n_styles}组({','.join(f'{k}={v}' for k,v in style_counts.items())})",
                "子策略评分": " / ".join(f"{item.get('source_strategy')}={float(item.get('score',0) or 0):.1f}" for item in items),
                "3分钟结构": " / ".join(m3_states) if m3_states else "-",
                "1H趋势时效": " / ".join(age_infos) if age_infos else "-",   # v2
                "3m回调时效": " / ".join(stale_infos) if stale_infos else "-", # v2
                "3m微观指标": " / ".join(micro_infos) if micro_infos else "-", # v3.2
                "评估": " | ".join(signals),
            },
            "ranking_factors": _merge_ranking_factors(items, score),
        }
        if build_opportunity_profile:
            try: result.update(build_opportunity_profile(score, direction, v24, result["ranking_factors"], signals))
            except Exception: pass
        # v4.1: 历史绩效标签
        perf_label = self._get_signal_performance_label(symbol, direction)
        if perf_label != "无历史记录":
            result.setdefault("details", {})
            result["details"]["历史绩效"] = perf_label

        return result if result.get("passed") else None

    # ── v4.1: 信号持久度过滤 ──────────────────────────────────────────────────
    def _apply_persistence_filter(self, results: List[Dict]) -> List[Dict]:
        """过滤闪现信号：同一币种同方向需在连续N次扫描中出现"""
        min_bars = max(1, int(self.config.get("min_signal_persistence_bars", 3) or 3))
        self._scan_counter += 1
        passed = []
        for item in results:
            sym = str(item.get("symbol", ""))
            d = str(item.get("direction", "WAIT")).upper()
            key = f"{sym}:{d}"
            last_seen = self._signal_persistence.get(key, -999)
            if self._scan_counter - last_seen <= 1:
                # 连续出现：累计计数
                streak = self._signal_persistence.get(f"{key}:streak", 0) + 1
                self._signal_persistence[f"{key}:streak"] = streak
                self._signal_persistence[key] = self._scan_counter
                if streak >= min_bars:
                    passed.append(item)
                    item.setdefault("_persistence_streak", streak)
            else:
                # 首次出现或中断：重置计数
                self._signal_persistence[f"{key}:streak"] = 1
                self._signal_persistence[key] = self._scan_counter
                if min_bars <= 1:
                    passed.append(item)
                    item.setdefault("_persistence_streak", 1)
                else:
                    logger.debug(f"[持续检查] {sym} {d} 首次出现(需连续{min_bars}次扫描确认)")
        return passed

    def report_trade_outcome(self, source_strategy: str, direction: str, pnl: float):
        """外部（回测引擎/实盘）调用：报告一笔交易的实际结果。

        Args:
            source_strategy: 子策略名称（如"DRL小时趋势启动"）
            direction: "BUY" 或 "SELL"
            pnl: 实际收益率（小数，如 0.03 = 3%）
        """
        if not bool(self.config.get("enable_engine_track_record", True)):
            return
        d = str(direction).upper()
        if d == "SELL": d = "SHORT"
        self._record_engine_performance(source_strategy, d, float(pnl))
        if d in {"BUY", "SHORT"}:
            ws = [self._get_engine_weight(sn, d)
                  for sn, _, _ in self.child_strategies]
            logger.debug(f"[引擎权重] {source_strategy} {d} PnL={pnl:+.3f} → 胜率权重: "
                        f"{' / '.join(f'{sn}={w:.2f}' for (sn,_,_), w in zip(self.child_strategies, ws))}")

    def _record_performance_for_scan_results(self, results: List[Dict]) -> None:
        """批量初始化引擎追踪条目（PnL=0.0，由 report_trade_outcome 更新）"""
        for item in results:
            child_results = item.get("child_results") if isinstance(item.get("child_results"), list) else []
            for cr in child_results:
                sn = str(cr.get("source_strategy", ""))
                cd = str(cr.get("direction", "WAIT")).upper()
                if sn and cd in {"BUY", "SELL"}:
                    # 只在首次出现时初始化，避免覆盖已有数据
                    track = self._engine_track.setdefault(sn, {}).setdefault(cd, [])
                    if not track:
                        pass  # 等 report_trade_outcome 来填充

    # ═══════════════════════════════════════════════════════════════════════════
    # v4.2 交易员视角 — 风险/市场环境过滤器
    # ═══════════════════════════════════════════════════════════════════════════

    # ── 1. BTC 相关性过滤 ────────────────────────────────────────────────────
    def _get_btc_context(self, symbols: List) -> Dict[str, float]:
        """从扫描品种列表中提取 BTC 基准数据"""
        btc = next((s for s in symbols if str(getattr(s, "inst_id", "")).upper() in
                     {"BTC-USDT-SWAP", "BTC-USDT"}), None)
        if not btc:
            return {}
        klines = (getattr(btc, "extra_data", {}) or {}).get("klines", {})
        h1 = klines.get("1H") or klines.get("1h") or []
        h4 = klines.get("4H") or klines.get("4h") or []
        d1 = klines.get("1D") or klines.get("1d") or []
        ctx = {}
        # BTC 1H 涨跌
        if len(h1) >= 2:
            try:
                ctx["btc_1h_move"] = (float(h1[-1][4]) / float(h1[-2][4]) - 1.0) * 100
            except: pass
        # BTC 4H EMA 状态
        if len(h4) >= 50:
            try:
                closes = [float(r[4]) for r in h4 if float(r[4]) > 0]
                ema20 = pd.Series(closes).ewm(span=20, adjust=False).mean().iloc[-1]
                ema50 = pd.Series(closes).ewm(span=50, adjust=False).mean().iloc[-1]
                ctx["btc_4h_bullish"] = float(ema20 > ema50)
            except: pass
        # BTC 24h 涨跌 (from ticker)
        ctx["btc_24h_move"] = _safe_number(getattr(btc, "price_change_24h", 0))
        return ctx

    def _apply_btc_correlation_filter(self, results: List[Dict], btc_ctx: Dict) -> List[Dict]:
        """BTC暴跌时降权山寨币多头信号"""
        if not bool(self.config.get("enable_btc_correlation_filter", True)) or not btc_ctx:
            return results
        threshold = float(self.config.get("btc_dump_threshold_pct", -2.5) or -2.5)
        penalty = float(self.config.get("btc_dump_penalty", 12.0) or 12.0)
        btc_move = btc_ctx.get("btc_1h_move", btc_ctx.get("btc_24h_move", 0))
        btc_4h_bull = btc_ctx.get("btc_4h_bullish", 1.0)
        is_dump = btc_move < threshold
        for item in results:
            sym = str(item.get("symbol", "")).upper()
            if "BTC" in sym:
                continue  # BTC自身不受影响
            direction = str(item.get("direction", "WAIT")).upper()
            if is_dump and direction in {"BUY", "LONG"}:
                item["score"] = round(max(0, float(item.get("score", 0) or 0) - penalty), 2)
                item["_btc_penalty"] = True
                d = item.setdefault("details", {})
                d["BTC环境"] = f"⚠ BTC 1H跌{btc_move:.1f}%，山寨多头降{penalty}分"
            elif not is_dump and btc_4h_bull:
                # BTC 稳定或上涨：不加分也不扣分，仅标注
                d = item.setdefault("details", {})
                d["BTC环境"] = f"✓ BTC 4H多头确认，山寨多头环境安全"
        return results

    # ── 2. 资金费率过滤 ──────────────────────────────────────────────────────
    def _apply_funding_filter(self, results: List[Dict], symbols: List) -> List[Dict]:
        """极端资金费率时降权同向信号（多头拥挤→降权多头，空头拥挤→降权空头）"""
        if not bool(self.config.get("enable_funding_filter", True)):
            return results
        threshold = float(self.config.get("funding_extreme_threshold", 0.10) or 0.10) / 100.0
        penalty = float(self.config.get("funding_penalty", 8.0) or 8.0)
        # P0-B1 修复：原代码用 next() 嵌套循环 O(N²)。改为预构建 sym_map 与其他过滤器一致
        sym_map = {str(getattr(s, "inst_id", "")): s for s in symbols}
        for item in results:
            sym_id = str(item.get("symbol", ""))
            sym = sym_map.get(sym_id)
            if not sym:
                continue
            funding = _safe_number(
                (getattr(sym, "extra_data", {}) or {}).get("funding_rate", 0)
            )
            if abs(funding) < threshold:
                continue
            direction = str(item.get("direction", "WAIT")).upper()
            d = item.setdefault("details", {})
            if funding > threshold and direction in {"BUY", "LONG"}:
                item["score"] = round(max(0, float(item.get("score", 0) or 0) - penalty), 2)
                d["资金费率"] = f"⚠ 多头拥挤({funding*100:.3f}%)，降{penalty}分"
            elif funding < -threshold and direction in {"SELL", "SHORT"}:
                item["score"] = round(max(0, float(item.get("score", 0) or 0) - penalty), 2)
                d["资金费率"] = f"⚠ 空头拥挤({funding*100:.3f}%)，降{penalty}分"
            else:
                d["资金费率"] = f"正常({funding*100:.3f}%)"
        return results

    # ── 3. 多周期趋势共振评分 ───────────────────────────────────────────────
    def _get_symbol_klines_indicators(self, symbol, bars=("1H","4H","1D")) -> Dict[str, float]:
        """提取单个品种的多周期EMA/ADX指标"""
        klines = (getattr(symbol, "extra_data", {}) or {}).get("klines", {})
        ind = {}
        for bar in bars:
            rows = klines.get(bar) or []
            if len(rows) < 26:
                continue
            try:
                closes = [float(r[4]) for r in rows if float(r[4]) > 0]
                if len(closes) < 26:
                    continue
                s = pd.Series(closes)
                ema12 = float(s.ewm(span=12, adjust=False).mean().iloc[-1])
                ema26 = float(s.ewm(span=26, adjust=False).mean().iloc[-1])
                ind[f"{bar}_bullish"] = 1.0 if ema12 > ema26 else 0.0
                ind[f"{bar}_gap_pct"] = abs(ema12 - ema26) / max(ema26, 1e-9) * 100
                # 简版ADX
                if len(closes) >= 28:
                    tr_vals = [max(closes[i]-closes[i-1], closes[i-1]-closes[i], 0)
                               for i in range(1, len(closes))]
                    if len(tr_vals) >= 14:
                        atr14 = float(pd.Series(tr_vals).rolling(14).mean().iloc[-1])
                        last_px = closes[-1]
                        if atr14 > 0 and last_px > 0:
                            ind[f"{bar}_adx_pct"] = atr14 / last_px * 100
            except: pass
        return ind

    def _apply_confluence_scoring(self, results: List[Dict], symbols: List) -> List[Dict]:
        """多周期EMA金叉/死叉共振加分：根据方向对称处理"""
        if not bool(self.config.get("enable_confluence_scoring", True)):
            return results
        max_bonus = float(self.config.get("confluence_bonus_max", 6.0) or 6.0)
        sym_map = {str(getattr(s, "inst_id", "")): s for s in symbols}
        for item in results:
            sym_id = str(item.get("symbol", ""))
            sym = sym_map.get(sym_id)
            if not sym:
                continue
            direction = str(item.get("direction", "WAIT")).upper()
            ind = self._get_symbol_klines_indicators(sym)
            if not ind:
                continue
            # 根据方向计算共振：多头=金叉数，空头=死叉数
            bullish = [ind.get(f"{b}_bullish", 0) for b in ("1H","4H","1D")]
            if direction in {"SELL", "SHORT"}:
                aligned_count = sum(1 for v in bullish if v == 0.0)  # 死叉=EMA12<EMA26
            else:
                aligned_count = sum(1 for v in bullish if v == 1.0)  # 金叉
            if aligned_count >= 3:
                bonus = max_bonus
                label = "★★★ 1H/4H/日线三周期共振"
            elif aligned_count >= 2:
                bonus = max_bonus * 0.60
                label = "★★ 双周期共振"
            else:
                bonus = 0
                label = f"★ {aligned_count}周期对齐"
            if bonus > 0:
                item["score"] = round(min(100, float(item.get("score", 0) or 0) + bonus), 2)
            d = item.setdefault("details", {})
            gaps = " | ".join(f"{b}={ind.get(b+'_gap_pct',0):.1f}%" for b in ("1H","4H","1D") if f"{b}_gap_pct" in ind)
            d["周期共振"] = f"{label} ({'金叉' if direction in {'BUY','LONG'} else '死叉'}) (+{bonus:.1f}分) [{gaps}]"

        return results

    # ── 4. ATR 动态止损建议 ──────────────────────────────────────────────────
    def _get_atr(self, rows: List, period: int = 14) -> float:
        """从原始K线列表计算 ATR"""
        if not rows or len(rows) < period + 2:
            return 0.0
        trs = []
        for i in range(1, min(len(rows), period + 12)):
            try:
                h, l, pc = float(rows[-i][2]), float(rows[-i][3]), float(rows[-i-1][4])
                trs.append(max(h - l, abs(h - pc), abs(l - pc)))
            except: pass
        if not trs:
            return 0.0
        return float(pd.Series(trs).ewm(span=period, adjust=False).mean().iloc[-1])

    def _apply_atr_stop_suggestion(self, results: List[Dict], symbols: List) -> List[Dict]:
        """为每条信号附加基于15m ATR的动态止损建议"""
        if not bool(self.config.get("enable_atr_stop_suggestion", True)):
            return results
        mult = float(self.config.get("stop_atr_multiplier", 1.8) or 1.8)
        sym_map = {str(getattr(s, "inst_id", "")): s for s in symbols}
        for item in results:
            sym_id = str(item.get("symbol", ""))
            sym = sym_map.get(sym_id)
            if not sym:
                continue
            klines = (getattr(sym, "extra_data", {}) or {}).get("klines", {})
            m15 = klines.get("15m") or klines.get("15M") or []
            atr = self._get_atr(m15)
            if atr <= 0:
                rows_3m = klines.get("3m") or []
                atr = self._get_atr(rows_3m, 14) * 2.0  # 3m ATR ×2 近似15m ATR
            if atr <= 0:
                continue
            lp = _safe_number(getattr(sym, "last_price", item.get("last_price", 0)))
            if lp <= 0:
                continue
            direction = str(item.get("direction", "WAIT")).upper()
            sl_price = lp - atr * mult if direction in {"BUY", "LONG"} else lp + atr * mult
            sl_pct = atr * mult / lp * 100
            d = item.setdefault("details", {})
            d["ATR止损建议"] = f"{sl_price:.6g} ({sl_pct:.2f}% | {mult}×ATR={atr:.6g})"
        return results

    # ── 5. 成交量质量 ────────────────────────────────────────────────────────
    def _apply_volume_quality_check(self, results: List[Dict], symbols: List) -> List[Dict]:
        """检测刷量：超过60%成交量集中在单根K线 → 降权"""
        if not bool(self.config.get("enable_volume_quality_check", True)):
            return results
        threshold = float(self.config.get("vol_conc_ratio_threshold", 0.60) or 0.60)
        sym_map = {str(getattr(s, "inst_id", "")): s for s in symbols}
        for item in results:
            sym_id = str(item.get("symbol", ""))
            sym = sym_map.get(sym_id)
            if not sym:
                continue
            klines = (getattr(sym, "extra_data", {}) or {}).get("klines", {})
            h1 = klines.get("1H") or klines.get("1h") or []
            if len(h1) < 12:
                continue
            try:
                vols = []
                for r in h1[-12:]:
                    v = float(r[5]) if len(r) > 5 else 0
                    if v > 0: vols.append(v)
                if len(vols) >= 8:
                    max_vol = max(vols)
                    total = sum(vols)
                    conc = max_vol / max(total, 1e-9)
                    d = item.setdefault("details", {})
                    if conc > threshold:
                        item["score"] = round(max(0, float(item.get("score", 0) or 0) - 4.0), 2)
                        d["成交量质量"] = f"⚠ 集中度{conc:.0%}(>{threshold:.0%})，疑似刷量，降4分"
                    else:
                        d["成交量质量"] = f"✓ 正常(集中度{conc:.0%})"
            except: pass
        return results

    # ── 6. 相对BTC强弱 ──────────────────────────────────────────────────────
    def _apply_btc_relative_strength(self, results: List[Dict], symbols: List, btc_ctx: Dict) -> List[Dict]:
        """山寨弱于BTC时标记或降权"""
        if not bool(self.config.get("enable_btc_relative_strength", True)) or not btc_ctx:
            return results
        min_ratio = float(self.config.get("btc_rs_min_ratio", -0.3) or -0.3)
        btc_24h = btc_ctx.get("btc_24h_move", 0)
        sym_map = {str(getattr(s, "inst_id", "")): s for s in symbols}
        for item in results:
            sym_id = str(item.get("symbol", ""))
            if "BTC" in sym_id.upper():
                continue
            sym = sym_map.get(sym_id)
            if not sym:
                continue
            alt_24h = _safe_number(getattr(sym, "price_change_24h", 0))
            if abs(btc_24h) < 0.5:
                continue  # BTC 静止时不做比较
            rs = alt_24h - btc_24h  # 山寨涨跌 - BTC涨跌
            d = item.setdefault("details", {})
            direction = str(item.get("direction", "WAIT")).upper()
            if direction in {"BUY", "LONG"} and rs < min_ratio:
                penalty = min(8.0, abs(rs - min_ratio) * 2.0)
                item["score"] = round(max(0, float(item.get("score", 0) or 0) - penalty), 2)
                d["相对BTC"] = f"⚠ BTC+{btc_24h:.1f}% vs 山寨{alt_24h:+.1f}%(差{rs:.1f}%)，弱于大盘降{penalty:.1f}分"
            elif direction in {"SELL", "SHORT"} and rs > -min_ratio:
                d["相对BTC"] = f"山寨{alt_24h:+.1f}% vs BTC+{btc_24h:.1f}%(差{rs:.1f}%)，相对强势，注意空头风险"
            else:
                d["相对BTC"] = f"山寨{alt_24h:+.1f}% vs BTC+{btc_24h:.1f}%(差{rs:.1f}%)"

        return results

    # ═══════════════════════════════════════════════════════════════════════════
    # v4.3 交易员视角 — 1H 小时线突破检测
    # ═══════════════════════════════════════════════════════════════════════════

    def _h1_breakout_detect(self, symbol) -> Tuple[bool, float, str]:
        """
        检测1H级别突破：
          ① 当前收盘突破最近 N 根1H K线最高点
          ② 突破K线成交量显著放大(>1.35x均量)
          ③ K线收盘在K线上半部(强势收盘)
        返回: (是否突破, 置信度0~1, 诊断描述)
        """
        if not bool(self.config.get("enable_h1_breakout_detect", True)):
            return False, 0.0, "未启用"
        rows = self._get_h1_klines(symbol)
        lookback = int(self.config.get("h1_breakout_lookback", 12) or 12)
        vol_ratio = float(self.config.get("h1_breakout_vol_ratio", 1.35) or 1.35)
        close_pos = float(self.config.get("h1_breakout_close_position", 0.65) or 0.65)
        if len(rows) < lookback + 3:
            return False, 0.0, f"1H数据不足({len(rows)}根，需{lookback+3})"

        def rv(row, idx):
            try: return float(row[idx])
            except: return 0.0

        # 最近一根K线
        cur_open, cur_high, cur_low, cur_close, cur_vol = rv(rows[-1],1), rv(rows[-1],2), rv(rows[-1],3), rv(rows[-1],4), rv(rows[-1],5)

        # 前 lookback 根K线(不含最新一根)的最高点
        prev_rows = rows[-lookback-1:-1]
        prev_high = max((rv(r, 2) for r in prev_rows), default=0.0)
        if prev_high <= 0:
            return False, 0.0, "无有效历史高点"

        # ① 突破检测
        breakthrough = cur_close > prev_high * 1.001 and cur_high > prev_high
        if not breakthrough:
            return False, 0.0, f"收盘{cur_close:.6g}未突破{lookback}根高点{prev_high:.6g}"

        # ② 放量检测
        # P2-1: 改用历史百分位（默认前 25%）替代固定倍数，适应不同币种的常态分布
        use_percentile = bool(self.config.get("h1_breakout_vol_use_percentile", True))
        if use_percentile and len(rows) >= 50:
            vols_hist = [rv(r, 5) for r in rows[-50:] if rv(r, 5) > 0]
            if len(vols_hist) >= 20:
                vol_pctile_thr = float(self.config.get("h1_breakout_vol_percentile", 0.75))
                vol_thr_value = float(np.percentile(vols_hist, vol_pctile_thr * 100))
                avg_vol = float(np.mean(vols_hist))
                vol_surge = cur_vol >= vol_thr_value
            else:
                # 历史不足时退回旧逻辑
                prev_vols = [rv(r, 5) for r in prev_rows[-8:] if rv(r, 5) > 0]
                avg_vol = sum(prev_vols) / len(prev_vols) if prev_vols else cur_vol
                vol_surge = cur_vol >= avg_vol * vol_ratio
        else:
            prev_vols = [rv(r, 5) for r in prev_rows[-8:] if rv(r, 5) > 0]
            avg_vol = sum(prev_vols) / len(prev_vols) if prev_vols else cur_vol
            vol_surge = cur_vol >= avg_vol * vol_ratio

        # ③ 强势收盘
        candle_range = cur_high - cur_low
        if candle_range > 0:
            position = (cur_close - cur_low) / candle_range
        else:
            position = 0.5
        strong_close = position >= close_pos and cur_close >= cur_open

        # 综合判断
        checks = [breakthrough, vol_surge, strong_close]
        passed = sum(checks)
        conf = passed / 3.0

        details = (
            f"突破{lookback}根高点{prev_high:.6g}→{cur_close:.6g}(+{(cur_close/prev_high-1)*100:.2f}%)"
            f" | 量{cur_vol:.0f}vs均{avg_vol:.0f}({cur_vol/avg_vol:.1f}x)"
            f" | 收盘位{position:.0%}"
            f" | 评分{passed}/3"
        )
        return True, conf, details

    def _h1_squeeze_detect(self, symbol) -> Tuple[bool, float, str]:
        """
        布林带压缩检测：带宽处于历史低位 → 即将突破
        """
        if not bool(self.config.get("enable_h1_squeeze_detect", True)):
            return False, 0.0, "未启用"
        rows = self._get_h1_klines(symbol)
        period = int(self.config.get("h1_squeeze_bb_period", 20) or 20)
        pctile = float(self.config.get("h1_squeeze_bb_width_percentile", 0.20) or 0.20)
        if len(rows) < period * 3:
            return False, 0.0, f"1H数据不足({len(rows)}根，需{period*3})"

        try:
            closes = []
            for r in rows[-period*3:]:
                c = float(r[4]) if float(r[4]) > 0 else None
                if c: closes.append(c)
            if len(closes) < period * 2:
                return False, 0.0, "有效收盘不足"

            s = pd.Series(closes)
            ma = s.rolling(period).mean()
            std = s.rolling(period).std()
            bbw = (std * 2) / ma  # 布林带宽

            cur_bbw = float(bbw.iloc[-1])
            if pd.isna(cur_bbw) or cur_bbw <= 0:
                return False, 0.0, "带宽计算异常"

            # 计算带宽在历史中的分位
            hist_bbw = bbw.dropna().values
            if len(hist_bbw) < 10:
                return False, 0.0, "带宽历史不足"
            rank = np.sum(hist_bbw <= cur_bbw) / len(hist_bbw)
            is_squeeze = rank <= pctile

            if not is_squeeze:
                return False, 0.0, f"带宽{cur_bbw:.4f}，历史分位{rank:.0%}，未压缩"

            # 方向判断：价格在均线上方=多头突破概率更高
            last_close = closes[-1]
            last_ma = float(ma.iloc[-1])
            bullish_bias = last_close > last_ma

            return True, 0.65 if bullish_bias else 0.45, (
                f"布林带宽压缩({cur_bbw:.4f}, 分位{rank:.1%})"
                f" | 价格在MA{'上' if bullish_bias else '下'}方"
            )
        except Exception as e:
            return False, 0.0, f"计算异常:{e}"

    def _h1_structure_score(self, symbol, direction: str) -> Tuple[float, str]:
        """
        1H 技术结构评分（0~100）:
          - 更高高点 + 更高低点 (上升趋势)
          - 多次测试支撑不破
          - 均线多头排列
        """
        if not bool(self.config.get("enable_h1_structure_score", True)):
            return 50.0, "未启用"
        rows = self._get_h1_klines(symbol)
        lookback = int(self.config.get("h1_structure_lookback", 24) or 24)
        if len(rows) < lookback + 5:
            return 50.0, f"1H数据不足({len(rows)}根)"
        direction_up = str(direction).upper() in {"BUY", "LONG"}

        try:
            closes = [float(r[4]) for r in rows[-lookback:] if float(r[4]) > 0]
            highs  = [float(r[2]) for r in rows[-lookback:] if float(r[2]) > 0]
            lows   = [float(r[3]) for r in rows[-lookback:] if float(r[3]) > 0]
            if len(closes) < 8:
                return 50.0, "有效K线不足"

            score = 50.0
            parts = []

            # ① HH/HL 结构（分成两段比较）
            half = len(highs) // 2
            if half >= 4:
                h1, h2 = highs[:half], highs[half:]
                l1, l2 = lows[:half], lows[half:]
                if direction_up:
                    if max(h2) > max(h1): score += 15; parts.append("更高高点✓")
                    else: parts.append("HH未确认")
                    if min(l2) > min(l1): score += 15; parts.append("更高低点✓")
                    else: parts.append("HL未确认")
                else:
                    if min(l2) < min(l1): score += 15; parts.append("更低低点✓")
                    else: parts.append("LL未确认")
                    if max(h2) < max(h1): score += 15; parts.append("更低高点✓")
                    else: parts.append("LH未确认")

            # ② 均线排列
            if len(closes) >= 50:
                s_closes = pd.Series(closes)
                ema12 = float(s_closes.ewm(span=12, adjust=False).mean().iloc[-1])
                ema26 = float(s_closes.ewm(span=26, adjust=False).mean().iloc[-1])
                ema50 = float(s_closes.ewm(span=50, adjust=False).mean().iloc[-1]) if len(closes) >= 50 else ema26
                if direction_up:
                    aligned = closes[-1] > ema12 > ema26
                    if aligned: score += 12; parts.append("EMA多头排列✓")
                    else: parts.append("EMA排列未完成")
                else:
                    aligned = closes[-1] < ema12 < ema26
                    if aligned: score += 12; parts.append("EMA空头排列✓")
                    else: parts.append("EMA排列未完成")

            # ③ 支撑/压力测试次数
            if direction_up:
                # 找最近低点，统计回踩不破次数
                recent_low = min(lows[-8:])
                bounce_count = sum(1 for l in lows[-12:] if abs(l - recent_low) / recent_low < 0.015)
                if bounce_count >= 3:
                    score += 8; parts.append(f"支撑测试{bounce_count}次✓")
                else:
                    parts.append(f"支撑测试{bounce_count}次")
            else:
                recent_high = max(highs[-8:])
                reject_count = sum(1 for h in highs[-12:] if abs(h - recent_high) / recent_high < 0.015)
                if reject_count >= 3:
                    score += 8; parts.append(f"压力测试{reject_count}次✓")
                else:
                    parts.append(f"压力测试{reject_count}次")

            return min(100.0, score), " | ".join(parts)
        except Exception as e:
            return 50.0, f"结构计算异常:{e}"

    def _apply_h1_breakout_scoring(self, results: List[Dict], symbols: List) -> List[Dict]:
        """
        统一应用1H突破检测：通过加分、预突破加分、结构评分
        分为 "确认突破" 和 "早期信号" 两个等级
        """
        if not bool(self.config.get("enable_h1_breakout_detect", True)):
            return results
        sym_map = {str(getattr(s, "inst_id", "")): s for s in symbols}
        breakout_bonus = float(self.config.get("h1_breakout_bonus", 8.0) or 8.0)
        early_bonus = float(self.config.get("h1_early_breakout_bonus", 4.0) or 4.0)
        squeeze_bonus = float(self.config.get("h1_squeeze_bonus", 4.0) or 4.0)
        structure_max = float(self.config.get("h1_structure_bonus_max", 5.0) or 5.0)

        for item in results:
            sym_id = str(item.get("symbol", ""))
            sym = sym_map.get(sym_id)
            if not sym:
                continue
            direction = str(item.get("direction", "WAIT")).upper()
            d = item.setdefault("details", {})

            # ① 突破检测
            is_breakout, conf, bk_detail = self._h1_breakout_detect(sym)
            if is_breakout:
                if conf >= 0.67:  # 2/3以上=确认突破
                    item["score"] = round(min(100, float(item.get("score", 0) or 0) + breakout_bonus), 2)
                    d["1H突破"] = f"✓ 确认突破(置信{conf:.0%}) +{breakout_bonus}分 | {bk_detail}"
                    item["_h1_breakout_level"] = "confirmed"
                else:
                    item["score"] = round(min(100, float(item.get("score", 0) or 0) + early_bonus), 2)
                    d["1H突破"] = f"▲ 早期突破(置信{conf:.0%}) +{early_bonus}分 | {bk_detail}"
                    item["_h1_breakout_level"] = "early"
            elif conf > 0:  # 突破已发生但检查未全通过
                # 放量不足或收盘位置不够：仍给一点分作为预突破信号
                item["score"] = round(min(100, float(item.get("score", 0) or 0) + 2.0), 2)
                d["1H突破"] = f"△ 潜在突破(待确认) +2分 | {bk_detail}"
                item["_h1_breakout_level"] = "potential"

            # ② 波动压缩检测
            is_squeeze, sq_conf, sq_detail = self._h1_squeeze_detect(sym)
            if is_squeeze:
                item["score"] = round(min(100, float(item.get("score", 0) or 0) + squeeze_bonus * sq_conf), 2)
                d["波动压缩"] = f"✓ 布林带压缩(置信{sq_conf:.0%}) +{squeeze_bonus*sq_conf:.1f}分 | {sq_detail}"
                # 压缩+突破=最强信号
                if is_breakout:
                    item["score"] = round(min(100, float(item.get("score", 0) or 0) + 2.0), 2)
                    d["波动压缩"] = d.get("波动压缩","") + " [+共振加成2分]"

            # ③ 结构评分
            struct_score, struct_detail = self._h1_structure_score(sym, direction)
            if struct_score > 50:
                bonus = (struct_score - 50) / 50 * structure_max
                item["score"] = round(min(100, float(item.get("score", 0) or 0) + bonus), 2)
                d["1H结构"] = f"评分{struct_score:.0f}/100 +{bonus:.1f}分 | {struct_detail}"

            # ④ 如果1H突破已确认，放宽3m过滤要求
            if item.get("_h1_breakout_level") in ("confirmed",):
                item["_relax_3m"] = True
                d["3m过滤"] = "1H已确认突破，3m要求放宽"

        return results

    # ── v4.4: 共识层市场状态推断（统一用 _state_adjustment，不再用硬编码 STATE_WEIGHTS）──
    def _infer_consensus_state(self, items: List[Dict]) -> str:
        """从子策略结果中提取市场状态，无标记时返回 neutral"""
        for it in items:
            st = str(it.get("market_state", it.get("state_mode", ""))).strip().lower()
            if st in {"trend", "range", "volatile"}:
                return st
        return "neutral"

    # ── v5.0: H1 入场时机评分（趋势刚启动 / 回调再启动）───────────────────
    def _h1_entry_timing_score(self, symbol, direction: str) -> Tuple[float, str, str]:
        """
        评估 H1 入场时机质量。返回 (加减分, 详情描述, 入场类型)。

        入场类型:
          "fresh_cross"    — EMA12/34 刚发生金叉/死叉（最优：趋势刚启动）
          "pullback_reentry" — 趋势中回踩EMA21后再次向上突破（次优：回调再入）
          "trend_extended"  — EMA间距过大，趋势已运行很久（差：拒绝/重惩）
          "neutral"         — 无明显特征
        """
        if not bool(self.config.get("h1_entry_timing_enabled", True)):
            return 0.0, "入场时机检测未启用", "neutral"

        rows = self._get_h1_klines(symbol)
        if len(rows) < 40:
            return 0.0, f"H1数据不足({len(rows)}根)", "neutral"

        try:
            closes = [float(r[4]) for r in rows if float(r[4]) > 0]
            highs  = [float(r[2]) for r in rows if float(r[2]) > 0]
            lows   = [float(r[3]) for r in rows if float(r[3]) > 0]
            vols   = [float(r[5]) if len(r) > 5 else 0.0 for r in rows]
            if len(closes) < 40:
                return 0.0, "H1有效数据不足", "neutral"

            s = pd.Series(closes)
            ema12 = s.ewm(span=12, adjust=False).mean()
            ema21 = s.ewm(span=21, adjust=False).mean()
            ema34 = s.ewm(span=34, adjust=False).mean()

            cur_price = closes[-1]
            cur_ema12 = float(ema12.iloc[-1])
            cur_ema21 = float(ema21.iloc[-1])
            cur_ema34 = float(ema34.iloc[-1])

            # ── EMA 间距（趋势老化程度）──────────────────────────────
            # P0-B8 修复：原代码用绝对 EMA gap %，未归一化到 ATR。
            # 高波动品种（ATR%=4%）的"EMA gap=8%"是正常的；低波动品种（ATR%=1%）的"8%"才是过度延伸。
            # 改为：以 ATR 倍数评估 → gap_atr_mult = ema_gap / atr，更符合波动适应性。
            ema_gap_pct = abs(cur_ema12 - cur_ema34) / max(cur_ema34, 1e-9) * 100.0
            # 计算 ATR%（H1）
            try:
                if len(closes) >= 15 and len(highs) >= 15 and len(lows) >= 15:
                    trs = [max(highs[i]-lows[i], abs(highs[i]-closes[i-1]), abs(lows[i]-closes[i-1]))
                           for i in range(-14, 0)]
                    atr_h1 = float(np.mean(trs))
                    atr_pct = atr_h1 / max(cur_price, 1e-9) * 100.0
                else:
                    atr_pct = 1.5  # 默认假设
            except Exception:
                atr_pct = 1.5
            # gap 归一化：ema_gap / atr_pct = "EMA 间距对应几个 ATR"
            gap_atr_mult = ema_gap_pct / max(atr_pct, 0.5)
            gap_limit_pct = float(self.config.get("h1_ema_gap_hard_limit_pct", 8.0))
            # 同时支持两套阈值：百分比（旧）和 ATR 倍数（新，gap_atr_mult ≥ 4 视为透支）
            gap_atr_limit = 4.0    # EMA12/34 间距 ≥ 4×ATR = 趋势已显著展开
            gap_penalty = float(self.config.get("h1_ema_gap_penalty", 15.0))

            is_extended = (ema_gap_pct >= gap_limit_pct) and (gap_atr_mult >= gap_atr_limit)
            if is_extended:
                excess = max(gap_atr_mult / gap_atr_limit - 1.0, ema_gap_pct / gap_limit_pct - 1.0)
                pen = gap_penalty * min(1.5, max(0.5, excess + 0.5))
                return -pen, (f"EMA间距过大({ema_gap_pct:.1f}%, {gap_atr_mult:.1f}×ATR≥{gap_atr_limit}×)，"
                              f"趋势已老化-{pen:.1f}分"), "trend_extended"

            # ── 类型A：EMA12/34 刚交叉 ───────────────────────────────
            fresh_bars = int(self.config.get("h1_fresh_cross_bars", 6))
            fresh_bonus = float(self.config.get("h1_fresh_cross_bonus", 12.0))
            diff = ema12 - ema34
            cross_age = None
            cross_dir = None
            for i in range(1, min(fresh_bars + 1, len(diff))):
                cur_d = float(diff.iloc[-i])
                prev_d = float(diff.iloc[-i - 1])
                if prev_d <= 0 < cur_d:   # 金叉
                    cross_age = i; cross_dir = "gold"; break
                if prev_d >= 0 > cur_d:   # 死叉
                    cross_age = i; cross_dir = "dead"; break

            direction_up = str(direction).upper() in {"BUY", "LONG"}
            if cross_age is not None:
                # 确认交叉方向与信号方向一致
                correct = (cross_dir == "gold" and direction_up) or (cross_dir == "dead" and not direction_up)
                if correct:
                    decay = 1.0 - (cross_age - 1) * 0.12  # 越近越满分
                    bonus = fresh_bonus * max(0.3, decay)
                    label = "金叉" if cross_dir == "gold" else "死叉"
                    return bonus, f"EMA12/34刚{label}({cross_age}根前)+{bonus:.1f}分 间距{ema_gap_pct:.1f}%", "fresh_cross"
                else:
                    # 方向相反的刚交叉 = 反向信号，惩罚
                    return -8.0, f"EMA刚反向交叉({cross_dir})，与{direction}方向冲突-8分", "trend_extended"

            # ── 类型B：回调再启动（EMA方向正确 + 价格回踩EMA21后反弹）──
            if bool(self.config.get("h1_pullback_reentry_enabled", True)):
                reentry_bonus = float(self.config.get("h1_pullback_reentry_bonus", 10.0))
                # 条件：趋势方向正确（EMA12>EMA34 for bull）
                trend_aligned = (direction_up and cur_ema12 > cur_ema34) or (not direction_up and cur_ema12 < cur_ema34)
                if trend_aligned and len(closes) >= 10:
                    # 检测过去8根内是否有回踩到EMA21附近（1.5%以内）
                    touched_ema21 = False
                    touch_bar = None
                    for i in range(2, min(9, len(closes))):
                        ema21_val = float(ema21.iloc[-i])
                        if direction_up:
                            bar_low = lows[-i] if len(lows) >= i else closes[-i]
                            dist = abs(bar_low - ema21_val) / max(ema21_val, 1e-9) * 100
                            if dist <= 1.5:
                                touched_ema21 = True; touch_bar = i; break
                        else:
                            bar_high = highs[-i] if len(highs) >= i else closes[-i]
                            dist = abs(bar_high - ema21_val) / max(ema21_val, 1e-9) * 100
                            if dist <= 1.5:
                                touched_ema21 = True; touch_bar = i; break

                    if touched_ema21:
                        # 价格当前已从EMA21反弹（回踩后恢复）
                        if direction_up:
                            rebounded = cur_price > cur_ema21 * 1.003  # 已反弹回EMA21上方
                        else:
                            rebounded = cur_price < cur_ema21 * 0.997
                        if rebounded:
                            bonus = reentry_bonus * (1.0 - (touch_bar - 2) * 0.1)
                            bonus = max(reentry_bonus * 0.4, bonus)
                            return bonus, f"H1回踩EMA21({touch_bar}根前)后再启动+{bonus:.1f}分 间距{ema_gap_pct:.1f}%", "pullback_reentry"

            # ── 无特征但 EMA 间距适中 → 小奖励 ─────────────────────
            if ema_gap_pct < gap_limit * 0.4:
                return 3.0, f"EMA间距适中({ema_gap_pct:.1f}%)，趋势较新+3分", "neutral"
            return 0.0, f"无明显早启动特征 间距{ema_gap_pct:.1f}%", "neutral"

        except Exception as e:
            return 0.0, f"入场时机计算异常:{e}", "neutral"

    def _apply_h1_entry_timing_filter(self, results: List[Dict], symbols: List) -> List[Dict]:
        """
        v5.0: 应用 H1 入场时机过滤：
          - trend_age 超过硬上限 → 直接拒绝
          - trend_extended（EMA间距过大）→ 惩罚
          - fresh_cross / pullback_reentry → 加分
        """
        if not bool(self.config.get("h1_entry_timing_enabled", True)):
            return results
        hard_limit = int(self.config.get("h1_trend_age_hard_limit", 16) or 16)
        sym_map = {str(getattr(s, "inst_id", "")): s for s in symbols}
        kept = []
        for item in results:
            sym_id = str(item.get("symbol", ""))
            sym = sym_map.get(sym_id)
            direction = str(item.get("direction", "WAIT")).upper()

            # ① trend_age 硬上限过滤
            trend_age = 0
            det = item.get("details") if isinstance(item.get("details"), dict) else {}
            for k in ("1H趋势延续根数", "h1_trend_age", "trend_age"):
                try:
                    v = det.get(k) or (item.get(k) or 0)
                    if v: trend_age = int(float(str(v).split()[0])); break
                except Exception: pass
            if hard_limit > 0 and trend_age > hard_limit:
                logger.info(f"[H1时机] 拒绝 {sym_id}: trend_age={trend_age}>{hard_limit}")
                continue

            # ② H1 入场时机评分
            if sym and direction in {"BUY", "SELL"}:
                adj, timing_det, entry_type = self._h1_entry_timing_score(sym, direction)
                gap_hard_thr = float(self.config.get("h1_ema_gap_penalty", 15.0)) * 0.8
                if entry_type == "trend_extended" and adj <= -gap_hard_thr:
                    logger.info(f"[H1时机] 拒绝 {sym_id}: {timing_det}")
                    continue
                # P0-B4 修复：原代码 `dict(item)` 是浅拷贝，details 仍是同一对象
                # 修改 details 会污染 by_strategy / consensus 等共享原对象的字典
                item = dict(item)
                item["details"] = dict(item.get("details") or {})    # 深拷贝 details
                item["score"] = round(max(0.0, min(100.0, float(item.get("score", 0) or 0) + adj)), 2)
                item["details"]["H1入场时机"] = f"[{entry_type}] {timing_det}"
                item["_h1_entry_type"] = entry_type
            kept.append(item)
        return kept

    # ── v4.4: 共识层市场状态推断（统一用 _state_adjustment，不再用硬编码 STATE_WEIGHTS）──（保留原位置标记）

    # ── P2-5: BTC 熔断检查 ──────────────────────────────────────────────────
    def _btc_circuit_breaker(self, btc_ctx: Dict) -> Tuple[bool, str]:
        """检查 BTC 是否处于剧烈波动状态。
        返回 (是否熔断, 原因描述)。熔断时所有信号被丢弃。
        """
        if not bool(self.config.get("btc_circuit_breaker_enabled", True)):
            return False, ""
        threshold = float(self.config.get("btc_circuit_threshold_pct", 5.0))
        btc_1h = abs(_safe_number(btc_ctx.get("btc_1h_move", 0)))
        if btc_1h >= threshold:
            return True, f"BTC 1H |{btc_1h:.2f}%| ≥ {threshold:.1f}%（熔断）"
        return False, ""

    # ── P2-2: H4 方向硬过滤 ──────────────────────────────────────────────────
    def _apply_h4_direction_hard_filter(self, results: List[Dict], symbols: List) -> List[Dict]:
        """H4 方向反向时直接淘汰（替代加分项）。
        利用 P1-2 缓存的 H1/H4 EMA 数据。
        """
        if not bool(self.config.get("h4_direction_hard_filter", True)):
            return results
        min_conf = float(self.config.get("h4_filter_min_confidence", 0.4))
        sym_map = {str(getattr(s, "inst_id", "")): s for s in symbols}
        kept = []
        for item in results:
            # 共振信号免过滤（多引擎共识应该比 H4 信号更强）
            if "共振" in str(item.get("category", "")):
                kept.append(item); continue
            sym_id = str(item.get("symbol", ""))
            sym = sym_map.get(sym_id)
            if not sym:
                kept.append(item); continue
            # 提取 H4 EMA
            klines = (getattr(sym, "extra_data", {}) or {}).get("klines", {})
            h4_rows = klines.get("4H") or klines.get("4h") or []
            if len(h4_rows) < 26:
                kept.append(item); continue   # 数据不足时不过滤
            try:
                h4_closes = [float(r[4]) for r in h4_rows if float(r[4]) > 0]
                if len(h4_closes) < 26:
                    kept.append(item); continue
                s = pd.Series(h4_closes)
                ema12 = float(s.ewm(span=12, adjust=False).mean().iloc[-1])
                ema26 = float(s.ewm(span=26, adjust=False).mean().iloc[-1])
                gap_pct = abs(ema12 - ema26) / max(ema26, 1e-9) * 100.0
                # 仅当 EMA gap 显著（>= min_conf）时才硬拒（避免边缘情况误杀）
                direction = str(item.get("direction", "WAIT")).upper()
                h4_bull = ema12 > ema26
                if gap_pct >= min_conf:
                    if direction in {"BUY", "LONG"} and not h4_bull:
                        logger.info(f"[H4硬过滤] 淘汰 {sym_id}: H4空头 EMA gap={gap_pct:.2f}%")
                        continue
                    if direction in {"SELL", "SHORT"} and h4_bull:
                        logger.info(f"[H4硬过滤] 淘汰 {sym_id}: H4多头 EMA gap={gap_pct:.2f}%")
                        continue
            except Exception:
                pass
            kept.append(item)
        return kept

    # ── 统一应用所有交易视角过滤器 ──────────────────────────────────────────
    def _apply_trading_filters(self, results: List[Dict], symbols: List) -> List[Dict]:
        """按顺序应用所有v4.2交易视角过滤器（v4.4: 加分上限保护）"""
        btc_ctx = self._get_btc_context(symbols)
        # P2-5: BTC 熔断检查（所有过滤器之前）
        is_break, break_reason = self._btc_circuit_breaker(btc_ctx)
        if is_break:
            logger.warning(f"[BTC熔断] 全面熔断 {len(results)} 条信号: {break_reason}")
            for r in results:
                r["passed"] = False
                r.setdefault("details", {})["BTC熔断"] = f"⚠ {break_reason}"
            return []
        # 记录每个结果的原始分，后续加分项总上限 15 分
        orig_scores = {id(r): float(r.get("score", 0) or 0) for r in results}
        # v5.0: 最先执行入场时机过滤（硬拒绝老趋势 + 早启动/回调再启动加减分）
        results = self._apply_h1_entry_timing_filter(results, symbols)
        # P2-2: H4 方向硬过滤（在 H1 时机之后，进入加分前）
        results = self._apply_h4_direction_hard_filter(results, symbols)
        results = self._apply_h1_breakout_scoring(results, symbols)
        results = self._apply_btc_correlation_filter(results, btc_ctx)
        results = self._apply_funding_filter(results, symbols)
        results = self._apply_confluence_scoring(results, symbols)
        results = self._apply_atr_stop_suggestion(results, symbols)
        results = self._apply_volume_quality_check(results, symbols)
        results = self._apply_btc_relative_strength(results, symbols, btc_ctx)
        # 加分封顶：总分最多比原始分高 18 分
        # 扣分下限：总分最多比原始分低 20 分（防止多项惩罚叠加淹没高质量信号）
        MAX_ADD = 18.0
        MAX_DED = 20.0
        for r in results:
            orig = orig_scores.get(id(r), float(r.get("score", 0) or 0))
            current = float(r.get("score", 0) or 0)
            if current > orig + MAX_ADD:
                r["score"] = round(orig + MAX_ADD, 2)
                r.setdefault("details", {})["加分封顶"] = f"原始{orig:.1f}→封顶{orig+MAX_ADD:.1f}"
            elif current < orig - MAX_DED:
                r["score"] = round(orig - MAX_DED, 2)
                r.setdefault("details", {})["扣分下限"] = f"原始{orig:.1f}→下限{orig-MAX_DED:.1f}(惩罚已封顶)"
        return results

    # ── 硬性过滤：辅助工具 ──────────────────────────────────────────────────
    def _get_m3_klines(self, symbol) -> List:
        """从 symbol.extra_data 中取出3m K线列表，格式：[ts, open, high, low, close, vol, ...]"""
        try:
            ed = getattr(symbol, "extra_data", None) or {}
            klines = ed.get("klines", {}) if isinstance(ed, dict) else {}
            rows = klines.get("3m") or klines.get("3M") or []
            return rows if isinstance(rows, (list, tuple)) else []
        except Exception:
            return []

    def _get_h1_klines(self, symbol) -> List:
        """从 symbol.extra_data 中取出1H K线列表。"""
        try:
            ed = getattr(symbol, "extra_data", None) or {}
            klines = ed.get("klines", {}) if isinstance(ed, dict) else {}
            rows = (klines.get("1H") or klines.get("1h")
                    or klines.get("60m") or klines.get("60M") or [])
            return rows if isinstance(rows, (list, tuple)) else []
        except Exception:
            return []

    # ── 硬性检查 1：3m 回调、不跌破突破点、企稳 ─────────────────────────────
    def _m3_pullback_hard_check(self, m3_rows: List, direction: str, relax: bool = False) -> Tuple[bool, str]:
        """
        三合一硬性检测（LONG 示例，SHORT 镜像）：
          ① 3m 存在真实脉冲高点（lookback 内）
          ② 当前回调幅度在 [min_pct, max_pct] 区间内
          ③ 企稳阶段（最后 stab_bars 根）最低收盘价不低于脉冲起点（突破点）
             —— 即"不跌破原突破点"（允许 m3_no_break_tolerance_pct 容差）
          ④ 企稳形态：区间紧缩 OR 末棒方向正确（阳线/阴线）
        SHORT 镜像：脉冲低点 → 反弹 → 不突破下跌起点 → 区间/末棒
        """
        lookback     = max(8, int(self.config.get("m3_impulse_lookback_bars", 15) or 15))
        min_pct      = float(self.config.get("m3_pullback_min_pct",      0.10) or 0.10)
        max_pct      = float(self.config.get("m3_pullback_max_pct",      5.00) or 5.00)
        stab_bars    = max(2, int(self.config.get("m3_stabilization_bars", 2) or 2))
        no_brk_tol   = float(self.config.get("m3_no_break_tolerance_pct", 0.60) or 0.60) / 100.0

        # v4.3: 1H突破已确认时放宽3m要求
        if relax:
            min_pct *= 0.5      # 最低回调从0.5%→0.25%
            max_pct *= 1.5      # 最高回调从2.2%→3.3%
            no_brk_tol *= 2.0   # 突破点容差翻倍

        if not m3_rows or len(m3_rows) < lookback + stab_bars:
            return False, f"3m数据不足({len(m3_rows)}根，需{lookback + stab_bars}根)"

        rows = list(m3_rows)[-(lookback + stab_bars):]

        def _val(row, idx, default=0.0):
            try: return float(row[idx])
            except (TypeError, ValueError, IndexError): return default

        # 诊断用数据摘要
        try:
            data_range = f"最新={_val(rows[-1],4):.6g} 最高={max(_val(r,2) for r in rows):.6g} 最低={min(_val(r,3) for r in rows):.6g}"
        except Exception:
            data_range = "解析失败"

        direction_up = str(direction).upper() in {"BUY", "LONG"}
        body_rows = rows[:lookback]   # 脉冲主体段
        tail_rows = rows[lookback:]   # 企稳候选段

        if direction_up:
            # ① 找脉冲高点及其位置
            peak_high = max((_val(r, 2) for r in body_rows), default=0.0)
            if peak_high <= 0:
                return False, f"3m无有效脉冲高点 [{data_range}]"

            peak_idx = max(range(len(body_rows)),
                           key=lambda i: _val(body_rows[i], 2))

            # ③ 突破起点 = 脉冲高点出现前最低收盘（脉冲的起跳平台）
            pre_peak_closes = [_val(body_rows[i], 4) for i in range(peak_idx)]
            breakout_base   = min((c for c in pre_peak_closes if c > 0), default=0.0)

            # ② 当前企稳段末棒收盘价
            cur_close = _val(tail_rows[-1], 4) if tail_rows else 0.0
            if cur_close <= 0:
                return False, "3m末棒价格无效"

            pullback_pct = (peak_high - cur_close) / peak_high * 100.0
            if pullback_pct < min_pct:
                return False, f"3m回调幅度不足({pullback_pct:.2f}%<{min_pct}%)"
            if pullback_pct > max_pct:
                return False, f"3m回调过深({pullback_pct:.2f}%>{max_pct}%，可能趋势反转)"

            # ③ 企稳段任意收盘不得低于突破起点（含容差）
            if breakout_base > 0:
                floor = breakout_base * (1.0 - no_brk_tol)
                tail_closes = [_val(r, 4) for r in tail_rows]
                min_tail_close = min((c for c in tail_closes if c > 0), default=cur_close)
                if min_tail_close < floor:
                    depth = (floor - min_tail_close) / floor * 100
                    return False, (
                        f"3m回调跌破突破起点"
                        f"(最低收盘{min_tail_close:.4f} < 突破点{breakout_base:.4f}"
                        f"-容差，低{depth:.2f}%)")

            # ④ 企稳形态
            tail_highs  = [_val(r, 2) for r in tail_rows]
            tail_lows   = [_val(r, 3) for r in tail_rows]
            range_pct   = (max(tail_highs) - min(tail_lows)) / max(peak_high, 1e-9) * 100.0
            last_bull   = _val(tail_rows[-1], 4) >= _val(tail_rows[-1], 1)
            tight_range = range_pct <= max_pct * 0.60
            if not (tight_range or last_bull):
                return False, (
                    f"3m未企稳(波动{range_pct:.2f}%，"
                    f"末棒{'阳' if last_bull else '阴'}，需紧缩或阳线收盘)")

            stab_desc = "区间紧缩" if tight_range else "末棒阳线"
            base_desc  = f"，突破点{breakout_base:.4f}" if breakout_base > 0 else ""
            return True, (
                f"3m多头回调企稳✓ 回调{pullback_pct:.2f}%{base_desc}，{stab_desc}")

        else:  # SHORT 镜像
            # ① 找脉冲低点
            trough_low = min(
                (_val(r, 3) for r in body_rows if _val(r, 3) > 0), default=0.0)
            if trough_low <= 0:
                return False, "3m无有效脉冲低点"

            trough_idx = min(range(len(body_rows)),
                             key=lambda i: _val(body_rows[i], 3) or float("inf"))

            # ③ 突破起点 = 脉冲低点出现前最高收盘（下跌起点的压力位）
            pre_trough_closes = [_val(body_rows[i], 4) for i in range(trough_idx)]
            breakout_base     = max((c for c in pre_trough_closes if c > 0), default=0.0)

            cur_close = _val(tail_rows[-1], 4) if tail_rows else 0.0
            if cur_close <= 0:
                return False, "3m末棒价格无效"

            bounce_pct = (cur_close - trough_low) / trough_low * 100.0
            if bounce_pct < min_pct:
                return False, f"3m反弹幅度不足({bounce_pct:.2f}%<{min_pct}%)"
            if bounce_pct > max_pct:
                return False, f"3m反弹过高({bounce_pct:.2f}%>{max_pct}%，可能趋势反转)"

            # ③ 企稳段任意收盘不得高于突破起点（含容差）
            if breakout_base > 0:
                ceiling = breakout_base * (1.0 + no_brk_tol)
                tail_closes = [_val(r, 4) for r in tail_rows]
                max_tail_close = max((c for c in tail_closes if c > 0), default=cur_close)
                if max_tail_close > ceiling:
                    height = (max_tail_close - ceiling) / ceiling * 100
                    return False, (
                        f"3m反弹突破下跌起点"
                        f"(最高收盘{max_tail_close:.4f} > 压力点{breakout_base:.4f}"
                        f"+容差，高{height:.2f}%)")

            # ④ 企稳形态
            tail_highs  = [_val(r, 2) for r in tail_rows]
            tail_lows   = [_val(r, 3) for r in tail_rows]
            range_pct   = (max(tail_highs) - min(tail_lows)) / max(cur_close, 1e-9) * 100.0
            last_bear   = _val(tail_rows[-1], 4) <= _val(tail_rows[-1], 1)
            tight_range = range_pct <= max_pct * 0.60
            if not (tight_range or last_bear):
                return False, (
                    f"3m未企稳(波动{range_pct:.2f}%，"
                    f"末棒{'阴' if last_bear else '阳'}，需紧缩或阴线收盘)")

            stab_desc = "区间紧缩" if tight_range else "末棒阴线"
            base_desc  = f"，压力点{breakout_base:.4f}" if breakout_base > 0 else ""
            return True, (
                f"3m空头反弹企稳✓ 反弹{bounce_pct:.2f}%{base_desc}，{stab_desc}")

    # ── 硬性检查 2：小时线趋势延续（EMA 排列） ───────────────────────────────
    def _h1_trend_check(self, h1_rows: List, direction: str) -> Tuple[bool, str]:
        """
        验证 H1 趋势仍在延续，不允许已出现趋势反转。
        用 EMA(fast) vs EMA(slow) 判断：
          LONG  → EMA(fast) > EMA(slow)，且最近2根H1收盘未连续低于EMA(fast)
          SHORT → EMA(fast) < EMA(slow)，且最近2根H1收盘未连续高于EMA(fast)
        数据不足时宽松处理（返回True），不因数据缺失误杀。
        """
        fast_span = max(3, int(self.config.get("h1_ema_fast", 12) or 12))
        slow_span = max(fast_span + 1, int(self.config.get("h1_ema_slow", 26) or 26))
        min_bars  = slow_span + 5

        if not h1_rows or len(h1_rows) < min_bars:
            return True, f"1H数据不足({len(h1_rows)}根)，跳过趋势验证"

        closes: List[float] = []
        for row in h1_rows:
            try:
                c = float(row[4])
                if np.isfinite(c) and c > 0:
                    closes.append(c)
            except (TypeError, ValueError, IndexError):
                pass

        if len(closes) < min_bars:
            return True, "1H有效收盘数据不足，跳过趋势验证"

        # 计算最终 EMA（Wilder 式指数平滑）
        def _ema_final(data: List[float], span: int) -> float:
            k, val = 2.0 / (span + 1), data[0]
            for v in data[1:]:
                val = v * k + val * (1.0 - k)
            return val

        ema_fast = _ema_final(closes, fast_span)
        ema_slow = _ema_final(closes, slow_span)
        last2    = closes[-2:]  # 最近两根 H1 收盘

        direction_up = str(direction).upper() in {"BUY", "LONG"}

        if direction_up:
            if ema_fast < ema_slow:
                gap = (ema_slow - ema_fast) / max(ema_slow, 1e-9) * 100
                return False, (
                    f"1H趋势已转空(EMA{fast_span} < EMA{slow_span}，"
                    f"偏离{gap:.2f}%)，多头信号无效")
            # 检测连续反转迹象：最近2根H1全部收盘低于快线EMA
            if len(last2) >= 2 and all(c < ema_fast for c in last2):
                return False, (
                    f"1H多头趋势减弱：最近2根收盘({last2[-1]:.4f})均低于EMA{fast_span}"
                    f"({ema_fast:.4f})，趋势动能不足")
            gap = (ema_fast - ema_slow) / max(ema_slow, 1e-9) * 100
            return True, f"1H多头趋势延续✓ EMA{fast_span}>{slow_span}，领先{gap:.2f}%"
        else:
            if ema_fast > ema_slow:
                gap = (ema_fast - ema_slow) / max(ema_fast, 1e-9) * 100
                return False, (
                    f"1H趋势已转多(EMA{fast_span} > EMA{slow_span}，"
                    f"偏离{gap:.2f}%)，空头信号无效")
            if len(last2) >= 2 and all(c > ema_fast for c in last2):
                return False, (
                    f"1H空头趋势减弱：最近2根收盘({last2[-1]:.4f})均高于EMA{fast_span}"
                    f"({ema_fast:.4f})，趋势动能不足")
            gap = (ema_slow - ema_fast) / max(ema_slow, 1e-9) * 100
            return True, f"1H空头趋势延续✓ EMA{fast_span}<{slow_span}，领先{gap:.2f}%"

    # ── 综合应用两项硬性过滤 ─────────────────────────────────────────────────
    def _apply_m3_hard_filter(self, result: Dict[str, Any], symbol) -> Dict[str, Any]:
        """
        同时执行两项硬性过滤，任一不通过则 passed=False：
          1. 3m 回调 + 不跌破突破点 + 企稳
          2. 1H 趋势仍在延续（EMA 排列）
        """
        if not bool(self.config.get("m3_hard_filter", True)):
            return result
        if not result.get("passed"):
            return result
        direction = str(result.get("direction", "WAIT")).upper()
        if direction not in {"BUY", "SELL"}:
            return result

        m3_rows = self._get_m3_klines(symbol)
        h1_rows = self._get_h1_klines(symbol)

        relax_m3 = bool(result.get("_relax_3m", False))
        m3_ok, m3_reason = self._m3_pullback_hard_check(m3_rows, direction, relax=relax_m3)
        if relax_m3 and not m3_ok:
            m3_reason += " (1H已突破，3m放宽后仍不通过)"
        h1_ok, h1_reason = (
            self._h1_trend_check(h1_rows, direction)
            if bool(self.config.get("h1_trend_hard_filter", True))
            else (True, "1H过滤已关闭")
        )

        result = dict(result)
        details = dict(result.get("details") or {})

        if m3_ok and h1_ok:
            details["3m硬性过滤"] = f"通过: {m3_reason}"
            details["1H趋势过滤"] = f"通过: {h1_reason}"
            result["details"] = details
            return result

        # 任一不通过
        result["passed"] = False
        fail_parts = []
        if not m3_ok:
            details["3m硬性过滤"] = f"未通过: {m3_reason}"
            fail_parts.append(f"[3m] {m3_reason}")
        else:
            details["3m硬性过滤"] = f"通过: {m3_reason}"
        if not h1_ok:
            details["1H趋势过滤"] = f"未通过: {h1_reason}"
            fail_parts.append(f"[1H] {h1_reason}")
        else:
            details["1H趋势过滤"] = f"通过: {h1_reason}"

        result["details"] = details
        sigs = list(result.get("signals") or [])
        sigs.append("[硬性过滤淘汰] " + " | ".join(fail_parts))
        result["signals"] = sigs
        return result

    # ── 子策略加载（v6.0 P1-5: 懒加载 + 失败追踪）────────────────────────
    # 子策略定义（class引用，不立即实例化）
    _CHILD_DEFS = [
        ("截面多因子",      "ACrossSectionalMultiFactorScannerStrategy",  970),
        ("AI因子挖掘",      "AIAutomatedAlphaCryptoScannerStrategy",       960),
        ("DRL小时趋势启动", "DRLMetaHourlyTrendStartScannerStrategy",      965),
        ("XGBoost截面排序", "XGBoostCrossSectionalRanker",                 955),
        ("AI订单流动量",    "AIOrderflowMomentumBreakoutScanner",          945),
    ]

    def _build_child_strategies(self):
        """v6.0 P1-5: 立即实例化所有子策略（保持原行为以兼容）。
        失败的子策略记录到 self._failed_engines 而非吞掉。
        """
        strategies = []
        # 通过模块全局名找类（避免硬编码 import 在文件顶部）
        glb = globals()
        # 失败追踪
        if not hasattr(self, "_failed_engines"):
            self._failed_engines: Dict[str, str] = {}
        for sn, cls_name, priority in self._CHILD_DEFS:
            cls = glb.get(cls_name)
            if cls is None:
                self._failed_engines[sn] = f"类 {cls_name} 未定义"
                logger.error(f"[组合] 跳过 {sn}: {self._failed_engines[sn]}")
                continue
            try:
                child_cfg = self._child_config(sn)
                strategies.append((sn, cls(child_cfg), priority))
                logger.info(f"[组合] 已加载 {sn}")
            except Exception as e:
                self._failed_engines[sn] = f"初始化失败: {type(e).__name__}: {e}"
                logger.error(f"[组合] 跳过 {sn}: {self._failed_engines[sn]}")
        return strategies

    # 子策略允许透传的键白名单（避免组合层 70+ 参数污染子策略命名空间）
    _CHILD_SAFE_KEYS = {
        "min_volume_24h", "min_score", "position_size", "allow_short", "max_atr_pct",
        # 时效性
        "max_h1_trend_age", "h1_trend_age_penalty", "max_m3_staleness_bars",
        "m3_freshness_penalty", "bonus_freshness_score",
        # 3m 回调企稳
        "require_m3_pullback_confirmation", "m3_pullback_min_pct", "m3_pullback_max_pct",
        "m3_stabilization_bars", "require_m3_freshness", "m3_min_impulse_pct",
        "vol_continuation_min_ratio",
        # 微观结构
        "enable_atr_squeeze_check", "atr_squeeze_ratio",
        "enable_volume_delta_check", "volume_delta_min_ratio",
        "enable_vwap_alignment_check",
        # 截面加速
        "use_dynamic_ic_weights", "ic_weight_blend",
        "use_orthogonalization", "enable_mfin_interactions",
        "enable_llm_factors", "enable_on_chain",
        "enable_early_trend_factors", "early_trend_min_trigger",
        # v5.0 早启动
        "h1_trend_age_hard_limit", "require_early_trend_entry", "early_trend_edge_discount",
    }

    def _child_config(self, strategy_name: str) -> Dict[str, Any]:
        """
        仅透传白名单内的安全参数给子策略，避免参数名冲突。
        截面子策略加速模式时可选关闭 m3 检查。
        """
        cfg = {k: self.config[k] for k in self._CHILD_SAFE_KEYS if k in self.config}
        if strategy_name == "截面多因子" and bool(self.config.get("accelerate_cross_section_child", True)):
            cfg["use_dynamic_ic_weights"] = False
            cfg["ic_weight_blend"] = 0.0
            if bool(self.config.get("accel_disable_m3", False)):
                cfg["require_m3_pullback_confirmation"] = False
        return cfg

    # ── 市场状态推断 ──────────────────────────────────────────────────────────
    def _normalize_scores_cross_sectional(self, items: List[Dict]) -> None:
        """跨引擎 z-score 归一化：让不同评分体系的引擎结果可比"""
        if not items or not bool(self.config.get("enable_score_normalization", True)):
            return
        scores = [float(it.get("score", 0) or 0) for it in items]
        mean_s = np.mean(scores) if scores else 50.0
        std_s = max(np.std(scores), 1e-6)
        for it in items:
            raw = float(it.get("score", 0) or 0)
            norm = (raw - mean_s) / std_s * 15.0 + 50.0
            it["_norm_score"] = round(max(0.0, min(100.0, norm)), 2)

    def _get_engine_weight(self, engine_name: str, direction: str) -> float:
        """获取引擎当前胜率权重（0.5~1.5）。
        P0-B5 修复：原 track<5 时直接返回 1.0，无平滑过渡。
        改为贝叶斯先验：先验胜率 50%，先验权重 5（等价于"假装看过 5 局，胜负各半"）。
        track 越多，实际胜率影响越大；track=0 时退化为先验 1.0。
        """
        if not bool(self.config.get("enable_engine_track_record", True)):
            return 1.0
        track = self._engine_track.get(engine_name, {}).get(direction, [])
        n = len(track)
        wins = sum(1 for p in track if p > 0)
        # 贝叶斯平滑：(wins + α) / (n + α + β)，先验 α=β=2.5（弱先验，避免长期支配）
        prior_alpha = 2.5
        prior_beta  = 2.5
        wr_smooth = (wins + prior_alpha) / max(n + prior_alpha + prior_beta, 1.0)
        return max(0.5, min(1.5, 0.5 + wr_smooth))

    def _record_engine_performance(self, engine_name: str, direction: str, pnl: float) -> None:
        """记录引擎信号的实际绩效，expire 旧记录"""
        if not bool(self.config.get("enable_engine_track_record", True)):
            return
        decay = float(self.config.get("engine_weight_decay", 0.85) or 0.85)
        track = self._engine_track.setdefault(engine_name, {}).setdefault(direction, [])
        track.append(pnl)
        # 保持最新 N 条
        while len(track) > self._max_perf_entries:
            track.pop(0)

    def _record_signal_performance(self, symbol: str, direction: str, score: float,
                                    pnl: Optional[float]) -> None:
        """记录单个币种的信号→PnL，供前端展示历史绩效"""
        cache = self._performance_cache.setdefault(symbol, [])
        cache.append((direction, score, pnl))
        while len(cache) > self._max_perf_entries:
            cache.pop(0)

    def _get_signal_performance_label(self, symbol: str, direction: str) -> str:
        """生成历史绩效标签文字"""
        cache = self._performance_cache.get(symbol, [])
        if not cache:
            return "无历史记录"
        recent = [r for r in cache[-10:] if r[0] == direction]
        if not recent:
            return "无同向历史记录"
        wins = sum(1 for r in recent if (r[2] or 0) > 0)
        pnls = [r[2] for r in recent if r[2] is not None]
        avg_pnl = sum(pnls) / len(pnls) * 100 if pnls else 0.0
        return f"近{len(recent)}次同向: {wins}赢{len(recent)-wins}亏, 均{avg_pnl:+.2f}%"
    def _state_adjustment(self, sn: str, state: str) -> Tuple[float, float]:
        table = {
            "trend":    {"DRL小时趋势启动": (2.8,1.03), "AI因子挖掘": (1.5,1.01), "截面多因子": (0.8,1.00),
                         "XGBoost截面排序": (1.2,1.01), "AI订单流动量": (1.0,1.00)},
            "range":    {"截面多因子": (2.1,1.02),       "AI因子挖掘": (1.2,1.01), "DRL小时趋势启动": (-1.8,0.98),
                         "XGBoost截面排序": (1.5,1.01), "AI订单流动量": (0.5,1.00)},
            "volatile": {"AI因子挖掘": (0.9,1.00),       "截面多因子": (0.6,1.00), "DRL小时趋势启动": (-1.2,0.97),
                         "AI订单流动量": (1.2,1.01), "XGBoost截面排序": (0.7,1.00)},
            "neutral":  {"DRL小时趋势启动": (0.4,1.00),  "AI因子挖掘": (0.4,1.00), "截面多因子": (0.4,1.00),
                         "XGBoost截面排序": (0.4,1.00), "AI订单流动量": (0.4,1.00)},
        }
        return table.get(str(state).lower(), {}).get(sn, (0.0, 1.0))

    def _normalize_state_mode(self, state_mode) -> str:
        mode = str(state_mode or "auto").strip().lower()
        return mode if mode in {"auto","trend","range","volatile","neutral"} else "auto"

    def _infer_market_state_from_data(self, data) -> str:
        if isinstance(data, dict):
            direct = str(data.get("market_state", data.get("state_mode",""))).strip().lower()
            if direct in {"trend","range","volatile","neutral"}:
                return direct
        closes = self._extract_closes_from_backtest_data(data)
        if len(closes) < 40:
            return "neutral"
        series = pd.Series(closes, dtype=float).replace([np.inf,-np.inf], np.nan).dropna()
        if len(series) < 40:
            return "neutral"
        fast = float(series.ewm(span=12, adjust=False).mean().iloc[-1])
        slow = float(series.ewm(span=34, adjust=False).mean().iloc[-1])
        last = max(abs(float(series.iloc[-1])), 1e-9)
        gap_pct = abs(fast - slow) / last * 100.0
        lb = min(24, len(series) - 1)
        dir_move = abs((float(series.iloc[-1]) / max(float(series.iloc[-lb-1]),1e-9) - 1.0)*100.0) if lb > 0 else 0.0
        rv = float(np.log(series).diff().dropna().tail(min(24,len(series)-1)).std() * np.sqrt(24) * 100.0)
        if rv >= 6.2: return "volatile"
        if gap_pct >= 1.25 and dir_move >= 1.6: return "trend"
        if gap_pct <= 0.85 and dir_move <= 1.2 and rv <= 5.4: return "range"
        return "neutral"

    def _extract_closes_from_backtest_data(self, data) -> List[float]:
        rows: List = []
        if isinstance(data, dict):
            km = data.get("klines_map") if isinstance(data.get("klines_map"), dict) else {}
            for key in ("1H","1h","60m","60M","15m","15M"):
                if key in km and isinstance(km.get(key), list) and km.get(key):
                    rows = list(km.get(key) or []); break
            if not rows and isinstance(data.get("klines"), list): rows = list(data.get("klines") or [])
            if not rows and isinstance(data.get("bars"), list): rows = list(data.get("bars") or [])
        elif isinstance(data, (list, tuple)): rows = list(data)
        closes = []
        for row in rows:
            val = row.get("c", row.get("close")) if isinstance(row, dict) else (row[4] if isinstance(row,(list,tuple)) and len(row)>=5 else row)
            try:
                c = float(val)
                if np.isfinite(c): closes.append(c)
            except (TypeError, ValueError): pass
        return closes

    # ── 回测对比 ──────────────────────────────────────────────────────────────
    def run_state_backtest_compare(self, dataset, *a, **kw) -> Dict[str, Any]:
        if isinstance(dataset, dict) and isinstance(dataset.get("samples"), list):
            samples = list(dataset.get("samples") or [])
        elif isinstance(dataset, (list,tuple)): samples = list(dataset)
        else: samples = [dataset]
        samples = [s for s in samples if s is not None]
        modes = ["auto","trend","range","volatile"]
        compare = {m: self._simulate_state_mode(samples, m, *a, **kw) for m in modes}
        best_mode = max(compare, key=lambda m: (
            float(compare[m].get("total_return_pct") or -1e9) if compare[m].get("total_return_pct") is not None
            else float(compare[m].get("avg_signal_score", 0) or 0)
        ))
        return {"type":"state_mode_backtest_compare","sample_count":len(samples),"compare":compare,"best_mode":best_mode}

    def _simulate_state_mode(self, samples, mode, *a, **kw):
        trades=wins=losses=0; score_sum=0.0; consensus_hits=0; rets=[]
        for sample in samples:
            sig = self.generate_signal(sample, *a, state_mode=mode, **kw)
            if not sig: continue
            if str(sig.get("action","")).upper() not in {"BUY","SHORT"}: continue
            trades += 1
            score_sum += float(sig.get("score",0) or 0)
            if int(sig.get("consensus_engines",0) or 0) >= int(self.config.get("min_consensus_engines",2) or 2):
                consensus_hits += 1
            ret = self._extract_realized_return(sample, sig)
            if ret is None: continue
            rets.append(ret)
            if ret > 0: wins += 1
            elif ret < 0: losses += 1
        avg = score_sum/trades if trades else 0.0
        wr = (wins/(wins+losses)*100) if (wins+losses) else None
        if rets:
            eq=peak=1.0; mdd=0.0
            for r in rets:
                eq*=(1+r); peak=max(peak,eq)
                mdd=max(mdd,(peak-eq)/peak if peak>0 else 0)
            tr=(eq-1)*100
        else: tr=mdd=None
        return {"mode":mode,"trades":trades,"wins":wins,"losses":losses,
                "win_rate_pct":round(float(wr),2) if wr is not None else None,
                "avg_signal_score":round(float(avg),2),
                "consensus_hits":consensus_hits,
                "consensus_ratio_pct":round(consensus_hits/trades*100,2) if trades else 0.0,
                "total_return_pct":round(float(tr),2) if tr is not None else None,
                "max_drawdown_pct":round(float(mdd)*100,2) if mdd is not None else None}

    def _extract_realized_return(self, sample, signal) -> Optional[float]:
        if not isinstance(sample, dict): return None
        candidates = [sample.get(k) for k in ("future_return","future_return_1h","next_return","label_return","target_return","ret_1h")]
        labels = sample.get("labels") if isinstance(sample.get("labels"), dict) else {}
        candidates += [labels.get(k) for k in ("future_return","future_return_1h","next_return","ret_1h")]
        ret = None
        for v in candidates:
            try:
                n = float(v)
                if np.isfinite(n): ret = n; break
            except (TypeError, ValueError): pass
        if ret is None: return None
        # P0-B3 修复：原代码用 50/50 猜测格式，改为元数据明确指定。
        # 优先级：sample.return_format / labels.return_format / 启发式
        fmt = str(sample.get("return_format",
                  labels.get("return_format", "auto") if labels else "auto")).strip().lower()
        if fmt == "pct":
            ret /= 100.0
        elif fmt == "decimal":
            pass   # 已是小数格式（0.05 = 5%）
        else:
            # auto 启发：|ret| > 100 一定是百分数；2 < |ret| <= 100 视情况
            if abs(ret) > 100.0:
                ret /= 100.0
            elif abs(ret) > 2.0:
                # 启发：单期回报 > 200% 极罕见，故视为百分数
                ret /= 100.0
        if str(signal.get("action","")).upper() in {"SHORT","SELL"}: ret = -ret
        return max(-0.99, min(10.0, ret))


# ══════════════════════════════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════════════════════════════

def _result_sort_key(item: Dict[str, Any]) -> Tuple[float, float, float, float, str]:
    return (
        float(item.get("group_sort_score", 0) or 0),
        float(item.get("opportunity_score", item.get("score", 0)) or 0),
        float(item.get("score", 0) or 0),
        float(item.get("volume_24h", 0) or 0),
        str(item.get("symbol", "")),  # 确定性 tiebreaker
    )

def _dedupe_results(results: List[Dict[str, Any]], dedupe_by_symbol: bool = True) -> List[Dict[str, Any]]:
    best: Dict = {}
    for item in results:
        sym = str(item.get("symbol",""))
        key = sym if dedupe_by_symbol else (sym, str(item.get("source_strategy","")))
        ex = best.get(key)
        if ex is None:
            best[key] = item
            continue
        new_key = _result_sort_key(item)
        old_key = _result_sort_key(ex)
        if new_key > old_key:
            best[key] = item
        elif new_key == old_key:
            # 同分时：优先保留共识结果（多引擎共振 > 单引擎个体）
            is_new_consensus = "共振" in str(item.get("category", ""))
            is_old_consensus = "共振" in str(ex.get("category", ""))
            if is_new_consensus and not is_old_consensus:
                best[key] = item
    return list(best.values())

def _safe_number(value, default: float = 0.0) -> float:
    try:
        n = float(value)
    except (TypeError, ValueError):
        return default
    return n if np.isfinite(n) else default

def _recent_kline_move_pct(rows, lookback: int = 6) -> float:
    if not rows or not isinstance(rows, (list,tuple)) or len(rows) < 2:
        return 0.0
    try:
        end = float(rows[-1][4]); start = float(rows[max(0,len(rows)-lookback-1)][4])
    except (TypeError, ValueError, IndexError):
        return 0.0
    if not np.isfinite(end) or not np.isfinite(start) or start <= 0:
        return 0.0
    return (end/start - 1.0)*100.0

def _first_number(items: List[Dict], key: str) -> float:
    for item in items:
        try:
            v = float(item.get(key, 0) or 0)
            if v: return v
        except (TypeError, ValueError): pass
    return 0.0

def _merge_ranking_factors(items: List[Dict], fallback: float) -> Dict[str, float]:
    keys = ["trend","trigger","volume","location","freshness","risk"]
    merged = {}
    for key in keys:
        vals = []
        for item in items:
            f = item.get("ranking_factors") or {}
            try: vals.append(float(f.get(key, fallback)))
            except (TypeError, ValueError): pass
        merged[key] = sum(vals)/len(vals) if vals else fallback
    return merged


STRATEGY_NAME = "AI截面五引擎组合扫描器"
STRATEGY_TYPE = "scan"
STRATEGY_CLASS = AICrossSectionDualFactorComboScanner
BACKTEST_CLASS = AICrossSectionDualFactorComboScanner
