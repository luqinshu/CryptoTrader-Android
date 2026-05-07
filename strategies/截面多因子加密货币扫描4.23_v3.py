"""
截面多因子加密货币交易对扫描策略（精修版 v3 → v3.1）

v3fix → v3.1 新增特性 (2025-04)
─────────────────────────────────────────────────────
【用户需求：避免建仓后遇到小时级别回调】
1. 1H 趋势时效性检测（新增）
   _measure_trend_age(close, fast=12, slow=34, direction) 计算 EMA 排列
   已连续维持多少根 1H K 线。超过 max_h1_trend_age（默认 12）后对
   score 施加递增惩罚，阻止在"老趋势末端"建仓。

2. 3m 回调时效性检测（强化）
   m3_staleness_bars 原本只输出到 details 做展示，v3.1 将其正式
   纳入 passed 条件和 score 惩罚：超过 max_m3_staleness_bars（默认 15
   根 = 45 分钟）的回调企稳信号会被惩罚，不再轻易放行。

3. 新增 CONFIG_SCHEMA 条目
   - max_h1_trend_age (default 12)：1H 趋势最大延续根数
   - h1_trend_age_penalty (default 8.0)：超出后 score 惩罚量
   - m3_freshness_penalty (default 6.0)：3m 回调过旧 score 惩罚量
   - bonus_freshness_score (default 3.0)：两项时效均通过时加分

4. FactorSnapshot 新增 slot: h1_trend_age

5. 修复 _sync_config 每次 on_bar 重建 config 导致 scanner 缓存
   （last_factor_weights 等）在不同 bar 之间丢失的问题。
   → 改为首次调用时同步一次（_synced 标志位）。

v3 → v3fix 逻辑错误修复 (2024-04-24)
─────────────────────────────────────────────────────
【关键逻辑修复】
1. _micro_pullback_continuation 硬编码窗口越界 [严重]
   [-28:-12]/[-12:-stab_bars]/[-6:-1] 固定偏移在 m3=36 根时切片重叠/空切片。
   → 改为动态窗口分段（与 AI v3/DRL v2 一致）。

2. is_monotonic_increasing/decreasing 过严 [严重]
   浮点数据几乎永远不满足严格单调，导致企稳确认永远失败。
   → 改为宽松判断：企稳段不创新低/高 + EMA方向 + 大方向收盘。

3. m3 score 单位混乱 [中等]
   impulse_pct * 0.45 + pullback_pct * 0.75（百分比直接乘）远超 [-2,2]，
   clamp 后几乎永远是满分，无区分度。
   → 先归一化到 [0,1] 再合成。

4. m3_pullback_min_pct 默认值不一致 [中等]
   CONFIG_SCHEMA=0.50，代码 config.get(..., 0.35) → 统一为 0.50。

5. _historical_h1_factors RSI alignment 硬编码 50.0 [严重]
   IC 计算中 rsi_alignment 始终为常数 50.0，corr 恒为 nan/0，
   导致该因子的 Rolling IC 完全无效。
   → 用 1H-14 和 1H-56 RSI 近似 1H/4H RSI 对齐。

6. IC 动量定义与实盘不一致 [严重]
   历史 IC：m4h = _pct_change(close, 24)（24 根 1H）
   实盘评分：m4h = _pct_change(h4['c'], 12)（12 根 4H ≈ 48 根 1H）
   → 统一为 48 根 1H，使 IC 权重校准到正确因子空间。

7. Orthogonalization 未在 IC 计算中应用 [严重]
   实盘评分在正交化后的因子空间计算 z-score，但 IC 在原始因子
   空间计算 → 动态权重校准到了错误的因子空间。
   → 每个历史时间截面也调用 _orthogonalize_factors 后再算 IC。

8. early_trend_trigger 双重计分 [中等]
   _single_asset_score 中 early_p 已包含 early_trend_trigger 贡献，
   后面又 if abs(trigger) >= min_trigger: score += bonus → 重复加分。
   → 移除重复的 bonus 加分。

9. momentum_decay 评分过于二值化 [低]
   _clamp(50 + decay * 15, ...) 中 *15 使正常 -3~+3 范围触发
   0/100 极端值，退化为二值信号。
   → 改为 *6，保持 [-5,+5] 范围映射到 [20,80]，保留连续区分度。

10. short_reversal + oi_heat 从未纳入单标的评分 [中等]
     _factor_weights 中有权重，_build_snapshot 中计算了，但
     _single_asset_score 完全没用到 → 因子信息丢失。
     → 补充 sr_p + oi_p 到评分公式。

11. 3m 回调无时效性检查 [低]
     v3 的 _micro_pullback_continuation 没有检测回调是否过旧。
     → 增加 staleness_bars 输出 + 信号文本展示。
─────────────────────────────────────────────────────

v2 → v3 新增特性
─────────────────────────────────────────────────────
A) 链上数据因子（on-chain）
   - whale_flow：大户净流入（>$100k 交易流入-流出）/市值，正=鲸鱼买入
   - exchange_netflow：交易所净流入，负=提币（看涨信号）
   - active_addresses：活跃地址数 log，衡量链上活跃度
   - nvt_signal：NVT 信号（市值/链上交易量），低=低估
   数据来源：从 symbol.extra_data.get('on_chain', {}) 读取。
   如果没有链上数据，这些因子填 nan → z-score 变 0，不影响总分。

B) 多周期动量衰减检测（momentum decay）
   - momentum_decay：对比 1H/4H/1D 三个周期的动量，
     如果短周期动量 < 长周期动量 → 动量在衰减（趋势可能转弱）
     decay = (momentum_1h/scale_1h) - (momentum_1d/scale_1d)
     负值 = 短周期弱于长周期 = 衰减 → 降低多头信号可信度
   - momentum_acceleration：动量二阶导（momentum_1h 近 2 个周期的变化）
     正=加速，负=减速

C) 因子正交化（orthogonalization）
   在截面排序前，对高相关因子做 Gram-Schmidt 正交化：
   - momentum_1h / momentum_4h / momentum_1d 三个动量因子高度相关，
     保留 momentum_4h 作为"锚因子"，正交化 1h 和 1d
   - trend_quality 和 efficiency_ratio 高度相关，正交化 efficiency_ratio
   正交化后每个因子只保留"独有信息"，避免同质因子重复投票。
   可通过 config 'use_orthogonalization' 开关控制。

D) IC 衰减半衰期（IC decay half-life）
   v2 的 Rolling IC 对所有历史截面等权平均（EWMA 只做了平滑）。
   v3 引入半衰期衰减：距今越远的截面 IC 贡献越小。
   ic_half_life_points 控制衰减速度（默认 15 个截面 ≈ 15 天）。
   效果：最近的因子有效性变化能更快反映到权重调整中。
─────────────────────────────────────────────────────
"""

from __future__ import annotations
from math import log, log2, exp
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from strategies._shared.indicators import (
    _to_df, _aggregate_bars, _ema, _rsi_wilder, _adx, _efficiency_ratio,
    _volume_zscore, _robust_zscore, _measure_trend_age, _micro_pullback_continuation,
    _safe_float, _clamp, _calc_atr, _calc_volume_delta, _calc_vwap,
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

CONFIG_SCHEMA = {
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
    'position_size':          {'type':'float','default':0.1,       'label':'回测仓位比例'},
    'take_profit_pct':        {'type':'float','default':5.5,       'label':'止盈%'},
    'stop_loss_pct':          {'type':'float','default':3.2,       'label':'止损%'},
}
_DEFAULT_CONFIG = {k: v['default'] for k, v in CONFIG_SCHEMA.items()}


def _backtest_config(config: Dict[str, Any]) -> Dict[str, Any]:
    cfg = dict(config or {})
    cfg['min_score'] = float(cfg.get('backtest_min_score', cfg.get('min_score', 70.0)) or 70.0)
    return cfg


def _scan_symbol_with_config(symbol, config: Dict[str, Any]) -> Dict:
    snap = _build_snapshot(symbol, config)
    if not snap.valid:
        return _failed_result(symbol, snap.reason)
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
        self.config = {**_DEFAULT_CONFIG, **(config or {})}
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
        sym = _symbol_from_backtest_data(data, cfg)
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
            snap = _build_snapshot(sym, self.config)
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
            if snap.momentum_4h <= 0 and snap.momentum_1d <= 2 and not early_ok: continue
            # v3: 动量衰减过滤 — 如果严重衰减则跳过
            if snap.momentum_decay < -3.0 and not early_ok: continue
            # v3.1: 趋势时效过滤 — 趋势延续过长则降低优先级（score惩罚在_single_asset_score中已处理）
            max_age = int(_safe_float(self.config.get('max_h1_trend_age', 12), 12))
            max_stale = int(_safe_float(self.config.get('max_m3_staleness_bars', 15), 15))
            if snap.h1_trend_age > max_age * 2 and snap.m3_staleness_bars > max_stale: continue  # 两项都严重超标则直接跳过
            edge = float(long_edge.loc[sn])
            score = _edge_to_score(edge, snap)
            if edge < min_edge or score < min_score: continue
            results.append(_build_scan_result(snapshot=snap,score=score,direction='BUY',edge=edge,
                factor_scores=z.loc[sn].to_dict(),passed=True,rank=rank,universe_size=usize,
                category="截面多因子多头",config=self.config,weights=weights,ic_snapshot=ic_snap))

        if allow_short:
            short_edge = -long_edge
            for rank, sn in enumerate(short_edge.sort_values(ascending=False).head(int(self.config.get('top_n_short',8))).index, 1):
                snap = snap_map.get(sn)
                if not snap or snap.atr_pct > hard_atr: continue
                early_ok = snap.early_trend_trigger <= -float(self.config.get('early_trend_min_trigger',0.18))
                if snap.momentum_4h >= 0 and snap.momentum_1d >= -3 and not early_ok: continue
                if snap.momentum_decay > 3.0 and not early_ok: continue  # v3: 空头也检查
                # v3.1: 空头时效过滤
                max_age_s = int(_safe_float(self.config.get('max_h1_trend_age', 12), 12))
                max_stale_s = int(_safe_float(self.config.get('max_m3_staleness_bars', 15), 15))
                if snap.h1_trend_age > max_age_s * 2 and snap.m3_staleness_bars > max_stale_s: continue
                edge = float(short_edge.loc[sn])
                score = _edge_to_score(edge, snap)
                if edge < min_edge or score < min_score: continue
                results.append(_build_scan_result(snapshot=snap,score=score,direction='SELL',edge=edge,
                    factor_scores=z.loc[sn].to_dict(),passed=True,rank=rank,universe_size=usize,
                    category="截面多因子空头",config=self.config,weights=weights,ic_snapshot=ic_snap))

        results.sort(key=lambda r:(float(r.get('opportunity_score',r.get('score',0)) or 0),
                                   float(r.get('volume_24h',0) or 0)), reverse=True)
        return {'type':'cross_section_multi_factor','all_opportunities':results}

    def get_config_schema(self) -> Dict: return dict(CONFIG_SCHEMA)


# ══════════════════════════════════════════════
# 回测适配器
# ══════════════════════════════════════════════
class ZCrossSectionalMultiFactorBacktestStrategy:
    required_bars = ['1D','4H','1H']
    name = "截面多因子加密货币扫描(回测)"
    strategy_type = "backtest"
    def __init__(self, config=None):
        self.config = _backtest_config({**_DEFAULT_CONFIG, **(config or {})})
        self._bars = []
        self.last_analysis = {}
        self._scanner = ACrossSectionalMultiFactorScannerStrategy(self.config)
        self._synced = False  # v3.1: 只同步一次，避免反复覆盖 scanner 缓存
    def on_bar(self, bar, *a, **kw): return self._handle_bar(bar)
    def generate_signal(self, data, *a, **kw):
        if isinstance(data, dict) and data.get('klines_map'):
            self._ensure_sync()
            sym = _symbol_from_backtest_data(data, self.config)
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
        sym = _MinimalSymbol(inst_id=str(self.config.get('inst_id','BT') or 'BT'),
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
            self.config = _backtest_config({**_DEFAULT_CONFIG, **(self.config or {})})
            self._scanner.config = self.config
            self._synced = True
    def get_config_schema(self): return dict(CONFIG_SCHEMA)


class _MinimalSymbol:
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


def _empty_early_trend_features() -> Dict[str, float]:
    return {
        'trigger': 0.0,
        'early_trend_trigger': 0.0,
        'ema_compression_breakout': 0.0,
        'rsi_midline_turn': 0.0,
        'macd_hist_turn': 0.0,
        'donchian_breakout': 0.0,
        'volume_price_confirm': 0.0,
    }


def _early_trend_features(h1: pd.DataFrame) -> Dict[str, float]:
    """
    小时级趋势早启动/转折因子。
    设计重点是捕捉“均线刚从收敛区扩散 + 价格刚突破短箱体 + RSI/MACD 拐头 + 量价确认”，
    尽量避免只在趋势已经拉开很远后才给分。
    """
    if h1 is None or len(h1) < 58:
        return _empty_early_trend_features()
    close = h1['c'].astype(float)
    high = h1['h'].astype(float)
    low = h1['l'].astype(float)
    vol = h1['vol'].astype(float)
    price = float(close.iloc[-1])
    if price <= 0:
        return _empty_early_trend_features()

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

    rsi_series = _rsi_series_wilder(close, 14)
    rsi_now = float(rsi_series.iloc[-1]) if not rsi_series.empty else 50.0
    rsi_prev = float(rsi_series.iloc[-4]) if len(rsi_series) >= 4 else 50.0
    rsi_mid = _clamp((rsi_now - 50.0) / 16.0 + (rsi_now - rsi_prev) / 12.0, -2.0, 2.0)

    macd_hist = _macd_hist_series(close)
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

    trigger = _clamp(
        ema_component * 0.30 + donchian * 0.24 + rsi_mid * 0.18 + macd_turn * 0.16 + volume_price * 0.12,
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
    }


def _rsi_series_wilder(close: pd.Series, period: int = 14) -> pd.Series:
    if len(close) < period + 2:
        return pd.Series([50.0] * len(close), index=close.index)
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100.0 - 100.0 / (1.0 + rs)).replace([np.inf, -np.inf], np.nan).fillna(50.0).clip(0.0, 100.0)


def _macd_hist_series(close: pd.Series) -> pd.Series:
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
        return _normalize_weights(weights)
    return _normalize_weights(usable)


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

def _build_snapshot(symbol, config) -> FactorSnapshot:
    klines = getattr(symbol, 'extra_data', {}).get('klines', {}) or {}
    m3 = _to_df(_get_klines(klines, '3m'))
    if m3.empty:
        m3 = _to_df(_get_klines(klines, '3M'))
    if m3.empty:
        m1 = _to_df(_get_klines(klines, '1m'))
        if len(m1) >= 120:
            m3 = _aggregate_bars(m1, 3)
    d1 = _to_df(_get_klines(klines, '1D'))
    h4 = _to_df(_get_klines(klines, '4H'))
    h1 = _to_df(_get_klines(klines, '1H'))
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
            btc_h1 = _to_df(_get_klines({'1H': btc_data} if isinstance(btc_data, list) else btc_data, '1H'))
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
    tq = _trend_quality(lp,ema21,ema55,spread,slope,adx)
    rv = _realized_vol_pct(h1['c'],48); atr = _atr_pct(h4)
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
    early = _early_trend_features(h1) if bool(config.get('enable_early_trend_factors', True)) else _empty_early_trend_features()

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
    # v3: 新增 short_reversal + oi_heat（之前被遗漏）
    sr_p      = _clamp(50+snap.short_reversal*15,0,100)*0.03
    oi_p      = _clamp(50+snap.oi_heat*5,0,100)*0.02
    # v3
    decay_p   = _clamp(50+snap.momentum_decay*6,0,100)*0.06      # v3fix: *15→*6 避免二值化
    accel_p   = _clamp(50+snap.momentum_acceleration*5,0,100)*0.04
    early_p   = _clamp(50+snap.early_trend_trigger*22,0,100)*0.07
    turn_p    = _clamp(50+snap.rsi_midline_turn*18+snap.macd_hist_turn*16,0,100)*0.04
    whale_p   = _clamp(50+_safe_float(snap.whale_flow,0)*200,0,100)*0.03
    nvt_p     = _clamp(50+_safe_float(snap.nvt_signal,0)*10,0,100)*0.02

    score = (trend_p+mom_p+lowvol_p+vol_p+fund_p+liq_p+macd_p+bb_p+eff_p+rsia_p+
             sr_p+oi_p+decay_p+accel_p+early_p+turn_p+whale_p+nvt_p)

    if snap.atr_pct>max_atr: score -= min(28,(snap.atr_pct-max_atr)*3.2)
    if snap.rsi_1h>78: score -= 5
    elif snap.rsi_1h<22: score -= 5
    if snap.momentum_decay < -2.0: score -= 4
    # v3fix: 移除 early_trend_trigger 的重复加分（early_p 已包含此因子）

    if snap.atr_pct>max_atr: direction='WAIT'; edge=0
    elif (snap.direction_bias=='BUY' and mom_raw>0) or snap.early_trend_trigger >= float(config.get('early_trend_min_trigger',0.18)):
        direction='BUY'; edge=score/100
    elif ((snap.direction_bias=='SELL' and mom_raw<0) or snap.early_trend_trigger <= -float(config.get('early_trend_min_trigger',0.18))) and bool(config.get('allow_short',True)):
        direction='SELL'; edge=score/100
    else: direction='WAIT'; edge=0

    # v3.1: 时效性调整（在确定方向后计算，惩罚/奖励应用到最终 score）
    time_adj = _timeliness_score_adjustment(snap, config, direction)
    score = _clamp(score + time_adj, 0.0, 100.0)

    fs = {'trend':trend_p,'momentum':mom_p,'low_vol':lowvol_p,'volume':vol_p,
          'funding':fund_p,'liquidity':liq_p,'macd':macd_p,'bb':bb_p,
          'efficiency':eff_p,'rsi_align':rsia_p,'short_reversal':sr_p,'oi_heat':oi_p,
          'decay':decay_p,'accel':accel_p,
          'early_trend':early_p,'turn':turn_p,'whale':whale_p,'nvt':nvt_p}
    return round(_clamp(score,0,100),2), direction, edge, fs


# ══════════════════════════════════════════════
# 因子权重 + Rolling IC + 半衰期
# ══════════════════════════════════════════════

def _factor_weights():
    return {
        'momentum_1h':0.07,'momentum_4h':0.14,'momentum_1d':0.11,
        'short_reversal':0.02,'trend_quality':0.12,'low_volatility':0.06,
        'liquidity':0.06,'volume_impulse':0.06,'funding_carry':0.02,
        'funding_contrarian':0.02,'oi_heat':0.02,
        'macd_momentum':0.04,'bb_percentb':0.02,'vol_zscore':0.02,
        'close_strength':0.02,'efficiency_ratio':0.02,'rsi_alignment':0.01,
        # v3
        'momentum_decay':0.05,'momentum_acceleration':0.03,
        'early_trend_trigger':0.07,'ema_compression_breakout':0.03,
        'rsi_midline_turn':0.025,'macd_hist_turn':0.025,
        'donchian_breakout':0.025,'volume_price_confirm':0.025,
        'whale_flow':0.03,'exchange_netflow':0.02,'active_addresses':0.02,'nvt_signal':0.02,
    }

def _resolve_factor_weights(symbols, config):
    base = _normalize_weights(_factor_weights())
    if not bool(config.get('use_dynamic_ic_weights',True)): return base, {}
    ic_snap = _rolling_ic_snapshot(symbols, config)
    if not ic_snap: return base, {}
    hist = set(ic_snap.keys())
    dyn = {n: base.get(n,0)*max(float(ic_snap.get(n,0) or 0),0) for n in hist}
    if sum(dyn.values())<=1e-12: return base, ic_snap
    ht = sum(base.get(n,0) for n in hist)
    hd = _normalize_weights(dyn)
    blend = _clamp(float(config.get('ic_weight_blend',0.62)),0,1)
    mixed = dict(base)
    for n in hist: mixed[n] = base.get(n,0)*(1-blend) + hd.get(n,0)*ht*blend
    capped = _cap_weights(_normalize_weights(mixed), float(config.get('max_dynamic_factor_weight',0.28)))
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
        h1 = _to_df(_get_klines(kl,'1H'))
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
        'trend_quality':_trend_quality(price,ema21,ema55,spread,slope,adx_v),
        'low_volatility':-_realized_vol_pct(close,48),
        'liquidity':log(max(float(getattr(symbol,'volume_24h',0) or hist['vol'].tail(24).sum()),1.0)),
        'volume_impulse':_volume_ratio(hist['vol'],24),
        'macd_momentum':_macd_momentum(close),'bb_percentb':_bb_percentb(close),
        'vol_zscore':_volume_zscore(hist['vol']),'close_strength':_avg_close_strength(hist,6),
        'efficiency_ratio':_efficiency_ratio(close,20),'rsi_alignment':ra,
        'momentum_decay':_momentum_decay(m1h,m4h,m1d),
        'momentum_acceleration':_momentum_acceleration(close,6),
        **_early_trend_features(hist),
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

def _failed_result(symbol, reason):
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
def _normalize_weights(w):
    c = {}
    for n, v in w.items():
        fv = _safe_float(v, 0.0)
        c[n] = max(fv, 0.0) if np.isfinite(fv) else 0.0
    t=sum(c.values())
    return {n:v/t for n,v in c.items()} if t>1e-12 else dict(_factor_weights())

def _cap_weights(w, cap):
    cap=_clamp(cap,0.05,1.0); w=_normalize_weights(w)
    capped={};rem=[];rt=0;ct=0
    for n,v in w.items():
        if v>cap: capped[n]=cap; ct+=cap
        else: rem.append(n); rt+=v
    if not rem or ct>=1: return _normalize_weights(capped)
    budget=1-ct
    for n in rem: capped[n]=w[n]/rt*budget if rt>0 else budget/len(rem)
    return _normalize_weights(capped)

def _edge_to_score(edge, snap):
    score = 58+min(max(edge,0),2.2)*16
    if snap.volume_impulse>=1.4: score+=2
    if snap.atr_pct<=5: score+=2
    if abs(snap.funding_rate)<=0.07: score+=1.5    # v3fix: funding_rate已*100，0.07≈0.07%费率
    if snap.efficiency_ratio>=0.3: score+=1.5
    if snap.macd_momentum>0.5: score+=1
    if snap.momentum_decay>0.5: score+=1
    score -= min(18, max(snap.atr_pct-7.5,0)*2.4)
    return round(_clamp(score,0,100),2)

def _get_klines(km, bar):
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
def _trend_quality(price,e21,e55,spread,slope,adx):
    bull=price>e21>e55; bear=price<e21<e55; sc=30.0
    if bull or bear: sc+=28
    sc+=_clamp(abs(spread)/2.8*18,0,18); sc+=_clamp(abs(slope)/2*14,0,14)
    sc+=_clamp((adx-12)/18*10,0,10)
    return round(_clamp(sc,0,100),2)
def _realized_vol_pct(close,window=48):
    if len(close)<3: return 0.0
    ret=close.pct_change().dropna().tail(window)
    return float(ret.std(ddof=0)*np.sqrt(max(len(ret),1))*100) if not ret.empty else 0.0
def _atr_pct(df,period=14):
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
def _symbol_from_backtest_data(data, config):
    km=data.get('klines_map') or {}; h1_rows=_get_klines(km,'1H') or data.get('klines') or []
    h1=_to_df(h1_rows); lp=float(h1['c'].iloc[-1]) if not h1.empty else 0
    return _MinimalSymbol(inst_id=str(config.get('inst_id','BT') or 'BT'),last_price=lp,
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
    cfg={**_DEFAULT_CONFIG,**(config or {})}
    d1=_aggregate_bars(h1 if isinstance(h1,pd.DataFrame) else _to_df(h1),24)
    h4_=h4 if isinstance(h4,pd.DataFrame) else _to_df(h4)
    h1_=h1 if isinstance(h1,pd.DataFrame) else _to_df(h1)
    kl={'1H':_df_to_rows(h1_),'4H':_df_to_rows(h4_),'1D':_df_to_rows(d1)}
    sym=_MinimalSymbol(inst_id='ANALYZE',last_price=float(last_price),
        volume_24h=float(h1_['vol'].tail(24).sum()) if not h1_.empty else 0,
        price_change_24h=_pct_change(h1_['c'],24) if not h1_.empty else 0,
        extra_data={'klines':kl})
    snap=_build_snapshot(sym,cfg)
    if not snap.valid: return {'valid':False,'reason':snap.reason,'score':0,'direction':'WAIT'}
    score,direction,edge,fs=_single_asset_score(snap,cfg)
    return {'valid':True,'score':score,'direction':direction,'edge':edge,'factor_scores':fs}
def klines_list_to_df(rows): return _to_df(rows)

STRATEGY_NAME = "截面多因子加密货币扫描"
STRATEGY_TYPE = "scan"
STRATEGY_CLASS = ACrossSectionalMultiFactorScannerStrategy
BACKTEST_CLASS = ZCrossSectionalMultiFactorBacktestStrategy
