"""
经典波段八策略组合扫描器 v2

一次扫描自动分为八类机会：
  1. 突破启动   2. 新高突破   3. 单边趋势   4. 趋势回踩
  5. 背离反转   6. 超跌反转   7. 中继再启动  8. 趋势回踩二次启动

═══════════════════ v2 审计缺陷修复 ═══════════════════
【Critical 修复】
  A. 漏写 STRATEGY_CLASS / BACKTEST_CLASS 导出 → 扫描器引擎无法加载本策略
  B. scan_symbol / scan_all_symbols 均未执行 _init_conditions 中声明的
     成交量过滤条件 → 0 成交量的垃圾对也能通过
  C. _annotate_result 调用 apply_strategy_lifecycle_guard 无 try/except →
     任何守门异常会让整个结果被丢弃，改为安全包裹

【High 修复】
  D. _build_child_strategies 仅 print 加载失败，加载 0 个子策略时调用方
     无任何感知 → 增加警告计数，结果字典加 'child_load_warnings' 字段
  E. scan_all_symbols 对每个交易对都重复获取 klines（通过子策略） →
     新增 _get_symbol_klines 统一提取，避免重复 dict.get 开销

【Medium 修复】
  F. 传给子策略的 child_config 含大量与其无关的组合键（如
     lifecycle_*、m3_reversal_window 等），子策略 .get() 虽然安全但
     造成语义污染 → 过滤，只传子策略需要的通用键
  G. group_sort_score 分离设计与子策略实际 opportunity_score 脱节，
     仅凭机械的 group_sort_score 无法判断"最佳机会" → 在 scan_symbol
     返回前改为按 opportunity_score 优先，group_sort_score 次之排序

═══════════════════ v2 新增两条硬性全局门槛 ═══════════════════

  Gate A — 3m 突破回踩确认（gate_a_enable，默认开启）
  ─────────────────────────────────────────────────────
  任何子策略评分通过后，在 3m 周期上必须同时满足：
    ① 找到最近一个方向性局部极值（多头=近期摆动高，空头=摆动低）
       作为"原突破位"
    ② 从该突破位之后出现了明显回调（多头价格下行 >= gate_a_min_pb_pct %；
       空头价格上行 >= gate_a_min_pb_pct %）
    ③ 回调极值不得穿越原突破位超过 gate_a_max_break_pct %（否则视为
       突破位已被打穿，结构失效）
    ④ 当前收盘仍在突破位同侧（多头 close > 突破位 × (1 - tol)；
       空头 close < 突破位 × (1 + tol)），回踩后价格已收回

  Gate B — H1 趋势延续无回调迹象（gate_b_enable，默认开启）
  ─────────────────────────────────────────────────────
  在小时线上必须同时满足：
    ① EMA8 > EMA21 > EMA55（多头完整顺排）或 EMA8 < EMA21 < EMA55（空头）
    ② 近 gate_b_lookback 根 H1 K 线中，反向收盘（多头=收盘低于EMA21；
       空头=收盘高于EMA21）不超过 1 根（允许 1 根噪声影线收盘）
    ③ H1 RSI 站稳趋势侧（多头 RSI >= gate_b_min_rsi_long；
       空头 RSI <= gate_b_max_rsi_short）
    ④ H1 MACD 柱状线与趋势方向一致（多头 hist > 0；空头 hist < 0）

两条 Gate 均可通过配置独立关闭（gate_a_enable / gate_b_enable = False）。
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from strategies._shared.indicators import _rsi_wilder, _to_df

logger = logging.getLogger(__name__)

try:
    from src.scanner.base_scanner import BaseScannerStrategy, ScanCondition, ScannerSymbol
    from src.scanner.strategy_lifecycle import apply_strategy_lifecycle_guard
    _HAS_SCANNER_BASE = True
except ImportError:
    BaseScannerStrategy = object
    ScanCondition = None
    ScannerSymbol = None
    apply_strategy_lifecycle_guard = None
    _HAS_SCANNER_BASE = False


# ══════════════════════════════════════════════
# 配置 Schema
# ══════════════════════════════════════════════
CONFIG_SCHEMA: Dict = {
    # ── 通用 ──
    'min_volume_24h':       {'type': 'float', 'default': 1_000_000,  'label': '组合最小24H成交额'},
    'min_score':            {'type': 'float', 'default': 72.0,       'label': '子策略最低触发评分'},
    'backtest_min_score':   {'type': 'float', 'default': 68.0,       'label': '回测最低入场评分'},
    'top_n_per_group':      {'type': 'int',   'default': 12,         'label': '每组保留数量'},
    'take_profit_pct':      {'type': 'float', 'default': 5.0,        'label': '回测固定止盈%'},
    'stop_loss_pct':        {'type': 'float', 'default': 3.0,        'label': '回测固定止损%'},
    'conservative_same_bar_exit': {'type': 'bool', 'default': True,  'label': '同K线止盈止损按保守处理'},

    # ── 策略生命周期守门 ──
    'enable_strategy_lifecycle_guard':          {'type': 'bool',  'default': True,  'label': '启用策略前提/失效守门'},
    'lifecycle_warning_penalty':                {'type': 'float', 'default': 4.0,   'label': '前提不足单项降权'},
    'lifecycle_failure_score_cap':              {'type': 'float', 'default': 67.0,  'label': '失效信号评分上限'},
    'lifecycle_new_high_min_trend_quality':     {'type': 'float', 'default': 68.0,  'label': '新高突破最低趋势质量'},
    'lifecycle_directional_min_trend_quality':  {'type': 'float', 'default': 70.0,  'label': '单边趋势最低趋势质量'},
    'lifecycle_pullback_min_trend_quality':     {'type': 'float', 'default': 68.0,  'label': '趋势回踩最低趋势质量'},
    'lifecycle_new_high_min_adx':               {'type': 'float', 'default': 18.0,  'label': '新高突破最低ADX'},
    'lifecycle_directional_min_adx':            {'type': 'float', 'default': 20.0,  'label': '单边趋势最低ADX'},
    'lifecycle_pullback_min_adx':               {'type': 'float', 'default': 16.0,  'label': '趋势回踩最低ADX'},
    'lifecycle_compression_max_atr_ratio':      {'type': 'float', 'default': 0.88,  'label': '压缩爆发ATR收缩上限'},
    'lifecycle_compression_max_band_ratio':     {'type': 'float', 'default': 0.86,  'label': '压缩爆发布林带收缩上限'},
    'lifecycle_continuation_max_contraction_ratio': {'type': 'float', 'default': 0.88, 'label': '中继再启动缩量系数上限'},

    # ── 子策略特有参数（透传） ──
    'require_3m_reversal_retest':   {'type': 'bool',  'default': True,  'label': '超跌反转要求3m突破回踩不破起点'},
    'm3_reversal_window':           {'type': 'int',   'default': 80,    'label': '超跌反转3m结构窗口'},
    'm3_pullback_window':           {'type': 'int',   'default': 12,    'label': '超跌反转3m回踩窗口'},
    'm3_neckline_bars':             {'type': 'int',   'default': 10,    'label': '超跌反转3m突破位样本'},
    'm3_breakout_buffer_pct':       {'type': 'float', 'default': 0.12,  'label': '超跌反转3m突破缓冲%'},
    'm3_origin_tolerance_pct':      {'type': 'float', 'default': 0.18,  'label': '超跌反转3m起点容忍跌破%'},
    'm3_min_pullback_bars':         {'type': 'int',   'default': 3,     'label': '3m最少回踩确认根数'},
    'require_3m_continuation':      {'type': 'bool',  'default': True,  'label': '趋势回踩要求3m三段式延续确认'},
    'm3_window':                    {'type': 'int',   'default': 60,    'label': '趋势回踩3m观察窗口'},
    'max_pullback_pct':             {'type': 'float', 'default': 3.2,   'label': '趋势回踩最大价格偏离%'},
    'max_pullback_atr':             {'type': 'float', 'default': 1.5,   'label': '趋势回踩最大ATR倍数'},
    'min_confirm_volume_ratio':     {'type': 'float', 'default': 1.3,   'label': '趋势回踩确认最小量比'},
    'require_3m_restart':           {'type': 'bool',  'default': True,  'label': '二次启动要求3m三段式确认'},
    'm3_swing_window':              {'type': 'int',   'default': 50,    'label': '二次启动3m观察窗口'},
    'max_pullback_distance_pct':    {'type': 'float', 'default': 3.5,   'label': '二次启动最大回踩距离%'},
    'max_key_level_retest_pct':     {'type': 'float', 'default': 2.2,   'label': '二次启动关键位回测容差%'},
    'min_restart_volume_ratio':     {'type': 'float', 'default': 1.25,  'label': '二次启动最小量比'},

    # ── Gate A：3m 突破回踩确认 ──
    'gate_a_enable':            {'type': 'bool',  'default': True,  'label': '[Gate A] 启用3m突破回踩确认门槛'},
    'gate_a_m3_window':         {'type': 'int',   'default': 80,    'label': '[Gate A] 3m搜索窗口（根数）'},
    'gate_a_swing_left':        {'type': 'int',   'default': 5,     'label': '[Gate A] 突破位摆动左侧确认根数'},
    'gate_a_swing_right':       {'type': 'int',   'default': 3,     'label': '[Gate A] 突破位摆动右侧确认根数'},
    'gate_a_min_pb_pct':        {'type': 'float', 'default': 0.25,  'label': '[Gate A] 最小回调幅度%（相对突破位）'},
    'gate_a_max_break_pct':     {'type': 'float', 'default': 0.20,  'label': '[Gate A] 允许穿越突破位最大容忍%'},
    'gate_a_recover_required':  {'type': 'bool',  'default': True,  'label': '[Gate A] 要求当前价格已回到突破位同侧'},

    # ── Gate B：H1 趋势延续无回调迹象 ──
    'gate_b_enable':            {'type': 'bool',  'default': True,  'label': '[Gate B] 启用H1趋势延续确认门槛'},
    'gate_b_lookback':          {'type': 'int',   'default': 6,     'label': '[Gate B] H1近N根检查反向收盘'},
    'gate_b_max_reverse_bars':  {'type': 'int',   'default': 1,     'label': '[Gate B] 允许最多N根反向收盘（噪声容忍）'},
    'gate_b_min_rsi_long':      {'type': 'float', 'default': 50.0,  'label': '[Gate B] 多头最低H1 RSI'},
    'gate_b_max_rsi_short':     {'type': 'float', 'default': 50.0,  'label': '[Gate B] 空头最高H1 RSI'},
    'gate_b_require_macd':      {'type': 'bool',  'default': True,  'label': '[Gate B] 要求MACD柱状线方向与趋势一致'},
    'gate_b_ema_fast':          {'type': 'int',   'default': 8,     'label': '[Gate B] EMA快线周期'},
    'gate_b_ema_mid':           {'type': 'int',   'default': 21,    'label': '[Gate B] EMA中线周期'},
    'gate_b_ema_slow':          {'type': 'int',   'default': 55,    'label': '[Gate B] EMA慢线周期'},

    # ── v3: 市场环境 + 质量增强 + 交易辅助 ──
    'enable_score_normalization':  {'type': 'bool',  'default': True,  'label': '[v3] 跨策略z-score评分归一化'},
    'enable_btc_market_filter':    {'type': 'bool',  'default': True,  'label': '[v3] BTC市场环境过滤(暴跌降权山寨多头)'},
    'btc_dump_threshold_pct':      {'type': 'float', 'default': -2.5,  'label': '[v3] BTC暴跌阈值%'},
    'btc_dump_penalty':            {'type': 'float', 'default': 10.0,  'label': '[v3] BTC暴跌时多头降分'},
    'enable_funding_filter':       {'type': 'bool',  'default': True,  'label': '[v3] 资金费率极端检测'},
    'funding_extreme_pct':         {'type': 'float', 'default': 0.10,  'label': '[v3] 极端费率阈值%'},
    'funding_penalty':             {'type': 'float', 'default': 6.0,   'label': '[v3] 极端费率降分'},
    'enable_vol_quality_check':    {'type': 'bool',  'default': True,  'label': '[v3] 成交量质量检测(防刷量)'},
    'vol_conc_threshold':          {'type': 'float', 'default': 0.65,  'label': '[v3] 成交量集中度阈值'},
    'enable_atr_stop_target':      {'type': 'bool',  'default': True,  'label': '[v3] 输出ATR止损/止盈建议'},
    'stop_atr_multiplier':         {'type': 'float', 'default': 2.0,   'label': '[v3] 止损ATR倍数'},
    'target_atr_multiplier':       {'type': 'float', 'default': 3.0,   'label': '[v3] 止盈ATR倍数'},
    'enable_signal_persistence':   {'type': 'bool',  'default': True,  'label': '[v3] 信号持续性追踪'},
    'persistence_scans':           {'type': 'int',   'default': 2,     'label': '[v3] 连续出现N次才稳定加分'},
    'persistence_bonus':           {'type': 'float', 'default': 4.0,   'label': '[v3] 稳定出现加分'},
}

# 只向子策略透传的通用键（避免语义污染）
_CHILD_COMMON_KEYS = {
    'min_score', 'min_volume_24h', 'backtest_min_score',
    'take_profit_pct', 'stop_loss_pct', 'conservative_same_bar_exit',
    'max_pullback_pct', 'max_pullback_atr', 'min_confirm_volume_ratio',
    'require_3m_continuation', 'm3_window', 'm3_neckline_bars',
    'm3_min_pullback_bars', 'm3_breakout_buffer_pct', 'm3_origin_tolerance_pct',
    'require_3m_reversal_retest', 'm3_reversal_window', 'm3_pullback_window',
    'require_3m_restart', 'm3_swing_window',
    'max_pullback_distance_pct', 'max_key_level_retest_pct', 'min_restart_volume_ratio',
}


# ══════════════════════════════════════════════
# 主扫描器类
# ══════════════════════════════════════════════
class SwingEightStrategyComboScanner(BaseScannerStrategy if _HAS_SCANNER_BASE else object):
    required_bars = ['1D', '4H', '1H', '3m']
    strategy_type = 'scan'
    name = '波段八策略组合扫描器'
    description = '突破启动/新高突破/单边趋势/趋势回踩/背离反转/超跌反转/中继再启动/二次启动 + 3m回踩不破位 + H1趋势延续'

    CATEGORY_ORDER = [
        ('突破启动',        '波段平台突破扫描_4.21_v2.py',        'BreakoutSwingScanner',              1000),
        ('新高突破',        '新高突破扫描.py',                     'NewHighBreakoutScanner',            999),
        ('单边趋势',        '单边趋势跟随扫描_4.21_v2.py',        'DirectionalTrendFollowScanner',     998),
        ('趋势回踩',        '波段趋势回踩扫描_4.21_v3.py',        'TrendPullbackSwingScanner',         997),
        ('背离反转',        '背离反转扫描_4.21_v2.py',            'DivergenceReversalScanner',         996),
        ('超跌反转',        '波段超跌反转扫描_4.21_v3.py',        'OversoldReversalSwingScanner',      995),
        ('中继再启动',      '波段缩量中继再启动扫描_4.21_v2.py',  'ContinuationCompressionSwingScanner', 994),
        ('趋势回踩二次启动','趋势回踩二次启动筛选_4.21_v6.py',   'TrendPullbackRestartScanner',       993),
        ('量价背离',        '量价背离扫描策略.py',                  'VolumePriceDivergenceScanner',      980),
    ]

    def __init__(self, config: Dict = None):
        self.config = {k: v['default'] for k, v in CONFIG_SCHEMA.items()}
        self.config.update(config or {})
        self.child_strategies: List[Tuple[str, object, int]] = []
        self._child_load_warnings: List[str] = []
        # v3: 信号持续性状态 + 扫描计数
        self._signal_persistence: Dict[str, int] = {}
        self._scan_count: int = 0
        if _HAS_SCANNER_BASE and hasattr(super(), '__init__'):
            try:
                super().__init__(self.config)
            except Exception:
                pass
        self.child_strategies = self._build_child_strategies()

    # ── BaseScannerStrategy 抽象方法实现 ──
    def _init_conditions(self):
        if ScanCondition is None:
            return
        self.add_condition(ScanCondition(
            name='基础流动性',
            description='组合扫描统一流动性过滤',
            field='volume_24h',
            operator='>=',
            value=float(self.config.get('min_volume_24h', 1_000_000)),
        ))

    # ── 单标的扫描 ──
    def scan_symbol(self, symbol) -> Dict:
        # 前置：成交量过滤（修复：v1 scan_symbol 跳过了 _init_conditions 中的过滤）
        vol24 = float(getattr(symbol, 'volume_24h', 0) or 0)
        min_vol = float(self.config.get('min_volume_24h', 1_000_000))
        if vol24 < min_vol:
            return _fail_result(symbol.inst_id, f'成交额不足({vol24:.0f} < {min_vol:.0f})')

        # 提取 klines（统一一次，避免各子策略重复 dict.get）
        klines_map = _get_symbol_klines(symbol)
        m3_df  = _to_df(klines_map.get('3m') or klines_map.get('3M') or [])
        h1_df  = _to_df(klines_map.get('1H') or klines_map.get('1h') or [])

        direction = 'WAIT'
        best_result: Optional[Dict] = None

        for category_name, strategy, group_sort_score in self.child_strategies:
            try:
                result = strategy.scan_symbol(symbol)
            except Exception as exc:
                continue
            if not result.get('passed'):
                continue

            direction = str(result.get('direction', 'WAIT'))
            # 应用策略生命周期守门
            annotated = self._annotate_result(result, category_name, group_sort_score)
            if not annotated.get('passed'):
                continue

            # ══ Gate A：3m 突破回踩确认 ══
            if bool(self.config.get('gate_a_enable', True)) and direction in ('BUY', 'SELL'):
                gate_a_ok, gate_a_msg = _check_m3_retest_hold(m3_df, direction, self.config)
                if not gate_a_ok:
                    annotated['gate_a_fail'] = gate_a_msg
                    annotated.setdefault('details', {})['Gate_A'] = f'❌ {gate_a_msg}'
                    continue
                annotated.setdefault('details', {})['Gate_A'] = f'✅ {gate_a_msg}'

            # ══ Gate B：H1 趋势延续无回调迹象 ══
            if bool(self.config.get('gate_b_enable', True)) and direction in ('BUY', 'SELL'):
                gate_b_ok, gate_b_msg = _check_h1_trend_continuation(h1_df, direction, self.config)
                if not gate_b_ok:
                    annotated['gate_b_fail'] = gate_b_msg
                    annotated.setdefault('details', {})['Gate_B'] = f'❌ {gate_b_msg}'
                    continue
                annotated.setdefault('details', {})['Gate_B'] = f'✅ {gate_b_msg}'

            osc = float(annotated.get('opportunity_score', annotated.get('score', 0)) or 0)
            if best_result is None or osc > float(
                best_result.get('opportunity_score', best_result.get('score', 0)) or 0
            ):
                best_result = annotated

        if best_result:
            return best_result

        return _fail_result(symbol.inst_id, '组合策略均未通过（含双门槛过滤）')

    # ── 全量扫描（分组汇总 + v3.2 并发优化） ──
    def scan_all_symbols(self, symbols: List) -> Dict:
        top_n = int(self.config.get('top_n_per_group', 12))
        gate_a_on = bool(self.config.get('gate_a_enable', True))
        gate_b_on = bool(self.config.get('gate_b_enable', True))
        self._scan_count += 1

        # v3: BTC市场环境预检
        btc_ctx = self._extract_btc_context(symbols)

        # 预提取每个 symbol 的 klines
        klines_cache: Dict[str, Dict] = {}
        for sym in symbols:
            klines_cache[sym.inst_id] = _get_symbol_klines(sym)

        all_results: List[Dict] = []

        # v3.2: 并发扫描 — 每策略加锁防内部状态竞争
        import threading
        self._strategy_locks = {id(st): threading.Lock() for _, st, _ in self.child_strategies}
        from concurrent.futures import ThreadPoolExecutor, as_completed
        tasks = []
        for category_name, strategy, group_sort_score in self.child_strategies:
            for sym in symbols:
                tasks.append((category_name, strategy, group_sort_score, sym))

        min_vol = float(self.config.get('min_volume_24h', 1_000_000))
        with ThreadPoolExecutor(max_workers=min(16, len(tasks) or 1)) as executor:
            futures = {
                executor.submit(
                    self._scan_one, category_name, strategy, group_sort_score,
                    sym, min_vol, gate_a_on, gate_b_on, klines_cache
                ): (category_name, sym.inst_id)
                for category_name, strategy, group_sort_score, sym in tasks
            }
            for future in as_completed(futures):
                try:
                    result = future.result(timeout=20)
                except Exception:
                    continue
                if result:
                    all_results.append(result)

        # v3.3: 先 enrich market（BTC/费率/量质），再归一化（确保归一化反映最终分数）
        for r in all_results:
            sym = next((s for s in symbols if getattr(s, 'inst_id', '') == r.get('symbol', '')), None)
            if sym:
                r = self._enrich_result_market(r, sym, btc_ctx)
                km = klines_cache.get(sym.inst_id, {})
                r = self._enrich_result_trading_aids(r, sym, r.get('direction', 'WAIT'), km)
                # Gate A+B 质量乘积分（双通过 > 单通过）
                ga = r.get('details', {}).get('Gate_A', '')
                gb = r.get('details', {}).get('Gate_B', '')
                r['gate_quality'] = _calc_gate_quality(ga, gb)

        # v3.2: 全局归一化（跨策略跨品种可比）—— 在 enrich 之后计算
        if bool(self.config.get('enable_score_normalization', True)) and len(all_results) >= 5:
            scores = [float(r.get('opportunity_score', r.get('score', 0)) or 0) for r in all_results]
            mean_s, std_s = np.mean(scores), max(np.std(scores), 1e-6)
            for r in all_results:
                raw = float(r.get('opportunity_score', r.get('score', 0)) or 0)
                r['_global_norm'] = round((raw - mean_s) / std_s * 12 + 70, 1)

        # 分组排序 + top_n
        grouped: Dict[str, List] = {}
        for r in all_results:
            cat = r.get('category', '未分类')
            grouped.setdefault(cat, []).append(r)

        final_results: List[Dict] = []
        for cat, items in grouped.items():
            items.sort(
                key=lambda r: (
                    float(r.get('_global_norm', r.get('opportunity_score', r.get('score', 0)) or 0)),
                    float(r.get('opportunity_score', r.get('score', 0)) or 0),
                ),
                reverse=True,
            )
            for idx, item in enumerate(items[:top_n], 1):
                item['group_rank'] = idx
                item.setdefault('details', {})['全局归一化'] = f"{item.get('_global_norm', '-')}分"
                final_results.append(item)

        # v3.3: 跨类别方向冲突检测（同一品种多空同时出现 → 降权）
        final_results = self._resolve_directional_conflicts(final_results)

        # 全局排序输出
        final_results.sort(
            key=lambda r: (
                float(r.get('_global_norm', r.get('opportunity_score', r.get('score', 0)) or 0)),
                float(r.get('gate_quality', 1.0) or 1.0),
            ),
            reverse=True,
        )

        # v3: 持久化追踪
        if bool(self.config.get('enable_signal_persistence', True)):
            self._track_persistence(final_results)

        logger.info(f'[组合扫描器] #{self._scan_count}: {len(symbols)}品种 × {len(self.child_strategies)}策略 → {len(final_results)}条信号')

        return {
            'type': 'grouped_combo_scan',
            'all_opportunities': final_results,
            'child_load_warnings': self._child_load_warnings,
            'scan_round': self._scan_count,
        }

    def _scan_one(self, category_name, strategy, group_sort_score,
                   sym, min_vol, gate_a_on, gate_b_on, klines_cache) -> Optional[Dict]:
        """单品种单策略扫描 + 双门过滤（并发安全，策略调用加锁）"""
        vol24 = float(getattr(sym, 'volume_24h', 0) or 0)
        if vol24 < min_vol:
            return None
        # 策略内部状态保护
        lock = getattr(self, '_strategy_locks', {}).get(id(strategy))
        try:
            if lock:
                lock.acquire()
            result = strategy.scan_symbol(sym)
        except Exception:
            return None
        finally:
            if lock:
                lock.release()
        if not result.get('passed'):
            return None

        direction = str(result.get('direction', 'WAIT'))
        # _annotate_result 放在同一锁内（防止 lifecycle_guard 状态竞争）
        if lock:
            lock.acquire()
        try:
            annotated = self._annotate_result(result, category_name, group_sort_score)
        finally:
            if lock:
                lock.release()
        if not annotated.get('passed'):
            return None

        km = klines_cache.get(sym.inst_id, {})
        m3_df = _to_df(km.get('3m') or km.get('3M') or [])
        h1_df = _to_df(km.get('1H') or km.get('1h') or [])

        if gate_a_on and direction in ('BUY', 'SELL'):
            gate_a_ok, gate_a_msg = _check_m3_retest_hold(m3_df, direction, self.config)
            if not gate_a_ok:
                annotated.setdefault('details', {})['Gate_A'] = f'❌ {gate_a_msg}'
                return None
            annotated.setdefault('details', {})['Gate_A'] = f'✅ {gate_a_msg}'

        if gate_b_on and direction in ('BUY', 'SELL'):
            gate_b_ok, gate_b_msg = _check_h1_trend_continuation(h1_df, direction, self.config)
            if not gate_b_ok:
                annotated.setdefault('details', {})['Gate_B'] = f'❌ {gate_b_msg}'
                return None
            annotated.setdefault('details', {})['Gate_B'] = f'✅ {gate_b_msg}'

        return annotated

    # ── 跨类别方向冲突检测 ───────────────────────────────────────────
    def _resolve_directional_conflicts(self, results: List[Dict]) -> List[Dict]:
        """同一品种出现多空方向冲突 → 同时降权并标注"""
        by_symbol: Dict[str, List[Dict]] = {}
        for r in results:
            by_symbol.setdefault(str(r.get('symbol', '')), []).append(r)
        for sym, items in by_symbol.items():
            if len(items) < 2:
                continue
            longs = [it for it in items if str(it.get('direction','')).upper() in ('BUY','LONG')]
            shorts = [it for it in items if str(it.get('direction','')).upper() in ('SELL','SHORT')]
            if longs and shorts:
                penalty = 8.0
                for it in items:
                    it['score'] = round(max(0, float(it.get('score',0) or 0) - penalty), 2)
                    it.setdefault('details', {})['方向冲突'] = (
                        f'⚠ {sym}同时出现多头({len(longs)})与空头({len(shorts)})信号，各降{penalty}分')
        return results

    # ── v3: 市场环境 + 交易辅助方法 ───
    def _extract_btc_context(self, symbols: List) -> Dict:
        """提取 BTC 基准数据"""
        btc = next((s for s in symbols if 'BTC' in str(getattr(s, 'inst_id', '')).upper()), None)
        if not btc:
            return {}
        return {
            'btc_24h': float(getattr(btc, 'price_change_24h', 0) or 0),
            'btc_klines': (getattr(btc, 'extra_data', {}) or {}).get('klines', {}),
        }

    def _enrich_result_market(self, result: Dict, sym, btc_ctx: Dict) -> Dict:
        """市场环境过滤：BTC暴跌降权、资金费率检测、成交量质量"""
        d = result.setdefault('details', {})
        direction = str(result.get('direction', 'WAIT')).upper()
        score = float(result.get('score', 0) or 0)

        # BTC 暴跌检测 + 空头奖励
        if bool(self.config.get('enable_btc_market_filter', True)):
            btc_24h = btc_ctx.get('btc_24h', 0)
            threshold = float(self.config.get('btc_dump_threshold_pct', -2.5))
            is_btc_dump = btc_24h < threshold
            if direction == 'BUY' and is_btc_dump:
                penalty = float(self.config.get('btc_dump_penalty', 10.0))
                result['score'] = round(max(0, score - penalty), 2)
                d['BTC环境'] = f'⚠ BTC{btc_24h:+.1f}%<{threshold}%，多头降{penalty}分'
            elif direction == 'SELL' and is_btc_dump:
                bonus = float(self.config.get('btc_dump_penalty', 10.0)) * 0.5
                result['score'] = round(min(100, score + bonus), 2)
                d['BTC环境'] = f'🚀 BTC{btc_24h:+.1f}%<{threshold}%，空头环境有利+{bonus}分'
            elif not is_btc_dump:
                d['BTC环境'] = f'✓ BTC{btc_24h:+.1f}%正常'

        # 资金费率极端检测
        if bool(self.config.get('enable_funding_filter', True)):
            funding = float((getattr(sym, 'extra_data', {}) or {}).get('funding_rate', 0) or 0)
            ext_pct = float(self.config.get('funding_extreme_pct', 0.10)) / 100.0
            if direction == 'BUY' and funding > ext_pct:
                penalty = float(self.config.get('funding_penalty', 6.0))
                result['score'] = round(max(0, float(result.get('score', 0) or 0) - penalty), 2)
                d['资金费率'] = f'⚠ 多头拥挤({funding*100:.3f}%)，降{penalty}分'
            elif direction == 'SELL' and funding < -ext_pct:
                penalty = float(self.config.get('funding_penalty', 6.0))
                result['score'] = round(max(0, float(result.get('score', 0) or 0) - penalty), 2)
                d['资金费率'] = f'⚠ 空头拥挤({funding*100:.3f}%)，降{penalty}分'

        # 成交量质量检测
        if bool(self.config.get('enable_vol_quality_check', True)):
            rows = (getattr(sym, 'extra_data', {}) or {}).get('klines', {}).get('1H', [])
            if len(rows) >= 12:
                try:
                    vols = [float(r[5]) for r in rows[-12:] if len(r) > 5 and float(r[5]) > 0]
                    if len(vols) >= 8:
                        conc = max(vols) / max(sum(vols), 1e-9)
                        if conc > float(self.config.get('vol_conc_threshold', 0.65)):
                            result['score'] = round(max(0, float(result.get('score', 0) or 0) - 4.0), 2)
                            d['成交量质量'] = f'⚠ 集中度{conc:.0%}，疑似刷量降4分'
                except Exception:
                    pass

        return result

    def _enrich_result_trading_aids(self, result: Dict, sym, direction: str, km: Dict) -> Dict:
        """ATR止损/止盈建议"""
        if not bool(self.config.get('enable_atr_stop_target', True)):
            return result
        d = result.setdefault('details', {})
        rows = km.get('15m') or km.get('3m') or []
        if len(rows) < 16:
            return result
        try:
            trs = []
            for i in range(1, min(len(rows), 20)):
                h, l, pc = float(rows[-i][2]), float(rows[-i][3]), float(rows[-i-1][4])
                trs.append(max(h-l, abs(h-pc), abs(l-pc)))
            atr = float(np.mean(trs)) if trs else 0.0
            if atr <= 0:
                return result
            lp = float(getattr(sym, 'last_price', 0) or d.get('last_price', 0) or 0)
            if lp <= 0:
                return result
            sl_m = float(self.config.get('stop_atr_multiplier', 2.0))
            tp_m = float(self.config.get('target_atr_multiplier', 3.0))
            if direction == 'BUY':
                d['ATR止损'] = f'{lp - atr*sl_m:.6g} (-{atr*sl_m/lp*100:.1f}%)'
                d['ATR止盈'] = f'{lp + atr*tp_m:.6g} (+{atr*tp_m/lp*100:.1f}%)'
            elif direction == 'SELL':
                d['ATR止损'] = f'{lp + atr*sl_m:.6g} (+{atr*sl_m/lp*100:.1f}%)'
                d['ATR止盈'] = f'{lp - atr*tp_m:.6g} (-{atr*tp_m/lp*100:.1f}%)'
        except Exception:
            pass
        return result

    def _track_persistence(self, results: List[Dict]) -> None:
        """追踪信号跨扫描稳定性（含方向 + 过期衰减）"""
        active_keys = set()
        for r in results:
            direction = str(r.get('direction', 'WAIT')).upper()
            key = f"{r.get('symbol','')}:{r.get('category','')}:{direction}"
            active_keys.add(key)
            self._signal_persistence[key] = self._signal_persistence.get(key, 0) + 1
            count = self._signal_persistence[key]
            if count >= int(self.config.get('persistence_scans', 2)):
                bonus = float(self.config.get('persistence_bonus', 4.0))
                r['score'] = round(min(100, float(r.get('score', 0) or 0) + bonus), 2)
                d = r.setdefault('details', {})
                d['稳定性'] = f'🔄 连续出现{count}次 +{bonus}分'
        # 清理已消失的信号计数器（指数衰减）
        for k in list(self._signal_persistence.keys()):
            if k not in active_keys:
                self._signal_persistence[k] = max(0, self._signal_persistence[k] - 1)
                if self._signal_persistence[k] <= 0:
                    del self._signal_persistence[k]

    # ── 结果标注 + 生命周期守门（带异常保护） ──
    def _annotate_result(self, result: Dict, category_name: str, group_sort_score: int) -> Dict:
        normalized = dict(result or {})
        normalized['category']       = category_name
        normalized['group_sort_score'] = group_sort_score
        normalized['priority_reason']  = normalized.get('priority_reason') or category_name
        details = normalized.get('details')
        if isinstance(details, dict):
            details.setdefault('机会类型', category_name)
            self._normalize_detail_aliases(details, category_name)
        if apply_strategy_lifecycle_guard and bool(self.config.get('enable_strategy_lifecycle_guard', True)):
            try:
                normalized = apply_strategy_lifecycle_guard(normalized, category_name, self.config)
            except Exception:
                pass  # 守门异常不应丢弃结果（修复 v1 缺失 try/except）
        return normalized

    def _normalize_detail_aliases(self, details: Dict, category_name: str) -> None:
        alias_pairs = [
            ('H4_ADX',    '4H_ADX'),
            ('H4_ADX',    'H4_ADX_内置'),
            ('平台宽度',   '平台宽度%'),
            ('延伸幅度',   '延伸ATR'),
            ('延伸幅度',   '延伸(ATR倍)'),
            ('前段涨幅',   '前段涨/跌幅'),
        ]
        for target, source in alias_pairs:
            if target not in details and source in details:
                details[target] = details[source]
        if category_name == '突破启动':
            details.setdefault('突破幅度', details.get('延伸ATR', details.get('延伸幅度', '0')))
        elif category_name == '中继再启动':
            details.setdefault('量比', details.get('启动量比', '0'))

    # ── 动态加载子策略 ──
    def _build_child_strategies(self) -> List[Tuple[str, object, int]]:
        strategies = []
        strategy_dir = Path(__file__).resolve().parent
        child_cfg = {k: self.config[k] for k in _CHILD_COMMON_KEYS if k in self.config}

        for category_name, filename, class_name, group_sort_score in self.CATEGORY_ORDER:
            module_path = strategy_dir / filename
            if not module_path.exists():
                msg = f'跳过 {category_name}: 未找到策略文件 {filename}'
                logger.warning(f'[组合扫描器] {msg}')
                self._child_load_warnings.append(msg)
                continue
            spec = importlib.util.spec_from_file_location(f'combo_{class_name}', str(module_path))
            if spec is None or spec.loader is None:
                msg = f'跳过 {category_name}: importlib 无法解析 {filename}'
                logger.warning(f'[组合扫描器] {msg}')
                self._child_load_warnings.append(msg)
                continue
            module = importlib.util.module_from_spec(spec)
            try:
                sys.modules[spec.name] = module
                spec.loader.exec_module(module)
            except Exception as exc:
                msg = f'跳过 {category_name}: 加载 {filename} 失败: {exc}'
                logger.error(f'[组合扫描器] {msg}')
                self._child_load_warnings.append(msg)
                continue
            strategy_class = getattr(module, class_name, None)
            if strategy_class is None:
                msg = f'跳过 {category_name}: {filename} 缺少类 {class_name}'
                logger.warning(f'[组合扫描器] {msg}')
                self._child_load_warnings.append(msg)
                continue
            try:
                strategies.append((category_name, strategy_class(child_cfg), group_sort_score))
                logger.info(f'[组合扫描器] ✅ 加载 {category_name} ({class_name})')
            except Exception as exc:
                msg = f'跳过 {category_name}: 初始化失败: {exc}'
                logger.error(f'[组合扫描器] {msg}')
                self._child_load_warnings.append(msg)

        if not strategies:
            logger.warning('[组合扫描器] ⚠️  警告：0 个子策略加载成功，扫描将返回空结果！')
        return strategies

    def get_config_schema(self) -> Dict:
        return dict(CONFIG_SCHEMA)


# ══════════════════════════════════════════════
# Gate A：3m 突破回踩确认
# ══════════════════════════════════════════════
def _check_m3_retest_hold(m3: pd.DataFrame, direction: str, cfg: dict) -> Tuple[bool, str]:
    """
    3m"突破-回踩-不破位"结构检测。

    核心逻辑（以多头 BUY 为例）：
      "原突破点" = 最近的 3m 摆动低点（price 从此支撑位向上突破）
      ① 在支撑位之后，price 出现了一次向上的爆发（peak > support）
      ② 之后出现明显的回调（peak → pullback_low，幅度 >= min_pb_pct%）
      ③ pullback_low >= support * (1 - tol)：回调未跌破原突破支撑位
      ④ 当前 close 已重新站回 support 上方（回踩结束，准备继续）

    空头 SELL 为镜像：摆动高点是"原跌破位"，反弹高点不能涨回该位。

    Returns: (passed: bool, reason_str: str)
    """
    window        = int(cfg.get('gate_a_m3_window', 80))
    sw_left       = int(cfg.get('gate_a_swing_left', 5))
    sw_right      = int(cfg.get('gate_a_swing_right', 3))
    min_pb_pct    = float(cfg.get('gate_a_min_pb_pct', 0.25))
    max_break_pct = float(cfg.get('gate_a_max_break_pct', 0.20))
    recover_req   = bool(cfg.get('gate_a_recover_required', True))

    # 波动率自适应：高波动时提高回调门槛
    if len(m3) >= 15:
        try:
            atr_pct = _quick_atr_pct(m3['h'], m3['l'], m3['c'], 14)
            min_pb_pct = max(min_pb_pct, atr_pct * 0.15)  # 高 ATR 时自动提门槛
        except Exception:
            pass

    min_bars = window + sw_left + sw_right + 5
    if len(m3) < min_bars:
        return False, f'3m数据不足({len(m3)}根，需{min_bars}根)'

    recent = m3.tail(window + sw_left + sw_right).reset_index(drop=True)
    n = len(recent)

    # 搜索范围：排除最右侧 sw_right 根（留给右侧确认），从右向左找最近的摆动点
    search_end = n - sw_right
    origin_idx: Optional[int] = None
    origin_level: float = 0.0

    if direction == 'BUY':
        # 多头："原突破位" = 最近的摆动低点（支撑位，price 从此向上突破）
        for i in range(search_end - 1, sw_left - 1, -1):
            if _is_swing_low(recent['l'].values, i, sw_left, sw_right):
                origin_idx = i
                origin_level = float(recent['l'].iloc[i])
                break
    else:
        # 空头："原突破位" = 最近的摆动高点（阻力位，price 从此向下突破）
        for i in range(search_end - 1, sw_left - 1, -1):
            if _is_swing_high(recent['h'].values, i, sw_left, sw_right):
                origin_idx = i
                origin_level = float(recent['h'].iloc[i])
                break

    if origin_idx is None or origin_level <= 0:
        return False, '3m未找到有效摆动突破位（样本不足或结构平坦）'

    # 突破位之后的数据段
    post = recent.iloc[origin_idx + 1:]
    if len(post) < 4:
        return False, f'突破位后数据过少({len(post)}根)，无法确认突破+回调结构'

    current_close = float(recent['c'].iloc[-1])
    tol = max_break_pct / 100.0

    if direction == 'BUY':
        # post 中最高点（突破后的峰值）—— 必须在 post 前半段，确保"突破已完成"
        post_half = max(len(post) // 2, 2)
        peak_post   = float(post['h'].iloc[:post_half].max())
        # post 后半段中最低点（回调低点）
        pullback_seg = post.iloc[post_half:]
        pullback_low = float(pullback_seg['l'].min()) if len(pullback_seg) > 0 else float(post['l'].min())

        # 价格必须在原突破位上方形成过峰值（确认向上突破发生）
        if peak_post <= origin_level:
            return False, (
                f'3m多头突破未发生: 突破位后前半段最高价{peak_post:.6g} <= 突破位{origin_level:.6g}'
            )

        # 从峰值到回调低点的幅度
        pb_pct = (peak_post - pullback_low) / peak_post * 100 if peak_post > 0 else 0.0
        if pb_pct < min_pb_pct:
            return False, (
                f'3m多头回调幅度不足({pb_pct:.2f}% < {min_pb_pct}%)'
            )

        # 回调低点不得跌破原突破支撑位
        floor = origin_level * (1.0 - tol)
        if pullback_low < floor:
            return False, (
                f'3m回调跌破突破位: 低点{pullback_low:.6g} < 突破支撑{origin_level:.6g}'
            )

        if recover_req and current_close < floor:
            return False, (
                f'当前收盘({current_close:.6g})仍在突破位以下，回踩未结束'
            )

        return True, (
            f'3m多头突破回踩确认: 突破支撑{origin_level:.6g}→峰值{peak_post:.6g}→'
            f'回调{pb_pct:.2f}%至{pullback_low:.6g}→当前{current_close:.6g}'
        )

    else:  # SELL
        # post 前半段中最低点（跌破后的谷值）
        post_half = max(len(post) // 2, 2)
        trough_post  = float(post['l'].iloc[:post_half].min())
        # post 后半段中最高点（反弹高点）
        bounce_seg = post.iloc[post_half:]
        bounce_high  = float(bounce_seg['h'].max()) if len(bounce_seg) > 0 else float(post['h'].max())

        if trough_post >= origin_level:
            return False, (
                f'3m空头跌破未发生: 跌破位后前半段最低价{trough_post:.6g} >= 跌破位{origin_level:.6g}'
            )

        pb_pct = (bounce_high - trough_post) / trough_post * 100 if trough_post > 0 else 0.0
        if pb_pct < min_pb_pct:
            return False, (
                f'3m空头反弹幅度不足({pb_pct:.2f}% < {min_pb_pct}%)'
            )

        ceiling = origin_level * (1.0 + tol)
        if bounce_high > ceiling:
            return False, (
                f'3m反弹涨破跌破位: 高点{bounce_high:.6g} > 跌破位{origin_level:.6g}'
            )

        if recover_req and current_close > ceiling:
            return False, (
                f'当前收盘({current_close:.6g})仍在跌破位以上，反弹未结束'
            )

        return True, (
            f'3m空头跌破反弹确认: 跌破位{origin_level:.6g}→谷值{trough_post:.6g}→'
            f'反弹{pb_pct:.2f}%至{bounce_high:.6g}→当前{current_close:.6g}'
        )


def _is_swing_high(highs: np.ndarray, idx: int, left: int, right: int) -> bool:
    """idx 是否为摆动高点（左 left 根均≤，右 right 根均≤，至少一侧严格<）。"""
    if idx < left or idx + right >= len(highs):
        return False
    peak = highs[idx]
    left_ok  = all(highs[idx - j] <= peak for j in range(1, left + 1))
    right_ok = all(highs[idx + j] <= peak for j in range(1, right + 1))
    if not (left_ok and right_ok):
        return False
    left_strict  = any(highs[idx - j] < peak for j in range(1, left + 1))
    right_strict = any(highs[idx + j] < peak for j in range(1, right + 1))
    return left_strict or right_strict


def _is_swing_low(lows: np.ndarray, idx: int, left: int, right: int) -> bool:
    """idx 是否为摆动低点（左 left 根均≥，右 right 根均≥，至少一侧严格>）。"""
    if idx < left or idx + right >= len(lows):
        return False
    trough = lows[idx]
    left_ok  = all(lows[idx - j] >= trough for j in range(1, left + 1))
    right_ok = all(lows[idx + j] >= trough for j in range(1, right + 1))
    if not (left_ok and right_ok):
        return False
    left_strict  = any(lows[idx - j] > trough for j in range(1, left + 1))
    right_strict = any(lows[idx + j] > trough for j in range(1, right + 1))
    return left_strict or right_strict


# ══════════════════════════════════════════════
# Gate B：H1 趋势延续无回调迹象
# ══════════════════════════════════════════════
def _check_h1_trend_continuation(h1: pd.DataFrame, direction: str, cfg: dict) -> Tuple[bool, str]:
    """
    检查 H1 趋势是否处于延续态（无回调迹象）：
      ① EMA8/21/55 完整顺排
      ② 近 N 根无超过允许数量的反向收盘
      ③ RSI 站稳趋势侧
      ④ MACD 柱状线方向一致

    Returns: (passed, reason_str)
    """
    lookback        = int(cfg.get('gate_b_lookback', 6))
    max_rev_bars    = int(cfg.get('gate_b_max_reverse_bars', 1))
    min_rsi_long    = float(cfg.get('gate_b_min_rsi_long', 50.0))
    max_rsi_short   = float(cfg.get('gate_b_max_rsi_short', 50.0))
    require_macd    = bool(cfg.get('gate_b_require_macd', True))
    ema_fast_p      = int(cfg.get('gate_b_ema_fast', 8))
    ema_mid_p       = int(cfg.get('gate_b_ema_mid', 21))
    ema_slow_p      = int(cfg.get('gate_b_ema_slow', 55))

    if len(h1) < 60:
        return False, f'H1数据不足({len(h1)}根，需60根)'

    c = h1['c']
    ema_fast_s  = c.ewm(span=ema_fast_p, adjust=False).mean()
    ema_mid_s   = c.ewm(span=ema_mid_p,  adjust=False).mean()
    ema_slow_s  = c.ewm(span=ema_slow_p, adjust=False).mean()

    ema_fast  = float(ema_fast_s.iloc[-1])
    ema_mid   = float(ema_mid_s.iloc[-1])
    ema_slow  = float(ema_slow_s.iloc[-1])
    rsi   = _rsi_wilder(c)

    # MACD 柱状线
    e12    = c.ewm(span=12, adjust=False).mean()
    e26    = c.ewm(span=26, adjust=False).mean()
    m_line = e12 - e26
    s_line = m_line.ewm(span=9, adjust=False).mean()
    hist   = m_line - s_line
    hist_last = float(hist.iloc[-1])
    hist_prev = float(hist.iloc[-2]) if len(hist) >= 2 else hist_last

    # 近 lookback 根的 EMA mid 序列
    ema_mid_recent = ema_mid_s.iloc[-lookback:].values
    closes_recent = c.iloc[-lookback:].values

    if direction == 'BUY':
        # ① EMA 排列检查：EMA8>EMA21（短期确认），EMA21>EMA55（中期确认，可选）
        ema_short_ok = ema_fast > ema_mid
        ema_mid_ok   = ema_mid > ema_slow
        if not ema_short_ok:
            return False, (
                f'H1 EMA短期排列未达标: EMA{ema_fast_p}={ema_fast:.4g} <= EMA{ema_mid_p}={ema_mid:.4g}'
            )
        ema_level = '完整顺排' if ema_mid_ok else '短期确认(EMA21>EMA55待金叉)'
        # ② 近 N 根反向收盘数量
        reverse_count = int(sum(1 for cl, e21 in zip(closes_recent, ema_mid_recent) if cl < e21))
        if reverse_count > max_rev_bars:
            return False, (
                f'H1近{lookback}根有{reverse_count}根收盘低于EMA{ema_mid_p}'
                f'（允许≤{max_rev_bars}根），存在回调迹象'
            )
        # ③ RSI 多头区间
        if rsi < min_rsi_long:
            return False, f'H1 RSI偏弱({rsi:.1f} < {min_rsi_long:.0f})，动能不足'
        # ④ MACD 方向 + 零线穿越加成
        if require_macd and hist_last < 0:
            return False, f'H1 MACD柱状线为负({hist_last:.4g})，多头趋势动能减弱'
        macd_bonus = ' 🔼零线上穿' if (hist_prev <= 0 < hist_last) else ''

        return True, (
            f'H1多头趋势延续[{ema_level}]: EMA{ema_fast_p}({ema_fast:.4g})>EMA{ema_mid_p}({ema_mid:.4g})'
            + (f'>EMA{ema_slow_p}({ema_slow:.4g})' if ema_mid_ok else f'(EMA{ema_slow_p}={ema_slow:.4g},待中周期确认)')
            + f', 近{lookback}根反向收盘{reverse_count}根, RSI={rsi:.1f}, MACD柱={hist_last:.4g}{macd_bonus}'
        )

    else:  # SELL
        # ① EMA 排列检查
        ema_short_ok = ema_fast < ema_mid
        ema_mid_ok   = ema_mid < ema_slow
        if not ema_short_ok:
            return False, (
                f'H1 EMA短期空头未达标: EMA{ema_fast_p}={ema_fast:.4g} >= EMA{ema_mid_p}={ema_mid:.4g}'
            )
        ema_level = '完整顺排' if ema_mid_ok else '短期确认(EMA21<EMA55待死叉)'
        # ② 近 N 根反向收盘数量
        reverse_count = int(sum(1 for cl, e21 in zip(closes_recent, ema_mid_recent) if cl > e21))
        if reverse_count > max_rev_bars:
            return False, (
                f'H1近{lookback}根有{reverse_count}根收盘高于EMA{ema_mid_p}'
                f'（允许≤{max_rev_bars}根），存在反弹迹象'
            )
        # ③ RSI 空头区间
        if rsi > max_rsi_short:
            return False, f'H1 RSI偏强({rsi:.1f} > {max_rsi_short:.0f})，空头动能不足'
        # ④ MACD 方向
        if require_macd and hist_last > 0:
            return False, f'H1 MACD柱状线为正({hist_last:.4g})，空头趋势动能减弱'
        macd_bonus = ' 🔽零线下穿' if (hist_prev >= 0 > hist_last) else ''

        return True, (
            f'H1空头趋势延续[{ema_level}]: EMA{ema_fast_p}({ema_fast:.4g})<EMA{ema_mid_p}({ema_mid:.4g})'
            + (f'<EMA{ema_slow_p}({ema_slow:.4g})' if ema_mid_ok else f'(EMA{ema_slow_p}={ema_slow:.4g},待中周期确认)')
            + f', 近{lookback}根反向收盘{reverse_count}根, RSI={rsi:.1f}, MACD柱={hist_last:.4g}{macd_bonus}'
        )


# ══════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════
def _get_symbol_klines(symbol) -> Dict:
    """统一从 symbol.extra_data 中提取 klines 映射。"""
    try:
        return symbol.extra_data.get('klines', {})
    except Exception:
        return {}


def _fail_result(inst_id: str, reason: str) -> Dict:
    return {
        'symbol':   inst_id,
        'passed':   False,
        'score':    0.0,
        'direction': 'WAIT',
        'category': '未分类',
        'group_sort_score': 0,
        'details':  {'状态': reason},
    }


# ══════════════════════════════════════════════
# v3.3 辅助函数
# ══════════════════════════════════════════════

def _calc_gate_quality(gate_a_str: str, gate_b_str: str) -> float:
    """Gate A+B 质量乘积分：双通过 > 单通过 > 未检查。"""
    a_pass = '✅' in str(gate_a_str)
    b_pass = '✅' in str(gate_b_str)
    if a_pass and b_pass:
        return 1.0
    elif a_pass:
        return 0.65
    elif b_pass:
        return 0.55
    return 1.0


def _quick_atr_pct(h: pd.Series, l: pd.Series, c: pd.Series, period: int = 14) -> float:
    """快速计算 ATR%（用于 Gate A 自适应阈值）"""
    try:
        if len(h) < period + 2:
            return 2.0
        pc = c.shift(1)
        tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
        atr = float(tr.tail(period).mean())
        last = float(c.iloc[-1]) if len(c) > 0 else 1.0
        return atr / max(last, 1e-9) * 100 if atr > 0 and last > 0 else 2.0
    except Exception:
        return 2.0


# ══════════════════════════════════════════════
# 模块导出（v1 漏写 STRATEGY_CLASS 导致引擎无法加载，v2 修复）
# ══════════════════════════════════════════════
STRATEGY_NAME  = '波段八策略组合扫描器'
STRATEGY_TYPE  = 'scan'
STRATEGY_CLASS = SwingEightStrategyComboScanner
BACKTEST_CLASS = SwingEightStrategyComboScanner
