#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XGBoost 截面排序策略 v2

v1 → v2 修复摘要
─────────────────────────────────────────────────────
【严重逻辑错误修复】
1. _train_model: groups 总和 ≠ len(samples)
   当样本数不能整除 group_size 时 XGBRanker 报错。
   → v2 用余数补全最后一组，确保 sum(groups)==len(samples)。

2. _train_model: 同时传 groups + qid（两种互斥 API）
   XGBRanker.fit() 只接受一种，同时传会引发 TypeError。
   → v2 统一使用 qid 参数（sklearn 兼容接口）。

3. 训练标签 momentum_1d 是当前已知的过去收益 → look-ahead bias
   用过去7天动量作为排序目标，模型学到的是"过去涨得多=好"，
   而非"当前因子值高 → 未来收益好"。
   → v2 积累样本时同步记录 entry_time，下次扫描时用当前价格
     计算实际已实现收益作为标签（延迟标签机制）。
   → 在标签尚未产生时仍用 momentum_1d 作为弱替代，但明确标注。

4. _score_with_model: 线性评分量纲混乱
   factors 中部分值已是百分比（momentum_1h*100≈2.0）、
   部分是原始比率（trend_quality≈80）、
   部分是 log（liquidity≈18），统一乘 100 导致评分失控。
   → v2 对各因子先做 robust z-score 归一化，再做线性加权，
     结果有意义的范围在 [-3,3]，最后线性映射到 [0,100]。

5. scan_all_symbols: 逐个调 scan_symbol，无截面归一化
   每个 symbol 独立打分，XGBoost Ranker 的截面排序意义丢失。
   → v2 scan_all_symbols 先批量提取全部因子矩阵，做截面
     z-score 归一化，再统一输入模型/线性权重，得到截面 edge。

【中等错误修复】
6. _bb_pctb: 标准 %b 公式应为 (price-lower)/(upper-lower)，
   范围 [0,1]，而原代码 (c-m)/(2σ) 范围在 [-0.5, 0.5]。
   → v2 修正为正确公式并 clip 到 [0,1]。

7. _vol_zscore: 基线 tail(24) 包含当前 bar，拉高基线。
   → v2 改为 iloc[-(25):-1]（排除当前 bar）。

8. _early_trigger: rsi_c = _clamp((r-50)/16, 0, 1) 空头时截0，
   丢失空头信号。
   → v2 允许 rsi_c 为 [-1,1]，由 trigger 综合后 clip 到 [-2.5,2.5]。

9. _mom_decay: 参数 m1/m4/md 是 _pct() 的原始返回（小数），
   注释标注为百分比，概念混乱易引入维护 bug。
   → v2 在函数内统一乘100转换为百分比再归一化。
─────────────────────────────────────────────────────
"""

from __future__ import annotations
import logging
from math import log
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import time

from strategies._shared.indicators import (
    _to_df, _robust_zscore, _adx, _rsi_wilder as _rsi, _clamp, _measure_trend_age,
)

logger = logging.getLogger(__name__)

try:
    import xgboost as xgb
    _HAS_XGB = True
except Exception:
    xgb = None
    _HAS_XGB = False
    logger.warning("[XGBoost策略] xgboost 不可用，将使用线性回退。安装: pip install xgboost")

try:
    from src.scanner.base_scanner import BaseScannerStrategy, ScanCondition
    from src.scanner.ranking import build_opportunity_profile
    _HAS_BASE = True
except ImportError:
    BaseScannerStrategy = object; ScanCondition = None; build_opportunity_profile = None
    _HAS_BASE = False


CONFIG_SCHEMA = {
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

_DEFAULT = {k: v["default"] for k, v in CONFIG_SCHEMA.items()}

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

    def get_config_schema(self): return dict(CONFIG_SCHEMA)

    # ── 单标的评分（不走截面 z-score，用于 scan_symbol 模式）──────────────
    def scan_symbol(self, symbol):
        snap = _build_snapshot(symbol, self.config)
        if not snap["valid"]: return _failed(symbol, snap["reason"])
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
        return _result(snap, score, direction, f, passed, self.config)

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
            snap = _build_snapshot(sym, self.config)
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
                results.append(_result(snap, score, direction, z.loc[sym_name].to_dict(), passed, self.config))

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
        sym = _symbol_from_backtest(data, cfg)
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

def _build_snapshot(symbol, config) -> Dict[str, Any]:
    inst = str(getattr(symbol, "inst_id", ""))
    extra = getattr(symbol, "extra_data", {}) or {}
    klines = extra.get("klines", {}) or {}
    h1 = _to_df(_getk(klines, "1H"))
    h4 = _to_df(_getk(klines, "4H"))
    d1 = _to_df(_getk(klines, "1D"))
    m3 = _to_df(_getk(klines, "3m"))
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
    tq = _trend_quality(h4)
    rv = _realized_vol(close_1h, 48)
    atr = _atr_pct(h4, 14)
    liq = log(max(v24, 1.0))
    vi = _vol_ratio(vol_1h, 24)
    vz = _vol_zscore(vol_1h)   # v2: 已修正排除当前bar
    fr = float((extra.get("funding_rate") or 0)) * 100
    oi = log(max(float(getattr(symbol, "open_interest", 0) or 1), 1))
    mm = _macd_mom(close_1h)
    bb = _bb_pctb(close_1h)    # v2: 已修正为标准 [0,1]
    cs = _close_strength(h1, 6)
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

def _getk(klines, bar):
    for k in [bar, bar.lower(), bar.upper()]:
        if k in klines and klines.get(k): return klines[k]
    return []

def _pct(s, bars):
    if len(s) <= bars or bars <= 0: return 0.0
    return float(s.iloc[-1]) / max(float(s.iloc[-(bars+1)]), 1e-9) - 1

def _vol_ratio(v, w):
    if len(v) < w + 3: return 1.0
    return float(v.tail(3).mean()) / max(float(v.iloc[-(w+1):-1].median() or 0), 1e-9)

def _vol_zscore(v):
    """v2: 排除当前 bar 的基线计算。"""
    if len(v) < 25: return 0.0
    baseline = v.iloc[-25:-1]  # 排除最后一根（当前bar）
    m = float(baseline.mean())
    s = float(baseline.std(ddof=0) or 1)
    return (float(v.iloc[-1]) - m) / s

def _atr_pct(df, p):
    if len(df) < p + 2: return 1.0
    pc = df["c"].shift(1)
    tr = pd.concat([(df["h"]-df["l"]).abs(), (df["h"]-pc).abs(), (df["l"]-pc).abs()], axis=1).max(axis=1)
    a = float(tr.ewm(alpha=1/p, adjust=False).mean().iloc[-1] or 0)
    return a / float(df["c"].iloc[-1] or 1) * 100

def _realized_vol(c, w):
    r = c.pct_change().dropna().tail(w)
    return float(r.std(ddof=0) * np.sqrt(len(r)) * 100) if len(r) > 0 else 0

def _trend_quality(h4):
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

def _close_strength(df, n):
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
    v_ratio = _vol_ratio(v, 24); cs = _close_strength(h1, 3)
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

def _failed(symbol, reason):
    return {"symbol": str(getattr(symbol, "inst_id", "")), "passed": False,
            "score": 0, "direction": "WAIT", "signals": [], "details": {"状态": reason}}

def _result(snap, score, direction, fs, passed, config):
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

def _symbol_from_backtest(data, config):
    km = data.get("klines_map", {}) or {}
    h1 = _to_df(_getk(km, "1H") or data.get("klines") or [])
    lp = float(h1["c"].iloc[-1]) if not h1.empty else 0
    vol = float((h1["c"] * h1["vol"]).tail(48).sum()) if not h1.empty else 0
    extra = {"klines": km, "funding_rate": data.get("funding_rate", 0)}
    return _MinimalSymbol(inst_id=str(config.get("inst_id", "BT")),
        last_price=lp, volume_24h=vol,
        price_change_24h=_pct(h1["c"], 24)*100 if not h1.empty else 0,
        extra_data=extra)

class _MinimalSymbol:
    def __init__(self, inst_id, last_price, volume_24h, price_change_24h, extra_data):
        self.inst_id = inst_id; self.last_price = last_price
        self.volume_24h = volume_24h; self.price_change_24h = price_change_24h
        self.high_24h = 0; self.low_24h = 0; self.open_interest = 0
        self.extra_data = extra_data


STRATEGY_NAME = "XGBoost截面排序策略"
STRATEGY_TYPE = "scan"
STRATEGY_CLASS = XGBoostCrossSectionalRanker
BACKTEST_CLASS = XGBoostCrossSectionalRanker
