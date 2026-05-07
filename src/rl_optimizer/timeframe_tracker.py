"""
多时间框架趋势追踪器: 记录扫描信号时各周期的趋势判断，
后续与实际走势对比，按周期统计准确率，驱动策略参数调整。
"""

from __future__ import annotations

import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class TrendClassification:
    """趋势分类结果"""
    BULLISH = "bullish"   # 看涨
    BEARISH = "bearish"   # 看跌
    RANGING = "ranging"   # 震荡
    UNKNOWN = "unknown"   # 数据不足


class MultiTimeframeTracker:
    """
    多时间框架追踪器：
    - 记录扫描时每个交易对在 1D/1H/3m 的趋势判断
    - 后续验证每个时间框架的预测是否正确
    - 按策略/周期统计准确率 → 反馈到参数优化器
    """

    VALIDATION_WINDOWS = {
        "3m": (1.0, 0.6),    # 1小时后验证，需要1.2%的变动判为正确
        "1H": (6.0, 1.5),    # 6小时后验证，1.5%变动
        "1D": (24.0, 3.0),   # 24小时后验证，3.0%变动
        "4H": (12.0, 2.0),   # 12小时后验证，2.0%变动
    }

    def __init__(self, data_dir: Optional[str] = None):
        self._data_dir = Path(data_dir) if data_dir else Path(__file__).resolve().parent.parent.parent / "data" / "rl_timeframes"
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._path = self._data_dir / "timeframe_predictions.json"
        self._predictions: List[Dict] = []
        self._load()

    def _load(self):
        if self._path.exists():
            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    self._predictions = json.load(f)
            except Exception:
                self._predictions = []

    def _save(self):
        try:
            with open(self._path, "w", encoding="utf-8") as f:
                json.dump(self._predictions[-8000:], f, indent=2, ensure_ascii=False, default=str)
        except Exception:
            pass

    @staticmethod
    def classify_trend(klines: List[List]) -> Tuple[str, float, Dict]:
        """
        分析K线趋势并返回判断结果。

        Returns:
            (trend_class, confidence, details_dict)
            trend_class: "bullish"/"bearish"/"ranging"/"unknown"
            confidence: -3 ~ +3 (正值=看涨强度，负值=看跌强度)
            details: included for debugging
        """
        if not klines or len(klines) < 10:
            return TrendClassification.UNKNOWN, 0.0, {"error": "K线不足"}

        closes = []
        highs = []
        lows = []
        for row in reversed(klines):
            if len(row) < 6:
                continue
            try:
                closes.append(float(row[4]))
                highs.append(float(row[2]))
                lows.append(float(row[3]))
            except (ValueError, TypeError):
                continue

        closes = closes[:100]
        highs = highs[:100]
        lows = lows[:100]

        if len(closes) < 10:
            return TrendClassification.UNKNOWN, 0.0, {"error": "有效K线不足"}

        price = closes[0]

        # EMA 排列（正向计算：oldest→newest）
        def _ema(series, span):
            alpha = 2.0 / (span + 1)
            result = []
            ema = series[0]  # 种子用最老的值
            for v in series:  # 正向遍历（oldest→newest）
                ema = alpha * v + (1 - alpha) * ema
                result.append(ema)
            return result

        ema8 = _ema(closes, 8)
        ema21 = _ema(closes, 21)
        ema55 = _ema(closes, 55)

        # 趋势排列得分
        if ema8[0] > ema21[0] > ema55[0] and closes[0] > ema8[0]:
            alignment = 1.0
        elif ema8[0] < ema21[0] < ema55[0] and closes[0] < ema8[0]:
            alignment = -1.0
        else:
            alignment = 0.0

        # 动量得分
        momentum_long = (price / max(closes[min(24, len(closes)-1)], 1e-9) - 1) * 100
        momentum_medium = (price / max(closes[min(12, len(closes)-1)], 1e-9) - 1) * 100
        momentum_score = np.clip(momentum_long * 0.03 + momentum_medium * 0.06, -2.0, 2.0)

        # 波动率判断震荡
        if len(closes) >= 20:
            returns = [closes[i] / max(closes[i+1], 1e-9) - 1 for i in range(min(20, len(closes)-1))]
            volatility = np.std(returns) * 100 if returns else 0
            if len(returns) >= 10:
                window_high = max(highs[:14]) if highs[:14] else 0.0
                window_low = min(lows[:14]) if lows[:14] else 0.0
                atr_ratio = (window_high / max(window_low, 1e-9) - 1) * 100
            else:
                atr_ratio = volatility * 2
        else:
            volatility = 0
            atr_ratio = 0

        # 综合判断
        total_score = alignment * 1.2 + momentum_score * 0.8

        # 震荡判定: 排列不明确 + 波动率偏低
        if abs(alignment) < 0.3 and atr_ratio < 2.0:
            return TrendClassification.RANGING, 0.0, {
                "alignment": round(alignment, 2), "momentum": round(momentum_score, 2),
                "volatility": round(volatility, 2), "atr_ratio": round(atr_ratio, 2),
            }

        if total_score > 0.4:
            return TrendClassification.BULLISH, min(total_score, 3.0), {
                "alignment": round(alignment, 2), "momentum": round(momentum_score, 2),
                "volatility": round(volatility, 2), "atr_ratio": round(atr_ratio, 2),
            }
        elif total_score < -0.4:
            return TrendClassification.BEARISH, max(total_score, -3.0), {
                "alignment": round(alignment, 2), "momentum": round(momentum_score, 2),
                "volatility": round(volatility, 2), "atr_ratio": round(atr_ratio, 2),
            }
        else:
            return TrendClassification.RANGING, 0.0, {
                "alignment": round(alignment, 2), "momentum": round(momentum_score, 2),
                "volatility": round(volatility, 2), "atr_ratio": round(atr_ratio, 2),
            }

    def record_signal_with_trends(self, signal: Dict[str, Any], klines_map: Dict[str, List]):
        """
        记录扫描信号,同时分析所有时间框架的趋势。

        signal: 扫描结果 dict (包含 symbol, direction, score, strategy_name 等)
        klines_map: {"1D": [...], "1H": [...], "3m": [...], "4H": [...]}
        """
        symbol = signal.get("symbol", "")
        strategy = signal.get("strategy_name") or signal.get("category", "unknown")
        now = time.time()

        predictions = {}
        for tf in ("1D", "4H", "1H", "3m"):
            candles = klines_map.get(tf, [])
            if not candles:
                predictions[tf] = {"trend": "unknown", "confidence": 0.0, "details": {"error": "无K线数据"}}
                continue
            trend, confidence, details = self.classify_trend(candles)
            predictions[tf] = {
                "trend": trend,
                "confidence": round(confidence, 2),
                "details": details,
                "price_at_signal": float(candles[0][4]) if candles and len(candles[0]) > 4 else 0,
            }

        # 去重
        for existing in reversed(self._predictions):
            if (existing.get("symbol") == symbol
                    and existing.get("strategy") == strategy
                    and now - existing.get("timestamp", 0) < 300):
                return

        entry = {
            "symbol": symbol,
            "strategy": strategy,
            "scan_direction": signal.get("direction", "WAIT"),
            "scan_score": float(signal.get("score", 0) or 0),
            "timestamp": now,
            "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "predictions": predictions,
            "validated": {tf: False for tf in predictions},
            "outcomes": {},
        }
        self._predictions.append(entry)
        self._save()

    def validate_predictions(self, okx_client=None) -> int:
        """验证所有未验证的时间框架预测"""
        now = time.time()
        updated = 0

        for pred in self._predictions:
            elapsed_h = (now - pred.get("timestamp", 0)) / 3600.0

            for tf, (wait_h, threshold_pct) in self.VALIDATION_WINDOWS.items():
                if tf not in pred.get("predictions", {}):
                    continue
                if pred.get("validated", {}).get(tf, False):
                    continue
                if elapsed_h < wait_h:
                    continue
                if not okx_client:
                    continue

                symbol = pred["symbol"]
                predicted_trend = pred["predictions"][tf]["trend"]
                start_price = pred["predictions"][tf].get("price_at_signal", 0)

                try:
                    ticker = okx_client.get_ticker(symbol)
                    if ticker.get("code") != "0" or not ticker.get("data"):
                        continue
                    current_price = float(ticker["data"][0]["last"])
                except Exception:
                    continue

                if start_price <= 0:
                    pred["validated"][tf] = True
                    pred["outcomes"][tf] = {"correct": False, "reason": "无效起始价格"}
                    updated += 1
                    continue

                pnl_pct = (current_price - start_price) / start_price * 100

                correct = False
                if predicted_trend == TrendClassification.BULLISH:
                    correct = pnl_pct > threshold_pct
                elif predicted_trend == TrendClassification.BEARISH:
                    correct = pnl_pct < -threshold_pct
                elif predicted_trend == TrendClassification.RANGING:
                    correct = abs(pnl_pct) <= threshold_pct * 1.5

                pred["validated"][tf] = True
                pred["outcomes"][tf] = {
                    "correct": correct,
                    "pnl_pct": round(pnl_pct, 2),
                    "predicted_trend": predicted_trend,
                    "current_price": current_price,
                    "validate_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
                updated += 1

        if updated > 0:
            self._save()
        return updated

    def accuracy_by_timeframe(self, strategy_name: Optional[str] = None, days: int = 30) -> Dict[str, Dict]:
        """按时间框架统计准确率"""
        cutoff = time.time() - days * 86400
        stats: Dict[str, Dict] = {}

        for tf in ("3m", "1H", "4H", "1D"):
            correct = 0
            total = 0
            for pred in self._predictions:
                if pred.get("timestamp", 0) < cutoff:
                    continue
                if strategy_name and pred.get("strategy") != strategy_name:
                    continue
                outcome = pred.get("outcomes", {}).get(tf, {})
                if not outcome or "correct" not in outcome:
                    continue
                if outcome["correct"]:
                    correct += 1
                total += 1
            stats[tf] = {
                "accuracy": round(correct / total * 100, 1) if total > 0 else 0.0,
                "total": total,
                "correct": correct,
            }
        return stats

    def accuracy_by_strategy(self, days: int = 30) -> Dict[str, Any]:
        """按策略统计各时间框架准确率"""
        cutoff = time.time() - days * 86400
        strategies = {}
        for pred in self._predictions:
            if pred.get("timestamp", 0) < cutoff:
                continue
            name = pred.get("strategy", "unknown")
            s = strategies.setdefault(name, {})
            for tf in ("3m", "1H", "4H", "1D"):
                outcome = pred.get("outcomes", {}).get(tf, {})
                if not outcome or "correct" not in outcome:
                    continue
                ts = s.setdefault(tf, {"correct": 0, "total": 0})
                if outcome["correct"]:
                    ts["correct"] += 1
                ts["total"] += 1
        result = {}
        for name, tf_data in strategies.items():
            result[name] = {
                tf: {
                    "accuracy": round(d["correct"] / d["total"] * 100, 1) if d["total"] > 0 else 0.0,
                    "total": d["total"],
                }
                for tf, d in tf_data.items()
            }
        return result

    def adjustment_recommendations(self, strategy_name: str, days: int = 30) -> List[str]:
        """根据各周期准确率生成参数调整建议"""
        tf_stats = self.accuracy_by_timeframe(strategy_name, days)
        recs = []

        for tf, weight_key, param_key in [
            ("3m", "m3_pullback", "m3_stabilization_bars"),
            ("1H", "h1_early_trend", "max_h1_trend_age"),
            ("4H", "momentum_4h", "h1_trend_age_penalty"),
            ("1D", "momentum_1d", "min_score"),
        ]:
            acc = tf_stats.get(tf, {}).get("accuracy", 0)
            total = tf_stats.get(tf, {}).get("total", 0)
            if total < 3:
                continue
            if acc < 40:
                recs.append(f"⚠️ {tf}预测准确率仅 {acc:.1f}%，建议降低 {param_key}")
            elif acc > 65:
                recs.append(f"✅ {tf}预测准确率 {acc:.1f}% 优秀，可适当提高 {param_key}")
            else:
                recs.append(f"📊 {tf}预测准确率 {acc:.1f}% 正常，保持 {param_key} 当前值")

        return recs

    def recent_predictions(self, strategy_name: str, limit: int = 20) -> List[Dict]:
        return [p for p in reversed(self._predictions) if p.get("strategy") == strategy_name][:limit]
