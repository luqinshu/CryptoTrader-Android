from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd

from strategies._shared.validation import validate_config


def normalize_direction(value: Any) -> str:
    direction = str(value or "").strip().upper()
    if direction in {"BUY", "LONG", "BUY_LONG"}:
        return "LONG"
    if direction in {"SELL", "SHORT", "SELL_SHORT"}:
        return "SHORT"
    return ""


def rows_to_df(rows) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    base_cols = ["ts", "open", "high", "low", "close", "volume"]
    extra = [f"_x{i}" for i in range(20)]
    width = len(rows[0]) if rows else 0
    cols = base_cols[:width] if width <= len(base_cols) else base_cols + extra[: width - len(base_cols)]
    df = pd.DataFrame(rows, columns=cols)
    for col in ("open", "high", "low", "close", "volume"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["ts"] = pd.to_numeric(df["ts"], errors="coerce")
    df = df.dropna(subset=["ts", "open", "high", "low", "close"]).copy()
    df["timestamp"] = pd.to_datetime(df["ts"].astype("int64"), unit="ms")
    return df.sort_values("timestamp").drop_duplicates(subset="timestamp", keep="last").reset_index(drop=True)


def evaluate_entry_rule_from_klines(
    klines_map: Dict[str, Any],
    direction: Any,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    h1_rows = (
        klines_map.get("1H")
        or klines_map.get("hourly")
        or klines_map.get("h1")
        or []
    )
    m3_rows = klines_map.get("3m") or klines_map.get("m3") or []
    h1_df = rows_to_df(h1_rows)
    m3_df = rows_to_df(m3_rows)
    return evaluate_entry_rule_from_frames(m3_df, h1_df, direction, config=config)


def evaluate_entry_rule_from_frames(
    m3_df: pd.DataFrame,
    h1_df: pd.DataFrame,
    direction: Any,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    cfg = config or {}
    errors = validate_config(cfg, {
        'min_score': {'min': 0, 'max': 100, 'type': float, 'required': False},
        'position_size': {'min': 0.001, 'max': 1.0, 'type': float, 'required': False},
        'max_leverage': {'min': 1, 'max': 125, 'type': int, 'required': False},
    })
    if errors:
        return {"ok": False, "reason": f"配置参数校验失败: {'; '.join(errors)}"}
    norm_dir = normalize_direction(direction)
    if norm_dir not in {"LONG", "SHORT"}:
        return {"ok": False, "reason": "方向缺失，无法执行3m/H1硬性检查"}
    if m3_df is None or m3_df.empty or len(m3_df) < 24:
        return {"ok": False, "reason": "3m数据不足，无法确认回调企稳"}
    if h1_df is None or h1_df.empty or len(h1_df) < 30:
        return {"ok": False, "reason": "1H数据不足，无法确认趋势延续"}
    if not _hourly_trend_continues(h1_df, norm_dir, cfg):
        return {"ok": False, "reason": "1H趋势未继续原趋势"}
    pullback = _detect_breakout_pullback_hold(m3_df, norm_dir, cfg)
    if not pullback["ok"]:
        return pullback
    return {
        "ok": True,
        "reason": (
            f"3m明显回调{pullback['pullback_pct']:.2f}%且未跌破原突破点"
            f" | 1H趋势继续"
        ),
        "pullback_pct": pullback["pullback_pct"],
        "breakout_point": pullback["breakout_point"],
    }


def _cfg_float(config: Dict[str, Any], key: str, default: float) -> float:
    value = config.get(key, default)
    if value in (None, ""):
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _cfg_int(config: Dict[str, Any], key: str, default: int) -> int:
    value = config.get(key, default)
    if value in (None, ""):
        return int(default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _ema(close: pd.Series, span: int) -> pd.Series:
    return close.ewm(span=max(int(span), 1), adjust=False).mean()


def _find_local_swing(series: pd.Series, direction: str, order: int = 5) -> int:
    values = series.values
    if len(values) < order * 2 + 1:
        return int(series.idxmax()) if direction == "LONG" else int(series.idxmin())
    if direction == "LONG":
        for i in range(len(values) - order - 1, order - 1, -1):
            window = values[max(0, i - order): i + order + 1]
            if values[i] == window.max():
                return i
        return int(series.idxmax())
    for i in range(len(values) - order - 1, order - 1, -1):
        window = values[max(0, i - order): i + order + 1]
        if values[i] == window.min():
            return i
    return int(series.idxmin())


def _hourly_trend_continues(h1_df: pd.DataFrame, direction: str, config: Dict[str, Any]) -> bool:
    close = h1_df["close"]
    fast = _ema(close, _cfg_int(config, "h1_fast_ema", 12))
    slow = _ema(close, _cfg_int(config, "h1_slow_ema", 26))
    if len(fast) >= 10:
        base = float(fast.iloc[-10]) or 1.0
        slope_pct = (float(fast.iloc[-1]) - base) / base
    else:
        slope_pct = 0.0
    slope_threshold = 0.0002
    last_close = float(close.iloc[-1])
    fast_last = float(fast.iloc[-1])
    slow_last = float(slow.iloc[-1])
    if direction == "LONG":
        return fast_last > slow_last and slope_pct > slope_threshold and last_close >= slow_last * 0.999
    return fast_last < slow_last and slope_pct < -slope_threshold and last_close <= slow_last * 1.001


def _detect_breakout_pullback_hold(
    m3_df: pd.DataFrame,
    direction: str,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    df = m3_df.tail(48).reset_index(drop=True)
    close = df["close"]
    high = df["high"]
    low = df["low"]
    fast = _ema(close, _cfg_int(config, "m3_fast_ema", 8))
    mid = _ema(close, _cfg_int(config, "m3_mid_ema", 13))
    stab_bars = max(_cfg_int(config, "m3_stabilization_bars", 3), 2)
    min_pb = _cfg_float(config, "m3_pullback_min_pct", 0.35)
    max_pb = _cfg_float(config, "m3_pullback_max_pct", 2.50)
    breakout_buffer = _cfg_float(config, "m3_breakout_buffer_pct", 0.15) / 100.0
    hold_buffer = _cfg_float(config, "m3_breakout_hold_buffer_pct", 0.12) / 100.0

    search_slice = close.iloc[:-stab_bars]
    if len(search_slice) < 12:
        return {"ok": False, "reason": "3m历史不足，无法识别原突破点"}

    if direction == "LONG":
        anchor_idx = _find_local_swing(search_slice, "LONG", order=5)
        pre_anchor = high.iloc[max(0, anchor_idx - 18):anchor_idx]
        if pre_anchor.empty:
            return {"ok": False, "reason": "未找到有效原突破点"}
        breakout_point = float(pre_anchor.max())
        anchor = float(high.iloc[anchor_idx])
        if anchor < breakout_point * (1.0 + breakout_buffer):
            return {"ok": False, "reason": "3m未形成有效突破，不属于突破后回调"}
        pb_slice = df.iloc[anchor_idx + 1:]
        if pb_slice.empty:
            return {"ok": False, "reason": "3m尚未出现明显回调"}
        trough = float(pb_slice["low"].min())
        pullback_pct = (anchor - trough) / anchor * 100 if anchor > 0 else 0.0
        stabilized = (
            bool((close.tail(stab_bars) >= fast.tail(stab_bars)).all())
            and bool((close.tail(stab_bars) >= mid.tail(stab_bars)).all())
            and float(close.iloc[-1]) >= float(close.iloc[-stab_bars])
        )
        hold_ok = trough >= breakout_point * (1.0 - hold_buffer)
        if pullback_pct < min_pb:
            return {"ok": False, "reason": f"3m回调不够明显（{pullback_pct:.2f}% < {min_pb:.2f}%）"}
        if pullback_pct > max_pb:
            return {"ok": False, "reason": f"3m回调过深（{pullback_pct:.2f}% > {max_pb:.2f}%）"}
        if not hold_ok:
            return {"ok": False, "reason": "3m回调跌破原突破点"}
        if not stabilized:
            return {"ok": False, "reason": "3m虽有回调但尚未企稳"}
    else:
        anchor_idx = _find_local_swing(search_slice, "SHORT", order=5)
        pre_anchor = low.iloc[max(0, anchor_idx - 18):anchor_idx]
        if pre_anchor.empty:
            return {"ok": False, "reason": "未找到有效原突破点"}
        breakout_point = float(pre_anchor.min())
        anchor = float(low.iloc[anchor_idx])
        if anchor > breakout_point * (1.0 - breakout_buffer):
            return {"ok": False, "reason": "3m未形成有效破位，不属于破位后回抽"}
        pb_slice = df.iloc[anchor_idx + 1:]
        if pb_slice.empty:
            return {"ok": False, "reason": "3m尚未出现明显反抽"}
        rebound = float(pb_slice["high"].max())
        pullback_pct = (rebound - anchor) / anchor * 100 if anchor > 0 else 0.0
        stabilized = (
            bool((close.tail(stab_bars) <= fast.tail(stab_bars)).all())
            and bool((close.tail(stab_bars) <= mid.tail(stab_bars)).all())
            and float(close.iloc[-1]) <= float(close.iloc[-stab_bars])
        )
        hold_ok = rebound <= breakout_point * (1.0 + hold_buffer)
        if pullback_pct < min_pb:
            return {"ok": False, "reason": f"3m反抽不够明显（{pullback_pct:.2f}% < {min_pb:.2f}%）"}
        if pullback_pct > max_pb:
            return {"ok": False, "reason": f"3m反抽过深（{pullback_pct:.2f}% > {max_pb:.2f}%）"}
        if not hold_ok:
            return {"ok": False, "reason": "3m反抽上破原突破点"}
        if not stabilized:
            return {"ok": False, "reason": "3m虽有反抽但尚未重新转弱"}

    return {
        "ok": True,
        "reason": "通过3m回调与H1延续硬性检查",
        "pullback_pct": pullback_pct,
        "breakout_point": breakout_point,
    }
