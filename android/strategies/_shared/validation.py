"""
共享输入校验模块 —— 所有策略文件统一引用。

消除各策略中缺失的价格/K 线校验，防止 NaN/Inf 和脏数据导致异常。
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd


def validate_price(price: float, label: str = "价格") -> bool:
    """校验价格是否有效（>0 且非 NaN/Inf）。

    Returns: True 表示有效。
    """
    try:
        p = float(price)
    except (TypeError, ValueError):
        return False
    return p > 0 and np.isfinite(p)


def validate_klines(klines, min_len: int = 3, label: str = "K线") -> bool:
    """校验 K 线数据是否有效：
    - 非空
    - 长度达标
    - 时间戳单调递增
    - 无 NaN/Inf 在 OHLC 列
    - 高 >= 低

    Returns: True 表示有效。
    """
    if klines is None:
        return False
    if isinstance(klines, pd.DataFrame):
        df = klines
    elif isinstance(klines, list):
        valid = [r[:6] for r in klines if isinstance(r, (list, tuple)) and len(r) >= 6]
        if len(valid) < min_len:
            return False
        df = pd.DataFrame(valid, columns=['ts', 'o', 'h', 'l', 'c', 'vol'])
    else:
        return False

    if len(df) < min_len:
        return False

    # OHLC 非 NaN/Inf
    for col in ['o', 'h', 'l', 'c']:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors='coerce')
            if vals.isna().any() or np.isinf(vals).any():
                return False

    # 高 >= 低
    if 'h' in df.columns and 'l' in df.columns:
        if (pd.to_numeric(df['h'], errors='coerce') <
                pd.to_numeric(df['l'], errors='coerce')).any():
            return False

    # 时间戳单调递增
    if 'ts' in df.columns and len(df) >= 2:
        ts = pd.to_numeric(df['ts'], errors='coerce').dropna()
        if len(ts) >= 2 and (ts.diff().dropna() <= 0).any():
            return False

    return True


def check_nan_inf(value: float, default: float = 0.0) -> float:
    """检查并修复 NaN/Inf，返回安全值。"""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    return v if np.isfinite(v) else default


def validate_config(
    config: Dict[str, Any],
    rules: Dict[str, Dict[str, Any]],
) -> List[str]:
    """校验策略配置参数是否在合法范围内。

    Args:
        config: 用户配置字典
        rules: 校验规则字典，格式：
            {
                'param_name': {
                    'min': float,
                    'max': float,
                    'type': float | int,
                    'required': bool,
                }
            }

    Returns: 错误信息列表，空列表表示无错误。
    """
    errors = []
    for key, rule in rules.items():
        if rule.get('required', False) and key not in config:
            errors.append(f"缺少必要参数: {key}")
            continue
        if key not in config or config.get(key) is None:
            continue

        val = config[key]
        expected_type = rule.get('type', float)

        try:
            if expected_type == float:
                val = float(val)
            elif expected_type == int:
                val = int(float(val))
        except (TypeError, ValueError):
            errors.append(f"参数 {key}={val} 类型错误，期望 {expected_type.__name__}")
            continue

        if not np.isfinite(val):
            errors.append(f"参数 {key}={val} 为非法值(NaN/Inf)")
            continue

        if 'min' in rule and val < rule['min']:
            errors.append(f"参数 {key}={val} 小于最小值 {rule['min']}")
        if 'max' in rule and val > rule['max']:
            errors.append(f"参数 {key}={val} 大于最大值 {rule['max']}")

    return errors
