"""
共享交易参数配置。
统一管理自动交易、模拟交易（测试网自动执行）和回测的风险参数，
明确不介入手动交易标签页的独立下单参数。
"""

from __future__ import annotations

import json
import threading
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict


DEFAULT_TRADE_SETTINGS: Dict[str, Dict[str, Any]] = {
    "common": {
        "position_size": 0.10,
        "leverage": 3,
        "take_profit_pct": 5.0,
        "stop_loss_pct": 3.0,
        "allow_short": True,
    },
    "backtest": {
        "fee_pct": 0.05,
        "slippage_pct": 0.03,
        "funding_rate_8h_pct": 0.0,
        "limit_miss_probability_pct": 0.0,
        "market_impact_pct": 0.02,
        "conservative_same_bar_exit": True,
        "respect_selected_bar_as_driver": True,
    },
    "auto_trading": {
        "apply_to_testnet_simulation": True,
        "prefer_market_order": True,
        "signal_cooldown_minutes": 15,
        "auto_trading_capital": 1000.0,
        "pilot_position_pct": 0.01,
        "add_position_pct": 0.10,
        "h4_fast_ema": 20,
        "h4_slow_ema": 60,
        "h1_fast_ema": 12,
        "h1_slow_ema": 26,
        "h1_rsi_min_long": 45.0,
        "h1_rsi_max_long": 70.0,
        "h1_rsi_min_short": 30.0,
        "h1_rsi_max_short": 55.0,
        "m3_fast_ema": 8,
        "m3_mid_ema": 13,
        "m3_slow_ema": 21,
        "m3_pullback_min_pct": 0.30,
        "m3_pullback_max_pct": 1.80,
        "m3_stabilization_bars": 3,
        "m3_breakout_buffer_pct": 0.18,
        "volume_confirm_ratio": 0.75,
        "stop_loss_floor_pct": 2.00,
        "cost_line_atr_multiplier": 2.00,
        "stop_loss_cap_pct": 5.00,
        "stage1_trail_activate_pct": 0.35,
        "stage1_trail_ratio": 0.40,
        "trail_stop_pct": 1.20,
        "trial_loss_block_pct": 0.30,
        "max_hold_hours": 24.0,
        "h1_large_pullback_pct": 3.50,
        "h1_reversal_min_hold_hours": 2.0,

        # ── v2: 动态风控参数 ──────────────────────────────────────────────
        "trail_enabled": True,                 # 追踪止损开关
        "trail_activate_atr": 1.5,            # 浮盈 ≥ N×ATR 后激活追踪
        "trail_distance_atr": 2.0,            # 追踪距离 = N×ATR
        "atr_stop_mult": 2.5,                 # ATR 止损倍数
        "atr_stop_floor_pct": 1.5,            # 保底止损%
        "atr_stop_ceiling_pct": 15.0,         # 止损上限%
        "vol_scale_enabled": True,            # 波动率仓位缩放
        "btc_crash_halt_pct": -5.0,           # BTC 暴跌熔断阈值%
        "btc_crash_halt_minutes": 30,         # 熔断持续时间
        "funding_reversal_threshold": 0.08,   # 费率极端阈值
        "oi_confirmation_enabled": True,       # OI 趋势确认
    },
}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


class SharedTradeSettingsManager:
    """共享交易参数持久化管理器。"""

    def __init__(self, settings_path: str | None = None):
        project_root = Path(__file__).resolve().parent.parent.parent
        self.settings_path = Path(settings_path) if settings_path else project_root / "data" / "trade_parameters.json"
        self.settings_path.parent.mkdir(parents=True, exist_ok=True)
        self._settings: Dict[str, Dict[str, Any]] = deepcopy(DEFAULT_TRADE_SETTINGS)
        self._lock = threading.RLock()
        self.load()

    def load(self) -> Dict[str, Dict[str, Any]]:
        """从磁盘加载配置，不存在时使用默认值。"""
        data: Dict[str, Any] = {}
        if self.settings_path.exists():
            try:
                with self.settings_path.open("r", encoding="utf-8") as f:
                    data = json.load(f) or {}
            except Exception:
                data = {}
        with self._lock:
            self._settings = _deep_merge(DEFAULT_TRADE_SETTINGS, data)
        return self.get_all()

    def save(self) -> None:
        """持久化当前配置（线程安全）。"""
        with self._lock:
            with self.settings_path.open("w", encoding="utf-8") as f:
                json.dump(self._settings, f, ensure_ascii=False, indent=2)

    def reset(self) -> Dict[str, Dict[str, Any]]:
        """恢复默认配置并保存。"""
        self._settings = deepcopy(DEFAULT_TRADE_SETTINGS)
        self.save()
        return self.get_all()

    def get_all(self) -> Dict[str, Dict[str, Any]]:
        """返回完整配置副本。"""
        return deepcopy(self._settings)

    def get_section(self, section: str) -> Dict[str, Any]:
        """返回指定分组配置。"""
        return deepcopy(self._settings.get(section, {}))

    def update(self, payload: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """更新配置并保存（线程安全）。"""
        with self._lock:
            self._settings = _deep_merge(self._settings, payload or {})
        self.save()
        return self.get_all()

    def get_common_risk(self) -> Dict[str, Any]:
        """获取自动交易/回测共用风险参数。"""
        return self.get_section("common")

    def get_backtest_overrides(self) -> Dict[str, Any]:
        """获取回测专用参数。"""
        return self.get_section("backtest")

    def build_backtest_config(self, base_config: Dict[str, Any]) -> Dict[str, Any]:
        """将共享参数合并到回测配置。

        合并优先级（低 → 高）：
          1. base_config（UI 面板原始值）
          2. common（杠杆/止盈/止损等通用风控）
          3. backtest（费率/滑点等成本参数）
          4. auto_trading（Runner 运行时参数：ATR止损、移动止盈等）

        注意：auto_trading 参数仅注入顶层 config（供 StrategyRunner 读取），
        **不再注入 strategy_params**，避免覆盖策略自身优化后的 EMA/RSI/回调参数。
        """
        merged = dict(base_config or {})
        merged.update(self.get_common_risk())
        merged.update(self.get_section("auto_trading"))
        # backtest 参数放最后：确保 fee_pct/slippage 不被其他 section 覆盖
        merged.update(self.get_backtest_overrides())
        # strategy_params 只继承 common 风控（杠杆/止盈/止损），
        # 不继承 auto_trading（防止覆盖策略内部参数如 m3_pullback_min_pct）
        strategy_params = dict(merged.get("strategy_params") or {})
        strategy_params.update(self.get_common_risk())
        merged["strategy_params"] = strategy_params
        return merged

    def build_auto_runtime_config(self, strategy_config: Dict[str, Any]) -> Dict[str, Any]:
        """将共享参数合并到自动交易运行配置。"""
        merged = dict(strategy_config or {})
        merged.update(self.get_common_risk())
        merged.update(self.get_section("auto_trading"))
        return merged
