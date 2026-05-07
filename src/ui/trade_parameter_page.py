"""
独立交易参数设定页。
用于统一管理自动交易、模拟交易（测试网自动执行）与回测的共享风险参数。
"""

from __future__ import annotations

from src.qt_compat import (
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
    Qt,
    Signal,
    QFont,
)


class TradeParameterPage(QWidget):
    """共享交易参数设定页面。"""

    settings_changed = Signal(dict)

    def __init__(self, settings_manager, parent=None):
        super().__init__(parent)
        self.settings_manager = settings_manager
        self.inputs = {}
        self.init_ui()
        self.load_settings_into_form()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(15)

        title = QLabel("交易参数设定")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setStyleSheet("color: #ffffff;")
        layout.addWidget(title)

        scope = QFrame()
        scope.setStyleSheet("""
            QFrame {
                background-color: #1f2a36;
                border: 1px solid #35506b;
                border-radius: 8px;
                padding: 10px;
            }
        """)
        scope_layout = QVBoxLayout(scope)
        scope_desc = QLabel(
            "这里的参数会统一作用于自动交易、测试网模拟自动交易、策略回测。"
            "手动交易标签页的下单金额、杠杆、止盈止损继续独立控制，不受这里影响。"
        )
        scope_desc.setWordWrap(True)
        scope_desc.setStyleSheet("color: #d7e6f5;")
        scope_layout.addWidget(scope_desc)
        layout.addWidget(scope)

        common_group = QGroupBox("通用风险参数")
        common_form = QFormLayout(common_group)

        self.inputs["common.position_size"] = self._make_double(0.01, 1.0, 4, " (资金占比)")
        common_form.addRow("单次仓位比例:", self.inputs["common.position_size"])

        self.inputs["common.leverage"] = self._make_int(1, 125)
        common_form.addRow("默认杠杆:", self.inputs["common.leverage"])

        self.inputs["common.take_profit_pct"] = self._make_double(0.0, 100.0, 2, " %")
        common_form.addRow("止盈比例:", self.inputs["common.take_profit_pct"])

        self.inputs["common.stop_loss_pct"] = self._make_double(0.0, 100.0, 2, " %")
        common_form.addRow("止损比例:", self.inputs["common.stop_loss_pct"])

        self.inputs["common.allow_short"] = QCheckBox("允许做空信号执行")
        common_form.addRow("空头控制:", self.inputs["common.allow_short"])
        layout.addWidget(common_group)

        backtest_group = QGroupBox("回测专用参数")
        backtest_form = QFormLayout(backtest_group)

        self.inputs["backtest.fee_pct"] = self._make_double(0.0, 5.0, 4, " %")
        backtest_form.addRow("手续费率:", self.inputs["backtest.fee_pct"])

        self.inputs["backtest.slippage_pct"] = self._make_double(0.0, 10.0, 4, " %")
        backtest_form.addRow("滑点:", self.inputs["backtest.slippage_pct"])

        self.inputs["backtest.funding_rate_8h_pct"] = self._make_double(-5.0, 5.0, 4, " %/8h")
        backtest_form.addRow("资金费率:", self.inputs["backtest.funding_rate_8h_pct"])

        self.inputs["backtest.limit_miss_probability_pct"] = self._make_double(0.0, 100.0, 2, " %")
        backtest_form.addRow("限价未成交概率:", self.inputs["backtest.limit_miss_probability_pct"])

        self.inputs["backtest.market_impact_pct"] = self._make_double(0.0, 10.0, 4, " %")
        backtest_form.addRow("市价冲击成本:", self.inputs["backtest.market_impact_pct"])

        self.inputs["backtest.conservative_same_bar_exit"] = QCheckBox("同一根K线同时触发止盈止损时，按保守止损处理")
        backtest_form.addRow("同K线风险处理:", self.inputs["backtest.conservative_same_bar_exit"])

        self.inputs["backtest.respect_selected_bar_as_driver"] = QCheckBox("回测优先按界面选择的周期驱动，不隐式降级到更低周期")
        backtest_form.addRow("驱动周期:", self.inputs["backtest.respect_selected_bar_as_driver"])
        layout.addWidget(backtest_group)

        auto_group = QGroupBox("自动交易 / 模拟交易")
        auto_form = QFormLayout(auto_group)
        self.inputs["auto_trading.auto_trading_capital"] = self._make_double(100.0, 100000000.0, 2, " USDT")
        auto_form.addRow("自动交易总资金池:", self.inputs["auto_trading.auto_trading_capital"])

        self.inputs["auto_trading.pilot_position_pct"] = self._make_double(0.001, 0.20, 4, " (试仓占比)")
        auto_form.addRow("首笔试仓比例:", self.inputs["auto_trading.pilot_position_pct"])

        self.inputs["auto_trading.add_position_pct"] = self._make_double(0.001, 1.0, 4, " (加仓占比)")
        auto_form.addRow("二次加仓比例:", self.inputs["auto_trading.add_position_pct"])

        self.inputs["auto_trading.h4_fast_ema"] = self._make_int(5, 200)
        auto_form.addRow("4H快线EMA:", self.inputs["auto_trading.h4_fast_ema"])
        self.inputs["auto_trading.h4_slow_ema"] = self._make_int(10, 300)
        auto_form.addRow("4H慢线EMA:", self.inputs["auto_trading.h4_slow_ema"])
        self.inputs["auto_trading.h1_fast_ema"] = self._make_int(5, 120)
        auto_form.addRow("1H快线EMA:", self.inputs["auto_trading.h1_fast_ema"])
        self.inputs["auto_trading.h1_slow_ema"] = self._make_int(10, 200)
        auto_form.addRow("1H慢线EMA:", self.inputs["auto_trading.h1_slow_ema"])
        self.inputs["auto_trading.h1_rsi_min_long"] = self._make_double(1.0, 99.0, 1)
        auto_form.addRow("1H做多最低RSI:", self.inputs["auto_trading.h1_rsi_min_long"])
        self.inputs["auto_trading.h1_rsi_max_long"] = self._make_double(1.0, 99.0, 1)
        auto_form.addRow("1H做多最高RSI:", self.inputs["auto_trading.h1_rsi_max_long"])
        self.inputs["auto_trading.h1_rsi_min_short"] = self._make_double(1.0, 99.0, 1)
        auto_form.addRow("1H做空最低RSI:", self.inputs["auto_trading.h1_rsi_min_short"])
        self.inputs["auto_trading.h1_rsi_max_short"] = self._make_double(1.0, 99.0, 1)
        auto_form.addRow("1H做空最高RSI:", self.inputs["auto_trading.h1_rsi_max_short"])
        self.inputs["auto_trading.m3_fast_ema"] = self._make_int(3, 100)
        auto_form.addRow("3m快线EMA:", self.inputs["auto_trading.m3_fast_ema"])
        self.inputs["auto_trading.m3_mid_ema"] = self._make_int(5, 120)
        auto_form.addRow("3m中线EMA:", self.inputs["auto_trading.m3_mid_ema"])
        self.inputs["auto_trading.m3_slow_ema"] = self._make_int(8, 200)
        auto_form.addRow("3m慢线EMA:", self.inputs["auto_trading.m3_slow_ema"])

        self.inputs["auto_trading.apply_to_testnet_simulation"] = QCheckBox("测试网模拟自动交易沿用同一套参数")
        auto_form.addRow("模拟交易联动:", self.inputs["auto_trading.apply_to_testnet_simulation"])
        self.inputs["auto_trading.prefer_market_order"] = QCheckBox("自动开仓优先使用市价单")
        auto_form.addRow("自动开仓类型:", self.inputs["auto_trading.prefer_market_order"])
        self.inputs["auto_trading.signal_cooldown_minutes"] = self._make_int(0, 1440)
        auto_form.addRow("同标的信号冷却:", self.inputs["auto_trading.signal_cooldown_minutes"])
        self.inputs["auto_trading.m3_pullback_min_pct"] = self._make_double(0.05, 10.0, 2, " %")
        auto_form.addRow("3m最小回调幅度:", self.inputs["auto_trading.m3_pullback_min_pct"])
        self.inputs["auto_trading.m3_pullback_max_pct"] = self._make_double(0.10, 20.0, 2, " %")
        auto_form.addRow("3m最大回调幅度:", self.inputs["auto_trading.m3_pullback_max_pct"])
        self.inputs["auto_trading.m3_stabilization_bars"] = self._make_int(2, 10)
        auto_form.addRow("3m企稳确认根数:", self.inputs["auto_trading.m3_stabilization_bars"])
        self.inputs["auto_trading.m3_breakout_buffer_pct"] = self._make_double(0.01, 5.0, 2, " %")
        auto_form.addRow("3m延续突破缓冲:", self.inputs["auto_trading.m3_breakout_buffer_pct"])
        self.inputs["auto_trading.volume_confirm_ratio"] = self._make_double(0.10, 3.0, 2, " x均量")
        auto_form.addRow("3m量能确认阈值:", self.inputs["auto_trading.volume_confirm_ratio"])
        self.inputs["auto_trading.h1_large_pullback_pct"] = self._make_double(0.50, 20.0, 2, " %")
        auto_form.addRow("H1大回调离场阈值:", self.inputs["auto_trading.h1_large_pullback_pct"])
        self.inputs["auto_trading.stop_loss_floor_pct"] = self._make_double(0.10, 20.0, 2, " %")
        auto_form.addRow("保底止损距离:", self.inputs["auto_trading.stop_loss_floor_pct"])
        self.inputs["auto_trading.cost_line_atr_multiplier"] = self._make_double(0.10, 10.0, 2, " xATR")
        auto_form.addRow("ATR止损倍数:", self.inputs["auto_trading.cost_line_atr_multiplier"])
        self.inputs["auto_trading.stage1_trail_activate_pct"] = self._make_double(0.10, 20.0, 2, " %")
        auto_form.addRow("试仓浮盈保护启动:", self.inputs["auto_trading.stage1_trail_activate_pct"])
        self.inputs["auto_trading.stage1_trail_ratio"] = self._make_double(0.05, 1.00, 2, " 回撤比例")
        auto_form.addRow("试仓浮盈回撤阈值:", self.inputs["auto_trading.stage1_trail_ratio"])
        self.inputs["auto_trading.trail_stop_pct"] = self._make_double(0.10, 20.0, 2, " %")
        auto_form.addRow("加仓后移动止盈:", self.inputs["auto_trading.trail_stop_pct"])
        self.inputs["auto_trading.max_hold_hours"] = self._make_double(0.5, 240.0, 1, " h")
        auto_form.addRow("最大持仓时长:", self.inputs["auto_trading.max_hold_hours"])
        layout.addWidget(auto_group)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        reset_btn = QPushButton("恢复默认")
        reset_btn.clicked.connect(self.reset_defaults)
        btn_row.addWidget(reset_btn)
        save_btn = QPushButton("保存参数")
        save_btn.setStyleSheet("background-color:#007f5f; color:white; font-weight:bold;")
        save_btn.clicked.connect(self.save_settings)
        btn_row.addWidget(save_btn)
        layout.addLayout(btn_row)
        layout.addStretch()

    def _make_double(self, minimum: float, maximum: float, decimals: int, suffix: str = "") -> QDoubleSpinBox:
        widget = QDoubleSpinBox()
        widget.setRange(minimum, maximum)
        widget.setDecimals(decimals)
        if suffix:
            widget.setSuffix(suffix)
        return widget

    def _make_int(self, minimum: int, maximum: int) -> QSpinBox:
        widget = QSpinBox()
        widget.setRange(minimum, maximum)
        return widget

    def load_settings_into_form(self):
        settings = self.settings_manager.get_all()
        for key, widget in self.inputs.items():
            section, field = key.split(".", 1)
            value = settings.get(section, {}).get(field)
            if value is None:
                continue
            if isinstance(widget, QCheckBox):
                widget.setChecked(bool(value))
            elif isinstance(widget, QSpinBox):
                widget.setValue(int(value))
            elif isinstance(widget, QDoubleSpinBox):
                widget.setValue(float(value))

    def collect_settings(self):
        payload = {"common": {}, "backtest": {}, "auto_trading": {}}
        for key, widget in self.inputs.items():
            section, field = key.split(".", 1)
            if isinstance(widget, QCheckBox):
                payload[section][field] = widget.isChecked()
            elif isinstance(widget, QSpinBox):
                payload[section][field] = widget.value()
            elif isinstance(widget, QDoubleSpinBox):
                payload[section][field] = widget.value()
        return payload

    def save_settings(self):
        payload = self.collect_settings()
        settings = self.settings_manager.update(payload)
        self.settings_changed.emit(settings)
        QMessageBox.information(self, "保存成功", "共享交易参数已保存，并会作用于自动交易、模拟交易和回测。")

    def reset_defaults(self):
        reply = QMessageBox.question(
            self,
            "恢复默认",
            "确定恢复默认交易参数吗？这会覆盖当前自动交易、模拟交易和回测的共享参数。",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        settings = self.settings_manager.reset()
        self.load_settings_into_form()
        self.settings_changed.emit(settings)
        QMessageBox.information(self, "已恢复", "共享交易参数已恢复为默认值。")
