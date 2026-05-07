"""
回测页面模块 - 修复版
"""

import csv
import concurrent.futures
import json
import os
import sys
import threading
from datetime import datetime, timedelta
from typing import Dict, Optional


from src.strategy.loader import StrategyLoader, StrategyInfo
from src.backtest.engine import Backtester, BacktestResult, BacktestAnalyzer
from src.qt_compat import QApplication, QCheckBox, QColor, QComboBox, QDate, QDateEdit, QDoubleSpinBox, QFileDialog, QFont, QFormLayout, QFrame, QGroupBox, QHBoxLayout, QHeaderView, QLabel, QListWidget, QListWidgetItem, QMenu, QAction, QMessageBox, QObject, QProgressBar, QPushButton, QScrollArea, QSpinBox, QSplitter, QTabWidget, QTableWidget, QTableWidgetItem, QTextEdit, QThread, QVBoxLayout, QWidget, Qt, Signal


class BacktestConfigWidget(QWidget):
    """回测配置组件"""

    start_backtest = Signal(dict)
    cancel_backtest = Signal()
    pause_backtest = Signal()
    symbols_loaded = Signal(list)
    symbols_load_failed = Signal(str)

    def __init__(self, strategy_loader: StrategyLoader = None, okx_client=None, data_manager=None, trade_settings_manager=None, open_trade_settings_callback=None):
        super().__init__()
        if strategy_loader:
            self.strategy_loader = strategy_loader
        else:
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            strategies_dir = os.path.join(project_root, 'strategies')
            self.strategy_loader = StrategyLoader(strategies_dir)
        self.okx_client = okx_client
        self.data_manager = data_manager
        self.trade_settings_manager = trade_settings_manager
        self.open_trade_settings_callback = open_trade_settings_callback
        self.selected_strategy: Optional[StrategyInfo] = None
        self.config_inputs: Dict[str, QWidget] = {}
        self.strategy_presets: Dict[str, Dict] = {}
        self.preset_label_map: Dict[str, str] = {}
        self._pinned_strategies: set = self._load_pinned_strategies()
        self.batch_presets_check: Optional[QCheckBox] = None
        self.best_rule_combo: Optional[QComboBox] = None
        self.return_weight_spin: Optional[QDoubleSpinBox] = None
        self.drawdown_weight_spin: Optional[QDoubleSpinBox] = None
        self.sharpe_weight_spin: Optional[QDoubleSpinBox] = None
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self._ui_state_path = os.path.join(project_root, "src", "backtest_ui_state.json")
        self._ui_state = self._load_ui_state()
        self.symbols_loaded.connect(self.on_symbols_loaded)
        self.symbols_load_failed.connect(self.on_symbols_load_failed)
        self.init_ui()
        self.refresh_pairs()
        self.refresh_shared_trade_settings()

    def init_ui(self):
        """初始化 UI"""
        layout = QVBoxLayout(self)
        layout.setSpacing(15)

        # 标题
        title = QLabel("回测配置")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setStyleSheet("color: #ffffff;")
        layout.addWidget(title)

        # 滚动区域
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: transparent;
            }
            QScrollBar:vertical {
                background: #2a2a2a;
                width: 12px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background: #555;
                border-radius: 6px;
                min-height: 30px;
            }
            QScrollBar::handle:vertical:hover {
                background: #666;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)

        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setSpacing(15)

        # 策略选择
        strategy_group = QGroupBox("策略选择")
        strategy_layout = QVBoxLayout(strategy_group)
        strategy_layout.setSpacing(10)

        self.strategy_list = QListWidget()
        self.strategy_list.setStyleSheet("""
            QListWidget {
                background-color: #2a2a2a;
                color: #ffffff;
                border: 1px solid #444;
                border-radius: 4px;
            }
            QListWidget::item {
                padding: 10px;
                border-bottom: 1px solid #333;
            }
            QListWidget::item:hover {
                background-color: #3a3a3a;
            }
            QListWidget::item:selected {
                background-color: #0066cc;
            }
        """)
        self.strategy_list.setMinimumHeight(120)
        self.strategy_list.itemClicked.connect(self.on_strategy_item_clicked)
        self.strategy_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.strategy_list.customContextMenuRequested.connect(self._on_strategy_context_menu)
        strategy_layout.addWidget(self.strategy_list)

        refresh_btn = QPushButton("刷新策略列表")
        refresh_btn.clicked.connect(self.refresh_strategies)
        strategy_layout.addWidget(refresh_btn)

        scroll_layout.addWidget(strategy_group)

        self.strategy_desc = QTextEdit()
        self.strategy_desc.setMaximumHeight(100)
        self.strategy_desc.setReadOnly(True)
        self.strategy_desc.setPlaceholderText("选择策略后显示说明...")
        self.strategy_desc.setStyleSheet("""
            QTextEdit {
                background-color: #2a2a2a;
                color: #cccccc;
                border: 1px solid #333;
                border-radius: 4px;
            }
        """)
        scroll_layout.addWidget(self.strategy_desc)

        market_group = QGroupBox("市场配置")
        market_layout = QFormLayout()

        self.pair_combo = QComboBox()
        self.pair_combo.setEditable(True)
        self.pair_combo.addItem("加载中...")
        market_layout.addRow("交易对:", self.pair_combo)

        self.bar_combo = QComboBox()
        self.bar_combo.addItems(["全周期(1m-1D)", "1m", "3m", "5m", "15m", "30m", "1H", "2H", "4H", "1D", "1W"])
        self.bar_combo.setCurrentText("全周期(1m-1D)")
        market_layout.addRow("K 线周期:", self.bar_combo)

        refresh_pairs_btn = QPushButton("刷新合约列表")
        refresh_pairs_btn.clicked.connect(self.refresh_pairs)
        market_layout.addRow("", refresh_pairs_btn)

        market_group.setLayout(market_layout)
        scroll_layout.addWidget(market_group)

        time_group = QGroupBox("回测时间")
        time_layout = QFormLayout()

        self.start_date = QDateEdit()
        self.start_date.setCalendarPopup(True)
        self.start_date.setDate(QDate.currentDate().addMonths(-6))
        time_layout.addRow("开始日期:", self.start_date)

        self.end_date = QDateEdit()
        self.end_date.setCalendarPopup(True)
        self.end_date.setDate(QDate.currentDate())
        time_layout.addRow("结束日期:", self.end_date)

        self.start_date.dateChanged.connect(lambda *_: self._save_ui_state())
        self.end_date.dateChanged.connect(lambda *_: self._save_ui_state())

        time_group.setLayout(time_layout)
        scroll_layout.addWidget(time_group)

        capital_group = QGroupBox("资金配置")
        capital_layout = QFormLayout()

        self.initial_capital = QDoubleSpinBox()
        self.initial_capital.setRange(100, 1000000)
        self.initial_capital.setValue(10000)
        self.initial_capital.setSuffix(" USDT")
        self.initial_capital.setDecimals(2)
        capital_layout.addRow("初始资金:", self.initial_capital)

        self.position_size = QDoubleSpinBox()
        self.position_size.setRange(0.01, 1.0)
        self.position_size.setValue(0.1)
        self.position_size.setSuffix(" (10%)")
        self.position_size.setDecimals(2)
        self.position_size.setEnabled(False)
        capital_layout.addRow("仓位比例:", self.position_size)

        self.shared_risk_summary = QLabel("共享风控参数加载中...")
        self.shared_risk_summary.setWordWrap(True)
        self.shared_risk_summary.setStyleSheet("color: #bbbbbb;")
        capital_layout.addRow("止盈/止损:", self.shared_risk_summary)

        self.open_trade_settings_btn = QPushButton("打开交易参数设定")
        self.open_trade_settings_btn.clicked.connect(self._open_trade_settings_page)
        capital_layout.addRow("", self.open_trade_settings_btn)

        capital_group.setLayout(capital_layout)
        scroll_layout.addWidget(capital_group)

        cost_group = QGroupBox("实盘成本模型")
        cost_layout = QFormLayout()

        self.fee_pct_spin = QDoubleSpinBox()
        self.fee_pct_spin.setRange(0, 5)
        self.fee_pct_spin.setDecimals(4)
        self.fee_pct_spin.setValue(0.05)
        self.fee_pct_spin.setSuffix(" %")
        self.fee_pct_spin.setEnabled(False)
        cost_layout.addRow("手续费率:", self.fee_pct_spin)

        self.slippage_pct_spin = QDoubleSpinBox()
        self.slippage_pct_spin.setRange(0, 10)
        self.slippage_pct_spin.setDecimals(4)
        self.slippage_pct_spin.setValue(0.03)
        self.slippage_pct_spin.setSuffix(" %")
        self.slippage_pct_spin.setEnabled(False)
        cost_layout.addRow("滑点:", self.slippage_pct_spin)

        self.funding_rate_spin = QDoubleSpinBox()
        self.funding_rate_spin.setRange(-5, 5)
        self.funding_rate_spin.setDecimals(4)
        self.funding_rate_spin.setValue(0.0)
        self.funding_rate_spin.setSuffix(" %/8h")
        self.funding_rate_spin.setEnabled(False)
        cost_layout.addRow("资金费率:", self.funding_rate_spin)

        self.limit_miss_spin = QDoubleSpinBox()
        self.limit_miss_spin.setRange(0, 100)
        self.limit_miss_spin.setDecimals(2)
        self.limit_miss_spin.setValue(0.0)
        self.limit_miss_spin.setSuffix(" %")
        self.limit_miss_spin.setEnabled(False)
        cost_layout.addRow("限价未成交概率:", self.limit_miss_spin)

        self.market_impact_spin = QDoubleSpinBox()
        self.market_impact_spin.setRange(0, 10)
        self.market_impact_spin.setDecimals(4)
        self.market_impact_spin.setValue(0.02)
        self.market_impact_spin.setSuffix(" %")
        self.market_impact_spin.setEnabled(False)
        cost_layout.addRow("市价冲击成本:", self.market_impact_spin)

        self.backtest_hint = QLabel("以上仓位与成本参数已改为由“交易参数设定”页统一控制。")
        self.backtest_hint.setWordWrap(True)
        self.backtest_hint.setStyleSheet("color: #bbbbbb;")
        cost_layout.addRow("", self.backtest_hint)

        cost_group.setLayout(cost_layout)
        scroll_layout.addWidget(cost_group)

        self.params_group = QGroupBox("策略参数")
        self.params_layout = QFormLayout()
        self.params_group.setLayout(self.params_layout)
        scroll_layout.addWidget(self.params_group)

        scroll_layout.addStretch()

        scroll.setWidget(scroll_content)
        layout.addWidget(scroll)

        # 控制按钮
        btn_layout = QHBoxLayout()

        self.start_btn = QPushButton("开始回测")
        self.start_btn.setMinimumHeight(50)
        self.start_btn.setStyleSheet("""
            QPushButton {
                background-color: #00aa00;
                color: white;
                border-radius: 8px;
                font-weight: bold;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #00cc00;
            }
            QPushButton:disabled {
                background-color: #555;
            }
        """)
        self.start_btn.clicked.connect(self.on_start_backtest)
        btn_layout.addWidget(self.start_btn)

        self.pause_btn = QPushButton("暂停")
        self.pause_btn.setMinimumHeight(50)
        self.pause_btn.setStyleSheet("""
            QPushButton {
                background-color: #ffaa00;
                color: white;
                border-radius: 8px;
                font-weight: bold;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #ffcc00;
            }
            QPushButton:disabled {
                background-color: #555;
            }
        """)
        self.pause_btn.clicked.connect(self.on_pause_backtest)
        self.pause_btn.setEnabled(False)
        btn_layout.addWidget(self.pause_btn)

        self.stop_btn = QPushButton("停止")
        self.stop_btn.setMinimumHeight(50)
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #aa0000;
                color: white;
                border-radius: 8px;
                font-weight: bold;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #cc0000;
            }
            QPushButton:disabled {
                background-color: #555;
            }
        """)
        self.stop_btn.clicked.connect(self.on_cancel_backtest)
        self.stop_btn.setEnabled(False)
        btn_layout.addWidget(self.stop_btn)

        layout.addLayout(btn_layout)

        self.refresh_strategies()
        self._restore_ui_state()

    def refresh_pairs(self):
        """刷新本地数据库中存在数据的交易对列表。"""
        self.pair_combo.clear()
        self.pair_combo.addItem("正在加载本地交易对...")
        thread = threading.Thread(target=self.fetch_local_symbols_with_data, daemon=True)
        thread.start()

    def fetch_local_symbols_with_data(self):
        """从本地数据库读取有数据的交易对。"""
        try:
            if self.data_manager:
                symbols = self.data_manager.get_symbols_with_local_data()
                self.symbols_loaded.emit(symbols)
            else:
                self.symbols_load_failed.emit("未配置本地数据管理器")
        except Exception as e:
            self.symbols_load_failed.emit(f"读取本地交易对列表失败: {e}")

    def on_symbols_loaded(self, symbols):
        """主线程更新交易对选择框"""
        current_text = self._ui_state.get("inst_id") or self.pair_combo.currentText()
        self.pair_combo.clear()
        if not symbols:
            self.pair_combo.addItem("本地数据库无可用交易对")
            return

        self.pair_combo.addItems(symbols)
        if current_text and current_text in symbols:
            self.pair_combo.setCurrentText(current_text)
        else:
            self.pair_combo.setCurrentIndex(0)

    def on_symbols_load_failed(self, message: str):
        """交易对列表加载失败后的降级处理"""
        current_text = self._ui_state.get("inst_id") or self.pair_combo.currentText()
        self.pair_combo.clear()
        fallback_symbols = []
        if current_text:
            fallback_symbols.append(current_text)
        if not fallback_symbols:
            self.pair_combo.addItem("本地数据库无可用交易对")
            self.strategy_desc.setPlaceholderText(message)
            return
        self.pair_combo.addItems(fallback_symbols)
        if current_text in fallback_symbols:
            self.pair_combo.setCurrentText(current_text)
        self.strategy_desc.setPlaceholderText(message)

    def refresh_strategies(self):
        """刷新策略列表（置顶策略优先排列）"""
        self.strategy_list.clear()
        self.strategy_list.addItem("请选择策略...")

        strategies = self.strategy_loader.discover_strategies()
        # 置顶策略排前面
        pinned = [s for s in strategies if s.name in self._pinned_strategies]
        others = [s for s in strategies if s.name not in self._pinned_strategies]
        for strategy in pinned:
            item = QListWidgetItem(f"📌 {strategy.name} ({strategy.type.value})")
            item.setData(Qt.UserRole, strategy)
            item.setForeground(QColor("#ffcc00"))
            self.strategy_list.addItem(item)
        for strategy in others:
            item = QListWidgetItem(f"{strategy.name} ({strategy.type.value})")
            item.setData(Qt.UserRole, strategy)
            self.strategy_list.addItem(item)

    # ── 策略置顶功能 ──────────────────────────────────────────────────────

    def _on_strategy_context_menu(self, pos):
        """策略列表右键菜单"""
        item = self.strategy_list.itemAt(pos)
        if not item:
            return
        strategy_info = item.data(Qt.UserRole)
        if not strategy_info:
            return

        menu = QMenu(self)
        menu.setStyleSheet("""
            QMenu {
                background-color: #2a2a2a;
                color: #ffffff;
                border: 1px solid #555;
                border-radius: 4px;
                padding: 4px;
            }
            QMenu::item {
                padding: 6px 24px;
            }
            QMenu::item:selected {
                background-color: #0066cc;
            }
        """)

        is_pinned = strategy_info.name in self._pinned_strategies
        if is_pinned:
            action = QAction("取消置顶", self)
            action.triggered.connect(lambda: self._toggle_pin_strategy(strategy_info.name, False))
        else:
            action = QAction("📌 置顶", self)
            action.triggered.connect(lambda: self._toggle_pin_strategy(strategy_info.name, True))
        menu.addAction(action)
        menu.exec_(self.strategy_list.viewport().mapToGlobal(pos))

    def _toggle_pin_strategy(self, name: str, pin: bool):
        """切换策略置顶状态"""
        if pin:
            self._pinned_strategies.add(name)
        else:
            self._pinned_strategies.discard(name)
        self._save_pinned_strategies()
        self.refresh_strategies()

    def _pinned_file_path(self) -> str:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        return os.path.join(project_root, "pinned_strategies.json")

    def _load_pinned_strategies(self) -> set:
        try:
            path = self._pinned_file_path()
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return set(data) if isinstance(data, list) else set()
        except Exception:
            pass
        return set()

    def _save_pinned_strategies(self):
        try:
            with open(self._pinned_file_path(), "w", encoding="utf-8") as f:
                json.dump(sorted(self._pinned_strategies), f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def on_strategy_item_clicked(self, item):
        """策略列表项点击事件"""
        text = item.text()
        if text == "请选择策略...":
            return

        self.selected_strategy = item.data(Qt.UserRole)
        self.strategy_presets = {}
        self.batch_presets_check = None
        self.best_rule_combo = None
        self.return_weight_spin = None
        self.drawdown_weight_spin = None
        self.sharpe_weight_spin = None

        # 清空参数配置
        while self.params_layout.count():
            child = self.params_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        self.config_inputs.clear()
        self.preset_label_map = {}

        if self.selected_strategy:
            # 显示策略说明
            desc = f"策略：{self.selected_strategy.name}\\n"
            if self.selected_strategy.description:
                desc += f"说明：{self.selected_strategy.description}\\n"
            if self.selected_strategy.author:
                desc += f"作者：{self.selected_strategy.author}\\n"
            if self.selected_strategy.version:
                desc += f"版本：{self.selected_strategy.version}"
            self.strategy_desc.setText(desc)

            # 加载策略并显示参数
            module = self.strategy_loader.load_strategy(self.selected_strategy.name)
            if module and self.selected_strategy.config_schema:
                self.strategy_presets = getattr(module, 'PRESET_CONFIGS', {}) or {}
                if self.strategy_presets:
                    batch_label = getattr(module, 'BATCH_PRESET_LABEL', "批量回测全部预设模板")
                    self.batch_presets_check = QCheckBox(str(batch_label))
                    self.batch_presets_check.setChecked(False)
                    self.params_layout.addRow("批量参数:", self.batch_presets_check)
                    self.best_rule_combo = QComboBox()
                    self.best_rule_combo.addItem("按收益选择最佳模板", "return")
                    self.best_rule_combo.addItem("按回撤选择最佳模板", "drawdown")
                    self.best_rule_combo.addItem("按夏普选择最佳模板", "sharpe")
                    self.best_rule_combo.addItem("按综合评分选择最佳模板", "composite")
                    self.params_layout.addRow("最佳模板规则:", self.best_rule_combo)

                    self.return_weight_spin = QDoubleSpinBox()
                    self.return_weight_spin.setRange(0.0, 1.0)
                    self.return_weight_spin.setSingleStep(0.05)
                    self.return_weight_spin.setDecimals(2)
                    self.return_weight_spin.setValue(0.5)
                    self.params_layout.addRow("收益权重:", self.return_weight_spin)

                    self.drawdown_weight_spin = QDoubleSpinBox()
                    self.drawdown_weight_spin.setRange(0.0, 1.0)
                    self.drawdown_weight_spin.setSingleStep(0.05)
                    self.drawdown_weight_spin.setDecimals(2)
                    self.drawdown_weight_spin.setValue(0.3)
                    self.params_layout.addRow("回撤权重:", self.drawdown_weight_spin)

                    self.sharpe_weight_spin = QDoubleSpinBox()
                    self.sharpe_weight_spin.setRange(0.0, 1.0)
                    self.sharpe_weight_spin.setSingleStep(0.05)
                    self.sharpe_weight_spin.setDecimals(2)
                    self.sharpe_weight_spin.setValue(0.2)
                    self.params_layout.addRow("夏普权重:", self.sharpe_weight_spin)
                for param_name, param_info in self.selected_strategy.config_schema.items():
                    label = param_info.get('label', param_name)
                    param_type = param_info.get('type', 'float')
                    default_value = param_info.get('default', 0)

                    if param_type == 'select':
                        input_widget = QComboBox()
                        options = param_info.get('options', [])
                        for option in options:
                            if isinstance(option, dict):
                                option_label = str(option.get('label', option.get('value', '')))
                                option_value = option.get('value')
                                input_widget.addItem(option_label, option_value)
                                if param_name == 'preset' and option_value is not None:
                                    self.preset_label_map[str(option_value)] = option_label
                            else:
                                input_widget.addItem(str(option), option)
                                if param_name == 'preset':
                                    self.preset_label_map[str(option)] = str(option)
                        default_index = input_widget.findData(default_value)
                        if default_index < 0:
                            default_index = input_widget.findText(str(default_value))
                        if default_index >= 0:
                            input_widget.setCurrentIndex(default_index)
                        if param_name == 'preset':
                            input_widget.currentIndexChanged.connect(self.apply_selected_preset)
                    elif param_type == 'int':
                        input_widget = QSpinBox()
                        input_widget.setRange(-999999, 999999)
                        input_widget.setValue(int(default_value))
                    elif param_type == 'float':
                        input_widget = QDoubleSpinBox()
                        input_widget.setRange(-999999.0, 999999.0)
                        input_widget.setDecimals(4)
                        input_widget.setValue(float(default_value))
                    elif param_type == 'bool':
                        input_widget = QCheckBox()
                        input_widget.setChecked(bool(default_value))
                    else:
                        input_widget = QDoubleSpinBox()
                        input_widget.setRange(-999999.0, 999999.0)
                        input_widget.setDecimals(4)
                        input_widget.setValue(float(default_value))

                    self.params_layout.addRow(f"{label}:", input_widget)
                    self.config_inputs[param_name] = input_widget

                if 'preset' in self.config_inputs:
                    self.apply_selected_preset()
        else:
            self.strategy_desc.clear()

    def apply_selected_preset(self):
        """应用预设参数模板"""
        preset_widget = self.config_inputs.get('preset')
        if not isinstance(preset_widget, QComboBox):
            return

        preset_key = preset_widget.currentData()
        if preset_key in (None, '', 'custom'):
            return

        preset_values = self.strategy_presets.get(str(preset_key), {})
        if not preset_values:
            return

        for param_name, param_value in preset_values.items():
            input_widget = self.config_inputs.get(param_name)
            if not input_widget or param_name == 'preset':
                continue
            if hasattr(input_widget, 'setChecked'):
                input_widget.setChecked(bool(param_value))
            elif isinstance(input_widget, QComboBox):
                idx = input_widget.findData(param_value)
                if idx < 0:
                    idx = input_widget.findText(str(param_value))
                if idx >= 0:
                    input_widget.setCurrentIndex(idx)
            elif hasattr(input_widget, 'setValue'):
                input_widget.setValue(param_value)

    def on_start_backtest(self):
        """开始回测"""
        config = self.get_config()
        if not config:
            QMessageBox.warning(self, "警告", "请先选择策略")
            return
        start_dt = self.start_date.date()
        end_dt = self.end_date.date()
        if start_dt > end_dt:
            QMessageBox.warning(self, "警告", "开始日期不能晚于结束日期")
            return
        if not self.pair_combo.currentText().strip():
            QMessageBox.warning(self, "警告", "请选择交易对")
            return
        self.start_backtest.emit(config)

    def on_cancel_backtest(self):
        """取消回测"""
        self.cancel_backtest.emit()

    def on_pause_backtest(self):
        """暂停/继续回测"""
        self.pause_backtest.emit()

    def get_config(self) -> Dict:
        """获取配置"""
        strategy = self.selected_strategy
        if not strategy:
            return {}

        strategy_params = {}
        for param_name, input_widget in self.config_inputs.items():
            if hasattr(input_widget, 'isChecked'):
                strategy_params[param_name] = input_widget.isChecked()
            elif isinstance(input_widget, QComboBox):
                strategy_params[param_name] = input_widget.currentData() or input_widget.currentText()
            elif hasattr(input_widget, 'value'):
                strategy_params[param_name] = input_widget.value()

        config = {
            'strategy_name': strategy.name,
            'strategy_path': strategy.path,
            'inst_id': self.pair_combo.currentText(),
            'bar': self.bar_combo.currentText(),
            'start_date': self.start_date.date().toString("yyyy-MM-dd"),
            'end_date': self.end_date.date().toString("yyyy-MM-dd"),
            'initial_capital': self.initial_capital.value(),
            'position_size': self.position_size.value(),
            'fee_pct': self.fee_pct_spin.value(),
            'slippage_pct': self.slippage_pct_spin.value(),
            'funding_rate_8h_pct': self.funding_rate_spin.value(),
            'limit_miss_probability_pct': self.limit_miss_spin.value(),
            'market_impact_pct': self.market_impact_spin.value(),
            'strategy_params': strategy_params,
            'preset_configs': dict(self.strategy_presets),
            'preset_labels': dict(self.preset_label_map),
            'batch_presets': [key for key in self.strategy_presets.keys() if key != 'custom'] if self.batch_presets_check and self.batch_presets_check.isChecked() else [],
            'best_template_rule': self.best_rule_combo.currentData() if self.best_rule_combo else 'return',
            'best_template_weights': {
                'return': self.return_weight_spin.value() if self.return_weight_spin else 0.5,
                'drawdown': self.drawdown_weight_spin.value() if self.drawdown_weight_spin else 0.3,
                'sharpe': self.sharpe_weight_spin.value() if self.sharpe_weight_spin else 0.2,
            },
            'backtest_log_level': 'KEY',
            'batch_parallel_workers': max(1, min(4, os.cpu_count() or 2)),
        }
        if self.trade_settings_manager:
            config = self.trade_settings_manager.build_backtest_config(config)
        self._save_ui_state()
        return config

    def _load_ui_state(self) -> Dict:
        try:
            if os.path.exists(self._ui_state_path):
                with open(self._ui_state_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        return data
        except Exception:
            pass
        return {}

    def _restore_ui_state(self):
        state = dict(self._ui_state or {})
        start_date = state.get("start_date")
        end_date = state.get("end_date")
        if start_date:
            qd = QDate.fromString(str(start_date), "yyyy-MM-dd")
            if qd.isValid():
                self.start_date.setDate(qd)
        if end_date:
            qd = QDate.fromString(str(end_date), "yyyy-MM-dd")
            if qd.isValid():
                self.end_date.setDate(qd)
        self.bar_combo.setCurrentText("全周期(1m-1D)")

    def _save_ui_state(self):
        state = {
            "inst_id": self.pair_combo.currentText().strip(),
            "start_date": self.start_date.date().toString("yyyy-MM-dd"),
            "end_date": self.end_date.date().toString("yyyy-MM-dd"),
        }
        try:
            with open(self._ui_state_path, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _open_trade_settings_page(self):
        if callable(self.open_trade_settings_callback):
            self.open_trade_settings_callback()

    def refresh_shared_trade_settings(self):
        if not self.trade_settings_manager:
            return
        settings = self.trade_settings_manager.get_all()
        common = settings.get('common', {})
        backtest = settings.get('backtest', {})

        self.position_size.setValue(float(common.get('position_size', 0.10) or 0.10))
        self.fee_pct_spin.setValue(float(backtest.get('fee_pct', 0.05) or 0.05))
        self.slippage_pct_spin.setValue(float(backtest.get('slippage_pct', 0.03) or 0.03))
        self.funding_rate_spin.setValue(float(backtest.get('funding_rate_8h_pct', 0.0) or 0.0))
        self.limit_miss_spin.setValue(float(backtest.get('limit_miss_probability_pct', 0.0) or 0.0))
        self.market_impact_spin.setValue(float(backtest.get('market_impact_pct', 0.02) or 0.02))
        self.shared_risk_summary.setText(
            f"止盈 {float(common.get('take_profit_pct', 5.0)):.2f}% / "
            f"止损 {float(common.get('stop_loss_pct', 3.0)):.2f}% / "
            f"默认杠杆 {int(common.get('leverage', 3) or 3)}x / "
            f"{'允许做空' if bool(common.get('allow_short', True)) else '仅做多'}"
        )

    def set_running(self, running: bool):
        """设置运行状态"""
        self.start_btn.setEnabled(not running)
        self.pause_btn.setEnabled(running)
        self.stop_btn.setEnabled(running)
    
    def set_paused(self, paused: bool):
        """设置暂停状态"""
        self.pause_btn.setText("继续" if paused else "暂停")


class BacktestResultWidget(QWidget):
    """回测结果展示组件"""

    def __init__(self):
        super().__init__()
        self.current_result: Optional[BacktestResult] = None
        self.current_config: Dict = {}
        self.batch_results = []
        self.init_ui()

    def init_ui(self):
        """初始化 UI"""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        title = QLabel("回测结果")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setStyleSheet("color: #ffffff;")
        layout.addWidget(title)

        self.tabs = QTabWidget()

        # 概览页面
        overview_widget = QWidget()
        overview_layout = QVBoxLayout(overview_widget)

        self.metrics_table = QTableWidget()
        self.metrics_table.setColumnCount(2)
        self.metrics_table.setHorizontalHeaderLabels(["指标", "数值"])
        self.metrics_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        overview_layout.addWidget(self.metrics_table)

        self.tabs.addTab(overview_widget, "概览")

        config_widget = QWidget()
        config_layout = QVBoxLayout(config_widget)
        self.config_table = QTableWidget()
        self.config_table.setColumnCount(2)
        self.config_table.setHorizontalHeaderLabels(["配置项", "数值"])
        self.config_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        config_layout.addWidget(self.config_table)
        self.tabs.addTab(config_widget, "配置")

        # 交易记录页面
        trades_widget = QWidget()
        trades_layout = QVBoxLayout(trades_widget)

        self.trades_table = QTableWidget()
        self.trades_table.setColumnCount(9)
        self.trades_table.setHorizontalHeaderLabels([
            "序号", "方向", "开仓时间", "平仓时间", "开仓市场价", "开仓成交价", "平仓触发价", "平仓成交价", "盈亏"
        ])
        self.trades_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        trades_layout.addWidget(self.trades_table)

        self.tabs.addTab(trades_widget, "交易记录")

        compare_widget = QWidget()
        compare_layout = QVBoxLayout(compare_widget)
        self.compare_table = QTableWidget()
        self.compare_table.setColumnCount(10)
        self.compare_table.setHorizontalHeaderLabels([
            "模板", "综合评分", "准入评分", "最终资金", "总收益率", "最大回撤", "夏普", "总交易", "胜率", "备注"
        ])
        self.compare_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.compare_table.setSortingEnabled(True)
        compare_layout.addWidget(self.compare_table)
        self.tabs.addTab(compare_widget, "参数对比")

        heatmap_widget = QWidget()
        heatmap_layout = QVBoxLayout(heatmap_widget)
        self.heatmap_hint = QLabel("颜色越亮代表相对表现越强；回撤列已自动按“越低越好”处理。")
        self.heatmap_hint.setStyleSheet("color: #bbbbbb;")
        heatmap_layout.addWidget(self.heatmap_hint)
        self.heatmap_table = QTableWidget()
        self.heatmap_table.setColumnCount(6)
        self.heatmap_table.setHorizontalHeaderLabels([
            "模板", "综合评分", "总收益率", "最大回撤", "夏普", "胜率"
        ])
        self.heatmap_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        heatmap_layout.addWidget(self.heatmap_table)
        self.tabs.addTab(heatmap_widget, "参数热力图")

        # 报告页面
        report_widget = QWidget()
        report_layout = QVBoxLayout(report_widget)

        self.report_text = QTextEdit()
        self.report_text.setReadOnly(True)
        self.report_text.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #00ff00;
                font-family: 'Menlo', 'Monaco';
                font-size: 12px;
                border: 1px solid #333;
            }
        """)
        report_layout.addWidget(self.report_text)

        self.tabs.addTab(report_widget, "详细报告")

        layout.addWidget(self.tabs)

        export_btn = QPushButton("导出报告")
        export_btn.clicked.connect(self.export_report)
        layout.addWidget(export_btn)

    def _preset_display_name(self, preset_key: str) -> str:
        labels = (self.current_config or {}).get('preset_labels') or {}
        return str(labels.get(str(preset_key), preset_key or ""))

    def _build_batch_report_summary(self, batch_items: list, best_payload: Dict) -> str:
        if not batch_items:
            return ""

        best_rule = (best_payload or {}).get('rule', 'return')
        weights = (best_payload or {}).get('weights') or {}
        best_item = (best_payload or {}).get('best') or {}
        best_result = best_item.get('result')
        lines = [
            "========================================",
            "         模板对比摘要",
            "========================================",
            f"最佳模板规则：{best_rule}",
        ]
        if best_rule == 'composite':
            lines.append(
                "综合权重："
                f"收益 {float(weights.get('return', 0.5)):.2f} / "
                f"回撤 {float(weights.get('drawdown', 0.3)):.2f} / "
                f"夏普 {float(weights.get('sharpe', 0.2)):.2f}"
            )
        if best_result:
            lines.append(
                f"当前最佳：{self._preset_display_name(best_item.get('preset', ''))} | "
                f"收益 {best_result.total_return:.2f}% | 回撤 {best_result.max_drawdown:.2f}% | "
                f"夏普 {best_result.sharpe_ratio:.2f} | 准入 {best_result.live_readiness_score:.2f}"
            )
        lines.append("----------------------------------------")
        lines.append("模板逐项对比")

        ranked_items = sorted(
            [item for item in batch_items if item.get('result')],
            key=lambda item: item.get('composite_score', 0.0),
            reverse=True,
        )
        baseline = ranked_items[0]['result'] if ranked_items else None
        for idx, item in enumerate(ranked_items, start=1):
            result = item['result']
            lines.append(
                f"{idx}. {self._preset_display_name(item.get('preset', ''))} | "
                f"综合 {item.get('composite_score', 0.0):.4f} | "
                f"收益 {result.total_return:.2f}% | 回撤 {result.max_drawdown:.2f}% | "
                f"夏普 {result.sharpe_ratio:.2f} | 胜率 {result.win_rate:.2f}% | "
                f"交易 {result.total_trades} | 准入 {result.live_readiness_score:.2f} | "
                f"{self._validation_note(item, best_item is item)}"
            )
            if baseline and idx > 1:
                lines.append(
                    f"   相对最佳差异：收益 {result.total_return - baseline.total_return:+.2f}% | "
                    f"回撤 {result.max_drawdown - baseline.max_drawdown:+.2f}% | "
                    f"夏普 {result.sharpe_ratio - baseline.sharpe_ratio:+.2f}"
                )

        if len(ranked_items) >= 2:
            top = ranked_items[0]['result']
            second = ranked_items[1]['result']
            lines.append("----------------------------------------")
            lines.append(
                f"优先结论：{self._preset_display_name(ranked_items[0].get('preset', ''))} "
                f"相比 {self._preset_display_name(ranked_items[1].get('preset', ''))}，"
                f"收益 {top.total_return - second.total_return:+.2f}% 、"
                f"回撤 {top.max_drawdown - second.max_drawdown:+.2f}% 、"
                f"夏普 {top.sharpe_ratio - second.sharpe_ratio:+.2f}。"
            )

        lines.append("========================================")
        return "\n".join(lines)

    def display_result(self, result: BacktestResult, config: Dict = None):
        """显示回测结果"""
        self.current_result = result
        self.current_config = config or {}
        self.batch_results = []

        self.metrics_table.setRowCount(0)

        metrics = [
            ("策略名称", result.strategy_name),
            ("交易对", result.inst_id),
            ("回测区间", f"{result.start_date.strftime('%Y-%m-%d')} 至 {result.end_date.strftime('%Y-%m-%d')}"),
            ("初始资金", f"{result.initial_capital:,.2f} USDT"),
            ("最终资金", f"{result.final_capital:,.2f} USDT"),
            ("总收益率", f"{result.total_return:.2f}%"),
            ("年化收益率", f"{result.annual_return:.2f}%"),
            ("最大回撤", f"{result.max_drawdown:.2f}%"),
            ("夏普比率", f"{result.sharpe_ratio:.2f}"),
            ("索提诺比率", f"{result.sortino_ratio:.2f}"),
            ("卡玛比率", f"{getattr(result, 'calmar_ratio', 0.0):.2f}"),
            ("VaR-95%（单笔最坏）", f"{getattr(result, 'var_95', 0.0):.2f}%"),
            ("恢复因子", f"{getattr(result, 'recovery_factor', 0.0):.2f}"),
            ("溃疡指数", f"{getattr(result, 'ulcer_index', 0.0):.2f}%"),
            ("实盘准入评分", f"{result.live_readiness_score:.2f}"),
            ("最近7天收益", f"{result.stats.get('return_7d', 0.0):.2f}%"),
            ("最近30天收益", f"{result.stats.get('return_30d', 0.0):.2f}%"),
            ("最近90天收益", f"{result.stats.get('return_90d', 0.0):.2f}%"),
            ("最大连续亏损", str(result.max_consecutive_losses)),
            ("总交易次数", str(result.total_trades)),
            ("胜率", f"{result.win_rate:.2f}%"),
            ("盈亏比", f"{result.profit_factor:.2f}"),
            ("平均盈利", f"{result.avg_win:,.2f} USDT"),
            ("平均亏损", f"{result.avg_loss:,.2f} USDT"),
            ("试仓次数", str(result.stats.get('pilot_trade_count', 0))),
            ("试仓成功率", f"{result.stats.get('pilot_success_rate', 0.0):.2f}%"),
            ("二次加仓次数", str(result.stats.get('add_on_trade_count', 0))),
            ("二次加仓成功率", f"{result.stats.get('add_on_success_rate', 0.0):.2f}%"),
            ("第一原则触发次数", str(result.stats.get('first_principle_trigger_count', 0))),
            ("手续费成本", f"{result.stats.get('total_fees', 0.0):,.2f} USDT"),
            ("滑点/冲击成本", f"{result.stats.get('total_slippage_cost', 0.0):,.2f} USDT"),
            ("资金费率成本", f"{result.stats.get('total_funding_cost', 0.0):,.2f} USDT"),
        ]

        for label, value in metrics:
            row = self.metrics_table.rowCount()
            self.metrics_table.insertRow(row)
            self.metrics_table.setItem(row, 0, QTableWidgetItem(label))
            self.metrics_table.setItem(row, 1, QTableWidgetItem(str(value)))

        self.config_table.setRowCount(0)
        config_items = []
        if config:
            config_items.extend([
                ("交易对", config.get('inst_id', '')),
                ("K线周期", config.get('bar', '')),
                ("开始日期", config.get('start_date', '')),
                ("结束日期", config.get('end_date', '')),
                ("初始资金", f"{config.get('initial_capital', 0):,.2f} USDT"),
                ("仓位比例", f"{config.get('position_size', 0):.2%}"),
                ("手续费率", f"{config.get('fee_pct', 0):.4f}%"),
                ("滑点", f"{config.get('slippage_pct', 0):.4f}%"),
                ("资金费率", f"{config.get('funding_rate_8h_pct', 0):.4f}%/8h"),
                ("限价未成交概率", f"{config.get('limit_miss_probability_pct', 0):.2f}%"),
                ("市价冲击成本", f"{config.get('market_impact_pct', 0):.4f}%"),
            ])
            for key, value in (config.get('strategy_params') or {}).items():
                config_items.append((f"策略参数:{key}", value))

        for label, value in config_items:
            row = self.config_table.rowCount()
            self.config_table.insertRow(row)
            self.config_table.setItem(row, 0, QTableWidgetItem(str(label)))
            self.config_table.setItem(row, 1, QTableWidgetItem(str(value)))

        self.trades_table.setRowCount(0)
        for i, trade in enumerate(result.trades):
            if trade.exit_time:
                row = self.trades_table.rowCount()
                self.trades_table.insertRow(row)
                direction = "做多" if trade.direction.name == "LONG" else "做空"
                self.trades_table.setItem(row, 0, QTableWidgetItem(str(i + 1)))
                self.trades_table.setItem(row, 1, QTableWidgetItem(direction))
                self.trades_table.setItem(row, 2, QTableWidgetItem(trade.entry_time.strftime("%Y-%m-%d %H:%M")))
                self.trades_table.setItem(row, 3, QTableWidgetItem(trade.exit_time.strftime("%Y-%m-%d %H:%M")))
                self.trades_table.setItem(row, 4, QTableWidgetItem(f"{getattr(trade, 'raw_entry_price', trade.entry_price):.4f}"))
                self.trades_table.setItem(row, 5, QTableWidgetItem(f"{trade.entry_price:.4f}"))
                self.trades_table.setItem(row, 6, QTableWidgetItem(f"{getattr(trade, 'raw_exit_price', trade.exit_price):.4f}"))
                self.trades_table.setItem(row, 7, QTableWidgetItem(f"{trade.exit_price:.4f}"))
                pnl_item = QTableWidgetItem(f"{trade.pnl:.2f} ({trade.pnl_percent:.2f}%)")
                self.trades_table.setItem(row, 8, pnl_item)

        report = BacktestAnalyzer.generate_report(result)
        # 零成本警告：全部成本为 0 时结果不真实
        cost_model = result.stats.get('cost_model', {})
        _all_zero = all(
            float(cost_model.get(k, 0.0) or 0.0) == 0.0
            for k in ('fee_pct', 'slippage_pct', 'funding_rate_8h_pct', 'market_impact_pct')
        )
        if _all_zero and result.total_trades > 0:
            report = "⚠️  警告：手续费/滑点/资金费率均为 0，回测结果偏于乐观，请在成本设置中填入真实费率后重跑！\n\n" + report
        self.report_text.setText(report)
        self._save_live_readiness_score(result, config or {})
        self.compare_table.setRowCount(0)
        self.heatmap_table.setRowCount(0)
        self.compare_table.setSortingEnabled(False)
        self.compare_table.setSortingEnabled(True)

    def display_batch_results(self, batch_results: list, best_payload: Dict, config: Dict = None):
        """显示批量参数回测结果"""
        batch_items = batch_results or []
        best_result = best_payload.get('result') if isinstance(best_payload, dict) else None
        best_config = best_payload.get('config') if isinstance(best_payload, dict) else config
        if best_result:
            self.display_result(best_result, best_config or config)
        self.batch_results = batch_items

        self.compare_table.setRowCount(0)
        self.compare_table.setSortingEnabled(False)
        for item in batch_items:
            result = item.get('result')
            preset_name = self._preset_display_name(item.get('preset', ''))
            if not result:
                continue
            row = self.compare_table.rowCount()
            self.compare_table.insertRow(row)
            values = [
                preset_name,
                f"{item.get('composite_score', 0.0):.4f}",
                f"{result.live_readiness_score:.2f}",
                f"{result.final_capital:,.2f}",
                f"{result.total_return:.2f}%",
                f"{result.max_drawdown:.2f}%",
                f"{result.sharpe_ratio:.2f}",
                str(result.total_trades),
                f"{result.win_rate:.2f}%",
                self._validation_note(item, best_result is result),
            ]
            for col, value in enumerate(values):
                item_widget = QTableWidgetItem(value)
                if col in (1, 2, 3, 4, 5, 6, 7, 8):
                    try:
                        numeric_value = float(str(value).replace(',', '').replace('%', ''))
                        item_widget.setData(Qt.EditRole, numeric_value)
                    except Exception:
                        pass
                self.compare_table.setItem(row, col, item_widget)

        sort_rule = (best_payload or {}).get('rule', 'return')
        sort_column = {'composite': 1, 'return': 4, 'drawdown': 5, 'sharpe': 6}.get(sort_rule, 4)
        sort_order = Qt.DescendingOrder if sort_rule in ('return', 'sharpe', 'composite') else Qt.AscendingOrder
        self.compare_table.setSortingEnabled(True)
        self.compare_table.sortItems(sort_column, sort_order)
        self._populate_heatmap(batch_items)
        base_report = self.report_text.toPlainText()
        batch_summary = self._build_batch_report_summary(batch_items, best_payload or {})
        if batch_summary:
            self.report_text.setText(batch_summary + "\n\n" + base_report)
        if best_result:
            self._save_live_readiness_score(best_result, best_config or config or {})

    def _save_live_readiness_score(self, result: BacktestResult, config: Dict):
        """保存最新策略准入评分，供实时交易页人工/自动读取参考"""
        try:
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            path = os.path.join(project_root, 'src', 'strategy_live_scores.json')
            payload = {}
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    payload = json.load(f)
            key = config.get('strategy_name') or result.strategy_name
            payload[key] = {
                'strategy_name': key,
                'inst_id': result.inst_id,
                'saved_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'live_readiness_score': result.live_readiness_score,
                'return_30d': result.stats.get('return_30d', 0.0),
                'return_90d': result.stats.get('return_90d', 0.0),
                'max_drawdown': result.max_drawdown,
                'win_rate': result.win_rate,
                'profit_factor': result.profit_factor,
                'sharpe_ratio': result.sharpe_ratio,
                'max_consecutive_losses': result.max_consecutive_losses,
                'total_trades': result.total_trades,
            }
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _populate_heatmap(self, batch_items: list):
        """渲染参数热力图"""
        self.heatmap_table.setRowCount(0)
        if not batch_items:
            return

        metric_values = {
            'composite': [item.get('composite_score', 0.0) for item in batch_items],
            'return': [item['result'].total_return for item in batch_items if item.get('result')],
            'drawdown': [item['result'].max_drawdown for item in batch_items if item.get('result')],
            'sharpe': [item['result'].sharpe_ratio for item in batch_items if item.get('result')],
            'win_rate': [item['result'].win_rate for item in batch_items if item.get('result')],
        }

        for item in batch_items:
            result = item.get('result')
            if not result:
                continue
            row = self.heatmap_table.rowCount()
            self.heatmap_table.insertRow(row)

            values = [
                ("text", self._preset_display_name(item.get('preset', '')), None, False),
                ("metric", f"{item.get('composite_score', 0.0):.4f}", item.get('composite_score', 0.0), False),
                ("metric", f"{result.total_return:.2f}%", result.total_return, False),
                ("metric", f"{result.max_drawdown:.2f}%", result.max_drawdown, True),
                ("metric", f"{result.sharpe_ratio:.2f}", result.sharpe_ratio, False),
                ("metric", f"{result.win_rate:.2f}%", result.win_rate, False),
            ]

            for col, (kind, text, numeric_value, reverse) in enumerate(values):
                cell = QTableWidgetItem(text)
                if kind == "metric":
                    intensity = self._normalized_heat(
                        numeric_value,
                        metric_values[['composite', 'return', 'drawdown', 'sharpe', 'win_rate'][col - 1]],
                        reverse=reverse,
                    )
                    self._apply_heat_style(cell, intensity)
                    cell.setData(Qt.EditRole, float(numeric_value))
                else:
                    cell.setBackground(QColor(35, 35, 35))
                    cell.setForeground(QColor(240, 240, 240))
                self.heatmap_table.setItem(row, col, cell)

    def _validation_note(self, item: Dict, is_best: bool) -> str:
        notes = []
        if is_best:
            notes.append("最佳")
        validation = item.get('validation') or {}
        if validation.get('overfit_risk'):
            notes.append("过拟合风险")
        live_score = float(getattr(item.get('result'), 'live_readiness_score', 0.0) or 0.0)
        notes.append("准入通过" if live_score >= 70 else "准入不足")
        return " / ".join(notes)

    def _normalized_heat(self, value: float, values: list, reverse: bool = False) -> float:
        """将指标标准化到 0-1，用于热力图着色"""
        if not values:
            return 0.5
        low = min(values)
        high = max(values)
        if high == low:
            return 1.0
        intensity = (value - low) / (high - low)
        return 1.0 - intensity if reverse else intensity

    def _apply_heat_style(self, item: QTableWidgetItem, intensity: float):
        """根据强弱上色"""
        intensity = max(0.0, min(1.0, float(intensity)))
        red = int(90 * (1.0 - intensity) + 25)
        green = int(110 + 110 * intensity)
        blue = int(45 + 35 * (1.0 - intensity))
        item.setBackground(QColor(red, green, blue))
        item.setForeground(QColor(255, 255, 255) if intensity < 0.75 else QColor(20, 20, 20))

    def export_report(self):
        """导出报告与交易明细"""
        if not self.current_result:
            QMessageBox.warning(self, "警告", "没有可导出的回测结果")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存回测报告", "",
            "Text Files (*.txt);;All Files (*)"
        )

        if file_path:
            try:
                report = self.report_text.toPlainText().strip() or BacktestAnalyzer.generate_report(self.current_result)
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(report)

                base_path = file_path[:-4] if file_path.lower().endswith('.txt') else file_path
                trades_csv = f"{base_path}_trades.csv"
                summary_csv = f"{base_path}_summary.csv"
                self._export_trades_csv(trades_csv)
                self._export_summary_csv(summary_csv)

                batch_csv = ""
                if self.batch_results:
                    batch_csv = f"{base_path}_batch_compare.csv"
                    self._export_batch_compare_csv(batch_csv)

                QMessageBox.information(
                    self,
                    "成功",
                    f"报告已保存到:\n{file_path}\n\n交易明细:\n{trades_csv}\n\n汇总数据:\n{summary_csv}" +
                    (f"\n\n批量对比:\n{batch_csv}" if batch_csv else "")
                )
            except Exception as e:
                QMessageBox.critical(self, "错误", f"导出失败:\\n{str(e)}")

    def _export_trades_csv(self, file_path: str):
        """导出逐笔交易明细"""
        with open(file_path, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "序号", "方向", "开仓时间", "平仓时间", "开仓市场价", "开仓成交价", "平仓触发价", "平仓成交价", "数量",
                "盈亏USDT", "盈亏%", "入场原因", "出场原因"
            ])
            row_num = 1
            for trade in self.current_result.trades:
                if not trade.exit_time:
                    continue
                writer.writerow([
                    row_num,
                    "LONG" if trade.direction.name == "LONG" else "SHORT",
                    trade.entry_time.strftime("%Y-%m-%d %H:%M:%S"),
                    trade.exit_time.strftime("%Y-%m-%d %H:%M:%S"),
                    f"{getattr(trade, 'raw_entry_price', trade.entry_price):.8f}",
                    f"{trade.entry_price:.8f}",
                    f"{getattr(trade, 'raw_exit_price', trade.exit_price):.8f}",
                    f"{trade.exit_price:.8f}",
                    f"{trade.size:.8f}",
                    f"{trade.pnl:.4f}",
                    f"{trade.pnl_percent:.4f}",
                    trade.entry_reason,
                    trade.exit_reason,
                ])
                row_num += 1

    def _export_summary_csv(self, file_path: str):
        """导出可核验的汇总信息"""
        rows = [
            ("策略名称", self.current_result.strategy_name),
            ("交易对", self.current_result.inst_id),
            ("回测开始", self.current_result.start_date.strftime("%Y-%m-%d %H:%M:%S")),
            ("回测结束", self.current_result.end_date.strftime("%Y-%m-%d %H:%M:%S")),
            ("初始资金", f"{self.current_result.initial_capital:.4f}"),
            ("最终资金", f"{self.current_result.final_capital:.4f}"),
            ("总盈亏", f"{self.current_result.total_pnl:.4f}"),
            ("总收益率%", f"{self.current_result.total_return:.4f}"),
            ("年化收益率%", f"{self.current_result.annual_return:.4f}"),
            ("最大回撤%", f"{self.current_result.max_drawdown:.4f}"),
            ("夏普比率", f"{self.current_result.sharpe_ratio:.4f}"),
            ("索提诺比率", f"{self.current_result.sortino_ratio:.4f}"),
            ("总交易次数", str(self.current_result.total_trades)),
            ("盈利交易", str(self.current_result.winning_trades)),
            ("亏损交易", str(self.current_result.losing_trades)),
            ("胜率%", f"{self.current_result.win_rate:.4f}"),
            ("盈亏比", f"{self.current_result.profit_factor:.4f}"),
            ("平均盈利", f"{self.current_result.avg_win:.4f}"),
            ("平均亏损", f"{self.current_result.avg_loss:.4f}"),
            ("平均持仓小时", f"{self.current_result.avg_trade_duration:.4f}"),
            ("试仓次数", str(self.current_result.stats.get('pilot_trade_count', 0))),
            ("试仓成功率%", f"{self.current_result.stats.get('pilot_success_rate', 0.0):.4f}"),
            ("二次加仓次数", str(self.current_result.stats.get('add_on_trade_count', 0))),
            ("二次加仓成功率%", f"{self.current_result.stats.get('add_on_success_rate', 0.0):.4f}"),
            ("第一原则触发次数", str(self.current_result.stats.get('first_principle_trigger_count', 0))),
            ("驱动周期", str(self.current_result.stats.get('driver_bar', ''))),
        ]

        for key, value in (self.current_result.stats.get('bars_loaded') or {}).items():
            rows.append((f"数据根数:{key}", value))
        for key, value in (self.current_result.stats.get('data_sources') or {}).items():
            rows.append((f"数据来源:{key}", value))
        for key, value in (self.current_result.stats.get('signal_counts') or {}).items():
            rows.append((f"信号统计:{key}", value))
        for key, value in (self.current_config.get('strategy_params') or {}).items():
            rows.append((f"策略参数:{key}", value))
        if self.batch_results:
            rows.append(("批量模板对比", ""))
            for item in self.batch_results:
                result = item.get('result')
                if not result:
                    continue
                preset_name = self._preset_display_name(item.get('preset', ''))
                rows.append((f"模板:{preset_name}:综合评分", f"{item.get('composite_score', 0.0):.4f}"))
                rows.append((f"模板:{preset_name}:总收益率%", f"{result.total_return:.4f}"))
                rows.append((f"模板:{preset_name}:最大回撤%", f"{result.max_drawdown:.4f}"))
                rows.append((f"模板:{preset_name}:夏普比率", f"{result.sharpe_ratio:.4f}"))
                rows.append((f"模板:{preset_name}:实盘准入评分", f"{result.live_readiness_score:.4f}"))
                rows.append((f"模板:{preset_name}:总交易次数", str(result.total_trades)))

        with open(file_path, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["项目", "数值"])
            writer.writerows(rows)

    def _export_batch_compare_csv(self, file_path: str):
        """导出批量参数回测对比"""
        with open(file_path, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["模板", "综合评分", "最终资金", "总收益率%", "最大回撤%", "夏普比率", "总交易次数", "胜率%", "总盈亏USDT"])
            if self.current_config.get('best_template_rule') == 'composite':
                weights = self.current_config.get('best_template_weights') or {}
                writer.writerow([
                    "权重设置",
                    "",
                    f"收益={weights.get('return', 0.5):.2f}",
                    f"回撤={weights.get('drawdown', 0.3):.2f}",
                    f"夏普={weights.get('sharpe', 0.2):.2f}",
                    "", "", "", ""
                ])
            for item in self.batch_results:
                result = item.get('result')
                if not result:
                    continue
                writer.writerow([
                    self._preset_display_name(item.get('preset', '')),
                    f"{item.get('composite_score', 0.0):.6f}",
                    f"{result.final_capital:.4f}",
                    f"{result.total_return:.4f}",
                    f"{result.max_drawdown:.4f}",
                    f"{result.sharpe_ratio:.4f}",
                    str(result.total_trades),
                    f"{result.win_rate:.4f}",
                    f"{result.total_pnl:.4f}",
                ])


class BacktestWorker(QObject):
    """回测工作线程"""

    finished = Signal(object)
    progress = Signal(int, str)
    error = Signal(str)
    log = Signal(str, str)

    def __init__(self, strategy_class, config: Dict, okx_client=None, data_manager=None):
        super().__init__()
        self.strategy_class = strategy_class
        self.config = config
        self.okx_client = okx_client
        self.data_manager = data_manager
        self._stop_flag = False
        self._pause_flag = False
        self._shared_backtester: Optional[Backtester] = None
        self._result = None  # 保存结果供 on_backtest_finished 读取
        self._log_level = str((config or {}).get('backtest_log_level', 'KEY') or 'KEY').upper()

    def _should_log(self, level: str) -> bool:
        threshold_map = {
            'VERBOSE': 0,
            'INFO': 1,
            'KEY': 2,
            'QUIET': 3,
        }
        event_map = {
            'DEBUG': 0,
            'INFO': 1,
            'WARNING': 2,
            'SUCCESS': 2,
            'ERROR': 3,
        }
        threshold = threshold_map.get(self._log_level, 2)
        event_level = event_map.get(str(level or 'INFO').upper(), 1)
        return event_level >= threshold

    def _emit_log(self, message: str, level: str = "INFO"):
        if self._should_log(level):
            self.log.emit(message, level)

    def _get_backtester(self, initial_capital: float) -> Backtester:
        if self._shared_backtester is None or self._shared_backtester.initial_capital != initial_capital:
            self._shared_backtester = Backtester(
                okx_client=self.okx_client,
                initial_capital=initial_capital
            )
            if self.data_manager:
                self._shared_backtester.data_manager = self.data_manager
        return self._shared_backtester

    def run(self):
        """运行回测"""
        try:
            self._emit_log("初始化回测引擎...", "INFO")
            batch_presets = self.config.get('batch_presets') or []
            if batch_presets:
                self._result = self._run_batch_backtest(batch_presets)
            else:
                self._result = self._run_single_backtest(self.config, progress_start=0, progress_span=100)
        except Exception as e:
            self.error.emit(f"回测失败：{str(e)}")
            self._emit_log(f"错误：{str(e)}", "ERROR")
            self._result = None
        finally:
            # 始终 emit finished，确保线程能正常退出
            self.finished.emit(self._result)

    def _run_single_backtest(self, run_config: Dict, progress_start: int = 0, progress_span: int = 100, silent_progress: bool = False) -> BacktestResult:
        """运行单次回测"""
        strategy = self.strategy_class(run_config.get('strategy_params', {}))
        backtester = self._get_backtester(run_config.get('initial_capital', 10000))

        self._emit_log(f"开始回测 {run_config['inst_id']}...", "INFO")

        def scoped_progress(percent: int, status: str):
            if silent_progress:
                return
            mapped = progress_start + int(progress_span * percent / 100)
            self.progress.emit(mapped, status)

        result = backtester.run_backtest(
            strategy=strategy,
            inst_id=run_config['inst_id'],
            start_date=run_config['start_date'],
            end_date=run_config['end_date'],
            bar=run_config['bar'],
            config={
                **(run_config.get('strategy_params', {}) or {}),
                'fee_pct': run_config.get('fee_pct', 0.05),
                'slippage_pct': run_config.get('slippage_pct', 0.03),
                'funding_rate_8h_pct': run_config.get('funding_rate_8h_pct', 0.0),
                'limit_miss_probability_pct': run_config.get('limit_miss_probability_pct', 0.0),
                'market_impact_pct': run_config.get('market_impact_pct', 0.02),
                'position_size': run_config.get('position_size', 0.1),
                'inst_id': run_config.get('inst_id', ''),
            },
            progress_callback=scoped_progress,
            should_stop=lambda: self._stop_flag,
            should_pause=lambda: self._pause_flag,
        )
        return result

    def _run_single_backtest_isolated(self, run_config: Dict, progress_callback=None) -> BacktestResult:
        strategy = self.strategy_class(run_config.get('strategy_params', {}))
        backtester = Backtester(
            okx_client=self.okx_client,
            initial_capital=run_config.get('initial_capital', 10000)
        )
        if self.data_manager:
            backtester.data_manager = self.data_manager
        return backtester.run_backtest(
            strategy=strategy,
            inst_id=run_config['inst_id'],
            start_date=run_config['start_date'],
            end_date=run_config['end_date'],
            bar=run_config['bar'],
            config={
                **(run_config.get('strategy_params', {}) or {}),
                'fee_pct': run_config.get('fee_pct', 0.05),
                'slippage_pct': run_config.get('slippage_pct', 0.03),
                'funding_rate_8h_pct': run_config.get('funding_rate_8h_pct', 0.0),
                'limit_miss_probability_pct': run_config.get('limit_miss_probability_pct', 0.0),
                'market_impact_pct': run_config.get('market_impact_pct', 0.02),
                'position_size': run_config.get('position_size', 0.1),
                'inst_id': run_config.get('inst_id', ''),
            },
            progress_callback=progress_callback,
            should_stop=lambda: self._stop_flag,
            should_pause=lambda: self._pause_flag,
        )

    def _run_batch_backtest(self, batch_presets: list) -> Dict:
        """并行运行多套参数模板回测。"""
        base_config = self.config
        batch_results = []
        total = max(len(batch_presets), 1)
        best_rule = base_config.get('best_template_rule', 'return')
        weights = base_config.get('best_template_weights') or {}
        jobs = []
        for idx, preset_name in enumerate(batch_presets):
            strategy_params = dict(base_config.get('strategy_params', {}))
            strategy_params['preset'] = preset_name
            preset_configs = base_config.get('preset_configs') or {}
            strategy_params.update(preset_configs.get(preset_name, {}))

            run_config = dict(base_config)
            run_config['strategy_params'] = strategy_params
            jobs.append((idx, preset_name, run_config))

        max_workers = int(base_config.get('batch_parallel_workers', max(1, min(4, total))) or max(1, min(4, total)))
        max_workers = max(1, min(max_workers, len(jobs)))
        self._emit_log(f"批量回测并行启动 {len(jobs)} 个模板，workers={max_workers}", "INFO")
        self.progress.emit(3, f"批量回测准备中 0/{total}")
        completed = 0
        ordered_results: Dict[int, Dict] = {}
        progress_lock = threading.Lock()
        job_progress = {idx: 0 for idx, _, _ in jobs}

        def make_progress_callback(preset_name: str, idx: int):
            last_percent = {'value': -1}

            def _callback(percent: int, status: str):
                pct = max(0, min(100, int(percent or 0)))
                with progress_lock:
                    if pct > job_progress.get(idx, 0):
                        job_progress[idx] = pct
                    avg_pct = sum(job_progress.values()) / max(len(job_progress), 1)
                mapped = 5 + int(avg_pct * 0.80)
                self.progress.emit(min(85, mapped), f"{self._display_preset_name(preset_name)} | {status}")
                milestone = min(100, (pct // 25) * 25)
                if milestone != last_percent['value'] and milestone in (0, 25, 50, 75, 100):
                    last_percent['value'] = milestone
                    self._emit_log(
                        f"模板 {self._display_preset_name(preset_name)} 进度 {pct}%：{status}",
                        "INFO"
                    )

            return _callback

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="backtest-batch") as executor:
            future_map = {
                executor.submit(
                    self._run_single_backtest_isolated,
                    run_config,
                    make_progress_callback(preset_name, idx),
                ): (idx, preset_name, run_config)
                for idx, preset_name, run_config in jobs
            }
            for future in concurrent.futures.as_completed(future_map):
                if self._stop_flag:
                    break
                idx, preset_name, run_config = future_map[future]
                self._emit_log(f"模板 {self._display_preset_name(preset_name)} 已完成主回测，开始样本外验证...", "INFO")
                result = future.result()
                completed += 1
                self.progress.emit(min(85, int(85 * completed / total)), f"批量回测 {completed}/{total}")
                if (run_config.get('strategy_params') or {}).get('skip_batch_validation', False):
                    validation = {'note': '已跳过额外样本验证', 'overfit_risk': False}
                else:
                    validation = self._validate_out_of_sample(run_config)
                ordered_results[idx] = {
                    'preset': preset_name,
                    'config': run_config,
                    'result': result,
                    'validation': validation,
                }

        batch_results = [ordered_results[idx] for idx, _, _ in jobs if idx in ordered_results]

        if not batch_results:
            raise RuntimeError("批量参数回测未产生结果")

        self._attach_composite_scores(batch_results, weights)

        if best_rule == 'drawdown':
            best = min(
                batch_results,
                key=lambda item: (
                    item['result'].max_drawdown,
                    -item['result'].total_return,
                    -item['result'].final_capital,
                )
            )
        elif best_rule == 'sharpe':
            best = max(
                batch_results,
                key=lambda item: (
                    item['result'].sharpe_ratio,
                    item['result'].total_return,
                    item['result'].final_capital,
                )
            )
        elif best_rule == 'composite':
            best = max(
                batch_results,
                key=lambda item: (
                    item.get('composite_score', 0.0),
                    item['result'].total_return,
                    item['result'].final_capital,
                )
            )
        else:
            best = max(
                batch_results,
                key=lambda item: (
                    item['result'].total_return,
                    item['result'].final_capital,
                    -item['result'].max_drawdown,
                )
            )
        self.progress.emit(100, "批量回测完成")
        self._emit_log(f"批量回测完成，最佳模板: {best['preset']} (规则: {best_rule})", "SUCCESS")
        return {
            'mode': 'batch',
            'items': batch_results,
            'best': best,
            'rule': best_rule,
            'weights': weights,
        }

    def _display_preset_name(self, preset_name: str) -> str:
        preset_labels = self.config.get('preset_labels') or {}
        return str(preset_labels.get(str(preset_name), preset_name or ""))

    def _validate_out_of_sample(self, run_config: Dict) -> Dict:
        """拆分样本内/样本外/近7天/近30天验证，用于识别过拟合"""
        try:
            start_dt = datetime.strptime(run_config['start_date'], "%Y-%m-%d")
            end_dt = datetime.strptime(run_config['end_date'], "%Y-%m-%d")
            if (end_dt - start_dt).days < 45:
                return {'note': '区间过短，跳过样本外验证'}
            split_dt = start_dt + (end_dt - start_dt) * 2 / 3

            def run_window(window_start: datetime, window_end: datetime) -> BacktestResult:
                cfg = dict(run_config)
                cfg['start_date'] = window_start.strftime("%Y-%m-%d")
                cfg['end_date'] = window_end.strftime("%Y-%m-%d")
                return self._run_single_backtest(cfg, progress_start=0, progress_span=1, silent_progress=True)

            self._emit_log("样本外验证 (1/4): 样本内回测...", "INFO")
            in_sample = run_window(start_dt, split_dt)
            self._emit_log("样本外验证 (2/4): 样本外回测...", "INFO")
            out_sample = run_window(split_dt + timedelta(days=1), end_dt)
            self._emit_log("样本外验证 (3/4): 近7天回测...", "INFO")
            recent_7 = run_window(max(start_dt, end_dt - timedelta(days=7)), end_dt)
            self._emit_log("样本外验证 (4/4): 近30天回测...", "INFO")
            recent_30 = run_window(max(start_dt, end_dt - timedelta(days=30)), end_dt)
            overfit = (
                in_sample.total_return > 8.0
                and (
                    out_sample.total_return < 0
                    or out_sample.live_readiness_score < in_sample.live_readiness_score * 0.55
                    or out_sample.max_drawdown > in_sample.max_drawdown * 1.8
                )
            )
            return {
                'in_sample_return': in_sample.total_return,
                'out_sample_return': out_sample.total_return,
                'recent_7_return': recent_7.total_return,
                'recent_30_return': recent_30.total_return,
                'in_sample_score': in_sample.live_readiness_score,
                'out_sample_score': out_sample.live_readiness_score,
                'overfit_risk': overfit,
            }
        except Exception as exc:
            return {'note': f'样本外验证失败: {exc}', 'overfit_risk': False}

    def _attach_composite_scores(self, batch_results: list, weights: Dict[str, float]):
        """按收益/回撤/夏普权重计算综合评分"""
        if not batch_results:
            return

        ret_values = [item['result'].total_return for item in batch_results]
        dd_values = [item['result'].max_drawdown for item in batch_results]
        sharpe_values = [item['result'].sharpe_ratio for item in batch_results]

        def normalize(value: float, values: list, reverse: bool = False) -> float:
            v_min = min(values)
            v_max = max(values)
            if v_max == v_min:
                return 1.0
            score = (value - v_min) / (v_max - v_min)
            return 1.0 - score if reverse else score

        return_weight = float(weights.get('return', 0.5))
        drawdown_weight = float(weights.get('drawdown', 0.3))
        sharpe_weight = float(weights.get('sharpe', 0.2))
        weight_sum = return_weight + drawdown_weight + sharpe_weight
        if weight_sum <= 0:
            return_weight, drawdown_weight, sharpe_weight = 0.5, 0.3, 0.2
            weight_sum = 1.0

        for item in batch_results:
            result = item['result']
            ret_score = normalize(result.total_return, ret_values)
            dd_score = normalize(result.max_drawdown, dd_values, reverse=True)
            sharpe_score = normalize(result.sharpe_ratio, sharpe_values)
            composite = (
                ret_score * return_weight +
                dd_score * drawdown_weight +
                sharpe_score * sharpe_weight
            ) / weight_sum
            item['composite_score'] = composite

    def stop(self):
        """停止回测"""
        self._stop_flag = True

    def pause(self, paused: bool):
        """暂停/继续回测"""
        self._pause_flag = paused


class BacktestPage(QWidget):
    """回测页面"""

    def __init__(self, okx_client=None, data_manager=None, trade_settings_manager=None, open_trade_settings_callback=None):
        super().__init__()
        self.okx_client = okx_client
        self.data_manager = data_manager
        self.trade_settings_manager = trade_settings_manager
        self.open_trade_settings_callback = open_trade_settings_callback
        import os
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        strategies_dir = os.path.join(project_root, 'strategies')
        self.strategy_loader = StrategyLoader(strategies_dir)
        self.backtest_worker: Optional[BacktestWorker] = None
        self.backtest_thread: Optional[QThread] = None
        self.current_config: Optional[Dict] = None
        self.is_paused = False
        self.init_ui()

    def init_ui(self):
        """初始化 UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(15)

        splitter = QSplitter(Qt.Horizontal)

        self.config_widget = BacktestConfigWidget(
            self.strategy_loader,
            self.okx_client,
            data_manager=self.data_manager,
            trade_settings_manager=self.trade_settings_manager,
            open_trade_settings_callback=self.open_trade_settings_callback,
        )
        splitter.addWidget(self.config_widget)

        self.result_widget = BacktestResultWidget()
        splitter.addWidget(self.result_widget)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)

        layout.addWidget(splitter)

        # 日志
        log_group = QGroupBox("回测日志")
        log_layout = QVBoxLayout(log_group)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(150)
        self.log_text.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #00ff00;
                font-family: 'Menlo', 'Monaco';
                font-size: 11px;
                border: 1px solid #333;
            }
        """)
        log_layout.addWidget(self.log_text)
        layout.addWidget(log_group)

        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        # 连接信号
        self.config_widget.start_backtest.connect(self.start_backtest)
        self.config_widget.cancel_backtest.connect(self.stop_backtest)
        self.config_widget.pause_backtest.connect(self.toggle_pause_backtest)

    def refresh_trade_settings_summary(self):
        if hasattr(self, "config_widget") and self.config_widget:
            self.config_widget.refresh_shared_trade_settings()

    def log(self, message: str, level: str = "INFO"):
        """添加日志"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        color = {
            "INFO": "#00ff00",
            "WARNING": "#ffaa00",
            "ERROR": "#ff4444",
            "SUCCESS": "#00ffaa"
        }.get(level, "#ffffff")
        self.log_text.append(f'<span style="color:#666666">[{timestamp}]</span> <span style="color:{color}">[{level}]</span> {message}')
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())

    def start_backtest(self, config: Dict):
        """开始回测"""
        if not config.get('strategy_name'):
            QMessageBox.warning(self, "警告", "请选择策略")
            return

        self.log(f"开始回测：{config['strategy_name']} @ {config['inst_id']}", "INFO")
        self.current_config = config
        self.config_widget.set_running(True)
        self.progress_bar.setValue(0)

        module = self.strategy_loader.load_strategy(config['strategy_name'])
        if not module:
            self.log("策略加载失败", "ERROR")
            self.config_widget.set_running(False)
            return

        strategy_class = self.strategy_loader.get_strategy_class(config['strategy_name'])
        if not strategy_class:
            self.log("未找到策略类", "ERROR")
            self.config_widget.set_running(False)
            return

        self.backtest_thread = QThread()
        self.backtest_worker = BacktestWorker(
            strategy_class=strategy_class,
            config=config,
            okx_client=self.okx_client,
            data_manager=self.data_manager
        )
        self.backtest_worker.moveToThread(self.backtest_thread)

        self.backtest_thread.started.connect(self.backtest_worker.run)
        # worker.finished 是自定义 Signal，从 run() 内部 emit，用于通知线程退出事件循环
        self.backtest_worker.finished.connect(self.backtest_thread.quit)
        # ⚠️ on_backtest_finished / deleteLater 必须挂在 thread.finished（OS 线程完全退出后）
        # 而不是 worker.finished（那时 run() 尚未 return，线程仍运行）→ 否则 SIGABRT
        self.backtest_thread.finished.connect(self.on_backtest_finished)
        self.backtest_thread.finished.connect(self.backtest_worker.deleteLater)
        self.backtest_thread.finished.connect(self.backtest_thread.deleteLater)
        self.backtest_worker.progress.connect(self.on_backtest_progress)
        self.backtest_worker.error.connect(self.on_backtest_error)
        self.backtest_worker.log.connect(self.log)

        self.backtest_thread.start()

    def stop_backtest(self):
        """停止回测"""
        if self.backtest_worker:
            self.backtest_worker.stop()
            self.log("正在停止回测...", "WARNING")
            self.is_paused = False
            self.config_widget.set_paused(False)
            self.config_widget.set_running(False)

    def toggle_pause_backtest(self):
        """暂停/继续回测"""
        if not self.backtest_worker:
            return
        self.is_paused = not self.is_paused
        self.backtest_worker.pause(self.is_paused)
        self.config_widget.set_paused(self.is_paused)
        self.log("回测已" + ("暂停" if self.is_paused else "继续"), "WARNING")

    def on_backtest_finished(self):
        """回测完成（由 QThread.finished 触发，无参数；从 worker._result 取结果）"""
        result_payload = getattr(self.backtest_worker, '_result', None) if self.backtest_worker else None
        if result_payload is None:
            self.log("回测结束（无有效结果）", "WARNING")
        elif isinstance(result_payload, dict) and result_payload.get('mode') == 'batch':
            self.log("回测完成", "SUCCESS")
            best = result_payload.get('best') or {}
            best_preset = best.get('preset', '')
            if best_preset:
                self.log(f"批量参数回测最佳模板：{best_preset}", "SUCCESS")
            self.result_widget.display_batch_results(
                result_payload.get('items', []),
                best,
                self.current_config or {}
            )
        else:
            self.log("回测完成", "SUCCESS")
            self.result_widget.display_result(result_payload, self.current_config or {})
        self.config_widget.set_running(False)
        self.config_widget.set_paused(False)
        self.is_paused = False
        self.progress_bar.setValue(100)

        if self.backtest_thread:
            self.backtest_thread.quit()
            self.backtest_thread.wait(3000)

    def on_backtest_progress(self, percent: int, status: str):
        """更新进度"""
        self.progress_bar.setValue(percent)
        self.log(f"进度：{status}", "INFO")

    def on_backtest_error(self, error: str):
        """回测错误"""
        self.log(error, "ERROR")
        self.config_widget.set_running(False)
        self.config_widget.set_paused(False)
        QMessageBox.critical(self, "错误", error)
