"""
量化交易主界面
支持自主加载策略、配置参数、执行交易
"""

import sys
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTextEdit, QTableWidget, QTableWidgetItem,
    QComboBox, QGroupBox, QFormLayout, QSpinBox, QDoubleSpinBox,
    QCheckBox, QFileDialog, QMessageBox, QProgressBar, QSplitter,
    QTabWidget, QScrollArea, QFrame, QHeaderView, QDialog,
    QListWidget, QListWidgetItem, QTextBrowser, QGridLayout
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject, QThread, QPoint
from PyQt5.QtGui import QFont, QColor, QMouseEvent

from src.api.okx_client import OKXClient
from src.strategy.loader import StrategyLoader, StrategyInfo, StrategyType
from src.trading.executor import TradeExecutor, PositionSide, OrderType
from src.strategy.runner import StrategyRunner, SimpleStrategyRunner
from src.ui.backtest_page import BacktestPage


class StrategyConfigWidget(QWidget):
    """策略配置组件"""

    def __init__(self, strategy_info: StrategyInfo = None):
        super().__init__()
        self.strategy_info = strategy_info
        self.config_inputs: Dict[str, QWidget] = {}
        self.init_ui()

    def init_ui(self):
        """初始化 UI"""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        if not self.strategy_info:
            label = QLabel("请先选择策略")
            label.setAlignment(Qt.AlignCenter)
            layout.addWidget(label)
            return

        # 策略信息
        info_frame = QFrame()
        info_frame.setStyleSheet("""
            QFrame {
                background-color: #f5f5f7;
                border: 1px solid #e5e5ea;
                border-radius: 8px;
                padding: 10px;
            }
        """)
        info_layout = QVBoxLayout(info_frame)

        name_label = QLabel(f"<b>策略名称:</b> {self.strategy_info.name}")
        info_layout.addWidget(name_label)

        if self.strategy_info.description:
            desc_label = QLabel(f"<b>说明:</b> {self.strategy_info.description}")
            desc_label.setWordWrap(True)
            info_layout.addWidget(desc_label)

        if self.strategy_info.author:
            author_label = QLabel(f"<b>作者:</b> {self.strategy_info.author}")
            info_layout.addWidget(author_label)

        if self.strategy_info.version:
            version_label = QLabel(f"<b>版本:</b> {self.strategy_info.version}")
            info_layout.addWidget(version_label)

        layout.addWidget(info_frame)

        # 配置参数
        if self.strategy_info.config_schema:
            config_group = QGroupBox("策略参数配置")
            config_layout = QFormLayout()
            config_layout.setSpacing(8)

            for param_name, param_info in self.strategy_info.config_schema.items():
                label = param_info.get('label', param_name)
                param_type = param_info.get('type', 'float')
                default_value = param_info.get('default', 0)

                if param_type == 'int':
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

                config_layout.addRow(f"{label}:", input_widget)
                self.config_inputs[param_name] = input_widget

            config_group.setLayout(config_layout)
            layout.addWidget(config_group)

        layout.addStretch()

    def get_config(self) -> Dict:
        """获取配置值"""
        config = {}
        for param_name, input_widget in self.config_inputs.items():
            if isinstance(input_widget, QCheckBox):
                config[param_name] = input_widget.isChecked()
            elif isinstance(input_widget, QSpinBox):
                config[param_name] = input_widget.value()
            elif isinstance(input_widget, QDoubleSpinBox):
                config[param_name] = input_widget.value()
            else:
                config[param_name] = input_widget
        return config


class TradeLogWidget(QTextEdit):
    """交易日志组件"""

    def __init__(self):
        super().__init__()
        self.setReadOnly(True)
        self.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #00ff00;
                font-family: 'Courier New', monospace;
                font-size: 12px;
                border: 1px solid #333;
            }
        """)

    def log(self, message: str, level: str = "INFO"):
        """添加日志"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        color = {
            "INFO": "#00ff00",
            "WARNING": "#ffaa00",
            "ERROR": "#ff4444",
            "SUCCESS": "#00ffaa",
            "TRADE": "#00ccff"
        }.get(level, "#ffffff")

        self.append(f'<span style="color:#666666">[{timestamp}]</span> <span style="color:{color}">[{level}]</span> {message}')
        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())

    def clear_log(self):
        """清空日志"""
        self.clear()


class PositionMonitorWidget(QWidget):
    """持仓监控组件（带滚动条）"""

    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        """初始化 UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # 创建滚动区域
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setStyleSheet("""
            QScrollArea {
                border: 1px solid #333;
                border-radius: 4px;
                background-color: transparent;
            }
            QScrollBar:vertical {
                background-color: #2a2a2a;
                width: 12px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background-color: #555;
                border-radius: 6px;
                min-height: 30px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #777;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
            QScrollBar:horizontal {
                background-color: #2a2a2a;
                height: 12px;
                border-radius: 6px;
            }
            QScrollBar::handle:horizontal {
                background-color: #555;
                border-radius: 6px;
                min-width: 30px;
            }
            QScrollBar::handle:horizontal:hover {
                background-color: #777;
            }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
                width: 0px;
            }
        """)

        # 持仓表格
        self.position_table = QTableWidget()
        self.position_table.setColumnCount(8)
        self.position_table.setHorizontalHeaderLabels([
            "交易对", "方向", "数量", "开仓价", "当前价", "未实现盈亏", "盈亏率%", "操作"
        ])
        self.position_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.position_table.setSelectionBehavior(QTableWidget.SelectRows)

        scroll.setWidget(self.position_table)
        layout.addWidget(scroll)

    def update_positions(self, positions: Dict):
        """更新持仓显示"""
        self.position_table.setRowCount(0)

        for inst_id, pos_info in positions.items():
            row = self.position_table.rowCount()
            self.position_table.insertRow(row)

            # 方向显示
            side_text = "做多" if pos_info.side == PositionSide.LONG else "做空"
            side_color = "#00ff00" if pos_info.side == PositionSide.LONG else "#ff4444"

            # 盈亏颜色
            pnl_color = "#00ff00" if pos_info.unrealized_pnl >= 0 else "#ff4444"

            self.position_table.setItem(row, 0, QTableWidgetItem(inst_id))
            self.position_table.setItem(row, 1, QTableWidgetItem(side_text))
            self.position_table.setItem(row, 2, QTableWidgetItem(f"{pos_info.size:.6f}"))
            self.position_table.setItem(row, 3, QTableWidgetItem(f"{pos_info.entry_price:.4f}"))
            self.position_table.setItem(row, 4, QTableWidgetItem(f"{pos_info.current_price:.4f}"))

            pnl_item = QTableWidgetItem(f"{pos_info.unrealized_pnl:.2f} USDT")
            pnl_item.setForeground(QColor(pnl_color))
            self.position_table.setItem(row, 5, pnl_item)

            pnl_pct_item = QTableWidgetItem(f"{pos_info.pnl_percent:.2f}%")
            pnl_pct_item.setForeground(QColor(pnl_color))
            self.position_table.setItem(row, 6, pnl_pct_item)

            # 平仓按钮
            close_btn = QPushButton("平仓")
            close_btn.setStyleSheet("""
                QPushButton {
                    background-color: #ff4444;
                    color: white;
                    border-radius: 4px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #ff6666;
                }
            """)
            close_btn.clicked.connect(lambda _, i=inst_id: self.on_close_position.emit(i))
            self.position_table.setCellWidget(row, 7, close_btn)

    # 自定义信号
    on_close_position = pyqtSignal(str)


class QuantTradeWindow(QMainWindow):
    """量化交易主窗口"""

    def __init__(self, okx_client: OKXClient = None):
        super().__init__()
        self.okx_client = okx_client
        self.trade_executor = None
        self.strategy_loader = StrategyLoader()
        self.current_strategy = None
        self.current_strategy_module = None
        self.strategy_config = {}
        self.is_running = False

        self.init_ui()
        self.init_components()

    def init_ui(self):
        """初始化 UI"""
        # 设置窗口标志 - 确保窗口可以正常拖动
        self.setWindowFlags(
            Qt.Window |
            Qt.WindowTitleHint |
            Qt.WindowMinimizeButtonHint |
            Qt.WindowMaximizeButtonHint |
            Qt.WindowCloseButtonHint
        )
        
        self.setWindowTitle("Crypto Trader - 量化交易系统")
        self.setGeometry(100, 100, 1400, 900)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
            }
            QLabel {
                color: #ffffff;
            }
            QGroupBox {
                color: #ffffff;
                font-weight: bold;
                border: 1px solid #333;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: #00ccff;
            }
            QTabWidget::pane {
                border: 1px solid #333;
                border-radius: 8px;
                background-color: #1e1e1e;
            }
            QTabWidget::tab-bar {
                alignment: left;
            }
            QTabBar::tab {
                background-color: #2a2a2a;
                color: #ffffff;
                padding: 10px 20px;
                border: 1px solid #333;
                border-bottom: none;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: #1e1e1e;
                border-bottom: none;
            }
            QTabBar::tab:hover {
                background-color: #333;
            }
        """)

        # 主容器
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # 顶部状态栏
        self.create_status_bar(main_layout)

        # 标签页容器
        self.main_tabs = QTabWidget()

        # 实时交易页面
        trade_widget = self.create_trade_page()
        self.main_tabs.addTab(trade_widget, "实时交易")

        # 回测页面
        self.backtest_page = BacktestPage(self.okx_client)
        self.main_tabs.addTab(self.backtest_page, "策略回测")

        # 扫描页面
        from src.ui.scanner_page import ScannerPage
        self.scanner_page = ScannerPage(self.okx_client)
        self.main_tabs.addTab(self.scanner_page, "交易对扫描")

        main_layout.addWidget(self.main_tabs)

    def create_status_bar(self, layout):
        """创建状态栏"""
        status_frame = QFrame()
        status_frame.setFixedHeight(45)  # 限制高度
        status_frame.setStyleSheet("""
            QFrame {
                background-color: #2a2a2a;
                border: 1px solid #333;
                border-radius: 4px;
                padding: 2px;
            }
        """)
        status_layout = QHBoxLayout(status_frame)
        status_layout.setContentsMargins(10, 2, 10, 2)
        status_layout.setSpacing(10)

        # 连接状态
        self.connection_label = QLabel("● 未连接")
        self.connection_label.setStyleSheet("color: #ff4444; font-weight: bold; font-size: 11px;")
        status_layout.addWidget(self.connection_label)

        # 账户余额标签
        balance_label = QLabel("余额:")
        balance_label.setStyleSheet("color: #aaa; font-size: 11px;")
        status_layout.addWidget(balance_label)

        # 账户余额数值
        self.balance_label = QLabel("0.00 USDT")
        self.balance_label.setStyleSheet("color: #00ffaa; font-weight: bold; font-size: 11px;")
        status_layout.addWidget(self.balance_label)

        # 刷新按钮
        refresh_btn = QPushButton("刷新")
        refresh_btn.setFixedHeight(24)
        refresh_btn.setStyleSheet("""
            QPushButton {
                background-color: #00ccff;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 2px 8px;
                font-size: 10px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #00aadd;
            }
        """)
        refresh_btn.clicked.connect(self.refresh_data)
        status_layout.addWidget(refresh_btn)

        status_layout.addStretch()

        # 运行状态
        self.status_label = QLabel("就绪")
        self.status_label.setStyleSheet("color: #00ccff; font-size: 11px;")
        status_layout.addWidget(self.status_label)

        layout.addWidget(status_frame)

    def create_strategy_panel(self) -> QWidget:
        """创建策略管理面板（带滚动条）"""
        # 创建滚动区域
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: transparent;
            }
            QScrollBar:vertical {
                background-color: #2a2a2a;
                width: 12px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background-color: #555;
                border-radius: 6px;
                min-height: 30px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #777;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)

        # 面板容器
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        # 策略列表
        strategy_group = QGroupBox("策略管理")
        strategy_layout = QVBoxLayout(strategy_group)

        # 策略列表框
        self.strategy_list = QListWidget()
        self.strategy_list.itemClicked.connect(self.on_strategy_selected)
        strategy_layout.addWidget(self.strategy_list)

        # 策略操作按钮
        btn_layout = QHBoxLayout()

        refresh_btn = QPushButton("刷新策略")
        refresh_btn.clicked.connect(self.refresh_strategies)
        btn_layout.addWidget(refresh_btn)

        load_btn = QPushButton("加载策略文件")
        load_btn.clicked.connect(self.load_custom_strategy)
        btn_layout.addWidget(load_btn)

        strategy_layout.addLayout(btn_layout)
        layout.addWidget(strategy_group)

        # 策略配置
        self.config_scroll = QScrollArea()
        self.config_scroll.setWidgetResizable(True)
        self.config_scroll.setStyleSheet("""
            QScrollArea {
                border: 1px solid #333;
                border-radius: 8px;
                background-color: #2a2a2a;
            }
        """)
        self.config_widget = QWidget()
        self.config_scroll.setWidget(self.config_widget)
        self.config_layout_inner = QVBoxLayout(self.config_widget)

        config_group = QGroupBox("策略配置")
        config_group.setLayout(self.config_layout_inner)
        layout.addWidget(config_group)

        # 交易对选择
        pair_group = QGroupBox("交易对配置")
        pair_layout = QFormLayout()

        self.pair_combo = QComboBox()
        self.pair_combo.setEditable(True)
        self.pair_combo.addItems([
            "BTC-USDT", "ETH-USDT", "SOL-USDT", "XRP-USDT",
            "ADA-USDT", "AVAX-USDT", "DOGE-USDT", "DOT-USDT"
        ])
        pair_layout.addRow("交易对:", self.pair_combo)

        self.leverage_spin = QSpinBox()
        self.leverage_spin.setRange(1, 100)
        self.leverage_spin.setValue(1)
        self.leverage_spin.setSuffix("x")
        pair_layout.addRow("杠杆倍数:", self.leverage_spin)

        pair_group.setLayout(pair_layout)
        layout.addWidget(pair_group)

        # 控制按钮
        control_group = QGroupBox("交易控制")
        control_layout = QVBoxLayout(control_group)

        self.start_btn = QPushButton("启动策略")
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
        self.start_btn.clicked.connect(self.toggle_strategy)
        control_layout.addWidget(self.start_btn)

        self.stop_btn = QPushButton("停止策略")
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
        self.stop_btn.clicked.connect(self.stop_strategy)
        self.stop_btn.setEnabled(False)
        control_layout.addWidget(self.stop_btn)

        layout.addWidget(control_group)
        layout.addStretch()

        # 将面板设置为滚动区域的内容
        scroll.setWidget(panel)
        return scroll

    def create_trade_page(self) -> QWidget:
        """创建实时交易页面"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(15)

        # 分割器
        splitter = QSplitter(Qt.Horizontal)

        # 左侧：策略管理
        left_panel = self.create_strategy_panel()
        splitter.addWidget(left_panel)

        # 右侧：交易执行和监控
        right_panel = self.create_trade_panel()
        splitter.addWidget(right_panel)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)

        layout.addWidget(splitter)

        # 底部日志
        log_group = QGroupBox("交易日志")
        log_layout = QVBoxLayout(log_group)
        self.trade_log = TradeLogWidget()
        log_layout.addWidget(self.trade_log)
        layout.addWidget(log_group, 1)

        return widget

    def create_trade_panel(self) -> QWidget:
        """创建交易面板（带滚动条）"""
        # 创建滚动区域
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: transparent;
            }
            QScrollBar:vertical {
                background-color: #2a2a2a;
                width: 12px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background-color: #555;
                border-radius: 6px;
                min-height: 30px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #777;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)

        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        # 持仓监控
        position_group = QGroupBox("持仓监控")
        position_layout = QVBoxLayout(position_group)
        self.position_monitor = PositionMonitorWidget()
        self.position_monitor.on_close_position.connect(self.close_position)
        position_layout.addWidget(self.position_monitor)
        layout.addWidget(position_group)

        # 手动交易
        manual_group = QGroupBox("手动交易")
        manual_layout = QFormLayout(manual_group)

        # 交易方向
        trade_layout = QHBoxLayout()
        self.buy_btn = QPushButton("买入/做多")
        self.buy_btn.setStyleSheet("""
            QPushButton {
                background-color: #00aa00;
                color: white;
                border-radius: 4px;
                font-weight: bold;
                padding: 10px;
            }
        """)
        self.buy_btn.clicked.connect(lambda: self.manual_trade("buy"))
        trade_layout.addWidget(self.buy_btn)

        self.sell_btn = QPushButton("卖出/平仓")
        self.sell_btn.setStyleSheet("""
            QPushButton {
                background-color: #ff4444;
                color: white;
                border-radius: 4px;
                font-weight: bold;
                padding: 10px;
            }
        """)
        self.sell_btn.clicked.connect(lambda: self.manual_trade("sell"))
        trade_layout.addWidget(self.sell_btn)

        manual_layout.addRow(trade_layout)

        # 仓位比例
        self.position_size_spin = QDoubleSpinBox()
        self.position_size_spin.setRange(0.01, 1.0)
        self.position_size_spin.setValue(0.1)
        self.position_size_spin.setSuffix(" (10%)")
        self.position_size_spin.setDecimals(2)
        manual_layout.addRow("仓位比例:", self.position_size_spin)

        manual_group.setLayout(manual_layout)
        layout.addWidget(manual_group)
        layout.addStretch()

        # 将面板设置为滚动区域的内容
        scroll.setWidget(panel)
        return scroll

    def init_components(self):
        """初始化组件"""
        # 发现策略
        self.refresh_strategies()

        # 连接 OKX
        if self.okx_client:
            self.connection_label.setText("● 已连接")
            self.connection_label.setStyleSheet("color: #00ff00;")
            self.trade_executor = TradeExecutor(self.okx_client)
            # 初始刷新一次
            self.refresh_balance()
            self.refresh_positions()

    def refresh_strategies(self):
        """刷新策略列表"""
        self.strategy_list.clear()
        strategies = self.strategy_loader.discover_strategies()

        for strategy_info in strategies:
            item_text = f"{strategy_info.name} ({strategy_info.type.value})"
            if strategy_info.description:
                item_text += f" - {strategy_info.description[:30]}"

            list_item = QListWidgetItem(item_text)
            list_item.setData(Qt.UserRole, strategy_info)
            self.strategy_list.addItem(list_item)

        self.trade_log.log(f"已发现 {len(strategies)} 个策略", "INFO")

    def load_custom_strategy(self):
        """加载自定义策略文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择策略文件", "",
            "Python Files (*.py);;All Files (*)"
        )

        if file_path:
            strategy_info = self.strategy_loader.load_custom_strategy(file_path)
            if strategy_info:
                self.refresh_strategies()
                self.trade_log.log(f"已加载策略：{strategy_info.name}", "SUCCESS")
            else:
                self.trade_log.log(f"加载策略失败：{file_path}", "ERROR")

    def on_strategy_selected(self, item):
        """策略选择事件"""
        strategy_info = item.data(Qt.UserRole)
        if not strategy_info:
            return

        self.current_strategy = strategy_info
        self.trade_log.log(f"选择策略：{strategy_info.name}", "INFO")

        # 更新配置面板
        while self.config_layout_inner.count():
            child = self.config_layout_inner.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        config_widget = StrategyConfigWidget(strategy_info)
        self.config_layout_inner.addWidget(config_widget)

    def toggle_strategy(self):
        """启动/停止策略"""
        if self.is_running:
            self.stop_strategy()
        else:
            self.start_strategy()

    def start_strategy(self):
        """启动策略"""
        if not self.current_strategy:
            QMessageBox.warning(self, "警告", "请先选择策略")
            return

        # 获取配置
        config_widget = self.config_scroll.findChild(StrategyConfigWidget)
        if config_widget:
            self.strategy_config = config_widget.get_config()

        inst_id = self.pair_combo.currentText()
        leverage = self.leverage_spin.value()

        self.trade_log.log(f"启动策略：{self.current_strategy.name}", "TRADE")
        self.trade_log.log(f"交易对：{inst_id}, 杠杆：{leverage}x", "INFO")
        self.trade_log.log(f"配置：{self.strategy_config}", "INFO")

        # 加载策略模块
        self.current_strategy_module = self.strategy_loader.load_strategy(self.current_strategy.name)
        if not self.current_strategy_module:
            self.trade_log.log("策略加载失败", "ERROR")
            return

        # 尝试获取策略类并初始化
        strategy_class = self.strategy_loader.get_strategy_class(self.current_strategy.name)
        if strategy_class:
            try:
                self.strategy_instance = strategy_class(self.strategy_config)
                self.trade_log.log(f"策略实例化成功", "SUCCESS")
            except Exception as e:
                self.trade_log.log(f"策略实例化失败：{e}", "ERROR")
                return
        else:
            # 使用模块作为策略
            self.strategy_instance = self.current_strategy_module
            self.trade_log.log(f"使用策略模块", "INFO")

        # 更新 UI 状态
        self.is_running = True
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_label.setText("运行中")

        # 启动策略执行线程
        self.strategy_thread = QThread()
        self.strategy_worker = StrategyRunner(
            self.strategy_instance,
            inst_id,
            self.okx_client,
            self.trade_executor,
            self.strategy_config
        )
        self.strategy_worker.moveToThread(self.strategy_thread)

        self.strategy_thread.started.connect(self.strategy_worker.run)
        self.strategy_worker.finished.connect(self.on_strategy_finished)
        self.strategy_worker.log_signal.connect(self.on_strategy_log)
        self.strategy_worker.trade_signal.connect(self.on_strategy_trade)

        self.strategy_thread.start()

    def stop_strategy(self):
        """停止策略"""
        self.is_running = False
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText("已停止")

        if hasattr(self, 'strategy_thread'):
            self.strategy_thread.quit()
            self.strategy_thread.wait(3000)

        self.trade_log.log("策略已停止", "WARNING")

    def on_strategy_finished(self):
        """策略执行完成"""
        self.stop_strategy()
        self.trade_log.log("策略执行完成", "INFO")

    def on_strategy_log(self, message: str, level: str):
        """接收策略日志"""
        self.trade_log.log(message, level)

    def on_strategy_trade(self, action: str, inst_id: str, price: float, size: float):
        """接收交易信号"""
        self.trade_log.log(f"交易信号：{action} {inst_id} @ {price}", "TRADE")

    def refresh_data(self):
        """刷新数据"""
        self.refresh_balance()
        self.refresh_positions()

    def refresh_balance(self):
        """刷新余额"""
        if self.trade_executor:
            balance = self.trade_executor.get_usdt_balance()
            self.balance_label.setText(f"{balance:.2f} USDT")

    def refresh_positions(self):
        """刷新持仓"""
        if self.trade_executor:
            positions = self.trade_executor.get_positions()
            self.position_monitor.update_positions(positions)

    def close_position(self, inst_id: str):
        """平仓"""
        if not self.trade_executor:
            return

        reply = QMessageBox.question(
            self, "确认平仓",
            f"确认平仓 {inst_id} 吗？",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            result = self.trade_executor.execute_stop_loss(inst_id)
            if result.success:
                self.trade_log.log(f"平仓成功：{inst_id}", "SUCCESS")
            else:
                self.trade_log.log(f"平仓失败：{result.message}", "ERROR")

    def manual_trade(self, action: str):
        """手动交易"""
        if not self.trade_executor:
            self.trade_log.log("请先连接 OKX", "ERROR")
            return

        inst_id = self.pair_combo.currentText()
        position_ratio = self.position_size_spin.value()

        if action == "buy":
            result = self.trade_executor.execute_buy(inst_id, position_ratio=position_ratio)
            if result.success:
                self.trade_log.log(f"买入成功：{inst_id}, 数量={result.filled_size}", "SUCCESS")
            else:
                self.trade_log.log(f"买入失败：{result.message}", "ERROR")
        else:
            positions = self.trade_executor.get_positions(inst_id)
            if inst_id in positions:
                result = self.trade_executor.execute_sell(inst_id, positions[inst_id].size)
                if result.success:
                    self.trade_log.log(f"卖出成功：{inst_id}, 数量={result.filled_size}", "SUCCESS")
                else:
                    self.trade_log.log(f"卖出失败：{result.message}", "ERROR")
            else:
                self.trade_log.log("无持仓", "WARNING")


# StrategyWorker 已替换为 StrategyRunner


def main():
    """主函数"""
    app = QApplication(sys.argv)

    # 设置全局样式
    app.setStyle('Fusion')

    # 初始化 OKX 客户端
    okx_client = OKXClient(
        api_key="ddafb223-6fe7-4ada-94f6-a31d58b23e1a",
        secret_key="C05E005B0B94EB17E44739C7302605C9",
        passphrase="!Lqs4381525",
        testnet=True,
        proxy_url="http://127.0.0.1:7897"
    )

    # 创建主窗口
    window = QuantTradeWindow(okx_client)
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
