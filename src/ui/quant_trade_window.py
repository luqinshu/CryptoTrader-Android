"""
量化交易主界面
支持自主加载策略、配置参数、执行交易
"""

import json
import os
import sys
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional

try:
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QPushButton, QLabel, QTextEdit, QTableWidget, QTableWidgetItem,
        QComboBox, QGroupBox, QFormLayout, QSpinBox, QDoubleSpinBox,
        QCheckBox, QFileDialog, QMessageBox, QProgressBar, QSplitter,
        QTabWidget, QScrollArea, QFrame, QHeaderView, QDialog, QInputDialog,
        QListWidget, QListWidgetItem, QTextBrowser, QGridLayout,
        QButtonGroup, QRadioButton
    )
    from PySide6.QtCore import Qt, QTimer, Signal as pyqtSignal, QObject, QThread, QPoint
    from PySide6.QtGui import QFont, QColor, QMouseEvent
except ImportError:
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QPushButton, QLabel, QTextEdit, QTableWidget, QTableWidgetItem,
        QComboBox, QGroupBox, QFormLayout, QSpinBox, QDoubleSpinBox,
        QCheckBox, QFileDialog, QMessageBox, QProgressBar, QSplitter,
        QTabWidget, QScrollArea, QFrame, QHeaderView, QDialog, QInputDialog,
        QListWidget, QListWidgetItem, QTextBrowser, QGridLayout,
        QButtonGroup, QRadioButton
    )
    from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject, QThread, QPoint
    from PyQt5.QtGui import QFont, QColor, QMouseEvent

from src.api.okx_client import OKXClient
from src.strategy.loader import StrategyLoader, StrategyInfo, StrategyType
from src.trading.executor import TradeExecutor, PositionSide, OrderType
from src.trading.shared_trade_settings import SharedTradeSettingsManager
from src.strategy.runner import StrategyRunner, SimpleStrategyRunner
from src.strategy.paper_runner import PaperStrategyRunner
from src.trading.paper_engine import PaperTradeEngine
from src.trading.scan_auto_trader import ScanDrivenAutoTrader
from src.ui.backtest_page import BacktestPage
from src.ui.trade_parameter_page import TradeParameterPage
from src.data.manager import DataManager


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
                elif param_type == 'select':
                    input_widget = QComboBox()
                    options = param_info.get('options', [])
                    for opt in options:
                        if isinstance(opt, dict):
                            input_widget.addItem(opt.get('label', opt.get('value', '')), opt.get('value', ''))
                        else:
                            input_widget.addItem(str(opt), opt)
                    # 选中默认值
                    def_val = str(default_value)
                    for i in range(input_widget.count()):
                        if input_widget.itemData(i) == def_val or input_widget.itemText(i) == def_val:
                            input_widget.setCurrentIndex(i)
                            break
                elif param_type == 'str':
                    input_widget = QLineEdit(str(default_value))
                else:
                    # 未知类型跳过，避免崩溃
                    input_widget = QLabel(f"(不支持的类型: {param_type})")
                    input_widget.setStyleSheet("color:#888")

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
            elif isinstance(input_widget, QComboBox):
                config[param_name] = input_widget.currentData() or input_widget.currentText()
            elif isinstance(input_widget, QLineEdit):
                config[param_name] = input_widget.text()
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


_SCROLL_STYLE = """
    QScrollArea { border: none; background: transparent; }
    QScrollBar:vertical {
        background: #1a1a1a;
        width: 10px;
        margin: 0;
        border-radius: 5px;
    }
    QScrollBar::handle:vertical {
        background: #444;
        border-radius: 5px;
        min-height: 28px;
    }
    QScrollBar::handle:vertical:hover { background: #666; }
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
    QScrollBar:horizontal {
        background: #1a1a1a;
        height: 8px;
        border-radius: 4px;
    }
    QScrollBar::handle:horizontal {
        background: #444;
        border-radius: 4px;
        min-width: 28px;
    }
    QScrollBar::handle:horizontal:hover { background: #666; }
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }
"""


class QuantTradeWindow(QMainWindow):
    """量化交易主窗口"""

    def __init__(self, okx_client: OKXClient = None):
        super().__init__()
        self.okx_client = okx_client
        self.trade_executor = None
        _proj_root = os.path.normpath(
            os.path.join(os.path.dirname(__file__), '..', '..')
        )
        _strategies_dir = os.path.join(_proj_root, 'strategies')
        self.strategy_loader = StrategyLoader(strategies_dir=_strategies_dir)
        self._last_strategy_file = os.path.join(
            os.path.dirname(__file__), '..', 'last_strategy.json'
        )
        self.current_strategy = None
        self.current_strategy_module = None
        self.strategy_config = {}
        self.is_running = False
        self._drag_pos = None   # 窗口拖动跟踪
        self.data_manager = DataManager()  # 共享数据库实例（供回测/数据库页共用）
        self.trade_settings_manager = SharedTradeSettingsManager()
        self._mt_daily_loss: float = 0.0          # 当日已亏损（USDT，正数=亏损）
        self._mt_daily_reset_date: str = ""        # 上次重置日期
        self._mt_order_thread: Optional[QThread] = None  # 异步下单线程
        self._scan_auto_trader: Optional[ScanDrivenAutoTrader] = None  # 扫描驱动自动交易编排器
        self._scan_paper_engine: Optional[PaperTradeEngine] = None     # 扫描驱动模拟引擎

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
        self.setMinimumSize(1100, 700)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
            }
            /* 全局滚动条样式 */
            QScrollBar:vertical {
                background: #1a1a1a; width: 10px; border-radius: 5px; margin: 0;
            }
            QScrollBar::handle:vertical {
                background: #444; border-radius: 5px; min-height: 28px;
            }
            QScrollBar::handle:vertical:hover { background: #666; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
            QScrollBar:horizontal {
                background: #1a1a1a; height: 8px; border-radius: 4px;
            }
            QScrollBar::handle:horizontal {
                background: #444; border-radius: 4px; min-width: 28px;
            }
            QScrollBar::handle:horizontal:hover { background: #666; }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }
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

        # 自动交易页面
        trade_widget = self.create_trade_page()
        self.main_tabs.addTab(self._make_scrollable(trade_widget), "自动交易")

        # 回测页面
        self.backtest_page = BacktestPage(
            self.okx_client,
            data_manager=self.data_manager,
            trade_settings_manager=self.trade_settings_manager,
            open_trade_settings_callback=self.open_trade_parameter_tab,
        )
        self.backtest_page_container = self._make_scrollable(self.backtest_page)
        self.main_tabs.addTab(self.backtest_page_container, "策略回测")

        # 共享交易参数页面
        self.trade_parameter_page = TradeParameterPage(self.trade_settings_manager, parent=self)
        self.trade_parameter_page.settings_changed.connect(self.on_trade_settings_changed)
        self.trade_parameter_page_container = self._make_scrollable(self.trade_parameter_page)
        self.main_tabs.addTab(self.trade_parameter_page_container, "交易参数设定")

        # 扫描页面
        from src.ui.scanner_page import ScannerPage
        self.scanner_page = ScannerPage(self.okx_client)
        self.main_tabs.addTab(self._make_scrollable(self.scanner_page), "交易对扫描")

        # 手动交易独立标签页
        self.manual_trade_tab = self.create_manual_trade_tab()
        self.main_tabs.addTab(self._make_scrollable(self.manual_trade_tab), "✋ 手动交易")

        # AI 智能交易助理
        try:
            from src.ui.assistant_page import AssistantPage
            self.assistant_page = AssistantPage(
                okx_client=self.okx_client,
                parent=self,
            )
            self.main_tabs.addTab(self._make_scrollable(self.assistant_page), "🧠 智能助理")
        except Exception as e:
            print(f"[WARN] AssistantPage load failed: {e}")
            import traceback; traceback.print_exc()

        # AI顾问页面
        try:
            from src.ui.ai_advisor_page import AIAdvisorPage
            self.ai_advisor_page = AIAdvisorPage(parent=self)
            self.main_tabs.addTab(self._make_scrollable(self.ai_advisor_page), "🤖 AI顾问")
        except Exception as e:
            print(f"[WARN] AIAdvisorPage load failed: {e}")

        # RL强化学习页面
        try:
            from src.ui.rl_learning_page import RLLearningPage
            self.rl_learning_page = RLLearningPage(okx_client=self.okx_client)
            self.main_tabs.addTab(self._make_scrollable(self.rl_learning_page), "🧠 RL学习")
        except Exception as e:
            print(f"[WARN] RLLearningPage load failed: {e}")

        # 交易池页面
        try:
            from src.ui.trade_pool_page import TradePoolPage
            self.trade_pool_page = TradePoolPage()
            self.main_tabs.addTab(self._make_scrollable(self.trade_pool_page), "📦 交易池")
            if hasattr(self, "scanner_page"):
                self.scanner_page.trade_pool_page = self.trade_pool_page
        except Exception as e:
            print(f"[WARN] TradePoolPage load failed: {e}")

        # 监控池页面
        try:
            from src.ui.monitor_pool_page import MonitorPoolPage
            self.monitor_pool_page = MonitorPoolPage(
                okx_client=self.okx_client,
                trade_executor=self.trade_executor,
                trade_settings_manager=self.trade_settings_manager,
            )
            self.main_tabs.addTab(self.monitor_pool_page, "📡 监控池")
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"[WARN] MonitorPoolPage load failed: {e}")

        if hasattr(self, "trade_pool_page") and hasattr(self, "monitor_pool_page"):
            self.trade_pool_page.monitor_pool_page = self.monitor_pool_page

        # 交易对数据库页面
        try:
            from src.ui.database_page import DatabasePage
            self.database_page = DatabasePage(self.okx_client)
            self.main_tabs.addTab(self._make_scrollable(self.database_page), "🗄️ 交易对数据库")
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"[WARN] DatabasePage load failed: {e}")

        # 等待所有页面创建后，给 AI 顾问注入完整系统上下文
        try:
            if hasattr(self, 'ai_advisor_page'):
                self.ai_advisor_page.wire_system_context(
                    trade_executor=(
                        self.trade_executor if hasattr(self, 'trade_executor') else None
                    ),
                    scanner=(
                        self.scanner_page if hasattr(self, 'scanner_page') else None
                    ),
                    trade_pool=(
                        self.trade_pool_page if hasattr(self, 'trade_pool_page') else None
                    ),
                    monitor_pool=(
                        self.monitor_pool_page if hasattr(self, 'monitor_pool_page') else None
                    ),
                    tracker=(
                        self.rl_learning_page.tracker
                        if hasattr(self, 'rl_learning_page') else None
                    ),
                    timeframe_tracker=(
                        self.rl_learning_page.timeframe_tracker
                        if hasattr(self, 'rl_learning_page') else None
                    ),
                    optimizer=(
                        self.rl_learning_page.optimizer
                        if hasattr(self, 'rl_learning_page') else None
                    ),
                    mutator=(
                        self.rl_learning_page.mutator
                        if hasattr(self, 'rl_learning_page') else None
                    ),
                )
        except Exception as e:
            print(f"[WARN] AI advisor wiring failed: {e}")

        main_layout.addWidget(self.main_tabs)

    # ── 工具：滚动区域包装 ────────────────────────────────────────
    def _make_scrollable(self, widget: QWidget) -> QScrollArea:
        """将任意页面包装进 QScrollArea，垂直滚动条始终显示"""
        sa = QScrollArea()
        sa.setWidget(widget)
        sa.setWidgetResizable(True)
        sa.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        sa.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        sa.setStyleSheet(_SCROLL_STYLE)
        return sa

    # ── 窗口拖动（点击状态栏任意空白区域拖动窗口） ────────────────
    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            try:
                self._drag_pos = event.globalPosition().toPoint()
            except AttributeError:
                self._drag_pos = event.globalPos()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._drag_pos is not None and (event.buttons() & Qt.LeftButton):
            try:
                cur = event.globalPosition().toPoint()
            except AttributeError:
                cur = event.globalPos()
            # 只在非最大化时允许拖动
            if not self.isMaximized():
                self.move(self.pos() + cur - self._drag_pos)
            self._drag_pos = cur
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        self._drag_pos = None
        super().mouseReleaseEvent(event)

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
            "BTC-USDT-SWAP", "ETH-USDT-SWAP", "SOL-USDT-SWAP", "XRP-USDT-SWAP",
            "ADA-USDT-SWAP", "AVAX-USDT-SWAP", "DOGE-USDT-SWAP", "DOT-USDT-SWAP",
        ])
        _pair_row = QWidget()
        _pair_row_lay = QHBoxLayout(_pair_row)
        _pair_row_lay.setContentsMargins(0, 0, 0, 0)
        _pair_row_lay.addWidget(self.pair_combo, 1)
        self._load_pairs_btn = QPushButton("🌐 全部合约")
        self._load_pairs_btn.setFixedWidth(90)
        self._load_pairs_btn.setCheckable(True)
        self._load_pairs_btn.setToolTip(
            "点击后高亮：扫描驱动将扫描整个交易所全部 USDT 永续合约\n"
            "再次点击取消"
        )
        self._load_pairs_btn.setStyleSheet("""
            QPushButton { background:#2a2a2a; color:#ccc; border:1px solid #555;
                          border-radius:4px; padding:3px 6px; }
            QPushButton:checked { background:#0066cc; color:#fff;
                                  border:1px solid #44aaff; font-weight:bold; }
            QPushButton:hover { background:#333; }
        """)
        self._load_pairs_btn.toggled.connect(self._on_all_contracts_toggled)
        _pair_row_lay.addWidget(self._load_pairs_btn)
        pair_layout.addRow("交易对:", _pair_row)

        self.leverage_spin = QSpinBox()
        self.leverage_spin.setRange(1, 100)
        self.leverage_spin.setValue(int(self.trade_settings_manager.get_common_risk().get("leverage", 3) or 3))
        self.leverage_spin.setSuffix("x")
        self.leverage_spin.setEnabled(False)
        pair_layout.addRow("杠杆倍数:", self.leverage_spin)

        self.auto_risk_hint = QLabel("自动交易杠杆、仓位、止盈止损由“交易参数设定”页统一控制。")
        self.auto_risk_hint.setWordWrap(True)
        self.auto_risk_hint.setStyleSheet("color:#aaaaaa; font-size:11px;")
        pair_layout.addRow("", self.auto_risk_hint)

        pair_group.setLayout(pair_layout)
        layout.addWidget(pair_group)

        # ── 交易模式选择 ────────────────────────────────────────────────────
        mode_group = QGroupBox("交易模式")
        mode_layout = QVBoxLayout(mode_group)

        self._trade_mode_group = QButtonGroup(self)
        self._radio_live = QRadioButton("🔴  实盘交易（使用真实账户资金）")
        self._radio_paper = QRadioButton("📊  模拟交易（不使用真实资金）")
        self._radio_live.setChecked(True)
        self._trade_mode_group.addButton(self._radio_live, 0)
        self._trade_mode_group.addButton(self._radio_paper, 1)
        mode_layout.addWidget(self._radio_live)
        mode_layout.addWidget(self._radio_paper)

        # 模拟资金设置（仅模拟模式显示）
        self._paper_settings_widget = QWidget()
        paper_form = QFormLayout(self._paper_settings_widget)
        paper_form.setContentsMargins(0, 4, 0, 0)
        self._paper_capital_spin = QDoubleSpinBox()
        self._paper_capital_spin.setRange(100, 10_000_000)
        self._paper_capital_spin.setDecimals(2)
        self._paper_capital_spin.setValue(10_000.0)
        self._paper_capital_spin.setPrefix("$ ")
        self._paper_capital_spin.setSingleStep(1000)
        paper_form.addRow("模拟初始资金:", self._paper_capital_spin)

        self._paper_review_btn = QPushButton("📂  查看历史模拟报告")
        self._paper_review_btn.clicked.connect(self._open_paper_report_dialog)
        paper_form.addRow("", self._paper_review_btn)
        self._paper_settings_widget.setVisible(False)
        mode_layout.addWidget(self._paper_settings_widget)

        self._radio_paper.toggled.connect(
            lambda checked: self._paper_settings_widget.setVisible(checked)
        )
        layout.addWidget(mode_group)

        # 控制按钮
        control_group = QGroupBox("交易控制")
        control_layout = QVBoxLayout(control_group)

        self.start_btn = QPushButton("🚀 一键启动自动交易")
        self.start_btn.setMinimumHeight(56)
        self.start_btn.setToolTip(
            "同时启动「策略执行」和「扫描驱动全市场自动交易」\n"
            "点击后将立即触发一次全市场扫描并开始自动建仓/平仓"
        )
        self.start_btn.setStyleSheet("""
            QPushButton {
                background-color: #00aa00;
                color: white;
                border-radius: 8px;
                font-weight: bold;
                font-size: 15px;
                padding: 4px 0;
            }
            QPushButton:hover {
                background-color: #00cc00;
            }
            QPushButton:disabled {
                background-color: #555;
                color: #999;
            }
        """)
        self.start_btn.clicked.connect(self.toggle_strategy)
        control_layout.addWidget(self.start_btn)

        self.stop_btn = QPushButton("⏹ 停止全部")
        self.stop_btn.setMinimumHeight(50)
        self.stop_btn.setToolTip("同时停止策略执行和扫描驱动自动交易")
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #aa0000;
                color: white;
                border-radius: 8px;
                font-weight: bold;
                font-size: 15px;
            }
            QPushButton:hover {
                background-color: #cc0000;
            }
            QPushButton:disabled {
                background-color: #555;
                color: #999;
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
        """创建自动交易页面"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(15)

        # 分割器
        splitter = QSplitter(Qt.Horizontal)

        # 左侧：策略管理
        left_panel = self.create_strategy_panel()
        splitter.addWidget(left_panel)

        # 右侧：标签页（实盘监控 + 模拟交易面板 + 扫描驱动）
        right_tabs = QTabWidget()
        right_tabs.addTab(self.create_trade_panel(), "📡 实盘监控")
        right_tabs.addTab(self._create_paper_panel(), "📊 模拟交易")
        right_tabs.addTab(self._create_scan_driven_panel(), "🔄 扫描驱动")
        splitter.addWidget(right_tabs)

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

    def _create_paper_panel(self) -> QWidget:
        """创建模拟交易面板"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # 汇总信息栏
        summary_group = QGroupBox("模拟账户汇总")
        summary_layout = QVBoxLayout(summary_group)
        self._paper_summary_label = QLabel("尚未启动模拟交易")
        self._paper_summary_label.setTextFormat(Qt.RichText)
        self._paper_summary_label.setStyleSheet("font-size:13px; padding:4px;")
        summary_layout.addWidget(self._paper_summary_label)
        layout.addWidget(summary_group)

        # 当前模拟持仓
        pos_group = QGroupBox("当前模拟持仓")
        pos_layout = QVBoxLayout(pos_group)
        self._paper_pos_table = QTableWidget()
        self._paper_pos_table.setColumnCount(9)
        self._paper_pos_table.setHorizontalHeaderLabels([
            "交易对", "方向", "开仓价", "当前价", "投入(USDT)",
            "杠杆", "浮动盈亏", "止盈价", "止损价",
        ])
        self._paper_pos_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._paper_pos_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._paper_pos_table.setAlternatingRowColors(True)
        self._paper_pos_table.setMaximumHeight(160)
        pos_layout.addWidget(self._paper_pos_table)
        layout.addWidget(pos_group)

        # 本次会话成交记录
        trade_group = QGroupBox("本次会话成交记录")
        trade_layout = QVBoxLayout(trade_group)
        self._paper_trade_table = QTableWidget()
        self._paper_trade_table.setColumnCount(10)
        self._paper_trade_table.setHorizontalHeaderLabels([
            "交易对", "方向", "开仓时间", "平仓时间",
            "开仓价", "平仓价", "投入(USDT)", "盈亏(USDT)", "盈亏%", "平仓原因",
        ])
        self._paper_trade_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self._paper_trade_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._paper_trade_table.setAlternatingRowColors(True)
        trade_layout.addWidget(self._paper_trade_table)
        layout.addWidget(trade_group, 1)

        # 底部操作按钮
        btn_layout = QHBoxLayout()
        review_btn = QPushButton("📂 查看历史模拟报告")
        review_btn.clicked.connect(self._open_paper_report_dialog)
        btn_layout.addWidget(review_btn)
        export_btn = QPushButton("💾 导出当前会话")
        export_btn.clicked.connect(self._export_current_paper_session)
        btn_layout.addWidget(export_btn)
        layout.addLayout(btn_layout)

        return widget

    def _export_current_paper_session(self):
        """手动保存当前模拟会话"""
        if not hasattr(self, '_paper_engine') or not self._paper_engine:
            QMessageBox.information(self, "提示", "当前没有正在运行的模拟交易会话")
            return
        path = self._paper_engine.save_final()
        QMessageBox.information(self, "已保存", f"模拟会话报告已保存至：\n{path}")

    def _open_paper_report_dialog(self):
        """打开历史模拟报告复盘窗口"""
        dialog = PaperReportDialog(self)
        dialog.exec_()


    # ═══════════════════════════════════════════════════════════════════════
    # 扫描驱动自动交易面板
    # ═══════════════════════════════════════════════════════════════════════

    def _create_scan_driven_panel(self) -> QWidget:
        """创建扫描驱动自动交易面板"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # ── 顶部：模式开关 + 状态 ────────────────────────────────────────
        header_layout = QHBoxLayout()

        # 扫描驱动由左侧「一键启动自动交易」按钮统一控制，此处只显示模式说明
        _sd_title = QLabel("🔄 扫描驱动自动交易")
        _sd_title.setStyleSheet("font-size:14px; font-weight:bold; color:#00ccff;")
        header_layout.addWidget(_sd_title)
        _sd_hint = QLabel("（由左侧「一键启动」按钮统一开关）")
        _sd_hint.setStyleSheet("font-size:11px; color:#888;")
        header_layout.addWidget(_sd_hint)
        # 保留内部引用供兼容旧代码（setChecked 调用变为无操作）
        self._scan_driven_enable_cb = None

        header_layout.addStretch()

        self._scan_driven_status_label = QLabel("● 未启用")
        self._scan_driven_status_label.setStyleSheet("color:#888; font-weight:bold;")
        header_layout.addWidget(self._scan_driven_status_label)

        layout.addLayout(header_layout)

        # ── 实时扫描状态栏（始终可见）───────────────────────────────────
        scan_status_frame = QFrame()
        scan_status_frame.setStyleSheet(
            "QFrame { background:#1a1a2e; border:1px solid #333; border-radius:6px; padding:4px; }"
        )
        scan_status_v = QVBoxLayout(scan_status_frame)
        scan_status_v.setContentsMargins(6, 4, 6, 4)
        scan_status_v.setSpacing(3)

        # 状态文字行
        scan_status_h = QHBoxLayout()
        self._sd_scan_state_label = QLabel("⏸ 扫描驱动未启用")
        self._sd_scan_state_label.setStyleSheet("color:#888; font-size:12px; font-weight:bold;")
        scan_status_h.addWidget(self._sd_scan_state_label, 1)
        self._sd_scan_pair_label = QLabel("")
        self._sd_scan_pair_label.setStyleSheet("color:#aaa; font-size:11px;")
        scan_status_h.addWidget(self._sd_scan_pair_label)
        scan_status_v.addLayout(scan_status_h)

        # 进度条
        self._sd_scan_progress_bar = QProgressBar()
        self._sd_scan_progress_bar.setRange(0, 100)
        self._sd_scan_progress_bar.setValue(0)
        self._sd_scan_progress_bar.setTextVisible(True)
        self._sd_scan_progress_bar.setMaximumHeight(14)
        self._sd_scan_progress_bar.setStyleSheet("""
            QProgressBar { background:#222; border:1px solid #444; border-radius:3px;
                           color:#fff; font-size:10px; text-align:center; }
            QProgressBar::chunk { background:#0066cc; border-radius:3px; }
        """)
        self._sd_scan_progress_bar.setVisible(False)
        scan_status_v.addWidget(self._sd_scan_progress_bar)

        layout.addWidget(scan_status_frame)

        # ── 交易模式选择 ─────────────────────────────────────────────────
        mode_group = QGroupBox("交易模式")
        mode_layout = QHBoxLayout(mode_group)
        self._sd_trade_mode_group = QButtonGroup(self)
        self._sd_radio_live = QRadioButton("🔴  实盘（真实账户）")
        self._sd_radio_paper = QRadioButton("📊  模拟（不动账户资金）")
        self._sd_radio_live.setChecked(True)
        self._sd_trade_mode_group.addButton(self._sd_radio_live, 0)
        self._sd_trade_mode_group.addButton(self._sd_radio_paper, 1)
        mode_layout.addWidget(self._sd_radio_live)
        mode_layout.addWidget(self._sd_radio_paper)
        mode_layout.addStretch()
        layout.addWidget(mode_group)

        # ── 扫描策略 & 间隔设置 ───────────────────────────────────────────
        scan_group = QGroupBox("扫描策略 & 间隔")
        scan_form = QFormLayout(scan_group)
        scan_form.setSpacing(6)

        # 当前选中策略展示
        self._sd_strategy_label = QLabel('（未选择，请先在"交易对扫描"页单击高亮一个策略）')
        self._sd_strategy_label.setStyleSheet("color:#ffcc44; font-size:12px;")
        self._sd_strategy_label.setWordWrap(True)
        scan_form.addRow("扫描策略:", self._sd_strategy_label)

        # 刷新显示当前策略名
        _refresh_strategy_btn = QPushButton("🔄 刷新")
        _refresh_strategy_btn.setFixedWidth(70)
        _refresh_strategy_btn.setToolTip("显示扫描页当前高亮的策略名")
        _refresh_strategy_btn.clicked.connect(self._sd_refresh_strategy_label)
        scan_form.addRow("", _refresh_strategy_btn)

        # 扫描间隔
        self._sd_scan_interval_spin = QSpinBox()
        self._sd_scan_interval_spin.setRange(1, 1440)
        self._sd_scan_interval_spin.setValue(15)
        self._sd_scan_interval_spin.setSuffix(" 分钟")
        self._sd_scan_interval_spin.setToolTip("启用后每隔此时长自动触发一次扫描")
        scan_form.addRow("扫描间隔:", self._sd_scan_interval_spin)

        # 倒计时显示
        self._sd_countdown_label = QLabel("下次扫描：--")
        self._sd_countdown_label.setStyleSheet("color:#aaa; font-size:12px;")
        scan_form.addRow("", self._sd_countdown_label)

        # 立即触发按钮
        _trigger_now_btn = QPushButton("▶ 立即触发一次扫描")
        _trigger_now_btn.clicked.connect(self._sd_trigger_scan_now)
        scan_form.addRow("", _trigger_now_btn)

        layout.addWidget(scan_group)

        # ── 参数设置 ─────────────────────────────────────────────────────
        settings_group = QGroupBox("筛选 & 风控参数")
        settings_form = QFormLayout(settings_group)
        settings_form.setSpacing(6)

        self._sd_min_score_spin = QDoubleSpinBox()
        self._sd_min_score_spin.setRange(0, 100)
        self._sd_min_score_spin.setValue(50.0)  # 扫描策略已做主要过滤，50 分即可入选
        self._sd_min_score_spin.setSuffix(" 分")
        self._sd_min_score_spin.setDecimals(1)
        settings_form.addRow("最低入选评分:", self._sd_min_score_spin)

        self._sd_max_pos_spin = QSpinBox()
        self._sd_max_pos_spin.setRange(1, 20)
        self._sd_max_pos_spin.setValue(3)
        settings_form.addRow("最大同时持仓数:", self._sd_max_pos_spin)

        self._sd_allow_short_cb = QCheckBox("允许做空")
        self._sd_allow_short_cb.setChecked(True)
        settings_form.addRow("方向:", self._sd_allow_short_cb)

        self._sd_expiry_spin = QDoubleSpinBox()
        self._sd_expiry_spin.setRange(0.5, 24)
        self._sd_expiry_spin.setValue(4.0)
        self._sd_expiry_spin.setSuffix(" 小时")
        self._sd_expiry_spin.setDecimals(1)
        settings_form.addRow("信号有效期:", self._sd_expiry_spin)

        self._sd_paper_capital_spin = QDoubleSpinBox()
        self._sd_paper_capital_spin.setRange(100, 10_000_000)
        self._sd_paper_capital_spin.setValue(10_000)
        self._sd_paper_capital_spin.setPrefix("$ ")
        self._sd_paper_capital_spin.setDecimals(0)
        self._sd_paper_capital_spin.setSingleStep(1000)
        settings_form.addRow("模拟初始资金:", self._sd_paper_capital_spin)

        layout.addWidget(settings_group)

        # ── 活跃 campaigns 表格 ──────────────────────────────────────────
        camp_group = QGroupBox("活跃交易对监控")
        camp_layout = QVBoxLayout(camp_group)

        self._sd_summary_label = QLabel("监控中：0 对  |  已建仓：0 个")
        self._sd_summary_label.setStyleSheet("color:#aaa; font-size:12px;")
        camp_layout.addWidget(self._sd_summary_label)

        self._sd_gate_reason_label = QLabel("过滤/等待原因：--")
        self._sd_gate_reason_label.setStyleSheet("color:#caa96b; font-size:12px;")
        self._sd_gate_reason_label.setWordWrap(True)
        camp_layout.addWidget(self._sd_gate_reason_label)

        self._sd_camp_table = QTableWidget()
        self._sd_camp_table.setColumnCount(8)
        self._sd_camp_table.setHorizontalHeaderLabels([
            "交易对", "方向", "评分", "状态",
            "当前价", "开仓价", "浮动盈亏", "信号原因",
        ])
        self._sd_camp_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._sd_camp_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._sd_camp_table.setAlternatingRowColors(True)
        self._sd_camp_table.setMaximumHeight(200)
        self._sd_camp_table.setSelectionBehavior(QTableWidget.SelectRows)
        camp_layout.addWidget(self._sd_camp_table)

        btn_row = QHBoxLayout()
        self._sd_stop_all_btn = QPushButton("⏹ 停止全部")
        self._sd_stop_all_btn.setEnabled(False)
        self._sd_stop_all_btn.clicked.connect(self._sd_stop_all)
        btn_row.addWidget(self._sd_stop_all_btn)

        self._sd_stop_one_btn = QPushButton("⏹ 停止选中")
        self._sd_stop_one_btn.setEnabled(False)
        self._sd_stop_one_btn.clicked.connect(self._sd_stop_selected)
        btn_row.addWidget(self._sd_stop_one_btn)

        self._sd_paper_review_btn = QPushButton("📂 查看模拟报告")
        self._sd_paper_review_btn.clicked.connect(self._open_paper_report_dialog)
        btn_row.addWidget(self._sd_paper_review_btn)
        camp_layout.addLayout(btn_row)

        layout.addWidget(camp_group)

        # ── 扫描驱动日志 ─────────────────────────────────────────────────
        log_group = QGroupBox("扫描驱动日志")
        log_layout = QVBoxLayout(log_group)
        self._sd_log_text = QTextEdit()
        self._sd_log_text.setReadOnly(True)
        self._sd_log_text.setMaximumHeight(150)
        self._sd_log_text.setStyleSheet(
            "background:#111; color:#ccc; font-family:Consolas,monospace; font-size:12px;"
        )
        log_layout.addWidget(self._sd_log_text)
        layout.addWidget(log_group, 1)

        return widget

    # ── 扫描驱动：事件处理 ────────────────────────────────────────────────

    def _on_scan_driven_toggled(self, checked: bool):
        """启用 / 禁用扫描驱动自动交易"""
        if checked:
            self._init_scan_auto_trader()
        else:
            self._destroy_scan_auto_trader()

    def _get_scan_driven_config(self) -> dict:
        """从 UI 读取扫描驱动参数，合并共享交易参数"""
        base = self.trade_settings_manager.get_all() if hasattr(self, 'trade_settings_manager') else {}
        cfg = dict(base)
        cfg['min_auto_score'] = self._sd_min_score_spin.value()
        cfg['max_concurrent_positions'] = self._sd_max_pos_spin.value()
        cfg['allow_short'] = self._sd_allow_short_cb.isChecked()
        cfg['signal_expiry_hours'] = self._sd_expiry_spin.value()
        return cfg

    def _init_scan_auto_trader(self):
        """创建并激活扫描驱动编排器"""
        # 先清理旧实例
        self._destroy_scan_auto_trader()

        cfg = self._get_scan_driven_config()
        paper_mode = self._sd_radio_paper.isChecked()

        if paper_mode:
            _cost = cfg.get('cost', {})
            self._scan_paper_engine = PaperTradeEngine(
                okx_client=self.okx_client,
                initial_capital=self._sd_paper_capital_spin.value(),
                fee_pct=float(_cost.get('fee_pct', 0.05) or 0.05),
                slippage_pct=float(_cost.get('slippage_pct', 0.03) or 0.03),
                market_impact_pct=float(_cost.get('market_impact_pct', 0.02) or 0.02),
                funding_rate_8h_pct=float(_cost.get('funding_rate_8h_pct', 0.01) or 0.01),
                strategy_name="扫描驱动模拟",
            )
            executor = self._scan_paper_engine
            mode_txt = "模拟"
        else:
            if not self.trade_executor:
                # checkbox 已合并进统一按钮，直接停止并提示
                self._sd_log("实盘模式需要先连接 OKX API", "ERROR")
                self.stop_strategy()
                return
            self._scan_paper_engine = None
            executor = self.trade_executor
            mode_txt = "实盘"

        self._scan_auto_trader = ScanDrivenAutoTrader(
            okx_client=self.okx_client,
            trade_executor=executor,
            config=cfg,
        )
        self._scan_auto_trader.log_signal.connect(self._sd_on_log)
        self._scan_auto_trader.state_updated.connect(self._sd_on_state_updated)
        self._scan_auto_trader.position_opened.connect(
            lambda inst, direction: self._sd_log(
                f"▶ {inst} {direction} 已建仓", "TRADE"
            )
        )
        self._scan_auto_trader.position_closed.connect(
            lambda inst: self._sd_log(f"■ {inst} 已完成", "INFO")
        )
        self._scan_auto_trader.conflict_signal.connect(self._sd_on_conflict_position)

        # 连接扫描结果信号 + 实时进度日志
        if hasattr(self, 'scanner_page'):
            try:
                self.scanner_page.scan_results_ready.connect(
                    self._scan_auto_trader.on_scan_results
                )
            except Exception:
                pass
            try:
                self.scanner_page.scan_log_signal.connect(self._sd_on_log)
            except Exception:
                pass

        # ── 监控池高质量信号 → 扫描驱动自动交易 ─────────────────────────────
        if hasattr(self, 'monitor_pool_page') and self.monitor_pool_page:
            self.monitor_pool_page.set_scan_auto_trader(self._scan_auto_trader)

        # ── 独立扫描定时器 ────────────────────────────────────────────────
        interval_ms = int(self._sd_scan_interval_spin.value() * 60 * 1000)
        self._sd_scan_timer = QTimer(self)
        self._sd_scan_timer.setInterval(interval_ms)
        # 主定时器触发时先重置倒计时，再执行扫描
        def _on_sd_timer():
            self._sd_countdown_remaining = int(self._sd_scan_interval_spin.value() * 60)
            self._sd_trigger_scan_now()
        self._sd_scan_timer.timeout.connect(_on_sd_timer)
        self._sd_scan_timer.start()

        # 倒计时显示定时器（每秒刷新一次）
        self._sd_countdown_remaining = int(self._sd_scan_interval_spin.value() * 60)
        self._sd_countdown_tick_timer = QTimer(self)
        self._sd_countdown_tick_timer.setInterval(1000)
        self._sd_countdown_tick_timer.timeout.connect(self._sd_update_countdown)
        self._sd_countdown_tick_timer.start()

        # 立刻触发第一次扫描
        QTimer.singleShot(500, self._sd_trigger_scan_now)

        self._scan_driven_status_label.setText(f"● 运行中（{mode_txt}）")
        self._scan_driven_status_label.setStyleSheet(
            "color:#00ff88; font-weight:bold;"
        )
        self._sd_stop_all_btn.setEnabled(True)
        self._sd_stop_one_btn.setEnabled(True)
        self._sd_log(
            f"[扫描驱动] 已启用（{mode_txt}模式）  "
            f"扫描间隔 {self._sd_scan_interval_spin.value():.0f} 分钟  "
            f"立即触发首次扫描…",
            "SUCCESS",
        )

    def _destroy_scan_auto_trader(self):
        """停止并销毁扫描驱动编排器"""
        # 先停定时器
        for attr in ('_sd_scan_timer', '_sd_countdown_tick_timer'):
            t = getattr(self, attr, None)
            if t is not None:
                try:
                    t.stop()
                except Exception:
                    pass
                setattr(self, attr, None)

        if self._scan_auto_trader:
            # 断开扫描信号
            if hasattr(self, 'scanner_page'):
                try:
                    self.scanner_page.scan_results_ready.disconnect(
                        self._scan_auto_trader.on_scan_results
                    )
                except Exception:
                    pass
                try:
                    self.scanner_page.scan_log_signal.disconnect(self._sd_on_log)
                except Exception:
                    pass
            # 断开监控池高质量信号桥接
            if hasattr(self, 'monitor_pool_page') and self.monitor_pool_page:
                try:
                    self.monitor_pool_page.set_scan_auto_trader(None)
                except Exception:
                    pass
            try:
                self._scan_auto_trader.stop_all()
            except Exception:
                pass
            self._scan_auto_trader = None

        if self._scan_paper_engine:
            try:
                path = self._scan_paper_engine.save_final()
                self._sd_log(f"[模拟] 报告已保存：{path}", "INFO")
            except Exception:
                pass
            self._scan_paper_engine = None

        self._scan_driven_status_label.setText("● 未启用")
        self._scan_driven_status_label.setStyleSheet("color:#888; font-weight:bold;")
        self._sd_stop_all_btn.setEnabled(False)
        self._sd_stop_one_btn.setEnabled(False)
        self._sd_camp_table.setRowCount(0)
        self._sd_summary_label.setText("监控中：0 对  |  已建仓：0 个")
        if hasattr(self, '_sd_gate_reason_label'):
            self._sd_gate_reason_label.setText("过滤/等待原因：--")
        if hasattr(self, '_sd_countdown_label'):
            self._sd_countdown_label.setText("下次扫描：--")
        if hasattr(self, '_sd_scan_state_label'):
            self._sd_scan_state_label.setText("⏸ 扫描驱动未启用")
            self._sd_scan_state_label.setStyleSheet("color:#888; font-size:12px; font-weight:bold;")
        if hasattr(self, '_sd_scan_progress_bar'):
            self._sd_scan_progress_bar.setVisible(False)
            self._sd_scan_progress_bar.setValue(0)
        if hasattr(self, '_sd_scan_pair_label'):
            self._sd_scan_pair_label.setText("")

    def _sd_refresh_strategy_label(self):
        """刷新「当前扫描策略」标签，显示扫描页的高亮策略"""
        if not hasattr(self, 'scanner_page'):
            return
        item = self.scanner_page.strategy_list.currentItem()
        if item:
            info = item.data(Qt.ItemDataRole.UserRole) if hasattr(Qt, 'ItemDataRole') \
                else item.data(0x0100)  # Qt.UserRole fallback
            name = getattr(info, 'name', str(info)) if info else item.text()
            self._sd_strategy_label.setText(f"✅ {name}")
            self._sd_strategy_label.setStyleSheet("color:#44ff88; font-size:12px;")
        else:
            self._sd_strategy_label.setText('（未选择，请在"交易对扫描"页单击高亮一个策略）')
            self._sd_strategy_label.setStyleSheet("color:#ffcc44; font-size:12px;")

    def _sd_trigger_scan_now(self):
        """立即触发一次扫描（使用扫描页当前高亮策略）"""
        # ── 诊断检查 1：扫描页是否存在 ───────────────────────────────────
        if not hasattr(self, 'scanner_page'):
            self._sd_log("❌ 未找到扫描页对象，无法触发扫描", "ERROR")
            return

        # ── 诊断检查 2：是否正在扫描中 ──────────────────────────────────
        if getattr(self.scanner_page, 'is_scanning', False):
            self._sd_log("⏳ 上轮扫描仍在进行中，本次跳过", "INFO")
            return

        # ── 诊断检查 3：策略列表是否有策略 ───────────────────────────────
        strat_count = 0
        if hasattr(self.scanner_page, 'strategy_list'):
            strat_count = self.scanner_page.strategy_list.count()
        if strat_count == 0:
            self._sd_log(
                '❌ 扫描页策略列表为空，请先切换到"交易对扫描"页等待策略加载',
                "ERROR"
            )
            return

        # ── 诊断检查 4：当前高亮策略 ─────────────────────────────────────
        item = self.scanner_page.strategy_list.currentItem()
        if not item:
            self._sd_log(
                f'⚠ 策略列表有 {strat_count} 个策略但无高亮项，'
                '请在"交易对扫描"页单击选中一个策略',
                "WARNING"
            )
            return

        # ── 读取策略信息 ─────────────────────────────────────────────────
        try:
            from src.qt_compat import Qt as _Qt
            _role = _Qt.ItemDataRole.UserRole if hasattr(_Qt, 'ItemDataRole') else _Qt.UserRole
            info = item.data(_role)
        except Exception as e:
            self._sd_log(f"❌ 读取策略信息失败: {e}", "ERROR")
            return

        if not info:
            self._sd_log("❌ 策略数据为空，请重新在扫描页选中策略", "ERROR")
            return

        strategy_name = getattr(info, 'name', str(info))
        strategy_path = getattr(info, 'path', '未知路径')

        self._sd_log(
            f"━━ 触发扫描 ━━  策略: {strategy_name}  "
            f"文件: {strategy_path.split('/')[-1]}  "
            f"扫描全部 USDT 永续合约",
            "INFO"
        )
        if hasattr(self, '_sd_strategy_label'):
            self._sd_strategy_label.setText(f"✅ {strategy_name}")
            self._sd_strategy_label.setStyleSheet("color:#44ff88; font-size:12px;")

        # ── 更新状态栏 ──────────────────────────────────────────────────
        if hasattr(self, '_sd_scan_state_label'):
            self._sd_scan_state_label.setText("⏳ 正在启动扫描…")
            self._sd_scan_state_label.setStyleSheet(
                "color:#ffcc44; font-size:12px; font-weight:bold;"
            )
        if hasattr(self, '_sd_scan_progress_bar'):
            self._sd_scan_progress_bar.setVisible(True)
            self._sd_scan_progress_bar.setValue(0)
            self._sd_scan_progress_bar.setFormat("启动中…")

        # ── 重置倒计时 ──────────────────────────────────────────────────
        self._sd_countdown_remaining = int(
            self._sd_scan_interval_spin.value() * 60
            if hasattr(self, '_sd_scan_interval_spin') else 900
        )

        # ── 触发扫描 ────────────────────────────────────────────────────
        try:
            self.scanner_page.start_scan(info)
            self._sd_log(
                f"✔ 扫描线程已启动，策略: {strategy_name}  正在后台扫描市场…",
                "SUCCESS"
            )
        except Exception as exc:
            import traceback as _tb
            self._sd_log(f"❌ 触发扫描失败: {exc}", "ERROR")
            self._sd_log(_tb.format_exc()[:300], "ERROR")
            if hasattr(self, '_sd_scan_state_label'):
                self._sd_scan_state_label.setText("❌ 启动失败")
                self._sd_scan_state_label.setStyleSheet(
                    "color:#ff4444; font-size:12px; font-weight:bold;"
                )

    def _sd_update_countdown(self):
        """每秒更新倒计时 + 扫描状态指示"""
        if not hasattr(self, '_sd_countdown_remaining'):
            return

        is_scanning = (
            getattr(self, 'scanner_page', None) is not None
            and getattr(self.scanner_page, 'is_scanning', False)
        )

        if is_scanning:
            # 扫描进行中：countdown 不递减，但刷新显示
            if hasattr(self, '_sd_countdown_label'):
                self._sd_countdown_label.setText("🔄 扫描运行中…")
            # 如果进度条还不可见（可能错过了 start 信号），强制显示
            if hasattr(self, '_sd_scan_progress_bar') and \
                    not self._sd_scan_progress_bar.isVisible():
                self._sd_scan_progress_bar.setVisible(True)
            return

        # 扫描空闲：递减倒计时
        self._sd_countdown_remaining = max(0, self._sd_countdown_remaining - 1)
        mins = self._sd_countdown_remaining // 60
        secs = self._sd_countdown_remaining % 60
        if hasattr(self, '_sd_countdown_label'):
            self._sd_countdown_label.setText(f"⏱ 下次扫描：{mins:02d}:{secs:02d}")

    def _sd_stop_all(self):
        """停止所有扫描驱动工作器（由面板内"停止所有"按钮调用，等同于全停）"""
        self.stop_strategy()

    def _sd_stop_selected(self):
        """停止选中行的交易对"""
        if not self._scan_auto_trader:
            return
        rows = self._sd_camp_table.selectedItems()
        if not rows:
            return
        row = self._sd_camp_table.currentRow()
        item = self._sd_camp_table.item(row, 0)
        if item:
            inst_id = item.text()
            self._scan_auto_trader.stop_one(inst_id)

    def _sd_on_log(self, msg: str, level: str = "INFO"):
        """接收扫描驱动日志：写入日志框 + 解析进度更新状态栏"""
        import re as _re

        # ── 解析进度格式 "[val%|scanned|total] ..." ──────────────────────
        _m = _re.match(r'^\[(\d+)%\|(\d+)\|(\d+)\](.*)$', msg)
        if _m:
            val   = int(_m.group(1))
            scnd  = int(_m.group(2))
            tot   = int(_m.group(3))
            rest  = _m.group(4).strip()

            if hasattr(self, '_sd_scan_progress_bar'):
                self._sd_scan_progress_bar.setVisible(True)
                self._sd_scan_progress_bar.setValue(val)
                self._sd_scan_progress_bar.setFormat(
                    f"{val}%  {scnd}/{tot}" if tot > 0 else f"{val}%"
                )
            if hasattr(self, '_sd_scan_state_label'):
                self._sd_scan_state_label.setText(f"🔄 扫描中 {val}%")
                self._sd_scan_state_label.setStyleSheet(
                    "color:#44ccff; font-size:12px; font-weight:bold;"
                )
            if hasattr(self, '_sd_scan_pair_label') and rest:
                # 取最后一段（交易对名）
                pair_part = rest.split()[-1] if rest.split() else rest
                self._sd_scan_pair_label.setText(pair_part[:30])
            # 精简版写入日志（去掉前缀）
            self._sd_log(f"[{val}%] {rest}" if rest else f"[{val}%]", level)
            return

        # ── 扫描开始 ──────────────────────────────────────────────────────
        if msg.startswith("▶ 开始扫描"):
            if hasattr(self, '_sd_scan_state_label'):
                self._sd_scan_state_label.setText("🔄 正在启动扫描…")
                self._sd_scan_state_label.setStyleSheet(
                    "color:#ffcc44; font-size:12px; font-weight:bold;"
                )
            if hasattr(self, '_sd_scan_progress_bar'):
                self._sd_scan_progress_bar.setVisible(True)
                self._sd_scan_progress_bar.setValue(0)
                self._sd_scan_progress_bar.setFormat("启动中…")

        # ── 扫描完成 ──────────────────────────────────────────────────────
        elif msg.startswith("■ 扫描完成"):
            if hasattr(self, '_sd_scan_state_label'):
                clr = "#44ff88" if "无信号" not in msg else "#aaa"
                self._sd_scan_state_label.setText("✅ 扫描完成，等待下轮")
                self._sd_scan_state_label.setStyleSheet(
                    f"color:{clr}; font-size:12px; font-weight:bold;"
                )
            if hasattr(self, '_sd_scan_progress_bar'):
                self._sd_scan_progress_bar.setValue(100)
                self._sd_scan_progress_bar.setFormat("完成")
            if hasattr(self, '_sd_scan_pair_label'):
                self._sd_scan_pair_label.setText("")

        # ── 命中信号 ──────────────────────────────────────────────────────
        elif msg.startswith("🎯 命中"):
            if hasattr(self, '_sd_scan_pair_label'):
                self._sd_scan_pair_label.setText(msg[3:40])

        # ── 错误 ─────────────────────────────────────────────────────────
        elif level == "ERROR":
            if hasattr(self, '_sd_scan_state_label'):
                self._sd_scan_state_label.setText("❌ 扫描出错")
                self._sd_scan_state_label.setStyleSheet(
                    "color:#ff4444; font-size:12px; font-weight:bold;"
                )

        self._sd_log(msg, level)
        if hasattr(self, 'trade_log') and level in ("ERROR", "WARNING", "TRADE", "SUCCESS"):
            self.trade_log.log(msg, level)

    def _sd_log(self, msg: str, level: str = "INFO"):
        """向扫描驱动日志文本框追加一行"""
        color_map = {
            "ERROR": "#ff4444",
            "WARNING": "#ffcc44",
            "SUCCESS": "#44ff88",
            "TRADE": "#44ccff",
            "INFO": "#cccccc",
        }
        color = color_map.get(level, "#cccccc")
        ts = datetime.now().strftime("%H:%M:%S")
        html = f'<span style="color:#666;">[{ts}]</span> <span style="color:{color};">{msg}</span>'
        if hasattr(self, '_sd_log_text'):
            self._sd_log_text.append(html)
            # 滚动到底部
            sb = self._sd_log_text.verticalScrollBar()
            sb.setValue(sb.maximum())

    def _sd_on_conflict_position(self, inst_id: str, new_dir: str, existing_dir: str):
        """
        扫描驱动检测到反向信号时的弹窗通知。
        平仓操作已在 ScanDrivenAutoTrader 中执行，此处仅做 UI 通知。
        """
        dir_zh = {'LONG': '做多', 'SHORT': '做空'}
        existing_zh = dir_zh.get(existing_dir, existing_dir)
        new_zh = dir_zh.get(new_dir, new_dir)

        # 写入扫描驱动日志
        self._sd_log(
            f"⚡ 反向信号冲突：{inst_id}  现有持仓={existing_zh}  新信号={new_zh}  → 已自动平仓",
            "WARNING",
        )

        # 非阻塞弹窗（information 不阻塞，用户点 OK 即可）
        box = QMessageBox(self)
        box.setWindowTitle("⚡ 反向信号 — 自动平仓通知")
        box.setIcon(QMessageBox.Warning)
        box.setText(
            f"<b>{inst_id}</b>"
            f"<br><br>"
            f"扫描策略发现新信号：<font color='#f0883e'><b>{new_zh}</b></font><br>"
            f"与当前持仓方向（<b>{existing_zh}</b>）<font color='#ff6b6b'>相反</font><br><br>"
            f"<b>已自动平仓。</b>如需开新仓，请等待下次扫描确认入场。"
        )
        box.setStandardButtons(QMessageBox.Ok)
        box.setModal(False)   # 非模态，不阻塞交易界面
        box.show()

    def _sd_on_state_updated(self, states: list):
        """扫描驱动状态更新 → 刷新 campaigns 表格"""
        if not hasattr(self, '_sd_camp_table'):
            return

        self._sd_camp_table.setRowCount(len(states))
        total_monitoring = len(states)
        total_with_pos = 0
        gate_reasons = []

        for row, state in enumerate(states):
            inst_id = state.get('inst_id', '')
            direction = state.get('direction', '')
            score = float(state.get('score', 0))
            stage = state.get('stage', '')
            current_price = float(state.get('current_price', 0))
            entry_price = float(state.get('entry_price', 0))
            pnl = float(state.get('unrealized_pnl', 0))
            reason = state.get('scan_reason', '')
            gate_reason = state.get('gate_reason', '')

            if stage == '已开仓':
                total_with_pos += 1
            elif gate_reason:
                gate_reasons.append(f"{inst_id}: {gate_reason}")

            pnl_color = "#44ff88" if pnl >= 0 else "#ff4444"
            stage_color = "#44ccff" if stage == '已开仓' else "#ffcc44" if stage == '等待入场' else "#888"

            def _item(text, align=Qt.AlignCenter):
                it = QTableWidgetItem(str(text))
                it.setTextAlignment(align)
                return it

            self._sd_camp_table.setItem(row, 0, _item(inst_id))
            dir_item = _item(direction)
            dir_item.setForeground(QColor("#44ff88" if direction == 'LONG' else "#ff6644"))
            self._sd_camp_table.setItem(row, 1, dir_item)
            self._sd_camp_table.setItem(row, 2, _item(f"{score:.1f}"))
            stage_item = _item(stage)
            stage_item.setForeground(QColor(stage_color))
            self._sd_camp_table.setItem(row, 3, stage_item)
            self._sd_camp_table.setItem(row, 4, _item(f"{current_price:.4g}"))
            self._sd_camp_table.setItem(row, 5, _item(f"{entry_price:.4g}" if entry_price else "--"))
            pnl_item = _item(f"{pnl:+.2f}" if stage == '已开仓' else "--")
            pnl_item.setForeground(QColor(pnl_color))
            self._sd_camp_table.setItem(row, 6, pnl_item)
            display_reason = gate_reason if stage != '已开仓' and gate_reason else reason
            self._sd_camp_table.setItem(row, 7, _item(display_reason[:60], Qt.AlignLeft))

        self._sd_summary_label.setText(
            f"监控中：{total_monitoring} 对  |  已建仓：{total_with_pos} 个"
        )
        if hasattr(self, '_sd_gate_reason_label'):
            if gate_reasons:
                self._sd_gate_reason_label.setText(f"过滤/等待原因：{gate_reasons[0][:120]}")
            else:
                self._sd_gate_reason_label.setText("过滤/等待原因：--")

        # 同步更新编排器摘要 (如果存在)
        if self._scan_auto_trader:
            summary = self._scan_auto_trader.active_summary()
            self._scan_driven_status_label.setText(
                f"● 运行中  监控:{summary['total_monitoring']}  建仓:{summary['with_position']}"
            )

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

        auto_group = QGroupBox("自动交易资金池")
        auto_layout = QFormLayout(auto_group)
        self.auto_pool_label = QLabel("-- USDT")
        self.auto_pool_label.setStyleSheet("color:#00ffaa; font-weight:bold;")
        auto_layout.addRow("总资金池:", self.auto_pool_label)
        self.auto_pilot_label = QLabel("--")
        self.auto_add_label = QLabel("--")
        self.auto_rule_label = QLabel(
            "第一原则：每次自动开仓前，必须先检查 3m 回调企稳、H1 趋势延续、"
            "且风险线不能跌破总开仓成本（开仓线+手续费+滑点/冲击）。"
            "若不满足，自动交易一律不开仓。"
        )
        self.auto_rule_label.setWordWrap(True)
        self.auto_rule_label.setStyleSheet("color:#ffcc66; font-size:11px;")
        auto_layout.addRow("1%试仓:", self.auto_pilot_label)
        auto_layout.addRow("10%加仓:", self.auto_add_label)
        auto_layout.addRow("硬规则:", self.auto_rule_label)
        self.auto_fund_hint_label = QLabel("自动交易使用独立总资金池，不读取手动交易页的下单金额逻辑。")
        self.auto_fund_hint_label.setWordWrap(True)
        self.auto_fund_hint_label.setStyleSheet("color:#aaaaaa; font-size:11px;")
        auto_layout.addRow("", self.auto_fund_hint_label)
        open_settings_btn = QPushButton("打开交易参数设定")
        open_settings_btn.clicked.connect(self.open_trade_parameter_tab)
        auto_layout.addRow("", open_settings_btn)
        auto_group.setLayout(auto_layout)
        layout.addWidget(auto_group)
        layout.addStretch()

        # 将面板设置为滚动区域的内容
        scroll.setWidget(panel)
        return scroll

    def init_components(self):
        """初始化组件"""
        self.refresh_strategies()
        self._restore_last_strategy()
        self._refresh_auto_trade_pool_summary()

        if self.okx_client:
            self.connection_label.setText("● 已连接")
            self.connection_label.setStyleSheet("color: #00ff00;")
            self.trade_executor = TradeExecutor(self.okx_client)
            if hasattr(self, "monitor_pool_page") and self.monitor_pool_page:
                self.monitor_pool_page.set_trade_runtime_dependencies(
                    trade_executor=self.trade_executor,
                    trade_settings_manager=self.trade_settings_manager,
                )
            # 延迟 500ms 异步刷新，避免启动时阻塞 UI
            QTimer.singleShot(500, self._async_refresh_balance)
            QTimer.singleShot(800, self.refresh_positions)

        # 注入依赖到智能助理
        if hasattr(self, "assistant_page"):
            self.assistant_page.inject_dependencies(
                okx_client=self.okx_client,
                trade_executor=getattr(self, "trade_executor", None),
                scanner_page=getattr(self, "scanner_page", None),
            )
            # 扫描完成 → 自动推送到智能助理
            if hasattr(self, "scanner_page"):
                self.scanner_page.scan_results_ready.connect(
                    self.assistant_page.on_scan_results_received
                )

        # 扫描驱动自动交易：扫描结果路由（动态连接，由 checkbox 控制）
        # 注：信号连接在 _init_scan_auto_trader() 中动态建立，此处无需额外操作

    def _async_refresh_balance(self):
        """异步刷新余额（直接在主线程调用，由 QTimer.singleShot 延迟触发）"""
        try:
            balance = self.trade_executor.get_usdt_balance()
            self.balance_label.setText(f"{balance:.2f} USDT")
        except Exception as e:
            print(f"[余额] 获取失败: {e}")
            self.balance_label.setText("⚠️ 获取失败")

    def refresh_strategies(self):
        """刷新策略列表（目录扫描 + 手动加载的自定义策略一并显示）"""
        self.strategy_list.clear()
        # 触发目录扫描；自定义策略已在 loader 内部保留
        self.strategy_loader.discover_strategies()
        # 展示 loader 中所有策略（目录扫描 + 自定义）
        all_strategies = sorted(
            self.strategy_loader.strategies.values(),
            key=lambda s: s.name
        )
        for strategy_info in all_strategies:
            prefix = "📌 " if strategy_info.path in self.strategy_loader._custom_paths else ""
            item_text = f"{prefix}{strategy_info.name} ({strategy_info.type.value})"
            if strategy_info.description:
                item_text += f" - {strategy_info.description[:30]}"
            list_item = QListWidgetItem(item_text)
            list_item.setData(Qt.UserRole, strategy_info)
            self.strategy_list.addItem(list_item)

        self.trade_log.log(f"已发现 {len(all_strategies)} 个策略", "INFO")

    def load_custom_strategy(self):
        """加载自定义策略文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择策略文件", "",
            "Python Files (*.py);;All Files (*)"
        )

        if not file_path:
            return

        strategy_info = self.strategy_loader.load_custom_strategy(file_path)
        if strategy_info:
            self.refresh_strategies()
            # 自动选中刚加载的策略
            for i in range(self.strategy_list.count()):
                item = self.strategy_list.item(i)
                info = item.data(Qt.UserRole)
                if info and info.name == strategy_info.name:
                    self.strategy_list.setCurrentItem(item)
                    self.on_strategy_selected(item)
                    break
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

        # 持久化：保存最后选择的策略
        self._save_last_strategy(strategy_info)

    def _save_last_strategy(self, strategy_info):
        """将选中策略的名称和路径写入持久化文件"""
        try:
            data = {
                'name': strategy_info.name,
                'path': strategy_info.path,
            }
            path = os.path.normpath(self._last_strategy_file)
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[策略持久化] 保存失败: {e}")

    def _restore_last_strategy(self):
        """启动时恢复上次选择的策略"""
        try:
            path = os.path.normpath(self._last_strategy_file)
            if not os.path.exists(path):
                return
            with open(path, encoding='utf-8') as f:
                data = json.load(f)
            last_name = data.get('name', '')
            last_path = data.get('path', '')
            if not last_name:
                return

            # 若文件已不在 strategies 目录，尝试重新加载（自定义策略）
            if (last_path and last_path not in self.strategy_loader.strategies.get(last_name, type('', (), {'path': None})()).path
                    and os.path.exists(last_path)
                    and last_name not in self.strategy_loader.strategies):
                self.strategy_loader.load_custom_strategy(last_path)
                self.refresh_strategies()

            # 在列表中找到并选中
            for i in range(self.strategy_list.count()):
                item = self.strategy_list.item(i)
                info = item.data(Qt.UserRole)
                if info and info.name == last_name:
                    self.strategy_list.setCurrentItem(item)
                    self.on_strategy_selected(item)
                    self.trade_log.log(f"已恢复上次策略：{last_name}", "INFO")
                    return
        except Exception as e:
            print(f"[策略持久化] 恢复失败: {e}")

    def toggle_strategy(self):
        """一键启动/停止：策略执行 + 扫描驱动自动交易"""
        if self.is_running:
            self.stop_strategy()
        else:
            self._unified_start()

    def _unified_start(self):
        """统一启动入口：先启动策略 Runner，成功后再激活扫描驱动并立即触发扫描"""
        self.start_strategy()
        # 仅当策略启动成功（is_running=True）时才继续激活扫描驱动
        if self.is_running:
            self._init_scan_auto_trader()
            self.trade_log.log(
                "✅ 一键启动完成：策略执行 + 扫描驱动已全部激活，立即触发首次全市场扫描",
                "SUCCESS",
            )

    def start_strategy(self):
        """启动策略"""
        if not self.current_strategy:
            QMessageBox.warning(self, "警告", "请先选择策略")
            return

        # 获取配置
        config_widget = self.config_scroll.findChild(StrategyConfigWidget)
        if config_widget:
            self.strategy_config = self.trade_settings_manager.build_auto_runtime_config(config_widget.get_config())
        else:
            self.strategy_config = self.trade_settings_manager.build_auto_runtime_config({})

        inst_id = self.pair_combo.currentText()
        leverage = int(self.strategy_config.get("leverage", self.leverage_spin.value()) or self.leverage_spin.value())

        self.trade_log.log(f"启动策略：{self.current_strategy.name}", "TRADE")
        self.trade_log.log(f"交易对：{inst_id}, 杠杆：{leverage}x", "INFO")
        self.trade_log.log(
            "共享风控："
            f" 仓位 {float(self.strategy_config.get('position_size', 0.1)):.2%},"
            f" 止盈 {float(self.strategy_config.get('take_profit_pct', 5.0)):.2f}%,"
            f" 止损 {float(self.strategy_config.get('stop_loss_pct', 3.0)):.2f}%,"
            f" {'允许做空' if bool(self.strategy_config.get('allow_short', True)) else '仅做多'}",
            "INFO",
        )
        self.trade_log.log(
            "自动交易资金池："
            f" 总额 {float(self.strategy_config.get('auto_trading_capital', 1000.0)):.2f} USDT,"
            f" 试仓 {float(self.strategy_config.get('pilot_position_pct', 0.01)):.2%},"
            f" 加仓 {float(self.strategy_config.get('add_position_pct', 0.10)):.2%}",
            "INFO",
        )
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
        self.start_btn.setText("🟢 运行中（策略 + 扫描驱动）")
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_label.setText("运行中")

        # 启动策略执行线程
        self.strategy_thread = QThread()
        _paper_mode = hasattr(self, '_radio_paper') and self._radio_paper.isChecked()

        if _paper_mode:
            # ── 模拟交易模式 ───────────────────────────────────────────────
            _paper_capital = float(
                self._paper_capital_spin.value()
                if hasattr(self, '_paper_capital_spin') else 10_000.0
            )
            _cost = self.trade_settings_manager.get_all().get('cost', {})
            self._paper_engine = PaperTradeEngine(
                okx_client=self.okx_client,
                initial_capital=_paper_capital,
                fee_pct=float(_cost.get('fee_pct', 0.05) or 0.05),
                slippage_pct=float(_cost.get('slippage_pct', 0.03) or 0.03),
                market_impact_pct=float(_cost.get('market_impact_pct', 0.02) or 0.02),
                funding_rate_8h_pct=float(_cost.get('funding_rate_8h_pct', 0.01) or 0.01),
                strategy_name=self.current_strategy.name,
            )
            self.strategy_worker = PaperStrategyRunner(
                self.strategy_instance,
                inst_id,
                self.okx_client,
                self._paper_engine,
                self.strategy_config,
            )
            self.strategy_worker.paper_state_signal.connect(self._on_paper_state_update)
            self.trade_log.log(
                f"[模拟] 启动模拟交易，初始资金 {_paper_capital:,.2f} USDT",
                "INFO"
            )
        else:
            # ── 实盘交易模式 ───────────────────────────────────────────────
            if not self.trade_executor:
                self.trade_log.log("实盘交易需要先连接 OKX API", "ERROR")
                self.is_running = False
                self.start_btn.setText("🚀 一键启动自动交易")
                self.start_btn.setEnabled(True)
                self.stop_btn.setEnabled(False)
                return
            self._paper_engine = None
            self.strategy_worker = StrategyRunner(
                self.strategy_instance,
                inst_id,
                self.okx_client,
                self.trade_executor,
                self.strategy_config,
            )

        self.strategy_worker.moveToThread(self.strategy_thread)
        self.strategy_thread.started.connect(self.strategy_worker.run)
        self.strategy_worker.finished.connect(self.on_strategy_finished)
        self.strategy_worker.finished.connect(self.strategy_thread.quit)
        # ⚠️ 修复：strategy_worker.finished 是自定义 Signal，在 run() 内部 emit，
        # 此时 OS 线程仍在运行。连接 deleteLater 到自定义 finished 会导致
        # "QThread: Destroyed while thread still running" SIGABRT。
        # 必须连到 strategy_thread.finished（Qt 内置信号，OS 线程完全退出后才发出）。
        self.strategy_thread.finished.connect(self.strategy_worker.deleteLater)
        self.strategy_worker.log_signal.connect(self.on_strategy_log)
        self.strategy_worker.trade_signal.connect(self.on_strategy_trade)
        # strategy_thread 本身在 stop_strategy / closeEvent 中通过 wait() 确保完成，
        # 不用 deleteLater 自毁（避免 stop_strategy 访问已删对象）。
        # 仍保留 deleteLater 在 finished 上，Qt 会在安全时机回收。
        self.strategy_thread.finished.connect(self.strategy_thread.deleteLater)

        self.strategy_thread.start()

    def open_trade_parameter_tab(self):
        if not hasattr(self, "trade_parameter_page_container"):
            return
        index = self.main_tabs.indexOf(self.trade_parameter_page_container)
        if index >= 0:
            self.main_tabs.setCurrentIndex(index)

    def on_trade_settings_changed(self, _settings: Dict):
        common = self.trade_settings_manager.get_common_risk()
        if hasattr(self, "leverage_spin"):
            self.leverage_spin.setValue(int(common.get("leverage", 3) or 3))
        self._refresh_auto_trade_pool_summary()
        if hasattr(self, "backtest_page") and self.backtest_page:
            self.backtest_page.refresh_trade_settings_summary()
        if hasattr(self, "monitor_pool_page") and self.monitor_pool_page:
            self.monitor_pool_page.set_trade_runtime_dependencies(
                trade_executor=self.trade_executor,
                trade_settings_manager=self.trade_settings_manager,
            )
        self.trade_log.log("共享交易参数已更新，后续自动交易与回测将使用新设置。", "SUCCESS")

    def _refresh_auto_trade_pool_summary(self):
        settings = self.trade_settings_manager.get_all()
        common = settings.get("common", {})
        auto = settings.get("auto_trading", {})
        total_capital = float(auto.get("auto_trading_capital", 1000.0) or 1000.0)
        pilot_pct = float(auto.get("pilot_position_pct", 0.01) or 0.01)
        add_pct = float(auto.get("add_position_pct", 0.10) or 0.10)
        if hasattr(self, "auto_pool_label"):
            self.auto_pool_label.setText(f"{total_capital:.2f} USDT")
        if hasattr(self, "auto_pilot_label"):
            self.auto_pilot_label.setText(f"{pilot_pct:.2%} ≈ {total_capital * pilot_pct:.2f} USDT")
        if hasattr(self, "auto_add_label"):
            self.auto_add_label.setText(f"{add_pct:.2%} ≈ {total_capital * add_pct:.2f} USDT")
        if hasattr(self, "auto_risk_hint"):
            self.auto_risk_hint.setText(
                "自动交易共享风控："
                f"杠杆 {int(common.get('leverage', 3) or 3)}x，"
                f"止盈 {float(common.get('take_profit_pct', 5.0)):.2f}%，"
                f"止损 {float(common.get('stop_loss_pct', 3.0)):.2f}%，"
                f"资金池 {total_capital:.2f} USDT。"
            )

    def stop_strategy(self):
        """停止全部：策略执行 + 扫描驱动自动交易"""
        self.is_running = False
        self.start_btn.setText("🚀 一键启动自动交易")
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText("已停止")

        if hasattr(self, 'strategy_worker') and self.strategy_worker:
            try:
                self.strategy_worker.stop()
            except Exception:
                pass

        if hasattr(self, 'strategy_thread'):
            self.strategy_thread.quit()
            self.strategy_thread.wait(3000)

        # 同步停止扫描驱动
        self._destroy_scan_auto_trader()

        self.trade_log.log("⏹ 策略执行 + 扫描驱动已全部停止", "WARNING")

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

    # ── 模拟交易状态更新 ─────────────────────────────────────────────────────

    def _on_paper_state_update(self, state: dict):
        """接收 PaperStrategyRunner 发来的模拟持仓/汇总状态，刷新 UI 面板。"""
        positions = state.get('positions', [])
        summary = state.get('summary', {})
        trades = state.get('trades', [])

        # 刷新持仓表格
        if hasattr(self, '_paper_pos_table'):
            tbl = self._paper_pos_table
            tbl.setRowCount(0)
            for pos in positions:
                row = tbl.rowCount()
                tbl.insertRow(row)
                pnl = float(pos.get('unrealized_pnl', 0))
                pnl_txt = f"{pnl:+.2f}"
                items = [
                    pos.get('inst_id', ''),
                    pos.get('direction', ''),
                    f"{pos.get('entry_price', 0):.6f}",
                    f"{pos.get('current_price', 0):.6f}",
                    f"{pos.get('usdt_amount', 0):.2f}",
                    f"{pos.get('leverage', 1)}x",
                    pnl_txt,
                    f"{pos.get('tp_price', 0):.6f}",
                    f"{pos.get('sl_price', 0):.6f}",
                ]
                for col, txt in enumerate(items):
                    item = QTableWidgetItem(txt)
                    if col == 6:
                        item.setForeground(QColor('#00ff88') if pnl >= 0 else QColor('#ff4444'))
                    tbl.setItem(row, col, item)

        # 刷新成交历史表格
        if hasattr(self, '_paper_trade_table'):
            tbl = self._paper_trade_table
            tbl.setRowCount(0)
            for t in reversed(trades):
                row = tbl.rowCount()
                tbl.insertRow(row)
                pnl = float(t.get('pnl', 0))
                items = [
                    t.get('inst_id', ''),
                    t.get('direction', ''),
                    t.get('entry_time', '')[:19].replace('T', ' '),
                    t.get('exit_time', '')[:19].replace('T', ' '),
                    f"{t.get('entry_price', 0):.6f}",
                    f"{t.get('exit_price', 0):.6f}",
                    f"{t.get('usdt_amount', 0):.2f}",
                    f"{pnl:+.2f}",
                    f"{t.get('pnl_pct', 0):+.2f}%",
                    t.get('exit_reason', ''),
                ]
                for col, txt in enumerate(items):
                    item = QTableWidgetItem(txt)
                    if col == 7:
                        item.setForeground(QColor('#00ff88') if pnl >= 0 else QColor('#ff4444'))
                    tbl.setItem(row, col, item)

        # 刷新汇总标签
        if hasattr(self, '_paper_summary_label') and summary:
            balance = float(summary.get('balance', 0))
            init_cap = float(summary.get('initial_capital', balance))
            total_return = float(summary.get('total_return', 0))
            color = '#00ff88' if total_return >= 0 else '#ff4444'
            self._paper_summary_label.setText(
                f"模拟余额: <b style='color:{color}'>{balance:,.2f} USDT</b>  |  "
                f"总收益: <b style='color:{color}'>{total_return:+.2f}%</b>  |  "
                f"总交易: {summary.get('total_trades', 0)}  |  "
                f"胜率: {summary.get('win_rate', 0):.1f}%  |  "
                f"盈亏: {summary.get('total_pnl', 0):+.2f} USDT"
            )

    def refresh_data(self):
        """刷新数据"""
        self.refresh_balance()
        self.refresh_positions()

    def refresh_balance(self):
        """刷新余额（异步，不阻塞 UI）"""
        if self.trade_executor:
            self._async_refresh_balance()

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
                if hasattr(self, "mt_log"):
                    self._mt_log(f"平仓成功：{inst_id}", "SUCCESS")
                if hasattr(self, "mt_position_table"):
                    self._mt_refresh_positions()
                if hasattr(self, "mt_pending_table"):
                    self._mt_refresh_pending_orders()
                if hasattr(self, "mt_history_orders_table"):
                    self._mt_refresh_history()
            else:
                self.trade_log.log(f"平仓失败：{result.message}", "ERROR")
                if hasattr(self, "mt_log"):
                    self._mt_log(f"平仓失败：{inst_id} → {result.message}", "ERROR")

    def closeEvent(self, event):
        """统一清理后台线程，避免退出时 QThread 仍在运行。"""
        try:
            if hasattr(self, "scanner_page") and self.scanner_page:
                self.scanner_page.shutdown()
        except Exception as e:
            print(f"[WARN] scanner_page shutdown failed: {e}")

        try:
            if hasattr(self, "assistant_page") and self.assistant_page:
                self.assistant_page.close()
        except Exception as e:
            print(f"[WARN] assistant_page close failed: {e}")

        try:
            if hasattr(self, "ai_advisor_page") and self.ai_advisor_page:
                if hasattr(self.ai_advisor_page, "_stop_kline_trader"):
                    self.ai_advisor_page._stop_kline_trader()
                if hasattr(self.ai_advisor_page, "_stop_kline_monitor"):
                    self.ai_advisor_page._stop_kline_monitor()
        except Exception as e:
            print(f"[WARN] ai_advisor_page cleanup failed: {e}")

        try:
            if hasattr(self, "rl_learning_page") and self.rl_learning_page:
                if hasattr(self.rl_learning_page, "_safe_stop_trainer"):
                    self.rl_learning_page._safe_stop_trainer()
        except Exception as e:
            print(f"[WARN] rl_learning_page cleanup failed: {e}")

        try:
            if getattr(self, "is_running", False):
                self.stop_strategy()
            elif hasattr(self, "strategy_thread"):
                if hasattr(self, "strategy_worker") and self.strategy_worker:
                    try:
                        self.strategy_worker.stop()
                    except Exception:
                        pass
                self.strategy_thread.quit()
                self.strategy_thread.wait(3000)
        except Exception as e:
            print(f"[WARN] strategy thread cleanup failed: {e}")

        super().closeEvent(event)

    def create_manual_trade_tab(self) -> QWidget:
        """创建手动交易独立标签页"""
        widget = QWidget()
        main_layout = QHBoxLayout(widget)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(12)

        # ── 左侧：下单面板 ──
        left = QWidget()
        left.setMaximumWidth(380)
        left_layout = QVBoxLayout(left)
        left_layout.setSpacing(10)

        # 交易对 + 行情
        pair_group = QGroupBox("交易对")
        pair_form = QFormLayout(pair_group)
        self.mt_pair_combo = QComboBox()
        self.mt_pair_combo.setEditable(True)
        self.mt_pair_combo.addItems([
            "BTC-USDT-SWAP", "ETH-USDT-SWAP", "SOL-USDT-SWAP",
            "XRP-USDT-SWAP", "ADA-USDT-SWAP", "AVAX-USDT-SWAP",
            "DOGE-USDT-SWAP", "DOT-USDT-SWAP",
        ])
        self.mt_pair_combo.currentTextChanged.connect(self._mt_on_symbol_changed)
        _mt_pair_row = QWidget()
        _mt_pair_row_lay = QHBoxLayout(_mt_pair_row)
        _mt_pair_row_lay.setContentsMargins(0, 0, 0, 0)
        _mt_pair_row_lay.addWidget(self.mt_pair_combo, 1)
        _mt_load_btn = QPushButton("🌐 全部合约")
        _mt_load_btn.setFixedWidth(90)
        _mt_load_btn.setToolTip("从 OKX 获取所有 USDT 永续合约并填充列表")
        _mt_load_btn.clicked.connect(lambda: self._load_all_swap_pairs(self.mt_pair_combo))
        _mt_pair_row_lay.addWidget(_mt_load_btn)
        pair_form.addRow("交易对:", _mt_pair_row)
        self.mt_price_label = QLabel("最新价：--")
        self.mt_price_label.setStyleSheet("color:#00ffaa; font-weight:bold;")
        pair_form.addRow(self.mt_price_label)
        self.mt_market_hint_label = QLabel("模式：仅支持 USDT 永续合约，卖出按钮用于开空，不用于现货卖出。")
        self.mt_market_hint_label.setStyleSheet("color:#888; font-size:11px;")
        self.mt_market_hint_label.setWordWrap(True)
        pair_form.addRow(self.mt_market_hint_label)
        refresh_price_btn = QPushButton("刷新报价")
        refresh_price_btn.clicked.connect(self._mt_refresh_price)
        pair_form.addRow(refresh_price_btn)
        left_layout.addWidget(pair_group)

        # 订单参数
        order_group = QGroupBox("下单参数")
        order_form = QFormLayout(order_group)

        self.mt_order_type = QComboBox()
        self.mt_order_type.addItems(["市价单", "限价单"])
        self.mt_order_type.currentIndexChanged.connect(self._mt_on_order_type_changed)
        order_form.addRow("订单类型:", self.mt_order_type)

        # 动态自适应小数位数（根据品种 tick 大小决定）
        self.mt_limit_price = QDoubleSpinBox()
        self.mt_limit_price.setRange(0, 10_000_000)
        self.mt_limit_price.setDecimals(8)  # 默认 8 位，最低价币也够用
        self.mt_limit_price.setValue(0)
        self.mt_limit_price.setPrefix("$ ")
        self.mt_limit_price_row_label = QLabel("限价价格:")
        order_form.addRow(self.mt_limit_price_row_label, self.mt_limit_price)
        self.mt_limit_price.setVisible(False)
        self.mt_limit_price_row_label.setVisible(False)

        self.mt_size_spin = QDoubleSpinBox()
        self.mt_size_spin.setRange(1, 1_000_000)
        self.mt_size_spin.setDecimals(4)
        self.mt_size_spin.setValue(25)
        self.mt_size_spin.setPrefix("$ ")
        self.mt_size_spin.setSingleStep(10)
        order_form.addRow("投入金额:", self.mt_size_spin)

        self.mt_leverage_spin = QSpinBox()
        self.mt_leverage_spin.setRange(1, 125)
        self.mt_leverage_spin.setValue(5)
        self.mt_leverage_spin.setSuffix("x")
        order_form.addRow("杠杆:", self.mt_leverage_spin)

        self.mt_estimate_label = QLabel("订单估算：等待报价")
        self.mt_estimate_label.setWordWrap(True)
        self.mt_estimate_label.setStyleSheet("color:#b0b0b0; font-size:11px;")
        order_form.addRow(self.mt_estimate_label)

        preview_btn = QPushButton("刷新订单估算")
        preview_btn.clicked.connect(self._mt_update_estimate)
        order_form.addRow(preview_btn)

        left_layout.addWidget(order_group)

        # 止盈止损
        sl_group = QGroupBox("止盈 / 止损")
        sl_form = QFormLayout(sl_group)

        self.mt_sl_check = QCheckBox("启用止损")
        self.mt_sl_price = QDoubleSpinBox()
        self.mt_sl_price.setRange(0, 10_000_000)
        self.mt_sl_price.setDecimals(8)
        self.mt_sl_price.setPrefix("$ ")
        self.mt_sl_check.toggled.connect(self.mt_sl_price.setEnabled)
        self.mt_sl_price.setEnabled(False)
        sl_form.addRow(self.mt_sl_check, self.mt_sl_price)

        self.mt_tp_check = QCheckBox("启用止盈")
        self.mt_tp_price = QDoubleSpinBox()
        self.mt_tp_price.setRange(0, 10_000_000)
        self.mt_tp_price.setDecimals(8)
        self.mt_tp_price.setPrefix("$ ")
        self.mt_tp_check.toggled.connect(self.mt_tp_price.setEnabled)
        self.mt_tp_price.setEnabled(False)
        sl_form.addRow(self.mt_tp_check, self.mt_tp_price)

        left_layout.addWidget(sl_group)

        ladder_group = QGroupBox("Ladder 分批挂单")
        ladder_form = QFormLayout(ladder_group)
        self.mt_ladder_check = QCheckBox("启用分批限价挂单")
        ladder_form.addRow(self.mt_ladder_check)
        self.mt_ladder_steps = QSpinBox()
        self.mt_ladder_steps.setRange(2, 10)
        self.mt_ladder_steps.setValue(3)
        ladder_form.addRow("分批层数:", self.mt_ladder_steps)
        self.mt_ladder_step_pct = QDoubleSpinBox()
        self.mt_ladder_step_pct.setRange(0.05, 20.0)
        self.mt_ladder_step_pct.setDecimals(2)
        self.mt_ladder_step_pct.setValue(0.5)
        self.mt_ladder_step_pct.setSuffix(" %")
        ladder_form.addRow("层间间隔:", self.mt_ladder_step_pct)
        self.mt_ladder_info_label = QLabel("做多按基准价向下分层，做空按基准价向上分层。仅限限价单。")
        self.mt_ladder_info_label.setWordWrap(True)
        self.mt_ladder_info_label.setStyleSheet("color:#888; font-size:11px;")
        ladder_form.addRow(self.mt_ladder_info_label)
        left_layout.addWidget(ladder_group)

        # 风控：日亏损限额
        risk_group = QGroupBox("风控限额")
        risk_form = QFormLayout(risk_group)
        self.mt_max_daily_loss_spin = QDoubleSpinBox()
        self.mt_max_daily_loss_spin.setRange(0, 100_000)
        self.mt_max_daily_loss_spin.setDecimals(2)
        self.mt_max_daily_loss_spin.setValue(0)
        self.mt_max_daily_loss_spin.setSuffix(" USDT")
        self.mt_max_daily_loss_spin.setToolTip("设置为 0 表示不限制；当日亏损超过此值时禁止开仓")
        risk_form.addRow("日亏损限额:", self.mt_max_daily_loss_spin)
        left_layout.addWidget(risk_group)

        # 买入 / 卖出按钮
        btn_layout = QHBoxLayout()
        self.mt_buy_btn = QPushButton("🟢 买入 / 做多")
        self.mt_buy_btn.setMinimumHeight(48)
        self.mt_buy_btn.setStyleSheet("""
            QPushButton { background:#00aa44; color:white; border-radius:6px;
                          font-weight:bold; font-size:15px; }
            QPushButton:hover { background:#00cc55; }
        """)
        self.mt_buy_btn.clicked.connect(lambda: self._mt_place_order("buy"))
        btn_layout.addWidget(self.mt_buy_btn)

        self.mt_sell_btn = QPushButton("🔴 卖出 / 做空")
        self.mt_sell_btn.setMinimumHeight(48)
        self.mt_sell_btn.setStyleSheet("""
            QPushButton { background:#cc2222; color:white; border-radius:6px;
                          font-weight:bold; font-size:15px; }
            QPushButton:hover { background:#ee3333; }
        """)
        self.mt_sell_btn.clicked.connect(lambda: self._mt_place_order("sell"))
        btn_layout.addWidget(self.mt_sell_btn)
        left_layout.addLayout(btn_layout)

        # 一键平仓
        close_all_btn = QPushButton("⚡ 一键平仓（全部持仓）")
        close_all_btn.setMinimumHeight(36)
        close_all_btn.setStyleSheet("""
            QPushButton { background:#555; color:#ffcc00; border-radius:5px;
                          font-weight:bold; }
            QPushButton:hover { background:#777; }
        """)
        close_all_btn.clicked.connect(self._mt_close_all)
        left_layout.addWidget(close_all_btn)

        left_layout.addStretch()
        main_layout.addWidget(left)

        # ── 右侧：持仓 + 日志 ──
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setSpacing(10)

        pos_group = QGroupBox("当前持仓")
        pos_layout = QVBoxLayout(pos_group)
        self.mt_position_table = QTableWidget(0, 7)
        self.mt_position_table.setHorizontalHeaderLabels(
            ["交易对", "方向", "数量", "开仓价", "当前价", "未实现盈亏", "仓位操作"])
        self.mt_position_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.mt_position_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.mt_position_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.mt_position_table.setSelectionMode(QTableWidget.SingleSelection)
        self.mt_position_table.setStyleSheet("QTableWidget { background:#1e1e1e; color:#eee; }")
        pos_layout.addWidget(self.mt_position_table)
        pos_action_row = QHBoxLayout()
        refresh_pos_btn = QPushButton("刷新持仓")
        refresh_pos_btn.clicked.connect(self._mt_refresh_positions)
        pos_action_row.addWidget(refresh_pos_btn)
        quick_partial_btn = QPushButton("选中仓位减仓")
        quick_partial_btn.clicked.connect(self._mt_partial_close_selected)
        pos_action_row.addWidget(quick_partial_btn)
        quick_reverse_btn = QPushButton("选中仓位反手")
        quick_reverse_btn.clicked.connect(self._mt_reverse_selected)
        pos_action_row.addWidget(quick_reverse_btn)
        pos_layout.addLayout(pos_action_row)
        right_layout.addWidget(pos_group, 2)

        pending_group = QGroupBox("当前挂单")
        pending_layout = QVBoxLayout(pending_group)
        self.mt_pending_table = QTableWidget(0, 8)
        self.mt_pending_table.setHorizontalHeaderLabels(
            ["订单ID", "交易对", "方向", "类型", "数量", "委托价", "状态", "操作"]
        )
        self.mt_pending_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.mt_pending_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.mt_pending_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.mt_pending_table.setSelectionMode(QTableWidget.SingleSelection)
        self.mt_pending_table.setStyleSheet("QTableWidget { background:#171717; color:#eee; }")
        pending_layout.addWidget(self.mt_pending_table)
        pending_action_row = QHBoxLayout()
        refresh_pending_btn = QPushButton("刷新挂单")
        refresh_pending_btn.clicked.connect(self._mt_refresh_pending_orders)
        pending_action_row.addWidget(refresh_pending_btn)
        cancel_selected_btn = QPushButton("撤销选中挂单")
        cancel_selected_btn.clicked.connect(self._mt_cancel_selected_order)
        pending_action_row.addWidget(cancel_selected_btn)
        pending_layout.addLayout(pending_action_row)
        right_layout.addWidget(pending_group, 1)

        history_group = QGroupBox("历史委托 / 历史成交")
        history_layout = QVBoxLayout(history_group)
        self.mt_history_tabs = QTabWidget()

        history_orders_page = QWidget()
        history_orders_layout = QVBoxLayout(history_orders_page)
        self.mt_history_orders_table = QTableWidget(0, 8)
        self.mt_history_orders_table.setHorizontalHeaderLabels(
            ["订单ID", "交易对", "方向", "类型", "成交/委托", "委托价", "状态", "时间"]
        )
        self.mt_history_orders_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.mt_history_orders_table.setEditTriggers(QTableWidget.NoEditTriggers)
        history_orders_layout.addWidget(self.mt_history_orders_table)
        self.mt_history_tabs.addTab(history_orders_page, "历史委托")

        fills_page = QWidget()
        fills_layout = QVBoxLayout(fills_page)
        self.mt_fills_table = QTableWidget(0, 8)
        self.mt_fills_table.setHorizontalHeaderLabels(
            ["成交ID", "交易对", "方向", "成交数量", "成交价", "手续费", "已实现收益", "时间"]
        )
        self.mt_fills_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.mt_fills_table.setEditTriggers(QTableWidget.NoEditTriggers)
        fills_layout.addWidget(self.mt_fills_table)
        self.mt_history_tabs.addTab(fills_page, "历史成交")

        history_layout.addWidget(self.mt_history_tabs)
        history_action_row = QHBoxLayout()
        refresh_history_btn = QPushButton("刷新历史")
        refresh_history_btn.clicked.connect(self._mt_refresh_history)
        history_action_row.addWidget(refresh_history_btn)
        history_layout.addLayout(history_action_row)
        right_layout.addWidget(history_group, 1)

        log_group = QGroupBox("交易记录")
        log_layout = QVBoxLayout(log_group)
        self.mt_log = QTextEdit()
        self.mt_log.setReadOnly(True)
        self.mt_log.setStyleSheet(
            "QTextEdit { background:#111; color:#ccc; font-family:monospace; font-size:11px; }")
        log_layout.addWidget(self.mt_log)
        right_layout.addWidget(log_group, 1)

        main_layout.addWidget(right, 1)
        self.mt_order_type.currentIndexChanged.connect(lambda _: self._mt_update_estimate())
        self.mt_size_spin.valueChanged.connect(lambda _: self._mt_update_estimate())
        self.mt_limit_price.valueChanged.connect(lambda _: self._mt_update_estimate())
        self.mt_leverage_spin.valueChanged.connect(lambda _: self._mt_update_estimate())
        self.mt_sl_check.toggled.connect(lambda _: self._mt_update_estimate())
        self.mt_tp_check.toggled.connect(lambda _: self._mt_update_estimate())
        self.mt_sl_price.valueChanged.connect(lambda _: self._mt_update_estimate())
        self.mt_tp_price.valueChanged.connect(lambda _: self._mt_update_estimate())
        self.mt_ladder_check.toggled.connect(lambda _: self._mt_update_estimate())
        self.mt_ladder_steps.valueChanged.connect(lambda _: self._mt_update_estimate())
        self.mt_ladder_step_pct.valueChanged.connect(lambda _: self._mt_update_estimate())
        QTimer.singleShot(0, self._mt_refresh_price)
        # 仅初始化时刷新一次，不自动轮询（避免主线程阻塞 UI）
        self._mt_poll_timer = None
        return widget

    def _mt_log(self, msg: str, level: str = "INFO"):
        color = {"INFO": "#aaa", "SUCCESS": "#00ff88", "ERROR": "#ff4444",
                 "WARNING": "#ffaa00", "TRADE": "#00ccff"}.get(level, "#aaa")
        ts = datetime.now().strftime("%H:%M:%S")
        self.mt_log.append(f'<span style="color:#666">[{ts}]</span> '
                           f'<span style="color:{color}">{msg}</span>')

    def _mt_on_order_type_changed(self, idx: int):
        is_limit = idx == 1
        self.mt_limit_price.setVisible(is_limit)
        self.mt_limit_price_row_label.setVisible(is_limit)
        self._mt_update_estimate()

    def _mt_on_symbol_changed(self, _text: str):
        self._mt_refresh_price(log_result=False)
        self._mt_update_estimate()
        self._mt_refresh_pending_orders()
        self._mt_refresh_history()

    def _mt_get_inst_id(self) -> str:
        return self.mt_pair_combo.currentText().strip().upper()

    def _mt_fetch_last_price(self, inst_id: str) -> float:
        ticker = self.okx_client.get_ticker(inst_id)
        if ticker.get("code") != "0" or not ticker.get("data"):
            raise ValueError(ticker.get("msg", "未获取到报价"))
        row = ticker["data"][0]
        last = row.get("last") or row.get("lastPr") or 0
        return float(last or 0)

    def _mt_validate_symbol(self, inst_id: str) -> Optional[str]:
        if not inst_id:
            return "请输入交易对"
        if not inst_id.endswith("-SWAP"):
            return "手动交易标签页当前仅支持 USDT 永续合约，请使用 *-SWAP 交易对"
        if "-USDT-" not in inst_id:
            return "当前手动交易页仅支持 USDT 永续合约"
        return None

    def _mt_validate_risk_prices(self, action: str, entry_price: float,
                                 sl_px: Optional[float], tp_px: Optional[float]) -> Optional[str]:
        if entry_price <= 0:
            return "参考价格无效"
        if action == "buy":
            if sl_px is not None and sl_px >= entry_price:
                return "做多止损价必须低于开仓价"
            if tp_px is not None and tp_px <= entry_price:
                return "做多止盈价必须高于开仓价"
        else:
            if sl_px is not None and sl_px <= entry_price:
                return "做空止损价必须高于开仓价"
            if tp_px is not None and tp_px >= entry_price:
                return "做空止盈价必须低于开仓价"
        return None

    def _mt_price_to_pct(self, action: str, entry_price: float, target_price: float) -> float:
        if entry_price <= 0 or target_price <= 0:
            return 0.0
        if action == "buy":
            return abs(target_price - entry_price) / entry_price
        return abs(entry_price - target_price) / entry_price

    def _mt_build_ladder_prices(self, action: str, base_price: float) -> List[float]:
        if base_price <= 0:
            return []
        levels = self.mt_ladder_steps.value()
        step_pct = self.mt_ladder_step_pct.value() / 100.0
        prices: List[float] = []
        for idx in range(levels):
            price = base_price * (1 - step_pct * idx) if action == "buy" else base_price * (1 + step_pct * idx)
            if price > 0:
                prices.append(price)
        return prices

    def _mt_estimate_side_color(self, side_text: str) -> QColor:
        return QColor("#00cc66" if side_text.lower() in {"buy", "long"} else "#ff6666")

    def _mt_format_ts(self, ts: str) -> str:
        try:
            raw = str(ts or "").strip()
            if not raw:
                return "-"
            if raw.isdigit():
                stamp = int(raw)
                if stamp > 10_000_000_000:
                    stamp //= 1000
                return datetime.fromtimestamp(stamp).strftime("%m-%d %H:%M:%S")
            return raw
        except Exception:
            return str(ts or "-")

    def _mt_update_estimate(self):
        if not self.trade_executor:
            self.mt_estimate_label.setText("订单估算：未连接 OKX")
            return
        inst_id = self._mt_get_inst_id()
        symbol_error = self._mt_validate_symbol(inst_id)
        if symbol_error:
            self.mt_estimate_label.setText(f"订单估算：{symbol_error}")
            return
        usdt_amount = self.mt_size_spin.value()
        is_limit = self.mt_order_type.currentIndex() == 1
        limit_px = self.mt_limit_price.value() if is_limit else None
        leverage = self.mt_leverage_spin.value()
        try:
            estimate = self.trade_executor.estimate_order(
                inst_id,
                usdt_amount,
                price=limit_px if is_limit and limit_px > 0 else None,
                leverage=leverage,
            )
            if not estimate.get("success"):
                self.mt_estimate_label.setText(f"订单估算：{estimate.get('message', '失败')}")
                return
            market_price = float(estimate.get("market_price") or 0)
            reference_price = float(estimate.get("reference_price") or 0)
            size = float(estimate.get("size") or 0)
            notional = float(estimate.get("estimated_notional") or 0)
            margin = float(estimate.get("estimated_margin") or 0)
            fee = float(estimate.get("estimated_fee") or 0)
            self.mt_estimate_label.setText(
                f"订单估算：现价 {market_price:.6f} | 参考价 {reference_price:.6f} | "
                f"预计数量 {size:.6f} | 名义价值 {notional:.2f} USDT | "
                f"保证金 {margin:.2f} USDT | 手续费约 {fee:.4f} USDT"
            )
            if self.mt_ladder_check.isChecked() and is_limit:
                ladder_prices = self._mt_build_ladder_prices("buy", reference_price)
                if ladder_prices:
                    self.mt_estimate_label.setText(
                        self.mt_estimate_label.text() +
                        f" | Ladder {len(ladder_prices)} 层"
                    )
        except Exception as e:
            self.mt_estimate_label.setText(f"订单估算：{e}")

    def _mt_get_tpsl_pct(self, action: str, reference_price: float) -> tuple:
        sl_px = self.mt_sl_price.value() if self.mt_sl_check.isChecked() else None
        tp_px = self.mt_tp_price.value() if self.mt_tp_check.isChecked() else None
        risk_error = self._mt_validate_risk_prices(action, reference_price, sl_px, tp_px)
        if risk_error:
            raise ValueError(risk_error)
        tp_pct = self._mt_price_to_pct(action, reference_price, tp_px) if tp_px else 0.05
        sl_pct = self._mt_price_to_pct(action, reference_price, sl_px) if sl_px else 0.03
        return tp_pct, sl_pct

    def _mt_refresh_price(self, log_result: bool = True):
        if not self.okx_client:
            self._mt_log("未连接 OKX", "ERROR")
            return
        inst_id = self._mt_get_inst_id()
        symbol_error = self._mt_validate_symbol(inst_id)
        if symbol_error:
            self.mt_price_label.setText("最新价：--")
            self.mt_estimate_label.setText(f"订单估算：{symbol_error}")
            if log_result:
                self._mt_log(symbol_error, "WARNING")
            return
        try:
            price = self._mt_fetch_last_price(inst_id)
            self.mt_price_label.setText(f"最新价：{price}")
            if self.mt_order_type.currentIndex() == 1 and self.mt_limit_price.value() <= 0:
                self.mt_limit_price.setValue(price)
            self._mt_update_estimate()
            if log_result:
                self._mt_log(f"{inst_id} 最新价：{price}", "INFO")
        except Exception as e:
            self._mt_log(f"获取报价失败：{e}", "ERROR")

    def _mt_refresh_pending_orders(self):
        if not self.trade_executor:
            self._mt_log("未连接 OKX", "ERROR")
            return
        try:
            inst_id = self._mt_get_inst_id()
            symbol_error = self._mt_validate_symbol(inst_id)
            query_inst = inst_id if not symbol_error else None
            orders = self.trade_executor.get_pending_orders(query_inst)
            self.mt_pending_table.setRowCount(0)
            for order in orders:
                row = self.mt_pending_table.rowCount()
                self.mt_pending_table.insertRow(row)
                ord_id = str(order.get("ordId") or order.get("algoId") or "-")
                inst = str(order.get("instId") or "-")
                side = str(order.get("side") or "-")
                ord_type = str(order.get("ordType") or "-")
                size = str(order.get("sz") or order.get("accFillSz") or "-")
                px = str(order.get("px") or order.get("fillPx") or "市价")
                state = str(order.get("state") or "-")
                self.mt_pending_table.setItem(row, 0, QTableWidgetItem(ord_id))
                self.mt_pending_table.setItem(row, 1, QTableWidgetItem(inst))
                side_item = QTableWidgetItem(side)
                side_item.setForeground(self._mt_estimate_side_color(side))
                self.mt_pending_table.setItem(row, 2, side_item)
                self.mt_pending_table.setItem(row, 3, QTableWidgetItem(ord_type))
                self.mt_pending_table.setItem(row, 4, QTableWidgetItem(size))
                self.mt_pending_table.setItem(row, 5, QTableWidgetItem(px))
                self.mt_pending_table.setItem(row, 6, QTableWidgetItem(state))
                cancel_btn = QPushButton("撤单")
                cancel_btn.setStyleSheet("QPushButton{background:#6a2d2d;color:white;border-radius:3px;}")
                cancel_btn.clicked.connect(lambda _, i=inst, o=ord_id: self._mt_cancel_order(i, o))
                self.mt_pending_table.setCellWidget(row, 7, cancel_btn)
            self._mt_log(f"挂单已刷新，共 {len(orders)} 条", "INFO")
        except Exception as e:
            self._mt_log(f"刷新挂单失败：{e}", "ERROR")

    def _mt_refresh_history(self):
        if not self.trade_executor:
            self._mt_log("未连接 OKX", "ERROR")
            return
        try:
            inst_id = self._mt_get_inst_id()
            symbol_error = self._mt_validate_symbol(inst_id)
            query_inst = inst_id if not symbol_error else None

            orders = self.trade_executor.get_order_history(query_inst, limit=50)
            self.mt_history_orders_table.setRowCount(0)
            for order in orders:
                row = self.mt_history_orders_table.rowCount()
                self.mt_history_orders_table.insertRow(row)
                ord_id = str(order.get("ordId") or "-")
                inst = str(order.get("instId") or "-")
                side = str(order.get("side") or "-")
                ord_type = str(order.get("ordType") or "-")
                fill_ratio = f"{order.get('accFillSz') or '0'}/{order.get('sz') or '-'}"
                px = str(order.get("px") or "市价")
                state = str(order.get("state") or "-")
                ts = self._mt_format_ts(order.get("cTime") or order.get("uTime"))
                self.mt_history_orders_table.setItem(row, 0, QTableWidgetItem(ord_id))
                self.mt_history_orders_table.setItem(row, 1, QTableWidgetItem(inst))
                side_item = QTableWidgetItem(side)
                side_item.setForeground(self._mt_estimate_side_color(side))
                self.mt_history_orders_table.setItem(row, 2, side_item)
                self.mt_history_orders_table.setItem(row, 3, QTableWidgetItem(ord_type))
                self.mt_history_orders_table.setItem(row, 4, QTableWidgetItem(fill_ratio))
                self.mt_history_orders_table.setItem(row, 5, QTableWidgetItem(px))
                self.mt_history_orders_table.setItem(row, 6, QTableWidgetItem(state))
                self.mt_history_orders_table.setItem(row, 7, QTableWidgetItem(ts))

            fills = self.trade_executor.get_fill_history(query_inst, limit=50)
            self.mt_fills_table.setRowCount(0)
            for fill in fills:
                row = self.mt_fills_table.rowCount()
                self.mt_fills_table.insertRow(row)
                fill_id = str(fill.get("tradeId") or fill.get("billId") or "-")
                inst = str(fill.get("instId") or "-")
                side = str(fill.get("side") or "-")
                fill_sz = str(fill.get("fillSz") or fill.get("sz") or "-")
                fill_px = str(fill.get("fillPx") or fill.get("px") or "-")
                fee = str(fill.get("fee") or fill.get("fillFee") or "-")
                pnl = str(fill.get("fillPnl") or fill.get("pnl") or "-")
                ts = self._mt_format_ts(fill.get("ts") or fill.get("fillTime"))
                self.mt_fills_table.setItem(row, 0, QTableWidgetItem(fill_id))
                self.mt_fills_table.setItem(row, 1, QTableWidgetItem(inst))
                side_item = QTableWidgetItem(side)
                side_item.setForeground(self._mt_estimate_side_color(side))
                self.mt_fills_table.setItem(row, 2, side_item)
                self.mt_fills_table.setItem(row, 3, QTableWidgetItem(fill_sz))
                self.mt_fills_table.setItem(row, 4, QTableWidgetItem(fill_px))
                self.mt_fills_table.setItem(row, 5, QTableWidgetItem(fee))
                pnl_item = QTableWidgetItem(pnl)
                try:
                    pnl_value = float(pnl)
                    pnl_item.setForeground(QColor("#00cc66" if pnl_value >= 0 else "#ff6666"))
                except Exception:
                    pass
                self.mt_fills_table.setItem(row, 6, pnl_item)
                self.mt_fills_table.setItem(row, 7, QTableWidgetItem(ts))

            self._mt_log(f"历史委托 {len(orders)} 条，历史成交 {len(fills)} 条", "INFO")
        except Exception as e:
            self._mt_log(f"刷新历史失败：{e}", "ERROR")

    def _mt_poll_refresh(self):
        """定时刷新持仓、挂单、历史（非阻塞）"""
        try:
            self._mt_refresh_positions()
            self._mt_refresh_pending_orders()
        except Exception:
            pass  # 静默处理，避免定时器崩溃

    def _mt_refresh_positions(self):
        if not self.trade_executor:
            self._mt_log("未连接 OKX", "ERROR")
            return
        try:
            positions = self.trade_executor.get_positions()
            self.mt_position_table.setRowCount(0)
            for inst_id, pos in positions.items():
                row = self.mt_position_table.rowCount()
                self.mt_position_table.insertRow(row)
                side = getattr(pos, "side", "--")
                size = getattr(pos, "size", "--")
                avg_px = getattr(pos, "entry_price", "--")
                last_px = getattr(pos, "current_price", "--")
                upl = getattr(pos, "unrealized_pnl", "--")
                self.mt_position_table.setItem(row, 0, QTableWidgetItem(inst_id))
                side_item = QTableWidgetItem(getattr(side, "value", str(side)))
                side_text = side_item.text().lower()
                if side_text == "long":
                    side_item.setForeground(QColor("#00cc66"))
                elif side_text == "short":
                    side_item.setForeground(QColor("#ff6666"))
                self.mt_position_table.setItem(row, 1, side_item)
                self.mt_position_table.setItem(row, 2, QTableWidgetItem(f"{float(size):.6f}" if isinstance(size, (int, float)) else str(size)))
                self.mt_position_table.setItem(row, 3, QTableWidgetItem(f"{float(avg_px):.6f}" if isinstance(avg_px, (int, float)) else str(avg_px)))
                self.mt_position_table.setItem(row, 4, QTableWidgetItem(f"{float(last_px):.6f}" if isinstance(last_px, (int, float)) else str(last_px)))
                upl_item = QTableWidgetItem(f"{float(upl):.4f}" if isinstance(upl, (int, float)) else str(upl))
                if isinstance(upl, (int, float)):
                    upl_item.setForeground(QColor("#00cc66" if float(upl) >= 0 else "#ff6666"))
                self.mt_position_table.setItem(row, 5, upl_item)
                action_widget = QWidget()
                action_layout = QHBoxLayout(action_widget)
                action_layout.setContentsMargins(2, 2, 2, 2)
                action_layout.setSpacing(4)
                reduce_btn = QPushButton("减仓")
                reduce_btn.setStyleSheet("QPushButton{background:#805d20;color:white;border-radius:3px;}")
                reduce_btn.clicked.connect(lambda _, i=inst_id: self._mt_partial_close_position(i))
                action_layout.addWidget(reduce_btn)
                reverse_btn = QPushButton("反手")
                reverse_btn.setStyleSheet("QPushButton{background:#245c85;color:white;border-radius:3px;}")
                reverse_btn.clicked.connect(lambda _, i=inst_id: self._mt_reverse_position(i))
                action_layout.addWidget(reverse_btn)
                close_btn = QPushButton("平仓")
                close_btn.setStyleSheet("QPushButton{background:#aa3300;color:white;border-radius:3px;}")
                close_btn.clicked.connect(lambda _, i=inst_id: self.close_position(i))
                action_layout.addWidget(close_btn)
                self.mt_position_table.setCellWidget(row, 6, action_widget)
            self._mt_log(f"持仓已刷新，共 {len(positions)} 条", "INFO")
        except Exception as e:
            self._mt_log(f"获取持仓失败：{e}", "ERROR")

    def _mt_get_selected_position_inst_id(self) -> Optional[str]:
        row = self._mt_get_selected_row(self.mt_position_table)
        if row is None:
            return None
        item = self.mt_position_table.item(row, 0)
        if item is None:
            return None
        return item.text().strip()

    def _mt_get_selected_pending_order(self) -> Optional[tuple]:
        row = self._mt_get_selected_row(self.mt_pending_table)
        if row is None:
            return None
        inst_item = self.mt_pending_table.item(row, 1)
        ord_item = self.mt_pending_table.item(row, 0)
        if inst_item is None or ord_item is None:
            return None
        return inst_item.text().strip(), ord_item.text().strip()

    def _mt_get_selected_row(self, table: QTableWidget) -> Optional[int]:
        try:
            selection_model = table.selectionModel()
            if selection_model is not None:
                rows = selection_model.selectedRows()
                if rows:
                    return rows[0].row()
        except Exception:
            pass
        try:
            items = table.selectedItems()
            if items:
                return items[0].row()
        except Exception:
            pass
        row = table.currentRow()
        return row if row >= 0 else None

    def _mt_partial_close_selected(self):
        inst_id = self._mt_get_selected_position_inst_id()
        if not inst_id:
            self._mt_log("请先选中一条持仓再执行减仓", "WARNING")
            return
        self._mt_partial_close_position(inst_id)

    def _mt_reverse_selected(self):
        inst_id = self._mt_get_selected_position_inst_id()
        if not inst_id:
            self._mt_log("请先选中一条持仓再执行反手", "WARNING")
            return
        self._mt_reverse_position(inst_id)

    def _mt_partial_close_position(self, inst_id: str):
        if not self.trade_executor:
            self._mt_log("请先连接 OKX", "ERROR")
            return
        ratio_pct, ok = QInputDialog.getDouble(
            self,
            "部分平仓",
            f"{inst_id} 减仓比例（%）",
            50.0,
            1.0,
            100.0,
            1
        )
        if not ok:
            self._mt_log("用户取消减仓", "WARNING")
            return
        ratio = ratio_pct / 100.0
        reply = QMessageBox.question(
            self,
            "确认减仓",
            f"确认对 {inst_id} 执行 {ratio_pct:.1f}% 的部分平仓吗？",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            self._mt_log("用户取消减仓确认", "WARNING")
            return
        try:
            result = self.trade_executor.close_position_partial(inst_id, ratio)
            if result.success:
                self._mt_log(
                    f"{inst_id} 部分平仓成功，减仓比例 {ratio_pct:.1f}%，成交数量={result.filled_size}",
                    "SUCCESS"
                )
                self._mt_refresh_positions()
                self._mt_refresh_pending_orders()
                self._mt_refresh_history()
            else:
                self._mt_log(f"{inst_id} 部分平仓失败：{result.message}", "ERROR")
        except Exception as e:
            self._mt_log(f"{inst_id} 部分平仓异常：{e}", "ERROR")

    def _mt_reverse_position(self, inst_id: str):
        if not self.trade_executor:
            self._mt_log("请先连接 OKX", "ERROR")
            return
        usdt_amount, ok = QInputDialog.getDouble(
            self,
            "一键反手",
            f"{inst_id} 反手后新仓投入金额（USDT）",
            float(self.mt_size_spin.value()),
            1.0,
            1_000_000.0,
            2
        )
        if not ok:
            self._mt_log("用户取消反手", "WARNING")
            return
        is_limit = self.mt_order_type.currentIndex() == 1
        limit_px = self.mt_limit_price.value() if is_limit else None
        leverage = self.mt_leverage_spin.value()
        try:
            reference_price = limit_px if is_limit and limit_px else self._mt_fetch_last_price(inst_id)
            pos = self.trade_executor.get_positions(inst_id).get(inst_id)
            if not pos:
                self._mt_log(f"{inst_id} 当前无持仓，无法反手", "WARNING")
                return
            next_action = "sell" if pos.side == PositionSide.LONG else "buy"
            tp_pct, sl_pct = self._mt_get_tpsl_pct(next_action, reference_price)
            reply = QMessageBox.question(
                self,
                "确认反手",
                f"确认反手 {inst_id} 吗？\n\n将先平掉当前仓位，再以 {usdt_amount:.2f} USDT 开立反向仓位。",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                self._mt_log("用户取消反手确认", "WARNING")
                return
            result = self.trade_executor.reverse_position(
                inst_id,
                usdt_amount=usdt_amount,
                leverage=leverage,
                tp_pct=tp_pct,
                sl_pct=sl_pct,
                order_type="limit" if is_limit else "market",
                price=limit_px if is_limit else None,
            )
            if result.success:
                self._mt_log(f"{inst_id} 反手成功，成交数量={result.filled_size}", "SUCCESS")
                if result.message:
                    self._mt_log(result.message, "INFO")
                self._mt_refresh_positions()
                self._mt_refresh_pending_orders()
                self._mt_refresh_history()
            else:
                self._mt_log(f"{inst_id} 反手失败：{result.message}", "ERROR")
        except Exception as e:
            self._mt_log(f"{inst_id} 反手异常：{e}", "ERROR")

    def _mt_cancel_selected_order(self):
        selected = self._mt_get_selected_pending_order()
        if not selected:
            self._mt_log("请先选中一条挂单再撤销", "WARNING")
            return
        inst_id, ord_id = selected
        self._mt_cancel_order(inst_id, ord_id)

    def _mt_cancel_order(self, inst_id: str, ord_id: str):
        if not self.trade_executor:
            self._mt_log("请先连接 OKX", "ERROR")
            return
        reply = QMessageBox.question(
            self,
            "确认撤单",
            f"确认撤销挂单？\n\n订单ID: {ord_id}\n交易对: {inst_id}",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            self._mt_log("用户取消撤单", "WARNING")
            return
        try:
            result = self.trade_executor.cancel_order(inst_id, ord_id)
            if result.success:
                self._mt_log(f"撤单成功：{inst_id} / {ord_id}", "SUCCESS")
                self._mt_refresh_pending_orders()
                self._mt_refresh_history()
            else:
                self._mt_log(f"撤单失败：{inst_id} / {ord_id} → {result.message}", "ERROR")
        except Exception as e:
            self._mt_log(f"撤单异常：{e}", "ERROR")

    def _mt_reset_daily_loss_if_needed(self):
        """每天首次下单时重置当日亏损计数器"""
        today = datetime.now().strftime("%Y-%m-%d")
        if self._mt_daily_reset_date != today:
            self._mt_daily_loss = 0.0
            self._mt_daily_reset_date = today

    def _mt_record_trade_pnl(self, pnl: float):
        """记录一笔成交的盈亏到当日亏损（pnl 负数=亏损）"""
        if pnl < 0:
            self._mt_daily_loss += abs(pnl)

    def _mt_place_order(self, action: str):
        if not self.trade_executor:
            self._mt_log("请先连接 OKX", "ERROR")
            return

        # 日亏损熔断检查
        self._mt_reset_daily_loss_if_needed()
        try:
            _max_daily = float(getattr(self, 'mt_max_daily_loss_spin', None) and
                               self.mt_max_daily_loss_spin.value() or 0)
        except Exception:
            _max_daily = 0.0
        if _max_daily > 0 and self._mt_daily_loss >= _max_daily:
            self._mt_log(
                f"日亏损熔断触发：当日已亏 {self._mt_daily_loss:.2f} USDT，"
                f"超过限额 {_max_daily:.2f} USDT，今日禁止开仓",
                "ERROR"
            )
            return

        inst_id = self._mt_get_inst_id()
        symbol_error = self._mt_validate_symbol(inst_id)
        if symbol_error:
            self._mt_log(symbol_error, "ERROR")
            return
        usdt_amount = self.mt_size_spin.value()
        is_limit = self.mt_order_type.currentIndex() == 1
        limit_px = self.mt_limit_price.value() if is_limit else None
        sl_px = self.mt_sl_price.value() if self.mt_sl_check.isChecked() else None
        tp_px = self.mt_tp_price.value() if self.mt_tp_check.isChecked() else None
        leverage = self.mt_leverage_spin.value()
        if usdt_amount <= 0:
            self._mt_log("投入金额必须大于 0", "ERROR")
            return
        if is_limit and (limit_px is None or limit_px <= 0):
            self._mt_log("限价单必须填写有效价格", "ERROR")
            return
        try:
            reference_price = limit_px if is_limit and limit_px else self._mt_fetch_last_price(inst_id)
        except Exception as e:
            self._mt_log(f"下单前获取价格失败：{e}", "ERROR")
            return
        risk_error = self._mt_validate_risk_prices(action, reference_price, sl_px, tp_px)
        if risk_error:
            self._mt_log(risk_error, "ERROR")
            return
        ladder_enabled = self.mt_ladder_check.isChecked()
        if ladder_enabled and not is_limit:
            self._mt_log("Ladder 分批挂单仅支持限价单", "ERROR")
            return

        # --- 估算阶段（放后台线程，避免网络调用阻塞 UI）---
        self._mt_log(f"正在估算订单…", "INFO")
        self.mt_buy_btn.setEnabled(False)
        self.mt_sell_btn.setEnabled(False)

        executor = self.trade_executor
        order_params = {
            'inst_id': inst_id, 'action': action, 'usdt_amount': usdt_amount,
            'is_limit': is_limit, 'limit_px': limit_px, 'sl_px': sl_px, 'tp_px': tp_px,
            'leverage': leverage, 'reference_price': reference_price,
            'ladder_enabled': ladder_enabled,
        }

        def _do_estimate():
            return executor.estimate_order(
                inst_id, usdt_amount,
                price=limit_px if is_limit else None,
                leverage=leverage,
            )

        def _on_estimate_done(estimate):
            self.mt_buy_btn.setEnabled(True)
            self.mt_sell_btn.setEnabled(True)
            if not estimate.get("success"):
                self._mt_log(f"订单估算失败：{estimate.get('message', '未知原因')}", "ERROR")
                return
            estimated_size = float(estimate.get("size") or 0)
            estimated_margin = float(estimate.get("estimated_margin") or 0)
            estimated_notional = float(estimate.get("estimated_notional") or 0)
            if estimated_size <= 0:
                self._mt_log("下单数量不足最小下单单位，请提高投入金额", "ERROR")
                return

            # 改进余额检查：扣除当前持仓已占用保证金
            try:
                balance = executor.get_usdt_balance()
                open_positions = executor.get_positions() or {}
                occupied_margin = sum(
                    float(p.get('margin', 0) or 0)
                    for p in open_positions.values()
                    if isinstance(p, dict)
                )
                available = max(0.0, balance - occupied_margin)
            except Exception:
                balance = 0.0
                available = 0.0
            if balance > 0 and estimated_margin > available * 1.02:
                self._mt_log(
                    f"可用保证金不足：预计占用 {estimated_margin:.2f} USDT，"
                    f"可用余额 {available:.2f} USDT（总权益 {balance:.2f} USDT，"
                    f"已占用 {balance - available:.2f} USDT）",
                    "ERROR"
                )
                return

            confirm_text = (
                f"确认提交订单？\n\n"
                f"交易对: {inst_id}\n"
                f"方向: {'做多' if action == 'buy' else '做空'}\n"
                f"类型: {'限价单' if is_limit else '市价单'}\n"
                f"投入金额: {usdt_amount:.2f} USDT\n"
                f"杠杆: {leverage}x\n"
                f"参考价格: {reference_price:.6f}\n"
                f"预计数量: {estimated_size:.6f}\n"
                f"名义价值: {estimated_notional:.2f} USDT\n"
                f"预计保证金: {estimated_margin:.2f} USDT\n"
                f"可用余额: {available:.2f} USDT"
            )
            if tp_px:
                confirm_text += f"\n止盈: {tp_px:.6g}"
            if sl_px:
                confirm_text += f"\n止损: {sl_px:.6g}"
            if not tp_px and not sl_px:
                confirm_text += "\n(无止盈止损设置)"
            reply = QMessageBox.question(
                self, "确认下单", confirm_text, QMessageBox.Yes | QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                self._mt_log("用户取消下单", "WARNING")
                return

            tp_pct = self._mt_price_to_pct(action, reference_price, tp_px) if tp_px else None
            sl_pct = self._mt_price_to_pct(action, reference_price, sl_px) if sl_px else None
            direction = "LONG" if action == "buy" else "SHORT"

            # --- 下单阶段（同样放后台线程）---
            self.mt_buy_btn.setEnabled(False)
            self.mt_sell_btn.setEnabled(False)

            def _do_execute():
                if ladder_enabled:
                    ladder_prices = self._mt_build_ladder_prices(
                        action, limit_px if limit_px else reference_price
                    )
                    return ('ladder', ladder_prices, usdt_amount, direction, leverage, tp_pct, sl_pct)
                else:
                    res = executor.execute_entry(
                        inst_id, direction,
                        usdt_amount=usdt_amount, leverage=leverage,
                        tp_pct=tp_pct, sl_pct=sl_pct,
                        order_type="limit" if is_limit else "market",
                        price=limit_px if is_limit else None,
                    )
                    return ('single', res)

            def _on_execute_done(payload):
                self.mt_buy_btn.setEnabled(True)
                self.mt_sell_btn.setEnabled(True)
                kind = payload[0]
                if kind == 'single':
                    res = payload[1]
                    if res.success:
                        self._mt_log(
                            f"{'做多' if action == 'buy' else '做空'}成功，"
                            f"委托ID={res.order_id or '-'}，成交数量={res.filled_size}",
                            "SUCCESS"
                        )
                        if res.message:
                            self._mt_log(res.message, "INFO")
                        pnl = float(getattr(res, 'pnl', 0.0) or 0.0)
                        self._mt_record_trade_pnl(pnl)
                    else:
                        self._mt_log(f"下单失败：{res.message}", "ERROR")
                elif kind == 'ladder':
                    _, ladder_prices, per_amount_total, _dir, _lev, _tp, _sl = payload
                    if len(ladder_prices) < 2:
                        self._mt_log("Ladder 价格层无效，请检查限价和层间间隔", "ERROR")
                        return
                    per_order_amount = per_amount_total / len(ladder_prices)
                    success_count = 0
                    for idx2, ladder_price in enumerate(ladder_prices, 1):
                        r2 = executor.execute_entry(
                            inst_id, _dir,
                            usdt_amount=per_order_amount, leverage=_lev,
                            tp_pct=_tp, sl_pct=_sl,
                            order_type="limit", price=ladder_price,
                        )
                        if r2.success:
                            success_count += 1
                            self._mt_log(
                                f"Ladder 第 {idx2}/{len(ladder_prices)} 层下单成功，"
                                f"价位={ladder_price:.6f}，订单ID={r2.order_id or '-'}",
                                "SUCCESS"
                            )
                        else:
                            self._mt_log(
                                f"Ladder 第 {idx2}/{len(ladder_prices)} 层失败，"
                                f"价位={ladder_price:.6f}：{r2.message}",
                                "ERROR"
                            )
                    if success_count > 0:
                        self._mt_log(
                            f"Ladder 提交完成：成功 {success_count}/{len(ladder_prices)} 层，"
                            f"每层约 {per_order_amount:.2f} USDT",
                            "INFO"
                        )
                    else:
                        self._mt_log("Ladder 挂单全部失败", "ERROR")
                self._mt_refresh_positions()
                self._mt_refresh_pending_orders()
                self._mt_refresh_history()

            self._mt_run_in_thread(_do_execute, _on_execute_done)

        self._mt_run_in_thread(_do_estimate, _on_estimate_done)

    def _on_all_contracts_toggled(self, checked: bool):
        """全部合约 Toggle：选中时加载合约列表并高亮；同步通知扫描驱动面板。"""
        if checked:
            # 加载全部合约到 combo
            self._load_all_swap_pairs(self.pair_combo)
            if hasattr(self, 'trade_log'):
                self.trade_log.log(
                    "✅ 全部合约模式已开启：扫描驱动将覆盖交易所全部 USDT 永续合约", "SUCCESS"
                )
            # 更新扫描驱动面板提示
            if hasattr(self, '_sd_strategy_label'):
                self._sd_strategy_label.setStyleSheet("color:#44ccff; font-size:12px;")
        else:
            if hasattr(self, 'trade_log'):
                self.trade_log.log("全部合约模式已关闭", "INFO")

    def _load_all_swap_pairs(self, combo: QComboBox):
        """异步从 OKX 获取全部 USDT 永续合约，按成交额排序后填入 combo。"""
        if not self.okx_client:
            QMessageBox.warning(self, "提示", "需要先连接 OKX API 才能加载合约列表")
            return

        current_text = combo.currentText()
        combo.setEnabled(False)
        combo.setToolTip("正在从 OKX 加载合约列表，请稍候…")

        okx = self.okx_client

        def _fetch():
            try:
                result = okx.get_tickers(instType="SWAP")
                if result and result.get('code') == '0':
                    tickers = result.get('data', [])
                    swaps = [t for t in tickers
                             if t.get('instId', '').endswith('-USDT-SWAP')]
                    swaps.sort(
                        key=lambda t: float(t.get('volCcyQuote') or t.get('vol24h') or 0),
                        reverse=True,
                    )
                    return [t['instId'] for t in swaps]
            except Exception as exc:
                print(f"[加载合约] 异常: {exc}")
            return []

        class _FetchWorker(QThread):
            done = pyqtSignal(object)
            def __init__(self, f):
                super().__init__()
                self._f = f
            def run(self):
                self.done.emit(self._f())

        worker = _FetchWorker(_fetch)

        def _on_done(inst_ids):
            combo.setEnabled(True)
            combo.setToolTip("")
            if inst_ids:
                combo.clear()
                combo.addItems(inst_ids)
                idx = combo.findText(current_text)
                combo.setCurrentIndex(idx if idx >= 0 else 0)
                msg = f"已加载 {len(inst_ids)} 个 USDT 永续合约"
                level = "SUCCESS"
            else:
                if current_text:
                    combo.setEditText(current_text)
                msg = "加载合约列表失败，请检查 API 连接"
                level = "ERROR"
            if hasattr(self, 'trade_log'):
                self.trade_log.log(msg, level)

        worker.done.connect(_on_done)
        # 必须用 finished 而不是 done：finished 在 OS 线程退出后由 Qt 发出，
        # 此时 deleteLater 才安全；done 在 run() 内部发出，线程仍在运行，
        # 用 done 连接 deleteLater 会导致 QThread::~QThread() 时 abort()。
        worker.finished.connect(worker.deleteLater)
        worker.start()
        # 持有引用，防止被 GC
        if not hasattr(self, '_pair_load_workers'):
            self._pair_load_workers = []
        self._pair_load_workers.append(worker)
        # 线程结束后从列表移除，避免内存泄漏（4GB 机器上尤其重要）
        worker.finished.connect(
            lambda w=worker: self._pair_load_workers.remove(w)
            if w in self._pair_load_workers else None
        )

    def _mt_run_in_thread(self, fn, callback):
        """在 QThread 中执行 fn()，完成后在主线程调用 callback(result)。"""
        class _Worker(QThread):
            done = pyqtSignal(object)
            def __init__(self, f):
                super().__init__()
                self._f = f
            def run(self):
                try:
                    self.done.emit(self._f())
                except Exception as exc:
                    self.done.emit({'_thread_error': str(exc)})

        worker = _Worker(fn)

        def _on_done(result):
            if isinstance(result, dict) and '_thread_error' in result:
                self._mt_log(f"下单异常：{result['_thread_error']}", "ERROR")
                self.mt_buy_btn.setEnabled(True)
                self.mt_sell_btn.setEnabled(True)
                return
            callback(result)

        worker.done.connect(_on_done)
        # finished 在 OS 线程退出后发出，此时 deleteLater 才安全
        worker.finished.connect(worker.deleteLater)
        # 用列表而不是单个引用：单个引用会覆盖上一个 worker，若上一个还在运行则被 GC 掉
        if not hasattr(self, '_mt_worker_pool'):
            self._mt_worker_pool = []
        self._mt_worker_pool.append(worker)
        worker.finished.connect(
            lambda w=worker: self._mt_worker_pool.remove(w)
            if w in self._mt_worker_pool else None
        )
        self._mt_order_thread = worker  # 保留向后兼容引用
        worker.start()

    def _mt_close_all(self):
        if not self.trade_executor:
            self._mt_log("请先连接 OKX", "ERROR")
            return
        reply = QMessageBox.question(self, "确认平仓", "确定要平掉所有持仓吗？",
                                     QMessageBox.Yes | QMessageBox.No)
        if reply != QMessageBox.Yes:
            return
        try:
            positions = self.trade_executor.get_positions()
            if not positions:
                self._mt_log("当前无持仓", "WARNING")
                return
            for inst_id in list(positions.keys()):
                result = self.trade_executor.execute_stop_loss(inst_id)
                if result.success:
                    self._mt_log(f"平仓成功：{inst_id}", "SUCCESS")
                else:
                    self._mt_log(f"平仓失败：{inst_id} → {result.message}", "ERROR")
            self._mt_refresh_positions()
            self._mt_refresh_pending_orders()
            self._mt_refresh_history()
        except Exception as e:
            self._mt_log(f"一键平仓异常：{e}", "ERROR")

    def manual_trade(self, action: str):
        """手动交易（已废弃——请使用手动交易标签页的 _mt_place_order）"""
        self.trade_log.log("请使用手动交易标签页操作", "WARNING")


# StrategyWorker 已替换为 StrategyRunner


def _load_okx_client() -> OKXClient:
    """从 .env 文件读取 OKX 配置，构建客户端"""
    import os
    from pathlib import Path

    # 找项目根目录下的 .env
    env_path = Path(__file__).resolve().parent.parent.parent / ".env"
    env = {}
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                env[k.strip()] = v.strip()

    api_key    = env.get("OKX_API_KEY")    or os.getenv("OKX_API_KEY", "")
    secret_key = env.get("OKX_SECRET_KEY") or os.getenv("OKX_SECRET_KEY", "")
    passphrase = env.get("OKX_PASSPHRASE") or os.getenv("OKX_PASSPHRASE", "")
    testnet    = (env.get("OKX_TESTNET")   or os.getenv("OKX_TESTNET", "false")).lower() == "true"
    proxy_url  = env.get("OKX_PROXY_URL")  or os.getenv("OKX_PROXY_URL", "") or None

    if not api_key:
        print("[WARN] 未找到 OKX_API_KEY，程序将以无凭证模式运行")
        return None

    print(f"[OKX] testnet={testnet}, proxy={proxy_url or '直连'}")
    return OKXClient(
        api_key=api_key,
        secret_key=secret_key,
        passphrase=passphrase,
        testnet=testnet,
        proxy_url=proxy_url,
    )


class PaperReportDialog(QDialog):
    """历史模拟交易报告复盘窗口"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("📊 模拟交易历史报告复盘")
        self.resize(1100, 700)
        self._sessions: list = []
        self._current_data: dict = {}
        self._build_ui()
        self._load_sessions()

    def _build_ui(self):
        layout = QHBoxLayout(self)

        # ── 左侧：会话列表 ───────────────────────────────────────────────────
        left = QWidget()
        left.setFixedWidth(280)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)

        left_layout.addWidget(QLabel("历史会话（最新在前）:"))
        self._session_list = QListWidget()
        self._session_list.itemClicked.connect(self._on_session_selected)
        left_layout.addWidget(self._session_list)

        refresh_btn = QPushButton("🔄 刷新")
        refresh_btn.clicked.connect(self._load_sessions)
        left_layout.addWidget(refresh_btn)
        layout.addWidget(left)

        # ── 右侧：详情 ──────────────────────────────────────────────────────
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)

        # 汇总信息
        self._summary_text = QTextBrowser()
        self._summary_text.setMaximumHeight(160)
        self._summary_text.setStyleSheet("font-size:12px;")
        right_layout.addWidget(QLabel("会话汇总:"))
        right_layout.addWidget(self._summary_text)

        # 成交明细表格
        right_layout.addWidget(QLabel("成交明细:"))
        self._trade_table = QTableWidget()
        self._trade_table.setColumnCount(11)
        self._trade_table.setHorizontalHeaderLabels([
            "#", "交易对", "方向", "开仓时间", "平仓时间",
            "开仓价", "平仓价", "投入(U)", "盈亏(U)", "盈亏%", "平仓原因",
        ])
        self._trade_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self._trade_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._trade_table.setAlternatingRowColors(True)
        right_layout.addWidget(self._trade_table, 1)

        # 底部按钮
        btn_layout = QHBoxLayout()
        export_btn = QPushButton("💾 导出文本报告")
        export_btn.clicked.connect(self._export_report)
        btn_layout.addWidget(export_btn)
        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(self.accept)
        btn_layout.addWidget(close_btn)
        right_layout.addLayout(btn_layout)
        layout.addWidget(right, 1)

    def _load_sessions(self):
        from src.trading.paper_engine import PaperTradeEngine
        self._sessions = PaperTradeEngine.list_sessions()
        self._session_list.clear()
        for s in self._sessions:
            pnl = float(s.get('total_pnl', 0))
            ret = float(s.get('total_return', 0))
            n = int(s.get('total_trades', 0))
            color = '🟢' if ret >= 0 else '🔴'
            ts = s.get('saved_at', '')[:16].replace('T', ' ')
            label = (
                f"{color} {ts}\n"
                f"  策略: {s.get('strategy_name', '-')}\n"
                f"  收益: {ret:+.2f}%  盈亏: {pnl:+.2f}U  ({n}笔)"
            )
            item = QListWidgetItem(label)
            item.setData(Qt.UserRole, s)
            self._session_list.addItem(item)

    def _on_session_selected(self, item):
        from src.trading.paper_engine import PaperTradeEngine
        session_meta = item.data(Qt.UserRole)
        if not session_meta:
            return
        try:
            data = PaperTradeEngine.load_session(session_meta['file'])
        except Exception as e:
            QMessageBox.warning(self, "加载失败", str(e))
            return
        self._current_data = data
        self._refresh_detail(data)

    def _refresh_detail(self, data: dict):
        summary = data.get('summary', {})
        trades = data.get('trades', [])

        # 汇总
        lines = [
            f"<b>会话ID:</b> {data.get('session_id', '-')}",
            f"<b>策略:</b> {data.get('strategy_name', '-')}",
            f"<b>开始时间:</b> {(data.get('start_time') or '-')[:19].replace('T', ' ')}",
            f"<b>保存时间:</b> {(data.get('saved_at') or '-')[:19].replace('T', ' ')}",
            f"<b>初始资金:</b> {data.get('initial_capital', 0):,.2f} USDT",
            f"<b>模拟余额:</b> {summary.get('balance', 0):,.2f} USDT",
            f"<b>总收益率:</b> {summary.get('total_return', 0):+.2f}%",
            f"<b>总盈亏:</b> {summary.get('total_pnl', 0):+.2f} USDT",
            f"<b>总交易:</b> {summary.get('total_trades', 0)} 笔  "
            f"(<span style='color:#00ff88'>盈利 {summary.get('win_trades', 0)}</span> / "
            f"<span style='color:#ff4444'>亏损 {summary.get('lose_trades', 0)}</span>)",
            f"<b>胜率:</b> {summary.get('win_rate', 0):.1f}%",
            f"<b>最大单笔盈利:</b> {summary.get('max_profit', 0):+.2f} USDT",
            f"<b>最大单笔亏损:</b> {summary.get('max_loss', 0):+.2f} USDT",
            f"<b>总手续费:</b> {summary.get('total_fees', 0):.4f} USDT",
        ]
        self._summary_text.setHtml("<br>".join(lines))

        # 成交明细
        self._trade_table.setRowCount(0)
        for idx, t in enumerate(trades, 1):
            row = self._trade_table.rowCount()
            self._trade_table.insertRow(row)
            pnl = float(t.get('pnl', 0))
            cells = [
                str(idx),
                t.get('inst_id', ''),
                t.get('direction', ''),
                (t.get('entry_time') or '')[:19].replace('T', ' '),
                (t.get('exit_time') or '')[:19].replace('T', ' '),
                f"{t.get('entry_price', 0):.6f}",
                f"{t.get('exit_price', 0):.6f}",
                f"{t.get('usdt_amount', 0):.2f}",
                f"{pnl:+.2f}",
                f"{t.get('pnl_pct', 0):+.2f}%",
                t.get('exit_reason', ''),
            ]
            for col, txt in enumerate(cells):
                item = QTableWidgetItem(txt)
                if col == 8:
                    item.setForeground(QColor('#00ff88') if pnl >= 0 else QColor('#ff4444'))
                self._trade_table.setItem(row, col, item)

    def _export_report(self):
        if not self._current_data:
            QMessageBox.information(self, "提示", "请先选择一个会话")
            return
        data = self._current_data
        summary = data.get('summary', {})
        trades = data.get('trades', [])
        lines = [
            "=" * 60,
            "       模拟交易复盘报告",
            "=" * 60,
            f"会话ID: {data.get('session_id', '-')}",
            f"策略:   {data.get('strategy_name', '-')}",
            f"开始:   {(data.get('start_time') or '-')[:19].replace('T', ' ')}",
            f"保存:   {(data.get('saved_at') or '-')[:19].replace('T', ' ')}",
            "-" * 60,
            f"初始资金:   {data.get('initial_capital', 0):>12,.2f} USDT",
            f"模拟余额:   {summary.get('balance', 0):>12,.2f} USDT",
            f"总收益率:   {summary.get('total_return', 0):>12.2f}%",
            f"总盈亏:     {summary.get('total_pnl', 0):>12.2f} USDT",
            f"总交易:     {summary.get('total_trades', 0):>12} 笔",
            f"胜率:       {summary.get('win_rate', 0):>12.1f}%",
            f"最大盈利:   {summary.get('max_profit', 0):>12.2f} USDT",
            f"最大亏损:   {summary.get('max_loss', 0):>12.2f} USDT",
            f"总手续费:   {summary.get('total_fees', 0):>12.4f} USDT",
            "-" * 60,
            "成交明细:",
        ]
        for idx, t in enumerate(trades, 1):
            pnl = float(t.get('pnl', 0))
            lines.append(
                f"{idx:3d}. {t.get('direction', ''):<6} {t.get('inst_id', ''):<15} "
                f"开 {t.get('entry_price', 0):.6f} → 平 {t.get('exit_price', 0):.6f}  "
                f"盈亏 {pnl:+.2f}U ({t.get('pnl_pct', 0):+.2f}%)  "
                f"原因: {t.get('exit_reason', '-')}"
            )
        lines.append("=" * 60)
        report_text = "\n".join(lines)

        path, _ = QFileDialog.getSaveFileName(
            self, "保存报告", f"paper_report_{data.get('session_id', 'export')}.txt",
            "Text Files (*.txt)"
        )
        if path:
            try:
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(report_text)
                QMessageBox.information(self, "已保存", f"报告已导出至：\n{path}")
            except Exception as e:
                QMessageBox.warning(self, "保存失败", str(e))


def main():
    """主函数"""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    okx_client = _load_okx_client()

    window = QuantTradeWindow(okx_client)
    window.show()

    sys.exit(app.exec() if hasattr(app, 'exec') and not hasattr(app, 'exec_') else app.exec_())


if __name__ == "__main__":
    main()
