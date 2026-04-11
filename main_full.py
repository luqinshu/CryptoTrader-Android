#!/usr/bin/env python3
"""Crypto Trader - 完整 PyQt5 版本"""

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QListWidget, QListWidgetItem, QTextEdit,
    QComboBox, QDateEdit, QDoubleSpinBox, QGroupBox, QFormLayout,
    QTableWidget, QTableWidgetItem, QSplitter, QProgressBar, QFrame,
    QHeaderView, QTabWidget
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject, QDate, QTimer
from PyQt5.QtGui import QFont

from src.strategy.loader import StrategyLoader
from src.backtest.engine import Backtester, BacktestAnalyzer
from src.api.okx_client import OKXClient
from src.trading.executor import TradeExecutor


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
    
    def log(self, message, level="INFO"):
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        color = {"INFO": "#00ff00", "WARNING": "#ffaa00", "ERROR": "#ff4444", "SUCCESS": "#00ffaa"}.get(level, "#ffffff")
        self.append(f'<span style="color:#666666">[{timestamp}]</span> <span style="color:{color}">[{level}]</span> {message}')
        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())


class BacktestThread(QThread):
    """回测线程"""
    finished = pyqtSignal(object)
    log = pyqtSignal(str)
    
    def __init__(self, strategy, config):
        super().__init__()
        self.strategy = strategy
        self.config = config
    
    def run(self):
        try:
            self.log.emit("开始回测...")
            backtester = Backtester(initial_capital=self.config.get('initial_capital', 10000))
            result = backtester.run_backtest(
                strategy=self.strategy,
                inst_id=self.config['inst_id'],
                start_date=self.config['start_date'],
                end_date=self.config['end_date'],
                bar=self.config['bar']
            )
            result = BacktestAnalyzer.analyze(result)
            self.log.emit("回测完成")
            self.finished.emit(result)
        except Exception as e:
            self.log.emit(f"错误：{e}")
            self.finished.emit(None)


class MainWindow(QMainWindow):
    def __init__(self, okx_client=None):
        super().__init__()
        self.setWindowTitle("Crypto Trader - 量化交易系统")
        self.setGeometry(100, 100, 1200, 800)
        self.setStyleSheet("""
            QMainWindow { background-color: #1e1e1e; }
            QLabel { color: #ffffff; }
            QGroupBox { 
                color: #ffffff; 
                font-weight: bold;
                border: 1px solid #444;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: #00aaff;
            }
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
            QPushButton {
                background-color: #0066cc;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 10px 20px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #0077ee;
            }
            QPushButton:disabled {
                background-color: #555;
            }
            QTableWidget {
                background-color: #2a2a2a;
                color: #ffffff;
                border: 1px solid #444;
            }
            QProgressBar {
                border: 1px solid #444;
                border-radius: 3px;
                text-align: center;
                background-color: #2a2a2a;
            }
            QProgressBar::chunk {
                background-color: #0066cc;
            }
            QTabWidget::pane {
                border: 1px solid #333;
                border-radius: 8px;
                background-color: #1e1e1e;
            }
            QTabBar::tab {
                background-color: #2a2a2a;
                color: #ffffff;
                padding: 12px 24px;
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
        
        self.okx_client = okx_client
        self.trade_executor = TradeExecutor(okx_client) if okx_client else None
        
        # 策略加载器
        self.strategy_loader = StrategyLoader('strategies')
        self.strategies = self.strategy_loader.discover_strategies()
        self.selected_strategy = None
        
        # 创建中央部件
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # 顶部状态栏
        self.create_status_bar(main_layout)
        
        # 标签页
        self.main_tabs = QTabWidget()
        
        # 实时交易页面
        trade_widget = self.create_trade_page()
        self.main_tabs.addTab(trade_widget, "实时交易")
        
        # 回测页面
        backtest_widget = self.create_backtest_page()
        self.main_tabs.addTab(backtest_widget, "策略回测")
        
        main_layout.addWidget(self.main_tabs)
        
        # 定时刷新持仓
        if self.trade_executor:
            self.refresh_timer = QTimer()
            self.refresh_timer.timeout.connect(self.refresh_positions)
            self.refresh_timer.start(5000)
    
    def create_status_bar(self, layout):
        """创建状态栏"""
        status_frame = QFrame()
        status_frame.setStyleSheet("""
            QFrame {
                background-color: #2a2a2a;
                border: 1px solid #333;
                border-radius: 8px;
                padding: 10px;
            }
        """)
        status_layout = QHBoxLayout(status_frame)
        
        # 测试 OKX 连接（使用 get_balance 更可靠）
        try:
            balance_result = self.okx_client.get_balance()
            if balance_result and balance_result.get('code') == '0':
                self.connection_label = QLabel("● 已连接 OKX")
                self.connection_label.setStyleSheet("color: #00ff00; font-weight: bold;")
                print('✅ OKX 连接成功')
                
                # 显示 USDT 余额
                usdt_balance = 0
                for detail in balance_result.get('data', []):
                    if 'details' in detail:
                        for asset in detail['details']:
                            if asset.get('ccy') == 'USDT':
                                usdt_balance = float(asset.get('availEq', 0))
                                break
                
                self.balance_label = QLabel(f"{usdt_balance:.2f} USDT")
                self.balance_label.setStyleSheet("color: #00ffaa; font-weight: bold; font-size: 14px;")
                print(f'USDT 余额：{usdt_balance:.2f}')
                status_layout.addWidget(self.balance_label)
            else:
                self.connection_label = QLabel("● 未连接")
                self.connection_label.setStyleSheet("color: #ff4444; font-weight: bold;")
                print(f'❌ OKX 连接失败：{balance_result}')
        except Exception as e:
            self.connection_label = QLabel("● 未连接")
            self.connection_label.setStyleSheet("color: #ff4444; font-weight: bold;")
            print(f'❌ OKX 连接异常：{e}')
        
        status_layout.addWidget(self.connection_label)
        
        balance_label = QLabel("账户余额:")
        status_layout.addWidget(balance_label)
        self.balance_label = QLabel("0.00 USDT")
        self.balance_label.setStyleSheet("color: #00ffaa; font-weight: bold; font-size: 14px;")
        status_layout.addWidget(self.balance_label)
        
        status_layout.addStretch()
        
        self.status_label = QLabel("就绪")
        self.status_label.setStyleSheet("color: #00ccff;")
        status_layout.addWidget(self.status_label)
        
        layout.addWidget(status_frame)
    
    def create_trade_page(self):
        """创建实时交易页面"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(15)
        
        splitter = QSplitter(Qt.Horizontal)
        
        # 左侧：策略管理
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        strategy_group = QGroupBox("策略管理")
        strategy_layout = QVBoxLayout(strategy_group)
        
        self.trade_strategy_list = QListWidget()
        self.trade_strategy_list.setMinimumHeight(150)
        self.trade_strategy_list.itemClicked.connect(self.on_trade_strategy_click)
        strategy_layout.addWidget(self.trade_strategy_list)
        
        refresh_btn = QPushButton("🔄 刷新策略")
        refresh_btn.clicked.connect(self.load_trade_strategies)
        strategy_layout.addWidget(refresh_btn)
        
        left_layout.addWidget(strategy_group)
        
        self.trade_strategy_desc = QTextEdit()
        self.trade_strategy_desc.setMaximumHeight(80)
        self.trade_strategy_desc.setReadOnly(True)
        self.trade_strategy_desc.setPlaceholderText("选择策略后显示说明...")
        left_layout.addWidget(self.trade_strategy_desc)
        
        config_group = QGroupBox("交易配置")
        config_layout = QFormLayout(config_group)
        
        self.trade_pair_combo = QComboBox()
        self.trade_pair_combo.setEditable(True)
        self.trade_pair_combo.addItems(["BTC-USDT", "ETH-USDT", "SOL-USDT", "XRP-USDT"])
        config_layout.addRow("交易对:", self.trade_pair_combo)
        
        self.position_size_spin = QDoubleSpinBox()
        self.position_size_spin.setRange(0.01, 1.0)
        self.position_size_spin.setValue(0.1)
        self.position_size_spin.setSuffix(" (10%)")
        config_layout.addRow("仓位比例:", self.position_size_spin)
        
        left_layout.addWidget(config_group)
        
        self.trade_start_btn = QPushButton("🚀 启动策略")
        self.trade_start_btn.setMinimumHeight(50)
        self.trade_start_btn.setStyleSheet("font-size: 16px; background-color: #00aa00;")
        self.trade_start_btn.clicked.connect(self.start_trade_strategy)
        left_layout.addWidget(self.trade_start_btn)
        
        left_layout.addStretch()
        splitter.addWidget(left_widget)
        
        # 右侧：持仓监控
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        position_group = QGroupBox("持仓监控")
        position_layout = QVBoxLayout(position_group)
        
        self.position_table = QTableWidget()
        self.position_table.setColumnCount(6)
        self.position_table.setHorizontalHeaderLabels(["交易对", "方向", "数量", "开仓价", "当前价", "盈亏"])
        self.position_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        position_layout.addWidget(self.position_table)
        
        right_layout.addWidget(position_group)
        
        manual_group = QGroupBox("手动交易")
        manual_layout = QHBoxLayout(manual_group)
        
        self.buy_btn = QPushButton("买入/做多")
        self.buy_btn.setStyleSheet("background-color: #00aa00;")
        self.buy_btn.clicked.connect(lambda: self.manual_trade("buy"))
        manual_layout.addWidget(self.buy_btn)
        
        self.sell_btn = QPushButton("卖出/平仓")
        self.sell_btn.setStyleSheet("background-color: #cc0000;")
        self.sell_btn.clicked.connect(lambda: self.manual_trade("sell"))
        manual_layout.addWidget(self.sell_btn)
        
        right_layout.addWidget(manual_group)
        
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        
        layout.addWidget(splitter)
        
        # 日志
        log_group = QGroupBox("交易日志")
        log_layout = QVBoxLayout(log_group)
        self.trade_log = TradeLogWidget()
        log_layout.addWidget(self.trade_log)
        layout.addWidget(log_group, 1)
        
        # 加载策略
        self.load_trade_strategies()
        self.trade_log.log("实时交易页面已就绪")
        
        return widget
    
    def create_backtest_page(self):
        """创建回测页面"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(15)
        
        splitter = QSplitter(Qt.Horizontal)
        
        # 左侧：配置
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        strategy_group = QGroupBox("策略选择")
        strategy_layout = QVBoxLayout(strategy_group)
        
        self.backtest_strategy_list = QListWidget()
        self.backtest_strategy_list.setMinimumHeight(150)
        self.backtest_strategy_list.itemClicked.connect(self.on_backtest_strategy_click)
        strategy_layout.addWidget(self.backtest_strategy_list)
        
        refresh_btn = QPushButton("🔄 刷新策略")
        refresh_btn.clicked.connect(self.load_backtest_strategies)
        strategy_layout.addWidget(refresh_btn)
        
        left_layout.addWidget(strategy_group)
        
        self.backtest_strategy_desc = QTextEdit()
        self.backtest_strategy_desc.setMaximumHeight(80)
        self.backtest_strategy_desc.setReadOnly(True)
        self.backtest_strategy_desc.setPlaceholderText("选择策略后显示说明...")
        left_layout.addWidget(self.backtest_strategy_desc)
        
        config_group = QGroupBox("回测配置")
        config_layout = QFormLayout(config_group)
        
        self.backtest_pair_combo = QComboBox()
        self.backtest_pair_combo.setEditable(True)
        self.backtest_pair_combo.addItems(["BTC-USDT", "ETH-USDT", "SOL-USDT", "XRP-USDT"])
        config_layout.addRow("交易对:", self.backtest_pair_combo)
        
        self.backtest_bar_combo = QComboBox()
        self.backtest_bar_combo.addItems(["1H", "4H", "1D"])
        self.backtest_bar_combo.setCurrentText("1H")
        config_layout.addRow("K 线周期:", self.backtest_bar_combo)
        
        self.backtest_start_date = QDateEdit()
        self.backtest_start_date.setCalendarPopup(True)
        self.backtest_start_date.setDate(QDate.currentDate().addMonths(-6))
        config_layout.addRow("开始日期:", self.backtest_start_date)
        
        self.backtest_end_date = QDateEdit()
        self.backtest_end_date.setCalendarPopup(True)
        self.backtest_end_date.setDate(QDate.currentDate())
        config_layout.addRow("结束日期:", self.backtest_end_date)
        
        self.backtest_capital_spin = QDoubleSpinBox()
        self.backtest_capital_spin.setRange(100, 1000000)
        self.backtest_capital_spin.setValue(10000)
        self.backtest_capital_spin.setSuffix(" USDT")
        config_layout.addRow("初始资金:", self.backtest_capital_spin)
        
        left_layout.addWidget(config_group)
        
        self.backtest_start_btn = QPushButton("🚀 开始回测")
        self.backtest_start_btn.setMinimumHeight(50)
        self.backtest_start_btn.setStyleSheet("font-size: 16px; background-color: #00aa00;")
        self.backtest_start_btn.clicked.connect(self.start_backtest)
        left_layout.addWidget(self.backtest_start_btn)
        
        left_layout.addStretch()
        splitter.addWidget(left_widget)
        
        # 右侧：结果
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        self.backtest_result_table = QTableWidget()
        self.backtest_result_table.setColumnCount(2)
        self.backtest_result_table.setHorizontalHeaderLabels(["指标", "数值"])
        self.backtest_result_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        right_layout.addWidget(self.backtest_result_table)
        
        self.backtest_progress = QProgressBar()
        self.backtest_progress.setMaximum(100)
        right_layout.addWidget(self.backtest_progress)
        
        self.backtest_log = TradeLogWidget()
        self.backtest_log.setMaximumHeight(150)
        right_layout.addWidget(self.backtest_log)
        
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        
        layout.addWidget(splitter)
        
        # 加载策略
        self.load_backtest_strategies()
        self.backtest_log.log("回测页面已就绪")
        
        return widget
    
    def load_trade_strategies(self):
        """加载交易策略列表"""
        self.trade_strategy_list.clear()
        self.strategies = self.strategy_loader.discover_strategies()
        
        for s in self.strategies:
            item = QListWidgetItem(f"{s.name} ({s.type.value})")
            item.setData(Qt.UserRole, s)
            self.trade_strategy_list.addItem(item)
        
        self.trade_log.log(f"已加载 {len(self.strategies)} 个策略")
    
    def on_trade_strategy_click(self, item):
        """交易策略点击"""
        self.selected_strategy = item.data(Qt.UserRole)
        if self.selected_strategy:
            desc = f"策略：{self.selected_strategy.name}\n"
            if self.selected_strategy.description:
                desc += f"说明：{self.selected_strategy.description}"
            self.trade_strategy_desc.setText(desc)
    
    def start_trade_strategy(self):
        """启动交易策略"""
        if not self.selected_strategy:
            self.trade_log.log("请先选择策略", "WARNING")
            return
        self.trade_log.log(f"启动策略：{self.selected_strategy.name}", "SUCCESS")
    
    def load_backtest_strategies(self):
        """加载回测策略列表"""
        self.backtest_strategy_list.clear()
        self.strategies = self.strategy_loader.discover_strategies()
        
        for s in self.strategies:
            item = QListWidgetItem(f"{s.name} ({s.type.value})")
            item.setData(Qt.UserRole, s)
            self.backtest_strategy_list.addItem(item)
        
        self.backtest_log.log(f"已加载 {len(self.strategies)} 个策略")
    
    def on_backtest_strategy_click(self, item):
        """回测策略点击"""
        self.selected_strategy = item.data(Qt.UserRole)
        if self.selected_strategy:
            desc = f"策略：{self.selected_strategy.name}\n"
            if self.selected_strategy.description:
                desc += f"说明：{self.selected_strategy.description}"
            self.backtest_strategy_desc.setText(desc)
    
    def start_backtest(self):
        """开始回测"""
        if not self.selected_strategy:
            self.backtest_log.log("请先选择策略", "WARNING")
            return
        
        self.backtest_log.log(f"开始回测：{self.selected_strategy.name}", "INFO")
        self.backtest_progress.setValue(10)
        self.backtest_start_btn.setEnabled(False)
        
        module = self.strategy_loader.load_strategy(self.selected_strategy.name)
        if not module:
            self.backtest_log.log("策略加载失败", "ERROR")
            self.backtest_start_btn.setEnabled(True)
            return
        
        strategy_class = self.strategy_loader.get_strategy_class(self.selected_strategy.name)
        if not strategy_class:
            self.backtest_log.log("未找到策略类", "ERROR")
            self.backtest_start_btn.setEnabled(True)
            return
        
        strategy = strategy_class({})
        
        config = {
            'inst_id': self.backtest_pair_combo.currentText(),
            'bar': self.backtest_bar_combo.currentText(),
            'start_date': self.backtest_start_date.date().toString("yyyy-MM-dd"),
            'end_date': self.backtest_end_date.date().toString("yyyy-MM-dd"),
            'initial_capital': self.backtest_capital_spin.value()
        }
        
        self.backtest_thread = BacktestThread(strategy, config)
        self.backtest_thread.finished.connect(self.on_backtest_finished)
        self.backtest_thread.log.connect(self.backtest_log.log)
        self.backtest_thread.start()
        self.backtest_progress.setValue(50)
    
    def on_backtest_finished(self, result):
        """回测完成"""
        self.backtest_progress.setValue(100)
        self.backtest_start_btn.setEnabled(True)
        
        if result is None:
            self.backtest_log.log("回测失败", "ERROR")
            return
        
        self.backtest_result_table.setRowCount(0)
        metrics = [
            ("策略名称", result.strategy_name),
            ("交易对", result.inst_id),
            ("最终资金", f"{result.final_capital:.2f} USDT"),
            ("总收益率", f"{result.total_return:.2f}%"),
            ("年化收益", f"{result.annual_return:.2f}%"),
            ("最大回撤", f"{result.max_drawdown:.2f}%"),
            ("夏普比率", f"{result.sharpe_ratio:.2f}"),
            ("胜率", f"{result.win_rate:.2f}%"),
            ("交易次数", str(result.total_trades)),
        ]
        
        for label, value in metrics:
            row = self.backtest_result_table.rowCount()
            self.backtest_result_table.insertRow(row)
            self.backtest_result_table.setItem(row, 0, QTableWidgetItem(label))
            self.backtest_result_table.setItem(row, 1, QTableWidgetItem(value))
        
        self.backtest_log.log(f"回测完成！收益率：{result.total_return:.2f}%", "SUCCESS")
    
    def manual_trade(self, action):
        """手动交易"""
        inst_id = self.trade_pair_combo.currentText()
        if action == "buy":
            self.trade_log.log(f"买入 {inst_id}", "INFO")
        else:
            self.trade_log.log(f"卖出 {inst_id}", "INFO")
    
    def refresh_positions(self):
        """刷新持仓"""
        # 实现持仓刷新逻辑
        pass


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    okx_client = OKXClient(
        api_key="ddafb223-6fe7-4ada-94f6-a31d58b23e1a",
        secret_key="C05E005B0B94EB17E44739C7302605C9",
        passphrase="!Lqs4381525",
        testnet=True,
        proxy_url="http://127.0.0.1:7897"  # 使用代理
    )
    
    window = MainWindow(okx_client)
    window.show()
    window.raise_()
    window.activateWindow()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
