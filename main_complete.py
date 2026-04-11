#!/usr/bin/env python3
"""Crypto Trader - 完整版"""

import sys
import os
import threading
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, 
    QWidget, QFrame, QTabWidget, QListWidget, QListWidgetItem, 
    QTextEdit, QComboBox, QDateEdit, QDoubleSpinBox, QGroupBox, 
    QFormLayout, QTableWidget, QTableWidgetItem, QSplitter, 
    QPushButton, QHeaderView, QFileDialog, QMessageBox, QProgressBar
)
from PyQt5.QtCore import Qt, QDate, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QColor

from src.strategy.loader import StrategyLoader
from src.api.okx_client import OKXClient
from src.trading.executor import TradeExecutor, PositionSide


class BacktestThread(QThread):
    """回测线程"""
    finished = pyqtSignal(object)
    log = pyqtSignal(str)
    progress = pyqtSignal(int)
    
    def __init__(self, strategy, config, okx_client):
        super().__init__()
        self.strategy = strategy
        self.config = config
        self.okx_client = okx_client
    
    def run(self):
        try:
            from src.backtest.engine import Backtester, BacktestAnalyzer
            
            self.log.emit(f"[信息] 开始回测：{self.config['strategy_name']}")
            self.progress.emit(20)
            
            backtester = Backtester(okx_client=self.okx_client, initial_capital=self.config['initial_capital'])
            result = backtester.run_backtest(
                strategy=self.strategy,
                inst_id=self.config['inst_id'],
                start_date=self.config['start_date'],
                end_date=self.config['end_date'],
                bar=self.config['bar']
            )
            
            self.progress.emit(80)
            result = BacktestAnalyzer.analyze(result)
            
            self.log.emit(f"[成功] 回测完成!")
            self.progress.emit(100)
            
            self.finished.emit(result)
        except Exception as e:
            self.log.emit(f"[错误] 回测失败：{e}")
            self.finished.emit(None)


class MainWindow(QMainWindow):
    def __init__(self, okx_client):
        super().__init__()
        self.setWindowTitle("Crypto Trader - 量化交易系统")
        self.setGeometry(100, 100, 1400, 900)
        
        self.okx_client = okx_client
        self.trade_executor = TradeExecutor(okx_client)
        self.strategy_loader = StrategyLoader('strategies')
        self.usdt_balance = 0.0
        self.positions = {}
        
        # 获取余额
        self.get_balance()
        
        # 创建 UI
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # 状态栏
        self.create_status_bar(main_layout)
        
        # 标签页
        self.tabs = QTabWidget()
        self.tabs.addTab(self.create_trade_tab(), "📈 实时交易")
        self.tabs.addTab(self.create_backtest_tab(), "📊 策略回测")
        self.tabs.addTab(self.create_strategy_tab(), "📁 策略管理")
        self.tabs.addTab(self.create_manual_tab(), "💰 手动交易")
        main_layout.addWidget(self.tabs)
        
        # 定时刷新
        self.refresh_timer = None
        
    def create_status_bar(self, layout):
        """创建状态栏"""
        bar = QFrame()
        bar.setStyleSheet("background-color: #2a2a2a; padding: 10px;")
        bar_layout = QHBoxLayout(bar)
        
        # 连接状态
        conn_label = QLabel("● 已连接 OKX")
        conn_label.setStyleSheet("color: #00ff00; font-weight: bold;")
        bar_layout.addWidget(conn_label)
        
        # 余额
        self.balance_label = QLabel(f"{self.usdt_balance:.2f} USDT")
        self.balance_label.setStyleSheet("color: #00ffaa; font-weight: bold; font-size: 14px;")
        bar_layout.addWidget(self.balance_label)
        
        bar_layout.addStretch()
        
        # 状态
        self.status_label = QLabel("就绪")
        self.status_label.setStyleSheet("color: #00ccff;")
        bar_layout.addWidget(self.status_label)
        
        layout.addWidget(bar)
    
    def get_balance(self):
        """获取 USDT 余额"""
        try:
            result = self.okx_client.get_balance()
            if result and result.get('code') == '0':
                for detail in result.get('data', []):
                    if 'details' in detail:
                        for asset in detail['details']:
                            if asset.get('ccy') == 'USDT':
                                self.usdt_balance = float(asset.get('availEq', 0))
                                return
        except Exception as e:
            print(f"获取余额失败：{e}")
    
    def refresh_balance(self):
        """刷新余额显示"""
        self.get_balance()
        self.balance_label.setText(f"{self.usdt_balance:.2f} USDT")
    
    def create_trade_tab(self):
        """创建实时交易标签页"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        splitter = QSplitter(Qt.Horizontal)
        
        # 左侧
        left = QWidget()
        left_layout = QVBoxLayout(left)
        
        # 策略列表
        strategy_group = QGroupBox("策略管理")
        strategy_layout = QVBoxLayout(strategy_group)
        
        self.trade_list = QListWidget()
        self.load_strategies(self.trade_list)
        self.trade_list.itemClicked.connect(self.on_trade_strategy_selected)
        strategy_layout.addWidget(self.trade_list)
        
        # 刷新和加载按钮
        btn_layout = QHBoxLayout()
        refresh_btn = QPushButton("🔄 刷新")
        refresh_btn.clicked.connect(lambda: self.load_strategies(self.trade_list))
        btn_layout.addWidget(refresh_btn)
        
        load_btn = QPushButton("📂 加载策略")
        load_btn.clicked.connect(self.load_custom_strategy)
        btn_layout.addWidget(load_btn)
        
        strategy_layout.addLayout(btn_layout)
        left_layout.addWidget(strategy_group)
        
        # 策略说明
        self.strategy_desc = QTextEdit()
        self.strategy_desc.setMaximumHeight(100)
        self.strategy_desc.setReadOnly(True)
        self.strategy_desc.setPlaceholderText("选择策略后显示说明...")
        left_layout.addWidget(self.strategy_desc)
        
        # 配置
        config_group = QGroupBox("交易配置")
        config_layout = QFormLayout(config_group)
        
        self.trade_pair = QComboBox()
        self.trade_pair.addItems(["BTC-USDT", "ETH-USDT", "SOL-USDT", "XRP-USDT"])
        config_layout.addRow("交易对:", self.trade_pair)
        
        self.trade_size = QDoubleSpinBox()
        self.trade_size.setRange(0.01, 1.0)
        self.trade_size.setValue(0.1)
        self.trade_size.setSuffix(" (10%)")
        config_layout.addRow("仓位比例:", self.trade_size)
        
        left_layout.addWidget(config_group)
        
        # 控制按钮
        self.trade_start_btn = QPushButton("🚀 启动策略")
        self.trade_start_btn.setStyleSheet("background-color: #00aa00; color: white; padding: 15px; font-size: 16px;")
        self.trade_start_btn.clicked.connect(self.start_trade_strategy)
        left_layout.addWidget(self.trade_start_btn)
        
        self.trade_stop_btn = QPushButton("⏹ 停止策略")
        self.trade_stop_btn.setStyleSheet("background-color: #aa0000; color: white; padding: 15px; font-size: 16px;")
        self.trade_stop_btn.clicked.connect(self.stop_trade_strategy)
        self.trade_stop_btn.setEnabled(False)
        left_layout.addWidget(self.trade_stop_btn)
        
        left_layout.addStretch()
        splitter.addWidget(left)
        
        # 右侧
        right = QWidget()
        right_layout = QVBoxLayout(right)
        
        # 持仓表格
        position_group = QGroupBox("持仓监控")
        position_layout = QVBoxLayout(position_group)
        
        self.position_table = QTableWidget()
        self.position_table.setColumnCount(7)
        self.position_table.setHorizontalHeaderLabels(["交易对", "方向", "数量", "开仓价", "当前价", "未实现盈亏", "操作"])
        self.position_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        position_layout.addWidget(self.position_table)
        
        # 刷新持仓按钮
        refresh_pos_btn = QPushButton("🔄 刷新持仓")
        refresh_pos_btn.clicked.connect(self.refresh_positions)
        position_layout.addWidget(refresh_pos_btn)
        
        right_layout.addWidget(position_group)
        
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        
        layout.addWidget(splitter)
        
        # 日志
        log_group = QGroupBox("交易日志")
        log_layout = QVBoxLayout(log_group)
        
        self.trade_log = QTextEdit()
        self.trade_log.setReadOnly(True)
        self.trade_log.setStyleSheet("background-color: #1e1e1e; color: #00ff00; font-family: 'Courier New';")
        self.trade_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] 实时交易页面已就绪")
        log_layout.addWidget(self.trade_log)
        
        layout.addWidget(log_group)
        
        return widget
    
    def create_backtest_tab(self):
        """创建回测标签页"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        splitter = QSplitter(Qt.Horizontal)
        
        # 左侧
        left = QWidget()
        left_layout = QVBoxLayout(left)
        
        # 策略列表
        strategy_group = QGroupBox("策略选择")
        strategy_layout = QVBoxLayout(strategy_group)
        
        self.backtest_list = QListWidget()
        self.load_strategies(self.backtest_list)
        self.backtest_list.itemClicked.connect(self.on_backtest_strategy_selected)
        strategy_layout.addWidget(self.backtest_list)
        
        # 刷新和加载按钮
        btn_layout = QHBoxLayout()
        refresh_btn = QPushButton("🔄 刷新")
        refresh_btn.clicked.connect(lambda: self.load_strategies(self.backtest_list))
        btn_layout.addWidget(refresh_btn)
        
        load_btn = QPushButton("📂 加载策略")
        load_btn.clicked.connect(self.load_custom_strategy_backtest)
        btn_layout.addWidget(load_btn)
        
        strategy_layout.addLayout(btn_layout)
        left_layout.addWidget(strategy_group)
        
        # 策略说明
        self.backtest_desc = QTextEdit()
        self.backtest_desc.setMaximumHeight(80)
        self.backtest_desc.setReadOnly(True)
        self.backtest_desc.setPlaceholderText("选择策略后显示说明...")
        left_layout.addWidget(self.backtest_desc)
        
        # 配置
        config_group = QGroupBox("回测配置")
        config_layout = QFormLayout(config_group)
        
        self.backtest_pair = QComboBox()
        self.backtest_pair.addItems(["BTC-USDT", "ETH-USDT", "SOL-USDT", "XRP-USDT"])
        config_layout.addRow("交易对:", self.backtest_pair)
        
        self.backtest_bar = QComboBox()
        self.backtest_bar.addItems(["3m", "5m", "15m", "30m", "1H", "2H", "4H", "1D"])
        self.backtest_bar.setCurrentText("1H")
        config_layout.addRow("K 线周期:", self.backtest_bar)
        
        self.backtest_start = QDateEdit()
        self.backtest_start.setCalendarPopup(True)
        self.backtest_start.setDate(QDate.currentDate().addMonths(-6))
        config_layout.addRow("开始日期:", self.backtest_start)
        
        self.backtest_end = QDateEdit()
        self.backtest_end.setCalendarPopup(True)
        self.backtest_end.setDate(QDate.currentDate())
        config_layout.addRow("结束日期:", self.backtest_end)
        
        self.backtest_capital = QDoubleSpinBox()
        self.backtest_capital.setRange(100, 1000000)
        self.backtest_capital.setValue(10000)
        self.backtest_capital.setSuffix(" USDT")
        config_layout.addRow("初始资金:", self.backtest_capital)
        
        left_layout.addWidget(config_group)
        
        # 开始按钮
        self.backtest_start_btn = QPushButton("🚀 开始回测")
        self.backtest_start_btn.setStyleSheet("background-color: #00aa00; color: white; padding: 15px; font-size: 16px;")
        self.backtest_start_btn.clicked.connect(self.start_backtest)
        left_layout.addWidget(self.backtest_start_btn)
        
        left_layout.addStretch()
        splitter.addWidget(left)
        
        # 右侧
        right = QWidget()
        right_layout = QVBoxLayout(right)
        
        # 进度条
        self.backtest_progress = QProgressBar()
        self.backtest_progress.setMaximum(100)
        right_layout.addWidget(self.backtest_progress)
        
        # 结果表格
        result_group = QGroupBox("回测结果")
        result_layout = QVBoxLayout(result_group)
        
        self.result_table = QTableWidget()
        self.result_table.setColumnCount(2)
        self.result_table.setHorizontalHeaderLabels(["指标", "数值"])
        self.result_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        result_layout.addWidget(self.result_table)
        
        # 导出按钮
        export_btn = QPushButton("📥 导出报告")
        export_btn.clicked.connect(self.export_backtest_report)
        result_layout.addWidget(export_btn)
        
        right_layout.addWidget(result_group)
        
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        
        layout.addWidget(splitter)
        
        # 日志
        log_group = QGroupBox("回测日志")
        log_layout = QVBoxLayout(log_group)
        
        self.backtest_log = QTextEdit()
        self.backtest_log.setReadOnly(True)
        self.backtest_log.setStyleSheet("background-color: #1e1e1e; color: #00ff00; font-family: 'Courier New';")
        self.backtest_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] 回测页面已就绪")
        log_layout.addWidget(self.backtest_log)
        
        layout.addWidget(log_group)
        
        return widget
    
    def create_strategy_tab(self):
        """创建策略管理标签页"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # 标题
        title = QLabel("📁 策略管理")
        title.setFont(QFont("Arial", 18, QFont.Bold))
        title.setStyleSheet("color: #00aaff;")
        layout.addWidget(title)
        
        # 策略列表
        self.strategy_manage_list = QListWidget()
        self.load_strategies(self.strategy_manage_list)
        layout.addWidget(self.strategy_manage_list)
        
        # 按钮
        btn_layout = QHBoxLayout()
        
        refresh_btn = QPushButton("🔄 刷新策略")
        refresh_btn.clicked.connect(lambda: self.load_strategies(self.strategy_manage_list))
        btn_layout.addWidget(refresh_btn)
        
        load_btn = QPushButton("📂 加载策略文件")
        load_btn.clicked.connect(self.load_custom_strategy_manage)
        btn_layout.addWidget(load_btn)
        
        view_btn = QPushButton("👁 查看策略详情")
        view_btn.clicked.connect(self.view_strategy_detail)
        btn_layout.addWidget(view_btn)
        
        layout.addLayout(btn_layout)
        
        # 策略详情
        detail_group = QGroupBox("策略详情")
        detail_layout = QVBoxLayout(detail_group)
        
        self.strategy_detail_text = QTextEdit()
        self.strategy_detail_text.setReadOnly(True)
        self.strategy_detail_text.setPlaceholderText("选择策略查看详情...")
        detail_layout.addWidget(self.strategy_detail_text)
        
        layout.addWidget(detail_group)
        
        return widget
    
    def create_manual_tab(self):
        """创建手动交易标签页"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # 标题
        title = QLabel("💰 手动交易")
        title.setFont(QFont("Arial", 18, QFont.Bold))
        title.setStyleSheet("color: #ffaa00;")
        layout.addWidget(title)
        
        # 配置
        config_group = QGroupBox("交易配置")
        config_layout = QFormLayout(config_group)
        
        self.manual_pair = QComboBox()
        self.manual_pair.addItems(["BTC-USDT", "ETH-USDT", "SOL-USDT", "XRP-USDT"])
        config_layout.addRow("交易对:", self.manual_pair)
        
        self.manual_size = QDoubleSpinBox()
        self.manual_size.setRange(0.01, 1.0)
        self.manual_size.setValue(0.1)
        self.manual_size.setSuffix(" (10%)")
        config_layout.addRow("仓位比例:", self.manual_size)
        
        layout.addWidget(config_group)
        
        # 交易按钮
        btn_layout = QHBoxLayout()
        
        buy_btn = QPushButton("🟢 买入/做多")
        buy_btn.setStyleSheet("background-color: #00aa00; color: white; padding: 20px; font-size: 18px;")
        buy_btn.clicked.connect(lambda: self.manual_trade("buy"))
        btn_layout.addWidget(buy_btn)
        
        sell_btn = QPushButton("🔴 卖出/平仓")
        sell_btn.setStyleSheet("background-color: #aa0000; color: white; padding: 20px; font-size: 18px;")
        sell_btn.clicked.connect(lambda: self.manual_trade("sell"))
        btn_layout.addWidget(sell_btn)
        
        layout.addLayout(btn_layout)
        
        # 持仓
        position_group = QGroupBox("当前持仓")
        position_layout = QVBoxLayout(position_group)
        
        self.manual_position_table = QTableWidget()
        self.manual_position_table.setColumnCount(6)
        self.manual_position_table.setHorizontalHeaderLabels(["交易对", "方向", "数量", "开仓价", "当前价", "盈亏"])
        self.manual_position_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        position_layout.addWidget(self.manual_position_table)
        
        refresh_btn = QPushButton("🔄 刷新持仓")
        refresh_btn.clicked.connect(self.refresh_manual_positions)
        position_layout.addWidget(refresh_btn)
        
        layout.addWidget(position_group)
        
        # 日志
        log_group = QGroupBox("交易日志")
        log_layout = QVBoxLayout(log_group)
        
        self.manual_log = QTextEdit()
        self.manual_log.setReadOnly(True)
        self.manual_log.setStyleSheet("background-color: #1e1e1e; color: #00ff00; font-family: 'Courier New';")
        self.manual_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] 手动交易页面已就绪")
        log_layout.addWidget(self.manual_log)
        
        layout.addWidget(log_group)
        
        return widget
    
    def load_strategies(self, list_widget):
        """加载策略列表"""
        list_widget.clear()
        strategies = self.strategy_loader.discover_strategies()
        for s in strategies:
            item = QListWidgetItem(f"{s.name} ({s.type.value})")
            item.setData(Qt.UserRole, s)
            list_widget.addItem(item)
        self.status_label.setText(f"已加载 {len(strategies)} 个策略")
    
    def on_trade_strategy_selected(self, item):
        """实时交易策略选择"""
        strategy_info = item.data(Qt.UserRole)
        if strategy_info:
            desc = f"策略：{strategy_info.name}\n"
            if strategy_info.description:
                desc += f"说明：{strategy_info.description}\n"
            if strategy_info.author:
                desc += f"作者：{strategy_info.author}"
            self.strategy_desc.setText(desc)
            self.trade_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] 选择策略：{strategy_info.name}")
    
    def on_backtest_strategy_selected(self, item):
        """回测策略选择"""
        strategy_info = item.data(Qt.UserRole)
        if strategy_info:
            desc = f"策略：{strategy_info.name}\n"
            if strategy_info.description:
                desc += f"说明：{strategy_info.description}\n"
            if strategy_info.author:
                desc += f"作者：{strategy_info.author}"
            self.backtest_desc.setText(desc)
            self.backtest_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] 选择策略：{strategy_info.name}")
    
    def load_custom_strategy(self):
        """加载自定义策略（实时交易）"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择策略文件", "",
            "Python Files (*.py);;All Files (*)"
        )
        if file_path:
            strategy_info = self.strategy_loader.load_custom_strategy(file_path)
            if strategy_info:
                self.load_strategies(self.trade_list)
                self.trade_log.append(f"[成功] 已加载策略：{strategy_info.name}")
            else:
                self.trade_log.append(f"[失败] 加载策略失败")
    
    def load_custom_strategy_backtest(self):
        """加载自定义策略（回测）"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择策略文件", "",
            "Python Files (*.py);;All Files (*)"
        )
        if file_path:
            strategy_info = self.strategy_loader.load_custom_strategy(file_path)
            if strategy_info:
                self.load_strategies(self.backtest_list)
                self.backtest_log.append(f"[成功] 已加载策略：{strategy_info.name}")
            else:
                self.backtest_log.append(f"[失败] 加载策略失败")
    
    def load_custom_strategy_manage(self):
        """加载自定义策略（管理）"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择策略文件", "",
            "Python Files (*.py);;All Files (*)"
        )
        if file_path:
            strategy_info = self.strategy_loader.load_custom_strategy(file_path)
            if strategy_info:
                self.load_strategies(self.strategy_manage_list)
                QMessageBox.information(self, "成功", f"已加载策略：{strategy_info.name}")
            else:
                QMessageBox.warning(self, "警告", "加载策略失败")
    
    def view_strategy_detail(self):
        """查看策略详情"""
        selected_items = self.strategy_manage_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "警告", "请先选择策略")
            return
        
        strategy_info = selected_items[0].data(Qt.UserRole)
        if strategy_info:
            detail = f"""策略名称：{strategy_info.name}
策略类型：{strategy_info.type.value}
策略描述：{strategy_info.description or '暂无'}
作者：{strategy_info.author or '未知'}
版本：{strategy_info.version or '1.0'}
文件路径：{strategy_info.path}

策略参数:
"""
            if strategy_info.config_schema:
                for param, info in strategy_info.config_schema.items():
                    detail += f"  - {param}: {info.get('label', param)} = {info.get('default', 'N/A')}\n"
            else:
                detail += "  暂无参数配置"
            
            self.strategy_detail_text.setText(detail)
    
    def start_trade_strategy(self):
        """启动交易策略"""
        selected_items = self.trade_list.selectedItems()
        if not selected_items:
            self.trade_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] [警告] 请先选择策略")
            return
        
        strategy_info = selected_items[0].data(Qt.UserRole)
        self.trade_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] [信息] 启动策略：{strategy_info.name}")
        self.trade_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] [信息] 交易对：{self.trade_pair.currentText()}")
        
        self.trade_start_btn.setEnabled(False)
        self.trade_stop_btn.setEnabled(True)
        self.status_label.setText("策略运行中")
    
    def stop_trade_strategy(self):
        """停止交易策略"""
        self.trade_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] [信息] 停止策略")
        self.trade_start_btn.setEnabled(True)
        self.trade_stop_btn.setEnabled(False)
        self.status_label.setText("已停止")
    
    def refresh_positions(self):
        """刷新持仓"""
        self.position_table.setRowCount(0)
        try:
            positions = self.trade_executor.get_positions()
            for inst_id, pos in positions.items():
                row = self.position_table.rowCount()
                self.position_table.insertRow(row)
                self.position_table.setItem(row, 0, QTableWidgetItem(inst_id))
                self.position_table.setItem(row, 1, QTableWidgetItem("做多" if pos.side == PositionSide.LONG else "做空"))
                self.position_table.setItem(row, 2, QTableWidgetItem(f"{pos.size:.6f}"))
                self.position_table.setItem(row, 3, QTableWidgetItem(f"{pos.entry_price:.4f}"))
                self.position_table.setItem(row, 4, QTableWidgetItem(f"{pos.current_price:.4f}"))
                
                pnl_item = QTableWidgetItem(f"{pos.unrealized_pnl:.2f} ({pos.pnl_percent:.2f}%)")
                pnl_item.setForeground(QColor("#00ff00" if pos.unrealized_pnl >= 0 else "#ff4444"))
                self.position_table.setItem(row, 5, pnl_item)
                
                close_btn = QPushButton("平仓")
                close_btn.setStyleSheet("background-color: #aa0000; color: white;")
                self.position_table.setCellWidget(row, 6, close_btn)
        except Exception as e:
            self.trade_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] [错误] 刷新持仓失败：{e}")
    
    def start_backtest(self):
        """开始回测"""
        selected_items = self.backtest_list.selectedItems()
        if not selected_items:
            self.backtest_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] [警告] 请先选择策略")
            return
        
        strategy_info = selected_items[0].data(Qt.UserRole)
        
        self.backtest_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] [信息] 开始回测：{strategy_info.name}")
        self.backtest_start_btn.setEnabled(False)
        self.backtest_progress.setValue(10)
        
        # 加载策略
        module = self.strategy_loader.load_strategy(strategy_info.name)
        if not module:
            self.backtest_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] [错误] 策略加载失败")
            self.backtest_start_btn.setEnabled(True)
            return
        
        strategy_class = self.strategy_loader.get_strategy_class(strategy_info.name)
        if not strategy_class:
            self.backtest_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] [错误] 未找到策略类")
            self.backtest_start_btn.setEnabled(True)
            return
        
        strategy = strategy_class({})
        
        config = {
            'strategy_name': strategy_info.name,
            'inst_id': self.backtest_pair.currentText(),
            'bar': self.backtest_bar.currentText(),
            'start_date': self.backtest_start.date().toString("yyyy-MM-dd"),
            'end_date': self.backtest_end.date().toString("yyyy-MM-dd"),
            'initial_capital': self.backtest_capital.value()
        }
        
        # 启动回测线程
        self.backtest_thread = BacktestThread(strategy, config, self.okx_client)
        self.backtest_thread.finished.connect(self.on_backtest_finished)
        self.backtest_thread.log.connect(self.backtest_log.append)
        self.backtest_thread.progress.connect(self.backtest_progress.setValue)
        self.backtest_thread.start()
    
    def on_backtest_finished(self, result):
        """回测完成"""
        self.backtest_start_btn.setEnabled(True)
        
        if result is None:
            self.backtest_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] [错误] 回测结果为空")
            return
        
        # 调试输出
        self.backtest_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] [调试] final_capital={result.final_capital}")
        self.backtest_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] [调试] total_trades={result.total_trades}")
        self.backtest_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] [调试] total_return={result.total_return}")
        
        # 显示结果
        self.result_table.setRowCount(0)
        metrics = [
            ('策略名称', result.strategy_name or 'N/A'),
            ('交易对', result.inst_id or 'N/A'),
            ('初始资金', f'{result.initial_capital:.2f} USDT'),
            ('最终资金', f'{result.final_capital:.2f} USDT'),
            ('总收益率', f'{result.total_return:.2f}%'),
            ('年化收益', f'{result.annual_return:.2f}%'),
            ('最大回撤', f'{result.max_drawdown:.2f}%'),
            ('夏普比率', f'{result.sharpe_ratio:.2f}'),
            ('胜率', f'{result.win_rate:.2f}%'),
            ('盈亏比', f'{result.profit_factor:.2f}'),
            ('交易次数', str(result.total_trades or 0)),
            ('盈利交易', str(result.winning_trades or 0)),
            ('亏损交易', str(result.losing_trades or 0)),
        ]
        
        for label, value in metrics:
            row = self.result_table.rowCount()
            self.result_table.insertRow(row)
            self.result_table.setItem(row, 0, QTableWidgetItem(label))
            self.result_table.setItem(row, 1, QTableWidgetItem(str(value)))
        
        self.backtest_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] [结果] 总收益率：{result.total_return:.2f}%")
        self.backtest_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] [结果] 最大回撤：{result.max_drawdown:.2f}%")
        self.backtest_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] [结果] 夏普比率：{result.sharpe_ratio:.2f}")
    
    def export_backtest_report(self):
        """导出回测报告"""
        if self.result_table.rowCount() == 0:
            QMessageBox.warning(self, "警告", "没有可导出的回测结果")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存回测报告", "",
            "Text Files (*.txt);;All Files (*)"
        )
        
        if file_path:
            try:
                report = "Crypto Trader 回测报告\n"
                report += "=" * 50 + "\n\n"
                for row in range(self.result_table.rowCount()):
                    label = self.result_table.item(row, 0).text()
                    value = self.result_table.item(row, 1).text()
                    report += f"{label}: {value}\n"
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(report)
                
                QMessageBox.information(self, "成功", f"报告已保存到:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"导出失败：{e}")
    
    def manual_trade(self, action):
        """手动交易"""
        inst_id = self.manual_pair.currentText()
        size = self.manual_size.value()
        
        try:
            if action == "buy":
                result = self.trade_executor.execute_buy(inst_id, position_ratio=size)
                if result.success:
                    self.manual_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] [成功] 买入 {inst_id} 成功")
                else:
                    self.manual_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] [错误] 买入失败：{result.message}")
            else:
                positions = self.trade_executor.get_positions(inst_id)
                if inst_id in positions:
                    result = self.trade_executor.execute_sell(inst_id, positions[inst_id].size)
                    if result.success:
                        self.manual_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] [成功] 卖出 {inst_id} 成功")
                    else:
                        self.manual_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] [错误] 卖出失败：{result.message}")
                else:
                    self.manual_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] [警告] 无持仓")
            
            self.refresh_balance()
            self.refresh_manual_positions()
        except Exception as e:
            self.manual_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] [错误] 交易失败：{e}")
    
    def refresh_manual_positions(self):
        """刷新手动交易持仓"""
        self.manual_position_table.setRowCount(0)
        try:
            positions = self.trade_executor.get_positions()
            for inst_id, pos in positions.items():
                row = self.manual_position_table.rowCount()
                self.manual_position_table.insertRow(row)
                self.manual_position_table.setItem(row, 0, QTableWidgetItem(inst_id))
                self.manual_position_table.setItem(row, 1, QTableWidgetItem("做多" if pos.side == PositionSide.LONG else "做空"))
                self.manual_position_table.setItem(row, 2, QTableWidgetItem(f"{pos.size:.6f}"))
                self.manual_position_table.setItem(row, 3, QTableWidgetItem(f"{pos.entry_price:.4f}"))
                self.manual_position_table.setItem(row, 4, QTableWidgetItem(f"{pos.current_price:.4f}"))
                
                pnl_item = QTableWidgetItem(f"{pos.unrealized_pnl:.2f} ({pos.pnl_percent:.2f}%)")
                pnl_item.setForeground(QColor("#00ff00" if pos.unrealized_pnl >= 0 else "#ff4444"))
                self.manual_position_table.setItem(row, 5, pnl_item)
        except Exception as e:
            self.manual_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] [错误] 刷新持仓失败：{e}")


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    okx_client = OKXClient(
        api_key="ddafb223-6fe7-4ada-94f6-a31d58b23e1a",
        secret_key="C05E005B0B94EB17E44739C7302605C9",
        passphrase="!Lqs4381525",
        testnet=True,
        proxy_url="http://127.0.0.1:7897"
    )
    
    win = MainWindow(okx_client)
    win.show()
    win.raise_()
    win.activateWindow()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
