#!/usr/bin/env python3
"""Crypto Trader - 简化版"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, QWidget, QFrame, QTabWidget, QListWidget, QListWidgetItem, QTextEdit, QComboBox, QDateEdit, QDoubleSpinBox, QGroupBox, QFormLayout, QTableWidget, QTableWidgetItem, QSplitter, QPushButton, QHeaderView
from PyQt5.QtCore import Qt, QDate
from PyQt5.QtGui import QFont

from src.strategy.loader import StrategyLoader
from src.api.okx_client import OKXClient

class MainWindow(QMainWindow):
    def __init__(self, okx_client):
        super().__init__()
        self.setWindowTitle("Crypto Trader - 量化交易系统")
        self.setGeometry(100, 100, 1200, 800)
        
        self.okx_client = okx_client
        self.strategy_loader = StrategyLoader('strategies')
        self.usdt_balance = 0.0
        
        # 获取余额
        self.get_balance()
        
        # 创建 UI
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # 状态栏
        self.create_status_bar(layout)
        
        # 标签页
        self.tabs = QTabWidget()
        self.tabs.addTab(self.create_trade_tab(), "实时交易")
        self.tabs.addTab(self.create_backtest_tab(), "策略回测")
        layout.addWidget(self.tabs)
    
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
                                print(f"✅ USDT 余额：{self.usdt_balance}")
                                return
            print("❌ 未找到 USDT 余额")
        except Exception as e:
            print(f"❌ 获取余额失败：{e}")
    
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
        
        status_label = QLabel("就绪")
        status_label.setStyleSheet("color: #00ccff;")
        bar_layout.addWidget(status_label)
        
        layout.addWidget(bar)
    
    def create_trade_tab(self):
        """创建交易标签页"""
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
        strategy_layout.addWidget(self.trade_list)
        
        left_layout.addWidget(strategy_group)
        
        # 配置
        config_group = QGroupBox("交易配置")
        config_layout = QFormLayout(config_group)
        
        self.trade_pair = QComboBox()
        self.trade_pair.addItems(["BTC-USDT", "ETH-USDT", "SOL-USDT"])
        config_layout.addRow("交易对:", self.trade_pair)
        
        self.trade_size = QDoubleSpinBox()
        self.trade_size.setRange(0.01, 1.0)
        self.trade_size.setValue(0.1)
        config_layout.addRow("仓位比例:", self.trade_size)
        
        left_layout.addWidget(config_group)
        
        # 开始按钮
        start_btn = QPushButton("🚀 启动策略")
        start_btn.setStyleSheet("background-color: #00aa00; color: white; padding: 15px; font-size: 16px;")
        left_layout.addWidget(start_btn)
        
        left_layout.addStretch()
        splitter.addWidget(left)
        
        # 右侧
        right = QWidget()
        right_layout = QVBoxLayout(right)
        
        # 持仓表格
        position_group = QGroupBox("持仓监控")
        position_layout = QVBoxLayout(position_group)
        
        self.position_table = QTableWidget()
        self.position_table.setColumnCount(5)
        self.position_table.setHorizontalHeaderLabels(["交易对", "方向", "数量", "价格", "盈亏"])
        self.position_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        position_layout.addWidget(self.position_table)
        
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
        self.trade_log.setStyleSheet("background-color: #1e1e1e; color: #00ff00;")
        self.trade_log.append("[系统] 实时交易页面已就绪")
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
        strategy_layout.addWidget(self.backtest_list)
        
        left_layout.addWidget(strategy_group)
        
        # 配置
        config_group = QGroupBox("回测配置")
        config_layout = QFormLayout(config_group)
        
        self.backtest_pair = QComboBox()
        self.backtest_pair.addItems(["BTC-USDT", "ETH-USDT", "SOL-USDT"])
        config_layout.addRow("交易对:", self.backtest_pair)
        
        self.backtest_bar = QComboBox()
        self.backtest_bar.addItems(["1H", "4H", "1D"])
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
        
        # 加载策略按钮
        load_btn_layout = QHBoxLayout()
        load_btn = QPushButton("📂 加载策略文件")
        load_btn.clicked.connect(self.load_custom_strategy)
        load_btn.setStyleSheet("background-color: #0066cc; color: white; padding: 10px;")
        load_btn_layout.addWidget(load_btn)
        
        refresh_btn = QPushButton("🔄 刷新")
        refresh_btn.clicked.connect(lambda: self.load_strategies(self.backtest_list))
        refresh_btn.setStyleSheet("background-color: #666; color: white; padding: 10px;")
        load_btn_layout.addWidget(refresh_btn)
        
        left_layout.addLayout(load_btn_layout)
        
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
        
        # 结果表格
        result_group = QGroupBox("回测结果")
        result_layout = QVBoxLayout(result_group)
        
        self.result_table = QTableWidget()
        self.result_table.setColumnCount(2)
        self.result_table.setHorizontalHeaderLabels(["指标", "数值"])
        self.result_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        result_layout.addWidget(self.result_table)
        
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
        self.backtest_log.setStyleSheet("background-color: #1e1e1e; color: #00ff00;")
        self.backtest_log.append("[系统] 回测页面已就绪")
        log_layout.addWidget(self.backtest_log)
        
        layout.addWidget(log_group)
        
        return widget
    
    def start_backtest(self):
        """开始回测"""
        # 获取选中的策略
        selected_items = self.backtest_list.selectedItems()
        if not selected_items:
            self.backtest_log.append("[警告] 请先选择策略")
            return
        
        strategy_info = selected_items[0].data(Qt.UserRole)
        if not strategy_info:
            self.backtest_log.append("[错误] 策略信息无效")
            return
        
        self.backtest_log.append(f"[信息] 开始回测：{strategy_info.name}")
        self.backtest_log.append(f"[信息] 交易对：{self.backtest_pair.currentText()}")
        self.backtest_log.append(f"[信息] K 线周期：{self.backtest_bar.currentText()}")
        self.backtest_log.append(f"[信息] 初始资金：{self.backtest_capital.value()} USDT")
        
        # 加载策略
        module = self.strategy_loader.load_strategy(strategy_info.name)
        if not module:
            self.backtest_log.append("[错误] 策略加载失败")
            return
        
        strategy_class = self.strategy_loader.get_strategy_class(strategy_info.name)
        if not strategy_class:
            self.backtest_log.append("[错误] 未找到策略类")
            return
        
        # 实例化策略
        strategy = strategy_class({})
        self.backtest_log.append("[成功] 策略加载完成")
        
        # 运行回测
        try:
            from src.backtest.engine import Backtester, BacktestAnalyzer
            
            backtester = Backtester(okx_client=self.okx_client, initial_capital=self.backtest_capital.value())
            result = backtester.run_backtest(
                strategy=strategy,
                inst_id=self.backtest_pair.currentText(),
                start_date=self.backtest_start.date().toString('yyyy-MM-dd'),
                end_date=self.backtest_end.date().toString('yyyy-MM-dd'),
                bar=self.backtest_bar.currentText()
            )
            
            # 分析结果
            result = BacktestAnalyzer.analyze(result)
            
            # 显示结果
            self.backtest_log.append(f"[成功] 回测完成！")
            self.backtest_log.append(f"[结果] 总收益率：{result.total_return:.2f}%")
            self.backtest_log.append(f"[结果] 最大回撤：{result.max_drawdown:.2f}%")
            self.backtest_log.append(f"[结果] 夏普比率：{result.sharpe_ratio:.2f}")
            self.backtest_log.append(f"[结果] 胜率：{result.win_rate:.2f}%")
            self.backtest_log.append(f"[结果] 交易次数：{result.total_trades}")
            
            # 更新结果表格
            self.result_table.setRowCount(0)
            
            # 调试输出
            self.backtest_log.append(f"[调试] result.final_capital={result.final_capital}")
            self.backtest_log.append(f"[调试] result.total_return={result.total_return}")
            self.backtest_log.append(f"[调试] result.total_trades={result.total_trades}")
            
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
                ('交易次数', str(result.total_trades or 0)),
            ]
            for label, value in metrics:
                row = self.result_table.rowCount()
                self.result_table.insertRow(row)
                item1 = QTableWidgetItem(label)
                item2 = QTableWidgetItem(str(value))
                self.result_table.setItem(row, 0, item1)
                self.result_table.setItem(row, 1, item2)
            
        except Exception as e:
            self.backtest_log.append(f"[错误] 回测失败：{e}")
    
    def load_custom_strategy(self):
        """加载自定义策略文件"""
        from PyQt5.QtWidgets import QFileDialog
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
                self.backtest_log.append(f"[失败] 加载策略失败：{file_path}")
    
    def load_strategies(self, list_widget):
        """加载策略列表"""
        strategies = self.strategy_loader.discover_strategies()
        for s in strategies:
            item = QListWidgetItem(f"{s.name} ({s.type.value})")
            item.setData(Qt.UserRole, s)
            list_widget.addItem(item)

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
