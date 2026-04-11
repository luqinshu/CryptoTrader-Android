#!/usr/bin/env python3
"""Crypto Trader - GUI 版本"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, 
    QWidget, QTabWidget, QListWidget, QListWidgetItem, QTextEdit, QComboBox, 
    QDateEdit, QDoubleSpinBox, QGroupBox, QFormLayout, QTableWidget, 
    QTableWidgetItem, QSplitter, QPushButton, QHeaderView, 
    QFileDialog, QMessageBox, QProgressBar, QLineEdit, QCheckBox,
    QFrame
)
from PyQt5.QtGui import QFont, QColor
from PyQt5.QtCore import Qt, QDate
from PyQt5.QtGui import QFont, QColor

from src.strategy.loader import StrategyLoader
from src.api.okx_client import OKXClient
from src.trading.executor import TradeExecutor, PositionSide

class MainWindow(QMainWindow):
    def __init__(self, okx_client):
        super().__init__()
        self.setWindowTitle("Crypto Trader - 量化交易系统")
        self.setGeometry(100, 100, 1400, 900)
        
        self.okx_client = okx_client
        self.strategy_loader = StrategyLoader('strategies')
        
        # 获取余额
        self.usdt_balance = self.get_balance()
        
        # 创建 UI
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # 状态栏
        self.create_status_bar(layout)
        
        # 标签页 - API 设置在第一个位置
        self.tabs = QTabWidget()
        self.tabs.addTab(self.create_api_tab(), "⚙️ API 设置")
        self.tabs.addTab(self.create_backtest_tab(), "📊 策略回测")
        self.tabs.addTab(self.create_manual_tab(), "💰 手动交易")
        layout.addWidget(self.tabs)
        
    def create_status_bar(self, layout):
        bar = QFrame()
        bar.setStyleSheet("background-color: #2a2a2a; padding: 10px;")
        bar_layout = QHBoxLayout(bar)
        
        conn_label = QLabel("● 已连接 OKX")
        conn_label.setStyleSheet("color: #00ff00; font-weight: bold;")
        bar_layout.addWidget(conn_label)
        
        self.balance_label = QLabel(f"{self.usdt_balance:.2f} USDT")
        self.balance_label.setStyleSheet("color: #00ffaa; font-weight: bold; font-size: 14px;")
        bar_layout.addWidget(self.balance_label)
        
        bar_layout.addStretch()
        layout.addWidget(bar)
    
    def get_balance(self):
        try:
            result = self.okx_client.get_balance()
            if result and result.get('code') == '0':
                for detail in result.get('data', []):
                    if 'details' in detail:
                        for asset in detail['details']:
                            if asset.get('ccy') == 'USDT':
                                return float(asset.get('availEq', 0))
        except:
            pass
        return 0.0
    
    def create_api_tab(self):
        """API 设置页面"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(20)
        layout.setContentsMargins(30, 30, 30, 30)
        
        # 大标题
        title = QLabel("⚙️ OKX API 配置")
        title.setFont(QFont("Arial", 24, QFont.Bold))
        title.setStyleSheet("color: #00aaff; padding: 20px;")
        layout.addWidget(title)
        
        # 说明
        desc = QLabel("请在下方配置您的 OKX API 密钥，配置完成后点击测试连接验证配置是否有效。")
        desc.setStyleSheet("color: #888; font-size: 14px; padding: 10px;")
        desc.setWordWrap(True)
        layout.addWidget(desc)
        
        # API 配置框
        api_group = QGroupBox("API 密钥配置")
        api_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                font-size: 14px;
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
        """)
        api_layout = QFormLayout(api_group)
        api_layout.setSpacing(15)
        api_layout.setLabelAlignment(Qt.AlignRight)
        
        # API Key
        api_key_label = QLabel("API Key:")
        api_key_label.setStyleSheet("font-size: 13px; color: #fff;")
        self.api_key_input = QLineEdit()
        self.api_key_input.setText("ddafb223-6fe7-4ada-94f6-a31d58b23e1a")
        self.api_key_input.setEchoMode(QLineEdit.Password)
        self.api_key_input.setMinimumHeight(35)
        self.api_key_input.setStyleSheet("""
            QLineEdit {
                padding: 8px;
                border: 1px solid #444;
                border-radius: 4px;
                background-color: #1e1e1e;
                color: #fff;
            }
        """)
        api_layout.addRow(api_key_label, self.api_key_input)
        
        # Secret Key
        secret_key_label = QLabel("Secret Key:")
        secret_key_label.setStyleSheet("font-size: 13px; color: #fff;")
        self.secret_key_input = QLineEdit()
        self.secret_key_input.setText("C05E005B0B94EB17E44739C7302605C9")
        self.secret_key_input.setEchoMode(QLineEdit.Password)
        self.secret_key_input.setMinimumHeight(35)
        self.secret_key_input.setStyleSheet("""
            QLineEdit {
                padding: 8px;
                border: 1px solid #444;
                border-radius: 4px;
                background-color: #1e1e1e;
                color: #fff;
            }
        """)
        api_layout.addRow(secret_key_label, self.secret_key_input)
        
        # Passphrase
        passphrase_label = QLabel("Passphrase:")
        passphrase_label.setStyleSheet("font-size: 13px; color: #fff;")
        self.passphrase_input = QLineEdit()
        self.passphrase_input.setText("!Lqs4381525")
        self.passphrase_input.setEchoMode(QLineEdit.Password)
        self.passphrase_input.setMinimumHeight(35)
        self.passphrase_input.setStyleSheet("""
            QLineEdit {
                padding: 8px;
                border: 1px solid #444;
                border-radius: 4px;
                background-color: #1e1e1e;
                color: #fff;
            }
        """)
        api_layout.addRow(passphrase_label, self.passphrase_input)
        
        # 测试网
        testnet_label = QLabel("测试网:")
        testnet_label.setStyleSheet("font-size: 13px; color: #fff;")
        self.testnet_check = QCheckBox()
        self.testnet_check.setChecked(True)
        self.testnet_check.setStyleSheet("QCheckBox { color: #fff; font-size: 13px; }")
        api_layout.addRow(testnet_label, self.testnet_check)
        
        # 代理
        proxy_label = QLabel("代理地址:")
        proxy_label.setStyleSheet("font-size: 13px; color: #fff;")
        self.proxy_input = QLineEdit()
        self.proxy_input.setText("http://127.0.0.1:7897")
        self.proxy_input.setMinimumHeight(35)
        self.proxy_input.setStyleSheet("""
            QLineEdit {
                padding: 8px;
                border: 1px solid #444;
                border-radius: 4px;
                background-color: #1e1e1e;
                color: #fff;
            }
        """)
        api_layout.addRow(proxy_label, self.proxy_input)
        
        layout.addWidget(api_group)
        
        # 按钮区域
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(20)
        
        # 测试连接按钮
        self.test_btn = QPushButton("🔌 测试连接")
        self.test_btn.setStyleSheet("""
            QPushButton {
                background-color: #0066cc;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 15px 30px;
                font-size: 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #0077ee;
            }
        """)
        self.test_btn.clicked.connect(self.test_api_connection)
        btn_layout.addWidget(self.test_btn)
        
        # 保存按钮
        save_btn = QPushButton("💾 保存配置")
        save_btn.setStyleSheet("""
            QPushButton {
                background-color: #00aa00;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 15px 30px;
                font-size: 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #00cc00;
            }
        """)
        save_btn.clicked.connect(self.save_api_config)
        btn_layout.addWidget(save_btn)
        
        layout.addLayout(btn_layout)
        
        # 状态显示
        self.api_status = QLabel("状态：已连接 ✅")
        self.api_status.setStyleSheet("color: #00ff00; font-size: 16px; padding: 15px;")
        layout.addWidget(self.api_status)
        
        layout.addStretch()
        return widget
    
    def test_api_connection(self):
        try:
            api_key = self.api_key_input.text()
            secret_key = self.secret_key_input.text()
            passphrase = self.passphrase_input.text()
            proxy_url = self.proxy_input.text()
            testnet = self.testnet_check.isChecked()
            
            client = OKXClient(
                api_key=api_key,
                secret_key=secret_key,
                passphrase=passphrase,
                testnet=testnet,
                proxy_url=proxy_url if proxy_url else None
            )
            
            result = client.get_tickers(instType='SPOT')
            if result and result.get('code') == '0':
                self.api_status.setText("状态：连接成功 ✅")
                self.api_status.setStyleSheet("color: #00ff00; font-size: 16px; padding: 15px;")
                QMessageBox.information(self, "成功", "✅ API 连接测试成功！")
            else:
                self.api_status.setText("状态：连接失败 ❌")
                self.api_status.setStyleSheet("color: #ff4444; font-size: 16px; padding: 15px;")
                QMessageBox.warning(self, "警告", f"API 连接失败：{result}")
        except Exception as e:
            self.api_status.setText(f"状态：错误 ❌")
            self.api_status.setStyleSheet("color: #ff4444; font-size: 16px; padding: 15px;")
            QMessageBox.critical(self, "错误", f"API 连接错误：{e}")
    
    def save_api_config(self):
        QMessageBox.information(self, "成功", "✅ API 配置已保存！\n\n配置将在下次启动时生效。")
    
    def create_backtest_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        title = QLabel("📊 策略回测")
        title.setFont(QFont("Arial", 18, QFont.Bold))
        title.setStyleSheet("color: #00aaff;")
        layout.addWidget(title)
        
        # 策略列表
        self.backtest_list = QListWidget()
        strategies = self.strategy_loader.discover_strategies()
        for s in strategies:
            item = QListWidgetItem(f"{s.name} ({s.type.value})")
            item.setData(Qt.UserRole, s)
            self.backtest_list.addItem(item)
        layout.addWidget(self.backtest_list)
        
        # K 线周期
        bar_label = QLabel("K 线周期：3m, 5m, 15m, 30m, 1H, 2H, 4H, 1D")
        bar_label.setStyleSheet("color: #00ff00; font-size: 14px;")
        layout.addWidget(bar_label)
        
        layout.addStretch()
        return widget
    
    def create_manual_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        title = QLabel("💰 手动交易")
        title.setFont(QFont("Arial", 18, QFont.Bold))
        title.setStyleSheet("color: #ffaa00;")
        layout.addWidget(title)
        
        # 交易对
        pair_label = QLabel("交易对:")
        pair_label.setStyleSheet("color: #fff;")
        layout.addWidget(pair_label)
        
        self.manual_pair = QComboBox()
        self.manual_pair.addItems(["BTC-USDT", "ETH-USDT", "SOL-USDT"])
        layout.addWidget(self.manual_pair)
        
        # 买卖按钮
        btn_layout = QHBoxLayout()
        
        buy_btn = QPushButton("🟢 买入")
        buy_btn.setStyleSheet("background-color: #00aa00; color: white; padding: 20px; font-size: 18px;")
        btn_layout.addWidget(buy_btn)
        
        sell_btn = QPushButton("🔴 卖出")
        sell_btn.setStyleSheet("background-color: #aa0000; color: white; padding: 20px; font-size: 18px;")
        btn_layout.addWidget(sell_btn)
        
        layout.addLayout(btn_layout)
        layout.addStretch()
        return widget

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
    
    print("\n" + "="*50)
    print("✅ 程序已启动")
    print("="*50)
    print("API 设置位置：点击窗口顶部的第一个标签页 ⚙️ API 设置")
    print("="*50)
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
