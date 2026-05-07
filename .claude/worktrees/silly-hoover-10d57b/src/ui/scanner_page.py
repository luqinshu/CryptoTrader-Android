"""
扫描页面 UI 组件 (完整版 - 含交易对池与排序)
提供完整的交易对扫描界面
"""

import sys
import time
import json
import os
import threading
from datetime import datetime
from typing import Dict, List, Optional

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QTableWidget, QTableWidgetItem, QComboBox, QGroupBox,
    QFormLayout, QSpinBox, QDoubleSpinBox, QProgressBar,
    QHeaderView, QFrame, QMessageBox, QSplitter, QCheckBox,
    QTextEdit, QTabWidget, QFileDialog, QDialog, QLineEdit,
    QListWidget, QListWidgetItem, QScrollArea
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QFont, QColor

from src.scanner.engine import ScanEngine
from src.scanner.base_scanner import BaseScannerStrategy, ScannerSymbol
from src.strategy.loader import StrategyLoader, StrategyInfo, StrategyType


class AutoCloseDialog(QDialog):
    """自动关闭的对话框"""
    
    def __init__(self, title, message, parent=None, timeout=10):
        super().__init__(parent)
        self.timeout = timeout
        self.remaining = timeout
        
        self.setWindowTitle(title)
        self.setMinimumWidth(400)
        self.setModal(True)
        
        layout = QVBoxLayout(self)
        
        # 消息标签
        msg_label = QLabel(message)
        msg_label.setWordWrap(True)
        msg_label.setStyleSheet("font-size: 13px; padding: 10px;")
        layout.addWidget(msg_label)
        
        # 倒计时标签
        self.countdown_label = QLabel(f"⏱ {self.remaining} 秒后自动关闭")
        self.countdown_label.setStyleSheet(
            "color: #00ccff; font-size: 12px; font-weight: bold; padding: 5px;"
        )
        self.countdown_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.countdown_label)
        
        # 按钮布局
        btn_layout = QHBoxLayout()
        
        # 立即关闭按钮
        close_btn = QPushButton("立即关闭")
        close_btn.setStyleSheet(
            "QPushButton { background-color: #00ccff; color: white; border: none; "
            "border-radius: 4px; padding: 8px 16px; font-weight: bold; }"
            "QPushButton:hover { background-color: #00aadd; }"
        )
        close_btn.clicked.connect(self.accept)
        btn_layout.addWidget(close_btn)
        
        layout.addLayout(btn_layout)
        
        # 倒计时定时器
        self.timer = QTimer()
        self.timer.timeout.connect(self._update_countdown)
        self.timer.start(1000)
    
    def _update_countdown(self):
        self.remaining -= 1
        if self.remaining <= 0:
            self.timer.stop()
            self.accept()
        else:
            self.countdown_label.setText(f"⏱ {self.remaining} 秒后自动关闭")
            if self.remaining <= 3:
                self.countdown_label.setStyleSheet(
                    "color: #ff4444; font-size: 12px; font-weight: bold; padding: 5px;"
                )


class ScannerPage(QWidget):
    """
    扫描页面 - 含交易对池与结果排序
    """

    def __init__(self, okx_client):
        super().__init__()
        self.okx_client = okx_client
        self.scan_engine = ScanEngine(okx_client)
        
        # 正确构建策略目录路径
        import pathlib
        current_file = pathlib.Path(__file__)
        strategies_dir = str(current_file.parent.parent.parent / 'strategies')
        self.strategy_loader = StrategyLoader(strategies_dir=strategies_dir)
        
        self.current_strategy = None
        self.is_scanning = False
        self.scan_thread: Optional[threading.Thread] = None
        self.scan_results = None
        self.scan_error = None

        # 交易对池数据
        self.trading_pool = []
        self.load_pool()

        # 邮件配置
        self.email_config = {
            'smtp_server': 'smtp.gmail.com',
            'smtp_port': 587,
            'sender_email': '',
            'sender_password': '',
            'recipient_email': '',
        }
        self.load_email_config()

        # 状态轮询定时器
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.check_scan_status)
        self.status_timer.start(200)

        self.init_ui()
        self.refresh_strategies()

    def load_email_config(self):
        import json, os
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'email_config.json')
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    self.email_config.update(json.load(f))
            except Exception as e:
                print(f"加载邮件配置失败：{e}")

    def save_email_config(self):
        import json, os
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'email_config.json')
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(self.email_config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"保存邮件配置失败：{e}")

    def show_email_config(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("📧 邮箱配置")
        dialog.setModal(True)
        dialog.resize(400, 300)
        layout = QFormLayout(dialog)
        
        smtp_server = QLineEdit(self.email_config.get('smtp_server', 'smtp.gmail.com'))
        layout.addRow("SMTP 服务器:", smtp_server)
        smtp_port = QSpinBox()
        smtp_port.setRange(1, 65535)
        smtp_port.setValue(self.email_config.get('smtp_port', 587))
        layout.addRow("SMTP 端口:", smtp_port)
        sender_email = QLineEdit(self.email_config.get('sender_email', ''))
        layout.addRow("发件人邮箱:", sender_email)
        sender_password = QLineEdit(self.email_config.get('sender_password', ''))
        sender_password.setEchoMode(QLineEdit.Password)
        layout.addRow("授权码/密码:", sender_password)
        recipient_email = QLineEdit(self.email_config.get('recipient_email', ''))
        layout.addRow("收件人邮箱:", recipient_email)
        
        note = QLabel("💡 提示：Gmail/QQ/163 需开启 SMTP 并使用授权码")
        note.setStyleSheet("color: #888; font-size: 11px;")
        layout.addRow(note)
        
        btn_layout = QHBoxLayout()
        save_btn = QPushButton("保存")
        save_btn.clicked.connect(dialog.accept)
        cancel_btn = QPushButton("取消")
        cancel_btn.clicked.connect(dialog.reject)
        btn_layout.addWidget(save_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addRow(btn_layout)
        
        if dialog.exec_() == QDialog.Accepted:
            self.email_config.update({
                'smtp_server': smtp_server.text(),
                'smtp_port': smtp_port.value(),
                'sender_email': sender_email.text(),
                'sender_password': sender_password.text(),
                'recipient_email': recipient_email.text(),
            })
            self.save_email_config()
            self.status_label.setText("✅ 邮箱配置已保存")

    def send_scan_result_email(self, results: list):
        import smtplib
        from email.mime.text import MIMEText
        from email.mime.multipart import MIMEMultipart
        
        if not self.email_config.get('sender_email') or not self.email_config.get('recipient_email'):
            return False
        
        try:
            msg = MIMEMultipart()
            msg['From'] = self.email_config['sender_email']
            msg['To'] = self.email_config['recipient_email']
            msg['Subject'] = f"📊 加密合约扫描结果 - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            body = self.build_email_body(results)
            msg.attach(MIMEText(body, 'html', 'utf-8'))
            
            server = smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port'])
            server.starttls()
            server.login(self.email_config['sender_email'], self.email_config['sender_password'])
            server.send_message(msg)
            server.quit()
            self.status_label.setText("📧 扫描结果已发送到邮箱")
            return True
        except Exception as e:
            print(f"发送邮件失败：{e}")
            self.status_label.setText(f"❌ 邮件发送失败：{e}")
            return False

    def build_email_body(self, results: list) -> str:
        html = f"""
        <html><head><style>
            body{{font-family:Arial,sans-serif;background:#f5f5f5;padding:20px;}}
            .container{{max-width:800px;margin:0 auto;background:white;padding:20px;border-radius:8px;}}
            h2{{color:#0066cc;}} table{{width:100%;border-collapse:collapse;margin:20px 0;}}
            th{{background:#f0f0f0;padding:10px;text-align:left;border-bottom:2px solid #ddd;}}
            td{{padding:8px 10px;border-bottom:1px solid #eee;}}
            .gain{{color:#00aa00;font-weight:bold;}} .loss{{color:#ff0000;font-weight:bold;}}
        </style></head><body><div class="container">
        <h2>📊 加密合约扫描结果</h2><p>扫描时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        """
        if results and results[0].get('type') == 'gainer_loser_ranking':
            data = results[0]
            if data.get('top_gainers'):
                html += "<h3>📈 涨幅榜 TOP</h3><table><tr><th>排名</th><th>交易对</th><th>价格</th><th>涨幅</th><th>量</th></tr>"
                for i in data['top_gainers']:
                    html += f"<tr><td>#{i.get('rank')}</td><td>{i.get('symbol','').replace('-USDT-SWAP','/USDT')}</td><td>{i.get('last_price',0):.4f}</td><td class='gain'>+{i.get('price_change_24h',0):.2f}%</td><td>{i.get('volume_24h',0)/1e6:.2f}M</td></tr>"
                html += "</table>"
            if data.get('top_losers'):
                html += "<h3>📉 跌幅榜 TOP</h3><table><tr><th>排名</th><th>交易对</th><th>价格</th><th>跌幅</th><th>量</th></tr>"
                for i in data['top_losers']:
                    html += f"<tr><td>#{i.get('rank')}</td><td>{i.get('symbol','').replace('-USDT-SWAP','/USDT')}</td><td>{i.get('last_price',0):.4f}</td><td class='loss'>{i.get('price_change_24h',0):.2f}%</td><td>{i.get('volume_24h',0)/1e6:.2f}M</td></tr>"
                html += "</table>"
        else:
            passed = [r for r in results if r.get('passed')]
            html += f"<h3>✅ 符合条件的交易对 ({len(passed)}个)</h3><table><tr><th>交易对</th><th>价格</th><th>涨跌</th><th>得分</th></tr>"
            for r in passed[:20]:
                html += f"<tr><td>{r.get('symbol','').replace('-USDT-SWAP','/USDT')}</td><td>{r.get('last_price',0):.4f}</td><td class='{'gain' if r.get('price_change_24h',0)>=0 else 'loss'}'>{r.get('price_change_24h',0):+.2f}%</td><td>{r.get('score',0):.1f}%</td></tr>"
            html += "</table>"
        html += "</div></body></html>"
        return html

    def load_pool(self):
        """加载交易对池"""
        pool_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'trading_pool.json')
        if os.path.exists(pool_path):
            try:
                with open(pool_path, 'r', encoding='utf-8') as f:
                    self.trading_pool = json.load(f)
            except Exception as e:
                print(f"加载交易池失败：{e}")
                self.trading_pool = []

    def save_pool(self):
        """保存交易对池"""
        pool_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'trading_pool.json')
        try:
            with open(pool_path, 'w', encoding='utf-8') as f:
                json.dump(self.trading_pool, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"保存交易池失败：{e}")

    def init_ui(self):
        """初始化 UI"""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)

        control_panel = self.create_control_panel()
        layout.addWidget(control_panel)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        status_frame = self.create_status_frame()
        layout.addWidget(status_frame)

        self.scan_info_frame = self.create_scan_info_frame()
        layout.addWidget(self.scan_info_frame)

        self.result_tabs = QTabWidget()

        self.gainers_table = self.create_ranking_table("涨幅榜")
        self.result_tabs.addTab(self.gainers_table, "📈 涨幅榜 TOP")

        self.losers_table = self.create_ranking_table("跌幅榜")
        self.result_tabs.addTab(self.losers_table, "📉 跌幅榜 TOP")

        self.result_table = self.create_result_table()
        self.result_tabs.addTab(self.result_table, "扫描结果")

        # 📦 交易对池标签页
        self.pool_table = self.create_pool_table()
        self.result_tabs.addTab(self.pool_table, "📦 交易对池")

        self.result_tabs.setCurrentIndex(3)  # 默认显示交易对池
        layout.addWidget(self.result_tabs)

        self.setStyleSheet("""
            QWidget { background-color: #1e1e1e; color: #ffffff; }
            QGroupBox { color: #00ccff; font-weight: bold; border: 1px solid #333; border-radius: 8px; margin-top: 10px; padding-top: 10px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; color: #00ccff; }
            QPushButton { background-color: #00ccff; color: white; border: none; border-radius: 4px; padding: 8px 16px; font-weight: bold; font-size: 13px; }
            QPushButton:hover { background-color: #00aadd; }
            QPushButton:pressed { background-color: #0088aa; }
            QPushButton:disabled { background-color: #555; color: #888; }
            QPushButton.stop { background-color: #ff4444; }
            QPushButton.stop:hover { background-color: #ff6666; }
            QTableWidget { background-color: #2a2a2a; border: 1px solid #333; border-radius: 4px; gridline-color: #444; }
            QHeaderView::section { background-color: #333; color: #00ccff; padding: 8px; border: none; font-weight: bold; }
            QComboBox, QSpinBox, QDoubleSpinBox { background-color: #2a2a2a; border: 1px solid #444; border-radius: 4px; padding: 5px; color: white; }
            QScrollBar:vertical { background-color: #2a2a2a; width: 12px; border-radius: 6px; }
            QScrollBar::handle:vertical { background-color: #555; border-radius: 6px; min-height: 30px; }
            QScrollBar::handle:vertical:hover { background-color: #777; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
        """)

    def create_control_panel(self) -> QWidget:
        panel = QWidget()
        layout = QHBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)

        strategy_group = QGroupBox("扫描策略")
        strategy_layout = QVBoxLayout(strategy_group)
        strategy_layout.setSpacing(8)

        # 策略选择模式
        self.single_strategy_radio = QCheckBox("单策略模式")
        self.single_strategy_radio.setChecked(True)
        self.single_strategy_radio.stateChanged.connect(self.on_strategy_mode_changed)
        strategy_layout.addWidget(self.single_strategy_radio)

        self.multi_strategy_radio = QCheckBox("多策略轮流模式")
        self.multi_strategy_radio.stateChanged.connect(self.on_strategy_mode_changed)
        strategy_layout.addWidget(self.multi_strategy_radio)

        # 单策略下拉框
        self.strategy_combo = QComboBox()
        self.strategy_combo.setMinimumWidth(300)
        self.strategy_combo.currentIndexChanged.connect(self.on_strategy_changed)
        strategy_layout.addWidget(self.strategy_combo)

        # 多策略列表（可多选）
        self.strategy_list_widget = QListWidget()
        self.strategy_list_widget.setSelectionMode(QListWidget.MultiSelection)
        self.strategy_list_widget.setVisible(False)
        self.strategy_list_widget.setMinimumHeight(100)
        strategy_layout.addWidget(self.strategy_list_widget)

        btn_layout = QHBoxLayout()
        refresh_btn = QPushButton("刷新策略列表")
        refresh_btn.setFixedWidth(120)
        refresh_btn.clicked.connect(self.refresh_strategies)
        btn_layout.addWidget(refresh_btn)

        load_custom_btn = QPushButton("加载自定义策略")
        load_custom_btn.setFixedWidth(140)
        load_custom_btn.clicked.connect(self.load_custom_strategy_file)
        btn_layout.addWidget(load_custom_btn)

        strategy_layout.addLayout(btn_layout)

        layout.addWidget(strategy_group, 1)

        settings_group = QGroupBox("扫描设置")
        settings_layout = QFormLayout(settings_group)
        settings_layout.setSpacing(8)

        self.interval_spin = QSpinBox()
        self.interval_spin.setRange(1, 60)
        self.interval_spin.setValue(1)
        self.interval_spin.setSuffix(" 分钟")
        settings_layout.addRow("扫描间隔:", self.interval_spin)

        # 多策略间隔设置
        self.multi_interval_spin = QSpinBox()
        self.multi_interval_spin.setRange(1, 60)
        self.multi_interval_spin.setValue(5)
        self.multi_interval_spin.setSuffix(" 分钟")
        self.multi_interval_spin.setVisible(False)
        settings_layout.addRow("策略切换间隔:", self.multi_interval_spin)

        self.show_passed_only = QCheckBox("仅显示通过的交易对")
        self.show_passed_only.setChecked(False)
        self.show_passed_only.stateChanged.connect(self.on_show_passed_changed)
        settings_layout.addRow(self.show_passed_only)

        self.email_notify = QCheckBox("扫描结果邮件通知")
        self.email_notify.setChecked(False)
        settings_layout.addRow(self.email_notify)

        email_config_btn = QPushButton("📧 配置邮箱")
        email_config_btn.setFixedWidth(100)
        email_config_btn.clicked.connect(self.show_email_config)
        settings_layout.addRow(email_config_btn)

        layout.addWidget(settings_group, 1)

        button_layout = QVBoxLayout()
        button_layout.setSpacing(8)

        self.start_btn = QPushButton("开始扫描")
        self.start_btn.setMinimumHeight(40)
        self.start_btn.clicked.connect(self.start_scan)
        button_layout.addWidget(self.start_btn)

        self.stop_btn = QPushButton("停止扫描")
        self.stop_btn.setMinimumHeight(40)
        self.stop_btn.setObjectName("stop")
        self.stop_btn.clicked.connect(self.stop_scan)
        self.stop_btn.setEnabled(False)
        button_layout.addWidget(self.stop_btn)

        self.auto_btn = QPushButton("自动扫描")
        self.auto_btn.setMinimumHeight(40)
        self.auto_btn.setCheckable(True)
        self.auto_btn.clicked.connect(self.toggle_auto_scan)
        button_layout.addWidget(self.auto_btn)

        layout.addLayout(button_layout, 0)
        return panel

    def create_status_frame(self) -> QWidget:
        frame = QFrame()
        frame.setStyleSheet("QFrame { background-color: #2a2a2a; border: 1px solid #333; border-radius: 4px; padding: 10px; }")
        layout = QHBoxLayout(frame)
        self.status_label = QLabel("就绪 - 请选择扫描策略")
        self.status_label.setStyleSheet("color: #00ccff; font-size: 14px;")
        layout.addWidget(self.status_label)
        self.scan_time_label = QLabel("上次扫描：从未")
        self.scan_time_label.setStyleSheet("color: #888;")
        layout.addWidget(self.scan_time_label)
        self.count_label = QLabel("通过：0 / 0")
        self.count_label.setStyleSheet("color: #00ffaa; font-weight: bold;")
        layout.addWidget(self.count_label)
        return frame

    def create_scan_info_frame(self) -> QFrame:
        frame = QFrame()
        frame.setStyleSheet("QFrame { background-color: #1a1a1a; border: 1px solid #444; border-radius: 4px; padding: 2px; }")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(5, 2, 5, 2)
        layout.setSpacing(2)
        title_label = QLabel("🔍 实时扫描信息")
        title_label.setStyleSheet("color: #ffaa00; font-weight: bold; font-size: 11px;")
        layout.addWidget(title_label)
        self.scanning_symbol_label = QLabel("等待扫描...")
        self.scanning_symbol_label.setStyleSheet("color: #00ffaa; font-size: 11px; font-weight: bold;")
        self.scanning_symbol_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.scanning_symbol_label)
        
        # 倒计时显示
        self.countdown_label = QLabel("")
        self.countdown_label.setStyleSheet("color: #00ccff; font-size: 12px; font-weight: bold;")
        self.countdown_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.countdown_label)
        
        info_layout = QHBoxLayout()
        info_layout.setContentsMargins(0, 0, 0, 0)
        self.scan_progress_detail = QLabel("")
        self.scan_progress_detail.setStyleSheet("color: #cccccc; font-size: 10px;")
        self.scan_progress_detail.setAlignment(Qt.AlignCenter)
        info_layout.addWidget(self.scan_progress_detail)
        layout.addLayout(info_layout)
        return frame

    def create_result_table(self) -> QTableWidget:
        table = QTableWidget()
        table.setColumnCount(10)
        table.setHorizontalHeaderLabels(["交易对", "最新价", "24h 成交量", "24h 涨跌幅", "24h 最高", "24h 最低", "方向", "得分", "评级", "触发信号"])
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table.setSelectionBehavior(QTableWidget.SelectRows)
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        table.setAlternatingRowColors(False)
        table.setSortingEnabled(True)
        table.setStyleSheet("""
            QTableWidget {
                background-color: #1a1a1a;
                color: #ffffff;
                gridline-color: #333333;
                border: 1px solid #333333;
            }
            QTableWidget::item {
                background-color: #1a1a1a;
                color: #ffffff;
                padding: 5px;
            }
            QTableWidget::item:selected {
                background-color: #00ccff;
                color: #000000;
            }
            QHeaderView::section {
                background-color: #2a2a2a;
                color: #ffffff;
                padding: 5px;
                border: 1px solid #333333;
                font-weight: bold;
            }
            QTableWidget QTableCornerButton::section {
                background-color: #2a2a2a;
                border: 1px solid #333333;
            }
        """)
        return table

    def create_ranking_table(self, table_type: str):
        table = QTableWidget()
        table.setColumnCount(7)
        table.setHorizontalHeaderLabels(["排名", "交易对", "最新价", "24h 涨跌幅", "24h 成交量", "24h 最高", "24h 最低"])
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table.setSelectionBehavior(QTableWidget.SelectRows)
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        table.setAlternatingRowColors(False)
        table.setStyleSheet("""
            QTableWidget {
                background-color: #1a1a1a;
                color: #ffffff;
                gridline-color: #333333;
                border: 1px solid #333333;
            }
            QTableWidget::item {
                background-color: #1a1a1a;
                color: #ffffff;
                padding: 5px;
            }
            QTableWidget::item:selected {
                background-color: #00ccff;
                color: #000000;
            }
            QHeaderView::section {
                background-color: #2a2a2a;
                color: #ffffff;
                padding: 5px;
                border: 1px solid #333333;
                font-weight: bold;
            }
            QTableWidget QTableCornerButton::section {
                background-color: #2a2a2a;
                border: 1px solid #333333;
            }
        """)
        return table

    def create_pool_table(self) -> QTableWidget:
        """创建交易对池表格"""
        table = QTableWidget()
        table.setColumnCount(7)
        table.setHorizontalHeaderLabels(["添加时间", "交易对", "最新价", "24h 涨跌幅", "得分", "来源策略", "操作"])
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table.setSelectionBehavior(QTableWidget.SelectRows)
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        table.setAlternatingRowColors(False)
        table.setStyleSheet("""
            QTableWidget {
                background-color: #1a1a1a;
                color: #ffffff;
                gridline-color: #333333;
                border: 1px solid #333333;
            }
            QTableWidget::item {
                background-color: #1a1a1a;
                color: #ffffff;
                padding: 5px;
            }
            QTableWidget::item:selected {
                background-color: #00ccff;
                color: #000000;
            }
            QHeaderView::section {
                background-color: #2a2a2a;
                color: #ffffff;
                padding: 5px;
                border: 1px solid #333333;
                font-weight: bold;
            }
            QTableWidget QTableCornerButton::section {
                background-color: #2a2a2a;
                border: 1px solid #333333;
            }
        """)
        return table

    def on_strategy_mode_changed(self, state):
        """策略模式切换"""
        if self.single_strategy_radio.isChecked():
            self.strategy_combo.setVisible(True)
            self.strategy_list_widget.setVisible(False)
            self.multi_interval_spin.setVisible(False)
        else:
            self.strategy_combo.setVisible(False)
            self.strategy_list_widget.setVisible(True)
            self.multi_interval_spin.setVisible(True)

    def on_show_passed_changed(self, state):
        """显示过滤切换"""
        self.refresh_result_table()

    def refresh_strategies(self):
        """刷新策略列表"""
        self.strategy_combo.clear()
        self.strategy_list_widget.clear()
        
        strategies = self.strategy_loader.discover_strategies()
        if not strategies:
            self.strategy_combo.addItem("未找到策略")
            return
        
        for s in strategies:
            # 单策略下拉框
            self.strategy_combo.addItem(s.name, s)
            
            # 多策略列表
            item = QListWidgetItem(s.name)
            item.setData(Qt.UserRole, s)
            self.strategy_list_widget.addItem(item)
        
        self.on_strategy_changed(0)

    def load_custom_strategy_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择策略文件", "", "Python Files (*.py);;All Files (*)")
        if file_path:
            info = self.strategy_loader.load_custom_strategy(file_path)
            if info:
                self.strategy_combo.addItem(f"{info.name} (自定义)", info)
                self.strategy_combo.setCurrentIndex(self.strategy_combo.count() - 1)
                self.on_strategy_changed(self.strategy_combo.currentIndex())
            else:
                QMessageBox.warning(self, "警告", f"加载策略失败：{file_path}")

    def on_strategy_changed(self, index: int):
        if index < 0: return
        info = self.strategy_combo.itemData(index)
        if not info: return
        try:
            module = self.strategy_loader.load_strategy(info.name)
            if not module:
                self.status_label.setText(f"加载策略失败：{info.name}")
                self.current_strategy = None
                return
            cls = None
            for name, obj in module.__dict__.items():
                if isinstance(obj, type) and name != 'BaseScannerStrategy':
                    try:
                        if issubclass(obj, BaseScannerStrategy): cls = obj; break
                    except: pass
            if not cls:
                for name, obj in module.__dict__.items():
                    if isinstance(obj, type) and ('Strategy' in name or 'Scanner' in name):
                        if obj.__module__ == module.__name__: cls = obj; break
            if not cls:
                self.current_strategy = None
                self.status_label.setText(f"⚠️ 策略 {info.name} 不是扫描策略")
                return
            config = {}
            if info.config_schema:
                for k, v in info.config_schema.items(): config[k] = v.get('default', 0)
            self.current_strategy = cls(config)
            self.scan_engine.set_strategy(self.current_strategy)
            self.status_label.setText(f"✓ 已选择策略：{info.name}")
        except Exception as e:
            import traceback; traceback.print_exc()
            self.current_strategy = None
            self.status_label.setText(f"❌ 策略加载错误：{e}")

    def _scan_worker(self, symbols=None):
        try:
            import time
            from concurrent.futures import ThreadPoolExecutor, as_completed
            engine = self.scan_engine
            engine.is_running = True

            strategy_name = "未知"
            if self.current_strategy:
                try: strategy_name = self.current_strategy.__class__.__name__
                except: pass

            self._scan_progress = {'status': 'fetching', 'message': f'✓ 真实扫描 [{strategy_name}] - 正在获取行情数据...', 'current': 0, 'total': 100}
            all_tickers = engine.get_all_tickers()
            if not all_tickers:
                self.scan_results = []; self.scan_error = "获取行情数据失败"; return

            total = len(all_tickers)
            self._scan_progress = {'status': 'processing', 'message': f'✓ 真实扫描 [{strategy_name}] - 正在解析数据...', 'current': 0, 'total': total}

            all_symbols_data = []
            for i, ticker in enumerate(all_tickers):
                if not engine.is_running: break
                symbol = engine.parse_ticker(ticker)
                open_24h = float(ticker.get('open24h', 0))
                if open_24h > 0: symbol.price_change_24h = ((symbol.last_price - open_24h) / open_24h) * 100
                
                # 如果是单边趋势扫描策略(高级版或早期发现版),获取K线数据
                is_unilateral_advanced = 'UnilateralTrendScannerAdvanced' in strategy_name
                is_unilateral_early = 'UnilateralTrendScannerEarly' in strategy_name
                is_swing_pro = 'OKXHourSwingScanner' in strategy_name
                
                if is_unilateral_advanced or is_unilateral_early or is_swing_pro:
                    try:
                        symbol.extra_data['klines'] = {
                            '5m': engine.get_klines(symbol.inst_id, bar='5m', limit=50),
                            '15m': engine.get_klines(symbol.inst_id, bar='15m', limit=50),
                            '1H': engine.get_klines(symbol.inst_id, bar='1H', limit=100),
                            '1D': engine.get_klines(symbol.inst_id, bar='1D', limit=200),
                            '3m': engine.get_klines(symbol.inst_id, bar='3m', limit=50),
                        }
                    except Exception as e:
                        print(f"获取 {symbol.inst_id} K线数据失败: {e}")
                        symbol.extra_data['klines'] = {}
                
                all_symbols_data.append(symbol)
                self._scan_progress = {'current': i + 1, 'total': total, 'symbol': symbol.inst_id, 'last_price': symbol.last_price, 'volume_24h': symbol.volume_24h, 'price_change_24h': symbol.price_change_24h}

            results = []
            is_ranking = hasattr(engine.strategy, 'scan_all_symbols')
            if is_ranking:
                self._scan_progress = {'status': 'analyzing', 'message': f'正在分析 {len(all_symbols_data)} 个交易对...', 'current': len(all_symbols_data), 'total': total}
                results = [engine.strategy.scan_all_symbols(all_symbols_data)]
            else:
                is_multi = hasattr(engine.strategy, '_check_daily_breakout') or hasattr(engine.strategy, '_check_daily_trend')
                is_vol = 'Volume' in strategy_name
                is_simple = 'SimpleMA' in strategy_name or '简单均线' in strategy_name

                self._scan_progress = {'status': 'filtering', 'message': '预过滤筛选...', 'current': 0, 'total': total}
                candidates = []
                for i, symbol in enumerate(all_symbols_data):
                    if not engine.is_running: break
                    if symbol.volume_24h < 500000: continue
                    if is_multi or is_vol or is_simple:
                        dk = engine.get_klines(symbol.inst_id, bar='1D', limit=30)
                        if dk and len(dk) >= 25:
                            passed = False
                            if is_multi: passed, _ = engine.strategy._check_daily_trend(dk) if hasattr(engine.strategy, '_check_daily_trend') else engine.strategy._check_daily_breakout(dk)
                            elif is_vol: passed = True # 成交量策略不依赖日线过滤
                            elif is_simple: passed = True
                            if passed: candidates.append(symbol)
                    else: candidates.append(symbol)
                    self._scan_progress = {'current': i + 1, 'total': total, 'symbol': symbol.inst_id, 'status': f'预过滤: {len(candidates)}个候选'}

                self._scan_progress = {'status': 'fetching_klines', 'message': f'获取{len(candidates)}个候选K线数据...'}
                def fetch_klines(sym):
                    try:
                        data = {}
                        if is_multi:
                            if 'MultiIndicator' in strategy_name: data = {'1D': engine.get_klines(sym.inst_id, '1D', 50), '1H': engine.get_klines(sym.inst_id, '1H', 50), '15m': engine.get_klines(sym.inst_id, '15m', 50)}
                            else: data = {'1D': engine.get_klines(sym.inst_id, '1D', 50), '1H': engine.get_klines(sym.inst_id, '1H', 100), '3m': engine.get_klines(sym.inst_id, '3m', 100)}
                        elif is_vol: data = {'1H': engine.get_klines(sym.inst_id, '1H', 30)}
                        elif is_simple: data = {'4H': engine.get_klines(sym.inst_id, '4H', 50), '1D': engine.get_klines(sym.inst_id, '1D', 50)}
                        elif 'Triple' in strategy_name: data = {'1D': engine.get_klines(sym.inst_id, '1D', 60), '1H': engine.get_klines(sym.inst_id, '1H', 65), '3m': engine.get_klines(sym.inst_id, '3m', 30)}
                        else: data = {'1D': engine.get_klines(sym.inst_id, '1D', 50), '1H': engine.get_klines(sym.inst_id, '1H', 50)}
                        sym.extra_data['klines'] = data
                        return sym, True
                    except: return sym, False

                with ThreadPoolExecutor(max_workers=10) as ex:
                    futures = {ex.submit(fetch_klines, s): s for s in candidates}
                    done = 0
                    for f in as_completed(futures):
                        s, ok = f.result(); done += 1
                        self._scan_progress = {'current': done, 'total': len(candidates), 'symbol': s.inst_id, 'status': f'K线: {done}/{len(candidates)}'}

                for i, sym in enumerate(candidates):
                    if not engine.is_running: break
                    if not sym.extra_data.get('klines'): continue
                    res = engine.strategy.scan_symbol(sym)
                    res.update({'last_price': sym.last_price, 'volume_24h': sym.volume_24h, 'price_change_24h': sym.price_change_24h, 'high_24h': sym.high_24h, 'low_24h': sym.low_24h})
                    results.append(res)
                    self._scan_progress = {'current': i + 1, 'total': len(candidates), 'symbol': sym.inst_id, 'result': res}

            self.scan_results = results; self.scan_error = None
        except Exception as e:
            import traceback; traceback.print_exc()
            self.scan_results = None; self.scan_error = str(e)

    def start_scan(self):
        if not self.current_strategy:
            QMessageBox.warning(self, "警告", "请先选择一个有效的扫描策略")
            return
        if self.is_scanning: return

        strategy_name = self.current_strategy.__class__.__name__
        strategy_module = self.current_strategy.__class__.__module__
        print(f"\n{'='*50}\n=== 开始真实扫描 ===\n策略名称: {strategy_name}\n策略模块: {strategy_module}\n{'='*50}\n")

        self.is_scanning = True; self.scan_results = None; self.scan_error = None
        self._scan_progress = {'status': 'starting', 'message': '正在启动扫描...', 'current': 0, 'total': 100}

        self.start_btn.setEnabled(False); self.stop_btn.setEnabled(True); self.auto_btn.setEnabled(False)
        self.progress_bar.setVisible(True); self.progress_bar.setValue(0)

        self.status_label.setText(f"✓ 真实扫描 [{strategy_name}] - 正在启动...")
        self.status_label.setStyleSheet("color: #00ff00; font-size: 16px; font-weight: bold;")
        self.scanning_symbol_label.setText(f"策略模块: {strategy_module}")
        self.scan_progress_detail.setText(f"API: {self.scan_engine.okx_client.base_url}")
        
        self.result_table.setRowCount(0); self.gainers_table.setRowCount(0); self.losers_table.setRowCount(0)

        self.scan_thread = threading.Thread(target=self._scan_worker, daemon=True)
        self.scan_thread.start()
        print("扫描线程已启动")

    def stop_scan(self):
        self.scan_engine.is_running = False
        if self.scan_thread: self.scan_thread.join(timeout=5); self.scan_thread = None
        self.is_scanning = False; self.start_btn.setEnabled(True); self.stop_btn.setEnabled(False); self.auto_btn.setEnabled(True); self.progress_bar.setVisible(False)
        self.status_label.setText("扫描已停止"); self.status_label.setStyleSheet("color: #00ccff; font-size: 14px;")

    def toggle_auto_scan(self, checked: bool):
        if checked:
            # 检查策略是否已选择
            if self.single_strategy_radio.isChecked():
                if not self.current_strategy:
                    QMessageBox.warning(self, "警告", "请先选择一个有效的扫描策略")
                    self.auto_btn.setChecked(False)
                    return
            else:
                # 多策略模式
                selected_items = self.strategy_list_widget.selectedItems()
                if not selected_items:
                    QMessageBox.warning(self, "警告", "请先在多策略列表中选择一个或多个策略")
                    self.auto_btn.setChecked(False)
                    return

            self.auto_scan_interval = self.interval_spin.value() * 60
            self.countdown_seconds = self.auto_scan_interval
            self.auto_scan_timer = QTimer()
            self.auto_scan_timer.timeout.connect(self._auto_scan_tick)
            self.auto_scan_timer.start(1000)

            # 多策略模式初始化
            self.multi_strategy_mode = self.multi_strategy_radio.isChecked()
            if self.multi_strategy_mode:
                self.selected_strategies = []
                for item in self.strategy_list_widget.selectedItems():
                    info = item.data(Qt.UserRole)
                    if info:
                        self.selected_strategies.append(info)
                self.current_strategy_index = 0
                self.strategy_switch_interval = self.multi_interval_spin.value() * 60
                self.strategy_switch_countdown = self.strategy_switch_interval
                self._switch_to_next_strategy()

            # 启动倒计时
            self._update_countdown()

            self.auto_btn.setText("⏹ 停止自动")
            self.auto_btn.setStyleSheet("QPushButton { background-color: #ff4444; color: white; border: none; border-radius: 4px; padding: 8px 16px; font-weight: bold; font-size: 13px; } QPushButton:hover { background-color: #ff6666; }")
            
            if self.multi_strategy_mode:
                self.status_label.setText(f"⏱ 多策略轮流扫描已启动（{len(self.selected_strategies)}个策略，切换间隔 {self.multi_interval_spin.value()} 分钟）")
            else:
                self.status_label.setText(f"⏱ 自动扫描已启动（间隔 {self.interval_spin.value()} 分钟）")
        else:
            if hasattr(self, 'auto_scan_timer'):
                self.auto_scan_timer.stop()
                del self.auto_scan_timer
            self.countdown_seconds = 0
            self.countdown_label.setText("")
            self.multi_strategy_mode = False
            self.selected_strategies = []
            self.auto_btn.setText("▶ 自动扫描")
            self.auto_btn.setStyleSheet("")
            self.status_label.setText("⏹ 自动扫描已停止")

    def _switch_to_next_strategy(self):
        """切换到下一个策略"""
        if not hasattr(self, 'selected_strategies') or not self.selected_strategies:
            return

        info = self.selected_strategies[self.current_strategy_index]
        try:
            module = self.strategy_loader.load_strategy(info.name)
            if not module:
                self.status_label.setText(f"加载策略失败：{info.name}")
                return

            cls = None
            for name, obj in module.__dict__.items():
                if isinstance(obj, type) and name != 'BaseScannerStrategy':
                    try:
                        if issubclass(obj, BaseScannerStrategy):
                            cls = obj
                            break
                    except:
                        pass

            if not cls:
                for name, obj in module.__dict__.items():
                    if isinstance(obj, type) and ('Strategy' in name or 'Scanner' in name):
                        if obj.__module__ == module.__name__:
                            cls = obj
                            break

            if not cls:
                self.current_strategy = None
                self.status_label.setText(f"⚠️ 策略 {info.name} 不是扫描策略")
                return

            config = {}
            if info.config_schema:
                for k, v in info.config_schema.items():
                    config[k] = v.get('default', 0)

            self.current_strategy = cls(config)
            self.scan_engine.set_strategy(self.current_strategy)
            self.status_label.setText(f"✓ 已切换到策略：{info.name}")

            # 更新索引
            self.current_strategy_index = (self.current_strategy_index + 1) % len(self.selected_strategies)

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.current_strategy = None
            self.status_label.setText(f"❌ 策略切换错误：{e}")

    def _update_countdown(self):
        """更新倒计时显示"""
        if not hasattr(self, 'countdown_seconds') or self.countdown_seconds <= 0:
            self.countdown_label.setText("")
            return
        
        minutes = int(self.countdown_seconds // 60)
        seconds = int(self.countdown_seconds % 60)
        time_str = f"{minutes:02d}:{seconds:02d}"
        
        # 根据剩余时间改变颜色
        if self.countdown_seconds <= 10:
            color = "#ff4444"  # 红色警告
        elif self.countdown_seconds <= 30:
            color = "#ffaa00"  # 黄色警告
        else:
            color = "#00ccff"  # 蓝色正常
        
        self.countdown_label.setText(f"⏱ 下次扫描倒计时: {time_str}")
        self.countdown_label.setStyleSheet(f"color: {color}; font-size: 12px; font-weight: bold;")
    
    def _auto_scan_tick(self):
        """自动扫描定时器触发"""
        # 如果正在扫描，只更新倒计时
        if self.is_scanning:
            return

        # 多策略模式：检查是否需要切换策略
        if hasattr(self, 'multi_strategy_mode') and self.multi_strategy_mode:
            if hasattr(self, 'strategy_switch_countdown'):
                self.strategy_switch_countdown -= 1
                if self.strategy_switch_countdown <= 0:
                    # 切换策略
                    self._switch_to_next_strategy()
                    self.strategy_switch_countdown = self.strategy_switch_interval

        # 倒计时更新
        if hasattr(self, 'countdown_seconds'):
            self.countdown_seconds -= 1

            if self.countdown_seconds <= 0:
                # 倒计时结束，开始扫描
                self.countdown_seconds = self.auto_scan_interval  # 重置倒计时
                self.scan_results = None
                self.scan_error = None
                self.result_table.setRowCount(0)
                self.gainers_table.setRowCount(0)
                self.losers_table.setRowCount(0)
                
                strategy_name = "未知"
                if self.current_strategy:
                    try:
                        strategy_name = self.current_strategy.__class__.__name__
                    except:
                        pass
                
                self.status_label.setText(f"🔄 自动扫描中... [{strategy_name}]")
                self.start_scan()
            else:
                # 更新倒计时显示
                self._update_countdown()

    def refresh_result_table(self):
        """刷新结果表格"""
        if not hasattr(self, 'scan_results') or not self.scan_results:
            return

        results = self.scan_results
        if not results:
            return

        # 清空表格
        self.result_table.setRowCount(0)

        # 按得分降序排序
        sorted_results = sorted(results, key=lambda x: x.get('score', 0), reverse=True)

        # 显示结果
        for r in sorted_results:
            # 根据show_passed_only过滤
            if self.show_passed_only.isChecked() and not r.get('passed'):
                continue
            self.add_result_row(r)

    def check_scan_status(self):
        if not self.is_scanning: return
        strategy_name = "未知"
        if self.current_strategy:
            try: strategy_name = self.current_strategy.__class__.__name__
            except: strategy_name = "未知策略"

        status_text = f"✓ 真实扫描 [{strategy_name}]"
        if hasattr(self, '_scan_progress') and self._scan_progress:
            prog = self._scan_progress
            current, total = prog.get('current', 0), prog.get('total', 1)
            symbol, status, message = prog.get('symbol', ''), prog.get('status', ''), prog.get('message', '')
            self.progress_bar.setValue(current)
            self.scanning_symbol_label.setText(f"正在扫描: {symbol.replace('-USDT-SWAP', '/USDT').replace('-SWAP', '')}")
            price, change, volume = prog.get('last_price', 0), prog.get('price_change_24h', 0), prog.get('volume_24h', 0)
            vol_str = f"{volume / 1000000:.2f}M" if volume >= 1000000 else f"{volume / 1000:.2f}K"
            change_color = "#00ff00" if change >= 0 else "#ff4444"
            self.scan_progress_detail.setText(f'进度: {current}/{total} ({current*100//total}%) | 价格: <span style="color:#00ccff">{price:.4f}</span> | 24h量: <span style="color:#ffaa00">{vol_str}</span> | 涨跌: <span style="color:{change_color}">{change:+.2f}%</span>')
            status_text = f"✓ 真实扫描 [{strategy_name}] - {message or status} ({current}/{total})"
        else:
            status_text = f"✓ 真实扫描 [{strategy_name}] - 正在启动..."

        self.status_label.setText(status_text)
        self.status_label.setStyleSheet("color: #00ff00; font-size: 16px; font-weight: bold;")

        if self.scan_results is not None or self.scan_error is not None:
            self.on_scan_finished_internal()

    def on_scan_finished_internal(self):
        self.is_scanning = False
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.auto_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.scanning_symbol_label.setText("扫描完成")

        # 如果是自动扫描模式，重置倒计时
        if hasattr(self, 'auto_scan_timer') and hasattr(self, 'auto_scan_interval'):
            self.countdown_seconds = self.auto_scan_interval
            self._update_countdown()

        if self.scan_error:
            self.scan_progress_detail.setText(f"❌ 错误: {self.scan_error}")
            self.status_label.setText(f"❌ 扫描失败: {self.scan_error}")
            self.status_label.setStyleSheet("color: #ff4444; font-size: 16px; font-weight: bold;")
            QMessageBox.critical(self, "扫描错误", self.scan_error)
            return

        results = self.scan_results or []
        if self.email_notify.isChecked():
            threading.Thread(target=self.send_scan_result_email, args=(results,), daemon=True).start()

        if results and results[0].get('type') in ['gainer_loser_ranking', 'unilateral_trend', 'unilateral_trend_advanced', 'unilateral_trend_early']:
            data = results[0]
            
            # 处理单边趋势扫描结果(包括早期发现版)
            if data.get('type') in ['unilateral_trend', 'unilateral_trend_advanced', 'unilateral_trend_early']:
                long_opp = len(data.get('long_opportunities', []))
                short_opp = len(data.get('short_opportunities', []))
                total_opp = data.get('total_opportunities', 0)
                early_count = data.get('early_signals', 0)
                
                # 在扫描结果标签页显示
                self.result_tabs.setCurrentIndex(2)  # 切换到扫描结果标签
                self.update_unilateral_trend_table(data)
                
                is_auto = hasattr(self, 'auto_scan_timer')
                early_note = f" (其中{early_count}个早期信号)" if early_count > 0 else ""
                self.status_label.setText(f"✓ 单边趋势扫描完成 - 做多机会 {long_opp} 个, 做空机会 {short_opp} 个{early_note}")
                self.scan_time_label.setText(f"上次扫描：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                self.count_label.setText(f"机会：{total_opp}")
                self.scan_progress_detail.setText(f"做多:{long_opp}个, 做空:{short_opp}个")
                
                if total_opp == 0:
                    dlg = AutoCloseDialog(
                        "提示",
                        "扫描完成，但未发现明显的单边趋势机会。",
                        self,
                        timeout=10
                    )
                    dlg.exec_()
            else:
                # 处理涨跌幅排行榜结果
                gc, lc = len(data.get('top_gainers', [])), len(data.get('top_losers', []))
                self.update_ranking_tables(data)
                is_auto = hasattr(self, 'auto_scan_timer')
                self.status_label.setText(f"{'✓ 真实扫描 - 自动' if is_auto else '✓ 真实扫描 - '}扫描完成 - 涨幅榜 {gc} 个，跌幅榜 {lc} 个")
                self.scan_time_label.setText(f"上次扫描：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                self.count_label.setText(f"通过：{gc + lc}")
                self.scan_progress_detail.setText(f"涨幅榜:{gc}行, 跌幅榜:{lc}行")
                self.result_tabs.setCurrentIndex(0)
                if gc == 0 and lc == 0:
                    # 使用自动关闭弹窗
                    dlg = AutoCloseDialog(
                        "提示",
                        "扫描完成，但未找到符合条件的交易对。",
                        self,
                        timeout=10
                    )
                    dlg.exec_()
        else:
            passed = sum(1 for r in results if r.get('passed'))
            is_auto = hasattr(self, 'auto_scan_timer')

            # 📊 结果排序并显示 - 只显示通过筛选的交易对
            table = self.result_table
            table.setRowCount(0)
            # 按得分降序排序，只显示通过的
            sorted_results = sorted(results, key=lambda x: x.get('score', 0), reverse=True)
            for r in sorted_results:
                # 只显示通过筛选的交易对
                if r.get('passed'):
                    self.add_result_row(r)

            self.scan_time_label.setText(f"上次扫描：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            self.count_label.setText(f"通过：{passed} / {len(results)}")
            self.scan_progress_detail.setText(f"结果数: {len(results)}, 通过: {passed}")
            self.status_label.setText(f"{'✓ 真实扫描 - 自动' if is_auto else '✓ 真实扫描 - '}扫描完成 - {passed}/{len(results)} 个交易对符合条件")
            self.result_tabs.setCurrentIndex(2)

            # 📦 自动入库 - 将通过筛选的交易对添加到交易对池
            self.update_pool(results)

            if passed == 0 and len(results) > 0:
                # 使用自动关闭弹窗
                dlg = AutoCloseDialog(
                    "提示",
                    f"扫描完成，共扫描 {len(results)} 个交易对，但其中没有符合条件的交易对。",
                    self,
                    timeout=10
                )
                dlg.exec_()
            elif len(results) == 0:
                # 使用自动关闭弹窗
                dlg = AutoCloseDialog(
                    "提示",
                    "扫描完成，但未找到任何交易对数据。",
                    self,
                    timeout=10
                )
                dlg.exec_()

    def update_pool(self, results):
        """将符合条件的交易对加入池中"""
        new_count = 0
        for r in results:
            # 只添加通过筛选的交易对
            if r.get('passed'):
                symbol = r.get('symbol', '')
                # 检查是否已存在（避免重复添加）
                if not any(p['symbol'] == symbol for p in self.trading_pool):
                    self.trading_pool.append({
                        'time': datetime.now().strftime('%H:%M:%S'),
                        'symbol': symbol,
                        'price': r.get('last_price', 0),
                        'change': r.get('price_change_24h', 0),
                        'score': r.get('score', 0),
                        'strategy': self.current_strategy.__class__.__name__ if self.current_strategy else '未知'
                    })
                    new_count += 1
                else:
                    # 如果已存在，更新数据
                    for p in self.trading_pool:
                        if p['symbol'] == symbol:
                            p['time'] = datetime.now().strftime('%H:%M:%S')
                            p['price'] = r.get('last_price', 0)
                            p['change'] = r.get('price_change_24h', 0)
                            p['score'] = r.get('score', 0)
                            p['strategy'] = self.current_strategy.__class__.__name__ if self.current_strategy else '未知'
                            break
        if new_count > 0:
            self.update_pool_table()
            self.save_pool()
            self.status_label.setText(f"✅ 新增 {new_count} 个交易对到交易对池")

    def update_pool_table(self):
        """刷新交易对池表格（按时间由新到旧排序）"""
        # 按时间降序排序（新到旧）
        sorted_pool = sorted(self.trading_pool, key=lambda x: x.get('time', ''), reverse=True)
        
        self.pool_table.setRowCount(0)
        for i, p in enumerate(sorted_pool):
            self.pool_table.insertRow(i)
            self.pool_table.setItem(i, 0, QTableWidgetItem(p['time']))
            self.pool_table.setItem(i, 1, QTableWidgetItem(p['symbol'].replace('-USDT-SWAP', '/USDT')))
            self.pool_table.setItem(i, 2, QTableWidgetItem(f"{p['price']:.6f}"))
            chg_item = QTableWidgetItem(f"{p['change']:+.2f}%")
            chg_item.setForeground(QColor("#00ff00" if p['change'] >= 0 else "#ff4444"))
            self.pool_table.setItem(i, 3, chg_item)
            score_item = QTableWidgetItem(f"{p['score']:.1f}%")
            score_item.setForeground(QColor("#00ff00" if p['score'] >= 70 else "#ffaa00" if p['score'] >= 50 else "#ff4444"))
            self.pool_table.setItem(i, 4, score_item)
            self.pool_table.setItem(i, 5, QTableWidgetItem(p['strategy']))
            
            btn = QPushButton("移除")
            btn.setStyleSheet("QPushButton { background-color: #ff4444; color: white; border-radius: 4px; padding: 4px; }")
            # 注意：移除按钮需要对应原始列表的索引，这里使用符号匹配查找
            btn.clicked.connect(lambda checked, sym=p['symbol']: self.remove_from_pool_by_symbol(sym))
            self.pool_table.setCellWidget(i, 6, btn)

    def remove_from_pool_by_symbol(self, symbol):
        """按交易对符号从池中移除"""
        self.trading_pool = [p for p in self.trading_pool if p['symbol'] != symbol]
        self.update_pool_table()
        self.save_pool()
        self.status_label.setText(f"🗑 已移除: {symbol}")

    def update_ranking_tables(self, ranking_data: dict):
        gt, lt = self.gainers_table, self.losers_table
        gt.setRowCount(0); lt.setRowCount(0)
        for item in ranking_data.get('top_gainers', []): self.add_ranking_row(gt, item, "gainer")
        for item in ranking_data.get('top_losers', []): self.add_ranking_row(lt, item, "loser")

    def update_unilateral_trend_table(self, data: dict):
        """更新单边趋势扫描结果表格"""
        table = self.result_table
        table.setRowCount(0)
        
        # 合并做多和做空机会
        all_opportunities = []
        for opp in data.get('long_opportunities', []):
            opp['position'] = '做多'
            all_opportunities.append(opp)
        for opp in data.get('short_opportunities', []):
            opp['position'] = '做空'
            all_opportunities.append(opp)
        
        # 按得分降序排序
        all_opportunities.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        # 添加到表格
        for opp in all_opportunities:
            row = table.rowCount()
            table.insertRow(row)
            
            # 交易对
            symbol = opp.get('symbol', '').replace('-USDT-SWAP', '/USDT')
            table.setItem(row, 0, QTableWidgetItem(symbol))
            
            # 最新价
            table.setItem(row, 1, QTableWidgetItem(f"{opp.get('last_price', 0):.6f}"))
            
            # 24h成交量
            vol = opp.get('volume_24h', 0)
            vol_str = f"{vol/1e6:.2f}M" if vol >= 1e6 else f"{vol/1e3:.2f}K"
            table.setItem(row, 2, QTableWidgetItem(vol_str))
            
            # 24h涨跌幅
            chg = opp.get('price_change_24h', 0)
            chg_item = QTableWidgetItem(f"{chg:+.2f}%")
            chg_item.setForeground(QColor("#00ff00" if chg >= 0 else "#ff4444"))
            table.setItem(row, 3, chg_item)
            
            # 24h最高
            table.setItem(row, 4, QTableWidgetItem(f"{opp.get('high_24h', 0):.6f}"))
            
            # 24h最低
            table.setItem(row, 5, QTableWidgetItem(f"{opp.get('low_24h', 0):.6f}"))
            
            # 方向
            pos_item = QTableWidgetItem(opp.get('position', ''))
            pos_item.setForeground(QColor("#00ff00" if opp.get('position') == '做多' else "#ff4444"))
            table.setItem(row, 6, pos_item)
            
            # 得分
            sc = opp.get('score', 0)
            sc_item = QTableWidgetItem(f"{sc}分")
            if sc >= 90:
                sc_item.setForeground(QColor("#00ff00"))  # 绿色
            elif sc >= 75:
                sc_item.setForeground(QColor("#00ccff"))  # 蓝色
            elif sc >= 65:
                sc_item.setForeground(QColor("#ffaa00"))  # 橙色
            else:
                sc_item.setForeground(QColor("#ff4444"))  # 红色
            sc_item.setData(Qt.UserRole, sc)
            table.setItem(row, 7, sc_item)
            
            # 评级
            rating = opp.get('rating', '')
            table.setItem(row, 8, QTableWidgetItem(rating))
            
            # 信号 (前3个)
            signals = opp.get('signals', [])
            signals_str = " | ".join(signals[:3]) if signals else "无信号"
            table.setItem(row, 9, QTableWidgetItem(signals_str))

    def add_ranking_row(self, table, item, rank_type):
        row = table.rowCount(); table.insertRow(row)
        table.setItem(row, 0, QTableWidgetItem(f"#{item.get('rank', '')}"))
        table.setItem(row, 1, QTableWidgetItem(item.get('symbol', '').replace('-USDT-SWAP', '/USDT')))
        table.setItem(row, 2, QTableWidgetItem(f"{item.get('last_price', 0):.6f}"))
        chg = item.get('price_change_24h', 0)
        chg_item = QTableWidgetItem(f"{chg:+.2f}%"); chg_item.setTextAlignment(Qt.AlignCenter); chg_item.setForeground(QColor("#00ff00" if rank_type=="gainer" else "#ff4444"))
        table.setItem(row, 3, chg_item)
        vol = item.get('volume_24h', 0)
        table.setItem(row, 4, QTableWidgetItem(f"{vol/1e6:.2f}M" if vol>=1e6 else f"{vol/1e3:.2f}K"))
        table.setItem(row, 5, QTableWidgetItem(f"{item.get('high_24h', 0):.6f}"))
        table.setItem(row, 6, QTableWidgetItem(f"{item.get('low_24h', 0):.6f}"))

    def add_result_row(self, result):
        table = self.result_table; row = table.rowCount(); table.insertRow(row)
        table.setItem(row, 0, QTableWidgetItem(result.get('symbol', '').replace('-USDT-SWAP', '/USDT')))
        table.setItem(row, 1, QTableWidgetItem(f"{result.get('last_price', 0):.6f}"))
        vol = result.get('volume_24h', 0)
        table.setItem(row, 2, QTableWidgetItem(f"{vol/1e6:.2f}M" if vol>=1e6 else f"{vol/1e3:.2f}K"))
        chg = result.get('price_change_24h', 0)
        chg_item = QTableWidgetItem(f"{chg:.2f}%"); chg_item.setForeground(QColor("#00ff00" if chg>=0 else "#ff4444")); table.setItem(row, 3, chg_item)
        table.setItem(row, 4, QTableWidgetItem(f"{result.get('high_24h', 0):.6f}"))
        table.setItem(row, 5, QTableWidgetItem(f"{result.get('low_24h', 0):.6f}"))
        table.setItem(row, 6, QTableWidgetItem(f"{result.get('conditions_met', 0)}/{result.get('conditions_total', 0)}"))
        sc = result.get('score', 0)
        sc_item = QTableWidgetItem(f"{sc:.1f}%"); sc_item.setForeground(QColor("#00ff00" if sc>=80 else "#ffaa00" if sc>=50 else "#ff4444")); sc_item.setData(Qt.UserRole, sc); table.setItem(row, 7, sc_item)
        st = QTableWidgetItem("✓ 通过" if result.get('passed') else "✗ 未通过"); st.setForeground(QColor("#00ff00" if result.get('passed') else "#ff4444")); table.setItem(row, 8, st)
        table.setItem(row, 9, QTableWidgetItem(" | ".join([f"{k}: {v}" for k, v in list(result.get('details', {}).items())[:2]])))