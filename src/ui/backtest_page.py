"""
回测页面模块 - 修复版
"""

import sys
from datetime import datetime
from typing import Dict, Optional

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QComboBox, QDateEdit, QSpinBox, QDoubleSpinBox, QGroupBox,
    QFormLayout, QTableWidget, QTableWidgetItem, QTextEdit,
    QProgressBar, QFileDialog, QMessageBox, QSplitter,
    QScrollArea, QFrame, QHeaderView, QTabWidget, QApplication,
    QListWidget, QListWidgetItem
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject, QDate
from PyQt5.QtGui import QFont, QColor

from src.strategy.loader import StrategyLoader, StrategyInfo
from src.backtest.engine import Backtester, BacktestResult, BacktestAnalyzer


class BacktestConfigWidget(QWidget):
    """回测配置组件"""

    start_backtest = pyqtSignal(dict)
    cancel_backtest = pyqtSignal()

    def __init__(self, strategy_loader: StrategyLoader = None):
        super().__init__()
        self.strategy_loader = strategy_loader or StrategyLoader()
        self.selected_strategy: Optional[StrategyInfo] = None
        self.config_inputs: Dict[str, QWidget] = {}
        self.init_ui()

    def init_ui(self):
        """初始化 UI"""
        layout = QVBoxLayout(self)
        layout.setSpacing(15)

        # 标题
        title = QLabel("回测配置")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setStyleSheet("color: #ffffff;")
        layout.addWidget(title)

        # 策略选择
        strategy_group = QGroupBox("策略选择")
        strategy_layout = QVBoxLayout(strategy_group)
        strategy_layout.setSpacing(10)

        # 使用 QListWidget
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
        strategy_layout.addWidget(self.strategy_list)

        # 刷新按钮
        refresh_btn = QPushButton("刷新策略列表")
        refresh_btn.clicked.connect(self.refresh_strategies)
        strategy_layout.addWidget(refresh_btn)

        layout.addWidget(strategy_group)

        # 策略说明
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
        layout.addWidget(self.strategy_desc)

        # 市场配置
        market_group = QGroupBox("市场配置")
        market_layout = QFormLayout()

        self.pair_combo = QComboBox()
        self.pair_combo.setEditable(True)
        self.pair_combo.addItems([
            "BTC-USDT", "ETH-USDT", "SOL-USDT", "XRP-USDT",
            "ADA-USDT", "AVAX-USDT", "DOGE-USDT", "DOT-USDT"
        ])
        market_layout.addRow("交易对:", self.pair_combo)

        self.bar_combo = QComboBox()
        self.bar_combo.addItems(["1m", "5m", "15m", "30m", "1H", "2H", "4H", "1D"])
        self.bar_combo.setCurrentText("1H")
        market_layout.addRow("K 线周期:", self.bar_combo)

        market_group.setLayout(market_layout)
        layout.addWidget(market_group)

        # 时间范围
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

        time_group.setLayout(time_layout)
        layout.addWidget(time_group)

        # 资金配置
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
        capital_layout.addRow("仓位比例:", self.position_size)

        capital_group.setLayout(capital_layout)
        layout.addWidget(capital_group)

        # 策略参数
        self.params_group = QGroupBox("策略参数")
        self.params_layout = QFormLayout()
        self.params_group.setLayout(self.params_layout)
        layout.addWidget(self.params_group)

        layout.addStretch()

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

        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.setMinimumHeight(50)
        self.cancel_btn.setStyleSheet("""
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
        """)
        self.cancel_btn.clicked.connect(self.on_cancel_backtest)
        self.cancel_btn.setEnabled(False)
        btn_layout.addWidget(self.cancel_btn)

        layout.addLayout(btn_layout)

        # 初始化
        self.refresh_strategies()

    def refresh_strategies(self):
        """刷新策略列表"""
        self.strategy_list.clear()
        self.strategy_list.addItem("请选择策略...")

        strategies = self.strategy_loader.discover_strategies()
        for strategy in strategies:
            item = QListWidgetItem(f"{strategy.name} ({strategy.type.value})")
            item.setData(Qt.UserRole, strategy)
            self.strategy_list.addItem(item)

    def on_strategy_item_clicked(self, item):
        """策略列表项点击事件"""
        text = item.text()
        if text == "请选择策略...":
            return

        self.selected_strategy = item.data(Qt.UserRole)

        # 清空参数配置
        while self.params_layout.count():
            child = self.params_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        self.config_inputs.clear()

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
                for param_name, param_info in self.selected_strategy.config_schema.items():
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

                    self.params_layout.addRow(f"{label}:", input_widget)
                    self.config_inputs[param_name] = input_widget
        else:
            self.strategy_desc.clear()

    def on_start_backtest(self):
        """开始回测"""
        config = self.get_config()
        self.start_backtest.emit(config)

    def on_cancel_backtest(self):
        """取消回测"""
        self.cancel_backtest.emit()

    def get_config(self) -> Dict:
        """获取配置"""
        strategy = self.selected_strategy
        if not strategy:
            return {}

        strategy_params = {}
        for param_name, input_widget in self.config_inputs.items():
            if hasattr(input_widget, 'isChecked'):
                strategy_params[param_name] = input_widget.isChecked()
            elif hasattr(input_widget, 'value'):
                strategy_params[param_name] = input_widget.value()

        return {
            'strategy_name': strategy.name,
            'strategy_path': strategy.path,
            'inst_id': self.pair_combo.currentText(),
            'bar': self.bar_combo.currentText(),
            'start_date': self.start_date.date().toString("yyyy-MM-dd"),
            'end_date': self.end_date.date().toString("yyyy-MM-dd"),
            'initial_capital': self.initial_capital.value(),
            'position_size': self.position_size.value(),
            'strategy_params': strategy_params
        }

    def set_running(self, running: bool):
        """设置运行状态"""
        self.start_btn.setEnabled(not running)
        self.cancel_btn.setEnabled(running)


class BacktestResultWidget(QWidget):
    """回测结果展示组件"""

    def __init__(self):
        super().__init__()
        self.current_result: Optional[BacktestResult] = None
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

        # 交易记录页面
        trades_widget = QWidget()
        trades_layout = QVBoxLayout(trades_widget)

        self.trades_table = QTableWidget()
        self.trades_table.setColumnCount(7)
        self.trades_table.setHorizontalHeaderLabels([
            "序号", "方向", "开仓时间", "平仓时间", "开仓价", "平仓价", "盈亏"
        ])
        self.trades_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        trades_layout.addWidget(self.trades_table)

        self.tabs.addTab(trades_widget, "交易记录")

        # 报告页面
        report_widget = QWidget()
        report_layout = QVBoxLayout(report_widget)

        self.report_text = QTextEdit()
        self.report_text.setReadOnly(True)
        self.report_text.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #00ff00;
                font-family: 'Courier New', monospace;
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

    def display_result(self, result: BacktestResult):
        """显示回测结果"""
        self.current_result = result

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
            ("总交易次数", str(result.total_trades)),
            ("胜率", f"{result.win_rate:.2f}%"),
            ("盈亏比", f"{result.profit_factor:.2f}"),
            ("平均盈利", f"{result.avg_win:,.2f} USDT"),
            ("平均亏损", f"{result.avg_loss:,.2f} USDT"),
        ]

        for label, value in metrics:
            row = self.metrics_table.rowCount()
            self.metrics_table.insertRow(row)
            self.metrics_table.setItem(row, 0, QTableWidgetItem(label))
            self.metrics_table.setItem(row, 1, QTableWidgetItem(str(value)))

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
                self.trades_table.setItem(row, 4, QTableWidgetItem(f"{trade.entry_price:.4f}"))
                self.trades_table.setItem(row, 5, QTableWidgetItem(f"{trade.exit_price:.4f}"))
                pnl_item = QTableWidgetItem(f"{trade.pnl:.2f} ({trade.pnl_percent:.2f}%)")
                self.trades_table.setItem(row, 6, pnl_item)

        report = BacktestAnalyzer.generate_report(result)
        self.report_text.setText(report)

    def export_report(self):
        """导出报告"""
        if not self.current_result:
            QMessageBox.warning(self, "警告", "没有可导出的回测结果")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存回测报告", "",
            "Text Files (*.txt);;All Files (*)"
        )

        if file_path:
            try:
                report = BacktestAnalyzer.generate_report(self.current_result)
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(report)
                QMessageBox.information(self, "成功", f"报告已保存到:\\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"导出失败:\\n{str(e)}")


class BacktestWorker(QObject):
    """回测工作线程"""

    finished = pyqtSignal(object)
    progress = pyqtSignal(int, str)
    error = pyqtSignal(str)
    log = pyqtSignal(str, str)

    def __init__(self, strategy, config: Dict, okx_client=None):
        super().__init__()
        self.strategy = strategy
        self.config = config
        self.okx_client = okx_client
        self._stop_flag = False

    def run(self):
        """运行回测"""
        try:
            self.log.emit("初始化回测引擎...", "INFO")

            backtester = Backtester(
                okx_client=self.okx_client,
                initial_capital=self.config.get('initial_capital', 10000)
            )

            self.log.emit(f"开始回测 {self.config['inst_id']}...", "INFO")
            self.progress.emit(10, "获取历史数据")

            result = backtester.run_backtest(
                strategy=self.strategy,
                inst_id=self.config['inst_id'],
                start_date=self.config['start_date'],
                end_date=self.config['end_date'],
                bar=self.config['bar'],
                config=self.config.get('strategy_params', {})
            )

            self.progress.emit(100, "回测完成")
            self.log.emit("回测完成，正在分析结果...", "SUCCESS")

            self.finished.emit(result)

        except Exception as e:
            self.error.emit(f"回测失败：{str(e)}")
            self.log.emit(f"错误：{str(e)}", "ERROR")

    def stop(self):
        """停止回测"""
        self._stop_flag = True


class BacktestPage(QWidget):
    """回测页面"""

    def __init__(self, okx_client=None):
        super().__init__()
        self.okx_client = okx_client
        self.strategy_loader = StrategyLoader()
        self.backtest_worker: Optional[BacktestWorker] = None
        self.backtest_thread: Optional[QThread] = None
        self.init_ui()

    def init_ui(self):
        """初始化 UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(15)

        splitter = QSplitter(Qt.Horizontal)

        self.config_widget = BacktestConfigWidget(self.strategy_loader)
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
                font-family: 'Courier New', monospace;
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
        self.config_widget.cancel_backtest.connect(self.cancel_backtest)

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

        strategy_instance = strategy_class(config.get('strategy_params', {}))

        self.backtest_thread = QThread()
        self.backtest_worker = BacktestWorker(
            strategy=strategy_instance,
            config=config,
            okx_client=self.okx_client
        )
        self.backtest_worker.moveToThread(self.backtest_thread)

        self.backtest_thread.started.connect(self.backtest_worker.run)
        self.backtest_worker.finished.connect(self.on_backtest_finished)
        self.backtest_worker.progress.connect(self.on_backtest_progress)
        self.backtest_worker.error.connect(self.on_backtest_error)
        self.backtest_worker.log.connect(self.log)

        self.backtest_thread.start()

    def cancel_backtest(self):
        """取消回测"""
        if self.backtest_worker:
            self.backtest_worker.stop()
            self.log("正在取消回测...", "WARNING")

    def on_backtest_finished(self, result: BacktestResult):
        """回测完成"""
        self.log("回测完成", "SUCCESS")
        self.result_widget.display_result(result)
        self.config_widget.set_running(False)
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
        QMessageBox.critical(self, "错误", error)
