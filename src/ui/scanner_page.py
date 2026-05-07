import os
import json
import threading
import time
import hashlib
from html import escape
from datetime import datetime

from src.strategy.loader import StrategyLoader
from src.scanner.engine import ScanEngine
from src.scanner.ranking import enrich_scan_result, sort_scan_results
from src.trading.entry_rule_guard import evaluate_entry_rule_from_klines, normalize_direction
from src.qt_compat import QCheckBox, QColor, QComboBox, QDoubleSpinBox, QFormLayout, QGridLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel, QListWidget, QListWidgetItem, QMenu, QMessageBox, QProgressBar, QPushButton, QScrollArea, QSpinBox, QTabWidget, QTableWidget, QTableWidgetItem, QTextEdit, QThread, QTimer, QVBoxLayout, QWidget, Qt, Signal

class ScanThread(QThread):
    """异步扫描线程 - 支持批量结果推送和实时进度"""
    finished = Signal(list)
    result_found = Signal(list)
    result_found_single = Signal(dict)
    progress = Signal(int, str, str, int, int, str)
    error = Signal(str)

    def __init__(self, engine, strategy):
        super().__init__()
        self.engine = engine
        self.strategy = strategy
        self._stop_flag = False
        self._buffer = []
        self._progress_counter = 0
        self._result_counter = 0

    def stop(self):
        self._stop_flag = True

    def _flush_buffer(self):
        if self._buffer:
            batch = list(self._buffer)
            self._buffer.clear()
            self.result_found.emit(batch)

    def run(self):
        import gc
        try:
            if self._stop_flag:
                self.finished.emit([])
                return

            def on_progress(val, msg, symbol, scanned, total, remaining):
                if self._stop_flag:
                    return True
                self._progress_counter += 1
                if self._progress_counter % 8 != 0 and scanned < total:
                    return False
                self.progress.emit(val, msg, symbol, scanned, total, remaining)
                return False

            def on_result(res):
                if self._stop_flag:
                    return
                self._buffer.append(res)
                self._result_counter += 1
                if len(self._buffer) >= 10:
                    self._flush_buffer()
                    gc.collect()

            results = self.engine.run_scan(
                self.strategy,
                progress_callback=on_progress,
                result_callback=on_result
            )

            self._flush_buffer()

            if self._stop_flag:
                return

            results = sort_scan_results(results)
            gc.collect()
            self.finished.emit(results)
        except Exception as e:
            if self._stop_flag:
                return
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))

class ScannerPage(QWidget):
    """交易对扫描页面"""
    # 扫描完成后向外广播结果（供智能助理等订阅）
    scan_results_ready = Signal(list)
    # 扫描状态日志（供外部面板订阅实时进度）：(message, level)
    scan_log_signal = Signal(str, str)

    RESULT_HEADERS = ["交易对", "价格", "24h涨幅", "机会类型", "信号强度", "连续轮数", "建议方向", "扫描理由", "更新时间"]
    CATEGORY_TABS = [
        "总览", "突破启动", "新高突破", "单边趋势", "趋势回踩", "背离反转", "超跌反转",
        "波动率收缩爆发", "趋势回踩二次启动", "中继再启动", "资金费率反转", "持仓量异常", "放量承接",
        "BTC/ETH牵引", "热度跃迁"
    ]
    LEVEL_OPTIONS = ["全部等级", "S", "A", "B", "C", "D"]
    DIRECTION_OPTIONS = ["全部方向", "BUY", "SELL", "LONG", "SHORT"]
    QUADRANT_OPTIONS = [
        "全部象限",
        "新出现且共振",
        "新出现但未共振",
        "连续强化且共振",
        "连续强化但未共振",
    ]
    SORT_OPTIONS = [
        "按机会评分",
        "按连续轮数",
        "按原始分数",
        "按24H成交额",
        "按24H涨跌幅",
        "按更新时间",
    ]

    def __init__(self, okx_client, trade_pool_page=None, rl_learning_page=None):
        super().__init__()
        self.okx_client = okx_client
        self.engine = ScanEngine(okx_client)
        import os
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.strategy_loader = StrategyLoader(os.path.join(project_root, 'strategies'))
        self.trade_pool_page = trade_pool_page
        self.rl_learning_page = rl_learning_page
        self.current_strategy_name = "未知策略"  # 保存当前执行的策略名称
        self._scan_thread = None
        self._log_progress_counter = 0   # scan_log_signal 节流计数器
        self.is_paused = False
        self.latest_results = []
        self.raw_results = []
        self.aggregate_scan_results = []
        self.batch_scan_active = False
        self.previous_scan_keys = set()
        self.previous_scan_streaks = {}
        self.result_tables = {}
        self.result_stats_labels = {}
        self.strategy_config_inputs = {}
        self.current_config_strategy_name = None
        self.show_advanced_strategy_params = False
        self._current_strategy_config = {}
        self._pool_live_keys = set()
        self._load_rejected_signals()
        # 策略置顶
        self._pinned_strategies: set = set()
        self._pin_config_path = os.path.join(project_root, 'src', 'scanner_pinned_strategies.json')
        self._load_pinned_strategies()
        self.init_ui()
        self.scan_log_signal.connect(self._append_scan_log_line)
        
        # 初始刷新余额
        QTimer.singleShot(500, self.refresh_balance)
        
        # 定时器设置
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.start_scan)

        # 防抖定时器：扫描结果批量到达时，延迟 300ms 再刷新表格，避免每批都重建
        self._refresh_debounce_timer = QTimer(self)
        self._refresh_debounce_timer.setSingleShot(True)
        self._refresh_debounce_timer.timeout.connect(self.refresh_result_view)

        # 僵尸线程列表：旧 ScanThread 未在 200ms 内退出时，暂存于此保持 Python 引用，
        # 防止 GC 在 OS 线程仍运行时销毁 QThread 对象（→ SIGABRT）。
        # 线程自然退出后通过 finished 信号自动从列表移除。
        self._zombie_scan_threads: list = []

    def _safe_discard_scan_thread(self, thread, wait_ms: int = 200):
        """
        安全丢弃旧 ScanThread 引用。

        等待最多 wait_ms 毫秒让线程自然退出。若仍在运行，则将其加入
        僵尸列表（_zombie_scan_threads）以维持 Python 引用，避免 GC 在
        OS 线程还在运行时销毁 QThread 对象（否则触发 SIGABRT）。
        线程最终退出时通过 finished 信号自动移出僵尸列表。
        """
        if thread is None:
            return
        if thread.isRunning():
            if not thread.wait(wait_ms):          # 返回 False = 超时，线程仍在运行
                # 保持引用直到线程自然结束
                if thread not in self._zombie_scan_threads:
                    self._zombie_scan_threads.append(thread)
                    thread.finished.connect(
                        lambda t=thread: (
                            self._zombie_scan_threads.remove(t)
                            if t in self._zombie_scan_threads else None
                        )
                    )

    def shutdown(self, wait_ms: int = 8000):
        """关闭页面前尽量停止后台扫描线程，避免 QThread 被提前销毁。"""
        try:
            self.auto_scan_enabled = False
            self.current_auto_strategies = []
            if hasattr(self, "timer"):
                self.timer.stop()
            if hasattr(self, "countdown_timer"):
                self.countdown_timer.stop()
            if hasattr(self, "_refresh_debounce_timer"):
                self._refresh_debounce_timer.stop()
        except Exception:
            pass

        old_thread = getattr(self, "_scan_thread", None)
        if old_thread is not None and old_thread.isRunning():
            try:
                self.engine.request_stop()
                old_thread.stop()
                self.status_label.setText("正在停止扫描线程...")
            except Exception:
                pass
            old_thread.wait(wait_ms)
        self._scan_thread = None
        self.is_scanning = False

    def closeEvent(self, event):
        self.shutdown()
        super().closeEvent(event)

    def refresh_balance(self):
        """刷新 USDT 余额（后台线程，避免阻塞 UI）"""
        import threading

        def _fetch():
            try:
                from src.trading.executor import TradeExecutor
                executor = TradeExecutor(self.okx_client)

                raw_balance = self.okx_client.get_balance()
                print(f"[调试] OKX 余额接口返回: {raw_balance}")

                balance = executor.get_usdt_balance()
                print(f"[调试] 解析后的 USDT 余额: {balance}")

                # 回到主线程更新 UI
                QTimer.singleShot(0, lambda: self._update_balance_ui(balance))
            except Exception as e:
                print(f"[错误] 获取余额失败: {e}")
                QTimer.singleShot(0, lambda: self._update_balance_ui_error(str(e)))

        threading.Thread(target=_fetch, daemon=True).start()

    def _update_balance_ui(self, balance: float):
        """主线程更新余额 UI"""
        if balance > 0:
            self.api_status_label.setText(f"💰 账户余额: {balance:.2f} USDT")
            self.api_status_label.setStyleSheet("color: #00ffaa; font-weight: bold; font-size: 13px;")
        else:
            self.api_status_label.setText(f"⚠️ 账户余额: {balance:.2f} USDT (请检查API密钥和账户类型)")
            self.api_status_label.setStyleSheet("color: #ffaa00; font-weight: bold; font-size: 13px;")

    def _update_balance_ui_error(self, error: str):
        """主线程更新余额错误 UI"""
        self.api_status_label.setText(f"❌ 账户余额: 获取失败 ({error})")
        self.api_status_label.setStyleSheet("color: #ff4444; font-weight: bold; font-size: 13px;")

    def init_ui(self):
        layout = QVBoxLayout(self)

        # 1. 顶部控制栏
        control_group = QGroupBox("扫描控制")
        control_group.setMaximumHeight(130)  # 限制高度
        control_layout = QVBoxLayout(control_group)
        control_layout.setContentsMargins(5, 5, 5, 5)
        control_layout.setSpacing(5)

        # 第一行：策略多选列表
        strategies_layout = QHBoxLayout()
        strategies_layout.addWidget(QLabel("选择策略:"))
        
        self.strategy_list = QListWidget()
        self.strategy_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        self.strategy_list.setMaximumHeight(80)
        self.strategy_list.itemSelectionChanged.connect(self.refresh_strategy_config_panel)
        self.strategy_list.currentItemChanged.connect(lambda *_: self.refresh_strategy_config_panel())
        self.strategy_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.strategy_list.customContextMenuRequested.connect(self._on_strategy_list_context_menu)
        strategies_layout.addWidget(self.strategy_list, 1)
        control_layout.addLayout(strategies_layout)

        # 第二行：控制按钮和间隔设置
        row2_layout = QHBoxLayout()
        
        # 间隔设置
        row2_layout.addWidget(QLabel("间隔(分):"))
        self.interval_spin = QSpinBox()
        self.interval_spin.setRange(1, 1440)
        self.interval_spin.setValue(5)
        row2_layout.addWidget(self.interval_spin)

        # 倒计时标签
        self.countdown_label = QLabel("⏱ 下次扫描：未启动")
        self.countdown_label.setStyleSheet("color: #ffaa00; font-size: 12px; font-weight: bold;")
        row2_layout.addWidget(self.countdown_label)

        row2_layout.addStretch()

        # 手动扫描按钮
        self.scan_btn = QPushButton("🚀 手动扫描")
        self.scan_btn.clicked.connect(lambda: self.start_scan())
        self.scan_btn.setStyleSheet("background-color: #007acc; color: white; font-weight: bold; padding: 5px 15px;")
        row2_layout.addWidget(self.scan_btn)

        # 暂停/恢复按钮（已禁用，因多线程稳定性问题）
        self.pause_btn = QPushButton("⏸ 暂停扫描")
        self.pause_btn.setEnabled(False)
        self.pause_btn.setStyleSheet("background-color: #555555; color: #888888; font-weight: bold; padding: 5px 15px;")
        row2_layout.addWidget(self.pause_btn)

        # 停止按钮
        self.stop_btn = QPushButton("⏹ 停止扫描")
        self.stop_btn.clicked.connect(lambda: self.stop_scan())
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("background-color: #dc3545; color: white; font-weight: bold; padding: 5px 15px;")
        row2_layout.addWidget(self.stop_btn)

        # 定时扫描按钮
        self.auto_scan_btn = QPushButton("⏱ 启动定时扫描")
        self.auto_scan_btn.clicked.connect(lambda: self.toggle_auto_scan())
        self.auto_scan_btn.setStyleSheet("background-color: #28a745; color: white; font-weight: bold; padding: 5px 15px;")
        row2_layout.addWidget(self.auto_scan_btn)

        control_layout.addLayout(row2_layout)
        layout.addWidget(control_group)

        # 1.5 策略参数面板
        self.strategy_config_group = QGroupBox("策略参数")
        strategy_config_outer_layout = QVBoxLayout(self.strategy_config_group)
        strategy_config_outer_layout.setContentsMargins(6, 4, 6, 4)

        self.strategy_config_scroll = QScrollArea()
        self.strategy_config_scroll.setWidgetResizable(True)
        self.strategy_config_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.strategy_config_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        self.strategy_config_container = QWidget()
        self.strategy_config_layout = QGridLayout(self.strategy_config_container)
        self.strategy_config_layout.setContentsMargins(8, 6, 8, 6)
        self.strategy_config_layout.setSpacing(6)
        self.strategy_config_layout.setColumnStretch(1, 1)
        self.strategy_config_layout.setColumnStretch(3, 1)
        self.strategy_config_scroll.setWidget(self.strategy_config_container)
        strategy_config_outer_layout.addWidget(self.strategy_config_scroll)
        self.strategy_config_group.setMinimumHeight(130)
        self.strategy_config_group.setMaximumHeight(220)
        layout.addWidget(self.strategy_config_group)
        self.refresh_strategies_list()
        self.refresh_strategy_config_panel()

        # 倒计时定时器
        self.countdown_timer = QTimer(self)
        self.countdown_timer.timeout.connect(self.update_countdown)
        self.countdown_remaining = 0
        self.auto_scan_enabled = False
        self.current_auto_strategies = [] # 待扫描的策略队列
        
        # 暂停/停止状态（暂停功能已移除）
        self.is_scanning = False

        # 2. 实时扫描信息栏 (增强版)
        info_group = QGroupBox("实时扫描状态")
        info_layout = QVBoxLayout(info_group)

        # 第一行：账户余额和当前扫描信息
        row1_layout = QHBoxLayout()

        # 账户余额 (代替之前的测试网显示)
        self.api_status_label = QLabel("💰 账户余额: 加载中...")
        self.api_status_label.setStyleSheet("color: #00ffaa; font-weight: bold; font-size: 13px;")
        row1_layout.addWidget(self.api_status_label)

        # 当前正在扫描的交易对
        self.current_symbol_label = QLabel("等待扫描...")
        self.current_symbol_label.setStyleSheet("color: #00ccff; font-weight: bold; font-size: 13px;")
        row1_layout.addWidget(self.current_symbol_label, 1)

        # 当前执行的策略
        self.current_strategy_label = QLabel("策略: -")
        self.current_strategy_label.setStyleSheet("color: #ffaa00; font-weight: bold; font-size: 13px;")
        row1_layout.addWidget(self.current_strategy_label)

        # 进度统计
        self.scan_stats_label = QLabel("")
        self.scan_stats_label.setStyleSheet("color: #ffaa00; font-size: 12px;")
        row1_layout.addWidget(self.scan_stats_label)

        info_layout.addLayout(row1_layout)

        # 第二行：扫描进度文本
        row2_layout = QHBoxLayout()
        self.scan_progress_label = QLabel("等待开始扫描...")
        self.scan_progress_label.setStyleSheet("color: #00ccff; font-size: 12px;")
        row2_layout.addWidget(self.scan_progress_label, 1)
        info_layout.addLayout(row2_layout)

        # 第三行：实时日志窗口 (新增)
        row3_layout = QHBoxLayout()
        self.scan_log_label = QLabel("📋 实时日志: 就绪")
        self.scan_log_label.setStyleSheet("color: #aaaaaa; font-size: 11px; font-family: 'Menlo', 'Monaco';")
        self.scan_log_label.setWordWrap(True)
        row3_layout.addWidget(self.scan_log_label, 1)
        info_layout.addLayout(row3_layout)

        self.scan_log_browser = QTextEdit()
        self.scan_log_browser.setReadOnly(True)
        self.scan_log_browser.setMaximumHeight(110)
        self.scan_log_browser.setStyleSheet("""
            QTextEdit {
                background-color: #15171a;
                color: #d7dde5;
                border: 1px solid #2b3138;
                border-radius: 6px;
                font-size: 11px;
                font-family: 'Menlo', 'Monaco', 'Courier New';
                padding: 4px;
            }
        """)
        info_layout.addWidget(self.scan_log_browser)

        # 被过滤信号按钮
        self.rejected_btn = QPushButton("📋 被过滤信号")
        self.rejected_btn.setStyleSheet("QPushButton{background:#3a2d2d;color:#ccc;border:1px solid #5a3a3a;border-radius:3px;padding:2px 8px;}")
        self.rejected_btn.setMaximumWidth(120)
        self.rejected_btn.clicked.connect(self._show_rejected_signals)
        info_layout.addWidget(self.rejected_btn)

        row4_layout = QHBoxLayout()
        self.live_signal_label = QLabel("🎯 最新命中: 暂无")
        self.live_signal_label.setStyleSheet("color: #7fd1ff; font-size: 11px;")
        self.live_signal_label.setWordWrap(True)
        row4_layout.addWidget(self.live_signal_label, 1)
        info_layout.addLayout(row4_layout)

        layout.addWidget(info_group)

        # 3. 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar { 
                height: 12px; 
                text-align: center; 
                border: 1px solid #444; 
                border-radius: 6px; 
                background-color: #1e1e1e;
            }
            QProgressBar::chunk {
                background-color: #007acc;
                border-radius: 5px;
            }
        """)
        layout.addWidget(self.progress_bar)

        # 4. 结果标签页
        filter_group = QGroupBox("结果过滤与排序")
        filter_layout = QHBoxLayout(filter_group)
        filter_layout.setContentsMargins(8, 6, 8, 6)
        filter_layout.setSpacing(8)

        filter_layout.addWidget(QLabel("方向:"))
        self.direction_filter_combo = QComboBox()
        self.direction_filter_combo.addItems(self.DIRECTION_OPTIONS)
        self.direction_filter_combo.currentTextChanged.connect(self.refresh_result_view)
        filter_layout.addWidget(self.direction_filter_combo)

        filter_layout.addWidget(QLabel("等级:"))
        self.level_filter_combo = QComboBox()
        self.level_filter_combo.addItems(self.LEVEL_OPTIONS)
        self.level_filter_combo.currentTextChanged.connect(self.refresh_result_view)
        filter_layout.addWidget(self.level_filter_combo)

        filter_layout.addWidget(QLabel("类型:"))
        self.category_filter_combo = QComboBox()
        self.category_filter_combo.addItems(["全部类型", "突破启动", "新高突破", "单边趋势", "趋势回踩", "趋势回踩二次启动", "背离反转", "超跌反转", "波动率收缩爆发", "中继再启动"])
        self.category_filter_combo.currentTextChanged.connect(self.refresh_result_view)
        filter_layout.addWidget(self.category_filter_combo)

        filter_layout.addWidget(QLabel("最低评分:"))
        self.min_score_filter_spin = QDoubleSpinBox()
        self.min_score_filter_spin.setRange(0.0, 100.0)
        self.min_score_filter_spin.setDecimals(1)
        self.min_score_filter_spin.setSingleStep(1.0)
        self.min_score_filter_spin.setValue(0.0)
        self.min_score_filter_spin.valueChanged.connect(self.refresh_result_view)
        filter_layout.addWidget(self.min_score_filter_spin)

        self.resonance_only_check = QCheckBox("只看共振")
        self.resonance_only_check.toggled.connect(self.refresh_result_view)
        filter_layout.addWidget(self.resonance_only_check)

        self.new_only_check = QCheckBox("只看最近新出现")
        self.new_only_check.toggled.connect(self.refresh_result_view)
        filter_layout.addWidget(self.new_only_check)

        filter_layout.addWidget(QLabel("象限:"))
        self.quadrant_filter_combo = QComboBox()
        self.quadrant_filter_combo.addItems(self.QUADRANT_OPTIONS)
        self.quadrant_filter_combo.currentTextChanged.connect(self.refresh_result_view)
        filter_layout.addWidget(self.quadrant_filter_combo)

        filter_layout.addWidget(QLabel("排序:"))
        self.sort_combo = QComboBox()
        self.sort_combo.addItems(self.SORT_OPTIONS)
        self.sort_combo.currentTextChanged.connect(self.refresh_result_view)
        filter_layout.addWidget(self.sort_combo)

        reset_filter_btn = QPushButton("重置")
        reset_filter_btn.clicked.connect(self.reset_result_filters)
        filter_layout.addWidget(reset_filter_btn)
        filter_layout.addStretch()
        layout.addWidget(filter_group)

        self.result_tabs = QTabWidget()
        for category_name in self.CATEGORY_TABS:
            tab_widget, stats_label, table = self._create_result_tab(category_name)
            self.result_tables[category_name] = table
            self.result_stats_labels[category_name] = stats_label
            self.result_tabs.addTab(tab_widget, category_name)
        self.result_table = self.result_tables["总览"]
        layout.addWidget(self.result_tabs)

        # 状态栏信息
        self.status_label = QLabel("就绪")
        layout.addWidget(self.status_label)

    # ── 策略列表右键菜单 ──
    def _on_strategy_list_context_menu(self, pos):
        """策略列表右键菜单（支持多选批量置顶/取消置顶）"""
        item = self.strategy_list.itemAt(pos)
        if not item:
            return
        # 如果右键点击的项不在已有选中中，则仅选中该项
        if not item.isSelected():
            self.strategy_list.clearSelection()
            item.setSelected(True)

        selected_items = self.strategy_list.selectedItems()
        if not selected_items:
            return

        names = []
        for si in selected_items:
            info = si.data(Qt.ItemDataRole.UserRole)
            if info and hasattr(info, 'name'):
                names.append(str(info.name))

        if not names:
            return

        any_pinned = any(n in getattr(self, '_pinned_strategies', set()) for n in names)
        all_pinned = all(n in getattr(self, '_pinned_strategies', set()) for n in names)

        menu = QMenu(self)
        menu.setStyleSheet("""
            QMenu { background-color: #2a2a2a; color: #dddddd; border: 1px solid #444; }
            QMenu::item:selected { background-color: #3a3a3a; }
        """)
        if all_pinned:
            label = f"📌 取消置顶（{len(names)}个）"
        elif any_pinned:
            label = f"📌 全部置顶（{len(names)}个）"
        else:
            label = f"📌 置顶（{len(names)}个）" if len(names) > 1 else "📌 置顶"
        pin_action = menu.addAction(label)
        pin_action.triggered.connect(lambda: self._toggle_batch_pin(names, all_pinned))
        menu.exec(self.strategy_list.viewport().mapToGlobal(pos))

    def _toggle_batch_pin(self, names: list, all_pinned: bool):
        """批量置顶/取消置顶"""
        for name in names:
            if all_pinned:
                self._pinned_strategies.discard(name)
            else:
                self._pinned_strategies.add(name)
        self._save_pinned_strategies()
        self.refresh_strategies_list()

    def _save_pinned_strategies(self):
        try:
            with open(self._pin_config_path, 'w', encoding='utf-8') as f:
                json.dump(list(self._pinned_strategies), f, ensure_ascii=False)
        except Exception:
            pass

    def _load_pinned_strategies(self):
        try:
            if os.path.exists(self._pin_config_path):
                with open(self._pin_config_path, 'r', encoding='utf-8') as f:
                    self._pinned_strategies = set(json.load(f))
        except Exception:
            self._pinned_strategies = set()

    def refresh_strategies_list(self):
        """加载所有扫描策略到多选列表（置顶策略排最前）"""
        if not hasattr(self, 'strategy_list'):
            return
        try:
            # 保存之前选中的策略名称
            previously_selected = set()
            for i in range(self.strategy_list.count()):
                item = self.strategy_list.item(i)
                if item and item.isSelected():
                    info = item.data(Qt.ItemDataRole.UserRole)
                    if info and hasattr(info, 'name'):
                        previously_selected.add(str(info.name))

            self.strategy_list.clear()
            strategies = self.strategy_loader.discover_strategies()
            if not strategies:
                return
            pinned = getattr(self, '_pinned_strategies', set())
            strategies = sorted(strategies, key=lambda s: (0 if str(s.name) in pinned else 1, str(s.name)))
            for s in strategies:
                name = str(s.name)
                tval = str(getattr(s.type, 'value', 'unknown'))
                prefix = '📌 ' if name in pinned else ''
                item = QListWidgetItem(f"{prefix}{name} ({tval})")
                item.setData(Qt.ItemDataRole.UserRole, s)
                if name in pinned:
                    item.setForeground(QColor("#ffd700"))
                self.strategy_list.addItem(item)
                # 恢复选中状态
                if name in previously_selected:
                    item.setSelected(True)
        except Exception as e:
            print(f"[ScannerPage] refresh_strategies_list 出错: {e}")
            import traceback; traceback.print_exc()
        if self.strategy_list.count() > 0 and self.strategy_list.currentItem() is None:
            self.strategy_list.setCurrentRow(0)
        if self.strategy_list.count() > 0 and not self.strategy_list.selectedItems():
            first_item = self.strategy_list.item(0)
            if first_item is not None:
                first_item.setSelected(True)
        if hasattr(self, 'strategy_config_layout') and hasattr(self, 'strategy_config_group'):
            self.refresh_strategy_config_panel()

    def refresh_strategy_config_panel(self):
        """刷新当前选中策略的参数面板。"""
        if not hasattr(self, 'strategy_config_layout') or not hasattr(self, 'strategy_config_group') or not hasattr(self, 'strategy_list'):
            return

        while self.strategy_config_layout.count():
            item = self.strategy_config_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        self.strategy_config_inputs = {}
        self.current_config_strategy_name = None

        selected_items = self.strategy_list.selectedItems()
        current_item = self.strategy_list.currentItem()
        target_item = current_item if current_item in selected_items else (selected_items[0] if selected_items else None)
        if not target_item:
            self.strategy_config_group.setTitle("策略参数")
            self.strategy_config_layout.addWidget(QLabel("请先选择一个策略"), 0, 0, 1, 4)
            return

        strategy_info = target_item.data(Qt.ItemDataRole.UserRole)
        if not strategy_info:
            self.strategy_config_layout.addWidget(QLabel("未找到策略信息"), 0, 0, 1, 4)
            return

        self.current_config_strategy_name = strategy_info.name
        self.strategy_config_group.setTitle(f"策略参数 - {strategy_info.name}")
        schema = strategy_info.config_schema or {}
        if not schema:
            self.strategy_config_layout.addWidget(QLabel("该策略没有可配置参数"), 0, 0, 1, 4)
            return

        if strategy_info.name != "AI截面双因子组合扫描器":
            self.show_advanced_strategy_params = False

        row_offset = 0
        if len(selected_items) > 1:
            note = QLabel("当前多选扫描时，仅此策略会使用这里的参数；其他策略使用各自默认值。")
            note.setWordWrap(True)
            note.setStyleSheet("color: #aaaaaa; font-size: 11px;")
            self.strategy_config_layout.addWidget(note, 0, 0, 1, 4)
            row_offset = 1

        core_items, advanced_items = self._split_strategy_schema(strategy_info.name, schema)
        row_offset = self._add_schema_section("核心参数", core_items, row_offset)
        if advanced_items:
            toggle_btn = QPushButton("展开高级参数" if not self.show_advanced_strategy_params else "收起高级参数")
            toggle_btn.setCheckable(True)
            toggle_btn.setChecked(self.show_advanced_strategy_params)
            toggle_btn.setStyleSheet("padding: 4px 10px;")
            toggle_btn.clicked.connect(self.toggle_advanced_strategy_params)
            self.strategy_config_layout.addWidget(toggle_btn, row_offset, 0, 1, 4)
            row_offset += 1
            if self.show_advanced_strategy_params:
                row_offset = self._add_schema_section("高级参数", advanced_items, row_offset)

    def toggle_advanced_strategy_params(self):
        self.show_advanced_strategy_params = not self.show_advanced_strategy_params
        self.refresh_strategy_config_panel()

    def _split_strategy_schema(self, strategy_name, schema):
        """按策略名称拆分核心/高级参数。"""
        items = list(schema.items())
        if strategy_name != "AI截面双因子组合扫描器":
            return items, []

        core_keys = [
            "mode",
            "min_volume_24h",
            "min_score",
            "top_n",
            "dedupe_by_symbol",
            "min_consensus_engines",
            "allow_short",
            "backtest_min_score",
            "top_n_per_strategy",
        ]
        core_set = set(core_keys)
        core_items = [(k, v) for k, v in items if k in core_set]
        advanced_items = [(k, v) for k, v in items if k not in core_set]
        return core_items, advanced_items

    def _add_schema_section(self, title, schema_items, row_offset):
        """将一组 schema 参数以双列形式加入面板。"""
        if not schema_items:
            return row_offset

        title_label = QLabel(title)
        title_label.setStyleSheet("color: #7fd1ff; font-size: 12px; font-weight: bold;")
        self.strategy_config_layout.addWidget(title_label, row_offset, 0, 1, 4)
        row_offset += 1

        for index, (param_name, param_info) in enumerate(schema_items):
            label = param_info.get('label', param_name)
            param_type = param_info.get('type', 'float')
            default_value = param_info.get('default', 0)

            if param_type == 'select':
                input_widget = QComboBox()
                options = param_info.get('options', [])
                for option in options:
                    if isinstance(option, dict):
                        input_widget.addItem(str(option.get('label', option.get('value', ''))), option.get('value'))
                    else:
                        input_widget.addItem(str(option), option)
                default_index = input_widget.findData(default_value)
                if default_index < 0:
                    default_index = input_widget.findText(str(default_value))
                if default_index >= 0:
                    input_widget.setCurrentIndex(default_index)
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

            label_widget = QLabel(f"{label}:")
            label_widget.setStyleSheet("color: #cccccc;")
            input_widget.setMinimumWidth(120)
            grid_row = row_offset + index // 2
            col_offset = (index % 2) * 2
            self.strategy_config_layout.addWidget(label_widget, grid_row, col_offset)
            self.strategy_config_layout.addWidget(input_widget, grid_row, col_offset + 1)
            self.strategy_config_inputs[param_name] = input_widget

        return row_offset + (len(schema_items) + 1) // 2 + 1

    def get_selected_strategy_config(self, strategy_name: str) -> dict:
        """返回当前参数面板对应策略的配置。"""
        if strategy_name != self.current_config_strategy_name:
            return {}
        config = {}
        for param_name, input_widget in self.strategy_config_inputs.items():
            if isinstance(input_widget, QCheckBox):
                config[param_name] = input_widget.isChecked()
            elif isinstance(input_widget, QComboBox):
                config[param_name] = input_widget.currentData()
            elif isinstance(input_widget, QSpinBox):
                config[param_name] = input_widget.value()
            elif isinstance(input_widget, QDoubleSpinBox):
                config[param_name] = input_widget.value()
        return config

    def toggle_auto_scan(self):
        """切换定时扫描"""
        try:
            print(f"[toggle_auto_scan] 当前状态：auto_scan_enabled={self.auto_scan_enabled}")
            
            if self.auto_scan_enabled:
                # 停止定时扫描
                print("[toggle_auto_scan] 停止定时扫描")
                self.auto_scan_enabled = False
                if hasattr(self, 'countdown_timer'):
                    self.countdown_timer.stop()
                self.auto_scan_btn.setText("⏱ 启动定时扫描")
                self.auto_scan_btn.setStyleSheet("background-color: #28a745; color: white; font-weight: bold; padding: 5px 15px;")
                self.countdown_label.setText("⏱ 下次扫描：已停止")
                self.status_label.setText("定时扫描已停止")
            else:
                # 启动定时扫描
                print("[toggle_auto_scan] 尝试启动定时扫描")
                
                # 检查必要组件是否存在
                if not hasattr(self, 'strategy_list'):
                    raise AttributeError("策略列表未初始化")
                if not hasattr(self, 'interval_spin'):
                    raise AttributeError("间隔设置未初始化")
                if not hasattr(self, 'countdown_timer'):
                    raise AttributeError("倒计时定时器未初始化")
                
                # 只使用当前高亮（currentItem）的策略，忽略多选
                current_item = self.strategy_list.currentItem()
                if not current_item:
                    print("[toggle_auto_scan] 警告：没有高亮任何策略")
                    QMessageBox.warning(
                        self,
                        "⚠️ 需要选择策略",
                        "请先在上方策略列表中单击选中（高亮）一个策略！\n\n"
                        "📋 操作步骤：\n"
                        "1️⃣ 单击策略名称（高亮显示）\n"
                        "2️⃣ 设置扫描间隔时间（默认5分钟）\n"
                        "3️⃣ 再点击「⏱ 启动定时扫描」\n\n"
                        "💡 定时扫描只运行当前高亮的策略，切换高亮即可更换策略。"
                    )
                    return

                strategy_info = current_item.data(Qt.ItemDataRole.UserRole)
                print(f"[toggle_auto_scan] 高亮策略：{strategy_info.name if strategy_info else '未知'}")

                self.auto_scan_enabled = True
                self.countdown_remaining = self.interval_spin.value() * 60  # 转换为秒
                print(f"[toggle_auto_scan] 间隔时间：{self.interval_spin.value()} 分钟 ({self.countdown_remaining} 秒)")

                self.auto_scan_btn.setText("⏹ 停止定时扫描")
                self.auto_scan_btn.setStyleSheet("background-color: #dc3545; color: white; font-weight: bold; padding: 5px 15px;")
                self.countdown_timer.start(1000)  # 每秒更新一次
                self.update_countdown()
                sname = strategy_info.name if strategy_info else "未知"
                self.status_label.setText(
                    f"✅ 定时扫描已启动  策略: {sname}  间隔: {self.interval_spin.value()} 分钟"
                )
                print("[toggle_auto_scan] 定时扫描已成功启动")
        except Exception as e:
            error_msg = f"自动扫描启动失败：{str(e)}"
            print(f"[错误] {error_msg}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "❌ 启动失败", f"{error_msg}\n\n请查看终端日志获取详细信息。")

    def update_countdown(self):
        """更新倒计时显示"""
        if self.is_scanning:
            self.countdown_label.setText("⏱ 正在扫描...")
            return

        if self.countdown_remaining > 0:
            self.countdown_remaining -= 1
            minutes = self.countdown_remaining // 60
            seconds = self.countdown_remaining % 60
            self.countdown_label.setText(f"⏱ 下次扫描：{minutes:02d}:{seconds:02d}")
            
            # 倒计时结束，自动扫描
            if self.countdown_remaining == 0:
                self.auto_scan_all()
        else:
            self.countdown_label.setText("⏱ 准备扫描...")

    def auto_scan_all(self):
        """定时自动扫描：只运行当前高亮显示的策略（单一策略模式）"""
        current_item = self.strategy_list.currentItem()
        if not current_item:
            self.status_label.setText("⚠️ 无高亮策略，跳过本轮扫描")
            # 仍需重置倒计时，否则定时器就停了
            self.countdown_remaining = self.interval_spin.value() * 60
            return

        strategy_info = current_item.data(Qt.ItemDataRole.UserRole)
        if not strategy_info:
            self.countdown_remaining = self.interval_spin.value() * 60
            return

        # 单策略模式：不做批量聚合
        self.current_auto_strategies = [strategy_info]
        self.batch_scan_active = False
        self.aggregate_scan_results = []
        self.raw_results = []
        self.latest_results = []
        self._pool_live_keys = set()

        self.start_next_auto_strategy()

    def start_next_auto_strategy(self):
        """开始执行队列中的下一个策略"""
        if not self.current_auto_strategies:
            # 所有策略扫描完成，重置倒计时
            self.countdown_remaining = self.interval_spin.value() * 60
            self.status_label.setText(f"所有策略扫描完成，下次扫描将在 {self.interval_spin.value()} 分钟后")
            return

        strategy_info = self.current_auto_strategies.pop(0)
        self.status_label.setText(f"正在执行自动扫描: {strategy_info.name}")
        self.start_scan(strategy_info)

    def start_scan(self, strategy_info=None):
        """开始扫描（支持手动传入策略或从列表获取）"""
        try:
            print(f"[start_scan] 被调用，strategy_info={strategy_info}")
            
            # 如果没有传入策略，从多选列表获取选中的策略
            if strategy_info is None:
                # 手动点击“手动扫描”按钮时的逻辑
                selected_items = self.strategy_list.selectedItems()
                if not selected_items and self.strategy_list.currentItem() is not None:
                    self.strategy_list.currentItem().setSelected(True)
                    selected_items = self.strategy_list.selectedItems()
                print(f"[start_scan] 选中项数量: {len(selected_items)}")
                
                if not selected_items:
                    self.status_label.setText("错误：请至少选择一个扫描策略")
                    QMessageBox.warning(self, "警告", "请先在策略列表中勾选至少一个扫描策略！")
                    return

                # 手动扫描时，也支持顺序扫描所有选中的
                self.current_auto_strategies = [item.data(Qt.ItemDataRole.UserRole) for item in selected_items]
                self.batch_scan_active = len(self.current_auto_strategies) > 1
                self.aggregate_scan_results = []
                self.raw_results = []
                self.latest_results = []
                self._pool_live_keys = set()
                strategy_info = self.current_auto_strategies.pop(0)
            
            if not strategy_info:
                self.status_label.setText("错误：无效的策略信息")
                self.on_scan_finished([])
                return

            # 保存当前策略名称
            self.current_strategy_name = strategy_info.name

            strategy_config = self.get_selected_strategy_config(strategy_info.name)
            strategy_instance = strategy_info.create_instance(strategy_config)
            if not strategy_instance:
                self.status_label.setText("错误：无法实例化策略")
                QMessageBox.critical(self, "启动失败", f"策略 {strategy_info.name} 实例化失败，请检查策略参数或策略文件。")
                return

            print(f"\n[扫描] 开始执行扫描: {strategy_info.name}")
            print(f"[扫描] 策略配置: {strategy_info.config_schema}")
            if strategy_config:
                print(f"[扫描] 当前生效参数: {strategy_config}")

            strategy_runtime_config = dict(getattr(strategy_instance, "config", {}) or strategy_config or {})
            self._current_strategy_config = dict(strategy_runtime_config)
            strategy_code_hash = ""
            try:
                with open(strategy_info.path, "r", encoding="utf-8") as f:
                    strategy_code_hash = hashlib.md5(f.read().encode("utf-8")).hexdigest()[:8]
            except Exception:
                strategy_code_hash = ""
            strategy_instance._rl_strategy_name = strategy_info.name
            strategy_instance._rl_strategy_path = strategy_info.path
            strategy_instance._rl_strategy_file = os.path.basename(strategy_info.path)
            strategy_instance._rl_param_snapshot = strategy_runtime_config
            strategy_instance._rl_code_hash = strategy_code_hash

            # 关键修复：启动新线程前，安全清理旧线程
            old_thread = getattr(self, '_scan_thread', None)
            if old_thread is not None:
                if old_thread.isRunning():
                    print("[start_scan] 警告：旧线程仍在运行，发送停止信号...")
                    self.engine.request_stop()
                    old_thread.stop()
                # 安全丢弃：若 200ms 内未退出则暂存僵尸列表，防止 GC 在线程运行时销毁对象
                self._safe_discard_scan_thread(old_thread, wait_ms=200)
                self._scan_thread = None

            # 开始扫描前重置 UI
            self.is_scanning = True
            self.is_paused = False
            self.scan_btn.setEnabled(False)
            self.pause_btn.setEnabled(True)
            self.stop_btn.setEnabled(True)
            self.auto_scan_btn.setEnabled(False)
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(10)
            if not (self.batch_scan_active and self.aggregate_scan_results):
                self._clear_all_result_tables()
                self.raw_results = []
                self.latest_results = []
            self.status_label.setText(f"正在扫描: {strategy_info.name}...")
            self.current_symbol_label.setText("准备开始扫描...")
            self.current_strategy_label.setText(f"📊 策略: {strategy_info.name}")
            self.scan_stats_label.setText("")
            self.scan_progress_label.setText("初始化扫描引擎...")
            self.live_signal_label.setText("🎯 最新命中: 扫描中，等待信号...")
            if hasattr(self, 'scan_log_browser'):
                self.scan_log_browser.clear()
            self._log_progress_counter = 0
            self.scan_log_signal.emit(
                f"▶ 开始扫描  策略: {strategy_info.name}  扫描全部 USDT 永续合约", "INFO"
            )
            self.scan_log_signal.emit(
                "[5%|0|0] 正在从 OKX 获取全市场行情，请稍候…", "INFO"
            )

            # 刷新一次余额
            self.refresh_balance()

            # 保持线程引用防止被垃圾回收
            self._scan_thread = ScanThread(self.engine, strategy_instance)
            self._scan_thread.progress.connect(self.update_progress)
            self._scan_thread.result_found.connect(self.add_realtime_result_batch)
            self._scan_thread.finished.connect(self.on_scan_finished)
            self._scan_thread.error.connect(self.on_scan_error)
            # 注意：已移除暂停/恢复功能，简化线程实现

            print(f"[扫描] 启动扫描线程...")
            self._scan_thread.start()
        except Exception as e:
            print(f"[错误] 启动扫描失败: {e}")
            import traceback
            traceback.print_exc()
            self.on_scan_error(f"启动失败: {str(e)}")

    def stop_scan(self):
        """停止扫描"""
        # 如果当前没有在扫描，但可能在倒计时
        if not self.is_scanning:
            if self.auto_scan_enabled:
                self.toggle_auto_scan()
            return

        reply = QMessageBox.question(
            self, '确认停止', 
            '确定要停止当前扫描吗？',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            # 清空队列，防止继续扫描下一个
            self.current_auto_strategies = []

            old_thread = getattr(self, '_scan_thread', None)
            if old_thread is not None and old_thread.isRunning():
                self.engine.request_stop()
                old_thread.stop()
                self.status_label.setText("正在停止扫描线程...")
            if old_thread is not None:
                # 安全丢弃：若 200ms 内未退出则暂存僵尸列表
                self._safe_discard_scan_thread(old_thread, wait_ms=200)
                self._scan_thread = None

            # 立即更新 UI 状态
            self.is_scanning = False
            self.scan_btn.setEnabled(True)
            self.pause_btn.setEnabled(False)
            self.stop_btn.setEnabled(False)
            self.auto_scan_btn.setEnabled(True)
            self.progress_bar.setVisible(False)
            self.status_label.setText("扫描已停止")
            self.scan_progress_label.setText("扫描已由用户手动停止")
            self.current_symbol_label.setText("等待扫描...")
            self.current_strategy_label.setText("策略: -")

    def add_realtime_result_batch(self, batch: list):
        """批量处理扫描结果，避免UI卡顿"""
        for res in batch:
            self._process_single_result(res, refresh=False)
        self.latest_results = list(self.raw_results)
        # 防抖：300ms 内多次调用只刷新一次，避免每批都重建全量表格
        self._refresh_debounce_timer.start(300)

    def add_realtime_result(self, res):
        """实时将发现的信号插入表格（单条，保留兼容性）"""
        self._process_single_result(res)

    def _append_scan_log_line(self, message: str, level: str = "INFO"):
        if not hasattr(self, "scan_log_browser") or self.scan_log_browser is None:
            return
        timestamp = datetime.now().strftime("%H:%M:%S")
        color = {
            "INFO": "#9ecbff",
            "WARNING": "#ffd27f",
            "ERROR": "#ff8f8f",
            "SUCCESS": "#89f0a5",
            "TRADE": "#7fdfff",
        }.get(level, "#d7dde5")
        self.scan_log_browser.append(
            f'<span style="color:#6f7a86">[{timestamp}]</span> '
            f'<span style="color:{color}">[{level}]</span> '
            f'{message}'
        )
        self.scan_log_browser.verticalScrollBar().setValue(
            self.scan_log_browser.verticalScrollBar().maximum()
        )

    def _store_rejected_result(self, res: dict, reason: str):
        """保存被过滤的信号到日志文件，供事后审查"""
        entry = {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "symbol": res.get("symbol", ""),
            "direction": res.get("direction", res.get("side", "")),
            "score": float(res.get("score", res.get("opportunity_score", 0)) or 0),
            "category": res.get("category", res.get("strategy_category", "")),
            "reason": reason,
            "last_price": float(res.get("last_price", 0) or 0),
        }
        # 内存保留
        if not hasattr(self, "_rejected_signals"):
            self._rejected_signals = []
        self._rejected_signals.append(entry)
        if len(self._rejected_signals) > 500:
            self._rejected_signals = self._rejected_signals[-500:]
        # 每 20 条批量写入磁盘
        if len(self._rejected_signals) % 20 == 0:
            self._save_rejected_signals()

    def _save_rejected_signals(self):
        try:
            import json
            path = Path(__file__).resolve().parent.parent / "rejected_signals_log.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self._rejected_signals[-500:], f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _load_rejected_signals(self):
        try:
            import json
            path = Path(__file__).resolve().parent.parent / "rejected_signals_log.json"
            if path.exists():
                with open(path, encoding="utf-8") as f:
                    self._rejected_signals = json.load(f)[-500:]
            else:
                self._rejected_signals = []
        except Exception:
            self._rejected_signals = []

    def _show_rejected_signals(self):
        """弹窗显示被过滤的信号列表"""
        signals = getattr(self, "_rejected_signals", [])
        if not signals:
            QMessageBox.information(self, "被过滤信号", "暂无被过滤的信号记录")
            return
        recent = signals[-50:]
        lines = [f"{s['time']} | {s['symbol']} | {s.get('direction','')} | 评分{s.get('score',0):.0f} | {s.get('reason','')}"
                 for s in reversed(recent)]
        QMessageBox.information(self, "被过滤信号", f"最近 {len(recent)} 条:\n\n" + "\n".join(lines))

    def _entry_rule_filter_result(self, res):
        direction = normalize_direction(res.get('direction', res.get('side', '')))
        if not direction:
            return False, "结果缺少明确方向"
        klines_map = dict(res.get("klines_map", {}) or {})
        if not klines_map:
            return False, "缺少3m/H1 K线，上线前硬性检查无法执行"
        guard = evaluate_entry_rule_from_klines(
            klines_map,
            direction,
            getattr(self, "_current_strategy_config", {}) or {},
        )
        return bool(guard.get("ok")), str(guard.get("reason", "未通过3m/H1硬性检查"))

    def _process_single_result(self, res, refresh=True):
        """处理单条扫描结果"""
        passed, reason = self._entry_rule_filter_result(res)
        if not passed:
            symbol = res.get('symbol', '')
            self.scan_log_signal.emit(
                f"🛑 过滤原因 | {symbol} | {reason}",
                "WARNING",
            )
            self._store_rejected_result(res, reason)
            return
        enrich_scan_result(res)
        res['updated_at'] = datetime.now().isoformat()
        symbol = res.get('symbol')
        category = res.get('category')
        strategy_name = res.get('strategy_name')
        direction = str(res.get('side', res.get('direction', 'WATCH')))
        pool_key = (symbol, direction, strategy_name or self.current_strategy_name)
        if symbol:
            self.raw_results = [
                item for item in self.raw_results
                if (item.get('symbol'), item.get('category'), item.get('strategy_name')) != (symbol, category, strategy_name)
            ]
            self.raw_results.append(res)
            score_val = float(res.get('opportunity_score', res.get('score', 0)) or 0)
            self.live_signal_label.setText(
                f"🎯 最新命中: {symbol} | {direction} | 评分 {score_val:.1f}"
            )
            category = res.get('category', '')
            reason_raw = res.get('signals', res.get('reason', ''))
            reason_txt = (', '.join(reason_raw[:2]) if isinstance(reason_raw, list)
                          else str(reason_raw)[:60])
            self.scan_log_signal.emit(
                f"🎯 命中 {symbol}  {direction}  评分:{score_val:.1f}"
                + (f"  [{category}]" if category else "")
                + (f"  {reason_txt}" if reason_txt else ""),
                "TRADE",
            )
            if self.trade_pool_page:
                self.trade_pool_page.add_results([res], strategy_name or self.current_strategy_name, increment_hits=False)
                self._pool_live_keys.add(pool_key)
            if self.rl_learning_page:
                signal = dict(res)
                signal.setdefault("strategy_name", strategy_name or self.current_strategy_name)
                klines_map = signal.get("klines_map", {})
                if klines_map:
                    self.rl_learning_page.record_signal_with_trends(signal, klines_map)
                else:
                    self.rl_learning_page.record_signal(signal)
        if refresh:
            self.latest_results = list(self.raw_results)
            self.refresh_result_view()

    def fill_table_row(self, table, i, res):
        """填充表格行的具体数据"""
        resonance_count = int(res.get('resonance_count', 1) or 1)
        is_resonance = bool(res.get('is_resonance', False) or resonance_count > 1)
        is_new_signal = bool(res.get('is_new_signal', False))

        # 交易对
        symbol_text = res.get('symbol', '-')
        if is_resonance:
            symbol_text = f"★ {symbol_text}"
        if is_new_signal:
            symbol_text = f"NEW {symbol_text}"
        symbol_item = QTableWidgetItem(symbol_text)
        table.setItem(i, 0, symbol_item)
        
        # 价格
        price_item = QTableWidgetItem(f"{res.get('last_price', 0):.4f}")
        price_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        table.setItem(i, 1, price_item)
        
        # 24h涨幅
        change = res.get('change_24h', res.get('price_change_24h', 0))
        change_item = QTableWidgetItem(f"{change:+.2f}%")
        change_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        if change > 0: change_item.setForeground(Qt.GlobalColor.green)
        elif change < 0: change_item.setForeground(Qt.GlobalColor.red)
        table.setItem(i, 2, change_item)

        # 机会类型
        category = res.get('category', '-')
        if is_resonance:
            category = f"{category} · 共振x{max(resonance_count, int(res.get('resonance_strategy_count', 1) or 1))}"
        category_item = QTableWidgetItem(str(category))
        category_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        table.setItem(i, 3, category_item)
        
        # 信号强度 (Score)
        opportunity_score = float(res.get('opportunity_score', res.get('score', 0)) or 0)
        opportunity_level = res.get('opportunity_level', '')
        score_text = f"{opportunity_score:.1f}"
        if opportunity_level:
            score_text = f"{score_text} ({opportunity_level})"
        if is_resonance:
            score_text = f"{score_text} ★"
        if is_new_signal:
            score_text = f"{score_text} NEW"
        score_item = QTableWidgetItem(score_text)
        score_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        if opportunity_score >= 84:
            score_item.setForeground(Qt.GlobalColor.cyan)
        table.setItem(i, 4, score_item)

        # 连续出现轮数
        streak_count = int(res.get('streak_count', 1) or 1)
        streak_text = f"{streak_count}"
        if is_new_signal:
            streak_text = f"{streak_text} (新)"
        streak_item = QTableWidgetItem(streak_text)
        streak_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        if streak_count >= 3:
            streak_item.setForeground(Qt.GlobalColor.yellow)
        table.setItem(i, 5, streak_item)
        
        # 建议方向
        side = res.get('side', res.get('direction', 'WATCH'))
        side_item = QTableWidgetItem(side)
        side_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        if side == 'BUY' or side == 'LONG': side_item.setBackground(Qt.GlobalColor.darkGreen)
        elif side == 'SELL' or side == 'SHORT': side_item.setBackground(Qt.GlobalColor.darkRed)
        table.setItem(i, 6, side_item)
        
        # 理由/详情
        reason = res.get('priority_reason', res.get('reason', res.get('details', '-')))
        if isinstance(reason, list): reason = ", ".join(reason)
        if is_resonance:
            strategy_name = str(res.get('strategy_name', '')).strip()
            strategy_desc = f"{strategy_name} | " if strategy_name else ""
            reason = f"[多策略共振] {strategy_desc}{reason}"
        if is_new_signal:
            reason = f"[最近新出现] {reason}"
        table.setItem(i, 7, QTableWidgetItem(str(reason)[:100]))
        
        # 时间
        time_str = datetime.now().strftime("%H:%M:%S")
        table.setItem(i, 8, QTableWidgetItem(time_str))

        if is_resonance:
            resonance_bg = Qt.GlobalColor.darkYellow
            for col in [0, 3, 4]:
                cell = table.item(i, col)
                if cell is not None:
                    cell.setBackground(resonance_bg)
        if is_new_signal:
            new_fg = Qt.GlobalColor.yellow
            for col in [0, 4, 5, 7]:
                cell = table.item(i, col)
                if cell is not None:
                    cell.setForeground(new_fg)

    def on_scan_finished(self, results):
        """扫描完成回调"""
        self.is_scanning = False
        self.is_paused = False
        self.scan_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.auto_scan_btn.setEnabled(True)
        self.pause_btn.setText("⏸ 暂停扫描")
        self.pause_btn.setStyleSheet("background-color: #ffc107; color: black; font-weight: bold; padding: 5px 15px;")
        self.progress_bar.setVisible(False)

        # 安全清理线程引用（不阻塞 UI）
        old_thread = getattr(self, '_scan_thread', None)
        if old_thread is not None:
            # on_scan_finished 是由 finished 信号触发，线程即将退出，200ms 应足够；
            # 若仍未退出则暂存僵尸列表保持引用，防止 GC 在运行时销毁 QThread。
            self._safe_discard_scan_thread(old_thread, wait_ms=200)
            self._scan_thread = None

        # 最终校准一次结果（按分数排序）
        finalized_results = []
        filtered_out = 0
        filtered_details = []
        for item in (results or []):
            item = dict(item)
            item.setdefault('updated_at', datetime.now().isoformat())
            item.setdefault('strategy_name', self.current_strategy_name)
            passed, reason = self._entry_rule_filter_result(item)
            if not passed:
                filtered_out += 1
                filtered_details.append((item.get('symbol', ''), reason))
                self.scan_log_signal.emit(
                    f"🛑 过滤原因 | {item.get('symbol', '')} | {reason}",
                    "WARNING",
                )
                self._store_rejected_result(item, reason)
                continue
            finalized_results.append(item)
        if self.batch_scan_active:
            self.aggregate_scan_results = self._merge_scan_results(self.aggregate_scan_results, finalized_results)
            self.raw_results = sort_scan_results(self.aggregate_scan_results)
        else:
            self.raw_results = sort_scan_results(finalized_results)
        import gc
        gc.collect()
        self.latest_results = list(self.raw_results)
        self.refresh_result_view()

        # 广播扫描结果供外部订阅（智能助理等）
        if self.raw_results:
            self.scan_results_ready.emit(list(self.raw_results))

        # 将结果添加到交易对池
        if self.trade_pool_page and finalized_results:
            pending_results = []
            for res in finalized_results:
                direction = str(res.get('direction', res.get('side', 'WATCH')))
                pool_key = (res.get('symbol'), direction, self.current_strategy_name)
                if pool_key not in self._pool_live_keys:
                    pending_results.append(res)
            if pending_results:
                self.trade_pool_page.add_results(pending_results, self.current_strategy_name)

        # 更新状态显示
        finish_msg = (
            f"✅ 策略扫描完成: {datetime.now().strftime('%H:%M:%S')}，"
            f"输出 {len(finalized_results)} 个信号"
            + (f"，过滤 {filtered_out} 个" if filtered_out else "")
        )
        self.scan_progress_label.setText(finish_msg)
        self.current_symbol_label.setText("✅ 当前策略扫描已完成")
        self.current_strategy_label.setText("策略: -")
        self.scan_stats_label.setText("")
        self.status_label.setText("就绪")
        if finalized_results:
            self.live_signal_label.setText(f"🎯 最新命中: 本轮完成，共 {len(finalized_results)} 个信号")
        else:
            self.live_signal_label.setText("🎯 最新命中: 本轮无信号")

        # 广播完成摘要
        if filtered_details:
            self.scan_log_signal.emit(
                f"📋 本轮过滤明细：共 {len(filtered_details)} 个交易对被过滤",
                "WARNING",
            )
            for symbol, reason in filtered_details[:20]:
                self.scan_log_signal.emit(
                    f"   - {symbol} | {reason}",
                    "WARNING",
                )
            if len(filtered_details) > 20:
                self.scan_log_signal.emit(
                    f"   - 其余 {len(filtered_details) - 20} 个过滤结果未展开显示",
                    "WARNING",
                )

        top_hits = []
        for r in finalized_results[:3]:
            sym = r.get('symbol', '')
            dir_ = r.get('direction', '')
            sc = float(r.get('score', r.get('opportunity_score', 0)) or 0)
            if sym:
                top_hits.append(f"{sym}/{dir_}({sc:.0f})")
        top_str = "  ".join(top_hits)
        level = "SUCCESS" if results else "INFO"
        self.scan_log_signal.emit(
            f"■ 扫描完成  共 {len(results)} 个信号"
            + (f"  前3: {top_str}" if top_hits else "  无信号"),
            level,
        )

        # 检查是否还有待执行的队列
        if self.current_auto_strategies:
            print(f"[on_scan_finished] 队列中还有 {len(self.current_auto_strategies)} 个策略待执行，2秒后开始下一个...")
            QTimer.singleShot(2000, self.start_next_auto_strategy)
        else:
            # 队列全部执行完毕
            self.batch_scan_active = False
            self.previous_scan_keys = {self._result_identity_key(item) for item in self.raw_results}
            self.previous_scan_streaks = {
                self._result_identity_key(item): int(item.get('streak_count', 1) or 1)
                for item in self.raw_results
            }
            if self.auto_scan_enabled:
                print(f"[on_scan_finished] 全部策略扫描完毕，开始计算下次扫描间隔 ({self.interval_spin.value()} 分钟)")
                # 只有在这里，当一轮扫描全部完成后，才重置倒计时
                self.countdown_remaining = self.interval_spin.value() * 60
                self.status_label.setText(f"所有扫描已完成，下次扫描将在 {self.interval_spin.value()} 分钟后")

    def display_results(self, results):
        """重新填充整个表格"""
        results = list(results or [])
        self._clear_all_result_tables()
        deduped = []
        seen = set()
        for res in results:
            key = (res.get('symbol'), res.get('category'), res.get('strategy_name'))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(res)
        grouped = {category: [] for category in self.CATEGORY_TABS}
        for res in deduped:
            grouped["总览"].append(res)
            category_name = res.get('category')
            if category_name in grouped:
                grouped[category_name].append(res)
        for category_name, items in grouped.items():
            self._update_result_stats(category_name, items)
            table = self.result_tables.get(category_name)
            if table is None:
                continue
            # ── 批量更新：预分配行数 + 禁用中间重绘，避免每行触发布局重计算 ──
            table.setUpdatesEnabled(False)
            table.setSortingEnabled(False)
            try:
                table.setRowCount(len(items))   # 预分配，替代逐行 insertRow
                for i, res in enumerate(items):
                    self.fill_table_row(table, i, res)
            finally:
                table.setSortingEnabled(True)
                table.setUpdatesEnabled(True)   # 一次性刷新整张表

    def refresh_result_view(self):
        annotated_results = self._annotate_resonance(self.raw_results)
        annotated_results = self._annotate_newness(annotated_results)
        filtered_results = self._apply_result_filters(annotated_results)
        sorted_results = self._sort_filtered_results(filtered_results)
        self.latest_results = sorted_results
        self.display_results(sorted_results)

    def reset_result_filters(self):
        # blockSignals 防止 8 个控件逐一触发 refresh_result_view，最后统一刷新一次
        _widgets = [
            self.direction_filter_combo, self.level_filter_combo,
            self.category_filter_combo, self.min_score_filter_spin,
            self.resonance_only_check, self.new_only_check,
            self.quadrant_filter_combo, self.sort_combo,
        ]
        for w in _widgets:
            w.blockSignals(True)
        try:
            self.direction_filter_combo.setCurrentIndex(0)
            self.level_filter_combo.setCurrentIndex(0)
            self.category_filter_combo.setCurrentIndex(0)
            self.min_score_filter_spin.setValue(0.0)
            self.resonance_only_check.setChecked(False)
            self.new_only_check.setChecked(False)
            self.quadrant_filter_combo.setCurrentIndex(0)
            self.sort_combo.setCurrentIndex(0)
        finally:
            for w in _widgets:
                w.blockSignals(False)
        self.refresh_result_view()  # 只触发一次

    def _apply_result_filters(self, results):
        filtered = []
        direction_filter = self.direction_filter_combo.currentText()
        level_filter = self.level_filter_combo.currentText()
        category_filter = self.category_filter_combo.currentText()
        min_score = float(self.min_score_filter_spin.value())
        resonance_only = self.resonance_only_check.isChecked()
        new_only = self.new_only_check.isChecked()
        quadrant_filter = self.quadrant_filter_combo.currentText()

        for item in results or []:
            direction = str(item.get('direction') or item.get('side') or '').upper()
            if direction_filter != "全部方向" and direction != direction_filter:
                continue

            level = str(item.get('opportunity_level', '')).upper()
            if level_filter != "全部等级" and level != level_filter:
                continue

            category = str(item.get('category', ''))
            if category_filter != "全部类型" and category != category_filter:
                continue

            opportunity_score = float(item.get('opportunity_score', item.get('score', 0)) or 0)
            if opportunity_score < min_score:
                continue

            if resonance_only and not bool(item.get('is_resonance', False)):
                continue

            if new_only and not bool(item.get('is_new_signal', False)):
                continue

            if quadrant_filter != "全部象限" and self._get_quadrant_name(item) != quadrant_filter:
                continue

            filtered.append(item)
        return filtered

    def _sort_filtered_results(self, results):
        sort_mode = self.sort_combo.currentText()
        if sort_mode == "按连续轮数":
            return sorted(
                results,
                key=lambda item: (
                    int(item.get('streak_count', 1)),
                    int(item.get('resonance_count', 1)),
                    float(item.get('opportunity_score', item.get('score', 0)) or 0),
                ),
                reverse=True,
            )
        if sort_mode == "按原始分数":
            return sorted(
                results,
                key=lambda item: (int(item.get('resonance_count', 1)), float(item.get('score', 0) or 0)),
                reverse=True,
            )
        if sort_mode == "按24H成交额":
            return sorted(
                results,
                key=lambda item: (int(item.get('resonance_count', 1)), float(item.get('volume_24h', 0) or 0)),
                reverse=True,
            )
        if sort_mode == "按24H涨跌幅":
            return sorted(
                results,
                key=lambda item: (
                    int(item.get('resonance_count', 1)),
                    abs(float(item.get('price_change_24h', item.get('change_24h', 0)) or 0)),
                ),
                reverse=True,
            )
        if sort_mode == "按更新时间":
            return sorted(
                results,
                key=lambda item: (
                    int(item.get('resonance_count', 1)),
                    str(item.get('updated_at', item.get('timestamp', '')))
                ),
                reverse=True,
            )
        return sorted(
            sort_scan_results(results or []),
            key=lambda item: (
                int(item.get('resonance_count', 1)),
                float(item.get('opportunity_score', item.get('score', 0)) or 0),
                float(item.get('score', 0) or 0),
            ),
            reverse=True,
        )

    def _annotate_resonance(self, results):
        annotated = [dict(item) for item in (results or [])]
        symbol_counts = {}
        symbol_strategies = {}
        for item in annotated:
            symbol = str(item.get('symbol', ''))
            if not symbol:
                continue
            symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1
            symbol_strategies.setdefault(symbol, set()).add(str(item.get('strategy_name', item.get('category', ''))))
        for item in annotated:
            symbol = str(item.get('symbol', ''))
            count = symbol_counts.get(symbol, 1)
            strategy_count = len(symbol_strategies.get(symbol, set()))
            item['resonance_count'] = count
            item['resonance_strategy_count'] = strategy_count
            item['is_resonance'] = count > 1 or strategy_count > 1
        return annotated

    def _annotate_newness(self, results):
        annotated = [dict(item) for item in (results or [])]
        has_previous_snapshot = bool(self.previous_scan_keys)
        for item in annotated:
            result_key = self._result_identity_key(item)
            item['is_new_signal'] = (result_key not in self.previous_scan_keys) if has_previous_snapshot else True
            item['streak_count'] = int(self.previous_scan_streaks.get(result_key, 0)) + 1
            item['quadrant'] = self._get_quadrant_name(item)
        return annotated

    def _merge_scan_results(self, existing_results, new_results):
        merged = {}
        for item in list(existing_results or []) + list(new_results or []):
            key = self._result_identity_key(item)
            merged[key] = item
        return list(merged.values())

    def _result_identity_key(self, item):
        return (
            item.get('symbol'),
            item.get('category'),
            item.get('strategy_name'),
        )

    def _get_quadrant_name(self, item):
        is_new_signal = bool(item.get('is_new_signal', False))
        is_resonance = bool(item.get('is_resonance', False))
        if is_new_signal and is_resonance:
            return "新出现且共振"
        if is_new_signal and not is_resonance:
            return "新出现但未共振"
        if (not is_new_signal) and is_resonance:
            return "连续强化且共振"
        return "连续强化但未共振"

    def _create_result_tab(self, category_name):
        tab_widget = QWidget()
        tab_layout = QVBoxLayout(tab_widget)
        tab_layout.setContentsMargins(0, 0, 0, 0)
        tab_layout.setSpacing(6)

        stats_label = QLabel(self._format_result_stats(category_name, []))
        stats_label.setTextFormat(Qt.TextFormat.RichText)
        stats_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
        stats_label.setOpenExternalLinks(False)
        stats_label.linkActivated.connect(self._handle_stats_link)
        stats_label.setStyleSheet("color: #8fd3ff; font-size: 12px; font-weight: bold; padding: 4px 2px;")
        tab_layout.addWidget(stats_label)

        table = self._create_result_table()
        table.cellClicked.connect(
            lambda row, column, current_table=table, current_category=category_name:
            self._on_result_cell_clicked(current_table, current_category, row, column)
        )
        table.cellDoubleClicked.connect(
            lambda row, column, current_table=table, current_category=category_name:
            self._on_result_cell_double_clicked(current_table, current_category, row, column)
        )
        table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        table.customContextMenuRequested.connect(
            lambda pos, current_table=table, current_category=category_name:
            self._on_result_context_menu(current_table, current_category, pos)
        )
        tab_layout.addWidget(table)
        return tab_widget, stats_label, table

    def _create_result_table(self):
        table = QTableWidget()
        table.setColumnCount(len(self.RESULT_HEADERS))
        table.setHorizontalHeaderLabels(self.RESULT_HEADERS)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        table.setAlternatingRowColors(True)
        table.setStyleSheet("QTableWidget { background-color: #252526; color: #cccccc; }")
        return table

    def _clear_all_result_tables(self):
        for table in self.result_tables.values():
            # 禁用重绘再清空，避免每行触发一次布局重计算
            table.setUpdatesEnabled(False)
            try:
                table.setRowCount(0)
            finally:
                table.setUpdatesEnabled(True)
        for category_name in self.CATEGORY_TABS:
            self._update_result_stats(category_name, [])

    def _update_result_stats(self, category_name, items):
        label = self.result_stats_labels.get(category_name)
        if label is None:
            return
        label.setText(self._format_result_stats(category_name, items))

    def _format_result_stats(self, category_name, items):
        count = len(items)
        if count == 0:
            return f"{category_name} | 数量: 0 | 最高评分: - | 平均评分: - | 最强交易对: -"

        scores = [float(item.get('opportunity_score', item.get('score', 0)) or 0) for item in items]
        best_item = max(
            items,
            key=lambda item: float(item.get('opportunity_score', item.get('score', 0)) or 0)
        )
        max_score = max(scores)
        avg_score = sum(scores) / count if count else 0.0
        best_symbol = str(best_item.get('symbol', '-'))
        escaped_symbol = escape(best_symbol)
        highlight_link = f"highlight||{category_name}||{best_symbol}"
        trade_link = f"trade||{category_name}||{best_symbol}"
        backtest_link = f"backtest||{category_name}||{best_symbol}"
        return (
            f"{category_name} | 数量: {count} | 最高评分: {max_score:.1f} | "
            f"平均评分: {avg_score:.1f} | 最强交易对: "
            f"<a href=\"{highlight_link}\" style=\"color:#ffd166; text-decoration:none;\">{escaped_symbol}</a> "
            f"[<a href=\"{trade_link}\" style=\"color:#7bd389; text-decoration:none;\">自动交易</a> / "
            f"<a href=\"{backtest_link}\" style=\"color:#7aa6ff; text-decoration:none;\">回测</a>]"
        )

    def _handle_stats_link(self, link: str):
        parts = str(link or "").split("||", 2)
        if len(parts) != 3:
            return
        action, category_name, symbol = parts
        if action == "highlight":
            self._highlight_symbol(symbol, category_name)
        elif action == "trade":
            self._send_symbol_to_trade(symbol, category_name)
        elif action == "backtest":
            self._send_symbol_to_backtest(symbol, category_name)

    def _highlight_symbol(self, symbol: str, category_name: str):
        target_tab = category_name if category_name in self.result_tables else "总览"
        table = self.result_tables.get(target_tab)
        if table is None:
            return
        self.result_tabs.setCurrentWidget(table.parentWidget())
        table.clearSelection()
        for row in range(table.rowCount()):
            item = table.item(row, 0)
            if item and item.text() == symbol:
                table.selectRow(row)
                table.scrollToItem(item, QTableWidget.ScrollHint.PositionAtCenter)
                return
        QMessageBox.information(self, "提示", f"未在 {target_tab} 中找到 {symbol}")

    def _send_symbol_to_trade(self, symbol: str, category_name: str):
        self._highlight_symbol(symbol, category_name)
        main_window = self.window()
        if not hasattr(main_window, 'main_tabs') or not hasattr(main_window, 'pair_combo'):
            QMessageBox.information(self, "提示", f"已高亮 {symbol}，但未找到自动交易页面入口")
            return
        main_window.pair_combo.setCurrentText(symbol)
        main_window.main_tabs.setCurrentIndex(0)
        QMessageBox.information(self, "已发送", f"{symbol} 已送入自动交易页面")

    def _send_symbol_to_backtest(self, symbol: str, category_name: str):
        self._highlight_symbol(symbol, category_name)
        main_window = self.window()
        backtest_page = getattr(main_window, 'backtest_page', None)
        main_tabs = getattr(main_window, 'main_tabs', None)
        if backtest_page is None or main_tabs is None or not hasattr(backtest_page, 'config_widget'):
            QMessageBox.information(self, "提示", f"已高亮 {symbol}，但未找到回测页面入口")
            return
        backtest_page.config_widget.pair_combo.setCurrentText(symbol)
        main_tabs.setCurrentWidget(backtest_page)
        QMessageBox.information(self, "已发送", f"{symbol} 已送入回测页面")

    def _on_result_cell_clicked(self, table, category_name: str, row: int, column: int):
        if column != 0 or row < 0:
            return
        context = self._get_row_symbol_context(table, category_name, row)
        if context is None:
            return
        _, symbol, actual_category = context
        self._show_symbol_quick_menu(table, row, symbol, actual_category)

    def _show_symbol_quick_menu(self, table, row: int, symbol: str, category_name: str):
        menu = QMenu(self)
        highlight_action = menu.addAction(f"高亮 {symbol}")
        trade_action = menu.addAction(f"送入自动交易: {symbol}")
        backtest_action = menu.addAction(f"送入回测: {symbol}")

        anchor_item = table.item(row, 0)
        if anchor_item is not None:
            popup_pos = table.viewport().mapToGlobal(table.visualItemRect(anchor_item).center())
        else:
            popup_pos = table.mapToGlobal(table.rect().center())

        selected_action = menu.exec(popup_pos)
        if selected_action == highlight_action:
            self._highlight_symbol(symbol, category_name)
        elif selected_action == trade_action:
            self._send_symbol_to_trade(symbol, category_name)
        elif selected_action == backtest_action:
            self._send_symbol_to_backtest(symbol, category_name)

    def _on_result_cell_double_clicked(self, table, category_name: str, row: int, column: int):
        context = self._get_row_symbol_context(table, category_name, row)
        if context is None:
            return
        _, symbol, actual_category = context
        self._send_symbol_to_backtest(symbol, actual_category)

    def _on_result_context_menu(self, table, category_name: str, pos):
        item = table.itemAt(pos)
        if item is None:
            return
        row = item.row()
        context = self._get_row_symbol_context(table, category_name, row)
        if context is None:
            return
        _, symbol, actual_category = context
        self._show_symbol_quick_menu_at(table, row, symbol, actual_category, table.viewport().mapToGlobal(pos))

    def _show_symbol_quick_menu_at(self, table, row: int, symbol: str, category_name: str, global_pos):
        menu = QMenu(self)
        highlight_action = menu.addAction(f"高亮 {symbol}")
        trade_action = menu.addAction(f"送入自动交易: {symbol}")
        backtest_action = menu.addAction(f"送入回测: {symbol}")

        selected_action = menu.exec(global_pos)
        if selected_action == highlight_action:
            self._highlight_symbol(symbol, category_name)
        elif selected_action == trade_action:
            self._send_symbol_to_trade(symbol, category_name)
        elif selected_action == backtest_action:
            self._send_symbol_to_backtest(symbol, category_name)

    def _get_row_symbol_context(self, table, category_name: str, row: int):
        if row < 0:
            return None
        symbol_item = table.item(row, 0)
        if symbol_item is None:
            return None
        symbol = symbol_item.text().strip()
        if not symbol:
            return None
        actual_category = category_name
        if category_name == "总览":
            category_item = table.item(row, 3)
            if category_item and category_item.text().strip():
                actual_category = category_item.text().strip()
        return row, symbol, actual_category

    def update_progress(self, val, msg, current_symbol, scanned, total, est_remaining):
        """更新扫描进度显示"""
        self.progress_bar.setValue(val)
        progress_text = msg
        if est_remaining:
            progress_text = f"{msg} | 剩余 {est_remaining}"
        self.scan_progress_label.setText(progress_text)
        self.status_label.setText(f"进度: {val}%")
        if val >= 100:
            return

        # 更新当前扫描的交易对
        if current_symbol:
            self.current_symbol_label.setText(f"📊 正在分析: {current_symbol}")
        else:
            self.current_symbol_label.setText("📊 正在分析: 市场截面与候选池")

        # 更新进度统计
        if scanned > 0 and total > 0:
            self.scan_stats_label.setText(f"已完成: {scanned}/{total}")
        elif total > 0:
            self.scan_stats_label.setText(f"总数: {total}")
        
        # 更新实时日志 - 显示详细进度
        log_msg = f"[{val}%] {msg}"
        if current_symbol and total > 0:
            log_msg += f" | 当前: {current_symbol} ({scanned}/{total})"
        elif total > 0:
            log_msg += f" | 进度: {scanned}/{total}"
        self.scan_log_label.setText(f"📋 实时日志: {log_msg}")

        # 向外广播进度（每 4 次发一条），同时携带进度数字供进度条使用
        self._log_progress_counter = getattr(self, '_log_progress_counter', 0) + 1
        if self._log_progress_counter % 4 == 1 or val >= 95:
            detail = ""
            if current_symbol and total > 0:
                detail = f"  {current_symbol} ({scanned}/{total})"
            elif total > 0:
                detail = f"  {scanned}/{total}"
            # 格式固定为 "[进度%|scanned|total] msg detail"，供外部解析进度条
            self.scan_log_signal.emit(
                f"[{val}%|{scanned}|{total}] {msg}{detail}",
                "INFO",
            )

    def on_scan_error(self, err_msg):
        """扫描错误回调"""
        self.is_scanning = False
        self.scan_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_label.setText(f"扫描出错: {err_msg}")
        self.current_symbol_label.setText(f"❌ 扫描失败")
        self.current_strategy_label.setText("策略: -")
        self.scan_stats_label.setText("")
        self.scan_progress_label.setText(f"错误: {err_msg}")
        self.scan_log_signal.emit(f"❌ 扫描出错: {err_msg}", "ERROR")

        # 安全清理线程引用（不阻塞 UI）
        old_thread = getattr(self, '_scan_thread', None)
        if old_thread is not None:
            # 错误回调时线程可能仍在退出路径上，暂存僵尸列表保持引用
            self._safe_discard_scan_thread(old_thread, wait_ms=200)
            self._scan_thread = None

        # 即使报错，如果队列里还有任务，也要继续执行下一个（延迟5秒）
        if self.current_auto_strategies:
            print(f"[on_scan_error] 扫描出错，5秒后尝试下一个策略...")
            QTimer.singleShot(5000, self.start_next_auto_strategy)
        else:
            if self.auto_scan_enabled:
                print(f"[on_scan_error] 扫描出错且队列为空，重置间隔...")
                self.countdown_remaining = self.interval_spin.value() * 60
