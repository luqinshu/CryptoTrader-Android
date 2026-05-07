from datetime import datetime
from src.data.manager import DataManager
from src.data.downloader import DataDownloader
from src.qt_compat import QComboBox, QDate, QDateEdit, QFormLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMessageBox, QProgressBar, QPushButton, QTableWidget, QTableWidgetItem, QThread, QVBoxLayout, QWidget, Qt, Signal

class DownloadThread(QThread):
    """异步下载线程 - 支持单周期或多周期批量下载"""
    progress = Signal(str, int, str) # 周期, 数量, 当前进度时间
    finished = Signal(int)
    error = Signal(str)

    def __init__(self, downloader, symbol, bars, start_date, end_date):
        super().__init__()
        self.downloader = downloader
        self.symbol = symbol
        self.bars = bars # 这是一个列表
        self.start_date = start_date
        self.end_date = end_date

    def run(self):
        try:
            total_all_bars = 0
            for bar in self.bars:
                if self.downloader._stop_flag:
                    break
                def on_progress(count, time_str):
                    self.progress.emit(bar, count, time_str)
                
                total = self.downloader.download_range(
                    self.symbol, bar, self.start_date, self.end_date, on_progress
                )
                total_all_bars += total
                if self.downloader._stop_flag:
                    break
            self.finished.emit(total_all_bars)
        except Exception as e:
            self.error.emit(str(e))

class DatabasePage(QWidget):
    """交易对数据库管理页面 - 增强版"""
    symbols_loaded = Signal(list) # 新增：用于安全传回品种列表
    open_backtest = Signal(str, str)

    def __init__(self, okx_client):
        super().__init__()
        self.okx_client = okx_client
        self.dm = DataManager()
        self.downloader = DataDownloader(okx_client, self.dm)
        self.inventory_cache = []
        self.init_ui()
        
        # 信号连接：确保 UI 更新在主线程执行
        self.symbols_loaded.connect(self.on_symbols_loaded)
        
        # 异步加载品种列表
        import threading
        thread = threading.Thread(target=self.fetch_all_symbols, daemon=True)
        thread.start()
        self.refresh_inventory()

    def on_symbols_loaded(self, symbols):
        """主线程槽函数：安全更新 UI"""
        self.symbol_input.clear()
        self.symbol_input.addItems(symbols)
        if "BTC-USDT-SWAP" in symbols:
            self.symbol_input.setCurrentText("BTC-USDT-SWAP")
        elif symbols:
            self.symbol_input.setCurrentIndex(0)
        self.status_label.setText(f"✅ 已成功加载 {len(symbols)} 个交易所品种")

    def init_ui(self):
        layout = QVBoxLayout(self)

        # 1. 下载控制栏
        dl_group = QGroupBox("下载/同步历史数据")
        dl_layout = QHBoxLayout(dl_group)
        
        form_layout = QFormLayout()
        self.symbol_input = QComboBox()
        self.symbol_input.setEditable(True)
        self.symbol_input.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.symbol_input.setPlaceholderText("正在获取交易所品种列表...")
        form_layout.addRow("交易对:", self.symbol_input)
        
        self.bar_input = QComboBox()
        # 增加“全周期”选项
        self.bar_input.addItems(["全周期(3m-1D)", "3m", "5m", "15m", "30m", "1H", "2H", "4H", "1D"])
        self.bar_input.setCurrentText("1H")
        form_layout.addRow("周期:", self.bar_input)
        
        dl_layout.addLayout(form_layout)

        form2_layout = QFormLayout()
        self.start_date = QDateEdit()
        self.start_date.setDate(QDate.currentDate().addMonths(-3))
        self.start_date.setCalendarPopup(True)
        form2_layout.addRow("开始日期:", self.start_date)
        
        self.end_date = QDateEdit()
        self.end_date.setDate(QDate.currentDate())
        self.end_date.setCalendarPopup(True)
        form2_layout.addRow("结束日期:", self.end_date)
        
        dl_layout.addLayout(form2_layout)

        self.dl_btn = QPushButton("📥 开始下载任务")
        self.dl_btn.setStyleSheet("background-color: #28a745; color: white; font-weight: bold; padding: 15px;")
        self.dl_btn.clicked.connect(self.start_download)
        dl_layout.addWidget(self.dl_btn)

        # 新增暂停按钮
        self.pause_btn = QPushButton("⏸ 暂停")
        self.pause_btn.setStyleSheet("background-color: #ffc107; color: black; font-weight: bold; padding: 15px;")
        self.pause_btn.clicked.connect(self.toggle_pause)
        self.pause_btn.setEnabled(False)
        dl_layout.addWidget(self.pause_btn)

        # 新增停止按钮
        self.stop_btn = QPushButton("⏹ 停止")
        self.stop_btn.setStyleSheet("background-color: #dc3545; color: white; font-weight: bold; padding: 15px;")
        self.stop_btn.clicked.connect(self.stop_download)
        self.stop_btn.setEnabled(False)
        dl_layout.addWidget(self.stop_btn)

        layout.addWidget(dl_group)

        # 2. 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        self.status_label = QLabel("就绪 (支持模糊搜索交易对，如输入 BTC 即可筛选)")
        self.status_label.setStyleSheet("color: #aaaaaa; font-style: italic;")
        layout.addWidget(self.status_label)

        # 3. 本地清单
        inv_group = QGroupBox("本地数据库资产清单")
        inv_layout = QVBoxLayout(inv_group)

        filter_layout = QHBoxLayout()
        self.inventory_search = QLineEdit()
        self.inventory_search.setPlaceholderText("搜索交易对，如 BTC / ETH / SWAP")
        self.inventory_search.textChanged.connect(self.apply_inventory_filters)
        filter_layout.addWidget(self.inventory_search)

        self.inventory_bar_filter = QComboBox()
        self.inventory_bar_filter.addItems(["全部周期", "3m", "5m", "15m", "30m", "1H", "2H", "4H", "1D"])
        self.inventory_bar_filter.currentTextChanged.connect(self.apply_inventory_filters)
        filter_layout.addWidget(self.inventory_bar_filter)

        self.inventory_sort_combo = QComboBox()
        self.inventory_sort_combo.addItems(["按交易对/周期", "按K线数量(高到低)", "按K线数量(低到高)"])
        self.inventory_sort_combo.currentTextChanged.connect(self.apply_inventory_filters)
        filter_layout.addWidget(self.inventory_sort_combo)
        inv_layout.addLayout(filter_layout)
        
        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["交易对", "周期", "本地K线数", "覆盖时间范围"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        inv_layout.addWidget(self.table)

        action_layout = QHBoxLayout()
        refresh_btn = QPushButton("🔄 刷新本地清单")
        refresh_btn.clicked.connect(self.refresh_inventory)
        action_layout.addWidget(refresh_btn)

        detect_gap_btn = QPushButton("🔍 检测缺口")
        detect_gap_btn.clicked.connect(self.detect_selected_gap)
        action_layout.addWidget(detect_gap_btn)

        fill_gap_btn = QPushButton("🩹 一键补齐")
        fill_gap_btn.clicked.connect(self.fill_selected_gap)
        action_layout.addWidget(fill_gap_btn)

        update_latest_btn = QPushButton("⏫ 更新到最新")
        update_latest_btn.clicked.connect(self.update_selected_to_latest)
        action_layout.addWidget(update_latest_btn)

        backtest_btn = QPushButton("📈 送入回测")
        backtest_btn.clicked.connect(self.send_selected_to_backtest)
        action_layout.addWidget(backtest_btn)

        delete_row_btn = QPushButton("🗑 删除选中周期")
        delete_row_btn.clicked.connect(self.delete_selected_bar)
        action_layout.addWidget(delete_row_btn)

        delete_symbol_btn = QPushButton("🗑 删除选中交易对全部周期")
        delete_symbol_btn.clicked.connect(self.delete_selected_symbol)
        action_layout.addWidget(delete_symbol_btn)

        download_all_btn = QPushButton("⬇ 一键下载全周期全部K线")
        download_all_btn.setStyleSheet(
            "background-color: #007bff; color: white; font-weight: bold; padding: 8px 16px;"
        )
        download_all_btn.clicked.connect(self.download_all_periods_for_selected_symbol)
        action_layout.addWidget(download_all_btn)
        inv_layout.addLayout(action_layout)
        
        layout.addWidget(inv_group)

    def fetch_all_symbols(self):
        """从交易所获取所有品种并填充选择框（后台线程执行）"""
        try:
            # 获取永续合约
            res_swap = self.okx_client.get_tickers(instType="SWAP")
            # 获取现货
            res_spot = self.okx_client.get_tickers(instType="SPOT")
            
            symbols = []
            if res_swap.get('code') == '0':
                symbols.extend([t['instId'] for t in res_swap['data']])
            if res_spot.get('code') == '0':
                symbols.extend([t['instId'] for t in res_spot['data']])
            
            symbols.sort()
            
            # 安全发射信号传回主线程
            self.symbols_loaded.emit(symbols)
            
        except Exception as e:
            print(f"获取品种列表失败: {e}")

    def refresh_inventory(self):
        """刷新本地数据表格"""
        inventory = self.dm.get_local_inventory()
        self.inventory_cache = inventory
        self.apply_inventory_filters()

    def apply_inventory_filters(self):
        """应用库存筛选与排序"""
        inventory = list(self.inventory_cache)
        keyword = self.inventory_search.text().strip().upper() if hasattr(self, 'inventory_search') else ""
        bar_filter = self.inventory_bar_filter.currentText() if hasattr(self, 'inventory_bar_filter') else "全部周期"
        sort_mode = self.inventory_sort_combo.currentText() if hasattr(self, 'inventory_sort_combo') else "按交易对/周期"

        if keyword:
            inventory = [item for item in inventory if keyword in item['symbol'].upper() or keyword in item['bar'].upper()]

        if bar_filter != "全部周期":
            inventory = [item for item in inventory if item['bar'] == bar_filter]

        if sort_mode == "按K线数量(高到低)":
            inventory.sort(key=lambda x: (-x['count'], x['symbol'], x['bar']))
        elif sort_mode == "按K线数量(低到高)":
            inventory.sort(key=lambda x: (x['count'], x['symbol'], x['bar']))
        else:
            inventory.sort(key=lambda x: (x['symbol'], x['bar']))

        self.table.setRowCount(0)
        for i, item in enumerate(inventory):
            self.table.insertRow(i)
            self.table.setItem(i, 0, QTableWidgetItem(item['symbol']))
            self.table.setItem(i, 1, QTableWidgetItem(item['bar']))
            self.table.setItem(i, 2, QTableWidgetItem(f"{item['count']:,}"))
            self.table.setItem(i, 3, QTableWidgetItem(item['range']))
        self.status_label.setText(f"📚 本地库存 {len(inventory)} 条，支持搜索、周期过滤和删除")

    def get_selected_inventory_row(self):
        """获取当前选中的库存项"""
        row = self.table.currentRow()
        if row < 0:
            QMessageBox.warning(self, "提示", "请先在本地清单中选中一行")
            return None

        symbol_item = self.table.item(row, 0)
        bar_item = self.table.item(row, 1)
        if not symbol_item or not bar_item:
            return None
        return symbol_item.text(), bar_item.text()

    def detect_selected_gap(self):
        """检测选中数据的时间缺口"""
        selected = self.get_selected_inventory_row()
        if not selected:
            return

        symbol, bar = selected
        gap_info = self.dm.detect_gaps(symbol, bar)
        if not gap_info.get("success"):
            QMessageBox.warning(self, "缺口检测", gap_info.get("message", "检测失败"))
            return

        if gap_info["gap_count"] == 0:
            message = f"{symbol} {bar} 未发现时间缺口，数据连续。"
        else:
            preview = "\n".join(
                f"{idx + 1}. {gap['start']} -> {gap['end']}，缺失 {gap['missing_bars']} 根"
                for idx, gap in enumerate(gap_info["gaps"][:5])
            )
            if gap_info["gap_count"] > 5:
                preview += f"\n... 其余 {gap_info['gap_count'] - 5} 个缺口未展开"
            message = (
                f"{symbol} {bar} 共发现 {gap_info['gap_count']} 个缺口，"
                f"累计缺失 {gap_info['missing_bars']} 根K线。\n\n{preview}"
            )

        self.status_label.setText(f"🔍 已完成 {symbol} {bar} 缺口检测")
        QMessageBox.information(self, "缺口检测结果", message)

    def fill_selected_gap(self):
        """一键补齐选中数据的时间缺口"""
        selected = self.get_selected_inventory_row()
        if not selected:
            return

        symbol, bar = selected
        gap_info = self.dm.detect_gaps(symbol, bar)
        if not gap_info.get("success"):
            QMessageBox.warning(self, "补齐失败", gap_info.get("message", "无法检测缺口"))
            return

        if gap_info["gap_count"] == 0:
            QMessageBox.information(self, "无需补齐", f"{symbol} {bar} 当前未检测到缺口。")
            return

        start = min(gap["start"][:10] for gap in gap_info["gaps"])
        end = max(gap["end"][:10] for gap in gap_info["gaps"])
        reply = QMessageBox.question(
            self,
            "确认补齐缺口",
            (
                f"检测到 {symbol} {bar} 有 {gap_info['gap_count']} 个缺口。\n"
                f"将重新下载 {start} 至 {end} 区间以补齐断档，是否继续？"
            ),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        self.begin_download_task(symbol, [bar], start, end, action_label="补齐缺口")

    def update_selected_to_latest(self):
        """将选中数据更新到最新"""
        selected = self.get_selected_inventory_row()
        if not selected:
            return

        symbol, bar = selected
        gap_info = self.dm.detect_gaps(symbol, bar)
        if not gap_info.get("success"):
            QMessageBox.warning(self, "更新失败", gap_info.get("message", "无法读取本地数据范围"))
            return

        start_date = gap_info["end_date"]
        end_date = datetime.now().strftime("%Y-%m-%d")
        if start_date >= end_date:
            QMessageBox.information(self, "已是最新", f"{symbol} {bar} 的本地数据已经覆盖到今天。")
            return

        reply = QMessageBox.question(
            self,
            "确认更新到最新",
            f"将从 {start_date} 开始增量更新 {symbol} {bar} 到 {end_date}，是否继续？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        self.begin_download_task(symbol, [bar], start_date, end_date, action_label="更新到最新")

    def send_selected_to_backtest(self):
        """将选中交易对和周期送入回测页面"""
        selected = self.get_selected_inventory_row()
        if not selected:
            return

        symbol, bar = selected
        self.open_backtest.emit(symbol, bar)
        self.status_label.setText(f"📈 已将 {symbol} {bar} 送入回测页面")

    def delete_selected_bar(self):
        """删除选中的交易对周期数据"""
        selected = self.get_selected_inventory_row()
        if not selected:
            return

        symbol, bar = selected
        reply = QMessageBox.question(
            self,
            "确认删除",
            f"确定删除 {symbol} 的 {bar} 周期本地数据吗？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        deleted = self.dm.delete_klines(symbol, bar)
        if deleted:
            self.status_label.setText(f"🗑 已删除 {symbol} {bar} 本地数据")
            self.refresh_inventory()
        else:
            QMessageBox.warning(self, "删除失败", "未找到对应的本地数据文件")

    def delete_selected_symbol(self):
        """删除选中交易对的全部周期数据"""
        selected = self.get_selected_inventory_row()
        if not selected:
            return

        symbol, _ = selected
        reply = QMessageBox.question(
            self,
            "确认删除全部周期",
            f"确定删除 {symbol} 的全部本地周期数据吗？此操作不可恢复。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        deleted_count = self.dm.delete_symbol(symbol)
        if deleted_count > 0:
            self.status_label.setText(f"🗑 已删除 {symbol} 的 {deleted_count} 个周期文件")
            self.refresh_inventory()
        else:
            QMessageBox.warning(self, "删除失败", "未找到该交易对的本地目录")

    def download_all_periods_for_selected_symbol(self):
        """一键下载所选交易对所有周期的全部历史K线"""
        selected = self.get_selected_inventory_row()
        if not selected:
            # 如果清单为空，尝试从表格当前选中行获取 symbol
            row = self.table.currentRow()
            if row < 0:
                QMessageBox.warning(self, "提示", "请先在本地清单中选中一行，或在上方输入框中选择交易对")
                return
            symbol = self.table.item(row, 0)
            symbol = symbol.text() if symbol else None
            if not symbol:
                return
        else:
            symbol, _ = selected

        # 全周期列表
        all_bars = ["3m", "5m", "15m", "30m", "1H", "2H", "4H", "1D"]

        # 从 OKX 上线日或最早可用日开始下载全部历史
        start_date = "2019-01-01"
        end_date = datetime.now().strftime("%Y-%m-%d")

        reply = QMessageBox.question(
            self,
            "确认全量下载",
            (
                f"将为 {symbol} 下载全部 {len(all_bars)} 个周期"
                f"（{'/'.join(all_bars)}）\n"
                f"时间范围：{start_date} 至 {end_date}\n\n"
                f"该操作可能需要较长时间，是否继续？"
            ),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        self.symbol_input.setCurrentText(symbol)
        self.status_label.setText(
            f"⬇ 开始为 {symbol} 下载全周期全部历史K线..."
        )
        self.begin_download_task(symbol, all_bars, start_date, end_date, action_label="全量下载")

    def start_download(self):
        """启动下载"""
        symbol = self.symbol_input.currentText()
        if not symbol:
            QMessageBox.warning(self, "警告", "请先选择或输入一个交易对")
            return

        if self.start_date.date() > self.end_date.date():
            QMessageBox.warning(self, "警告", "开始日期不能晚于结束日期")
            return

        bar_selection = self.bar_input.currentText()
        start = self.start_date.date().toString("yyyy-MM-dd")
        end = self.end_date.date().toString("yyyy-MM-dd")

        # 判定是否全周期
        if "全周期" in bar_selection:
            bars = ["3m", "5m", "15m", "30m", "1H", "2H", "4H", "1D"]
        else:
            bars = [bar_selection]

        self.begin_download_task(symbol, bars, start, end, action_label="下载")

    def begin_download_task(self, symbol: str, bars: list, start: str, end: str, action_label: str = "下载"):
        """统一启动下载/补齐/更新任务"""
        self.downloader.reset_state()
        self.dl_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)
        self.stop_btn.setEnabled(True)
        self.pause_btn.setText("⏸ 暂停")
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.status_label.setText(f"🚀 正在{action_label} {symbol} {'/'.join(bars)} 数据...")

        self.thread = DownloadThread(self.downloader, symbol, bars, start, end)
        self.thread.progress.connect(self.on_download_progress)
        self.thread.finished.connect(self.on_download_finished)
        self.thread.error.connect(self.on_download_error)
        self.thread.start()

    def toggle_pause(self):
        """切换暂停/恢复"""
        if self.downloader._pause_flag:
            self.downloader.resume()
            self.pause_btn.setText("⏸ 暂停")
            self.status_label.setText("▶️ 任务已恢复...")
        else:
            self.downloader.pause()
            self.pause_btn.setText("▶️ 恢复")
            self.status_label.setText("⏸ 任务已暂停")

    def stop_download(self):
        """强制停止下载"""
        reply = QMessageBox.question(self, "确认停止", "确定要强制停止当前的下载任务吗？", 
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            self.downloader.stop()
            self.status_label.setText("⏹ 正在请求停止...")
            self.pause_btn.setEnabled(False)
            self.stop_btn.setEnabled(False)

    def on_download_progress(self, bar, count, time_str):
        self.status_label.setText(f"🚀 正在批量下载 [{bar}] ... 已获取 {count:,} 根，进度点: {time_str}")

    def on_download_finished(self, total):
        self.dl_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setVisible(False)
        if self.downloader._stop_flag:
            self.status_label.setText(f"⏹ 下载已停止，已存入 {total:,} 根 K 线数据")
        else:
            self.status_label.setText(f"✅ 所有任务完成！累计存入 {total:,} 根 K 线数据")
        self.refresh_inventory()
        if not self.downloader._stop_flag:
            QMessageBox.information(self, "批量下载成功", f"所选周期数据已全部同步至本地数据库。")

    def on_download_error(self, err):
        self.dl_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setVisible(False)
        self.status_label.setText(f"❌ 下载中断: {err}")
        QMessageBox.critical(self, "错误", f"下载过程中发生异常:\n{err}")
