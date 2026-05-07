"""
交易对池页面 - 收集、导出并定时推送扫描结果。
"""

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict


from src.qt_compat import (
    QCheckBox, QComboBox, QFileDialog, QGroupBox, QHBoxLayout, QHeaderView,
    QLabel, QLineEdit, QMenu, QMessageBox, QPushButton, QSpinBox,
    QTableWidget, QTableWidgetItem, QTimer, QVBoxLayout, QWidget, Qt, Signal,
)
from src.utils.trade_pool_report import (
    export_trade_pool_to_excel,
    send_email_with_attachment,
    send_telegram_document,
    send_server_chan_message,
)


class TradePoolPage(QWidget):
    """交易对池页面"""

    push_result_signal = Signal(str, str)

    def __init__(self):
        super().__init__()
        self.pool_data = []
        self.monitor_pool_page = None
        self._pool_data_raw: Dict[str, dict] = {}
        self._pool_data_path = Path(__file__).resolve().parent.parent / "trade_pool_data.json"
        self.max_pool_size = 200
        self.push_timer = QTimer(self)
        self.push_timer.timeout.connect(self.push_trade_pool_report)
        self.config_path = Path(__file__).resolve().parent.parent / "trade_pool_push_config.json"
        self.telegram_config_path = Path(__file__).resolve().parent.parent / "telegram_config.json"
        self.init_ui()
        self.push_result_signal.connect(self._handle_push_result)
        self.load_push_config()
        self._load_pool_data()

    @staticmethod
    def _timestamp_sort_key(item: dict):
        timestamp = str(item.get('time', '') or '')
        try:
            return datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return datetime.min

    def init_ui(self):
        layout = QVBoxLayout(self)

        header_group = QGroupBox("交易对池管理")
        header_layout = QHBoxLayout(header_group)

        title = QLabel("📊 交易对池 - 扫描历史记录")
        title.setStyleSheet("color: #00aaff; font-size: 14px; font-weight: bold;")
        header_layout.addWidget(title)

        header_layout.addStretch()

        self.stats_label = QLabel("总计：0 条记录")
        self.stats_label.setStyleSheet("color: #aaaaaa; font-size: 12px;")
        header_layout.addWidget(self.stats_label)

        self.export_btn = QPushButton("📤 导出 Excel")
        self.export_btn.clicked.connect(self.export_trade_pool_excel)
        self.export_btn.setStyleSheet("background-color: #28a745; color: white; font-weight: bold; padding: 5px 15px;")
        header_layout.addWidget(self.export_btn)

        clear_btn = QPushButton("🗑️ 清空池")
        clear_btn.clicked.connect(self.clear_pool)
        clear_btn.setStyleSheet("background-color: #dc3545; color: white; font-weight: bold; padding: 5px 15px;")
        header_layout.addWidget(clear_btn)

        layout.addWidget(header_group)

        push_group = QGroupBox("Excel 定时推送")
        push_layout = QVBoxLayout(push_group)
        push_layout.setContentsMargins(8, 8, 8, 8)
        push_layout.setSpacing(6)

        def add_compact_field(row_layout, label_text, widget, stretch=0):
            label = QLabel(label_text)
            label.setStyleSheet("color: #aaaaaa; font-size: 11px;")
            row_layout.addWidget(label)
            row_layout.addWidget(widget, stretch)

        telegram_row = QHBoxLayout()
        telegram_row.setSpacing(6)

        self.push_method_combo = QComboBox()
        self.push_method_combo.addItems(["Telegram(手机)", "邮箱", "微信(Server酱)"])
        add_compact_field(telegram_row, "渠道:", self.push_method_combo)

        self.push_scope_combo = QComboBox()
        self.push_scope_combo.addItems(["精选(最近新增/高评分/共振)", "最近新增", "高评分", "共振", "全量"])
        add_compact_field(telegram_row, "范围:", self.push_scope_combo)

        self.push_interval_spin = QSpinBox()
        self.push_interval_spin.setRange(1, 1440)
        self.push_interval_spin.setValue(30)
        self.push_interval_spin.setSuffix(" 分钟")
        add_compact_field(telegram_row, "间隔:", self.push_interval_spin)

        self.recent_minutes_spin = QSpinBox()
        self.recent_minutes_spin.setRange(1, 10080)
        self.recent_minutes_spin.setValue(180)
        self.recent_minutes_spin.setSuffix(" 分钟")
        add_compact_field(telegram_row, "新增:", self.recent_minutes_spin)

        self.min_push_score_spin = QSpinBox()
        self.min_push_score_spin.setRange(0, 1000)
        self.min_push_score_spin.setValue(84)
        add_compact_field(telegram_row, "评分:", self.min_push_score_spin)

        self.min_resonance_hits_spin = QSpinBox()
        self.min_resonance_hits_spin.setRange(2, 20)
        self.min_resonance_hits_spin.setValue(2)
        add_compact_field(telegram_row, "共振:", self.min_resonance_hits_spin)

        self.export_dir_edit = QLineEdit(str((Path(__file__).resolve().parent.parent.parent / "reports" / "trade_pool").resolve()))
        add_compact_field(telegram_row, "目录:", self.export_dir_edit, 2)
        browse_btn = QPushButton("选择目录")
        browse_btn.clicked.connect(self.choose_export_dir)
        telegram_row.addWidget(browse_btn)

        self.telegram_token_edit = QLineEdit()
        self.telegram_token_edit.setPlaceholderText("Telegram Bot Token")
        add_compact_field(telegram_row, "Token:", self.telegram_token_edit, 2)

        self.telegram_chat_id_edit = QLineEdit()
        self.telegram_chat_id_edit.setPlaceholderText("聊天 chat_id")
        add_compact_field(telegram_row, "Chat:", self.telegram_chat_id_edit)

        self.server_chan_key_edit = QLineEdit()
        self.server_chan_key_edit.setPlaceholderText("Server酱 SendKey")
        add_compact_field(telegram_row, "微信Key:", self.server_chan_key_edit, 2)
        push_layout.addLayout(telegram_row)

        email_row = QHBoxLayout()
        email_row.setSpacing(6)

        self.smtp_host_edit = QLineEdit()
        self.smtp_host_edit.setPlaceholderText("smtp.example.com")
        add_compact_field(email_row, "SMTP:", self.smtp_host_edit, 2)

        self.smtp_port_spin = QSpinBox()
        self.smtp_port_spin.setRange(1, 65535)
        self.smtp_port_spin.setValue(587)
        add_compact_field(email_row, "端口:", self.smtp_port_spin)

        self.smtp_user_edit = QLineEdit()
        self.smtp_user_edit.setPlaceholderText("邮箱账号")
        add_compact_field(email_row, "用户:", self.smtp_user_edit, 2)

        self.smtp_password_edit = QLineEdit()
        self.smtp_password_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.smtp_password_edit.setPlaceholderText("邮箱密码/授权码")
        add_compact_field(email_row, "密码:", self.smtp_password_edit, 2)

        self.email_to_edit = QLineEdit()
        self.email_to_edit.setPlaceholderText("接收邮箱")
        add_compact_field(email_row, "收件:", self.email_to_edit, 2)

        self.email_tls_check = QCheckBox()
        self.email_tls_check.setChecked(True)
        email_row.addWidget(QLabel("TLS:"))
        email_row.addWidget(self.email_tls_check)

        self.save_push_config_btn = QPushButton("保存配置")
        self.save_push_config_btn.clicked.connect(self.save_push_config)
        email_row.addWidget(self.save_push_config_btn)

        self.push_now_btn = QPushButton("立即推送")
        self.push_now_btn.clicked.connect(self.push_trade_pool_report)
        email_row.addWidget(self.push_now_btn)

        self.start_push_btn = QPushButton("启动定时推送")
        self.start_push_btn.clicked.connect(self.start_scheduled_push)
        email_row.addWidget(self.start_push_btn)

        self.stop_push_btn = QPushButton("停止定时推送")
        self.stop_push_btn.clicked.connect(self.stop_scheduled_push)
        email_row.addWidget(self.stop_push_btn)
        push_layout.addLayout(email_row)

        self.push_status_label = QLabel("当前未启动定时推送。")
        self.push_status_label.setStyleSheet("color: #ffd166;")
        push_layout.addWidget(self.push_status_label)

        layout.addWidget(push_group)

        self.pool_table = QTableWidget()
        self.pool_table.setColumnCount(10)
        self.pool_table.setHorizontalHeaderLabels([
            "时间", "交易对", "方向", "价格", "24h涨幅", "评分", "命中次数", "信号理由", "策略来源", "移除"
        ])
        self.pool_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.pool_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.pool_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.pool_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.pool_table.horizontalHeader().setSectionResizeMode(9, QHeaderView.ResizeMode.ResizeToContents)
        self.pool_table.setAlternatingRowColors(True)
        self.pool_table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.pool_table.customContextMenuRequested.connect(self._show_pool_item_menu)
        self.pool_table.setStyleSheet("""
            QTableWidget {
                background-color: #1e1e1e;
                color: #cccccc;
                gridline-color: #333333;
            }
            QTableWidget::item {
                padding: 5px;
            }
        """)
        layout.addWidget(self.pool_table)

        self.status_label = QLabel("就绪 - 等待扫描结果...")
        self.status_label.setStyleSheet("color: #00ccff; font-size: 11px;")
        layout.addWidget(self.status_label)

    def add_results(self, results, strategy_name="未知策略", increment_hits=True):
        if not results:
            return

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for res in results:
            symbol = res.get('symbol', res.get('inst_id', '-'))
            direction = res.get('direction', res.get('side', 'N/A'))
            reason = ', '.join(res.get('signals', [])) if isinstance(res.get('signals'), list) else str(res.get('reason', '-'))
            signal_hint = self._extract_signal_hint(res, reason)
            pool_key = f"{symbol}:{direction}:{strategy_name}"
            self._pool_data_raw[pool_key] = res
            existing = next((item for item in self.pool_data if item['symbol'] == symbol and item['direction'] == direction and item['strategy'] == strategy_name), None)
            if existing:
                new_hits = existing.get('hits', 1) + (1 if increment_hits else 0)
                existing.update({
                    'time': timestamp,
                    'price': res.get('last_price', existing.get('price', 0)),
                    'change_24h': res.get('price_change_24h', res.get('change_24h', existing.get('change_24h', 0))),
                    'score': max(existing.get('score', 0), res.get('score', 0)),
                    'opportunity_score': max(existing.get('opportunity_score', 0), res.get('opportunity_score', res.get('score', 0))),
                    'reason': reason,
                    'signal_hint': signal_hint or existing.get('signal_hint', ''),
                    'hits': new_hits,
                    'is_new_signal': bool(res.get('is_new_signal', existing.get('is_new_signal', False))),
                    'is_resonance': bool(res.get('is_resonance', existing.get('is_resonance', False))),
                    'resonance_count': max(int(existing.get('resonance_count', 1) or 1), int(res.get('resonance_count', 1) or 1)),
                    'streak_count': max(int(existing.get('streak_count', 1) or 1), int(res.get('streak_count', 1) or 1)),
                })
                continue

            self.pool_data.insert(0, {
                'time': timestamp,
                'symbol': symbol,
                'direction': direction,
                'price': res.get('last_price', 0),
                'change_24h': res.get('price_change_24h', res.get('change_24h', 0)),
                'score': res.get('score', 0),
                'opportunity_score': res.get('opportunity_score', res.get('score', 0)),
                'hits': 1,
                'reason': reason,
                'signal_hint': signal_hint,
                'strategy': strategy_name,
                'is_new_signal': bool(res.get('is_new_signal', False)),
                'is_resonance': bool(res.get('is_resonance', False)),
                'resonance_count': int(res.get('resonance_count', 1) or 1),
                'streak_count': int(res.get('streak_count', 1) or 1),
            })

        self.pool_data.sort(key=self._timestamp_sort_key, reverse=True)
        if len(self.pool_data) > self.max_pool_size:
            self.pool_data = self.pool_data[:self.max_pool_size]

        self.refresh_table()
        self.status_label.setText(f"✅ 已添加 {len(results)} 条记录到交易对池")
        self.stats_label.setText(f"总计：{len(self.pool_data)} 条记录")
        self._save_pool_data()

    def refresh_table(self):
        # 按时间由新到旧排序
        self.pool_data.sort(key=self._timestamp_sort_key, reverse=True)
        self.pool_table.setRowCount(len(self.pool_data))

        for row, item in enumerate(self.pool_data):
            time_item = QTableWidgetItem(item['time'])
            time_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.pool_table.setItem(row, 0, time_item)

            symbol_item = QTableWidgetItem(item['symbol'])
            symbol_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.pool_table.setItem(row, 1, symbol_item)

            direction = item['direction']
            direction_item = QTableWidgetItem(direction)
            direction_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            if direction in ['BUY', 'LONG']:
                direction_item.setBackground(Qt.GlobalColor.darkGreen)
                direction_item.setForeground(Qt.GlobalColor.white)
            elif direction in ['SELL', 'SHORT']:
                direction_item.setBackground(Qt.GlobalColor.darkRed)
                direction_item.setForeground(Qt.GlobalColor.white)
            self.pool_table.setItem(row, 2, direction_item)

            price_item = QTableWidgetItem(f"{item['price']:.4f}")
            price_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.pool_table.setItem(row, 3, price_item)

            change = item['change_24h']
            change_item = QTableWidgetItem(f"{change:+.2f}%")
            change_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            if change > 0:
                change_item.setForeground(Qt.GlobalColor.green)
            elif change < 0:
                change_item.setForeground(Qt.GlobalColor.red)
            self.pool_table.setItem(row, 4, change_item)

            score = item['score']
            opportunity_score = float(item.get('opportunity_score', score) or score)
            score_item = QTableWidgetItem(f"{opportunity_score:.1f}")
            score_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            if score >= 80:
                score_item.setForeground(Qt.GlobalColor.cyan)
            elif score >= 60:
                score_item.setForeground(Qt.GlobalColor.yellow)
            self.pool_table.setItem(row, 5, score_item)

            hits_item = QTableWidgetItem(str(item.get('hits', 1)))
            hits_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.pool_table.setItem(row, 6, hits_item)

            reason = item['reason'][:80] + '...' if len(item['reason']) > 80 else item['reason']
            self.pool_table.setItem(row, 7, QTableWidgetItem(reason))

            strategy_item = QTableWidgetItem(item['strategy'])
            strategy_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.pool_table.setItem(row, 8, strategy_item)

            remove_btn = QPushButton("🗑️ 移除")
            remove_btn.setStyleSheet("""
                QPushButton {
                    background-color: #dc3545;
                    color: white;
                    border: none;
                    border-radius: 3px;
                    padding: 3px 10px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #c82333;
                }
            """)
            remove_btn.clicked.connect(lambda checked, r=row: self.remove_item(r))
            btn_widget = QWidget()
            btn_layout = QHBoxLayout(btn_widget)
            btn_layout.addWidget(remove_btn)
            btn_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
            btn_layout.setContentsMargins(0, 0, 0, 0)
            self.pool_table.setCellWidget(row, 9, btn_widget)

    def _show_pool_item_menu(self, pos):
        """右键菜单：查看详细评分理由"""
        row = self.pool_table.rowAt(pos.y())
        if row < 0 or row >= len(self.pool_data):
            return
        item = self.pool_data[row]
        menu = QMenu(self)
        menu.setStyleSheet("""
            QMenu { background-color: #2a2a2a; color: #dddddd; border: 1px solid #444; }
            QMenu::item:selected { background-color: #3a3a3a; }
        """)
        detail_action = menu.addAction("📊 查看详细评分理由")
        detail_action.triggered.connect(lambda: self._show_trade_detail_dialog(item))
        if self.monitor_pool_page:
            monitor_action = menu.addAction("📡 加入监控池")
            monitor_action.triggered.connect(lambda: self._add_item_to_monitor_pool(item))
            preset_auto_action = menu.addAction("⚡ 加入监控池并预设自动开仓")
            preset_auto_action.triggered.connect(lambda: self._add_item_to_monitor_pool_with_auto_open(item))
        menu.exec(self.pool_table.viewport().mapToGlobal(pos))

    def _add_item_to_monitor_pool(self, item: dict):
        symbol = str(item.get('symbol', '') or '').strip()
        if not symbol or not self.monitor_pool_page:
            return
        try:
            self.monitor_pool_page.add_pair(symbol)
            self.status_label.setText(f"📡 已将 {symbol} 加入监控池")
        except Exception as exc:
            QMessageBox.warning(self, "加入监控池失败", f"{symbol} 加入监控池失败：{exc}")

    def _add_item_to_monitor_pool_with_auto_open(self, item: dict):
        symbol = str(item.get('symbol', '') or '').strip()
        if not symbol or not self.monitor_pool_page:
            return
        signal_type = self._infer_monitor_signal_type(item)
        if not signal_type:
            QMessageBox.information(
                self,
                "无法预设自动开仓",
                f"{symbol} 已可加入监控池，但当前记录无法可靠识别对应监控信号类型，请先使用普通“加入监控池”。",
            )
            return
        try:
            self.monitor_pool_page.add_pair_with_action_preset(symbol, signal_type, "提醒+自动开仓")
            self.status_label.setText(f"⚡ 已将 {symbol} 加入监控池，并把“{signal_type}”预设为自动开仓")
        except Exception as exc:
            QMessageBox.warning(self, "预设自动开仓失败", f"{symbol} 预设自动开仓失败：{exc}")

    @staticmethod
    def _infer_monitor_signal_type(item: dict) -> str:
        """从交易池记录里尽量推断监控池信号类型。"""
        text_parts = [
            str(item.get('signal_hint', '') or ''),
            str(item.get('reason', '') or ''),
            str(item.get('strategy', '') or ''),
            str(item.get('symbol', '') or ''),
        ]
        combined = " | ".join(text_parts)
        candidates = [
            "趋势突破",
            "大幅回调",
            "企稳突破",
            "放量异动",
            "动量背离",
        ]
        for candidate in candidates:
            if candidate in combined:
                return candidate
        return ""

    @staticmethod
    def _extract_signal_hint(res: dict, fallback_reason: str) -> str:
        """从扫描结果中提取更适合监控池映射的信号提示。"""
        candidates = []
        raw_signal = res.get('signal_type')
        if raw_signal:
            candidates.append(str(raw_signal))
        signals = res.get('signals')
        if isinstance(signals, list):
            candidates.extend(str(item) for item in signals if item)
        category = res.get('category') or res.get('strategy_category')
        if category:
            candidates.append(str(category))
        candidates.append(str(fallback_reason or ''))
        combined = " | ".join(candidates)
        for label in ["趋势突破", "大幅回调", "企稳突破", "放量异动", "动量背离"]:
            if label in combined:
                return label
        return ""

    def _show_trade_detail_dialog(self, item: dict):
        """弹窗显示交易对的详细评分理由"""
        from PySide6.QtWidgets import QDialog, QVBoxLayout, QTextEdit, QPushButton, QHBoxLayout

        dialog = QDialog(self)
        dialog.setWindowTitle(f"📊 详细评分分析 — {item.get('symbol', '-')}")
        dialog.setMinimumSize(700, 600)
        dialog.setStyleSheet("background-color: #1a1a2e; color: #ddd;")

        layout = QVBoxLayout(dialog)

        text = QTextEdit(dialog)
        text.setReadOnly(True)
        text.setStyleSheet("""
            QTextEdit {
                background-color: #0d1117; color: #cccccc;
                border: 1px solid #333; border-radius: 6px;
                font-family: 'Menlo', 'Monaco', monospace;
                font-size: 12px; padding: 12px;
            }
        """)

        html = self._build_detail_html(item)
        text.setHtml(html)
        layout.addWidget(text)

        btn_layout = QHBoxLayout()
        close_btn = QPushButton("关闭")
        close_btn.setStyleSheet("""
            QPushButton { background-color: #444; color: white; padding: 6px 20px;
                          border-radius: 4px; font-weight: bold; }
            QPushButton:hover { background-color: #555; }
        """)
        close_btn.clicked.connect(dialog.accept)
        btn_layout.addStretch()
        btn_layout.addWidget(close_btn)
        layout.addLayout(btn_layout)

        dialog.exec()

    def _build_detail_html(self, item: dict) -> str:
        """构建详细评分分析的 HTML"""
        symbol = item.get('symbol', '-')
        direction = item.get('direction', 'N/A')
        score = float(item.get('opportunity_score', item.get('score', 0)) or 0)
        strategy = item.get('strategy', '-')
        hits = item.get('hits', 1)
        price = float(item.get('price', 0) or 0)
        change_24h = float(item.get('change_24h', 0) or 0)
        reason = item.get('reason', '-')
        is_resonance = bool(item.get('is_resonance', False))
        resonance_count = int(item.get('resonance_count', 1) or 1)
        streak = int(item.get('streak_count', 1) or 1)

        # 方向颜色
        dir_color = "#00ff00" if direction in ('BUY', 'LONG') else "#ff4444"
        score_color = "#00ff88" if score >= 80 else ("#ffd700" if score >= 60 else "#ff6666")

        html = f"""
        <div style="padding: 10px;">
            <h2 style="color: #00aaff; margin-bottom: 8px;">{symbol}</h2>
            <table style="width: 100%; border-collapse: collapse; font-size: 13px;">
                <tr><td style="padding: 4px 8px; color: #888;">方向</td>
                    <td style="padding: 4px 8px; color: {dir_color}; font-weight: bold;">{direction}</td></tr>
                <tr><td style="padding: 4px 8px; color: #888;">综合评分</td>
                    <td style="padding: 4px 8px; color: {score_color}; font-weight: bold;">{score:.1f} / 100</td></tr>
                <tr><td style="padding: 4px 8px; color: #888;">当前价格</td>
                    <td style="padding: 4px 8px;">${price:.4f}</td></tr>
                <tr><td style="padding: 4px 8px; color: #888;">24h涨幅</td>
                    <td style="padding: 4px 8px; color: {'#00ff00' if change_24h >= 0 else '#ff4444'};">
                        {change_24h:+.2f}%</td></tr>
                <tr><td style="padding: 4px 8px; color: #888;">入选策略</td>
                    <td style="padding: 4px 8px; color: #ffd700;">{strategy}</td></tr>
                <tr><td style="padding: 4px 8px; color: #888;">命中次数</td>
                    <td style="padding: 4px 8px;">{hits} 次</td></tr>
                <tr><td style="padding: 4px 8px; color: #888;">共振引擎</td>
                    <td style="padding: 4px 8px;">{'是 ✨' if is_resonance else '否'} {f'(x{resonance_count})' if is_resonance else ''}</td></tr>
                <tr><td style="padding: 4px 8px; color: #888;">连续信号</td>
                    <td style="padding: 4px 8px;">{streak} 次</td></tr>
            </table>
            <hr style="border: 1px solid #333; margin: 12px 0;">
        """

        # 原始详情
        pool_key = f"{symbol}:{direction}:{strategy}"
        raw = self._pool_data_raw.get(pool_key, {})
        if raw:
            html += "<h3 style='color: #00aaff; margin: 8px 0;'>📋 策略评分明细</h3>"
            details = raw.get('details', {})
            if details and isinstance(details, dict):
                html += "<table style='width:100%; border-collapse:collapse; font-size:12px;'>"
                for dk, dv in details.items():
                    if dv is None or dv == '' or dv == '-':
                        continue
                    dk_str = str(dk)
                    dv_str = str(dv)
                    if len(dv_str) > 120:
                        dv_str = dv_str[:120] + '…'
                    row_color = "#1a1a2e"
                    html += f"<tr style='background:{row_color};'>"
                    html += f"<td style='padding:3px 8px; color:#888; white-space:nowrap;'>{dk_str}</td>"
                    html += f"<td style='padding:3px 8px; color:#ddd;'>{dv_str}</td></tr>"
                html += "</table>"
            else:
                html += f"<p style='color:#888;'>无详细评分数据</p>"

            # 因子 z-score
            factor_scores = raw.get('factor_scores') or raw.get('factor_score', {})
            if factor_scores and isinstance(factor_scores, dict):
                html += "<h3 style='color: #00aaff; margin: 12px 0 8px;'>🧮 因子 z-score</h3>"
                html += "<table style='width:100%; border-collapse:collapse; font-size:12px;'>"
                sorted_factors = sorted(factor_scores.items(), key=lambda x: abs(x[1] if x[1] else 0), reverse=True)
                for fk, fv in sorted_factors[:20]:
                    if fv is None or fv == 0:
                        continue
                    fv_f = float(fv)
                    f_color = "#00ff88" if fv_f > 0.5 else ("#ff6666" if fv_f < -0.5 else "#aaa")
                    html += f"<tr><td style='padding:2px 8px; color:#888;'>{fk}</td>"
                    html += f"<td style='padding:2px 8px; color:{f_color};'>{fv_f:+.3f}</td></tr>"
                html += "</table>"

        # 信号理由
        html += f"""
            <h3 style='color: #00aaff; margin: 12px 0 8px;'>💡 信号理由</h3>
            <p style="background: #0d1117; padding: 10px; border-radius: 4px; 
                      border-left: 3px solid #00aaff; color: #ccc; line-height: 1.5;">
                {reason}
            </p>
        </div>
        """
        return html

    def choose_export_dir(self):
        directory = QFileDialog.getExistingDirectory(self, "选择 Excel 导出目录", self.export_dir_edit.text().strip())
        if directory:
            self.export_dir_edit.setText(directory)

    def _default_caption(self) -> str:
        return f"交易对池扫描结果 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

    def export_trade_pool_excel(self):
        if not self.pool_data:
            QMessageBox.warning(self, "提示", "交易对池为空，暂无可导出的结果。")
            return
        try:
            file_path = export_trade_pool_to_excel(self.pool_data, self.export_dir_edit.text().strip())
            self.status_label.setText(f"✅ 已导出 Excel: {file_path}")
            self.save_push_config()
        except Exception as exc:
            QMessageBox.warning(self, "导出失败", str(exc))

    def _telegram_defaults(self):
        if not self.telegram_config_path.exists():
            return {}
        try:
            return json.loads(self.telegram_config_path.read_text(encoding='utf-8'))
        except Exception:
            return {}

    def load_push_config(self):
        telegram_defaults = self._telegram_defaults()
        if telegram_defaults.get("chat_id"):
            self.telegram_chat_id_edit.setText(str(telegram_defaults.get("chat_id", "")))

        if not self.config_path.exists():
            return
        try:
            config = json.loads(self.config_path.read_text(encoding='utf-8'))
        except Exception:
            return

        self.push_method_combo.setCurrentText(config.get("push_method", "Telegram(手机)"))
        self.push_scope_combo.setCurrentText(config.get("push_scope", "精选(最近新增/高评分/共振)"))
        self.push_interval_spin.setValue(int(config.get("push_interval_min", 30)))
        self.recent_minutes_spin.setValue(int(config.get("recent_minutes", 180)))
        self.min_push_score_spin.setValue(int(config.get("min_push_score", 84)))
        self.min_resonance_hits_spin.setValue(int(config.get("min_resonance_hits", 2)))
        self.export_dir_edit.setText(config.get("export_dir", self.export_dir_edit.text()))
        self.telegram_token_edit.setText(config.get("telegram_token", ""))
        self.telegram_chat_id_edit.setText(config.get("telegram_chat_id", self.telegram_chat_id_edit.text()))
        self.server_chan_key_edit.setText(config.get("server_chan_key", ""))
        self.smtp_host_edit.setText(config.get("smtp_host", ""))
        self.smtp_port_spin.setValue(int(config.get("smtp_port", 587)))
        self.smtp_user_edit.setText(config.get("smtp_user", ""))
        self.smtp_password_edit.setText(config.get("smtp_password", ""))
        self.email_to_edit.setText(config.get("email_to", ""))
        self.email_tls_check.setChecked(bool(config.get("email_use_tls", True)))
        if bool(config.get("push_enabled", False)):
            self.start_scheduled_push(silent=True)

    def save_push_config(self):
        config = {
            "push_method": self.push_method_combo.currentText(),
            "push_scope": self.push_scope_combo.currentText(),
            "push_interval_min": self.push_interval_spin.value(),
            "recent_minutes": self.recent_minutes_spin.value(),
            "min_push_score": self.min_push_score_spin.value(),
            "min_resonance_hits": self.min_resonance_hits_spin.value(),
            "export_dir": self.export_dir_edit.text().strip(),
            "telegram_token": self.telegram_token_edit.text().strip(),
            "telegram_chat_id": self.telegram_chat_id_edit.text().strip(),
            "server_chan_key": self.server_chan_key_edit.text().strip(),
            "smtp_host": self.smtp_host_edit.text().strip(),
            "smtp_port": self.smtp_port_spin.value(),
            "smtp_user": self.smtp_user_edit.text().strip(),
            "smtp_password": self.smtp_password_edit.text(),
            "email_to": self.email_to_edit.text().strip(),
            "email_use_tls": self.email_tls_check.isChecked(),
            "push_enabled": self.push_timer.isActive(),
        }
        self.config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding='utf-8')
        self.push_status_label.setText("配置已保存。")

    def start_scheduled_push(self, silent: bool = False):
        self.save_push_config()
        interval_ms = max(self.push_interval_spin.value(), 1) * 60 * 1000
        self.push_timer.start(interval_ms)
        self.save_push_config()
        if not silent:
            self.push_status_label.setText(
                f"定时推送已启动：每 {self.push_interval_spin.value()} 分钟通过 {self.push_method_combo.currentText()} 推送一次"
            )

    def stop_scheduled_push(self):
        self.push_timer.stop()
        self.save_push_config()
        self.push_status_label.setText("当前未启动定时推送。")

    def push_trade_pool_report(self):
        filtered_data = self._build_push_dataset()
        if not filtered_data:
            self.push_status_label.setText("交易对池为空，跳过本次推送。")
            return
        self.save_push_config()
        threading.Thread(target=self._push_trade_pool_report_worker, args=(filtered_data,), daemon=True).start()

    def _push_trade_pool_report_worker(self, report_rows):
        try:
            method = self.push_method_combo.currentText()
            caption = self._default_caption()
            if method == "Telegram(手机)":
                file_path = export_trade_pool_to_excel(report_rows, self.export_dir_edit.text().strip())
                send_telegram_document(
                    self.telegram_token_edit.text().strip(),
                    self.telegram_chat_id_edit.text().strip(),
                    file_path,
                    caption=caption,
                )
                self.push_result_signal.emit("success", f"{method}|{Path(file_path).name}")
            elif method == "微信(Server酱)":
                title = f"交易对池预警: {len(report_rows)} 个品种"
                content = f"### {caption}\n\n"
                for i, row in enumerate(report_rows[:30]): # 最多发送前30个以防消息过长
                    content += f"{i+1}. **{row.get('symbol')}** ({row.get('direction')}) - 评分: {row.get('score', row.get('opportunity_score', 0))}\n"
                if len(report_rows) > 30:
                    content += f"\n... 以及其他 {len(report_rows)-30} 个品种。"
                
                send_server_chan_message(
                    self.server_chan_key_edit.text().strip(),
                    title,
                    content
                )
                self.push_result_signal.emit("success", f"{method}|已发送文字摘要")
            else:
                file_path = export_trade_pool_to_excel(report_rows, self.export_dir_edit.text().strip())
                send_email_with_attachment(
                    smtp_host=self.smtp_host_edit.text().strip(),
                    smtp_port=self.smtp_port_spin.value(),
                    smtp_user=self.smtp_user_edit.text().strip(),
                    smtp_password=self.smtp_password_edit.text(),
                    to_email=self.email_to_edit.text().strip(),
                    file_path=file_path,
                    subject="交易对池扫描结果 Excel",
                    body=f"{caption}\n\n附件为交易对池扫描结果 Excel 文件。",
                    use_tls=self.email_tls_check.isChecked(),
                )
                self.push_result_signal.emit("success", f"{method}|{Path(file_path).name}")
        except Exception as exc:
            self.push_result_signal.emit("error", str(exc))

    def _handle_push_result(self, level: str, payload: str):
        """处理后台推送线程返回结果"""
        if level == "success":
            method, file_name = payload.split("|", 1) if "|" in payload else ("推送", payload)
            self.push_status_label.setText(f"✅ 推送成功：{file_name}")
            self.status_label.setText(f"✅ 已完成 {method} 推送")
        else:
            self.push_status_label.setText(f"❌ 推送失败：{payload}")
            self.status_label.setText(f"❌ 推送失败：{payload}")

    def _build_push_dataset(self):
        scope = self.push_scope_combo.currentText()
        recent_minutes = int(self.recent_minutes_spin.value())
        min_score = float(self.min_push_score_spin.value())
        min_resonance_hits = int(self.min_resonance_hits_spin.value())
        now = datetime.now()

        def is_recent(item):
            try:
                ts = datetime.strptime(str(item.get("time", "")), "%Y-%m-%d %H:%M:%S")
            except Exception:
                return False
            return (now - ts).total_seconds() <= recent_minutes * 60

        def is_high_score(item):
            return float(item.get("opportunity_score", item.get("score", 0)) or 0) >= min_score

        def is_resonance(item):
            return (bool(item.get("is_resonance", False)) or
                    int(item.get("resonance_count", item.get("consensus_engines", 0)) or 0) >= min_resonance_hits)

        if scope == "全量":
            rows = list(self.pool_data)
        elif scope == "最近新增":
            rows = [item for item in self.pool_data if bool(item.get("is_new_signal", False)) or is_recent(item)]
        elif scope == "高评分":
            rows = [item for item in self.pool_data if is_high_score(item)]
        elif scope == "共振":
            rows = [item for item in self.pool_data if is_resonance(item)]
        else:
            rows = [
                item for item in self.pool_data
                if (bool(item.get("is_new_signal", False)) or is_recent(item) or is_high_score(item) or is_resonance(item))
            ]

        rows = sorted(
            rows,
            key=lambda item: (
                float(item.get("opportunity_score", item.get("score", 0)) or 0),
                int(item.get("resonance_count", item.get("hits", 1)) or 1),
                int(item.get("streak_count", 1) or 1),
                item.get("time", ""),
            ),
            reverse=True,
        )
        return rows[: min(len(rows), self.max_pool_size)]

    def remove_item(self, row):
        if 0 <= row < len(self.pool_data):
            item = self.pool_data[row]
            reply = QMessageBox.question(
                self, '确认移除',
                f'确定要移除交易对 {item["symbol"]} 吗？',
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.pool_data.pop(row)
                self.refresh_table()
                self._save_pool_data()
                self.status_label.setText(f"✅ 已移除交易对: {item['symbol']}")
                self.stats_label.setText(f"总计：{len(self.pool_data)} 条记录")

    def clear_pool(self):
        reply = QMessageBox.question(
            self, '确认清空',
            '确定要清空交易对池吗？此操作不可恢复。',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            self.pool_data.clear()
            self._pool_data_raw.clear()
            self.pool_table.setRowCount(0)
            self._save_pool_data()
            self.status_label.setText("🗑️ 交易对池已清空")
            self.stats_label.setText("总计：0 条记录")

    def _save_pool_data(self):
        """保存交易对池数据到文件"""
        try:
            self.pool_data.sort(key=self._timestamp_sort_key, reverse=True)
            data = self.pool_data[:self.max_pool_size]
            # 同步截断 _pool_data_raw（防止内存泄漏）
            keep_keys = {self._make_pool_key(it) for it in data}
            for k in list(self._pool_data_raw.keys()):
                if k not in keep_keys:
                    del self._pool_data_raw[k]
            with open(self._pool_data_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

    def _load_pool_data(self):
        """加载保存的交易对池数据"""
        if not self._pool_data_path.exists():
            return
        try:
            with open(self._pool_data_path, 'r', encoding='utf-8') as f:
                loaded = json.load(f)
            if isinstance(loaded, list):
                self.pool_data = loaded[:self.max_pool_size]
                self.pool_data.sort(key=self._timestamp_sort_key, reverse=True)
                self.refresh_table()
                self.stats_label.setText(f"总计：{len(self.pool_data)} 条记录")
                self.status_label.setText("📂 已恢复上次保存的交易对池")
        except Exception:
            pass
