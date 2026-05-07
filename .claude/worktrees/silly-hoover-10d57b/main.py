import os
import threading
import pandas as pd
import numpy as np
import ta
from datetime import datetime

# Kivy & KivyMD 核心组件
from kivymd.app import MDApp
from kivymd.uix.screen import MDScreen
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.button import MDRaisedButton, MDIconButton
from kivymd.uix.label import MDLabel
from kivymd.uix.toolbar import MDTopAppBar
from kivymd.uix.list import ThreeLineListItem, MDList
from kivymd.uix.scrollview import MDScrollView
from kivymd.uix.selectioncontrol import MDCheckbox
from kivymd.uix.textfield import MDTextField
from kivymd.uix.datatables import MDDataTable
from kivy.metrics import dp
from kivy.clock import Clock

# 导入您现有的 OKX 逻辑 (需确保 okx_client.py 也在打包目录)
from src.api.okx_client import OKXClient
from strategies.OKX小时线波段共振策略 import OKXHourSwingScanner
from src.scanner.base_scanner import ScannerSymbol

class MobileScannerApp(MDApp):
    def build(self):
        self.theme_cls.theme_style = "Dark"
        self.theme_cls.primary_palette = "Cyan"
        
        # 初始数据
        self.okx_client = None
        self.scanner = OKXHourSwingScanner()
        self.is_scanning = False
        
        # 构建主界面
        screen = MDScreen()
        layout = MDBoxLayout(orientation='vertical')
        
        # 1. 顶部工具栏
        toolbar = MDTopAppBar(title="Crypto Scanner Pro")
        layout.addWidget(toolbar)
        
        # 2. 配置区域 (API & 代理)
        config_box = MDBoxLayout(orientation='vertical', padding=dp(10), spacing=dp(5), adaptive_height=True)
        self.api_key_input = MDTextField(hint_text="OKX API Key", text="ddafb223-6fe7-4ada-94f6-a31d58b23e1a")
        self.secret_key_input = MDTextField(hint_text="Secret Key", password=True, text="C05E005B0B94EB17E44739C7302605C9")
        self.pass_input = MDTextField(hint_text="Passphrase", text="!Lqs4381525")
        
        config_box.add_widget(self.api_key_input)
        config_box.add_widget(self.secret_key_input)
        config_box.add_widget(self.pass_input)
        layout.addWidget(config_box)
        
        # 3. 状态显示
        self.status_label = MDLabel(
            text="就绪: 点击下方按钮开始全市场扫描",
            halign="center",
            theme_text_color="Hint",
            size_hint_y=None,
            height=dp(40)
        )
        layout.addWidget(self.status_label)
        
        # 4. 扫描结果列表
        scroll = MDScrollView()
        self.result_list = MDList()
        scroll.add_widget(self.result_list)
        layout.addWidget(scroll)
        
        # 5. 底部控制按钮
        self.scan_btn = MDRaisedButton(
            text="🚀 开始小时线波段扫描",
            pos_hint={'center_x': .5},
            size_hint_x=0.9,
            padding=dp(20),
            on_release=self.start_scan_thread
        )
        layout.addWidget(self.scan_btn)
        layout.add_widget(MDBoxLayout(size_hint_y=None, height=dp(10)))
        
        screen.add_widget(layout)
        return screen

    def start_scan_thread(self, instance):
        if self.is_scanning: return
        
        self.is_scanning = True
        self.scan_btn.disabled = True
        self.result_list.clear_widgets()
        self.status_label.text = "正在连接 OKX 获取行情..."
        
        # 开启后台线程执行扫描，防止安卓界面卡死
        threading.Thread(target=self.run_scanner_logic, daemon=True).start()

    def run_scanner_logic(self):
        try:
            # 1. 初始化客户端
            if not self.okx_client:
                self.okx_client = OKXClient(
                    api_key=self.api_key_input.text,
                    secret_key=self.secret_key_input.text,
                    passphrase=self.pass_input.text,
                    testnet=True,
                    proxy_url="http://127.0.0.1:7897" # 注意：手机端通常需配置正确代理或不使用代理
                )
            
            # 2. 获取行情
            tickers_res = self.okx_client.get_tickers('SWAP')
            tickers = tickers_res.get('data', []) if isinstance(tickers_res, dict) else tickers_res
            
            # 过滤高活跃
            active_tickers = [t for t in tickers if float(t.get('volCcy24h', 0)) > 5000000][:50] # 手机端限制前50个以节省流量
            
            total = len(active_tickers)
            for i, t in enumerate(active_tickers):
                inst_id = t['instId']
                self.update_status(f"分析中 [{i+1}/{total}]: {inst_id}")
                
                try:
                    # 获取多周期数据
                    klines = {
                        '1D': self.okx_client.get_kline(inst_id, '1D', limit=200),
                        '1H': self.okx_client.get_kline(inst_id, '1H', limit=100),
                        '3m': self.okx_client.get_kline(inst_id, '3m', limit=50)
                    }
                    
                    symbol_obj = ScannerSymbol(
                        inst_id=inst_id,
                        last_price=float(t['last']),
                        volume_24h=float(t['volCcy24h']),
                        extra_data={'klines': klines}
                    )
                    
                    # 执行策略逻辑
                    res = self.scanner.scan_symbol(symbol_obj)
                    
                    # 如果得分 > 60，显示在界面上
                    if res.get('score', 0) >= 60:
                        self.add_result_item(res)
                        
                except: continue
                
            self.update_status(f"✅ 扫描完成，发现 {len(self.result_list.children)} 个机会")
            
        except Exception as e:
            self.update_status(f"❌ 错误: {str(e)}")
        finally:
            self.is_scanning = False
            Clock.schedule_once(lambda dt: setattr(self.scan_btn, 'disabled', False))

    def update_status(self, text):
        Clock.schedule_once(lambda dt: setattr(self.status_label, 'text', text))

    def add_result_item(self, res):
        def add_to_ui(dt):
            item = ThreeLineListItem(
                text=f"{res['symbol']} - 评分: {res['score']:.1f}",
                secondary_text=f"方向: {res['direction']} | 评级: {res['rating']}",
                tertiary_text=f"信号: {' | '.join(res['signals'][:2])}",
                on_release=lambda x: self.show_detail(res)
            )
            self.result_list.add_widget(item)
        Clock.schedule_once(add_to_ui)

    def show_detail(self, res):
        # 手机端点击可弹出详细止盈止损建议
        from kivymd.uix.dialog import MDDialog
        risk = res.get('risk_management', {})
        msg = f"止损: {risk.get('stop_loss')}\n止盈: {risk.get('take_profit')}\n盈亏比: {risk.get('rr_ratio')}\n\n全部信号:\n" + "\n".join(res['signals'])
        
        dialog = MDDialog(title=f"{res['symbol']} 详情", text=msg)
        dialog.open()

if __name__ == "__main__":
    MobileScannerApp().run()
