"""
Crypto Scanner Pro - Android 独立版本
专为Android打包优化的交易对扫描应用
"""

import os
import json
import threading
import pandas as pd
import numpy as np
import ta
from datetime import datetime

# Kivy & KivyMD 核心组件
from kivymd.app import MDApp
from kivymd.uix.screen import MDScreen
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.button import MDRaisedButton
from kivymd.uix.label import MDLabel
from kivymd.uix.toolbar import MDTopAppBar
from kivymd.uix.list import ThreeLineListItem, MDList
from kivymd.uix.scrollview import MDScrollView
from kivymd.uix.textfield import MDTextField
from kivymd.uix.dialog import MDDialog
from kivy.metrics import dp
from kivy.clock import Clock
from kivy.storage.jsonstore import JsonStore

# 导入核心逻辑
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.api.okx_client import OKXClient
from strategies.OKX小时线波段共振策略 import OKXHourSwingScanner
from src.scanner.base_scanner import ScannerSymbol


class AndroidScannerApp(MDApp):
    """Android 扫描器应用"""
    
    def build(self):
        self.theme_cls.theme_style = "Dark"
        self.theme_cls.primary_palette = "Cyan"
        
        # 初始化数据
        self.okx_client = None
        self.scanner = OKXHourSwingScanner()
        self.is_scanning = False
        self.config_file = "scanner_config.json"
        
        # 加载保存的配置
        self.load_config()
        
        # 构建主界面
        screen = MDScreen()
        main_layout = MDBoxLayout(
            orientation='vertical',
            padding=dp(10),
            spacing=dp(10)
        )
        
        # 1. 顶部工具栏
        toolbar = MDTopAppBar(
            title="📊 Crypto Scanner Pro",
            pos_hint={"top": 1},
            elevation=4
        )
        main_layout.add_widget(toolbar)
        
        # 2. 滚动视图（包含所有内容）
        scroll = MDScrollView()
        content_layout = MDBoxLayout(
            orientation='vertical',
            padding=dp(10),
            spacing=dp(15),
            size_hint_y=None,
            adaptive_height=True
        )
        
        # API 配置区域
        config_box = MDBoxLayout(
            orientation='vertical',
            spacing=dp(10),
            size_hint_y=None,
            adaptive_height=True
        )
        
        config_title = MDLabel(
            text="⚙️ API 配置（首次使用需填写）",
            font_style="H6",
            theme_text_color="Primary",
            size_hint_y=None,
            height=dp(30)
        )
        config_box.add_widget(config_title)
        
        self.api_key_input = MDTextField(
            hint_text="OKX API Key",
            text=self.saved_config.get('api_key', ''),
            mode="rectangle",
            size_hint_y=None,
            height=dp(40)
        )
        config_box.add_widget(self.api_key_input)
        
        self.secret_key_input = MDTextField(
            hint_text="Secret Key",
            password=True,
            text=self.saved_config.get('secret_key', ''),
            mode="rectangle",
            size_hint_y=None,
            height=dp(40)
        )
        config_box.add_widget(self.secret_key_input)
        
        self.pass_input = MDTextField(
            hint_text="Passphrase",
            password=True,
            text=self.saved_config.get('passphrase', ''),
            mode="rectangle",
            size_hint_y=None,
            height=dp(40)
        )
        config_box.add_widget(self.pass_input)
        
        # 代理开关
        self.proxy_hint = MDLabel(
            text="💡 提示：国内用户可能需要配置代理",
            theme_text_color="Hint",
            size_hint_y=None,
            height=dp(25)
        )
        config_box.add_widget(self.proxy_hint)
        
        content_layout.add_widget(config_box)
        
        # 3. 状态显示
        self.status_label = MDLabel(
            text="✅ 就绪：点击下方按钮开始扫描",
            halign="center",
            font_style="Body1",
            theme_text_color="Secondary",
            size_hint_y=None,
            height=dp(40)
        )
        content_layout.add_widget(self.status_label)
        
        # 4. 进度条
        self.progress_label = MDLabel(
            text="",
            halign="center",
            theme_text_color="Hint",
            size_hint_y=None,
            height=dp(30)
        )
        content_layout.add_widget(self.progress_label)
        
        # 5. 扫描结果列表
        result_title = MDLabel(
            text="📈 扫描结果",
            font_style="H6",
            theme_text_color="Primary",
            size_hint_y=None,
            height=dp(30)
        )
        content_layout.add_widget(result_title)
        
        self.result_list = MDList()
        content_layout.add_widget(self.result_list)
        
        scroll.add_widget(content_layout)
        main_layout.add_widget(scroll)
        
        # 6. 底部控制按钮
        btn_layout = MDBoxLayout(
            orientation='horizontal',
            spacing=dp(10),
            size_hint_y=None,
            height=dp(50),
            padding=dp(10)
        )
        
        self.scan_btn = MDRaisedButton(
            text="🚀 开始扫描",
            size_hint_x=0.7,
            on_release=self.start_scan_thread
        )
        btn_layout.add_widget(self.scan_btn)
        
        save_btn = MDRaisedButton(
            text="💾 保存配置",
            size_hint_x=0.3,
            md_bg_color=self.theme_cls.primaryColor,
            on_release=self.save_config_manual
        )
        btn_layout.add_widget(save_btn)
        
        main_layout.add_widget(btn_layout)
        
        screen.add_widget(main_layout)
        return screen
    
    def load_config(self):
        """加载本地配置"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    self.saved_config = json.load(f)
            else:
                self.saved_config = {}
        except:
            self.saved_config = {}
    
    def save_config(self):
        """保存配置到本地"""
        config = {
            'api_key': self.api_key_input.text,
            'secret_key': self.secret_key_input.text,
            'passphrase': self.pass_input.text
        }
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
            return True
        except:
            return False
    
    def save_config_manual(self, instance):
        """手动保存配置"""
        if self.save_config():
            self.show_dialog("✅ 成功", "配置已保存到本地")
        else:
            self.show_dialog("❌ 失败", "配置保存失败")
    
    def start_scan_thread(self, instance):
        """启动扫描线程"""
        if self.is_scanning:
            return
        
        # 验证输入
        if not self.api_key_input.text or not self.secret_key_input.text:
            self.show_dialog("⚠️ 提示", "请先填写 API 配置")
            return
        
        # 保存配置
        self.save_config()
        
        self.is_scanning = True
        self.scan_btn.disabled = True
        self.scan_btn.text = "⏳ 扫描中..."
        self.result_list.clear_widgets()
        self.status_label.text = "🔄 正在连接 OKX..."
        self.progress_label.text = ""
        
        # 开启后台线程
        threading.Thread(target=self.run_scanner_logic, daemon=True).start()
    
    def run_scanner_logic(self):
        """执行扫描逻辑"""
        try:
            # 1. 初始化客户端（Android 环境不使用代理）
            if not self.okx_client:
                self.okx_client = OKXClient(
                    api_key=self.api_key_input.text,
                    secret_key=self.secret_key_input.text,
                    passphrase=self.pass_input.text,
                    testnet=False,  # 使用正式网
                    proxy_url=None  # Android 不使用代理
                )
            
            self.update_status("📡 正在获取交易对列表...")
            
            # 2. 获取永续合约行情
            tickers_res = self.okx_client.get_tickers('SWAP')
            
            if not isinstance(tickers_res, dict) or tickers_res.get('code') != '0':
                self.update_status(f"❌ API错误: {tickers_res}")
                return
            
            tickers = tickers_res.get('data', [])
            
            # 过滤高活跃度交易对
            active_tickers = [
                t for t in tickers 
                if float(t.get('volCcy24h', 0)) > 5000000
            ][:30]  # Android 限制前30个
            
            total = len(active_tickers)
            self.update_status(f"✅ 找到 {total} 个高活跃度交易对")
            
            # 3. 逐个分析
            found_count = 0
            for i, t in enumerate(active_tickers):
                inst_id = t['instId']
                self.update_progress(f"分析中 [{i+1}/{total}]: {inst_id}")
                
                try:
                    # 获取多周期K线数据
                    klines = {
                        '1D': self.okx_client.get_kline(inst_id, '1D', limit=200),
                        '1H': self.okx_client.get_kline(inst_id, '1H', limit=100),
                        '3m': self.okx_client.get_kline(inst_id, '3m', limit=50)
                    }
                    
                    # 检查数据有效性
                    valid_data = True
                    for timeframe, data in klines.items():
                        if not isinstance(data, dict) or data.get('code') != '0':
                            valid_data = False
                            break
                        if not data.get('data') or len(data.get('data', [])) < 30:
                            valid_data = False
                            break
                    
                    if not valid_data:
                        continue
                    
                    # 构建交易对对象
                    symbol_obj = ScannerSymbol(
                        inst_id=inst_id,
                        last_price=float(t['last']),
                        volume_24h=float(t['volCcy24h']),
                        extra_data={'klines': klines}
                    )
                    
                    # 执行策略扫描
                    res = self.scanner.scan_symbol(symbol_obj)
                    
                    # 如果评分 >= 70，显示结果
                    if res.get('score', 0) >= 70:
                        found_count += 1
                        self.add_result_item(res)
                
                except Exception as e:
                    print(f"分析 {inst_id} 失败: {e}")
                    continue
            
            self.update_status(f"✅ 扫描完成！发现 {found_count} 个交易机会")
            self.update_progress("")
            
        except Exception as e:
            self.update_status(f"❌ 扫描失败: {str(e)}")
            import traceback
            print(traceback.format_exc())
        
        finally:
            self.is_scanning = False
            Clock.schedule_once(lambda dt: self.reset_scan_button())
    
    def reset_scan_button(self):
        """重置扫描按钮"""
        self.scan_btn.disabled = False
        self.scan_btn.text = "🚀 开始扫描"
    
    def update_status(self, text):
        """更新状态文本（线程安全）"""
        Clock.schedule_once(lambda dt: setattr(self.status_label, 'text', text))
    
    def update_progress(self, text):
        """更新进度文本（线程安全）"""
        Clock.schedule_once(lambda dt: setattr(self.progress_label, 'text', text))
    
    def add_result_item(self, res):
        """添加结果项（线程安全）"""
        def add_to_ui(dt):
            score = res.get('score', 0)
            direction = res.get('direction', 'NEUTRAL')
            rating = res.get('rating', '普通')
            signals = res.get('signals', [])
            
            # 根据方向设置颜色
            if direction == "LONG":
                direction_icon = "📈"
            elif direction == "SHORT":
                direction_icon = "📉"
            else:
                direction_icon = "⚖️"
            
            item_text = f"{res['symbol']} | 评分: {score:.0f}"
            
            item = ThreeLineListItem(
                text=item_text,
                secondary_text=f"{direction_icon} {direction} | {rating}",
                tertiary_text=f"信号: {' | '.join(signals[:2])}" if signals else "信号: 无",
                on_release=lambda x: self.show_detail(res)
            )
            self.result_list.add_widget(item)
        
        Clock.schedule_once(add_to_ui)
    
    def show_detail(self, res):
        """显示详细信息"""
        risk = res.get('risk_management', {})
        signals = res.get('signals', [])
        
        detail_text = f"""
📊 交易对: {res['symbol']}
💯 评分: {res.get('score', 0):.0f}
🎯 方向: {res.get('direction', 'NEUTRAL')}
⭐ 评级: {res.get('rating', '普通')}

💰 当前价格: {res.get('last_price', 0)}

🛡️ 风险管理:
  • 止损: {risk.get('stop_loss', 'N/A')}
  • 止盈: {risk.get('take_profit', 'N/A')}
  • 止损距离: {risk.get('sl_distance_pct', 'N/A')}%
  • 盈亏比: 1:{risk.get('rr_ratio', 'N/A')}

📡 信号列表:
{chr(10).join([f"  • {s}" for s in signals])}
        """
        
        self.show_dialog(f"{res['symbol']} 详情", detail_text)
    
    def show_dialog(self, title, text):
        """显示对话框"""
        def close_dialog(dialog, *args):
            dialog.dismiss()
        
        dialog = MDDialog(
            title=title,
            text=text,
            buttons=[
                MDRaisedButton(
                    text="关闭",
                    on_release=close_dialog
                )
            ]
        )
        dialog.open()


if __name__ == "__main__":
    AndroidScannerApp().run()
