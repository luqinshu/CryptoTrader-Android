"""
CryptoScanner Pro - Android MVP
Kivy 加密货币扫描器
"""

import os
import sys
import traceback

# Write crash log for adb logcat debugging
def _log(msg):
    try:
        with open('/sdcard/cryptoscanner_crash.log', 'a') as f:
            f.write(msg + '\n')
    except Exception:
        pass

_log("CryptoScanner starting...")
_log(f"Python {sys.version}")
_log(f"sys.path: {sys.path}")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import json
    import threading
    import time
    import glob

    from kivy.app import App
    from kivy.uix.boxlayout import BoxLayout
    from kivy.uix.button import Button
    from kivy.uix.label import Label
    from kivy.uix.textinput import TextInput
    from kivy.uix.scrollview import ScrollView
    from kivy.uix.progressbar import ProgressBar
    from kivy.uix.popup import Popup
    from kivy.uix.spinner import Spinner
    from kivy.clock import Clock
    from kivy.metrics import dp, sp
    from kivy.core.window import Window
    from kivy.graphics import Color, Rectangle

    _log("Kivy imports OK")

    from src.api.okx_client import OKXClient
    _log("OKXClient import OK")

    from src.scanner.base_scanner import ScannerSymbol
    _log("ScannerSymbol import OK")

    from strategies.okx_swing import OKXHourSwingScanner
    _log("OKXHourSwingScanner import OK")

    _imports_ok = True
except Exception as e:
    _log(f"IMPORT ERROR: {e}")
    _log(traceback.format_exc())
    _imports_ok = False
    _import_error = str(e)
    _import_traceback = traceback.format_exc()

P20_WIDTH = 360
P20_HEIGHT = 748
P20_ASPECT_RATIO = 18.7 / 9

FONT_NAME = None
try:
    from kivy.core.text import LabelBase
    FONT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fonts', 'PingFang.ttc')
    if os.path.exists(FONT_PATH):
        LabelBase.register(name='PingFang', fn_regular=FONT_PATH)
        FONT_NAME = 'PingFang'
except Exception:
    pass

Window.clearcolor = (0.12, 0.12, 0.14, 1)


class ResultItem(BoxLayout):
    def __init__(self, result, on_touch_callback=None, **kwargs):
        super().__init__(orientation='vertical', size_hint_y=None, height=dp(80), **kwargs)
        self.result = result
        self._callback = on_touch_callback
        with self.canvas.before:
            Color(0.18, 0.18, 0.22, 1)
            self._bg = Rectangle(pos=self.pos, size=self.size)
        self.bind(pos=self._update_rect, size=self._update_rect)

        score = result.get('score', 0)
        direction = result.get('direction', 'NEUTRAL')
        rating = result.get('rating', '')
        signals = result.get('signals', [])
        symbol = result.get('symbol', '?')

        dir_icon = "+" if direction == "LONG" else ("-" if direction == "SHORT" else "=")
        dir_color = (0.2, 0.8, 0.4, 1) if direction == "LONG" else ((0.9, 0.2, 0.2, 1) if direction == "SHORT" else (0.7, 0.7, 0.7, 1))

        header = BoxLayout(size_hint_y=None, height=dp(28))
        sym_lbl = Label(text=f"[b]{symbol}[/b]", markup=True, color=(1, 1, 1, 1), font_size=sp(15), halign='left', valign='middle')
        if FONT_NAME:
            sym_lbl.font_name = FONT_NAME
        sym_lbl.bind(size=sym_lbl.setter('text_size'))
        dir_lbl = Label(text=f"[b]{dir_icon} {direction}[/b]", markup=True, color=dir_color, font_size=sp(14), halign='center', valign='middle', size_hint_x=0.3)
        if FONT_NAME:
            dir_lbl.font_name = FONT_NAME
        dir_lbl.bind(size=dir_lbl.setter('text_size'))
        score_lbl = Label(text=f"{score:.0f}分", color=(1, 0.85, 0.2, 1), font_size=sp(14), halign='center', valign='middle', size_hint_x=0.25)
        if FONT_NAME:
            score_lbl.font_name = FONT_NAME
        score_lbl.bind(size=score_lbl.setter('text_size'))
        header.add_widget(sym_lbl)
        header.add_widget(dir_lbl)
        header.add_widget(score_lbl)
        self.add_widget(header)

        sig_text = " | ".join(signals[:3]) if signals else "无信号"
        sig_lbl = Label(text=sig_text, color=(0.6, 0.6, 0.65, 1), font_size=sp(11), halign='left', valign='top', size_hint_y=None, height=dp(22))
        if FONT_NAME:
            sig_lbl.font_name = FONT_NAME
        sig_lbl.bind(size=sig_lbl.setter('text_size'))
        self.add_widget(sig_lbl)

        rating_lbl = Label(text=rating, color=(0.5, 0.75, 1, 1), font_size=sp(11), halign='left', valign='middle', size_hint_y=None, height=dp(20))
        if FONT_NAME:
            rating_lbl.font_name = FONT_NAME
        rating_lbl.bind(size=rating_lbl.setter('text_size'))
        self.add_widget(rating_lbl)

    def _update_rect(self, instance, value):
        self._bg.pos = instance.pos
        self._bg.size = instance.size

    def on_touch_down(self, touch):
        if self.collide_point(*touch.pos) and self._callback:
            self._callback(self.result)
            return True
        return super().on_touch_down(touch)


class CryptoScannerApp(App):

    def build(self):
        Window.size = (P20_WIDTH, P20_HEIGHT)
        Window.minimum_width = P20_WIDTH
        Window.minimum_height = P20_HEIGHT
        
        self.okx_client = None
        self.scanner = OKXHourSwingScanner()
        self.is_scanning = False
        self.is_auto_scanning = False
        self._was_auto_scanning = False
        self.auto_scan_timer = None
        self.auto_scan_interval = 600
        self.available_strategies = []
        self._data_dir = os.path.dirname(os.path.abspath(__file__))
        self.config_file = os.path.join(self._data_dir, "scanner_config.json")
        self.saved_config = self._load_config()
        self._has_sensitive_config_warning = self._check_sensitive_config()

        root = BoxLayout(orientation='vertical', padding=dp(8), spacing=dp(4), size_hint=(1, 1))

        title_bar = BoxLayout(size_hint_y=None, height=dp(44))
        with title_bar.canvas.before:
            Color(0.08, 0.08, 0.1, 1)
            self._title_bg = Rectangle(pos=title_bar.pos, size=title_bar.size)
        title_bar.bind(pos=lambda i, v: setattr(self._title_bg, 'pos', v),
                       size=lambda i, v: setattr(self._title_bg, 'size', v))
        title_lbl = Label(text="CryptoScanner Pro", font_size=sp(20), bold=True, color=(0.3, 0.9, 1, 1))
        if FONT_NAME:
            title_lbl.font_name = FONT_NAME
        title_bar.add_widget(title_lbl)
        root.add_widget(title_bar)

        scroll = ScrollView(size_hint=(1, 1))
        self.content = BoxLayout(orientation='vertical', size_hint_x=1, size_hint_y=None, spacing=dp(6))
        self.content.bind(minimum_height=self.content.setter('height'))

        section_lbl = Label(text="API 配置", font_size=sp(14), bold=True, color=(0.5, 0.8, 1, 1), halign='left', valign='middle', size_hint_y=None, height=dp(24))
        if FONT_NAME:
            section_lbl.font_name = FONT_NAME
        section_lbl.bind(size=section_lbl.setter('text_size'))
        self.content.add_widget(section_lbl)

        self.api_key_input = self._make_input("OKX API Key")
        self.content.add_widget(self.api_key_input)

        self.secret_key_input = self._make_input("Secret Key", password=True)
        self.content.add_widget(self.secret_key_input)

        self.passphrase_input = self._make_input("Passphrase", password=True)
        self.content.add_widget(self.passphrase_input)

        self.proxy_input = self._make_input("代理地址（可选）", self.saved_config.get('proxy_url', ''))
        self.content.add_widget(self.proxy_input)

        strategy_section = Label(text="扫描策略", font_size=sp(14), bold=True, color=(0.5, 0.8, 1, 1), halign='left', valign='middle', size_hint_y=None, height=dp(24))
        if FONT_NAME:
            strategy_section.font_name = FONT_NAME
        strategy_section.bind(size=strategy_section.setter('text_size'))
        self.content.add_widget(strategy_section)

        strategy_row = BoxLayout(size_hint_y=None, height=dp(48), spacing=dp(6))
        self.strategy_spinner = Spinner(
            text='选择策略',
            values=self._load_strategies(),
            size_hint_x=0.65,
            background_color=(0.16, 0.16, 0.2, 1),
            color=(1, 1, 1, 1),
        )
        if FONT_NAME:
            self.strategy_spinner.font_name = FONT_NAME
        strategy_row.add_widget(self.strategy_spinner)

        self.load_strategy_btn = Button(
            text="加载策略", font_size=sp(13),
            background_color=(0.25, 0.45, 0.3, 1), color=(0.9, 0.9, 0.95, 1),
            size_hint_x=0.35,
        )
        if FONT_NAME:
            self.load_strategy_btn.font_name = FONT_NAME
        self.load_strategy_btn.bind(on_release=self._on_load_strategy)
        strategy_row.add_widget(self.load_strategy_btn)
        self.content.add_widget(strategy_row)

        test_connection_row = BoxLayout(size_hint_y=None, height=dp(48), spacing=dp(6))
        self.test_connection_btn = Button(
            text="测试连接", font_size=sp(13),
            background_color=(0.4, 0.4, 0.25, 1), color=(0.95, 0.95, 0.95, 1),
            size_hint_x=1.0,
        )
        if FONT_NAME:
            self.test_connection_btn.font_name = FONT_NAME
        self.test_connection_btn.bind(on_release=self._on_test_connection)
        test_connection_row.add_widget(self.test_connection_btn)
        self.content.add_widget(test_connection_row)

        timer_section = Label(text="定时扫描设置", font_size=sp(14), bold=True, color=(0.5, 0.8, 1, 1), halign='left', valign='middle', size_hint_y=None, height=dp(24))
        if FONT_NAME:
            timer_section.font_name = FONT_NAME
        timer_section.bind(size=timer_section.setter('text_size'))
        self.content.add_widget(timer_section)

        timer_row = BoxLayout(size_hint_y=None, height=dp(48), spacing=dp(6))
        self.timer_input = TextInput(
            hint_text="间隔时间（秒）",
            text=str(self.saved_config.get('auto_scan_interval', '600')),
            multiline=False,
            size_hint_x=0.4,
            font_size=sp(13),
            background_color=(0.16, 0.16, 0.2, 1),
            foreground_color=(1, 1, 1, 1),
            input_filter='int',
        )
        if FONT_NAME:
            self.timer_input.font_name = FONT_NAME
        timer_row.add_widget(self.timer_input)

        timer_unit = Label(text="秒", font_size=sp(13), color=(0.6, 0.6, 0.65, 1), size_hint_x=0.15, halign='center', valign='middle')
        if FONT_NAME:
            timer_unit.font_name = FONT_NAME
        timer_row.add_widget(timer_unit)

        self.auto_scan_btn = Button(
            text="开启定时", font_size=sp(13),
            background_color=(0.45, 0.35, 0.25, 1), color=(0.9, 0.9, 0.95, 1),
            size_hint_x=0.45,
        )
        if FONT_NAME:
            self.auto_scan_btn.font_name = FONT_NAME
        self.auto_scan_btn.bind(on_release=self._on_toggle_auto_scan)
        timer_row.add_widget(self.auto_scan_btn)
        self.content.add_widget(timer_row)

        self.progress_bar = ProgressBar(value=0, size_hint_y=None, height=dp(18))
        self.content.add_widget(self.progress_bar)

        self.status_label = Label(
            text="就绪：配置 API 后点击扫描",
            font_size=sp(13), color=(0.6, 0.6, 0.65, 1),
            halign='center', valign='middle', size_hint_y=None, height=dp(30),
        )
        if FONT_NAME:
            self.status_label.font_name = FONT_NAME
        self.status_label.bind(size=self.status_label.setter('text_size'))
        self.content.add_widget(self.status_label)

        result_header = Label(text="扫描结果", font_size=sp(14), bold=True, color=(0.5, 0.8, 1, 1), halign='left', valign='middle', size_hint_y=None, height=dp(24))
        if FONT_NAME:
            result_header.font_name = FONT_NAME
        result_header.bind(size=result_header.setter('text_size'))
        self.content.add_widget(result_header)

        self.result_container = BoxLayout(orientation='vertical', size_hint_y=None, spacing=dp(4))
        self.result_container.bind(minimum_height=self.result_container.setter('height'))
        self.content.add_widget(self.result_container)

        scroll.add_widget(self.content)
        root.add_widget(scroll)

        btn_row = BoxLayout(size_hint_y=None, height=dp(48), spacing=dp(6))
        self.scan_btn = Button(
            text="开始扫描", font_size=sp(15), bold=True,
            background_color=(0.15, 0.55, 0.85, 1), color=(1, 1, 1, 1),
            size_hint_x=0.65,
        )
        if FONT_NAME:
            self.scan_btn.font_name = FONT_NAME
        self.scan_btn.bind(on_release=self._on_scan)
        btn_row.add_widget(self.scan_btn)

        save_btn = Button(
            text="保存配置", font_size=sp(13),
            background_color=(0.25, 0.25, 0.3, 1), color=(0.8, 0.8, 0.85, 1),
            size_hint_x=0.35,
        )
        if FONT_NAME:
            save_btn.font_name = FONT_NAME
        save_btn.bind(on_release=self._on_save_config)
        btn_row.add_widget(save_btn)
        root.add_widget(btn_row)

        if self._has_sensitive_config_warning:
            Clock.schedule_once(lambda dt: self._show_sensitive_config_warning(), 0.5)

        return root

    def _make_input(self, hint, text='', password=False):
        ti = TextInput(
            hint_text=hint, text=text, password=password,
            multiline=False, size_hint_y=None, height=dp(40),
            font_size=sp(13),
            background_color=(0.16, 0.16, 0.2, 1),
            foreground_color=(1, 1, 1, 1),
            cursor_color=(0.3, 0.9, 1, 1),
        )
        if FONT_NAME:
            ti.font_name = FONT_NAME
        return ti

    def _load_strategies(self):
        strategies_dir = os.path.join(self._data_dir, 'strategies')
        self.available_strategies = []
        try:
            py_files = glob.glob(os.path.join(strategies_dir, '*.py'))
            for f in py_files:
                name = os.path.basename(f)
                if not name.startswith('_') and name != '__init__.py' and name != 'okx_swing.py':
                    display_name = name.replace('.py', '')
                    self.available_strategies.append(display_name)
        except Exception:
            pass
        if not self.available_strategies:
            self.available_strategies = ['OKX小时线波段共振策略']
        return self.available_strategies

    def _import_strategy(self, strategy_name):
        import importlib.util
        # Map known strategies to their files
        filename = strategy_name + '.py'
        filepath = os.path.join(self._data_dir, 'strategies', filename)
        if os.path.exists(filepath):
            spec = importlib.util.spec_from_file_location(strategy_name, filepath)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
        raise ImportError(f"Strategy file not found: {filepath}")

    def _load_config(self):
        cfg = {}
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    raw = json.load(f)
                cfg['proxy_url'] = raw.get('proxy_url', '')
                cfg['auto_scan_interval'] = raw.get('auto_scan_interval', '600')
        except Exception:
            pass
        return cfg

    def _check_sensitive_config(self):
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    raw = json.load(f)
                has_keys = bool(raw.get('api_key') or raw.get('secret_key') or raw.get('passphrase'))
                if has_keys:
                    os.remove(self.config_file)
                return has_keys
        except Exception:
            pass
        return False

    def _save_config(self):
        cfg = {
            'proxy_url': self.proxy_input.text,
            'auto_scan_interval': self.timer_input.text,
        }
        try:
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(cfg, f, ensure_ascii=False, indent=2)
            return True
        except Exception:
            return False

    def _on_save_config(self, instance):
        if self._save_config():
            self._show_popup("成功", "配置已保存")
        else:
            self._show_popup("失败", "配置保存失败")

    def _on_test_connection(self, instance):
        """测试 API 连接"""
        self._set_status("正在测试连接...")
        threading.Thread(target=self._test_connection_thread, daemon=True).start()

    def _test_connection_thread(self):
        """后台测试连接线程"""
        try:
            proxy = self.proxy_input.text.strip() or None
            client = OKXClient(
                api_key=self.api_key_input.text,
                secret_key=self.secret_key_input.text,
                passphrase=self.passphrase_input.text,
                testnet=self.saved_config.get('testnet', True),
                proxy_url=proxy,
            )

            ticker_result = client.get_tickers('SWAP')
            if isinstance(ticker_result, dict) and ticker_result.get('code') == '0':
                data = ticker_result.get('data', [])
                Clock.schedule_once(lambda dt: self._show_popup("连接成功", f"获取到 {len(data)} 个交易对"))
                Clock.schedule_once(lambda dt: self._set_status("API 连接正常 ✓"))
            else:
                msg = ticker_result.get('msg', '未知错误')
                Clock.schedule_once(lambda dt: self._show_popup("连接失败", f"{msg}"))
                Clock.schedule_once(lambda dt: self._set_status(f"API 连接失败: {msg}"))
        except Exception as e:
            Clock.schedule_once(lambda dt: self._show_popup("连接错误", str(e)))
            Clock.schedule_once(lambda dt: self._set_status(f"连接错误: {str(e)}"))

    def _on_load_strategy(self, instance):
        strategy_name = self.strategy_spinner.text
        if strategy_name == '选择策略' or not strategy_name:
            self._show_popup("提示", "请选择一个策略")
            return
        
        try:
            module = self._import_strategy(strategy_name)
            
            scanner_class = None
            possible_class_names = [
                strategy_name + 'Scanner',
                'XiaoYueBollMacdScanner',
                'OKXHourSwingScanner'
            ]
            
            for class_name in possible_class_names:
                if hasattr(module, class_name):
                    scanner_class = getattr(module, class_name)
                    break
            
            if scanner_class:
                self.scanner = scanner_class()
                self._set_status(f"策略已加载: {strategy_name}")
                self._show_popup("成功", f"已加载策略: {strategy_name}")
            else:
                self._show_popup("失败", f"无法找到策略类")
        except Exception as e:
            self._show_popup("错误", f"加载策略失败: {str(e)}")

    def _on_toggle_auto_scan(self, instance):
        if self.is_auto_scanning:
            self._stop_auto_scan()
        else:
            self._start_auto_scan()

    def _start_auto_scan(self):
        try:
            interval = int(self.timer_input.text)
            if interval < 60:
                self._show_popup("提示", "最小间隔时间为 60 秒")
                return
            
            self.auto_scan_interval = interval
            self.is_auto_scanning = True
            self.auto_scan_btn.text = "停止定时"
            self.auto_scan_btn.background_color = (0.6, 0.25, 0.25, 1)
            self._set_status(f"定时扫描已开启，间隔 {interval} 秒")
            
            self._schedule_next_scan()
        except ValueError:
            self._show_popup("提示", "请输入有效的数字")

    def _stop_auto_scan(self):
        self.is_auto_scanning = False
        if self.auto_scan_timer:
            self.auto_scan_timer.cancel()
            self.auto_scan_timer = None
        self.auto_scan_btn.text = "开启定时"
        self.auto_scan_btn.background_color = (0.45, 0.35, 0.25, 1)
        self._set_status("定时扫描已停止")

    def _schedule_next_scan(self):
        if not self.is_auto_scanning:
            return
        
        self.auto_scan_timer = threading.Timer(self.auto_scan_interval, self._auto_scan_trigger)
        self.auto_scan_timer.start()

    def _auto_scan_trigger(self):
        if self.is_auto_scanning and not self.is_scanning:
            self._on_scan(None)
        
        self._schedule_next_scan()

    def _on_scan(self, instance):
        if self.is_scanning:
            return
        if not self.api_key_input.text or not self.secret_key_input.text:
            self._show_popup("提示", "请先填写 API Key 和 Secret Key")
            return

        self._save_config()
        self.is_scanning = True
        self.scan_btn.disabled = True
        self.scan_btn.text = "扫描中..."
        self.scan_btn.background_color = (0.5, 0.5, 0.5, 1)
        self.result_container.clear_widgets()
        self.progress_bar.value = 0
        self._set_status("正在连接 OKX...")

        threading.Thread(target=self._run_scan, daemon=True).start()

    def _init_client(self):
        proxy = self.proxy_input.text.strip() or None
        return OKXClient(
            api_key=self.api_key_input.text,
            secret_key=self.secret_key_input.text,
            passphrase=self.passphrase_input.text,
            testnet=True,
            proxy_url=proxy,
        )

    def _run_scan(self):
        try:
            if not self.okx_client:
                self.okx_client = self._init_client()

            self._set_status("获取行情数据...")
            self._set_progress(5)

            tickers_res = self.okx_client.get_tickers('SWAP')
            if not isinstance(tickers_res, dict) or tickers_res.get('code') != '0':
                msg = tickers_res.get('msg', '未知错误') if isinstance(tickers_res, dict) else '网络连接失败'
                self._show_error(f"API 错误: {msg}")
                return

            tickers = tickers_res.get('data', [])
            usdt_swaps = [t for t in tickers if t.get('instId', '').endswith('-USDT-SWAP')]
            active = [t for t in usdt_swaps if float(t.get('volCcyQuote') or t.get('vol24h') or 0) > 5000000]
            active = sorted(active, key=lambda t: float(t.get('volCcyQuote') or t.get('vol24h') or 0), reverse=True)
            active = active[:30]

            total = len(active)
            self._set_status(f"找到 {total} 个活跃品种，开始分析...")
            self._set_progress(10)

            found = 0
            for i, t in enumerate(active):
                if not self.is_scanning:
                    break

                inst_id = t['instId']
                pct = 10 + int(85 * (i + 1) / total)
                self._set_progress(pct)
                self._set_status(f"[{i+1}/{total}] {inst_id}")

                try:
                    klines = {}
                    for bar in ['1D', '1H', '15m', '3m']:
                        res = self.okx_client.get_kline(inst_id, bar=bar, limit=200)
                        if isinstance(res, dict) and res.get('code') == '0' and res.get('data'):
                            klines[bar] = res['data']

                    if not klines.get('1D') or not klines.get('1H'):
                        continue

                    symbol = ScannerSymbol(
                        inst_id=inst_id,
                        last_price=float(t.get('last', 0)),
                        volume_24h=float(t.get('volCcyQuote') or t.get('vol24h') or 0),
                        extra_data={'klines': klines},
                    )

                    result = self.scanner.scan_symbol(symbol)

                    if result.get('passed', False) or result.get('score', 0) >= 60:
                        found += 1
                        self._add_result(result)

                except Exception as e:
                    print(f"分析 {inst_id} 失败: {e}")
                    continue

                time.sleep(0.15)

            self._set_status(f"扫描完成！发现 {found} 个交易机会")
            self._set_progress(100)

        except Exception as e:
            err_msg = str(e)
            if 'Connection' in err_msg or 'Timeout' in err_msg or 'timeout' in err_msg.lower():
                self._show_error("网络连接失败，请检查网络后重试")
            elif 'Proxy' in err_msg or 'proxy' in err_msg.lower():
                self._show_error(f"代理连接失败: {err_msg}")
            else:
                self._show_error(f"扫描失败: {err_msg}")
        finally:
            self.is_scanning = False
            Clock.schedule_once(lambda dt: self._reset_btn())

    def _reset_btn(self):
        self.scan_btn.disabled = False
        self.scan_btn.text = "开始扫描"
        self.scan_btn.background_color = (0.15, 0.55, 0.85, 1)

    def _set_status(self, text):
        Clock.schedule_once(lambda dt: setattr(self.status_label, 'text', text))

    def _set_progress(self, value):
        Clock.schedule_once(lambda dt: setattr(self.progress_bar, 'value', value))

    def _add_result(self, res):
        def _add(dt):
            item = ResultItem(res, on_touch_callback=self._show_detail)
            self.result_container.add_widget(item)
        Clock.schedule_once(_add)

    def _show_detail(self, res):
        risk = res.get('risk_management', {})
        signals = res.get('signals', [])
        detail = (
            f"交易对: {res.get('symbol', '?')}\n"
            f"评分: {res.get('score', 0):.0f}\n"
            f"方向: {res.get('direction', 'NEUTRAL')}\n"
            f"评级: {res.get('rating', '-')}\n"
            f"当前价: {res.get('last_price', 'N/A')}\n\n"
            f"止损: {risk.get('stop_loss', 'N/A')}\n"
            f"止盈: {risk.get('take_profit', 'N/A')}\n"
            f"止损距离: {risk.get('sl_distance_pct', 'N/A')}%\n"
            f"盈亏比: 1:{risk.get('rr_ratio', 'N/A')}\n\n"
            f"信号:\n" + "\n".join([f"  - {s}" for s in signals])
        )
        self._show_popup(f"{res.get('symbol', '?')} 详情", detail)

    def _show_popup(self, title, text):
        content = BoxLayout(orientation='vertical', padding=dp(10), spacing=dp(8), size_hint=(1, 1))
        lbl = Label(text=text, font_size=sp(12), halign='left', valign='top', color=(1, 1, 1, 1), size_hint_y=1)
        if FONT_NAME:
            lbl.font_name = FONT_NAME
        lbl.bind(size=lbl.setter('text_size'))
        content.add_widget(lbl)
        close_btn = Button(text="关闭", font_name=FONT_NAME if FONT_NAME else None, size_hint_y=None, height=dp(36), background_color=(0.3, 0.3, 0.35, 1))
        content.add_widget(close_btn)

        popup = Popup(
            title=title, content=content,
            size_hint=(0.9, 0.7),
            background_color=(0.15, 0.15, 0.18, 0.95),
            separator_color=(0.3, 0.9, 1, 1),
        )
        close_btn.bind(on_release=popup.dismiss)
        popup.open()

    def _show_error(self, msg):
        def _show(dt):
            self._show_popup("错误", msg)
            self._set_status(msg)
        Clock.schedule_once(_show)

    def _show_sensitive_config_warning(self):
        warning_text = (
            "检测到旧版配置文件包含 API 凭证明文，已自动删除。\n\n"
            "请立即去 OKX 官网控制台删除该 API Key 并创建新 Key：\n"
            "https://www.okx.com/account/my-api\n\n"
            "之后在本 App 的 API 配置区手动输入新的 Key。\n"
            "凭据仅存内存，关闭 App 即自动清除。"
        )
        content = BoxLayout(orientation='vertical', padding=dp(12), spacing=dp(8), size_hint=(1, 1))
        lbl = Label(text=warning_text, font_size=sp(12), halign='left', valign='top', color=(1, 0.85, 0.3, 1))
        if FONT_NAME:
            lbl.font_name = FONT_NAME
        lbl.bind(size=lbl.setter('text_size'))
        content.add_widget(lbl)
        close_btn = Button(text="我知道了", font_name=FONT_NAME if FONT_NAME else None, size_hint_y=None, height=dp(40), background_color=(0.8, 0.4, 0.2, 1))
        content.add_widget(close_btn)
        popup = Popup(
            title="安全提示", content=content,
            size_hint=(0.9, 0.5),
            background_color=(0.15, 0.15, 0.18, 0.95),
            separator_color=(0.9, 0.5, 0.2, 1),
        )
        close_btn.bind(on_release=popup.dismiss)
        popup.open()

    def on_pause(self):
        self._was_auto_scanning = self.is_auto_scanning
        if self.is_auto_scanning:
            self._stop_auto_scan()
        return True

    def on_resume(self):
        if getattr(self, '_was_auto_scanning', False):
            Clock.schedule_once(lambda dt: self._start_auto_scan(), 1)

    def on_stop(self):
        self._stop_auto_scan()


if __name__ == "__main__":
    if not _imports_ok:
        from kivy.app import App
        from kivy.uix.label import Label
        from kivy.uix.boxlayout import BoxLayout
        from kivy.uix.button import Button
        from kivy.uix.scrollview import ScrollView
        from kivy.uix.popup import Popup
        from kivy.metrics import dp, sp
        from kivy.core.window import Window

        class ErrorApp(App):
            def build(self):
                Window.size = (360, 748)
                root = BoxLayout(orientation='vertical', padding=dp(20), spacing=dp(10))
                lbl = Label(
                    text=f"启动失败\n\n{_import_error}\n\n请检查 /sdcard/cryptoscanner_crash.log",
                    font_size=sp(12), color=(1, 0.3, 0.3, 1), halign='left', valign='top'
                )
                lbl.bind(size=lbl.setter('text_size'))
                scroll = ScrollView()
                scroll.add_widget(lbl)
                root.add_widget(scroll)
                btn = Button(text="退出", size_hint_y=None, height=dp(40),
                           background_color=(0.6, 0.2, 0.2, 1))
                btn.bind(on_release=lambda x: sys.exit(0))
                root.add_widget(btn)
                return root

        ErrorApp().run()
    else:
        CryptoScannerApp().run()
