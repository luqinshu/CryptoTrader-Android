"""
CryptoScanner Pro - Android Full Version
Kivy multi-tab trading platform
"""
import os, sys, json, threading, time, glob, traceback, importlib.util

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── crash log ──────────────────────────────────────────────
def _crash(msg):
    try:
        with open('/sdcard/cs_crash.log', 'a') as f:
            f.write(msg + '\n')
    except Exception:
        pass

_crash("App starting")
_crash(f"Python {sys.version}")

# ── imports ────────────────────────────────────────────────
try:
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

    _crash("Kivy imports OK")

    from src.api.okx_client import OKXClient
    from src.scanner.base_scanner import ScannerSymbol
    from strategies.okx_swing import OKXHourSwingScanner

    _crash("App imports OK")
    _IMPORTS_OK = True
except Exception as e:
    _crash(f"Import error: {e}\n{traceback.format_exc()}")
    _IMPORTS_OK = False
    _IMPORT_ERR = str(e)

# ── font setup ─────────────────────────────────────────────
from kivy.core.text import LabelBase
FONT_NAME = None
_font_paths = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fonts', 'PingFang.ttc'),
    '/system/fonts/NotoSansCJK-Regular.ttc',
    '/system/fonts/DroidSansFallback.ttf',
]
for fp in _font_paths:
    try:
        if os.path.exists(fp):
            LabelBase.register(name='CNFont', fn_regular=fp)
            FONT_NAME = 'CNFont'
            _crash(f"Font loaded: {fp}")
            break
    except Exception:
        pass
if not FONT_NAME:
    _crash("No Chinese font found, using default")

Window.clearcolor = (0.10, 0.10, 0.13, 1)

# ── colors ─────────────────────────────────────────────────
C_BG   = (0.10, 0.10, 0.13, 1)
C_CARD = (0.16, 0.16, 0.20, 1)
C_BTN  = (0.18, 0.50, 0.80, 1)
C_ACC  = (0.20, 0.80, 0.40, 1)
C_WARN = (0.90, 0.60, 0.10, 1)
C_RED  = (0.85, 0.20, 0.20, 1)
C_TEXT = (0.90, 0.90, 0.95, 1)
C_SUB  = (0.55, 0.55, 0.60, 1)
C_TAB_ACTIVE = (0.18, 0.55, 0.85, 1)
C_TAB_INACTIVE = (0.12, 0.12, 0.16, 1)

def _font(widget):
    if FONT_NAME:
        widget.font_name = FONT_NAME

def _lbl(text, size=13, color=C_TEXT, bold=False, halign='left'):
    lbl = Label(text=text, font_size=sp(size), color=color, bold=bold, halign=halign, valign='middle')
    lbl.bind(size=lbl.setter('text_size'))
    _font(lbl)
    return lbl

def _btn(text, bg=C_BTN, size=13, bold=False):
    btn = Button(text=text, font_size=sp(size), bold=bold,
                 background_color=bg, color=C_TEXT,
                 background_normal='')
    _font(btn)
    return btn

def _input(hint='', text='', pw=False):
    ti = TextInput(hint_text=hint, text=text, password=pw,
                   multiline=False, font_size=sp(13),
                   background_color=C_CARD, foreground_color=C_TEXT,
                   cursor_color=C_TAB_ACTIVE, padding=[dp(8), dp(8)])
    _font(ti)
    return ti


class CryptoApp(App):
    def build(self):
        Window.size = (360, 748)
        self._data_dir = os.path.dirname(os.path.abspath(__file__))
        self.config_file = os.path.join(self._data_dir, "scanner_config.json")
        self._cfg = self._load_cfg()
        self.okx_client = None
        self.scanner = OKXHourSwingScanner()
        self.is_scanning = False

        self.root = BoxLayout(orientation='vertical')

        # ── title bar ──────────────────────────────────
        title_bar = BoxLayout(size_hint_y=None, height=dp(44))
        with title_bar.canvas.before:
            Color(*C_TAB_INACTIVE)
            self._tb_bg = Rectangle(pos=title_bar.pos, size=title_bar.size)
        title_bar.bind(pos=lambda i, v: setattr(self._tb_bg, 'pos', v),
                       size=lambda i, v: setattr(self._tb_bg, 'size', v))
        title_bar.add_widget(_lbl("CryptoScanner Pro", 18, C_TAB_ACTIVE, True))
        self.root.add_widget(title_bar)

        # ── tab bar ────────────────────────────────────
        self.tab_names = ['交易对扫描', '交易池', '监控池', '数据库']
        self.tab_buttons = []
        tab_bar = BoxLayout(size_hint_y=None, height=dp(40), spacing=dp(2))
        for i, name in enumerate(self.tab_names):
            btn = _btn(name, C_TAB_INACTIVE if i > 0 else C_TAB_ACTIVE, 13)
            btn.tab_idx = i
            btn.bind(on_release=self._switch_tab)
            self.tab_buttons.append(btn)
            tab_bar.add_widget(btn)
        self.root.add_widget(tab_bar)

        # ── content area ───────────────────────────────
        self.tab_content = BoxLayout()
        self.root.add_widget(self.tab_content)

        # ── status bar ─────────────────────────────────
        self.status_lbl = _lbl("就绪", 11, C_SUB)
        self.root.add_widget(BoxLayout(size_hint_y=None, height=dp(24), children=[self.status_lbl]))

        # ── build pages ────────────────────────────────
        self._build_scanner_page()
        self._build_trade_pool_page()
        self._build_monitor_page()
        self._build_database_page()

        self._switch_tab(self.tab_buttons[0])
        return self.root

    # ── tab switching ───────────────────────────────────
    def _switch_tab(self, btn):
        for b in self.tab_buttons:
            b.background_color = C_TAB_ACTIVE if b is btn else C_TAB_INACTIVE
        self.tab_content.clear_widgets()
        idx = btn.tab_idx
        pages = [self._scanner_page, self._pool_page, self._monitor_page, self._db_page]
        self.tab_content.add_widget(pages[idx])

    # ═══════════════════════════════════════════════════════
    # TAB 1: 交易对扫描
    # ═══════════════════════════════════════════════════════
    def _build_scanner_page(self):
        page = BoxLayout(orientation='vertical', padding=dp(8), spacing=dp(6))
        scroll = ScrollView()
        content = BoxLayout(orientation='vertical', size_hint_y=None, spacing=dp(6))
        content.bind(minimum_height=content.setter('height'))

        content.add_widget(_lbl("API 配置", 14, C_TAB_ACTIVE, True))
        self.api_key_ti = _input("OKX API Key")
        self.secret_ti = _input("Secret Key", pw=True)
        self.phrase_ti = _input("Passphrase", pw=True)
        self.proxy_ti = _input("代理(可选)", self._cfg.get('proxy_url', ''))
        content.add_widget(self.api_key_ti)
        content.add_widget(self.secret_ti)
        content.add_widget(self.phrase_ti)
        content.add_widget(self.proxy_ti)

        content.add_widget(_lbl("策略选择", 14, C_TAB_ACTIVE, True))
        row = BoxLayout(size_hint_y=None, height=dp(40), spacing=dp(6))
        self.strat_spinner = Spinner(text='OKX小时线波段共振策略',
            values=['OKX小时线波段共振策略'], size_hint_x=0.65,
            background_color=C_CARD, color=C_TEXT)
        _font(self.strat_spinner)
        row.add_widget(self.strat_spinner)
        row.add_widget(_btn("加载", C_BTN, 12))
        content.add_widget(row)

        row2 = BoxLayout(size_hint_y=None, height=dp(40), spacing=dp(6))
        row2.add_widget(_btn("测试连接", (0.40, 0.40, 0.25, 1), 12))
        row2.add_widget(_btn("保存配置", (0.20, 0.20, 0.25, 1), 12))
        content.add_widget(row2)

        content.add_widget(_lbl("定时扫描", 14, C_TAB_ACTIVE, True))
        tr = BoxLayout(size_hint_y=None, height=dp(40), spacing=dp(6))
        self.timer_ti = _input("间隔(秒)", self._cfg.get('auto_scan_interval', '600'))
        self.timer_ti.size_hint_x = 0.4
        tr.add_widget(self.timer_ti)
        tr.add_widget(_lbl("秒", 12, C_SUB))
        tr.add_widget(_btn("开启定时", (0.45, 0.35, 0.25, 1), 12))
        content.add_widget(tr)

        self.scan_progress = ProgressBar(value=0, size_hint_y=None, height=dp(14))
        content.add_widget(self.scan_progress)
        self.scan_status = _lbl("就绪：配置 API 后点击扫描", 12, C_SUB)
        content.add_widget(self.scan_status)

        content.add_widget(_lbl("扫描结果", 14, C_TAB_ACTIVE, True))
        self.result_box = BoxLayout(orientation='vertical', size_hint_y=None, spacing=dp(4))
        self.result_box.bind(minimum_height=self.result_box.setter('height'))
        content.add_widget(self.result_box)

        # bottom scan button
        row3 = BoxLayout(size_hint_y=None, height=dp(44), spacing=dp(6))
        self.scan_btn = _btn("开始扫描", C_BTN, 14, True)
        self.scan_btn.bind(on_release=self._do_scan)
        row3.add_widget(self.scan_btn)
        row3.add_widget(_btn("保存配置到文件", (0.20, 0.20, 0.25, 1), 12))
        content.add_widget(row3)

        scroll.add_widget(content)
        page.add_widget(scroll)
        self._scanner_page = page

    def _init_okx(self):
        if not self.okx_client:
            self.okx_client = OKXClient(
                api_key=self.api_key_ti.text,
                secret_key=self.secret_ti.text,
                passphrase=self.phrase_ti.text,
                testnet=True,
                proxy_url=self.proxy_ti.text.strip() or None,
            )

    def _do_scan(self, btn):
        if self.is_scanning:
            return
        if not self.api_key_ti.text or not self.secret_ti.text:
            self._popup("提示", "请先填写 API Key 和 Secret Key")
            return
        self.is_scanning = True
        self.scan_btn.disabled = True
        self.scan_btn.text = "扫描中..."
        self.result_box.clear_widgets()
        self.scan_progress.value = 0
        self._status("正在连接 OKX...")
        threading.Thread(target=self._scan_thread, daemon=True).start()

    def _scan_thread(self):
        try:
            self._init_okx()
            self._status("获取行情...")
            self._set_progress(5)
            res = self.okx_client.get_tickers('SWAP')
            if not isinstance(res, dict) or res.get('code') != '0':
                msg = res.get('msg', '网络错误') if isinstance(res, dict) else '连接失败'
                self._show_err(f"API错误: {msg}")
                return
            tickers = res.get('data', [])
            swaps = [t for t in tickers if t.get('instId', '').endswith('-USDT-SWAP')]
            active = sorted(
                [t for t in swaps if float(t.get('volCcyQuote') or t.get('vol24h') or 0) > 5000000],
                key=lambda t: float(t.get('volCcyQuote') or t.get('vol24h') or 0), reverse=True
            )[:30]
            self._status(f"{len(active)} 个活跃品种, 开始分析...")
            self._set_progress(10)
            found = 0
            for i, t in enumerate(active):
                inst_id = t['instId']
                pct = 10 + int(85 * (i + 1) / len(active))
                self._set_progress(pct)
                self._status(f"[{i+1}/{len(active)}] {inst_id}")
                try:
                    klines = {}
                    for bar in ['1D', '1H', '15m', '3m']:
                        r = self.okx_client.get_kline(inst_id, bar=bar, limit=200)
                        if isinstance(r, dict) and r.get('code') == '0' and r.get('data'):
                            klines[bar] = r['data']
                    if not klines.get('1D') or not klines.get('1H'):
                        continue
                    sym = ScannerSymbol(
                        inst_id=inst_id,
                        last_price=float(t.get('last', 0)),
                        volume_24h=float(t.get('volCcyQuote') or t.get('vol24h') or 0),
                        extra_data={'klines': klines},
                    )
                    result = self.scanner.scan_symbol(sym)
                    if result.get('passed', False) or result.get('score', 0) >= 60:
                        found += 1
                        self._add_result(result)
                except Exception:
                    continue
                time.sleep(0.15)
            self._status(f"扫描完成！发现 {found} 个交易机会")
            self._set_progress(100)
        except Exception as e:
            self._show_err(f"扫描失败: {e}")
        finally:
            self.is_scanning = False
            Clock.schedule_once(lambda dt: self._reset_scan_btn())

    def _reset_scan_btn(self):
        self.scan_btn.disabled = False
        self.scan_btn.text = "开始扫描"

    def _add_result(self, r):
        def _f(dt):
            box = BoxLayout(orientation='vertical', size_hint_y=None, height=dp(60), spacing=dp(2))
            with box.canvas.before:
                Color(*C_CARD)
                box._bg = Rectangle(pos=box.pos, size=box.size)
            box.bind(pos=lambda i, v: setattr(box._bg, 'pos', v),
                     size=lambda i, v: setattr(box._bg, 'size', v))
            d = r.get('direction', 'NEUTRAL')
            dc = C_ACC if d == 'LONG' else (C_RED if d == 'SHORT' else C_SUB)
            hdr = BoxLayout(size_hint_y=None, height=dp(24))
            hdr.add_widget(_lbl(r.get('symbol', '?'), 13, C_TEXT, True))
            hdr.add_widget(_lbl(f"{d} {r.get('score', 0):.0f}分", 12, dc))
            box.add_widget(hdr)
            sigs = ' | '.join(r.get('signals', [])[:3]) or '无信号'
            box.add_widget(_lbl(sigs, 10, C_SUB))
            self.result_box.add_widget(box)
        Clock.schedule_once(_f)

    # ═══════════════════════════════════════════════════════
    # TAB 2: 交易池
    # ═══════════════════════════════════════════════════════
    def _build_trade_pool_page(self):
        page = BoxLayout(orientation='vertical', padding=dp(8), spacing=dp(6))
        page.add_widget(_lbl("交易池 - 扫描结果历史", 14, C_TAB_ACTIVE, True))
        scroll = ScrollView()
        self.pool_content = BoxLayout(orientation='vertical', size_hint_y=None, spacing=dp(4))
        self.pool_content.bind(minimum_height=self.pool_content.setter('height'))
        self.pool_content.add_widget(_lbl("暂无数据。扫描后会在此显示结果。", 12, C_SUB))
        scroll.add_widget(self.pool_content)
        page.add_widget(scroll)
        bar = BoxLayout(size_hint_y=None, height=dp(40), spacing=dp(6))
        bar.add_widget(_btn("导出扫描记录", (0.25, 0.40, 0.30, 1), 12))
        bar.add_widget(_btn("清空记录", (0.30, 0.30, 0.35, 1), 12))
        page.add_widget(bar)
        self._pool_page = page

    # ═══════════════════════════════════════════════════════
    # TAB 3: 监控池
    # ═══════════════════════════════════════════════════════
    def _build_monitor_page(self):
        page = BoxLayout(orientation='vertical', padding=dp(8), spacing=dp(6))
        page.add_widget(_lbl("监控池 - 实时信号监控", 14, C_TAB_ACTIVE, True))
        page.add_widget(_lbl("添加交易对到监控列表，自动检测趋势信号", 11, C_SUB))
        row = BoxLayout(size_hint_y=None, height=dp(40), spacing=dp(6))
        self.monitor_input = _input("输入交易对 如 BTC-USDT-SWAP")
        self.monitor_input.size_hint_x = 0.6
        row.add_widget(self.monitor_input)
        row.add_widget(_btn("添加", C_BTN, 12))
        page.add_widget(row)
        scroll = ScrollView()
        self.monitor_content = BoxLayout(orientation='vertical', size_hint_y=None, spacing=dp(4))
        self.monitor_content.bind(minimum_height=self.monitor_content.setter('height'))
        self.monitor_content.add_widget(_lbl("暂无监控交易对", 12, C_SUB))
        scroll.add_widget(self.monitor_content)
        page.add_widget(scroll)
        bar = BoxLayout(size_hint_y=None, height=dp(40), spacing=dp(6))
        bar.add_widget(_btn("开始监控", C_ACC, 12))
        bar.add_widget(_btn("停止监控", C_RED, 12))
        page.add_widget(bar)
        self._monitor_page = page

    # ═══════════════════════════════════════════════════════
    # TAB 4: 交易对数据库
    # ═══════════════════════════════════════════════════════
    def _build_database_page(self):
        page = BoxLayout(orientation='vertical', padding=dp(8), spacing=dp(6))
        page.add_widget(_lbl("交易对数据库 - K线数据管理", 14, C_TAB_ACTIVE, True))
        page.add_widget(_lbl("下载并管理交易对历史K线数据", 11, C_SUB))
        row = BoxLayout(size_hint_y=None, height=dp(40), spacing=dp(6))
        self.db_input = _input("交易对 如 BTC-USDT-SWAP")
        self.db_input.size_hint_x = 0.5
        row.add_widget(self.db_input)
        row.add_widget(_btn("下载K线", C_BTN, 12))
        page.add_widget(row)
        scroll = ScrollView()
        self.db_content = BoxLayout(orientation='vertical', size_hint_y=None, spacing=dp(4))
        self.db_content.bind(minimum_height=self.db_content.setter('height'))
        self.db_content.add_widget(_lbl("暂无下载数据", 12, C_SUB))
        scroll.add_widget(self.db_content)
        page.add_widget(scroll)
        bar = BoxLayout(size_hint_y=None, height=dp(40), spacing=dp(6))
        bar.add_widget(_btn("刷新列表", (0.20, 0.20, 0.25, 1), 12))
        bar.add_widget(_btn("清空数据", C_RED, 12))
        page.add_widget(bar)
        self._db_page = page

    # ── helpers ──────────────────────────────────────────
    def _status(self, msg):
        Clock.schedule_once(lambda dt: setattr(self.status_lbl, 'text', msg))
        Clock.schedule_once(lambda dt: setattr(self.scan_status, 'text', msg))

    def _set_progress(self, v):
        Clock.schedule_once(lambda dt: setattr(self.scan_progress, 'value', v))

    def _show_err(self, msg):
        def _f(dt):
            self._popup("错误", msg)
            self._status(msg)
        Clock.schedule_once(_f)

    def _popup(self, title, text):
        content = BoxLayout(orientation='vertical', padding=dp(10), spacing=dp(8))
        lbl = Label(text=text, font_size=sp(12), halign='left', valign='top', color=C_TEXT)
        _font(lbl)
        lbl.bind(size=lbl.setter('text_size'))
        content.add_widget(lbl)
        btn = Button(text="关闭", size_hint_y=None, height=dp(36),
                     background_color=C_CARD, color=C_TEXT)
        _font(btn)
        content.add_widget(btn)
        popup = Popup(title=title, content=content, size_hint=(0.9, 0.6),
                      background_color=(0.15, 0.15, 0.18, 0.95),
                      separator_color=C_TAB_ACTIVE)
        btn.bind(on_release=popup.dismiss)
        popup.open()

    def _load_cfg(self):
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception:
            pass
        return {}

    def _save_cfg(self):
        cfg = {'proxy_url': self.proxy_ti.text, 'auto_scan_interval': self.timer_ti.text}
        try:
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(cfg, f, ensure_ascii=False, indent=2)
            return True
        except Exception:
            return False

    def on_pause(self): return True
    def on_resume(self): pass


if __name__ == '__main__':
    if not _IMPORTS_OK:
        class ErrApp(App):
            def build(self):
                from kivy.uix.boxlayout import BoxLayout
                from kivy.uix.label import Label
                from kivy.uix.button import Button
                from kivy.uix.scrollview import ScrollView
                Window.size = (360, 748)
                root = BoxLayout(orientation='vertical', padding=dp(20), spacing=dp(10))
                lbl = Label(text=f"启动失败\n\n{_IMPORT_ERR}\n\n查看 /sdcard/cs_crash.log",
                           font_size=sp(12), color=C_RED, halign='left', valign='top')
                lbl.bind(size=lbl.setter('text_size'))
                scr = ScrollView()
                scr.add_widget(lbl)
                root.add_widget(scr)
                btn = Button(text="退出", size_hint_y=None, height=dp(40),
                           background_color=C_RED, color=C_TEXT)
                btn.bind(on_release=lambda x: sys.exit(0))
                root.add_widget(btn)
                return root
        ErrApp().run()
    else:
        CryptoApp().run()
