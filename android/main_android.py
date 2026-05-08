"""
CryptoScanner Pro - Android Full Version
Kivy multi-tab trading platform
"""
import os, sys, json, threading, time, glob, traceback, importlib.util, warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── clipboard helper ──────────────────────────────────────
def _get_clipboard():
    """Get text from Android clipboard via pyjnius"""
    try:
        from jnius import autoclass
        PythonActivity = autoclass('org.kivy.android.PythonActivity')
        Context = autoclass('android.content.Context')
        activity = PythonActivity.mActivity
        clipboard = activity.getSystemService(Context.CLIPBOARD_SERVICE)
        if clipboard.hasPrimaryClip():
            clip = clipboard.getPrimaryClip()
            if clip.getItemCount() > 0:
                return clip.getItemAt(0).getText().toString()
    except Exception:
        pass
    return ''

def _paste_to(ti):
    """Paste clipboard into text input"""
    text = _get_clipboard()
    if text:
        ti.text = text

def _crash(msg):
    try:
        with open('/sdcard/cs_crash.log', 'a') as f:
            f.write(msg + '\n')
    except Exception:
        pass

_crash("App starting")
_crash(f"Python {sys.version}")

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

# font
from kivy.core.text import LabelBase
FONT_NAME = None
for fp in [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fonts', 'PingFang.ttc'),
    '/system/fonts/NotoSansCJK-Regular.ttc',
    '/system/fonts/DroidSansFallback.ttf',
]:
    try:
        if os.path.exists(fp):
            LabelBase.register(name='CNFont', fn_regular=fp)
            FONT_NAME = 'CNFont'
            break
    except Exception:
        pass

# theme
C_BG   = (0.10, 0.10, 0.13, 1)
C_CARD = (0.16, 0.16, 0.20, 1)
C_BTN  = (0.18, 0.50, 0.80, 1)
C_ACC  = (0.20, 0.80, 0.40, 1)
C_WARN = (0.90, 0.60, 0.10, 1)
C_RED  = (0.85, 0.20, 0.20, 1)
C_TEXT = (0.90, 0.90, 0.95, 1)
C_SUB  = (0.55, 0.55, 0.60, 1)
C_TAB  = (0.18, 0.55, 0.85, 1)
C_TAB_OFF = (0.12, 0.12, 0.16, 1)

def _font(w):
    if FONT_NAME: w.font_name = FONT_NAME

def _lbl(text, s=15, c=C_TEXT, b=False, h='left'):
    l = Label(text=text, font_size=sp(s), color=c, bold=b, halign=h, valign='middle')
    l.bind(size=l.setter('text_size'))
    _font(l)
    return l

def _btn(text, bg=C_BTN, s=14, b=False, cb=None):
    btn = Button(text=text, font_size=sp(s), bold=b, background_color=bg, color=C_TEXT, background_normal='')
    _font(btn)
    if cb: btn.bind(on_release=cb)
    return btn

def _in(hint='', text='', pw=False):
    t = TextInput(hint_text=hint, text=text, password=pw, multiline=False,
                  font_size=sp(14), background_color=C_CARD, foreground_color=C_TEXT,
                  cursor_color=C_TAB, padding=[dp(12), dp(12)],
                  size_hint_y=None, height=dp(52))
    _font(t)
    return t


class CryptoApp(App):
    def build(self):
        Window.minimum_width, Window.minimum_height = 320, 480
        Window.softinput_mode = 'below_target'

        self._data_dir = os.path.dirname(os.path.abspath(__file__))
        self.cfg_file = os.path.join(self._data_dir, "scanner_config.json")
        self._cfg = self._load_cfg()
        self.okx_client = None
        self.scanner = OKXHourSwingScanner()
        self.is_scanning = False
        self.pool_results = []  # trade pool
        self.monitor_list = []  # monitor symbols

        root = BoxLayout(orientation='vertical')

        # title
        tb = BoxLayout(size_hint_y=None, height=dp(52))
        with tb.canvas.before:
            Color(*C_TAB_OFF)
            self._tbbg = Rectangle(pos=tb.pos, size=tb.size)
        tb.bind(pos=lambda i, v: setattr(self._tbbg, 'pos', v),
                size=lambda i, v: setattr(self._tbbg, 'size', v))
        tb.add_widget(_lbl("CryptoScanner Pro", 20, C_TAB, True))
        root.add_widget(tb)

        # tabs
        self.tabs = ['交易对扫描', '交易池', '监控池', '数据库']
        self.tab_btns = []
        tr = BoxLayout(size_hint_y=None, height=dp(48), spacing=dp(2))
        for i, n in enumerate(self.tabs):
            b = Button(text=n, font_size=sp(14), color=C_TEXT,
                       background_color=C_TAB if i == 0 else C_TAB_OFF,
                       background_normal='')
            b.tab_idx = i
            b.bind(on_release=self._switch_tab)
            _font(b)
            self.tab_btns.append(b)
            tr.add_widget(b)
        root.add_widget(tr)

        # pages
        self.pages = [self._scanner(), self._pool(), self._monitor(), self._database()]
        self.content = BoxLayout()
        self.content.add_widget(self.pages[0])
        root.add_widget(self.content)

        # status
        self.slbl = _lbl("就绪", 11, C_SUB)
        root.add_widget(BoxLayout(size_hint_y=None, height=dp(24), children=[self.slbl]))
        return root

    def _switch_tab(self, btn):
        for b in self.tab_btns:
            b.background_color = C_TAB if b is btn else C_TAB_OFF
        self.content.clear_widgets()
        self.content.add_widget(self.pages[btn.tab_idx])

    # ═══════════════════════ scanner ═══════════════════════
    def _scanner(self):
        p = BoxLayout(orientation='vertical', padding=dp(10), spacing=dp(8))

        # Fixed API config section (not scrollable)
        cfg_section = BoxLayout(orientation='vertical', size_hint_y=None, spacing=dp(6))
        cfg_section.bind(minimum_height=cfg_section.setter('height'))

        cfg_section.add_widget(_lbl("API 配置", 16, C_TAB, True))

        def _paste_row(input_widget):
            r = BoxLayout(size_hint_y=None, height=dp(48), spacing=dp(6))
            ti = input_widget
            ti.size_hint_x = 0.75
            r.add_widget(ti)
            pb = _btn("粘贴", (0.25, 0.25, 0.35, 1), 12, cb=lambda x: _paste_to(ti))
            pb.size_hint_x = 0.25
            r.add_widget(pb)
            return r

        self.api_ti = _in("OKX API Key")
        self.sec_ti = _in("Secret Key", pw=True)
        self.phr_ti = _in("Passphrase", pw=True)
        self.prx_ti = _in("代理(可选)", self._cfg.get('proxy_url', ''))
        cfg_section.add_widget(_paste_row(self.api_ti))
        cfg_section.add_widget(_paste_row(self.sec_ti))
        cfg_section.add_widget(_paste_row(self.phr_ti))
        cfg_section.add_widget(self.prx_ti)

        cfg_section.add_widget(_lbl("策略: OKX小时线波段共振策略", 13, C_SUB))

        r1 = BoxLayout(size_hint_y=None, height=dp(56), spacing=dp(8))
        r1.add_widget(_btn("测试连接", (0.40, 0.40, 0.25, 1), 14, cb=self._test_conn))
        r1.add_widget(_btn("保存配置", (0.20, 0.20, 0.25, 1), 14, cb=lambda x: self._popup("配置", "已保存" if self._save_cfg() else "失败")))
        cfg_section.add_widget(r1)

        cfg_section.add_widget(_lbl("定时扫描(秒)", 16, C_TAB, True))
        self.tmr_ti = TextInput(text=self._cfg.get('auto_scan_interval', '600'),
                                hint_text="600", multiline=False,
                                font_size=sp(14), background_color=C_CARD,
                                foreground_color=C_TEXT, size_hint_y=None, height=dp(52),
                                input_filter='int')
        _font(self.tmr_ti)
        r2 = BoxLayout(size_hint_y=None, height=dp(56), spacing=dp(8))
        r2.add_widget(self.tmr_ti)
        self.auto_btn = _btn("开启定时", (0.45, 0.35, 0.25, 1), 14, cb=self._toggle_auto)
        r2.add_widget(self.auto_btn)
        cfg_section.add_widget(r2)

        self.pbar = ProgressBar(value=0, size_hint_y=None, height=dp(20))
        cfg_section.add_widget(self.pbar)
        self.ss = _lbl("就绪：配置 API 后点击扫描", 13, C_SUB)
        cfg_section.add_widget(self.ss)

        p.add_widget(cfg_section)

        # Scrollable results area
        p.add_widget(_lbl("扫描结果", 16, C_TAB, True))
        sv = ScrollView()
        self.rb = BoxLayout(orientation='vertical', size_hint_y=None, spacing=dp(4))
        self.rb.bind(minimum_height=self.rb.setter('height'))
        sv.add_widget(self.rb)
        p.add_widget(sv)

        # Fixed scan button
        r3 = BoxLayout(size_hint_y=None, height=dp(56), spacing=dp(8))
        self.scan_btn = _btn("开始扫描", C_BTN, 16, True, cb=self._do_scan)
        r3.add_widget(self.scan_btn)
        r3.add_widget(_btn("保存配置到文件", (0.20, 0.20, 0.25, 1), 14, cb=lambda x: self._popup("配置", "已保存" if self._save_cfg() else "失败")))
        p.add_widget(r3)
        return p

    # ═══════════════════════ pool ═══════════════════════
    def _pool(self):
        p = BoxLayout(orientation='vertical', padding=dp(10), spacing=dp(8))
        p.add_widget(_lbl("交易池", 16, C_TAB, True))
        p.add_widget(_lbl("扫描通过的结果自动汇集到此", 13, C_SUB))
        sv = ScrollView()
        self.pc = BoxLayout(orientation='vertical', size_hint_y=None, spacing=dp(3))
        self.pc.bind(minimum_height=self.pc.setter('height'))
        self.pc.add_widget(_lbl("暂无数据", 12, C_SUB))
        sv.add_widget(self.pc)
        p.add_widget(sv)
        bar = BoxLayout(size_hint_y=None, height=dp(52), spacing=dp(8))
        bar.add_widget(_btn("导出扫描记录", (0.25, 0.40, 0.30, 1), 12, cb=lambda x: self._popup("导出", "功能开发中")))
        bar.add_widget(_btn("清空记录", (0.30, 0.30, 0.35, 1), 12, cb=self._clear_pool))
        p.add_widget(bar)
        return p

    # ═══════════════════════ monitor ═══════════════════════
    def _monitor(self):
        p = BoxLayout(orientation='vertical', padding=dp(10), spacing=dp(8))
        p.add_widget(_lbl("监控池", 16, C_TAB, True))
        p.add_widget(_lbl("添加交易对到监控列表", 13, C_SUB))
        r = BoxLayout(size_hint_y=None, height=dp(52), spacing=dp(8))
        self.mi = TextInput(hint_text="如 BTC-USDT-SWAP", multiline=False,
                            font_size=sp(14), background_color=C_CARD, foreground_color=C_TEXT)
        _font(self.mi)
        r.add_widget(self.mi)
        r.add_widget(_btn("添加", C_BTN, 12, cb=self._add_monitor))
        p.add_widget(r)
        sv = ScrollView()
        self.mc = BoxLayout(orientation='vertical', size_hint_y=None, spacing=dp(3))
        self.mc.bind(minimum_height=self.mc.setter('height'))
        self.mc.add_widget(_lbl("暂无监控交易对", 12, C_SUB))
        sv.add_widget(self.mc)
        p.add_widget(sv)
        bar = BoxLayout(size_hint_y=None, height=dp(52), spacing=dp(8))
        bar.add_widget(_btn("开始监控", C_ACC, 12, cb=lambda x: self._popup("监控", "功能开发中")))
        bar.add_widget(_btn("停止监控", C_RED, 12, cb=lambda x: self._popup("监控", "已停止")))
        p.add_widget(bar)
        return p

    # ═══════════════════════ database ═══════════════════════
    def _database(self):
        p = BoxLayout(orientation='vertical', padding=dp(10), spacing=dp(8))
        p.add_widget(_lbl("交易对数据库", 16, C_TAB, True))
        p.add_widget(_lbl("下载历史K线数据", 13, C_SUB))
        r = BoxLayout(size_hint_y=None, height=dp(52), spacing=dp(8))
        self.dbi = TextInput(hint_text="如 BTC-USDT-SWAP", multiline=False,
                             font_size=sp(14), background_color=C_CARD, foreground_color=C_TEXT)
        _font(self.dbi)
        r.add_widget(self.dbi)
        r.add_widget(_btn("下载K线", C_BTN, 12, cb=lambda x: self._popup("数据库", "功能开发中")))
        p.add_widget(r)
        sv = ScrollView()
        self.dc = BoxLayout(orientation='vertical', size_hint_y=None, spacing=dp(3))
        self.dc.bind(minimum_height=self.dc.setter('height'))
        self.dc.add_widget(_lbl("暂无数据", 12, C_SUB))
        sv.add_widget(self.dc)
        p.add_widget(sv)
        bar = BoxLayout(size_hint_y=None, height=dp(52), spacing=dp(8))
        bar.add_widget(_btn("刷新", (0.20, 0.20, 0.25, 1), 12, cb=lambda x: self._popup("数据库", "已刷新")))
        bar.add_widget(_btn("清空", C_RED, 12, cb=lambda x: self._popup("数据库", "已清空")))
        p.add_widget(bar)
        return p

    # ── scanner actions ──────────────────────────────────
    def _test_conn(self, btn):
        self._status("测试连接...")
        threading.Thread(target=self._test_conn_thread, daemon=True).start()

    def _test_conn_thread(self):
        try:
            client = OKXClient(api_key=self.api_ti.text, secret_key=self.sec_ti.text,
                               passphrase=self.phr_ti.text, testnet=True,
                               proxy_url=self.prx_ti.text.strip() or None)
            res = client.get_tickers('SWAP')
            if isinstance(res, dict) and res.get('code') == '0':
                n = len(res.get('data', []))
                Clock.schedule_once(lambda dt: self._popup("成功", f"获取 {n} 个交易对"))
                Clock.schedule_once(lambda dt: self._status("API 连接正常"))
            else:
                msg = res.get('msg', '?') if isinstance(res, dict) else '网络错误'
                Clock.schedule_once(lambda dt: self._popup("失败", msg))
                Clock.schedule_once(lambda dt: self._status(f"连接失败: {msg}"))
        except Exception as e:
            Clock.schedule_once(lambda dt: self._popup("错误", str(e)))
            Clock.schedule_once(lambda dt: self._status(f"错误: {e}"))

    def _do_scan(self, btn):
        if self.is_scanning: return
        if not self.api_ti.text or not self.sec_ti.text:
            self._popup("提示", "请填写 API Key 和 Secret Key"); return
        self._save_cfg()
        self.is_scanning = True
        self.scan_btn.disabled = True
        self.scan_btn.text = "扫描中..."
        self.rb.clear_widgets()
        self.pbar.value = 0
        self._status("连接 OKX...")
        threading.Thread(target=self._scan_thread, daemon=True).start()

    def _scan_thread(self):
        try:
            if not self.okx_client:
                self.okx_client = OKXClient(api_key=self.api_ti.text, secret_key=self.sec_ti.text,
                                            passphrase=self.phr_ti.text, testnet=True,
                                            proxy_url=self.prx_ti.text.strip() or None)
            self._status("获取行情..."); self._set_progress(5)
            res = self.okx_client.get_tickers('SWAP')
            if not isinstance(res, dict) or res.get('code') != '0':
                msg = res.get('msg', '?') if isinstance(res, dict) else '连接失败'
                self._err(f"API错误: {msg}"); return
            tickers = res.get('data', [])
            swaps = [t for t in tickers if t.get('instId', '').endswith('-USDT-SWAP')]
            active = sorted([t for t in swaps if float(t.get('volCcyQuote') or t.get('vol24h') or 0) > 5000000],
                            key=lambda t: float(t.get('volCcyQuote') or t.get('vol24h') or 0), reverse=True)[:30]
            self._status(f"{len(active)} 个品种分析中..."); self._set_progress(10)
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
                    if not klines.get('1D') or not klines.get('1H'): continue
                    sym = ScannerSymbol(inst_id=inst_id,
                                        last_price=float(t.get('last', 0)),
                                        volume_24h=float(t.get('volCcyQuote') or t.get('vol24h') or 0),
                                        extra_data={'klines': klines})
                    result = self.scanner.scan_symbol(sym)
                    if result.get('passed', False) or result.get('score', 0) >= 60:
                        found += 1
                        self._add_result(result)
                        self.pool_results.append(result)
                except Exception:
                    continue
                time.sleep(0.15)
            self._status(f"扫描完成！{found} 个机会")
            self._set_progress(100)
        except Exception as e:
            self._err(f"扫描失败: {e}")
        finally:
            self.is_scanning = False
            Clock.schedule_once(lambda dt: self._reset_scan())

    def _reset_scan(self):
        self.scan_btn.disabled = False
        self.scan_btn.text = "开始扫描"

    def _add_result(self, r):
        def _f(dt):
            bx = BoxLayout(orientation='vertical', size_hint_y=None, height=dp(56), spacing=dp(2))
            with bx.canvas.before:
                Color(*C_CARD)
                bx._b = Rectangle(pos=bx.pos, size=bx.size)
            bx.bind(pos=lambda i, v: setattr(bx._b, 'pos', v), size=lambda i, v: setattr(bx._b, 'size', v))
            d = r.get('direction', 'NEUTRAL')
            dc = C_ACC if d == 'LONG' else (C_RED if d == 'SHORT' else C_SUB)
            h = BoxLayout(size_hint_y=None, height=dp(24))
            h.add_widget(_lbl(r.get('symbol', '?'), 12, C_TEXT, True))
            h.add_widget(_lbl(f"{d} {r.get('score', 0):.0f}分", 12, dc))
            bx.add_widget(h)
            sigs = ' | '.join(r.get('signals', [])[:3]) or '无信号'
            bx.add_widget(_lbl(sigs, 10, C_SUB))
            self.rb.add_widget(bx)
        Clock.schedule_once(_f)

    # ── monitor ──────────────────────────────────────────
    def _add_monitor(self, btn):
        s = self.mi.text.strip()
        if not s: return
        if s not in self.monitor_list:
            self.monitor_list.append(s)
            if len(self.mc.children) == 1 and isinstance(self.mc.children[0], Label):
                self.mc.clear_widgets()
            bx = BoxLayout(size_hint_y=None, height=dp(36), spacing=dp(6))
            bx.add_widget(_lbl(s, 12, C_TEXT))
            bx.add_widget(_btn("删除", C_RED, 10, cb=lambda x, sym=s: self._rm_monitor(sym)))
            self.mc.add_widget(bx)
            self.mi.text = ''

    def _rm_monitor(self, s):
        if s in self.monitor_list:
            self.monitor_list.remove(s)
        self.mc.clear_widgets()
        for sym in self.monitor_list:
            bx = BoxLayout(size_hint_y=None, height=dp(36), spacing=dp(6))
            bx.add_widget(_lbl(sym, 12, C_TEXT))
            bx.add_widget(_btn("删除", C_RED, 10, cb=lambda x, m=sym: self._rm_monitor(m)))
            self.mc.add_widget(bx)
        if not self.monitor_list:
            self.mc.add_widget(_lbl("暂无监控交易对", 12, C_SUB))

    # ── pool ─────────────────────────────────────────────
    def _clear_pool(self, btn):
        self.pool_results = []
        self.pc.clear_widgets()
        self.pc.add_widget(_lbl("已清空", 12, C_SUB))

    # ── auto scan ────────────────────────────────────────
    def _toggle_auto(self, btn):
        if hasattr(self, '_auto_timer') and self._auto_timer:
            self._auto_timer.cancel()
            self._auto_timer = None
            self.auto_btn.text = "开启定时"
            self.auto_btn.background_color = (0.45, 0.35, 0.25, 1)
            self._status("定时已停止")
        else:
            try:
                sec = int(self.tmr_ti.text)
                if sec < 60: sec = 60
            except ValueError:
                sec = 600
            self.auto_btn.text = "停止定时"
            self.auto_btn.background_color = C_RED
            self._status(f"定时 {sec}秒 已开启")
            self._auto_timer = Clock.schedule_interval(lambda dt: self._do_scan(None), sec)

    # ── helpers ──────────────────────────────────────────
    def _status(self, msg):
        Clock.schedule_once(lambda dt: setattr(self.slbl, 'text', msg))
        Clock.schedule_once(lambda dt: setattr(self.ss, 'text', msg))

    def _set_progress(self, v):
        Clock.schedule_once(lambda dt: setattr(self.pbar, 'value', v))

    def _err(self, msg):
        def _f(dt): self._popup("错误", msg); self._status(msg)
        Clock.schedule_once(_f)

    def _popup(self, title, text):
        c = BoxLayout(orientation='vertical', padding=dp(12), spacing=dp(8))
        l = Label(text=text, font_size=sp(13), halign='left', valign='top', color=C_TEXT)
        _font(l); l.bind(size=l.setter('text_size'))
        c.add_widget(l)
        b = Button(text="关闭", size_hint_y=None, height=dp(36),
                   background_color=C_BTN, color=C_TEXT); _font(b)
        c.add_widget(b)
        pp = Popup(title=title, content=c, size_hint=(0.85, 0.5),
                   background_color=(0.15, 0.15, 0.18, 0.95), separator_color=C_TAB)
        b.bind(on_release=pp.dismiss)
        pp.open()

    def _load_cfg(self):
        try:
            if os.path.exists(self.cfg_file):
                with open(self.cfg_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception: pass
        return {}

    def _save_cfg(self):
        cfg = {'proxy_url': self.prx_ti.text, 'auto_scan_interval': self.tmr_ti.text}
        try:
            os.makedirs(os.path.dirname(self.cfg_file), exist_ok=True)
            with open(self.cfg_file, 'w', encoding='utf-8') as f:
                json.dump(cfg, f, ensure_ascii=False, indent=2)
            return True
        except Exception: return False

    def on_pause(self):
        if hasattr(self, '_auto_timer') and self._auto_timer:
            self._auto_timer.cancel()
            self._auto_timer = None
        return True

    def on_resume(self): pass


if __name__ == '__main__':
    if not _IMPORTS_OK:
        class ErrApp(App):
            def build(self):
                r = BoxLayout(orientation='vertical', padding=dp(20), spacing=dp(10))
                l = Label(text=f"启动失败\n\n{_IMPORT_ERR}\n\n/sdcard/cs_crash.log",
                          font_size=sp(12), color=C_RED, halign='left', valign='top')
                l.bind(size=l.setter('text_size'))
                s = ScrollView(); s.add_widget(l); r.add_widget(s)
                b = Button(text="退出", size_hint_y=None, height=dp(40),
                           background_color=C_RED, color=C_TEXT)
                b.bind(on_release=lambda x: sys.exit(0)); r.add_widget(b)
                return r
        ErrApp().run()
    else:
        CryptoApp().run()
