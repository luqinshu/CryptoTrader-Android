"""
CryptoScanner Pro - Android Full Version
"""
import os, sys
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass
import json, threading, time, glob, traceback, importlib, warnings, logging
warnings.filterwarnings('ignore')
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('requests').setLevel(logging.WARNING)

from kivy.config import Config
Config.set('graphics', 'width', '390')
Config.set('graphics', 'height', '844')
Config.set('graphics', 'resizable', True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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
    from src.api.okx_client import OKXClient
    from src.scanner.base_scanner import ScannerSymbol
    from strategies.okx_swing import OKXHourSwingScanner
    IMP_OK = True
except Exception as e:
    IMP_OK = False
    IMP_ERR = str(e)

# font
from kivy.core.text import LabelBase
FN = None
for fp in [os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fonts', 'PingFang.ttc'),
           '/system/fonts/NotoSansCJK-Regular.ttc', '/system/fonts/DroidSansFallback.ttf']:
    try:
        if os.path.exists(fp): LabelBase.register(name='C', fn_regular=fp); FN = 'C'; break
    except Exception: pass

C_TAB = (0.18, 0.55, 0.85, 1)
C_BG  = (0.10, 0.10, 0.13, 1)
C_CRD = (0.16, 0.16, 0.20, 1)
C_BTN = (0.18, 0.50, 0.80, 1)
C_ACC = (0.20, 0.80, 0.40, 1)
C_RED = (0.85, 0.20, 0.20, 1)
C_TXT = (0.90, 0.90, 0.95, 1)
C_SUB = (0.55, 0.55, 0.60, 1)
C_WARN = (0.90, 0.60, 0.10, 1)
C_OFF = (0.12, 0.12, 0.16, 1)

def _f(w):
    if FN: w.font_name = FN

def L(text, s=14, c=C_TXT, b=False):
    l = Label(text=text, font_size=sp(s), color=c, bold=b, halign='left', valign='middle',
              size_hint_y=None, height=dp(s+10))
    l.bind(size=l.setter('text_size')); _f(l); return l

def B(text, bg=C_BTN, s=14, cb=None):
    btn = Button(text=text, font_size=sp(s), background_color=bg, color=C_TXT,
                 background_normal='', size_hint_y=None, height=dp(48))
    _f(btn)
    if cb: btn.bind(on_release=cb)
    return btn

def TI(hint='', text='', pw=False):
    t = TextInput(hint_text=hint, text=text, password=pw, multiline=False,
                  font_size=sp(14), background_color=C_CRD, foreground_color=C_TXT,
                  cursor_color=C_TAB, padding=[dp(12), dp(12)], size_hint_y=None, height=dp(52))
    _f(t); return t


class App(App):
    def build(self):
        Window.softinput_mode = 'below_target'
        self.dir = os.path.dirname(os.path.abspath(__file__))
        self.cfg_path = os.path.join(self.dir, "app_config.json")
        self.cfg = self._load()
        self.okx = None
        self.scanner = OKXHourSwingScanner()
        self.scanning = False
        self.pool = []
        self.monitors = []
        self.positions_data = []
        self._tmon_balance_text = "账户: ----"
        self._tmon_equity_text = "权益: ----"
        self.auto_timer = None
        self._auto_seconds = 0
        self._selected_strat_names = list(self.cfg.get('selected_strategies', ['OKX小时线波段共振策略']))
        self._strat_scanners = {}
        self._smap = {
            'okx_hour_strategy': 'OKX小时线波段共振策略',
            'xiaoyue_boll': '小月期货多周期布林趋势转折',
            'three_min_pullback': '三分钟多周期回调企稳策略',
            'trend_squeeze': '趋势挤压突破前4_30_v2',
            'AI五引擎合并独立版': 'AI五引擎合并独立版',
            'AI五引擎合并独立版5.9': 'AI五引擎合并独立版v5.9',
            'xiaoyue_boll_5.9': '小月期货多周期布林v5.9',
            'trend_squeeze_5.9': '趋势挤压突破前v5.9',
        }
        self._pool_monitoring = False
        self._pool_monitor_timer = None
        self._monitor_lock = threading.Lock()
        self._at_balance_cache = "账户: ----"
        self._at_equity_cache = "权益: ----"

        root = BoxLayout(orientation='vertical')

        # title
        tb = BoxLayout(size_hint_y=None, height=dp(48))
        tb.add_widget(L("CryptoScanner Pro", 18, C_TAB, True))
        root.add_widget(tb)

        # tab bar
        self.tabs = ['交易扫描', '交易池', '交易监控', '自动交易']
        self.tbtns = []
        tr = BoxLayout(size_hint_y=None, height=dp(44), spacing=dp(2))
        for i,n in enumerate(self.tabs):
            b = Button(text=n, font_size=sp(13), color=C_TXT, background_color=C_TAB if i==0 else C_OFF, background_normal='')
            b.t = i; b.bind(on_release=self._tab); _f(b); self.tbtns.append(b); tr.add_widget(b)
        root.add_widget(tr)

        self.content = BoxLayout()
        self.content.add_widget(self._scan_page())
        root.add_widget(self.content)

        # status
        self.sl = L("就绪", 11, C_SUB)
        root.add_widget(BoxLayout(size_hint_y=None, height=dp(22), children=[self.sl]))
        # show API popup on first launch
        if not self.cfg.get('api_key'):
            Clock.schedule_once(lambda dt: self._show_api_popup(), 0.5)
        return root

    def _tab(self, btn):
        for b in self.tbtns: b.background_color = C_TAB if b is btn else C_OFF
        self.content.clear_widgets()
        self.content.add_widget([self._scan_page(), self._pool_page(), self._trade_mon_page(), self._auto_trade_page()][btn.t])

    # ═══════════════════ API Popup ═══════════════════
    def _show_api_popup(self):
        c = BoxLayout(orientation='vertical', padding=dp(12), spacing=dp(8))
        c.add_widget(L("OKX API 配置", 16, C_TAB, True))
        c.add_widget(L("仅保存在本地，App重启自动加载", 12, C_SUB))

        def _add_row(label, field):
            row = BoxLayout(size_hint_y=None, height=dp(52), spacing=dp(4))
            row.add_widget(field)
            pb = Button(text="粘贴", font_size=sp(11), color=C_TXT,
                        background_color=(0.25, 0.25, 0.30, 1), background_normal='',
                        size_hint_x=None, width=dp(52))
            _f(pb)
            pb.bind(on_release=lambda x, f=field: self._paste_to(f))
            row.add_widget(pb)
            c.add_widget(row)

        ak = TI("API Key", self.cfg.get('api_key',''))
        sk = TI("Secret Key", self.cfg.get('secret_key',''))
        ph = TI("Passphrase", self.cfg.get('passphrase',''))
        px = TI("代理(可选)", self.cfg.get('proxy_url',''))
        _add_row("API Key", ak)
        _add_row("Secret Key", sk)
        _add_row("Passphrase", ph)
        _add_row("代理", px)

        bb = BoxLayout(size_hint_y=None, height=dp(52), spacing=dp(8))
        bb.add_widget(B("测试连接", (0.40, 0.40, 0.25, 1), 13, cb=lambda x: self._test(ak.text, sk.text, ph.text, px.text)))
        bb.add_widget(B("保存", C_BTN, 13))
        c.add_widget(bb)

        pp = Popup(title="API 设置", content=c, size_hint=(0.9, 0.75),
                   background_color=(0.12, 0.12, 0.15, 0.97), separator_color=C_TAB, auto_dismiss=False)
        def save(_):
            self.cfg['api_key'] = ak.text.strip(); self.cfg['secret_key'] = sk.text.strip()
            self.cfg['passphrase'] = ph.text.strip(); self.cfg['proxy_url'] = px.text.strip()
            self._save()
            pp.dismiss()
            self._status("API 已保存")
        bb.children[0].bind(on_release=save)  # "保存" button
        pp.open()

    def _paste_to(self, field):
        try:
            from kivy.core.clipboard import Clipboard
            text = (Clipboard.get('text/plain') or Clipboard.get() or '').strip()
            if text:
                field.text = text
        except Exception:
            pass

    def _test(self, k, s, p, x):
        if not k or not s:
            self._pop("提示", "请填写 API Key 和 Secret Key"); return
        self._status("测试连接...")
        threading.Thread(target=self._test_thread, args=(k,s,p,x), daemon=True).start()

    def _test_thread(self, k, s, p, x):
        try:
            c = OKXClient(api_key=k, secret_key=s, passphrase=p, testnet=False, proxy_url=x.strip() or None)
            r = c.get_tickers('SWAP')
            if isinstance(r, dict) and r.get('code') == '0':
                n = len(r.get('data', []))
                Clock.schedule_once(lambda dt: self._pop("成功", f"获取 {n} 个交易对"))
            else:
                msg = r.get('msg', '?') if isinstance(r, dict) else '网络错误'
                Clock.schedule_once(lambda dt: self._pop("失败", msg))
        except Exception as e:
            Clock.schedule_once(lambda dt: self._pop("错误", str(e)))

    # ═══════════════════ Scanner ═══════════════════
    def _scan_page(self):
        p = BoxLayout(orientation='vertical', padding=dp(6), spacing=dp(3))

        # row 1: settings
        sr = BoxLayout(size_hint_y=None, height=dp(40), spacing=dp(4))
        sr.add_widget(B("API设置", (0.30, 0.30, 0.35, 1), 12, cb=lambda x: self._show_api_popup()))
        sr.add_widget(B("保存", (0.20, 0.20, 0.25, 1), 12, cb=lambda x: self._pop("配置", "已保存" if self._save() else "失败")))
        p.add_widget(sr)

        # row 2: strategy picker button -> popup
        sr2 = BoxLayout(size_hint_y=None, height=dp(40), spacing=dp(4))
        self._cur_strat_name = self._selected_strat_names[0] if self._selected_strat_names else 'OKX小时线波段共振策略'
        n = len(self._selected_strat_names)
        btn_txt = f"策略({n}): {self._cur_strat_name}" if n == 1 else f"策略: {n}个已选"
        self._strat_btn = B(btn_txt, (0.20, 0.25, 0.30, 1), 12, cb=self._show_strat_popup)
        sr2.add_widget(self._strat_btn)
        sr2.add_widget(B("从文件加载", (0.25, 0.45, 0.30, 1), 12, cb=self._load_from_file))
        p.add_widget(sr2)

        # row 3: interval + countdown
        ir = BoxLayout(size_hint_y=None, height=dp(40), spacing=dp(6))
        lbl_int = Label(text="间隔(秒)", font_size=sp(12), color=C_SUB,
                        halign='left', valign='middle', size_hint_x=None, width=dp(60))
        _f(lbl_int)
        ir.add_widget(lbl_int)
        self.tm = TextInput(text=str(self.cfg.get('interval','600')), hint_text="秒", multiline=False,
                            font_size=sp(12), background_color=C_CRD, foreground_color=C_TXT,
                            size_hint_y=None, height=dp(40), input_filter='int', size_hint_x=0.12)
        _f(self.tm)
        ir.add_widget(self.tm)
        self._cd_label = Label(text="", font_size=sp(13), color=C_WARN, bold=True,
                               halign='right', valign='middle', size_hint_x=0.35)
        self._cd_label.bind(size=lambda w, v: setattr(w, 'text_size', (v[0], None)))
        _f(self._cd_label)
        ir.add_widget(self._cd_label)
        p.add_widget(ir)

        # progress + status
        self.pb = ProgressBar(value=0, size_hint_y=None, height=dp(10))
        p.add_widget(self.pb)
        self.st = L("就绪 ─ 配置 API 后扫描", 11, C_SUB)
        p.add_widget(self.st)

        # results log (scrollable, fills most space)
        sv = ScrollView(size_hint_y=1, scroll_type=['bars','content'], bar_width=dp(6))
        self.rlog = Label(text="", font_size=sp(11), color=C_TXT, halign='left', valign='top',
                          size_hint_y=None, markup=True, text_size=(dp(370), None))
        _f(self.rlog)
        self.rlog.bind(width=lambda w, v: setattr(w, 'text_size', (v, None)))
        self.rlog.bind(texture_size=self.rlog.setter('size'))
        sv.add_widget(self.rlog)
        p.add_widget(sv)

        # three scan buttons at bottom
        br = BoxLayout(size_hint_y=None, height=dp(50), spacing=dp(4))
        br.add_widget(B("手动扫描", C_BTN, 14, cb=lambda x: self._scan(False)))
        br.add_widget(B("停止", C_RED, 14, cb=self._stop_scan))
        self.ab = B("定时扫描", C_WARN, 14, cb=self._start_auto)
        br.add_widget(self.ab)
        p.add_widget(br)
        self._update_cd()
        return p

    def _init_okx(self):
        if not self.okx:
            self.okx = OKXClient(api_key=self.cfg.get('api_key','').strip(), secret_key=self.cfg.get('secret_key','').strip(),
                                 passphrase=self.cfg.get('passphrase','').strip(), testnet=False,
                                 proxy_url=self.cfg.get('proxy_url','').strip() or None)

    def _scan(self, auto=False):
        if self.scanning: return
        k = self.cfg.get('api_key',''); s = self.cfg.get('secret_key','')
        if not k or not s: self._pop("提示", "请先在 API 设置中填写 Key"); return
        self.scanning = True
        self._cancel_flag = False
        self._status("连接 OKX..."); self.pb.value = 0
        t = time.strftime("%H:%M:%S")
        Clock.schedule_once(lambda dt: setattr(self.rlog, 'text', f"━━━ 扫描开始 {t} ━━━\n"))
        threading.Thread(target=self._scan_thread, daemon=True).start()

    def _stop_scan(self, btn):
        if self.scanning:
            self._cancel_flag = True
            self._status("正在停止...")
        else:
            self._status("没有正在进行的扫描")

    def _start_auto(self, btn):
        if self.auto_timer:
            self.auto_timer.cancel(); self.auto_timer = None
            self._auto_seconds = 0
            self.ab.text = "定时扫描"; self.ab.background_color = C_WARN
            self._cd_label.text = ""
            self._status("定时已停止")
        else:
            try: sec = max(10, int(self.tm.text))
            except ValueError: sec = 600
            self._auto_seconds = sec
            self.ab.text = "停止"; self.ab.background_color = C_RED
            self._save()
            self._status(f"定时 {sec}秒")
            self._scan(auto=True)
            self._update_cd()
            self.auto_timer = Clock.schedule_interval(self._cd_tick, 1)

    def _cd_tick(self, dt):
        if self.scanning:
            self._update_cd()
            return
        self._auto_seconds -= 1
        if self._auto_seconds <= 0:
            self._scan(auto=True)
            try: sec = max(10, int(self.tm.text))
            except ValueError: sec = 600
            self._auto_seconds = sec
        self._update_cd()

    def _update_cd(self):
        if not hasattr(self, '_cd_label') or not self._cd_label:
            return
        if self.scanning:
            self._cd_label.text = "扫描中..."
            self._cd_label.color = C_TAB
        elif self._auto_seconds > 0:
            self._cd_label.text = f"下次: {self._auto_seconds}s"
            self._cd_label.color = C_ACC if self._auto_seconds <= 10 else C_WARN
        else:
            self._cd_label.text = ""
            self._cd_label.color = C_WARN

    def _scan_thread(self):
        try:
            self._init_okx(); self._status("获取行情..."); self._prog(5)
            if self._cancel_flag: return
            r = self.okx.get_tickers('SWAP')
            if not isinstance(r, dict) or r.get('code') != '0':
                self._err(f"API错误: {r.get('msg','?') if isinstance(r,dict) else '连接失败'}"); return
            ticks = r.get('data', [])
            swaps = [t for t in ticks if t.get('instId','').endswith('-USDT-SWAP')]
            active = sorted([t for t in swaps if float(t.get('volCcyQuote') or t.get('vol24h') or 0) > 5000000],
                            key=lambda t: float(t.get('volCcyQuote') or t.get('vol24h') or 0), reverse=True)[:30]
            self._prog(10)
            selected = list(self._selected_strat_names)
            total_found = 0

            for si, strat_name in enumerate(selected):
                if self._cancel_flag:
                    self._status(f"已取消"); self._prog(0); return
                scanner = self._load_strat_scanner(strat_name)
                if not scanner:
                    self._add_log(f"⚠ 策略加载失败: {strat_name}")
                    continue
                self.scanner = scanner
                self._status(f"[{si+1}/{len(selected)}] {strat_name} 扫描中..."); self._prog(10)
                found = 0
                for i, t in enumerate(active):
                    if self._cancel_flag:
                        self._status(f"已取消 (分析 {total_found} 个)"); self._prog(0); return
                    iid = t['instId']
                    pct = 10 + int(85 * (i + 1) / len(active))
                    self._prog(pct)
                    self._status(f"[{si+1}/{len(selected)}] {iid} [{strat_name[:8]}]")
                    try:
                        kls = {}
                        for bar in ['1D', '1H', '15m', '3m']:
                            if self._cancel_flag: return
                            rr = self.okx.get_kline(iid, bar=bar, limit=200)
                            if isinstance(rr, dict) and rr.get('code') == '0' and rr.get('data'):
                                kls[bar] = rr['data']
                        if not kls.get('1D') or not kls.get('1H'): continue
                        sym = ScannerSymbol(inst_id=iid, last_price=float(t.get('last', 0)),
                                            volume_24h=float(t.get('volCcyQuote') or t.get('vol24h') or 0),
                                            extra_data={'klines': kls})
                        try:
                            res = scanner.scan_symbol(sym)
                            if isinstance(res, dict):
                                if res.get('passed', False) or res.get('score', 0) >= 60:
                                    found += 1
                                    self._add_res(res, strat_name)
                        except Exception as e2:
                            self._add_log(f"{iid} 分析出错: {e2}")
                    except Exception as e1:
                        self._add_log(f"{iid} 获取数据失败"); continue
                    time.sleep(0.15)
                self._add_log(f"━ {strat_name}: {found} 个机会")
                total_found += found

            self._status(f"完成！{total_found} 个机会"); self._prog(100)
            if total_found == 0:
                Clock.schedule_once(lambda dt: self._add_log("未发现符合条件的交易机会"))
        except Exception as e: self._err(f"扫描失败: {e}")
        finally: self.scanning = False

    def _add_log(self, msg):
        def _f(dt):
            self.rlog.text += msg + "\n"
        Clock.schedule_once(_f)

    def _add_res(self, r, strat_name=None):
        r['scan_time'] = time.strftime("%H:%M:%S")
        r['strategy'] = strat_name or self._cur_strat_name
        self.pool.append(r)
        def _f(dt):
            d = r.get('direction','NEUTRAL')
            arrow = "↑" if d == 'LONG' else ("↓" if d == 'SHORT' else "→")
            sigs = ' | '.join(r.get('signals',[])[:3]) or '无信号'
            line = f"[b]{r.get('symbol','?')}[/b] {arrow}{d} {r.get('score',0):.0f}分  {sigs}"
            self.rlog.text += line + "\n"
            self._refresh_pool_display()
        Clock.schedule_once(_f)

    def _list_strats(self):
        names = []
        for k, v in self._smap.items():
            if os.path.exists(os.path.join(self.dir, 'strategies', k+'.py')):
                names.append(v)
        return names or ['OKX小时线波段共振策略']

    def _load_strat(self, btn):
        """Load selected strategy from spinner"""
        name = self.ssp.text
        if not name: return
        rev = {v:k for k,v in self._smap.items()}
        fname = rev.get(name, name)
        fpath = os.path.join(self.dir, 'strategies', fname+'.py')
        self._do_load_file(fpath, fname+'.py')

    def _load_from_file(self, btn):
        from kivy.uix.filechooser import FileChooserListView

        c = BoxLayout(orientation='vertical', padding=dp(6), spacing=dp(4))
        c.add_widget(L("浏览选择策略文件 (.py)", 13, C_TAB, True))

        # start path: internal storage on Android, strategy dir on desktop
        start = self.dir
        for rp in ['/sdcard', '/storage/emulated/0', os.path.expanduser('~')]:
            if os.path.isdir(rp):
                start = rp; break

        fc = FileChooserListView(path=start, filters=['*.py'], size_hint_y=1)
        c.add_widget(fc)

        pp = Popup(title="加载策略文件", content=c, size_hint=(0.92, 0.78),
                   background_color=(0.12, 0.12, 0.15, 0.97), separator_color=C_TAB, auto_dismiss=False)

        br = BoxLayout(size_hint_y=None, height=dp(44), spacing=dp(6))
        br.add_widget(B("取消", (0.25, 0.25, 0.30, 1), 13, cb=pp.dismiss))
        br.add_widget(B("加载选中", C_BTN, 13,
                        cb=lambda x: self._on_file_selected(fc.selection, pp)))
        c.add_widget(br)
        pp.open()

    def _on_file_selected(self, selection, popup):
        if not selection:
            self._pop("提示", "请先选择一个文件"); return
        path = selection[0]
        if not path.endswith('.py'):
            self._pop("提示", "请选择 .py 策略文件"); return
        popup.dismiss()
        self._do_load_file(path, os.path.basename(path))

    def _show_strat_popup(self, btn):
        """Show popup with multi-select strategies for batch scanning"""
        names = self._list_strats()
        if not names:
            self._pop("提示", "无可用策略"); return

        c = BoxLayout(orientation='vertical', padding=dp(8), spacing=dp(4))
        c.add_widget(L("多选策略（依次扫描）", 14, C_TAB, True))
        sv = ScrollView(scroll_type=['bars','content'], bar_width=dp(6))
        flist = BoxLayout(orientation='vertical', size_hint_y=None, spacing=dp(2))
        flist.bind(minimum_height=flist.setter('height'))

        pp = Popup(title="策略列表", content=c, size_hint=(0.88, 0.65),
                   background_color=(0.12, 0.12, 0.15, 0.97), separator_color=C_TAB, auto_dismiss=False)

        for name in names:
            is_selected = name in self._selected_strat_names
            bg = C_TAB if is_selected else C_CRD
            btn_text = f"✓ {name}" if is_selected else f"  {name}"
            b = Button(text=btn_text, font_size=sp(13), color=C_TXT,
                       background_color=bg, background_normal='',
                       size_hint_y=None, height=dp(46), halign='left')
            b.bind(size=b.setter('text_size'))
            _f(b)
            b.bind(on_release=lambda x, n=name, btn=None: self._toggle_strat(n, pp))
            flist.add_widget(b)
        sv.add_widget(flist)
        c.add_widget(sv)

        # bottom row
        br = BoxLayout(size_hint_y=None, height=dp(44), spacing=dp(4))
        br.add_widget(B("全选", (0.20, 0.20, 0.25, 1), 12, cb=lambda x: self._sel_all_strat(pp)))
        br.add_widget(B("清空", (0.20, 0.20, 0.25, 1), 12, cb=lambda x: self._sel_none_strat(pp)))
        br.add_widget(B("确定", C_BTN, 13, cb=lambda x: self._confirm_strats(pp)))
        c.add_widget(br)
        pp.open()

    def _toggle_strat(self, name, popup):
        if name in self._selected_strat_names:
            self._selected_strat_names.remove(name)
        else:
            self._selected_strat_names.append(name)
        if hasattr(self, '_strat_btn'):
            n = len(self._selected_strat_names)
            self._strat_btn.text = f"策略({n}): {self._selected_strat_names[0]}" if n == 1 else f"策略: {n}个已选"
        popup.dismiss()
        self._show_strat_popup(None)

    def _sel_all_strat(self, popup):
        self._selected_strat_names = list(self._list_strats())
        popup.dismiss()
        self._show_strat_popup(None)

    def _sel_none_strat(self, popup):
        self._selected_strat_names = []
        popup.dismiss()
        self._show_strat_popup(None)

    def _confirm_strats(self, popup):
        popup.dismiss()
        if not self._selected_strat_names:
            self._pop("提示", "请至少选择一个策略")
            return
        self._strat_btn.text = f"策略: {len(self._selected_strat_names)}个已选"
        self._save()
        self._load_strat_scanner(self._selected_strat_names[0])

    def _load_strat_scanner(self, name):
        """Load and cache scanner class for a given display name. Returns scanner instance."""
        if name in self._strat_scanners:
            return self._strat_scanners[name]
        rev = {v: k for k, v in self._smap.items()}
        fname = rev.get(name, name)
        fpath = os.path.join(self.dir, 'strategies', fname + '.py')
        try:
            spec = importlib.util.spec_from_file_location(fname, fpath)
            if not spec: return None
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            for cls_name in ['OKXHourSwingScanner', 'XiaoYueBollMacdScanner',
                             'TrendSqueezeBreakoutScannerV3', 'AICrossSectionDualFactorComboScanner',
                             'ThreeMinuteMultiTimeframePullbackStrategy']:
                if hasattr(mod, cls_name):
                    scanner = getattr(mod, cls_name)()
                    self._strat_scanners[name] = scanner
                    return scanner
        except Exception:
            pass
        return None

    def _do_load_file(self, path, filename, popup=None, silent=False):
        if hasattr(self, '_file_popup') and self._file_popup:
            self._file_popup.dismiss()
        if popup: popup.dismiss()
        try:
            name = filename.replace('.py', '')
            spec = importlib.util.spec_from_file_location(name, path)
            if not spec: self._pop("错误", "无法解析文件"); return
            mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
            scanner_cls = None
            for cls_name in ['OKXHourSwingScanner', 'XiaoYueBollMacdScanner',
                             'TrendSqueezeBreakoutScannerV3', 'AICrossSectionDualFactorComboScanner',
                             'ThreeMinuteMultiTimeframePullbackStrategy']:
                if hasattr(mod, cls_name):
                    scanner_cls = getattr(mod, cls_name); break
            if scanner_cls:
                self.scanner = scanner_cls()
                self._cur_strat_name = filename.replace('.py','')
                self._selected_strat_names = [self._cur_strat_name]
                self._strat_scanners[self._cur_strat_name] = self.scanner
                self._strat_btn.text = f"策略(1): {self._cur_strat_name}"
                self._save()
                if not silent: self._pop("成功", f"已加载: {filename}")
            else:
                if not silent: self._pop("提示", f"文件中未找到策略类: {filename}")
        except Exception as e:
            if not silent: self._pop("错误", f"加载失败: {e}")

    # ═══════════════════ Pool ═══════════════════
    def _pool_page(self):
        p = BoxLayout(orientation='vertical', padding=dp(6), spacing=dp(4))
        p.add_widget(L("交易池 - 扫描结果汇集", 15, C_TAB, True))
        self._pool_count = L("共 0 条", 12, C_SUB)
        p.add_widget(self._pool_count)

        # scrollable item list
        sv = ScrollView(size_hint_y=1, scroll_type=['bars','content'], bar_width=dp(6))
        self._pool_list = BoxLayout(orientation='vertical', size_hint_y=None, spacing=dp(2))
        self._pool_list.bind(minimum_height=self._pool_list.setter('height'))
        sv.add_widget(self._pool_list)
        p.add_widget(sv)

        # bottom buttons
        bar = BoxLayout(size_hint_y=None, height=dp(40), spacing=dp(6))
        bar.add_widget(B("清空全部", (0.30, 0.30, 0.35, 1), 12, cb=self._clr_pool))
        self._pool_mon_btn = B("启用监控", (0.20, 0.55, 0.25, 1), 12, cb=self._toggle_pool_monitor)
        bar.add_widget(self._pool_mon_btn)
        p.add_widget(bar)

        self._refresh_pool_display()
        return p

    def _clr_pool(self, btn):
        self.pool = []
        if self._pool_monitoring:
            self._pool_monitoring = False
            if self._pool_monitor_timer:
                self._pool_monitor_timer.cancel()
                self._pool_monitor_timer = None
            self._pool_mon_btn.text = "启用监控"
            self._pool_mon_btn.background_color = (0.20, 0.55, 0.25, 1)
        self._pool_sel_label.text = "未选中"
        self._pool_selected = None
        self._refresh_pool_display()

    def _remove_pool_selected(self, btn):
        if self._pool_selected is not None and 0 <= self._pool_selected < len(self.pool):
            del self.pool[self._pool_selected]
            self._pool_selected = None
            self._pool_sel_label.text = "未选中"
            self._refresh_pool_display()

    def _refresh_pool_display(self):
        if not hasattr(self, '_pool_list') or not self._pool_list:
            return
        self._pool_list.clear_widgets()
        if not self.pool:
            self._pool_list.add_widget(L("暂无数据", 12, C_SUB))
            if hasattr(self, '_pool_count') and self._pool_count:
                self._pool_count.text = "共 0 条"
            return
        if hasattr(self, '_pool_count') and self._pool_count:
            self._pool_count.text = f"共 {len(self.pool)} 条"

        # table header
        hdr = BoxLayout(size_hint_y=None, height=dp(28), spacing=dp(2))
        cols = [("时间", 0.12), ("交易对", 0.16), ("方向", 0.10), ("评分", 0.09), ("策略来源", 0.16),
                ("日", 0.09), ("时", 0.09), ("分", 0.09), ("操作", 0.10)]
        for title, w in cols:
            lbl = Label(text=title, font_size=sp(11), color=C_TAB, bold=True,
                        halign='center', valign='middle', size_hint_x=w)
            _f(lbl)
            hdr.add_widget(lbl)
        self._pool_list.add_widget(hdr)

        # data rows
        for idx, r in enumerate(reversed(self.pool)):
            real_idx = len(self.pool) - 1 - idx
            t = r.get('scan_time', '-')
            sym = r.get('symbol', '?').replace('-USDT-SWAP', '')
            d = r.get('direction', '-')
            score = f"{r.get('score',0):.0f}"
            strategy_src = r.get('strategy', self._cur_strat_name)[:12]
            outlook = r.get('rating', d)[:12]
            dcolor = C_ACC if d == 'LONG' else (C_RED if d == 'SHORT' else C_SUB)
            arrow = "↑" if d == 'LONG' else ("↓" if d == 'SHORT' else "→")

            # monitoring trend indicators
            td = r.get('trend_d', '') or '→'
            th = r.get('trend_h', '') or '→'
            tm = r.get('trend_m', '') or '→'
            td_color = C_ACC if '↑' in td else (C_RED if '↓' in td else C_SUB)
            th_color = C_ACC if '↑' in th else (C_RED if '↓' in th else C_SUB)
            tm_color = C_ACC if '↑' in tm else (C_RED if '↓' in tm else C_SUB)

            row = BoxLayout(size_hint_y=None, height=dp(30), spacing=dp(2))
            vals = [t, sym, f"{arrow}{d}", score, strategy_src, td, th, tm]
            for i, (val, (_, w)) in enumerate(zip(vals, cols[:-1])):
                if i == 2:      color = dcolor
                elif i == 5:    color = td_color
                elif i == 6:    color = th_color
                elif i == 7:    color = tm_color
                else:           color = C_TXT
                lbl = Label(text=val, font_size=sp(10), color=color,
                           halign='center', valign='middle', size_hint_x=w)
                _f(lbl)
                row.add_widget(lbl)
            # delete button
            del_btn = Button(text="✕", font_size=sp(10), color=C_RED,
                            background_color=C_CRD, background_normal='',
                            size_hint_x=cols[-1][1])
            _f(del_btn)
            del_btn.bind(on_release=lambda x, i=real_idx: self._del_pool_item(i))
            row.add_widget(del_btn)
            self._pool_list.add_widget(row)

    def _del_pool_item(self, idx):
        if 0 <= idx < len(self.pool):
            del self.pool[idx]
            self._refresh_pool_display()

    def _toggle_pool_monitor(self, btn):
        if self._pool_monitoring:
            self._pool_monitoring = False
            if self._pool_monitor_timer:
                self._pool_monitor_timer.cancel()
                self._pool_monitor_timer = None
            self._pool_mon_btn.text = "启用监控"
            self._pool_mon_btn.background_color = (0.20, 0.55, 0.25, 1)
            self._status("池监控已停止")
        else:
            if not self.pool:
                self._pop("提示", "池子为空，请先扫描"); return
            self._pool_monitoring = True
            self._pool_mon_btn.text = "停止监控"
            self._pool_mon_btn.background_color = C_RED
            self._status("池监控已启动")
            self._pool_monitor_timer = Clock.schedule_interval(
                lambda dt: threading.Thread(target=self._pool_monitor_thread, daemon=True).start(), 30)

    def _pool_monitor_thread(self):
        import pandas as pd
        if not self._pool_monitoring or not self.pool:
            return
        if not self._monitor_lock.acquire(blocking=False):
            return  # previous monitor still running, skip this round
        try:
            self._init_okx()
            symbols = list(set(r.get('symbol', '') for r in self.pool if r.get('symbol')))
            if not symbols:
                return

            for sym in symbols:
                if not self._pool_monitoring:
                    return
                inst_id = sym if '-USDT-SWAP' in sym else f"{sym}-USDT-SWAP"
                try:
                    kls = {}
                    for bar, limit in [('1D', 60), ('1H', 60), ('15m', 60)]:
                        rr = self.okx.get_kline(inst_id, bar=bar, limit=limit)
                        if isinstance(rr, dict) and rr.get('code') == '0' and rr.get('data'):
                            kls[bar] = rr['data']
                    trend_d = trend_h = trend_m = '→'
                    if kls.get('1D') and len(kls['1D']) >= 26:
                        d1 = pd.DataFrame([r[:6] for r in kls['1D'] if len(r) >= 6],
                                           columns=['ts','o','h','l','c','vol']).astype(float)
                        if len(d1) >= 26:
                            e12 = d1['c'].ewm(span=12, adjust=False).mean().iloc[-1]
                            e26 = d1['c'].ewm(span=26, adjust=False).mean().iloc[-1]
                            p = d1['c'].iloc[-1]
                            trend_d = '↑' if p > e12 > e26 else ('↓' if p < e12 < e26 else '→')
                    if kls.get('1H') and len(kls['1H']) >= 26:
                        h1 = pd.DataFrame([r[:6] for r in kls['1H'] if len(r) >= 6],
                                           columns=['ts','o','h','l','c','vol']).astype(float)
                        if len(h1) >= 26:
                            e12 = h1['c'].ewm(span=12, adjust=False).mean().iloc[-1]
                            e26 = h1['c'].ewm(span=26, adjust=False).mean().iloc[-1]
                            p = h1['c'].iloc[-1]
                            trend_h = '↑' if p > e12 > e26 else ('↓' if p < e12 < e26 else '→')
                    if kls.get('15m') and len(kls['15m']) >= 26:
                        m15 = pd.DataFrame([r[:6] for r in kls['15m'] if len(r) >= 6],
                                            columns=['ts','o','h','l','c','vol']).astype(float)
                        if len(m15) >= 26:
                            e12 = m15['c'].ewm(span=12, adjust=False).mean().iloc[-1]
                            e26 = m15['c'].ewm(span=26, adjust=False).mean().iloc[-1]
                            p = m15['c'].iloc[-1]
                            trend_m = '↑' if p > e12 > e26 else ('↓' if p < e12 < e26 else '→')
                    for r in self.pool:
                        if r.get('symbol', '').replace('-USDT-SWAP', '') == sym.replace('-USDT-SWAP', ''):
                            r['trend_d'] = trend_d
                            r['trend_h'] = trend_h
                            r['trend_m'] = trend_m
                    time.sleep(0.3)
                except Exception:
                    continue
            Clock.schedule_once(lambda dt: self._refresh_pool_display())
        except Exception:
            pass
        finally:
            self._monitor_lock.release()

    # ═══════════════════ Trading Monitor ═══════════════════
    def _trade_mon_page(self):
        p = BoxLayout(orientation='vertical', padding=dp(8), spacing=dp(4))
        p.add_widget(L("交易监控", 15, C_TAB, True))

        # account summary
        self._tmon_balance = L("账户: ----", 14, C_TXT, True)
        p.add_widget(self._tmon_balance)
        self._tmon_equity = L("权益: ----", 13, C_TXT)
        p.add_widget(self._tmon_equity)
        self._tmon_pnl = L("未实现盈亏: ----", 13, C_TXT)
        p.add_widget(self._tmon_pnl)

        br = BoxLayout(size_hint_y=None, height=dp(44), spacing=dp(6))
        br.add_widget(B("刷新", C_BTN, 13, cb=lambda x: self._refresh_trade_mon()))
        br.add_widget(B("持仓列表", (0.25, 0.45, 0.30, 1), 13, cb=lambda x: self._fetch_positions()))
        p.add_widget(br)

        # positions table
        sv = ScrollView(size_hint_y=1, scroll_type=['bars','content'], bar_width=dp(6))
        self._tmon_list = BoxLayout(orientation='vertical', size_hint_y=None, spacing=dp(2))
        self._tmon_list.bind(minimum_height=self._tmon_list.setter('height'))
        sv.add_widget(self._tmon_list)
        p.add_widget(sv)
        self._restore_tmon_display()
        return p

    def _refresh_trade_mon(self):
        k = self.cfg.get('api_key',''); s = self.cfg.get('secret_key','')
        if not k or not s: self._pop("提示", "请先在 API 设置中填写 Key"); return
        self._status("刷新账户...")
        threading.Thread(target=self._refresh_trade_thread, daemon=True).start()

    def _refresh_trade_thread(self):
        try:
            self._init_okx()
            # balance
            bal = self.okx.get_balance()
            if isinstance(bal, dict) and bal.get('code') == '0':
                data = bal.get('data', [])
                if data and len(data) > 0:
                    d = data[0]
                    eq = d.get('totalEq', '0')
                    bal_usdt = d.get('details', [])
                    usdt = next((x for x in bal_usdt if x.get('ccy') == 'USDT'), {})
                    avail = usdt.get('availEq', '0')
                    self._tmon_balance_text = f"可用: {avail} USDT"
                    self._tmon_equity_text = f"总权益: {eq} USDT"
                else:
                    self._tmon_balance_text = "无余额数据"
                    self._tmon_equity_text = "权益: ----"
            else:
                self._tmon_balance_text = "获取余额失败"
                self._tmon_equity_text = "权益: ----"
            Clock.schedule_once(lambda dt: self._restore_tmon_display())

            # positions
            self._fetch_positions_thread()
            Clock.schedule_once(lambda dt: self._status("已刷新"))
        except Exception as e:
            Clock.schedule_once(lambda dt: self._pop("错误", str(e)))

    def _fetch_positions(self):
        k = self.cfg.get('api_key',''); s = self.cfg.get('secret_key','')
        if not k or not s: self._pop("提示", "请先填写 API Key"); return
        self._status("获取持仓...")
        threading.Thread(target=self._fetch_positions_thread, daemon=True).start()

    def _fetch_positions_thread(self):
        try:
            self._init_okx()
            pos = self.okx.get_positions()
            def update(dt):
                if isinstance(pos, dict) and pos.get('code') == '0':
                    self.positions_data = pos.get('data', [])
                    self._refresh_positions_display()
                else:
                    self.positions_data = []
                    self._refresh_positions_display()
            Clock.schedule_once(update)
        except Exception as e:
            self.positions_data = []
            Clock.schedule_once(lambda dt: self._refresh_positions_display())
            Clock.schedule_once(lambda dt: self._pop("错误", str(e)))

    def _refresh_positions_display(self):
        if not hasattr(self, '_tmon_list') or not self._tmon_list:
            return
        self._tmon_list.clear_widgets()
        if not self.positions_data:
            self._tmon_list.add_widget(L("暂无持仓", 12, C_SUB))
            return

        # table header
        hdr = BoxLayout(size_hint_y=None, height=dp(28), spacing=dp(2))
        cols = [("交易对", 0.20), ("方向", 0.10), ("张数", 0.10), ("杠杆", 0.08),
                ("均价", 0.15), ("标记价", 0.15), ("未实现盈亏", 0.22)]
        for title, w in cols:
            lbl = Label(text=title, font_size=sp(11), color=C_TAB, bold=True,
                        halign='center', valign='middle', size_hint_x=w)
            _f(lbl)
            hdr.add_widget(lbl)
        self._tmon_list.add_widget(hdr)

        # data rows
        for p in self.positions_data:
            iid = p.get('instId', '?').replace('-USDT-SWAP', '').replace('-USDT', '')
            side = p.get('posSide', 'long')
            pos_qty = float(p.get('pos', 0))
            # 单向持仓模式 posSide='net'，用 pos 正负判断方向
            if side == 'net':
                is_long = pos_qty > 0
            else:
                is_long = side == 'long'
            side_cn = "多" if is_long else "空"
            arrow = "↑" if is_long else "↓"
            qty = str(abs(int(pos_qty)))
            lev = p.get('lever', '1')
            avg_px = p.get('avgPx', '0')
            mark_px = p.get('markPx', '0')
            pnl = float(p.get('upl', 0))
            sign = "+" if pnl >= 0 else ""
            pnl_color = C_ACC if pnl >= 0 else C_RED
            side_color = C_ACC if is_long else C_RED

            row = BoxLayout(size_hint_y=None, height=dp(32), spacing=dp(2))
            vals = [iid, f"{arrow}{side_cn}", qty, f"{lev}x", avg_px, mark_px, f"{sign}{pnl:.2f}"]
            for i, (val, (_, w)) in enumerate(zip(vals, cols)):
                color = side_color if i == 1 else (pnl_color if i == 6 else C_TXT)
                lbl = Label(text=val, font_size=sp(11), color=color,
                           halign='center', valign='middle', size_hint_x=w)
                _f(lbl)
                row.add_widget(lbl)
            self._tmon_list.add_widget(row)

    def _restore_tmon_display(self):
        """Restore cached account info and positions to widgets (survives tab switches)"""
        if hasattr(self, '_tmon_balance') and self._tmon_balance:
            self._tmon_balance.text = getattr(self, '_tmon_balance_text', "账户: ----")
        if hasattr(self, '_tmon_equity') and self._tmon_equity:
            self._tmon_equity.text = getattr(self, '_tmon_equity_text', "权益: ----")
        self._refresh_positions_display()

    # ═══════════════════ Auto Trade ═══════════════════
    def _auto_trade_page(self):
        p = BoxLayout(orientation='vertical', padding=dp(6), spacing=dp(4))

        # header
        p.add_widget(L("自动交易", 15, C_TAB, True))
        self._at_balance = L("账户: ----", 13, C_TXT, True)
        p.add_widget(self._at_balance)
        self._at_equity = L("权益: ----", 12, C_SUB)
        p.add_widget(self._at_equity)

        # settings row
        sr = BoxLayout(size_hint_y=None, height=dp(44), spacing=dp(4))
        sr.add_widget(L("保证金", 11, C_SUB))
        self._at_margin = TextInput(text=self.cfg.get('at_margin','50'), multiline=False,
                                     font_size=sp(12), background_color=C_CRD, foreground_color=C_TXT,
                                     size_hint_y=None, height=dp(40), input_filter='int', size_hint_x=0.12)
        _f(self._at_margin); sr.add_widget(self._at_margin)
        sr.add_widget(L("杠杆", 11, C_SUB))
        self._at_lever = TextInput(text=self.cfg.get('at_lever','3'), multiline=False,
                                    font_size=sp(12), background_color=C_CRD, foreground_color=C_TXT,
                                    size_hint_y=None, height=dp(40), input_filter='int', size_hint_x=0.10)
        _f(self._at_lever); sr.add_widget(self._at_lever)
        sr.add_widget(B("刷新账户", (0.20,0.45,0.30,1), 12, cb=lambda x: self._at_refresh_account()))
        p.add_widget(sr)

        # pool candidate label
        self._at_cand_count = L("池候选: 0", 12, C_SUB)
        p.add_widget(self._at_cand_count)

        # scrollable candidate + position list
        sv = ScrollView(size_hint_y=1, scroll_type=['bars','content'], bar_width=dp(6))
        self._at_list = BoxLayout(orientation='vertical', size_hint_y=None, spacing=dp(2))
        self._at_list.bind(minimum_height=self._at_list.setter('height'))
        sv.add_widget(self._at_list)
        p.add_widget(sv)

        # bottom buttons
        bar = BoxLayout(size_hint_y=None, height=dp(40), spacing=dp(6))
        bar.add_widget(B("刷新候选", (0.22,0.45,0.30,1), 12, cb=lambda x: self._at_refresh_candidates()))
        bar.add_widget(B("平全部仓", C_RED, 12, cb=lambda x: self._at_close_all()))
        p.add_widget(bar)

        self._at_refresh_candidates()
        self._at_restore_display()
        return p

    def _at_refresh_account(self):
        self._status("获取账户...")
        threading.Thread(target=self._at_refresh_account_thread, daemon=True).start()

    def _at_refresh_account_thread(self):
        try:
            self._init_okx()
            bal = self.okx.get_balance()
            if isinstance(bal, dict) and bal.get('code') == '0':
                data = bal.get('data', [])
                if data and len(data) > 0:
                    d = data[0]
                    eq = d.get('totalEq', '0')
                    bals = d.get('details', [])
                    usdt = next((x for x in bals if x.get('ccy') == 'USDT'), {})
                    avail = usdt.get('availEq', '0')
                    self._at_balance_cache = f"可用: {avail} USDT"
                    self._at_equity_cache = f"总权益: {eq} USDT"
                else:
                    self._at_balance_cache = "无余额数据"
                    self._at_equity_cache = "权益: ----"
            else:
                self._at_balance_cache = "获取失败"
                self._at_equity_cache = ""
            Clock.schedule_once(lambda dt: self._at_restore_display())
        except Exception as e:
            Clock.schedule_once(lambda dt: self._pop("错误", str(e)))

    def _at_restore_display(self):
        if hasattr(self, '_at_balance') and self._at_balance:
            self._at_balance.text = getattr(self, '_at_balance_cache', "账户: ----")
        if hasattr(self, '_at_equity') and self._at_equity:
            self._at_equity.text = getattr(self, '_at_equity_cache', "权益: ----")

    def _at_refresh_candidates(self):
        if not hasattr(self, '_at_list') or not self._at_list:
            return
        self._at_list.clear_widgets()

        # filter pool items with monitoring data and direction
        candidates = [r for r in self.pool if r.get('trend_d') in ('↑','↓')]
        if hasattr(self, '_at_cand_count') and self._at_cand_count:
            self._at_cand_count.text = f"池候选: {len(candidates)}/{len(self.pool)}"

        if not candidates:
            self._at_list.add_widget(L("无候选 — 请先扫描并启用池监控", 12, C_SUB))
            return

        # header
        hdr = BoxLayout(size_hint_y=None, height=dp(28), spacing=dp(2))
        cols = [("交易对", 0.16), ("方向", 0.10), ("日/时/分", 0.18), ("评分", 0.09),
                ("策略", 0.14), ("保证金", 0.13), ("开仓", 0.10), ("杠杆", 0.10)]
        for title, w in cols:
            lbl = Label(text=title, font_size=sp(10), color=C_TAB, bold=True,
                        halign='center', valign='middle', size_hint_x=w)
            _f(lbl)
            hdr.add_widget(lbl)
        self._at_list.add_widget(hdr)

        for r in candidates:
            sym = r.get('symbol', '?').replace('-USDT-SWAP', '')
            d = r.get('direction', '-')
            dcolor = C_ACC if d == 'LONG' else (C_RED if d == 'SHORT' else C_SUB)
            arrow = "↑" if d == 'LONG' else ("↓" if d == 'SHORT' else "→")
            td = r.get('trend_d', '→'); th = r.get('trend_h', '→'); tm = r.get('trend_m', '→')
            score = f"{r.get('score',0):.0f}"
            strat = r.get('strategy', '')[:10]

            row = BoxLayout(size_hint_y=None, height=dp(34), spacing=dp(2))
            vals = [sym, f"{arrow}{d}", f"{td} {th} {tm}", score, strat]
            for i, (val, (_, w)) in enumerate(zip(vals, cols)):
                if i == 1: color = dcolor
                elif i == 2: color = C_ACC if '↑' in td+th+tm else C_SUB
                else: color = C_TXT
                lbl = Label(text=val, font_size=sp(10), color=color,
                           halign='center', valign='middle', size_hint_x=w)
                _f(lbl)
                row.add_widget(lbl)

            # margin input
            mi = TextInput(text=self._at_margin.text, multiline=False,
                           font_size=sp(10), background_color=C_CRD, foreground_color=C_TXT,
                           size_hint_y=None, height=dp(34), input_filter='int', size_hint_x=cols[5][1])
            _f(mi); row.add_widget(mi)

            # open button — long
            op = Button(text="多", font_size=sp(10), color=C_TXT,
                        background_color=C_ACC, background_normal='', size_hint_x=cols[6][1])
            _f(op)
            op.bind(on_release=lambda x, sym=sym, d='long', mi=mi: self._at_open(sym, d, mi.text))
            row.add_widget(op)

            # lever input
            li = TextInput(text=self._at_lever.text, multiline=False,
                           font_size=sp(10), background_color=C_CRD, foreground_color=C_TXT,
                           size_hint_y=None, height=dp(34), input_filter='int', size_hint_x=cols[7][1])
            _f(li); row.add_widget(li)
            self._at_list.add_widget(row)

    def _at_open(self, sym, direction, margin_text):
        k = self.cfg.get('api_key',''); s = self.cfg.get('secret_key','')
        if not k or not s: self._pop("提示", "请先配置 API"); return
        try: margin_usdt = max(1, int(margin_text))
        except: margin_usdt = 50
        inst_id = sym if '-USDT-SWAP' in sym else f"{sym}-USDT-SWAP"
        threading.Thread(target=self._at_open_thread, args=(inst_id, direction, margin_usdt), daemon=True).start()

    def _at_open_thread(self, inst_id, direction, margin_usdt):
        try:
            self._init_okx()
            lever = self._at_lever.text
            td_mode = 'cross'

            # set leverage
            self.okx._request("POST", "/api/v5/account/set-leverage", data={
                "instId": inst_id, "lever": lever, "mgnMode": td_mode
            })

            # get ticker for price
            tk = self.okx.get_ticker(inst_id)
            price = 0
            if isinstance(tk, dict) and tk.get('code') == '0' and tk.get('data'):
                price = float(tk['data'][0].get('last', 0))

            if price <= 0:
                Clock.schedule_once(lambda dt: self._pop("错误", "获取价格失败"))
                return

            # calculate size
            sz = margin_usdt * int(lever) / price

            side = 'buy' if direction == 'long' else 'sell'
            posSide = 'long' if direction == 'long' else 'short'

            r = self.okx.place_order(instId=inst_id, tdMode=td_mode, side=side,
                                     ordType='market', sz=str(round(sz, 0)), posSide=posSide)

            if isinstance(r, dict) and r.get('code') == '0':
                Clock.schedule_once(lambda dt: self._pop("开仓成功", f"{inst_id} {direction} {sz:.0f}张"))
                Clock.schedule_once(lambda dt: self._at_refresh_candidates())
            else:
                msg = r.get('msg', '?') if isinstance(r, dict) else '失败'
                Clock.schedule_once(lambda dt: self._pop("开仓失败", msg))
        except Exception as e:
            Clock.schedule_once(lambda dt: self._pop("错误", str(e)))

    def _at_close_all(self):
        k = self.cfg.get('api_key',''); s = self.cfg.get('secret_key','')
        if not k or not s: self._pop("提示", "请先配置 API"); return
        threading.Thread(target=self._at_close_all_thread, daemon=True).start()

    def _at_close_all_thread(self):
        try:
            self._init_okx()
            pos = self.okx.get_positions()
            if not isinstance(pos, dict) or pos.get('code') != '0':
                return
            data = pos.get('data', [])
            for p in data:
                inst_id = p.get('instId', '')
                pos_side = p.get('posSide', 'long')
                qty = p.get('pos', '0')
                if float(qty) <= 0: continue
                side = 'sell' if pos_side == 'long' else 'buy'
                # close by reversing the side with posSide
                side_close = 'sell' if pos_side == 'long' else 'buy'
                self.okx.place_order(instId=inst_id, tdMode='cross', side=side_close,
                                     ordType='market', sz=qty, posSide=pos_side, reduceOnly=True)
                time.sleep(0.3)
            Clock.schedule_once(lambda dt: self._status("平仓完成"))
        except Exception as e:
            Clock.schedule_once(lambda dt: self._pop("错误", str(e)))

    # ── helpers ──────────────────────────────────────────
    def _status(self, m):
        Clock.schedule_once(lambda dt: setattr(self.sl, 'text', m))
        Clock.schedule_once(lambda dt: setattr(self.st, 'text', m))

    def _prog(self, v): Clock.schedule_once(lambda dt: setattr(self.pb, 'value', v))
    def _err(self, m):
        def _f(dt): self._pop("错误", m); self._status(m)
        Clock.schedule_once(_f)

    def _pop(self, title, text):
        c = BoxLayout(orientation='vertical', padding=dp(12), spacing=dp(8))
        l = TextInput(text=text, font_size=sp(13), readonly=True,
                      background_color=(0,0,0,0), foreground_color=C_TXT, size_hint_y=1)
        _f(l); c.add_widget(l)
        b = Button(text="关闭", size_hint_y=None, height=dp(40),
                   background_color=C_BTN, color=C_TXT)
        _f(b); c.add_widget(b)
        pp = Popup(title=title, content=c, size_hint=(0.85, 0.45),
                   background_color=(0.15, 0.15, 0.18, 0.95), separator_color=C_TAB)
        b.bind(on_release=pp.dismiss); pp.open()

    def _load(self):
        try:
            if os.path.exists(self.cfg_path):
                with open(self.cfg_path, 'r', encoding='utf-8') as f: return json.load(f)
        except Exception: pass
        return {}

    def _save(self):
        try:
            os.makedirs(os.path.dirname(self.cfg_path), exist_ok=True)
            if hasattr(self, 'tm') and self.tm:
                self.cfg['interval'] = self.tm.text
            self.cfg['selected_strategies'] = list(self._selected_strat_names)
            with open(self.cfg_path, 'w', encoding='utf-8') as f:
                json.dump(self.cfg, f, ensure_ascii=False, indent=2)
            return True
        except Exception: return False

    def on_pause(self):
        if self.auto_timer: self.auto_timer.cancel(); self.auto_timer = None
        return True
    def on_resume(self): pass


if __name__ == '__main__':
    if not IMP_OK:
        class ErrApp(App):
            def build(self):
                r = BoxLayout(orientation='vertical', padding=dp(20), spacing=dp(10))
                l = Label(text=f"启动失败\n\n{IMP_ERR}"); _f(l); l.bind(size=l.setter('text_size'))
                s = ScrollView(); s.add_widget(l); r.add_widget(s)
                b = Button(text="退出", size_hint_y=None, height=dp(40),
                           background_color=C_RED, color=C_TXT); _f(b)
                b.bind(on_release=lambda x: sys.exit(0)); r.add_widget(b)
                return r
        ErrApp().run()
    else:
        App().run()
