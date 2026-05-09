"""
CryptoScanner Pro - Android Full Version
"""
import os, sys, json, threading, time, glob, traceback, importlib, warnings
warnings.filterwarnings('ignore')

from kivy.config import Config
Config.set('graphics', 'width', '390')
Config.set('graphics', 'height', '844')
Config.set('graphics', 'resizable', False)

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
        Window.minimum_width, Window.minimum_height = 320, 480
        Window.softinput_mode = 'below_target'
        self.dir = os.path.dirname(os.path.abspath(__file__))
        self.cfg_path = os.path.join(self.dir, "app_config.json")
        self.cfg = self._load()
        self.okx = None
        self.scanner = OKXHourSwingScanner()
        self.scanning = False
        self.pool = []
        self.monitors = []
        self.auto_timer = None

        root = BoxLayout(orientation='vertical')

        # title
        tb = BoxLayout(size_hint_y=None, height=dp(48))
        with tb.canvas.before: Color(*C_OFF); self._tbg = Rectangle(pos=tb.pos, size=tb.size)
        tb.bind(pos=lambda i,v: setattr(self._tbg,'pos',v), size=lambda i,v: setattr(self._tbg,'size',v))
        tb.add_widget(L("CryptoScanner Pro", 18, C_TAB, True))
        root.add_widget(tb)

        # tab bar
        self.tabs = ['交易扫描', '交易池', '监控池', '数据']
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
        self.content.add_widget([self._scan_page(), self._pool_page(), self._mon_page(), self._data_page()][btn.t])

    # ═══════════════════ API Popup ═══════════════════
    def _show_api_popup(self):
        c = BoxLayout(orientation='vertical', padding=dp(12), spacing=dp(8))
        c.add_widget(L("OKX API 配置", 16, C_TAB, True))
        c.add_widget(L("仅保存在本地，App重启自动加载", 12, C_SUB))

        ak = TI("API Key", self.cfg.get('api_key',''))
        sk = TI("Secret Key", self.cfg.get('secret_key',''), pw=True)
        ph = TI("Passphrase", self.cfg.get('passphrase',''), pw=True)
        px = TI("代理(可选)", self.cfg.get('proxy_url',''))
        c.add_widget(ak); c.add_widget(sk); c.add_widget(ph); c.add_widget(px)

        bb = BoxLayout(size_hint_y=None, height=dp(52), spacing=dp(8))
        bb.add_widget(B("测试连接", (0.40, 0.40, 0.25, 1), 13, cb=lambda x: self._test(ak.text, sk.text, ph.text, px.text)))
        bb.add_widget(B("保存", C_BTN, 13))
        c.add_widget(bb)

        pp = Popup(title="API 设置", content=c, size_hint=(0.9, 0.75),
                   background_color=(0.12, 0.12, 0.15, 0.97), separator_color=C_TAB, auto_dismiss=False)
        def save(_):
            self.cfg['api_key'] = ak.text; self.cfg['secret_key'] = sk.text
            self.cfg['passphrase'] = ph.text; self.cfg['proxy_url'] = px.text
            self._save()
            pp.dismiss()
            self._status("API 已保存")
        bb.children[0].bind(on_release=save)  # "保存" button
        pp.open()

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
        p = BoxLayout(orientation='vertical', padding=dp(8), spacing=dp(4))

        # settings row
        sr = BoxLayout(size_hint_y=None, height=dp(42), spacing=dp(6))
        sr.add_widget(B("API 设置", (0.30, 0.30, 0.35, 1), 12, cb=lambda x: self._show_api_popup()))
        sr.add_widget(B("保存配置", (0.20, 0.20, 0.25, 1), 12, cb=lambda x: self._pop("配置", "已保存" if self._save() else "失败")))
        p.add_widget(sr)

        # strategy
        sr2 = BoxLayout(size_hint_y=None, height=dp(42), spacing=dp(4))
        self.ssp = Spinner(text='OKX小时线波段共振策略', values=self._list_strats(),
                           size_hint_x=0.55, background_color=C_CRD, color=C_TXT, font_size=sp(12))
        _f(self.ssp)
        sr2.add_widget(self.ssp)
        sr2.add_widget(B("加载", C_BTN, 12, cb=self._load_strat))
        sr2.add_widget(B("从文件", (0.25, 0.45, 0.30, 1), 12, cb=self._load_from_file))
        p.add_widget(sr2)

        # timer + auto
        tr = BoxLayout(size_hint_y=None, height=dp(42), spacing=dp(6))
        self.tm = TextInput(text=str(self.cfg.get('interval','600')), hint_text="间隔秒", multiline=False,
                            font_size=sp(13), background_color=C_CRD, foreground_color=C_TXT,
                            size_hint_y=None, height=dp(42), input_filter='int', size_hint_x=0.35)
        _f(self.tm)
        tr.add_widget(self.tm)
        self.ab = B("启动定时扫描", C_WARN, 12, cb=self._start_auto)
        tr.add_widget(self.ab)
        p.add_widget(tr)

        self.pb = ProgressBar(value=0, size_hint_y=None, height=dp(12))
        p.add_widget(self.pb)
        self.st = L("就绪 ─ 配置 API 后扫描", 11, C_SUB)
        p.add_widget(self.st)

        # results (scrollable)
        sv = ScrollView(size_hint_y=1)
        self.rbox = BoxLayout(orientation='vertical', size_hint_y=None, spacing=dp(3))
        self.rbox.bind(minimum_height=self.rbox.setter('height'))
        sv.add_widget(self.rbox)
        p.add_widget(sv)

        # bottom buttons
        br = BoxLayout(size_hint_y=None, height=dp(50), spacing=dp(4))
        br.add_widget(B("手动扫描", C_BTN, 13, cb=lambda x: self._scan(False)))
        br.add_widget(B("停止扫描", C_RED, 13, cb=self._stop_scan))
        br.add_widget(B("定时扫描", C_WARN, 13, cb=self._start_auto))
        p.add_widget(br)
        return p

    def _init_okx(self):
        if not self.okx:
            self.okx = OKXClient(api_key=self.cfg.get('api_key',''), secret_key=self.cfg.get('secret_key',''),
                                 passphrase=self.cfg.get('passphrase',''), testnet=False,
                                 proxy_url=self.cfg.get('proxy_url','').strip() or None)

    def _scan(self, auto=False):
        if self.scanning: return
        k = self.cfg.get('api_key',''); s = self.cfg.get('secret_key','')
        if not k or not s: self._pop("提示", "请先在 API 设置中填写 Key"); return
        self.scanning = True
        self._cancel_flag = False
        self._status("连接 OKX..."); self.pb.value = 0
        self.rbox.clear_widgets()
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
            self.ab.text = "启动定时扫描"; self.ab.background_color = C_WARN
            self._status("定时已停止")
        else:
            try: sec = max(60, int(self.tm.text))
            except ValueError: sec = 600
            self.ab.text = "停止定时扫描"; self.ab.background_color = C_RED
            self._save()
            self._status(f"定时 {sec}秒"); self._scan(auto=True)
            self.auto_timer = Clock.schedule_interval(lambda dt: self._scan(auto=True), sec)

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
            self._status(f"{len(active)} 品种分析中..."); self._prog(10)
            found = 0
            for i, t in enumerate(active):
                if self._cancel_flag:
                    self._status(f"已取消 (分析 {found} 个)"); self._prog(0); return
                iid = t['instId']; pct = 10 + int(85*(i+1)/len(active))
                self._prog(pct); self._status(f"[{i+1}/{len(active)}] {iid}")
                try:
                    kls = {}
                    for bar in ['1D','1H','15m','3m']:
                        if self._cancel_flag: return
                        rr = self.okx.get_kline(iid, bar=bar, limit=200)
                        if isinstance(rr, dict) and rr.get('code')=='0' and rr.get('data'):
                            kls[bar] = rr['data']
                    if not kls.get('1D') or not kls.get('1H'): continue
                    sym = ScannerSymbol(inst_id=iid, last_price=float(t.get('last',0)),
                                        volume_24h=float(t.get('volCcyQuote') or t.get('vol24h') or 0),
                                        extra_data={'klines': kls})
                    res = self.scanner.scan_symbol(sym)
                    if res.get('passed', False) or res.get('score',0) >= 60:
                        found += 1; self._add_res(res); self.pool.append(res)
                except Exception: continue
                time.sleep(0.15)
            self._status(f"完成！{found} 个机会"); self._prog(100)
        except Exception as e: self._err(f"扫描失败: {e}")
        finally: self.scanning = False

    def _add_res(self, r):
        def _f(dt):
            bx = BoxLayout(orientation='vertical', size_hint_y=None, height=dp(52), spacing=dp(2))
            with bx.canvas.before: Color(*C_CRD); bx._b = Rectangle(pos=bx.pos, size=bx.size)
            bx.bind(pos=lambda i,v: setattr(bx._b,'pos',v), size=lambda i,v: setattr(bx._b,'size',v))
            d = r.get('direction','NEUTRAL'); dc = C_ACC if d=='LONG' else (C_RED if d=='SHORT' else C_SUB)
            h = BoxLayout(size_hint_y=None, height=dp(22))
            h.add_widget(L(r.get('symbol','?'), 12, C_TXT, True))
            h.add_widget(L(f"{d} {r.get('score',0):.0f}分", 12, dc))
            bx.add_widget(h)
            bx.add_widget(L(' | '.join(r.get('signals',[])[:3]) or '无信号', 10, C_SUB))
            self.rbox.add_widget(bx)
        Clock.schedule_once(_f)

    def _list_strats(self):
        self._smap = {
            'okx_hour_strategy': 'OKX小时线波段共振策略',
            'xiaoyue_boll': '小月期货多周期布林趋势转折',
            'three_min_pullback': '三分钟多周期回调企稳策略',
            'trend_squeeze': '趋势挤压突破前4_30_v2',
            'AI截面五引擎组合扫描器4_28_v2': 'AI截面五引擎组合扫描器4_28_v2',
        }
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
        strat_dir = os.path.join(self.dir, 'strategies')
        files = sorted(glob.glob(os.path.join(strat_dir, '*.py')))
        display = [(os.path.basename(f), f) for f in files if not os.path.basename(f).startswith('_')]

        if not display:
            self._pop("提示", "无可用策略文件"); return

        c = BoxLayout(orientation='vertical', padding=dp(8), spacing=dp(6))
        c.add_widget(L("选择策略文件", 14, C_TAB, True))
        sv = ScrollView(size_hint_y=1)
        flist = BoxLayout(orientation='vertical', size_hint_y=None, spacing=dp(4))
        flist.bind(minimum_height=flist.setter('height'))

        self._file_popup = Popup(title="加载策略文件", content=c, size_hint=(0.88, 0.7),
                                 background_color=(0.12, 0.12, 0.15, 0.97), separator_color=C_TAB, auto_dismiss=True)

        for fn, fp in display:
            btn = Button(text=fn, font_size=sp(13), color=C_TXT,
                         background_color=C_CRD, background_normal='',
                         size_hint_y=None, height=dp(48))
            _f(btn)
            btn.bind(on_release=lambda x, p=fp, n=fn: self._do_load_file(p, n))
            flist.add_widget(btn)
        sv.add_widget(flist)
        c.add_widget(sv)
        c.add_widget(B("关闭", C_BTN, 13, cb=self._file_popup.dismiss))
        self._file_popup.open()

    def _do_load_file(self, path, filename, popup=None):
        if hasattr(self, '_file_popup') and self._file_popup:
            self._file_popup.dismiss()
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
                self._pop("成功", f"已加载: {filename}")
            else:
                self._pop("提示", f"文件中未找到策略类: {filename}")
        except Exception as e:
            self._pop("错误", f"加载失败: {e}")
        name = self.ssp.text
        if not name: return
        # reverse map display -> filename
        rev = {v:k for k,v in self._smap.items()}
        fname = rev.get(name, name)
        fpath = os.path.join(self.dir, 'strategies', fname+'.py')
        if not os.path.exists(fpath):
            self._pop("错误", f"文件未找到: {fname}.py"); return
        try:
            spec = importlib.util.spec_from_file_location(fname, fpath)
            if not spec: self._pop("错误", "策略加载失败"); return
            mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
            for cls_name in ['OKXHourSwingScanner', 'XiaoYueBollMacdScanner',
                             'TrendSqueezeBreakoutScannerV3', 'AICrossSectionDualFactorComboScanner',
                             'ThreeMinuteMultiTimeframePullbackStrategy']:
                if hasattr(mod, cls_name):
                    self.scanner = getattr(mod, cls_name)(); self._pop("成功", f"已加载: {name}"); return
            self._pop("提示", "策略类未找到")
        except Exception as e: self._pop("错误", f"加载失败: {e}")

    # ═══════════════════ Pool ═══════════════════
    def _pool_page(self):
        p = BoxLayout(orientation='vertical', padding=dp(10), spacing=dp(8))
        p.add_widget(L("交易池 - 扫描结果汇集", 15, C_TAB, True))
        sv = ScrollView(size_hint_y=1)
        self.pc = BoxLayout(orientation='vertical', size_hint_y=None, spacing=dp(3))
        self.pc.bind(minimum_height=self.pc.setter('height'))
        if not self.pool: self.pc.add_widget(L("暂无数据, 扫描后自动汇集", 12, C_SUB))
        sv.add_widget(self.pc)
        p.add_widget(sv)
        bar = BoxLayout(size_hint_y=None, height=dp(48), spacing=dp(8))
        bar.add_widget(B("导出记录", (0.25, 0.40, 0.30, 1), 13, cb=lambda x: self._pop("导出", "开发中")))
        bar.add_widget(B("清空", C_RED, 13, cb=self._clr_pool))
        p.add_widget(bar)
        return p

    def _clr_pool(self, btn):
        self.pool = []; self.pc.clear_widgets(); self.pc.add_widget(L("已清空", 12, C_SUB))

    # ═══════════════════ Monitor ═══════════════════
    def _mon_page(self):
        p = BoxLayout(orientation='vertical', padding=dp(10), spacing=dp(8))
        p.add_widget(L("监控池", 15, C_TAB, True))
        r = BoxLayout(size_hint_y=None, height=dp(48), spacing=dp(8))
        self.mi = TextInput(hint_text="如 BTC-USDT-SWAP", multiline=False,
                            font_size=sp(14), background_color=C_CRD, foreground_color=C_TXT)
        _f(self.mi); r.add_widget(self.mi)
        r.add_widget(B("添加", C_BTN, 13, cb=self._add_mon))
        p.add_widget(r)
        sv = ScrollView(size_hint_y=1)
        self.mc = BoxLayout(orientation='vertical', size_hint_y=None, spacing=dp(3))
        self.mc.bind(minimum_height=self.mc.setter('height'))
        self._refresh_mon_view()
        sv.add_widget(self.mc)
        p.add_widget(sv)
        bar = BoxLayout(size_hint_y=None, height=dp(48), spacing=dp(8))
        bar.add_widget(B("开始监控", C_ACC, 13, cb=lambda x: self._pop("监控", "开发中")))
        bar.add_widget(B("停止", C_RED, 13, cb=lambda x: self._pop("监控", "已停止")))
        p.add_widget(bar)
        return p

    def _add_mon(self, btn):
        s = self.mi.text.strip()
        if s and s not in self.monitors:
            self.monitors.append(s); self.mi.text = ''; self._refresh_mon_view()

    def _refresh_mon_view(self):
        self.mc.clear_widgets()
        if not self.monitors: self.mc.add_widget(L("暂无监控交易对", 12, C_SUB)); return
        for s in self.monitors:
            r = BoxLayout(size_hint_y=None, height=dp(42), spacing=dp(6))
            r.add_widget(L(s, 13, C_TXT))
            r.add_widget(B("删除", C_RED, 11, cb=lambda x, sym=s: self._rm_mon(sym)))
            self.mc.add_widget(r)

    def _rm_mon(self, s):
        if s in self.monitors: self.monitors.remove(s)
        self._refresh_mon_view()

    # ═══════════════════ Data ═══════════════════
    def _data_page(self):
        p = BoxLayout(orientation='vertical', padding=dp(10), spacing=dp(8))
        p.add_widget(L("交易对数据库", 15, C_TAB, True))
        r = BoxLayout(size_hint_y=None, height=dp(48), spacing=dp(8))
        self.di = TextInput(hint_text="如 BTC-USDT-SWAP", multiline=False,
                            font_size=sp(14), background_color=C_CRD, foreground_color=C_TXT)
        _f(self.di); r.add_widget(self.di)
        r.add_widget(B("下载K线", C_BTN, 13, cb=lambda x: self._pop("数据库", "开发中")))
        p.add_widget(r)
        sv = ScrollView(size_hint_y=1)
        self.dc = BoxLayout(orientation='vertical', size_hint_y=None, spacing=dp(3))
        self.dc.bind(minimum_height=self.dc.setter('height'))
        self.dc.add_widget(L("暂无数据", 12, C_SUB))
        sv.add_widget(self.dc)
        p.add_widget(sv)
        bar = BoxLayout(size_hint_y=None, height=dp(48), spacing=dp(8))
        bar.add_widget(B("刷新", (0.20, 0.20, 0.25, 1), 13, cb=lambda x: self._pop("数据库", "已刷新")))
        bar.add_widget(B("清空", C_RED, 13, cb=lambda x: self._pop("数据库", "已清空")))
        p.add_widget(bar)
        return p

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
            self.cfg['interval'] = self.tm.text
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
