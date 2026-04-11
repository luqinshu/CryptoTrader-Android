#!/usr/bin/env python3
"""Crypto Trader - macOS 原生版本"""

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from src.strategy.loader import StrategyLoader
from src.backtest.engine import Backtester, BacktestAnalyzer

from AppKit import *
from Foundation import NSObject

class AppDelegate(NSObject):
    def applicationDidFinishLaunching_(self, notification):
        return True
    
    def applicationShouldTerminateAfterLastWindowClosed_(self, sender):
        return True

def main():
    app = NSApplication.sharedApplication()
    app.setActivationPolicy_(NSApplicationActivationPolicyRegular)
    
    delegate = AppDelegate.alloc().init()
    app.setDelegate_(delegate)
    
    # 创建主窗口
    window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
        ((100, 100), (1000, 700)),
        NSTitledWindowMask | NSClosableWindowMask | NSMiniaturizableWindowMask | NSResizableWindowMask,
        NSBackingStoreBuffered,
        False
    )
    window.setTitle_("Crypto Trader - 量化交易系统")
    
    # 创建视图
    view = NSView.alloc().initWithFrame_(((0, 0), (1000, 700)))
    view.setBackgroundColor_(NSColor.colorWithCalibratedRed_green_blue_alpha_(0.12, 0.12, 0.12, 1.0))
    
    # 标题
    title = NSTextField.alloc().initWithFrame_(((50, 620), (900, 50)))
    title.setStringValue_("Crypto Trader - 量化交易系统")
    title.setEditable_(False)
    title.setBezeled_(False)
    title.setDrawsBackground_(False)
    title.setTextColor_(NSColor.colorWithCalibratedRed_green_blue_alpha_(0.0, 0.67, 1.0, 1.0))
    title.setFont_(NSFont.boldSystemFontOfSize_(24))
    view.addSubview_(title)
    
    # 策略列表标题
    list_title = NSTextField.alloc().initWithFrame_(((50, 570), (200, 30)))
    list_title.setStringValue_("策略选择:")
    list_title.setEditable_(False)
    list_title.setBezeled_(False)
    list_title.setDrawsBackground_(False)
    list_title.setTextColor_(NSColor.whiteColor())
    list_title.setFont_(NSFont.boldSystemFontOfSize_(16))
    view.addSubview_(list_title)
    
    # 策略列表
    strategy_list = NSScrollView.alloc().initWithFrame_(((50, 350), (300, 200)))
    strategy_list.setHasVerticalScroller_(True)
    view.addSubview_(strategy_list)
    
    # 日志标题
    log_title = NSTextField.alloc().initWithFrame_(((400, 310), (100, 30)))
    log_title.setStringValue_("回测日志:")
    log_title.setEditable_(False)
    log_title.setBezeled_(False)
    log_title.setDrawsBackground_(False)
    log_title.setTextColor_(NSColor.whiteColor())
    log_title.setFont_(NSFont.boldSystemFontOfSize_(14))
    view.addSubview_(log_title)
    
    # 日志文本
    log_text = NSTextView.alloc().initWithFrame_(((400, 100), (550, 200)))
    log_text.setEditable_(False)
    log_text.setBackgroundColor_(NSColor.colorWithCalibratedRed_green_blue_alpha_(0.05, 0.05, 0.05, 1.0))
    log_text.setTextColor_(NSColor.colorWithCalibratedRed_green_blue_alpha_(0.0, 1.0, 0.0, 1.0))
    log_text.setFont_(NSFont.fontWithName_size_("Menlo", 12))
    view.addSubview_(log_text)
    
    def log(message):
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        text = f"[{timestamp}] {message}\n"
        log_text.setString_(log_text.string() + text)
        log_text.scrollToEndOfDocument_(None)
    
    # 策略加载器
    strategy_loader = StrategyLoader('strategies')
    strategies = strategy_loader.discover_strategies()
    
    log("系统已启动")
    log(f"发现 {len(strategies)} 个策略:")
    for s in strategies:
        log(f"  - {s.name}: {s.description or '暂无描述'}")
    log("就绪")
    
    window.setContentView_(view)
    window.makeKeyAndOrderFront_(None)
    window.orderFrontRegardless()
    
    app.activateIgnoringOtherApps_(True)
    log("窗口已创建")
    
    app.run()

if __name__ == "__main__":
    main()
