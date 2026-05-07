#!/usr/bin/env python3
"""Crypto Trader - Tkinter 版本（macOS 兼容）"""

import sys
import os
import threading
import time
from datetime import datetime

# 添加项目根目录到 Python 路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog

from src.strategy.loader import StrategyLoader
from src.backtest.engine import Backtester, BacktestAnalyzer

class CryptoTraderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Crypto Trader - 量化交易系统")
        self.root.geometry("1200x800")
        
        # 深色主题
        self.colors = {
            'bg': '#1e1e1e',
            'fg': '#ffffff',
            'accent': '#0066cc',
            'success': '#00aa00',
            'error': '#cc0000',
            'card': '#2a2a2a'
        }
        
        self.root.configure(bg=self.colors['bg'])
        
        # 策略加载器
        self.strategy_loader = StrategyLoader('strategies')
        self.strategies = self.strategy_loader.discover_strategies()
        self.selected_strategy = None
        
        # 创建 UI
        self.create_ui()
        
        # 加载策略列表
        self.refresh_strategy_list()
        
    def create_ui(self):
        """创建 UI"""
        # 标题
        title_frame = tk.Frame(self.root, bg=self.colors['accent'], height=60)
        title_frame.pack(fill='x', padx=0, pady=0)
        title_frame.pack_propagate(False)
        
        title_label = tk.Label(
            title_frame, 
            text="🚀 Crypto Trader - 量化交易系统",
            font=("Arial", 20, "bold"),
            bg=self.colors['accent'],
            fg='white'
        )
        title_label.pack(pady=15)
        
        # 标签页
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=20, pady=20)
        
        # 回测页面
        self.backtest_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.backtest_frame, text="  策略回测  ")
        self.create_backtest_page()
        
        # 策略管理页面
        self.strategy_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.strategy_frame, text="  策略管理  ")
        self.create_strategy_page()
        
        # 日志区域
        log_frame = tk.LabelFrame(self.root, text="日志", bg=self.colors['card'], fg=self.colors['fg'])
        log_frame.pack(fill='x', padx=20, pady=(0, 20))
        
        self.log_text = scrolledtext.ScrolledText(
            log_frame, 
            height=6,
            bg='#0d0d0d',
            fg='#00ff00',
            font=('Courier', 10),
            insertbackground='white'
        )
        self.log_text.pack(fill='x', padx=10, pady=10)
        
        self.add_log("系统已就绪")
        
    def create_backtest_page(self):
        """创建回测页面"""
        # 左侧面板
        left_frame = tk.Frame(self.backtest_frame, bg=self.colors['card'], width=400)
        left_frame.pack(side='left', fill='y', padx=(0, 10), pady=10)
        left_frame.pack_propagate(False)
        
        # 策略选择
        strategy_label = tk.Label(
            left_frame, 
            text="选择策略",
            font=("Arial", 12, "bold"),
            bg=self.colors['card'],
            fg=self.colors['fg']
        )
        strategy_label.pack(padx=15, pady=(15, 10), anchor='w')
        
        self.strategy_listbox = tk.Listbox(
            left_frame,
            bg=self.colors['card'],
            fg=self.colors['fg'],
            selectbackground=self.colors['accent'],
            font=("Arial", 11),
            height=8,
            borderwidth=1,
            relief='solid'
        )
        self.strategy_listbox.pack(fill='x', padx=15, pady=10)
        self.strategy_listbox.bind('<<ListboxSelect>>', self.on_strategy_select)
        
        # 刷新按钮
        refresh_btn = tk.Button(
            left_frame,
            text="🔄 刷新策略",
            command=self.refresh_strategy_list,
            bg=self.colors['card'],
            fg=self.colors['fg'],
            relief='flat',
            cursor='hand2'
        )
        refresh_btn.pack(padx=15, pady=5, fill='x')
        
        # 策略说明
        self.strategy_desc = tk.Text(
            left_frame,
            height=4,
            bg='#1a1a1a',
            fg='#888888',
            font=("Arial", 10),
            borderwidth=0,
            state='disabled'
        )
        self.strategy_desc.pack(padx=15, pady=10, fill='x')
        
        # 配置表单
        form_frame = tk.LabelFrame(
            left_frame, 
            text="回测配置",
            bg=self.colors['card'],
            fg=self.colors['fg'],
            font=("Arial", 10, "bold")
        )
        form_frame.pack(fill='both', expand=True, padx=15, pady=10)
        
        # 交易对
        tk.Label(form_frame, text="交易对:", bg=self.colors['card'], fg=self.colors['fg']).pack(anchor='w', padx=10, pady=(10, 5))
        self.pair_var = tk.StringVar(value="BTC-USDT")
        pair_combo = ttk.Combobox(form_frame, textvariable=self.pair_var, values=[
            "BTC-USDT", "ETH-USDT", "SOL-USDT", "XRP-USDT", "ADA-USDT"
        ], state='readonly')
        pair_combo.pack(fill='x', padx=10, pady=5)
        
        # K 线周期
        tk.Label(form_frame, text="K 线周期:", bg=self.colors['card'], fg=self.colors['fg']).pack(anchor='w', padx=10, pady=5)
        self.bar_var = tk.StringVar(value="1H")
        bar_combo = ttk.Combobox(form_frame, textvariable=self.bar_var, values=[
            "1m", "5m", "15m", "30m", "1H", "2H", "4H", "1D"
        ], state='readonly')
        bar_combo.pack(fill='x', padx=10, pady=5)
        
        # 开始日期
        tk.Label(form_frame, text="开始日期:", bg=self.colors['card'], fg=self.colors['fg']).pack(anchor='w', padx=10, pady=5)
        self.start_date_var = tk.StringVar(value="2025-10-07")
        start_date_entry = tk.Entry(form_frame, textvariable=self.start_date_var, bg='#1a1a1a', fg='white')
        start_date_entry.pack(fill='x', padx=10, pady=5)
        
        # 结束日期
        tk.Label(form_frame, text="结束日期:", bg=self.colors['card'], fg=self.colors['fg']).pack(anchor='w', padx=10, pady=5)
        self.end_date_var = tk.StringVar(value="2026-04-07")
        end_date_entry = tk.Entry(form_frame, textvariable=self.end_date_var, bg='#1a1a1a', fg='white')
        end_date_entry.pack(fill='x', padx=10, pady=5)
        
        # 初始资金
        tk.Label(form_frame, text="初始资金 (USDT):", bg=self.colors['card'], fg=self.colors['fg']).pack(anchor='w', padx=10, pady=5)
        self.capital_var = tk.StringVar(value="10000")
        capital_entry = tk.Entry(form_frame, textvariable=self.capital_var, bg='#1a1a1a', fg='white')
        capital_entry.pack(fill='x', padx=10, pady=5)
        
        # 开始回测按钮
        start_btn = tk.Button(
            left_frame,
            text="🚀 开始回测",
            command=self.start_backtest,
            bg=self.colors['success'],
            fg='white',
            font=("Arial", 12, "bold"),
            relief='flat',
            cursor='hand2',
            pady=10
        )
        start_btn.pack(fill='x', padx=15, pady=15)
        
        # 右侧结果面板
        right_frame = tk.Frame(self.backtest_frame, bg=self.colors['card'])
        right_frame.pack(side='right', fill='both', expand=True, padx=(0, 10), pady=10)
        
        # 结果标签
        result_label = tk.Label(
            right_frame,
            text="📈 回测结果",
            font=("Arial", 14, "bold"),
            bg=self.colors['card'],
            fg=self.colors['fg']
        )
        result_label.pack(padx=15, pady=15, anchor='w')
        
        # 结果表格
        self.result_tree = ttk.Treeview(right_frame, columns=('指标', '数值'), show='headings', height=15)
        self.result_tree.heading('指标', text='指标')
        self.result_tree.heading('数值', text='数值')
        self.result_tree.column('指标', width=150)
        self.result_tree.column('数值', width=150)
        self.result_tree.pack(fill='both', expand=True, padx=15, pady=10)
        
        # 导出按钮
        export_btn = tk.Button(
            right_frame,
            text="📥 导出报告",
            command=self.export_report,
            bg=self.colors['accent'],
            fg='white',
            relief='flat',
            cursor='hand2'
        )
        export_btn.pack(pady=10)
        
    def create_strategy_page(self):
        """创建策略管理页面"""
        # 策略列表
        list_frame = tk.Frame(self.strategy_frame, bg=self.colors['card'])
        list_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        title_label = tk.Label(
            list_frame,
            text="📁 可用策略列表",
            font=("Arial", 16, "bold"),
            bg=self.colors['card'],
            fg=self.colors['fg']
        )
        title_label.pack(pady=(10, 20))
        
        # 策略列表框
        self.strategy_manage_listbox = tk.Listbox(
            list_frame,
            bg=self.colors['card'],
            fg=self.colors['fg'],
            selectbackground=self.colors['accent'],
            font=("Arial", 12),
            borderwidth=1,
            relief='solid'
        )
        self.strategy_manage_listbox.pack(fill='both', expand=True, padx=20, pady=10)
        self.refresh_strategy_list_manage()
        
    def refresh_strategy_list(self):
        """刷新策略列表"""
        self.strategy_listbox.delete(0, 'end')
        self.strategies = self.strategy_loader.discover_strategies()
        
        for s in self.strategies:
            self.strategy_listbox.insert('end', f"{s.name} ({s.type.value})")
            
        self.add_log(f"已加载 {len(self.strategies)} 个策略")
        
    def refresh_strategy_list_manage(self):
        """刷新策略管理列表"""
        self.strategy_manage_listbox.delete(0, 'end')
        self.strategies = self.strategy_loader.discover_strategies()
        
        for s in self.strategies:
            desc = s.description if s.description else "暂无描述"
            self.strategy_manage_listbox.insert('end', f"{s.name}\n  {desc}")
            
    def on_strategy_select(self, event):
        """策略选择事件"""
        selection = self.strategy_listbox.curselection()
        if selection:
            index = selection[0]
            self.selected_strategy = self.strategies[index]
            
            # 显示策略说明
            self.strategy_desc.config(state='normal')
            self.strategy_desc.delete('1.0', 'end')
            desc_text = f"策略：{self.selected_strategy.name}\n"
            if self.selected_strategy.description:
                desc_text += f"说明：{self.selected_strategy.description}\n"
            if self.selected_strategy.author:
                desc_text += f"作者：{self.selected_strategy.author}"
            self.strategy_desc.insert('1.0', desc_text)
            self.strategy_desc.config(state='disabled')
            
            self.add_log(f"选择策略：{self.selected_strategy.name}")
            
    def start_backtest(self):
        """开始回测"""
        if not self.selected_strategy:
            messagebox.showwarning("警告", "请先选择策略")
            return
            
        self.add_log(f"开始回测：{self.selected_strategy.name} @ {self.pair_var.get()}")
        
        # 禁用按钮
        # 在后台线程中运行回测
        def run_backtest():
            try:
                # 加载策略
                module = self.strategy_loader.load_strategy(self.selected_strategy.name)
                if not module:
                    self.add_log("策略加载失败", "error")
                    return
                    
                strategy_class = self.strategy_loader.get_strategy_class(self.selected_strategy.name)
                if not strategy_class:
                    self.add_log("策略类未找到", "error")
                    return
                    
                strategy = strategy_class({})
                
                # 运行回测
                backtester = Backtester(initial_capital=float(self.capital_var.get()))
                result = backtester.run_backtest(
                    strategy=strategy,
                    inst_id=self.pair_var.get(),
                    start_date=self.start_date_var.get(),
                    end_date=self.end_date_var.get(),
                    bar=self.bar_var.get()
                )
                
                # 分析结果
                result = BacktestAnalyzer.analyze(result)
                
                # 更新 UI
                self.root.after(0, lambda: self.display_result(result))
                self.root.after(0, lambda: self.add_log("回测完成!", "success"))
                
            except Exception as e:
                self.root.after(0, lambda: self.add_log(f"回测失败：{str(e)}", "error"))
                
        thread = threading.Thread(target=run_backtest)
        thread.start()
        
    def display_result(self, result):
        """显示回测结果"""
        # 清空表格
        for item in self.result_tree.get_children():
            self.result_tree.delete(item)
            
        # 添加数据
        metrics = [
            ("策略名称", result.strategy_name),
            ("交易对", result.inst_id),
            ("初始资金", f"{result.initial_capital:,.2f} USDT"),
            ("最终资金", f"{result.final_capital:,.2f} USDT"),
            ("总收益率", f"{result.total_return:.2f}%"),
            ("年化收益率", f"{result.annual_return:.2f}%"),
            ("最大回撤", f"{result.max_drawdown:.2f}%"),
            ("夏普比率", f"{result.sharpe_ratio:.2f}"),
            ("胜率", f"{result.win_rate:.2f}%"),
            ("盈亏比", f"{result.profit_factor:.2f}"),
            ("总交易次数", str(result.total_trades)),
            ("盈利交易", str(result.winning_trades)),
            ("亏损交易", str(result.losing_trades)),
        ]
        
        for label, value in metrics:
            self.result_tree.insert('', 'end', values=(label, value))
            
    def export_report(self):
        """导出报告"""
        messagebox.showinfo("提示", "导出功能开发中...")
        
    def add_log(self, message, level="info"):
        """添加日志"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        color = {
            "info": "#00aaff",
            "success": "#00ff88",
            "error": "#ff4444"
        }.get(level, "#ffffff")
        
        self.log_text.insert('end', f"[{timestamp}] {message}\n")
        self.log_text.see('end')

def main():
    root = tk.Tk()
    
    # 设置窗口样式
    root.configure(bg='#1e1e1e')
    
    # 创建应用
    app = CryptoTraderApp(root)
    
    # 居中显示
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f'+{x}+{y}')
    
    root.mainloop()

if __name__ == "__main__":
    main()
