"""
Telegram 机器人主模块
通过 Telegram 实现量化交易扫描和通知功能
"""

import os
import json
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler, CallbackQueryHandler,
    ContextTypes, filters
)
from telegram.request import HTTPXRequest

from src.api.okx_client import OKXClient
from src.scanner.engine import ScanEngine
from src.strategy.loader import StrategyLoader


class CryptoTraderBot:
    """Telegram 量化交易机器人"""

    def __init__(self, token: str, okx_client: OKXClient, strategies_dir: str = None):
        """
        初始化机器人

        Args:
            token: Telegram Bot Token
            okx_client: OKX 客户端实例
            strategies_dir: 策略目录路径
        """
        self.token = token
        self.okx_client = okx_client
        self.scan_engine = ScanEngine(okx_client)
        
        # 正确构建策略目录路径
        if strategies_dir is None:
            current_file = Path(__file__)
            strategies_dir = str(current_file.parent.parent.parent / 'strategies')
        
        self.strategy_loader = StrategyLoader(strategies_dir=strategies_dir)
        self.strategies = self.strategy_loader.discover_strategies()
        
        # 配置
        self.config = self._load_config()
        self.chat_id = self.config.get('chat_id')
        
        # 运行状态
        self.is_scanning = False
        self.current_strategy = None
        self.scan_results = []
        
        # 代理配置（中国用户需要）
        proxy_url = "http://127.0.0.1:7897"
        
        # 初始化 Telegram 应用（v22 版本，带代理）
        self.application = (
            ApplicationBuilder()
            .token(token)
            .proxy(proxy_url)
            .get_updates_proxy(proxy_url)
            .connection_pool_size(8)
            .get_updates_connection_pool_size(8)
            .build()
        )
        self._setup_handlers()
    
    def _load_config(self) -> Dict:
        """加载配置文件"""
        config_path = Path(__file__).parent.parent / 'telegram_config.json'
        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"加载配置失败：{e}")
        return {}
    
    def _save_config(self):
        """保存配置文件"""
        config_path = Path(__file__).parent.parent / 'telegram_config.json'
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"保存配置失败：{e}")
    
    def _setup_handlers(self):
        """设置命令处理器"""
        # 命令处理器
        self.application.add_handler(CommandHandler("start", self.cmd_start))
        self.application.add_handler(CommandHandler("help", self.cmd_help))
        self.application.add_handler(CommandHandler("status", self.cmd_status))
        self.application.add_handler(CommandHandler("strategies", self.cmd_strategies))
        self.application.add_handler(CommandHandler("scan", self.cmd_scan))
        self.application.add_handler(CommandHandler("auto", self.cmd_auto_scan))
        self.application.add_handler(CommandHandler("stop", self.cmd_stop))
        self.application.add_handler(CommandHandler("pool", self.cmd_pool))
        self.application.add_handler(CommandHandler("config", self.cmd_config))
        
        # 按钮回调
        self.application.add_handler(CallbackQueryHandler(self.button_callback))
        
        # 消息处理器（用于设置 chat_id）
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))
    
    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """启动命令"""
        chat_id = update.effective_chat.id
        self.config['chat_id'] = str(chat_id)
        self._save_config()
        
        message = (
            "🤖 **Crypto Trader 量化交易机器人**\n\n"
            "欢迎使用！我是您的量化交易助手。\n\n"
            "📋 **可用命令：**\n"
            "/start - 启动机器人\n"
            "/help - 查看帮助\n"
            "/status - 查看当前状态\n"
            "/strategies - 查看策略列表\n"
            "/scan - 手动扫描\n"
            "/auto - 自动扫描设置\n"
            "/stop - 停止扫描\n"
            "/pool - 查看交易对池\n"
            "/config - 配置设置\n\n"
            "💡 点击下面的按钮快速操作："
        )
        
        keyboard = [
            [InlineKeyboardButton("🔍 立即扫描", callback_data="scan_now")],
            [InlineKeyboardButton("📊 查看策略", callback_data="show_strategies")],
            [InlineKeyboardButton("⚙️ 自动扫描", callback_data="auto_scan_menu")],
            [InlineKeyboardButton("📦 交易对池", callback_data="show_pool")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(message, parse_mode='Markdown', reply_markup=reply_markup)
    
    async def cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """帮助命令"""
        message = (
            "📖 **使用帮助**\n\n"
            "**1. 手动扫描**\n"
            "发送 `/scan` 或点击「立即扫描」\n"
            "机器人会使用上次选择的策略进行扫描\n\n"
            "**2. 自动扫描**\n"
            "发送 `/auto` 设置自动扫描\n"
            "可选择策略和扫描间隔\n\n"
            "**3. 查看结果**\n"
            "扫描完成后会自动推送结果\n"
            "符合条件的交易对会加入交易对池\n\n"
            "**4. 交易对池**\n"
            "发送 `/pool` 查看累计的交易对\n\n"
            "**5. 配置**\n"
            "发送 `/config` 设置参数\n\n"
            "💬 如需帮助，请联系管理员。"
        )
        await update.message.reply_text(message, parse_mode='Markdown')
    
    async def cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """状态命令"""
        status_text = "📊 **当前状态**\n\n"
        status_text += f"🤖 机器人：{'运行中' if self.chat_id else '未配置'}\n"
        status_text += f"🔍 扫描状态：{'扫描中' if self.is_scanning else '空闲'}\n"
        
        if self.current_strategy:
            status_text += f"📈 当前策略：{self.current_strategy.__class__.__name__}\n"
        else:
            status_text += "📈 当前策略：未选择\n"
        
        status_text += f"📦 交易对池：{len(self.scan_engine.last_results)} 个\n"
        
        # 账户信息
        try:
            balance = self.okx_client.get_account_balance()
            if balance and balance.get('code') == '0':
                details = balance.get('data', [{}])[0]
                total_eq = details.get('totalEq', '0')
                status_text += f"💰 账户余额：{total_eq} USD\n"
        except:
            pass
        
        await update.message.reply_text(status_text, parse_mode='Markdown')
    
    async def cmd_strategies(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """策略列表命令"""
        message = "📈 **可用策略列表**\n\n"
        
        for i, strategy in enumerate(self.strategies, 1):
            message += f"{i}. **{strategy.name}** ({strategy.type.value})\n"
            if strategy.description:
                message += f"   {strategy.description[:50]}\n"
            message += "\n"
        
        message += "💡 使用 `/scan 策略名` 开始扫描"
        
        keyboard = []
        row = []
        for i, strategy in enumerate(self.strategies, 1):
            row.append(InlineKeyboardButton(
                strategy.name[:20],
                callback_data=f"select_strategy_{i-1}"
            ))
            if len(row) == 2:
                keyboard.append(row)
                row = []
        if row:
            keyboard.append(row)
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.message.reply_text(message, parse_mode='Markdown', reply_markup=reply_markup)
    
    async def cmd_scan(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """手动扫描命令"""
        if self.is_scanning:
            await update.message.reply_text("⚠️ 正在扫描中，请稍候...")
            return
        
        # 检查是否有策略选择
        if not self.current_strategy:
            if self.strategies:
                self._set_strategy(0)
                await update.message.reply_text(f"✅ 已选择策略：{self.current_strategy.__class__.__name__}")
            else:
                await update.message.reply_text("❌ 未找到可用策略")
                return
        
        # 发送开始消息
        msg = await update.message.reply_text(f"🔍 开始扫描：{self.current_strategy.__class__.__name__}")
        
        # 启动扫描
        threading.Thread(target=self._run_scan, args=(update.effective_chat.id, msg.message_id), daemon=True).start()
    
    async def cmd_auto_scan(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """自动扫描设置"""
        message = (
            "⚙️ **自动扫描设置**\n\n"
            "请选择扫描间隔："
        )
        
        keyboard = [
            [InlineKeyboardButton("1 分钟", callback_data="auto_60")],
            [InlineKeyboardButton("5 分钟", callback_data="auto_300")],
            [InlineKeyboardButton("15 分钟", callback_data="auto_900")],
            [InlineKeyboardButton("30 分钟", callback_data="auto_1800")],
            [InlineKeyboardButton("1 小时", callback_data="auto_3600")],
            [InlineKeyboardButton("❌ 关闭自动扫描", callback_data="auto_off")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(message, parse_mode='Markdown', reply_markup=reply_markup)
    
    async def cmd_stop(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """停止扫描"""
        if not self.is_scanning:
            await update.message.reply_text("⚠️ 当前未在扫描")
            return
        
        self.scan_engine.is_running = False
        self.is_scanning = False
        await update.message.reply_text("✅ 扫描已停止")
    
    async def cmd_pool(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """查看交易对池"""
        pool = self._load_trading_pool()
        
        if not pool:
            await update.message.reply_text("📦 交易对池为空")
            return
        
        message = f"📦 **交易对池** ({len(pool)} 个)\n\n"
        
        # 按时间倒序显示前20个
        sorted_pool = sorted(pool, key=lambda x: x.get('time', ''), reverse=True)[:20]
        
        for i, p in enumerate(sorted_pool, 1):
            symbol = p['symbol'].replace('-USDT-SWAP', '/USDT')
            message += f"{i}. **{symbol}**\n"
            message += f"   价格：{p.get('price', 0):.2f} | 涨跌：{p.get('change', 0):+.2f}%\n"
            message += f"   得分：{p.get('score', 0):.1f} | 策略：{p.get('strategy', '未知')}\n"
            message += f"   时间：{p.get('time', '')}\n\n"
        
        keyboard = [
            [InlineKeyboardButton("🗑 清空交易对池", callback_data="clear_pool")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(message, parse_mode='Markdown', reply_markup=reply_markup)
    
    async def cmd_config(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """配置设置"""
        message = (
            "⚙️ **配置选项**\n\n"
            "请选择要配置的项目："
        )
        
        keyboard = [
            [InlineKeyboardButton("📧 邮件通知设置", callback_data="config_email")],
            [InlineKeyboardButton("🔔 推送通知设置", callback_data="config_notify")],
            [InlineKeyboardButton("📊 默认策略设置", callback_data="config_default_strategy")],
            [InlineKeyboardButton("💾 保存并返回", callback_data="back_to_main")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(message, parse_mode='Markdown', reply_markup=reply_markup)
    
    async def button_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """按钮回调处理"""
        query = update.callback_query
        await query.answer()
        
        data = query.data
        
        if data == "scan_now":
            # 立即扫描
            if self.is_scanning:
                await query.edit_message_text("⚠️ 正在扫描中，请稍候...")
                return
            
            if not self.current_strategy and self.strategies:
                self._set_strategy(0)
            
            await query.edit_message_text(f"🔍 开始扫描：{self.current_strategy.__class__.__name__}")
            threading.Thread(target=self._run_scan, args=(query.message.chat_id, query.message.message_id), daemon=True).start()
        
        elif data == "show_strategies":
            await self.cmd_strategies(update, context)
        
        elif data == "auto_scan_menu":
            await self.cmd_auto_scan(update, context)
        
        elif data == "show_pool":
            await self.cmd_pool(update, context)
        
        elif data.startswith("select_strategy_"):
            idx = int(data.split('_')[-1])
            self._set_strategy(idx)
            await query.edit_message_text(f"✅ 已选择策略：{self.current_strategy.__class__.__name__}\n\n使用 /scan 开始扫描")
        
        elif data.startswith("auto_"):
            interval = data.split('_')[1]
            if interval == "off":
                if hasattr(self, 'auto_timer'):
                    self.auto_timer.cancel()
                await query.edit_message_text("✅ 自动扫描已关闭")
            else:
                self.auto_scan_interval = int(interval)
                self._start_auto_scan(query.message.chat_id)
                await query.edit_message_text(f"✅ 自动扫描已启动，间隔：{int(interval)//60} 分钟")
        
        elif data == "clear_pool":
            self._clear_trading_pool()
            await query.edit_message_text("✅ 交易对池已清空")
        
        elif data == "back_to_main":
            await self.cmd_start(update, context)
    
    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """处理普通消息"""
        text = update.message.text.strip()
        
        # 如果消息是策略名称，切换到该策略
        for i, strategy in enumerate(self.strategies):
            if text.lower() in strategy.name.lower():
                self._set_strategy(i)
                await update.message.reply_text(f"✅ 已选择策略：{strategy.name}")
                return
        
        await update.message.reply_text("❓ 未知命令，发送 /help 查看帮助")
    
    def _set_strategy(self, index: int):
        """设置当前策略"""
        if 0 <= index < len(self.strategies):
            info = self.strategies[index]
            try:
                module = self.strategy_loader.load_strategy(info.name)
                if module:
                    cls = None
                    for name, obj in module.__dict__.items():
                        if isinstance(obj, type) and name != 'BaseScannerStrategy':
                            try:
                                if issubclass(obj, BaseScannerStrategy):
                                    cls = obj
                                    break
                            except:
                                pass
                    
                    if cls:
                        config = {}
                        if info.config_schema:
                            for k, v in info.config_schema.items():
                                config[k] = v.get('default', 0)
                        
                        self.current_strategy = cls(config)
                        self.scan_engine.set_strategy(self.current_strategy)
            except Exception as e:
                print(f"设置策略失败：{e}")
    
    def _run_scan(self, chat_id: int, message_id: int = None):
        """执行扫描（在线程中运行）"""
        self.is_scanning = True
        self.scan_results = []
        
        try:
            strategy_name = self.current_strategy.__class__.__name__
            print(f"\n{'='*50}")
            print(f"=== Telegram 扫描开始 ===")
            print(f"策略：{strategy_name}")
            print(f"{'='*50}\n")
            
            # 获取交易对
            symbols = self.scan_engine.get_contract_symbols()
            if not symbols:
                self._send_message(chat_id, "❌ 获取交易对失败")
                return
            
            # 执行扫描
            all_tickers = self.scan_engine.get_all_tickers()
            if not all_tickers:
                self._send_message(chat_id, "❌ 获取行情数据失败")
                return
            
            # 扫描所有交易对
            results = []
            for ticker in all_tickers:
                if not self.scan_engine.is_running:
                    break
                
                symbol = self.scan_engine.parse_ticker(ticker)
                open_24h = float(ticker.get('open24h', 0))
                if open_24h > 0:
                    symbol.price_change_24h = ((symbol.last_price - open_24h) / open_24h) * 100
                
                # 执行策略扫描
                result = self.scan_engine.strategy.scan_symbol(symbol)
                result.update({
                    'last_price': symbol.last_price,
                    'volume_24h': symbol.volume_24h,
                    'price_change_24h': symbol.price_change_24h,
                    'high_24h': symbol.high_24h,
                    'low_24h': symbol.low_24h
                })
                
                if result.get('passed'):
                    results.append(result)
                    # 添加到交易对池
                    self._add_to_pool(result)
            
            self.scan_results = results
            self.is_scanning = False
            
            # 发送结果
            if results:
                self._send_scan_results(chat_id, results)
            else:
                self._send_message(chat_id, f"✅ 扫描完成，未找到符合条件的交易对")
        
        except Exception as e:
            print(f"扫描错误：{e}")
            import traceback
            traceback.print_exc()
            self._send_message(chat_id, f"❌ 扫描错误：{str(e)}")
            self.is_scanning = False
    
    def _send_scan_results(self, chat_id: int, results: List[Dict]):
        """发送扫描结果"""
        # 按得分排序
        results.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        message = f"🎉 **扫描完成！**\n\n"
        message += f"📊 通过：**{len(results)}** 个交易对\n"
        message += f"📈 策略：{self.current_strategy.__class__.__name__}\n\n"
        message += "**详细结果：**\n\n"
        
        # 显示前10个结果
        for i, r in enumerate(results[:10], 1):
            symbol = r.get('symbol', '').replace('-USDT-SWAP', '/USDT')
            message += f"{i}. **{symbol}**\n"
            message += f"   价格：{r.get('last_price', 0):.2f}\n"
            message += f"   24h涨跌：{r.get('price_change_24h', 0):+.2f}%\n"
            message += f"   得分：{r.get('score', 0):.1f}\n"
            message += f"   详情：{r.get('details', {})}\n\n"
        
        if len(results) > 10:
            message += f"... 共 {len(results)} 个结果\n"
        
        # 由于消息长度限制，分段发送
        self._send_long_message(chat_id, message)
    
    def _send_long_message(self, chat_id: int, message: str):
        """发送长消息（分段）"""
        max_length = 4000
        for i in range(0, len(message), max_length):
            chunk = message[i:i+max_length]
            self._send_message(chat_id, chunk)
    
    def _send_message(self, chat_id: int, text: str):
        """发送消息到 Telegram"""
        try:
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(
                self.application.bot.send_message(chat_id=chat_id, text=text, parse_mode='Markdown')
            )
        except Exception as e:
            print(f"发送消息失败：{e}")
    
    def _load_trading_pool(self) -> List[Dict]:
        """加载交易对池"""
        pool_path = Path(__file__).parent.parent / 'trading_pool.json'
        if pool_path.exists():
            try:
                with open(pool_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        return []
    
    def _save_trading_pool(self, pool: List[Dict]):
        """保存交易对池"""
        pool_path = Path(__file__).parent.parent / 'trading_pool.json'
        try:
            with open(pool_path, 'w', encoding='utf-8') as f:
                json.dump(pool, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"保存交易对池失败：{e}")
    
    def _add_to_pool(self, result: Dict):
        """添加交易对到池中"""
        pool = self._load_trading_pool()
        
        symbol = result.get('symbol', '')
        # 检查是否已存在
        if not any(p['symbol'] == symbol for p in pool):
            pool.append({
                'time': datetime.now().strftime('%H:%M:%S'),
                'symbol': symbol,
                'price': result.get('last_price', 0),
                'change': result.get('price_change_24h', 0),
                'score': result.get('score', 0),
                'strategy': self.current_strategy.__class__.__name__ if self.current_strategy else '未知'
            })
            self._save_trading_pool(pool)
    
    def _clear_trading_pool(self):
        """清空交易对池"""
        self._save_trading_pool([])
    
    def _start_auto_scan(self, chat_id: int):
        """启动自动扫描"""
        def auto_scan_loop():
            while hasattr(self, 'auto_scan_interval'):
                if not self.is_scanning:
                    self._run_scan(chat_id)
                time.sleep(self.auto_scan_interval)
        
        self.auto_timer = threading.Thread(target=auto_scan_loop, daemon=True)
        self.auto_timer.start()
    
    def run(self):
        """启动机器人"""
        print("🤖 Telegram 机器人启动中...")
        print(f"📱 用户名: @lqsjiamisaomiao_bot")
        print("💬 请在 Telegram 中发送 /start 开始使用")
        print("="*50)
        
        # 使用 run_polling 启动（v22 推荐方式）
        self.application.run_polling(
            poll_interval=1,
            timeout=30,
            drop_pending_updates=True
        )
