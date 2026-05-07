"""
Telegram 机器人 - 简单轮询版本（不依赖 python-telegram-bot）
"""

import os
import sys
import json
import time
import threading
import requests
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path

# 代理配置
PROXY = {
    'http': 'http://127.0.0.1:7897',
    'https': 'http://127.0.0.1:7897'
}

class SimpleTelegramBot:
    """简单的 Telegram 机器人（使用 requests 库）"""
    
    def __init__(self, token: str):
        self.token = token
        self.base_url = f"https://api.telegram.org/bot{token}"
        self.last_update_id = 0
        self.chat_id = None
        self.is_running = False
        
        # 加载配置
        self._load_config()
    
    def _load_config(self):
        """加载配置"""
        config_path = Path(__file__).parent.parent / 'telegram_bot_config.json'
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    self.last_update_id = config.get('last_update_id', 0)
                    self.chat_id = config.get('chat_id')
            except:
                pass
    
    def _save_config(self):
        """保存配置"""
        config_path = Path(__file__).parent.parent / 'telegram_bot_config.json'
        try:
            with open(config_path, 'w') as f:
                json.dump({
                    'last_update_id': self.last_update_id,
                    'chat_id': self.chat_id
                }, f, indent=2)
        except Exception as e:
            print(f"保存配置失败：{e}")
    
    def _request(self, method: str, params: Dict = None, data: Dict = None) -> Dict:
        """发送请求到 Telegram API"""
        url = f"{self.base_url}/{method}"
        try:
            if params:
                response = requests.get(url, params=params, proxies=PROXY, timeout=30)
            elif data:
                response = requests.post(url, json=data, proxies=PROXY, timeout=30)
            else:
                response = requests.get(url, proxies=PROXY, timeout=30)
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"请求失败：{response.status_code} - {response.text}")
                return {}
        except requests.exceptions.RequestException as e:
            print(f"请求异常：{e}")
            return {}
        except Exception as e:
            print(f"未知异常：{e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def get_me(self) -> Dict:
        """获取机器人信息"""
        return self._request("getMe")
    
    def send_message(self, chat_id: int, text: str):
        """发送消息（纯文本）"""
        data = {
            'chat_id': chat_id,
            'text': text
        }
        return self._request("sendMessage", data=data)
    
    def send_message_with_keyboard(self, chat_id: int, text: str, keyboard: List[List[str]]):
        """发送带按钮的消息"""
        # 构建 InlineKeyboardMarkup
        inline_keyboard = []
        for row in keyboard:
            inline_row = []
            for btn_text, callback_data in row:
                inline_row.append({
                    'text': btn_text,
                    'callback_data': callback_data
                })
            inline_keyboard.append(inline_row)
        
        data = {
            'chat_id': chat_id,
            'text': text,
            'parse_mode': 'Markdown',
            'reply_markup': json.dumps({'inline_keyboard': inline_keyboard})
        }
        return self._request("sendMessage", data=data)
    
    def answer_callback(self, callback_query_id: str, text: str = ""):
        """回答回调"""
        data = {
            'callback_query_id': callback_query_id,
            'text': text
        }
        return self._request("answerCallbackQuery", data=data)
    
    def get_updates(self, offset: int = None, timeout: int = 5) -> List[Dict]:
        """获取更新"""
        params = {
            'timeout': timeout,
            'allowed_updates': ['message', 'callback_query']
        }
        if offset:
            params['offset'] = offset
        
        result = self._request("getUpdates", params=params)
        return result.get('result', [])
    
    def start_polling(self):
        """开始轮询"""
        print("🤖 机器人启动轮询...")
        print(f"📱 最后更新 ID: {self.last_update_id}")
        
        self.is_running = True
        offset = self.last_update_id + 1 if self.last_update_id > 0 else None
        
        while self.is_running:
            try:
                updates = self.get_updates(offset=offset, timeout=5)
                
                for update in updates:
                    self._handle_update(update)
                    self.last_update_id = update['update_id']
                    self._save_config()
                    offset = self.last_update_id + 1
                
            except KeyboardInterrupt:
                print("\n⏹️ 收到停止信号")
                self.is_running = False
                break
            except Exception as e:
                print(f"轮询错误：{e}")
                time.sleep(3)
        
        print("🛑 机器人已停止")
    
    def _handle_update(self, update: Dict):
        """处理更新"""
        # 处理消息
        if 'message' in update:
            message = update['message']
            chat_id = message['chat']['id']
            user = message['from']
            text = message.get('text', '')
            
            self.chat_id = chat_id
            self._save_config()
            
            print(f"📨 收到消息 from {user.get('first_name', 'Unknown')}: {text}")
            
            # 处理命令
            if text.startswith('/'):
                self._handle_command(chat_id, text, user)
            else:
                self._handle_text(chat_id, text, user)
        
        # 处理按钮点击
        elif 'callback_query' in update:
            query = update['callback_query']
            chat_id = query['message']['chat']['id']
            user = query['from']
            data = query.get('data', '')
            
            print(f"🔘 按钮点击 from {user.get('first_name', 'Unknown')}: {data}")
            
            self.answer_callback(query['id'])
            self._handle_callback(chat_id, data, user)
    
    def _handle_command(self, chat_id: int, command: str, user: Dict):
        """处理命令"""
        if command == '/start':
            self._cmd_start(chat_id, user)
        elif command == '/help':
            self._cmd_help(chat_id)
        elif command == '/scan':
            self._cmd_scan(chat_id)
        elif command == '/status':
            self._cmd_status(chat_id)
        else:
            self.send_message(chat_id, f"❓ 未知命令：`{command}`\n\n发送 /help 查看帮助")
    
    def _handle_text(self, chat_id: int, text: str, user: Dict):
        """处理普通文本消息"""
        self.send_message(chat_id, f"💬 收到您的消息：\n\n`{text}`\n\n发送 /help 查看可用命令")
    
    def _handle_callback(self, chat_id: int, data: str, user: Dict):
        """处理按钮回调"""
        if data == 'test_button':
            self.send_message(chat_id, "✅ 按钮测试成功！")
        else:
            self.send_message(chat_id, f"🔘 您点击了：`{data}`")
    
    def _cmd_start(self, chat_id: int, user: Dict):
        """处理 /start 命令"""
        first_name = user.get('first_name', '用户')
        message = (
            f"🎉 欢迎，{first_name}！\n\n"
            f"我是 **加密扫描机器人**，您的量化交易助手。\n\n"
            f"📋 **可用命令：**\n"
            f"/start - 启动机器人\n"
            f"/help - 查看帮助\n"
            f"/scan - 手动扫描\n"
            f"/status - 查看状态\n\n"
            f"💡 点击下方按钮测试："
        )
        
        keyboard = [
            [("🔍 立即扫描", "scan_now")],
            [("📊 查看状态", "check_status")],
            [("💬 测试按钮", "test_button")]
        ]
        
        self.send_message_with_keyboard(chat_id, message, keyboard)
    
    def _cmd_help(self, chat_id: int):
        """处理 /help 命令"""
        message = (
            "📖 **使用帮助**\n\n"
            "**命令列表：**\n"
            "/start - 启动机器人\n"
            "/help - 查看帮助\n"
            "/scan - 手动扫描市场\n"
            "/status - 查看运行状态\n\n"
            "**功能说明：**\n"
            "• 自动扫描符合策略的交易对\n"
            "• 符合条件的交易对自动加入交易对池\n"
            "• 支持多策略轮流扫描\n\n"
            "💬 有任何问题请联系管理员。"
        )
        self.send_message(chat_id, message)
    
    def _cmd_scan(self, chat_id: int):
        """处理 /scan 命令"""
        self.send_message(chat_id, "🔍 正在扫描市场...\n\n⏳ 这可能需要几分钟，请稍候...")
        
        # 这里可以调用实际的扫描逻辑
        # 暂时发送测试消息
        time.sleep(2)
        self.send_message(chat_id, "✅ 扫描完成！\n\n📊 本次扫描未发现符合条件的交易对。\n\n💡 提示：您可以调整策略参数以获得更多结果。")
    
    def _cmd_status(self, chat_id: int):
        """处理 /status 命令"""
        message = (
            "📊 **运行状态**\n\n"
            f"🤖 机器人：✅ 运行中\n"
            f"👤 您的 ID：`{chat_id}`\n"
            f"🕐 当前时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            f"💡 发送 /scan 开始扫描"
        )
        self.send_message(chat_id, message)


def main():
    """主函数"""
    # 读取 Token
    current_dir = os.path.dirname(os.path.abspath(__file__))
    token_path = os.path.join(current_dir, 'telegram_config.txt')
    if not os.path.exists(token_path):
        print("❌ 未找到 telegram_config.txt")
        print(f"   当前目录：{current_dir}")
        sys.exit(1)
    
    with open(token_path, 'r') as f:
        token = f.read().strip()
    
    if not token:
        print("❌ Token 为空")
        sys.exit(1)
    
    print("="*50)
    print("🤖 加密扫描 Telegram 机器人")
    print("="*50)
    
    # 创建机器人
    bot = SimpleTelegramBot(token)
    
    # 测试连接
    print("正在测试连接...")
    me = bot.get_me()
    if me.get('ok'):
        result = me['result']
        print(f"✅ 连接成功！")
        print(f"   用户名：@{result['username']}")
        print(f"   名称：{result['first_name']}")
        print(f"   ID：{result['id']}")
    else:
        print(f"❌ 连接失败：{me}")
        sys.exit(1)
    
    print("\n" + "="*50)
    print(f"📱 在 Telegram 中搜索: @{me['result']['username']}")
    print(f"💬 发送 /start 开始使用")
    print("="*50)
    print("\n⏹️ 按 Ctrl+C 停止机器人\n")
    
    # 开始轮询
    bot.start_polling()


if __name__ == "__main__":
    main()
