#!/usr/bin/env python3
"""
Telegram 机器人启动脚本
"""

import sys
import os

# 添加项目根目录到 Python 路径
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from src.api.okx_client import OKXClient
from src.telegram_bot import CryptoTraderBot


def main():
    """主函数"""
    # 检查依赖
    try:
        import telegram
        import telegram.ext
    except ImportError:
        print("❌ 缺少 python-telegram-bot 库")
        print("请运行：pip3 install python-telegram-bot")
        sys.exit(1)
    
    # 设置代理（中国用户必须）
    os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7897'
    os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7897'
    
    # 从环境变量或配置文件读取 Token
    token = os.environ.get('TELEGRAM_BOT_TOKEN')
    
    # 如果环境变量未设置，尝试从配置文件读取
    if not token:
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'telegram_config.txt')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                token = f.read().strip()
    
    if not token:
        print("❌ 未设置 TELEGRAM_BOT_TOKEN")
        print("\n请选择一种方式配置 Token：")
        print("1. 环境变量：export TELEGRAM_BOT_TOKEN='your_token'")
        print("2. 配置文件：在 telegram_config.txt 中写入 Token")
        sys.exit(1)
    
    # OKX 配置（硬编码，方便使用）
    okx_api_key = os.environ.get('OKX_API_KEY', 'ddafb223-6fe7-4ada-94f6-a31d58b23e1a')
    okx_secret_key = os.environ.get('OKX_SECRET_KEY', 'C05E005B0B94EB17E44739C7302605C9')
    okx_passphrase = os.environ.get('OKX_PASSPHRASE', '!Lqs4381525')
    okx_testnet = os.environ.get('OKX_TESTNET', 'true').lower() == 'true'
    proxy_url = os.environ.get('HTTP_PROXY', 'http://127.0.0.1:7897')
    
    # 初始化 OKX 客户端
    okx_client = OKXClient(
        api_key=okx_api_key,
        secret_key=okx_secret_key,
        passphrase=okx_passphrase,
        testnet=okx_testnet,
        proxy_url=proxy_url
    )
    
    # 创建机器人
    bot = CryptoTraderBot(
        token=token,
        okx_client=okx_client
    )
    
    print("✅ Telegram 机器人已就绪")
    print(f"📱 用户名: @lqsjiamisaomiao_bot")
    print(f"💬 请在 Telegram 中发送 /start 开始使用")
    print("="*50)
    
    # 启动机器人
    bot.run()


if __name__ == "__main__":
    main()
