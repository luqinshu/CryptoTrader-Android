#!/usr/bin/env python3
"""Crypto Trader 主入口文件 - 修复版"""

import sys
import os

# 添加项目根目录到 Python 路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# 设置环境变量以修复 macOS 显示问题
os.environ['QT_MAC_WANT_LAYERED_RENDERING'] = '1'
os.environ['QT_MAC_DISABLE_FOREGROUND_APPLICATION_TRANSFORM'] = '1'

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QCoreApplication
from PyQt5.QtGui import QGuiApplication

# 在创建 QApplication 之前设置属性
QCoreApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
QCoreApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

from src.ui.quant_trade_window import QuantTradeWindow
from src.api.okx_client import OKXClient

def main():
    """主函数"""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    # 设置应用信息
    QCoreApplication.setApplicationName("Crypto Trader")
    QCoreApplication.setApplicationVersion("1.0.0")
    QCoreApplication.setOrganizationName("Crypto Trader")

    # 初始化 OKX 客户端
    okx_client = OKXClient(
        api_key="ddafb223-6fe7-4ada-94f6-a31d58b23e1a",
        secret_key="C05E005B0B94EB17E44739C7302605C9",
        passphrase="!Lqs4381525",
        testnet=True,
        proxy_url="http://127.0.0.1:7897"
    )

    # 创建主窗口
    window = QuantTradeWindow(okx_client)
    window.show()
    window.raise_()
    window.activateWindow()
    
    # 确保窗口在前台
    import time
    time.sleep(0.1)
    window.activateWindow()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
