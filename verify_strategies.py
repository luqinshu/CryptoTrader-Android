import sys
import os
import traceback

# 添加项目根目录到 Python 路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from src.api.okx_client import OKXClient
from strategies.三周期共振趋势扫描 import TripleTimeframeTrendScanner
from strategies.单边大趋势捕捉扫描_早期发现版 import UnilateralTrendScannerEarly

def test_scanner():
    print("🚀 开始策略实时验证...")
    
    # 初始化客户端 (使用测试账号配置)
    okx_client = OKXClient(
        api_key="ddafb223-6fe7-4ada-94f6-a31d58b23e1a",
        secret_key="C05E005B0B94EB17E44739C7302605C9",
        passphrase="!Lqs4381525",
        testnet=True,
        proxy_url="http://127.0.0.1:7897"
    )

    # 1. 测试币种: BTC-USDT-SWAP
    symbol_id = "BTC-USDT-SWAP"
    print(f"\n🔍 正在获取 {symbol_id} 的多周期数据...")
    
    try:
        # 模拟扫描器获取数据 (1D, 1H, 3m)
        klines = {
            '1D': okx_client.get_kline(symbol_id, '1D', limit=50),
            '1H': okx_client.get_kline(symbol_id, '1H', limit=50),
            '5m': okx_client.get_kline(symbol_id, '5m', limit=50),
            '3m': okx_client.get_kline(symbol_id, '3m', limit=50)
        }
        
        from src.scanner.base_scanner import ScannerSymbol
        ticker = okx_client.get_ticker(symbol_id)
        
        symbol_obj = ScannerSymbol(
            inst_id=symbol_id,
            last_price=float(ticker['last']),
            volume_24h=float(ticker['volCcy24h']),
            price_change_24h=0.0, # 简化处理
            extra_data={'klines': klines}
        )

        # 2. 运行 三周期共振策略 V3.5
        print("\n--- [策略1: 三周期共振 V3.5] ---")
        scanner_triple = TripleTimeframeTrendScanner(config={'min_score': 60})
        res_triple = scanner_triple.scan_symbol(symbol_obj)
        print(f"得分: {res_triple.get('score', 0)}")
        print(f"评级: {res_triple.get('rating', 'N/A')}")
        print(f"方向: {res_triple.get('direction', 'N/A')}")
        print(f"信号: {', '.join(res_triple.get('signals', []))}")

        # 3. 运行 早期发现策略 V3.0
        print("\n--- [策略2: 早期发现 V3.0] ---")
        scanner_early = UnilateralTrendScannerEarly(config={'min_score': 50})
        res_early = scanner_early.scan_symbol(symbol_obj)
        print(f"得分: {res_early.get('score', 0)}")
        print(f"评级: {res_early.get('rating', 'N/A')}")
        print(f"信号: {', '.join(res_early.get('signals', []))}")

    except Exception as e:
        print(f"❌ 运行出错: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    test_scanner()
