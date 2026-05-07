import sys
import os
import time
from datetime import datetime

# 添加项目根目录到 Python 路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from src.api.okx_client import OKXClient
from src.scanner.base_scanner import ScannerSymbol
from strategies.OKX小时线波段共振策略 import OKXHourSwingScanner

def run_verification():
    print("="*100)
    print(f"🕵️ OKX 完美波段策略验证程序 - 启动: {datetime.now().strftime('%H:%M:%S')}")
    print("="*100)

    # 1. 初始化 OKX 客户端 (包含代理)
    client = OKXClient(
        api_key="ddafb223-6fe7-4ada-94f6-a31d58b23e1a",
        secret_key="C05E005B0B94EB17E44739C7302605C9",
        passphrase="!Lqs4381525",
        testnet=True,
        proxy_url="http://127.0.0.1:7897"
    )

    # 2. 初始化完美优化版策略
    scanner = OKXHourSwingScanner(config={'min_score': 70}) # 调低门槛以便观察过程

    try:
        # 获取 OKX 所有 SWAP 交易对
        print("📡 [步骤1/3] 正在从 OKX 获取全市场 SWAP 合约列表...")
        response = client.get_tickers('SWAP')
        tickers = response.get('data', []) if isinstance(response, dict) else response
        
        if not tickers:
            print("❌ 错误: 未能获取到交易对，请检查网络或 API 密钥。")
            return

        print(f"✅ 成功发现 {len(tickers)} 个 SWAP 合约。")

        # 过滤成交量，只扫描有流动性的币种
        active_tickers = [t for t in tickers if float(t.get('volCcy24h', 0)) > 5000000]
        print(f"🎯 [步骤2/3] 过滤后进入扫描池的币种 (24h成交量>5M): {len(active_tickers)} 个")
        print("\n⏳ 开始逐一比对多周期数据 (1D + 1H + 3m)...")
        print("-" * 100)
        print(f"{'交易对':<18} | {'方向':<8} | {'评分':<6} | {'信号与逻辑比对反馈'}")
        print("-" * 100)

        found_count = 0
        for i, ticker in enumerate(active_tickers, 1):
            symbol_id = ticker['instId']
            
            # 显示实时扫描进度
            sys.stdout.write(f"\r🔍 [{i}/{len(active_tickers)}] 正在调取数据并比对: {symbol_id}...")
            sys.stdout.flush()

            try:
                # 验证是否真正拉取了三个周期的 K 线
                klines = {
                    '1D': client.get_kline(symbol_id, '1D', limit=200),  # 需200根计算EMA200
                    '1H': client.get_kline(symbol_id, '1H', limit=100),  # 需100根计算布林带带宽百分位
                    '3m': client.get_kline(symbol_id, '3m', limit=50)    # 需50根计算3m EMA和VWAP
                }
                
                # 构造扫描对象
                symbol_obj = ScannerSymbol(
                    inst_id=symbol_id,
                    last_price=float(ticker['last']),
                    volume_24h=float(ticker['volCcy24h']),
                    price_change_24h=0.0,
                    extra_data={'klines': klines}
                )

                # 执行策略比对逻辑
                res = scanner.scan_symbol(symbol_obj)
                
                # 如果得分较高，或者满足基本通过条件，则详细输出
                if res.get('score', 0) >= 50:
                    found_count += 1
                    print(f"\r" + " " * 80, end="") # 清行
                    direction = res.get('direction', 'NEUTRAL')
                    score = res.get('score', 0)
                    signals = " | ".join(res.get('signals', []))
                    
                    # 格式化输出
                    print(f"\r⭐ {symbol_id:<15} | {direction:<8} | {score:<6.1f} | {signals}")
                
                # 稍微延时，尊重 OKX API 限速 (Public 约 10-20 次/秒)
                time.sleep(0.1)

            except Exception as e:
                # 跳过异常币种
                continue

        print("\n" + "="*100)
        print(f"✅ 验证完成！共比对 {len(active_tickers)} 个活跃合约。")
        print(f"📢 发现符合波段启动特征的交易对: {found_count} 个")
        print("="*100)

    except Exception as e:
        print(f"\n❌ 验证中断: {e}")

if __name__ == "__main__":
    run_verification()
