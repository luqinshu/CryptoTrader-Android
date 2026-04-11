"""
单边大趋势扫描策略 - 实战使用示例

展示如何在实际交易中使用这个策略
"""

from strategies.单边大趋势捕捉扫描 import UnilateralTrendScanner
from strategies.单边大趋势捕捉扫描_高级版 import UnilateralTrendScannerAdvanced
from src.scanner.base_scanner import ScannerSymbol
from typing import List, Dict


class UnilateralTrendTrader:
    """
    单边大趋势交易者

    结合扫描策略和自动交易
    """

    def __init__(self, okx_client, strategy_type='advanced'):
        """
        初始化

        Args:
            okx_client: OKX 客户端
            strategy_type: 策略类型 'basic' 或 'advanced'
        """
        self.okx_client = okx_client
        self.strategy_type = strategy_type

        # 初始化扫描器
        if strategy_type == 'advanced':
            self.scanner = UnilateralTrendScannerAdvanced(config={
                'min_volume_24h': 1000000,
                'min_score': 75,
                'top_n': 10,
            })
        else:
            self.scanner = UnilateralTrendScanner(config={
                'min_volume_24h': 500000,
                'min_score': 70,
                'top_n': 15,
            })

        # 交易配置
        self.trade_config = {
            'max_positions': 3,  # 最大同时持仓数
            'position_size': 0.1,  # 单笔仓位比例(10%)
            'stop_loss_pct': 0.05,  # 止损5%
            'take_profit_pct': 0.15,  # 止盈15%
            'trailing_stop_pct': 0.08,  # 移动止损8%
        }

        # 持仓记录
        self.positions = {}

    def scan_and_trade(self, symbols_data: List[ScannerSymbol]) -> Dict:
        """
        扫描并执行交易

        Args:
            symbols_data: 交易对数据列表

        Returns:
            扫描和交易结果
        """
        # 执行扫描
        scan_result = self.scanner.scan_all_symbols(symbols_data)

        # 获取机会列表
        opportunities = scan_result.get('long_opportunities', [])

        # 过滤掉已持仓的
        new_opportunities = [
            opp for opp in opportunities
            if opp['symbol'] not in self.positions
        ]

        # 执行交易
        trades = []
        for opp in new_opportunities[:self.trade_config['max_positions'] - len(self.positions)]:
            if opp['score'] >= 75:  # 只交易高分机会
                trade = self._execute_trade(opp)
                if trade:
                    trades.append(trade)

        return {
            'scan_result': scan_result,
            'trades': trades,
            'total_positions': len(self.positions),
        }

    def _execute_trade(self, opportunity: Dict) -> Dict:
        """
        执行单笔交易

        Args:
            opportunity: 机会信息

        Returns:
            交易结果
        """
        symbol = opportunity['symbol']
        direction = opportunity['direction']
        entry_price = opportunity['last_price']
        score = opportunity['score']

        # 计算仓位大小(根据得分调整)
        if score >= 90:
            position_ratio = self.trade_config['position_size'] * 1.5
        elif score >= 80:
            position_ratio = self.trade_config['position_size']
        else:
            position_ratio = self.trade_config['position_size'] * 0.5

        # 计算止损止盈价
        if direction == 'LONG':
            stop_loss_price = entry_price * (1 - self.trade_config['stop_loss_pct'])
            take_profit_price = entry_price * (1 + self.trade_config['take_profit_pct'])
        else:
            stop_loss_price = entry_price * (1 + self.trade_config['stop_loss_pct'])
            take_profit_price = entry_price * (1 - self.trade_config['take_profit_pct'])

        # TODO: 实际调用交易API
        # result = self.okx_client.open_position(...)

        # 记录持仓
        self.positions[symbol] = {
            'direction': direction,
            'entry_price': entry_price,
            'position_size': position_ratio,
            'stop_loss': stop_loss_price,
            'take_profit': take_profit_price,
            'entry_time': opportunity['analysis_time'],
            'score': score,
            'signals': opportunity.get('signals', []),
        }

        print(f"✅ 开仓: {symbol} {direction}")
        print(f"   入场价: {entry_price}")
        print(f"   止损: {stop_loss_price}")
        print(f"   止盈: {take_profit_price}")
        print(f"   仓位: {position_ratio*100:.1f}%")
        print(f"   得分: {score}")
        print(f"   信号: {', '.join(opportunity.get('signals', [])[:3])}")
        print("-" * 60)

        return {
            'symbol': symbol,
            'action': 'OPEN',
            'direction': direction,
            'entry_price': entry_price,
            'stop_loss': stop_loss_price,
            'take_profit': take_profit_price,
            'position_size': position_ratio,
            'score': score,
        }

    def check_positions(self, current_prices: Dict[str, float]):
        """
        检查持仓,执行止损止盈

        Args:
            current_prices: 当前价格字典 {symbol: price}
        """
        symbols_to_close = []

        for symbol, position in self.positions.items():
            if symbol not in current_prices:
                continue

            current_price = current_prices[symbol]
            entry_price = position['entry_price']
            direction = position['direction']
            stop_loss = position['stop_loss']
            take_profit = position['take_profit']

            # 计算盈亏
            if direction == 'LONG':
                pnl_pct = (current_price - entry_price) / entry_price
            else:
                pnl_pct = (entry_price - current_price) / entry_price

            should_close = False
            close_reason = ''

            # 检查止损
            if direction == 'LONG' and current_price <= stop_loss:
                should_close = True
                close_reason = f"止损 ({pnl_pct*100:.2f}%)"
            elif direction == 'SHORT' and current_price >= stop_loss:
                should_close = True
                close_reason = f"止损 ({pnl_pct*100:.2f}%)"

            # 检查止盈
            if direction == 'LONG' and current_price >= take_profit:
                should_close = True
                close_reason = f"止盈 ({pnl_pct*100:.2f}%)"
            elif direction == 'SHORT' and current_price <= take_profit:
                should_close = True
                close_reason = f"止盈 ({pnl_pct*100:.2f}%)"

            # 移动止损逻辑
            if pnl_pct > 0.08:  # 盈利超过8%启用移动止损
                if direction == 'LONG':
                    trailing_stop = current_price * (1 - self.trade_config['trailing_stop_pct'])
                    if current_price <= trailing_stop:
                        should_close = True
                        close_reason = f"移动止损 ({pnl_pct*100:.2f}%)"
                else:
                    trailing_stop = current_price * (1 + self.trade_config['trailing_stop_pct'])
                    if current_price >= trailing_stop:
                        should_close = True
                        close_reason = f"移动止损 ({pnl_pct*100:.2f}%)"

            if should_close:
                # TODO: 实际调用平仓API
                # self.okx_client.close_position(symbol)

                print(f"❌ 平仓: {symbol}")
                print(f"   原因: {close_reason}")
                print(f"   盈亏: {pnl_pct*100:.2f}%")
                print("-" * 60)

                symbols_to_close.append(symbol)

        # 移除已平仓的
        for symbol in symbols_to_close:
            del self.positions[symbol]


# ==================== 使用示例 ====================

if __name__ == '__main__':
    """
    实战使用示例
    """

    # 示例1: 手动分析
    print("=" * 60)
    print("示例1: 手动分析单个交易对")
    print("=" * 60)

    # 模拟数据
    rave_symbol = ScannerSymbol(
        inst_id='RAVE-USDT-SWAP',
        last_price=1.01446,
        volume_24h=539000000,
        price_change_24h=123.59,
        high_24h=1.23800,
        low_24h=0.30497,
    )

    # 使用基础版
    basic_scanner = UnilateralTrendScanner(config={
        'min_volume_24h': 500000,
        'min_score': 60,
    })

    result = basic_scanner.scan_symbol(rave_symbol)
    print(f"\n交易对: {result['symbol']}")
    print(f"方向: {result['direction']}")
    print(f"得分: {result['score']}")
    print(f"评级: {result['rating']}")
    print(f"建议: {result['action']}")

    print("\n" + "=" * 60)
    print("示例2: 批量扫描")
    print("=" * 60)

    # 模拟多个交易对
    symbols_data = [
        ScannerSymbol(
            inst_id='RAVE-USDT-SWAP',
            last_price=1.01446,
            volume_24h=539000000,
            price_change_24h=123.59,
            high_24h=1.23800,
            low_24h=0.30497,
        ),
        ScannerSymbol(
            inst_id='BTC-USDT-SWAP',
            last_price=82500,
            volume_24h=2500000000,
            price_change_24h=3.5,
            high_24h=83000,
            low_24h=79500,
        ),
        ScannerSymbol(
            inst_id='ETH-USDT-SWAP',
            last_price=1850,
            volume_24h=1200000000,
            price_change_24h=5.2,
            high_24h=1870,
            low_24h=1750,
        ),
    ]

    all_results = basic_scanner.scan_all_symbols(symbols_data)
    print(f"\n扫描总数: {all_results['total_scanned']}")
    print(f"过滤后: {all_results['total_filtered']}")
    print(f"机会数: {all_results['total_opportunities']}")

    if all_results['long_opportunities']:
        print("\n🔥 做多机会:")
        for opp in all_results['long_opportunities'][:3]:
            print(f"  {opp['symbol']}: {opp['score']}分 - {opp['rating']}")

    if all_results['short_opportunities']:
        print("\n🔥 做空机会:")
        for opp in all_results['short_opportunities'][:3]:
            print(f"  {opp['symbol']}: {opp['score']}分 - {opp['rating']}")

    print("\n" + "=" * 60)
    print("示例3: 结合自动交易")
    print("=" * 60)

    # 初始化交易者
    # trader = UnilateralTrendTrader(okx_client, strategy_type='advanced')

    # 执行扫描和交易
    # result = trader.scan_and_trade(symbols_data)
    # print(f"执行了 {len(result['trades'])} 笔交易")
    # print(f"当前持仓: {result['total_positions']}")

    print("\n💡 提示: 取消注释上面的代码并传入真实的 okx_client 即可实战!")
