# Crypto Trader - 量化交易系统

一个功能完整的加密货币量化交易平台，支持桌面端和移动端（Android）。

## 📱 移动端（Android）

### 快速打包（3步）

```bash
# 1. 检查环境
./check_env.sh

# 2. 一键打包
./build_android.sh

# 3. 安装到手机
adb install bin/CryptoScannerPro-1.0.0-debug.apk
```

### 移动端功能
- ✅ 全市场永续合约扫描
- ✅ 多周期技术分析（1D/1H/3m）
- ✅ 智能评分系统
- ✅ 动态止损止盈计算
- ✅ 本地配置自动保存
- ✅ 移动端友好界面

### 📖 完整文档
- [快速开始指南](QUICK_START.md) - 3步打包APK
- [Android打包完整指南](ANDROID_BUILD_GUIDE.md) - 三种打包方式
- [文件清单说明](ANDROID_FILE_MANIFEST.md) - 所有文件说明
- [打包总结](ANDROID_PACKAGING_SUMMARY.md) - 一站式汇总

---

## 🖥️ 桌面端（macOS/Linux）

## 功能特点

### 🎯 核心功能
- **自主策略加载**: 支持动态加载任意 Python 策略文件
- **策略配置界面**: 可视化配置策略参数
- **实时交易执行**: 连接 OKX 交易所执行买卖操作
- **持仓监控**: 实时显示持仓盈亏
- **交易日志**: 记录所有交易和策略运行日志
- **手动交易**: 支持手动买入/卖出操作
- **策略回测**: 完整的历史数据回测功能

### 📊 回测功能
- **多周期支持**: 1m, 5m, 15m, 30m, 1H, 2H, 4H, 1D
- **绩效分析**: 收益率、最大回撤、夏普比率、索提诺比率
- **交易统计**: 胜率、盈亏比、平均持仓时间
- **权益曲线**: 完整的资金变化记录
- **报告导出**: 支持导出文本格式回测报告

### 📈 技术支持
- OKX 交易所 API 对接（支持测试网）
- 多周期 K 线数据分析
- 动态策略发现和加载
- 多线程策略执行
- PyQt5 专业交易界面

## 安装

### 1. 安装依赖

```bash
pip3 install -r requirements.txt
```

### 2. 配置 OKX API

在 `main.py` 或代码中配置您的 OKX API 密钥：

```python
okx_client = OKXClient(
    api_key="您的 API_KEY",
    secret_key="您的 SECRET_KEY",
    passphrase="您的 PASSPHRASE",
    testnet=True,  # 测试网设为 True
    proxy_url="http://127.0.0.1:7897"  # 代理地址
)
```

## 使用方法

### 启动程序

```bash
cd /Users/apple/Desktop/minimax\ 龙虾目录/CryptoTrader
python3 main.py
```

### 使用步骤

1. **实时交易**:
   - 点击"实时交易"标签页
   - 在左侧策略列表中选择要使用的策略
   - 在策略配置区域设置策略参数
   - 选择交易对（如 BTC-USDT）
   - 设置杠杆倍数（现货为 1x）
   - 点击"启动策略"按钮开始自动交易
   - 在右侧查看持仓和交易日志

2. **策略回测**:
   - 点击"策略回测"标签页
   - 选择要回测的策略
   - 配置回测参数：
     - 交易对
     - K 线周期
     - 回测时间范围
     - 初始资金
     - 仓位比例
   - 点击"开始回测"
   - 查看回测结果：
     - 概览：总收益率、年化收益、最大回撤、夏普比率等
     - 交易记录：所有交易的详细信息
     - 详细报告：完整的回测分析报告
   - 可导出回测报告

### 策略开发

### 示例策略

项目包含两个示例策略：

1. **简单均线策略** (`simple_ma_strategy.py`)
   - 基于双均线金叉死叉的交易策略
   - 参数：快线周期、慢线周期、止损、止盈、仓位比例

2. **RSI 超买超卖策略** (`rsi_strategy.py`)
   - 基于 RSI 指标的均值回归策略
   - 参数：RSI 周期、超买线、超卖线、止损、止盈、仓位比例

### 策略文件结构

策略文件应包含一个策略类，建议命名包含 `Strategy` 或 `策略`：

```python
"""
策略说明文档
作者：作者名
版本：1.0
"""

class MyTradingStrategy:
    def __init__(self, config=None):
        """初始化策略"""
        self.config = config or {}
        # 策略参数
        self.stop_loss = self.config.get('stop_loss', 2.0)
        self.take_profit = self.config.get('take_profit', 8.0)
        
    def generate_signal(self, klines: dict) -> dict:
        """
        生成交易信号
        
        Args:
            klines: K 线数据字典，包含 daily, hourly, m15 等周期数据
            
        Returns:
            信号字典，包含 action, entry_price, position_size 等
        """
        # 分析市场数据
        # 生成交易信号
        
        return {
            'action': 'BUY',  # BUY, SELL, HOLD, STOP_LOSS
            'entry_price': 50000.0,
            'position_size': 0.1,  # 仓位比例
            'confidence': 0.8,
            'reason': '多周期共振买入信号'
        }
```

#### 策略配置模式

在策略文件中定义参数，系统会自动识别并生成配置界面：

```python
class MyStrategy:
    def __init__(self, config=None):
        self.stop_loss = self.config.get('stop_loss', 2.0)  # 止损百分比
        self.take_profit = self.config.get('take_profit', 10.0)  # 止盈百分比
        self.position_size = self.config.get('position_size', 0.1)  # 仓位比例
        self.rsi_period = self.config.get('rsi_period', 14)  # RSI 周期
```

系统会自动识别以下参数并生成配置输入框：
- `stop_loss`: 止损百分比
- `take_profit`: 止盈百分比
- `position_size`: 仓位比例
- `rsi_period`: RSI 周期
- 其他数值型参数

## 项目结构

```
CryptoTrader/
├── main.py                     # 主入口文件
├── requirements.txt            # Python 依赖
├── README.md                   # 项目说明
├── strategies/                 # 策略目录
│   ├── 专业波段策略 V4.0.py
│   ├── 波段交易策略 0.1 版.py
│   └── ...
├── src/
│   ├── api/
│   │   └── okx_client.py      # OKX API 客户端
│   ├── strategy/
│   │   ├── loader.py          # 策略加载器
│   │   └── runner.py          # 策略运行器
│   ├── trading/
│   │   └── executor.py        # 交易执行器
│   ├── backtest/
│   │   └── engine.py          # 回测引擎
│   └── ui/
│       ├── quant_trade_window.py  # 量化交易主窗口
│       └── backtest_page.py       # 回测页面
└── data/                      # 数据目录
```

## 界面说明

### 实时交易页面

#### 左侧面板 - 策略管理
- **策略列表**: 显示所有可加载的策略
- **策略配置**: 设置策略参数
- **交易对配置**: 选择交易对和杠杆
- **交易控制**: 启动/停止策略按钮

#### 右侧面板 - 交易监控
- **持仓监控**: 显示当前持仓、盈亏
- **手动交易**: 手动买入/卖出按钮

#### 底部 - 交易日志
- 显示策略运行日志
- 显示交易执行记录
- 显示错误和警告信息

### 策略回测页面

#### 左侧面板 - 回测配置
- **策略选择**: 选择要回测的策略
- **策略说明**: 显示策略详细信息
- **市场配置**: 交易对、K 线周期
- **回测时间**: 开始/结束日期
- **资金配置**: 初始资金、仓位比例
- **策略参数**: 策略特定参数配置

#### 右侧面板 - 回测结果
- **概览标签页**: 关键绩效指标
- **交易记录标签页**: 所有交易详情
- **详细报告标签页**: 完整回测报告
- **导出报告**: 保存回测结果

## 安全提示

⚠️ **重要**: 
1. 首次使用请在测试网测试
2. 不要投入超过您能承受的资金
3. 策略可能有风险，请充分测试
4. 保管好您的 API 密钥
5. 建议设置 API 密钥只能交易，不能提现

## 常见问题

### Q: 如何添加新策略？
A: 将策略 Python 文件放入 `strategies/` 目录，重启程序后会在策略列表中显示。

### Q: 策略不生效怎么办？
A: 检查策略类是否包含 `Strategy` 或 `策略` 字样，确保有 `generate_signal` 方法。

### Q: 如何停止策略？
A: 点击"停止策略"按钮，策略会立即停止运行。

### Q: 支持哪些交易所？
A: 当前仅支持 OKX 交易所。

## 许可证

MIT License

## 免责声明

本软件仅供学习和研究使用，不构成任何投资建议。使用本软件进行交易存在风险，您可能损失全部或部分本金。请谨慎使用。
