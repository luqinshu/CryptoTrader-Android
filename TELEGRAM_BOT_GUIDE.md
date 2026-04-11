# Telegram 机器人使用指南

## 一、安装依赖

```bash
pip3 install python-telegram-bot
```

## 二、创建 Telegram 机器人

1. 在 Telegram 中搜索 `@BotFather`
2. 发送 `/newbot` 创建新机器人
3. 按提示设置机器人名称和用户名
4. 复制获得的 **Bot Token**（格式类似：`123456789:ABCdefGHIjklMNOpqrSTUvwxyz`）

## 三、配置环境变量

```bash
# 设置 Telegram Bot Token
export TELEGRAM_BOT_TOKEN='your_bot_token_here'

# 设置 OKX API（如果使用）
export OKX_API_KEY='your_api_key'
export OKX_SECRET_KEY='your_secret_key'
export OKX_PASSPHRASE='your_passphrase'
export OKX_TESTNET='true'  # 测试网设为 true，实盘设为 false
export HTTP_PROXY='http://127.0.0.1:7897'  # 如果需要代理
```

## 四、启动机器人

```bash
python3 run_telegram_bot.py
```

## 五、在 Telegram 中使用

1. 在 Telegram 中搜索您创建的机器人用户名
2. 点击「开始」或发送 `/start`
3. 按照提示操作

## 六、可用命令

| 命令 | 功能 |
|------|------|
| `/start` | 启动机器人，显示主菜单 |
| `/help` | 查看帮助信息 |
| `/status` | 查看当前运行状态 |
| `/strategies` | 查看可用策略列表 |
| `/scan` | 手动执行扫描 |
| `/auto` | 设置自动扫描 |
| `/stop` | 停止当前扫描 |
| `/pool` | 查看交易对池 |
| `/config` | 配置设置 |

## 七、功能特点

✅ **手动扫描**：点击按钮或发送 `/scan` 立即扫描  
✅ **自动扫描**：支持定时自动扫描（1分钟/5分钟/15分钟/30分钟/1小时）  
✅ **结果推送**：扫描完成后自动推送符合条件的交易对  
✅ **交易对池**：累计所有扫描结果，随时查看  
✅ **多策略支持**：可切换不同扫描策略  
✅ **状态监控**：实时查看运行状态和账户信息  

## 八、注意事项

⚠️ **网络要求**：需要稳定的网络连接  
⚠️ **代理设置**：中国大陆用户需要配置 HTTP 代理  
⚠️ **API 限流**：频繁扫描可能触发 OKX API 限流  
⚠️ **Token 安全**：不要泄露 Bot Token，否则他人可以控制您的机器人  

## 九、常见问题

### Q: 机器人无响应？
A: 检查以下几点：
- Token 是否正确
- 网络连接是否正常
- 代理是否配置正确

### Q: 扫描结果为空？
A: 可能原因：
- 策略条件过于严格
- 当前市场不符合任何策略
- 尝试调整策略参数

### Q: 如何停止机器人？
A: 在终端按 `Ctrl+C` 即可停止

## 十、进阶配置

如需更详细的配置，请编辑 `src/telegram_bot/bot.py` 文件，修改以下参数：

- `self.auto_scan_interval`：自动扫描间隔（秒）
- `max_results_per_message`：每条消息显示的最大结果数
- 自定义命令处理逻辑
