# Android 打包完整指南

## 📱 应用说明

**CryptoScanner Pro** 是一个基于 OKX API 的加密货币交易对扫描器，可以自动分析数百个交易对，找出符合技术形态的交易机会。

### 核心功能
- ✅ 全市场永续合约扫描
- ✅ 多周期技术分析（1D/1H/3m）
- ✅ 智能评分系统
- ✅ 动态止损止盈计算
- ✅ 本地配置保存
- ✅ 移动端友好界面

---

## 🛠️ 打包环境要求

### 方式一：使用 macOS/Linux 本地打包（推荐）

#### 1. 系统要求
```bash
- macOS 10.15+ 或 Ubuntu 20.04+
- Python 3.8 - 3.10
- 至少 10GB 磁盘空间
- 稳定的网络连接（首次需下载 ~2GB Android SDK）
```

#### 2. 安装依赖

```bash
# 安装 Homebrew (macOS)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 安装必要工具
brew install python3 java wget autoconf automake libtool pkg-config cmake

# 安装 buildozer
pip3 install --user buildozer cython

# 安装项目依赖
pip3 install pandas numpy ta requests kivymd kivy
```

#### 3. 打包步骤

```bash
# 进入项目目录
cd "/Users/apple/Desktop/minimax 龙虾目录/CryptoTrader"

# 首次打包（会自动下载 Android SDK/NDK）
buildozer -v android debug

# 后续快速打包
buildozer android debug
```

**预计首次打包时间：30-60 分钟**（取决于网速）

#### 4. 生成的 APK 位置
```
bin/CryptoScannerPro-1.0.0-debug.apk
```

---

### 方式二：使用 Docker 打包（最稳定）

#### 1. 安装 Docker
```bash
# macOS
brew install --cask docker

# 启动 Docker Desktop
open -a Docker
```

#### 2. 使用官方 Buildozer 镜像
```bash
cd "/Users/apple/Desktop/minimax 龙虾目录/CryptoTrader"

# 运行打包
docker run -v "$(pwd)":/home/user/hostcwd \
  -v /tmp/.buildozer-cache:/home/user/.buildozer \
  --name buildozer \
  kivytoolchain/buildozer \
  buildozer -v android debug
```

---

### 方式三：使用 GitHub Actions 云打包（无需本地环境）

#### 1. 创建 `.github/workflows/build-android.yml`

```yaml
name: Build Android APK

on:
  push:
    branches: [ main ]
  workflow_dispatch:

jobs:
  build:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install buildozer cython
    
    - name: Cache Buildozer
      uses: actions/cache@v3
      with:
        path: |
          .buildozer
          ~/.buildozer
        key: ${{ runner.os }}-buildozer-${{ hashFiles('**/buildozer_android.spec') }}
    
    - name: Build APK
      run: |
        buildozer -v android debug
    
    - name: Upload APK
      uses: actions/upload-artifact@v3
      with:
        name: CryptoScannerPro-APK
        path: bin/*.apk
```

#### 2. 推送代码到 GitHub
```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/CryptoTrader.git
git push -u origin main
```

#### 3. 在 GitHub Actions 页面手动触发构建
- 访问：`https://github.com/YOUR_USERNAME/CryptoTrader/actions`
- 点击 "Build Android APK" → "Run workflow"

---

## 📦 安装到手机

### 方法 1：USB 传输
```bash
# macOS 使用 Android File Transfer
# 1. 手机开启 USB 调试
# 2. 连接电脑
# 3. 将 APK 复制到手机
# 4. 在手机上点击安装

# 或使用 adb 直接安装
adb install bin/CryptoScannerPro-1.0.0-debug.apk
```

### 方法 2：网络传输
```bash
# 1. 将 APK 上传到云盘（Google Drive、百度网盘等）
# 2. 手机下载并安装
```

### 方法 3：HTTP 服务器
```bash
# 在电脑启动临时 HTTP 服务器
cd bin
python3 -m http.server 8000

# 手机浏览器访问：http://YOUR_COMPUTER_IP:8000
# 点击下载 APK 并安装
```

---

## ⚙️ 应用配置

### 首次使用需配置
1. **OKX API Key** - 从 OKX 获取
2. **Secret Key** - API 密钥
3. **Passphrase** - 密码短语

### 获取 OKX API 密钥
1. 访问 https://www.okx.com
2. 登录账户
3. 用户中心 → API → 创建 API
4. 权限：仅开启"读取"（安全起见不要开启交易权限）
5. 复制 API Key、Secret Key、Passphrase

### 代理配置（国内用户）
当前版本为简化版，未内置代理功能。如需代理，请修改 `main_android.py` 中的 `OKXClient` 初始化部分：

```python
self.okx_client = OKXClient(
    api_key=self.api_key_input.text,
    secret_key=self.secret_key_input.text,
    passphrase=self.pass_input.text,
    testnet=False,
    proxy_url="http://YOUR_PROXY_IP:PORT"  # 添加此行
)
```

---

## 🔧 常见问题

### Q1: 打包失败 "SDK not found"
```bash
# 手动下载 Android SDK
wget https://dl.google.com/android/repository/commandlinetools-mac-9477386_latest.zip
unzip commandlinetools-mac-9477386_latest.zip
export ANDROID_HOME=$HOME/Android/Sdk
export PATH=$PATH:$ANDROID_HOME/cmdline-tools/latest/bin
```

### Q2: 打包时依赖安装失败
```bash
# 清理缓存重新打包
buildozer android clean
buildozer -v android debug
```

### Q3: APK 安装失败
```bash
# 检查是否开启"允许安装未知来源应用"
# 卸载旧版本后重新安装
adb uninstall org.cryptotrader.cryptoscannerpro
adb install bin/CryptoScannerPro-1.0.0-debug.apk
```

### Q4: 运行时崩溃
```bash
# 查看日志
adb logcat | python -m kivymd.tools.log

# 或在手机上使用 logcat 应用查看日志
```

### Q5: 扫描无结果
- 检查 API Key 是否正确
- 确保网络畅通
- 降低评分阈值（修改代码中 `score >= 70` 为更低的值）

---

## 📊 性能优化建议

### 减少扫描数量
在 `main_android.py` 中修改：
```python
][:30]  # 改为更小的数字，如 10 或 20
```

### 简化技术指标
修改策略文件 `strategies/OKX小时线波段共振策略.py`，减少指标计算。

### 使用缓存
添加交易对数据缓存，避免重复请求。

---

## 🚀 发布正式版

### 签名 APK
```bash
# 1. 生成密钥
keytool -genkey -v -keystore my-release-key.jks -keyalg RSA -keysize 2048 -validity 10000 -alias my-alias

# 2. 修改 buildozer_android.spec
android.release_discharge = True
android.keystore = my-release-key.jks

# 3. 打包发布版
buildozer android release
```

### 上架应用市场
- 准备应用截图和描述
- 生成应用图标（1024x1024 PNG）
- 提交到各大应用市场

---

## 📝 版本更新日志

### v1.0.0 (2026-04-11)
- ✅ 首次发布
- ✅ 支持 OKX 永续合约扫描
- ✅ 多周期技术分析
- ✅ 智能评分系统
- ✅ 本地配置保存
- ✅ 移动端友好界面

---

## 📞 技术支持

如遇到问题，请提供以下信息：
1. 系统环境（macOS/Ubuntu 版本）
2. Python 版本
3. 完整错误日志
4. buildozer 版本

---

## ⚠️ 免责声明

本工具仅供学习和技术研究使用，使用本工具进行交易决策需自行承担风险。作者不对任何交易损失负责。

---

**祝交易顺利！📈**
