# 🚀 快速开始 - 3步打包Android APK

## 方式一：一键打包脚本（最简单）

```bash
cd "/Users/apple/Desktop/minimax 龙虾目录/CryptoTrader"
./build_android.sh
```

**就这么简单！** 脚本会自动：
- ✅ 检查Python环境
- ✅ 安装Buildozer（如未安装）
- ✅ 下载Android SDK/NDK（首次）
- ✅ 编译打包APK
- ✅ 显示安装指南

**预计时间**：首次30-60分钟，后续5-10分钟

---

## 方式二：手动打包

```bash
# 1. 安装依赖
pip3 install buildozer cython

# 2. 进入项目目录
cd "/Users/apple/Desktop/minimax 龙虾目录/CryptoTrader"

# 3. 开始打包
buildozer -v android debug
```

生成的APK位置：`bin/CryptoScannerPro-1.0.0-debug.apk`

---

## 方式三：云打包（无需本地环境）

1. 将代码推送到GitHub
2. 在GitHub Actions页面点击 "Run workflow"
3. 等待15-30分钟
4. 下载生成的APK

详细步骤见：[ANDROID_BUILD_GUIDE.md](ANDROID_BUILD_GUIDE.md)

---

## 📱 安装到手机

### 方法1：USB安装
```bash
# 启用USB调试后运行
adb install bin/CryptoScannerPro-1.0.0-debug.apk
```

### 方法2：HTTP传输
```bash
cd bin
python3 -m http.server 8000
# 手机浏览器访问：http://你的电脑IP:8000
```

---

## ⚙️ 首次使用配置

1. 打开应用
2. 填写OKX API配置：
   - API Key
   - Secret Key  
   - Passphrase
3. 点击"💾 保存配置"
4. 点击"🚀 开始扫描"

### 获取API密钥
访问 https://www.okx.com → 登录 → 用户中心 → API → 创建API

**安全提示**：仅开启"读取"权限，不要开启交易权限！

---

## ❓ 常见问题

**Q: 打包卡在下载SDK？**  
A: 首次需下载约2GB，请保持网络畅通或使用代理

**Q: 应用闪退？**  
A: 检查是否填写了正确的API密钥

**Q: 扫描无结果？**  
A: 确保API密钥有效且网络畅通

**Q: 国内无法访问OKX？**  
A: 需要配置代理，修改代码中的`proxy_url`参数

---

## 📚 更多帮助

- 完整打包指南：[ANDROID_BUILD_GUIDE.md](ANDROID_BUILD_GUIDE.md)
- 项目说明：[README.md](README.md)

---

**祝你使用愉快！📈**
