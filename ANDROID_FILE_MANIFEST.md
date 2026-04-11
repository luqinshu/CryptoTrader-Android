# CryptoScanner Pro - Android 打包文件清单

## 📁 核心文件（打包必需）

### 入口文件
- `main_android.py` - Android版本的主程序入口
  - 移除硬编码API密钥
  - 添加配置保存功能
  - 优化移动端UI体验
  - 线程安全的扫描逻辑

### 配置文件
- `buildozer_android.spec` - Buildozer打包配置
  - 应用名称、包名、版本
  - Python依赖列表
  - Android SDK/NDK版本
  - 权限和屏幕方向

### 依赖模块
- `src/api/okx_client.py` - OKX API客户端
- `strategies/OKX小时线波段共振策略.py` - 核心扫描策略
- `src/scanner/base_scanner.py` - 扫描器基类

---

## 📦 打包工具

### 自动化脚本
- `build_android.sh` - 一键打包脚本
  - 环境检查
  - 依赖安装
  - 自动打包
  - 安装指南

### GitHub Actions
- `.github/workflows/build-android.yml` - 云打包工作流
  - 自动构建
  - APK上传
  - 版本发布

---

## 📖 文档

### 快速开始
- `QUICK_START.md` - 3步快速开始指南
  - 最简单的打包流程
  - 常见问题解答

### 详细指南
- `ANDROID_BUILD_GUIDE.md` - 完整打包指南
  - 三种打包方式详解
  - 环境配置
  - 故障排除
  - 发布指南

### 项目文档
- `README.md` - 项目总体说明
- `TELEGRAM_BOT_GUIDE.md` - Telegram机器人配置

---

## 🗑️ 不需要打包的文件

以下文件在Android打包时**不需要**：

### 其他平台入口
- `main.py` - 原始Kivy版本
- `main_gui.py` - PyQt5版本
- `main_tk.py` - Tkinter版本
- `main_simple.py` - 简化版
- `main_complete.py` - 完整版
- `main_macos.py` - macOS版本
- `main_fixed.py` - 修复版
- `main_full.py` - 全功能版

### 其他功能
- `cli_scanner.py` - 命令行扫描器
- `mobile_scanner.py` - 旧版移动端
- `run_simple_bot.py` - 简单机器人
- `run_telegram_bot.py` - Telegram机器人

### 测试和配置
- `test_keep_alive.py` - 测试脚本
- `buildozer.spec` - 旧版配置
- `更新说明_倒计时功能.md` - 更新日志

---

## 📊 文件依赖关系

```
main_android.py (入口)
    ├── src/api/okx_client.py (API请求)
    ├── strategies/OKX小时线波段共振策略.py (扫描逻辑)
    │   └── src/scanner/base_scanner.py (基类)
    └── KivyMD UI组件
         └── 打包为 APK
```

---

## ✅ 打包前检查清单

在运行 `build_android.sh` 之前，确保：

- [ ] `main_android.py` 存在且可运行
- [ ] `buildozer_android.spec` 配置正确
- [ ] `src/api/okx_client.py` 存在
- [ ] `strategies/OKX小时线波段共振策略.py` 存在
- [ ] `src/scanner/base_scanner.py` 存在
- [ ] Python 3.8+ 已安装
- [ ] 网络连接稳定

---

## 📝 打包后生成的文件

```
.buildozer/          # 构建缓存（可删除）
bin/
    └── CryptoScannerPro-1.0.0-debug.apk  # 最终APK
scanner_config.json  # 用户配置（运行时生成）
```

---

## 🔧 可选优化文件

### 应用图标（推荐添加）
- `icon.png` - 应用图标（1024x1024 PNG）
- 在 `buildozer_android.spec` 中添加：
  ```
  icon.filename = icon.png
  ```

### 启动画面（可选）
- `splash.png` - 启动画面
- 在配置中启用：
  ```
  android.presplash_filename = splash.png
  ```

---

## 💡 提示

1. **首次打包**会下载Android SDK/NDK（约2GB），请耐心等待
2. **APK大小**约30-50MB，包含所有Python依赖
3. **架构支持**：arm64-v8a, armeabi-v7a（覆盖99%的Android手机）
4. **最低版本**：Android 5.0 (API 21)

---

**准备好开始了吗？运行 `./build_android.sh` 吧！🚀**
