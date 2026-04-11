# GitHub Actions 云打包指南

## 📋 前置要求

- GitHub 账号（免费）
- Git 已安装
- 项目代码

---

## 🚀 快速开始（5步）

### 步骤1：初始化Git仓库（如果还没有）

```bash
cd "/Users/apple/Desktop/minimax 龙虾目录/CryptoTrader"
git init
```

### 步骤2：创建 .gitignore 文件

创建文件 `.gitignore`，排除不必要的文件：

```
# Python
__pycache__/
*.py[cod]
*.egg-info/
.venv/

# Buildozer
.buildozer/
bin/

# IDE
.vscode/
.idea/
*.swp

# OS
.DS_Store
Thumbs.db

# 敏感文件
scanner_config.json
*.key
*.jks
```

### 步骤3：提交代码

```bash
git add .
git commit -m "准备Android打包：添加云打包配置"
```

### 步骤4：创建GitHub仓库

1. 访问 https://github.com/new
2. 创建新仓库（例如：`CryptoTrader-Android`）
3. **不要**勾选"Initialize with README"

### 步骤5：推送代码到GitHub

```bash
# 替换YOUR_USERNAME为你的GitHub用户名
git remote add origin https://github.com/YOUR_USERNAME/CryptoTrader-Android.git
git branch -M main
git push -u origin main
```

---

## ⚙️ 触发自动打包

### 方式1：自动触发（推送代码时）

推送代码后会自动开始打包：

```bash
git add .
git commit -m "更新功能"
git push
```

### 方式2：手动触发

1. 访问你的GitHub仓库页面
2. 点击顶部 **Actions** 标签
3. 点击左侧 **Build Android APK**
4. 点击右上角 **Run workflow** 按钮
5. 选择分支（通常是 `main`）
6. 点击 **Run workflow**

---

## 📦 下载APK

打包完成后（约15-30分钟）：

### 方法1：从Actions下载

1. 访问仓库的 **Actions** 页面
2. 点击最新的工作流运行记录
3. 在页面底部的 **Artifacts** 部分
4. 点击 **CryptoScannerPro-Debug-APK** 下载
5. 解压后得到 `.apk` 文件

### 方法2：使用GitHub CLI下载

```bash
# 安装GitHub CLI
brew install gh

# 登录
gh auth login

# 列出最近的构建
gh run list

# 下载Artifacts
gh run download RUN_ID
```

---

## 🔧 打包配置说明

### 当前配置

工作流文件：`.github/workflows/build-android.yml`

**构建环境**：
- 操作系统：Ubuntu 22.04
- Python版本：3.10
- Android SDK：API 33
- NDK版本：25b

**缓存优化**：
- 首次构建：15-30分钟
- 后续构建：5-10分钟（有缓存）

**触发条件**：
- ✅ 推送到 main/master 分支
- ✅ 创建Pull Request
- ✅ 手动触发（workflow_dispatch）
- ✅ 创建Tag（会自动发布Release）

---

## 📊 打包状态

### 查看构建日志

1. 访问 **Actions** 标签
2. 点击具体的工作流运行
3. 点击 **build-android** job
4. 查看实时日志

### 常见问题

**Q: 构建失败**
- 查看日志中的错误信息
- 检查 `buildozer_android.spec` 配置
- 确认所有依赖文件存在

**Q: 构建时间过长**
- 首次构建需要下载Android SDK（正常）
- 后续构建会使用缓存，速度更快

**Q: 找不到APK**
- 确认构建成功（绿色✓）
- 检查 Artifacts 部分
- Artifacts保留30天

---

## 🎯 高级用法

### 发布版本（创建Release）

```bash
# 创建版本标签
git tag v1.0.0
git push origin v1.0.0
```

这会自动：
1. 触发构建
2. 创建GitHub Release
3. 上传APK到Release

### 自定义构建配置

编辑 `.github/workflows/build-android.yml` 文件：

```yaml
# 修改Python版本
python-version: '3.11'

# 修改Android API级别
android.api = 34

# 添加更多依赖
requirements = python3,kivy,kivymd,pandas,numpy,...
```

---

## 📱 安装APK

下载APK后：

### USB安装
```bash
adb install CryptoScannerPro-1.0.0-debug.apk
```

### 传输到手机
1. 通过微信/QQ发送APK文件
2. 在手机上点击下载
3. 允许安装未知来源应用
4. 完成安装

---

## ⚠️ 注意事项

1. **GitHub Actions限制**（免费账号）：
   - 每月2000分钟构建时间
   - 单次构建最长6小时
   - 并发5个job

2. **APK大小**：
   - 约30-50MB
   - 包含完整Python运行时

3. **网络要求**：
   - GitHub仓库需公开（免费Actions）
   - 或私有仓库（有分钟数限制）

4. **安全提醒**：
   - 不要在代码中硬编码API密钥
   - 使用GitHub Secrets存储敏感信息
   - APK中的默认配置应留空

---

## 📞 获取帮助

### 查看构建日志
```bash
# 使用GitHub CLI
gh run view --log

# 或在网页上查看Actions标签
```

### 常见问题排查
1. 检查workflows文件语法
2. 确认所有依赖文件存在
3. 查看错误日志
4. 在Issues中提问

---

## 🎉 完成！

设置完成后，每次推送代码都会自动：
- ✅ 构建Android APK
- ✅ 上传到Artifacts
- ✅ 保留30天可下载

**祝你使用愉快！📱✨**
