#!/bin/bash

#############################################################################
# CryptoScanner Pro - 打包环境检查工具
# 用于检查是否具备Android打包条件
#############################################################################

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# 计数
PASS=0
WARN=0
FAIL=0

print_check() {
    echo -e "${CYAN}[检查]${NC} $1"
}

print_pass() {
    echo -e "${GREEN}  ✓ $1${NC}"
    PASS=$((PASS + 1))
}

print_warn() {
    echo -e "${YELLOW}  ⚠ $1${NC}"
    WARN=$((WARN + 1))
}

print_fail() {
    echo -e "${RED}  ✗ $1${NC}"
    FAIL=$((FAIL + 1))
}

echo ""
echo -e "${BLUE}╔══════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║     CryptoScanner Pro - Android 打包环境检查       ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════╝${NC}"
echo ""

# 1. 检查核心文件
echo -e "${BLUE}[1/6] 核心文件检查${NC}"
echo ""

if [ -f "main_android.py" ]; then
    print_pass "main_android.py 存在"
else
    print_fail "main_android.py 不存在（必需）"
fi

if [ -f "buildozer_android.spec" ]; then
    print_pass "buildozer_android.spec 存在"
else
    print_fail "buildozer_android.spec 不存在（必需）"
fi

if [ -f "src/api/okx_client.py" ]; then
    print_pass "okx_client.py 存在"
else
    print_fail "okx_client.py 不存在（必需）"
fi

if [ -f "strategies/OKX小时线波段共振策略.py" ]; then
    print_pass "扫描策略文件存在"
else
    print_fail "扫描策略文件不存在（必需）"
fi

if [ -f "src/scanner/base_scanner.py" ]; then
    print_pass "base_scanner.py 存在"
else
    print_fail "base_scanner.py 不存在（必需）"
fi

echo ""

# 2. 检查Python
echo -e "${BLUE}[2/6] Python 环境检查${NC}"
echo ""

if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | awk '{print $2}')
    print_pass "Python3 已安装: $PYTHON_VERSION"
    
    # 检查版本是否 >= 3.8
    PY_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
    PY_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
    
    if [ "$PY_MAJOR" -ge 3 ] && [ "$PY_MINOR" -ge 8 ]; then
        print_pass "Python 版本符合要求 (>= 3.8)"
    else
        print_fail "Python 版本过低，需要 3.8+"
    fi
else
    print_fail "未安装 Python3"
    print_warn "安装方法: brew install python3 (macOS)"
fi

echo ""

# 3. 检查依赖包
echo -e "${BLUE}[3/6] Python 依赖包检查${NC}"
echo ""

DEPS=("pandas" "numpy" "ta" "requests" "kivymd" "kivy")

for dep in "${DEPS[@]}"; do
    if python3 -c "import $dep" 2>/dev/null; then
        VERSION=$(python3 -c "import $dep; print($dep.__version__)" 2>/dev/null || echo "unknown")
        print_pass "$dep 已安装 ($VERSION)"
    else
        print_warn "$dep 未安装（打包时会自动包含）"
    fi
done

echo ""

# 4. 检查Buildozer
echo -e "${BLUE}[4/6] Buildozer 检查${NC}"
echo ""

if command -v buildozer &> /dev/null; then
    BUILDOZER_VERSION=$(buildozer --version 2>/dev/null || echo "unknown")
    print_pass "Buildozer 已安装: $BUILDOZER_VERSION"
    
    # 检查Java（Buildozer依赖）
    if command -v java &> /dev/null; then
        JAVA_VERSION=$(java -version 2>&1 | head -n 1 | cut -d'"' -f2)
        print_pass "Java 已安装: $JAVA_VERSION"
    else
        print_warn "Java 未安装（Buildozer需要）"
        print_warn "安装方法: brew install --cask temurin (macOS)"
    fi
else
    print_warn "Buildozer 未安装"
    print_warn "安装命令: pip3 install --user buildozer cython"
fi

echo ""

# 5. 检查磁盘空间
echo -e "${BLUE}[5/6] 磁盘空间检查${NC}"
echo ""

DISK_AVAILABLE=$(df -h . | tail -1 | awk '{print $4}' | sed 's/G//')
if [ -n "$DISK_AVAILABLE" ]; then
    if (( $(echo "$DISK_AVAILABLE > 10" | bc -l 2>/dev/null || echo 1) )); then
        print_pass "可用磁盘空间: ${DISK_AVAILABLE}GB (建议 > 10GB)"
    else
        print_warn "可用磁盘空间: ${DISK_AVAILABLE}GB (建议 > 10GB)"
        print_warn "首次打包需要下载约2GB的Android SDK/NDK"
    fi
else
    print_warn "无法检测磁盘空间"
fi

echo ""

# 6. 检查网络
echo -e "${BLUE}[6/6] 网络连接检查${NC}"
echo ""

if ping -c 1 -W 3 www.okx.com &> /dev/null; then
    print_pass "可以访问 OKX API"
else
    print_warn "无法访问 OKX API（国内可能需要代理）"
fi

if ping -c 1 -W 3 pypi.org &> /dev/null; then
    print_pass "可以访问 PyPI（安装依赖需要）"
else
    print_warn "无法访问 PyPI"
fi

echo ""

# 总结
echo -e "${BLUE}══════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${GREEN}通过: $PASS${NC}"
echo -e "${YELLOW}警告: $WARN${NC}"
echo -e "${RED}失败: $FAIL${NC}"
echo ""

if [ $FAIL -eq 0 ]; then
    echo -e "${GREEN}╔══════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║           ✅ 环境检查通过！可以开始打包            ║${NC}"
    echo -e "${GREEN}╚══════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo "运行以下命令开始打包："
    echo ""
    echo -e "${CYAN}  ./build_android.sh${NC}"
    echo ""
else
    echo -e "${RED}╔══════════════════════════════════════════════════════╗${NC}"
    echo -e "${RED}║           ❌ 存在关键问题，请先解决                ║${NC}"
    echo -e "${RED}╚══════════════════════════════════════════════════════╝${NC}"
    echo ""
fi

if [ $WARN -gt 0 ]; then
    echo -e "${YELLOW}提示：${NC}有 $WARN 个警告，这些在打包过程中会自动解决"
    echo ""
fi

# 推荐下一步
echo -e "${BLUE}推荐操作：${NC}"
echo ""
echo "  1. 查看快速开始指南: cat QUICK_START.md"
echo "  2. 运行一键打包: ./build_android.sh"
echo "  3. 查看完整文档: cat ANDROID_BUILD_GUIDE.md"
echo ""
