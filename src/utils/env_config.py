"""
环境变量与 .env 配置加载工具。
"""

import os
from typing import Dict


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ENV_FILE_PATH = os.path.join(PROJECT_ROOT, ".env")


def load_dotenv_file(path: str = ENV_FILE_PATH) -> Dict[str, str]:
    """手动加载 .env 文件到环境变量，不依赖 python-dotenv。"""
    loaded = {}
    if not os.path.exists(path):
        return loaded

    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if not key:
                continue
            os.environ[key] = value
            loaded[key] = value
    return loaded


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def get_okx_config() -> Dict[str, object]:
    """获取 OKX 客户端配置。"""
    load_dotenv_file()
    return {
        "api_key": os.getenv("OKX_API_KEY", "").strip(),
        "secret_key": os.getenv("OKX_SECRET_KEY", "").strip(),
        "passphrase": os.getenv("OKX_PASSPHRASE", "").strip(),
        "testnet": _env_bool("OKX_TESTNET", True),
        "proxy_url": os.getenv("OKX_PROXY_URL", "").strip() or None,
    }


def missing_okx_config_fields(config: Dict[str, object]) -> list[str]:
    """返回缺失的关键配置字段。"""
    missing = []
    if not str(config.get("api_key", "")).strip():
        missing.append("OKX_API_KEY")
    if not str(config.get("secret_key", "")).strip():
        missing.append("OKX_SECRET_KEY")
    if not str(config.get("passphrase", "")).strip():
        missing.append("OKX_PASSPHRASE")
    return missing
