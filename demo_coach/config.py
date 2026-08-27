# demo_coach/config.py
import os
from dataclasses import dataclass

DEFAULT_BASE_URL = "https://api.moonshot.cn/v1"
DEFAULT_MODEL = "kimi-k2-0711-preview"


@dataclass(frozen=True)
class Settings:
    api_key: str | None
    base_url: str
    model: str


def get_settings() -> Settings:
    return Settings(
        api_key=os.environ.get("MOONSHOT_API_KEY"),
        base_url=os.environ.get("KIMI_BASE_URL", DEFAULT_BASE_URL),
        model=os.environ.get("KIMI_MODEL", DEFAULT_MODEL),
    )
