# tests/test_config.py
import os
from demo_coach.config import get_settings


def test_defaults(monkeypatch):
    monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)
    monkeypatch.delenv("KIMI_MODEL", raising=False)
    s = get_settings()
    assert s.api_key is None
    assert s.base_url == "https://api.moonshot.cn/v1"
    assert s.model == "kimi-k2-0711-preview"


def test_env_override(monkeypatch):
    monkeypatch.setenv("MOONSHOT_API_KEY", "sk-test")
    monkeypatch.setenv("KIMI_MODEL", "kimi-test-model")
    s = get_settings()
    assert s.api_key == "sk-test"
    assert s.model == "kimi-test-model"
