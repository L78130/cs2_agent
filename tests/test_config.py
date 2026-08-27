# tests/test_config.py
from demo_coach.config import get_settings


def test_defaults(monkeypatch):
    monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)
    monkeypatch.delenv("KIMI_MODEL", raising=False)
    monkeypatch.delenv("KIMI_BASE_URL", raising=False)
    s = get_settings()
    assert s.api_key is None
    assert s.base_url == "https://api.kimi.com/coding/v1"
    assert s.model == "k3-256k"


def test_env_override(monkeypatch):
    monkeypatch.setenv("MOONSHOT_API_KEY", "sk-test")
    monkeypatch.setenv("KIMI_MODEL", "kimi-test-model")
    s = get_settings()
    assert s.api_key == "sk-test"
    assert s.model == "kimi-test-model"
