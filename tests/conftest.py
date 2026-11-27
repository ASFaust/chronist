import pytest


@pytest.fixture(autouse=True)
def _isolated_credentials(tmp_path, monkeypatch):
    """Never read or write the real ~/.config/chronist/credentials.json in tests."""
    monkeypatch.setenv("CHRONIST_CREDENTIALS", str(tmp_path / "credentials.json"))
    monkeypatch.delenv("CHRONIST_TOKEN", raising=False)
    monkeypatch.delenv("CHRONIST_PASSWORD", raising=False)
