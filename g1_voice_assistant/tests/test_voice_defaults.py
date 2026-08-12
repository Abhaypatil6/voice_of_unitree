import importlib
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_default_piper_model_is_clearer_voice(monkeypatch):
    import config.settings as settings

    monkeypatch.delenv("PIPER_MODEL", raising=False)

    with patch("dotenv.load_dotenv", return_value=None):
        reloaded = importlib.reload(settings)

    assert reloaded.config.tts.piper_model == "en_US-lessac-medium"
