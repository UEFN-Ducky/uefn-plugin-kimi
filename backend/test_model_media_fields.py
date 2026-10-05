"""Kimi model records report image/video/audio limits only when present."""

from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path

ROOT = Path(os.path.abspath(__file__)).parents[1]
_APP = None
for _p in ROOT.parents:
    for _name in ("UEFN-Ducky-Release", "UEFN-Ducky-video", "UEFN-Ducky"):
        _app = _p / _name / "ducky_app"
        if (_app / "backend" / "agent").is_dir():
            _APP = str(_app)
            break
    else:
        continue
    break


def _load():
    # The plugin dir is itself a ``backend`` package; let the host's win while importing.
    saved = {k: sys.modules.pop(k) for k in list(sys.modules) if k == "backend" or k.startswith("backend.")}
    saved_path = list(sys.path)
    sys.path[:] = [_APP] + [p for p in sys.path if os.path.abspath(p) != str(ROOT)]
    try:
        return _load_module()
    finally:
        sys.path[:] = saved_path
        for k in [k for k in sys.modules if k == "backend" or k.startswith("backend.")]:
            del sys.modules[k]
        sys.modules.update(saved)


def _load_module():
    pkg = types.ModuleType("kimi_gw")
    pkg.__path__ = [str(ROOT / "backend")]
    sys.modules["kimi_gw"] = pkg
    prov = types.ModuleType("kimi_gw.kimi_provider")  # heavy; thinking helpers are irrelevant here
    prov.KIMI_BASE_URL = "https://api.moonshot.ai/v1"
    prov.kimi_supports_thinking = lambda mid: False
    prov.thinking_menu = lambda mid: None
    sys.modules["kimi_gw.kimi_provider"] = prov
    spec = importlib.util.spec_from_file_location("kimi_gw.model_fetch", ROOT / "backend" / "model_fetch.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _info(record=None):
    return _load()._info_from_id("kimi-x", record)


def test_media_fields_from_record():
    info = _info({"max_images": 10, "input_modalities": ["text", "image", "Video"]})
    assert (info.max_images, info.supports_video, info.supports_audio) == (10, True, False)


def test_audio_from_modalities():
    assert _info({"input_modalities": ["text", "audio"]}).supports_audio is True


def test_media_fields_unknown_without_record_data():
    for rec in (None, {}, {"id": "kimi-x"}):
        info = _info(rec)
        assert (info.max_images, info.supports_video, info.supports_audio) == (None, None, None)
