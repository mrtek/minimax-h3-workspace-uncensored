import json
import tempfile
from pathlib import Path
import engine
import server


def test_broken_config_json_quarantined():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "config.json"
        p.write_text("{", encoding="utf-8")
        cfg = engine.Config(p)
        assert cfg.data == {}
        broken_files = list(Path(td).glob("config.json.broken-*"))
        assert len(broken_files) == 1
        assert broken_files[0].read_text(encoding="utf-8") == "{"
        # Atomic update works
        cfg.update(key="val")
        assert p.exists()
        assert json.loads(p.read_text(encoding="utf-8")) == {"key": "val"}


def test_invalid_type_config_json_quarantined():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "config.json"
        p.write_text('["not", "a", "dict"]', encoding="utf-8")
        cfg = engine.Config(p)
        assert cfg.data == {}
        broken_files = list(Path(td).glob("config.json.broken-*"))
        assert len(broken_files) == 1


def test_valid_config_json_loaded():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "config.json"
        p.write_text('{"existing": 123}', encoding="utf-8")
        cfg = engine.Config(p)
        assert cfg.data == {"existing": 123}
        broken_files = list(Path(td).glob("config.json.broken-*"))
        assert len(broken_files) == 0


def test_broken_library_json_quarantined():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "library.json"
        p.write_text("{", encoding="utf-8")
        lib = server.Library(p)
        assert lib.items == []
        broken_files = list(Path(td).glob("library.json.broken-*"))
        assert len(broken_files) == 1
        assert broken_files[0].read_text(encoding="utf-8") == "{"


def test_dict_instead_of_list_library_json_quarantined():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "library.json"
        p.write_text('{"id": "item"}', encoding="utf-8")
        lib = server.Library(p)
        assert lib.items == []
        broken_files = list(Path(td).glob("library.json.broken-*"))
        assert len(broken_files) == 1


def test_valid_library_json_loaded():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "library.json"
        p.write_text('[{"id": "1", "name": "test"}, {"bad": "no id"}]', encoding="utf-8")
        lib = server.Library(p)
        assert len(lib.items) == 1
        assert lib.items[0]["id"] == "1"
        broken_files = list(Path(td).glob("library.json.broken-*"))
        assert len(broken_files) == 0
