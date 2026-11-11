import os
import sys
import tempfile
from pathlib import Path

APP = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP))

_tmp = tempfile.mkdtemp(prefix="h3test_")
os.environ.setdefault("H3STUDIO_DATA", _tmp)   # server.py reads this at import time
