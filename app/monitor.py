"""Live system readings for the sidebar: CPU, RAM, GPU (via nvidia-smi) and free disk space.

Nothing here needs extra packages: CPU load comes from GetSystemTimes, RAM from GlobalMemoryStatusEx.
"""

import ctypes
import shutil
import subprocess
import time
from pathlib import Path

import engine

GB = 1024 ** 3


class _FileTime(ctypes.Structure):
    _fields_ = [("lo", ctypes.c_ulong), ("hi", ctypes.c_ulong)]


class _MemStatus(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


def _ticks(ft):
    return (ft.hi << 32) | ft.lo


def _num(text):
    try:
        return float(text)
    except (TypeError, ValueError):
        return None  # "[N/A]" and the like


class Monitor:
    def __init__(self):
        self._cpu_prev = None
        self._cache, self._at = None, 0.0

    def cpu_percent(self):
        idle, kernel, user = _FileTime(), _FileTime(), _FileTime()
        if not ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)):
            return None
        now = (_ticks(idle), _ticks(kernel) + _ticks(user))  # kernel time already includes idle time
        prev, self._cpu_prev = self._cpu_prev, now
        if not prev or now[1] <= prev[1]:
            return None
        return round(100 * (1 - (now[0] - prev[0]) / (now[1] - prev[1])), 1)

    @staticmethod
    def ram():
        m = _MemStatus(dwLength=ctypes.sizeof(_MemStatus))
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        total = m.ullTotalPhys / GB
        return {"total_gb": round(total, 1), "used_gb": round(total - m.ullAvailPhys / GB, 1)}

    @staticmethod
    def gpu():
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit",
                 "--format=csv,noheader,nounits"],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, timeout=4,
                creationflags=engine.CREATE_NO_WINDOW).stdout
        except Exception:
            return None
        rows = [[x.strip() for x in line.split(",")] for line in out.strip().splitlines() if line.strip()]
        if not rows:
            return None
        r = max(rows, key=lambda r: _num(r[3]) or 0)  # the card with the most memory is the one in use
        used, total = _num(r[2]), _num(r[3])
        return {"name": r[0].replace("NVIDIA GeForce ", ""), "util": _num(r[1]),
                "vram_used_gb": round(used / 1024, 1) if used is not None else None,
                "vram_total_gb": round(total / 1024, 1) if total is not None else None,
                "temp": _num(r[4]), "power": _num(r[5]), "power_limit": _num(r[6])}

    @staticmethod
    def disk(path):
        p = Path(path)
        while not p.exists() and p.parent != p:
            p = p.parent
        try:
            u = shutil.disk_usage(p)
        except OSError:
            return None
        return {"free_gb": round(u.free / GB), "total_gb": round(u.total / GB), "drive": p.drive or str(p)}

    def sample(self, disk_path):
        """At most one reading per second, however many windows ask."""
        if self._cache and time.monotonic() - self._at < 1.0:
            return self._cache
        self._cache = {"cpu": self.cpu_percent(), "ram": self.ram(), "gpu": self.gpu(), "disk": self.disk(disk_path)}
        self._at = time.monotonic()
        return self._cache
