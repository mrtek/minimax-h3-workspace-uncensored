"""Live system readings for the sidebar: CPU, RAM, GPU (via nvidia-smi) and free disk space.

Nothing here needs extra packages: CPU load comes from GetSystemTimes, RAM from GlobalMemoryStatusEx.
"""

import ctypes
import shutil
import subprocess
import threading
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
        self._lock = threading.Lock()
        self._cpu_prev = None
        self._cache, self._at = None, 0.0

    def cpu_percent(self):
        try:
            idle, kernel, user = _FileTime(), _FileTime(), _FileTime()
            if not ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)):
                return None
            now = (_ticks(idle), _ticks(kernel) + _ticks(user))  # kernel time already includes idle time
            prev, self._cpu_prev = self._cpu_prev, now
            if not prev or now[1] <= prev[1]:
                return None
            val = round(100 * (1 - (now[0] - prev[0]) / (now[1] - prev[1])), 1)
            return max(0.0, min(100.0, val))
        except Exception:
            return None

    def cpu_temp(self):
        """CPU temperature in degrees Celsius, or None if unavailable.

        Windows user-mode application runs with PrivilegesRequired=lowest.
        Without Ring-0 drivers or administrative elevation, direct CPU DTS/MSR readings
        are unavailable. Returns None safely without blocking or raising exceptions.
        """
        try:
            return None
        except Exception:
            return None

    @staticmethod
    def ram():
        try:
            m = _MemStatus(dwLength=ctypes.sizeof(_MemStatus))
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
                return None
            total = m.ullTotalPhys / GB
            if total <= 0:
                return None
            used = total - (m.ullAvailPhys / GB)
            return {"total_gb": round(total, 1), "used_gb": round(max(0.0, used), 1)}
        except Exception:
            return None

    @staticmethod
    def gpu():
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit",
                 "--format=csv,noheader,nounits"],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, timeout=4,
                creationflags=engine.CREATE_NO_WINDOW).stdout
            rows = [[x.strip() for x in line.split(",")] for line in out.strip().splitlines() if line.strip()]
            valid_rows = [r for r in rows if len(r) >= 7]
            if not valid_rows:
                return None
            r = max(valid_rows, key=lambda r: _num(r[3]) or 0)  # the card with the most memory is the one in use
            used, total = _num(r[2]), _num(r[3])
            return {
                "name": r[0].replace("NVIDIA GeForce ", ""),
                "util": _num(r[1]),
                "vram_used_gb": round(used / 1024, 1) if used is not None else None,
                "vram_total_gb": round(total / 1024, 1) if total is not None else None,
                "temp": _num(r[4]),
                "power": _num(r[5]),
                "power_limit": _num(r[6]),
            }
        except Exception:
            return None

    @staticmethod
    def disk(path):
        try:
            p = Path(path)
            while not p.exists() and p.parent != p:
                p = p.parent
            u = shutil.disk_usage(p)
            return {"free_gb": round(u.free / GB), "total_gb": round(u.total / GB), "drive": p.drive or str(p)}
        except Exception:
            return None

    def invalidate_cache(self):
        """Invalidate cached metrics so the next sample() call fetches fresh data."""
        with self._lock:
            self._cache = None
            self._at = 0.0

    def sample(self, disk_path):
        """At most one reading per second, however many windows ask."""
        with self._lock:
            now = time.monotonic()
            if self._cache is not None and (now - self._at) < 1.0:
                return self._cache

            try:
                c_cpu = self.cpu_percent()
            except Exception:
                c_cpu = None

            try:
                c_temp = self.cpu_temp()
            except Exception:
                c_temp = None

            try:
                c_ram = self.ram()
            except Exception:
                c_ram = None

            try:
                c_gpu = self.gpu()
            except Exception:
                c_gpu = None

            try:
                c_disk = self.disk(disk_path)
            except Exception:
                c_disk = None

            fresh = {
                "cpu": c_cpu,
                "cpu_temp": c_temp,
                "ram": c_ram,
                "gpu": c_gpu,
                "disk": c_disk,
            }
            self._cache = fresh
            self._at = now
            return fresh
