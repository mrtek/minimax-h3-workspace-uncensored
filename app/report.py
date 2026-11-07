"""A text report for the author: what computer this is, what the app sees and what went wrong.

The user presses one button, gets a .txt file and sends it. Nothing is sent by the app itself.
Private things are left out: the RunningHub key and instance ids are never written, and the Windows user
name inside paths is replaced with <user>.
"""
import json
import platform
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

try:
    import winreg
except ImportError:  # not Windows (tests)
    winreg = None

from version import APP_VERSION

NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
SECRET_KEYS = {"rh_key", "rh_instance", "rh_workflow_id", "rh_host"}
WEBVIEW2_ID = "{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}"


def webview2_version():
    """Version of the Edge WebView2 runtime that draws the window (an old one breaks the layout)."""
    if not winreg:
        return "нет данных"
    for hive, sub in ((winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients"),
                      (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\EdgeUpdate\Clients"),
                      (winreg.HKEY_CURRENT_USER, r"Software\Microsoft\EdgeUpdate\Clients")):
        try:
            with winreg.OpenKey(hive, sub + "\\" + WEBVIEW2_ID) as k:
                return str(winreg.QueryValueEx(k, "pv")[0])
        except OSError:
            continue
    return "не найден (окно откроется в обычном браузере или не откроется)"


def windows_version():
    try:
        out = subprocess.run(["cmd", "/c", "ver"], capture_output=True, text=True, timeout=5,
                             creationflags=NO_WINDOW).stdout.strip()
    except Exception:
        out = ""
    return f"{platform.system()} {platform.release()} ({platform.version()}) {out}".strip()


def nvidia_smi_info():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,compute_cap",
                              "--format=csv,noheader"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, timeout=10,
                             creationflags=NO_WINDOW).stdout.strip()
        return out or "nvidia-smi ничего не вернул (нет видеокарты NVIDIA или не установлен драйвер)"
    except FileNotFoundError:
        return "nvidia-smi не найден (нет драйвера NVIDIA)"
    except Exception as e:
        return f"nvidia-smi: ошибка {type(e).__name__}"


def cpu_name():
    try:
        out = subprocess.run(["wmic", "cpu", "get", "name"], capture_output=True, text=True, timeout=8,
                             creationflags=NO_WINDOW).stdout.split("\n")
        return next((x.strip() for x in out[1:] if x.strip()), platform.processor())
    except Exception:
        return platform.processor()


def mask(text):
    """Hide the Windows user name in paths and anything that looks like a key."""
    text = str(text)
    user = Path.home().name
    if user:
        text = re.sub(re.escape(user), "<user>", text, flags=re.I)
    return re.sub(r"\b(sk-|rh_|key[-_=: ]+)[A-Za-z0-9_\-]{12,}", r"\1***", text)


def safe_config(cfg):
    out = {}
    for k, v in (cfg or {}).items():
        out[k] = "<скрыто>" if k in SECRET_KEYS else v
    return out


def build(info):
    """info: dict made by the server; returns the text of the report."""
    L = []

    def sec(title):
        L.extend(["", "=== " + title + " ==="])

    L.append("ОТЧЁТ ДЛЯ РАЗРАБОТЧИКА — Minimax H3 Workspace")
    L.append("Создан: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    sec("ПРИЛОЖЕНИЕ")
    L.append(f"Версия: {APP_VERSION}")
    L.append(f"Собранная версия (exe): {'да' if getattr(sys, 'frozen', False) else 'нет, запуск из кода'}")
    L.append(f"Python: {sys.version.split()[0]}")
    L.append(f"Папка программы: {info.get('home')}")
    L.append(f"Папка данных: {info.get('data_dir')}")

    sec("КОМПЬЮТЕР")
    L.append(f"Windows: {windows_version()}")
    L.append(f"Процессор: {cpu_name()}")
    hw = info.get("hw") or {}
    L.append(f"Память (ОЗУ), ГБ: {hw.get('ram_gb')}")
    L.append(f"Свободно на диске установки, ГБ: {hw.get('disk_free_gb')}")
    L.append("Видеокарта NVIDIA (name, driver, memory, compute): " + nvidia_smi_info())
    L.append(f"Определено приложением: видеокарта={hw.get('gpu')}, видеопамять ГБ={hw.get('vram_gb')}, "
             f"драйвер={hw.get('driver')}, compute={hw.get('compute_cap')}")
    L.append(f"Движок окна (WebView2): {webview2_version()}")

    sec("ОКНО (что видит страница)")
    for k, v in (info.get("page") or {}).items():
        L.append(f"{k}: {v}")

    sec("ДВИЖОК ComfyUI")
    for k, v in (info.get("comfy") or {}).items():
        L.append(f"{k}: {v}")

    sec("МОДЕЛИ")
    for m in info.get("models") or []:
        L.append(f"[{'есть' if m['have'] else 'НЕТ '}] {m['title']}: {m['name']}")
    if not info.get("models"):
        L.append("нет данных (движок не отвечает)")

    sec("НАСТРОЙКИ (без ключей)")
    L.append(json.dumps(safe_config(info.get("config")), ensure_ascii=False, indent=1))

    sec("ПОСЛЕДНИЕ ОШИБКИ В ПРИЛОЖЕНИИ")
    for e in info.get("errors") or ["ошибок не было с момента запуска"]:
        L.append("- " + str(e))

    sec("ПОСЛЕДНИЕ СООБЩЕНИЯ НА ЭКРАНЕ")
    for e in info.get("toasts") or ["нет"]:
        L.append("- " + str(e))

    sec("ЖУРНАЛ ДВИЖКА (последние строки)")
    L.append(info.get("comfy_log") or "журнала нет (свой ComfyUI пишет его сам в своём окне)")

    sec("ЖУРНАЛ АССИСТЕНТА (последние строки)")
    L.append(info.get("llm_log") or "нет")

    return mask("\n".join(L)) + "\n"
