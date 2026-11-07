"""Minimax H3 Workspace desktop entry point: runs the local server and shows it in its own window.

Falls back to the default browser if the WebView2 window cannot be created.
"""

import asyncio
import os
import socket
import sys
import threading
import webbrowser
from pathlib import Path

if getattr(sys, "frozen", False):
    # installed app: settings live in the install folder the user chose (see datadir.py), never in the Windows profile
    import datadir
    os.environ.setdefault("H3STUDIO_DATA", str(datadir.locate(Path(sys.executable).parent, os.environ.get("LOCALAPPDATA", "."))))

from aiohttp import web  # noqa: E402

import server  # noqa: E402

_JOB_HANDLE = None


def init_job_object():
    """Bind the current process to a Windows Job Object with KILL_ON_JOB_CLOSE.
    This guarantees that whenever the main app process closes or exits, the Windows kernel
    automatically and unconditionally terminates all child processes (ComfyUI python, llama-server).
    """
    if sys.platform != "win32":
        return
    global _JOB_HANDLE
    try:
        import ctypes
        from ctypes import wintypes

        class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", wintypes.LARGE_INTEGER),
                ("PerJobUserTimeLimit", wintypes.LARGE_INTEGER),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class IO_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("ReadOperationCount", ctypes.c_ulonglong),
                ("WriteOperationCount", ctypes.c_ulonglong),
                ("OtherOperationCount", ctypes.c_ulonglong),
                ("ReadTransferCount", ctypes.c_ulonglong),
                ("WriteTransferCount", ctypes.c_ulonglong),
                ("OtherTransferCount", ctypes.c_ulonglong),
            ]

        class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
                ("IoCounters", IO_COUNTERS),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryLimit", ctypes.c_size_t),
                ("PeakJobMemoryLimit", ctypes.c_size_t),
            ]

        kernel32 = ctypes.windll.kernel32
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        kernel32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        kernel32.SetInformationJobObject.restype = wintypes.BOOL
        kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        kernel32.AssignProcessToJobObject.restype = wintypes.BOOL

        hJob = kernel32.CreateJobObjectW(None, None)
        if not hJob:
            return

        info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        info.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        res = kernel32.SetInformationJobObject(hJob, 9, ctypes.byref(info), ctypes.sizeof(info))
        if res:
            kernel32.AssignProcessToJobObject(hJob, kernel32.GetCurrentProcess())
            _JOB_HANDLE = hJob
        else:
            kernel32.CloseHandle(hJob)
    except Exception:
        pass


def cleanup_stale_environment():
    """Kill lingering background processes from previous crashed sessions
    and remove stale webview lock files.
    """
    if sys.platform != "win32":
        return
    import subprocess
    try:
        subprocess.run(["taskkill", "/F", "/IM", "llama-server.exe"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       creationflags=0x08000000, timeout=1)
    except Exception:
        pass

    try:
        wv_dir = server.DATA_DIR / "webview"
        if wv_dir.exists():
            for lock in wv_dir.rglob("LOCK"):
                try:
                    if lock.is_file() and lock.stat().st_size == 0:
                        lock.unlink(missing_ok=True)
                except Exception:
                    pass
    except Exception:
        pass


def resolve_webview_storage_path():
    """Return a usable webview user data folder path. If the primary folder is locked
    by an unreleased process, fall back to an alternate directory so WebView2 never hangs or shows a black window.
    """
    base_dir = server.DATA_DIR / "webview"
    lock_file = base_dir / "EBWebView" / "Default" / "LOCK"
    if lock_file.exists():
        try:
            with open(lock_file, "r+b"):
                pass
        except OSError:
            alt_dir = server.DATA_DIR / "webview_session"
            alt_dir.mkdir(parents=True, exist_ok=True)
            return str(alt_dir)
    return str(base_dir)


def free_port(start=8765):
    for port in range(start, start + 50):
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError("no free port")


class ServerThread(threading.Thread):
    def __init__(self, port):
        super().__init__(daemon=True)
        self.port = port
        self.ready = threading.Event()
        self.loop = None
        self.runner = None
        self.site = None

    def run(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.runner = web.AppRunner(server.build_app())
        self.loop.run_until_complete(self.runner.setup())
        self.site = web.TCPSite(self.runner, "127.0.0.1", self.port)
        self.loop.run_until_complete(self.site.start())
        self.ready.set()
        self.loop.run_forever()

    def shutdown(self):
        if not self.loop or not self.runner:
            return
        async def _stop():
            try:
                if self.site:
                    await self.site.stop()
            except Exception:
                pass
            try:
                await self.runner.cleanup()
            except Exception:
                pass
        try:
            fut = asyncio.run_coroutine_threadsafe(_stop(), self.loop)
            try:
                fut.result(timeout=1.5)
            except Exception:
                pass
        except Exception:
            pass
        finally:
            try:
                self.loop.call_soon_threadsafe(self.loop.stop)
            except Exception:
                pass


class JsApi:
    # pywebview exposes every public attribute to JS and walks it recursively,
    # so everything except the methods JS may call stays private (underscore)
    def __init__(self, port):
        self._window = None
        self._port = port

    def pick_folder(self, start=""):
        import webview
        res = self._window.create_file_dialog(webview.FOLDER_DIALOG, directory=start if os.path.isdir(start) else "")
        return res[0] if res else None

    def _copy(self, item_id, dest):
        """Stream a library video from the local server into the file the user chose."""
        import urllib.request
        with urllib.request.urlopen(f"http://127.0.0.1:{self._port}/api/file/{item_id}", timeout=60) as r, \
                open(dest, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)

    @staticmethod
    def _free_name(folder, name):
        stem, ext = os.path.splitext(name)
        path, n = os.path.join(folder, name), 1
        while os.path.exists(path):  # never overwrite files the user already has
            path, n = os.path.join(folder, f"{stem} ({n}){ext}"), n + 1
        return path

    def save_video(self, item_id, filename):
        """The usual Windows "Save as" window, then the video is copied there."""
        import webview
        ext = os.path.splitext(filename)[1].lower() or ".mp4"
        kinds = (f"Картинка {ext[1:].upper()} (*{ext})",) if ext != ".mp4" else ("Видео MP4 (*.mp4)",)
        res = self._window.create_file_dialog(webview.SAVE_DIALOG, save_filename=filename, file_types=kinds)
        if not res:
            return {"ok": False, "cancelled": True}
        dest = res if isinstance(res, str) else res[0]
        if not dest.lower().endswith(ext):
            dest += ext
        try:
            self._copy(item_id, dest)
        except Exception as e:
            return {"ok": False, "error": str(e)}
        return {"ok": True, "path": dest}

    def save_videos(self, items):
        """Several videos: one folder is chosen, every video is saved into it."""
        import webview
        res = self._window.create_file_dialog(webview.FOLDER_DIALOG)
        if not res:
            return {"ok": False, "cancelled": True}
        folder = res if isinstance(res, str) else res[0]
        saved, failed = [], 0
        for it in items:
            try:
                path = self._free_name(folder, it["filename"])
                self._copy(it["id"], path)
                saved.append(path)
            except Exception:
                failed += 1
        return {"ok": bool(saved), "folder": folder, "count": len(saved), "failed": failed}


def hard_exit(code=0):
    try:
        import ctypes
        from ctypes import wintypes
        k = ctypes.windll.kernel32
        k.GetCurrentProcess.restype = wintypes.HANDLE
        k.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
        k.TerminateProcess.restype = wintypes.BOOL
        k.TerminateProcess(k.GetCurrentProcess(), code)
    except Exception:
        pass
    os._exit(code)


def main():
    import ctypes
    if sys.platform == "win32":
        # no Windows error dialogs; child processes inherit this
        ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002)  # SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX
        init_job_object()
        cleanup_stale_environment()

    port = free_port()
    srv = ServerThread(port)
    srv.start()
    srv.ready.wait(30)
    url = f"http://127.0.0.1:{port}"

    def explain_no_webview():
        try:
            ctypes.windll.user32.MessageBoxW(
                0, "Не найден компонент Windows «Microsoft Edge WebView2», он рисует окно программы.\n\n"
                   "Скачайте и установите «WebView2 Runtime» (бесплатно, с сайта Microsoft: microsoft.com/edge/webview2), "
                   "затем запустите программу снова.\n\nПока программа откроется в обычном браузере.",
                "Minimax H3 Workspace", 0x40)
        except Exception:
            pass

    try:
        import webview
        if sys.platform == "win32":
            try:
                import report
                if report.webview2_version().startswith("не найден"):
                    explain_no_webview()
                    raise RuntimeError("no WebView2")
            except ImportError:
                pass
        try:
            webview.settings["ALLOW_DOWNLOADS"] = True  # a plain <a download> also works as a fallback
        except Exception:
            pass

        storage_path = resolve_webview_storage_path()
        api = JsApi(port)
        window = webview.create_window("Minimax H3 Workspace (Uncensored)", url, width=1440, height=920, min_size=(1000, 680),
                                       background_color="#12070a", js_api=api,
                                       draggable=True)  # pywebview blocks dragging of every picture by default; the app drags pictures into slots
        api._window = window

        def on_closed():
            # Watchdog: if anything hangs after window is closed, force-kill the process in 1.5 seconds
            threading.Timer(1.5, lambda: hard_exit(0)).start()
            try:
                srv.shutdown()
            except Exception:
                pass
            hard_exit(0)

        window.events.closed += on_closed
        webview.start(gui="edgechromium", private_mode=False,
                      storage_path=storage_path)
    except Exception:
        # no WebView2: use the browser and keep running until the console/process is closed
        webbrowser.open(url)
        threading.Event().wait()
    finally:
        try:
            srv.shutdown()
        except Exception:
            pass
        hard_exit(0)


if __name__ == "__main__":
    main()
