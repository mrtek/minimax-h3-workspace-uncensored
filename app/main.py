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

    def run(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.runner = web.AppRunner(server.build_app())
        self.loop.run_until_complete(self.runner.setup())
        self.loop.run_until_complete(web.TCPSite(self.runner, "127.0.0.1", self.port).start())
        self.ready.set()
        self.loop.run_forever()

    def shutdown(self):
        # cleanup also stops the managed ComfyUI process
        fut = asyncio.run_coroutine_threadsafe(self.runner.cleanup(), self.loop)
        try:
            fut.result(timeout=30)
        finally:
            self.loop.call_soon_threadsafe(self.loop.stop)


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


def main():
    import ctypes
    if sys.platform == "win32":
        # no Windows error dialogs (e.g. nvidia-smi.exe 0xc0000142 under memory pressure); child processes inherit this
        ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002)  # SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX
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
        api = JsApi(port)
        api._window = webview.create_window("Minimax H3 Workspace (Uncensored)", url, width=1440, height=920, min_size=(1000, 680),
                                           background_color="#12070a", js_api=api,
                                           draggable=True)  # pywebview blocks dragging of every picture by default; the app drags pictures into slots
        webview.start(gui="edgechromium", private_mode=False,
                      storage_path=str(server.DATA_DIR / "webview"))
    except Exception:
        # no WebView2: use the browser and keep running until the console/process is closed
        webbrowser.open(url)
        threading.Event().wait()
    finally:
        srv.shutdown()


if __name__ == "__main__":
    main()
