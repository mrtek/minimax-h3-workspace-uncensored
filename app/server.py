"""Minimax H3 Workspace local server.

Serves the UI and talks to ComfyUI over its HTTP/WebSocket API: uploads inputs,
queues workflows, relays progress and live previews to the browser, and keeps a
small library of finished videos. On first run it either connects to the user's
own ComfyUI or installs one (plus node packs and models) and manages it.

    python app/server.py [--port 8765] [--open]
"""

import argparse
import asyncio
import json
import os
import random
import re
import shutil
import struct
import threading
import time
import subprocess
import sys
import uuid
import urllib.parse
import webbrowser
from datetime import datetime
from pathlib import Path

import aiohttp
from aiohttp import web

import agent as ag
import catalog
import cloud
import customwf
import engine
import monitor
import report
import speech
import updater
from version import APP_VERSION

def split_frames(text):
    """One frame per line / numbered item: "1: ...", "Frame 2. ...", "Кадр 3 - ..."."""
    parts = re.split(r"(?m)^\s*(?:frame|кадр)?\s*\d+\s*[:.)\-–]\s*", text.strip(), flags=re.I)
    frames = [p.strip().replace("\n", " ") for p in parts if p.strip()]
    if len(frames) < 2:
        frames = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return frames


def _ssl_context():
    # Windows store + the bundled certifi list: helps PCs with an outdated root-certificate store
    import ssl
    ctx = ssl.create_default_context()
    try:
        import certifi
        ctx.load_verify_locations(cafile=certifi.where())
    except Exception:
        pass
    return ctx


class JobStore(dict):
    """prompt_id -> job; every change is written to disk so a closed program or a power cut does not lose the queue."""

    def __init__(self, on_change):
        super().__init__()
        self.on_change = on_change

    def __setitem__(self, k, v):
        super().__setitem__(k, v)
        self.on_change()

    def pop(self, k, *a):
        r = super().pop(k, *a)
        self.on_change()
        return r


def round_size(form, step=32, low=256, high=2048, max_px=2200000):
    """The user's own width/height from the form, rounded to the model's step (None when not given)."""
    w, h = str(form.get("width", "")), str(form.get("height", ""))
    if not (w.isdigit() and h.isdigit()):
        return None
    w, h = int(w), int(h)
    if min(w, h) < 64:
        return None
    k = (max_px / (w * h)) ** 0.5 if w * h > max_px else 1.0
    r = lambda v: min(high, max(low, round(v * k / step) * step))
    return r(w), r(h)


SUPPORT_URL = "https://pay.cloudtips.ru/p/0e4d4824"
SUPPORT_UNCENSORED_URL = "https://pay.cloudtips.ru/p/10956f76"
import workflows as wf

# frozen builds keep code inside the bundle but user data next to the exe
ROOT = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
HOME = Path(sys.executable).parent if getattr(sys, "frozen", False) else ROOT
UI_DIR = ROOT / "ui"
DATA_DIR = Path(os.environ.get("H3STUDIO_DATA", HOME / "data"))
LONG_DIR = DATA_DIR / "long"   # saved scene states of the long video made scene by scene
LIBRARY_FILE = DATA_DIR / "library.json"
QUEUE_FILE = DATA_DIR / "queue.json"
WORKFLOWS_DIR = DATA_DIR / "workflows"
MAX_SECONDS = 25.0  # H3 itself is made for up to 15 s; longer clips are allowed, quality is the user's call

CLIENT_ID = "h3studio-" + uuid.uuid4().hex[:8]

# ComfyUI binary websocket event types
BIN_PREVIEW = 1
BIN_PREVIEW_META = 4


class Library:
    def __init__(self, path):
        self.path = path
        self.items = []
        if path.exists():
            self.items = json.loads(path.read_text(encoding="utf-8"))

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.items, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(self.path)

    def get(self, item_id):
        return next((x for x in self.items if x["id"] == item_id), None)

    def add(self, item):
        self.items.insert(0, item)
        self.save()

    def remove(self, item_id):
        self.items = [x for x in self.items if x["id"] != item_id]
        self.save()


class App:
    def __init__(self, comfy_override=None):
        self.config = engine.Config(DATA_DIR / "config.json")
        self.comfy_override = comfy_override.rstrip("/") if comfy_override else None
        self.session = None
        self.browsers = set()
        self.saved_queue = self.load_queue()  # unfinished jobs of the previous run (asked about at start)
        self.jobs = JobStore(self.save_queue)  # prompt_id -> job dict
        self.library = Library(LIBRARY_FILE)
        self.output_dir = self._detect_output_dir()
        self.comfy_online = False
        self.process = None  # engine.ComfyProcess in managed mode
        self._bg = set()  # background tasks (kept so they are not garbage collected)
        self._stop_awake = threading.Event()
        self._comfy_update_cache = None
        self._comfy_updating = False
        self._comfy_update_status = {"status": "idle", "log": ""}
        self.installer = None
        self.setup_log = []  # last setup events, replayed to a reopened window
        self.model_names = {}  # role -> name as ComfyUI lists it
        self.missing = []  # catalog entries ComfyUI does not see
        self.skills = ag.Skills(ROOT / "skills", DATA_DIR / "skills", self.config)
        self.llm = ag.LlamaServer(self.llm_root(), DATA_DIR / "llama.log")
        self.agent_installer = None
        self.hw = None
        self.rh_models = None  # model paths found in the user's RunningHub workflow
        self.monitor = monitor.Monitor()
        self.recent_errors = []  # last problems, for the report to the author

    def llm_root(self):
        if self.config.get("llm_dir"):
            return Path(self.config.get("llm_dir"))
        if self.config.get("engine") == "managed" and self.config.get("install_dir"):
            return Path(self.config.get("install_dir")) / "llm"
        return DATA_DIR.parent / "llm"

    async def llm_variant(self):
        if not self.llm.alive():
            self.llm.root = self.llm_root()  # follows the install folder chosen in setup
            self.llm.custom = self.config.get("llm_custom")
        if self.config.get("llm_variant") in ag.LLM_VARIANTS:
            return self.config.get("llm_variant")
        if self.hw is None:
            self.hw = await asyncio.to_thread(engine.detect_hardware)
        return ag.pick_variant(self.hw)

    async def stop_llm(self):
        if self.llm.alive():
            await asyncio.to_thread(self.llm.stop)

    @property
    def comfy(self):
        if self.comfy_override:
            return self.comfy_override
        if self.process:
            return self.process.url
        return self.config.get("comfy_url", "http://127.0.0.1:8188").rstrip("/")

    def comfy_root(self):
        d = self.config.get("install_dir")
        return Path(d) / catalog.COMFY_DIRNAME if d else None

    def choice(self):
        return {**catalog.DEFAULT_CHOICE, **(self.config.get("choice") or {}), **(self.config.get("choice_override") or {})}

    def modes(self):
        modes = list(self.config.get("modes") or catalog.MODE_ROLES)
        if "actor" not in modes and ("refs" in modes or "lipsync" in modes):
            modes.append("actor")  # same models, so set-ups made before this mode existed get it too
        if "swap" not in modes and "refs" in modes:
            modes.append("swap")   # character-swap preset: References models + a 0.16 GB LoRA, fetched by "Докачать"
        return modes + [m for m in getattr(self, "auto_modes", ()) if m not in modes]

    # ---------- ComfyUI connection ----------

    def apply_mirror(self):
        engine.HF_MIRROR = "hf-mirror.com" if self.config.get("hf_mirror") else ""

    async def h_mirror(self, req):
        on = bool((await req.json()).get("on"))
        self.config.update(hf_mirror=on)
        self.apply_mirror()
        return web.json_response({"ok": True, "on": on})

    def _bg_run(self, coro):
        task = asyncio.create_task(coro)
        self._bg.add(task)
        task.add_done_callback(self._bg.discard)
        return task

    def awake_loop(self):
        """While generations are queued or running (and for ten minutes after the last one), Windows is not allowed to go to sleep:
        a long video made scene after scene overnight must not be stopped by the power plan. The screen may still switch off."""
        import ctypes
        import time
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        last_busy = 0.0
        while not self._stop_awake.is_set():
            if self.jobs:
                last_busy = time.time()
            keep = time.time() - last_busy < 600
            try:
                ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if keep else 0))
            except Exception:
                return
            for _ in range(15):
                if self._stop_awake.is_set():
                    break
                time.sleep(1)

    async def start(self, _app):
        if sys.platform == "win32":
            import threading
            threading.Thread(target=self.awake_loop, daemon=True, name="keep-awake").start()
        self.apply_mirror()
        self.session = aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=_ssl_context()))
        if self.config.get("engine") == "managed" and self.config.get("setup_done") and not self.comfy_override:
            if self.missing_node_packs():
                self.packs_task = asyncio.create_task(self.install_missing_packs())
            else:
                self.start_managed()
        self.ws_task = asyncio.create_task(self.comfy_ws_loop())

    def missing_node_packs(self):
        """Own ComfyUI: node packs from the catalog that are not in its custom_nodes yet (a newer app version may need more)."""
        root = self.comfy_root()
        if not root:
            return []
        nodes_dir = root / "ComfyUI" / "custom_nodes"
        if not nodes_dir.is_dir():
            return []
        return [p["name"] for p in catalog.NODE_PACKS if not (nodes_dir / p["name"]).exists()]

    async def install_missing_packs(self):
        """Put the missing node packs in place, then start ComfyUI. Runs in the background so the app opens at once;
        ComfyUI is started even when the download fails (no internet), only the new sections stay unavailable."""
        async def quiet(_ev):
            pass
        try:
            await engine.Installer(ROOT, self.session, quiet).install_nodes(self.comfy_root())
        except Exception as e:
            print(f"Не удалось докачать ноды ({type(e).__name__}): {e}", file=sys.stderr)
        if not self.process:
            self.start_managed()

    async def stop(self, _app):
        self._stop_awake.set()
        for task in list(self._bg):
            task.cancel()
        self.ws_task.cancel()
        if self.installer:
            self.installer.cancelled = True
        if self.process:
            await asyncio.to_thread(self.process.stop)
        await asyncio.to_thread(self.llm.stop)
        await self.session.close()

    def start_managed(self):
        root = self.comfy_root()
        self.process = engine.ComfyProcess(root, DATA_DIR / "comfyui.log")
        self.process.start()
        self.output_dir = root / "ComfyUI" / "output"

    async def wait_comfy(self, timeout=300):
        """Wait until ComfyUI answers; raise with its log if the process died."""
        loop = asyncio.get_running_loop()
        end = loop.time() + timeout
        while loop.time() < end:
            if self.process and not self.process.alive():
                tail = self.process.log_tail()
                hint = ("\n\nПохоже, проблема в драйвере видеокарты. Обновите драйвер NVIDIA (nvidia.com/drivers), "
                        "перезагрузите компьютер и нажмите «Продолжить».") if re.search(r"cuda|nvidia|driver", tail, re.I) else ""
                raise RuntimeError("ComfyUI не запустился:\n" + tail + hint)
            try:
                await self.system_stats()
                return
            except Exception:
                await asyncio.sleep(2)
        raise RuntimeError("ComfyUI долго не отвечает:\n" + (self.process.log_tail() if self.process else ""))

    async def resolve_models(self):
        """Match the chosen model files against what ComfyUI sees (any subfolder)."""
        visible = {}
        for cat in {v["category"] for variants in catalog.VARIANTS.values() for v in variants.values()}:
            try:
                _, lst = await self.comfy_json("GET", f"/models/{cat}")
            except Exception:
                lst = []
            visible[cat] = lst or []
        self.visible = {cat: {n.replace("/", "\\").split("\\")[-1] for n in lst} for cat, lst in visible.items()}
        names, found, entries = {}, set(), {}
        for role, variant in self.choice().items():
            entry = entries[role] = catalog.model_entry(role, variant)
            picked = (self.config.get("model_pick") or {}).get(role)  # a file the user chose by hand
            def seen(file):
                ok = {file, *catalog.FILE_ALIASES.get(file, ())}  # same file under another name (e.g. upscaler from Hugging Face)
                return next((n for n in visible[entry["category"]] if n.replace("/", "\\").split("\\")[-1] in ok), None)
            others = {v["file"] for r, vs in catalog.VARIANTS.items() if r != role for v in vs.values()}
            if picked and picked.replace("/", "\\").split("\\")[-1] in others:  # the file of ANOTHER model of the app: never feed it to this role
                picked = None
            hit = picked if picked in visible[entry["category"]] else seen(entry["file"])
            if not hit and role not in (self.config.get("choice_override") or {}):  # another variant the user already has (v1.3 instead of v2); never for a version picked by hand
                hit = next(filter(None, (seen(v["file"]) for v in catalog.VARIANTS[role].values())), None)
            names[role] = hit or entry["name"]
            if hit:
                found.add(role)
        # modes added after a set-up was made: with your own ComfyUI they switch on by themselves when the files are there
        self.auto_modes = {m for m in ("krea", "kreareal", "qwen", "flux", "upscale", "long") if self.config.get("engine") == "external"
                           and all(r in found for r in catalog.MODE_ROLES[m])}
        needed = catalog.roles_for(self.modes())
        missing = [entries[r] for r in needed if r in entries and r not in found]
        self.model_names, self.missing = names, missing
        return names, missing

    async def comfy_json(self, method, path, **kw):
        async with self.session.request(method, self.comfy + path, **kw) as r:
            body = await r.text()
            data = json.loads(body) if body else None
            return r.status, data

    async def system_stats(self):
        try:
            status, data = await self.comfy_json("GET", "/system_stats", timeout=aiohttp.ClientTimeout(total=8))
            if status != 200 or not isinstance(data, dict) or "system" not in data:
                raise ValueError("no stats")
        except Exception:
            # some builds hide or slow down /system_stats; a queue answer still means ComfyUI is alive
            status, _ = await self.comfy_json("GET", "/queue", timeout=aiohttp.ClientTimeout(total=8))
            if status != 200:
                raise
            data = {"system": {}, "devices": []}
        argv = data["system"].get("argv", [])
        if "--output-directory" in argv:
            self.output_dir = Path(argv[argv.index("--output-directory") + 1])
        return data

    async def comfy_ws_loop(self):
        while True:
            if not self.config.get("setup_done") and not self.comfy_override:
                await asyncio.sleep(2)
                continue
            ws_url = self.comfy.replace("http", "ws", 1) + f"/ws?clientId={CLIENT_ID}"
            try:
                async with self.session.ws_connect(ws_url, max_msg_size=0, heartbeat=20) as ws:
                    self.comfy_online = True
                    await self.resolve_models()
                    await self.broadcast({"type": "engine", "online": True})
                    await self.warn_weak_gpu()
                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            await self.on_comfy_event(json.loads(msg.data))
                        elif msg.type == aiohttp.WSMsgType.BINARY:
                            await self.on_comfy_binary(msg.data)
            except asyncio.CancelledError:
                raise
            except Exception:
                pass
            if self.comfy_online:
                self.comfy_online = False
                await self.broadcast({"type": "engine", "online": False})
            await asyncio.sleep(2)

    async def warn_weak_gpu(self):
        """The 'max' profile (full ~32 GB Singularity) is meant for 24 GB cards. Say so once per run on a weaker card."""
        if getattr(self, "_weak_warned", False) or self.config.get("choice", {}).get("lipsync_unet") != "full":
            return
        try:
            if self.hw is None:
                self.hw = await asyncio.to_thread(engine.detect_hardware)
            vram = self.hw.get("vram_gb") or 0
        except Exception:
            return
        if 0 < vram < 24:
            self._weak_warned = True
            text = (f"Выбрана самая тяжёлая модель (профиль «Максимум»), а у видеокарты {vram:g} ГБ памяти: "
                    "генерация может зависать на первом шаге. Откройте «Модели», выберите для Lipsync/Референсов "
                    "облегчённую версию (pruned) и скачайте её.")
            self.note_error(text)
            await self.broadcast({"type": "notice", "text": text})

    async def on_comfy_binary(self, data):
        if len(data) < 8:
            return
        kind = struct.unpack(">I", data[:4])[0]
        if kind == BIN_PREVIEW:
            image = data[8:]
        elif kind == BIN_PREVIEW_META:
            meta_len = struct.unpack(">I", data[4:8])[0]
            meta = json.loads(data[8:8 + meta_len])
            if meta.get("prompt_id") not in self.jobs:
                return
            image = data[8 + meta_len:]
        else:
            return
        if not any(j["state"] == "running" for j in self.jobs.values()):
            return
        for b in list(self.browsers):
            try:
                await b.send_bytes(image)
            except Exception:
                self.browsers.discard(b)

    async def on_comfy_event(self, msg):
        t, d = msg.get("type"), msg.get("data") or {}
        if t == "kj_preview_override":   # animated live preview of KJNodes (no prompt id in it: it belongs to the running job)
            run = next((j for j in self.jobs.values() if j["state"] == "running"), None)
            if run and d.get("image"):
                await self.broadcast({"type": "preview_anim", "prompt_id": run["prompt_id"], "image": d["image"],
                                      "mime": d.get("mime") or "image/jpeg", "step": d.get("step"), "total": d.get("total")})
            return
        job = self.jobs.get(d.get("prompt_id"))
        if not job:
            return
        pid = job["prompt_id"]
        if t == "execution_start":
            job["state"] = "running"
            await self.send_stage(job, "load")
        elif t == "executing" and d.get("node"):
            stage = job["stages"].get(d["node"])
            if stage:
                await self.send_stage(job, stage, node=d["node"])
        elif t == "progress":
            stage = job["stages"].get(str(d.get("node")), job.get("stage"))
            if stage in ("sample", "sample2"):
                await self.send_stage(job, stage, d.get("value"), d.get("max"), node=str(d.get("node")))
        elif t == "execution_success":
            await self.finish_job(job)
        elif t == "execution_error":
            job["state"] = "error"
            self.note_error(f'Ошибка генерации, узел {d.get("node_type", "")}: {d.get("exception_message", "").strip()}')
            hint = self.model_hint((job.get("meta") or {}).get("mode", ""), d.get("exception_message", ""))
            await self.broadcast({"type": "error", "prompt_id": pid,
                                  "message": f'{d.get("node_type", "")}: {d.get("exception_message", "").strip()}' + hint})
            self.jobs.pop(pid, None)
        elif t == "execution_interrupted":
            await self.broadcast({"type": "interrupted", "prompt_id": pid})
            self.jobs.pop(pid, None)

    async def send_stage(self, job, stage, step=None, total=None, node=None):
        job["stage"] = stage
        msg = {"type": "stage", "prompt_id": job["prompt_id"], "stage": stage, "step": step, "max": total}
        sc = job.get("scenes")
        if sc and node is not None and str(node) in sc.get("map", {}):
            job["scene_now"] = sc["map"][str(node)] + 1
        if sc and job.get("scene_now"):
            msg.update(scene=job["scene_now"], scenes=sc["n"])
        await self.broadcast(msg)

    async def finish_job(self, job):
        pid = job["prompt_id"]
        self.jobs.pop(pid, None)
        _, hist = await self.comfy_json("GET", f"/history/{pid}")
        outputs = (hist or {}).get(pid, {}).get("outputs", {})
        tagged = [(nid, f) for nid, out in outputs.items() for lst in out.values() if isinstance(lst, list)
                  for f in lst if isinstance(f, dict) and "filename" in f]
        files = [f for _, f in tagged]
        node_of = {id(f): nid for nid, f in tagged}
        videos = [f for f in files if f["filename"].lower().endswith((".mp4", ".webm", ".mkv", ".mov"))]
        videos = [f for f in videos if f.get("type", "output") == "output"] or videos  # not the previews of loaded clips
        lg = (job.get("meta") or {}).get("long")
        if lg:  # a scene of the long video: keep the tail of its latent for the next scene
            lat = next((f for f in files if f["filename"].lower().endswith(".latent")), None)
            if lat and videos:
                try:
                    async with self.session.get(self.comfy + "/view", params={
                            "filename": lat["filename"], "subfolder": lat.get("subfolder", ""), "type": lat.get("type", "output")}) as r:
                        body = await r.read()
                    if r.status == 200 and body:
                        d = LONG_DIR / lg["run"]
                        d.mkdir(parents=True, exist_ok=True)
                        (d / f"state_{lg['part']}.latent").write_bytes(body)
                        (d / f"state_{lg['part']}.json").write_text(
                            json.dumps({"res": (job.get("meta") or {}).get("res", "")}, ensure_ascii=False), encoding="utf-8")
                except Exception as e:
                    self.note_error(f"Не удалось сохранить состояние сцены: {e}")
        images = [f for f in files if f["filename"].lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
                  and f.get("type", "output") == "output"]
        if not videos and not images:
            self.note_error("ComfyUI закончил работу, но не вернул ни видео, ни картинку")
            await self.broadcast({"type": "error", "prompt_id": pid, "message": "ComfyUI не вернул ни видео, ни картинку"})
            return
        if videos:
            picked = [videos[-1] if job.get("last_output") else videos[0]]
        else:  # a workflow that makes pictures: every saved picture goes to the library
            picked = images[:12]
            def order(f):  # the glued sheet first, then the frames 1, 2, 3...
                t = str((job.get("titles") or {}).get(node_of.get(id(f)), ""))
                return (0 if t.startswith("Лист") else 1, t)
            picked.sort(key=order)
        items = []
        for f in picked:
            item = dict(job["meta"], id=uuid.uuid4().hex[:12], prompt_id=pid,
                        date=datetime.now().isoformat(timespec="seconds"), fav=False, file=f,
                        ext=Path(f["filename"]).suffix.lstrip(".").lower() or "mp4")
            if not videos:
                item["kind"] = "image"
                title = (job.get("titles") or {}).get(node_of.get(id(f)), "")  # "Результат: апскейл (после…)" -> "апскейл"
                label = re.split(r"\s*[(\[]", title.split(":")[-1].strip())[0].strip()[:24]
                if label:
                    item["label"] = label
            meta = f.get("metadata") or {}
            if meta.get("width") and meta.get("height") and not (item.get("mode") == "upscale" and item.get("res")):
                item["res"] = f'{meta["width"]}×{meta["height"]}'
            self.library.add(item)
            await self.archive_item(item)
            items.append(self.public(item))
        await self.broadcast({"type": "done", "prompt_id": pid, "item": items[0], "items": items})

    async def broadcast(self, payload):
        text = json.dumps(payload, ensure_ascii=False)
        for b in list(self.browsers):
            try:
                await b.send_str(text)
            except Exception:
                self.browsers.discard(b)

    # ---------- helpers ----------

    def public(self, item):
        p = self.file_path(item)
        # a video deleted from disk stays in the list until the user removes it, but is marked
        return {k: v for k, v in item.items() if k != "file"} | {"url": f"/api/file/{item['id']}",
                                                                 "missing": bool(p) and not p.exists()}

    def file_path(self, item):
        if item["file"].get("type") == "local":
            return Path(item["file"]["path"])
        orig = None
        if self.output_dir and item["file"].get("type") == "output":
            orig = self.output_dir / item["file"].get("subfolder", "") / item["file"]["filename"]
        if orig and orig.exists():
            return orig
        local = item.get("local")   # the program's own copy: it stays when ComfyUI's output folder is cleaned
        if local and Path(local).exists():
            return Path(local)
        out = self.output_dir or self._detect_output_dir()
        if out and out.exists():
            fn = item["file"]["filename"]
            sf = item["file"].get("subfolder", "")
            for cand in [out / sf / fn, out / fn]:
                if cand.exists():
                    return cand
        return orig

    # ---------- own copies of the results (they must not vanish when ComfyUI's output folder is cleaned) ----------

    def default_results_dir(self):
        """Videos\\Minimax H3\\output in the user's folder: it does not depend on where the program lies and survives updates."""
        return Path.home() / "Videos" / "Minimax H3" / "output"

    def results_dir(self):
        d = self.config.get("results_dir")
        return Path(d) if d else self.default_results_dir()

    def _detect_output_dir(self):
        cr = self.comfy_root()
        if cr and (cr / "ComfyUI" / "output").exists():
            return cr / "ComfyUI" / "output"
        return None


    def keep_copies(self):
        return self.config.get("keep_copies") is not False

    async def archive_item(self, item):
        """Copy the finished file of a library item into the program's own results folder."""
        if not self.keep_copies() or item.get("local") and Path(item["local"]).exists():
            return False
        f = item["file"]
        if f.get("type") == "local":
            return False
        try:
            dest_dir = self.results_dir()
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / f"{str(item.get('date', ''))[:10].replace('-', '')}_{item['id']}_{f['filename']}"
            src = self.output_dir / f.get("subfolder", "") / f["filename"] if self.output_dir and f.get("type") == "output" else None
            if src and src.exists():
                await asyncio.to_thread(shutil.copy2, src, dest)
            else:   # output folder unknown (own ComfyUI): take the file from ComfyUI itself
                params = {"filename": f["filename"], "subfolder": f.get("subfolder", ""), "type": f.get("type", "output")}
                async with self.session.get(self.comfy + "/view", params=params) as r:
                    if r.status != 200:
                        return False
                    with open(dest, "wb") as out:
                        async for chunk in r.content.iter_chunked(1 << 20):
                            out.write(chunk)
            item["local"] = str(dest)
            self.library.save()
            return True
        except Exception as e:
            self.note_error(f"Не удалось сохранить копию результата: {e}")
            return False

    async def upload_state(self, path, name):
        """Put a saved scene state (.latent) into ComfyUI's input folder, where its LoadLatent node looks for it."""
        data = aiohttp.FormData()
        data.add_field("image", path.read_bytes(), filename=name, content_type="application/octet-stream")
        data.add_field("type", "input")
        data.add_field("overwrite", "true")
        status, res = await self.comfy_json("POST", "/upload/image", data=data)
        if status != 200:
            raise web.HTTPBadGateway(text="Не удалось передать состояние сцены в ComfyUI")
        return res["name"]

    async def upload_input(self, field, rh=None):
        """Put an image/audio/video file into ComfyUI's input/h3studio folder (or upload it to RunningHub)."""
        data = aiohttp.FormData()
        name = f"h3s_{uuid.uuid4().hex[:8]}{Path(field.filename or 'img.png').suffix or '.png'}"
        content_type = field.headers.get("Content-Type", "application/octet-stream")
        raw = await field.read()
        if getattr(self, "_capture", None) is not None:
            self._capture[field.name] = (Path(field.filename or "x.png").suffix.lower() or ".png", raw)
        if rh:
            try:
                return await rh.upload(name, raw, content_type)
            except cloud.CloudError as e:
                raise web.HTTPBadGateway(text=str(e))
        data.add_field("image", raw, filename=name, content_type=content_type)
        data.add_field("subfolder", "h3studio")
        data.add_field("type", "input")
        status, res = await self.comfy_json("POST", "/upload/image", data=data)
        if status != 200:
            raise web.HTTPBadGateway(text="Не удалось загрузить файл в ComfyUI")
        return f'{res["subfolder"]}/{res["name"]}' if res.get("subfolder") else res["name"]

    # ---------- HTTP handlers ----------

    async def h_status(self, _req):
        base = {"configured": bool(self.config.configured or self.comfy_override),
                "engine": self.config.get("engine"), "installing": bool(self.installer)}
        try:
            stats = await self.system_stats()
        except Exception:
            return web.json_response({**base, "online": False})
        dev = (stats.get("devices") or [{}])[0]
        await self.resolve_models()
        missing = {m["role"] for m in self.missing}
        needed = catalog.roles_for(self.modes())
        models = [{"role": role, "title": catalog.ROLE_TITLES[role], "name": self.model_names[role],
                   "category": catalog.VARIANTS[role][variant]["category"],
                   "folder": catalog.VARIANTS[role][variant]["category"], "marks": catalog.marks_text(role),
                   "file": catalog.VARIANTS[role][variant]["file"], "url": engine.mirrored(catalog.VARIANTS[role][variant]["url"]),
                   "size": catalog.VARIANTS[role][variant]["size"], "have": role not in missing,
                   "needed": role in needed, "picked": role in (self.config.get("model_pick") or {}),
                   "modes": [m for m, roles in catalog.MODE_ROLES.items() if role in roles],
                   "variants": [{"key": k, "label": catalog.VARIANT_LABELS.get(role, {}).get(k, k), "size": v["size"],
                                 "current": k == variant, "have": v["file"] in getattr(self, "visible", {}).get(v["category"], ())}
                                for k, v in catalog.VARIANTS[role].items()] if len(catalog.VARIANTS[role]) > 1 else []}
                  for role, variant in self.choice().items() if role in needed]
        choice = self.choice()
        def _mode_gb(m):
            diff = (catalog.plan_size(choice, self.modes() + [m]) - catalog.plan_size(choice, self.modes())) / 1e9
            gb = round(diff, 1)
            return int(gb) if gb.is_integer() else gb

        add_modes = {m: _mode_gb(m) for m in catalog.MODE_ROLES if m not in self.modes()}
        return web.json_response({
            **base, "online": True, "modes": self.modes(), "add_modes": add_modes, "hf_mirror": bool(self.config.get("hf_mirror")),
            "offer_modes": [{"mode": m, "gb": add_modes[m]} for m in catalog.OFFER_MODES
                            if m in add_modes and m not in (self.config.get("offered_modes") or [])],
            "gpu": gpu_name(dev.get("name", "")),
            "vram_total": dev.get("vram_total"), "vram_free": dev.get("vram_free"),
            "comfy_version": stats["system"].get("comfyui_version"),
            "output_dir": str(self.output_dir) if self.output_dir else None,
            "paths": {"install": self.config.get("install_dir"), "models": self.config.get("models_dir"),
                      "ext_models": self.config.get("ext_models_dir"),
                      "assistant": str(self.llm.root), "data": str(DATA_DIR)},
            "models": models,
        })

    async def nodes_status(self):
        """Which custom nodes the chosen sections need, and whether this ComfyUI knows them."""
        modes = set(self.modes())
        out = []
        for c in catalog.NODE_CHECKS:
            used = [m for m in c["modes"] if m in modes]
            if not used:
                continue
            have = True
            for cls in c["classes"]:
                try:
                    status, data = await self.comfy_json("GET", f"/object_info/{cls}", timeout=aiohttp.ClientTimeout(total=8))
                    have = have and status == 200 and isinstance(data, dict) and cls in data
                except Exception:
                    have = False
            repo = catalog.node_repo(c["pack"])
            out.append({"pack": c["pack"], "have": have, "optional": c["optional"], "why": c["why"], "modes": used,
                        "url": f"https://github.com/{repo}"})
        return out

    async def h_nodes(self, _req):
        try:
            return web.json_response({"nodes": await self.nodes_status()})
        except aiohttp.ClientError:
            return web.json_response({"error": "ComfyUI не отвечает"}, status=503)

    async def h_models_files(self, req):
        """Files ComfyUI sees in the folder this model role lives in, for picking one by hand."""
        role = req.query.get("role", "")
        if role not in catalog.VARIANTS:
            raise web.HTTPBadRequest(text="Неизвестная модель")
        cat = next(iter(catalog.VARIANTS[role].values()))["category"]
        try:
            _, lst = await self.comfy_json("GET", f"/models/{cat}")
        except Exception:
            raise web.HTTPServiceUnavailable(text="ComfyUI не отвечает")
        return web.json_response({"files": lst or [], "folder": cat})

    async def h_models_download(self, req):
        """Own ComfyUI: download missing model files of the app's catalog into the user's models folder (resumable)."""
        body = await req.json()
        managed = self.config.get("engine") == "managed"
        if managed:  # the app's own ComfyUI: the folder is known
            folder = str(Path(self.config.get("install_dir") or "") / catalog.COMFY_DIRNAME / "ComfyUI" / "models")
        else:
            folder = (body.get("folder") or self.config.get("ext_models_dir") or "").strip()
        p = Path(folder) if folder else None
        if not p or not p.is_dir():
            raise web.HTTPBadRequest(text="Папка не найдена. Укажите папку models вашего ComfyUI.")
        if not any((p / sub).is_dir() for sub in ("diffusion_models", "loras", "vae", "text_encoders", "checkpoints")):
            raise web.HTTPBadRequest(text="Это не похоже на папку models ComfyUI: внутри нет diffusion_models, loras, vae. Выберите папку, где они лежат.")
        if getattr(self, "dl_task", None) and not self.dl_task.done():
            raise web.HTTPConflict(text="Скачивание уже идёт. Дождитесь окончания или остановите его.")
        choice = self.choice()
        roles = [r for r in (body.get("roles") or [body.get("role")]) if r in catalog.VARIANTS and r in choice]
        if not roles:
            raise web.HTTPBadRequest(text="Неизвестная модель")
        if not managed:
            self.config.update(ext_models_dir=str(p))
        entries = [catalog.model_entry(r, choice[r]) for r in roles]

        async def emit(ev):
            await self.broadcast({"type": "mdl", "state": ev.get("state"), "label": ev.get("label"),
                                  "done": ev.get("overall_done", ev.get("done")), "total": ev.get("overall_total", ev.get("total")),
                                  "index": ev.get("index"), "count": ev.get("count")})

        inst = engine.Installer(ROOT, self.session, emit)
        self.dl_installer = inst

        async def run():
            total = sum(e["size"] for e in entries)
            base = 0
            try:
                for i, e in enumerate(entries):
                    dest = p / e["category"] / Path(e["name"].replace("\\", "/"))
                    label = catalog.ROLE_TITLES.get(e["role"], e["role"])

                    async def emit_overall(ev, base=base, idx=i, label=label):
                        if ev.get("done") is not None:
                            ev = dict(ev, overall_done=base + ev["done"], overall_total=total, index=idx + 1, count=len(entries), label=label)
                        await emit(ev)
                    inst.emit = emit_overall
                    await inst.download(e["url"], dest, "models", label, e["size"])
                    base += e["size"]
                await self.resolve_models()
                await self.broadcast({"type": "mdl", "state": "done", "label": "Готово: файлы скачаны и лежат в вашей папке моделей"})
            except engine.Cancelled:
                await self.broadcast({"type": "mdl", "state": "error", "label": "Скачивание остановлено. Скачанное сохранено: нажмите «Скачать» снова, и оно продолжится."})
            except Exception as ex:
                await self.broadcast({"type": "mdl", "state": "error", "label": str(ex)[:300]})
        self.dl_task = asyncio.create_task(run())
        return web.json_response({"ok": True, "folder": str(p), "count": len(entries), "size": sum(e["size"] for e in entries)})

    async def h_models_download_cancel(self, _req):
        inst = getattr(self, "dl_installer", None)
        if inst:
            inst.cancelled = True
        return web.json_response({"ok": True})

    async def h_models_open(self, req):
        """Open the download page of a model in the browser (only links of the app's own catalog)."""
        body = await req.json()
        role = body.get("role")
        if role not in catalog.VARIANTS:
            raise web.HTTPBadRequest(text="Неизвестная модель")
        url = catalog.VARIANTS[role][self.choice().get(role) or next(iter(catalog.VARIANTS[role]))]["url"]
        url = engine.mirrored(url)
        webbrowser.open(url)
        return web.json_response({"ok": True, "url": url})

    async def h_models_pick(self, req):
        body = await req.json()
        role, name = body.get("role"), body.get("name") or ""
        if role not in catalog.VARIANTS:
            raise web.HTTPBadRequest(text="Неизвестная модель")
        if name and not body.get("force") and not catalog.looks_right(role, name):
            title = catalog.ROLE_TITLES.get(role, role)
            return web.json_response({"ok": False, "warn": (
                f"Название файла «{name.replace('/', chr(92)).split(chr(92))[-1]}» не похоже на нужное для «{title}». "
                f"В названии обычно есть {catalog.marks_text(role)}. Если выбрать не тот файл, генерация сломается с "
                f"непонятной ошибкой. Всё равно выбрать этот файл?")})
        picks = dict(self.config.get("model_pick") or {})
        if name:
            picks[role] = name
        else:
            picks.pop(role, None)
        self.config.update(model_pick=picks)
        await self.resolve_models()
        return web.json_response({"ok": True})

    def model_hint(self, mode, text):
        """After a ComfyUI error that smells like a wrong model file: say which models of this mode to check."""
        if not re.search(r"shape|size|conv|dtype|mat1|mat2|cannot be|expected|mismatch|tensor|dimension|state_dict|missing key|unexpected key|invalid", text or "", re.I):
            return ""
        roles = catalog.MODE_ROLES.get(mode, [])
        if not roles or not self.model_names:
            return ""
        picked = self.config.get("model_pick") or {}
        sus = [r for r in roles if r in self.model_names and (r in picked or not catalog.looks_right(r, self.model_names[r]))]
        bad = [r for r in sus if not catalog.looks_right(r, self.model_names[r])]
        lines = ["", "Подсказка: похоже, одна из моделей этого раздела не подходит (ошибка про размеры или формат данных)."]
        for r in bad:
            lines.append(f"• «{catalog.ROLE_TITLES.get(r, r)}»: сейчас стоит {self.model_names[r].replace('/', chr(92)).split(chr(92))[-1]}, "
                         f"а в названии нужного файла есть {catalog.marks_text(r)}.")
        for r in sus:
            if r not in bad:
                lines.append(f"• «{catalog.ROLE_TITLES.get(r, r)}»: вы выбирали файл вручную, это "
                             f"{self.model_names[r].replace('/', chr(92)).split(chr(92))[-1]}. Проверьте его.")
        lines.append("Откройте «Модели» и для нужной строки нажмите «Заменить…», выберите файл «как по умолчанию» или тот, "
                     "что указан в подсказке строки. Если непонятно, нажмите «Отчёт для разработчика» и пришлите файл.")
        return "\n".join(lines)

    async def h_models_variant(self, req):
        """Use another version of the same model (Singularity v1.3 -> v2, a lighter one...). The file is fetched by "Докачать"."""
        body = await req.json()
        role, variant = body.get("role"), body.get("variant") or ""
        if role not in catalog.VARIANTS or (variant and variant not in catalog.VARIANTS[role]):
            raise web.HTTPBadRequest(text="Неизвестная версия модели")
        over = dict(self.config.get("choice_override") or {})
        if variant:
            over[role] = variant
        else:
            over.pop(role, None)
        picks = dict(self.config.get("model_pick") or {})
        picks.pop(role, None)  # a file picked by hand belonged to the old choice
        self.config.update(choice_override=over, model_pick=picks)
        await self.resolve_models()
        entry = catalog.model_entry(role, self.choice()[role])
        have = role not in {m["role"] for m in self.missing}
        return web.json_response({"ok": True, "have": have, "size": entry["size"], "file": entry["file"]})

    async def h_update_check(self, _req):
        info = await updater.check(self.session)
        self.update_info = info if info.get("available") else None
        return web.json_response({**{k: v for k, v in info.items() if k not in ("asset", "sha256")},
                                  "can_update": updater.can_update()})

    async def h_update_install(self, _req):
        info = getattr(self, "update_info", None)
        if not info or not updater.can_update():
            raise web.HTTPBadRequest(text="Обновление недоступно")
        if self.jobs:
            raise web.HTTPConflict(text="Сейчас идёт генерация. Дождитесь её конца и нажмите «Обновить» снова.")
        try:
            installer = await updater.download(self.session, info)
        except Exception as e:
            raise web.HTTPBadGateway(text=str(e))
        updater.run(installer)
        # the installer is already running on its own: leave, so it can replace the files and open the app again
        asyncio.get_running_loop().call_later(3, lambda: os._exit(0))
        return web.json_response({"ok": True})

    async def h_system(self, _req):
        """CPU / RAM / GPU / disk readings for the sidebar."""
        where = self.output_dir or self.config.get("install_dir") or HOME
        return web.json_response(await asyncio.to_thread(self.monitor.sample, where))

    # ---------- first-run setup ----------

    async def setup_emit(self, ev):
        if ev.get("state") == "error":
            self.note_error(f'Установка, шаг {ev.get("step")}: {ev.get("label")}')
        if ev.get("state") != "progress" or ev.get("step") != getattr(self, "_last_step", None):
            self.setup_log = (self.setup_log + [ev])[-40:]
        self._last_step = ev.get("step")
        self.setup_last = ev
        await self.broadcast(ev)

    async def h_setup_info(self, req):
        install_dir = req.query.get("install_dir") or self.config.get("install_dir") or engine.default_install_dir()
        modes = [m for m in (req.query.get("modes") or ",".join(catalog.MODE_ROLES)).split(",") if m in catalog.MODE_ROLES]
        hw = await asyncio.to_thread(engine.detect_hardware, install_dir)
        choice, notes, verdict, profile = catalog.choose_models(hw, req.query.get("profile"))
        plan = [catalog.model_entry(r, choice[r]) for r in catalog.roles_for(modes or ["video"])]
        need_gb = round((catalog.COMFY_SIZE * 2.5 + sum(m["size"] for m in plan)) / 1e9)
        # download size of each profile for each set of modes, for the size picker
        sizes = {}
        for p in catalog.PROFILES:
            c = catalog.choose_models(hw, p)[0]
            sizes[p] = {m: round(catalog.plan_size(c, [m]) / 1e9) for m in catalog.MODE_ROLES}
            sizes[p]["selected"] = round(catalog.plan_size(c, modes or ["video"]) / 1e9)
        return web.json_response({
            "hardware": hw, "choice": choice, "notes": notes, "verdict": verdict,
            "profile": profile, "recommended": catalog.recommended_profile(hw), "modes": modes,
            "profiles": {k: {"title": v["title"], "about": v["about"], "sizes": sizes[k]} for k, v in catalog.PROFILES.items()},
            "install_dir": install_dir, "need_gb": need_gb,
            "plan": [{k: m[k] for k in ("role", "title", "file", "size", "variant")} for m in plan],
            "config": {k: self.config.get(k) for k in ("engine", "install_dir", "models_dir", "comfy_url", "setup_done")},
            "installing": bool(self.installer), "log": self.setup_log,
            "last": getattr(self, "setup_last", None),
        })

    async def h_setup_external(self, req):
        body = await req.json()
        url = (body.get("url") or "http://127.0.0.1:8188").strip().rstrip("/")
        if not url.startswith("http"):
            url = "http://" + url
        old = self.config.get("comfy_url")
        self.config.data["comfy_url"] = url
        self.process = None
        try:
            stats = await self.system_stats()
        except Exception as e:
            self.config.data["comfy_url"] = old
            return web.json_response({"error": f"ComfyUI не отвечает по адресу {url} ({type(e).__name__}). Откройте этот адрес "
                                      "в браузере: если страница ComfyUI не открывается, проверьте адрес и порт в окне, "
                                      "где запущен ComfyUI."}, status=400)
        modes = [m for m in (body.get("modes") or []) if m in catalog.MODE_ROLES] or list(catalog.MODE_ROLES)
        self.config.update(engine="external", comfy_url=url, setup_done=True, choice=catalog.DEFAULT_CHOICE, modes=modes)
        names, missing = await self.resolve_models()
        try:
            nodes = [n["pack"] for n in await self.nodes_status() if not n["have"] and not n["optional"]]
        except Exception:
            nodes = []
        return web.json_response({"ok": True, "version": stats["system"].get("comfyui_version"),
                                  "missing": [m["file"] for m in missing], "nodes_missing": nodes})

    async def h_mode_offered(self, req):
        mode = (await req.json()).get("mode")
        self.config.update(offered_modes=sorted(set(self.config.get("offered_modes") or []) | {mode}))
        return web.json_response({"ok": True})

    async def h_mode_add(self, req):
        """Own ComfyUI: switch on a mode added after the set-up, so its model files show up in Models (with download buttons)."""
        mode = (await req.json()).get("mode")
        if mode not in catalog.MODE_ROLES:
            return web.json_response({"error": "Неизвестный режим"}, status=400)
        modes = sorted(set(self.modes()) | {mode}, key=list(catalog.MODE_ROLES).index)
        self.config.update(modes=modes)
        await self.resolve_models()
        return web.json_response({"ok": True})

    async def h_setup_install(self, req):
        if self.installer:
            return web.json_response({"error": "Установка уже идёт"}, status=409)
        body = await req.json()
        install_dir = str(Path(body.get("install_dir") or engine.default_install_dir()))
        models_dir = (body.get("models_dir") or "").strip() or None
        if models_dir and not Path(models_dir).is_dir():
            return web.json_response({"error": f"Папка не найдена: {models_dir}"}, status=400)
        hw = await asyncio.to_thread(engine.detect_hardware, install_dir)
        profile = body.get("profile") or self.config.get("profile")
        choice, _, verdict, profile = catalog.choose_models(hw, profile)
        choice = {**choice, **(self.config.get("choice_override") or {})}  # a version the user picked by hand stays
        modes = [m for m in (body.get("modes") or self.modes()) if m in catalog.MODE_ROLES] or ["video"]
        if body.get("add_mode"):
            modes = sorted(set(self.modes()) | {body["add_mode"]}, key=list(catalog.MODE_ROLES).index)
        self.config.update(engine="managed", install_dir=install_dir, models_dir=models_dir,
                           choice=choice, profile=profile, modes=modes, setup_done=False)
        self.setup_log = []
        self.installer = engine.Installer(ROOT, self.session, self.setup_emit)
        self.installer.task = asyncio.create_task(self.run_install(install_dir, models_dir))
        return web.json_response({"ok": True})

    async def run_install(self, install_dir, models_dir):
        inst = self.installer
        try:
            Path(install_dir).mkdir(parents=True, exist_ok=True)
            root = await inst.install_comfy(install_dir)
            await inst.install_nodes(root)
            if models_dir:
                inst.write_extra_paths(root, models_dir)
            await inst.step("start", "progress", label="Запускаем ComfyUI…")
            if self.process:
                await asyncio.to_thread(self.process.stop)
            self.start_managed()
            await self.wait_comfy()
            await inst.step("start", "done", label="ComfyUI запущен")
            _, missing = await self.resolve_models()
            # shared files and the video mode first, so the first mode is usable sooner
            order = ["text_encoder", "video_vae", "audio_vae", "preview", "fast_unet"]
            missing.sort(key=lambda m: order.index(m["role"]) if m["role"] in order else len(order))
            if missing:
                gb = sum(m["size"] for m in missing) / 1e9
                await inst.step("models", "progress", label=f"Скачиваем модели: {len(missing)} файлов, {gb:.0f} ГБ")
                await inst.download_models(root / "ComfyUI" / "models", missing)
                await self.resolve_models()
            else:
                await inst.step("models", "done", label="Все модели уже есть")
            self.config.update(setup_done=True)
            await self.setup_emit({"type": "setup", "step": "finish", "state": "done", "label": "Готово!"})
        except engine.Cancelled:
            await self.setup_emit({"type": "setup", "step": "finish", "state": "cancelled",
                                   "label": "Установка остановлена. Скачанное сохранено — можно продолжить позже."})
        except Exception as e:
            await self.setup_emit({"type": "setup", "step": "finish", "state": "error", "label": str(e)})
        finally:
            self.installer = None

    def note_error(self, text):
        self.recent_errors = (self.recent_errors + [f"{datetime.now():%H:%M:%S} {text}"])[-15:]

    async def h_report(self, req):
        """Collect everything the author needs into one text file and show it in Explorer."""
        try:
            page = await req.json()
        except Exception:
            page = {}
        comfy = {"адрес": self.comfy, "режим": self.config.get("engine"), "онлайн": self.comfy_online,
                 "настройка завершена": bool(self.config.get("setup_done"))}
        try:
            st = await self.system_stats()
            sysinfo = st.get("system", {})
            comfy.update({"версия ComfyUI": sysinfo.get("comfyui_version"), "версия PyTorch": sysinfo.get("pytorch_version"),
                          "запуск": " ".join(map(str, sysinfo.get("argv", []))),
                          "устройства": "; ".join(f'{d.get("name")} (всего {round(d.get("vram_total", 0) / 1024 ** 3, 1)} ГБ, '
                                                  f'свободно {round(d.get("vram_free", 0) / 1024 ** 3, 1)} ГБ)' for d in st.get("devices", []))})
            await self.resolve_models()
            needed = catalog.roles_for(self.modes())
            missing = {m["role"] for m in self.missing}
            models = [{"title": catalog.ROLE_TITLES[r], "name": self.model_names.get(r), "have": r not in missing}
                      for r in self.choice() if r in needed]
        except Exception as e:
            comfy["ошибка связи"] = f"{type(e).__name__}: {e}"
            models = []
        hw = await asyncio.to_thread(engine.detect_hardware, self.config.get("install_dir"))
        long_states = []
        try:
            for d in sorted(LONG_DIR.iterdir()) if LONG_DIR.is_dir() else []:
                files = sorted(d.glob("state_*.latent"))
                if not files:
                    continue
                sizes = ", ".join(f"{f.name}: {round(f.stat().st_size / 1024, 1)} КБ" for f in files)
                long_states.append(f"{d.name}: {sizes}")
        except OSError as e:
            long_states.append(f"не удалось прочитать папку: {e}")
        text = await asyncio.to_thread(report.build, {
            "long_states": long_states,
            "home": HOME, "data_dir": DATA_DIR, "hw": hw, "page": page.get("page"), "comfy": comfy, "models": models,
            "config": self.config.data, "errors": self.recent_errors, "toasts": page.get("toasts"),
            "comfy_log": self.process.log_tail(60) if self.process else "",
            "llm_log": self.llm.log_tail(12)})
        folder = DATA_DIR / "reports"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"Отчёт для разработчика {datetime.now():%Y-%m-%d %H-%M-%S}.txt"
        path.write_text(text, encoding="utf-8")
        try:
            subprocess.Popen(["explorer", "/select,", str(path)])
        except Exception:
            pass
        return web.json_response({"ok": True, "path": str(path), "name": path.name})

    async def h_open_driver(self, _req):
        webbrowser.open("https://www.nvidia.com/Download/index.aspx?lang=ru")
        return web.json_response({"ok": True})

    async def h_search_web(self, req):
        """Open a web search in the user's browser (missing model file / custom node in 'Свои workflow')."""
        body = await req.json()
        q = str(body.get("q", "")).strip()[:200]
        if not q:
            raise web.HTTPBadRequest(text="Пустой запрос")
        webbrowser.open("https://www.google.com/search?q=" + urllib.parse.quote_plus(q))
        return web.json_response({"ok": True})

    def load_queue(self):
        try:
            data = json.loads(QUEUE_FILE.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        week = datetime.now().timestamp() - 7 * 86400
        return [j for j in data if isinstance(j, dict) and j.get("graph") and j.get("saved", 0) > week]

    def save_queue(self):
        items = list(self.saved_queue) + [
            {"graph": j["graph"], "stages": j.get("stages") or {}, "meta": j.get("meta") or {}, "last_output": j.get("last_output", False),
             "titles": j.get("titles") or {}, "pid": j.get("prompt_id"), "saved": datetime.now().timestamp()}
            for j in dict.values(self.jobs) if j.get("graph")]
        try:
            if items:
                QUEUE_FILE.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
            elif QUEUE_FILE.exists():
                QUEUE_FILE.unlink()
        except OSError:
            pass

    async def h_queue_saved(self, _req):
        return web.json_response({"count": len(self.saved_queue),
                                  "items": [{"mode": j["meta"].get("mode"), "prompt": (j["meta"].get("prompt") or "")[:80]} for j in self.saved_queue]})

    async def h_queue_resume(self, _req):
        items, self.saved_queue = list(self.saved_queue), []
        self.save_queue()
        out = []
        try:  # a job that ComfyUI still has (or already finished) while only the program was closed is not run twice
            _, q = await self.comfy_json("GET", "/queue")
            alive = {str(x[1]) for key in ("queue_running", "queue_pending") for x in (q or {}).get(key, []) if len(x) > 1}
        except Exception:
            alive = set()
        for j in items:
            if j.get("pid") and (j["pid"] in alive):
                continue
            resp = await self.queue(j["graph"], j.get("stages") or {}, j.get("meta") or {}, j.get("last_output", False), j.get("titles"))
            pid = json.loads(resp.text)["prompt_id"]
            out.append({"prompt_id": pid, "mode": j["meta"].get("mode"), "prompt": j["meta"].get("prompt", "")})
        return web.json_response({"jobs": out})

    async def h_queue_discard(self, _req):
        self.saved_queue = []
        self.save_queue()
        return web.json_response({"ok": True})

    async def h_support(self, req):
        """Open the author's donation page."""
        target = req.query.get("target") or ""
        url = SUPPORT_UNCENSORED_URL if target == "uncensored" else SUPPORT_URL
        webbrowser.open(url)
        return web.json_response({"ok": True})

    async def h_clipboard(self, _req):
        """Text from the Windows clipboard (the web view may refuse to let the page read it)."""
        text = ""
        try:
            import ctypes
            u, k = ctypes.windll.user32, ctypes.windll.kernel32
            u.GetClipboardData.restype = ctypes.c_void_p
            k.GlobalLock.restype = ctypes.c_void_p
            k.GlobalLock.argtypes = [ctypes.c_void_p]
            k.GlobalUnlock.argtypes = [ctypes.c_void_p]
            if u.OpenClipboard(0):
                try:
                    h = u.GetClipboardData(13)  # CF_UNICODETEXT
                    if h:
                        p = k.GlobalLock(h)
                        if p:
                            text = ctypes.wstring_at(p)
                            k.GlobalUnlock(h)
                finally:
                    u.CloseClipboard()
        except Exception:
            pass
        return web.json_response({"text": text})

    async def h_setup_cancel(self, _req):
        if self.installer:
            self.installer.cancelled = True
        return web.json_response({"ok": True})

    async def h_setup_reset(self, _req):
        """Back to the first-run wizard; installed files stay where they are."""
        if self.installer:
            return web.json_response({"error": "Сначала остановите установку"}, status=409)
        if self.process:
            await asyncio.to_thread(self.process.stop)
            self.process = None
        self.config.update(setup_done=False)
        return web.json_response({"ok": True})

    # ---------- cloud (RunningHub) ----------

    def rh(self):
        """A RunningHub client when cloud generation is switched on, else None."""
        if self.config.get("rh_enabled") and self.config.get("rh_key"):
            return cloud.RunningHub(self.session, self.config.get("rh_key"), self.config.get("rh_host", "ai"))
        return None

    async def h_cloud_get(self, _req):
        c = self.config
        return web.json_response({"enabled": bool(c.get("rh_enabled")), "has_key": bool(c.get("rh_key")),
                                  "host": c.get("rh_host", "ai"), "workflow_id": c.get("rh_workflow_id", ""),
                                  "instance": c.get("rh_instance", "default")})

    async def h_cloud_set(self, req):
        b = await req.json()
        upd = {}
        if b.get("key"):
            upd["rh_key"] = b["key"].strip()
        if b.get("clear_key"):
            upd["rh_key"] = None
            upd["rh_enabled"] = False
        if b.get("host") in cloud.HOSTS:
            upd["rh_host"] = b["host"]
        if "workflow_id" in b:
            upd["rh_workflow_id"] = str(b["workflow_id"]).strip()
        if b.get("instance") in cloud.INSTANCE_TYPES:
            upd["rh_instance"] = b["instance"]
        if "enabled" in b:
            upd["rh_enabled"] = bool(b["enabled"]) and bool(upd.get("rh_key", self.config.get("rh_key")))
        self.config.update(**upd)
        self.rh_models = None
        return await self.h_cloud_get(req)

    async def h_setup_cloud(self, _req):
        """First-run choice "cloud only": nothing installed locally."""
        if not (self.config.get("rh_key") and self.config.get("rh_enabled")):
            return web.json_response({"error": "Сначала сохраните ключ и включите облако"}, status=400)
        self.config.update(engine="cloud", setup_done=True, modes=list(catalog.MODE_ROLES))
        return web.json_response({"ok": True})

    async def h_cloud_test(self, _req):
        key = self.config.get("rh_key")
        if not key:
            return web.json_response({"error": "Вставьте API-ключ"}, status=400)
        rh = cloud.RunningHub(self.session, key, self.config.get("rh_host", "ai"))
        try:
            acc = await rh.account()
            names = await rh.model_names(self.config.get("rh_workflow_id")) if self.config.get("rh_workflow_id") else set()
        except cloud.CloudError as e:
            return web.json_response({"error": str(e)}, status=400)
        return web.json_response({"ok": True, "account": acc, "models": sorted(names)[:40]})

    async def queue_cloud(self, rh, graph, stages, meta):
        if not self.config.get("rh_workflow_id"):
            return web.json_response({"error": "Укажите ID любого вашего workflow на RunningHub (раздел «Модели» → «Облако»)"},
                                     status=400)
        if getattr(self, "rh_models", None) is None:
            try:
                self.rh_models = await rh.model_names(self.config.get("rh_workflow_id"))
            except cloud.CloudError:
                self.rh_models = set()
        cloud.remap_models(graph, self.rh_models)
        try:
            task = await rh.create(graph, self.config.get("rh_workflow_id"), self.config.get("rh_instance", "default"))
        except cloud.CloudError as e:
            return web.json_response({"error": str(e)}, status=400)
        pid = "rh-" + task
        self.jobs[pid] = {"prompt_id": pid, "state": "queued", "stages": {}, "stage": None,
                          "meta": dict(meta, cloud=True), "task": task}
        asyncio.create_task(self.poll_cloud(rh, pid))
        return web.json_response({"prompt_id": pid, "seed": meta["seed"]})

    async def poll_cloud(self, rh, pid):
        job = self.jobs.get(pid)
        last = None
        try:
            while pid in self.jobs:
                status = await rh.status(job["task"])
                if status != last:
                    last = status
                    stage = {"QUEUED": "cloud_queued", "RUNNING": "cloud_run"}.get(status)
                    if stage:
                        await self.send_stage(job, stage)
                if status == "SUCCESS":
                    outs = await rh.outputs(job["task"]) or []
                    video = next((o for o in outs if str(o.get("fileType", "")).lower() in ("mp4", "webm", "mov", "mkv")
                                  or str(o.get("fileUrl", "")).lower().split("?")[0].endswith((".mp4", ".webm", ".mov"))), None)
                    if not video:
                        raise cloud.CloudError("RunningHub не вернул видео")
                    dest = DATA_DIR / "cloud" / f'{pid}.mp4'
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    async with self.session.get(video["fileUrl"]) as r:
                        dest.write_bytes(await r.read())
                    self.jobs.pop(pid, None)
                    item = dict(job["meta"], id=uuid.uuid4().hex[:12], prompt_id=pid,
                                date=datetime.now().isoformat(timespec="seconds"), fav=False,
                                file={"type": "local", "path": str(dest), "filename": dest.name})
                    if video.get("consumeCoins"):
                        item["sub"] = f'{item.get("sub", "")} · ☁ {video["consumeCoins"]} RH'.strip(" ·")
                    self.library.add(item)
                    await self.broadcast({"type": "done", "prompt_id": pid, "item": self.public(item)})
                    return
                if status == "FAILED":
                    await rh.outputs(job["task"])  # raises with the failure reason
                    raise cloud.CloudError("Задание на RunningHub завершилось ошибкой")
                await asyncio.sleep(5)
        except Exception as e:
            if pid in self.jobs:
                self.jobs.pop(pid, None)
                await self.broadcast({"type": "error", "prompt_id": pid, "message": str(e)})

    async def read_form(self, req, file_fields):
        form, files = {}, {}
        rh = self.rh()
        async for field in await req.multipart():
            if field.name in file_fields and field.filename:
                files[field.name] = await self.upload_input(field, rh)
            else:
                form[field.name] = await field.text()
        prompt = form.get("prompt", "").strip()
        aspect, quality = form.get("aspect", "16:9"), form.get("quality", "720p")
        if not prompt:
            raise web.HTTPBadRequest(text="Напишите промпт")
        if aspect not in wf.ASPECTS or quality not in wf.QUALITY:
            raise web.HTTPBadRequest(text="Неизвестный формат или качество")
        seed = int(form["seed"]) if form.get("seed", "").isdigit() else random.randrange(2 ** 50)
        return form, files, prompt, aspect, quality, seed

    async def queue(self, graph, stages, meta, last_output=False, titles=None):
        scenes = stages.pop("__scenes__", None)   # which sampler is which scene (long video), for the progress text
        rh = self.rh()
        if rh:
            return await self.queue_cloud(rh, graph, stages, meta)
        await self.stop_llm()  # the agent's model and H3 do not fit in VRAM together
        mode_key = (meta or {}).get("mode", "")
        same = mode_key == getattr(self, "last_mode", None) and mode_key in ("krea", "kreareal", "qwen", "flux")
        self.last_mode = mode_key
        # nothing is running or waiting: drop the previous job's models and cache so each job starts clean.
        # Not between pictures of the same mode: the same big models would just be loaded from disk again.
        if not self.jobs and not same:
            try:
                await self.comfy_json("POST", "/free", json={"unload_models": True, "free_memory": True})
                await asyncio.sleep(1.5)
                gpu = await asyncio.to_thread(self.monitor.gpu)
                used = (gpu or {}).get("vram_used_gb")
                if used is not None and used > 2.5:  # right after /free ComfyUI itself holds well under this
                    await self.broadcast({"type": "notice", "text": (
                        f"Перед заданием занято {used:g} ГБ видеопамяти. Скорее всего, её держат другие программы "
                        "(браузер, Adobe, игры). Закройте их, иначе генерация может идти медленнее.")})
            except Exception:
                pass
        status, res = await self.comfy_json("POST", "/prompt", json={
            "prompt": graph, "client_id": CLIENT_ID,
            "extra_data": {"preview_method": "auto"}})
        if status != 200:
            return web.json_response({"error": comfy_error_text(res)}, status=400)
        pid = res["prompt_id"]
        self.jobs[pid] = {"prompt_id": pid, "state": "queued", "stages": stages, "stage": None, "meta": meta, "last_output": last_output, "titles": titles or {},
                          "graph": graph, "scenes": scenes}
        return web.json_response({"prompt_id": pid, "seed": meta["seed"], "queue_number": res.get("number")})

    async def need_models(self, mode):
        if self.rh():  # models live on RunningHub
            if not self.model_names:
                self.model_names = {r: catalog.VARIANTS[r][v]["name"] for r, v in self.choice().items()}
            return
        if mode not in self.modes():
            raise web.HTTPBadRequest(text="Этот режим не установлен. Добавьте его в разделе «Модели».")
        if not self.model_names:
            await self.resolve_models()
        lost = [m["file"] for m in self.missing if m["role"] in catalog.MODE_ROLES[mode]]
        if lost:
            raise web.HTTPBadRequest(text="ComfyUI не видит модели: " + ", ".join(lost)
                                     + ". Откройте раздел «Модели».")

    # ---------- the user's own workflows (API format), for those with their own ComfyUI ----------

    def wf_path(self, wid):
        if not wid.isalnum():
            raise web.HTTPNotFound()
        return WORKFLOWS_DIR / f"{wid}.json"

    def wf_load(self, wid):
        p = self.wf_path(wid)
        if not p.exists():
            raise web.HTTPNotFound(text="Workflow не найден")
        return json.loads(p.read_text(encoding="utf-8"))

    async def h_wf_list(self, _req):
        items = []
        for p in sorted(WORKFLOWS_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True) if WORKFLOWS_DIR.exists() else []:
            try:
                items.append({"id": p.stem, "name": json.loads(p.read_text(encoding="utf-8")).get("name", p.stem)})
            except (OSError, ValueError):
                pass
        return web.json_response(items)

    async def h_wf_add(self, req):
        body = await req.json()
        graph = body.get("graph")
        try:
            customwf.validate_api(graph)
        except ValueError as e:
            raise web.HTTPBadRequest(text=str(e))
        WORKFLOWS_DIR.mkdir(parents=True, exist_ok=True)
        wid = uuid.uuid4().hex[:10]
        name = (body.get("name") or "Мой workflow").strip()[:80]
        self.wf_path(wid).write_text(json.dumps({"name": name, "graph": graph}, ensure_ascii=False), encoding="utf-8")
        return web.json_response({"id": wid, "name": name})

    async def h_wf_delete(self, req):
        p = self.wf_path(req.match_info["id"])
        p.unlink(missing_ok=True)
        return web.json_response({"ok": True})

    async def wf_infos(self, graph):
        """/object_info of every node class the workflow uses (a class ComfyUI does not know is left out)."""
        async def one(cls):
            try:
                status, data = await self.comfy_json("GET", f"/object_info/{cls}", timeout=aiohttp.ClientTimeout(total=15))
            except Exception:
                raise web.HTTPServiceUnavailable(text="ComfyUI не отвечает")
            return cls, (data[cls] if status == 200 and isinstance(data, dict) and cls in data else None)
        pairs = await asyncio.gather(*(one(c) for c in {n["class_type"] for n in graph.values()}))
        return {c: i for c, i in pairs if i is not None}

    async def h_wf_get(self, req):
        wf_file = self.wf_load(req.match_info["id"])
        report = customwf.analyze(wf_file["graph"], await self.wf_infos(wf_file["graph"]))
        return web.json_response({"id": req.match_info["id"], "name": wf_file["name"], **report})

    async def h_generate_custom(self, req):
        form, uploads = {}, {}
        async for field in await req.multipart():
            if field.name.startswith("f:") and field.filename:
                uploads[field.name[2:]] = await self.upload_input(field)
            else:
                form[field.name] = await field.text()
        wf_file = self.wf_load(form.get("wf", ""))
        graph = wf_file["graph"]
        report = customwf.analyze(graph, await self.wf_infos(graph))
        if report["missing_nodes"]:
            raise web.HTTPBadRequest(text="В вашем ComfyUI нет нод: " + ", ".join(sorted({m["cls"] for m in report["missing_nodes"]})))
        try:
            values = json.loads(form.get("values") or "{}")
        except ValueError:
            values = {}
        graph = customwf.apply_values(graph, values, uploads)
        await self.stop_llm()
        prompt = next((str(values.get(w["key"], w["value"])) for w in report["widgets"] if w["role"] == "prompt"), "")
        seed = next((values.get(w["key"], w["value"]) for w in report["widgets"] if w["role"] == "seed"), None)
        return await self.queue(graph, customwf.stage_map(graph), {
            "mode": "custom", "sub": wf_file["name"], "wf": form.get("wf"), "cw_values": values,
            "prompt": prompt, "aspect": "", "q": "", "len": 0.0, "res": "",
            "seed": seed if isinstance(seed, (int, float)) else None}, last_output=True,
            titles={nid: (n.get("_meta") or {}).get("title", "") for nid, n in graph.items() if n["class_type"].startswith("Save")})

    async def read_image_form(self, req):
        """Multipart form of the picture sections: text fields plus one optional file `image`."""
        if self.rh():
            raise web.HTTPBadRequest(text="Картинки работают только на своём ComfyUI (не в облаке RunningHub).")
        form, image, extra = {}, None, []
        async for field in await req.multipart():
            if field.name == "image" and field.filename:
                image = await self.upload_input(field)
            elif field.name in ("image2", "image3") and field.filename:
                extra.append(await self.upload_input(field))
            else:
                form[field.name] = await field.text()
        form["_extra"] = extra
        seed = int(form["seed"]) if form.get("seed", "").isdigit() else random.randrange(2 ** 50)
        return form, image, seed

    async def h_upscale(self, req):
        """Improve the quality of a finished video or picture from the library (SeedVR2), the sound stays."""
        peek = await req.json()
        soft = peek.get("method") == "soft2k"  # "up to 2K", soft (pictures only)
        relay = peek.get("engine") == "relay"   # the fast RelayVSR x2 (video only) instead of SeedVR2
        await self.need_models("relayvsr" if relay else "upscale")
        if self.rh():
            raise web.HTTPBadRequest(text="Увеличение работает только на своём ComfyUI (не в облаке RunningHub).")
        body = peek
        item = self.library.get(body.get("id", ""))
        if not item:
            raise web.HTTPBadRequest(text="Файл не найден")
        if soft and item.get("kind") != "image":
            raise web.HTTPBadRequest(text="«До 2К» работает для картинок. Для видео выберите «Увеличить в 1,5 или 2 раза».")
        path = self.file_path(item)
        content = None
        if path and path.exists():
            content = path.read_bytes()
            suffix = path.suffix
        else:
            # output folder unknown (own ComfyUI without --output-directory): take the file from ComfyUI itself
            f = item["file"]
            params = {"filename": f.get("filename", ""), "subfolder": f.get("subfolder", ""), "type": f.get("type", "output")}
            try:
                async with self.session.get(self.comfy + "/view", params=params) as r:
                    if r.status == 200:
                        content = await r.read()
            except aiohttp.ClientError:
                content = None
            suffix = Path(f.get("filename", "")).suffix
        if not content:
            raise web.HTTPBadRequest(text="Файл не найден")
        return await self._upscale_core(body, item, content, suffix, relay, soft)

    async def h_upscale_file(self, req):
        """The same enlarge for ANY file of the user (made elsewhere): the file comes with the request, not from the library."""
        form, content, name = {}, None, ""
        async for part in await req.multipart():
            if part.name == "file":
                name = part.filename or ""
                content = await part.read()
            else:
                form[part.name] = (await part.text()).strip()
        if not content:
            raise web.HTTPBadRequest(text="Файл не получен")
        suffix = Path(name).suffix.lower()
        pic = suffix in (".png", ".jpg", ".jpeg", ".webp", ".bmp")
        relay = form.get("engine") == "relay"
        soft = form.get("method") == "soft2k"
        await self.need_models("relayvsr" if relay else "upscale")
        if self.rh():
            raise web.HTTPBadRequest(text="Увеличение работает только на своём ComfyUI (не в облаке RunningHub).")
        if soft and not pic:
            raise web.HTTPBadRequest(text="«До 2К» работает для картинок. Для видео выберите «Увеличить в 1,5 или 2 раза».")
        item = {"id": "", "kind": "image" if pic else "video", "res": form.get("res", ""), "len": float(form.get("len") or 0),
                "prompt": "Увеличение: " + name[:60], "aspect": "", "q": ""}
        return await self._upscale_core(form, item, content, suffix, relay, soft)

    async def _upscale_core(self, body, item, content, suffix, relay, soft):
        pic = item.get("kind") == "image"
        if relay:
            if pic:
                raise web.HTTPBadRequest(text="Быстрое увеличение работает только для видео. Для картинки выберите SeedVR2.")
            try:
                w0, h0 = (int(x) for x in str(item.get("res", "")).split("×"))
                secs = float(item.get("len") or 0)
            except ValueError:
                w0 = h0 = 0; secs = 0.0
            if w0 and secs and w0 * h0 * secs > wf.RELAY_PIXEL_SECONDS:   # decoded in memory as a whole: longer clips do not fit
                ok_secs = wf.RELAY_PIXEL_SECONDS / (w0 * h0)
                raise web.HTTPBadRequest(text=f"Видео слишком большое для быстрого увеличения: оно целиком загружается в память. "
                                              f"При размере {w0}×{h0} можно не больше {ok_secs:.0f} с (у вас {secs:.0f} с). "
                                              f"Выберите SeedVR2 или сделайте ролик короче.")
        scale = 2.0 if relay else wf.UPSCALE_SCALES.get(str(body.get("scale")), 1.5)
        data = aiohttp.FormData()
        name = f"h3s_{uuid.uuid4().hex[:8]}{suffix or ('.png' if pic else '.mp4')}"
        data.add_field("image", content, filename=name, content_type="image/png" if pic else "video/mp4")
        data.add_field("subfolder", "h3studio")
        data.add_field("type", "input")
        status, res = await self.comfy_json("POST", "/upload/image", data=data)
        if status != 200:
            raise web.HTTPBadGateway(text="Не удалось передать файл в ComfyUI")
        video = f'{res["subfolder"]}/{res["name"]}' if res.get("subfolder") else res["name"]
        seed = random.randrange(2 ** 40)
        try:
            w, h = (int(x) for x in str(item.get("res", "")).split("×"))
            res = f"{round(w * scale / 2) * 2}×{round(h * scale / 2) * 2}"
        except ValueError:
            res = ""
        if soft:  # long side 2048, SeedVR2 mixed half with the plain enlarge
            try:
                w0, h0 = (int(x) for x in str(item.get("res", "")).split("×"))
            except ValueError:
                raise web.HTTPBadRequest(text="Не знаю размер картинки, увеличить до 2К не получится.")
            scale = max(1.0, 2048 / max(w0, h0))
            graph, stages = wf.upscale_image_graph(self.model_names, video, scale, seed, soften=0.5)
            return await self.queue(graph, stages, {
                "mode": "upscale", "sub": "До 2К, мягко", "prompt": item.get("prompt", "") or "Увеличение до 2К",
                "aspect": item.get("aspect", ""), "q": "2K", "len": 0.0, "res": f"{round(w0 * scale / 2) * 2}×{round(h0 * scale / 2) * 2}", "seed": seed, "src": item.get("id", "")})
        def num(key, default):   # a slider value 0..1 from the window (a string when it comes with an uploaded file)
            try:
                return min(1.0, max(0.0, float(body.get(key))))
            except (TypeError, ValueError):
                return default
        if relay:
            graph, stages = wf.relay_graph(self.model_names, video, seed, detail=num("detail", 0.7))
        else:
            soften = round(1.0 - num("sharp", 1.0), 2)   # 0 = full sharpness of SeedVR2, more = mixed with the plain enlarge
            graph, stages = (wf.upscale_image_graph if pic else wf.upscale_graph)(self.model_names, video, scale, seed, soften=soften)
        return await self.queue(graph, stages, {
            "mode": "upscale", "sub": "Быстрое увеличение ×2 (RelayVSR)" if relay else f"Увеличение ×{body.get('scale', '1.5')}", "prompt": item.get("prompt", "") or "Увеличение",
            "aspect": item.get("aspect", ""), "q": item.get("q", ""), "len": item.get("len", 0.0), "res": res, "seed": seed,
            "src": item.get("id", "")})

    async def h_generate_krea(self, req):
        await self.need_models("krea")
        form, _, seed = await self.read_image_form(req)
        prompt = form.get("prompt", "").strip()
        if not prompt:
            raise web.HTTPBadRequest(text="Напишите промпт")
        lab = {"label": form["batch_label"][:20]} if form.get("batch_label") else {}  # "3 из 8" for a series from a prompt list
        aspect = form.get("aspect", "16:9")
        if aspect not in wf.KREA_SIZES:
            raise web.HTTPBadRequest(text="Неизвестный формат")
        width, height = wf.KREA_SIZES[aspect]
        custom = round_size(form)
        if custom:
            width, height = custom
        engine = form.get("engine", "krea")
        if engine == "flux":
            await self.need_models("flux")
            graph, stages = wf.flux_t2i_graph(self.model_names, prompt, width, height, seed)
            return await self.queue(graph, stages, {
                "mode": "krea", "sub": "Картинка (FLUX.2)", "prompt": prompt, "aspect": aspect, "q": "1K",
                "len": 0.0, "res": f"{width}×{height}", "seed": seed, **lab})
        if engine == "qwen":
            await self.need_models("qwen")
            graph, stages = wf.qwen_t2i_graph(self.model_names, prompt, form.get("negative", "").strip()[:600], width, height, seed)
            return await self.queue(graph, stages, {
                "mode": "krea", "sub": "Картинка (Qwen 2.1)", "prompt": prompt, "aspect": aspect, "q": "1K",
                "len": 0.0, "res": f"{width}×{height}", "seed": seed, **lab})
        if form.get("style") == "real":
            await self.need_models("kreareal")
            scale = wf.KREA_REAL_SCALES.get(form.get("scale", "2k"), 1.5)
            graph, stages = wf.krea_real_graph(self.model_names, prompt, width, height, seed, scale)
            return await self.queue(graph, stages, {
                "mode": "krea", "sub": "Картинка: реализм", "prompt": prompt, "aspect": aspect, "q": "4K" if scale > 2 else "2K",
                "len": 0.0, "res": f"{int(width * scale)}×{int(height * scale)}", "seed": seed, **lab})
        graph, stages = wf.krea_graph(self.model_names, prompt, width, height, seed)
        return await self.queue(graph, stages, {
            "mode": "krea", "sub": "Картинка по тексту", "prompt": prompt, "aspect": aspect, "q": "1K",
            "len": 0.0, "res": f"{width}×{height}", "seed": seed, **lab})

    async def h_generate_qwen(self, req):
        form, image, seed = await self.read_image_form(req)
        flux = form.get("kind") != "sheet" and form.get("engine") == "flux"
        await self.need_models("flux" if flux else "qwen")
        if form.get("kind") == "story" and form.get("frames") == "1":
            # storyboard frame by frame (FLUX.2 klein): every frame is its own picture, the sheet is glued from them
            use_qwen = form.get("engine") == "qwen"
            await self.need_models("qwen" if use_qwen else "flux")
            frames = split_frames(form.get("prompt", ""))
            if len(frames) not in wf.FRAME_COLS:
                raise web.HTTPBadRequest(text=f"Нужно 2, 3, 4, 6, 8 или 9 кадров, а в описании их {len(frames)}. Напишите каждый кадр с новой строки: «1: ...», «2: ...».")
            refs = [x for x in ([image] if image else []) + list(form.get("_extra") or [])]
            build = wf.qwen_frames_graph if use_qwen else wf.flux_frames_graph
            graph, stages, titles = build(self.model_names, refs, frames, seed, size=round_size(form) or wf.FRAME_SIZE, line=form.get("line", "white"))
            return await self.queue(graph, stages, {
                "mode": "qwen", "sub": "Раскадровка по кадрам", "prompt": form.get("prompt", "").strip(), "aspect": "",
                "q": "", "len": 0.0, "res": "", "seed": seed}, titles=titles)
        if not image:
            raise web.HTTPBadRequest(text="Загрузите картинку")
        if form.get("kind") == "story":
            prompt = form.get("prompt", "").strip()
            if not prompt:
                raise web.HTTPBadRequest(text="Опишите историю по кадрам")
            refs = [image] + list(form.get("_extra") or [])
            sz = round_size(form) or wf.STORY_SIZE
            if flux:
                graph, stages = wf.flux_story_graph(self.model_names, refs, prompt, seed, sz)
            else:
                graph, stages = wf.qwen_story_graph(self.model_names, refs, prompt, seed, sz)
            return await self.queue(graph, stages, {
                "mode": "qwen", "sub": "Раскадровка по картинкам", "prompt": prompt, "aspect": "",
                "q": "", "len": 0.0, "res": "%d×%d" % wf.STORY_SIZE, "seed": seed})
        if flux:
            prompt = form.get("prompt", "").strip()
            if not prompt:
                raise web.HTTPBadRequest(text="Напишите, что изменить на картинке")
            mp = wf.QWEN_MP.get(form.get("size", "1"), 1.0)
            graph, stages = wf.flux_edit_graph(self.model_names, prompt, image, mp, seed, form.get("_extra") or (), round_size(form))
            return await self.queue(graph, stages, {
                "mode": "qwen", "sub": "Правка картинки (FLUX.2)", "prompt": prompt, "aspect": "",
                "q": "", "len": 0.0, "res": "", "seed": seed})
        if form.get("kind") == "sheet":
            panels = 6 if form.get("panels") == "6" else 4
            graph, stages, prompt = wf.qwen_sheet_graph(
                self.model_names, image, seed, headless=form.get("headless", "1") != "0",
                background=form.get("background", "grey"), marks=form.get("marks", "")[:300], panels=panels)
            sub, res = "Лист персонажа", "%d×%d" % (wf.SHEET6_SIZE if panels == 6 else wf.SHEET_SIZE)
            prompt = "Лист персонажа" + (" (6 панелей)" if panels == 6 else "") + (": " + form["marks"].strip()[:80] if form.get("marks", "").strip() else "")
        else:
            prompt = form.get("prompt", "").strip()
            if not prompt:
                raise web.HTTPBadRequest(text="Напишите, что изменить на картинке")
            mp = wf.QWEN_MP.get(form.get("size", "1"), 1.0)
            size = round_size(form) or wf.qwen_edit_size(form.get("aspect", ""), mp)
            graph, stages = wf.qwen_edit_graph(self.model_names, prompt, image, mp, seed, size, form.get("_extra") or ())
            sub, res = "Правка картинки", ("%d×%d" % size if size else "")
        return await self.queue(graph, stages, {
            "mode": "qwen", "sub": sub, "prompt": prompt, "aspect": form.get("aspect", "") if form.get("kind") != "sheet" else "",
            "q": "", "len": 0.0, "res": res, "seed": seed})

    async def h_generate_video(self, req):
        await self.need_models("video")
        form, images, prompt, aspect, quality, seed = await self.read_form(req, ("first", "last"))
        seconds = min(MAX_SECONDS, max(3.0, float(form.get("duration", 5))))
        width, height = wf.resolution(aspect, quality, form)
        first, last = images.get("first"), images.get("last") if images.get("first") else None
        loop = bool(first) and form.get("loop") == "1"
        if loop:  # seamless loop (live wallpaper): the video ends on the same picture it starts with
            last = first

        fast = None
        if form.get("accel") == "sla":
            miss = {mm["role"] for mm in self.missing}
            if "sla_lora" in miss or "sla_lora" not in self.model_names:
                raise web.HTTPBadRequest(text="Для быстрого режима нужен файл Turbo-SLA (около 2 ГБ). Нажмите «Скачать» под переключателем «Ускорение» "
                                              "или выберите «Обычное».")
            fast = self.model_names["sla_lora"]
        graph, stages = wf.video_graph(self.model_names, prompt, width, height, seconds, seed, first, last, self.custom_loras(form), fast)
        sub = "Зацикленное видео" if loop else "Кадр → кадр" if last else "Картинка → видео" if first else "Текст → видео"
        if fast:
            sub += " · быстрый черновик"
        return await self.queue(graph, stages, {
            "mode": "video", "sub": sub, "prompt": prompt, "aspect": aspect, "q": quality,
            "len": seconds, "res": f"{width}×{height}", "seed": seed})

    async def h_generate_lipsync(self, req):
        await self.need_models("lipsync")
        ref_fields = tuple(f"ref{i}" for i in range(9))
        form, files, prompt, aspect, quality, seed = await self.read_form(req, ref_fields + ("audio",))
        if "audio" not in files:
            raise web.HTTPBadRequest(text="Загрузите аудио")
        refs = [files[f] for f in ref_fields if f in files]
        if not refs:
            raise web.HTTPBadRequest(text="Добавьте хотя бы один референс — лицо персонажа")
        seconds = float(form.get("audio_len", 0))
        if not 1 <= seconds <= 15:
            raise web.HTTPBadRequest(text="Фрагмент аудио должен быть от 1 до 15 секунд")

        ref_size = form.get("ref_size") if form.get("ref_size") in ("match", "max") else "max"
        scale = wf.lipsync_scale(quality)
        out_w, out_h = wf.resolution(aspect, quality, form)
        width, height = wf.r32(out_w / scale), wf.r32(out_h / scale)
        extra = self.optional_loras(form, (("realism", "realism_lora"),)) + self.custom_loras(form)
        graph, stages = wf.lipsync_graph(self.model_names, wf.lipsync_prompt(prompt, len(refs)), width, height,
                                         seconds, seed, files["audio"], refs, scale, ref_size, extra)
        speed = "максимум" if ref_size == "max" else "быстро"
        return await self.queue(graph, stages, {
            "mode": "lipsync", "sub": f"{len(refs)} реф. ({speed}) · аудио {seconds:.1f} с", "prompt": prompt,
            "aspect": aspect, "q": quality, "len": seconds, "res": f"{out_w}×{out_h}", "seed": seed,
            "ref_size": ref_size})

    async def read_ref_form(self, req, max_images):
        fields = tuple(f"img{i}" for i in range(max_images)) + tuple(f"vid{i}" for i in range(3)) + \
            tuple(f"aud{i}" for i in range(3))
        form, files, prompt, aspect, quality, seed = await self.read_form(req, fields)
        pick = lambda p, n: [files[f"{p}{i}"] for i in range(n) if f"{p}{i}" in files]
        images, videos, audios = pick("img", max_images), pick("vid", 3), pick("aud", 3)
        if not images and not videos:
            raise web.HTTPBadRequest(text="Добавьте хотя бы одну референс-картинку")
        if len(images) + len(videos) + len(audios) > 12:
            raise web.HTTPBadRequest(text="Всего не больше 12 референсов (картинки + видео + аудио)")
        seconds = min(MAX_SECONDS, max(3.0, float(form.get("duration", 6))))
        return form, prompt, aspect, quality, seed, images, videos, audios, seconds

    def custom_loras(self, form=None):
        """(file, strength) for the LoRAs the user added by hand on the Models screen or in the form."""
        if self.rh():
            return []
        configured = [(x["file"], float(x["strength"])) for x in self.config.get("custom_loras") or [] if x.get("on", True) and x.get("file")]
        if form and form.get("custom_loras"):
            import json
            try:
                raw = json.loads(form["custom_loras"]) if isinstance(form["custom_loras"], str) else form["custom_loras"]
                return [(x["file"] if "file" in x else x.get("name"), float(x.get("strength", x.get("v", 0.8))))
                        for x in raw if (x.get("file") or x.get("name")) and x.get("on", True)]
            except Exception:
                pass
        return configured

    async def h_loras_custom(self, _req):
        import loras_registry
        files_set = set()
        try:
            _, comfy_files = await self.comfy_json("GET", "/models/loras")
            if comfy_files:
                for f in comfy_files:
                    files_set.add(f)
        except Exception:
            pass
        root = self.comfy_root()
        loras_dir = None
        if root:
            loras_dir = root / "ComfyUI" / "models" / "loras"
            if loras_dir.exists():
                for f in loras_dir.rglob("*.safetensors"):
                    files_set.add(str(f.relative_to(loras_dir)).replace("/", "\\"))
        clean = sorted(x for x in files_set if "ref2v_turbo" not in x.lower())
        custom_meta = self.config.get("custom_lora_meta") or {}
        meta_map = {}
        for f in clean:
            fp = (loras_dir / f) if loras_dir else None
            meta_map[f] = loras_registry.get_lora_metadata(f, full_file_path=fp, custom_meta=custom_meta)
        for it in (self.config.get("custom_loras") or []):
            f = it.get("file")
            if f and f not in meta_map:
                fp = (loras_dir / f) if loras_dir else None
                meta_map[f] = loras_registry.get_lora_metadata(f, full_file_path=fp, custom_meta=custom_meta)
        items = self.config.get("custom_loras") or []
        conflicts = loras_registry.analyze_lora_conflicts(items, custom_meta=custom_meta)
        return web.json_response({
            "files": clean,
            "items": items,
            "meta": meta_map,
            "conflicts": conflicts,
            "civitai_token": self.config.get("civitai_token") or "",
            "hf_token": self.config.get("hf_token") or "",
        })

    async def h_loras_conflicts(self, req):
        import loras_registry
        data = await req.json()
        items = data.get("items") or []
        builtins = data.get("builtins") or {}
        custom_meta = self.config.get("custom_lora_meta") or {}
        conflicts = loras_registry.analyze_lora_conflicts(items, custom_meta=custom_meta, builtins=builtins)
        return web.json_response({"conflicts": conflicts})

    async def h_loras_open_folder(self, _req):
        root = self.comfy_root()
        if not root:
            return web.json_response({"error": "Папка ComfyUI не найдена"}, status=400)
        loras_dir = root / "ComfyUI" / "models" / "loras"
        loras_dir.mkdir(parents=True, exist_ok=True)
        try:
            import os
            os.startfile(str(loras_dir))
        except Exception:
            import subprocess
            subprocess.Popen(["explorer", str(loras_dir)])
        return web.json_response({"ok": True, "path": str(loras_dir)})

    async def h_loras_meta_save(self, req):
        data = await req.json()
        f = data.get("file")
        if not f:
            return web.json_response({"error": "No file specified"}, status=400)
        all_meta = dict(self.config.get("custom_lora_meta") or {})
        entry = dict(all_meta.get(f) or {})
        if "name" in data:
            entry["name"] = data["name"].strip()
        if "desc" in data:
            entry["desc"] = data["desc"].strip()
        if "triggers" in data:
            tr = data["triggers"]
            if isinstance(tr, str):
                tr = [t.strip() for t in tr.split(",") if t.strip()]
            entry["triggers"] = tr
        if "category" in data:
            entry["category"] = data["category"].strip()
        all_meta[f] = entry
        self.config.update(custom_lora_meta=all_meta)
        return web.json_response({"ok": True})

    async def h_civitai_search(self, req):
        import civitai_client
        query = req.query.get("query", "").strip()
        cursor = req.query.get("cursor")
        page = int(req.query.get("page", 1)) if req.query.get("page") else 1
        mirror = req.query.get("mirror", "red")
        nsfw = req.query.get("nsfw", "true").lower() in ("true", "1")
        token = (req.query.get("token") or self.config.get("civitai_token") or "").strip()
        try:
            res = await civitai_client.search_loras(self.session, query=query, page=page, limit=20, mirror=mirror, nsfw=nsfw, cursor=cursor, token=token)
            return web.json_response(res)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def h_civitai_resolve(self, req):
        body = await req.json()
        link = (body.get("link") or "").strip()
        mirror = body.get("mirror", "red")
        if "huggingface.co" in link or (re.match(r"^[a-zA-Z0-9_\-\.]+/[a-zA-Z0-9_\-\.]+", link) and "civitai" not in link):
            import hf_client
            token = self.config.get("hf_token")
            try:
                res = await hf_client.resolve_url_or_repo(self.session, link, token=token)
                return web.json_response(res)
            except Exception as e:
                return web.json_response({"error": str(e)}, status=400)

        import civitai_client
        try:
            res = await civitai_client.resolve_url_or_id(self.session, link, mirror=mirror)
            return web.json_response(res)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=400)

    async def h_hf_search(self, req):
        import hf_client
        query = req.query.get("query", "").strip()
        limit = int(req.query.get("limit", 30)) if req.query.get("limit") else 30
        token = (req.query.get("token") or self.config.get("hf_token") or "").strip()
        try:
            res = await hf_client.search_loras(self.session, query=query, limit=limit, token=token)
            return web.json_response(res)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def h_hf_resolve(self, req):
        import hf_client
        body = await req.json()
        link = (body.get("link") or "").strip()
        token = (body.get("token") or self.config.get("hf_token") or "").strip()
        try:
            res = await hf_client.resolve_url_or_repo(self.session, link, token=token)
            return web.json_response(res)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=400)

    async def h_loras_tokens_save(self, req):
        data = await req.json()
        updates = {}
        if "civitai_token" in data:
            updates["civitai_token"] = str(data["civitai_token"] or "").strip()
        if "hf_token" in data:
            updates["hf_token"] = str(data["hf_token"] or "").strip()
        if updates:
            self.config.update(**updates)
        return web.json_response({"ok": True, **updates})

    async def h_civitai_download(self, req):
        import civitai_client
        body = await req.json()
        url = body.get("url")
        filename = body.get("filename")
        source = body.get("source") or ("HuggingFace" if "huggingface" in str(url) else "Civitai")

        if "huggingface" in str(url) or str(source).lower() == "huggingface":
            token = (body.get("token") or self.config.get("hf_token") or "").strip()
            if body.get("token") is not None and "hf_" in token:
                self.config.update(hf_token=token)
        else:
            token = (body.get("token") or self.config.get("civitai_token") or "").strip()
            if body.get("token") is not None and "hf_" not in token:
                self.config.update(civitai_token=token)

        if not url or not filename:
            return web.json_response({"error": "Не указан URL или имя файла"}, status=400)

        root = self.comfy_root()
        if not root:
            return web.json_response({"error": "ComfyUI не найден"}, status=500)

        dest_dir = root / "ComfyUI" / "models" / "loras" / "MinimaxH3"
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / filename

        meta_name = body.get("name")
        meta_desc = body.get("desc")
        meta_triggers = body.get("triggers")
        if meta_name or meta_triggers:
            all_meta = dict(self.config.get("custom_lora_meta") or {})
            entry = {
                "name": meta_name or filename,
                "desc": meta_desc or "",
                "triggers": meta_triggers or [],
                "category": f"{source} LoRA",
                "source": str(source).lower(),
            }
            all_meta[f"MinimaxH3\\{filename}"] = entry
            all_meta[filename] = entry
            self.config.update(custom_lora_meta=all_meta)

        if civitai_client.CURRENT_DOWNLOAD["active"]:
            return web.json_response({"error": "Уже идёт скачивание другой LoRA"}, status=409)

        civitai_client.CURRENT_DOWNLOAD["active"] = True
        civitai_client.CURRENT_DOWNLOAD["name"] = meta_name or filename

        async def run_dl():
            try:
                await civitai_client.download_file_stream(self.session, url, dest_path, token=token)
            except Exception as e:
                self.note_error(f"Ошибка скачивания LoRA {filename}: {e}")

        asyncio.create_task(run_dl())
        return web.json_response({"ok": True, "file": filename})

    async def h_civitai_download_status(self, _req):
        import civitai_client
        st = dict(civitai_client.CURRENT_DOWNLOAD)
        st.pop("task", None)
        return web.json_response(st)

    async def h_civitai_download_cancel(self, _req):
        import civitai_client
        civitai_client.CURRENT_DOWNLOAD["active"] = False
        civitai_client.CURRENT_DOWNLOAD["status"] = "idle"
        return web.json_response({"ok": True})

    async def h_loras_custom_set(self, req):
        items = []
        for x in (await req.json()).get("items", [])[:10]:
            try:
                strength = max(-4.0, min(4.0, float(x.get("strength", 0.8))))
            except (TypeError, ValueError):
                strength = 0.8
            if isinstance(x.get("file"), str) and x["file"]:
                items.append({"file": x["file"], "strength": strength, "on": bool(x.get("on", True))})
        self.config.update(custom_loras=items)
        return web.json_response({"ok": True})

    def optional_loras(self, form, pairs):
        """(file name, strength) for the optional LoRAs the user switched on and this ComfyUI has."""
        missing = {m["role"] for m in self.missing}
        out = []
        for key, role in pairs:
            v = lora_strength(form, key, None)
            if v is not None and role not in missing and role in self.model_names:
                out.append((self.model_names[role], v))
        return out

    async def h_generate_refs(self, req):
        await self.need_models("refs")
        form, prompt, aspect, quality, seed, images, videos, audios, seconds = await self.read_ref_form(req, 9)
        steps = int(form.get("steps", 6)) if form.get("steps", "6").isdigit() else 6
        steps = min(8, max(4, steps))
        ref_size = form.get("ref_size") if form.get("ref_size") in ("match", "max") else "max"
        extra = self.optional_loras(form, (("realism", "realism_lora"), ("motion", "motion_repair_lora"))) + self.custom_loras(form)
        width, height = wf.resolution(aspect, quality, form)
        pika = None
        if form.get("accel") == "pika":
            if "pika_lora" in {m["role"] for m in self.missing}:
                raise web.HTTPBadRequest(text="Файл Pika (pika_h3_flex_turbo_8step_hd_ref2va) не найден в ComfyUI. "
                                              "Скачайте его в разделе «Модели» или выберите прежнее ускорение.")
            pika, steps = self.model_names["pika_lora"], max(6, steps)
        graph, stages = wf.ref_graph(self.model_names, wf.lipsync_prompt(prompt, len(images), len(videos)), width, height, seconds, seed, images, videos, audios,
                                     steps, ref_size, extra, pika)
        sub = f"{len(images)} карт." + (f", {len(videos)} видео" if videos else "") + (f", {len(audios)} аудио" if audios else "") \
            + f" · {steps} шагов" + (" · Pika" if pika else "")
        return await self.queue(graph, stages, {
            "mode": "refs", "sub": sub, "prompt": prompt, "aspect": aspect, "q": quality, "len": seconds,
            "res": f"{width}×{height}", "seed": seed, "steps": steps, "ref_size": ref_size})

    async def long_source(self, item_id, uploaded, first_sec):
        """A video the user already has (from the library, or a file from the computer) to continue from: put it into ComfyUI and
        find out how many frames it has. Returns (file name in ComfyUI input, frames)."""
        if uploaded:
            name = uploaded   # read_form already uploaded it to ComfyUI
        else:
            it = self.library.get(item_id)
            if not it or it.get("kind") == "image":
                raise web.HTTPBadRequest(text="Видео для продолжения не найдено в Библиотеке")
            data = await self.item_bytes(it)
            fd = aiohttp.FormData()
            nm = f"h3s_c_{uuid.uuid4().hex[:10]}.{it.get('ext') or 'mp4'}"
            fd.add_field("image", data, filename=nm, content_type="video/mp4")
            fd.add_field("type", "input")
            status, res = await self.comfy_json("POST", "/upload/image", data=fd)
            if status != 200:
                raise web.HTTPBadGateway(text="Не удалось передать видео в ComfyUI")
            name = res["name"]
        # how long is it: a tiny job that only opens the video and reports its number of frames
        probe = {"L": {"class_type": "LoadVideo", "inputs": {"file": name}},
                 "G": {"class_type": "GetVideoComponents", "inputs": {"video": ["L", 0]}},
                 "Z": {"class_type": "GetImageSize", "inputs": {"image": ["G", 0]}},
                 "N": {"class_type": "PreviewAny", "inputs": {"source": ["Z", 2]}},
                 "F": {"class_type": "PreviewAny", "inputs": {"source": ["G", 2]}}}
        status, res = await self.comfy_json("POST", "/prompt", json={"prompt": probe, "client_id": CLIENT_ID})
        if status != 200:
            raise web.HTTPBadRequest(text="ComfyUI не смог открыть это видео: " + comfy_error_text(res))
        pid = res["prompt_id"]
        loop = asyncio.get_running_loop()
        end = loop.time() + 180
        meta = None
        while loop.time() < end and not meta:
            await asyncio.sleep(0.7)
            _, hist = await self.comfy_json("GET", f"/history/{pid}")
            e = (hist or {}).get(pid)
            if e:
                if e.get("status", {}).get("status_str") == "error":
                    break
                try:
                    meta = {"frame_count": int(float(e["outputs"]["N"]["text"][0])), "fps": float(e["outputs"]["F"]["text"][0])}
                except (KeyError, IndexError, ValueError):
                    break
        if not meta:
            raise web.HTTPBadRequest(text="Не удалось прочитать это видео. Подойдёт обычный mp4.")
        frames = int(meta["frame_count"])
        need = wf.long_plan([first_sec], cont_first=True)[0]["ov"]
        if frames < need:
            raise web.HTTPBadRequest(text=f"Видео слишком короткое для продолжения: в нём {frames} кадров, нужно не меньше {need} (около 2,5 секунд).")
        return (name, frames)

    async def preview_opts(self):
        """Settings for the animated live preview, or None when ComfyUI has no KJNodes' Model Preview Override."""
        return None   # switched off: the animated preview made generation noticeably slower; the usual single-frame preview is used
        try:
            status, oi = await self.comfy_json("GET", "/object_info/ModelPreviewOverrideKJ")
        except Exception:
            return None
        node = (oi or {}).get("ModelPreviewOverrideKJ") if status == 200 else None
        if not node:
            return None
        tiny = "none"
        try:
            spec = node["input"].get("optional", {}).get("tiny_vae") or []
            options = (spec[1] or {}).get("options") if len(spec) > 1 and isinstance(spec[1], dict) else spec[0]
            tiny = next((o for o in options or [] if o.replace("\\", "/").split("/")[-1].startswith("taeh3")), "none")
        except Exception:
            pass
        return {"tiny_vae": tiny}

    def keep_long_refs(self, run, captured):
        """The reference pictures (and the sound) of a long video project, next to its saved scene states."""
        if not run:
            return
        d = LONG_DIR / run / "refs"
        d.mkdir(parents=True, exist_ok=True)
        for key, (ext, raw) in captured.items():
            if key.startswith(("img", "aud", "simg")) and len(raw) < 80_000_000:
                for old in d.glob(key + ".*"):
                    old.unlink(missing_ok=True)
                (d / (key + ext)).write_bytes(raw)

    async def h_long_project(self, req):
        """The project of a long video (scenes, settings) is kept on disk too, so it survives anything that happens to the browser storage."""
        if req.method == "POST":
            body = await req.json()
            run = re.sub(r"[^A-Za-z0-9]", "", str(body.get("run", "")))[:32]
            if not run or not isinstance(body.get("project"), dict):
                raise web.HTTPBadRequest(text="bad project")
            d = LONG_DIR / run
            d.mkdir(parents=True, exist_ok=True)
            (d / "project.json").write_text(json.dumps(body["project"], ensure_ascii=False), encoding="utf-8")
            return web.json_response({"ok": True})
        run = re.sub(r"[^A-Za-z0-9]", "", req.query.get("run", ""))[:32]
        if not run and req.query.get("last"):   # the most recently saved project
            files = sorted(LONG_DIR.glob("*/project.json"), key=lambda f: f.stat().st_mtime, reverse=True) if LONG_DIR.is_dir() else []
            run = files[0].parent.name if files else ""
        f = LONG_DIR / run / "project.json" if run else None
        if not f or not f.exists():
            return web.json_response({})
        refs = sorted(p.name for p in (LONG_DIR / run / "refs").glob("*")) if (LONG_DIR / run / "refs").is_dir() else []
        return web.json_response({"run": run, "project": json.loads(f.read_text(encoding="utf-8")), "refs": refs})

    async def h_long_ref(self, req):
        run = re.sub(r"[^A-Za-z0-9]", "", req.match_info["run"])[:32]
        name = Path(req.match_info["name"]).name
        f = LONG_DIR / run / "refs" / name
        if not f.is_file():
            raise web.HTTPNotFound()
        return web.FileResponse(f)

    async def h_generate_long(self, req):
        await self.need_models("long")
        self._capture = {}
        try:
            return await self.generate_long(req)
        finally:
            captured, self._capture = self._capture or {}, None
            run = getattr(self, "_long_run", None)
            if run:
                self.keep_long_refs(run, captured)
                self._long_run = None

    async def generate_long(self, req):
        form, files, prompt, aspect, quality, seed = await self.read_form(req, ("img0", "img1", "img2", "img3", "img4", "aud0", "srcvid") + tuple(f"simg{i}" for i in range(12)))
        self._long_run = re.sub(r"[^A-Za-z0-9]", "", form.get("run", ""))[:32]   # the references are kept even if the request is refused later
        try:
            scenes = [(str(x.get("prompt", "")).strip(), float(x.get("sec", 10))) for x in json.loads(form.get("scenes") or "[]")]
        except (ValueError, TypeError, AttributeError):
            scenes = []
        scenes = [(p, min(wf.LONG_MAX_SCENE, max(3.0, sec)), files.get(f"simg{i}")) for i, (p, sec) in enumerate(scenes) if p][:12]
        if not scenes:
            raise web.HTTPBadRequest(text="Напишите хотя бы одну сцену")
        images = [files[f"img{i}"] for i in range(5) if f"img{i}" in files]
        audio = files.get("aud0")
        try:
            start = max(0.0, float(form.get("start") or 0))
        except ValueError:
            start = 0.0
        vocals = None
        if audio and form.get("vocals") == "1":
            if "vocal_model" in {m["role"] for m in self.missing} or "vocal_model" not in self.model_names:
                raise web.HTTPBadRequest(text="Для «Только вокал» нужна модель MelBand. Скачайте её в разделе «Модели».")
            vocals = self.model_names["vocal_model"]
        if "pika_lora" in {m["role"] for m in self.missing}:
            raise web.HTTPBadRequest(text="Файл Pika (pika_h3_flex_turbo_8step_hd_ref2va) не найден. Скачайте его в разделе «Модели».")
        width, height = wf.resolution(aspect, quality, form)
        scenes = [(wf.lipsync_prompt(p, len(images) + (1 if extra else 0)), sec, extra) for p, sec, extra in scenes]
        part = form.get("part", "")
        if part.isdigit():  # one scene at a time: scene k continues from the saved tail of scene k-1
            k = int(part)
            run = re.sub(r"[^A-Za-z0-9]", "", form.get("run", ""))[:32]
            self._long_run = run
            if not run or k >= len(scenes):
                raise web.HTTPBadRequest(text="Неверный номер сцены")
            state, cont = None, None
            if k == 0 and (form.get("src_item") or files.get("srcvid")):
                cont = await self.long_source(form.get("src_item"), files.get("srcvid"), scenes[0][1])
            if k:
                src = LONG_DIR / run / f"state_{k - 1}.latent"
                if not src.exists():
                    raise web.HTTPBadRequest(text=f"Сначала сделайте сцену {k}: следующая продолжает её конец.")
                try:  # the saved tail only fits a scene of the same frame size
                    was = json.loads(src.with_suffix(".json").read_text(encoding="utf-8")).get("res")
                except (OSError, ValueError):
                    was = None
                if was and was != f"{width}×{height}":
                    raise web.HTTPBadRequest(text=f"Сцена {k} сделана в формате {was}, а сейчас выбран {width}×{height}. "
                                                  "Верните прежний формат и качество или переделайте сцену 1 в новом.")
                state = await self.upload_state(src, f"h3s_long_{run}_{k - 1}.latent")
            graph, stages = wf.long_part_graph(self.model_names, scenes, k, width, height, seed, images, audio, start,
                                               self.model_names["pika_lora"], self.custom_loras(form), "match", vocals, state,
                                               await self.preview_opts(), cont, form.get("src_project") == "1")
            plan = wf.long_plan([sc[1] for sc in scenes], cont_first=bool(cont) or form.get("src_project") == "1")
            sub = f"сцена {k + 1} из {len(scenes)}" + (" · музыка" if audio else " · звук от модели")
            return await self.queue(graph, stages, {
                "mode": "long", "sub": sub, "prompt": scenes[k][0], "aspect": aspect, "q": quality,
                "len": round((plan[k]["total"] - plan[k]["ov"]) / wf.FPS, 1), "res": f"{width}×{height}", "seed": seed,
                "long": {"run": run, "part": k, "n": len(scenes)}})
        graph, stages = wf.long_graph(self.model_names, scenes, width, height, seed, images, audio, start,
                                      self.model_names["pika_lora"], self.custom_loras(form), "match", vocals, await self.preview_opts())
        run = re.sub(r"[^A-Za-z0-9]", "", form.get("run", ""))[:32]
        self._long_run = run
        total = wf.long_plan([sc[1] for sc in scenes])[-1]["acc"] / wf.FPS
        sub = f"{len(scenes)} сцен · {len(images)} карт." + (", музыка" if audio else ", звук от модели")
        meta = {"mode": "long", "sub": sub, "prompt": "\n\n".join(sc[0] for sc in scenes), "aspect": aspect, "q": quality,
                "len": round(total, 1), "res": f"{width}×{height}", "seed": seed}
        if run:
            meta["long"] = {"run": run, "part": len(scenes) - 1, "n": len(scenes), "all": True}
        return await self.queue(graph, stages, meta)

    async def h_long_state(self, req):
        """Which scenes of a long video project have a saved continuation point (so the project can go on after a restart)."""
        run = re.sub(r"[^A-Za-z0-9]", "", req.query.get("run", ""))[:32]
        d = LONG_DIR / run
        parts = sorted(int(m.group(1)) for f in d.glob("state_*.latent") if (m := re.match(r"state_(\d+)\.latent$", f.name))) if run and d.is_dir() else []
        return web.json_response({"parts": parts})

    async def item_bytes(self, item):
        """The file of a library item: from disk, or from ComfyUI when its output folder is not known to the app."""
        p = self.file_path(item)
        if p and p.exists():
            return await asyncio.to_thread(p.read_bytes)
        f = item["file"]
        async with self.session.get(self.comfy + "/view", params={
                "filename": f["filename"], "subfolder": f.get("subfolder", ""), "type": f.get("type", "output")}) as r:
            if r.status != 200:
                raise web.HTTPBadRequest(text="Не удалось найти один из клипов: он удалён с диска?")
            return await r.read()

    async def h_montage(self, req):
        """Join library clips in the given order with ComfyUI (no ffmpeg in the app)."""
        form = {}
        async for field in await req.multipart():
            form[field.name] = await field.text()
        try:
            ids = [i for i in json.loads(form.get("ids") or "[]") if isinstance(i, str)][:30]
        except ValueError:
            ids = []
        items = [self.library.get(i) for i in ids]
        if len(items) < 2 or any(not it or it.get("kind") == "image" for it in items):
            raise web.HTTPBadRequest(text="Добавьте хотя бы два видео")
        if self.rh():
            raise web.HTTPBadRequest(text="Монтаж работает только с локальным ComfyUI, не с облаком.")
        names, sizes = [], []
        for it in items:
            data = await self.item_bytes(it)
            name = f"h3s_m_{uuid.uuid4().hex[:10]}.{it.get('ext') or 'mp4'}"
            fd = aiohttp.FormData()
            fd.add_field("image", data, filename=name, content_type="video/mp4")
            fd.add_field("type", "input")
            status, res = await self.comfy_json("POST", "/upload/image", data=fd)
            if status != 200:
                raise web.HTTPBadGateway(text="Не удалось передать клип в ComfyUI")
            names.append(res["name"])
            mm = re.match(r"\s*(\d+)\D+(\d+)", it.get("res") or "")
            sizes.append((int(mm.group(1)), int(mm.group(2))) if mm else None)
        graph, stages = wf.montage_graph(names, sizes)
        total = sum(float(it.get("len") or 0) for it in items)
        return await self.queue(graph, stages, {
            "mode": "montage", "sub": f"{len(items)} клипов", "prompt": "", "aspect": items[0].get("aspect", ""), "q": "",
            "len": round(total, 1), "res": items[0].get("res", ""), "seed": 0}, last_output=True)

    async def h_generate_actor(self, req):
        await self.need_models("actor")
        form, prompt, aspect, quality, seed, images, videos, audios, seconds = await self.read_ref_form(req, 3)
        realism = None
        v = lora_strength(form, "realism", 0.7)  # on by default, as in the original workflow
        missing = {m["role"] for m in self.missing}
        if v is not None and "realism_lora" not in missing and "realism_lora" in self.model_names:
            realism = (self.model_names["realism_lora"], v)
        width, height = wf.resolution(aspect, quality, form)
        graph, stages = wf.actor_graph(self.model_names, wf.lipsync_prompt(prompt, len(images)), width, height, seconds, seed,
                                       images, audios, realism, self.custom_loras(form))
        sub = f"{len(images)} карт." + (f", голос ({len(audios)})" if audios else ", без голоса") + " · 10 шагов"
        return await self.queue(graph, stages, {
            "mode": "actor", "sub": sub, "prompt": prompt, "aspect": aspect, "q": quality, "len": seconds,
            "res": f"{width}×{height}", "seed": seed})

    async def h_generate_swap(self, req):
        """Preset «Замена персонажа»: one video + up to two character pictures, the app writes the prompt."""
        await self.need_models("swap")
        form, prompt, aspect, quality, seed, images, videos, audios, seconds = await self.read_ref_form(req, 2)
        if not videos:
            raise web.HTTPBadRequest(text="Добавьте видео, в котором нужно заменить человека")
        if not images:
            raise web.HTTPBadRequest(text="Добавьте картинку персонажа, на которого заменяем")
        shots = []
        if form.get("shots"):   # several plans (shots): every one is swapped on its own, then they are joined
            try:
                for x in json.loads(form["shots"])[:12]:
                    s0, e0 = float(x["s"]), float(x["e"])
                    if e0 - s0 < 0.5:
                        continue
                    shots.append({"s": s0, "e": e0, "pic": int(x.get("pic", 0)), "who": str(x.get("who") or "").strip(),
                                  "a_end": float(x["a_end"]) if x.get("a_end") not in (None, "") else None})
            except (ValueError, KeyError, TypeError):
                raise web.HTTPBadRequest(text="Не удалось разобрать список планов")
            if any(0 <= x["pic"] and x["e"] - x["s"] < 2 for x in shots):
                raise web.HTTPBadRequest(text="Каждый план для замены должен быть не короче 2 секунд: объедините короткие планы с соседними")
        whos = [(form.get(f"who{i}") or (form.get("who") if i == 0 else "") or "").strip() for i in range(len(images))]
        try:
            cuts = [float(x) for x in (form.get("cuts") or "").replace(";", ",").split(",") if x.strip()]
        except ValueError:
            cuts = []
        text = wf.swap_prompt(whos, (form.get("extra") or "").strip(), cuts, form.get("prompt_style") != "plain")
        if form.get("raw_prompt", "").strip():   # a prompt written by hand in the MiniMax format (tests, experts): used as it is
            text = form["raw_prompt"].strip()
        pika = None
        if form.get("accel") == "pika":
            if "pika_lora" in {m["role"] for m in self.missing}:
                raise web.HTTPBadRequest(text="Файл Pika (pika_h3_flex_turbo_8step_hd_ref2va) не найден в ComfyUI. "
                                              "Скачайте его в разделе «Модели» или выберите обычное ускорение.")
            pika = self.model_names["pika_lora"]
        names = self.model_names
        if form.get("unet"):   # another base model for this job (tests): only a file this ComfyUI really has
            try:
                _, have = await self.comfy_json("GET", "/models/diffusion_models")
            except Exception:
                have = []
            pick = next((f for f in (have or []) if f.replace("/", "\\") == form["unet"].replace("/", "\\")), None)
            if not pick:
                raise web.HTTPBadRequest(text="Такой модели нет в ComfyUI: " + form["unet"])
            names = {**self.model_names, "lipsync_unet": pick}
        width, height = wf.resolution(aspect, quality, form)
        seconds = min(15.0, max(3.0, seconds))
        loras = self.optional_loras(form, (("realism", "realism_lora"), ("motion", "motion_repair_lora"))) + self.custom_loras(form)
        if shots:
            snd = form.get("sound") if form.get("sound") in ("original", "lock") else "original"
            graph, stages = wf.swap_shots_graph(names, shots, width, height, seed, images, videos[0],
                                                (form.get("extra") or "").strip(), pika, loras, snd,
                                                form.get("small_ref") == "1")
            seconds = max(x["e"] for x in shots) - min(x["s"] for x in shots)
        else:
            snd = form.get("sound") if form.get("sound") in ("original", "lock", "model") else "original"
            graph, stages = wf.swap_graph(names, text, width, height, seconds, seed, images, videos[0], loras, pika, snd,
                                             small_ref=form.get("small_ref") == "1")
        return await self.queue(graph, stages, {
            "mode": "swap", "sub": "Замена персонажа" + (f", планов: {len(shots)}" if shots else (f" ({len(images)})" if len(images) > 1 else "")) + (" · Pika" if pika else ""),
            "prompt": prompt or whos[0], "aspect": aspect, "q": quality, "len": seconds,
            "res": f"{width}×{height}", "seed": seed})

    async def action_setup(self):
        """What the original action workflow needs and what this ComfyUI has."""
        if not self.model_names:
            await self.resolve_models()
        _, oi = await self.comfy_json("GET", "/object_info/SelfLiftH3Sampler")
        _, oi2 = await self.comfy_json("GET", "/object_info/H3SigmaRefiner")
        _, loras = await self.comfy_json("GET", "/models/loras")
        dmad = next((n for n in loras or [] if catalog.ACTION_DMAD_LORA in n), None)
        missing = {m["role"] for m in self.missing}
        have_up = "action_upscaler" not in missing
        nodes = bool(oi) and bool(oi2)
        return {
            "faithful": nodes and have_up,
            "selflift": bool(oi), "refiner": bool(oi2), "refiner_def": (oi2 or {}).get("H3SigmaRefiner"),
            "upscaler": have_up, "dmad": dmad,
            "speed": None if "speed_slider_lora" in missing else self.model_names.get("speed_slider_lora"),
        }

    async def h_action_info(self, _req):
        try:
            s = await self.action_setup()
        except aiohttp.ClientError:
            return web.json_response({"error": "ComfyUI не отвечает"}, status=503)
        s.pop("refiner_def", None)
        return web.json_response(s)

    async def h_generate_action(self, req):
        await self.need_models("action")
        form, prompt, aspect, quality, seed, images, videos, audios, seconds = await self.read_ref_form(req, 6)
        if self.rh():
            # on RunningHub only built-in nodes are assumed; LoRAs are used if the user's RH workflow has them
            rh_files = {n.replace("\\", "/").split("/")[-1].lower(): n for n in (getattr(self, "rh_models", None) or [])}
            speed = rh_files.get(catalog.VARIANTS["speed_slider_lora"]["v1.1"]["file"].lower())
            dmad = next((n for b, n in rh_files.items() if catalog.ACTION_DMAD_LORA in b), None)
            s = {"faithful": False, "dmad": dmad, "speed": speed}
        else:
            s = await self.action_setup()
        loras = []
        for key, name, default in (("dmad", s["dmad"], 0.8), ("speed", s["speed"], 4.0)):
            v = lora_strength(form, key, default)  # the user can switch them off or change the strength
            if name and v is not None:
                loras.append((name, v))
        loras += self.custom_loras(form)
        names = dict(self.model_names)
        width, height = wf.resolution(aspect, quality, form)
        graph, stages = wf.action_graph(names, wf.lipsync_prompt(prompt, len(images)), width, height, seconds, seed, images, audios,
                                        s["faithful"], loras)
        if s["faithful"]:
            # the refiner's input names come from the installed node; values as in the original workflow
            req_inputs = [k for k in (s["refiner_def"]["input"].get("required") or {}) if k != "sigmas"]
            graph["13"]["inputs"] = {"sigmas": ["10", 0], **dict(zip(req_inputs, [1, 0.7, 0.0, "cosine"]))}
        sub = ("точная копия workflow" if s["faithful"] else "упрощённая схема") + f" · {len(images)} карт."
        return await self.queue(graph, stages, {
            "mode": "action", "sub": sub, "prompt": prompt, "aspect": aspect, "q": quality, "len": seconds,
            "res": f"{width}×{height}", "seed": seed, "faithful": s["faithful"]})

    # ---------- audio: voice / music separation (MelBand RoFormer through ComfyUI) ----------

    async def separation_ready(self):
        """(ok, reason) - is the node pack there and is the model visible to ComfyUI."""
        if self.rh():
            return False, "Отделение голоса работает на этом компьютере, а сейчас включено облако."
        try:
            _, node = await self.comfy_json("GET", "/object_info/MelBandRoFormerSampler")
        except aiohttp.ClientError:
            return False, "ComfyUI не запущен."
        if not node:
            return False, "В ComfyUI нет ноды MelBandRoFormer (ComfyUI-MelBandRoFormer)."
        if not self.model_names:
            await self.resolve_models()
        if "vocal_model" in {m["role"] for m in self.missing}:
            return False, "Не найдена модель MelBandRoformer_fp16.safetensors (папка diffusion_models)."
        return True, ""

    async def h_audio_info(self, _req):
        ok, reason = await self.separation_ready()
        return web.json_response({"available": ok, "reason": reason})

    async def h_audio_separate(self, req):
        """WAV in -> the voice only (FLAC) out. Blocks until ComfyUI is done (usually well under a minute)."""
        ok, reason = await self.separation_ready()
        if not ok:
            return web.json_response({"error": reason}, status=400)
        name = None
        async for field in await req.multipart():
            if field.name == "file" and field.filename:
                name = await self.upload_input(field)
        if not name:
            return web.json_response({"error": "Нет аудио"}, status=400)
        await self.stop_llm()  # the agent's model would compete for video memory
        graph = {
            "1": {"class_type": "LoadAudio", "inputs": {"audio": name}},
            "2": {"class_type": "MelBandRoFormerModelLoader", "inputs": {"model_name": self.model_names["vocal_model"]}},
            "3": {"class_type": "MelBandRoFormerSampler", "inputs": {"model": ["2", 0], "audio": ["1", 0]}},
            "4": {"class_type": "SaveAudio", "inputs": {"audio": ["3", 0], "filename_prefix": "H3Workspace/audio/vocals"}},
        }
        status, res = await self.comfy_json("POST", "/prompt", json={"prompt": graph, "client_id": CLIENT_ID})
        if status != 200:
            return web.json_response({"error": comfy_error_text(res)}, status=400)
        pid = res["prompt_id"]
        loop = asyncio.get_running_loop()
        end = loop.time() + 600
        while loop.time() < end:
            await asyncio.sleep(1.5)
            _, hist = await self.comfy_json("GET", f"/history/{pid}")
            entry = (hist or {}).get(pid)
            if not entry:
                continue
            st = entry.get("status", {})
            if st.get("status_str") == "error":
                msgs = [m[1].get("exception_message", "") for m in st.get("messages", []) if m[0] == "execution_error"]
                return web.json_response({"error": "Не удалось отделить голос: " + (msgs[0].strip() if msgs else "ошибка ComfyUI")}, status=500)
            audio = next((f for out in entry.get("outputs", {}).values() for f in out.get("audio", [])), None)
            if audio:
                async with self.session.get(self.comfy + "/view", params={
                        "filename": audio["filename"], "subfolder": audio.get("subfolder", ""), "type": audio.get("type", "output")}) as r:
                    data = await r.read()
                return web.Response(body=data, content_type="audio/flac")
        return web.json_response({"error": "Отделение голоса заняло слишком много времени"}, status=504)

    async def h_interrupt(self, req):
        pid = (await req.json()).get("prompt_id")
        job = self.jobs.get(pid)
        if job and job.get("task"):  # cloud task
            self.jobs.pop(pid, None)
            rh = self.rh()
            if rh:
                try:
                    await rh.cancel(job["task"])
                except cloud.CloudError:
                    pass
            await self.broadcast({"type": "interrupted", "prompt_id": pid})
        elif job and job["state"] == "queued":
            await self.comfy_json("POST", "/queue", json={"delete": [pid]})
            self.jobs.pop(pid, None)
            await self.broadcast({"type": "interrupted", "prompt_id": pid})
        else:
            await self.comfy_json("POST", "/interrupt", json={"prompt_id": pid})
            task = asyncio.create_task(self.interrupt_watch(pid))
            self._bg.add(task)
            task.add_done_callback(self._bg.discard)
        return web.json_response({"ok": True})

    async def interrupt_watch(self, pid, wait=15):
        """ComfyUI sometimes ignores «Cancel» (stuck on a heavy step). If the job is still there after `wait` seconds:
        managed engine - restart it; the user's own ComfyUI - only tell the user."""
        await asyncio.sleep(wait)
        if pid not in self.jobs:
            return
        if not self.process:
            text = ("ComfyUI не отвечает на «Отмена» уже больше 15 секунд. Это ваш собственный ComfyUI: "
                    "закройте его окно и запустите заново.")
            self.note_error(text)
            await self.broadcast({"type": "notice", "text": text})
            return
        text = "ComfyUI не отвечает на «Отмена». Перезапускаем движок, это займёт около минуты."
        self.note_error(text)
        await self.broadcast({"type": "notice", "text": text})
        try:
            await asyncio.to_thread(self.process.stop)
            await asyncio.to_thread(self.process.start)
            await self.wait_comfy()
        except Exception as e:
            self.note_error(f"Не удалось перезапустить движок после отмены: {e}")
        # the restarted engine has an empty queue: nothing of what was running or waiting is there any more
        for jid in [j for j, job in self.jobs.items() if not job.get("task")]:
            self.jobs.pop(jid, None)
            await self.broadcast({"type": "interrupted", "prompt_id": jid})

    async def h_library(self, _req):
        return web.json_response([self.public(x) for x in self.library.items])

    async def h_library_patch(self, req):
        item = self.library.get(req.match_info["id"])
        if not item:
            raise web.HTTPNotFound()
        body = await req.json()
        if "fav" in body:
            item["fav"] = bool(body["fav"])
            self.library.save()
        return web.json_response(self.public(item))

    async def h_library_delete(self, req):
        item = self.library.get(req.match_info["id"])
        if not item:
            raise web.HTTPNotFound()
        recycled = False
        for path in {p for p in (self.file_path(item), Path(item["local"]) if item.get("local") else None) if p}:
            if path.exists():   # the original and the program's own copy both go to the Recycle Bin
                ok = await asyncio.to_thread(send_to_recycle_bin, path)
                recycled = recycled or ok
                if not ok:
                    return web.json_response({"error": "Не удалось переместить файл в корзину"}, status=500)
        self.library.remove(item["id"])
        return web.json_response({"ok": True, "recycled": recycled})

    async def h_results_get(self, _req):
        items = self.library.items
        lost = sum(1 for x in items if (p := self.file_path(x)) and not p.exists())
        nocopy = sum(1 for x in items if not (x.get("local") and Path(x["local"]).exists()) and (p := self.file_path(x)) and p.exists())
        return web.json_response({"dir": str(self.results_dir()), "keep": self.keep_copies(), "lost": lost, "nocopy": nocopy,
                                  "default": str(self.default_results_dir())})

    async def h_results_set(self, req):
        body = await req.json()
        if "keep" in body:
            self.config.update(keep_copies=bool(body["keep"]))
        d = (body.get("dir") or "").strip().strip('"')
        if d:
            try:
                Path(d).mkdir(parents=True, exist_ok=True)
            except OSError:
                return web.json_response({"error": "Не получается создать папку. Проверьте путь и диск"}, status=400)
            self.config.update(results_dir=d)
        elif body.get("reset"):
            self.config.update(results_dir=None)
        return await self.h_results_get(req)

    async def h_results_backfill(self, req):
        """Make copies of the files that are still in ComfyUI's output folder but have no copy of their own yet."""
        n = 0
        for x in list(self.library.items):
            p = self.file_path(x)
            if p and p.exists() and not (x.get("local") and Path(x["local"]).exists()):
                n += 1 if await self.archive_item(x) else 0
        return web.json_response({"copied": n, **json.loads((await self.h_results_get(req)).text)})

    async def h_results_clean(self, req):
        """Remove the Library records whose files are gone from the disk (nothing on the disk is touched)."""
        gone = [x["id"] for x in self.library.items if (p := self.file_path(x)) and not p.exists()]
        for i in gone:
            self.library.remove(i)
        return web.json_response({"removed": len(gone), **json.loads((await self.h_results_get(req)).text)})

    async def h_file(self, req):
        item = self.library.get(req.match_info["id"])
        if not item:
            raise web.HTTPNotFound()
        f = item["file"]
        headers = {}
        if req.query.get("download"):
            headers["Content-Disposition"] = f'attachment; filename="{f["filename"]}"'
        path = self.file_path(item)
        if path and path.exists():
            return web.FileResponse(path, headers=headers)
        # output folder unknown: stream through ComfyUI
        params = {"filename": f["filename"], "subfolder": f.get("subfolder", ""), "type": f.get("type", "output")}
        async with self.session.get(self.comfy + "/view", params=params) as r:
            if r.status != 200:
                raise web.HTTPNotFound(text="Файл не найден")
            resp = web.StreamResponse(headers={"Content-Type": r.headers.get("Content-Type", "video/mp4"), **headers})
            await resp.prepare(req)
            async for chunk in r.content.iter_chunked(1 << 16):
                await resp.write(chunk)
            return resp

    async def h_events(self, req):
        ws = web.WebSocketResponse(heartbeat=20)
        await ws.prepare(req)
        self.browsers.add(ws)
        await ws.send_str(json.dumps({"type": "engine", "online": self.comfy_online}))
        try:
            async for _ in ws:
                pass
        finally:
            self.browsers.discard(ws)
        return ws

    # ---------- agent ----------

    def skill_public(self, s):
        return {k: s[k] for k in ("id", "name", "description", "builtin", "mode", "keywords")} | \
               {"refs": [r.name for r in s["refs"]]}

    async def h_agent_status(self, _req):
        variant = await self.llm_variant()
        v = ag.LLM_VARIANTS[variant]
        custom = self.config.get("llm_custom")
        vram = (self.hw or {}).get("vram_gb") or 0
        candidates = []
        if custom and custom.get("dir") and Path(custom["dir"]).is_dir():
            _, _, found = await asyncio.to_thread(ag.find_gguf, custom["dir"])
            candidates = [{"path": str(p), "name": p.name, "gb": round(p.stat().st_size / 1e9, 1),
                           "fits": ag.fits(p, vram), "current": str(p) == custom["model"]} for p in found]
        options = []
        for k, ov in ag.LLM_VARIANTS.items():
            mp, pp = ag.variant_files(self.llm.root, ov)
            options.append({"key": k, "title": ov["title"], "uncensored": bool(ov.get("uncensored")),
                            "size": ov["model_size"] + ov["mmproj_size"], "have": mp.exists() and pp.exists(),
                            "current": k == variant and not custom, "fits": vram >= ov.get("min_vram", 0),
                            "min_vram": ov.get("min_vram", 0)})
        return web.json_response({
            "variants": options,
            "my_model": ({"title": Path((custom or self.config.get("llm_custom_saved") or {}).get("model", "")).stem,
                          "current": bool(custom)} if (custom or (self.config.get("llm_custom_saved") or {}).get("model")) else None),
            "variant": variant, "title": Path(custom["model"]).stem if custom else v["title"],
            "size": 37e6 if custom else v["model_size"] + v["mmproj_size"], "candidates": candidates, "vram": vram,
            "custom": custom, "vision": self.llm.vision(variant), "has_model": self.llm.has_model(variant),
            "speech_ready": self.speech_status()["ready"], "speech_size": speech.MODEL_SIZE + speech.WHISPER_ZIP_SIZE,
            "speech": self.speech_status(),
            "installed": self.llm.installed(variant), "running": self.llm.alive(),
            "installing": bool(self.agent_installer), "dir": str(self.llm.root), "managed": self.config.get("engine") == "managed",
            "dir_chosen": bool(self.config.get("llm_dir")) or self.config.get("engine") == "managed",
            "skills": [self.skill_public(s) for s in self.skills.list()],
        })

    async def h_agent_install(self, req):
        if self.agent_installer:
            return web.json_response({"error": "Уже скачивается"}, status=409)
        try:
            body = await req.json()
        except Exception:
            body = {}
        wanted = (body.get("dir") or "").strip()
        req_variant = body.get("variant")
        if req_variant and req_variant in ag.LLM_VARIANTS:
            self.config.update(llm_variant=req_variant, llm_custom=None)
            self.llm.custom = None
            variant = req_variant
        else:
            variant = await self.llm_variant()
        if self.config.get("engine") != "managed" and not self.llm.installed(variant):
            # with the user's own ComfyUI nothing goes anywhere the user did not name
            if not wanted and not self.config.get("llm_dir"):
                return web.json_response({"error": "Сначала укажите папку, куда скачать помощника"}, status=400)
            if wanted:
                target = Path(wanted) if Path(wanted).name == "MinimaxH3_assistant" else Path(wanted) / "MinimaxH3_assistant"
                try:
                    target.mkdir(parents=True, exist_ok=True)
                except OSError:
                    return web.json_response({"error": "Не получается создать папку. Проверьте путь и диск"}, status=400)
                self.config.update(llm_dir=str(target))
                await self.llm_variant()

        async def emit(ev):
            await self.broadcast(dict(ev, type="agent_setup"))

        self.agent_installer = engine.Installer(ROOT, self.session, emit)

        async def run():
            try:
                await ag.install(self.agent_installer, self.llm.root, variant,
                                 runtime_only=bool(self.config.get("llm_custom")),
                                 whisper_dir=self.config.get("whisper_dir"))
                await emit({"step": "agent", "state": "done", "label": "Модель агента установлена"})
            except engine.Cancelled:
                await emit({"step": "agent", "state": "cancelled", "label": "Скачивание остановлено"})
            except Exception as e:
                await emit({"step": "agent", "state": "error", "label": str(e)})
            finally:
                self.agent_installer = None

        asyncio.create_task(run())
        return web.json_response({"ok": True})

    def speech_status(self):
        s = speech.locate(self.llm.root, self.config.get("whisper_dir"))
        return {"ready": s["ready"], "missing": s["missing"], "custom": s["custom"],
                "exe": str(s["exe"]) if s["exe"] else None, "model": s["model"].name if s["model"] else None}

    async def h_speech_folder(self, req):
        """Use whisper files the user already has (or go back to the app's own copy)."""
        body = await req.json()
        await self.llm_variant()
        if body.get("reset"):
            self.config.update(whisper_dir=None)
        else:
            folder = (body.get("dir") or "").strip().strip('"')
            if not Path(folder).is_dir():
                return web.json_response({"error": f"Папка не найдена: {folder}"}, status=400)
            found = await asyncio.to_thread(speech.locate, self.llm.root, folder)
            if not (speech._find_exe(folder) or speech._find_model(folder)):
                return web.json_response({"error": "В папке нет ни whisper-cli.exe, ни модели ggml-*.bin "
                                                   "(нужна многоязычная модель, не «.en»)."}, status=400)
            self.config.update(whisper_dir=folder)
        return await self.h_agent_status(req)

    async def h_transcribe(self, req):
        """Words from an audio clip (WAV from the browser) - lyrics for lipsync prompts and voice input."""
        await self.llm_variant()  # keeps llm.root in sync with the install folder
        if not self.speech_status()["ready"]:
            return web.json_response({"error": "Распознавание речи ещё не скачано (карточка «Модель» в разделе «Агент»)."},
                                     status=400)
        wav, lang = None, "auto"
        async for field in await req.multipart():
            if field.name == "file":
                wav = await field.read()
            elif field.name == "lang":
                lang = (await field.text()).strip() or "auto"
        if not wav:
            return web.json_response({"error": "Нет аудио"}, status=400)
        try:
            text, code, name = await speech.transcribe(self.llm.root, wav, lang, self.config.get("whisper_dir"))
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)
        return web.json_response({"text": text, "lang": code, "lang_name": name})

    async def h_agent_variant(self, req):
        """Choose which of the app's own models the agent uses (Gemma / Qwen, with or without refusals)."""
        key = (await req.json()).get("variant")
        if key not in ag.LLM_VARIANTS:
            return web.json_response({"error": "Неизвестная модель"}, status=400)
        cur = self.config.get("llm_custom")
        # the user's own model is not lost: it stays as «Моя модель» and can be switched back on
        self.config.update(llm_variant=key, llm_custom=None, **({"llm_custom_saved": cur} if cur else {}))
        await self.stop_llm()
        return await self.h_agent_status(req)

    async def h_agent_model(self, req):
        """Use a GGUF model the user already has (or go back to the app's own Gemma)."""
        body = await req.json()
        if body.get("reset"):
            cur = self.config.get("llm_custom")
            self.config.update(llm_custom=None, **({"llm_custom_saved": cur} if cur else {}))
        elif body.get("restore"):
            saved = self.config.get("llm_custom_saved")
            if not saved or not Path(saved.get("model", "")).exists():
                return web.json_response({"error": "Файл вашей модели не найден"}, status=400)
            self.config.update(llm_custom=saved)
        else:
            folder = (body.get("dir") or "").strip().strip('"')
            if not Path(folder).is_dir():
                return web.json_response({"error": f"Папка не найдена: {folder}"}, status=400)
            if self.hw is None:
                self.hw = await asyncio.to_thread(engine.detect_hardware)
            model, proj, models = await asyncio.to_thread(ag.find_gguf, folder, body.get("model"), self.hw.get("vram_gb"))
            if not model:
                return web.json_response({"error": "В папке нет моделей .gguf. Нужна обычная Gemma в формате GGUF "
                                                   "(версия для LTX в text_encoders не подойдёт)."}, status=400)
            self.config.update(llm_custom={"model": str(model), "mmproj": str(proj) if proj else None, "dir": folder})
        await self.stop_llm()
        return await self.h_agent_status(req)

    async def h_agent_install_cancel(self, _req):
        if self.agent_installer:
            self.agent_installer.cancelled = True
        return web.json_response({"ok": True})

    async def h_agent_chat(self, req):
        body = await req.json()
        messages = body.get("messages") or []
        target = body.get("target") if body.get("target") in ag.AGENT_TARGETS else "video"
        ctx = body.get("ctx") or {}
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"})
        await resp.prepare(req)

        async def send(obj):
            await resp.write(f"data: {ag.dumps(obj)}\n\n".encode())

        try:
            if self.jobs:
                raise RuntimeError("Сейчас идёт генерация видео — агент запустится, когда она закончится.")
            variant = await self.llm_variant()
            if not self.llm.installed(variant):
                raise RuntimeError("Модель агента ещё не скачана.")
            if not self.llm.alive():
                await send({"type": "status", "text": "Освобождаем видеопамять…"})
                try:
                    await self.comfy_json("POST", "/free", json={"unload_models": True, "free_memory": True})
                except Exception:
                    pass
                await send({"type": "status", "text": f'Загружаем {ag.LLM_VARIANTS[variant]["title"]}… (до минуты)'})
                self.llm.start(variant)
                await ag.wait_ready(self.session, self.llm)

            if not self.llm.vision(variant) and any(m.get("images") for m in messages):
                # without a vision projector the model cannot take pictures
                messages = [dict(m, images=None) for m in messages]
                await send({"type": "status", "text": "У этой модели нет файла «зрения» (mmproj) — картинки она не увидит"})
            # frames cut out of a video are the last pictures of a message and are not hero pictures (Picture N)
            ctx["images"] = sum(max(0, len(m.get("images") or []) - int(m.get("frames") or 0)) for m in messages if m["role"] == "user")
            active = self.skills.active(ag.chat_text(messages), target)
            if target == "actor" and not any(s["id"] == "akterskaya-igra" for s in active):
                # this mode is acting-first: its skill is always on here
                active += [s for s in self.skills.list() if s["id"] == "akterskaya-igra" and s["mode"] != "off"]
            await send({"type": "skills", "names": [s["name"] for s in active]})
            system = ag.system_prompt("\n\n".join(self.skills.text(s) for s in active), target, ctx)
            await send({"type": "status", "text": "Читаю скиллы и думаю…"})

            thinking = False
            payload = {"messages": [{"role": "system", "content": system}] + ag.to_openai(messages),
                       "stream": True, "temperature": 0.6, "cache_prompt": True}
            if body.get("think") != "deep":
                # "fast" (default): no hidden reasoning before the answer. On the user's machine a light request took 17 s with it
                # (about 2000 characters of thoughts) and 1.5 s without; the long-video rules took minutes with it
                payload["chat_template_kwargs"] = {"enable_thinking": False}
                payload["thinking_budget_tokens"] = 0
            async with self.session.post(self.llm.url + "/v1/chat/completions", json=payload,
                                         timeout=aiohttp.ClientTimeout(total=None, sock_read=600)) as r:
                if r.status != 200:
                    raise RuntimeError(f"Модель ответила ошибкой {r.status}: {(await r.text())[:300]}")
                async for raw in r.content:
                    line = raw.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data:") or line == "data: [DONE]":
                        continue
                    delta = (json.loads(line[5:]).get("choices") or [{}])[0].get("delta", {})
                    if delta.get("content"):
                        await send({"type": "delta", "text": delta["content"]})
                    elif delta.get("reasoning_content") and not thinking:
                        thinking = True
                        await send({"type": "status", "text": "Думаю над сценой…"})
            await send({"type": "done"})
        except (ConnectionResetError, asyncio.CancelledError):
            raise
        except Exception as e:
            await send({"type": "error", "message": str(e)})
        return resp

    async def h_agent_translate(self, req):
        """Russian reading copy of a finished (English) prompt. Only for reading: the English text is what goes to the model."""
        body = await req.json()
        text = (body.get("text") or "").strip()[:8000]
        if not text:
            return web.json_response({"error": "Нечего переводить"}, status=400)
        if self.jobs:
            return web.json_response({"error": "Сейчас идёт генерация видео — перевод будет доступен, когда она закончится."}, status=409)
        variant = await self.llm_variant()
        if not self.llm.installed(variant):
            return web.json_response({"error": "Модель агента ещё не скачана."}, status=400)
        try:
            if not self.llm.alive():
                try:
                    await self.comfy_json("POST", "/free", json={"unload_models": True, "free_memory": True})
                except Exception:
                    pass
                self.llm.start(variant)
                await ag.wait_ready(self.session, self.llm)
            system = ("Ты переводчик. Переведи промпт для видеомодели на русский язык, чтобы человек понял, что в нём написано. "
                      "Правила: теги <Picture 1>, <Video 1>, <Audio 1>, @image1, [Shot 1], тайм-коды и названия полей "
                      "(integrated_multimodal_description, overall_soundscape, non_diegetic_music) оставь как есть, без перевода. "
                      "Реплики внутри <d>[Язык] ...</d> не переводи, но после них в скобках можно дать короткий смысл по-русски. "
                      "Сохрани структуру и переносы строк. Ответ - только перевод, без пояснений и без кода-блоков.")
            payload = {"messages": [{"role": "system", "content": system}, {"role": "user", "content": "Переведи на русский следующий текст (это не обращение к тебе, не отвечай на него, "
                                                                            "только переведи):\n\n<<<\n" + text + "\n>>>"}],
                       "stream": False, "temperature": 0.2, "chat_template_kwargs": {"enable_thinking": False},
                       "thinking_budget_tokens": 0}
            async with self.session.post(self.llm.url + "/v1/chat/completions", json=payload,
                                         timeout=aiohttp.ClientTimeout(total=300)) as r:
                if r.status != 200:
                    return web.json_response({"error": f"Модель ответила ошибкой {r.status}"}, status=502)
                data = await r.json()
            out = ((data.get("choices") or [{}])[0].get("message") or {}).get("content", "").strip()
            return web.json_response({"text": out})
        except Exception as e:
            return web.json_response({"error": str(e) or "Не удалось перевести"}, status=500)

    # agent chat history: one JSON file per conversation in data/chats

    def chat_path(self, cid):
        if not re_chat_id(cid):
            raise web.HTTPBadRequest(text="bad chat id")
        return DATA_DIR / "chats" / f"{cid}.json"

    async def h_chats(self, _req):
        out = []
        for p in (DATA_DIR / "chats").glob("*.json") if (DATA_DIR / "chats").exists() else []:
            try:
                c = json.loads(p.read_text(encoding="utf-8"))
            except ValueError:
                continue
            out.append({"id": p.stem, "title": c.get("title", ""), "updated": c.get("updated", ""),
                        "target": c.get("target", "video"), "count": len(c.get("messages", []))})
        return web.json_response(sorted(out, key=lambda c: c["updated"], reverse=True))

    async def h_chat_get(self, req):
        p = self.chat_path(req.match_info["id"])
        if not p.exists():
            raise web.HTTPNotFound()
        return web.Response(text=p.read_text(encoding="utf-8"), content_type="application/json")

    async def h_chat_put(self, req):
        p = self.chat_path(req.match_info["id"])
        body = await req.json()
        msgs = [{k: m.get(k) for k in ("role", "content", "images", "skills", "error", "nframes") if m.get(k)}
                for m in body.get("messages") or [] if m.get("role") in ("user", "assistant")]
        first = next((m.get("content", "") for m in msgs if m["role"] == "user"), "") or "Без названия"
        data = {"title": (body.get("title") or first).strip().replace("\n", " ")[:80], "target": body.get("target", "video"),
                "updated": datetime.now().isoformat(timespec="seconds"), "messages": msgs}
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        return web.json_response({"ok": True, "title": data["title"]})

    async def h_chat_delete(self, req):
        self.chat_path(req.match_info["id"]).unlink(missing_ok=True)
        return web.json_response({"ok": True})

    async def h_chats_clear(self, _req):
        for p in (DATA_DIR / "chats").glob("*.json") if (DATA_DIR / "chats").exists() else []:
            p.unlink(missing_ok=True)
        return web.json_response({"ok": True})

    async def h_skill_upload(self, req):
        files = []
        async for field in await req.multipart():
            if field.filename:
                files.append((field.filename.replace("\\", "/"), await field.read()))
        try:
            sid = await asyncio.to_thread(self.skills.save_upload, files)
        except ValueError as e:
            return web.json_response({"error": str(e)}, status=400)
        return web.json_response({"ok": True, "id": sid})

    async def h_skill_patch(self, req):
        body = await req.json()
        self.skills.save_settings(req.match_info["id"], body.get("mode"), body.get("keywords"))
        return web.json_response({"ok": True})

    async def h_skill_delete(self, req):
        self.skills.delete(req.match_info["id"])
        return web.json_response({"ok": True})


    def get_comfy_version(self):
        """Returns the local ComfyUI git tag or commit hash, e.g. 'v0.43.0'."""
        root = self.comfy_root()
        if not root:
            return ""
        py = root / "python_embeded" / "python.exe"
        repo_dir = root / "ComfyUI"
        if not repo_dir.is_dir():
            return ""
        if py.is_file():
            try:
                code = (
                    "import pygit2\n"
                    "pygit2.option(pygit2.GIT_OPT_SET_OWNER_VALIDATION, 0)\n"
                    f"r = pygit2.Repository(r'{str(repo_dir)}')\n"
                    "h = r.head.target\n"
                    "found = ''\n"
                    "for k in r.references:\n"
                    "    if k.startswith('refs/tags/'):\n"
                    "        t = r.lookup_reference(k).target\n"
                    "        obj = r.get(t)\n"
                    "        if getattr(obj, 'target', None) == h or t == h:\n"
                    "            found = k[10:]; break\n"
                    "print(found or str(h)[:8])"
                )
                res = subprocess.run([str(py), "-c", code], capture_output=True, text=True, timeout=5)
                if res.returncode == 0 and res.stdout.strip():
                    return res.stdout.strip()
            except Exception:
                pass
        try:
            res = subprocess.run(["git", "-C", str(repo_dir), "describe", "--tags", "--always"],
                                 capture_output=True, text=True, timeout=3)
            if res.returncode == 0 and res.stdout.strip():
                return res.stdout.strip()
        except Exception:
            pass
        return ""

    async def check_comfy_update(self, force=False):
        """Check if a new ComfyUI release is available on GitHub."""
        now = time.time()
        cached = getattr(self, "_comfy_update_cache", None)
        if not force and cached and (now - cached.get("time", 0) < 3600):
            return cached["data"]

        current = await asyncio.to_thread(self.get_comfy_version)
        data = {
            "current": current or "v0.0.0",
            "latest": current or "v0.0.0",
            "has_update": False,
            "url": "https://github.com/Comfy-Org/ComfyUI/releases",
            "notes": "",
            "published_at": "",
        }
        try:
            url = "https://api.github.com/repos/Comfy-Org/ComfyUI/releases/latest"
            headers = {"User-Agent": "MinimaxH3Workspace", "Accept": "application/vnd.github.v3+json"}
            async with self.session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=6)) as resp:
                if resp.status == 200:
                    rel = await resp.json()
                    latest = rel.get("tag_name", "").strip()
                    data["latest"] = latest
                    data["url"] = rel.get("html_url") or data["url"]
                    data["notes"] = (rel.get("body") or "")[:1000]
                    data["published_at"] = rel.get("published_at") or ""

                    def parse_v(v_str):
                        m = re.findall(r'\d+', v_str)
                        return tuple(map(int, m)) if m else (0,)

                    if parse_v(latest) > parse_v(current):
                        data["has_update"] = True
        except Exception as e:
            data["error"] = str(e)

        self._comfy_update_cache = {"time": now, "data": data}
        return data

    async def h_comfy_update_check(self, req):
        force = req.query.get("force") == "1"
        res = await self.check_comfy_update(force=force)
        return web.json_response(res)

    async def h_comfy_update_status(self, _req):
        return web.json_response(getattr(self, "_comfy_update_status", {"status": "idle", "log": ""}))

    async def h_comfy_update_run(self, _req):
        if getattr(self, "_comfy_updating", False):
            return web.json_response({"error": "Обновление уже выполняется", "status": "running"}, status=409)
        if self.jobs:
            return web.json_response({"error": "Нельзя обновить ComfyUI во время активной генерации", "status": "busy"}, status=400)

        root = self.comfy_root()
        if not root:
            return web.json_response({"error": "ComfyUI не найден в портативной папке"}, status=400)

        py = root / "python_embeded" / "python.exe"
        update_py = root / "update" / "update.py"
        if not py.exists() or not update_py.exists():
            return web.json_response({"error": "Скрипт обновления update\\update.py не найден"}, status=400)

        self._comfy_updating = True
        self._comfy_update_status = {"status": "running", "log": "Остановка ComfyUI...\n"}

        repo_dir = root / "ComfyUI"

        async def do_update():
            log_lines = ["Остановка ComfyUI..."]
            was_managed = bool(self.process)
            try:
                if self.process:
                    self.process.stop()
                    await asyncio.sleep(1)

                log_lines.append("Запуск скрипта обновления ComfyUI...")
                self._comfy_update_status = {"status": "running", "log": "\n".join(log_lines)}

                cmd = [str(py), str(update_py), str(repo_dir), "--stable", "--skip_self_update"]
                env = os.environ.copy()
                proxy = engine.system_proxy()
                if proxy:
                    env["HTTP_PROXY"] = proxy
                    env["HTTPS_PROXY"] = proxy

                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    cwd=str(root),
                    env=env,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT
                )
                while True:
                    line = await proc.stdout.readline()
                    if not line:
                        break
                    line_str = line.decode("utf-8", errors="replace").strip()
                    if line_str:
                        log_lines.append(line_str)
                        self._comfy_update_status = {"status": "running", "log": "\n".join(log_lines[-60:])}
                await proc.wait()

                new_v = self.get_comfy_version()
                log_lines.append(f"Готово! Текущая версия: {new_v}")
                self._comfy_update_cache = None  # invalidate cache
                self._comfy_update_status = {"status": "done", "log": "\n".join(log_lines), "version": new_v}
            except Exception as e:
                log_lines.append(f"Ошибка обновления: {e}")
                self._comfy_update_status = {"status": "error", "log": "\n".join(log_lines)}
            finally:
                self._comfy_updating = False
                if was_managed and not self.process:
                    self.start_managed()

        self._bg_run(do_update())
        return web.json_response({"ok": True, "status": "running"})


    async def h_index(self, _req):
        return web.FileResponse(UI_DIR / "index.html", headers={"Cache-Control": "no-store"})


def lora_strength(form, key, default):
    """Strength sent by the UI: a number when the LoRA is on, "off" when off, nothing -> the default (None = off)."""
    v = form.get(f"lora_{key}")
    if v in (None, ""):
        return default
    if v == "off":
        return None
    try:
        return float(v)
    except ValueError:
        return default


def re_chat_id(cid):
    return cid.isalnum() and 4 <= len(cid) <= 40


def gpu_name(raw):
    # "cuda:0 NVIDIA GeForce RTX 4060 Ti : cudaMallocAsync" -> "NVIDIA GeForce RTX 4060 Ti"
    name = raw.split(" : ")[0]
    return name.split(" ", 1)[1] if name.startswith("cuda:") and " " in name else name


def comfy_error_text(res):
    if not isinstance(res, dict):
        return "ComfyUI отклонил задание"
    err = res.get("error") or {}
    parts = [err.get("message", "") + (": " + err.get("details", "") if err.get("details") else "")]
    for node_id, ne in (res.get("node_errors") or {}).items():
        for e in ne.get("errors", []):
            parts.append(f'{ne.get("class_type", node_id)}: {e.get("message")} {e.get("details", "")}'.strip())
    return "\n".join(p for p in parts if p) or "ComfyUI отклонил задание"


def send_to_recycle_bin(path):
    ps = ("Add-Type -AssemblyName Microsoft.VisualBasic; "
          "[Microsoft.VisualBasic.FileIO.FileSystem]::DeleteFile($env:H3_DEL, 'OnlyErrorDialogs', 'SendToRecycleBin')")
    r = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                       env={**os.environ, "H3_DEL": str(path)}, capture_output=True)
    return r.returncode == 0 and not path.exists()


def build_app(comfy_override=None):
    state = App(comfy_override)
    app = web.Application(client_max_size=300 * 1024 * 1024)
    app.on_startup.append(state.start)
    app.on_cleanup.append(state.stop)
    app.add_routes([
        web.get("/", state.h_index),
        web.get("/api/status", state.h_status),
        web.get("/api/system", state.h_system),
        web.get("/api/audio/info", state.h_audio_info),
        web.post("/api/audio/separate", state.h_audio_separate),
        web.get("/api/setup/info", state.h_setup_info),
        web.post("/api/setup/external", state.h_setup_external),
        web.post("/api/setup/install", state.h_setup_install),
        web.post("/api/mode/add", state.h_mode_add),
        web.post("/api/settings/mirror", state.h_mirror),
        web.post("/api/mode/offered", state.h_mode_offered),
        web.post("/api/setup/cancel", state.h_setup_cancel),
        web.post("/api/open-driver-page", state.h_open_driver),
        web.post("/api/clipboard", state.h_clipboard),
        web.post("/api/support", state.h_support),
        web.get("/api/queue/saved", state.h_queue_saved),
        web.post("/api/queue/resume", state.h_queue_resume),
        web.post("/api/queue/discard", state.h_queue_discard),
        web.post("/api/search-web", state.h_search_web),
        web.post("/api/report", state.h_report),
        web.post("/api/setup/reset", state.h_setup_reset),
        web.post("/api/generate/video", state.h_generate_video),
        web.post("/api/generate/krea", state.h_generate_krea),
        web.post("/api/upscale", state.h_upscale),
        web.post("/api/upscale_file", state.h_upscale_file),
        web.get("/api/version", lambda _req: web.json_response({"version": APP_VERSION})),
        web.get("/api/update/check", state.h_update_check),
        web.post("/api/update/install", state.h_update_install),
        web.post("/api/generate/qwen", state.h_generate_qwen),
        web.post("/api/generate/lipsync", state.h_generate_lipsync),
        web.get("/api/nodes", state.h_nodes),
        web.get("/api/models/files", state.h_models_files),
        web.post("/api/models/pick", state.h_models_pick),
        web.post("/api/models/open", state.h_models_open),
        web.post("/api/models/download", state.h_models_download),
        web.post("/api/models/download/cancel", state.h_models_download_cancel),
        web.post("/api/models/variant", state.h_models_variant),
        web.get("/api/wf/list", state.h_wf_list),
        web.post("/api/wf/add", state.h_wf_add),
        web.get("/api/wf/{id}", state.h_wf_get),
        web.delete("/api/wf/{id}", state.h_wf_delete),
        web.post("/api/generate/custom", state.h_generate_custom),
        web.get("/api/loras/custom", state.h_loras_custom),
        web.post("/api/loras/custom", state.h_loras_custom_set),
        web.post("/api/loras/conflicts", state.h_loras_conflicts),
        web.post("/api/loras/open-folder", state.h_loras_open_folder),
        web.post("/api/loras/meta/save", state.h_loras_meta_save),
        web.get("/api/civitai/search", state.h_civitai_search),
        web.post("/api/civitai/resolve", state.h_civitai_resolve),
        web.get("/api/hf/search", state.h_hf_search),
        web.post("/api/hf/resolve", state.h_hf_resolve),
        web.post("/api/loras/tokens/save", state.h_loras_tokens_save),
        web.post("/api/civitai/download", state.h_civitai_download),
        web.get("/api/civitai/download/status", state.h_civitai_download_status),
        web.post("/api/civitai/download/cancel", state.h_civitai_download_cancel),
        web.get("/api/comfy/update/check", state.h_comfy_update_check),
        web.post("/api/comfy/update/run", state.h_comfy_update_run),
        web.get("/api/comfy/update/status", state.h_comfy_update_status),
        web.post("/api/generate/refs", state.h_generate_refs),
        web.post("/api/generate/actor", state.h_generate_actor),
        web.post("/api/generate/swap", state.h_generate_swap),
        web.post("/api/generate/long", state.h_generate_long),
        web.post("/api/montage", state.h_montage),
        web.get("/api/long/state", state.h_long_state),
        web.get("/api/long/project", state.h_long_project),
        web.post("/api/long/project", state.h_long_project),
        web.get("/api/long/ref/{run}/{name}", state.h_long_ref),
        web.post("/api/generate/action", state.h_generate_action),
        web.get("/api/action/info", state.h_action_info),
        web.get("/api/cloud", state.h_cloud_get),
        web.post("/api/cloud", state.h_cloud_set),
        web.post("/api/cloud/test", state.h_cloud_test),
        web.post("/api/setup/cloud", state.h_setup_cloud),
        web.post("/api/interrupt", state.h_interrupt),
        web.get("/api/agent/status", state.h_agent_status),
        web.post("/api/agent/translate", state.h_agent_translate),
        web.post("/api/agent/install", state.h_agent_install),
        web.post("/api/agent/install/cancel", state.h_agent_install_cancel),
        web.post("/api/agent/model", state.h_agent_model),
        web.post("/api/agent/variant", state.h_agent_variant),
        web.post("/api/speech/transcribe", state.h_transcribe),
        web.post("/api/speech/folder", state.h_speech_folder),
        web.post("/api/agent/chat", state.h_agent_chat),
        web.get("/api/agent/chats", state.h_chats),
        web.delete("/api/agent/chats", state.h_chats_clear),
        web.get("/api/agent/chats/{id}", state.h_chat_get),
        web.put("/api/agent/chats/{id}", state.h_chat_put),
        web.delete("/api/agent/chats/{id}", state.h_chat_delete),
        web.post("/api/agent/skills", state.h_skill_upload),
        web.patch("/api/agent/skills/{id}", state.h_skill_patch),
        web.delete("/api/agent/skills/{id}", state.h_skill_delete),
        web.get("/api/results", state.h_results_get),
        web.post("/api/results", state.h_results_set),
        web.post("/api/results/backfill", state.h_results_backfill),
        web.post("/api/results/clean", state.h_results_clean),
        web.get("/api/library", state.h_library),
        web.patch("/api/library/{id}", state.h_library_patch),
        web.delete("/api/library/{id}", state.h_library_delete),
        web.get("/api/file/{id}", state.h_file),
        web.get("/api/events", state.h_events),
        web.static("/ui", UI_DIR),
    ])
    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comfy", help="always use this ComfyUI URL (skips the setup wizard)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--open", action="store_true", help="open the UI in the default browser")
    args = ap.parse_args()

    app = build_app(args.comfy)
    if args.open:
        async def open_browser(_app):
            asyncio.get_running_loop().call_later(1, webbrowser.open, f"http://127.0.0.1:{args.port}")
        app.on_startup.append(open_browser)
    web.run_app(app, host="127.0.0.1", port=args.port, print=lambda *_: print(f"Minimax H3 Workspace: http://127.0.0.1:{args.port}"))


if __name__ == "__main__":
    main()
