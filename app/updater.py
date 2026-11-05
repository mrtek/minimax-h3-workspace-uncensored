"""Updates from GitHub Releases: look for a newer release, download its installer, check it, run it.

The release must carry an installer starting with Minimax.H3.Workspace.Uncensored-Setup (GitHub shows its sha256 next to it).
"""

import hashlib
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import aiohttp

from version import APP_VERSION

REPO = "mrtek/minimax-h3-workspace-uncensored"
API = f"https://api.github.com/repos/{REPO}/releases/latest"
DOWNLOAD_PREFIX = f"https://github.com/{REPO}/releases/download/"
ASSET = "Minimax.H3.Workspace.Uncensored-Setup.exe"
HEADERS = {"User-Agent": "Minimax-H3-Workspace", "Accept": "application/vnd.github+json"}


def parse(v):
    """'v1.2.3' -> (1, 2, 3); anything unreadable -> None."""
    m = re.match(r"^v?(\d+)\.(\d+)(?:\.(\d+))?", str(v or "").strip())
    return tuple(int(x or 0) for x in m.groups()) if m else None


async def check(session):
    """{"available": bool, "current", "version", "url", "asset", "size", "sha256"}; never raises (no internet is normal)."""
    out = {"available": False, "current": APP_VERSION}
    try:
        async with session.get(API, headers=HEADERS, timeout=aiohttp.ClientTimeout(total=8)) as r:
            if r.status != 200:
                return out
            rel = await r.json()
    except Exception:
        return out
    new, cur = parse(rel.get("tag_name")), parse(APP_VERSION)
    asset = next((a for a in rel.get("assets", [])
                  if a.get("name", "").startswith("Minimax.H3.Workspace.Uncensored-Setup") and a.get("name", "").endswith(".exe")), None)
    if not new or not cur or not asset or rel.get("draft") or rel.get("prerelease"):
        return out
    url = asset.get("browser_download_url", "")
    if new > cur and url.startswith(DOWNLOAD_PREFIX):
        digest = str(asset.get("digest") or "")
        out.update(available=True, version=".".join(map(str, new)), url=rel.get("html_url", ""), asset=url,
                   name=asset.get("name") or ASSET,
                   size=int(asset.get("size") or 0), sha256=digest.split(":", 1)[1] if digest.startswith("sha256:") else "")
    return out


async def download(session, info):
    """Save the installer to a temp folder and check it against the size and sha256 GitHub published. Returns its path."""
    if not info.get("asset", "").startswith(DOWNLOAD_PREFIX):
        raise ValueError("Адрес обновления не с GitHub")
    folder = Path(tempfile.gettempdir()) / "h3studio_update"
    folder.mkdir(exist_ok=True)
    filename = info.get("name") or ASSET
    dest = folder / filename
    sha = hashlib.sha256()
    size = 0
    async with session.get(info["asset"], headers=HEADERS, timeout=aiohttp.ClientTimeout(total=900)) as r:
        if r.status != 200:
            raise ValueError(f"GitHub ответил {r.status}")
        with open(dest, "wb") as f:
            async for chunk in r.content.iter_chunked(1 << 20):
                f.write(chunk)
                sha.update(chunk)
                size += len(chunk)
    if (info.get("size") and size != info["size"]) or (info.get("sha256") and sha.hexdigest() != info["sha256"]):
        dest.unlink(missing_ok=True)
        raise ValueError("Файл обновления скачался с ошибкой, попробуйте ещё раз")
    return dest


def run(installer):
    """Start the installer detached and quietly; it closes this app, replaces the files and opens the app again."""
    flags = 0x00000008 | 0x00000200  # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
    subprocess.Popen([str(installer), "/SILENT", "/SUPPRESSMSGBOXES", "/CLOSEAPPLICATIONS", "/NORESTART"],
                     creationflags=flags, close_fds=True)


def can_update():
    """Only an installed build updates itself: a run from source or the maker's folder does not."""
    return bool(getattr(sys, "frozen", False))
