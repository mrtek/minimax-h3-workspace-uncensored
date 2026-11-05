"""Civitai API integration for MiniMax H3 LoRAs.

Allows searching, inspecting, and downloading LoRAs directly from civitai.red or civitai.com
with automatic trigger word extraction and metadata caching.
"""

import asyncio
import json
import os
import re
import time
from pathlib import Path
import aiohttp

DEFAULT_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

# In-memory download task state
CURRENT_DOWNLOAD = {
    "active": False,
    "file": "",
    "name": "",
    "done": 0,
    "total": 0,
    "percent": 0,
    "speed": "",
    "status": "idle", # "idle", "downloading", "done", "error"
    "error": None,
    "task": None,
}


def clean_html(raw_html):
    if not raw_html:
        return ""
    text = re.sub(r"<[^>]+>", " ", raw_html)
    text = re.sub(r"&[a-zA-Z0-9#]+;", " ", text)
    return " ".join(text.split()).strip()


async def search_loras(session, query="", page=1, limit=20, mirror="red", nsfw=True, cursor=None, token=None):
    """Searches Civitai API for MiniMax H3 LoRAs."""
    domain = "civitai.red" if mirror == "red" else "civitai.com"
    url = f"https://{domain}/api/v1/models"
    params = {
        "types": "LORA",
        "baseModels": "MiniMax H3",
        "nsfw": "true" if nsfw else "false",
        "limit": str(min(50, max(5, limit))),
    }
    q = (query or "").strip()
    # Civitai rule: Cannot use page param with query search!
    if q:
        params["query"] = q
        if cursor:
            params["cursor"] = str(cursor)
    else:
        if cursor:
            params["cursor"] = str(cursor)
        elif page:
            params["page"] = str(max(1, page))

    if token:
        params["token"] = token.strip()

    headers = {"User-Agent": DEFAULT_USER_AGENT, "Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token.strip()}"
    try:
        async with session.get(url, params=params, headers=headers, timeout=aiohttp.ClientTimeout(total=20)) as resp:
            if resp.status != 200:
                err_body = (await resp.text())[:200]
                # If civitai.red failed, try civitai.com as fallback
                if mirror == "red":
                    return await search_loras(session, query, page, limit, mirror="com", nsfw=nsfw, cursor=cursor, token=token)
                raise RuntimeError(f"Civitai API HTTP {resp.status}: {err_body}")
            data = await resp.json()
    except Exception as e:
        if mirror == "red":
            try:
                return await search_loras(session, query, page, limit, mirror="com", nsfw=nsfw, cursor=cursor, token=token)
            except Exception:
                pass
        raise RuntimeError(f"Ошибка поиска Civitai: {e}")

    items = []
    for item in data.get("items", []):
        mid = item.get("id")
        name = item.get("name", "")
        desc = clean_html(item.get("description", ""))
        creator = (item.get("creator") or {}).get("username", "")
        
        # Collect preview images
        images = []
        for v in item.get("modelVersions", []):
            for img in v.get("images", []):
                if img.get("url"):
                    images.append({
                        "url": img["url"],
                        "nsfw": img.get("nsfw", False) or img.get("nsfwLevel", 0) > 1,
                        "width": img.get("width"),
                        "height": img.get("height"),
                    })
            if images:
                break

        # Filter versions for MiniMax H3
        versions = []
        for v in item.get("modelVersions", []):
            base_model = v.get("baseModel", "")
            if "minimax" not in base_model.lower():
                continue
            
            # Find primary model file
            files = v.get("files", [])
            model_file = next((f for f in files if f.get("type") == "Model" and f.get("name", "").endswith(".safetensors")), None)
            if not model_file and files:
                model_file = next((f for f in files if f.get("name", "").endswith(".safetensors")), files[0])

            if not model_file:
                continue

            trained_words = v.get("trainedWords") or []
            if isinstance(trained_words, str):
                trained_words = [w.strip() for w in trained_words.split(",") if w.strip()]

            size_mb = round(float(model_file.get("sizeKB", 0)) / 1024, 1)

            versions.append({
                "id": v.get("id"),
                "name": v.get("name", "Default"),
                "baseModel": base_model,
                "trainedWords": trained_words,
                "filename": model_file.get("name") or f"lora_{v.get('id')}.safetensors",
                "sizeMB": size_mb,
                "downloadUrl": model_file.get("downloadUrl") or f"https://{domain}/api/download/models/{v.get('id')}",
                "description": clean_html(v.get("description", "")),
            })

        if versions:
            items.append({
                "id": mid,
                "name": name,
                "description": desc,
                "creator": creator,
                "images": images[:4],
                "versions": versions,
                "civitaiUrl": f"https://{domain}/models/{mid}",
                "source": "civitai",
            })

    metadata = data.get("metadata", {})
    return {
        "items": items,
        "page": metadata.get("currentPage", page),
        "totalPages": metadata.get("totalPages", 1),
        "totalItems": metadata.get("totalItems", len(items)),
        "mirror": mirror,
    }


async def resolve_url_or_id(session, text, mirror="red"):
    """Parses a Civitai model link or ID and returns the model & version info."""
    text = str(text or "").strip()
    domain = "civitai.red" if mirror == "red" else "civitai.com"
    headers = {"User-Agent": DEFAULT_USER_AGENT, "Accept": "application/json"}

    # Check if text contains modelVersionId
    v_match = re.search(r"modelVersionId=(\d+)", text)
    if v_match:
        vid = v_match.group(1)
        url = f"https://{domain}/api/v1/model-versions/{vid}"
        async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=15)) as r:
            if r.status == 200:
                v = await r.json()
                files = v.get("files", [])
                mf = next((f for f in files if f.get("name", "").endswith(".safetensors")), files[0] if files else {})
                return {
                    "id": v.get("modelId"),
                    "name": v.get("name"),
                    "version": {
                        "id": v.get("id"),
                        "name": v.get("name"),
                        "trainedWords": v.get("trainedWords") or [],
                        "filename": mf.get("name", f"lora_{vid}.safetensors"),
                        "sizeMB": round(float(mf.get("sizeKB", 0)) / 1024, 1),
                        "downloadUrl": mf.get("downloadUrl") or f"https://{domain}/api/download/models/{vid}",
                        "description": clean_html(v.get("description", "")),
                    }
                }

    # Check if text contains models/<id>
    m_match = re.search(r"models/(\d+)", text) or re.match(r"^(\d+)$", text)
    if m_match:
        mid = m_match.group(1)
        url = f"https://{domain}/api/v1/models/{mid}"
        async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=15)) as r:
            if r.status == 200:
                m = await r.json()
                versions = m.get("modelVersions", [])
                h3_v = next((v for v in versions if "minimax" in v.get("baseModel", "").lower()), versions[0] if versions else None)
                if h3_v:
                    files = h3_v.get("files", [])
                    mf = next((f for f in files if f.get("name", "").endswith(".safetensors")), files[0] if files else {})
                    return {
                        "id": m.get("id"),
                        "name": m.get("name"),
                        "version": {
                            "id": h3_v.get("id"),
                            "name": h3_v.get("name"),
                            "trainedWords": h3_v.get("trainedWords") or [],
                            "filename": mf.get("name", f"lora_{h3_v.get('id')}.safetensors"),
                            "sizeMB": round(float(mf.get("sizeKB", 0)) / 1024, 1),
                            "downloadUrl": mf.get("downloadUrl") or f"https://{domain}/api/download/models/{h3_v.get('id')}",
                            "description": clean_html(h3_v.get("description", "")),
                        }
                    }

    raise ValueError(f"Не удалось распознать ссылку или ID Civitai: {text}")


async def download_file_stream(session, url, dest_path, token=None):
    """Downloads a file with resume support, token auth, and updates CURRENT_DOWNLOAD."""
    global CURRENT_DOWNLOAD
    dest = Path(dest_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")

    CURRENT_DOWNLOAD["status"] = "downloading"
    CURRENT_DOWNLOAD["error"] = None
    CURRENT_DOWNLOAD["file"] = dest.name

    download_url = url
    if token and "token=" not in download_url:
        delim = "&" if "?" in download_url else "?"
        download_url = f"{download_url}{delim}token={token.strip()}"

    headers = {"User-Agent": DEFAULT_USER_AGENT}
    if token:
        headers["Authorization"] = f"Bearer {token.strip()}"

    have = part.stat().st_size if part.exists() else 0
    if have:
        headers["Range"] = f"bytes={have}-"

    start_time = time.time()
    downloaded_this_session = 0

    try:
        async with session.get(download_url, headers=headers, timeout=aiohttp.ClientTimeout(total=None, sock_read=120), allow_redirects=True) as resp:
            # Check for Civitai login / 403 / 401 requirement
            if resp.status in (401, 403):
                raise RuntimeError("Civitai требует авторизацию (API-токен) для скачивания этой модели. Введите API-токен Civitai в настройках каталога.")
            if "auth.civitai.com" in str(resp.url) or "login" in str(resp.url):
                raise RuntimeError("Civitai перенаправил на вход в аккаунт. Для этой модели требуется бесплатный API-токен Civitai (Settings -> API Keys).")
            if resp.status not in (200, 206) and resp.status != 416:
                raise RuntimeError(f"HTTP ошибка при скачивании: {resp.status}")

            total = 0
            if resp.status == 206:
                cr = resp.headers.get("Content-Range", "")
                m = re.search(r"/(\d+)$", cr)
                total = int(m.group(1)) if m else have + int(resp.headers.get("Content-Length", 0))
            elif resp.status == 200:
                have = 0
                total = int(resp.headers.get("Content-Length", 0))
            elif resp.status == 416:
                # Already complete
                pass

            CURRENT_DOWNLOAD["total"] = total
            CURRENT_DOWNLOAD["done"] = have

            mode = "ab" if have and resp.status == 206 else "wb"
            with open(part, mode) as f:
                async for chunk in resp.content.iter_chunked(256 * 1024):
                    if chunk:
                        f.write(chunk)
                        have += len(chunk)
                        downloaded_this_session += len(chunk)
                        CURRENT_DOWNLOAD["done"] = have
                        if total > 0:
                            CURRENT_DOWNLOAD["percent"] = min(100, round((have / total) * 100, 1))
                        
                        elapsed = time.time() - start_time
                        if elapsed > 0.5:
                            speed_mb = (downloaded_this_session / (1024 * 1024)) / elapsed
                            CURRENT_DOWNLOAD["speed"] = f"{speed_mb:.1f} МБ/с"

        # Rename .part to dest
        if dest.exists():
            dest.unlink()
        part.rename(dest)

        CURRENT_DOWNLOAD["status"] = "done"
        CURRENT_DOWNLOAD["percent"] = 100
        CURRENT_DOWNLOAD["speed"] = "Завершено"
    except Exception as e:
        CURRENT_DOWNLOAD["status"] = "error"
        CURRENT_DOWNLOAD["error"] = str(e)
        raise
    finally:
        CURRENT_DOWNLOAD["active"] = False
