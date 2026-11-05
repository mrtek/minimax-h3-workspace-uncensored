"""Hugging Face API integration for MiniMax H3 LoRAs.

Allows searching, inspecting, and downloading LoRAs directly from huggingface.co
with automatic trigger word extraction, preview detection, and metadata caching.
"""

import asyncio
import json
import re
import urllib.parse
from pathlib import Path
import aiohttp

DEFAULT_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"


def clean_text(text):
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&[a-zA-Z0-9#]+;", " ", text)
    return " ".join(text.split()).strip()


def extract_triggers_and_desc(readme_text):
    """Extracts trigger words and an excerpt description from README.md."""
    if not readme_text:
        return [], ""
    triggers = []
    desc = ""

    # 1. Frontmatter
    body = readme_text
    if readme_text.startswith("---"):
        parts = readme_text.split("---", 2)
        if len(parts) >= 3:
            fm = parts[1]
            body = parts[2]
            ip = re.search(r'(?i)instance_prompt:\s*>?-?\s*([^\n\r]+)', fm)
            if ip:
                val = ip.group(1).strip()
                if val and val != ">-":
                    triggers.append(val)

    # 2. Trigger words regex
    patterns = [
        r'(?i)\*\*Trigger\s*words?:\*\*\s*[`"\'*]*([^`"\'\n\r*#]{2,80})[`"\'*]*',
        r'(?i)(?:trigger(?: words?| keywords?)?|activation keywords?|trained words?)[:\s*]+[`"\'*]*([^`"\'\n\r*#]{2,80})[`"\'*]*',
    ]
    for p in patterns:
        for m in re.findall(p, readme_text):
            clean = m.strip().strip("`'\"*")
            if clean and not clean.startswith(">-") and clean not in triggers and len(clean) > 1:
                triggers.append(clean)

    # 3. Clean description from body (first meaningful paragraph)
    for p in body.split("\n\n"):
        p_strip = p.strip()
        if not p_strip or p_strip.startswith("#") or p_strip.startswith("---") or p_strip.startswith("<"):
            continue
        p_clean = clean_text(re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', p_strip))
        if len(p_clean) > 25:
            desc = p_clean[:240]
            break

    return triggers[:10], desc


async def fetch_readme(session, mid, token=None):
    """Fetches raw README.md for a model repo."""
    url = f"https://huggingface.co/{mid}/raw/main/README.md"
    headers = {"User-Agent": DEFAULT_USER_AGENT}
    if token:
        headers["Authorization"] = f"Bearer {token.strip()}"
    try:
        async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=4)) as r:
            if r.status == 200:
                return mid, await r.text()
    except Exception:
        pass
    return mid, ""


async def search_loras(session, query="", limit=30, token=None):
    """Searches Hugging Face for MiniMax H3 LoRA models."""
    q_raw = (query or "").strip()
    if q_raw:
        search_query = f"{q_raw} minimax h3"
    else:
        search_query = "minimax h3 lora"

    url = (
        f"https://huggingface.co/api/models"
        f"?search={urllib.parse.quote_plus(search_query)}"
        f"&limit={max(10, min(50, limit))}"
        f"&cardData=true&full=true"
    )
    headers = {"User-Agent": DEFAULT_USER_AGENT, "Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token.strip()}"

    try:
        async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=20)) as resp:
            if resp.status != 200:
                err = (await resp.text())[:200]
                raise RuntimeError(f"Hugging Face API HTTP {resp.status}: {err}")
            models_data = await resp.json()
    except Exception as e:
        raise RuntimeError(f"Ошибка поиска Hugging Face: {e}")

    # Gather READMEs for top models to get accurate triggers and descriptions
    mids_to_fetch = [
        m.get("id") for m in models_data[:20]
        if m.get("id") and not (m.get("cardData") or {}).get("instance_prompt")
    ]
    readmes = {}
    if mids_to_fetch:
        tasks = [fetch_readme(session, mid, token=token) for mid in mids_to_fetch]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for res in results:
            if isinstance(res, tuple) and len(res) == 2:
                readmes[res[0]] = res[1]

    items = []
    for m in models_data:
        mid = m.get("id", "")
        if not mid:
            continue

        tags = m.get("tags") or []
        siblings = [s.get("rfilename", "") for s in m.get("siblings") or []]
        card = m.get("cardData") or {}

        # Look for safetensors files that are LoRAs (exclude base model multi-part shards)
        st_files = [
            f for f in siblings
            if f.endswith(".safetensors")
            and not re.search(r"model-\d{5}-of-\d{5}", f)
            and not any(p in f.lower() for p in ["text_encoder/", "transformer/", "audio_vae/", "video_vae/", "vae/"])
        ]

        # Is this a LoRA or related weights?
        is_lora = ("lora" in tags or "lora" in mid.lower() or any("lora" in f.lower() for f in st_files) or "adapter" in tags)
        if not st_files and not is_lora:
            continue

        if not st_files:
            continue

        # Extract triggers
        triggers = []
        if card.get("instance_prompt"):
            ip = str(card["instance_prompt"]).strip()
            if ip and ip != ">-":
                triggers.append(ip)
        if card.get("trigger_words"):
            tw = card["trigger_words"]
            if isinstance(tw, list):
                triggers.extend([str(w).strip() for w in tw if str(w).strip()])
            elif isinstance(tw, str) and tw.strip():
                triggers.append(tw.strip())

        # Check README if triggers or desc are empty
        readme_txt = readmes.get(mid, "")
        readme_triggers, readme_desc = extract_triggers_and_desc(readme_txt)
        for rt in readme_triggers:
            if rt not in triggers:
                triggers.append(rt)

        # Preview media (mp4 / webm / png / jpg / webp)
        media_files = [
            f for f in siblings
            if f.lower().endswith((".mp4", ".webm", ".jpg", ".png", ".jpeg", ".webp"))
            and not any(p in f.lower() for p in ["vae", "weights"])
        ]
        images = []
        # Check widget outputs first (often hero preview)
        widget_data = card.get("widget") or []
        for w in widget_data:
            out = (w.get("output") or {}).get("url")
            if out:
                images.append({"url": out})
                break

        if not images and media_files:
            # Pick first preview media file
            prev_url = f"https://huggingface.co/{mid}/resolve/main/{media_files[0]}"
            images.append({"url": prev_url})

        # Creator and title
        parts = mid.split("/", 1)
        creator = parts[0] if len(parts) > 1 else ""
        name = parts[1] if len(parts) > 1 else mid
        name = name.replace("-", " ").replace("_", " ")

        # Description
        desc = card.get("description") or card.get("summary") or readme_desc or ""
        if not desc:
            desc = f"Hugging Face LoRA модель от {creator or 'сообщества'}. Теги: {', '.join(tags[:4])}"
        desc = clean_text(str(desc))[:260]

        # Versions / files
        versions = []
        for sf in st_files:
            clean_fn = sf.split("/")[-1]
            dl_url = f"https://huggingface.co/{mid}/resolve/main/{sf}"
            versions.append({
                "id": f"{mid}@{clean_fn}",
                "name": clean_fn.replace(".safetensors", ""),
                "baseModel": "MiniMax H3",
                "trainedWords": triggers,
                "filename": clean_fn,
                "sizeMB": 0,
                "downloadUrl": dl_url,
                "description": desc,
            })

        items.append({
            "id": mid,
            "name": name,
            "description": desc,
            "creator": creator,
            "images": images,
            "versions": versions,
            "civitaiUrl": f"https://huggingface.co/{mid}",
            "source": "huggingface",
            "downloads": m.get("downloads", 0),
            "likes": m.get("likes", 0),
        })

    return {
        "items": items,
        "page": 1,
        "totalPages": 1,
        "totalItems": len(items),
        "source": "huggingface",
    }


async def resolve_url_or_repo(session, text, token=None):
    """Resolves a Hugging Face link or repo ID to a downloadable LoRA version."""
    text = str(text or "").strip()
    headers = {"User-Agent": DEFAULT_USER_AGENT, "Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token.strip()}"

    # Extract author and model:
    m = re.search(r"huggingface\.co/([^/]+/[^/?#]+)", text)
    if m:
        repo_and_rest = m.group(1)
    else:
        m2 = re.match(r"^([a-zA-Z0-9_\-\.]+/[a-zA-Z0-9_\-\.]+)", text)
        if m2:
            repo_and_rest = m2.group(1)
        else:
            raise ValueError(f"Не удалось распознать ссылку Hugging Face: {text}")

    parts = repo_and_rest.split("/")
    author = parts[0]
    repo_name = parts[1]
    repo_id = f"{author}/{repo_name}"

    # Check if a specific filename was requested
    target_filename = None
    fn_match = re.search(r"/(?:blob|resolve)/[^/]+/(.+\.safetensors)", text)
    if fn_match:
        target_filename = fn_match.group(1)

    url = f"https://huggingface.co/api/models/{repo_id}?cardData=true&full=true"
    async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=20)) as resp:
        if resp.status != 200:
            err = (await resp.text())[:200]
            raise RuntimeError(f"Hugging Face API HTTP {resp.status}: {err}")
        m_data = await resp.json()

    siblings = [s.get("rfilename", "") for s in m_data.get("siblings") or []]
    st_files = [
        f for f in siblings
        if f.endswith(".safetensors")
        and not re.search(r"model-\d{5}-of-\d{5}", f)
        and not any(p in f.lower() for p in ["text_encoder/", "transformer/", "audio_vae/", "video_vae/", "vae/"])
    ]

    if not st_files:
        raise RuntimeError(f"В репозитории {repo_id} не найдено .safetensors файлов LoRA.")

    if target_filename and target_filename in st_files:
        chosen_file = target_filename
    else:
        chosen_file = st_files[0]

    clean_fn = chosen_file.split("/")[-1]
    card = m_data.get("cardData") or {}

    # Extract triggers
    triggers = []
    if card.get("instance_prompt"):
        ip = str(card["instance_prompt"]).strip()
        if ip and ip != ">-":
            triggers.append(ip)
    if card.get("trigger_words"):
        tw = card["trigger_words"]
        if isinstance(tw, list):
            triggers.extend([str(w).strip() for w in tw if str(w).strip()])
        elif isinstance(tw, str) and tw.strip():
            triggers.append(tw.strip())

    # Try fetching README.md for triggers and desc
    mid, readme_txt = await fetch_readme(session, repo_id, token=token)
    readme_triggers, readme_desc = extract_triggers_and_desc(readme_txt)
    for rt in readme_triggers:
        if rt not in triggers:
            triggers.append(rt)

    desc = card.get("description") or card.get("summary") or readme_desc or f"LoRA {clean_fn} из Hugging Face ({repo_id})"
    desc = clean_text(str(desc))[:260]

    return {
        "id": repo_id,
        "name": repo_name.replace("-", " ").replace("_", " "),
        "source": "huggingface",
        "version": {
            "id": f"{repo_id}@{clean_fn}",
            "name": clean_fn.replace(".safetensors", ""),
            "trainedWords": triggers,
            "filename": clean_fn,
            "sizeMB": 0,
            "downloadUrl": f"https://huggingface.co/{repo_id}/resolve/main/{chosen_file}",
            "description": desc,
        }
    }
