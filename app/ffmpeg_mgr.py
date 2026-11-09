"""Portable FFmpeg manager and video montage engine for Minimax H3 Workspace.

Provides:
- Detection of portable FFmpeg strictly in the workspace (never uses system PATH);
- Automatic download, extraction, and update checks of official Windows FFmpeg builds;
- High-performance video concatenation:
  - Instant stream copy (-c copy) when video codecs and resolutions match (~0.5-1s, 0% quality loss);
  - Smart scaling & padding when resolutions differ, powered by NVIDIA NVENC hardware acceleration;
  - Real-time progress reporting parsed from FFmpeg stdout (-progress pipe:1);
  - Optional background audio / music attachment and audio mixing.
"""

import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path

# Known reliable static essentials builds (GyanD) & mirror fallbacks
DEFAULT_RELEASE_URL = "https://github.com/GyanD/codexffmpeg/releases/download/9.0.2/ffmpeg-9.0.2-essentials_build.zip"
DEFAULT_RELEASE_VERSION = "9.0.2"
MIRROR_RELEASE_URL = "https://ghproxy.net/https://github.com/GyanD/codexffmpeg/releases/download/9.0.2/ffmpeg-9.0.2-essentials_build.zip"
BTBN_FALLBACK_URL = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl-shared.zip"


def get_default_ffmpeg_dir(workspace_root=None):
    """Default location: <WORKSPACE_ROOT>/ffmpeg.
    E.g. G:/MinimaxH3Workspace/ffmpeg or detected workspace drive.
    """
    if workspace_root:
        return Path(workspace_root) / "ffmpeg"

    # Search for an existing MinimaxH3Workspace folder on fixed drives
    for letter in "GDEFGHIJKLMNOPQRSTUVWXYZ":
        candidate = Path(f"{letter}:\\MinimaxH3Workspace")
        if candidate.exists():
            return candidate / "ffmpeg"

    local = os.environ.get("LOCALAPPDATA", Path.home())
    return Path(local) / "MinimaxH3Workspace" / "ffmpeg"


def get_ffmpeg_dir(config, workspace_root=None):
    """Returns the active FFmpeg directory: custom user folder if configured, else default."""
    if config:
        custom = config.get("ffmpeg_dir")
        if custom and str(custom).strip():
            return Path(str(custom).strip())
        install_dir = config.get("install_dir")
        if install_dir and Path(install_dir).exists():
            return Path(install_dir) / "ffmpeg"
    return get_default_ffmpeg_dir(workspace_root)


def find_ffmpeg(folder):
    """Finds ffmpeg.exe and ffprobe.exe inside the specified folder only.
    Checks folder/bin/ffmpeg.exe and folder/ffmpeg.exe.
    DOES NOT check system PATH (per user requirement).
    """
    if not folder:
        return None, None
    folder = Path(folder)
    if not folder.is_dir():
        return None, None

    cand_ffmpeg = [
        folder / "bin" / "ffmpeg.exe",
        folder / "ffmpeg.exe",
    ]
    cand_ffprobe = [
        folder / "bin" / "ffprobe.exe",
        folder / "ffprobe.exe",
    ]
    ff = next((p for p in cand_ffmpeg if p.is_file()), None)
    fp = next((p for p in cand_ffprobe if p.is_file()), None)
    return ff, fp


def get_ffmpeg_version(ffmpeg_exe):
    """Executes ffmpeg.exe -version and parses the version string."""
    if not ffmpeg_exe or not Path(ffmpeg_exe).is_file():
        return None
    try:
        creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        res = subprocess.run([str(ffmpeg_exe), "-version"], capture_output=True, text=True, timeout=5, creationflags=creationflags)
        first_line = res.stdout.splitlines()[0] if res.stdout else ""
        m = re.search(r"ffmpeg version\s+([^\s]+)", first_line, re.I)
        return m.group(1) if m else first_line[:30]
    except Exception:
        return None


def get_ffmpeg_status(config, workspace_root=None):
    """Returns a dict with the full status of portable FFmpeg for the UI."""
    cur_dir = get_ffmpeg_dir(config, workspace_root)
    def_dir = get_default_ffmpeg_dir(workspace_root or (config.get("install_dir") if config else None))
    ff, fp = find_ffmpeg(cur_dir)
    version = get_ffmpeg_version(ff) if ff else None
    return {
        "installed": bool(ff and fp),
        "ffmpeg_path": str(ff) if ff else None,
        "ffprobe_path": str(fp) if fp else None,
        "version": version,
        "dir": str(cur_dir),
        "default_dir": str(def_dir),
        "is_default": str(cur_dir).lower().rstrip("\\/") == str(def_dir).lower().rstrip("\\/"),
    }


def check_ffmpeg_update(current_version=None):
    """Checks GitHub for the latest release tag."""
    api_url = "https://api.github.com/repos/GyanD/codexffmpeg/releases/latest"
    try:
        req = urllib.request.Request(api_url, headers={"User-Agent": "MinimaxH3Workspace"})
        with urllib.request.urlopen(req, timeout=6) as r:
            data = json.loads(r.read().decode("utf-8"))
            tag = data.get("tag_name", "").lstrip("v")
            has_update = bool(current_version and tag and current_version.strip() != tag.strip())
            return {"ok": True, "latest_version": tag, "has_update": has_update, "current_version": current_version}
    except Exception as e:
        return {"ok": False, "latest_version": DEFAULT_RELEASE_VERSION, "has_update": False, "error": str(e)}


def probe_media(ffprobe_exe, file_path):
    """Probes video/audio properties using ffprobe."""
    if not ffprobe_exe or not Path(ffprobe_exe).is_file():
        return {}
    creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    cmd = [
        str(ffprobe_exe),
        "-v", "error",
        "-show_entries", "stream=index,codec_type,codec_name,width,height,r_frame_rate,duration",
        "-show_entries", "format=duration,size",
        "-of", "json",
        str(file_path)
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=10, creationflags=creationflags)
        data = json.loads(res.stdout) if res.stdout else {}
        info = {
            "duration": float(data.get("format", {}).get("duration") or 0.0),
            "size": int(data.get("format", {}).get("size") or 0),
            "video": None,
            "audio": None
        }
        for st in data.get("streams", []):
            if st.get("codec_type") == "video" and not info["video"]:
                w = int(st.get("width") or 0)
                h = int(st.get("height") or 0)
                fps_str = st.get("r_frame_rate", "24/1")
                fps = 24.0
                try:
                    num, den = fps_str.split("/")
                    fps = round(float(num) / float(den), 2) if float(den) > 0 else 24.0
                except Exception:
                    pass
                info["video"] = {
                    "width": w,
                    "height": h,
                    "codec": st.get("codec_name", "h264"),
                    "fps": fps,
                    "duration": float(st.get("duration") or info["duration"])
                }
            elif st.get("codec_type") == "audio" and not info["audio"]:
                info["audio"] = {
                    "codec": st.get("codec_name", "aac"),
                    "duration": float(st.get("duration") or info["duration"])
                }
        return info
    except Exception as e:
        print(f"[FFmpeg probe error] {file_path}: {e}", file=sys.stderr)
        return {}


def has_nvenc(ffmpeg_exe):
    """Checks whether the portable ffmpeg has hardware NVENC encoder support."""
    if not ffmpeg_exe:
        return False
    try:
        creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        res = subprocess.run([str(ffmpeg_exe), "-hide_banner", "-encoders"], capture_output=True, text=True, timeout=5, creationflags=creationflags)
        return "h264_nvenc" in (res.stdout or "")
    except Exception:
        return False


def download_and_extract_ffmpeg_sync(target_dir, progress_cb=None, cancel_event=None):
    """Synchronous worker that downloads portable FFmpeg zip and extracts it cleanly."""
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    temp_zip = target_dir / "_ffmpeg_download.tmp.zip"
    extract_dir = target_dir / "_extract_tmp"

    urls = [
        DEFAULT_RELEASE_URL,
        MIRROR_RELEASE_URL,
        BTBN_FALLBACK_URL
    ]

    last_err = None
    downloaded_ok = False

    for url in urls:
        if cancel_event and cancel_event.is_set():
            return {"ok": False, "error": "Отменено пользователем"}
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 MinimaxH3Workspace"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                total_bytes = int(resp.headers.get("Content-Length") or 0)
                downloaded = 0
                start_time = time.time()
                last_update = start_time

                with open(temp_zip, "wb") as f:
                    while True:
                        if cancel_event and cancel_event.is_set():
                            raise asyncio.CancelledError("Отменено")
                        chunk = resp.read(512 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)

                        now = time.time()
                        if progress_cb and (now - last_update >= 0.25 or downloaded == total_bytes):
                            last_update = now
                            elapsed = max(0.001, now - start_time)
                            speed_mb = (downloaded / (1024 * 1024)) / elapsed
                            pct = min(99, int((downloaded / total_bytes * 100))) if total_bytes > 0 else 50
                            progress_cb({
                                "status": "downloading",
                                "pct": pct,
                                "bytes": downloaded,
                                "total": total_bytes,
                                "speed_mb": round(speed_mb, 1)
                            })
                downloaded_ok = True
                break
        except Exception as e:
            last_err = e
            if temp_zip.exists():
                try:
                    temp_zip.unlink()
                except OSError:
                    pass
            continue

    if not downloaded_ok:
        return {"ok": False, "error": f"Не удалось скачать FFmpeg: {last_err}"}

    if cancel_event and cancel_event.is_set():
        if temp_zip.exists():
            temp_zip.unlink()
        return {"ok": False, "error": "Отменено пользователем"}

    # Extracting
    if progress_cb:
        progress_cb({"status": "extracting", "pct": 99, "bytes": 0, "total": 0, "speed_mb": 0.0})

    try:
        if extract_dir.exists():
            shutil.rmtree(extract_dir, ignore_errors=True)
        extract_dir.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(temp_zip, "r") as zf:
            zf.extractall(extract_dir)

        # Locate binaries inside the extracted archive
        found_ffmpeg = None
        found_ffprobe = None
        for p in extract_dir.rglob("ffmpeg.exe"):
            found_ffmpeg = p
            break
        for p in extract_dir.rglob("ffprobe.exe"):
            found_ffprobe = p
            break

        if not found_ffmpeg or not found_ffprobe:
            raise RuntimeError("В скачанном архиве не найдены ffmpeg.exe / ffprobe.exe")

        # Destination bin folder
        bin_dir = target_dir / "bin"
        bin_dir.mkdir(parents=True, exist_ok=True)

        # Source bin dir containing exe and any dlls
        src_bin = found_ffmpeg.parent
        for item in src_bin.iterdir():
            dst = bin_dir / item.name
            if item.is_file():
                shutil.copy2(item, dst)
            elif item.is_dir():
                if dst.exists():
                    shutil.rmtree(dst, ignore_errors=True)
                shutil.copytree(item, dst)

        # Also copy ffmpeg.exe and ffprobe.exe directly to target_dir root for convenience
        try:
            shutil.copy2(bin_dir / "ffmpeg.exe", target_dir / "ffmpeg.exe")
            shutil.copy2(bin_dir / "ffprobe.exe", target_dir / "ffprobe.exe")
        except Exception:
            pass

        # Cleanup
        try:
            temp_zip.unlink()
            shutil.rmtree(extract_dir, ignore_errors=True)
        except Exception:
            pass

        ff, fp = find_ffmpeg(target_dir)
        version = get_ffmpeg_version(ff)
        return {
            "ok": True,
            "version": version,
            "ffmpeg_path": str(ff),
            "ffprobe_path": str(fp),
            "dir": str(target_dir)
        }
    except Exception as e:
        return {"ok": False, "error": f"Ошибка распаковки: {e}"}


def run_concat(ffmpeg_exe, ffprobe_exe, clip_paths, output_path, audio_path=None, audio_opts=None, progress_cb=None, cancel_event=None):
    """Performs video montage / concatenation using portable FFmpeg.
    - If codecs and resolutions match and no audio track added: ultra-fast stream copy (-c copy).
    - If resolutions differ: Lanczos scale & pad to first clip's resolution + NVENC/x264 encode.
    - If background audio provided: mixes or attaches background music.
    - Reports percentage via progress_cb.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audio_opts = audio_opts or {}

    # 1. Probe all input clips
    probes = [probe_media(ffprobe_exe, p) for p in clip_paths]
    total_duration = sum((pr.get("duration") or 0.0) for pr in probes)
    if total_duration <= 0.0:
        total_duration = 5.0 * len(clip_paths)

    # Check if stream copy is possible
    can_stream_copy = True
    first_v = probes[0].get("video") if probes else None
    if not first_v or audio_path:
        can_stream_copy = False
    else:
        for pr in probes[1:]:
            v = pr.get("video")
            if not v:
                can_stream_copy = False
                break
            if v["width"] != first_v["width"] or v["height"] != first_v["height"] or v["codec"] != first_v["codec"]:
                can_stream_copy = False
                break

    creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0

    if can_stream_copy:
        # Fast path: Stream Copy via concat demuxer
        concat_txt = output_path.parent / f"_concat_{int(time.time()*1000)}.txt"
        try:
            with open(concat_txt, "w", encoding="utf-8") as f:
                for cp in clip_paths:
                    norm = str(Path(cp).resolve()).replace("\\", "/")
                    f.write(f"file '{norm}'\n")

            cmd = [
                str(ffmpeg_exe),
                "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", str(concat_txt),
                "-c", "copy",
                "-movflags", "+faststart",
                str(output_path)
            ]

            if progress_cb:
                progress_cb(50, "Быстрая склейка без потери качества...")

            res = subprocess.run(cmd, capture_output=True, text=True, timeout=120, creationflags=creationflags)
            if res.returncode != 0:
                raise RuntimeError(f"FFmpeg copy failed: {res.stderr[:300]}")

            if progress_cb:
                progress_cb(100, "Готово!")

            return {
                "ok": True,
                "fast_copy": True,
                "duration": round(total_duration, 1),
                "res": f"{first_v['width']}×{first_v['height']}"
            }
        finally:
            if concat_txt.exists():
                try:
                    concat_txt.unlink()
                except OSError:
                    pass

    # Re-encode path: Auto-scaling and/or Audio mixing
    target_w = first_v["width"] if first_v else 1280
    target_h = first_v["height"] if first_v else 720
    # ensure even dimensions
    target_w = (target_w // 2) * 2
    target_h = (target_h // 2) * 2

    inputs = []
    for cp in clip_paths:
        inputs.extend(["-i", str(Path(cp).resolve())])

    has_bg_audio = bool(audio_path and Path(audio_path).is_file())
    music_idx = len(clip_paths)
    if has_bg_audio:
        loop_music = audio_opts.get("loop", True)
        if loop_music:
            inputs.extend(["-stream_loop", "-1", "-i", str(Path(audio_path).resolve())])
        else:
            inputs.extend(["-i", str(Path(audio_path).resolve())])

    # Build filter_complex
    filter_parts = []
    # Video scale & pad
    for i in range(len(clip_paths)):
        filter_parts.append(
            f"[{i}:v]scale={target_w}:{target_h}:force_original_aspect_ratio=decrease,"
            f"pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=24[v{i}]"
        )

    # Check which clips have audio
    clips_have_audio = any(pr.get("audio") for pr in probes)
    n = len(clip_paths)

    if clips_have_audio:
        for i in range(n):
            if probes[i].get("audio"):
                filter_parts.append(f"[{i}:a]aformat=sample_rates=44100:channel_layouts=stereo[a{i}]")
            else:
                # generate silent audio if clip lacks audio
                dur = probes[i].get("duration") or 5.0
                filter_parts.append(f"anullsrc=r=44100:cl=stereo,atrim=0:{dur}[a{i}]")

        v_cat_in = "".join(f"[v{i}][a{i}]" for i in range(n))
        filter_parts.append(f"{v_cat_in}concat=n={n}:v=1:a=1[vcat][acat]")
    else:
        v_cat_in = "".join(f"[v{i}]" for i in range(n))
        filter_parts.append(f"{v_cat_in}concat=n={n}:v=1:a=0[vcat]")

    map_v = "[vcat]"
    map_a = None

    if has_bg_audio:
        music_vol = float(audio_opts.get("music_vol", 0.8))
        video_vol = float(audio_opts.get("video_vol", 1.0))
        audio_mode = audio_opts.get("mode", "mix")  # "mix", "replace", or "music_only"

        if clips_have_audio and audio_mode == "mix":
            filter_parts.append(f"[{music_idx}:a]volume={music_vol}[mus];[acat]volume={video_vol}[vid_a];[vid_a][mus]amix=inputs=2:duration=first:dropout_transition=2[aout]")
            map_a = "[aout]"
        else:
            filter_parts.append(f"[{music_idx}:a]volume={music_vol}[aout]")
            map_a = "[aout]"
    elif clips_have_audio:
        map_a = "[acat]"

    filter_complex = ";".join(filter_parts)

    use_nvenc = has_nvenc(ffmpeg_exe)
    vcodec_args = ["-c:v", "h264_nvenc", "-preset", "p5", "-cq", "19", "-b:v", "0"] if use_nvenc else ["-c:v", "libx264", "-preset", "fast", "-crf", "19"]

    cmd = [str(ffmpeg_exe), "-y"] + inputs + [
        "-filter_complex", filter_complex,
        "-map", map_v
    ]
    if map_a:
        cmd.extend(["-map", map_a, "-c:a", "aac", "-b:a", "192k"])
    else:
        cmd.extend(["-an"])

    cmd.extend(vcodec_args)
    cmd.extend([
        "-t", str(round(total_duration, 2)),
        "-movflags", "+faststart",
        "-progress", "pipe:1",
        str(output_path)
    ])

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1, creationflags=creationflags)

    last_pct = 0
    while True:
        if cancel_event and cancel_event.is_set():
            proc.kill()
            if output_path.exists():
                output_path.unlink()
            return {"ok": False, "error": "Отменено пользователем"}

        line = proc.stdout.readline()
        if not line and proc.poll() is not None:
            break
        line = line.strip()
        if line.startswith("out_time_ms="):
            try:
                ms = int(line.split("=")[1])
                cur_sec = ms / 1_000_000
                pct = min(99, int((cur_sec / total_duration) * 100))
                if pct > last_pct and progress_cb:
                    last_pct = pct
                    progress_cb(pct, f"Монтаж FFmpeg: {pct}%")
            except Exception:
                pass
        elif line == "progress=end":
            if progress_cb:
                progress_cb(100, "Готово!")

    proc.wait()
    if proc.returncode != 0:
        err = proc.stderr.read()[:500] if proc.stderr else "Unknown error"
        raise RuntimeError(f"FFmpeg execution failed (code {proc.returncode}): {err}")

    return {
        "ok": True,
        "fast_copy": False,
        "duration": round(total_duration, 1),
        "res": f"{target_w}×{target_h}"
    }
