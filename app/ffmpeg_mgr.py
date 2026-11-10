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
import threading
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


def parse_version_tuple(v_str):
    """Extracts numeric version tuple from version string, e.g. '9.0.2-essentials...' -> (9, 0, 2)."""
    if not v_str:
        return (0,)
    m = re.search(r"(\d+(?:\.\d+)+)", str(v_str))
    if m:
        return tuple(map(int, m.group(1).split(".")))
    digits = re.findall(r"\d+", str(v_str))
    return tuple(map(int, digits)) if digits else (0,)


def get_ffmpeg_version(ffmpeg_exe):
    """Executes ffmpeg.exe -version and parses clean version string."""
    if not ffmpeg_exe or not Path(ffmpeg_exe).is_file():
        return None
    try:
        creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        res = subprocess.run([str(ffmpeg_exe), "-version"], capture_output=True, text=True, timeout=5, creationflags=creationflags)
        first_line = res.stdout.splitlines()[0] if res.stdout else ""
        m_ver = re.search(r"ffmpeg version\s+[vn]?(\d+(?:\.\d+)+)", first_line, re.I)
        if m_ver:
            return m_ver.group(1)
        m = re.search(r"ffmpeg version\s+([^\s]+)", first_line, re.I)
        if m:
            token = m.group(1)
            prefix = token.split("-")[0].lstrip("vn")
            return prefix if prefix and any(c.isdigit() for c in prefix) else token
        return first_line[:30]
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
            tag = data.get("tag_name", "").lstrip("vn").strip()
            v_cur = parse_version_tuple(current_version)
            v_tag = parse_version_tuple(tag)
            has_update = bool(current_version and tag and v_tag > v_cur)
            clean_cur = ".".join(map(str, v_cur)) if v_cur != (0,) else (current_version or "")
            return {
                "ok": True,
                "latest_version": tag,
                "has_update": has_update,
                "current_version": clean_cur,
                "url": data.get("html_url") or "https://github.com/GyanD/codexffmpeg/releases",
            }
    except Exception as e:
        v_cur = parse_version_tuple(current_version)
        clean_cur = ".".join(map(str, v_cur)) if v_cur != (0,) else (current_version or "")
        return {"ok": False, "latest_version": DEFAULT_RELEASE_VERSION, "has_update": False, "current_version": clean_cur, "error": str(e)}


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


def run_ffmpeg_with_progress(cmd, total_duration, output_path, progress_cb=None, cancel_event=None, creationflags=0, label="Монтаж FFmpeg"):
    """Runs FFmpeg process while draining stderr in background to completely avoid pipe buffer deadlocks,
    and parses stdout (-progress pipe:1) to report smooth real-time percentage to the UI."""
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        creationflags=creationflags
    )

    stderr_lines = []
    def drain_stderr():
        try:
            for l in iter(proc.stderr.readline, ''):
                if not l:
                    break
                stderr_lines.append(l)
                if len(stderr_lines) > 100:
                    stderr_lines.pop(0)
        except Exception:
            pass

    t_err = threading.Thread(target=drain_stderr, daemon=True)
    t_err.start()

    last_pct = 5
    if progress_cb:
        progress_cb(5, f"{label}: запуск...")

    try:
        while True:
            if cancel_event and cancel_event.is_set():
                proc.kill()
                if output_path and Path(output_path).exists():
                    try:
                        Path(output_path).unlink()
                    except OSError:
                        pass
                return {"ok": False, "error": "Отменено пользователем", "stderr": ""}

            line = proc.stdout.readline()
            if not line and proc.poll() is not None:
                break
            if not line:
                continue
            line = line.strip()

            cur_sec = None
            if line.startswith("out_time_ms="):
                try:
                    ms = int(line.split("=")[1])
                    cur_sec = ms / 1_000_000 if ms > 100_000 else ms / 1000
                except (ValueError, IndexError):
                    pass
            elif line.startswith("out_time_us="):
                try:
                    us = int(line.split("=")[1])
                    cur_sec = us / 1_000_000
                except (ValueError, IndexError):
                    pass
            elif line.startswith("out_time="):
                try:
                    parts = line.split("=")[1].split(":")
                    if len(parts) == 3:
                        h, m, s = float(parts[0]), float(parts[1]), float(parts[2])
                        cur_sec = h * 3600 + m * 60 + s
                except (ValueError, IndexError):
                    pass

            if cur_sec is not None and total_duration > 0:
                pct = min(99, max(5, int((cur_sec / total_duration) * 100)))
                if pct > last_pct and progress_cb:
                    last_pct = pct
                    progress_cb(pct, f"{label}: {pct}%")
            elif line == "progress=end":
                if progress_cb:
                    progress_cb(100, f"{label}: готово!")
    finally:
        pass

    proc.wait()
    t_err.join(timeout=1.0)
    full_err = "".join(stderr_lines).strip()

    if proc.returncode != 0:
        return {"ok": False, "error": f"FFmpeg code {proc.returncode}: {full_err[-600:]}", "stderr": full_err}

    return {"ok": True, "stderr": full_err}


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
    if use_nvenc:
        vcodec_args = ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "20", "-b:v", "0", "-pix_fmt", "yuv420p"]
    else:
        vcodec_args = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p"]

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

    res_run = run_ffmpeg_with_progress(
        cmd,
        total_duration,
        output_path,
        progress_cb=progress_cb,
        cancel_event=cancel_event,
        creationflags=creationflags,
        label="Монтаж FFmpeg"
    )
    if not res_run["ok"]:
        raise RuntimeError(res_run.get("error", "Ошибка выполнения FFmpeg"))

    return {
        "ok": True,
        "fast_copy": False,
        "duration": round(total_duration, 1),
        "res": f"{target_w}×{target_h}"
    }


def run_studio_montage(ffmpeg_exe, ffprobe_exe, clip_paths, output_path, options=None, progress_cb=None, cancel_event=None):
    """Full-featured FFmpeg video studio montage engine:
    - Transitions via xfade & acrossfade (fade, wipe, slide, circle, zoom, dissolve);
    - Aspect ratio conversion (16:9, 9:16, 1:1, 21:9) with smart blurred background, crop or pad;
    - Speed control (0.5x, 0.75x, 1.0x, 1.25x, 1.5x, 2.0x);
    - Color grading (brightness, contrast, saturation) and filters (B&W, vignette, warm, cold, vintage);
    - Text title / subtitle overlay with customizable position, size, and styling;
    - Advanced audio mixing: background music, separate volumes, fade in/out, looping;
    - NVIDIA NVENC hardware acceleration with fallback to libx264.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    options = options or {}

    cuts = options.get("cuts") or []
    speed = float(options.get("speed") or 1.0)
    if speed <= 0.1 or speed > 5.0:
        speed = 1.0

    transition = str(options.get("transition") or "none").strip().lower()
    transition_dur = float(options.get("transition_dur") or 0.5)
    aspect = str(options.get("aspect") or "original").strip().lower()
    fit_mode = str(options.get("fit_mode") or "blur").strip().lower()
    color = options.get("color") or {}
    title = options.get("title") or {}
    audio_path = options.get("audio_path")
    audio_opts = options.get("audio_opts") or {}

    creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0

    # 1. Probe all input clips
    probes = [probe_media(ffprobe_exe, p) for p in clip_paths]
    first_v = probes[0].get("video") if probes else None

    # Calculate raw & effective durations
    raw_durs = []
    eff_durs = []
    for i, pr in enumerate(probes):
        d = float(pr.get("duration") or 5.0)
        raw_durs.append(d)
        c = float(cuts[i]) if (i < len(cuts) and cuts[i] is not None) else None
        if c is not None and 0.2 < c < d:
            eff_d = c
        else:
            eff_d = d
        eff_d = eff_d / speed
        eff_durs.append(eff_d)

    n = len(clip_paths)
    if n == 0:
        raise ValueError("Нет клипов для монтажа")

    # Target resolution
    if aspect == "16:9":
        tw, th = 1920, 1080
    elif aspect == "9:16":
        tw, th = 1080, 1920
    elif aspect == "1:1":
        tw, th = 1080, 1080
    elif aspect == "21:9":
        tw, th = 2560, 1080
    else:
        tw = first_v["width"] if first_v else 1280
        th = first_v["height"] if first_v else 720
    tw = (tw // 2) * 2
    th = (th // 2) * 2

    # Fast stream copy check
    can_stream_copy = (
        transition == "none"
        and speed == 1.0
        and aspect == "original"
        and not audio_path
        and color.get("filter") in (None, "none")
        and float(color.get("brightness", 0)) == 0.0
        and float(color.get("contrast", 1.0)) == 1.0
        and float(color.get("saturation", 1.0)) == 1.0
        and not (title.get("text") or "").strip()
        and not any(cuts[i] is not None and 0.2 < cuts[i] < raw_durs[i] for i in range(len(cuts)))
    )
    if can_stream_copy:
        for pr in probes:
            v = pr.get("video")
            if not v or v["width"] != tw or v["height"] != th or v["codec"] != first_v["codec"]:
                can_stream_copy = False
                break

    if can_stream_copy:
        return run_concat(ffmpeg_exe, ffprobe_exe, clip_paths, output_path, progress_cb=progress_cb, cancel_event=cancel_event)

    # Studio filter graph construction
    use_nvenc = has_nvenc(ffmpeg_exe)
    inputs = []
    for cp in clip_paths:
        if use_nvenc:
            inputs.extend(["-hwaccel", "auto"])
        inputs.extend(["-i", str(Path(cp).resolve())])

    has_bg_audio = bool(audio_path and Path(audio_path).is_file())
    music_idx = len(clip_paths)
    if has_bg_audio:
        loop_music = audio_opts.get("loop", True)
        if loop_music:
            inputs.extend(["-stream_loop", "-1", "-i", str(Path(audio_path).resolve())])
        else:
            inputs.extend(["-i", str(Path(audio_path).resolve())])

    filter_parts = []
    fps = 24

    for i in range(n):
        v_pre = []
        c = float(cuts[i]) if (i < len(cuts) and cuts[i] is not None) else None
        if c is not None and 0.2 < c < raw_durs[i]:
            v_pre.append(f"trim=0:{c:0.3f},setpts=PTS-STARTPTS")
        if speed != 1.0:
            v_pre.append(f"setpts=(1.0/{speed:0.3f})*PTS")
        v_pre_str = (",".join(v_pre) + ",") if v_pre else ""

        # Framing / Scaling
        clip_v = probes[i].get("video")
        cw = clip_v["width"] if clip_v else tw
        ch = clip_v["height"] if clip_v else th
        same_aspect = abs((cw / ch) - (tw / th)) < 0.02

        if fit_mode == "blur" and not same_aspect:
            bw, bh = 160, max(90, int(160 * (th / tw)))
            bw = (bw // 2) * 2
            bh = (bh // 2) * 2
            filter_parts.append(
                f"[{i}:v]{v_pre_str}split[bg_in{i}][fg_in{i}];"
                f"[bg_in{i}]scale={bw}:{bh}:force_original_aspect_ratio=increase,crop={bw}:{bh},boxblur=2:1,scale={tw}:{th}:flags=fast_bilinear[bg{i}];"
                f"[fg_in{i}]scale={tw}:{th}:force_original_aspect_ratio=decrease,setsar=1[fg{i}];"
                f"[bg{i}][fg{i}]overlay=(W-w)/2:(H-h)/2,fps={fps},format=yuv420p,settb=AVTB[v_prep{i}]"
            )
        elif fit_mode == "crop" and not same_aspect:
            filter_parts.append(
                f"[{i}:v]{v_pre_str}scale={tw}:{th}:force_original_aspect_ratio=increase,crop={tw}:{th},setsar=1,fps={fps},format=yuv420p,settb=AVTB[v_prep{i}]"
            )
        elif not same_aspect:
            filter_parts.append(
                f"[{i}:v]{v_pre_str}scale={tw}:{th}:force_original_aspect_ratio=decrease,pad={tw}:{th}:(ow-iw)/2:(oh-ih)/2:black,setsar=1,fps={fps},format=yuv420p,settb=AVTB[v_prep{i}]"
            )
        else:
            filter_parts.append(
                f"[{i}:v]{v_pre_str}scale={tw}:{th},setsar=1,fps={fps},format=yuv420p,settb=AVTB[v_prep{i}]"
            )

        # Audio prep
        d_i = eff_durs[i]
        if probes[i].get("audio"):
            a_filters = ["aformat=sample_rates=44100:channel_layouts=stereo"]
            if c is not None and 0.2 < c < raw_durs[i]:
                a_filters.append(f"atrim=0:{c:0.3f}")
            if speed != 1.0:
                at = max(0.5, min(2.0, speed))
                a_filters.append(f"atempo={at:0.3f}")
            a_filters.append("asetpts=PTS-STARTPTS")
            filter_parts.append(f"[{i}:a]{','.join(a_filters)}[a_prep{i}]")
        else:
            filter_parts.append(f"anullsrc=r=44100:cl=stereo,atrim=0:{d_i:0.3f},asetpts=PTS-STARTPTS[a_prep{i}]")

    # Transitions
    min_eff_dur = min(eff_durs) if eff_durs else 1.0
    actual_tdur = min(transition_dur, max(0.2, min_eff_dur / 2.1))

    if transition != "none" and n >= 2:
        cur_v = "v_prep0"
        cur_a = "a_prep0"
        accum_dur = eff_durs[0]

        for k in range(1, n):
            off = accum_dur - actual_tdur
            next_v = f"v_x{k}"
            next_a = f"a_x{k}"
            filter_parts.append(f"[{cur_v}][v_prep{k}]xfade=transition={transition}:duration={actual_tdur:0.3f}:offset={off:0.3f}[{next_v}]")
            filter_parts.append(f"[{cur_a}][a_prep{k}]acrossfade=d={actual_tdur:0.3f}:c1=tri:c2=tri[{next_a}]")
            cur_v = next_v
            cur_a = next_a
            accum_dur += eff_durs[k] - actual_tdur
        total_duration = accum_dur
    elif n == 1:
        cur_v = "v_prep0"
        cur_a = "a_prep0"
        total_duration = eff_durs[0]
    else:
        v_cat_in = "".join(f"[v_prep{i}][a_prep{i}]" for i in range(n))
        filter_parts.append(f"{v_cat_in}concat=n={n}:v=1:a=1[v_cat][a_cat]")
        cur_v = "v_cat"
        cur_a = "a_cat"
        total_duration = sum(eff_durs)

    # Color grading
    v_effects = []
    b = float(color.get("brightness", 0.0))
    c = float(color.get("contrast", 1.0))
    s = float(color.get("saturation", 1.0))
    if b != 0.0 or c != 1.0 or s != 1.0:
        v_effects.append(f"eq=brightness={b:0.3f}:contrast={c:0.3f}:saturation={s:0.3f}")

    flt = str(color.get("filter") or "none").lower()
    if flt == "bw":
        v_effects.append("hue=s=0")
    elif flt == "vignette":
        v_effects.append("vignette=PI/4")
    elif flt == "warm":
        v_effects.append("colorbalance=rs=0.12:gs=0.04:bs=-0.08")
    elif flt == "cold":
        v_effects.append("colorbalance=rs=-0.08:gs=0.0:bs=0.15")
    elif flt == "vintage":
        v_effects.append("colorchannelmixer=.393:.769:.189:0:.349:.686:.168:0:.272:.534:.131")

    if v_effects:
        filter_parts.append(f"[{cur_v}]{','.join(v_effects)}[v_col]")
        cur_v = "v_col"

    # Text overlay
    text = (title.get("text") or "").strip()
    if text:
        clean_text = text.replace("'", "\\'").replace(":", "\\:").replace("%", "\\%")
        pos = title.get("position", "bottom")
        if pos == "top":
            xy = "x=(w-text_w)/2:y=60"
        elif pos == "center":
            xy = "x=(w-text_w)/2:y=(h-text_h)/2"
        else:
            xy = "x=(w-text_w)/2:y=h-text_h-60"

        sz_map = {"small": 30, "medium": 44, "large": 64}
        fontsize = sz_map.get(title.get("size", "medium"), 44)
        col = title.get("color", "white").lower()
        font_arg = "fontfile='C\\:/Windows/Fonts/arial.ttf':" if sys.platform == "win32" and Path("C:/Windows/Fonts/arial.ttf").exists() else ""
        filter_parts.append(
            f"[{cur_v}]drawtext={font_arg}text='{clean_text}':fontsize={fontsize}:fontcolor={col}:box=1:boxcolor=black@0.6:boxborderw=10:{xy}[v_txt]"
        )
        cur_v = "v_txt"

    # Audio Mixing
    map_a = cur_a
    if has_bg_audio:
        music_vol = float(audio_opts.get("music_vol", 0.8))
        video_vol = float(audio_opts.get("video_vol", 1.0))
        audio_mode = audio_opts.get("mode", "mix")
        fade = audio_opts.get("fade", True)

        m_filters = [f"volume={music_vol:0.3f}"]
        if fade and total_duration > 3.0:
            m_filters.append(f"afade=t=in:ss=0:d=1.5,afade=t=out:st={total_duration-1.5:0.3f}:d=1.5")
        filter_parts.append(f"[{music_idx}:a]{','.join(m_filters)}[mus]")

        if audio_mode == "mix":
            filter_parts.append(f"[{cur_a}]volume={video_vol:0.3f}[vid_a];[vid_a][mus]amix=inputs=2:duration=first:dropout_transition=2[aout]")
            map_a = "aout"
        elif audio_mode == "replace":
            map_a = "mus"
        else:
            map_a = cur_a

    filter_complex = ";".join(filter_parts)

    if use_nvenc:
        vcodec_args = ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "20", "-b:v", "0", "-pix_fmt", "yuv420p"]
    else:
        vcodec_args = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p"]

    cmd = [str(ffmpeg_exe), "-y"] + inputs + [
        "-filter_complex", filter_complex,
        "-map", f"[{cur_v}]",
        "-map", f"[{map_a}]",
        "-c:a", "aac", "-b:a", "192k"
    ]
    cmd.extend(vcodec_args)
    cmd.extend([
        "-t", str(round(total_duration, 2)),
        "-movflags", "+faststart",
        "-progress", "pipe:1",
        str(output_path)
    ])

    res_run = run_ffmpeg_with_progress(
        cmd,
        total_duration,
        output_path,
        progress_cb=progress_cb,
        cancel_event=cancel_event,
        creationflags=creationflags,
        label="Студия FFmpeg"
    )
    if not res_run["ok"]:
        raise RuntimeError(res_run.get("error", "Ошибка монтажа в Студии FFmpeg"))

    return {
        "ok": True,
        "fast_copy": False,
        "duration": round(total_duration, 1),
        "res": f"{tw}×{th}"
    }
