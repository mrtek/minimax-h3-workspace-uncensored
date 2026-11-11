"""ComfyUI API graphs for Minimax H3 Workspace modes.

Each builder returns (graph, stages): the API-format prompt and a map
node_id -> UI stage name, so progress can be shown in plain words.
The graphs mirror the user's workflows in the project root.
"""

import math

FPS = 24

# Model files come in as `m`: role -> name as ComfyUI lists it (see catalog.py).

ASPECTS = {"16:9": (16, 9), "9:16": (9, 16), "1:1": (1, 1), "4:3": (4, 3), "3:4": (3, 4), "21:9": (21, 9)}
# sizes of the original workflow's table (16:9 base, multiples of 32); the four old ids are kept for saved settings
QUALITY = {"0.2": 608 * 352, "0.3": 736 * 416, "480p": 864 * 480, "0.5": 960 * 544, "0.6": 1056 * 608, "0.7": 1152 * 640,
           "0.8": 1216 * 672, "720p": 1280 * 736, "HD+": 1376 * 768, "1.2": 1504 * 832, "1.5": 1664 * 928,
           "1.8": 1824 * 1024, "1080p": 1920 * 1088}


def r32(v):
    return max(32, round(v / 32) * 32)


def resolution(aspect, quality, form=None):
    """Size from the format + quality, or the user's own width/height (rounded to a multiple of 32)."""
    if form and str(form.get("width", "")).isdigit() and str(form.get("height", "")).isdigit():
        cw, ch = int(form["width"]), int(form["height"])
        if cw >= 64 and ch >= 64:
            return min(2048, r32(cw)), min(2048, r32(ch))
    w, h = ASPECTS[aspect]
    px = QUALITY[quality]
    return r32(math.sqrt(px * w / h)), r32(math.sqrt(px * h / w))


def frames(seconds):
    # same formula as the workflow's ComfyMathExpression: 24 fps, length = 17k + 5
    f = max(5, round(seconds * FPS))
    return f + (5 - f % 17) % 17


def video_graph(m, prompt, width, height, seconds, seed, first=None, last=None, extra_loras=(), fast=None):
    """MINIMAXH3_FASTvideo_I2V: text / first frame / first + last frame."""
    g = {
        "1": {"class_type": "UNETLoader", "inputs": {
            "unet_name": m["fast_unet"], "weight_dtype": "default"}},
        "2": {"class_type": "MiniMaxH3SigmaShift", "inputs": {
            "model": ["1", 0], "shift_video": 10, "shift_audio": 3}},
        "3": {"class_type": "ModelAttentionBackend", "inputs": {
            "model": ["2", 0], "attention": "comfy kitchen attention"}},
        "4": {"class_type": "BlockSparseAttention", "inputs": {
            "model": ["3", 0], "selection": "vsa", "selection.keep_percent": 10.0,
            "start_percent": 0.2, "end_percent": 1.0, "dense_blocks": "",
            "min_tokens": 12288, "extra_tokens": 256,
            "sink_conditioning": "exact_kv_and_rows", "verbose": False}},
        "5": {"class_type": "CLIPLoader", "inputs": {
            "clip_name": m["text_encoder"], "type": "minimax", "device": "default"}},
        "6": {"class_type": "VAELoader", "inputs": {"vae_name": m["video_vae"]}},
        "7": {"class_type": "VAELoader", "inputs": {"vae_name": m["audio_vae"]}},
        "8": {"class_type": "MiniMaxH3ImageToVideo", "inputs": {
            "clip": ["5", 0], "vae": ["6", 0], "prompt": prompt,
            "width": width, "height": height, "length": frames(seconds)}},
        "9": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "10": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "res_multistep"}},
        "11": {"class_type": "BasicScheduler", "inputs": {
            "model": ["4", 0], "scheduler": "simple", "steps": 8, "denoise": 1.0}},
        "12": {"class_type": "BasicGuider", "inputs": {"model": ["4", 0], "conditioning": ["8", 0]}},
        "13": {"class_type": "SamplerCustomAdvanced", "inputs": {
            "noise": ["9", 0], "guider": ["12", 0], "sampler": ["10", 0],
            "sigmas": ["11", 0], "latent_image": ["8", 1]}},
        "14": {"class_type": "VAEDecode", "inputs": {"samples": ["13", 0], "vae": ["6", 0]}},
        "15": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["13", 0], "vae": ["7", 0]}},
        "16": {"class_type": "CreateVideo", "inputs": {
            "images": ["14", 0], "audio": ["15", 0], "fps": float(FPS),
            "bit_depth": "auto", "color_space": "sRGB", "codec": "none"}},
        "17": {"class_type": "SaveVideo", "inputs": {
            "video": ["16", 0], "filename_prefix": "H3Workspace/video",
            "format": "auto", "format.codec": "auto"}},
    }
    for slot, (load_id, scale_id, image) in {
        "first_frame": ("20", "21", first), "last_frame": ("22", "23", last)}.items():
        if not image:
            continue
        g[load_id] = {"class_type": "LoadImage", "inputs": {"image": image}}
        g[scale_id] = {"class_type": "ImageScale", "inputs": {
            "image": [load_id, 0], "upscale_method": "lanczos",
            "width": width, "height": height, "crop": "center"}}
        g["8"]["inputs"][slot] = [scale_id, 0]
    loras = ([(fast, 1.0)] if fast else []) + list(extra_loras)  # Turbo-SLA first, then the user's own LoRAs
    if loras:
        g["2"]["inputs"]["model"] = _lora_chain(g, ["1", 0], loras)
    if fast:  # "fast draft": Turbo-SLA 4 steps, flow shifts 6 / 3; block-sparse attention is switched OFF (it left square
        # mosaic artifacts on busy scenes: debris, explosions), dense attention is cleaner and still about 1.3x faster
        g["2"]["inputs"].update(shift_video=6, shift_audio=3)
        g["11"]["inputs"]["model"] = ["3", 0]
        g["12"]["inputs"]["model"] = ["3", 0]
        g.pop("4", None)
        g["11"]["inputs"]["steps"] = 4

    stages = {n: "load" for n in ("1", "2", "3", "4", "5", "6", "7")}
    stages.update({n: "encode" for n in ("8", "20", "21", "22", "23")})
    stages.update({"13": "sample", "14": "decode", "15": "decode", "16": "save", "17": "save"})
    return g, stages


def _refs(g, cond_node, images, videos=(), audios=(), base=100, vsize=None):
    """LoadImage / VHS_LoadVideo (24 fps) / LoadAudio nodes wired into MiniMaxH3ReferenceToVideo."""
    inputs = g[cond_node]["inputs"]
    for i, name in enumerate(images[:9]):
        nid = str(base + i)
        g[nid] = {"class_type": "LoadImage", "inputs": {"image": name}}
        inputs[f"ref_images.ref_image_{i}"] = [nid, 0]
    for i, name in enumerate(videos[:3]):
        nid = str(base + 10 + i)
        g[nid] = {"class_type": "VHS_LoadVideo", "inputs": {
            "video": name, "force_rate": 24, "custom_width": vsize[0] if vsize else 0, "custom_height": vsize[1] if vsize else 0,
            "frame_load_cap": 360, "skip_first_frames": 0, "select_every_nth": 1, "format": "None"}}
        inputs[f"ref_videos.ref_video_{i}"] = [nid, 0]
    for i, name in enumerate(audios[:3]):
        nid = str(base + 20 + i)
        g[nid] = {"class_type": "LoadAudio", "inputs": {"audio": name}}
        inputs[f"ref_audios.ref_audio_{i}"] = [nid, 0]
    return [str(base + i) for i in range(len(images[:9]))] + [str(base + 10 + i) for i in range(len(videos[:3]))] \
        + [str(base + 20 + i) for i in range(len(audios[:3]))]


def _lora_chain(g, start, loras, base=60):
    """Stack LoraLoaderModelOnly nodes; returns the last model output."""
    prev = start
    for i, (name, strength) in enumerate(loras):
        nid = str(base + i)
        g[nid] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": prev, "lora_name": name, "strength_model": strength}}
        prev = [nid, 0]
    return prev


def _sdxl_lora_chain(g, model_start, clip_start, loras, base=60):
    """Stack LoraLoader nodes (patching both model and clip) for SDXL and FLUX."""
    m_prev, c_prev = model_start, clip_start
    for i, (name, strength) in enumerate(loras):
        nid = str(base + i)
        g[nid] = {"class_type": "LoraLoader", "inputs": {
            "model": m_prev, "clip": c_prev, "lora_name": name,
            "strength_model": strength, "strength_clip": strength}}
        m_prev, c_prev = [nid, 0], [nid, 1]
    return m_prev, c_prev


def ref_graph(m, prompt, width, height, seconds, seed, images, videos, audios, steps=6,
              ref_size="max", extra_loras=(), pika=None):
    """MinimaxH3_Ref to Video (ULTRA SPEED Singularity 4-step), single pass.

    The original samples with T8's Dual Clock node (shift video 12 / audio 4, beta);
    here the same shifts go through ComfyUI's built-in MiniMaxH3SigmaShift + beta scheduler.
    """
    g = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": m["lipsync_unet"], "weight_dtype": "default"}},
        "2": {"class_type": "ModelAttentionBackend", "inputs": {"model": ["1", 0], "attention": "comfy kitchen attention"}},
        "4": {"class_type": "CLIPLoader", "inputs": {"clip_name": m["text_encoder"], "type": "minimax", "device": "default"}},
        "5": {"class_type": "VAELoader", "inputs": {"vae_name": m["video_vae"]}},
        "6": {"class_type": "VAELoader", "inputs": {"vae_name": m["audio_vae"]}},
        "8": {"class_type": "MiniMaxH3ReferenceToVideo", "inputs": {
            "clip": ["4", 0], "vae": ["5", 0], "audio_vae": ["6", 0], "prompt": prompt,
            "width": width, "height": height, "length": frames(seconds), "ref_image_size": ref_size}},
    }
    if pika:  # Pika H3 Flex Turbo (8 steps, HD): its own LoRA instead of the 4-step one; no sigma shift, scheduler "simple"
        mref = _lora_chain(g, ["2", 0], list(extra_loras) + [(pika, 1.0)])
        steps, scheduler = max(6, steps), "simple"
    else:
        model = _lora_chain(g, ["2", 0], list(extra_loras) + [(m["ref2v_turbo_lora"], 0.7)])
        g["9"] = {"class_type": "MiniMaxH3SigmaShift", "inputs": {"model": model, "shift_video": 12.0, "shift_audio": 4.0}}
        mref, scheduler = ["9", 0], "beta"
    g.update({
        "10": {"class_type": "BasicScheduler", "inputs": {"model": mref, "scheduler": scheduler, "steps": steps, "denoise": 1.0}},
        "11": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "12": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "13": {"class_type": "BasicGuider", "inputs": {"model": mref, "conditioning": ["8", 0]}},
        "14": {"class_type": "SamplerCustomAdvanced", "inputs": {
            "noise": ["11", 0], "guider": ["13", 0], "sampler": ["12", 0], "sigmas": ["10", 0], "latent_image": ["8", 1]}},
        "15": {"class_type": "VAEDecode", "inputs": {"samples": ["14", 1], "vae": ["5", 0]}},
        "16": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["14", 1], "vae": ["6", 0]}},
        "17": {"class_type": "CreateVideo", "inputs": {
            "images": ["15", 0], "audio": ["16", 0], "fps": float(FPS), "bit_depth": "auto", "color_space": "sRGB", "codec": "none"}},
        "18": {"class_type": "SaveVideo", "inputs": {
            "video": ["17", 0], "filename_prefix": "H3Workspace/refs", "format": "auto", "format.codec": "auto"}},
    })
    loaders = _refs(g, "8", images, videos, audios)
    stages = {n: "load" for n in ["1", "2", "4", "5", "6"] + [k for k in g if k.isdigit() and 60 <= int(k) < 70]}
    stages.update({n: "encode" for n in ["8"] + loaders})
    stages.update({"14": "sample", "15": "decode", "16": "decode", "17": "save", "18": "save"})
    return g, stages


# MiniMax_H3_Direct-ai-Actor: one pass, 10 steps with the author's own sigma list
ACTOR_SIGMAS = "1, .9882, .973, .9524, .9231, .878, .8, .6316, .4737, .1579, 0"


def actor_graph(m, prompt, width, height, seconds, seed, images, audios, realism=None, extra_loras=()):
    """Acting scenes: first frame + a voice sample, new speech written in the prompt (<d>...</d>).

    Same Singularity Ref2V model as the references mode, but with sol-attn sparse attention,
    sigma shift 12.2 / 3, er_sde and the turbo LoRA at full strength, as in the original workflow.
    `realism` is the (file, strength) of the people-realism LoRA when it is on.
    """
    g = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": m["lipsync_unet"], "weight_dtype": "default"}},
        "2": {"class_type": "ModelAttentionBackend", "inputs": {"model": ["1", 0], "attention": "comfy kitchen attention"}},
        "3": {"class_type": "BlockSparseAttention", "inputs": {
            "model": ["2", 0], "selection": "sol-attn", "selection.tau": 1.3, "start_percent": 0.2, "end_percent": 1.0,
            "dense_blocks": "", "min_tokens": 12288, "extra_tokens": 256, "sink_conditioning": "exact_kv_and_rows",
            "verbose": False}},
        "4": {"class_type": "CLIPLoader", "inputs": {"clip_name": m["text_encoder"], "type": "minimax", "device": "default"}},
        "5": {"class_type": "VAELoader", "inputs": {"vae_name": m["video_vae"]}},
        "6": {"class_type": "VAELoader", "inputs": {"vae_name": m["audio_vae"]}},
        "7": {"class_type": "MiniMaxH3SigmaShift", "inputs": {"model": ["3", 0], "shift_video": 12.2, "shift_audio": 3.0}},
        "8": {"class_type": "MiniMaxH3ReferenceToVideo", "inputs": {
            "clip": ["4", 0], "vae": ["5", 0], "audio_vae": ["6", 0], "prompt": prompt,
            "width": width, "height": height, "length": frames(seconds), "ref_image_size": "match"}},
    }
    loras = [(m["ref2v_turbo_lora"], 1.0)] + ([realism] if realism else []) + list(extra_loras)
    model = _lora_chain(g, ["7", 0], loras)
    g.update({
        "10": {"class_type": "ManualSigmas", "inputs": {"sigmas": ACTOR_SIGMAS}},
        "11": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "12": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "er_sde"}},
        "13": {"class_type": "BasicGuider", "inputs": {"model": model, "conditioning": ["8", 0]}},
        "14": {"class_type": "SamplerCustomAdvanced", "inputs": {
            "noise": ["11", 0], "guider": ["13", 0], "sampler": ["12", 0], "sigmas": ["10", 0], "latent_image": ["8", 1]}},
        "15": {"class_type": "VAEDecode", "inputs": {"samples": ["14", 1], "vae": ["5", 0]}},
        "16": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["14", 1], "vae": ["6", 0]}},
        "17": {"class_type": "CreateVideo", "inputs": {
            "images": ["15", 0], "audio": ["16", 0], "fps": float(FPS), "bit_depth": "auto", "color_space": "sRGB", "codec": "none"}},
        "18": {"class_type": "SaveVideo", "inputs": {
            "video": ["17", 0], "filename_prefix": "H3Workspace/actor", "format": "auto", "format.codec": "auto"}},
    })
    loaders = _refs(g, "8", images, (), audios)
    stages = {n: "load" for n in ["1", "2", "3", "4", "5", "6", "7"] + [k for k in g if k.isdigit() and 60 <= int(k) < 70]}
    stages.update({n: "encode" for n in ["8"] + loaders})
    stages.update({"14": "sample", "15": "decode", "16": "decode", "17": "save", "18": "save"})
    return g, stages


def swap_prompt(whos, extra="", cuts=None, structured=True):
    """Prompt of the character swap preset. `whos`: who to replace with <Picture 1>, <Picture 2>... (one phrase each, in the
    user's words; a single string is accepted too). `extra`: optional addition. `cuts`: seconds where shot 2, 3... start (when
    several people are swapped in one pass: one person per shot). Tags follow the order the files are connected.
    `structured`: MiniMax's official three-field prompt (alignment line, [Shot N] timeline, soundscape, music), as References use."""
    whos = [whos] if isinstance(whos, str) else list(whos)
    tail = (" Retain the identity, attire, and style of the "
            + ("character from <Picture 1>." if len(whos) == 1 else "replacement characters from their pictures.")
            + " Simultaneously, preserve the camera angle, background, lighting, objects, and all other figures from the original video."
            " Ensure each replacement character aligns with the original figure in terms of position, scale, pose, and movement."
            + (" Only this one character is added: do not add any other figures, copies or people from the picture, and keep the background of <Video 1>."
               if len(whos) == 1 else " Do not add any other figures, copies or people, and keep the background of <Video 1>.")
            + " Do not display the reference images or their backgrounds.")
    names = [(w or "").strip() or ("the main person" if len(whos) == 1 else f"person {i + 1}") for i, w in enumerate(whos)]
    if not structured:
        t = " ".join(f"Replace {'only ' if len(whos) == 1 else ''}{w} in <Video 1> with the character from <Picture {i + 1}>."
                     for i, w in enumerate(names)) + tail
        return t + (" " + extra.strip() if extra and extra.strip() else "")
    pics = " ".join(f"<Picture {i + 1}> (from [Shot {i + 1}]) is fully referenced." for i in range(len(names)))
    head = (f"For the target video, at 0.00 seconds into the target video, {pics} <Video 1> is the video to edit: its motion, camera, "
            "timing, setting and sound are kept. The faces and identities of the replaced people must match the reference pictures exactly.")
    shots = []
    for i, w in enumerate(names):
        if i == 0:
            lead = "[Shot 1] Live-action, "
        else:
            t0 = (cuts[i - 1] if cuts and len(cuts) >= i else None)
            lead = (f"[Shot {i + 1}] At {int(t0 // 60):02d}:{t0 % 60:06.3f}, the camera cuts to the next shot of <Video 1>. " if t0 is not None
                    else f"[Shot {i + 1}] After the cut in <Video 1>, ")
        verb = "Replace" if lead.endswith(". ") else "replace"
        shots.append(lead + f"{verb} {'only ' if len(names) == 1 else ''}{w} in <Video 1> with the character from <Picture {i + 1}>.")
    body = " ".join(shots) + tail + (" " + extra.strip() if extra and extra.strip() else "")
    return f"{head}\n\nintegrated_multimodal_description: {body}\n\noverall_soundscape: N/A\n\nnon_diegetic_music: N/A"


def swap_graph(m, prompt, width, height, seconds, seed, images, video, extra_loras=(), pika=None, audio="original", small_ref=False):
    """Character swap preset (fallback group of Smart Bobo's «MiniMax H3 Character Swap» workflow):
    Ref2V model + swap LoRA 1.0 + turbo LoRA, sol-attn, sigma shift 12.2 / 3, er_sde, 10 manual sigmas.
    With `pika` (file name): the Pika H3 Flex Turbo LoRA instead of the turbo one, 8 steps, scheduler simple (as in References).
    `audio`: "original" - the sound of the video is put into the result; "lock" - the same, and Audio Lock makes the mouths follow
    that exact sound (lip sync); "model" - the sound the model invents (for videos that have no sound)."""
    g = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": m["lipsync_unet"], "weight_dtype": "default"}},
        "2": {"class_type": "ModelAttentionBackend", "inputs": {"model": ["1", 0], "attention": "comfy kitchen attention"}},
        "4": {"class_type": "CLIPLoader", "inputs": {"clip_name": m["text_encoder"], "type": "minimax", "device": "default"}},
        "5": {"class_type": "VAELoader", "inputs": {"vae_name": m["video_vae"]}},
        "6": {"class_type": "VAELoader", "inputs": {"vae_name": m["audio_vae"]}},
        "8": {"class_type": "MiniMaxH3ReferenceToVideo", "inputs": {
            "clip": ["4", 0], "vae": ["5", 0], "audio_vae": ["6", 0], "prompt": prompt,
            "width": width, "height": height, "length": frames(seconds), "ref_image_size": "match"}},
    }
    if pika:
        model = _lora_chain(g, ["2", 0], [(m["swap_lora"], 1.0), (pika, 1.0)] + list(extra_loras))
        g.update({
            "10": {"class_type": "BasicScheduler", "inputs": {"model": model, "scheduler": "simple", "steps": 8, "denoise": 1.0}},
            "12": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        })
    else:
        g["3"] = {"class_type": "BlockSparseAttention", "inputs": {
            "model": ["2", 0], "selection": "sol-attn", "selection.tau": 1.3, "start_percent": 0.2, "end_percent": 1.0,
            "dense_blocks": "", "min_tokens": 12288, "extra_tokens": 256, "sink_conditioning": "exact_kv_and_rows",
            "verbose": False}}
        g["7"] = {"class_type": "MiniMaxH3SigmaShift", "inputs": {"model": ["3", 0], "shift_video": 12.2, "shift_audio": 3.0}}
        model = _lora_chain(g, ["7", 0], [(m["swap_lora"], 1.0), (m["ref2v_turbo_lora"], 1.0)] + list(extra_loras))
        g.update({
            "10": {"class_type": "ManualSigmas", "inputs": {"sigmas": ACTOR_SIGMAS}},
            "12": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "er_sde"}},
        })
    latent, snd = ["8", 1], ["110", 2]
    if audio == "lock":   # exact sound of the video encoded into the latent: video is denoised while listening to it
        g["9"] = {"class_type": "MiniMaxH3NativeAudioLock", "inputs": {
            "model": ["1", 0], "av_latent": ["8", 1], "audio_vae": ["6", 0], "audio": ["110", 2]}}
        g["10"]["inputs"]["model"] = ["9", 0]
        latent, snd = ["9", 1], ["9", 2]
    g.update({
        "11": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "13": {"class_type": "BasicGuider", "inputs": {"model": model, "conditioning": ["8", 0]}},
        "14": {"class_type": "SamplerCustomAdvanced", "inputs": {
            "noise": ["11", 0], "guider": ["13", 0], "sampler": ["12", 0], "sigmas": ["10", 0], "latent_image": latent}},
        "15": {"class_type": "VAEDecode", "inputs": {"samples": ["14", 1], "vae": ["5", 0]}},
        "17": {"class_type": "CreateVideo", "inputs": {
            "images": ["15", 0], "audio": snd, "fps": float(FPS), "bit_depth": "auto", "color_space": "sRGB", "codec": "none"}},
        "18": {"class_type": "SaveVideo", "inputs": {
            "video": ["17", 0], "filename_prefix": "H3Workspace/swap", "format": "auto", "format.codec": "auto"}},
    })
    images = [images] if isinstance(images, str) else list(images)
    if audio == "model":   # a video without sound: the result gets the sound the model invents
        g["16"] = {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["14", 1], "vae": ["6", 0]}}
        g["17"]["inputs"]["audio"] = ["16", 0]
    loaders = _refs(g, "8", images[:2], [video], (), vsize=(width, height) if small_ref else None)   # node 110 = the video loader; its sound goes into the result
    stages = {n: "load" for n in ["1", "2", "3", "4", "5", "6", "7"] if n in g}
    stages.update({k: "load" for k in g if k.isdigit() and 60 <= int(k) < 70})
    stages.update({n: "encode" for n in ["8"] + loaders})
    stages.update({"14": "sample", "15": "decode", "17": "save", "18": "save"})
    return g, stages


def swap_shots_graph(m, shots, width, height, seed, images, video, extra="", pika=None, extra_loras=(), audio="original", small_ref=False):
    """Character swap over several shots (plans) of one video. The LoRA is made for one continuous shot, so every shot
    is swapped on its own and the pieces are joined again.

    shots: [{"s": start sec, "e": end sec, "pic": picture index or -1 (keep the original shot), "who": words}].
    VHS_LoadVideo takes just that part of the video (skip_first_frames / frame_load_cap, 24 fps), so no cutting is needed."""
    g = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": m["lipsync_unet"], "weight_dtype": "default"}},
        "2": {"class_type": "ModelAttentionBackend", "inputs": {"model": ["1", 0], "attention": "comfy kitchen attention"}},
        "4": {"class_type": "CLIPLoader", "inputs": {"clip_name": m["text_encoder"], "type": "minimax", "device": "default"}},
        "5": {"class_type": "VAELoader", "inputs": {"vae_name": m["video_vae"]}},
        "6": {"class_type": "VAELoader", "inputs": {"vae_name": m["audio_vae"]}},
    }
    if pika:
        model = _lora_chain(g, ["2", 0], [(m["swap_lora"], 1.0), (pika, 1.0)] + list(extra_loras))
        g["10"] = {"class_type": "BasicScheduler", "inputs": {"model": model, "scheduler": "simple", "steps": 8, "denoise": 1.0}}
        g["12"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}}
    else:
        g["3"] = {"class_type": "BlockSparseAttention", "inputs": {
            "model": ["2", 0], "selection": "sol-attn", "selection.tau": 1.3, "start_percent": 0.2, "end_percent": 1.0,
            "dense_blocks": "", "min_tokens": 12288, "extra_tokens": 256, "sink_conditioning": "exact_kv_and_rows",
            "verbose": False}}
        g["7"] = {"class_type": "MiniMaxH3SigmaShift", "inputs": {"model": ["3", 0], "shift_video": 12.2, "shift_audio": 3.0}}
        model = _lora_chain(g, ["7", 0], [(m["swap_lora"], 1.0), (m["ref2v_turbo_lora"], 1.0)] + list(extra_loras))
        g["10"] = {"class_type": "ManualSigmas", "inputs": {"sigmas": ACTOR_SIGMAS}}
        g["12"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "er_sde"}}
    stages = {n: "load" for n in ["1", "2", "3", "4", "5", "6", "7"] if n in g}
    stages.update({k: "load" for k in g if k.isdigit() and 60 <= int(k) < 70})
    pieces = {}
    for k, sh in enumerate(shots):
        n = max(5, int(round((sh["e"] - sh["s"]) * 24)))
        skip = int(round(sh["s"] * 24))
        sec = min(15.0, max(2.0, (sh["e"] - sh["s"])))
        vid = f"V{k}"
        g[vid] = {"class_type": "VHS_LoadVideo", "inputs": {
            "video": video, "force_rate": 24, "custom_width": width if small_ref else 0, "custom_height": height if small_ref else 0,
            "frame_load_cap": min(360, n),
            "skip_first_frames": skip, "select_every_nth": 1, "format": "None"}}
        stages[vid] = "encode"
        if sh.get("pic", -1) < 0 or sh["pic"] >= len(images):   # this shot stays as it is, with its own sound
            g[f"C{k}"] = {"class_type": "CreateVideo", "inputs": {
                "images": [vid, 0], "audio": [vid, 2], "fps": float(FPS), "bit_depth": "auto", "color_space": "sRGB", "codec": "none"}}
            stages[f"C{k}"] = "decode"
            pieces[k] = [f"C{k}", 0]
            continue
        prompt = swap_prompt(sh.get("who") or "", extra)
        pic = f"I{k}"
        g[pic] = {"class_type": "LoadImage", "inputs": {"image": images[sh["pic"]]}}
        ref = f"R{k}"
        g[ref] = {"class_type": "MiniMaxH3ReferenceToVideo", "inputs": {
            "clip": ["4", 0], "vae": ["5", 0], "audio_vae": ["6", 0], "prompt": prompt, "width": width, "height": height,
            "length": frames(sec), "ref_image_size": "match", "ref_images.ref_image_0": [pic, 0], "ref_videos.ref_video_0": [vid, 0]}}
        latent, sigmas, snd = [ref, 1], ["10", 0], [vid, 2]
        if audio == "lock":   # this shot's own sound is locked in: the mouths follow it, and the right person speaks the right lines
            lock_audio = [vid, 2]
            a_end = sh.get("a_end")
            if a_end is not None and sh["s"] < a_end < sh["e"] - 0.1:
                # the sound leads the picture (the next person starts to speak while this one is still on the screen):
                # only the part up to a_end drives this person's mouth, the rest is silence; the real sound stays in the video
                g[f"X{k}"] = {"class_type": "TrimAudioDuration", "inputs": {"audio": [vid, 2], "start_index": 0.0, "duration": round(a_end - sh["s"], 3)}}
                g[f"Y{k}"] = {"class_type": "EmptyAudio", "inputs": {"duration": round(sh["e"] - a_end, 3), "sample_rate": 44100, "channels": 2}}
                g[f"W{k}"] = {"class_type": "AudioConcat", "inputs": {"audio1": [f"X{k}", 0], "audio2": [f"Y{k}", 0], "direction": "after"}}
                lock_audio = [f"W{k}", 0]
                stages.update({f"X{k}": "encode", f"Y{k}": "encode", f"W{k}": "encode"})
            g[f"K{k}"] = {"class_type": "MiniMaxH3NativeAudioLock", "inputs": {
                "model": ["1", 0], "av_latent": [ref, 1], "audio_vae": ["6", 0], "audio": lock_audio}}
            sch = dict(g["10"]["inputs"]); sch["model"] = [f"K{k}", 0]
            g[f"H{k}"] = {"class_type": g["10"]["class_type"], "inputs": sch}
            latent, sigmas = [f"K{k}", 1], [f"H{k}", 0]   # the video keeps the real sound of the shot (snd stays the original)
            stages[f"K{k}"] = "encode"
        g[f"G{k}"] = {"class_type": "BasicGuider", "inputs": {"model": model, "conditioning": [ref, 0]}}
        g[f"N{k}"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}}
        g[f"S{k}"] = {"class_type": "SamplerCustomAdvanced", "inputs": {
            "noise": [f"N{k}", 0], "guider": [f"G{k}", 0], "sampler": ["12", 0], "sigmas": sigmas, "latent_image": latent}}
        g[f"D{k}"] = {"class_type": "VAEDecode", "inputs": {"samples": [f"S{k}", 1], "vae": ["5", 0]}}
        g[f"T{k}"] = {"class_type": "ImageFromBatch", "inputs": {"image": [f"D{k}", 0], "batch_index": 0, "length": n}}
        g[f"A{k}"] = {"class_type": "VAEDecodeAudio", "inputs": {"samples": [f"S{k}", 1], "vae": ["6", 0]}}
        g[f"C{k}"] = {"class_type": "CreateVideo", "inputs": {
            "images": [f"T{k}", 0], "audio": snd, "fps": float(FPS), "bit_depth": "auto", "color_space": "sRGB", "codec": "none"}}
        del g[f"A{k}"]   # the sound of the original video is kept, the one the model invents is not used
        stages.update({pic: "encode", ref: "encode", f"S{k}": "sample", f"D{k}": "decode", f"T{k}": "decode", f"C{k}": "decode"})
        pieces[k] = [f"C{k}", 0]
    if len(pieces) == 1:
        last = pieces[0]
    else:
        g["J"] = {"class_type": "ConcatenateVideo", "inputs": {**{f"videos.video{k}": v for k, v in pieces.items()}, "codec": "auto"}}
        stages["J"] = "decode"
        last = ["J", 0]
    g["Z"] = {"class_type": "SaveVideo", "inputs": {"video": last, "filename_prefix": "H3Workspace/swap",
                                                    "format": "auto", "format.codec": "auto"}}
    stages["Z"] = "save"
    return g, stages


# ---------- long video: scenes continue each other (Music-Video / Long-Shot workflow) ----------

def grain_graph(name, strength, seed):
    """Film-like grain on one clip: KJNodes' per-pixel noise (it differs from frame to frame). strength 0..1; the sound stays."""
    # Plain per-pixel colour noise looked like flicker. Film grain is a single-colour (brightness) noise with grains of 2-3 pixels:
    # noise is made at half size on mid-grey, taken from one channel, enlarged back and laid over the picture with «soft light».
    g = {"L": {"class_type": "LoadVideo", "inputs": {"file": name}},
         "N": {"class_type": "GetVideoComponents", "inputs": {"video": ["L", 0]}},
         "H": {"class_type": "ImageScaleBy", "inputs": {"image": ["N", 0], "upscale_method": "area", "scale_by": 0.5}},
         "Z": {"class_type": "ImageBlend", "inputs": {"image1": ["H", 0], "image2": ["H", 0], "blend_factor": 1.0, "blend_mode": "difference"}},   # black, same size and frame count
         "I": {"class_type": "ImageInvert", "inputs": {"image": ["Z", 0]}},
         "G": {"class_type": "ImageBlend", "inputs": {"image1": ["Z", 0], "image2": ["I", 0], "blend_factor": 0.5, "blend_mode": "normal"}},     # mid-grey
         "R": {"class_type": "ImageNoiseAugmentation", "inputs": {"image": ["G", 0], "noise_aug_strength": 0.3, "seed": seed}},
         "M": {"class_type": "ImageToMask", "inputs": {"image": ["R", 0], "channel": "red"}},
         "K": {"class_type": "MaskToImage", "inputs": {"mask": ["M", 0]}},
         "B": {"class_type": "ImageScaleBy", "inputs": {"image": ["K", 0], "upscale_method": "bilinear", "scale_by": 2.0}},
         "F": {"class_type": "ImageBlend", "inputs": {"image1": ["N", 0], "image2": ["B", 0], "blend_factor": round(min(1.0, strength * 2), 3), "blend_mode": "soft_light"}},   # measured: 30% (factor 0.3) gave only about 5 of 255 levels, too weak
         "W": {"class_type": "CreateVideo", "inputs": {"images": ["F", 0], "fps": ["N", 2], "audio": ["N", 1]}},
         "S": {"class_type": "SaveVideo", "inputs": {"video": ["W", 0], "filename_prefix": "H3Workspace/grain", "format": "auto", "codec": "auto"}}}
    return g, {k: "encode" for k in "LNHZIGRMKBF"} | {"W": "decode", "S": "save"}


def montage_graph(names, sizes, cuts=None):
    """Join clips end to end with ComfyUI's own nodes (no ffmpeg). Clips of another size than the first one are scaled to it
    first: the join node refuses clips of different sizes. `sizes`: (width, height) per clip or None when unknown."""
    g, ins, stages = {}, {}, {}
    first = next((s for s in sizes[:1] if s), None)
    for i, name in enumerate(names):
        g[f"L{i}"] = {"class_type": "LoadVideo", "inputs": {"file": name}}
        stages[f"L{i}"] = "encode"
        src = [f"L{i}", 0]
        if cuts and i < len(cuts) and cuts[i]:   # the clip is trimmed: only its first seconds are used
            g[f"T{i}"] = {"class_type": "Video Slice", "inputs": {"video": src, "start_time": 0.0, "duration": float(cuts[i]), "strict_duration": False}}
            stages[f"T{i}"] = "encode"
            src = [f"T{i}", 0]
        if first and sizes[i] and tuple(sizes[i]) != tuple(first):
            g[f"G{i}"] = {"class_type": "GetVideoComponents", "inputs": {"video": src}}
            g[f"Z{i}"] = {"class_type": "ImageScale", "inputs": {"image": [f"G{i}", 0], "upscale_method": "lanczos",
                                                                 "width": first[0], "height": first[1], "crop": "center"}}
            g[f"V{i}"] = {"class_type": "CreateVideo", "inputs": {"images": [f"Z{i}", 0], "fps": [f"G{i}", 2], "audio": [f"G{i}", 1]}}
            stages.update({f"G{i}": "encode", f"Z{i}": "encode", f"V{i}": "decode"})
            src = [f"V{i}", 0]
        ins[f"videos.video{i}"] = src
    g["C"] = {"class_type": "ConcatenateVideo", "inputs": {**ins, "codec": "auto"}}
    g["S"] = {"class_type": "SaveVideo", "inputs": {"video": ["C", 0], "filename_prefix": "H3Workspace/montage",
                                                    "format": "auto", "codec": "auto"}}
    stages.update({"C": "decode", "S": "save"})
    return g, stages


LONG_MAX_SCENE = 15      # seconds per scene (the model is trained for about 5-15 s)
LONG_OVERLAP = 1.0       # seconds of the previous scene that the next one continues from


def _snap(f):
    f = int(round(f))
    return f + (5 - f % 17) % 17


def _tokens(f):
    return round((f - 5) / 17 * 5 + 2)


def long_plan(scene_seconds, overlap=LONG_OVERLAP, cont_first=False, fresh=()):
    """Frame arithmetic of the original workflow: per scene (frames to generate, overlap frames, frames kept so far).
    `cont_first`: the first scene continues an existing video, so it is planned like a later scene.
    `fresh[i]`: scene i is a NEW SHOT: it does not continue the previous scene (no overlap, no saved tail), it starts from the
    reference pictures like scene 1, so the quality of a long video does not wear off from scene to scene."""
    plan, acc = [], 0
    for i, sec in enumerate(scene_seconds):
        seg = frames(min(LONG_MAX_SCENE, max(1, sec)))
        if i > 0 and i < len(fresh) and fresh[i]:
            plan.append({"total": seg, "ov": 0, "acc_before": acc, "acc": acc + seg})
            acc += seg
        elif i == 0 and not cont_first:
            plan.append({"total": seg, "ov": 0, "acc_before": 0, "acc": seg})
            acc = seg
        else:
            total = _snap(seg + _snap(max(5, round(overlap * FPS))))
            ov = _snap(total - seg)
            plan.append({"total": total, "ov": ov, "acc_before": acc, "acc": acc - ov + total})
            acc = acc - ov + total
    return plan


def _with_preview(g, model, preview):
    """KJNodes' Model Preview Override: an ANIMATED live preview (a few frames of the video latent per step), like in the
    ComfyUI workflow. Without it ComfyUI shows only the first frame. `preview` is None when that node is not installed."""
    if not preview:
        return model
    g["14"] = {"class_type": "ModelPreviewOverrideKJ", "inputs": {
        "model": model, "max_resolution": 512, "jpeg_quality": 75, "suppress_default_preview": False,
        "preview_frames": 12, "preview_fps": 8, "tiny_vae": preview.get("tiny_vae", "none")}}
    return ["14", 0]


def _long_base(m, pika, extra_loras, draft=False):
    """Model chain shared by the long-video graphs: Singularity Ref2V + memory savers + Pika (or the quality route)."""
    g = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": m["lipsync_unet"], "weight_dtype": "default"}},
        "2": {"class_type": "ModelAttentionBackend", "inputs": {"model": ["1", 0], "attention": "comfy kitchen attention"}},
        "3": {"class_type": "MiniMaxChunkFeedForward", "inputs": {"model": ["2", 0], "chunks": 2, "seq_threshold": 4096}},
        "4": {"class_type": "MiniMaxLowVRAMAttention", "inputs": {"model": ["3", 0], "head_chunks": 4}},
        "5": {"class_type": "CLIPLoader", "inputs": {"clip_name": m["text_encoder"], "type": "minimax", "device": "default"}},
        "6": {"class_type": "VAELoader", "inputs": {"vae_name": m["video_vae"]}},
        "7": {"class_type": "VAELoader", "inputs": {"vae_name": m["audio_vae"]}},
    }
    steps = 8
    if pika:
        model = _lora_chain(g, ["4", 0], list(extra_loras) + [(pika, 1.0)])
        sched, sampler_name = "simple", "euler"
        if draft:   # a quick draft: `pika` is the DMAD 4-step LoRA here (lcm sampler, shifts as in the editor it comes from)
            g["8"] = {"class_type": "MiniMaxH3SigmaShift", "inputs": {"model": model, "shift_video": 12.0, "shift_audio": 2.0}}
            model, sampler_name, steps = ["8", 0], "lcm", 4
    else:  # quality route without a speed LoRA, as in the original (er_sde, beta, 12 steps)
        model = _lora_chain(g, ["4", 0], list(extra_loras)) if extra_loras else ["4", 0]
        g["8"] = {"class_type": "MiniMaxH3SigmaShift", "inputs": {"model": model, "shift_video": 12.0, "shift_audio": 4.0}}
        model, sched, sampler_name, steps = ["8", 0], "beta", "er_sde", 12
    return g, model, sched, sampler_name, steps


def _audio_tokens(frames):
    return round(frames * 5 / 3)   # the audio latent of H3 has 40 steps per second, a video frame is 1/24 s


FADE_DB = (-4, -9, -18, -40)   # 4 steps of 50 ms: the sound dies away (or swells up) around a join of two scenes, no click or jump
FADE_STEP = 0.05


def _audio_fade(add, audio, dur, head=False, tail=False):
    """The same sound with a short fade-in at the start and/or a fade-out at the end (core nodes only: volume steps of 50 ms)."""
    n = len(FADE_DB)
    edge = n * FADE_STEP
    if dur < 2 * edge + 0.1 or not (head or tail):
        return audio
    t0 = edge if head else 0.0
    t1 = dur - (edge if tail else 0.0)
    parts = []
    if head:
        for j in range(n):
            piece = add("TrimAudioDuration", audio=audio, start_index=round(j * FADE_STEP, 3), duration=FADE_STEP)
            parts.append([add("AudioAdjustVolume", audio=[piece, 0], volume=FADE_DB[n - 1 - j]), 0])
    parts.append([add("TrimAudioDuration", audio=audio, start_index=round(t0, 3), duration=round(t1 - t0, 3)), 0])
    if tail:
        for j in range(n):
            piece = add("TrimAudioDuration", audio=audio, start_index=round(t1 + j * FADE_STEP, 3), duration=FADE_STEP)
            parts.append([add("AudioAdjustVolume", audio=[piece, 0], volume=FADE_DB[j]), 0])
    out = parts[0]
    for nxt in parts[1:]:
        out = [add("AudioConcat", audio1=out, audio2=nxt, direction="after"), 0]
    return out


def _audio_continue(add, old_audio, new_audio, ov, total):
    """The audio latent of a scene starts with the end of the previous scene's audio (kept as it is) and continues with new sound:
    the same trick as for the picture. Returns the audio latent for the sampler."""
    a_ov, a_tot = _audio_tokens(ov), _audio_tokens(total)
    cut = add("LatentCut", samples=old_audio, dim="y", index=-a_ov, amount=a_ov)
    cat = add("LatentConcat", samples1=[cut, 0], samples2=new_audio, dim="y")
    cut2 = add("LatentCut", samples=[cat, 0], dim="y", index=0, amount=a_tot)
    keep = add("SolidMask", value=0.0, width=a_ov, height=2)
    ones = add("SolidMask", value=1.0, width=a_tot, height=2)
    mask = add("MaskComposite", destination=[ones, 0], source=[keep, 0], x=0, y=0, operation="multiply")
    return [add("SetLatentNoiseMask", samples=[cut2, 0], mask=[mask, 0]), 0]


def long_graph(m, scenes, width, height, seed, images, audio=None, start=0.0, pika=None, extra_loras=(), ref_size="match", vocals=None, preview=None, fresh=(), audio_cont=False, draft=False):
    """Scenes [(prompt, seconds), ...] rendered one after another; every next scene continues the last second of the
    previous one (latent cut + noise mask), with the same reference pictures, and optionally follows a music track.

    Singularity Ref2V + Pika H3 Flex Turbo (8 steps), as chosen for the app. Without `audio` the model makes its own sound
    for every scene (voices and music are written in the prompts).
    """
    g, model, sched, sampler_name, steps = _long_base(m, pika, extra_loras, draft)
    model = _with_preview(g, model, preview)
    g["9"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": sampler_name}}
    g["10"] = {"class_type": "BasicScheduler", "inputs": {"model": model, "scheduler": sched, "steps": steps, "denoise": 1.0}}
    heard = ["11", 0]  # what the model hears; the finished video always carries the full track
    if audio:
        g["11"] = {"class_type": "LoadAudio", "inputs": {"audio": audio}}
        if vocals:  # voice only: helps lip sync, the music in the result stays complete
            g["12"] = {"class_type": "MelBandRoFormerModelLoader", "inputs": {"model_name": vocals}}
            g["13"] = {"class_type": "MelBandRoFormerSampler", "inputs": {"model": ["12", 0], "audio": ["11", 0]}}
            heard = ["13", 0]
    stages = {n: "load" for n in g}
    plan = long_plan([sc[1] for sc in scenes], fresh=fresh)
    nid = [200]

    def add(cls, **inputs):
        nid[0] += 1
        g[str(nid[0])] = {"class_type": cls, "inputs": inputs}
        return str(nid[0])

    prev_lat = prev_img = prev_aud = None
    scene_of = {}
    for i, (sc, p) in enumerate(zip(scenes, plan)):
        prompt = sc[0]
        own = list(sc[2]) if len(sc) > 2 and sc[2] else []  # this scene's own references; none: the common ones are used
        total, ov = p["total"], p["ov"]
        ref = add("MiniMaxH3ReferenceToVideo", clip=["5", 0], vae=["6", 0], audio_vae=["7", 0], prompt=prompt,
                  width=width, height=height, length=total, ref_image_size=ref_size)
        _refs(g, ref, own or images, base=1000 + 10 * i)
        new = add("LTXVSeparateAVLatent", av_latent=[ref, 1])
        if audio:
            ptr = start + (p["acc_before"] - ov) / FPS if i else start
            trim = add("TrimAudioDuration", audio=heard, start_index=round(ptr, 3), duration=round(total / FPS, 3))
            enc = add("VAEEncodeAudio", audio=[trim, 0], vae=["7", 0])
            amask = add("SolidMask", value=0.0, width=round(total * 5 / 3), height=2)
            a_lat = [add("SetLatentNoiseMask", samples=[enc, 0], mask=[amask, 0]), 0]
        else:
            a_lat = [new, 1]
        if i == 0 or ov == 0:   # the first scene, or a new shot: starts from the references only
            v_lat = [new, 0]
        else:
            old = add("LTXVSeparateAVLatent", av_latent=prev_lat)
            t_ov, t_tot = _tokens(ov), _tokens(total)
            cut = add("LatentCut", samples=[old, 0], dim="t", index=-t_ov, amount=t_ov)
            cat = add("LatentConcat", samples1=[cut, 0], samples2=[new, 0], dim="t")
            cut2 = add("LatentCut", samples=[cat, 0], dim="t", index=0, amount=t_tot)
            lw, lh = round(width / 16), round(height / 16)
            black = add("EmptyImage", width=lw, height=lh, batch_size=t_ov, color=0)
            white = add("EmptyImage", width=lw, height=lh, batch_size=max(1, t_tot - t_ov), color=16777215)
            both = add("ImageBatch", image1=[black, 0], image2=[white, 0])
            vmask = add("ImageToMask", image=[both, 0], channel="red")
            v_lat = [add("SetLatentNoiseMask", samples=[cut2, 0], mask=[vmask, 0]), 0]
            if audio_cont and not audio:   # the sound made by the model continues the sound of the previous scene (no jump at the join)
                a_lat = _audio_continue(add, [old, 1], [new, 1], ov, total)
        lat = add("LTXVConcatAVLatent", video_latent=v_lat, audio_latent=a_lat)
        noise = add("RandomNoise", noise_seed=seed + 100 * i)
        guider = add("BasicGuider", model=model, conditioning=[ref, 0])
        smp = add("SamplerCustomAdvanced", noise=[noise, 0], guider=[guider, 0], sampler=["9", 0], sigmas=["10", 0],
                  latent_image=[lat, 0])
        img = add("VAEDecode", samples=[smp, 1], vae=["6", 0])
        if i == 0:
            all_img = [img, 0]
        else:
            keep = add("ImageFromBatch", image=prev_img, batch_index=0, length=p["acc_before"] - ov)
            all_img = [add("ImageBatch", image1=[keep, 0], image2=[img, 0]), 0]
        if not audio:
            aud = add("VAEDecodeAudio", samples=[smp, 1], vae=["7", 0])
            if i == 0:
                all_aud = [aud, 0]
            else:
                kdur = round((p["acc_before"] - ov) / FPS, 3)
                kept = add("TrimAudioDuration", audio=prev_aud, start_index=0.0, duration=kdur)
                a_out = _audio_fade(add, [kept, 0], kdur, tail=True)           # the old scene dies away...
                a_in = _audio_fade(add, [aud, 0], round(total / FPS, 3), head=True)   # ...the new one swells up
                all_aud = [add("AudioConcat", audio1=a_out, audio2=a_in, direction="after"), 0]
            prev_aud = all_aud
            stages[aud] = "decode"
        stages.update({ref: "encode", smp: "sample", img: "decode"})
        scene_of[smp] = i
        prev_lat, prev_img = [smp, 1], all_img
    if audio:
        final_a = add("TrimAudioDuration", audio=["11", 0], start_index=float(start), duration=round(plan[-1]["acc"] / FPS, 3))
        final_a = [final_a, 0]
    else:
        final_a = prev_aud
    vid = add("CreateVideo", images=prev_img, audio=final_a, fps=float(FPS), bit_depth="auto", color_space="sRGB", codec="none")
    save = add("SaveVideo", video=[vid, 0], filename_prefix="H3Workspace/long", format="auto", **{"format.codec": "auto"})
    # the tail of the last scene's latent: lets the finished video be continued with one more scene later
    sep2 = add("LTXVSeparateAVLatent", av_latent=prev_lat)
    tail = add("LatentCut", samples=[sep2, 0], dim="t", index=-LONG_STATE_TOKENS, amount=LONG_STATE_TOKENS)
    keep = add("SaveLatent", samples=[tail, 0], filename_prefix="H3Workspace/long_state")
    stages.update({vid: "save", save: "save", sep2: "save", tail: "save", keep: "save"})
    stages["__scenes__"] = {"n": len(scenes), "map": scene_of}   # progress text "scene 2 of 4"
    return g, stages


LONG_STATE_TOKENS = 17   # latent tokens of the last second(s) of a scene that the next scene continues from


def long_part_graph(m, scenes, k, width, height, seed, images, audio=None, start=0.0, pika=None, extra_loras=(),
                    ref_size="match", vocals=None, state=None, preview=None, src=None, cont_plan=False, fresh=(), draft=False):
    """Scene number `k` (0-based) of the long video as its own job. Scene 0 starts fresh; scene k > 0 continues from the saved
    latent tail of scene k-1 (`state`, a .latent file in ComfyUI's input folder). The clip it saves has the overlap with the
    previous scene cut off, so the clips join end to end. Also saves the tail of its own latent for scene k+1."""
    g, model, sched, sampler_name, steps = _long_base(m, pika, extra_loras, draft)
    model = _with_preview(g, model, preview)
    g["9"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": sampler_name}}
    g["10"] = {"class_type": "BasicScheduler", "inputs": {"model": model, "scheduler": sched, "steps": steps, "denoise": 1.0}}
    heard = ["11", 0]
    if audio:
        g["11"] = {"class_type": "LoadAudio", "inputs": {"audio": audio}}
        if vocals:
            g["12"] = {"class_type": "MelBandRoFormerModelLoader", "inputs": {"model_name": vocals}}
            g["13"] = {"class_type": "MelBandRoFormerSampler", "inputs": {"model": ["12", 0], "audio": ["11", 0]}}
            heard = ["13", 0]
    stages = {n: "load" for n in g}
    cont = bool(src) and k == 0   # the first scene continues a video the user already has (src = (file in ComfyUI input, frames))
    plan = long_plan([sc[1] for sc in scenes], cont_first=bool(src) or cont_plan, fresh=fresh)   # a project that continues an existing video
    p, sc = plan[k], scenes[k]
    new_shot = k > 0 and k < len(fresh) and bool(fresh[k])   # does not continue scene k-1
    total, ov = p["total"], p["ov"]
    nid = [200]

    def add(cls, **inputs):
        nid[0] += 1
        g[str(nid[0])] = {"class_type": cls, "inputs": inputs}
        return str(nid[0])

    ref = add("MiniMaxH3ReferenceToVideo", clip=["5", 0], vae=["6", 0], audio_vae=["7", 0], prompt=sc[0],
              width=width, height=height, length=total, ref_image_size=ref_size)
    own = list(sc[2]) if len(sc) > 2 and sc[2] else []   # this scene's own references; none: the common ones are used
    _refs(g, ref, own or images, base=100)
    new = add("LTXVSeparateAVLatent", av_latent=[ref, 1])
    if audio:
        ptr = max(0.0, start - ov / FPS) if cont else start + ((p["acc_before"] - ov) / FPS if k else 0)
        trim = add("TrimAudioDuration", audio=heard, start_index=round(ptr, 3), duration=round(total / FPS, 3))
        enc = add("VAEEncodeAudio", audio=[trim, 0], vae=["7", 0])
        amask = add("SolidMask", value=0.0, width=round(total * 5 / 3), height=2)
        a_lat = [add("SetLatentNoiseMask", samples=[enc, 0], mask=[amask, 0]), 0]
    else:
        a_lat = [new, 1]
    if (k == 0 and not cont) or new_shot:
        v_lat = [new, 0]
    else:
        if cont:   # the tail of the existing video, encoded back into a latent
            vid_in = add("LoadVideo", file=src[0])
            comp = add("GetVideoComponents", video=[vid_in, 0])
            last = add("ImageFromBatch", image=[comp, 0], batch_index=max(0, src[1] - ov), length=ov)
            fit = add("ImageScale", image=[last, 0], upscale_method="lanczos", width=width, height=height, crop="center")
            old = add("VAEEncode", pixels=[fit, 0], vae=["6", 0])
            stages.update({vid_in: "encode", comp: "encode", last: "encode", fit: "encode", old: "encode"})
        else:
            old = add("LoadLatent", latent=state)
        t_ov, t_tot = _tokens(ov), _tokens(total)
        cut = add("LatentCut", samples=[old, 0], dim="t", index=-t_ov, amount=t_ov)
        cat = add("LatentConcat", samples1=[cut, 0], samples2=[new, 0], dim="t")
        cut2 = add("LatentCut", samples=[cat, 0], dim="t", index=0, amount=t_tot)
        lw, lh = round(width / 16), round(height / 16)
        black = add("EmptyImage", width=lw, height=lh, batch_size=t_ov, color=0)
        white = add("EmptyImage", width=lw, height=lh, batch_size=max(1, t_tot - t_ov), color=16777215)
        both = add("ImageBatch", image1=[black, 0], image2=[white, 0])
        vmask = add("ImageToMask", image=[both, 0], channel="red")
        v_lat = [add("SetLatentNoiseMask", samples=[cut2, 0], mask=[vmask, 0]), 0]
        stages[old] = "load"
    lat = add("LTXVConcatAVLatent", video_latent=v_lat, audio_latent=a_lat)
    noise = add("RandomNoise", noise_seed=seed + 100 * k)
    guider = add("BasicGuider", model=model, conditioning=[ref, 0])
    smp = add("SamplerCustomAdvanced", noise=[noise, 0], guider=[guider, 0], sampler=["9", 0], sigmas=["10", 0],
              latent_image=[lat, 0])
    img = add("VAEDecode", samples=[smp, 1], vae=["6", 0])
    clip_len = total - ov
    frames_out = [img, 0] if ((not k and not cont) or new_shot) else [add("ImageFromBatch", image=[img, 0], batch_index=ov, length=clip_len), 0]
    if audio:
        a0 = start + (p["acc_before"] / FPS if k else 0)
        aud = [add("TrimAudioDuration", audio=["11", 0], start_index=round(a0, 3), duration=round(clip_len / FPS, 3)), 0]
    else:
        dec = add("VAEDecodeAudio", samples=[smp, 1], vae=["7", 0])
        stages[dec] = "decode"
        aud = [dec, 0] if ((not k and not cont) or new_shot) else [add("TrimAudioDuration", audio=[dec, 0], start_index=round(ov / FPS, 3),
                                                          duration=round(clip_len / FPS, 3)), 0]
        # soft edges: this clip is joined to the others later, a jump of sound at the join is hidden by a short fade
        aud = _audio_fade(add, aud, round(clip_len / FPS, 3), head=bool(k or cont), tail=True)
    vid = add("CreateVideo", images=frames_out, audio=aud, fps=float(FPS), bit_depth="auto", color_space="sRGB", codec="none")
    save = add("SaveVideo", video=[vid, 0], filename_prefix="H3Workspace/long_part", format="auto", **{"format.codec": "auto"})
    sep2 = add("LTXVSeparateAVLatent", av_latent=[smp, 1])
    tail = add("LatentCut", samples=[sep2, 0], dim="t", index=-LONG_STATE_TOKENS, amount=LONG_STATE_TOKENS)
    keep = add("SaveLatent", samples=[tail, 0], filename_prefix="H3Workspace/long_state")
    stages.update({ref: "encode", smp: "sample", img: "decode", vid: "save", save: "save", sep2: "save", tail: "save", keep: "save"})
    stages["__scenes__"] = {"n": len(scenes), "map": {smp: k}}
    return g, stages


ACTION_LOWRES = 0.6  # SelfLift lowres_scale in the original


def action_graph(m, prompt, width, height, seconds, seed, images, audios, faithful, loras):
    """PM Minimax H3 Best Action Workflow (Prompt Mastery): hybrid model, sol-attn, 8 low-res + 4 high-res steps.

    faithful=True runs the original SelfLiftH3Sampler + H3SigmaRefiner; otherwise the same schedule is
    done with built-in nodes: pass 1 at 0.6x, learned latent upscale, pass 2 at full size.
    width/height are the final size.
    """
    g = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": m["action_unet"], "weight_dtype": "default"}},
        "4": {"class_type": "CLIPLoader", "inputs": {"clip_name": m["text_encoder"], "type": "minimax", "device": "default"}},
        "5": {"class_type": "VAELoader", "inputs": {"vae_name": m["video_vae"]}},
        "6": {"class_type": "VAELoader", "inputs": {"vae_name": m["audio_vae"]}},
    }
    model = _lora_chain(g, ["1", 0], loras)  # dmad 0.8, speed slider 4 in the original
    g.update({
        "2": {"class_type": "ModelAttentionBackend", "inputs": {"model": model, "attention": "comfy kitchen attention"}},
        "3": {"class_type": "BlockSparseAttention", "inputs": {
            "model": ["2", 0], "selection": "sol-attn", "selection.tau": 1.3, "start_percent": 0.2, "end_percent": 1.0,
            "dense_blocks": "", "min_tokens": 12288, "extra_tokens": 256, "sink_conditioning": "exact_kv_and_rows",
            "verbose": False}},
        "7": {"class_type": "MiniMaxH3SigmaShift", "inputs": {"model": ["3", 0], "shift_video": 12.0, "shift_audio": 3.0}},
        "10": {"class_type": "BasicScheduler", "inputs": {"model": ["7", 0], "scheduler": "simple", "steps": 12, "denoise": 1.0}},
        "11": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "12": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
    })
    if faithful:
        g.update({
            "8": {"class_type": "MiniMaxH3ReferenceToVideo", "inputs": {
                "clip": ["4", 0], "vae": ["5", 0], "audio_vae": ["6", 0], "prompt": prompt,
                "width": width, "height": height, "length": frames(seconds), "ref_image_size": "max"}},
            "9": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["8", 0]}},
            "13": {"class_type": "H3SigmaRefiner", "inputs": {
                "sigmas": ["10", 0], "extra_steps": 1, "strength": 0.7, "offset": 0.0, "curve": "cosine"}},
            "14": {"class_type": "SelfLiftH3Sampler", "inputs": {
                "model": ["7", 0], "positive": ["8", 0], "negative": ["9", 0], "vae": ["5", 0],
                "latent_image": ["8", 1], "sampler": ["12", 0], "sigmas": ["13", 0], "seed": seed, "cfg": 1.0,
                "transition_step": 8, "lowres_scale": ACTION_LOWRES, "rho": 0.0, "w_min": 0.5, "w_max": 1.0,
                "upscaler_model": m["action_upscaler"], "highres_tiling": True}},
            "20": {"class_type": "VAEDecode", "inputs": {"samples": ["14", 0], "vae": ["5", 0]}},
            "21": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["14", 0], "vae": ["6", 0]}},
        })
        sample_stages = {"14": "sample"}
    else:
        lw, lh = r32(width * ACTION_LOWRES), r32(height * ACTION_LOWRES)
        g.update({
            "8": {"class_type": "MiniMaxH3ReferenceToVideo", "inputs": {
                "clip": ["4", 0], "vae": ["5", 0], "audio_vae": ["6", 0], "prompt": prompt,
                "width": lw, "height": lh, "length": frames(seconds), "ref_image_size": "max"}},
            "13": {"class_type": "BasicGuider", "inputs": {"model": ["7", 0], "conditioning": ["8", 0]}},
            "9": {"class_type": "SplitSigmas", "inputs": {"sigmas": ["10", 0], "step": 8}},
            "14": {"class_type": "SamplerCustomAdvanced", "inputs": {
                "noise": ["11", 0], "guider": ["13", 0], "sampler": ["12", 0], "sigmas": ["9", 0], "latent_image": ["8", 1]}},
            "15": {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["14", 1]}},
            "16": {"class_type": "MinimaxH3LatentUpscaler3D", "inputs": {
                "latent": ["15", 0], "model_name": m["upscaler"], "mode": "scale by multiplier",
                "mode.scale": round(1 / ACTION_LOWRES, 3), "align": 32, "enable_temporal_chunking": True,
                "force_unload": True, "device": "cuda", "precision": "fp16"}},
            "17": {"class_type": "LTXVConcatAVLatent", "inputs": {"video_latent": ["16", 0], "audio_latent": ["15", 1]}},
            "18": {"class_type": "SamplerCustomAdvanced", "inputs": {
                "noise": ["11", 0], "guider": ["13", 0], "sampler": ["12", 0], "sigmas": ["9", 1], "latent_image": ["17", 0]}},
            "20": {"class_type": "VAEDecode", "inputs": {"samples": ["18", 1], "vae": ["5", 0]}},
            "21": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["18", 1], "vae": ["6", 0]}},
        })
        sample_stages = {"14": "sample", "15": "upscale", "16": "upscale", "17": "upscale", "18": "sample2"}
    g.update({
        "22": {"class_type": "CreateVideo", "inputs": {
            "images": ["20", 0], "audio": ["21", 0], "fps": float(FPS), "bit_depth": "auto", "color_space": "sRGB", "codec": "none"}},
        "23": {"class_type": "SaveVideo", "inputs": {
            "video": ["22", 0], "filename_prefix": "H3Workspace/action", "format": "auto", "format.codec": "auto"}},
    })
    loaders = _refs(g, "8", images, (), audios)
    stages = {n: "load" for n in ["1", "2", "3", "4", "5", "6", "7"] + [k for k in g if k.isdigit() and 60 <= int(k) < 70]}
    stages.update({n: "encode" for n in ["8"] + loaders})
    stages.update(sample_stages)
    stages.update({"20": "decode", "21": "decode", "22": "save", "23": "save"})
    return g, stages


def lipsync_scale(quality):
    return 2.0 if QUALITY.get(quality, 0) >= 1824 * 1024 else 1.5


def lipsync_prompt(text, n_refs, n_videos=0):
    """Wrap free text into H3's structured prompt unless the user already wrote one.
    Reference videos are tagged <Video N> as the official workflow asks; with pictures present the face comes from the pictures."""
    if "integrated_multimodal_description" in text:
        return text
    refs = " ".join(f"<Picture {i}> (from [Shot 1]) is fully referenced." for i in range(1, n_refs + 1))
    if n_videos:
        vids = ", ".join(f"<Video {i}>" for i in range(1, n_videos + 1))
        refs += f" {vids} {'is' if n_videos == 1 else 'are'} referenced for motion, camera and setting only."
        if n_refs:
            refs += " The faces and identities must match the reference pictures exactly."
    return (f"For the target video, at 0.00 seconds into the target video, {refs}\n\n"
            f"integrated_multimodal_description: [Shot 1] {text}\n\n"
            "overall_soundscape: N/A\n\nnon_diegetic_music: N/A")


def lipsync_graph(m, prompt, width, height, seconds, seed, audio, ref_images, scale, ref_size="max", extra_loras=()):
    """MiniMax H3 Singularity Ref2V + Audio Lock, two passes.

    width/height are pass 1 size; pass 2 upscales the latent by `scale`.
    `audio` is the already trimmed clip; the video length follows it.
    ref_size: "max" (best identity, several times slower) or "match" (refs scaled to the video area).
    """
    g = {
        "1": {"class_type": "UNETLoader", "inputs": {
            "unet_name": m["lipsync_unet"], "weight_dtype": "default"}},
        # the original workflow uses KJNodes' SageAttention patch; the built-in INT8 attention
        # needs no extra install (sageattention is hard to set up on Windows)
        "2": {"class_type": "ModelAttentionBackend", "inputs": {
            "model": ["1", 0], "attention": "comfy kitchen attention"}},
        "3": {"class_type": "LoraLoaderModelOnly", "inputs": {
            "model": ["2", 0], "lora_name": m["ref2v_turbo_lora"], "strength_model": 0.55}},
        "4": {"class_type": "CLIPLoader", "inputs": {
            "clip_name": m["text_encoder"], "type": "minimax", "device": "default"}},
        "5": {"class_type": "VAELoader", "inputs": {"vae_name": m["video_vae"]}},
        "6": {"class_type": "VAELoader", "inputs": {"vae_name": m["audio_vae"]}},
        "7": {"class_type": "LoadAudio", "inputs": {"audio": audio}},
        "8": {"class_type": "MiniMaxH3ReferenceToVideo", "inputs": {
            "clip": ["4", 0], "vae": ["5", 0], "audio_vae": ["6", 0], "prompt": prompt,
            "width": width, "height": height, "length": frames(seconds), "ref_image_size": ref_size}},
        "9": {"class_type": "MiniMaxH3NativeAudioLock", "inputs": {
            "model": ["1", 0], "av_latent": ["8", 1], "audio_vae": ["6", 0], "audio": ["7", 0]}},
        "10": {"class_type": "BasicScheduler", "inputs": {
            "model": ["9", 0], "scheduler": "beta", "steps": 12, "denoise": 1.0}},
        "11": {"class_type": "SplitSigmas", "inputs": {"sigmas": ["10", 0], "step": 8}},
        "12": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "13": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "14": {"class_type": "BasicGuider", "inputs": {"model": ["3", 0], "conditioning": ["8", 0]}},
        # pass 1
        "15": {"class_type": "SamplerCustomAdvanced", "inputs": {
            "noise": ["12", 0], "guider": ["14", 0], "sampler": ["13", 0],
            "sigmas": ["11", 0], "latent_image": ["9", 1]}},
        # pass 2: upscale the video latent, keep the audio latent, finish the low sigmas
        "16": {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["15", 1]}},
        "17": {"class_type": "MinimaxH3LatentUpscaler3D", "inputs": {
            "latent": ["16", 0], "model_name": m["upscaler"],
            "mode": "scale by multiplier", "mode.scale": scale, "align": 32,
            "enable_temporal_chunking": True, "force_unload": True,
            "device": "cuda", "precision": "fp16"}},
        "18": {"class_type": "LTXVConcatAVLatent", "inputs": {"video_latent": ["17", 0], "audio_latent": ["16", 1]}},
        "19": {"class_type": "SamplerCustomAdvanced", "inputs": {
            "noise": ["12", 0], "guider": ["14", 0], "sampler": ["13", 0],
            "sigmas": ["11", 1], "latent_image": ["18", 0]}},
        "20": {"class_type": "VAEDecode", "inputs": {"samples": ["19", 1], "vae": ["5", 0]}},
        # the saved video carries the original audio, not a regenerated one
        "21": {"class_type": "CreateVideo", "inputs": {
            "images": ["20", 0], "audio": ["9", 2], "fps": float(FPS),
            "bit_depth": "auto", "color_space": "sRGB", "codec": "none"}},
        "22": {"class_type": "SaveVideo", "inputs": {
            "video": ["21", 0], "filename_prefix": "H3Workspace/lipsync",
            "format": "auto", "format.codec": "auto"}},
    }
    for i, image in enumerate(ref_images[:9]):
        node_id = str(30 + i)
        g[node_id] = {"class_type": "LoadImage", "inputs": {"image": image}}
        g["8"]["inputs"][f"ref_images.ref_image_{i}"] = [node_id, 0]
    if extra_loras:  # optional LoRAs go in front of the speed LoRA, as in the original workflow
        g["3"]["inputs"]["model"] = _lora_chain(g, ["2", 0], list(extra_loras))

    stages = {n: "load" for n in ("1", "2", "3", "4", "5", "6", "7")}
    stages.update({n: "load" for n in g if n.isdigit() and 60 <= int(n) < 70})
    stages.update({n: "encode" for n in ["8", "9"] + [str(30 + i) for i in range(9)]})
    stages.update({"15": "sample", "16": "upscale", "17": "upscale", "18": "upscale",
                   "19": "sample2", "20": "decode", "21": "save", "22": "save"})
    return g, stages


# ---------- pictures: Krea 2 (text) and Qwen Image 2.1 (edit, character sheet) ----------

# Krea 2 works at about 1 MP only
KREA_SIZES = {"16:9": (1376, 768), "9:16": (768, 1376), "1:1": (1024, 1024), "4:3": (1184, 896),
              "3:4": (896, 1184), "3:2": (1248, 832), "2:3": (832, 1248), "21:9": (1568, 672)}
QWEN_MP = {"1": 1.0, "2": 2.0}
SHEET_SIZE = (1920, 1088)
SHEET6_SIZE = (2112, 960)  # six tall panels in one row, about 2 MP like the four-panel sheet
SHEET_BG = {"grey": "light grey", "white": "white", "black": "black"}


def krea_graph(m, prompt, width, height, seed):
    """Krea 2 Turbo, text to picture: 10 steps, CFG 1, no negative prompt (the model has none)."""
    g = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": m["krea_unet"], "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": m["krea_clip"], "type": "krea2", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": m["krea_vae"]}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["2", 0]}},
        "5": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["4", 0]}},
        "6": {"class_type": "EmptyLatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
        "7": {"class_type": "KSampler", "inputs": {
            "seed": seed, "steps": 10, "cfg": 1, "sampler_name": "euler", "scheduler": "simple", "denoise": 1,
            "model": ["1", 0], "positive": ["4", 0], "negative": ["5", 0], "latent_image": ["6", 0]}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["3", 0]}},
        "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": "H3Workspace/krea", "images": ["8", 0]}},
    }
    stages = {"1": "load", "2": "load", "3": "load", "4": "encode", "5": "encode", "6": "encode",
              "7": "sample", "8": "decode", "9": "save"}
    return g, stages


KREA_REAL_SCALES = {"2k": 1.5, "4k": 2.5}


def krea_real_graph(m, prompt, width, height, seed, scale):
    """Krea 2 realism (RedCraft RedMix + Realism Engine LoRA 0.5), two passes: draw ~1 MP, enlarge the latent, refine with denoise 0.6."""
    g = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": m["kreareal_unet"], "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": m["krea_clip"], "type": "krea2", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": m["krea_vae"]}},
        "4": {"class_type": "LoraLoaderModelOnly", "inputs": {"lora_name": m["kreareal_lora"], "strength_model": 0.5, "model": ["1", 0]}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["2", 0]}},
        "6": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["5", 0]}},
        "7": {"class_type": "EmptyLatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
        "8": {"class_type": "KSampler", "inputs": {
            "seed": seed, "steps": 8, "cfg": 1, "sampler_name": "euler_ancestral", "scheduler": "sgm_uniform", "denoise": 1,
            "model": ["4", 0], "positive": ["5", 0], "negative": ["6", 0], "latent_image": ["7", 0]}},
        "9": {"class_type": "LatentUpscaleBy", "inputs": {"upscale_method": "bicubic", "scale_by": scale, "samples": ["8", 0]}},
        "10": {"class_type": "KSampler", "inputs": {
            "seed": seed, "steps": 6, "cfg": 1, "sampler_name": "er_sde", "scheduler": "simple", "denoise": 0.6,
            "model": ["4", 0], "positive": ["5", 0], "negative": ["6", 0], "latent_image": ["9", 0]}},
        "11": {"class_type": "VAEDecode", "inputs": {"samples": ["10", 0], "vae": ["3", 0]}},
        "12": {"class_type": "SaveImage", "inputs": {"filename_prefix": "H3Workspace/krea_real", "images": ["11", 0]}},
    }
    stages = {"1": "load", "2": "load", "3": "load", "4": "load", "5": "encode", "6": "encode", "7": "encode",
              "8": "sample", "9": "upscale", "10": "sample2", "11": "decode", "12": "save"}
    return g, stages


def _qwen_base(m, image, megapixels, prompt, seed, size=None, extras=(), extra_loras=()):
    """The graph of the "One Image Edit" / "Character Sheet" workflows on plain ComfyUI nodes.
    `size` (width, height) gives an empty canvas of its own; otherwise the result keeps the picture's shape."""
    g = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": m["qwen_unet"], "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": m["qwen_clip"], "type": "qwen_image", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": m["qwen_vae"]}},
        "4": {"class_type": "QwenImage21Cache", "inputs": {"device": "auto", "dtype": "default", "model": ["1", 0]}},
        "5": {"class_type": "LoadImage", "inputs": {"image": image}},
        "6": {"class_type": "ImageScaleToTotalPixels", "inputs": {
            "upscale_method": "lanczos", "megapixels": megapixels, "resolution_steps": 32, "image": ["5", 0]}},
        "7": {"class_type": "TextEncodeQwenImage21", "inputs": {
            "prompt": prompt, "negative_prompt": "", "resolution": 0, "clip": ["2", 0],
            "images.image_1": ["6", 0], "vae": ["3", 0]}},
        "9": {"class_type": "KSampler", "inputs": {
            "seed": seed, "steps": 25, "cfg": 1, "sampler_name": "euler", "scheduler": "simple", "denoise": 1,
            "model": ["4", 0], "positive": ["7", 0], "negative": ["7", 1], "latent_image": ["7", 2]}},
        "10": {"class_type": "VAEDecode", "inputs": {"samples": ["9", 0], "vae": ["3", 0]}},
        "11": {"class_type": "SaveImage", "inputs": {"filename_prefix": "H3Workspace/qwen", "images": ["10", 0]}},
    }
    if extra_loras:
        g["4"]["inputs"]["model"] = _lora_chain(g, ["1", 0], extra_loras, base=60)
    for k, img in enumerate(list(extras)[:2], 2):  # more reference pictures: <image2>, <image3>
        a, b = f"12{k}", f"13{k}"
        g[a] = {"class_type": "LoadImage", "inputs": {"image": img}}
        g[b] = {"class_type": "ImageScaleToTotalPixels", "inputs": {
            "upscale_method": "lanczos", "megapixels": megapixels, "resolution_steps": 32, "image": [a, 0]}}
        g["7"]["inputs"][f"images.image_{k}"] = [b, 0]
    if size:
        g["8"] = {"class_type": "EmptyLatentImage", "inputs": {"width": size[0], "height": size[1], "batch_size": 1}}
        g["9"]["inputs"]["latent_image"] = ["8", 0]
    stages = {"1": "load", "2": "load", "3": "load", "4": "load", "5": "encode", "6": "encode", "7": "encode",
              "8": "encode", "9": "sample", "10": "decode", "11": "save"}
    if extra_loras:
        for i in range(len(extra_loras)):
            stages[str(60 + i)] = "load"
    for k in (2, 3):
        stages[f"12{k}"] = stages[f"13{k}"] = "encode"
    return g, stages


def sdxl_t2i_graph(m, prompt, negative, width, height, seed, extra_loras=(), steps=28, cfg=6.5):
    """SDXL / Pony checkpoint, text to image: full uncensored photorealism, LoRA support."""
    g = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": m["sdxl_checkpoint"]}},
        "4": {"class_type": "EmptyLatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
        "5": {"class_type": "KSampler", "inputs": {
            "seed": seed, "steps": steps, "cfg": cfg, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 1.0,
            "model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0], "latent_image": ["4", 0]}},
        "6": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
        "7": {"class_type": "SaveImage", "inputs": {"filename_prefix": "H3Workspace/sdxl_t2i", "images": ["6", 0]}},
    }
    model_out, clip_out = ["1", 0], ["1", 1]
    stages = {"1": "load", "4": "encode", "5": "sample", "6": "decode", "7": "save"}
    if extra_loras:
        model_out, clip_out = _sdxl_lora_chain(g, model_out, clip_out, extra_loras, base=60)
        for i in range(len(extra_loras)):
            stages[str(60 + i)] = "load"
    g["2"] = {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": clip_out}}
    g["3"] = {"class_type": "CLIPTextEncode", "inputs": {"text": negative, "clip": clip_out}}
    g["5"]["inputs"]["model"] = model_out
    g["5"]["inputs"]["positive"] = ["2", 0]
    g["5"]["inputs"]["negative"] = ["3", 0]
    stages.update({"2": "encode", "3": "encode"})
    return g, stages


def sdxl_inpaint_graph(m, prompt, negative, image, seed, mask=None, denoise=0.75, size=None, extra_loras=(), steps=28, cfg=6.5):
    """SDXL / Pony checkpoint: inpaint / clothing removal / img2img edit.
    If mask is provided, uses VAEEncodeForInpaint to regenerate only the masked region.
    If mask is None, uses VAEEncode + denoise for img2img editing (preserving overall pose/composition).
    """
    g = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": m["sdxl_checkpoint"]}},
        "10": {"class_type": "LoadImage", "inputs": {"image": image}},
    }
    model_out, clip_out = ["1", 0], ["1", 1]
    stages = {"1": "load", "10": "load"}
    if extra_loras:
        model_out, clip_out = _sdxl_lora_chain(g, model_out, clip_out, extra_loras, base=60)
        for i in range(len(extra_loras)):
            stages[str(60 + i)] = "load"

    g["2"] = {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": clip_out}}
    g["3"] = {"class_type": "CLIPTextEncode", "inputs": {"text": negative, "clip": clip_out}}
    stages.update({"2": "encode", "3": "encode"})

    # SDXL was taught on ~1024x1024. A native 4000 px photo is far outside that, and the model answers
    # with invented anatomy and melted details - the "it hallucinates" complaint. So: the user's own
    # width/height when given, otherwise the picture's own shape brought to about one megapixel.
    img_ref = ["10", 0]
    if size:
        g["11"] = {"class_type": "ImageScale", "inputs": {
            "image": img_ref, "upscale_method": "lanczos", "width": size[0], "height": size[1], "crop": "center"}}
    else:
        g["11"] = {"class_type": "ImageScaleToTotalPixels", "inputs": {
            "image": img_ref, "upscale_method": "lanczos", "megapixels": 1.0}}
    img_ref = ["11", 0]
    stages["11"] = "encode"

    if mask:
        g["12"] = {"class_type": "LoadImage", "inputs": {"image": mask}}
        mask_img = ["12", 0]
        stages["12"] = "load"
        # the mask must land on exactly the pixels above: same node, same settings, same source shape
        if size:
            g["13"] = {"class_type": "ImageScale", "inputs": {
                "image": mask_img, "upscale_method": "lanczos", "width": size[0], "height": size[1], "crop": "center"}}
        else:
            g["13"] = {"class_type": "ImageScaleToTotalPixels", "inputs": {
                "image": mask_img, "upscale_method": "lanczos", "megapixels": 1.0}}
        mask_img = ["13", 0]
        stages["13"] = "encode"
        g["14"] = {"class_type": "ImageToMask", "inputs": {"image": mask_img, "channel": "red"}}
        g["15"] = {"class_type": "VAEEncodeForInpaint", "inputs": {
            "pixels": img_ref, "vae": ["1", 2], "mask": ["14", 0], "grow_mask_by": 6}}
        latent_ref = ["15", 0]
        stages.update({"14": "encode", "15": "encode"})
        run_denoise = denoise   # the slider is the user's wish: VAEEncodeForInpaint carries its own noise mask,
                                # so a value below 1.0 keeps the rest of the picture instead of repainting it
    else:
        g["15"] = {"class_type": "VAEEncode", "inputs": {"pixels": img_ref, "vae": ["1", 2]}}
        latent_ref = ["15", 0]
        stages["15"] = "encode"
        run_denoise = denoise

    g["20"] = {"class_type": "KSampler", "inputs": {
        "seed": seed, "steps": steps, "cfg": cfg, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": run_denoise,
        "model": model_out, "positive": ["2", 0], "negative": ["3", 0], "latent_image": latent_ref}}
    g["21"] = {"class_type": "VAEDecode", "inputs": {"samples": ["20", 0], "vae": ["1", 2]}}
    g["22"] = {"class_type": "SaveImage", "inputs": {"filename_prefix": "H3Workspace/sdxl_inpaint", "images": ["21", 0]}}
    stages.update({"20": "sample", "21": "decode", "22": "save"})
    return g, stages


def flux_t2i_graph(m, prompt, width, height, seed, extra_loras=(), guidance=3.5, steps=20):
    """FLUX.1 [dev] FP8 (12B), picture by text: 20 steps, FluxGuidance 3.5, full LoRA support."""
    g = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": m["flux_unet"], "weight_dtype": "default"}},
        "2": {"class_type": "DualCLIPLoader", "inputs": {
            "clip_name1": m["flux_clip"], "clip_name2": m["flux_clip_l"], "type": "flux", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": m["flux_vae"]}},
        "7": {"class_type": "EmptySD3LatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
    }
    model_out, clip_out = ["1", 0], ["2", 0]
    stages = {"1": "load", "2": "load", "3": "load", "7": "encode"}
    if extra_loras:
        model_out, clip_out = _sdxl_lora_chain(g, model_out, clip_out, extra_loras, base=60)
        for i in range(len(extra_loras)):
            stages[str(60 + i)] = "load"
    g["4"] = {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": clip_out}}
    g["5"] = {"class_type": "FluxGuidance", "inputs": {"guidance": guidance, "conditioning": ["4", 0]}}
    g["6"] = {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["5", 0]}}
    g["10"] = {"class_type": "KSampler", "inputs": {
        "seed": seed, "steps": steps, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0,
        "model": model_out, "positive": ["5", 0], "negative": ["6", 0], "latent_image": ["7", 0]}}
    g["11"] = {"class_type": "VAEDecode", "inputs": {"samples": ["10", 0], "vae": ["3", 0]}}
    g["12"] = {"class_type": "SaveImage", "inputs": {"filename_prefix": "H3Workspace/flux_t2i", "images": ["11", 0]}}
    stages.update({"4": "encode", "5": "encode", "6": "encode", "10": "sample", "11": "decode", "12": "save"})
    return g, stages


def qwen_t2i_graph(m, prompt, negative, width, height, seed, extra_loras=()):
    """Qwen Image 2.1, picture by text (no reference). With a negative prompt CFG goes above 1 so that it works."""
    g = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": m["qwen_unet"], "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": m["qwen_clip"], "type": "qwen_image", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": m["qwen_vae"]}},
        "4": {"class_type": "QwenImage21Cache", "inputs": {"device": "auto", "dtype": "default", "model": ["1", 0]}},
        "7": {"class_type": "TextEncodeQwenImage21", "inputs": {
            "prompt": prompt, "negative_prompt": negative, "resolution": 0, "clip": ["2", 0], "vae": ["3", 0]}},
        "8": {"class_type": "EmptyLatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
        "9": {"class_type": "KSampler", "inputs": {
            "seed": seed, "steps": 25, "cfg": 3 if negative.strip() else 1, "sampler_name": "euler", "scheduler": "simple", "denoise": 1,
            "model": ["4", 0], "positive": ["7", 0], "negative": ["7", 1], "latent_image": ["8", 0]}},
        "10": {"class_type": "VAEDecode", "inputs": {"samples": ["9", 0], "vae": ["3", 0]}},
        "11": {"class_type": "SaveImage", "inputs": {"filename_prefix": "H3Workspace/qwen_t2i", "images": ["10", 0]}},
    }
    stages = {"1": "load", "2": "load", "3": "load", "4": "load", "7": "encode", "8": "encode",
              "9": "sample", "10": "decode", "11": "save"}
    if extra_loras:
        g["4"]["inputs"]["model"] = _lora_chain(g, ["1", 0], extra_loras, base=60)
        for i in range(len(extra_loras)):
            stages[str(60 + i)] = "load"
    return g, stages


STORY_SIZE = (1920, 1088)


def qwen_story_graph(m, images, prompt, seed, size=STORY_SIZE, extra_loras=()):
    """Storyboard from 1-3 pictures (character, location...) on Qwen Image 2.1: <image1>, <image2> in the prompt."""
    g = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": m["qwen_unet"], "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": m["qwen_clip"], "type": "qwen_image", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": m["qwen_vae"]}},
        "4": {"class_type": "QwenImage21Cache", "inputs": {"device": "auto", "dtype": "default", "model": ["1", 0]}},
    }
    stages = {"1": "load", "2": "load", "3": "load", "4": "load"}
    if extra_loras:
        g["4"]["inputs"]["model"] = _lora_chain(g, ["1", 0], extra_loras, base=60)
        for i in range(len(extra_loras)):
            stages[str(60 + i)] = "load"
    enc = {"prompt": prompt, "negative_prompt": "", "resolution": 0, "clip": ["2", 0], "vae": ["3", 0]}
    for i, img in enumerate(images, 1):
        a, b = f"10{i}", f"11{i}"
        g[a] = {"class_type": "LoadImage", "inputs": {"image": img}}
        g[b] = {"class_type": "ImageScaleToTotalPixels", "inputs": {
            "upscale_method": "lanczos", "megapixels": 1.0, "resolution_steps": 32, "image": [a, 0]}}
        enc[f"images.image_{i}"] = [b, 0]
        stages[a] = stages[b] = "encode"
    g["7"] = {"class_type": "TextEncodeQwenImage21", "inputs": enc}
    g["8"] = {"class_type": "EmptyLatentImage", "inputs": {"width": size[0], "height": size[1], "batch_size": 1}}
    g["9"] = {"class_type": "KSampler", "inputs": {
        "seed": seed, "steps": 25, "cfg": 1, "sampler_name": "euler", "scheduler": "simple", "denoise": 1,
        "model": ["4", 0], "positive": ["7", 0], "negative": ["7", 1], "latent_image": ["8", 0]}}
    g["10"] = {"class_type": "VAEDecode", "inputs": {"samples": ["9", 0], "vae": ["3", 0]}}
    g["11"] = {"class_type": "SaveImage", "inputs": {"filename_prefix": "H3Workspace/qwen_story", "images": ["10", 0]}}
    stages.update({"7": "encode", "8": "encode", "9": "sample", "10": "decode", "11": "save"})
    return g, stages


FRAME_COLS = {2: 2, 3: 3, 4: 2, 6: 3, 8: 4, 9: 3}  # frames -> columns of the sheet (rows follow)
FRAME_SIZE = (1376, 768)
FRAME_LINE = {"white": ("white", 6), "black": ("black", 4), "none": ("white", 0)}


def _glue(g, stages, titles, frame_nodes, line):
    """Glue the decoded frames into one sheet with ImageStitch (exact thin line), saved as "Лист раскадровки"."""
    n = len(frame_nodes)
    color, gap = FRAME_LINE.get(line, FRAME_LINE["white"])
    cols = FRAME_COLS.get(n, n)
    rows = [frame_nodes[k:k + cols] for k in range(0, n, cols)]
    row_out = []
    for r, items in enumerate(rows):
        cur = items[0]
        for k, nxt in enumerate(items[1:]):
            nid = f"4{r}{k}"
            g[nid] = {"class_type": "ImageStitch", "inputs": {
                "image1": cur, "direction": "right", "match_image_size": True, "spacing_width": gap, "spacing_color": color, "image2": nxt}}
            stages[nid] = "decode"
            cur = [nid, 0]
        row_out.append(cur)
    cur = row_out[0]
    for r, nxt in enumerate(row_out[1:]):
        nid = f"49{r}"
        g[nid] = {"class_type": "ImageStitch", "inputs": {
            "image1": cur, "direction": "down", "match_image_size": True, "spacing_width": gap, "spacing_color": color, "image2": nxt}}
        stages[nid] = "decode"
        cur = [nid, 0]
    g["500"] = {"class_type": "SaveImage", "inputs": {"filename_prefix": "H3Workspace/story_sheet", "images": cur}}
    stages["500"] = "save"
    titles["500"] = "Лист раскадровки"


def flux_frames_graph(m, refs, prompts, seed, size=FRAME_SIZE, line="white"):
    """Storyboard frame by frame on FLUX.1 [dev]: every frame is its own picture (saved separately, full size) and the
    sheet is glued from them by ImageStitch with an exact thin line, so there are no margins or double frames."""
    n = len(prompts)
    g = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": m["flux_unet"], "weight_dtype": "default"}},
        "2": {"class_type": "DualCLIPLoader", "inputs": {
            "clip_name1": m["flux_clip"], "clip_name2": m["flux_clip_l"], "type": "flux", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": m["flux_vae"]}},
        "13": {"class_type": "EmptySD3LatentImage", "inputs": {"width": size[0], "height": size[1], "batch_size": 1}},
        "8": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["2", 0]}},
    }
    stages = {"1": "load", "2": "load", "3": "load", "13": "encode", "8": "encode"}
    lat = []
    for i, img in enumerate(refs[:3], 1):
        a, b, v = f"10{i}", f"11{i}", f"12{i}"
        g[a] = {"class_type": "LoadImage", "inputs": {"image": img}}
        g[b] = {"class_type": "ImageScaleToTotalPixels", "inputs": {
            "upscale_method": "lanczos", "megapixels": 1.0, "resolution_steps": 16, "image": [a, 0]}}
        g[v] = {"class_type": "VAEEncode", "inputs": {"pixels": [b, 0], "vae": ["3", 0]}}
        lat.append(v)
        for k in (a, b, v):
            stages[k] = "encode"
    g["9"] = {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["8", 0]}}
    neg = ["9", 0]
    for j, v in enumerate(lat):  # the negative side does not depend on the frame: build it once
        nid = f"15{j}"
        g[nid] = {"class_type": "ReferenceLatent", "inputs": {"conditioning": neg, "latent": [v, 0]}}
        neg = [nid, 0]
        stages[nid] = "encode"
    stages["9"] = "encode"
    titles, frame_nodes = {}, []
    for i, text in enumerate(prompts):
        base = 200 + i * 10
        t = str(base)
        g[t] = {"class_type": "CLIPTextEncode", "inputs": {"text": text, "clip": ["2", 0]}}
        gd = f"{base}_gd"
        g[gd] = {"class_type": "FluxGuidance", "inputs": {"guidance": 3.5, "conditioning": [t, 0]}}
        pos = [gd, 0]
        for j, v in enumerate(lat):
            nid = str(base + 1 + j)
            g[nid] = {"class_type": "ReferenceLatent", "inputs": {"conditioning": pos, "latent": [v, 0]}}
            pos = [nid, 0]
            stages[nid] = "encode"
        sm, dc, sv = (str(base + k) for k in (7, 8, 9))
        g[sm] = {"class_type": "KSampler", "inputs": {
            "seed": seed + i, "steps": 20, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0,
            "model": ["1", 0], "positive": pos, "negative": neg, "latent_image": ["13", 0]}}
        g[dc] = {"class_type": "VAEDecode", "inputs": {"samples": [sm, 0], "vae": ["3", 0]}}
        g[sv] = {"class_type": "SaveImage", "inputs": {"filename_prefix": "H3Workspace/story_frame", "images": [dc, 0]}}
        stages.update({t: "encode", gd: "encode", sm: "sample", dc: "decode", sv: "save"})
        titles[sv] = f"Кадр {i + 1}"
        frame_nodes.append([dc, 0])
    _glue(g, stages, titles, frame_nodes, line)
    return g, stages, titles


def qwen_frames_graph(m, refs, prompts, seed, size=FRAME_SIZE, line="white", extra_loras=()):
    """The same frame-by-frame storyboard on Qwen Image 2.1 (slower than FLUX, no commercial license)."""
    n = len(prompts)
    g = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": m["qwen_unet"], "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": m["qwen_clip"], "type": "qwen_image", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": m["qwen_vae"]}},
        "4": {"class_type": "QwenImage21Cache", "inputs": {"device": "auto", "dtype": "default", "model": ["1", 0]}},
        "8": {"class_type": "EmptyLatentImage", "inputs": {"width": size[0], "height": size[1], "batch_size": 1}},
    }
    stages = {"1": "load", "2": "load", "3": "load", "4": "load", "8": "encode"}
    if extra_loras:
        g["4"]["inputs"]["model"] = _lora_chain(g, ["1", 0], extra_loras, base=60)
        for i in range(len(extra_loras)):
            stages[str(60 + i)] = "load"
    scaled = []
    for i, img in enumerate(refs[:3], 1):
        a, b = f"10{i}", f"11{i}"
        g[a] = {"class_type": "LoadImage", "inputs": {"image": img}}
        g[b] = {"class_type": "ImageScaleToTotalPixels", "inputs": {
            "upscale_method": "lanczos", "megapixels": 1.0, "resolution_steps": 32, "image": [a, 0]}}
        scaled.append(b)
        stages[a] = stages[b] = "encode"
    titles, frame_nodes = {}, []
    for i, text in enumerate(prompts):
        base = 200 + i * 10
        en, ks, dc, sv = (str(base + k) for k in (0, 1, 2, 3))
        enc = {"prompt": text, "negative_prompt": "", "resolution": 0, "clip": ["2", 0], "vae": ["3", 0]}
        for k, b in enumerate(scaled, 1):
            enc[f"images.image_{k}"] = [b, 0]
        g[en] = {"class_type": "TextEncodeQwenImage21", "inputs": enc}
        g[ks] = {"class_type": "KSampler", "inputs": {
            "seed": seed + i, "steps": 25, "cfg": 1, "sampler_name": "euler", "scheduler": "simple", "denoise": 1,
            "model": ["4", 0], "positive": [en, 0], "negative": [en, 1], "latent_image": ["8", 0]}}
        g[dc] = {"class_type": "VAEDecode", "inputs": {"samples": [ks, 0], "vae": ["3", 0]}}
        g[sv] = {"class_type": "SaveImage", "inputs": {"filename_prefix": "H3Workspace/story_frame", "images": [dc, 0]}}
        stages.update({en: "encode", ks: "sample", dc: "decode", sv: "save"})
        titles[sv] = f"Кадр {i + 1}"
        frame_nodes.append([dc, 0])
    _glue(g, stages, titles, frame_nodes, line)
    return g, stages, titles


def flux_story_graph(m, images, prompt, seed, size=STORY_SIZE):
    """Storyboard from 1-3 pictures on FLUX.1 [dev]: every picture is one more ReferenceLatent (image 1, image 2 ...)."""
    g = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": m["flux_unet"], "weight_dtype": "default"}},
        "2": {"class_type": "DualCLIPLoader", "inputs": {
            "clip_name1": m["flux_clip"], "clip_name2": m["flux_clip_l"], "type": "flux", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": m["flux_vae"]}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["2", 0]}},
        "7g": {"class_type": "FluxGuidance", "inputs": {"guidance": 3.5, "conditioning": ["7", 0]}},
        "8": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["7g", 0]}},
    }
    stages = {"1": "load", "2": "load", "3": "load", "7": "encode", "7g": "encode", "8": "encode"}
    pos, neg = ["7g", 0], ["8", 0]
    for i, img in enumerate(images, 1):
        a, b, v, rp, rn = f"10{i}", f"11{i}", f"12{i}", f"13{i}", f"14{i}"
        g[a] = {"class_type": "LoadImage", "inputs": {"image": img}}
        g[b] = {"class_type": "ImageScaleToTotalPixels", "inputs": {
            "upscale_method": "lanczos", "megapixels": 1.0, "resolution_steps": 16, "image": [a, 0]}}
        g[v] = {"class_type": "VAEEncode", "inputs": {"pixels": [b, 0], "vae": ["3", 0]}}
        g[rp] = {"class_type": "ReferenceLatent", "inputs": {"conditioning": pos, "latent": [v, 0]}}
        g[rn] = {"class_type": "ReferenceLatent", "inputs": {"conditioning": neg, "latent": [v, 0]}}
        pos, neg = [rp, 0], [rn, 0]
        for k in (a, b, v, rp, rn):
            stages[k] = "encode"
    g["13"] = {"class_type": "EmptySD3LatentImage", "inputs": {"width": size[0], "height": size[1], "batch_size": 1}}
    g["17"] = {"class_type": "KSampler", "inputs": {
        "seed": seed, "steps": 20, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0,
        "model": ["1", 0], "positive": pos, "negative": neg, "latent_image": ["13", 0]}}
    g["18"] = {"class_type": "VAEDecode", "inputs": {"samples": ["17", 0], "vae": ["3", 0]}}
    g["19"] = {"class_type": "SaveImage", "inputs": {"filename_prefix": "H3Workspace/flux_story", "images": ["18", 0]}}
    stages.update({"13": "encode", "17": "sample", "18": "decode", "19": "save"})
    return g, stages


def flux_edit_graph(m, prompt, image, megapixels, seed, extras=(), size=None, extra_loras=(), guidance=3.5, steps=20):
    """FLUX.1 [dev] FP8 (12B), edit a picture: reference latents, FluxGuidance 3.5, optional LoRA."""
    g = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": m["flux_unet"], "weight_dtype": "default"}},
        "2": {"class_type": "DualCLIPLoader", "inputs": {
            "clip_name1": m["flux_clip"], "clip_name2": m["flux_clip_l"], "type": "flux", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": m["flux_vae"]}},
        "4": {"class_type": "LoadImage", "inputs": {"image": image}},
        "5": {"class_type": "ImageScaleToTotalPixels", "inputs": {
            "upscale_method": "lanczos", "megapixels": megapixels, "resolution_steps": 16, "image": ["4", 0]}},
        "6": {"class_type": "GetImageSize", "inputs": {"image": ["5", 0]}},
        "9": {"class_type": "VAEEncode", "inputs": {"pixels": ["5", 0], "vae": ["3", 0]}},
    }
    model_out, clip_out = ["1", 0], ["2", 0]
    stages = {"1": "load", "2": "load", "3": "load", "4": "load", "5": "encode", "6": "encode", "9": "encode"}
    if extra_loras:
        model_out, clip_out = _sdxl_lora_chain(g, model_out, clip_out, extra_loras, base=60)
        for i in range(len(extra_loras)):
            stages[str(60 + i)] = "load"
    g["7"] = {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": clip_out}}
    g["7g"] = {"class_type": "FluxGuidance", "inputs": {"guidance": guidance, "conditioning": ["7", 0]}}
    g["8"] = {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["7g", 0]}}
    g["10"] = {"class_type": "ReferenceLatent", "inputs": {"conditioning": ["7g", 0], "latent": ["9", 0]}}
    g["11"] = {"class_type": "ReferenceLatent", "inputs": {"conditioning": ["8", 0], "latent": ["9", 0]}}
    stages.update({"7": "encode", "7g": "encode", "8": "encode", "10": "encode", "11": "encode"})
    pos, neg = ["10", 0], ["11", 0]
    for k, img in enumerate(list(extras)[:2]):  # more reference pictures: image 2, image 3
        a, b, v, rp, rn = (f"{p}{k}" for p in ("30", "31", "32", "33", "34"))
        g[a] = {"class_type": "LoadImage", "inputs": {"image": img}}
        g[b] = {"class_type": "ImageScaleToTotalPixels", "inputs": {
            "upscale_method": "lanczos", "megapixels": megapixels, "resolution_steps": 16, "image": [a, 0]}}
        g[v] = {"class_type": "VAEEncode", "inputs": {"pixels": [b, 0], "vae": ["3", 0]}}
        g[rp] = {"class_type": "ReferenceLatent", "inputs": {"conditioning": pos, "latent": [v, 0]}}
        g[rn] = {"class_type": "ReferenceLatent", "inputs": {"conditioning": neg, "latent": [v, 0]}}
        pos, neg = [rp, 0], [rn, 0]
        for nid in (a, b, v, rp, rn):
            stages[nid] = "encode"
    g["13"] = {"class_type": "EmptySD3LatentImage", "inputs": {"width": ["6", 0], "height": ["6", 1], "batch_size": 1}}
    g["17"] = {"class_type": "KSampler", "inputs": {
        "seed": seed, "steps": steps, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0,
        "model": model_out, "positive": pos, "negative": neg, "latent_image": ["13", 0]}}
    g["18"] = {"class_type": "VAEDecode", "inputs": {"samples": ["17", 0], "vae": ["3", 0]}}
    g["19"] = {"class_type": "SaveImage", "inputs": {"filename_prefix": "H3Workspace/flux", "images": ["18", 0]}}
    if size:  # own size: the picture is scaled to it, the result follows
        g["5"] = {"class_type": "ImageScale", "inputs": {
            "image": ["4", 0], "upscale_method": "lanczos", "width": size[0], "height": size[1], "crop": "center"}}
    stages.update({"13": "encode", "17": "sample", "18": "decode", "19": "save"})
    return g, stages


def qwen_edit_size(aspect, megapixels):
    """Canvas for a chosen aspect ratio (about 1 or 2 megapixels, sides in steps of 32); None = keep the picture's own shape."""
    if aspect not in KREA_SIZES:
        return None
    w, h = KREA_SIZES[aspect]
    k = (megapixels or 1.0) ** 0.5
    return (round(w * k / 32) * 32, round(h * k / 32) * 32)


def qwen_edit_graph(m, prompt, image, megapixels, seed, size=None, extras=(), extra_loras=()):
    return _qwen_base(m, image, megapixels, prompt, seed, size, extras, extra_loras=extra_loras)


def sheet_prompt(headless=True, background="grey", marks="", panels=4):
    """Four panels: full body front (without the head by default), full body back, close-up front, close-up profile.
    Tested on Qwen Image 2.1 with a cartoon character: same style and outfit on all panels."""
    bg = SHEET_BG.get(background, "light grey")
    first = ("full-body front view, the figure has no head at all, not a mannequin and not a neck stump, the collar is "
             "simply empty and the plain background shows through the neckline, standing in a relaxed neutral pose, "
             "showing the whole outfit from the neckline to the shoes." if headless else
             "full-body front view facing the viewer, standing in a relaxed neutral pose, showing the whole outfit "
             "from head to shoes.")
    keep = "Keep the same face, hairstyle, " + (marks.strip().rstrip(".,") + ", " if marks.strip() else "") \
        + "clothing, colors, proportions and art style as in <image1> in every panel."
    if panels == 6:
        return (
            "A character reference sheet of the character from <image1>, split into six equal tall panels side by side in one "
            "row, all at the same scale, in the same art style as <image1>.\n"
            f"Panel one: {first}\n"
            "Panel two: full-body three-quarter view of the same character, turned about forty-five degrees to the left, "
            "same pose and outfit, head fully visible.\n"
            "Panel three: full-body side view in profile facing left, same pose and outfit, head fully visible.\n"
            "Panel four: full-body back view of the same character, same pose and outfit, seen from behind, head fully visible.\n"
            "Panel five: close-up head-and-shoulders portrait facing the viewer with a neutral expression.\n"
            "Panel six: close-up head-and-shoulders portrait in three-quarter view, with the same lighting and scale as panel five.\n"
            f"{keep}\n"
            f"Plain solid {bg} background, soft even lighting, no text, no labels.")
    return (
        "A character reference sheet of the character from <image1>, split into four equal panels side by side in one "
        "row, all at the same scale, in the same art style as <image1>.\n"
        f"Panel one: {first}\n"
        "Panel two: full-body back view of the same character in the same pose and outfit, seen from behind, head fully visible.\n"
        "Panel three: close-up head-and-shoulders portrait facing the viewer with a neutral expression.\n"
        "Panel four: close-up head-and-shoulders portrait in profile, with the same lighting and scale as panel three.\n"
        f"{keep}\n"
        f"Plain solid {bg} background, soft even lighting, no text, no labels.")


def qwen_sheet_graph(m, image, seed, headless=True, background="grey", marks="", panels=4, extra_loras=()):
    prompt = sheet_prompt(headless, background, marks, panels)
    size = SHEET6_SIZE if panels == 6 else SHEET_SIZE
    g, stages = _qwen_base(m, image, size[0] * size[1] / 1048576, prompt, seed, size=size, extra_loras=extra_loras)
    return g, stages, prompt


# ---------- improve quality of a finished video: SeedVR2 on plain ComfyUI nodes, one step, tiled ----------

UPSCALE_SCALES = {"1.5": 1.5, "2": 2.0}
UPSCALE_SEC_PER_MP_FRAME = 3.24  # measured on an RTX 4060 Ti 16 GB: 5 s of 1056x608 -> 1584x912 took ~10 minutes


def upscale_image_graph(m, image, scale, seed, soften=0.0):
    """The same SeedVR2 one-step enlarge for a single picture."""
    g = {
        "1": {"class_type": "LoadImage", "inputs": {"image": image}},
        "3": {"class_type": "ImageScaleBy", "inputs": {"upscale_method": "lanczos", "scale_by": scale, "image": ["1", 0]}},
        "4": {"class_type": "SeedVR2Preprocess", "inputs": {"resized_images": ["3", 0]}},
        "5": {"class_type": "VAELoader", "inputs": {"vae_name": m["seedvr_vae"]}},
        "6": {"class_type": "UNETLoader", "inputs": {"unet_name": m["seedvr_unet"], "weight_dtype": "default"}},
        "7": {"class_type": "VAEEncodeTiled", "inputs": {
            "tile_size": 512, "overlap": 128, "temporal_size": 64, "temporal_overlap": 8, "pixels": ["4", 0], "vae": ["5", 0]}},
        "8": {"class_type": "SeedVR2Conditioning", "inputs": {"model": ["6", 0], "vae_conditioning": ["7", 0]}},
        "9": {"class_type": "KSampler", "inputs": {
            "seed": seed, "steps": 1, "cfg": 1, "sampler_name": "euler", "scheduler": "simple", "denoise": 1,
            "model": ["6", 0], "positive": ["8", 0], "negative": ["8", 1], "latent_image": ["7", 0]}},
        "10": {"class_type": "VAEDecodeTiled", "inputs": {
            "tile_size": 512, "overlap": 128, "temporal_size": 64, "temporal_overlap": 8, "samples": ["9", 0], "vae": ["5", 0]}},
        "11": {"class_type": "SeedVR2PostProcessing", "inputs": {
            "color_correction_method": "lab", "images": ["10", 0], "original_resized_images": ["3", 0]}},
        "12": {"class_type": "SaveImage", "inputs": {"filename_prefix": "H3Workspace/upscale", "images": ["11", 0]}},
    }
    stages = {"1": "load", "5": "load", "6": "load", "3": "encode", "4": "encode", "7": "encode", "8": "encode",
              "9": "sample", "10": "decode", "11": "decode", "12": "save"}
    if soften:  # soft result: SeedVR2 mixed with the plain enlarged picture (less "digital" sharpness)
        g["13"] = {"class_type": "ImageBlend", "inputs": {"image1": ["11", 0], "image2": ["3", 0], "blend_factor": soften, "blend_mode": "normal"}}
        g["12"]["inputs"]["images"] = ["13", 0]
        stages["13"] = "decode"
    return g, stages


RELAY_PIXEL_SECONDS = 10 * 1376 * 768  # the whole clip is decoded in memory: about 10 s at 1376x768 is the limit (author's note)
RELAY_SEC_PER_MP_FRAME = 0.7   # the author: 5 s of 720p in ~22 s on an RTX 5090 (0.2 s per megapixel-frame); a 4060 Ti is ~3.5x slower: an estimate, not measured here


def relay_graph(m, video, seed, detail=1.0):
    """RelayVSR x2: the whole video enlarged 2x in one sampling step (ComfyUI's built-in nodes), sound and fps kept."""
    g = {
        "1": {"class_type": "LoadVideo", "inputs": {"file": video}},
        "2": {"class_type": "GetVideoComponents", "inputs": {"video": ["1", 0]}},
        "3": {"class_type": "UNETLoader", "inputs": {"unet_name": m["relayvsr_unet"], "weight_dtype": "default"}},
        "4": {"class_type": "RelayVSRConditioning", "inputs": {"model": ["3", 0], "images": ["2", 0]}},
        "5": {"class_type": "KSampler", "inputs": {
            "seed": seed, "steps": 1, "cfg": 1, "sampler_name": "euler", "scheduler": "simple", "denoise": 1,
            "model": ["3", 0], "positive": ["4", 0], "negative": ["4", 0], "latent_image": ["4", 1]}},
        "6": {"class_type": "VAELoader", "inputs": {"vae_name": m["relayvsr_vae"]}},
        "7": {"class_type": "VSRDecode", "inputs": {"vae": ["6", 0], "samples": ["5", 0], "images": ["2", 0], "detail": detail}},
        "8": {"class_type": "CreateVideo", "inputs": {"images": ["7", 0], "audio": ["2", 1], "fps": ["2", 2]}},
        "9": {"class_type": "SaveVideo", "inputs": {
            "video": ["8", 0], "filename_prefix": "H3Workspace/upscale", "format": "auto", "format.codec": "auto"}},
    }
    stages = {"1": "load", "2": "load", "3": "load", "6": "load", "4": "encode", "5": "sample", "7": "decode", "8": "save", "9": "save"}
    return g, stages


def upscale_graph(m, video, scale, seed, soften=0.0):
    g = {
        "1": {"class_type": "LoadVideo", "inputs": {"file": video}},
        "2": {"class_type": "GetVideoComponents", "inputs": {"video": ["1", 0]}},
        "3": {"class_type": "ImageScaleBy", "inputs": {"upscale_method": "lanczos", "scale_by": scale, "image": ["2", 0]}},
        "4": {"class_type": "SeedVR2Preprocess", "inputs": {"resized_images": ["3", 0]}},
        "5": {"class_type": "VAELoader", "inputs": {"vae_name": m["seedvr_vae"]}},
        "6": {"class_type": "UNETLoader", "inputs": {"unet_name": m["seedvr_unet"], "weight_dtype": "default"}},
        "7": {"class_type": "VAEEncodeTiled", "inputs": {
            "tile_size": 512, "overlap": 128, "temporal_size": 64, "temporal_overlap": 8, "pixels": ["4", 0], "vae": ["5", 0]}},
        "8": {"class_type": "SeedVR2Conditioning", "inputs": {"model": ["6", 0], "vae_conditioning": ["7", 0]}},
        "9": {"class_type": "KSampler", "inputs": {
            "seed": seed, "steps": 1, "cfg": 1, "sampler_name": "euler", "scheduler": "simple", "denoise": 1,
            "model": ["6", 0], "positive": ["8", 0], "negative": ["8", 1], "latent_image": ["7", 0]}},
        "10": {"class_type": "VAEDecodeTiled", "inputs": {
            "tile_size": 512, "overlap": 128, "temporal_size": 64, "temporal_overlap": 8, "samples": ["9", 0], "vae": ["5", 0]}},
        "11": {"class_type": "SeedVR2PostProcessing", "inputs": {
            "color_correction_method": "lab", "images": ["10", 0], "original_resized_images": ["3", 0]}},
        "12": {"class_type": "CreateVideo", "inputs": {"images": ["11", 0], "audio": ["2", 1], "fps": ["2", 2]}},
        "13": {"class_type": "SaveVideo", "inputs": {
            "video": ["12", 0], "filename_prefix": "H3Workspace/upscale", "format": "auto", "format.codec": "auto"}},
    }
    stages = {"1": "load", "2": "load", "5": "load", "6": "load", "3": "encode", "4": "encode", "7": "encode", "8": "encode",
              "9": "sample", "10": "decode", "11": "decode", "12": "save", "13": "save"}
    if soften:  # softer result: SeedVR2 mixed with the plain enlarged video (the share of the plain one is `soften`)
        g["14"] = {"class_type": "ImageBlend", "inputs": {"image1": ["11", 0], "image2": ["3", 0], "blend_factor": soften, "blend_mode": "normal"}}
        g["12"]["inputs"]["images"] = ["14", 0]
        stages["14"] = "decode"
    return g, stages
