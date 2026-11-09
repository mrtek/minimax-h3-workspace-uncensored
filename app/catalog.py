"""What a fresh install needs: ComfyUI build, node packs, and model variants per GPU class.

Every model has a role (what the workflows use it for). A role can have several
variants; `choose_models` picks one per role from the detected hardware.
"""

COMFY_VERSION = "0.43.0"
MIN_DRIVER = 580  # the bundled PyTorch is built for CUDA 13
COMFY_URL = ("https://github.com/Comfy-Org/ComfyUI/releases/download/"
             f"v{COMFY_VERSION}/ComfyUI_windows_portable_nvidia.7z")
COMFY_SIZE = 2_040_000_000
COMFY_DIRNAME = "ComfyUI_windows_portable"

# Downloaded as GitHub archives of a fixed commit, so users need no git.
NODE_PACKS = [
    {"name": "ComfyUI-H3-NativeAudioLock", "repo": "SveSop/ComfyUI-H3-NativeAudioLock",
     "commit": "699e58e51b1c8e12b8c0f0dd55da603d870d20e6"},
    {"name": "Comfyui_Minimax_h3_latent_Upscaler", "repo": "LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler",
     "commit": "40316cf008b2fd8663263270669eb4da23f89d2c"},
    # separates the voice from music in the lipsync audio editor
    {"name": "ComfyUI-MelBandRoFormer", "repo": "kijai/ComfyUI-MelBandRoFormer",
     "commit": "92c86854e6654f4aacc97484471af95c98ea16d4"},
    # loads reference videos at 24 fps for the references mode
    {"name": "ComfyUI-VideoHelperSuite", "repo": "Kosinkadink/ComfyUI-VideoHelperSuite",
     "commit": "4d907bee61e92c2e65af3bd6383a4e4d356126d1"},
    # memory savers (MiniMaxChunkFeedForward, MiniMaxLowVRAMAttention) used by the long video graphs
    {"name": "ComfyUI-KJNodes", "repo": "kijai/ComfyUI-KJNodes",
     "commit": "ab72eda7ffad6280422d71ab7d54d3fb87c0ba18"},
]

# What each mode needs from custom nodes: checked in a user's own ComfyUI (a new install gets them by itself).
# `classes` are node names ComfyUI must know; `optional` ones only switch a feature on.
NODE_CHECKS = [
    {"pack": "ComfyUI-H3-NativeAudioLock", "classes": ["MiniMaxH3NativeAudioLock"], "modes": ["lipsync"],
     "optional": False, "why": "точный липсинг под ваше аудио"},
    {"pack": "Comfyui_Minimax_h3_latent_Upscaler", "classes": ["MinimaxH3LatentUpscaler3D"], "modes": ["lipsync", "action"],
     "optional": False, "why": "второй проход (увеличение кадра)"},
    {"pack": "ComfyUI-MelBandRoFormer", "classes": ["MelBandRoFormerSampler", "MelBandRoFormerModelLoader"],
     "modes": ["lipsync"], "optional": True, "why": "кнопка «Отделить голос» в редакторе аудио"},
    {"pack": "ComfyUI-VideoHelperSuite", "classes": ["VHS_LoadVideo"], "modes": ["refs", "swap"],
     "optional": False, "why": "видео, в котором меняем персонажа"},
    {"pack": "ComfyUI-KJNodes", "classes": ["MiniMaxChunkFeedForward", "MiniMaxLowVRAMAttention"], "modes": ["long"],
     "optional": False, "why": "экономия видеопамяти в длинном видео"},
]


def node_repo(pack):
    return next(p["repo"] for p in NODE_PACKS if p["name"] == pack)


HF = "https://huggingface.co/{repo}/resolve/main/{path}"
SUB = "MinimaxH3"  # subfolder inside each model category, same as the original workflows
IMG_SUB = "Images"


def _m(repo, path, size_gb, category, name=None, sub=SUB):
    """A downloadable file. `name` is how it is saved (defaults to the file name in the repo)."""
    fname = name or path.rsplit("/", 1)[-1]
    return {"url": HF.format(repo=repo, path=path), "size": int(size_gb * 1e9), "category": category,
            "file": fname, "name": f"{sub}\\{fname}" if sub else fname}


# the same file is published under two names; either one counts as present
FILE_ALIASES = {
    "minimax_h3_latent_upscaler_3d_bf16.safetensors": ("minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors",),
}

VARIANTS = {
    "fast_unet": {
        "int8": _m("FastVideo/FastVideo-FastH3-Comfy",
                   "diffusion_models/fastvideo_fasth3_8step_v2_pruned_int8_convrot.safetensors", 22.13, "diffusion_models"),
        "nvfp4": _m("FastVideo/FastVideo-FastH3-8-Step-V2-NVFP4-Comfy",
                    "diffusion_models/fastvideo_fasth3_8step_v2_pruned_nvfp4.safetensors", 13.61, "diffusion_models"),
    },
    "lipsync_unet": {
        "full": _m("WarmBloodAban/Minimax-h3_Singularity",
                   "Minimax-h3_Singularity_ref2va_v1.3_int8.safetensors", 34.0, "diffusion_models"),
        "pruned": _m("WarmBloodAban/Minimax-h3_Singularity",
                     "Minimax-h3_Singularity_ref2va_Pruned_v1.3_int8.safetensors", 20.97, "diffusion_models"),
        "lite": _m("WarmBloodAban/Minimax-h3_Singularity",
                   "Minimax-h3_Singularity_ref2va_v1.3_Pruned_w4a8.safetensors", 11.77, "diffusion_models"),
        # v2: viewers reported that v1.3 generates sound badly
        "v2": _m("WarmBloodAban/Minimax-h3_Singularity",
                 "Minimax-h3_Singularity_ref2va_v2_pruned_int8.safetensors", 20.97, "diffusion_models"),
        "v2_full": _m("WarmBloodAban/Minimax-h3_Singularity",
                      "Minimax-h3_Singularity_ref2va_v2_int8.safetensors", 34.0, "diffusion_models"),
    },
    "text_encoder": {
        "nvfp4": _m("Comfy-Org/MiniMax-H3", "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
                    15.69, "text_encoders"),
    },
    "video_vae": {
        "int8": _m("Comfy-Org/MiniMax-H3", "vae/minimax_h3_video_vae_int8_convrot.safetensors", 2.81, "vae"),
    },
    "audio_vae": {
        "fp32": _m("Comfy-Org/MiniMax-H3", "vae/minimax_h3_audio_vae_fp32.safetensors", 0.61, "vae"),
    },
    "ref2v_turbo_lora": {
        "bf16": _m("Comfy-Org/MiniMax-H3", "loras/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors",
                   1.96, "loras"),
    },
    "upscaler": {
        # same file (same sha256) the original workflow knows as minimax_h3_latent_upscaler_3d_bf16
        "bf16": _m("LBH-123-AI/Minimax_h3_latent_Upscaler",
                   "minimax_h3_latent_upscaler_3d_conv_v1/minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors",
                   0.69, "latent_upscale_models", name="minimax_h3_latent_upscaler_3d_bf16.safetensors", sub=None),
    },
    # Turbo-SLA (lightx2v): 4-step LoRA for the first/last frame model; used on top of FastH3 for the "fast" switch
    "sla_lora": {
        "bf16": _m("lightx2v/Minimax-h3-Turbo-SLA", "minimax_h3_fl2v_turbo_4step_v0.1_768p_sla_comfyui_bf16.safetensors",
                   1.96, "loras"),
    },
    "preview": {
        "taeh3": _m("Alissonerdx/Minimax-H3-ComfyUI", "vae_approx/taeh3.safetensors", 0.02, "vae_approx", sub=None),
    },
    # voice / music separation (MelBand RoFormer), used by the audio editor in Lipsync
    "vocal_model": {
        "fp16": _m("Kijai/MelBandRoFormer_comfy", "MelBandRoformer_fp16.safetensors", 0.456, "diffusion_models"),
    },
    # action scenes (Prompt Mastery workflow): FL2VA/Ref2VA hybrid
    "action_unet": {
        "int8": _m("smhfacct/Minimax-H3-fl2va-ref2va-hybrid-models",
                   "minimax_h3_hybrid_fl2va_ref2va_b25-49-int8.safetensors", 20.97, "diffusion_models"),
        "lite": _m("binglingzhimeng/minimax_h3_hybrid_fl2va_ref2va_b25-49_w6a8",
                   "minimax_h3_hybrid_fl2va_ref2va_b25-49_w6a8.safetensors", 15.98, "diffusion_models"),
    },
    "speed_slider_lora": {
        "v1.1": _m("RunningHubAI/rh-h3-speed-slider-1.1-lora", "H3_speed_slider_1.1.safetensors", 0.01, "loras"),
    },
    "action_upscaler": {
        "lms": _m("Alissonerdx/Minimax-H3-ComfyUI", "latent_upscaler/h3_upscaler_lms_v0.1.safetensors",
                  0.69, "latent_upscale_models", sub=None),
    },
    # optional LoRAs for the references mode
    "realism_lora": {
        "v1": _m("fal/MiniMax-H3-Realism-People-LoRA", "h3-realism-people-t2v-i2v-r2v.safetensors", 0.13, "loras"),
    },
    "motion_repair_lora": {
        "v1": _m("JOKER141/MiniMax-H3-General-Motion-Continuity-Repair", "Motion_Repair.safetensors", 0.16, "loras"),
    },
    # Character swap (akatz-ai): replaces a person in a video with a character from a picture
    "swap_lora": {
        "v15": _m("akatz-ai/MiniMax-H3-Character-Swap-LoRA", "h3_character_swap_v1.5.safetensors", 0.16, "loras"),
    },
    # Pika H3 Flex Turbo: 8-step HD LoRA for references; a viewer favourite for the generated sound
    "pika_lora": {
        "hd8": _m("Pika-Labs/Pika-H3-Flex-Turbo-LoRA", "pika_h3_flex_turbo_8step_hd_ref2va.safetensors", 3.91, "loras"),
    },
    # improve quality of a finished video (SeedVR2, plain ComfyUI nodes)
    "seedvr_unet": {
        "int8": _m("Comfy-Org/SeedVR2", "diffusion_models/seedvr2_3b_int8_convrot.safetensors", 3.46,
                   "diffusion_models", sub=IMG_SUB),
    },
    "seedvr_vae": {
        "fp16": _m("Comfy-Org/SeedVR2", "vae/seedvr2_ema_vae_fp16.safetensors", 0.5, "vae", sub=IMG_SUB),
    },
    # fast video enlarge x2 in one sampling step (RelayVSR, built into ComfyUI); video only
    "relayvsr_unet": {
        "x2": _m("Comfy-Org/RelayVSR", "diffusion_models/relayvsr_x2_lightx2v_int8_convrot.safetensors", 5.16,
                 "diffusion_models", sub=IMG_SUB),
    },
    "relayvsr_vae": {
        "x2": _m("Comfy-Org/RelayVSR", "vae/relayvsr_flash_vae_x2_fp32.safetensors", 0.13, "vae", sub=IMG_SUB),
    },
    # pictures: Krea 2 (by text) and Qwen Image 2.1 (edit a picture, character sheet); both are plain ComfyUI nodes
    "krea_unet": {
        "int8": _m("Comfy-Org/Krea-2", "diffusion_models/krea2_turbo_int8_convrot.safetensors", 13.49,
                   "diffusion_models", sub=IMG_SUB),
    },
    "krea_clip": {
        "fp8": _m("Comfy-Org/Krea-2", "text_encoders/qwen3vl_4b_fp8_scaled.safetensors", 5.24,
                  "text_encoders", sub=IMG_SUB),
    },
    "krea_vae": {
        "bf16": _m("Comfy-Org/Krea-2", "vae/qwen_image_vae.safetensors", 0.25, "vae", sub=IMG_SUB),
    },
    # Krea 2 for realistic people and photos: RedCraft RedMix + the Realism Engine LoRA (two passes)
    "kreareal_unet": {
        "fp8": _m("Kerstal/krea-2_model", "redcraftKREA2RedMix_krea2Edition.safetensors", 13.14,
                  "diffusion_models", sub=IMG_SUB),
    },
    "kreareal_lora": {
        "v2": _m("uzumix/realism_engine_krea2_v2", "realism_engine_krea2_v2.safetensors", 1.56, "loras", sub=IMG_SUB),
    },
    "qwen_unet": {
        "int8": _m("Comfy-Org/Qwen-Image-2.1", "diffusion_models/qwen_image_2.1_int8_convrot.safetensors", 7.26,
                   "diffusion_models", sub=IMG_SUB),
    },
    "qwen_clip": {
        "uncensored": _m("paralaif/Huihui-Qwen3-VL-8B-Instruct-abliterated-comfy-int8-convrot",
                         "Huihui-Qwen3-VL-8B-Instruct-abliterated_int8_convrot.safetensors", 9.29,
                         "text_encoders", sub=IMG_SUB),
        "int8": _m("Comfy-Org/Qwen-Image-2.1", "text_encoders/qwen3vl_8b_int8_convrot.safetensors", 9.35,
                   "text_encoders", sub=IMG_SUB),
    },
    "qwen_vae": {
        "bf16": _m("Comfy-Org/Qwen-Image-2.1", "vae/qwen_image_2.1_vae_bf16.safetensors", 0.68, "vae", sub=IMG_SUB),
    },
    # FLUX.2 klein 4B (Apache 2.0, free for commercial use): edit a picture by text
    "flux_unet": {
        "fp8": _m("black-forest-labs/FLUX.2-klein-4b-fp8", "flux-2-klein-4b-fp8.safetensors", 4.07,
                  "diffusion_models", sub=IMG_SUB),
    },
    "flux_clip": {
        "uncensored": _m("ussoewwin/qwen3_4b_8b_abliterated_fp16_convrot_int8",
                         "qwen3_4b_abliterated_fp16_converted.safetensors", 8.04,
                         "text_encoders", sub=IMG_SUB),
        "bf16": _m("Comfy-Org/z_image_turbo", "split_files/text_encoders/qwen_3_4b.safetensors", 8.04,
                   "text_encoders", sub=IMG_SUB),
    },
    "flux_vae": {
        "bf16": _m("Comfy-Org/flux2-dev", "split_files/vae/flux2-vae.safetensors", 0.34, "vae", sub=IMG_SUB),
    },
}

# What a file name of each role must contain (any of these words, any case). Used only to warn when a file is picked
# by hand that does not look like the right one, and to say what to check when ComfyUI fails with a shape error.
ROLE_MARKS = {
    "fast_unet": ["fasth3", "h3"], "lipsync_unet": ["singularity", "h3"], "text_encoder": ["32b"],
    "video_vae": ["video_vae"], "audio_vae": ["audio_vae"], "ref2v_turbo_lora": ["turbo"],
    "upscaler": ["upscaler"], "preview": ["taeh3"], "action_unet": ["hybrid", "fl2va"],
    "speed_slider_lora": ["speed"], "action_upscaler": ["upscaler"], "vocal_model": ["melband"],
    "realism_lora": ["realism"], "motion_repair_lora": ["motion"], "pika_lora": ["pika"], "swap_lora": ["swap"], "sla_lora": ["sla"],
    "seedvr_unet": ["seedvr"], "seedvr_vae": ["seedvr"], "relayvsr_unet": ["relayvsr"], "relayvsr_vae": ["relayvsr"], "krea_clip": ["qwen3vl", "qwen3-vl"],
    "krea_vae": ["qwen_image", "qwen-image"], "kreareal_unet": ["krea"], "kreareal_lora": ["realism"],
    "qwen_unet": ["qwen"], "qwen_clip": ["qwen3vl", "qwen3-vl", "abliterated"], "qwen_vae": ["qwen"],
    "flux_unet": ["flux"], "flux_clip": ["qwen_3", "qwen3", "abliterated"], "flux_vae": ["flux"],
}


def looks_right(role, filename):
    """False only when the role has a name rule and the file name breaks it."""
    marks = ROLE_MARKS.get(role)
    n = filename.replace("/", "\\").split("\\")[-1].lower()
    return not marks or any(m in n for m in marks)


def marks_text(role):
    return " или ".join(f"«{m}»" for m in ROLE_MARKS.get(role, [])[:2])


# Plain words for the "another version" choice in the Models section (only versions of the same model replace each other)
VARIANT_LABELS = {
    "fast_unet": {"int8": "обычная (int8)", "nvfp4": "лёгкая NVFP4 — только для видеокарт RTX 50-й серии"},
    "lipsync_unet": {"full": "v1.3 полная (самая тяжёлая)", "pruned": "v1.3 облегчённая", "lite": "v1.3 лёгкая (слабее качество)",
                     "v2": "v2 облегчённая — по отзывам, звук лучше", "v2_full": "v2 полная"},
    "action_unet": {"int8": "обычная (int8)", "lite": "лёгкая (меньше памяти, чуть слабее)"},
    "qwen_clip": {"uncensored": "Uncensored (Abliterated int8)", "int8": "оригинальная (Comfy-Org int8)"},
    "flux_clip": {"uncensored": "Uncensored (Abliterated fp16)", "bf16": "оригинальная (Comfy-Org bf16)"},
}

# LoRA from the original action workflow that has no public download; used when the user has it
ACTION_DMAD_LORA = "minimax_h3_dmad_full_dareties_v4_step600_comfy_fro"

ROLE_TITLES = {
    "fast_unet": "Основная модель — Image-to-Video (FastH3)",
    "lipsync_unet": "Основная модель — Lipsync (Singularity)",
    "text_encoder": "Текстовый энкодер (Qwen3-VL 32B)",
    "video_vae": "Видео VAE",
    "audio_vae": "Аудио VAE",
    "ref2v_turbo_lora": "LoRA ускорения для Lipsync",
    "upscaler": "Апскейлер второго прохода",
    "preview": "Декодер живого превью",
    "action_unet": "Основная модель — Экшн-сцены (Hybrid FL2VA/Ref2VA)",
    "speed_slider_lora": "LoRA Speed Slider (экшн)",
    "action_upscaler": "Апскейлер SelfLift (экшн, точный режим)",
    "vocal_model": "Отделение голоса от музыки (MelBand RoFormer)",
    "realism_lora": "LoRA «Реализм людей» (по желанию)",
    "motion_repair_lora": "LoRA «Исправление движения» (по желанию)",
    "swap_lora": "LoRA «Замена персонажа» (akatz-ai)",
    "pika_lora": "LoRA Pika H3 Flex Turbo (ускорение для Референсов, хороший звук)",
    "sla_lora": "LoRA Turbo-SLA (быстрое ускорение для Image-to-Video)",
    "seedvr_unet": "Увеличение видео и картинок — модель SeedVR2 (3B)",
    "seedvr_vae": "Увеличение видео и картинок — VAE SeedVR2",
    "relayvsr_unet": "Быстрое увеличение видео — модель RelayVSR x2",
    "relayvsr_vae": "Быстрое увеличение видео — VAE RelayVSR",
    "krea_unet": "Основная модель — Картинка по тексту (Krea 2)",
    "krea_clip": "Текстовый энкодер Krea 2 (Qwen3-VL 4B)",
    "krea_vae": "VAE для Krea 2",
    "kreareal_unet": "Основная модель — Картинка: реализм (RedCraft, Krea 2)",
    "kreareal_lora": "LoRA Realism Engine для Krea 2 (реализм)",
    "qwen_unet": "Основная модель — Правка картинки (Qwen Image 2.1)",
    "qwen_clip": "Текстовый энкодер Qwen (Qwen3-VL 8B)",
    "qwen_vae": "VAE для Qwen Image 2.1",
    "flux_unet": "Основная модель — Правка картинки для коммерции (FLUX.2 klein 4B)",
    "flux_clip": "Текстовый энкодер FLUX.2 klein (Qwen3 4B)",
    "flux_vae": "VAE для FLUX.2",
}

MODE_TITLES = {"video": "Image-to-Video", "lipsync": "Lipsync", "refs": "Референсы", "actor": "Актёрская игра", "swap": "Замена персонажа", "action": "Экшн-сцены",
               "krea": "Картинка по тексту", "kreareal": "Картинка: реализм", "qwen": "Правка картинки", "flux": "Правка картинки: FLUX.2 (можно в коммерцию)",
               "upscale": "Увеличение видео и картинок", "relayvsr": "Быстрое увеличение видео (RelayVSR)", "long": "Длинное видео"}

# Which roles each mode needs.
MODE_ROLES = {
    "video": ["fast_unet", "text_encoder", "video_vae", "audio_vae"],
    "lipsync": ["lipsync_unet", "text_encoder", "video_vae", "audio_vae", "ref2v_turbo_lora", "upscaler"],
    "refs": ["lipsync_unet", "text_encoder", "video_vae", "audio_vae", "ref2v_turbo_lora"],
    # same models as references: dialogue scenes with a voice sample, acting-first prompts
    "actor": ["lipsync_unet", "text_encoder", "video_vae", "audio_vae", "ref2v_turbo_lora"],
    "action": ["action_unet", "text_encoder", "video_vae", "audio_vae", "upscaler"],
    # character swap preset: same Ref2V model as References + a small swap LoRA
    "swap": ["lipsync_unet", "text_encoder", "video_vae", "audio_vae", "ref2v_turbo_lora", "swap_lora"],
    # long video / clips: scenes that continue each other (Singularity Ref2V + Pika LoRA, same files as References)
    "long": ["lipsync_unet", "text_encoder", "video_vae", "audio_vae", "pika_lora"],
    "krea": ["krea_unet", "krea_clip", "krea_vae"],
    "kreareal": ["kreareal_unet", "kreareal_lora", "krea_clip", "krea_vae"],
    "qwen": ["qwen_unet", "qwen_clip", "qwen_vae"],
    "flux": ["flux_unet", "flux_clip", "flux_vae"],
    "upscale": ["seedvr_unet", "seedvr_vae"],
    "relayvsr": ["relayvsr_unet", "relayvsr_vae"],
}
# Small extras downloaded with a mode but not required to run it.
MODE_OPTIONAL = {
    "video": ["sla_lora"],
    "lipsync": ["realism_lora", "vocal_model"],
    "refs": ["pika_lora", "realism_lora", "motion_repair_lora"],
    "actor": ["realism_lora"],
    "swap": ["pika_lora", "realism_lora", "motion_repair_lora"],
    "action": ["speed_slider_lora", "action_upscaler"],
    "long": ["vocal_model"],
}
ALL_ROLES = list(VARIANTS)

# The user's original files, for connecting to an existing ComfyUI that already has them.
DEFAULT_CHOICE = {"fast_unet": "int8", "lipsync_unet": "full", "text_encoder": "nvfp4", "video_vae": "int8",
                  "audio_vae": "fp32", "ref2v_turbo_lora": "bf16", "upscaler": "bf16", "preview": "taeh3",
                  "action_unet": "int8", "speed_slider_lora": "v1.1", "action_upscaler": "lms", "vocal_model": "fp16",
                  "realism_lora": "v1", "motion_repair_lora": "v1", "pika_lora": "hd8", "swap_lora": "v15", "sla_lora": "bf16",
                  "krea_unet": "int8", "krea_clip": "fp8", "krea_vae": "bf16", "kreareal_unet": "fp8", "kreareal_lora": "v2",
                  "qwen_unet": "int8", "qwen_clip": "uncensored", "qwen_vae": "bf16",
                  "flux_unet": "fp8", "flux_clip": "uncensored", "flux_vae": "bf16",
                  "seedvr_unet": "int8", "seedvr_vae": "fp16", "relayvsr_unet": "x2", "relayvsr_vae": "x2"}


PROFILES = {
    "lite": {"title": "Лёгкий", "lipsync_unet": "lite", "action_unet": "lite",
             "about": "Самые лёгкие версии: меньше качать и меньше памяти, качество немного ниже"},
    "standard": {"title": "Стандарт", "lipsync_unet": "v2", "action_unet": "int8",
                 "about": "Облегчённая (pruned) Singularity v2: ~20 ГБ вместо 34, качество почти как у полной. Подходит для видеокарт 12–16 ГБ"},
    "max": {"title": "Максимум", "lipsync_unet": "full", "action_unet": "int8",
            "about": "Полные модели, как в исходных workflow"},
}


def recommended_profile(hw):
    vram, ram = hw.get("vram_gb") or 0, hw.get("ram_gb") or 0
    if vram + ram >= 48 and vram >= 24:  # "max" puts the full ~32 GB Singularity: too heavy for 16 GB cards
        return "max"
    if vram + ram >= 40:
        return "standard"
    return "lite"


def roles_for(modes, optional=True):
    roles = {r for m in modes for r in MODE_ROLES[m]} | {"preview"}
    if optional:
        roles |= {r for m in modes for r in MODE_OPTIONAL.get(m, [])}
    return [r for r in ALL_ROLES if r in roles]


def plan_size(choice, modes):
    return sum(VARIANTS[r][choice[r]]["size"] for r in roles_for(modes))


def choose_models(hw, profile=None):
    """Pick a variant per role for this computer. Returns (choice, notes, verdict, profile).

    ComfyUI keeps what does not fit in VRAM in system RAM, so the real budget is VRAM + RAM.
    """
    vram, ram = hw.get("vram_gb") or 0, hw.get("ram_gb") or 0
    blackwell = (hw.get("compute_cap") or 0) >= 12
    profile = profile if profile in PROFILES else recommended_profile(hw)
    choice = dict(DEFAULT_CHOICE, lipsync_unet=PROFILES[profile]["lipsync_unet"],
                  action_unet=PROFILES[profile]["action_unet"])
    notes = []

    if blackwell and vram >= 16:
        choice["fast_unet"] = "nvfp4"
        notes.append("Видеокарта RTX 50-й серии: для генерации видео берём версию NVFP4 — она легче и быстрее на этих картах.")

    if hw.get("nvidia") and 0 < (hw.get("driver") or 0) < MIN_DRIVER:
        notes.insert(0, f"Драйвер видеокарты устарел ({hw['driver']:g}). Нужен {MIN_DRIVER} или новее, иначе ComfyUI не запустится — обновите драйвер.")

    cap = hw.get("compute_cap") or 0
    if not hw.get("nvidia"):
        verdict = "unsupported"
        notes.insert(0, "Приложение работает только с видеокартами NVIDIA (RTX 20-й серии и новее, от 8 ГБ памяти). "
                        "Видеокарты AMD Radeon, встроенная графика Intel и компьютеры Mac не поддерживаются.")
    elif 0 < cap < 7.5:
        verdict = "unsupported"
        notes.insert(0, "Ваша видеокарта слишком старая для этой программы. Нужна NVIDIA RTX 20-й серии или новее "
                        "(например, RTX 2060, 3060, 4060) с памятью от 8 ГБ.")
    elif vram < 8:
        verdict = "unsupported"
        notes.insert(0, f"Видеопамяти {vram:g} ГБ — для работы нужно минимум 8 ГБ (лучше 16). "
                        "На такой видеокарте генерация видео не поместится в память.")
    elif vram < 12 or ram < 32:
        verdict = "weak"
        if vram < 12:
            notes.insert(0, f"Видеопамяти {vram:g} ГБ — генерация будет медленной, лучше выбирать 480p или 720p.")
        if ram < 32:
            notes.insert(0, f"Оперативной памяти {ram:g} ГБ, а рекомендуется 32 ГБ: модели могут не поместиться, "
                            "Windows начнёт использовать файл подкачки и всё сильно замедлится.")
    else:
        verdict = "ok"
    return choice, notes, verdict, profile


def model_entry(role, variant):
    return dict(VARIANTS[role][variant], role=role, variant=variant, title=ROLE_TITLES[role])


# modes that appeared after the first release: the app offers them once after an update
OFFER_MODES = ["flux", "long"]
