"""Knowledge base and metadata extractor for MiniMax H3 LoRAs.

Provides human-readable names, categories, descriptions, and trigger words for known LoRAs,
reads safetensors metadata headers, and formats LoRA guidance for the prompt-writing Agent.
"""

import json
import os
import re
from pathlib import Path

# Built-in knowledge base for known MiniMax H3 LoRAs
KNOWN_LORAS = [
    {
        "pattern": r"dr34ml4y",
        "name": "DR34ML4Y NSFW Позы & Секс",
        "category": "NSFW & Позы",
        "desc": "Специализированная LoRA для взрослых поз и интимных сцен (миссионерская, догги, наездница, оральные ласки).",
        "triggers": ["m15510n4ry", "d0gg1e", "c0wg1rl", "bl0wj0b", "d0ubl3_bj"],
        "recommended_weight": 0.8,
    },
    {
        "pattern": r"unlocked.*v?2|nsfw_unlocked|minimax.*nsfw.*unlocked",
        "name": "NSFW Unlocked V2.1 (Снятие цензуры)",
        "category": "NSFW & Анатомия",
        "desc": "Полная разблокировка анатомии: детализация женской и мужской груди, сосков, половых органов и обнажённых тел.",
        "triggers": ["nude", "bare breasts", "perky nipples", "navel", "pussy", "naked", "topless"],
        "recommended_weight": 0.85,
    },
    {
        "pattern": r"mistermr.*latex|latex.*outfit",
        "name": "MisterMR Latex Outfit (Костюмы из латекса)",
        "category": "Одежда & Стили",
        "desc": "Облегающие комбинезоны, боди и платья из глянцевого латекса от MisterMR с глубокими отражениями света.",
        "triggers": ["latex", "latex catsuit", "shiny black latex", "latex outfit", "glossy latex"],
        "recommended_weight": 0.8,
    },
    {
        "pattern": r"latexclothes|latex.*minimax",
        "name": "Latex Clothes (Латексная одежда)",
        "category": "Одежда & Стили",
        "desc": "Одевает персонажей в блестящую облегающую одежду из латекса (боди, корсеты, перчатки, чулки).",
        "triggers": ["latex", "latex clothes", "shiny black latex", "latex bodysuit"],
        "recommended_weight": 0.8,
    },
    {
        "pattern": r"realism.*people|h3-realism",
        "name": "Реализм людей (Кожа и лица)",
        "category": "Качество & Реализм",
        "desc": "Улучшает естественную текстуру кожи, микрорельеф лица, убирает искусственный «пластиковый» вид нейросети.",
        "triggers": ["realism", "hyperrealistic skin texture", "natural human skin", "detailed face pores"],
        "recommended_weight": 0.6,
    },
    {
        "pattern": r"speed_slider|speedslider",
        "name": "Speed Slider (Слайдер скорости)",
        "category": "Управление движением",
        "desc": "Управление динамикой: положительный вес (+1..+3) делает движения быстрыми и резкими; отрицательный (-1..-2) замедляет до слоу-мо.",
        "triggers": [],
        "recommended_weight": 1.5,
    },
    {
        "pattern": r"motion_repair",
        "name": "Motion Repair (Исправление движения)",
        "category": "Исправление артефактов",
        "desc": "Устраняет дрожание, смазанные руки, артефакты «желе» и сглаживает искажения при динамичном движении.",
        "triggers": [],
        "recommended_weight": 0.5,
    },
    {
        "pattern": r"pika_h3_flex|pika.*turbo",
        "name": "Pika Flex Turbo HD (8 шагов)",
        "category": "Турбо-ускорители",
        "desc": "Высококачественный 8-шаговый HD турбо-ускоритель от Pika для динамичных экшн и референс-сцен.",
        "triggers": [],
        "recommended_weight": 1.0,
    },
    {
        "pattern": r"ref2v_turbo.*4step|turbo.*4step",
        "name": "Ref2V Turbo (4 шага)",
        "category": "Турбо-ускорители",
        "desc": "Сверхбыстрый 4-шаговый генератор для режима референсов (Ref2V), сокращает время генерации в 3-4 раза.",
        "triggers": [],
        "recommended_weight": 0.7,
    },
    {
        "pattern": r"pbs.*minimax|ltx_pbs",
        "name": "PBS (Физика и поверхности)",
        "category": "Качество & Физика",
        "desc": "Физически корректный рендеринг поверхностей: ткани, металла, влажной кожи, реалистичный свет и тени.",
        "triggers": ["pbs", "physically based shading", "detailed texture"],
        "recommended_weight": 0.7,
    },
    {
        "pattern": r"splits.*os|splits",
        "name": "Splits & Angle Shifts (Смена ракурсов / Шпагаты)",
        "category": "Движение & Ракурсы",
        "desc": "Плавные динамические смены ракурсов, переходы между планами и гибкие гимнастические движения (шпагаты).",
        "triggers": ["split", "full split", "dynamic camera angle", "camera shift"],
        "recommended_weight": 0.7,
    },
    {
        "pattern": r"hmnsfw|aio.*sex",
        "name": "HMNSFW - All-In-One Sex LoRA",
        "category": "NSFW & Позы",
        "desc": "Комплексная взрослая модель: реалистичные интимные движения, страстные позы, сцепка тел и натуральная кожа.",
        "triggers": ["hmnsfw", "hardcore", "intercourse", "penetration", "passionate sex", "detailed skin"],
        "recommended_weight": 0.8,
    },
    {
        "pattern": r"hmcumshot|cumshot",
        "name": "HMCumshot (Эякуляция и следы)",
        "category": "NSFW Эффекты",
        "desc": "Реалистичные визуальные эффекты семенной жидкости на теле, одежде или лице партнёра.",
        "triggers": ["cumshot", "facial", "cum on body", "semen", "ejaculation"],
        "recommended_weight": 0.8,
    },
    {
        "pattern": r"hmpussy|pussy",
        "name": "HMPussy (Детализация анатомии)",
        "category": "NSFW Анатомия",
        "desc": "Высокая детализация женской и мужской анатомии крупным планом.",
        "triggers": ["detailed pussy", "vagina", "spread legs", "pussy closeup"],
        "recommended_weight": 0.75,
    },
    {
        "pattern": r"breast.*play|breast.*jiggle",
        "name": "Breast Play & Jiggle (Физика груди)",
        "category": "NSFW Физика",
        "desc": "Реалистичная физика колыхания женской груди при движении, взаимодействие рук с телом.",
        "triggers": ["breast jiggle", "groping breasts", "breast bounce", "cleavage"],
        "recommended_weight": 0.8,
    },
    {
        "pattern": r"deepthroat|blowjob",
        "name": "Daring's Deepthroat (Оральные сцены)",
        "category": "NSFW Позы",
        "desc": "Оральный секс, глубокий минет, взаимодействие партнёров крупным планом.",
        "triggers": ["deepthroat", "blowjob", "oral sex", "giving head"],
        "recommended_weight": 0.8,
    },
    {
        "pattern": r"motion_booster|motion.*booster",
        "name": "H3 Motion Booster (Усилитель динамики)",
        "category": "Управление движением",
        "desc": "Усиливает амплитуду движений тел и камеры, убирает замирания и статичность сцены.",
        "triggers": ["motion boost", "fast motion", "dynamic movement"],
        "recommended_weight": 0.75,
    },
    {
        "pattern": r"cinematic.*v2|cinematic.*look",
        "name": "Cinematic Look & Lighting (Киношный свет)",
        "category": "Качество & Кинематограф",
        "desc": "Кинематографичная атмосфера, киношная глубина резкости, анаморфотные блики и благородная цветокоррекция.",
        "triggers": ["cinematic lighting", "film grain", "depth of field", "movie still", "anamorphic"],
        "recommended_weight": 0.7,
    },
    {
        "pattern": r"male.*pov",
        "name": "Male POV (Вид от первого лица мужчины)",
        "category": "Ракурсы & Камера",
        "desc": "Ракурс от первого лица (глазами мужчины) для реалистичных сцен контакта и интимного взаимодействия.",
        "triggers": ["pov", "male pov", "first-person view", "looking down at partner"],
        "recommended_weight": 0.8,
    },
    {
        "pattern": r"mystic.*xxx|mysticxxx",
        "name": "Mystic XXX (NSFW Режиссура)",
        "category": "NSFW & Позы",
        "desc": "Высокодетализированная взрослая режиссура и движения для откровенных роликов.",
        "triggers": ["mysticxxx", "explicit sex", "passionate intimacy"],
        "recommended_weight": 0.8,
    },
]


def read_safetensors_metadata(filepath):
    """Safely extracts JSON metadata from the first bytes of a .safetensors file."""
    try:
        path = Path(filepath)
        if not path.is_file():
            return {}
        with open(path, "rb") as f:
            header_size_bytes = f.read(8)
            if len(header_size_bytes) < 8:
                return {}
            header_size = int.from_bytes(header_size_bytes, "little")
            # Sanity check: header shouldn't be gigabytes (usually a few KB to 5MB)
            if header_size <= 0 or header_size > 20 * 1024 * 1024:
                return {}
            header_raw = f.read(header_size)
            header_json = header_raw.decode("utf-8", errors="ignore")
            data = json.loads(header_json)
            meta = data.get("__metadata__", {})
            return meta if isinstance(meta, dict) else {}
    except Exception:
        return {}


def format_clean_name(raw_name):
    """Turns a raw filename like `my_lora_name_v1.safetensors` into a human title."""
    s = Path(raw_name).stem
    s = re.sub(r"[_\-]+", " ", s)
    s = re.sub(r"\b(v\d+(\.\d+)?|safetensors|minimax|h3)\b", "", s, flags=re.I)
    s = " ".join(s.split()).strip()
    return s.title() if s else Path(raw_name).stem


def get_lora_metadata(file_path_or_name, full_file_path=None, custom_meta=None):
    """Returns a rich metadata dictionary for any LoRA file.
    
    Checks custom user overrides first, then the built-in knowledge base,
    then parses the .safetensors header, and falls back to a clean formatted name.
    """
    clean_name = Path(file_path_or_name).name
    lower_name = clean_name.lower()
    
    # 1. Custom user overrides from config
    overrides = (custom_meta or {}).get(file_path_or_name) or (custom_meta or {}).get(clean_name) or {}
    
    # 2. Match against built-in knowledge base
    matched = None
    for entry in KNOWN_LORAS:
        if re.search(entry["pattern"], lower_name):
            matched = entry
            break
            
    # 3. If file exists, try reading header for trained words if not matched or triggers empty
    embedded_triggers = []
    embedded_desc = ""
    if full_file_path and Path(full_file_path).is_file():
        meta = read_safetensors_metadata(full_file_path)
        if meta:
            # Check civitai trainedWords
            tw = meta.get("modelspec.trained_words") or meta.get("trainedWords") or meta.get("ss_trained_words")
            if isinstance(tw, str):
                embedded_triggers = [w.strip() for w in tw.split(",") if w.strip()]
            elif isinstance(tw, list):
                embedded_triggers = [str(w).strip() for w in tw if str(w).strip()]
            
            # Check ss_tag_frequency
            if not embedded_triggers and meta.get("ss_tag_frequency"):
                try:
                    freq = json.loads(meta["ss_tag_frequency"]) if isinstance(meta["ss_tag_frequency"], str) else meta["ss_tag_frequency"]
                    for _, tags in freq.items():
                        if isinstance(tags, dict):
                            top_tags = sorted(tags.keys(), key=lambda k: tags[k], reverse=True)[:10]
                            embedded_triggers.extend(top_tags)
                except Exception:
                    pass

            embedded_desc = meta.get("modelspec.description") or meta.get("description") or ""

    # Synthesize metadata
    title = overrides.get("name") or (matched["name"] if matched else format_clean_name(clean_name))
    category = overrides.get("category") or (matched["category"] if matched else "Пользовательская LoRA")
    desc = overrides.get("desc") or (matched["desc"] if matched else (embedded_desc[:250] if embedded_desc else "Пользовательская LoRA для MiniMax H3."))
    
    triggers = overrides.get("triggers")
    if triggers is None:
        triggers = matched["triggers"] if matched else embedded_triggers
    if isinstance(triggers, str):
        triggers = [t.strip() for t in triggers.split(",") if t.strip()]

    recommended_weight = overrides.get("recommended_weight") or (matched["recommended_weight"] if matched else 0.8)

    return {
        "file": file_path_or_name,
        "filename": clean_name,
        "name": title,
        "category": category,
        "desc": desc,
        "triggers": triggers,
        "recommended_weight": recommended_weight,
    }


def format_loras_for_agent(active_loras, custom_meta=None):
    """Formats active LoRAs into an authoritative instruction block for the Agent."""
    if not active_loras:
        return ""

    lines = []
    has_triggers = False
    for item in active_loras:
        f = item.get("file") if isinstance(item, dict) else str(item)
        if not f:
            continue
        strength = float(item.get("strength", 0.8)) if isinstance(item, dict) else 0.8
        info = get_lora_metadata(f, custom_meta=custom_meta)
        trig = info.get("triggers") or []
        if trig:
            has_triggers = True
            trig_str = f"Ключевые триггерные слова: {', '.join(trig)}"
        else:
            trig_str = "Триггерные слова не требуются (управляет динамикой, стилем или физикой)"

        lines.append(
            f"- LoRA «{info['name']}» (файл: {info['filename']}, сила: {strength:.2f}):\n"
            f"  Назначение: {info['desc']}\n"
            f"  {trig_str}"
        )

    if not lines:
        return ""

    rule = (
        "ИНСТРУКЦИЯ ПО ИСПОЛЬЗОВАНИЮ LORA ДЛЯ АГЕНТА:\n"
        "1. В видеогенерации АКТИВНЫ указанные выше LoRA. Если тема сцены, персонаж, одежда, ракурс или действия "
        "пользователя совпадают с назначением LoRA — ОБЯЗАТЕЛЬНО используй её точные триггерные слова в английском описании "
        "integrated_multimodal_description.\n"
        "2. Вплетай триггеры органично (например: 'in d0gg1e position on the bed', 'wearing a shiny black latex bodysuit', 'hyperrealistic skin texture').\n"
        "3. Не используй триггеры не к месту, если сцена не имеет к ним отношения. Но если пользователь просит соответствующую сцену — активация триггеров обязательна!"
    )

    return (
        "ПОДКЛЮЧЕННЫЕ И АКТИВНЫЕ LORA В ПРОЕКТЕ:\n"
        + "\n".join(lines)
        + "\n\n"
        + rule
        + "\n"
    )
