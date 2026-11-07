"""Agent mode: a local LLM (Gemma 4 via llama.cpp) that writes MiniMax H3 prompts using skills.

Skills are folders with SKILL.md (+ optional references/*.md). Built-in ones ship in
app/skills, the user's own are uploaded into data/skills. Each skill is "always" on,
"auto" (included when the chat mentions one of its keywords) or "off".
"""

import asyncio
import json
import re
import shutil
import subprocess
import zipfile
from pathlib import Path

import aiohttp

import engine

LLAMA_BUILD = "b11985"
LLAMA_URL = f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_BUILD}/llama-{LLAMA_BUILD}-bin-win-vulkan-x64.zip"

HF = "https://huggingface.co/{repo}/resolve/main/{path}"
LLM_VARIANTS = {
    "gemma4-12b": {"title": "Gemma 4 12B (Uncensored Heretic)", "repo": "llmfan46/gemma-4-12B-it-qat-q4_0-uncensored-heretic-GGUF",
                   "model": "gemma-4-12B-it-qat-q4_0-uncensored-heretic-Q4_0.gguf", "model_size": 7.60e9,
                   "mmproj": "gemma-4-12B-it-qat-q4_0-uncensored-heretic-mmproj-BF16.gguf", "mmproj_size": 0.18e9, "ctx": 49152},
    "gemma4-12b-orig": {"title": "Gemma 4 12B (Google Original)", "repo": "ggml-org/gemma-4-12B-it-GGUF",
                   "model": "gemma-4-12B-it-Q4_0.gguf", "model_size": 7.22e9,
                   "mmproj": "mmproj-gemma-4-12B-it-Q8_0.gguf", "mmproj_size": 0.16e9, "ctx": 49152},
    "gemma4-e4b": {"title": "Gemma 4 E4B (лёгкая)", "repo": "ggml-org/gemma-4-E4B-it-GGUF",
                   "model": "gemma-4-E4B-it-Q4_0.gguf", "model_size": 4.59e9,
                   "mmproj": "mmproj-gemma-4-E4B-it-Q8_0.gguf", "mmproj_size": 0.56e9, "ctx": 32768},
}

# Keywords that switch on the acting skill in "auto" mode (matched as word starts, any case).
DEFAULT_SKILL_SETTINGS = {
    "hailuo-h3": {"mode": "always", "keywords": []},
    # picture skills: used only when the chat is set to a picture target (see IMAGE_SKILLS), never for video prompts
    "karta-personazha": {"mode": "off", "keywords": []},
    "krea-2": {"mode": "off", "keywords": []},
    "flux2-klein": {"mode": "off", "keywords": []},
    "storyboard": {"mode": "off", "keywords": []},
    "qwen-image-2-1-director-prompt": {"mode": "off", "keywords": []},
    "akterskaya-igra": {"mode": "auto", "keywords": [
        "актер", "актёр", "диалог", "реплик", "эмоци", "игра", "драм", "ссор", "спор", "плач", "слез", "слёз",
        "прощан", "признан", "разговор", "говорит", "говорят", "кричит", "шепчет", "обид", "ревн", "страх",
        "гнев", "грусть", "печал", "радост", "микромимик", "dialog", "dialogue", "emotion", "acting", "actor"]},
}


# "Prompt for" targets that make pictures: which skill writes the prompt for each
IMAGE_SKILLS = {"plist": "krea-2", "krea": "krea-2", "fluxt": "flux2-klein", "qwent": "qwen-image-2-1-director-prompt",
                "qwen": "qwen-image-2-1-director-prompt", "sheet": "karta-personazha", "sheet6": "karta-personazha",
                "story": "storyboard"}
AGENT_TARGETS = ("video", "lipsync", "refs", "actor", "action", "long") + tuple(IMAGE_SKILLS)
# long extra reference files that would not fit next to the rest into the model's context
SKIP_REFS = {"platform-adapters.md", "long-form-writing.md"}


def pick_variant(hw):
    return "gemma4-12b" if (hw.get("vram_gb") or 0) >= 12 else "gemma4-e4b"


def parse_frontmatter(text):
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n", text, re.S)
    meta = {}
    if m:
        for line in m.group(1).splitlines():
            k, _, v = line.partition(":")
            if v:
                meta[k.strip()] = v.strip().strip("'\"")
        text = text[m.end():]
    return meta, text.strip()


class Skills:
    def __init__(self, builtin_dir, user_dir, config):
        self.builtin_dir, self.user_dir, self.config = Path(builtin_dir), Path(user_dir), config

    def settings(self, sid):
        saved = (self.config.get("skills") or {}).get(sid, {})
        return {**DEFAULT_SKILL_SETTINGS.get(sid, {"mode": "always", "keywords": []}), **saved}

    def list(self):
        out = []
        for base, builtin in ((self.builtin_dir, True), (self.user_dir, False)):
            if not base.exists():
                continue
            for d in sorted(p for p in base.iterdir() if (p / "SKILL.md").exists()):
                if not builtin and (self.builtin_dir / d.name).exists():
                    continue
                meta, _ = parse_frontmatter((d / "SKILL.md").read_text(encoding="utf-8", errors="replace"))
                refs = sorted((d / "references").glob("*.md")) if (d / "references").exists() else []
                out.append({"id": d.name, "name": meta.get("name", d.name), "description": meta.get("description", ""),
                            "builtin": builtin, "path": d, "refs": refs, **self.settings(d.name)})
        return out

    def save_settings(self, sid, mode=None, keywords=None):
        all_s = dict(self.config.get("skills") or {})
        cur = dict(all_s.get(sid, {}))
        if mode in ("always", "auto", "off"):
            cur["mode"] = mode
        if keywords is not None:
            cur["keywords"] = [k.strip().lower() for k in keywords if k.strip()]
        all_s[sid] = cur
        self.config.update(skills=all_s)

    def active(self, chat_text, target="video"):
        if target in IMAGE_SKILLS:  # a picture prompt: only its own skill, the video skills would mislead the model
            return [s for s in self.list() if s["id"] == IMAGE_SKILLS[target]]
        text = chat_text.lower()
        res = []
        for s in self.list():
            if s["id"] in IMAGE_SKILLS.values():
                continue
            if s["mode"] == "always":
                res.append(s)
            elif s["mode"] == "auto" and any(re.search(r"\b" + re.escape(k), text) for k in s["keywords"]):
                res.append(s)
        # always-on skills first: the prompt prefix stays the same, so the model reuses its cache
        return sorted(res, key=lambda s: s["mode"] != "always")

    def text(self, skill):
        _, body = parse_frontmatter((skill["path"] / "SKILL.md").read_text(encoding="utf-8", errors="replace"))
        parts = [f"# СКИЛЛ «{skill['name']}»\n\n{body}"]
        for ref in skill["refs"]:
            if ref.name in SKIP_REFS:  # only for platforms without their own skill / too long
                continue
            parts.append(f"## Справочник {skill['name']}: {ref.name}\n\n"
                         + ref.read_text(encoding="utf-8", errors="replace").strip())
        return "\n\n".join(parts)

    def save_upload(self, files):
        """files: list of (relative_path, bytes). Returns the new skill id."""
        skill_md = next((p for p, _ in files if Path(p).name.lower() == "skill.md"), None)
        if not skill_md:
            raise ValueError("В загрузке нет файла SKILL.md")
        root = Path(skill_md).parent
        meta, _ = parse_frontmatter(next(b for p, b in files if p == skill_md).decode("utf-8", errors="replace"))
        sid = re.sub(r"[^\w\-]+", "-", meta.get("name") or root.name or "skill").strip("-").lower() or "skill"
        dest = self.user_dir / sid
        if dest.exists():
            shutil.rmtree(dest)
        for p, data in files:
            rel = Path(p).relative_to(root) if root != Path(".") and Path(p).is_relative_to(root) else Path(Path(p).name)
            if rel.suffix.lower() not in (".md", ".txt"):
                continue
            target = (dest / rel).resolve()
            if dest.resolve() not in target.parents:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        return sid

    def delete(self, sid):
        d = self.user_dir / sid
        if d.exists() and (d / "SKILL.md").exists():
            shutil.rmtree(d)


def system_prompt(skills_text, target, ctx):
    """App rules first (they decide the output format), then the skills."""
    dur = ctx.get("duration")
    n_img = ctx.get("images", 0)
    if target == "lipsync":
        mode_rules = (
            "Режим приложения: LIPSYNC (Singularity Ref2V + Audio Lock). Видео делается по референс-картинкам и "
            "аудиофайлу пользователя, длина видео = длине аудиофрагмента"
            + (f" ({dur:.1f} с)" if dur else "") + ".\n"
            "- Первая строка промпта - выравнивание референсов: для каждой картинки по порядку "
            "`<Picture N> (from [Shot 1]) is fully referenced.` после фразы `For the target video, at 0.00 seconds "
            "into the target video,`. Затем пустая строка и три поля H3.\n"
            "- Реплики внутри <d>[Язык] ...</d> должны дословно совпадать со словами в аудио. Если в сообщении есть блок "
            "`[Аудио ... Слова из аудио ...]`, возьми слова из него и вставь ДОСЛОВНО (без перевода и правок), язык "
            "укажи тот, что назван в блоке. Если слов нет и пользователь не дал текст - спроси его, не выдумывай.\n"
            "- Звук берётся из аудио пользователя, поэтому `overall_soundscape: N/A` и `non_diegetic_music: N/A`, "
            "если пользователь не просит иначе.\n"
            "- Рот говорящего должен оставаться видимым, без больших поворотов головы.")
    elif target == "actor":
        voice = ctx.get("voice")
        variant = ("ВАРИАНТ 1 (с голос-референсом): в разделе загружен образец голоса, поэтому у говорящих строка ролей с `@audio1`, "
                   "привязка к голосу перед каждой репликой, `(S1)`/`(S2)`." if voice else
                   "ВАРИАНТ 2 (без голоса): образец голоса не загружен, поэтому строку про `@audio1` и (S1) не пиши; звук сцены "
                   "строй из дыхания, ткани, шагов, комнаты." if voice is not None else
                   "Если пользователь не сказал, загружен ли образец голоса, выбери вариант по тексту: есть реплики - вариант 1, "
                   "нет - вариант 2.")
        mode_rules = (
            "Режим приложения: АКТЁРСКАЯ ИГРА (ComfyUI, MiniMax H3 Reference to Video). Это режим ComfyUI из скилла "
            "«Актёрская игра»: примени его раздел «Режим ComfyUI (H3 Reference to Video)» строго как написано, он важнее общих "
            "правил формата. Первая картинка - стартовый кадр видео (`@image1 - keyframe, first frame of the target video`), "
            "модель продолжает сцену именно с неё. Голос берётся из загруженного образца только как тембр, а слова произносятся "
            "новые - те, что ты пишешь в <d>[Язык] ...</d>. "
            f"Длительность клипа {dur or 14} с (максимум 25) - таймкоды не выходят за неё.\n"
            f"- {variant}\n"
            "- Ответ - один английский блок prompt: строка ролей, строка выравнивания, `integrated_multimodal_description` "
            "с шотами и таймкодами, `overall_soundscape`, `non_diegetic_music: N/A`. Без русской версии.\n"
            "- Внешность персонажей не описывай - её закрывает картинка. Никаких слов про шёпот: речь должна звучать.\n"
            "- Если реплики не помещаются в длительность, одной строкой скажи, сколько секунд поставить в разделе.")
    elif target == "plist":
        mode_rules = (
            "Режим приложения: СПИСОК КАДРОВ (серия отдельных картинок, раздел «Картинка по тексту», режим «Список промптов»). "
            "Пользователь описывает историю или набор картинок, а ты выдаёшь промпт для КАЖДОЙ картинки. Используй скилл «krea-2» "
            "для написания каждого промпта.\n"
            "- Ответ - ОДИН блок prompt, внутри отдельные промпты, по одному на картинку, разделённые ПУСТОЙ СТРОКОЙ. Без номеров, "
            "заголовков и подписей: приложение само делит текст по пустым строкам.\n"
            "- Каждая картинка рисуется отдельно и ничего не знает о других, поэтому в КАЖДОМ промпте заново опиши героя, место и "
            "стиль одними и теми же словами (так серия получится единой), меняя только действие, ракурс и свет.\n"
            "- Сколько картинок: столько, сколько назвал пользователь, иначе 4-8; не больше 50.\n"
            "- Каждый промпт - один абзац, это картинка, не видео: без таймкодов, шотов и звука.")
    elif target == "krea":
        mode_rules = (
            "Режим приложения: КАРТИНКА ПО ТЕКСТУ (Krea 2, раздел «Картинка по тексту»). Нужен промпт для ОДНОЙ КАРТИНКИ, "
            "это не видео: без таймкодов, шотов, звука и формата H3. Используй скилл «krea-2» для самого промпта.\n"
            "- В приложении нужен только английский промпт одним блоком prompt. Русскую версию, слайдеры, creativity, "
            "референсы стиля и выбор варианта модели не выводи: в приложении их нет.\n"
            "- Одной строкой перед блоком подскажи формат кадра из списка 16:9, 9:16, 1:1, 4:3, 3:4, 3:2, 2:3 "
            "(он выбирается в разделе). Если нужны реалистичные люди или фото, добавь, что в разделе есть стиль «Реализм».\n"
            "- Картинки, приложенные к чату, это образцы настроения и композиции, их не пересказывай подробно.")
    elif target == "story" and ctx.get("frames"):
        mode_rules = (
            f"Режим приложения: РАСКАДРОВКА ПО КАДРАМ ({'Qwen Image 2.1' if ctx.get('model') == 'qwen' else 'FLUX.2 klein'}). Каждый кадр рисуется ОТДЕЛЬНОЙ картинкой, а лист склеивает "
            "программа. Используй скилл «storyboard», раздел «Режим по кадрам».\n"
            "- Ответ: одной строкой по-русски сколько кадров; затем ОДИН блок prompt, в котором каждый кадр на своей строке: "
            "`1: ...`, `2: ...` и так далее. Число кадров строго 2, 3, 4, 6, 8 или 9 (если пользователь назвал другое число, "
            "выбери ближайшее из этих и скажи об этом). Каждая строка это полный самостоятельный английский промпт ОДНОГО кадра: "
            "стиль, героя (одинаково во всех кадрах), действие, план, свет. Никаких рамок, листов, линий и текста на картинке.\n"
            + ("- Ссылки на картинки в тексте: " + ('пиши <image1>, <image2>, <image3> (Qwen)' if ctx.get("model") == "qwen" else 'пиши "image 1", "image 2", "image 3" без скобок (FLUX)') + ".\n")
            + ("- К чату ПРИЛОЖЕНЫ картинки (%d): они подгружаются как референсы. <image1> первая приложенная (герой или лист "
               "персонажа), <image2> вторая (локация), <image3> третья. В КАЖДОЙ строке пиши героя и ссылкой (\"the character from "
               "image 1\") и коротко словами (одежда, причёска, цвета), пиши, что кадр новая композиция, а не копия image 2, и "
               "начинай каждую строку с \"In the same art style as image 1, a full-bleed picture that fills the entire frame right to its edges with no white border, no paper margin and no vignette.\" Пиши \"one single\" перед героем, чтобы он не раздваивался." % n_img if n_img else
               "- Картинок нет: опиши героя коротко и ОДИНАКОВО в каждой строке, стиль один во всех; каждую строку начинай с \"a full-bleed picture that fills the entire frame right to its edges with no white border, no paper margin and no vignette\". Если есть карта персонажа "
               "или локация, одной строкой подскажи приложить их (скрепка).")
        )
    elif target == "story":
        model = {"krea": "Krea 2", "flux": "FLUX.2 klein", "qwen": "Qwen Image 2.1"}.get(ctx.get("model"), "Krea 2")
        mode_rules = (
            f"Режим приложения: РАСКАДРОВКА (одна картинка 16:9 из нескольких кадров, модель {model}). Используй скилл "
            f"«storyboard» и в разделе «Шаг 4» ОДИН шаблон: для {model}. Пользователь рассказывает историю по кадрам.\n"
            "- Ответ: одной строкой по-русски число кадров и сетка, затем один английский блок prompt. Без номеров, подписей, "
            "стрелок и любого текста на картинке, без слов про видео. Негатив не выводи (у Qwen он вписывается в приложении). "
            "Русскую версию не выводи.\n"
            "- Если пользователь не назвал число кадров и не перечислил кадры, возьми 6.\n"
            + ("- К чату ПРИЛОЖЕНЫ картинки (%d): они подгружаются в генерацию как референсы. Порядок задан пользователем: "
               "<image1> первая приложенная (обычно герой или лист персонажа), <image2> вторая (обычно локация), <image3> третья. "
               "Ссылайся на них в промпте как <image1>, <image2> (для FLUX.2 пиши \"image 1\", \"image 2\"), героя и место "
               "словами не описывай, пиши \"the character from <image1>\", \"the location from <image2>\", \"same art style as "
               "<image1>\". Это работает только для Qwen 2.1 и FLUX.2 klein: если выбрана Krea 2, одной строкой скажи переключить "
               "модель. Используй раздел скилла про референсы." % n_img
               if n_img and ctx.get("model") != "krea" else
               "- Картинки, приложенные к чату, это образцы стиля или героя: кратко опиши героя словами, в раскадровку по тексту "
               "они не подгружаются. Если нужна точность по картинкам, подскажи выбрать Qwen 2.1 или FLUX.2 klein." if n_img else
               "- Если у пользователя есть карта персонажа или локация, одной строкой подскажи приложить их (скрепка) и выбрать "
               "Qwen 2.1 или FLUX.2 klein: тогда герой и место возьмутся с картинок.")
        )
    elif target == "fluxt":
        mode_rules = (
            "Режим приложения: КАРТИНКА ПО ТЕКСТУ (FLUX.2 klein, раздел «Картинка по тексту», модель FLUX.2 klein). Нужен промпт "
            "для ОДНОЙ КАРТИНКИ, это не видео: без таймкодов, шотов и звука. Используй скилл «flux2-klein».\n"
            "- В приложении нужен только английский промпт одним блоком prompt. Негатива у модели нет, не выводи его. "
            "Русскую версию и настройки модели не выводи.\n"
            "- Одной строкой перед блоком подскажи формат кадра из списка 16:9, 9:16, 1:1, 4:3, 3:4, 3:2, 2:3 (выбирается в разделе).\n"
            "- Картинки, приложенные к чату, это образцы настроения и композиции, их не пересказывай подробно.")
    elif target == "qwent":
        mode_rules = (
            "Режим приложения: КАРТИНКА ПО ТЕКСТУ (Qwen Image 2.1, раздел «Картинка по тексту», модель Qwen 2.1). Нужен промпт "
            "для ОДНОЙ КАРТИНКИ без референса, это не видео. Используй скилл Qwen Image для формулировки (наблюдательный стиль).\n"
            "- В приложении нужен только английский промпт одним блоком prompt. Негативный промпт в приложении вписывается "
            "отдельным полем, поэтому НЕ выводи блок negative. Русскую версию и RGBA/мультиреференс не выводи.\n"
            "- Одной строкой перед блоком подскажи формат кадра из списка 16:9, 9:16, 1:1, 4:3, 3:4, 3:2, 2:3.\n"
            "- Картинки, приложенные к чату, это образцы настроения и композиции, их не пересказывай подробно.")
    elif target == "qwen":
        mode_rules = (
            "Режим приложения: ПРАВКА КАРТИНКИ (Qwen Image 2.1, раздел «Правка картинки»). Пользователь прикладывает к чату "
            "картинку (Picture 1), которую надо изменить, и пишет, что поменять. Используй скилл Qwen Image для формулировки.\n"
            "- Если картинки в чате нет, попроси приложить её (скрепка внизу или «Из Библиотеки»), не выдумывай.\n"
            "- Ответ: один английский блок prompt с инструкцией правки: что изменить и что обязательно сохранить "
            "(лицо, причёска, одежда, поза, композиция, свет). Одна правка за раз. Если приложено несколько картинок: <image1> "
            "первая (её меняем), остальные (<image2>, <image3>) это референсы (герой, предмет, одежда, место): ссылайся на них "
            "по номеру, например «put the character from <image2> into the scene of <image1>».\n"
            "- RGBA, мультиреференс, размеры и другие настройки не выводи: в приложении их нет. Без видео-терминов.")
    elif target == "sheet":
        mode_rules = (
            "Режим приложения: ЛИСТ ПЕРСОНАЖА по описанию (картинка Krea 2 в разделе «Картинка по тексту», формат 16:9). "
            "Используй скилл «Карта персонажа».\n"
            "- Ответ: один английский блок prompt на ОДНУ картинку-лист из 4 панелей, без видео-терминов. Русскую версию "
            "и промпты для других моделей не выводи.\n"
            "- Если к чату приложена картинка персонажа, опираясь на неё, опиши его внешность (DNA) подробно: лист рисуется "
            "с нуля по тексту. Если персонаж уже есть картинкой и нужен лист именно по ней, одной строкой подскажи "
            "раздел «Правка картинки» → «Лист персонажа»: там это делается по самой картинке.")
    elif target == "sheet6":
        mode_rules = (
            "Режим приложения: ЛИСТ ПЕРСОНАЖА из 6 ПАНЕЛЕЙ по описанию (картинка Krea 2 в разделе «Картинка по тексту», "
            "широкий формат 21:9). Используй скилл «Карта персонажа», раздел «Вариант из 6 панелей» (шаблон Krea 2).\n"
            "- Ответ: один английский блок prompt на ОДНУ картинку-лист из шести панелей в один ряд, без видео-терминов. "
            "Русскую версию и промпты для других моделей не выводи.\n"
            "- Если к чату приложена картинка персонажа, опираясь на неё, опиши его внешность (DNA) подробно: лист рисуется "
            "с нуля по тексту. Если лист нужен именно по готовой картинке, одной строкой подскажи раздел «Правка картинки» → "
            "«Лист персонажа» → 6 панелей: там это делается по самой картинке и точнее.")
    elif target == "long":
        lg = ctx.get("long") or {}
        sound = lg.get("sound")
        secs = lg.get("secs") or []
        snd = ("ЗВУК: пользователь загрузит свою песню или голос, видео подстроится под них. Слова песни в текст НЕ пиши, "
               "модель их слышит. Пиши только игру: пение, эмоции, движения и камеру в ритм. Если пользователь назвал части песни "
               "(куплет, припев, проигрыш), раздели сцены по ним, иначе делай сцены по 10 секунд."
               if sound == "own" else
               "ЗВУК: своего аудио нет, модель сама делает звук по тексту. Реплики героев пиши в <d>[Язык] ...</d> на языке "
               "пользователя, и одной короткой фразой в конце сцены напиши фоновый звук и музыку за кадром."
               if sound == "none" else
               "ЗВУК: если пользователь не сказал, с песней ролик или без, спроси одним вопросом. С песней слова песни в текст "
               "не пиши (модель их слышит), без песни реплики героев пиши в <d>[Язык] ...</d>.")
        mode_rules = (
            "Режим приложения: ДЛИННОЕ ВИДЕО (сцены идут одна за другой, Singularity Ref2V + Pika). Ролик состоит из сцен, "
            "длина сцены от 3 до 15 секунд (по умолчанию 10), сцен не больше 12"
            + (f"; сейчас в разделе {len(secs)} сцен: {', '.join(str(int(x)) for x in secs)} с" if secs else "")
            + ". Если пользователь не назвал число сцен, предложи его сам по описанию (обычно 2-6).\n"
            "ГЛАВНОЕ: каждая следующая сцена продолжает прошлую, но модель видит только последнюю секунду прошлой сцены и "
            "референс-картинки. Текст соседних сцен она НЕ видит. Поэтому:\n"
            "- в КАЖДОЙ сцене коротко заново назови героя и место, ссылаясь на референс-картинки как <Picture 1>, <Picture 2> и т.д. "
            f"(приложено картинок: {n_img}, можно до 5; пользователь сам решает, что на какой: персонаж, локация, предмет); "
            "внешность подробно не расписывай, её даёт картинка;\n"
            "- одно главное действие на сцену, не пять;\n"
            "- камера отдельной фразой (наезд, облёт, отъезд), свет и настроение одной фразой;\n"
            "- начало сцены продолжает конец прошлой: на стыке не меняй место, одежду и позу резко, смену места делай плавной;\n"
            f"- {snd}\n"
            "ФОРМАТ ОТВЕТА: ровно ОДИН блок prompt. Внутри сцены по порядку, каждая начинается отдельной строкой "
            "`SCENE N | S`, где N номер сцены, S длина в секундах (целое 3-15), а дальше один абзац обычного английского "
            "текста. БЕЗ полей `integrated_multimodal_description`, `[Shot 1]`, таймкодов и `overall_soundscape`: приложение "
            "добавит обёртку само. После последней сцены НИЧЕГО не пиши в блоке (никаких overall_soundscape, non_diegetic_music "
            "и других полей): звук описывай внутри текста сцены. Пример:\n"
            "SCENE 1 | 10\nA knight <Picture 1> stands in the ruined throne room <Picture 2>, slowly raising a glowing guitar. "
            "Camera pushes in from wide to medium shot. Red glow, drifting ash.\n\n"
            "SCENE 2 | 10\nThe knight <Picture 1> in the ruined throne room <Picture 2> starts playing fiercely, head tilted back. "
            "Camera circles him at chest height. Flickering ghostly light on the walls.")
    elif target in ("refs", "action"):
        mode_rules = (
            ("Режим приложения: ЭКШН-СЦЕНЫ (Hybrid FL2VA/Ref2VA, сэмплинг 8 шагов в низком разрешении + 4 в высоком): "
             "бой, погоня, бег, прыжки, стрельба. Пиши энергичное действие по нарастанию, по 1 ключевому движению на шот, "
             "быстрые, но мотивированные движения камеры (tracking, pull out, truck) с указанием скорости. "
             if target == "action" else
             "Режим приложения: РЕФЕРЕНСЫ (Singularity Ref2V, один проход, 4-8 шагов). ")
            + f"Видео строится по референсам Omni-Reference: картинки (персонажи, локация, предметы), может быть "
            f"референс-видео (движение/камера) и референс-аудио (голос/ритм). Длительность {dur or 6} с - таймкоды "
            "шотов не выходят за неё.\n"
            "- Перед тремя полями H3 перечисли роли референсов по правилам скилла: `@image1 - character reference (...)` "
            "и т.д., одна роль на файл; на картинки в тексте ссылайся как <Picture N>.\n"
            "- Пиши примерно одну реплику на 2,5 секунды, если в сцене есть речь.")
    else:
        if n_img >= 2:
            align = "есть первый и последний кадр - используй строку выравнивания FL2VA"
        elif n_img == 1:
            align = "есть стартовый кадр - используй строку выравнивания I2VA"
        else:
            align = "картинок нет - T2VA, без строки выравнивания"
        mode_rules = (f"Режим приложения: ГЕНЕРАЦИЯ ВИДЕО (FastH3: текст / первый кадр / первый и последний кадр). "
                      f"Сейчас {align}. Длительность клипа {dur or 5} с - таймкоды шотов не выходят за неё.")
    intro = ("Ты - агент-художник внутри приложения Minimax H3 Workspace. Пользователь описывает картинку или правку, может приложить "
             "картинки. Ты пишешь готовый промпт для модели картинок по правилам скилла ниже."
             if target in IMAGE_SKILLS else
             "Ты - агент-режиссёр внутри приложения Minimax H3 Workspace. Пользователь описывает сцену, сценарий, персонажей, "
             "может приложить картинки. Ты пишешь готовый промпт для локальной видеомодели MiniMax H3 по правилам скиллов ниже.")

    loras_block = ""
    if ctx.get("loras_text"):
        loras_block = f"\n{ctx['loras_text']}\n"
    elif ctx.get("loras"):
        try:
            import loras_registry
            loras_block = "\n" + loras_registry.format_loras_for_agent(ctx.get("loras"), ctx.get("loras_meta")) + "\n"
        except Exception:
            pass

    return f"""{intro}

ПРАВИЛА ПРИЛОЖЕНИЯ (важнее указаний скиллов о формате ответа):
1. Общайся на языке пользователя, коротко и по делу. Если не хватает критичного (что происходит, кто говорит, текст реплик) - задай 1-2 конкретных вопроса. Иначе сразу пиши промпт.
2. Итоговый промпт выдай РОВНО ОДНИМ блоком кода с меткой prompt:
```prompt
...промпт...
```
Промпт на английском, реплики внутри <d>[Язык] ...</d> - на языке оригинала. Не дублируй промпт вторым блоком на русском, если пользователь сам не попросил. Перед блоком - максимум 2-3 строки (допущения, если что-то придумал).
3. Промпт не длиннее 7000 символов. Без синтаксиса весов, без имён знаменитостей и чужих персонажей.
4. Картинки, приложенные к чату, - это Picture 1, Picture 2... в порядке прикрепления (сейчас приложено: {n_img}). Внешность с картинок подробно не расписывай - модель видит их сама; бери только позу, взгляд, предметы, свет.
5. Скиллы могут ссылаться на файлы references/... - все нужные справочники уже включены ниже, файлы открывать не нужно.

{mode_rules}
{loras_block}
{skills_text}
"""


def fits(model, vram_gb):
    """A model of this file size runs comfortably when it takes at most ~60% of video memory
    (the rest goes to the context, which the skills need to be large)."""
    return not vram_gb or Path(model).stat().st_size <= vram_gb * 1024 ** 3 * 0.6


def find_gguf(folder, prefer=None, vram_gb=None):
    """Pick a chat model and its vision projector from a folder the user already has.

    Returns (model, mmproj_or_None, all_models). Draft/MTP helper files are skipped.
    With no explicit choice, a Gemma that fits the user's video memory wins, never simply the biggest file.
    """
    folder = Path(folder)
    files = [p for p in folder.rglob("*.gguf") if len(p.relative_to(folder).parts) <= 3]
    mmproj = [p for p in files if "mmproj" in p.name.lower()]
    models = [p for p in files if p not in mmproj and not p.name.lower().startswith(("mtp-", "dflash-"))]
    if not models:
        return None, None, []
    models.sort(key=lambda p: p.name.lower())
    model = next((p for p in models if prefer and str(p) == str(prefer)), None)
    if model is None:
        ok = [p for p in models if fits(p, vram_gb)]
        pool = ok or models
        # among suitable files prefer Gemma, then the largest that still fits; if nothing fits, the smallest
        pool.sort(key=lambda p: ("gemma" not in p.name.lower(), -p.stat().st_size if ok else p.stat().st_size))
        model = pool[0]
    # a projector from the same folder belongs to the same model most of the time
    proj = sorted(mmproj, key=lambda p: (p.parent != model.parent, p.stat().st_size))
    return model, (proj[0] if proj else None), models


def explain_failure(log):
    """Turn a llama.cpp start-up log into one sentence a normal user can act on."""
    low = log.lower()
    if "outofdevicememory" in low or "failed to allocate" in low or "out of memory" in low:
        return ("Не хватило видеопамяти: эта модель слишком большая для вашей видеокарты. "
                "Выберите модель поменьше, например Gemma 4 12B в сжатии Q4 (около 7 ГБ), "
                "или вернитесь к встроенной Gemma кнопкой «Вернуть встроенную Gemma».")
    if "unknown model architecture" in low or "unsupported" in low:
        return "Эта модель не поддерживается движком llama.cpp. Выберите другую модель (лучше Gemma 4 в формате GGUF)."
    if "failed to load model" in low or "error loading model" in low:
        return "Не удалось открыть файл модели: возможно, он скачан не до конца или повреждён."
    return None


class LlamaServer:
    def __init__(self, root, log_path):
        self.root = Path(root)
        self.log_path = log_path
        self.proc = None
        self.port = None
        self.variant = None
        self.custom = None  # {"model": path, "mmproj": path|None} when the user points to their own GGUF

    def exe(self):
        hits = list((self.root / "llama.cpp").rglob("llama-server.exe")) if (self.root / "llama.cpp").exists() else []
        return hits[0] if hits else None

    def model_paths(self, variant):
        if self.custom:
            return Path(self.custom["model"]), Path(self.custom["mmproj"]) if self.custom.get("mmproj") else None
        v = LLM_VARIANTS[variant]
        m = self.root / "models" / v["model"]
        p = self.root / "models" / v["mmproj"]
        if not m.exists():
            for other_var, other_v in LLM_VARIANTS.items():
                cand_m = self.root / "models" / other_v["model"]
                if cand_m.exists():
                    m = cand_m
                    p = self.root / "models" / other_v["mmproj"]
                    break
        if p and not p.exists():
            for cand_p in [
                self.root / "models" / "gemma-4-12B-it-qat-q4_0-uncensored-heretic-mmproj-BF16.gguf",
                self.root / "models" / "mmproj-gemma-4-12B-it-Q8_0.gguf",
            ] + list((self.root / "models").glob("*mmproj*.gguf")):
                if cand_p.exists():
                    p = cand_p
                    break
            else:
                p = None
        return m, p

    def has_model(self, variant):
        m, p = self.model_paths(variant)
        return m.exists()

    def installed(self, variant):
        return bool(self.exe()) and self.has_model(variant)

    def vision(self, variant):
        return self.model_paths(variant)[1] is not None

    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    @property
    def url(self):
        return f"http://127.0.0.1:{self.port}"

    def start(self, variant):
        key = (variant, json.dumps(self.custom, sort_keys=True))
        if self.alive() and self.variant == key:
            return
        self.stop()
        m, p = self.model_paths(variant)
        self.port = engine.free_port(8280)
        self.variant = key
        # -ngl is left unset on purpose: llama.cpp then fits the layers to free video memory by itself
        # (heavier models run partly on the CPU, slower but they start); the context stays fixed
        # because the skills need about 20k tokens
        args = [str(self.exe()), "-m", str(m), "--host", "127.0.0.1", "--port", str(self.port),
                "-c", str(LLM_VARIANTS[variant]["ctx"]), "-fit", "on", "-fitt", "1536", "--jinja", "-np", "1"]
        if p:
            args += ["--mmproj", str(p)]
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        log = open(self.log_path, "w", encoding="utf-8", errors="replace")
        self.proc = subprocess.Popen(args, cwd=self.exe().parent, stdout=log, stderr=subprocess.STDOUT,
                                     creationflags=engine.CREATE_NO_WINDOW, close_fds=True)
        try:
            log.close()
        except Exception:
            pass

    def stop(self):
        if self.proc:
            try:
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(self.proc.pid)],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               creationflags=engine.CREATE_NO_WINDOW, timeout=2)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass
        self.proc = None

    def log_tail(self, n=12):
        try:
            return "\n".join(self.log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-n:])
        except OSError:
            return ""


async def install(installer, root, variant, runtime_only=False, whisper_dir=None):
    """Download llama.cpp (Vulkan build, works on any NVIDIA/AMD card) and, unless the user has their own, Gemma."""
    root = Path(root)
    llama_dir = root / "llama.cpp"
    if not (llama_dir.exists() and list(llama_dir.rglob("llama-server.exe"))):
        zpath = root / "downloads" / f"llama-{LLAMA_BUILD}.zip"
        await installer.download(LLAMA_URL, zpath, "agent", "llama.cpp", 37e6)
        with zipfile.ZipFile(zpath) as z:
            z.extractall(llama_dir)
        zpath.unlink(missing_ok=True)
    import speech
    await speech.install(installer, root, whisper_dir)  # words from audio + voice input (~200 MB)
    if runtime_only:
        return
    v = LLM_VARIANTS[variant]
    for key, size in (("mmproj", v["mmproj_size"]), ("model", v["model_size"])):
        dest = root / "models" / v[key]
        await installer.download(HF.format(repo=v["repo"], path=v[key]), dest, "agent", f'{v["title"]}: {v[key]}', size)


async def wait_ready(session, server, timeout=240):
    loop = asyncio.get_running_loop()
    end = loop.time() + timeout
    while loop.time() < end:
        if not server.alive():
            log = server.log_tail(80)
            raise RuntimeError(explain_failure(log) or "Модель не запустилась:\n" + server.log_tail(6))
        try:
            async with session.get(server.url + "/health", timeout=aiohttp.ClientTimeout(total=3)) as r:
                if r.status == 200:
                    return
        except Exception:
            pass
        await asyncio.sleep(1)
    raise RuntimeError("Модель долго загружается:\n" + server.log_tail())


def to_openai(messages):
    out = []
    for m in messages:
        if m.get("images") and m["role"] == "user":
            content = [{"type": "text", "text": m.get("content", "")}]
            content += [{"type": "image_url", "image_url": {"url": u}} for u in m["images"]]
            out.append({"role": "user", "content": content})
        else:
            out.append({"role": m["role"], "content": m.get("content", "")})
    return out


def chat_text(messages):
    return "\n".join(m.get("content", "") for m in messages if m["role"] == "user")


def dumps(obj):
    return json.dumps(obj, ensure_ascii=False)
