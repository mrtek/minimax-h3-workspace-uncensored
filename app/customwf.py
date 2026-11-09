"""The user's own ComfyUI workflow (API format): what to show in the simple form and what is missing.

Works only from the workflow itself and ComfyUI's /object_info, so it knows nothing about H3.
"""
import copy
import re

MEDIA = {  # (node class, input) -> kind of file the input takes
    ("LoadImage", "image"): "image", ("LoadImageMask", "image"): "image", ("LoadImageOutput", "image"): "image",
    ("LoadAudio", "audio"): "audio", ("VHS_LoadAudioUpload", "audio"): "audio",
    ("VHS_LoadVideo", "video"): "video", ("LoadVideo", "file"): "video",
}
OUTPUTS_VIDEO = {"SaveVideo", "VHS_VideoCombine", "SaveAnimatedWEBP", "SaveAnimatedPNG", "SaveWEBM"}
PROMPT_NAMES = {"text", "prompt", "positive_prompt", "negative_prompt", "text_g", "text_l", "caption"}
SIZE_NAMES = {"width", "height"}
LENGTH_NAMES = {"length", "frames", "num_frames", "frame_count", "video_length", "duration", "seconds", "frame_rate_length"}


def is_link(v):
    return isinstance(v, list) and len(v) == 2 and isinstance(v[0], (str, int)) and isinstance(v[1], int)


def validate_api(graph):
    if isinstance(graph, dict) and isinstance(graph.get("nodes"), list):
        raise ValueError("Это обычный файл workflow. Нужен формат API: в ComfyUI меню «Workflow» → «Export (API)».")
    if not isinstance(graph, dict) or not graph:
        raise ValueError("Файл не похож на workflow ComfyUI.")
    for node in graph.values():
        if not isinstance(node, dict) or not isinstance(node.get("class_type"), str) or not isinstance(node.get("inputs"), dict):
            raise ValueError("Файл не похож на workflow в формате API (меню «Workflow» → «Export (API)»).")


def spec(info, name):
    """Widget description of one input from /object_info, or None when the node does not list it."""
    for section in ("required", "optional"):
        entry = ((info.get("input") or {}).get(section) or {}).get(name)
        if not entry:
            continue
        typ = entry[0]
        opts = entry[1] if len(entry) > 1 and isinstance(entry[1], dict) else {}
        if isinstance(typ, list):
            return {"kind": "combo", "options": typ}
        if typ == "COMBO":
            return {"kind": "combo", "options": opts.get("options") or []}
        if typ in ("INT", "FLOAT"):
            return {"kind": typ.lower(), "min": opts.get("min"), "max": opts.get("max"), "step": opts.get("step")}
        if typ == "STRING":
            return {"kind": "string", "multiline": bool(opts.get("multiline"))}
        if typ == "BOOLEAN":
            return {"kind": "bool"}
        return {"kind": "link"}
    return None


def infer(value):
    if isinstance(value, bool):
        return {"kind": "bool"}
    if isinstance(value, int):
        return {"kind": "int"}
    if isinstance(value, float):
        return {"kind": "float"}
    return {"kind": "string", "multiline": isinstance(value, str) and len(value) > 60}


def basename(name):
    return str(name).replace("\\", "/").split("/")[-1]


RATIO = re.compile(r"\d{1,2}(\.\d+)?\s*[:/]\s*\d{1,2}(\.\d+)?|\d{3,4}\s*[x×]\s*\d{3,4}", re.I)
SYSTEM_TEXT = re.compile(r"system|rewrit|системн|переписыв|инструкц", re.I)
SIZE_WORDS = re.compile(r"megapix|^mp$|resolution|scale_by|upscale_by|^scale$|longest|shortest|target_size|max_size|side_length", re.I)


def role_of(cls, name, title, sp, negative):
    """What this field is for: prompt / negative / media / aspect / size / length / steps / cfg / seed / toggle / param / other."""
    if (cls, name) in MEDIA:
        return "media"
    label = name
    if cls.startswith("Primitive") and name == "value":  # primitives are named by their title
        label = re.sub(r"[^a-z]+", "_", title.lower()).strip("_") if re.search(r"[a-z]", title.lower()) else title.lower()
    low, t = label.lower(), title.lower()
    kind = sp["kind"]
    if kind == "string":
        texty = name in PROMPT_NAMES or "prompt" in low or cls.startswith("CLIPTextEncode") or "prompt" in t or "промпт" in t
        if texty and not SYSTEM_TEXT.search(name + " " + title):
            return "negative" if negative or "negative" in low or "negative" in t or "негатив" in t else "prompt"
        if cls.startswith("Primitive") and not SYSTEM_TEXT.search(title) and len(str(sp.get("value", ""))) < 80:
            return "param"
        return "other"
    if kind == "combo":
        opts = [str(o) for o in sp.get("options") or []]
        if sum(1 for o in opts if RATIO.search(o)) >= 2 or re.search(r"aspect|ratio|orientation", low):
            return "aspect"
        return "other"
    if kind == "bool":
        return "toggle" if cls.startswith("Primitive") else "other"
    if kind in ("int", "float"):
        if "seed" in low:
            return "seed"
        if low in SIZE_NAMES or low in ("w", "h") or SIZE_WORDS.search(low):
            return "size"
        if low in LENGTH_NAMES or re.fullmatch(r"(video_)?(length|frames|duration|seconds)", low) or "секунд" in t or "длин" in t:
            return "length"
        if low in ("steps", "step_count"):
            return "steps"
        if low in ("cfg", "guidance", "cfg_scale"):
            return "cfg"
        if cls.startswith("Primitive"):
            return "param"
    return "other"


def analyze(graph, infos):
    """infos: class name -> its /object_info entry (absent when ComfyUI does not know the class)."""
    negatives = {str(v[0]) for n in graph.values() for k, v in n["inputs"].items() if k.startswith("negative") and is_link(v)}
    widgets, missing_nodes, notes = [], [], []
    for nid, node in graph.items():
        cls = node["class_type"]
        title = str((node.get("_meta") or {}).get("title") or cls)
        info = infos.get(cls)
        if info is None:
            missing_nodes.append({"id": nid, "cls": cls, "title": title})
            continue
        if re.search(r"upscal|upsampl", cls, re.I):
            notes.append(f"Есть апскейл: «{title}». Он сработает как в ComfyUI, его модели проверены ниже.")
        for name, value in node["inputs"].items():
            if is_link(value):
                continue
            sp = spec(info, name) or infer(value)
            if sp["kind"] == "link":
                continue
            if sp["kind"] == "combo":  # the form compares and shows these as text; numbers and true/false come back through coerce()
                sp = dict(sp, options=[o if isinstance(o, str) else str(o) for o in sp["options"]])
                value = value if isinstance(value, str) else str(value)
            role = role_of(cls, name, title, sp, nid in negatives)
            w = {"key": f"{nid}.{name}", "node": nid, "title": title, "cls": cls, "name": name, "value": value,
                 "role": role, **sp}
            if role == "media":
                w["mkind"] = MEDIA[(cls, name)]
                w["has_default"] = value in (sp.get("options") or [])
            elif sp["kind"] == "combo" and value not in sp["options"]:
                w["missing"] = True
                same = [o for o in sp["options"] if basename(o) == basename(value)]
                w["suggest"] = same[0] if len(same) == 1 else ""
            if role == "media":
                w.pop("options", None)
            widgets.append(w)
    outputs = [n["class_type"] for n in graph.values() if n["class_type"] in OUTPUTS_VIDEO]
    if not outputs and not any(n["class_type"] == "SaveImage" for n in graph.values()):
        notes.append("В этом workflow нет узла сохранения (видео или картинки): результат не попадёт в Библиотеку.")
    return {"widgets": widgets, "missing_nodes": missing_nodes, "notes": notes}


def stage_map(graph):
    """node id -> progress stage used by the app's progress line."""
    depth = {}

    def dep(nid, seen=()):
        if nid in depth:
            return depth[nid]
        if nid in seen or nid not in graph:
            return 0
        d = 0
        for v in graph[nid]["inputs"].values():
            if is_link(v):
                d = max(d, dep(str(v[0]), seen + (nid,)) + 1)
        depth[nid] = d
        return d

    samplers = []
    stages = {}
    for nid, node in graph.items():
        c = node["class_type"]
        low = c.lower()
        if "sampler" in low and "select" not in low:
            samplers.append((dep(nid), nid))
        elif "decode" in low:
            stages[nid] = "decode"
        elif re.search(r"upscal|upsampl", low):
            stages[nid] = "upscale"
        elif c in OUTPUTS_VIDEO or low.startswith("save") or "videocombine" in low:
            stages[nid] = "save"
        elif (low.endswith("loader") or low.startswith("load")) and (c, "image") not in MEDIA and not c.startswith("LoadAudio"):
            stages[nid] = "load"
        else:
            stages[nid] = "encode"
    for i, (_, nid) in enumerate(sorted(samplers)):
        stages[nid] = "sample" if i == 0 else "sample2"
    return stages


def coerce(value, original):
    if isinstance(original, bool):
        return value in (True, "true", "True", "1", 1)
    if isinstance(original, int):
        return int(float(value))
    if isinstance(original, float):
        return float(value)
    return value if isinstance(value, str) else str(value)


def apply_values(graph, values, uploads):
    """A copy of the workflow with the form's values and uploaded file names put in."""
    g = copy.deepcopy(graph)
    for source in (values, uploads):
        for key, value in source.items():
            nid, _, name = key.partition(".")
            node = g.get(nid)
            if not node or name not in node["inputs"] or is_link(node["inputs"][name]):
                continue
            try:
                node["inputs"][name] = coerce(value, node["inputs"][name])
            except (TypeError, ValueError):
                pass
    return g
