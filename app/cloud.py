"""Cloud generation on RunningHub (https://www.runninghub.ai) through its OpenAPI.

The app sends the same API-format graph it would send to a local ComfyUI, as the
`workflow` field of /task/openapi/create (a `workflowId` of any of the user's
RunningHub workflows is still required by the API). Input files go through
/task/openapi/upload; the finished video is downloaded into the local library.

The API key stays in the local config file and is never returned to the UI.
"""

import json

import aiohttp

HOSTS = {"ai": "https://www.runninghub.ai", "cn": "https://www.runninghub.cn"}
INSTANCE_TYPES = ("default", "plus", "ultra")  # 24 / 48 / 84 GB VRAM


MESSAGES = {
    "APIKEY_USER_NOT_FOUND": "Ключ не найден — проверьте, что скопировали его целиком",
    "APIKEY_INVALID_NODE_INFO": "RunningHub не принял параметры узлов",
    "APIKEY_TASK_NOT_FOUND": "Задание не найдено на RunningHub",
    "APIKEY_UNAUTHORIZED": "Ключ не даёт доступа к API — нужна подписка с API",
    "APIKEY_LIMIT_NOT_ENOUGH": "Закончился баланс или лимит на RunningHub",
    "NOT_ENOUGH_BALANCE": "На кошельке RunningHub (Wallet, в долларах) нет денег. Кредиты тарифа через API не списываются — пополните Wallet",
    "WORKFLOW_NOT_EXISTS": "Workflow с таким ID не найден на RunningHub",
    "TASK_QUEUE_MAXED": "Очередь RunningHub переполнена, попробуйте позже",
    "PERSONAL_QUEUE_COUNT_LIMIT": "Превышен лимит личной очереди RunningHub",
}


class CloudError(Exception):
    def __init__(self, msg):
        super().__init__(MESSAGES.get(msg, msg))


class RunningHub:
    def __init__(self, session, key, host="ai"):
        self.session = session
        self.key = key
        self.base = HOSTS.get(host, HOSTS["ai"])

    @property
    def headers(self):
        return {"Authorization": f"Bearer {self.key}"}

    async def post(self, path, body=None, data=None, timeout=60):
        kw = {"data": data} if data is not None else {"json": {"apiKey": self.key, **(body or {})}}
        try:
            async with self.session.post(self.base + path, headers=self.headers,
                                         timeout=aiohttp.ClientTimeout(total=timeout), **kw) as r:
                text = await r.text()
        except aiohttp.ClientError as e:
            raise CloudError(f"RunningHub недоступен: {e}")
        try:
            res = json.loads(text)
        except ValueError:
            raise CloudError(f"RunningHub ответил не JSON (HTTP {r.status}): {text[:200]}")
        return res

    async def account(self):
        res = await self.post("/uc/openapi/accountStatus", {"apikey": self.key})
        if res.get("code") != 0:
            raise CloudError(res.get("msg") or "Ключ не принят")
        return res.get("data") or {}

    async def upload(self, filename, content, content_type):
        form = aiohttp.FormData()
        form.add_field("apiKey", self.key)
        form.add_field("fileType", "input")
        form.add_field("file", content, filename=filename, content_type=content_type)
        res = await self.post("/task/openapi/upload", data=form, timeout=300)
        if res.get("code") != 0:
            raise CloudError(f"Не удалось загрузить {filename}: {res.get('msg')}")
        return res["data"]["fileName"]

    async def model_names(self, workflow_id):
        """File names the user's RunningHub workflow uses, to reuse their exact paths there."""
        res = await self.post("/api/openapi/getJsonApiFormat", {"workflowId": workflow_id})
        if res.get("code") != 0:
            raise CloudError(res.get("msg") or "Не удалось прочитать workflow")
        prompt = json.loads((res.get("data") or {}).get("prompt") or "{}")
        names = set()
        for node in prompt.values():
            for v in (node.get("inputs") or {}).values():
                if isinstance(v, str) and v.lower().endswith((".safetensors", ".gguf", ".ckpt", ".pt", ".pth")):
                    names.add(v)
        return names

    async def create(self, graph, workflow_id, instance_type="default"):
        body = {"workflowId": str(workflow_id), "workflow": json.dumps(graph, ensure_ascii=False),
                "addMetadata": False}
        if instance_type in INSTANCE_TYPES and instance_type != "default":
            body["instanceType"] = instance_type
        res = await self.post("/task/openapi/create", body)
        if res.get("code") != 0:
            raise CloudError(res.get("msg") or "RunningHub отклонил задание")
        data = res.get("data") or {}
        tips = data.get("promptTips")
        if tips:
            try:
                t = json.loads(tips)
                if t.get("node_errors"):
                    raise CloudError("Ошибка в графе на RunningHub: " + json.dumps(t["node_errors"], ensure_ascii=False)[:500])
            except ValueError:
                pass
        return str(data["taskId"])

    async def status(self, task_id):
        res = await self.post("/task/openapi/status", {"taskId": task_id})
        return res.get("data") if res.get("code") == 0 else None

    async def outputs(self, task_id):
        res = await self.post("/task/openapi/outputs", {"taskId": task_id})
        code = res.get("code")
        if code == 0:
            return res.get("data") or []
        if code in (804, 813):  # still running / queued
            return None
        reason = (res.get("data") or {}).get("failedReason") if isinstance(res.get("data"), dict) else None
        if isinstance(reason, dict):
            reason = f'{reason.get("node_name", "")}: {reason.get("exception_message", "")}'
        raise CloudError(reason or res.get("msg") or "Задание на RunningHub завершилось ошибкой")

    async def cancel(self, task_id):
        await self.post("/task/openapi/cancel", {"taskId": task_id})


def remap_models(graph, available):
    """Swap model file names for the paths the user's RunningHub workflow uses (matched by file name)."""
    by_base = {n.replace("\\", "/").split("/")[-1].lower(): n for n in available}
    for node in graph.values():
        for k, v in list(node.get("inputs", {}).items()):
            if isinstance(v, str) and v.lower().endswith((".safetensors", ".gguf")):
                hit = by_base.get(v.replace("\\", "/").split("/")[-1].lower())
                if hit:
                    node["inputs"][k] = hit
    return graph
