"""Основной движок: визуально-языковая модель (VLM), которая «читает» документ целиком.

Модель запускается локально — в LM Studio, Ollama, llama.cpp или vLLM. Все они
умеют OpenAI-совместимый API, поэтому клиент один. Документ никуда за пределы
компьютера не уходит: это важно, в ПТС и СТС персональные данные.

По умолчанию — Qwen3-VL 8B: открытая модель (Apache 2.0), читает 32 языка,
в сжатом виде занимает ~6 ГБ видеопамяти.
"""
from __future__ import annotations

import base64
import io
import json
import os
import re
import time

import requests
from PIL import Image

from ..normalize import clean_fields, detect_doc_type_by_fields
from ..schema import DOC_FIELDS, DOC_TITLES, DOC_TYPES, FIELDS
from .base import Engine, Extraction

DEFAULT_URL = os.environ.get("OKO_VLM_URL", "http://127.0.0.1:1234/v1")
DEFAULT_MODEL = os.environ.get("OKO_VLM_MODEL", "qwen3-vl-8b")


def build_prompt() -> str:
    doc_lines = "\n".join(f'- "{dt}" — {DOC_TITLES[dt]}: поля {", ".join(DOC_FIELDS[dt])}' for dt in DOC_TYPES)
    field_lines = "\n".join(f'- "{f.key}": {f.title}' + (f" ({f.hint})" if f.hint else "") for f in FIELDS)
    return f"""Ты — система извлечения данных из российских документов на автомобиль.
На картинке один документ. Определи его вид и перепиши значения полей.

Виды документов:
{doc_lines}

Поля:
{field_lines}

Правила:
1. Переписывай значения ровно так, как они напечатаны, заглавными буквами. Ничего не додумывай.
2. Если поля в документе нет или его не видно — пиши null.
3. Мощность в документе записана парой («78 / 106» или «106 (78)»): кВт клади в power_kw, л.с. — в power_hp.
4. ФИО собственника раздели на owner_last, owner_first, owner_middle.
5. Адрес раздели: region — область/край/республика, city — населённый пункт, street_address — улица, дом, квартира.
6. В госномере и номере ПТС буквы русские, в VIN — латинские.
7. Подписи полей, водяной знак «ОБРАЗЕЦ», печати и пометки на полях не переписывай.

Ответь только JSON без пояснений:
{{"doc_type": "<вид документа>", "fields": {{"<поле>": "<значение или null>", ...}}}}"""


def json_schema() -> dict:
    props = {f.key: {"type": ["string", "null"]} for f in FIELDS}
    return {
        "name": "vehicle_document",
        "strict": False,
        "schema": {
            "type": "object",
            "properties": {
                "doc_type": {"type": "string", "enum": DOC_TYPES},
                "fields": {"type": "object", "properties": props},
            },
            "required": ["doc_type", "fields"],
        },
    }


def parse_answer(text: str) -> dict | None:
    """Достаём JSON из ответа модели, даже если она обернула его в ```json ... ```."""
    if not text:
        return None
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)
    candidates = [m.group(1)] if m else []
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start:end + 1])
    for c in candidates:
        try:
            data = json.loads(c)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            continue
    return None


def image_to_data_url(img: Image.Image, max_side: int = 1600) -> str:
    img = img.convert("RGB")
    scale = max_side / max(img.size)
    if scale < 1:
        img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


class VLMEngine(Engine):
    def __init__(self, base_url: str = DEFAULT_URL, model: str = DEFAULT_MODEL, api_key: str = "local",
                 timeout: int = 240, max_side: int = 1600, use_schema: bool = True):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.max_side = max_side
        self.use_schema = use_schema
        self.prompt = build_prompt()
        self.name = f"vlm:{model}"

    def available(self) -> tuple[bool, str]:
        try:
            r = requests.get(f"{self.base_url}/models", timeout=4,
                             headers={"Authorization": f"Bearer {self.api_key}"})
            r.raise_for_status()
            ids = [m.get("id", "") for m in r.json().get("data", [])]
        except Exception:  # noqa: BLE001
            return False, (f"Сервер модели не отвечает по адресу {self.base_url}. "
                           "Запусти LM Studio (вкладка Developer → Start Server) или Ollama.")
        if ids and self.model not in ids:
            # LM Studio называет модель «qwen/qwen3-vl-8b», Ollama — «qwen3-vl:8b»: ищем похожее имя
            key = re.sub(r"[^a-z0-9]", "", self.model.lower())
            similar = [i for i in ids if key and key in re.sub(r"[^a-z0-9]", "", i.lower())]
            if len(similar) != 1:
                return False, f"На сервере нет модели «{self.model}». Есть: {', '.join(ids[:8])}"
            self.model = similar[0]
            self.name = f"vlm:{self.model}"
        return True, ""

    def _request(self, img: Image.Image, with_schema: bool) -> requests.Response:
        body = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 1200,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": self.prompt},
                    {"type": "image_url", "image_url": {"url": image_to_data_url(img, self.max_side)}},
                ],
            }],
        }
        if with_schema:
            body["response_format"] = {"type": "json_schema", "json_schema": json_schema()}
        return requests.post(f"{self.base_url}/chat/completions", json=body, timeout=self.timeout,
                             headers={"Authorization": f"Bearer {self.api_key}"})

    def extract(self, img: Image.Image) -> Extraction:
        t0 = time.time()
        try:
            r = self._request(img, self.use_schema)
            if r.status_code == 400 and self.use_schema:
                # не все серверы понимают JSON-схему — тогда просто просим JSON в тексте
                self.use_schema = False
                r = self._request(img, False)
            r.raise_for_status()
            text = r.json()["choices"][0]["message"]["content"] or ""
        except Exception as e:  # noqa: BLE001
            return Extraction(None, {}, time.time() - t0, self.name, error=f"{type(e).__name__}: {e}")
        data = parse_answer(text)
        if data is None:
            return Extraction(None, {}, time.time() - t0, self.name, raw=text, error="Ответ модели не JSON")
        raw_fields = data.get("fields") if isinstance(data.get("fields"), dict) else data
        fields = clean_fields({k: v for k, v in raw_fields.items() if k in {f.key for f in FIELDS}})
        doc_type = data.get("doc_type") if data.get("doc_type") in DOC_TYPES else detect_doc_type_by_fields(fields)
        return Extraction(doc_type, fields, time.time() - t0, self.name, raw=text)
