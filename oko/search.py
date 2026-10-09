"""Поиск по базе запросом на обычном языке.

«белые киа 2018–2020 года из Саратова мощнее 120 л.с.» превращается в набор
условий: цвет = БЕЛЫЙ, марка ~ KIA, год между 2018 и 2020, регион или город ~ САРАТОВ,
мощность > 120. Условия собираются в SQL с параметрами — текст запроса никогда
не подставляется в SQL напрямую.

Два способа разбора:
- правила (`parse_rules`) — работают всегда, без нейросети;
- модель (`parse_llm`) — та же локальная модель переводит запрос в условия;
  её ответ проверяется: допускаются только известные колонки и операции.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

import requests

from .db import VEHICLE_COLUMNS, Store
from .normalize import canon

OPS = {"=": "=", "<": "<", ">": ">", "<=": "<=", ">=": ">=", "like": "LIKE", "between": "BETWEEN", "len": "LEN"}

BRANDS = {
    "киа": "KIA", "кия": "KIA", "хендай": "HYUNDAI", "хендэ": "HYUNDAI", "хундай": "HYUNDAI", "хёндай": "HYUNDAI",
    "хюндай": "HYUNDAI", "тойота": "TOYOTA", "тоёта": "TOYOTA", "лада": "LADA", "ваз": "LADA",
    "фольксваген": "VOLKSWAGEN", "фольц": "VOLKSWAGEN", "шкода": "SKODA", "рено": "RENAULT", "ниссан": "NISSAN",
    "чери": "CHERY", "хавейл": "HAVAL", "хавал": "HAVAL", "хавэйл": "HAVAL", "джили": "GEELY", "бмв": "BMW",
    "мерседес": "MERCEDES-BENZ", "мерс": "MERCEDES-BENZ", "митсубиси": "MITSUBISHI", "мицубиси": "MITSUBISHI",
    "форд": "FORD", "солярис": "SOLARIS", "рио": "RIO", "веста": "VESTA", "гранта": "GRANTA", "ларгус": "LARGUS",
    "камри": "CAMRY", "рав4": "RAV4", "поло": "POLO", "октавия": "OCTAVIA", "логан": "LOGAN", "дастер": "DUSTER",
    "кашкай": "QASHQAI", "тигго": "TIGGO", "джолион": "JOLION", "кулрей": "COOLRAY", "аутлендер": "OUTLANDER",
    "фокус": "FOCUS", "крета": "CRETA",
}

COLORS = [  # порядок важен: «темно-сер» раньше «сер», «серебр» раньше «сер»
    ("темно-сер", "ТЕМНО-СЕРЫЙ"), ("тёмно-сер", "ТЕМНО-СЕРЫЙ"), ("серебр", "СЕРЕБРИСТЫЙ"), ("сер", "СЕРЫЙ"),
    ("бел", "БЕЛЫЙ"), ("черн", "ЧЕРНЫЙ"), ("чёрн", "ЧЕРНЫЙ"), ("син", "СИНИЙ"), ("красн", "КРАСНЫЙ"),
    ("зелен", "ЗЕЛЕНЫЙ"), ("зелён", "ЗЕЛЕНЫЙ"), ("корич", "КОРИЧНЕВЫЙ"), ("беж", "БЕЖЕВЫЙ"),
    ("оранж", "ОРАНЖЕВЫЙ"), ("голуб", "ГОЛУБОЙ"),
]

TITLES = {"year": "год", "power_hp": "мощность, л.с.", "engine_volume": "объём, см³", "make_model": "марка/модель",
          "color": "цвет", "region": "регион", "city": "город", "region|city": "регион или город", "owner_last": "фамилия владельца", "plate": "госномер",
          "vin": "VIN", "engine_type": "двигатель", "eco_class": "экокласс", "category": "категория",
          "pts_number": "ПТС"}


@dataclass
class Cond:
    column: str        # колонка или «region|city» (любая из двух)
    op: str
    value: object

    def human(self) -> str:
        t = TITLES.get(self.column, self.column)
        if self.op == "between":
            return f"{t}: {self.value[0]}–{self.value[1]}"
        if self.op == "len":
            return "электронный ПТС" if self.value == 15 else "бумажный ПТС"
        if self.op == "like":
            return f"{t} ~ {self.value}"
        sign = {"=": ":", ">": " >", "<": " <", ">=": " ≥", "<=": " ≤"}[self.op]
        return f"{t}{sign} {self.value}"


def _num(s: str) -> int:
    return int(s.replace(" ", ""))


def _take(q: str, pattern: str):
    """Ищет в запросе все куски, подходящие под шаблон. Возвращает найденное и запрос без этих кусков."""
    found = list(re.finditer(pattern, q))
    q = re.sub(pattern, " ", q)
    return found, q


def _find_color(w: str):
    """Цвет по началу слова: «белые», «белую» → БЕЛЫЙ."""
    for stem, val in COLORS:
        if w.startswith(stem) and len(w) <= len(stem) + 6:
            return val
    return None


def _find_brand(w: str):
    """Марка или модель по русскому названию: «тойоты», «солярисы» — тоже подходят."""
    for k, v in BRANDS.items():
        if w == k:
            return v
        if len(k) >= 4 and w.startswith(k[:-1]) and len(w) <= len(k) + 2:
            return v
    return None


def parse_rules(text: str, store: Store | None = None) -> tuple[list[Cond], list[str]]:
    """Разбор запроса правилами. Возвращает условия и слова, которые не удалось понять."""
    q = text.lower().replace("ё", "е")
    q = re.sub(r"[«»\"!?]", " ", q)
    conds: list[Cond] = []
    # Каждое правило: найти кусок запроса по шаблону, сделать из него условие, вырезать кусок из запроса.
    # Порядок важен: сначала длинные формулировки («с 2015 по 2018»), потом короткие («2015»).

    yr = r"((?:19|20)\d{2})"

    # годы диапазоном: «с 2015 по 2018», «2015-2018»
    range_rules = [
        rf"\b(?:с|от)\s+{yr}\s*(?:года?|г\.?)?\s*(?:по|до|-|–)\s*{yr}\s*(?:годов|года?|г\.?)?",
        rf"{yr}\s*[-–]\s*{yr}\s*(?:годов|года?|г\.?)?",
    ]
    for pattern in range_rules:
        found, q = _take(q, pattern)
        for m in found:
            conds.append(Cond("year", "between", (int(m.group(1)), int(m.group(2)))))

    # один год: (шаблон, операция)
    year_rules = [
        (rf"\bне\s+старше\s+{yr}\s*(?:года?|г\.?)?", ">="),
        (rf"\b(?:новее|после|позже|моложе)\s+{yr}\s*(?:года?|г\.?)?", ">"),
        (rf"\b(?:старше|до|раньше)\s+{yr}\s*(?:года?|г\.?)?", "<"),
        (rf"\b(?:с|от)\s+{yr}\s*(?:года?|г\.?)?", ">="),
        (rf"(?<!\d){yr}\s*(?:года?|г\.?|выпуска)?(?!\d)", "="),
    ]
    for pattern, op in year_rules:
        found, q = _take(q, pattern)
        for m in found:
            conds.append(Cond("year", op, int(m.group(1))))

    # мощность: (шаблон, операция)
    hp = r"(\d{2,3})\s*(?:л\.?\s*с\.?|лс|лошад\w*|сил\w*)"
    power_rules = [
        (rf"\b(?:мощнее|больше|более|свыше|от)\s+{hp}", ">"),
        (rf"\b(?:слабее|меньше|менее|до)\s+{hp}", "<"),
        (rf"{hp}", "="),
    ]
    for pattern, op in power_rules:
        found, q = _take(q, pattern)
        for m in found:
            conds.append(Cond("power_hp", op, _num(m.group(1))))

    # объём в литрах: «1.6» → от 1540 до 1660 см³
    found, q = _take(q, r"(\d[.,]\d)\s*(?:л\b|литр\w*|объем\w*)?")
    for m in found:
        liters = float(m.group(1).replace(",", "."))
        conds.append(Cond("engine_volume", "between", (int(liters * 1000 - 60), int(liters * 1000 + 60))))

    # слова с готовым значением: (шаблон, колонка, операция, значение)
    word_rules = [
        (r"дизел\w*", "engine_type", "like", "ДИЗЕЛ"),
        (r"бензин\w*", "engine_type", "like", "БЕНЗИН"),
        (r"(?:\bс\s+)?\b(?:эптс|электронн\w+\s+(?:птс|паспорт\w*))", "pts_number", "len", 15),
        (r"(?:\bс\s+)?\bбумажн\w+\s+(?:птс|паспорт\w*)", "pts_number", "len", 10),
        (r"(?:евро|экокласс\w*|экологическ\w+\s+класс\w*)[\s-]*(5|пят\w*)", "eco_class", "=", "ПЯТЫЙ"),
        (r"(?:евро|экокласс\w*|экологическ\w+\s+класс\w*)[\s-]*(4|четверт\w*)", "eco_class", "=", "ЧЕТВЕРТЫЙ"),
    ]
    for pattern, column, op, value in word_rules:
        found, q = _take(q, pattern)
        for m in found:
            conds.append(Cond(column, op, value))

    # госномер целиком или кусок: «а123вс», «номер 123»
    found, q = _take(q, r"\b[авекмнорстух]\d{3}[авекмнорстух]{2}\d{2,3}\b")
    for m in found:
        conds.append(Cond("plate", "=", canon("plate", m.group(0))))
    found, q = _take(q, r"(?:номер|госномер|гос\.?\s*номер)\s+([авекмнорстух]?\d{3}[авекмнорстух]{0,2})")
    for m in found:
        conds.append(Cond("plate", "like", canon("plate", m.group(1))))

    # VIN целиком (17 знаков) или кусок после слова «vin»
    found, q = _take(q, r"\b(?=[a-z0-9]*\d)(?=[a-z0-9]*[a-z])[a-z0-9]{17}\b")
    for m in found:
        conds.append(Cond("vin", "=", canon("vin", m.group(0))))
    found, q = _take(q, r"\bvin\s+([a-hj-npr-z0-9]{4,16})\b")
    for m in found:
        conds.append(Cond("vin", "like", canon("vin", m.group(1))))

    words = re.findall(r"[a-zа-я0-9][a-zа-я0-9\-]*", q)
    rest: list[str] = []
    makes = [m.upper() for m in (store.distinct("make_model") if store else [])]
    regions = store.distinct("region") if store else []
    cities = store.distinct("city") if store else []
    owners = set(store.distinct("owner_last")) if store else set()
    skip = {"все", "всех", "покажи", "найди", "машины", "машин", "авто", "автомобили", "автомобиль", "года", "год",
            "из", "в", "на", "с", "и", "у", "по", "от", "до", "области", "обл", "город", "края", "владелец",
            "владельца", "собственник", "собственника", "цвета", "цвет", "какие", "есть", "мне", "лс", "кто",
            "категории", "категория", "тачки", "тачку", "ли", "область", "край", "республика", "респ",
            "или", "а", "также", "всё", "которые", "который", "выпуска", "г", "гг", "мне", "нужны", "нужна"}
    for w in words:
        if w in skip:
            continue
        # цвет проверяем раньше марки: «черные» — это цвет, а не CHERY
        color = _find_color(w)
        if color and not (color == "СЕРЫЙ" and w.startswith("серг")):
            conds.append(Cond("color", "=", color))
            continue
        brand = _find_brand(w)
        if brand:                                   # «тойоты», «солярисы» — тоже марка
            conds.append(Cond("make_model", "like", brand))
            continue
        if w.upper() in {m.split()[0] for m in makes} or any(w.upper() in m.split() for m in makes):
            conds.append(Cond("make_model", "like", w.upper()))
            continue
        if w in ("b", "в") and "категор" in text.lower():
            conds.append(Cond("category", "=", "B"))
            continue
        up = w.upper()
        # фамилия в любом падеже: «Морозовой» → МОРОЗОВА, «Морозову» → МОРОЗОВ, «Иванова» → ИВАНОВ или ИВАНОВА
        cand = [up]
        if len(up) > 4:
            if up.endswith(("ОЙ", "ЕЙ")):
                cand.append(up[:-2] + "А")
            elif up.endswith(("ЫМ", "ОМ", "ЫХ")):
                cand.append(up[:-2])
            elif up.endswith(("А", "У", "Е")):
                cand.append(up[:-1])
        found = [c for c in dict.fromkeys(cand) if c in owners]
        if found:
            conds += [Cond("owner_last", "=", c) for c in found]
            continue
        if len(w) >= 4:
            stem = up[: max(4, len(up) - 2)]
            if any(stem in c for c in cities):            # сначала город: «из Саратова» — это город
                conds.append(Cond("city", "like", stem))
                continue
            if any(stem in r for r in regions):
                conds.append(Cond("region", "like", stem))
                continue
        rest.append(w)
    return _merge_brand_model(conds), rest


def _merge_brand_model(conds: list[Cond]) -> list[Cond]:
    """«хендай крета» — это одна модель, а не «любой хендай или любая крета»: марку убираем, модель остаётся."""
    from .fakedata import CATALOG
    full = [m.name for m in CATALOG]
    likes = {c.value for c in conds if c.column == "make_model" and c.op == "like"}
    implied = set()
    for v in likes:
        for b in likes - {v}:
            if any(f.startswith(b + " ") and v in f.split()[1:] for f in full):
                implied.add(b)
    return [c for c in conds if not (c.column == "make_model" and c.op == "like" and c.value in implied)]


def to_sql(conds: list[Cond]) -> tuple[str, list]:
    parts, params = [], []
    for c in conds:
        cols = c.column.split("|")
        if any(col not in VEHICLE_COLUMNS for col in cols) or c.op not in OPS:
            continue
        sub = []
        for col in cols:
            if c.op == "between":
                sub.append(f"{col} BETWEEN ? AND ?")
                params += list(c.value)
            elif c.op == "like":
                sub.append(f"UPPER({col}) LIKE ?")
                params.append(f"%{str(c.value).upper()}%")
            elif c.op == "len":
                sub.append(f"LENGTH({col}) = ?")
                params.append(int(c.value))
            else:
                sub.append(f"{col} {OPS[c.op]} ?")
                params.append(c.value)
        parts.append("(" + " OR ".join(sub) + ")")
    # одинаковые колонки с «=» (два цвета, две марки) объединяем через ИЛИ — так ожидает человек
    return (" AND ".join(parts) if parts else "1=1"), params


def search(store: Store, text: str, conds: list[Cond] | None = None, limit: int = 200):
    """Ищет автомобили. Возвращает (условия, непонятые слова, найденные карточки)."""
    rest: list[str] = []
    if conds is None:
        conds, rest = parse_rules(text, store)
    # одинаковые условия «колонка = значение» с разными значениями — это «или»
    merged: dict[tuple[str, str], list[Cond]] = {}
    for c in conds:
        merged.setdefault((c.column, c.op), []).append(c)
    where_parts, params = [], []
    for (col, op), group in merged.items():
        if op in ("=", "like") and len(group) > 1:
            sqls = [to_sql([g]) for g in group]
            where_parts.append("(" + " OR ".join(s for s, _ in sqls) + ")")
            for _, p in sqls:
                params += p
        else:
            s, p = to_sql(group)
            where_parts.append(s)
            params += p
    if rest and not conds:
        # ничего не поняли — ищем слова в любых текстовых колонках
        text_cols = [c for c, t in VEHICLE_COLUMNS.items() if t == "TEXT"]
        for w in rest:
            where_parts.append("(" + " OR ".join(f"UPPER({c}) LIKE ?" for c in text_cols) + ")")
            params += [f"%{w.upper()}%"] * len(text_cols)
    where = " AND ".join(where_parts) if where_parts else "1=1"
    return conds, rest, store.query(where, params, limit)


# ------------------------------------------------------------------ разбор моделью

LLM_PROMPT = """Переведи поисковый запрос по базе автомобилей в JSON-список условий.
Колонки: {columns}.
Операции: "=", ">", "<", ">=", "<=", "like" (часть строки), "between" (значение — [от, до]).
Значения текстовых колонок пиши заглавными буквами по-русски, марку и модель — латиницей (KIA RIO).
Регион пиши как в базе: САРАТОВСКАЯ ОБЛ, Г. МОСКВА; для города используй колонку city (Г. САРАТОВ).
Ответь только JSON: {{"conditions": [{{"column": "...", "op": "...", "value": ...}}]}}
Запрос: {query}"""


def parse_llm(text: str, base_url: str, model: str, timeout: int = 60) -> list[Cond]:
    body = {"model": model, "temperature": 0, "max_tokens": 400,
            "messages": [{"role": "user", "content": LLM_PROMPT.format(columns=", ".join(VEHICLE_COLUMNS), query=text)}]}
    r = requests.post(f"{base_url.rstrip('/')}/chat/completions", json=body, timeout=timeout)
    r.raise_for_status()
    from .engines.vlm import parse_answer
    data = parse_answer(r.json()["choices"][0]["message"]["content"]) or {}
    conds = []
    for c in data.get("conditions", []):
        col, op, val = c.get("column"), str(c.get("op", "")).lower(), c.get("value")
        if col not in VEHICLE_COLUMNS or op not in OPS or op == "len":
            continue                                   # всё, чего нет в белом списке, отбрасываем
        if op == "between" and not (isinstance(val, list) and len(val) == 2):
            continue
        conds.append(Cond(col, op, tuple(val) if op == "between" else val))
    return conds


def explain(conds: list[Cond]) -> list[str]:
    return [c.human() for c in conds]


def as_json(conds: list[Cond]) -> str:
    return json.dumps([c.__dict__ for c in conds], ensure_ascii=False)
