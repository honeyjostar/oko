"""Приведение значений к единому виду и проверки данных.

1. `canon()` — каноническая форма значения. Модель может написать «Д.12»
   вместо «Д. 12» или латинскую «A» вместо русской «А» в госномере.
   Это не ошибка распознавания, поэтому сравниваем канонические формы.
2. `checks()` — проверки документа: формат VIN и номеров, сверка полей
   между собой (мощность в кВт и л.с., год по VIN и год выпуска и т. д.).
"""
from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass

from .fakedata import PLATE_CODE_TO_REGION, VIN_YEAR, vin_check_digit
from .schema import EPTS, FIELD_BY_KEY, PTS, STS_BACK, STS_FRONT

CYR_TO_LAT = str.maketrans("АВЕКМНОРСТУХ", "ABEKMHOPCTYX")
LAT_TO_CYR = str.maketrans("ABEKMHOPCTYX", "АВЕКМНОРСТУХ")
PLATE_LETTERS = "АВЕКМНОРСТУХ"

# Поля, которые пишутся латиницей, и поля, которые пишутся кириллицей.
# Распознавание часто путает похожие буквы (русская «С» и латинская «C»),
# поэтому после распознавания приводим буквы к алфавиту поля.
LATIN_FIELDS = {"vin", "body", "make_model", "category"}
CYRILLIC_FIELDS = {"vehicle_type", "color", "eco_class", "chassis", "engine_type", "owner_last", "owner_first",
                   "owner_middle", "region", "city", "street_address", "epts_status"}

PLATE_RE = re.compile(r"^[АВЕКМНОРСТУХ]\d{3}[АВЕКМНОРСТУХ]{2}\d{2,3}$")
VIN_RE = re.compile(r"^[A-HJ-NPR-Z0-9]{17}$")
STS_NUMBER_RE = re.compile(r"^\d{10}$")
PTS_NUMBER_RE = re.compile(r"^\d{2}[А-Я]{2}\d{6}$")
EPTS_NUMBER_RE = re.compile(r"^\d{15}$")
DATE_RE = re.compile(r"(\d{1,2})\s*[./\-]\s*(\d{1,2})\s*[./\-]\s*(\d{4})")


def _base(value) -> str:
    if value is None:
        return ""
    s = str(value).upper().replace("Ё", "Е").replace(" ", " ")
    s = re.sub(r"\s+", " ", s).strip()
    if s in {"NULL", "NONE", "N/A", "НЕТ ДАННЫХ"}:
        return ""
    return s.strip(" .,;:")


def fix_vin(s: str) -> str:
    s = re.sub(r"[\s\-]", "", s).translate(CYR_TO_LAT)
    return s.replace("O", "0").replace("I", "1").replace("Q", "0")


def fix_plate(s: str) -> str:
    """Госномер: буквы — кириллица, цифры — цифры. Чиним по позициям: Л ЦЦЦ ЛЛ ЦЦ(Ц)."""
    s = re.sub(r"[\s\-|]", "", s).translate(LAT_TO_CYR)
    as_letter = {"0": "О", "8": "В"}
    as_digit = {"О": "0", "З": "3", "Б": "6"}
    out = []
    for i, ch in enumerate(s):
        if i in (0, 4, 5):
            ch = as_letter.get(ch, ch)
        else:
            ch = as_digit.get(ch, ch)
        out.append(ch)
    return "".join(out)


def fix_series(s: str) -> str:
    """Номер ПТС «63 ОХ 123456»: буквы кириллицей, без пробелов."""
    s = re.sub(r"[\s\-№N]", "", s)
    mid = s[2:4]
    # буквы серии бывают только у бумажного ПТС; номер СТС — одни цифры
    if len(s) == 10 and s[:2].isdigit() and s[4:].isdigit() and not mid.isdigit():
        s = s[:2] + mid.translate(LAT_TO_CYR).replace("0", "О") + s[4:]
    return s


def canon(key: str, value) -> str:
    """Каноническая форма значения поля — для сравнения с правильным ответом."""
    s = _base(value)
    if not s:
        return ""
    kind = FIELD_BY_KEY[key].kind if key in FIELD_BY_KEY else "text"
    if key in ("vin", "body"):
        v = fix_vin(s)
        return v if (key == "vin" or len(v) == 17) else re.sub(r"[^А-ЯA-Z0-9]", "", s)
    if key == "plate":
        return fix_plate(s)
    if key in ("doc_number", "pts_number"):
        return fix_series(s)
    if kind == "code":
        return re.sub(r"[\s\-]", "", s)
    if kind == "int":
        digits = re.sub(r"\D", "", s)
        return str(int(digits)) if digits else ""
    if kind == "date":
        m = DATE_RE.search(s)
        if not m:
            return re.sub(r"\s", "", s)
        d, mth, y = m.groups()
        return f"{int(d):02d}.{int(mth):02d}.{y}"
    if key in LATIN_FIELDS:
        s = s.translate(CYR_TO_LAT)
    elif key in CYRILLIC_FIELDS:
        s = s.translate(LAT_TO_CYR)
    # обычный текст: сравниваем только буквы и цифры
    return re.sub(r"[^А-ЯA-Z0-9]", "", s)


def clean_fields(fields: dict) -> dict[str, str]:
    """Чистка результата любого движка: пробелы, алфавит полей, пустые значения.
    Применяется одинаково ко всем движкам, поэтому сравнение честное."""
    out: dict[str, str] = {}
    for key, value in fields.items():
        v = _base(value)
        if not v:
            continue
        if key in ("vin", "body") and len(re.sub(r"[\s\-]", "", v)) == 17:
            v = fix_vin(v)
        elif key in LATIN_FIELDS:
            v = v.translate(CYR_TO_LAT)
        elif key in CYRILLIC_FIELDS:
            v = v.translate(LAT_TO_CYR)
        elif key == "plate":
            v = fix_plate(v)
        out[key] = v
    return out


def same(key: str, pred, truth) -> bool:
    return canon(key, pred) == canon(key, truth)


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(key: str, pred, truth) -> float:
    """Доля ошибочных символов (Character Error Rate) после приведения к единому виду."""
    t, p = canon(key, truth), canon(key, pred)
    if not t:
        return 0.0 if not p else 1.0
    return min(1.0, levenshtein(p, t) / len(t))


# ------------------------------------------------------------------ проверки

@dataclass
class Check:
    level: str      # ok | info | warn | error
    field: str
    message: str


def _int(v) -> int | None:
    d = re.sub(r"\D", "", str(v or ""))
    return int(d) if d else None


def _date(v) -> dt.date | None:
    m = DATE_RE.search(str(v or ""))
    if not m:
        return None
    d, mth, y = (int(x) for x in m.groups())
    try:
        return dt.date(y, mth, d)
    except ValueError:
        return None


def checks(doc_type: str, fields: dict) -> list[Check]:
    """Проверки одного документа. Возвращает список замечаний (и подтверждений)."""
    out: list[Check] = []
    f = {k: v for k, v in fields.items() if _base(v)}

    if "vin" in f:
        vin = canon("vin", f["vin"])
        if not VIN_RE.match(vin):
            out.append(Check("error", "vin", f"VIN «{vin}» не похож на VIN: нужно 17 символов без I, O, Q"))
        else:
            out.append(Check("ok", "vin", "Формат VIN верный"))
            if vin_check_digit(vin) != vin[8]:
                out.append(Check("info", "vin", "9-й символ VIN — не контрольная цифра. Она обязательна только "
                                                "для машин рынка Северной Америки, так что это не ошибка"))
            year = _int(f.get("year"))
            if year and VIN_YEAR.get(year) and VIN_YEAR[year] != vin[9]:
                out.append(Check("warn", "year", f"Год выпуска {year} не совпадает с годом, зашитым в VIN (10-й символ «{vin[9]}»)"))
            elif year and VIN_YEAR.get(year):
                out.append(Check("ok", "year", "Год выпуска совпадает с годом в VIN"))
        if "body" in f and canon("body", f["body"]) not in ("", "ОТСУТСТВУЕТ") and canon("body", f["body"]) != vin:
            out.append(Check("warn", "body", "Номер кузова отличается от VIN"))

    if "plate" in f:
        plate = canon("plate", f["plate"])
        if not PLATE_RE.match(plate):
            out.append(Check("error", "plate", f"Госномер «{plate}» не соответствует формату А123ВС164"))
        else:
            region = PLATE_CODE_TO_REGION.get(plate[6:])
            out.append(Check("ok", "plate", f"Госномер верного формата, регион {plate[6:]}" + (f" — {region}" if region else "")))

    kw, hp = _int(f.get("power_kw")), _int(f.get("power_hp"))
    if kw and hp:
        if abs(round(hp * 0.7355) - kw) > 1:
            out.append(Check("warn", "power_kw", f"{kw} кВт ≠ {hp} л.с. (должно быть около {round(hp * 0.7355)} кВт)"))
        else:
            out.append(Check("ok", "power_kw", "Мощность в кВт и л.с. сходится"))

    cm, mm = _int(f.get("curb_mass")), _int(f.get("max_mass"))
    if cm and mm:
        out.append(Check("ok", "curb_mass", "Масса без нагрузки меньше разрешённой") if cm < mm else
                   Check("warn", "curb_mass", "Масса без нагрузки больше разрешённой максимальной"))

    year = _int(f.get("year"))
    if year is not None and not (1950 <= year <= dt.date.today().year + 1):
        out.append(Check("error", "year", f"Странный год выпуска: {year}"))

    if "issue_date" in f:
        d = _date(f["issue_date"])
        if not d:
            out.append(Check("error", "issue_date", "Дата выдачи не читается как ДД.ММ.ГГГГ"))
        elif year and d.year < year - 1:
            out.append(Check("warn", "issue_date", "Документ выдан раньше, чем выпущена машина"))

    if "doc_number" in f:
        num = canon("doc_number", f["doc_number"])
        if doc_type in (STS_FRONT, STS_BACK):
            ok = STS_NUMBER_RE.match(num)
            what = "СТС: 10 цифр"
        elif doc_type == PTS:
            ok = PTS_NUMBER_RE.match(num)
            what = "ПТС: 2 цифры, 2 буквы, 6 цифр"
        else:
            ok = EPTS_NUMBER_RE.match(num)
            what = "ЭПТС: 15 цифр"
        out.append(Check("ok", "doc_number", "Номер документа верного формата") if ok else
                   Check("error", "doc_number", f"Номер документа «{num}» не похож на номер {what}"))

    if "pts_number" in f:
        num = canon("pts_number", f["pts_number"])
        if not (PTS_NUMBER_RE.match(num) or EPTS_NUMBER_RE.match(num)):
            out.append(Check("warn", "pts_number", "Номер ПТС в СТС не похож ни на бумажный ПТС, ни на ЭПТС"))

    for key in ("owner_last", "owner_first", "owner_middle"):
        if key in f and not re.fullmatch(r"[А-ЯЁ\- ]+", _base(f[key])):
            out.append(Check("warn", key, f"В поле «{FIELD_BY_KEY[key].title}» есть не только русские буквы"))
    return out


def detect_doc_type_by_fields(fields: dict) -> str | None:
    """Если модель не назвала вид документа — угадываем по набору полей."""
    keys = {k for k, v in fields.items() if _base(v)}
    if "epts_status" in keys:
        return EPTS
    if "plate" in keys:
        return STS_FRONT
    if {"owner_last", "gibdd_code"} & keys and "vin" not in keys:
        return STS_BACK
    if "vin" in keys:
        return PTS
    return None
