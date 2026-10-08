"""Базовая линия: классическое распознавание текста (Tesseract) + правила.

Как работает:
1. Tesseract превращает картинку в строки слов с координатами.
2. В каждой строке ищем самый большой промежуток между словами: слева от него
   подпись поля («Год выпуска ТС»), справа — значение («2011»).
3. Подпись нечётко сравниваем со списком известных подписей (OCR часто ошибается
   в буквах, поэтому точное совпадение не годится).
4. Значение чистим: значения в документах печатаются заглавными, а мусор от
   водяного знака и печатей — обычно строчные буквы и одиночные символы.

Перед распознаванием картинка готовится классическими методами обработки:
выравнивание освещения, бинаризация (Оцу), выпрямление наклона, удаление линий таблиц.

Это то, что можно сделать без нейросетей. Её задача — дать точку отсчёта.
"""
from __future__ import annotations

import os
import re
import shutil
import time
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image
from rapidfuzz import fuzz

from ..normalize import clean_fields
from ..schema import DOC_FIELDS, EPTS, PTS, STS_BACK, STS_FRONT
from .base import Engine, Extraction

# Подписи полей. Особые ключи: power_* — две мощности в одной строке,
# owner_full — ФИО одной строкой, address — «регион, город» (улица на следующей строке).
LABELS: dict[str, list[tuple[str, str]]] = {
    STS_FRONT: [
        ("plate", "регистрационный знак"),
        ("vin", "идентификационный номер vin"),
        ("make_model", "марка модель"),
        ("vehicle_type", "тип тс"),
        ("category", "категория тс a b c d прицеп"),
        ("year", "год выпуска тс"),
        ("chassis", "шасси рама"),
        ("body", "кузов кабина прицеп"),
        ("color", "цвет"),
        ("power_kw_hp", "мощность двигателя квт лс"),
        ("eco_class", "экологический класс"),
        ("pts_number", "паспорт тс серия"),
        ("max_mass", "разрешенная max масса кг"),
        ("curb_mass", "масса без нагрузки кг"),
    ],
    STS_BACK: [
        ("owner_last", "фамилия"),
        ("owner_first", "имя"),
        ("owner_middle", "отчество"),
        ("region", "республика край область"),
        ("city", "населенный пункт"),
        ("street_address", "улица дом корп кв"),
        ("gibdd_code", "код подразделения гибдд"),
        ("issue_date", "дата выдачи"),
    ],
    PTS: [
        ("vin", "1 идентификационный номер vin"),
        ("make_model", "2 марка модель тс"),
        ("vehicle_type", "3 наименование тип тс"),
        ("category", "4 категория тс a b c d прицеп"),
        ("year", "5 год изготовления тс"),
        ("engine_model", "6 модель двигателя"),
        ("chassis", "7 шасси рама"),
        ("body", "8 кузов кабина прицеп"),
        ("color", "9 цвет кузова кабины прицепа"),
        ("power_hp_kw", "10 мощность двигателя лс квт"),
        ("engine_volume", "11 рабочий объем двигателя куб см"),
        ("engine_type", "12 тип двигателя"),
        ("eco_class", "13 экологический класс"),
        ("max_mass", "14 разрешенная max масса кг"),
        ("curb_mass", "15 масса без нагрузки кг"),
        ("manufacturer", "16 организация изготовитель тс страна"),
        ("_approval", "17 одобрение типа тс"),
        ("_country", "18 страна вывоза тс"),
        ("_customs_doc", "19 серия тд тпо"),
        ("_customs", "20 таможенные ограничения"),
        ("owner_full", "21 наименование фио собственника тс"),
        ("address", "22 адрес"),
        ("_issuer", "23 наименование организации выдавшей паспорт"),
        ("issue_date", "24 дата выдачи паспорта"),
    ],
    EPTS: [
        ("doc_number", "номер электронного паспорта"),
        ("epts_status", "статус электронного паспорта"),
        ("issue_date", "дата оформления"),
        ("vin", "идентификационный номер vin"),
        ("make_model", "марка коммерческое наименование"),
        ("category", "категория"),
        ("year", "год изготовления"),
        ("color", "цвет"),
        ("engine_volume", "рабочий объем цилиндров см3"),
        ("power_kw_hp", "максимальная мощность квт лс"),
        ("engine_type", "тип двигателя"),
        ("eco_class", "экологический класс"),
        ("max_mass", "технически допустимая макс масса кг"),
        ("manufacturer", "изготовитель"),
        ("owner_last", "фамилия"),
        ("owner_first", "имя"),
        ("owner_middle", "отчество"),
        ("region", "субъект российской федерации"),
    ],
}

KEYWORDS = {
    EPTS: ["ВЫПИСКА", "ЭЛЕКТРОННОГО ПАСПОРТА"],
    PTS: ["ПАСПОРТ ТРАНСПОРТНОГО СРЕДСТВА", "ОДОБРЕНИЕ ТИПА", "ТАМОЖЕННЫЕ"],
    STS_BACK: ["СОБСТВЕННИК (ВЛАДЕЛЕЦ)", "ОСОБЫЕ ОТМЕТКИ", "КОД ПОДРАЗДЕЛЕНИЯ"],
    STS_FRONT: ["РЕГИСТРАЦИОННЫЙ ЗНАК", "ЭКОЛОГИЧЕСКИЙ КЛАСС", "СВИДЕТЕЛЬСТВО О РЕГИСТРАЦИИ"],
}


def _deskew_angle(binary_inv: np.ndarray) -> float:
    """Угол наклона текста: при правильном повороте строки дают самые «резкие» горизонтальные полосы."""
    h, w = binary_inv.shape
    scale = 500 / max(w, h)
    small = cv2.resize(binary_inv, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    sh, sw = small.shape
    center = (sw / 2, sh / 2)

    def score(a: float) -> float:
        m = cv2.getRotationMatrix2D(center, a, 1.0)
        rot = cv2.warpAffine(small, m, (sw, sh))
        return float(np.var(rot.sum(axis=1)))

    best = max(np.arange(-25, 25.5, 1.0), key=score)
    return float(max(np.arange(best - 1, best + 1.01, 0.25), key=score))


def preprocess(img: Image.Image, binarize: bool = True, scale: float = 1.0) -> Image.Image:
    """Подготовка картинки для Tesseract.
    binarize=True — чёрно-белая картинка: лучше всего на чистых сканах.
    binarize=False — серая с увеличением: мелкий размытый текст после бинаризации пропадает, а так читается."""
    gray = cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2GRAY)
    if scale != 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    # 1. выравниваем освещение: делим на размытый фон (убирает тени и блики)
    bg = cv2.medianBlur(cv2.resize(gray, None, fx=0.25, fy=0.25), 21)
    bg = cv2.resize(bg, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_LINEAR)
    flat = cv2.divide(gray, bg, scale=255)
    # 2. бинаризация по Оцу: светлый водяной знак и узор фона уходят в белое
    _, binary = cv2.threshold(flat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inv = 255 - binary
    # 3. выпрямляем наклон
    angle = _deskew_angle(inv)
    h, w = inv.shape
    if abs(angle) > 0.3:
        m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        inv = cv2.warpAffine(inv, m, (w, h))
        flat = cv2.warpAffine(flat, m, (w, h), borderValue=255)
    # 4. убираем тонкие длинные линии (рамки таблиц). Толстые полосы — это размытый текст, их не трогаем
    hl = cv2.morphologyEx(inv, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(40, w // 18), 1)))
    vl = cv2.morphologyEx(inv, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(40, h // 18))))
    t = max(5, round(5 * scale))
    hl[cv2.morphologyEx(hl, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, t))) > 0] = 0
    vl[cv2.morphologyEx(vl, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (t, 1))) > 0] = 0
    lines = (hl > 0) | (vl > 0)
    if binarize:
        inv[lines] = 0
        return Image.fromarray(255 - inv)
    flat[lines] = 255
    return Image.fromarray(flat)


@dataclass
class Word:
    text: str
    left: int
    right: int
    top: int
    height: int


_LOOKALIKE = str.maketrans("acekmhopbtxy", "асекмнорвтху")


def _norm_label(s: str) -> str:
    """Подпись в простой вид: строчные буквы, латиница-двойник → кириллица, без знаков."""
    s = s.lower().replace("ё", "е").translate(_LOOKALIKE)
    s = re.sub(r"[^a-zа-я0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _is_value_token(t: str) -> bool:
    """Значения печатаются заглавными; строчные буквы и одиночные символы — мусор."""
    if not re.search(r"[A-ZА-ЯЁ0-9]", t):
        return False
    return not re.search(r"[a-zа-яё]", t)


def _find_tesseract() -> str | None:
    env = os.environ.get("OKO_TESSERACT")
    if env and os.path.exists(env):
        return env
    found = shutil.which("tesseract")
    if found:
        return found
    local = os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "Tesseract-OCR", "tesseract.exe")
    for p in (r"C:\Program Files\Tesseract-OCR\tesseract.exe", r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe", local):
        if os.path.exists(p):
            return p
    return None


class TesseractEngine(Engine):
    name = "tesseract"

    def __init__(self, lang: str = "rus+eng", psm: int = 6):
        self.lang = lang
        self.psm = psm
        self.labels = {dt: [(k, _norm_label(lab)) for k, lab in labs] for dt, labs in LABELS.items()}
        import pytesseract
        self._pt = pytesseract
        exe = _find_tesseract()
        if exe:
            pytesseract.pytesseract.tesseract_cmd = exe

    def available(self) -> tuple[bool, str]:
        if not _find_tesseract():
            return False, "Tesseract не установлен (Windows: установщик UB Mannheim, Linux: apt install tesseract-ocr)"
        try:
            langs = self._pt.get_languages()
        except Exception as e:  # noqa: BLE001
            return False, f"Tesseract не запускается: {e}"
        if "rus" not in langs:
            return False, "У Tesseract нет русского языка (rus.traineddata)"
        return True, ""

    # ------------------------------------------------------------- распознавание
    def _lines(self, img: Image.Image) -> list[list[Word]]:
        d = self._pt.image_to_data(img, lang=self.lang, config=f"--psm {self.psm}",
                                   output_type=self._pt.Output.DICT)
        lines: dict[tuple, list[Word]] = {}
        for i, txt in enumerate(d["text"]):
            if not txt.strip():
                continue
            key = (d["block_num"][i], d["par_num"][i], d["line_num"][i])
            lines.setdefault(key, []).append(
                Word(txt.strip(), d["left"][i], d["left"][i] + d["width"][i], d["top"][i], d["height"][i]))
        return [sorted(ws, key=lambda w: w.left) for ws in lines.values()]

    @staticmethod
    def _split(line: list[Word], width: int) -> tuple[str, list[Word]]:
        """Делим строку на подпись и значение по первому большому промежутку.
        Строка, которая начинается правее трети листа, — значение без подписи."""
        if line[0].left > width * 0.3:
            return "", line
        heights = sorted(w.height for w in line)
        threshold = max(28, 1.8 * heights[len(heights) // 2])
        for i in range(len(line) - 1):
            if line[i + 1].left - line[i].right > threshold:
                return " ".join(w.text for w in line[: i + 1]), line[i + 1:]
        return " ".join(w.text for w in line), []

    @staticmethod
    def _clean(value: list[Word]) -> str:
        """Оставляем непрерывный кусок значения, мусор после большого разрыва отбрасываем."""
        out: list[Word] = []
        for w in value:
            if re.search(r"[a-zа-яё]", w.text):          # строчные буквы — мусор
                if out:
                    break
                continue
            if not re.search(r"[A-ZА-ЯЁ0-9]", w.text):    # «/», «(», «=» — пропускаем
                continue
            if out:
                h = max(out[-1].height, w.height, 10)
                if w.left - out[-1].right > 3.2 * h:
                    break
            out.append(w)
        return " ".join(w.text for w in out).strip(" =|‚,")

    def _doc_type(self, text: str, lines_labels: list[str]) -> str:
        up = text.upper().replace("Ё", "Е")
        scores = {}
        for dt, kws in KEYWORDS.items():
            scores[dt] = sum(2 for k in kws if k in up)
            labels = [lab for _, lab in self.labels[dt]]
            for ll in lines_labels:
                if ll and max(fuzz.ratio(ll, lab) for lab in labels) >= 80:
                    scores[dt] += 1
        return max(scores, key=scores.get)

    def extract(self, img: Image.Image) -> Extraction:
        """Два прохода: сначала чёрно-белая картинка; если прочитано меньше 3/4 полей —
        ещё раз по серой увеличенной, и недостающие поля берём оттуда."""
        t0 = time.time()
        try:
            doc_type, fields, text = self._read(preprocess(img, binarize=True))
            expected = len(DOC_FIELDS.get(doc_type, ())) or 16
            if len(fields) < 0.75 * expected:
                dt2, fields2, text2 = self._read(preprocess(img, binarize=False, scale=1.6))
                if len(fields2) > len(fields):
                    doc_type, fields, fields2, text = dt2, fields2, fields, text2
                for k, v in fields2.items():
                    fields.setdefault(k, v)
        except Exception as e:  # noqa: BLE001
            return Extraction(None, {}, time.time() - t0, self.name, error=str(e))
        return Extraction(doc_type, clean_fields(fields), time.time() - t0, self.name, raw=text)

    def _read(self, img: Image.Image) -> tuple[str, dict[str, str], str]:
        lines = self._lines(img)
        width = img.width
        parsed = []
        for ln in lines:
            label, value = self._split(ln, width)
            parsed.append((_norm_label(label), self._clean(value), ln))
        text = "\n".join(" ".join(w.text for w in ln) for ln in lines)
        doc_type = self._doc_type(text, [p[0] for p in parsed])

        fields: dict[str, str] = {}
        best: dict[str, float] = {}
        labels = self.labels[doc_type]
        for i, (label, value, ln) in enumerate(parsed):
            if not label or not value:
                continue
            score, key = max((fuzz.ratio(label, lab), k) for k, lab in labels)
            if len(label) <= 6:   # короткие подписи («Имя», «Цвет») сравниваем строже
                score -= 10
            if score < 70 or score <= best.get(key, 0):
                continue
            best[key] = score
            self._put(fields, key, value, parsed, i, width)

        self._regex_fallbacks(doc_type, text, fields)
        return doc_type, fields, text

    @staticmethod
    def _numbers(s: str) -> list[str]:
        return re.findall(r"\d+", s)

    def _put(self, fields, key, value, parsed, i, width):
        if key.startswith("_"):
            return
        if key == "power_kw_hp":
            nums = self._numbers(value)
            if len(nums) >= 2:
                fields["power_kw"], fields["power_hp"] = nums[0], nums[1]
        elif key == "power_hp_kw":
            nums = self._numbers(value)
            if len(nums) >= 2:
                fields["power_hp"], fields["power_kw"] = nums[0], nums[1]
        elif key == "owner_full":
            parts = value.replace(",", " ").split()
            for k, p in zip(("owner_last", "owner_first", "owner_middle"), parts):
                fields[k] = p
        elif key == "address":
            region, _, city = value.partition(",")
            fields["region"], fields["city"] = region.strip(), city.strip()
            if i + 1 < len(parsed):          # улица — на следующей строке без подписи
                nlabel, nvalue, _ = parsed[i + 1]
                if not nlabel and nvalue:
                    fields["street_address"] = nvalue
        elif key in ("year", "max_mass", "curb_mass", "engine_volume", "gibdd_code"):
            nums = self._numbers(value)
            if nums:
                fields[key] = max(nums, key=len)
        else:
            fields[key] = value

    @staticmethod
    def _regex_fallbacks(doc_type: str, text: str, fields: dict):
        up = text.upper()
        if doc_type in (STS_FRONT, STS_BACK):
            m = list(re.finditer(r"(?<!\d)(\d{2})\s?(\d{2})\s?(\d{6})(?!\d)", up))
            if m:
                g = m[-1].groups()
                fields["doc_number"] = f"{g[0]} {g[1]} {g[2]}"
        elif doc_type == PTS:
            m = re.search(r"(?<!\d)(\d{2})\s?([А-ЯA-Z0О]{2})\s?(\d{6})(?!\d)", up)
            if m:
                fields["doc_number"] = " ".join(m.groups())
        if "vin" not in fields and doc_type != STS_BACK:
            m = re.search(r"\b[A-HJ-NPR-Z0-9]{17}\b", up)
            if m:
                fields["vin"] = m.group(0)
        if "issue_date" not in fields and doc_type in (STS_BACK, PTS, EPTS):
            dates = re.findall(r"\d{2}\.\d{2}\.\d{4}", up)
            if dates:
                fields["issue_date"] = dates[-1]
