"""Рисование синтетических документов.

Каждый бланк — приблизительный макет, а не копия настоящего: без гербов,
голограмм и защитных сеток. На каждом листе крупный водяной знак
«ОБРАЗЕЦ · СИНТЕТИЧЕСКИЕ ДАННЫЕ», чтобы картинку нельзя было выдать за документ.

Для каждого значения запоминается рамка (x0, y0, x1, y1) — это разметка
набора данных: где на картинке лежит каждое поле.
"""
from __future__ import annotations

import math
import random
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

from .fakedata import Vehicle, doc_truth
from .schema import EPTS, PTS, STS_BACK, STS_FRONT

FONT_DIR = Path(__file__).parent / "fonts"

Box = tuple[int, int, int, int]


@lru_cache(maxsize=128)
def font(name: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_DIR / name), size)


LABEL_FONT = "DejaVuSansCondensed.ttf"
LABEL_BOLD = "DejaVuSansCondensed-Bold.ttf"
VALUE_FONTS = {
    STS_FRONT: ["LiberationMono-Bold.ttf", "DejaVuSansMono.ttf", "LiberationSans-Bold.ttf", "Carlito-Bold.ttf"],
    STS_BACK: ["LiberationMono-Bold.ttf", "DejaVuSansMono.ttf", "LiberationSans-Bold.ttf", "Carlito-Bold.ttf"],
    PTS: ["LiberationMono-Regular.ttf", "LiberationMono-Bold.ttf", "DejaVuSansMono.ttf", "DejaVuSerif.ttf"],
    EPTS: ["LiberationSans-Regular.ttf", "Carlito-Regular.ttf", "DejaVuSans.ttf"],
}


class Sheet:
    """Лист, на котором рисуем. Помнит рамки значений полей."""

    def __init__(self, size: tuple[int, int], paper: tuple[int, int, int], rng: random.Random):
        self.rng = rng
        self.img = Image.new("RGB", size, paper)
        self.draw = ImageDraw.Draw(self.img)
        self.boxes: dict[str, Box] = {}

    def text(self, xy, text, fnt, fill=(70, 70, 80), anchor="la"):
        self.draw.text(xy, text, font=fnt, fill=fill, anchor=anchor)
        return self.draw.textbbox(xy, text, font=fnt, anchor=anchor)

    def value(self, key: str | None, xy, text: str, fnt, fill) -> Box:
        """Печатает значение поля с небольшим сдвигом, как принтер на бланке."""
        x, y = xy
        x += self.rng.randint(-3, 5)
        y += self.rng.randint(-3, 3)
        box = self.text((x, y), text, fnt, fill=fill)
        if key:
            self.boxes[key] = tuple(int(v) for v in box)
        return box

    def values_inline(self, parts: list[tuple[str | None, str]], xy, fnt, fill, max_w: int | None = None) -> None:
        """Несколько полей в одной строке: «78 / 106» или «ИВАНОВ ИВАН ИВАНОВИЧ».
        Если строка не влезает в max_w, шрифт уменьшается."""
        x, y = xy
        x += self.rng.randint(-3, 5)
        y += self.rng.randint(-3, 3)
        if max_w:
            width = self.draw.textlength("".join(t for _, t in parts), font=fnt)
            if width > max_w:
                fnt = ImageFont.truetype(fnt.path, max(12, int(fnt.size * max_w / width)))
        for key, text in parts:
            box = self.text((x, y), text, fnt, fill=fill)
            if key:
                self.boxes[key] = tuple(int(v) for v in box)
            x = box[2]


def _ink(rng: random.Random) -> tuple[int, int, int]:
    """Цвет «краски» принтера: почти чёрный или тёмно-синий."""
    base = rng.choice([(25, 25, 30), (20, 28, 60), (35, 35, 40), (15, 20, 45)])
    return tuple(max(0, min(255, c + rng.randint(-8, 8))) for c in base)


def _waves(sheet: Sheet, box: Box, color, step=14, amp=5.0, period=180.0):
    """Тонкие волнистые линии фона — просто чтобы бумага не была идеально пустой."""
    x0, y0, x1, y1 = box
    phase = sheet.rng.random() * 6.28
    for y in range(y0, y1, step):
        pts = [(x, y + amp * math.sin(x / period * 6.28 + phase + y / 90)) for x in range(x0, x1, 8)]
        sheet.draw.line(pts, fill=color, width=1)


def _watermark(img: Image.Image, rng: random.Random) -> Image.Image:
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    w, h = img.size
    big = font("DejaVuSans-Bold.ttf", int(min(w, h) * 0.16))
    small = font("DejaVuSansCondensed.ttf", int(min(w, h) * 0.028))
    d.text((w / 2, h / 2), "ОБРАЗЕЦ", font=big, fill=(200, 40, 40, 42), anchor="mm")
    d.text((w / 2, h / 2 + min(w, h) * 0.12), "СИНТЕТИЧЕСКИЕ ДАННЫЕ · OKO · НЕ ЯВЛЯЕТСЯ ДОКУМЕНТОМ",
           font=small, fill=(200, 40, 40, 70), anchor="mm")
    layer = layer.rotate(rng.uniform(18, 28), resample=Image.BICUBIC, center=(w / 2, h / 2))
    out = img.convert("RGBA")
    out.alpha_composite(layer)
    return out.convert("RGB")


def _signature(sheet: Sheet, x: int, y: int, w: int = 150):
    r = sheet.rng
    pts = []
    for i in range(40):
        t = i / 39
        pts.append((x + t * w, y + 14 * math.sin(t * r.uniform(9, 14) + r.random()) * (1 - t * 0.5)))
    sheet.draw.line(pts, fill=(30, 40, 110), width=2, joint="curve")


def _stamp(sheet: Sheet, cx: int, cy: int, r: int, text: str):
    color = (60, 80, 170)
    d = sheet.draw
    d.ellipse((cx - r, cy - r, cx + r, cy + r), outline=color, width=3)
    d.ellipse((cx - r + 10, cy - r + 10, cx + r - 10, cy + r - 10), outline=color, width=2)
    f = font(LABEL_BOLD, max(10, r // 6))
    n = len(text)
    for i, ch in enumerate(text):
        a = -math.pi / 2 + 2 * math.pi * i / n
        d.text((cx + (r - 22) * math.cos(a), cy + (r - 22) * math.sin(a)), ch, font=f, fill=color, anchor="mm")
    d.text((cx, cy), "★", font=font("DejaVuSans.ttf", r // 3), fill=color, anchor="mm")


def _scan_bed(card: Image.Image, rng: random.Random, pad: int = 60) -> tuple[Image.Image, int, int]:
    """Кладём карточку на «стекло сканера»: светлый фон и мягкая тень."""
    w, h = card.size
    bed = Image.new("RGB", (w + 2 * pad, h + 2 * pad), (238 + rng.randint(-6, 6),) * 3)
    shadow = Image.new("L", bed.size, 0)
    ImageDraw.Draw(shadow).rounded_rectangle((pad + 6, pad + 8, pad + w + 6, pad + h + 8), 18, fill=90)
    shadow = shadow.filter(ImageFilter.GaussianBlur(10))
    bed.paste((200, 200, 200), mask=shadow)
    mask = Image.new("L", card.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, w - 1, h - 1), 22, fill=255)
    bed.paste(card, (pad, pad), mask)
    return bed, pad, pad


def _shift_boxes(boxes: dict[str, Box], dx: int, dy: int) -> dict[str, Box]:
    return {k: (b[0] + dx, b[1] + dy, b[2] + dx, b[3] + dy) for k, b in boxes.items()}


# ---------------------------------------------------------------- СТС

STS_SIZE = (1240, 800)


def _sts_base(rng: random.Random, title_right: str) -> Sheet:
    paper = (246, 233 + rng.randint(-4, 4), 236 + rng.randint(-4, 4))
    s = Sheet(STS_SIZE, paper, rng)
    w, h = STS_SIZE
    _waves(s, (0, 0, w, h), (236, 214, 220), step=11)
    s.draw.rounded_rectangle((18, 18, w - 18, h - 18), 16, outline=(205, 160, 170), width=2)
    s.text((w / 2, 46), "РОССИЙСКАЯ ФЕДЕРАЦИЯ", font(LABEL_FONT, 17), fill=(120, 70, 80), anchor="mm")
    s.text((w / 2, 78), "СВИДЕТЕЛЬСТВО О РЕГИСТРАЦИИ ТРАНСПОРТНОГО СРЕДСТВА", font(LABEL_BOLD, 24),
           fill=(110, 50, 60), anchor="mm")
    s.text((w - 50, 46), title_right, font(LABEL_FONT, 15), fill=(140, 90, 100), anchor="rm")
    return s


def _rows(s: Sheet, rows, x_label: int, x_value: int, y0: int, step: int, vfont, ink, lfont):
    y = y0
    max_w = s.img.width - x_value - 60
    for label, parts in rows:
        s.text((x_label, y + 4), label, lfont, fill=(105, 85, 90))
        if isinstance(parts, str):
            parts = [(None, parts)]
        s.values_inline(parts, (x_value, y), vfont, ink, max_w)
        y += step
    return y


def render_sts_front(v: Vehicle, rng: random.Random):
    s = _sts_base(rng, "лицевая сторона")
    t = doc_truth(v, STS_FRONT)
    ink = _ink(rng)
    vf = font(rng.choice(VALUE_FONTS[STS_FRONT]), 23)
    lf = font(LABEL_FONT, 16)
    s.text((60, 128), "Регистрационный знак", font(LABEL_FONT, 18), fill=(105, 85, 90))
    s.value("plate", (470, 116), t["plate"], font(rng.choice(VALUE_FONTS[STS_FRONT]), 36), ink)
    rows = [
        ("Идентификационный номер (VIN)", [("vin", t["vin"])]),
        ("Марка, модель", [("make_model", t["make_model"])]),
        ("Тип ТС", [("vehicle_type", t["vehicle_type"])]),
        ("Категория ТС (A, B, C, D, прицеп)", [("category", t["category"])]),
        ("Год выпуска ТС", [("year", t["year"])]),
        ("Шасси (рама) №", [("chassis", t["chassis"])]),
        ("Кузов (кабина, прицеп) №", [("body", t["body"])]),
        ("Цвет", [("color", t["color"])]),
        ("Мощность двигателя, кВт / л.с.", [("power_kw", t["power_kw"]), (None, " / "), ("power_hp", t["power_hp"])]),
        ("Экологический класс", [("eco_class", t["eco_class"])]),
        ("Паспорт ТС серия №", [("pts_number", t["pts_number"])]),
        ("Разрешенная max масса, кг", [("max_mass", t["max_mass"])]),
        ("Масса без нагрузки, кг", [("curb_mass", t["curb_mass"])]),
    ]
    _rows(s, rows, 60, 470, 180, 41, vf, ink, lf)
    s.value("doc_number", (STS_SIZE[0] - 330, STS_SIZE[1] - 70), t["doc_number"],
            font("LiberationMono-Bold.ttf", 30), (185, 30, 40))
    return s, t


def render_sts_back(v: Vehicle, rng: random.Random):
    s = _sts_base(rng, "оборотная сторона")
    t = doc_truth(v, STS_BACK)
    ink = _ink(rng)
    vf = font(rng.choice(VALUE_FONTS[STS_BACK]), 24)
    lf = font(LABEL_FONT, 17)
    s.text((60, 130), "СОБСТВЕННИК (ВЛАДЕЛЕЦ)", font(LABEL_BOLD, 19), fill=(110, 50, 60))
    rows = [
        ("Фамилия", [("owner_last", t["owner_last"])]),
        ("Имя", [("owner_first", t["owner_first"])]),
        ("Отчество", [("owner_middle", t["owner_middle"])]),
        ("Республика, край, область", [("region", t["region"])]),
        ("Населенный пункт", [("city", t["city"])]),
        ("Улица, дом, корп., кв.", [("street_address", t["street_address"])]),
    ]
    y = _rows(s, rows, 60, 400, 172, 46, vf, ink, lf)
    s.text((60, y + 10), "Особые отметки", lf, fill=(105, 85, 90))
    s.text((400, y + 8), rng.choice(["—", "ЗАМЕНА ЦВЕТА НЕ ПРОИЗВОДИЛАСЬ", "ВЫДАНО ВЗАМЕН УТРАЧЕННОГО"]),
           font(rng.choice(VALUE_FONTS[STS_BACK]), 19), fill=ink)
    y += 70
    s.text((60, y + 4), "Код подразделения ГИБДД", lf, fill=(105, 85, 90))
    s.value("gibdd_code", (400, y), t["gibdd_code"], vf, ink)
    s.text((60, y + 50), "Дата выдачи", lf, fill=(105, 85, 90))
    s.value("issue_date", (400, y + 46), t["issue_date"], vf, ink)
    _signature(s, 760, y + 40)
    _stamp(s, 1060, y - 40, 78, "ГОСАВТОИНСПЕКЦИЯ · РЭО · ")
    s.value("doc_number", (STS_SIZE[0] - 330, STS_SIZE[1] - 70), t["doc_number"],
            font("LiberationMono-Bold.ttf", 30), (185, 30, 40))
    return s, t


# ---------------------------------------------------------------- ПТС

PTS_SIZE = (1100, 1600)


def render_pts(v: Vehicle, rng: random.Random):
    paper = (226 + rng.randint(-4, 4), 238, 228 + rng.randint(-4, 4))
    s = Sheet(PTS_SIZE, paper, rng)
    w, h = PTS_SIZE
    _waves(s, (0, 0, w, h), (210, 226, 212), step=12, period=220)
    t = doc_truth(v, PTS)
    ink = _ink(rng)
    vf = font(rng.choice(VALUE_FONTS[PTS]), 21)
    lf = font(LABEL_FONT, 15)
    s.text((w / 2, 60), "ПАСПОРТ ТРАНСПОРТНОГО СРЕДСТВА", font(LABEL_BOLD, 30), fill=(40, 80, 55), anchor="mm")
    s.value("doc_number", (w - 330, 98), t["doc_number"], font("LiberationMono-Bold.ttf", 28), (185, 30, 40))
    owner = [("owner_last", t["owner_last"]), (None, " "), ("owner_first", t["owner_first"]), (None, " "),
             ("owner_middle", t["owner_middle"])]
    rows = [
        ("1. Идентификационный номер (VIN)", [("vin", t["vin"])]),
        ("2. Марка, модель ТС", [("make_model", t["make_model"])]),
        ("3. Наименование (тип ТС)", [("vehicle_type", t["vehicle_type"])]),
        ("4. Категория ТС (A, B, C, D, прицеп)", [("category", t["category"])]),
        ("5. Год изготовления ТС", [("year", t["year"])]),
        ("6. Модель, № двигателя", [("engine_model", t["engine_model"])]),
        ("7. Шасси (рама) №", [("chassis", t["chassis"])]),
        ("8. Кузов (кабина, прицеп) №", [("body", t["body"])]),
        ("9. Цвет кузова (кабины, прицепа)", [("color", t["color"])]),
        ("10. Мощность двигателя, л.с. (кВт)", [("power_hp", t["power_hp"]), (None, " ("), ("power_kw", t["power_kw"]), (None, ")")]),
        ("11. Рабочий объем двигателя, куб. см", [("engine_volume", t["engine_volume"])]),
        ("12. Тип двигателя", [("engine_type", t["engine_type"])]),
        ("13. Экологический класс", [("eco_class", t["eco_class"])]),
        ("14. Разрешенная max масса, кг", [("max_mass", t["max_mass"])]),
        ("15. Масса без нагрузки, кг", [("curb_mass", t["curb_mass"])]),
        ("16. Организация-изготовитель ТС (страна)", [("manufacturer", t["manufacturer"])]),
        ("17. Одобрение типа ТС №", f"ТС RU E-RU.МТ02.00{rng.randint(100, 999)}.Р{rng.randint(1, 9)}"),
        ("18. Страна вывоза ТС", "—"),
        ("19. Серия, № ТД, ТПО", "—"),
        ("20. Таможенные ограничения", "НЕ УСТАНОВЛЕНЫ"),
        ("21. Наименование (ф.и.о.) собственника ТС", owner),
        ("22. Адрес", [("region", t["region"]), (None, ", "), ("city", t["city"])]),
        ("", [("street_address", t["street_address"])]),
        ("23. Наименование организации, выдавшей паспорт", t["manufacturer"].split(" (")[0]),
        ("24. Дата выдачи паспорта", [("issue_date", t["issue_date"])]),
    ]
    x_label, x_value, y, step = 44, 470, 150, 52
    s.draw.rectangle((30, y - 14, w - 30, y - 14 + step * len(rows)), outline=(120, 160, 130), width=2)
    for label, parts in rows:
        if label:
            s.draw.line((30, y - 14, w - 30, y - 14), fill=(150, 185, 158), width=1)
        s.draw.line((x_value - 16, y - 14, x_value - 16, y - 14 + step), fill=(150, 185, 158), width=1)
        _wrap_label(s, label, (x_label, y - 4), lf, x_value - x_label - 30)
        if isinstance(parts, str):
            parts = [(None, parts)]
        s.values_inline(parts, (x_value, y), vf, ink, w - 40 - x_value)
        y += step
    _signature(s, 140, y + 40)
    _stamp(s, 860, y + 60, 80, "ИЗГОТОВИТЕЛЬ · ОТК · ")
    return s, t


def _wrap_label(s: Sheet, text: str, xy, fnt, width: int):
    """Длинные подписи переносим на две строки."""
    if not text:
        return
    words, lines, cur = text.split(), [], ""
    for wd in words:
        test = (cur + " " + wd).strip()
        if s.draw.textlength(test, font=fnt) > width and cur:
            lines.append(cur)
            cur = wd
        else:
            cur = test
    lines.append(cur)
    x, y = xy
    for i, line in enumerate(lines[:2]):
        s.text((x, y + i * 18), line, fnt, fill=(70, 95, 78))


# ---------------------------------------------------------------- ЭПТС

EPTS_SIZE = (1100, 1500)


def _fake_qr(s: Sheet, x: int, y: int, size: int):
    """Квадрат из случайных клеток, похожий на QR-код (ничего не кодирует)."""
    n = 25
    cell = size // n
    r = s.rng
    for i in range(n):
        for j in range(n):
            corner = (i < 7 and j < 7) or (i < 7 and j >= n - 7) or (i >= n - 7 and j < 7)
            if corner:
                ii, jj = i % (n - 7) if i >= n - 7 else i, j % (n - 7) if j >= n - 7 else j
                on = ii in (0, 6) or jj in (0, 6) or (2 <= ii <= 4 and 2 <= jj <= 4)
            else:
                on = r.random() < 0.45
            if on:
                s.draw.rectangle((x + j * cell, y + i * cell, x + (j + 1) * cell - 1, y + (i + 1) * cell - 1),
                                 fill=(20, 20, 20))


def render_epts(v: Vehicle, rng: random.Random):
    s = Sheet(EPTS_SIZE, (252, 252, 250), rng)
    w, h = EPTS_SIZE
    t = doc_truth(v, EPTS)
    ink = (25, 25, 25)
    vf = font(rng.choice(VALUE_FONTS[EPTS]), 21)
    lf = font("LiberationSans-Regular.ttf", 17)
    s.text((60, 70), "ВЫПИСКА ИЗ ЭЛЕКТРОННОГО ПАСПОРТА", font("LiberationSans-Bold.ttf", 28), fill=(20, 20, 20))
    s.text((60, 106), "ТРАНСПОРТНОГО СРЕДСТВА", font("LiberationSans-Bold.ttf", 28), fill=(20, 20, 20))
    _fake_qr(s, w - 230, 50, 175)
    s.text((60, 170), "Номер электронного паспорта", lf, fill=(90, 90, 90))
    s.value("doc_number", (420, 166), t["doc_number"], font("LiberationSans-Bold.ttf", 24), ink)
    s.text((60, 210), "Статус электронного паспорта", lf, fill=(90, 90, 90))
    s.value("epts_status", (420, 206), t["epts_status"], vf, ink)
    s.text((60, 250), "Дата оформления", lf, fill=(90, 90, 90))
    s.value("issue_date", (420, 246), t["issue_date"], vf, ink)

    def table(title: str, rows, y: int) -> int:
        s.draw.rectangle((50, y, w - 50, y + 40), fill=(232, 236, 240))
        s.text((64, y + 9), title, font("LiberationSans-Bold.ttf", 19), fill=(30, 30, 30))
        y += 40
        for label, parts in rows:
            s.draw.rectangle((50, y, w - 50, y + 46), outline=(200, 205, 210), width=1)
            s.draw.line((470, y, 470, y + 46), fill=(200, 205, 210), width=1)
            s.text((64, y + 13), label, lf, fill=(70, 70, 70))
            if isinstance(parts, str):
                parts = [(None, parts)]
            s.values_inline(parts, (486, y + 10), vf, ink, w - 50 - 486 - 14)
            y += 46
        return y + 24

    y = table("Сведения о транспортном средстве", [
        ("Идентификационный номер (VIN)", [("vin", t["vin"])]),
        ("Марка, коммерческое наименование", [("make_model", t["make_model"])]),
        ("Категория", [("category", t["category"])]),
        ("Год изготовления", [("year", t["year"])]),
        ("Цвет", [("color", t["color"])]),
        ("Рабочий объем цилиндров, см³", [("engine_volume", t["engine_volume"])]),
        ("Максимальная мощность, кВт (л.с.)", [("power_kw", t["power_kw"]), (None, " ("), ("power_hp", t["power_hp"]), (None, ")")]),
        ("Тип двигателя", [("engine_type", t["engine_type"])]),
        ("Экологический класс", [("eco_class", t["eco_class"])]),
        ("Технически допустимая макс. масса, кг", [("max_mass", t["max_mass"])]),
        ("Изготовитель", [("manufacturer", t["manufacturer"])]),
    ], 300)
    y = table("Сведения о собственнике", [
        ("Фамилия", [("owner_last", t["owner_last"])]),
        ("Имя", [("owner_first", t["owner_first"])]),
        ("Отчество", [("owner_middle", t["owner_middle"])]),
        ("Субъект Российской Федерации", [("region", t["region"])]),
    ], y)
    s.text((60, y + 10), "Выписка сформирована автоматически и действительна без подписи.",
           font("LiberationSans-Regular.ttf", 15), fill=(110, 110, 110))
    return s, t


RENDERERS = {STS_FRONT: render_sts_front, STS_BACK: render_sts_back, PTS: render_pts, EPTS: render_epts}


def render(v: Vehicle, doc_type: str, seed: int) -> tuple[Image.Image, dict[str, str], dict[str, Box]]:
    """Рисует документ. Возвращает картинку, правильные значения полей и рамки полей."""
    rng = random.Random(seed)
    sheet, truth = RENDERERS[doc_type](v, rng)
    card = _watermark(sheet.img, rng)
    boxes = sheet.boxes
    if doc_type in (STS_FRONT, STS_BACK):
        card, dx, dy = _scan_bed(card, rng)
        boxes = _shift_boxes(boxes, dx, dy)
    return card, truth, boxes
