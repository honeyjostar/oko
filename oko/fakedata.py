"""Выдуманные, но правдоподобные данные автомобилей и владельцев.

Всё генерируется из seed: один и тот же seed всегда даёт тот же набор.
Реальных людей и машин здесь нет: ФИО собираются случайно, VIN и номера
тоже случайные, только соблюдают формат.
"""
from __future__ import annotations

import datetime as dt
import random
from dataclasses import asdict, dataclass

from faker import Faker

# Буквы, которые бывают в российских госномерах (совпадают по виду с латиницей)
PLATE_LETTERS = "АВЕКМНОРСТУХ"
# В VIN нет букв I, O, Q
VIN_ALPHABET = "ABCDEFGHJKLMNPRSTUVWXYZ0123456789"

# 10-й символ VIN — модельный год
VIN_YEAR = {2001 + i: str(i + 1) for i in range(9)}
VIN_YEAR.update(dict(zip(range(2010, 2031), "ABCDEFGHJKLMNPRSTVWXY")))

_VIN_VALUES = dict(zip("ABCDEFGH", range(1, 9)))
_VIN_VALUES.update(dict(zip("JKLMN", range(1, 6))))
_VIN_VALUES.update({"P": 7, "R": 9})
_VIN_VALUES.update(dict(zip("STUVWXYZ", range(2, 10))))
_VIN_VALUES.update({str(d): d for d in range(10)})
_VIN_WEIGHTS = [8, 7, 6, 5, 4, 3, 2, 10, 0, 9, 8, 7, 6, 5, 4, 3, 2]


def vin_check_digit(vin: str) -> str:
    """Контрольная цифра VIN (9-я позиция) по ISO 3779.
    Многие VIN для российского рынка её не соблюдают, поэтому это только подсказка."""
    total = sum(_VIN_VALUES.get(c, 0) * w for c, w in zip(vin, _VIN_WEIGHTS))
    r = total % 11
    return "X" if r == 10 else str(r)


@dataclass(frozen=True)
class CarModel:
    name: str
    wmi: str                    # первые 3 символа VIN (код производителя)
    types: tuple[str, ...]
    engines: tuple[tuple[int, int], ...]   # (объём см³, мощность л.с.)
    engine_prefix: str
    manufacturer: str
    curb: tuple[int, int]       # масса без нагрузки, кг
    years: tuple[int, int]


CATALOG: list[CarModel] = [
    CarModel("LADA GRANTA", "XTA", ("ЛЕГКОВОЙ СЕДАН", "ЛЕГКОВОЙ ХЭТЧБЕК", "ЛЕГКОВОЙ УНИВЕРСАЛ"),
             ((1596, 87), (1596, 98), (1596, 106)), "21116", "ПАО «АВТОВАЗ» (РОССИЯ)", (1110, 1185), (2012, 2024)),
    CarModel("LADA VESTA", "XTA", ("ЛЕГКОВОЙ СЕДАН", "ЛЕГКОВОЙ УНИВЕРСАЛ"),
             ((1596, 106), (1774, 122)), "21129", "ПАО «АВТОВАЗ» (РОССИЯ)", (1230, 1350), (2015, 2024)),
    CarModel("LADA LARGUS", "XTA", ("ЛЕГКОВОЙ УНИВЕРСАЛ",),
             ((1596, 90), (1596, 106)), "K4M", "ПАО «АВТОВАЗ» (РОССИЯ)", (1260, 1370), (2012, 2023)),
    CarModel("KIA RIO", "Z94", ("ЛЕГКОВОЙ СЕДАН", "ЛЕГКОВОЙ ХЭТЧБЕК"),
             ((1396, 100), (1591, 123)), "G4FA", "ООО «ХММР» (РОССИЯ)", (1110, 1220), (2011, 2022)),
    CarModel("HYUNDAI SOLARIS", "Z94", ("ЛЕГКОВОЙ СЕДАН", "ЛЕГКОВОЙ ХЭТЧБЕК"),
             ((1396, 100), (1591, 123)), "G4FC", "ООО «ХММР» (РОССИЯ)", (1100, 1210), (2011, 2022)),
    CarModel("HYUNDAI CRETA", "Z94", ("ЛЕГКОВОЙ УНИВЕРСАЛ",),
             ((1591, 123), (1999, 149)), "G4NA", "ООО «ХММР» (РОССИЯ)", (1280, 1390), (2016, 2022)),
    CarModel("TOYOTA CAMRY", "XW7", ("ЛЕГКОВОЙ СЕДАН",),
             ((1998, 150), (2494, 181), (2494, 200)), "2AR", "ООО «ТОЙОТА МОТОР МАНУФЭКЧУРИНГ РОССИЯ» (РОССИЯ)",
             (1470, 1590), (2012, 2022)),
    CarModel("TOYOTA RAV4", "XW7", ("ЛЕГКОВОЙ УНИВЕРСАЛ",),
             ((1987, 149), (2494, 199)), "M20A", "ООО «ТОЙОТА МОТОР МАНУФЭКЧУРИНГ РОССИЯ» (РОССИЯ)",
             (1540, 1680), (2016, 2022)),
    CarModel("VOLKSWAGEN POLO", "XW8", ("ЛЕГКОВОЙ СЕДАН", "ЛЕГКОВОЙ ХЭТЧБЕК"),
             ((1598, 90), (1598, 110), (1395, 125)), "CWV", "ООО «ФОЛЬКСВАГЕН ГРУП РУС» (РОССИЯ)",
             (1160, 1270), (2011, 2022)),
    CarModel("SKODA OCTAVIA", "XW8", ("ЛЕГКОВОЙ ХЭТЧБЕК", "ЛЕГКОВОЙ УНИВЕРСАЛ"),
             ((1395, 150), (1798, 180)), "CZD", "ООО «ФОЛЬКСВАГЕН ГРУП РУС» (РОССИЯ)", (1280, 1420), (2013, 2022)),
    CarModel("RENAULT LOGAN", "X7L", ("ЛЕГКОВОЙ СЕДАН",),
             ((1598, 82), (1598, 102), (1598, 113)), "K7M", "АО «РЕНО РОССИЯ» (РОССИЯ)", (1090, 1180), (2010, 2022)),
    CarModel("RENAULT DUSTER", "X7L", ("ЛЕГКОВОЙ УНИВЕРСАЛ",),
             ((1598, 114), (1998, 143)), "H4M", "АО «РЕНО РОССИЯ» (РОССИЯ)", (1270, 1420), (2012, 2022)),
    CarModel("NISSAN QASHQAI", "SJN", ("ЛЕГКОВОЙ УНИВЕРСАЛ",),
             ((1197, 115), (1997, 144)), "MR20", "NISSAN MANUFACTURING UK LTD (ВЕЛИКОБРИТАНИЯ)", (1340, 1480), (2014, 2022)),
    CarModel("CHERY TIGGO 7 PRO", "LVV", ("ЛЕГКОВОЙ УНИВЕРСАЛ",),
             ((1498, 147),), "SQRE4T15C", "CHERY AUTOMOBILE CO., LTD (КИТАЙ)", (1480, 1560), (2020, 2024)),
    CarModel("HAVAL JOLION", "XZG", ("ЛЕГКОВОЙ УНИВЕРСАЛ",),
             ((1497, 143),), "GW4G15K", "ООО «ХАВЕЙЛ МОТОР МАНУФЭКЧУРИНГ РУС» (РОССИЯ)", (1440, 1530), (2021, 2024)),
    CarModel("GEELY COOLRAY", "Y4K", ("ЛЕГКОВОЙ УНИВЕРСАЛ",),
             ((1477, 150),), "JLH-3G15TD", "ЗАО «БЕЛДЖИ» (БЕЛАРУСЬ)", (1340, 1415), (2020, 2024)),
    CarModel("BMW 320I", "WBA", ("ЛЕГКОВОЙ СЕДАН",),
             ((1998, 184),), "B48B20", "BMW AG (ГЕРМАНИЯ)", (1475, 1560), (2015, 2022)),
    CarModel("MERCEDES-BENZ E 200", "W1K", ("ЛЕГКОВОЙ СЕДАН",),
             ((1991, 197),), "M274", "MERCEDES-BENZ AG (ГЕРМАНИЯ)", (1605, 1700), (2016, 2022)),
    CarModel("MITSUBISHI OUTLANDER", "Z8T", ("ЛЕГКОВОЙ УНИВЕРСАЛ",),
             ((1998, 146), (2360, 167)), "4B11", "ООО «ПСМА РУС» (РОССИЯ)", (1440, 1560), (2013, 2022)),
    CarModel("FORD FOCUS", "X9F", ("ЛЕГКОВОЙ СЕДАН", "ЛЕГКОВОЙ ХЭТЧБЕК", "ЛЕГКОВОЙ УНИВЕРСАЛ"),
             ((1596, 105), (1596, 125)), "IQDB", "ООО «ФОРД СОЛЛЕРС ХОЛДИНГ» (РОССИЯ)", (1250, 1360), (2011, 2019)),
]

COLORS = ["БЕЛЫЙ", "ЧЕРНЫЙ", "СЕРЫЙ", "СЕРЕБРИСТЫЙ", "СИНИЙ", "КРАСНЫЙ", "КОРИЧНЕВЫЙ",
          "ЗЕЛЕНЫЙ", "БЕЖЕВЫЙ", "ОРАНЖЕВЫЙ", "ТЕМНО-СЕРЫЙ", "ГОЛУБОЙ"]
COLOR_WEIGHTS = [22, 18, 14, 14, 8, 6, 4, 3, 3, 2, 4, 2]

ECO = {2011: "ЧЕТВЕРТЫЙ", 2014: "ПЯТЫЙ"}

# Регион собственника, коды регионов на номерах, города
REGIONS: list[tuple[str, tuple[str, ...], tuple[str, ...]]] = [
    ("САРАТОВСКАЯ ОБЛ", ("64", "164"), ("Г. САРАТОВ", "Г. ЭНГЕЛЬС", "Г. БАЛАКОВО", "Г. ВОЛЬСК")),
    ("Г. МОСКВА", ("77", "97", "99", "177", "197", "199", "777", "797", "799", "977"), ("Г. МОСКВА",)),
    ("МОСКОВСКАЯ ОБЛ", ("50", "90", "150", "190", "750"), ("Г. ПОДОЛЬСК", "Г. ХИМКИ", "Г. МЫТИЩИ", "Г. БАЛАШИХА")),
    ("Г. САНКТ-ПЕТЕРБУРГ", ("78", "98", "178", "198"), ("Г. САНКТ-ПЕТЕРБУРГ",)),
    ("САМАРСКАЯ ОБЛ", ("63", "163", "763"), ("Г. САМАРА", "Г. ТОЛЬЯТТИ", "Г. СЫЗРАНЬ")),
    ("ВОЛГОГРАДСКАЯ ОБЛ", ("34", "134"), ("Г. ВОЛГОГРАД", "Г. ВОЛЖСКИЙ", "Г. КАМЫШИН")),
    ("РЕСП. ТАТАРСТАН", ("16", "116", "716"), ("Г. КАЗАНЬ", "Г. НАБЕРЕЖНЫЕ ЧЕЛНЫ", "Г. АЛЬМЕТЬЕВСК")),
    ("ПЕНЗЕНСКАЯ ОБЛ", ("58",), ("Г. ПЕНЗА", "Г. КУЗНЕЦК")),
    ("НИЖЕГОРОДСКАЯ ОБЛ", ("52", "152"), ("Г. НИЖНИЙ НОВГОРОД", "Г. ДЗЕРЖИНСК", "Г. АРЗАМАС")),
    ("СВЕРДЛОВСКАЯ ОБЛ", ("66", "96", "196"), ("Г. ЕКАТЕРИНБУРГ", "Г. НИЖНИЙ ТАГИЛ")),
    ("КРАСНОДАРСКИЙ КРАЙ", ("23", "93", "123", "193"), ("Г. КРАСНОДАР", "Г. СОЧИ", "Г. НОВОРОССИЙСК")),
    ("НОВОСИБИРСКАЯ ОБЛ", ("54", "154"), ("Г. НОВОСИБИРСК", "Г. БЕРДСК")),
    ("РОСТОВСКАЯ ОБЛ", ("61", "161", "761"), ("Г. РОСТОВ-НА-ДОНУ", "Г. ТАГАНРОГ", "Г. ШАХТЫ")),
    ("УЛЬЯНОВСКАЯ ОБЛ", ("73", "173"), ("Г. УЛЬЯНОВСК", "Г. ДИМИТРОВГРАД")),
]
REGION_WEIGHTS = [30, 10, 8, 6, 8, 7, 6, 5, 5, 4, 4, 3, 4, 4]

# Код региона по номеру → регион (для перекрёстной проверки номера и адреса)
PLATE_CODE_TO_REGION = {code: name for name, codes, _ in REGIONS for code in codes}


@dataclass
class Vehicle:
    """Одна машина со всеми данными, из которых рисуются её документы."""
    plate: str
    vin: str
    make_model: str
    vehicle_type: str
    category: str
    year: int
    engine_model: str
    chassis: str
    body: str
    color: str
    power_kw: int
    power_hp: int
    engine_volume: int
    engine_type: str
    eco_class: str
    max_mass: int
    curb_mass: int
    manufacturer: str
    owner_last: str
    owner_first: str
    owner_middle: str
    region: str
    city: str
    street_address: str
    gibdd_code: str
    sts_number: str
    sts_issue_date: str
    pts_kind: str            # "pts" — бумажный, "epts" — электронный
    pts_number: str
    pts_issue_date: str
    epts_status: str

    def as_dict(self) -> dict:
        return asdict(self)


def hp_to_kw(hp: int) -> int:
    return round(hp * 0.7355)


def _date(rng: random.Random, start: dt.date, end: dt.date) -> dt.date:
    days = max(0, (end - start).days)
    return start + dt.timedelta(days=rng.randint(0, days))


class VehicleFactory:
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.fake = Faker("ru_RU")
        self.fake.seed_instance(seed)

    def _vin(self, model: CarModel, year: int) -> str:
        r = self.rng
        body = model.wmi + "".join(r.choice(VIN_ALPHABET) for _ in range(5))
        tail = "".join(r.choice(VIN_ALPHABET) for _ in range(1)) + "".join(r.choice("0123456789") for _ in range(6))
        vin = body + "0" + VIN_YEAR.get(year, "A") + tail
        # примерно у 60% машин контрольная цифра посчитана, у остальных — нет, как в жизни
        check = vin_check_digit(vin) if r.random() < 0.6 else r.choice("0123456789")
        return vin[:8] + check + vin[9:]

    def _plate(self, region_codes: tuple[str, ...]) -> str:
        r = self.rng
        letters = [r.choice(PLATE_LETTERS) for _ in range(3)]
        digits = f"{r.randint(1, 999):03d}"
        return f"{letters[0]}{digits}{letters[1]}{letters[2]}{r.choice(region_codes)}"

    def make(self) -> Vehicle:
        r, f = self.rng, self.fake
        model = r.choice(CATALOG)
        year = r.randint(*model.years)
        volume, hp = r.choice(model.engines)
        curb = r.randint(*model.curb) // 5 * 5
        region, codes, cities = r.choices(REGIONS, weights=REGION_WEIGHTS)[0]
        female = r.random() < 0.38
        if female:
            last, first, middle = f.last_name_female(), f.first_name_female(), f.middle_name_female()
        else:
            last, first, middle = f.last_name_male(), f.first_name_male(), f.middle_name_male()
        street = f.street_title().upper().replace("Ё", "Е")
        street_kind = r.choice(["УЛ.", "УЛ.", "УЛ.", "ПР-КТ", "ПЕР.", "Ш."])
        flat = f", КВ. {r.randint(1, 240)}" if r.random() < 0.8 else ""
        street_address = f"{street_kind} {street}, Д. {r.randint(1, 180)}{flat}"

        vin = self._vin(model, year)
        produced = dt.date(year, r.randint(1, 12), r.randint(1, 28))
        today = dt.date(2026, 9, 1)
        sts_date = _date(r, max(produced, dt.date(year, 1, 1)), today)
        pts_kind = "epts" if year >= 2021 or (year >= 2019 and r.random() < 0.3) else "pts"
        pts_date = _date(r, produced, min(today, produced + dt.timedelta(days=120)))
        if pts_kind == "pts":
            pts_number = f"{r.randint(10, 99)} {r.choice(PLATE_LETTERS)}{r.choice(PLATE_LETTERS)} {r.randint(0, 999999):06d}"
        else:
            pts_number = f"{r.choice(['164', '163', '177', '178', '116', '152'])}3010{r.randint(0, 99999999):08d}"
        sts_number = f"{r.randint(10, 99)} {r.randint(10, 99)} {r.randint(0, 999999):06d}"
        engine_model = f"{model.engine_prefix}, {r.choice('ABCDEFGHJKLMNPRSTVWXYZ0123456789')}{r.randint(100000, 9999999)}"
        eco = "ПЯТЫЙ" if year >= 2014 else "ЧЕТВЕРТЫЙ"
        return Vehicle(
            plate=self._plate(codes),
            vin=vin,
            make_model=model.name,
            vehicle_type=r.choice(model.types),
            category="B",
            year=year,
            engine_model=engine_model,
            chassis="ОТСУТСТВУЕТ",
            body=vin,
            color=r.choices(COLORS, weights=COLOR_WEIGHTS)[0],
            power_kw=hp_to_kw(hp),
            power_hp=hp,
            engine_volume=volume,
            engine_type=r.choices(["БЕНЗИНОВЫЙ", "ДИЗЕЛЬНЫЙ"], weights=[94, 6])[0],
            eco_class=eco,
            max_mass=curb + r.randint(80, 110) * 5,
            curb_mass=curb,
            manufacturer=model.manufacturer,
            owner_last=last.upper().replace("Ё", "Е"),
            owner_first=first.upper().replace("Ё", "Е"),
            owner_middle=middle.upper().replace("Ё", "Е"),
            region=region,
            city=r.choice(cities),
            street_address=street_address,
            gibdd_code=f"{r.randint(1100000, 1199999)}",
            sts_number=sts_number,
            sts_issue_date=sts_date.strftime("%d.%m.%Y"),
            pts_kind=pts_kind,
            pts_number=pts_number,
            pts_issue_date=pts_date.strftime("%d.%m.%Y"),
            epts_status="ДЕЙСТВУЮЩИЙ" if r.random() < 0.93 else "ПОГАШЕН",
        )


def doc_truth(v: Vehicle, doc_type: str) -> dict[str, str]:
    """Правильные значения полей для документа данного вида (то, что должна найти модель)."""
    from .schema import DOC_FIELDS, EPTS, PTS, STS_BACK, STS_FRONT

    d = v.as_dict()
    d["year"] = str(v.year)
    for k in ("power_kw", "power_hp", "engine_volume", "max_mass", "curb_mass"):
        d[k] = str(d[k])
    if doc_type in (STS_FRONT, STS_BACK):
        d["doc_number"] = v.sts_number
        d["issue_date"] = v.sts_issue_date
    elif doc_type in (PTS, EPTS):
        d["doc_number"] = v.pts_number
        d["issue_date"] = v.pts_issue_date
    return {k: d[k] for k in DOC_FIELDS[doc_type]}
