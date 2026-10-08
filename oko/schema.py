"""Схема данных: какие бывают документы и какие поля в них есть.

Ключи полей одинаковые для всех документов. Так поле `vin` из СТС
и поле `vin` из ПТС потом сводятся в одну карточку автомобиля.
"""
from __future__ import annotations

from dataclasses import dataclass

# Виды документов
STS_FRONT = "sts_front"   # СТС, лицевая сторона: данные машины
STS_BACK = "sts_back"     # СТС, оборотная сторона: собственник
PTS = "pts"               # бумажный ПТС
EPTS = "epts"             # выписка из электронного ПТС

DOC_TYPES = [STS_FRONT, STS_BACK, PTS, EPTS]

DOC_TITLES = {
    STS_FRONT: "СТС, лицевая сторона",
    STS_BACK: "СТС, оборот",
    PTS: "ПТС (бумажный)",
    EPTS: "Выписка из ЭПТС",
}


@dataclass(frozen=True)
class Field:
    key: str
    title: str          # как поле называется у нас в интерфейсе
    kind: str = "text"  # text | code | int | date — от этого зависит сравнение
    hint: str = ""      # подсказка для нейросети: как выглядит значение


FIELDS: list[Field] = [
    Field("doc_number", "Серия и номер документа", "code",
          "СТС: 4 цифры серии + 6 цифр номера (99 21 123456); ПТС: 2 цифры, 2 буквы, 6 цифр (63 ОХ 123456); ЭПТС: 15 цифр"),
    Field("plate", "Регистрационный знак", "code", "госномер, например А123ВС164"),
    Field("vin", "VIN", "code", "17 символов латиницей и цифрами, без I, O, Q"),
    Field("make_model", "Марка, модель", "text", "например KIA RIO"),
    Field("vehicle_type", "Тип ТС", "text", "например ЛЕГКОВОЙ СЕДАН"),
    Field("category", "Категория", "text", "A, B, C, D, BE..."),
    Field("year", "Год выпуска", "int", "четыре цифры"),
    Field("engine_model", "Модель и номер двигателя", "text", "например 21129, 3456789"),
    Field("chassis", "Шасси (рама) №", "text", "номер или ОТСУТСТВУЕТ"),
    Field("body", "Кузов (кабина, прицеп) №", "code", "обычно совпадает с VIN"),
    Field("color", "Цвет", "text", "например БЕЛЫЙ"),
    Field("power_kw", "Мощность, кВт", "int", "целое число"),
    Field("power_hp", "Мощность, л.с.", "int", "целое число"),
    Field("engine_volume", "Рабочий объём, см³", "int", "целое число"),
    Field("engine_type", "Тип двигателя", "text", "например БЕНЗИНОВЫЙ"),
    Field("eco_class", "Экологический класс", "text", "например ПЯТЫЙ"),
    Field("max_mass", "Разрешённая макс. масса, кг", "int", "целое число"),
    Field("curb_mass", "Масса без нагрузки, кг", "int", "целое число"),
    Field("manufacturer", "Изготовитель", "text", "организация-изготовитель со страной"),
    Field("pts_number", "ПТС (серия, номер)", "code", "номер ПТС или ЭПТС, указанный в СТС"),
    Field("owner_last", "Фамилия собственника", "text", ""),
    Field("owner_first", "Имя собственника", "text", ""),
    Field("owner_middle", "Отчество собственника", "text", ""),
    Field("region", "Регион", "text", "например САРАТОВСКАЯ ОБЛ"),
    Field("city", "Населённый пункт", "text", "например Г. САРАТОВ"),
    Field("street_address", "Улица, дом, кв.", "text", "например УЛ. ЛЕНИНА, Д. 12, КВ. 45"),
    Field("gibdd_code", "Код подразделения ГИБДД", "code", "7 цифр"),
    Field("issue_date", "Дата выдачи", "date", "ДД.ММ.ГГГГ"),
    Field("epts_status", "Статус ЭПТС", "text", "например ДЕЙСТВУЮЩИЙ"),
]

FIELD_BY_KEY = {f.key: f for f in FIELDS}

# Какие поля есть в каждом виде документа
DOC_FIELDS: dict[str, list[str]] = {
    STS_FRONT: ["plate", "vin", "make_model", "vehicle_type", "category", "year", "chassis", "body",
                "color", "power_kw", "power_hp", "eco_class", "pts_number", "max_mass", "curb_mass",
                "doc_number"],
    STS_BACK: ["owner_last", "owner_first", "owner_middle", "region", "city", "street_address",
               "gibdd_code", "issue_date", "doc_number"],
    PTS: ["vin", "make_model", "vehicle_type", "category", "year", "engine_model", "chassis", "body",
          "color", "power_kw", "power_hp", "engine_volume", "engine_type", "eco_class", "max_mass",
          "curb_mass", "manufacturer", "owner_last", "owner_first", "owner_middle", "region", "city",
          "street_address", "issue_date", "doc_number"],
    EPTS: ["doc_number", "epts_status", "vin", "make_model", "category", "year", "color", "power_kw",
           "power_hp", "engine_volume", "engine_type", "eco_class", "max_mass", "manufacturer",
           "owner_last", "owner_first", "owner_middle", "region", "issue_date"],
}
