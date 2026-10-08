from oko.normalize import canon, cer, checks, clean_fields, detect_doc_type_by_fields, same
from oko.schema import EPTS, PTS, STS_BACK, STS_FRONT


def test_plate_letters_and_digits_by_position():
    assert canon("plate", "a123bc 164") == "А123ВС164"     # латиница → кириллица
    assert canon("plate", "0123ВС64") == "О123ВС64"        # ноль на месте буквы → «О»
    assert canon("plate", "АО2ЗВС164") == "А023ВС164"      # «О» и «З» на месте цифр → 0 и 3


def test_vin_lookalikes():
    assert canon("vin", "xta 21170O k1234567") == "XTA211700K1234567"
    assert canon("vin", "ХТА21170ОК1234567") == "XTA211700K1234567"   # кириллица


def test_series_numbers():
    assert canon("doc_number", "99 01 123456") == "9901123456"         # номер СТС — только цифры
    assert canon("pts_number", "63 OX 123456") == "63ОХ123456"         # серия ПТС — кириллица
    assert canon("pts_number", "63 0Х 123456") == "63ОХ123456"


def test_dates_ints_and_text():
    assert same("issue_date", "1.2.2020", "01.02.2020")
    assert same("power_hp", "106 л.с.", "106")
    assert same("category", "В", "B")
    assert same("city", "Г. САРАТОВ", "г.Саратов")
    assert same("owner_last", None, "") and not same("owner_last", "ИВАНОВ", "ИВАНОВА")


def test_cer():
    assert cer("vin", "XTA211700K1234567", "XTA211700K1234567") == 0
    assert 0 < cer("vin", "XTA211700K1234568", "XTA211700K1234567") < 0.1
    assert cer("city", "", "Г. САРАТОВ") == 1


def test_clean_fields():
    out = clean_fields({"plate": "a123bc164", "vin": "xta21170ok1234567", "make_model": "КIА RIO",
                        "owner_last": None, "color": "белый"})
    assert out == {"plate": "А123ВС164", "vin": "XTA211700K1234567", "make_model": "KIA RIO", "color": "БЕЛЫЙ"}


def test_checks_catch_inconsistencies():
    found = {(c.level, c.field) for c in checks(STS_FRONT, {
        "vin": "XTA21170OK123456",           # 16 символов
        "plate": "A12BC164",
        "power_kw": "200", "power_hp": "106",
        "doc_number": "99 01 12345",
    })}
    assert ("error", "vin") in found
    assert ("error", "plate") in found
    assert ("warn", "power_kw") in found
    assert ("error", "doc_number") in found


def test_year_vs_vin():
    found = {(c.level, c.field) for c in checks(PTS, {"vin": "XTA21170CK1234567", "year": "2015"})}
    assert ("warn", "year") in found              # 10-й символ VIN «K» — это 2019 год


def test_detect_doc_type():
    assert detect_doc_type_by_fields({"epts_status": "ДЕЙСТВУЮЩИЙ"}) == EPTS
    assert detect_doc_type_by_fields({"plate": "А123ВС164", "vin": "X"}) == STS_FRONT
    assert detect_doc_type_by_fields({"owner_last": "ИВАНОВ"}) == STS_BACK
    assert detect_doc_type_by_fields({"vin": "XTA211700K1234567"}) == PTS
