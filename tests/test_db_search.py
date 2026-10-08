import pytest

from oko.db import Store
from oko.fakedata import VehicleFactory, doc_truth
from oko.search import Cond, parse_rules, search, to_sql
from oko.schema import EPTS, PTS, STS_BACK, STS_FRONT


def _docs(v):
    return [(dt, doc_truth(v, dt)) for dt in (STS_FRONT, STS_BACK, EPTS if v.pts_kind == "epts" else PTS)]


@pytest.fixture()
def store(tmp_path):
    return Store(tmp_path / "t.db")


def test_three_documents_make_one_vehicle(store):
    f = VehicleFactory(1)
    cars = [f.make() for _ in range(5)]
    for v in cars:
        for dt, fields in _docs(v):
            store.add_document(dt, fields, "test")
    assert store.stats() == {"vehicles": 5, "documents": 15}
    for row in store.query("1=1", []):
        assert row["doc_count"] == 3
        assert row["owner_last"] and row["vin"]       # владелец пришёл с оборота СТС, VIN — с лицевой


def test_back_side_first_then_merge(store):
    """Порядок не важен: оборот СТС без VIN, потом ПТС, потом лицевая — всё сходится в одну карточку."""
    v = VehicleFactory(2).make()
    docs = dict(_docs(v))
    pts_type = EPTS if v.pts_kind == "epts" else PTS
    store.add_document(STS_BACK, docs[STS_BACK])
    store.add_document(pts_type, docs[pts_type])
    assert store.stats()["vehicles"] == 2
    store.add_document(STS_FRONT, docs[STS_FRONT])     # связывает обе по номеру СТС и номеру ПТС
    assert store.stats() == {"vehicles": 1, "documents": 3}


def _filled(store, n=40, seed=3):
    f = VehicleFactory(seed)
    cars = [f.make() for _ in range(n)]
    for v in cars:
        for dt, fields in _docs(v):
            store.add_document(dt, fields, "test")
    return cars


def test_search_matches_python_filter(store):
    cars = _filled(store)
    _, _, rows = search(store, "белые машины новее 2015 года")
    expect = {c.vin for c in cars if c.color == "БЕЛЫЙ" and c.year > 2015}
    assert {r["vin"] for r in rows} == expect


def test_search_brand_word_forms_and_or(store):
    cars = _filled(store)
    _, _, rows = search(store, "тойоты и киа")
    expect = {c.vin for c in cars if c.make_model.startswith(("TOYOTA", "KIA"))}
    assert {r["vin"] for r in rows} == expect


def test_search_owner_case_and_city(store):
    cars = _filled(store)
    c = cars[0]
    conds, rest, rows = search(store, f"машины {c.owner_last.lower()}")
    assert any(x.column == "owner_last" for x in conds) and c.vin in {r["vin"] for r in rows}
    city = c.city.replace("Г. ", "").lower()
    conds, rest, rows = search(store, f"из {city}")
    assert c.vin in {r["vin"] for r in rows}


@pytest.mark.parametrize("q, expect", [
    ("белые киа 2018 года", {("year", "=", 2018), ("color", "=", "БЕЛЫЙ"), ("make_model", "like", "KIA")}),
    ("с ЭПТС 2021–2024", {("year", "between", (2021, 2024)), ("pts_number", "len", 15)}),
    ("мощнее 140 л.с. до 2015 г.", {("power_hp", ">", 140), ("year", "<", 2015)}),
    ("серебристые солярисы с 2015 по 2019", {("year", "between", (2015, 2019)), ("color", "=", "СЕРЕБРИСТЫЙ"),
                                             ("make_model", "like", "SOLARIS")}),
    ("дизельные 2.0 л", {("engine_type", "like", "ДИЗЕЛ"), ("engine_volume", "between", (1940, 2060))}),
    ("а123вс164", {("plate", "=", "А123ВС164")}),
    ("черные тойоты", {("color", "=", "ЧЕРНЫЙ"), ("make_model", "like", "TOYOTA")}),     # не CHERY
    ("хендай крета с ЭПТС", {("make_model", "like", "CRETA"), ("pts_number", "len", 15)}),  # марка + модель
    ("xta21170ok1234567", {("vin", "=", "XTA211700K1234567")}),
])
def test_parse_rules(q, expect):
    conds, rest = parse_rules(q)
    assert {(c.column, c.op, c.value) for c in conds} == expect
    assert rest == []


def test_sql_is_parameterized():
    sql, params = to_sql([Cond("owner_last", "=", "'; DROP TABLE vehicles; --")])
    assert "DROP" not in sql and params == ["'; DROP TABLE vehicles; --"]
