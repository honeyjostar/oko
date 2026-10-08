import json
import re

from oko.dataset import load
from oko.fakedata import PLATE_CODE_TO_REGION, VIN_YEAR, VehicleFactory, doc_truth, vin_check_digit
from oko.normalize import PLATE_RE, VIN_RE, checks
from oko.render import render
from oko.schema import DOC_FIELDS, EPTS, PTS, STS_BACK, STS_FRONT


def test_vin_check_digit_reference():
    # классический пример из стандарта ISO 3779 / NHTSA
    assert vin_check_digit("1M8GDM9AXKP042788") == "X"


def test_factory_is_deterministic():
    a = [VehicleFactory(123).make().as_dict() for _ in range(1)]
    b = [VehicleFactory(123).make().as_dict() for _ in range(1)]
    assert a == b


def test_vehicle_values_are_consistent():
    f = VehicleFactory(5)
    for _ in range(60):
        v = f.make()
        assert VIN_RE.match(v.vin), v.vin
        assert v.vin[9] == VIN_YEAR[v.year]                  # 10-й символ VIN — год выпуска
        assert PLATE_RE.match(v.plate), v.plate
        assert v.plate[6:] in PLATE_CODE_TO_REGION
        assert abs(round(v.power_hp * 0.7355) - v.power_kw) <= 1
        assert v.curb_mass < v.max_mass
        assert re.fullmatch(r"\d{2} \d{2} \d{6}", v.sts_number)
        if v.pts_kind == "epts":
            assert re.fullmatch(r"\d{15}", v.pts_number)
        else:
            assert re.fullmatch(r"\d{2} [А-Я]{2} \d{6}", v.pts_number)


def test_truth_has_exactly_the_document_fields():
    v = VehicleFactory(9).make()
    for dt in (STS_FRONT, STS_BACK, PTS if v.pts_kind == "pts" else EPTS):
        assert list(doc_truth(v, dt)) == DOC_FIELDS[dt]


def test_clean_truth_passes_all_checks():
    f = VehicleFactory(11)
    for _ in range(30):
        v = f.make()
        for dt in (STS_FRONT, STS_BACK, EPTS if v.pts_kind == "epts" else PTS):
            bad = [c for c in checks(dt, doc_truth(v, dt)) if c.level in ("warn", "error")]
            assert not bad, (dt, [c.message for c in bad])


def test_render_boxes_inside_image():
    v = VehicleFactory(3).make()
    for dt in (STS_FRONT, STS_BACK, PTS, EPTS):
        img, truth, boxes = render(v, dt, seed=1)
        assert set(boxes) <= set(truth)
        assert len(boxes) >= len(truth) - 1
        for x0, y0, x1, y1 in boxes.values():
            assert 0 <= x0 < x1 <= img.width and 0 <= y0 < y1 <= img.height


def test_generated_dataset(tiny_data, tiny_items):
    assert len(tiny_items) == 4 * 3
    for it in tiny_items:
        assert (tiny_data / it.image).exists() and (tiny_data / it.photo).exists()
        assert list(it.fields) == DOC_FIELDS[it.doc_type]
    # одна машина — целиком в одной части набора
    split_of = {}
    for it in tiny_items:
        assert split_of.setdefault(it.vehicle_id, it.split) == it.split
    assert {it.split for it in load(tiny_data, "test")} == {"test"}


def test_coco_export(tiny_data, tiny_items):
    coco = json.loads((tiny_data / "coco.json").read_text(encoding="utf-8"))
    assert len(coco["images"]) == len(tiny_items)
    assert len(coco["annotations"]) == sum(len(it.boxes) for it in tiny_items)
    cat_ids = {c["id"] for c in coco["categories"]}
    assert all(a["category_id"] in cat_ids and a["bbox"][2] > 0 and a["bbox"][3] > 0 for a in coco["annotations"])


def test_dataset_is_reproducible(tiny_items):
    """Тот же seed — та же разметка на любом компьютере. Если тест упал после обновления Faker,
    результаты в results/ больше не сравнимы с новым набором."""
    from oko.dataset import fingerprint
    assert fingerprint(tiny_items) == "822de09f6280"
