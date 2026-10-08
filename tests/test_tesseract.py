"""Сквозной тест базовой линии: генерируем скан, читаем Tesseract, сравниваем с разметкой."""
import pytest

from oko.dataset import open_image
from oko.engines import get_engine
from oko.normalize import same

eng = get_engine("tesseract")
ok, why = eng.available()
pytestmark = pytest.mark.skipif(not ok, reason=f"Tesseract не установлен: {why}")


def test_reads_clean_scans(tiny_data, tiny_items):
    total = good = types = 0
    for it in tiny_items[:6]:
        r = eng.extract(open_image(tiny_data, it.image))
        assert r.error is None
        types += r.doc_type == it.doc_type
        for k, v in it.fields.items():
            total += 1
            good += same(k, r.fields.get(k), v)
    assert types >= 5
    assert good / total > 0.6, f"верных полей {good}/{total}"
