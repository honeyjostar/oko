import pytest

from oko import distort
from oko.engines.base import Engine, Extraction
from oko.evaluate import load_records, records_path, run, score, variant_image
from oko.dataset import open_image


def test_score_perfect_and_one_mistake(tiny_items):
    truth = {it.id: it for it in tiny_items}
    perfect = [{"id": it.id, "variant": "clean", "pred_doc_type": it.doc_type, "fields": dict(it.fields),
                "seconds": 1.0} for it in tiny_items]
    s = score(perfect, truth)["variants"]["clean"]
    assert s["field_accuracy"] == 1 and s["document_accuracy"] == 1 and s["doc_type_accuracy"] == 1
    assert s["mean_cer"] == 0

    spoiled = [dict(r, fields=dict(r["fields"])) for r in perfect]
    spoiled[0]["fields"]["doc_number"] = "0000000000"
    s = score(spoiled, truth)["variants"]["clean"]
    n_fields = sum(len(it.fields) for it in tiny_items)
    assert s["field_accuracy"] == pytest.approx((n_fields - 1) / n_fields)
    assert s["document_accuracy"] == pytest.approx((len(tiny_items) - 1) / len(tiny_items))


def test_robustness_curve_starts_at_clean(tiny_items):
    truth = {it.id: it for it in tiny_items}
    recs = []
    for it in tiny_items:
        recs.append({"id": it.id, "variant": "clean", "pred_doc_type": it.doc_type, "fields": dict(it.fields)})
        recs.append({"id": it.id, "variant": "blur:4", "pred_doc_type": it.doc_type, "fields": {}})
    rob = score(recs, truth)["robustness"]
    assert rob["blur"] == {0: 1.0, 4: 0.0}


def test_variants_are_reproducible(tiny_data, tiny_items):
    it = tiny_items[0]
    a = variant_image(tiny_data, it, "glare:3")
    b = variant_image(tiny_data, it, "glare:3")
    assert a.tobytes() == b.tobytes()
    assert variant_image(tiny_data, it, "clean").size == open_image(tiny_data, it.image).size


def test_all_distortions_run(tiny_data, tiny_items):
    img = open_image(tiny_data, tiny_items[0].image)
    for k in distort.KINDS:
        for lvl in (0, 4):
            out = distort.apply(img, k, lvl, seed=1)
            assert out.mode == "RGB" and min(out.size) > 100


class Truthful(Engine):
    name = "truthful"

    def __init__(self, items):
        self.items = list(items)
        self.calls = 0

    def available(self):
        return True, ""

    def extract(self, img):
        self.calls += 1
        it = self.items[0]
        return Extraction(it.doc_type, dict(it.fields), 0.01, self.name)


def test_run_is_resumable(tmp_path, tiny_data, tiny_items):
    eng = Truthful(tiny_items)
    items = tiny_items[:2]
    run(eng, tiny_data, items, ["clean"], tmp_path, workers=1)
    assert eng.calls == 2
    run(eng, tiny_data, items, ["clean", "photo"], tmp_path, workers=1)   # clean уже посчитан
    assert eng.calls == 4
    recs = load_records(records_path(tmp_path, eng.name))
    assert sorted((r["id"], r["variant"]) for r in recs) == sorted((it.id, v) for it in items for v in ("clean", "photo"))


def test_results_bound_to_dataset(tmp_path, tiny_data, tiny_items):
    from oko.evaluate import DataMismatch, check_meta
    eng = Truthful(tiny_items)
    run(eng, tiny_data, tiny_items[:1], ["clean"], tmp_path, workers=1)
    meta = tmp_path / "truthful" / "meta.json"
    assert meta.exists()
    meta.write_text('{"data_fingerprint": "000000000000"}', encoding="utf-8")
    with pytest.raises(DataMismatch):
        check_meta(tmp_path / "truthful", tiny_data)
