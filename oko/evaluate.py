"""Оценка качества: прогон движка по набору и подсчёт метрик.

Варианты картинок:
    clean        — чистый скан
    photo        — «фото телефоном» (несколько мягких искажений вместе)
    <вид>:<уровень> — одно искажение заданной силы, например blur:3

Каждый результат сразу дописывается в results/<движок>/records.jsonl.
Если прогон прервать, при повторном запуске уже сделанное пропускается.
"""
from __future__ import annotations

import json
import re
import statistics
import zlib
from collections import defaultdict
from pathlib import Path

from PIL import Image

from . import distort
from .dataset import Item, fingerprint, load, open_image
from .engines.base import Engine
from .normalize import cer, same
from .schema import DOC_TYPES


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


def variant_image(data_dir: str | Path, item: Item, variant: str) -> Image.Image:
    if variant == "clean":
        return open_image(data_dir, item.image)
    if variant == "photo":
        return open_image(data_dir, item.photo)
    kind, level = variant.split(":")
    seed = zlib.crc32(f"{item.id}|{variant}".encode())
    return distort.apply(open_image(data_dir, item.image), kind, int(level), seed)


def records_path(results_dir: str | Path, engine_name: str) -> Path:
    return Path(results_dir) / safe_name(engine_name) / "records.jsonl"


def load_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


class DataMismatch(RuntimeError):
    pass


def check_meta(engine_dir: Path, data_dir: str | Path, write: bool = False) -> None:
    """Результаты движка привязаны к отпечатку набора. Другой набор — другая разметка, мешать нельзя."""
    fp = fingerprint(load(data_dir))
    meta_path = engine_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    if meta.get("data_fingerprint") and meta["data_fingerprint"] != fp:
        raise DataMismatch(f"Результаты в {engine_dir} посчитаны на другом наборе (отпечаток {meta['data_fingerprint']}, "
                           f"а у набора {data_dir} — {fp}). Пересоздай набор с тем же seed или удали эту папку.")
    if write and not meta:
        meta_path.write_text(json.dumps({"data_fingerprint": fp}, indent=2), encoding="utf-8")


def run(engine: Engine, data_dir: str | Path, items: list[Item], variants: list[str],
        results_dir: str | Path = "results", workers: int = 1, progress=None) -> list[dict]:
    path = records_path(results_dir, engine.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    check_meta(path.parent, data_dir, write=True)
    done = {(r["id"], r["variant"]) for r in load_records(path)}
    todo = [(it, v) for v in variants for it in items if (it.id, v) not in done]
    # Документы обрабатываются по одному; параметр workers оставлен, чтобы не менять вызовы.
    n = 0
    for it, v in todo:
        res = engine.extract(variant_image(data_dir, it, v))
        rec = {"id": it.id, "variant": v, "doc_type": it.doc_type, "pred_doc_type": res.doc_type,
               "fields": res.fields, "seconds": round(res.seconds, 3), "error": res.error}
        # Каждый результат сразу дописывается в файл: прерванный прогон можно продолжить.
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        n += 1
        if progress:
            progress(n, len(todo))
    return load_records(path)


def robustness_items(items: list[Item], per_type: int) -> list[Item]:
    """Для стресс-теста берём поровну документов каждого вида."""
    out = []
    for dt in DOC_TYPES:
        out += [it for it in items if it.doc_type == dt][:per_type]
    return out


# ------------------------------------------------------------------ метрики

def score(records: list[dict], truth: dict[str, Item]) -> dict:
    """Сводка метрик по всем записям одного движка."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        if r["id"] in truth:
            groups[r["variant"]].append(r)

    def summarize(recs: list[dict]) -> dict:
        fields_total = fields_ok = docs_ok = types_ok = errors = 0
        cers, secs = [], []
        per_field = defaultdict(lambda: [0, 0])
        per_type = defaultdict(lambda: [0, 0])
        for r in recs:
            it = truth[r["id"]]
            pred = r.get("fields") or {}
            all_ok = True
            for key, val in it.fields.items():
                ok = same(key, pred.get(key), val)
                fields_total += 1
                fields_ok += ok
                all_ok &= ok
                cers.append(cer(key, pred.get(key), val))
                per_field[key][0] += ok
                per_field[key][1] += 1
                per_type[it.doc_type][0] += ok
                per_type[it.doc_type][1] += 1
            docs_ok += all_ok
            types_ok += r.get("pred_doc_type") == it.doc_type
            errors += bool(r.get("error"))
            secs.append(r.get("seconds") or 0)
        n = len(recs)
        return {
            "documents": n,
            "field_accuracy": fields_ok / fields_total if fields_total else None,
            "document_accuracy": docs_ok / n if n else None,
            "doc_type_accuracy": types_ok / n if n else None,
            "mean_cer": statistics.fmean(cers) if cers else None,
            "median_seconds": statistics.median(secs) if secs else None,
            "errors": errors,
            "per_field": {k: v[0] / v[1] for k, v in sorted(per_field.items())},
            "per_doc_type": {k: v[0] / v[1] for k, v in per_type.items()},
        }

    out = {"variants": {v: summarize(rs) for v, rs in groups.items()}}
    out["robustness"] = robustness(records, truth)
    return out


def robustness(records: list[dict], truth: dict[str, Item], ids: set[str] | None = None) -> dict:
    """Кривые устойчивости: доля верных полей для каждого искажения и силы.
    Сила 0 — чистый скан тех же самых документов, поэтому кривые честно начинаются из одной точки."""
    by_variant: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in records:
        if r["id"] in truth and (ids is None or r["id"] in ids):
            by_variant[r["variant"]][r["id"]] = r

    def acc(recs) -> float | None:
        total = ok = 0
        for r in recs:
            for key, val in truth[r["id"]].fields.items():
                total += 1
                ok += same(key, (r.get("fields") or {}).get(key), val)
        return ok / total if total else None

    curves: dict[str, dict[int, float]] = defaultdict(dict)
    docs: dict[str, set[str]] = defaultdict(set)
    for v, recs in by_variant.items():
        if ":" in v:
            kind, level = v.split(":")
            curves[kind][int(level)] = acc(recs.values())
            docs[kind] |= set(recs)
    clean = by_variant.get("clean", {})
    for kind in curves:
        base = [clean[i] for i in docs[kind] if i in clean]
        if base:
            curves[kind][0] = acc(base)
    return {k: dict(sorted(v.items())) for k, v in curves.items()}


def summarize_engine(results_dir: str | Path, engine_dir: str, data_dir: str | Path) -> dict:
    check_meta(Path(results_dir) / engine_dir, data_dir)
    truth = {it.id: it for it in load(data_dir)}
    recs = load_records(Path(results_dir) / engine_dir / "records.jsonl")
    s = score(recs, truth)
    s["engine"] = engine_dir
    (Path(results_dir) / engine_dir / "summary.json").write_text(
        json.dumps(s, ensure_ascii=False, indent=2), encoding="utf-8")
    return s
