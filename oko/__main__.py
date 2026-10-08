"""Командная строка OKO.

    python -m oko generate --out data --vehicles 300     набор синтетических документов
    python -m oko extract фото.jpg                        распознать документ
    python -m oko eval --engine vlm                       качество на тестовой части
    python -m oko robustness --engine vlm                 стресс-тест искажениями
    python -m oko report                                  графики и отчёт в docs/
    python -m oko db-fill                                 заполнить демо-базу
    python -m oko search "белые киа 2018 года"            поиск по базе
    python -m oko doctor                                  проверить, всё ли установлено
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("OMP_THREAD_LIMIT", "1")   # Tesseract быстрее в несколько потоков, чем в один многопоточный


def _progress(prefix: str):
    start = time.time()

    def cb(i: int, n: int):
        el = time.time() - start
        eta = el / i * (n - i) if i else 0
        sys.stdout.write(f"\r{prefix}: {i}/{n}  прошло {el / 60:.1f} мин, осталось ~{eta / 60:.1f} мин   ")
        sys.stdout.flush()
        if i == n:
            sys.stdout.write("\n")
    return cb


def _engine(args):
    from .engines import get_engine
    if args.engine == "vlm":
        kw = {}
        if args.url:
            kw["base_url"] = args.url
        if args.model:
            kw["model"] = args.model
        e = get_engine("vlm", **kw)
    else:
        e = get_engine(args.engine)
    ok, why = e.available()
    if not ok:
        sys.exit(f"Движок «{e.name}» недоступен: {why}")
    return e


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m oko", description="OKO — анализ документов на автомобиль")
    sub = p.add_subparsers(dest="cmd", required=True)

    def engine_args(sp):
        sp.add_argument("--engine", choices=["vlm", "tesseract"], default="vlm")
        sp.add_argument("--url", help="адрес OpenAI-совместимого сервера, по умолчанию http://127.0.0.1:1234/v1")
        sp.add_argument("--model", help="имя модели на сервере, по умолчанию qwen3-vl-8b")

    g = sub.add_parser("generate", help="создать синтетический набор")
    g.add_argument("--out", default="data")
    g.add_argument("--vehicles", type=int, default=300)
    g.add_argument("--seed", type=int, default=42)

    x = sub.add_parser("extract", help="распознать документ(ы)")
    engine_args(x)
    x.add_argument("images", nargs="+")

    ev = sub.add_parser("eval", help="оценка качества")
    engine_args(ev)
    ev.add_argument("--data", default="data")
    ev.add_argument("--split", default="test")
    ev.add_argument("--variants", default="clean,photo")
    ev.add_argument("--limit", type=int, default=0, help="взять только первые N документов")
    ev.add_argument("--workers", type=int, default=0, help="параллельных потоков (0 — авто)")
    ev.add_argument("--results", default="results")

    rb = sub.add_parser("robustness", help="стресс-тест искажениями")
    engine_args(rb)
    rb.add_argument("--data", default="data")
    rb.add_argument("--per-type", type=int, default=6, help="документов каждого вида")
    rb.add_argument("--levels", default="1,2,3,4")
    rb.add_argument("--kinds", default="all")
    rb.add_argument("--workers", type=int, default=0)
    rb.add_argument("--results", default="results")

    rp = sub.add_parser("report", help="графики и отчёт")
    rp.add_argument("--results", default="results")
    rp.add_argument("--data", default="data")
    rp.add_argument("--out", default="docs")

    df = sub.add_parser("db-fill", help="заполнить демо-базу документами набора")
    df.add_argument("--data", default="data")
    df.add_argument("--db", default="oko.db")
    df.add_argument("--from-results", help="брать поля из результатов движка (папка results/<движок>), а не из разметки")
    df.add_argument("--variant", default="photo")

    se = sub.add_parser("search", help="поиск по базе")
    se.add_argument("query")
    se.add_argument("--db", default="oko.db")

    sub.add_parser("doctor", help="проверить окружение")

    args = p.parse_args(argv)

    if args.cmd == "generate":
        from .dataset import generate
        items = generate(args.out, args.vehicles, args.seed, progress=_progress("Документы"))
        from .dataset import fingerprint
        print(f"Готово: {len(items)} документов в {args.out}, отпечаток набора {fingerprint(items)}")

    elif args.cmd == "extract":
        from PIL import Image
        from .normalize import checks
        e = _engine(args)
        for path in args.images:
            r = e.extract(Image.open(path))
            print(json.dumps({"file": path, "doc_type": r.doc_type, "fields": r.fields,
                              "seconds": round(r.seconds, 2), "error": r.error}, ensure_ascii=False, indent=2))
            for c in checks(r.doc_type or "", r.fields):
                print(f"  [{c.level}] {c.message}")

    elif args.cmd == "eval":
        from .dataset import load
        from .evaluate import run
        e = _engine(args)
        items = load(args.data, args.split)
        if args.limit:
            items = items[: args.limit]
        workers = args.workers or (2 if args.engine == "tesseract" else 1)
        run(e, args.data, items, args.variants.split(","), args.results, workers, _progress(e.name))
        _print_summary(args.results, args.data)

    elif args.cmd == "robustness":
        from . import distort
        from .dataset import load
        from .evaluate import robustness_items, run
        e = _engine(args)
        items = robustness_items(load(args.data, "test"), args.per_type)
        kinds = distort.KINDS if args.kinds == "all" else args.kinds.split(",")
        variants = ["clean"] + [f"{k}:{lvl}" for k in kinds for lvl in args.levels.split(",")]
        workers = args.workers or (2 if args.engine == "tesseract" else 1)
        run(e, args.data, items, variants, args.results, workers, _progress(e.name))
        _print_summary(args.results, args.data)

    elif args.cmd == "report":
        from .report import build
        build(args.results, args.data, args.out)
        print(f"Отчёт: {args.out}/REPORT.md")

    elif args.cmd == "db-fill":
        from pathlib import Path
        from .dataset import load
        from .db import Store
        from .evaluate import load_records
        st = Store(args.db)
        st.clear()
        items = {it.id: it for it in load(args.data)}
        if args.from_results:
            recs = [r for r in load_records(Path(args.from_results) / "records.jsonl") if r["variant"] == args.variant]
            for r in recs:
                st.add_document(r.get("pred_doc_type") or items[r["id"]].doc_type, r["fields"], r["id"],
                                Path(args.from_results).name)
        else:
            for it in items.values():
                st.add_document(it.doc_type, it.fields, it.id, "разметка")
        print(st.stats())

    elif args.cmd == "search":
        from .db import Store
        from .search import explain, search
        conds, rest, rows = search(Store(args.db), args.query)
        print("Понял так:", "; ".join(explain(conds)) or "—", "| не понял:", ", ".join(rest) or "—")
        for r in rows:
            print(f"  {r['plate'] or '—':10} {r['make_model'] or '—':22} {r['year'] or '—'}  {r['color'] or '—':12} "
                  f"{(r['owner_last'] or '—')} {(r['city'] or '')}")
        print(f"Найдено: {len(rows)}")

    elif args.cmd == "doctor":
        from .engines import get_engine
        for name in ("tesseract", "vlm"):
            ok, why = get_engine(name).available()
            print(f"{'✓' if ok else '✗'} {name}: {'готов' if ok else why}")


def _print_summary(results_dir, data_dir):
    from pathlib import Path
    from .evaluate import summarize_engine
    for d in sorted(Path(results_dir).iterdir()):
        if (d / "records.jsonl").exists():
            s = summarize_engine(results_dir, d.name, data_dir)
            for v, r in s["variants"].items():
                if ":" not in v:
                    print(f"{d.name:28} {v:8} документов {r['documents']:4}  верных полей {r['field_accuracy'] * 100:5.1f}%  "
                          f"документов без ошибок {r['document_accuracy'] * 100:5.1f}%  {r['median_seconds']:.1f} с/док")


if __name__ == "__main__":
    main()
