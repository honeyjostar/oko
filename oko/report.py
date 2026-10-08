"""Отчёт: графики и таблицы по результатам прогонов.

Каждый график рисуется в двух вариантах — для светлой и тёмной темы GitHub.
Цвета закреплены за движками и не меняются от графика к графику:
основная модель — синий, базовая линия Tesseract — оранжевый.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .distort import TITLES as DISTORT_TITLES  # noqa: E402
from .dataset import load  # noqa: E402
from .evaluate import load_records, robustness, summarize_engine  # noqa: E402
from .schema import DOC_TITLES, FIELD_BY_KEY  # noqa: E402

THEMES = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781", "grid": "#e1e0d9",
              "axis": "#c3c2b7", "vlm": "#2a78d6", "tesseract": "#eb6834"},
    "dark": {"surface": "#1a1a19", "ink": "#ffffff", "ink2": "#c3c2b7", "muted": "#898781", "grid": "#2c2c2a",
             "axis": "#383835", "vlm": "#3987e5", "tesseract": "#d95926"},
}
VARIANT_TITLES = {"clean": "Чистый скан", "photo": "Фото телефоном"}


def engine_label(engine_dir: str) -> str:
    if engine_dir.startswith("vlm"):
        model = engine_dir.split("_", 1)[1] if "_" in engine_dir else engine_dir
        m = re.search(r"qwen(\d(?:\.\d)?)[-_]?vl[-_:]?(\d+b)?", model, flags=re.I)
        if m:
            return f"Qwen{m.group(1)}-VL" + (f" {m.group(2).upper()}" if m.group(2) else "")
        return model
    if engine_dir == "tesseract":
        return "Tesseract 5 + правила"
    return engine_dir


def engine_color(engine_dir: str, theme: dict) -> str:
    return theme["vlm"] if engine_dir.startswith("vlm") else theme["tesseract"]


def _style(ax, t):
    ax.set_facecolor(t["surface"])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(t["axis"])
    ax.tick_params(colors=t["ink2"], labelsize=9, length=0)
    ax.grid(axis="x", color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)


def _fig(w, h, t):
    fig = plt.figure(figsize=(w, h), dpi=150)
    fig.patch.set_facecolor(t["surface"])
    return fig


def chart_overview(summaries: dict[str, dict], t: dict, path: Path):
    variants = [v for v in ("clean", "photo") if any(v in s["variants"] for s in summaries.values())]
    engines = list(summaries)
    fig = _fig(7.2, 1.0 + 0.9 * len(variants), t)
    ax = fig.add_subplot(111)
    _style(ax, t)
    bar_h = min(0.8 / max(1, len(engines)), 0.42)
    for ei, e in enumerate(engines):
        for vi, v in enumerate(variants):
            s = summaries[e]["variants"].get(v)
            if not s:
                continue
            y = vi + (ei - (len(engines) - 1) / 2) * bar_h
            val = s["field_accuracy"] * 100
            ax.barh(y, val, height=bar_h - 0.06, color=engine_color(e, t), label=engine_label(e) if vi == 0 else None)
            ax.text(val + 1, y, f"{val:.1f}%", va="center", ha="left", fontsize=9, color=t["ink"])
    ax.set_yticks(range(len(variants)))
    ax.set_yticklabels([VARIANT_TITLES[v] for v in variants], color=t["ink"], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(0, 108)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_title("Доля верно распознанных полей", loc="left", color=t["ink"], fontsize=12, pad=26)
    leg = ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=len(engines), frameon=False, fontsize=9,
                    handlelength=1.2, borderaxespad=0.2)
    for txt in leg.get_texts():
        txt.set_color(t["ink2"])
    fig.tight_layout()
    fig.savefig(path, facecolor=t["surface"])
    plt.close(fig)


def chart_robustness(summaries: dict[str, dict], t: dict, path: Path):
    kinds = [k for k in DISTORT_TITLES if any(k in s.get("robustness", {}) for s in summaries.values())]
    if not kinds:
        return False
    cols = 4
    rows = (len(kinds) + cols - 1) // cols
    fig = _fig(11, 2.35 * rows + 0.8, t)
    axes = fig.subplots(rows, cols, sharey=True, squeeze=False)
    for i, ax in enumerate(axes.flat):
        if i >= len(kinds):
            ax.set_visible(False)
            continue
        k = kinds[i]
        _style(ax, t)
        ax.grid(axis="y", color=t["grid"], linewidth=0.8)
        ax.grid(axis="x", visible=False)
        for e, s in summaries.items():
            curve = s.get("robustness", {}).get(k)
            if not curve:
                continue
            xs = sorted(int(x) for x in curve)
            ys = [curve[x] * 100 if x in curve else curve[str(x)] * 100 for x in xs]
            ax.plot(xs, ys, color=engine_color(e, t), linewidth=2, marker="o", markersize=4.5, label=engine_label(e))
        ax.set_title(DISTORT_TITLES[k], loc="left", color=t["ink"], fontsize=9.5)
        ax.set_ylim(0, 100)
        ax.set_xticks([0, 1, 2, 3, 4])
        ax.set_yticks([0, 50, 100])
        ax.set_yticklabels(["0%", "50%", "100%"])
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.suptitle("Устойчивость к искажениям: доля верных полей при силе искажения от 0 до 4",
                 x=0.01, ha="left", color=t["ink"], fontsize=12)
    leg = fig.legend(handles, labels, loc="upper right", ncol=len(labels), frameon=False, fontsize=9)
    for txt in leg.get_texts():
        txt.set_color(t["ink2"])
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, facecolor=t["surface"])
    plt.close(fig)
    return True


def chart_fields(summaries: dict[str, dict], t: dict, path: Path, variant: str):
    engines = [e for e in summaries if variant in summaries[e]["variants"]]
    if not engines:
        return False
    keys = sorted({k for e in engines for k in summaries[e]["variants"][variant]["per_field"]})
    best = {k: max(summaries[e]["variants"][variant]["per_field"].get(k, 0) for e in engines) for k in keys}
    keys.sort(key=lambda k: best[k])
    fig = _fig(7.2, 0.27 * len(keys) + 1.3, t)
    ax = fig.add_subplot(111)
    _style(ax, t)
    for yi, k in enumerate(keys):
        vals = [summaries[e]["variants"][variant]["per_field"].get(k) for e in engines]
        vals = [v * 100 for v in vals if v is not None]
        if len(vals) > 1:
            ax.plot([min(vals), max(vals)], [yi, yi], color=t["grid"], linewidth=2, zorder=1)
    for e in engines:
        pf = summaries[e]["variants"][variant]["per_field"]
        xs = [pf.get(k, float("nan")) * 100 for k in keys]
        ax.scatter(xs, range(len(keys)), s=34, color=engine_color(e, t), zorder=2, label=engine_label(e),
                   edgecolors=t["surface"], linewidths=1.2)
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels([FIELD_BY_KEY[k].title for k in keys], color=t["ink"], fontsize=8.5)
    ax.set_xlim(-2, 102)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_title(f"Точность по полям: {VARIANT_TITLES.get(variant, variant).lower()}", loc="left",
                 color=t["ink"], fontsize=12, pad=24)
    leg = ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=len(engines), frameon=False, fontsize=9,
                    handletextpad=0.3, borderaxespad=0.2)
    for txt in leg.get_texts():
        txt.set_color(t["ink2"])
    fig.tight_layout()
    fig.savefig(path, facecolor=t["surface"])
    plt.close(fig)
    return True


def _pct(v):
    return "—" if v is None or v is False else f"{v * 100:.1f}%"


def build(results_dir: str | Path = "results", data_dir: str | Path = "data", out_dir: str | Path = "docs") -> dict:
    results_dir, out_dir = Path(results_dir), Path(out_dir)
    img_dir = out_dir / "img"
    img_dir.mkdir(parents=True, exist_ok=True)
    engines = sorted((p.name for p in results_dir.iterdir() if (p / "records.jsonl").exists()),
                     key=lambda n: (not n.startswith("vlm"), n))
    summaries = {e: summarize_engine(results_dir, e, data_dir) for e in engines}

    # стресс-тест сравниваем только на документах, которые прошли все движки
    truth = {it.id: it for it in load(data_dir)}
    stress_ids = {}
    for e in engines:
        ids = {r["id"] for r in load_records(results_dir / e / "records.jsonl") if ":" in r["variant"]}
        if ids:
            stress_ids[e] = ids
    common = set.intersection(*stress_ids.values()) if stress_ids else set()
    for e in stress_ids:
        if len(stress_ids) > 1:
            summaries[e]["robustness"] = robustness(load_records(results_dir / e / "records.jsonl"), truth, common)
    n_stress = len(common)

    charts = {}
    for mode, t in THEMES.items():
        if summaries:
            chart_overview(summaries, t, img_dir / f"overview-{mode}.png")
            charts["overview"] = True
            charts["robustness"] = chart_robustness(summaries, t, img_dir / f"robustness-{mode}.png")
            charts["fields_photo"] = chart_fields(summaries, t, img_dir / f"fields-photo-{mode}.png", "photo")
            charts["fields_clean"] = chart_fields(summaries, t, img_dir / f"fields-clean-{mode}.png", "clean")

    lines = ["# Отчёт о качестве", "",
             "Все цифры получены прогоном на синтетическом наборе (тестовая часть). "
             "Файлы с сырыми результатами: `results/<движок>/records.jsonl`, сводки: `summary.json`.", ""]

    def pic(name: str, alt: str) -> str:
        return (f'<picture><source media="(prefers-color-scheme: dark)" srcset="img/{name}-dark.png">'
                f'<img src="img/{name}-light.png" alt="{alt}"></picture>')

    lines += ["## Итог", "", "| Движок | Вариант | Документов | Верных полей | Документов без единой ошибки | "
              "Вид документа определён | CER | Время на документ |", "|---|---|---|---|---|---|---|---|"]
    for e, s in summaries.items():
        for v in ("clean", "photo"):
            r = s["variants"].get(v)
            if r:
                lines.append(f"| {engine_label(e)} | {VARIANT_TITLES[v]} | {r['documents']} | {_pct(r['field_accuracy'])} | "
                             f"{_pct(r['document_accuracy'])} | {_pct(r['doc_type_accuracy'])} | "
                             f"{r['mean_cer']:.3f} | {r['median_seconds']:.1f} с |")
    lines += ["", "CER — доля неверных символов в значении поля (0 — идеально).", ""]
    if charts.get("overview"):
        lines += [pic("overview", "Доля верно распознанных полей"), ""]

    lines += ["## По видам документов", "", "| Движок | Вариант | " + " | ".join(DOC_TITLES.values()) + " |",
              "|---|---|" + "---|" * len(DOC_TITLES)]
    for e, s in summaries.items():
        for v in ("clean", "photo"):
            r = s["variants"].get(v)
            if r:
                cells = [_pct(r["per_doc_type"].get(dt)) for dt in DOC_TITLES]
                lines.append(f"| {engine_label(e)} | {VARIANT_TITLES[v]} | " + " | ".join(cells) + " |")
    lines.append("")

    if charts.get("robustness"):
        lines += ["## Устойчивость к искажениям", "",
                  f"Каждое искажение применялось отдельно, с силой от 1 до 4, к одним и тем же {n_stress} документам "
                  f"(поровну каждого вида). Уровень 0 — чистый скан этих же документов.", "",
                  pic("robustness", "Устойчивость к искажениям"), ""]
        lines += ["| Искажение | " + " | ".join(f"{engine_label(e)}, сила {lv}" for e in summaries for lv in (2, 4)) + " |",
                  "|---|" + "---|" * (2 * len(summaries))]
        for k, title in DISTORT_TITLES.items():
            vals = []
            for e, s in summaries.items():
                c = s.get("robustness", {}).get(k, {})
                for lv in (2, 4):
                    vals.append(_pct(c.get(lv, c.get(str(lv)))))
            if any(v != "—" for v in vals):
                lines.append(f"| {title} | " + " | ".join(vals) + " |")
        lines.append("")
    if charts.get("fields_photo"):
        lines += ["## Точность по полям", "", pic("fields-photo", "Точность по полям на фото"), ""]
    if charts.get("fields_clean"):
        lines += [pic("fields-clean", "Точность по полям на сканах"), ""]
    (out_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    update_readme(out_dir.parent / "README.md", summaries, charts, out_dir.name)
    return summaries


START, END = "<!-- results:start -->", "<!-- results:end -->"


def _docs_word(n: int) -> str:
    if n % 10 == 1 and n % 100 != 11:
        return "документ"
    if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        return "документа"
    return "документов"


def update_readme(readme: Path, summaries: dict, charts: dict, docs: str = "docs") -> bool:
    """Таблица результатов в README между метками обновляется при каждом `python -m oko report`."""
    if not readme.exists():
        return False
    text = readme.read_text(encoding="utf-8")
    if START not in text or END not in text:
        return False

    def pic(name: str, alt: str) -> str:
        return (f'<picture><source media="(prefers-color-scheme: dark)" srcset="{docs}/img/{name}-dark.png">'
                f'<img src="{docs}/img/{name}-light.png" alt="{alt}"></picture>')

    rows = ["| Движок | Скан: верных полей | Фото: верных полей | Фото: документов без ошибок | Время на одно фото |",
            "|---|---|---|---|---|"]
    for e, s in summaries.items():
        c, ph = s["variants"].get("clean"), s["variants"].get("photo")
        if not (c or ph):
            continue
        rows.append(f"| **{engine_label(e)}** | {_pct(c and c['field_accuracy'])} | {_pct(ph and ph['field_accuracy'])} | "
                    f"{_pct(ph and ph['document_accuracy'])} | "
                    f"{(ph or c)['median_seconds']:.1f} с |")
    n = max((s["variants"].get("clean") or {}).get("documents", 0) for s in summaries.values()) if summaries else 0
    block = [START, "", f"Тестовая часть набора: {n} {_docs_word(n)}, каждый — в двух вариантах: скан и фото телефоном.", ""]
    block += rows + [""]
    if not any(e.startswith("vlm") for e in summaries):
        block += ["> Замер основной модели Qwen3-VL 8B делается на компьютере с видеокартой: `scripts\\eval_vlm.bat`. "
                  "После прогона таблица и графики здесь обновятся сами (`python -m oko report`).", ""]
    if charts.get("overview"):
        block += [pic("overview", "Доля верно распознанных полей"), ""]
    if charts.get("robustness"):
        block += [pic("robustness", "Устойчивость к искажениям"), ""]
    block += [f"Все таблицы: [{docs}/REPORT.md]({docs}/REPORT.md). Сырые ответы движков: `results/<движок>/records.jsonl`.",
              "", END]
    a, b = text.index(START), text.index(END) + len(END)
    readme.write_text(text[:a] + "\n".join(block) + text[b:], encoding="utf-8")
    return True
