"""OKO — демо-приложение.  Запуск:  streamlit run app.py"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageOps

from oko import distort
from oko.dataset import load, open_image
from oko.db import Store
from oko.engines import get_engine
from oko.normalize import checks, same
from oko.schema import DOC_FIELDS, DOC_TITLES, FIELD_BY_KEY
from oko.search import explain, parse_llm, search

ROOT = Path(__file__).parent
DATA = Path(os.environ.get("OKO_DATA", ROOT / "data" if (ROOT / "data" / "labels.jsonl").exists() else ROOT / "data_sample"))
RESULTS = Path(os.environ.get("OKO_RESULTS", ROOT / "results"))
DOCS = ROOT / "docs"
DB = Path(os.environ.get("OKO_DB", ROOT / "oko.db"))

st.set_page_config(page_title="OKO — анализ документов", page_icon="◉", layout="wide")

st.markdown("""
<style>
  .block-container {padding-top: 3.4rem; max-width: 1280px;}
  .oko-title {font-size: 2.1rem; font-weight: 700; letter-spacing: .02em; margin: 0;}
  .oko-sub {color: #4a5866; margin: .2rem 0 1rem;}
  .chip {display: inline-block; padding: 2px 10px; margin: 2px 4px 2px 0; border-radius: 999px;
         background: #e3e9f7; color: #1f3a8a; font-size: .85rem;}
  .chk {padding: 6px 10px; border-radius: 6px; margin-bottom: 4px; font-size: .9rem;}
  .chk-ok {background: #e8f5ec;} .chk-info {background: #eef2f7;}
  .chk-warn {background: #fff4dc;} .chk-error {background: #fde8ea;}
  .muted {color: #5b6875; font-size: .9rem;}
</style>""", unsafe_allow_html=True)

st.markdown('<div class="oko-title">◉ OKO</div><div class="oko-sub">Распознавание СТС, ПТС и ЭПТС, извлечение полей и поиск '
            'по базе на обычном языке. Всё работает локально — документы не покидают компьютер.</div>',
            unsafe_allow_html=True)


# ------------------------------------------------------------------ ресурсы

@st.cache_resource
def engine(name: str):
    return get_engine(name)


@st.cache_resource
def store() -> Store:
    s = Store(DB)
    if s.stats()["documents"] == 0:
        # пустая база — заполняем разметкой набора, чтобы было что искать
        items = load(DATA) if (DATA / "labels.jsonl").exists() else []
        if len({it.vehicle_id for it in items}) >= 100:
            for it in items:
                s.add_document(it.doc_type, it.fields, it.id, "разметка набора")
        else:   # в репозитории лежит только маленькая выборка — берём те же 300 машин прямо из генератора
            from oko.fakedata import VehicleFactory, doc_truth
            from oko.schema import EPTS, PTS, STS_BACK, STS_FRONT
            f = VehicleFactory(42)
            for i in range(300):
                v = f.make()
                for dt in (STS_FRONT, STS_BACK, EPTS if v.pts_kind == "epts" else PTS):
                    s.add_document(dt, doc_truth(v, dt), f"v{i:04d}_{dt}", "разметка набора")
    return s


@st.cache_data
def dataset():
    return load(DATA) if (DATA / "labels.jsonl").exists() else []


def vlm_ready() -> bool:
    return engine("vlm").available()[0]


@st.cache_data
def _records(engine_dir: str) -> dict:
    from oko.evaluate import load_records
    return {(r["id"], r["variant"]): r for r in load_records(RESULTS / engine_dir / "records.jsonl")}


def saved_result(eng_name: str, item_id: str, variant: str):
    """Результат из сохранённого прогона — чтобы демо работало, даже когда модель не запущена."""
    from oko.engines.base import Extraction
    dirs = sorted(p.name for p in RESULTS.glob("vlm*" if eng_name == "vlm" else "tesseract") if p.is_dir())
    for d in dirs:
        r = _records(d).get((item_id, variant))
        if r:
            return Extraction(r.get("pred_doc_type"), r.get("fields") or {}, r.get("seconds") or 0,
                              f"{d} (сохранённый прогон)", error=r.get("error"))
    return None


CARD_TITLES = {"sts_number": "СТС (серия, номер)", "pts_number": "ПТС / ЭПТС, номер"}

EXAMPLES = ["черные тойоты", "мерседесы и бмв новее 2017", "из Саратова мощнее 140 л.с.",
            "хендай крета с ЭПТС", "машины Пановой", "серебристые солярисы с 2015 по 2019"]

ICON = {"ok": "✓", "info": "ℹ", "warn": "⚠", "error": "✕"}


def show_checks(found):
    for c in found:
        st.markdown(f'<div class="chk chk-{c.level}">{ICON[c.level]} {c.message}</div>', unsafe_allow_html=True)


def draw_boxes(img: Image.Image, boxes: dict, verdict: dict | None = None) -> Image.Image:
    """Рамки полей: без вердикта — синие (разметка), с вердиктом — зелёные/красные."""
    img = img.copy()
    d = ImageDraw.Draw(img)
    for k, b in boxes.items():
        if verdict is None:
            d.rectangle(b, outline=(47, 79, 176), width=2)
        elif k in verdict:
            d.rectangle(b, outline=(22, 150, 70) if verdict[k] else (210, 40, 50), width=3)
    return img


tab_rec, tab_search, tab_quality, tab_data, tab_about = st.tabs(
    ["Распознать", "Поиск", "Качество", "Набор данных", "О проекте"])

# ------------------------------------------------------------------ распознать
with tab_rec:
    items = dataset()
    left, right = st.columns([1.05, 1])
    with left:
        source = st.radio("Документ", ["Пример из набора", "Свой файл"], horizontal=True)
        truth = None
        variant = None
        if source == "Свой файл":
            up = st.file_uploader("Скан или фото документа", type=["jpg", "jpeg", "png", "webp"])
            img = Image.open(up).convert("RGB") if up else None
        elif items:
            test_items = [it for it in items if it.split == "test"] or items
            labels = {f"{it.id} · {DOC_TITLES[it.doc_type]}": it for it in test_items}
            pick = labels[st.selectbox("Пример", list(labels))]
            mode = st.segmented_control("Как снят", ["Скан", "Фото телефоном", "Своё искажение"], default="Скан")
            variant = {"Скан": "clean", "Фото телефоном": "photo", None: "clean"}.get(mode)
            if mode == "Фото телефоном":
                img = open_image(DATA, pick.photo)
            elif mode == "Своё искажение":
                c1, c2 = st.columns(2)
                kind = c1.selectbox("Искажение", distort.KINDS, format_func=lambda k: distort.TITLES[k])
                level = c2.slider("Сила", 0, 4, 2)
                img = distort.apply(open_image(DATA, pick.image), kind, level, seed=7)
            else:
                img = open_image(DATA, pick.image)
            truth = pick
        else:
            img = None
            st.info("Набор данных не найден. Создай его: `python -m oko generate --out data`")
        if img is not None:
            st.image(img, width="stretch")

    with right:
        eng_name = st.radio("Движок", ["vlm", "tesseract"], horizontal=True, index=0 if vlm_ready() else 1,
                            format_func=lambda n: "Qwen3-VL (основной)" if n == "vlm" else "Tesseract (базовая линия)")
        ok, why = engine(eng_name).available()
        saved = None
        if not ok:
            st.warning(why)
            if truth is not None and variant in ("clean", "photo"):
                saved = saved_result(eng_name, truth.id, variant)
        if saved is not None:
            go = st.button("Показать результат из замера", type="primary", width="stretch",
                           help="Движок сейчас не запущен, но этот документ есть в сохранённом прогоне results/")
        else:
            go = st.button("Распознать", type="primary", disabled=img is None or not ok, width="stretch")
        if go:
            if saved is not None:
                res = saved
            else:
                with st.spinner("Читаю документ…"):
                    res = engine(eng_name).extract(img)
            st.session_state["last"] = (res, truth.id if truth else None, source)
        if "last" in st.session_state:
            res, tid, src = st.session_state["last"]
            tr = next((it for it in items if it.id == tid), None) if tid else None
            if res.error:
                st.error(res.error)
            dt = res.doc_type
            st.markdown(f"**{DOC_TITLES.get(dt, 'вид не определён')}** · {res.seconds:.1f} с · движок `{res.engine}`")
            rows, verdict = [], {}
            keys = list(tr.fields) if tr else list(res.fields)
            for k in keys:
                v = res.fields.get(k, "")
                row = {"Поле": FIELD_BY_KEY[k].title, "Распознано": v}
                if tr:
                    good = same(k, v, tr.fields[k])
                    verdict[k] = good
                    row[" "] = "✓" if good else "✕"
                    row["Должно быть"] = "" if good else tr.fields[k]
                rows.append(row)
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", height=35 * len(rows) + 38,
                         column_config={"Поле": st.column_config.TextColumn(width="medium"),
                                        "Распознано": st.column_config.TextColumn(width="medium"),
                                        " ": st.column_config.TextColumn(width=32),
                                        "Должно быть": st.column_config.TextColumn(width="medium")})
            if tr:
                n_ok = sum(verdict.values())
                st.markdown(f'<span class="muted">Верно {n_ok} из {len(verdict)} полей. '
                            f'Зелёные рамки на скане — верно, красные — ошибка.</span>', unsafe_allow_html=True)
                with st.expander("Где эти поля на документе"):
                    st.image(draw_boxes(open_image(DATA, tr.image), tr.boxes, verdict), width="stretch")
            st.markdown("**Проверки**")
            missing = [FIELD_BY_KEY[k].title for k in DOC_FIELDS.get(dt, []) if not res.fields.get(k)]
            if missing:
                st.markdown(f'<div class="chk chk-warn">⚠ Не прочитано полей: {len(missing)} — '
                            f'{", ".join(missing)}</div>', unsafe_allow_html=True)
            show_checks(checks(dt or "", res.fields))
            c1, c2 = st.columns(2)
            if c1.button("Сохранить в базу", width="stretch", disabled=not res.fields):
                vid = store().add_document(dt or "", res.fields, tid or "загрузка", res.engine)
                st.success(f"Сохранено в карточку автомобиля №{vid}")
            c2.download_button("Скачать JSON", json.dumps({"doc_type": dt, "fields": res.fields}, ensure_ascii=False, indent=2),
                               file_name="oko_result.json", width="stretch")

# ------------------------------------------------------------------ поиск
with tab_search:
    s = store()
    st.markdown(f'<span class="muted">В базе {s.stats()["vehicles"]} автомобилей и {s.stats()["documents"]} документов. '
                'Документы одной машины связаны по VIN, номеру СТС и номеру ПТС.</span>', unsafe_allow_html=True)
    cols = st.columns(3)
    for i, ex in enumerate(EXAMPLES):
        if cols[i % 3].button(ex, width="stretch"):
            st.session_state["q"] = ex
    q = st.text_input("Запрос", key="q", placeholder="например: чёрные тойоты из Самарской области")
    use_llm = st.toggle("Разбирать запрос моделью", value=False,
                        help="Правила работают всегда. Модель понимает более свободные формулировки, но нужен запущенный сервер.")
    if q:
        conds = None
        if use_llm:
            vlm = engine("vlm")
            try:
                conds = parse_llm(q, vlm.base_url, vlm.model)
            except Exception as e:  # noqa: BLE001
                st.warning(f"Модель недоступна, разбираю правилами ({e.__class__.__name__})")
        conds, rest, rows = search(s, q, conds)
        chips = "".join(f'<span class="chip">{t}</span>' for t in explain(conds)) or "—"
        st.markdown(f"Понял так: {chips}" + (f' <span class="muted">· не понял: {", ".join(rest)}</span>' if rest else ""),
                    unsafe_allow_html=True)
        st.markdown(f"**Найдено: {len(rows)}**")
        if rows:
            df = pd.DataFrame([{
                "№": r["id"], "Госномер": r["plate"], "Марка, модель": r["make_model"], "Год": r["year"],
                "Цвет": r["color"], "л.с.": r["power_hp"], "Владелец": " ".join(x for x in (r["owner_last"], r["owner_first"]) if x),
                "Город": r["city"], "Документы": r["doc_count"]} for r in rows])
            st.dataframe(df, hide_index=True, width="stretch")
            vid = st.selectbox("Открыть карточку", [r["id"] for r in rows],
                               format_func=lambda i: next(f"№{r['id']} · {r['make_model']} · {r['plate']}" for r in rows if r["id"] == i))
            v = s.vehicle(vid)
            c1, c2 = st.columns([1, 1])
            with c1:
                st.markdown("**Карточка автомобиля**")
                card = [{"Поле": CARD_TITLES.get(k) or FIELD_BY_KEY[k].title, "Значение": v[k]}
                        for k in v if k not in ("id", "updated") and v[k] not in (None, "")]
                st.dataframe(pd.DataFrame(card), hide_index=True, width="stretch", height=35 * len(card) + 38)
            with c2:
                st.markdown("**Документы**")
                for d in s.documents(vid):
                    with st.expander(f"{DOC_TITLES.get(d['doc_type'], d['doc_type'])} · {d['source']}"):
                        st.json(d["fields"], expanded=False)
                        show_checks([type("C", (), c)() for c in d["checks"]])

# ------------------------------------------------------------------ качество
with tab_quality:
    summary_path = DOCS / "summary.json"
    if not summary_path.exists():
        st.info("Отчёт ещё не собран: `python -m oko eval`, затем `python -m oko report`")
    else:
        summaries = json.loads(summary_path.read_text(encoding="utf-8"))
        from oko.report import VARIANT_TITLES, engine_label
        cols = st.columns(max(1, len(summaries) * 2))
        i = 0
        for e, s_ in summaries.items():
            for v in ("clean", "photo"):
                r = s_["variants"].get(v)
                if r:
                    cols[i].metric(f"{engine_label(e)} · {VARIANT_TITLES[v].lower()}", f"{r['field_accuracy'] * 100:.1f}%",
                                   help=f"Верных полей. Документов: {r['documents']}, {r['median_seconds']:.1f} с на документ")
                    i += 1
        for name in ("overview", "robustness", "fields-photo"):
            p = DOCS / "img" / f"{name}-light.png"
            if p.exists():
                st.image(str(p), width="stretch")
        st.markdown("Подробности и таблицы — в `docs/REPORT.md`.")

# ------------------------------------------------------------------ набор данных
with tab_data:
    items = dataset()
    if not items:
        st.info("Набор не найден.")
    else:
        n_veh = len({it.vehicle_id for it in items})
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Автомобилей", n_veh)
        c2.metric("Документов", len(items))
        c3.metric("Размеченных полей", sum(len(it.boxes) for it in items))
        c4.metric("В тестовой части", sum(it.split == "test" for it in items))
        if n_veh < 100:
            st.caption("Это выборка из репозитория. Полный набор (300 машин, 900 документов): "
                       "`python -m oko generate --out data`")
        st.markdown('<span class="muted">Все данные выдуманы, на каждом листе водяной знак «ОБРАЗЕЦ». '
                    'Разметка — `labels.jsonl` и `coco.json` (открывается в CVAT).</span>', unsafe_allow_html=True)
        it = st.selectbox("Документ", items, format_func=lambda i: f"{i.id} · {DOC_TITLES[i.doc_type]}")
        c1, c2 = st.columns(2)
        c1.image(draw_boxes(open_image(DATA, it.image), it.boxes),
                 caption=f"{it.id} · скан с разметкой", width="stretch")
        c2.image(open_image(DATA, it.photo), caption="он же «сфотографированный телефоном»", width="stretch")
        st.markdown("**Все искажения стресс-теста, сила 3**")
        base = open_image(DATA, it.image)
        for row in range(0, len(distort.KINDS), 4):
            for col, k in zip(st.columns(4), distort.KINDS[row:row + 4]):
                pic = ImageOps.pad(distort.apply(base, k, 3, seed=1), base.size, color=(236, 236, 232))
                col.image(pic, caption=distort.TITLES[k], width="stretch")

# ------------------------------------------------------------------ о проекте
with tab_about:
    st.markdown((ROOT / "docs" / "ABOUT.md").read_text(encoding="utf-8") if (ROOT / "docs" / "ABOUT.md").exists()
                else "См. README.md")
