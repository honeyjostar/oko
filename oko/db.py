"""Хранилище: распознанные документы и собранные из них карточки автомобилей.

Документы одной машины связываются автоматически:
- лицевая сторона СТС и ПТС — по VIN;
- две стороны СТС — по номеру СТС (он напечатан на обеих);
- СТС и ПТС/ЭПТС — ещё и по номеру ПТС, который записан в СТС.
Так оборот СТС (там только владелец) попадает в ту же карточку, что и машина.
"""
from __future__ import annotations

import datetime as dt
import json
import sqlite3
from pathlib import Path

from .normalize import canon, checks
from .schema import EPTS, PTS, STS_BACK, STS_FRONT

VEHICLE_COLUMNS = {
    "vin": "TEXT", "plate": "TEXT", "sts_number": "TEXT", "pts_number": "TEXT",
    "make_model": "TEXT", "vehicle_type": "TEXT", "category": "TEXT", "year": "INTEGER", "color": "TEXT",
    "power_kw": "INTEGER", "power_hp": "INTEGER", "engine_volume": "INTEGER", "engine_type": "TEXT",
    "eco_class": "TEXT", "max_mass": "INTEGER", "curb_mass": "INTEGER", "manufacturer": "TEXT",
    "owner_last": "TEXT", "owner_first": "TEXT", "owner_middle": "TEXT", "region": "TEXT", "city": "TEXT",
    "street_address": "TEXT",
}
INT_COLUMNS = {k for k, t in VEHICLE_COLUMNS.items() if t == "INTEGER"}


class Store:
    def __init__(self, path: str | Path = "oko.db"):
        self.path = str(path)
        self.con = sqlite3.connect(self.path, check_same_thread=False)
        self.con.row_factory = sqlite3.Row
        cols = ", ".join(f"{k} {t}" for k, t in VEHICLE_COLUMNS.items())
        self.con.executescript(f"""
            CREATE TABLE IF NOT EXISTS vehicles (id INTEGER PRIMARY KEY, {cols}, updated TEXT);
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY, vehicle_id INTEGER REFERENCES vehicles(id),
                doc_type TEXT, source TEXT, engine TEXT, created TEXT, fields TEXT, checks TEXT);
            CREATE INDEX IF NOT EXISTS idx_v_vin ON vehicles(vin);
            CREATE INDEX IF NOT EXISTS idx_v_sts ON vehicles(sts_number);
            CREATE INDEX IF NOT EXISTS idx_v_pts ON vehicles(pts_number);
        """)
        self.con.commit()

    # ------------------------------------------------------------- запись
    @staticmethod
    def _vehicle_values(doc_type: str, fields: dict) -> dict:
        """Какие колонки карточки автомобиля заполняет этот документ."""
        vals = {}
        for k, v in fields.items():
            if k in VEHICLE_COLUMNS and v:
                vals[k] = v
        if "vin" in vals:
            vals["vin"] = canon("vin", vals["vin"])
        if "plate" in vals:
            vals["plate"] = canon("plate", vals["plate"])
        num = canon("doc_number", fields.get("doc_number"))
        if doc_type in (STS_FRONT, STS_BACK) and num:
            vals["sts_number"] = num
        if doc_type in (PTS, EPTS) and num:
            vals["pts_number"] = num
        if doc_type == STS_FRONT and fields.get("pts_number"):
            vals["pts_number"] = canon("pts_number", fields["pts_number"])
        for k in INT_COLUMNS & vals.keys():
            digits = "".join(c for c in str(vals[k]) if c.isdigit())
            vals[k] = int(digits) if digits else None
        return {k: v for k, v in vals.items() if v not in (None, "")}

    def _find(self, vals: dict) -> list[int]:
        ids: list[int] = []
        for col in ("vin", "sts_number", "pts_number"):
            if vals.get(col):
                for row in self.con.execute(f"SELECT id FROM vehicles WHERE {col} = ?", (vals[col],)):
                    if row["id"] not in ids:
                        ids.append(row["id"])
        return ids

    def add_document(self, doc_type: str, fields: dict, source: str = "", engine: str = "") -> int:
        """Сохраняет документ и привязывает его к карточке автомобиля. Возвращает id карточки."""
        now = dt.datetime.now().isoformat(timespec="seconds")
        vals = self._vehicle_values(doc_type, fields)
        ids = self._find(vals)
        if ids:
            vid = ids[0]
            for other in ids[1:]:          # документ связал две карточки — сливаем их
                self._merge(vid, other)
        else:
            vid = self.con.execute("INSERT INTO vehicles (updated) VALUES (?)", (now,)).lastrowid
        if vals:
            sets = ", ".join(f"{k} = ?" for k in vals)
            self.con.execute(f"UPDATE vehicles SET {sets}, updated = ? WHERE id = ?", (*vals.values(), now, vid))
        found = [c.__dict__ for c in checks(doc_type, fields)]
        self.con.execute(
            "INSERT INTO documents (vehicle_id, doc_type, source, engine, created, fields, checks) VALUES (?,?,?,?,?,?,?)",
            (vid, doc_type, source, engine, now, json.dumps(fields, ensure_ascii=False),
             json.dumps(found, ensure_ascii=False)))
        self.con.commit()
        return vid

    def _merge(self, keep: int, drop: int) -> None:
        a = dict(self.con.execute("SELECT * FROM vehicles WHERE id = ?", (keep,)).fetchone())
        b = dict(self.con.execute("SELECT * FROM vehicles WHERE id = ?", (drop,)).fetchone())
        fill = {k: b[k] for k in VEHICLE_COLUMNS if a.get(k) in (None, "") and b.get(k) not in (None, "")}
        if fill:
            sets = ", ".join(f"{k} = ?" for k in fill)
            self.con.execute(f"UPDATE vehicles SET {sets} WHERE id = ?", (*fill.values(), keep))
        self.con.execute("UPDATE documents SET vehicle_id = ? WHERE vehicle_id = ?", (keep, drop))
        self.con.execute("DELETE FROM vehicles WHERE id = ?", (drop,))

    # ------------------------------------------------------------- чтение
    def vehicle(self, vid: int) -> dict | None:
        row = self.con.execute("SELECT * FROM vehicles WHERE id = ?", (vid,)).fetchone()
        return dict(row) if row else None

    def documents(self, vid: int) -> list[dict]:
        rows = self.con.execute("SELECT * FROM documents WHERE vehicle_id = ? ORDER BY id", (vid,)).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["fields"] = json.loads(d["fields"])
            d["checks"] = json.loads(d["checks"])
            out.append(d)
        return out

    def stats(self) -> dict:
        q = lambda sql: self.con.execute(sql).fetchone()[0]  # noqa: E731
        return {"vehicles": q("SELECT COUNT(*) FROM vehicles"), "documents": q("SELECT COUNT(*) FROM documents")}

    def distinct(self, column: str) -> list[str]:
        if column not in VEHICLE_COLUMNS:
            raise ValueError(column)
        return [r[0] for r in self.con.execute(f"SELECT DISTINCT {column} FROM vehicles WHERE {column} IS NOT NULL")]

    def query(self, where: str, params: list, limit: int = 200) -> list[dict]:
        sql = ("SELECT v.*, (SELECT COUNT(*) FROM documents d WHERE d.vehicle_id = v.id) AS doc_count, "
               "(SELECT GROUP_CONCAT(doc_type, ',') FROM documents d WHERE d.vehicle_id = v.id) AS doc_types "
               f"FROM vehicles v WHERE {where} ORDER BY v.id LIMIT {int(limit)}")
        return [dict(r) for r in self.con.execute(sql, params)]

    def clear(self) -> None:
        self.con.executescript("DELETE FROM documents; DELETE FROM vehicles;")
        self.con.commit()
