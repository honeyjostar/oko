"""Общий интерфейс движков извлечения."""
from __future__ import annotations

from dataclasses import dataclass, field

from PIL import Image


@dataclass
class Extraction:
    doc_type: str | None
    fields: dict[str, str]
    seconds: float
    engine: str
    raw: str = ""                 # сырой ответ (текст OCR или ответ модели) — для отладки
    error: str | None = None
    extra: dict = field(default_factory=dict)


class Engine:
    """Движок получает картинку документа и возвращает найденные поля."""
    name = "base"

    def extract(self, img: Image.Image) -> Extraction:  # pragma: no cover - интерфейс
        raise NotImplementedError

    def available(self) -> tuple[bool, str]:
        """Можно ли сейчас пользоваться движком, и если нет — почему."""
        return True, ""
