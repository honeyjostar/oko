"""Движки извлечения полей."""
from __future__ import annotations

from .base import Engine, Extraction


def get_engine(name: str, **kwargs) -> Engine:
    """`tesseract` — базовая линия, `vlm` — основная модель (через LM Studio / Ollama)."""
    if name == "tesseract":
        from .tesseract import TesseractEngine
        return TesseractEngine(**kwargs)
    if name == "vlm":
        from .vlm import VLMEngine
        return VLMEngine(**kwargs)
    raise ValueError(f"Неизвестный движок «{name}». Есть: tesseract, vlm")


__all__ = ["Engine", "Extraction", "get_engine"]
