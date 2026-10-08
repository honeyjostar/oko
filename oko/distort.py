"""Искажения документов для проверки устойчивости.

У каждого искажения пять уровней: 0 — без изменений, 4 — очень сильно.
Все функции принимают и возвращают картинку PIL в RGB.
"""
from __future__ import annotations

import math

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from .render import FONT_DIR

LEVELS = [0, 1, 2, 3, 4]

TITLES = {
    "blur": "Размытие (не в фокусе)",
    "motion": "Смаз (рука дрогнула)",
    "rotate": "Поворот",
    "perspective": "Перспектива (снято под углом)",
    "noise": "Шум матрицы",
    "jpeg": "Сильное сжатие JPEG",
    "lowres": "Низкое разрешение",
    "glare": "Блик",
    "shadow": "Тень",
    "lowlight": "Слабое освещение",
    "occlusion": "Перекрытия (печати, пальцы, стикеры)",
    "crumple": "Мятая бумага",
}
KINDS = list(TITLES)


def _np(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGB"), dtype=np.uint8)


def _pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")


def _bg_color(arr: np.ndarray) -> tuple[int, int, int]:
    corners = np.concatenate([arr[:8, :8].reshape(-1, 3), arr[-8:, -8:].reshape(-1, 3)])
    return tuple(int(c) for c in corners.mean(axis=0))


def blur(img, level, rng):
    sigma = [0, 1.0, 1.8, 2.8, 4.0][level]
    return img if sigma == 0 else img.filter(ImageFilter.GaussianBlur(sigma))


def motion(img, level, rng):
    k = [0, 7, 11, 17, 25][level]
    if k == 0:
        return img
    kernel = np.zeros((k, k), np.float32)
    kernel[k // 2, :] = 1.0
    angle = rng.uniform(0, 180)
    m = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), angle, 1.0)
    kernel = cv2.warpAffine(kernel, m, (k, k))
    kernel /= max(kernel.sum(), 1e-6)
    return _pil(cv2.filter2D(_np(img), -1, kernel))


def rotate(img, level, rng):
    a = [0, 3, 7, 12, 20][level]
    if a == 0:
        return img
    arr = _np(img)
    angle = a * rng.choice([-1, 1]) * rng.uniform(0.8, 1.0)
    return img.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=_bg_color(arr))


def perspective(img, level, rng):
    j = [0, 0.03, 0.06, 0.10, 0.15][level]
    if j == 0:
        return img
    arr = _np(img)
    h, w = arr.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    d = lambda s: rng.uniform(0.3, 1.0) * j * s
    dst = np.float32([[d(w), d(h)], [w - d(w), d(h)], [w - d(w), h - d(h)], [d(w), h - d(h)]])
    m = cv2.getPerspectiveTransform(src, dst)
    out = cv2.warpPerspective(arr, m, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=_bg_color(arr))
    return _pil(out)


def noise(img, level, rng):
    std = [0, 8, 16, 26, 38][level]
    if std == 0:
        return img
    arr = _np(img).astype(np.float32)
    lum = rng.normal(0, std, arr.shape[:2])[..., None]
    col = rng.normal(0, std * 0.35, arr.shape)
    return _pil(arr + lum + col)


def jpeg(img, level, rng):
    q = [95, 45, 22, 12, 6][level]
    if level == 0:
        return img
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(_np(img), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, q])
    return _pil(cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB))


def lowres(img, level, rng):
    s = [1, 0.7, 0.5, 0.36, 0.26][level]
    if s == 1:
        return img
    w, h = img.size
    small = img.resize((max(1, int(w * s)), max(1, int(h * s))), Image.BILINEAR)
    return small.resize((w, h), Image.BILINEAR)


def glare(img, level, rng):
    a = [0, 0.5, 0.7, 0.85, 0.97][level]
    if a == 0:
        return img
    arr = _np(img).astype(np.float32)
    h, w = arr.shape[:2]
    cx, cy = rng.uniform(0.3, 0.7) * w, rng.uniform(0.3, 0.7) * h
    rx, ry = rng.uniform(0.22, 0.38) * w, rng.uniform(0.16, 0.28) * h
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dist = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2
    mask = np.exp(-dist * 1.6) * a
    return _pil(arr * (1 - mask[..., None]) + 255 * mask[..., None])


def shadow(img, level, rng):
    a = [0, 0.25, 0.4, 0.55, 0.7][level]
    if a == 0:
        return img
    w, h = img.size
    mask = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(mask)
    # тень от руки или телефона: многоугольник от края листа
    side = rng.integers(0, 4)
    pts = {
        0: [(0, 0), (rng.uniform(0.3, 0.7) * w, 0), (rng.uniform(0.1, 0.5) * w, h), (0, h)],
        1: [(w, 0), (rng.uniform(0.3, 0.7) * w, 0), (rng.uniform(0.5, 0.9) * w, h), (w, h)],
        2: [(0, 0), (w, 0), (w, rng.uniform(0.2, 0.6) * h), (0, rng.uniform(0.4, 0.8) * h)],
        3: [(0, h), (w, h), (w, rng.uniform(0.4, 0.8) * h), (0, rng.uniform(0.2, 0.6) * h)],
    }[int(side)]
    d.polygon(pts, fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(max(w, h) * 0.03))
    m = np.asarray(mask, np.float32)[..., None] / 255 * a
    arr = _np(img).astype(np.float32)
    tint = np.array([0.92, 0.95, 1.0], np.float32)
    return _pil(arr * (1 - m) + arr * tint * (1 - a) * m)


def lowlight(img, level, rng):
    k = [1, 0.8, 0.62, 0.48, 0.36][level]
    if k == 1:
        return img
    arr = _np(img).astype(np.float32)
    mean = arr.mean()
    arr = (arr - mean) * (0.55 + 0.45 * k) + mean * k     # темнее и меньше контраста
    arr *= np.array([1.0, 0.94, 0.82], np.float32)        # тёплая лампа
    return noise(_pil(arr), max(0, level - 1), rng)


def occlusion(img, level, rng):
    n = [0, 1, 2, 3, 5][level]
    if n == 0:
        return img
    base = img.convert("RGBA")
    w, h = base.size
    for i in range(n):
        layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        kind = ["stamp", "finger", "sticker", "pen"][int(rng.integers(0, 4)) if i else 0]
        cx, cy = rng.uniform(0.25, 0.8) * w, rng.uniform(0.2, 0.85) * h
        if kind == "stamp":
            r = rng.uniform(0.07, 0.11) * w * (1 + level * 0.12)
            col = (40, 70, 170, int(rng.uniform(140, 200)))
            d.ellipse((cx - r, cy - r, cx + r, cy + r), outline=col, width=5)
            d.ellipse((cx - r * 0.78, cy - r * 0.78, cx + r * 0.78, cy + r * 0.78), outline=col, width=3)
            f = ImageFont.truetype(str(FONT_DIR / "DejaVuSansCondensed-Bold.ttf"), int(r * 0.22))
            d.text((cx, cy), "ОПЛАЧЕНО", font=f, fill=col, anchor="mm")
            layer = layer.rotate(rng.uniform(-30, 30), center=(cx, cy))
        elif kind == "finger":
            r = rng.uniform(0.06, 0.09) * w * (1 + level * 0.1)
            ex = rng.choice([0.0, 1.0]) * w
            d.ellipse((ex - r * 1.6, cy - r, ex + r * 1.6, cy + r * 1.1), fill=(214, 168, 140, 255))
            layer = layer.filter(ImageFilter.GaussianBlur(6))
        elif kind == "sticker":
            sw, sh = rng.uniform(0.12, 0.2) * w, rng.uniform(0.06, 0.1) * h
            d.rectangle((cx, cy, cx + sw, cy + sh), fill=(250, 232, 120, 245))
            f = ImageFont.truetype(str(FONT_DIR / "Carlito-Regular.ttf"), int(sh * 0.3))
            d.text((cx + 12, cy + 10), "позвонить в 15:00", font=f, fill=(40, 40, 120, 255))
            layer = layer.rotate(rng.uniform(-12, 12), center=(cx + sw / 2, cy + sh / 2))
        else:
            pts = [(cx + t * 0.3 * w, cy + 18 * math.sin(t * 9) + rng.uniform(-3, 3)) for t in np.linspace(0, 1, 40)]
            d.line(pts, fill=(20, 30, 120, 230), width=4)
        base.alpha_composite(layer)
    return base.convert("RGB")


def crumple(img, level, rng):
    alpha = [0, 5, 10, 16, 24][level]
    if alpha == 0:
        return img
    arr = _np(img)
    h, w = arr.shape[:2]
    sigma = 40
    dx = cv2.GaussianBlur(rng.uniform(-1, 1, (h, w)).astype(np.float32), (0, 0), sigma)
    dy = cv2.GaussianBlur(rng.uniform(-1, 1, (h, w)).astype(np.float32), (0, 0), sigma)
    dx *= alpha / (np.abs(dx).max() + 1e-6)
    dy *= alpha / (np.abs(dy).max() + 1e-6)
    xx, yy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    out = cv2.remap(arr, xx + dx, yy + dy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    # складки: светлые и тёмные полосы
    shade = np.ones((h, w), np.float32)
    for _ in range(level):
        x0 = rng.uniform(0, w)
        ang = rng.uniform(-0.4, 0.4)
        dist = (xx - x0 - (yy - h / 2) * ang)
        shade *= 1 - 0.10 * np.exp(-(dist / 6) ** 2) + 0.05 * np.exp(-((dist - 10) / 14) ** 2)
    return _pil(out.astype(np.float32) * shade[..., None])


FUNCS = {
    "blur": blur, "motion": motion, "rotate": rotate, "perspective": perspective, "noise": noise,
    "jpeg": jpeg, "lowres": lowres, "glare": glare, "shadow": shadow, "lowlight": lowlight,
    "occlusion": occlusion, "crumple": crumple,
}


def apply(img: Image.Image, kind: str, level: int, seed: int = 0) -> Image.Image:
    """Одно искажение заданной силы."""
    rng = np.random.default_rng(seed)
    return FUNCS[kind](img, int(level), rng)


def phone_photo(img: Image.Image, seed: int = 0) -> Image.Image:
    """Как будто документ сфотографировали телефоном: 2–4 мягких искажения вместе."""
    rng = np.random.default_rng(seed)
    kinds = list(rng.choice(["perspective", "rotate", "shadow", "glare", "lowlight", "noise", "blur",
                             "jpeg", "crumple"], size=int(rng.integers(2, 5)), replace=False))
    # геометрию применяем первой, сжатие — последним, как в жизни
    order = ["crumple", "perspective", "rotate", "shadow", "glare", "lowlight", "blur", "noise", "jpeg"]
    for k in sorted(kinds, key=order.index):
        img = FUNCS[k](img, int(rng.integers(1, 3)), rng)
    return img
