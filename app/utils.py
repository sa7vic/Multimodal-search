import io
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from config import SUPPORTED_EXTS


def load_image_from_bytes(data):
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
        img.load()
        return img
    except (UnidentifiedImageError, Exception) as exc:
        raise ValueError(f"cannot decode image: {exc}") from exc


def load_image_from_path(path):
    try:
        img = Image.open(path).convert("RGB")
        img.load()
        return img
    except Exception as exc:
        raise ValueError(f"cannot load {path}: {exc}") from exc


def is_supported_image(filename):
    return Path(filename).suffix.lower() in SUPPORTED_EXTS
