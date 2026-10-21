"""Đọc / ghi ảnh an toàn với đường dẫn Unicode trên Windows."""

from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np

from helmet_ppe.config import IMAGE_EXTENSIONS


class ImageReadError(ValueError):
    """Tệp không phải ảnh hợp lệ hoặc không đọc được."""


def natural_key(path: Path) -> list:
    """Khóa sắp xếp tự nhiên: 'anh (2)' đứng trước 'anh (10)'."""
    return [int(tok) if tok.isdigit() else tok.lower() for tok in re.split(r"(\d+)", path.name)]


def list_images(folder: str | Path, recursive: bool = False) -> list[Path]:
    """Liệt kê ảnh hợp lệ (JPG, JPEG, PNG, WEBP, BMP) trong thư mục."""
    folder = Path(folder)
    if not folder.is_dir():
        raise NotADirectoryError(f"Không tìm thấy thư mục ảnh: {folder}")
    pattern = "**/*" if recursive else "*"
    files = [p for p in folder.glob(pattern)
             if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    return sorted(files, key=natural_key)


def decode_image(data: bytes, name: str = "<bytes>") -> np.ndarray:
    """Giải mã ảnh từ bytes thành mảng BGR."""
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise ImageReadError(f"Không giải mã được ảnh: {name}")
    return image


def read_image(path: str | Path) -> np.ndarray:
    """Đọc ảnh BGR; dùng imdecode để hỗ trợ đường dẫn có dấu tiếng Việt."""
    path = Path(path)
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise ImageReadError(f"Không đọc được tệp {path}: {exc}") from exc
    return decode_image(data, str(path))


def write_image(path: str | Path, image: np.ndarray, quality: int = 92) -> None:
    """Ghi ảnh; định dạng theo đuôi tệp."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower() or ".jpg"
    params = [cv2.IMWRITE_JPEG_QUALITY, quality] if ext in {".jpg", ".jpeg"} else []
    ok, buffer = cv2.imencode(ext, image, params)
    if not ok:
        raise OSError(f"Không mã hóa được ảnh để ghi: {path}")
    buffer.tofile(str(path))
