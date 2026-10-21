"""Các phép tính hình học trên bounding box dạng (x1, y1, x2, y2) theo pixel."""

from __future__ import annotations

from typing import Sequence

BBox = tuple[float, float, float, float]


def to_bbox(values: Sequence[float]) -> BBox:
    """Chuẩn hóa một dãy 4 số thành BBox với x1 ≤ x2, y1 ≤ y2."""
    if len(values) != 4:
        raise ValueError(f"BBox cần đúng 4 giá trị, nhận được {len(values)}")
    x1, y1, x2, y2 = (float(v) for v in values)
    return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)


def width(box: BBox) -> float:
    return max(0.0, box[2] - box[0])


def height(box: BBox) -> float:
    return max(0.0, box[3] - box[1])


def area(box: BBox) -> float:
    return width(box) * height(box)


def center(box: BBox) -> tuple[float, float]:
    return (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0


def intersection_area(a: BBox, b: BBox) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def iou(a: BBox, b: BBox) -> float:
    """Intersection over Union của hai hộp."""
    inter = intersection_area(a, b)
    union = area(a) + area(b) - inter
    return inter / union if union > 0 else 0.0


def inside_ratio(inner: BBox, outer: BBox) -> float:
    """Tỉ lệ diện tích của ``inner`` nằm trong ``outer`` (0..1)."""
    inner_area = area(inner)
    return intersection_area(inner, outer) / inner_area if inner_area > 0 else 0.0


def expand(box: BBox, left: float = 0.0, top: float = 0.0,
           right: float = 0.0, bottom: float = 0.0) -> BBox:
    """Nới hộp theo tỉ lệ kích thước của chính nó ở từng cạnh."""
    w, h = width(box), height(box)
    return box[0] - left * w, box[1] - top * h, box[2] + right * w, box[3] + bottom * h

