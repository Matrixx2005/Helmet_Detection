"""Vẽ kết quả lên ảnh: khung người theo trạng thái, mũ, no_helmet và tóm tắt."""

from __future__ import annotations

import cv2
import numpy as np

from helmet_ppe.config import HELMET, NO_HELMET, Status
from helmet_ppe.schemas import ImageResult

# Màu BGR
STATUS_COLORS: dict[Status, tuple[int, int, int]] = {
    Status.HELMET_DETECTED: (60, 180, 75),       # xanh lá
    Status.SUSPECTED_NO_HELMET: (40, 40, 220),   # đỏ
    Status.UNKNOWN: (0, 165, 255),               # cam
}
HELMET_COLOR = (230, 200, 0)       # xanh ngọc
NO_HELMET_COLOR = (200, 0, 200)    # tím
UNMATCHED_COLOR = (160, 160, 160)  # xám

# cv2.putText không hỗ trợ tiếng Việt có dấu → dùng nhãn ASCII ngắn.
STATUS_SHORT = {
    Status.HELMET_DETECTED: "HELMET",
    Status.SUSPECTED_NO_HELMET: "NO HELMET?",
    Status.UNKNOWN: "UNKNOWN",
}


def _scale(image: np.ndarray) -> tuple[float, int]:
    """Cỡ chữ và độ dày nét theo kích thước ảnh."""
    base = max(image.shape[:2]) / 1000.0
    return max(0.45, 0.6 * base), max(1, int(round(2 * base)))


def _label(image: np.ndarray, text: str, x: int, y: int, color: tuple[int, int, int],
           font_scale: float, thickness: int) -> None:
    """Nhãn chữ trắng trên nền màu, tự đẩy vào trong ảnh nếu tràn mép."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x = int(min(max(0, x), max(0, image.shape[1] - tw - 4)))
    y = int(max(th + baseline + 2, y))
    cv2.rectangle(image, (x, y - th - baseline - 4), (x + tw + 4, y), color, -1)
    cv2.putText(image, text, (x + 2, y - baseline - 2), font, font_scale,
                (255, 255, 255), thickness, cv2.LINE_AA)


def _box(image: np.ndarray, bbox, color, thickness: int) -> None:
    x1, y1, x2, y2 = (int(round(v)) for v in bbox)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)


def draw_result(image: np.ndarray, result: ImageResult) -> np.ndarray:
    """Trả về bản sao ảnh đã vẽ kết quả (không sửa ảnh gốc)."""
    canvas = image.copy()
    font_scale, thickness = _scale(canvas)
    small = font_scale * 0.8
    by_id = {det.det_id: det for det in result.detections}

    unmatched = set(result.unmatched_helmet_ids) | set(result.unmatched_no_helmet_ids)
    for det in result.detections:
        if det.label not in (HELMET, NO_HELMET):
            continue
        if det.det_id in unmatched:
            color = UNMATCHED_COLOR
        else:
            color = HELMET_COLOR if det.label == HELMET else NO_HELMET_COLOR
        _box(canvas, det.bbox, color, max(1, thickness - 1))
        tag = "helmet" if det.label == HELMET else "no_helmet"
        _label(canvas, f"{tag} {det.confidence:.2f}", int(det.bbox[0]), int(det.bbox[1]),
               color, small, max(1, thickness - 1))

    for person in result.persons:
        color = STATUS_COLORS[person.status]
        _box(canvas, person.bbox, color, thickness + 1)
        evidence = person.helmet if person.status == Status.HELMET_DETECTED else person.no_helmet
        conf = evidence.confidence if evidence else person.person_confidence
        text = f"P{person.person_id} {STATUS_SHORT[person.status]} {conf:.2f}"
        _label(canvas, text, int(person.bbox[0]), int(person.bbox[3]), color,
               font_scale, thickness)
        # Đường nối người → mũ được ghép giúp kiểm tra association bằng mắt.
        for item in (person.helmet, person.no_helmet):
            if item is not None and item.det_id in by_id:
                hx = int((item.bbox[0] + item.bbox[2]) / 2)
                hy = int(item.bbox[3])
                px = int((person.bbox[0] + person.bbox[2]) / 2)
                py = int(person.bbox[1] + 0.25 * (person.bbox[3] - person.bbox[1]))
                cv2.line(canvas, (hx, hy), (px, py), color, 1, cv2.LINE_AA)

    _draw_summary(canvas, result, font_scale, thickness)
    return canvas


def _draw_summary(canvas: np.ndarray, result: ImageResult, font_scale: float,
                  thickness: int) -> None:
    helmet_ok = result.count_status(Status.HELMET_DETECTED)
    suspected = result.count_status(Status.SUSPECTED_NO_HELMET)
    unknown = result.count_status(Status.UNKNOWN)
    if not result.persons:
        text = "NO PERSON DETECTED"
        color = (90, 90, 90)
    else:
        text = (f"Persons: {len(result.persons)} | Helmet: {helmet_ok} | "
                f"No helmet?: {suspected} | Unknown: {unknown}")
        color = STATUS_COLORS.get(result.image_status, (90, 90, 90))
    _label(canvas, text, 4, 4, color, font_scale, thickness)
