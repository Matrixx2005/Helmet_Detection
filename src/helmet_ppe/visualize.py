"""Vẽ kết quả lên ảnh: khung người theo trạng thái, mũ, no_helmet và tóm tắt."""

from __future__ import annotations

import cv2
import numpy as np

from helmet_ppe.config import HELMET, NO_HELMET, PERSON, POSE_PERSON, AssociationConfig, Status
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
        # no_helmet độ tin cậy thấp không thuộc ai không phải bằng chứng → không vẽ cho đỡ rối
        # (vẫn được ghi trong detections.csv với role = unmatched).
        if det.label == NO_HELMET and det.det_id in unmatched:
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
        # Điểm đầu ước lượng từ pose (nếu có) và đường nối tới mũ/đầu trần được ghép,
        # giúp kiểm tra bước association bằng mắt.
        if person.head is not None:
            anchor = (int(person.head.cx), int(person.head.cy))
            cv2.circle(canvas, anchor, max(3, thickness + 2), (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(canvas, anchor, max(3, thickness + 2), color, max(1, thickness - 1), cv2.LINE_AA)
        else:
            anchor = (int((person.bbox[0] + person.bbox[2]) / 2),
                      int(person.bbox[1] + 0.25 * (person.bbox[3] - person.bbox[1])))
        for item in (person.helmet, person.no_helmet):
            if item is not None and item.det_id in by_id:
                top = (int((item.bbox[0] + item.bbox[2]) / 2), int(item.bbox[3]))
                cv2.line(canvas, top, anchor, color, 1, cv2.LINE_AA)

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


def draw_analysis_stage(image: np.ndarray, result: ImageResult, stage: int,
                        cfg: AssociationConfig, person_id: int | None = None) -> np.ndarray:
    """Vẽ đầu ra mô hình, liên kết bằng chứng hoặc luật trên cùng hệ tọa độ.

    Các lớp detection giữ ID gốc D…, người sau hợp nhất dùng ID P….
    Chọn một người sẽ làm mờ nền và chỉ vẽ bằng chứng của người đó.
    """
    if stage == 0:
        return image.copy()
    canvas = image.copy()
    font, thickness = _scale(canvas)
    neutral = (180, 115, 45)
    if stage == 1:
        detections = result.raw_detections or result.detections
        colors = {PERSON: neutral, POSE_PERSON: neutral,
                  HELMET: HELMET_COLOR, NO_HELMET: NO_HELMET_COLOR}
        for det in detections:
            if det.label not in colors:
                continue
            color = colors[det.label]
            _box(canvas, det.bbox, color, thickness)
            _label(canvas, f"D{det.det_id} {det.label} {det.confidence:.2f}",
                   int(det.bbox[0]), int(det.bbox[1]), color, font * 0.8, thickness)
        return canvas

    people = [p for p in result.persons if person_id is None or p.person_id == person_id]
    if person_id is not None:
        canvas = (canvas * 0.35).astype(np.uint8)
        selected = next((p for p in people), None)
        if selected is not None:
            x1, y1, x2, y2 = person_view_bounds(result, selected.person_id)
            canvas[y1:y2, x1:x2] = image[y1:y2, x1:x2]
    for person in people:
        color = STATUS_COLORS[person.status] if stage == 3 else neutral
        _box(canvas, person.bbox, color, thickness + 1)
        tag = f"P{person.person_id} / D{person.person_det_id}"
        if stage == 3:
            tag += f" {person.rule_id} {STATUS_SHORT[person.status]}"
        _label(canvas, tag, int(person.bbox[0]), int(person.bbox[3]), color, font, thickness)
        if person.head is not None:
            anchor = (int(person.head.cx), int(person.head.cy))
            cv2.circle(canvas, anchor, max(3, thickness + 2), color, -1, cv2.LINE_AA)
        else:
            anchor = (int((person.bbox[0] + person.bbox[2]) / 2),
                      int(person.bbox[1] + cfg.expected_rel_y * (person.bbox[3] - person.bbox[1])))
            cv2.drawMarker(canvas, anchor, color, cv2.MARKER_CROSS,
                           max(8, thickness * 5), thickness)
        for name, evidence, item_color in (("helmet", person.helmet, HELMET_COLOR),
                                            ("no_helmet", person.no_helmet, NO_HELMET_COLOR)):
            if evidence is None:
                continue
            _box(canvas, evidence.bbox, item_color, thickness + 1)
            target = (int((evidence.bbox[0] + evidence.bbox[2]) / 2),
                      int((evidence.bbox[1] + evidence.bbox[3]) / 2))
            cv2.line(canvas, anchor, target, item_color, thickness, cv2.LINE_AA)
            _label(canvas, f"D{evidence.det_id} {name} match={evidence.match_score:.2f}",
                   int(evidence.bbox[0]), int(evidence.bbox[1]), item_color, font * 0.8, thickness)
    if person_id is None and stage == 2:
        unmatched = set(result.unmatched_helmet_ids + result.unmatched_no_helmet_ids)
        for det in result.detections:
            if det.det_id in unmatched:
                _box(canvas, det.bbox, UNMATCHED_COLOR, thickness)
                _label(canvas, f"D{det.det_id} unmatched", int(det.bbox[0]), int(det.bbox[1]),
                       UNMATCHED_COLOR, font * 0.8, thickness)
    return canvas


def person_view_bounds(result: ImageResult, person_id: int) -> tuple[int, int, int, int]:
    """Khung cắt người và bằng chứng, có lề và giới hạn trong ảnh."""
    person = next(p for p in result.persons if p.person_id == person_id)
    boxes = [person.bbox] + [e.bbox for e in (person.helmet, person.no_helmet) if e is not None]
    padding = max(16, int((person.bbox[2] - person.bbox[0]) * 0.15))
    return (max(0, int(min(b[0] for b in boxes)) - padding),
            max(0, int(min(b[1] for b in boxes)) - padding),
            min(result.width, int(max(b[2] for b in boxes)) + padding),
            min(result.height, int(max(b[3] for b in boxes)) + padding))
