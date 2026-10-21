"""Lọc detection theo ngưỡng và hợp nhất người từ hai mô hình.

* Mô hình PPE (``best.pt``) cho lớp ``Person`` với khung theo kiểu gán nhãn của
  Construction-PPE nhưng hay bỏ sót người bị che khuất trong đám đông.
* Mô hình pose (COCO) cho ``pose_person`` kèm 17 điểm khớp, phát hiện người
  trong đám đông tốt hơn và biết đầu nằm ở đâu.

Hợp nhất: mỗi người pose được ghép (tham lam theo IoU, một-một) với người PPE
trùng khung. Người đã ghép giữ khung PPE và nhận thêm điểm khớp của pose; người
chỉ có ở một mô hình được giữ nguyên.

Ngoài ra, ở ngưỡng confidence thấp mô hình PPE hay bắn ``no_helmet`` vào *khuôn
mặt ngay dưới một chiếc mũ* (cùng một cái đầu). :func:`suppress_faces_under_helmets`
loại các hộp ``no_helmet`` như vậy để chúng không bị ghép nhầm thành vi phạm.
"""

from __future__ import annotations

from dataclasses import replace

from helmet_ppe import geometry
from helmet_ppe.config import HELMET, NO_HELMET, PERSON, POSE_PERSON, PipelineConfig
from helmet_ppe.schemas import Detection


def filter_detections(detections: list[Detection], cfg: PipelineConfig) -> list[Detection]:
    """Giữ detection đạt ngưỡng confidence của lớp tương ứng."""
    thresholds = cfg.detector.class_thresholds
    return [
        det for det in detections
        if det.label in thresholds and det.confidence >= thresholds[det.label]
        and (cfg.detector.use_pose or det.label != POSE_PERSON)
    ]


def is_face_under_helmet(no_helmet: Detection, helmet: Detection, min_overlap_x: float = 0.5) -> bool:
    """True nếu hộp ``no_helmet`` chỉ là phần mặt nằm ngay dưới hộp ``helmet``.

    Điều kiện: hai hộp chồng nhau theo chiều ngang ≥ ``min_overlap_x`` bề rộng hộp
    hẹp hơn; tâm hộp đầu trần nằm thấp hơn đáy mũ (tức là phần mặt bên dưới, không
    phải hai hộp chồng khít lên nhau — trường hợp đó để luật R3 xử lý); khoảng hở
    dọc giữa đáy mũ và đỉnh hộp đầu trần không quá nửa chiều cao mũ; và mũ tin cậy
    hơn hẳn.
    """
    hx1, hy1, hx2, hy2 = helmet.bbox
    nx1, ny1, nx2, ny2 = no_helmet.bbox
    overlap_x = min(hx2, nx2) - max(hx1, nx1)
    narrower = min(hx2 - hx1, nx2 - nx1)
    if narrower <= 0 or overlap_x < min_overlap_x * narrower:
        return False
    if (ny1 + ny2) / 2 <= hy2:
        return False
    if ny1 - hy2 > 0.5 * (hy2 - hy1):
        return False
    return helmet.confidence > no_helmet.confidence


def suppress_faces_under_helmets(detections: list[Detection]) -> list[Detection]:
    """Bỏ các hộp ``no_helmet`` thực chất là khuôn mặt dưới một chiếc mũ."""
    helmets = [d for d in detections if d.label == HELMET]
    return [d for d in detections
            if d.label != NO_HELMET or not any(is_face_under_helmet(d, h) for h in helmets)]


def fuse_persons(detections: list[Detection], merge_iou: float) -> tuple[list[Detection], dict[int, int]]:
    """Hợp nhất người PPE và người pose.

    Returns:
        (danh sách người dùng cho bước ghép, ánh xạ det_id người → det_id pose đã gộp vào)
    """
    ppe = [d for d in detections if d.label == PERSON]
    pose = [d for d in detections if d.label == POSE_PERSON]
    pairs = sorted(((geometry.iou(p.bbox, q.bbox), p.det_id, q.det_id) for p in ppe for q in pose),
                   reverse=True)
    merged: dict[int, int] = {}
    used_pose: set[int] = set()
    for overlap, ppe_id, pose_id in pairs:
        if overlap < merge_iou:
            break
        if ppe_id in merged or pose_id in used_pose:
            continue
        merged[ppe_id] = pose_id
        used_pose.add(pose_id)

    by_id = {d.det_id: d for d in pose}
    persons = []
    for person in ppe:
        if person.det_id in merged:
            match = by_id[merged[person.det_id]]
            person = replace(person, source="ppe+pose", keypoints=match.keypoints,
                             confidence=max(person.confidence, match.confidence))
        persons.append(person)
    persons += [replace(q, class_name="Person", source="pose") for q in pose
                if q.det_id not in used_pose]
    return persons, merged
