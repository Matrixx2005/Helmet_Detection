"""Ghép mũ (helmet) và vùng đầu không mũ (no_helmet) với từng người.

Thuật toán (cho từng loại vật thể vùng đầu, độc lập nhau):

1. Với mỗi cặp (người, vật thể), tính vị trí tương đối của tâm vật thể trong
   khung người và loại bỏ cặp vi phạm các điều kiện cứng (gate):
   tâm nằm ngoài bề ngang người, nằm quá thấp (không còn là vùng đầu — ví dụ
   mũ cầm trên tay), kích thước không hợp lý, hoặc phần lớn nằm ngoài khung người.
2. Cặp hợp lệ được chấm điểm 0..1 bằng tổ hợp có trọng số của:
   độ gần vị trí đầu kỳ vọng theo chiều dọc, độ lệch ngang so với trục người
   và tỉ lệ diện tích vật thể nằm trong khung người.
3. Ghép tham lam một-một theo điểm giảm dần: mỗi người tối đa một mũ, mỗi mũ
   tối đa một người. Khi nhiều người đứng sát nhau, mũ được trao cho người có
   vùng đầu khớp nhất thay vì cho người được duyệt trước.
"""

from __future__ import annotations

from typing import Iterable

from helmet_ppe import geometry
from helmet_ppe.config import HELMET, NO_HELMET, PERSON, AssociationConfig
from helmet_ppe.schemas import (Association, AssociationResult, Candidate,
                                Detection, HeadEvidence)


def score_pair(person: Detection, obj: Detection,
               cfg: AssociationConfig) -> Candidate | None:
    """Chấm điểm việc ``obj`` (mũ / no_helmet) nằm trên đầu ``person``.

    Trả về ``None`` nếu cặp không thỏa điều kiện hình học tối thiểu.
    """
    pw, ph = geometry.width(person.bbox), geometry.height(person.bbox)
    if pw <= 0 or ph <= 0:
        return None

    cx, cy = geometry.center(obj.bbox)
    person_cx = (person.bbox[0] + person.bbox[2]) / 2.0
    rel_x = (cx - person_cx) / (pw / 2.0)
    rel_y = (cy - person.bbox[1]) / ph
    width_ratio = geometry.width(obj.bbox) / pw
    # Cho phép mũ nhô lên trên khung người một chút trước khi tính phần nằm trong.
    allowed = geometry.expand(person.bbox, top=abs(cfg.min_rel_y))
    inside = geometry.inside_ratio(obj.bbox, allowed)

    if not (cfg.min_rel_y <= rel_y <= cfg.max_rel_y):
        return None
    if abs(rel_x) > cfg.max_abs_rel_x:
        return None
    if not (cfg.min_width_ratio <= width_ratio <= cfg.max_width_ratio):
        return None
    if inside < cfg.min_inside_ratio:
        return None

    vertical = max(0.0, 1.0 - abs(rel_y - cfg.expected_rel_y) / cfg.vertical_tolerance)
    horizontal = max(0.0, 1.0 - min(abs(rel_x), 1.0))
    score = (cfg.weight_vertical * vertical
             + cfg.weight_horizontal * horizontal
             + cfg.weight_inside * inside)
    score /= cfg.weight_vertical + cfg.weight_horizontal + cfg.weight_inside

    return Candidate(
        person_det_id=person.det_id,
        object_det_id=obj.det_id,
        score=score,
        rel_x=rel_x,
        rel_y=rel_y,
        width_ratio=width_ratio,
        inside_ratio=inside,
    )


def match_one_to_one(persons: list[Detection], objects: list[Detection],
                     cfg: AssociationConfig) -> dict[int, Candidate]:
    """Ghép tham lam một-một theo điểm giảm dần; trả về person_det_id → Candidate."""
    candidates = [
        cand
        for person in persons
        for obj in objects
        if (cand := score_pair(person, obj, cfg)) is not None
    ]
    # Sắp xếp ổn định: điểm cao trước; hòa điểm thì ưu tiên id nhỏ (tái lập được).
    ranked = sorted(candidates,
                    key=lambda c: (-c.score, c.person_det_id, c.object_det_id))

    chosen: dict[int, Candidate] = {}
    used_objects: set[int] = set()
    for cand in ranked:
        if cand.score < cfg.min_match_score:
            break
        if cand.person_det_id in chosen or cand.object_det_id in used_objects:
            continue
        chosen[cand.person_det_id] = cand
        used_objects.add(cand.object_det_id)
    return chosen


def _evidence(cand: Candidate | None, by_id: dict[int, Detection]) -> HeadEvidence | None:
    if cand is None:
        return None
    obj = by_id[cand.object_det_id]
    return HeadEvidence(det_id=obj.det_id, confidence=obj.confidence,
                        match_score=cand.score, bbox=obj.bbox)


def _by_label(detections: Iterable[Detection], label: str) -> list[Detection]:
    return [det for det in detections if det.label == label]


def associate(detections: list[Detection],
              cfg: AssociationConfig | None = None) -> AssociationResult:
    """Ghép mũ và no_helmet với từng người trong ảnh.

    Người được sắp xếp từ trái sang phải để Person ID ổn định và dễ đối chiếu
    trên ảnh. Mũ / no_helmet không ghép được với ai được trả về riêng (ví dụ
    mũ đặt trên bàn, mũ cầm trên tay, hoặc người đội mũ bị YOLO bỏ sót).
    """
    cfg = cfg or AssociationConfig()
    persons = sorted(_by_label(detections, PERSON), key=lambda d: (d.bbox[0], d.bbox[1]))
    helmets = _by_label(detections, HELMET)
    no_helmets = _by_label(detections, NO_HELMET)
    by_id = {det.det_id: det for det in detections}

    helmet_match = match_one_to_one(persons, helmets, cfg)
    no_helmet_match = match_one_to_one(persons, no_helmets, cfg)

    associations = [
        Association(
            person=person,
            helmet=_evidence(helmet_match.get(person.det_id), by_id),
            no_helmet=_evidence(no_helmet_match.get(person.det_id), by_id),
        )
        for person in persons
    ]

    used_helmets = {c.object_det_id for c in helmet_match.values()}
    used_no_helmets = {c.object_det_id for c in no_helmet_match.values()}
    return AssociationResult(
        associations=associations,
        unmatched_helmet_ids=[d.det_id for d in helmets if d.det_id not in used_helmets],
        unmatched_no_helmet_ids=[d.det_id for d in no_helmets if d.det_id not in used_no_helmets],
    )
