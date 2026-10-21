"""Ghép mũ (helmet) và vùng đầu không mũ (no_helmet) với từng người.

Mỗi cặp (người, vật thể vùng đầu) được chấm điểm 0..1 theo một trong hai cách:

* **Theo đầu (head)** — khi người có điểm khớp từ mô hình pose: vị trí đầu thật
  được ước lượng từ mắt/mũi/tai (``head.py``). Mũ phải nằm ngay trên đầu đó:
  tâm mũ phải rơi vào một elip quanh vị trí kỳ vọng (bán trục ngang
  ``head_tolerance_x·s``, dọc ``head_tolerance_y·s``, với ``s`` là kích thước
  đầu) và mũ không được nhỏ hơn hẳn đầu. Cách này loại được mũ của người
  đứng phía sau lọt vào khung người phía trước, và mũ cầm tay.
* **Theo khung (box)** — khi không có điểm khớp: dùng vị trí tương đối của tâm
  vật thể trong khung người (vùng đầu ước lượng ở phần trên khung), như sau:
  loại cặp vi phạm điều kiện cứng (quá thấp, ngoài bề ngang, kích thước bất
  hợp lý, phần lớn nằm ngoài khung), rồi chấm điểm theo độ gần vị trí đầu kỳ
  vọng, độ lệch ngang và tỉ lệ nằm trong khung.

Sau đó ghép tham lam một-một theo điểm giảm dần: mỗi người tối đa một mũ, mỗi
mũ tối đa một người. Mũ và no_helmet được ghép độc lập; hệ luật phân xử nếu một
người có cả hai.
"""

from __future__ import annotations

import math
from typing import Iterable

from helmet_ppe import geometry
from helmet_ppe.config import HELMET, NO_HELMET, PERSON, AssociationConfig
from helmet_ppe.head import estimate_head
from helmet_ppe.schemas import (Association, AssociationResult, Candidate,
                                Detection, HeadEstimate, HeadEvidence)


def score_pair(person: Detection, obj: Detection, cfg: AssociationConfig,
               head: HeadEstimate | None = None) -> Candidate | None:
    """Chấm điểm việc ``obj`` (mũ / no_helmet) nằm trên đầu ``person``.

    Dùng cách "theo đầu" nếu có ``head``, ngược lại "theo khung".
    Trả về ``None`` nếu cặp không thỏa điều kiện hình học tối thiểu.
    """
    if head is not None:
        return _score_by_head(person, obj, head, cfg)
    return _score_by_box(person, obj, cfg)


def _score_by_head(person: Detection, obj: Detection, head: HeadEstimate,
                   cfg: AssociationConfig) -> Candidate | None:
    is_helmet = obj.label == HELMET
    offset = cfg.helmet_head_offset if is_helmet else cfg.no_helmet_head_offset
    obj_width = geometry.width(obj.bbox)
    size_ratio = obj_width / head.size
    # Chuẩn hóa bằng kích thước đầu, nhưng không nhỏ hơn kích thước suy từ chính vật thể
    # (điểm khớp trên mặt nhỏ thường làm ước lượng đầu bị thiếu).
    scale = max(head.size, obj_width / (cfg.helmet_to_head_ratio if is_helmet
                                        else cfg.no_helmet_to_head_ratio))
    cx, cy = geometry.center(obj.bbox)
    du = (cx - head.cx) / (cfg.head_tolerance_x * scale)
    dv = (cy - (head.cy - offset * scale)) / (cfg.head_tolerance_y * scale)
    distance = math.hypot(du, dv)  # ≤ 1 nghĩa là nằm trong elip chấp nhận
    if distance > 1.0:
        return None
    # Chỉ chặn dưới: ước lượng đầu hay bị thiếu nên mũ "to hơn đầu nhiều lần" vẫn hợp lệ,
    # còn mũ nhỏ hơn hẳn đầu thì không thể là mũ đang đội trên đầu đó.
    if size_ratio < cfg.min_head_size_ratio:
        return None
    return Candidate(person.det_id, obj.det_id, 1.0 - 0.5 * distance ** 2, "head")


def _score_by_box(person: Detection, obj: Detection, cfg: AssociationConfig) -> Candidate | None:
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

    return Candidate(person.det_id, obj.det_id, score, "box")


def match_one_to_one(persons: list[Detection], objects: list[Detection],
                     cfg: AssociationConfig,
                     heads: dict[int, HeadEstimate | None] | None = None) -> dict[int, Candidate]:
    """Ghép tham lam một-một theo điểm giảm dần; trả về person_det_id → Candidate."""
    heads = heads or {}
    candidates = [
        cand
        for person in persons
        for obj in objects
        if (cand := score_pair(person, obj, cfg, heads.get(person.det_id))) is not None
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
    heads = {p.det_id: estimate_head(p.keypoints, cfg.keypoint_conf) for p in persons}

    helmet_match = match_one_to_one(persons, helmets, cfg, heads)
    no_helmet_match = match_one_to_one(persons, no_helmets, cfg, heads)

    associations = [
        Association(
            person=person,
            helmet=_evidence(helmet_match.get(person.det_id), by_id),
            no_helmet=_evidence(no_helmet_match.get(person.det_id), by_id),
            head=heads[person.det_id],
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
