"""Phương pháp đối chứng "ngây thơ" để đo đóng góp của Association + hệ luật.

Luật đối chứng: người nào có tâm của *bất kỳ* hộp mũ nằm trong khung người →
``HELMET_DETECTED``; ngược lại → ``SUSPECTED_NO_HELMET`` (coi "không thấy mũ"
là vi phạm). Không dùng lớp ``no_helmet``, không ghép một-một, không có UNKNOWN.
Chạy trên đúng các detection đã lưu trong ``results.json`` của một lần batch, nên
khác biệt so với hệ thống chỉ đến từ bước suy luận.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from helmet_ppe import geometry
from helmet_ppe.config import HELMET, PERSON, Status
from helmet_ppe.rule_engine import summarize_image
from helmet_ppe.schemas import HeadEvidence, ImageResult


def _center_inside(point: tuple[float, float], box) -> bool:
    x, y = point
    return box[0] <= x <= box[2] and box[1] <= y <= box[3]


def naive_image_result(result: ImageResult) -> ImageResult:
    """Đối chứng trên cùng người sau hợp nhất và hộp sau lọc của pipeline.

    Giữ nguyên ID/khung người, dùng tâm mũ trong khung để kết luận. Một hộp mũ
    có thể nằm trong nhiều người. Không dùng no_helmet hay ngưỡng bằng chứng.
    API đọc JSON cũ bên dưới tiếp tục dùng danh sách người PPE trong tệp.
    """
    helmets = [d for d in result.detections if d.label == HELMET]
    persons = []
    used = set()
    for person in result.persons:
        matches = [d for d in helmets if _center_inside(geometry.center(d.bbox), person.bbox)]
        evidence = None
        reason = "Không có tâm hộp mũ trong khung người nên đối chứng gán nghi không đội mũ."
        if matches:
            strongest = max(matches, key=lambda d: (d.confidence, -d.det_id))
            evidence = HeadEvidence(strongest.det_id, strongest.confidence, 1.0, strongest.bbox)
            used.add(strongest.det_id)
            ids = ", ".join(f"D{d.det_id}" for d in matches)
            reason = f"Tâm hộp mũ {ids} nằm trong khung người nên đối chứng gán đội mũ."
        persons.append(replace(
            person, status=Status.HELMET_DETECTED if matches else Status.SUSPECTED_NO_HELMET,
            rule_id="BASELINE", reason=reason, helmet=evidence, no_helmet=None, head=None, flags=[],
        ))
    return replace(result, persons=persons, image_status=summarize_image(persons),
                   unmatched_helmet_ids=[d.det_id for d in helmets if d.det_id not in used],
                   unmatched_no_helmet_ids=[])


def naive_person_rows(results_json: Path) -> list[dict]:
    """Sinh các dòng dạng persons.csv theo luật đối chứng."""
    data = json.loads(Path(results_json).read_text(encoding="utf-8"))
    rows = []
    for image in data["images"]:
        dets = image["detections"]
        persons = [d for d in dets if d["class_name"].lower() == PERSON]
        helmet_centers = [geometry.center(d["bbox"]) for d in dets if d["class_name"].lower() == HELMET]
        for person_id, person in enumerate(persons, start=1):
            has_helmet = any(_center_inside(c, person["bbox"]) for c in helmet_centers)
            x1, y1, x2, y2 = person["bbox"]
            rows.append({
                "image_name": image["image_name"], "person_id": person_id,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "status": str(Status.HELMET_DETECTED if has_helmet else Status.SUSPECTED_NO_HELMET),
                "rule_id": "BASELINE", "reason": "",
            })
    return rows
