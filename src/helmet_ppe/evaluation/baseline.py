"""Phương pháp đối chứng "ngây thơ" để đo đóng góp của Association + hệ luật.

Luật đối chứng: người nào có tâm của *bất kỳ* hộp mũ nằm trong khung người →
``HELMET_DETECTED``; ngược lại → ``SUSPECTED_NO_HELMET`` (coi "không thấy mũ"
là vi phạm). Không dùng lớp ``no_helmet``, không ghép một-một, không có UNKNOWN.
Chạy trên đúng các detection đã lưu trong ``results.json`` của một lần batch, nên
khác biệt so với hệ thống chỉ đến từ bước suy luận.
"""

from __future__ import annotations

import json
from pathlib import Path

from helmet_ppe import geometry
from helmet_ppe.config import HELMET, PERSON, Status


def _center_inside(point: tuple[float, float], box) -> bool:
    x, y = point
    return box[0] <= x <= box[2] and box[1] <= y <= box[3]


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
