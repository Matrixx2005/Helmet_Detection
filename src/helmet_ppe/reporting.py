"""Chuyển ImageResult thành các dòng CSV, ghi tệp và tính thống kê tổng hợp.

Mỗi người chỉ xuất hiện đúng một lần trong ``persons.csv`` (khóa
``image_name + person_id``), mỗi detection đúng một lần trong
``detections.csv`` (khóa ``image_name + det_id``), nên các thống kê cộng dồn
không bị đếm trùng.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

from helmet_ppe.config import HELMET, NO_HELMET, PERSON, PERSON_STATUSES, POSE_PERSON, Status
from helmet_ppe.schemas import ImageResult

DETECTION_FIELDS = [
    "image_name", "det_id", "class_id", "class_name", "source", "confidence",
    "x1", "y1", "x2", "y2", "assigned_person_id", "role", "keypoints",
]
PERSON_FIELDS = [
    "image_name", "person_id", "status", "rule_id", "source", "person_confidence",
    "x1", "y1", "x2", "y2", "head_x1", "head_y1", "head_x2", "head_y2",
    "helmet_det_id", "helmet_confidence", "helmet_match_score", "helmet_evidence",
    "no_helmet_det_id", "no_helmet_confidence", "no_helmet_match_score", "no_helmet_evidence",
    "flags", "reason",
]
SUMMARY_FIELDS = [
    "image_name", "image_status", "width", "height",
    "num_detections", "num_persons", "num_helmets", "num_no_helmet",
    "helmet_detected", "suspected_no_helmet", "unknown",
    "unmatched_helmets", "unmatched_no_helmet", "inference_ms", "total_ms", "annotated_image",
]
ERROR_FIELDS = ["image_name", "image_path", "stage", "error_type", "message"]


def _r(value: float | None, digits: int = 4):
    return "" if value is None else round(float(value), digits)


def detection_rows(result: ImageResult) -> list[dict]:
    """Một dòng cho mỗi detection, kèm vai trò sau bước ghép."""
    roles: dict[int, tuple[int, str]] = {}
    for person in result.persons:
        roles[person.person_det_id] = (person.person_id, "person")
        if person.person_det_id in result.merged_pose_ids:
            roles[result.merged_pose_ids[person.person_det_id]] = (person.person_id, "pose_of_person")
        if person.helmet:
            roles[person.helmet.det_id] = (person.person_id, "helmet_of_person")
        if person.no_helmet:
            roles[person.no_helmet.det_id] = (person.person_id, "no_helmet_of_person")

    rows = []
    for det in result.detections:
        person_id, role = roles.get(det.det_id, ("", "unmatched"))
        if det.label not in (PERSON, HELMET, NO_HELMET, POSE_PERSON):
            role = "not_used"
        x1, y1, x2, y2 = det.bbox
        keypoints = ("" if det.keypoints is None else
                     json.dumps([[round(v, 1) for v in kp] for kp in det.keypoints],
                                separators=(",", ":")))
        rows.append({
            "image_name": result.image_name, "det_id": det.det_id, "class_id": det.class_id,
            "class_name": det.class_name, "source": det.source, "confidence": _r(det.confidence),
            "x1": _r(x1, 1), "y1": _r(y1, 1), "x2": _r(x2, 1), "y2": _r(y2, 1),
            "assigned_person_id": person_id, "role": role, "keypoints": keypoints,
        })
    return rows


def person_rows(result: ImageResult) -> list[dict]:
    """Một dòng cho mỗi người với trạng thái và bằng chứng."""
    rows = []
    for p in result.persons:
        x1, y1, x2, y2 = p.bbox
        h, n = p.helmet, p.no_helmet
        head = p.head_box or ("", "", "", "")
        rows.append({
            "image_name": result.image_name, "person_id": p.person_id, "status": str(p.status),
            "rule_id": p.rule_id, "source": p.source, "person_confidence": _r(p.person_confidence),
            "x1": _r(x1, 1), "y1": _r(y1, 1), "x2": _r(x2, 1), "y2": _r(y2, 1),
            **{f"head_{k}": ("" if v == "" else _r(v, 1))
               for k, v in zip(("x1", "y1", "x2", "y2"), head)},
            "helmet_det_id": h.det_id if h else "",
            "helmet_confidence": _r(h.confidence) if h else "",
            "helmet_match_score": _r(h.match_score) if h else "",
            "helmet_evidence": _r(h.strength) if h else "",
            "no_helmet_det_id": n.det_id if n else "",
            "no_helmet_confidence": _r(n.confidence) if n else "",
            "no_helmet_match_score": _r(n.match_score) if n else "",
            "no_helmet_evidence": _r(n.strength) if n else "",
            "flags": ";".join(p.flags), "reason": p.reason,
        })
    return rows


def summary_row(result: ImageResult, annotated_image: str = "") -> dict:
    """Một dòng thống kê cho mỗi ảnh."""
    return {
        "image_name": result.image_name, "image_status": str(result.image_status),
        "width": result.width, "height": result.height,
        "num_detections": len(result.detections), "num_persons": len(result.persons),
        "num_helmets": result.count(HELMET), "num_no_helmet": result.count(NO_HELMET),
        "helmet_detected": result.count_status(Status.HELMET_DETECTED),
        "suspected_no_helmet": result.count_status(Status.SUSPECTED_NO_HELMET),
        "unknown": result.count_status(Status.UNKNOWN),
        "unmatched_helmets": len(result.unmatched_helmet_ids),
        "unmatched_no_helmet": len(result.unmatched_no_helmet_ids),
        "inference_ms": _r(result.inference_ms, 2), "total_ms": _r(result.total_ms, 2),
        "annotated_image": annotated_image,
    }


def write_csv(path: Path, rows: Iterable[dict], fields: Sequence[str]) -> Path:
    """Ghi CSV UTF-8 có BOM để Excel trên Windows hiển thị đúng tiếng Việt."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)
    return path


def read_csv(path: Path) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def write_json(path: Path, data) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def compute_statistics(results: Sequence[ImageResult], num_failed: int,
                       wall_time_s: float) -> dict:
    """Thống kê tổng hợp của một lần chạy (dùng cho results.json và biểu đồ)."""
    person_status = Counter(str(p.status) for r in results for p in r.persons)
    image_status = Counter(str(r.image_status) for r in results)
    det_classes = Counter(d.class_name for r in results for d in r.detections)
    person_sources = Counter(p.source for r in results for p in r.persons)
    processed = len(results)
    persons_total = sum(person_status.values())
    unknown = person_status.get(str(Status.UNKNOWN), 0)
    return {
        "images_total": processed + num_failed,
        "images_processed": processed,
        "images_failed": num_failed,
        "persons_total": persons_total,
        "persons_by_source": dict(sorted(person_sources.items())),
        "helmet_objects_detected": det_classes.get("helmet", 0),
        "no_helmet_objects_detected": det_classes.get("no_helmet", 0),
        "detections_by_class": dict(sorted(det_classes.items())),
        "person_status_counts": {str(s): person_status.get(str(s), 0) for s in PERSON_STATUSES},
        "image_status_counts": {str(s): image_status.get(str(s), 0) for s in Status},
        "unknown_rate": round(unknown / persons_total, 4) if persons_total else None,
        "unmatched_helmets": sum(len(r.unmatched_helmet_ids) for r in results),
        "unmatched_no_helmet": sum(len(r.unmatched_no_helmet_ids) for r in results),
        "wall_time_s": round(wall_time_s, 3),
        "avg_ms_per_image": round(wall_time_s * 1000 / processed, 2) if processed else None,
        "avg_inference_ms": (round(sum(r.inference_ms for r in results) / processed, 2)
                             if processed else None),
    }
