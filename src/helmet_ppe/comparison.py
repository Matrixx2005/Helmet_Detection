"""So sánh hai cách kết luận và tạo nhận xét từ đúng số liệu đã tính."""

from __future__ import annotations

import csv
import io
import math
from collections import Counter
from dataclasses import dataclass
from typing import Sequence

from helmet_ppe.config import HELMET, NO_HELMET, PERSON_STATUSES, Status
from helmet_ppe.evaluation.baseline import naive_image_result
from helmet_ppe.evaluation.ground_truth import GT_STATUSES, REQUIRED_GT_FIELDS
from helmet_ppe.evaluation.person_eval import evaluate_persons
from helmet_ppe.reporting import person_rows
from helmet_ppe.schemas import ImageResult

STATUS_LABELS = {
    str(Status.HELMET_DETECTED): "Đội mũ",
    str(Status.SUSPECTED_NO_HELMET): "Nghi không đội mũ",
    str(Status.UNKNOWN): "Chưa đủ bằng chứng",
}
COMPARISON_FIELDS = [
    "image_name", "person_id", "before_status", "after_status", "changed",
    "before_reason", "after_rule", "after_reason", "helmet_evidence", "no_helmet_evidence",
]


@dataclass
class ComparisonReport:
    before: list[ImageResult]
    after: list[ImageResult]
    rows: list[dict]
    summary: dict
    comments: list[str]

    def to_dict(self) -> dict:
        return {
            "method": "Cùng hộp sau lọc và cùng người sau hợp nhất; ghép đơn giản so với ghép vùng đầu + hệ luật.",
            "summary": self.summary, "comments": self.comments, "persons": self.rows,
            "before": [r.to_dict() for r in self.before],
            "after": [r.to_dict() for r in self.after],
        }


def compare_results(results: Sequence[ImageResult]) -> ComparisonReport:
    """Sinh đối chứng, bảng thay đổi và nhận xét, không chạy lại mô hình."""
    results = list(results)
    if len({r.image_name for r in results}) != len(results):
        raise ValueError("Tên ảnh trong lượt so sánh bị trùng.")
    before = [naive_image_result(r) for r in results]
    rows = []
    for original, final in zip(before, results):
        for previous, current in zip(original.persons, final.persons):
            rows.append({
                "image_name": final.image_name, "person_id": current.person_id,
                "before_status": str(previous.status), "after_status": str(current.status),
                "changed": previous.status != current.status,
                "before_reason": previous.reason, "after_rule": current.rule_id,
                "after_reason": current.reason,
                "helmet_evidence": current.helmet.strength if current.helmet else 0.0,
                "no_helmet_evidence": current.no_helmet.strength if current.no_helmet else 0.0,
            })
    old_counts = Counter(row["before_status"] for row in rows)
    new_counts = Counter(row["after_status"] for row in rows)
    transitions = Counter((row["before_status"], row["after_status"]) for row in rows if row["changed"])
    summary = {
        "images": len(results), "persons": len(rows),
        "before_counts": {str(s): old_counts[str(s)] for s in PERSON_STATUSES},
        "after_counts": {str(s): new_counts[str(s)] for s in PERSON_STATUSES},
        "changed_persons": sum(transitions.values()),
        "helmet_objects": sum(r.count(HELMET) for r in results),
        "no_helmet_objects": sum(r.count(NO_HELMET) for r in results),
    }
    suspected = str(Status.SUSPECTED_NO_HELMET)
    comments = [
        f"Đã so sánh {summary['images']} ảnh, với cùng {summary['persons']} người ở hai phương án.",
        f"Đầu vào chung sau lọc có {summary['helmet_objects']} hộp mũ và "
        f"{summary['no_helmet_objects']} hộp đầu không mũ.",
        f"Số người nghi không đội mũ thay đổi từ {old_counts[suspected]} "
        f"xuống {new_counts[suspected]}." if new_counts[suspected] < old_counts[suspected] else
        f"Số người nghi không đội mũ trước/sau là {old_counts[suspected]}/{new_counts[suspected]}.",
        f"Có {summary['changed_persons']} người thay đổi trạng thái kết luận.",
    ]
    comments.extend(f"{count} người chuyển từ {STATUS_LABELS[old]} sang {STATUS_LABELS[new]}."
                    for (old, new), count in sorted(transitions.items()))
    return ComparisonReport(before, results, rows, summary, comments)


def read_comparison_gt(content: bytes) -> list[dict]:
    """Đọc CSV nhãn do người dùng cung cấp, kiểm tra trước khi tính chỉ số."""
    try:
        reader = csv.DictReader(io.StringIO(content.decode("utf-8-sig")), strict=True)
        headers = reader.fieldnames or []
        missing = set(REQUIRED_GT_FIELDS) - set(headers)
        if missing:
            raise ValueError(f"CSV thiếu cột {', '.join(sorted(missing))}.")
        if len(set(headers)) != len(headers):
            raise ValueError("CSV có tên cột bị trùng.")
        rows = list(reader)
    except UnicodeDecodeError as exc:
        raise ValueError("CSV cần được lưu bằng UTF-8.") from exc
    except csv.Error as exc:
        raise ValueError(f"Không đọc được CSV: {exc}") from exc
    seen = set()
    for index, row in enumerate(rows, start=2):
        if None in row or any(value is None for value in row.values()):
            raise ValueError(f"Dòng {index} có số cột không hợp lệ.")
        row.update({k: v.strip() for k, v in row.items()})
        if not row["image_name"] or not row["gt_person_id"]:
            raise ValueError(f"Dòng {index} thiếu tên ảnh hoặc ID người.")
        if row["gt_status"] not in GT_STATUSES:
            raise ValueError(f"Dòng {index} có gt_status không hợp lệ.")
        identity = (row["image_name"], row["gt_person_id"])
        if identity in seen:
            raise ValueError(f"Dòng {index} trùng tên ảnh và ID người.")
        seen.add(identity)
        for prefix in ("", "head_"):
            values = [row.get(f"{prefix}{key}", "") for key in ("x1", "y1", "x2", "y2")]
            if prefix and not any(values):
                continue
            try:
                x1, y1, x2, y2 = map(float, values)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Dòng {index} có tọa độ {prefix or 'người'} không hợp lệ.") from exc
            if not all(math.isfinite(v) for v in (x1, y1, x2, y2)) or x2 <= x1 or y2 <= y1:
                raise ValueError(f"Dòng {index} có khung {prefix or 'người'} rỗng hoặc không hữu hạn.")
    return rows


def evaluate_comparison(report: ComparisonReport, gt_rows: list[dict], *,
                        covered_images: set[str]) -> dict:
    """Đánh giá cùng khung người/IoU cho hai phía, loại ảnh chưa được gán nhãn.

    covered_images là phạm vi được xác nhận có nhãn, bao gồm ảnh không có người
    nếu nguồn dữ liệu có danh sách ảnh riêng. Với CSV tải lên, dùng các tên ảnh
    xuất hiện trong CSV. Ghép theo khung người ở cả hai phía để tránh việc chọn
    mũ khác nhau làm thay đổi cặp người chuẩn.
    """
    names = {r.image_name for r in report.after}
    evaluated = names & covered_images
    if not evaluated:
        raise ValueError("Nhãn chuẩn chưa có ảnh nào trùng tên với kết quả so sánh.")
    metrics = {}
    for label, results in (("before", report.before), ("after", report.after)):
        predictions = [row for result in results for row in person_rows(result)]
        predictions = [{k: v for k, v in row.items() if not k.startswith("head_")}
                       for row in predictions]
        metrics[label] = evaluate_persons(predictions, gt_rows, images=evaluated, iou_threshold=0.5)
    selected_gt = [r for r in gt_rows if r["image_name"] in evaluated]
    comments = [
        f"Đánh giá {len(evaluated)} ảnh có nhãn, gồm {metrics['after']['gt_persons']} người chuẩn. "
        f"Có {len(names - evaluated)} ảnh so sánh chưa có nhãn được loại khỏi phép đánh giá.",
        f"Hai phương án dùng cùng khung người và ghép với nhãn chuẩn theo IoU ≥ 0,50.",
    ]
    for key, label in (("precision", "Precision vi phạm"), ("recall", "Recall vi phạm"),
                       ("f1", "F1 vi phạm")):
        old, new = metrics["before"]["violation"][key], metrics["after"]["violation"][key]
        if old is None or new is None:
            comments.append(f"{label} chưa xác định ở ít nhất một phương án do mẫu số bằng 0.")
        else:
            delta = (new - old) * 100
            change = "tăng" if delta > 0 else "giảm"
            ending = f"{change} {abs(delta):.2f} điểm phần trăm" if delta else "không đổi"
            comments.append(f"{label} từ {old:.2%} lên {new:.2%}, {ending}." if delta > 0 else
                            f"{label} trước/sau {old:.2%}/{new:.2%}, {ending}.")
    old, new = metrics["before"], metrics["after"]
    comments.extend([
        f"Cảnh báo vi phạm sai (FP) trước/sau là {old['violation']['fp']}/{new['violation']['fp']}.",
        f"Người chuẩn không đội mũ chưa được cảnh báo đúng (FN) trước/sau là "
        f"{old['violation']['fn']}/{new['violation']['fn']}.",
        f"Người chuẩn bị bỏ sót trước/sau là {old['missed_persons']}/{new['missed_persons']}; "
        f"người dự đoán không khớp nhãn là {old['false_persons']}/{new['false_persons']}.",
    ])
    return {
        **metrics, "evaluated_images": sorted(evaluated), "unlabeled_images": sorted(names - evaluated),
        "gt_sources": dict(Counter(r.get("source") or "không khai báo" for r in selected_gt)),
        "comments": comments,
    }
