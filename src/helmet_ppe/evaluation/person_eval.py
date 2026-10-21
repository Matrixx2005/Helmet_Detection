"""So sánh kết luận cấp người (persons.csv) với nhãn chuẩn cấp người.

1. Trong từng ảnh, ghép người dự đoán với người chuẩn theo IoU (tham lam,
   một-một, IoU ≥ ngưỡng).
2. Lập ma trận nhầm lẫn: hàng = nhãn chuẩn (HELMET / NO_HELMET /
   UNDETERMINED), cột = dự đoán (HELMET_DETECTED / SUSPECTED_NO_HELMET /
   UNKNOWN / MISSED). Người dự đoán không khớp ai được đếm riêng là
   "false person".
3. Tính Precision / Recall / F1 cho hai lớp quan tâm:
   * Vi phạm: chuẩn NO_HELMET ↔ dự đoán SUSPECTED_NO_HELMET.
   * Đội mũ:  chuẩn HELMET    ↔ dự đoán HELMET_DETECTED.
   Recall tính trên **toàn bộ** người chuẩn của lớp đó, nên UNKNOWN và
   MISSED đều làm giảm recall (không được "giấu" ca khó vào UNKNOWN).
   Người chuẩn UNDETERMINED không dùng để tính P/R.
"""

from __future__ import annotations

from collections import defaultdict

from helmet_ppe.config import Status
from helmet_ppe.evaluation.ground_truth import GT_HELMET, GT_NO_HELMET, GT_STATUSES, GT_UNDETERMINED
from helmet_ppe.geometry import iou

MISSED = "MISSED"
PRED_COLUMNS = (str(Status.HELMET_DETECTED), str(Status.SUSPECTED_NO_HELMET),
                str(Status.UNKNOWN), MISSED)


def _box(row: dict) -> tuple[float, float, float, float]:
    return float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"])


def match_persons(preds: list[dict], gts: list[dict], iou_threshold: float
                  ) -> tuple[list[tuple[dict, dict, float]], list[dict], list[dict]]:
    """Ghép tham lam theo IoU giảm dần. Trả về (cặp khớp, gt bị sót, pred thừa)."""
    pairs = sorted(
        ((iou(_box(p), _box(g)), pi, gi)
         for pi, p in enumerate(preds) for gi, g in enumerate(gts)),
        reverse=True,
    )
    used_p, used_g, matches = set(), set(), []
    for score, pi, gi in pairs:
        if score < iou_threshold:
            break
        if pi in used_p or gi in used_g:
            continue
        used_p.add(pi)
        used_g.add(gi)
        matches.append((preds[pi], gts[gi], score))
    missed = [g for gi, g in enumerate(gts) if gi not in used_g]
    extra = [p for pi, p in enumerate(preds) if pi not in used_p]
    return matches, missed, extra


def _ratio(num: float, den: float) -> float | None:
    return round(num / den, 4) if den else None


def _prf(tp: int, fp: int, fn: int) -> dict:
    """Precision / Recall / F1; giá trị None khi mẫu số bằng 0 (không định nghĩa)."""
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    f1 = None
    if precision is not None and recall is not None:
        f1 = _ratio(2 * precision * recall, precision + recall) or 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def _case(kind: str, image: str, gt: dict | None, pred: dict | None, overlap: float | None) -> dict:
    return {
        "error_type": kind, "image_name": image,
        "gt_person_id": gt["gt_person_id"] if gt else "",
        "gt_status": gt["gt_status"] if gt else "",
        "pred_person_id": pred["person_id"] if pred else "",
        "pred_status": pred["status"] if pred else "",
        "rule_id": pred.get("rule_id", "") if pred else "",
        "iou": "" if overlap is None else round(overlap, 3),
        "reason": pred.get("reason", "") if pred else "",
    }


def evaluate_persons(pred_rows: list[dict], gt_rows: list[dict], *,
                     iou_threshold: float = 0.5, images: set[str] | None = None) -> dict:
    """Đánh giá suy luận cấp người.

    Args:
        pred_rows: các dòng persons.csv.
        gt_rows: các dòng nhãn chuẩn cấp người.
        images: chỉ đánh giá các ảnh này (mặc định: các ảnh có trong summary dự đoán
            hoặc nhãn). Nên truyền danh sách ảnh đã xử lý thành công.
    """
    by_image_pred: dict[str, list[dict]] = defaultdict(list)
    by_image_gt: dict[str, list[dict]] = defaultdict(list)
    for row in pred_rows:
        by_image_pred[row["image_name"]].append(row)
    for row in gt_rows:
        by_image_gt[row["image_name"]].append(row)
    image_names = sorted(images if images is not None else set(by_image_pred) | set(by_image_gt))

    cm = {g: {c: 0 for c in PRED_COLUMNS} for g in GT_STATUSES}
    false_persons = {c: 0 for c in PRED_COLUMNS[:3]}
    cases: list[dict] = []

    for image in image_names:
        matches, missed, extra = match_persons(by_image_pred.get(image, []),
                                               by_image_gt.get(image, []), iou_threshold)
        for pred, gt, overlap in matches:
            cm[gt["gt_status"]][pred["status"]] += 1
            kind = _classify(gt["gt_status"], pred["status"])
            if kind:
                cases.append(_case(kind, image, gt, pred, overlap))
        for gt in missed:
            cm[gt["gt_status"]][MISSED] += 1
            cases.append(_case("MISSED_PERSON", image, gt, None, None))
        for pred in extra:
            false_persons[pred["status"]] += 1
            cases.append(_case("FALSE_PERSON", image, None, pred, None))

    helmet, violation = str(Status.HELMET_DETECTED), str(Status.SUSPECTED_NO_HELMET)
    unknown = str(Status.UNKNOWN)
    gt_helmet_total = sum(cm[GT_HELMET].values())
    gt_violation_total = sum(cm[GT_NO_HELMET].values())
    gt_total = sum(sum(row.values()) for row in cm.values())
    matched = gt_total - sum(cm[g][MISSED] for g in GT_STATUSES)
    matched_unknown = sum(cm[g][unknown] for g in GT_STATUSES)
    decided_labeled = sum(cm[g][c] for g in (GT_HELMET, GT_NO_HELMET) for c in (helmet, violation))
    correct = cm[GT_HELMET][helmet] + cm[GT_NO_HELMET][violation]

    return {
        "iou_threshold": iou_threshold,
        "num_images": len(image_names),
        "gt_persons": gt_total,
        "gt_status_counts": {g: sum(cm[g].values()) for g in GT_STATUSES},
        "pred_persons": sum(len(by_image_pred.get(image, [])) for image in image_names),
        "matched_persons": matched,
        "missed_persons": gt_total - matched,
        "false_persons": sum(false_persons.values()),
        "false_persons_by_status": false_persons,
        "person_detection_recall": _ratio(matched, gt_total),
        "unknown_rate_matched": _ratio(matched_unknown, matched),
        "decided_accuracy": _ratio(correct, decided_labeled),
        "violation": _prf(
            tp=cm[GT_NO_HELMET][violation],
            fp=cm[GT_HELMET][violation] + false_persons[violation],
            fn=gt_violation_total - cm[GT_NO_HELMET][violation]),
        "helmet": _prf(
            tp=cm[GT_HELMET][helmet],
            fp=cm[GT_NO_HELMET][helmet] + false_persons[helmet],
            fn=gt_helmet_total - cm[GT_HELMET][helmet]),
        "confusion_matrix": {"rows": list(GT_STATUSES), "cols": list(PRED_COLUMNS),
                             "values": [[cm[g][c] for c in PRED_COLUMNS] for g in GT_STATUSES]},
        "cases": cases,
    }


def _classify(gt_status: str, pred_status: str) -> str | None:
    """Loại lỗi cho một cặp đã khớp (None nếu đúng hoặc không xác định được)."""
    helmet, violation = str(Status.HELMET_DETECTED), str(Status.SUSPECTED_NO_HELMET)
    if gt_status == GT_HELMET and pred_status == violation:
        return "FALSE_VIOLATION"
    if gt_status == GT_NO_HELMET and pred_status == helmet:
        return "MISSED_VIOLATION"
    if gt_status == GT_NO_HELMET and pred_status == str(Status.UNKNOWN):
        return "VIOLATION_AS_UNKNOWN"
    if gt_status == GT_HELMET and pred_status == str(Status.UNKNOWN):
        return "HELMET_AS_UNKNOWN"
    if gt_status == GT_UNDETERMINED and pred_status in (helmet, violation):
        return "DECIDED_ON_UNDETERMINED"
    return None
