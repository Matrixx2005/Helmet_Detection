"""Dò ngưỡng hệ luật trên tập validation (không dùng tập test để chọn tham số).

Ý tưởng: chạy YOLO một lần với confidence thấp (vd. 0.05) để lưu mọi detection
vào ``detections.csv``. Sau đó với mỗi tổ hợp (ngưỡng confidence, ngưỡng bằng
chứng), lọc lại detection, chạy lại association + rule engine *offline* và so
với nhãn chuẩn cấp người. Không cần chạy lại mô hình nên rất nhanh.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from itertools import product
from pathlib import Path
from typing import Iterable

from helmet_ppe.config import PipelineConfig
from helmet_ppe.evaluation.person_eval import evaluate_persons
from helmet_ppe.pipeline import analyze_detections
from helmet_ppe.reporting import person_rows, read_csv
from helmet_ppe.schemas import Detection, ImageResult


def load_run(run_dir: Path) -> tuple[dict[str, list[Detection]], dict[str, tuple[int, int]]]:
    """Đọc detections.csv + summary.csv của một lần chạy batch."""
    sizes = {r["image_name"]: (int(r["width"]), int(r["height"]))
             for r in read_csv(Path(run_dir) / "summary.csv")}
    detections: dict[str, list[Detection]] = defaultdict(list)
    for r in read_csv(Path(run_dir) / "detections.csv"):
        detections[r["image_name"]].append(Detection(
            det_id=int(r["det_id"]), class_id=int(r["class_id"]), class_name=r["class_name"],
            confidence=float(r["confidence"]),
            bbox=(float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"]))))
    return detections, sizes


def reanalyze(detections: dict[str, list[Detection]], sizes: dict[str, tuple[int, int]],
              conf: float, cfg: PipelineConfig) -> list[ImageResult]:
    """Chạy lại association + rules sau khi lọc detection theo ``conf``."""
    return [
        analyze_detections([d for d in detections.get(name, []) if d.confidence >= conf],
                           image_name=name, image_path=name, width=w, height=h, cfg=cfg)
        for name, (w, h) in sizes.items()
    ]


def objective(metrics: dict) -> float:
    """Trung bình F1 của hai lớp kết luận (vi phạm và đội mũ)."""
    return ((metrics["violation"]["f1"] or 0.0) + (metrics["helmet"]["f1"] or 0.0)) / 2


def sweep(run_dir: Path, gt_rows: list[dict], *, confs: Iterable[float],
          evidences: Iterable[float], margins: Iterable[float],
          iou_threshold: float = 0.5) -> list[dict]:
    """Thử mọi tổ hợp tham số; trả về bảng kết quả sắp theo mục tiêu giảm dần."""
    detections, sizes = load_run(run_dir)
    images = set(sizes)
    gt_rows = [g for g in gt_rows if g["image_name"] in images]
    base = PipelineConfig()
    table = []
    for conf, evidence, margin in product(confs, evidences, margins):
        rules = replace(base.rules, min_helmet_evidence=evidence,
                        min_no_helmet_evidence=evidence, decision_margin=margin)
        results = reanalyze(detections, sizes, conf, replace(base, rules=rules))
        preds = [row for r in results for row in person_rows(r)]
        m = evaluate_persons(preds, gt_rows, iou_threshold=iou_threshold, images=images)
        table.append({
            "conf": conf, "min_evidence": evidence, "margin": margin,
            "objective": round(objective(m), 4),
            "violation_precision": m["violation"]["precision"],
            "violation_recall": m["violation"]["recall"], "violation_f1": m["violation"]["f1"],
            "helmet_precision": m["helmet"]["precision"],
            "helmet_recall": m["helmet"]["recall"], "helmet_f1": m["helmet"]["f1"],
            "unknown_rate": m["unknown_rate_matched"],
            "person_recall": m["person_detection_recall"],
        })
    # Hòa điểm: ưu tiên ngưỡng conf cao hơn và bằng chứng chặt hơn (thận trọng hơn).
    return sorted(table, key=lambda r: (-r["objective"], -r["conf"], -r["min_evidence"]))
