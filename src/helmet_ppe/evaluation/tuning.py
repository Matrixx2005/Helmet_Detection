"""Dò ngưỡng trên tập validation (không dùng tập test để chọn tham số).

Ý tưởng: chạy YOLO một lần với confidence rất thấp (vd. 0.02) để lưu mọi
detection (của cả mô hình PPE và mô hình pose) vào ``detections.csv``. Sau đó,
với mỗi tổ hợp (bật/tắt pose, confidence theo từng lớp, ngưỡng bằng chứng, biên
chênh lệch), lọc lại detection, chạy lại hợp nhất người + association + rule
engine *offline* và so với nhãn chuẩn cấp người. Không cần
chạy lại mô hình nên rất nhanh.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, fields, replace
from itertools import product
from pathlib import Path

from helmet_ppe.config import AssociationConfig, DetectorConfig, PipelineConfig, RuleConfig
from helmet_ppe.evaluation.person_eval import evaluate_persons
from helmet_ppe.pipeline import analyze_detections
from helmet_ppe.reporting import person_rows, read_csv
from helmet_ppe.schemas import Detection, ImageResult


@dataclass(frozen=True)
class SweepGrid:
    """Các giá trị được thử cho từng tham số (tích Descartes)."""

    use_pose: tuple[bool, ...] = (False, True)
    conf_pose_person: tuple[float, ...] = (0.3, 0.4, 0.5)
    conf_person: tuple[float, ...] = (0.3, 0.4, 0.5)
    conf_helmet: tuple[float, ...] = (0.2, 0.25, 0.3)
    conf_no_helmet: tuple[float, ...] = (0.02, 0.03, 0.05, 0.1)
    min_helmet_evidence: tuple[float, ...] = (0.1,)
    min_no_helmet_evidence: tuple[float, ...] = (0.0, 0.02, 0.05)
    margin: tuple[float, ...] = (0.0,)
    # Hình học ghép theo đầu (chỉ có tác dụng khi bật pose)
    head_tolerance_x: tuple[float, ...] = (0.9, 1.2)
    head_tolerance_y: tuple[float, ...] = (0.55, 0.8)
    min_head_size_ratio: tuple[float, ...] = (0.5, 0.8)
    suppress_faces_under_helmets: tuple[bool, ...] = (True, False)

    def combinations(self):
        """Tích Descartes; khi tắt pose, các tham số chỉ dùng cho pose được giữ cố định."""
        names = [f.name for f in fields(self)]
        pose_only = ("conf_pose_person", "head_tolerance_x", "head_tolerance_y", "min_head_size_ratio")
        for values in product(*(getattr(self, n) for n in names)):
            params = dict(zip(names, values))
            if not params["use_pose"] and any(params[n] != getattr(self, n)[0] for n in pose_only):
                continue
            yield params


def load_run(run_dir: Path) -> tuple[dict[str, list[Detection]], dict[str, tuple[int, int]]]:
    """Đọc detections.csv + summary.csv của một lần chạy batch."""
    sizes = {r["image_name"]: (int(r["width"]), int(r["height"]))
             for r in read_csv(Path(run_dir) / "summary.csv")}
    detections: dict[str, list[Detection]] = defaultdict(list)
    for r in read_csv(Path(run_dir) / "detections.csv"):
        keypoints = r.get("keypoints") or ""
        detections[r["image_name"]].append(Detection(
            det_id=int(r["det_id"]), class_id=int(r["class_id"]), class_name=r["class_name"],
            confidence=float(r["confidence"]),
            bbox=(float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"])),
            source=r.get("source") or "ppe",
            keypoints=tuple(tuple(kp) for kp in json.loads(keypoints)) if keypoints else None))
    return detections, sizes


def reanalyze(detections: dict[str, list[Detection]], sizes: dict[str, tuple[int, int]],
              cfg: PipelineConfig) -> list[ImageResult]:
    """Chạy lại lọc ngưỡng + hợp nhất người + association + rules với cấu hình ``cfg``."""
    return [
        analyze_detections(detections.get(name, []), image_name=name, image_path=name,
                           width=w, height=h, cfg=cfg)
        for name, (w, h) in sizes.items()
    ]


def objective(metrics: dict) -> float:
    """Trung bình F1 của hai lớp kết luận (vi phạm và đội mũ)."""
    return ((metrics["violation"]["f1"] or 0.0) + (metrics["helmet"]["f1"] or 0.0)) / 2


def config_from(params: dict) -> PipelineConfig:
    """Tạo PipelineConfig từ một dòng tham số của lưới."""
    return PipelineConfig(
        detector=DetectorConfig(conf_person=params["conf_person"], conf_helmet=params["conf_helmet"],
                                conf_no_helmet=params["conf_no_helmet"],
                                use_pose=params["use_pose"],
                                conf_pose_person=params["conf_pose_person"]),
        association=AssociationConfig(head_tolerance_x=params["head_tolerance_x"],
                                      head_tolerance_y=params["head_tolerance_y"],
                                      min_head_size_ratio=params["min_head_size_ratio"],
                                      suppress_faces_under_helmets=params.get(
                                          "suppress_faces_under_helmets", True)),
        rules=replace(RuleConfig(), min_helmet_evidence=params["min_helmet_evidence"],
                      min_no_helmet_evidence=params["min_no_helmet_evidence"],
                      decision_margin=params["margin"]),
    )


def sweep(run_dir: Path, gt_rows: list[dict], grid: SweepGrid | None = None, *,
          iou_threshold: float = 0.5) -> list[dict]:
    """Thử mọi tổ hợp tham số; trả về bảng kết quả sắp theo mục tiêu giảm dần."""
    grid = grid or SweepGrid()
    detections, sizes = load_run(run_dir)
    images = set(sizes)
    gt_rows = [g for g in gt_rows if g["image_name"] in images]
    table = []
    for params in grid.combinations():
        results = reanalyze(detections, sizes, config_from(params))
        preds = [row for r in results for row in person_rows(r)]
        m = evaluate_persons(preds, gt_rows, iou_threshold=iou_threshold, images=images)
        table.append({
            **params,
            "objective": round(objective(m), 4),
            "violation_precision": m["violation"]["precision"],
            "violation_recall": m["violation"]["recall"], "violation_f1": m["violation"]["f1"],
            "helmet_precision": m["helmet"]["precision"],
            "helmet_recall": m["helmet"]["recall"], "helmet_f1": m["helmet"]["f1"],
            "unknown_rate": m["unknown_rate_matched"],
            "person_recall": m["person_detection_recall"],
        })
    # Hòa điểm: ưu tiên ngưỡng cao hơn (thận trọng hơn, ít báo nhầm hơn).
    names = [f.name for f in fields(SweepGrid)]
    return sorted(table, key=lambda r: (-r["objective"], *(-r[n] for n in names)))
