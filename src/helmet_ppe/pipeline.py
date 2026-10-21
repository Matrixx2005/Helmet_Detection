"""Ghép ba module thành một pipeline: Detection → Association → Rule Engine."""

from __future__ import annotations

import logging
import time
from dataclasses import replace
from pathlib import Path
from typing import Sequence

import numpy as np

from helmet_ppe.association import associate
from helmet_ppe.config import HELMET, NO_HELMET, PipelineConfig
from helmet_ppe.detector import HelmetDetector
from helmet_ppe.fusion import filter_detections, fuse_persons, suppress_faces_under_helmets
from helmet_ppe.pose import PoseEstimator
from helmet_ppe.rule_engine import evaluate_image
from helmet_ppe.schemas import Detection, ImageResult

log = logging.getLogger(__name__)


def analyze_detections(detections: list[Detection], *, image_name: str, image_path: str,
                       width: int, height: int, cfg: PipelineConfig | None = None,
                       inference_ms: float = 0.0) -> ImageResult:
    """Lọc theo ngưỡng → bỏ mặt dưới mũ → hợp nhất người → association → rule engine.

    Hàm thuần (không cần mô hình), dùng chung cho pipeline, kiểm thử, dò ngưỡng
    offline và để sinh nhãn cấp người từ nhãn hộp chuẩn.
    """
    cfg = cfg or PipelineConfig()
    kept = filter_detections(detections, cfg)
    if cfg.association.suppress_faces_under_helmets:
        kept = suppress_faces_under_helmets(kept)
    persons, merged = fuse_persons(kept, cfg.association.person_merge_iou)
    heads = [d for d in kept if d.label in (HELMET, NO_HELMET)]
    matched = associate(persons + heads, cfg.association)
    results, image_status = evaluate_image(matched.associations, cfg.rules)
    return ImageResult(
        image_name=image_name,
        image_path=image_path,
        width=width,
        height=height,
        detections=kept,
        persons=results,
        image_status=image_status,
        unmatched_helmet_ids=matched.unmatched_helmet_ids,
        unmatched_no_helmet_ids=matched.unmatched_no_helmet_ids,
        merged_pose_ids=merged,
        inference_ms=inference_ms,
    )


class HelmetPipeline:
    """Pipeline đầy đủ; mô hình YOLO chỉ được nạp một lần khi khởi tạo."""

    def __init__(self, cfg: PipelineConfig | None = None) -> None:
        cfg = cfg or PipelineConfig()
        if cfg.detector.use_pose and not Path(cfg.detector.pose_model_path).is_file():
            log.warning("Không có %s → chạy không dùng mô hình pose", cfg.detector.pose_model_path)
            cfg = replace(cfg, detector=replace(cfg.detector, use_pose=False))
        self.cfg = cfg
        self.detector = HelmetDetector(cfg.detector)
        self.pose = PoseEstimator(cfg.detector) if cfg.detector.use_pose else None

    def analyze_batch(self, images: Sequence[np.ndarray],
                      paths: Sequence[str | Path]) -> list[ImageResult]:
        """Phân tích một lô ảnh BGR đã đọc sẵn (paths dùng để đặt tên kết quả)."""
        start = time.perf_counter()
        outputs = self._detect(images)
        results = [
            analyze_detections(
                detections,
                image_name=Path(path).name,
                image_path=str(path),
                width=int(image.shape[1]),
                height=int(image.shape[0]),
                cfg=self.cfg,
                inference_ms=inference_ms,
            )
            for image, path, (detections, inference_ms) in zip(images, paths, outputs)
        ]
        per_image_ms = (time.perf_counter() - start) * 1000.0 / max(len(results), 1)
        for result in results:
            result.total_ms = per_image_ms
        return results

    def _detect(self, images: Sequence[np.ndarray]) -> list[tuple[list[Detection], float]]:
        """Detection PPE (+ người pose nếu bật) cho từng ảnh, kèm thời gian suy luận (ms)."""
        outputs = self.detector.detect_batch(images)
        if self.pose is None:
            return outputs
        poses = self.pose.estimate_batch(images)
        return [(dets + PoseEstimator.as_detections(people, first_id=len(dets)), ms + pose_ms)
                for (dets, ms), (people, pose_ms) in zip(outputs, poses)]

    def analyze(self, image: np.ndarray, path: str | Path = "image.jpg") -> ImageResult:
        """Phân tích một ảnh BGR."""
        return self.analyze_batch([image], [path])[0]
