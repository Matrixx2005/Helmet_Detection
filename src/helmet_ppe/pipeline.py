"""Ghép ba module thành một pipeline: Detection → Association → Rule Engine."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Sequence

import numpy as np

from helmet_ppe.association import associate
from helmet_ppe.config import PipelineConfig
from helmet_ppe.detector import HelmetDetector
from helmet_ppe.rule_engine import evaluate_image
from helmet_ppe.schemas import Detection, ImageResult


def analyze_detections(detections: list[Detection], *, image_name: str, image_path: str,
                       width: int, height: int, cfg: PipelineConfig | None = None,
                       inference_ms: float = 0.0) -> ImageResult:
    """Chạy association + rule engine trên danh sách detection có sẵn.

    Hàm thuần (không cần mô hình), dùng chung cho pipeline, kiểm thử và để
    sinh nhãn cấp người từ nhãn hộp chuẩn.
    """
    cfg = cfg or PipelineConfig()
    matched = associate(detections, cfg.association)
    persons, image_status = evaluate_image(matched.associations, cfg.rules)
    return ImageResult(
        image_name=image_name,
        image_path=image_path,
        width=width,
        height=height,
        detections=detections,
        persons=persons,
        image_status=image_status,
        unmatched_helmet_ids=matched.unmatched_helmet_ids,
        unmatched_no_helmet_ids=matched.unmatched_no_helmet_ids,
        inference_ms=inference_ms,
    )


class HelmetPipeline:
    """Pipeline đầy đủ; mô hình YOLO chỉ được nạp một lần khi khởi tạo."""

    def __init__(self, cfg: PipelineConfig | None = None) -> None:
        self.cfg = cfg or PipelineConfig()
        self.detector = HelmetDetector(self.cfg.detector)

    def analyze_batch(self, images: Sequence[np.ndarray],
                      paths: Sequence[str | Path]) -> list[ImageResult]:
        """Phân tích một lô ảnh BGR đã đọc sẵn (paths dùng để đặt tên kết quả)."""
        start = time.perf_counter()
        outputs = self.detector.detect_batch(images)
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

    def analyze(self, image: np.ndarray, path: str | Path = "image.jpg") -> ImageResult:
        """Phân tích một ảnh BGR."""
        return self.analyze_batch([image], [path])[0]
