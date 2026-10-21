"""Bọc mô hình YOLO (Ultralytics) thành danh sách ``Detection``."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np

from helmet_ppe.config import RELEVANT_CLASSES, DetectorConfig
from helmet_ppe.geometry import to_bbox
from helmet_ppe.schemas import Detection

log = logging.getLogger(__name__)


class ModelError(RuntimeError):
    """Mô hình không tải được hoặc không có các lớp cần thiết."""


class HelmetDetector:
    """Nạp YOLO một lần và phát hiện người / mũ / no_helmet trên nhiều ảnh."""

    def __init__(self, cfg: DetectorConfig | None = None) -> None:
        self.cfg = cfg or DetectorConfig()
        model_path = Path(self.cfg.model_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"Không tìm thấy mô hình: {model_path}")

        try:
            from ultralytics import YOLO  # import chậm: chỉ khi thật sự cần mô hình
        except ImportError as exc:  # pragma: no cover - phụ thuộc môi trường
            raise ModelError("Chưa cài ultralytics. Chạy: pip install -r requirements.txt") from exc

        try:
            self.model = YOLO(str(model_path))
        except Exception as exc:  # checkpoint hỏng / sai phiên bản
            raise ModelError(f"Không nạp được mô hình {model_path}: {exc}") from exc

        self.names: dict[int, str] = {int(k): str(v) for k, v in self.model.names.items()}
        self.class_ids = self._relevant_class_ids() if self.cfg.only_relevant_classes else None
        log.info("Đã nạp %s (%d lớp)", model_path.name, len(self.names))

    def _relevant_class_ids(self) -> list[int]:
        lookup = {name.lower(): idx for idx, name in self.names.items()}
        missing = [name for name in RELEVANT_CLASSES if name not in lookup]
        if missing:
            raise ModelError(f"Mô hình thiếu lớp bắt buộc {missing}; các lớp hiện có: {self.names}")
        return [lookup[name] for name in RELEVANT_CLASSES]

    def detect_batch(self, images: Sequence[np.ndarray]) -> list[tuple[list[Detection], float]]:
        """Phát hiện trên một lô ảnh BGR.

        Returns:
            Với mỗi ảnh: (danh sách Detection sắp theo confidence giảm dần,
            thời gian suy luận mô hình tính bằng ms).
        """
        if not images:
            return []
        results = self.model.predict(
            source=list(images),
            conf=self.cfg.conf,
            iou=self.cfg.iou,
            imgsz=self.cfg.imgsz,
            device=self.cfg.device,
            classes=self.class_ids,
            verbose=False,
        )
        return [(self._to_detections(r), float(r.speed.get("inference", 0.0))) for r in results]

    def detect(self, image: np.ndarray) -> tuple[list[Detection], float]:
        """Phát hiện trên một ảnh BGR."""
        return self.detect_batch([image])[0]

    def _to_detections(self, result) -> list[Detection]:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)
        order = np.argsort(-confs, kind="stable")
        return [
            Detection(
                det_id=det_id,
                class_id=int(classes[i]),
                class_name=self.names[int(classes[i])],
                confidence=float(confs[i]),
                bbox=to_bbox(xyxy[i]),
            )
            for det_id, i in enumerate(order)
        ]
