"""Phát hiện người kèm 17 điểm khớp bằng YOLO26n-pose (huấn luyện sẵn trên COCO)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np

from helmet_ppe.config import POSE_PERSON, DetectorConfig
from helmet_ppe.detector import ModelError
from helmet_ppe.geometry import to_bbox
from helmet_ppe.schemas import Detection

log = logging.getLogger(__name__)

POSE_CLASS_ID = -1  # không thuộc 11 lớp của mô hình PPE


class PoseEstimator:
    """Nạp mô hình pose một lần; trả về người (hộp, độ tin cậy, điểm khớp)."""

    def __init__(self, cfg: DetectorConfig) -> None:
        self.cfg = cfg
        path = Path(cfg.pose_model_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"Không tìm thấy mô hình pose: {path}. Đặt yolo26n-pose.pt vào models/ "
                "hoặc tắt bằng --no-pose.")
        from ultralytics import YOLO

        try:
            self.model = YOLO(str(path))
        except Exception as exc:  # checkpoint hỏng / sai phiên bản
            raise ModelError(f"Không nạp được mô hình pose {path}: {exc}") from exc
        log.info("Đã nạp mô hình pose %s", path.name)

    def estimate_batch(self, images: Sequence[np.ndarray]) -> list[tuple[list[tuple], float]]:
        """Với mỗi ảnh: (danh sách (bbox, conf, keypoints), thời gian suy luận ms)."""
        if not images:
            return []
        results = self.model.predict(source=list(images), conf=self.cfg.min_conf,
                                     imgsz=self.cfg.imgsz, device=self.cfg.device, verbose=False)
        output = []
        for result in results:
            people = []
            if result.boxes is not None and len(result.boxes) and result.keypoints is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                kps = result.keypoints.data.cpu().numpy()
                for i in np.argsort(-confs, kind="stable"):
                    keypoints = tuple(tuple(float(v) for v in kp) for kp in kps[i])
                    people.append((to_bbox(boxes[i]), float(confs[i]), keypoints))
            output.append((people, float(result.speed.get("inference", 0.0))))
        return output

    @staticmethod
    def as_detections(people: list[tuple], first_id: int) -> list[Detection]:
        """Chuyển người pose thành Detection lớp ``pose_person`` với det_id tiếp nối."""
        return [
            Detection(det_id=first_id + i, class_id=POSE_CLASS_ID, class_name=POSE_PERSON,
                      confidence=conf, bbox=bbox, source="pose", keypoints=keypoints)
            for i, (bbox, conf, keypoints) in enumerate(people)
        ]
