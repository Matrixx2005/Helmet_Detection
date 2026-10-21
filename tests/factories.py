"""Hàm tạo dữ liệu giả cho kiểm thử (không cần mô hình YOLO)."""

from __future__ import annotations

from itertools import count
from pathlib import Path

import numpy as np

from helmet_ppe.config import PipelineConfig
from helmet_ppe.image_io import write_image
from helmet_ppe.pipeline import analyze_detections
from helmet_ppe.schemas import Detection, ImageResult

CLASS_IDS = {"helmet": 0, "Person": 6, "no_helmet": 7}


class Scene:
    """Xây dựng một "ảnh" bằng cách thêm các hộp phát hiện."""

    def __init__(self, width: int = 1000, height: int = 800) -> None:
        self.width, self.height = width, height
        self.detections: list[Detection] = []
        self._ids = count()

    def add(self, class_name: str, bbox, conf: float = 0.9) -> int:
        det_id = next(self._ids)
        self.detections.append(Detection(det_id, CLASS_IDS[class_name], class_name, conf,
                                         tuple(float(v) for v in bbox)))
        return det_id

    def person(self, x1, y1, x2, y2, conf=0.9) -> int:
        return self.add("Person", (x1, y1, x2, y2), conf)

    def helmet_on(self, person_box, conf=0.9, dx: float = 0.0) -> int:
        """Mũ đặt đúng vùng đầu của khung người (dx: lệch ngang theo pixel)."""
        x1, y1, x2, y2 = person_box
        w, h = x2 - x1, y2 - y1
        cx = (x1 + x2) / 2 + dx
        hw = 0.35 * w / 2
        return self.add("helmet", (cx - hw, y1 - 0.02 * h, cx + hw, y1 + 0.18 * h), conf)

    def no_helmet_on(self, person_box, conf=0.9) -> int:
        x1, y1, x2, y2 = person_box
        w, h = x2 - x1, y2 - y1
        cx = (x1 + x2) / 2
        hw = 0.3 * w / 2
        return self.add("no_helmet", (cx - hw, y1, cx + hw, y1 + 0.2 * h), conf)

    def analyze(self, name: str = "scene.jpg") -> ImageResult:
        return analyze_detections(self.detections, image_name=name, image_path=name,
                                  width=self.width, height=self.height)


class FakePipeline:
    """Thay thế HelmetPipeline: mỗi ảnh có 1 người đội mũ và 1 người không đội mũ."""

    cfg = PipelineConfig()

    def analyze_batch(self, images, paths):
        results = []
        for image, path in zip(images, paths):
            scene = Scene(image.shape[1], image.shape[0])
            left, right = (5, 10, 25, 60), (40, 10, 60, 60)
            scene.person(*left)
            scene.helmet_on(left)
            scene.person(*right)
            scene.no_helmet_on(right)
            results.append(analyze_detections(scene.detections, image_name=Path(path).name,
                                              image_path=str(path), width=image.shape[1],
                                              height=image.shape[0], inference_ms=5.0))
        return results

    def analyze(self, image, path):
        return self.analyze_batch([image], [path])[0]


def make_images(folder: Path, n: int) -> None:
    """Tạo n ảnh xám đơn giản trong thư mục."""
    for i in range(n):
        write_image(folder / f"img ({i + 1}).jpg", np.full((80, 100, 3), 120, np.uint8))
