"""Nhãn chuẩn cấp người (person-level ground truth).

Định dạng CSV (có thể tự gán nhãn tay bằng Excel)::

    image_name, gt_person_id, x1, y1, x2, y2, gt_status, source[, head_x1, head_y1, head_x2, head_y2]

``gt_status`` ∈ {HELMET, NO_HELMET, UNDETERMINED}. Tọa độ theo pixel. Các cột
``head_*`` (hộp mũ / đầu trần của người đó) là tùy chọn, dùng để ghép người theo
đầu khi đánh giá.

Bộ Construction-PPE chỉ có nhãn *hộp* (Person / helmet / no_helmet), không có
nhãn "người này có đội mũ không". Hàm :func:`derive_person_gt` suy ra nhãn
cấp người từ nhãn hộp chuẩn bằng **cùng quy tắc ghép hình học** của hệ thống
(với confidence = 1). Vì vậy đánh giá trên nhãn suy ra này đo được *ảnh
hưởng của lỗi phát hiện YOLO lên kết luận cuối cùng*, chứ không kiểm chứng
độc lập quy tắc ghép. Để kiểm chứng độc lập, hãy sửa tay cột ``gt_status``
và đặt ``source = manual``.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from helmet_ppe.config import PipelineConfig, Status
from helmet_ppe.image_io import list_images
from helmet_ppe.pipeline import analyze_detections
from helmet_ppe.reporting import read_csv, write_csv
from helmet_ppe.schemas import Detection

GT_FIELDS = ["image_name", "gt_person_id", "x1", "y1", "x2", "y2", "gt_status", "source",
             "head_x1", "head_y1", "head_x2", "head_y2"]
REQUIRED_GT_FIELDS = GT_FIELDS[:7]
GT_HELMET, GT_NO_HELMET, GT_UNDETERMINED = "HELMET", "NO_HELMET", "UNDETERMINED"
GT_STATUSES = (GT_HELMET, GT_NO_HELMET, GT_UNDETERMINED)

_STATUS_TO_GT = {
    Status.HELMET_DETECTED: GT_HELMET,
    Status.SUSPECTED_NO_HELMET: GT_NO_HELMET,
    Status.UNKNOWN: GT_UNDETERMINED,
}


def load_yolo_labels(label_path: Path, img_w: int, img_h: int,
                     names: dict[int, str]) -> list[Detection]:
    """Đọc tệp nhãn YOLO (class cx cy w h, chuẩn hóa) thành Detection pixel."""
    detections: list[Detection] = []
    if not label_path.is_file():
        return detections
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        cx, cy, w, h = (float(v) for v in parts[1:5])
        bbox = ((cx - w / 2) * img_w, (cy - h / 2) * img_h,
                (cx + w / 2) * img_w, (cy + h / 2) * img_h)
        detections.append(Detection(det_id=len(detections), class_id=cls,
                                    class_name=names.get(cls, str(cls)),
                                    confidence=1.0, bbox=bbox))
    return detections


def derive_person_gt(images_dir: Path, labels_dir: Path, names: dict[int, str],
                     cfg: PipelineConfig | None = None) -> list[dict]:
    """Suy ra nhãn cấp người từ nhãn hộp chuẩn của một split."""
    rows: list[dict] = []
    for image_path in list_images(images_dir):
        with Image.open(image_path) as img:
            width, height = img.size
        labels = load_yolo_labels(labels_dir / f"{image_path.stem}.txt", width, height, names)
        result = analyze_detections(labels, image_name=image_path.name,
                                    image_path=str(image_path), width=width, height=height,
                                    cfg=cfg)
        for person in result.persons:
            x1, y1, x2, y2 = person.bbox
            head = person.head_box or ("", "", "", "")
            rows.append({
                "image_name": image_path.name, "gt_person_id": person.person_id,
                "x1": round(x1, 1), "y1": round(y1, 1), "x2": round(x2, 1), "y2": round(y2, 1),
                "gt_status": _STATUS_TO_GT[person.status],
                "source": "derived_from_box_labels",
                **{f"head_{k}": ("" if v == "" else round(v, 1))
                   for k, v in zip(("x1", "y1", "x2", "y2"), head)},
            })
    return rows


def write_person_gt(path: Path, rows: list[dict]) -> Path:
    return write_csv(path, rows, GT_FIELDS)


def read_person_gt(path: Path) -> list[dict]:
    """Đọc và kiểm tra CSV nhãn cấp người."""
    rows = read_csv(path)
    for i, row in enumerate(rows, start=2):
        missing = [f for f in REQUIRED_GT_FIELDS if f not in row]
        if missing:
            raise ValueError(f"{path}: thiếu cột {missing}")
        if row["gt_status"] not in GT_STATUSES:
            raise ValueError(f"{path} dòng {i}: gt_status không hợp lệ '{row['gt_status']}'")
    return rows
