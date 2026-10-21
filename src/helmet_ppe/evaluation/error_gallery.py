"""Tạo ảnh minh họa các ca sai để phân tích lỗi trong báo cáo.

Ảnh dự đoán (đã vẽ) được phủ thêm khung người chuẩn màu trắng kèm nhãn chuẩn,
giúp so sánh trực quan "hệ thống thấy gì" với "đáp án là gì".
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import cv2

from helmet_ppe.image_io import read_image, write_image

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


def build_gallery(cases: list[dict], gt_rows: list[dict], annotated_dir: Path,
                  out_dir: Path, max_per_type: int = 6) -> list[Path]:
    """Lưu tối đa ``max_per_type`` ảnh cho mỗi loại lỗi."""
    gt_by_image: dict[str, list[dict]] = defaultdict(list)
    for row in gt_rows:
        gt_by_image[row["image_name"]].append(row)

    seen: dict[str, set[str]] = defaultdict(set)
    saved: list[Path] = []
    for case in cases:
        kind, image_name = case["error_type"], case["image_name"]
        if image_name in seen[kind] or len(seen[kind]) >= max_per_type:
            continue
        source = Path(annotated_dir) / image_name
        if not source.is_file():
            continue
        canvas = read_image(source)
        for gt in gt_by_image.get(image_name, []):
            x1, y1, x2, y2 = (int(float(gt[k])) for k in ("x1", "y1", "x2", "y2"))
            cv2.rectangle(canvas, (x1, y1), (x2, y2), BLACK, 4)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), WHITE, 2)
            cv2.putText(canvas, f"GT {gt['gt_status']}", (x1 + 3, max(14, y1 + 16)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLACK, 3, cv2.LINE_AA)
            cv2.putText(canvas, f"GT {gt['gt_status']}", (x1 + 3, max(14, y1 + 16)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)
        target = Path(out_dir) / kind / image_name
        write_image(target, canvas)
        seen[kind].add(image_name)
        saved.append(target)
    return saved
