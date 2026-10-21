"""Xem nhanh kết quả YOLO pose (người + điểm khớp) trên một thư mục ảnh.

    python scripts/pose_preview.py [thu_muc_anh] [ten_mo_hinh_pose]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ultralytics import YOLO  # noqa: E402

from helmet_ppe.image_io import list_images, read_image, write_image  # noqa: E402


def main() -> int:
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "crowd_samples"
    weights = sys.argv[2] if len(sys.argv) > 2 else "yolo26n-pose.pt"
    out = ROOT / "outputs" / "pose_preview"
    model = YOLO(str(ROOT / "models" / weights) if (ROOT / "models" / weights).is_file() else weights)
    summary = {}
    for path in list_images(folder):
        result = model.predict(read_image(path), conf=0.25, verbose=False)[0]
        write_image(out / path.name, result.plot())
        kps = result.keypoints.data.cpu().numpy().round(1).tolist() if result.keypoints is not None else []
        summary[path.name] = {"persons": len(result.boxes),
                              "boxes": result.boxes.xyxy.cpu().numpy().round(1).tolist(),
                              "conf": result.boxes.conf.cpu().numpy().round(3).tolist(),
                              "keypoints": kps}
        print(path.name, "người:", len(result.boxes))
    (out / "pose.json").write_text(json.dumps(summary), encoding="utf-8")
    print("weights:", model.ckpt_path if hasattr(model, "ckpt_path") else weights)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
