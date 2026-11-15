"""Sửa tên lớp trong trọng số YOLO và loại bỏ optimizer (giảm dung lượng).

    python scripts/fix_model_names.py <best.pt vào> <best.pt ra>

Dùng khi mô hình được huấn luyện trên bộ Construction-PPE phiên bản mới (Ultralytics, 11/2026) nhưng
notebook gán tên lớp theo thứ tự cũ. Thứ tự đúng của bộ mới:
  0 helmet, 1 gloves, 2 vest, 3 boots, 4 goggles, 5 Person, 6 no_helmet, 7 no_goggle,
  8 no_gloves, 9 no_boots, 10 forklift   (nhãn lớp 11 = no_vest bị bỏ vì mô hình chỉ có 11 lớp)
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import torch

NAMES = {0: "helmet", 1: "gloves", 2: "vest", 3: "boots", 4: "goggles", 5: "Person",
         6: "no_helmet", 7: "no_goggle", 8: "no_gloves", 9: "no_boots", 10: "forklift"}


def main() -> int:
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    for key in ("model", "ema"):
        if ckpt.get(key) is not None:
            ckpt[key].names = dict(NAMES)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_pt = Path(tmp) / "fixed.pt"
        torch.save(ckpt, tmp_pt)
        from ultralytics.utils.torch_utils import strip_optimizer
        strip_optimizer(tmp_pt)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(tmp_pt, dst)
    print(f"Đã ghi {dst} ({dst.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
