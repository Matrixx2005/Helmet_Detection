"""Tạo bản val + test của Construction-PPE (bản cũ) với thứ tự lớp của bản mới (Ultralytics, 11/2026).

    python scripts/convert_labels_new_scheme.py                                  # -> data/ppe-eval
    python scripts/convert_labels_new_scheme.py --src data/construction-ppe --dest data/ppe-eval

Bộ nguồn KHÔNG bị sửa: ảnh được sao chép, nhãn được ghi sang thư mục đích với id đã đổi.
Ánh xạ id cũ -> mới:  0..4 giữ nguyên, 5 (none) bỏ, 6 Person->5, 7 no_helmet->6, 8 no_goggle->7,
9 no_gloves->8, 10 no_boots->9. Chỉ gồm val và test (tập train không dùng để đánh giá).
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import yaml

OLD_TO_NEW = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9}
NEW_NAMES = {0: "helmet", 1: "gloves", 2: "vest", 3: "boots", 4: "goggles", 5: "Person",
             6: "no_helmet", 7: "no_goggle", 8: "no_gloves", 9: "no_boots", 10: "forklift"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=Path("data/construction-ppe"))
    ap.add_argument("--dest", type=Path, default=Path("data/ppe-eval"))
    args = ap.parse_args()

    old = yaml.safe_load((args.src / "data.yaml").read_text(encoding="utf-8"))["names"]
    if next(int(k) for k, v in old.items() if v == "Person") != 6:
        sys.exit("Bộ nguồn không theo thứ tự lớp cũ (Person != 6); không chuyển.")
    if args.dest.exists():
        sys.exit(f"{args.dest} đã tồn tại; xóa hoặc đổi --dest rồi chạy lại.")

    dropped = {}
    for split in ("val", "test"):
        (args.dest / "images" / split).mkdir(parents=True)
        (args.dest / "labels" / split).mkdir(parents=True)
        n = 0
        for img in sorted((args.src / "images" / split).iterdir()):
            shutil.copy2(img, args.dest / "images" / split / img.name)
            lbl = args.src / "labels" / split / f"{img.stem}.txt"
            out = []
            if lbl.is_file():
                for line in lbl.read_text(encoding="utf-8").splitlines():
                    p = line.split()
                    if len(p) >= 5 and int(float(p[0])) in OLD_TO_NEW:
                        out.append(" ".join([str(OLD_TO_NEW[int(float(p[0]))]), *p[1:5]]))
            (args.dest / "labels" / split / f"{img.stem}.txt").write_text(
                "\n".join(out) + ("\n" if out else ""), encoding="utf-8")
            n += 1
        dropped[split] = n
    (args.dest / "images" / "train").mkdir(parents=True)  # rỗng, chỉ để data.yaml hợp lệ
    spec = {"train": "images/train", "val": "images/val", "test": "images/test", "names": NEW_NAMES}
    (args.dest / "data.yaml").write_text(yaml.safe_dump(spec, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"Đã tạo {args.dest}: " + ", ".join(f"{s}={n} ảnh" for s, n in dropped.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
