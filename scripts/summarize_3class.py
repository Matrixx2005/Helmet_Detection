"""Tổng hợp kết quả đánh giá chỉ cho 3 lớp hệ thống dùng: helmet, Person, no_helmet.

    python scripts/summarize_3class.py [thu_muc_ket_qua]      # mặc định reports/detection_eval300

Đọc ``<thư mục>/<mô hình>/per_class_*_{val,test}.csv`` (do ``eval-detector`` sinh ra) và tính mAP trung bình
(macro) trên 3 lớp. Số "all" của Ultralytics tính trên cả 11 lớp nên bị kéo xuống khi ảnh nguồn Roboflow
không có nhãn cho 8 lớp phụ; số 3 lớp này mới phản ánh đúng mục tiêu của đề tài.
Kết quả: ``<thư mục>/summary_3class.csv``.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from helmet_ppe.reporting import write_csv  # noqa: E402

CLASSES = ("helmet", "Person", "no_helmet")


def main() -> int:
    base = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "reports" / "detection_eval300"
    rows = []
    for f in sorted(base.glob("*/per_class_*_*.csv")):
        model_dir, split = f.parent.name, f.stem.rsplit("_", 1)[1]
        per = {r["class_name"]: r for r in csv.DictReader(open(f, encoding="utf-8-sig"))}
        if not all(c in per for c in CLASSES):
            continue
        avg = lambda key: sum(float(per[c][key]) for c in CLASSES) / len(CLASSES)
        rows.append({
            "model": model_dir, "split": split,
            "mAP50_helmet": per["helmet"]["mAP50"], "mAP50_Person": per["Person"]["mAP50"],
            "mAP50_no_helmet": per["no_helmet"]["mAP50"],
            "mAP50_3class": round(avg("mAP50"), 4), "mAP50-95_3class": round(avg("mAP50-95"), 4),
            "precision_3class": round(avg("precision"), 4), "recall_3class": round(avg("recall"), 4),
        })
    fields = list(rows[0]) if rows else []
    out = write_csv(base / "summary_3class.csv", rows, fields)
    for r in rows:
        print(f"{r['model']:10s} {r['split']:5s} mAP50: helmet={r['mAP50_helmet']} Person={r['mAP50_Person']} "
              f"no_helmet={r['mAP50_no_helmet']} | TB 3 lop: mAP50={r['mAP50_3class']} mAP50-95={r['mAP50-95_3class']}")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
