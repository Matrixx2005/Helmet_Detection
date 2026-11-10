"""Đánh giá YOLO riêng từng nguồn ảnh của bộ gộp ``data/ppe-extended``.

    python scripts/eval_by_source.py [--model models/best.pt] [--dataset data/ppe-extended]

Đọc ``sources.csv`` (do merge_roboflow.py tạo), chia ảnh val/test theo nguồn rồi chạy
Ultralytics val cho từng nhóm. Chỉ báo cáo 3 lớp hệ thống dùng (helmet, Person, no_helmet).
Kết quả: ``reports/detection_eval_extended/by_source.csv``.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from helmet_ppe.reporting import read_csv, write_csv  # noqa: E402

CLASSES = {0: "helmet", 6: "Person", 7: "no_helmet"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=str(ROOT / "models" / "best.pt"))
    ap.add_argument("--dataset", type=Path, default=ROOT / "data" / "ppe-extended")
    ap.add_argument("--output", type=Path, default=ROOT / "reports" / "detection_eval_extended" / "by_source.csv")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    from ultralytics import YOLO  # import chậm

    dataset = args.dataset.resolve()
    spec = yaml.safe_load((dataset / "data.yaml").read_text(encoding="utf-8"))
    sources = read_csv(dataset / "sources.csv")
    model = YOLO(args.model)
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for split in ("val", "test"):
            for source in sorted({r["source"] for r in sources}):
                imgs = [dataset / "images" / split / r["image"] for r in sources
                        if r["split"] == split and r["source"] == source]
                if not imgs:
                    continue
                listing = Path(tmp) / f"{source}_{split}.txt"
                listing.write_text("\n".join(str(p) for p in imgs), encoding="utf-8")
                yaml_path = Path(tmp) / f"{source}_{split}.yaml"
                yaml_path.write_text(yaml.safe_dump({
                    "path": str(dataset), "train": str(listing), "val": str(listing),
                    "names": spec["names"]}, allow_unicode=True), encoding="utf-8")
                m = model.val(data=str(yaml_path), split="val", imgsz=args.imgsz, device=args.device,
                              plots=False, verbose=False, project=tmp, name=f"{source}_{split}", exist_ok=True)
                by_cls = {int(c): i for i, c in enumerate(m.box.ap_class_index)}
                for cid, cname in CLASSES.items():
                    if cid not in by_cls:
                        continue
                    i = by_cls[cid]
                    p, r, ap50, ap = m.box.class_result(i)
                    rows.append({"source": source, "split": split, "images": len(imgs), "class": cname,
                                 "instances": int(m.nt_per_class[cid]), "precision": round(float(p), 4),
                                 "recall": round(float(r), 4), "mAP50": round(float(ap50), 4),
                                 "mAP50-95": round(float(ap), 4)})
    for r in rows:
        print(r)
    write_csv(args.output, rows, ["source", "split", "images", "class", "instances",
                                  "precision", "recall", "mAP50", "mAP50-95"])
    print(f"-> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
