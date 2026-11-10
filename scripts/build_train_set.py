"""Ghép tập huấn luyện: train của Construction-PPE + train của dataset Roboflow.

    python scripts/build_train_set.py data/roboflow_bac_hien            # tạo data/ppe-train-combined
    python scripts/build_train_set.py data/roboflow_bac_hien --zip      # + nén để tải lên Colab
    python scripts/build_train_set.py data/roboflow_bac_hien --dry-run  # chỉ thống kê

Nguyên tắc chống rò rỉ dữ liệu:
* Ảnh Roboflow nào trùng / gần trùng (average hash) với BẤT KỲ ảnh val hoặc test trong
  ``data/ppe-extended`` đều bị loại khỏi train. Ngưỡng ``--threshold`` rộng hơn ngưỡng
  khử trùng thông thường vì dataset Roboflow thường là các khung hình liên tiếp của video.
* Tập val của bộ mới chính là val của ``data/ppe-extended`` (dùng để chọn checkpoint tốt nhất).
  Tập test KHÔNG được đưa vào gói huấn luyện.

Kết quả: ``data/ppe-train-combined/`` (images|labels × train|val, data.yaml 11 lớp, không có
khóa ``path`` để chạy được ở mọi máy, kể cả Colab).
"""

from __future__ import annotations

import argparse
import collections
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from find_dataset_duplicates import average_hash  # noqa: E402
from helmet_ppe.image_io import list_images  # noqa: E402
from helmet_ppe.reporting import write_csv  # noqa: E402
from merge_roboflow import build_class_map, remap_label  # noqa: E402


def count_boxes(label_files, names: dict[int, str]) -> dict[str, int]:
    c: collections.Counter = collections.Counter()
    for f in label_files:
        if f.is_file():
            for line in f.read_text(encoding="utf-8").splitlines():
                if line.split():
                    c[int(float(line.split()[0]))] += 1
    return {names[k]: v for k, v in sorted(c.items())}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roboflow", type=Path, help="Thư mục Roboflow đã giải nén (có data.yaml)")
    ap.add_argument("--base", type=Path, default=ROOT / "data" / "construction-ppe")
    ap.add_argument("--heldout", type=Path, default=ROOT / "data" / "ppe-extended",
                    help="Bộ val/test cố định (đầu ra của merge_roboflow.py)")
    ap.add_argument("--dest", type=Path, default=ROOT / "data" / "ppe-train-combined")
    ap.add_argument("--rf-splits", nargs="+", default=["train"], help="Các tập Roboflow dùng để huấn luyện")
    ap.add_argument("--class-map", nargs="*", default=[], metavar="TEN=LOP")
    ap.add_argument("--threshold", type=int, default=12, help="Khoảng cách Hamming (/256) coi là gần trùng val/test")
    ap.add_argument("--max-rf", type=int, default=4000,
                    help="Số ảnh Roboflow tối đa (0 = không giới hạn). Giữ ảnh có no_helmet trước, còn lại lấy ngẫu nhiên")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--keep-empty", action="store_true", help="Giữ cả ảnh Roboflow không có nhãn lớp nào dùng được")
    ap.add_argument("--zip", action="store_true", help="Nén thành <dest>.zip")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    base_spec = yaml.safe_load((args.base / "data.yaml").read_text(encoding="utf-8"))
    names = {int(k): str(v) for k, v in base_spec["names"].items()}
    target = {v: k for k, v in names.items()}
    rf_spec = yaml.safe_load((args.roboflow / "data.yaml").read_text(encoding="utf-8"))
    rf_names = rf_spec["names"]
    rf_names = dict(enumerate(rf_names)) if isinstance(rf_names, list) else {int(k): v for k, v in rf_names.items()}
    print(f"Lớp Roboflow: {rf_names}")
    mapping = build_class_map(rf_names, target, dict(kv.split("=", 1) for kv in args.class_map))
    if not mapping:
        sys.exit("Không ánh xạ được lớp nào.")

    heldout = [p for s in ("val", "test") for p in list_images(args.heldout / "images" / s)]
    if not heldout:
        sys.exit(f"{args.heldout} chưa có ảnh val/test; chạy merge_roboflow.py trước.")
    print(f"Tính hash {len(heldout)} ảnh val/test cần bảo vệ...")
    held_hashes = np.array([average_hash(p) for p in heldout])

    kept, excluded, empty = [], [], 0
    for rf_split in args.rf_splits:
        imgs = list_images(args.roboflow / rf_split / "images")
        print(f"Quét {len(imgs)} ảnh Roboflow/{rf_split}...")
        for n, img in enumerate(imgs, 1):
            if n % 2000 == 0:
                print(f"  {n}/{len(imgs)}")
            lines = remap_label(args.roboflow / rf_split / "labels" / f"{img.stem}.txt", mapping)
            if not lines and not args.keep_empty:
                empty += 1
                continue
            dist = (held_hashes != average_hash(img)).sum(axis=1)
            j = int(dist.argmin())
            if dist[j] <= args.threshold:
                excluded.append({"image": f"{rf_split}/{img.name}", "giong_voi": heldout[j].name,
                                 "hamming": int(dist[j])})
                continue
            kept.append((img, rf_split, lines))

    if args.max_rf and len(kept) > args.max_rf:
        rng = np.random.default_rng(args.seed)
        nh = str(target["no_helmet"])
        pref = [k for k in kept if any(l.split()[0] == nh for l in k[2])]
        rest = [k for k in kept if not any(l.split()[0] == nh for l in k[2])]
        pref = [pref[i] for i in rng.permutation(len(pref))][: args.max_rf]
        rest = [rest[i] for i in rng.permutation(len(rest))][: args.max_rf - len(pref)]
        print(f"Giới hạn --max-rf {args.max_rf}: giữ {len(pref)} ảnh có no_helmet + {len(rest)} ảnh khác "
              f"(bỏ {len(kept) - len(pref) - len(rest)})")
        kept = pref + rest

    base_train = list_images(args.base / "images" / "train")
    base_lbl = [args.base / "labels" / "train" / f"{p.stem}.txt" for p in base_train]
    print(f"\nConstruction-PPE train: {len(base_train)} ảnh | hộp: {count_boxes(base_lbl, names)}")
    kept_counter: collections.Counter = collections.Counter(int(l.split()[0]) for _, _, ls in kept for l in ls)
    print(f"Roboflow giữ lại: {len(kept)} ảnh | hộp: { {names[k]: v for k, v in sorted(kept_counter.items())} }")
    print(f"Roboflow bị loại: {len(excluded)} ảnh gần trùng val/test, {empty} ảnh không có nhãn dùng được")
    total = collections.Counter(count_boxes(base_lbl, names)) + collections.Counter({names[k]: v for k, v in kept_counter.items()})
    print(f"TỔNG train: {len(base_train) + len(kept)} ảnh | hộp: {dict(total)}")
    if args.dry_run:
        return 0

    if args.dest.exists():
        sys.exit(f"{args.dest} đã tồn tại; xóa hoặc đổi --dest rồi chạy lại.")
    for sub in ("images/train", "labels/train", "images/val", "labels/val"):
        (args.dest / sub).mkdir(parents=True)
    sources = []
    for img, lbl in zip(base_train, base_lbl):
        shutil.copy2(img, args.dest / "images" / "train" / img.name)
        if lbl.is_file():
            shutil.copy2(lbl, args.dest / "labels" / "train" / lbl.name)
        sources.append({"split": "train", "image": img.name, "source": "construction-ppe"})
    tag = args.roboflow.name
    for img, rf_split, lines in kept:
        new = f"{tag}_{rf_split}_{img.name}"
        shutil.copy2(img, args.dest / "images" / "train" / new)
        (args.dest / "labels" / "train" / f"{Path(new).stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        sources.append({"split": "train", "image": new, "source": tag})
    for img in list_images(args.heldout / "images" / "val"):
        shutil.copy2(img, args.dest / "images" / "val" / img.name)
        lbl = args.heldout / "labels" / "val" / f"{img.stem}.txt"
        if lbl.is_file():
            shutil.copy2(lbl, args.dest / "labels" / "val" / lbl.name)

    spec = {"train": "images/train", "val": "images/val", "names": {k: v for k, v in names.items()}}
    (args.dest / "data.yaml").write_text(yaml.safe_dump(spec, allow_unicode=True, sort_keys=False), encoding="utf-8")
    write_csv(args.dest / "sources.csv", sources, ["split", "image", "source"])
    if excluded:
        write_csv(ROOT / "reports" / "build_train_set_excluded.csv", excluded, ["image", "giong_voi", "hamming"])
    print(f"Đã tạo {args.dest}")

    if args.zip:
        out = args.dest.with_suffix(".zip")
        with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as z:  # ảnh đã nén sẵn, không cần nén thêm
            for f in sorted(args.dest.rglob("*")):
                if f.is_file():
                    z.write(f, f.relative_to(args.dest.parent))
        print(f"Đã nén -> {out} ({out.stat().st_size / 1e6:.0f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
