"""Gộp dataset Roboflow (định dạng YOLO) vào val/test của Construction-PPE.

    python scripts/merge_roboflow.py data/roboflow_hardhat
    python scripts/merge_roboflow.py data/roboflow_hardhat --class-map head=no_helmet --dry-run

Việc script làm:
1. Đọc ``data.yaml`` của Roboflow, ánh xạ tên lớp về bộ 11 lớp của mô hình
   (mặc định ``head -> no_helmet``, ``person -> Person``, ``hardhat/helmet -> helmet``).
2. Bỏ ảnh gần trùng (average hash) với train của Construction-PPE, với val/test gốc,
   và với ảnh đã nhận trước đó, để không rò rỉ dữ liệu.
3. Tạo bộ mới ``data/ppe-extended`` (val + test = gốc + Roboflow). Bộ Construction-PPE
   gốc không bị sửa. Ghi ``sources.csv`` để báo cáo riêng từng nguồn.

Đánh giá sau đó:  python -m helmet_ppe eval-detector --dataset data/ppe-extended
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from find_dataset_duplicates import MAX_DISTANCE, average_hash  # noqa: E402
from helmet_ppe.image_io import list_images  # noqa: E402
from helmet_ppe.reporting import write_csv  # noqa: E402

DEFAULT_ALIASES = {
    "head": "no_helmet", "nohelmet": "no_helmet", "nohardhat": "no_helmet", "bare": "no_helmet",
    "person": "Person", "worker": "Person",
    "helmet": "helmet", "hardhat": "helmet", "hat": "helmet",
}
# split của Roboflow -> split của dự án (mặc định); đổi bằng --split-map
SPLIT_MAP = {"valid": "val", "test": "test"}


def norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def build_class_map(rf_names: dict[int, str], target: dict[str, int], overrides: dict[str, str]) -> dict[int, int]:
    aliases = dict(DEFAULT_ALIASES)
    aliases.update({norm(k): v for k, v in overrides.items()})
    mapping = {}
    for idx, name in rf_names.items():
        dest = aliases.get(norm(name))
        if dest is None:
            print(f"  lớp '{name}' (id {idx}) không có trong ánh xạ -> bỏ qua")
        elif dest not in target:
            sys.exit(f"Lớp đích '{dest}' không thuộc bộ lớp của mô hình: {list(target)}")
        else:
            mapping[idx] = target[dest]
            print(f"  '{name}' ({idx}) -> {dest} ({target[dest]})")
    return mapping


def remap_label(src: Path, mapping: dict[int, int]) -> list[str]:
    lines = []
    if not src.is_file():
        return lines
    for line in src.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 5 and int(float(parts[0])) in mapping:
            lines.append(" ".join([str(mapping[int(float(parts[0]))]), *parts[1:5]]))
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roboflow", type=Path, help="Thư mục Roboflow đã giải nén (có data.yaml)")
    ap.add_argument("--base", type=Path, default=ROOT / "data" / "construction-ppe")
    ap.add_argument("--dest", type=Path, default=ROOT / "data" / "ppe-extended")
    ap.add_argument("--tag", default=None, help="Nhãn nguồn (mặc định: tên thư mục Roboflow)")
    ap.add_argument("--class-map", nargs="*", default=[], metavar="TEN=LOP", help="vd: head=no_helmet")
    ap.add_argument("--split-map", nargs="*", default=[], metavar="RF:DICH",
                    help="vd: valid:val test:test train:test (có thể lấy thêm ảnh train của Roboflow)")
    ap.add_argument("--limit", type=int, default=None, help="Số ảnh Roboflow tối đa nhận cho mỗi tập val/test")
    ap.add_argument("--prefer", default="no_helmet", help="Ưu tiên ảnh có lớp này khi dùng --limit")
    ap.add_argument("--prefer-frac", type=float, default=0.4,
                    help="Tỉ lệ tối thiểu ảnh có lớp ưu tiên trong số ảnh nhận (khi dùng --limit)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="Chỉ thống kê, không ghi gì")
    args = ap.parse_args()

    tag = args.tag or args.roboflow.name
    overrides = dict(kv.split("=", 1) for kv in args.class_map)
    base_spec = yaml.safe_load((args.base / "data.yaml").read_text(encoding="utf-8"))
    target = {str(v): int(k) for k, v in base_spec["names"].items()}
    rf_spec = yaml.safe_load((args.roboflow / "data.yaml").read_text(encoding="utf-8"))
    rf_names = rf_spec["names"]
    rf_names = dict(enumerate(rf_names)) if isinstance(rf_names, list) else {int(k): v for k, v in rf_names.items()}
    print(f"Lớp Roboflow: {rf_names}")
    mapping = build_class_map(rf_names, target, overrides)
    if not mapping:
        sys.exit("Không ánh xạ được lớp nào; dùng --class-map ten=lop.")

    print("Tính hash ảnh Construction-PPE...")
    base_images = {s: list_images(args.base / "images" / s) for s in ("train", "val", "test")}
    seen_names = [(s, p) for s, ps in base_images.items() for p in ps]
    seen_hashes = [average_hash(p) for _, p in seen_names]

    split_map = dict(SPLIT_MAP)
    if args.split_map:
        split_map = dict(kv.split(":", 1) for kv in args.split_map)
    prefer_id = target.get(args.prefer)

    # Gom ứng viên theo tập đích; sắp ảnh có lớp ưu tiên trước, còn lại xáo ngẫu nhiên (cố định seed).
    rng = np.random.default_rng(args.seed)
    cands: dict[str, list] = {"val": [], "test": []}
    for rf_split, split in split_map.items():
        img_dir = args.roboflow / rf_split / "images"
        if not img_dir.is_dir():
            continue
        for img in list_images(img_dir):
            lines = remap_label(args.roboflow / rf_split / "labels" / f"{img.stem}.txt", mapping)
            has_pref = any(l.split()[0] == str(prefer_id) for l in lines)
            cands[split].append((img, rf_split, lines, has_pref))
    kept, rows = [], []
    for split, items in cands.items():
        items = [items[i] for i in rng.permutation(len(items))]
        pref = [it for it in items if it[3]]
        rest = [it for it in items if not it[3]]
        quota = int(args.limit * args.prefer_frac) if args.limit else len(pref)
        # xen kẽ: trước hết đủ quota ảnh có lớp ưu tiên, phần còn lại trộn ngẫu nhiên
        head, tail = pref[:quota], pref[quota:] + rest
        tail = [tail[i] for i in rng.permutation(len(tail))]
        items = head + tail
        taken = 0
        for img, rf_split, lines, _ in items:
            if args.limit is not None and taken >= args.limit:
                break
            h = average_hash(img)
            dist = (np.array(seen_hashes) != h).sum(axis=1)
            j = int(dist.argmin())
            if dist[j] <= MAX_DISTANCE:
                rows.append({"image": f"{rf_split}/{img.name}", "split": split, "status": "bo_trung",
                             "trung_voi": f"{seen_names[j][0]}/{seen_names[j][1].name}"})
                continue
            kept.append((img, split, f"{tag}_{rf_split}_{img.name}", lines))
            seen_names.append((f"{tag}-{split}", img))
            seen_hashes.append(h)
            rows.append({"image": f"{rf_split}/{img.name}", "split": split, "status": "nhan", "trung_voi": ""})
            taken += 1

    n_drop = sum(r["status"] == "bo_trung" for r in rows)
    for s in ("val", "test"):
        print(f"{s}: +{sum(k[1] == s for k in kept)} ảnh Roboflow, bỏ {sum(r['split'] == s and r['status'] == 'bo_trung' for r in rows)} ảnh trùng")
    print(f"Tổng: nhận {len(kept)}, bỏ trùng {n_drop}")
    if args.dry_run:
        return 0

    if args.dest.exists():
        sys.exit(f"{args.dest} đã tồn tại; xóa hoặc đổi --dest rồi chạy lại.")
    sources = []
    for s in ("val", "test"):
        (args.dest / "images" / s).mkdir(parents=True)
        (args.dest / "labels" / s).mkdir(parents=True)
        for img in base_images[s]:
            shutil.copy2(img, args.dest / "images" / s / img.name)
            lbl = args.base / "labels" / s / f"{img.stem}.txt"
            if lbl.is_file():
                shutil.copy2(lbl, args.dest / "labels" / s / lbl.name)
            sources.append({"split": s, "image": img.name, "source": "construction-ppe"})
    (args.dest / "images" / "train").mkdir(parents=True)  # rỗng: chỉ để data.yaml hợp lệ
    for img, split, new_name, lines in kept:
        shutil.copy2(img, args.dest / "images" / split / new_name)
        (args.dest / "labels" / split / f"{Path(new_name).stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        sources.append({"split": split, "image": new_name, "source": tag})

    spec = dict(base_spec)
    spec.pop("download", None)
    spec["path"] = str(args.dest.resolve())
    spec["train"], spec["val"], spec["test"] = "images/train", "images/val", "images/test"
    (args.dest / "data.yaml").write_text(yaml.safe_dump(spec, allow_unicode=True, sort_keys=False), encoding="utf-8")
    write_csv(args.dest / "sources.csv", sources, ["split", "image", "source"])
    write_csv(ROOT / "reports" / f"merge_{tag}.csv", rows, ["image", "split", "status", "trung_voi"])
    print(f"Đã tạo {args.dest}. Đánh giá: python -m helmet_ppe eval-detector --dataset {args.dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
