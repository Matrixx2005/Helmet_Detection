"""Chia lại train/test của Construction-PPE theo tỉ lệ test:train = 1:4, GIỮ NGUYÊN tập val.

    python scripts/resplit_construction_ppe.py --dry-run     # chỉ thống kê
    python scripts/resplit_construction_ppe.py               # thực hiện (di chuyển file tại chỗ)

Gộp ảnh train + test rồi xáo ngẫu nhiên (seed cố định) và chia lại: test = 20% số ảnh gộp.
Mỗi ảnh đi cùng nhãn ``labels/<split>/<tên>.txt``. Nhãn thừa không có ảnh tương ứng được để nguyên.
Ghi lịch sử di chuyển ra ``reports/construction_ppe_resplit.csv`` để truy vết / hoàn tác.
Khôi phục bản gốc bằng cách giải nén lại construction-ppe.zip.

Lưu ý: ảnh test mới có thể là ảnh mô hình cũ đã học khi huấn luyện, nên kết quả đo trước đó
không so sánh được với tập test mới.
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from find_dataset_duplicates import MAX_DISTANCE, average_hash  # noqa: E402
from helmet_ppe.image_io import list_images  # noqa: E402
from helmet_ppe.reporting import write_csv  # noqa: E402


def grow_val(ds: Path, args) -> int:
    """Chuyển ngẫu nhiên ảnh từ train sang val (kèm nhãn) cho đến khi val có ``grow_val_to`` ảnh."""
    train, val = list_images(ds / "images" / "train"), list_images(ds / "images" / "val")
    need = args.grow_val_to - len(val)
    if need <= 0:
        print(f"Val đã có {len(val)} ảnh (>= {args.grow_val_to}); không làm gì.")
        return 0
    chosen = random.Random(args.seed).sample(train, need)
    test_n = len(list_images(ds / "images" / "test"))
    print(f"Trước: train {len(train)}, val {len(val)}, test {test_n}")
    print(f"Sau  : train {len(train) - need}, val {len(val) + need}, test {test_n}")
    if args.dry_run:
        return 0
    rows = []
    for img in chosen:
        shutil.move(str(img), str(ds / "images" / "val" / img.name))
        lbl = ds / "labels" / "train" / f"{img.stem}.txt"
        if lbl.is_file():
            shutil.move(str(lbl), str(ds / "labels" / "val" / lbl.name))
        rows.append({"image": img.name, "old_split": "train", "new_split": "val"})
    for cache in (ds / "labels").glob("*.cache"):
        cache.unlink()
    write_csv(ROOT / "reports" / "construction_ppe_resplit_val.csv", rows, ["image", "old_split", "new_split"])
    for s in ("train", "val", "test"):
        print(f"{s}: {len(list_images(ds / 'images' / s))} ảnh")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, default=ROOT / "data" / "construction-ppe")
    ap.add_argument("--test-fraction", type=float, default=0.2, help="Tỉ lệ test trong (train + test); 0,2 = 1:4")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grow-val-to", type=int, default=None,
                    help="Chỉ chuyển ảnh từ train sang val cho đến khi val đủ N ảnh (giữ nguyên test)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    ds = args.dataset
    if args.grow_val_to is not None:
        return grow_val(ds, args)
    pool = [(split, p) for split in ("train", "test") for p in list_images(ds / "images" / split)]
    rng = random.Random(args.seed)
    order = list(range(len(pool)))
    rng.shuffle(order)
    n_test = round(len(pool) * args.test_fraction)
    new_split = {i: ("test" if rank < n_test else "train") for rank, i in enumerate(order)}

    before = {s: sum(1 for sp, _ in pool if sp == s) for s in ("train", "test")}
    after = {s: sum(1 for v in new_split.values() if v == s) for s in ("train", "test")}
    val_n = len(list_images(ds / "images" / "val"))
    print(f"Trước: train {before['train']}, val {val_n}, test {before['test']}")
    print(f"Sau  : train {after['train']}, val {val_n}, test {after['test']}  "
          f"(test:train = 1:{after['train'] / after['test']:.2f})")
    moves = [(i, pool[i][0], new_split[i]) for i in range(len(pool)) if pool[i][0] != new_split[i]]
    print(f"Số ảnh đổi tập: {len(moves)}")

    # Kiểm tra gần trùng giữa train mới và test mới (chỉ để báo cáo)
    hashes = np.array([average_hash(p) for _, p in pool])
    tr = [i for i, s in new_split.items() if s == "train"]
    te = [i for i, s in new_split.items() if s == "test"]
    near = sum(int(((hashes[tr] != hashes[i]).sum(axis=1) <= MAX_DISTANCE).any()) for i in te)
    print(f"Ảnh test mới có bản gần trùng trong train mới: {near}/{len(te)}")
    if args.dry_run:
        return 0

    rows = []
    for i, old, new in moves:
        img = pool[i][1]
        shutil.move(str(img), str(ds / "images" / new / img.name))
        lbl = ds / "labels" / old / f"{img.stem}.txt"
        if lbl.is_file():
            shutil.move(str(lbl), str(ds / "labels" / new / lbl.name))
        rows.append({"image": img.name, "old_split": old, "new_split": new})
    for cache in (ds / "labels").glob("*.cache"):  # cache nhãn đã cũ
        cache.unlink()
    write_csv(ROOT / "reports" / "construction_ppe_resplit.csv", rows, ["image", "old_split", "new_split"])
    for s in ("train", "val", "test"):
        print(f"{s}: {len(list_images(ds / 'images' / s))} ảnh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
