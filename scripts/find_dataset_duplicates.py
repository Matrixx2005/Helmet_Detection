"""Kiểm tra ảnh trùng lặp (gần như giống hệt) bằng average hash 16×16.

    python scripts/find_dataset_duplicates.py [thu_muc_anh] [thu_muc_dataset]

1. Ảnh trong ``thu_muc_anh`` (mặc định data/raw) trùng với ảnh nào của Construction-PPE
   → reports/data_raw_duplicates.csv
2. Ảnh val/test của Construction-PPE có bản gần trùng trong train hay không (rò rỉ dữ liệu)
   → reports/dataset_split_overlap.csv

Khoảng cách Hamming ≤ 5/256 bit được coi là trùng.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from helmet_ppe.image_io import list_images, read_image  # noqa: E402
from helmet_ppe.reporting import write_csv  # noqa: E402

HASH_SIZE = 16
MAX_DISTANCE = 5


def average_hash(path: Path) -> np.ndarray:
    gray = cv2.cvtColor(read_image(path), cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (HASH_SIZE, HASH_SIZE), interpolation=cv2.INTER_AREA).astype(np.float32)
    return (small > small.mean()).flatten()


def main() -> int:
    images_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "raw"
    dataset_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "data" / "construction-ppe"
    refs = [(split, p) for split in ("train", "val", "test")
            for p in list_images(dataset_dir / "images" / split)]
    ref_hashes = np.array([average_hash(p) for _, p in refs])

    rows = []
    for path in list_images(images_dir):
        distances = (ref_hashes != average_hash(path)).sum(axis=1)
        best = int(distances.argmin())
        split, ref = refs[best]
        rows.append({"image_name": path.name, "closest_dataset_image": ref.name, "split": split,
                     "hamming_distance": int(distances[best]),
                     "duplicate": int(distances[best]) <= MAX_DISTANCE})
    out = write_csv(ROOT / "reports" / "data_raw_duplicates.csv", rows, list(rows[0].keys()))
    dup = [r for r in rows if r["duplicate"]]
    by_split = {s: sum(r["split"] == s for r in dup) for s in ("train", "val", "test")}
    print(f"{len(dup)}/{len(rows)} ảnh trùng với Construction-PPE {by_split} → {out}")
    check_split_overlap(refs, ref_hashes)
    return 0


def check_split_overlap(refs: list[tuple[str, Path]], hashes: np.ndarray) -> None:
    """Đếm ảnh val/test có bản gần trùng trong tập train."""
    train_idx = [i for i, (split, _) in enumerate(refs) if split == "train"]
    rows = []
    for i, (split, path) in enumerate(refs):
        if split == "train":
            continue
        distances = (hashes[train_idx] != hashes[i]).sum(axis=1)
        best = int(distances.argmin())
        if distances[best] <= MAX_DISTANCE:
            rows.append({"split": split, "image_name": path.name,
                         "train_image": refs[train_idx[best]][1].name,
                         "hamming_distance": int(distances[best])})
    fields = ["split", "image_name", "train_image", "hamming_distance"]
    out = write_csv(ROOT / "reports" / "dataset_split_overlap.csv", rows, fields)
    for split in ("val", "test"):
        total = sum(s == split for s, _ in refs)
        print(f"{split}: {sum(r['split'] == split for r in rows)}/{total} ảnh gần trùng ảnh train")
    print(f"→ {out}")


if __name__ == "__main__":
    raise SystemExit(main())
