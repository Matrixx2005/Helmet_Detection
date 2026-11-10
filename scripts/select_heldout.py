"""Chọn val/test ~300 ảnh: giữ NGUYÊN val/test Construction-PPE, bổ sung ảnh Roboflow chất lượng tốt nhất.

    python scripts/select_heldout.py                 # data/ppe-extended -> data/ppe-eval300
    python scripts/select_heldout.py --target 300 --dry-run

Ứng viên Roboflow chỉ lấy trong ``data/ppe-extended`` (đã lọc không trùng / gần trùng với tập
train), nên val/test mới vẫn tách khỏi ``data/ppe-train-combined`` mà không cần huấn luyện lại.

Tiêu chí chọn (KHÔNG dựa trên kết quả của mô hình nào, để không làm lệch phép đo):
* ảnh sắc nét (phương sai Laplacian cao);
* có người đủ lớn (chiều cao Person trung vị so với ảnh), không quá đông (số Person trong ngưỡng);
* mũ / đầu trần không quá nhỏ;
* đa dạng: bỏ ảnh gần giống ảnh đã chọn (average hash);
* giữ tỉ lệ ảnh có no_helmet >= ``--nohelmet-frac`` để lớp này đo được.
Lưu ý: chọn ảnh rõ nét làm tập dễ hơn ảnh thực tế; cần nêu điều này trong báo cáo.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from find_dataset_duplicates import average_hash  # noqa: E402
from helmet_ppe.image_io import read_image  # noqa: E402
from helmet_ppe.reporting import read_csv, write_csv  # noqa: E402

HELMET, PERSON, NO_HELMET = 0, 6, 7


def read_boxes(path: Path) -> list[tuple[int, float, float, float, float]]:
    out = []
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            p = line.split()
            if len(p) >= 5:
                out.append((int(float(p[0])), *map(float, p[1:5])))  # cls, cx, cy, w, h
    return out


def analyse(img_path: Path, lbl_path: Path):
    boxes = read_boxes(lbl_path)
    persons = [b for b in boxes if b[0] == PERSON]
    heads = [b for b in boxes if b[0] in (HELMET, NO_HELMET)]
    if not persons or not heads:
        return None
    image = read_image(img_path)
    h, w = image.shape[:2]
    gray = cv2.cvtColor(cv2.resize(image, (640, max(1, int(640 * h / w)))), cv2.COLOR_BGR2GRAY)
    return {
        "sharp": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "person_h": float(np.median([b[4] for b in persons])),
        "n_person": len(persons),
        "min_head_w": min(b[3] for b in heads),
        "has_no_helmet": any(b[0] == NO_HELMET for b in boxes),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=ROOT / "data" / "ppe-extended")
    ap.add_argument("--dest", type=Path, default=ROOT / "data" / "ppe-eval300")
    ap.add_argument("--target", type=int, default=300, help="Số ảnh mỗi tập val/test")
    ap.add_argument("--nohelmet-frac", type=float, default=0.35, help="Tỉ lệ tối thiểu ảnh Roboflow có no_helmet")
    ap.add_argument("--min-person-h", type=float, default=0.12)
    ap.add_argument("--min-head-w", type=float, default=0.02)
    ap.add_argument("--max-persons", type=int, default=8)
    ap.add_argument("--dup-distance", type=int, default=20, help="Hamming (/256) tối thiểu giữa các ảnh được chọn")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    sources = read_csv(args.src / "sources.csv")
    spec = yaml.safe_load((args.src / "data.yaml").read_text(encoding="utf-8"))
    selected: dict[str, list[tuple[str, str]]] = {}
    for split in ("val", "test"):
        keep = [r["image"] for r in sources if r["split"] == split and r["source"] == "construction-ppe"]
        cands = [r for r in sources if r["split"] == split and r["source"] != "construction-ppe"]
        need = max(0, args.target - len(keep))
        print(f"\n[{split}] giữ {len(keep)} ảnh Construction-PPE, cần bổ sung {need} ảnh từ {len(cands)} ứng viên")

        stats = []
        for r in cands:
            img = args.src / "images" / split / r["image"]
            a = analyse(img, args.src / "labels" / split / f"{Path(r['image']).stem}.txt")
            if a and a["person_h"] >= args.min_person_h and a["min_head_w"] >= args.min_head_w \
                    and a["n_person"] <= args.max_persons:
                stats.append((r, img, a))
        print(f"  {len(stats)}/{len(cands)} ứng viên đạt ngưỡng chất lượng")
        if not stats:
            sys.exit("Không còn ứng viên; nới --min-person-h / --min-head-w / --max-persons.")

        # điểm = trung bình hạng của độ nét và kích thước người (hạng cao = tốt hơn)
        sharp_rank = np.argsort(np.argsort([s[2]["sharp"] for s in stats])) / max(1, len(stats) - 1)
        size_rank = np.argsort(np.argsort([s[2]["person_h"] for s in stats])) / max(1, len(stats) - 1)
        order = np.argsort(-(0.6 * sharp_rank + 0.4 * size_rank))

        # hash các ảnh đã giữ để chọn đa dạng
        chosen_hashes = [average_hash(args.src / "images" / split / n) for n in keep]
        picked, nh_count = [], 0
        quota_nh = int(need * args.nohelmet_frac)
        for pass_nh in (True, False):  # lượt 1: ưu tiên ảnh có no_helmet cho đủ quota; lượt 2: lấp phần còn lại
            for i in order:
                if len(picked) >= need:
                    break
                r, img, a = stats[i]
                if r["image"] in {p[0] for p in picked}:
                    continue
                if pass_nh and (nh_count >= quota_nh or not a["has_no_helmet"]):
                    continue
                h = average_hash(img)
                if chosen_hashes and min((h != c).sum() for c in chosen_hashes) < args.dup_distance:
                    continue
                picked.append((r["image"], r["source"]))
                chosen_hashes.append(h)
                nh_count += a["has_no_helmet"]
        print(f"  chọn {len(picked)}/{need} ảnh Roboflow ({nh_count} ảnh có no_helmet)")
        selected[split] = [(n, "construction-ppe") for n in keep] + picked

    if args.dry_run:
        return 0
    if args.dest.exists():
        sys.exit(f"{args.dest} đã tồn tại; xóa hoặc đổi --dest rồi chạy lại.")
    rows = []
    for split, items in selected.items():
        (args.dest / "images" / split).mkdir(parents=True)
        (args.dest / "labels" / split).mkdir(parents=True)
        for name, src in items:
            shutil.copy2(args.src / "images" / split / name, args.dest / "images" / split / name)
            lbl = args.src / "labels" / split / f"{Path(name).stem}.txt"
            if lbl.is_file():
                shutil.copy2(lbl, args.dest / "labels" / split / lbl.name)
            rows.append({"split": split, "image": name, "source": src})
    (args.dest / "images" / "train").mkdir(parents=True)  # rỗng, chỉ để data.yaml hợp lệ
    new_spec = dict(spec)
    new_spec["path"] = str(args.dest.resolve())
    (args.dest / "data.yaml").write_text(yaml.safe_dump(new_spec, allow_unicode=True, sort_keys=False), encoding="utf-8")
    write_csv(args.dest / "sources.csv", rows, ["split", "image", "source"])
    print(f"\nĐã tạo {args.dest}: " + ", ".join(f"{s}={len(v)}" for s, v in selected.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
