"""Huấn luyện (fine-tune) YOLO26n trên bộ dữ liệu Construction-PPE.

Tái lập lần huấn luyện đã tạo ``models/best_v2.pt``. Các siêu tham số lấy từ
``train_args`` lưu trong chính tệp trọng số (xem docs/05_experiments.md).
Ban đầu mô hình được huấn luyện trên Google Colab (GPU); script này chạy được cả
trên Colab lẫn máy cá nhân.

Cách chạy:
    pip install ultralytics
    python scripts/train.py                      # 100 epoch, batch 16
    python scripts/train.py --epochs 10 --batch 8  # cấu hình của models/best.pt
    python scripts/train.py --data data/construction-ppe/data.yaml --device cpu
"""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", default="yolo26n.pt", help="Trọng số tiền huấn luyện để fine-tune")
    p.add_argument(
        "--data",
        default="construction-ppe.yaml",
        help="data.yaml; mặc định dùng cấu hình có sẵn của Ultralytics (tự tải dataset)",
    )
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="0", help="'0' (GPU), 'cpu', ...")
    p.add_argument("--project", default="runs", help="Thư mục lưu kết quả")
    p.add_argument("--name", default="helmet_yolo26n_v2")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    from ultralytics import YOLO  # import chậm: chỉ khi thật sự huấn luyện

    model = YOLO(args.base)
    model.train(
        data=args.data,
        epochs=args.epochs,
        patience=args.patience,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project=args.project,
        name=args.name,
        optimizer="AdamW",
        lr0=0.000667,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        close_mosaic=10,
        seed=0,
        deterministic=True,
        amp=True,
        workers=8,
    )
    metrics = model.val()
    print(f"mAP50={metrics.box.map50:.3f}  mAP50-95={metrics.box.map:.3f}")


if __name__ == "__main__":
    main()
