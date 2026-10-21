"""Đánh giá mô hình YOLO trên tập có nhãn (Precision, Recall, mAP50, mAP50-95).

Dùng ``YOLO.val`` của Ultralytics trên split val/test của Construction-PPE.
Chỉ chạy khi có ``data.yaml`` và nhãn YOLO hợp lệ; nếu không, KHÔNG có số liệu
độ chính xác nào được tạo ra.
"""

from __future__ import annotations

import logging
import shutil
import zipfile
from pathlib import Path

import yaml

log = logging.getLogger(__name__)

FOCUS_CLASSES = ("Person", "helmet", "no_helmet")
COPIED_PLOTS = ("confusion_matrix.png", "confusion_matrix_normalized.png",
                "BoxPR_curve.png", "BoxF1_curve.png")


def prepare_dataset(zip_path: Path, dest_dir: Path) -> Path:
    """Giải nén bộ dữ liệu (nếu chưa) và tạo ``data.yaml`` trỏ đường dẫn tuyệt đối.

    Tệp zip gốc không bị thay đổi. Trả về đường dẫn ``data.yaml`` dùng để đánh giá.
    """
    dest_dir = Path(dest_dir)
    if not (dest_dir / "images").is_dir():
        if not Path(zip_path).is_file():
            raise FileNotFoundError(f"Không tìm thấy tệp dữ liệu: {zip_path}")
        log.info("Giải nén %s → %s", zip_path, dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(dest_dir)
    return write_eval_yaml(dest_dir)


def write_eval_yaml(dataset_dir: Path) -> Path:
    """Tạo ``data_local.yaml`` với ``path`` tuyệt đối, giữ nguyên danh sách lớp gốc."""
    dataset_dir = Path(dataset_dir).resolve()
    source = dataset_dir / "data.yaml"
    if not source.is_file():
        raise FileNotFoundError(f"Thiếu {source}: không đủ điều kiện tính mAP")
    spec = yaml.safe_load(source.read_text(encoding="utf-8"))
    for split in ("train", "val", "test"):
        if split in spec and not (dataset_dir / spec[split]).is_dir():
            raise FileNotFoundError(f"Thiếu thư mục ảnh {split}: {dataset_dir / spec[split]}")
    spec["path"] = str(dataset_dir)
    spec.pop("download", None)
    target = dataset_dir / "data_local.yaml"
    target.write_text(yaml.safe_dump(spec, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return target


def class_names(data_yaml: Path) -> dict[int, str]:
    spec = yaml.safe_load(Path(data_yaml).read_text(encoding="utf-8"))
    names = spec["names"]
    return dict(enumerate(names)) if isinstance(names, list) else {int(k): v for k, v in names.items()}


def validate_model(model_path: Path, data_yaml: Path, split: str, out_dir: Path, *,
                   imgsz: int = 640, batch: int = 16, device: str | None = None) -> dict:
    """Chạy YOLO val và trả về số liệu tổng + từng lớp (giá trị thực đo được)."""
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    run_name = f"{Path(model_path).stem}_{split}"
    metrics = model.val(data=str(data_yaml), split=split, imgsz=imgsz, batch=batch,
                        device=device, plots=True, project=str(Path(out_dir).resolve()),
                        name=run_name, exist_ok=True, verbose=False)
    box = metrics.box
    names = metrics.names
    instances = getattr(metrics, "nt_per_class", None)
    per_class = []
    for i, cls in enumerate(box.ap_class_index):
        p, r, ap50, ap = box.class_result(i)
        per_class.append({
            "class_id": int(cls), "class_name": names[int(cls)],
            "instances": int(instances[int(cls)]) if instances is not None else None,
            "precision": round(float(p), 4), "recall": round(float(r), 4),
            "mAP50": round(float(ap50), 4), "mAP50-95": round(float(ap), 4),
        })

    save_dir = Path(metrics.save_dir)
    return {
        "model": str(model_path),
        "model_name": Path(model_path).name,
        "split": split,
        "imgsz": imgsz,
        "num_images": len(list((Path(data_yaml).parent / "images" / split).glob("*"))),
        "precision": round(float(box.mp), 4),
        "recall": round(float(box.mr), 4),
        "mAP50": round(float(box.map50), 4),
        "mAP50-95": round(float(box.map), 4),
        "speed_ms": {k: round(float(v), 2) for k, v in metrics.speed.items()},
        "per_class": per_class,
        "plots_dir": str(save_dir),
        "confusion_matrix": metrics.confusion_matrix.matrix.astype(int).tolist(),
    }


def copy_plots(result: dict, dest_dir: Path) -> list[Path]:
    """Sao chép các biểu đồ Ultralytics vào thư mục báo cáo với tên rõ ràng."""
    copied = []
    src = Path(result["plots_dir"])
    prefix = f"{Path(result['model_name']).stem}_{result['split']}"
    for name in COPIED_PLOTS:
        if (src / name).is_file():
            target = Path(dest_dir) / f"{prefix}_{name}"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src / name, target)
            copied.append(target)
    return copied
