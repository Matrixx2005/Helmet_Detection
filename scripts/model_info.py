"""Thông tin các mô hình trong models/: kiến trúc, số tham số, GFLOPs, tham số huấn luyện.

    python scripts/model_info.py  ->  reports/model_info.json
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_flops, get_num_params

ROOT = Path(__file__).resolve().parents[1]
TRAIN_KEYS = ("model", "data", "epochs", "imgsz", "batch", "optimizer", "lr0", "lrf", "momentum",
              "weight_decay", "warmup_epochs", "patience", "close_mosaic", "mosaic", "mixup",
              "fliplr", "hsv_h", "hsv_s", "hsv_v", "degrees", "scale", "translate", "seed", "device",
              "pretrained", "amp", "cos_lr", "name")


def describe(path: Path) -> dict:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = YOLO(str(path))
    net = model.model
    layers = len(list(net.modules()))
    params = get_num_params(net)
    gflops = get_flops(net, imgsz=640)
    args = ckpt.get("train_args") or {}
    yaml = getattr(model.model, "yaml", {}) or {}
    return {
        "file": path.name,
        "size_mb": round(path.stat().st_size / 1e6, 2),
        "task": model.task,
        "num_classes": len(model.names),
        "classes": list(model.names.values()),
        "yaml_file": yaml.get("yaml_file"),
        "scale": yaml.get("scale"),
        "layers": layers,
        "params": params,
        "gflops": round(gflops, 2) if gflops else None,
        "date": ckpt.get("date"),
        "ultralytics_version": ckpt.get("version"),
        "epoch": ckpt.get("epoch"),
        "best_fitness": None if ckpt.get("best_fitness") is None else float(ckpt["best_fitness"]),
        "train_metrics": {k: float(v) for k, v in (ckpt.get("train_metrics") or {}).items()},
        "train_args": {k: (str(args[k]) if k in ("data", "model", "device") else args[k])
                       for k in TRAIN_KEYS if k in args},
    }


if __name__ == "__main__":
    info = [describe(p) for p in sorted((ROOT / "models").glob("*.pt"))]
    out = ROOT / "reports" / "model_info.json"
    out.write_text(json.dumps(info, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    for m in info:
        print(m["file"], m["yaml_file"], m["params"], m["gflops"], m["train_args"].get("epochs"))
