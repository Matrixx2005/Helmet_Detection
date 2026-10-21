"""Giao diện dòng lệnh: ``python -m helmet_ppe <lệnh> [tùy chọn]``.

Các lệnh:
    predict          Phân tích một ảnh.
    batch            Xử lý hàng loạt một thư mục ảnh.
    prepare-dataset  Giải nén Construction-PPE và tạo data_local.yaml.
    eval-detector    Đánh giá YOLO (Precision, Recall, mAP50, mAP50-95).
    make-person-gt   Sinh nhãn chuẩn cấp người từ nhãn hộp chuẩn.
    tune-rules       Dò ngưỡng confidence / bằng chứng trên tập validation.
    eval-rules       Đánh giá kết luận cấp người so với nhãn chuẩn.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from helmet_ppe.config import (DEFAULT_MODEL_PATH, DEFAULT_OUTPUT_DIR, PROJECT_ROOT,
                               DetectorConfig, PipelineConfig, RuleConfig)

log = logging.getLogger("helmet_ppe")

DEFAULT_DATASET_DIR = PROJECT_ROOT / "data" / "construction-ppe"
DEFAULT_REPORTS_DIR = PROJECT_ROOT / "reports"


# --------------------------------------------------------------------------- helpers
def _pipeline_config(args: argparse.Namespace) -> PipelineConfig:
    return PipelineConfig(
        detector=DetectorConfig(model_path=str(args.model), conf=args.conf, iou=args.iou,
                                imgsz=args.imgsz, device=args.device),
        rules=RuleConfig(min_helmet_evidence=args.min_evidence,
                         min_no_helmet_evidence=args.min_evidence,
                         decision_margin=args.margin),
    )


def _add_model_args(parser: argparse.ArgumentParser) -> None:
    det, rules = DetectorConfig(), RuleConfig()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help="Tệp trọng số .pt")
    parser.add_argument("--conf", type=float, default=det.conf, help="Ngưỡng confidence YOLO")
    parser.add_argument("--iou", type=float, default=det.iou, help="Ngưỡng IoU của NMS")
    parser.add_argument("--imgsz", type=int, default=det.imgsz, help="Kích thước ảnh đầu vào")
    parser.add_argument("--device", default=None, help="cpu, 0 (GPU đầu tiên)…; mặc định tự chọn")
    parser.add_argument("--min-evidence", type=float, default=rules.min_helmet_evidence,
                        help="Ngưỡng bằng chứng (confidence × điểm ghép) để kết luận")
    parser.add_argument("--margin", type=float, default=rules.decision_margin,
                        help="Chênh lệch tối thiểu giữa bằng chứng mũ và không-mũ")


def _print_json(data: dict) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2))


# --------------------------------------------------------------------------- commands
def cmd_predict(args: argparse.Namespace) -> int:
    from helmet_ppe.image_io import read_image, write_image
    from helmet_ppe.pipeline import HelmetPipeline
    from helmet_ppe.visualize import draw_result

    pipeline = HelmetPipeline(_pipeline_config(args))
    image = read_image(args.image)
    result = pipeline.analyze(image, args.image)
    out_dir = Path(args.output)
    out_path = out_dir / Path(args.image).name
    write_image(out_path, draw_result(image, result))
    (out_dir / f"{Path(args.image).stem}.json").write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nẢnh: {result.image_name} → {result.image_status}")
    print(f"Số người: {len(result.persons)} | Số mũ phát hiện: {result.count('helmet')}")
    for p in result.persons:
        print(f"  P{p.person_id}: {p.status} [{p.rule_id}] {p.reason}")
    print(f"Đã lưu: {out_path}")
    return 0


def cmd_batch(args: argparse.Namespace) -> int:
    from helmet_ppe.batch import next_experiment_dir, run_folder
    from helmet_ppe.pipeline import HelmetPipeline

    output = Path(args.output) if args.output else next_experiment_dir(DEFAULT_OUTPUT_DIR)
    pipeline = HelmetPipeline(_pipeline_config(args))
    report = run_folder(pipeline, args.input, output, recursive=args.recursive,
                        batch_size=args.batch_size, save_images=not args.no_images)
    print(f"\nKết quả đã lưu tại: {report.output_dir}")
    _print_json(report.statistics)
    return 0 if report.statistics["images_processed"] else 1


def cmd_prepare_dataset(args: argparse.Namespace) -> int:
    from helmet_ppe.evaluation.detection_eval import prepare_dataset

    print(f"data.yaml dùng để đánh giá: {prepare_dataset(Path(args.zip), Path(args.dest))}")
    return 0


def cmd_eval_detector(args: argparse.Namespace) -> int:
    from helmet_ppe import charts
    from helmet_ppe.evaluation import detection_eval as de
    from helmet_ppe.reporting import write_csv, write_json

    data_yaml = de.write_eval_yaml(Path(args.dataset))
    out_dir = Path(args.output)
    results = []
    for model in args.models:
        for split in args.splits:
            log.info("Đánh giá %s trên split %s", model, split)
            res = de.validate_model(Path(model), data_yaml, split, out_dir / "ultralytics_runs",
                                    imgsz=args.imgsz, batch=args.batch, device=args.device)
            de.copy_plots(res, out_dir / "plots")
            results.append(res)
            tag = f"{Path(model).stem}_{split}"
            write_csv(out_dir / f"per_class_{tag}.csv", res["per_class"],
                      list(res["per_class"][0].keys()))
            focus = [c for c in res["per_class"] if c["class_name"] in de.FOCUS_CLASSES]
            charts.grouped_bar_chart(
                [c["class_name"] for c in focus],
                {m: [c[m] for c in focus] for m in ("precision", "recall", "mAP50", "mAP50-95")},
                out_dir / "charts" / f"focus_classes_{tag}.png",
                title=f"{Path(model).name} – {split}: Person / helmet / no_helmet",
                xlabel="Lớp", ylabel="Giá trị")

    overall = [{k: r[k] for k in ("model_name", "split", "num_images", "precision", "recall",
                                  "mAP50", "mAP50-95")} | {"inference_ms": r["speed_ms"].get("inference")}
               for r in results]
    write_csv(out_dir / "overall_metrics.csv", overall, list(overall[0].keys()))
    write_json(out_dir / "detection_metrics.json", results)
    labels = [f"{r['model_name']}\n{r['split']}" for r in results]
    charts.grouped_bar_chart(
        labels, {m: [r[m] for r in results] for m in ("precision", "recall", "mAP50", "mAP50-95")},
        out_dir / "charts" / "overall_metrics.png",
        title="Chỉ số phát hiện tổng thể (11 lớp)", xlabel="Mô hình / split", ylabel="Giá trị")
    for row in overall:
        print(row)
    return 0


def cmd_make_person_gt(args: argparse.Namespace) -> int:
    from helmet_ppe.evaluation import detection_eval as de
    from helmet_ppe.evaluation.ground_truth import derive_person_gt, write_person_gt

    dataset = Path(args.dataset)
    names = de.class_names(dataset / "data.yaml")
    rows = derive_person_gt(dataset / "images" / args.split, dataset / "labels" / args.split, names)
    path = write_person_gt(Path(args.output), rows)
    print(f"Đã ghi {len(rows)} người chuẩn → {path}")
    return 0


def cmd_tune_rules(args: argparse.Namespace) -> int:
    from helmet_ppe.evaluation.ground_truth import read_person_gt
    from helmet_ppe.evaluation.tuning import sweep
    from helmet_ppe.reporting import write_csv, write_json

    table = sweep(Path(args.run), read_person_gt(Path(args.gt)), confs=args.confs,
                  evidences=args.evidences, margins=args.margins, iou_threshold=args.iou)
    out_dir = Path(args.output)
    write_csv(out_dir / "rule_sweep.csv", table, list(table[0].keys()))
    write_json(out_dir / "best_params.json", table[0])
    print("Top 5 tổ hợp (sắp theo trung bình F1 của 'vi phạm' và 'đội mũ'):")
    for row in table[:5]:
        print(row)
    return 0


def cmd_eval_rules(args: argparse.Namespace) -> int:
    from helmet_ppe import charts
    from helmet_ppe.evaluation.error_gallery import build_gallery
    from helmet_ppe.evaluation.ground_truth import read_person_gt
    from helmet_ppe.evaluation.person_eval import evaluate_persons
    from helmet_ppe.reporting import read_csv, write_csv, write_json

    run_dir, out_dir = Path(args.run), Path(args.output)
    preds = read_csv(run_dir / "persons.csv")
    processed = {row["image_name"] for row in read_csv(run_dir / "summary.csv")}
    gts = [g for g in read_person_gt(Path(args.gt)) if g["image_name"] in processed]
    result = evaluate_persons(preds, gts, iou_threshold=args.iou, images=processed)
    cases = result.pop("cases")

    write_json(out_dir / "rule_metrics.json", result)
    write_csv(out_dir / "error_cases.csv", cases,
              ["error_type", "image_name", "gt_person_id", "gt_status", "pred_person_id",
               "pred_status", "rule_id", "iou", "reason"])
    cm = result["confusion_matrix"]
    charts.confusion_matrix_chart(cm["values"], cm["rows"], cm["cols"],
                                  out_dir / "charts" / "rule_confusion_matrix.png",
                                  title="Ma trận nhầm lẫn cấp người")
    charts.grouped_bar_chart(
        ["Vi phạm (no_helmet)", "Đội mũ (helmet)"],
        {m: [result["violation"][m] or 0, result["helmet"][m] or 0]
         for m in ("precision", "recall", "f1")},
        out_dir / "charts" / "rule_prf.png",
        title="Precision / Recall / F1 của hệ luật", xlabel="Lớp kết luận", ylabel="Giá trị")
    gallery = build_gallery(cases, gts, run_dir / "annotated_images", out_dir / "error_examples")
    print(f"Đã lưu {len(gallery)} ảnh minh họa lỗi.")
    _print_json(result)
    return 0


# --------------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="helmet_ppe",
                                     description="Phát hiện mũ bảo hộ và suy luận vi phạm bằng hệ luật")
    parser.add_argument("-v", "--verbose", action="store_true", help="In log chi tiết")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("predict", help="Phân tích một ảnh")
    p.add_argument("image", type=Path)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR / "single")
    _add_model_args(p)
    p.set_defaults(func=cmd_predict)

    p = sub.add_parser("batch", help="Xử lý hàng loạt một thư mục ảnh")
    p.add_argument("--input", type=Path, default=PROJECT_ROOT / "data" / "test_images")
    p.add_argument("--output", type=Path, default=None,
                   help="Thư mục kết quả (mặc định outputs/experiment_XX mới)")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--recursive", action="store_true", help="Quét cả thư mục con")
    p.add_argument("--no-images", action="store_true", help="Không lưu ảnh đã vẽ")
    _add_model_args(p)
    p.set_defaults(func=cmd_batch)

    p = sub.add_parser("prepare-dataset", help="Giải nén dataset và tạo data_local.yaml")
    p.add_argument("--zip", type=Path, required=True)
    p.add_argument("--dest", type=Path, default=DEFAULT_DATASET_DIR)
    p.set_defaults(func=cmd_prepare_dataset)

    p = sub.add_parser("eval-detector", help="Đánh giá YOLO trên tập có nhãn")
    p.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_DIR)
    p.add_argument("--models", nargs="+", default=[str(DEFAULT_MODEL_PATH)])
    p.add_argument("--splits", nargs="+", default=["val", "test"], choices=["train", "val", "test"])
    p.add_argument("--output", type=Path, default=DEFAULT_REPORTS_DIR / "detection_eval")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default=None)
    p.set_defaults(func=cmd_eval_detector)

    p = sub.add_parser("make-person-gt", help="Sinh nhãn chuẩn cấp người từ nhãn hộp")
    p.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_DIR)
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--output", type=Path, default=PROJECT_ROOT / "data" / "person_gt" / "test_person_gt.csv")
    p.set_defaults(func=cmd_make_person_gt)

    p = sub.add_parser("tune-rules", help="Dò ngưỡng hệ luật trên tập validation")
    p.add_argument("--run", type=Path, required=True,
                   help="Kết quả batch trên ảnh val, chạy với --conf thấp (vd. 0.05)")
    p.add_argument("--gt", type=Path, required=True, help="CSV nhãn chuẩn cấp người của val")
    p.add_argument("--output", type=Path, default=DEFAULT_REPORTS_DIR / "rule_tuning")
    p.add_argument("--confs", type=float, nargs="+",
                   default=[0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5])
    p.add_argument("--evidences", type=float, nargs="+", default=[0.1, 0.15, 0.2, 0.25, 0.3, 0.4])
    p.add_argument("--margins", type=float, nargs="+", default=[0.0, 0.05, 0.1, 0.2])
    p.add_argument("--iou", type=float, default=0.5)
    p.set_defaults(func=cmd_tune_rules)

    p = sub.add_parser("eval-rules", help="Đánh giá kết luận cấp người")
    p.add_argument("--run", type=Path, required=True, help="Thư mục kết quả batch (có persons.csv)")
    p.add_argument("--gt", type=Path, required=True, help="CSV nhãn chuẩn cấp người")
    p.add_argument("--output", type=Path, default=DEFAULT_REPORTS_DIR / "rule_eval")
    p.add_argument("--iou", type=float, default=0.5, help="Ngưỡng IoU ghép người dự đoán ↔ chuẩn")
    p.set_defaults(func=cmd_eval_rules)
    return parser


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):  # tiếng Việt trên console Windows
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    try:
        return args.func(args)
    except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
        log.error("%s", exc)
        return 2
