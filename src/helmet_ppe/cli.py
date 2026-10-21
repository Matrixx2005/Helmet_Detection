"""Giao diện dòng lệnh: ``python -m helmet_ppe <lệnh> [tùy chọn]``.

Các lệnh:
    predict          Phân tích một ảnh.
    batch            Xử lý hàng loạt một thư mục ảnh.
    prepare-dataset  Giải nén Construction-PPE và tạo data_local.yaml.
    eval-detector    Đánh giá YOLO (Precision, Recall, mAP50, mAP50-95).
    make-person-gt   Sinh nhãn chuẩn cấp người từ nhãn hộp chuẩn.
    tune-rules       Dò ngưỡng confidence / bằng chứng trên tập validation.
    eval-rules       Đánh giá kết luận cấp người so với nhãn chuẩn.
    compare-rules    Bảng + biểu đồ so sánh nhiều kết quả eval-rules.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

from helmet_ppe.config import (DEFAULT_MODEL_PATH, DEFAULT_OUTPUT_DIR, PROJECT_ROOT,
                               AssociationConfig, DetectorConfig, PipelineConfig, RuleConfig)
from helmet_ppe.evaluation.tuning import SweepGrid

log = logging.getLogger("helmet_ppe")

DEFAULT_DATASET_DIR = PROJECT_ROOT / "data" / "construction-ppe"
DEFAULT_REPORTS_DIR = PROJECT_ROOT / "reports"

# Tên hiển thị cho các thư mục kết quả do scripts/run_evaluation.bat tạo ra.
RUN_DISPLAY_NAMES = {
    "best_untuned": "Ngưỡng chung 0,25",
    "baseline_naive": "Đối chứng ngây thơ",
    "best_v2": "best_v2.pt + luật",
    "best_nopose": "best.pt + luật (không pose)",
    "best": "best.pt + luật + pose (đề xuất)",
}


# --------------------------------------------------------------------------- helpers
def _pipeline_config(args: argparse.Namespace) -> PipelineConfig:
    """Cấu hình từ tham số dòng lệnh; ``--params`` (best_params.json) ghi đè các ngưỡng."""
    if args.params:
        tuned = json.loads(Path(args.params).read_text(encoding="utf-8"))
        for key in ("conf_person", "conf_helmet", "conf_no_helmet", "conf_pose_person",
                    "min_helmet_evidence", "min_no_helmet_evidence", "margin"):
            setattr(args, key, tuned[key])
        args.no_pose = not tuned["use_pose"]
        association = AssociationConfig(head_tolerance_x=tuned["head_tolerance_x"],
                                        head_tolerance_y=tuned["head_tolerance_y"],
                                        min_head_size_ratio=tuned["min_head_size_ratio"],
                                        suppress_faces_under_helmets=tuned.get(
                                            "suppress_faces_under_helmets", True))
    else:
        association = AssociationConfig()
    return PipelineConfig(
        detector=DetectorConfig(model_path=str(args.model), conf_person=args.conf_person,
                                conf_helmet=args.conf_helmet, conf_no_helmet=args.conf_no_helmet,
                                use_pose=not args.no_pose, pose_model_path=str(args.pose_model),
                                conf_pose_person=args.conf_pose_person,
                                iou=args.iou, imgsz=args.imgsz, device=args.device),
        association=association,
        rules=RuleConfig(min_helmet_evidence=args.min_helmet_evidence,
                         min_no_helmet_evidence=args.min_no_helmet_evidence,
                         decision_margin=args.margin),
    )


def _add_model_args(parser: argparse.ArgumentParser) -> None:
    det, rules = DetectorConfig(), RuleConfig()
    group = parser.add_argument_group("mô hình và ngưỡng (mặc định = giá trị dò trên tập val)")
    group.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help="Tệp trọng số .pt")
    group.add_argument("--conf-person", type=float, default=det.conf_person)
    group.add_argument("--conf-helmet", type=float, default=det.conf_helmet)
    group.add_argument("--conf-no-helmet", type=float, default=det.conf_no_helmet)
    group.add_argument("--conf-pose-person", type=float, default=det.conf_pose_person)
    group.add_argument("--pose-model", type=Path, default=Path(det.pose_model_path),
                       help="Mô hình pose (COCO) dùng để thêm người và định vị đầu")
    group.add_argument("--no-pose", action="store_true", default=not det.use_pose,
                       help="Tắt mô hình pose, chỉ dùng mô hình PPE")
    group.add_argument("--iou", type=float, default=det.iou, help="Ngưỡng IoU của NMS")
    group.add_argument("--imgsz", type=int, default=det.imgsz, help="Kích thước ảnh đầu vào")
    group.add_argument("--device", default=None, help="cpu, 0 (GPU đầu tiên)…; mặc định tự chọn")
    group.add_argument("--min-helmet-evidence", type=float, default=rules.min_helmet_evidence,
                       help="Ngưỡng bằng chứng mũ (confidence × điểm ghép)")
    group.add_argument("--min-no-helmet-evidence", type=float, default=rules.min_no_helmet_evidence,
                       help="Ngưỡng bằng chứng không-mũ")
    group.add_argument("--margin", type=float, default=rules.decision_margin,
                       help="Chênh lệch tối thiểu giữa hai loại bằng chứng")
    group.add_argument("--params", type=Path, default=None,
                       help="Nạp các ngưỡng từ best_params.json do lệnh tune-rules tạo ra")


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

    def on_off(choice: str) -> tuple[bool, ...]:
        return (False, True) if choice == "both" else (choice == "on",)

    grid = SweepGrid(use_pose=on_off(args.use_pose), conf_pose_person=args.conf_pose_person,
                     head_tolerance_x=args.head_tolerance_x, head_tolerance_y=args.head_tolerance_y,
                     min_head_size_ratio=args.min_head_size_ratio,
                     suppress_faces_under_helmets=on_off(args.suppress_faces),
                     conf_person=args.conf_person, conf_helmet=args.conf_helmet,
                     conf_no_helmet=args.conf_no_helmet, min_helmet_evidence=args.helmet_evidence,
                     min_no_helmet_evidence=args.no_helmet_evidence, margin=args.margins)
    table = sweep(Path(args.run), read_person_gt(Path(args.gt)), grid, iou_threshold=args.iou)
    out_dir = Path(args.output)
    write_csv(out_dir / "rule_sweep.csv", table, list(table[0].keys()))
    write_json(out_dir / "best_params.json", table[0])
    print(f"Đã thử {len(table)} tổ hợp. Top 5 (theo trung bình F1 'vi phạm' và 'đội mũ'):")
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
    if args.baseline:
        from helmet_ppe.evaluation.baseline import naive_person_rows

        preds = naive_person_rows(run_dir / "results.json")
    else:
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
    if not args.baseline:  # ảnh đã vẽ là của hệ thống, không minh họa được cho đối chứng
        gallery_dir = out_dir / "error_examples"
        shutil.rmtree(gallery_dir, ignore_errors=True)  # chỉ xóa ảnh minh họa do lần trước sinh ra
        gallery = build_gallery(cases, gts, run_dir / "annotated_images", gallery_dir)
        print(f"Đã lưu {len(gallery)} ảnh minh họa lỗi.")
    _print_json(result)
    return 0


def cmd_compare_rules(args: argparse.Namespace) -> int:
    from helmet_ppe import charts
    from helmet_ppe.reporting import write_csv

    if args.labels and len(args.labels) != len(args.runs):
        raise ValueError("Số --labels phải bằng số thư mục kết quả")
    labels = args.labels or [RUN_DISPLAY_NAMES.get(Path(r).name, Path(r).name) for r in args.runs]
    rows = []
    for label, run in zip(labels, args.runs):
        m = json.loads((Path(run) / "rule_metrics.json").read_text(encoding="utf-8"))
        rows.append({
            "config": label,
            **{f"violation_{k}": m["violation"][k] for k in ("precision", "recall", "f1")},
            **{f"helmet_{k}": m["helmet"][k] for k in ("precision", "recall", "f1")},
            "unknown_rate": m["unknown_rate_matched"], "decided_accuracy": m["decided_accuracy"],
        })
    out_dir = Path(args.output)
    write_csv(out_dir / "comparison.csv", rows, list(rows[0].keys()))
    charts.grouped_bar_chart(
        labels, {name: [r[key] or 0 for r in rows] for name, key in (
            ("F1 vi phạm", "violation_f1"), ("Precision vi phạm", "violation_precision"),
            ("Recall vi phạm", "violation_recall"), ("F1 đội mũ", "helmet_f1"))},
        out_dir / "comparison.png", title="So sánh cấu hình suy luận trên tập test",
        xlabel="Cấu hình", ylabel="Giá trị")
    for row in rows:
        print(row)
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
                   help="Kết quả batch trên ảnh val, chạy với ngưỡng confidence thấp (vd. 0.02)")
    p.add_argument("--gt", type=Path, required=True, help="CSV nhãn chuẩn cấp người của val")
    p.add_argument("--output", type=Path, default=DEFAULT_REPORTS_DIR / "rule_tuning")
    grid = SweepGrid()
    p.add_argument("--use-pose", choices=["both", "on", "off"], default="both",
                   help="Thử cả bật và tắt mô hình pose (mặc định), hoặc cố định")
    p.add_argument("--conf-pose-person", type=float, nargs="+", default=list(grid.conf_pose_person))
    p.add_argument("--conf-person", type=float, nargs="+", default=list(grid.conf_person))
    p.add_argument("--conf-helmet", type=float, nargs="+", default=list(grid.conf_helmet))
    p.add_argument("--conf-no-helmet", type=float, nargs="+", default=list(grid.conf_no_helmet))
    p.add_argument("--helmet-evidence", type=float, nargs="+", default=list(grid.min_helmet_evidence))
    p.add_argument("--no-helmet-evidence", type=float, nargs="+",
                   default=list(grid.min_no_helmet_evidence))
    p.add_argument("--margins", type=float, nargs="+", default=list(grid.margin))
    p.add_argument("--suppress-faces", choices=["both", "on", "off"], default="both",
                   help="Thử bật/tắt bước bỏ hộp no_helmet ngay dưới mũ")
    p.add_argument("--head-tolerance-x", type=float, nargs="+", default=list(grid.head_tolerance_x))
    p.add_argument("--head-tolerance-y", type=float, nargs="+", default=list(grid.head_tolerance_y))
    p.add_argument("--min-head-size-ratio", type=float, nargs="+",
                   default=list(grid.min_head_size_ratio))
    p.add_argument("--iou", type=float, default=0.5)
    p.set_defaults(func=cmd_tune_rules)

    p = sub.add_parser("eval-rules", help="Đánh giá kết luận cấp người")
    p.add_argument("--run", type=Path, required=True, help="Thư mục kết quả batch (có persons.csv)")
    p.add_argument("--gt", type=Path, required=True, help="CSV nhãn chuẩn cấp người")
    p.add_argument("--output", type=Path, default=DEFAULT_REPORTS_DIR / "rule_eval")
    p.add_argument("--iou", type=float, default=0.5, help="Ngưỡng IoU ghép người dự đoán ↔ chuẩn")
    p.add_argument("--baseline", action="store_true",
                   help="Đánh giá luật đối chứng ngây thơ (không thấy mũ = vi phạm) trên cùng detection")
    p.set_defaults(func=cmd_eval_rules)
    p = sub.add_parser("compare-rules", help="So sánh nhiều kết quả eval-rules")
    p.add_argument("runs", nargs="+", type=Path, help="Các thư mục chứa rule_metrics.json")
    p.add_argument("--labels", nargs="+", help="Tên hiển thị cho từng thư mục")
    p.add_argument("--output", type=Path, default=DEFAULT_REPORTS_DIR / "rule_eval" / "comparison")
    p.set_defaults(func=cmd_compare_rules)
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
