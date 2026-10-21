"""Sao chép các biểu đồ / ảnh minh họa đã sinh vào docs/figures để chèn vào báo cáo.

    python scripts/collect_figures.py

Chỉ sao chép tệp đã tồn tại (sinh bởi run_batch / run_evaluation); không tạo số liệu mới.
"""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "docs" / "figures"

FIGURES = {
    "reports/detection_eval/charts/overall_metrics.png": "detection_overall.png",
    "reports/detection_eval/charts/focus_classes_best_test.png": "detection_focus_test.png",
    "reports/detection_eval/plots/best_test_BoxPR_curve.png": "detection_pr_curve_test.png",
    "reports/detection_eval/plots/best_test_confusion_matrix_normalized.png":
        "detection_confusion_test.png",
    "reports/rule_eval/best/charts/rule_confusion_matrix.png": "rule_confusion_test.png",
    "reports/rule_eval/best/charts/rule_prf.png": "rule_prf_test.png",
    "reports/rule_eval/comparison/comparison.png": "rule_comparison_test.png",
    "outputs/experiment_01/charts/detections_by_class.png": "exp01_detections_by_class.png",
    "outputs/experiment_01/charts/person_status_distribution.png": "exp01_person_status.png",
    "outputs/experiment_01/charts/image_status_distribution.png": "exp01_image_status.png",
    "outputs/experiment_01/charts/images_success_failure.png": "exp01_success_failure.png",
    "outputs/experiment_01/charts/inference_time.png": "exp01_inference_time.png",
    "outputs/ppe_test_best/annotated_images/image611.jpg": "example_all_helmets.jpg",
    "outputs/ppe_test_best/annotated_images/image1239.jpg": "example_no_helmet.jpg",
    "outputs/experiment_01/annotated_images/anh_ct_nh (3).jpg": "example_raw_no_helmet.jpg",
    "reports/rule_eval/best/error_examples/FALSE_VIOLATION/image1354.jpg":
        "error_false_violation.jpg",
    "reports/rule_eval/best/error_examples/FALSE_PERSON/image1145.jpg": "error_box_mismatch.jpg",
    "reports/rule_eval/best/error_examples/MISSED_PERSON/image116.jpg": "error_missed_person.jpg",
}


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    missing = []
    for source, name in FIGURES.items():
        path = ROOT / source
        if path.is_file():
            shutil.copy2(path, DEST / name)
        else:
            missing.append(source)
    print(f"Đã sao chép {len(FIGURES) - len(missing)}/{len(FIGURES)} hình vào {DEST}")
    for source in missing:
        print(f"  [thiếu] {source}")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
