"""Kiểm thử nhãn chuẩn cấp người và đánh giá hệ luật."""

import numpy as np
import pytest

from helmet_ppe.evaluation.ground_truth import (derive_person_gt, load_yolo_labels,
                                                read_person_gt, write_person_gt)
from helmet_ppe.evaluation.person_eval import evaluate_persons, match_persons
from helmet_ppe.image_io import write_image

NAMES = {0: "helmet", 6: "Person", 7: "no_helmet"}


def gt(image, pid, box, status):
    return {"image_name": image, "gt_person_id": pid, "x1": box[0], "y1": box[1],
            "x2": box[2], "y2": box[3], "gt_status": status, "source": "test"}


def pred(image, pid, box, status):
    return {"image_name": image, "person_id": pid, "x1": box[0], "y1": box[1],
            "x2": box[2], "y2": box[3], "status": status, "rule_id": "", "reason": ""}


A, B, C = (0, 0, 100, 300), (200, 0, 300, 300), (400, 0, 500, 300)


def test_load_yolo_labels_converts_to_pixels(tmp_path):
    label = tmp_path / "x.txt"
    label.write_text("6 0.5 0.5 0.2 0.4\n\n0 0.5 0.25 0.1 0.1\n")
    dets = load_yolo_labels(label, 1000, 500, NAMES)
    assert [d.class_name for d in dets] == ["Person", "helmet"]
    assert dets[0].bbox == pytest.approx((400, 150, 600, 350))


def test_derive_person_gt_from_box_labels(tmp_path):
    images, labels = tmp_path / "images", tmp_path / "labels"
    labels.mkdir()
    write_image(images / "a.jpg", np.zeros((500, 1000, 3), np.uint8))
    # người trái đội mũ, người phải có no_helmet, người giữa không có bằng chứng
    (labels / "a.txt").write_text(
        "6 0.15 0.5 0.1 0.8\n0 0.15 0.15 0.04 0.08\n"
        "6 0.85 0.5 0.1 0.8\n7 0.85 0.15 0.04 0.08\n"
        "6 0.5 0.5 0.1 0.8\n")
    rows = derive_person_gt(images, labels, NAMES)
    assert [r["gt_status"] for r in rows] == ["HELMET", "UNDETERMINED", "NO_HELMET"]

    path = write_person_gt(tmp_path / "gt.csv", rows)
    assert len(read_person_gt(path)) == 3


def test_read_person_gt_rejects_bad_status(tmp_path):
    path = write_person_gt(tmp_path / "gt.csv", [gt("a.jpg", 1, A, "MAYBE")])
    with pytest.raises(ValueError):
        read_person_gt(path)


def test_match_persons_by_iou():
    matches, missed, extra = match_persons([pred("i", 1, (5, 0, 105, 300), "UNKNOWN")],
                                           [gt("i", 1, A, "HELMET"), gt("i", 2, B, "HELMET")], 0.5)
    assert len(matches) == 1 and len(missed) == 1 and extra == []


def test_evaluate_persons_metrics():
    gts = [gt("i", 1, A, "HELMET"), gt("i", 2, B, "NO_HELMET"), gt("i", 3, C, "NO_HELMET"),
           gt("j", 1, A, "HELMET")]
    preds = [pred("i", 1, A, "HELMET_DETECTED"),
             pred("i", 2, B, "SUSPECTED_NO_HELMET"),
             pred("i", 3, C, "UNKNOWN"),
             pred("j", 1, (600, 0, 700, 300), "SUSPECTED_NO_HELMET")]  # người ảo
    m = evaluate_persons(preds, gts, images={"i", "j"})

    assert (m["gt_persons"], m["matched_persons"], m["missed_persons"], m["false_persons"]) == (4, 3, 1, 1)
    assert m["violation"] == {"tp": 1, "fp": 1, "fn": 1, "precision": 0.5, "recall": 0.5, "f1": 0.5}
    assert m["helmet"]["tp"] == 1 and m["helmet"]["fn"] == 1
    assert m["unknown_rate_matched"] == pytest.approx(1 / 3, abs=1e-4)
    assert m["decided_accuracy"] == 1.0
    kinds = sorted(c["error_type"] for c in m["cases"])
    assert kinds == ["FALSE_PERSON", "MISSED_PERSON", "VIOLATION_AS_UNKNOWN"]


def test_evaluate_persons_without_positives_gives_none_not_zero():
    m = evaluate_persons([], [gt("i", 1, A, "HELMET")], images={"i"})
    assert m["violation"]["precision"] is None and m["violation"]["recall"] is None


def test_tuning_sweep_reanalyses_saved_detections(tmp_path):
    from factories import FakePipeline, make_images

    from helmet_ppe.batch import run_batch
    from helmet_ppe.evaluation.tuning import SweepGrid, sweep
    from helmet_ppe.image_io import list_images

    src = tmp_path / "img"
    src.mkdir()
    make_images(src, 2)
    run_batch(FakePipeline(), list_images(src), tmp_path / "run", make_charts=False)
    gts = [gt(f"img ({i}).jpg", 1, (5, 10, 25, 60), "HELMET") for i in (1, 2)]
    gts += [gt(f"img ({i}).jpg", 2, (40, 10, 60, 60), "NO_HELMET") for i in (1, 2)]

    grid = SweepGrid(use_pose=(False,), conf_pose_person=(0.5,),
                     conf_person=(0.5,), conf_helmet=(0.5, 0.95), conf_no_helmet=(0.5,),
                     min_helmet_evidence=(0.2,), min_no_helmet_evidence=(0.2,), margin=(0.1,),
                     suppress_faces_under_helmets=(True,))
    table = sweep(tmp_path / "run", gts, grid)
    assert [row["conf_helmet"] for row in table] == [0.5, 0.95]  # 0.95 loại hết mũ (conf 0.9)
    assert table[0]["violation_f1"] == 1.0 and table[0]["helmet_f1"] == 1.0
    assert table[1]["helmet_recall"] == 0.0


def test_naive_baseline_treats_missing_helmet_as_violation(tmp_path):
    import json

    from helmet_ppe.evaluation.baseline import naive_person_rows

    image = {"image_name": "a.jpg", "detections": [
        {"class_name": "Person", "bbox": [0, 0, 100, 300]},
        {"class_name": "Person", "bbox": [200, 0, 300, 300]},
        {"class_name": "helmet", "bbox": [30, 0, 70, 40]},
    ]}
    path = tmp_path / "results.json"
    path.write_text(json.dumps({"images": [image]}), encoding="utf-8")
    assert [r["status"] for r in naive_person_rows(path)] == ["HELMET_DETECTED", "SUSPECTED_NO_HELMET"]
