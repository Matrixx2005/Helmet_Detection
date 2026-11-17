"""Kiểm tra tính nhất quán của đối chứng, nhận xét và đánh giá có nhãn."""

import csv
import io
from dataclasses import replace

import pytest
from factories import Scene

from helmet_ppe.comparison import compare_results, evaluate_comparison, read_comparison_gt
from helmet_ppe.config import Status


def ground_truth(image="scene.jpg", status="NO_HELMET"):
    return {"image_name": image, "gt_person_id": "1", "x1": 100, "y1": 100,
            "x2": 300, "y2": 700, "gt_status": status, "source": "manual"}


def make_scene(name="scene.jpg"):
    scene = Scene()
    scene.person(100, 100, 300, 700)
    scene.add("helmet", (140, 450, 210, 500))  # tâm mũ trong thân người, ngoài vùng đầu
    scene.no_helmet_on((100, 100, 300, 700))
    return scene.analyze(name=name)


def test_comparison_keeps_people_and_uses_actual_evidence_without_mutating_input():
    result = make_scene()
    original = result.to_dict()
    report = compare_results([result])
    assert report.before[0].persons[0].status == Status.HELMET_DETECTED
    assert report.after[0].persons[0].status == Status.SUSPECTED_NO_HELMET
    assert report.before[0].persons[0].person_id == result.persons[0].person_id
    assert report.before[0].persons[0].bbox == result.persons[0].bbox
    assert report.rows[0]["after_reason"] == result.persons[0].reason
    assert report.rows[0]["no_helmet_evidence"] == result.persons[0].no_helmet.strength
    assert report.summary["changed_persons"] == 1
    assert any("1 người chuyển từ Đội mũ sang Nghi không đội mũ" in c for c in report.comments)
    assert result.to_dict() == original


def test_comparison_includes_pose_only_people_and_preserves_ids():
    scene = Scene()
    scene.person(100, 100, 300, 700)
    scene.pose_person(600, 100, 800, 700)
    result = scene.analyze()
    report = compare_results([result])
    assert report.summary["persons"] == 2
    assert [(p.person_id, p.bbox) for p in report.before[0].persons] == [
        (p.person_id, p.bbox) for p in result.persons]
    assert report.summary["changed_persons"] == 2
    assert all(p.status == Status.UNKNOWN for p in result.persons)


def test_empty_image_and_empty_report_are_valid():
    report = compare_results([Scene().analyze()])
    assert report.summary["persons"] == 0
    assert report.before[0].image_status == Status.NO_PERSON_DETECTED
    assert report.summary["changed_persons"] == 0
    assert compare_results([]).to_dict()["summary"]["images"] == 0


def test_duplicate_image_names_are_rejected():
    with pytest.raises(ValueError, match="trùng"):
        compare_results([make_scene(), make_scene()])


def test_evaluation_uses_same_body_matching_and_reports_improvement_from_counts():
    report = compare_results([make_scene()])
    evaluation = evaluate_comparison(report, [ground_truth()], covered_images={"scene.jpg"})
    assert evaluation["before"]["matched_persons"] == evaluation["after"]["matched_persons"] == 1
    assert evaluation["before"]["violation"]["fn"] == 1
    assert evaluation["after"]["violation"]["tp"] == 1
    assert evaluation["after"]["violation"]["recall"] == 1
    assert any("tăng 100.00 điểm phần trăm" in c for c in evaluation["comments"])


def test_evaluation_comments_also_report_worse_results():
    scene = Scene()
    scene.person(100, 100, 300, 700)
    result = scene.analyze()
    result.persons = [replace(result.persons[0], status=Status.HELMET_DETECTED)]
    evaluation = evaluate_comparison(compare_results([result]), [ground_truth()], covered_images={"scene.jpg"})
    assert evaluation["before"]["violation"]["recall"] == 1
    assert evaluation["after"]["violation"]["recall"] == 0
    assert any("giảm 100.00 điểm phần trăm" in c for c in evaluation["comments"])


def test_evaluation_excludes_unlabeled_images_and_keeps_undefined_metrics():
    report = compare_results([make_scene(), make_scene("unlabeled.jpg")])
    evaluation = evaluate_comparison(report, [ground_truth(status="UNDETERMINED")],
                                     covered_images={"scene.jpg"})
    assert evaluation["unlabeled_images"] == ["unlabeled.jpg"]
    assert evaluation["after"]["num_images"] == 1
    assert evaluation["after"]["violation"]["recall"] is None
    assert any("chưa xác định" in c for c in evaluation["comments"])


def test_evaluation_counts_a_labeled_empty_image_as_false_person():
    evaluation = evaluate_comparison(compare_results([make_scene()]), [], covered_images={"scene.jpg"})
    assert evaluation["after"]["gt_persons"] == 0
    assert evaluation["after"]["false_persons"] == 1
    with pytest.raises(ValueError, match="trùng tên"):
        evaluate_comparison(compare_results([make_scene()]), [], covered_images={"other.jpg"})


def csv_content(rows):
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue().encode("utf-8-sig")


def test_gt_reader_accepts_optional_source_and_head_and_rejects_duplicates():
    row = ground_truth()
    del row["source"]
    assert read_comparison_gt(csv_content([row]))[0]["gt_status"] == "NO_HELMET"
    with pytest.raises(ValueError, match="trùng"):
        read_comparison_gt(csv_content([row, row]))
    with pytest.raises(ValueError, match="thiếu cột"):
        read_comparison_gt(b"image_name\nscene.jpg\n")


@pytest.mark.parametrize("updates", [
    {"x1": "nan"}, {"x2": "inf"}, {"x2": 90}, {"image_name": ""},
    {"gt_status": "WRONG"}, {"head_x1": 1},
])
def test_gt_reader_rejects_invalid_labels(updates):
    row = {**ground_truth(), **updates}
    with pytest.raises(ValueError):
        read_comparison_gt(csv_content([row]))
