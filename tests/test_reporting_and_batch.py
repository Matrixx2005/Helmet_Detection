"""Kiểm thử xuất CSV/JSON và batch processing với pipeline giả (không cần YOLO)."""

import json

import cv2
import numpy as np
from factories import FakePipeline, Scene, make_images

from helmet_ppe import reporting
from helmet_ppe.batch import next_experiment_dir, run_batch
from helmet_ppe.config import Status
from helmet_ppe.image_io import list_images, read_image, write_image
from helmet_ppe.visualize import draw_result


def test_list_images_filters_extensions_and_sorts_naturally(tmp_path):
    make_images(tmp_path, 3)
    write_image(tmp_path / "x.png", np.zeros((10, 10, 3), np.uint8))
    (tmp_path / "notes.txt").write_text("không phải ảnh")
    names = [p.name for p in list_images(tmp_path)]
    assert names == ["img (1).jpg", "img (2).jpg", "img (3).jpg", "x.png"]


def test_read_image_supports_unicode_path(tmp_path):
    path = tmp_path / "ảnh công trường.jpg"
    write_image(path, np.zeros((12, 16, 3), np.uint8))
    assert read_image(path).shape == (12, 16, 3)


def test_rows_are_consistent_and_not_double_counted():
    scene = Scene()
    box = (400, 100, 600, 700)
    scene.person(*box)
    scene.helmet_on(box)
    scene.add("helmet", (800, 650, 880, 720))  # mũ không thuộc ai
    result = scene.analyze(name="a.jpg")

    det_rows = reporting.detection_rows(result)
    assert len(det_rows) == len(result.detections)
    assert sorted(r["role"] for r in det_rows) == ["helmet_of_person", "person", "unmatched"]
    assert len(reporting.person_rows(result)) == 1
    summary = reporting.summary_row(result)
    assert (summary["num_persons"], summary["num_helmets"], summary["unmatched_helmets"]) == (1, 2, 1)


def test_draw_result_does_not_modify_source():
    scene = Scene(200, 200)
    scene.person(50, 20, 150, 190)
    image = np.zeros((200, 200, 3), np.uint8)
    drawn = draw_result(image, scene.analyze())
    assert image.sum() == 0 and drawn.sum() > 0


def test_run_batch_writes_all_outputs_and_survives_bad_image(tmp_path):
    src = tmp_path / "images"
    src.mkdir()
    make_images(src, 5)
    (src / "broken.jpg").write_bytes(b"day khong phai anh")
    before = {p.name: p.read_bytes() for p in src.iterdir()}

    out = tmp_path / "out"
    report = run_batch(FakePipeline(), list_images(src), out, batch_size=2)

    for name in ("detections.csv", "persons.csv", "summary.csv", "errors.csv",
                 "results.json", "config.json"):
        assert (out / name).is_file(), name
    assert len(list((out / "annotated_images").iterdir())) == 5
    assert len(list((out / "charts").glob("*.png"))) == 5

    stats = report.statistics
    assert (stats["images_total"], stats["images_processed"], stats["images_failed"]) == (6, 5, 1)
    assert stats["persons_total"] == 10
    assert stats["person_status_counts"][str(Status.HELMET_DETECTED)] == 5
    assert stats["person_status_counts"][str(Status.SUSPECTED_NO_HELMET)] == 5

    persons = reporting.read_csv(out / "persons.csv")
    assert len(persons) == 10
    assert len({(r["image_name"], r["person_id"]) for r in persons}) == 10  # không trùng
    errors = reporting.read_csv(out / "errors.csv")
    assert [e["image_name"] for e in errors] == ["broken.jpg"]
    data = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert len(data["images"]) == 5
    assert {p.name: p.read_bytes() for p in src.iterdir()} == before  # ảnh nguồn nguyên vẹn
    assert cv2.imread(str(out / "annotated_images" / "img (1).jpg")) is not None


def test_next_experiment_dir_increments(tmp_path):
    assert next_experiment_dir(tmp_path).name == "experiment_01"
    (tmp_path / "experiment_01").mkdir()
    (tmp_path / "experiment_07").mkdir()
    assert next_experiment_dir(tmp_path).name == "experiment_08"
