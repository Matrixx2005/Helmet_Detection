"""Kiểm thử tích hợp với mô hình thật. Tự bỏ qua nếu thiếu ultralytics hoặc best.pt."""

import json

import pytest

from helmet_ppe.config import (DEFAULT_MODEL_PATH, PROJECT_ROOT, AssociationConfig,
                               DetectorConfig, PipelineConfig, RuleConfig, Status)
from helmet_ppe.image_io import list_images, read_image

pytest.importorskip("ultralytics")
pytestmark = pytest.mark.skipif(not DEFAULT_MODEL_PATH.is_file(),
                                reason="Không có models/best.pt")

EXPECTED_CLASSES = {0: "helmet", 1: "gloves", 2: "vest", 3: "boots", 4: "goggles", 5: "none",
                    6: "Person", 7: "no_helmet", 8: "no_goggle", 9: "no_gloves", 10: "no_boots"}


@pytest.fixture(scope="module")
def pipeline():
    from helmet_ppe.pipeline import HelmetPipeline

    return HelmetPipeline(PipelineConfig(detector=DetectorConfig(device="cpu")))


def test_model_has_expected_classes(pipeline):
    assert pipeline.detector.names == EXPECTED_CLASSES
    assert sorted(pipeline.detector.class_ids) == [0, 6, 7]


def test_pose_model_returns_keypoints(pipeline):
    if pipeline.pose is None:
        pytest.skip("Chưa có models/yolo26n-pose.pt")
    images = list_images(PROJECT_ROOT / "data" / "raw")
    if not images:
        pytest.skip("data/raw không có ảnh")
    people, _ = pipeline.pose.estimate_batch([read_image(images[0])])[0]
    assert all(len(kps) == 17 for _, _, kps in people)


def test_pipeline_runs_on_real_image(pipeline):
    images = list_images(PROJECT_ROOT / "data" / "raw")
    if not images:
        pytest.skip("data/raw không có ảnh")
    result = pipeline.analyze(read_image(images[0]), images[0])
    assert result.width > 0 and result.height > 0
    assert {d.label for d in result.detections} <= {"person", "helmet", "no_helmet", "pose_person"}
    valid = {Status.HELMET_DETECTED, Status.SUSPECTED_NO_HELMET, Status.UNKNOWN}
    assert all(p.status in valid for p in result.persons)


def test_default_thresholds_match_validation_tuning():
    """Ngưỡng mặc định trong config.py phải đúng bằng kết quả dò trên tập val."""
    path = PROJECT_ROOT / "reports" / "rule_tuning" / "best" / "best_params.json"
    if not path.is_file():
        pytest.skip("Chưa chạy tune-rules")
    tuned = json.loads(path.read_text(encoding="utf-8"))
    det, assoc, rules = DetectorConfig(), AssociationConfig(), RuleConfig()
    defaults = {
        "conf_person": det.conf_person, "conf_helmet": det.conf_helmet,
        "conf_no_helmet": det.conf_no_helmet, "use_pose": det.use_pose,
        "conf_pose_person": det.conf_pose_person, "head_tolerance_x": assoc.head_tolerance_x,
        "head_tolerance_y": assoc.head_tolerance_y, "min_head_size_ratio": assoc.min_head_size_ratio,
        "min_helmet_evidence": rules.min_helmet_evidence,
        "min_no_helmet_evidence": rules.min_no_helmet_evidence, "margin": rules.decision_margin,
        "suppress_faces_under_helmets": assoc.suppress_faces_under_helmets,
    }
    if not tuned.get("use_pose", True):  # tham số riêng của pose không có ý nghĩa khi tắt pose
        defaults = {k: v for k, v in defaults.items()
                    if k not in ("conf_pose_person", "head_tolerance_x", "head_tolerance_y",
                                 "min_head_size_ratio")}
    assert {k: v for k, v in defaults.items() if k in tuned} == {
        k: tuned[k] for k in defaults if k in tuned}
