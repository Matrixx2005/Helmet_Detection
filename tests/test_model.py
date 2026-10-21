"""Kiểm thử tích hợp với mô hình thật. Tự bỏ qua nếu thiếu ultralytics hoặc best.pt."""

import pytest

from helmet_ppe.config import DEFAULT_MODEL_PATH, PROJECT_ROOT, DetectorConfig, PipelineConfig, Status
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


def test_pipeline_runs_on_real_image(pipeline):
    images = list_images(PROJECT_ROOT / "data" / "raw")
    if not images:
        pytest.skip("data/raw không có ảnh")
    result = pipeline.analyze(read_image(images[0]), images[0])
    assert result.width > 0 and result.height > 0
    assert {d.label for d in result.detections} <= {"person", "helmet", "no_helmet"}
    valid = {Status.HELMET_DETECTED, Status.SUSPECTED_NO_HELMET, Status.UNKNOWN}
    assert all(p.status in valid for p in result.persons)
