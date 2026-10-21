import pytest

from helmet_ppe import geometry


def test_to_bbox_normalises_corner_order():
    assert geometry.to_bbox([10, 20, 0, 5]) == (0.0, 5.0, 10.0, 20.0)


def test_to_bbox_rejects_wrong_length():
    with pytest.raises(ValueError):
        geometry.to_bbox([1, 2, 3])


def test_iou_identical_disjoint_and_partial():
    a = (0, 0, 10, 10)
    assert geometry.iou(a, a) == pytest.approx(1.0)
    assert geometry.iou(a, (20, 20, 30, 30)) == 0.0
    assert geometry.iou(a, (5, 0, 15, 10)) == pytest.approx(50 / 150)


def test_inside_ratio_and_degenerate_box():
    assert geometry.inside_ratio((0, 0, 10, 10), (0, 0, 5, 10)) == pytest.approx(0.5)
    assert geometry.inside_ratio((3, 3, 3, 3), (0, 0, 10, 10)) == 0.0


def test_expand_grows_each_side_by_ratio():
    assert geometry.expand((0, 0, 10, 20), top=0.5, right=0.1) == (0, -10, 11, 20)
